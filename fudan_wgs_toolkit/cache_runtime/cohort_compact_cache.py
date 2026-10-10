"""Bounded, exact cohort compact cache; never stores dosage or model products.

One immutable blob preserves the arrays returned by ``CachedGenotypeAdapter._prepare``.
Identity includes the complete ordered variant/sample axes, their dtypes, the
MAC scalar, and an explicit source/decoder binding supplied by the caller.
SQLite reservations cover unfinished writes as well as published blobs. Fixed
striped flock files serialize every key without creating files on cache misses.
Blob data and a conservative per-entry index allowance count against max_bytes;
the fixed SQLite schema/lock-directory overhead is outside this data budget.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import threading
import time
import uuid

import numpy as np


FORMAT = "private-cohort-compact-v1"
MAGIC = b"WGSCPC01"
DEFAULT_MAX_BYTES = 64 * 2**30
_ALIGNMENT = 64
_BLOCK = 4096
_INDEX_ALLOWANCE = 4096
_MAX_HEADER_BYTES = 2**20
_ARRAY_NAMES = ("columns", "exception_col", "exception_row", "exception_state",
                "summary_0", "summary_1", "summary_2", "summary_3", "summary_4")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _align(value, alignment=_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def _axis(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(name + " must be a one-dimensional nonnegative integer axis")
    contiguous = np.ascontiguousarray(values)
    return contiguous, dict(dtype=values.dtype.str, shape=[len(values)],
                            sha256=hashlib.sha256(contiguous.view(np.uint8)).hexdigest())


def _mac(value):
    if value is None:
        return None
    scalar = np.asarray(value)
    if (scalar.ndim != 0 or scalar.dtype.kind not in "iuf" or not np.isfinite(scalar)
            or scalar < 0):
        raise ValueError("minimum_mac must be None or a finite nonnegative numeric scalar")
    return dict(dtype=scalar.dtype.str, bytes=scalar.tobytes().hex())


def _same_array_bytes(left, right):
    """Compare unchanged dtype/NaN bits with at most 1 MiB temporary chunks."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    step = max(1, 2**20 // left.dtype.itemsize)
    for start in range(0, len(left), step):
        a = np.ascontiguousarray(left[start:start + step]).view(np.uint8)
        b = np.ascontiguousarray(right[start:start + step]).view(np.uint8)
        if not np.array_equal(a, b):
            return False
    return True


def _prepared_arrays(prepared, variants, samples, *, snapshot=True):
    required = {"cache_variant_count", "cache_sample_count", "columns", "samples",
                "exception_col", "exception_row", "exception_state", "summaries",
                "full_union_summaries"}
    if set(prepared) != required:
        raise ValueError("compact prepared fields differ from the exact cache contract")
    if (type(prepared["cache_variant_count"]) is not int
            or prepared["cache_variant_count"] != len(variants)
            or type(prepared["cache_sample_count"]) is not int
            or prepared["cache_sample_count"] != len(samples)):
        raise ValueError("compact geometry differs from requested axes")
    if type(prepared["full_union_summaries"]) is not bool:
        raise ValueError("full_union_summaries must be boolean")
    if not isinstance(prepared["summaries"], (tuple, list)) or len(prepared["summaries"]) != 5:
        raise ValueError("five unchanged allele summaries are required")
    arrays = {name: np.asarray(prepared[name]) for name in _ARRAY_NAMES[:4]}
    arrays.update({"summary_%d" % index: np.asarray(array)
                   for index, array in enumerate(prepared["summaries"])})
    output_samples = np.asarray(prepared["samples"])
    if (output_samples.ndim != 1 or output_samples.dtype.kind not in "iu"
            or len(output_samples) != len(samples) or np.any(output_samples < 0)
            or np.any(output_samples >= len(samples))):
        raise ValueError("invalid compact logical sample axis")
    columns = arrays["columns"]
    if (columns.ndim != 1 or columns.dtype.kind not in "iu" or np.any(columns < 0)
            or np.any(columns >= len(variants)) or len(np.unique(columns)) != len(columns)):
        raise ValueError("invalid compact retained columns")
    retained = len(columns)
    ec, er, es = (arrays[name] for name in _ARRAY_NAMES[1:4])
    if (any(a.ndim != 1 for a in (ec, er, es)) or len(ec) != len(er) or len(ec) != len(es)
            or ec.dtype.kind not in "iu" or er.dtype.kind not in "iu" or es.dtype != np.uint8
            or np.any(ec < 0) or np.any(ec >= retained)
            or np.any(er < 0) or np.any(er >= len(samples))
            or np.any(es < 1) or np.any(es > 5)):
        raise ValueError("invalid compact six-state exceptions")
    for name in _ARRAY_NAMES[4:]:
        array = arrays[name]
        if array.ndim != 1 or len(array) != retained or array.dtype.kind not in "iuf":
            raise ValueError("invalid unchanged compact allele summary")
    standard = np.array_equal(output_samples, np.arange(len(samples), dtype=output_samples.dtype))
    if not standard and len(np.unique(output_samples)) != len(output_samples):
        raise ValueError("compact logical sample axis contains duplicate rows")
    sample_descriptor = dict(reconstructed_arange=bool(standard), dtype=output_samples.dtype.str,
                             length=len(output_samples))
    if not standard:
        arrays["samples"] = output_samples
    # Owned immutable snapshots prevent a caller's later writes changing the blob.
    if snapshot:
        arrays = {name: np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=a.dtype)
                  for name, a in arrays.items()}
    return arrays, sample_descriptor


class CohortCompactCache:
    """Exact disk cache for a fixed ordered cohort; max_bytes is shared globally.

    A shared directory must be opened with the same max_bytes by every process.
    No eviction is performed: budget exhaustion returns False from store. A
    missing key returns None; a published corrupt/missing blob raises ValueError.
    Source binding must include population/cache and decoder-semantic hashes.
    """
    def __init__(self, directory, source_binding, max_bytes=DEFAULT_MAX_BYTES, metrics=None):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.directory = Path(directory)
        self.source_binding = json.loads(_json(source_binding))
        if not isinstance(self.source_binding, dict) or not self.source_binding:
            raise ValueError("an explicit nonempty source binding is required")
        self.max_bytes = max_bytes
        self.metrics = metrics if metrics is not None else {}
        self._metrics_lock = threading.Lock()
        self._sample_axis_lock = threading.Lock()
        self._sample_axis = None
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.directory / "locks").mkdir(exist_ok=True, mode=0o700)
        self.index = self.directory / "index.sqlite"
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, format TEXT NOT NULL, max_bytes INTEGER NOT NULL, used_bytes INTEGER NOT NULL CHECK(used_bytes>=0))")
            connection.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, state TEXT NOT NULL CHECK(state IN ('writing','ready')), blob_bytes INTEGER NOT NULL, charged_bytes INTEGER NOT NULL, sha256 TEXT, temporary TEXT)")
            row = connection.execute("SELECT format,max_bytes FROM config WHERE id=1").fetchone()
            if row is None:
                connection.execute("INSERT INTO config VALUES (1,?,?,0)", (FORMAT, max_bytes))
            elif row != (FORMAT, max_bytes):
                raise ValueError("shared compact cache format/global byte budget differs")
            connection.commit()
        try:
            os.chmod(self.index, 0o600)
        except FileNotFoundError:
            raise ValueError("compact cache index disappeared")

    @contextmanager
    def _database(self):
        connection = sqlite3.connect(self.index, timeout=60, isolation_level=None)
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _count(self, name, value=1):
        key = "cohort_compact_" + name
        with self._metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + value

    def close(self):
        """No persistent connections: mapped returned arrays remain valid."""

    def _identity(self, variants, samples, minimum_mac):
        variants, variant_descriptor = _axis(variants, "variants")
        samples = np.asarray(samples)
        if samples.ndim != 1 or samples.dtype.kind not in "iu" or np.any(samples < 0):
            raise ValueError("samples must be a one-dimensional nonnegative integer axis")
        with self._sample_axis_lock:
            previous = self._sample_axis
            if (previous is not None and samples.dtype == previous[0].dtype
                    and len(samples) == len(previous[0]) and np.array_equal(samples, previous[0])):
                sample_descriptor = previous[1]
                self._count("sample_axis_hash_reuse")
            else:
                samples, sample_descriptor = _axis(samples, "samples")
                snapshot = np.frombuffer(samples.tobytes(), dtype=samples.dtype)
                self._sample_axis = snapshot, sample_descriptor
                self._count("sample_axis_hashes")
        identity = dict(format=FORMAT, source_binding=self.source_binding,
                        variants=variant_descriptor, samples=sample_descriptor,
                        minimum_mac=_mac(minimum_mac))
        return hashlib.sha256(_json(identity).encode()).hexdigest(), identity, variants, samples

    @contextmanager
    def _key_lock(self, key):
        # A fixed 256-way flock table bounds control files even for cache misses.
        # Every occurrence of a key takes exactly the same exclusive file lock.
        path = self.directory / "locks" / (key[:2] + ".lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _row(self, key):
        with self._database() as connection:
            row = connection.execute("SELECT state,blob_bytes,charged_bytes,sha256,temporary FROM entries WHERE key=?", (key,)).fetchone()
            if row and row[0] == "writing":
                # The matching flock is held: no living writer can own this key.
                temporary = row[4]
                if (not isinstance(temporary, str) or not temporary.startswith(key + ".")
                        or not temporary.endswith(".tmp") or Path(temporary).name != temporary):
                    raise ValueError("invalid unfinished compact-cache reservation")
                for path in (self.directory / temporary, self.directory / (key + ".cpc")):
                    path.unlink(missing_ok=True)
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM entries WHERE key=? AND state='writing'", (key,))
                connection.execute("UPDATE config SET used_bytes=used_bytes-? WHERE id=1", (row[2],))
                connection.commit()
                self._count("recovered_writes")
                return None
            return row

    @staticmethod
    def _layout(identity, arrays, sample_descriptor, prepared):
        offset = 0
        descriptors = {}
        for name, array in arrays.items():
            offset = _align(offset)
            descriptors[name] = dict(dtype=array.dtype.str, shape=[len(array)],
                                     offset=offset, bytes=array.nbytes)
            offset += array.nbytes
        header = dict(format=FORMAT, identity=identity, arrays=descriptors,
                      payload_bytes=offset, samples=sample_descriptor,
                      cache_variant_count=prepared["cache_variant_count"],
                      cache_sample_count=prepared["cache_sample_count"],
                      full_union_summaries=prepared["full_union_summaries"])
        encoded = _json(header).encode()
        if len(encoded) > _MAX_HEADER_BYTES:
            raise ValueError("compact cache header exceeds bounded format limit")
        base = _align(16 + len(encoded))
        return header, encoded, base, base + offset + 32

    def _read(self, key, identity, variants, samples, row):
        path = self.directory / (key + ".cpc")
        try:
            with path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size != row[1]:
                    raise ValueError("compact cache blob size differs from committed index")
                prefix = stream.read(16)
                if len(prefix) != 16 or prefix[:8] != MAGIC:
                    raise ValueError("invalid compact cache blob magic")
                length = struct.unpack("<Q", prefix[8:])[0]
                if length > _MAX_HEADER_BYTES or length < 2:
                    raise ValueError("invalid compact cache header size")
                encoded = stream.read(length)
                header = json.loads(encoded)
                if header.get("format") != FORMAT or header.get("identity") != identity:
                    raise ValueError("compact cache blob source/cohort identity differs")
                digest = hashlib.sha256()
                stream.seek(0)
                remaining = row[1] - 32
                while remaining:
                    chunk = stream.read(min(2**20, remaining))
                    if not chunk:
                        raise ValueError("truncated compact cache blob")
                    digest.update(chunk)
                    remaining -= len(chunk)
                expected = stream.read(32)
                if digest.digest() != expected or digest.hexdigest() != row[3]:
                    raise ValueError("compact cache blob checksum mismatch")
                base = _align(16 + length)
                payload_bytes = header.get("payload_bytes")
                if type(payload_bytes) is not int or payload_bytes < 0 or base + payload_bytes + 32 != row[1]:
                    raise ValueError("invalid compact cache payload geometry")
                descriptors = header.get("arrays")
                sample_descriptor = header.get("samples")
                if not isinstance(descriptors, dict) or not isinstance(sample_descriptor, dict):
                    raise ValueError("missing compact cache array descriptors")
                expected_names = set(_ARRAY_NAMES)
                if sample_descriptor.get("reconstructed_arange") is False:
                    expected_names.add("samples")
                if set(descriptors) != expected_names:
                    raise ValueError("compact cache array set differs")
                # Map once using the already checksummed file descriptor. Views
                # share this mapping after the stream closes, retaining its inode.
                mapping = np.memmap(stream, mode="r", dtype=np.uint8, shape=(row[1],))
                arrays = {}
                end = 0
                for name, descriptor in descriptors.items():
                    dtype = np.dtype(descriptor["dtype"])
                    shape, offset, size = descriptor["shape"], descriptor["offset"], descriptor["bytes"]
                    if (dtype.kind not in "iuf" or not isinstance(shape, list) or len(shape) != 1
                            or type(shape[0]) is not int or shape[0] < 0
                            or type(offset) is not int or offset < 0 or offset % _ALIGNMENT
                            or type(size) is not int or size != shape[0] * dtype.itemsize
                            or offset + size > payload_bytes):
                        raise ValueError("invalid compact cache array geometry/dtype")
                    # JSON is sorted alphabetically; check overlaps separately.
                    arrays[name] = (np.ndarray(tuple(shape), dtype=dtype, buffer=mapping, offset=base + offset)
                                    if size else np.empty(shape, dtype=dtype))
                    arrays[name].flags.writeable = False
                    end = max(end, offset + size)
                intervals = sorted((d["offset"], d["offset"] + d["bytes"]) for d in descriptors.values() if d["bytes"])
                if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])) or end != payload_bytes:
                    raise ValueError("overlapping/incomplete compact cache payload")
                if (type(sample_descriptor.get("reconstructed_arange")) is not bool
                        or sample_descriptor.get("length") != len(samples)
                        or np.dtype(sample_descriptor["dtype"]).kind not in "iu"):
                    raise ValueError("invalid compact cached sample-axis descriptor")
                logical_samples = (np.arange(len(samples), dtype=np.dtype(sample_descriptor["dtype"]))
                                   if sample_descriptor["reconstructed_arange"] else arrays.pop("samples"))
                logical_samples.flags.writeable = False
                prepared = dict(cache_variant_count=header["cache_variant_count"],
                                cache_sample_count=header["cache_sample_count"],
                                full_union_summaries=header["full_union_summaries"],
                                samples=logical_samples,
                                summaries=tuple(arrays.pop("summary_%d" % i) for i in range(5)),
                                **arrays)
                # Verify semantic bounds without copying the mapped arrays.
                _prepared_arrays(prepared, variants, samples, snapshot=False)
                return prepared
        except (OSError, KeyError, TypeError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("published compact cache entry is missing or invalid") from error

    def load(self, variants, samples, minimum_mac):
        """Return exact read-only compact arrays, or None for an absent entry."""
        started = time.perf_counter()
        key, identity, variants, samples = self._identity(variants, samples, minimum_mac)
        try:
            with self._key_lock(key):
                row = self._row(key)
                if row is None:
                    self._count("misses")
                    return None
                prepared = self._read(key, identity, variants, samples, row)
                self._count("hits")
                self._count("read_bytes", row[1])
                return prepared
        except ValueError:
            self._count("errors")
            raise
        finally:
            self._count("load_seconds", time.perf_counter() - started)

    def store(self, variants, samples, minimum_mac, prepared):
        """Publish compact arrays once; False means shared budget exhausted."""
        started = time.perf_counter()
        key, identity, variants, samples = self._identity(variants, samples, minimum_mac)
        # Validate views and reserve globally before copying the compact payload.
        # A full cache must not add a payload-sized copy to every new request.
        arrays, sample_descriptor = _prepared_arrays(prepared, variants, samples, snapshot=False)
        header, encoded, base, size = self._layout(identity, arrays, sample_descriptor, prepared)
        charge = _align(size, _BLOCK) + _INDEX_ALLOWANCE
        temporary = key + "." + uuid.uuid4().hex + ".tmp"
        temporary_path = self.directory / temporary
        published_path = self.directory / (key + ".cpc")
        try:
            with self._key_lock(key):
                row = self._row(key)
                if row is not None:
                    existing = self._read(key, identity, variants, samples, row)
                    existing_arrays, existing_samples = _prepared_arrays(existing, variants, samples, snapshot=False)
                    if (existing_samples != sample_descriptor
                            or existing["full_union_summaries"] != prepared["full_union_summaries"]
                            or any(not _same_array_bytes(existing_arrays[name], array)
                                   for name, array in arrays.items())):
                        raise ValueError("immutable compact entry differs from recomputed exact arrays")
                    self._count("existing_writes")
                    return True
                with self._database() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    used = connection.execute("SELECT used_bytes FROM config WHERE id=1").fetchone()[0]
                    if charge > self.max_bytes - used:
                        connection.rollback()
                        self._count("budget_skips")
                        return False
                    connection.execute("INSERT INTO entries VALUES (?,?,?,?,?,?)", (key, "writing", size, charge, None, temporary))
                    connection.execute("UPDATE config SET used_bytes=used_bytes+? WHERE id=1", (charge,))
                    connection.commit()
                arrays, captured_samples = _prepared_arrays(prepared, variants, samples, snapshot=True)
                captured_header, captured_encoded, captured_base, captured_size = self._layout(
                    identity, arrays, captured_samples, prepared)
                if (captured_header != header or captured_encoded != encoded
                        or captured_base != base or captured_size != size):
                    # Retain the writing reservation for _row's locked recovery;
                    # a changed layout must never publish a ready entry.
                    raise ValueError("compact snapshot layout changed after byte-budget reservation")
                digest = hashlib.sha256()
                fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    def write(data):
                        stream.write(data)
                        digest.update(data)
                    write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
                    write(b"\0" * (base - stream.tell()))
                    for name, array in arrays.items():
                        position = base + header["arrays"][name]["offset"]
                        write(b"\0" * (position - stream.tell()))
                        write(memoryview(array).cast("B"))
                    stream.write(digest.digest())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, published_path)
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                with self._database() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("UPDATE entries SET state='ready',sha256=?,temporary=NULL WHERE key=? AND state='writing'", (digest.hexdigest(), key))
                    connection.commit()
                self._count("writes")
                self._count("written_bytes", size)
                return True
        finally:
            self._count("store_seconds", time.perf_counter() - started)
