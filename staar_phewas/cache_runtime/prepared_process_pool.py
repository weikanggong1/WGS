"""Ordered CPU-only spawn preparation with verified, bounded file handoff.

Children own independent readers. Only variant axes and small scalar reports
cross IPC; prepared arrays are verified and mapped by the parent loader. A
shared-cache budget skip hands off the existing preparation once, then disables
new pool dispatch. Closing waits for reads/stores before removing private files.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import atexit
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np


_CHILD = None
_SAMPLES = None
_HANDOFF_ROOT = None


def _identity():
    result = dict(pid=os.getpid(), rss_bytes=0)
    try:
        status = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
        result["start_ticks"] = int(status[19])
        result["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        result["rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    return result


def _handoff_usage(directory, delta=0):
    """Account transient charged payloads across independent CPU children."""
    with (Path(directory) / "handoff_usage.json").open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        value = json.load(stream)
        current = value["current_bytes"] + delta
        if current < 0:
            raise RuntimeError("negative transient handoff accounting")
        if delta > 0 and current > value["limit_bytes"] and value["current_bytes"]:
            raise MemoryError("transient handoffs exceed bounded pool memory budget")
        value["current_bytes"] = current
        value["peak_bytes"] = max(value["peak_bytes"], current)
        if delta:
            stream.seek(0)
            json.dump(value, stream)
            stream.truncate()
            stream.flush()
        return value


def _parent_alive(owner):
    # Check the process, not the transient thread that spawned this pool.
    if os.getppid() != owner["pid"]:
        return False
    if "start_ticks" not in owner:
        return True
    try:
        fields = Path("/proc/%d/stat" % owner["pid"]).read_text().split(") ", 1)[1].split()
        return (fields[0] != "Z" and int(fields[19]) == owner["start_ticks"]
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == owner["boot_id"])
    except (OSError, ValueError, IndexError):
        return False


def _watch_parent(owner):
    # A process exit releases its file locks. An interrupted atomic-cache store
    # remains a recoverable 'writing' reservation, never a published result.
    sleeper = threading.Event()
    while _parent_alive(owner):
        sleeper.wait(1.0)
    os._exit(70)


def _initialize(descriptor, sample_path, sample_sha, handoff_root, owner, startup_barrier):
    global _CHILD, _SAMPLES, _HANDOFF_ROOT
    # spawn imports the package before this initializer; set the environment in
    # the parent during spawn as well, then enforce any already-loaded BLAS pool.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if not _parent_alive(owner):
        os._exit(70)
    threading.Thread(target=_watch_parent, args=(owner,), daemon=True,
        name="torchstaar-cpu-parent-watch").start()
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation child inherited an initialized CUDA context")
    _SAMPLES = np.load(sample_path, mmap_mode="r", allow_pickle=False)
    if hashlib.sha256(_SAMPLES.tobytes()).hexdigest() != sample_sha:
        raise ValueError("CPU preparation sample axis checksum differs")
    from .portable import PortableCachedGDS
    _CHILD = PortableCachedGDS(descriptor["container_directory"], descriptor["metadata_directory"],
        device="cpu", compact_cache_bytes=descriptor["compact_cache_bytes"],
        verify_checksums=descriptor["verify_checksums"],
        prepared_cache_directory=descriptor["prepared_cache_directory"],
        prepared_cache_max_bytes=descriptor["prepared_cache_max_bytes"],
        prefetch_depth=0, prefetch_processes=0)
    if _CHILD._prepared_cache.source_binding != descriptor["source_binding"]:
        _CHILD.close()
        raise ValueError("CPU preparation decoder/population source binding differs")
    _HANDOFF_ROOT = Path(handoff_root)
    atexit.register(_CHILD.close)
    from multiprocessing.util import Finalize
    Finalize(_CHILD, _CHILD.close, exitpriority=10)
    ready = dict(_identity(), cuda_initialized=torch.cuda.is_initialized(),
        parent_identity_guard="process_pid_start_ticks_boot_id_1s",
        source_binding_sha256=hashlib.sha256(json.dumps(
            _CHILD._prepared_cache.source_binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    ready_path = _HANDOFF_ROOT / ("child-%d.ready.json" % os.getpid())
    temporary = ready_path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(ready, stream)
    os.replace(temporary, ready_path)
    # All N initializers must complete before any lightweight ready task can be
    # reused by the fastest child. Failure/timeout is fatal to the executor.
    startup_barrier.wait(timeout=60)


def _ready():
    import torch
    return dict(_identity(), cuda_initialized=torch.cuda.is_initialized())


def _prepare_task(ordinal, variant_bytes, minimum_mac, estimate):
    from .cohort_compact_cache import CohortCompactCache, _prepared_arrays, _align, _BLOCK, _INDEX_ALLOWANCE
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation child initialized CUDA")
    variants = np.frombuffer(variant_bytes, dtype=np.int64)
    before = dict(_CHILD._metrics)
    wall, cpu = time.perf_counter(), time.process_time()
    prepared, samples, vv = _CHILD._prepare(variants, _SAMPLES, minimum_mac)
    delta = {key: value - before.get(key, 0) for key, value in _CHILD._metrics.items()}
    key, identity, _, _ = _CHILD._prepared_cache._identity(vv, samples, minimum_mac)
    handoff = None
    handoff_budget = None
    if delta.get("cohort_compact_budget_skips", 0):
        # Reuse the result already computed: no second population decode.
        handoff = _HANDOFF_ROOT / str(ordinal)
        arrays, sample_descriptor = _prepared_arrays(prepared, vv, samples, snapshot=False)
        _, _, _, size = CohortCompactCache._layout(identity, arrays, sample_descriptor, prepared)
        handoff_budget = _align(size, _BLOCK) + _INDEX_ALLOWANCE
        if handoff_budget > estimate:
            raise MemoryError("transient compact size exceeds conservative dispatch estimate")
        _handoff_usage(_HANDOFF_ROOT, handoff_budget)
        transient = CohortCompactCache(handoff, _CHILD._prepared_cache.source_binding, handoff_budget)
        if not transient.store(vv, samples, minimum_mac, prepared):
            raise RuntimeError("bounded transient compact handoff unexpectedly exhausted")
        transient.close()
    return dict(ordinal=ordinal, key=key, handoff=None if handoff is None else str(handoff),
                handoff_budget=handoff_budget, metrics=delta, identity=_identity(),
                wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                cuda_initialized=torch.cuda.is_initialized())


class PreparedProcessPool:
    """One reusable pool for one reader and immutable analysis sample axis.

    ``memory_bytes`` bounds estimated transient CPU preparation and ready-file
    handoff bytes, in addition to the parent prepared queue. Child interpreters,
    mapped metadata/sample axes and each frame LRU have separate baseline RSS;
    production host admission must include the complete process tree. An
    oversized request runs alone and is reported, rather than deadlocking.
    """
    def __init__(self, adapter, samples, processes, memory_bytes):
        self.adapter, self.processes = adapter, processes
        self.memory_bytes = memory_bytes
        self.pending_limit = 2 * processes
        self.samples, _ = adapter._bind_samples(samples)
        self.descriptor = adapter._process_descriptor
        self._lock = threading.Lock()
        self._workers = {}
        self._executor = None
        self._closed = False
        self._disabled = False
        self._ordinal = 0
        self._pending_bytes = 0
        self._headers = {}
        self._parent_identity = _identity()
        self._temporary = Path(tempfile.mkdtemp(prefix=".cpu-prepared-", dir=adapter._prepared_cache.directory))
        os.chmod(self._temporary, 0o700)
        sample_path = self._temporary / "samples.npy"
        np.save(sample_path, self.samples, allow_pickle=False)
        os.chmod(sample_path, 0o600)
        usage_path = self._temporary / "handoff_usage.json"
        usage_path.write_text(json.dumps(dict(current_bytes=0, peak_bytes=0, limit_bytes=memory_bytes)))
        os.chmod(usage_path, 0o600)
        self._initializer = (self.descriptor, str(sample_path),
            hashlib.sha256(self.samples.tobytes()).hexdigest(), str(self._temporary), self._parent_identity)
        from .cohort_compact_cache import CohortCompactCache
        self._verification_cache = CohortCompactCache(adapter._prepared_cache.directory,
            self.descriptor["source_binding"], adapter._prepared_cache.max_bytes)
        # Below 8192 bytes no valid blob can be charged; also recognize a full
        # existing shared cache before paying spawn startup or duplicate work.
        with adapter._prepared_cache._database() as connection:
            budget, used = connection.execute("SELECT max_bytes,used_bytes FROM config WHERE id=1").fetchone()
        if budget-used < 8192:
            self._disabled = True

    def _add(self, name, amount=1):
        name = "prefetch_process_" + name
        with self._lock:
            self.adapter._metrics[name] = self.adapter._metrics.get(name, 0) + amount

    def metadata(self):
        with self._lock:
            workers = [dict(value) for value in self._workers.values()]
        usage = self._usage() if not self._closed else dict(current_bytes=0,
            peak_bytes=self.adapter._metrics.get("prefetch_process_transient_peak_bytes", 0))
        active = len(self._executor._processes) if self._executor is not None else 0
        return dict(requested_processes=self.processes, active_processes=active,
            disabled_after_budget_skip=self._disabled, closed=self._closed,
            pending_limit=self.pending_limit, pending_memory_budget_bytes=self.memory_bytes,
            pending_estimated_bytes_current=self._pending_bytes,
            parent_startup_rss_bytes=self._parent_identity["rss_bytes"], workers=workers,
            child_parent_death_guard="process_pid_start_ticks_boot_id_1s",
            transient_current_bytes=usage["current_bytes"], transient_peak_bytes=usage["peak_bytes"],
            transient_limit_bytes=self.memory_bytes, transient_oversized_single_request_allowed=True,
            memory_contract="estimated scratch/ready handoff bounded separately from parent queue; child baseline, metadata mmap and frame LRU require process-tree host admission")

    def _usage(self):
        value = _handoff_usage(self._temporary)
        with self._lock:
            self.adapter._metrics["prefetch_process_transient_current_bytes"] = value["current_bytes"]
            self.adapter._metrics["prefetch_process_transient_peak_bytes"] = value["peak_bytes"]
        return value

    def _check_executor(self):
        """Propagate a failed child even when a request hits cache or falls back."""
        if self._closed:
            raise RuntimeError("CPU preparation pool is closed")
        executor = self._executor
        if executor is None:
            return
        from concurrent.futures.process import BrokenProcessPool
        if executor._broken:
            raise BrokenProcessPool(executor._broken)
        processes = executor._processes
        if processes is None or any(not child.is_alive() for child in processes.values()):
            raise BrokenProcessPool("CPU preparation child exited unexpectedly")

    def _start(self):
        if self._executor is None:
            # OpenBLAS may initialize while spawn imports NumPy before the
            # initializer. Restore parent environment immediately after spawn.
            names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            prior = {name: os.environ.get(name) for name in names}
            try:
                for name in names:
                    os.environ[name] = "1"
                context = multiprocessing.get_context("spawn")
                self._executor = ProcessPoolExecutor(max_workers=self.processes,
                    mp_context=context, initializer=_initialize,
                    initargs=(*self._initializer, context.Barrier(self.processes)))
                # Submit while the environment is constrained: NumPy is
                # imported before the initializer in a spawned interpreter.
                startup = time.perf_counter()
                futures = [self._executor.submit(_ready) for _ in range(self.processes)]
                for future in futures:
                    future.result()
                processes = self._executor._processes
                if len(processes) != self.processes:
                    raise RuntimeError("CPU preparation startup did not initialize every requested process")
                expected_binding = hashlib.sha256(json.dumps(self.descriptor["source_binding"],
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                for pid, process in processes.items():
                    value = json.loads((self._temporary / ("child-%d.ready.json" % pid)).read_text())
                    if (value["pid"] != pid or value["cuda_initialized"] is not False
                            or value["source_binding_sha256"] != expected_binding or not process.is_alive()):
                        raise RuntimeError("CPU preparation child startup identity/CUDA/source verification failed")
                    with self._lock:
                        self._workers[pid] = value
                self._add("startup_seconds", time.perf_counter()-startup)
            finally:
                for name, value in prior.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def _estimate(self, variants):
        from . import sparse_codec_fast
        frames = np.unique(np.searchsorted(self.adapter._starts, variants, side="right")-1)
        raw = 0
        headers_path = Path(self.descriptor["container_directory"]) / "headers.bin"
        for frame in frames:
            frame = int(frame)
            if frame not in self._headers:
                row = self.adapter._index[frame]
                with headers_path.open("rb") as stream:
                    stream.seek(int(row["header_offset"]))
                    header = json.loads(stream.read(int(row["header_size"])))
                meta = header["csr_meta"]
                m, n, _, raw_bytes = sparse_codec_fast._header(meta, sparse_codec_fast.DEFAULT_MAX_BYTES)
                if (m != int(row["m"]) or n != len(self.adapter._samples)
                        or meta["compressed_bytes"] != int(row["size"])
                        or type(header["counts_raw_bytes"]) is not int or header["counts_raw_bytes"] != m*24
                        or header["counts_compressed_bytes"] != int(row["counts_size"])):
                    raise ValueError("CPU preparation estimate source geometry differs")
                self._headers[frame] = raw_bytes + header["counts_raw_bytes"]
            raw += self._headers[frame]
        # Six-state reindexing/concatenation and immutable store snapshot expand
        # compact uint indices. Include ordered request/sample bookkeeping.
        return 32*raw + 48*len(self.samples) + 64*len(variants) + 2**20

    def _collect(self, future, variants, minimum_mac):
        start = time.perf_counter()
        try:
            report = future.result()  # BrokenProcessPool propagates; never silent retry.
        except BaseException:
            self._add("future_errors")
            raise
        self._add("future_wait_seconds", time.perf_counter()-start)
        with self._lock:
            identity = report["identity"]
            previous = self._workers[identity["pid"]]
            if any(identity.get(key) != previous.get(key) for key in ("start_ticks", "boot_id")):
                raise RuntimeError("CPU preparation child process identity changed")
            self._workers[identity["pid"]] = dict(previous, **identity,
                cuda_initialized=report["cuda_initialized"])
        for key, value in report["metrics"].items():
            label = {"cohort_compact_store_seconds": "child_store_seconds"}.get(key, "child_"+key)
            self._add(label, value)
        self._add("child_cpu_seconds", report["cpu_seconds"])
        self._add("child_wall_seconds", report["wall_seconds"])
        self._add("cold_prepared", int(not report["metrics"].get("cohort_compact_hits", 0)))
        self.adapter._add_compact_prepare_seconds(report["metrics"]["compact_prepare_seconds"])
        # Existing decoder counters now cover parent + child work honestly.
        for key in ("frame_loads", "cache_read_validate_seconds"):
            self.adapter._metrics[key] += report["metrics"][key]
        for key in ("cohort_compact_writes", "cohort_compact_written_bytes", "cohort_compact_store_seconds",
                    "cohort_compact_budget_skips", "cohort_compact_errors"):
            value = report["metrics"].get(key, 0)
            self.adapter._metrics[key] = self.adapter._metrics.get(key, 0) + value
        if report["handoff"] is not None:
            self._disabled = True
            self._add("budget_skips")
            self._add("handoff_requests")
            self._usage()
            from .cohort_compact_cache import CohortCompactCache
            path = Path(report["handoff"])
            if path.parent != self._temporary:
                raise ValueError("CPU handoff escaped its private pool directory")
            cache = CohortCompactCache(path, self.descriptor["source_binding"], report["handoff_budget"])
        else:
            path, cache = None, self._verification_cache
        before_load = {key: cache.metrics.get(key, 0) for key in
            ("cohort_compact_load_seconds", "cohort_compact_read_bytes")}
        start = time.perf_counter()
        try:
            prepared = cache.load(variants, self.samples, minimum_mac)
            if prepared is None:
                raise ValueError("CPU prepared result missing from committed cache")
            key, *_ = cache._identity(variants, self.samples, minimum_mac)
            if key != report["key"]:
                raise ValueError("CPU prepared result request identity differs")
        finally:
            self._add("parent_load_seconds", time.perf_counter()-start)
            if path is None:
                # Cold publication verification is IO, not a warm cache hit.
                for key, previous in before_load.items():
                    self.adapter._metrics[key] = self.adapter._metrics.get(key, 0) + cache.metrics.get(key, 0)-previous
            if path is not None:
                cache.close()
                # Arrays keep their verified mapping inode after Linux unlink.
                shutil.rmtree(path)
                _handoff_usage(self._temporary, -report["handoff_budget"])
                self._usage()
        return prepared, self.samples, variants

    def iter_prepared(self, requests, minimum_mac):
        from ..gds import _indices
        source = iter(requests)
        pending = deque()
        pending_bytes = 0
        held = None
        done = False
        try:
            while pending or not done or held is not None:
                self._check_executor()
                while not done and len(pending) < self.pending_limit:
                    if held is None:
                        try:
                            request = next(source)
                        except StopIteration:
                            done = True
                            break
                        variants = self.adapter._immutable_axis(_indices(request, self.adapter.n_variants, "variant_indices"))
                        estimate = self._estimate(variants)
                        held = variants, estimate
                    variants, estimate = held
                    if pending and pending_bytes+estimate > self.memory_bytes:
                        break
                    if self._disabled or variants.nbytes > 8*2**20:
                        # Drain already dispatched requests before falling back.
                        if pending:
                            break
                        held = None
                        self._add("fallback_requests")
                        yield self.adapter._prepare(variants, self.samples, minimum_mac)
                        continue
                    start = time.perf_counter()
                    cached = self.adapter._prepared_cache.load(variants, self.samples, minimum_mac)
                    self._add("parent_load_seconds", time.perf_counter()-start)
                    if cached is not None:
                        self._add("warm_hits")
                        pending.append((None, variants, estimate, cached))
                    else:
                        self._start()
                        ordinal = self._ordinal
                        self._ordinal += 1
                        future = self._executor.submit(_prepare_task, ordinal, variants.tobytes(), minimum_mac, estimate)
                        pending.append((future, variants, estimate, None))
                        self._add("requests_submitted")
                    held = None
                    pending_bytes += estimate
                    with self._lock:
                        self._pending_bytes = pending_bytes
                        self.adapter._metrics["prefetch_process_pending_estimated_bytes_current"] = pending_bytes
                        key = "prefetch_process_pending_estimated_bytes_max"
                        self.adapter._metrics[key] = max(self.adapter._metrics.get(key, 0), pending_bytes)
                    if pending_bytes >= self.memory_bytes:
                        if estimate > self.memory_bytes:
                            self._add("oversized_requests")
                        break
                if pending:
                    future, variants, estimate, cached = pending.popleft()
                    try:
                        value = ((cached, self.samples, variants) if future is None else
                                 self._collect(future, variants, minimum_mac))
                    finally:
                        pending_bytes -= estimate
                        with self._lock:
                            self._pending_bytes = pending_bytes
                            self.adapter._metrics["prefetch_process_pending_estimated_bytes_current"] = pending_bytes
                    yield value
        finally:
            # Do not close underlying readers while children read or own locks.
            for future, variants, _, _ in pending:
                if future is None or future.cancel():
                    continue
                try:
                    self._collect(future, variants, minimum_mac)
                except BaseException:
                    # Preserve the consumer/producer's primary exception. A
                    # subsequent use of a broken executor still raises visibly.
                    pass
            with self._lock:
                self._pending_bytes = 0
                self.adapter._metrics["prefetch_process_pending_estimated_bytes_current"] = 0
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
        finally:
            shutil.rmtree(self._temporary)
