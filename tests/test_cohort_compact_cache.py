"""Exact compact persistence, corruption and multiprocess admission contracts."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fudan_wgs_toolkit.cache_runtime.cohort_compact_cache import CohortCompactCache, MAGIC
from fudan_wgs_toolkit.cache_runtime import cohort_compact_cache, sparse_codec_fast, sparse_decode_fast


BINDING = {"population_sha256": "a" * 64, "decoder_sha256": "b" * 64}
RAW = np.array([[0, 1, 2, 3, 4, 5, 0], [4, 0, 0, 1, 5, 0, 0],
                [5, 4, 0, 0, 2, 1, 0], [0, 0, 0, 0, 0, 0, 0],
                [2, 2, 1, 4, 5, 3, 0], [3, 3, 3, 3, 3, 3, 3]], dtype=np.uint8)


def prepared(variants=None, samples=None, minimum_mac=None):
    vv = np.array([4, 0, 1, 5], dtype=np.int64) if variants is None else variants
    ss = np.array([6, 3, 1, 0, 5], dtype=np.int64) if samples is None else samples
    offsets, indices, states = sparse_codec_fast.compact(RAW)
    counts = sparse_codec_fast.integer_counts(offsets, indices, states, *RAW.shape)
    source = sparse_decode_fast.validate_source(offsets, indices, states,
                counts["reference_alleles"], counts["called_alleles"], RAW.shape[1])
    value = sparse_decode_fast.prepare_validated(source, vv, ss, minimum_mac=minimum_mac)
    lookup = {int(column): position for position, column in enumerate(vv)}
    value["columns"] = np.array([lookup[int(column)] for column in value["columns"]], dtype=np.int64)
    value.update(cache_variant_count=len(vv), cache_sample_count=len(ss), samples=np.arange(len(ss), dtype=np.int64))
    return vv, ss, value


def concurrent_store(args):
    directory, variant, budget, identical = args
    vv = np.array([0 if identical else variant], dtype=np.int64)
    ss = np.array([0, 1, 3, 5], dtype=np.int64)
    _, _, value = prepared(vv, ss)
    cache = CohortCompactCache(directory, BINDING, budget)
    result = cache.store(vv, ss, None, value)
    loaded = cache.load(vv, ss, None) if result else None
    return result, loaded is not None


class CohortCompactCacheContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.cache = CohortCompactCache(self.directory, BINDING, max_bytes=2**20)

    def tearDown(self):
        self.cache.close()
        self.temporary.cleanup()

    def assertExact(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in ("cache_variant_count", "cache_sample_count", "full_union_summaries"):
            self.assertEqual(actual[name], expected[name])
        for name in ("columns", "samples", "exception_col", "exception_row", "exception_state"):
            self.assertEqual(actual[name].dtype, expected[name].dtype)
            np.testing.assert_array_equal(actual[name].view(np.uint8), expected[name].view(np.uint8))
            self.assertFalse(actual[name].flags.writeable)
        for left, right in zip(actual["summaries"], expected["summaries"]):
            self.assertEqual(left.dtype, right.dtype)
            np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))
            self.assertFalse(left.flags.writeable)

    def test_six_states_odd_called_summaries_and_mmap_exact(self):
        for cutoff in (None, 0, 2, 20):
            vv, ss, value = prepared(minimum_mac=cutoff)
            self.assertIsNone(self.cache.load(vv, ss, cutoff))
            self.assertTrue(self.cache.store(vv, ss, cutoff, value))
            actual = self.cache.load(vv, ss, cutoff)
            self.assertExact(actual, value)
            if len(actual["exception_state"]):
                self.assertIsInstance(actual["exception_state"].base, np.memmap)
                self.assertIs(actual["exception_state"].base, actual["columns"].base)
            self.assertTrue(self.cache.store(vv, ss, cutoff, value))
            self.cache.close()
            self.assertExact(actual, value)
        vv, ss, value = prepared()
        self.assertTrue(np.any(value["exception_state"] >= 4))
        self.assertTrue(np.any(value["summaries"][4] % 2 == 1))

    def test_order_dtype_source_and_mac_scalar_are_exact_identity(self):
        vv, ss, value = prepared()
        self.assertTrue(self.cache.store(vv, ss, None, value))
        for axes in ((vv[::-1], ss), (vv, ss[::-1]), (vv.astype(np.int32), ss),
                     (vv, ss.astype(np.int32)), (vv[:2], ss), (vv, ss[:2])):
            self.assertIsNone(self.cache.load(*axes, None))
        other = CohortCompactCache(self.directory, dict(BINDING, decoder_sha256="c" * 64), 2**20)
        self.assertIsNone(other.load(vv, ss, None))
        _, _, value = prepared(minimum_mac=2)
        self.assertTrue(self.cache.store(vv, ss, 2, value))
        for cutoff in (np.int32(2), 2.0, np.float32(2), None, 3):
            if cutoff is None:
                self.assertIsNotNone(self.cache.load(vv, ss, cutoff))
            else:
                self.assertIsNone(self.cache.load(vv, ss, cutoff))

    def test_sample_hash_reuse_revalidates_values_after_caller_mutation(self):
        vv, ss, value = prepared()
        self.cache.store(vv, ss, None, value)
        self.assertIsNotNone(self.cache.load(vv, ss, None))
        self.assertEqual(self.cache.metrics["cohort_compact_sample_axis_hashes"], 1)
        self.assertGreater(self.cache.metrics["cohort_compact_sample_axis_hash_reuse"], 0)
        ss[[0, 1]] = ss[[1, 0]]
        self.assertIsNone(self.cache.load(vv, ss, None))
        self.assertEqual(self.cache.metrics["cohort_compact_sample_axis_hashes"], 2)

    def test_arange_samples_are_not_stored_and_noncanonical_axis_is_preserved(self):
        vv, ss, value = prepared()
        self.cache.store(vv, ss, None, value)
        key, *_ = self.cache._identity(vv, ss, None)
        with (self.directory / (key + ".cpc")).open("rb") as stream:
            prefix = stream.read(16)
            self.assertEqual(prefix[:8], MAGIC)
            header = json.loads(stream.read(struct.unpack("<Q", prefix[8:])[0]))
        self.assertTrue(header["samples"]["reconstructed_arange"])
        self.assertNotIn("samples", header["arrays"])
        value["samples"] = np.arange(len(ss)-1, -1, -1, dtype=np.int32)
        other = CohortCompactCache(self.directory, dict(BINDING, decoder_sha256="d" * 64), 2**20)
        self.assertTrue(other.store(vv, ss, None, value))
        self.assertExact(other.load(vv, ss, None), value)

    def test_bitwise_nan_payload_and_original_integer_dtypes(self):
        vv, ss, value = prepared()
        value["columns"] = value["columns"].astype(">u4")
        value["exception_col"] = value["exception_col"].astype(np.uint16)
        value["exception_row"] = value["exception_row"].astype(np.uint32)
        summaries = list(value["summaries"])
        summaries[0] = np.array([0x7ff8000000000072] * len(value["columns"]), dtype=np.uint64).view(np.float64)
        value["summaries"] = tuple(summaries)
        self.assertTrue(self.cache.store(vv, ss, None, value))
        self.assertExact(self.cache.load(vv, ss, None), value)

    def test_corruption_or_missing_published_file_fails_closed(self):
        vv, ss, value = prepared()
        self.cache.store(vv, ss, None, value)
        key, *_ = self.cache._identity(vv, ss, None)
        path = self.directory / (key + ".cpc")
        with path.open("r+b") as stream:
            stream.seek(-35, 2)
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.cache.load(vv, ss, None)
        with self.assertRaises(ValueError):
            self.cache.store(vv, ss, None, value)
        path.unlink()
        with self.assertRaises(ValueError):
            self.cache.load(vv, ss, None)

    def test_immutable_key_rejects_different_recomputed_arrays(self):
        vv, ss, value = prepared()
        self.cache.store(vv, ss, None, value)
        summaries = list(value["summaries"])
        summaries[0] = summaries[0] + .01
        value["summaries"] = tuple(summaries)
        with self.assertRaisesRegex(ValueError, "recomputed"):
            self.cache.store(vv, ss, None, value)

    def test_invalid_object_and_bad_exception_bounds_never_publish(self):
        vv, ss, value = prepared()
        value["exception_state"] = value["exception_state"].astype(object)
        with self.assertRaises(ValueError):
            self.cache.store(vv, ss, None, value)
        vv, ss, value = prepared()
        value["exception_row"][0] = len(ss)
        with self.assertRaises(ValueError):
            self.cache.store(vv, ss, None, value)
        self.assertEqual(list(self.directory.glob("*.cpc")), [])

    def test_empty_mask_and_empty_sample_axes_preserve_exact_empty_arrays(self):
        for vv, ss in ((np.empty(0, dtype=np.int64), np.array([0, 2], dtype=np.int64)),
                       (np.array([0, 1], dtype=np.int64), np.empty(0, dtype=np.int64))):
            _, _, value = prepared(vv, ss)
            self.assertTrue(self.cache.store(vv, ss, None, value))
            self.assertExact(self.cache.load(vv, ss, None), value)

    def test_budget_and_shared_configuration_are_global(self):
        tiny = CohortCompactCache(self.directory / "tiny", BINDING, max_bytes=1)
        vv, ss, value = prepared()
        self.assertFalse(tiny.store(vv, ss, None, value))
        self.assertIsNone(tiny.load(vv, ss, None))
        with self.assertRaisesRegex(ValueError, "budget"):
            CohortCompactCache(self.directory, BINDING, max_bytes=2**19)
        with sqlite3.connect(tiny.index) as connection:
            self.assertEqual(connection.execute("SELECT used_bytes FROM config").fetchone()[0], 0)

    def test_budget_skip_and_existing_key_never_snapshot_whole_payload(self):
        vv, ss, value = prepared()
        tiny = CohortCompactCache(self.directory / "tiny", BINDING, max_bytes=1)
        original = cohort_compact_cache._prepared_arrays
        with patch.object(cohort_compact_cache, "_prepared_arrays", wraps=original) as capture:
            self.assertFalse(tiny.store(vv, ss, None, value))
            self.assertTrue(capture.call_args_list)
            self.assertTrue(all(call.kwargs.get("snapshot", True) is False for call in capture.call_args_list))
        self.assertTrue(self.cache.store(vv, ss, None, value))
        with patch.object(cohort_compact_cache, "_prepared_arrays", wraps=original) as capture:
            self.assertTrue(self.cache.store(vv, ss, None, value))
            self.assertTrue(all(call.kwargs.get("snapshot", True) is False for call in capture.call_args_list))

    def test_snapshot_layout_drift_cannot_publish_ready_entry(self):
        vv, ss, value = prepared()
        original = cohort_compact_cache._prepared_arrays

        def drift(prepared_value, variants, samples, *, snapshot=True):
            arrays, descriptor = original(prepared_value, variants, samples, snapshot=snapshot)
            if snapshot:
                arrays["exception_row"] = arrays["exception_row"].astype(np.uint32)
            return arrays, descriptor

        with patch.object(cohort_compact_cache, "_prepared_arrays", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "layout changed"):
                self.cache.store(vv, ss, None, value)
        self.assertEqual(list(self.directory.glob("*.cpc")), [])
        with sqlite3.connect(self.cache.index) as connection:
            self.assertEqual(connection.execute("SELECT state FROM entries").fetchone()[0], "writing")
        self.assertIsNone(self.cache.load(vv, ss, None))
        self.assertTrue(self.cache.store(vv, ss, None, value))

    def test_stale_unfinished_reservation_is_recovered_under_key_lock(self):
        vv, ss, value = prepared()
        key, *_ = self.cache._identity(vv, ss, None)
        filename = key + ".dead.tmp"
        (self.directory / filename).write_bytes(b"unfinished")
        with sqlite3.connect(self.cache.index) as connection:
            connection.execute("INSERT INTO entries VALUES (?,?,?,?,?,?)", (key, "writing", 12, 8192, None, filename))
            connection.execute("UPDATE config SET used_bytes=8192")
        self.assertIsNone(self.cache.load(vv, ss, None))
        self.assertFalse((self.directory / filename).exists())
        self.assertTrue(self.cache.store(vv, ss, None, value))
        with sqlite3.connect(self.cache.index) as connection:
            self.assertEqual(connection.execute("SELECT used_bytes FROM config").fetchone()[0],
                             connection.execute("SELECT SUM(charged_bytes) FROM entries").fetchone()[0])

    def test_crossprocess_budget_and_same_key_publication(self):
        context = multiprocessing.get_context("fork")
        for identical in (False, True):
            target = self.directory / ("same" if identical else "different")
            budget = 2 * 8192
            args = [(str(target), index % len(RAW), budget, identical) for index in range(8)]
            with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
                results = list(pool.map(concurrent_store, args))
            self.assertTrue(all(success == readable for success, readable in results))
            with sqlite3.connect(target / "index.sqlite") as connection:
                used = connection.execute("SELECT used_bytes FROM config").fetchone()[0]
                rows = connection.execute("SELECT COUNT(*),COALESCE(SUM(charged_bytes),0) FROM entries").fetchone()
            self.assertLessEqual(used, budget)
            self.assertEqual(used, rows[1])
            self.assertEqual(rows[0], 1 if identical else 2)
            if identical:
                self.assertTrue(all(success for success, _ in results))
            self.assertEqual(list(target.glob("*.tmp")), [])

    def test_prefetch_threads_share_key_lock_and_metrics(self):
        vv, ss, value = prepared()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.cache.store(vv, ss, None, value), range(16)))
        self.assertTrue(all(results))
        self.assertEqual(self.cache.metrics["cohort_compact_writes"], 1)
        self.assertEqual(self.cache.metrics["cohort_compact_existing_writes"], 15)
        self.assertEqual(len(list(self.directory.glob("*.cpc"))), 1)

    def test_failed_atomic_publish_keeps_budget_reserved_until_recovery(self):
        vv, ss, value = prepared()
        with patch("fudan_wgs_toolkit.cache_runtime.cohort_compact_cache.os.replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.cache.store(vv, ss, None, value)
        with sqlite3.connect(self.cache.index) as connection:
            self.assertEqual(connection.execute("SELECT state FROM entries").fetchone()[0], "writing")
            self.assertGreater(connection.execute("SELECT used_bytes FROM config").fetchone()[0], 0)
        self.assertIsNone(self.cache.load(vv, ss, None))
        self.assertEqual(list(self.directory.glob("*.tmp")), [])
        self.assertTrue(self.cache.store(vv, ss, None, value))


if __name__ == "__main__":
    unittest.main()
