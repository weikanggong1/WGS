"""CPU concurrency contracts; these fixtures are not a scientific benchmark."""
import threading
import time
import unittest

import numpy as np

from fudan_wgs_toolkit.cache_runtime.prepared_prefetch import _prepared_nbytes, iter_prepared


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("prefetch condition did not become true")
        threading.Event().wait(0.005)


class Adapter:
    def __init__(self, item_bytes=6, hook=None):
        self._metrics = {"existing_decoder_counter": 7}
        self.item_bytes = item_bytes
        self.hook = hook
        self.calls = []
        self.results = []

    def _prepare(self, request, samples, minimum_mac):
        self.calls.append((request.copy(), samples, minimum_mac, threading.get_ident()))
        if self.hook is not None:
            self.hook(len(self.calls) - 1)
        prepared = {"payload": np.full(self.item_bytes, int(request[0]), dtype=np.uint8),
                    "axis_view": samples[::2], "variant_view": request[:],
                    "summaries": ()}
        result = prepared, samples, request
        self.results.append(result)
        return result

    def close(self):
        raise AssertionError("prefetch must not close the adapter")

    def materialize(self, *args, **kwargs):
        raise AssertionError("prefetch must not materialize data")


def requests(count):
    return [np.array([index], dtype=np.int64) for index in range(count)]


class PreparedPrefetchContracts(unittest.TestCase):
    def setUp(self):
        self.samples = np.array([4, 1, 6], dtype=np.int64)

    def test_exact_tuple_order_axis_and_cpu_thread(self):
        adapter = Adapter()
        expected_requests = requests(9)
        actual = list(iter_prepared(adapter, expected_requests, self.samples, 20))
        self.assertEqual(len(actual), 9)
        for index, result in enumerate(actual):
            self.assertIs(result, adapter.results[index])
            self.assertIs(result[1], self.samples)
            self.assertIs(result[2], expected_requests[index])
            np.testing.assert_array_equal(result[0]["payload"], np.full(6, index, dtype=np.uint8))
            call = adapter.calls[index]
            np.testing.assert_array_equal(call[0], expected_requests[index])
            self.assertIs(call[1], self.samples)
            self.assertEqual(call[2], 20)
            self.assertNotEqual(call[3], threading.get_ident())
        self.assertEqual(len({call[3] for call in adapter.calls}), 1)
        self.assertEqual(adapter._metrics["existing_decoder_counter"], 7)
        self.assertEqual(adapter._metrics["prefetch_yielded_items"], 9)
        self.assertEqual(adapter._metrics["prefetch_total_queued_bytes"], 54)
        self.assertEqual(adapter._metrics["prefetch_queued_bytes"], 0)
        self.assertEqual(adapter._metrics["prefetch_queued_items"], 0)

    def test_empty_requests_and_close_before_first_next(self):
        adapter = Adapter()
        self.assertEqual(list(iter_prepared(adapter, [], self.samples, None)), [])
        self.assertEqual(adapter.calls, [])
        self.assertEqual(adapter._metrics["prefetch_sessions"], 1)
        self.assertEqual(adapter._metrics["prefetch_enqueued_items"], 0)
        unopened = iter_prepared(adapter, requests(2), self.samples, None)
        unopened.close()
        self.assertEqual(adapter.calls, [])
        self.assertEqual(adapter._metrics["prefetch_sessions"], 1)

    def test_positive_integer_limits_are_checked_before_start(self):
        for name in ("max_items", "max_bytes"):
            for invalid in (0, -1, 1.5, True, None, np.int64(2)):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    iter_prepared(Adapter(), [], self.samples, None, **{name: invalid})

    def test_item_limit_and_early_close_wake_full_queue(self):
        adapter = Adapter()
        iterator = iter_prepared(adapter, requests(20), self.samples, 20, max_items=2)
        try:
            next(iterator)
            wait_for(lambda: adapter._metrics.get("prefetch_queued_items") == 2)
            self.assertEqual(len(adapter.calls), 3)
            self.assertEqual(adapter._metrics["prefetch_peak_queued_items"], 2)
        finally:
            iterator.close()
        self.assertEqual(len(adapter.calls), 3)
        self.assertEqual(adapter._metrics["prefetch_queued_bytes"], 0)
        self.assertEqual(adapter._metrics["prefetch_queued_items"], 0)
        self.assertGreater(adapter._metrics["prefetch_producer_wait_seconds"], 0)

    def test_byte_limit_includes_only_unique_prepared_buffers(self):
        adapter = Adapter(item_bytes=6)
        iterator = iter_prepared(adapter, requests(20), self.samples, None, max_items=8, max_bytes=10)
        try:
            next(iterator)
            wait_for(lambda: len(adapter.calls) >= 3)
            self.assertEqual(adapter._metrics["prefetch_queued_items"], 1)
            self.assertEqual(adapter._metrics["prefetch_queued_bytes"], 6)
            self.assertEqual(adapter._metrics["prefetch_peak_queued_bytes"], 6)
        finally:
            iterator.close()
        self.assertEqual(len(adapter.calls), 3)

    def test_oversized_item_waits_alone_without_further_preparation(self):
        adapter = Adapter(item_bytes=16)
        iterator = iter_prepared(adapter, requests(20), self.samples, None, max_items=4, max_bytes=8)
        try:
            first = next(iterator)
            self.assertEqual(first[0]["payload"].nbytes, 16)
            wait_for(lambda: adapter._metrics.get("prefetch_queued_bytes") == 16)
            self.assertEqual(len(adapter.calls), 2)
            self.assertEqual(adapter._metrics["prefetch_queued_items"], 1)
            second = next(iterator)
            self.assertEqual(int(second[2][0]), 1)
            wait_for(lambda: len(adapter.calls) == 3)
            self.assertLessEqual(adapter._metrics["prefetch_peak_queued_items"], 1)
            self.assertEqual(adapter._metrics["prefetch_peak_queued_bytes"], 16)
        finally:
            iterator.close()

    def test_close_joins_in_flight_prepare_before_reader_can_close(self):
        entered = threading.Event()
        release = threading.Event()
        exited = threading.Event()
        closed = threading.Event()

        def block_second(index):
            if index == 1:
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test did not release preparation")
                exited.set()

        adapter = Adapter(hook=block_second)
        iterator = iter_prepared(adapter, requests(10), self.samples, None)
        next(iterator)
        self.assertTrue(entered.wait(3))

        def close_iterator():
            iterator.close()
            closed.set()

        closer = threading.Thread(target=close_iterator)
        closer.start()
        try:
            self.assertFalse(closed.wait(0.03))
            self.assertFalse(exited.is_set())
        finally:
            release.set()
            closer.join(3)
        self.assertFalse(closer.is_alive())
        self.assertTrue(exited.is_set())
        self.assertTrue(closed.is_set())
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(adapter._metrics["prefetch_queued_bytes"], 0)

    def test_consumer_error_stops_and_joins_producer(self):
        adapter = Adapter()
        iterator = iter_prepared(adapter, requests(30), self.samples, None, max_items=1)
        next(iterator)
        wait_for(lambda: adapter._metrics.get("prefetch_queued_items") == 1)
        error = RuntimeError("consumer failed")
        with self.assertRaises(RuntimeError) as caught:
            iterator.throw(error)
        self.assertIs(caught.exception, error)
        self.assertEqual(adapter._metrics["prefetch_queued_bytes"], 0)
        self.assertEqual(len(adapter.calls), 2)

    def test_prepare_error_preserves_preceding_results_and_original_error(self):
        error = ValueError("CPU prepare failed")

        def fail_third(index):
            if index == 2:
                raise error

        adapter = Adapter(hook=fail_third)
        iterator = iter_prepared(adapter, requests(10), self.samples, None, max_items=4)
        self.assertEqual(int(next(iterator)[2][0]), 0)
        self.assertEqual(int(next(iterator)[2][0]), 1)
        with self.assertRaises(ValueError) as caught:
            next(iterator)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(adapter.calls), 3)
        self.assertEqual(adapter._metrics["prefetch_queued_items"], 0)

    def test_request_iterator_error_is_propagated(self):
        error = RuntimeError("request iteration failed")

        def source():
            yield np.array([0], dtype=np.int64)
            raise error

        adapter = Adapter()
        iterator = iter_prepared(adapter, source(), self.samples, None)
        self.assertEqual(int(next(iterator)[2][0]), 0)
        with self.assertRaises(RuntimeError) as caught:
            next(iterator)
        self.assertIs(caught.exception, error)

    def test_metrics_accumulate_across_sessions_without_touching_decoder_keys(self):
        adapter = Adapter()
        for count in (2, 3):
            list(iter_prepared(adapter, requests(count), self.samples, None))
        self.assertEqual(adapter._metrics["prefetch_sessions"], 2)
        self.assertEqual(adapter._metrics["prefetch_total_queued_bytes"], 30)
        self.assertEqual(adapter._metrics["prefetch_enqueued_items"], 5)
        self.assertEqual(adapter._metrics["prefetch_yielded_items"], 5)
        self.assertEqual(adapter._metrics["existing_decoder_counter"], 7)
        self.assertGreater(adapter._metrics["prefetch_prepare_wall_seconds"], 0)
        self.assertGreaterEqual(adapter._metrics["prefetch_producer_wall_seconds"],
                                adapter._metrics["prefetch_prepare_wall_seconds"])

    def test_numpy_backing_aliases_and_shared_axes_are_counted_once(self):
        owner = np.arange(16, dtype=np.int64)
        variants = np.arange(10, dtype=np.int64)
        prepared = {"one": owner[:4], "two": owner[4:], "same": owner,
                    "nested": (self.samples[:], {"variants": variants[::2]})}
        self.assertEqual(_prepared_nbytes(prepared, self.samples, variants), owner.nbytes)
        immutable = bytes(range(16))
        prepared = (np.frombuffer(immutable, dtype=np.uint8, count=4),
                    np.frombuffer(immutable, dtype=np.uint8, offset=4))
        self.assertEqual(_prepared_nbytes(prepared, self.samples, variants), len(immutable))

    def test_actual_cpu_adapter_matches_synchronous_six_state_preparation(self):
        from fudan_wgs_toolkit.cache_runtime.adapter_fast import CachedGenotypeAdapter
        from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast

        class Reader:
            n_variants = 7
            n_samples = 6

        class Frames:
            complete = True
            samples = np.arange(6, dtype=np.int64)
            index = np.array([(0, 3), (3, 3), (6, 1)], dtype=[("start", "i8"), ("m", "i8")])
            raw = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0],
                            [3, 3, 3, 3, 3, 3], [5, 4, 3, 2, 1, 0],
                            [0, 1, 0, 1, 0, 1], [2, 2, 2, 2, 2, 2],
                            [1, 4, 5, 0, 2, 3]], dtype=np.uint8)

            def read_frame(self, index):
                start, count = self.index[index]
                raw = self.raw[start:start + count]
                offsets, rows, state = sparse_codec_fast.compact(raw)
                counts = sparse_codec_fast.integer_counts(offsets, rows, state, *raw.shape)
                return dict(start=int(start), m=int(count), n=6, offsets=offsets,
                            sample_index=rows, state=state, reference_alleles=counts["reference_alleles"],
                            called_alleles=counts["called_alleles"])

        for samples in (np.arange(6), np.array([5, 1, 3]), np.empty(0, dtype=np.int64)):
            for cutoff in (None, 0, 2, 20):
                with self.subTest(samples=samples, cutoff=cutoff):
                    adapter = CachedGenotypeAdapter(Reader(), Frames(), own_reader=False)
                    synchronous = CachedGenotypeAdapter(Reader(), Frames(), own_reader=False)
                    blocks = [np.array([6, 0, 3]), np.array([4, 1, 5, 2]), np.empty(0, dtype=np.int64)]
                    actual = list(iter_prepared(adapter, blocks, samples, cutoff, max_bytes=64))
                    expected = [synchronous._prepare(block, samples, cutoff) for block in blocks]
                    for (prepared, ss, vv), (reference, rss, rvv) in zip(actual, expected):
                        np.testing.assert_array_equal(ss, rss)
                        np.testing.assert_array_equal(vv, rvv)
                        self.assertEqual(set(prepared), set(reference))
                        for name, value in prepared.items():
                            if isinstance(value, np.ndarray):
                                np.testing.assert_array_equal(value, reference[name])
                                self.assertEqual(value.dtype, reference[name].dtype)
                            elif name == "summaries":
                                for summary, rsummary in zip(value, reference[name]):
                                    np.testing.assert_array_equal(summary, rsummary)
                                    self.assertEqual(summary.dtype, rsummary.dtype)
                            else:
                                self.assertEqual(value, reference[name])


if __name__ == "__main__":
    unittest.main()
