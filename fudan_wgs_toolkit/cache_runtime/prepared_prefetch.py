"""Bounded CPU compact preparation ahead of a GPU consumer.

Only ``adapter._prepare`` runs in the producer thread. Materialization, CUDA,
metadata reads and result writing remain the caller's responsibility. Close the
returned generator in the caller's ``finally`` block, before closing the reader.
Cancellation wakes queue waits and joins any in-flight preparation; it cannot
interrupt a CPU read safely.
"""
from collections import deque
from collections.abc import Mapping, MutableMapping
import threading
import time

import numpy as np


_METRICS_LOCK = threading.Lock()
_DEFAULT_METRICS = {
    "prefetch_sessions": 0,
    "prefetch_producer_wall_seconds": 0.0,
    "prefetch_prepare_wall_seconds": 0.0,
    "prefetch_producer_wait_seconds": 0.0,
    "prefetch_consumer_wait_seconds": 0.0,
    "prefetch_queued_bytes": 0,
    "prefetch_queued_items": 0,
    "prefetch_peak_queued_bytes": 0,
    "prefetch_peak_queued_items": 0,
    "prefetch_total_queued_bytes": 0,
    "prefetch_enqueued_items": 0,
    "prefetch_yielded_items": 0,
}


def _record(metrics, **increments):
    # All prefetch metric read/modify/write operations share this lock, including
    # successive iterators using the same adapter. Existing decoder keys are not
    # read or rewritten here.
    with _METRICS_LOCK:
        for name, initial in _DEFAULT_METRICS.items():
            metrics.setdefault(name, initial)
        for name, amount in increments.items():
            metrics[name] += amount
        metrics["prefetch_peak_queued_bytes"] = max(
            metrics["prefetch_peak_queued_bytes"], metrics["prefetch_queued_bytes"])
        metrics["prefetch_peak_queued_items"] = max(
            metrics["prefetch_peak_queued_items"], metrics["prefetch_queued_items"])


def _storage(array):
    """Identify a NumPy backing allocation, rather than each aliasing view."""
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    backing = owner.base
    if backing is None:
        return id(owner), owner.nbytes
    while isinstance(backing, memoryview):
        backing = backing.obj
    try:
        size = memoryview(backing).nbytes
    except TypeError:
        size = owner.nbytes
    return id(backing), size


def _prepared_nbytes(prepared, samples, variants):
    """Count unique prepared NumPy buffers, excluding shared sample/ID axes.

    The budget covers NumPy backing storage, including nested summaries. Python
    container overhead, the consumer's current item, the adapter's frame cache
    and one possible in-flight preparation are separate from this queue budget.
    """
    excluded = {_storage(axis)[0] for axis in (samples, variants)
                if isinstance(axis, np.ndarray)}
    allocations = {}
    visited = set()

    def visit(value):
        if isinstance(value, np.ndarray):
            key, size = _storage(value)
            if key not in excluded:
                allocations[key] = max(size, allocations.get(key, 0))
        elif isinstance(value, (Mapping, list, tuple)):
            if id(value) in visited:
                return
            visited.add(id(value))
            children = value.values() if isinstance(value, Mapping) else value
            for child in children:
                visit(child)

    visit(prepared)
    return sum(allocations.values())


def iter_prepared(adapter, requests, samples, minimum_mac, *, max_items=2,
                  max_bytes=256 * 2**20, _prepared_results=None):
    """Yield unchanged ``(prepared, ss, vv)`` tuples in request order.

    ``requests`` supplies variant-index arrays to ``adapter._prepare``. At most
    ``max_items`` completed requests wait in the queue, with at most ``max_bytes``
    unique prepared NumPy backing bytes. An item exceeding the byte budget can
    enter an empty queue alone; no following preparation starts while it waits.

    The adapter must have one preparation producer and remain open until this
    iterator has been exhausted or explicitly closed. Shared sample/request
    arrays must not be mutated while the iterator is active. Producer errors
    propagate after preceding successful items have been consumed. No copying,
    filtering, CUDA calls or numerical transformations are added here.

    ``prefetch_*_wall_seconds`` and wait counters accumulate in ``_metrics``.
    Producer wall includes its queue waits; prepare wall times only ``_prepare``.
    Queue gauges exclude yielded and in-flight items; peaks and total enqueued
    bytes remain cumulative across iterators. These overlapping times must not
    be added as independent end-to-end stages.
    """
    if type(max_items) is not int or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    metrics = adapter._metrics
    if not isinstance(metrics, MutableMapping):
        raise TypeError("adapter._metrics must be a mutable mapping")

    def generate():
        condition = threading.Condition()
        queue = deque()
        queued_bytes = 0
        stopped = False
        finished = False
        failure = None
        _record(metrics, prefetch_sessions=1)

        def producer_wait():
            start = time.perf_counter()
            condition.wait()
            _record(metrics, prefetch_producer_wait_seconds=time.perf_counter() - start)

        def produce():
            nonlocal queued_bytes, finished, failure
            start = time.perf_counter()
            source = None
            try:
                source = iter(requests if _prepared_results is None else _prepared_results)
                while True:
                    with condition:
                        while not stopped and (len(queue) >= max_items or queued_bytes >= max_bytes):
                            producer_wait()
                        if stopped:
                            return
                    try:
                        request = next(source)
                    except StopIteration:
                        return
                    prepare_start = time.perf_counter()
                    if _prepared_results is None:
                        try:
                            result = adapter._prepare(request, samples, minimum_mac)
                        finally:
                            _record(metrics, prefetch_prepare_wall_seconds=time.perf_counter() - prepare_start)
                    else:
                        result = request
                    prepared, ss, vv = result
                    size = _prepared_nbytes(prepared, ss, vv)
                    with condition:
                        while not stopped and (len(queue) >= max_items or
                                (queue and queued_bytes + size > max_bytes)):
                            producer_wait()
                        if stopped:
                            return
                        queue.append((result, size))
                        queued_bytes += size
                        _record(metrics, prefetch_queued_bytes=size, prefetch_queued_items=1,
                                prefetch_total_queued_bytes=size, prefetch_enqueued_items=1)
                        condition.notify_all()
            except BaseException as error:
                with condition:
                    failure = error
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as error:
                        if failure is None:
                            failure = error
                _record(metrics, prefetch_producer_wall_seconds=time.perf_counter() - start)
                with condition:
                    finished = True
                    condition.notify_all()

        producer = threading.Thread(target=produce, name="fudan_wgs_toolkit-compact-prefetch", daemon=True)
        try:
            producer.start()
            while True:
                with condition:
                    while not queue and not finished:
                        start = time.perf_counter()
                        condition.wait()
                        _record(metrics, prefetch_consumer_wait_seconds=time.perf_counter() - start)
                    if not queue:
                        if failure is not None:
                            raise failure
                        return
                    result, size = queue.popleft()
                    queued_bytes -= size
                    _record(metrics, prefetch_queued_bytes=-size, prefetch_queued_items=-1,
                            prefetch_yielded_items=1)
                    condition.notify_all()
                yield result
        finally:
            with condition:
                stopped = True
                _record(metrics, prefetch_queued_bytes=-queued_bytes,
                        prefetch_queued_items=-len(queue))
                queue.clear()
                queued_bytes = 0
                condition.notify_all()
            if producer.ident is not None:
                producer.join()

    return generate()
