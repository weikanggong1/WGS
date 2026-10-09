"""Preparation/packing counter concurrency, without CUDA or benchmark claims."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np
import pytest

from staar_phewas.cache_runtime.adapter_fast import CachedGDSAdapter
from staar_phewas.cache_runtime.single_batches import iter_effective_minor_blocks
from test_cache_adapter import Container, MetadataReader


class YieldingMetrics(dict):
    """Force a scheduling opportunity between a counter read and its write."""
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if key == 'compact_prepare_seconds':
            time.sleep(0.0001)
        return value


def test_concurrent_prepare_and_pack_accumulation_does_not_lose_updates():
    adapter = CachedGDSAdapter(MetadataReader(), Container())
    adapter._metrics = YieldingMetrics(adapter._metrics)
    initial = 17.0
    adapter._metrics['compact_prepare_seconds'] = initial
    barrier = Barrier(2)

    def contribute(amount):
        barrier.wait(timeout=3)
        for _ in range(120):
            adapter._add_compact_prepare_seconds(amount)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contribute, amount) for amount in (1.0, 2.0)]
        for future in futures:
            future.result(timeout=5)
    assert adapter._metrics['compact_prepare_seconds'] == initial + 120 * (1.0 + 2.0)


def test_real_prefetch_and_single_pack_share_atomic_counter():
    adapter = CachedGDSAdapter(MetadataReader(), Container(), prefetch_depth=2)
    samples = np.array([3, 9, 1], dtype=np.int64)
    variants = np.array([8, 0, 5, 3, 7], dtype=np.int64)
    original = adapter._add_compact_prepare_seconds
    contributions = []

    def record(seconds):
        contributions.append((get_ident(), seconds))
        original(seconds)

    def materialize(prepared, bound_samples, selected_variants, **kwargs):
        return SimpleNamespace(shape=(len(bound_samples), len(prepared['columns'])),
                               variant_indices=selected_variants[prepared['columns']].copy())

    with patch.object(adapter, '_add_compact_prepare_seconds', side_effect=record), \
            patch.object(adapter._fast, 'to_minor_block', side_effect=materialize):
        blocks = list(iter_effective_minor_blocks(adapter, variants, samples,
                      block_size=2, effective_block_size=2, resident=True, device='cpu'))
    np.testing.assert_array_equal(np.concatenate([block.variant_indices for block in blocks]), variants)
    consumer = [seconds for identity, seconds in contributions if identity == get_ident()]
    producer = [seconds for identity, seconds in contributions if identity != get_ident()]
    assert len(consumer) == 3 and len(producer) == 3
    assert adapter._metrics['compact_prepare_seconds'] == pytest.approx(
        sum(consumer) + sum(producer), rel=0, abs=1e-12)
    assert adapter._metrics['single_pack_seconds'] == pytest.approx(sum(consumer), rel=0, abs=1e-12)
    assert adapter._metrics['prefetch_queued_bytes'] == 0
    assert adapter._metrics['prefetch_queued_items'] == 0


def test_single_synchronous_adapter_without_atomic_helper_remains_compatible():
    original = CachedGDSAdapter(MetadataReader(), Container())
    legacy = SimpleNamespace(_closed=False, _device='cpu', _starts=original._starts,
        _sizes=original._sizes, _fast=original._fast, _metrics=original._metrics,
        _prepare=original._prepare, n_variants=original.n_variants, n_samples=original.n_samples)
    samples = np.array([3, 9, 1], dtype=np.int64)
    variants = np.array([8, 0, 5, 3, 7], dtype=np.int64)
    contributions = []
    helper = original._add_compact_prepare_seconds

    def record(seconds):
        contributions.append(seconds)
        helper(seconds)

    def materialize(prepared, bound_samples, selected_variants, **kwargs):
        return SimpleNamespace(shape=(len(bound_samples), len(prepared['columns'])),
                               variant_indices=selected_variants[prepared['columns']].copy())

    with patch.object(original, '_add_compact_prepare_seconds', side_effect=record), \
            patch.object(original._fast, 'to_minor_block', side_effect=materialize):
        blocks = list(iter_effective_minor_blocks(legacy, variants, samples,
                      block_size=2, effective_block_size=2, resident=True, device='cpu'))
    np.testing.assert_array_equal(np.concatenate([block.variant_indices for block in blocks]), variants)
    assert len(contributions) == 3
    assert legacy._metrics['compact_prepare_seconds'] == pytest.approx(
        sum(contributions) + legacy._metrics['single_pack_seconds'], rel=0, abs=1e-12)
