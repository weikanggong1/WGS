"""CPU correctness units for compact Single packing; never a benchmark.

The fake reader uses the actual six-state allele meanings, including partial
calls. The materializer creates only tiny CPU tensors; no GPU or GDS is opened.
"""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staar_phewas.cache_runtime.single_batches import iter_effective_minor_blocks
from staar_phewas.gds import _allele_frequency_summary
from staar_phewas.gds_device import DeviceMinorBlock
from staar_phewas.pipeline import AnalysisOptions, PheWASPipeline


class FakeMaterializer:
    def __init__(self):
        self.snapshots = []

    def to_minor_block(self, prepared, samples, variants, *, device):
        assert device == 'cpu'
        source_samples, source_variants = np.asarray(samples), np.asarray(variants)
        # Mirror the real materializer's original-axis admission before
        # checking the selected compact axes. The fake expands only on CPU.
        assert len(source_samples) == prepared['cache_sample_count']
        assert len(source_variants) == prepared['cache_variant_count']
        assert len(np.unique(source_samples)) == len(source_samples)
        assert len(np.unique(source_variants)) == len(source_variants)
        assert np.all((prepared['samples'] >= 0) & (prepared['samples'] < len(source_samples)))
        assert np.all((prepared['columns'] >= 0) & (prepared['columns'] < len(source_variants)))
        selected_samples = source_samples[prepared['samples']]
        indices = source_variants[prepared['columns']]
        self.snapshots.append((
            {key: (tuple(a.copy() for a in value) if key == 'summaries'
                   else value.copy() if isinstance(value, np.ndarray) else value)
             for key, value in prepared.items()},
            selected_samples.copy(), indices.copy()))
        af, missing, mac, ref_ac, called = prepared['summaries']
        n, m = len(selected_samples), len(indices)
        # Encoding: RR, RA, AA, no call, R/NA, A/NA. Partial calls keep
        # their source allele evidence but become whole-genotype sentinel 3.
        dosage = np.broadcast_to(np.where(af >= .5, 0, 2), (n, m)).astype(np.uint8).copy()
        col, row, state = (prepared[key] for key in
                           ('exception_col', 'exception_row', 'exception_state'))
        values = np.where(state < 3, 2-state, 3)
        values = np.where((values != 3) & (af[col] >= .5), 2-values, values)
        dosage[row, col] = values
        return DeviceMinorBlock(torch.from_numpy(dosage), selected_samples.copy(), indices.copy(),
            af.copy(), mac.copy(), missing.copy(), ref_ac.copy(), called.copy())


class FakeAdapter:
    def __init__(self, n_variants=18, frame_width=4, n_samples=8):
        self.n_variants, self.n_samples = n_variants, n_samples
        self._starts = np.arange(0, n_variants, frame_width, dtype=np.int64)
        self._sizes = np.minimum(frame_width, n_variants-self._starts)
        self._closed, self._device = False, 'cpu'
        self._metrics = defaultdict(float)
        self._fast = FakeMaterializer()
        self.prepare_requests = []

    def make_prepared(self, variants, samples, minimum_mac):
        variants, samples = np.asarray(variants), np.asarray(samples)
        pattern = np.asarray([0, 1, 2, 4, 5, 3, 1, 0], dtype=np.uint8)
        states = pattern[(samples[:, None]+variants[None, :]) % len(pattern)]
        states[:, variants % 8 == 3] = 0  # genuinely monomorphic, MAC=0
        states[:, variants % 8 == 7] = 3  # genuinely all missing
        if self.n_samples >= 100:
            # Separate rare ALT, rare REF, and rare high-missing ALT cases
            # exercise every original extraction/group-order marker.
            for variant, default, partial in [(2, 0, False), (13, 2, False), (14, 0, True)]:
                selected = np.flatnonzero(variants == variant)
                if len(selected):
                    states[:, selected] = default
                    states[np.flatnonzero(samples == 0)[:, None], selected] = 1
                    if partial:
                        states[np.flatnonzero((samples >= 1) & (samples <= 8))[:, None], selected] = 4
        mapping = np.asarray([(2, 2), (1, 2), (0, 2), (0, 0), (1, 1), (0, 1)])
        ref_ac = mapping[states, 0].sum(0, dtype=np.int64)
        called = mapping[states, 1].sum(0, dtype=np.int64)
        summaries = _allele_frequency_summary(ref_ac, called, len(samples))
        columns = (np.arange(len(variants), dtype=np.int64) if minimum_mac is None
                   else np.flatnonzero(summaries[2] >= minimum_mac))
        states = states[:, columns]
        exception_col, exception_row, exception_state = [], [], []
        for j in range(len(columns)):
            rows = np.flatnonzero(states[:, j] != 0)
            exception_col.extend([j] * len(rows))
            exception_row.extend(rows.tolist())
            exception_state.extend(states[rows, j].tolist())
        prepared = dict(cache_variant_count=len(variants), cache_sample_count=len(samples),
            columns=columns, samples=np.arange(len(samples), dtype=np.int64),
            exception_col=np.asarray(exception_col, dtype=np.int64),
            exception_row=np.asarray(exception_row, dtype=np.int64),
            exception_state=np.asarray(exception_state, dtype=np.uint8),
            summaries=tuple(a[columns].copy() for a in summaries),
            full_union_summaries=len(samples) == self.n_samples)
        return prepared, samples.copy(), variants.copy()

    def _prepare(self, variants, samples, minimum_mac):
        self.prepare_requests.append(np.asarray(variants).copy())
        return self.make_prepared(variants, samples, minimum_mac)


def packed(adapter, variants, samples, *, raw_size=4, effective_size=3, minimum_mac=2):
    return list(iter_effective_minor_blocks(adapter, variants, samples,
        block_size=raw_size, effective_block_size=effective_size,
        device='cpu', minimum_mac=minimum_mac, resident=True))


def assert_exact_compact_axis(adapter, variants, samples, outputs, minimum_mac=2):
    expected, _, original = adapter.make_prepared(variants, samples, minimum_mac)
    retained = original[expected['columns']]
    np.testing.assert_array_equal(np.concatenate([x.variant_indices for x in outputs]), retained)
    for block in outputs:
        np.testing.assert_array_equal(block.sample_indices, samples)
    names = ('union_ref_af', 'union_missing_rate', 'union_initial_mac',
             'union_ref_ac', 'union_called_alleles')
    for i, name in enumerate(names):
        np.testing.assert_array_equal(np.concatenate([getattr(x, name) for x in outputs]),
                                      expected['summaries'][i])
    # Rebase packed exception columns to the original retained global axis.
    cols, rows, states = [], [], []
    offset = 0
    for prep, bound, actual in adapter._fast.snapshots:
        np.testing.assert_array_equal(bound, samples)
        np.testing.assert_array_equal(prep['columns'], np.arange(len(actual)))
        assert prep['cache_variant_count'] == len(actual)
        assert prep['cache_sample_count'] == len(samples)
        assert prep['exception_col'].dtype == np.int64
        assert np.all((prep['exception_col'] >= 0) & (prep['exception_col'] < len(actual)))
        cols.extend((prep['exception_col']+offset).tolist())
        rows.extend(prep['exception_row'].tolist())
        states.extend(prep['exception_state'].tolist())
        offset += len(actual)
    for actual, key in [(cols, 'exception_col'), (rows, 'exception_row'), (states, 'exception_state')]:
        np.testing.assert_array_equal(actual, expected[key])


def test_sorted_frames_prepare_once_pack_cross_frame_and_flush_tail():
    adapter = FakeAdapter()
    variants, samples = np.arange(18), np.asarray([7, 2, 5, 0, 6])
    outputs = packed(adapter, variants, samples)
    assert_exact_compact_axis(adapter, variants, samples, outputs)
    expected_count = sum(len(x.variant_indices) for x in outputs)
    assert [x.shape[1] for x in outputs] == [3]*(expected_count//3)+([expected_count % 3] if expected_count % 3 else [])
    assert len(adapter.prepare_requests) == len(adapter._starts)
    assert adapter._metrics['returned_variants'] == expected_count
    assert adapter._metrics['requested_variants'] == len(variants)
    assert adapter._metrics['single_effective_blocks'] == len(outputs)


def test_permuted_variant_and_sample_axes_are_not_sorted_by_packer():
    adapter = FakeAdapter()
    variants = np.asarray([16, 2, 12, 7, 1, 9, 15, 5, 0, 17, 10])
    samples = np.asarray([6, 1, 7, 0, 5])
    outputs = packed(adapter, variants, samples, effective_size=2)
    assert_exact_compact_axis(adapter, variants, samples, outputs)
    assert all(x.shape[1] <= 2 for x in outputs)


def test_sorted_sparse_axis_skips_frames_and_keeps_original_bindings():
    adapter = FakeAdapter()
    variants, samples = np.asarray([1, 2, 8, 9, 16, 17]), np.asarray([7, 0, 6, 1])
    outputs = packed(adapter, variants, samples, effective_size=3, minimum_mac=None)
    assert_exact_compact_axis(adapter, variants, samples, outputs, minimum_mac=None)
    assert [request.tolist() for request in adapter.prepare_requests] == [[1, 2], [8, 9], [16, 17]]
    assert [block.shape[1] for block in outputs] == [3, 3]


def test_oversized_compact_part_is_sliced_before_materialize():
    adapter = FakeAdapter(frame_width=18)
    variants, samples = np.arange(18), np.arange(8)
    outputs = packed(adapter, variants, samples, raw_size=18, effective_size=2)
    assert len(adapter.prepare_requests) == 1
    assert all(x.shape[1] <= 2 for x in outputs)
    assert_exact_compact_axis(adapter, variants, samples, outputs)


def test_half_missing_source_mac_is_not_recomputed_from_dosage():
    adapter = FakeAdapter()
    variants, samples = np.asarray([0, 1, 2]), np.arange(8)
    outputs = packed(adapter, variants, samples, effective_size=3)
    assert_exact_compact_axis(adapter, variants, samples, outputs)
    block = outputs[0]
    assert np.any(block.initial_mac() != block.observed_mac())
    assert any(np.any(np.isin(prep['exception_state'], [4, 5]))
               for prep, _, _ in adapter._fast.snapshots)
    for frequency in ('reference', 'count'):
        for imputation in ('mean', 'minor'):
            rows = np.asarray([6, 2, 0]) if frequency == 'count' else np.arange(8)
            expected, _, vv = adapter.make_prepared(variants, samples, 2)
            golden = FakeMaterializer().to_minor_block(expected, samples, vv, device='cpu')
            a, b = block.trait_dense(rows, imputation, frequency_mode=frequency), golden.trait_dense(rows, imputation, frequency_mode=frequency)
            for left, right in zip(a, b):
                np.testing.assert_array_equal(left.numpy() if torch.is_tensor(left) else left,
                                              right.numpy() if torch.is_tensor(right) else right)


@pytest.mark.parametrize('variants', [np.asarray([], dtype=np.int64), np.asarray([3, 7, 11, 15])])
def test_empty_or_all_filtered_requests_do_not_materialize(variants):
    adapter = FakeAdapter()
    assert packed(adapter, variants, np.arange(8)) == []
    assert adapter._fast.snapshots == []
    assert adapter._metrics['minor_block_calls'] == 0


def test_changed_sample_binding_is_rejected_before_materialize():
    adapter = FakeAdapter()
    original = adapter._prepare
    def changed(variants, samples, minimum_mac):
        prep, bound, vv = original(variants, samples, minimum_mac)
        return prep, bound[::-1].copy(), vv
    adapter._prepare = changed
    with pytest.raises(RuntimeError, match='sample axis'):
        packed(adapter, np.arange(4), np.arange(8))
    assert adapter._fast.snapshots == []


@pytest.mark.parametrize('size', [True, 0, -1, 3., '3'])
def test_invalid_effective_width_rejected_before_prepare(size):
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match='effective_block_size'):
        packed(adapter, np.arange(4), np.arange(8), effective_size=size)
    assert adapter.prepare_requests == []


def test_large_sample_geometry_rejected_before_any_device_allocation():
    adapter = FakeAdapter()
    adapter.n_samples = 339013
    with pytest.raises(MemoryError, match='int32 CUDA indexing'):
        packed(adapter, np.asarray([0]), np.arange(339013), effective_size=8192)
    assert adapter.prepare_requests == [] and adapter._fast.snapshots == []
    # No large tensor is constructed. Test the actual core estimate by shape.
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options = AnalysisOptions(memory_limit_gib=20.)
    pipeline.gds = SimpleNamespace(genotype_raw_memory_bytes=256*2**20)
    model = SimpleNamespace(n=339013, n_pheno=1, matmul_mode='tf32',
                            device='cpu', x=SimpleNamespace(shape=(339013, 23)))
    pipeline._limit(model, 2048, individual=True)
    with pytest.raises(MemoryError, match='Single workspace'):
        pipeline._limit(model, 4096, individual=True)


def pipeline_records(blocks, samples, n_variants, mac_cutoff=2):
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    n = len(samples)
    def score_variance(genotype):
        # Deterministic tiny formula tests routing/order only; never benchmark.
        return genotype.sum(0)+.25, (genotype*genotype).sum(0)+1
    pipeline.models = [SimpleNamespace(n=n, n_pheno=1, use_spa=False,
        device='cpu', matmul_mode='tf32', x=torch.ones((n, 1), dtype=torch.float32),
        individual_score_variance=score_variance)]
    pipeline.options = AnalysisOptions(wrapper_semantics='base')
    pipeline.resident_genotypes = True
    pipeline.trait_rows, pipeline.union_rows = [np.arange(n)], samples
    pipeline.position = np.arange(n_variants)*10+100
    pipeline._base_mask = lambda *args: np.ones(n_variants, dtype=bool)
    pipeline._minor_blocks = lambda *args, **kwargs: iter(blocks)
    pipeline._limit = lambda *args, **kwargs: None
    labels = np.asarray(['T', 'A', 'CG', 'G'])
    pipeline.gds = SimpleNamespace(n_variants=n_variants,
        read_field=lambda name, selected: np.full(len(selected), '1'),
        read_ref_alt=lambda selected: (labels[selected % 4], labels[(selected+1) % 4]))
    return [row for _, rows in pipeline.iter_individual_records('1', mac_cutoff=mac_cutoff,
                                                               subset_variants_num=3) for row in rows]


@pytest.mark.parametrize('effective_size', [2, 3, 5])
def test_packing_preserves_global_chunk_groups_native_row_names_and_factors(effective_size):
    variants, samples = np.arange(18), np.asarray([7, 2, 5, 0, 6])
    raw = FakeAdapter();raw_blocks = []
    for start in range(0, len(variants), 4):
        prep, bound, vv = raw.make_prepared(variants[start:start+4], samples, 2)
        raw_blocks.append(raw._fast.to_minor_block(prep, bound, vv, device='cpu'))
    compact = FakeAdapter()
    merged = packed(compact, variants, samples, effective_size=effective_size)
    a, b = pipeline_records(raw_blocks, samples, len(variants)), pipeline_records(merged, samples, len(variants))
    assert len(a) == len(b)
    for left, right in zip(a, b):
        for key in ('CHR', 'POS', 'REF', 'ALT', 'N', '_chunk', '_common', '_base_group'):
            assert left[key] == right[key]
        for key in set(left)-{'CHR', 'POS', 'REF', 'ALT', 'N', '_chunk', '_common', '_base_group'}:
            assert left[key] == pytest.approx(right[key], abs=1e-7, rel=1e-7)
    old = PheWASPipeline.individual_tables([a])[0]
    new = PheWASPipeline.individual_tables([b])[0]
    assert old.row_names == new.row_names
    assert old.factor_levels == new.factor_levels
    assert [(r['CHR'], r['POS'], r['REF'], r['ALT']) for r in old] == [
            (r['CHR'], r['POS'], r['REF'], r['ALT']) for r in new]


def test_rare_common_and_ref_orientation_groups_survive_pack_boundaries():
    variants, samples = np.arange(18), np.arange(255, -1, -1)
    raw = FakeAdapter(n_samples=256);raw_blocks = []
    for start in range(0, len(variants), 4):
        prep, bound, vv = raw.make_prepared(variants[start:start+4], samples, 1)
        raw_blocks.append(raw._fast.to_minor_block(prep, bound, vv, device='cpu'))
    compact = FakeAdapter(n_samples=256)
    merged = packed(compact, variants, samples, effective_size=2, minimum_mac=1)
    a = pipeline_records(raw_blocks, samples, len(variants), mac_cutoff=1)
    b = pipeline_records(merged, samples, len(variants), mac_cutoff=1)
    assert {row['_common'] for row in a} == {False, True}
    assert {row['_base_group'] for row in a} == {0, 1, 2}
    for key in ('POS', 'REF', 'ALT', 'N', '_chunk', '_common', '_base_group'):
        assert [row[key] for row in a] == [row[key] for row in b]
    old = PheWASPipeline.individual_tables([a])[0]
    new = PheWASPipeline.individual_tables([b])[0]
    assert old.row_names == new.row_names
    assert old.factor_levels == new.factor_levels
