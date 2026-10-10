"""Edge-case unit checks, separate from genuine data benchmarks."""
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype import SparseMinorBlock
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
import fudan_wgs_toolkit.pipeline as pipeline_module
import fudan_wgs_toolkit.batch_statistics as batch_module


def sparse_block(genotype, ref_af, indices=None):
    rows, cols = np.nonzero((genotype != 0) | np.isnan(genotype))
    return SparseMinorBlock(rows, cols, genotype[rows, cols], np.arange(len(genotype)),
        np.arange(genotype.shape[1]) if indices is None else indices, np.asarray(ref_af))


@pytest.mark.parametrize("imputation", ["mean", "minor"])
def test_reference_frequency_tie_uses_alt_and_keeps_observed_mac20(imputation):
    calls = np.asarray([2.]*5 + [0.]*5 + [1.]*10 + [np.nan])[:, None]
    block = sparse_block(calls, [0.5])
    old, old_maf, old_mac, old_missing, old_alt = block.trait_dense(np.arange(21), imputation)
    ref, ref_maf, ref_mac, ref_missing, ref_alt = block.trait_dense(np.arange(21), imputation,
        frequency_mode="reference")
    expected_old = calls.copy()
    expected_ref = calls.copy()
    replacement = 1. if imputation == "mean" else 0.
    expected_old[-1] = expected_ref[-1] = replacement
    np.testing.assert_array_equal(old, expected_old)
    np.testing.assert_array_equal(ref, expected_ref)
    np.testing.assert_array_equal(old_mac, [20.])
    np.testing.assert_array_equal(ref_mac, old_mac)
    np.testing.assert_array_equal(ref_missing, old_missing)
    np.testing.assert_array_equal(ref_maf, old_maf)
    assert old_alt[0] and ref_alt[0]
    reversed_calls = block.trait_dense(np.arange(20, -1, -1), imputation, frequency_mode="reference")[0]
    np.testing.assert_array_equal(reversed_calls, expected_ref[::-1])


@pytest.mark.parametrize("semantics,expected", [
    ("base", [1, 4, 2, 3, 0]), ("phewas", [0, 1, 2, 3, 4])])
@pytest.mark.parametrize("imputation", ["mean", "minor"])
@pytest.mark.parametrize("has_kinship", [False, True])
def test_base_groups_stably_reorder_genotype_frequency_and_annotations(monkeypatch, semantics, expected, imputation, has_kinship):
    genotype = np.zeros((100, 5))
    genotype[0] = [1, 1, 2, 1, 1]
    genotype[1, 2] = 2
    genotype[-2:, 3] = np.nan
    count = 100 - np.isnan(genotype).sum(axis=0)
    minor_af = np.nansum(genotype, axis=0) / (2 * count)
    ref_af = np.where(np.arange(5) == 0, minor_af, 1 - minor_af)
    blocks = [sparse_block(genotype[:, :3], ref_af[:3]),
              sparse_block(genotype[:, 3:], ref_af[3:], np.arange(3, 5))]
    captured = []

    reductions = []

    def score_covariance(g, **kwargs):
        captured.append(g.copy())
        reductions.append(kwargs)
        return torch.zeros(g.shape[1], dtype=torch.float64), torch.eye(g.shape[1], dtype=torch.float64)

    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options = AnalysisOptions(rare_maf_cutoff=0.05, genotype_block_size=3,
        wrapper_semantics=semantics, imputation=imputation)
    pipeline.models = [SimpleNamespace(n=100, n_pheno=1, use_spa=False, has_kinship=has_kinship,
        spectrum=SimpleNamespace(blocks=[]), score_covariance=score_covariance)]
    monkeypatch.setattr(pipeline_module, "GaussianNullModel", SimpleNamespace)
    pipeline.trait_rows = [np.arange(100)]
    pipeline.union_rows = np.arange(100)
    pipeline.annotation_names = ["marker"]
    pipeline.genotype = SimpleNamespace(iter_minor_blocks=lambda *args, **kwargs: iter(blocks))
    pipeline.annotations = lambda *args, **kwargs: SimpleNamespace(annotations={})
    pipeline._limit = lambda *args: None
    monkeypatch.setattr(pipeline_module, "annotation_phred_matrix",
        lambda *args, **kwargs: (np.arange(5, dtype=float)[:, None], ["marker"]))
    payload = pipeline._prepare_test_set(np.arange(5))[0]
    frequency = {"frequency_mode": "reference"} if semantics == "base" else {}
    baseline = [block.trait_dense(np.arange(100), imputation, **frequency) for block in blocks]
    expected_g = np.concatenate([item[0] for item in baseline], axis=1)[:, expected]
    expected_maf = np.concatenate([item[1] for item in baseline])[expected]
    np.testing.assert_array_equal(captured[0], expected_g)
    np.testing.assert_array_equal(payload["maf"], expected_maf)
    np.testing.assert_array_equal(payload["annotations"].ravel(), expected)
    if semantics == "base" and has_kinship:
        assert reductions[0]["reduction"] == "reference_sparse"
        assert 0 < reductions[0]["max_workspace_bytes"] <= 256 * 2**20
    else:
        assert reductions == [{}]


@pytest.mark.parametrize("cache_existing", [False, True])
def test_batch_identical_indices_prepare_once_and_preserve_aliases(monkeypatch, cache_existing):
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.models = [SimpleNamespace(n_pheno=1, use_spa=False)] * 2
    pipeline._test_set_cache = OrderedDict()
    pipeline.batch_diagnostics = []
    prepared_keys = []
    batch_keys = []

    def prepare(indices):
        key = tuple(indices)
        prepared_keys.append(key)
        if not key:
            return [None, None]
        return [dict(tag=(key, trait), score=torch.zeros(1, dtype=torch.float64),
            covariance=torch.ones((1, 1), dtype=torch.float64)) for trait in range(2)]

    def statistics(items, **kwargs):
        batch_keys.extend(item["tag"] for item in items)
        return [dict(tag=item["tag"]) for item in items], {"masks": len(items)}

    pipeline._prepare_test_set = prepare
    monkeypatch.setattr(batch_module, "association_test_batch", statistics)
    a, b, empty = np.asarray([1, 3]), np.asarray([2]), np.asarray([], dtype=np.int64)
    if cache_existing:
        pipeline._cache_set(pipeline._set_key(b), [dict(tag=((2,), 0)), dict(tag=((2,), 1))])
    output = pipeline._test_sets_batch([a, b, a.copy(), empty, b.copy(), empty.copy()], max_workspace_bytes=16)
    assert prepared_keys == ([(1, 3), ()] if cache_existing else [(1, 3), (2,), ()])
    assert len(batch_keys) == (2 if cache_existing else 4)
    assert output[0] == output[2] and output[1] == output[4]
    assert output[3] == output[5] == [None, None]
    for left, right in [(0, 2), (1, 4)]:
        assert output[left] is not output[right]
        assert all(x is not y for x, y in zip(output[left], output[right]))
    output[2][0]["changed"] = True
    assert "changed" not in output[0][0]
    assert "changed" not in pipeline._test_set_cache[pipeline._set_key(a)][0]
