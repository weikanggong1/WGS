"""Hybrid dispatch/precision contracts; CPU mocks are not GPU benchmarks."""
from types import SimpleNamespace
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline, _cmac_scalar_sum
from fudan_wgs_toolkit.null_model import GaussianNullModel
from fudan_wgs_toolkit.precision_audit import (
    DenseProductAudit, explicit_fp64_spectral_refinement,
)
from fudan_wgs_toolkit.profiling import StageProfiler


def _dispatch_fixture(backend="cached"):
    analysis = PheWASPipeline.__new__(PheWASPipeline)
    analysis.options = AnalysisOptions(covariance_backend=backend)
    analysis.profiler = StageProfiler("cpu", enabled=False)
    analysis.local_mask_reuse_counters = {}
    analysis.covariance_diagnostics = []
    calls = []
    model = GaussianNullModel.__new__(GaussianNullModel)
    model.x = torch.ones((2, 1), dtype=torch.float32)
    model.spectrum = SimpleNamespace(blocks=[])
    model.matmul_mode = "tf32"
    analysis._limit = lambda unused_model, m, **kwargs: calls.append(("limit", m))
    def ordinary(g):
        calls.append(("ordinary", g))
        return "ordinary_U", "ordinary_V"
    def legacy(g, **kwargs):
        calls.append(("legacy", g, kwargs))
        return "legacy_U", "legacy_V"
    def cached(g, **kwargs):
        calls.append(("cached", g, kwargs))
        return "cached_U", "cached_V", {"variants": g.shape[1], "backend": "cached_native_tf32"}
    model.score_covariance = ordinary
    model.score_covariance_tiled = legacy
    model.score_covariance_cached = cached
    return analysis, model, calls


@pytest.mark.parametrize("m, expected", [(4999, "ordinary"), (5000, "ordinary"), (5001, "cached")])
def test_hybrid_dispatch_uses_actual_mask_size_and_keeps_threshold_inclusive(m, expected):
    analysis, model, calls = _dispatch_fixture()
    genotype = np.zeros((2, m), dtype=np.float32, order="F")
    assert analysis._hybrid_host_products(model, genotype) == (expected + "_U", expected + "_V")
    products = [row for row in calls if row[0] != "limit"]
    assert len(products) == 1
    assert products[0][0] == expected
    if expected == "ordinary":
        assert isinstance(products[0][1], torch.Tensor)
        assert analysis.covariance_diagnostics == []
    else:
        assert calls[0] == ("limit", m)
        assert products[0][1] is genotype
        assert products[0][2]["variant_tile_size"] == 4096
        assert products[0][2]["memory_limit_gib"] == 40
        assert analysis.covariance_diagnostics[0]["variants"] == m


def test_explicit_legacy_backend_preserves_512_covariance_route():
    analysis, model, calls = _dispatch_fixture("legacy")
    assert analysis._hybrid_host_products(model, np.zeros((2, 5001), dtype=np.float32)) == ("legacy_U", "legacy_V")
    assert calls[1][0] == "legacy"
    assert calls[1][2] == {"variant_tile_size": 512}
    assert analysis.covariance_diagnostics == []


def test_cached_numerical_failure_propagates_without_changing_backend():
    analysis, model, calls = _dispatch_fixture()
    def fail(*args, **kwargs):
        raise ArithmeticError("cache numerical guard")
    model.score_covariance_cached = fail
    with pytest.raises(ArithmeticError, match="cache numerical guard"):
        analysis._hybrid_host_products(model, np.zeros((2, 5001), dtype=np.float32))
    assert [row[0] for row in calls] == ["limit"]


def test_native_masks_never_enter_generic_fp64_statistics_batch(monkeypatch):
    import fudan_wgs_toolkit.batch_statistics as batch
    analysis, model, _ = _dispatch_fixture()
    analysis.options = AnalysisOptions(long_mask_threshold=2)
    analysis.models = [model]
    analysis._test_set_cache = OrderedDict()
    analysis.batch_diagnostics = []
    events = []
    def prepare(indices):
        m = len(indices)
        return [{"score": torch.ones(m), "covariance": torch.eye(m)}]
    def evaluate(payload, unused_model):
        m = len(payload["score"])
        route = "long" if m > 2 else "native_ordinary"
        events.append((route, m))
        return {"route": route, "M": m}
    def ordinary_batch(items, **kwargs):
        pytest.fail("forced TF32 entered generic FP64 statistics batch")
    analysis._prepare_test_set = prepare
    analysis._evaluate_prepared = evaluate
    monkeypatch.setattr(batch, "association_test_batch", ordinary_batch)
    masks = [np.array([10, 11]), np.array([0, 1, 2]), np.array([20, 21]), np.array([0, 1, 2])]
    result = analysis._test_sets_batch(masks, max_workspace_bytes=2**20)
    assert [row[0]["route"] for row in result] == ["native_ordinary", "long", "native_ordinary", "long"]
    assert events == [("native_ordinary", 2), ("long", 3), ("native_ordinary", 2)]


def test_explicit_fp64_controls_keep_generic_batch_and_duplicate_alias(monkeypatch):
    import fudan_wgs_toolkit.batch_statistics as batch
    analysis, model, _ = _dispatch_fixture()
    model.matmul_mode = "fp64"
    analysis.models = [model]
    analysis._test_set_cache = OrderedDict()
    analysis.batch_diagnostics = []
    events = []
    analysis._prepare_test_set = lambda indices: [
        {"score": torch.ones(len(indices), dtype=torch.float64),
         "covariance": torch.eye(len(indices), dtype=torch.float64)}]
    def fp64_batch(items, **kwargs):
        events.append([len(item["score"]) for item in items])
        return ([{"M": len(item["score"])} for item in items], {})
    analysis._evaluate_prepared = lambda *args: pytest.fail("FP64 mask lost its generic batch route")
    monkeypatch.setattr(batch, "association_test_batch", fp64_batch)
    masks = [np.array([10, 11]), np.array([0, 1, 2]), np.array([20, 21]), np.array([0, 1, 2])]
    result = analysis._test_sets_batch(masks, max_workspace_bytes=2**20)
    assert [row[0]["M"] for row in result] == [2, 3, 2, 3]
    assert events == [[2, 3, 2]]


@pytest.mark.parametrize("tile", [1, 513, 4095])
def test_bad_cache_tile_rejected_when_options_are_constructed(tile):
    with pytest.raises(ValueError, match="512"):
        AnalysisOptions(cached_variant_tile_size=tile)


def test_single_workspace_remains_linear_and_independent_of_cache_backend():
    model = SimpleNamespace(n=339013, n_pheno=1, matmul_mode="tf32",
                            x=SimpleNamespace(shape=(339013, 23)))
    estimates = []
    for backend in ("cached", "legacy"):
        analysis = PheWASPipeline.__new__(PheWASPipeline)
        analysis.options = AnalysisOptions(covariance_backend=backend)
        small = analysis._workspace_estimate(model, 5000, individual=True)
        large = analysis._workspace_estimate(model, 10000, individual=True)
        baseline = analysis._workspace_estimate(model, 0, individual=True)
        assert large - baseline == 2 * (small - baseline)
        estimates.append((small, large, baseline))
    assert estimates[0] == estimates[1]


def test_cached_workspace_estimate_counts_genotype_cache_instead_of_legacy512():
    # Shape-only Gaussian state; this test performs no model fit or GPU work.
    model = GaussianNullModel.__new__(GaussianNullModel)
    model.sample_ids = np.empty(339013, dtype="S1")
    model.x = SimpleNamespace(shape=(339013, 23), device=torch.device("cpu"))
    model.precision_x = SimpleNamespace(shape=(339013, 23))
    model.spectrum = SimpleNamespace(blocks=[])
    model.matmul_mode = "tf32"
    analysis = PheWASPipeline.__new__(PheWASPipeline)
    analysis.options = AnalysisOptions(covariance_backend="cached")
    estimate = analysis._workspace_estimate(model, 9854)
    assert 28 * 2**30 < estimate < 40 * 2**30
    analysis.options = AnalysisOptions(covariance_backend="legacy")
    assert analysis._workspace_estimate(model, 9854) < estimate


def test_large_fp64_binary_state_uses_public_dense_estimate_without_tf32_fields():
    analysis = PheWASPipeline.__new__(PheWASPipeline)
    analysis.options = AnalysisOptions()
    model = SimpleNamespace(n=100, n_pheno=1, device="cpu", family="binomial", use_spa=True)
    expected = 8 * (4 * 100 * 6001 + 6 * 6001**2 + 100)
    assert analysis._workspace_estimate(model, 6001) == expected


def test_fp32_cmac_metadata_uses_fp64_sum_without_changing_storage():
    dosage = np.full((100, 9), 0.1, dtype=np.float32)
    before = dosage.copy()
    assert _cmac_scalar_sum(dosage) == dosage.sum(dtype=np.float64)
    tensor = torch.from_numpy(dosage)
    assert _cmac_scalar_sum(tensor) == tensor.sum(dtype=torch.float64).item()
    np.testing.assert_array_equal(dosage, before)
    assert tensor.dtype == torch.float32


def test_declared_spectral_fp64_is_scoped_and_hidden_products_still_fail():
    value = torch.eye(3, dtype=torch.float64)
    with DenseProductAudit(forced=True) as audit:
        with explicit_fp64_spectral_refinement():
            assert torch.equal(value @ value, value)
        with pytest.raises(RuntimeError, match="hidden FP64"):
            value @ value
    assert audit.report()["explicit_fastskat_spectral_fp64_products"] == 1
    assert audit.report()["hidden_fp64_dense_products"] == 1
