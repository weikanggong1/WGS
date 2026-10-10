"""Keep normal TF32 contractions strict while admitting fitted FP64 / SPA work."""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import binary_single_phewas, staar_binary_phewas
from fudan_wgs_toolkit.binary_null import (
    ImmutableFittedBinarySource, compact_binary_normal_from_fitted_arrays,
    compact_spa_state_from_fitted_arrays,
)
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.phewas_storage import ModelRepository, save_model_store
from fudan_wgs_toolkit.precision_audit import (
    DenseProductAudit, explicit_binary_fitted_projection_fp64,
    explicit_binary_spa_fp64,
)


def _fitted(device="cpu"):
    n = 48
    x = np.column_stack((np.ones(n), np.linspace(-1., 1., n)))
    y = (np.arange(n) % 3 == 0).astype(float)
    return prepare_phewas_model(y, np.arange(n).astype(str), x,
        family="binomial", device=device, association_mode="tf32")


def _source_arguments(model, sparse=False):
    precision = model.precision
    if sparse:
        precision = torch.diag(precision).to_sparse_coo().coalesce()
    return dict(sample_ids=model.sample_ids, covariates=model.x,
        residual=model.scaled_residuals, fitted_probability=model.fitted_probability,
        fixed_effect_covariance=model.fixed_effect_covariance, precision=precision,
        null_fit_source_sha256=model.null_fit_source_sha256,
        verified_fit_source_sha256=model.null_fit_source_sha256,
        device=model.device, coefficients=model.coefficients)


def _cpu_normal_oracle(monkeypatch):
    # CPU proves dtype/scope/formula contracts; CUDA parameters below retain
    # the real native TF32 implementation and PTX checks unchanged.
    calls = []
    def product(left, right, *, mode):
        assert mode == "tf32"
        assert left.dtype == right.dtype == torch.float32
        calls.append((left.shape, right.shape))
        return left @ right
    monkeypatch.setattr("fudan_wgs_toolkit.binary_null.matmul", product)
    monkeypatch.setattr("fudan_wgs_toolkit.tf32.matmul", product)
    return calls


@pytest.mark.parametrize("scope,counter", [
    (explicit_binary_fitted_projection_fp64, "explicit_binary_fitted_projection_fp64_products"),
    (explicit_binary_spa_fp64, "explicit_binary_spa_fp64_products"),
])
def test_binary_fp64_scope_counts_and_resets_after_exception(scope, counter):
    value = torch.eye(2, dtype=torch.float64)
    with DenseProductAudit(forced=True) as audit:
        with pytest.raises(ValueError, match="fixture"):
            with scope():
                assert torch.equal(value @ value, value)
                raise ValueError("fixture")
        with pytest.raises(RuntimeError, match="hidden FP64"):
            value @ value
    report = audit.report()
    assert report[counter] == 1
    assert report["hidden_fp64_dense_products"] == 1
    assert report["explicit_fastskat_spectral_fp64_products"] == 0


def test_fitted_reconstruction_is_not_mislabelled_as_selected_spa():
    value = torch.eye(2, dtype=torch.float64)
    with DenseProductAudit(forced=True) as audit:
        with explicit_binary_spa_fp64():
            value @ value
            with explicit_binary_fitted_projection_fp64():
                value @ value
            value @ value
    report = audit.report()
    assert report["explicit_binary_fitted_projection_fp64_products"] == 1
    assert report["explicit_binary_spa_fp64_products"] == 2
    assert report["hidden_fp64_dense_products"] == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_compact_and_immutable_fitted_projection_admission_is_scoped(sparse, monkeypatch):
    fitted = _fitted()
    arguments = _source_arguments(fitted.spa_model, sparse=sparse)
    calls = _cpu_normal_oracle(monkeypatch)
    genotype = torch.arange(48*4).reshape(48, 4).remainder(3).float()
    with DenseProductAudit(forced=True) as audit:
        compact_binary_normal_from_fitted_arrays(**arguments)
        compact_spa_state_from_fitted_arrays(**arguments)
        source = ImmutableFittedBinarySource(**arguments)
        normal, spa = source.normal(), source.spa()
        normal.score_covariance(genotype)
        normal.individual_score_variance(genotype)
    report = audit.report()
    assert report["explicit_binary_fitted_projection_fp64_products"] >= 3
    assert report["explicit_binary_spa_fp64_products"] == 0
    assert report["hidden_fp64_dense_products"] == 0
    assert calls and spa.x.dtype == torch.float64 and normal.x.dtype == torch.float32
    # The fitted-source scope cannot authorize an accidental FP64 normal GEMM.
    monkeypatch.setattr("fudan_wgs_toolkit.binary_null.matmul",
        lambda left, right, *, mode: left.double() @ right.double())
    with DenseProductAudit(forced=True):
        with pytest.raises(RuntimeError, match="hidden FP64"):
            normal.score_covariance(genotype)


@pytest.mark.parametrize("kind", ["individual", "burden"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repository_normal_and_selected_spa_use_separate_audit_scopes(
        kind, device, tmp_path, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires real native TF32 CUDA products")
    fitted = _fitted(device)
    if device == "cpu":
        calls = _cpu_normal_oracle(monkeypatch)
    entry = save_model_store(fitted.model, tmp_path/"model", sample_rows=np.arange(48),
                            fit_metadata={}, spa_model=fitted.spa_model)
    repository = ModelRepository([entry])
    genotype = torch.zeros((48, 6), dtype=torch.float32, device=device)
    for j in range(6):
        genotype[(j*5+np.arange(7)) % 48, j] = 1
    maf = torch.linspace(.001, .009, 6, dtype=torch.float64, device=device)
    def evaluate(normal, sidecar=None, factory=None):
        if kind == "individual":
            return binary_single_phewas(normal, genotype, spa_model=sidecar,
                spa_acquire=factory, p_filter_cutoff=1)["pvalues"]
        return staar_binary_phewas(normal, genotype, maf, spa_model=sidecar,
            spa_acquire=factory, p_filter_cutoff=1)
    expected = evaluate(fitted.model, sidecar=fitted.spa_model)
    with DenseProductAudit(forced=True) as audit:
        with repository.acquire([0], device) as models:
            actual = evaluate(models[0], factory=lambda: repository.acquire_spa(0, device))
    if kind == "individual":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected
    report = audit.report()
    assert report["explicit_binary_fitted_projection_fp64_products"] >= 2
    assert report["explicit_binary_spa_fp64_products"] > 0
    assert report["hidden_fp64_dense_products"] == 0
    assert report["explicit_fastskat_spectral_fp64_products"] == 0
    assert repository.spa_pinned[(0, device)] == 0
    if device == "cpu":
        assert calls


def test_unselected_binary_single_does_not_enter_spa_or_leak_scope(monkeypatch):
    fitted = _fitted()
    _cpu_normal_oracle(monkeypatch)
    entered = []
    @contextmanager
    def sidecar():
        entered.append(True)
        yield fitted.spa_model
    with DenseProductAudit(forced=True) as audit:
        result = binary_single_phewas(fitted.model, torch.ones((48, 3)),
            spa_acquire=sidecar, normal_score=torch.zeros(3),
            normal_variance=torch.ones(3), normal_pvalues=torch.full((3,), .8))
        assert not bool(result["spa_selected"].any())
        before = audit.report()
        with pytest.raises(RuntimeError, match="hidden FP64"):
            torch.ones((2, 2), dtype=torch.float64) @ torch.ones((2, 2), dtype=torch.float64)
    assert not entered
    assert before["explicit_binary_spa_fp64_products"] == 0
    assert before["explicit_binary_fitted_projection_fp64_products"] == 0
    assert before["hidden_fp64_dense_products"] == 0
