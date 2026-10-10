"""Typed binary cache and fitted projection contracts, independent of SPA."""
import numpy as np
import pytest
import torch

from staar_phewas.binary_null import BinaryNullModel, binary_prefitted_state
from staar_phewas.io import load_null_model, save_null_model


def state(*, mode="fp64", device="cpu", use_spa=False, sparse=False):
    x = np.column_stack((np.ones(6), np.arange(6) - 2.5))
    precision = np.array([.11, .17, .09, .21, .14, .19])
    sx = precision[:, None] * x
    # This supplied covariance deliberately differs from a newly solved
    # projection, representing a fixed state returned at finite tolerance.
    cov = np.linalg.inv(x.T @ sx) * .93
    y = np.array([0., 1., 0., 1., 1., 0.])
    mu = np.array([.21, .42, .37, .61, .53, .32])
    residual = (y - mu) * .83
    inverse = precision
    if sparse:
        inverse = torch.diag(torch.tensor(precision)).to_sparse_coo()
    model = binary_prefitted_state(sample_ids=[f"s{i}" for i in range(6)],
        covariates=x, residual=residual, fitted_probability=mu,
        xw=sx.T, projection_left=x @ cov, fixed_effect_covariance=cov,
        precision=inverse, precision_covariates=sx, phenotype=y,
        working_phenotype=(y - mu) / (mu * (1 - mu)),
        has_kinship=True, use_spa=use_spa, device=device, matmul_mode=mode)
    return model


@pytest.mark.parametrize("sparse", [False, True])
def test_fixed_binary_projection_uses_supplied_covariance(sparse):
    model = state(sparse=sparse)
    g = torch.tensor([[0., 1.], [1., 0.], [0., 0.], [2., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    inverse = model.precision.to_dense() if sparse else torch.diag(model.precision)
    p = inverse.numpy() - model.precision_x.numpy() @ model.fixed_effect_covariance.numpy() @ model.precision_x.numpy().T
    expected_score = g.numpy().T @ model.scaled_residuals.numpy()
    expected_cov = g.numpy().T @ p @ g.numpy()
    score, cov = model.score_covariance(g)
    single_score, variance = model.individual_score_variance(g)
    np.testing.assert_allclose(score.numpy(), expected_score, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cov.numpy(), expected_cov, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(single_score, score, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(variance, cov.diagonal(), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("use_spa", [False, True], ids=["control", "spa"])
@pytest.mark.parametrize("precision_kind", ["none", "diagonal", "sparse"])
def test_binary_fp64_single_preserves_covariance_arithmetic_exactly(use_spa, precision_kind):
    model = state(mode="fp64", use_spa=use_spa, sparse=precision_kind == "sparse")
    if precision_kind == "none":
        weights = model.fitted_probability * (1 - model.fitted_probability)
        weighted_design = weights[:, None] * model.x
        supplied_covariance = torch.linalg.inv(model.x.T @ weighted_design) * .93
        model = binary_prefitted_state(sample_ids=model.sample_ids, covariates=model.x,
            residual=model.scaled_residuals, fitted_probability=model.fitted_probability,
            xw=weighted_design.T, projection_left=model.x @ supplied_covariance,
            fixed_effect_covariance=supplied_covariance, phenotype=model.phenotype,
            working_phenotype=model.working_phenotype, use_spa=use_spa,
            device="cpu", matmul_mode="fp64")
    # Fractional values exercise an imputed genotype as well as integral calls.
    g = torch.tensor([[0., 1., .17, 0.], [1., 0., 1., 0.], [0., 0., .17, 0.],
                      [2., 0., 0., 0.], [0., 1., 2., 0.], [1., 1., .17, 0.]], dtype=torch.float64)
    expected_score, covariance = model.score_covariance(g, matmul_mode="fp64")
    single_score, variance = model.individual_score_variance(g, matmul_mode="fp64")
    assert single_score.dtype == variance.dtype == torch.float64
    torch.testing.assert_close(single_score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, covariance.diagonal(), rtol=0, atol=0)


def test_binary_mode_conversion_preserves_fitted_state():
    model = state()
    original = model.precision_x.clone()
    model.set_matmul_mode("tf32")
    assert model.matmul_mode == "tf32"
    assert model.source_matmul_mode == "fp64"
    assert model.precision_x.dtype == torch.float32
    assert model.x.dtype == torch.float32
    torch.testing.assert_close(model.precision_x, original.float(), rtol=0, atol=0)
    model.set_matmul_mode("fp64")
    assert model.source_matmul_mode == "fp64"
    assert model.precision_x.dtype == torch.float64


@pytest.mark.parametrize("mode", ["fp64", "tf32"])
@pytest.mark.parametrize("sparse", [False, True])
def test_binary_typed_load_roundtrip(tmp_path, mode, sparse):
    model = state(sparse=sparse)
    model.gds_sample_ids = np.asarray([f"raw_{i}" for i in range(model.n)])
    model.set_matmul_mode(mode)
    path = tmp_path / "model.npz"
    save_null_model(model, path)
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored["model_kind"]) == "binary_state"
        assert str(stored["matmul_mode"]) == mode
        assert str(stored["source_matmul_mode"]) == "fp64"
    loaded = load_null_model(path)
    assert isinstance(loaded, BinaryNullModel)
    assert loaded.family == "binomial" and not loaded.use_spa
    assert loaded.matmul_mode == mode and loaded.source_matmul_mode == "fp64"
    assert loaded.has_kinship
    np.testing.assert_array_equal(loaded.gds_sample_ids, model.gds_sample_ids)
    for field in ("x", "scaled_residuals", "precision_x", "fixed_effect_covariance",
                  "phenotype", "working_phenotype", "precision"):
        original, restored = getattr(model, field), getattr(loaded, field)
        if original.layout != torch.strided:
            original, restored = original.to_dense(), restored.to_dense()
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        assert restored.dtype == (torch.float32 if mode == "tf32" else torch.float64)


def test_binary_explicit_tf32_load_retains_source_mode(tmp_path):
    path = tmp_path / "legacy.npz"
    save_null_model(state(), path)
    loaded = load_null_model(path, matmul_mode="tf32")
    assert loaded.x.dtype == torch.float32 and loaded.matmul_mode == "tf32"
    assert loaded.source_matmul_mode == "fp64"
    second = tmp_path / "native.npz"
    save_null_model(loaded, second)
    again = load_null_model(second)
    assert again.x.dtype == torch.float32 and again.source_matmul_mode == "fp64"


def test_binary_spa_keeps_fp64_and_rejects_tf32(tmp_path):
    model = state(use_spa=True)
    with pytest.raises(ValueError, match="SPA requires fp64"):
        model.set_matmul_mode("tf32")
    assert model.x.dtype == torch.float64 and model.matmul_mode == "fp64"
    path = tmp_path / "spa.npz"
    save_null_model(model, path)
    restored = load_null_model(path)
    assert restored.use_spa and restored.x.dtype == torch.float64
    with pytest.raises(ValueError, match="SPA requires fp64"):
        load_null_model(path, matmul_mode="tf32")
    with pytest.raises(ValueError, match="SPA requires fp64"):
        state(mode="tf32", use_spa=True)


@pytest.mark.parametrize("labels", [np.array([0., 1., -.2, 1., 0., 1.]),
                                   np.array([0., 1., 0., 1., 0., 1.0000000001])])
def test_binary_working_response_cannot_be_typed_as_labels(labels):
    model = state()
    with pytest.raises(ValueError, match="finite 0/1 labels"):
        binary_prefitted_state(sample_ids=model.sample_ids, covariates=model.x,
            residual=model.scaled_residuals, fitted_probability=model.fitted_probability,
            xw=model.xw, projection_left=model.projection_left,
            fixed_effect_covariance=model.fixed_effect_covariance,
            phenotype=labels, use_spa=False, device="cpu", matmul_mode="tf32")


def test_binary_tf32_core_calls_selected_products(monkeypatch):
    model = state(mode="tf32")
    calls = []
    def selected_product(a, b, *, mode):
        calls.append((mode, a.dtype, b.dtype))
        return a @ b
    monkeypatch.setattr("staar_phewas.binary_null.matmul", selected_product)
    g = torch.tensor([[0., 1.], [1., 0.], [0., 0.], [2., 0.], [0., 1.], [1., 1.]])
    score, covariance = model.score_covariance(g)
    single_score, variance = model.individual_score_variance(g)
    assert all(mode == "tf32" and a == b == torch.float32 for mode, a, b in calls)
    assert score.dtype == covariance.dtype == variance.dtype == torch.float32
    torch.testing.assert_close(single_score, score)
    torch.testing.assert_close(variance, covariance.diagonal())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA native TF32")
def test_binary_native_tf32_on_cuda():
    model = state(mode="tf32", device="cuda")
    g = torch.tensor([[0., 1.], [1., 0.], [0., 0.], [2., 0.], [0., 1.], [1., 1.]], device="cuda")
    score, covariance = model.score_covariance(g)
    single_score, variance = model.individual_score_variance(g)
    assert score.dtype == covariance.dtype == variance.dtype == torch.float32
    torch.testing.assert_close(single_score, score)
    torch.testing.assert_close(variance, covariance.diagonal(), rtol=1e-3, atol=1e-4)
