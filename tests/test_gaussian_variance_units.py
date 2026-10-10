"""AI coordinate changes and sigma-zero limits, not population benchmarks."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.null_model import fit_gaussian_null


def _interior_data():
    n = 100
    x = np.column_stack((np.ones(n), np.linspace(-1., 1., n)))
    y = np.sin(np.arange(n) * 1.7) + .3 * x[:, 1]
    kinship = dict(kinship_diagonal=np.linspace(.8, 1.2, n),
                   edge_rows=[0, 2], edge_cols=[1, 3], edge_values=[.02, .03])
    return x, y, kinship


def test_small_nonconstant_phenotype_uses_relative_variance_units():
    x, y, kinship = _interior_data()
    with pytest.raises(ValueError, match="covariance is singular"):
        fit_gaussian_null(y * .001, covariates=x, **kinship)
    model = fit_gaussian_null(y * .001, covariates=x,
                              variance_normalization="unit", **kinship)
    assert model.converged and float(model.theta[0]) > 0
    assert torch.isfinite(model.scaled_residuals).all()
    assert model.variance_normalization == "unit"
    assert model.phenotype_scale == pytest.approx(np.std(y * .001, ddof=1))


@pytest.mark.parametrize("scale", [1e-4, .001, 1000.])
def test_normalized_mixed_fit_restores_every_physical_fitted_field(scale):
    x, y, kinship = _interior_data()
    reference = fit_gaussian_null(y, covariates=x,
                                  variance_normalization="unit", **kinship)
    actual = fit_gaussian_null(y * scale, covariates=x,
                               variance_normalization="unit", **kinship)
    torch.testing.assert_close(actual.x, reference.x, atol=0, rtol=0)
    for field in ("theta", "precision_theta", "fixed_effect_covariance"):
        torch.testing.assert_close(getattr(actual, field) / scale**2,
                                   getattr(reference, field), atol=2e-11, rtol=2e-10)
    for field in ("coefficients", "phenotype", "fitted_values", "working_phenotype"):
        torch.testing.assert_close(getattr(actual, field) / scale,
                                   getattr(reference, field), atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(actual.scaled_residuals * scale,
                               reference.scaled_residuals, atol=2e-10, rtol=2e-10)
    for field in ("inverse_variance", "precision_x"):
        torch.testing.assert_close(getattr(actual, field) * scale**2,
                                   getattr(reference, field), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual.spectrum.eigenvalues,
                               reference.spectrum.eigenvalues, atol=0, rtol=0)
    for (_, a), (_, b) in zip(actual.spectrum.blocks, reference.spectrum.blocks):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def _boundary_data():
    n = 96
    diagonal = np.linspace(.03, 3., n)
    x = np.column_stack((np.ones(n), np.linspace(-1., 1., n)))
    y = np.random.default_rng(0).normal(size=n) * np.sqrt(diagonal)
    return x, y, diagonal


@pytest.mark.parametrize("normalization", ["none", "unit"])
@pytest.mark.parametrize("related", [False, True])
def test_zero_sigma_positive_definite_grm_has_finite_dense_py_limit(normalization, related):
    x, y, diagonal = _boundary_data()
    rows = np.arange(0, 14, 2) if related else np.array([], dtype=np.int64)
    edges = dict(edge_rows=rows, edge_cols=rows + 1, edge_values=np.full(len(rows), .005))
    model = fit_gaussian_null(y, covariates=x, kinship_diagonal=diagonal,
                              variance_normalization=normalization, **edges)
    assert model.converged and model.has_kinship
    assert float(model.theta[0]) == 0 and float(model.theta[1]) > 0
    assert float(model.precision_theta[0]) == 0
    xd = torch.tensor(x, dtype=torch.float64)
    yd = torch.tensor(y, dtype=torch.float64)
    dense_kinship = torch.diag(torch.tensor(diagonal))
    dense_kinship[rows, rows + 1] = .005
    dense_kinship[rows + 1, rows] = .005
    variance = (model.precision_theta[0] * torch.eye(len(y), dtype=torch.float64)
                + model.precision_theta[1] * dense_kinship)
    precision = torch.linalg.inv(variance)
    covariance = torch.linalg.inv(xd.T @ precision @ xd)
    projection = precision - precision @ xd @ covariance @ xd.T @ precision
    expected_py = projection @ yd
    torch.testing.assert_close(model.scaled_residuals, expected_py, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(model.fixed_effect_covariance, covariance, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(model.coefficients, covariance @ xd.T @ precision @ yd,
                               atol=2e-12, rtol=2e-12)
    genotype = torch.tensor(np.random.default_rng(6).integers(0, 3, size=(len(y), 4)),
                            dtype=torch.float64)
    score, score_covariance = model.score_covariance(genotype)
    torch.testing.assert_close(score, genotype.T @ expected_py, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(score_covariance, genotype.T @ projection @ genotype,
                               atol=2e-11, rtol=2e-11)


def test_normalized_trace_declares_units_and_returned_state_is_original():
    x, y, kinship = _interior_data()
    traces = []
    model = fit_gaussian_null(y * .001, covariates=x,
                              variance_normalization="unit", trace_callback=traces.append,
                              **kinship)
    assert traces and all(t["trace_units"] == "unit_variance" for t in traces)
    assert all(t["variance_normalization"] == "unit" for t in traces)
    assert all(t["phenotype_scale"] == model.phenotype_scale for t in traces)
    torch.testing.assert_close(model.precision_theta,
                               traces[-1]["tau_old"] * model.phenotype_scale**2,
                               atol=0, rtol=0)


def test_normalized_no_kinship_returns_same_ols_state_in_physical_units():
    x, y, _ = _interior_data()
    normal = fit_gaussian_null(y * .001, covariates=x)
    normalized = fit_gaussian_null(y * .001, covariates=x, variance_normalization="unit")
    for field in ("theta", "precision_theta", "coefficients", "scaled_residuals",
                  "fixed_effect_covariance", "inverse_variance", "precision_x"):
        torch.testing.assert_close(getattr(normalized, field), getattr(normal, field),
                                   atol=2e-10, rtol=2e-12)
    assert not normalized.has_kinship


def test_unit_normalization_does_not_hide_constant_data_or_invalid_covariance():
    with pytest.raises(ValueError, match="zero or nonfinite variance"):
        fit_gaussian_null(np.ones(32), variance_normalization="unit")
    with pytest.raises(ValueError, match="zero mean diagonal"):
        fit_gaussian_null(np.sin(np.arange(32)), kinship_diagonal=np.zeros(32),
                          variance_normalization="unit")
    with pytest.raises(ValueError, match="variance_normalization"):
        fit_gaussian_null(np.sin(np.arange(32)), variance_normalization="clamp")


def test_normalized_maxiter_failure_still_raises_no_fallback():
    x, y, kinship = _interior_data()
    with pytest.raises(ArithmeticError, match="did not converge"):
        fit_gaussian_null(y, covariates=x, variance_normalization="unit", maxiter=2,
                          **kinship)
