"""Numerical unit checks; these fixtures are not a real-data benchmark."""

import math

import numpy as np
import pytest
import torch
from scipy.special import gammaincc
from scipy.stats import chi2, t

from fudan_wgs_toolkit.statistics import (
    DegenerateTestError,
    annotation_weights,
    cct,
    quadratic_form_sf,
    score_covariance,
    association_test,
    _student_t_two_sided,
)


@pytest.mark.parametrize("dof", [2, 29, 42651])
def test_student_t_torch_incomplete_beta_matches_independent_distribution(dof):
    # Covers the ordinary Gaussian t branch, including the real sample scale.
    squared = torch.tensor([0, 1e-8, .01, 1, 10, 100, 1000], dtype=torch.float64)
    expected = 2 * t.sf(np.sqrt(squared.numpy()), dof)
    actual = _student_t_two_sided(squared, dof)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-11, rtol=2e-10)
    if torch.cuda.is_available():
        gpu = _student_t_two_sided(squared.cuda(), dof)
        assert gpu.device.type == "cuda"
        torch.testing.assert_close(gpu.cpu(), actual, atol=2e-11, rtol=2e-10)


def test_score_covariance_matches_explicit_weighted_projector():
    genotype = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 2]], dtype=torch.float64)
    residual = torch.tensor([0.4, -0.1, 0.2, -0.5], dtype=torch.float64)
    covariates = torch.tensor([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=torch.float64)
    weights = torch.tensor([0.2, 0.3, 0.15, 0.25], dtype=torch.float64)
    diagonal = torch.diag(weights)
    fixed_covariance = torch.linalg.inv(covariates.T @ diagonal @ covariates)
    projector = diagonal - diagonal @ covariates @ fixed_covariance @ covariates.T @ diagonal
    score, covariance = score_covariance(genotype, residual, covariates=covariates,
                                         working_weights=weights)
    torch.testing.assert_close(score, genotype.T @ residual, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(covariance, genotype.T @ projector @ genotype, rtol=1e-13, atol=1e-13)
    sparse_score, sparse_covariance = score_covariance(
        genotype, residual, precision=diagonal.to_sparse_coo(),
        precision_covariates=diagonal @ covariates,
        fixed_effect_covariance=fixed_covariance,
    )
    torch.testing.assert_close(sparse_score, score)
    torch.testing.assert_close(sparse_covariance, covariance)


def test_gaussian_score_covariance_includes_dispersion_once():
    genotype = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.float64)
    residual = torch.tensor([0.3, -0.2, 0.1, -0.2], dtype=torch.float64)
    intercept = torch.ones((4, 1), dtype=torch.float64)
    score, covariance = score_covariance(genotype, residual, covariates=intercept, dispersion=2.5)
    centered = genotype - genotype.mean(0)
    torch.testing.assert_close(covariance, 2.5 * centered.T @ centered)
    torch.testing.assert_close(score, genotype.T @ residual)


def test_annotation_weight_formulas_and_column_order():
    maf = torch.tensor([0.001, 0.005], dtype=torch.float64)
    annotations = torch.tensor([[10, 20], [30, 40]], dtype=torch.float64)
    wb, ws, wa = annotation_weights(maf, annotations)
    rank = 1 - 10 ** (-annotations / 10)
    beta = 25 * (1 - maf) ** 24
    assert wb.shape == ws.shape == wa.shape == (2, 6)
    torch.testing.assert_close(wb[:, 0], beta)
    torch.testing.assert_close(wb[:, 1:3], rank * beta[:, None])
    torch.testing.assert_close(ws[:, 1:3].square(), rank * beta[:, None].square())
    torch.testing.assert_close(wb[:, 3], torch.ones_like(maf))
    torch.testing.assert_close(wa[:, 3], math.pi ** 2 * maf * (1 - maf))


def test_cct_exact_boundaries_and_extreme_tail():
    assert cct([0, 0.4]) == 0
    with pytest.raises(ValueError, match="both exact"):
        cct([0, 1], [1, 0])
    with pytest.warns(RuntimeWarning):
        assert cct([0.01, 1]) == 1
    assert cct([1e-200, 1e-200]) == pytest.approx(1e-200, rel=1e-13)
    # Independently evaluated from cot(pi*p), using 50-digit arithmetic.
    assert cct([0.01, 0.1, 0.3]) == pytest.approx(0.026742142154721940, rel=1e-13)
    assert cct([0.3, 0.3], [1, 5]) == pytest.approx(0.3, abs=1e-15)
    with pytest.raises(DegenerateTestError):
        cct([0.1, 0.2], [0, 0])
    # Internal ACAT-V keeps the C++ formula; public R CCT has a separate exact-1 rule.
    assert cct([0.1, 1], internal=True) > 0.999999999999999


@pytest.mark.parametrize("statistic", [0.001, 0.1, 1.0, 5.0, 20.0])
def test_saddlepoint_equal_spectrum_has_valid_monotone_tail(statistic):
    probability = quadratic_form_sf(statistic, [1.0, 1.0])
    assert 0 <= probability <= 1
    # WGS uses a saddlepoint approximation away from its moment switch.
    assert probability == pytest.approx(chi2.sf(statistic, 2), abs=0.003)


def test_saddlepoint_at_mean_uses_fourth_moment_fallback():
    eigenvalues = np.array([0.2, 1.0, 2.0])
    c2 = np.sum(eigenvalues ** 2)
    c4 = np.sum(eigenvalues ** 4)
    dof = c2 ** 2 / c4
    assert quadratic_form_sf(float(eigenvalues.sum()), eigenvalues) == pytest.approx(
        gammaincc(dof / 2, dof / 2), rel=1e-13
    )
    with pytest.raises(DegenerateTestError):
        quadratic_form_sf(1.0, [1e-12, 0.0])
    assert quadratic_form_sf(0.0, [1.0, 0.1]) == 1


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_saddle_negative_bound_preserves_original_scalar_division(device):
    # A small arithmetic fixture, unrelated to any private study data.
    # Original WGS 0.9.9 Bisection/Saddle gives the values below. With
    # scalar/Tensor reverse division, reciprocal*m changes this bound by
    # one ulp and the resulting near-mean tail by about 3.66e-6.
    eigenvalues = torch.tensor([1/3, 2/3, 1.], dtype=torch.float64, device=device)
    statistic = eigenvalues.new_tensor(1.9996)
    lower = -torch.full_like(statistic, eigenvalues.numel()) / (2 * statistic)
    assert float(lower) == -0.75015003000600122
    expected = 0.39548257230568734
    assert quadratic_form_sf(statistic, eigenvalues) == pytest.approx(expected, abs=1e-10, rel=1e-7)


def test_burden_and_acat_very_rare_collapse_match_manual_calculation():
    score = np.array([0.4, -0.8, 0.6])
    covariance = np.array([[1.0, 0.2, 0.1], [0.2, 1.2, 0.3], [0.1, 0.3, 0.9]])
    maf = np.array([0.001, 0.005, 0.009])
    mac = np.array([2, 11, 4])
    result = association_test(score, covariance, maf, mac)
    assert result["Burden(1,1)"] == pytest.approx(chi2.sf(score.sum() ** 2 / covariance.sum(), 1), rel=1e-13)
    rare = np.array([0, 2])
    collapsed = chi2.sf(score[rare].sum() ** 2 / covariance[np.ix_(rare, rare)].sum(), 1)
    common = chi2.sf(score[1] ** 2 / covariance[1, 1], 1)
    wa = math.pi ** 2 * maf * (1 - maf)
    expected = cct([common, collapsed], [wa[1], wa[rare].mean()], internal=True)
    assert result["ACAT-V(1,1)"] == pytest.approx(expected, rel=1e-13)
    assert result["num_variant"] == 3
    assert result["cMAC"] == 17
    assert len(result) == 16


def test_gaussian_acat_uses_student_t_only_for_common_mac():
    score = np.array([0.4, -0.8, 0.6])
    covariance = np.diag([1.0, 1.2, 0.9])
    maf = np.array([0.001, 0.005, 0.009])
    mac = np.array([11, 12, 13])
    result = association_test(score, covariance, maf, mac, acat_calibration="gaussian_glm", dof=30)
    q = score ** 2 / covariance.diagonal()
    common = 2 * t.sf(np.sqrt(q / (30 - q) * 29), 29)
    wa = math.pi ** 2 * maf * (1 - maf)
    assert result["ACAT-V(1,1)"] == pytest.approx(cct(common, wa, internal=True), rel=1e-13)
    assert result["Burden(1,1)"] == pytest.approx(chi2.sf(score.sum() ** 2 / covariance.sum(), 1), rel=1e-13)


def test_output_names_annotation_shape_filtering_and_rejections():
    result = association_test([0.4, 0.6, 0.8], np.eye(3), [0.001, 0.003, 0.1], [1, 3, 100],
                        [[10], [20], [30]], ["functional"], cmac=4.25)
    assert result["num_variant"] == 2
    assert result["cMAC"] == 4.25
    assert "SKAT(1,25)-functional" in result
    assert "WGS-O" in result
    with pytest.raises(ValueError, match="residual dof"):
        association_test([1, 2], np.eye(2), [0.001, 0.002], [20, 20], acat_calibration="gaussian_glm")
    with pytest.raises(ValueError, match="rare-variant count"):
        association_test([1], [[1]], [0.001], [2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_float64_for_score_and_final_columns():
    genotype = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 2]], dtype=torch.float64)
    residual = torch.tensor([0.4, -0.1, 0.2, -0.5], dtype=torch.float64)
    covariates = torch.ones((4, 1), dtype=torch.float64)
    cpu_u, cpu_v = score_covariance(genotype, residual, covariates=covariates)
    gpu_u, gpu_v = score_covariance(genotype.cuda(), residual.cuda(), covariates=covariates.cuda())
    torch.testing.assert_close(gpu_u.cpu(), cpu_u, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gpu_v.cpu(), cpu_v, rtol=1e-12, atol=1e-12)
    cpu = association_test(cpu_u, cpu_v, [0.001, 0.005], [2, 10])
    gpu = association_test(gpu_u, gpu_v, [0.001, 0.005], [2, 10])
    for name in cpu:
        assert gpu[name] == pytest.approx(cpu[name], rel=1e-11, abs=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", [3, 32, 33])
def test_cuda_small_spectra_use_real_weight_batch_and_preserve_large_route(monkeypatch, size):
    # The CUDA solver changes between single and small batched matrices. Check
    # real annotation matrices, including the dispatch boundary, without study
    # data or artificial duplicate matrices used to trigger a solver.
    device = "cuda"
    diagonal = torch.linspace(.3, 2., size, dtype=torch.float64, device=device)
    score = torch.sqrt(diagonal * .995)
    covariance = torch.diag(diagonal)
    maf = torch.linspace(.001, .009, size, dtype=torch.float64, device=device)
    annotations = torch.linspace(10., 30., size, dtype=torch.float64, device=device)[:, None]
    weights = annotation_weights(maf, annotations)[1]
    matrices = torch.stack([covariance * weight[:, None] * weight[None, :]
                            for weight in weights.T])
    original = torch.linalg.eigvalsh
    spectra = original(matrices, UPLO="U") if size <= 32 else torch.stack([
        original(matrix, UPLO="U") for matrix in matrices])
    expected = [quadratic_form_sf(torch.sum(score.square() * weight.square()), eigen)
                for weight, eigen in zip(weights.T, spectra)]
    calls = []

    def record(matrix, **kwargs):
        calls.append(matrix.detach().clone())
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigvalsh", record)
    result = association_test(score, covariance, maf, torch.full_like(maf, 20),
                        annotations, ["functional"])
    if size <= 32:
        assert [tuple(matrix.shape) for matrix in calls] == [(4, size, size)]
        torch.testing.assert_close(calls[0], matrices, atol=0., rtol=0.)
    else:
        assert [tuple(matrix.shape) for matrix in calls] == [(size, size)] * 4
    fields = ["SKAT(1,25)", "SKAT(1,25)-functional",
              "SKAT(1,1)", "SKAT(1,1)-functional"]
    for field, probability in zip(fields, expected):
        assert result[field] == pytest.approx(probability, abs=1e-10, rel=1e-7)
