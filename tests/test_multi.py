"""Independent joint-model algebra checks; not real-data benchmarks."""
import math
import pytest
import torch
from scipy.stats import chi2

from fudan_wgs_toolkit.multi import (JointGaussianNullModel, fit_joint_gaussian_null,
                               joint_chi_square, joint_rank_inverse_normal,
                               joint_association_test, _ai_score_information, _components,
                               _gls_state)


def _inputs():
    y = torch.tensor([[.1, .4], [.2, -.3], [-.5, .9], [.7, .1], [.4, -.5],
                      [-.7, -.2], [.3, 1.2], [-.2, -.8]], dtype=torch.float64)
    x = torch.stack((torch.ones(8), torch.arange(8)), dim=1).double()
    g = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 0],
                      [0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float64)
    return y, x, g


def test_joint_ordinary_matches_full_kronecker_projector():
    y, x, g = _inputs()
    model = fit_joint_gaussian_null(y, x, device="cpu")
    score, covariance = model.score_covariance(g)
    t, n = y.shape[1], y.shape[0]
    sigma_i = torch.kron(torch.linalg.inv(model.theta[0]).contiguous(), torch.eye(n).double())
    x2 = torch.kron(torch.eye(t).double(), x)
    g2 = torch.kron(torch.eye(t).double(), g)
    projector = sigma_i - sigma_i @ x2 @ torch.linalg.inv(x2.T @ sigma_i @ x2) @ x2.T @ sigma_i
    torch.testing.assert_close(score, g2.T @ projector @ y.T.contiguous().reshape(-1), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(covariance, g2.T @ projector @ g2, atol=1e-13, rtol=1e-13)
    assert model.n_pheno == 2 and not model.relatedness


def test_joint_single_variant_uses_correlation_and_two_degrees_of_freedom():
    score = torch.tensor([.4, -.7], dtype=torch.float64)
    covariance = torch.tensor([[2., .6], [.6, 1.]], dtype=torch.float64)
    statistic = float(score @ torch.linalg.solve(covariance, score))
    assert float(joint_chi_square(score, covariance)) == pytest.approx(chi2.sf(statistic, 2), rel=1e-14)
    independent = joint_chi_square(score, covariance.diag().diag())
    assert abs(float(independent - joint_chi_square(score, covariance))) > .05


def test_joint_burden_matches_explicit_two_trait_score():
    y, x, g = _inputs()
    model = fit_joint_gaussian_null(y, x, device="cpu")
    u, v = model.score_covariance(g)
    maf = torch.tensor([.001, .002, .003], dtype=torch.float64)
    result = joint_association_test(u, v, maf, [3, 3, 4], cmac=float(g.sum()))
    total_score = u.reshape(2, 3).sum(1)
    total_covariance = v.reshape(2, 3, 2, 3).sum((1, 3))
    assert result["Burden(1,1)"] == pytest.approx(float(joint_chi_square(total_score, total_covariance)), rel=1e-13)
    # All three MACs are <=10: ACAT-V consists of the joint burden only.
    assert result["ACAT-V(1,1)"] == pytest.approx(result["Burden(1,1)"], rel=1e-13)
    assert result["cMAC"] == float(g.sum())


def test_rank_inverse_normal_averages_ties_after_complete_case_selection():
    y = torch.tensor([[1., 5.], [1., 2.], [3., 2.], [7., 8.]], dtype=torch.float64)
    z = joint_rank_inverse_normal(y)
    ranks = torch.tensor([[1.5, 3.], [1.5, 1.5], [3., 1.5], [4., 4.]], dtype=torch.float64)
    torch.testing.assert_close(z, torch.special.ndtri((ranks - .375) / 4.25))


def test_joint_model_roundtrip(tmp_path):
    y, x, g = _inputs()
    model = fit_joint_gaussian_null(y, x, sample_ids=[f"sample_{j}" for j in range(8)], device="cpu")
    path = tmp_path / "joint.pt"
    model.save(path)
    restored = JointGaussianNullModel.load(path, device="cpu")
    assert restored.sample_ids == model.sample_ids
    for actual, expected in zip(restored.score_covariance(g), model.score_covariance(g)):
        torch.testing.assert_close(actual, expected)


def test_joint_diagonal_rejects_nonzero_edges_and_unidentifiable_grm():
    y, x, _ = _inputs()
    with pytest.raises(NotImplementedError, match="no nonzero edges"):
        fit_joint_gaussian_null(y, x, kinship_diagonal=torch.arange(8) + 1,
                                edge_values=[.1], device="cpu")
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_joint_gaussian_null(y, x, kinship_diagonal=torch.ones(8), device="cpu")


def test_diagonal_joint_ai_matches_explicit_reml_derivatives():
    y, x, _ = _inputs()
    n, t = y.shape
    basis, pairs = _components(t, device=y.device)
    k = torch.linspace(.3, .9, n, dtype=y.dtype)
    derivative = torch.stack([(torch.ones_like(k) if c == 0 else k)[:, None, None] * basis[j]
                              for j, (c, _, _) in enumerate(pairs)], dim=1)
    theta = y.new_tensor([.6, .1, .8, .3, .05, .4])
    sigma = torch.einsum("j,njab->nab", theta, derivative)
    w, cov, _, py = _gls_state(y, x, sigma)
    score, information = _ai_score_information(y, x, w, cov, py, derivative)
    sigma_i = torch.zeros((n * t, n * t), dtype=y.dtype)
    for a in range(t):
        for b in range(t):
            sigma_i[a*n:(a+1)*n, b*n:(b+1)*n] = torch.diag(w[:, a, b])
    x2 = torch.kron(torch.eye(t).double(), x)
    projector = sigma_i - sigma_i @ x2 @ cov @ x2.T @ sigma_i
    py2 = projector @ y.T.contiguous().reshape(-1)
    torch.testing.assert_close(py.T.contiguous().reshape(-1), py2)
    derivatives = [torch.kron(basis[j], torch.diag(torch.ones_like(k) if c == 0 else k))
                   for j, (c, _, _) in enumerate(pairs)]
    expected_score = torch.stack([py2 @ d @ py2 - torch.trace(projector @ d) for d in derivatives])
    expected_information = torch.stack([torch.stack([py2 @ a @ projector @ b @ py2
                                                     for b in derivatives]) for a in derivatives])
    torch.testing.assert_close(score, expected_score, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(information, expected_information, atol=1e-12, rtol=1e-12)


def test_joint_individual_determinant_zero_and_log_underflow():
    from fudan_wgs_toolkit.multi import joint_individual_logp
    from scipy.special import erfcx
    score=torch.tensor([1.,2.],dtype=torch.float64)
    assert float(joint_individual_logp(score,torch.zeros((2,2),dtype=torch.float64)))==0
    score=torch.tensor([100.,100.],dtype=torch.float64)
    assert float(joint_individual_logp(score,torch.eye(2,dtype=torch.float64)))==10000.
    score=torch.tensor([100.,100.,100.],dtype=torch.float64)
    actual=float(joint_individual_logp(score,torch.eye(3,dtype=torch.float64)))
    # Independent closed form for chi-square(df=3), using scaled erfc.
    x=15000.
    expected=x-math.log(erfcx(math.sqrt(x))+2*math.sqrt(x)/math.sqrt(math.pi))
    assert abs(actual-expected)<2e-11
