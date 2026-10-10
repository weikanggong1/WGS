"""Numerical link contracts, not population or official-R benchmarks."""
import math

import numpy as np
import pytest
import torch

import fudan_wgs_toolkit.binary_null as binary_null


def _r_logit_scalar(eta):
    """Independent scalar formula from stats' logit link conventions."""
    epsilon = np.finfo(np.float64).eps
    exponential = epsilon if eta < -30 else 1/epsilon if eta > 30 else math.exp(eta)
    mu = exponential/(1+exponential)
    derivative = epsilon if abs(eta) > 30 else math.exp(eta)/(1+math.exp(eta))**2
    return mu, derivative, derivative**2/(mu*(1-mu))


def test_logit_means_derivative_and_working_weights_follow_r_boundary():
    eta = torch.tensor([-1000., -31., -30., -1., 0., 1., 30., 31., 1000.], dtype=torch.float64)
    actual = binary_null._binomial_logit_state(eta)
    expected = torch.tensor([_r_logit_scalar(float(z)) for z in eta], dtype=torch.float64).T
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=3e-16, atol=0)
    assert torch.all((actual[0] > 0) & (actual[0] < 1))
    assert torch.all(actual[2] > 0)
    # The derivative is independent at an extreme eta. This is not just
    # replacing zero sigmoid variance by an arbitrary positive floor.
    assert actual[1][-1] != actual[0][-1]*(1-actual[0][-1])


@pytest.mark.parametrize("eta", [torch.tensor([math.nan], dtype=torch.float64),
                                  torch.tensor([math.inf], dtype=torch.float64),
                                  torch.tensor([0.], dtype=torch.float32)])
def test_logit_nonfinite_or_wrong_precision_fails_closed(eta):
    with pytest.raises(ArithmeticError, match="finite FP64"):
        binary_null._binomial_logit_state(eta)


def test_rare_nonseparated_mixed_fit_does_not_misclassify_transient_saturation(monkeypatch):
    n = 1000
    y = np.zeros(n)
    y[[0, 499, 999]] = 1
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    parameters = dict(kinship_diagonal=np.ones(n), device="cpu")
    original = binary_null._binomial_logit_state

    def old_unbounded_link(eta):
        mu = torch.sigmoid(eta)
        weights = mu*(1-mu)
        if bool((weights <= 0).any()):
            raise ArithmeticError("old sigmoid saturated")
        return mu, weights, weights

    monkeypatch.setattr(binary_null, "_binomial_logit_state", old_unbounded_link)
    with pytest.raises(ArithmeticError, match="old sigmoid saturated"):
        binary_null.fit_logistic_mixed_null(y, x, **parameters)
    monkeypatch.setattr(binary_null, "_binomial_logit_state", original)
    trace = []
    fitted = binary_null.fit_logistic_mixed_null(y, x, trace_callback=trace.append, **parameters)
    assert fitted.converged and fitted.has_kinship
    assert fitted.fit_method == "logistic_block_sparse_GMMAT_PQL_AI_REML"
    assert fitted.theta[1] > 0
    assert max(float(s["eta"].max()) for s in trace) > 37
    assert max(s["link_boundary_count"] for s in trace) > 0
    assert fitted.binomial_link_boundary_count == 0
    assert fitted.binomial_link_method == "R_stats_logit_linkinv_mu_eta"
    assert torch.isfinite(fitted.scaled_residuals).all()
    assert torch.isfinite(fitted.precision).all()


def test_provable_fixed_effect_separation_is_not_hidden_by_link_boundary():
    x = np.column_stack((np.ones(32), np.linspace(-1, 1, 32)))
    y = (x[:, 1] > 0).astype(float)
    with pytest.raises(ArithmeticError, match="separation|converge"):
        binary_null.fit_logistic_mixed_null(y, x, kinship_diagonal=np.ones(32), device="cpu")
