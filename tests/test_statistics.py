"""Independent analytic/scalar references for the GPU association inference."""

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, erf
from scipy.stats import chi2, multivariate_normal
import torch

from torchwgs.statistics import (
    DEFAULT_RHOS, acat_logp, chi2_isf_logp, chi2_logsf, chi_bar_weights,
    eigen_compatible_column_norm,
    davies_logsf, kuonen_logsf, nnls_coefficients, normal_orthant_probability, numerical_diagnostics,
    sbat_logp, skat_logp, skato_logp, weighted_chi2_logsf,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device=DEVICE)


def acat_reference(pvalues, weights=None):
    pvalues = np.minimum(np.asarray(pvalues), .999)
    weights = np.ones_like(pvalues) if weights is None else np.asarray(weights)
    statistic = np.sum(weights * np.tan(np.pi * (.5 - pvalues))) / weights.sum()
    return -math.log10(math.atan2(1., statistic) / math.pi)


class TailTests(unittest.TestCase):
    def test_gpu_ordered_norm_matches_frozen_eigen_sse2_bits(self):
        # Public, participant-free numeric fixture evaluated independently
        # by Eigen3.4 compiled g++ -O3/SSE2, without fast-math.  The test does
        # not implement the four-chain sum or require Eigen at runtime.
        x = tensor([.001234567890123, 1.1234567890123, .0098765432109876]).repeat(10003, 1)
        indices = torch.arange(10003, device=DEVICE)
        x[indices % 11 == 0] = 0
        x[indices % 101 == 0] = tensor([1., 2., 3.])
        expected_squared = np.array([4636738256954075706, 4667692492504630837,
                                     4651135424334627105], dtype=np.uint64)
        expected_norm = np.array([4621819503815498247, 4637332484827610672,
                                  4629141585869923296], dtype=np.uint64)
        for values in (x, x.T.contiguous().T, x.cpu()):
            observed = eigen_compatible_column_norm(values, squared=True)
            np.testing.assert_array_equal(observed.cpu().numpy().view(np.uint64), expected_squared)
            observed = eigen_compatible_column_norm(values)
            np.testing.assert_array_equal(observed.cpu().numpy().view(np.uint64), expected_norm)
            self.assertEqual(observed.device, values.device)
        vector = eigen_compatible_column_norm(x[:, 0])
        self.assertEqual(vector.ndim, 0)
        self.assertEqual(vector.cpu().numpy().view(np.uint64).item(), expected_norm[0])

    def test_chi_square_matches_scipy_and_keeps_extreme_tail(self):
        x = tensor([0., .1, 1., 10., 100., 1000., 2000.])
        expected = -(math.log(2.) + log_ndtr(-np.sqrt(x.cpu().numpy()))) / math.log(10.)
        np.testing.assert_allclose(chi2_logsf(x).cpu(), expected, rtol=2e-13, atol=2e-13)
        self.assertGreater(float(chi2_logsf(x[-1])), 308.)
        # Even df=4 has a closed form exp(-x/2)*(1+x/2).
        expected4 = (x / 2 - torch.log1p(x / 2)) / math.log(10.)
        torch.testing.assert_close(chi2_logsf(x, 4), expected4, rtol=2e-13, atol=2e-13)
        self.assertEqual(chi2_logsf(x).device, DEVICE)

    def test_log_inverse_never_forms_underflowed_p(self):
        logp = tensor([0., .01, 1., 20., 300., 1000.])
        for df in (1., 2., 5.5):
            reconstructed = chi2_logsf(chi2_isf_logp(logp, df), df)
            torch.testing.assert_close(reconstructed, logp, rtol=2e-11, atol=3e-12)

    def test_acat_extreme_and_ordinary_signed_combination(self):
        lp = tensor([10., 1000.])
        self.assertAlmostEqual(float(acat_logp(lp)), 1000. - math.log10(2.), places=10)
        for p in ([.9, .01, .4], [.001, .002], [.5, .5], [1., .01]):
            observed = float(acat_logp(tensor(-np.log10(p))))
            self.assertAlmostEqual(observed, acat_reference(p), places=10)
        observed = acat_logp(tensor([float("nan"), -1., 2.]), tensor([1., 1., 3.]))
        self.assertAlmostEqual(float(observed), 2., places=12)
        self.assertEqual(float(acat_logp(tensor([0.]))), 0.)
        self.assertEqual(float(acat_logp(tensor([0., -1.]))), 0.)
        self.assertTrue(bool(torch.isnan(acat_logp(tensor([-1., float("nan")])))))

    def test_davies_gpu_fourier_against_frozen_scalar_reference(self):
        # Numeric fixtures evaluated with official 3.4.1 AS155, nc=0/df=1,
        # accuracy=1e-6/limit=10000.  The reference is not a runtime dependency.
        cases = [([.2, .5, 1.], 1.7, .4376331345005964),
                 ([.2, .5, 1.], 5.1, 1.363677216585403),
                 ([.2, .2, .8, .8, 1., 1.], 4., .3892731621546953),
                 ([.2, .2, .8, .8, 1., 1.], 12., 1.9016187887339528)]
        for spectrum, q, expected in cases:
            observed, fault = davies_logsf(tensor(q), tensor(spectrum))
            self.assertEqual(int(fault), 0)
            self.assertAlmostEqual(float(observed), expected, delta=2e-12)
            self.assertEqual(observed.device, DEVICE)

    def test_davies_budget_failure_switches_moderate_p_to_kuonen(self):
        spectrum = tensor([1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .1, 1.])
        q = tensor(.0441776)
        numerical_diagnostics(reset=True)
        lp, fault = davies_logsf(q, spectrum)
        self.assertEqual(int(fault), 1)  # native consumes its auxiliary budget
        self.assertTrue(bool(torch.isnan(lp)))
        score = torch.zeros_like(spectrum)
        score[0] = q.sqrt()
        observed = skat_logp(score, torch.diag(spectrum))
        expected = kuonen_logsf(q, spectrum)
        self.assertAlmostEqual(float(observed), float(expected), places=12)
        self.assertLess(float(observed), 1.)  # failure is unrelated to tiny P
        self.assertEqual(numerical_diagnostics()["cpu_tail_fallback_calls"], 0)
        self.assertEqual(numerical_diagnostics()["kuonen_tail_values"], 1)

    def test_weighted_rank_two_against_positive_angular_integral(self):
        values = tensor([.2, 2.])
        qs = tensor([.1, 1., 10., 100., 1000.])
        observed = weighted_chi2_logsf(qs, values)
        reference = []
        for q in qs.cpu().tolist():
            shift = -q / 4
            integral, error = quad(lambda theta: math.exp(
                -q / (2 * (.2 * math.cos(theta)**2 + 2 * math.sin(theta)**2)) - shift),
                0, math.pi/2, epsabs=1e-12, epsrel=1e-12)
            self.assertLess(error, 1e-9)
            reference.append(-(shift + math.log(2 * integral / math.pi)) / math.log(10.))
        np.testing.assert_allclose(observed.cpu(), reference, rtol=1e-8, atol=2e-8)

    def test_weighted_rank_three_against_closed_form_convolution(self):
        # chi2_1 + 2*chi2_2 has an elementary survival function.
        q = tensor([.1, 1., 5., 20., 100., 1000., 4000.])
        x = q.cpu().numpy()
        log_a = math.log(2.) + log_ndtr(-np.sqrt(x))
        log_b = .5 * math.log(2.) - x/4 + np.log(erf(np.sqrt(x/4)))
        reference = -np.logaddexp(log_a, log_b) / math.log(10.)
        observed = weighted_chi2_logsf(q, tensor([1., 2., 2.]))
        np.testing.assert_allclose(observed.cpu(), reference, rtol=2e-6, atol=8e-6)
        self.assertGreater(float(observed[-1]), 308.)

    def test_weighted_rank_four_against_hypoexponential_closed_form(self):
        q = tensor([.1, 1., 5., 20., 100., 1000., 4000.])
        # chi2_2 + 2*chi2_2: sf(q)=2*exp(-q/4)-exp(-q/2).
        reference = (q/4-torch.log(2-torch.exp(-q/4))) / math.log(10.)
        observed = weighted_chi2_logsf(q, tensor([1., 1., 2., 2.]))
        torch.testing.assert_close(observed, reference, rtol=2e-10, atol=2e-10)

    def test_weighted_rank_six_against_three_exponential_convolution(self):
        q = tensor([.1, 1., 5., 20., 100., 1000., 5000.])
        # chi2_2 + 2*chi2_2 + 3*chi2_2: factor out the slowest
        # exponential to retain a closed form at probabilities below 1e-308.
        reference = (q/6 - torch.log(4.5 - 4*torch.exp(-q/12)
                                    + .5*torch.exp(-q/3))) / math.log(10.)
        observed = weighted_chi2_logsf(q, tensor([1., 1., 2., 2., 3., 3.]))
        torch.testing.assert_close(observed, reference, rtol=2e-6, atol=8e-6)
        self.assertGreater(float(observed[-1]), 308.)

    def test_explicit_cpu_fallback_and_rejecting_inaccurate_gpu_integral(self):
        numerical_diagnostics(reset=True)
        ev = tensor([1., 2., 3.])
        with self.assertRaises(ArithmeticError):
            weighted_chi2_logsf(tensor(20.), ev, max_order=64, allow_cpu_fallback=False)
        result = weighted_chi2_logsf(tensor(20.), ev, max_order=64)
        self.assertTrue(bool(torch.isfinite(result)))
        diagnostics = numerical_diagnostics()
        self.assertEqual(diagnostics["cpu_tail_fallback_calls"], 1)
        self.assertGreater(diagnostics["cpu_tail_fallback_seconds"], 0.)


class GeneInferenceTests(unittest.TestCase):
    def test_skat_with_weights_equals_weighted_chi_square(self):
        u = tensor([2., -.5, 1.])
        k = tensor([[2., .1, 0.], [.1, 1., .2], [0., .2, 3.]])
        w = tensor([1., 2., .5])
        observed = skat_logp(u, k, w, tail_method="exact")
        target = weighted_chi2_logsf((u*w).square().sum(),
                    torch.linalg.eigvalsh(k*w[:, None]*w[None, :]))
        self.assertAlmostEqual(float(observed), float(target), places=8)

    def test_regenie_small_tail_policy_uses_gpu_kuonen(self):
        k = torch.diag(tensor([1., 2., 3.]))
        u = tensor([10., 0., 0.])
        numerical_diagnostics(reset=True)
        observed = skat_logp(u, k)
        expected = kuonen_logsf(u.square().sum(), k.diagonal())
        self.assertAlmostEqual(float(observed), float(expected), places=12)
        self.assertEqual(numerical_diagnostics()["kuonen_tail_values"], 1)
        exact = skat_logp(u, k, tail_method="exact")
        self.assertGreater(abs(float(exact-observed)), 1e-3)
        self.assertGreater(float(kuonen_logsf(tensor(6000.), k.diagonal())), 308.)

    def test_skato_integral_against_independent_two_dimensional_null(self):
        u = tensor([2., -1.])
        result = skato_logp(u, torch.eye(2, dtype=torch.float64, device=DEVICE))
        rhos = result["rhos"].cpu().numpy()
        self.assertEqual(rhos[-1], .999)  # source caps its logged rho=1 internally
        pmin = 10 ** -float(result["rho_log10ps"].max())
        eigs = np.array([[1-r, 1+r] for r in rhos])
        c1, c2, c4 = eigs.sum(1), (eigs**2).sum(1), (eigs**4).sum(1)
        dfs = c2**2 / c4
        critical = c1 + (chi2.isf(pmin, dfs)-dfs) * np.sqrt(c2/dfs)
        # Under identity K, the burden direction and its orthogonal direction
        # are independent. Integrate over the angle of an isotropic normal;
        # its squared radius is chi2_2 and has elementary survival.
        probability, error = quad(lambda theta: math.exp(-.5 * np.min(
            critical / ((1-rhos)*math.sin(theta)**2 + (1+rhos)*math.cos(theta)**2))),
            0, math.pi/2, epsabs=1e-10, epsrel=1e-8, limit=500)
        probability *= 2 / math.pi
        probability = min(probability, len(rhos)*pmin)
        self.assertLess(error, 1e-7)
        self.assertAlmostEqual(float(result["SKATO"]), -math.log10(probability), delta=2e-4)
        self.assertGreater(abs(float(result["SKATO"]-result["SKATO-ACAT"])), 1e-3)
        self.assertEqual(result["SKATO"].device, DEVICE)

    def test_skato_one_variant_degeneracy(self):
        result = skato_logp(tensor([3.]), tensor([[2.]]))
        for key in ("SKAT", "BURDEN", "SKATO", "SKATO-ACAT"):
            self.assertAlmostEqual(float(result[key]), float(chi2_logsf(tensor(4.5))), places=11)
        zero = skato_logp(tensor([0.]), tensor([[2.]]))
        self.assertEqual(float(zero["SKATO-ACAT"]), 0.)

    def test_skato_extreme_probability_and_scale_invariance(self):
        u = tensor([50., -30.])
        k = torch.eye(2, dtype=torch.float64, device=DEVICE)
        result = skato_logp(u, k)
        self.assertGreater(float(result["SKATO"]), 308.)
        self.assertTrue(bool(torch.isfinite(result["SKATO"])))
        self.assertGreaterEqual(float(result["SKATO"]),
                                float(result["rho_log10ps"].max())-math.log10(8)-1e-9)
        scaled = skato_logp(u*1e-9, k*1e-18)
        self.assertAlmostEqual(float(scaled["SKATO"]), float(result["SKATO"]), delta=1e-7)

    def test_sbat_nnls_kkt_and_independent_mixture(self):
        k = tensor([[1., .6, .2], [.6, 1., .3], [.2, .3, 1.]])
        u = tensor([1., -.5, .8])
        b = nnls_coefficients(u, k)
        gradient = k @ b-u
        self.assertTrue(bool((b >= 0).all()))
        self.assertLess(float((b*gradient).abs().max()), 1e-10)
        self.assertTrue(bool((gradient[b == 0] >= -1e-10).all()))
        identity = torch.eye(2, dtype=torch.float64, device=DEVICE)
        result = sbat_logp(tensor([2., -1.]), identity)
        np.testing.assert_allclose(result["mixture_weights"].cpu(), [.25, .5, .25], atol=1e-13)
        pp = .5*chi2.sf(4., 1) + .25*chi2.sf(4., 2)
        pn = .5*chi2.sf(1., 1) + .25*chi2.sf(1., 2)
        self.assertAlmostEqual(float(result["SBAT_POS"]), -math.log10(pp), places=11)
        self.assertAlmostEqual(float(result["SBAT_NEG"]), -math.log10(pn), places=11)
        self.assertAlmostEqual(float(result["SBAT"]), acat_reference([pp, pn]), places=11)
        zero = sbat_logp(tensor([0., 0.]), identity)
        self.assertAlmostEqual(float(zero["SBAT_POS"]), -math.log10(.75), places=12)

    def test_sbat_variance_and_orthant_weights(self):
        k = tensor([[1., .6], [.6, 1.]])
        w = chi_bar_weights(k, max_subsets=0)
        # w0 is the positive-score orthant; w2 is the inverse-Gram orthant.
        expected0 = .25 + math.asin(.6)/(2*math.pi)
        expected2 = .25 - math.asin(.6)/(2*math.pi)
        np.testing.assert_allclose(w.cpu(), [expected0, .5, expected2], atol=1e-13)
        result = sbat_logp(tensor([3., 2.]), k, variance_scale=2.)
        b = result["coefficients_positive"]
        self.assertAlmostEqual(float(result["statistic_positive"]), float(b@k@b)/2., places=12)

    def test_normal_orthant_qmc_against_scalar_mvn_reference(self):
        k = tensor([[1., .35, .2, .1], [.35, 1., .25, .15],
                    [.2, .25, 1., .3], [.1, .15, .3, 1.]])
        observed = normal_orthant_probability(k, qmc_samples=32768, seed=73)
        reference = multivariate_normal.cdf(np.zeros(4), mean=np.zeros(4), cov=k.cpu().numpy(),
                     maxpts=2000000, abseps=2e-6, releps=2e-6)
        self.assertAlmostEqual(float(observed), reference, delta=2e-4)
        self.assertEqual(observed.device, DEVICE)
        w = chi_bar_weights(torch.eye(4, dtype=torch.float64, device=DEVICE), max_subsets=10)
        np.testing.assert_allclose(w.cpu(), [1/16, 4/16, 6/16, 4/16, 1/16], atol=2e-12)


if __name__ == "__main__":
    unittest.main()
