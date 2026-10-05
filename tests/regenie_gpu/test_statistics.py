"""Independent analytic/scalar references for the GPU association inference."""

import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr, erf, ndtr
from scipy.stats import chi2, multivariate_normal
import torch

from torchwgs.statistics import (
    DEFAULT_RHOS, acat_logp, chi2_isf_logp, chi2_logsf, chi_bar_weights,
    eigen_compatible_column_norm,
    davies_logsf, diagnostics_scope, kuonen_logsf, nnls_coefficients,
    normal_orthant_probability, numerical_diagnostics,
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
    def test_nested_numerical_diagnostic_scopes_restore_the_outer_ledger(self):
        q = torch.tensor([1., 2.], dtype=torch.float64)
        eigenvalues = torch.tensor([.2, .5], dtype=torch.float64)
        before = numerical_diagnostics()
        with diagnostics_scope() as outer:
            davies_logsf(q, eigenvalues)
            self.assertEqual(outer["davies_tail_values"], 2)
            with diagnostics_scope() as inner:
                self.assertEqual(numerical_diagnostics()["davies_tail_values"], 0)
                davies_logsf(q[:1], eigenvalues)
            self.assertEqual(inner["davies_tail_values"], 1)
            self.assertEqual(numerical_diagnostics()["davies_tail_values"], 2)
        self.assertEqual(outer["davies_tail_values"], 2)
        self.assertEqual(numerical_diagnostics(), before)

    def test_parallel_diagnostics_do_not_reset_or_count_another_worker(self):
        barrier = Barrier(2)
        def worker(count):
            with diagnostics_scope() as ledger:
                davies_logsf(torch.arange(1, count+1, dtype=torch.float64),
                             torch.tensor([.2, .5], dtype=torch.float64))
                barrier.wait()
                actual = numerical_diagnostics()["davies_tail_values"]
                numerical_diagnostics(reset=True)
                barrier.wait()
                self.assertEqual(numerical_diagnostics()["davies_tail_values"], 0)
            self.assertEqual(ledger["davies_tail_values"], 0)
            return actual
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, count) for count in (1, 3)]
            self.assertEqual([future.result() for future in futures], [1, 3])

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

    def test_df1_inverse_independent_cdf_near_one_and_extreme_tails(self):
        values = np.array([0., 1e-30, 1e-20, 1e-17, 1e-12, 1e-6, .001,
                           .099999, .1, .100001, .3, 1., 5., 20., 100.,
                           299.999, 300., 300.000001, 500., 1000.])
        for device in (DEVICE, torch.device('cpu')):
            observed = chi2_isf_logp(torch.tensor(values, device=device)).cpu().numpy()
            self.assertEqual(observed[0], 0.)
            self.assertTrue(np.all(observed[1:] > 0))
            near_one = values < .1
            # SciPy's chi-square inverse incomplete gamma is independent of
            # the direct normal quantile used by the GPU implementation.
            expected_lower = chi2.ppf(-np.expm1(-values[near_one]*math.log(10)), 1)
            np.testing.assert_allclose(observed[near_one], expected_lower, rtol=4e-14, atol=0)
            recovered_lower = -np.log1p(-erf(np.sqrt(observed[near_one]/2)))/math.log(10)
            np.testing.assert_allclose(recovered_lower, values[near_one], rtol=4e-14, atol=0)
            ordinary = (values >= .1) & (values <= 300)
            expected = chi2.isf(np.exp(-values[ordinary]*math.log(10)), 1)
            np.testing.assert_allclose(observed[ordinary], expected, rtol=5e-14, atol=2e-14)
            extreme = values > 300
            expected_extreme = [brentq(
                lambda x: -(math.log(2)+log_ndtr(-np.sqrt(x)))/math.log(10)-lp,
                0., 2*lp*math.log(10)+100., xtol=1e-11) for lp in values[extreme]]
            np.testing.assert_allclose(observed[extreme], expected_extreme, rtol=2e-14, atol=2e-11)
            recovered = -(math.log(2)+log_ndtr(-np.sqrt(observed[~near_one])))/math.log(10)
            np.testing.assert_allclose(recovered, values[~near_one], rtol=2e-13, atol=2e-13)

    def test_inverse_broadcasts_mixed_degrees_and_rejects_invalid_df(self):
        values = tensor([[.001, 2., 20.], [100., 300., 1000.]])
        degrees = tensor([[1.], [2.]])
        observed = chi2_isf_logp(values, degrees)
        expected = np.vstack([chi2.isf(np.exp(-values[0].cpu().numpy()*math.log(10)), 1),
                              2*math.log(10)*values[1].cpu().numpy()])
        np.testing.assert_allclose(observed.cpu(), expected, rtol=3e-12, atol=2e-14)
        self.assertEqual(observed.device, DEVICE)
        self.assertEqual(float(chi2_isf_logp(tensor(0.))), 0.)
        for df in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                chi2_isf_logp(tensor(1.), df)

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
            for controller in ('scalar','numpy','auto'):
                observed, fault = davies_logsf(tensor(q), tensor(spectrum), controller=controller)
                self.assertEqual(int(fault), 0)
                self.assertAlmostEqual(float(observed), expected, delta=2e-12)
                self.assertEqual(observed.device, DEVICE)

    def test_davies_large_spectrum_controllers_agree_with_chi_square(self):
        spectrum=tensor([1.]*1024)
        q=tensor([900.,1024.,1150.])
        expected=tensor(-np.log10(chi2.sf(q.cpu().numpy(),1024)))
        scalar,scalar_fault=davies_logsf(q,spectrum,controller='scalar')
        for controller in ('numpy','auto'):
            value,fault=davies_logsf(q,spectrum,controller=controller)
            torch.testing.assert_close(value,scalar,rtol=0,atol=2e-13)
            torch.testing.assert_close(fault,scalar_fault,rtol=0,atol=0)
            torch.testing.assert_close(value,expected,rtol=0,atol=2e-5)
        with self.assertRaises(ValueError):
            davies_logsf(q,spectrum,controller='invalid')

    def test_davies_budget_failure_switches_moderate_p_to_kuonen(self):
        spectrum = tensor([1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .1, 1.])
        q = tensor(.0441776)
        numerical_diagnostics(reset=True)
        lp, fault = davies_logsf(q, spectrum)
        self.assertEqual(int(fault), 1)  # native consumes its auxiliary budget
        self.assertTrue(bool(torch.isnan(lp)))
        for controller in ('scalar','numpy'):
            alternate,alternate_fault=davies_logsf(q,spectrum,controller=controller)
            self.assertTrue(bool(torch.isnan(alternate)))
            self.assertEqual(int(alternate_fault),1)
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
    def test_sbat_subset_policy_is_repeatable_and_preserves_independent_mixture(self):
        gram = torch.eye(6, dtype=torch.float64)
        expected = torch.tensor([math.comb(6, df)/64 for df in range(7)], dtype=torch.float64)
        for policy in ("unique", "with_replacement"):
            first = chi_bar_weights(gram, max_subsets=2, qmc_samples=32, seed=19,
                                    subset_sampling=policy)
            repeated = chi_bar_weights(gram, max_subsets=2, qmc_samples=32, seed=19,
                                       subset_sampling=policy)
            torch.testing.assert_close(first, repeated, atol=0, rtol=0)
            torch.testing.assert_close(first, expected, atol=2e-15, rtol=2e-15)
        with self.assertRaises(ValueError):
            chi_bar_weights(gram, subset_sampling="invalid")

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

    def test_native_skato_gate_distinguishes_one_site_from_rank_one_multisite(self):
        # Native get_lambdas accepts its scalar specialization even when the
        # burden-deflated scalar is zero. A multi-site zero spectrum fails.
        single = skato_logp(tensor([3.]), tensor([[2.]]), native_validity=True)
        self.assertTrue(single["kernel_valid"])
        self.assertAlmostEqual(float(single["SKATO"]),
                               float(chi2_logsf(tensor(4.5))), places=11)
        duplicated = tensor([[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]])
        score = tensor([1., 1., 1.])
        failed = skato_logp(score, duplicated, native_validity=True)
        self.assertFalse(failed["kernel_valid"])
        self.assertEqual(failed["kernel_failure"], "empty_residual_spectrum")
        self.assertEqual(failed["rho_log10ps"].numel(), 0)
        self.assertNotIn("SKAT", failed)
        # The standalone mathematical API still reports the valid rank-one
        # chi-square distribution when native output validity is not requested.
        mathematical = skato_logp(score, duplicated)
        self.assertAlmostEqual(float(mathematical["SKAT"]),
                               float(chi2_logsf(tensor(.125))), places=11)
        # REGENIE's fixed-rho route does not need a deflated spectrum.
        fixed = skato_logp(score, duplicated, rhos=(.3,), native_validity=True)
        self.assertTrue(fixed["kernel_valid"])
        self.assertAlmostEqual(float(fixed["SKATO"]),
                               float(chi2_logsf(tensor(.125))), places=11)

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

    def test_adaptive_skato_backends_match_independent_null_and_keep_extreme_tails(self):
        reference = skato_logp(tensor([2., -1.]), torch.eye(2, device=DEVICE, dtype=torch.float64),
                              tail_method="exact", integral_rtol=2e-6)
        for backend in ("adaptive_x", "adaptive_sqrt"):
            result = skato_logp(tensor([2., -1.]), torch.eye(2, device=DEVICE, dtype=torch.float64),
                tail_method="exact", integral_backend=backend, integral_epsabs=0., integral_epsrel=2e-6)
            self.assertTrue(result["integral_diagnostics"]["converged"])
            self.assertAlmostEqual(float(result["SKATO"]), float(reference["SKATO"]), delta=2e-5)
            self.assertEqual(result["SKATO"].device, DEVICE)
            extreme = skato_logp(tensor([50., -30.]), torch.eye(2, device=DEVICE, dtype=torch.float64),
                integral_backend=backend, integral_epsabs=0., integral_epsrel=2e-6)
            self.assertTrue(bool(torch.isfinite(extreme["SKATO"])))
            self.assertGreater(float(extreme["SKATO"]), 308.)
            self.assertGreaterEqual(float(extreme["SKATO"]),
                float(extreme["rho_log10ps"].max())-math.log10(8)-1e-9)
        with self.assertRaises(ValueError):
            skato_logp(tensor([1.]), tensor([[1.]]), integral_backend="unknown")

    def test_adaptive_skato_budget_failure_uses_source_fallback(self):
        for score, available in (([3., -2.], True), ([.4, -.2], False)):
            result = skato_logp(tensor(score), torch.eye(2, device=DEVICE, dtype=torch.float64),
                integral_backend="adaptive_sqrt", integral_epsabs=0., integral_epsrel=1e-12,
                integral_max_intervals=1)
            self.assertFalse(result["integral_diagnostics"]["converged"])
            self.assertTrue(result["kernel_valid"])
            self.assertIn("SKAT", result)
            self.assertIn("SKATO-ACAT", result)
            if available:
                self.assertEqual(result["integral_fallback"], "bonferroni")
                self.assertAlmostEqual(float(result["SKATO"]),
                    float(result["rho_log10ps"].max())-math.log10(8), places=12)
            else:
                self.assertEqual(result["integral_fallback"], "unavailable")
                self.assertIsNone(result["SKATO"])

    def test_successful_skato_integral_rejects_probability_above_one(self):
        # Inject an over-one integral estimate without changing the actual
        # score, spectra, rho tests, or Bonferroni bound. This exercises the
        # original get_logp validity gate after a successful integration.
        def integration_patch(backend):
            if backend == "segmented":
                return patch("torchwgs.statistics._legendre", return_value=(
                    tensor([0.]), tensor([1e6])))
            return patch("torchwgs._quadrature.integrate_log_gk21", return_value=(
                tensor(math.log(2.)), {"converged": True, "status": "converged",
                                     "evaluations": 21, "intervals": 1}))

        covariance = torch.eye(2, device=DEVICE, dtype=torch.float64)
        for backend in ("adaptive_x", "adaptive_sqrt", "segmented"):
            with self.subTest(backend=backend), integration_patch(backend):
                mathematical = skato_logp(tensor([.4, -.2]), covariance,
                                          integral_backend=backend)
                native = skato_logp(tensor([.4, -.2]), covariance,
                                    integral_backend=backend, native_validity=True)
                self.assertLess(float(native["rho_log10ps"].max()), math.log10(8))
                self.assertIsNone(native["SKATO"])
                self.assertEqual(float(mathematical["SKATO"]), 0.)
                self.assertTrue(native["kernel_valid"])
                for name in ("SKAT", "BURDEN", "SKATO-ACAT", "rho_log10ps", "rhos"):
                    torch.testing.assert_close(native[name], mathematical[name])
                if backend != "segmented":
                    self.assertTrue(native["integral_diagnostics"]["converged"])
                    self.assertNotIn("integral_fallback", native)
                restored = skato_logp(tensor([3., -2.]), covariance,
                                      integral_backend=backend, native_validity=True)
                expected = float(restored["rho_log10ps"].max())-math.log10(8)
                self.assertGreater(expected, 0.)
                self.assertAlmostEqual(float(restored["SKATO"]), expected, places=12)
                # Bonferroni P=1 is valid: the rejection is strictly P>1.
                with patch("torchwgs.statistics._association_tail",
                           return_value=tensor(math.log10(8))):
                    boundary = skato_logp(tensor([.4, -.2]), covariance,
                        integral_backend=backend, native_validity=True)
                self.assertEqual(float(boundary["SKATO"]), 0.)

    def test_native_skato_near_one_gate_includes_its_source_boundary(self):
        source_boundary = -math.log10(1-torch.finfo(torch.float32).eps)
        with patch("torchwgs.statistics._association_tail", return_value=tensor(source_boundary)), \
                patch("torchwgs._quadrature.integrate_log_gk21") as integral:
            result = skato_logp(tensor([.4, -.2]),
                torch.eye(2, device=DEVICE, dtype=torch.float64),
                integral_backend="adaptive_x", native_validity=True)
        self.assertEqual(float(result["SKATO"]), 0.)
        integral.assert_not_called()

    def test_native_skato_conditional_underflow_preserves_source_failure(self):
        from torchwgs import statistics as stats
        original_tail = stats._association_tail

        def underflow_tail(q, eigenvalues, method, **kwargs):
            if kwargs.get("rank_one_exact") is False:
                return torch.full_like(q, 1000.)  # finite LP, ordinary SF=0
            return original_tail(q, eigenvalues, method, **kwargs)

        def checked_integral(function, upper, **kwargs):
            function((upper*.1).reshape(1))
            return tensor(math.log(.5)), {"converged": True, "status": "converged",
                                         "evaluations": 21, "intervals": 1}

        covariance = torch.eye(2, device=DEVICE, dtype=torch.float64)
        for backend in ("adaptive_x", "adaptive_sqrt", "segmented"):
            integration = (patch("torchwgs.statistics._legendre", return_value=(tensor([0.]), tensor([2.])))
                           if backend == "segmented" else
                           patch("torchwgs._quadrature.integrate_log_gk21", side_effect=checked_integral))
            with self.subTest(backend=backend), integration, \
                    patch("torchwgs.statistics._association_tail", side_effect=underflow_tail):
                raw = skato_logp(tensor([.4, -.2]), covariance, integral_backend=backend)
                failed = skato_logp(tensor([.4, -.2]), covariance,
                                    integral_backend=backend, native_validity=True)
                self.assertIsNone(failed["SKATO"])
                self.assertEqual(failed["integral_fallback"], "unavailable")
                self.assertTrue(failed["kernel_valid"])
                self.assertNotIn("integral_fallback", raw)
                self.assertIsNotNone(raw["SKATO"])
                for name in ("SKAT", "BURDEN", "SKATO-ACAT", "rho_log10ps", "rhos"):
                    torch.testing.assert_close(failed[name], raw[name])
                if backend != "segmented":
                    self.assertEqual(failed["integral_diagnostics"]["status"], "integrand_failure")
                restored = skato_logp(tensor([3., -2.]), covariance,
                                      integral_backend=backend, native_validity=True)
                self.assertEqual(restored["integral_fallback"], "bonferroni")
                self.assertAlmostEqual(float(restored["SKATO"]),
                    float(restored["rho_log10ps"].max())-math.log10(8), places=12)

    def test_native_skato_intentional_zero_is_exempt_from_underflow_failure(self):
        from torchwgs import statistics as stats
        original_tail = stats._association_tail
        intentional_underflow_values = []
        forced_logp = 1000.

        def intentional_zero_tail(q, eigenvalues, method, **kwargs):
            if kwargs.get("rank_one_exact") is False:
                # Residual mu is 1e-6 for this covariance; q>.1 implies
                # envelope>mu*1e4, where the source deliberately returns zero.
                large = q > .1
                intentional_underflow_values.append(int(large.sum()))
                return torch.where(large, torch.full_like(q, forced_logp), torch.zeros_like(q))
            return original_tail(q, eigenvalues, method, **kwargs)

        def checked_integral(function, upper, **kwargs):
            values = function((upper*.01).reshape(1))
            self.assertTrue(bool(torch.isneginf(values).all()))
            return tensor(-math.inf), {"converged": True, "status": "converged",
                                      "evaluations": 21, "intervals": 1}

        covariance = tensor([[1., .999999], [.999999, 1.]])
        for backend in ("adaptive_x", "adaptive_sqrt", "segmented"):
            results = []
            for forced_logp in (1000., 0.):
                intentional_underflow_values.clear()
                integration = (patch("torchwgs.statistics._legendre", return_value=(tensor([0.]), tensor([2.])))
                               if backend == "segmented" else
                               patch("torchwgs._quadrature.integrate_log_gk21", side_effect=checked_integral))
                with self.subTest(backend=backend, forced_logp=forced_logp), integration, \
                        patch("torchwgs.statistics._association_tail", side_effect=intentional_zero_tail):
                    result = skato_logp(tensor([1., 1.]), covariance,
                                        integral_backend=backend, native_validity=True)
                self.assertGreater(sum(intentional_underflow_values), 0)
                self.assertNotIn("integral_fallback", result)
                self.assertIsNotNone(result["SKATO"])
                results.append(result["SKATO"])
            # The source forces S=0 here even when the tail calculator
            # would produce the representable probability S=1.
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_native_skato_underflow_boundary_matches_cpu_ordinary_probability(self):
        from torchwgs import statistics as stats
        original_tail = stats._association_tail
        cutoff = -1075*math.log(2.)
        requested_logs = [-745., math.nextafter(cutoff, math.inf), cutoff,
                          math.nextafter(cutoff, -math.inf), -1000.]
        conditional_lp = tensor(0.)

        def boundary_tail(q, eigenvalues, method, **kwargs):
            if kwargs.get("rank_one_exact") is False:
                return torch.full_like(q, float(conditional_lp))
            return original_tail(q, eigenvalues, method, **kwargs)

        def checked_integral(function, upper, **kwargs):
            function((upper*.1).reshape(1))
            return tensor(-math.inf), {"converged": True, "status": "converged",
                                      "evaluations": 21, "intervals": 1}

        covariance = torch.eye(2, device=DEVICE, dtype=torch.float64)
        for backend in ("adaptive_x", "adaptive_sqrt", "segmented"):
            for requested_log in requested_logs:
                conditional_lp = tensor(-requested_log/math.log(10.))
                # Use the actual tensor log after the LP round trip. libm
                # exp is independent of the CUDA exp underflow boundary.
                actual_log = float(-conditional_lp*math.log(10.))
                ordinary_probability_failed = math.exp(actual_log) == 0
                integration = (patch("torchwgs.statistics._legendre", return_value=(tensor([0.]), tensor([2.])))
                               if backend == "segmented" else
                               patch("torchwgs._quadrature.integrate_log_gk21", side_effect=checked_integral))
                with self.subTest(backend=backend, requested_log=requested_log), integration, \
                        patch("torchwgs.statistics._association_tail", side_effect=boundary_tail):
                    native = skato_logp(tensor([.4, -.2]), covariance,
                        integral_backend=backend, native_validity=True)
                    raw = skato_logp(tensor([.4, -.2]), covariance, integral_backend=backend)
                if ordinary_probability_failed:
                    self.assertIsNone(native["SKATO"])
                    self.assertEqual(native["integral_fallback"], "unavailable")
                else:
                    self.assertIsNotNone(native["SKATO"])
                    self.assertNotIn("integral_fallback", native)
                if requested_log == -745.:
                    self.assertGreater(math.exp(actual_log), 0.)
                    self.assertIsNotNone(native["SKATO"])
                self.assertIsNotNone(raw["SKATO"])
                self.assertNotIn("integral_fallback", raw)
                for name in ("SKAT", "SKATO-ACAT", "rho_log10ps"):
                    torch.testing.assert_close(native[name], raw[name])

    def test_native_skato_probability_floor_leaves_bypasses_and_raw_api_unchanged(self):
        from torchwgs import statistics as stats
        original_chi2_tail = stats.chi2_logsf
        maximum_native_logp = -math.log10(10*torch.finfo(torch.float64).tiny)

        def mixture_tail(q, eigenvalues, method, **kwargs):
            return torch.full_like(q, 0. if kwargs.get("rank_one_exact") is False else 1000.)

        def extreme_scalar_tail(q, df=1.):
            if torch.as_tensor(q).numel() == 1 and torch.as_tensor(df).numel() == 1 and float(df) == 1:
                return torch.full_like(torch.as_tensor(q), 1000.)
            return original_chi2_tail(q, df)

        covariance = torch.eye(2, device=DEVICE, dtype=torch.float64)
        for backend in ("adaptive_x", "adaptive_sqrt", "segmented"):
            integration = (patch("torchwgs.statistics._legendre", return_value=(tensor([0.]), tensor([0.])))
                           if backend == "segmented" else
                           patch("torchwgs._quadrature.integrate_log_gk21", return_value=(
                               tensor(-math.inf), {"converged": True, "status": "converged",
                                                   "evaluations": 21, "intervals": 1})))
            with self.subTest(backend=backend), integration, \
                    patch("torchwgs.statistics._association_tail", side_effect=mixture_tail), \
                    patch("torchwgs.statistics.chi2_logsf", side_effect=extreme_scalar_tail), \
                    patch("torchwgs.statistics.chi2_isf_logp", wraps=stats.chi2_isf_logp) as inverse:
                result = skato_logp(tensor([2., -1.]), covariance,
                                    integral_backend=backend, native_validity=True)
                self.assertTrue(bool((inverse.call_args[0][0] == maximum_native_logp).all()))
                self.assertEqual(float(result["SKATO"]), maximum_native_logp)
                inverse.reset_mock()
                raw = skato_logp(tensor([2., -1.]), covariance, integral_backend=backend)
                self.assertTrue(bool((inverse.call_args[0][0] == 1000.).all()))
                self.assertAlmostEqual(float(raw["SKATO"]), 1000., places=12)
                inverse.reset_mock()
                single = skato_logp(tensor([2.]), tensor([[1.]]), native_validity=True)
                fixed = skato_logp(tensor([2., -1.]), covariance,
                                  rhos=(.3,), native_validity=True)
                self.assertEqual(float(single["SKATO"]), 1000.)
                self.assertEqual(float(fixed["SKATO"]), 1000.)
                inverse.assert_not_called()

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

    def test_four_dimensional_orthant_against_independent_conditional_integral(self):
        # Equicorrelation admits a separate one-factor conditional integral.
        # Integrate the boundary deficit, which stays resolved even when the
        # shared normal factor makes the probability approach one half.
        for rho in (.2, .8, .99, .9999999, 1-1e-12):
            factor = math.sqrt(rho/(1-rho))
            deficit = quad(lambda u: math.exp(-(u/factor)**2/2)/math.sqrt(2*math.pi)
                           * (1-ndtr(u)**4-ndtr(-u)**4)/factor,
                           0, 12, epsabs=1e-12, epsrel=1e-12)[0]
            covariance = torch.eye(4, dtype=torch.float64, device=DEVICE)*(1-rho)+rho
            observed = normal_orthant_probability(covariance, seed=13)
            self.assertAlmostEqual(float(observed), .5-deficit, delta=3e-9)
            self.assertEqual(float(observed), float(normal_orthant_probability(covariance, seed=177)))
        pairs = tensor([[1., .7, 0, 0], [.7, 1., 0, 0],
                        [0, 0, 1., -.8], [0, 0, -.8, 1.]])
        expected = (.25+math.asin(.7)/(2*math.pi))*(.25-math.asin(.8)/(2*math.pi))
        self.assertAlmostEqual(float(normal_orthant_probability(pairs)), expected, delta=2e-12)

    def test_four_dimensional_orthant_against_scalar_mvn_reference(self):
        k = tensor([[1., .35, .2, .1], [.35, 1., .25, .15],
                    [.2, .25, 1., .3], [.1, .15, .3, 1.]])
        observed = normal_orthant_probability(k, qmc_samples=32768, seed=73)
        reference = multivariate_normal.cdf(np.zeros(4), mean=np.zeros(4), cov=k.cpu().numpy(),
                     maxpts=2000000, abseps=2e-6, releps=2e-6)
        self.assertAlmostEqual(float(observed), reference, delta=2e-4)
        self.assertEqual(observed.device, DEVICE)
        w = chi_bar_weights(torch.eye(4, dtype=torch.float64, device=DEVICE), max_subsets=10)
        np.testing.assert_allclose(w.cpu(), [1/16, 4/16, 6/16, 4/16, 1/16], atol=2e-12)



class SecularSkatoIntegrationTests(unittest.TestCase):
    def test_complete_skato_matches_dense_with_duplicate_spectrum(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            covariance=torch.tensor([[2.,.5,.5,0.],[.5,2.,.5,0.],[.5,.5,2.,0.],[0.,0.,0.,1.5]],dtype=torch.float64,device=device)
            score=torch.tensor([1.1,-.7,.3,.2],dtype=torch.float64,device=device)
            dense=skato_logp(score,covariance,eigen_backend='dense')
            secular=skato_logp(score,covariance,eigen_backend='secular')
            for key in ('SKAT','BURDEN','SKATO','SKATO-ACAT','rho_log10ps'):
                torch.testing.assert_close(dense[key],secular[key],atol=2e-8,rtol=2e-8)
            with self.assertRaises(ValueError):
                skato_logp(score,covariance,eigen_backend='unknown')

if __name__ == "__main__":
    unittest.main()
