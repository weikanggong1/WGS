"""Analytic CPU/CUDA checks for the independent positive-integral engine."""
import math
import unittest

import torch

from torchwgs._quadrature import integrate_log_gk21


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


class QuadratureTests(unittest.TestCase):
    def test_singular_endpoint_and_gaussian_density_match_analytic_integrals(self):
        for device in DEVICES:
            with self.subTest(device=device):
                upper = torch.tensor(1., device=device, dtype=torch.float64)
                integral, info = integrate_log_gk21(lambda x: -.5*x.log(), upper,
                    epsabs=0., epsrel=1e-6)
                self.assertTrue(info["converged"])
                self.assertAlmostEqual(float(integral), math.log(2.), delta=1e-6)
                self.assertGreater(info["intervals"], 1)
                gaussian, info = integrate_log_gk21(lambda x: -x.square()/2,
                    upper*3, epsabs=0., epsrel=1e-8)
                expected = math.sqrt(math.pi/2)*math.erf(3/math.sqrt(2))
                self.assertTrue(info["converged"])
                self.assertAlmostEqual(float(gaussian), math.log(expected), delta=1e-8)
                self.assertEqual(gaussian.device, device)

    def test_logarithmic_scaling_preserves_probability_below_float64_range(self):
        for device in DEVICES:
            with self.subTest(device=device):
                upper = torch.tensor(1., device=device, dtype=torch.float64)
                integral, info = integrate_log_gk21(lambda x: -1000+2*x.log(), upper,
                    epsabs=0., epsrel=1e-8)
                self.assertTrue(info["converged"])
                self.assertAlmostEqual(float(integral), -1000-math.log(3.), delta=1e-10)
                self.assertEqual(float(integral.exp()), 0.)

    def test_source_coordinate_resolves_narrow_endpoint_feature(self):
        # Independent analytic example: a narrow feature can fall between all
        # first-rule sqrt-coordinate nodes, despite a tiny error estimate.
        # The original x-coordinate refinement resolves this particular case.
        epsilon = 1e-7
        expected = (math.erf(math.sqrt(.5)) - (1 + 2/epsilon)**-.5 *
                    math.erf(math.sqrt(.5 + 1/epsilon)))
        for device in DEVICES:
            with self.subTest(device=device):
                upper = torch.tensor(1., device=device, dtype=torch.float64)
                def log_density_x(x):
                    return (-x/2 - .5*(math.log(2*math.pi) + x.log()) +
                            torch.log(-torch.expm1(-x/epsilon)))
                def log_density_sqrt(t):
                    return (-t.square()/2 + .5*math.log(2/math.pi) +
                            torch.log(-torch.expm1(-t.square()/epsilon)))
                source, info = integrate_log_gk21(log_density_x, upper,
                    epsabs=1e-25, epsrel=2**-13, max_intervals=1000)
                transformed, transformed_info = integrate_log_gk21(
                    log_density_sqrt, upper, epsabs=1e-25,
                    epsrel=2**-13, max_intervals=1000)
                self.assertTrue(info["converged"])
                self.assertGreater(info["intervals"], 1)
                self.assertAlmostEqual(float(source.exp()), expected, delta=1e-7)
                # Record the known limitation instead of asserting that the
                # quadrature error estimate always bounds the actual error.
                self.assertTrue(transformed_info["converged"])
                self.assertLess(transformed_info["relative_error_estimate"], 1e-10)
                self.assertGreater(abs(float(transformed.exp())-expected), 2e-4)

    def test_exhausted_budget_is_reported_and_parameters_are_validated(self):
        for device in DEVICES:
            upper = torch.tensor(1., device=device, dtype=torch.float64)
            _, info = integrate_log_gk21(lambda x: -.5*x.log(), upper,
                epsabs=0., epsrel=1e-8, max_intervals=1)
            self.assertFalse(info["converged"])
            self.assertEqual(info["status"], "subdivision_limit")
            self.assertEqual(info["intervals"], 1)
            self.assertEqual(info["evaluations"], 21)
            zero, info = integrate_log_gk21(lambda x: x, upper*0)
            self.assertTrue(bool(torch.isneginf(zero)))
            self.assertTrue(info["converged"])
        for values in ({"epsabs": -1}, {"epsrel": 0}, {"epsrel": math.nan},
                       {"max_intervals": 0}, {"max_intervals": True}):
            with self.assertRaises(ValueError):
                integrate_log_gk21(lambda x: x, torch.tensor(1.), **values)


if __name__ == "__main__":
    unittest.main()
