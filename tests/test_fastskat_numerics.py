"""Numerical unit checks only; these are not a real WGS benchmark."""

from __future__ import annotations

import unittest

import torch

from staar_phewas._fastskat_numerics import (FastSKATNumericsError,
                              refine_spectrum_and_moments,
                              scaled_residual_cumulants)


class RefinementTests(unittest.TestCase):
    def test_matched_ritz_vectors_and_blocked_fp64_moments(self):
        matrix = torch.diag(torch.tensor([8., 3., 1., .5], dtype=torch.float32))
        basis = torch.tensor([[1., -1.], [1., 1.], [0., 0.], [0., 0.]])
        result = refine_spectrum_and_moments(matrix, basis, block_rows=1)
        torch.testing.assert_close(result["top"], torch.tensor([8., 3.], dtype=torch.float64))
        torch.testing.assert_close(result["residual_mean"], torch.tensor(1.5, dtype=torch.float64))
        torch.testing.assert_close(result["residual_second"], torch.tensor(1.25, dtype=torch.float64))
        self.assertLess(float(result["residual_norm"]), 1e-14)
        self.assertLess(float(result["orthogonality"]), 1e-14)
        self.assertEqual(result["metadata"]["blocked_matmul_calls"], 12)
        self.assertFalse(result["metadata"]["full_dense_fp64_materialized"])

    def test_severe_cancellation_uses_direct_deflation(self):
        matrix = torch.diag(torch.tensor([8., 3., 1e-9, 1e-10], dtype=torch.float32))
        basis = torch.eye(4, dtype=torch.float32)[:, :2]
        result = refine_spectrum_and_moments(matrix, basis, block_rows=2)
        expected = matrix[2:, 2:].double()
        torch.testing.assert_close(result["residual_mean"], expected.trace(), rtol=1e-14, atol=0)
        torch.testing.assert_close(result["residual_second"], expected.square().sum(), rtol=1e-14, atol=0)
        self.assertTrue(result["metadata"]["direct_deflation_used"])
        self.assertEqual(result["metadata"]["deflation_blocks"], 2)

    def test_exact_zero_remainder_removes_both_moments(self):
        matrix = torch.diag(torch.tensor([8., 3., 0., 0.]))
        result = refine_spectrum_and_moments(matrix, torch.eye(4)[:, :2])
        self.assertEqual(float(result["residual_mean"]), 0)
        self.assertEqual(float(result["residual_second"]), 0)

    def test_small_actual_negative_residual_is_not_clamped(self):
        matrix = torch.diag(torch.tensor([8., 3., -1e-9, 0.]))
        with self.assertRaisesRegex(FastSKATNumericsError, "negative residual moment") as caught:
            refine_spectrum_and_moments(matrix, torch.eye(4)[:, :2])
        self.assertTrue(caught.exception.diagnostics["direct_deflation_used"])
        self.assertFalse(caught.exception.diagnostics["negative_residual_clipped"])

    def test_material_negative_top_is_rejected(self):
        matrix = torch.diag(torch.tensor([8., -.1, 0., 0.]))
        with self.assertRaisesRegex(FastSKATNumericsError, "spectrum is materially negative"):
            refine_spectrum_and_moments(matrix, torch.eye(4)[:, :2])

    def test_frobenius_square_occurs_after_float64_conversion(self):
        diagonal = torch.tensor([11.1251, 9.8179, 2.0113, .8812], dtype=torch.float32)
        matrix = torch.diag(diagonal)
        result = refine_spectrum_and_moments(matrix, torch.eye(4)[:, :2], block_rows=1)
        expected = diagonal.double().square().sum()
        self.assertEqual(float(result["trace2"]), float(expected))
        self.assertNotEqual(float(expected), float(diagonal.square().sum(dtype=torch.float64)))

    def test_dense_rotated_covariance_matches_full_fp64_spectrum(self):
        vector = torch.arange(1, 7, dtype=torch.float64)
        vector = vector / torch.linalg.vector_norm(vector)
        rotation = torch.eye(6, dtype=torch.float64) - 2 * vector[:, None] * vector[None, :]
        eigenvalues = torch.tensor([100., 30., 1., .5, .2, .1], dtype=torch.float64)
        matrix = ((rotation * eigenvalues) @ rotation.T).float()
        matrix = (matrix + matrix.T) * .5
        result = refine_spectrum_and_moments(matrix, rotation[:, :2].float(), block_rows=2)
        dense = torch.linalg.eigvalsh(matrix.double()).flip(0)
        torch.testing.assert_close(result["top"], dense[:2], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(result["residual_mean"], dense[2:].sum(), rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(result["residual_second"], dense[2:].square().sum(), rtol=1e-9, atol=1e-11)


class ResidualCumulantTests(unittest.TestCase):
    def test_matches_continuous_chi_square_form(self):
        mu, second, root = [torch.tensor(v, dtype=torch.float64) for v in (3., .6, .1)]
        scale, dof = second / mu, mu.square() / second
        cumulant, first, curvature = scaled_residual_cumulants(root, mu, second, scale)
        torch.testing.assert_close(cumulant, -.5 * dof * torch.log1p(-2 * scale * root))
        torch.testing.assert_close(first, dof * scale / (1 - 2 * scale * root))
        torch.testing.assert_close(curvature, 2 * dof * scale.square() / (1 - 2 * scale * root).square())

    def test_tiny_component_scale_retains_its_mean(self):
        mu, second, root = [torch.tensor(v, dtype=torch.float64) for v in (3., 3e-30, .1)]
        cumulant, first, curvature = scaled_residual_cumulants(root, mu, second, second / mu)
        torch.testing.assert_close(cumulant, mu * root, rtol=1e-14, atol=0)
        torch.testing.assert_close(first, mu, rtol=1e-14, atol=0)
        torch.testing.assert_close(curvature, 2 * second, rtol=1e-14, atol=0)

    def test_zero_root_uses_analytic_limit(self):
        root = torch.tensor(0., dtype=torch.float64)
        mu = torch.tensor(3., dtype=torch.float64)
        second = torch.tensor(.6, dtype=torch.float64)
        cumulant, first, curvature = scaled_residual_cumulants(root, mu, second, second / mu)
        self.assertEqual(float(cumulant), 0)
        self.assertEqual(float(first), 3)
        self.assertEqual(float(curvature), 1.2)


if __name__ == "__main__":
    unittest.main()
