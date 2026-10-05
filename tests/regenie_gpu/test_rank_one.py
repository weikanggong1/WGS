"""Mathematical eigenvalue tests; these are not real-data benchmarks."""
import unittest

import torch

from torchwgs._rank_one import rank_one_eigvalsh


class RankOneTests(unittest.TestCase):
    def fixtures(self, device):
        generator = torch.Generator().manual_seed(934)
        d = torch.rand(64, generator=generator, dtype=torch.float64)
        z = torch.randn(64, generator=generator, dtype=torch.float64) / 8
        yield d.to(device), z.to(device), .5
        yield d.to(device), z.to(device), -.5
        d = torch.tensor([0., 0., 0., 1., 1., 2., 2., 2.], dtype=torch.float64)
        z = torch.tensor([0., 1., 2., 0., 0., 3., 4., 0.], dtype=torch.float64)
        yield d.to(device), z.to(device), 1.
        yield d.to(device), z.to(device), -1.
        eps = torch.finfo(torch.float64).eps
        d = torch.tensor([0., 1e-300, 1e-100, 1e-30, 1e-14, 1.,
                          1.+eps, 1.+2*eps, 2.], dtype=torch.float64)
        z = torch.tensor([1e-150, 1e-150, 1e-50, 1e-15, 1e-7,
                          1., 1e-15, 0., .5], dtype=torch.float64)
        yield d.to(device), z.to(device), .5
        yield d.to(device), z.to(device), -.5
        for exponent in (-150, 150):
            scale = 10. ** exponent
            d = torch.linspace(.01, 1., 64, dtype=torch.float64) * scale
            z = torch.randn(64, generator=generator, dtype=torch.float64) * (scale/64)**.5
            yield d.to(device), z.to(device), .9
            yield d.to(device), z.to(device), -.9
        d = torch.logspace(-12, 3, 128, dtype=torch.float64)
        a = torch.randn(128, generator=generator, dtype=torch.float64)
        a /= a.norm()
        yield d.to(device), (d.sqrt()*a).to(device), -1.

    def compare_fixtures(self, device):
        for d, z, coefficient in self.fixtures(device):
            expected = rank_one_eigvalsh(d, z, coefficient, backend='dense')
            for chunk in (7, 256):
                result = rank_one_eigvalsh(d, z, coefficient, backend='secular',
                                           root_chunk=chunk, return_bounds=True)
                norm = max(float(d.abs().max()) + abs(coefficient)*float(z.square().sum()), 1e-300)
                self.assertLess(float((result.values-expected).abs().max()) / norm, 5e-12)
                self.assertTrue(bool((result.lower <= result.values).all()))
                self.assertTrue(bool((result.values <= result.upper).all()))
                self.assertLess(float((result.upper-result.lower).max()) / norm, 1e-13)
                self.assertEqual(result.backend, 'secular')

    def test_cpu_spectra_repeated_singular_scaled_and_downdated(self):
        self.compare_fixtures('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA needed for device math')
    def test_cuda_spectra_repeated_singular_scaled_and_downdated(self):
        self.compare_fixtures('cuda')

    def test_zero_update_and_exact_deflation(self):
        d = torch.tensor([2., 1., 1., 0., 0.], dtype=torch.float64)
        z = torch.zeros_like(d)
        for backend in ('dense', 'secular'):
            self.assertTrue(torch.equal(rank_one_eigvalsh(d, z, backend=backend), d.sort().values))
            self.assertTrue(torch.equal(rank_one_eigvalsh(d, z+1, 0., backend=backend), d.sort().values))
        z = torch.tensor([0., 0., 0., 1., 2.], dtype=torch.float64)
        result = rank_one_eigvalsh(d, z, backend='secular')
        self.assertTrue(torch.equal(result, torch.tensor([0., 1., 1., 2., 5.], dtype=torch.float64)))
        self.assertEqual(rank_one_eigvalsh(d[:0], z[:0], backend='secular').numel(), 0)

    def test_auto_requires_an_explicit_validated_threshold(self):
        d = torch.tensor([0., 1.], dtype=torch.float64)
        z = torch.tensor([.1, .2], dtype=torch.float64)
        default = rank_one_eigvalsh(d, z, backend='auto', return_bounds=True)
        self.assertEqual(default.backend, 'dense')
        self.assertIsNone(default.lower)
        enabled = rank_one_eigvalsh(d, z, backend='auto', auto_min_size=2, return_bounds=True)
        self.assertEqual(enabled.backend, 'secular')

    def test_lower_iteration_budget_exposes_root_brackets(self):
        d = torch.tensor([0., 1., 3.], dtype=torch.float64)
        z = torch.tensor([.5, .7, .2], dtype=torch.float64)
        expected = rank_one_eigvalsh(d, z)
        for coefficient in (1., -1.):
            expected = rank_one_eigvalsh(d, z, coefficient)
            result = rank_one_eigvalsh(d, z, coefficient, backend='secular',
                                       max_iterations=4, return_bounds=True)
            self.assertTrue(bool((result.lower <= expected+1e-14).all()))
            self.assertTrue(bool((expected-1e-14 <= result.upper).all()))
            self.assertGreater(float((result.upper-result.lower).max()), 1e-3)

    def test_input_validation(self):
        d = torch.tensor([0., 1.], dtype=torch.float64)
        z = torch.ones_like(d)
        invalid = [dict(backend='unknown'), dict(root_chunk=0), dict(root_chunk=True),
                   dict(max_iterations=0), dict(auto_min_size=0), dict(coefficient=float('nan'))]
        for options in invalid:
            with self.assertRaises(ValueError): rank_one_eigvalsh(d, z, **options)
        with self.assertRaises(ValueError): rank_one_eigvalsh(d.float(), z.float())
        with self.assertRaises(ValueError): rank_one_eigvalsh(d[:,None], z[:,None])
        with self.assertRaises(ValueError): rank_one_eigvalsh(d, z[:1])
        with self.assertRaises(ValueError): rank_one_eigvalsh(d, z*float('inf'))
        with self.assertRaises(TypeError): rank_one_eigvalsh([0., 1.], z)
        huge = z * 1e200
        with self.assertRaises(ArithmeticError): rank_one_eigvalsh(d, huge, backend='secular')


if __name__ == '__main__':
    unittest.main()
