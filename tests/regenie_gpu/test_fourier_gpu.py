import math
import unittest
import torch

from torchwgs._fourier_gpu import available, fourier_rules
from torchwgs import statistics


def reference(spectrum, points, terms, spacing, gaussian, extra, auxiliary, maximum):
    offset = torch.arange(maximum+1, device=spectrum.device, dtype=torch.float64)
    frequency = (offset[None, :]+.5)*spacing[:, None]
    arguments = 2*frequency[:, :, None]*spectrum
    angles = arguments.atan()
    phase = -points[:, None]*frequency+angles.sum(-1)/2
    log_magnitude = -gaussian[:, None]*frequency.square()/2
    log_magnitude -= arguments.square().log1p().sum(-1)/4
    magnitude = torch.where((offset <= terms[:, None]) & (log_magnitude >= -50),
                            log_magnitude.exp(), 0.)
    amplitude = (spacing[:, None]/math.pi)*magnitude/frequency
    damping = extra[:, None]*frequency.square()/2
    factor = torch.where(damping > 50, 1., -torch.expm1(-damping))
    amplitude *= torch.where(auxiliary[:, None], factor, 1.)
    return ((phase.sin()*amplitude).sum(-1),
            ((points[:, None]*frequency+angles.abs().sum(-1)/2)*amplitude).sum(-1))


@unittest.skipUnless(available('cuda') and torch.cuda.is_available(), 'NVIDIA CUDA/Triton required')
class FourierGpu(unittest.TestCase):
    def test_variable_rule_spectra_auxiliary_and_gaussian_match_torch(self):
        device = 'cuda'
        for rank in (1, 4, 17, 129, 1025):
            with self.subTest(rank=rank):
                spectrum = torch.logspace(-3, 0, rank, dtype=torch.float64, device=device)
                points = torch.tensor([.1, 1., 3., 10., 25.], device=device, dtype=torch.float64)
                terms = torch.tensor([0., 17., 63., 129., 1024.], device=device, dtype=torch.float64)
                spacing = torch.tensor([.5, .25, .01, .1, .015], device=device, dtype=torch.float64)
                gaussian = torch.tensor([0., .02, 0., .1, .001], device=device, dtype=torch.float64)
                extra = torch.tensor([0., .2, 100., 0., .0001], device=device, dtype=torch.float64)
                auxiliary = torch.tensor([False, True, True, False, True], device=device)
                expected = reference(spectrum, points, terms, spacing, gaussian, extra, auxiliary, 1024)
                actual = fourier_rules(spectrum, points, terms, spacing, gaussian, extra, auxiliary,
                                       maximum_terms=1024)
                for a, b in zip(actual, expected):
                    self.assertEqual(a.dtype, torch.float64)
                    torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)

    def test_noncontiguous_vectors_empty_jobs_and_device_precision_guard(self):
        g = torch.arange(1, 9, device='cuda', dtype=torch.float64)[::2]
        points = torch.tensor([.1, 100., 1., 100.], device='cuda', dtype=torch.float64)[::2]
        terms = torch.tensor([65., 2., 17., 2.], device='cuda', dtype=torch.float64)[::2]
        spacing = torch.ones_like(points)*.01
        gaussian = torch.zeros_like(points)
        auxiliary = torch.zeros(2, dtype=torch.bool, device='cuda')
        expected = reference(g, points, terms, spacing, gaussian, gaussian, auxiliary, 65)
        actual = fourier_rules(g, points, terms, spacing, gaussian, gaussian, auxiliary, maximum_terms=65)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
        empty = points[:0]
        self.assertEqual(fourier_rules(g, empty, empty, empty, empty, empty, auxiliary[:0],
                                      maximum_terms=0)[0].numel(), 0)
        with self.assertRaises(ValueError):
            fourier_rules(g.float(), points, terms, spacing, gaussian, gaussian, auxiliary, maximum_terms=65)

    def test_davies_faults_and_tail_hierarchy_match_torch(self):
        for rank in (2, 17, 257):
            spectrum = torch.logspace(-2, 0, rank, device='cuda', dtype=torch.float64)
            points = spectrum.sum()*torch.tensor([.5, 1., 2., 8.], device='cuda', dtype=torch.float64)
            old, old_fault = statistics.davies_logsf(points, spectrum, fourier_backend='torch')
            new, new_fault = statistics.davies_logsf(points, spectrum, fourier_backend='fused')
            self.assertTrue(torch.equal(old_fault, new_fault))
            torch.testing.assert_close(old, new, atol=2e-10, rtol=2e-10, equal_nan=True)
            old = statistics._association_tail(points, spectrum, 'regenie', fourier_backend='torch')
            new = statistics._association_tail(points, spectrum, 'regenie', fourier_backend='fused')
            torch.testing.assert_close(old, new, atol=2e-10, rtol=2e-10, equal_nan=True)

    def test_skato_preserves_kernel_values_and_prepared_backend_guard(self):
        score = torch.tensor([.4, .8, 1.], device='cuda', dtype=torch.float64)
        covariance = torch.eye(3, device='cuda', dtype=torch.float64)
        kwargs = dict(native_validity=True, integral_backend='qags_x')
        old = statistics.skato_logp(score, covariance, davies_fourier_backend='torch', **kwargs)
        new = statistics.skato_logp(score, covariance, davies_fourier_backend='fused', **kwargs)
        for key in ('SKAT', 'BURDEN', 'SKATO', 'SKATO-ACAT', 'rho_log10ps'):
            torch.testing.assert_close(old[key], new[key], atol=2e-10, rtol=2e-10)
        prepared = statistics._prepare_davies_spectrum(covariance.diag(), 'auto', 'fused')
        with self.assertRaises(ValueError):
            statistics.davies_logsf(score, covariance.diag(), _prepared_spectrum=prepared)


if __name__ == '__main__':
    unittest.main()
