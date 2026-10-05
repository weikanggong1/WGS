"""QAGS analytic oracles, device ownership and SKAT-O failure compatibility."""
import math
import unittest
from unittest.mock import patch

import torch
from scipy.integrate import quad

from torchwgs import statistics
from torchwgs._quadrature import _EpsilonTable, integrate_log_qags
from torchwgs.masks import (Annotation, GeneConfig, MaskDefinition, _vc_tensor_guard,
                            build_gene_masks)


class QagsCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def upper(self):
        return torch.tensor(1., dtype=torch.float64)

    def test_slow_endpoint_convergence_matches_closed_form_and_scipy(self):
        for power in (.5, .9):
            with self.subTest(power=power):
                value, info = integrate_log_qags(lambda x: -power*x.log(), self.upper())
                oracle = quad(lambda x: x**-power, 0., 1., epsabs=1e-25,
                              epsrel=2**-13, limit=1000, full_output=True)
                self.assertTrue(info['converged'], info)
                self.assertEqual(info['ier'], 0)
                self.assertTrue(info['used_extrapolation'])
                self.assertAlmostEqual(float(value.exp()), 1/(1-power), delta=2e-8)
                self.assertAlmostEqual(float(value.exp()), oracle[0], delta=2e-8)
                self.assertEqual(info['intervals'], oracle[2]['last'])
                self.assertLess(info['intervals'], 20)

    def test_both_endpoint_singularities_and_interior_kink(self):
        value, info = integrate_log_qags(
            lambda x: -.5*(x.log()+torch.log1p(-x)), self.upper())
        self.assertTrue(info['converged'], info)
        self.assertAlmostEqual(float(value.exp()), math.pi, delta=1e-7)
        kink = .37
        value, info = integrate_log_qags(lambda x: (1+(x-kink).abs()).log(),
                                        self.upper(), epsrel=1e-8)
        expected = 1+(kink*kink+(1-kink)**2)/2
        self.assertTrue(info['converged'], info)
        self.assertAlmostEqual(float(value.exp()), expected, delta=1e-8)

    def test_device_float64_callback_never_uses_numpy_or_cpu_copy(self):
        calls = []
        def callback(x):
            self.assertEqual(x.dtype, torch.float64)
            self.assertEqual(x.device.type, 'cpu')
            self.assertTrue(bool(((x > 0) & (x < 1)).all()))
            calls.append(x.numel())
            return -.5*x.log()-x
        with patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy conversion')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU conversion')):
            value, info = integrate_log_qags(callback, self.upper())
        self.assertTrue(info['converged'], info)
        self.assertEqual(value.device.type, 'cpu')
        self.assertEqual(value.dtype, torch.float64)
        self.assertEqual(sum(calls), info['evaluations'])
        self.assertAlmostEqual(float(value.exp()), math.sqrt(math.pi)*math.erf(1), delta=2e-8)

    def test_subdivision_transfers_only_changed_error_metadata(self):
        original_tolist = torch.Tensor.tolist
        transfers = []
        def transfer(value):
            transfers.append((value.dtype, tuple(value.shape)))
            return original_tolist(value)
        with patch.object(torch.Tensor, 'tolist', transfer):
            value, info = integrate_log_qags(lambda x: -.9*x.log(), self.upper())
        self.assertTrue(info['converged'], info)
        self.assertAlmostEqual(float(value.exp()), 10., delta=2e-8)
        # One constant-size flag/error packet for each subdivision, followed
        # by the final two diagnostic scalars. No full interval-error table.
        self.assertEqual(transfers[:-1], [(torch.float64, (5,))]*(info['intervals']-1))
        self.assertEqual(transfers[-1], (torch.float64, (2,)))

    def test_fixed_scale_preserves_below_range_probability_and_absolute_budget(self):
        for shift in (0., -1000.):
            with self.subTest(shift=shift):
                value, info = integrate_log_qags(lambda x: shift-.5*x.log(), self.upper(),
                                                epsabs=0., epsrel=1e-8)
                self.assertTrue(info['converged'], info)
                self.assertAlmostEqual(float(value), shift+math.log(2), delta=2e-10)
        self.assertEqual(float(value.exp()), 0.)
        value, info = integrate_log_qags(lambda x: -1000-.5*x.log(), self.upper())
        self.assertTrue(info['converged'], info)
        self.assertEqual(float(value.exp()), 0.)
        self.assertLessEqual(info['intervals'], 2)

    def test_nonfinite_zero_budget_and_roundoff_status(self):
        for invalid in (math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ArithmeticError, 'Nonfinite quadrature integrand'):
                integrate_log_qags(lambda x: torch.full_like(x, invalid), self.upper())
        value, info = integrate_log_qags(lambda x: torch.full_like(x, -math.inf), self.upper())
        self.assertTrue(info['converged'], info)
        self.assertTrue(bool(torch.isneginf(value)))
        _, info = integrate_log_qags(lambda x: -.5*x.log(), self.upper(),
                                    max_intervals=2, epsrel=1e-12)
        self.assertFalse(info['converged'])
        self.assertEqual(info['ier'], 1)
        self.assertEqual(info['evaluations'], 63)
        _, info = integrate_log_qags(lambda x: torch.zeros_like(x), self.upper(),
                                    epsabs=0., epsrel=1e-16)
        self.assertFalse(info['converged'])
        self.assertEqual(info['ier'], 2)
        self.assertEqual(info['status'], 'roundoff')

    def test_empty_range_and_parameter_validation(self):
        value, info = integrate_log_qags(lambda x: x, self.upper()*0)
        self.assertTrue(bool(torch.isneginf(value)))
        self.assertTrue(info['converged'])
        self.assertEqual(info['evaluations'], 0)
        for parameters in ({'epsabs': -1.}, {'epsrel': 0.}, {'epsrel': math.nan},
                           {'max_intervals': 0}, {'max_intervals': True}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                integrate_log_qags(lambda x: x, self.upper(), **parameters)
        for upper in (torch.tensor(-1.), torch.tensor(math.nan), torch.tensor([1., 2.])):
            with self.subTest(upper=upper), self.assertRaises(ValueError):
                integrate_log_qags(lambda x: x, upper)

    def test_epsilon_uses_ordinary_areas_and_handles_coincident_diagonals(self):
        table = _EpsilonTable(self.upper()*2, self.upper()*1.5)
        estimate, error = table.append(self.upper()*1.25)
        self.assertAlmostEqual(float(estimate), 1., delta=1e-14)
        for index in range(3, 70):
            estimate, error = table.append(self.upper()*(1+2.**-index))
            self.assertLessEqual(table.size, 49)
            self.assertTrue(bool(torch.isfinite(estimate)))
            self.assertGreaterEqual(float(error), 0.)
        same = _EpsilonTable(self.upper(), self.upper())
        estimate, error = same.append(self.upper())
        self.assertEqual(float(estimate), 1.)
        self.assertGreaterEqual(float(error), 5*torch.finfo(torch.float64).eps)

    def test_legacy_backend_and_all_non_skato_kernel_results_are_preserved(self):
        self.assertEqual(GeneConfig().skato_integral_backend, 'adaptive_x')
        self.assertEqual(GeneConfig(skato_integral_backend='qags_x').skato_integral_backend, 'qags_x')
        score = torch.tensor([.4, .8, 1.], dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)
        with patch.object(statistics, '_fourier_tail_scalar', side_effect=AssertionError('CPU tail fallback')):
            old = statistics.skato_logp(score, covariance, native_validity=True,
                                        integral_backend='adaptive_x')
            new = statistics.skato_logp(score, covariance, native_validity=True,
                                        integral_backend='qags_x', integral_max_intervals=1)
        self.assertFalse(new['integral_diagnostics']['converged'])
        self.assertIn(new['integral_fallback'], ('bonferroni', 'unavailable'))
        self.assertTrue(torch.equal(old['rho_log10ps'], new['rho_log10ps']))
        for name in ('SKAT', 'BURDEN', 'SKATO-ACAT', 'rhos'):
            self.assertTrue(torch.equal(old[name], new[name]), name)

    def test_nonzero_integrator_error_keeps_bonferroni_and_unavailable_semantics(self):
        covariance = torch.eye(3, dtype=torch.float64)
        for magnitude in (.02, 3.):
            score = torch.tensor([.4, .8, 1.], dtype=torch.float64)*magnitude
            for failure in ('nonconvergence', 'integrand_failure'):
                with self.subTest(magnitude=magnitude, failure=failure):
                    if failure == 'integrand_failure':
                        arguments = {'side_effect': ArithmeticError('Synthetic invalid integrand')}
                    else:
                        arguments = {'return_value': (self.upper()*0, {
                            'converged': False, 'status': 'roundoff', 'ier': 2,
                            'evaluations': 21, 'intervals': 1})}
                    with patch('torchwgs._quadrature.integrate_log_qags', **arguments) as integrator:
                        result = statistics.skato_logp(score, covariance, native_validity=True,
                                                      integral_backend='qags_x')
                    integrator.assert_called_once()
                    self.assertFalse(result['integral_diagnostics']['converged'])
                    bonferroni = result['rho_log10ps'].max()-math.log10(result['rhos'].numel())
                    if float(bonferroni) >= 0:
                        self.assertTrue(torch.equal(result['SKATO'], bonferroni))
                        self.assertEqual(result['integral_fallback'], 'bonferroni')
                    else:
                        self.assertIsNone(result['SKATO'])
                        self.assertEqual(result['integral_fallback'], 'unavailable')


class PreparedMaskReuseProvenanceTests(unittest.TestCase):
    def masks(self, *, sparse=False, distinct_rare=False):
        identifiers = ('ordinary', 'rare_a', 'rare_b')
        matrix = torch.zeros((32, 3), dtype=torch.float64)
        matrix[:5, 0] = 1.
        matrix[7, 1:] = 1.
        annotations = [Annotation('ordinary', 'FixtureGene', category) for category in ('A', 'B')]
        if distinct_rare:
            annotations += [Annotation('rare_a', 'FixtureGene', 'A'),
                            Annotation('rare_b', 'FixtureGene', 'B')]
        else:
            annotations += [Annotation(identifier, 'FixtureGene', category)
                            for identifier in identifiers[1:] for category in ('A', 'B')]
        config = GeneConfig(aaf_bins=(.5,), vc_max_aaf=.5, collapse_mac=2.,
                            include_singletons=False, include_domains=False,
                            variant_block_size=1, vc_storage='sparse' if sparse else 'dense',
                            vc_score_method='crossproduct')
        definitions = [MaskDefinition('First', frozenset(('A',))),
                       MaskDefinition('Second', frozenset(('B',)))]
        return build_gene_masks(matrix, identifiers, annotations, definitions, config)

    def test_equal_members_have_equal_keys_but_distinct_builders_never_share(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                masks = self.masks(sparse=sparse)
                self.assertEqual(len(masks), 2)
                first, second = masks
                self.assertEqual(first.vc_reuse_key, second.vc_reuse_key)
                self.assertEqual(first.burden_reuse_key, second.burden_reuse_key)
                self.assertNotEqual(first.vc_reuse_key, self.masks(sparse=sparse)[0].vc_reuse_key)
                self.assertNotEqual(first.burden_reuse_key, self.masks(sparse=sparse)[0].burden_reuse_key)
                for mask in masks:
                    self.assertEqual(mask.vc_reuse_tensor_guard, _vc_tensor_guard(mask.vc_genotypes))
                    self.assertEqual(mask.burden_reuse_tensor_guard, _vc_tensor_guard(mask.burden))
                # Weights are deliberately excluded: only unweighted products
                # may be shared. A caller changing beta weights must recompute
                # the statistical kernel even with identical genotype keys.
                second.beta_b = 7.
                self.assertEqual(first.vc_reuse_key, second.vc_reuse_key)
                self.assertFalse(torch.equal(first.vc_weights, second.vc_weights))

    def test_distinct_rare_members_cannot_alias_even_when_collapse_values_match(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                first, second = self.masks(sparse=sparse, distinct_rare=True)
                self.assertNotEqual(first.vc_reuse_key, second.vc_reuse_key)
                self.assertNotEqual(first.burden_reuse_key, second.burden_reuse_key)
                a = first.vc_genotypes.to_dense() if sparse else first.vc_genotypes
                b = second.vc_genotypes.to_dense() if sparse else second.vc_genotypes
                self.assertTrue(torch.equal(a, b))

    def test_replacement_dense_inplace_and_sparse_component_mutation_invalidate_guards(self):
        mask = self.masks()[0]
        expected = mask.vc_reuse_tensor_guard
        replacement = mask.vc_genotypes.clone()
        self.assertNotEqual(expected, _vc_tensor_guard(replacement))
        mask.vc_genotypes.add_(.25)
        self.assertNotEqual(expected, _vc_tensor_guard(mask.vc_genotypes))
        mask.burden.mul_(2.)
        self.assertNotEqual(mask.burden_reuse_tensor_guard, _vc_tensor_guard(mask.burden))
        for component in ('values', 'indices'):
            with self.subTest(component=component):
                sparse = self.masks(sparse=True)[0]
                bound = sparse.vc_reuse_tensor_guard
                if component == 'values':
                    sparse.vc_genotypes.values().add_(.25)
                else:
                    sparse.vc_genotypes.indices()[0, 0].add_(1)
                self.assertNotEqual(bound, _vc_tensor_guard(sparse.vc_genotypes))

    def test_inference_and_uncoalesced_tensors_are_not_trusted_for_reuse(self):
        with torch.inference_mode():
            dense = torch.ones((2, 2))
            sparse = dense.to_sparse_coo().coalesce()
        self.assertIsNone(_vc_tensor_guard(dense))
        self.assertIsNone(_vc_tensor_guard(sparse))
        uncoalesced = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 0]]),
                                            torch.tensor([1., 2.]), (2, 2))
        self.assertIsNone(_vc_tensor_guard(uncoalesced))
        self.assertIsNone(_vc_tensor_guard(None))

    def test_cache_bytes_validate_and_zero_disables_without_changing_masks(self):
        self.assertEqual(GeneConfig().product_cache_bytes, 256*1024**2)
        self.assertEqual(GeneConfig(product_cache_bytes=0).product_cache_bytes, 0)
        for invalid in (-1, True, 1.25, '1024'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'product_cache_bytes'):
                GeneConfig(product_cache_bytes=invalid)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for device checks')
class QagsCudaCompatibilityTests(unittest.TestCase):
    def test_integrand_epsilon_and_result_remain_on_cuda(self):
        device = torch.device('cuda:0')
        upper = torch.tensor(1., device=device, dtype=torch.float64)
        calls = []
        def callback(x):
            self.assertEqual(x.device, device)
            self.assertEqual(x.dtype, torch.float64)
            calls.append(x.numel())
            return -.5*x.log()-x
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU conversion')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy conversion')):
            value, info = integrate_log_qags(callback, upper)
            table = _EpsilonTable(upper*2, upper*1.5)
            estimate, error = table.append(upper*1.25)
        self.assertTrue(info['converged'], info)
        self.assertEqual(sum(calls), info['evaluations'])
        for result in (value, estimate, error, table.values, table.history):
            self.assertEqual(result.device, device)
            self.assertEqual(result.dtype, torch.float64)
        self.assertAlmostEqual(float(value.exp()), math.sqrt(math.pi)*math.erf(1), delta=2e-8)
        self.assertAlmostEqual(float(estimate), 1., delta=1e-14)


if __name__ == '__main__':
    unittest.main()
