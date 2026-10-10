"""CPU contracts for failure-only retry; actual matrix evidence is separate."""
from types import SimpleNamespace
import unittest

import torch
from fudan_wgs_toolkit.cuda_eigen.small_solver import FP32SmallSpectrumSolver
from fudan_wgs_toolkit.cuda_eigen import _backend as backend
from fudan_wgs_toolkit import _weighted_spectra as weighted
from test_cuda_eigen_backend import API, Flag, Tensor, Torch

class MockLinAlgError(RuntimeError):
    pass

LinAlgError = getattr(getattr(torch, '_C', None), '_LinAlgError', MockLinAlgError)

class RetryTorch(Torch):
    def __init__(self, first_error=None, retry_error=None):
        super().__init__()
        self._C = SimpleNamespace(_LinAlgError=LinAlgError, _has_magma=True)
        self.preference = 'cusolver'
        self.preference_events = []
        self.seen = []
        self.first_error = first_error
        self.retry_error = retry_error
        self.finite = True
        def preference(value=None):
            if value is not None:
                self.preference_events.append(value)
                self.preference = value
            return self.preference
        self.backends = SimpleNamespace(cuda=SimpleNamespace(preferred_linalg_library=preference))
        def eig(matrix, **kwargs):
            self.original_calls += 1
            self.seen.append((matrix, dict(kwargs), self.preference))
            if self.original_calls == 1 and self.first_error is not None:
                raise self.first_error
            if self.original_calls == 2 and self.retry_error is not None:
                raise self.retry_error
            return Tensor(matrix.shape[:-1], dtype=matrix.dtype)
        self.linalg = SimpleNamespace(eigvalsh=eig)
    def isfinite(self, matrix):
        return Flag(self.finite)

class MagmaRetryContracts(unittest.TestCase):
    def solver(self, module):
        return FP32SmallSpectrumSolver(torch_module=module, memory_limit=40*2**30,
            _backend_factory=lambda **kwargs: self.fail('outside path loaded Jacobi backend'))

    def test_normal_torch_route_has_one_call_and_no_preference_change(self):
        t=RetryTorch(); s=self.solver(t); a=Tensor((2,3,3))
        out=s.eigvalsh(a)
        self.assertEqual(out.shape,(2,3))
        self.assertEqual(t.original_calls,1)
        self.assertEqual(t.preference_events,[])
        self.assertEqual(s.last_execution_backend,'torch.linalg.eigvalsh')
        self.assertEqual(s.report()['magma_retry_attempts'],0)
        self.assertEqual(s.memory_limit,40*2**30)

    def test_actual_linalg_type_fp32_retry_keeps_input_upper_triangle_and_restores(self):
        t=RetryTorch(LinAlgError('test convergence error')); s=self.solver(t)
        a=Tensor((2,513,513)); original_values=a.values[:]
        out=s.eigvalsh(a,UPLO='U')
        self.assertEqual(out.dtype,t.float32)
        self.assertEqual(out.shape,(2,513))
        self.assertEqual(a.values,original_values)
        self.assertTrue(all(item[0] is a for item in t.seen))
        self.assertEqual([item[1] for item in t.seen],[{'UPLO':'U'},{'UPLO':'U'}])
        self.assertEqual(t.preference_events,['magma','cusolver'])
        self.assertEqual(t.preference,'cusolver')
        r=s.report()
        self.assertEqual(r['magma_retry_attempts'],1)
        self.assertEqual(r['magma_retry_successes'],1)
        self.assertEqual(r['magma_retry_failures'],0)
        self.assertEqual(r['magma_retry_matrices'],2)
        self.assertEqual(r['magma_retry_max_n'],513)
        self.assertEqual(r['magma_retry_shapes'],[[2,513,513]])
        self.assertEqual(r['last_execution_backend'],'torch.linalg.eigvalsh:magma_retry')
        self.assertTrue(r['preference_restored'])
        self.assertTrue(r['preference_modified'])
        self.assertFalse(r['retry_scaling'] or r['retry_regularization'] or r['retry_fp64'])

    def test_generic_runtime_oom_io_and_baseexception_never_retry(self):
        for error in (RuntimeError('not a convergence error'),MemoryError('oom'),
                      OSError('io'),KeyboardInterrupt()):
            with self.subTest(type=type(error).__name__):
                t=RetryTorch(error);s=self.solver(t)
                with self.assertRaises(type(error)) as found:s.eigvalsh(Tensor((3,3)))
                self.assertIs(found.exception,error)
                self.assertEqual(t.original_calls,1)
                self.assertEqual(t.preference_events,[])
                self.assertEqual(s.report()['magma_retry_attempts'],0)

    def test_cpu_fp64_lower_grad_or_layout_error_propagates_unchanged(self):
        for invalid in ('cpu','fp64','lower','grad','layout'):
            with self.subTest(invalid=invalid):
                original=LinAlgError('original error');t=RetryTorch(original);s=self.solver(t)
                a=Tensor((3,3));kwargs={}
                if invalid=='cpu':a.is_cuda=False
                elif invalid=='fp64':a.dtype='float64'
                elif invalid=='lower':kwargs['UPLO']='L'
                elif invalid=='grad':a.requires_grad=True
                elif invalid=='layout':a.layout='sparse_coo'
                with self.assertRaises(LinAlgError) as found:s.eigvalsh(a,**kwargs)
                self.assertIs(found.exception,original)
                self.assertEqual(t.original_calls,1)
                self.assertEqual(t.preference_events,[])

    def test_unavailable_magma_or_preference_api_preserves_original_error(self):
        for invalid in ('magma','api'):
            with self.subTest(invalid=invalid):
                original=LinAlgError('original error');t=RetryTorch(original);s=self.solver(t)
                if invalid=='magma':t._C._has_magma=False
                else:t.backends.cuda.preferred_linalg_library=None
                with self.assertRaises(LinAlgError) as found:s.eigvalsh(Tensor((3,3)))
                self.assertIs(found.exception,original)
                self.assertEqual(t.original_calls,1)
                self.assertEqual(t.preference_events,[])

    def test_nonfinite_input_cannot_retry_or_replace_original_error(self):
        original=LinAlgError('original error');t=RetryTorch(original);s=self.solver(t)
        t.finite=False
        with self.assertRaises(LinAlgError) as found:s.eigvalsh(Tensor((3,3)))
        self.assertIs(found.exception,original)
        self.assertEqual(t.preference_events,[])
        self.assertEqual(s.report()['magma_nonfinite_rejections'],1)
        self.assertEqual(s.report()['magma_retry_attempts'],0)

    def test_failed_retry_restores_preference_and_propagates_retry_exception(self):
        for retry_error in (LinAlgError('magma failed'),RuntimeError('magma runtime'),
                            MemoryError('magma oom'),KeyboardInterrupt()):
            with self.subTest(type=type(retry_error).__name__):
                t=RetryTorch(LinAlgError('original failed'),retry_error);s=self.solver(t)
                with self.assertRaises(type(retry_error)) as found:s.eigvalsh(Tensor((3,3)))
                self.assertIs(found.exception,retry_error)
                self.assertEqual(t.preference_events,['magma','cusolver'])
                self.assertEqual(t.preference,'cusolver')
                self.assertEqual(s.report()['magma_retry_failures'],1)
                self.assertEqual(s.report()['magma_retry_successes'],0)
                self.assertTrue(s.report()['preference_restored'])
                self.assertIsNone(s.last_execution_backend)

    def test_selected_33_to_512_path_keeps_owned_solver_and_never_changes_preference(self):
        t=RetryTorch(LinAlgError('must not reach original'));api=API()
        s=FP32SmallSpectrumSolver(torch_module=t,memory_limit=40*2**30,
            _backend_factory=lambda **kwargs:backend.BatchedEigenBackend(api=api,**kwargs))
        out=s.eigvalsh(Tensor((2,33,33)))
        self.assertEqual(out.shape,(2,33))
        self.assertEqual(t.original_calls,0)
        self.assertEqual(t.preference_events,[])
        self.assertEqual(s.report()['magma_retry_attempts'],0)
        self.assertEqual(s.last_execution_backend,'cusolverDnSsyevjBatched')
        s.close()

    def test_selected_failure_does_not_enter_magma_recovery(self):
        t=RetryTorch();api=API();api.info=4
        s=FP32SmallSpectrumSolver(torch_module=t,memory_limit=40*2**30,
            _backend_factory=lambda **kwargs:backend.BatchedEigenBackend(api=api,**kwargs))
        with self.assertRaises(RuntimeError):s.eigvalsh(Tensor((1,33,33)))
        self.assertEqual(t.original_calls,0)
        self.assertEqual(t.preference_events,[])
        self.assertEqual(s.report()['magma_retry_attempts'],0)
        s.close()

    def test_context_metadata_labels_retry_as_magma_instead_of_custom_cusolver(self):
        from unittest.mock import patch
        t=RetryTorch(LinAlgError('original failed'));s=self.solver(t)
        with patch.object(weighted,'_make_solver',lambda _:s):
            with weighted.eigensolver_context('cusolver_batched',memory_limit=40*2**30) as state:
                out,route=weighted._complete_eigvalsh(Tensor((2,3,3)))
                self.assertEqual(route,'torch.linalg.eigvalsh:magma_retry')
                self.assertEqual(out.shape,(2,3))
        r=weighted.execution_metadata()
        self.assertEqual(r['actual_backend_calls'],{'torch.linalg.eigvalsh:magma_retry':1})
        self.assertEqual(r['actual_backend_matrices'],{'torch.linalg.eigvalsh:magma_retry':2})
        self.assertEqual(state['selector']['magma_retry_successes'],1)
        self.assertTrue(state['selector']['preference_restored'])
        self.assertIsNone(state['selector']['backend'])

if __name__=='__main__':unittest.main()
