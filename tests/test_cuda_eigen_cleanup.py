"""CPU-only lifetime contracts; tensors and CUDA calls are mock objects."""
import threading
import unittest

from fudan_wgs_toolkit.cuda_eigen import FP32SmallSpectrumSolver
from fudan_wgs_toolkit.cuda_eigen import _backend as backend_module
from test_cuda_eigen_backend import API, Torch, Tensor


class DestroyFailure(API):
    def __init__(self, error_type):
        super().__init__()
        self.error_type = error_type

    def destroy(self, handle, params):
        # A real destroy may already have released params before handle destroy
        # returns an error. Retrying either retained pointer would be unsafe.
        super().destroy(handle, params)
        raise self.error_type('mock cleanup error; never include this raw text in report')


class Contracts(unittest.TestCase):
    def backend(self, api=None):
        api = API() if api is None else api
        solver = backend_module.BatchedEigenBackend(torch_module=Torch(), api=api)
        return solver, api

    def assert_closed_once(self, solver, api):
        self.assertTrue(solver.closed)
        self.assertIsNone(solver.handle)
        self.assertIsNone(solver.params)
        count = api.calls
        solver.close()
        self.assertEqual(api.calls, count)
        self.assertEqual(sum(call[0] == 'destroy' for call in api.seen), 1)
        with self.assertRaisesRegex(RuntimeError, 'closed solver backend'):
            solver(Tensor((2, 2)))

    def test_success_close_detaches_and_repeated_close_does_no_work(self):
        solver, api = self.backend()
        solver(Tensor((2, 2)))
        solver.close()
        self.assert_closed_once(solver, api)
        report = solver.report()
        self.assertTrue(report['cleanup_attempted'])
        self.assertTrue(report['cleanup_completed'])
        self.assertFalse(report['cleanup_failed'])
        self.assertIsNone(report['cleanup_error_type'])
        self.assertEqual(report['actual_call_status'], 'selected_calls_succeeded_info0')

    def test_exception_cleanup_failclosed_and_success_status_independent(self):
        solver, api = self.backend(DestroyFailure(RuntimeError))
        solver(Tensor((2, 2)))
        with self.assertRaisesRegex(RuntimeError, 'mock cleanup error'):
            solver.close()
        self.assert_closed_once(solver, api)
        report = solver.report()
        self.assertTrue(report['cleanup_attempted'])
        self.assertFalse(report['cleanup_completed'])
        self.assertTrue(report['cleanup_failed'])
        self.assertEqual(report['cleanup_error_type'], 'RuntimeError')
        self.assertNotIn('mock cleanup error', repr(report))
        self.assertEqual(report['actual_call_status'], 'selected_calls_succeeded_info0')

    def test_baseexception_cleanup_failclosed_and_no_second_destroy(self):
        solver, api = self.backend(DestroyFailure(KeyboardInterrupt))
        solver(Tensor((2, 2)))
        with self.assertRaises(KeyboardInterrupt):
            solver.close()
        self.assert_closed_once(solver, api)
        self.assertTrue(solver.report()['cleanup_failed'])
        self.assertEqual(solver.report()['cleanup_error_type'], 'KeyboardInterrupt')

    def test_selector_context_closes_even_if_destroy_fails_during_body_error(self):
        torch = Torch()
        api = DestroyFailure(RuntimeError)
        def factory(**kwargs):
            return backend_module.BatchedEigenBackend(api=api, **kwargs)
        solver = FP32SmallSpectrumSolver(torch_module=torch, _backend_factory=factory)
        with self.assertRaisesRegex(RuntimeError, 'mock cleanup error') as caught:
            with solver:
                solver.eigvalsh(Tensor((1, 33, 33)))
                raise ValueError('independent body failure')
        self.assertIsInstance(caught.exception.__context__, ValueError)
        self.assertTrue(solver.closed)
        self.assert_closed_once(solver.backend, api)
        solver.close()
        report = solver.report()
        self.assertTrue(report['backend']['cleanup_failed'])
        self.assertFalse(report['backend']['cleanup_completed'])
        self.assertEqual(torch.original_calls, 0)

    def test_unused_close_no_destroy_and_wrong_thread_cannot_close_live_backend(self):
        solver, api = self.backend()
        errors = []
        def other_thread():
            try:
                solver.close()
            except RuntimeError as error:
                errors.append(type(error).__name__)
        thread = threading.Thread(target=other_thread)
        thread.start(); thread.join()
        self.assertEqual(errors, ['RuntimeError'])
        self.assertFalse(solver.closed)
        solver.close(); solver.close()
        self.assertEqual(api.calls, 0)
        self.assertFalse(solver.report()['cleanup_attempted'])
        self.assertTrue(solver.report()['cleanup_completed'])
        self.assertFalse(solver.report()['cleanup_failed'])

    def test_failed_solve_releases_lock_then_cleanup_once(self):
        solver, api = self.backend()
        api.info = 1
        with self.assertRaises(RuntimeError):
            solver(Tensor((2, 2)))
        api.info = 0
        solver(Tensor((2, 2)))
        solver.close()
        self.assert_closed_once(solver, api)
        report = solver.report()
        self.assertEqual(report['failed_calls'], 1)
        self.assertEqual(report['successful_calls'], 1)
        self.assertEqual(report['actual_call_status'], 'selected_calls_failed')
        self.assertTrue(report['cleanup_completed'])


if __name__ == '__main__':
    unittest.main()
