"""Minimal explicit full-spectrum selector for integration into weighted_spectra."""
import threading
from ._backend import BatchedEigenBackend

class FP32SmallSpectrumSolver:
    """Own one serialized solver, selected n33..512; other shapes use original Torch.

    No global Torch functions/preferences/TF32 flags are changed. Backend and
    official CUDA library are loaded lazily at the first selected CUDA call.
    """
    def __init__(self,*,torch_module=None,memory_limit=20*2**30,
                 expected_solver_sha256=None,expected_runtime_sha256=None,
                 _backend_factory=None):
        if torch_module is None:
            import torch as torch_module
        if type(memory_limit) is not int or not 0<memory_limit<=20*2**30:
            raise ValueError('memory_limit must be positive bytes <=20GiB')
        self.torch=torch_module;self.original=torch_module.linalg.eigvalsh
        self.memory_limit=memory_limit;self.expected_solver_sha256=expected_solver_sha256
        self.expected_runtime_sha256=expected_runtime_sha256
        self._factory=BatchedEigenBackend if _backend_factory is None else _backend_factory
        self.backend=None;self.closed=False;self.owner=threading.get_ident()
        self.selected_calls=0;self.outside_calls=0
        self.last_execution_backend=None

    def eigvalsh(self,matrix,*,UPLO='U'):
        if self.closed:raise RuntimeError('Closed FP32 spectrum solver')
        if threading.get_ident()!=self.owner:raise RuntimeError('Solver belongs to its creating thread')
        self.last_execution_backend=None
        shape=tuple(matrix.shape)
        selected=len(shape) in (2,3) and shape[-2]==shape[-1] and 33<=shape[-1]<=512
        if not selected:
            self.outside_calls+=1
            result=self.original(matrix,UPLO=UPLO)
            self.last_execution_backend='torch.linalg.eigvalsh'
            return result
        self.selected_calls+=1
        if matrix.dtype!=self.torch.float32 or not matrix.is_cuda or matrix.layout!=self.torch.strided or matrix.requires_grad:
            raise ValueError('Selected API requires strided CUDA FP32 without autograd')
        if UPLO!='U':raise ValueError('Selected FP32 API requires original UPLO=U')
        if self.backend is None:
            self.backend=self._factory(torch_module=self.torch,max_n=512,memory_limit=self.memory_limit,
                expected_solver_sha256=self.expected_solver_sha256,expected_runtime_sha256=self.expected_runtime_sha256)
        result=self.backend(matrix,UPLO='U')  # selected errors propagate
        self.last_execution_backend='cusolverDnSsyevjBatched'
        return result

    def report(self):
        return dict(selected_range=[33,512],outside_policy='original torch.linalg.eigvalsh',
                    selected_calls=self.selected_calls,outside_calls=self.outside_calls,
                    last_execution_backend=self.last_execution_backend,
                    closed=self.closed,global_torch_modified=False,preference_modified=False,
                    backend=None if self.backend is None else self.backend.report(),
                    scientific_gate_scope='external real-data run report; no acceptance implied by successful API calls')

    def close(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('Close on creating thread only')
        if self.closed:return
        try:
            if self.backend is not None:self.backend.close()
        finally:self.closed=True

    def __enter__(self):
        if self.closed:raise RuntimeError('Cannot reenter closed solver')
        return self
    def __exit__(self,*args):
        self.close();return False
