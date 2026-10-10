"""Minimal explicit full-spectrum selector for integration into weighted_spectra."""
import threading
from ._backend import BatchedEigenBackend

class FP32SmallSpectrumSolver:
    """Own one serialized solver, selected n33..512; other shapes use original Torch.

    Only a CUDA FP32 upper-triangle Torch convergence error permits a scoped
    MAGMA retry. Torch functions and TF32 flags remain unchanged; the library
    preference is restored even when retry fails. Backend and official CUDA
    library are loaded lazily at the first selected CUDA call.
    """
    def __init__(self,*,torch_module=None,memory_limit=40*2**30,
                 expected_solver_sha256=None,expected_runtime_sha256=None,
                 _backend_factory=None):
        if torch_module is None:
            import torch as torch_module
        if type(memory_limit) is not int or memory_limit <= 0:
            raise ValueError('memory_limit must be positive integer bytes')
        self.torch=torch_module;self.original=torch_module.linalg.eigvalsh
        self.memory_limit=memory_limit;self.expected_solver_sha256=expected_solver_sha256
        self.expected_runtime_sha256=expected_runtime_sha256
        self._factory=BatchedEigenBackend if _backend_factory is None else _backend_factory
        self.backend=None;self.closed=False;self.owner=threading.get_ident()
        self.selected_calls=0;self.outside_calls=0
        self.last_execution_backend=None
        self._linalg_error=getattr(getattr(torch_module,'_C',None),'_LinAlgError',())
        self.magma_retry_attempts=0;self.magma_retry_successes=0;self.magma_retry_failures=0
        self.magma_retry_matrices=0;self.magma_retry_max_n=0;self.magma_retry_shapes=[]
        self.magma_preference_restores=0;self.magma_nonfinite_rejections=0

    def _original_eigvalsh(self,matrix,*,UPLO):
        try:
            result=self.original(matrix,UPLO=UPLO)
        except self._linalg_error:
            shape=tuple(matrix.shape)
            preference=getattr(getattr(getattr(self.torch,'backends',None),'cuda',None),
                               'preferred_linalg_library',None)
            if (matrix.dtype!=self.torch.float32 or not matrix.is_cuda or
                matrix.layout!=self.torch.strided or matrix.requires_grad or UPLO!='U' or
                len(shape) not in (2,3) or shape[-2]!=shape[-1] or shape[-1]<=0 or
                not callable(preference) or not getattr(self.torch._C,'_has_magma',False)):
                raise
            if not bool(self.torch.isfinite(matrix).all().item()):
                self.magma_nonfinite_rejections+=1
                raise
            previous=preference()
            self.magma_retry_attempts+=1
            self.magma_retry_matrices+=1 if len(shape)==2 else shape[0]
            self.magma_retry_max_n=max(self.magma_retry_max_n,shape[-1])
            self.magma_retry_shapes.append(list(shape))
            try:
                try:
                    preference('magma')
                    result=self.original(matrix,UPLO=UPLO)
                finally:
                    preference(previous)
                    self.magma_preference_restores+=1
            except BaseException:
                self.magma_retry_failures+=1
                raise
            self.magma_retry_successes+=1
            self.last_execution_backend='torch.linalg.eigvalsh:magma_retry'
            return result
        self.last_execution_backend='torch.linalg.eigvalsh'
        return result

    def eigvalsh(self,matrix,*,UPLO='U'):
        if self.closed:raise RuntimeError('Closed FP32 spectrum solver')
        if threading.get_ident()!=self.owner:raise RuntimeError('Solver belongs to its creating thread')
        self.last_execution_backend=None
        shape=tuple(matrix.shape)
        selected=len(shape) in (2,3) and shape[-2]==shape[-1] and 33<=shape[-1]<=512
        if not selected:
            self.outside_calls+=1
            return self._original_eigvalsh(matrix,UPLO=UPLO)
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
        return dict(selected_range=[33,512],outside_policy='original Torch; CUDA FP32 LinAlgError-only scoped MAGMA retry',
                    selected_calls=self.selected_calls,outside_calls=self.outside_calls,
                    last_execution_backend=self.last_execution_backend,
                    closed=self.closed,global_torch_modified=False,
                    preference_modified=self.magma_retry_attempts>0,
                    preference_restored=self.magma_preference_restores==self.magma_retry_attempts,
                    magma_retry_attempts=self.magma_retry_attempts,
                    magma_retry_successes=self.magma_retry_successes,
                    magma_retry_failures=self.magma_retry_failures,
                    magma_retry_matrices=self.magma_retry_matrices,
                    magma_retry_max_n=self.magma_retry_max_n,
                    magma_retry_shapes=list(self.magma_retry_shapes),
                    magma_preference_restores=self.magma_preference_restores,
                    magma_nonfinite_rejections=self.magma_nonfinite_rejections,
                    magma_retry_backend='torch.linalg.eigvalsh:magma_retry',
                    retry_input_dtype='float32',retry_output_dtype='float32',
                    retry_scaling=False,retry_regularization=False,retry_fp64=False,
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
