"""Owned FP32 full-spectrum backend; library discovery is explicit.

No Torch/library/CUDA work on module import. Scientific acceptance is external.
"""
import ctypes as C
import hashlib
import math
from pathlib import Path
import threading
import time
from .library_resolution import resolve_libraries, query_versions

PROCESS_LIMIT = 20 * 2**30
RESERVE_BYTES = 256 * 2**20
NOVECTOR = 0
UPPER = 1


def library_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''): h.update(part)
    return h.hexdigest()


def available_bytes(allocated, reserved, free, *, limit=PROCESS_LIMIT, reserve=RESERVE_BYTES):
    if any(type(x) is not int or x < 0 for x in (allocated, reserved, free, limit, reserve)):
        raise ValueError('memory snapshot must contain nonnegative integer bytes')
    if limit > PROCESS_LIMIT or limit == 0: raise ValueError('process limit must be in (0,20 GiB]')
    return max(0, min(limit - allocated, free + max(0, reserved - allocated) - reserve))


def geometry(shape, *, max_n=512):
    if type(max_n) is not int or not 1 <= max_n <= 512: raise ValueError('max_n must be in [1,512]')
    shape = tuple(shape)
    if len(shape) not in (2, 3) or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('require nonempty 2D or 3D square matrix batch')
    n = shape[-1]; batch = shape[0] if len(shape) == 3 else 1
    if shape[-2] != n or n > max_n: raise ValueError('square matrix exceeds explicit experiment size range')
    if batch > 2**31-1: raise ValueError('batch exceeds cuSOLVER int32 API')
    return n, batch, shape[:-1], 4 * batch * (n*n + n + 1)


def column_major_copy(a, torch_module):
    """Pure storage conversion; always owns separate storage, including n=1."""
    return a.transpose(-1,-2).clone(memory_format=torch_module.contiguous_format)


class CuSolverAPI:
    """Checked documented FP32 SsyevjBatched ABI; handles are created only on call."""
    def __init__(self, path, *, loader=C.CDLL):
        self.path = Path(path).resolve(); self.sha256 = library_sha(self.path)
        self.lib = loader(str(self.path)); self.calls = 0
        ptr = C.c_void_p; integer = C.c_int
        signatures = {
            'cusolverDnCreate': [C.POINTER(ptr)], 'cusolverDnDestroy': [ptr],
            'cusolverDnSetStream': [ptr, ptr],
            'cusolverDnCreateSyevjInfo': [C.POINTER(ptr)], 'cusolverDnDestroySyevjInfo': [ptr],
            'cusolverDnXsyevjSetMaxSweeps': [ptr, integer],
            'cusolverDnXsyevjSetSortEig': [ptr, integer],
            'cusolverDnSsyevjBatched_bufferSize': [ptr, integer, integer, integer, ptr, integer, ptr,
                                                 C.POINTER(integer), ptr, integer],
            'cusolverDnSsyevjBatched': [ptr, integer, integer, integer, ptr, integer, ptr, ptr,
                                        integer, ptr, ptr, integer],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = integer

    def call(self, name, *args):
        self.calls += 1
        status = getattr(self.lib, name)(*args)
        if status != 0: raise RuntimeError('cuSOLVER API status nonzero: '+name+' status='+str(status))

    def create(self):
        handle, params = C.c_void_p(), C.c_void_p()
        self.call('cusolverDnCreate', C.byref(handle))
        try:
            self.call('cusolverDnCreateSyevjInfo', C.byref(params))
            # Leave tolerance at cuSOLVER's default machine accuracy, exactly as Torch does.
            self.call('cusolverDnXsyevjSetMaxSweeps', params, 100)
            self.call('cusolverDnXsyevjSetSortEig', params, 1)
        except BaseException:
            try:
                if params.value: self.call('cusolverDnDestroySyevjInfo', params)
            finally: self.call('cusolverDnDestroy', handle)
            raise
        return handle, params

    def set_stream(self, handle, stream): self.call('cusolverDnSetStream', handle, C.c_void_p(stream))

    def buffer_size(self, handle, params, n, batch, a, w):
        size = C.c_int()
        self.call('cusolverDnSsyevjBatched_bufferSize', handle, NOVECTOR, UPPER, n,
                  C.c_void_p(a), n, C.c_void_p(w), C.byref(size), params, batch)
        if size.value < 0: raise RuntimeError('cuSOLVER returned negative workspace')
        return size.value

    def solve(self, handle, params, n, batch, a, w, work, lwork, info):
        self.call('cusolverDnSsyevjBatched', handle, NOVECTOR, UPPER, n, C.c_void_p(a), n,
                  C.c_void_p(w), C.c_void_p(work), lwork, C.c_void_p(info), params, batch)

    def destroy(self, handle, params):
        try: self.call('cusolverDnDestroySyevjInfo', params)
        finally: self.call('cusolverDnDestroy', handle)


class BatchedEigenBackend:
    def __init__(self, *, torch_module=None, library=None, api=None, max_n=512,
                 memory_limit=PROCESS_LIMIT, reserve_bytes=RESERVE_BYTES,
                 expected_solver_sha256=None,expected_runtime_sha256=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        geometry((1,1), max_n=max_n)
        available_bytes(0,0,0,limit=memory_limit,reserve=reserve_bytes)
        self.library_metadata = None
        if api is None:
            if library is not None:raise ValueError('Only constrained installed official CUDA library resolution is supported')
            libraries=resolve_libraries(self.torch,expected_solver_sha256=expected_solver_sha256,
                                        expected_runtime_sha256=expected_runtime_sha256)
            self.library_metadata=query_versions(libraries)
            api = CuSolverAPI(libraries.solver)
            if api.sha256!=libraries.solver_sha256:raise RuntimeError("cuSOLVER SHA changed between version proof and ABI load")
        self.api = api; self.max_n = max_n; self.limit = memory_limit; self.reserve = reserve_bytes
        self.device = None; self.handle = None; self.params = None; self.closed = False
        self._cleanup = dict(attempted=False, completed=False, failed=False, error_type=None)
        self._lock = threading.Lock(); self._owner = threading.get_ident()
        self._metrics = dict(calls=0,successful_calls=0,failed_calls=0,matrices=0,eigenvalues=0,
            max_matrix_n=0,max_batch_size=0,max_workspace_bytes=0,max_clone_output_info_bytes=0,
            fresh_memory_queries=0,workspace_rejections=0,info_D2H_calls=0,info_nonzero=0,
            output_validation_D2H_calls=0,host_wall_seconds=0.,api_calls=0,
            out_of_scope_original_calls=0,max_validation_scratch_bytes=0,n_above32_successful_calls=0)

    def _guard(self, required):
        t = self.torch
        allocated=int(t.cuda.memory_allocated(self.device)); reserved=int(t.cuda.memory_reserved(self.device))
        free,_=t.cuda.mem_get_info(self.device)
        self._metrics['fresh_memory_queries'] += 1
        if required > available_bytes(allocated,reserved,int(free),limit=self.limit,reserve=self.reserve):
            self._metrics['workspace_rejections'] += 1
            raise MemoryError('complete FP32 eigensolver workspace exceeds fresh live/process cap; no fallback')

    def __call__(self, a, *, UPLO='U', out=None):
        t=self.torch
        if self.closed: raise RuntimeError('closed solver backend')
        if threading.get_ident()!=self._owner: raise RuntimeError('solver context is bound to its creating thread')
        if UPLO!='U' or out is not None: raise ValueError('candidate requires UPLO=U and fresh output')
        if a.dtype!=t.float32 or not a.is_cuda or a.layout!=t.strided or a.requires_grad:
            raise ValueError('candidate requires strided CUDA FP32 tensor without autograd')
        n,batch,output_shape,base_bytes=geometry(tuple(a.shape),max_n=self.max_n)
        if self.device is not None and a.device!=self.device: raise ValueError('context cannot change CUDA device')
        if not self._lock.acquire(blocking=False): raise RuntimeError('overlapping solver calls unsupported')
        self.device=a.device;self._metrics['calls']+=1;start=time.perf_counter()
        try:
            with t.cuda.device(self.device):
                validation_bytes=3*batch*n+65536
                self._guard(base_bytes+validation_bytes)
                # Always clone. Even n=1 or an already column-contiguous input never aliases a.
                column_major=column_major_copy(a,t)
                w=t.empty(output_shape,dtype=t.float32,device=self.device)
                info=t.empty((batch,),dtype=t.int32,device=self.device)
                if self.handle is None:self.handle,self.params=self.api.create()
                self.api.set_stream(self.handle,int(t.cuda.current_stream(self.device).cuda_stream))
                lwork=self.api.buffer_size(self.handle,self.params,n,batch,column_major.data_ptr(),w.data_ptr())
                if type(lwork) is not int or lwork<0 or lwork>2**31-1:raise RuntimeError('invalid cuSOLVER workspace size')
                workspace_bytes=4*lwork
                self._guard(workspace_bytes+validation_bytes)
                work=t.empty((lwork,),dtype=t.float32,device=self.device)
                self.api.solve(self.handle,self.params,n,batch,column_major.data_ptr(),w.data_ptr(),
                               work.data_ptr(),lwork,info.data_ptr())
                values=info.cpu().tolist();self._metrics['info_D2H_calls']+=1
                if len(values)!=batch or any(type(x) is not int for x in values):raise RuntimeError('malformed cuSOLVER info result')
                nonzero=sum(x!=0 for x in values);self._metrics['info_nonzero']+=nonzero
                if nonzero:raise RuntimeError('cuSOLVER eigen batch failed/nonconverged; info must be zero for every matrix')
                self._metrics['output_validation_D2H_calls']+=1
                if not bool(t.isfinite(w).all().item()):raise RuntimeError('nonfinite full eigen spectrum')
                if n>1:
                    self._metrics['output_validation_D2H_calls']+=1
                    if not bool((w[...,1:]>=w[...,:-1]).all().item()):raise RuntimeError('cuSOLVER returned unsorted spectrum')
                self._metrics['successful_calls']+=1;self._metrics['matrices']+=batch
                self._metrics['n_above32_successful_calls']+=int(n>32)
                self._metrics['max_validation_scratch_bytes']=max(self._metrics['max_validation_scratch_bytes'],validation_bytes)
                self._metrics['eigenvalues']+=batch*n
                self._metrics['max_matrix_n']=max(self._metrics['max_matrix_n'],n)
                self._metrics['max_batch_size']=max(self._metrics['max_batch_size'],batch)
                self._metrics['max_workspace_bytes']=max(self._metrics['max_workspace_bytes'],workspace_bytes)
                self._metrics['max_clone_output_info_bytes']=max(self._metrics['max_clone_output_info_bytes'],base_bytes)
                return w
        except BaseException:
            self._metrics['failed_calls']+=1;raise
        finally:
            self._metrics['host_wall_seconds']+=time.perf_counter()-start;self._lock.release()

    def close(self):
        if self.closed:return
        if threading.get_ident()!=self._owner:raise RuntimeError('close on creating thread only')
        # Destroy can release only part of the resources before raising. Detach
        # both pointers and stop all calls before invoking it; never retry a
        # possibly freed pointer on a second close.
        handle,params=self.handle,self.params
        self.handle=None;self.params=None
        self.closed=True
        if handle is None:
            self._cleanup['completed']=True
            return
        self._cleanup['attempted']=True
        try:
            with self.torch.cuda.device(self.device):self.api.destroy(handle,params)
        except BaseException as error:
            self._cleanup['failed']=True
            self._cleanup['error_type']=type(error).__name__
            raise
        else:
            self._cleanup['completed']=True

    def report(self):
        return dict(self._metrics,api_calls=getattr(self.api,'calls',None),backend='cusolverDnSsyevjBatched',
            library_sha256=getattr(self.api,'sha256',None),torch_version=str(self.torch.__version__),
            cuda_version=self.torch.version.cuda,input_dtype='float32',output_dtype='float32',
            matrix_uplo='U',jobz='NOVECTOR',sort_eigenvalues=True,
            tolerance='cuSOLVER CreateSyevjInfo default machine accuracy; identical default setup to Torch2.5.1',
            max_sweeps=100,max_allowed_n=self.max_n,process_limit_bytes=self.limit,reserve_bytes=self.reserve,
            library_metadata=self.library_metadata,closed=self.closed,
            cleanup_attempted=self._cleanup['attempted'],cleanup_completed=self._cleanup['completed'],
            cleanup_failed=self._cleanup['failed'],cleanup_error_type=self._cleanup['error_type'],
            input_mutated=False,eigenvalues_truncated=False,fp64_dense_products=False,implicit_fallback=False,
            actual_call_status=('selected_calls_succeeded_info0' if self._metrics['successful_calls'] and not self._metrics['failed_calls'] else
                'selected_calls_failed' if self._metrics['failed_calls'] else 'not_called'),
            scientific_gate_scope='external real-data original-software comparison, recorded by run report',
            timing_scope='host wall includes synchronization from info/finite/sorted validation; not pure GPU kernel time',
            memory_scope='fresh free/process guards for cloned column-major input/output/info and solver workspace; not a reservation or proof of hidden internal scratch limit')

