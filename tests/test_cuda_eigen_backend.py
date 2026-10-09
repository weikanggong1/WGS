"""CPU/mock contracts only; no Torch import, CUDA library load or GPU benchmark."""
from contextlib import nullcontext
import ctypes as C
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from staar_phewas.cuda_eigen import _backend as m

REGISTRY={}

class Flag:
    def __init__(self,v):self.value=v
    def all(self):return self
    def item(self):return self.value

class Tensor:
    def __init__(self,shape,values=None,dtype='float32',is_cuda=True):
        self.shape=tuple(shape);self.ndim=len(self.shape);self.values=list(values or [0]*math.prod(shape));self.dtype=dtype
        self.is_cuda=is_cuda;self.device='cuda:0';self.layout='strided';self.requires_grad=False
        REGISTRY[id(self)]=self
    def data_ptr(self):return id(self)
    def transpose(self,*axes):
        assert axes==(-1,-2)
        n=self.shape[-1];batch=math.prod(self.shape[:-2]) if len(self.shape)>2 else 1
        vals=[]
        for b in range(batch):
            a=self.values[b*n*n:(b+1)*n*n]
            vals.extend(a[i*n+j] for j in range(n) for i in range(n))
        return Tensor(self.shape,vals,dtype=self.dtype)
    def clone(self,**kwargs):
        assert kwargs=={'memory_format':'contiguous'}
        return Tensor(self.shape,self.values,dtype=self.dtype)
    def cpu(self):return self
    def tolist(self):return self.values
    def __getitem__(self,item):
        assert item[0] is Ellipsis
        n=self.shape[-1];batch=math.prod(self.shape[:-1])
        values=[]
        for b in range(batch):values.extend(self.values[b*n:(b+1)*n][item[1]])
        return Tensor(self.shape[:-1]+(n-1,),values)
    def __ge__(self,other):return Flag(all(a>=b for a,b in zip(self.values,other.values)))

class CUDA:
    free=32*2**30;allocated=100*2**20;reserved=100*2**20
    def device(self,device):return nullcontext()
    def memory_allocated(self,device):return self.allocated
    def memory_reserved(self,device):return self.reserved
    def mem_get_info(self,device):return self.free,40*2**30
    def current_stream(self,device):return SimpleNamespace(cuda_stream=123)

class Torch:
    __version__='2.5.1+cu118';version=SimpleNamespace(cuda='11.8');__file__='/unused/torch/__init__.py'
    float32='float32';int32='int32';strided='strided';contiguous_format='contiguous'
    def __init__(self):
        self.cuda=CUDA();self.original_calls=0
        def eig(*args,**kwargs):self.original_calls+=1;return 'original'
        self.linalg=SimpleNamespace(eigvalsh=eig)
    def empty(self,shape,*,dtype,device):return Tensor(shape,dtype=dtype)
    def isfinite(self,t):return Flag(all(math.isfinite(v) for v in t.values))

class API:
    sha256='a'*64
    def __init__(self):self.calls=0;self.seen=[];self.info=0;self.lwork=24;self.invalid=None
    def create(self):self.seen.append(('create',));self.calls+=1;return 1,2
    def set_stream(self,h,s):self.seen.append(('stream',h,s));self.calls+=1
    def buffer_size(self,h,p,n,b,a,w):
        self.seen.append(('buffer',n,b,REGISTRY[a].values[:],a));self.calls+=1;return self.lwork
    def solve(self,h,p,n,b,a,w,work,lwork,info):
        self.seen.append(('solve',n,b,lwork));self.calls+=1
        REGISTRY[info].values=[self.info]*b
        REGISTRY[w].values=[float(j-1) for _ in range(b) for j in range(n)]
        if self.invalid=='nan':REGISTRY[w].values[0]=float('nan')
        if self.invalid=='unsorted':REGISTRY[w].values[0]=99.
    def destroy(self,h,p):self.seen.append(('destroy',h,p));self.calls+=1

class Function:
    def __init__(self,name,seen,fail):self.name=name;self.seen=seen;self.fail=fail
    def __call__(self,*args):
        self.seen.append((self.name,args))
        if self.name in ('cusolverDnCreate','cusolverDnCreateSyevjInfo'):
            C.cast(args[0],C.POINTER(C.c_void_p))[0]=C.c_void_p(44)
        if self.name.endswith('bufferSize'):C.cast(args[7],C.POINTER(C.c_int))[0]=24
        return 7 if self.name==self.fail else 0

class Library:
    def __init__(self,fail=None):self.functions={};self.seen=[];self.fail=fail
    def __getattr__(self,name):
        if name not in self.functions:self.functions[name]=Function(name,self.seen,self.fail)
        return self.functions[name]

class Contracts(unittest.TestCase):
    def backend(self):
        t=Torch();api=API();return m.BatchedEigenBackend(torch_module=t,api=api),t,api
    def test_geometry_full_spectrum_and_guards(self):
        self.assertEqual(m.geometry((3,4,4)),(4,3,(3,4),252))
        self.assertEqual(m.geometry((1,1)),(1,1,(1,),12))
        for shape in [(0,2,2),(2,3),(2,513,513),(1,1,1,1)]:
            with self.assertRaises(ValueError):m.geometry(shape)
        with self.assertRaises(ValueError):m.available_bytes(0,0,0,limit=0)
        self.assertEqual(m.available_bytes(100,200,300,limit=1000,reserve=50),350)
    def test_column_major_copy_input_preserved_full_negative_values(self):
        b,t,api=self.backend();a=Tensor((2,2),[1,9,3,4]);before=a.values[:]
        w=b(a,UPLO='U');b.close()
        self.assertEqual(api.seen[2][3],[1,3,9,4]);self.assertNotEqual(api.seen[2][4],a.data_ptr())
        self.assertEqual(a.values,before);self.assertEqual(w.values,[-1.,0.]);self.assertEqual(w.shape,(2,))
        self.assertEqual(b.report()['eigenvalues'],2);self.assertEqual(api.seen[1],('stream',1,123))
        self.assertEqual(b.report()['actual_call_status'],'selected_calls_succeeded_info0')
    def test_singleton_always_copy_and_full_batch_shape(self):
        b,t,api=self.backend();a=Tensor((1,1),[7]);b(a)
        self.assertNotEqual(api.seen[2][4],a.data_ptr());self.assertEqual(a.values,[7])
        w=b(Tensor((3,2,2)));self.assertEqual(w.shape,(3,2));self.assertEqual(len(w.values),6);b.close()
    def test_fp64_cpu_grad_uplo_device_shape_never_fallback(self):
        for change in ('dtype','is_cuda','requires_grad','UPLO','out'):
            b,t,api=self.backend();a=Tensor((2,2));kwargs={}
            if change=='dtype':a.dtype='float64'
            elif change=='is_cuda':a.is_cuda=False
            elif change=='requires_grad':a.requires_grad=True
            elif change=='UPLO':kwargs['UPLO']='L'
            else:kwargs['out']=a
            with self.assertRaises(ValueError):b(a,**kwargs)
            self.assertEqual(t.original_calls,0);self.assertEqual(api.calls,0)
    def test_info_nonzero_nonfinite_unsorted_fail_closed(self):
        for failure in ('info','nan','unsorted'):
            b,t,api=self.backend()
            if failure=='info':api.info=3
            else:api.invalid=failure
            with self.assertRaises(RuntimeError):b(Tensor((2,2)))
            self.assertEqual(t.original_calls,0);self.assertEqual(b.report()['failed_calls'],1);b.close()
    def test_fresh_memory_before_clone_and_before_solver_workspace(self):
        b,t,api=self.backend();t.cuda.free=0
        with self.assertRaises(MemoryError):b(Tensor((2,2)))
        self.assertEqual(api.calls,0)
        b,t,api=self.backend();old=api.buffer_size
        def size(*args):t.cuda.free=0;return old(*args)
        api.buffer_size=size
        with self.assertRaises(MemoryError):b(Tensor((2,2)))
        self.assertEqual(b.report()['fresh_memory_queries'],2)
        self.assertFalse(any(x[0]=='solve' for x in api.seen));b.close()
    def test_workspace_invalid_no_solver(self):
        b,t,api=self.backend();api.lwork=-1
        with self.assertRaises(RuntimeError):b(Tensor((2,2)))
        self.assertFalse(any(x[0]=='solve' for x in api.seen));b.close()
    def test_ctypes_signature_default_tolerance_sweeps_sort_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mock.so';path.write_bytes(b'mock_abi_not_real_library');lib=Library()
            a=m.CuSolverAPI(path,loader=lambda _:lib);h,p=a.create();a.set_stream(h,123)
            self.assertEqual(a.buffer_size(h,p,64,25,1,2),24);a.solve(h,p,64,25,1,2,3,24,4);a.destroy(h,p)
            calls=[x[0] for x in lib.seen]
            self.assertNotIn('cusolverDnXsyevjSetTolerance',calls)
            self.assertEqual(lib.seen[2][1][1],100);self.assertEqual(lib.seen[3][1][1],1)
            args=next(x[1] for x in lib.seen if x[0]=='cusolverDnSsyevjBatched')
            self.assertEqual(args[1:4],(0,1,64));self.assertEqual(args[-1],25)
            self.assertEqual(len(lib.cusolverDnSsyevjBatched.argtypes),12)
    def test_api_error_cleans_created_handle(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mock.so';path.write_bytes(b'mock');lib=Library('cusolverDnXsyevjSetSortEig')
            a=m.CuSolverAPI(path,loader=lambda _:lib)
            with self.assertRaises(RuntimeError):a.create()
            self.assertEqual([x[0] for x in lib.seen][-2:],['cusolverDnDestroySyevjInfo','cusolverDnDestroy'])

if __name__=='__main__':unittest.main()
