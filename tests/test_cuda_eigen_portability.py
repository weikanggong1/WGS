import base64
from contextlib import nullcontext
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from staar_phewas.cuda_eigen import FP32SmallSpectrumSolver
from staar_phewas.cuda_eigen import library_resolution as r

class Record(str):pass
class Distribution:
 def __init__(self,site,relative,content):
  self.site=site;self.version='11.4.1.48';item=Record(relative)
  item.hash=SimpleNamespace(mode='sha256',value=base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('='));self.files=[item]
 def locate_file(self,item):return self.site/item
class VersionFunction:
 def __init__(self,value):self.value=value
 def __call__(self,pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0]=self.value;return 0
class Backend:
 def __init__(self,**kwargs):self.closed=False;self.fail=False;self.calls=0
 def __call__(self,a,**kwargs):
  self.calls+=1
  if self.fail:raise RuntimeError('selected fail')
  return 'selected'
 def close(self):self.closed=True
 def report(self):return {'actual_call_status':'selected_calls_succeeded_info0'}
class Matrix:
 def __init__(self,n):self.shape=(1,n,n);self.dtype='float32';self.is_cuda=True;self.layout='strided';self.requires_grad=False
class Contracts(unittest.TestCase):
 def fixture(self,root):
  site=root/'site';torchfile=site/'torch/__init__.py';torchfile.parent.mkdir(parents=True);torchfile.write_text('mock')
  solver=site/'nvidia/cusolver/lib/libcusolver.so.11';runtime=site/'nvidia/cuda_runtime/lib/libcudart.so.11.0'
  for path,content in ((solver,b'solver'),(runtime,b'runtime')):path.parent.mkdir(parents=True);path.write_bytes(content)
  torch=SimpleNamespace(__file__=str(torchfile),version=SimpleNamespace(cuda='11.8'))
  distributions={'nvidia-cusolver-cu11':Distribution(site,solver.relative_to(site).as_posix(),b'solver'),'nvidia-cuda-runtime-cu11':Distribution(site,runtime.relative_to(site).as_posix(),b'runtime')}
  return torch,solver,runtime,distributions
 def test_wheel_pair_record_hash_and_version_abi(self):
  with tempfile.TemporaryDirectory() as td:
   torch,solver,runtime,d=self.fixture(Path(td));libraries=r.resolve_libraries(torch,environ={},distribution_getter=d.__getitem__)
   self.assertTrue(libraries.package_hash_verified)
   libs={str(solver):SimpleNamespace(cusolverGetVersion=VersionFunction(11401)),str(runtime):SimpleNamespace(cudaRuntimeGetVersion=VersionFunction(11080))}
   report=r.query_versions(libraries,loader=libs.__getitem__);self.assertEqual(report['cuda_runtime_version'],11080)
   self.assertEqual(len(libs[str(solver)].cusolverGetVersion.argtypes),1)
   libs[str(runtime)].cudaRuntimeGetVersion.value=11070
   with self.assertRaisesRegex(RuntimeError,'Runtime CUDA'):r.query_versions(libraries,loader=libs.__getitem__)
   solver.write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'RECORD SHA'):r.resolve_libraries(torch,environ={},distribution_getter=d.__getitem__)
 def test_expected_sha_and_missing_owner_reject(self):
  with tempfile.TemporaryDirectory() as td:
   torch,solver,runtime,d=self.fixture(Path(td))
   with self.assertRaisesRegex(RuntimeError,'Expected cuSOLVER SHA'):r.resolve_libraries(torch,environ={},distribution_getter=d.__getitem__,expected_solver_sha256='0'*64)
   d['nvidia-cusolver-cu11'].files=[]
   with self.assertRaisesRegex(RuntimeError,'ownership'):r.resolve_libraries(torch,environ={},distribution_getter=d.__getitem__)
 def test_active_official_conda_pair(self):
  with tempfile.TemporaryDirectory() as td:
   prefix=Path(td);torchfile=prefix/'lib/python3.12/site-packages/torch/__init__.py';torchfile.parent.mkdir(parents=True);torchfile.write_text('mock');meta=prefix/'conda-meta';meta.mkdir()
   for name,relative,content,version in [('libcusolver','lib/libcusolver.so.11',b'solver','11.4.1.48'),('cuda-cudart','lib/libcudart.so.11.0',b'runtime','11.8.89')]:
    path=prefix/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(content)
    (meta/(name+'-0.json')).write_text(json.dumps({'name':name,'channel':'https://conda.anaconda.org/nvidia/linux-64','version':version,'files':[relative],'paths_data':{'paths':[{'_path':relative,'sha256':r.file_sha256(path),'path_type':'hardlink'}]}}))
   t=SimpleNamespace(__file__=str(torchfile),version=SimpleNamespace(cuda='11.8'))
   result=r.resolve_libraries(t,environ={'CONDA_PREFIX':str(prefix)})
   self.assertEqual(result.origin,'active official NVIDIA Conda pair');self.assertTrue(result.package_hash_verified)
   with self.assertRaises(RuntimeError):r.resolve_libraries(t,environ={})
   (meta/'libcusolver-0.json').write_text(json.dumps({'name':'libcusolver','channel':'untrusted','files':['lib/libcusolver.so.11'],'version':'11.4.1.48'}))
   with self.assertRaisesRegex(RuntimeError,'official NVIDIA'):r.resolve_libraries(t,environ={'CONDA_PREFIX':str(prefix)})
 def test_selector_explicit_policy_failure_and_restore(self):
  calls=[];t=SimpleNamespace(float32='float32',strided='strided',linalg=SimpleNamespace(eigvalsh=lambda a,**kwargs:calls.append(a.shape[-1]) or 'original'))
  original=t.linalg.eigvalsh
  with FP32SmallSpectrumSolver(torch_module=t,_backend_factory=Backend) as solver:
   for n in (1,32,513):self.assertEqual(solver.eigvalsh(Matrix(n)),'original')
   for n in (33,512):self.assertEqual(solver.eigvalsh(Matrix(n)),'selected')
   solver.backend.fail=True
   with self.assertRaisesRegex(RuntimeError,'selected fail'):solver.eigvalsh(Matrix(128))
   self.assertEqual(calls,[1,32,513]);self.assertIs(t.linalg.eigvalsh,original)
  self.assertTrue(solver.closed);self.assertTrue(solver.backend.closed)
  with self.assertRaises(RuntimeError):solver.eigvalsh(Matrix(33))
 def test_conda_missing_content_proof_stays_unverified_even_with_caller_pin(self):
  with tempfile.TemporaryDirectory() as td:
   prefix=Path(td);torchfile=prefix/'lib/python3.12/site-packages/torch/__init__.py';torchfile.parent.mkdir(parents=True);torchfile.write_text('mock');meta=prefix/'conda-meta';meta.mkdir()
   records=[('libcusolver','lib/libcusolver.so.11',b'solver','11.4.1.48'),('cuda-cudart','lib/libcudart.so.11.0',b'runtime','11.8.89')]
   for name,relative,content,version in records:
    path=prefix/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(content)
    (meta/(name+'-0.json')).write_text(json.dumps(dict(name=name,channel='nvidia',version=version,files=[relative])))
   t=SimpleNamespace(__file__=str(torchfile),version=SimpleNamespace(cuda='11.8'))
   libraries=r.resolve_libraries(t,environ={'CONDA_PREFIX':str(prefix)})
   self.assertFalse(libraries.package_hash_verified)
   pinned=r.resolve_libraries(t,environ={'CONDA_PREFIX':str(prefix)},expected_solver_sha256=libraries.solver_sha256,expected_runtime_sha256=libraries.runtime_sha256)
   self.assertFalse(pinned.package_hash_verified)
   for name,relative,_,_ in records:
    record=meta/(name+'-0.json');metadata=json.loads(record.read_text())
    metadata['paths_data']={'paths':[dict(_path=relative,path_type='softlink',sha256=r.file_sha256(prefix/relative))]}
    record.write_text(json.dumps(metadata))
   self.assertFalse(r.resolve_libraries(t,environ={'CONDA_PREFIX':str(prefix)}).package_hash_verified)
 def test_baseexception_close_no_flags_mutated(self):
  t=SimpleNamespace(float32='float32',strided='strided',linalg=SimpleNamespace(eigvalsh=lambda a,**kwargs:'original'))
  with self.assertRaises(KeyboardInterrupt):
   with FP32SmallSpectrumSolver(torch_module=t,_backend_factory=Backend) as solver:
    solver.eigvalsh(Matrix(33));raise KeyboardInterrupt()
  self.assertTrue(solver.backend.closed);self.assertTrue(solver.closed)
 def test_import_no_torch_cuda_or_cdll(self):
  code='import sys,ctypes,torch,staar_phewas; ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(AssertionError("no load")); import staar_phewas.cuda_eigen; assert not torch.cuda.is_initialized()'
  subprocess.run([sys.executable,'-c',code],check=True)
if __name__=='__main__':unittest.main()
