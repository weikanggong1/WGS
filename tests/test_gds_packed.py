"""Packed-reader fail-closed and decoder semantic contracts, CPU only."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
import torch
from staar_phewas import gds_packed as packed
from staar_phewas import gds_cuda


def binding_fixture(tmp_path):
    source=tmp_path/'source';(source/'pygds/include').mkdir(parents=True);(source/'src/CoreArray').mkdir(parents=True)
    for name in ('PyGDS.h','PyGDS2.h','dType.h','CoreDEF.h'):(source/'pygds/include'/name).write_text(name)
    (source/'src/CoreArray/CoreArray.h').write_text('official fixture')
    sdk=tmp_path/'sdk.so';sdk.write_bytes(b'sdk');compiled=tmp_path/('staar_gds_packed'+packed._platform_binding()['extension_suffix']);compiled.write_bytes(b'adapter')
    headers=packed._headers(source)
    b=dict(schema=1,platform=packed._platform_binding(),extension_filename=compiled.name,extension_sha256=packed._sha(compiled),source_directory=str(source),source_built_sdk=str(sdk),sdk_binary_sha256=packed._sha(sdk),source_files=packed._source_files(source),headers=headers,official_headers_sha256=packed._header_digest(headers),layout=dict(iterator=24,allocator=128,pointer=8,position=8),adapter_source_sha256=packed._sha(Path(packed.__file__).with_name('_gds_packed.cpp')))
    (tmp_path/'packed_binding.json').write_text(json.dumps(b))
    return b,sdk,source,compiled


def patches(b,sdk,source):
    exported=json.loads(json.dumps({k:b[k] for k in ('sdk_binary_sha256','official_headers_sha256','layout')}))
    module=SimpleNamespace(binding=lambda:exported)
    return mock.patch.object(packed.importlib,'import_module',return_value=SimpleNamespace(__file__=str(sdk))),mock.patch.dict(sys.modules,{'pygds':SimpleNamespace(get_include=lambda:str(source/'pygds/include'))}),mock.patch.object(packed,'_import_binary',return_value=module)


def test_unconfigured_reader_has_no_import_or_IO():
    with mock.patch.object(packed.importlib,'import_module',side_effect=AssertionError),mock.patch.object(Path,'read_text',side_effect=AssertionError):
        assert packed.load_packed_reader() is None


@pytest.mark.parametrize('mismatch',['platform','binary','built','header','layout','source','filename'])
def test_explicit_binding_rejected(tmp_path,mismatch):
    b,sdk,source,compiled=binding_fixture(tmp_path);contexts=patches(b,sdk,source)
    if mismatch=='platform':b['platform']['machine']='different'
    elif mismatch=='binary':compiled.write_bytes(b'wrong')
    elif mismatch=='built':b['source_built_sdk']=str(tmp_path/'missing.so')
    elif mismatch=='header':(source/'src/CoreArray/CoreArray.h').write_text('changed')
    elif mismatch=='layout':b['layout']['pointer']=4
    elif mismatch=='source':b['adapter_source_sha256']='0'*64
    else:b['extension_filename']='../adapter.so'
    (tmp_path/'packed_binding.json').write_text(json.dumps(b))
    with contexts[0],contexts[1],contexts[2],pytest.raises((RuntimeError,FileNotFoundError)):
        packed.load_packed_reader(tmp_path)


def test_valid_binding_returns_only_public_metadata(tmp_path):
    b,sdk,source,_=binding_fixture(tmp_path);contexts=patches(b,sdk,source)
    with contexts[0],contexts[1],contexts[2]:module=packed.load_packed_reader(tmp_path)
    assert module._packed_metadata['packed_binding_verified'] is True
    assert str(tmp_path) not in json.dumps(module._packed_metadata)


def test_explicit_import_path_not_sys_path(tmp_path):
    binary=tmp_path/'module.so';loaded=SimpleNamespace(__file__=str(binary));loader=SimpleNamespace(exec_module=mock.Mock())
    spec=SimpleNamespace(loader=loader)
    with mock.patch.object(importlib.util,'spec_from_file_location',return_value=spec) as create,mock.patch.object(importlib.util,'module_from_spec',return_value=loaded):
        assert packed._import_binary(binary) is loaded
    create.assert_called_once_with('staar_gds_packed',binary)
    loader.exec_module.assert_called_once_with(loaded)


class Reader:
    n_samples=5;ploidy=2;genotype_max_gap_layers=3;_stage_profiler=None
    def __init__(self,use_packed,samples):
        self._file=SimpleNamespace(fileid=1);self._flat_reader=self;self._packed_reader=self if use_packed else None
        self._genotype_steps=np.array([0,1,2,3,1],np.uint8);self._genotype_offsets=np.r_[0,np.cumsum(self._genotype_steps,dtype=np.int64)]
        codes=[None,np.array([[0,0],[0,3],[1,1],[3,3],[0,1]]),np.array([[0,15],[1,1],[0,0],[2,3],[15,15]]),np.array([[0,63],[0,0],[1,1],[3,0],[63,63]]),np.array([[0,1],[0,1],[3,3],[0,0],[1,1]])]
        self.raw=np.stack([(codes[v]>>(2*l))&3 for v in range(1,5) for l in range(int(self._genotype_steps[v]))]).astype(np.uint8)
        x=self.raw.reshape(-1);self.packed=np.zeros((len(x)+3)//4,np.uint8)
        for i,c in enumerate(x):self.packed[i//4]|=int(c)<<(2*(i%4))
        self.coverage=[];n=len(samples);self.genotype_raw_memory_bytes=10+5+16*n+12*(n*2)+2*(10+4*(n*2))
    def _prepare_genotype_index(self):pass
    def _record_minor_coverage(self,*a):self.coverage.append(a)
    def _sample_selection(self,s):return np.repeat(np.isin(np.arange(self.n_samples),s),2),np.argsort(np.argsort(s))
    def read_flat_path(self,f,p,o,c,d):return self.raw.reshape(-1)[o:o+c].copy()
    def read_selected_rows_path(self,f,p,o,rows,width,select,d):return self.raw.reshape(-1)[o:o+rows*width].reshape(rows,width)[:,select].copy().reshape(-1)
    def read_packed_path(self,f,p,off,count,N):
        assert N==self.n_samples
        first=off//4;size=(off%4+count+3)//4
        return self.packed[first:first+size].copy(),2*(off%4),count,first,len(self.packed)


def cpu_unpack(reader,raw,info,samples,selected,layers,device,measure):
    # Semantic fixture only. Real runtime uses one GPU integer kernel.
    s=np.sort(samples) if selected else samples;q=2*(np.arange(layers)[:,None,None]*reader.n_samples*2+s[None,:,None]*2+np.arange(2)[None,None,:])+info[0]
    return torch.as_tensor(((raw[q>>3]>>(q&7))&3).astype(np.uint8),device=device)


@pytest.mark.parametrize('samples',[[4,0],[4,0,2],[4,3,2,1,0],[]])
@pytest.mark.parametrize('variants',[[4,0,1,3],[2,4,3],[0],[3]])
@pytest.mark.parametrize('minimum',[None,2])
@pytest.mark.parametrize('resident',[False,True])
def test_packed_preserves_decoder_semantics(samples,variants,minimum,resident):
    samples=np.asarray(samples,dtype=np.int64);variants=np.asarray(variants,dtype=np.int64);control=Reader(False,samples);candidate=Reader(True,samples)
    with mock.patch.object(gds_cuda,'_packed_raw_tensor',side_effect=cpu_unpack):
        left=gds_cuda.native_minor_block(control,variants,samples,device='cpu',minimum_mac=minimum,resident=resident)
        right=gds_cuda.native_minor_block(candidate,variants,samples,device='cpu',minimum_mac=minimum,resident=resident)
    fields=['sample_indices','variant_indices','union_ref_af','union_initial_mac','union_missing_rate','union_ref_ac','union_called_alleles']+(['dosage'] if resident else ['row','col','value'])
    for field in fields:
        a=getattr(left,field);b=getattr(right,field)
        if isinstance(a,torch.Tensor):a=a.numpy();b=b.numpy()
        assert np.array_equal(a,b,equal_nan=True),field
    assert len(control.coverage)==len(candidate.coverage)
    if len(samples) and np.any(candidate._genotype_steps[variants]):assert candidate._reader_io_counts['packed']['calls']>0


def test_read_error_not_fallback_and_requires_reopen():
    samples=np.array([4,0]);reader=Reader(True,samples);reader.read_packed_path=mock.Mock(side_effect=RuntimeError('read failed'))
    with pytest.raises(RuntimeError,match='read failed'):gds_cuda.native_minor_block(reader,np.array([1,3]),samples,device='cpu')
    assert reader._packed_read_failed
    with pytest.raises(RuntimeError,match='reopen'):gds_cuda.native_minor_block(reader,np.array([1]),samples,device='cpu')
    assert reader.read_packed_path.call_count==1


def test_builder_mismatch_refuses_compilation(tmp_path):
    b,sdk,source,_=binding_fixture(tmp_path);wrong=tmp_path/'different_sdk.so';wrong.write_bytes(b'different')
    with mock.patch.object(packed.importlib,'import_module',return_value=SimpleNamespace(__file__=str(sdk))),mock.patch.dict(sys.modules,{'pygds':SimpleNamespace()}),mock.patch.object(packed.subprocess,'run') as compile_call,pytest.raises(RuntimeError,match='differ'):
        packed.build_packed_reader(source_dir=source,source_built_sdk=wrong,output_dir=tmp_path/'output')
    compile_call.assert_not_called()
    assert not (tmp_path/'output').exists()


def test_constructor_config_fails_before_open():
    from staar_phewas import gds
    file_factory=mock.Mock()
    with mock.patch.object(gds,'load_flat_reader',return_value=None),mock.patch.object(packed,'load_packed_reader',side_effect=RuntimeError('binding mismatch')),mock.patch.dict(sys.modules,{'pygds':SimpleNamespace(gdsfile=file_factory)}),pytest.raises(RuntimeError,match='binding mismatch'):
        gds.SeqArrayGDS('unused',packed_reader_directory='configured')
    file_factory.assert_not_called()


def test_constructor_storage_failure_closes_file():
    from staar_phewas import gds
    dimensions={'sample.id':[5],'variant.id':[4],'genotype/data':[7,5,2]}
    f=SimpleNamespace(fileid=1,open=mock.Mock(),close=mock.Mock(),index=lambda path:SimpleNamespace(description=lambda:dict(dim=dimensions[path])))
    backend=SimpleNamespace(read_packed_path=mock.Mock(side_effect=TypeError('not Bit2')))
    with mock.patch.object(gds,'load_flat_reader',return_value=None),mock.patch.object(packed,'load_packed_reader',return_value=backend),mock.patch.dict(sys.modules,{'pygds':SimpleNamespace(gdsfile=lambda:f)}),pytest.raises(TypeError,match='Bit2'):
        gds.SeqArrayGDS('unused',packed_reader_directory='configured')
    backend.read_packed_path.assert_called_once_with(1,'genotype/data',0,0,5)
    f.close.assert_called_once()


def test_packed_guard_honors_configured_lower_limit_without_GPU():
    from contextlib import nullcontext
    from staar_phewas import tf32
    reader=SimpleNamespace(n_samples=5);raw=np.zeros(4,np.uint8)
    with mock.patch.object(tf32,'_memory_limit_bytes',1024),mock.patch.object(torch.cuda,'memory_allocated',return_value=0),mock.patch.object(torch.cuda,'memory_reserved',return_value=0),mock.patch.object(torch.cuda,'mem_get_info',return_value=(2**30,2**30)),pytest.raises(MemoryError,match='workspace'):
        gds_cuda._packed_raw_tensor(reader,raw,(0,10,0,4),np.array([4,0]),False,1,'cuda:0',lambda *a,**k:nullcontext())
    assert reader._packed_workspace_snapshot['process_limit_bytes']==1024
    assert reader._packed_workspace_snapshot['reserve_bytes']==tf32._memory_reserve_bytes
    assert reader._packed_workspace_snapshot['required_bytes']>reader._packed_workspace_snapshot['available_bytes']


def test_compiled_stream_policy_unknown_known_and_invalid_sizes(tmp_path):
    """Exercise the actual C++ policy used by read_packed, without SDK or data."""
    import shutil
    import subprocess
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('Optional adapter requires a C++ compiler')
    cpp=Path(packed.__file__).with_name('_gds_packed.cpp').read_text()
    policy=cpp[cpp.index('static bool check_stream_bounds'):cpp.index('// End stream bounds policy.')]
    harness='''#include <cassert>
namespace CoreArray { using SIZE64 = long long; }
int runtime_error=1,index_error=2,last_error=0;
int* PyExc_RuntimeError=&runtime_error;
int* PyExc_IndexError=&index_error;
void PyErr_SetString(int* kind,const char*) {last_error=*kind;}
'''+policy+'''
int main(){
 assert(check_stream_bounds(-1,10,0,0));
 assert(check_stream_bounds(-1,10,0,1));
 assert(check_stream_bounds(-1,10,9,1));
 assert(check_stream_bounds(10,10,9,1));
 assert(check_stream_bounds(12,10,0,10));
 assert(!check_stream_bounds(-2,10,0,0));assert(last_error==runtime_error);
 assert(!check_stream_bounds(9,10,0,0));assert(last_error==index_error);
 assert(!check_stream_bounds(-1,10,9,2));assert(last_error==index_error);
 assert(!check_stream_bounds(-1,10,11,0));assert(last_error==index_error);
}
'''
    source=tmp_path/'policy.cpp';source.write_text(harness);binary=tmp_path/'policy'
    subprocess.run([compiler,'-std=c++11',str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)
