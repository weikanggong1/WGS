"""Index lifetime/decoder CPU contracts; synthetic cases are not benchmarks."""
from types import SimpleNamespace
import os
import numpy as np
import pytest
import torch
from staar_phewas import gds_cuda
from test_gds_flat_decode import make_reader


def test_lazy_single_entry_reuses_owns_mutation_key_and_evicts():
    reader=SimpleNamespace();samples=np.array([7,0,4],dtype=np.int64)
    first=gds_cuda._device_sample_index(reader,samples,'cpu')
    assert first.dtype==torch.int64
    assert reader._cuda_sample_index_cache['permutation'] is None
    assert gds_cuda._device_sample_index(reader,samples.copy(),'cpu') is first
    permutation=np.argsort(np.argsort(samples))
    order=gds_cuda._device_sample_index(reader,samples,'cpu',permutation=permutation)
    assert gds_cuda._device_sample_index(reader,samples,'cpu',permutation=permutation) is order
    assert torch.equal(order,torch.tensor(permutation))
    permutation[:]=0
    assert torch.equal(order,torch.tensor([2,0,1]))
    old=reader._cuda_sample_index_cache;samples[0]=8
    replacement=gds_cuda._device_sample_index(reader,samples,'cpu')
    assert replacement is not first and reader._cuda_sample_index_cache is not old
    assert reader._cuda_sample_index_cache['permutation'] is None
    assert torch.equal(first,torch.tensor([7,0,4]))
    assert torch.equal(replacement,torch.tensor([8,0,4]))


def test_device_key_resolves_current_index_and_never_reuses_other_device(monkeypatch):
    reader=SimpleNamespace();samples=np.array([3,1]);requested=[];current=[0]
    original=torch.tensor
    def cpu_tensor_for_device_contract(*args,**kwargs):
        requested.append(kwargs.pop('device'));return original(*args,**kwargs)
    monkeypatch.setattr(torch,'tensor',cpu_tensor_for_device_contract)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:current[0])
    first=gds_cuda._device_sample_index(reader,samples,'cuda')
    assert gds_cuda._device_sample_index(reader,samples,'cuda:0') is first
    current[0]=1
    second=gds_cuda._device_sample_index(reader,samples,'cuda')
    assert second is not first and requested==[torch.device('cuda:0'),torch.device('cuda:1')]
    assert reader._cuda_sample_index_cache['device']==('cuda',1)
    gds_cuda._device_sample_index(reader,samples,'cpu')
    assert reader._cuda_sample_index_cache['device']==('cpu',None)


@pytest.mark.parametrize('variants,samples,fields',[
    ([1,0],[7,0],{'permutation'}),
    ([4,1,0],[7,0],{'sample_indices','permutation'}),
    ([1,0],[15,1,4,7,3,12,8,0],{'sample_indices'}),
])
def test_real_decoder_routes_preserve_calls_order_counts_and_only_required_indices(monkeypatch,variants,samples,fields):
    a,_,_,an=make_reader(4);b,_,_,bn=make_reader(4)
    a.genotype_raw_memory_bytes=b.genotype_raw_memory_bytes=8192
    vv=np.array(variants);ss=np.array(samples)
    got=gds_cuda.native_minor_block(a,vv,ss,device='cpu')
    entry=a._cuda_sample_index_cache
    assert {k for k in ('sample_indices','permutation') if entry[k] is not None}==fields
    # Original index creation semantics, without the cache, as a CPU control.
    monkeypatch.setattr(gds_cuda,'_device_sample_index',lambda reader,samples,device,permutation=None:
        torch.as_tensor(samples if permutation is None else permutation,dtype=torch.int64,device=device))
    expected=gds_cuda.native_minor_block(b,vv,ss,device='cpu')
    for got_array,expected_array in zip(got.trait_dense(np.arange(len(ss))),expected.trait_dense(np.arange(len(ss)))):
        np.testing.assert_array_equal(got_array,expected_array)
    for attr in ('union_ref_af','source_missing_rate','variant_indices','initial_minor_ac'):
        if hasattr(got,attr):np.testing.assert_array_equal(getattr(got,attr),getattr(expected,attr))
    np.testing.assert_array_equal(got.initial_mac(),expected.initial_mac())
    assert an.calls==bn.calls
    for route in ('flat','selected'):
        assert a._reader_io_counts[route]['calls']==b._reader_io_counts[route]['calls']
        assert a._reader_io_counts[route]['returned_bytes']==b._reader_io_counts[route]['returned_bytes']


def test_empty_decode_does_not_create_index_cache():
    reader,_,_,_=make_reader(4)
    gds_cuda.native_minor_block(reader,np.array([],dtype=np.int64),np.array([7,0]),device='cpu')
    assert not hasattr(reader,'_cuda_sample_index_cache')


@pytest.mark.skipif(os.environ.get('WGS_RUN_CUDA_CONTRACTS')!='1',reason='Explicit root GPU queue authorization required')
def test_cuda_index_lifetime_device_and_cpu_values():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    reader=SimpleNamespace();samples=np.array([7,0,4]);device=torch.device('cuda',torch.cuda.current_device())
    first=gds_cuda._device_sample_index(reader,samples,device)
    assert first.device==device and first.dtype==torch.int64
    assert gds_cuda._device_sample_index(reader,samples.copy(),device) is first
    np.testing.assert_array_equal(first.cpu().numpy(),samples)
    samples[0]=8
    second=gds_cuda._device_sample_index(reader,samples,device)
    assert second is not first
    np.testing.assert_array_equal(first.cpu().numpy(),[7,0,4])



def test_close_drops_owned_integer_cache_even_if_file_close_raises():
    from staar_phewas.gds import SeqArrayGDS
    reader=SeqArrayGDS.__new__(SeqArrayGDS)
    gds_cuda._device_sample_index(reader,np.array([2,0]),'cpu')
    def fail():raise RuntimeError('close failure contract')
    reader._file=SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError,match='close failure'):reader.close()
    assert reader._cuda_sample_index_cache is None
    reader._file=None;reader.close()
    assert reader._cuda_sample_index_cache is None
