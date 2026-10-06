"""CPU-generated cache tests; no external GDS, data or GPU execution."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
from staar_phewas.cache_runtime import CacheSpec, make_reader_factory, run_cached_configuration
from staar_phewas.cache_runtime import store, sparse_codec_fast
from staar_phewas.cache_runtime.binding import make_source_binding
from staar_phewas.cache_runtime.export import parser, convert
from staar_phewas.cache_runtime.state_reader import build_reader, state_code

def fixture(tmp_path):
    states=np.array([[0,1,4,5],[3,2,0,1]],dtype=np.uint8)
    samples=np.array([7,1,3,5],dtype=np.int64)
    binding={'source':'synthetic-proof'}
    path=tmp_path/'cache'
    writer=store.Writer(path,binding,samples,len(states),source_bytes=10**8)
    writer.append(states,sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(states),*states.shape))
    writer.finish()
    opened=[]
    class Reader:
        n_samples=8;n_variants=2;reader_metadata={}
        closed=False
        def __init__(self,*args,**kwargs):opened.append(self)
        def close(self):self.closed=True
    source=tmp_path/'source.gds'
    spec=CacheSpec(path,binding,lambda:dict(binding),samples)
    return source,spec,Reader,opened

def test_import_paths_decoder_clone_and_semantic_state_codes():
    original=list(sys.path)
    importlib.import_module('staar_phewas.cache_runtime.sparse_decode_fast')
    reader=build_reader()
    assert sys.path==original and len(reader.decoder_ast_sha256)==64
    assert [state_code(*pair) for pair in [(2,2),(1,2),(0,2),(0,0),(1,1),(0,1)]]==list(range(6))
    with pytest.raises(ValueError):state_code(2,1)

def test_factory_exact_binding_metadata_route_and_final_proof(tmp_path):
    source,spec,Reader,opened=fixture(tmp_path)
    factory=make_reader_factory(Reader,{source:spec},device='cpu')
    with factory(source) as adapter:
        block=adapter.minor_block(np.array([1,0]),spec.expected_samples[::-1])
        np.testing.assert_array_equal(block.variant_indices,[1,0])
        np.testing.assert_array_equal(block.sample_indices,spec.expected_samples[::-1])
    assert opened[-1].closed
    with pytest.raises(ValueError):factory(tmp_path/'unbound.gds')
    current=dict(spec.expected_binding)
    mutable=CacheSpec(spec.directory,spec.expected_binding,lambda:dict(current),spec.expected_samples)
    factory=make_reader_factory(Reader,{source:mutable},device='cpu')
    adapter=factory(source);current['source']='changed'
    with pytest.raises(ValueError,match='changed during analysis'):adapter.close()
    assert opened[-1].closed and all(s.handle.closed for s in adapter._container._streams.values())
    before=len(opened)
    with pytest.raises(ValueError,match='Current source'):factory(source)
    assert len(opened)==before

def test_cli_context_restores_on_exception(tmp_path):
    from staar_phewas import cli
    source,spec,Reader,opened=fixture(tmp_path)
    class Pipeline:
        def __init__(self,*args,**kwargs):self._annotation_indexes={}
    def run(*args,**kwargs):
        with cli.SeqArrayGDS(source) as reader:
            cli.PheWASPipeline(reader,[])
            raise RuntimeError('injected-scientific-error')
    with mock.patch.object(cli,'SeqArrayGDS',Reader), mock.patch.object(cli,'PheWASPipeline',Pipeline), mock.patch.object(cli,'run_configuration',run):
        for _ in range(2):
            with pytest.raises(RuntimeError,match='injected-scientific-error'):
                run_cached_configuration({},cache_specs={source:spec},device='cpu')
            assert cli.SeqArrayGDS is Reader and cli.PheWASPipeline is Pipeline
            assert opened[-1].closed

@mock.patch('staar_phewas.cache_runtime.binding.load_packed_reader',
            return_value=SimpleNamespace(_packed_metadata={'fixture_sdk_sha':'a'*64}))
def test_full_source_packed_inputs_stat_and_plan(loader,tmp_path):
    source=tmp_path/'source.gds';source.write_bytes(b'synthetic')
    packed=tmp_path/'packed';packed.mkdir()
    (packed/'packed_binding.json').write_text('{}')
    (packed/'fixture.so').write_bytes(b'synthetic-sdk')
    config=tmp_path/'config.json';config.write_text('{}')
    before=make_source_binding(source,packed_directory=packed,input_files={'config':config})
    assert 'cache_runtime/store.py' in before['source_files_sha256']
    assert set(before['gds_stat'])=={'device','inode','size','mtime_ns','ctime_ns'}
    config.write_text('{"changed":true}')
    assert make_source_binding(source,packed_directory=packed,input_files={'config':config})!=before
    assert loader.call_count==2 and 'packed_sdk_binding' in before
    args=parser().parse_args(['--gds',str(source),'--packed-build',str(packed),
                             '--output',str(tmp_path/'never_created'),'--samples-npy','fixture.npy'])
    result=convert(args)
    assert not result['execute'] and not result['association_computation']
    assert not args.output.exists()
