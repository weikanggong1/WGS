"""Shared-runtime scheduled indexes retain order and avoid unused inputs."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fudan_wgs_toolkit import cli
from fudan_wgs_toolkit.masks import NONCODING_CATEGORIES
from fudan_wgs_toolkit.phewas_runtime import runtime

@pytest.mark.parametrize('specs,expected', [
    ([('individual', {}), ('coding', {'start': 1, 'end': 3})], []),
    ([('noncoding', {'start': 1, 'end': 3}), ('ncrna', {'start': 1, 'end': 3})], []),
    ([('ncrna', {}), ('ncrna', {})], ['ncRNA']),
    ([('noncoding', {'category': 'upstream'}), ('noncoding', {'category': 'UTR'}),
      ('noncoding', {'category': 'upstream', 'include_ncrna': True})], ['upstream', 'UTR', 'ncRNA']),
    ([('coding', {'start': 1, 'end': 3}), ('noncoding', {}), ('ncrna', {}), ('individual', {})], [*NONCODING_CATEGORIES, 'ncRNA']),
])
def test_shared_runtime_prepares_scheduled_index_once_in_order(tmp_path, monkeypatch, specs, expected):
    events=[]
    class Reader:
        n_samples=3
        n_variants=3
        reader_metadata={}
        def __init__(self,*args,**kwargs):
            assert kwargs=={'container_directory':str(tmp_path/'container')}
            events.append(('open',))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read_field(self,field,rows=None):return np.arange(3) if field=='position' else np.full(3,'PASS')
    class Container:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Broker(Container):
        metrics={}
        def __init__(self,reader,*args,**kwargs):self.reader=reader
        def reader_view(self,rows):return self.reader
    class Pipeline:
        def __init__(self,*args,**kwargs):
            events.append(('pipeline',))
            self._base_masks={};self._category_codes=None;self._annotation_indexes={};self._coding_masks_cache={}
            self.skipped_sets=[];self.local_mask_reuse_counters={}
        def prepare_annotation_index(self,chromosome,**kwargs):
            events.append(('index',kwargs['categories']))
            assert kwargs['include_ncrna'] is False
        def __getattr__(self,name):
            if name in {'coding','noncoding','ncrna'}:
                def job(**kwargs):
                    events.append(('job',name));return [[]]
                return job
            raise AttributeError(name)
    for name in ('init','is_available','reset_peak_memory_stats','synchronize','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,(lambda *args:True) if name=='is_available' else (lambda *args:0))
    model=SimpleNamespace(n=3,x=torch.ones((3,1)),family='gaussian',use_spa=False)
    monkeypatch.setattr(runtime,'_load_models',lambda *args:([model],[None]))
    monkeypatch.setattr(runtime,'PortableMetadataReader',Reader)
    monkeypatch.setattr(runtime,'Container',Container)
    monkeypatch.setattr(runtime,'SharedStateBroker',Broker)
    monkeypatch.setattr(runtime,'LimitedMaskPipeline',Pipeline)
    monkeypatch.setattr(cli,'_bind_genotype_samples',lambda *args:np.arange(3))
    def single(*args,**kwargs):events.append(('job','individual'));return [[[]]]
    monkeypatch.setattr(runtime,'_single',single)
    monkeypatch.setattr(runtime,'write_association_batch',lambda *args,**kwargs:{'rows':0})
    promoter=tmp_path/'promoters.tsv'
    if any(category.startswith('promoter_') for category in expected):promoter.write_text('1\t1\t3\n')
    jobs=[]
    for i,(kind,arguments) in enumerate(specs):
        args=dict(arguments)
        if kind!='individual':args['gene_name']='GENE_A'
        if kind=='coding':args.update(start=1,end=3)
        jobs.append(dict(kind=kind,arguments=args,output=str(tmp_path/f'{i}.csv')))
    source=str(tmp_path/'metadata')
    config=dict(weighted_eigensolver='torch',phenotypes=[dict(model='state.npz')],
        chromosomes=[dict(name='1',genotype=source,jobs=jobs,
                          annotation_index=dict(promoter_intervals_file=str(promoter)))])
    spec=SimpleNamespace(directory=str(tmp_path/'container'),expected_binding={},source_proof=lambda:{},expected_samples=None)
    report=runtime.run_configuration([config],cache_specs={source:spec},device='cuda:0')
    assert [event for event in events if event[0]=='index']==([('index',expected)] if expected else [])
    assert [event[1] for event in events if event[0]=='job']==[kind for kind,_ in specs]
    assert events.count(('open',))==events.count(('pipeline',))==1
    assert [job['kind'] for job in report['jobs']]==[kind for kind,_ in specs]
    if not expected:assert report['annotation_seconds']==0
