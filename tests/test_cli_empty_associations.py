"""Empty native outputs retain NULL topology and honest execution evidence."""
import pytest
import rdata
from staar_phewas import cli
from staar_phewas.r_output import write_association_batch
from staar_phewas.results import TraitRows


def metadata(**overrides):
    values=dict(logical_product_count=0,ptx_verified_tf32_gemm_count=0,
                tf32_gemm_call_count=0,fp64_gemm_fallback_count=0)
    values.update(overrides)
    return values


@pytest.mark.parametrize('kind,result', [('coding',{'plof':[[],[]],'missense':[[],[]]}),
    ('noncoding',{'upstream':[[],[]]}),('ncrna',[[],[]]),
    ('individual',[TraitRows(),TraitRows()]),('singlevariant',[[]])])
def test_empty_known_topologies(kind,result):
    assert cli._association_result_row_count(result,kind=kind)==0


@pytest.mark.parametrize('pvalue',[0.,1.,float('nan')])
def test_nonempty_count_does_not_depend_on_pvalue(pvalue):
    count=cli._association_result_row_count({'plof':[[],[{'STAAR-O':pvalue}]]},kind='coding')
    assert count==1
    with pytest.raises(RuntimeError,match='no matrix/vector products'):
        cli._native_execution_status(metadata(),[{'eligible_association_tests':count}],planned_jobs=1)


@pytest.mark.parametrize('result,kind',[(None,'ncrna'),([None],'ncrna'),([],'ncrna'),
    ([[{}]],'ncrna'),([{'STAAR-O':.5}],'ncrna'),({'category':[[]]},'individual')])
def test_malformed_results_cannot_certify_empty(result,kind):
    with pytest.raises(ValueError):cli._association_result_row_count(result,kind=kind)


@pytest.mark.parametrize('jobs,planned', [([],0),([],1),
    ([{'eligible_association_tests':0}],2),([{}],1),
    ([{'eligible_association_tests':False}],1),
    ([{'eligible_association_tests':0},{'eligible_association_tests':1}],2)])
def test_zero_products_requires_complete_explicit_empty_schedule(jobs,planned):
    with pytest.raises(RuntimeError,match='no matrix/vector products'):
        cli._native_execution_status(metadata(),jobs,planned_jobs=planned)


@pytest.mark.parametrize('changes',[{'tf32_gemm_call_count':1},{'fp64_gemm_fallback_count':1}])
def test_empty_schedule_keeps_precision_verification(changes):
    with pytest.raises(RuntimeError,match='verification failed'):
        cli._native_execution_status(metadata(**changes),[{'eligible_association_tests':0}],planned_jobs=1)


def test_fresh_fit_products_are_reported_even_without_association_rows():
    assert cli._native_execution_status(metadata(logical_product_count=1),
        [{'eligible_association_tests':0}],planned_jobs=1)=='executed_products'


@pytest.mark.parametrize('layout,expected',[('base',None),('phewas',[None,None])])
def test_completed_empty_jobs_write_actual_native_null(tmp_path,layout,expected):
    results=[[[]],[[]]]
    counts=[cli._association_result_row_count(result,kind='ncrna') for result in results]
    path=tmp_path/'empty.Rdata'
    write_association_batch(path,results,kind='ncrna',object_name='empty_results',layout=layout)
    assert rdata.read_rda(path)=={'empty_results':expected}
    state=metadata()
    assert cli._native_execution_status(state,[{'eligible_association_tests':n} for n in counts],
        planned_jobs=2)=='no_eligible_analysis_products'
    assert state==metadata()


def test_loaded_null_cli_empty_job_integration_cpu_mock(tmp_path,monkeypatch):
    """Mock CUDA/GDS boundary only; use the actual native writer and CLI audit."""
    class Model:
        n=3; n_pheno=1; family='gaussian';matmul_mode='tf32'
        def set_matmul_mode(self,mode):self.matmul_mode=mode
    class Reader:
        reader_metadata={}
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Pipeline:
        def __init__(self,*args,**kwargs):pass
        def ncrna(self,**kwargs):return [[]]
    monkeypatch.setattr(cli,'GaussianNullModel',Model)
    monkeypatch.setattr(cli,'load_null_model',lambda *args,**kwargs:Model())
    monkeypatch.setattr(cli,'SeqArrayGDS',Reader)
    monkeypatch.setattr(cli,'PheWASPipeline',Pipeline)
    monkeypatch.setattr(cli,'_bind_gds_samples',lambda *args:None)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True)
    for name in ('init','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(cli.torch.cuda,name,lambda *args:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(cli.torch.cuda,name,lambda *args:0)
    monkeypatch.setattr(cli.torch.cuda,'get_device_name',lambda *args:'CPU mocked boundary')
    path=tmp_path/'empty.Rdata'
    config={'phenotypes':[{'name':'trait','model':'cache.npz'}],
        'chromosomes':[{'name':1,'gds':'input.gds','jobs':[{'kind':'ncrna','arguments':{},
            'output':str(path),'object_name':'empty_results','layout':'base'}]}]}
    report=cli.run_configuration(config,device='cuda')
    assert rdata.read_rda(path)=={'empty_results':None}
    assert report['eligible_association_tests']==0
    assert report['jobs'][0]['eligible_association_tests']==0
    assert report['native_execution_status']=='no_eligible_analysis_products'
    assert report['tf32_execution']['logical_product_count']==0
    assert report['dense_product_audit']['enabled'] is True
