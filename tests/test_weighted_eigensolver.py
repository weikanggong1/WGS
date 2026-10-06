"""CPU contracts for public solver selection/lifetime; not a GPU benchmark."""
import threading

import pytest
import torch

from staar_phewas import cli, _weighted_spectra as weighted
from staar_phewas.cuda_eigen import FP32SmallSpectrumSolver
from staar_phewas.cuda_eigen import _backend as backend_module
from test_cuda_eigen_backend import API, Torch, Tensor


@pytest.mark.parametrize('requested,mode,device,weight_batch,expected,reason', [
    ('auto','tf32','cuda:0',True,'cusolver_batched',None),
    ('cusolver_batched','tf32','cuda',True,'cusolver_batched',None),
    ('torch','tf32','cuda',True,'torch','explicit_torch'),
    ('auto','tf32','cpu',True,'torch','device_not_cuda'),
    ('cusolver_batched','tf32','cpu',True,'torch','device_not_cuda'),
    ('auto','fp64','cuda',True,'torch','matmul_mode_not_tf32'),
    ('cusolver_batched','fp64','cpu',True,'torch','matmul_mode_not_tf32'),
    ('auto','tf32','cuda',False,'torch','weight_batch_optimization_disabled'),
    ('cusolver_batched','tf32','cuda',False,'torch','weight_batch_optimization_disabled'),
])
def test_config_selection_has_explicit_control_and_cpu_policy(requested,mode,device,weight_batch,expected,reason):
    settings=cli._weighted_eigensolver_settings(
        dict(weighted_eigensolver=requested,weight_batch_optimization=weight_batch),mode,device)
    assert settings['requested']==requested
    assert settings['effective']==expected and settings['inactive_reason']==reason
    assert cli._weighted_eigensolver_settings({},'tf32','cuda')['requested']=='auto'


@pytest.mark.parametrize('value',[None,True,[],{},'jacobi','CUDA'])
def test_bad_config_rejected_before_gpu(value,monkeypatch):
    monkeypatch.setattr(cli,'_run_configuration',lambda *args,**kwargs:pytest.fail('analysis entered'))
    with pytest.raises(ValueError,match='weighted_eigensolver'):
        cli.run_configuration({'weighted_eigensolver':value},device='cuda')


def mock_solver_factory(monkeypatch,api=None):
    module=Torch();api=API() if api is None else api;instances=[];limits=[]
    def factory(memory_limit):
        limits.append(memory_limit)
        def backend_factory(**kwargs):
            return backend_module.BatchedEigenBackend(api=api,**kwargs)
        solver=FP32SmallSpectrumSolver(torch_module=module,memory_limit=memory_limit,
            _backend_factory=backend_factory)
        instances.append(solver)
        return solver
    monkeypatch.setattr(weighted,'_make_solver',factory)
    return module,api,instances,limits


def test_one_context_reuses_one_owned_handle_and_reports_actual_routes(monkeypatch):
    module,api,instances,limits=mock_solver_factory(monkeypatch)
    original=torch.linalg.eigvalsh
    with weighted.eigensolver_context('cusolver_batched',memory_limit=2*2**30) as state:
        assert len(instances)==1 and instances[0].backend is None
        for n in (33,64,33):
            values,route=weighted._complete_eigvalsh(Tensor((2,n,n)))
            assert route=='cusolverDnSsyevjBatched' and values.shape==(2,n)
        outside,route=weighted._complete_eigvalsh(Tensor((2,32,32)))
        assert outside=='original' and route=='torch.linalg.eigvalsh'
        assert sum(call[0]=='create' for call in api.seen)==1
    assert limits==[2*2**30] and len(instances)==1
    assert torch.linalg.eigvalsh is original
    assert state['selector']['closed'] and state['selector']['backend']['closed']
    assert state['selector']['backend']['cleanup_completed']
    assert state['selector']['selected_calls']==3 and state['selector']['outside_calls']==1
    assert state['selector']['backend']['matrices']==6
    assert module.original_calls==1
    report=weighted.execution_metadata()
    assert report['actual_backend_calls']=={'cusolverDnSsyevjBatched':3,'torch.linalg.eigvalsh':1}
    assert report['actual_backend_matrices']=={'cusolverDnSsyevjBatched':6,'torch.linalg.eigvalsh':2}
    assert report['eigensolver_context']['selector']['backend']['closed']


def test_selected_error_propagates_with_no_torch_retry_and_context_restores(monkeypatch):
    module,api,instances,_=mock_solver_factory(monkeypatch)
    api.info=4
    with pytest.raises(RuntimeError,match='info must be zero'):
        with weighted.eigensolver_context('cusolver_batched'):
            weighted._complete_eigvalsh(Tensor((1,33,33)))
    assert module.original_calls==0 and instances[0].backend.closed
    report=weighted.execution_metadata()
    assert report['actual_backend_calls']=={}
    assert report['eigensolver_context']['selector']['backend']['info_nonzero']==1
    assert not weighted._SOLVER_CONTEXT_ACTIVE and weighted._SOLVER is None
    with weighted.eigensolver_context('torch'):
        actual,route=weighted._complete_eigvalsh(torch.eye(2)[None])
        assert route=='torch.linalg.eigvalsh' and torch.equal(actual,torch.ones((1,2)))


def test_baseexception_body_closes_handle_and_nested_context_cannot_replace_it(monkeypatch):
    _,api,instances,_=mock_solver_factory(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        with weighted.eigensolver_context('cusolver_batched'):
            with pytest.raises(RuntimeError,match='nested'):
                with weighted.eigensolver_context('torch'):
                    pytest.fail('nested scope entered')
            weighted._complete_eigvalsh(Tensor((1,33,33)))
            raise KeyboardInterrupt()
    assert instances[0].backend.closed
    assert sum(call[0]=='destroy' for call in api.seen)==1
    assert weighted._SOLVER is None and not weighted._SOLVER_CONTEXT_ACTIVE


def test_other_thread_cannot_enter_or_use_active_pipeline_context(monkeypatch):
    _,api,instances,_=mock_solver_factory(monkeypatch)
    failures=[]
    with weighted.eigensolver_context('cusolver_batched'):
        def other_thread():
            try:
                with weighted.eigensolver_context('torch'):
                    failures.append('unexpected context entry')
            except RuntimeError:
                failures.append('context rejected')
            try:
                weighted._complete_eigvalsh(Tensor((1,33,33)))
            except RuntimeError:
                failures.append('solve rejected')
        thread=threading.Thread(target=other_thread)
        thread.start();thread.join()
    assert failures==['context rejected','solve rejected'] and api.calls==0
    assert instances[0].closed and instances[0].backend is None


def test_cpu_native_values_preserved_inside_selected_context_without_backend_load(monkeypatch):
    _,api,instances,_=mock_solver_factory(monkeypatch)
    from staar_phewas import statistics
    covariance=torch.tensor([[2.,.25],[.25,3.]],dtype=torch.float32)
    weights=torch.tensor([[1.,2.,1.,3.],[2.,4.,3.,6.]],dtype=torch.float32)
    expected=weighted.native_weighted_spectra(covariance,weights)
    with weighted.eigensolver_context('cusolver_batched'):
        actual=statistics._native_weighted_spectra(covariance,weights)
    assert torch.equal(actual,expected)
    assert instances[0].backend is None and api.calls==0
    assert weighted.execution_metadata()['actual_backend_calls']=={'torch.linalg.eigvalsh':1}


def test_successful_selected_route_never_calls_original_torch_shape_recorder(monkeypatch):
    from staar_phewas import statistics
    original=torch.linalg.eigvalsh
    def actual_api_boundary(matrices):
        return original(matrices,UPLO='U'),'cusolverDnSsyevjBatched'
    monkeypatch.setattr(weighted,'_complete_eigvalsh',actual_api_boundary)
    monkeypatch.setattr(statistics,'record_gpu_eigen_route',
        lambda _:pytest.fail('selected API reported as original Torch'))
    # Only the solve boundary is mocked; weighting and full spectrum scaling
    # use native CPU Torch. This checks metadata plumbing, not GPU accuracy.
    result=weighted.native_weighted_spectra(torch.eye(2,dtype=torch.float32),
        torch.ones((2,1),dtype=torch.float32))
    assert torch.equal(result,torch.ones((1,2),dtype=torch.float64))


def test_cli_owns_whole_run_and_post_close_report(monkeypatch):
    _,api,instances,_=mock_solver_factory(monkeypatch)
    def analysis(config,**kwargs):
        for _ in range(3):weighted._complete_eigvalsh(Tensor((1,33,33)))
        return {'total_seconds':1.}
    monkeypatch.setattr(cli,'_run_configuration',analysis)
    report=cli.run_configuration({'weighted_eigensolver':'cusolver_batched'},device='cuda:0')
    state=report['weighted_eigensolver_execution']
    assert state['requested']=='cusolver_batched' and state['effective']=='cusolver_batched'
    assert state['selector']['backend']['closed'] and state['selector']['backend']['cleanup_completed']
    assert state['selector']['backend']['successful_calls']==3
    assert report['weighted_spectrum_execution']['actual_backend_calls']=={'cusolverDnSsyevjBatched':3}
    assert report['total_seconds']>=1.
    assert len(instances)==1 and sum(call[0]=='destroy' for call in api.seen)==1


def test_cli_body_failure_always_closes_and_restores(monkeypatch):
    _,api,instances,_=mock_solver_factory(monkeypatch)
    def analysis(config,**kwargs):
        weighted._complete_eigvalsh(Tensor((1,33,33)))
        raise KeyboardInterrupt()
    monkeypatch.setattr(cli,'_run_configuration',analysis)
    with pytest.raises(KeyboardInterrupt):cli.run_configuration({},device='cuda')
    assert instances[0].backend.closed and weighted._SOLVER is None
    assert sum(call[0]=='destroy' for call in api.seen)==1


def test_cli_reference_torch_does_not_create_selector(monkeypatch):
    monkeypatch.setattr(weighted,'_make_solver',lambda _:pytest.fail('CPU/control loaded selector'))
    monkeypatch.setattr(cli,'_run_configuration',lambda *args,**kwargs:{})
    report=cli.run_configuration({'matmul_mode':'fp64','precision_control':True,
        'weighted_eigensolver':'cusolver_batched'},device='cpu')
    state=report['weighted_eigensolver_execution']
    assert state['effective']=='torch' and state['selector'] is None


def test_cli_flag_explicitly_overrides_config(tmp_path,monkeypatch,capsys):
    import json,sys
    filename=tmp_path/'config.json'
    filename.write_text(json.dumps({'weighted_eigensolver':'auto'}))
    captured=[]
    monkeypatch.setattr(cli,'run_configuration',lambda config,**kwargs:captured.append(config) or {})
    monkeypatch.setattr(sys,'argv',['staar-phewas-torch',str(filename),'--weighted-eigensolver','torch'])
    cli.main();capsys.readouterr()
    assert captured==[{'weighted_eigensolver':'torch'}]
    assert json.loads(filename.read_text())=={'weighted_eigensolver':'auto'}


def test_failed_context_factory_restores_scope_before_next_run(monkeypatch):
    def factory(_):raise ValueError('mock factory failure')
    monkeypatch.setattr(weighted,'_make_solver',factory)
    with pytest.raises(ValueError,match='factory failure'):
        with weighted.eigensolver_context('cusolver_batched'):
            pytest.fail('failed factory entered body')
    assert weighted._SOLVER is None and not weighted._SOLVER_CONTEXT_ACTIVE
    with weighted.eigensolver_context('torch'):
        pass


def test_context_destroy_failure_is_reported_and_lock_is_restored(monkeypatch):
    from test_cuda_eigen_cleanup import DestroyFailure
    _,api,instances,_=mock_solver_factory(monkeypatch,DestroyFailure(RuntimeError))
    with pytest.raises(RuntimeError,match='mock cleanup error'):
        with weighted.eigensolver_context('cusolver_batched'):
            weighted._complete_eigvalsh(Tensor((1,33,33)))
    report=weighted.execution_metadata()['eigensolver_context']['selector']['backend']
    assert report['closed'] and report['cleanup_attempted'] and report['cleanup_failed']
    assert not report['cleanup_completed']
    assert report['actual_call_status']=='selected_calls_succeeded_info0'
    instances[0].close()
    assert sum(call[0]=='destroy' for call in api.seen)==1
    with weighted.eigensolver_context('torch'):
        pass
