import time
import numpy as np
import pytest
import torch
from staar_phewas.cli import run_configuration
from staar_phewas.chromosome import chromosome_configuration
from staar_phewas.null_model import fit_gaussian_null
from staar_phewas.io import save_null_model, load_null_model
from staar_phewas.precision_audit import DenseProductAudit
from staar_phewas.profiling import StageProfiler

def test_forced_cli_cannot_silently_run_cpu_or_unmarked_fp64():
    with pytest.raises(ValueError, match='Forced TF32 requires CUDA'):
        run_configuration({}, device='cpu')
    with pytest.raises(ValueError, match='precision_control=true'):
        run_configuration({'matmul_mode': 'fp64'}, device='cpu')

def test_cache_mode_survives_save_and_legacy_defaults_fp64(tmp_path):
    model = fit_gaussian_null(np.arange(20.0) ** 0.8, sample_ids=np.arange(20).astype(str))
    model.set_matmul_mode('tf32')
    path = tmp_path / 'model.npz'
    save_null_model(model, path)
    restored = load_null_model(path)
    assert restored.matmul_mode == 'tf32'
    with np.load(path) as values:
        legacy = {name: values[name] for name in values.files if name != 'matmul_mode'}
    np.savez(tmp_path / 'legacy.npz', **legacy)
    assert load_null_model(tmp_path / 'legacy.npz').matmul_mode == 'fp64'
    assert torch.equal(restored.scaled_residuals, model.scaled_residuals)

@pytest.mark.parametrize('operation', [lambda a: a @ a, lambda a: torch.addmm(a, a, a), lambda a: torch.mv(a, a[0]), lambda a: torch.dot(a[0], a[0])])
def test_dispatch_audit_rejects_hidden_fp64_contractions(operation):
    values = torch.eye(2, dtype=torch.float64)
    with DenseProductAudit(forced=True) as audit:
        with pytest.raises(RuntimeError, match='hidden FP64 product'):
            operation(values)
    assert audit.report()['hidden_fp64_dense_products'] == 1

def test_dispatch_audit_allows_fp64_elementwise_and_solvers():
    values = torch.eye(2, dtype=torch.float64)
    with DenseProductAudit(forced=True) as audit:
        assert (values * values).sum().item() == 2
        torch.linalg.eigh(values)
        torch.linalg.solve(values, values[:, 0])
    assert audit.report()['hidden_fp64_dense_products'] == 0

def test_profile_cpu_scopes_and_disabled_behavior():
    profiler = StageProfiler()
    with profiler.measure('score_covariance'):
        time.perf_counter()
    record = profiler.report()
    assert record['stages']['score_covariance']['calls'] == 1
    assert record['stages']['score_covariance']['host_wall_seconds'] >= 0
    assert record['stages']['score_covariance']['cuda_stream_seconds'] == 0
    assert record['pending_cuda_intervals'] == 0
    disabled = StageProfiler(enabled=False)
    with disabled.measure('score_covariance', gpu=True):
        pass
    assert disabled.report()['stages'] == {}

@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_forwards_optional_statistics_tail_flag(monkeypatch, enabled):
    from types import SimpleNamespace
    import staar_phewas.pipeline as module
    pipeline = module.PheWASPipeline.__new__(module.PheWASPipeline)
    pipeline.statistics_tail_optimization = enabled
    captured = {}

    def statistic(**kwargs):
        captured.update(kwargs)
        return {'STAAR-O': 0.5}
    monkeypatch.setattr(module, 'staar_test', statistic)
    model = SimpleNamespace(use_spa=False, n_pheno=1, matmul_mode='tf32')
    assert pipeline._evaluate_prepared({'score': 'private_fake_input_marker'}, model) == {'STAAR-O': 0.5}
    assert captured['tail_optimization'] is enabled
    assert captured['matmul_mode'] == 'tf32'
    assert getattr(pipeline, 'statistics_tail_optimization_calls', 0) == int(enabled)

def test_statistics_tail_flag_rejects_truthy_strings():
    with pytest.raises(ValueError, match='JSON boolean'):
        run_configuration({'matmul_mode': 'fp64', 'precision_control': True, 'statistics_tail_optimization': 'true'}, device='cpu')

@pytest.mark.parametrize('kind', ['sliding', 'window', 'fixed_window', 'multiple_window'])
def test_pipeline_rejects_removed_window_jobs_before_fitting(kind):
    config = {'matmul_mode': 'fp64', 'precision_control': True, 'chromosomes': [{'jobs': [{'kind': kind}]}]}
    with pytest.raises(ValueError, match='Pipeline jobs must be'):
        run_configuration(config, device='cpu')

@pytest.mark.parametrize('flag', ['local_mask_reuse', 'weight_batch_optimization'])
def test_reuse_flags_require_json_boolean(flag):
    with pytest.raises(ValueError, match='JSON boolean'):
        run_configuration({'matmul_mode': 'fp64', 'precision_control': True, flag: 'false'}, device='cpu')


@pytest.mark.parametrize('mode,effective', [('tf32',0),('fp64',None)])
def test_native_cli_resets_memory_limit_and_reports_unsplit(monkeypatch,mode,effective):
    from staar_phewas import cli,tf32
    tf32.configure_tf32(memory_limit_gib=1)
    monkeypatch.setattr(cli,'_run_configuration',lambda *a,**kw:{})
    report=cli.run_configuration({'matmul_mode':mode},device='cpu')
    assert report['tf32_configuration']['tf32_memory_limit_bytes']==20*2**30
    assert report['tf32_configuration']['effective_split_k']==effective
    assert not report['tf32_configuration']['split_k_applies']

@pytest.mark.parametrize('settings', [ {'matmul_mode':'tf32_binned'}, {'matmul_mode':'tf32x3'},
    {'tf32_binned_tile_shape':[32,64]}, {'tf32_binned_fused_small':True}, {'tf32_split_k':1024} ])
def test_removed_reconstruction_controls_fail_before_gpu(settings):
    with pytest.raises((ValueError,TypeError)):
        run_configuration(settings,device='cpu')

def test_old_cache_is_converted_on_load_before_native_computation(tmp_path):
    model=fit_gaussian_null(np.arange(20.)**.8)
    original=tmp_path/'original.npz';save_null_model(model,original)
    with np.load(original) as values:
        legacy={name:values[name] for name in values.files if name!='matmul_mode'}
    path=tmp_path/'legacy.npz';np.savez(path,**legacy)
    before=path.read_bytes()
    converted=load_null_model(path,matmul_mode='tf32')
    assert converted.source_matmul_mode=='fp64' and converted.matmul_mode=='tf32'
    for name in ('x','scaled_residuals','coefficients','theta','precision_theta','fixed_effect_covariance',
                 'inverse_variance','precision_x','phenotype','fitted_values','working_phenotype'):
        value=getattr(converted,name)
        if value is not None:assert value.dtype==torch.float32,name
    assert converted.spectrum.eigenvalues.dtype==torch.float32
    assert path.read_bytes()==before
    new=tmp_path/'new.npz';save_null_model(converted,new)
    with np.load(new) as values:assert values['x'].dtype==np.float32 and str(values['matmul_mode'])=='tf32'


def test_native_workspace_linear_single_and_fp32_budget():
    from types import SimpleNamespace
    from staar_phewas.pipeline import PheWASPipeline,AnalysisOptions
    pipeline=PheWASPipeline.__new__(PheWASPipeline)
    model=SimpleNamespace(n=42652,n_pheno=1,device='cpu',matmul_mode='fp64')
    fp64=pipeline._workspace_estimate(model,1216)
    model.matmul_mode='tf32';native=pipeline._workspace_estimate(model,1216)
    assert native*2==fp64 and native<20*2**30
    small=pipeline._workspace_estimate(model,100,individual=True)
    large=pipeline._workspace_estimate(model,200,individual=True)
    assert large-4*model.n==2*(small-4*model.n)
    pipeline.options=AnalysisOptions(memory_limit_gib=(native-1)/2**30)
    with pytest.raises(MemoryError):pipeline._limit(model,1216)


def test_native_model_formula_dtype_and_single_diagonal_cpu_mock(monkeypatch):
    import staar_phewas.null_model as module
    seen=[]
    def products(a,b,*,mode):
        assert mode=='tf32' and a.dtype==torch.float32 and b.dtype==torch.float32
        seen.append((a.shape,b.shape));return a@b
    monkeypatch.setattr(module,'matmul',products)
    model=module.fit_gaussian_null(np.arange(30.)**.8,matmul_mode='tf32')
    assert model.x.dtype==torch.float32 and model.fixed_effect_covariance.dtype==torch.float32
    genotype=torch.arange(90,dtype=torch.float64).reshape(30,3)%3
    u,v=model.score_covariance(genotype)
    s,diag=model.individual_score_variance(genotype)
    assert u.dtype==v.dtype==s.dtype==diag.dtype==torch.float32
    torch.testing.assert_close(u,s);torch.testing.assert_close(v.diagonal(),diag)
    assert model.precision_theta.dtype==torch.float32
    assert seen


def test_native_null_serialization_widens_values_only(monkeypatch):
    from staar_phewas.compat import gaussian_null_r_object
    model=fit_gaussian_null(np.arange(20.)**.8).set_matmul_mode('tf32')
    obj=gaussian_null_r_object(model)
    assert obj.value['theta'].value.dtype==np.float64
    assert obj.value['X'].value.values.dtype==np.float64
    assert model.theta.dtype==model.x.dtype==torch.float32


def test_native_mixed_model_preserves_last_preupdate_precision_cpu_mock(monkeypatch):
    import staar_phewas.null_model as module
    calls=[]
    def product(a,b,*,mode):
        assert mode=='tf32' and a.dtype==b.dtype==torch.float32
        return a@b
    monkeypatch.setattr(module,'matmul',product)
    rng=np.random.default_rng(7)
    model=module.fit_gaussian_null(rng.normal(size=80),kinship_diagonal=np.linspace(.6,1.4,80),
        edge_rows=[0,2],edge_cols=[1,3],edge_values=[.02,.03],matmul_mode='tf32',trace_callback=calls.append)
    assert calls and model.converged
    assert torch.equal(model.precision_theta,calls[-1]['tau_old'])
    assert torch.equal(model.inverse_variance,calls[-1]['inverse_variance'])
    assert torch.equal(model.fixed_effect_covariance,calls[-1]['cov'])
    assert all(rotation.dtype==torch.float32 and rows.dtype==torch.int64 for rows,rotation in model.spectrum.blocks)


@pytest.mark.parametrize('converged', [False, True])
def test_gaussian_cache_preserves_converged_and_legacy_default(tmp_path, converged):
    model = fit_gaussian_null(np.arange(20.) ** .8)
    model.converged = converged
    path = tmp_path / 'state.npz'
    save_null_model(model, path)
    assert load_null_model(path, matmul_mode='tf32').converged is converged
    with np.load(path) as values:
        legacy = {name: values[name] for name in values.files if name != 'converged'}
    np.savez(tmp_path / 'legacy_state.npz', **legacy)
    assert load_null_model(tmp_path / 'legacy_state.npz').converged is True


def test_native_identity_rotation_alias_and_readonly_consumers(monkeypatch):
    import staar_phewas.null_model as module
    monkeypatch.setattr(module, 'matmul', lambda a, b, *, mode: a @ b)
    model = module.fit_gaussian_null(np.arange(30.) ** .8, matmul_mode='tf32')
    genotype = (torch.arange(90).reshape(30, 3) % 3).float()
    original = genotype.clone()
    rotated = model.spectrum.rotate(genotype, matmul_mode='tf32')
    assert rotated is genotype
    model.score_covariance(genotype)
    model.individual_score_variance(genotype)
    assert torch.equal(genotype, original)
    # A nonidentity block still requires independent output storage.
    model.spectrum.blocks = [(torch.tensor([0, 1]), torch.tensor([[0., 1.], [1., 0.]]))]
    changed = model.spectrum.rotate(genotype, matmul_mode='tf32')
    assert changed.data_ptr() != genotype.data_ptr()
    assert torch.equal(genotype, original)
    assert torch.equal(changed[:2], genotype[[1, 0]])


def test_single_tail_uses_fp64_probability_vectors_only():
    from staar_phewas.pipeline import _individual_log_probabilities
    score = torch.tensor([1., 20., 38., 100., 1., 1., 1.], dtype=torch.float32)
    variance = torch.tensor([1., .7, 1., .03, 0., -1., float('nan')], dtype=torch.float32)
    saved_score, saved_variance = score.clone(), variance.clone()
    actual = _individual_log_probabilities(score, variance)
    expected = -np.log(2.) - torch.special.log_ndtr(-score[:4].double().abs() / variance[:4].double().sqrt())
    assert actual.dtype == torch.float64
    assert torch.equal(actual[:4], expected)
    assert torch.isfinite(actual[:4]).all() and actual[3] > 100000
    assert torch.equal(actual[4:6], torch.zeros(2, dtype=torch.float64))
    assert torch.isnan(actual[6])
    assert torch.equal(score, saved_score)
    torch.testing.assert_close(variance, saved_variance, equal_nan=True)


def test_cached_null_residual_uses_source_dtype_without_changing_native_state(tmp_path):
    from staar_phewas.compat import gaussian_null_r_object
    model = fit_gaussian_null(np.arange(20.) ** .8)
    original = tmp_path / "model.npz"
    save_null_model(model, original)
    with np.load(original, allow_pickle=False) as values:
        expected = np.subtract(values["phenotype"], values["fitted_values"]).astype(np.float64)
    native = load_null_model(original, matmul_mode="tf32")
    y_before, fit_before = native.phenotype.clone(), native.fitted_values.clone()
    result = gaussian_null_r_object(native)
    np.testing.assert_array_equal(result.value["residuals"].value, expected)
    assert native.phenotype.dtype == torch.float32
    torch.testing.assert_close(native.phenotype, y_before, rtol=0, atol=0)
    torch.testing.assert_close(native.fitted_values, fit_before, rtol=0, atol=0)
