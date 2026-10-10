"""Resource admission contracts; synthetic fixtures are not WGS benchmarks."""
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.phewas_storage import ModelRepository, save_model_store
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from test_phewas_batched_runtime import (
    Repository, cpu_product_oracle, descriptor, model_for)
from test_phewas_shared_state import MetadataReader, make_cache, fixture_states
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader


def simulated_cuda(monkeypatch, repository, *, total=1000):
    """Track owned cache bytes and external allocations without GPU allocation."""
    state = dict(external=0, csr=0, emptied=0)
    def used():
        return state['external']+state['csr']+sum(
            value.bytes for bank in (repository.models, repository.spa_sources)
            for key, value in bank.items() if key[1] == 'cuda:0')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (total-used(), total))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: used())
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: used())
    def empty():
        state['emptied'] += 1
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty)
    return state


def entry(bytes):
    return SimpleNamespace(bytes=bytes)


def test_dynamic_reservation_evicts_only_device_owned_unpinned_models_and_sources(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    repository.models = OrderedDict([
        ((0, 'cpu'), entry(10)), ((1, 'cuda:0'), entry(200)), ((2, 'cuda:0'), entry(250))])
    repository.pinned[(1, 'cuda:0')] = 1
    repository.spa_sources[(3, 'cuda:0')] = entry(200)
    repository.spa_pinned[(3, 'cuda:0')] = 1
    state = simulated_cuda(monkeypatch, repository)
    with repository.reserve_workspace('cuda:0', 550, reserve_bytes=0):
        assert (2, 'cuda:0') not in repository.models
        assert (0, 'cpu') in repository.models
        assert (1, 'cuda:0') in repository.models
        assert (3, 'cuda:0') in repository.spa_sources
        assert repository.summary()['workspace_evictions'] == 1
        with pytest.raises(MemoryError, match='pinned states are retained'):
            with repository.reserve_workspace('cuda:0', 60, reserve_bytes=0):
                pass
        assert repository.summary()['active_workspace_reservations'] == 1
    assert state['emptied'] and repository.summary()['active_workspace_reservations'] == 0


def test_reservations_restore_after_exception_and_phase_failure(monkeypatch):
    repository = ModelRepository([], workspace_bytes=50)
    simulated_cuda(monkeypatch, repository)
    with repository.reserve_workspace('cuda:0', 200, reserve_bytes=0) as outer:
        with pytest.raises(RuntimeError, match='consumer'):
            with repository.reserve_workspace('cuda:0', 300, reserve_bytes=0):
                assert repository._workspace_for(torch.device('cuda:0'))[0] == 500
                raise RuntimeError('consumer')
        assert repository._workspace_for(torch.device('cuda:0'))[0] == 200
        with pytest.raises(MemoryError):
            outer.set_required(1001)
        assert outer.required_bytes == 200
    assert repository._workspace_for(torch.device('cuda:0'))[0] == 50
    with pytest.raises(RuntimeError, match='closed'):
        outer.ensure_free()


@pytest.mark.parametrize('source', [False, True])
def test_hit_admission_protects_requested_hit_and_respects_live_pressure(monkeypatch, source):
    repository = ModelRepository([], workspace_bytes=0)
    bank = repository.spa_sources if source else repository.models
    bank[(0, 'cuda:0')], bank[(1, 'cuda:0')] = entry(200), entry(300)
    repository._descriptors = [dict(spa_available=True, family='gaussian')]
    state = simulated_cuda(monkeypatch, repository)
    with repository.reserve_workspace('cuda:0', 300, reserve_bytes=0):
        state['external'] = 250  # Pressure appeared after initial admission.
        acquire = (repository._acquire_binary_source(0, 'cuda:0') if source
                   else repository.acquire([0], 'cuda:0'))
        with acquire as result:
            actual = result if source else result[0]
            assert actual.bytes == 200
            assert (1, 'cuda:0') not in bank and (0, 'cuda:0') in bank
        assert repository.summary()['workspace_evictions'] == 1


def test_miss_reserves_fitted_state_and_workspace_before_load(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.phewas_storage as storage
    model = model_for(np.arange(20))
    saved = save_model_store(model, tmp_path/'model', sample_rows=np.arange(20), fit_metadata={})
    repository = ModelRepository([saved], workspace_bytes=0)
    repository.models[(9, 'cuda:0')] = entry(2000)
    state = simulated_cuda(monkeypatch, repository, total=3000)
    called = []
    def load(*args, **kwargs):
        assert (9, 'cuda:0') not in repository.models
        called.append(True)
        return entry(1000)
    monkeypatch.setattr(storage, 'load_model_store', load)
    with repository.reserve_workspace('cuda:0', 900, reserve_bytes=0):
        # The cached bank leaves 1000 bytes, less than fitted state + 900.
        with repository.acquire([0], 'cuda:0') as models:
            assert models[0].bytes == 1000
    assert called == [True] and repository.pinned[(0, 'cuda:0')] == 0


def test_coordinated_csr_retirement_never_evicts_pinned_model(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    repository.models[(0, 'cuda:0')] = entry(300)
    repository.pinned[(0, 'cuda:0')] = 1
    state = simulated_cuda(monkeypatch, repository)
    state['csr'] = 300
    released = []
    def release():
        released.append(state['csr'])
        state['csr'] = 0
    with repository.reserve_workspace('cuda:0', 500, reserve_bytes=0, cache_release=release):
        assert released == [300] and (0, 'cuda:0') in repository.models
    assert repository.summary()['workspace_cache_release_calls'] == 1


def test_allocated_covariance_is_not_recharged_as_future_bytes(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    state = simulated_cuda(monkeypatch, repository)
    with repository.reserve_workspace('cuda:0', 800, reserve_bytes=0) as workspace:
        state['external'] = 300  # This is now the allocated input covariance.
        workspace.set_required(500, phase='gene_tail')
        assert workspace.required_bytes == 500
        workspace.ensure_free(500, phase='gene_tail')
    phases = repository.summary()['workspace_phase_counts']
    assert phases['gene_tail'] == 2


@pytest.mark.parametrize('value', [-1, True, 1.5, None])
def test_invalid_workspace_bytes_are_rejected(value):
    repository = ModelRepository([])
    with pytest.raises(ValueError):
        with repository.reserve_workspace('cpu', value):
            pass


def test_broker_restores_owner_hook_and_active_slab_survives_cache_retirement(tmp_path):
    raw = fixture_states(m=7, n=11)
    with SharedStateBroker(MetadataReader(11, 7), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        slab = broker.read_states(np.arange(7))
        hooks = []
        with broker.workspace_guard(lambda required, **kwargs: hooks.append(required)):
            assert broker._workspace_admission is not None
            with pytest.raises(ValueError):
                with broker.workspace_guard(lambda *args, **kwargs: None):
                    raise ValueError('consumer')
            assert broker._workspace_admission is not None
            broker.release_device_cache()
            actual = broker.trait_block(slab, np.arange(11))
            assert actual.variant_indices.tolist() == list(range(7))
        assert broker._workspace_admission is None


def test_tail_plan_includes_staar_mask_copies_and_excludes_input_covariance():
    options = AnalysisOptions()
    n, m = 339013, 40000
    tail = runtime._gene_tail_new_bytes(n, m, options)
    assert tail == 16*m*m+64*m*512+4*n
    assert tail > 8*m*m+24*m*512  # Later FastSKAT-only allowance is insufficient.
    assert runtime._gene_host_covariance_new_bytes(n, m, 30, options) > 4*m*m


def test_gene_runtime_uses_covariance_then_incremental_tail_and_cleans_workspace(
        tmp_path, cpu_product_oracle, monkeypatch):
    n, m = 256, 6
    raw = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        raw[i, i] = 1
    class Reader(MetadataReader):
        def read_field(self, name, indices=None):
            values = np.full(m, 'PASS') if name == 'annotation/filter' else np.arange(m)
            if name == 'chromosome':
                values = np.full(m, '21')
            return values if indices is None else values[indices]
    class WorkspaceRepository(Repository):
        @contextmanager
        def reserve_workspace(self, device, required_bytes, **kwargs):
            phases.append(('enter', required_bytes))
            workspace = SimpleNamespace(
                set_required=lambda value, **kw: phases.append((kw['phase'], value)),
                ensure_free=lambda *args, **kw: None)
            try:
                yield workspace
            finally:
                phases.append(('exit', None))
    phases, writes = [], []
    ds = [descriptor(i, np.arange(n)) for i in range(3)]
    repository = WorkspaceRepository(ds, [model_for(np.arange(n), .7+i) for i in range(3)])
    _, groups = runtime._descriptor_groups(repository)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks', lambda self, *args: {'ncRNA': np.arange(m)})
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_long_mask_products',
                        lambda self, model, host: model.score_covariance(torch.from_numpy(host)))
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared',
                        lambda self, payload, model: dict(num_variant=len(payload['maf']),
                                                         cMAC=payload['cmac'], **{'STAAR-O': .25}))
    with SharedStateBroker(SharedMetadataReader(Reader(n, m)), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.arange(n))
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(axis, repository.models[0], axis._axis.samples, options, {})
        structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
        report = dict(association_rows={'ncrna': 0}, trait_rows={i: {'ncrna': 0} for i in range(3)},
                      binary_spa_selected=0, binary_spa_failed=0)
        runtime._batched_gene(groups, repository, broker, broker._reader, options, {},
            dict(gene_name='fixture_gene', start=0, end=5), SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)),
            report, set(), '21', 0, 2, structural, 'ncrna')
        tail = runtime._gene_tail_new_bytes(n, m, options)
        assert [value for phase, value in phases if phase == 'gene_tail'] == [tail]*3
        assert len(writes) == 3 and repository.active == 0
        assert phases[-1] == ('exit', None) and broker._workspace_admission is None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                    reason='actual long-mask live admission requires Ampere CUDA')
def test_gpu_live_eviction_then_long_mask_rank512_admission(tmp_path):
    from fudan_wgs_toolkit import statistics, tf32
    device = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(device).total_memory
    tf32.configure_tf32(split_k=0, memory_limit_gib=total/2**30)
    entry = save_model_store(model_for(np.arange(32)), tmp_path/'model',
                             sample_rows=np.arange(32), fit_metadata={})
    repository = ModelRepository([entry], workspace_bytes=0)
    with repository.acquire([0], device) as models:
        models[0].admission_fixture_ballast = torch.zeros(32*2**20//4, dtype=torch.float32, device=device)
    del models
    torch.cuda.synchronize(device)
    free, _ = torch.cuda.mem_get_info(device)
    unused = max(0, torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    available = min(free+unused, total-torch.cuda.memory_allocated(device))
    # Logical future demand forces retirement of a real owned GPU allocation;
    # no tensor fills the rest of the GPU, and no CUDA memory query is mocked.
    with repository.reserve_workspace(device, int(available)+1, reserve_bytes=0) as workspace:
        assert repository.summary()['workspace_evictions'] == 1
        options = AnalysisOptions()
        m, rank = 5120, 512
        tail = runtime._gene_tail_new_bytes(32, m, options)
        workspace.set_required(4*m*m+tail, phase='gene_covariance')
        diagonal = torch.full((m,), .5, dtype=torch.float32, device=device)
        diagonal[:rank] = torch.linspace(4, 2, rank, device=device)
        covariance = torch.diag(diagonal)
        workspace.set_required(tail, phase='gene_tail')
        q = (diagonal.double().sum()+3.2*torch.sqrt(2*diagonal.double().square().sum())).reshape(1)
        p = statistics._fastskat_hybrid_pvalues(q, covariance,
            torch.ones((m, 1), device=device), rank=rank, seed=1729, threshold=5000)
        reference = statistics._quadratic_form_sf_tensor(q[0], diagonal.double())
        assert torch.isfinite(p).all() and 0 < float(p[0]) < 1
        assert float((-torch.log10(p[0])+torch.log10(reference)).abs()) < .001
        del covariance, diagonal, q, p, reference
    assert repository.summary()['active_workspace_reservations'] == 0
