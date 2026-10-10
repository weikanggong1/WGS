"""CPU-only bounds for native diagonal-Gaussian mask workspace admission."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staar_phewas.binary_null import BinaryNullModel
from staar_phewas.null_model import GaussianNullModel, KinshipSpectrum
from staar_phewas.pipeline import AnalysisOptions, PheWASPipeline
from staar_phewas.phewas_runtime.mask_limit import LimitedMaskPipeline


def gaussian_model(*, n=64, mode="tf32", rotations=False, n_pheno=1):
    dtype = torch.float32 if mode == "tf32" else torch.float64
    design = torch.ones((1, 3), dtype=dtype).expand(n, 3)
    inverse = torch.ones(n, dtype=dtype)
    blocks = [(torch.tensor([0, 1]), torch.eye(2, dtype=dtype))] if rotations else []
    return GaussianNullModel(sample_ids=np.arange(n), x=design,
        scaled_residuals=torch.zeros(n, dtype=dtype), coefficients=torch.zeros(3, dtype=dtype),
        theta=torch.ones(2, dtype=dtype), precision_theta=torch.ones(2, dtype=dtype),
        fixed_effect_covariance=torch.eye(3, dtype=dtype),
        spectrum=KinshipSpectrum(inverse, blocks), inverse_variance=inverse,
        precision_x=design, iterations=0, converged=True, n_pheno=n_pheno,
        matmul_mode=mode)


def binary_model():
    n = 64
    design = torch.ones((1, 3)).expand(n, 3)
    return BinaryNullModel(sample_ids=np.arange(n), x=design,
        scaled_residuals=torch.zeros(n), fitted_probability=torch.full((n,), .5),
        xw=design.T, projection_left=design, fixed_effect_covariance=torch.eye(3),
        precision=torch.ones(n), precision_x=design, use_spa=False, matmul_mode="tf32")


def mask_pipeline(model, *, budget=20):
    pipeline = LimitedMaskPipeline.__new__(LimitedMaskPipeline)
    pipeline.models = [model]
    pipeline.options = AnalysisOptions(memory_limit_gib=budget, wrapper_semantics="base")
    pipeline.maximum_mask_variants = 4999
    pipeline._bounded_resident_workspace = True
    return pipeline


@pytest.mark.parametrize("m", [1, 63, 128, 192, 193, 4096, 4999, 5000])
def test_native_diagonal_gaussian_mask_bound_keeps_original_non_genotype_terms(m):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    original = PheWASPipeline._workspace_estimate(pipeline, model, m)
    estimate = pipeline._workspace_estimate(model, m)
    block_columns = min(m, pipeline.options.genotype_block_size)
    phase_bound = max(8 * model.n * m, 4 * model.n * m + 6 * model.n * block_columns)
    assert original == 4 * (4 * model.n * m + 6 * m * m + model.n)
    assert estimate == phase_bound + 4 * (6 * m * m + model.n)
    assert original - estimate == 16 * model.n * m - phase_bound


@pytest.mark.parametrize("block_size", [1, 32, 128, 512])
@pytest.mark.parametrize("m", [17, 127, 193, 4999])
def test_mask_workspace_covers_decode_and_score_phases_for_configured_block_size(block_size, m):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    pipeline.options = AnalysisOptions(genotype_block_size=block_size, wrapper_semantics="base")
    estimate = pipeline._workspace_estimate(model, m)
    unchanged_terms = 24 * m * m + 4 * model.n
    assert estimate - unchanged_terms == max(8 * model.n * m,
        4 * model.n * m + 6 * model.n * min(m, block_size))
    assert estimate >= 8 * model.n * m + unchanged_terms
    assert estimate >= 4 * model.n * m + 6 * model.n * min(m, block_size) + unchanged_terms


@pytest.mark.parametrize("route", ["fp64", "rotation", "individual", "binary", "unknown", "joint"])
def test_mask_workspace_optimization_leaves_other_routes_unchanged(route):
    model = gaussian_model(mode="fp64" if route == "fp64" else "tf32",
                           rotations=route == "rotation", n_pheno=2 if route == "joint" else 1)
    if route == "binary":
        model = binary_model()
    elif route == "unknown":
        model = SimpleNamespace(n=64, n_pheno=1, matmul_mode="tf32", device="cpu")
    pipeline = mask_pipeline(model)
    individual = route == "individual"
    original = PheWASPipeline._workspace_estimate(pipeline, model, 127, individual=individual)
    assert pipeline._workspace_estimate(model, 127, individual=individual) == original


@pytest.mark.parametrize("route", ["host", "zero_m", "long_m"])
def test_workspace_reduction_requires_known_resident_scope_and_small_actual_m(route):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    m = 127
    if route == "host":
        pipeline._bounded_resident_workspace = False
    elif route == "zero_m":
        m = 0
    else:
        m = 5001
    assert pipeline._workspace_estimate(model, m) == PheWASPipeline._workspace_estimate(pipeline, model, m)


@pytest.mark.parametrize("limit", [None, 4999, 5000, 10000])
@pytest.mark.parametrize("m", [127, 4999, 5000])
def test_resident_small_mask_budget_is_independent_of_optional_mask_cap(limit, m):
    model = gaussian_model(n=340_000)
    pipeline = mask_pipeline(model)
    pipeline.maximum_mask_variants = limit
    original = PheWASPipeline._workspace_estimate(pipeline, model, m)
    revised = pipeline._workspace_estimate(model, m)
    assert revised < original
    if m >= 4999:
        assert revised < 20 * 2**30 < original
    pipeline._limit(model, m)


def test_unlimited_configuration_delegates_every_mask_without_count_prefilter(monkeypatch):
    pipeline = mask_pipeline(gaussian_model())
    pipeline.maximum_mask_variants = None
    masks = [np.arange(2), np.arange(5001), np.arange(10001)]
    marker = object()
    def complete(self, actual):
        assert actual == masks
        return marker
    monkeypatch.setattr(PheWASPipeline, "_run_mask_sets", complete)
    assert pipeline._run_mask_sets(masks) is marker


@pytest.mark.parametrize("field", ["x", "precision_x", "inverse_variance", "fixed_effect_covariance",
                                   "scaled_residuals"])
@pytest.mark.parametrize("state", ["requires_grad", "missing", "float64"])
def test_mask_workspace_keeps_original_bound_for_unsupported_fitted_tensor(field, state):
    model = gaussian_model()
    if state == "requires_grad":
        setattr(model, field, getattr(model, field).clone().requires_grad_(True))
    elif state == "missing":
        delattr(model, field)
    else:
        setattr(model, field, getattr(model, field).double())
    pipeline = mask_pipeline(model)
    assert pipeline._workspace_estimate(model, 127) == PheWASPipeline._workspace_estimate(pipeline, model, 127)


@pytest.mark.parametrize("method", ["_materialize_resident_gene", "_calculate_local_union"])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_resident_workspace_scope_restores_previous_flag_after_return_and_exception(monkeypatch, method,
                                                                                  resident, previous, fail):
    pipeline = mask_pipeline(gaussian_model())
    pipeline._bounded_resident_workspace = previous
    prepared = {"_resident_blocks": []} if resident else {"_genotype_host": None}
    sentinel = object()
    observed = []
    def mature_method(self, *args):
        observed.append(self._bounded_resident_workspace)
        if fail:
            raise ArithmeticError("materialization failed")
        return sentinel
    monkeypatch.setattr(PheWASPipeline, method, mature_method)
    arguments = (prepared,) if method == "_materialize_resident_gene" else ([], prepared)
    if fail:
        with pytest.raises(ArithmeticError, match="materialization failed"):
            getattr(pipeline, method)(*arguments)
    else:
        assert getattr(pipeline, method)(*arguments) is sentinel
    assert observed == [resident]
    assert pipeline._bounded_resident_workspace is previous


def test_large_eligible_mask_estimate_fits_twenty_gib_without_large_dense_allocation(monkeypatch):
    model = gaussian_model(n=340_000)
    pipeline = mask_pipeline(model)
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU admission must not inspect or allocate CUDA")
    monkeypatch.setattr(torch.cuda, "memory_allocated", no_cuda)
    original = PheWASPipeline._workspace_estimate(pipeline, model, 4999)
    revised = pipeline._workspace_estimate(model, 4999)
    assert original > 20 * 2**30
    assert revised < 20 * 2**30
    assert model.x.untyped_storage().nbytes() == 3 * 4
    pipeline._limit(model, 4999)
    assert pipeline.memory_guard_max_estimated_bytes == revised


def test_mask_budget_still_includes_batch_reserve_and_rejects_one_byte_over_limit():
    model = gaussian_model(n=340_000)
    pipeline = mask_pipeline(model)
    estimate = pipeline._workspace_estimate(model, 4999)
    pipeline._batch_workspace_reserve = 20 * 2**30 - estimate
    pipeline._limit(model, 4999)
    assert pipeline.memory_guard_max_estimated_bytes == 20 * 2**30
    pipeline._batch_workspace_reserve += 1
    with pytest.raises(MemoryError, match="exceeding configured memory budget"):
        pipeline._limit(model, 4999)


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_small_mask_host_recovery_uses_twenty_gib_admission_and_restores_scope(monkeypatch, previous, fail):
    model = gaussian_model(n=340_000)
    pipeline = mask_pipeline(model)
    pipeline._bounded_resident_workspace = previous
    # Broadcast input supplies the real dimensions without allocating a
    # multi-gigabyte test matrix; admission is independent of dosage values.
    host = np.broadcast_to(np.float32(0), (model.n, 4999))
    observed = []
    sentinel = object()

    def original_host_products(self, fitted, genotype):
        observed.append(self._bounded_resident_workspace)
        assert fitted is model and genotype is host
        assert PheWASPipeline._workspace_estimate(self, fitted, 4999) > 20 * 2**30
        self._limit(fitted, 4999)
        if fail:
            raise ArithmeticError("host recovery failed")
        return sentinel

    monkeypatch.setattr(PheWASPipeline, "_hybrid_host_products", original_host_products)
    if fail:
        with pytest.raises(ArithmeticError, match="host recovery failed"):
            pipeline._hybrid_host_products(model, host)
    else:
        assert pipeline._hybrid_host_products(model, host) is sentinel
    assert observed == [True]
    assert pipeline._bounded_resident_workspace is previous


def test_host_recovery_retains_conservative_bound_for_gradient_input(monkeypatch):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    host = torch.zeros((model.n, 31), requires_grad=True)

    def original_host_products(self, fitted, genotype):
        assert genotype is host
        assert self._workspace_estimate(fitted, 31) == PheWASPipeline._workspace_estimate(self, fitted, 31)

    monkeypatch.setattr(PheWASPipeline, "_hybrid_host_products", original_host_products)
    pipeline._hybrid_host_products(model, host)


@pytest.mark.parametrize("route", ["sample_block", "configured_long_mask"])
def test_workspace_override_preserves_explicit_new_main_backends(route):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    options = dict(wrapper_semantics="base")
    if route == "sample_block":
        options["sample_block_size"] = 8
    else:
        options.update(long_mask_threshold=16, covariance_backend="legacy")
    pipeline.options = AnalysisOptions(**options)
    # Special backends budget their own tiles rather than four whole Gs.
    # Applying the ordinary two-G correction to that budget would underflow.
    expected = PheWASPipeline._workspace_estimate(pipeline, model, 31)
    assert expected > 0
    assert pipeline._workspace_estimate(model, 31) == expected


def test_small_mask_decode_peak_rejects_budget_that_only_covers_two_g_score_phase():
    model = gaussian_model()
    m = 63
    score_phase = 8 * model.n * m + 24 * m * m + 4 * model.n
    pipeline = mask_pipeline(model, budget=(score_phase + 1) / 2**30)
    assert pipeline._workspace_estimate(model, m) > score_phase + 1
    with pytest.raises(MemoryError, match="exceeding configured memory budget"):
        pipeline._limit(model, m)


def test_twenty_gib_mask_budget_keeps_existing_resident_allocation_query(monkeypatch):
    model = gaussian_model(n=340_000)
    pipeline = mask_pipeline(model)
    resident = {"bytes": 6 * 2**30}
    queries = []
    monkeypatch.setattr(GaussianNullModel, "device", property(lambda self: torch.device("cuda:0")))
    def allocated(device):
        queries.append(str(device))
        return resident["bytes"]
    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated)
    estimate = pipeline._workspace_estimate(model, 4999)
    pipeline._limit(model, 4999)
    assert pipeline.memory_guard_max_estimated_bytes == estimate + 6 * 2**30
    resident["bytes"] = 7 * 2**30
    with pytest.raises(MemoryError, match="exceeding configured memory budget"):
        pipeline._limit(model, 4999)
    assert queries == ["cuda:0", "cuda:0"]


def test_strict_eligible_m_limit_is_independent_of_workspace_reduction(monkeypatch):
    model = gaussian_model()
    pipeline = mask_pipeline(model)
    pipeline.union_rows = np.arange(model.n)
    pipeline.trait_rows = [np.arange(model.n)]
    pipeline.skipped_sets = []
    visited = []
    class RareBlock:
        def __init__(self, count):
            self.union_ref_af = np.full(count, .995)
        def trait_summary(self, *args, **kwargs):
            return (np.full(len(self.union_ref_af), .005),)
    def minor_blocks(indices, samples, **options):
        assert options["resident"] is True
        yield RareBlock(len(indices))
    def mature_mask_core(self, sets):
        visited.extend(len(indices) for indices in sets)
        return [[{"eligible_M": len(indices)}] for indices in sets]
    monkeypatch.setattr(pipeline, "_minor_blocks", minor_blocks)
    monkeypatch.setattr(PheWASPipeline, "_run_mask_sets", mature_mask_core)
    result = pipeline._run_mask_sets([np.arange(4999), np.arange(5000)])
    assert visited == [4999]
    assert result == [[{"eligible_M": 4999}], [None]]
    assert pipeline.skipped_sets == [{"mask_position": 1, "eligible_variants_lower_bound": 5000,
                                     "reason": "eligible_M_exceeds_configured_limit"}]
