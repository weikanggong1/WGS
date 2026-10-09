"""Cache admission contracts; CUDA comparisons are component tests only."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staar_phewas._cached_covariance import (
    cached_workspace_estimate, plan_cached_workspace, score_covariance_cached,
)
from staar_phewas.null_model import GaussianNullModel, KinshipSpectrum


GIB = 2**30


def test_admission_selects_full_then_two_panels_for_same_population():
    full = plan_cached_workspace(339013, 9854, 23, free_bytes=80 * GIB)
    assert full["full_resident"]
    assert full["panel_variant_size"] == 9854
    long = plan_cached_workspace(339013, 21231, 23, free_bytes=80 * GIB)
    assert not long["full_resident"]
    assert long["panel_variant_size"] == 4096
    assert long["conservative_new_workspace_bytes"] < long["available_new_bytes"]
    # A full resident long matrix cannot be admitted even on an 80-GiB card.
    full_long = cached_workspace_estimate(
        339013, 21231, 23, variant_tile_size=4096,
        panel_variant_size=21231, full_resident=True)
    assert full_long["new_storage_bytes"] > 40 * GIB


def test_live_gpu_guard_counts_other_allocations_and_unused_reservations():
    args = (339013, 21231, 23)
    with pytest.raises(MemoryError, match="cannot admit"):
        plan_cached_workspace(*args, free_bytes=24 * GIB)
    recovered = plan_cached_workspace(
        *args, free_bytes=24 * GIB, allocated_bytes=GIB,
        reserved_bytes=4 * GIB)
    assert recovered["panel_variant_size"] == 4096
    assert recovered["available_new_bytes"] == 27 * GIB - 256 * 2**20
    # An allocator reservation never grants bytes beyond the process cap.
    with pytest.raises(MemoryError, match="cannot admit"):
        plan_cached_workspace(*args, free_bytes=80 * GIB,
                              allocated_bytes=15 * GIB, reserved_bytes=30 * GIB)


def test_explicit_panel_and_too_large_dense_matrix_rejected_before_allocation():
    with pytest.raises(MemoryError, match="requires"):
        plan_cached_workspace(339013, 21231, 23, panel_variant_size=8192,
                              free_bytes=80 * GIB)
    with pytest.raises(MemoryError, match="cannot admit"):
        plan_cached_workspace(339013, 100000, 23, free_bytes=80 * GIB)
    empty = plan_cached_workspace(33, 0, 3)
    assert empty["cache_mode"] == "empty"
    assert empty["conservative_new_workspace_bytes"] == 0


@pytest.mark.parametrize("kwargs", [
    {"variant_tile_size": True}, {"variant_tile_size": 4095},
    {"panel_variant_size": 2048}, {"memory_limit_gib": 41},
    {"memory_limit_gib": float("nan")}, {"allocated_bytes": -1},
    {"free_bytes": -1},
])
def test_invalid_cache_options_have_no_implicit_fallback(kwargs):
    with pytest.raises(ValueError):
        plan_cached_workspace(33, 100, 3, **kwargs)


def test_cached_backend_rejects_cpu_fp64_and_unaccepted_symmetry():
    model = SimpleNamespace(
        family="gaussian", n_pheno=1, use_spa=False,
        spectrum=SimpleNamespace(blocks=[]), device=torch.device("cpu"),
        matmul_mode="tf32")
    with pytest.raises(ValueError, match="CUDA model"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="native TF32"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32),
                                matmul_mode="fp64")
    with pytest.raises(ValueError, match="symmetry=average"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32),
                                symmetry="mirror")


def _cuda_model(n):
    device = torch.device("cuda:0")
    x = torch.stack((torch.ones(n), torch.arange(n).float() / n,
                     (torch.arange(n) % 7).float() / 7), dim=1).to(device)
    inverse = (1.0 + (torch.arange(n) % 5).float() / 5).to(device)
    residual = ((torch.arange(n) % 11).float() / 11 - 0.5).to(device)
    zeros = torch.zeros(3, dtype=torch.float32, device=device)
    spectrum = KinshipSpectrum(torch.ones(n, device=device), [])
    return GaussianNullModel(
        sample_ids=np.arange(n).astype(str), x=x, scaled_residuals=residual,
        coefficients=zeros, theta=zeros, precision_theta=zeros,
        fixed_effect_covariance=torch.eye(3, device=device) / n,
        spectrum=spectrum, inverse_variance=inverse,
        precision_x=inverse[:, None] * x, iterations=1, converged=True,
        matmul_mode="tf32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA component comparison")
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("m", [513, 4097, 4609])
def test_cuda_full_and_two4096_panels_match_legacy512_at_tail_boundary(layout, m):
    from staar_phewas import tf32
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("native TF32 requires Ampere or newer")
    pytest.importorskip("triton")
    tf32.configure_tf32(memory_limit_gib=40, split_k=0)
    n = 1025
    generator = np.random.default_rng(173)
    genotype = np.array(generator.integers(0, 3, size=(n, m)),
                        dtype=np.float32, order=layout)
    model = _cuda_model(n)
    baseline_u, baseline_v = model.score_covariance_tiled(
        genotype, variant_tile_size=512)
    for panel_size in (None, 4096):
        u, v, report = model.score_covariance_cached(
            genotype, panel_variant_size=panel_size, profile=True)
        assert torch.equal(u, baseline_u)
        assert torch.equal(v, baseline_v)
        assert report["preparation_tile_size"] == 512
        assert report["variant_tile_size"] == 4096
        assert report["cache_mode"] == (
            "full_original_and_weighted" if panel_size is None
            else "two_original_and_weighted_panels")
        assert report["prepared_variant_tiles"] == (m + 511) // 512
        assert report["covariance_d2h_bytes"] == 0
        assert report["max_observed_allocated_bytes"] <= 40 * GIB
        assert report["symmetry"] == "average"
        assert report["singleton_product_blocks"] > 0
        assert report["singleton_covariance_tile_size"] == 512
        del u, v
    # Validate the final column of the second panel, before returning outputs.
    bad = genotype.copy(order=layout)
    bad[-1, -1] = np.nan
    with pytest.raises(ValueError, match="genotype must be finite"):
        model.score_covariance_cached(bad, panel_variant_size=4096)
