import pytest
import torch

from fudan_wgs_toolkit import sparse_numerics
from fudan_wgs_toolkit.sparse_numerics import (
    _ordered_rows, _ordered_rows_torch, reference_sparse_score_covariance,
    sparse_execution_metadata,
)


def test_execution_metadata_records_calls_not_availability(monkeypatch):
    monkeypatch.setattr(sparse_numerics, "_ordered_addition_counts",
                        {"triton_cuda": 0, "torchscript_cuda": 0, "torchscript_cpu": 0})
    monkeypatch.setattr(sparse_numerics, "_cuda_backend_disabled", None)
    assert sparse_execution_metadata()["ordered_addition_backend"] == "not_used"
    _ordered_rows(torch.ones((2, 3), dtype=torch.float64))
    metadata = sparse_execution_metadata()
    assert metadata["ordered_addition_backend"] == "torchscript"
    assert metadata["ordered_addition_call_count"] == 1
    assert metadata["ordered_addition_call_counts"] == {"triton": 0, "torchscript": 1}
    assert metadata["ordered_addition_device_call_counts"]["torchscript_cpu"] == 1
    assert metadata["ordered_addition_fallback_reason"] is None


def test_zero_genotype_reports_no_ordered_kernel(monkeypatch):
    monkeypatch.setattr(sparse_numerics, "_ordered_addition_counts",
                        {"triton_cuda": 0, "torchscript_cuda": 0, "torchscript_cpu": 0})
    g = torch.zeros((10, 2), dtype=torch.float64)
    _, _, diagnostics = reference_sparse_score_covariance(
        g, torch.ones(10), torch.ones(10), torch.ones((10, 1)), torch.ones((1, 1)),
        max_workspace_bytes=4096, return_diagnostics=True)
    assert diagnostics["ordered_addition"] == "not_used"
    assert sparse_execution_metadata()["ordered_addition_call_count"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bounded_sparse_matches_explicit_fixed_effect_projector(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    n, m = 11, 7
    dtype = torch.float64
    g = torch.arange(n*m, dtype=dtype, device=device).reshape(n, m).remainder(3)
    g[:, 0] = 0  # A zero column must contribute no score or projection.
    inv = torch.linspace(.7, 1.3, n, dtype=dtype, device=device)
    x = torch.stack((torch.ones_like(inv), torch.linspace(-1, 1, n, dtype=dtype, device=device)), dim=1)
    sx = inv[:, None] * x
    cov = torch.linalg.inv(x.T @ sx)
    residual = torch.linspace(-.3, .2, n, dtype=dtype, device=device)
    projector = torch.diag(inv) - sx @ cov @ sx.T
    u, v, info = reference_sparse_score_covariance(g, residual, inv, sx, cov,
                                                  max_workspace_bytes=4096, return_diagnostics=True)
    torch.testing.assert_close(u, g.T @ residual, rtol=0, atol=2e-14)
    torch.testing.assert_close(v, g.T @ projector @ g, rtol=0, atol=2e-14)
    assert info["column_block_size"] < m
    assert info["maximum_planned_workspace_bytes"] <= 4096
    assert torch.count_nonzero(u[:1]) == 0 and torch.count_nonzero(v[:1]) == 0


def test_workspace_rejection_precedes_sorting():
    g = torch.ones((10, 2), dtype=torch.float64)
    with pytest.raises(MemoryError, match="one ordered sparse column"):
        reference_sparse_score_covariance(g, g[:, 0], g[:, 0], g[:, :1], g[:1, :1],
                                          max_workspace_bytes=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_single_kernel_retains_sequential_addition_and_strided_tails():
    for rows, width in ((1, 1), (2, 17), (79, 130), (131, 3)):
        values = torch.randn((rows, width * 2), dtype=torch.float64, device="cuda")[:, ::2]
        actual = _ordered_rows(values)
        expected = _ordered_rows_torch(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("message", ["Triton Error [CUDA]: device kernel image is invalid", "unrelated analysis error"])
def test_only_kernel_load_errors_select_exact_torchscript(monkeypatch, message):
    from fudan_wgs_toolkit import _ordered_cuda
    monkeypatch.setattr(sparse_numerics, "_cuda_backend_disabled", None)
    monkeypatch.setattr(sparse_numerics, "_ordered_addition_counts",
                        {"triton_cuda": 0, "torchscript_cuda": 0, "torchscript_cpu": 0})
    def unavailable(*args):
        raise RuntimeError(message)
    monkeypatch.setattr(_ordered_cuda, "ordered_rows_cuda", unavailable)
    values = torch.randn((79, 17), dtype=torch.float64, device="cuda")
    if message == "unrelated analysis error":
        with pytest.raises(RuntimeError, match=message):
            _ordered_rows(values)
        assert sparse_numerics._cuda_backend_disabled is None
        assert sparse_execution_metadata()["ordered_addition_backend"] == "not_used"
    else:
        with pytest.warns(RuntimeWarning, match="same device"):
            actual = _ordered_rows(values)
        torch.testing.assert_close(actual, _ordered_rows_torch(values), rtol=0, atol=0)
        assert sparse_numerics._cuda_backend_disabled == "incompatible_cuda_kernel_image"
        execution = sparse_execution_metadata()
        assert execution["ordered_addition_backend"] == "torchscript"
        assert execution["ordered_addition_device_call_counts"]["torchscript_cuda"] == 1
        assert execution["ordered_addition_fallback_reason"] == "incompatible_cuda_kernel_image"
        _, _, info = reference_sparse_score_covariance(
            values, values[:, 0], torch.ones_like(values[:, 0]),
            torch.ones_like(values[:, :1]), values.new_ones((1, 1)),
            max_workspace_bytes=1024**2, return_diagnostics=True,
        )
        assert info["ordered_addition"] == "torchscript_sequential"
        assert info["ordered_addition_fallback_reason"] == "incompatible_cuda_kernel_image"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_missing_triton_selects_exact_torchscript(monkeypatch):
    import builtins
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "_ordered_cuda":
            raise ModuleNotFoundError("No module named triton", name="triton")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(sparse_numerics, "_cuda_backend_disabled", None)
    monkeypatch.setattr(builtins, "__import__", missing)
    values = torch.randn((79, 17), dtype=torch.float64, device="cuda")
    with pytest.warns(RuntimeWarning, match="missing_triton"):
        actual = _ordered_rows(values)
    torch.testing.assert_close(actual, _ordered_rows_torch(values), rtol=0, atol=0)
