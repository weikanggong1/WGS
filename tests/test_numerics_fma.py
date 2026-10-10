"""Integer-exact rational oracle for the FP64 FMA lane rounding contract.

These small arithmetic units are not performance or real-data benchmarks.
"""
from fractions import Fraction

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.numerics import reference_crossprod, reference_dot_execution_metadata


def rational_fma(a, b, accumulator):
    return float(Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                 + Fraction.from_float(float(accumulator)))


def reference_lanes(left, right):
    right = np.asarray(right).reshape(len(left), -1)
    output = []
    for column in range(right.shape[1]):
        lanes = np.zeros(16, dtype=np.float64)
        end = len(left) // 16 * 16
        for offset in range(0, end, 16):
            for lane in range(16):
                lanes[lane] = rational_fma(left[offset + lane], right[offset + lane, column], lanes[lane])
        if end + 8 <= len(left):
            for lane in range(8):
                lanes[lane] = rational_fma(left[end + lane], right[end + lane, column], lanes[lane])
            end += 8
        for lane in range(len(left) - end):
            lanes[lane] = rational_fma(left[end + lane], right[end + lane, column], lanes[lane])
        first = lanes[:8] + lanes[8:]
        second = first[:4] + first[4:]
        output.append((second[0] + second[1]) + (second[2] + second[3]))
    return np.asarray(output, dtype=np.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires a CUDA device and compatible Triton compiler')
@pytest.mark.parametrize('length', [1, 4, 7, 8, 12, 15, 16, 17, 24, 28, 31, 32, 33, 47, 48, 49, 64, 65])
def test_fma_all_tail_lengths_and_strided_columns(length):
    # Exact binary fractions make cancellation and fused rounding visible.
    index = np.arange(length, dtype=np.float64)
    left = np.where(index % 3 == 0, -1., 1. + 2. ** -27)
    base_right = np.where(index % 5 == 0, 1., 1. - 2. ** -27)
    right = np.stack((base_right, -base_right, base_right * (1. + 2. ** -52)), axis=1)
    expected = reference_lanes(left, right)
    a = torch.empty((length, 2), dtype=torch.float64, device='cuda')
    a[:, 0] = torch.as_tensor(left, device='cuda')
    a[:, 1] = 3.
    b = torch.empty((length, 6), dtype=torch.float64, device='cuda')
    b[:, ::2] = torch.as_tensor(right, device='cuda')
    b[:, 1::2] = 7.
    result = reference_crossprod(a[:, :1], b[:, ::2])
    np.testing.assert_array_equal(result.cpu().numpy(), expected[None, :])
    vector = reference_crossprod(a[:, :1], b[:, 0])
    np.testing.assert_array_equal(vector.cpu().numpy(), expected[:1])
    assert reference_dot_execution_metadata()['reference_dot_backend'] == 'triton_fp64_fma16'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
def test_fma_boundary_arithmetic_requires_hardware_fusion():
    left = np.ones(17, dtype=np.float64)
    right = np.zeros(17, dtype=np.float64)
    left[0], right[0] = -1., 1.
    left[-1], right[-1] = 1. + 2. ** -27, 1. - 2. ** -27
    expected = reference_lanes(left, right[:, None])
    assert expected[0] == -2. ** -54
    result = reference_crossprod(torch.as_tensor(left[:, None], device='cuda'),
                                torch.as_tensor(right, device='cuda'))
    np.testing.assert_array_equal(result.cpu().numpy(), expected)


def test_ordinary_multiple_left_columns_keep_matrix_multiplication():
    left = torch.arange(18, dtype=torch.float64).reshape(6, 3)
    right = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    torch.testing.assert_close(reference_crossprod(left, right), left.T @ right, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
@pytest.mark.parametrize('error_type', [RuntimeError, torch.OutOfMemoryError])
def test_fma_preserves_other_execution_failures(monkeypatch, error_type):
    from fudan_wgs_toolkit import _fma_cuda
    error = error_type('diagnostic execution failure')

    class FailingKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise error
            return fail

    monkeypatch.setattr(_fma_cuda, '_fma16_kernel', FailingKernel())
    before = reference_dot_execution_metadata()['reference_dot_cuda_call_count']
    left = torch.ones((2, 1), dtype=torch.float64, device='cuda')
    with pytest.raises(error_type) as captured:
        _fma_cuda.fma16_dot(left, left)
    assert captured.value is error
    assert reference_dot_execution_metadata()['reference_dot_cuda_call_count'] == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
def test_fma_reports_incompatible_toolchain_without_fallback(monkeypatch):
    from fudan_wgs_toolkit import _fma_cuda
    error = RuntimeError('Triton Error [CUDA]: device kernel image is invalid')

    class FailingKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise error
            return fail

    monkeypatch.setattr(_fma_cuda, '_fma16_kernel', FailingKernel())
    before = reference_dot_execution_metadata()['reference_dot_cuda_call_count']
    left = torch.ones((2, 1), dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError, match='compatible with the current GPU driver') as captured:
        _fma_cuda.fma16_dot(left, left)
    assert captured.value.__cause__ is error
    assert reference_dot_execution_metadata()['reference_dot_cuda_call_count'] == before
