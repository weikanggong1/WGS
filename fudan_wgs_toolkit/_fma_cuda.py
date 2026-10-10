"""Explicit FP64 FMA dot products with the reference sixteen-lane order.

SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _fma16_kernel(left, right, output, N: tl.constexpr, C: tl.constexpr,
                      LEFT_ROW: tl.constexpr, RIGHT_ROW: tl.constexpr,
                      RIGHT_COL: tl.constexpr, BLOCK_COLS: tl.constexpr):
        column = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        end = N // 16 * 16
        # Independent accumulators make the final horizontal order explicit.
        acc0 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc1 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc2 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc3 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc4 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc5 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc6 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc7 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc8 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc9 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc10 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc11 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc12 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc13 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc14 = tl.full((BLOCK_COLS,), 0, tl.float64)
        acc15 = tl.full((BLOCK_COLS,), 0, tl.float64)
        for offset in range(0, end, 16):
            a0 = tl.load(left + (offset + 0) * LEFT_ROW)
            b0 = tl.load(right + (offset + 0) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc0 = libdevice.fma_rn(a0, b0, acc0)
            a1 = tl.load(left + (offset + 1) * LEFT_ROW)
            b1 = tl.load(right + (offset + 1) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc1 = libdevice.fma_rn(a1, b1, acc1)
            a2 = tl.load(left + (offset + 2) * LEFT_ROW)
            b2 = tl.load(right + (offset + 2) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc2 = libdevice.fma_rn(a2, b2, acc2)
            a3 = tl.load(left + (offset + 3) * LEFT_ROW)
            b3 = tl.load(right + (offset + 3) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc3 = libdevice.fma_rn(a3, b3, acc3)
            a4 = tl.load(left + (offset + 4) * LEFT_ROW)
            b4 = tl.load(right + (offset + 4) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc4 = libdevice.fma_rn(a4, b4, acc4)
            a5 = tl.load(left + (offset + 5) * LEFT_ROW)
            b5 = tl.load(right + (offset + 5) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc5 = libdevice.fma_rn(a5, b5, acc5)
            a6 = tl.load(left + (offset + 6) * LEFT_ROW)
            b6 = tl.load(right + (offset + 6) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc6 = libdevice.fma_rn(a6, b6, acc6)
            a7 = tl.load(left + (offset + 7) * LEFT_ROW)
            b7 = tl.load(right + (offset + 7) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc7 = libdevice.fma_rn(a7, b7, acc7)
            a8 = tl.load(left + (offset + 8) * LEFT_ROW)
            b8 = tl.load(right + (offset + 8) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc8 = libdevice.fma_rn(a8, b8, acc8)
            a9 = tl.load(left + (offset + 9) * LEFT_ROW)
            b9 = tl.load(right + (offset + 9) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc9 = libdevice.fma_rn(a9, b9, acc9)
            a10 = tl.load(left + (offset + 10) * LEFT_ROW)
            b10 = tl.load(right + (offset + 10) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc10 = libdevice.fma_rn(a10, b10, acc10)
            a11 = tl.load(left + (offset + 11) * LEFT_ROW)
            b11 = tl.load(right + (offset + 11) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc11 = libdevice.fma_rn(a11, b11, acc11)
            a12 = tl.load(left + (offset + 12) * LEFT_ROW)
            b12 = tl.load(right + (offset + 12) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc12 = libdevice.fma_rn(a12, b12, acc12)
            a13 = tl.load(left + (offset + 13) * LEFT_ROW)
            b13 = tl.load(right + (offset + 13) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc13 = libdevice.fma_rn(a13, b13, acc13)
            a14 = tl.load(left + (offset + 14) * LEFT_ROW)
            b14 = tl.load(right + (offset + 14) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc14 = libdevice.fma_rn(a14, b14, acc14)
            a15 = tl.load(left + (offset + 15) * LEFT_ROW)
            b15 = tl.load(right + (offset + 15) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc15 = libdevice.fma_rn(a15, b15, acc15)
        if end + 8 <= N:
            a0 = tl.load(left + (end + 0) * LEFT_ROW)
            b0 = tl.load(right + (end + 0) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc0 = libdevice.fma_rn(a0, b0, acc0)
            a1 = tl.load(left + (end + 1) * LEFT_ROW)
            b1 = tl.load(right + (end + 1) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc1 = libdevice.fma_rn(a1, b1, acc1)
            a2 = tl.load(left + (end + 2) * LEFT_ROW)
            b2 = tl.load(right + (end + 2) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc2 = libdevice.fma_rn(a2, b2, acc2)
            a3 = tl.load(left + (end + 3) * LEFT_ROW)
            b3 = tl.load(right + (end + 3) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc3 = libdevice.fma_rn(a3, b3, acc3)
            a4 = tl.load(left + (end + 4) * LEFT_ROW)
            b4 = tl.load(right + (end + 4) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc4 = libdevice.fma_rn(a4, b4, acc4)
            a5 = tl.load(left + (end + 5) * LEFT_ROW)
            b5 = tl.load(right + (end + 5) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc5 = libdevice.fma_rn(a5, b5, acc5)
            a6 = tl.load(left + (end + 6) * LEFT_ROW)
            b6 = tl.load(right + (end + 6) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc6 = libdevice.fma_rn(a6, b6, acc6)
            a7 = tl.load(left + (end + 7) * LEFT_ROW)
            b7 = tl.load(right + (end + 7) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
            acc7 = libdevice.fma_rn(a7, b7, acc7)
            end += 8
        if end < N:
            remaining = N - end
            if remaining > 0:
                a0 = tl.load(left + (end + 0) * LEFT_ROW)
                b0 = tl.load(right + (end + 0) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc0 = libdevice.fma_rn(a0, b0, acc0)
            if remaining > 1:
                a1 = tl.load(left + (end + 1) * LEFT_ROW)
                b1 = tl.load(right + (end + 1) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc1 = libdevice.fma_rn(a1, b1, acc1)
            if remaining > 2:
                a2 = tl.load(left + (end + 2) * LEFT_ROW)
                b2 = tl.load(right + (end + 2) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc2 = libdevice.fma_rn(a2, b2, acc2)
            if remaining > 3:
                a3 = tl.load(left + (end + 3) * LEFT_ROW)
                b3 = tl.load(right + (end + 3) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc3 = libdevice.fma_rn(a3, b3, acc3)
            if remaining > 4:
                a4 = tl.load(left + (end + 4) * LEFT_ROW)
                b4 = tl.load(right + (end + 4) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc4 = libdevice.fma_rn(a4, b4, acc4)
            if remaining > 5:
                a5 = tl.load(left + (end + 5) * LEFT_ROW)
                b5 = tl.load(right + (end + 5) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc5 = libdevice.fma_rn(a5, b5, acc5)
            if remaining > 6:
                a6 = tl.load(left + (end + 6) * LEFT_ROW)
                b6 = tl.load(right + (end + 6) * RIGHT_ROW + column * RIGHT_COL, mask=column < C, other=0)
                acc6 = libdevice.fma_rn(a6, b6, acc6)
        first0 = acc0 + acc8
        first1 = acc1 + acc9
        first2 = acc2 + acc10
        first3 = acc3 + acc11
        first4 = acc4 + acc12
        first5 = acc5 + acc13
        first6 = acc6 + acc14
        first7 = acc7 + acc15
        second0 = first0 + first4
        second1 = first1 + first5
        second2 = first2 + first6
        second3 = first3 + first7
        result = (second0 + second1) + (second2 + second3)
        tl.store(output + column, result, mask=column < C)


_successful_calls = 0


def fma16_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return one dot per right column, on CUDA, with no compiler-fusion dependence.

    left is [samples, 1], right is [samples, columns], both float64 on the
    same CUDA device. Strided views are supported. Incompatible Triton/CUDA
    compilation raises an explicit error; a different rounding algorithm is
    never substituted for a successful reference dot.
    """
    global _successful_calls
    if triton is None:
        raise RuntimeError('Reference FP64 FMA dots require Triton >= 3.0 from a compatible CUDA PyTorch installation')
    if not left.is_cuda or right.device != left.device:
        raise ValueError('FMA16 inputs must use the same CUDA device')
    if left.dtype != torch.float64 or right.dtype != torch.float64:
        raise ValueError('FMA16 inputs must use float64')
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 1 or left.shape[0] != right.shape[0]:
        raise ValueError('FMA16 requires samples-by-one and samples-by-columns arrays')
    result = torch.empty(right.shape[1], dtype=left.dtype, device=left.device)
    if right.shape[1] == 0:
        return result
    try:
        with torch.cuda.device(left.device):
            _fma16_kernel[(triton.cdiv(right.shape[1], 4),)](
                left, right, result, left.shape[0], right.shape[1], left.stride(0),
                right.stride(0), right.stride(1), BLOCK_COLS=4,
                num_warps=4, enable_fp_fusion=False)
    except RuntimeError as exc:
        detail = str(exc).lower()
        incompatible = ('device kernel image is invalid', 'invalid device function',
                        'unsupported ptx version', 'unsupported toolchain',
                        'cuda_error_invalid_image', 'cuda_error_unsupported_ptx_version',
                        'no kernel image is available')
        if any(message in detail for message in incompatible):
            raise RuntimeError('Reference FP64 FMA dots require a Triton >= 3.0 CUDA compiler compatible with the current GPU driver') from exc
        raise
    _successful_calls += 1
    return result


def fma16_execution_metadata() -> dict:
    """Count successful explicit FMA kernel launches in this process."""
    return {'reference_dot_backend': 'triton_fp64_fma16' if _successful_calls else 'not_used',
            'reference_dot_cuda_call_count': _successful_calls}
