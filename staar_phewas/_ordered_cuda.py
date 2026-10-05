"""Single-kernel, sequential float64 addition of precomputed Torch products.

SPDX-License-Identifier: GPL-3.0-only
The caller computes each product in a separate Torch operation. Each CUDA
lane then adds one output column in increasing row order, without FMA or a
parallel sum tree. This module is imported only for CUDA execution.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sequential_rows_kernel(values, output, length, width, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0.0, tl.float64)
    for row in range(length):
        value = tl.load(values + row * width + offset, offset < width, other=0.0)
        accumulator = accumulator + value
    tl.store(output + offset, accumulator, offset < width)


def ordered_rows_cuda(values):
    """Reduce rows in their original order; trailing dimensions are outputs."""
    if not values.is_cuda or values.dtype != torch.float64:
        raise ValueError("ordered CUDA addition requires a float64 CUDA tensor")
    if not values.shape[0]:
        raise ValueError("ordered CUDA addition requires at least one row")
    contiguous = values.contiguous()
    width = contiguous.numel() // contiguous.shape[0]
    result = torch.empty(values.shape[1:], dtype=values.dtype, device=values.device)
    _sequential_rows_kernel[(triton.cdiv(width, 128),)](
        contiguous, result, contiguous.shape[0], width, BLOCK=128,
        num_warps=4, enable_fp_fusion=False,
    )
    return result
