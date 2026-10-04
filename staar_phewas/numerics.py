"""Float64 Torch reductions for the frozen reference's scalar projections.

SPDX-License-Identifier: GPL-3.0-only
The sixteen-lane fused dot and horizontal reduction reproduce the reference
BLAS vector kernel's rounding order; they never call a native BLAS library.
"""
from __future__ import annotations
import torch


def _two_sum(a, b):
    total = a + b
    part = total - a
    return total, (a - (total - part)) + (b - part)


def _two_product(a, b):
    """Dekker's exact product expansion, using separate Torch operations."""
    high = a * b
    split_a, split_b = a * 134217729., b * 134217729.
    a_high = split_a - (split_a - a)
    b_high = split_b - (split_b - b)
    a_low, b_low = a - a_high, b - b_high
    low = ((a_high * b_high - high) + a_high * b_low + a_low * b_high) + a_low * b_low
    return high, low


def _extended_sum_pair(values, errors=None):
    """Return a compensated high/low pair without rounding it to one double."""
    high = values.reshape(-1)
    low = torch.zeros_like(high) if errors is None else errors.reshape(-1)
    while high.numel() > 1:
        if high.numel() % 2:
            zero = torch.zeros(1, dtype=high.dtype, device=high.device)
            high, low = torch.cat((high, zero)), torch.cat((low, zero))
        total, error = _two_sum(high[0::2], high[1::2])
        correction = error + (low[0::2] + low[1::2])
        new = total + correction
        low, high = correction - (new - total), new
    return high.reshape(()), low.reshape(())


def _extended_divide(high, low, denominator):
    denominator_tensor = torch.full_like(high, denominator)
    quotient = high / denominator_tensor
    product, product_error = _two_product(quotient, denominator_tensor)
    correction = ((high - product) + (low - product_error)) / denominator_tensor
    return quotient + correction


def extended_variance(values):
    """Sample variance with extended subtraction, products and division.

    R's cov.c retains extended precision through the centered products and
    division by n-1. Double-double Torch intermediates prevent premature
    rounding of those operations; the output is one float64 scalar.
    """
    if values.ndim != 1 or values.numel() < 2 or values.dtype != torch.float64:
        raise ValueError("extended variance requires a float64 vector of length at least two")
    high, low = _extended_sum_pair(values)
    mean = _extended_divide(high, low, values.numel())
    centered, center_error = _two_sum(values, -mean)
    products, product_error = _two_product(centered, centered)
    product_error = product_error + (2 * centered * center_error + center_error.square())
    high, low = _extended_sum_pair(products, product_error)
    return _extended_divide(high, low, values.numel() - 1)


@torch.jit.script
def _sixteen_lane_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    acc = torch.zeros((16, a.size(1)), dtype=a.dtype, device=a.device)
    end = n // 16 * 16
    for off in range(0, end, 16):
        acc = torch.addcmul(acc, a[off:off + 16], b[off:off + 16])
    if end + 8 <= n:
        acc[:8] = torch.addcmul(acc[:8], a[end:end + 8], b[end:end + 8])
        end += 8
    if end < n:
        remaining = n - end
        acc[:remaining] = torch.addcmul(acc[:remaining], a[end:], b[end:])
    first = acc[:8] + acc[8:]
    second = first[:4] + first[4:]
    return (second[0] + second[1]) + (second[2] + second[3])


_warmed_layouts: set[tuple] = set()


def reference_crossprod(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """A'B with the frozen BLAS scalar-column reduction order.

    A has shape [samples, 1]; B is [samples] or [samples, columns]. All
    values are float64. Independent columns share a GPU launch. Other
    left-column counts use the ordinary Torch matrix multiplication.
    Strided inputs are supported. No sample count or data values are fixed.
    """
    if left.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("cross-product sample dimensions do not match")
    if left.shape[1] != 1:
        return left.T @ right
    if left.dtype != torch.float64 or right.dtype != torch.float64:
        raise ValueError("reference scalar cross-products require float64")
    vector = right.ndim == 1
    if right.ndim not in (1, 2):
        raise ValueError("cross-product right input must be a vector or matrix")
    r = right[:, None] if vector else right
    a, b = torch.broadcast_tensors(left, r)
    key = (str(a.device), a.shape, a.stride(), b.stride())
    if a.is_cuda and key not in _warmed_layouts:
        # TorchScript's first profiling executions precede CUDA fusion. Warm
        # this layout before selecting its fused multiply-add result, so the
        # first fitted model and subsequent fits have the same rounding.
        _sixteen_lane_dot(a, b)
        _sixteen_lane_dot(a, b)
        _warmed_layouts.add(key)
    result = _sixteen_lane_dot(a, b)
    return result if vector else result[None, :]
