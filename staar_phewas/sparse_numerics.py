"""Bounded float64 Torch reductions in the reference sparse product order.

SPDX-License-Identifier: GPL-3.0-only
Algorithmic provenance: STAAR contributors, STAAR_O_SMMAT_sparse.cpp,
https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05
and Armadillo's sparse matrix multiplication: increasing column/row indices.
Products are computed in Torch; CUDA row additions use one Triton kernel.
CPU row additions use TorchScript. There are no R-runtime calls.
"""
from __future__ import annotations

import torch
import warnings


_cuda_backend_disabled = None
_ordered_addition_counts = {"triton_cuda": 0, "torchscript_cuda": 0, "torchscript_cpu": 0}


def sparse_execution_metadata():
    """Return actual successful ordered-addition calls in this process.

    Importing the module or enabling Triton does not count as execution.
    Counters are cumulative, require no device synchronization and never
    change a numerical result. A fallback reason is reported only after
    TorchScript has actually run on CUDA following a recognized load failure.
    """
    counts = dict(_ordered_addition_counts)
    triton_calls = counts["triton_cuda"]
    torchscript_calls = counts["torchscript_cuda"] + counts["torchscript_cpu"]
    backend = ("mixed" if triton_calls and torchscript_calls else
               "triton" if triton_calls else "torchscript" if torchscript_calls else "not_used")
    return {"ordered_addition_backend": backend,
            "ordered_addition_call_count": triton_calls + torchscript_calls,
            "ordered_addition_call_counts": {"triton": triton_calls, "torchscript": torchscript_calls},
            "ordered_addition_device_call_counts": counts,
            "ordered_addition_fallback_reason": _cuda_backend_disabled if counts["torchscript_cuda"] else None}


@torch.jit.script
def _ordered_rows_torch(values: torch.Tensor) -> torch.Tensor:
    accumulator = torch.zeros_like(values[0])
    for row in range(values.size(0)):
        accumulator = accumulator + values[row]
    return accumulator


def _ordered_rows(values):
    global _cuda_backend_disabled
    if values.is_cuda and _cuda_backend_disabled is None:
        try:
            from ._ordered_cuda import ordered_rows_cuda
            result = ordered_rows_cuda(values)
            _ordered_addition_counts["triton_cuda"] += 1
            return result
        except ModuleNotFoundError as error:
            if error.name is None or not (error.name == "triton" or error.name.startswith("triton.")):
                raise
            _cuda_backend_disabled = "missing_triton"
        except RuntimeError as error:
            # Only recognized kernel/toolchain load failures select another
            # implementation. Analysis, allocation and CUDA execution errors
            # must remain visible instead of being swallowed by a fallback.
            message = str(error).lower()
            if not any(fragment in message for fragment in (
                "device kernel image is invalid", "cuda_error_invalid_image",
                "ptx was compiled with an unsupported toolchain",
            )):
                raise
            _cuda_backend_disabled = "incompatible_cuda_kernel_image"
        warnings.warn(
            "Triton ordered addition is unavailable; using float64 sequential "
            "TorchScript additions on the same device (" + _cuda_backend_disabled + ").",
            RuntimeWarning, stacklevel=2,
        )
    result = _ordered_rows_torch(values)
    _ordered_addition_counts["torchscript_cuda" if values.is_cuda else "torchscript_cpu"] += 1
    return result


@torch.no_grad()
def reference_sparse_score_covariance(
    genotype, residual, inverse_variance, precision_covariates,
    fixed_effect_covariance, *, max_workspace_bytes=256 * 1024**2,
    return_diagnostics=False,
):
    """Return U and V for diagonal precision, retaining sparse sum order.

    Genotype has shape [samples, variants], residual/inverse_variance have
    shape [samples], precision_covariates has shape [samples, fixed_effects],
    and fixed_effect_covariance is [fixed_effects, fixed_effects]. The inverse
    variance is diagonal Sigma_i. Covariance pairs and covariate features are
    processed in blocks; no max_nonzeros*variants*variants tensor is built.

    max_workspace_bytes bounds planned temporary workspace; the inputs,
    [variants, variants] result and [fixed_effects, variants] cross-product
    are separate. CUDA allocator reservations can retain earlier allocations.
    This function does not handle off-diagonal precision or choose a fallback.
    """
    g = torch.as_tensor(genotype, dtype=torch.float64)
    if g.ndim != 2 or not g.shape[0] or not g.shape[1]:
        raise ValueError("genotype must be nonempty samples-by-variants")
    r, inv, sx, cov = [torch.as_tensor(value, dtype=torch.float64, device=g.device)
                       for value in (residual, inverse_variance, precision_covariates,
                                     fixed_effect_covariance)]
    n, m = g.shape
    if r.shape != (n,) or inv.shape != (n,) or sx.ndim != 2 or sx.shape[0] != n:
        raise ValueError("residual, precision and fixed-effect rows must match genotype")
    p = sx.shape[1]
    if not p or cov.shape != (p, p):
        raise ValueError("fixed-effect covariance dimensions do not match")
    if any(not bool(torch.isfinite(value).all()) for value in (g, r, inv, sx, cov)):
        raise ValueError("ordered score inputs must contain only finite values")
    budget = int(max_workspace_bytes)
    # Conservative allowance for row keys, indices and CUDA sorting workspace.
    # Choose columns for the worst case length=n before allocating row keys.
    available = budget - 8 * n
    column_block = min(32, m, available // (144 * n + 64))
    if column_block < 1:
        raise MemoryError("workspace cannot hold one ordered sparse column")
    rows = torch.arange(n, device=g.device)[:, None]
    score, cross = g.new_zeros(m), g.new_zeros((p, m))
    covariance = g.new_zeros((m, m))
    calls_before = dict(_ordered_addition_counts)
    diagnostics = {"execution": "reference_sparse", "dtype": "float64",
                   "ordered_addition": "not_used",
                   "workspace_budget_bytes": budget, "maximum_planned_workspace_bytes": 0,
                   "column_block_size": column_block, "pair_row_block_size_max": 0,
                   "maximum_nonzero_rows": 0, "covariate_count": p}

    def packed(first, last):
        nonzero = g[:, first:last] != 0
        counts = nonzero.sum(dim=0)
        length = int(counts.max())
        if not length:
            return None
        keys = torch.where(nonzero, rows, rows + n)
        # A clone releases the full samples-by-block sorted index storage.
        full_indices = torch.argsort(keys, dim=0)
        indices = full_indices[:length].clone()
        del full_indices, keys, nonzero
        valid = torch.arange(length, device=g.device)[:, None] < counts[None, :]
        columns = torch.arange(first, last, device=g.device)[None, :]
        values = torch.where(valid, g[indices, columns], 0.)
        width = last - first
        packing = 8 * n + 48 * length * width
        sort_allowance = 8 * n + 96 * n * width
        remaining = budget - packing
        pair_block = remaining // (32 * length * width + 16 * width)
        if pair_block < 1:
            raise MemoryError("workspace cannot hold an ordered covariance pair")
        diagnostics["maximum_nonzero_rows"] = max(diagnostics["maximum_nonzero_rows"], length)
        diagnostics["maximum_planned_workspace_bytes"] = max(
            diagnostics["maximum_planned_workspace_bytes"], sort_allowance)
        return indices, values, length, packing, pair_block

    # First obtain every cross-product before projecting covariance blocks.
    for first in range(0, m, column_block):
        last = min(first + column_block, m)
        data = packed(first, last)
        if data is None:
            continue
        indices, values, length, packing, width = data
        score[first:last] = _ordered_rows(r[indices] * values)
        feature_block = min(p, width)
        for feature in range(0, p, feature_block):
            end = min(feature + feature_block, p)
            products = sx[indices, feature:end] * values[:, :, None]
            cross[feature:end, first:last] = _ordered_rows(products).T
            diagnostics["maximum_planned_workspace_bytes"] = max(
                diagnostics["maximum_planned_workspace_bytes"],
                packing + (32 * length * (last - first) + 16 * (last - first)) * (end - feature))
            del products
        del data, indices, values

    for first in range(0, m, column_block):
        last = min(first + column_block, m)
        data = packed(first, last)
        if data is None:
            continue
        indices, values, length, packing, width = data
        pair_block = min(m, width)
        diagnostics["pair_row_block_size_max"] = max(diagnostics["pair_row_block_size_max"], pair_block)
        for other in range(0, m, pair_block):
            end = min(other + pair_block, m)
            weighted = inv[indices, None] * g[indices, other:end]
            products = weighted * values[:, :, None]
            del weighted
            raw = _ordered_rows(products).T
            del products
            if p == 1:
                projection = ((cross[:, other:end].T * cov[0, 0]) * cross[:, first:last])
            else:
                projection = (cross[:, other:end].T @ cov) @ cross[:, first:last]
            covariance[other:end, first:last] = raw - projection
            diagnostics["maximum_planned_workspace_bytes"] = max(
                diagnostics["maximum_planned_workspace_bytes"],
                packing + (32 * length * (last - first) + 16 * (last - first)) * (end - other))
            del raw, projection
        del data, indices, values
    triton_calls = _ordered_addition_counts["triton_cuda"] - calls_before["triton_cuda"]
    torchscript_cuda_calls = _ordered_addition_counts["torchscript_cuda"] - calls_before["torchscript_cuda"]
    torchscript_calls = torchscript_cuda_calls + _ordered_addition_counts["torchscript_cpu"] - calls_before["torchscript_cpu"]
    diagnostics["ordered_addition"] = ("mixed_sequential" if triton_calls and torchscript_calls else
                                      "triton_sequential" if triton_calls else
                                      "torchscript_sequential" if torchscript_calls else "not_used")
    diagnostics["ordered_addition_call_counts"] = {"triton": triton_calls, "torchscript": torchscript_calls}
    if torchscript_cuda_calls and _cuda_backend_disabled is not None:
        diagnostics["ordered_addition_fallback_reason"] = _cuda_backend_disabled
    if return_diagnostics:
        return score, covariance, diagnostics
    return score, covariance
