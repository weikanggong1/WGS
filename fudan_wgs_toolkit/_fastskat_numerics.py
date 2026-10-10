"""Bounded FP64 refinement of a represented FP32 FastSKAT covariance.

The covariance construction and initial LOBPCG basis remain outside this module.
Only the final retained subspace, its dense moment ledger and deflation use
FP64.  No full M-by-M FP64 copy is made.  This helper must be invoked within the
caller's explicit spectral-refinement precision-audit boundary.
"""

from __future__ import annotations

import math
import time

import torch


class FastSKATNumericsError(ArithmeticError):
    """Numerical inconsistency; diagnostics do not contain sample identifiers."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def refine_spectrum_and_moments(weighted, basis, *, power_iterations=2,
                                block_rows=256, cancellation_threshold=1e-6):
    """Refine a retained subspace and calculate a consistent residual ledger.

    ``weighted`` is the represented, symmetric ``(M, M)`` covariance, normally
    FP32 on CUDA. ``basis`` is an ``(M, rank)`` initial subspace. Products use
    at most ``block_rows`` dense rows converted to FP64 at once. The returned
    dict contains matched descending Ritz values/vectors, ``A @ Q``, dense
    first/second moments, residual moments and guard diagnostics.

    Severe cancellation triggers direct deflation ``E=A-Q diag(top) Q.T``.
    For an orthonormal Q and its Rayleigh-Ritz B, ``trace(E)=trace(A)-trace(B)``
    and ``||E||_F^2=||A||_F^2-||B||_F^2``. Computing E by row blocks avoids
    subtracting almost equal second moments. Negative residual moments are
    checked again against this direct calculation and never clamped.

    This is numerical refinement of the same represented covariance. It does
    not repair a materially indefinite input covariance or establish the
    scientific accuracy of the FastSKAT approximation.
    """
    if weighted.ndim != 2 or weighted.shape[0] != weighted.shape[1]:
        raise ValueError("weighted must be a square matrix")
    if weighted.dtype not in (torch.float32, torch.float64):
        raise ValueError("weighted must use FP32 or FP64")
    m = int(weighted.shape[0])
    if basis.ndim != 2 or basis.shape[0] != m or basis.device != weighted.device:
        raise ValueError("basis must be an M-by-rank tensor on the matrix device")
    k = int(basis.shape[1])
    if not 0 < k < m:
        raise ValueError("rank must be positive and smaller than M")
    if type(power_iterations) is not int or power_iterations < 0:
        raise ValueError("power_iterations must be a nonnegative integer")
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    if not math.isfinite(cancellation_threshold) or not 0 < cancellation_threshold < 1:
        raise ValueError("cancellation_threshold must be finite and between zero and one")

    def synchronize():
        if weighted.is_cuda:
            torch.cuda.synchronize(weighted.device)

    def number(value):
        return float(value.detach().cpu())

    synchronize()
    began = time.perf_counter()
    metadata = {
        "refinement_dtype": "float64",
        "dense_input_dtype": str(weighted.dtype).removeprefix("torch."),
        "full_dense_fp64_materialized": False,
        "block_rows": min(block_rows, m),
        "M": m,
        "rank": k,
        "power_iterations": power_iterations,
        "blocked_matmul_calls": 0,
        "dense_reduction_blocks": 0,
        "deflation_blocks": 0,
        "direct_deflation_used": False,
        "negative_residual_clipped": False,
        "cancellation_threshold": cancellation_threshold,
        "moment_relative_failure_tolerance": 1e-5,
        "timing_scope": "synchronized host wall time; not isolated CUDA kernel time",
    }

    def fail(message, **values):
        metadata.update(values)
        synchronize()
        metadata["wall_seconds"] = time.perf_counter() - began
        raise FastSKATNumericsError(message, dict(metadata))

    def blocked_product(vectors):
        out = torch.empty((m, vectors.shape[1]), dtype=torch.float64,
                          device=weighted.device)
        for start in range(0, m, block_rows):
            stop = min(m, start + block_rows)
            block = weighted[start:stop].to(dtype=torch.float64)
            out[start:stop] = block @ vectors
            metadata["blocked_matmul_calls"] += 1
            del block
        return out

    phase = time.perf_counter()
    q = torch.linalg.qr(basis.to(dtype=torch.float64), mode="reduced").Q
    for _ in range(power_iterations):
        projected = blocked_product(q)
        q = torch.linalg.qr(projected, mode="reduced").Q
        del projected
    projected = blocked_product(q)
    ritz = q.T @ projected
    ritz = (ritz + ritz.T) * 0.5
    top, rotation = torch.linalg.eigh(ritz, UPLO="U")
    top = top.flip(0)
    rotation = rotation.flip(1)
    q = q @ rotation
    projected = projected @ rotation
    if not bool(torch.isfinite(top).all()) or not bool(torch.isfinite(q).all()):
        fail("FP64 Rayleigh-Ritz refinement produced non-finite eigenpairs")

    # Only errors at the FP64 Ritz-operation rounding scale can be treated as
    # zero. Do this before every moment calculation, so the entire ledger uses
    # precisely the same retained values. Actual negative Ritz values fail.
    spectral_scale = top.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
    top_roundoff_bound = 64 * (k + 1) * torch.finfo(torch.float64).eps * spectral_scale
    if bool(top.min() < -top_roundoff_bound):
        fail("FP64 Rayleigh-Ritz spectrum is materially negative",
             minimum_ritz_value=number(top.min()),
             negative_ritz_roundoff_bound=number(top_roundoff_bound))
    negative = top < 0
    metadata["zeroed_roundoff_ritz_count"] = int(negative.sum().detach().cpu())
    metadata["zeroed_roundoff_ritz_sum"] = number(top[negative].sum())
    metadata["negative_ritz_roundoff_bound"] = number(top_roundoff_bound)
    if bool(negative.any()):
        top = torch.where(negative, top.new_zeros(()), top)
    synchronize()
    metadata["refinement_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    trace = weighted.diagonal().to(dtype=torch.float64).sum()
    trace2 = trace.new_zeros(())
    for start in range(0, m, block_rows):
        block = weighted[start:min(m, start + block_rows)].to(dtype=torch.float64)
        trace2 += block.square().sum()
        metadata["dense_reduction_blocks"] += 1
        del block
    top_trace = top.sum()
    top_second = top.square().sum()
    residual_mean = trace - top_trace
    residual_second = trace2 - top_second
    if not bool(torch.isfinite(torch.stack((trace, trace2, residual_mean, residual_second))).all()):
        fail("FP64 dense moment ledger contains non-finite values")
    if bool(trace2 <= 0):
        fail("FastSKAT covariance has zero dense second moment")
    tiny = torch.finfo(torch.float64).tiny
    mean_fraction = residual_mean.abs() / trace.abs().clamp_min(tiny)
    second_fraction = residual_second.abs() / trace2.abs().clamp_min(tiny)
    metadata["subtracted_residual_mean"] = number(residual_mean)
    metadata["subtracted_residual_second"] = number(residual_second)
    metadata["subtracted_residual_mean_fraction"] = number(mean_fraction)
    metadata["subtracted_residual_second_fraction"] = number(second_fraction)
    synchronize()
    metadata["dense_moments_seconds"] = time.perf_counter() - phase

    direct = (bool(residual_mean < 0) or bool(residual_second < 0)
              or bool(mean_fraction <= cancellation_threshold)
              or bool(second_fraction <= cancellation_threshold)
              or metadata["zeroed_roundoff_ritz_count"] > 0)
    phase = time.perf_counter()
    if direct:
        metadata["direct_deflation_used"] = True
        stable_mean = trace.new_zeros(())
        stable_second = trace.new_zeros(())
        for start in range(0, m, block_rows):
            stop = min(m, start + block_rows)
            remainder = weighted[start:stop].to(dtype=torch.float64)
            remainder = remainder - ((q[start:stop] * top) @ q.T)
            local = torch.arange(stop - start, device=weighted.device)
            stable_mean += remainder[local, local + start].sum()
            stable_second += remainder.square().sum()
            metadata["deflation_blocks"] += 1
            del remainder
        metadata["deflation_mean_ledger_delta"] = number(stable_mean - residual_mean)
        metadata["deflation_second_ledger_delta"] = number(stable_second - residual_second)
        residual_mean, residual_second = stable_mean, stable_second
    synchronize()
    metadata["direct_deflation_seconds"] = time.perf_counter() - phase

    trace_tolerance = 1e-5 * trace.abs().clamp_min(1)
    second_tolerance = 1e-5 * trace2.abs().clamp_min(1)
    if bool(residual_mean < -trace_tolerance) or bool(residual_second < -second_tolerance):
        fail("top-k moments exceed dense trace moments after FP64 refinement",
             residual_mean=number(residual_mean), residual_second=number(residual_second),
             trace=number(trace), trace2=number(trace2),
             trace_tolerance=number(trace_tolerance),
             second_tolerance=number(second_tolerance))
    if bool(residual_mean < 0) or bool(residual_second < 0):
        fail("direct FP64 deflation retains a negative residual moment",
             residual_mean=number(residual_mean), residual_second=number(residual_second),
             trace=number(trace), trace2=number(trace2))
    if bool((residual_mean > 0) != (residual_second > 0)):
        fail("direct FP64 residual moments are inconsistent",
             residual_mean=number(residual_mean), residual_second=number(residual_second))

    phase = time.perf_counter()
    residual_norm = (torch.linalg.vector_norm(projected - q * top)
                     / torch.linalg.vector_norm(projected).clamp_min(tiny))
    orthogonality = (torch.linalg.vector_norm(q.T @ q - torch.eye(
        k, dtype=torch.float64, device=weighted.device)) / math.sqrt(k))
    if not bool(torch.isfinite(torch.stack((residual_norm, orthogonality))).all()):
        fail("FP64 spectral validation produced non-finite diagnostics")
    if bool(residual_norm > 0.25) or bool(orthogonality > 1e-2):
        fail("FP64 convergence or orthogonality guard failed",
             residual_norm=number(residual_norm), orthogonality=number(orthogonality))
    synchronize()
    metadata["validation_seconds"] = time.perf_counter() - phase
    metadata["wall_seconds"] = time.perf_counter() - began
    return {
        "top": top, "basis": q, "projected": projected,
        "trace": trace, "trace2": trace2,
        "residual_mean": residual_mean, "residual_second": residual_second,
        "residual_norm": residual_norm, "orthogonality": orthogonality,
        "metadata": metadata,
    }


def scaled_residual_cumulants(root, residual_mean_scaled,
                             residual_second_scaled, residual_scale_scaled):
    """Stable K, K' and K'' for the scaled Satterthwaite component.

    Inputs are t, mu/scale, s2/scale**2 and (s2/mu)/scale. Avoid forming a very
    large chi-square dof multiplied by a vanishing scale/logarithm. This is
    exactly the same cumulant-generating function; x=0 uses its analytic limit.
    The caller remains responsible for validating positive residual moments
    and keeping t below the component's positive pole.
    """
    x = -2 * residual_scale_scaled * root
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    ratio = torch.where(x == 0, torch.ones_like(x), torch.log1p(x) / safe_x)
    denominator = 1 + x
    cumulant = residual_mean_scaled * root * ratio
    first = residual_mean_scaled / denominator
    second = 2 * residual_second_scaled / denominator.square()
    return cumulant, first, second
