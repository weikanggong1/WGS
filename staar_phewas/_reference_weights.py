"""Original weight definitions: native FP32 device and explicit FP64 control.

The weight API consumes a complete PHRED matrix unchanged.  Complementary PHRED
construction is a separate helper for the raw annotation reader.  Constants are
the binary doubles exported from the locked original R 3.6.1 implementation.
"""
from __future__ import annotations

import math
import threading
import time
from typing import MutableMapping, Optional

import numpy as np
import torch

R_LBETA_1_25 = -3.218875824868201
R_LBETA_HALF = 1.1447298858494004
_COUNTER_LOCK = threading.Lock()
_COUNTERS = {
    'native_weight_calls': 0,
    'weight_calls': 0,
    'weight_rows': 0,
    'weight_annotation_cells': 0,
    'weight_d2h_calls': 0,
    'weight_d2h_bytes': 0,
    'weight_h2d_calls': 0,
    'weight_h2d_bytes': 0,
    'weight_conversion_wall_seconds': 0.0,
    'weight_scalar_transform_seconds': 0.0,
    'complementary_phred_calls': 0,
    'complementary_phred_cells': 0,
    'complementary_phred_d2h_calls': 0,
    'complementary_phred_d2h_bytes': 0,
    'complementary_phred_h2d_calls': 0,
    'complementary_phred_h2d_bytes': 0,
    'complementary_phred_wall_seconds': 0.0,
}


def _record(**increments):
    with _COUNTER_LOCK:
        for key, increment in increments.items():
            _COUNTERS[key] += increment


def reference_weights_execution_metadata(*, reset=False):
    """Read successful transform counts without exposing any input values.

    Conversion times measure host wall time, including validation and implicit
    synchronization/copies.  Byte counts describe logical float64 payloads;
    they are not hardware bus measurements.  Reset is intended for an explicit
    new execution boundary, rather than for concurrent per-job measurement.
    """
    with _COUNTER_LOCK:
        counters = dict(_COUNTERS)
        if reset:
            for key in _COUNTERS:
                _COUNTERS[key] = 0.0 if key.endswith('_seconds') else 0
    return {
        'implementation': ('device_fp32_native' if counters['native_weight_calls'] else 'scalar_libm_float64_control'),
        'arithmetic_reference': ('original STAAR weight definitions; native FP32 arithmetic' if counters['native_weight_calls'] else 'R 3.6.1 STAAR 0.9.9 explicit control'),
        'native_timing_scope': 'asynchronous device transformations; CPU scalar timing counters exclude native calls',
        'timing_scope': 'host wall time includes validation, implicit synchronization and copies',
        'transfer_bytes_scope': 'logical float64 payloads',
        **counters,
    }


def _host_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def reference_complementary_phred(values):
    """Return R's -10*log10(1-10^(-raw_phred/10)) in the original order.

    This function is used only when deriving a complementary annotation from raw
    PHRED.  It must not replace a complementary column supplied by the caller.
    A Tensor input returns a float64 Tensor on the same device; NumPy input
    returns a NumPy array.  Zero maps to positive infinity, as in R.
    """
    start = time.perf_counter()
    raw = _host_array(values)
    result = np.empty(raw.shape, dtype=np.float64)
    for index, value in enumerate(raw.flat):
        value = float(value)
        if math.isnan(value):
            result.flat[index] = value + 0.0
            continue
        try:
            power = math.pow(10.0, -value / 10.0)
        except OverflowError:
            result.flat[index] = math.nan
            continue
        complement = 1.0 - power
        result.flat[index] = (
            math.inf if complement == 0.0
            else math.nan if complement < 0.0
            else -10.0 * math.log10(complement)
        )
    cuda_input = isinstance(values, torch.Tensor) and values.device.type == 'cuda'
    if isinstance(values, torch.Tensor):
        result = torch.as_tensor(result, dtype=torch.float64, device=values.device)
    _record(
        complementary_phred_calls=1,
        complementary_phred_cells=raw.size,
        complementary_phred_d2h_calls=int(cuda_input),
        complementary_phred_d2h_bytes=raw.size * 8 if cuda_input else 0,
        complementary_phred_h2d_calls=int(cuda_input),
        complementary_phred_h2d_bytes=raw.size * 8 if cuda_input else 0,
        complementary_phred_wall_seconds=time.perf_counter() - start,
    )
    return result


def reference_annotation_weights(
    maf,
    annotations=None,
    *,
    device=None,
    profile: Optional[MutableMapping] = None,
):
    """Return (Burden, SKAT, ACAT-V) weights, preserving row/column order.

    ``maf`` is an m-vector; ``annotations`` is an m-by-k complete PHRED matrix.
    Output matrices are m-by-2*(k+1): Beta(1,25) base and its annotation columns,
    then Beta(1,1) base and its annotation columns.  Tensor input selects its
    device by default.  Scalar host libm transformations reproduce the original
    R arithmetic; score/covariance, eigensolver and p-values are outside this API.

    Optional profile records synchronous boundaries for this call only.  These
    timings are local weight conversion timings, not pipeline benchmark times.
    """
    target = torch.device(device) if device is not None else (
        maf.device if isinstance(maf, torch.Tensor) else torch.device("cpu")
    )
    cuda_target = target.type == "cuda"
    start = time.perf_counter()
    frequencies = _host_array(maf)
    if frequencies.ndim != 1 or not frequencies.size:
        raise ValueError("maf must be a nonempty vector")
    phred = (
        np.empty((frequencies.size, 0), dtype=np.float64)
        if annotations is None else _host_array(annotations)
    )
    if phred.ndim != 2 or phred.shape[0] != frequencies.size:
        raise ValueError("annotations must be an m-by-k complete PHRED matrix")
    if np.any(~np.isfinite(frequencies)):
        raise ValueError("maf must be finite")
    if np.any((frequencies <= 0) | (frequencies > 0.5)):
        raise ValueError("weights require minor-allele frequencies in (0, 0.5]")
    if np.any(~np.isfinite(phred)):
        raise ValueError("annotations must be finite")
    if np.any(phred < 0):
        raise ValueError("PHRED annotations cannot be negative")
    host_ready = time.perf_counter()

    m, k = phred.shape
    width = 2 * (k + 1)
    burden = np.empty((m, width), dtype=np.float64)
    skat = np.empty_like(burden)
    acat = np.empty_like(burden)
    for row in range(m):
        frequency = float(frequencies[row])
        log1p_frequency = math.log1p(-frequency)
        beta25 = math.exp(24.0 * log1p_frequency - R_LBETA_1_25)
        beta_half = math.exp(
            (-0.5 * math.log(frequency))
            + (-0.5 * log1p_frequency)
            - R_LBETA_HALF
        )
        beta25_squared = beta25 * beta25
        half_squared = beta_half * beta_half
        burden[row, 0] = skat[row, 0] = beta25
        burden[row, k + 1] = skat[row, k + 1] = 1.0
        acat[row, 0] = beta25_squared / half_squared
        acat[row, k + 1] = 1.0 / half_squared
        for column in range(k):
            rank = 1.0 - math.pow(10.0, -float(phred[row, column]) / 10.0)
            square_root = math.sqrt(rank)
            burden[row, column + 1] = rank * beta25
            burden[row, k + column + 2] = rank * 1.0
            skat[row, column + 1] = square_root * beta25
            skat[row, k + column + 2] = square_root * 1.0
            acat[row, column + 1] = (rank * beta25_squared) / half_squared
            acat[row, k + column + 2] = (rank * 1.0) / half_squared
    transformed = time.perf_counter()
    outputs = tuple(torch.as_tensor(a, dtype=torch.float64, device=target)
                    for a in (burden, skat, acat))
    if cuda_target and profile is not None:
        torch.cuda.synchronize(target)
    done = time.perf_counter()
    cuda_inputs = [value for value in (maf, annotations)
                   if isinstance(value, torch.Tensor) and value.device.type == "cuda"]
    _record(
        weight_calls=1,
        weight_rows=m,
        weight_annotation_cells=m * k,
        weight_d2h_calls=len(cuda_inputs),
        weight_d2h_bytes=sum(value.numel() * 8 for value in cuda_inputs),
        weight_h2d_calls=3 if cuda_target else 0,
        weight_h2d_bytes=3 * m * width * 8 if cuda_target else 0,
        weight_conversion_wall_seconds=done - start,
        weight_scalar_transform_seconds=transformed - host_ready,
    )
    if profile is not None:
        profile.update({
            "scope": "one annotation-weight conversion; synchronous boundaries",
            "input_to_host_seconds": host_ready - start,
            "host_scalar_transform_seconds": transformed - host_ready,
            "output_to_device_seconds": done - transformed,
            "total_seconds": done - start,
            "maf_cells": m,
            "annotation_cells": m * k,
            "output_cells": 3 * m * width,
            "d2h_calls": len(cuda_inputs),
            "d2h_bytes": sum(value.numel() * 8 for value in cuda_inputs),
            "h2d_calls": 3 if cuda_target else 0,
            "h2d_bytes": 3 * m * width * 8 if cuda_target else 0,
            "target_device": str(target),
            "cached": False,
        })
    return outputs


def native_annotation_weights(maf, annotations=None):
    """Original three weight definitions, vectorized on the input device in FP32.

    expm1 expresses 1-exp(-PHRED*log(10)/10) without small-PHRED cancellation.
    The two beta families and Burden/SKAT/ACAT-V annotation powers are distinct.
    No host libm loop or host copy participates in this native path.
    """
    f = torch.as_tensor(maf, dtype=torch.float32)
    phred = (torch.empty((len(f), 0), device=f.device, dtype=f.dtype) if annotations is None
             else torch.as_tensor(annotations, device=f.device, dtype=f.dtype))
    beta25 = torch.exp(24 * torch.log1p(-f) + math.log(25.))
    beta_half = torch.exp(-.5 * torch.log(f) - .5 * torch.log1p(-f) - math.log(math.pi))
    rank = -torch.expm1(-phred * (math.log(10.) / 10.))
    annotations_rank = torch.cat((torch.ones((len(f), 1), dtype=f.dtype, device=f.device), rank), dim=1)
    base = torch.stack((beta25, torch.ones_like(f)), dim=1)
    burden = (base[:, :, None] * annotations_rank[:, None, :]).reshape(len(f), -1)
    skat = (base[:, :, None] * annotations_rank.sqrt()[:, None, :]).reshape(len(f), -1)
    acat = ((base.square() / beta_half.square()[:, None])[:, :, None] * annotations_rank[:, None, :]).reshape(len(f), -1)
    _record(native_weight_calls=1, weight_calls=1, weight_rows=len(f), weight_annotation_cells=phred.numel())
    return burden, skat, acat
