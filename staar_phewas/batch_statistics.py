"""Optional float64 GPU batches for STAAR mask SKAT calculations.

SPDX-License-Identifier: GPL-3.0-only
Algorithms follow the same frozen STAAR sources cited in statistics.py.
Batching changes execution grouping, never the statistical thresholds.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import torch
from .statistics import (
    DegenerateTestError, _double, _finite, _quadratic_form_sf_tensor,
    annotation_weights, staar_test, _ordered_sum,
)

from ._precision_eigen import (near_mean_mask, refine_near_mean_spectrum,
                               record_gpu_eigen_route)


def _saddle_batch(q_raw: torch.Tensor, raw: torch.Tensor):
    """Run independent original bisections without per-row CPU decisions."""
    if bool((~torch.isfinite(raw)).any()) or bool((~torch.isfinite(q_raw) | (q_raw < 0)).any()):
        raise ValueError("quadratic form inputs must be finite and nonnegative")
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max(dim=1).values
    if bool((maximum <= 0).any()):
        raise DegenerateTestError("SKAT covariance has no eigenvalue at or above 1e-8")
    scaled = spectrum / maximum[:, None]
    q = q_raw / maximum
    # A zero statistic returns 1, after the same positive-spectrum check.
    positive = q_raw > 0
    safe_q = torch.where(positive, q, torch.ones_like(q))
    lower = torch.where(q > scaled.sum(dim=1), torch.full_like(q, -.01), -torch.full_like(safe_q, raw.shape[1]) / (2 * safe_q))
    upper = torch.full_like(q, .499995)
    root = torch.zeros_like(q)
    live = positive.clone()
    # A bisection halves its initial interval each update. Two spare steps
    # cover floating-point interval-width rounding. Every row still stops at
    # its original width/derivative condition, using work rather than CPU
    # decisions in the loop. Only the initial bound and final convergence
    # check synchronize the batch.
    initial_width = float(torch.where(positive, (upper-lower).abs(), 0.).max())
    if not math.isfinite(initial_width):
        raise ArithmeticError("STAAR saddlepoint bisection did not converge")
    iterations = 0 if initial_width <= 1e-8 else min(2048, max(0, math.ceil(math.log2(initial_width)-math.log2(1e-8)))+2)
    for _ in range(iterations):
        work = live & ((upper - lower).abs() > 1e-8)
        middle = (upper + lower) / 2
        derivative = (scaled / (1 - 2 * scaled * middle[:, None])).sum(dim=1) - q
        root = torch.where(work, middle, root)
        zero = derivative == 0
        upper = torch.where(work & ~zero & (derivative > 0), middle, upper)
        lower = torch.where(work & ~zero & (derivative <= 0), middle, lower)
        live = live & ~zero
    if bool((live & ((upper-lower).abs() > 1e-8)).any()):
        raise ArithmeticError("STAAR saddlepoint bisection did not converge")
    moment = root.abs() < 1e-4
    c1, c2, c4 = raw.sum(dim=1), raw.square().sum(dim=1), raw.pow(4).sum(dim=1)
    dof = c2.square() / c4
    adjusted = (q_raw - c1) / torch.sqrt(2 * c2) * torch.sqrt(2 * dof) + dof
    moment_p = torch.where(adjusted <= 0, torch.ones_like(q), torch.special.gammaincc(dof / 2, adjusted / 2))
    cumulant = -.5 * torch.log(1 - 2 * scaled * root[:, None]).sum(dim=1)
    w2 = 2 * (root * q - cumulant)
    signed_root = torch.copysign(torch.sqrt(torch.clamp_min(w2, torch.finfo(q.dtype).tiny)), root)
    second = 2 * (scaled.square() / (1 - 2 * scaled * root[:, None]).square()).sum(dim=1)
    v = root * torch.sqrt(second)
    z = signed_root + torch.log(v / signed_root) / signed_root
    saddle_p = .5 * torch.erfc(z / math.sqrt(2))
    p = torch.where(moment, moment_p, saddle_p)
    p = torch.where(positive, p, torch.ones_like(p))
    # Near the mean the upstream log(v/w)/w amplifies changes in eigensolver
    # or parallel reductions. Re-evaluate these rows through the original
    # serial float64 implementation; this does NOT alter its 1e-4 moment
    # switch or 1e-8 root/eigenvalue cutoffs.
    compatibility = positive & (near_mean_mask(q_raw, raw) | ((w2 <= 0) & ~moment) | ~torch.isfinite(p))
    return p, compatibility


def _prepare(item):
    if not isinstance(item, Mapping):
        raise TypeError("each batch item must be a mapping of staar_test arguments")
    u = _double(item["score"])
    v = _double(item["covariance"], device=u.device)
    f = _double(item["maf"], device=u.device)
    mac = _double(item["mac"], device=u.device)
    if u.ndim != 1 or v.shape != (len(u), len(u)) or f.shape != u.shape or mac.shape != u.shape:
        raise ValueError("score/maf/mac must be vectors and covariance must be variants-by-variants")
    for x, label in ((u, "score"), (v, "covariance"), (f, "maf"), (mac, "mac")):
        _finite(x, label)
    if bool(((f < 0) | (f > .5) | (mac < 0)).any()):
        raise ValueError("maf must lie in [0,.5] and mac cannot be negative")
    cutoff = item.get("rare_maf_cutoff", .01)
    least, most = item.get("rv_num_cutoff", 2), item.get("rv_num_cutoff_max", 1_000_000_000)
    if not 0 < cutoff <= .5 or not 1 <= least < most:
        raise ValueError("invalid rare-variant cutoffs")
    keep = (f > 0) & (f < cutoff)
    count = int(keep.sum())
    if count < least or count >= most:
        raise ValueError("rare-variant count is outside the allowed interval")
    annotations = item.get("annotations")
    if annotations is not None:
        annotations = _double(annotations, device=u.device)
        if annotations.ndim != 2 or annotations.shape[0] != len(u):
            raise ValueError("annotations must have one row per original variant")
        annotations = annotations[keep]
    u, v, f = u[keep], v[keep][:, keep], f[keep]
    if not bool(torch.allclose(v, v.T, atol=1e-10, rtol=1e-10)):
        raise ValueError("covariance must be symmetric")
    reference_covariance = v
    v = (v + v.T) * .5
    _, ws, _ = annotation_weights(f, annotations)
    return u, v, ws, reference_covariance


def staar_test_batch(
    items: Sequence[Mapping], *, max_workspace_bytes: int = 256 * 1024**2,
    return_diagnostics: bool = False,
):
    """Return one original-format STAAR dict per input mask, in input order.

    Each item is a mapping containing the arguments accepted by staar_test.
    Matrices remain float64. Exact variant counts are grouped rather than
    padded, so the reference bisection intervals and spectrum lengths stay
    unchanged. The workspace limit controls additional weighted matrices;
    genotype buffers and caller-owned covariance matrices are separate.
    Very large matrices use the original serial implementation.
    """
    if max_workspace_bytes < 1:
        raise ValueError("max_workspace_bytes must be positive")
    prepared = [_prepare(item) for item in items]
    outputs = [None] * len(items)
    skat_values = [torch.empty(ws.shape[1], dtype=u.dtype, device=u.device) for u, _, ws, _ in prepared]
    groups = defaultdict(list)
    serial = set()
    diagnostics = {"masks": len(items), "weighted_matrices": 0, "eigen_batches": 0,
                   "compatibility_rows": 0, "precision_reference_rows": 0, "workspace_serial_masks": 0,
                   "max_weighted_matrix_bytes": 0, "dtype": "float64"}
    for index, (u, v, ws, _) in enumerate(prepared):
        # Account for weighted input, solver copy/workspace and staging.
        if 4 * 8 * len(u)**2 > max_workspace_bytes:
            serial.add(index)
            continue
        for column in range(ws.shape[1]):
            groups[(str(u.device), len(u))].append((index, column))
    # A singleton CUDA eigensolve uses a different solver from a natural
    # small-matrix batch. When workspace or the final chunk would force that
    # switch, send the whole affected mask through the declared serial path;
    # its real annotation weights are still computed together.
    for (device, size), rows in list(groups.items()):
        if device.startswith("cuda") and size <= 32:
            capacity = max_workspace_bytes // (4 * 8 * size**2)
            if capacity < 2:
                serial.update(index for index, _ in rows)
                rows = []
            else:
                while len(rows) % capacity == 1:
                    serial.add(rows[-1][0])
                    rows = [row for row in rows if row[0] not in serial]
            groups[(device, size)] = rows
    diagnostics["workspace_serial_masks"] = len(serial)
    for (_, size), rows in groups.items():
        batch_size = max(1, max_workspace_bytes // (4 * 8 * size**2))
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            matrices, statistics = [], []
            for index, column in chunk:
                u, v, ws, _ = prepared[index]
                weight = ws[:, column]
                matrices.append(v * weight[:, None] * weight[None, :])
                # Retain the original 1-D reduction for each quadratic form.
                statistics.append(_ordered_sum(u.square() * weight.square()))
            stacked = torch.stack(matrices)
            eigenvalues = torch.linalg.eigvalsh(stacked, UPLO="U")
            record_gpu_eigen_route(stacked)
            probabilities, compatibility = _saddle_batch(torch.stack(statistics), eigenvalues)
            fallback_rows = torch.nonzero(compatibility, as_tuple=True)[0].cpu().tolist()
            for row in fallback_rows:
                index, column = chunk[row]
                _, _, ws, reference_covariance = prepared[index]
                weight = ws[:, column]
                original_eigenvalues, refined = refine_near_mean_spectrum(
                    reference_covariance, eigenvalues[row], statistics[row], weights=weight)
                if not refined and not (stacked.is_cuda and size <= 32 and len(chunk) > 1):
                    original_eigenvalues = torch.linalg.eigvalsh(matrices[row], UPLO="U")
                    record_gpu_eigen_route(matrices[row])
                probabilities[row] = _quadratic_form_sf_tensor(
                    statistics[row], original_eigenvalues, reference_reduction=refined)
                diagnostics["precision_reference_rows"] += int(refined)
            for row, (index, column) in enumerate(chunk):
                skat_values[index][column] = probabilities[row]
            diagnostics["weighted_matrices"] += len(chunk)
            diagnostics["eigen_batches"] += 1
            diagnostics["compatibility_rows"] += len(fallback_rows)
            diagnostics["max_weighted_matrix_bytes"] = max(diagnostics["max_weighted_matrix_bytes"], stacked.numel() * stacked.element_size())
    for index, item in enumerate(items):
        if "_skat_pvalues" in item:
            raise ValueError("batch items cannot supply private precomputed SKAT values")
        outputs[index] = staar_test(**item) if index in serial else staar_test(**item, _skat_pvalues=skat_values[index])
    return (outputs, diagnostics) if return_diagnostics else outputs
