"""Index-only contracts for local mask UV reuse (no precision changes).

SPDX-License-Identifier: GPL-3.0-only

Pipeline counters distinguish fallback_cuda_oom (allocation failure),
fallback_memory (declared workspace/raw budget) and fallback_geometry
(actual rare union geometry is rejected after its applicable native tile check).
geometry_checks and geometry_*_covariance_cells aggregate these actual rare
geometry checks, including rejected unions; aggregate cell ratios are not
per-family timing or speedup measurements. union_score_calls includes a
completed UV product even if a later mask tail fails; reused_masks counts
only complete accepted union output, so partial discarded output is excluded.
"""
from __future__ import annotations
import numpy as np


def valid_index_sets(index_sets):
    """Each physical mask must identify each variant once and unambiguously."""
    for values in index_sets:
        rows = np.asarray(values)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            return False
        if np.any(rows < 0) or len(np.unique(rows)) != len(rows):
            return False
    return True


def ordered_mask_columns(physical_indices, extraction_groups, mask_indices, *, grouped):
    """Recover mask input order, then apply its original stable group sort."""
    lookup = {int(row): column for column, row in enumerate(physical_indices)}
    columns = np.asarray([lookup[int(row)] for row in mask_indices if int(row) in lookup], dtype=np.int64)
    if grouped and len(columns):
        columns = columns[np.argsort(np.asarray(extraction_groups)[columns], kind="stable")]
    return columns


class UnionGeometryRejected(Exception):
    """Actual rare union would compute more covariance cells than its masks."""


def union_covariance_geometry(physical_indices, index_sets, *, minimum_variants):
    """Use eligible rare columns only; ignore independently insufficient masks.

    Cache hits and same-order aliases must already have been removed. This is
    a conservative covariance-cost gate, not a measured runtime prediction.
    """
    physical = np.asarray(physical_indices, dtype=np.int64)
    counts = [int(np.isin(physical, indices).sum()) for indices in index_sets]
    counts = [count for count in counts if count >= minimum_variants]
    union_cells = len(physical) ** 2
    mask_cells = sum(count ** 2 for count in counts)
    return {"union_covariance_cells": union_cells, "mask_covariance_cells": mask_cells,
            "beneficial": union_cells <= mask_cells}



def small_native_union_work(physical_indices, index_sets, *, minimum_variants, samples, covariates):
    """Shape-only UV work estimate from the current native backend policy.

    Work is padded MMA lane multiplies or unpadded GEMV/dot/outer multiplies,
    never a runtime prediction. Every corresponding route must agree so unlike
    execution classes are never compared as if their throughput were equal.
    Full per-mask eigenspectra/tails and genotype preparation are outside this cost.
    """
    from .tf32 import _native_geometry, _native_route
    physical = np.asarray(physical_indices)
    m, n, p = len(physical), int(samples), int(covariates)
    counts = [int(np.isin(physical, indices).sum()) for indices in index_sets]
    counts = [count for count in counts if count >= minimum_variants]
    if not 2 <= m <= 64 or p < 1 or n < 2 or len(counts) < 2:
        return {"beneficial": False, "union": {}, "separate": {}, "union_calls": 0, "separate_calls": 0}
    def products(size):
        shapes = {"cross": ((p, n), (n, size)),
                  "covariance": ((size, n), (n, size)),
                  "projection_left": ((size, p), (p, p)),
                  "projection": ((size, p), (p, size)),
                  "score": ((size, n), (n, 1))}
        result = {}
        for name, (left, right) in shapes.items():
            rows, inner, columns = left[0], left[1], right[1]
            route = _native_route(left, right)
            if route == "tf32_mma":
                bm, bn, bk, _, _ = _native_geometry(rows, columns, inner)
                work = ((rows + bm - 1) // bm * bm
                        * ((columns + bn - 1) // bn * bn)
                        * ((inner + bk - 1) // bk * bk))
            elif route == "outer":
                work = rows * columns
            elif route == "empty":
                work = 0
            else:
                work = rows * columns * inner
            result[name] = {"route": route, "work": work}
        return result
    union = products(m)
    separate_products = [products(count) for count in counts]
    separate = {name: sum(cost[name]["work"] for cost in separate_products) for name in union}
    union_calls, separate_calls = len(union), sum(len(cost) for cost in separate_products)
    extra = 0
    if p == 1:
        # Keep independent route requirements; never offset an outer increment
        # against MMA work or pretend the routes have equal throughput.
        extra = max(0, union["projection"]["work"] - separate["projection"])
        beneficial = (union_calls <= separate_calls
            and union["covariance"]["route"] == "tf32_mma"
            and all(cost["covariance"]["route"] == "tf32_mma" for cost in separate_products)
            and union["covariance"]["work"] < separate["covariance"]
            and all(union[name]["route"] in ("gemv", "dot")
                    and all(cost[name]["route"] == union[name]["route"] for cost in separate_products)
                    and union[name]["work"] <= separate[name]
                    for name in ("cross", "score", "projection_left"))
            and union["projection"]["route"] == "outer"
            and all(cost["projection"]["route"] == "outer" for cost in separate_products)
            and union["projection"]["work"] <= 4096 and extra <= 4096)
    else:
        beneficial = union_calls <= separate_calls and all(
            union[name]["work"] <= separate[name]
            and all(cost[name]["route"] == union[name]["route"] for cost in separate_products)
            for name in union)
    return {"beneficial": beneficial, "union": union, "separate": separate,
            "union_calls": union_calls, "separate_calls": separate_calls,
            "policy": "intercept_independent_routes" if p == 1 else "per_product_nonincrease",
            "outer_extra_output_bytes": 4 * extra,
            "outer_extra_output_bytes_scope": "shape difference only; not complete allocation/workspace"}


def release_failed_union(device):
    """Run only after exception/attempt frames no longer own GPU temporaries."""
    import gc
    import torch
    gc.collect()  # Also releases unreachable cycles containing failed tensors.
    if str(device).startswith("cuda"):
        # One recovery synchronization lets queued users of released buffers
        # finish before inactive allocator blocks are returned to the driver.
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def attempt_union(operation, *, device):
    """Return failure reason after unwinding; never retry inside an except.

    Catch only CUDA allocation failures, declared workspace limits and the
    covariance geometry gate. Numerical/data errors must still propagate.
    The callback must keep all union tensors in its own frame; no exception
    object/traceback or partial output is returned to the fallback caller.
    """
    import torch
    reason = None
    result = None
    try:
        result = operation()
    except torch.OutOfMemoryError:
        reason = "fallback_cuda_oom"
    except MemoryError:
        reason = "fallback_memory"
    except UnionGeometryRejected:
        reason = "fallback_geometry"
    # The except has finished: Python's active exception and its traceback
    # no longer retain operation/model/math-kernel frames or their tensors.
    if reason in ("fallback_cuda_oom", "fallback_memory"):
        release_failed_union(device)
    return result, reason
