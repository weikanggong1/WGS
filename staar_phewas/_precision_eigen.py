"""Complete Torch eigenspectrum route counters; no CPU precision refinement."""
from __future__ import annotations
import copy

_METADATA = {
    "backend": "torch_complete_eigvalsh",
    "cpu_precision_refinement": "retired_not_used",
    "workspace_policy": "bounded_weighted_matrix_batches",
    "cpu_eigen_calls": 0, "cpu_solve_seconds": 0.,
    "transfer_and_solve_seconds": 0., "selected_weighted_rows": 0,
    "library_sha256": None, "library_version": None,
    "gpu_eigen_calls": 0, "gpu_eigen_matrices": 0,
    "gpu_eigen_routes": {}, "gpu_ordered_tail_calls": 0,
    "eigen_dtype_calls": {},
}


def record_gpu_eigen_route(matrix):
    """Record successful complete Torch solves without changing their precision."""
    if not matrix.is_cuda:return
    count = 1 if matrix.ndim == 2 else matrix.shape[0]
    route = "small_batched_jacobi" if matrix.ndim == 3 and count > 1 and matrix.shape[-1] <= 32 else "syevd"
    _METADATA["gpu_eigen_calls"] += 1
    _METADATA["gpu_eigen_matrices"] += count
    routes = _METADATA["gpu_eigen_routes"]
    routes[route] = routes.get(route, 0) + count
    dtype = str(matrix.dtype)
    counts = _METADATA["eigen_dtype_calls"]
    counts[dtype] = counts.get(dtype, 0) + count


def record_ordered_tail():
    _METADATA["gpu_ordered_tail_calls"] += 1


def precision_eigen_execution_metadata(*, reset=False):
    """Compatibility counters explicitly report zero CPU refinement boundaries."""
    result = copy.deepcopy(_METADATA)
    if reset:
        for key in ("gpu_eigen_calls", "gpu_eigen_matrices", "gpu_ordered_tail_calls"):
            _METADATA[key] = 0
        _METADATA["gpu_eigen_routes"] = {}
        _METADATA["eigen_dtype_calls"] = {}
    return result


def near_mean_mask(statistic, eigenvalues):
    """Legacy batch branch predicate only; never selects CPU refinement."""
    import torch
    spectrum = torch.where(eigenvalues < 1e-8, 0., eigenvalues)
    maximum = spectrum.max(dim=-1).values
    safe = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scaled, q = spectrum / safe[..., None], statistic / safe
    lower = (scaled / (1 + 2 * (.01 + 1e-8) * scaled)).sum(dim=-1)
    upper = (scaled / (1 - 2 * (.01 + 1e-8) * scaled)).sum(dim=-1)
    return ((q >= lower) & (q <= upper) & (statistic > 0) & (maximum > 0)
            & torch.isfinite(statistic) & torch.isfinite(eigenvalues).all(dim=-1))


def refine_near_mean_spectrum(matrix, eigenvalues, statistic, *, weights=None):
    """Import compatibility for explicit legacy controls: retain Torch spectrum.

    CPU refinement and pinned external LAPACK loading have been removed.
    Native STAAR does not call this compatibility hook.
    """
    return eigenvalues, False
