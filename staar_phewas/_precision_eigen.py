"""Reference spectra for numerically sensitive CUDA saddle calculations.

SPDX-License-Identifier: GPL-3.0-only

A monotonic derivative bracket selects near-mean rows on the input device.
Only their symmetric eigenvalues use the validated CPU LAPACK implementation;
score covariance, quadratic forms and tail probabilities remain CUDA work.
Annotation-weight CPU boundaries are reported by ``_reference_weights``.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

_REFERENCE_LIBRARY_SHA256 = "f66c03e52d8a9e0102c515c148cf36941016c6dcdd8e38f20a9f56ab9b7f2fd8"
_ROOT_THRESHOLD = 0.01
_BRACKET_MARGIN = 1e-8
_LIBRARY = None
_METADATA = {
    "backend": "conditional_reference_lapack",
    "workspace_policy": "fixed_66n_original_armadillo",
    "cpu_eigen_calls": 0,
    "cpu_solve_seconds": 0.0,
    "transfer_and_solve_seconds": 0.0,
    "nearmean_root_threshold": _ROOT_THRESHOLD,
    "bracket_margin": _BRACKET_MARGIN,
    "selection": "monotonic K1 brackets",
    "selected_weighted_rows": 0,
    "library_sha256": None,
    "library_version": None,
    "gpu_eigen_calls": 0,
    "gpu_eigen_matrices": 0,
    "gpu_eigen_routes": {},
    "gpu_ordered_tail_calls": 0,
}


def _library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    supplied = os.environ.get("STAAR_REFERENCE_LAPACK_LIBRARY")
    if not supplied:
        raise RuntimeError(
            "Near-mean STAAR parity requires the validated MKL 2019.2 library; "
            "set STAAR_REFERENCE_LAPACK_LIBRARY after installing the reference environment"
        )
    path = Path(supplied).expanduser()
    if not path.is_file():
        raise RuntimeError("Configured reference LAPACK library does not exist")
    binary_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if binary_sha != _REFERENCE_LIBRARY_SHA256:
        raise RuntimeError(
            "Reference LAPACK must use the validated MKL 2019.2 library; "
            "the configured binary differs"
        )
    # Separate the pinned library symbols from the Torch process BLAS symbols.
    mode = ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
    try:
        library = ctypes.CDLL(str(path.resolve()), mode=mode)
    except OSError as error:
        raise RuntimeError("Cannot load the configured reference LAPACK library") from error
    pointer = ctypes.POINTER(ctypes.c_double)
    function = library.dsyev_
    function.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), pointer,
        ctypes.POINTER(ctypes.c_int), pointer, pointer,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None
    library.MKL_Set_Num_Threads.argtypes = [ctypes.c_int]
    library.MKL_Set_Num_Threads.restype = None
    library.MKL_Set_Num_Threads(ctypes.c_int(1))
    _LIBRARY = library
    _METADATA["library_sha256"] = binary_sha
    _METADATA["library_version"] = "MKL 2019.2 intel_187"
    return library


def cpu_reference_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """Return DSYEV(N,U) eigenvalues on the matrix's original device.

    The unaveraged upper triangle is authoritative.  The matching numerical
    library is installed separately; this routine never starts R or silently
    substitutes another eigensolver.
    """
    if (matrix.ndim != 2 or matrix.shape[0] < 1
            or matrix.shape[0] != matrix.shape[1] or matrix.dtype != torch.float64):
        raise ValueError("Reference spectrum requires a nonempty square float64 matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Reference spectrum matrix must contain only finite values")
    started = time.perf_counter()
    source = matrix.detach().cpu().numpy()
    function = _library().dsyev_
    pointer = ctypes.POINTER(ctypes.c_double)
    n = ctypes.c_int(len(source))
    a = np.array(source, dtype=np.float64, order="F", copy=True)
    values = np.empty(n.value, dtype=np.float64)
    # Match the original Armadillo values-only DSYEV workspace contract.
    size, info = ctypes.c_int(66 * n.value), ctypes.c_int(0)
    work = np.empty(size.value, dtype=np.float64)
    solve_started = time.perf_counter()
    function(
        b"N", b"U", ctypes.byref(n), a.ctypes.data_as(pointer), ctypes.byref(n),
        values.ctypes.data_as(pointer), work.ctypes.data_as(pointer),
        ctypes.byref(size), ctypes.byref(info),
    )
    seconds = time.perf_counter() - solve_started
    if info.value:
        raise RuntimeError(f"Reference DSYEV eigen solve failed: {info.value}")
    result = torch.as_tensor(values, device=matrix.device, dtype=matrix.dtype)
    _METADATA["cpu_eigen_calls"] += 1
    _METADATA["cpu_solve_seconds"] += seconds
    _METADATA["transfer_and_solve_seconds"] += time.perf_counter() - started
    return result


def near_mean_mask(statistic: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    """Select a conservative K1 bracket entirely on the input device.

    Eigenvalues have shape [..., variants]; the statistic has matching leading
    dimensions. K1 is monotonic, so bounds at ±(0.01+1e-8) select every root
    within the compatibility interval. Zero or degenerate inputs retain the
    original tail path and its existing validation.
    """
    spectrum = torch.where(eigenvalues < 1e-8, 0.0, eigenvalues)
    maximum = spectrum.max(dim=-1).values
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scaled = spectrum / safe_maximum[..., None]
    q = statistic / safe_maximum
    threshold = _ROOT_THRESHOLD + _BRACKET_MARGIN
    lower = (scaled / (1 + 2 * threshold * scaled)).sum(dim=-1)
    upper = (scaled / (1 - 2 * threshold * scaled)).sum(dim=-1)
    return (
        (q >= lower) & (q <= upper) & (statistic > 0) & (maximum > 0)
        & torch.isfinite(statistic) & torch.isfinite(eigenvalues).all(dim=-1)
    )


def refine_near_mean_spectrum(matrix, eigenvalues, statistic, *, weights=None):
    """Refine a sensitive CUDA spectrum using the original upper triangle.

    Optional weights are applied only after selection, avoiding a second
    weighted covariance allocation on the ordinary GPU path.
    """
    if matrix.is_cuda and bool(near_mean_mask(statistic, eigenvalues)):
        weighted = matrix if weights is None else matrix * weights[:, None] * weights[None, :]
        result = cpu_reference_eigenvalues(weighted)
        _METADATA["selected_weighted_rows"] += 1
        return result, True
    return eigenvalues, False


def record_gpu_eigen_route(matrix):
    """Record a completed Torch CUDA eigenvalues-only solve and input grouping."""
    if not matrix.is_cuda:
        return
    count = 1 if matrix.ndim == 2 else matrix.shape[0]
    # The pinned Torch 2.4/2.5 cuSolver dispatch uses small batched Jacobi;
    # its remaining float64 eigenvalues-only paths use SYEVD.
    route = (
        "small_batched_jacobi"
        if matrix.ndim == 3 and count > 1 and matrix.shape[-1] <= 32 else "syevd"
    )
    _METADATA["gpu_eigen_calls"] += 1
    _METADATA["gpu_eigen_matrices"] += count
    routes = _METADATA["gpu_eigen_routes"]
    routes[route] = routes.get(route, 0) + count


def record_ordered_tail():
    _METADATA["gpu_ordered_tail_calls"] += 1


def precision_eigen_execution_metadata(*, reset=False):
    """Return actual CPU spectrum boundaries and CUDA solve counts.

    CPU solve time excludes transfers. Transfer-and-solve wall time includes
    synchronization, copies, loader initialization and workspace allocation. It is
    separate from CUDA kernel time; private library paths are omitted.
    """
    result = copy.deepcopy(_METADATA)
    if reset:
        for key in (
            "cpu_eigen_calls", "selected_weighted_rows", "gpu_eigen_calls",
            "gpu_eigen_matrices", "gpu_ordered_tail_calls",
        ):
            _METADATA[key] = 0
        _METADATA["cpu_solve_seconds"] = 0.0
        _METADATA["transfer_and_solve_seconds"] = 0.0
        _METADATA["gpu_eigen_routes"] = {}
    return result
