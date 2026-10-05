"""Float64 eigenvalues of a diagonal matrix with a symmetric rank-one update.

The secular backend is an independent implementation of the secular equation
and eigenvalue interlacing. It does not copy or invoke LAPACK source. Exact
repeated diagonal entries and zero update components are deflated; a negative
update is reflected to a positive update. Computation uses bounded Torch
workspaces on the input device, with no M x M matrix in the secular backend.

Dense remains the default. ``auto`` also stays dense unless the caller provides
``auto_min_size`` after its own real-data precision and timing validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional

import torch


@dataclass(frozen=True)
class RankOneEigenvalues:
    """Values plus secular interlacing brackets in ascending eigenvalue order.

    For secular results, ``lower <= values <= upper`` and half the bracket
    width bounds the root approximation in exact arithmetic. Brackets receive
    one outward floating-point step when rescaled. They do not certify all
    upstream rounding or relative accuracy below eps*||matrix||. Dense results
    have no interval certificate and use ``lower=upper=None``.
    """
    values: torch.Tensor
    lower: Optional[torch.Tensor]
    upper: Optional[torch.Tensor]
    backend: str


def _positive_secular(diagonal, vector, coefficient, root_chunk, max_iterations):
    order = diagonal.argsort()
    raw_diagonal = diagonal[order]
    squared_update = coefficient * vector[order].square()
    if not bool(torch.isfinite(squared_update).all()):
        raise ArithmeticError('Rank-one squared update overflows float64')
    if not bool((squared_update > 0).any()):
        return RankOneEigenvalues(raw_diagonal, raw_diagonal, raw_diagonal, 'secular')
    scale = torch.maximum(raw_diagonal.abs().max(), squared_update.sum())
    if not bool(torch.isfinite(scale)):
        raise ArithmeticError('Rank-one eigenvalue scale overflows float64')

    # Group exact original poles before normalization. The norm of the update
    # in each repeated eigenspace is the only component that changes its pole.
    poles, inverse, counts = torch.unique_consecutive(raw_diagonal,
                                                     return_inverse=True,
                                                     return_counts=True)
    grouped = torch.zeros_like(poles).scatter_add_(0, inverse, squared_update)
    grouped = grouped / scale
    active = grouped > 0
    retained = torch.repeat_interleave(poles, counts - active.to(counts.dtype))
    d = poles[active] / scale
    q = grouped[active]
    if not d.numel():
        return RankOneEigenvalues(retained, retained, retained, 'secular')
    widths = torch.cat((d[1:] - d[:-1], q.sum().reshape(1)))
    lower_offsets = torch.zeros_like(d)
    upper_offsets = widths.clone()
    eps = torch.finfo(diagonal.dtype).eps
    for start in range(0, d.numel(), root_chunk):
        stop = min(start + root_chunk, d.numel())
        origins = d[start:stop]
        interval_widths = widths[start:stop]
        # Interlacing already gives an absolute <=4eps*scale midpoint error
        # on such intervals, even with nearly coincident or singular poles.
        selected = (interval_widths > 8 * eps).nonzero(as_tuple=True)[0]
        if not selected.numel():
            continue
        origins = origins[selected]
        low = torch.zeros_like(origins)
        high = interval_widths[selected].clone()
        differences = d[None, :] - origins[:, None]
        for _ in range(max_iterations):
            midpoint = low + (high - low) * .5
            denominator = differences - midpoint[:, None]
            # The secular function is monotone increasing on each interval:
            # f(t)=1+sum_j q_j/(d_j-origin-t). Keeping t as an offset avoids
            # adding a tiny root displacement to a large pole before division.
            value = 1 + (q[None, :] / denominator).sum(1)
            negative = value < 0
            low = torch.where(negative, midpoint, low)
            high = torch.where(negative, high, midpoint)
        lower_offsets[start + selected] = low
        upper_offsets[start + selected] = high

    roots = (d + lower_offsets + (upper_offsets - lower_offsets) * .5) * scale
    negative_infinity = torch.full_like(d, -math.inf)
    positive_infinity = torch.full_like(d, math.inf)
    lower = torch.nextafter((d + lower_offsets) * scale, negative_infinity)
    upper = torch.nextafter((d + upper_offsets) * scale, positive_infinity)
    values = torch.cat((roots, retained))
    order = values.argsort()
    return RankOneEigenvalues(values[order], torch.cat((lower, retained))[order],
                              torch.cat((upper, retained))[order], 'secular')


def rank_one_eigvalsh(diagonal, vector, coefficient=1., *, backend='dense',
                     root_chunk=256, max_iterations=64, auto_min_size=None,
                     return_bounds=False):
    """Eigenvalues of ``diag(diagonal)+coefficient*vector*vector.T``.

    Inputs are matching finite one-dimensional float64 Torch tensors on one
    device. Output stays on that device and contains all eigenvalues, including
    zero and negative values; callers apply their own covariance cutoff.

    ``backend`` is dense/secular/auto. Dense constructs one M x M matrix and
    calls Torch eigvalsh. Secular requires O(max_iterations*M**2) operations and
    O(root_chunk*M) workspace. Auto chooses secular only when an explicit
    positive ``auto_min_size`` is supplied and M reaches that threshold.
    ``return_bounds=True`` returns RankOneEigenvalues instead of its values.
    With fewer iterations, wider brackets expose the remaining root error.
    No relative accuracy is promised for eigenvalues below eps*matrix norm.
    """
    if not isinstance(diagonal, torch.Tensor) or not isinstance(vector, torch.Tensor):
        raise TypeError('Rank-one inputs must be Torch tensors')
    if diagonal.ndim != 1 or vector.shape != diagonal.shape:
        raise ValueError('Rank-one inputs must be matching one-dimensional vectors')
    if diagonal.dtype != torch.float64 or vector.dtype != torch.float64:
        raise ValueError('Rank-one eigenvalues require float64 inputs')
    if diagonal.device != vector.device:
        raise ValueError('Rank-one inputs must be on the same device')
    if backend not in ('dense', 'secular', 'auto'):
        raise ValueError('Rank-one backend must be dense, secular, or auto')
    for name, value in [('root_chunk', root_chunk), ('max_iterations', max_iterations)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if auto_min_size is not None and (isinstance(auto_min_size, bool) or
            not isinstance(auto_min_size, Integral) or auto_min_size < 1):
        raise ValueError('auto_min_size must be a positive integer or None')
    try:
        coefficient = float(coefficient)
    except (TypeError, ValueError) as error:
        raise ValueError('Rank-one coefficient must be a finite scalar') from error
    if not math.isfinite(coefficient):
        raise ValueError('Rank-one coefficient must be a finite scalar')
    if not bool(torch.isfinite(diagonal).all() & torch.isfinite(vector).all()):
        raise ValueError('Rank-one inputs must be finite')
    chosen = backend
    if chosen == 'auto':
        chosen = ('secular' if auto_min_size is not None and
                  diagonal.numel() >= auto_min_size else 'dense')
    if not diagonal.numel() or coefficient == 0:
        values = diagonal.sort().values
        result = RankOneEigenvalues(values, values if chosen == 'secular' else None,
                                    values if chosen == 'secular' else None, chosen)
    elif chosen == 'dense':
        matrix = (coefficient * vector)[:, None] * vector[None, :]
        matrix.diagonal().add_(diagonal)
        if not bool(torch.isfinite(matrix).all()):
            raise ArithmeticError('Rank-one dense matrix overflows float64')
        result = RankOneEigenvalues(torch.linalg.eigvalsh(matrix), None, None, 'dense')
    elif coefficient < 0:
        positive = _positive_secular(-diagonal, vector, -coefficient,
                                     root_chunk, max_iterations)
        result = RankOneEigenvalues(-positive.values.flip(0), -positive.upper.flip(0),
                                    -positive.lower.flip(0), 'secular')
    else:
        result = _positive_secular(diagonal, vector, coefficient,
                                   root_chunk, max_iterations)
    return result if return_bounds else result.values
