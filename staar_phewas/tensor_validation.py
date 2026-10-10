"""Finite-value checks with bounded CUDA temporaries and no layout copies.

SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

from itertools import product
from math import prod
import operator

import torch


_MAX_BLOCK_ELEMENTS = 2**23
_MAX_SCRATCH_BYTES = 64 * 1024**2


def _block_element_limit(number_elements, element_size):
    # isfinite can hold abs plus three boolean tensors. Using at most half
    # of a non-scalar input keeps this phase below one extra FP32 G copy.
    return min(_MAX_BLOCK_ELEMENTS, max(1, number_elements // 2),
               max(1, _MAX_SCRATCH_BYTES // (element_size + 3)))


def finite_check_scratch_bytes(shape, *, element_size=4):
    """Conservative CUDA finite-check workspace for a dense tensor shape.

    ``shape`` contains nonnegative integer axis lengths; ``element_size``
    is the storage bytes per element (4 for the native FP32 genotypes).
    The returned integer bounds abs and three boolean block temporaries;
    the reduction and aggregate also hold two scalar boolean tensors.
    This does not include the already resident input tensor.
    """
    axes = tuple(operator.index(length) for length in shape)
    size = operator.index(element_size)
    if any(length < 0 for length in axes) or size <= 0:
        raise ValueError("finite-check shape and element size must be nonnegative and positive")
    number_elements = prod(axes)
    if number_elements == 0:
        return 0
    return _block_element_limit(number_elements, size) * (size + 3) + 2


def _iter_finite_blocks(value, maximum_elements):
    """Yield storage-sharing blocks without cloning an arbitrary strided view."""
    if type(maximum_elements) is not int or maximum_elements < 1:
        raise ValueError("maximum_elements must be a positive integer")
    if value.numel() == 0:
        return
    if value.is_contiguous():
        flattened = value.view(-1)
    elif value.ndim == 2 and value.T.is_contiguous():
        flattened = value.T.view(-1)
    else:
        # Fill the fastest-striding axes first. Every rectangular slice is
        # a view, including stepped, offset and broadcast input layouts.
        lengths = [1] * value.ndim
        remaining = maximum_elements
        for axis in sorted(range(value.ndim), key=lambda number: value.stride(number)):
            lengths[axis] = min(value.shape[axis], remaining)
            remaining //= lengths[axis]
        starts = (range(0, length, step) for length, step in zip(value.shape, lengths))
        for corner in product(*starts):
            slices = tuple(slice(start, min(start + step, length))
                           for start, step, length in zip(corner, lengths, value.shape))
            yield value[slices]
        return
    for start in range(0, flattened.numel(), maximum_elements):
        yield flattened[start:start + maximum_elements]


def _all_finite_blocked(value, maximum_elements):
    """Check all blocks, reducing only the final scalar to the host."""
    valid = torch.ones((), dtype=torch.bool, device=value.device)
    for block in _iter_finite_blocks(value, maximum_elements):
        # Keep no elementwise flags alive between blocks. In particular,
        # do not retain the previous flags while constructing the next ones.
        finite = torch.isfinite(block).all()
        valid.logical_and_(finite)
        del finite
    return bool(valid)


def tensor_all_finite(value):
    """Return whether a tensor is finite, without a full CUDA flag matrix.

    CPU tensors retain the established ``torch.isfinite(value).all()``
    expression. Dense CUDA floating/complex tensors use bounded views,
    inspect every logical element, and synchronize the host once. Integer
    and boolean CUDA tensors are always finite. Neither the input storage,
    its strides nor its gradient state is changed.
    """
    if value.device.type != "cuda" or value.layout != torch.strided:
        return bool(torch.isfinite(value).all())
    if not (value.is_floating_point() or value.is_complex()):
        return True
    if value.numel() == 0:
        return True
    return _all_finite_blocked(value, _block_element_limit(value.numel(), value.element_size()))
