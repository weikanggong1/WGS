"""Ordered float64 sum of squares for the frozen SSE2 reference protocol.

The four independent addition chains implement a specified floating-point
reduction order.  They are executed inside one GPU kernel, rather than one
CUDA launch per addition.  This file contains no Eigen implementation.
"""

import triton
import triton.language as tl

try:  # PyTorch 2.0 / Triton 2.0 on the validation server.
    from triton.language import libdevice as _ld
except ImportError:  # Newer NVIDIA Triton distributions.
    from triton.language.extra.cuda import libdevice as _ld


@triton.jit
def _ordered_squared_norm(values, output, rows: tl.constexpr, columns: tl.constexpr,
                          row_stride: tl.constexpr, column_stride: tl.constexpr,
                          squared: tl.constexpr, block: tl.constexpr):
    col = tl.program_id(0)*block+tl.arange(0, block)
    valid = col < columns
    if rows == 0:
        result = tl.full((block,), 0., tl.float64)
    elif rows == 1:
        x = tl.load(values+col*column_stride, valid, 0.).to(tl.float64)
        result = _ld.mul_rn(x, x)
    else:
        a = tl.load(values+col*column_stride, valid, 0.).to(tl.float64)
        b = tl.load(values+row_stride+col*column_stride, valid, 0.).to(tl.float64)
        a, b = _ld.mul_rn(a, a), _ld.mul_rn(b, b)
        if rows >= 4:
            c = tl.load(values+2*row_stride+col*column_stride, valid, 0.).to(tl.float64)
            d = tl.load(values+3*row_stride+col*column_stride, valid, 0.).to(tl.float64)
            c, d = _ld.mul_rn(c, c), _ld.mul_rn(d, d)
            for row in range(4, rows//4*4, 4):
                x0 = tl.load(values+row*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                x1 = tl.load(values+(row+1)*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                x2 = tl.load(values+(row+2)*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                x3 = tl.load(values+(row+3)*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                a = _ld.add_rn(a, _ld.mul_rn(x0, x0))
                b = _ld.add_rn(b, _ld.mul_rn(x1, x1))
                c = _ld.add_rn(c, _ld.mul_rn(x2, x2))
                d = _ld.add_rn(d, _ld.mul_rn(x3, x3))
            a, b = _ld.add_rn(a, c), _ld.add_rn(b, d)
            if rows % 4 >= 2:
                tail = rows//4*4
                x0 = tl.load(values+tail*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                x1 = tl.load(values+(tail+1)*row_stride+col*column_stride, valid, 0.).to(tl.float64)
                a = _ld.add_rn(a, _ld.mul_rn(x0, x0))
                b = _ld.add_rn(b, _ld.mul_rn(x1, x1))
        result = _ld.add_rn(a, b)
        if rows % 2:
            x = tl.load(values+(rows-1)*row_stride+col*column_stride, valid, 0.).to(tl.float64)
            result = _ld.add_rn(result, _ld.mul_rn(x, x))
    if not squared:
        result = _ld.sqrt_rn(result)
    tl.store(output+col, result, valid)


def ordered_column_norm(matrix, output, squared=False):
    _ordered_squared_norm[(triton.cdiv(matrix.shape[1], 32),)](
        matrix, output, matrix.shape[0], matrix.shape[1], matrix.stride(0),
        matrix.stride(1), squared, 32, num_warps=1)
    return output
