"""NumPy error bounds for the positive central Davies controller.

Only scalar bound evaluation is vectorized. Left-to-right float64 sums use
NumPy accumulate rather than a pairwise reduction; all budget/bracketing,
convergence coefficients and Fourier integration remain the frozen methods.
"""
import math

import numpy as np

from torchwgs.statistics import _DaviesPlan as FrozenDaviesPlan


def sequential_sum(values):
    """Match Python 3.9 sum's initial-zero, left-to-right additions."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.add.accumulate(values)[-1]) if values.size else 0.


class NumpyDaviesPlan(FrozenDaviesPlan):
    """Preserve ordered sums and original bracketing/budget/fault decisions."""
    def __init__(self, spectrum, statistic, accuracy, limit):
        super().__init__(spectrum, statistic, accuracy, limit)
        self._weights = np.asarray(self.spectrum, dtype=np.float64)

    def truncation_error(self, frequency, extra_variance=0.0):
        self._count()
        normal = (self.gaussian_variance+extra_variance)*frequency*frequency
        with np.errstate(over="ignore", invalid="ignore"):
            bases = (2*frequency)*self._weights
            arguments = np.square(bases)
        # Python float **2 raises on a finite overflowing base. Infinity
        # already present in the base remains infinity in the scalar method.
        if np.any(np.isfinite(bases) & np.isinf(arguments)):
            raise OverflowError(34, "Numerical result out of range")
        small_values = arguments[arguments <= 1]
        large = arguments[arguments > 1]
        small = sequential_sum(np.log1p(small_values))
        product_bound = 2*normal+small+sequential_sum(np.log(large))
        smooth_bound = 2*normal+small+sequential_sum(np.log1p(large))
        a = self._bounded_exp(-product_bound/4)/math.pi
        b = self._bounded_exp(-smooth_bound/4)/math.pi
        polynomial = 2*a/large.size if large.size else 1.0
        combined = 2.5*b if smooth_bound > 1 else 1.0
        gaussian = b/(normal/2) if normal/2 > b else 1.0
        return min(polynomial, combined, gaussian)

    def tail_bound(self, parameter):
        self._count()
        with np.errstate(over="ignore", invalid="ignore"):
            offsets = (2*parameter)*self._weights
        values = np.empty_like(offsets)
        small = np.abs(offsets) < .01
        x = offsets[small]
        series = np.zeros_like(x)
        repeated_power = x.copy()
        for power in range(2, 13):
            repeated_power *= x
            term = repeated_power
            series += ((power-1)/power)*term
        values[small] = series
        x = offsets[~small]
        invalid = np.flatnonzero(x >= 1)
        if invalid.size:
            if x[invalid[0]] == 1:
                raise ZeroDivisionError("float division by zero")
            raise ValueError("math domain error")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values[~small] = x/(1-x)+np.log1p(-x)
        exponent = parameter*parameter*self.gaussian_variance+sequential_sum(values)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            boundary = parameter*self.gaussian_variance+sequential_sum(self._weights/(1-offsets))
        return self._bounded_exp(-exponent/2), boundary

