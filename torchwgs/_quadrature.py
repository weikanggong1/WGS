"""Adaptive positive integrals with float64 PyTorch Gauss-Kronrod rules.

Function evaluations and rule sums stay on the input device. A scalar
controller bisects the interval with the largest estimated error. Keeping
areas/errors in logarithms also supports probabilities below float64's
ordinary range. These are mathematical Gauss-10/Kronrod-21 constants;
this implementation does not call QUADPACK or a CPU integrand.
"""
from functools import lru_cache
import math
from numbers import Integral

import torch


_POSITIVE_NODES = (
    .9956571630258080807355, .9739065285171717200779,
    .9301574913557082260012, .8650633666889845107321,
    .7808177265864168970637, .6794095682990244062343,
    .5627571346686046833390, .4333953941292471907993,
    .2943928627014601981311, .1488743389816312108848)
_KRONROD_WEIGHTS = (
    .0116946388673718742781, .0325581623079647274788,
    .0547558965743519960314, .0750396748109199527670,
    .0931254545836976055351, .1093871588022976418992,
    .1234919762620658510779, .1347092173114733259281,
    .1427759385770600807971, .1477391049013384913748,
    .1494455540029169056649)
_GAUSS_WEIGHTS = (
    .0666713443086881375936, .1494513491505805931458,
    .2190863625159820439955, .2692667193099963550912,
    .2955242247147528701739)


def validate_quadrature_parameters(epsabs, epsrel, max_intervals):
    if not (math.isfinite(epsabs) and epsabs >= 0):
        raise ValueError("Integral epsabs must be finite and nonnegative.")
    if not (math.isfinite(epsrel) and epsrel > 0):
        raise ValueError("Integral epsrel must be finite and positive.")
    if isinstance(max_intervals, bool) or not isinstance(max_intervals, Integral) or max_intervals < 1:
        raise ValueError("Integral max_intervals must be a positive integer.")


@lru_cache(maxsize=16)
def _rule_constants(device):
    nodes = torch.tensor(tuple(-x for x in _POSITIVE_NODES) + (0.,) +
                         tuple(reversed(_POSITIVE_NODES)), device=device, dtype=torch.float64)
    kronrod = torch.tensor(_KRONROD_WEIGHTS[:10] + (_KRONROD_WEIGHTS[10],) +
                           tuple(reversed(_KRONROD_WEIGHTS[:10])), device=device, dtype=torch.float64)
    gauss = torch.zeros(21, device=device, dtype=torch.float64)
    gauss[[1, 3, 5, 7, 9]] = torch.tensor(_GAUSS_WEIGHTS, device=device, dtype=torch.float64)
    gauss[[11, 13, 15, 17, 19]] = torch.tensor(tuple(reversed(_GAUSS_WEIGHTS)),
                                            device=device, dtype=torch.float64)
    return nodes, kronrod, gauss


def _logsum(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    largest = max(finite)
    return largest + math.log(math.fsum(math.exp(value-largest) for value in finite))


def integrate_log_gk21(log_function, upper, *, epsabs=1e-25,
                       epsrel=2.**-13, max_intervals=1000):
    """Integrate exp(log_function(x)) from zero to scalar upper.

    Return (log_integral, diagnostics). The callback accepts/returns vectors
    on upper's device. ``converged`` compares the global error estimate to
    max(epsabs, epsrel*integral); it is false if the interval budget or
    floating-point subdivision limit is reached. No unchecked estimate is
    silently labeled converged.
    """
    validate_quadrature_parameters(epsabs, epsrel, max_intervals)
    reference = torch.as_tensor(upper, dtype=torch.float64)
    if reference.numel() != 1 or not bool(torch.isfinite(reference)) or float(reference) < 0:
        raise ValueError("Integral upper bound must be a finite nonnegative scalar.")
    upper_float = float(reference)
    if upper_float == 0:
        return reference.new_tensor(-math.inf), {
            "converged": True, "status": "converged", "evaluations": 0,
            "intervals": 0, "relative_error_estimate": 0., "log_error_estimate": -math.inf}
    nodes, kronrod_weights, gauss_weights = _rule_constants(str(reference.device))
    evaluations = 0

    def rule(bounds):
        nonlocal evaluations
        limits = reference.new_tensor(bounds)
        half_width = (limits[:, 1]-limits[:, 0])/2
        positions = (limits[:, 1]+limits[:, 0])[:, None]/2 + half_width[:, None]*nodes
        log_values = log_function(positions.flatten()).reshape(positions.shape)
        evaluations += positions.numel()
        if bool((torch.isnan(log_values) | torch.isposinf(log_values)).any()):
            raise ArithmeticError("Nonfinite quadrature integrand.")
        shift = log_values.amax(1)
        shift = torch.where(torch.isneginf(shift), torch.zeros_like(shift), shift)
        values = (log_values-shift[:, None]).exp()
        kronrod = (values*kronrod_weights).sum(1)
        gauss = (values*gauss_weights).sum(1)
        absolute = (values.abs()*kronrod_weights).sum(1)
        deviation = ((values-kronrod[:, None]/2).abs()*kronrod_weights).sum(1)
        raw_error = (kronrod-gauss).abs()
        denominator = deviation.clamp_min(torch.finfo(torch.float64).tiny)
        corrected = deviation*torch.minimum(torch.ones_like(deviation),
                                             (200*raw_error/denominator).pow(1.5))
        error = torch.where((deviation > 0) & (raw_error > 0), corrected, raw_error)
        error = torch.maximum(error, 50*torch.finfo(torch.float64).eps*absolute)
        log_area = kronrod.log() + half_width.log() + shift
        log_error = error.log() + half_width.log() + shift
        # Only per-interval scalar totals enter the adaptive controller.
        return log_area.tolist(), log_error.tolist()

    areas, errors = rule([(0., upper_float)])
    intervals = [(0., upper_float, areas[0], errors[0])]
    absolute_log = math.log(epsabs) if epsabs > 0 else -math.inf
    relative_log = math.log(epsrel)
    status = "subdivision_limit"
    for _ in range(max_intervals):
        total = _logsum(interval[2] for interval in intervals)
        error = _logsum(interval[3] for interval in intervals)
        if error <= max(absolute_log, relative_log+total):
            status = "converged"
            break
        if len(intervals) >= max_intervals:
            break
        index = max(range(len(intervals)), key=lambda i: intervals[i][3])
        left, right, _, _ = intervals[index]
        midpoint = (left+right)/2
        if midpoint == left or midpoint == right:
            status = "roundoff"
            break
        areas, errors = rule([(left, midpoint), (midpoint, right)])
        intervals[index:index+1] = [(left, midpoint, areas[0], errors[0]),
                                   (midpoint, right, areas[1], errors[1])]
    total = _logsum(interval[2] for interval in intervals)
    error = _logsum(interval[3] for interval in intervals)
    relative_error = math.exp(error-total) if math.isfinite(error-total) else 0.
    return reference.new_tensor(total), {
        "converged": status == "converged", "status": status,
        "evaluations": evaluations, "intervals": len(intervals),
        "relative_error_estimate": relative_error, "log_error_estimate": error}
