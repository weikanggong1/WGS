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
        totals = torch.stack((log_area, log_error), dim=1).tolist()
        return [value[0] for value in totals], [value[1] for value in totals]

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


class _EpsilonTable:
    """Bounded Wynn epsilon diagonals for ordinary, consistently scaled areas.

    The recurrence uses three successive differences; nearly coincident or
    unstable diagonals are discarded. Only indexing and convergence flags
    reach Python. Areas, reciprocal differences and the three-result error
    history remain float64 tensors on the integration device.
    """
    def __init__(self, first, second):
        self.values = first.new_zeros(52)
        self.values[:2] = torch.stack((first, second))
        self.size, self.calls = 2, 0
        self.history = first.new_zeros(3)

    def append(self, area):
        self.values[self.size] = area
        self.size += 1
        self.calls += 1
        original_size = self.size
        diagonal_count = (original_size-1)//2
        huge = area.new_tensor(torch.finfo(torch.float64).max)
        epsilon = torch.finfo(torch.float64).eps
        result, error = area.clone(), huge.clone()
        self.values[self.size+1] = area
        self.values[self.size-1] = huge
        position = self.size-1
        for diagonal in range(diagonal_count):
            e0 = self.values[position-2].clone()
            e1 = self.values[position-1].clone()
            e2 = self.values[position+2].clone()
            delta2, delta3 = e2-e1, e1-e0
            error2, error3 = delta2.abs(), delta3.abs()
            close2 = error2 <= torch.maximum(e2.abs(), e1.abs())*epsilon
            close3 = error3 <= torch.maximum(e1.abs(), e0.abs())*epsilon
            if bool(close2 & close3):
                return e2, torch.maximum(error2+error3, 5*epsilon*e2.abs())
            e3 = self.values[position].clone()
            self.values[position] = e1
            delta1 = e1-e3
            close1 = delta1.abs() <= torch.maximum(e1.abs(), e3.abs())*epsilon
            if bool(close1 | close2 | close3):
                self.size = 2*diagonal+1
                break
            reciprocal = delta1.reciprocal()+delta2.reciprocal()-delta3.reciprocal()
            if bool((reciprocal*e1).abs() <= 1e-3):
                self.size = 2*diagonal+1
                break
            estimate = e1+reciprocal.reciprocal()
            self.values[position] = estimate
            estimate_error = error2+(estimate-e2).abs()+error3
            if bool(estimate_error <= error):
                result, error = estimate, estimate_error
            position -= 2
        if self.size == 50:
            self.size = 49
        start = 1 if original_size % 2 == 0 else 0
        for index in range(start, start+2*(diagonal_count+1), 2):
            self.values[index] = self.values[index+2].clone()
        if self.size != original_size:
            offset = original_size-self.size
            self.values[:self.size] = self.values[offset:offset+self.size].clone()
        if self.calls < 4:
            self.history[self.calls-1] = result
            error = huge
        else:
            error = (result-self.history).abs().sum()
            self.history[:2] = self.history[1:].clone()
            self.history[2] = result
        return result, torch.maximum(error, 5*epsilon*result.abs())


def integrate_log_qags(log_function, upper, *, epsabs=1e-25,
                        epsrel=2.**-13, max_intervals=1000):
    """Positive original-coordinate QAGS with device-side Wynn extrapolation.

    This independent tensor implementation keeps the QAGS large-interval
    error selection, extrapolation history, roundoff counters and checked
    termination. It does not call SciPy, native code or a CPU integrand.
    One fixed logarithmic scale normalizes *all* rules, areas, epsilon
    diagonals and absolute tolerances. Existing GK21 behavior is unchanged.
    """
    validate_quadrature_parameters(epsabs, epsrel, max_intervals)
    reference = torch.as_tensor(upper, dtype=torch.float64)
    if reference.numel() != 1 or not bool(torch.isfinite(reference)) or float(reference) < 0:
        raise ValueError("Integral upper bound must be a finite nonnegative scalar.")
    boundary = float(reference)
    if boundary == 0:
        return reference.new_tensor(-math.inf), {
            "converged": True, "status": "converged", "evaluations": 0,
            "intervals": 0, "relative_error_estimate": 0., "log_error_estimate": -math.inf,
            "ier": 0, "extrapolation_calls": 0}
    nodes, kronrod_weights, gauss_weights = _rule_constants(str(reference.device))
    machine = torch.finfo(torch.float64)
    evaluations, scale = 0, None

    def rule(bounds):
        nonlocal evaluations, scale
        limits = reference.new_tensor(bounds)
        half = (limits[:, 1]-limits[:, 0])/2
        positions = (limits[:, 1]+limits[:, 0])[:, None]/2+half[:, None]*nodes
        logarithms = log_function(positions.flatten()).reshape(positions.shape)
        evaluations += positions.numel()
        if bool((torch.isnan(logarithms) | torch.isposinf(logarithms)).any()):
            raise ArithmeticError("Nonfinite quadrature integrand.")
        if scale is None:
            largest = logarithms.amax()
            scale = torch.where(torch.isneginf(largest), largest.new_zeros(()), largest)
        values = (logarithms-scale).exp()
        if not bool(torch.isfinite(values).all()):
            raise ArithmeticError("QAGS fixed-scale integrand overflowed.")
        kronrod = (values*kronrod_weights).sum(1)
        gauss = (values*gauss_weights).sum(1)
        absolute = (values.abs()*kronrod_weights).sum(1)*half
        deviation = ((values-kronrod[:, None]/2).abs()*kronrod_weights).sum(1)*half
        error = ((kronrod-gauss)*half).abs()
        adjusted = deviation*torch.minimum(torch.ones_like(deviation),
            (200*error/deviation.clamp_min(machine.tiny)).pow(1.5))
        error = torch.where((deviation > 0) & (error > 0), adjusted, error)
        # The original absolute rule floor does not apply below this physical
        # resabs boundary. Scaling must not invent an underflow error floor.
        floor_active = absolute.log()+scale > math.log(machine.tiny/(50*machine.eps))
        error = torch.where(floor_active, torch.maximum(error, 50*machine.eps*absolute), error)
        return torch.stack((kronrod*half, error, absolute, deviation), dim=1)

    first = rule([(0., boundary)])[0]
    area, error_sum, absolute, deviation = (value.clone() for value in first)
    absolute_tolerance = reference.new_tensor(math.log(epsabs) if epsabs else -math.inf).sub(scale).exp()
    tolerance = torch.maximum(absolute_tolerance, epsrel*area.abs())
    intervals = [(0., boundary)]
    areas, errors = [area.clone()], [error_sum.clone()]
    order, priority, selected = [0], 0, 0
    ier = 0
    if bool((error_sum <= 100*machine.eps*absolute) & (error_sum > tolerance)):
        ier = 2
    if max_intervals == 1:
        ier = 1
    initial_complete = bool(((error_sum <= tolerance) & (error_sum != deviation)) | (error_sum == 0))
    epsilon_table = None
    extrapolated = area.clone()
    extrapolated_error = reference.new_tensor(machine.max)
    extrapolate = disabled = False
    small = boundary*.375
    large_error = error_sum.clone()
    extrapolation_tolerance = tolerance.clone()
    correction = reference.new_zeros(())
    roundoff_regular = roundoff_extrapolated = roundoff_growth = 0
    extrapolation_roundoff = stagnation = 0
    prefer_global = initial_complete or bool(ier)
    if not prefer_global:
        # Only the ordering metadata has a host mirror. Unchanged interval
        # errors need not be transferred again after every subdivision.
        error_metadata = [float(error_sum)]
        while len(intervals) < max_intervals:
            left, right = intervals[selected]
            midpoint = (left+right)/2
            if midpoint == left or midpoint == right:
                ier = 4
                break
            old_error = errors[selected]
            old_area = areas[selected]
            paired = rule([(left, midpoint), (midpoint, right)])
            area_left, error_left, _, deviation_left = paired[0]
            area_right, error_right, _, deviation_right = paired[1]
            pair_area, pair_error = area_left+area_right, error_left+error_right
            error_sum = error_sum+pair_error-old_error
            area = area+pair_area-old_area
            regular = (deviation_left != error_left) & (deviation_right != error_right)
            unchanged = (old_area-pair_area).abs() <= 1e-5*pair_area.abs()
            # Return only this subdivision's flags and its two error totals
            # in one bounded transfer. All area/error arithmetic stays on
            # the device; these doubles are used solely for error ordering.
            flags = torch.stack((regular & unchanged & (pair_error >= .99*old_error),
                                 regular & (pair_error > old_error), error_right > error_left,
                                 error_left, error_right)).tolist()
            if flags[0]:
                if extrapolate:
                    roundoff_extrapolated += 1
                else:
                    roundoff_regular += 1
            count = len(intervals)+1
            if count > 10 and flags[1]:
                roundoff_growth += 1
            if flags[2]:
                intervals[selected] = (midpoint, right)
                intervals.append((left, midpoint))
                areas[selected], errors[selected] = area_right, error_right
                areas.append(area_left); errors.append(error_left)
                error_metadata[selected] = flags[4]
                error_metadata.append(flags[3])
            else:
                intervals[selected] = (left, midpoint)
                intervals.append((midpoint, right))
                areas[selected], errors[selected] = area_left, error_left
                areas.append(area_right); errors.append(error_right)
                error_metadata[selected] = flags[3]
                error_metadata.append(flags[4])
            tolerance = torch.maximum(absolute_tolerance, epsrel*area.abs())
            if roundoff_regular+roundoff_extrapolated >= 10 or roundoff_growth >= 20:
                ier = 2
            if roundoff_extrapolated >= 5:
                extrapolation_roundoff = 3
            if count == max_intervals:
                ier = 1
            if max(abs(left), abs(right)) <= (1+100*machine.eps)*(abs(midpoint)+1000*machine.tiny):
                ier = 4
            # QAGS can raise the current priority when subdivision increases
            # its error. Sorting is metadata control; error arithmetic stays
            # on the device. Equal errors retain the original interval index.
            while priority and error_metadata[selected] > error_metadata[order[priority-1]]:
                priority -= 1
            order = sorted(range(count), key=lambda index: (-error_metadata[index], index))
            selected = order[priority]
            if bool(error_sum <= tolerance):
                prefer_global = True
                break
            if ier:
                break
            if count == 2:
                epsilon_table = _EpsilonTable(first[0], area)
                large_error, extrapolation_tolerance = error_sum.clone(), tolerance.clone()
                continue
            if disabled:
                continue
            large_error = large_error-old_error
            if midpoint-left > small:
                large_error = large_error+pair_error
            if not extrapolate:
                if intervals[selected][1]-intervals[selected][0] > small:
                    continue
                extrapolate = True
                priority = 1
            if extrapolation_roundoff != 3 and bool(large_error > extrapolation_tolerance):
                search_end = count if count <= max_intervals//2+2 else max_intervals+3-count
                while priority < search_end:
                    selected = order[priority]
                    if intervals[selected][1]-intervals[selected][0] > small:
                        break
                    priority += 1
                if priority < search_end:
                    continue
            estimate, estimate_error = epsilon_table.append(area)
            stagnation += 1
            if stagnation > 5 and bool(extrapolated_error < 1e-3*error_sum):
                ier = 5
            if bool(estimate_error < extrapolated_error):
                stagnation = 0
                extrapolated, extrapolated_error = estimate, estimate_error
                correction = large_error.clone()
                extrapolation_tolerance = torch.maximum(absolute_tolerance, epsrel*estimate.abs())
                if bool(extrapolated_error <= extrapolation_tolerance):
                    break
            disabled = epsilon_table.size == 1
            if ier == 5:
                break
            priority, selected, extrapolate = 0, order[0], False
            small *= .5
            large_error = error_sum.clone()
        else:
            ier = 1
    if bool(extrapolated_error == machine.max):
        prefer_global = True
    if not prefer_global and (ier or extrapolation_roundoff):
        if extrapolation_roundoff == 3:
            extrapolated_error = extrapolated_error+correction
        if not ier:
            ier = 3
        if bool((extrapolated != 0) & (area != 0)):
            prefer_global = bool(extrapolated_error/extrapolated.abs() > error_sum/area.abs())
        else:
            prefer_global = bool(extrapolated_error > error_sum)
    if prefer_global:
        result = reference.new_zeros(())
        for value in areas:
            result = result+value
        result_error = error_sum
    else:
        result, result_error = extrapolated, extrapolated_error
        if bool((result < .01*area) | (result > 100*area) | (error_sum > area.abs())):
            ier = 6
    if ier > 2:
        ier -= 1
    final_tolerance = torch.maximum(absolute_tolerance, epsrel*result.abs())
    if not ier and not bool(torch.isfinite(result) & (result >= 0) & (result_error <= final_tolerance)):
        ier = 4
    log_result = result.log()+scale
    log_error = result_error.log()+scale
    relative_error = torch.where(result > 0, result_error/result, torch.zeros_like(result))
    summary = torch.stack((relative_error, log_error)).tolist()
    statuses = {0: "converged", 1: "subdivision_limit", 2: "roundoff", 3: "bad_integrand",
                4: "extrapolation_failure", 5: "divergent"}
    return log_result, {
        "converged": ier == 0, "status": statuses[ier], "ier": ier,
        "evaluations": evaluations, "intervals": len(intervals),
        "relative_error_estimate": summary[0], "log_error_estimate": summary[1],
        "extrapolation_calls": 0 if epsilon_table is None else epsilon_table.calls,
        "used_extrapolation": not prefer_global}
