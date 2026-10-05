"""Float64 association inference on the input tensor's device.

All public ``*_logp``/``*_logsf`` routines return -log10(P).  Genotype
cross-products, eigenproblems, NNLS and normal orthant integration stay in
PyTorch, including on CUDA.  Default association tails use central positive
Davies AS155 Fourier summation on the tensor device, with scalar CPU error
bound planning, then the original Kuonen/stringent-Davies/Liu hierarchy.
The optional exact backend uses checked exponentially tilted inversion;
its scalar SciPy quadrature fallback is available only for CPU inputs. Backend
counts and controller/fallback elapsed time are recorded explicitly.

SKAT-O is the rho-search integral with the moment/variance correction used by
REGENIE 3.4.1 (Lee et al., 2012), and is separate from SKAT-O-ACAT.  SBAT uses
the positive and negative NNLS chi-bar-square tests, combined by ACAT.
High-dimensional chi-bar weights use Genz orthant QMC and sampled active
subsets (REGENIE's default subset count is ten); those weights are numerical
approximations, not an exact closed-form distribution.

This implementation was written from the mathematical definitions.  No
upstream implementation is included here.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
import threading
from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import lru_cache
from numbers import Integral
from typing import Optional

import torch

_LN10 = math.log(10.0)
_LOGPI = math.log(math.pi)
DEFAULT_RHOS = (0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.5, 1.0)
_DIAGNOSTIC_DEFAULTS = {"gpu_tail_calls": 0, "cpu_tail_fallback_calls": 0,
                "torch_cpu_tail_calls": 0,
                "davies_tail_values": 0, "davies_failure_values": 0,
                "davies_spectrum_preparations": 0,
                "davies_fused_rules": 0, "davies_torch_rules": 0,
                "gene_vc_product_cache_hits": 0, "gene_vc_product_cache_misses": 0,
                "gene_burden_product_cache_hits": 0, "gene_burden_product_cache_misses": 0,
                "davies_integration_terms": 0, "davies_controller_seconds": 0.0,
                "kuonen_tail_values": 0, "liu_fallback_values": 0,
                "cpu_tail_fallback_seconds": 0.0, "orthant_qmc_calls": 0,
                "orthant_plackett_calls": 0,
                "orthant_sobol_cache_hits": 0, "orthant_sobol_cache_misses": 0,
                "skato_integral_calls": 0, "skato_quadrature_values": 0,
                "skato_quadrature_intervals": 0, "skato_integral_failures": 0,
                "secular_eigen_calls": 0, "dense_rho_eigen_calls": 0}


class _ThreadDiagnostics(MutableMapping):
    """Counters belong to the current worker, rather than other CUDA streams."""
    def __init__(self):
        self.local = threading.local()

    def current(self):
        if not hasattr(self.local, "counters"):
            self.local.counters = dict(_DIAGNOSTIC_DEFAULTS)
        return self.local.counters

    def __getitem__(self, key):
        return self.current()[key]

    def __setitem__(self, key, value):
        self.current()[key] = value

    def __delitem__(self, key):
        del self.current()[key]

    def __iter__(self):
        return iter(self.current())

    def __len__(self):
        return len(self.current())


_DIAGNOSTICS = _ThreadDiagnostics()


@contextmanager
def diagnostics_scope():
    """Isolate one analysis' tail counters in the current thread.

    The yielded dictionary keeps its final counters after leaving the scope.
    Nested scopes restore the caller's ledger; a worker cannot reset or add to
    another worker's counters. The existing numerical_diagnostics() API reads
    the currently active ledger and remains unchanged for serial callers.
    """
    previous = _DIAGNOSTICS.current()
    counters = dict(_DIAGNOSTIC_DEFAULTS)
    _DIAGNOSTICS.local.counters = counters
    try:
        yield counters
    finally:
        _DIAGNOSTICS.local.counters = previous


def numerical_diagnostics(reset: bool = False) -> dict:
    """Return this thread/scope's backend counts and fallback wall time.

    Tensor-tail counters count quadrature batches; the fallback counter
    counts scalar statistics.  Their ratio is not a GPU/CPU time fraction.
    Fallback seconds include transfer/setup and the checked SciPy integral.
    """
    result = dict(_DIAGNOSTICS)
    if reset:
        for key in _DIAGNOSTICS:
            _DIAGNOSTICS[key] = 0.0 if key.endswith("seconds") else 0
    return result


def _tensor(value, reference=None):
    return torch.as_tensor(value, dtype=torch.float64,
                           device=None if reference is None else reference.device)


def _mixture_device(q, eigenvalues):
    """Keep any CUDA input on CUDA; a CUDA statistic selects its own device."""
    if isinstance(q, torch.Tensor) and q.is_cuda:
        return q.device
    if isinstance(eigenvalues, torch.Tensor) and eigenvalues.is_cuda:
        return eigenvalues.device
    return None


def _mixture_spectrum(q, eigenvalues):
    device = _mixture_device(q, eigenvalues)
    return (eigenvalues if device is None else
            torch.as_tensor(eigenvalues, dtype=torch.float64, device=device))


def eigen_compatible_column_norm(matrix, *, squared=False):
    """Ordered float64 column norms for the frozen Eigen3.4/SSE2 protocol.

    Accept [N,M] and return [M], or [N] and return a scalar.  Four serial
    addition chains, followed by two pairwise merges and the odd scalar
    tail, reproduce that reference's cwiseAbs2 reduction.  CUDA uses one
    Triton kernel with separately rounded multiplication/addition; CPU uses
    the same recurrence in PyTorch.  Participant data stays on its device.
    This is intended for compatibility-sensitive SBAT pivoting, not for
    replacing general-purpose fast tensor norms.
    """
    values = _tensor(matrix)
    vector = values.ndim == 1
    if vector:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("column norms require a vector or [N,M] matrix")
    result = values.new_empty(values.shape[1])
    if values.device.type == "cuda":
        from ._norm_gpu import ordered_column_norm
        ordered_column_norm(values, result, squared)
    else:
        rows = values.shape[0]
        if rows < 2:
            result = values.square().sum(0)
        else:
            a, b = values[0].square(), values[1].square()
            if rows >= 4:
                c, d = values[2].square(), values[3].square()
                for row in range(4, rows//4*4, 4):
                    a = a+values[row].square()
                    b = b+values[row+1].square()
                    c = c+values[row+2].square()
                    d = d+values[row+3].square()
                a, b = a+c, b+d
                if rows % 4 >= 2:
                    a = a+values[rows//4*4].square()
                    b = b+values[rows//4*4+1].square()
            result = a+b
            if rows % 2:
                result = result+values[-1].square()
        if not squared:
            result = result.sqrt()
    return result[0] if vector else result


def chi2_logsf(chisq, df=1.0):
    """Stable chi-square upper tail, including P much smaller than 1e-308."""
    x = _tensor(chisq)
    degrees = _tensor(df, x)
    x, degrees = torch.broadcast_tensors(x, degrees)
    if bool(((x < 0) | (degrees <= 0)).any()):
        raise ValueError("chi-square statistic must be nonnegative and df positive")
    # The normal log-CDF has an asymptotic implementation, unlike erfc.
    if bool((degrees == 1).all()):
        return -(math.log(2.0) + torch.special.log_ndtr(-x.sqrt())) / _LN10
    a, z = degrees / 2.0, x / 2.0
    safe_z = z.clamp_min(torch.finfo(torch.float64).tiny)
    regular = torch.special.gammaincc(a, z)
    log_q = regular.log()
    needs_cf = (regular == 0) & (z > 0)
    if bool(needs_cf.any()):
        # Lentz continued fraction for Gamma(a,z)/Gamma(a).  Evaluate only
        # underflowed entries, whose large z makes this rapidly convergent.
        aa, zz = a[needs_cf], safe_z[needs_cf]
        tiny = torch.finfo(torch.float64).tiny / torch.finfo(torch.float64).eps
        b = zz + 1.0 - aa
        c = torch.full_like(b, 1.0 / tiny)
        d = 1.0 / b
        h = d.clone()
        for k in range(1, 257):
            an = -float(k) * (float(k) - aa)
            b = b + 2.0
            d = an * d + b
            d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
            c = b + an / c
            c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
            d = 1.0 / d
            delta = d * c
            h = h * delta
            if bool((delta - 1).abs().max() < 2e-15):
                break
        else:
            raise ArithmeticError("upper incomplete gamma fraction did not converge")
        log_q = log_q.clone()
        log_q[needs_cf] = aa * zz.log() - zz - torch.lgamma(aa) + h.log()
    return (-log_q / _LN10).clamp_min(0.0)


def chi2_isf_logp(logp, df=1.0):
    """Inverse chi-square upper tail from -log10(P), including extreme tails.

    For df=1 and -log10(P)<=300, a float64 normal quantile gives the same
    inverse directly. Near P=1, expm1/erfinv preserves the small lower-tail
    probability that would be lost by subtracting from one. Other degrees
    of freedom and stronger tails retain the stable log-tail bisection.
    """
    lp = _tensor(logp)
    degrees = _tensor(df, lp)
    lp, degrees = torch.broadcast_tensors(lp, degrees)
    if bool(((lp < 0) | ~torch.isfinite(lp)).any()):
        raise ValueError("inverse upper tail requires finite nonnegative -log10(P)")
    if bool(((degrees <= 0) | ~torch.isfinite(degrees)).any()):
        raise ValueError("inverse upper tail requires finite positive df")
    direct = (degrees == 1) & (lp <= 300.0)

    def normal_inverse(values):
        upper = torch.special.ndtri(torch.exp(-values * _LN10 - math.log(2.0))).square()
        # chi2_1 CDF(x)=erf(sqrt(x/2)); no subtraction from P near one.
        lower = 2.0 * torch.erfinv(-torch.expm1(-values.clamp(max=.1) * _LN10)).square()
        return torch.where(values < .1, lower, upper)

    if bool(direct.all()):
        return normal_inverse(lp)
    result = torch.empty_like(lp)
    if bool(direct.any()):
        result[direct] = normal_inverse(lp[direct])
    slow_lp, slow_df = lp[~direct], degrees[~direct]
    lo = torch.zeros_like(slow_lp)
    hi = 2.0 * slow_lp * _LN10 + 10.0 * slow_df + 100.0
    while bool((chi2_logsf(hi, slow_df) < slow_lp).any()):
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        below = chi2_logsf(mid, slow_df) < slow_lp
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    result[~direct] = torch.where(slow_lp == 0, torch.zeros_like(slow_lp), (lo + hi) / 2.0)
    return result


def acat_logp(log10ps, weights=None, dim=-1):
    """Cauchy combination in signed log space.

    ``log10ps`` contains -log10(P), not log(P).  Nonnegative weights are
    normalized over valid entries.  REGENIE's p<=0.999 convention is retained
    for the negative Cauchy tail; small P is never clipped or exponentiated.
    Negative/NaN entries are excluded, and an empty combination returns NaN.
    """
    lp = _tensor(log10ps)
    if lp.ndim == 0:
        lp = lp.unsqueeze(0)
    w = torch.ones_like(lp) if weights is None else _tensor(weights, lp)
    lp, w = torch.broadcast_tensors(lp, w)
    if bool((w < 0).any()):
        raise ValueError("ACAT weights must be nonnegative")
    valid = (lp >= 0) & ~torch.isnan(lp) & (w > 0)
    w = torch.where(valid, w, 0.0)
    total_weight = w.sum(dim=dim, keepdim=True)
    wn = w / total_weight.clamp_min(torch.finfo(w.dtype).tiny)
    p = torch.exp(-lp.clamp(max=15) * _LN10).clamp(max=0.999)
    cauchy = torch.tan(math.pi * (0.5 - p))
    sign = torch.where(lp >= 15, torch.ones_like(lp), cauchy.sign())
    sign = torch.where(valid, sign, torch.zeros_like(sign))
    logabs = torch.where(lp >= 15, lp * _LN10 - _LOGPI,
                         cauchy.abs().log()) + wn.log()
    logabs = torch.where(valid, logabs, -torch.inf)
    maximum = logabs.amax(dim=dim, keepdim=True)
    finite_max = torch.where(torch.isfinite(maximum), maximum, 0.0)
    signed_sum = (sign * torch.exp(logabs - finite_max)).sum(dim=dim)
    log_t = finite_max.squeeze(dim) + signed_sum.abs().log()
    sign_t = signed_sum.sign()
    # atan2 avoids cancellation when T is negative or very close to zero.
    t = sign_t * torch.exp(log_t.clamp(max=36))
    lp_result = -torch.log(torch.atan2(torch.ones_like(t), t) / math.pi) / _LN10
    lp_result = torch.where((sign_t > 0) & (log_t > 36),
                            (log_t + _LOGPI) / _LN10, lp_result)
    lp_result = torch.where(torch.isposinf(maximum.squeeze(dim)),
                            torch.full_like(lp_result, torch.inf), lp_result)
    single = valid.sum(dim=dim) == 1
    sole_value = torch.where(valid, lp, -torch.inf).amax(dim=dim)
    lp_result = torch.where(single, sole_value, lp_result)
    return torch.where(total_weight.squeeze(dim) > 0, lp_result.clamp_min(0),
                       torch.full_like(lp_result, torch.nan))


@lru_cache(maxsize=24)
def _legendre_cpu(order):
    # Scalar integration nodes are setup constants, never sample matrices.
    from scipy.special import roots_legendre
    nodes, weights = roots_legendre(order)
    return torch.from_numpy(nodes.copy()), torch.from_numpy(weights.copy())


def _legendre(order, reference):
    nodes, weights = _legendre_cpu(order)
    return nodes.to(device=reference.device), weights.to(device=reference.device)


def _positive_eigenvalues(eigenvalues, threshold=0.0):
    ev = _tensor(eigenvalues).flatten()
    if not ev.numel():
        return ev
    if bool((~torch.isfinite(ev)).any()):
        raise ValueError("finite eigenvalues are required")
    scale = ev.abs().max()
    if bool((ev < -scale * 1e-10).any()):
        raise ValueError("covariance has materially negative eigenvalues")
    nonnegative = ev[ev >= 0]
    cutoff = nonnegative.mean() * threshold if threshold and nonnegative.numel() else 0.
    ev = ev[ev > cutoff]
    return ev


def _tilt(q, ev):
    mean = ev.sum()
    low = torch.zeros_like(q)
    high = torch.full_like(q, 0.5 * (1 - 1e-14))
    if bool((q > mean).any()):
        for _ in range(48):
            mid = (low + high) / 2.0
            derivative = (ev[None, :] / (1 - 2 * mid[:, None] * ev)).sum(-1)
            low = torch.where(derivative < q, mid, low)
            high = torch.where(derivative < q, high, mid)
    tilt = torch.where(q > mean, (low + high) / 2,
                       torch.ones_like(q) * (0.25 / mean))
    # Above the mean but extremely close to it, t~0 is a sharp pole.  A
    # positive contour remains exact and avoids resolving that narrow pole.
    minimum_tilt = 0.025 / (2 * ev.square().sum()).sqrt()
    tilt = torch.maximum(tilt, minimum_tilt)
    denom = 1 - 2 * tilt[:, None] * ev
    tilted_ev = ev[None, :] / denom
    variance = 2 * tilted_ev.square().sum(-1)
    log_prefactor = -0.5 * denom.log().sum(-1) - tilt * q
    return tilt, tilted_ev, variance.sqrt(), log_prefactor


def _tilted_integral(q, ev, order, tilt_parameters=None):
    tilt, tilted_ev, scale, log_prefactor = _tilt(q, ev) if tilt_parameters is None else tilt_parameters
    nodes, weights = _legendre(order, q)
    angle = (nodes + 1) * (math.pi / 4)
    freq = angle.tan()[None, :] / scale[:, None]
    jac = (math.pi / 4) / angle.cos().square()[None, :] / scale[:, None]
    logamp = torch.zeros_like(freq)
    phase = -freq * q[:, None]
    # Bound intermediate storage when a set has thousands of variants.
    for start in range(0, ev.numel(), 256):
        argument = 2 * freq[:, :, None] * tilted_ev[:, None, start:start+256]
        logamp -= 0.25 * torch.log1p(argument.square()).sum(-1)
        phase += 0.5 * argument.atan().sum(-1)
    integrand = logamp.exp() * (tilt[:, None] * phase.cos() + freq * phase.sin())
    integrand /= tilt[:, None].square() + freq.square()
    integral = (integrand * jac * weights).sum(-1) / math.pi
    logsf = log_prefactor + integral.log()
    return logsf, integral > 0


def _small_rank_integral(q, ev, order):
    """Positive spherical/pair-angle integrals for ranks two through four.

    These avoid the slowly decaying, oscillatory Fourier tail at low rank.
    Rank three conditions on a sphere direction and uses chi-square(3).
    Rank four conditions on the angles of two independent normal pairs;
    the remaining two scaled chi-square(2) variables have an elementary
    hypoexponential survival function, evaluated with expm1 at equal scales.
    """
    nodes, weights = _legendre(order, q)
    theta = (nodes + 1) * math.pi / 4
    cosine2, sine2 = theta.cos().square(), theta.sin().square()
    ev = ev.sort().values
    if ev.numel() == 2:
        scale = ev[0] * cosine2 + ev[1] * sine2
        return torch.logsumexp(-q[:, None] / (2 * scale) + (weights / 2).log(), -1)
    log_weights = ((weights / 2).log()[:, None] + (weights / 2).log()[None, :])
    if ev.numel() == 3:
        z = (nodes + 1) / 2
        pair_scale = ev[0] * cosine2 + ev[1] * sine2
        scale = ev[2] * z.square()[:, None] + (1-z.square())[:, None] * pair_scale
        x = q[:, None, None] / scale
        logsf = torch.logaddexp(math.log(2.) + torch.special.log_ndtr(-x.sqrt()),
                    0.5 * torch.log(2 * x / math.pi) - x / 2)
    elif ev.numel() == 4:
        first = ev[0] * cosine2 + ev[1] * sine2
        second = ev[2] * cosine2 + ev[3] * sine2
        aa = torch.maximum(first[:, None], second[None, :])
        bb = torch.minimum(first[:, None], second[None, :])
        difference = aa - bb
        argument = q[:, None, None] * difference / (2 * aa * bb)
        # expm1(argument)/difference has a continuous limit at zero.
        safe_difference = torch.where(difference > 0, difference, 1.)
        correction = bb / safe_difference * -torch.expm1(-argument)
        correction = torch.where(difference > 0, correction, q[:, None, None] / (2 * aa))
        logsf = -q[:, None, None] / (2 * aa) + torch.log1p(correction)
    else:
        raise ValueError("positive low-rank integral requires rank 2, 3 or 4")
    return torch.logsumexp((logsf + log_weights).flatten(1), -1)


def _fourier_tail_scalar(q, ev, rtol):
    """Checked scalar Fourier quadrature for explicitly CPU inputs only."""
    if q.is_cuda or ev.is_cuda:
        raise ArithmeticError("CPU integrand fallback is disabled for CUDA inputs")
    import numpy as np
    from scipy.integrate import IntegrationWarning, quad
    beginning = time.perf_counter()
    tilt_t, _, _, prefactor = _tilt(q.reshape(1), ev)
    t = float(tilt_t.item())
    values = ev.detach().cpu().numpy()
    qt = float(q.item())
    tilted = values / (1 - 2 * t * values)

    def components(w):
        argument = 2 * w * tilted
        magnitude = math.exp(-0.25 * float(np.log1p(argument * argument).sum()))
        phase = 0.5 * float(np.arctan(argument).sum())
        c, s = magnitude * math.cos(phase), magnitude * math.sin(phase)
        denominator = t * t + w * w
        return (t * c + w * s) / denominator, (t * s - w * c) / denominator

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", IntegrationWarning)
        cosine, e1 = quad(lambda w: components(w)[0], 0, np.inf,
                          weight="cos", wvar=qt, epsabs=2e-11,
                          limlst=500, limit=500, maxp1=200)
        sine, e2 = quad(lambda w: components(w)[1], 0, np.inf,
                        weight="sin", wvar=qt, epsabs=2e-11,
                        limlst=500, limit=500, maxp1=200)
    integral = (cosine + sine) / math.pi
    _DIAGNOSTICS["cpu_tail_fallback_calls"] += 1
    _DIAGNOSTICS["cpu_tail_fallback_seconds"] += time.perf_counter() - beginning
    if integral <= 0 or e1 + e2 > max(2e-9, rtol * abs(cosine + sine)):
        raise ArithmeticError("weighted chi-square Fourier integral failed its error check")
    logsf = float(prefactor.item()) + math.log(integral)
    if logsf > 2e-8:
        raise ArithmeticError("weighted chi-square tail exceeded one")
    return max(0.0, -logsf / _LN10)


def weighted_chi2_logsf(q, eigenvalues, *, rtol=1e-5, max_order=8192,
                       allow_cpu_fallback=True):
    """P(sum(lambda_i*chi2_1) >= q), with convergence checked quadrature.

    Rank one and equal eigenvalues are exact chi-square tails. Ranks two to
    four use positive angular integrals. Other ranks use tilted inversion on the
    tensor's device. CPU inputs may use an explicitly counted scalar Fourier
    fallback; CUDA inputs report nonconvergence without a CPU integrand.
    The output shape is the shape of ``q``.
    """
    ev = _positive_eigenvalues(_mixture_spectrum(q, eigenvalues))
    statistic = _tensor(q, ev)
    if bool(((statistic < 0) | ~torch.isfinite(statistic)).any()):
        raise ValueError("weighted chi-square statistic must be finite and nonnegative")
    if not ev.numel():
        return torch.where(statistic == 0, torch.zeros_like(statistic),
                           torch.full_like(statistic, torch.inf))
    largest = ev.max()
    ev = ev / largest
    flat = (statistic / largest).flatten()
    if ev.numel() == 1:
        return chi2_logsf(flat, 1).reshape(statistic.shape)
    if bool((ev.max() - ev.min()) <= ev.max() * 1e-12):
        return chi2_logsf(flat / ev.mean(), float(ev.numel())).reshape(statistic.shape)
    result = torch.zeros_like(flat)
    for start in range(0, flat.numel(), 16):
        chunk = flat[start:start+16]
        nonzero = chunk > 0
        if not bool(nonzero.any()):
            continue
        qq = chunk[nonzero]
        _DIAGNOSTICS["gpu_tail_calls" if qq.is_cuda else "torch_cpu_tail_calls"] += 1
        previous = None
        accepted = torch.zeros_like(qq, dtype=torch.bool)
        current = torch.full_like(qq, torch.nan)
        tilt_parameters = _tilt(qq, ev) if ev.numel() > 4 else None
        for order in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            if order > max_order:
                break
            if ev.numel() <= 4 and order > 2048:
                break  # spherical rules are order^2, unlike the 1-D contour
            if ev.numel() <= 4:
                logsf = _small_rank_integral(qq, ev, order)
                valid = torch.isfinite(logsf)
            else:
                logsf, valid = _tilted_integral(qq, ev, order, tilt_parameters)
                valid &= torch.isfinite(logsf) & (logsf <= 2e-8)
            if previous is not None:
                converged = valid & torch.isfinite(previous) & ((logsf - previous).abs() < rtol)
                # Low-rank positive integrals converge regularly. For an
                # oscillatory contour, demand a minimum order as protection
                # against an accidental agreement between coarse rules.
                if ev.numel() > 4 and order < 512:
                    converged &= False
                update = converged & ~accepted
                current[update] = logsf[update]
                accepted |= converged
                if bool(accepted.all()):
                    break
            previous = logsf
        if bool((~accepted).any()):
            if statistic.is_cuda or not allow_cpu_fallback:
                raise ArithmeticError("weighted chi-square quadrature did not converge; "
                                      "CPU integrand fallback is disabled for this input")
            for index in torch.where(~accepted)[0].tolist():
                current[index] = -_fourier_tail_scalar(qq[index], ev, rtol) * _LN10
        values = torch.zeros_like(chunk)
        values[nonzero] = (-current / _LN10).clamp_min(0)
        result[start:start+chunk.numel()] = values
    return result.reshape(statistic.shape)


def kuonen_logsf(q, eigenvalues):
    """Float64 Kuonen saddlepoint tail on the input device.

    This is the normal-tail approximation used by REGENIE for mixture P
    <=1e-5.  Invalid/central saddlepoints return NaN so that the caller can
    take a checked exact fallback.  It is not an exact chi-square mixture.
    """
    ev = _positive_eigenvalues(_mixture_spectrum(q, eigenvalues))
    statistic = _tensor(q, ev)
    if not ev.numel() or bool(((statistic < 0) | ~torch.isfinite(statistic)).any()):
        raise ValueError("Kuonen requires positive eigenvalues and finite nonnegative q")
    maximum = ev.max()
    ev, flat = ev / maximum, (statistic / maximum).flatten()
    result = torch.full_like(flat, torch.nan)
    positive = flat > 0
    result[~positive] = 0
    if not bool(positive.any()):
        return result.reshape(statistic.shape)
    qq = flat[positive]
    lo = torch.where(qq > ev.sum(), torch.zeros_like(qq), -0.5*ev.numel()/qq)
    hi = torch.full_like(qq, 0.5-1e-8)
    max_derivative = (ev[None, :] / (1-2*hi[:, None]*ev)).sum(-1)
    valid_bound = max_derivative >= qq
    for _ in range(72):
        midpoint = (lo+hi)/2
        derivative = (ev[None, :] / (1-2*midpoint[:, None]*ev)).sum(-1)
        lower = derivative < qq
        lo, hi = torch.where(lower, midpoint, lo), torch.where(lower, hi, midpoint)
    root = (lo+hi)/2
    denominator = 1-2*root[:, None]*ev
    cumulant = -0.5*denominator.log().sum(-1)
    curvature = 2*(ev[None, :]/denominator).square().sum(-1)
    radicand = 2*(qq*root-cumulant)
    w = root.sign()*radicand.clamp_min(0).sqrt()
    u = root*curvature.sqrt()
    ratio = u/w
    transformed = w+ratio.log()/w
    usable = valid_bound & (radicand > 0) & (u.abs() >= 1e-4) & (ratio > 0)
    approximation = -torch.special.log_ndtr(-transformed)/_LN10
    result[positive] = torch.where(usable & torch.isfinite(approximation), approximation,
                                   torch.full_like(approximation, torch.nan))
    return result.reshape(statistic.shape)


def _liu_logsf(q, eigenvalues):
    # For positive eigenvalues, Cauchy-Schwarz gives s1^2<=s2, hence the
    # modified Liu distribution is central, df=(sum(lambda^2))^2/sum(lambda^4).
    c1, c2 = eigenvalues.sum(), eigenvalues.square().sum()
    df = c2.square()/eigenvalues.pow(4).sum()
    transformed = (q-c1)*(df/c2).sqrt()+df
    if bool((transformed < 0).any()):
        raise ArithmeticError("modified Liu fallback has a negative chi-square argument")
    return chi2_logsf(transformed, df)


class _DaviesSpectrum:
    """Read-only spectrum invariants, shared by independent statistic plans."""
    def __init__(self, spectrum):
        self.spectrum = tuple(sorted(spectrum))
        # Keep the original Python left-to-right sums and sqrt arithmetic.
        self.mean = sum(self.spectrum)
        self.sd = math.sqrt(2*sum(x*x for x in self.spectrum))
        self.numpy_weights = None


class _DaviesPlan:
    """Scalar error bounds for the positive, central AS155 special case.

    The characteristic function and all Fourier summation are evaluated by
    PyTorch below.  This controller only handles a kernel's eigenvalues and
    statistic; it never receives participant data.  Its error bounds are
    those of Davies (1980), doi:10.2307/2346911.  The .0866 approximation in
    the Gaussian convergence-factor bound is retained for REGENIE's budget
    decisions.  This is an independent mathematical implementation, not a
    translation or inclusion of the general upstream qfc implementation.
    """
    def __init__(self, spectrum, statistic, accuracy, limit):
        self._spectrum_state = (spectrum if isinstance(spectrum, _DaviesSpectrum)
                                else _DaviesSpectrum(spectrum))
        self.spectrum = self._spectrum_state.spectrum
        self.statistic = statistic
        self.accuracy = accuracy
        self.limit = limit
        self.mean = self._spectrum_state.mean
        self.sd = self._spectrum_state.sd
        self.gaussian_variance = 0.0
        self.bound_evaluations = 0
        self.rules = []

    def _count(self):
        self.bound_evaluations += 1
        if self.bound_evaluations > self.limit:
            raise ArithmeticError("Davies integration parameter budget exhausted")

    @staticmethod
    def _bounded_exp(log_value):
        return 0.0 if log_value < -50 else math.exp(log_value)

    def truncation_error(self, frequency, extra_variance=0.0):
        self._count()
        normal = (self.gaussian_variance+extra_variance)*frequency*frequency
        arguments = [(2*frequency*x)**2 for x in self.spectrum]
        small = sum(math.log1p(x) for x in arguments if x <= 1)
        large = [x for x in arguments if x > 1]
        product_bound = 2*normal+small+sum(math.log(x) for x in large)
        smooth_bound = 2*normal+small+sum(math.log1p(x) for x in large)
        a = self._bounded_exp(-product_bound/4)/math.pi
        b = self._bounded_exp(-smooth_bound/4)/math.pi
        polynomial = 2*a/len(large) if large else 1.0
        combined = 2.5*b if smooth_bound > 1 else 1.0
        gaussian = b/(normal/2) if normal/2 > b else 1.0
        return min(polynomial, combined, gaussian)

    def frequency_limit(self, guess, target_error):
        # Geometric bracketing, followed by four local multiplicative
        # refinements, preserves the error-budget protocol of AS155.
        if self.truncation_error(guess/4) > target_error:
            while self.truncation_error(guess) > target_error:
                guess *= 4
        else:
            guess /= 4
            while self.truncation_error(guess/4) <= target_error:
                guess /= 4
        for factor in (2., 1.4, 1.2, 1.1):
            candidate = guess/factor
            if self.truncation_error(candidate) <= target_error:
                guess = candidate
        return guess

    def convergence_coefficient(self, point):
        self._count()
        distance = abs(point)
        effective_df = 0.
        if point > 0:
            for index, weight in enumerate(self.spectrum):
                remainder = distance-weight
                safe_distance = weight/.0866
                if remainder > safe_distance:
                    distance = remainder
                    continue
                distance = min(distance, safe_distance)
                effective_df = (distance-remainder)/weight+len(self.spectrum)-index-1
                break
        if effective_df > 100:
            return None
        return 2**(effective_df/4)/(math.pi*distance*distance) if distance else math.inf

    def tail_bound(self, parameter):
        self._count()
        offsets = [2*parameter*x for x in self.spectrum]
        # x/(1-x)+log(1-x) is the exact central quadratic-form
        # Chernoff exponent.  A series avoids subtracting near zero.
        def remainder(x):
            if abs(x) < .01:
                return sum((k-1)/k*x**k for k in range(2, 13))
            return x/(1-x)+math.log1p(-x)
        exponent = parameter*parameter*self.gaussian_variance+sum(remainder(x) for x in offsets)
        boundary = parameter*self.gaussian_variance+sum(
                   x/(1-offset) for x, offset in zip(self.spectrum, offsets))
        return self._bounded_exp(-exponent/2), boundary

    def probability_range(self, guess, target_error):
        # Reparameterization keeps positive mgf evaluations away from the
        # first eigenvalue pole; the lower-tail parameter needs no pole cap.
        pole = 2*self.spectrum[-1] if guess > 0 else 0.
        lower_parameter, lower_boundary = 0., self.mean
        upper_parameter = guess
        probability, upper_boundary = self.tail_bound(guess/(1+guess*pole))
        while probability > target_error:
            lower_parameter, lower_boundary = upper_parameter, upper_boundary
            upper_parameter *= 2
            probability, upper_boundary = self.tail_bound(upper_parameter/(1+upper_parameter*pole))
        while (lower_boundary-self.mean)/(upper_boundary-self.mean) < .9:
            midpoint = (lower_parameter+upper_parameter)/2
            probability, boundary = self.tail_bound(midpoint/(1+midpoint*pole))
            if probability > target_error:
                lower_parameter, lower_boundary = midpoint, boundary
            else:
                upper_parameter, upper_boundary = midpoint, boundary
        return upper_parameter, upper_boundary

    def build(self):
        cutoff = self.frequency_limit(16/self.sd, self.accuracy/2)
        if self.statistic and self.spectrum[-1] > .07*self.sd:
            coefficient = self.convergence_coefficient(self.statistic)
            if coefficient is not None:
                variance = self.accuracy/(4*coefficient)
                if self.truncation_error(cutoff, variance) < self.accuracy/5:
                    self.gaussian_variance += variance
                    cutoff = self.frequency_limit(cutoff, self.accuracy/4)
        target = self.accuracy/2
        positive_parameter = 4.5/self.sd
        negative_parameter = -positive_parameter
        remaining_terms = float(self.limit)
        while True:
            positive_parameter, upper = self.probability_range(positive_parameter, target)
            if upper < self.statistic:
                return 0.
            negative_parameter, lower = self.probability_range(negative_parameter, target)
            if lower > self.statistic:
                return 1.
            spacing = 2*math.pi/max(upper-self.statistic, self.statistic-lower)
            required = cutoff/spacing
            auxiliary = 3/math.sqrt(target)
            use_auxiliary = required > 1.5*auxiliary
            if use_auxiliary and auxiliary > remaining_terms:
                raise ArithmeticError("Davies auxiliary term budget exhausted")
            if use_auxiliary:
                auxiliary_terms = int(math.floor(auxiliary+.5))
                auxiliary_spacing = cutoff/auxiliary_terms
                period = 2*math.pi/auxiliary_spacing
                if period > abs(self.statistic):
                    left = self.convergence_coefficient(self.statistic-period)
                    right = self.convergence_coefficient(self.statistic+period)
                    if left is not None and right is not None:
                        variance = .33*target/(1.1*(left+right))
                        self.rules.append((auxiliary_terms, auxiliary_spacing,
                                           self.gaussian_variance, variance))
                        remaining_terms -= auxiliary
                        self.gaussian_variance += variance
                        target *= .67
                        cutoff = self.frequency_limit(cutoff, target/4)
                        target *= .75
                        continue
            if required > remaining_terms:
                raise ArithmeticError("Davies main term budget exhausted")
            self.rules.append((int(math.floor(required+.5)), spacing,
                               self.gaussian_variance, None))
            return None


class _PreparedDaviesSpectrum:
    """Mask-local device/host state; mutable budget state stays in each plan."""
    def __init__(self, eigenvalues, controller, fourier_backend="torch"):
        if controller not in {"auto", "scalar", "numpy"}:
            raise ValueError("Davies controller must be auto, scalar or numpy")
        self.controller = controller
        if fourier_backend not in {"torch", "fused", "auto"}:
            raise ValueError("Fourier backend must be torch, fused or auto")
        self.fourier_backend = fourier_backend
        self.eigenvalues = _positive_eigenvalues(eigenvalues)
        self.maximum = (self.eigenvalues.max() if self.eigenvalues.numel()
                        else self.eigenvalues.new_tensor(1.))
        self.normalized = self.eigenvalues/self.maximum
        self.spectrum = _DaviesSpectrum(self.normalized.detach().cpu().tolist())
        self.plan_class = _DaviesPlan
        if controller == "numpy" or (controller == "auto" and len(self.spectrum.spectrum) >= 1024):
            from ._davies_bounds import NumpyDaviesPlan
            self.plan_class = NumpyDaviesPlan


def _prepare_davies_spectrum(eigenvalues, controller, fourier_backend="torch"):
    """Prepare one immutable spectrum without reusing statistic-specific rules."""
    start = time.perf_counter()
    prepared = _PreparedDaviesSpectrum(eigenvalues, controller, fourier_backend)
    _DIAGNOSTICS["davies_spectrum_preparations"] += 1
    _DIAGNOSTICS["davies_controller_seconds"] += time.perf_counter()-start
    return prepared


def _davies_spectrum_for_inputs(q, eigenvalues, controller, prepared, fourier_backend="torch"):
    if prepared is None:
        return _prepare_davies_spectrum(_mixture_spectrum(q, eigenvalues), controller, fourier_backend)
    if prepared.controller != controller:
        raise ValueError("Prepared Davies spectrum uses a different controller")
    if prepared.fourier_backend != fourier_backend:
        raise ValueError("Prepared Davies spectrum uses a different Fourier backend")
    device = _mixture_device(q, eigenvalues)
    if device is not None and prepared.eigenvalues.device != device:
        return _prepare_davies_spectrum(prepared.eigenvalues.to(device), controller, fourier_backend)
    return prepared


def davies_logsf(q, eigenvalues, *, accuracy=1e-6, limit=10000, controller="auto", fourier_backend="torch",
                 _prepared_spectrum=None):
    """Davies AS155 tails for positive central chi-square(1) mixtures.

    Return ``(-log10(P), ifault)`` tensors.  NaN and a nonzero fault mark a
    failed budget, invalid probability or rounding check.  Fourier inversion
    is float64 PyTorch on the input device.  Scalar error-bound planning is
    CPU; diagnostics include its actual wall time separately. ``controller``
    is auto/scalar/numpy; auto vectorizes bounds for spectra of 1024 or more
    values, retaining ordered float64 sums and the original budget protocol.
    No matrix or
    participant data enters the CPU controller.  These defaults and failures
    drive REGENIE's Kuonen switch even for moderate P.
    """
    if controller not in {"auto", "scalar", "numpy"}:
        raise ValueError("Davies controller must be auto, scalar or numpy")
    prepared = _davies_spectrum_for_inputs(q, eigenvalues, controller, _prepared_spectrum, fourier_backend)
    ev = prepared.eigenvalues
    statistic = _tensor(q, ev)
    if accuracy <= 0 or limit < 1 or bool(((statistic < 0) | ~torch.isfinite(statistic)).any()):
        raise ValueError("Davies needs finite nonnegative q, accuracy>0 and limit>=1")
    if not ev.numel():
        return weighted_chi2_logsf(statistic, ev), torch.zeros_like(statistic, dtype=torch.int64)
    normalized = prepared.normalized
    scaled = (statistic/prepared.maximum).flatten()
    start = time.perf_counter()
    scalar_q = scaled.detach().cpu().tolist()
    jobs, probabilities, fault_codes = [], [], []
    for i, point in enumerate(scalar_q):
        plan = prepared.plan_class(prepared.spectrum, point, accuracy, limit)
        try:
            known_probability = plan.build()
            fault_code = 0
        except (ArithmeticError, OverflowError, ZeroDivisionError, ValueError):
            known_probability = math.nan
            fault_code = 1
        if known_probability is None:
            jobs.extend((i, *rule) for rule in plan.rules)
        probabilities.append(known_probability)
        fault_codes.append(fault_code)
    _DIAGNOSTICS["davies_controller_seconds"] += time.perf_counter()-start
    faults = torch.tensor(fault_codes, dtype=torch.int64, device=ev.device)
    probability = _tensor([math.nan if p is None else p for p in probabilities], ev)
    integrated = torch.zeros_like(scaled, dtype=torch.bool)
    integral, absolute = torch.zeros_like(scaled), torch.zeros_like(scaled)
    # Batch rules with different spacing and Gaussian factors.  Keep the
    # frequency/eigenvalue workspace under about one million doubles.
    from ._fourier_gpu import available as fused_available, fourier_rules
    use_fused = fourier_backend != "torch" and fused_available(ev.device)
    if fourier_backend == "fused" and not use_fused:
        raise RuntimeError("Fused Davies Fourier requires NVIDIA CUDA and Triton")
    batch_size = 64 if use_fused else max(1, min(64, 1048576//(4096*ev.numel())))
    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start:batch_start+batch_size]
        indices = torch.tensor([job[0] for job in batch], dtype=torch.int64, device=ev.device)
        terms = _tensor([job[1] for job in batch], ev)
        spacing = _tensor([job[2] for job in batch], ev)[:, None]
        gaussian = _tensor([job[3] for job in batch], ev)[:, None]
        auxiliary = torch.tensor([job[4] is not None for job in batch], device=ev.device)[:, None]
        extra = _tensor([0. if job[4] is None else job[4] for job in batch], ev)[:, None]
        point = scaled[indices, None]
        batch_integral, batch_absolute = torch.zeros_like(terms), torch.zeros_like(terms)
        if use_fused:
            batch_integral, batch_absolute = fourier_rules(
                normalized, point[:, 0], terms, spacing[:, 0], gaussian[:, 0],
                extra[:, 0], auxiliary[:, 0], maximum_terms=max(job[1] for job in batch))
        else:
            for first in range(0, max(job[1] for job in batch)+1, 4096):
                offsets = torch.arange(first, min(first+4096, max(job[1] for job in batch)+1),
                                       dtype=torch.float64, device=ev.device)
                frequency = (offsets+.5)*spacing
                active = offsets <= terms[:, None]
                arguments = 2*frequency[:, :, None]*normalized
                angles = arguments.atan()
                phase = -point*frequency+angles.sum(-1)/2
                log_magnitude = -gaussian*frequency.square()/2
                log_magnitude -= arguments.square().log1p().sum(-1)/4
                magnitude = torch.where(active & (log_magnitude >= -50), log_magnitude.exp(), 0.)
                amplitudes = (spacing/math.pi)*magnitude/frequency
                damping = extra*frequency.square()/2
                factor = torch.where(damping > 50, 1., -torch.expm1(-damping))
                amplitudes *= torch.where(auxiliary, factor, 1.)
                batch_integral += (phase.sin()*amplitudes).sum(-1)
                batch_absolute += ((point*frequency+angles.abs().sum(-1)/2)*amplitudes).sum(-1)
        integral.index_add_(0, indices, batch_integral)
        absolute.index_add_(0, indices, batch_absolute)
        integrated[indices] = True
        _DIAGNOSTICS["davies_integration_terms"] += sum(job[1]+1 for job in batch)
        _DIAGNOSTICS["davies_fused_rules" if use_fused else "davies_torch_rules"] += len(batch)
    probability = torch.where(integrated, .5+integral, probability)
    rounding_lost = integrated & (absolute+accuracy/10 == absolute)
    faults = torch.where(rounding_lost, 2, faults)
    invalid = (probability < 0) | (probability > 1) | ~torch.isfinite(probability)
    faults = torch.where((faults == 0) & invalid, 1, faults)
    output = torch.where(faults == 0, -probability.log()/_LN10, torch.nan)
    _DIAGNOSTICS["davies_tail_values"] += scaled.numel()
    _DIAGNOSTICS["davies_failure_values"] += int((faults != 0).sum())
    return output.reshape(statistic.shape), faults.reshape(statistic.shape)


def _association_tail(q, eigenvalues, tail_method, *, rank_one_exact=True,
                      davies_controller="auto", fourier_backend="torch", _prepared_spectrum=None):
    """Davies(1e-6/10000) -> Kuonen -> Davies(1e-9/1e6) -> Liu.

    The Kuonen branch is also taken after a Davies budget/rounding failure,
    regardless of P.  Fixed-rho rank-one tests bypass this hierarchy, as in
    REGENIE; the SKAT-O conditional residual integral does not bypass it.
    """
    if tail_method not in ("regenie", "exact"):
        raise ValueError("tail_method must be 'regenie' or 'exact'")
    if _prepared_spectrum is None:
        ev = _positive_eigenvalues(_mixture_spectrum(q, eigenvalues))
    else:
        _prepared_spectrum = _davies_spectrum_for_inputs(q, eigenvalues,
                                                       davies_controller, _prepared_spectrum, fourier_backend)
        ev = _prepared_spectrum.eigenvalues
    qq = _tensor(q, ev)
    if tail_method == "exact" or (rank_one_exact and ev.numel() <= 1):
        return weighted_chi2_logsf(qq, ev)
    if not ev.numel():
        return weighted_chi2_logsf(qq, ev)
    if _prepared_spectrum is None:
        _prepared_spectrum = _prepare_davies_spectrum(ev, davies_controller, fourier_backend)
    flat = qq.flatten()
    output, faults = davies_logsf(flat, ev, controller=davies_controller,
                                 fourier_backend=fourier_backend,
                                 _prepared_spectrum=_prepared_spectrum)
    need_spa = (faults != 0) | ~torch.isfinite(output) | (output >= 5)
    output[need_spa] = torch.nan
    if bool(need_spa.any()):
        spa = kuonen_logsf(flat[need_spa], ev)
        valid = torch.isfinite(spa) & (spa >= 0)
        indices = torch.where(need_spa)[0]
        output[indices[valid]] = spa[valid]
        _DIAGNOSTICS["kuonen_tail_values"] += int(valid.sum())
    failed = ~torch.isfinite(output)
    if bool(failed.any()):
        strict, strict_fault = davies_logsf(flat[failed], ev, accuracy=1e-9, limit=1000000,
                                           controller=davies_controller,
                                           fourier_backend=fourier_backend,
                                           _prepared_spectrum=_prepared_spectrum)
        valid_strict = (strict_fault == 0) & torch.isfinite(strict)
        indices = torch.where(failed)[0]
        output[indices[valid_strict]] = strict[valid_strict]
        if bool((~valid_strict).any()):
            output[indices[~valid_strict]] = _liu_logsf(flat[indices[~valid_strict]], ev)
            _DIAGNOSTICS["liu_fallback_values"] += int((~valid_strict).sum())
    return output.reshape(qq.shape)


def _weighted_inputs(score, covariance, weights=None):
    u = _tensor(score).flatten()
    k = _tensor(covariance, u)
    if k.shape != (u.numel(), u.numel()):
        raise ValueError("score/covariance dimensions differ")
    if bool((~torch.isfinite(u)).any() | (~torch.isfinite(k)).any()):
        raise ValueError("finite score and covariance are required")
    k = k + k.T
    k.mul_(.5)
    if weights is not None:
        w = _tensor(weights, u).flatten()
        if w.shape != u.shape or bool((w < 0).any()):
            raise ValueError("variant weights must match score and be nonnegative")
        u = u*w
        k.mul_(w[:, None])
        k.mul_(w[None, :])
    return u, k


def skat_logp(score_vec, cov_mat, weights=None, *, tail_method="regenie",
              davies_controller="auto", davies_fourier_backend="torch"):
    """SKAT score form, using the source's small-P Kuonen rule by default."""
    u, k = _weighted_inputs(score_vec, cov_mat, weights)
    eigenvalues = _positive_eigenvalues(torch.linalg.eigvalsh(k), 1e-5)
    return _association_tail(u.square().sum(), eigenvalues, tail_method,
                             davies_controller=davies_controller, fourier_backend=davies_fourier_backend)


def skato_logp(score_vec, cov_mat, weights=None, rhos=None, *, integral_rtol=2e-4,
              tail_method="regenie", eigen_backend="dense", secular_min_size=4096,
              secular_root_chunk=256, secular_iterations=64, davies_controller="auto",
              davies_fourier_backend="torch",
              native_validity=False, integral_backend="segmented",
              integral_epsabs=1e-25, integral_epsrel=2.**-13,
              integral_max_intervals=1000):
    """SKAT, burden, rho p-values, actual SKAT-O and SKAT-O-ACAT.

    Returns a dict with SKAT/BURDEN/SKATO/SKATO-ACAT and ``rho_log10ps``/
    ``rhos``.  ACAT-V/ACAT-O require separate allele-frequency weights and
    must be combined by the caller.  The one-dimensional SKAT-O integral
    follows the REGENIE 3.4.1 moment-corrected null approximation. Multi-rho
    searches cap rho at .999, as REGENIE does internally (Data.cpp), rather
    than use an exact rho=1 endpoint. ``rhos`` reports the effective values.
    ``tail_method='regenie'`` uses Davies with the original budget failures
    and P<=1e-5 Kuonen switch; ``'exact'`` retains checked mixture tails.
    ``native_validity=True`` also applies REGENIE's multi-rho validity gate.
    An empty deflated spectrum or negative moment variance returns
    ``kernel_valid=False`` without numerical kernel results. ACAT-V is a
    separate test and remains available to the gene caller. The default
    retains the mathematical API's treatment of degenerate kernels.
    ``integral_backend`` is segmented (the original mathematical API),
    adaptive_x (original chi-square coordinate), adaptive_sqrt (remove
    the density singularity), or qags_x (original coordinate with QAGS
    error selection and Wynn extrapolation). These backends use independent
    device-side Gauss-Kronrod 21 rules with an absolute/relative error budget. An unmet
    budget uses REGENIE's Bonferroni fallback, or returns SKATO=None when
    that fallback is unavailable. Native validity also preserves the
    conditional probability underflow failure and the 10*DBL_MIN probability
    floor, and omits SKATO if its integrated probability exceeds one.
    Other kernel tests remain valid.
    """
    from ._rank_one import rank_one_eigvalsh
    from ._quadrature import integrate_log_gk21, integrate_log_qags, validate_quadrature_parameters
    if integral_backend not in {"segmented", "adaptive_x", "adaptive_sqrt", "qags_x"}:
        raise ValueError("SKAT-O integral backend must be segmented, adaptive_x, adaptive_sqrt or qags_x")
    validate_quadrature_parameters(integral_epsabs, integral_epsrel, integral_max_intervals)
    if davies_controller not in {"auto", "scalar", "numpy"}:
        raise ValueError("Davies controller must be auto, scalar or numpy")
    if eigen_backend not in {"dense", "secular", "auto"}:
        raise ValueError("SKAT-O eigen_backend must be dense, secular, or auto")
    for name,value in (("secular_min_size",secular_min_size),("secular_root_chunk",secular_root_chunk),
                       ("secular_iterations",secular_iterations)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
            raise ValueError(f"{name} must be a positive integer")
    u, k = _weighted_inputs(score_vec, cov_mat, weights)
    use_secular=eigen_backend=="secular" or (eigen_backend=="auto" and k.is_cuda and u.numel()>=secular_min_size)
    rho = _tensor(DEFAULT_RHOS if rhos is None else rhos, u).flatten()
    if not rho.numel() or bool(((rho < 0) | (rho > 1)).any()):
        raise ValueError("SKAT-O rho values must be in [0,1]")
    if rho.numel() > 1:
        rho = rho.clamp(max=0.999)
    ev, basis = torch.linalg.eigh(k)
    raw_ev=ev
    _positive_eigenvalues(ev)
    skat_eigenvalues = _positive_eigenvalues(ev, 1e-5)
    ev = ev.clamp_min(0)
    basis_sum=basis.sum(0)
    v = ev.sqrt() * basis_sum
    del basis  # the M x M eigenvector matrix is no longer needed
    row_sum = k.sum(-1) if native_validity else None
    gamma1 = row_sum.sum() if native_validity else k.sum()
    residual_ev = None
    native_ve = None
    if native_validity and u.numel() > 1 and rho.numel() > 1:
        invalid = {"kernel_valid": False, "rhos": rho,
                   "rho_log10ps": u.new_empty(0)}
        if float(gamma1) <= 0:
            return dict(invalid, kernel_failure="empty_residual_spectrum")
        gamma2 = row_sum.square().sum()
        gamma3 = row_sum @ k @ row_sum
        if use_secular:
            residual_vector = raw_ev*basis_sum/gamma1.sqrt()
            residual_values = rank_one_eigvalsh(raw_ev, residual_vector, -1.,
                    backend="secular", root_chunk=secular_root_chunk,
                    max_iterations=secular_iterations)
            _DIAGNOSTICS["secular_eigen_calls"] += 1
        else:
            # Match get_ztz_evals' division before the outer product. Dividing
            # a completed outer product can turn an exactly zero rank-one
            # residual into small positive eigenvalues and invent kernel rows.
            residual = row_sum[:, None] * (row_sum/gamma1)[None, :]
            residual.neg_().add_(k)
            residual_values = torch.linalg.eigvalsh(residual)
            del residual
            _DIAGNOSTICS["dense_rho_eigen_calls"] += 1
        nonnegative = residual_values[residual_values >= 0]
        cutoff = nonnegative.mean()*1e-5 if nonnegative.numel() else u.new_tensor(0.)
        residual_ev = residual_values[residual_values > cutoff]
        if not residual_ev.numel():
            return dict(invalid, kernel_failure="empty_residual_spectrum")
        native_ve = 4*(gamma3/gamma1 - gamma2.square()/gamma1.square())
        if float(2*residual_ev.square().sum() + native_ve) < 0:
            return dict(invalid, kernel_failure="negative_moment_variance")
    qskat, qburden = u.square().sum(), u.sum().square()
    q = (1 - rho) * qskat + rho * qburden
    rho_eigenvalues = []
    lp = []
    zero_rho_tail = None
    for ri, qi in zip(rho, q):
        is_zero_rho = float(ri) == 0
        if is_zero_rho:
            values = skat_eigenvalues
        elif float(ri) == 1:
            values = v.square().sum().reshape(1)
        elif use_secular:
            values=_positive_eigenvalues(rank_one_eigvalsh((1-ri)*ev,v,float(ri),
                    backend="secular",root_chunk=secular_root_chunk,
                    max_iterations=secular_iterations),1e-5)
            _DIAGNOSTICS["secular_eigen_calls"]+=1
        else:
            _DIAGNOSTICS["dense_rho_eigen_calls"]+=1
            # One dense rho kernel, with the same separately rounded multiply
            # then add. Building a diagonal matrix and multiple M x M scaled
            # terms can exceed the budget for large noncoding masks.
            transformed = (ri*v)[:, None] * v[None, :]
            transformed.diagonal().add_((1-ri)*ev)
            values = _positive_eigenvalues(torch.linalg.eigvalsh(transformed), 1e-5)
            del transformed
        rho_eigenvalues.append(values)
        rho_tail = _association_tail(qi, values, tail_method, davies_controller=davies_controller,
                                     fourier_backend=davies_fourier_backend)
        lp.append(rho_tail)
        if is_zero_rho and zero_rho_tail is None:
            zero_rho_tail = rho_tail
    rho_lp = torch.stack(lp)
    skat = (zero_rho_tail if zero_rho_tail is not None else
            _association_tail(qskat, skat_eigenvalues, tail_method, davies_controller=davies_controller,
                              fourier_backend=davies_fourier_backend))
    burden = chi2_logsf(qburden / gamma1) if float(gamma1) > 0 else u.new_tensor(0.)
    result = {"SKAT": skat, "BURDEN": burden, "SKATO-ACAT": acat_logp(rho_lp),
              "rho_log10ps": rho_lp, "rhos": rho, "kernel_valid": True}
    if u.numel() == 1 or rho.numel() == 1:
        result["SKATO"] = rho_lp.max()
        result["SKATO-ACAT"] = rho_lp[0]
        return result
    if not native_validity and float(gamma1) <= 1e-14 * float(k.diagonal().sum()):
        result["SKATO"] = skat
        return result
    row_sum = k.sum(-1)
    gamma2 = row_sum.square().sum()
    gamma3 = row_sum @ k @ row_sum
    if residual_ev is not None:
        pass  # The native validity gate already computed the deflated spectrum.
    elif use_secular:
        # K=B diag(lambda) B.T and K@1=B(lambda*(B.T@1)). Keep
        # raw lambda for this representation, including harmless zero rounding.
        residual_vector=raw_ev*basis_sum/gamma1.sqrt()
        residual_ev=_positive_eigenvalues(rank_one_eigvalsh(raw_ev,residual_vector,-1.,
                    backend="secular",root_chunk=secular_root_chunk,
                    max_iterations=secular_iterations),1e-5)
        _DIAGNOSTICS["secular_eigen_calls"]+=1
    else:
        _DIAGNOSTICS["dense_rho_eigen_calls"]+=1
        residual = row_sum[:, None] * row_sum[None, :]
        residual.div_(gamma1)
        residual.neg_()
        residual.add_(k)
        residual_ev = _positive_eigenvalues(torch.linalg.eigvalsh(residual), 1e-5)
    if not residual_ev.numel() or (not native_validity and
            float(residual_ev.square().sum()) < float(ev.square().sum()) * 1e-20):
        result["SKATO"] = rho_lp.max()
        return result
    mu = residual_ev.sum()
    v0 = 2 * residual_ev.square().sum()
    ve = native_ve if native_ve is not None else (
        4 * (gamma3 / gamma1 - gamma2.square() / gamma1.square())).clamp_min(0)
    correction = (v0 / (v0 + ve)).sqrt()
    tau = gamma1 * rho + gamma2 / gamma1 * (1 - rho)
    minlp = rho_lp.max()
    maximum_native_logp = -math.log10(10*torch.finfo(torch.float64).tiny)
    if native_validity:
        # Source get_Qmin receives max(10*DBL_MIN, minP). Keep this after
        # the single-site/fixed-rho bypasses, which do not use that integral.
        minlp = minlp.clamp(max=maximum_native_logp)
    minimum_logp = -math.log10(1-torch.finfo(torch.float32).eps) if native_validity else 1e-14
    if float(minlp) <= minimum_logp:
        result["SKATO"] = u.new_tensor(0.)
        return result
    moments1 = torch.stack([e.sum() for e in rho_eigenvalues])
    moments2 = torch.stack([e.square().sum() for e in rho_eigenvalues])
    moments4 = torch.stack([e.pow(4).sum() for e in rho_eigenvalues])
    dfs = moments2.square() / moments4
    critical = moments1 + (chi2_isf_logp(minlp.expand_as(dfs), dfs) - dfs) * (moments2 / dfs).sqrt()
    upper = ((critical + (1 - rho) * mu * (1 - correction) / correction) / tau).min()
    if float(upper) <= 0:
        result["SKATO"] = u.new_tensor(0.)
        return result
    conditional_spectrum = (_prepare_davies_spectrum(residual_ev, davies_controller, davies_fourier_backend)
                            if tail_method == "regenie" else None)
    if integral_backend != "segmented":
        _DIAGNOSTICS["skato_integral_calls"] += 1

        def log_integrand(coordinate):
            x = coordinate.square() if integral_backend == "adaptive_sqrt" else coordinate
            # Preserve source subtraction before dividing by 1-rho. Integrate
            # the entire interval, as the original adaptive route does.
            envelope = ((critical[:, None]-tau[:, None]*x[None, :]) /
                        (1-rho[:, None])).amin(0)
            threshold = ((envelope-mu)*correction+mu).clamp_min(0)
            conditional_lp = _association_tail(threshold, residual_ev, tail_method,
                                              rank_one_exact=False, davies_controller=davies_controller,
                                              fourier_backend=davies_fourier_backend,
                                              _prepared_spectrum=conditional_spectrum)
            log_survival = -conditional_lp*_LN10
            active = (threshold > 0) & (envelope <= mu*1e4)
            # IEEE float64 rounds half the smallest subnormal to zero.
            # CUDA exp can round to zero earlier at its extreme lower end.
            if native_validity and bool((active & (log_survival <= -1075*math.log(2.))).any()):
                # SKATO_integral_fn treats S<=0 as a failed integral. Its
                # deliberate zero for envelope>mu*1e4 is exempt from failure.
                raise ArithmeticError("SKAT-O conditional probability underflowed.")
            log_survival = torch.where(envelope > mu*1e4,
                                        torch.full_like(log_survival, -torch.inf), log_survival)
            if integral_backend == "adaptive_sqrt":
                return log_survival-x/2 + .5*math.log(2/math.pi)
            return log_survival-x/2 - .5*(math.log(2*math.pi)+x.log())

        integration_upper = upper.sqrt() if integral_backend == "adaptive_sqrt" else upper
        try:
            integrator = integrate_log_qags if integral_backend == "qags_x" else integrate_log_gk21
            log_integral, integral_info = integrator(log_integrand, integration_upper,
                epsabs=integral_epsabs, epsrel=integral_epsrel, max_intervals=integral_max_intervals)
        except ArithmeticError:
            integral_info = {"converged": False, "status": "integrand_failure",
                             "evaluations": 0, "intervals": 0}
            log_integral = u.new_tensor(-torch.inf)
        result["integral_diagnostics"] = integral_info
        _DIAGNOSTICS["skato_quadrature_values"] += integral_info["evaluations"]
        _DIAGNOSTICS["skato_quadrature_intervals"] += integral_info["intervals"]
        bonferroni_lp = minlp-math.log10(rho.numel())
        if not integral_info["converged"]:
            _DIAGNOSTICS["skato_integral_failures"] += 1
            result["SKATO"] = bonferroni_lp if float(bonferroni_lp) >= 0 else None
            result["integral_fallback"] = "bonferroni" if result["SKATO"] is not None else "unavailable"
            return result
        log_probability = torch.logaddexp(log_integral, -chi2_logsf(upper)*_LN10)
        selected_lp = torch.maximum(-log_probability/_LN10, bonferroni_lp)
        # REGENIE first applies Bonferroni, then rejects a probability > 1.
        # Clamping either candidate beforehand would invent a P=1 result.
        result["SKATO"] = (None if native_validity and float(selected_lp) < 0
                           else selected_lp.clamp(min=0, max=maximum_native_logp)
                           if native_validity else selected_lp.clamp_min(0))
        return result
    keep = rho < 1
    intercept = critical[keep] / (1 - rho[keep])
    slope = tau[keep] / (1 - rho[keep])
    # Partition at changes in the lower envelope; sqrt(x) removes the
    # chi-square(1) density singularity at zero.
    delta_slope = slope[:, None] - slope[None, :]
    crossings = (intercept[:, None] - intercept[None, :]) / delta_slope
    crossings = crossings[torch.isfinite(crossings) & (crossings > 0) & (crossings < upper)]
    points = sorted(set([0.0, float(upper)] + crossings.tolist()))
    if len(points) > 2:
        mids = u.new_tensor([(a+b)/2 for a, b in zip(points[:-1], points[1:])])
        active_lines = (intercept[:, None] - slope[:, None] * mids).argmin(0).tolist()
        points = [points[0]] + [points[i] for i in range(1, len(points)-1)
                    if active_lines[i-1] != active_lines[i]] + [points[-1]]
    previous = None
    log_probability = None
    _DIAGNOSTICS["skato_integral_calls"] += 1
    for order in (24, 48, 96, 192):
        nodes, quadrature_weights = _legendre(order, u)
        logparts = []
        for left, right in zip(points[:-1], points[1:]):
            lo, hi = math.sqrt(left), math.sqrt(right)
            t = lo + (nodes + 1) * (hi - lo) / 2
            envelope = (intercept[:, None] - slope[:, None] * t.square()).amin(0)
            threshold = ((envelope - mu) * correction + mu).clamp_min(0)
            conditional_lp = _association_tail(threshold, residual_ev, tail_method,
                                              rank_one_exact=False, davies_controller=davies_controller,
                                              fourier_backend=davies_fourier_backend,
                                              _prepared_spectrum=conditional_spectrum)
            log_survival = -conditional_lp*_LN10
            active = (threshold > 0) & (envelope <= mu*1e4)
            if native_validity and bool((active & (log_survival <= -1075*math.log(2.))).any()):
                _DIAGNOSTICS["skato_integral_failures"] += 1
                bonferroni_lp = minlp-math.log10(rho.numel())
                result["SKATO"] = bonferroni_lp if float(bonferroni_lp) >= 0 else None
                result["integral_fallback"] = "bonferroni" if result["SKATO"] is not None else "unavailable"
                return result
            terms = log_survival - t.square() / 2
            if native_validity:
                terms = torch.where(envelope > mu*1e4,
                                    torch.full_like(terms, -torch.inf), terms)
            terms += (quadrature_weights * (hi - lo) / 2).log() + 0.5 * math.log(2 / math.pi)
            logparts.append(torch.logsumexp(terms, 0))
        logparts.append(-chi2_logsf(upper) * _LN10)
        log_probability = torch.logsumexp(torch.stack(logparts), 0)
        if previous is not None and float((log_probability - previous).abs()) < integral_rtol:
            break
        previous = log_probability
    else:
        raise ArithmeticError("SKAT-O rho integral failed its convergence check")
    # This is the same upper bound on the approximation used by REGENIE.
    bonferroni_lp = minlp - math.log10(rho.numel())
    selected_lp = torch.maximum(-log_probability / _LN10, bonferroni_lp)
    result["SKATO"] = (None if native_validity and float(selected_lp) < 0
                       else selected_lp.clamp(min=0, max=maximum_native_logp)
                       if native_validity else selected_lp.clamp_min(0))
    return result


def _normal_quantile_logp(logp):
    """Inverse normal lower CDF from its natural logarithm."""
    moderate = logp > -30
    x = -(-2 * logp).sqrt()
    regular = torch.special.ndtri(logp.exp())
    for _ in range(8):
        logcdf = torch.special.log_ndtr(x)
        derivative = torch.exp(-x.square() / 2 - 0.5 * math.log(2 * math.pi) - logcdf)
        x -= (logcdf - logp) / derivative
    return torch.where(moderate, regular, x)


def _normal_orthant_four(correlation, *, absolute_tolerance=2e-10):
    """Plackett's correlation derivative integrated along I+t*(R-I).

    For each of the six pairs, the derivative is its bivariate density at
    zero times the remaining conditional bivariate orthant probability.
    Reference: Plackett (1954), Biometrika 41, 351-360,
    https://doi.org/10.1093/biomet/41.3-4.351.
    The substitution t=sin(theta) smooths the square-root singularity when
    a pair correlation approaches one. Adaptive 32/64-point integration
    checks each interval on the tensor device; no sample matrix leaves it.
    """
    corr = (correlation + correlation.T) / 2
    torch.linalg.cholesky(corr)
    pairs = tuple(itertools.combinations(range(4), 2))
    pair = torch.tensor(pairs, device=corr.device)
    remaining = torch.tensor([[k for k in range(4) if k not in ij]
                              for ij in pairs], device=corr.device)
    i, j, a, b = pair[:, 0], pair[:, 1], remaining[:, 0], remaining[:, 1]
    rij = corr[i, j]
    ria, rja, rib, rjb, rab = (corr[i, a], corr[j, a], corr[i, b], corr[j, b], corr[a, b])

    def integrate(intervals, order):
        nodes, weights = _legendre(order, corr)
        half = (intervals[:, 1] - intervals[:, 0]) / 2
        theta = (intervals[:, 1] + intervals[:, 0])[:, None] / 2 + half[:, None] * nodes
        t = theta.sin()[..., None]
        rho = t * rij
        denominator = (1-rho) * (1+rho)
        ua, va, ub, vb = t*ria, t*rja, t*rib, t*rjb
        # Diagonalize the conditioned 2x2 pair. Forming the usual quadratic
        # numerator before dividing loses two powers of (1-|rho|) near one.
        plus_a, plus_b = (ua+va)/(2*(1+rho)).sqrt(), (ub+vb)/(2*(1+rho)).sqrt()
        minus_a, minus_b = (ua-va)/(2*(1-rho)).sqrt(), (ub-vb)/(2*(1-rho)).sqrt()
        conditional_a = 1-plus_a.square()-minus_a.square()
        conditional_b = 1-plus_b.square()-minus_b.square()
        conditional_ab = t*rab-plus_a*plus_b-minus_a*minus_b
        conditional_rho = conditional_ab / (conditional_a*conditional_b).clamp_min(
            torch.finfo(corr.dtype).tiny).sqrt()
        conditional_probability = .25 + conditional_rho.clamp(-1, 1).asin()/(2*math.pi)
        derivative = rij/(2*math.pi) / denominator.sqrt() * conditional_probability
        values = derivative.sum(-1) * theta.cos()
        return half * (values * weights).sum(-1)

    intervals = corr.new_tensor([[0., math.pi/2]])
    for _ in range(25):
        low, high = integrate(intervals, 32), integrate(intervals, 64)
        error = (high-low).abs()
        if float(error.sum()) <= absolute_tolerance:
            probability = corr.new_tensor(1/16) + high.sum()
            if not bool(torch.isfinite(probability)) or not -absolute_tolerance <= float(probability) <= .5+absolute_tolerance:
                raise ArithmeticError("four-dimensional orthant integral is outside its probability range")
            return probability.clamp(0, .5)
        if intervals.shape[0] > 4096:
            break
        refine = error > absolute_tolerance / intervals.shape[0]
        selected = intervals[refine]
        midpoint = selected.mean(-1)
        intervals = torch.cat((intervals[~refine], torch.stack((selected[:, 0], midpoint), -1),
                               torch.stack((midpoint, selected[:, 1]), -1)))
    raise ArithmeticError("four-dimensional orthant integral failed its convergence check")


_SOBOL_UNIFORM_CACHE_LIMIT_BYTES = 64*1024*1024
_SOBOL_UNIFORM_CACHE_MAX_ENTRIES = 128
_SOBOL_UNIFORM_CACHE = OrderedDict()
_SOBOL_UNIFORM_CACHE_BYTES = 0
_SOBOL_UNIFORM_CACHE_LOCK = threading.Lock()


def _sobol_uniform(dimension, samples, seed, reference):
    """Read-only, byte-bounded constants with the original Sobol draw order.

    An unspecified seed preserves the original fresh random scramble. CUDA
    events and record_stream also keep cached constants valid for callers
    using different streams, without a device-wide synchronization.
    """
    global _SOBOL_UNIFORM_CACHE_BYTES

    def draw():
        engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
        return engine.draw(samples, dtype=torch.float64).to(reference.device)

    if seed is None or not isinstance(seed, Integral) or not isinstance(samples, Integral):
        _DIAGNOSTICS["orthant_sobol_cache_misses"] += 1
        return draw()
    size_bytes = dimension*samples*8  # The cached draw always uses float64.
    if not 0 < size_bytes <= _SOBOL_UNIFORM_CACHE_LIMIT_BYTES:
        _DIAGNOSTICS["orthant_sobol_cache_misses"] += 1
        return draw()
    key = (dimension, samples, int(seed), str(reference.device), torch.float64)
    with _SOBOL_UNIFORM_CACHE_LOCK:
        if key in _SOBOL_UNIFORM_CACHE:
            uniform, event = _SOBOL_UNIFORM_CACHE.pop(key)
            _SOBOL_UNIFORM_CACHE[key] = (uniform, event)
            if event is not None:
                stream = torch.cuda.current_stream(reference.device)
                stream.wait_event(event)
                uniform.record_stream(stream)
            _DIAGNOSTICS["orthant_sobol_cache_hits"] += 1
            return uniform
        _DIAGNOSTICS["orthant_sobol_cache_misses"] += 1
        # Release retained entries before allocating the next draw. Failed
        # draws do not register a cache entry or increase the byte ledger.
        while (_SOBOL_UNIFORM_CACHE_BYTES+size_bytes > _SOBOL_UNIFORM_CACHE_LIMIT_BYTES
               or len(_SOBOL_UNIFORM_CACHE) >= _SOBOL_UNIFORM_CACHE_MAX_ENTRIES):
            _, (old_uniform, _) = _SOBOL_UNIFORM_CACHE.popitem(last=False)
            _SOBOL_UNIFORM_CACHE_BYTES -= old_uniform.numel()*old_uniform.element_size()
            del old_uniform
        uniform = draw()
        event = None
        if uniform.is_cuda:
            stream = torch.cuda.current_stream(reference.device)
            event = torch.cuda.Event()
            event.record(stream)
            uniform.record_stream(stream)
        _SOBOL_UNIFORM_CACHE[key] = (uniform, event)
        _SOBOL_UNIFORM_CACHE_BYTES += size_bytes
        return uniform


def normal_orthant_probability(covariance, *, qmc_samples=8192, seed=0):
    """Zero-mean positive orthant probability, analytic for dimension<=3.

    Dimension four uses checked deterministic Plackett integration. Larger
    dimensions use a scrambled Sobol Genz conditional integral. Only
    the Sobol constants originate on CPU; triangular integration is on the
    covariance device.  ``qmc_samples`` controls numerical approximation.
    """
    cov = _tensor(covariance)
    n = cov.shape[0]
    if n == 0:
        return cov.new_tensor(1.)
    if cov.shape != (n, n) or bool((cov.diagonal() <= 0).any()):
        raise ValueError("orthant covariance must be positive definite")
    norm = cov.diagonal().sqrt()
    corr = (cov / norm[:, None] / norm[None, :]).clamp(-1, 1)
    if n == 1:
        return cov.new_tensor(0.5)
    if n == 2:
        return 0.25 + torch.asin(corr[0, 1]) / (2 * math.pi)
    if n == 3:
        return 0.125 + (torch.asin(corr[0, 1]) + torch.asin(corr[0, 2])
                        + torch.asin(corr[1, 2])) / (4 * math.pi)
    if n == 4:
        _DIAGNOSTICS["orthant_plackett_calls"] += 1
        return _normal_orthant_four(corr)
    _DIAGNOSTICS["orthant_qmc_calls"] += 1
    chol = torch.linalg.cholesky((corr + corr.T) / 2)
    uniform = _sobol_uniform(n-1, qmc_samples, seed, cov)
    z = torch.zeros((qmc_samples, n), dtype=torch.float64, device=cov.device)
    log_product = torch.zeros(qmc_samples, dtype=torch.float64, device=cov.device)
    for index in range(n):
        mean = z[:, :index] @ chol[index, :index]
        log_conditional = torch.special.log_ndtr(mean / chol[index, index])
        log_product += log_conditional
        if index < n - 1:
            log_tail = log_conditional + torch.log1p(-uniform[:, index])
            z[:, index] = -_normal_quantile_logp(log_tail)
    return torch.exp(torch.logsumexp(log_product, 0) - math.log(qmc_samples))


def chi_bar_weights(gram, *, max_subsets=10, qmc_samples=8192, seed=0,
                    subset_sampling="unique"):
    """NNLS null weights indexed by df=0,...,m.

    Enumerate each active-set size when it has <=max_subsets combinations;
    otherwise sample subsets uniformly, multiply by their number, and
    normalize only the sampled weights to preserve the exact endpoint
    weights. Set max_subsets=0 for complete subset enumeration. Native
    REGENIE draws subsets with replacement; subset_sampling='with_replacement'
    selects that policy, while 'unique' preserves the earlier release's policy.
    The local generator/seed makes each analysis independent of worker order.
    """
    k = _tensor(gram)
    if subset_sampling not in {"unique", "with_replacement"}:
        raise ValueError("subset_sampling must be unique or with_replacement")
    m = k.shape[0]
    norm = k.diagonal().sqrt()
    k = k / norm[:, None] / norm[None, :]
    inverse = torch.linalg.inv(k)
    w = k.new_zeros(m + 1)
    w[0] = normal_orthant_probability(k, qmc_samples=qmc_samples, seed=seed)
    w[m] = normal_orthant_probability(inverse, qmc_samples=qmc_samples, seed=seed+1)
    sampled = []
    generator = torch.Generator(device=k.device)
    generator.manual_seed(seed)
    indices = list(range(m))
    for active_count in range(1, m):
        total = math.comb(m, active_count)
        approximate = max_subsets > 0 and total > max_subsets
        if approximate:
            sampled.append(active_count)
            if subset_sampling == "with_replacement":
                subsets = [tuple(sorted(torch.randperm(m, generator=generator,
                                       device=k.device)[:active_count].tolist()))
                           for _ in range(max_subsets)]
            else:
                subsets = set()
                while len(subsets) < max_subsets:
                    subset = torch.randperm(m, generator=generator, device=k.device)[:active_count]
                    subsets.add(tuple(sorted(subset.tolist())))
        else:
            subsets = itertools.combinations(indices, active_count)
        contributions = []
        for number, subset in enumerate(subsets):
            active = torch.tensor(subset, device=k.device)
            inactive = torch.tensor([i for i in indices if i not in subset], device=k.device)
            kaa = k[active[:, None], active[None, :]]
            kbb = k[inactive[:, None], inactive[None, :]]
            kba = k[inactive[:, None], active[None, :]]
            inverse_active = torch.linalg.inv(kaa)
            conditional = kbb - kba @ inverse_active @ kba.T
            subseed = seed + active_count * 1009 + number * 2
            probability = normal_orthant_probability(inverse_active, qmc_samples=qmc_samples, seed=subseed)
            probability *= normal_orthant_probability(conditional, qmc_samples=qmc_samples, seed=subseed+1)
            contributions.append(probability)
        summed = torch.stack(contributions).sum()
        w[active_count] = summed * (total / len(contributions))
    if sampled:
        mask = torch.zeros_like(w, dtype=torch.bool)
        mask[sampled] = True
        w[mask] *= (1 - w[~mask].sum()) / w[mask].sum()
    # Small QMC integration errors affect even completely enumerated weights.
    if bool((w < -1e-10).any()) or not bool(torch.isfinite(w).all()):
        raise ArithmeticError("invalid chi-bar mixture weights")
    w = w.clamp_min(0)
    largest = w.argmax()
    w[largest] = 1 - (w.sum() - w[largest])
    if bool((w < 0).any()):
        raise ArithmeticError("chi-bar normalization produced a negative weight")
    return w


def nnls_coefficients(score, covariance, *, tolerance=1e-10, max_iterations=1000):
    """GPU active-set minimizer of b' K b /2 - U'b, subject to b>=0."""
    u, k = _weighted_inputs(score, covariance)
    m = u.numel()
    b = torch.zeros_like(u)
    active = torch.zeros(m, dtype=torch.bool, device=u.device)
    gradient = u.clone()
    for _ in range(max_iterations):
        eligible = ~active & (gradient > tolerance)
        if not bool(eligible.any()):
            return b
        enter = torch.where(eligible, gradient, -torch.inf).argmax()
        active[enter] = True
        for _ in range(max_iterations):
            ids = torch.where(active)[0]
            z = torch.zeros_like(b)
            z[ids] = torch.linalg.solve(k[ids[:, None], ids[None, :]], u[ids])
            bad = active & (z <= 0)
            if not bool(bad.any()):
                b = z
                break
            step = (b[bad] / (b[bad] - z[bad])).min()
            b = b + step * (z - b)
            leave = active & (b <= tolerance)
            active[leave] = False
            b[leave] = 0
        else:
            raise ArithmeticError("NNLS inner active-set iteration failed")
        gradient = u - k @ b
    raise ArithmeticError("NNLS active-set iteration failed")


def _chi_bar_logsf(statistic, weights):
    # REGENIE uses the strict upper tail and omits the df=0 atom even at
    # statistic zero: P(Q>0)=1-w0, rather than the inclusive tail P(Q>=0)=1.
    dfs = torch.arange(1, weights.numel(), dtype=torch.float64, device=weights.device)
    lp = chi2_logsf(statistic.expand_as(dfs), dfs)
    return -torch.logsumexp(weights[1:].log() - lp * _LN10, 0) / _LN10


def sbat_logp(score_vec, cov_mat, *, variance_scale=1.0, max_subsets=10,
              qmc_samples=8192, seed=0, mixture_weights=None, subset_sampling="unique"):
    """SBAT from mask score/covariance, with an explicit full-OLS variance.

    To reproduce QT REGENIE, pass ``variance_scale = (r'r-U'K^-1 U)/(N-q-m)``;
    m is the rank after dependent masks have been removed.  The default 1
    instead performs inference under known residual variance.  Returns the
    two directional -log10(P), their ACAT combination, coefficients and
    chi-bar weights.  High-dimensional weights are QMC/subset approximations.
    """
    u, k = _weighted_inputs(score_vec, cov_mat)
    scale = _tensor(variance_scale, u)
    if float(scale) <= 0 or not bool(torch.isfinite(scale)):
        raise ValueError("SBAT full-OLS residual variance must be positive")
    norm = k.diagonal().sqrt()
    if bool((norm <= 0).any()):
        raise ValueError("SBAT requires nonzero independent burden columns")
    normalized = k / norm[:, None] / norm[None, :]
    torch.linalg.cholesky(normalized)  # reject rank deficiency; caller must QR-filter
    normalized_u = u / norm
    positive = nnls_coefficients(normalized_u, normalized)
    negative = nnls_coefficients(-normalized_u, normalized)
    w = chi_bar_weights(normalized, max_subsets=max_subsets, qmc_samples=qmc_samples,
                        seed=seed, subset_sampling=subset_sampling) if mixture_weights is None else _tensor(mixture_weights, u)
    if (w.shape != (u.numel() + 1,) or bool((w < 0).any())
            or not bool(torch.isfinite(w).all()) or not bool(w.sum() > 0)):
        raise ValueError("SBAT mixture weights must have m+1 finite nonnegative entries")
    w = w / w.sum()
    positive_stat = positive @ normalized @ positive / scale
    negative_stat = negative @ normalized @ negative / scale
    pos_lp = _chi_bar_logsf(positive_stat, w)
    neg_lp = _chi_bar_logsf(negative_stat, w)
    return {"SBAT_POS": pos_lp, "SBAT_NEG": neg_lp,
            "SBAT": acat_logp(torch.stack([pos_lp, neg_lp])),
            "mixture_weights": w, "coefficients_positive": positive / norm,
            "coefficients_negative": -negative / norm,
            "statistic_positive": positive_stat, "statistic_negative": negative_stat}


# Spellings accepted by older callers; they have identical mathematical meaning.
skat_o_logp = skato_logp
sbat = sbat_logp
chi2_isf_log10p = chi2_isf_logp
