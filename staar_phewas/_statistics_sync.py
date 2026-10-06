"""FP64 scalar bisection with device flags and unchanged Torch reductions."""
from __future__ import annotations
import math
import torch


def bisection_root(scaled, q, reduce):
    """Preserve each active step; synchronize only the bound and final status.

    The upper/lower midpoint, derivative and reduction are the same operations
    as the reference loop. Inactive iterations copy the converged root without
    changing its bits. No masked-zero reduction or altered stopping root occurs.
    """
    mean = reduce(scaled)
    lower = torch.where(q > mean, q.new_tensor(-.01), -torch.full_like(q, scaled.numel()) / (2*q))
    upper, root = q.new_tensor(.499995), q.new_tensor(0.)
    initial_width = float((upper-lower).abs())
    if initial_width <= 1e-8:
        return root
    iterations = (min(2048, max(1, math.ceil(math.log2(initial_width) - math.log2(1e-8)) + 2))
                  if math.isfinite(initial_width) else 2048)
    alive = torch.ones_like(q, dtype=torch.bool)
    for _ in range(iterations):
        active = alive & ~((upper-lower).abs() <= 1e-8)
        midpoint = (upper+lower)/2
        root = torch.where(active, midpoint, root)
        derivative = reduce(scaled / (1-2*scaled*root))-q
        updating = active & (derivative != 0)
        upper = torch.where(updating & (derivative > 0), root, upper)
        lower = torch.where(updating & ~(derivative > 0), root, lower)
        alive = updating
    if bool(alive & ~((upper-lower).abs() <= 1e-8)):
        raise ArithmeticError("STAAR saddlepoint bisection did not converge")
    return root


def host_flags(*flags):
    """Transfer scalar predicates together; never transfer input observations."""
    return torch.stack(flags).detach().to(device='cpu').tolist()


def quadratic_form_sf_batch(statistic, eigenvalues):
    """Original complete-spectrum Saddle/moment formulas, one row per weight.

    Eigenvalues are computed in the caller's core dtype. This probability-only
    boundary uses FP64 because signed-root cancellation near the mean and
    positive small tails cannot be stored reliably in FP32.
    """
    from .statistics import DegenerateTestError
    raw = eigenvalues.to(torch.float64)
    q_raw = statistic.to(torch.float64)
    if raw.ndim != 2 or q_raw.shape != (len(raw),):
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    finite_spectrum, finite_q, negative_q = host_flags(
        torch.isfinite(raw).all(), torch.isfinite(q_raw).all(), (q_raw < 0).any())
    if not finite_spectrum or not finite_q or negative_q:
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max(dim=1).values
    nonzero = q_raw != 0
    invalid_spectrum, has_nonzero = host_flags((maximum <= 0).any(), nonzero.any())
    if invalid_spectrum:
        raise DegenerateTestError('SKAT covariance has no eigenvalue at or above 1e-8')
    result = torch.ones_like(q_raw)
    if not has_nonzero:return result
    raw, spectrum, q_raw, maximum = raw[nonzero], spectrum[nonzero], q_raw[nonzero], maximum[nonzero]
    scaled, q = spectrum / maximum[:, None], q_raw / maximum
    lower = torch.where(q > scaled.sum(dim=1), q.new_tensor(-.01), -torch.full_like(q, scaled.shape[1]) / (2*q))
    upper, root = torch.full_like(q,.499995), torch.zeros_like(q)
    width = float((upper-lower).abs().max())
    iterations = min(2048, max(1, math.ceil(math.log2(width)-math.log2(1e-8))+2)) if math.isfinite(width) and width>0 else 2048
    alive = torch.ones_like(q,dtype=torch.bool)
    for _ in range(iterations):
        active = alive & ~((upper-lower).abs()<=1e-8)
        root = torch.where(active,(upper+lower)/2,root)
        derivative = (scaled/(1-2*scaled*root[:,None])).sum(dim=1)-q
        updating = active & (derivative!=0)
        upper = torch.where(updating & (derivative>0),root,upper)
        lower = torch.where(updating & ~(derivative>0),root,lower)
        alive = updating
    moment = root.abs()<1e-4
    saddle = ~moment
    # One small transfer preserves the convergence failure priority and both
    # original branch predicates without changing any reduction or root step.
    unconverged, has_moment, has_saddle = host_flags(
        (alive & ~((upper-lower).abs()<=1e-8)).any(), moment.any(), saddle.any())
    if unconverged:
        raise ArithmeticError('STAAR saddlepoint bisection did not converge')
    probabilities = torch.empty_like(q)
    if has_moment:
        moments = raw[moment]
        c1,c2,c4 = moments.sum(dim=1),moments.square().sum(dim=1),moments.pow(4).sum(dim=1)
        if bool(((c2<=0)|(c4<=0)).any()):raise DegenerateTestError('SKAT moment fallback has zero variance')
        dof = c2.square()/c4
        adjusted = (q_raw[moment]-c1)/torch.sqrt(2*c2)*torch.sqrt(2*dof)+dof
        probabilities[moment] = torch.where(adjusted<=0,1.,torch.special.gammaincc(dof/2,adjusted/2))
    if has_saddle:
        s,r = scaled[saddle],root[saddle]
        denominator = 1-2*s*r[:,None]
        cumulant = -.5*torch.log(denominator).sum(dim=1)
        w2 = 2*(r*q[saddle]-cumulant)
        if bool((w2<=0).any()):raise ArithmeticError('STAAR saddlepoint has an invalid signed root')
        signed_root = torch.copysign(torch.sqrt(w2),r)
        second_derivative = 2*(s.square()/denominator.square()).sum(dim=1)
        v = r*torch.sqrt(second_derivative)
        z = signed_root+torch.log(v/signed_root)/signed_root
        probabilities[saddle] = .5*torch.erfc(z/math.sqrt(2))
    result[nonzero] = probabilities
    return result
