"""PyTorch binary saddlepoint score tests.

SPDX-License-Identifier: GPL-3.0-only
Algorithm attribution is recorded in NOTICE.md.
Retains alternate cumulant formulas and failure diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
import torch
from .statistics import _double, _finite, _cct_tensor, _chi1_from_score, annotation_weights


@dataclass
class SPAResult:
    """Device-resident p values, branch flags and Newton iteration counts."""
    pvalues: torch.Tensor
    used_bisection: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor
    iteration_limit: torch.Tensor


def _cumulant(x, mu, g, *, alternate=False):
    z = g * x[None, :]
    if alternate:
        # Preserve the author's alternate expression, including its centering.
        return torch.log((1 - mu) * torch.exp(-z) + mu).sum(0)
    return (-z * mu + torch.log(1 - mu + mu * torch.exp(z))).sum(0)


def _first(x, mu, g, q, *, alternate=False):
    if alternate:
        e = torch.exp(g * x[None, :])
        return (-mu * g + mu * g * e / (mu * e + 1 - mu)).sum(0) - q
    e = torch.exp(-g * x[None, :])
    return (-mu * g + mu * g / (mu + (1 - mu) * e)).sum(0) - q


def _second(x, mu, g, *, alternate=False):
    if alternate:
        e = torch.exp(g * x[None, :])
        # This is the frozen author's expression, without adding exp(z).
        return (mu * (1 - mu) * g.square() / (mu * e + 1 - mu).square()).sum(0)
    e = torch.exp(-g * x[None, :])
    return (mu * (1 - mu) * g.square() * e / (mu + (1 - mu) * e).square()).sum(0)


def _first_finite(x, mu, g, q):
    value = _first(x, mu, g, q)
    bad = ~torch.isfinite(value)
    if bool(bad.any()):
        value = torch.where(bad, _first(x, mu, g, q, alternate=True), value)
    return value


def _second_finite(x, mu, g):
    value = _second(x, mu, g)
    bad = ~torch.isfinite(value)
    if bool(bad.any()):
        value = torch.where(bad, _second(x, mu, g, alternate=True), value)
    return value


def _newton(mu, g, q, tol, max_iter):
    current = torch.zeros_like(q)
    first = _first(current, mu, g, q)
    update = torch.where(first.abs() > tol, current - first / _second(current, mu, g), current)
    counts = torch.zeros_like(q, dtype=torch.int64)
    while True:
        active = (torch.isfinite(update) & ((update-current).abs() > tol)
                  & (_first(update, mu, g, q).abs() > tol) & (counts < max_iter))
        if not bool(active.any()):
            break
        counts = counts + active.to(torch.int64)
        current = torch.where(active, update, current)
        candidate = current - _first_finite(current, mu, g, q) / _second_finite(current, mu, g)
        update = torch.where(active, candidate, update)
    # The original max-iteration guard evaluates its w at xhat=0, yielding
    # an undefined normal statistic; it leaves the last iterate unchanged.
    return torch.where(torch.isfinite(update), update, current), counts


def _same_sign(a, b):
    return (a >= 0) == (b >= 0)


def _bisection(mu, g, q, tol, max_iter):
    a, b = torch.full_like(q, -100.), torch.full_like(q, 100.)
    left, right = a.clone(), b.clone()
    fl, fr = _first_finite(left, mu, g, q), _first_finite(right, mu, g, q)
    phi = (1 + math.sqrt(5)) / 2
    for _ in range(max_iter):
        active = ((right-left).abs() > tol) & _same_sign(fl, fr)
        if not bool(active.any()):
            break
        right = torch.where(active, b - (b-a)/phi, right)
        left = torch.where(active, a + (b-a)/phi, left)
        fl, fr = _first_finite(left, mu, g, q), _first_finite(right, mu, g, q)
        b = torch.where(active & (fr < fl), left, b)
        a = torch.where(active & ~(fr < fl), right, a)
    lo, hi = torch.minimum(left, right), torch.maximum(left, right)
    fl, fr = _first_finite(lo, mu, g, q), _first_finite(hi, mu, g, q)
    valid = torch.isfinite(fl) & torch.isfinite(fr) & ~_same_sign(fl, fr)
    root, derivative = torch.zeros_like(q), torch.ones_like(q)
    # Width 200 / tolerance guarantees fewer than 1075 iterations even at
    # double precision's smallest positive tolerance.
    for _ in range(2048):
        active = valid & ((hi-lo).abs() > tol) & (derivative.abs() > tol)
        if not bool(active.any()):
            return root
        root = torch.where(active, (hi+lo)/2, root)
        derivative = _first_finite(root, mu, g, q)
        same = _same_sign(fl, derivative)
        lo = torch.where(active & same, root, lo)
        hi = torch.where(active & ~same, root, hi)
        fl = _first_finite(lo, mu, g, q)
    raise ArithmeticError("binary SPA bisection did not converge")


def _tail(root, mu, g, q, *, lower):
    w = torch.sqrt(2 * (root*q - _cumulant(root, mu, g)))
    bad = ~torch.isfinite(w)
    if bool(bad.any()):
        alternate = torch.sqrt(2 * (root*q - _cumulant(root, mu, g, alternate=True)))
        w = torch.where(bad, alternate, w)
    w = torch.where(root < 0, -w, w)
    ki = root * torch.sqrt(_second(root, mu, g))
    bad = ~torch.isfinite(ki)
    if bool(bad.any()):
        ki = torch.where(bad, root*torch.sqrt(_second(root, mu, g, alternate=True)), ki)
    z = w + torch.log(ki/w)/w
    probability = 0.5 * torch.erfc((-z if lower else z) / math.sqrt(2))
    return torch.where(root.abs() < 1e-4, 1., probability)


def binary_spa(score, projected_genotype, fitted_probability, *, tol=2**-13,
               max_iter=1000, warn_failures=True) -> SPAResult:
    """Original two-sided SPA for U and residualized genotype columns.

    ``projected_genotype`` is G-XXWX_inv@(XW@G), shape [n,p]; fitted
    probabilities have shape [n]. The null state must come from the same
    samples and design. Every numeric operation retains the input device.
    The frozen package uses Newton, then golden-section/bisection on failure.
    Its failure p=1 is reported by ``failed`` and an optional warning.
    """
    g = _double(projected_genotype)
    u, mu = _double(score, device=g.device), _double(fitted_probability, device=g.device)
    if g.ndim != 2 or not g.shape[0] or not g.shape[1] or u.shape != (g.shape[1],) or mu.shape != (g.shape[0],):
        raise ValueError("SPA requires projected_genotype[n,p], score[p], fitted_probability[n]")
    for value,name in ((g,'projected_genotype'),(u,'score'),(mu,'fitted_probability')):
        _finite(value,name)
    if bool(((mu <= 0) | (mu >= 1)).any()):
        raise ValueError("fitted probabilities must lie strictly between zero and one")
    if not math.isfinite(tol) or tol <= 0 or not isinstance(max_iter,int) or max_iter < 1:
        raise ValueError("tol must be positive and max_iter a positive integer")
    mu = mu[:,None]
    used = torch.zeros_like(u,dtype=torch.bool)
    iteration_limit = torch.zeros_like(used)
    tails, counts = [], torch.zeros_like(u,dtype=torch.int64)
    for lower, q in ((False,u.abs()),(True,-u.abs())):
        root, iterations = _newton(mu,g,q,tol,max_iter)
        counts += iterations
        iteration_limit |= iterations >= max_iter
        value = _tail(root,mu,g,q,lower=lower)
        retry = ~torch.isfinite(value) | (value == 1)
        if bool(retry.any()):
            alternate = _bisection(mu,g[:,retry],q[retry],tol,max_iter)
            value[retry] = _tail(alternate,mu,g[:,retry],q[retry],lower=lower)
            used |= retry
        tails.append(value)
    failed = ~torch.isfinite(tails[0]) | (tails[0]==1) | ~torch.isfinite(tails[1]) | (tails[1]==1)
    result = torch.where(failed, 1., torch.clamp(tails[0]+tails[1],max=1.))
    if warn_failures and bool(failed.any()):
        warnings.warn(f"original binary SPA failed for {int(failed.sum())} tests; returned p=1",RuntimeWarning,stacklevel=2)
    if warn_failures and bool(iteration_limit.any()):
        warnings.warn(f"original binary SPA reached the Newton iteration limit for {int(iteration_limit.sum())} tests",RuntimeWarning,stacklevel=2)
    return SPAResult(result,used,failed,counts,iteration_limit)


def individual_score_test_spa(genotype, residual, fitted_probability, xw,
                              projection_left, *, tol=2**-13,max_iter=1000,
                              normal_pvalues=None,p_filter_cutoff=.05,
                              return_diagnostics=False,_score=None):
    """Reproduce Individual_Score_Test_SPA, optionally applying its P filter.

    xw is [k,n], projection_left is original XXWX_inv [n,k] (or sparse
    mixed-model equivalent). residual is the appropriate original scaled
    residual for related samples, and y-mu for an ordinary binary model.
    ``normal_pvalues`` explicitly requests recalculation only below cutoff.
    """
    g = _double(genotype)
    r, xw = _double(residual,device=g.device), _double(xw,device=g.device)
    left = _double(projection_left,device=g.device)
    if g.ndim != 2 or r.shape != (g.shape[0],) or xw.ndim != 2 or xw.shape[1] != g.shape[0] or left.shape != (g.shape[0],xw.shape[0]):
        raise ValueError("incompatible genotype, residual, xw or projection_left shapes")
    for value,name in ((g,'genotype'),(r,'residual'),(xw,'xw'),(left,'projection_left')):
        _finite(value,name)
    mask = torch.ones(g.shape[1],dtype=torch.bool,device=g.device)
    p = torch.ones(g.shape[1],dtype=g.dtype,device=g.device)
    if normal_pvalues is not None:
        p = _double(normal_pvalues,device=g.device).clone()
        if p.shape != mask.shape or not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()) or not 0 < p_filter_cutoff <= 1:
            raise ValueError("normal_pvalues must be valid p values; cutoff must lie in (0,1]")
        mask = p < p_filter_cutoff
    used, failed = torch.zeros_like(mask), torch.zeros_like(mask)
    limit = torch.zeros_like(mask)
    counts = torch.zeros_like(mask,dtype=torch.int64)
    if bool(mask.any()):
        columns = g[:,mask]
        score = columns.T@r if _score is None else _double(_score,device=g.device)[mask]
        state = binary_spa(score,columns-left@(xw@columns),fitted_probability,tol=tol,max_iter=max_iter)
        p[mask],used[mask],failed[mask],counts[mask] = state.pvalues,state.used_bisection,state.failed,state.iterations
        limit[mask] = state.iteration_limit
    state = SPAResult(p,used,failed,counts,limit)
    return state if return_diagnostics else p


def _spa_combination(values):
    accepted = values[torch.isfinite(values) & (values < 1)]
    # Original wrapper deliberately omits failed ones and NA values. Its
    # sum(...)>0 check also returns one when all accepted values are zero.
    return _cct_tensor(accepted) if accepted.numel() and bool(accepted.sum()>0) else values.new_tensor(1.)


def association_binary_spa(genotype, maf, residual, fitted_probability, xw,
                     projection_left, annotations=None,names=None, *,
                     rare_maf_cutoff=.01,rv_num_cutoff=2,rv_num_cutoff_max=10**9,
                     tol=2**-13,max_iter=1000,spa_p_filter=False,
                     p_filter_cutoff=.05,covariance=None,return_diagnostics=False):
    """WGS_Binary_SPA_sp burden outputs, without an R runtime.

    Genotype is already minor-oriented and mean-imputed. The SPA method
    returns two Beta burden groups and their omnibus WGS-B, as the
    original does. Filtering requires an explicitly supplied score
    covariance from the same fitted null; it is never inferred or replaced.
    """
    g = _double(genotype); f = _double(maf,device=g.device)
    if g.ndim != 2 or f.shape != (g.shape[1],):
        raise ValueError("genotype must have shape [n,p] and maf shape [p]")
    _finite(f,'maf')
    if bool(((f<0)|(f>.5)).any()) or not 0 < rare_maf_cutoff <= .5 or not 1 <= rv_num_cutoff < rv_num_cutoff_max:
        raise ValueError("invalid MAF or variant cutoffs")
    mask = (f>0)&(f<rare_maf_cutoff); count = int(mask.sum())
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    annotation = None if annotations is None else _double(annotations,device=g.device)[mask]
    g,f = g[:,mask],f[mask]
    k = 0 if annotation is None else annotation.shape[1]
    labels = [f'annotation_{i+1}' for i in range(k)] if names is None else list(names)
    if len(labels)!=k or len(set(labels))!=k or any(not isinstance(s,str) or not s for s in labels):
        raise ValueError("names must be distinct nonempty annotation names")
    weights = annotation_weights(f,annotation)[0]
    burdens = g@weights
    normal = None
    if spa_p_filter:
        if covariance is None:
            raise ValueError("SPA filtering requires the fitted score covariance")
        v = _double(covariance,device=g.device)
        if v.shape != (mask.numel(),mask.numel()):
            raise ValueError("covariance must match the original variants")
        v = v[mask][:,mask]
        r = _double(residual,device=g.device)
        scores = weights.T@(g.T@r)
        normal = _chi1_from_score(scores,(weights.T@v@weights).diagonal())
    state = individual_score_test_spa(burdens,residual,fitted_probability,xw,projection_left,
        tol=tol,max_iter=max_iter,normal_pvalues=normal,p_filter_cutoff=p_filter_cutoff,return_diagnostics=True,
        _score=weights.T@(g.T@_double(residual,device=g.device)))
    result = {'num_variant':count,'cMAC':float(g.sum())}
    width = k+1
    for index,beta in enumerate(('1,25','1,1')):
        values = state.pvalues[index*width:(index+1)*width]
        result[f'Burden({beta})'] = float(values[0])
        for name,value in zip(labels,values[1:]):
            result[f'Burden({beta})-{name}'] = float(value)
        result[f'WGS-B({beta})'] = float(_spa_combination(values))
    result['WGS-B'] = float(_spa_combination(state.pvalues))
    return (result,state) if return_diagnostics else result
