"""Joint Gaussian JointAssociation inference in float64 PyTorch.

SPDX-License-Identifier: GPL-3.0-only
Algorithm sources: xihaoli/JointAssociation at
c372e135d88d5537c43af2d0f3e935f47cafd11c (R/JointAssociation.R,
src/JointAssociation_O_SMMAT_sparse.cpp) and GMMAT's glmmkin.multi.ai.
The PheWAS fork calls an unavailable JointAssociation_sp symbol. This module
implements the installed upstream JointAssociation mathematics with explicit MAF.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .statistics import (DegenerateTestError, annotation_weights, cct,
                         quadratic_form_sf)
from .rint import rank_inverse_normal_tensor

MULTIWGS_COMMIT = "c372e135d88d5537c43af2d0f3e935f47cafd11c"


def _double(value, device=None):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def joint_rank_inverse_normal(phenotypes, *, device=None):
    """Columnwise average-tie RINT: Phi^-1((rank-3/8)/(n+1/4)).

    Rank only after forming the shared complete-case sample set. Sorting,
    tie averaging and the inverse normal CDF execute on the tensor device.
    """
    y = _double(phenotypes, device)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2:
        raise ValueError("phenotypes must have shape samples-by-traits with at least two traits")
    _finite(y, "phenotypes")
    return rank_inverse_normal_tensor(y)


def _gls_state(y, x, subject_covariance):
    """Subject blocks of Sigma^-1 and the small trait-by-covariate GLS fit."""
    n, t = y.shape
    p = x.shape[1]
    chol = torch.linalg.cholesky(subject_covariance)
    precision = torch.cholesky_inverse(chol)
    gram = torch.einsum("nab,np,nq->apbq", precision, x, x).reshape(t * p, t * p)
    cov = torch.linalg.inv(gram)
    rhs = torch.einsum("nab,np,nb->ap", precision, x, y).reshape(t * p)
    coefficients = (cov @ rhs).reshape(t, p).T
    py = torch.einsum("nab,nb->na", precision, y - x @ coefficients)
    return precision, cov, coefficients, py


@dataclass
class JointGaussianNullModel:
    """Joint Gaussian fit; covariance across traits is estimated jointly.

    precision has shape [sample, trait, trait]. No sample-by-sample dense
    matrix is formed. Coefficients have shape [covariate, trait].
    """
    sample_ids: tuple
    covariates: torch.Tensor
    scaled_residuals: torch.Tensor
    coefficients: torch.Tensor
    theta: torch.Tensor
    working_theta: torch.Tensor
    fixed_effect_covariance: torch.Tensor
    precision: torch.Tensor
    iterations: int = 0
    converged: bool = True
    relatedness: bool = False
    family: str = "gaussian"
    use_spa: bool = False
    residual_covariance_singular: bool = False
    fit_method: str = "ordinary_REML"
    phenotype: torch.Tensor | None = None
    fitted_values: torch.Tensor | None = None
    phenotype_names: tuple | None = None

    @property
    def n(self):
        return self.covariates.shape[0]

    @property
    def n_pheno(self):
        return self.scaled_residuals.shape[1]

    @property
    def device(self):
        return self.covariates.device

    @property
    def dof(self):
        return self.n - self.covariates.shape[1]

    def save(self, path):
        """Save fitted tensors and alignment IDs; this file is individual data."""
        state = {name: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                 for name, value in vars(self).items()}
        torch.save({"model_type": "JointGaussianNullModel", "version": 1, "state": state}, Path(path))

    @classmethod
    def load(cls, path, *, device="cuda"):
        """Load the model and move its tensors to the requested device."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get("model_type") != "JointGaussianNullModel" or payload.get("version") != 1:
            raise ValueError("file is not a supported joint Gaussian null model")
        return cls(**payload["state"])

    def score_covariance(self, genotype):
        """Return score [traits*variants] and covariance in trait-major order.

        Input genotype is the aligned samples-by-variants minor dosage,
        already imputed and oriented. The trait covariance participates in
        both score and covariance; these are joint association statistics.
        """
        g = _double(genotype, self.device)
        if g.layout != torch.strided:
            g = g.to_dense()
        if g.ndim != 2 or g.shape[0] != self.n or g.shape[1] < 1:
            raise ValueError("genotype must have shape aligned-samples-by-variants")
        _finite(g, "genotype")
        t, p, m = self.n_pheno, self.covariates.shape[1], g.shape[1]
        score = (self.scaled_residuals.T @ g).reshape(t * m)
        # G2=I_traits kron G; compute its cross-products by subject blocks.
        first = torch.einsum("nab,nm,nl->ambl", self.precision, g, g).reshape(t * m, t * m)
        cross = torch.einsum("nab,np,nm->apbm", self.precision, self.covariates, g).reshape(t * p, t * m)
        covariance = first - cross.T @ self.fixed_effect_covariance @ cross
        return score, (covariance + covariance.T) * 0.5


def _components(t, *, device):
    basis, pairs = [], []
    for component in range(2):
        for a in range(t):
            for b in range(a, t):
                item = torch.zeros((t, t), dtype=torch.float64, device=device)
                item[a, b] = item[b, a] = 1
                basis.append(item)
                pairs.append((component, a, b))
    return torch.stack(basis), pairs


def _ai_score_information(y, x, precision, cov, py, derivative):
    # derivative[n,j,a,b] is d Sigma_n / d theta_j.
    n, t = y.shape
    p = x.shape[1]
    dpy = torch.einsum("njab,nb->nja", derivative, py)
    w_dpy = torch.einsum("nab,njb->nja", precision, dpy)
    cross = torch.einsum("np,nja->jap", x, w_dpy).reshape(derivative.shape[1], t * p)
    information = torch.einsum("nia,nja->ij", dpy, w_dpy) - cross @ cov @ cross.T
    quadratic = torch.einsum("na,nja->j", py, dpy)
    trace_wd = torch.einsum("nab,njba->j", precision, derivative)
    wdw = torch.einsum("nab,njbc,ncd->njad", precision, derivative, precision)
    fixed_trace = torch.einsum("njab,np,nq->japbq", wdw, x, x).reshape(derivative.shape[1], t * p, t * p)
    trace_pd = trace_wd - torch.einsum("ab,jba->j", cov, fixed_trace)
    # GMMAT cancels the identical factor 1/2 in both score and AI.
    return quadratic - trace_pd, (information + information.T) * 0.5


def _diagonal_ai(y, x, kinship, maxiter, tol):
    """GMMAT-compatible initialization, AI steps and variance boundary refits."""
    n, t = y.shape
    basis, pairs = _components(t, device=y.device)
    q = len(pairs)
    factors = torch.stack((torch.ones_like(kinship), kinship), dim=1)
    derivative = torch.stack([factors[:, c, None, None] * basis[j]
                              for j, (c, _, _) in enumerate(pairs)], dim=1)
    diag_indices = [j for j, (_, a, b) in enumerate(pairs) if a == b]
    diagonal = torch.tensor(diag_indices, dtype=torch.int64, device=y.device)
    lookup = {(c, a, b): j for j, (c, a, b) in enumerate(pairs)}
    correlation = [(j, lookup[c, a, a], lookup[c, b, b])
                   for j, (c, a, b) in enumerate(pairs) if a < b]
    fixed = torch.zeros(q, dtype=torch.bool, device=y.device)
    fixed_rho = torch.zeros(len(correlation), dtype=y.dtype, device=y.device)
    ordinary = torch.linalg.lstsq(x, y).solution
    total_iterations = 0

    def state(tau):
        sigma = torch.einsum("j,njab->nab", tau, derivative)
        return _gls_state(y, x, sigma)

    def bound(tau, old, rho, *, finish=False):
        tau = tau.clone()
        small = (tau[diagonal] < tol) & ((old[diagonal] < tol) | finish)
        tau[diagonal[small]] = 0
        near = []
        for k, (j, a, b) in enumerate(correlation):
            limit = torch.sqrt(torch.clamp(tau[a] * tau[b], min=0))
            near_old = abs(old[j]) > (1 - 1.01 * tol) * torch.sqrt(torch.clamp(old[a] * old[b], min=0))
            near_new = abs(tau[j]) > (1 - 1.01 * tol) * limit
            if float(rho[k]) != 0:
                tau[j] = rho[k] * limit
            elif bool(near_new & (near_old | finish)):
                tau[j] = torch.sign(tau[j]) * limit
            near.append(near_new)
        valid = bool((tau[diagonal] >= 0).all())
        for j, a, b in correlation:
            valid = valid and bool(abs(tau[j]) <= torch.sqrt(torch.clamp(tau[a] * tau[b], min=0)))
        return tau, valid

    for _refit in range(2 * q + 2):
        free = torch.nonzero(~fixed, as_tuple=True)[0]
        tau = torch.zeros(q, dtype=y.dtype, device=y.device)
        tau[free] = y.reshape(-1).var(unbiased=True) / q
        for j, a, b in correlation:
            tau[j] = 0
        w, cov, coefficients, py = state(tau)
        score, _ = _ai_score_information(y, x, w, cov, py, derivative)
        em = torch.clamp(tau + tau.square() * score / n, min=0)
        tau[free] = em[free]
        for j, _, _ in correlation:
            tau[j] = 0
        previous_alpha = ordinary
        converged = False
        for iteration in range(1, maxiter + 1):
            total_iterations += 1
            old = tau.clone()
            w, cov, coefficients, py = state(old)
            score, information = _ai_score_information(y, x, w, cov, py, derivative)
            if len(free):
                delta = torch.linalg.solve(information[free][:, free], score[free])
                for _step in range(2048):
                    candidate = old.clone()
                    candidate[free] += delta
                    candidate, valid = bound(candidate, old, fixed_rho)
                    if valid:
                        candidate, valid = bound(candidate, old, fixed_rho, finish=True)
                        if valid:
                            tau = candidate
                            break
                    delta /= 2
                else:
                    raise ArithmeticError("joint AI update cannot remain inside covariance parameter space")
            change_alpha = (abs(coefficients - previous_alpha) /
                            (abs(coefficients) + abs(previous_alpha) + tol)).max()
            change_theta = (abs(tau - old) / (abs(tau) + abs(old) + tol)).max()
            previous_alpha = coefficients
            if float(2 * torch.maximum(change_alpha, change_theta)) < tol:
                converged = iteration < maxiter
                break
            if float(abs(tau).max()) > tol ** -2:
                break
        new_fixed = torch.zeros_like(fixed)
        new_fixed[diagonal] = tau[diagonal] < 1.01 * tol
        new_rho = torch.zeros_like(fixed_rho)
        for k, (j, a, b) in enumerate(correlation):
            if bool(abs(tau[j]) > (1 - 1.01 * tol) * torch.sqrt(torch.clamp(tau[a] * tau[b], min=0))):
                new_rho[k] = torch.sign(tau[j])
        if bool((new_fixed == fixed).all() & (new_rho == fixed_rho).all()):
            break
        fixed, fixed_rho = new_fixed, new_rho
    else:
        raise ArithmeticError("joint covariance boundary refits did not stabilize")
    if not converged:
        raise ArithmeticError("joint Gaussian AI REML did not converge")
    components = torch.stack([torch.einsum("j,jab->ab", tau[c * q // 2:(c + 1) * q // 2],
                                         basis[c * q // 2:(c + 1) * q // 2]) for c in range(2)])
    old_components = torch.stack([torch.einsum("j,jab->ab", old[c * q // 2:(c + 1) * q // 2],
                                             basis[c * q // 2:(c + 1) * q // 2]) for c in range(2)])
    # GMMAT returns precision from the last pre-update state; eta uses the
    # old residual covariance, then scaled.residuals uses the final one.
    residual = py @ old_components[0]
    _, info = torch.linalg.cholesky_ex(components[0])
    singular_residual = bool(info != 0)
    # A singular residual component can coexist with positive-definite
    # total covariance. GMMAT's residual/Te representation cannot express
    # this boundary. The joint Gaussian score is still exactly P*y.
    if singular_residual:
        raise ArithmeticError("joint residual covariance is singular: upstream GMMAT's scaled-residual representation fails; robust=True explicitly uses total-covariance P*y instead")
    scaled = torch.linalg.solve(components[0], residual.T).T
    return w, cov, coefficients, scaled, components, old_components, total_iterations, singular_residual


def _factor_reml(y, x, kinship, maxiter, tol):
    """Explicit repaired REML fit with PSD covariance factors on device.

    This optimizer is separate from upstream AI. It avoids the original
    pairwise-correlation update's invalid >=3-trait covariance matrices.
    """
    n, t = y.shape
    beta = torch.linalg.lstsq(x, y).solution
    residual = y - x @ beta
    empirical = residual.T @ residual / (n - x.shape[1])
    whitening = torch.linalg.cholesky(empirical)
    white_y = torch.linalg.solve_triangular(whitening, y.T, upper=False).T
    identity = torch.eye(t, dtype=y.dtype, device=y.device)
    initial = torch.stack((identity / 2, identity / (2 * kinship.mean())))
    initial_factor = torch.linalg.cholesky(initial)
    rows, columns = torch.tril_indices(t, t, device=y.device)
    is_diagonal = rows == columns
    raw = initial_factor[:, rows, columns].clone()
    raw[:, is_diagonal] = torch.log(raw[:, is_diagonal])
    raw.requires_grad_()

    def materialize():
        factors = torch.zeros((2, t, t), dtype=y.dtype, device=y.device)
        entries = torch.where(is_diagonal[None], torch.exp(raw), raw)
        factors[:, rows, columns] = entries
        return factors @ factors.transpose(1, 2)

    def objective():
        components = materialize()
        sigma = components[0] + kinship[:, None, None] * components[1]
        chol = torch.linalg.cholesky(sigma)
        w = torch.cholesky_inverse(chol)
        p = x.shape[1]
        gram = torch.einsum("nab,np,nq->apbq", w, x, x).reshape(t*p, t*p)
        gram_chol = torch.linalg.cholesky(gram)
        rhs = torch.einsum("nab,np,nb->ap", w, x, white_y).reshape(t*p)
        beta = torch.cholesky_solve(rhs[:, None], gram_chol)[:, 0].reshape(t, p).T
        r = white_y - x @ beta
        quadratic = torch.einsum("na,nab,nb->", r, w, r)
        logdet = 2 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum()
        fixed_logdet = 2 * torch.log(gram_chol.diagonal()).sum()
        return (logdet + fixed_logdet + quadratic) / n

    optimizer = torch.optim.LBFGS([raw], max_iter=maxiter, history_size=20,
                                 tolerance_grad=tol * .01,
                                 tolerance_change=tol ** 2,
                                 line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError("factorized joint REML objective is not finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    # A small covariance-factor gradient is the stationarity criterion;
    # an iteration cap or a tiny blocked step is not a convergence claim.
    gradient = float(raw.grad.abs().max())
    if gradient > max(10 * tol, 1e-6):
        raise ArithmeticError(f"factorized joint REML did not converge (maximum gradient {gradient:.3g})")
    with torch.no_grad():
        components = whitening[None] @ materialize().detach() @ whitening.T[None]
        sigma = components[0] + kinship[:, None, None] * components[1]
        w, cov, beta, py = _gls_state(y, x, sigma)
        singular_residual = float(torch.linalg.eigvalsh(components[0]).min()) <= 1e-10
    iterations = int(optimizer.state[raw].get("n_iter", 0))
    return w, cov, beta, py, components, components, iterations, singular_residual


def fit_joint_gaussian_null(phenotypes, covariates=None, *, sample_ids=None,
                            kinship_diagonal=None, edge_rows=None, edge_cols=None,
                            edge_values=None, device="cuda", apply_rint=False,
                            maxiter=500, tol=1e-5, robust=False):
    """Fit an ordinary or diagonal-GRM joint Gaussian REML model on device.

    Input Y is a finite samples-by-traits complete-case matrix. X includes
    the intercept if supplied; otherwise an intercept is added. A supplied
    kinship_diagonal is retained in the covariance model. Nonzero GRM
    edges are rejected because this implementation uses subject blocks.
    At a singular residual-covariance boundary the default raises, matching
    upstream's inability to provide scaled residuals. Explicit robust=True
    selects covariance-factor REML and total-covariance P*y; this repaired
    optimizer is not an upstream numerical-equivalence claim.
    Binary outcomes, repeated observations and multiple GRMs are outside
    this Gaussian diagonal-GRM interface.
    """
    y = _double(phenotypes, device)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("joint Gaussian phenotypes must have at least two columns")
    _finite(y, "phenotypes")
    n, t = y.shape
    x = torch.ones((n, 1), dtype=y.dtype, device=y.device) if covariates is None else _double(covariates, y.device)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 1:
        raise ValueError("covariates must have shape samples-by-covariates including intercept")
    _finite(x, "covariates")
    p = x.shape[1]
    if int(torch.linalg.matrix_rank(x)) != p or n - p < t:
        raise ValueError("joint design must have full column rank and enough residual degrees of freedom")
    ids = tuple(str(i) for i in range(n)) if sample_ids is None else tuple(map(str, sample_ids))
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("sample_ids must contain one distinct ID per complete-case row")
    if not 0 < tol < 1 or maxiter < 2:
        raise ValueError("invalid AI tolerance or maximum iterations")
    if edge_values is not None and bool((_double(edge_values, y.device) != 0).any()):
        raise NotImplementedError("joint Gaussian currently requires a diagonal GRM with no nonzero edges")
    if apply_rint:
        y = joint_rank_inverse_normal(y)
    if kinship_diagonal is None:
        coefficients = torch.linalg.lstsq(x, y).solution
        residual = y - x @ coefficients
        trait_covariance = residual.T @ residual / (n - p)
        precision_trait = torch.linalg.inv(trait_covariance)
        precision = precision_trait.expand(n, t, t)
        cov = torch.kron(trait_covariance.contiguous(), torch.linalg.inv(x.T @ x).contiguous())
        scaled = residual @ precision_trait
        theta = trait_covariance[None]
        return JointGaussianNullModel(ids, x, scaled, coefficients, theta, theta, cov, precision,
                                      phenotype=y, fitted_values=x @ coefficients)
    kinship = _double(kinship_diagonal, y.device)
    if kinship.shape != (n,):
        raise ValueError("kinship_diagonal must have one value per aligned sample")
    _finite(kinship, "kinship_diagonal")
    if bool((kinship < 0).any()) or float(kinship.mean()) <= 0:
        raise ValueError("kinship diagonal must be nonnegative with positive mean")
    if float(kinship.max() - kinship.min()) <= 1e-12:
        raise ValueError("constant diagonal GRM has unidentifiable separate residual and genetic covariance")
    fit = _factor_reml if robust else _diagonal_ai
    precision, cov, coefficients, scaled, theta, working_theta, iterations, singular_residual = fit(y, x, kinship, maxiter, tol)
    return JointGaussianNullModel(ids, x, scaled, coefficients, theta, working_theta, cov,
                                  precision, iterations, True, True,
                                  residual_covariance_singular=singular_residual,
                                  fit_method="factor_REML" if robust else "AI_REML",
                                  phenotype=y,
                                  fitted_values=y - scaled @ theta[0] if not robust else x @ coefficients)


def joint_chi_square(score, covariance):
    """Joint score test, with df equal to the number of trait scores."""
    u = _double(score)
    v = _double(covariance, u.device)
    if u.ndim != 1 or v.shape != (u.numel(), u.numel()):
        raise ValueError("joint score covariance has incompatible dimensions")
    _finite(u, "joint score")
    _finite(v, "joint covariance")
    chol, info = torch.linalg.cholesky_ex((v + v.T) / 2)
    if bool((info != 0).any()):
        raise DegenerateTestError("joint trait score covariance must be positive definite")
    statistic = u @ torch.cholesky_solve(u[:, None], chol)[:, 0]
    return torch.special.gammaincc(u.new_tensor(u.numel() / 2), torch.clamp(statistic, min=0) / 2)


def joint_individual_logp(score,covariance):
    """Positive -log(P) of Individual_Score_Test_sp_multi on the tensor device.

    The original exact determinant-zero branch returns P=1. Integer degrees
    of freedom permit a log-space survival recurrence when P underflows.
    """
    u=_double(score);v=_double(covariance,u.device)
    if u.ndim!=1 or u.numel()<2 or v.shape!=(u.numel(),u.numel()):
        raise ValueError("joint individual score dimensions are incompatible")
    _finite(u,"joint score");_finite(v,"joint covariance")
    if bool(torch.linalg.det(v)==0):return u.new_tensor(0.)
    q=torch.clamp(u@torch.linalg.solve(v,u),min=0)
    if not bool(torch.isfinite(q)):raise ArithmeticError("joint individual score is nonfinite")
    x=q/2;df=u.numel()
    probability=torch.special.gammaincc(u.new_tensor(df/2),x)
    if bool(probability>0):return -torch.log(probability)
    # chi-square with an integer df: exact gamma recurrence in log space.
    if df%2==0:
        j=torch.arange(df//2,dtype=u.dtype,device=u.device)
        terms=torch.where(j==0,torch.zeros_like(j),j*torch.log(x))-torch.lgamma(j+1)
        logsf=-x+torch.logsumexp(terms,dim=0)
    else:
        base=torch.special.log_ndtr(-torch.sqrt(q))+u.new_tensor(2.).log()
        j=torch.arange(df//2,dtype=u.dtype,device=u.device)
        terms=(j+.5)*torch.log(x)-x-torch.lgamma(j+1.5)
        logsf=torch.logsumexp(torch.cat((base.reshape(1),terms)),dim=0)
    return -logsf


def joint_association_test(score, covariance, maf, mac, annotations=None,
                     names: Sequence[str] | None = None, *, n_pheno=None,
                     mac_threshold=10, rare_maf_cutoff=0.01,
                     rv_num_cutoff=2, rv_num_cutoff_max=1_000_000_000,
                     cmac=None, acat_calibration="chi2", **unused):
    """Joint JointAssociation-O, SKAT, burden and ACAT-V from trait-major scores.

    Return column names match single-trait WGS for pipeline assembly.
    SKAT combines trait scores before evaluating its quadratic form;
    burden and ACAT-V each use multivariate chi-square tests. MAC is the
    upstream rounded MAF*2*n, with the <=10 ultra-rare burden rule.
    """
    u = _double(score)
    v, f, mac = (_double(value, u.device) for value in (covariance, maf, mac))
    if f.ndim != 1 or f.numel() < 1 or mac.shape != f.shape or u.ndim != 1:
        raise ValueError("maf/mac must be per-variant vectors and score a trait-major vector")
    m = f.numel()
    t = u.numel() // m if n_pheno is None else n_pheno
    if t < 2 or u.numel() != t * m or v.shape != (t * m, t * m):
        raise ValueError("joint score and covariance must have traits*variants dimensions")
    if acat_calibration != "chi2":
        raise ValueError("JointAssociation uses chi-square calibration for its joint ACAT-V components")
    for value, name in ((u, "score"), (v, "covariance"), (f, "maf"), (mac, "mac")):
        _finite(value, name)
    if bool(((f < 0) | (f > .5) | (mac < 0)).any()):
        raise ValueError("maf must be in [0,.5] and mac nonnegative")
    if not 0 < rare_maf_cutoff <= .5 or not 1 <= rv_num_cutoff < rv_num_cutoff_max or mac_threshold < 0:
        raise ValueError("invalid rare-variant or MAC thresholds")
    if not bool(torch.allclose(v, v.T, rtol=1e-10, atol=1e-10)):
        raise ValueError("joint covariance must be symmetric")
    rare = (f > 0) & (f < rare_maf_cutoff)
    count = int(rare.sum())
    if not rv_num_cutoff <= count < rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    indices = torch.nonzero(rare.repeat(t), as_tuple=True)[0]
    u, v, f, mac = u[indices], v[indices][:, indices], f[rare], mac[rare]
    phred = None if annotations is None else _double(annotations, u.device)
    if phred is not None:
        if phred.ndim != 2 or phred.shape[0] != m:
            raise ValueError("annotations must have one row per original variant")
        phred = phred[rare]
    k = 0 if phred is None else phred.shape[1]
    labels = [f"annotation_{i+1}" for i in range(k)] if names is None else list(names)
    if len(labels) != k or len(set(labels)) != k or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("annotation names must be nonempty and distinct")
    wb, ws, wa = annotation_weights(f, phred)
    scores = u.reshape(t, count)
    cov4 = v.reshape(t, count, t, count)
    common = mac > mac_threshold
    ultra = ~common
    common_indices = torch.nonzero(common, as_tuple=True)[0]
    common_p = torch.stack([joint_chi_square(scores[:, j], cov4[:, j, :, j]) for j in common_indices]) if len(common_indices) else u.new_empty(0)
    number_weights = wb.shape[1]
    pvalues = u.new_empty((3, number_weights))
    for j in range(number_weights):
        skat_weight = ws[:, j].repeat(t)
        weighted_covariance = v * skat_weight[:, None] * skat_weight[None, :]
        spectrum = torch.linalg.eigvalsh(weighted_covariance)
        pvalues[0, j] = quadratic_form_sf(torch.sum((u * skat_weight).square()), spectrum)
        burden_weight = wb[:, j]
        burden_score = scores @ burden_weight
        burden_covariance = torch.einsum("m,ambn,n->ab", burden_weight, cov4, burden_weight)
        pvalues[1, j] = joint_chi_square(burden_score, burden_covariance)
        acat_p, acat_w = common_p, wa[common, j]
        if bool(ultra.any()):
            ultra_weight = burden_weight * ultra
            ultra_score = scores @ ultra_weight
            ultra_covariance = torch.einsum("m,ambn,n->ab", ultra_weight, cov4, ultra_weight)
            acat_p = torch.cat((common_p, joint_chi_square(ultra_score, ultra_covariance).reshape(1)))
            acat_w = torch.cat((acat_w, wa[ultra, j].mean().reshape(1)))
        pvalues[2, j] = cct(acat_p, acat_w, internal=True)
    result = {"num_variant": count, "cMAC": float(mac.sum()) if cmac is None else float(cmac)}
    width = k + 1
    for row, method, combined in ((0, "SKAT", "WGS-S"), (1, "Burden", "WGS-B"), (2, "ACAT-V", "WGS-A")):
        for beta_index, beta in enumerate(("1,25", "1,1")):
            values = pvalues[row, beta_index * width:(beta_index + 1) * width]
            base = f"{method}({beta})"
            result[base] = float(values[0])
            result.update({f"{base}-{name}": float(value) for name, value in zip(labels, values[1:])})
            result[f"{combined}({beta})"] = cct(values)
    result["ACAT-O"] = cct(pvalues[:, (0, width)].reshape(-1))
    result["WGS-O"] = cct(pvalues.reshape(-1))
    return result
