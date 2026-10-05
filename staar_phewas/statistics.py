"""Float64 STAAR score tests, without an R runtime.

SPDX-License-Identifier: GPL-3.0-only
Mathematical/algorithmic source: Xihao Li, Zilin Li and STAAR contributors,
https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05
(R/STAAR_sp.R; src/STAAR_O*.cpp;
src/Saddle.cpp; src/Bisection.cpp; R/CCT.R; src/CCT_pval.cpp).
This module preserves their saddlepoint and fourth-moment fallback choices.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch

from ._precision_eigen import (refine_near_mean_spectrum, record_gpu_eigen_route,
                               record_ordered_tail)


class DegenerateTestError(ValueError):
    """The requested test has no positive variance or valid weight mass."""


def _double(value, *, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def score_covariance(
    genotype,
    residual,
    *,
    covariates=None,
    working_weights=None,
    dispersion: float = 1.0,
    projector=None,
    precision=None,
    precision_covariates=None,
    fixed_effect_covariance=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute U=G' r and V=G' P G on G's device in float64.

    Supply exactly one of (a) covariates with optional working_weights,
    (b) an already fitted projector P, or (c) precision=Sigma_i,
    precision_covariates=Sigma_iX, fixed_effect_covariance=cov.
    The latter two accept sparse COO/CSR matrices. For ordinary Gaussian
    models, dispersion is sigma**2 and residual is the unscaled residual.
    For mixed models, residual must already be the fitted scaled residual;
    dispersion must be 1 because P/Sigma_i include the fitted scale.
    No null-model fitting, sample alignment, imputation or allele flipping
    takes place here.
    """
    g = _double(genotype)
    if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1:
        raise ValueError("genotype must be a nonempty samples-by-variants matrix")
    if g.layout != torch.strided:
        g = g.to_dense()
    r = _double(residual, device=g.device)
    if r.ndim != 1 or r.shape[0] != g.shape[0]:
        raise ValueError("residual must have one value per genotype row")
    _finite(g, "genotype")
    _finite(r, "residual")
    if not math.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("dispersion must be positive and finite")
    modes = sum((covariates is not None, projector is not None, precision is not None))
    if modes != 1:
        raise ValueError("supply exactly one of covariates, projector, precision")
    if projector is not None:
        if dispersion != 1 or working_weights is not None:
            raise ValueError("an explicit projector already contains fitted scaling")
        p = _double(projector, device=g.device)
        if p.shape != (g.shape[0], g.shape[0]):
            raise ValueError("projector must have shape samples-by-samples")
        projected_g = p @ g
        covariance = projected_g.T @ g
    elif precision is not None:
        if dispersion != 1 or working_weights is not None:
            raise ValueError("precision already contains fitted scaling")
        if precision_covariates is None or fixed_effect_covariance is None:
            raise ValueError("precision also requires precision_covariates and fixed_effect_covariance")
        inverse_sigma = _double(precision, device=g.device)
        inverse_sigma_x = _double(precision_covariates, device=g.device)
        cov = _double(fixed_effect_covariance, device=g.device)
        if inverse_sigma.shape != (g.shape[0], g.shape[0]):
            raise ValueError("precision must have shape samples-by-samples")
        if inverse_sigma_x.ndim != 2 or inverse_sigma_x.shape[0] != g.shape[0]:
            raise ValueError("precision_covariates must have one row per sample")
        if cov.shape != (inverse_sigma_x.shape[1], inverse_sigma_x.shape[1]):
            raise ValueError("fixed_effect_covariance has an incompatible shape")
        cross = inverse_sigma_x.T @ g
        covariance = (inverse_sigma @ g).T @ g - cross.T @ cov @ cross
    else:
        x = _double(covariates, device=g.device)
        if x.ndim != 2 or x.shape[0] != g.shape[0]:
            raise ValueError("covariates must have one row per genotype sample")
        _finite(x, "covariates")
        v = torch.ones(g.shape[0], dtype=g.dtype, device=g.device)
        if working_weights is not None:
            v = _double(working_weights, device=g.device)
            if v.shape != (g.shape[0],):
                raise ValueError("working_weights must have one value per sample")
            _finite(v, "working_weights")
            if bool((v < 0).any()):
                raise ValueError("working_weights cannot be negative")
        wg = v[:, None] * g
        cross = x.T @ wg
        if x.shape[1]:
            gram = x.T @ (v[:, None] * x)
            covariance = g.T @ wg - cross.T @ torch.linalg.solve(gram, cross)
        else:
            covariance = g.T @ wg
        covariance = dispersion * covariance
    score = g.T @ r
    _finite(covariance, "score covariance")
    return score, (covariance + covariance.T) * 0.5


def _cauchy_sf_tensor(statistic: torch.Tensor) -> torch.Tensor:
    reciprocal = torch.atan(1 / statistic) / math.pi
    far = torch.where(statistic > 0, reciprocal, 1 + reciprocal)
    return torch.where(statistic.abs() > 1, far, 0.5 - torch.atan(statistic) / math.pi)


def _cct_tensor(pvalues, weights=None, *, internal: bool = False) -> torch.Tensor:
    p = _double(pvalues).reshape(-1)
    if not p.numel() or not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
    if not internal:
        if bool((p == 0).any()) and bool((p == 1).any()):
            raise ValueError("cannot combine both exact 0 and exact 1 pvalues")
        if bool((p == 0).any()):
            return p.new_tensor(0.)
        if bool((p == 1).any()):
            warnings.warn("there are pvalues that are exactly 1", RuntimeWarning, stacklevel=3)
            return p.new_tensor(1.)
    if weights is None:
        w = torch.full_like(p, 1 / p.numel())
    else:
        w = _double(weights, device=p.device).reshape(-1)
        if w.shape != p.shape or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("weights must be finite, nonnegative and match pvalues")
        mass = w.sum()
        if not bool(torch.isfinite(mass)) or bool(mass <= 0):
            raise DegenerateTestError("CCT requires a positive finite weight sum")
        w = w / mass
    small = p < 1e-16
    statistic = (w[small] / p[small] / math.pi).sum()
    statistic = statistic + (w[~small] * torch.tan((0.5 - p[~small]) * math.pi)).sum()
    if bool(torch.isnan(statistic)):
        raise DegenerateTestError("CCT statistic is undefined")
    if bool(statistic > 1e15):
        return (1 / statistic) / math.pi
    tail = _cauchy_sf_tensor(statistic)
    return tail if internal or bool(statistic <= 0) else 1 - (1 - tail)


def cct(pvalues, weights=None, *, internal: bool = False) -> float:
    """Cauchy combination on the input device, with R/C++ boundary rules.

    Default matches exported R CCT: any exact 0 returns 0, any exact 1
    returns 1 with a warning, and simultaneous 0/1 is an error (even for
    zero-weight elements). internal=True matches CCT_pval.cpp used inside
    ACAT-V. Invalid or zero weight mass raises an error instead of R NaN.
    """
    return float(_cct_tensor(pvalues, weights, internal=internal))


def _ordered_sum(values: torch.Tensor) -> torch.Tensor:
    """Original scalar-addition order for CUDA rows; CPU retains Torch sum."""
    if values.is_cuda:
        from ._ordered_cuda import ordered_rows_cuda
        return ordered_rows_cuda(values.reshape(values.shape[0], -1)).reshape(values.shape[1:])
    return values.sum(dim=0)


def _quadratic_form_sf_tensor(statistic, eigenvalues, *, moment_eigenvalues=None, reference_reduction=False) -> torch.Tensor:
    raw = _double(eigenvalues).reshape(-1)
    q_raw = _double(statistic, device=raw.device)
    if not raw.numel() or not bool(torch.isfinite(raw).all()) or q_raw.numel() != 1 or not bool(torch.isfinite(q_raw)) or bool(q_raw < 0):
        raise ValueError("quadratic form requires a nonnegative finite statistic and finite spectrum")
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max()
    if bool(maximum <= 0):
        raise DegenerateTestError("SKAT covariance has no eigenvalue at or above 1e-8")
    if bool(q_raw == 0):
        return raw.new_tensor(1.)
    if reference_reduction and raw.is_cuda:
        record_ordered_tail()
    reduce = _ordered_sum if reference_reduction else lambda values: values.sum()
    scaled, q = spectrum / maximum, q_raw / maximum
    lower = q.new_tensor(-0.01) if bool(q > reduce(scaled)) else -torch.full_like(q, scaled.numel()) / (2 * q)
    upper, root = q.new_tensor(0.499995), q.new_tensor(0.)
    for _ in range(2048):
        if bool((upper - lower).abs() <= 1e-8):
            break
        root = (upper + lower) / 2
        derivative = reduce(scaled / (1 - 2 * scaled * root)) - q
        if bool(derivative == 0):
            break
        upper = torch.where(derivative > 0, root, upper)
        lower = torch.where(derivative > 0, lower, root)
    else:
        raise ArithmeticError("STAAR saddlepoint bisection did not converge")
    if bool(root.abs() < 1e-4):
        moments = raw if moment_eigenvalues is None else _double(moment_eigenvalues, device=raw.device).reshape(-1)
        _finite(moments, "moment_eigenvalues")
        c1, c2, c4 = moments.sum(), moments.square().sum(), moments.pow(4).sum()
        if bool(c2 <= 0) or bool(c4 <= 0):
            raise DegenerateTestError("SKAT moment fallback has zero variance")
        dof = c2.square() / c4
        adjusted = (q_raw - c1) / torch.sqrt(2 * c2) * torch.sqrt(2 * dof) + dof
        return raw.new_tensor(1.) if bool(adjusted <= 0) else torch.special.gammaincc(dof / 2, adjusted / 2)
    cumulant = -0.5 * reduce(torch.log(1 - 2 * scaled * root))
    w2 = 2 * (root * q - cumulant)
    if bool(w2 <= 0):
        raise ArithmeticError("STAAR saddlepoint has an invalid signed root")
    signed_root = torch.copysign(torch.sqrt(w2), root)
    second_derivative = 2 * reduce(scaled.square() / (1 - 2 * scaled * root).square())
    v = root * torch.sqrt(second_derivative)
    z = signed_root + torch.log(v / signed_root) / signed_root
    return 0.5 * torch.erfc(z / math.sqrt(2))


def quadratic_form_sf(statistic: float, eigenvalues, *, moment_eigenvalues=None) -> float:
    """STAAR Saddle tail and fourth-moment fallback in float64 PyTorch.

    The absolute eigenvalue cutoff is 1e-8, the bisection tolerance 1e-8,
    and the moment switch abs(root)<1e-4, matching the original package.
    The complete unthresholded spectrum supplies the fallback moments.
    """
    return float(_quadratic_form_sf_tensor(statistic, eigenvalues, moment_eigenvalues=moment_eigenvalues))


def _beta_continued_fraction(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Modified Lentz evaluation of the incomplete-beta continued fraction."""
    tiny = 1e-300
    def nonzero(value):
        return torch.where(value.abs() < tiny, torch.copysign(torch.full_like(value, tiny), value), value)
    qab, qap, qam = a + b, a + 1, a - 1
    c = torch.ones_like(x)
    d = 1 / nonzero(1 - qab * x / qap)
    h = d
    for m in range(1, 10001):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d, c = 1 / nonzero(1 + aa * d), nonzero(1 + aa / c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d, c = 1 / nonzero(1 + aa * d), nonzero(1 + aa / c)
        delta = d * c
        h = h * delta
        if bool(((delta - 1).abs() <= 8 * torch.finfo(x.dtype).eps).all()):
            return h
    raise ArithmeticError("Student t incomplete-beta continued fraction did not converge")


def _student_t_two_sided(t_squared: torch.Tensor, dof: int) -> torch.Tensor:
    """P(|T_dof| >= sqrt(t_squared)) via the regularized incomplete beta."""
    x = dof / (dof + t_squared)
    complement = t_squared / (dof + t_squared)
    result = torch.ones_like(x)
    interior = (x > 0) & (x < 1)
    z = x[interior]
    if z.numel():
        a, b = z.new_tensor(dof / 2), z.new_tensor(0.5)
        flip = z > (a + 1) / (a + b + 2)
        aa, bb = torch.where(flip, b, a), torch.where(flip, a, b)
        # Compute the complement directly: 1-x loses small t^2/dof digits.
        xx = torch.where(flip, complement[interior], z)
        leading = torch.exp(torch.lgamma(aa + bb) - torch.lgamma(aa) - torch.lgamma(bb)
                            + aa * torch.log(xx) + bb * torch.log1p(-xx))
        tail = leading * _beta_continued_fraction(aa, bb, xx) / aa
        result[interior] = torch.where(flip, 1 - tail, tail)
    result[x == 0] = 0
    return result


def annotation_weights(maf, annotations=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return burden, SKAT and ACAT-V weights in beta-then-annotation order."""
    f = _double(maf)
    if f.ndim != 1 or not f.numel():
        raise ValueError("maf must be a nonempty vector")
    _finite(f, "maf")
    if bool(((f <= 0) | (f > 0.5)).any()):
        raise ValueError("weights require minor-allele frequencies in (0, 0.5]")
    phred = None
    if annotations is not None:
        phred = _double(annotations, device=f.device)
        if phred.ndim != 2 or phred.shape[0] != f.numel():
            raise ValueError("annotations must have shape variants-by-annotations")
        _finite(phred, "annotations")
        if bool((phred < 0).any()):
            raise ValueError("PHRED annotations cannot be negative")
    # The original scalar libm transformations are retained at this small
    # weight boundary. Score/covariance and probability calculations stay on
    # the input device; the helper reports its CPU and transfer work.
    from ._reference_weights import reference_annotation_weights
    return reference_annotation_weights(f, phred)


def _chi1_from_score(score: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    if bool((variance <= 0).any()):
        raise DegenerateTestError("score test variance must be positive")
    return torch.erfc(torch.sqrt(score.square() / variance / 2))


def staar_test(
    score,
    covariance,
    maf,
    mac,
    annotations=None,
    names: Sequence[str] | None = None,
    *,
    acat_calibration: str = "chi2",
    dof: int | None = None,
    n: int | None = None,
    covariate_count: int | None = None,
    mac_threshold: int = 10,
    rare_maf_cutoff: float = 0.01,
    rv_num_cutoff: int = 2,
    rv_num_cutoff_max: int = 1_000_000_000,
    cmac: float | None = None,
    _skat_pvalues: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Return STAAR columns from already oriented, imputed score inputs.

    covariance includes the fitted dispersion. acat_calibration='chi2'
    is the SMMAT/relatedness (or binary ordinary) branch;
    'gaussian_glm' requires dof=n-rank(X), and uses t(dof-1) only for
    ACAT-V variants with MAC>mac_threshold. It does not change burden.
    annotations are PHRED values, not pre-transformed ranks. This routine
    never fits or approximates a binary SPA or mixed-model null fit.
    cmac optionally supplies exact sum(G_rare), important after imputation;
    otherwise the sum of the caller's MAC values is returned.
    """
    u = _double(score)
    v = _double(covariance, device=u.device)
    f = _double(maf, device=u.device)
    m = _double(mac, device=u.device)
    if u.ndim != 1 or v.shape != (u.numel(), u.numel()) or f.shape != u.shape or m.shape != u.shape:
        raise ValueError("score/maf/mac must be vectors and covariance must be variants-by-variants")
    for value, name in ((u, "score"), (v, "covariance"), (f, "maf"), (m, "mac")):
        _finite(value, name)
    if bool(((f < 0) | (f > 0.5) | (m < 0)).any()):
        raise ValueError("maf must lie in [0,.5] and mac cannot be negative")
    if not 0 < rare_maf_cutoff <= 0.5 or not 1 <= rv_num_cutoff < rv_num_cutoff_max:
        raise ValueError("invalid rare-variant cutoffs")
    if acat_calibration not in ("chi2", "gaussian_glm"):
        raise ValueError("acat_calibration must be chi2 or gaussian_glm")
    if acat_calibration == "gaussian_glm":
        if dof is None and n is not None and covariate_count is not None:
            dof = n - covariate_count
        if dof is None or dof <= 1:
            raise ValueError("gaussian_glm ACAT-V requires residual dof=n-rank(X)>1")
    if mac_threshold < 0:
        raise ValueError("mac_threshold cannot be negative")
    mask = (f > 0) & (f < rare_maf_cutoff)
    count = int(mask.sum())
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    phred = None
    if annotations is not None:
        phred = _double(annotations, device=u.device)
        if phred.ndim != 2 or phred.shape[0] != u.numel():
            raise ValueError("annotations must have one row per original variant")
        phred = phred[mask]
    u, v, f, m = u[mask], v[mask][:, mask], f[mask], m[mask]
    if not bool(torch.allclose(v, v.T, atol=1e-10, rtol=1e-10)):
        raise ValueError("covariance must be symmetric")
    reference_covariance = v
    v = (v + v.T) * 0.5
    k = 0 if phred is None else phred.shape[1]
    labels = [f"annotation_{i + 1}" for i in range(k)] if names is None else list(names)
    if len(labels) != k or len(set(labels)) != k or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("names must be nonempty distinct annotation names")
    burden_weights, skat_weights, acat_weights = annotation_weights(f, phred)
    common = m > mac_threshold
    very_rare = ~common
    common_p = _chi1_from_score(u[common], v.diagonal()[common])
    if acat_calibration == "gaussian_glm" and bool(common.any()):
        explained = u[common].square() / v.diagonal()[common]
        if bool((explained > dof).any()):
            raise ArithmeticError("Gaussian single-variant explained sum of squares exceeds the null total")
        t_squared = explained / (dof - explained) * (dof - 1)
        common_p = _student_t_two_sided(t_squared, dof - 1)
    number_weights = burden_weights.shape[1]
    pvalues = torch.empty((3, number_weights), dtype=u.dtype, device=u.device)
    if _skat_pvalues is not None:
        _skat_pvalues = _double(_skat_pvalues, device=u.device)
        if _skat_pvalues.shape != (number_weights,):
            raise ValueError("precomputed SKAT values do not match the annotation weights")
    small_spectra = None
    if _skat_pvalues is None and u.is_cuda and len(u) <= 32 and number_weights > 1:
        # A natural batch of annotation weights selects the small-matrix
        # CUDA Jacobi solver. Reuse that spectrum through the scalar tail.
        matrices = torch.stack([v * ws[:, None] * ws[None, :]
                                for ws in skat_weights.T])
        small_spectra = torch.linalg.eigvalsh(matrices, UPLO="U")
        record_gpu_eigen_route(matrices)
    q_values = (_ordered_sum(u.square()[:, None] * skat_weights.square())
                if _skat_pvalues is None and u.is_cuda else None)
    for i in range(number_weights):
        if _skat_pvalues is None:
            ws = skat_weights[:, i]
            if small_spectra is None:
                weighted_covariance = v * ws[:, None] * ws[None, :]
                eigenvalues = torch.linalg.eigvalsh(weighted_covariance, UPLO="U")
                record_gpu_eigen_route(weighted_covariance)
            else:
                eigenvalues = small_spectra[i]
            q = q_values[i] if q_values is not None else torch.sum(u.square() * ws.square())
            eigenvalues, refined = refine_near_mean_spectrum(
                reference_covariance, eigenvalues, q, weights=ws)
            pvalues[0, i] = _quadratic_form_sf_tensor(q, eigenvalues, reference_reduction=refined)
        else:
            pvalues[0, i] = _skat_pvalues[i]
        wb = burden_weights[:, i]
        burden_score = torch.sum(u * wb)
        burden_variance = wb @ v @ wb
        pvalues[1, i] = _chi1_from_score(burden_score, burden_variance)
        acat_p = common_p
        acat_w = acat_weights[common, i]
        if bool(very_rare.any()):
            rare_wb = wb[very_rare]
            rare_score = torch.sum(u[very_rare] * rare_wb)
            rare_variance = rare_wb @ v[very_rare][:, very_rare] @ rare_wb
            rare_p = _chi1_from_score(rare_score, rare_variance).reshape(1)
            rare_weight = acat_weights[very_rare, i].mean().reshape(1)
            acat_p = torch.cat((acat_p, rare_p))
            acat_w = torch.cat((acat_w, rare_weight))
        pvalues[2, i] = _cct_tensor(acat_p, acat_w, internal=True)
    result: dict[str, float | int] = {"num_variant": count, "cMAC": float(m.sum()) if cmac is None else float(cmac)}
    width = k + 1
    for row, method, combined in ((0, "SKAT", "STAAR-S"), (1, "Burden", "STAAR-B"), (2, "ACAT-V", "STAAR-A")):
        for beta_index, beta_name in enumerate(("1,25", "1,1")):
            values = pvalues[row, beta_index * width:(beta_index + 1) * width]
            base = f"{method}({beta_name})"
            result[base] = float(values[0])
            for label, value in zip(labels, values[1:]):
                result[f"{base}-{label}"] = float(value)
            result[f"{combined}({beta_name})"] = cct(values)
    result["ACAT-O"] = cct(pvalues[:, (0, width)].reshape(-1))
    result["STAAR-O"] = cct(pvalues.reshape(-1))
    return result
