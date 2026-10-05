"""Gaussian STAAR null models with float64 PyTorch block-sparse REML.

SPDX-License-Identifier: GPL-3.0-only
AI initialization, steps, boundary refits and stopping rule follow GMMAT
R/glmmkin.R, https://github.com/hanchenphd/GMMAT.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from .rint import rank_inverse_normal_tensor
from .numerics import reference_crossprod, extended_variance, _extended_sum_pair


def _r_sum(values):
    """Compensated float64 tree for R's extended-precision sum reductions."""
    high, low = _extended_sum_pair(values)
    return high + low


def rank_inverse_normal(values, *, device="cpu"):
    """REGENIE --apply-rint: average tied ranks and (rank - 3/8) / (N + 1/4).

    Call this after selecting the analysis samples; missing values must have
    already been excluded. Returns float64, without additional rescaling.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("RINT requires at least two finite phenotype values")
    return rank_inverse_normal_tensor(y, device=device).cpu().numpy()


@dataclass
class KinshipSpectrum:
    """Diagonal entries plus disjoint small relatedness blocks.

    edge_rows/edge_cols are zero-based indices. Provide each off-diagonal
    edge once, or a symmetric COO with the same value in both directions.
    No cutoff is applied to an already sparse relationship matrix.
    """
    eigenvalues: torch.Tensor
    blocks: list[tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def from_sparse(cls, diagonal, edge_rows=(), edge_cols=(), edge_values=(), *, device="cpu", max_block_size=2048):
        d = np.asarray(diagonal, dtype=np.float64)
        if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
            raise ValueError("kinship diagonal must be a finite nonnegative vector")
        rows = np.asarray(edge_rows, dtype=np.int64)
        cols = np.asarray(edge_cols, dtype=np.int64)
        vals = np.asarray(edge_values, dtype=np.float64)
        if rows.shape != cols.shape or rows.shape != vals.shape or rows.ndim != 1:
            raise ValueError("kinship edge arrays must have equal one-dimensional shapes")
        if ((rows < 0) | (rows >= len(d)) | (cols < 0) | (cols >= len(d))).any() or not np.isfinite(vals).all():
            raise ValueError("kinship edge indices or values are invalid")
        parent = {}
        def find(i):
            parent.setdefault(i, i)
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        edges = {}
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if r == c:
                raise ValueError("diagonal entries must be supplied separately")
            if v == 0:
                continue
            key = (min(r, c), max(r, c))
            if key in edges and edges[key] != v:
                raise ValueError("inconsistent symmetric kinship entries")
            edges[key] = v
            parent[find(r)] = find(c)
        groups = {}
        for i in parent:
            groups.setdefault(find(i), []).append(i)
        eigenvalues = torch.as_tensor(d.copy(), dtype=torch.float64, device=device)
        blocks = []
        grouped_edges = {}
        for (r, c), value in edges.items():
            grouped_edges.setdefault(find(r), []).append((r, c, value))
        for root, group in groups.items():
            group.sort()
            if len(group) > max_block_size:
                raise ValueError("relatedness block exceeds max_block_size; increase the limit only after checking memory")
            loc = {i: j for j, i in enumerate(group)}
            block = np.diag(d[group])
            for r, c, value in grouped_edges[root]:
                block[loc[r], loc[c]] = block[loc[c], loc[r]] = value
            tensor = torch.as_tensor(block, dtype=torch.float64, device=device)
            eigen, rotation = torch.linalg.eigh(tensor)
            if float(eigen.min()) < -1e-10:
                raise ValueError("kinship matrix is not positive semidefinite")
            idx = torch.as_tensor(group, dtype=torch.int64, device=device)
            eigenvalues[idx] = eigen.clamp_min(0)
            blocks.append((idx, rotation))
        return cls(eigenvalues, blocks)

    def rotate(self, values, *, inverse=False):
        value = torch.as_tensor(values, dtype=torch.float64, device=self.eigenvalues.device)
        if value.shape[0] != len(self.eigenvalues):
            raise ValueError("value rows do not match kinship samples")
        result = value.clone()
        for idx, rotation in self.blocks:
            result[idx] = (rotation if inverse else rotation.T) @ value[idx]
        return result


@dataclass
class GaussianNullModel:
    """Single Gaussian phenotype with fitted sparse mixed-model covariance."""
    sample_ids: np.ndarray
    x: torch.Tensor
    scaled_residuals: torch.Tensor
    coefficients: torch.Tensor
    theta: torch.Tensor
    precision_theta: torch.Tensor
    fixed_effect_covariance: torch.Tensor
    spectrum: KinshipSpectrum
    inverse_variance: torch.Tensor
    precision_x: torch.Tensor
    iterations: int
    converged: bool
    family: str = "gaussian"
    n_pheno: int = 1
    relatedness: bool = True
    use_spa: bool = False
    phenotype: torch.Tensor | None = None
    has_kinship: bool = True
    fitted_values: torch.Tensor | None = None
    working_phenotype: torch.Tensor | None = None

    @property
    def n(self):
        return len(self.sample_ids)

    @property
    def device(self):
        return self.x.device

    def score_covariance(self, genotype, *, reduction="blas", max_workspace_bytes=256 * 1024**2):
        """U=G' scaled.residuals, V=G' Sigma_i G-X projection.

        The block eigensystem avoids materializing an N by N projector.
        Final precision and residuals preserve GMMAT's finite-tolerance
        return convention (precision is from the last pre-update step).
        reduction='reference_sparse' uses increasing sparse row sums for
        diagonal precision, with bounded temporary covariance pair blocks.
        Off-diagonal relatedness blocks require reduction='blas'.
        """
        g = torch.as_tensor(genotype, dtype=torch.float64, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not bool(torch.isfinite(g).all()):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if reduction == "reference_sparse":
            if self.spectrum.blocks:
                raise NotImplementedError("ordered sparse reduction requires diagonal precision")
            from .sparse_numerics import reference_sparse_score_covariance
            return reference_sparse_score_covariance(
                g, self.scaled_residuals, self.inverse_variance, self.precision_x,
                self.fixed_effect_covariance, max_workspace_bytes=max_workspace_bytes)
        if reduction != "blas":
            raise ValueError("reduction must be 'blas' or 'reference_sparse'")
        rotated = self.spectrum.rotate(g)
        cross = self.precision_x.T @ g
        covariance = rotated.T @ (self.inverse_variance[:, None] * rotated) - cross.T @ self.fixed_effect_covariance @ cross
        return g.T @ self.scaled_residuals, (covariance + covariance.T) / 2


    def individual_score_variance(self, genotype):
        """Single-variant scores and variances without an M by M matrix.

        Genotypes have the same oriented, imputed [samples, variants] layout
        as score_covariance. All computations remain float64 on this model's
        device; the block eigensystem applies the same precision operator.
        """
        g = torch.as_tensor(genotype, dtype=torch.float64, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not bool(torch.isfinite(g).all()):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        rotated = self.spectrum.rotate(g)
        weighted = self.inverse_variance[:, None] * rotated
        cross = self.precision_x.T @ g
        projected = cross.T @ self.fixed_effect_covariance
        variance = (rotated * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
        return g.T @ self.scaled_residuals, variance


def fit_gaussian_null(phenotype, *, sample_ids=None, covariates=None, kinship_diagonal=None,
                      edge_rows=(), edge_cols=(), edge_values=(), device="cpu", tol=1e-5,
                      maxiter=500, max_block_size=2048, trace_callback=None):
    """Fit intercept/covariate Gaussian null model using GMMAT AI REML.

    covariates must explicitly contain an intercept if one is wanted. None
    selects an intercept. All rows are already aligned; no implicit missing
    removal, phenotype transformation, or kinship threshold takes place.
    With no kinship, reproduces fit_nullmodel(kins=NULL)'s sparse Gaussian
    object and chi-square association calibration.
    """
    y = torch.as_tensor(phenotype, dtype=torch.float64, device=device)
    if y.ndim != 1 or not bool(torch.isfinite(y).all()):
        raise ValueError("phenotype must be a finite vector")
    n = len(y)
    x = torch.ones((n, 1), dtype=y.dtype, device=device) if covariates is None else torch.as_tensor(covariates, dtype=y.dtype, device=device)
    if x.ndim != 2 or x.shape[0] != n or not bool(torch.isfinite(x).all()) or x.shape[1] >= n:
        raise ValueError("covariates must be a finite full-rank samples-by-columns matrix")
    if int(torch.linalg.matrix_rank(x)) != x.shape[1]:
        raise ValueError("covariates are rank deficient")
    ids = np.arange(n).astype(str) if sample_ids is None else np.asarray(sample_ids).astype(str)
    if ids.shape != (n,) or len(set(ids)) != n:
        raise ValueError("sample_ids must be unique and match phenotype rows")
    if tol <= 0 or maxiter < 2:
        raise ValueError("tol must be positive and maxiter at least 2")
    alpha0 = torch.linalg.solve(x.T @ x, x.T @ y)
    if kinship_diagonal is None:
        residual = y - x @ alpha0
        dispersion = residual.square().sum() / (n - x.shape[1])
        if float(dispersion) <= 0:
            raise ValueError("phenotype has zero residual variance")
        kin = KinshipSpectrum.from_sparse(np.zeros(n), device=device)
        inv = torch.ones_like(y) / dispersion
        cov = torch.cholesky_inverse(torch.linalg.cholesky(x.T @ (inv[:, None] * x)))
        return GaussianNullModel(ids, x, residual / dispersion, alpha0, torch.stack((dispersion, dispersion * 0)),
                                 torch.stack((dispersion, dispersion * 0)), cov, kin, inv, inv[:, None] * x, 0, True, phenotype=y, has_kinship=False, fitted_values=x @ alpha0, working_phenotype=y)
    if len(kinship_diagonal) != n:
        raise ValueError("kinship diagonal does not match phenotype")
    kin = KinshipSpectrum.from_sparse(kinship_diagonal, edge_rows, edge_cols, edge_values, device=device, max_block_size=max_block_size)
    eigen = kin.eigenvalues
    yr = kin.rotate(y)
    xr = kin.rotate(x)
    mean_diag = _r_sum(eigen) / n
    if mean_diag <= 0:
        raise ValueError("kinship has zero mean diagonal; omit it for an ordinary model")

    def state(tau):
        variance = tau[0] + tau[1] * eigen
        if bool((variance <= 0).any()):
            raise ValueError("mixed-model covariance is singular")
        inv = torch.reciprocal(torch.sqrt(variance)).square()
        sx = inv[:, None] * xr
        cov = torch.cholesky_inverse(torch.linalg.cholesky(reference_crossprod(xr, sx)))
        alpha = cov @ reference_crossprod(sx, yr)
        sx_cov = sx @ cov
        py = inv * yr - sx @ reference_crossprod(sx_cov, yr)
        return inv, sx, cov, alpha, py

    total_iterations = 0
    fixed = torch.zeros(2, dtype=torch.bool, device=device)
    converged = False
    for _refit in range(4):
        free = torch.nonzero(~fixed, as_tuple=True)[0]
        tau = torch.zeros(2, dtype=y.dtype, device=device)
        # R cov.c keeps centered products and the final division in extended
        # precision; rounding the squares before summation shifts the AI seed.
        variance_y = extended_variance(y)
        tau[free] = variance_y / 2
        tau[1] /= mean_diag
        inv, sx, cov, alpha, py = state(tau)
        # GMMAT's single EM initialization, then average-information steps.
        diag_p = inv - torch.sum(sx * (sx @ cov), dim=1)
        derivative = torch.stack((torch.ones_like(eigen), eigen))
        sx_cov = sx @ cov
        apy_k = eigen * py
        papy_k = inv * apy_k - sx @ reference_crossprod(sx_cov, apy_k)
        scores = torch.stack((_r_sum(py.square()) - _r_sum(diag_p),
            _r_sum(yr * papy_k) - (_r_sum(inv * eigen) - _r_sum(sx * (eigen[:, None] * sx_cov)))))
        tau[free] = torch.maximum(tau[free] + tau[free].square() * scores[free] / n, torch.zeros_like(tau[free]))
        alpha_prev = alpha0.clone()
        for iteration in range(1, maxiter + 1):
            total_iterations += 1
            old = tau.clone()
            old_y = yr.clone()
            inv, sx, cov, alpha, py = state(old)
            if len(free):
                sx_cov = sx @ cov
                apy = derivative.T * py[:, None]
                papy = inv[:, None] * apy - sx @ reference_crossprod(sx_cov, apy)
                diag_p = inv - torch.sum(sx * sx_cov, dim=1)
                all_scores = torch.stack((_r_sum(py.square()-diag_p),
                    _r_sum(yr*papy[:, 1]) - (_r_sum(inv*eigen)-_r_sum(sx*(eigen[:,None]*sx_cov)))))
                ai00 = _r_sum(py*(inv*py)) - _r_sum(reference_crossprod(sx_cov, py)*reference_crossprod(sx, py))
                ai01 = _r_sum(py*papy[:, 1])
                ai11 = _r_sum(py*(eigen*papy[:, 1]))
                all_ai = torch.stack((torch.stack((ai00, ai01)), torch.stack((ai01, ai11))))
                # R passes a column-major AI matrix to DGESV. cuSOLVER uses
                # a different transpose route for row-major input; preserving
                # column-major strides also preserves its scalar rounding.
                selected_ai = all_ai[free][:, free].T.contiguous().T
                delta = torch.linalg.solve(selected_ai, all_scores[free])
                for _step in range(2048):
                    tau[free] = old[free] + delta
                    tau[(tau < tol) & (old < tol)] = 0
                    if not bool((tau < 0).any()):
                        break
                    delta /= 2
                else:
                    raise ArithmeticError("AI step could not remain in the parameter space")
                tau[tau < tol] = 0
            eta_spectral = old_y - old[0] * (inv * old_y - sx @ alpha)
            eta = kin.rotate(eta_spectral, inverse=True)
            working_y = eta + (y - eta)
            yr = kin.rotate(working_y)
            if trace_callback is not None:
                trace_callback({"refit": _refit, "iteration": iteration, "tau_old": old,
                    "tau": tau.clone(), "Y_old": old_y, "Y": yr, "PY": py,
                    "inverse_variance": inv, "cov": cov, "alpha": alpha,
                    "score": all_scores if len(free) else None,
                    "AI": all_ai if len(free) else None, "eta": eta})
            change = max(float((abs(alpha - alpha_prev) / (abs(alpha) + abs(alpha_prev) + tol)).max()),
                         float((abs(tau - old) / (abs(tau) + abs(old) + tol)).max())) * 2
            alpha_prev = alpha
            if change < tol:
                converged = iteration < maxiter
                break
            if float(abs(tau).max()) > tol ** -2:
                break
        new_fixed = tau < 1.01 * tol
        if bool((new_fixed == fixed).all()):
            break
        fixed = new_fixed
    else:
        raise ArithmeticError("variance boundary refits did not stabilize")
    if not converged:
        raise ArithmeticError("AI REML did not converge; no implicit alternative optimizer is used")
    if float(tau[0]) <= 0:
        raise ValueError("zero residual dispersion cannot produce GMMAT scaled residuals")
    # eta=y-old_dispersion*P_old*y; returned residual divides by final tau.
    scaled = (y - eta) / tau[0]
    precision_x = kin.rotate(sx, inverse=True)
    return GaussianNullModel(ids, x, scaled, alpha, tau, old, cov, kin, inv, precision_x, total_iterations, converged, phenotype=y, fitted_values=eta, working_phenotype=working_y)
