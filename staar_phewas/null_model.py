"""Gaussian STAAR block-sparse REML: native FP32 or explicit FP64 controls.

SPDX-License-Identifier: GPL-3.0-only
AI initialization, steps, boundary refits and stopping rule follow GMMAT
R/glmmkin.R, https://github.com/hanchenphd/GMMAT.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from .tf32 import matmul, validate_mode
from .rint import rank_inverse_normal_tensor
from .numerics import reference_crossprod, extended_variance, _extended_sum_pair
from .tensor_validation import tensor_all_finite


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
    def from_sparse(cls, diagonal, edge_rows=(), edge_cols=(), edge_values=(), *, device="cpu", max_block_size=2048, dtype=torch.float64):
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
        eigenvalues = torch.as_tensor(d.copy(), dtype=dtype, device=device)
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
            tensor = torch.as_tensor(block, dtype=dtype, device=device)
            eigen, rotation = torch.linalg.eigh(tensor)
            if float(eigen.min()) < -1e-10:
                raise ValueError("kinship matrix is not positive semidefinite")
            idx = torch.as_tensor(group, dtype=torch.int64, device=device)
            eigenvalues[idx] = eigen.clamp_min(0)
            blocks.append((idx, rotation))
        return cls(eigenvalues, blocks)

    def rotate(self, values, *, inverse=False, matmul_mode="fp64"):
        dtype = torch.float32 if validate_mode(matmul_mode) == "tf32" else torch.float64
        value = torch.as_tensor(values, dtype=dtype, device=self.eigenvalues.device)
        if value.shape[0] != len(self.eigenvalues):
            raise ValueError("value rows do not match kinship samples")
        # Rotation callers consume this tensor without mutating it.
        if not self.blocks:
            return value
        result = value.clone()
        for idx, rotation in self.blocks:
            result[idx] = matmul((rotation if inverse else rotation.T).to(dtype=dtype), value[idx], mode=matmul_mode)
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
    matmul_mode: str = "fp64"

    @property
    def n(self):
        return len(self.sample_ids)

    @property
    def device(self):
        return self.x.device

    def set_matmul_mode(self, mode):
        """Convert the complete floating fitted state without refitting it.

        Legacy FP64 caches become FP32 for native execution; no precision
        control or arithmetic fallback is inferred from a cache's metadata.
        """
        mode = validate_mode(mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        for name in ("x", "scaled_residuals", "coefficients", "theta", "precision_theta",
                     "fixed_effect_covariance", "inverse_variance", "precision_x",
                     "phenotype", "fitted_values", "working_phenotype"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(dtype=dtype))
        self.spectrum.eigenvalues = self.spectrum.eigenvalues.to(dtype=dtype)
        self.spectrum.blocks = [(rows, rotation.to(dtype=dtype)) for rows, rotation in self.spectrum.blocks]
        self.matmul_mode = mode
        self._centered_single_projection_cache = None
        return self

    def score_covariance_sample_block(self, genotype, *, sample_block_size=8192, matmul_mode=None):
        """TF32 Gaussian score/covariance by streaming sample blocks.

        The host keeps the N-by-M imputed genotype matrix. A K-by-M sample
        block is copied to CUDA, and its score and weighted cross-product are
        accumulated into the complete M-by-M FP32 covariance. Projection
        cross-products are accumulated with the same TF32 products before the
        final dense projection. This route intentionally does not mirror
        upper-triangular blocks and is restricted to a single Gaussian null
        without SPA or kinship blocks.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32":
            raise ValueError("sample-block covariance requires native TF32")
        if self.spectrum.blocks or self.n_pheno != 1 or self.use_spa or self.family != "gaussian":
            raise NotImplementedError("sample-block covariance requires a single Gaussian model without SPA or kinship blocks")
        if type(sample_block_size) is not int or sample_block_size < 1:
            raise ValueError("sample_block_size must be a positive integer")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        device = self.device
        inv = self.inverse_variance.to(dtype=torch.float32, device=device)
        residual = self.scaled_residuals.to(dtype=torch.float32, device=device)
        precision_x = self.precision_x.to(dtype=torch.float32, device=device)
        fixed_cov = self.fixed_effect_covariance.to(dtype=torch.float32, device=device)
        score = torch.zeros((m,), dtype=torch.float32, device=device)
        covariance = torch.zeros((m, m), dtype=torch.float32, device=device)
        cross = torch.zeros((precision_x.shape[1], m), dtype=torch.float32, device=device)
        for start in range(0, n, sample_block_size):
            stop = min(start + sample_block_size, n)
            block = torch.as_tensor(source[start:stop], dtype=torch.float32, device=device)
            score.add_(matmul(block.T, residual[start:stop], mode="tf32"))
            covariance.add_(matmul(block.T, inv[start:stop, None] * block, mode="tf32"))
            cross.add_(matmul(precision_x[start:stop].T, block, mode="tf32"))
            del block
        projected = matmul(matmul(cross.T, fixed_cov, mode="tf32"), cross, mode="tf32")
        covariance.sub_(projected)
        covariance = (covariance + covariance.T) / 2
        del projected, cross, source, inv, residual, precision_x, fixed_cov
        return score, covariance


    def score_covariance_tiled(self, genotype, *, variant_tile_size=256, matmul_mode=None):
        """TF32 Gaussian covariance using bounded variant tiles.

        genotype remains on host memory. Each N-by-B tile is copied to the
        CUDA device, and only B-by-B products are resident while the complete
        M-by-M covariance is assembled in host FP32 storage. This preserves
        variant order, native TF32 products, and dense covariance/eigenspectrum
        semantics without changing rare-variant selection. This route is
        restricted to a single Gaussian model with diagonal precision.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32":
            raise ValueError("tiled covariance requires native TF32")
        if self.spectrum.blocks or self.n_pheno != 1 or self.use_spa or self.family != "gaussian":
            raise NotImplementedError("tiled covariance requires a single Gaussian model without SPA or kinship blocks")
        if type(variant_tile_size) is not int or variant_tile_size < 1:
            raise ValueError("variant_tile_size must be a positive integer")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        if m < 1:
            return (torch.empty((0,), dtype=torch.float32, device=self.device),
                    torch.empty((0, 0), dtype=torch.float32, device=self.device))
        device = self.device
        inv = self.inverse_variance.to(dtype=torch.float32, device=device)
        residual = self.scaled_residuals.to(dtype=torch.float32, device=device)
        precision_x = self.precision_x.to(dtype=torch.float32, device=device)
        fixed_cov = self.fixed_effect_covariance.to(dtype=torch.float32, device=device)
        score = torch.empty((m,), dtype=torch.float32, device=device)
        covariance_host = torch.empty((m, m), dtype=torch.float32, device="cpu")
        for start in range(0, m, variant_tile_size):
            stop = min(start + variant_tile_size, m)
            left = torch.as_tensor(source[:, start:stop], dtype=torch.float32, device=device)
            score[start:stop] = matmul(left.T, residual, mode="tf32")
            left_cross = matmul(precision_x.T, left, mode="tf32")
            for other in range(start, m, variant_tile_size):
                other_stop = min(other + variant_tile_size, m)
                right = torch.as_tensor(source[:, other:other_stop], dtype=torch.float32, device=device)
                right_cross = matmul(precision_x.T, right, mode="tf32")
                block = (matmul(left.T, inv[:, None] * right, mode="tf32")
                         - matmul(matmul(left_cross.T, fixed_cov, mode="tf32"),
                                  right_cross, mode="tf32"))
                if start == other:
                    block = (block + block.T) / 2
                else:
                    reverse = (matmul(right.T, inv[:, None] * left, mode="tf32")
                               - matmul(matmul(right_cross.T, fixed_cov, mode="tf32"), left_cross, mode="tf32"))
                    block = (block + reverse.T) / 2
                    del reverse
                covariance_host[start:stop, other:other_stop].copy_(block.detach().cpu())
                if other != start:
                    covariance_host[other:other_stop, start:stop].copy_(block.T.detach().cpu())
                del right, right_cross, block
            del left, left_cross
        covariance = covariance_host.to(device=device, dtype=torch.float32)
        del covariance_host, source, inv, residual, precision_x, fixed_cov
        return score, covariance


    def score_covariance_cached(self, genotype, *, variant_tile_size=4096,
                                panel_variant_size=None, memory_limit_gib=40,
                                matmul_mode=None, profile=False):
        """Return score, covariance and a bounded TF32 cache report.

        The host input is an already oriented/imputed FP32 [samples, variants]
        dosage matrix aligned to this model. Output score [variants] and dense
        covariance [variants, variants] remain CUDA FP32. Cache selection is
        automatic: retain original/weighted full genotypes if admitted, or
        stream two panels. Covariance tiles default to 4096 columns; score and
        covariate projection tiles remain 512. ``panel_variant_size`` is None
        or a positive multiple of ``variant_tile_size``. ``memory_limit_gib``
        bounds live allocated storage, including the resident model, and must
        be at most 40. ``profile`` adds CUDA stream timings with one final
        synchronization; stage intervals overlap and are not pure kernel time.

        Callers choose this backend explicitly. Ordinary score_covariance,
        legacy tiled and Single interfaces retain their existing behavior.
        No CPU, lower-precision or alternate statistical fallback is used.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        from ._cached_covariance import score_covariance_cached
        return score_covariance_cached(
            self, genotype, variant_tile_size=variant_tile_size,
            panel_variant_size=panel_variant_size,
            memory_limit_gib=memory_limit_gib, matmul_mode=mode,
            symmetry="average", profile=profile)

    def score_covariance(self, genotype, *, reduction="blas", max_workspace_bytes=256 * 1024**2, matmul_mode=None):
        """U=G' scaled.residuals, V=G' Sigma_i G-X projection.

        The block eigensystem avoids materializing an N by N projector.
        Final precision and residuals preserve GMMAT's finite-tolerance
        return convention (precision is from the last pre-update step).
        reduction='reference_sparse' uses increasing sparse row sums for
        diagonal precision, with bounded temporary covariance pair blocks.
        Off-diagonal relatedness blocks require reduction='blas'.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if mode != "fp64" and reduction == "reference_sparse":
            raise ValueError("reference_sparse is an FP64 control; forced TF32 requires reduction=blas")
        if reduction == "reference_sparse":
            if self.spectrum.blocks:
                raise NotImplementedError("ordered sparse reduction requires diagonal precision")
            from .sparse_numerics import reference_sparse_score_covariance
            return reference_sparse_score_covariance(
                g, self.scaled_residuals, self.inverse_variance, self.precision_x,
                self.fixed_effect_covariance, max_workspace_bytes=max_workspace_bytes)
        if reduction != "blas":
            raise ValueError("reduction must be 'blas' or 'reference_sparse'")
        rotated = self.spectrum.rotate(g, matmul_mode=mode)
        cross = matmul(self.precision_x.to(dtype=dtype).T, g, mode=mode)
        covariance = matmul(rotated.T, self.inverse_variance.to(dtype=dtype)[:, None] * rotated, mode=mode) - matmul(matmul(cross.T, self.fixed_effect_covariance.to(dtype=dtype), mode=mode), cross, mode=mode)
        return matmul(g.T, self.scaled_residuals, mode=mode), (covariance + covariance.T) / 2


    def _centered_single_projection_state(self):
        """FP32 constant-direction correction for this fitted diagonal model."""
        from .single_projection import centered_projection_state
        return centered_projection_state(
            self, self.inverse_variance, self.precision_x, product=matmul,
            blocked=bool(self.spectrum.blocks))

    def individual_score_variance(self, genotype, *, matmul_mode=None):
        """Single-variant scores and variances without an M by M matrix.

        Genotypes have the same oriented, imputed [samples, variants] layout
        as score_covariance. Storage and elementwise reductions follow the selected FP32/FP64 mode;
        matmul_mode selects explicit TF32 products or the FP64 control. For
        diagonal FP32 precision and an explicit intercept, compute the same P
        quadratic form on centered genotypes and retain its finite-state P1
        correction. Scores always use the original oriented genotypes.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        centered_state = (self._centered_single_projection_state()
                          if mode == "tf32" and not g.requires_grad else None)
        if centered_state is not None:
            score = matmul(g.T, self.scaled_residuals, mode=mode)
            mean = g.mean(dim=0)
            centered = g - mean[None, :]
            weighted = self.inverse_variance[:, None] * centered
            cross = matmul(self.precision_x.T, centered, mode=mode)
            projected = matmul(cross.T, self.fixed_effect_covariance, mode=mode)
            variance = ((centered * weighted).sum(dim=0)
                        - (projected * cross.T).sum(dim=1))
            del weighted, cross, projected
            p_one, p_one_sum = centered_state
            variance += (2 * mean * matmul(centered.T, p_one, mode=mode)
                         + mean.square() * p_one_sum)
            return score, variance
        rotated = self.spectrum.rotate(g, matmul_mode=mode)
        weighted = self.inverse_variance.to(dtype=dtype)[:, None] * rotated
        cross = matmul(self.precision_x.to(dtype=dtype).T, g, mode=mode)
        projected = matmul(cross.T, self.fixed_effect_covariance.to(dtype=dtype), mode=mode)
        variance = (rotated * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
        return matmul(g.T, self.scaled_residuals, mode=mode), variance


def fit_gaussian_null(phenotype, *, sample_ids=None, covariates=None, kinship_diagonal=None,
                      edge_rows=(), edge_cols=(), edge_values=(), device="cpu", tol=1e-5,
                      maxiter=500, max_block_size=2048, trace_callback=None, matmul_mode="fp64"):
    """Fit intercept/covariate Gaussian null model using GMMAT AI REML.

    covariates must explicitly contain an intercept if one is wanted. None
    selects an intercept. All rows are already aligned; no implicit missing
    removal, phenotype transformation, or kinship threshold takes place.
    With no kinship, reproduces fit_nullmodel(kins=NULL)'s sparse Gaussian
    object and chi-square association calibration.
    """
    validate_mode(matmul_mode)
    def mm(a, b):
        return matmul(a, b, mode=matmul_mode)
    def cp(a, b):
        return reference_crossprod(a, b) if matmul_mode == "fp64" else mm(a.T, b)
    dtype = torch.float32 if matmul_mode == "tf32" else torch.float64
    rsum = _r_sum if matmul_mode == "fp64" else lambda values: values.sum()
    y = torch.as_tensor(phenotype, dtype=dtype, device=device)
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
    alpha0 = torch.linalg.solve(mm(x.T, x), mm(x.T, y))
    if kinship_diagonal is None:
        residual = y - mm(x, alpha0)
        dispersion = residual.square().sum() / (n - x.shape[1])
        if float(dispersion) <= 0:
            raise ValueError("phenotype has zero residual variance")
        kin = KinshipSpectrum.from_sparse(np.zeros(n), device=device, dtype=dtype)
        inv = torch.ones_like(y) / dispersion
        cov = torch.cholesky_inverse(torch.linalg.cholesky(mm(x.T, inv[:, None] * x)))
        return GaussianNullModel(ids, x, residual / dispersion, alpha0, torch.stack((dispersion, dispersion * 0)),
                                 torch.stack((dispersion, dispersion * 0)), cov, kin, inv, inv[:, None] * x, 0, True, phenotype=y, has_kinship=False, fitted_values=mm(x, alpha0), working_phenotype=y, matmul_mode=matmul_mode)
    if len(kinship_diagonal) != n:
        raise ValueError("kinship diagonal does not match phenotype")
    kin = KinshipSpectrum.from_sparse(kinship_diagonal, edge_rows, edge_cols, edge_values, device=device, max_block_size=max_block_size, dtype=dtype)
    eigen = kin.eigenvalues
    yr = kin.rotate(y, matmul_mode=matmul_mode)
    xr = kin.rotate(x, matmul_mode=matmul_mode)
    mean_diag = rsum(eigen) / n
    if mean_diag <= 0:
        raise ValueError("kinship has zero mean diagonal; omit it for an ordinary model")

    def state(tau):
        variance = tau[0] + tau[1] * eigen
        if bool((variance <= 0).any()):
            raise ValueError("mixed-model covariance is singular")
        inv = torch.reciprocal(torch.sqrt(variance)).square()
        sx = inv[:, None] * xr
        cov = torch.cholesky_inverse(torch.linalg.cholesky(cp(xr, sx)))
        alpha = mm(cov, cp(sx, yr))
        sx_cov = mm(sx, cov)
        py = inv * yr - mm(sx, cp(sx_cov, yr))
        return inv, sx, cov, alpha, py

    total_iterations = 0
    fixed = torch.zeros(2, dtype=torch.bool, device=device)
    converged = False
    for _refit in range(4):
        free = torch.nonzero(~fixed, as_tuple=True)[0]
        tau = torch.zeros(2, dtype=y.dtype, device=device)
        # R cov.c keeps centered products and the final division in extended
        # precision; rounding the squares before summation shifts the AI seed.
        variance_y = extended_variance(y) if matmul_mode == "fp64" else y.var(correction=1)
        tau[free] = variance_y / 2
        tau[1] /= mean_diag
        inv, sx, cov, alpha, py = state(tau)
        # GMMAT's single EM initialization, then average-information steps.
        diag_p = inv - torch.sum(sx * (mm(sx, cov)), dim=1)
        derivative = torch.stack((torch.ones_like(eigen), eigen))
        sx_cov = mm(sx, cov)
        apy_k = eigen * py
        papy_k = inv * apy_k - mm(sx, cp(sx_cov, apy_k))
        scores = torch.stack((rsum(py.square()) - rsum(diag_p),
            rsum(yr * papy_k) - (rsum(inv * eigen) - rsum(sx * (eigen[:, None] * sx_cov)))))
        tau[free] = torch.maximum(tau[free] + tau[free].square() * scores[free] / n, torch.zeros_like(tau[free]))
        alpha_prev = alpha0.clone()
        for iteration in range(1, maxiter + 1):
            total_iterations += 1
            old = tau.clone()
            old_y = yr.clone()
            inv, sx, cov, alpha, py = state(old)
            if len(free):
                sx_cov = mm(sx, cov)
                apy = derivative.T * py[:, None]
                papy = inv[:, None] * apy - mm(sx, cp(sx_cov, apy))
                diag_p = inv - torch.sum(sx * sx_cov, dim=1)
                all_scores = torch.stack((rsum(py.square()-diag_p),
                    rsum(yr*papy[:, 1]) - (rsum(inv*eigen)-rsum(sx*(eigen[:,None]*sx_cov)))))
                ai00 = rsum(py*(inv*py)) - rsum(cp(sx_cov, py)*cp(sx, py))
                ai01 = rsum(py*papy[:, 1])
                ai11 = rsum(py*(eigen*papy[:, 1]))
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
            eta_spectral = old_y - old[0] * (inv * old_y - mm(sx, alpha))
            eta = kin.rotate(eta_spectral, inverse=True, matmul_mode=matmul_mode)
            working_y = eta + (y - eta)
            yr = kin.rotate(working_y, matmul_mode=matmul_mode)
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
    precision_x = kin.rotate(sx, inverse=True, matmul_mode=matmul_mode)
    return GaussianNullModel(ids, x, scaled, alpha, tau, old, cov, kin, inv, precision_x, total_iterations, converged, phenotype=y, fitted_values=eta, working_phenotype=working_y, matmul_mode=matmul_mode)
