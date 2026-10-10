"""Ordinary logistic IRLS and explicit prefitted binary association state.

SPDX-License-Identifier: GPL-3.0-only
IRLS follows the binomial identity used by R's stats::glm; this module
does not fit a binary mixed model or replace a supplied GRM implicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .statistics import score_covariance
from .tf32 import matmul, validate_mode
from .tensor_validation import tensor_all_finite


@dataclass
class BinaryNullModel:
    sample_ids: np.ndarray
    x: torch.Tensor
    scaled_residuals: torch.Tensor
    fitted_probability: torch.Tensor
    xw: torch.Tensor
    projection_left: torch.Tensor
    fixed_effect_covariance: torch.Tensor
    precision: torch.Tensor | None = None
    precision_x: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    phenotype: torch.Tensor | None = None
    iterations: int = 0
    converged: bool = True
    has_kinship: bool = False
    family: str = "binomial"
    n_pheno: int = 1
    use_spa: bool = True
    fit_method: str = "prefitted_binary_state"
    matmul_mode: str = "fp64"
    working_phenotype: torch.Tensor | None = None

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def device(self):
        return self.x.device

    @property
    def dof(self):
        return self.n - self.x.shape[1]

    def set_matmul_mode(self, mode):
        """Convert fitted tensors without replacing or refitting binary state.

        Native TF32 applies only to non-SPA association; the established SPA
        interface keeps its FP64 arithmetic. Source mode is retained separately
        from the selected association storage mode.
        """
        mode = validate_mode(mode)
        if mode == "tf32" and self.use_spa:
            raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
        if not hasattr(self, "source_matmul_mode"):
            self.source_matmul_mode = self.matmul_mode
        dtype = torch.float32 if mode == "tf32" else torch.float64
        for name in ("x", "scaled_residuals", "fitted_probability", "xw",
                     "projection_left", "fixed_effect_covariance", "precision",
                     "precision_x", "coefficients", "phenotype", "working_phenotype"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(dtype=dtype))
        self.matmul_mode = mode
        self._centered_single_projection_cache = None
        return self

    def _association_genotype(self, genotype, matmul_mode):
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode == "tf32" and self.use_spa:
            raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        return g, mode

    def _fitted_projection(self, genotype, mode):
        """Apply supplied Sigma_i, retaining original Sigma_iX and cov.

        Diagonal precision uses a rowwise product. Explicit sparse precision
        uses a sparse FP32/FP64 product without materializing an N by N matrix.
        Dense products and fixed-effect projection use the selected mode.
        """
        g = genotype
        precision = self.precision
        if precision is None:
            precision = self.fitted_probability * (1 - self.fitted_probability)
        precision = precision.to(dtype=g.dtype)
        if precision.ndim == 1:
            weighted = precision[:, None] * g
        elif precision.layout != torch.strided:
            weighted = torch.sparse.mm(precision, g)
        else:
            weighted = matmul(precision, g, mode=mode)
        sx = self.precision_x if self.precision_x is not None else self.xw.T
        cross = matmul(sx.to(dtype=g.dtype).T, g, mode=mode)
        return weighted, cross

    def score_covariance(self, genotype, *, matmul_mode=None):
        """Return u=G.T r and V=G.T Sigma_i G-cross.T cov cross.

        Supplied precision_covariates and fixed_effect_covariance are used
        directly; no projection is recomputed from the design matrix.
        """
        g, mode = self._association_genotype(genotype, matmul_mode)
        # Keep the historical ordinary-logistic/SPA path unchanged in FP64.
        if mode == "fp64" and self.precision is None:
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.fitted_probability * (1-self.fitted_probability))
        if mode == "fp64" and self.precision.ndim == 1 and (self.precision_x is None or self.use_spa):
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.precision)
        weighted, cross = self._fitted_projection(g, mode)
        cov = self.fixed_effect_covariance.to(dtype=g.dtype)
        covariance = matmul(g.T, weighted, mode=mode) - matmul(
            matmul(cross.T, cov, mode=mode), cross, mode=mode)
        return (matmul(g.T, self.scaled_residuals.to(dtype=g.dtype), mode=mode),
                (covariance + covariance.T) / 2)

    def _diagonal_tf32_covariance_state(self, matmul_mode=None):
        """Validate an explicit fitted diagonal state for bounded covariance.

        Return the original precision, Sigma_iX, residual and fitted covariance.
        This never derives or fits a replacement projection and never treats a
        dense or sparse precision as diagonal. Cached and tiled association are
        inference interfaces; tensors requiring gradients are rejected so panel
        lifetimes remain bounded.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32" or self.matmul_mode != "tf32":
            raise ValueError("bounded binary covariance requires a native TF32 model")
        if self.family != "binomial" or self.n_pheno != 1 or self.use_spa:
            raise NotImplementedError("bounded binary covariance requires one non-SPA binomial model")
        precision, sx = self.precision, self.precision_x
        if precision is None or sx is None:
            raise NotImplementedError("bounded binary covariance requires explicit fitted precision and precision_covariates")
        n, q = self.x.shape
        values = (("x", self.x, (n, q)), ("precision", precision, (n,)),
                  ("precision_x", sx, (n, q)),
                  ("scaled_residuals", self.scaled_residuals, (n,)),
                  ("fixed_effect_covariance", self.fixed_effect_covariance, (q, q)))
        for name, value, shape in values:
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != shape or value.dtype != torch.float32
                    or value.device != self.device or value.requires_grad):
                raise ValueError(f"model.{name} must be aligned dense FP32 without gradients on the model device")
            if not tensor_all_finite(value):
                raise ValueError(f"model.{name} must be finite")
        if n < 1 or bool((precision <= 0).any()):
            raise ValueError("binary diagonal precision must be positive")
        return precision, sx, self.scaled_residuals, self.fixed_effect_covariance

    def score_covariance_cached(self, genotype, *, variant_tile_size=4096,
                                panel_variant_size=None, memory_limit_gib=40,
                                matmul_mode=None, profile=False):
        """Return fitted binary score, covariance and a bounded cache report.

        Host FP32 dosage is already oriented/imputed and ordered [N, M]. The
        shared native TF32 cache retains 512-column score/projection geometry,
        bounds original/weighted panels against live CUDA memory, and averages
        both directed covariance products. All supplied fitted state is reused.
        Output score [M] and covariance [M, M] stay CUDA FP32. Only explicit
        diagonal non-SPA state is supported; no fitting or statistical fallback
        occurs. Parameter meanings and admission rules match the Gaussian cache.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        from ._cached_covariance import score_covariance_cached
        return score_covariance_cached(
            self, genotype, variant_tile_size=variant_tile_size,
            panel_variant_size=panel_variant_size, memory_limit_gib=memory_limit_gib,
            matmul_mode=mode, symmetry="average", profile=profile)

    def score_covariance_tiled(self, genotype, *, variant_tile_size=256, matmul_mode=None):
        """Bounded host-to-device variant tiles using the fitted binary formula.

        This uses the existing Gaussian tiled product order with binary fitted
        diagonal precision: U=G.T r and V=G.T Sigma_i G-cross.T cov cross.
        Both directions are averaged before host FP32 assembly. The full [M,M]
        covariance is returned on the model device; callers retain their dense
        output/solver workspace guards. This does not alter SPA or FP64 paths.
        """
        inv, precision_x, residual, fixed_cov = self._diagonal_tf32_covariance_state(matmul_mode)
        if type(variant_tile_size) is not int or variant_tile_size < 1:
            raise ValueError("variant_tile_size must be a positive integer")
        if isinstance(genotype, torch.Tensor) and (genotype.is_cuda or genotype.requires_grad):
            raise ValueError("tiled binary genotype must be host data without gradients")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a samples-by-variants matrix aligned to the model")
        if not tensor_all_finite(source):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        if not m:
            return (torch.empty((0,), dtype=torch.float32, device=self.device),
                    torch.empty((0, 0), dtype=torch.float32, device=self.device))
        device = self.device
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
                         - matmul(matmul(left_cross.T, fixed_cov, mode="tf32"), right_cross, mode="tf32"))
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
        return score, covariance

    def _centered_single_projection_state(self):
        """Constant-direction correction for a fixed non-SPA diagonal model."""
        from .single_projection import centered_projection_state
        return centered_projection_state(
            self, self.precision, self.precision_x, product=matmul,
            blocked=self.use_spa)

    def individual_score_variance(self, genotype, *, matmul_mode=None):
        """Use the same fitted score and only diag(V), without M by M output."""
        g, mode = self._association_genotype(genotype, matmul_mode)
        if mode == "fp64":
            # Keep the established control/SPA score_covariance branch and its
            # projection arithmetic. Diagonal-only optimization is native TF32.
            score, covariance = self.score_covariance(g, matmul_mode=mode)
            return score, covariance.diagonal()
        centered_state = (self._centered_single_projection_state()
                          if not g.requires_grad else None)
        if centered_state is not None:
            score = matmul(g.T, self.scaled_residuals, mode=mode)
            mean = g.mean(dim=0)
            centered = g - mean[None, :]
            weighted = self.precision[:, None] * centered
            cross = matmul(self.precision_x.T, centered, mode=mode)
            projected = matmul(cross.T, self.fixed_effect_covariance, mode=mode)
            variance = ((centered * weighted).sum(dim=0)
                        - (projected * cross.T).sum(dim=1))
            del weighted, cross, projected
            p_one, p_one_sum = centered_state
            variance += (2 * mean * matmul(centered.T, p_one, mode=mode)
                         + mean.square() * p_one_sum)
            return score, variance
        weighted, cross = self._fitted_projection(g, mode)
        projected = matmul(cross.T, self.fixed_effect_covariance.to(dtype=g.dtype), mode=mode)
        variance = (g * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
        return matmul(g.T, self.scaled_residuals.to(dtype=g.dtype), mode=mode), variance


def binary_prefitted_state(*, sample_ids, covariates, residual, fitted_probability,
                           xw, projection_left, fixed_effect_covariance,
                           precision=None, precision_covariates=None,
                           coefficients=None, phenotype=None, has_kinship=False,
                           use_spa=True, device="cuda", provenance="prefitted_binary_state",
                           matmul_mode="fp64", working_phenotype=None):
    """Use explicitly supplied state; this function performs no null fitting.

    Ordinary xw=X.T*mu*(1-mu), left=X*inv(X.T*W*X). Mixed xw=X.T*Sigma_i,
    left=X*cov. A mixed state additionally needs its fitted precision and
    precision_covariates for score-based association and SPA filtering.
    """
    mode = validate_mode(matmul_mode)
    if mode == "tf32" and use_spa:
        raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
    dtype = torch.float32 if mode == "tf32" else torch.float64
    def tensor(value):
        return None if value is None else torch.as_tensor(value, dtype=dtype, device=device)
    x,r,mu,wx,left,cov = map(tensor,(covariates,residual,fitted_probability,xw,projection_left,fixed_effect_covariance))
    if x.ndim!=2 or x.shape[1]<1:
        raise ValueError("binary covariates must include a fixed-effect design")
    n,p=x.shape;ids=np.asarray(sample_ids,dtype=str)
    if ids.shape!=(n,) or len(set(ids))!=n or r.shape!=(n,) or mu.shape!=(n,) or wx.shape!=(p,n) or left.shape!=(n,p) or cov.shape!=(p,p):
        raise ValueError("prefitted binary arrays and sample IDs have incompatible shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (x,r,mu,wx,left,cov)) or bool(((mu<=0)|(mu>=1)).any()):
        raise ValueError("binary state must be finite with fitted probabilities strictly in (0,1)")
    inverse,sx = tensor(precision),tensor(precision_covariates)
    if has_kinship and (inverse is None or sx is None):
        raise ValueError("mixed prefitted binary state requires fitted precision and precision_covariates")
    if inverse is not None and inverse.shape not in ((n,),(n,n)):
        raise ValueError("binary precision must be aligned diagonal or samples-by-samples")
    if sx is not None and sx.shape!=(n,p):
        raise ValueError("binary precision_covariates must have shape samples-by-covariates")
    for value in (inverse, sx):
        if value is not None:
            finite_value = value if value.layout == torch.strided else value.to_sparse_coo().coalesce().values()
            if not bool(torch.isfinite(finite_value).all()):
                raise ValueError("binary fitted precision state must be finite")
    if inverse is not None and inverse.ndim == 1 and bool((inverse <= 0).any()):
        raise ValueError("binary diagonal precision must be positive")
    # Check labels before FP32 conversion; rounding a working response to 0/1
    # must not turn an invalid typed phenotype into apparently valid labels.
    raw_y = None if phenotype is None else torch.as_tensor(phenotype, dtype=torch.float64, device=device)
    if raw_y is not None and (raw_y.shape != (n,) or not bool(torch.isfinite(raw_y).all()) or bool(((raw_y != 0) & (raw_y != 1)).any())):
        raise ValueError("binary phenotype must be aligned finite 0/1 labels; working response is separate")
    y, working = tensor(phenotype), tensor(working_phenotype)
    if working is not None and (working.shape != (n,) or not bool(torch.isfinite(working).all())):
        raise ValueError("binary working_phenotype must be an aligned finite vector")
    return BinaryNullModel(ids,x,r,mu,wx,left,cov,inverse,sx,tensor(coefficients),y,
                           has_kinship=has_kinship,use_spa=use_spa,fit_method=provenance,
                           matmul_mode=mode,working_phenotype=working)


def fit_logistic_null(phenotype,covariates=None,*,sample_ids=None,device="cuda",
                      use_spa=True,maxiter=100,tol=1e-8):
    """Float64 GPU binomial IRLS, with stable likelihood step halving."""
    y=torch.as_tensor(phenotype,dtype=torch.float64,device=device)
    if y.ndim!=1 or y.numel()<2 or not bool(torch.isfinite(y).all()) or bool(((y!=0)&(y!=1)).any()) or not 0<float(y.mean())<1:
        raise ValueError("logistic phenotype must contain both finite 0 and 1 outcomes")
    n=y.numel();x=torch.ones((n,1),dtype=y.dtype,device=y.device) if covariates is None else torch.as_tensor(covariates,dtype=y.dtype,device=y.device)
    if x.ndim!=2 or x.shape[0]!=n or not bool(torch.isfinite(x).all()) or int(torch.linalg.matrix_rank(x))!=x.shape[1]:
        raise ValueError("logistic design must be finite, aligned and full rank")
    if not 0<tol<1 or maxiter<2:
        raise ValueError("invalid logistic fitting tolerance or iteration limit")
    ids=np.asarray([str(j) for j in range(n)] if sample_ids is None else sample_ids,dtype=str)
    beta=torch.zeros(x.shape[1],dtype=y.dtype,device=y.device)
    # stats::binomial initialization for an unweighted 0/1 response.
    eta=torch.logit((y+.5)/2)
    def loss(z):return (torch.nn.functional.softplus(z)-y*z).sum()
    current_loss=loss(eta)
    for iteration in range(1,maxiter+1):
        mu=torch.sigmoid(eta);w=mu*(1-mu)
        if bool((w<=0).any()):raise ArithmeticError("logistic fit saturated; possible separation")
        z=eta+(y-mu)/w
        sqrtw=torch.sqrt(w)
        candidate=torch.linalg.lstsq(x*sqrtw[:,None],z*sqrtw,driver="gels").solution
        delta=candidate-beta
        for _ in range(64):
            candidate=beta+delta;next_eta=x@candidate;next_loss=loss(next_eta)
            if bool(torch.isfinite(next_loss)&(((iteration==1)|(next_loss<=current_loss+1e-10)))):
                break
            delta/=2
        else:raise ArithmeticError("logistic IRLS step halving did not find a finite likelihood improvement")
        change=float((next_loss-current_loss).abs()/(.05+next_loss.abs()))
        beta,eta,current_loss=candidate,next_eta,next_loss
        if change<tol:break
    else:raise ArithmeticError("logistic IRLS did not converge")
    mu=torch.sigmoid(eta);w=mu*(1-mu)
    if bool((w<=0).any()):raise ArithmeticError("logistic fit saturated; possible separation")
    cov=torch.linalg.inv(x.T@(w[:,None]*x))
    model=binary_prefitted_state(sample_ids=ids,covariates=x,residual=y-mu,fitted_probability=mu,
        xw=(x*w[:,None]).T,projection_left=x@cov,fixed_effect_covariance=cov,
        coefficients=beta,phenotype=y,device=device,use_spa=use_spa,provenance="ordinary_logistic_IRLS")
    model.iterations=iteration
    return model
