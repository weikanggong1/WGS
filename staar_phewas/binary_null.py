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

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def device(self):
        return self.x.device

    @property
    def dof(self):
        return self.n - self.x.shape[1]

    def score_covariance(self, genotype):
        g = torch.as_tensor(genotype, dtype=torch.float64, device=self.device)
        if self.precision is None:
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.fitted_probability * (1-self.fitted_probability))
        if self.precision.ndim == 1:
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.precision)
        return score_covariance(g, self.scaled_residuals, precision=self.precision,
                                precision_covariates=self.precision_x,
                                fixed_effect_covariance=self.fixed_effect_covariance)


def binary_prefitted_state(*, sample_ids, covariates, residual, fitted_probability,
                           xw, projection_left, fixed_effect_covariance,
                           precision=None, precision_covariates=None,
                           coefficients=None, phenotype=None, has_kinship=False,
                           use_spa=True, device="cuda", provenance="prefitted_binary_state"):
    """Use explicitly supplied state; this function performs no null fitting.

    Ordinary xw=X.T*mu*(1-mu), left=X*inv(X.T*W*X). Mixed xw=X.T*Sigma_i,
    left=X*cov. A mixed state additionally needs its fitted precision and
    precision_covariates for score-based association and SPA filtering.
    """
    def tensor(value):
        return None if value is None else torch.as_tensor(value, dtype=torch.float64, device=device)
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
    return BinaryNullModel(ids,x,r,mu,wx,left,cov,inverse,sx,tensor(coefficients),tensor(phenotype),
                           has_kinship=has_kinship,use_spa=use_spa,fit_method=provenance)


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
