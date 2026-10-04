"""Explicit raw-IDP and pre-residualized phenotype preparation."""
from dataclasses import dataclass
import torch
from .step1 import rank_inverse_normal as inverse_normal_transform, covariate_basis


@dataclass
class PreparedPhenotype:
    values: torch.Tensor
    sample_indices: torch.Tensor
    metadata: dict


def prepare_phenotype(phenotype, *, mode='residual', covariates=None,
                      outlier_sd=5., device='cuda', quantile_normalize=True):
    """Raw: filter 5SD, normalize, remove supplied paper covariates.

    Residual: filter missing only. REGENIE-stage RINT is handled by fit_null and
    create_test_context. Call separately for each discovery phenotype/cohort.
    """
    if mode not in ('raw', 'residual'):
        raise ValueError('mode must be raw or residual')
    y = torch.as_tensor(phenotype, dtype=torch.float64, device=device)
    valid = torch.isfinite(y)
    x = None if covariates is None else torch.as_tensor(covariates, dtype=torch.float64, device=device)
    if x is not None:
        if x.ndim == 1: x = x[:, None]
        valid &= torch.isfinite(x).all(1)
    if mode == 'raw':
        if x is None:
            raise ValueError('Raw mode needs explicitly supplied paper confound columns')
        values = y[valid]
        mean, sd = values.mean(), values.std(correction=1)
        valid &= (y-mean).abs() <= outlier_sd*sd
    indices = valid.nonzero().flatten()
    y = y[indices]
    if mode == 'raw':
        if quantile_normalize:
            y = inverse_normal_transform(y)
        q = covariate_basis(x[indices], len(y), device=device)
        y = y-q@(q.T@y)
    return PreparedPhenotype(y.cpu(), indices.cpu(), {
        'mode': mode, 'n_input': len(phenotype), 'n_valid': len(y),
        'outlier_sd': outlier_sd if mode == 'raw' else None,
        'quantile_normalize': quantile_normalize if mode == 'raw' else False})
