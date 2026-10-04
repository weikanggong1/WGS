"""PyTorch quantitative-trait score association; no REGENIE runtime calls."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .phenotype import inverse_normal_transform, covariate_basis


@dataclass
class SingleVariantConfig:
    block_size: int = 1000
    maf_min: float = 0.001
    min_mac: float = 20.
    apply_rint: bool = True
    device: str = 'cuda'
    dtype: str = 'float32'
    tf32: bool = True

    def __post_init__(self):
        if self.block_size<1 or self.min_mac<0 or not 0<=self.maf_min<=.5:
            raise ValueError('Require block_size>0, min_mac>=0, and 0<=maf_min<=0.5')
        if self.dtype not in ('float32','float64'):
            raise ValueError('Supported analysis dtypes: float32, float64')


@dataclass
class TestContext:
    y: torch.Tensor
    y_scale: float
    residual_scale: float
    covariates_q: torch.Tensor
    sample_indices: torch.Tensor
    sample_ids: list
    metadata: dict
    y_float64: torch.Tensor | None = None
    covariates_q_float64: torch.Tensor | None = None

    def residualize(self, genotypes, *, dtype=None):
        selected_dtype=self.y.dtype if dtype is None else dtype
        g = torch.as_tensor(genotypes, device=self.y.device, dtype=selected_dtype)
        q=self.covariates_q_float64 if selected_dtype==torch.float64 and self.covariates_q_float64 is not None else self.covariates_q
        q=q.to(selected_dtype)
        means = torch.nanmean(g, dim=0)
        means = torch.nan_to_num(means)
        g = torch.where(torch.isfinite(g), g, means)
        return g-q@(q.T@g)


def create_test_context(phenotype, loco=None, *, covariates=None, sample_ids=None,
                        apply_rint=True, device='cuda', dtype='float32', tf32=True):
    if dtype not in ('float32', 'float64'):
        raise ValueError('Supported analysis dtypes: float32, float64')
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32 and dtype == 'float32')
    torch.backends.cudnn.allow_tf32 = bool(tf32 and dtype == 'float32')
    target_dtype = getattr(torch, dtype)
    # Covariate projection and trait scaling use double precision even in GPU
    # float32 mode; the genotype block/matrix workload uses the requested dtype.
    y = torch.as_tensor(phenotype, device=device, dtype=torch.float64)
    x = None if covariates is None else torch.as_tensor(covariates, device=device, dtype=torch.float64)
    if x is not None and x.ndim == 1: x = x[:, None]
    valid = torch.isfinite(y)
    if x is not None: valid &= torch.isfinite(x).all(1)
    predictions = torch.zeros_like(y) if loco is None else torch.as_tensor(loco, device=device, dtype=torch.float64)
    if predictions.shape != y.shape:
        raise ValueError('LOCO and phenotype must have equal aligned sample counts')
    valid &= torch.isfinite(predictions)
    indices = valid.nonzero().flatten()
    y, predictions = y[indices], predictions[indices]
    if apply_rint: y = inverse_normal_transform(y)
    q = covariate_basis(None if x is None else x[indices], len(y), device=device)
    df = len(y)-q.shape[1]
    if df < 2: raise ValueError('Insufficient residual degrees of freedom')
    y = y-q@(q.T@y)
    y_scale = torch.linalg.vector_norm(y)/math.sqrt(df)
    residual = y/y_scale-predictions
    residual_scale = torch.linalg.vector_norm(residual)/math.sqrt(df)
    if not (y_scale > 0 and residual_scale > 0):
        raise ValueError('Constant phenotype or zero residual variance')
    ids = [] if sample_ids is None else [sample_ids[i] for i in indices.cpu().tolist()]
    return TestContext((residual/residual_scale).to(target_dtype), float(y_scale),
                       float(residual_scale), q.to(target_dtype), indices.cpu(), ids,
                       {'n': len(y), 'n_input':len(phenotype), 'df': df, 'apply_rint': apply_rint,
                        'dtype': dtype, 'device': device, 'tf32': tf32},residual/residual_scale,q)


def score_genotypes(genotypes, context):
    """Return observed N/AAF/MAC and QT score statistics on original A1 scale."""
    g = torch.as_tensor(genotypes, device=context.y.device, dtype=context.y.dtype)
    called = torch.isfinite(g)
    n_called = called.sum(0)
    alt_count = torch.where(called, g, 0.).sum(0)
    aaf = alt_count/(2*n_called.clamp_min(1))
    maf = torch.minimum(aaf, 1-aaf)
    mac = torch.minimum(alt_count, 2*n_called-alt_count)
    centered = context.residualize(g)
    # Float64 reductions retain small/rare score signals; large projection is GPU.
    u = (centered.double()*context.y.double()[:, None]).sum(0)
    v = centered.double().square().sum(0)
    scale = context.y_scale*context.residual_scale
    beta = u*scale/v
    se = scale/v.sqrt()
    chi2 = u.square()/v
    from .statistics import chi2_logsf
    logp = chi2_logsf(chi2, 1.)
    return {'N': n_called, 'A1FREQ': aaf, 'MAF': maf, 'MAC': mac,
            'BETA': beta, 'SE': se, 'CHISQ': chi2, 'LOG10P': logp,
            'VALID': (n_called > 0) & (v > torch.finfo(torch.float64).eps)}


def iter_single_variant_results(reader, context, *, config=None, variant_indices=None):
    config = SingleVariantConfig() if config is None else config
    sample_rows=None
    if context.sample_ids and list(reader.sample_ids)!=list(context.sample_ids):
        lookup={sid:i for i,sid in enumerate(reader.sample_ids)}
        try:sample_rows=torch.tensor([lookup[sid] for sid in context.sample_ids],dtype=torch.int64)
        except KeyError as error:raise ValueError('Context sample absent from genotype reader') from error
    elif reader.n_samples!=len(context.y):
        sample_rows=context.sample_indices
    for variants, genotypes in reader.iter_variant_blocks(config.block_size, variant_indices):
        if sample_rows is not None:genotypes=genotypes[sample_rows]
        stats = score_genotypes(genotypes, context)
        keep = stats['VALID'] & (stats['MAC'] >= config.min_mac) & (stats['MAF'] > config.maf_min)
        columns = {k: v.detach().cpu().tolist() for k, v in stats.items() if k != 'VALID'}
        for j in keep.nonzero().flatten().cpu().tolist():
            variant = variants[j]
            yield {'CHROM': variant.chrom, 'GENPOS': variant.position, 'ID': variant.id,
                   'ALLELE0': variant.allele0, 'ALLELE1': variant.allele1,
                   'TEST': 'ADD', 'EXTRA': 'NA', **{k: values[j] for k, values in columns.items()}}


def test_single_variant(reader, context, *, config=None, output_path=None, variant_indices=None):
    """Write streaming TSV or collect a bounded result table if no path is given."""
    import pandas as pd
    rows = iter_single_variant_results(reader, context, config=config, variant_indices=variant_indices)
    if output_path is None:
        return pd.DataFrame(rows)
    from .output import RESULT_COLUMNS, _native
    from pathlib import Path
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write(' '.join(RESULT_COLUMNS)+'\n')
        for row in rows: stream.write(' '.join(_native(row[k]) for k in RESULT_COLUMNS)+'\n')
    return path
