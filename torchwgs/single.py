"""PyTorch quantitative-trait score association; no REGENIE runtime calls."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .phenotype import inverse_normal_transform, covariate_basis


@dataclass
class SingleVariantConfig:
    block_size: int = 1000
    maf_min: float = 0.
    min_mac: float = 20.
    apply_rint: bool = True
    device: str = 'cuda'
    dtype: str = 'float32'
    tf32: bool = True
    genotype_reader: str = 'cpu'

    def __post_init__(self):
        if self.block_size<1 or self.min_mac<0 or not 0<=self.maf_min<=.5:
            raise ValueError('Require block_size>0, min_mac>=0, and 0<=maf_min<=0.5')
        if self.dtype not in ('float32','float64'):
            raise ValueError('Supported analysis dtypes: float32, float64')
        if self.genotype_reader not in ('cpu','cuda_packed'):
            raise ValueError('genotype_reader must be cpu or cuda_packed')


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
    if not indices.numel():
        raise ValueError('No finite aligned phenotype/LOCO samples remain')
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


def _allele_counts(g):
    """Exact observed counts before imputation or projection."""
    called = torch.isfinite(g)
    n_called = called.sum(0)
    alt_count = torch.where(called, g, 0.).sum(0, dtype=torch.float64)
    return _counts_from_observed(n_called, alt_count)


def _counts_from_observed(n_called, alt_count):
    """Use identical float64 frequency/MAC arithmetic for both BED readers."""
    alt_count = alt_count.to(torch.float64)
    aaf = alt_count/(2*n_called.clamp_min(1))
    maf = torch.minimum(aaf, 1-aaf)
    mac = torch.minimum(alt_count, 2*n_called-alt_count)
    return {'N': n_called, 'A1FREQ': aaf, 'MAF': maf, 'MAC': mac}


def _score_counted_genotypes(g, context, counts):
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
    return {**counts,
            'BETA': beta, 'SE': se, 'CHISQ': chi2, 'LOG10P': logp,
            'VALID': (counts['N'] > 0) & (v > torch.finfo(torch.float64).eps)}


def score_genotypes(genotypes, context):
    """Return observed N/AAF/MAC and QT score statistics on original A1 scale."""
    g = torch.as_tensor(genotypes, device=context.y.device, dtype=context.y.dtype)
    return _score_counted_genotypes(g, context, _allele_counts(g))


def _copy_result_columns(statistics):
    """Copy floating statistics and validity together, preserving integer N.

    Integer sample counts keep their original Python integer representation.
    Floating values are promoted losslessly to float64 for one bounded block
    transfer; this also avoids a separate nonzero/host-copy for validity.
    """
    names = [name for name in statistics if name not in ('N', 'VALID')]
    floating = torch.stack([statistics[name].to(torch.float64) for name in names]
                           + [statistics['VALID'].to(torch.float64)])
    copied = floating.detach().cpu().tolist()
    columns = dict(zip(names, copied[:-1]))
    columns['N'] = statistics['N'].detach().cpu().tolist()
    columns = {name: columns[name] for name in statistics if name != 'VALID'}
    return columns, copied[-1]


def iter_single_variant_results(reader, context, *, config=None, variant_indices=None):
    config = SingleVariantConfig() if config is None else config
    sample_rows=None
    if context.sample_ids and list(reader.sample_ids)!=list(context.sample_ids):
        lookup={sid:i for i,sid in enumerate(reader.sample_ids)}
        try:sample_rows=torch.tensor([lookup[sid] for sid in context.sample_ids],dtype=torch.int64)
        except KeyError as error:raise ValueError('Context sample absent from genotype reader') from error
    elif reader.n_samples!=len(context.y):
        sample_rows=context.sample_indices
    if config.genotype_reader=='cuda_packed' and context.y.device.type!='cuda':
        raise ValueError('cuda_packed single-variant reader requires a CUDA context')
    packed_blocks = config.genotype_reader == 'cuda_packed' and hasattr(reader, 'iter_packed_variant_blocks')
    if packed_blocks:
        blocks = reader.iter_packed_variant_blocks(config.block_size, variant_indices,
                 sample_rows=sample_rows, device=context.y.device, dtype=context.y.dtype)
    else:
        blocks = reader.iter_variant_blocks(config.block_size,variant_indices,
                 genotype_reader=config.genotype_reader,sample_rows=sample_rows,
                 device=context.y.device,dtype=context.y.dtype)
    for variants, genotypes in blocks:
        if packed_blocks:
            observed = genotypes.allele_counts()
            counts = _counts_from_observed(observed['N'], observed['AAC'])
        else:
            g = torch.as_tensor(genotypes, device=context.y.device, dtype=context.y.dtype)
            counts = _allele_counts(g)
        # WGS contains many ultra-rare variants.  Reject them before building
        # projected N x B matrices; the observed-count filter is unchanged.
        selected = ((counts['N'] > 0) & (counts['MAC'] >= config.min_mac)
                    & (counts['MAF'] > config.maf_min)).nonzero().flatten()
        if not selected.numel():
            continue
        selected_rows = selected.cpu().tolist()
        variants = [variants[j] for j in selected_rows]
        counts = {name: values[selected] for name, values in counts.items()}
        # The packed CUDA path only materializes the sites that survive MAC/MAF.
        # The CPU/reference path retains its previous decode and selection order.
        g = genotypes.decode(selected_rows) if packed_blocks else g[:, selected]
        stats = _score_counted_genotypes(g, context, counts)
        columns, valid_rows = _copy_result_columns(stats)
        # Do not keep this floating-point matrix alive while the reader uploads
        # and counts the next packed block, including consecutive empty blocks.
        del g
        for j, valid in enumerate(valid_rows):
            if not valid:
                continue
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
