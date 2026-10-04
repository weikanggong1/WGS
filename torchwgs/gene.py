"""Quantitative-trait gene tests with genotype algebra performed in PyTorch.

Public entry points accept either an in-memory ALT dosage matrix or a BedReader.
Tail algorithms live in statistics.py; no REGENIE executable is used at runtime.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from . import statistics as stats
from .masks import (Annotation, GeneConfig, GeneMaskBuilder, GeneSet, MaskDefinition,
                    PreparedMask, load_annotations, load_mask_definitions,
                    load_setlist, orient_alt)


@dataclass
class GeneArtifacts:
    """Optional in-memory masks for the caller's PLINK mask writer."""
    gene: GeneSet
    masks: list[PreparedMask]


def _logp_chisq(logp: torch.Tensor) -> torch.Tensor:
    inverse = getattr(stats, "chi2_isf_log10p", None)
    if inverse is not None:
        return inverse(logp)
    # Monotone GPU inversion used only when the statistics module has no inverse.
    target = torch.as_tensor(logp, dtype=torch.float64)
    lower = torch.zeros_like(target)
    upper = (target * (2 * 2.302585092994046) + 10).clamp_min(10)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        lower = torch.where(stats.chi2_logsf(midpoint) < target, midpoint, lower)
        upper = torch.where(stats.chi2_logsf(midpoint) >= target, midpoint, upper)
    return (lower + upper) / 2


def _score_covariance(genotypes: torch.Tensor, context):
    # Applying Q before products avoids introducing an extra intercept or LOCO term.
    # Native gene tests project small mask matrices in double precision.  Merely
    # converting a float32 projection to double would retain its rounding error.
    if getattr(context, "covariates_q_float64", None) is not None:
        residual = context.residualize(genotypes, dtype=torch.float64)
    else:
        residual = context.residualize(genotypes).to(torch.float64)
    y = _trait_float64(context).to(residual.device).reshape(-1)
    return residual.T @ y, residual.T @ residual, residual


def _trait_float64(context):
    original = getattr(context, "y_float64", None)
    return (context.y if original is None else original).to(torch.float64)


def _row(gene: GeneSet, n: int, test: str, logp: torch.Tensor | float | None,
         *, mask: PreparedMask | None = None, beta=None, se=None, chisq=None,
         df: int | None = 1, strongest: str | None = None):
    value = None if logp is None else float(torch.as_tensor(logp).detach().cpu().item())
    if value is not None and (value < 0 or not torch.isfinite(torch.tensor(value))):
        value = None
    if chisq is None and value is not None:
        chisq = float(_logp_chisq(torch.as_tensor(logp)).detach().cpu().item())
    if chisq is not None:
        chisq = float(torch.as_tensor(chisq).detach().cpu().item())
    if mask is None:
        identifier, allele0, allele1, aaf = gene.gene, None, None, None
    else:
        frequency = "all" if mask.aaf_upper == 1 else mask.frequency
        suffix = f"{mask.base_name}.{frequency}"
        identifier = f"{gene.gene}.{mask.name}.{frequency}"
        allele0, allele1 = "ref", suffix
        aaf = mask.aaf if test == "ADD" else None
    extra = "DF=NA" if df is None or value is None else f"DF={df}"
    if strongest:
        extra += f";STRONGEST_MASK={strongest}"
    return {"CHROM": gene.chrom, "GENPOS": gene.position, "ID": identifier,
            "ALLELE0": allele0, "ALLELE1": allele1, "A1FREQ": aaf, "N": n,
            "TEST": test, "BETA": None if beta is None else float(beta),
            "SE": None if se is None else float(se), "CHISQ": chisq,
            "LOG10P": value, "EXTRA": extra}


def _independent_columns(matrix: torch.Tensor, tolerance: float,
                         tie_tolerance: float = 0.) -> list[int]:
    """Column-pivoted Householder QR, retaining original signed burden columns.

    Stable partial norm downdates follow the Businger-Golub QR algorithm used
    by Eigen's ColPivHouseholderQR.  No CPU matrix factorization is called.
    """
    qr = matrix.clone()
    # Equal standardized columns can select different signed cones if their
    # long-vector norms use a different floating-point reduction order.
    updated = stats.eigen_compatible_column_norm(qr)
    direct = updated.clone()
    if updated.numel() == 0 or float(updated.max()) <= 0:
        return []
    permutation = list(range(matrix.shape[1]))
    diagonal = []
    downdate_threshold = torch.finfo(qr.dtype).eps ** .5
    for k in range(min(qr.shape)):
        if tie_tolerance:
            maximum = updated[k:].max()
            ties = (updated[k:] >= maximum * (1-tie_tolerance)).nonzero().flatten().tolist()
            pivot = k + min(ties, key=lambda index: permutation[k+index])
        else:
            pivot = k + int(updated[k:].argmax().item())
        if pivot != k:
            qr[:, [k, pivot]] = qr[:, [pivot, k]]
            updated[[k, pivot]] = updated[[pivot, k]]
            direct[[k, pivot]] = direct[[pivot, k]]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
        vector = qr[k:, k].clone()
        first = vector[0].clone()
        tail_norm2 = vector[1:].square().sum()
        if float(tail_norm2) == 0:
            beta, tau = first, torch.zeros_like(first)
            vector.zero_()
            vector[0] = 1
        else:
            beta = -torch.copysign((first.square() + tail_norm2).sqrt(), first)
            vector[1:] /= first - beta
            vector[0] = 1
            tau = (beta - first) / beta
        qr[k, k] = beta
        qr[k+1:, k] = vector[1:]
        diagonal.append(beta.abs())
        if k + 1 == qr.shape[1]:
            continue
        trailing = qr[k:, k+1:]
        trailing -= tau * vector[:, None] * (vector @ trailing)[None, :]
        current = updated[k+1:]
        ratio = qr[k, k+1:].abs() / current.clamp_min(torch.finfo(qr.dtype).tiny)
        remaining_fraction = ((1 + ratio) * (1 - ratio)).clamp_min(0)
        accuracy = remaining_fraction * (current / direct[k+1:].clamp_min(torch.finfo(qr.dtype).tiny)).square()
        recompute = (accuracy <= downdate_threshold) & (current > 0)
        if bool(recompute.any()):
            recalculated = qr[k+1:, k+1:][:, recompute].norm(dim=0)
            direct[k+1:][recompute] = recalculated
            updated[k+1:][recompute] = recalculated
        keep_update = ~recompute
        updated[k+1:][keep_update] *= remaining_fraction[keep_update].sqrt()
    diagonal = torch.stack(diagonal)
    rank = int((diagonal > diagonal.max() * tolerance).sum().item())
    # Native SBAT keeps input order when QR removes no column.
    return list(range(matrix.shape[1])) if rank == matrix.shape[1] else permutation[:rank]


def test_prepared_gene(gene: GeneSet, prepared: Sequence[PreparedMask], context,
                       config: GeneConfig | None = None, *, qr_sample_rows=None,
                       qr_n_samples: int | None = None) -> list[dict[str, Any]]:
    """Test one gene's masks, then apply the four-component GENE-P strategy."""
    config = config or GeneConfig()
    n = context.y.numel()
    rows, burden_info, vc_info = [], [], []
    scale = float(context.y_scale) * float(context.residual_scale)
    for mask in prepared:
        score, covariance, residual = _score_covariance(mask.burden[:, None], context)
        variance = covariance[0, 0]
        residual_df = max(1, n - context.covariates_q.shape[1])
        if float(variance) <= residual_df * config.genotype_scale_tolerance ** 2:
            continue  # A constant mask has no valid ADD or corresponding VC output.
        chisq = score[0].square() / variance
        lp = stats.chi2_logsf(chisq)
        burden_row = _row(gene, n, "ADD", lp, mask=mask, beta=score[0] * scale / variance,
                          se=scale / variance.sqrt(), chisq=chisq)
        burden_info.append((mask, lp, residual[:, 0]))
        if mask.vc_genotypes is None:
            rows.append(burden_row)
            continue
        vc_score, vc_cov, _ = _score_covariance(mask.vc_genotypes, context)
        usable = vc_cov.diag() > torch.finfo(torch.float64).eps
        if not bool(usable.any()):
            rows.append(burden_row)
            continue
        vc_score, vc_cov = vc_score[usable], vc_cov[usable][:, usable]
        weights = mask.vc_weights[usable].to(vc_score.device)
        acat_weights = mask.acat_weights[usable].to(vc_score.device)
        single_logps = stats.chi2_logsf(vc_score.square() / vc_cov.diag())
        acatv = stats.acat_logp(single_logps, weights=acat_weights)
        kernel_tests = stats.skato_logp(vc_score, vc_cov, weights=weights,
                                       rhos=config.skato_rhos, tail_method=config.tail_method)
        rho_logps = kernel_tests["rho_log10ps"]
        rhos = torch.as_tensor(kernel_tests["rhos"], device=vc_score.device)
        if vc_score.numel() == 1:
            # Native one-site masks bypass ACAT's P≈1 clipping entirely.
            acatv = single_logps[0]
            acato = single_logps[0]
            kernel_tests = dict(kernel_tests)
            for test in ("SKAT", "SKATO", "SKATO-ACAT"):
                kernel_tests[test] = single_logps[0]
        elif config.acato_full:
            acato = stats.acat_logp(torch.cat([acatv.reshape(1), rho_logps.reshape(-1)]))
        else:
            # Multi-rho REGENIE internally caps the upper rho at 0.999.
            endpoint = (rhos == rhos.min()) | (rhos == rhos.max())
            acato = stats.acat_logp(torch.cat([acatv.reshape(1), rho_logps[endpoint]]))
        tests = {"ACATO": acato, "ACATV": acatv,
                 "SKAT": kernel_tests["SKAT"], "SKATO": kernel_tests["SKATO"],
                 "SKATO-ACAT": kernel_tests["SKATO-ACAT"]}
        for name, pvalue in tests.items():
            rows.append(_row(gene, n, "ADD-" + name, pvalue, mask=mask))
        rows.append(burden_row)
        vc_info.append((mask, acatv, kernel_tests["SKATO-ACAT"]))
    if not config.gene_p or not burden_info:
        return rows
    groups = {"": {m.base_name for m, _, _ in burden_info}}
    if config.gene_p_groups:
        groups = {str(name): set(names) for name, names in config.gene_p_groups.items()}
    for group, base_names in groups.items():
        burden = [(m, p, r) for m, p, r in burden_info if m.base_name in base_names]
        vc = [(m, a, s) for m, a, s in vc_info if m.base_name in base_names]
        if not burden:
            continue
        suffix = "" if not group else "_" + group
        components = []
        burden_acat = stats.acat_logp(torch.stack([p for _, p, _ in burden]))
        components.append(burden_acat)
        rows.append(_row(gene, n, "ADD-BURDEN-ACAT" + suffix, burden_acat, df=len(burden)))
        if config.run_sbat:
            matrix = torch.stack([r for _, _, r in burden], 1)
            # REGENIE projects and normalizes each ADD genotype before QR.
            # Pivoting unnormalized masks can choose a different positive cone
            # when an overall burden is a linear combination of domain burdens.
            # Native QR retains excluded phenotype rows as zeros in their BED
            # order.  Keeping those rows also preserves its norm/pivot arithmetic.
            metadata = getattr(context, "metadata", {})
            full_n = qr_n_samples if qr_n_samples is not None else metadata.get("n_input", n)
            positions = qr_sample_rows
            if positions is None and full_n > n:
                positions = getattr(context, "sample_indices", None)
            qr_matrix = matrix
            if full_n > n:
                if positions is None or len(positions) != n:
                    raise ValueError("Native SBAT QR padding requires aligned original sample positions.")
                padded = torch.zeros((full_n, matrix.shape[1]), device=matrix.device,
                                     dtype=matrix.dtype)
                padded[torch.as_tensor(positions, device=matrix.device, dtype=torch.long)] = matrix
                qr_matrix = padded
            norm = stats.eigen_compatible_column_norm(qr_matrix)
            residual_df = n - context.covariates_q.shape[1]
            normalized = qr_matrix / (norm / residual_df ** .5)[None, :]
            independent = _independent_columns(normalized, config.rank_tolerance,
                                              config.qr_tie_tolerance)
            if independent:
                matrix = matrix[:, independent]
                y = _trait_float64(context).to(matrix.device).reshape(-1)
                score, covariance = matrix.T @ y, matrix.T @ matrix
                q = context.covariates_q.shape[1]
                degrees = n - q - len(independent)
                if degrees > 0:
                    explained = score @ torch.linalg.solve(covariance, score)
                    variance_scale = (y.square().sum() - explained) / degrees
                    sbat = stats.sbat_logp(score, covariance, variance_scale=variance_scale,
                                          max_subsets=config.sbat_max_subsets,
                                          qmc_samples=config.sbat_qmc_samples,
                                          seed=config.sbat_seed)
                    components.append(sbat["SBAT"])
                    for name in ("SBAT", "SBAT_POS", "SBAT_NEG"):
                        rows.append(_row(gene, n, "ADD-BURDEN-" + name + suffix,
                                         sbat[name], df=len(independent)))
        if vc:
            for test, position in (("ACATV-ACAT", 1), ("SKATO-ACAT", 2)):
                pvalue = stats.acat_logp(torch.stack([item[position] for item in vc]))
                components.append(pvalue)
                rows.append(_row(gene, n, "ADD-" + test + suffix, pvalue, df=len(vc)))
        overall = stats.acat_logp(torch.stack(components))
        strongest_candidates = [(m.base_name, p) for m, p, _ in burden]
        strongest_candidates += [(m.base_name, p) for m, a, s in vc for p in (a, s)]
        candidate_name, candidate_logp = max(strongest_candidates, key=lambda pair: float(pair[1]))
        strongest = candidate_name if float(candidate_logp) > 0 else None
        rows.append(_row(gene, n, "GENE_P" + suffix, overall, df=len(components), strongest=strongest))
    return rows


def _normalize_annotations(annotation, genes) -> list[Annotation]:
    if isinstance(annotation, (str, Path)):
        return load_annotations(annotation, genes=genes)
    if hasattr(annotation, "to_dict"):
        annotation = annotation.to_dict("records")
    result = []
    for record in annotation:
        if isinstance(record, Annotation):
            result.append(record)
        else:
            result.append(Annotation(str(record["variant_id"]), str(record["gene"]),
                                     str(record["category"]), record.get("domain")))
    return result


def _context_rows(reader, context) -> torch.Tensor | None:
    if not context.sample_ids:
        if reader.n_samples == context.y.numel():
            return None
        indices = getattr(context, "sample_indices", None)
        if indices is None or len(indices) != context.y.numel():
            raise ValueError("BED/context sample alignment requires sample_ids or sample_indices.")
        return torch.as_tensor(indices, dtype=torch.long, device="cpu")
    if tuple(reader.sample_ids) == tuple(context.sample_ids):
        return None
    index = {tuple(sample): i for i, sample in enumerate(reader.sample_ids)}
    try:
        selected = [index[tuple(sample)] for sample in context.sample_ids]
    except KeyError as error:
        raise ValueError("Analysis context contains a sample absent from the BED file.") from error
    return torch.tensor(selected, dtype=torch.long)


def test_gene_based(genotypes, context, annotation, setlist, masks,
                    config: GeneConfig | None = None, *, variants: Sequence[object] | None = None,
                    artifact_callback=None) -> Iterator[dict[str, Any]]:
    """Yield REGENIE-shaped rows for all selected genes in one chromosome context.

    ``genotypes`` is [N,V] ALT dosage when orientation='alt', or a BedReader.
    Matrix input requires explicit Variant metadata in ``variants``.  Annotation,
    setlist and mask arguments accept parsed dataclasses or original text paths.
    Sub analysis is supplied by MaskDefinition.extract_variants or config's global
    whitelist; category/score names are deliberately not hard-coded.
    """
    config = config or GeneConfig()
    sets = load_setlist(setlist, config.extract_genes) if isinstance(setlist, (str, Path)) else list(setlist)
    if config.extract_genes is not None:
        allowed = set(config.extract_genes)
        sets = [s for s in sets if s.gene in allowed]
    annotations = _normalize_annotations(annotation, {s.gene for s in sets})
    definitions = load_mask_definitions(masks) if isinstance(masks, (str, Path)) else list(masks)
    by_gene = defaultdict(list)
    for record in annotations:
        by_gene[record.gene].append(record)
    reader = hasattr(genotypes, "read_variants")
    if reader:
        needed = {record.variant_id for record in annotations}
        lookup = genotypes.find_variants(needed)
        sample_rows = _context_rows(genotypes, context)
    else:
        if variants is None:
            raise ValueError("Matrix input requires explicit variants metadata.")
        matrix = torch.as_tensor(genotypes)
        if matrix.shape != (context.y.numel(), len(variants)):
            raise ValueError("Matrix rows must match context samples and columns must match variants.")
        lookup = {variant.id: variant for variant in variants}
        matrix_columns = {variant.id: j for j, variant in enumerate(variants)}
    for gene in sets:
        gene_annotations = by_gene[gene.gene]
        if not gene_annotations:
            continue
        # Native set-list parsing sorts retained sites by BED/BIM index.
        present = sorted((lookup[v] for v in gene.variant_ids if v in lookup),
                         key=lambda variant: variant.index)
        if not present:
            continue
        builder = GeneMaskBuilder(gene_annotations, definitions, context.y.numel(),
                                  context.y.device, torch.float64, config)
        for start in range(0, len(present), config.variant_block_size):
            block = present[start:start + config.variant_block_size]
            if reader:
                raw = genotypes.read_variants([v.index for v in block])
                if sample_rows is not None:
                    raw = raw[sample_rows]
            else:
                columns = [matrix_columns[v.id] for v in block]
                raw = matrix[:, columns]
            raw = torch.as_tensor(raw, device=context.y.device, dtype=torch.float64)
            raw = orient_alt(raw, block, config.genotype_orientation)
            builder.update([v.id for v in block], raw)
        prepared = builder.finish()
        if artifact_callback is not None:
            artifact_callback(GeneArtifacts(gene, prepared))
        yield from test_prepared_gene(gene, prepared, context, config,
                                      qr_sample_rows=sample_rows if reader else None,
                                      qr_n_samples=genotypes.n_samples if reader else None)
