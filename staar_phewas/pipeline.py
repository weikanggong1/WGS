"""Native GDS -> Gaussian PheWAS association pipeline.

SPDX-License-Identifier: GPL-3.0-only
Analysis rules follow the frozen STAARpipelinePheWAS 0.9.7.1 source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import numpy as np
import torch

from .gds import SeqArrayGDS
from .masks import (VariantAnnotations, variant_filter, coding_masks, noncoding_masks,
                    ncRNA_mask, annotation_phred_matrix, sample_union, sample_indices)
from .null_model import GaussianNullModel
from .statistics import staar_test
from .multi import multi_staar_test, joint_individual_logp
from .binary import staar_binary_spa, individual_score_test_spa
from .results import coding_record, window_record, single_variant_record, assemble_phewas_results, TraitRows


@dataclass(frozen=True)
class AnalysisOptions:
    rare_maf_cutoff: float = 0.01
    rv_num_cutoff: int = 2
    rv_num_cutoff_max: int = 1_000_000_000
    rv_num_cutoff_max_prefilter: int = 1_000_000_000
    variant_type: str = "SNV"
    imputation: str = "mean"
    genotype_block_size: int = 128
    annotation_block_size: int = 250_000
    memory_limit_gib: float = 20.0
    spa_p_filter: bool = True
    p_filter_cutoff: float = 0.05
    spa_tol: float = 2**-13
    spa_max_iter: int = 1000

    def __post_init__(self):
        if not 0 < self.rare_maf_cutoff <= 0.5:
            raise ValueError("rare_maf_cutoff must be in (0, 0.5]")
        if self.rv_num_cutoff < 1 or self.rv_num_cutoff_max <= self.rv_num_cutoff:
            raise ValueError("invalid variant-set size limits")
        if self.rv_num_cutoff_max_prefilter < 1 or self.genotype_block_size < 1 or self.annotation_block_size < 1 or self.memory_limit_gib <= 0:
            raise ValueError("prefilter, block size and memory budget must be positive")
        if self.variant_type not in ("SNV", "Indel", "variant") or self.imputation not in ("mean", "minor"):
            raise ValueError("invalid variant type or missing imputation")
        if not 0<self.p_filter_cutoff<=1 or self.spa_tol<=0 or self.spa_max_iter<1:
            raise ValueError("invalid binary SPA settings")


class PheWASPipeline:
    """Reuse union genotype blocks across single and correlated joint models.

    annotation_catalog maps functional annotation names to native GDS nodes.
    gds_sample_indices optionally gives aligned, zero-based GDS rows for each
    model, allowing an explicit external ID normalization during preparation.
    A joint Gaussian model is one analysis in the list and retains its
    complete trait covariance. Explicit binary state can use its SPA kernel.
    """
    def __init__(self, gds: SeqArrayGDS, models: Sequence[GaussianNullModel], *,
                 qc_path="annotation/filter", annotation_catalog: Mapping[str, str] | None = None,
                 annotation_names: Sequence[str] = (), gds_sample_indices=None, options=None):
        self.gds, self.models = gds, list(models)
        if not self.models:
            raise ValueError("provide at least one fitted null model")
        if any(m.family not in ("gaussian","binomial") or m.n_pheno<1 or
               (m.n_pheno>1 and (m.family!="gaussian" or m.use_spa)) for m in self.models):
            raise NotImplementedError("supported models are Gaussian single/joint and binary single-trait state")
        self.options = options or AnalysisOptions()
        self.annotation_catalog = dict(annotation_catalog or {})
        self.annotation_names = list(annotation_names)
        canonical=[];aligned_rows=[]
        if gds_sample_indices is not None and len(gds_sample_indices)!=len(self.models):
            raise ValueError("provide GDS row indices for every model")
        available=None
        for index,model in enumerate(self.models):
            if gds_sample_indices is None:
                identifiers=np.asarray(getattr(model,"gds_sample_ids",model.sample_ids),dtype=str)
                rows=gds.sample_indices(identifiers)
            else:
                raw=np.asarray(gds_sample_indices[index])
                if raw.shape!=(model.n,) or not np.issubdtype(raw.dtype,np.integer):
                    raise ValueError("model GDS row indices must be an aligned integer vector")
                rows=raw.astype(np.int64)
                if len(np.unique(rows))!=model.n or np.any(rows<0) or np.any(rows>=gds.n_samples):
                    raise ValueError("model GDS row indices must be unique and in range")
                if available is None:available=gds.sample_ids()
                identifiers=available[rows]
                bound=getattr(model,"gds_sample_ids",None)
                if bound is not None and not np.array_equal(np.asarray(bound,dtype=str),identifiers):
                    raise ValueError("model canonical IDs disagree with supplied chromosome indices")
            canonical.append(identifiers);aligned_rows.append(rows)
        # Models may have normalized external IDs; union on actual GDS IDs.
        self.union_ids=sample_union(canonical)
        self.trait_rows=[sample_indices(identifiers,self.union_ids) for identifiers in canonical]
        mapping={}
        for identifiers,rows in zip(canonical,aligned_rows):
            for identifier,row in zip(identifiers,rows):
                if identifier in mapping and mapping[identifier]!=row:
                    raise ValueError("one GDS sample maps to inconsistent rows")
                mapping[identifier]=int(row)
        self.union_rows=np.asarray([mapping[identifier] for identifier in self.union_ids],dtype=np.int64)
        self.position = np.asarray(gds.read_field("position"), dtype=np.int64)
        self.qc = gds.read_field(qc_path)
        self.qc_path = qc_path
        self.skipped_sets: list[dict] = []

    def _assemble(self, records, *, kind, **kwargs):
        flags=[model.use_spa for model in self.models]
        if len(set(flags))==1:
            return assemble_phewas_results(records,kind=kind,use_spa=flags[0],**kwargs)
        assembled=[assemble_phewas_results([rows],kind=kind,use_spa=flag,**kwargs)
                   for rows,flag in zip(records,flags)]
        if isinstance(assembled[0],dict):
            return {category:[result[category][0] for result in assembled] for category in assembled[0]}
        return [result[0] for result in assembled]

    def region_indices(self, start: int, end: int):
        if start > end:
            raise ValueError("region start must not exceed end")
        return np.flatnonzero((self.position >= start) & (self.position <= end))

    def annotations(self, indices, *, mask_fields=(), include_weights=True):
        indices = np.asarray(indices, dtype=np.int64)
        names = list(dict.fromkeys(list(mask_fields) + (self.annotation_names if include_weights else [])))
        annotations = {}
        for name in names:
            if name in self.annotation_catalog:
                annotations[name] = self.gds.read_field(self.annotation_catalog[name], indices)
            elif name in mask_fields:
                raise ValueError(f"required mask annotation missing from catalog: {name}")
        return VariantAnnotations(
            self.position[indices], self.qc[indices], annotations,
            ref=self.gds.read_field("$ref", indices), alt=self.gds.read_field("$alt", indices),
            chromosome=self.gds.read_field("chromosome", indices),
            variant_id=self.gds.read_field("variant.id", indices),
        )

    def _select_mask_chunks(self,indices,mask_fields,selector):
        """Read candidate metadata in order; load PHRED only for selected sites."""
        total=self.gds.n_variants if indices is None else len(indices)
        selected={}
        for offset in range(0,total,self.options.annotation_block_size):
            end=min(offset+self.options.annotation_block_size,total)
            rows=np.arange(offset,end,dtype=np.int64) if indices is None else indices[offset:end]
            annotations=self.annotations(rows,mask_fields=mask_fields,include_weights=False)
            for category,local in selector(annotations).items():
                selected.setdefault(category,[])
                if len(local):selected[category].append(rows[local])
        return {category:np.concatenate(chunks) if chunks else np.empty(0,dtype=np.int64)
                for category,chunks in selected.items()}

    def _limit(self, model, number_variants):
        # Dense G, rotated G, weighted G, plus covariance/eigensolver workspace.
        t=model.n_pheno
        estimate = 8 * (4 * model.n * number_variants + 6 * (t*number_variants) ** 2 + model.n*t*t)
        if estimate > self.options.memory_limit_gib * 2**30:
            raise MemoryError("variant set exceeds the configured memory budget; increase the budget or analyze smaller genomic sets")

    def test_set(self, indices, annotations=None):
        """Return one STAAR result per model, None for insufficient rare variants.

        Union MAF prefilter (<0.05 when cutoff<=0.01, otherwise <1) precedes
        per-trait MAF; allele orientation is never flipped again per trait.
        Exceptions other than insufficient variant count propagate to callers.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if annotations is None:
            annotations = self.annotations(indices)
        phred, names = annotation_phred_matrix(annotations.annotations, self.annotation_names,
                                               variant_type=self.options.variant_type,
                                               number_variants=len(indices))
        blocks = []
        columns = []
        prefilter = 0.05 if self.options.rare_maf_cutoff <= 0.01 else 1.0
        for offset, block in enumerate(self.gds.iter_minor_blocks(indices, self.union_rows, block_size=self.options.genotype_block_size)):
            union_maf = np.minimum(block.union_ref_af, 1 - block.union_ref_af)
            keep = np.isfinite(union_maf) & (union_maf < prefilter)
            if not keep.any():
                continue
            start = offset * self.options.genotype_block_size
            blocks.append((block, keep))
            columns.extend((start + np.flatnonzero(keep)).tolist())
        if len(columns) >= self.options.rv_num_cutoff_max_prefilter:
            raise ValueError("union-prefilter variant count reaches rv_num_cutoff_max_prefilter")
        results = []
        for model, rows in zip(self.models, self.trait_rows):
            pieces, frequencies = [], []
            for block, keep in blocks:
                g, maf, _, _, _ = block.trait_dense(rows, self.options.imputation)
                pieces.append(g[:, keep]); frequencies.append(maf[keep])
            if not pieces:
                results.append(None); continue
            maf = np.concatenate(frequencies)
            rare = np.isfinite(maf) & (maf > 0) & (maf < self.options.rare_maf_cutoff)
            count = int(rare.sum())
            if count < self.options.rv_num_cutoff:
                results.append(None); continue
            if count >= self.options.rv_num_cutoff_max:
                raise ValueError("rare variant count reaches rv_num_cutoff_max")
            self._limit(model, count)
            g = np.concatenate(pieces, axis=1)[:, rare]
            u, v = model.score_covariance(g)
            annotation=phred[np.asarray(columns)[rare]]
            cutoffs=dict(rare_maf_cutoff=self.options.rare_maf_cutoff,rv_num_cutoff=self.options.rv_num_cutoff,
                         rv_num_cutoff_max=self.options.rv_num_cutoff_max)
            if model.use_spa:
                result=staar_binary_spa(torch.as_tensor(g,dtype=torch.float64,device=model.device),maf[rare],
                    model.scaled_residuals,model.fitted_probability,model.xw,model.projection_left,
                    annotation,names,spa_p_filter=self.options.spa_p_filter,p_filter_cutoff=self.options.p_filter_cutoff,
                    tol=self.options.spa_tol,max_iter=self.options.spa_max_iter,covariance=v,**cutoffs)
            else:
                test=multi_staar_test if model.n_pheno>1 else staar_test
                result=test(u,v,maf[rare],np.rint(maf[rare]*2*model.n),annotation,names=names,
                            acat_calibration="chi2",cmac=float(g.sum()),**cutoffs)
            results.append(result)
        return results

    def coding(self, chromosome, gene_name, start, end, *, category="all_categories", include_ptv=False):
        if category == "all_categories_incl_ptv":
            category, include_ptv = "all_categories", True
        indices = self.region_indices(start, end)
        masks=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.EXONIC.Category","MetaSVM"),
            lambda a:coding_masks(a,gene_name,start,end,variant_type=self.options.variant_type,
                                  chromosome=chromosome,include_ptv=include_ptv))
        records = [[] for _ in self.models]
        for mask, selected in masks.items():
            if category != "all_categories" and mask not in (category, "disruptive_missense" if category == "missense" else category):
                continue
            if not len(selected):continue
            for trait,stats in enumerate(self.test_set(selected)):
                if stats is not None:records[trait].append(coding_record(chromosome,gene_name,mask,stats))
        return self._assemble(records, kind="coding", category=category, include_ptv=include_ptv)

    def noncoding(self, chromosome, gene_name, start=None, end=None, *, category="all_categories", promoter_intervals=None, include_ncrna=False):
        indices = None if start is None and end is None else self.region_indices(start,end)
        def selector(a):
            overlaps=None
            if promoter_intervals is not None:
                from .masks import promoter_overlaps
                overlaps=promoter_overlaps(a.position,a.chromosome,promoter_intervals)
            return noncoding_masks(a,gene_name,promoter_overlap=overlaps,variant_type=self.options.variant_type,
                                   chromosome=chromosome,include_ncrna=include_ncrna,
                                   requested_categories=None if category=="all_categories" else [category])
        masks=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.Info","CAGE","DHS","GeneHancer"),selector)
        records=[[] for _ in self.models]
        for mask,selected in masks.items():
            if category != "all_categories" and mask != category or not len(selected):continue
            for trait,stats in enumerate(self.test_set(selected)):
                if stats is not None:records[trait].append(coding_record(chromosome,gene_name,mask,stats))
        return self._assemble(records, kind="noncoding", category=category, include_ncrna=include_ncrna)

    def ncrna(self, chromosome, gene_name, start=None, end=None):
        indices=None if start is None and end is None else self.region_indices(start,end)
        masks=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.Info"),
            lambda a:{"ncRNA":ncRNA_mask(a,gene_name,variant_type=self.options.variant_type,chromosome=chromosome)})
        selected=masks.get("ncRNA",np.empty(0,dtype=np.int64))
        return [[coding_record(chromosome,gene_name,"ncRNA",stats)] if stats is not None else []
                for stats in self.test_set(selected)]

    def sliding(self, chromosome, start, end, *, window_length=None):
        """One inclusive region, or half-overlapping fixed-length windows."""
        if window_length is None:
            windows = [(start, end)]
        else:
            if window_length < 2 or window_length % 2 or end-start+1 < window_length:
                raise ValueError("window length must be positive/even and fit the region")
            step = window_length // 2
            number = (end-start+1) // step - 1
            windows = [(start+k*step, start+k*step+window_length-1) for k in range(number)]
        records = [[] for _ in self.models]
        for left, right in windows:
            indices = self.region_indices(left, right)
            a = self.annotations(indices,include_weights=False)
            local = np.flatnonzero(variant_filter(a, self.options.variant_type))
            for trait, stats in enumerate(self.test_set(indices[local])):
                if stats is not None:
                    records[trait].append(window_record(chromosome, left, right, stats))
        return self._assemble(records, kind="sliding")

    def individual(self, chromosome, start, end, *, mac_cutoff=20, variant_type="variant", subset_variants_num=5000):
        indices = self.region_indices(start, end)
        a = self.annotations(indices,include_weights=False)
        local = np.flatnonzero(variant_filter(a, variant_type))
        indices = indices[local]
        if subset_variants_num < 1:
            raise ValueError("subset_variants_num must be positive")
        records = [[] for _ in self.models]
        union_ordinal = 0
        for block in self.gds.iter_minor_blocks(indices, self.union_rows, block_size=self.options.genotype_block_size):
            _, _, union_mac, _, _ = block.trait_dense(np.arange(len(self.union_rows)))
            keep_union = union_mac >= mac_cutoff
            ordinals = union_ordinal + np.cumsum(keep_union) - 1
            union_ordinal += int(keep_union.sum())
            for trait, (model, rows) in enumerate(zip(self.models, self.trait_rows)):
                g, maf, mac, missing, is_alt = block.trait_dense(rows, self.options.imputation)
                keep = keep_union & (mac >= mac_cutoff)
                if not keep.any():
                    continue
                columns = np.flatnonzero(keep)
                selected = block.variant_indices[columns]
                u, v = model.score_covariance(g[:, keep])
                if model.n_pheno>1:
                    scores=u.reshape(model.n_pheno,len(columns))
                    cov4=v.reshape(model.n_pheno,len(columns),model.n_pheno,len(columns))
                    log_probabilities=torch.stack([joint_individual_logp(scores[:,j],cov4[:,j,:,j]) for j in range(len(columns))])
                else:
                    variance=v.diagonal()
                    if bool((variance < -1e-10).any()):raise ArithmeticError("individual score variance is negative")
                    positive=variance>0
                    standard_error=torch.sqrt(torch.clamp(variance,min=0))
                    z=torch.where(positive,u/torch.clamp(standard_error,min=1e-300),0.)
                    log_probabilities=-math.log(2)-torch.special.log_ndtr(-z.abs())
                    log_probabilities=torch.where(positive,log_probabilities,0.)
                    if model.use_spa:
                        probabilities=individual_score_test_spa(torch.as_tensor(g[:,keep],dtype=torch.float64,device=model.device),
                            model.scaled_residuals,model.fitted_probability,model.xw,model.projection_left,
                            normal_pvalues=torch.exp(-log_probabilities) if self.options.spa_p_filter else None,
                            p_filter_cutoff=self.options.p_filter_cutoff,tol=self.options.spa_tol,max_iter=self.options.spa_max_iter)
                chrom = self.gds.read_field("chromosome", selected)
                ref = self.gds.read_field("$ref", selected); alt = self.gds.read_field("$alt", selected)
                for j, column in enumerate(columns):
                    if model.n_pheno>1:
                        statistic={"pvalue_log":float(log_probabilities[j]),"Score":scores[:,j].detach().cpu().tolist()}
                    elif model.use_spa:
                        statistic={"pvalue":float(probabilities[j])}
                    else:
                        value,score,se=float(variance[j]),float(u[j]),float(standard_error[j])
                        statistic={"pvalue_log":float(log_probabilities[j]),"Score":score,"Score_se":se,
                                   "Est":score/value if value>0 else 0.,"Est_se":1/se if se>0 else 0.}
                    row = single_variant_record(chrom[j], self.position[selected[j]], ref[j], alt[j],
                                                maf[column] if is_alt[column] else 1-maf[column], maf[column], model.n,
                                                statistic,number_phenotypes=model.n_pheno,use_spa=model.use_spa)
                    # R appends common/high-missing groups before rare groups,
                    # then orders by POS without renumbering data.frame rows.
                    row["_chunk"] = int(ordinals[column] // subset_variants_num)
                    row["_common"] = bool(maf[column] >= 0.01 or missing[column] / model.n >= 0.01)
                    records[trait].append(row)
        tables = []
        for rows in records:
            levels = {"REF": [], "ALT": []}
            ordered = []
            for chunk in sorted({row["_chunk"] for row in rows}):
                common = [row for row in rows if row["_chunk"] == chunk and row["_common"]]
                rare = [row for row in rows if row["_chunk"] == chunk and not row["_common"]]
                groups = [common[i:i+200] for i in range(0, len(common), 200)] + ([rare] if rare else [])
                for group in groups:
                    for field in levels:
                        for value in sorted({row[field] for row in group}):
                            if value not in levels[field]:
                                levels[field].append(value)
                    ordered.extend(group)
            for index, row in enumerate(ordered, 1):
                row.pop("_chunk"); row.pop("_common")
                row["_r_row_name"] = index
            ordered.sort(key=lambda row: row["POS"])
            row_names = [row.pop("_r_row_name") for row in ordered]
            tables.append(TraitRows(ordered, row_names=row_names, factor_levels=levels))
        return tables
