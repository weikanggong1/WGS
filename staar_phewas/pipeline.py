"""Native GDS -> Gaussian PheWAS association pipeline.

SPDX-License-Identifier: GPL-3.0-only
Analysis rules follow the frozen STAARpipelinePheWAS 0.9.7.1 source.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import math
from typing import Mapping, Sequence
import numpy as np
import torch

from .gds import SeqArrayGDS
from .masks import (VariantAnnotations, variant_filter, coding_masks, noncoding_masks,
                    ncRNA_mask, annotation_phred_matrix, sample_union, sample_indices,
                    gene_assignments, NONCODING_CATEGORIES, _strings, _chromosome)
from .annotation_index import CandidateAnnotationIndex
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
    wrapper_semantics: str = "phewas"

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
        if self.wrapper_semantics not in ("phewas", "base"):
            raise ValueError("wrapper_semantics must be phewas or base")


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
        if self.options.wrapper_semantics == "base" and len(self.models) != 1:
            raise ValueError("base wrapper semantics requires exactly one null model")
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
        self._position_sorted = bool(np.all(self.position[1:] >= self.position[:-1]))
        self._base_masks = {}
        self._category_codes = None
        self._annotation_indexes = {}
        self._coding_masks_cache = {}
        self._test_set_cache = OrderedDict()
        self.batch_diagnostics = []
        self.statistics_execution = "serial"

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
        if self._position_sorted:
            left = np.searchsorted(self.position, start, side="left")
            right = np.searchsorted(self.position, end, side="right")
            return np.arange(left, right, dtype=np.int64)
        return np.flatnonzero((self.position >= start) & (self.position <= end))

    def annotations(self, indices, *, mask_fields=(), include_weights=True, metadata="full"):
        indices = np.asarray(indices, dtype=np.int64)
        names = list(dict.fromkeys(list(mask_fields) + (self.annotation_names if include_weights else [])))
        annotations = {}
        for name in names:
            if name in self.annotation_catalog:
                annotations[name] = self.gds.read_field(self.annotation_catalog[name], indices)
            elif name in mask_fields:
                raise ValueError(f"required mask annotation missing from catalog: {name}")
        if metadata not in ("full", "mask", "weights"):
            raise ValueError("metadata must be full, mask, or weights")
        ref, alt = self.gds.read_ref_alt(indices) if metadata != "weights" else (None, None)
        return VariantAnnotations(
            self.position[indices], self.qc[indices], annotations,
            ref=ref, alt=alt,
            chromosome=self.gds.read_field("chromosome", indices) if metadata != "weights" else None,
            variant_id=self.gds.read_field("variant.id", indices) if metadata == "full" else None,
        )

    def _select_mask_chunks(self,indices,mask_fields,selector):
        """Read candidate metadata in order; load PHRED only for selected sites."""
        total=self.gds.n_variants if indices is None else len(indices)
        selected={}
        for offset in range(0,total,self.options.annotation_block_size):
            end=min(offset+self.options.annotation_block_size,total)
            rows=np.arange(offset,end,dtype=np.int64) if indices is None else indices[offset:end]
            annotations=self.annotations(rows,mask_fields=mask_fields,include_weights=False,metadata="mask")
            for category,local in selector(annotations).items():
                selected.setdefault(category,[])
                if len(local):selected[category].append(rows[local])
        return {category:np.concatenate(chunks) if chunks else np.empty(0,dtype=np.int64)
                for category,chunks in selected.items()}

    def _field(self, name, rows):
        if name not in self.annotation_catalog:
            raise ValueError(f"required mask annotation missing from catalog: {name}")
        return _strings(self.gds.read_field(self.annotation_catalog[name], rows))

    def _base_mask(self, chromosome, variant_type):
        key = (_chromosome(chromosome), variant_type)
        if key not in self._base_masks:
            output = np.zeros(self.gds.n_variants, dtype=bool)
            for offset in range(0, self.gds.n_variants, self.options.annotation_block_size):
                rows = np.arange(offset, min(offset + self.options.annotation_block_size, self.gds.n_variants))
                ref, alt = self.gds.read_ref_alt(rows) if variant_type != "variant" else (None, None)
                a = VariantAnnotations(self.position[rows], self.qc[rows], {}, ref=ref, alt=alt)
                on_chromosome = np.fromiter((_chromosome(x) == key[0] for x in self.gds.read_field("chromosome", rows)),
                                           dtype=bool, count=len(rows))
                output[rows] = variant_filter(a, variant_type) & on_chromosome
            self._base_masks[key] = output
        return self._base_masks[key]

    def _annotation_categories(self):
        categories = ("upstream", "downstream", "UTR3", "UTR5", "UTR5;UTR3",
                      "ncRNA_exonic", "ncRNA_exonic;splicing", "ncRNA_splicing")
        if self._category_codes is None:
            self._category_codes = np.zeros(self.gds.n_variants, dtype=np.uint8)
            for offset in range(0, self.gds.n_variants, self.options.annotation_block_size):
                rows = np.arange(offset, min(offset + self.options.annotation_block_size, self.gds.n_variants))
                values = self._field("GENCODE.Category", rows)
                for code, category in enumerate(categories, 1):
                    self._category_codes[rows[values == category]] = code
        return self._category_codes

    def prepare_annotation_index(self, chromosome, *, promoter_intervals=None,
                                 include_ncrna=True, categories=None):
        """Parse each requested gene-assignment category once per chromosome.

        Candidate counts exclude QC/type failures but precede MAF filtering.
        Only category candidates receive expensive GENCODE.Info parsing.
        """
        requested = list(NONCODING_CATEGORIES if categories is None else categories)
        if include_ncrna and "ncRNA" not in requested:
            requested.append("ncRNA")
        if any(name not in (*NONCODING_CATEGORIES, "ncRNA") for name in requested):
            raise ValueError("unknown indexed annotation category")
        key = (_chromosome(chromosome), self.options.variant_type)
        index = self._annotation_indexes.setdefault(key, CandidateAnnotationIndex(*key))
        needed = [name for name in requested if name not in index.prepared_categories]
        promoters = [name for name in requested if name.startswith("promoter_")]
        promoter_mask = None
        if promoters:
            if promoter_intervals is None:
                raise ValueError("exact promoter intervals are required for promoter categories")
            signature = tuple(sorted((int(start), int(end)) for c, start, end in promoter_intervals
                                     if _chromosome(c) == key[0]))
            if any(start > end for start, end in signature):
                raise ValueError("promoter interval start exceeds end")
            if index.promoter_signature is not None and index.promoter_signature != signature:
                raise ValueError("one annotation index cannot mix different promoter references")
            index.promoter_signature = signature
            if any(name in needed for name in promoters):
                promoter_mask = np.zeros(self.gds.n_variants, dtype=bool)
                if signature:
                    starts = np.asarray([item[0] for item in signature])
                    ends = np.maximum.accumulate(np.asarray([item[1] for item in signature]))
                    previous = np.searchsorted(starts, self.position, side="right") - 1
                    valid = previous >= 0
                    promoter_mask[valid] = self.position[valid] <= ends[previous[valid]]
        if not needed:
            return index
        base = self._base_mask(chromosome, self.options.variant_type)
        class_codes = self._annotation_categories() if any(name in ("upstream", "downstream", "UTR", "ncRNA") for name in needed) else None
        chunks = {}
        code_groups = {"upstream": (1,), "downstream": (2,), "UTR": (3, 4, 5), "ncRNA": (6, 7, 8)}
        for offset in range(0, self.gds.n_variants, self.options.annotation_block_size):
            stop = min(offset + self.options.annotation_block_size, self.gds.n_variants)
            rows = offset + np.flatnonzero(base[offset:stop])
            if not len(rows):
                continue
            signals = {}
            for signal in ("CAGE", "DHS"):
                if any(name.endswith("_" + signal) for name in needed):
                    signals[signal] = self._field(signal, rows) != ""
            info_masks = {}
            for category in needed:
                if category in code_groups:
                    info_masks[category] = np.isin(class_codes[rows], code_groups[category])
                elif category.startswith("promoter_"):
                    info_masks[category] = promoter_mask[rows] & signals[category.split("_")[1]]
            if info_masks:
                local = np.flatnonzero(np.logical_or.reduce(list(info_masks.values())))
                if len(local):
                    info = self._field("GENCODE.Info", rows[local])
                    lookup = np.full(len(rows), -1, dtype=np.int64); lookup[local] = np.arange(len(local))
                    for category, accepted in info_masks.items():
                        selected = np.flatnonzero(accepted)
                        index.add_assignments(chunks, category, rows[selected],
                                              gene_assignments(category, info[lookup[selected]]))
            enhancer_names = [name for name in needed if name.startswith("enhancer_")]
            if enhancer_names:
                local = np.flatnonzero(np.logical_or.reduce([signals[name.split("_")[1]] for name in enhancer_names]))
                if len(local):
                    values = self._field("GeneHancer", rows[local])
                    assignments = gene_assignments("enhancer", values)
                    for category in enhancer_names:
                        selected = np.flatnonzero(signals[category.split("_")[1]][local])
                        index.add_assignments(chunks, category, rows[local[selected]], [assignments[j] for j in selected])
        index.finish(chunks, needed)
        return index

    def annotation_gene_manifest(self, chromosome, **kwargs):
        return self.prepare_annotation_index(chromosome, **kwargs).manifest()

    def _limit(self, model, number_variants):
        # Dense G, rotated G, weighted G, plus covariance/eigensolver workspace.
        t=model.n_pheno
        estimate = 8 * (4 * model.n * number_variants + 6 * (t*number_variants) ** 2 + model.n*t*t)
        if estimate + getattr(self,"_batch_workspace_reserve",0) > self.options.memory_limit_gib * 2**30:
            raise MemoryError("variant set exceeds the configured memory budget; increase the budget or analyze smaller genomic sets")

    def _prepare_test_set(self, indices, annotations=None):
        """Return one STAAR result per model, None for insufficient rare variants.

        Union MAF prefilter (<0.05 when cutoff<=0.01, otherwise <1) precedes
        per-trait MAF; allele orientation is never flipped again per trait.
        Exceptions other than insufficient variant count propagate to callers.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if annotations is None:
            annotations = self.annotations(indices, metadata="weights")
        phred, names = annotation_phred_matrix(annotations.annotations, self.annotation_names,
                                               variant_type=self.options.variant_type,
                                               number_variants=len(indices))
        blocks = []
        columns = []
        prefilter = (self.options.rare_maf_cutoff if self.options.wrapper_semantics == "base" else
                     0.05 if self.options.rare_maf_cutoff <= 0.01 else 1.0)
        for offset, block in enumerate(self.gds.iter_minor_blocks(indices, self.union_rows, block_size=self.options.genotype_block_size)):
            union_alt_af = 1 - block.union_ref_af
            union_maf = np.where(block.union_ref_af >= union_alt_af, union_alt_af, block.union_ref_af)
            keep = np.isfinite(union_maf) & (union_maf > 0) & (union_maf < prefilter)
            if not keep.any():
                continue
            start = offset * self.options.genotype_block_size
            blocks.append((block, keep))
            columns.extend((start + np.flatnonzero(keep)).tolist())
        if len(columns) >= self.options.rv_num_cutoff_max_prefilter:
            raise ValueError("union-prefilter variant count reaches rv_num_cutoff_max_prefilter")
        results = []
        for model, rows in zip(self.models, self.trait_rows):
            pieces, frequencies, extraction_groups, source_frequencies = [], [], [], []
            for block, keep in blocks:
                frequency = {"frequency_mode": "reference"} if self.options.wrapper_semantics == "base" else {}
                g, maf, _, missing, _ = block.trait_dense(rows, self.options.imputation, **frequency)
                pieces.append(g[:, keep]); frequencies.append(maf[keep])
                if self.options.wrapper_semantics == "base":
                    original_alt_af = 1 - block.union_ref_af
                    original_maf = np.where(block.union_ref_af >= original_alt_af, original_alt_af, block.union_ref_af)
                    source_frequencies.append(original_maf[keep])
                    allele_missing = block.allele_missing_rate()
                    # The original callers leave the extractor's default 0.01
                    # MAF and missing-rate group thresholds unchanged.
                    groups = np.where(original_alt_af > 0.5, 2,
                        np.where((original_maf >= 0.01) | (allele_missing >= 0.01), 1, 0))
                    extraction_groups.append(groups[keep])
            if not pieces:
                results.append(None); continue
            maf = np.concatenate(frequencies)
            rare = np.isfinite(maf) & (maf > 0) & (maf < self.options.rare_maf_cutoff)
            if self.options.wrapper_semantics == "base":
                source_maf = np.concatenate(source_frequencies)
                rare &= np.isfinite(source_maf) & (source_maf > 0) & (source_maf < self.options.rare_maf_cutoff)
            count = int(rare.sum())
            if count < self.options.rv_num_cutoff:
                results.append(None); continue
            if count >= self.options.rv_num_cutoff_max:
                raise ValueError("rare variant count reaches rv_num_cutoff_max")
            self._limit(model, count)
            g = np.concatenate(pieces, axis=1)[:, rare]
            selected_maf = maf[rare]
            annotation = phred[np.asarray(columns)[rare]]
            if self.options.wrapper_semantics == "base":
                order = np.argsort(np.concatenate(extraction_groups)[rare], kind="stable")
                g = g[:, order]
                selected_maf = selected_maf[order]
                annotation = annotation[order]
            reduction_options = {}
            if (self.options.wrapper_semantics == "base" and isinstance(model, GaussianNullModel)
                    and model.has_kinship and not model.spectrum.blocks):
                estimate = 8 * (4 * model.n * count + 6 * count**2 + model.n)
                available = int(self.options.memory_limit_gib * 2**30) - estimate - getattr(self, "_batch_workspace_reserve", 0)
                workspace = min(256 * 2**20, available)
                if workspace <= 0:
                    raise MemoryError("no workspace remains for reference sparse score reduction")
                reduction_options = {"reduction": "reference_sparse", "max_workspace_bytes": workspace}
            u, v = model.score_covariance(g, **reduction_options)
            cutoffs=dict(rare_maf_cutoff=self.options.rare_maf_cutoff,rv_num_cutoff=self.options.rv_num_cutoff,
                         rv_num_cutoff_max=self.options.rv_num_cutoff_max)
            payload = dict(score=u, covariance=v, maf=selected_maf, mac=np.rint(selected_maf*2*model.n),
                           annotations=annotation, names=names, acat_calibration="chi2",
                           cmac=float(g.sum()), **cutoffs)
            if model.use_spa:
                payload["_genotype"] = g
            results.append(payload)
        return results

    def _evaluate_prepared(self, payload, model):
        if payload is None:
            return None
        if model.use_spa:
            cutoffs={name:payload[name] for name in ("rare_maf_cutoff", "rv_num_cutoff", "rv_num_cutoff_max")}
            return staar_binary_spa(torch.as_tensor(payload["_genotype"], dtype=torch.float64, device=model.device),
                payload["maf"], model.scaled_residuals, model.fitted_probability, model.xw, model.projection_left,
                payload["annotations"], payload["names"], spa_p_filter=self.options.spa_p_filter,
                p_filter_cutoff=self.options.p_filter_cutoff, tol=self.options.spa_tol,
                max_iter=self.options.spa_max_iter, covariance=payload["covariance"], **cutoffs)
        return (multi_staar_test if model.n_pheno>1 else staar_test)(**payload)

    def _set_key(self, indices):
        rows=np.asarray(indices,dtype=np.int64)
        return (len(rows), hashlib.blake2b(rows.tobytes(), digest_size=16).digest())

    def _cache_set(self, key, results):
        self._test_set_cache[key] = [None if result is None else dict(result) for result in results]
        self._test_set_cache.move_to_end(key)
        while len(self._test_set_cache)>256:
            self._test_set_cache.popitem(last=False)

    def test_set(self, indices, annotations=None):
        key=self._set_key(indices)
        if annotations is None and key in self._test_set_cache:
            self._test_set_cache.move_to_end(key)
            return [None if value is None else dict(value) for value in self._test_set_cache[key]]
        results=[self._evaluate_prepared(payload,model) for payload,model in
                 zip(self._prepare_test_set(indices,annotations),self.models)]
        if annotations is None:
            self._cache_set(key,results)
        return results

    def test_sets_batch(self, index_sets, *, max_workspace_bytes=256*2**20):
        """Batch Gaussian mask kernels; preserve set/model order and None entries.

        Genotype blocks and MAF remain model-specific. Pending score/covariance
        storage is bounded and original result columns are retained.
        """
        if max_workspace_bytes<=0:
            raise ValueError("max_workspace_bytes must be positive")
        previous_reserve=getattr(self,"_batch_workspace_reserve",0)
        # Pending covariance, prepared copies and solver workspace are separate
        # allocations. Keep additional margin for the model's resident state.
        self._batch_workspace_reserve=4*max_workspace_bytes
        try:
            return self._test_sets_batch(index_sets,max_workspace_bytes=max_workspace_bytes)
        finally:
            self._batch_workspace_reserve=previous_reserve

    def _test_sets_batch(self,index_sets,*,max_workspace_bytes):
        from .batch_statistics import staar_test_batch
        index_sets=list(index_sets)
        output=[[None]*len(self.models) for _ in index_sets]
        pending=[]; pending_bytes=0
        keys = [self._set_key(indices) for indices in index_sets]
        representatives = {}
        aliases = {}
        def flush():
            if not pending:
                return
            results,diagnostics=staar_test_batch([item[2] for item in pending],
                max_workspace_bytes=max_workspace_bytes,return_diagnostics=True)
            self.batch_diagnostics.append(diagnostics)
            for (set_index,trait,_),result in zip(pending,results):
                output[set_index][trait]=result
            pending.clear()
        for set_index,indices in enumerate(index_sets):
            key = keys[set_index]
            if key in representatives:
                aliases[set_index] = representatives[key]
                continue
            representatives[key] = set_index
            if key in self._test_set_cache:
                output[set_index]=[None if value is None else dict(value) for value in self._test_set_cache[key]]
                continue
            prepared=self._prepare_test_set(indices)
            for trait,(model,payload) in enumerate(zip(self.models,prepared)):
                if payload is None:
                    continue
                if model.n_pheno!=1 or model.use_spa:
                    output[set_index][trait]=self._evaluate_prepared(payload,model)
                    continue
                size=8*(payload["score"].numel()+payload["covariance"].numel())
                if pending and pending_bytes+size>max_workspace_bytes:
                    flush();pending_bytes=0
                pending.append((set_index,trait,payload));pending_bytes+=size
        flush()
        for alias, representative in aliases.items():
            output[alias] = [None if value is None else dict(value) for value in output[representative]]
        for key,results in zip(keys,output):
            self._cache_set(key,results)
        return output

    def _run_mask_sets(self, index_sets):
        if self.statistics_execution in ("batch", "batched"):
            return self.test_sets_batch(index_sets)
        if self.statistics_execution != "serial":
            raise ValueError("statistics_execution must be serial or batched")
        return [self.test_set(indices) for indices in index_sets]

    def coding(self, chromosome, gene_name, start, end, *, category="all_categories", include_ptv=False):
        if category == "all_categories_incl_ptv":
            category, include_ptv = "all_categories", True
        cache_key=(_chromosome(chromosome),gene_name,int(start),int(end),bool(include_ptv))
        if cache_key not in self._coding_masks_cache:
            indices = self.region_indices(start, end)
            self._coding_masks_cache[cache_key]=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.EXONIC.Category","MetaSVM"),
                lambda a:coding_masks(a,gene_name,start,end,variant_type=self.options.variant_type,
                                      chromosome=chromosome,include_ptv=include_ptv))
        masks=self._coding_masks_cache[cache_key]
        records = [[] for _ in self.models]
        selected_masks=[(mask,selected) for mask,selected in masks.items() if len(selected) and
                        (category=="all_categories" or mask in (category,"disruptive_missense" if category=="missense" else category))]
        for (mask,selected),results in zip(selected_masks,self._run_mask_sets([rows for _,rows in selected_masks])):
            for trait,stats in enumerate(results):
                if stats is not None:records[trait].append(coding_record(chromosome,gene_name,mask,stats))
        return self._assemble(records, kind="coding", category=category, include_ptv=include_ptv)

    def noncoding(self, chromosome, gene_name, start=None, end=None, *, category="all_categories", promoter_intervals=None, include_ncrna=False):
        if start is None and end is None:
            categories=list(NONCODING_CATEGORIES) if category=="all_categories" else [category]
            index=self.prepare_annotation_index(chromosome,promoter_intervals=promoter_intervals,
                                               include_ncrna=include_ncrna,categories=categories)
            masks={name:index.indices(gene_name,name) for name in categories+(["ncRNA"] if include_ncrna else [])}
        else:
            return self._noncoding_region(chromosome,gene_name,start,end,category=category,
                                          promoter_intervals=promoter_intervals,include_ncrna=include_ncrna)
        return self._noncoding_results(chromosome,gene_name,masks,category,include_ncrna)

    def _noncoding_region(self, chromosome, gene_name, start, end, *, category, promoter_intervals, include_ncrna):
        indices=self.region_indices(start,end)
        def selector(a):
            overlaps=None
            if promoter_intervals is not None:
                from .masks import promoter_overlaps
                overlaps=promoter_overlaps(a.position,a.chromosome,promoter_intervals)
            return noncoding_masks(a,gene_name,promoter_overlap=overlaps,variant_type=self.options.variant_type,
                                   chromosome=chromosome,include_ncrna=include_ncrna,
                                   requested_categories=None if category=="all_categories" else [category])
        masks=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.Info","CAGE","DHS","GeneHancer"),selector)
        return self._noncoding_results(chromosome,gene_name,masks,category,include_ncrna)

    def _noncoding_results(self,chromosome,gene_name,masks,category,include_ncrna):
        records=[[] for _ in self.models]
        selected_masks=[(mask,rows) for mask,rows in masks.items() if len(rows) and
                        (category=="all_categories" or mask==category)]
        for (mask,selected),results in zip(selected_masks,self._run_mask_sets([rows for _,rows in selected_masks])):
            for trait,stats in enumerate(results):
                if stats is not None:records[trait].append(coding_record(chromosome,gene_name,mask,stats))
        return self._assemble(records, kind="noncoding", category=category, include_ncrna=include_ncrna)

    def ncrna(self, chromosome, gene_name, start=None, end=None):
        if start is None and end is None:
            selected=self.prepare_annotation_index(chromosome,categories=["ncRNA"],include_ncrna=False).indices(gene_name,"ncRNA")
        else:
            indices=self.region_indices(start,end)
            masks=self._select_mask_chunks(indices,("GENCODE.Category","GENCODE.Info"),
                lambda a:{"ncRNA":ncRNA_mask(a,gene_name,variant_type=self.options.variant_type,chromosome=chromosome)})
            selected=masks.get("ncRNA",np.empty(0,dtype=np.int64))
        return [[coding_record(chromosome,gene_name,"ncRNA",stats)] if stats is not None else []
                for stats in self._run_mask_sets([selected])[0]]

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
        selected_windows=[]
        for left, right in windows:
            indices = self.region_indices(left, right)
            a = self.annotations(indices,include_weights=False,metadata="mask")
            local = np.flatnonzero(variant_filter(a, self.options.variant_type))
            selected_windows.append(indices[local])
        for (left,right),results in zip(windows,self._run_mask_sets(selected_windows)):
            for trait, stats in enumerate(results):
                if stats is not None:
                    records[trait].append(window_record(chromosome, left, right, stats))
        return self._assemble(records, kind="sliding")

    def iter_individual_records(self, chromosome, start=None, end=None, *, mac_cutoff=20,
                                variant_type="variant", subset_variants_num=5000):
        """Yield (model index, bounded record block) in GDS variant order.

        Omit both endpoints for the complete chromosome. Internal grouping
        markers retain the original R factor-level and row-name convention.
        Consumers can write each block immediately without retaining all rows.
        """
        if (start is None)!=(end is None):
            raise ValueError("provide both region endpoints or omit both")
        genotype_device = next((model.device for model in self.models
                                if str(getattr(model, "device", "cpu")).startswith("cuda")), None)
        reader_options = {"device": genotype_device, "minimum_mac": mac_cutoff} if genotype_device is not None else {}
        if start is None:
            base=self._base_mask(chromosome,variant_type)
            def blocks():
                for offset in range(0,self.gds.n_variants,self.options.annotation_block_size):
                    stop=min(offset+self.options.annotation_block_size,self.gds.n_variants)
                    indices=offset+np.flatnonzero(base[offset:stop])
                    yield from self.gds.iter_minor_blocks(indices,self.union_rows,block_size=self.options.genotype_block_size, **reader_options)
            genotype_blocks=blocks()
        else:
            indices = self.region_indices(start, end)
            a = self.annotations(indices,include_weights=False,metadata="mask")
            indices = indices[np.flatnonzero(variant_filter(a, variant_type))]
            genotype_blocks=self.gds.iter_minor_blocks(indices,self.union_rows,block_size=self.options.genotype_block_size, **reader_options)
        if subset_variants_num < 1:
            raise ValueError("subset_variants_num must be positive")
        union_ordinal = 0
        for block in genotype_blocks:
            keep_union = block.initial_mac() >= mac_cutoff
            ordinals = union_ordinal + np.cumsum(keep_union) - 1
            union_ordinal += int(keep_union.sum())
            union_columns = np.flatnonzero(keep_union)
            if len(union_columns) == 0:
                continue
            block = block.select_columns(union_columns)
            ordinals = ordinals[union_columns]
            source_alt_af = 1 - block.union_ref_af
            source_maf = np.where(block.union_ref_af >= source_alt_af, source_alt_af, block.union_ref_af)
            extractable = np.isfinite(source_maf) & (source_maf > 0) & (source_maf < 1)
            for trait, (model, rows) in enumerate(zip(self.models, self.trait_rows)):
                base_mode = self.options.wrapper_semantics == "base"
                eligible = extractable.copy()
                if not base_mode:
                    eligible &= block.observed_mac(rows) >= mac_cutoff
                trait_columns = np.flatnonzero(eligible)
                if len(trait_columns) == 0:
                    continue
                trait_block = block.select_columns(trait_columns)
                trait_ordinals = ordinals[trait_columns]
                frequency = {"frequency_mode": "reference"} if base_mode else {}
                g, maf, mac, missing, is_alt = trait_block.trait_dense(rows, self.options.imputation, **frequency)
                if base_mode:
                    original_alt_af = 1 - trait_block.union_ref_af
                    original_maf = np.where(trait_block.union_ref_af >= original_alt_af, original_alt_af, trait_block.union_ref_af)
                    allele_missing = trait_block.allele_missing_rate()
                    base_group = np.where(original_alt_af > 0.5, 2,
                        np.where((original_maf >= 0.01) | (allele_missing >= 0.01), 1, 0))
                # Retain the existing F-contiguous boolean-column copy before
                # invoking the unchanged association kernel.
                keep = np.ones(len(trait_columns), dtype=bool)
                columns = np.arange(len(trait_columns))
                selected = trait_block.variant_indices
                trait_records=[]
                if model.n_pheno==1 and hasattr(model,"individual_score_variance"):
                    u,variance=model.individual_score_variance(g[:,keep])
                else:
                    u,v=model.score_covariance(g[:,keep])
                    if model.n_pheno==1:variance=v.diagonal()
                if model.n_pheno>1:
                    scores=u.reshape(model.n_pheno,len(columns))
                    cov4=v.reshape(model.n_pheno,len(columns),model.n_pheno,len(columns))
                    log_probabilities=torch.stack([joint_individual_logp(scores[:,j],cov4[:,j,:,j]) for j in range(len(columns))])
                else:
                    positive=variance>0
                    standard_error=torch.sqrt(variance)
                    z=torch.where(positive,u/torch.clamp(standard_error,min=1e-300),0.)
                    log_probabilities=-math.log(2)-torch.special.log_ndtr(-z.abs())
                    log_probabilities=torch.where(torch.isnan(variance),variance,torch.where(positive,log_probabilities,0.))
                    if model.use_spa:
                        probabilities=individual_score_test_spa(torch.as_tensor(g[:,keep],dtype=torch.float64,device=model.device),
                            model.scaled_residuals,model.fitted_probability,model.xw,model.projection_left,
                            normal_pvalues=torch.exp(-log_probabilities) if self.options.spa_p_filter else None,
                            p_filter_cutoff=self.options.p_filter_cutoff,tol=self.options.spa_tol,max_iter=self.options.spa_max_iter)
                # Transfer one result block, avoiding four CUDA synchronizations
                # per variant. float64 values and row assembly are unchanged.
                if model.n_pheno>1:
                    values_cpu=torch.cat((scores,log_probabilities[None,:]),dim=0).detach().cpu().numpy()
                elif model.use_spa:
                    values_cpu=probabilities.detach().cpu().numpy()
                else:
                    values_cpu=torch.stack((variance,u,standard_error,log_probabilities),dim=1).detach().cpu().numpy()
                chrom = self.gds.read_field("chromosome", selected)
                ref,alt=self.gds.read_ref_alt(selected)
                for j, column in enumerate(columns):
                    if model.n_pheno>1:
                        statistic={"pvalue_log":float(values_cpu[-1,j]),"Score":values_cpu[:-1,j].tolist()}
                    elif model.use_spa:
                        statistic={"pvalue":float(values_cpu[j])}
                    else:
                        value,score,se,logp=map(float,values_cpu[j])
                        statistic={"pvalue_log":logp,"Score":score,"Score_se":se,
                                   "Est":0. if value==0 else score/value,"Est_se":0. if se==0 else 1/se}
                    row = single_variant_record(chrom[j], self.position[selected[j]], ref[j], alt[j],
                                                original_alt_af[column] if base_mode else (maf[column] if is_alt[column] else 1-maf[column]), maf[column], model.n,
                                                statistic,number_phenotypes=model.n_pheno,use_spa=model.use_spa)
                    # R appends common/high-missing groups before rare groups,
                    # then orders by POS without renumbering data.frame rows.
                    row["_chunk"] = int(trait_ordinals[column] // subset_variants_num)
                    row["_common"] = bool(maf[column] >= 0.01 or
                                          (allele_missing[column] if base_mode else missing[column] / model.n) >= 0.01)
                    if base_mode:
                        row["_base_group"] = int(base_group[column])
                    trait_records.append(row)
                yield trait,trait_records

    def individual(self, chromosome, start=None, end=None, *, mac_cutoff=20, variant_type="variant", subset_variants_num=5000):
        """Collect the streaming kernel and preserve the original R table metadata."""
        records=[[] for _ in self.models]
        for trait,rows in self.iter_individual_records(chromosome,start,end,mac_cutoff=mac_cutoff,
                variant_type=variant_type,subset_variants_num=subset_variants_num):
            records[trait].extend(rows)
        return self.individual_tables(records)

    @staticmethod
    def individual_tables(records):
        """Finalize original grouping markers, factor levels and position ordering."""
        tables = []
        for rows in records:
            levels = {"REF": [], "ALT": []}
            seen_levels = {"REF": set(), "ALT": set()}
            chunks = {}
            for row in rows:
                # Base extraction prepends ALT sparse, ALT dense, then REF
                # dosage columns. PheWAS has no extraction-group marker.
                common, rare = chunks.setdefault(row["_chunk"],
                                                 ([[], [], []], [[], [], []]))
                (common if row["_common"] else rare)[row.get("_base_group", 0)].append(row)
            ordered = []
            for chunk in sorted(chunks):
                common_parts, rare_parts = chunks[chunk]
                common = [row for part in common_parts for row in part]
                rare = [row for part in rare_parts for row in part]
                groups = [common[i:i+200] for i in range(0, len(common), 200)] + ([rare] if rare else [])
                for group in groups:
                    for field in levels:
                        for value in sorted({row[field] for row in group}):
                            if value not in seen_levels[field]:
                                levels[field].append(value)
                                seen_levels[field].add(value)
                    ordered.extend(group)
            for index, row in enumerate(ordered, 1):
                row.pop("_chunk"); row.pop("_common")
                row.pop("_base_group", None)
                row["_r_row_name"] = index
            ordered.sort(key=lambda row: row["POS"])
            row_names = [row.pop("_r_row_name") for row in ordered]
            tables.append(TraitRows(ordered, row_names=row_names, factor_levels=levels))
        return tables
