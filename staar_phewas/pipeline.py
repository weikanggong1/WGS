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

from .gds import SeqArrayGDS, SparseMinorBlock
from .gds_device import DeviceMinorBlock
from .profiling import StageProfiler
from .masks import (VariantAnnotations, variant_filter, coding_masks, noncoding_masks,
                    ncRNA_mask, annotation_phred_matrix, sample_union, sample_indices,
                    gene_assignments, NONCODING_CATEGORIES, _strings, _chromosome)
from .annotation_index import CandidateAnnotationIndex
from .null_model import GaussianNullModel
from .statistics import staar_test
from .multi import multi_staar_test, joint_individual_logp
from .binary import staar_binary_spa, individual_score_test_spa
from .results import coding_record, single_variant_record, assemble_phewas_results, TraitRows


def _individual_log_probabilities(score, variance):
    """Evaluate the original normal tail from already-computed core vectors."""
    score, variance = score.to(torch.float64), variance.to(torch.float64)
    positive = variance > 0
    standard_error = torch.sqrt(variance)
    z = torch.where(positive, score / torch.clamp(
        standard_error, min=torch.finfo(standard_error.dtype).tiny), 0.)
    log_probabilities = -math.log(2) - torch.special.log_ndtr(-z.abs())
    return torch.where(torch.isnan(variance), variance,
                       torch.where(positive, log_probabilities, 0.))


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
        self.statistics_tail_optimization = False
        self.weight_batch_optimization = False
        self.local_mask_reuse = False
        self.local_mask_reuse_counters = {"families": 0, "union_score_calls": 0,
            "reused_masks": 0, "input_variant_columns": 0, "union_variant_columns": 0, "prepared_union_variant_columns": 0,
            "fallback_memory": 0, "fallback_unsupported": 0, "fallback_mapping": 0,
            "cache_hits": 0, "duplicate_mask_hits": 0, "single_mask_paths": 0,
            "fallback_cuda_oom": 0, "fallback_geometry": 0, "fallback_host_reused_masks": 0, "geometry_checks": 0,
            "geometry_union_covariance_cells": 0, "geometry_mask_covariance_cells": 0}
        self.statistics_tail_optimization_calls = 0
        self.profiler = StageProfiler(getattr(self.models[0], "device", "cpu"), enabled=False)
        self.resident_genotypes = False

    @property
    def profiler(self):
        if not hasattr(self, "_profiler"):
            self._profiler = StageProfiler(enabled=False)
        return self._profiler

    @profiler.setter
    def profiler(self, value):
        self._profiler = value

    def _minor_blocks(self, *args, **kwargs):
        """Time iterator work, excluding the caller's association computation."""
        iterator = iter(self.gds.iter_minor_blocks(*args, **kwargs))
        while True:
            with self.profiler.measure("gds_sdk_decode_prepare", gpu=bool(kwargs.get("device"))):
                try:
                    block = next(iterator)
                except StopIteration:
                    return
            yield block

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

    def _workspace_estimate(self, model, number_variants, *, individual=False):
        """Conservative live core buffers; Single has no M-by-M term.

        Native TF32 uses FP32 inputs/outputs and no reconstruction planes.
        Raw SDK slabs and already resident model tensors are independent.
        """
        n, m, t = model.n, int(number_variants), model.n_pheno
        native = getattr(model, "matmul_mode", "fp64") == "tf32"
        bytes_per_element = 4 if native else 8
        genotype_copies, covariance_copies = (4, 6)
        model_terms = n * t * t
        if individual:
            # Covariate projection and per-site scores/variance/tails need
            # linear storage, independently of a joint or Single model.
            model_terms += (getattr(getattr(model, "x", None), "shape", (n, 1))[1] + 8 * t) * m
            covariance_terms = 0
        else:
            covariance_terms = covariance_copies * (t * m) ** 2
        return bytes_per_element * (genotype_copies * n * m + covariance_terms + model_terms)

    def _limit(self, model, number_variants, *, individual=False):
        estimate = self._workspace_estimate(model, number_variants, individual=individual)
        mode = getattr(model, "matmul_mode", "fp64")
        resident_bytes = raw_bytes = 0
        if mode != "fp64":
            # Existing model/resident dosage are outside new product buffers.
            # Reading the allocator counter does not synchronize the stream.
            if str(getattr(model, "device", "cpu")).startswith("cuda"):
                resident_bytes = torch.cuda.memory_allocated(model.device)
            if individual:
                raw_bytes = int(getattr(getattr(self, "gds", None), "genotype_raw_memory_bytes", 0))
        required = estimate + resident_bytes + raw_bytes + getattr(self, "_batch_workspace_reserve", 0)
        self.memory_guard_max_estimated_bytes = max(getattr(self, "memory_guard_max_estimated_bytes", 0), required)
        if required > self.options.memory_limit_gib * 2**30:
            raise MemoryError(f"{mode} {'Single' if individual else 'mask'} workspace requires estimated {required} bytes, exceeding configured memory budget; precision and chunks were not changed")


    def _mask_union_geometry(self, physical, index_sets, model):
        """Retain the rare-cell gate; admit only bounded native tile savings."""
        from .local_mask_reuse import union_covariance_geometry, small_native_union_work
        geometry = union_covariance_geometry(physical, index_sets, minimum_variants=self.options.rv_num_cutoff)
        counters = self.local_mask_reuse_counters
        counters["geometry_checks"] += 1
        counters["geometry_union_covariance_cells"] += geometry["union_covariance_cells"]
        counters["geometry_mask_covariance_cells"] += geometry["mask_covariance_cells"]
        if (not geometry["beneficial"] and 2 <= len(physical) <= 64 and len(self.models) == 1
                and isinstance(model, GaussianNullModel) and model.n_pheno == 1 and not model.use_spa
                and model.matmul_mode == "tf32" and not model.spectrum.blocks and model.x.shape[1] >= 1):
            cost = small_native_union_work(physical, index_sets, minimum_variants=self.options.rv_num_cutoff,
                                          samples=model.n, covariates=model.x.shape[1])
            counters["geometry_small_tile_checks"] = counters.get("geometry_small_tile_checks", 0) + 1
            for name, value in cost["union"].items():
                for prefix, work in (("union", value["work"]), ("mask", cost["separate"][name])):
                    key = f"geometry_small_tile_{prefix}_{name}_work"
                    counters[key] = counters.get(key, 0) + work
            for prefix, calls in (("union", cost["union_calls"]), ("mask", cost["separate_calls"])):
                key = f"geometry_small_tile_{prefix}_logical_calls"
                counters[key] = counters.get(key, 0) + calls
            geometry["small_tile_work"] = cost
            if cost["beneficial"]:
                geometry["beneficial"] = True
                key = "geometry_small_tile_accepted"
                counters[key] = counters.get(key, 0) + 1
                if model.x.shape[1] == 1:
                    key = "geometry_small_tile_p1_accepted"
                    counters[key] = counters.get(key, 0) + 1
                    # Shape-only incremental outer output bytes, not workspace.
                    key = "geometry_small_tile_outer_extra_output_bytes"
                    counters[key] = counters.get(key, 0) + cost["outer_extra_output_bytes"]
        return geometry

    def _resident_gene_reader_options(self, indices):
        """Choose a bounded IO route before reading; raw count is storage only."""
        if (not getattr(self, "resident_genotypes", False) or len(self.models) != 1
                or not isinstance(self.models[0], GaussianNullModel)):
            return {}
        model = self.models[0]
        if (model.n_pheno != 1 or model.use_spa or model.matmul_mode != "tf32"
                or not str(model.device).startswith("cuda")
                or getattr(self.gds, "_flat_reader", None) is None):
            return {}
        n, b = len(self.union_rows), min(len(indices), self.options.genotype_block_size)
        storage = n * len(indices) + 64 * len(indices)
        # uint8 slabs plus int64 decoder operands/codes, masks, sample indexes.
        # The raw budget may span gaps/layers beyond this block's selected sites.
        reserve = 256 * 2**20
        scratch = 96 * n * b + 4 * self.gds.genotype_raw_memory_bytes + 8 * self.gds.n_samples
        allocated = torch.cuda.memory_allocated(model.device)
        unused = max(0, torch.cuda.memory_reserved(model.device) - allocated)
        free, _ = torch.cuda.mem_get_info(model.device)
        needed = storage + scratch + reserve
        counters = self.local_mask_reuse_counters
        key = "resident_gene_storage_reserve_bytes_max"
        counters[key] = max(counters.get(key, 0), needed)
        if (allocated + needed > self.options.memory_limit_gib * 2**30
                or needed > free + unused):
            key = "resident_gene_cpu_route_budget"
            counters[key] = counters.get(key, 0) + 1
            return {}
        key = "resident_gene_families"
        counters[key] = counters.get(key, 0) + 1
        return dict(device=model.device, resident=True)

    def _resident_gene_cmac(self, genotype):
        """Same imputed-G sum definition, using native FP32 device reduction."""
        counters = self.local_mask_reuse_counters
        key = "resident_gene_cmac_device_reductions"
        counters[key] = counters.get(key, 0) + 1
        return float(genotype.sum())

    def _materialize_resident_gene(self, prepared, columns=None):
        """Build only selected rare FP32 columns, never concatenate candidates."""
        model = self.models[0]
        mapping = prepared["_resident_mapping"]
        columns = np.arange(len(mapping)) if columns is None else np.asarray(columns, dtype=np.int64)
        self._limit(model, len(columns))
        # Original NumPy rare/column indexing yields sample-by-variant F
        # layout. Keep those strides for GEMV without allocating a clone.
        genotype = torch.empty((len(columns), model.n), dtype=model.x.dtype, device=model.device).T
        selected_mapping = mapping[columns]
        frequency = {"frequency_mode": "reference"} if self.options.wrapper_semantics == "base" else {}
        with self.profiler.measure("genotype_trait_prepare", gpu=True):
            for block_index, block in enumerate(prepared["_resident_blocks"]):
                destination = np.flatnonzero(selected_mapping[:, 0] == block_index)
                if not len(destination):
                    continue
                block_columns = selected_mapping[destination, 1]
                selected = block.select_columns(block_columns)
                dense = selected.trait_dense(self.trait_rows[0], self.options.imputation,
                                            dtype=model.x.dtype, **frequency)[0]
                genotype.index_copy_(1, torch.as_tensor(destination, device=model.device), dense)
                del dense, selected
        return genotype

    def _prepare_resident_gene(self, indices, annotations, reader_options, local_masks, defer_score):
        """Filter from small summaries before any floating genotype allocation."""
        model, rows = self.models[0], self.trait_rows[0]
        if annotations is None:
            annotations = self.annotations(indices, metadata="weights")
        phred, names = annotation_phred_matrix(annotations.annotations, self.annotation_names,
            variant_type=self.options.variant_type, number_variants=len(indices))
        prefilter = (self.options.rare_maf_cutoff if self.options.wrapper_semantics == "base" else
                     .05 if self.options.rare_maf_cutoff <= .01 else 1.)
        blocks, physical, sources, frequencies, groups, mapping = [], [], [], [], [], []
        frequency = {"frequency_mode": "reference"} if self.options.wrapper_semantics == "base" else {}
        for offset, block in enumerate(self._minor_blocks(indices, self.union_rows,
                block_size=self.options.genotype_block_size, **reader_options)):
            source_alt = 1 - block.union_ref_af
            source_maf = np.where(block.union_ref_af >= source_alt, source_alt, block.union_ref_af)
            keep = np.isfinite(source_maf) & (source_maf > 0) & (source_maf < prefilter)
            positions = np.flatnonzero(keep)
            if not len(positions):
                del block
                continue
            group = np.where(source_alt > .5, 2,
                np.where((source_maf >= .01) | (block.allele_missing_rate() >= .01), 1, 0))
            selected = block.select_columns(positions)
            del block
            with self.profiler.measure("genotype_trait_metadata", gpu=True):
                maf = selected.trait_summary(rows, self.options.imputation, **frequency)[0]
            physical.extend((offset * self.options.genotype_block_size + positions).tolist())
            sources.extend(source_maf[positions]); frequencies.extend(maf); groups.extend(group[positions])
            mapping.extend((len(blocks), column) for column in range(len(positions)))
            blocks.append(selected)
            del selected
        physical = np.asarray(physical, dtype=np.int64)
        mask_counts = ([len(physical)] if local_masks is None else
                       [int(np.isin(indices[physical], mask).sum()) for mask in local_masks])
        if any(count >= self.options.rv_num_cutoff_max_prefilter for count in mask_counts):
            raise ValueError("union-prefilter variant count reaches rv_num_cutoff_max_prefilter")
        maf = np.asarray(frequencies, dtype=np.float64)
        rare = np.isfinite(maf) & (maf > 0) & (maf < self.options.rare_maf_cutoff)
        if self.options.wrapper_semantics == "base":
            source = np.asarray(sources)
            rare &= np.isfinite(source) & (source > 0) & (source < self.options.rare_maf_cutoff)
        if local_masks is not None:
            included = np.zeros(len(rare), dtype=bool)
            for mask in local_masks:
                member = np.isin(indices[physical], mask)
                count = int((rare & member).sum())
                if count >= self.options.rv_num_cutoff_max:
                    raise ValueError("rare variant count reaches rv_num_cutoff_max")
                if count >= self.options.rv_num_cutoff:
                    included |= member
            rare &= included
        count = int(rare.sum())
        if count < self.options.rv_num_cutoff:
            return [None]
        if local_masks is None and count >= self.options.rv_num_cutoff_max:
            raise ValueError("rare variant count reaches rv_num_cutoff_max")
        mapping = np.asarray(mapping, dtype=np.int64).reshape(-1, 2)[rare]
        physical, maf = physical[rare], maf[rare]
        group = np.asarray(groups, dtype=np.int64)[rare] if self.options.wrapper_semantics == "base" else np.zeros(count, dtype=np.int64)
        order = np.argsort(group, kind="stable")
        mapping, physical, maf, group = mapping[order], physical[order], maf[order], group[order]
        # Compact to actual rare columns before retaining recovery state.
        for bi, block in enumerate(blocks):
            positions = np.flatnonzero(mapping[:, 0] == bi)
            original = mapping[positions, 1]
            blocks[bi] = block.select_columns(original)
            mapping[positions, 1] = np.arange(len(positions))
        del block
        geometry = None
        if local_masks is not None:
            geometry = self._mask_union_geometry(indices[physical], local_masks, model)
        prepared = dict(maf=maf, mac=np.rint(maf * 2 * model.n), annotations=phred[physical], names=names,
            acat_calibration="chi2", rare_maf_cutoff=self.options.rare_maf_cutoff,
            rv_num_cutoff=self.options.rv_num_cutoff, rv_num_cutoff_max=self.options.rv_num_cutoff_max,
            _variant_indices=indices[physical], _extraction_groups=group, _union_geometry=geometry,
            _resident_blocks=blocks, _resident_mapping=mapping)
        counters = self.local_mask_reuse_counters
        counters["resident_gene_prepared_columns"] = counters.get("resident_gene_prepared_columns", 0) + count
        if defer_score:
            return [prepared]
        genotype = self._materialize_resident_gene(prepared)
        payload = {key: value for key, value in prepared.items() if not key.startswith("_")}
        payload["cmac"] = self._resident_gene_cmac(genotype)
        with self.profiler.measure("score_covariance", gpu=True):
            payload["score"], payload["covariance"] = model.score_covariance(genotype)
        return [payload]

    def _prepare_test_set(self, indices, annotations=None, *, _local_masks=None, _defer_score=False):
        """Return one STAAR result per model, None for insufficient rare variants.

        Union MAF prefilter (<0.05 when cutoff<=0.01, otherwise <1) precedes
        per-trait MAF; allele orientation is never flipped again per trait.
        Exceptions other than insufficient variant count propagate to callers.
        """
        indices = np.asarray(indices, dtype=np.int64)
        reader_options = self._resident_gene_reader_options(indices)
        if reader_options:
            return self._prepare_resident_gene(indices, annotations, reader_options, _local_masks, _defer_score)
        if annotations is None:
            annotations = self.annotations(indices, metadata="weights")
        phred, names = annotation_phred_matrix(annotations.annotations, self.annotation_names,
                                               variant_type=self.options.variant_type,
                                               number_variants=len(indices))
        blocks = []
        columns = []
        prefilter = (self.options.rare_maf_cutoff if self.options.wrapper_semantics == "base" else
                     0.05 if self.options.rare_maf_cutoff <= 0.01 else 1.0)
        # Mask preparation retains bounded host blocks until the final rare
        # count passes the memory guard. Single scans use resident CUDA dosage.
        for offset, block in enumerate(self._minor_blocks(indices, self.union_rows, block_size=self.options.genotype_block_size)):
            union_alt_af = 1 - block.union_ref_af
            union_maf = np.where(block.union_ref_af >= union_alt_af, union_alt_af, block.union_ref_af)
            keep = np.isfinite(union_maf) & (union_maf > 0) & (union_maf < prefilter)
            if not keep.any():
                continue
            start = offset * self.options.genotype_block_size
            blocks.append((block, keep))
            columns.extend((start + np.flatnonzero(keep)).tolist())
        if _local_masks is None:
            prefilter_counts = [len(columns)]
        else:
            prefilter_indices = indices[np.asarray(columns, dtype=np.int64)]
            prefilter_counts = [int(np.isin(prefilter_indices, mask).sum()) for mask in _local_masks]
        if any(count >= self.options.rv_num_cutoff_max_prefilter for count in prefilter_counts):
            raise ValueError("union-prefilter variant count reaches rv_num_cutoff_max_prefilter")
        results = []
        for model, rows in zip(self.models, self.trait_rows):
            pieces, frequencies, extraction_groups, source_frequencies = [], [], [], []
            for block, keep in blocks:
                frequency = {"frequency_mode": "reference"} if self.options.wrapper_semantics == "base" else {}
                if getattr(model, "matmul_mode", "fp64") == "tf32" and isinstance(block, SparseMinorBlock):
                    frequency["dtype"] = np.float32
                with self.profiler.measure("genotype_trait_prepare", gpu=True):
                    g, maf, _, missing, _ = block.trait_dense(rows, self.options.imputation, **frequency)
                    if getattr(model, "matmul_mode", "fp64") == "tf32":
                        g = g.to(dtype=torch.float32) if isinstance(g, torch.Tensor) else g.astype(np.float32, copy=False)
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
            if _local_masks is not None:
                # Set limits apply to each original mask, never to their union.
                physical = indices[np.asarray(columns, dtype=np.int64)]
                included = np.zeros(len(rare), dtype=bool)
                for mask in _local_masks:
                    member = np.isin(physical, mask)
                    mask_count = int((rare & member).sum())
                    if mask_count >= self.options.rv_num_cutoff_max:
                        raise ValueError("rare variant count reaches rv_num_cutoff_max")
                    if mask_count >= self.options.rv_num_cutoff:
                        included |= member
                rare &= included
            count = int(rare.sum())
            if count < self.options.rv_num_cutoff:
                results.append(None); continue
            if _local_masks is None and count >= self.options.rv_num_cutoff_max:
                raise ValueError("rare variant count reaches rv_num_cutoff_max")
            if _local_masks is not None:
                from .local_mask_reuse import UnionGeometryRejected
                geometry = self._mask_union_geometry(indices[np.asarray(columns, dtype=np.int64)[rare]],
                                                     _local_masks, model)
                if not geometry["beneficial"] and not _defer_score:
                    raise UnionGeometryRejected("actual rare union covariance exceeds separate-mask cells")
            if not _defer_score:
                self._limit(model, count)
            g = (torch.cat(pieces, dim=1) if isinstance(pieces[0], torch.Tensor) else np.concatenate(pieces, axis=1))[:, rare]
            selected_maf = maf[rare]
            annotation = phred[np.asarray(columns)[rare]]
            physical_indices = indices[np.asarray(columns, dtype=np.int64)[rare]]
            selected_groups = np.zeros(count, dtype=np.int64)
            if self.options.wrapper_semantics == "base":
                selected_groups = np.concatenate(extraction_groups)[rare]
                order = np.argsort(selected_groups, kind="stable")
                g = g[:, order]
                selected_maf = selected_maf[order]
                annotation = annotation[order]
                physical_indices = physical_indices[order]
                selected_groups = selected_groups[order]
            cutoffs=dict(rare_maf_cutoff=self.options.rare_maf_cutoff,rv_num_cutoff=self.options.rv_num_cutoff,
                         rv_num_cutoff_max=self.options.rv_num_cutoff_max)
            if _defer_score:
                results.append(dict(maf=selected_maf, mac=np.rint(selected_maf*2*model.n),
                    annotations=annotation, names=names, acat_calibration="chi2", **cutoffs,
                    _genotype_host=g, _variant_indices=physical_indices,
                    _extraction_groups=selected_groups,
                    _union_geometry=geometry if _local_masks is not None else None))
                continue
            reduction_options = {}
            if (self.options.wrapper_semantics == "base" and isinstance(model, GaussianNullModel)
                    and model.has_kinship and not model.spectrum.blocks
                    and getattr(model, "matmul_mode", "fp64") == "fp64"):
                estimate = 8 * (4 * model.n * count + 6 * count**2 + model.n)
                available = int(self.options.memory_limit_gib * 2**30) - estimate - getattr(self, "_batch_workspace_reserve", 0)
                workspace = min(256 * 2**20, available)
                if workspace <= 0:
                    raise MemoryError("no workspace remains for reference sparse score reduction")
                reduction_options = {"reduction": "reference_sparse", "max_workspace_bytes": workspace}
            # Preserve the established host summation order for native CMAC.
            cmac = float(g.sum())
            if str(getattr(model, "device", "cpu")).startswith("cuda") or getattr(model, "matmul_mode", "fp64") == "tf32":
                with self.profiler.measure("genotype_h2d", gpu=True):
                    g = torch.as_tensor(g, dtype=model.x.dtype, device=model.device)
            with self.profiler.measure("score_covariance", gpu=True):
                u, v = model.score_covariance(g, **reduction_options)
            cutoffs=dict(rare_maf_cutoff=self.options.rare_maf_cutoff,rv_num_cutoff=self.options.rv_num_cutoff,
                         rv_num_cutoff_max=self.options.rv_num_cutoff_max)
            payload = dict(score=u, covariance=v, maf=selected_maf, mac=np.rint(selected_maf*2*model.n),
                           annotations=annotation, names=names, acat_calibration="chi2",
                           cmac=cmac, **cutoffs)
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
        with self.profiler.measure("eigen_tail", gpu=True):
            if model.n_pheno > 1:
                return multi_staar_test(**payload)
            tail_optimization = getattr(self, "statistics_tail_optimization", False)
            result = staar_test(**payload, matmul_mode=getattr(model, "matmul_mode", "fp64"),
                               tail_optimization=tail_optimization,
                               weight_batch_optimization=getattr(self, "weight_batch_optimization", False))
            if tail_optimization:
                self.statistics_tail_optimization_calls = getattr(self, "statistics_tail_optimization_calls", 0) + 1
            return result

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
        index_sets = list(index_sets)
        if getattr(self, "local_mask_reuse", False) and len(index_sets) > 1:
            return self._test_sets_local_union(index_sets)
        return [self.test_set(indices) for indices in index_sets]

    def _test_sets_local_union(self, index_sets):
        """Resolve existing cache and family aliases before any union work."""
        counters = self.local_mask_reuse_counters
        counters["families"] += 1
        representatives, aliases, pending = {}, {}, []
        output = [None] * len(index_sets)
        def copy_results(results):
            return [None if result is None else dict(result) for result in results]
        for position, indices in enumerate(index_sets):
            key = self._set_key(indices)
            if key in representatives:
                aliases[position] = representatives[key]
                counters["duplicate_mask_hits"] += 1
                continue
            representatives[key] = position
            if key in self._test_set_cache:
                self._test_set_cache.move_to_end(key)
                output[position] = copy_results(self._test_set_cache[key])
                counters["cache_hits"] += 1
            else:
                pending.append((position, key, indices))
        if len(pending) == 1:
            counters["single_mask_paths"] += 1
            position, _, indices = pending[0]
            output[position] = self.test_set(indices)
        elif pending:
            results = self._test_sets_uncached_union([indices for _, _, indices in pending])
            for (position, key, _), result in zip(pending, results):
                output[position] = result
                # Include insufficient/NULL sets in the existing bounded LRU.
                # Same-mode fallback test_set may already have cached them.
                if key not in self._test_set_cache:
                    self._cache_set(key, result)
        for position, representative in aliases.items():
            output[position] = copy_results(output[representative])
        # Match the original per-mask access order, including alias accesses.
        for indices in index_sets:
            key = self._set_key(indices)
            if key in self._test_set_cache:
                self._test_set_cache.move_to_end(key)
        return output

    def _test_sets_uncached_union(self, index_sets):
        """Reuse one local family UV; evaluate each unique mask separately.

        This experimental path only supports one Gaussian trait/model. A union
        changes output tiles, not the contraction/sample order or per-variant
        matrix semantics. Native comparisons must still validate every family.
        """
        from .local_mask_reuse import attempt_union, valid_index_sets
        counters = self.local_mask_reuse_counters
        def fallback(reason):
            counters[reason] += 1
            # This preserves the configured model mode, including TF32 audit.
            return [self.test_set(indices) for indices in index_sets]
        if (len(self.models) != 1 or not isinstance(self.models[0], GaussianNullModel)
                or self.models[0].n_pheno != 1 or self.models[0].use_spa
                or getattr(self.models[0], "matmul_mode", "fp64") == "fp64"):
            return fallback("fallback_unsupported")
        if not valid_index_sets(index_sets):
            return fallback("fallback_mapping")
        self._local_union_host_prepared = None
        self._local_union_device_prepared = None
        try:
            union_indices = np.unique(np.concatenate(index_sets))
            counters["input_variant_columns"] += sum(len(indices) for indices in index_sets)
            counters["union_variant_columns"] += len(union_indices)
            prepared = self._prepare_test_set(union_indices, _local_masks=index_sets, _defer_score=True)[0]
            if prepared is None:
                return [[None] for _ in index_sets]
            if "_resident_blocks" in prepared:
                self._local_union_device_prepared = prepared
            else:
                self._local_union_host_prepared = prepared
            result, reason = attempt_union(lambda: self._calculate_local_union(index_sets, prepared),
                                           device=self.models[0].device)
            if reason is not None:
                # Only NumPy host state survives the unwound GPU frame.
                # Reuse its exact filtering/imputation instead of decoding
                # the same physical variants again for each mask.
                device_prepared = self._local_union_device_prepared
                if device_prepared is not None:
                    counters[reason] += 1
                    return self._test_masks_from_union_device(device_prepared, index_sets)
                prepared = self._local_union_host_prepared
                if prepared is not None:
                    counters[reason] += 1
                    return self._test_masks_from_union_host(prepared, index_sets)
                return fallback(reason)
            return result
        finally:
            self._local_union_host_prepared = None
            self._local_union_device_prepared = None

    def _test_masks_from_union_device(self, prepared, index_sets):
        from .local_mask_reuse import ordered_mask_columns
        model, output = self.models[0], []
        for indices in index_sets:
            columns = ordered_mask_columns(prepared["_variant_indices"], prepared["_extraction_groups"],
                indices, grouped=self.options.wrapper_semantics == "base")
            if len(columns) < self.options.rv_num_cutoff:
                output.append([None]); continue
            genotype = self._materialize_resident_gene(prepared, columns)
            payload = {key: value for key, value in prepared.items() if not key.startswith("_")}
            for key in ("maf", "mac", "annotations"):
                payload[key] = prepared[key][columns]
            payload["cmac"] = self._resident_gene_cmac(genotype)
            with self.profiler.measure("score_covariance", gpu=True):
                payload["score"], payload["covariance"] = model.score_covariance(genotype)
            del genotype
            output.append([self._evaluate_prepared(payload, model)])
            counters = self.local_mask_reuse_counters
            counters["fallback_device_reused_masks"] = counters.get("fallback_device_reused_masks", 0) + 1
        return output

    def _test_masks_from_union_host(self, prepared, index_sets):
        """Same-mode per-mask products from already prepared host columns."""
        from .local_mask_reuse import ordered_mask_columns
        model = self.models[0]
        physical = prepared["_variant_indices"]
        output = []
        for indices in index_sets:
            columns = ordered_mask_columns(physical, prepared["_extraction_groups"], indices,
                                           grouped=self.options.wrapper_semantics == "base")
            if len(columns) < self.options.rv_num_cutoff:
                output.append([None]); continue
            self._limit(model, len(columns))
            host = prepared["_genotype_host"][:, columns]
            payload = {key: value for key, value in prepared.items() if not key.startswith("_")}
            for key in ("maf", "mac", "annotations"):
                payload[key] = prepared[key][columns]
            payload["cmac"] = float(host.sum())
            with self.profiler.measure("genotype_h2d", gpu=True):
                genotype = torch.as_tensor(host, dtype=model.x.dtype, device=model.device)
            with self.profiler.measure("score_covariance", gpu=True):
                payload["score"], payload["covariance"] = model.score_covariance(genotype)
            del genotype
            output.append([self._evaluate_prepared(payload, model)])
            counter = "fallback_host_reused_masks"
            self.local_mask_reuse_counters[counter] = self.local_mask_reuse_counters.get(counter, 0) + 1
        return output

    def _calculate_local_union(self, index_sets, prepared):
        """All floating union temporaries die before same-mode recovery."""
        from .local_mask_reuse import ordered_mask_columns
        counters, model = self.local_mask_reuse_counters, self.models[0]
        prepared = dict(prepared)
        resident = "_resident_blocks" in prepared
        host_genotype = None if resident else prepared.pop("_genotype_host")
        physical_indices = prepared.pop("_variant_indices")
        counters["prepared_union_variant_columns"] += len(physical_indices)
        groups = prepared.pop("_extraction_groups")
        geometry = prepared.pop("_union_geometry")
        if geometry is not None and not geometry["beneficial"]:
            from .local_mask_reuse import UnionGeometryRejected
            raise UnionGeometryRejected("actual rare union covariance exceeds separate-mask cells")
        self._limit(model, len(physical_indices))
        if resident:
            genotype = self._materialize_resident_gene(self._local_union_device_prepared)
            prepared.pop("_resident_blocks"); prepared.pop("_resident_mapping")
        else:
            with self.profiler.measure("genotype_h2d", gpu=True):
                genotype = torch.as_tensor(host_genotype, dtype=model.x.dtype, device=model.device)
        with self.profiler.measure("score_covariance", gpu=True):
            union_score, union_covariance = model.score_covariance(genotype)
        counters["union_score_calls"] += 1
        if not resident:
            del genotype  # Resident G supplies the original imputed-mask sum.
        output = []
        evaluated_masks = 0
        for indices in index_sets:
            columns = ordered_mask_columns(physical_indices, groups, indices,
                                           grouped=self.options.wrapper_semantics == "base")
            if len(columns) < self.options.rv_num_cutoff:
                result = None
            else:
                device_columns = torch.as_tensor(columns, dtype=torch.int64, device=model.device)
                payload = dict(prepared)
                payload.update(score=union_score.index_select(0, device_columns),
                    covariance=union_covariance.index_select(0, device_columns).index_select(1, device_columns),
                    maf=prepared["maf"][columns], mac=prepared["mac"][columns],
                    annotations=prepared["annotations"][columns],
                    # Advanced column indexing reproduces the original host
                    # mask layout and sum order, not a sum of rounded MACs.
                    cmac=self._resident_gene_cmac(genotype[:, device_columns]) if resident else float(host_genotype[:, columns].sum()))
                result = self._evaluate_prepared(payload, model)
                evaluated_masks += 1
            output.append([result])
        counters["reused_masks"] += evaluated_masks
        return output

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
        reader_options = {"device": genotype_device, "minimum_mac": mac_cutoff, "resident": self.resident_genotypes} if genotype_device is not None else {}
        if start is None:
            base=self._base_mask(chromosome,variant_type)
            def blocks():
                for offset in range(0,self.gds.n_variants,self.options.annotation_block_size):
                    stop=min(offset+self.options.annotation_block_size,self.gds.n_variants)
                    indices=offset+np.flatnonzero(base[offset:stop])
                    yield from self._minor_blocks(indices,self.union_rows,block_size=self.options.genotype_block_size, **reader_options)
            genotype_blocks=blocks()
        else:
            indices = self.region_indices(start, end)
            a = self.annotations(indices,include_weights=False,metadata="mask")
            indices = indices[np.flatnonzero(variant_filter(a, variant_type))]
            genotype_blocks=self._minor_blocks(indices,self.union_rows,block_size=self.options.genotype_block_size, **reader_options)
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
            if len(union_columns) != len(block.variant_indices):
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
                self._limit(model, len(trait_columns), individual=True)
                trait_block = block if len(trait_columns) == len(block.variant_indices) else block.select_columns(trait_columns)
                trait_ordinals = ordinals[trait_columns]
                frequency = {"frequency_mode": "reference"} if base_mode else {}
                if isinstance(trait_block, DeviceMinorBlock):
                    frequency["dtype"] = model.x.dtype
                with self.profiler.measure("genotype_trait_prepare", gpu=True):
                    g, maf, mac, missing, is_alt = trait_block.trait_dense(rows, self.options.imputation, **frequency)
                if base_mode:
                    original_alt_af = 1 - trait_block.union_ref_af
                    original_maf = np.where(trait_block.union_ref_af >= original_alt_af, original_alt_af, trait_block.union_ref_af)
                    allele_missing = trait_block.allele_missing_rate()
                    base_group = np.where(original_alt_af > 0.5, 2,
                        np.where((original_maf >= 0.01) | (allele_missing >= 0.01), 1, 0))
                # Retain the reference host layout only for FP64 controls;
                # forced TF32 can pass the already selected CUDA tensor.
                keep = np.ones(len(trait_columns), dtype=bool)
                columns = np.arange(len(trait_columns))
                selected = trait_block.variant_indices
                trait_records=[]
                analysis_g = (g[:, keep] if getattr(model, "matmul_mode", "fp64") == "fp64" else
                              torch.as_tensor(g, dtype=torch.float32, device=model.device))
                with self.profiler.measure("score_covariance", gpu=True):
                    if model.n_pheno==1 and hasattr(model,"individual_score_variance"):
                        u,variance=model.individual_score_variance(analysis_g)
                    else:
                        u,v=model.score_covariance(analysis_g)
                        if model.n_pheno==1:variance=v.diagonal()
                with self.profiler.measure("individual_tail", gpu=True):
                    if model.n_pheno>1:
                        scores=u.reshape(model.n_pheno,len(columns))
                        cov4=v.reshape(model.n_pheno,len(columns),model.n_pheno,len(columns))
                        log_probabilities=torch.stack([joint_individual_logp(scores[:,j],cov4[:,j,:,j]) for j in range(len(columns))])
                    else:
                        # Preserve sqrt's IEEE values for native R output; only
                        # the probability helper protects its division.
                        standard_error=torch.sqrt(variance.to(torch.float64))
                        log_probabilities=_individual_log_probabilities(u, variance)
                        if model.use_spa:
                            probabilities=individual_score_test_spa(torch.as_tensor(g[:,keep],dtype=torch.float64,device=model.device),
                                model.scaled_residuals,model.fitted_probability,model.xw,model.projection_left,
                                normal_pvalues=torch.exp(-log_probabilities) if self.options.spa_p_filter else None,
                                p_filter_cutoff=self.options.p_filter_cutoff,tol=self.options.spa_tol,max_iter=self.options.spa_max_iter)
                # Transfer one result block, avoiding four CUDA synchronizations
                # per variant. Computed values are serialized as R doubles with original row metadata.
                with self.profiler.measure("result_d2h", gpu=True):
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
