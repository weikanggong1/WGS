"""Gene annotations and streamed ALT burden / minor-allele VC masks.

Annotation domains are opaque labels supplied by the analysis.  In particular,
this module never guesses what C, R, or UR means from those letters.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
from numbers import Integral
from pathlib import Path
from typing import Collection, Iterable, Iterator, Mapping, Sequence

import torch


@dataclass(frozen=True)
class Annotation:
    variant_id: str
    gene: str
    category: str
    domain: str | None = None


@dataclass(frozen=True)
class GeneSet:
    gene: str
    chrom: str
    position: int
    variant_ids: tuple[str, ...]


@dataclass(frozen=True)
class MaskDefinition:
    name: str
    categories: frozenset[str]
    extract_variants: frozenset[str] | None = None
    score: str | None = None
    # Native ##MASKS headers retain category order from the input file.
    category_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FrequencyDomain:
    """Explicit MAF domain for analyses that construct their own annotations."""
    lower: float = 0.0
    upper: float = 1.0
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def contains(self, value: float) -> bool:
        low = value >= self.lower if self.lower_inclusive else value > self.lower
        high = value <= self.upper if self.upper_inclusive else value < self.upper
        return low and high


@dataclass
class GeneConfig:
    # Context construction applies RINT once; gene tests consume that context.
    apply_rint: bool = True
    aaf_bins: tuple[float, ...] = (0.01,)
    vc_max_aaf: float = 0.01
    min_mac: float = 1.0
    collapse_mac: float = 10.0
    beta_a: float = 1.0
    beta_b: float = 25.0
    include_singletons: bool = True
    singleton_carrier: bool = False
    include_domains: bool = True
    include_overall: bool = True
    domain_mapping: Mapping[str, FrequencyDomain] | None = None
    extract_variants: Collection[str] | None = None
    extract_genes: Collection[str] | None = None
    variant_block_size: int = 1000
    max_matrix_bytes: int = 8 * 1024**3
    vc_score_method: str = "residual"
    vc_storage: str = "dense"
    vc_score_block_size: int = 256
    genotype_reader: str = "cpu"
    genotype_orientation: str = "allele1"
    skato_rhos: tuple[float, ...] = (0., .01, .04, .09, .16, .25, .5, 1.)
    tail_method: str = "regenie"
    davies_controller: str = "auto"
    davies_fourier_backend: str = "torch"
    eigen_backend: str = "dense"
    secular_min_size: int = 4096
    secular_root_chunk: int = 256
    secular_iterations: int = 64
    skato_integral_backend: str = "adaptive_x"
    skato_integral_epsabs: float = 1e-25
    skato_integral_epsrel: float = 2.**-13
    skato_integral_max_intervals: int = 1000
    acato_full: bool = True
    gene_p: bool = True
    gene_p_groups: Mapping[str, Collection[str]] | None = None
    run_sbat: bool = True
    sbat_max_subsets: int = 10
    sbat_qmc_samples: int = 8192
    sbat_seed: int = 0
    sbat_subset_sampling: str = "unique"
    rank_tolerance: float = 1e-7
    qr_tie_tolerance: float = 0.0
    genotype_scale_tolerance: float = 1e-6
    product_cache_bytes: int = 256 * 1024**2

    def __post_init__(self):
        from ._quadrature import validate_quadrature_parameters
        if self.skato_integral_backend not in {"segmented", "adaptive_x", "adaptive_sqrt", "qags_x"}:
            raise ValueError("SKAT-O integral backend must be segmented, adaptive_x, adaptive_sqrt or qags_x.")
        validate_quadrature_parameters(self.skato_integral_epsabs, self.skato_integral_epsrel,
                                       self.skato_integral_max_intervals)
        if self.davies_controller not in {"auto", "scalar", "numpy"}:
            raise ValueError("davies_controller must be auto, scalar or numpy.")
        if self.davies_fourier_backend not in {"auto", "torch", "fused"}:
            raise ValueError("davies_fourier_backend must be auto, torch or fused.")
        # JSON serializes tuples as lists. Normalize at the public config
        # boundary so both CLI JSON and direct Python calls build the same masks.
        try:
            self.aaf_bins = tuple(float(value) for value in self.aaf_bins)
            self.skato_rhos = tuple(float(value) for value in self.skato_rhos)
        except (TypeError, ValueError) as error:
            raise ValueError("AAF bins and SKAT-O rho values must be numeric sequences.") from error
        if not self.aaf_bins or any(not 0 < x <= 1 for x in self.aaf_bins):
            raise ValueError("aaf_bins must contain upper bounds in (0, 1].")
        if not self.skato_rhos or any(not 0 <= x <= 1 for x in self.skato_rhos):
            raise ValueError("skato_rhos must contain rho values in [0, 1].")
        if not 0 < self.vc_max_aaf <= 1:
            raise ValueError("vc_max_aaf must be in (0, 1].")
        if self.min_mac < 0 or self.collapse_mac < 0:
            raise ValueError("MAC thresholds must be nonnegative.")
        if not (math.isfinite(self.beta_a) and math.isfinite(self.beta_b)
                and self.beta_a > 0 and self.beta_b > 0):
            raise ValueError("Beta weight parameters must be finite and positive.")
        if self.variant_block_size < 1 or self.max_matrix_bytes < 1:
            raise ValueError("Block size and matrix memory limit must be positive.")
        if (isinstance(self.product_cache_bytes, bool)
                or not isinstance(self.product_cache_bytes, Integral)
                or self.product_cache_bytes < 0):
            raise ValueError("product_cache_bytes must be a nonnegative integer (0 disables caching).")
        if self.vc_score_method not in {"residual", "crossproduct"}:
            raise ValueError("vc_score_method must be residual or crossproduct.")
        if self.vc_storage not in {"dense", "sparse"}:
            raise ValueError("vc_storage must be dense or sparse.")
        if self.genotype_reader not in {"cpu", "cuda_packed"}:
            raise ValueError("genotype_reader must be cpu or cuda_packed.")
        if self.vc_storage == "sparse" and self.vc_score_method != "crossproduct":
            raise ValueError("Sparse VC storage requires the crossproduct score method.")
        if not isinstance(self.vc_score_block_size, Integral) or self.vc_score_block_size < 1:
            raise ValueError("VC score block size must be a positive integer.")
        if self.sbat_subset_sampling not in {"unique", "with_replacement"}:
            raise ValueError("SBAT subset sampling must be unique or with_replacement.")
        if self.genotype_orientation not in {"variant_id", "allele1", "alt"}:
            raise ValueError("genotype_orientation must be variant_id, allele1, or alt.")
        if self.eigen_backend not in {"dense", "secular", "auto"}:
            raise ValueError("eigen_backend must be dense, secular, or auto.")
        for name in ("secular_min_size", "secular_root_chunk", "secular_iterations"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.tail_method not in {"regenie", "exact"}:
            raise ValueError("tail_method must be regenie or exact.")
        if not isinstance(self.sbat_max_subsets, Integral) or self.sbat_max_subsets < 0:
            raise ValueError("SBAT subset threshold must be a nonnegative integer.")
        if not isinstance(self.sbat_qmc_samples, Integral) or self.sbat_qmc_samples < 1:
            raise ValueError("SBAT QMC sample count must be a positive integer.")
        if not isinstance(self.sbat_seed, Integral):
            raise ValueError("SBAT seed must be an integer.")
        if not (math.isfinite(self.genotype_scale_tolerance) and self.genotype_scale_tolerance >= 0):
            raise ValueError("Genotype scale tolerance must be finite and nonnegative.")
        if not (math.isfinite(self.qr_tie_tolerance) and 0 <= self.qr_tie_tolerance < 1):
            raise ValueError("QR tie tolerance must be finite and in [0, 1).")


def _lines(path: str | Path):
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if text and not text.startswith("#"):
                yield number, text.split()


def load_annotations(path: str | Path, genes: Collection[str] | None = None) -> list[Annotation]:
    """Read REGENIE three/four-column annotations: ID gene [domain] category."""
    selected = set(genes) if genes is not None else None
    records = []
    for number, parts in _lines(path):
        if len(parts) not in (3, 4):
            raise ValueError(f"Annotation line {number} must have 3 or 4 columns.")
        variant, gene = parts[:2]
        if selected is not None and gene not in selected:
            continue
        domain, category = (None, parts[2]) if len(parts) == 3 else (parts[2], parts[3])
        records.append(Annotation(variant, gene, category, domain))
    return records


def load_setlist(path: str | Path, genes: Collection[str] | None = None) -> list[GeneSet]:
    selected = set(genes) if genes is not None else None
    sets = []
    for number, parts in _lines(path):
        if len(parts) != 4:
            raise ValueError(f"Setlist line {number} must have 4 columns.")
        gene, chrom, position, ids = parts
        if selected is None or gene in selected:
            sets.append(GeneSet(gene, chrom.removeprefix("chr"), int(position),
                                tuple(dict.fromkeys(ids.split(",")))))
    return sets


def load_mask_definitions(path: str | Path, *, extract_variants: Collection[str] | None = None,
                          score: str | None = None) -> list[MaskDefinition]:
    whitelist = None if extract_variants is None else frozenset(extract_variants)
    masks = []
    for number, parts in _lines(path):
        if len(parts) != 2:
            raise ValueError(f"Mask line {number} must have name and category list.")
        category_order = tuple(parts[1].split(","))
        masks.append(MaskDefinition(parts[0], frozenset(category_order), whitelist, score,
                                    category_order=category_order))
    if len({m.name for m in masks}) != len(masks):
        raise ValueError("Mask names must be unique.")
    return masks


def effective_mask_definitions(definitions: Sequence[MaskDefinition],
                               annotations: Iterable[Annotation],
                               available_variants: Collection[str]) -> list[MaskDefinition]:
    """Return native output-header masks without changing their source definitions.

    Supply the complete annotation records and the BIM IDs remaining after the
    global/score variant filters. Native category registration precedes setlist,
    gene, AAF and MAC filtering, so those filters must not restrict this input.
    ``NULL`` is registered even without an annotation record. Unknown categories
    are removed in their original order, and masks with none left are omitted.
    The remaining fields, including per-mask variant filters, are preserved.
    """
    available = (available_variants if isinstance(available_variants, (set, frozenset, Mapping))
                 else set(available_variants))
    registered = {"NULL"}
    registered.update(record.category for record in annotations
                      if record.variant_id in available)
    effective = []
    for definition in definitions:
        categories = definition.categories.intersection(registered)
        if not categories:
            continue
        source_order = (definition.category_order if definition.category_order is not None
                        else tuple(sorted(definition.categories)))
        category_order = tuple(category for category in source_order if category in categories)
        effective.append(replace(definition, categories=frozenset(categories),
                                 category_order=category_order))
    return effective


def load_variant_whitelist(path: str | Path, variant_ids: Collection[str] | None = None) -> frozenset[str]:
    """Read an original score whitelist, retaining optional annotation candidates.

    Every source line is parsed with the same first-column semantics. Filtering
    changes storage rather than membership for the candidate variants; the
    default still returns the complete whitelist for existing Python callers.
    """
    allowed = (variant_ids if isinstance(variant_ids, (set, frozenset)) else set(variant_ids)) if variant_ids is not None else None
    return frozenset(parts[0] for _, parts in _lines(path)
                     if allowed is None or parts[0] in allowed)


def beta_maf_weights(maf: torch.Tensor, a: float = 1., b: float = 25.) -> torch.Tensor:
    """Beta(MAF; a, b) density on the tensor's device, not its square."""
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("Beta weight parameters must be finite and positive.")
    if a == 1. and b == 25.:
        return 25.0 * (1.0 - maf).pow(24)
    parameters = maf.new_tensor([a, b, a+b])
    log_normalizer = torch.lgamma(parameters[:2]).sum() - torch.lgamma(parameters[2])
    x = maf.clamp(torch.finfo(maf.dtype).tiny, 1-torch.finfo(maf.dtype).eps)
    return ((a-1)*x.log() + (b-1)*torch.log1p(-x) - log_normalizer).exp()


def orient_alt(genotypes: torch.Tensor, variants: Sequence[object], orientation: str) -> torch.Tensor:
    """Convert BIM allele1 dosages to ALT from a chr:pos:REF:ALT variant ID."""
    if orientation in {"alt", "allele1"}:
        return genotypes
    flip = []
    for variant in variants:
        parts = str(variant.id).split(":")
        if len(parts) != 4:
            raise ValueError(f"ALT orientation requires a chr:pos:REF:ALT ID: {variant.id}")
        alt = parts[3]
        if str(variant.allele1) == alt:
            flip.append(False)
        elif str(variant.allele0) == alt:
            flip.append(True)
        else:
            raise ValueError(f"Variant {variant.id} ALT does not match either BIM allele.")
    to_flip = torch.tensor(flip, device=genotypes.device, dtype=torch.bool)
    valid = torch.isfinite(genotypes) & (genotypes >= 0)
    return torch.where(to_flip[None, :] & valid, 2.0 - genotypes, genotypes)


@dataclass
class PreparedMask:
    name: str
    base_name: str
    domain: str | None
    frequency: str
    aaf_upper: float | None
    variant_ids: tuple[str, ...]
    burden: torch.Tensor
    aaf: float
    mac: float
    n_observed: int
    vc_genotypes: torch.Tensor | None
    vc_mafs: torch.Tensor | None
    score: str | None = None
    raw_burden: torch.Tensor | None = None
    beta_a: float = 1.
    beta_b: float = 25.
    # Internal trusted builder provenance. External masks default to no reuse.
    # The identity guard binds each token to its original, unmodified tensor.
    vc_reuse_key: tuple | None = field(default=None, repr=False, compare=False)
    vc_reuse_tensor_guard: tuple | None = field(default=None, repr=False, compare=False)
    burden_reuse_key: tuple | None = field(default=None, repr=False, compare=False)
    burden_reuse_tensor_guard: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def vc_weights(self):
        return None if self.vc_mafs is None else beta_maf_weights(self.vc_mafs, self.beta_a, self.beta_b)

    @property
    def acat_weights(self):
        if self.vc_mafs is None:
            return None
        return self.vc_weights.square() * self.vc_mafs * (1.0 - self.vc_mafs)


def _vc_tensor_guard(tensor: torch.Tensor | None) -> tuple | None:
    """Metadata-only identity/version guard for a cached matrix or vector.

    Dense strides and sparse component pointers/versions are included without
    transferring numerical data. Inference tensors have no version counter;
    unsupported layouts and uncoalesced COO inputs are conservatively uncached.
    """
    if tensor is None:
        return None
    try:
        identity = (id(tensor), tensor._version, tensor.layout, tuple(tensor.shape),
                    tensor.device, tensor.dtype)
        if tensor.layout == torch.strided:
            return identity + (tensor.data_ptr(), tuple(tensor.stride()), tensor.storage_offset())
        if tensor.layout == torch.sparse_coo and tensor.is_coalesced():
            indices, values = tensor.indices(), tensor.values()
            return identity + (indices.data_ptr(), indices._version, tuple(indices.shape),
                               values.data_ptr(), values._version, tuple(values.shape))
    except RuntimeError:
        # Inference-mode tensors deliberately omit mutation version counters.
        return None
    return None


@dataclass
class _SharedVC:
    values: torch.Tensor | None
    mafs: torch.Tensor | None
    references: int = 0


@dataclass
class _VCSelection:
    shared: _SharedVC
    columns: torch.Tensor


def _matrix_storage_bytes(matrix: torch.Tensor) -> int:
    if matrix.is_sparse:
        return (matrix.values().numel() * matrix.values().element_size()
                + matrix.indices().numel() * matrix.indices().element_size())
    return matrix.numel() * matrix.element_size()


def _select_vc_columns(matrix: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    if not matrix.is_sparse:
        return matrix[:, columns]
    mapping = torch.full((matrix.shape[1],), -1, device=matrix.device, dtype=torch.long)
    mapping[columns] = torch.arange(columns.numel(), device=matrix.device)
    coordinates, values = matrix.indices(), matrix.values()
    selected = mapping[coordinates[1]] >= 0
    indices = torch.stack([coordinates[0, selected], mapping[coordinates[1, selected]]])
    return torch.sparse_coo_tensor(indices, values[selected], (matrix.shape[0], columns.numel()),
                                  device=matrix.device, dtype=matrix.dtype).coalesce()


def _concatenate_vc_chunks(chunks: Sequence[torch.Tensor], *, sparse=False) -> torch.Tensor:
    if not sparse:
        return torch.cat(chunks, 1)
    indices, values, offset = [], [], 0
    for chunk in chunks:
        source = chunk.coalesce() if chunk.is_sparse else chunk.to_sparse_coo().coalesce()
        coordinates = source.indices().clone()
        coordinates[1] += offset
        indices.append(coordinates)
        values.append(source.values())
        offset += source.shape[1]
    return torch.sparse_coo_tensor(torch.cat(indices, 1), torch.cat(values),
                                  (chunks[0].shape[0], offset), device=chunks[0].device,
                                  dtype=chunks[0].dtype).coalesce()


def _copy_mask_variant_metadata(aaf: torch.Tensor, maf: torch.Tensor,
                                mac: torch.Tensor, ac: torch.Tensor,
                                counts: torch.Tensor, singletons: torch.Tensor,
                                *, max_count: int) -> list[list]:
    """Copy computed variant metadata once, without rounding its original values."""
    values = (aaf, maf, mac, ac, counts, singletons)
    # Float64 holds every float32 value and counts up to 2**53 exactly. Keep
    # full int64 count semantics even for a theoretical larger sample set.
    if max_count > 2**53:
        return [value.detach().cpu().tolist() for value in values]
    copied = torch.stack([value.to(torch.float64) for value in values]).detach().cpu().tolist()
    copied[4] = [int(value) for value in copied[4]]
    copied[5] = [bool(value) for value in copied[5]]
    return copied


def _copy_mask_frequency(ac: torch.Tensor, n: int) -> list[float]:
    # Retain the original Python-scalar denominator after n.item(): replacing
    # it with a device tensor can change floating-point division rounding.
    return torch.stack((ac / (2 * max(n, 1)),
                        torch.minimum(ac, 2 * n - ac))).detach().cpu().tolist()


@dataclass
class _State:
    definition: MaskDefinition
    domain: str | None
    upper: float | None
    frequency: str
    raw_alt: torch.Tensor
    raw_minor_rare: torch.Tensor
    variant_ids: list[str] = field(default_factory=list)
    vc_chunks: list[_VCSelection] = field(default_factory=list)
    raw_member_ordinals: list[int] = field(default_factory=list)
    vc_regular_ordinals: list[int] = field(default_factory=list)
    vc_rare_ordinals: list[int] = field(default_factory=list)


class GeneMaskBuilder:
    """Accumulate a gene in blocks, retaining only noncollapsed VC genotypes."""
    def __init__(self, annotations: Sequence[Annotation], definitions: Sequence[MaskDefinition],
                 n_samples: int, device: torch.device | str, dtype: torch.dtype,
                 config: GeneConfig | None = None):
        self.config = config or GeneConfig()
        self.device, self.dtype = torch.device(device), dtype
        if not definitions:
            raise ValueError("At least one mask definition is required.")
        self.n_samples = n_samples
        self.annotations = defaultdict(list)
        for record in annotations:
            self.annotations[record.variant_id].append(record)
        domains = sorted({a.domain for a in annotations if a.domain is not None})
        if self.config.domain_mapping is not None:
            domains = sorted(set(domains) | set(self.config.domain_mapping))
        domains = domains if self.config.include_domains else []
        if self.config.include_overall:
            domains = domains + [None]
        if not domains:
            raise ValueError("No annotation domains or overall mask were selected.")
        cutoffs = sorted(set(self.config.aaf_bins + (self.config.vc_max_aaf,)))
        bins = [(x, format(x, ".12g")) for x in cutoffs]
        if self.config.include_singletons:
            bins = [(None, "singleton")] + bins
        self.states = []
        for definition in definitions:
            for domain in domains:
                for upper, label in bins:
                    empty = torch.full((n_samples,), -1., device=self.device, dtype=dtype)
                    self.states.append(_State(definition, domain, upper, label, empty,
                                               empty.clone()))
        self.whitelist = None if self.config.extract_variants is None else set(self.config.extract_variants)
        self.retained_bytes = 0
        self.finished = False
        # An opaque namespace prevents equal member positions in different
        # genes/builders/sample contexts from sharing a products cache entry.
        self._reuse_namespace = object()
        self._next_variant_ordinal = 0

    def _eligible_annotation(self, variant_id: str, state: _State, maf: float) -> bool:
        if self.whitelist is not None and variant_id not in self.whitelist:
            return False
        if state.definition.extract_variants is not None and variant_id not in state.definition.extract_variants:
            return False
        matching = [a for a in self.annotations[variant_id] if a.category in state.definition.categories]
        if not matching:
            return False
        if state.domain is None:
            return True
        if self.config.domain_mapping is not None and state.domain in self.config.domain_mapping:
            return self.config.domain_mapping[state.domain].contains(maf)
        return any(a.domain == state.domain for a in matching)

    def update(self, variant_ids: Sequence[str], genotypes: torch.Tensor):
        if self.finished:
            raise RuntimeError("A finalized gene mask builder cannot accept new genotypes.")
        raw = torch.as_tensor(genotypes, device=self.device, dtype=self.dtype)
        if raw.ndim != 2 or raw.shape != (self.n_samples, len(variant_ids)):
            raise ValueError("Genotype block must have shape [analysis samples, variants].")
        ordinal_base = self._next_variant_ordinal
        self._next_variant_ordinal += len(variant_ids)
        valid = torch.isfinite(raw) & (raw >= 0) & (raw <= 2)
        raw = torch.where(valid, raw, -1.)
        counts = valid.sum(0)
        ac = torch.where(valid, raw, 0.).sum(0)
        aaf = ac / (2 * counts.clamp_min(1))
        maf = torch.minimum(aaf, 1.0 - aaf)
        mac = torch.minimum(ac, 2 * counts - ac)
        # Frozen 3.4.1 marks singletons from AAC before minor-allele flipping.
        singletons = ((raw >= .5) & valid).sum(0) == 1 if self.config.singleton_carrier else (ac + .5).floor() == 1
        aafs, mafs, macs, aacs, ns, single = _copy_mask_variant_metadata(
            aaf, maf, mac, ac, counts, singletons, max_count=self.n_samples)
        minor = torch.where((aaf > .5)[None, :] & valid, 2.0 - raw, raw)
        # Several masks and domains can include the same ordinary VC sites.
        # Keep one imputed block and cheap column selections until each mask
        # is tested, rather than duplicating N x V matrices for every mask.
        shared, shared_lookup = None, {}
        for state in self.states:
            indices = [j for j, vid in enumerate(variant_ids)
                       if ns[j] and macs[j] >= .5 and self._eligible_annotation(vid, state, mafs[j])
                       and (single[j] if state.upper is None else aafs[j] <= state.upper)]
            if not indices:
                continue
            state.variant_ids.extend(variant_ids[j] for j in indices)
            state.raw_member_ordinals.extend(ordinal_base+j for j in indices)
            selected = torch.tensor(indices, device=self.device)
            state.raw_alt = torch.maximum(state.raw_alt, raw[:, selected].amax(1))
            if state.upper != self.config.vc_max_aaf:
                continue
            # REGENIE's mac1 used for collapsing is AAC; it equals MAC in the
            # requested AAF <= 1% analysis.  Ordinary VC columns are flipped below.
            rare = [j for j in indices if 0 < aacs[j] <= self.config.collapse_mac]
            if rare:
                state.vc_rare_ordinals.extend(ordinal_base+j for j in rare)
                state.raw_minor_rare = torch.maximum(state.raw_minor_rare,
                                                    minor[:, rare].amax(1))
            regular = [j for j in indices if aacs[j] > self.config.collapse_mac]
            if regular:
                state.vc_regular_ordinals.extend(ordinal_base+j for j in regular)
                if shared is None:
                    # Only sites eligible for at least one mask need storage.
                    reusable = [j for j, vid in enumerate(variant_ids)
                                if ns[j] and macs[j] >= .5 and aacs[j] > self.config.collapse_mac
                                and aafs[j] <= self.config.vc_max_aaf
                                and any(self._eligible_annotation(vid, candidate, mafs[j])
                                        for candidate in self.states
                                        if candidate.upper == self.config.vc_max_aaf)]
                    retained = minor[:, reusable]
                    retained = torch.where(retained >= 0, retained, 2 * maf[reusable][None, :])
                    if self.config.vc_storage == "sparse":
                        retained = retained.to_sparse_coo().coalesce()
                    shared = _SharedVC(retained, maf[reusable].clone())
                    shared_lookup = {column: offset for offset, column in enumerate(reusable)}
                    self.retained_bytes += _matrix_storage_bytes(retained)
                    if self.retained_bytes > self.config.max_matrix_bytes:
                        raise MemoryError("Unique gene VC genotypes exceed max_matrix_bytes.")
                shared.references += 1
                state.vc_chunks.append(_VCSelection(shared, torch.tensor(
                    [shared_lookup[j] for j in regular], device=self.device, dtype=torch.long)))

    def _release_vc(self, state: _State):
        for selection in state.vc_chunks:
            shared = selection.shared
            shared.references -= 1
            if shared.references == 0:
                self.retained_bytes -= _matrix_storage_bytes(shared.values)
                shared.values, shared.mafs = None, None
        state.vc_chunks.clear()

    def finish_iter(self) -> Iterator[PreparedMask]:
        """Materialize one mask's VC matrix at a time in original mask order.

        Consumers should test a yielded mask before requesting the next one.
        Previously used shared blocks are released after their last mask.
        Burden/raw burden tensors stay available for SBAT and mask artifacts.
        """
        if self.finished:
            raise RuntimeError("A gene mask builder can only be finalized once.")
        self.finished = True
        for state in self.states:
            if not state.variant_ids:
                continue
            observed = state.raw_alt >= 0
            n = int(observed.sum().item())
            ac = torch.where(observed, state.raw_alt, 0.).sum()
            af, mac = _copy_mask_frequency(ac, n)
            if n == 0 or mac < self.config.min_mac:
                self._release_vc(state)
                continue
            burden = torch.where(observed, state.raw_alt, 2 * af)
            vc, mafs, vc_key = None, None, None
            if state.upper == self.config.vc_max_aaf:
                chunks, maf_chunks = [], []
                rare_observed = state.raw_minor_rare >= 0
                has_rare = bool((state.raw_minor_rare > 0).any())
                width = sum(selection.columns.numel() for selection in state.vc_chunks) + int(has_rare)
                matrix_bytes = self.n_samples * width * burden.element_size()
                if self.config.vc_storage == "sparse":
                    matrix_bytes = sum(_matrix_storage_bytes(selection.shared.values)
                                       for selection in state.vc_chunks) + self.n_samples * 24 * int(has_rare)
                covariance_bytes = width ** 2 * 8
                # Check before allocating selections, concatenation and the
                # double-precision projection/eigensolver working matrices.
                workspace_bytes = 2 * matrix_bytes
                if self.config.vc_score_method == "crossproduct":
                    workspace_bytes = matrix_bytes
                    if self.config.vc_storage == "sparse":
                        workspace_bytes += self.n_samples * min(width, self.config.vc_score_block_size) * 8
                if self.retained_bytes + workspace_bytes + 6 * covariance_bytes > self.config.max_matrix_bytes:
                    raise MemoryError("One gene mask and covariance matrix exceed max_matrix_bytes.")
                for selection in state.vc_chunks:
                    shared = selection.shared
                    chunks.append(_select_vc_columns(shared.values, selection.columns))
                    maf_chunks.append(shared.mafs[selection.columns])
                if has_rare:
                    rare_mean = torch.where(rare_observed, state.raw_minor_rare, 0.).sum() / rare_observed.sum()
                    rare_maf = torch.minimum(rare_mean / 2, 1.0 - rare_mean / 2)
                    chunks.append(torch.where(rare_observed, state.raw_minor_rare,
                                              rare_mean)[:, None])
                    maf_chunks.append(rare_maf.reshape(1))
                if chunks:
                    vc = _concatenate_vc_chunks(chunks, sparse=self.config.vc_storage == "sparse")
                    mafs = torch.cat(maf_chunks).to(torch.float64)
                    vc_key = (self._reuse_namespace, tuple(state.vc_regular_ordinals),
                              tuple(state.vc_rare_ordinals) if has_rare else ())
                # Free an imputed input block as soon as its last selection is
                # materialized. The concatenated output owns independent data.
                self._release_vc(state)
                chunks.clear()
                maf_chunks.clear()
            name = ("" if state.domain is None else state.domain + ".") + state.definition.name
            yield PreparedMask(name, state.definition.name, state.domain, state.frequency,
                               state.upper, tuple(dict.fromkeys(state.variant_ids)), burden,
                               af, mac, n, vc, mafs, state.definition.score,
                               raw_burden=torch.where(observed, state.raw_alt, float("nan")),
                               beta_a=self.config.beta_a, beta_b=self.config.beta_b,
                               vc_reuse_key=vc_key, vc_reuse_tensor_guard=_vc_tensor_guard(vc),
                               burden_reuse_key=(self._reuse_namespace, tuple(state.raw_member_ordinals)),
                               burden_reuse_tensor_guard=_vc_tensor_guard(burden))

    def finish(self) -> list[PreparedMask]:
        """Return all materialized masks for the in-memory Python interface."""
        output, retained_outputs = [], 0
        for mask in self.finish_iter():
            if mask.vc_genotypes is not None:
                retained_outputs += _matrix_storage_bytes(mask.vc_genotypes)
                if retained_outputs + self.retained_bytes > self.config.max_matrix_bytes:
                    raise MemoryError("All materialized gene masks exceed max_matrix_bytes; use finish_iter().")
            output.append(mask)
        return output


def build_gene_masks(genotypes: torch.Tensor, variant_ids: Sequence[str],
                     annotation: Sequence[Annotation], masks: Sequence[MaskDefinition],
                     config: GeneConfig | None = None) -> list[PreparedMask]:
    config = config or GeneConfig()
    genotypes = torch.as_tensor(genotypes)
    dtype = genotypes.dtype if genotypes.is_floating_point() else torch.float32
    builder = GeneMaskBuilder(annotation, masks, genotypes.shape[0], genotypes.device, dtype, config)
    for start in range(0, len(variant_ids), config.variant_block_size):
        end = start + config.variant_block_size
        builder.update(variant_ids[start:end], genotypes[:, start:end])
    return builder.finish()
