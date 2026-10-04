"""Gene annotations and streamed ALT burden / minor-allele VC masks.

Annotation domains are opaque labels supplied by the analysis.  In particular,
this module never guesses what C, R, or UR means from those letters.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from numbers import Integral
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

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
    genotype_orientation: str = "allele1"
    skato_rhos: tuple[float, ...] = (0., .01, .04, .09, .16, .25, .5, 1.)
    tail_method: str = "regenie"
    acato_full: bool = True
    gene_p: bool = True
    gene_p_groups: Mapping[str, Collection[str]] | None = None
    run_sbat: bool = True
    sbat_max_subsets: int = 10
    sbat_qmc_samples: int = 8192
    sbat_seed: int = 0
    rank_tolerance: float = 1e-7
    qr_tie_tolerance: float = 0.0
    genotype_scale_tolerance: float = 1e-6

    def __post_init__(self):
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
        if self.genotype_orientation not in {"variant_id", "allele1", "alt"}:
            raise ValueError("genotype_orientation must be variant_id, allele1, or alt.")
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


def load_variant_whitelist(path: str | Path) -> frozenset[str]:
    return frozenset(parts[0] for _, parts in _lines(path))


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

    @property
    def vc_weights(self):
        return None if self.vc_mafs is None else beta_maf_weights(self.vc_mafs, self.beta_a, self.beta_b)

    @property
    def acat_weights(self):
        if self.vc_mafs is None:
            return None
        return self.vc_weights.square() * self.vc_mafs * (1.0 - self.vc_mafs)


@dataclass
class _State:
    definition: MaskDefinition
    domain: str | None
    upper: float | None
    frequency: str
    raw_alt: torch.Tensor
    raw_minor_rare: torch.Tensor
    variant_ids: list[str] = field(default_factory=list)
    vc_chunks: list[torch.Tensor] = field(default_factory=list)
    maf_chunks: list[torch.Tensor] = field(default_factory=list)


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
        raw = torch.as_tensor(genotypes, device=self.device, dtype=self.dtype)
        if raw.ndim != 2 or raw.shape != (self.n_samples, len(variant_ids)):
            raise ValueError("Genotype block must have shape [analysis samples, variants].")
        valid = torch.isfinite(raw) & (raw >= 0) & (raw <= 2)
        raw = torch.where(valid, raw, -1.)
        counts = valid.sum(0)
        ac = torch.where(valid, raw, 0.).sum(0)
        aaf = ac / (2 * counts.clamp_min(1))
        maf = torch.minimum(aaf, 1.0 - aaf)
        mac = torch.minimum(ac, 2 * counts - ac)
        # Frozen 3.4.1 marks singletons from AAC before minor-allele flipping.
        singletons = ((raw >= .5) & valid).sum(0) == 1 if self.config.singleton_carrier else (ac + .5).floor() == 1
        aafs, mafs, macs, aacs, ns, single = [t.detach().cpu().tolist() for t in (aaf, maf, mac, ac, counts, singletons)]
        minor = torch.where((aaf > .5)[None, :] & valid, 2.0 - raw, raw)
        for state in self.states:
            indices = [j for j, vid in enumerate(variant_ids)
                       if ns[j] and macs[j] >= .5 and self._eligible_annotation(vid, state, mafs[j])
                       and (single[j] if state.upper is None else aafs[j] <= state.upper)]
            if not indices:
                continue
            state.variant_ids.extend(variant_ids[j] for j in indices)
            selected = torch.tensor(indices, device=self.device)
            state.raw_alt = torch.maximum(state.raw_alt, raw[:, selected].amax(1))
            if state.upper != self.config.vc_max_aaf:
                continue
            # REGENIE's mac1 used for collapsing is AAC; it equals MAC in the
            # requested AAF <= 1% analysis.  Ordinary VC columns are flipped below.
            rare = [j for j in indices if 0 < aacs[j] <= self.config.collapse_mac]
            if rare:
                state.raw_minor_rare = torch.maximum(state.raw_minor_rare,
                                                    minor[:, rare].amax(1))
            regular = [j for j in indices if aacs[j] > self.config.collapse_mac]
            if regular:
                retained = minor[:, regular].clone()
                retained = torch.where(retained >= 0, retained, 2 * maf[regular][None, :])
                self.retained_bytes += retained.numel() * retained.element_size()
                if self.retained_bytes > self.config.max_matrix_bytes:
                    raise MemoryError("Gene VC matrices exceed max_matrix_bytes; reduce the gene/variant selection.")
                state.vc_chunks.append(retained)
                state.maf_chunks.append(maf[regular].clone())

    def finish(self) -> list[PreparedMask]:
        output = []
        for state in self.states:
            if not state.variant_ids:
                continue
            observed = state.raw_alt >= 0
            n = int(observed.sum().item())
            ac = torch.where(observed, state.raw_alt, 0.).sum()
            af = float((ac / (2 * max(n, 1))).item())
            mac = float(torch.minimum(ac, 2 * n - ac).item())
            if n == 0 or mac < self.config.min_mac:
                continue
            burden = torch.where(observed, state.raw_alt, 2 * af)
            vc, mafs = None, None
            if state.upper == self.config.vc_max_aaf:
                rare_observed = state.raw_minor_rare >= 0
                if bool((state.raw_minor_rare > 0).any()):
                    rare_mean = torch.where(rare_observed, state.raw_minor_rare, 0.).sum() / rare_observed.sum()
                    rare_maf = torch.minimum(rare_mean / 2, 1.0 - rare_mean / 2)
                    state.vc_chunks.append(torch.where(rare_observed, state.raw_minor_rare,
                                                        rare_mean)[:, None])
                    state.maf_chunks.append(rare_maf.reshape(1))
                if state.vc_chunks:
                    vc = torch.cat(state.vc_chunks, 1)
                    mafs = torch.cat(state.maf_chunks).to(torch.float64)
                    covariance_bytes = vc.shape[1] ** 2 * 8
                    # Eigensolvers, rho kernels and residualized G need working space.
                    if self.retained_bytes + 2 * vc.numel() * vc.element_size() + 6 * covariance_bytes > self.config.max_matrix_bytes:
                        raise MemoryError("Gene genotype and covariance matrix exceed max_matrix_bytes.")
                    state.vc_chunks.clear()
                    state.maf_chunks.clear()
            name = ("" if state.domain is None else state.domain + ".") + state.definition.name
            output.append(PreparedMask(name, state.definition.name, state.domain, state.frequency,
                                       state.upper, tuple(dict.fromkeys(state.variant_ids)), burden,
                                       af, mac, n, vc, mafs, state.definition.score,
                                       raw_burden=torch.where(observed, state.raw_alt, float("nan")),
                                       beta_a=self.config.beta_a, beta_b=self.config.beta_b))
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
