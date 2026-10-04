"""Variant masks and annotation weights matching STAARpipelinePheWAS.

GPL-3.0-only. Adapted from Li Lab STAARpipelinePheWAS, commit
7a2c49617b3791c35a20504e260c038c02a4c643, by Xihao Li, Zilin Li,
and Yuxin Yuan. Source: https://github.com/li-lab-genetics/STAARpipelinePheWAS

Selectors return zero-based indices in input order. They do not apply a
trait-specific frequency filter: this must follow sample selection and dosage
imputation in the statistical pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Sequence

import numpy as np


CODING_CATEGORIES = ("plof", "plof_ds", "missense", "disruptive_missense", "synonymous")
NONCODING_CATEGORIES = (
    "upstream", "downstream", "UTR", "promoter_CAGE", "promoter_DHS",
    "enhancer_CAGE", "enhancer_DHS",
)
_SPLICE = ("splicing", "exonic;splicing", "ncRNA_splicing", "ncRNA_exonic;splicing")
_CODING_SPLICE = ("splicing", "exonic;splicing")
_STOP = ("stopgain", "stoploss")
_FRAMESHIFT = ("frameshift deletion", "frameshift insertion")


def _strings(values: Sequence) -> np.ndarray:
    """Keep missing strings unmatched, rather than turning them into names."""
    array = np.asarray(values)
    if array.dtype.kind == "U":
        return array
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return np.asarray([
        "" if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
        else value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def _chromosome(value: object) -> str:
    result = str(value)
    if result.lower().startswith("chr"):
        result = result[3:]
    return result.upper()


@dataclass
class VariantAnnotations:
    """Aligned metadata, with semantic annotation names from the R catalog.

    Positions and gene coordinates use one-based, inclusive genomic coordinates.
    ``snv`` can provide the exact classification exported by SeqVarTools.
    Otherwise REF and ALT must each be one base long, matching the default
    biallelic SeqVarTools isSNV() test. Multiallelic one-base sites are excluded.
    """

    position: Sequence
    qc: Sequence
    annotations: Mapping[str, Sequence]
    ref: Sequence | None = None
    alt: Sequence | None = None
    chromosome: Sequence | None = None
    variant_id: Sequence | None = None
    snv: Sequence | None = None
    _string_columns: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.int64)
        self.qc = _strings(self.qc)
        if self.position.ndim != 1:
            raise ValueError("position must be a one-dimensional array")
        number = len(self.position)
        for name in ("qc", "ref", "alt", "chromosome", "variant_id", "snv"):
            value = getattr(self, name)
            if value is not None and len(value) != number:
                raise ValueError(f"{name} is not aligned with position")
        for name, value in self.annotations.items():
            if len(value) != number:
                raise ValueError(f"annotation {name!r} is not aligned with position")

    def column(self, name: str) -> np.ndarray:
        if name not in self.annotations:
            raise KeyError(f"required annotation {name!r} is absent")
        if name not in self._string_columns:
            self._string_columns[name] = _strings(self.annotations[name])
        return self._string_columns[name]


def variant_filter(variants: VariantAnnotations, variant_type: str = "SNV") -> np.ndarray:
    """Return PASS variants, intersecting with SNV or its complement if requested."""
    if variant_type not in ("SNV", "Indel", "variant"):
        raise ValueError("variant_type must be SNV, Indel, or variant")
    accepted = variants.qc == "PASS"
    if variant_type == "variant":
        return accepted
    if variants.snv is not None:
        snv = np.asarray(variants.snv, dtype=bool)
    else:
        if variants.ref is None or variants.alt is None:
            raise ValueError("snv or both ref and alt are required for variant classification")
        ref, alt = _strings(variants.ref), _strings(variants.alt)
        snv = np.fromiter(
            (len(r) == 1 and len(a) == 1
             for r, a in zip(ref, alt)), dtype=bool, count=len(ref),
        )
    return accepted & (snv if variant_type == "SNV" else ~snv)


def _on_chromosome(variants: VariantAnnotations, chromosome: object | None) -> np.ndarray:
    if chromosome is None:
        return np.ones(len(variants.position), dtype=bool)
    if variants.chromosome is None:
        raise ValueError("chromosome metadata are required when selecting a chromosome")
    target = _chromosome(chromosome)
    return np.fromiter((_chromosome(x) == target for x in variants.chromosome),
                       dtype=bool, count=len(variants.position))


def coding_masks(
    variants: VariantAnnotations, gene_name: str, gene_start: int, gene_end: int,
    *, variant_type: str = "SNV", include_ptv: bool = False,
    chromosome: object | None = None,
) -> dict[str, np.ndarray]:
    """Select the five coding masks, with the optional two PTV masks.

    Coding selection uses the supplied gene span, as in the upstream genes_info
    table. It does not reinterpret GENCODE.Info to assign overlapping genes.
    """
    if gene_start > gene_end:
        raise ValueError("gene_start must not exceed gene_end")
    if not gene_name:
        raise ValueError("gene_name must be nonempty")
    base = variant_filter(variants, variant_type) & _on_chromosome(variants, chromosome)
    base &= (variants.position >= gene_start) & (variants.position <= gene_end)
    category = variants.column("GENCODE.Category")
    exonic = variants.column("GENCODE.EXONIC.Category")
    disruptive = (exonic == "nonsynonymous SNV") & (variants.column("MetaSVM") == "D")
    plof = np.isin(exonic, _STOP) | np.isin(category, _SPLICE)
    selected = {
        "plof": plof,
        "plof_ds": plof | disruptive,
        "missense": exonic == "nonsynonymous SNV",
        "disruptive_missense": disruptive,
        "synonymous": exonic == "synonymous SNV",
    }
    if include_ptv:
        snv_ptv = np.isin(exonic, _STOP) | np.isin(category, _CODING_SPLICE)
        indel_ptv = np.isin(exonic, _FRAMESHIFT)
        ptv = snv_ptv if variant_type == "SNV" else indel_ptv if variant_type == "Indel" else snv_ptv | indel_ptv
        selected.update(ptv=ptv, ptv_ds=ptv | disruptive)
    return {name: np.flatnonzero(base & condition) for name, condition in selected.items()}


def promoter_overlaps(
    positions: Sequence, chromosomes: Sequence,
    promoter_intervals: Sequence[Sequence],
) -> np.ndarray:
    """Overlap positions with exact TxDb promoter intervals (chr, start, end).

    Intervals are one-based and inclusive. Export them from the same TxDb used
    for the R comparison: the upstream selector uses the union of all promoters.
    """
    positions = np.asarray(positions, dtype=np.int64)
    chromosomes = np.asarray([_chromosome(x) for x in chromosomes], dtype=object)
    if positions.ndim != 1 or len(positions) != len(chromosomes):
        raise ValueError("positions and chromosomes must be aligned vectors")
    grouped: dict[str, list[tuple[int, int]]] = {}
    for chromosome, start, end in promoter_intervals:
        if int(start) > int(end):
            raise ValueError("promoter interval start exceeds end")
        grouped.setdefault(_chromosome(chromosome), []).append((int(start), int(end)))
    output = np.zeros(len(positions), dtype=bool)
    for chromosome, intervals in grouped.items():
        intervals.sort()
        starts = np.asarray([x[0] for x in intervals], dtype=np.int64)
        ends = np.maximum.accumulate(np.asarray([x[1] for x in intervals], dtype=np.int64))
        rows = np.flatnonzero(chromosomes == chromosome)
        previous = np.searchsorted(starts, positions[rows], side="right") - 1
        valid = previous >= 0
        output[rows[valid]] = positions[rows[valid]] <= ends[previous[valid]]
    return output


def ncRNA_mask(variants: VariantAnnotations, gene_name: str, *,
               variant_type: str = "SNV", chromosome: object | None = None) -> np.ndarray:
    """Select upstream ncRNA categories and the first three named genes."""
    category = variants.column("GENCODE.Category")
    info = variants.column("GENCODE.Info")
    belongs = np.fromiter(
        (gene_name in re.sub(r"\(.*\)", "", text.split(";")[0]).split(",")[:3]
         for text in info), dtype=bool, count=len(info),
    )
    base = variant_filter(variants, variant_type) & _on_chromosome(variants, chromosome)
    base &= np.isin(category, ("ncRNA_exonic", "ncRNA_exonic;splicing", "ncRNA_splicing"))
    return np.flatnonzero(base & belongs)


def noncoding_masks(
    variants: VariantAnnotations, gene_name: str, *, promoter_overlap: Sequence | None = None,
    variant_type: str = "SNV", include_ncrna: bool = False,
    chromosome: object | None = None,
    requested_categories: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Select the seven noncoding masks, optionally adding the ncRNA mask.

    ``promoter_overlap`` must describe membership of every input position in
    the union of the exact TxDb promoter intervals. It must not approximate
    promoters from the coding gene span. ``requested_categories`` can restrict
    selection; promoter intervals are required only when a promoter mask is
    requested. Its default selects all seven masks.
    """
    requested = list(NONCODING_CATEGORIES) if requested_categories is None else list(requested_categories)
    if include_ncrna and "ncRNA" not in requested:
        requested.append("ncRNA")
    allowed = set(NONCODING_CATEGORIES) | {"ncRNA"}
    if any(name not in allowed for name in requested):
        raise ValueError("unknown noncoding category in requested_categories")
    base = variant_filter(variants, variant_type) & _on_chromosome(variants, chromosome)
    category, info = variants.column("GENCODE.Category"), variants.column("GENCODE.Info")
    neighbour = np.fromiter((gene_name in text.split(",") for text in info),
                            dtype=bool, count=len(info))
    utr_gene = np.fromiter((text.split("(")[0] == gene_name for text in info),
                           dtype=bool, count=len(info))
    promoter_gene = np.fromiter((re.split(r"\(|,|;|-", text)[0] == gene_name for text in info),
                                dtype=bool, count=len(info))
    if promoter_overlap is None:
        if any(name.startswith("promoter_") for name in requested):
            raise ValueError("exact promoter_overlap is required for promoter masks")
        promoter_overlap = np.zeros(len(base), dtype=bool)
    promoter_overlap = np.asarray(promoter_overlap, dtype=bool)
    if promoter_overlap.shape != base.shape:
        raise ValueError("promoter_overlap is not aligned with variants")
    genehancer = variants.column("GeneHancer")
    enhancer_gene = np.fromiter(
        ((text.split("=")[3].split(";")[0] if len(text.split("=")) > 3 else "") == gene_name
         for text in genehancer), dtype=bool, count=len(info),
    )
    cage = variants.column("CAGE") != ""
    dhs = variants.column("DHS") != ""
    selected = {
        "upstream": (category == "upstream") & neighbour,
        "downstream": (category == "downstream") & neighbour,
        "UTR": np.isin(category, ("UTR3", "UTR5", "UTR5;UTR3")) & utr_gene,
        "promoter_CAGE": promoter_overlap & promoter_gene & cage,
        "promoter_DHS": promoter_overlap & promoter_gene & dhs,
        "enhancer_CAGE": (genehancer != "") & enhancer_gene & cage,
        "enhancer_DHS": (genehancer != "") & enhancer_gene & dhs,
    }
    output = {name: np.flatnonzero(base & condition) for name, condition in selected.items() if name in requested}
    if "ncRNA" in requested:
        output["ncRNA"] = ncRNA_mask(variants, gene_name, variant_type=variant_type, chromosome=chromosome)
    return output


def annotation_phred_matrix(
    annotations: Mapping[str, Sequence], annotation_names: Sequence[str], *,
    indices: Sequence[int] | None = None, variant_type: str = "SNV",
    use_annotation_weights: bool = True, number_variants: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Construct the upstream ordered PHRED matrix and column names.

    Unknown requested names are skipped as in the R catalog membership check.
    CADD missing scores become zero. LocalDiversity gains the complementary
    PHRED column. Other missing scores are preserved, not silently filled.
    """
    if number_variants is None:
        number_variants = len(next(iter(annotations.values()))) if annotations else 0
    rows = np.arange(number_variants) if indices is None else np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("indices must be a one-dimensional vector")
    if variant_type != "SNV" or not use_annotation_weights:
        return np.empty((len(rows), 0), dtype=np.float64), []
    columns, names = [], []
    for name in annotation_names:
        if name not in annotations:
            continue
        values = np.asarray(annotations[name], dtype=np.float64)[rows].copy()
        if name == "CADD":
            values[np.isnan(values)] = 0
        columns.append(values)
        names.append(name)
        if name == "aPC.LocalDiversity":
            with np.errstate(divide="ignore", invalid="ignore"):
                complement = -10 * np.log10(1 - np.power(10.0, -values / 10))
            columns.append(complement)
            names.append(name + "(-)")
    return (np.column_stack(columns) if columns else np.empty((len(rows), 0))), names


def staar_weights(maf: Sequence, annotation_phred: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return B/S/A weights, beta(1,25) block followed by beta(1,1).

    This function expects the trait-specific frequencies of already selected
    rare variants. It preserves the upstream formulas without renormalisation.
    """
    maf = np.asarray(maf, dtype=np.float64)
    if maf.ndim != 1 or np.any((maf <= 0) | (maf >= 1)):
        raise ValueError("maf must be a vector with values strictly between 0 and 1")
    annotation = np.empty((len(maf), 0)) if annotation_phred is None else np.asarray(annotation_phred, dtype=np.float64)
    if annotation.ndim != 2 or annotation.shape[0] != len(maf):
        raise ValueError("annotation_phred must have one row per variant")
    rank = np.column_stack((np.ones(len(maf)), 1 - np.power(10.0, -annotation / 10)))
    w25, w1 = 25 * np.power(1 - maf, 24), np.ones(len(maf))
    density_inverse_square = np.pi**2 * maf * (1 - maf)
    with np.errstate(invalid="ignore"):
        return {
            "B": np.column_stack((rank * w25[:, None], rank * w1[:, None])),
            "S": np.column_stack((np.sqrt(rank) * w25[:, None], np.sqrt(rank) * w1[:, None])),
            "A": np.column_stack((rank * (w25**2 * density_inverse_square)[:, None],
                                  rank * (w1**2 * density_inverse_square)[:, None])),
        }


def sample_union(sample_lists: Sequence[Sequence]) -> list[str]:
    """R Reduce(union, ...): first-seen order across null model sample IDs."""
    return list(dict.fromkeys(str(sample) for samples in sample_lists for sample in samples))


def sample_indices(samples: Sequence, union_samples: Sequence) -> np.ndarray:
    """Match one null model's order to the shared union, retaining repeat IDs."""
    lookup = {str(sample): index for index, sample in enumerate(union_samples)}
    try:
        return np.asarray([lookup[str(sample)] for sample in samples], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"sample {error.args[0]!r} is absent from the sample union") from error
