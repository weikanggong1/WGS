"""Read-only native SeqArray GDS access, with bounded genotype blocks.

Genotype and annotation conventions follow CoreArray/SeqArray and
STAARpipelinePheWAS.  No R subprocess is used by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .gds_flat import load_flat_reader, flat_reader_metadata

COREARRAY_PYGDS_COMMIT = "b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd"


def _allele_code_dtype(bit_layers: int):
    """Smallest signed dtype retaining every called code and the -1 sentinel."""
    if not 0 <= bit_layers <= 16:
        raise ValueError("Bit2 layer count must be between 0 and 16")
    return np.int16 if bit_layers <= 7 else np.int32 if bit_layers <= 15 else np.int64


def _allele_frequency_summary(reference_ac, called_alleles, n_samples: int):
    """Match frozen SeqArray allele summaries and the R initial-MAC formula."""
    called = np.asarray(called_alleles, dtype=np.int64)
    reference = np.asarray(reference_ac, dtype=np.float64).copy()
    reference[called == 0] = np.nan
    ref_af = np.divide(reference, called, out=np.full(reference.shape, np.nan), where=called > 0)
    total_alleles = 2 * n_samples
    missing_alleles = total_alleles - called
    missing_rate = (missing_alleles / total_alleles if total_alleles else
                    np.full(reference.shape, np.nan))
    # Preserve division, complement, multiplication, and ties-to-even round.
    # With an odd called-allele count this differs from round(called / 2).
    alt_ac = 2 * np.rint(n_samples * (1 - missing_rate)) - reference
    initial_mac = np.where(reference >= alt_ac, alt_ac, reference)
    return ref_af, missing_rate, initial_mac, reference, called


def _indices(values, size: int, label: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= size):
        raise IndexError(f"{label} is outside the GDS dimension")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{label} contains duplicate indices")
    return result


def _factor(values: np.ndarray, attrs: dict) -> np.ndarray:
    classes = np.atleast_1d(attrs.get("R.class", []))
    if "factor" not in classes:
        if values.dtype.kind == "i":
            missing = values == np.iinfo(np.int32).min
            if missing.any():
                values = values.astype(np.float64)
                values[missing] = np.nan
        return values
    # CoreArray pygds already resolves factors when conversion is omitted.
    if values.dtype.kind not in "iu":
        return values
    levels = np.atleast_1d(attrs.get("R.levels", []))
    result = np.empty(values.shape, dtype=object)
    result.fill(None)
    good = (values > 0) & (values <= len(levels))
    result[good] = levels[values[good].astype(np.int64) - 1]
    return result


@dataclass
class SparseMinorBlock:
    """COO genotypes oriented to the minor allele in the union of samples.

    Explicit NaN entries represent missing calls; omitted entries are zero.
    row indices refer to ``sample_indices``; columns to ``variant_indices``.
    The allele orientation remains fixed when selecting individual traits.
    """

    row: np.ndarray
    col: np.ndarray
    value: np.ndarray
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    union_ref_af: np.ndarray
    union_initial_mac: np.ndarray | None = None
    union_missing_rate: np.ndarray | None = None
    union_ref_ac: np.ndarray | None = None
    union_called_alleles: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.sample_indices), len(self.variant_indices)

    def observed_mac(self, trait_rows=None) -> np.ndarray:
        """Sum unfilled called dosages in canonical COO columns, ignoring NaN.

        Reader-produced COO has one entry per sample/site; implicit entries
        are called zero. This integer-valued sum never depends on imputation.
        """
        keep = ~np.isnan(self.value)
        if trait_rows is not None:
            rows = _indices(trait_rows, self.shape[0], "trait_rows")
            members = np.zeros(self.shape[0], dtype=bool)
            members[rows] = True
            keep &= members[self.row]
        return np.bincount(self.col[keep], weights=self.value[keep], minlength=self.shape[1])

    def initial_mac(self) -> np.ndarray:
        """Return source allele-wise MAC, distinct from whole-dosage MAC."""
        if self.union_initial_mac is not None:
            return self.union_initial_mac
        # Backwards-compatible manually constructed complete-diploid COO.
        # A partial allele call requires the explicit source summary fields.
        return self.observed_mac()

    def allele_missing_rate(self) -> np.ndarray:
        """Return source allele missing rates; supplied by every GDS reader."""
        if self.union_missing_rate is not None:
            return self.union_missing_rate
        missing = np.bincount(self.col[np.isnan(self.value)], minlength=self.shape[1])
        return missing / self.shape[0] if self.shape[0] else np.full(self.shape[1], np.nan)

    def select_columns(self, column_indices):
        """Select local variant columns, retaining missing entries and order."""
        columns = _indices(column_indices, self.shape[1], "column_indices")
        mapping = np.full(self.shape[1], -1, dtype=np.int64)
        mapping[columns] = np.arange(len(columns))
        keep = mapping[self.col] >= 0
        summaries = {name: None if getattr(self, name) is None else getattr(self, name)[columns]
                     for name in ("union_initial_mac", "union_missing_rate", "union_ref_ac", "union_called_alleles")}
        return SparseMinorBlock(self.row[keep], mapping[self.col[keep]], self.value[keep],
                                self.sample_indices, self.variant_indices[columns],
                                self.union_ref_af[columns], **summaries)

    def trait_dense(self, trait_rows, imputation: str = "mean", *, frequency_mode: str = "count", dtype=np.float64):
        """Return genotypes, MAF, observed MAC, missing counts, and ALT orientation.

        ``trait_rows`` contains zero-based positions in this block's union
        sample list, in the null model's order.  Mean imputation retains the
        observed-call MAF; minor imputation uses zero and divides MAC by 2N.
        ``dtype`` selects float64 control or direct float32 storage. Frequency
        reductions remain float64; mean fills cast the original double value
        once, matching float64 preparation followed by float32 conversion.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError("dosage dtype must be float64 or float32")
        trait_rows = _indices(trait_rows, self.shape[0], "trait_rows")
        if imputation not in ("mean", "minor"):
            raise ValueError("imputation must be 'mean' or 'minor'")
        if frequency_mode not in ("count", "reference"):
            raise ValueError("frequency_mode must be count or reference")
        if frequency_mode == "reference" and len(trait_rows) != self.shape[0]:
            raise ValueError("reference frequency requires the complete union sample set")
        mapping = np.full(self.shape[0], -1, dtype=np.int64)
        mapping[trait_rows] = np.arange(len(trait_rows))
        keep = mapping[self.row] >= 0
        genotype = np.zeros((len(trait_rows), self.shape[1]), dtype=dtype)
        genotype[mapping[self.row[keep]], self.col[keep]] = self.value[keep]
        missing = np.isnan(genotype)
        mac = np.nansum(genotype, axis=0, dtype=np.float64)
        count = len(trait_rows) - missing.sum(axis=0)
        maf = np.divide(mac, 2 * count, out=np.full_like(mac, np.nan), where=count > 0)
        if frequency_mode == "reference":
            # Base STAARpipeline retains the SeqArray REF_AF subtraction order.
            # PheWAS recomputes each trait frequency from its observed MAC.
            alt_af = 1 - self.union_ref_af
            maf = np.where(self.union_ref_af >= alt_af, alt_af, self.union_ref_af)
            # Both original wrappers select ALT at a union AF tie.
            # The block already retains that global orientation.
        if imputation == "mean":
            row, col = np.nonzero(missing)
            genotype[row, col] = 2 * maf[col]
        else:
            genotype[missing] = 0
            if frequency_mode == "reference":
                restored_mac = np.rint(((2 * maf) * (1 - self.allele_missing_rate())) * len(trait_rows))
                maf = restored_mac / (2 * len(trait_rows)) if len(trait_rows) else restored_mac * np.nan
            else:
                maf = mac / (2 * len(trait_rows)) if len(trait_rows) else mac * 0
        is_alt = self.union_ref_af >= 0.5
        return genotype, maf, mac, missing.sum(axis=0), is_alt

    def to_torch_sparse(self, device="cpu"):
        """Return a float64 COO tensor, retaining explicit missing entries."""
        import torch

        coordinates = torch.as_tensor(np.vstack((self.row, self.col)), device=device)
        values = torch.as_tensor(self.value, dtype=torch.float64, device=device)
        return torch.sparse_coo_tensor(coordinates, values, self.shape).coalesce()


class SeqArrayGDS:
    """Read a native GDS file; all sample/variant indices are zero-based.

    Genotype reads preserve the requested order.  Only selected genotype
    blocks are materialized. packed_reader_directory optionally names a local
    SDK-bound extension build; None keeps the existing reader. Its binding is
    checked before opening data, and configured read errors propagate.
    The format's one-dimensional indices may be
    loaded for decoding; a whole sample-by-variant matrix is never loaded.
    """

    def __init__(self, path: str | Path, *, genotype_raw_memory_bytes: int = 256 * 2**20,
                 genotype_max_gap_layers: int = 8, packed_reader_directory=None):
        if not isinstance(genotype_raw_memory_bytes, int) or genotype_raw_memory_bytes < 1:
            raise ValueError("genotype_raw_memory_bytes must be a positive integer")
        if not isinstance(genotype_max_gap_layers, int) or genotype_max_gap_layers < 0:
            raise ValueError("genotype_max_gap_layers must be a nonnegative integer")
        self.genotype_raw_memory_bytes = genotype_raw_memory_bytes
        self.genotype_max_gap_layers = genotype_max_gap_layers
        self._flat_reader = load_flat_reader()
        from .gds_packed import load_packed_reader
        self._packed_reader = load_packed_reader(packed_reader_directory)
        self._packed_read_failed = False
        try:
            import pygds
        except ImportError as error:
            raise ImportError("Install CoreArray/pygds at the documented commit") from error
        if not hasattr(pygds, "gdsfile"):
            raise ImportError("The installed pygds is not the CoreArray GDS package")
        self._file = pygds.gdsfile()
        self._file.open(str(path), readonly=True)
        self.n_samples = self._dim("sample.id")[0]
        self.n_variants = self._dim("variant.id")[0]
        dims = self._dim("genotype/data")
        if len(dims) != 3 or dims[1] != self.n_samples:
            self.close()
            raise ValueError("Invalid SeqArray genotype dimensions")
        self.ploidy = dims[2]
        if self._packed_reader is not None:
            try:
                # Validate storage, logical stream and axes without reading genotypes.
                self._packed_reader.read_packed_path(self._file.fileid, "genotype/data", 0, 0, self.n_samples)
            except Exception:
                self.close()
                raise
        self._genotype_steps = None
        self._genotype_offsets = None
        self._sample_lookup = None
        self._annotation_indices = {}

    def close(self):
        self._cuda_sample_index_cache = None
        self._packed_cuda_sample_index_cache = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _dim(self, path: str) -> tuple[int, ...]:
        return tuple(self._file.index(path).description()["dim"])

    def describe(self, path: str) -> dict:
        """Return node structure and attributes, without genotype payloads."""
        node = self._file.index(path)
        return {"description": node.description(), "attributes": node.getattr() or {}}

    def sample_ids(self) -> np.ndarray:
        return np.asarray(self._file.index("sample.id").read()).astype(str)

    def sample_indices(self, sample_ids: Sequence[str]) -> np.ndarray:
        """Map sample IDs into file indices, retaining the requested order."""
        if self._sample_lookup is None:
            ids = self.sample_ids()
            if len(set(ids)) != len(ids):
                raise ValueError("GDS sample IDs are not unique")
            self._sample_lookup = {value: i for i, value in enumerate(ids)}
        requested = list(map(str, sample_ids))
        if len(set(requested)) != len(requested):
            raise ValueError("Requested sample IDs are not unique")
        try:
            return np.asarray([self._sample_lookup[x] for x in requested], dtype=np.int64)
        except KeyError as error:
            raise ValueError("A requested sample ID is absent from the GDS") from None

    def _read_axis(self, node, indices: np.ndarray, size: int, axis=0):
        dims = tuple(node.description()["dim"])
        if len(indices)==0:
            shape=list(dims);shape[axis]=0
            return np.empty(shape)
        if len(indices)==1 or np.all(np.diff(indices)==1):
            start=[0]*len(dims);count=list(dims)
            start[axis]=int(indices[0]);count[axis]=len(indices)
            return np.asarray(node.read(start,count))
        # pygds readex accepts full-axis masks and scans that axis on every
        # call. Bound sparse metadata reads instead; no genotype is read here.
        order = np.argsort(indices)
        sorted_indices = indices[order]
        pieces = []
        offset = 0
        while offset < len(sorted_indices):
            boundary = np.searchsorted(sorted_indices, sorted_indices[offset] + 65_536)
            rows = sorted_indices[offset:boundary]
            start = [0] * len(dims); count = list(dims)
            start[axis] = int(rows[0]); count[axis] = int(rows[-1] - rows[0] + 1)
            values = np.asarray(node.read(start, count))
            pieces.append(np.take(values, rows - rows[0], axis=axis))
            offset = boundary
        result = np.concatenate(pieces, axis=axis)
        return np.take(result, np.argsort(order), axis=axis)

    def read_ref_alt(self, variant_indices):
        """Read the allele node once; preserve complete comma-delimited ALT."""
        alleles = self.read_field("allele", variant_indices)
        ref = np.empty(len(alleles), dtype=object)
        alt = np.empty(len(alleles), dtype=object)
        for index, value in enumerate(alleles):
            value = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            ref[index], _, alt[index] = value.partition(",")
        return ref, alt

    def read_field(self, path: str, variant_indices=None):
        """Read aligned variant fields, decoding factors and variable lengths.

        Fixed fields return arrays with variant on axis 0.  Variable-length
        fields return a list of arrays, one per requested variant; zero
        lengths return empty arrays.  REF/ALT preserve all comma-delimited
        alternate alleles, rather than splitting one site into new sites.
        """
        if variant_indices is None:
            variant_indices = np.arange(self.n_variants, dtype=np.int64)
        indices = _indices(variant_indices, self.n_variants, "variant_indices")
        if path in ("$ref", "$alt", "$num_allele"):
            if path in ("$ref", "$alt"):
                ref, alt = self.read_ref_alt(indices)
                return ref if path == "$ref" else alt
            alleles = self.read_field("allele", indices)
            return np.asarray([x.count(",") + 1 for x in alleles], dtype=np.int64)
        node = self._file.index(path)
        dims = tuple(node.description()["dim"] or ())
        if not dims:
            raise ValueError("A field must refer to an array node, not a folder")
        parent, _, name = path.rpartition("/")
        if path.startswith("annotation/info/") and parent:
            if path not in self._annotation_indices:
                index_node=self._file.index(f"{parent}/@{name}",silent=True)
                cached=None
                if index_node is not None:
                    lengths=np.asarray(index_node.read(),dtype=np.int64)
                    if lengths.shape!=(self.n_variants,) or np.any(lengths<0):
                        raise ValueError("Invalid variable-length annotation index")
                    if np.all(lengths==1):
                        cached=("unit",)
                    else:
                        offsets=np.concatenate(([0],np.cumsum(lengths,dtype=np.int64)))
                        if offsets[-1]!=dims[0]:
                            raise ValueError("Annotation index does not match the data dimension")
                        cached=("variable",lengths,offsets)
                self._annotation_indices[path]=cached
            cached=self._annotation_indices[path]
            if cached is not None:
                if cached[0]=="unit":
                    return _factor(self._read_axis(node,indices,self.n_variants),node.getattr() or {})
                _,lengths,offsets=cached
                result=[];attrs=node.getattr() or {}
                for index in indices:
                    start=[int(offsets[index])]+[0]*(len(dims)-1)
                    count=[int(lengths[index])]+list(dims[1:])
                    value=np.asarray(node.read(start,count)) if count[0] else np.empty((0,*dims[1:]))
                    result.append(_factor(value,attrs))
                return result
        if dims[0] != self.n_variants:
            raise ValueError("Field's leading dimension is not the variant count")
        return _factor(self._read_axis(node, indices, self.n_variants), node.getattr() or {})

    def _prepare_genotype_index(self):
        if self._genotype_steps is not None:
            return
        node = self._file.index("genotype/@data", silent=True)
        raw_count = self._dim("genotype/data")[0]
        if node is None:
            if raw_count != self.n_variants:
                raise ValueError("A variable-layer genotype array needs genotype/@data")
            steps = np.ones(self.n_variants, dtype=np.uint8)
        else:
            steps = np.asarray(node.read())
        if steps.shape != (self.n_variants,) or np.any(steps > 16):
            raise ValueError("Invalid genotype bit-layer index")
        offsets = np.concatenate(([0], np.cumsum(steps, dtype=np.int64)))
        if offsets[-1] != raw_count:
            raise ValueError("Genotype bit-layer index does not match genotype/data")
        self._genotype_steps, self._genotype_offsets = steps, offsets

    @property
    def reader_metadata(self) -> dict:
        """Report the actual I/O backend, binary checksum, and read counters."""
        metadata = flat_reader_metadata(self._flat_reader)
        if getattr(self, "_packed_reader", None) is not None:
            packed_metadata = dict(self._packed_reader._packed_metadata)
            packed_backend = packed_metadata.pop("reader_backend")
            metadata.update(packed_metadata)
            metadata["packed_reader_configured"] = True
            if getattr(self, "_reader_io_counts", {}).get("packed", {}).get("calls", 0):
                metadata["reader_backend"] = packed_backend
        metadata["genotype_raw_memory_bytes"] = self.genotype_raw_memory_bytes
        metadata["genotype_max_gap_layers"] = self.genotype_max_gap_layers
        counts = getattr(self, "_minor_decode_counts", {})
        used = [name for name, count in counts.items() if count]
        metadata["minor_genotype_decode_backend"] = used[0] if len(used) == 1 else "mixed" if used else "not_used"
        metadata["minor_genotype_decode_calls"] = dict(counts)
        metadata["minor_genotype_decode_fallback_reason"] = getattr(self, "_minor_decode_fallback_reason", None)
        metadata["allele_code_dtype_policy"] = "int16: 0-7 layers; int32: 8-15; int64: 16"
        if self._flat_reader is not None or getattr(self, "_packed_reader", None) is not None:
            if getattr(self, "_packed_reader", None) is None:
                metadata["reader_backend"] = "native_auto"
            metadata["native_reads"] = {name: dict(record) for name, record in
                                        getattr(self, "_reader_io_counts", {}).items()}
        metadata["individual_decode_coverage"] = dict(getattr(self, "_minor_decode_coverage", {}))
        return metadata

    def _record_minor_coverage(self, steps, summaries, half_missing: int, eligible: int):
        """Record aggregate integer-decoding coverage, without individual data."""
        counts = getattr(self, "_minor_decode_coverage", None)
        if counts is None:
            counts = self._minor_decode_coverage = {
                "decoded_variants": 0, "mac_eligible_variants": 0,
                "half_missing_genotypes": 0, "ref_af_tie_variants": 0,
                "all_missing_variants": 0, "max_bit2_layers": 0}
        counts["decoded_variants"] += len(steps)
        counts["mac_eligible_variants"] += eligible
        counts["half_missing_genotypes"] += half_missing
        counts["ref_af_tie_variants"] += int(np.count_nonzero(summaries[0] == 0.5))
        counts["all_missing_variants"] += int(np.count_nonzero(summaries[4] == 0))
        counts["max_bit2_layers"] = max(counts["max_bit2_layers"], int(steps.max()) if len(steps) else 0)

    def _sample_selection(self, samples):
        """Reuse one host SDK mask/permutation across consecutive bounded blocks.

        Cache owns its indices and never relies on mutable caller arrays. One
        entry bounds storage and changing the union evicts the old selection.
        """
        cached = getattr(self, "_sample_selection_cache", None)
        if cached is None or not np.array_equal(cached[0], samples):
            mask = np.zeros(self.n_samples, dtype=bool)
            mask[samples] = True
            cached = (samples.copy(), np.repeat(mask, self.ploidy), np.argsort(np.argsort(samples)))
            self._sample_selection_cache = cached
        return cached[1], cached[2]

    def read_genotype(self, variant_indices, sample_indices) -> np.ndarray:
        """Return [variant, sample, ploidy] signed allele codes; missing is -1.

        The optional SDK adapter reads local Bit2 slabs within
        ``genotype_raw_memory_bytes`` of temporary decoding storage. Slabs
        with at least two selected sites and fewer than half the file's
        samples use native row selection; other slabs use continuous reads.
        Sample and variant order are retained. Missing adapter installations
        use the official pygds rectangular reader; read errors propagate.
        """
        if self._flat_reader is None:
            return self._read_genotype_readex(variant_indices, sample_indices)
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        samples = _indices(sample_indices, self.n_samples, "sample_indices")
        if len(variants) == 0 or len(samples) == 0:
            return np.empty((len(variants), len(samples), self.ploidy), dtype=np.int64)
        self._prepare_genotype_index()
        bytes_per_layer = self.n_samples * self.ploidy
        selected_cells = len(samples) * self.ploidy
        # Reserve optional masks/permutation and shifted uint32 operands.
        # Full-sample uint8 slabs bound either native route conservatively.
        # The final signed int64 output is separate from this decoding bound.
        mask_bytes = bytes_per_layer + self.n_samples + 16 * len(samples)
        max_layers = (self.genotype_raw_memory_bytes - mask_bytes - 12 * selected_cells) // (
            bytes_per_layer + 4 * selected_cells)
        if max_layers < 1:
            raise MemoryError("one genotype Bit2 layer exceeds the raw read budget")
        variant_order = np.argsort(variants)
        sorted_variants = variants[variant_order]
        result = np.full((len(variants), len(samples), self.ploidy), -1,
                         dtype=_allele_code_dtype(int(self._genotype_steps[variants].max())))
        selected = [(int(variant_order[output]), int(self._genotype_offsets[index]),
                     int(self._genotype_offsets[index + 1]))
                    for output, index in enumerate(sorted_variants)
                    if self._genotype_steps[index]]
        groups = []
        for entry in selected:
            if (groups and entry[1] - groups[-1][-1][2] <= self.genotype_max_gap_layers
                    and entry[2] - groups[-1][0][1] <= max_layers):
                groups[-1].append(entry)
            else:
                groups.append([entry])
        selection = sample_order = None
        if len(samples) * 2 < self.n_samples and any(len(group) >= 2 for group in groups):
            selection, sample_order = self._sample_selection(samples)
        counters = getattr(self, "_reader_io_counts", None)
        if counters is None:
            counters = self._reader_io_counts = {name: {"calls": 0, "returned_bytes": 0}
                                                for name in ("flat", "selected")}
        for group in groups:
            use_selected = len(group) >= 2 and selection is not None
            route = "selected" if use_selected else "flat"
            raw_start, raw_end = group[0][1], group[-1][2]
            values = {output: np.zeros((len(samples), self.ploidy), dtype=np.uint32)
                      for output, _, _ in group}
            # A 16-layer site is split under a smaller budget. Reconstruct
            # into uint32, assign to signed int64, then replace the missing
            # all-bits-set sentinel. High called alleles never wrap negative.
            for lower in range(raw_start, raw_end, max_layers):
                upper = min(raw_end, lower + max_layers)
                if use_selected:
                    raw = self._flat_reader.read_selected_rows_path(
                        self._file.fileid, "genotype/data", lower * bytes_per_layer,
                        upper - lower, bytes_per_layer, selection, "uint8")
                    raw = np.asarray(raw, dtype=np.uint8).reshape(
                        upper - lower, len(samples), self.ploidy)
                else:
                    raw = self._flat_reader.read_flat_path(
                        self._file.fileid, "genotype/data", lower * bytes_per_layer,
                        (upper - lower) * bytes_per_layer, "uint8")
                    raw = np.asarray(raw, dtype=np.uint8).reshape(
                        upper - lower, self.n_samples, self.ploidy)
                counters[route]["calls"] += 1
                counters[route]["returned_bytes"] += raw.nbytes
                for output, first, last in group:
                    for layer in range(max(first, lower), min(last, upper)):
                        value = raw[layer - lower] if use_selected else raw[layer - lower, samples, :]
                        values[output] |= value.astype(np.uint32) << (2 * (layer - first))
                del raw
            for output, first, last in group:
                value = values[output][sample_order] if use_selected else values[output]
                result[output] = value
                result[output][value == (1 << (2 * (last - first))) - 1] = -1
        return result

    def _read_genotype_readex(self, variant_indices, sample_indices) -> np.ndarray:
        """Return allele codes as [variant, sample, ploidy]; missing is -1.

        Multiple Bit2 layers are reconstructed before interpreting missing
        codes, so allele code 3 is retained at sites with multiple layers.
        """
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        samples = _indices(sample_indices, self.n_samples, "sample_indices")
        if len(variants) == 0 or len(samples) == 0:
            return np.empty((len(variants), len(samples), self.ploidy), dtype=np.int64)
        self._prepare_genotype_index()
        sorted_variants = np.sort(variants)
        raw_mask = np.zeros(int(self._genotype_offsets[-1]), dtype=bool)
        for index in sorted_variants:
            raw_mask[self._genotype_offsets[index]:self._genotype_offsets[index + 1]] = True
        sample_mask = np.zeros(self.n_samples, dtype=bool)
        sample_mask[samples] = True
        raw = np.asarray(self._file.index("genotype/data").readex([raw_mask, sample_mask, None]), dtype=np.uint32)
        result = np.full((len(variants), len(samples), self.ploidy), -1,
                         dtype=_allele_code_dtype(int(self._genotype_steps[variants].max())))
        cursor = 0
        for output_index, index in enumerate(sorted_variants):
            steps = int(self._genotype_steps[index])
            if steps:
                value = np.zeros((len(samples), self.ploidy), dtype=np.uint32)
                for layer in range(steps):
                    value |= raw[cursor + layer] << (2 * layer)
                missing_code = (1 << (2 * steps)) - 1
                result[output_index] = np.where(value == missing_code, -1, value)
                cursor += steps
        result = result[np.argsort(np.argsort(variants))]
        return result[:, np.argsort(np.argsort(samples))]

    def read_ref_dosage(self, variant_indices, sample_indices) -> np.ndarray:
        """Return [sample, variant] REF-copy dosage, with missing calls as NaN."""
        genotype = self.read_genotype(variant_indices, sample_indices)
        missing = (genotype < 0).any(axis=2)
        dosage = (genotype == 0).sum(axis=2).astype(np.float64)
        dosage[missing] = np.nan
        return dosage.T

    def minor_block(self, variant_indices, union_sample_indices, *, device=None, minimum_mac=None, resident=False) -> SparseMinorBlock:
        """Read minor dosages with source allele-wise AF/MAC/missing summaries.

        The optional CUDA I/O path performs only integer decoding/counting;
        source floating-point frequency calculations remain NumPy float64.
        Mean/minor imputation occurs later in each null model's sample rows.
        ``resident=True`` returns a compact DeviceMinorBlock on CUDA, retaining
        dosage there through trait selection; metadata remains on the host.
        """
        if self.ploidy != 2:
            raise ValueError("STAARpipelinePheWAS requires diploid genotype dosage")
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        samples = _indices(union_sample_indices, self.n_samples, "union_sample_indices")
        if minimum_mac is not None and not np.isfinite(minimum_mac):
            raise ValueError("minimum_mac must be finite")
        use_cuda = device is not None and str(device).startswith("cuda") and (self._flat_reader is not None or getattr(self, "_packed_reader", None) is not None)
        if resident and not use_cuda:
            raise ValueError("resident dosage requires CUDA and the native SDK adapter")
        if use_cuda:
            from .gds_cuda import native_minor_block
            block = native_minor_block(self, variants, samples, device=device, minimum_mac=minimum_mac, resident=resident)
            backend = "cuda"
        else:
            if device is not None and str(device).startswith("cuda") and self._flat_reader is None:
                self._minor_decode_fallback_reason = "native SDK adapter is not installed"
            genotype = self.read_genotype(variants, samples)
            reference = genotype == 0
            reference_ac = reference.sum(axis=(1, 2), dtype=np.int64)
            called_alleles = (genotype >= 0).sum(axis=(1, 2), dtype=np.int64)
            summaries = _allele_frequency_summary(reference_ac, called_alleles, len(samples))
            if minimum_mac is not None:
                half_missing = int(np.count_nonzero((genotype < 0).sum(axis=2) == 1))
                eligible = len(variants) if minimum_mac is None else int(np.count_nonzero(summaries[2] >= minimum_mac))
                self._record_minor_coverage(self._genotype_steps[variants], summaries, half_missing, eligible)
            dosage = reference.sum(axis=2).astype(np.float64)
            dosage[(genotype < 0).any(axis=2)] = np.nan
            dosage = dosage.T
            del genotype, reference
            ref_af, missing_rate, initial_mac, ref_ac, called = summaries
            if minimum_mac is not None:
                keep = np.flatnonzero(initial_mac >= minimum_mac)
                variants, dosage = variants[keep], dosage[:, keep]
                ref_af, missing_rate, initial_mac, ref_ac, called = [value[keep] for value in summaries]
            flip = ref_af >= 0.5
            dosage[:, flip] = 2 - dosage[:, flip]
            row, col = np.nonzero((dosage != 0) | np.isnan(dosage))
            block = SparseMinorBlock(row, col, dosage[row, col], samples, variants, ref_af,
                initial_mac, missing_rate, ref_ac, called)
            backend = "cpu"
        counters = getattr(self, "_minor_decode_counts", None)
        if counters is None:
            counters = self._minor_decode_counts = {"cpu": 0, "cuda": 0}
        counters[backend] += 1
        return block

    def iter_minor_blocks(self, variant_indices, union_sample_indices, block_size=256, *, device=None, minimum_mac=None, resident=False) -> Iterator[SparseMinorBlock]:
        """Read bounded blocks; optional early MAC filtering preserves site order."""
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        for start in range(0, len(variants), block_size):
            yield self.minor_block(variants[start:start + block_size], union_sample_indices,
                                  device=device, minimum_mac=minimum_mac, resident=resident)


def main():
    """Inspect dimensions and a node's structure, without printing payloads."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect a native SeqArray GDS file")
    parser.add_argument("--gds", required=True, help="Input native GDS file")
    parser.add_argument("--node", help="Optional node whose structure is inspected")
    args = parser.parse_args()
    with SeqArrayGDS(args.gds) as reader:
        result = {"samples": reader.n_samples, "variants": reader.n_variants, "ploidy": reader.ploidy}
        if args.node:
            result["node"] = reader.describe(args.node)
        print(json.dumps(result, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))


if __name__ == "__main__":
    main()
