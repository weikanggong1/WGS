"""Read-only native SeqArray GDS access, with bounded genotype blocks.

Genotype and annotation conventions follow CoreArray/SeqArray and
STAARpipelinePheWAS.  No R subprocess is used by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

COREARRAY_PYGDS_COMMIT = "b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd"


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

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.sample_indices), len(self.variant_indices)

    def trait_dense(self, trait_rows, imputation: str = "mean"):
        """Return genotypes, MAF, observed MAC, missing counts, and ALT orientation.

        ``trait_rows`` contains zero-based positions in this block's union
        sample list, in the null model's order.  Mean imputation retains the
        observed-call MAF; minor imputation uses zero and divides MAC by 2N.
        """
        trait_rows = _indices(trait_rows, self.shape[0], "trait_rows")
        if imputation not in ("mean", "minor"):
            raise ValueError("imputation must be 'mean' or 'minor'")
        mapping = np.full(self.shape[0], -1, dtype=np.int64)
        mapping[trait_rows] = np.arange(len(trait_rows))
        keep = mapping[self.row] >= 0
        genotype = np.zeros((len(trait_rows), self.shape[1]), dtype=np.float64)
        genotype[mapping[self.row[keep]], self.col[keep]] = self.value[keep]
        missing = np.isnan(genotype)
        mac = np.nansum(genotype, axis=0)
        count = len(trait_rows) - missing.sum(axis=0)
        maf = np.divide(mac, 2 * count, out=np.full_like(mac, np.nan), where=count > 0)
        if imputation == "mean":
            row, col = np.nonzero(missing)
            genotype[row, col] = 2 * maf[col]
        else:
            genotype[missing] = 0
            maf = mac / (2 * len(trait_rows)) if len(trait_rows) else mac * 0
        return genotype, maf, mac, missing.sum(axis=0), self.union_ref_af > 0.5

    def to_torch_sparse(self, device="cpu"):
        """Return a float64 COO tensor, retaining explicit missing entries."""
        import torch

        coordinates = torch.as_tensor(np.vstack((self.row, self.col)), device=device)
        values = torch.as_tensor(self.value, dtype=torch.float64, device=device)
        return torch.sparse_coo_tensor(coordinates, values, self.shape).coalesce()


class SeqArrayGDS:
    """Read a native GDS file; all sample/variant indices are zero-based.

    Genotype reads preserve the requested order.  Only selected genotype
    blocks are materialized.  The format's one-dimensional indices may be
    loaded for decoding; a whole sample-by-variant matrix is never loaded.
    """

    def __init__(self, path: str | Path):
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
        self._genotype_steps = None
        self._genotype_offsets = None
        self._sample_lookup = None
        self._annotation_indices = {}

    def close(self):
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
        selection = np.zeros(size, dtype=bool)
        selection[indices] = True
        masks = [None] * len(dims)
        masks[axis] = selection
        result = np.asarray(node.readex(masks))
        return np.take(result, np.argsort(np.argsort(indices)), axis=axis)

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
            alleles = self.read_field("allele", indices).astype(str)
            if path == "$ref":
                return np.asarray([x.split(",", 1)[0] for x in alleles])
            if path == "$alt":
                return np.asarray([x.partition(",")[2] for x in alleles])
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

    def read_genotype(self, variant_indices, sample_indices) -> np.ndarray:
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
        result = np.full((len(variants), len(samples), self.ploidy), -1, dtype=np.int64)
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

    def minor_block(self, variant_indices, union_sample_indices) -> SparseMinorBlock:
        """Extract sparse minor genotypes, oriented once on all union samples.

        Calls remain missing until each trait selects its own rows.  This is
        the order used by STAARpipelinePheWAS Genotype_sp_extraction.
        """
        if self.ploidy != 2:
            raise ValueError("STAARpipelinePheWAS requires diploid genotype dosage")
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        samples = _indices(union_sample_indices, self.n_samples, "union_sample_indices")
        dosage = self.read_ref_dosage(variants, samples)
        observed = np.isfinite(dosage)
        ref_ac = np.nansum(dosage, axis=0)
        counts = observed.sum(axis=0)
        ref_af = np.divide(ref_ac, 2 * counts, out=np.full(len(variants), np.nan), where=counts > 0)
        flip = ref_af > 0.5
        dosage[:, flip] = 2 - dosage[:, flip]
        row, col = np.nonzero((dosage != 0) | ~observed)
        return SparseMinorBlock(row, col, dosage[row, col], samples, variants, ref_af)

    def iter_minor_blocks(self, variant_indices, union_sample_indices, block_size=256) -> Iterator[SparseMinorBlock]:
        """Extract bounded sparse blocks in requested variant order."""
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        variants = _indices(variant_indices, self.n_variants, "variant_indices")
        for start in range(0, len(variants), block_size):
            yield self.minor_block(variants[start:start + block_size], union_sample_indices)


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
