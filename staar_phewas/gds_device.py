"""Device-resident lossless minor dosage blocks for bounded Single scans.

Only small allele summaries cross to the host. The uint8 matrix uses 3 for
whole-genotype missingness and otherwise contains minor-copy counts 0..2.
Source AF/MAC and trait frequency formulas retain NumPy float64 operation order.
"""
from dataclasses import dataclass, field
import numpy as np
from .gds import _indices


@dataclass
class DeviceMinorBlock:
    dosage: object
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    union_ref_af: np.ndarray
    union_initial_mac: np.ndarray
    union_missing_rate: np.ndarray
    union_ref_ac: np.ndarray
    union_called_alleles: np.ndarray
    _counts_cache: object = field(default=None, init=False, repr=False)

    @property
    def shape(self):
        return len(self.sample_indices), len(self.variant_indices)

    def initial_mac(self):
        return self.union_initial_mac

    def allele_missing_rate(self):
        return self.union_missing_rate

    def _dosage_stamp(self):
        # Normal Torch in-place writes invalidate cached counts. Inference
        # tensors without version counters conservatively bypass caching.
        try:
            return (id(self.dosage), self.dosage._version, tuple(self.dosage.shape))
        except RuntimeError:
            return None

    def _cached_counts(self, rows):
        stamp = self._dosage_stamp()
        cached = self._counts_cache
        if stamp is not None and cached is not None and stamp == cached[0] and np.array_equal(rows, cached[1]):
            return cached[2], cached[3]
        self._counts_cache = None
        return None

    def _store_counts(self, rows, mac, missing_count):
        stamp = self._dosage_stamp()
        if stamp is not None:
            # Only small host summaries are cached, never selected CUDA
            # dosage/missing matrices. Public return arrays remain independent.
            self._counts_cache = (stamp, rows.copy(), mac.copy(), missing_count.copy())

    def _trait_counts(self, trait_rows):
        import torch
        rows = _indices(trait_rows, self.shape[0], "trait_rows")
        identity = len(rows) == self.shape[0] and np.array_equal(rows, np.arange(self.shape[0]))
        selected = self.dosage if identity else self.dosage.index_select(0, torch.as_tensor(rows, device=self.dosage.device))
        missing = selected == 3
        cached = self._cached_counts(rows)
        if cached is None:
            called = selected.masked_fill(missing, 0)
            # Count whole-genotype missingness from dosage, not allele AF/AC/AN.
            summary = torch.stack((called.sum(0, dtype=torch.int64),
                                   missing.sum(0, dtype=torch.int64))).cpu().numpy()
            mac, missing_count = summary[0].astype(np.float64), summary[1]
            self._store_counts(rows, mac, missing_count)
        else:
            mac, missing_count = cached
        return rows, selected, missing, mac.copy(), missing_count.copy()

    def observed_mac(self, trait_rows=None):
        rows = _indices(np.arange(self.shape[0]) if trait_rows is None else trait_rows,
                        self.shape[0], "trait_rows")
        cached = self._cached_counts(rows)
        return cached[0].copy() if cached is not None else self._trait_counts(rows)[3]

    def select_columns(self, column_indices):
        """Select in caller order; an identity selection shares this block.

        Lossless dosage is read-only during analysis. Normal in-place changes
        invalidate its count cache; returned host summary arrays are copies.
        """
        import torch
        columns = _indices(column_indices, self.shape[1], "column_indices")
        if len(columns) == self.shape[1] and np.array_equal(columns, np.arange(self.shape[1])):
            return self
        dosage = self.dosage.index_select(1, torch.as_tensor(columns, device=self.dosage.device))
        result = DeviceMinorBlock(dosage, self.sample_indices, self.variant_indices[columns],
            *[getattr(self, name)[columns] for name in ("union_ref_af", "union_initial_mac",
               "union_missing_rate", "union_ref_ac", "union_called_alleles")])
        cached = self._counts_cache
        if cached is not None and cached[0] == self._dosage_stamp():
            result._store_counts(cached[1], cached[2][columns], cached[3][columns])
        return result

    def _trait_frequency(self, rows, mac, missing_count, imputation, frequency_mode):
        count = len(rows) - missing_count
        maf = np.divide(mac, 2 * count, out=np.full_like(mac, np.nan), where=count > 0)
        if frequency_mode == "reference":
            alt_af = 1 - self.union_ref_af
            maf = np.where(self.union_ref_af >= alt_af, alt_af, self.union_ref_af)
        if imputation == "minor":
            if frequency_mode == "reference":
                restored_mac = np.rint(((2 * maf) * (1 - self.allele_missing_rate())) * len(rows))
                maf = restored_mac / (2 * len(rows)) if len(rows) else restored_mac * np.nan
            else:
                maf = mac / (2 * len(rows)) if len(rows) else mac * 0
        return maf

    def _trait_rows(self, trait_rows, imputation, frequency_mode):
        if imputation not in ("mean", "minor"):
            raise ValueError("imputation must be 'mean' or 'minor'")
        if frequency_mode not in ("count", "reference"):
            raise ValueError("frequency_mode must be count or reference")
        rows = _indices(trait_rows, self.shape[0], "trait_rows")
        if frequency_mode == "reference" and len(rows) != self.shape[0]:
            raise ValueError("reference frequency requires the complete union sample set")
        return rows

    def trait_summary(self, trait_rows, imputation="mean", *, frequency_mode="count"):
        """Return original host summaries without allocating floating dosage.

        Counts follow the same integer reductions and cache as trait_dense.
        Temporary row selections/missing masks are released on return.
        """
        rows = self._trait_rows(trait_rows, imputation, frequency_mode)
        cached = self._cached_counts(rows)
        if cached is None:
            _, selected, missing, mac, missing_count = self._trait_counts(rows)
            del selected, missing
        else:
            mac, missing_count = [value.copy() for value in cached]
        maf = self._trait_frequency(rows, mac, missing_count, imputation, frequency_mode)
        return maf, mac, missing_count, self.union_ref_af >= 0.5

    def trait_dense(self, trait_rows, imputation="mean", *, frequency_mode="count", dtype=None):
        """Return device dosage and original host frequency/count summaries.

        FP32 fills the original double mean cast once. No sample-by-variant
        payload is sent to CPU. float16/BF16 are rejected.
        """
        import torch
        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("dosage dtype must be torch.float64 or torch.float32")
        rows = self._trait_rows(trait_rows, imputation, frequency_mode)
        rows, selected, missing, mac, missing_count = self._trait_counts(rows)
        maf = self._trait_frequency(rows, mac, missing_count, imputation, frequency_mode)
        genotype = selected.to(dtype=dtype, copy=True)
        if imputation == "mean":
            means = torch.as_tensor(2 * maf, dtype=dtype, device=genotype.device)
            torch.where(missing, means[None, :], genotype, out=genotype)
        else:
            genotype.masked_fill_(missing, 0)
        return genotype, maf, mac, missing_count, self.union_ref_af >= 0.5
