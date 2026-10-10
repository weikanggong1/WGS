"""Explicit eligible-M limit shared by standalone and PheWAS runs."""
from contextlib import contextmanager
import numpy as np
import torch
from ..null_model import GaussianNullModel
from ..pipeline import PheWASPipeline


class LimitedMaskPipeline(PheWASPipeline):
    maximum_mask_variants = None

    @contextmanager
    def _resident_workspace(self, prepared, *, host=False):
        previous = getattr(self, "_bounded_resident_workspace", False)
        self._bounded_resident_workspace = host or "_resident_blocks" in prepared
        try:
            yield
        finally:
            self._bounded_resident_workspace = previous

    def _materialize_resident_gene(self, prepared, columns=None):
        with self._resident_workspace(prepared):
            return super()._materialize_resident_gene(prepared, columns)

    def _calculate_local_union(self, index_sets, prepared):
        with self._resident_workspace(prepared):
            return super()._calculate_local_union(index_sets, prepared)

    def _hybrid_host_products(self, model, host):
        # The mature hybrid reader can release a large uint8 local union to
        # host storage while all its individual masks remain below the limit.
        # Those small masks use the same two-G native covariance phases as
        # resident masks; no long-mask backend or precision choice changes.
        bounded_host = not isinstance(host, torch.Tensor) or not host.requires_grad
        with self._resident_workspace({}, host=bounded_host):
            return super()._hybrid_host_products(model, host)

    def _workspace_estimate(self, model, number_variants, *, individual=False):
        estimate = super()._workspace_estimate(model, number_variants, individual=individual)
        m = int(number_variants)
        tensors = ("x", "precision_x", "inverse_variance", "fixed_effect_covariance", "scaled_residuals")
        if (individual or not getattr(self, "_bounded_resident_workspace", False)
                or not 0 < m <= 5000
                or self.options.sample_block_size is not None
                or m > self.options.long_mask_threshold
                or not isinstance(model, GaussianNullModel) or model.n_pheno != 1
                or model.matmul_mode != "tf32" or model.spectrum.blocks
                or any(getattr(getattr(model, name, None), "requires_grad", True)
                       or getattr(getattr(model, name, None), "dtype", None) != torch.float32
                       for name in tensors)):
            return estimate
        # With diagonal precision rotate(G) aliases G. The inline Sigma_iG
        # operand dies after its covariance product, before the final Score
        # GEMV can need a layout copy. These are separate two-G phases.
        # Materialization also holds one uint8 selection, a missing mask and
        # one FP32 block. Cover its separate peak, retaining the original
        # covariance/model allowance and all live allocator/product guards.
        n = model.n
        block_columns = min(m, self.options.genotype_block_size)
        genotype_bytes = max(8 * n * m, 4 * n * m + 6 * n * block_columns)
        return estimate - 16 * n * m + genotype_bytes

    def _run_mask_sets(self, index_sets):
        sets = list(index_sets)
        limit = self.maximum_mask_variants
        if limit is None:
            return super()._run_mask_sets(sets)
        if type(limit) is not int or limit < 1 or len(self.models) != 1:
            raise ValueError('maximum_mask_variants requires a positive integer and singleton pipeline')
        kept, positions = [], []
        output = [[None] for _ in sets]
        for number, indices in enumerate(sets):
            indices = np.asarray(indices, dtype=np.int64)
            count = 0
            if len(indices) > limit:
                for block in self._minor_blocks(indices, self.union_rows, block_size=self.options.genotype_block_size,
                                                device=self.models[0].device, resident=True):
                    alt = 1 - block.union_ref_af
                    source = np.where(block.union_ref_af >= alt, alt, block.union_ref_af)
                    frequency = block.trait_summary(self.trait_rows[0], self.options.imputation,
                                                    frequency_mode='reference')[0]
                    count += int(np.count_nonzero(np.isfinite(source) & (source > 0) &
                        (source < self.options.rare_maf_cutoff) & np.isfinite(frequency) &
                        (frequency > 0) & (frequency < self.options.rare_maf_cutoff)))
                    del block
                    if count > limit:
                        break
            if count > limit:
                self.skipped_sets.append({'mask_position': number,
                    'eligible_variants_lower_bound': count, 'reason': 'eligible_M_exceeds_configured_limit'})
            else:
                kept.append(indices)
                positions.append(number)
        if kept:
            for number, result in zip(positions, super()._run_mask_sets(kept)):
                output[number] = result
        return output
