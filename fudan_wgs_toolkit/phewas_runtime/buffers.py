"""Independent bounded eligible-column buffers; no disk cache is created."""
from collections import deque
import numpy as np
import torch
from ..genotype_device import DeviceMinorBlock


class EffectiveBuffer:
    def __init__(self, samples, block_size, *, allocation_guard=None):
        self.samples = np.asarray(samples, dtype=np.int64).copy()
        if type(block_size) is not int or block_size < 1:
            raise ValueError('effective block size must be positive')
        if len(self.samples) * block_size > np.iinfo(np.int32).max:
            raise MemoryError('effective dosage exceeds validated int32 geometry')
        self.block_size, self.count, self.parts = block_size, 0, deque()
        self.allocation_guard = allocation_guard

    def append(self, block):
        if not np.array_equal(block.sample_indices, self.samples):
            raise ValueError('buffer block changed the singleton sample order')
        if block.shape[1]:
            self.parts.append(block)
            self.count += block.shape[1]

    def take(self, *, tail=False):
        count = min(self.count, self.block_size)
        if count == 0 or (count < self.block_size and not tail):
            return None
        if self.allocation_guard is not None:
            largest = max(part.shape[1] for part in self.parts)
            self.allocation_guard(len(self.samples) * (count + largest) + 64 * 2**20)
        selected = []
        remaining = count
        while remaining:
            part = self.parts.popleft()
            if part.shape[1] > remaining:
                selected.append(part.select_columns(np.arange(remaining)))
                self.parts.appendleft(part.select_columns(np.arange(remaining, part.shape[1])))
                remaining = 0
            else:
                selected.append(part)
                remaining -= part.shape[1]
        self.count -= count
        if len(selected) == 1:
            return selected[0]
        names = ('union_ref_af', 'union_initial_mac', 'union_missing_rate', 'union_ref_ac', 'union_called_alleles')
        block = DeviceMinorBlock(torch.cat([x.dosage for x in selected], dim=1), self.samples,
            np.concatenate([x.variant_indices for x in selected]),
            *[np.concatenate([getattr(x, name) for x in selected]) for name in names])
        rows = np.arange(len(self.samples))
        counts = [x._cached_counts(rows) for x in selected]
        if all(c is not None for c in counts):
            block._store_counts(rows, np.concatenate([c[0] for c in counts]), np.concatenate([c[1] for c in counts]))
        return block
