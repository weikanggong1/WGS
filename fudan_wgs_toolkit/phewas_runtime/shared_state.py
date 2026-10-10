"""Read-only shared six-state cache transport for independent phenotypes.

The physical cache keeps partial allele calls. A phenotype obtains its own
source AF/MAC, orientation and ordered sample axis before the existing Single
or gene statistic receives a DeviceMinorBlock. CPU mode is a small contract
oracle; production CUDA uses bounded Triton count/scatter kernels.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import time

import numpy as np
import torch

from ..cache_runtime import sparse_decode_fast, store
from ..genotype import _allele_frequency_summary, _indices
from ..genotype_device import DeviceMinorBlock

_KERNELS = None


def _immutable(values):
    a = np.asarray(values, dtype=np.int64)
    return np.frombuffer(a.tobytes(), dtype=np.int64)


def _stamp(tensor):
    return (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype, tensor.device)


def _kernels():
    global _KERNELS, triton, tl
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def scatter_raw(O, OFFSETS, R, S, COLS, M, BLOCK: tl.constexpr):
        col = tl.program_id(1)
        source_col = tl.load(COLS + col)
        begin = tl.load(OFFSETS + source_col).to(tl.int64)
        end = tl.load(OFFSETS + source_col + 1).to(tl.int64)
        x = begin + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = x < end
        row = tl.load(R + x, valid, other=0).to(tl.int64)
        state = tl.load(S + x, valid, other=0)
        tl.store(O + row * M + col, state, valid)

    @triton.jit
    def count_trait(S, ROWS, O, M, N, TILES, BLOCK: tl.constexpr):
        tile, col = tl.program_id(0), tl.program_id(1)
        r = tile * BLOCK + tl.arange(0, BLOCK)
        valid = r < N
        source_row = tl.load(ROWS + r, valid, other=0).to(tl.int64)
        state = tl.load(S + source_row * M + col, valid, other=3)
        ref = tl.where(state == 0, 2, tl.where((state == 1) | (state == 4), 1, 0))
        called = tl.where(state < 3, 2, tl.where(state >= 4, 1, 0))
        whole_ref = tl.where(state < 3, 2 - state.to(tl.int32), 0)
        missing = state >= 3
        offset = (col * TILES + tile) * 4
        tl.store(O + offset, tl.sum(tl.where(valid, ref, 0).to(tl.int64), 0))
        tl.store(O + offset + 1, tl.sum(tl.where(valid, called, 0).to(tl.int64), 0))
        tl.store(O + offset + 2, tl.sum(tl.where(valid, whole_ref, 0).to(tl.int64), 0))
        tl.store(O + offset + 3, tl.sum((valid & missing).to(tl.int64), 0))

    @triton.jit
    def materialize_trait(S, ROWS, COLS, FLIP, O, SOURCE_M, M, N, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = x < M * N
        col = x % M
        source_col = tl.load(COLS + col, valid, other=0)
        source_row = tl.load(ROWS + x // M, valid, other=0).to(tl.int64)
        state = tl.load(S + source_row * SOURCE_M + source_col, valid, other=3)
        flip = tl.load(FLIP + col, valid, other=0)
        dosage = tl.where(state < 3, tl.where(flip, state, 2 - state), 3)
        tl.store(O + x, dosage, valid)

    _KERNELS = scatter_raw, count_trait, materialize_trait
    return _KERNELS


@dataclass(frozen=True)
class RawStateBlock:
    """Lossless uint8 [physical cache samples, requested variants] states."""
    states: object
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    _seal: object = field(default=None, repr=False, compare=False)
    _state_stamp: object = field(default=None, repr=False, compare=False)

    @property
    def shape(self):
        return tuple(self.states.shape)


@dataclass
class _Axis:
    original: object
    samples: np.ndarray
    cache_rows: np.ndarray
    device_rows: object


@dataclass(frozen=True)
class _FrameSummaries:
    offsets: np.ndarray
    ref_ac: np.ndarray
    called: np.ndarray
    n: int


class SharedStateBroker:
    """Share verified frames and CSR H2D across phenotype-specific views.

    ``sample_indices`` means rows of the metadata reader. For a prepared
    population these are logical rows; original source rows remain provenance. ``read_states``
    preserves caller variant order. Its optional MAC bound rejects only sites
    that no cache sample subset can retain; final MAC is always cohort-specific.
    This object never writes cache files and never falls back to SDK genotypes.
    """
    def __init__(self, original_reader, container, *, device='cuda:0',
                 memory_limit_gib=20, device_cache_bytes=512 * 2**20,
                 compact_cache_bytes=64 * 2**20, own_reader=False, own_container=True):
        if not isinstance(container, store.Container) or not container.complete:
            raise ValueError('a complete verified six-state Container is required')
        if (not isinstance(memory_limit_gib, (int, float)) or isinstance(memory_limit_gib, bool)
                or not math.isfinite(memory_limit_gib) or not 0 < memory_limit_gib <= 20):
            raise ValueError('memory_limit_gib must be finite and in (0, 20]')
        for value in (device_cache_bytes, compact_cache_bytes):
            if type(value) is not int or value < 0:
                raise ValueError('shared cache capacities must be nonnegative integers')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('shared six-state device must be cpu or cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA shared cache requested but CUDA is unavailable')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._reader, self._container = original_reader, container
        self._own_reader, self._own_container = own_reader, own_container
        self._limit = int(memory_limit_gib * 2**30)
        self._device_capacity, self._compact_capacity = device_cache_bytes, compact_cache_bytes
        self._sources, self._device_sources = OrderedDict(), OrderedDict()
        self._source_bytes = self._device_bytes = 0
        self._closed, self._seal, self._axes = False, object(), []
        # Portable metadata exposes logical rows of the prepared population.
        # Original source FAM rows remain exclusively in the container proof.
        logical_axis = getattr(original_reader, 'sample_axis_kind', None) == 'prepared_population'
        self.samples = _immutable(np.arange(container.manifest['n'], dtype=np.int64)
                                  if logical_axis else container.samples)
        self._starts = np.asarray(container.index['start'], dtype=np.int64)
        self._sizes = np.asarray(container.index['m'], dtype=np.int64)
        if (self.samples.shape != (container.manifest['n'],)
                or np.any(self.samples < 0) or np.any(self.samples >= original_reader.n_samples)
                or len(np.unique(self.samples)) != len(self.samples)):
            raise ValueError('physical cache sample binding differs from source')
        if (container.manifest['m'] != original_reader.n_variants
                or (len(self._starts) and (self._starts[0] != 0
                    or np.any(self._sizes < 1)
                    or np.any(self._starts[1:] != self._starts[:-1] + self._sizes[:-1])
                    or self._starts[-1] + self._sizes[-1] != original_reader.n_variants))):
            raise ValueError('physical cache variant coverage differs from source')
        self._sample_sort = np.argsort(self.samples)
        self._sorted_samples = self.samples[self._sample_sort]
        self._metrics = dict(frame_reads=0, compact_frame_hits=0, compact_frame_evictions=0,
            gpu_csr_uploads=0, gpu_csr_hits=0, gpu_csr_evictions=0,
            device_csr_uploads=0, device_csr_hits=0,
            csr_h2d_bytes=0, index_h2d_bytes=0, h2d_bytes=0, state_read_calls=0,
            requested_variants=0, bound_rejected_variants=0, returned_raw_variants=0,
            trait_block_calls=0, trait_count_calls=0, returned_trait_variants=0,
            bound_sample_axes=0, sample_axis_reuse_calls=0, sample_axis_validation_calls=0,
            compact_bytes_highwater=0, gpu_csr_bytes_highwater=0,
            raw_slab_bytes_highwater=0, memory_required_bytes_highwater=0,
            frame_read_validate_seconds=0., csr_upload_host_seconds=0.,
            raw_materialize_host_seconds=0., trait_count_host_seconds=0.,
            trait_materialize_host_seconds=0., trait_summary_d2h_calls=0,
            trait_summary_d2h_bytes=0, trait_summary_d2h_host_seconds=0.,
            batched_trait_summary_calls=0, batched_trait_summary_traits=0,
            trait_pending_summary_bytes_highwater=0, batched_trait_count_guard_calls=0)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError('shared six-state broker is closed')

    @property
    def metrics(self):
        return dict(self._metrics, backend='shared-lossless-six-state-CSR', device=str(self.device),
            compact_cache_limit_bytes=self._compact_capacity,
            gpu_csr_cache_limit_bytes=self._device_capacity,
            current_compact_cache_bytes=self._source_bytes,
            current_gpu_csr_cache_bytes=self._device_bytes,
            memory_limit_bytes=self._limit, genotype_sdk_fallback_count=0,
            timer_contract='host times include enqueue; summary D2H includes wait/copy; stages may overlap')

    def _evict_device(self):
        _, (_, size, _) = self._device_sources.popitem(last=False)
        self._device_bytes -= size
        self._metrics['gpu_csr_evictions'] += 1

    def _guard(self, required):
        self._metrics['memory_required_bytes_highwater'] = max(
            self._metrics['memory_required_bytes_highwater'], required)
        if required > self._limit:
            raise MemoryError('shared six-state workspace exceeds configured memory budget')
        if self.device.type == 'cuda':
            # Own LRU entries may be evicted; independent model allocations are
            # never removed or precision-adjusted to satisfy this guard.
            while self._device_sources and torch.cuda.memory_allocated(self.device) + required > self._limit:
                self._evict_device()
            allocated = torch.cuda.memory_allocated(self.device)
            unused = torch.cuda.memory_reserved(self.device) - allocated
            free, _ = torch.cuda.mem_get_info(self.device)
            if allocated + required > self._limit or required > free + unused:
                raise MemoryError('shared six-state workspace exceeds live CUDA budget')

    def _source(self, frame):
        if frame in self._sources:
            self._metrics['compact_frame_hits'] += 1
            self._sources.move_to_end(frame)
            return self._sources[frame][0]
        if frame in self._device_sources:
            # Tiny source summaries remain with the device CSR. They do not
            # retain the large decompressed exception buffers after CPU LRU
            # eviction, and avoid another disk/decompress cycle on GPU hits.
            return self._device_sources[frame][2]
        started = time.perf_counter()
        f = self._container.read_frame(int(frame))
        if (f['start'] != self._starts[frame] or f['m'] != self._sizes[frame]
                or f['n'] != len(self.samples)):
            raise ValueError('verified frame geometry differs from cache index')
        source = sparse_decode_fast.validate_source(f['offsets'], f['sample_index'], f['state'],
            f['reference_alleles'], f['called_alleles'], int(f['n']))
        self._metrics['frame_reads'] += 1
        self._metrics['frame_read_validate_seconds'] += time.perf_counter() - started
        size = sum(a.nbytes for a in (source.offsets, source.sample_index, source.state,
                                      source.ref_ac, source.called))
        if size <= self._compact_capacity:
            while self._sources and self._source_bytes + size > self._compact_capacity:
                _, (_, old) = self._sources.popitem(last=False)
                self._source_bytes -= old
                self._metrics['compact_frame_evictions'] += 1
            self._sources[frame] = (source, size)
            self._source_bytes += size
            self._metrics['compact_bytes_highwater'] = max(
                self._metrics['compact_bytes_highwater'], self._source_bytes)
        return source

    def _upload(self, frame, source):
        if frame in self._device_sources:
            self._metrics['device_csr_hits'] += 1
            if self.device.type == 'cuda':
                self._metrics['gpu_csr_hits'] += 1
            self._device_sources.move_to_end(frame)
            return self._device_sources[frame][0]
        if isinstance(source, _FrameSummaries):
            # A live-budget eviction between metadata selection and upload
            # may remove the CSR. Revalidate its immutable frame if needed.
            source = self._source(frame)
        started = time.perf_counter()
        payload_bytes = source.offsets.nbytes + source.sample_index.nbytes + source.state.nbytes
        self._guard(payload_bytes + 64 * 2**20)
        payload = tuple(torch.from_numpy(np.array(a, copy=True)).to(self.device)
                        for a in (source.offsets, source.sample_index, source.state))
        self._metrics['device_csr_uploads'] += 1
        if self.device.type == 'cuda':
            self._metrics['gpu_csr_uploads'] += 1
            self._metrics['csr_h2d_bytes'] += payload_bytes
            self._metrics['h2d_bytes'] += payload_bytes
        self._metrics['csr_upload_host_seconds'] += time.perf_counter() - started
        if payload_bytes <= self._device_capacity:
            while self._device_sources and self._device_bytes + payload_bytes > self._device_capacity:
                self._evict_device()
            metadata = _FrameSummaries(source.offsets, source.ref_ac, source.called, source.n)
            self._device_sources[frame] = (payload, payload_bytes, metadata)
            self._device_bytes += payload_bytes
            self._metrics['gpu_csr_bytes_highwater'] = max(
                self._metrics['gpu_csr_bytes_highwater'], self._device_bytes)
        return payload

    def _indices_to_device(self, values):
        a = np.asarray(values)
        result = torch.from_numpy(np.array(a, copy=True)).to(self.device)
        if self.device.type == 'cuda':
            self._metrics['index_h2d_bytes'] += a.nbytes
            self._metrics['h2d_bytes'] += a.nbytes
        return result

    @staticmethod
    def _minimum_mac(value):
        if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value < 0):
            raise ValueError('minimum MAC must be finite and nonnegative')

    def read_states(self, variant_indices, minimum_mac_bound=None):
        """Read bounded physical state slabs, sharing verified CSR transport.

        A subset can have at most ``min(full_REF, full_ALT + 1)`` initial MAC.
        The extra allele preserves the original odd called-count rounding.
        """
        self._check_open()
        self._minimum_mac(minimum_mac_bound)
        variants = _indices(variant_indices, self._reader.n_variants, 'variant_indices')
        self._metrics['state_read_calls'] += 1
        self._metrics['requested_variants'] += len(variants)
        frames = np.searchsorted(self._starts, variants, side='right') - 1
        parts, positions = [], []
        for frame in np.unique(frames):
            source = self._source(int(frame))
            request_positions = np.flatnonzero(frames == frame)
            columns = variants[request_positions] - self._starts[frame]
            if minimum_mac_bound is not None:
                upper = np.minimum(source.ref_ac[columns], source.called[columns] - source.ref_ac[columns] + 1)
                keep = upper >= minimum_mac_bound
                self._metrics['bound_rejected_variants'] += int((~keep).sum())
                columns, request_positions = columns[keep], request_positions[keep]
            if not len(columns):
                continue
            cells = len(self.samples) * len(columns)
            if cells > np.iinfo(np.int32).max:
                raise MemoryError('raw six-state slab exceeds validated int32 geometry')
            self._guard(cells * 2 + 64 * 2**20)
            payload = self._upload(int(frame), source)
            self._guard(cells + 64 * 2**20)
            started = time.perf_counter()
            output = torch.zeros((len(self.samples), len(columns)), dtype=torch.uint8, device=self.device)
            c = self._indices_to_device(columns)
            offsets, r, state = payload
            if len(state):
                if self.device.type == 'cuda':
                    scatter, _, _ = _kernels()
                    largest_column = int(np.max(np.diff(source.offsets)[columns], initial=0))
                    with torch.cuda.device(self.device):
                        if largest_column:
                            scatter[(triton.cdiv(largest_column, 256), len(columns))](
                                output, offsets, r, state, c, len(columns), BLOCK=256)
                else:
                    for destination, column in enumerate(columns):
                        begin, end = map(int, source.offsets[column:column+2])
                        output[r[begin:end].to(torch.int64), destination] = state[begin:end]
            self._metrics['raw_materialize_host_seconds'] += time.perf_counter() - started
            self._metrics['raw_slab_bytes_highwater'] = max(
                self._metrics['raw_slab_bytes_highwater'], output.numel())
            parts.append(output)
            positions.append(request_positions)
        if parts:
            all_positions = np.concatenate(positions)
            order = np.argsort(all_positions)
            if len(parts) == 1 and np.array_equal(order, np.arange(len(order))):
                states = parts[0]
            else:
                cells = len(self.samples) * len(all_positions)
                self._guard(cells * 2 + 64 * 2**20)
                states = torch.cat(parts, 1).index_select(1, self._indices_to_device(order))
            selected = variants[all_positions[order]]
        else:
            states = torch.empty((len(self.samples), 0), dtype=torch.uint8, device=self.device)
            selected = np.empty(0, dtype=np.int64)
        self._metrics['returned_raw_variants'] += len(selected)
        return RawStateBlock(states, self.samples, _immutable(selected), self._seal, _stamp(states))

    def _axis(self, samples):
        candidate = np.asarray(samples)
        if candidate.ndim != 1 or candidate.dtype.kind not in 'iu':
            raise ValueError('sample_indices must be a one-dimensional integer array')
        # Integer comparison must never pass through floating point. In
        # particular uint64 vs int64 can promote to double in NumPy; reject
        # unrepresentable unsigned values before casting to the canonical axis.
        if candidate.dtype.kind == 'u' and np.any(candidate > np.uint64(np.iinfo(np.int64).max)):
            raise IndexError('sample_indices exceeds the canonical integer dimension')
        candidate_values = candidate.astype(np.int64, copy=False)
        for axis in self._axes:
            if samples is axis.samples:
                if (candidate.dtype != np.int64 or candidate.shape != axis.cache_rows.shape
                        or candidate.flags.writeable):
                    raise ValueError('immutable sample binding metadata was altered')
                self._metrics['sample_axis_reuse_calls'] += 1
                return axis
            # Pipeline construction may create an equal ordered copy. Its
            # integer dtype, full shape and every value prove the same already
            # validated axis; identity or shape alone never grants admission.
            if (candidate_values.shape == axis.samples.shape
                    and np.array_equal(candidate_values, axis.samples)):
                self._metrics['sample_axis_reuse_calls'] += 1
                return axis
        self._metrics['sample_axis_validation_calls'] += 1
        selected = _indices(samples, self._reader.n_samples, 'sample_indices')
        positions = np.searchsorted(self._sorted_samples, selected)
        if (np.any(positions >= len(self.samples)) or
                np.any(self._sorted_samples[np.minimum(positions, max(0, len(self.samples)-1))] != selected)):
            raise ValueError('phenotype samples are outside the physical cache binding')
        rows = _immutable(self._sample_sort[positions])
        self._guard(rows.nbytes + 64 * 2**20)
        axis = _Axis(samples, _immutable(selected), rows, self._indices_to_device(rows))
        self._axes.append(axis)
        self._metrics['bound_sample_axes'] += 1
        return axis

    def _validate_raw(self, raw):
        if (not isinstance(raw, RawStateBlock) or raw._seal is not self._seal
                or raw._state_stamp != _stamp(raw.states)
                or raw.sample_indices is not self.samples or raw.states.dtype != torch.uint8
                or raw.variant_indices.ndim != 1 or raw.variant_indices.dtype != np.int64
                or raw.variant_indices.flags.writeable
                or raw.states.device != self.device or raw.shape != (len(self.samples), len(raw.variant_indices))):
            raise ValueError('unaltered raw block from this shared broker required')

    def _count_device(self, raw, axis, *, guard_counts=True):
        """Keep integer summaries on device; release each trait's partials."""
        n, m = len(axis.samples), raw.shape[1]
        if not m:
            return torch.empty((4, 0), dtype=torch.int64, device=self.device)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            _, count, _ = _kernels()
            tiles = triton.cdiv(n, 1024)
            if guard_counts:
                self._guard(m * (max(1, tiles) + 2) * 32 + 64 * 2**20)
            partial = torch.empty((m, tiles, 4), dtype=torch.int64, device=self.device)
            if tiles:
                with torch.cuda.device(self.device):
                    count[(tiles, m)](raw.states, axis.device_rows, partial, m, n, tiles, BLOCK=1024)
            result = partial.sum(1).T.contiguous()
            del partial
        else:
            # Column chunks bound oracle scratch, including table lookups.
            result = torch.empty((4, m), dtype=torch.int64)
            ref = torch.tensor([2, 1, 0, 0, 1, 0], dtype=torch.int64)
            called = torch.tensor([2, 2, 2, 0, 1, 1], dtype=torch.int64)
            for begin in range(0, m, 128):
                end = min(begin + 128, m)
                if guard_counts:
                    self._guard(40 * n * (end-begin) + 64 * 2**20)
                s = raw.states[:, begin:end].index_select(0, axis.device_rows).to(torch.int64)
                values = torch.stack((ref[s].sum(0), called[s].sum(0),
                    torch.where(s < 3, 2-s, 0).sum(0), (s >= 3).sum(0)))
                result[:, begin:end] = values
        self._metrics['trait_count_calls'] += 1
        self._metrics['trait_count_host_seconds'] += time.perf_counter() - started
        return result

    def _summaries_to_cpu(self, counts):
        started = time.perf_counter()
        result = counts.detach().cpu().numpy()
        if self.device.type == 'cuda' and counts.numel():
            self._metrics['trait_summary_d2h_calls'] += 1
            self._metrics['trait_summary_d2h_bytes'] += counts.numel() * counts.element_size()
        self._metrics['trait_summary_d2h_host_seconds'] += time.perf_counter()-started
        return result

    def _count(self, raw, axis):
        return self._summaries_to_cpu(self._count_device(raw, axis))

    def trait_block(self, raw_block, sample_indices, minimum_mac=None):
        """Return an independently oriented cohort block, preserving row order."""
        self._check_open()
        self._validate_raw(raw_block)
        self._minimum_mac(minimum_mac)
        axis = self._axis(sample_indices)
        counts = self._count(raw_block, axis)
        return self._materialize_from_counts(raw_block, axis, counts, minimum_mac)

    def trait_blocks(self, raw_block, sample_axes, minimum_mac=None):
        """Count independent cohorts, then transfer their small summaries once.

        Each existing integer count kernel retains its sample order and
        half-call meanings. Only [trait, four summaries, variants] tensors are
        stacked. No model, genotype imputation or matrix K dimension is merged.
        """
        self._check_open()
        self._validate_raw(raw_block)
        self._minimum_mac(minimum_mac)
        axes = [self._axis(samples) for samples in sample_axes]
        self._metrics['batched_trait_summary_calls'] += 1
        self._metrics['batched_trait_summary_traits'] += len(axes)
        if not axes:
            return []
        pending = []
        m = raw_block.shape[1]
        if m:
            largest_axis = max(len(axis.samples) for axis in axes)
            if self.device.type == 'cuda':
                partial = m * (max(1, (largest_axis+1023)//1024) + 2) * 32
            else:
                partial = 40 * largest_axis * min(m, 128) + 2 * m * 32
            # Worst trait partial/reduction plus every retained summary and
            # the stack output. Check once before the owned sequence so live
            # memory queries do not insert a per-trait count synchronization.
            self._guard(partial + 2 * len(axes) * 4 * m * 8 + 64 * 2**20)
            self._metrics['batched_trait_count_guard_calls'] += 1
        for axis in axes:
            pending.append(self._count_device(raw_block, axis, guard_counts=False))
            pending_bytes = len(pending) * 4 * m * 8
            self._metrics['trait_pending_summary_bytes_highwater'] = max(
                self._metrics['trait_pending_summary_bytes_highwater'], pending_bytes)
        # The ownership guard already includes pending summaries and stack.
        stacked = torch.stack(pending)
        counts_cpu = self._summaries_to_cpu(stacked)
        pending.clear()
        del stacked
        return [self._materialize_from_counts(raw_block, axis, counts, minimum_mac)
                for axis, counts in zip(axes, counts_cpu)]

    def _materialize_from_counts(self, raw_block, axis, counts, minimum_mac):
        reference, called, whole_ref, missing = counts
        summaries = _allele_frequency_summary(reference, called, len(axis.samples))
        columns = (np.arange(raw_block.shape[1], dtype=np.int64) if minimum_mac is None
                   else np.flatnonzero(summaries[2] >= minimum_mac))
        af, missing_rate, initial_mac, reference_ac, called_alleles = tuple(a[columns] for a in summaries)
        n, m = len(axis.samples), len(columns)
        if n * m > np.iinfo(np.int32).max:
            raise MemoryError('trait dosage exceeds validated int32 geometry')
        self._guard(n * m * (1 if self.device.type == 'cuda' else 20) + 64 * 2**20)
        started = time.perf_counter()
        dosage = torch.empty((n, m), dtype=torch.uint8, device=self.device)
        if n * m:
            c = self._indices_to_device(columns)
            flip = self._indices_to_device(af >= .5)
            if self.device.type == 'cuda':
                _, _, materialize = _kernels()
                with torch.cuda.device(self.device):
                    materialize[(triton.cdiv(n*m, 256),)](raw_block.states, axis.device_rows,
                        c, flip, dosage, raw_block.shape[1], m, n, BLOCK=256)
            else:
                s = raw_block.states.index_select(0, axis.device_rows).index_select(1, c)
                dosage[:] = torch.where(s < 3, torch.where(flip[None, :], s, 2-s), 3)
        result = DeviceMinorBlock(dosage, axis.samples, raw_block.variant_indices[columns],
            af, initial_mac, missing_rate, reference_ac, called_alleles)
        observed = np.where(af >= .5, 2*(n-missing[columns])-whole_ref[columns], whole_ref[columns])
        result._store_counts(np.arange(n, dtype=np.int64), observed.astype(np.float64), missing[columns])
        self._metrics['trait_block_calls'] += 1
        self._metrics['returned_trait_variants'] += m
        self._metrics['trait_materialize_host_seconds'] += time.perf_counter()-started
        return result

    def reader_view(self, trait_rows):
        """Metadata-compatible genotype facade fixed to one ordered cohort axis."""
        self._check_open()
        return SharedTraitReader(self, self._axis(trait_rows))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sources.clear()
        self._device_sources.clear()
        self._axes.clear()
        self._source_bytes = self._device_bytes = 0
        try:
            if self._own_container:
                self._container.close()
        finally:
            if self._own_reader:
                self._reader.close()


class SharedTraitReader:
    """One single-model reader view; closing it never closes another view."""
    def __init__(self, broker, axis):
        self._broker, self._axis, self._closed = broker, axis, False
        self._starts, self._sizes = broker._starts, broker._sizes
        self._device = str(broker.device)
        self._metrics = dict(minor_block_calls=0, single_effective_blocks=0, single_effective_columns=0)

    def __getattr__(self, name):
        return getattr(self._broker._reader, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._closed = True

    @property
    def reader_metadata(self):
        result = dict(self._broker._reader.reader_metadata)
        result['analysis_cache'] = dict(self._metrics, backend='shared-lossless-six-state-view',
            genotype_sdk_fallback_count=0)
        result['phewas_shared_state'] = self._broker.metrics
        return result

    def read_genotype(self, *args, **kwargs):
        raise RuntimeError('shared cache views never fall back to SDK genotypes')

    def read_ref_dosage(self, *args, **kwargs):
        raise RuntimeError('shared cache views require cohort minor_block')

    def _samples(self, values):
        if self._closed:
            raise RuntimeError('shared trait reader is closed')
        self._broker._check_open()
        a = np.asarray(values)
        # The only admitted axis is already validated and immutable. Equality
        # proves range/uniqueness/order; rerunning np.unique on hundreds of
        # thousands of samples for every mask would repeat the old bottleneck.
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or a.shape != self._axis.samples.shape
                or not np.array_equal(a, self._axis.samples)):
            raise ValueError('trait reader sample order differs from its fixed cohort')
        return self._axis.samples

    def minor_block(self, variant_indices, union_sample_indices, *, device=None,
                    minimum_mac=None, resident=True):
        samples = self._samples(union_sample_indices)
        if device is not None:
            requested = torch.device(device)
            if requested.type == 'cuda' and requested.index is None:
                requested = torch.device('cuda', torch.cuda.current_device())
            if requested != self._broker.device:
                raise ValueError('trait reader device differs from shared broker')
        raw = self._broker.read_states(variant_indices, minimum_mac_bound=minimum_mac)
        result = self._broker.trait_block(raw, samples, minimum_mac=minimum_mac)
        self._metrics['minor_block_calls'] += 1
        return result

    def iter_minor_blocks(self, variant_indices, union_sample_indices, block_size=256, **options):
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        variants = _indices(variant_indices, self.n_variants, 'variant_indices')
        for start in range(0, len(variants), block_size):
            yield self.minor_block(variants[start:start+block_size], union_sample_indices, **options)

    def iter_effective_minor_blocks(self, variant_indices, union_sample_indices, block_size=1024,
                                    *, effective_block_size=1024, **options):
        """Standalone view contract; multi-trait scheduler owns shared packing."""
        if type(effective_block_size) is not int or effective_block_size < 1:
            raise ValueError('effective_block_size must be a positive integer')
        from ..cache_runtime.single_batches import _physical_requests
        variants = _indices(variant_indices, self.n_variants, 'variant_indices')
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        parts, pending = [], 0

        def merge():
            cells = len(self._axis.samples) * sum(part.shape[1] for part in parts)
            if cells > np.iinfo(np.int32).max:
                raise MemoryError('effective dosage exceeds validated int32 geometry')
            self._broker._guard(cells + 64 * 2**20)
            dosage = torch.cat([part.dosage for part in parts], 1)
            result = DeviceMinorBlock(dosage, self._axis.samples,
                np.concatenate([part.variant_indices for part in parts]),
                *[np.concatenate([getattr(part, name) for part in parts]) for name in
                  ('union_ref_af', 'union_initial_mac', 'union_missing_rate', 'union_ref_ac', 'union_called_alleles')])
            self._metrics['single_effective_blocks'] += 1
            self._metrics['single_effective_columns'] += result.shape[1]
            return result

        for request in _physical_requests(self, variants, block_size):
            block = self.minor_block(request, union_sample_indices, **options)
            begin = 0
            while begin < block.shape[1]:
                stop = min(block.shape[1], begin + effective_block_size-pending)
                part = block.select_columns(np.arange(begin, stop, dtype=np.int64))
                parts.append(part)
                pending += stop-begin
                begin = stop
                if pending == effective_block_size:
                    result = merge()
                    parts.clear()
                    pending = 0
                    yield result
                    del result
        if pending:
            yield merge()
