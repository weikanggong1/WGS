"""Frame-aligned compact Single reads with bounded eligible-column batches.

The reader retains source/cohort MAC, allele summaries and caller order. Only
CPU compact arrays are accumulated; no genotype matrix is re-cached on disk.
"""
import time
import numpy as np


def _slice_prepared(prepared, start, stop):
    """Select a contiguous eligible-column interval before dense expansion."""
    columns = prepared['exception_col']
    keep = (columns >= start) & (columns < stop)
    return dict(prepared, columns=prepared['columns'][start:stop],
                exception_col=columns[keep] - start,
                exception_row=prepared['exception_row'][keep],
                exception_state=prepared['exception_state'][keep],
                summaries=tuple(a[start:stop] for a in prepared['summaries']))


def _merge_prepared(parts, sample_count):
    """Rebind compact columns to their exact retained physical variant axis."""
    # Each part is (prepared, physical_variant_indices), in original order.
    count = sum(len(indices) for _, indices in parts)
    if len(parts) == 1:
        item, variants = parts[0]
        return dict(item, cache_variant_count=count, cache_sample_count=sample_count,
                    columns=np.arange(count, dtype=np.int64),
                    samples=np.arange(sample_count, dtype=np.int64)), variants
    columns, rows, states, variants = [], [], [], []
    offset = 0
    for prepared, indices in parts:
        columns.append(prepared['exception_col'] + offset)
        rows.append(prepared['exception_row'])
        states.append(prepared['exception_state'])
        variants.append(indices)
        offset += len(indices)
    return dict(cache_variant_count=count, cache_sample_count=sample_count,
                columns=np.arange(count, dtype=np.int64), samples=np.arange(sample_count, dtype=np.int64),
                exception_col=np.concatenate(columns), exception_row=np.concatenate(rows),
                exception_state=np.concatenate(states),
                summaries=tuple(np.concatenate([p['summaries'][i] for p, _ in parts]) for i in range(5)),
                full_union_summaries=all(p['full_union_summaries'] for p, _ in parts)), np.concatenate(variants)


def _physical_requests(adapter, variants, block_size):
    """Avoid reading adjacent compressed frames twice for sorted Single axes."""
    if len(variants) < 2 or np.all(variants[1:] > variants[:-1]):
        offset = 0
        while offset < len(variants):
            frame = int(np.searchsorted(adapter._starts, variants[offset], side='right') - 1)
            stop = int(np.searchsorted(variants, adapter._starts[frame] + adapter._sizes[frame], side='left'))
            for begin in range(offset, stop, block_size):
                yield variants[begin:min(begin + block_size, stop)]
            offset = stop
    else:
        # Unsorted valid requests retain caller order rather than being sorted.
        for begin in range(0, len(variants), block_size):
            yield variants[begin:begin + block_size]


def iter_effective_minor_blocks(adapter, variant_indices, union_sample_indices,
                                block_size=1024, *, effective_block_size=1024,
                                device=None, minimum_mac=None, resident=True):
    """Yield at most effective_block_size retained columns per device block.

    Physical input reads remain bounded by block_size and compressed frames.
    MAC filtering happens in the original cohort decoder before packing. The
    subsequent pipeline keeps its global MAC-filtered ordinal, independently
    of these compute batches, preserving native chunk/factor ordering.
    """
    from ..gds import _indices
    if adapter._closed:
        raise RuntimeError('cache reader is closed')
    if not resident:
        raise ValueError('effective Single batches require resident CUDA dosage')
    if type(block_size) is not int or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if type(effective_block_size) is not int or effective_block_size < 1:
        raise ValueError('effective_block_size must be a positive integer')
    variants = _indices(variant_indices, adapter.n_variants, 'variant_indices')
    samples = _indices(union_sample_indices, adapter.n_samples, 'union_sample_indices')
    if len(samples) * effective_block_size > np.iinfo(np.int32).max:
        raise MemoryError('effective Single dosage geometry exceeds validated int32 CUDA indexing')
    metrics = adapter._metrics
    parts, pending_count = [], 0

    def materialize():
        began = time.perf_counter()
        prepared, selected_variants = _merge_prepared(parts, len(samples))
        pack_seconds = time.perf_counter() - began
        metrics['single_pack_seconds'] = metrics.get('single_pack_seconds', 0.) + pack_seconds
        add_prepare_seconds = getattr(adapter, '_add_compact_prepare_seconds', None)
        if add_prepare_seconds is not None:
            add_prepare_seconds(pack_seconds)
        else:
            # Preserve compatibility with synchronous third-party/mock adapters.
            metrics['compact_prepare_seconds'] += pack_seconds
        began = time.perf_counter()
        result = adapter._fast.to_minor_block(prepared, samples, selected_variants,
                                               device=device or adapter._device)
        seconds = time.perf_counter() - began
        metrics['materialize_seconds'] += seconds
        metrics['minor_block_wall_seconds'] += pack_seconds + seconds
        metrics['minor_block_calls'] += 1
        metrics['cuda_resident_calls'] += 1
        metrics['returned_variants'] += result.shape[1]
        metrics['single_effective_blocks'] = metrics.get('single_effective_blocks', 0) + 1
        metrics['single_effective_columns'] = metrics.get('single_effective_columns', 0) + result.shape[1]
        return result

    requests = _physical_requests(adapter, variants, block_size)
    prepare_requests = getattr(adapter, '_prepared_requests', None)
    iterator = (prepare_requests(requests, samples, minimum_mac) if prepare_requests is not None else
                (adapter._prepare(request, samples, minimum_mac) for request in requests))
    try:
        while True:
            began = time.perf_counter()
            item = next(iterator, None)
            metrics['minor_block_wall_seconds'] += time.perf_counter() - began
            if item is None:
                break
            prepared, bound_samples, bound_variants = item
            metrics['requested_variants'] += len(bound_variants)
            metrics['single_compact_requests'] = metrics.get('single_compact_requests', 0) + 1
            if not np.array_equal(bound_samples, samples):
                raise RuntimeError('Single compact request changed its bound sample axis')
            count = len(prepared['columns'])
            begin = 0
            while begin < count:
                stop = min(count, begin + effective_block_size - pending_count)
                part = prepared if begin == 0 and stop == count else _slice_prepared(prepared, begin, stop)
                parts.append((part, bound_variants[part['columns']]))
                pending_count += stop - begin
                begin = stop
                if pending_count == effective_block_size:
                    result = materialize()
                    parts.clear()
                    pending_count = 0
                    yield result
                    del result
    finally:
        iterator.close()
    if pending_count:
        result = materialize()
        parts.clear()
        yield result
