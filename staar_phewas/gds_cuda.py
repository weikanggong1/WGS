"""Lossless CUDA Bit2 decoding for early individual-variant MAC selection.

The SDK reads bounded uint8 slabs. CUDA computes integer allele counts and
transfers only eligible dosages; source AF and MAC formulas stay float64 on
the CPU. Association calculations and imputation are performed elsewhere.
"""

from __future__ import annotations

import numpy as np
import time
from contextlib import nullcontext


def _device_sample_index(reader,samples,device,*,permutation=None):
    """One reader-owned entry of integer indices; never retain genotypes.

    Resolve bare CUDA devices on every request so a changed current device
    cannot reuse another device's indices. The owning CPU key detects caller
    mutation; both device tensors are lazy and evicted together.
    """
    import torch
    target=torch.device(device)
    if target.type=='cuda' and target.index is None:
        target=torch.device('cuda',torch.cuda.current_device())
    key=(target.type,target.index)
    indices=np.asarray(samples,dtype=np.int64)
    cached=getattr(reader,'_cuda_sample_index_cache',None)
    if cached is None or cached['device']!=key or not np.array_equal(cached['samples'],indices):
        cached=dict(device=key,samples=indices.copy(),sample_indices=None,permutation=None)
        reader._cuda_sample_index_cache=cached
    field='sample_indices' if permutation is None else 'permutation'
    if cached[field] is None:
        values=cached['samples'] if permutation is None else np.asarray(permutation,dtype=np.int64)
        # copy=True avoids exposing mutable NumPy storage in CPU contracts.
        cached[field]=torch.tensor(values,dtype=torch.int64,device=target)
    return cached[field]


_packed_kernel = None


def _get_packed_kernel():
    global _packed_kernel, tl
    if _packed_kernel is None:
        import triton
        import triton.language as tl

        # Block length and leading bit are runtime integers. Varying GDS
        # groups must reuse a compiled kernel, including length-one groups.
        @triton.jit(do_not_specialize=[3, 6, 7])
        def unpack(P, S, O, C, W: tl.constexpr, NS: tl.constexpr,
                   L, BIT, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            live = i < L * NS * 2
            row = i // (NS * 2)
            sample = tl.load(S + (i // 2) % NS, live, other=0).to(tl.int64)
            logical = row.to(tl.int64) * W + sample * 2 + i % 2
            q = logical * 2 + BIT
            valid = live & (logical < C)
            byte = tl.load(P + (q >> 3), valid, other=0).to(tl.uint32)
            value = tl.where(valid, (byte >> (q & 7)) & 3, 3).to(tl.uint8)
            tl.store(O + i, value, live)

        _packed_kernel = unpack
    return _packed_kernel


def _packed_raw_tensor(reader, raw, info, samples, selected, layers, device, measure):
    """Decode packed logical bytes directly into the original selected layout.

    Integer sample indices are the only cached CUDA payload. The existing
    layer-combination and dosage code consumes unchanged uint8 allele codes.
    The 20 GiB/live-free preflight is an estimate, not an allocator guarantee.
    """
    import torch
    from . import tf32
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Packed GPU decoding requires a CUDA device")
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    indices = np.asarray(samples, dtype=np.int64)
    cached = getattr(reader, "_packed_cuda_sample_index_cache", None)
    if (cached is None or cached['device'] != target.index
            or not np.array_equal(cached['samples'], indices)):
        cached = dict(samples=indices.copy(), device=target.index, tensors={})
        reader._packed_cuda_sample_index_cache = cached
    count = layers * len(indices) * 2
    required = raw.nbytes + count + len(indices) * 8 + 2**20
    allocated = torch.cuda.memory_allocated(target)
    reserved = torch.cuda.memory_reserved(target)
    free, _ = torch.cuda.mem_get_info(target)
    snapshot = tf32._product_workspace_availability(
        allocated=allocated, reserved=reserved, free=free,
        limit=min(20 * 2**30, tf32._memory_limit_bytes),
        reserve=tf32._memory_reserve_bytes)
    available = snapshot['available_bytes']
    reader._packed_workspace_snapshot = dict(snapshot, required_bytes=required)
    if required > available:
        raise MemoryError("Packed decoder exceeds the live/20 GiB workspace estimate")
    import triton
    with measure("gds_packed_raw_h2d", gpu=True):
        if selected not in cached['tensors']:
            values = np.sort(indices) if selected else indices
            cached['tensors'][selected] = torch.tensor(values, dtype=torch.int64, device=target)
        sample_tensor = cached['tensors'][selected]
        raw_tensor = torch.as_tensor(raw, device=target)
    with measure("gds_packed_unpack", gpu=True):
        output = torch.empty((layers, len(indices), 2), dtype=torch.uint8, device=target)
        kernel = _get_packed_kernel()
        kernel[(triton.cdiv(count, 256),)](raw_tensor, sample_tensor, output,
            info[1], reader.n_samples * 2, len(indices), layers, info[0], 256)
    return output


def native_minor_block(reader, variants, samples, *, device, minimum_mac=None, resident=False):
    """Return canonical COO minor dosages, preserving caller sample/site order."""
    import torch
    from .gds import SparseMinorBlock, _allele_frequency_summary

    packed_reader = getattr(reader, "_packed_reader", None)
    if packed_reader is not None and getattr(reader, "_packed_read_failed", False):
        raise RuntimeError("Packed read failed; close and reopen the reader")
    profiler = getattr(reader, "_stage_profiler", None)
    def measure(name, *, gpu=False):
        return profiler.measure(name, gpu=gpu) if profiler is not None else nullcontext()

    reader._prepare_genotype_index()
    steps = reader._genotype_steps[variants]
    if np.any(steps > 16):
        raise ValueError("Bit2 layer count exceeds the supported 16 layers")
    n, m = len(samples), len(variants)
    reference_ac = np.zeros(m, dtype=np.int64)
    called_alleles = np.zeros(m, dtype=np.int64)
    dosages = {}
    half_missing = 0
    row_width = reader.n_samples * reader.ploidy
    selected_cells = n * reader.ploidy
    mask_bytes = row_width + reader.n_samples + 16 * n
    max_layers = (reader.genotype_raw_memory_bytes - mask_bytes - 12 * selected_cells) // (
        row_width + 4 * selected_cells)
    if n and m and max_layers < 1:
        raise MemoryError("one genotype Bit2 layer exceeds the raw read budget")
    variant_order = np.argsort(variants)
    selected = [(int(output), int(reader._genotype_offsets[variants[output]]),
                 int(reader._genotype_offsets[variants[output] + 1]))
                for output in variant_order if steps[output]]
    groups = []
    for entry in selected:
        if (groups and entry[1] - groups[-1][-1][2] <= reader.genotype_max_gap_layers
                and entry[2] - groups[-1][0][1] <= max_layers):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    selection = sample_order = None
    if n * 2 < reader.n_samples and any(len(group) >= 2 for group in groups):
        selection, permutation = reader._sample_selection(samples)
    sample_tensor = None
    counters = getattr(reader, "_reader_io_counts", None)
    if counters is None:
        counters = reader._reader_io_counts = {name: {"calls": 0, "returned_bytes": 0}
                                               for name in ("flat", "selected")}
    for group in groups if n else []:
        outputs = np.asarray([entry[0] for entry in group], dtype=np.int64)
        first = np.asarray([entry[1] for entry in group], dtype=np.int64)
        last = np.asarray([entry[2] for entry in group], dtype=np.int64)
        group_steps = last - first
        use_selected = len(group) >= 2 and selection is not None
        route = "packed" if packed_reader is not None else ("selected" if use_selected else "flat")
        counters.setdefault(route, {"calls": 0, "returned_bytes": 0})
        codes = torch.zeros((len(group), n, reader.ploidy), dtype=torch.int64, device=device)
        for lower in range(int(first[0]), int(last[-1]), max_layers):
            upper = min(int(last[-1]), lower + max_layers)
            read_started = time.perf_counter()
            with measure("gds_packed_allocator_read" if packed_reader is not None else "gds_sdk_read", gpu=False):
                if packed_reader is not None:
                    try:
                        packed_info = packed_reader.read_packed_path(
                            reader._file.fileid, "genotype/data", lower * row_width,
                            (upper - lower) * row_width, reader.n_samples)
                    except Exception:
                        reader._packed_read_failed = True
                        raise
                    raw = packed_info[0]
                elif use_selected:
                    raw = reader._flat_reader.read_selected_rows_path(
                        reader._file.fileid, "genotype/data", lower * row_width,
                        upper - lower, row_width, selection, "uint8")
                    raw = np.asarray(raw, dtype=np.uint8).reshape(upper - lower, n, reader.ploidy)
                else:
                    raw = reader._flat_reader.read_flat_path(
                        reader._file.fileid, "genotype/data", lower * row_width,
                        (upper - lower) * row_width, "uint8")
                    raw = np.asarray(raw, dtype=np.uint8).reshape(
                        upper - lower, reader.n_samples, reader.ploidy)
            counters[route]["seconds"] = counters[route].get("seconds", 0.0) + time.perf_counter() - read_started
            counters[route]["calls"] += 1
            counters[route]["returned_bytes"] += raw.nbytes
            if packed_reader is not None:
                raw_tensor = _packed_raw_tensor(reader, raw, packed_info[1:], samples,
                                                use_selected, upper - lower, device, measure)
            else:
                with measure("gds_raw_h2d", gpu=True):
                    raw_tensor = torch.as_tensor(raw, device=device)
            with measure("gds_decode", gpu=True):
                if not use_selected and packed_reader is None:
                    if sample_tensor is None:
                        sample_tensor = _device_sample_index(reader,samples,device)
                    raw_tensor = raw_tensor.index_select(1, sample_tensor)
                for layer in range(int(group_steps.max())):
                    raw_indices = first + layer
                    live = np.flatnonzero((group_steps > layer) & (raw_indices >= lower) & (raw_indices < upper))
                    if not len(live):
                        continue
                    live_tensor = torch.as_tensor(live, device=device)
                    take_indices = torch.as_tensor(raw_indices[live] - lower, device=device)
                    operands = raw_tensor.index_select(0, take_indices).to(torch.int64)
                    operands = torch.bitwise_left_shift(operands, 2 * layer)
                    values = torch.bitwise_or(codes.index_select(0, live_tensor), operands)
                    codes.index_copy_(0, live_tensor, values)
                    del operands, values
            del raw_tensor, raw
        with measure("gds_decode", gpu=True):
            sentinel = torch.as_tensor([(1 << (2 * int(value))) - 1 for value in group_steps],
                                       dtype=torch.int64, device=device)[:, None, None]
            missing = codes == sentinel
            reference = (codes == 0) & ~missing
            group_reference = reference.sum(dim=(1, 2), dtype=torch.int64)
            group_called = (~missing).sum(dim=(1, 2), dtype=torch.int64)
            missing_count = missing.sum(dim=2, dtype=torch.int8)
            group_half_missing = (missing_count == 1).sum(dtype=torch.int64)
        # A single integer transfer per group; no float32/TF32 frequency path.
        with measure("gds_metadata_d2h", gpu=True):
            summary = torch.cat((group_reference, group_called, group_half_missing.reshape(1))).cpu().numpy()
        reference_ac[outputs] = summary[:len(group)]
        called_alleles[outputs] = summary[len(group):2 * len(group)]
        half_missing += int(summary[-1])
        group_summary = _allele_frequency_summary(reference_ac[outputs], called_alleles[outputs], n)
        eligible = np.arange(len(group)) if minimum_mac is None else np.flatnonzero(group_summary[2] >= minimum_mac)
        if len(eligible):
            with measure("gds_decode", gpu=True):
                eligible_tensor = torch.as_tensor(eligible, device=device)
                dosage = reference.index_select(0, eligible_tensor).sum(dim=2, dtype=torch.uint8)
                whole_missing = missing_count.index_select(0, eligible_tensor) > 0
                dosage.masked_fill_(whole_missing, 3)
                if use_selected:
                    if sample_order is None:
                        sample_order = _device_sample_index(reader,samples,device,permutation=permutation)
                    dosage = dosage.index_select(1, sample_order)
            if resident:
                selected_dosage = dosage
            else:
                with measure("gds_legacy_dosage_d2h", gpu=True):
                    selected_dosage = dosage.cpu().numpy()
            for output, row in zip(outputs[eligible], selected_dosage):
                dosages[int(output)] = row
            del eligible_tensor, dosage, whole_missing, selected_dosage
        del codes, sentinel, missing, reference, missing_count
        del group_reference, group_called, group_half_missing
    summaries = _allele_frequency_summary(reference_ac, called_alleles, n)
    positions = np.arange(m) if minimum_mac is None else np.flatnonzero(summaries[2] >= minimum_mac)
    ref_af, missing_rate, initial_mac, ref_ac, called = [value[positions] for value in summaries]
    if resident:
        from .gds_device import DeviceMinorBlock
        # Compact integer payload remains on the decoder device, in caller order.
        with measure("gds_resident_prepare", gpu=True):
            absent = torch.full((n,), 3, dtype=torch.uint8, device=device)
            dosage = (torch.stack([dosages.get(int(output), absent) for output in positions], dim=1)
                      if len(positions) else torch.empty((n, 0), dtype=torch.uint8, device=device))
            flip = torch.as_tensor(ref_af >= 0.5, device=device)
            dosage = torch.where((dosage != 3) & flip[None, :], 2 - dosage, dosage)
        if minimum_mac is not None:
            reader._record_minor_coverage(steps, summaries, half_missing, len(positions))
        return DeviceMinorBlock(dosage, samples, variants[positions], ref_af,
                                initial_mac, missing_rate, ref_ac, called)
    dosage = np.empty((n, len(positions)), dtype=np.float64)
    for column, output in enumerate(positions):
        # Zero-layer sites and empty sample sets retain all-missing semantics.
        dosage[:, column] = dosages.get(int(output), 3)
    dosage[dosage == 3] = np.nan
    ref_af, missing_rate, initial_mac, ref_ac, called = [value[positions] for value in summaries]
    dosage[:, ref_af >= 0.5] = 2 - dosage[:, ref_af >= 0.5]
    row, col = np.nonzero((dosage != 0) | np.isnan(dosage))
    if minimum_mac is not None:
        reader._record_minor_coverage(steps, summaries, half_missing, len(positions))
    return SparseMinorBlock(row, col, dosage[row, col], samples, variants[positions], ref_af,
                            initial_mac, missing_rate, ref_ac, called)
