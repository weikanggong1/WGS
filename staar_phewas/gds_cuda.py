"""Lossless CUDA Bit2 decoding for early individual-variant MAC selection.

The SDK reads bounded uint8 slabs. CUDA computes integer allele counts and
transfers only eligible dosages; source AF and MAC formulas stay float64 on
the CPU. Association calculations and imputation are performed elsewhere.
"""

from __future__ import annotations

import numpy as np


def native_minor_block(reader, variants, samples, *, device, minimum_mac=None):
    """Return canonical COO minor dosages, preserving caller sample/site order."""
    import torch
    from .gds import SparseMinorBlock, _allele_frequency_summary

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
        sample_mask = np.zeros(reader.n_samples, dtype=bool)
        sample_mask[samples] = True
        selection = np.repeat(sample_mask, reader.ploidy)
        sample_order = torch.as_tensor(np.argsort(np.argsort(samples)), device=device)
    sample_tensor = torch.as_tensor(samples, device=device)
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
        route = "selected" if use_selected else "flat"
        codes = torch.zeros((len(group), n, reader.ploidy), dtype=torch.int64, device=device)
        for lower in range(int(first[0]), int(last[-1]), max_layers):
            upper = min(int(last[-1]), lower + max_layers)
            if use_selected:
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
            counters[route]["calls"] += 1
            counters[route]["returned_bytes"] += raw.nbytes
            raw_tensor = torch.as_tensor(raw, device=device)
            if not use_selected:
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
        sentinel = torch.as_tensor([(1 << (2 * int(value))) - 1 for value in group_steps],
                                   dtype=torch.int64, device=device)[:, None, None]
        missing = codes == sentinel
        reference = (codes == 0) & ~missing
        group_reference = reference.sum(dim=(1, 2), dtype=torch.int64)
        group_called = (~missing).sum(dim=(1, 2), dtype=torch.int64)
        missing_count = missing.sum(dim=2, dtype=torch.int8)
        group_half_missing = (missing_count == 1).sum(dtype=torch.int64)
        # A single integer transfer per group; no float32/TF32 frequency path.
        summary = torch.cat((group_reference, group_called, group_half_missing.reshape(1))).cpu().numpy()
        reference_ac[outputs] = summary[:len(group)]
        called_alleles[outputs] = summary[len(group):2 * len(group)]
        half_missing += int(summary[-1])
        group_summary = _allele_frequency_summary(reference_ac[outputs], called_alleles[outputs], n)
        eligible = np.arange(len(group)) if minimum_mac is None else np.flatnonzero(group_summary[2] >= minimum_mac)
        if len(eligible):
            eligible_tensor = torch.as_tensor(eligible, device=device)
            dosage = reference.index_select(0, eligible_tensor).sum(dim=2, dtype=torch.uint8)
            whole_missing = missing_count.index_select(0, eligible_tensor) > 0
            dosage.masked_fill_(whole_missing, 3)
            if use_selected:
                dosage = dosage.index_select(1, sample_order)
            selected_dosage = dosage.cpu().numpy()
            for output, row in zip(outputs[eligible], selected_dosage):
                dosages[int(output)] = row
            del eligible_tensor, dosage, whole_missing, selected_dosage
        del codes, sentinel, missing, reference, missing_count
        del group_reference, group_called, group_half_missing
    summaries = _allele_frequency_summary(reference_ac, called_alleles, n)
    positions = np.arange(m) if minimum_mac is None else np.flatnonzero(summaries[2] >= minimum_mac)
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
