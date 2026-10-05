"""Integer BED counts and selected-column decoding on the current device.

The CUDA kernels never materialize the full sample-by-variant code matrix.
Counts use bounded sample tiles and exact integer reductions.  The decoder
writes the retained float32/float64 matrix directly in sample-major order.
Only BED hardcalls are handled: 00=A1/A1, 01=missing, 10=A1/A2, 11=A2/A2.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU readers and GPU PyTorch-only environments.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _count_codes(packed, sample_bytes, sample_shifts, partials,
                     source_stride: tl.constexpr, samples: tl.constexpr,
                     variants: tl.constexpr, tiles: tl.constexpr,
                     block: tl.constexpr):
        variant = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        row = tile * block + tl.arange(0, block)
        valid = row < samples
        byte = tl.load(sample_bytes + row, valid, 0)
        shift = tl.load(sample_shifts + row, valid, 0)
        raw = tl.load(packed + variant * source_stride + byte, valid, 0)
        code = (raw.to(tl.int32) >> shift.to(tl.int32)) & 3
        missing = tl.sum((valid & (code == 1)).to(tl.int32), 0)
        hom_a1 = tl.sum((valid & (code == 0)).to(tl.int32), 0)
        heterozygous = tl.sum((valid & (code == 2)).to(tl.int32), 0)
        hom_a0 = tl.sum((valid & (code == 3)).to(tl.int32), 0)
        offset = variant * tiles + tile
        plane = variants * tiles
        tl.store(partials + offset, missing)
        tl.store(partials + plane + offset, hom_a1)
        tl.store(partials + 2 * plane + offset, heterozygous)
        tl.store(partials + 3 * plane + offset, hom_a0)

    @triton.jit(do_not_specialize=['kept'])
    def _decode_columns(packed, sample_bytes, sample_shifts, columns, output,
                        source_stride: tl.constexpr, samples: tl.constexpr,
                        kept, block: tl.constexpr):
        # The MAC filter changes the retained width almost every block. Keep
        # it dynamic to avoid compiling a decoder for every observed width.
        # Cast before multiplication so large low-level outputs remain safe.
        kept = kept.to(tl.int64)
        program = tl.program_id(0).to(tl.int64)
        offset = program * block + tl.arange(0, block)
        valid = offset < samples * kept
        row = offset // kept
        col = offset % kept
        variant = tl.load(columns + col, valid, 0)
        byte = tl.load(sample_bytes + row, valid, 0)
        shift = tl.load(sample_shifts + row, valid, 0)
        raw = tl.load(packed + variant * source_stride + byte, valid, 0)
        code = (raw.to(tl.int32) >> shift.to(tl.int32)) & 3
        dosage = tl.where(code == 0, 2., tl.where(code == 2, 1., 0.))
        dosage = tl.where(code == 1, float('nan'), dosage)
        tl.store(output + offset, dosage, valid)


def packed_genotype_counts(packed, sample_bytes, sample_shifts):
    """Return int64 [missing, hom_A1, heterozygous, hom_A0] counts per site.

    ``packed`` is contiguous uint8 [variants, ceil(source_samples/4)].
    The sample byte/shift vectors select the aligned analysis rows, including
    arbitrary order or repeated rows. Padding calls are never counted.
    """
    variants, samples = packed.shape[0], sample_bytes.numel()
    if variants == 0 or samples == 0:
        return torch.zeros((4, variants), device=packed.device, dtype=torch.int64)
    block = 2048
    if packed.device.type == 'cuda' and triton is not None:
        tiles = triton.cdiv(samples, block)
        partials = torch.empty((4, variants, tiles), device=packed.device,
                               dtype=torch.int32)
        with torch.cuda.device(packed.device):
            _count_codes[(variants, tiles)](
                packed, sample_bytes, sample_shifts, partials,
                packed.stride(0), samples, variants, tiles, block, num_warps=4)
        return partials.sum(2, dtype=torch.int64)
    # Device-local reference fallback. Tile the selected samples so even this
    # path avoids an expanded floating-point genotype matrix before MAC filtering.
    result = torch.zeros((4, variants), device=packed.device, dtype=torch.int64)
    for start in range(0, samples, block):
        raw = packed[:, sample_bytes[start:start + block]]
        code = torch.bitwise_right_shift(raw, sample_shifts[None, start:start + block])
        code.bitwise_and_(3)
        for plane, value in enumerate((1, 0, 2, 3)):
            result[plane] += (code == value).sum(1)
    return result


def decode_packed_columns(packed, sample_bytes, sample_shifts, columns, lookup):
    """Decode validated block-column indices into contiguous [samples, kept]."""
    samples, kept = sample_bytes.numel(), columns.numel()
    if not samples or not kept:
        return torch.empty((samples, kept), device=packed.device, dtype=lookup.dtype)
    if packed.device.type == 'cuda' and triton is not None:
        output = torch.empty((samples, kept), device=packed.device, dtype=lookup.dtype)
        with torch.cuda.device(packed.device):
            _decode_columns[(triton.cdiv(samples * kept, 1024),)](
                packed, sample_bytes, sample_shifts, columns, output,
                packed.stride(0), samples, kept, 1024, num_warps=4)
        return output
    raw = packed[columns[:, None], sample_bytes[None, :]]
    codes = torch.bitwise_right_shift(raw, sample_shifts[None, :])
    codes.bitwise_and_(3)
    return lookup[codes.long()].T.contiguous()
