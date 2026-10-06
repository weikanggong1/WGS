"""Private six-state CUDA adapter. No CUDA/Triton import at module import."""
import numpy as np

STATE_MAPPING = 'ref-homo-first-six-v1'
LIMIT_BYTES = 20 * 2**30
_UNPACK_KERNEL = None


def _indices(values, size):
    if values is None:
        return np.arange(size, dtype=np.int64)
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a >= size) or np.any(a < 0):
        raise ValueError('indices must be one-dimensional in-range integers')
    if len(np.unique(a)) != len(a):
        raise ValueError('duplicate indices are not original reader semantics')
    return a.astype(np.int64, copy=False)


def validate_payload(packed, n, columns=None, samples=None, *, mapping=STATE_MAPPING, layout='plane-major', bitorder='little'):
    if mapping != STATE_MAPPING or layout != 'plane-major' or bitorder != 'little':
        raise ValueError('unsupported explicit cache encoding')
    if type(n) is not int or n < 0:
        raise ValueError('invalid sample count')
    a = np.asarray(packed)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[0] != 3 or a.shape[2] != (n+7)//8:
        raise ValueError('require uint8 P[3,m,ceil(n/8)]')
    if n % 8 and np.any(a[:, :, -1] & np.uint8((255 << (n % 8)) & 255)):
        raise ValueError('nonzero high padding bits')
    return a, _indices(columns, a.shape[1]), _indices(samples, n)


def workspace_bytes(m, n, B):
    """Conservative payload, output, conversion intermediates and index budget.

    32 bytes/cell covers raw state, int64 index expansion, lookup/count/dosage
    temporaries; 64 MiB reserve covers small reductions/allocator workspace.
    This is a fail-closed allocation estimate, not an observed peak guarantee.
    """
    return 3*m*B + 32*m*n + 8*n + 256*m + 64*2**20


def check_budget(required, allocated, free, *, limit=LIMIT_BYTES):
    if required > limit-allocated or required > free:
        raise MemoryError('six-state cache workspace exceeds live memory or 20 GiB process budget')


def unpack_numpy(packed, n, columns=None, samples=None, **encoding):
    """Small CPU oracle only; never a CUDA fallback."""
    a, cols, rows = validate_payload(packed, n, columns, samples, **encoding)
    planes = np.unpackbits(a[:, cols, :], axis=2, bitorder='little')[:, :, :n]
    state = (planes[0] | planes[1] << 1 | planes[2] << 2)[:, rows]
    if np.any(state > 5):
        raise ValueError('invalid six-state code')
    return state


def _get_unpack_kernel():
    """Build one JITFunction per process; Triton reuses its compiled signatures.

    M/N/C are runtime scalars. First launch still includes real compilation;
    caller must record that separately from warm replay, without dummy data.
    """
    global triton, tl, _UNPACK_KERNEL
    if _UNPACK_KERNEL is not None:
        return _UNPACK_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def unpack(P, S, O, M, N, C, BLOCK: tl.constexpr):
        idx = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        valid = idx < M*N
        v = idx // N
        s = tl.load(S + idx % N, valid, other=0)
        offset = v*C + s//8
        shift = s % 8
        b0 = (tl.load(P+offset, valid, other=0) >> shift) & 1
        b1 = (tl.load(P+M*C+offset, valid, other=0) >> shift) & 1
        b2 = (tl.load(P+2*M*C+offset, valid, other=0) >> shift) & 1
        tl.store(O+idx, b0 | (b1 << 1) | (b2 << 2), valid)

    _UNPACK_KERNEL = unpack
    return _UNPACK_KERNEL


def decode_cuda(packed, n, columns=None, samples=None, *, device='cuda:0', **encoding):
    """Return CUDA uint8 states [requested_variant, requested_sample]."""
    a, cols, rows = validate_payload(packed, n, columns, samples, **encoding)
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('CUDA decoder requires a CUDA device')
    free, _ = torch.cuda.mem_get_info(device)
    check_budget(workspace_bytes(len(cols), len(rows), a.shape[2]),
                 torch.cuda.memory_allocated(device), free)
    # Missing backend errors propagate, with no old-reader fallback.
    unpack = _get_unpack_kernel()

    p = torch.from_numpy(np.ascontiguousarray(a[:, cols, :])).to(device)
    s = torch.as_tensor(rows, device=device)
    out = torch.empty((len(cols), len(rows)), dtype=torch.uint8, device=device)
    if out.numel():
        with torch.cuda.device(device):
            unpack[(triton.cdiv(out.numel(), 256),)](p, s, out, len(cols), len(rows), a.shape[2], BLOCK=256)
        if bool((out > 5).any()):
            raise ValueError('invalid six-state code')
    return out


def states_to_minor_block(states, sample_indices, variant_indices, *, minimum_mac=None):
    """Raw state -> original union-oriented DeviceMinorBlock, no imputation."""
    import torch
    from ..gds import _allele_frequency_summary
    from staar_phewas.gds_device import DeviceMinorBlock
    if states.dtype != torch.uint8 or states.ndim != 2 or states.device.type != 'cuda':
        raise ValueError('require CUDA uint8 states[variant,sample]')
    variants, samples = np.asarray(variant_indices), np.asarray(sample_indices)
    if variants.ndim != 1 or samples.ndim != 1 or variants.dtype.kind not in 'iu' or samples.dtype.kind not in 'iu':
        raise ValueError('require integer source index vectors')
    if states.shape != (len(variants), len(samples)):
        raise ValueError('index binding shape mismatch')
    if np.any(variants < 0) or np.any(samples < 0) or len(np.unique(samples)) != len(samples) or len(np.unique(variants)) != len(variants):
        raise ValueError('invalid original index binding')
    if minimum_mac is not None and (not np.isfinite(minimum_mac) or minimum_mac < 0):
        raise ValueError('invalid minimum MAC')
    free, _ = torch.cuda.mem_get_info(states.device)
    check_budget(workspace_bytes(*states.shape, 0), torch.cuda.memory_allocated(states.device), free)
    if bool((states > 5).any()):
        raise ValueError('invalid six-state code')
    summary = _integer_summary(states).cpu().numpy()
    summaries = _allele_frequency_summary(summary[0], summary[1], len(samples))
    selected = np.arange(len(variants)) if minimum_mac is None else np.flatnonzero(summaries[2] >= minimum_mac)
    af, missing_rate, initial_mac, ref_ac, called_alleles = [v[selected] for v in summaries]
    selected_states = states.index_select(0, torch.as_tensor(selected, device=states.device))
    dosage = _minor_dosage(selected_states, af)
    return DeviceMinorBlock(dosage, samples.copy(), variants[selected].copy(), af,
                            initial_mac, missing_rate, ref_ac, called_alleles)


def _integer_summary(states):
    """Shared arithmetic helper; CPU allowed only for differential contracts."""
    import torch
    ref = torch.tensor([2, 1, 0, 0, 1, 0], dtype=torch.uint8, device=states.device)
    called = torch.tensor([2, 2, 2, 0, 1, 1], dtype=torch.uint8, device=states.device)
    codes = states.long()
    return torch.stack((ref[codes].sum(1, dtype=torch.int64), called[codes].sum(1, dtype=torch.int64)))


def _minor_dosage(states, af):
    """Shared integer orientation; all partial/whole missing remain sentinel3."""
    import torch
    dosage = torch.where(states < 3, 2-states, 3).to(torch.uint8).T.contiguous()
    flip = torch.as_tensor(af >= .5, device=states.device)
    return torch.where((dosage != 3) & flip[None, :], 2-dosage, dosage)
