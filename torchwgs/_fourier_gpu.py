"""Bounded float64 Davies Fourier rules without a terms-by-spectrum workspace.

Each CUDA program accumulates a small frequency tile through the spectrum.
NVIDIA double-precision libdevice functions are used throughout; TF32 and
fast approximate transcendental functions are not used for tail inference.
"""
import torch

try:
    import triton
    import triton.language as tl
    try:
        from triton.language import libdevice as _ld
    except ImportError:
        from triton.language.extra.cuda import libdevice as _ld
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _fourier_rules(spectrum, points, terms, spacing, gaussian, extra,
                       auxiliary, output, eigen_count: tl.constexpr,
                       chunks, frequency_tile: tl.constexpr,
                       eigen_tile: tl.constexpr):
        job, chunk = tl.program_id(0), tl.program_id(1)
        offset = chunk*frequency_tile+tl.arange(0, frequency_tile)
        last = tl.load(terms+job)
        valid = offset <= last
        step = tl.load(spacing+job)
        point = tl.load(points+job)
        frequency = _ld.mul_rn(offset.to(tl.float64)+0.5, step)
        phase_sum = tl.full((frequency_tile,), 0., tl.float64)
        log_sum = tl.full((frequency_tile,), 0., tl.float64)
        for start in range(0, eigen_count, eigen_tile):
            col = start+tl.arange(0, eigen_tile)
            eigen = tl.load(spectrum+col, col < eigen_count, 0.).to(tl.float64)
            argument = 2.*frequency[:, None]*eigen[None, :]
            phase_sum += tl.sum(_ld.atan(argument), 1)
            log_sum += tl.sum(_ld.log1p(argument*argument), 1)
        # Explicit round-to-nearest operations prevent multiply/add
        # contraction on Triton 2 as well as newer compilers. Triton 2's
        # launcher does not accept the enable_fp_fusion option.
        phase = _ld.add_rn(_ld.mul_rn(-point, frequency), phase_sum/2.)
        log_magnitude = _ld.add_rn(
            -tl.load(gaussian+job)*frequency*frequency/2., -log_sum/4.)
        magnitude = tl.where(valid & (log_magnitude >= -50.), _ld.exp(log_magnitude), 0.)
        # Triton 2 cannot construct a nonzero fp64 full() constant. Bitcast
        # the exact IEEE-754 double bits instead of first rounding pi to fp32.
        pi = tl.full((frequency_tile,), 0x400921FB54442D18, tl.int64).to(tl.float64, bitcast=True)
        amplitude = (step/pi)*magnitude/frequency
        damping = tl.load(extra+job)*frequency*frequency/2.
        factor = tl.where(damping > 50., 1., -_ld.expm1(-damping))
        amplitude *= tl.where(tl.load(auxiliary+job), factor, 1.)
        integral = tl.sum(_ld.sin(phase)*amplitude, 0)
        absolute = tl.sum(_ld.mul_rn(
            _ld.add_rn(_ld.mul_rn(point, frequency), phase_sum/2.), amplitude), 0)
        index = job*chunks+chunk
        tl.store(output+index, integral)
        tl.store(output+index+tl.num_programs(0)*chunks, absolute)


def available(device):
    return (triton is not None and torch.device(device).type == 'cuda'
            and torch.version.hip is None)


def fourier_rules(spectrum, points, terms, spacing, gaussian, extra, auxiliary,
                  *, maximum_terms):
    """Return one integral and absolute-error sum per independent rule.

    Inputs have already been checked by the Davies controller. ``terms``
    includes the last frequency index, as in the original Fourier loop.
    The largest temporary is two doubles per rule and frequency tile;
    spectrum length does not multiply the global workspace.
    """
    if not available(spectrum.device):
        raise RuntimeError('Fused Fourier rules require CUDA and NVIDIA Triton.')
    if spectrum.dtype != torch.float64 or points.dtype != torch.float64:
        raise ValueError('Davies Fourier rules require float64 inputs.')
    if spectrum.ndim != 1 or points.ndim != 1 or spectrum.numel() == 0:
        raise ValueError('A nonempty spectrum and a point vector are required.')
    jobs = points.numel()
    if any(t.numel() != jobs or t.device != spectrum.device for t in
           (points, terms, spacing, gaussian, extra, auxiliary)):
        raise ValueError('All Fourier rule vectors must share length and device.')
    if maximum_terms < 0:
        raise ValueError('maximum_terms must be nonnegative.')
    if jobs == 0:
        return points.clone(), points.clone()
    frequency_tile = 32 if spectrum.numel() <= 64 else 16
    eigen_tile = min(128, triton.next_power_of_2(spectrum.numel()))
    chunks = triton.cdiv(maximum_terms+1, frequency_tile)
    output = torch.empty((2, jobs, chunks), device=spectrum.device, dtype=torch.float64)
    vectors = [t.reshape(-1).contiguous() for t in
               (points, terms, spacing, gaussian, extra, auxiliary)]
    with torch.cuda.device(spectrum.device):
        _fourier_rules[(jobs, chunks)](
            spectrum.contiguous(), *vectors, output, spectrum.numel(), chunks,
            frequency_tile, eigen_tile, num_warps=4)
    summed = output.sum(-1)
    return summed[0], summed[1]
