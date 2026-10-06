"""Private source29-pinned fused FP64 probability-root candidate.

No dense/eigen/core arithmetic is replaced. GPU root/branch/P equivalence is
validated on the chr21 significant-union benchmark; reduction order may differ from Torch sum. CPU contracts
verify formulas and branches, not GPU rounding or performance.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None

SOURCE_SHA256='be86c19d1f40003f66f34833f7501fc68a7b5149088a2596092d926f4be7f802'
MAX_FUSED_SPECTRUM=4096
_METRICS=dict(candidate_calls=0,fused_root_kernel_launches=0,fused_root_rows=0,
              original_large_spectrum_calls=0,max_root_workspace_bytes=0)
_PINNED_ORIGINAL=None

def _pinned_original():
    global _PINNED_ORIGINAL
    if _PINNED_ORIGINAL is None:
        import staar_phewas._statistics_sync as original
        path=Path(original.__file__).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256:
            raise RuntimeError('fused root source29 probability implementation hash mismatch')
        # Root runtime wrapper must capture this function before replacing it.
        # Access the immutable saved original to avoid recursion after install.
        _PINNED_ORIGINAL=original.quadratic_form_sf_batch
    return _PINNED_ORIGINAL

if triton is not None:
    @triton.jit
    def _root_kernel(S,Q,ROOT,UNCONVERGED,M,SS0,SS1,QS,
                     BLOCK:tl.constexpr):
        row=tl.program_id(0)
        index=tl.arange(0,BLOCK)
        spectrum=tl.load(S+row*SS0+index*SS1,index<M,other=0).to(tl.float64)
        q=tl.load(Q+row*QS).to(tl.float64)
        mean=tl.sum(spectrum,0)
        lower=tl.where(q>mean,-0.01,-M.to(tl.float64)/(2.0*q))
        upper=tl.full((),0.499995,tl.float64)
        root=tl.full((),0.0,tl.float64)
        alive=tl.full((),True,tl.int1)
        iteration=tl.full((),0,tl.int32)
        # Same original width and exact derivative-zero stopping predicates.
        # A row exits when inactive; source29's later inactive loop iterations
        # only copy its last root and therefore do not change the result.
        while (alive & (tl.abs(upper-lower)>1e-8)) & (iteration<2048):
            root=(upper+lower)/2.0
            derivative=tl.sum(spectrum/(1.0-2.0*spectrum*root),0)-q
            updating=derivative!=0.0
            upper=tl.where(updating & (derivative>0.0),root,upper)
            lower=tl.where(updating & ~(derivative>0.0),root,lower)
            alive=updating
            iteration+=1
        tl.store(ROOT+row,root)
        tl.store(UNCONVERGED+row,alive & (tl.abs(upper-lower)>1e-8))

def fused_roots_cuda(scaled,q):
    """One program per FP64 spectrum; root + convergence flag outputs only."""
    if triton is None:raise RuntimeError('fused probability roots require Triton')
    if not scaled.is_cuda or q.device!=scaled.device:raise ValueError('same CUDA device required')
    if scaled.dtype!=torch.float64 or q.dtype!=torch.float64:raise ValueError('FP64 probability inputs required')
    if scaled.ndim!=2 or q.shape!=(len(scaled),) or not 1<=scaled.shape[1]<=MAX_FUSED_SPECTRUM:
        raise ValueError('unsupported fused probability geometry')
    rows,m=scaled.shape
    root=torch.empty((rows,),dtype=torch.float64,device=scaled.device)
    unconverged=torch.empty((rows,),dtype=torch.bool,device=scaled.device)
    _METRICS['max_root_workspace_bytes']=max(_METRICS['max_root_workspace_bytes'],9*rows)
    with torch.cuda.device(scaled.device):
        _root_kernel[(rows,)](scaled,q,root,unconverged,m,*scaled.stride(),q.stride(0),
                             BLOCK=triton.next_power_of_2(m),num_warps=4,num_stages=1,
                             enable_fp_fusion=False)
    _METRICS['fused_root_kernel_launches']+=1
    _METRICS['fused_root_rows']+=rows
    return root,unconverged

def reference_roots_cpu(scaled,q):
    """CPU mathematical/control reference, preserving source29 Torch reductions."""
    if scaled.device.type!='cpu' or q.device.type!='cpu':raise ValueError('CPU reference only')
    if scaled.dtype!=torch.float64 or q.dtype!=torch.float64:raise ValueError('FP64 probability inputs required')
    lower = torch.where(q > scaled.sum(dim=1), q.new_tensor(-.01), -torch.full_like(q, scaled.shape[1]) / (2*q))
    upper, root = torch.full_like(q,.499995), torch.zeros_like(q)
    width = float((upper-lower).abs().max())
    iterations = min(2048, max(1, math.ceil(math.log2(width)-math.log2(1e-8))+2)) if math.isfinite(width) and width>0 else 2048
    alive = torch.ones_like(q,dtype=torch.bool)
    for _ in range(iterations):
        active = alive & ~((upper-lower).abs()<=1e-8)
        root = torch.where(active,(upper+lower)/2,root)
        derivative = (scaled/(1-2*scaled*root[:,None])).sum(dim=1)-q
        updating = active & (derivative!=0)
        upper = torch.where(updating & (derivative>0),root,upper)
        lower = torch.where(updating & ~(derivative>0),root,lower)
        alive = updating
    return root,alive & ~((upper-lower).abs()<=1e-8)

def host_flags(*flags):
    return torch.stack(flags).detach().to(device='cpu').tolist()

def quadratic_form_sf_batch(statistic, eigenvalues):
    """Source29 complete-spectrum formulas with candidate fused FP64 roots.

    Eigenvalues are computed in the caller's core dtype. This probability-only
    boundary uses FP64 because signed-root cancellation near the mean and
    positive small tails cannot be stored reliably in FP32.
    """
    original=_pinned_original()
    if not eigenvalues.is_cuda or not statistic.is_cuda:
        return original(statistic,eigenvalues)
    if eigenvalues.ndim==2 and eigenvalues.shape[1]>MAX_FUSED_SPECTRUM:
        _METRICS['original_large_spectrum_calls']+=1
        return original(statistic,eigenvalues)
    _METRICS['candidate_calls']+=1
    from staar_phewas.statistics import DegenerateTestError
    raw = eigenvalues.to(torch.float64)
    q_raw = statistic.to(torch.float64)
    if raw.ndim != 2 or q_raw.shape != (len(raw),):
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    finite_spectrum, finite_q, negative_q = host_flags(
        torch.isfinite(raw).all(), torch.isfinite(q_raw).all(), (q_raw < 0).any())
    if not finite_spectrum or not finite_q or negative_q:
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max(dim=1).values
    nonzero = q_raw != 0
    invalid_spectrum, has_nonzero = host_flags((maximum <= 0).any(), nonzero.any())
    if invalid_spectrum:
        raise DegenerateTestError('SKAT covariance has no eigenvalue at or above 1e-8')
    result = torch.ones_like(q_raw)
    if not has_nonzero:return result
    raw, spectrum, q_raw, maximum = raw[nonzero], spectrum[nonzero], q_raw[nonzero], maximum[nonzero]
    scaled, q = spectrum / maximum[:, None], q_raw / maximum
    root,unconverged_rows=fused_roots_cuda(scaled,q)
    moment = root.abs()<1e-4
    saddle = ~moment
    # One small transfer preserves the convergence failure priority and both
    # original branch predicates; the root reduction is the candidate kernel.
    unconverged, has_moment, has_saddle = host_flags(
        unconverged_rows.any(), moment.any(), saddle.any())
    if unconverged:
        raise ArithmeticError('STAAR saddlepoint bisection did not converge')
    probabilities = torch.empty_like(q)
    if has_moment:
        moments = raw[moment]
        c1,c2,c4 = moments.sum(dim=1),moments.square().sum(dim=1),moments.pow(4).sum(dim=1)
        if bool(((c2<=0)|(c4<=0)).any()):raise DegenerateTestError('SKAT moment fallback has zero variance')
        dof = c2.square()/c4
        adjusted = (q_raw[moment]-c1)/torch.sqrt(2*c2)*torch.sqrt(2*dof)+dof
        probabilities[moment] = torch.where(adjusted<=0,1.,torch.special.gammaincc(dof/2,adjusted/2))
    if has_saddle:
        s,r = scaled[saddle],root[saddle]
        denominator = 1-2*s*r[:,None]
        cumulant = -.5*torch.log(denominator).sum(dim=1)
        w2 = 2*(r*q[saddle]-cumulant)
        if bool((w2<=0).any()):raise ArithmeticError('STAAR saddlepoint has an invalid signed root')
        signed_root = torch.copysign(torch.sqrt(w2),r)
        second_derivative = 2*(s.square()/denominator.square()).sum(dim=1)
        v = r*torch.sqrt(second_derivative)
        z = signed_root+torch.log(v/signed_root)/signed_root
        probabilities[saddle] = .5*torch.erfc(z/math.sqrt(2))
    result[nonzero] = probabilities
    return result

def reference_probability_cpu(statistic, eigenvalues):
    """Original complete-spectrum Saddle/moment formulas, one row per weight.

    Eigenvalues are computed in the caller's core dtype. This probability-only
    boundary uses FP64 because signed-root cancellation near the mean and
    positive small tails cannot be stored reliably in FP32.
    """
    if eigenvalues.device.type!='cpu' or statistic.device.type!='cpu':raise ValueError('CPU reference only')
    from staar_phewas.statistics import DegenerateTestError
    raw = eigenvalues.to(torch.float64)
    q_raw = statistic.to(torch.float64)
    if raw.ndim != 2 or q_raw.shape != (len(raw),):
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    finite_spectrum, finite_q, negative_q = host_flags(
        torch.isfinite(raw).all(), torch.isfinite(q_raw).all(), (q_raw < 0).any())
    if not finite_spectrum or not finite_q or negative_q:
        raise ValueError('quadratic form requires a nonnegative finite statistic and finite spectrum')
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max(dim=1).values
    nonzero = q_raw != 0
    invalid_spectrum, has_nonzero = host_flags((maximum <= 0).any(), nonzero.any())
    if invalid_spectrum:
        raise DegenerateTestError('SKAT covariance has no eigenvalue at or above 1e-8')
    result = torch.ones_like(q_raw)
    if not has_nonzero:return result
    raw, spectrum, q_raw, maximum = raw[nonzero], spectrum[nonzero], q_raw[nonzero], maximum[nonzero]
    scaled, q = spectrum / maximum[:, None], q_raw / maximum
    root,unconverged_rows=reference_roots_cpu(scaled,q)
    moment = root.abs()<1e-4
    saddle = ~moment
    # One small transfer preserves the convergence failure priority and both
    # original branch predicates without changing any reduction or root step.
    unconverged, has_moment, has_saddle = host_flags(
        unconverged_rows.any(), moment.any(), saddle.any())
    if unconverged:
        raise ArithmeticError('STAAR saddlepoint bisection did not converge')
    probabilities = torch.empty_like(q)
    if has_moment:
        moments = raw[moment]
        c1,c2,c4 = moments.sum(dim=1),moments.square().sum(dim=1),moments.pow(4).sum(dim=1)
        if bool(((c2<=0)|(c4<=0)).any()):raise DegenerateTestError('SKAT moment fallback has zero variance')
        dof = c2.square()/c4
        adjusted = (q_raw[moment]-c1)/torch.sqrt(2*c2)*torch.sqrt(2*dof)+dof
        probabilities[moment] = torch.where(adjusted<=0,1.,torch.special.gammaincc(dof/2,adjusted/2))
    if has_saddle:
        s,r = scaled[saddle],root[saddle]
        denominator = 1-2*s*r[:,None]
        cumulant = -.5*torch.log(denominator).sum(dim=1)
        w2 = 2*(r*q[saddle]-cumulant)
        if bool((w2<=0).any()):raise ArithmeticError('STAAR saddlepoint has an invalid signed root')
        signed_root = torch.copysign(torch.sqrt(w2),r)
        second_derivative = 2*(s.square()/denominator.square()).sum(dim=1)
        v = r*torch.sqrt(second_derivative)
        z = signed_root+torch.log(v/signed_root)/signed_root
        probabilities[saddle] = .5*torch.erfc(z/math.sqrt(2))
    result[nonzero] = probabilities
    return result

def execution_metadata(*,reset=False):
    report=dict(_METRICS,pinned_probability_source_sha256=SOURCE_SHA256,
                max_fused_spectrum=MAX_FUSED_SPECTRUM,probability_dtype='float64',
                dense_core_changed=False,eigen_solver_changed=False,
                root_reduction='Triton tl.sum FP64; chr21 significant-union gate passed',
                root_tolerance=1e-8,moment_threshold=1e-4,
                invalid_signed_root_clamped=False,
                functions_retained='torch.special.gammaincc and torch.erfc',
                adoption_requires='GPU root/branch/P differential on failed subset and complete suite')
    if reset:
        for key in _METRICS:_METRICS[key]=0
    return report
