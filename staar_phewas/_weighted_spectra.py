"""Complete FP32 weighted spectra and exact proportional-weight reuse.

Original FP32 weighted matrix, exact proportional proof and full eigvalsh stay
unchanged. Real chr21 significant-P validation is described in the benchmark;
CPU contracts check interfaces separately.
"""
from contextlib import contextmanager,nullcontext
import math
import time
import torch

WEIGHTED_BATCH_LIMIT_BYTES=512*2**20
_PINNED=None
_PROFILE_HOOK=None
_METRICS=dict(calls=0,batch_sizes=[],eigen_api_calls=0,complete_representative_spectra=0,
              exact_relation_reused_spectra=0,live_workspace_queries=0,workspace_rejections=0,
              max_estimated_batch_workspace_bytes=0,max_weighted_tensor_bytes=0,
              workspace_query_host_seconds=0.)

def _pinned_statistics():
    from . import statistics
    return statistics

def _profile(label):
    return _PROFILE_HOOK(label) if _PROFILE_HOOK is not None else nullcontext()

@contextmanager
def profile_stages(hook):
    global _PROFILE_HOOK
    old=_PROFILE_HOOK;_PROFILE_HOOK=hook
    try:yield
    finally:_PROFILE_HOOK=old

def workspace_unit_bytes(covariance):
    # Three weighted-matrix-sized buffers plus one full FP32 spectrum output.
    # CUDA allocator cap/live margin remain final controls for solver scratch.
    return 3*covariance.numel()*covariance.element_size()+4*covariance.shape[0]

def _batch_capacity(covariance,remaining):
    budget=WEIGHTED_BATCH_LIMIT_BYTES
    if covariance.is_cuda:
        from staar_phewas import tf32
        start=time.perf_counter()
        allocated=torch.cuda.memory_allocated(covariance.device)
        reserved=torch.cuda.memory_reserved(covariance.device)
        with torch.cuda.device(covariance.device):free,_=torch.cuda.mem_get_info()
        _METRICS['workspace_query_host_seconds']+=time.perf_counter()-start
        _METRICS['live_workspace_queries']+=1
        available=tf32._product_workspace_availability(allocated=allocated,reserved=reserved,free=free,
            limit=min(20*2**30,tf32._memory_limit_bytes),reserve=tf32._memory_reserve_bytes)['available_bytes']
        budget=min(budget,available)
    unit=workspace_unit_bytes(covariance)
    capacity=min(remaining,budget//max(unit,1))
    if capacity<1:
        _METRICS['workspace_rejections']+=1
        raise MemoryError('full FP32 weighted eigensolve batch cannot fit fresh live/process workspace; no truncation/precision retry')
    _METRICS['max_estimated_batch_workspace_bytes']=max(_METRICS['max_estimated_batch_workspace_bytes'],capacity*unit)
    _METRICS['max_weighted_tensor_bytes']=max(_METRICS['max_weighted_tensor_bytes'],capacity*covariance.numel()*covariance.element_size())
    return capacity

def scale_exact_spectra(solved,mapping,exact,pivots):
    """Vectorize original independent FP64 scalars; no changed reductions."""
    representative_rows=torch.tensor(mapping,dtype=torch.int64,device=exact.device)
    representative_pivots=pivots[representative_rows]
    numerators=exact.gather(1,representative_pivots[:,None])[:,0]
    denominators=exact[representative_rows,representative_pivots]
    scales=numerators/denominators
    full_fp32=torch.stack([solved[i] for i in mapping])
    return full_fp32.double()*scales.square()[:,None]

def native_weighted_spectra(covariance, weights):
    """Complete FP32 spectra; reuse only exact proportional input weights.

    Products of two FP32 numbers are represented exactly in FP64 (at most
    48 significand bits). Equal cross-products therefore prove proportionality,
    without a tolerance or annotation grouping approximation. Dense weighted
    matrices and eigvalsh remain FP32. Only spectrum scalar scaling uses FP64.
    """
    original=_pinned_statistics()
    _native_weight_relations=original._native_weight_relations
    _STATISTICS_METADATA=original._STATISTICS_METADATA
    record_gpu_eigen_route=original.record_gpu_eigen_route
    _METRICS['calls']+=1
    rows = weights.T.contiguous()
    with _profile('weight_relations'):
        exact, pivots, related = _native_weight_relations(rows)
    representatives, mapping = [], []
    for j in range(len(rows)):
        found = next((i for i in representatives if related[i][j]), None)
        if found is None:
            representatives.append(j); found = j
        mapping.append(found)
    # Bound only weighted-matrix materialization; eigensolver workspace is
    # additionally covered by the pipeline's process allocation limit.
    if covariance.dtype!=torch.float32 or covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or covariance.shape[0]!=rows.shape[1] or covariance.device!=weights.device:
        raise ValueError('weighted spectrum candidate requires aligned square FP32 covariance')
    per_matrix = covariance.numel() * covariance.element_size()
    solved = {}
    start=0
    while start<len(representatives):
        batch_size=_batch_capacity(covariance,len(representatives)-start)
        indices = representatives[start:start + batch_size]
        w = rows[indices]
        with _profile('weighted_matrix_build'):
            matrices = covariance[None, :, :] * w[:, :, None] * w[:, None, :]
        with _profile('eigvalsh'):
            spectra = torch.linalg.eigvalsh(matrices, UPLO="U")
        record_gpu_eigen_route(matrices)
        _METRICS['batch_sizes'].append(len(indices))
        _METRICS['eigen_api_calls']+=1
        for index, spectrum in zip(indices, spectra):
            solved[index] = spectrum
        start+=len(indices)
    _STATISTICS_METADATA["native_complete_spectra"] += len(representatives)
    _STATISTICS_METADATA["native_spectra_reused"] += len(rows) - len(representatives)
    _METRICS['complete_representative_spectra']+=len(representatives)
    _METRICS['exact_relation_reused_spectra']+=len(rows)-len(representatives)
    with _profile('spectrum_scaling'):
        return scale_exact_spectra(solved,mapping,exact,pivots)


def execution_metadata(*,reset=False):
    report=dict(_METRICS,batch_sizes=list(_METRICS['batch_sizes']),
        batch_limit_bytes=WEIGHTED_BATCH_LIMIT_BYTES,
        weighted_matrix_dtype='float32',eigensolver='torch.linalg.eigvalsh UPLO=U full spectrum',
        spectrum_scalar_scale_dtype='float64',exact_proportional_relations=True,
        eigenvalues_truncated=False,fp64_dense_matrix_products=False,
        scalar_scale_reductions_changed=False,
        workspace_policy='three matrix buffers + full spectrum; fresh live/process cap before each batch',
        workspace_limitation='estimate is not CUDA reservation or measured solver scratch bound; OOM propagates',
        GPU_batch_equivalence='chr21 significant-union gate passed; see published benchmark scope')
    if reset:
        for key,value in _METRICS.items():_METRICS[key]=[] if isinstance(value,list) else 0. if isinstance(value,float) else 0
    return report
