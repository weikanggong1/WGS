"""Native IEEE FP32 Burden products with a bounded CUDA workspace.

Score/covariance products keep native TF32. No component reconstruction or
FP64 dense multiplication is performed.
"""
from contextlib import contextmanager
import math
import os
import threading
import time
import torch

_LOCK=threading.RLock()
_METRICS=dict(selected_fp32_mm_calls=0,live_workspace_queries=0,workspace_rejections=0,
              max_estimated_new_workspace_bytes=0,max_output_bytes=0,query_host_seconds=0.)


def estimate_workspace(left_shape,right_shape):
    """Output plus conservative potential copies of both FP32 operands."""
    if len(left_shape)!=2 or len(right_shape)!=2 or left_shape[1]!=right_shape[0]:raise ValueError('Burden product dimensions do not match')
    if any(d<0 for d in (*left_shape,*right_shape)):raise ValueError('invalid product dimension')
    m,k=left_shape;_,n=right_shape
    return dict(output_bytes=4*m*n,layout_copy_allowance_bytes=4*(m*k+k*n),
                required_bytes=4*(m*n+m*k+k*n),cast_bytes=0,fp64_dense_bytes=0)


def guard_workspace(left,right,*,limit_bytes=20*2**30,reserve_bytes=256*2**20):
    from fudan_wgs_toolkit.tf32 import _product_workspace_availability
    estimate=estimate_workspace(left.shape,right.shape)
    _METRICS['max_estimated_new_workspace_bytes']=max(_METRICS['max_estimated_new_workspace_bytes'],estimate['required_bytes'])
    _METRICS['max_output_bytes']=max(_METRICS['max_output_bytes'],estimate['output_bytes'])
    if not estimate['required_bytes']:return
    start=time.perf_counter()
    allocated=torch.cuda.memory_allocated(left.device);reserved=torch.cuda.memory_reserved(left.device)
    with torch.cuda.device(left.device):free,_=torch.cuda.mem_get_info()
    _METRICS['query_host_seconds']+=time.perf_counter()-start
    _METRICS['live_workspace_queries']+=1
    snapshot=_product_workspace_availability(allocated=allocated,reserved=reserved,free=free,
                                            limit=limit_bytes,reserve=reserve_bytes)
    if estimate['required_bytes']>snapshot['available_bytes']:
        _METRICS['workspace_rejections']+=1
        raise MemoryError('IEEE FP32 Burden workspace exceeds live/process budget; no precision/chunk fallback')


@contextmanager
def _ieee_flags():
    precision=torch.get_float32_matmul_precision();allow=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        yield
    finally:
        torch.set_float32_matmul_precision(precision);torch.backends.cuda.matmul.allow_tf32=allow


def ieee_burden_product(left,right,*,require_cuda=True,memory_limit_gib=20):
    if not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor):raise TypeError('Burden operands must be tensors')
    if left.dtype!=torch.float32 or right.dtype!=torch.float32:raise ValueError('Burden control requires float32 tensors; no dtype fallback')
    if left.layout!=torch.strided or right.layout!=torch.strided or left.ndim!=2 or right.ndim!=2:raise ValueError('Burden control requires dense matrices')
    if left.device!=right.device or (require_cuda and not left.is_cuda):raise ValueError('Burden matrices must be on the same CUDA device')
    estimate_workspace(left.shape,right.shape)
    if left.is_cuda and os.getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: native Burden requires IEEE FP32")
    if left.is_cuda:
        from . import tf32
        guard_workspace(left,right,limit_bytes=min(int(memory_limit_gib*2**30), tf32._memory_limit_bytes))
    # Source association_test has already checked finite score/covariance/metadata;
    # keep its validation priority rather than adding per-product host checks.
    # Frozen CLI sets allow_tf32=True after an outer scope starts. Protect
    # every selected product itself so that setter cannot weaken this control.
    with _LOCK, _ieee_flags():result=torch.mm(left,right)
    _METRICS['selected_fp32_mm_calls']+=1
    return result


def execution_metadata():
    return dict(_METRICS, selected_backend="IEEE FP32", fp64_dense_product_count=0, tf32_component_reconstruction=False)
