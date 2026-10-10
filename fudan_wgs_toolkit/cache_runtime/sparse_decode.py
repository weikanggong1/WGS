"""Private default-REF-homo sparse cache -> direct uint8 minor dosage.

No full states or full int64 counts on CUDA. No reader fallback. Six-state
encoding is ref-homo-first-six-v1; nonzero exceptions are CSR by variant.
"""
import numpy as np
from .device_decode import STATE_MAPPING, check_budget
from .sparse_codec_fast import _sample_dtype

_KERNELS = None


def prepare(offsets, sample_index, state, ref_ac, called_alleles, n,
            columns=None, samples=None, *, minimum_mac=None, mapping=STATE_MAPPING):
    """CPU compact preparation; summaries recomputed for a sample subset.

    Contract: offsets uint32[m+1], sample_index uint16[E] for n<65536 and
    uint32[E] for larger axes, matching the verified CSR codec; state
    uint8[E] in1..5; strictly increasing sample indices within each variant.
    ref_ac/called_alleles int64[m] refer to the complete cache sample union.
    These must come from the codec's verified header/payload binding.
    """
    from ..genotype import _allele_frequency_summary
    from .device_decode import _indices
    if mapping != STATE_MAPPING or type(n) is not int or not 0 <= n <= 2**32:
        raise ValueError('unsupported sparse cache encoding/sample count')
    off, idx, st, R, A = map(np.asarray,(offsets,sample_index,state,ref_ac,called_alleles))
    if (off.ndim!=1 or len(off)<1 or off.dtype!=np.uint32 or idx.ndim!=1 or idx.dtype!=_sample_dtype(n)
        or st.ndim!=1 or st.dtype!=np.uint8 or R.dtype!=np.int64 or A.dtype!=np.int64):
        raise ValueError('incorrect CSR dtype/shape')
    m=len(off)-1
    if R.shape!=(m,) or A.shape!=(m,) or len(idx)!=len(st) or off[0]!=0 or off[-1]!=len(st) or np.any(off[1:]<off[:-1]):
        raise ValueError('invalid CSR geometry')
    if np.any(idx>=n) or np.any(st<1) or np.any(st>5) or np.any(R<0) or np.any(A<R) or np.any(A>2*n):
        raise ValueError('invalid exception/summary range')
    # Only cross-variant boundaries may have equal/decreasing sample indices.
    if len(idx)>1:
        bad=np.flatnonzero(idx[1:]<=idx[:-1])+1
        boundaries=off[1:-1]
        if np.any(~np.isin(bad,boundaries)):
            raise ValueError('duplicate/unsorted within-variant exceptions')
    cols, rows=_indices(columns,m),_indices(samples,n)
    if minimum_mac is not None and (not np.isscalar(minimum_mac) or not np.isfinite(minimum_mac) or minimum_mac<0):
        raise ValueError('invalid minimum MAC')
    full=len(rows)==n  # unique in-range rows implies complete union, any order
    rowmap=np.full(n,-1,dtype=np.int64);rowmap[rows]=np.arange(len(rows))
    outcol=[];outrow=[];outstate=[]
    r=R[cols].copy() if full else np.full(len(cols),2*len(rows),dtype=np.int64)
    a=A[cols].copy() if full else r.copy()
    dref=np.array([0,1,2,2,1,2],dtype=np.int64)
    dcalled=np.array([0,0,0,2,1,1],dtype=np.int64)
    for j,c in enumerate(cols):
        lo,hi=int(off[c]),int(off[c+1]);mapped=rowmap[idx[lo:hi]];keep=mapped>=0
        ss=st[lo:hi][keep];rr=mapped[keep]
        if not full:
            r[j]-=dref[ss].sum(dtype=np.int64);a[j]-=dcalled[ss].sum(dtype=np.int64)
        outcol.append(np.full(len(ss),j,dtype=np.int64));outrow.append(rr);outstate.append(ss)
    summaries=_allele_frequency_summary(r,a,len(rows))
    eligible=np.arange(len(cols)) if minimum_mac is None else np.flatnonzero(summaries[2]>=minimum_mac)
    remap=np.full(len(cols),-1,dtype=np.int64);remap[eligible]=np.arange(len(eligible))
    ec=np.concatenate(outcol) if outcol else np.empty(0,dtype=np.int64)
    er=np.concatenate(outrow) if outrow else np.empty(0,dtype=np.int64)
    es=np.concatenate(outstate) if outstate else np.empty(0,dtype=np.uint8)
    keep=remap[ec]>=0
    return dict(cache_variant_count=m, cache_sample_count=n,
                columns=cols[eligible], samples=rows, exception_col=remap[ec[keep]],
                exception_row=er[keep], exception_state=es[keep],
                summaries=tuple(s[eligible] for s in summaries), full_union_summaries=full)


def workspace_bytes(prepared):
    m,n=len(prepared['columns']),len(prepared['samples']);e=len(prepared['exception_state'])
    return 2*m*n + 40*e + 256*m + 64*2**20


def _get_kernels():
    global _KERNELS, triton, tl
    if _KERNELS is not None:return _KERNELS
    import triton
    import triton.language as tl
    @triton.jit
    def fill(O,F,M,N,BLOCK:tl.constexpr):
        x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=x<M*N
        flip=tl.load(F+x%M,ok,other=0)
        tl.store(O+x,tl.where(flip,0,2),ok)
    @triton.jit
    def scatter(O,F,C,R,S,M,E,BLOCK:tl.constexpr):
        x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=x<E
        c=tl.load(C+x,ok,other=0);r=tl.load(R+x,ok,other=0);s=tl.load(S+x,ok,other=3)
        flip=tl.load(F+c,ok,other=0)
        d=tl.where(s<3,2-s,3)
        d=tl.where((d!=3)&flip,2-d,d)
        tl.store(O+r*M+c,d,ok)
    _KERNELS=fill,scatter
    return _KERNELS


def to_minor_block(prepared, sample_indices, variant_indices, *, device='cuda:0'):
    """Materialize only MAC-eligible columns. No full sparse-state reduction."""
    import torch
    from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
    source_samples=np.asarray(sample_indices);source_variants=np.asarray(variant_indices)
    if source_samples.ndim!=1 or source_variants.ndim!=1 or source_samples.dtype.kind not in 'iu' or source_variants.dtype.kind not in 'iu':
        raise ValueError('require original integer source index binding')
    if len(source_samples)!=prepared['cache_sample_count'] or len(source_variants)!=prepared['cache_variant_count']:
        raise ValueError('source bindings must cover complete cache axes')
    if len(np.unique(source_samples))!=len(source_samples) or len(np.unique(source_variants))!=len(source_variants) or np.any(source_samples<0) or np.any(source_variants<0):
        raise ValueError('invalid original source index binding')
    rows,cols=prepared['samples'],prepared['columns']
    if np.any(rows>=len(source_samples)) or np.any(cols>=len(source_variants)):
        raise ValueError('source binding coverage mismatch')
    dev=torch.device(device)
    if dev.type!='cuda':raise ValueError('CUDA device required')
    free,_=torch.cuda.mem_get_info(dev)
    check_budget(workspace_bytes(prepared),torch.cuda.memory_allocated(dev),free)
    m,n=len(cols),len(rows);af,miss,mac,R,A=prepared['summaries']
    out=torch.empty((n,m),dtype=torch.uint8,device=dev)
    if out.numel():
        fill,scatter=_get_kernels()
        flip=torch.as_tensor(af>=.5,device=dev)
        with torch.cuda.device(dev):
            fill[(triton.cdiv(out.numel(),256),)](out,flip,m,n,BLOCK=256)
            e=len(prepared['exception_state'])
            if e:
                c=torch.as_tensor(prepared['exception_col'],device=dev)
                r=torch.as_tensor(prepared['exception_row'],device=dev)
                s=torch.as_tensor(prepared['exception_state'],device=dev)
                scatter[(triton.cdiv(e,256),)](out,flip,c,r,s,m,e,BLOCK=256)
    return DeviceMinorBlock(out,source_samples[rows].copy(),source_variants[cols].copy(),af,mac,miss,R,A)


def dosage_numpy(prepared):
    """CPU differential oracle; never a CUDA fallback."""
    af=prepared['summaries'][0];m,n=len(af),len(prepared['samples'])
    out=np.broadcast_to(np.where(af>=.5,0,2).astype(np.uint8),(n,m)).copy()
    c,r,s=(prepared[k] for k in ('exception_col','exception_row','exception_state'))
    d=np.where(s<3,2-s,3).astype(np.uint8)
    d=np.where((d!=3)&(af[c]>=.5),2-d,d).astype(np.uint8)
    out[r,c]=d
    return out
