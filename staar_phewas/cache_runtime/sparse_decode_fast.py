"""Independent vectorized private CSR preparation; old decoder unchanged."""
from dataclasses import dataclass, field
import numpy as np
from .device_decode import STATE_MAPPING, _indices
from .sparse_decode import to_minor_block, dosage_numpy, workspace_bytes
_VALIDATION_SEAL = object()


@dataclass(frozen=True)
class ValidatedSource:
    offsets: np.ndarray
    sample_index: np.ndarray
    state: np.ndarray
    ref_ac: np.ndarray
    called: np.ndarray
    n: int
    _seal: object = field(default=None, repr=False, compare=False)


def _immutable(a):
    # Bytes backing makes write-enable impossible; no unchecked mutable aliases.
    return np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)


def validate_source(offsets, sample_index, state, ref_ac, called_alleles, n, *, mapping=STATE_MAPPING):
    if mapping!=STATE_MAPPING or type(n) is not int or not 0<=n<65536:
        raise ValueError('unsupported sparse cache encoding')
    off,idx,st,R,A=map(np.asarray,(offsets,sample_index,state,ref_ac,called_alleles))
    if (off.ndim!=1 or not len(off) or off.dtype!=np.uint32 or idx.ndim!=1 or idx.dtype!=np.uint16
        or st.ndim!=1 or st.dtype!=np.uint8 or R.dtype!=np.int64 or A.dtype!=np.int64):
        raise ValueError('incorrect CSR dtype/shape')
    m=len(off)-1
    if R.shape!=(m,) or A.shape!=(m,) or len(idx)!=len(st) or off[0]!=0 or off[-1]!=len(st) or np.any(off[1:]<off[:-1]):
        raise ValueError('invalid CSR geometry')
    if np.any(idx>=n) or np.any(st<1) or np.any(st>5) or np.any(R<0) or np.any(A<R) or np.any(A>2*n):
        raise ValueError('invalid exception/summary range')
    if len(idx)>1:
        bad=idx[1:]<=idx[:-1]
        boundary=off[1:-1]
        boundary=boundary[(boundary>0)&(boundary<len(idx))]
        bad[boundary.astype(np.int64)-1]=False
        if np.any(bad):raise ValueError('duplicate/unsorted within-variant exceptions')
    # Immutable token remains tied to this verified payload; header/hash/source
    # binding remains codec/driver responsibility, as in the original adapter.
    return ValidatedSource(*[_immutable(a) for a in (off,idx,st,R,A)],n,_VALIDATION_SEAL)


def _exceptions(source, cols):
    start=source.offsets[cols].astype(np.int64)
    length=source.offsets[cols+1].astype(np.int64)-start
    count=int(length.sum())
    col=np.repeat(np.arange(len(cols),dtype=np.int64),length)
    if not count:return col,np.empty(0,dtype=np.uint16),np.empty(0,dtype=np.uint8)
    cumulative=np.cumsum(length)-length
    take=np.repeat(start-cumulative,length)+np.arange(count,dtype=np.int64)
    return col,source.sample_index[take],source.state[take]


def prepare_validated(source, columns=None, samples=None, *, minimum_mac=None):
    """MAC first for full union; sparse vectorized counts for sample subsets."""
    from ..gds import _allele_frequency_summary
    if not isinstance(source,ValidatedSource) or source._seal is not _VALIDATION_SEAL:
        raise ValueError('validated immutable source required')
    m=len(source.offsets)-1
    cols,rows=_indices(columns,m),_indices(samples,source.n)
    if minimum_mac is not None and (not np.isscalar(minimum_mac) or not np.isfinite(minimum_mac) or minimum_mac<0):
        raise ValueError('invalid minimum MAC')
    full=len(rows)==source.n
    identity=samples is None or np.array_equal(rows,np.arange(source.n,dtype=np.int64))
    if full:
        summaries=_allele_frequency_summary(source.ref_ac[cols],source.called[cols],len(rows))
        eligible=np.arange(len(cols)) if minimum_mac is None else np.flatnonzero(summaries[2]>=minimum_mac)
        cols=cols[eligible];summaries=tuple(s[eligible] for s in summaries)
        ec,rawrows,es=_exceptions(source,cols)
        if identity:er=rawrows.astype(np.int64)
        else:
            rowmap=np.empty(source.n,dtype=np.int64);rowmap[rows]=np.arange(len(rows))
            er=rowmap[rawrows]
    else:
        ec,rawrows,es=_exceptions(source,cols)
        rowmap=np.full(source.n,-1,dtype=np.int64);rowmap[rows]=np.arange(len(rows))
        er=rowmap[rawrows];keep=er>=0;ec,er,es=ec[keep],er[keep],es[keep]
        # These are integer-valued totals <131072 per variant, exactly
        # represented by bincount's float64 weights, then restored to int64.
        dref=np.array([0,1,2,2,1,2],dtype=np.int64)
        dcalled=np.array([0,0,0,2,1,1],dtype=np.int64)
        R=2*len(rows)-np.bincount(ec,weights=dref[es],minlength=len(cols)).astype(np.int64)
        A=2*len(rows)-np.bincount(ec,weights=dcalled[es],minlength=len(cols)).astype(np.int64)
        summaries=_allele_frequency_summary(R,A,len(rows))
        eligible=np.arange(len(cols)) if minimum_mac is None else np.flatnonzero(summaries[2]>=minimum_mac)
        remap=np.full(len(cols),-1,dtype=np.int64);remap[eligible]=np.arange(len(eligible))
        keep=remap[ec]>=0;ec,er,es=remap[ec[keep]],er[keep],es[keep]
        cols=cols[eligible];summaries=tuple(s[eligible] for s in summaries)
    return dict(cache_variant_count=m,cache_sample_count=source.n,columns=cols,samples=rows,
                exception_col=ec,exception_row=er,exception_state=es,
                summaries=summaries,full_union_summaries=full)


def prepare(offsets,sample_index,state,ref_ac,called_alleles,n,columns=None,samples=None,*,minimum_mac=None,mapping=STATE_MAPPING):
    """Compatibility entry: validates once per invocation; no implicit cache."""
    return prepare_validated(validate_source(offsets,sample_index,state,ref_ac,called_alleles,n,mapping=mapping),
                             columns,samples,minimum_mac=minimum_mac)
