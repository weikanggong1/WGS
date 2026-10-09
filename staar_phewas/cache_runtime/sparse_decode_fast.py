"""Independent vectorized private CSR preparation; old decoder unchanged."""
from dataclasses import dataclass, field
import numpy as np
from .device_decode import STATE_MAPPING, _indices
from .sparse_decode import to_minor_block, dosage_numpy, workspace_bytes
from .sparse_codec_fast import _sample_dtype
_VALIDATION_SEAL = object()
_SAMPLE_BINDING_SEAL = object()


@dataclass(frozen=True)
class ValidatedSource:
    offsets: np.ndarray
    sample_index: np.ndarray
    state: np.ndarray
    ref_ac: np.ndarray
    called: np.ndarray
    n: int
    _seal: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SampleBinding:
    """One verified immutable cache-row axis and its inverse lookup.

    Rows preserve the requested order. The inverse lookup is absent only for
    a complete identity axis; subsets and complete permutations keep it. A
    private seal prevents an unchecked caller-supplied array being treated as
    a previously validated mapping.
    """
    rows: np.ndarray
    rowmap: np.ndarray | None
    n: int
    full: bool
    identity: bool
    _rows_address: int = field(default=0, repr=False, compare=False)
    _map_address: int = field(default=0, repr=False, compare=False)
    _seal: object = field(default=None, repr=False, compare=False)


def _immutable(a):
    # Bytes backing makes write-enable impossible; no unchecked mutable aliases.
    return np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)


def _count(metrics, key, value=1):
    if metrics is not None:
        metrics[key] = metrics.get(key, 0) + value


def bind_samples(samples, n, *, metrics=None):
    """Validate a sample axis once, copying it to immutable bytes backing."""
    if type(n) is not int or not 0 <= n <= 2**32:
        raise ValueError('invalid sample binding dimension')
    _count(metrics, 'decoder_sample_axis_validations')
    rows = _immutable(_indices(samples, n))
    full = len(rows) == n
    identity = samples is None or (full and np.array_equal(rows, np.arange(n, dtype=np.int64)))
    rowmap = None
    if not identity:
        inverse = np.full(n, -1, dtype=np.int64)
        inverse[rows] = np.arange(len(rows), dtype=np.int64)
        rowmap = _immutable(inverse)
        _count(metrics, 'decoder_sample_map_builds')
    if metrics is not None:
        metrics['decoder_sample_map_cache_bytes'] = 0 if rowmap is None else rowmap.nbytes
    return SampleBinding(rows, rowmap, n, full, identity,
        rows.__array_interface__['data'][0],
        0 if rowmap is None else rowmap.__array_interface__['data'][0],
        _SAMPLE_BINDING_SEAL)


def _bound_rows(binding, samples, n, metrics):
    # Frozen fields and immutable bytes protect values. Check ndarray metadata
    # too: shape/dtype can be reassigned even when the data is read-only.
    if (not isinstance(binding, SampleBinding) or binding._seal is not _SAMPLE_BINDING_SEAL
        or binding.n != n or binding.rows.ndim != 1 or binding.rows.dtype != np.int64
        or binding.rows.flags.writeable or not binding.rows.flags.c_contiguous
        or (len(binding.rows) and binding.rows.strides != (8,))
        or binding.rows.__array_interface__['data'][0] != binding._rows_address
        or len(binding.rows) > n
        or binding.full != (len(binding.rows) == n)
        or (binding.identity and (not binding.full or binding.rowmap is not None))
        or (not binding.identity and (not isinstance(binding.rowmap, np.ndarray)
            or binding.rowmap.shape != (n,) or binding.rowmap.dtype != np.int64
            or binding.rowmap.flags.writeable or not binding.rowmap.flags.c_contiguous
            or (n and binding.rowmap.strides != (8,))
            or binding.rowmap.__array_interface__['data'][0] != binding._map_address))):
        raise ValueError('validated immutable sample binding required')
    if samples is binding.rows or (samples is None and binding.identity):
        _count(metrics, 'decoder_sample_axis_cache_hits')
    else:
        # The fast path requires the canonical immutable object. A different
        # caller array is fully validated and compared, never trusted by shape.
        _count(metrics, 'decoder_sample_axis_validations')
        if not np.array_equal(_indices(samples, n), binding.rows):
            raise ValueError('sample axis differs from validated binding')
    return binding.rows


def validate_source(offsets, sample_index, state, ref_ac, called_alleles, n, *, mapping=STATE_MAPPING):
    if mapping!=STATE_MAPPING or type(n) is not int or not 0<=n<=2**32:
        raise ValueError('unsupported sparse cache encoding')
    off,idx,st,R,A=map(np.asarray,(offsets,sample_index,state,ref_ac,called_alleles))
    if (off.ndim!=1 or not len(off) or off.dtype!=np.uint32 or idx.ndim!=1 or idx.dtype!=_sample_dtype(n)
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
    if not count:return col,np.empty(0,dtype=source.sample_index.dtype),np.empty(0,dtype=np.uint8)
    cumulative=np.cumsum(length)-length
    take=np.repeat(start-cumulative,length)+np.arange(count,dtype=np.int64)
    return col,source.sample_index[take],source.state[take]


def prepare_validated(source, columns=None, samples=None, *, minimum_mac=None,
                      sample_binding=None, metrics=None):
    """Exact cohort summaries, with safe full-cache MAC bounds for subsets.

    A subset cannot exceed full-cache reference count or true alternate count.
    The original initial-MAC expression rounds the called count through a
    missing-rate calculation; an odd called count may add one allele. Its
    safe upper bound is therefore min(full_REF, full_called-full_REF+1), not
    the unrounded full-cache MAC. Final cohort filtering uses the unchanged
    original summary expression after exact integer subset counting.
    """
    from ..gds import _allele_frequency_summary
    if not isinstance(source,ValidatedSource) or source._seal is not _VALIDATION_SEAL:
        raise ValueError('validated immutable source required')
    m=len(source.offsets)-1
    cols=_indices(columns,m)
    if sample_binding is None:
        _count(metrics, 'decoder_sample_axis_validations')
        rows=_indices(samples,source.n)
        full=len(rows)==source.n
        identity=samples is None or (full and np.array_equal(rows,np.arange(source.n,dtype=np.int64)))
        rowmap=None
    else:
        rows=_bound_rows(sample_binding,samples,source.n,metrics)
        full,identity=sample_binding.full,sample_binding.identity
        rowmap=sample_binding.rowmap
    if minimum_mac is not None and (not np.isscalar(minimum_mac) or not np.isfinite(minimum_mac) or minimum_mac<0):
        raise ValueError('invalid minimum MAC')
    if not full and minimum_mac is not None:
        _count(metrics, 'safe_mac_prescreen_calls')
        _count(metrics, 'safe_mac_prescreen_input_variants', len(cols))
        upper=np.minimum(source.ref_ac[cols], source.called[cols]-source.ref_ac[cols]+1)
        candidate=upper>=minimum_mac
        _count(metrics, 'safe_mac_prescreen_rejected_variants', int((~candidate).sum()))
        _count(metrics, 'safe_mac_prescreen_candidate_variants', int(candidate.sum()))
        cols=cols[candidate]
    if full:
        summaries=_allele_frequency_summary(source.ref_ac[cols],source.called[cols],len(rows))
        eligible=np.arange(len(cols)) if minimum_mac is None else np.flatnonzero(summaries[2]>=minimum_mac)
        cols=cols[eligible];summaries=tuple(s[eligible] for s in summaries)
        ec,rawrows,es=_exceptions(source,cols)
        if identity:er=rawrows.astype(np.int64)
        else:
            if rowmap is None:
                rowmap=np.empty(source.n,dtype=np.int64);rowmap[rows]=np.arange(len(rows))
                _count(metrics, 'decoder_sample_map_builds')
            er=rowmap[rawrows]
    else:
        ec,rawrows,es=_exceptions(source,cols)
        if rowmap is None:
            rowmap=np.full(source.n,-1,dtype=np.int64);rowmap[rows]=np.arange(len(rows))
            _count(metrics, 'decoder_sample_map_builds')
        er=rowmap[rawrows];keep=er>=0;ec,er,es=ec[keep],er[keep],es[keep]
        # These are integer-valued totals <=2**33 per variant, exactly
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
