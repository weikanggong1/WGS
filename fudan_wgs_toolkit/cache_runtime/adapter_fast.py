"""Private CSR facade with immutable sample-map reuse and safe MAC prefilter.

Final cohort MAC, ordering, six-state decoding and materialization remain the
original paths; prefiltering uses a conservative integer upper bound only.
"""
from collections import OrderedDict
from pathlib import Path
import threading
import time
import numpy as np


class CachedGenotypeAdapter:
    def __init__(self, original_reader, container, *, device='cuda:0', compact_cache_bytes=64*2**20,
                 own_reader=True, prepared_cache=None, prefetch_depth=0,
                 prefetch_memory_bytes=256*2**20, prefetch_processes=0,
                 process_descriptor=None):
        from . import sparse_decode_fast
        self._fast=sparse_decode_fast
        self._reader=original_reader;self._container=container;self._device=device
        self._own_reader=own_reader;self._closed=False;self._lru=OrderedDict();self._lru_bytes=0
        self._compact_prepare_lock = threading.Lock()
        if type(prefetch_depth) is not int or prefetch_depth < 0:
            raise ValueError('prefetch_depth must be a nonnegative integer')
        if type(prefetch_memory_bytes) is not int or prefetch_memory_bytes < 1:
            raise ValueError('prefetch_memory_bytes must be a positive integer')
        if type(prefetch_processes) is not int or prefetch_processes < 0:
            raise ValueError('prefetch_processes must be a nonnegative integer')
        self._prepared_cache = prepared_cache
        self._prefetch_depth, self._prefetch_memory_bytes = prefetch_depth, prefetch_memory_bytes
        self._prefetch_processes = prefetch_processes
        self._process_descriptor = process_descriptor
        self._prepare_pool = None
        if type(compact_cache_bytes) is not int or compact_cache_bytes<1:raise ValueError('invalid compact cache budget')
        self._capacity=compact_cache_bytes
        self._sample_binding=None  # One axis only; immutable bytes-backed snapshots.
        self._decoder_binding=None
        if not getattr(container,'complete',False):raise RuntimeError('complete verified cache container required')
        self._index=container.index
        starts=np.asarray(self._index['start'],dtype=np.int64);sizes=np.asarray(self._index['m'],dtype=np.int64)
        if not len(starts) or starts[0]!=0 or np.any(sizes<=0) or np.any(starts[1:]!=starts[:-1]+sizes[:-1]):
            raise RuntimeError('cache variant coverage is not contiguous and complete')
        self._starts=starts;self._sizes=sizes
        if int(starts[-1]+sizes[-1])!=original_reader.n_variants:raise RuntimeError('cache/source variant coverage differs')
        self._samples=np.asarray(container.samples)
        if self._samples.ndim!=1 or self._samples.dtype.kind not in 'iu' or np.any(self._samples>=original_reader.n_samples) or np.any(self._samples<0) or len(np.unique(self._samples))!=len(self._samples):
            raise RuntimeError('invalid cache/source sample binding')
        self._sample_sort=np.argsort(self._samples);self._sorted_samples=self._samples[self._sample_sort]
        self._metrics=dict(minor_block_calls=0,frame_loads=0,frame_cache_hits=0,frame_cache_evictions=0,
            cache_read_validate_seconds=0.,compact_prepare_seconds=0.,materialize_seconds=0.,
            minor_block_wall_seconds=0.,cpu_sparse_calls=0,cuda_resident_calls=0,
            requested_variants=0,returned_variants=0,cache_bytes_highwater=0,
            sample_bind_calls=0,sample_bind_cache_hits=0,sample_bind_validations=0,
            sample_bind_seconds=0.,sample_validate_seconds=0.,sample_bind_cache_bytes=0,
            decoder_sample_axis_validations=0,decoder_sample_axis_cache_hits=0,
            decoder_sample_map_builds=0,decoder_sample_map_cache_bytes=0,
            safe_mac_prescreen_calls=0,safe_mac_prescreen_input_variants=0,
            safe_mac_prescreen_rejected_variants=0,safe_mac_prescreen_candidate_variants=0)

    def __getattr__(self,name):
        return getattr(self._reader,name)

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def close(self):
        if self._closed:return
        if self._prepare_pool is not None:
            self._prepare_pool.close()
        self._closed=True;self._lru.clear();self._lru_bytes=0
        self._sample_binding=None;self._metrics['sample_bind_cache_bytes']=0
        self._decoder_binding=None;self._metrics['decoder_sample_map_cache_bytes']=0
        if self._prepared_cache is not None:
            self._prepared_cache.close()
        try:
            if hasattr(self._container,'close'):self._container.close()
        finally:
            if self._own_reader:self._reader.close()

    @property
    def reader_metadata(self):
        result=dict(self._reader.reader_metadata)
        result['analysis_cache']=dict(self._metrics,backend='complete-reference-six-state-CSR',
            genotype_sdk_fallback_count=0,compact_cache_limit_bytes=self._capacity,
            timer_contract='minor_block wall is consumer wait plus materialization; with prefetch, producer '
                           'read_validate/compact_prepare/cache IO overlap consumer work and are not additive; '
                           'sample_bind is another minor_block wall component; sample_validate is nested in sample_bind; '
                           'metadata timing is retained; CUDA materialize host time may enqueue work')
        result['analysis_cache']['prefetch_processes_requested'] = self._prefetch_processes
        if self._prepare_pool is not None:
            result['analysis_cache']['process_pool'] = self._prepare_pool.metadata()
        return result

    def read_genotype(self,*args,**kwargs):raise RuntimeError('cache stores collapsed REF states, not raw allele identities; no SDK genotype fallback')
    def read_ref_dosage(self,*args,**kwargs):raise RuntimeError('use cache minor_block; no SDK genotype fallback')

    def _frame(self,j):
        if j in self._lru:
            self._metrics['frame_cache_hits']+=1;self._lru.move_to_end(j);return self._lru[j][0]
        start=time.perf_counter()
        f=self._container.read_frame(int(j))  # codec validates frame/header/count hashes
        if f['start']!=self._starts[j] or f['m']!=self._sizes[j] or f['n']!=len(self._samples):
            raise RuntimeError('cache frame/index/sample geometry mismatch')
        source=self._fast.validate_source(f['offsets'],f['sample_index'],f['state'],f['reference_alleles'],f['called_alleles'],int(f['n']))
        self._metrics['cache_read_validate_seconds']+=time.perf_counter()-start
        self._metrics['frame_loads']+=1
        size=sum(a.nbytes for a in (source.offsets,source.sample_index,source.state,source.ref_ac,source.called))
        if size<=self._capacity:
            while self._lru and self._lru_bytes+size>self._capacity:
                _,(_,oldsize)=self._lru.popitem(last=False);self._lru_bytes-=oldsize;self._metrics['frame_cache_evictions']+=1
            self._lru[j]=(source,size);self._lru_bytes+=size
            self._metrics['cache_bytes_highwater']=max(self._metrics['cache_bytes_highwater'],self._lru_bytes)
        return source

    @staticmethod
    def _immutable_axis(values):
        # Unlike writeable=False on an owned array, bytes backing cannot be
        # write-enabled by a caller and has no alias to their mutable input.
        return np.frombuffer(values.tobytes(order='C'),dtype=values.dtype)

    def _bind_samples(self,samples):
        from ..genotype import _indices
        start=time.perf_counter();self._metrics['sample_bind_calls']+=1
        try:
            candidate=np.asarray(samples)
            binding=self._sample_binding
            # Require the original integer dtype, complete values and geometry.
            # Object identity/shape alone never establishes a validated binding.
            if (binding is not None and candidate.ndim==1 and candidate.dtype.kind in 'iu'
                and candidate.dtype==binding[0].dtype and len(candidate)==len(binding[0])
                and self.n_samples==binding[3] and np.array_equal(candidate,binding[0])):
                self._metrics['sample_bind_cache_hits']+=1
                return binding[1],binding[2]
            validate_start=time.perf_counter()
            try:
                ss=_indices(samples,self.n_samples,'union_sample_indices')
                positions=np.searchsorted(self._sorted_samples,ss)
                if np.any(positions>=len(self._samples)) or np.any(self._sorted_samples[np.minimum(positions,len(self._samples)-1)]!=ss):
                    raise RuntimeError('requested samples outside complete cache sample binding')
                cache_rows=self._sample_sort[positions]
            finally:
                self._metrics['sample_bind_validations']+=1
                self._metrics['sample_validate_seconds']+=time.perf_counter()-validate_start
            snapshot=self._immutable_axis(candidate)
            validated=self._immutable_axis(ss)
            decoder_binding=self._fast.bind_samples(cache_rows,len(self._samples),metrics=self._metrics)
            rows=decoder_binding.rows
            full_identity=decoder_binding.identity
            self._sample_binding=(snapshot,validated,rows,self.n_samples,full_identity)
            self._decoder_binding=decoder_binding
            self._metrics['sample_bind_cache_bytes']=snapshot.nbytes+validated.nbytes+rows.nbytes
            return validated,rows
        finally:
            self._metrics['sample_bind_seconds']+=time.perf_counter()-start

    def _prepare(self,variants,samples,minimum_mac):
        from ..genotype import _indices
        vv=_indices(variants,self.n_variants,'variant_indices')
        ss,cache_rows=self._bind_samples(samples)
        if self._prepared_cache is not None:
            cached = self._prepared_cache.load(vv, ss, minimum_mac)
            if cached is not None:
                return cached, ss, vv
        # Only this immutable canonical axis uses the sealed decoder binding.
        # Full permutations/subsets retain exact requested sample order while
        # reusing their already validated inverse map across cache frames.
        binding=self._sample_binding
        decoder_samples=None if binding[2] is cache_rows and binding[4] else cache_rows
        frames=np.searchsorted(self._starts,vv,side='right')-1
        parts=[];request_positions=[]
        read_before=self._metrics['cache_read_validate_seconds'];start=time.perf_counter()
        for j in np.unique(frames):
            pos=np.flatnonzero(frames==j);local=vv[pos]-self._starts[j]
            part=self._fast.prepare_validated(self._frame(int(j)),local,decoder_samples,
                minimum_mac=minimum_mac,sample_binding=self._decoder_binding,metrics=self._metrics)
            # local columns after MAC filtering map back to request order.
            localmap=np.full(int(self._sizes[j]),-1,dtype=np.int64);localmap[local]=pos
            request_positions.append(localmap[part['columns']]);parts.append(part)
        selected=np.concatenate(request_positions) if parts else np.empty(0,dtype=np.int64)
        order=np.argsort(selected);inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
        ec=[];er=[];es=[];base=0
        for part in parts:
            ec.append(part['exception_col']+base);er.append(part['exception_row']);es.append(part['exception_state']);base+=len(part['columns'])
        col=np.concatenate(ec) if ec else np.empty(0,dtype=np.int64)
        row=np.concatenate(er) if er else np.empty(0,dtype=np.int64)
        state=np.concatenate(es) if es else np.empty(0,dtype=np.uint8)
        summaries=tuple(np.concatenate([p['summaries'][i] for p in parts])[order] if parts else np.empty(0,dtype=np.int64 if i==4 else np.float64) for i in range(5))
        prepared=dict(cache_variant_count=len(vv),cache_sample_count=len(ss),columns=selected[order],samples=np.arange(len(ss),dtype=np.int64),
            exception_col=inverse[col],exception_row=row,exception_state=state,summaries=summaries,full_union_summaries=len(cache_rows)==len(self._samples))
        self._add_compact_prepare_seconds(time.perf_counter()-start-(self._metrics['cache_read_validate_seconds']-read_before))
        if self._prepared_cache is not None:
            self._prepared_cache.store(vv, ss, minimum_mac, prepared)
        return prepared,ss,vv

    def _add_compact_prepare_seconds(self, seconds):
        """Serialize producer preparation and consumer Single packing clocks."""
        with self._compact_prepare_lock:
            self._metrics['compact_prepare_seconds'] += seconds

    def _prepared_requests(self, requests, samples, minimum_mac):
        """Keep CUDA materialization on the consumer thread."""
        if self._prefetch_depth:
            from .prepared_prefetch import iter_prepared
            if (self._prefetch_processes and self._prepared_cache is not None
                    and self._process_descriptor is not None):
                from .prepared_process_pool import PreparedProcessPool
                ss, _ = self._bind_samples(samples)
                if self._prepare_pool is not None and not np.array_equal(ss, self._prepare_pool.samples):
                    self._prepare_pool.close()
                    self._prepare_pool = None
                if self._prepare_pool is None:
                    self._prepare_pool = PreparedProcessPool(self, ss, self._prefetch_processes,
                        max(self._prefetch_memory_bytes, self._prefetch_processes*2**30))
                results = self._prepare_pool.iter_prepared(requests, minimum_mac)
                yield from iter_prepared(self, (), ss, minimum_mac,
                    max_items=self._prefetch_depth, max_bytes=self._prefetch_memory_bytes,
                    _prepared_results=results)
            else:
                yield from iter_prepared(self, requests, samples, minimum_mac,
                    max_items=self._prefetch_depth, max_bytes=self._prefetch_memory_bytes)
        else:
            for request in requests:
                yield self._prepare(request, samples, minimum_mac)

    @staticmethod
    def _sparse(prepared,samples,variants):
        from ..genotype import SparseMinorBlock
        m,n=len(prepared['columns']),len(samples);af,miss,mac,R,A=prepared['summaries']
        ec,er,es=(prepared[k] for k in ('exception_col','exception_row','exception_state'))
        defaults=np.flatnonzero(~(af>=.5))
        count=n*len(defaults)+len(es)
        if count*64>20*2**30:raise MemoryError('CPU sparse cache materialization exceeds conservative 20 GiB budget')
        # Only nonzero baseline columns expand; rare-ALT defaults remain implicit0.
        baseline=(np.arange(n,dtype=np.int64)[:,None]*m+defaults[None,:]).reshape(-1)
        exceptions=er*m+ec
        baseline=baseline[~np.isin(baseline,exceptions)]
        d=np.where(es<3,2-es,3).astype(np.uint8)
        d=np.where((d!=3)&(af[ec]>=.5),2-d,d).astype(np.uint8)
        keep=d!=0
        locations=np.concatenate((baseline,exceptions[keep]));values=np.concatenate((np.full(len(baseline),2.,dtype=np.float64),np.where(d[keep]==3,np.nan,d[keep]).astype(np.float64)))
        order=np.argsort(locations);locations=locations[order];values=values[order]
        row=locations//m if m else np.empty(0,dtype=np.int64);col=locations%m if m else np.empty(0,dtype=np.int64)
        return SparseMinorBlock(row,col,values,samples.copy(),variants[prepared['columns']].copy(),af,mac,miss,R,A)

    def minor_block(self,variant_indices,union_sample_indices,*,device=None,minimum_mac=None,resident=False):
        if self._closed:raise RuntimeError('cache reader is closed')
        start=time.perf_counter();prepared,samples,variants=self._prepare(variant_indices,union_sample_indices,minimum_mac)
        materialize_start=time.perf_counter()
        if resident:
            result=self._fast.to_minor_block(prepared,samples,variants,device=device or self._device)
            self._metrics['cuda_resident_calls']+=1
        else:
            result=self._sparse(prepared,samples,variants);self._metrics['cpu_sparse_calls']+=1
        self._metrics['materialize_seconds']+=time.perf_counter()-materialize_start
        self._metrics['minor_block_calls']+=1;self._metrics['requested_variants']+=len(variants);self._metrics['returned_variants']+=result.shape[1]
        self._metrics['minor_block_wall_seconds']+=time.perf_counter()-start
        return result

    def iter_minor_blocks(self,variant_indices,union_sample_indices,block_size=256,*,device=None,minimum_mac=None,resident=False):
        from ..genotype import _indices
        vv=_indices(variant_indices,self.n_variants,'variant_indices');ss=_indices(union_sample_indices,self.n_samples,'union_sample_indices')
        if type(block_size) is not int or block_size<1:raise ValueError('block_size must be positive integer')
        requests=(vv[start:start+block_size] for start in range(0,len(vv),block_size))
        iterator=self._prepared_requests(requests,ss,minimum_mac)
        try:
            while True:
                waiting=time.perf_counter()
                item=next(iterator,None)
                self._metrics['minor_block_wall_seconds']+=time.perf_counter()-waiting
                if item is None:
                    break
                prepared,bound_samples,variants=item
                start=time.perf_counter()
                if resident:
                    result=self._fast.to_minor_block(prepared,bound_samples,variants,device=device or self._device)
                    self._metrics['cuda_resident_calls']+=1
                else:
                    result=self._sparse(prepared,bound_samples,variants)
                    self._metrics['cpu_sparse_calls']+=1
                elapsed=time.perf_counter()-start
                self._metrics['materialize_seconds']+=elapsed
                self._metrics['minor_block_wall_seconds']+=elapsed
                self._metrics['minor_block_calls']+=1
                self._metrics['requested_variants']+=len(variants)
                self._metrics['returned_variants']+=result.shape[1]
                yield result
        finally:
            iterator.close()

    def iter_effective_minor_blocks(self,*args,**kwargs):
        from .single_batches import iter_effective_minor_blocks
        return iter_effective_minor_blocks(self,*args,**kwargs)
