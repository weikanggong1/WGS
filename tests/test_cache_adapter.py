from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from fudan_wgs_toolkit.cache_runtime.adapter_fast import CachedGenotypeAdapter
from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast, sparse_decode, store

def fixture(raw):
    offsets,indices,states=sparse_codec_fast.compact(raw)
    counts=sparse_codec_fast.integer_counts(offsets,indices,states,*raw.shape)
    return offsets,indices,states,counts['reference_alleles'],counts['called_alleles']

class MetadataReader:
    n_variants=9;n_samples=12
    reader_metadata={'native_reads':{'calls':0}}
    def read_field(self,path,variants=None):return path,variants
    def sample_ids(self):return np.arange(12).astype(str)
    def close(self):self.closed=True


class Container:
    complete=True
    def __init__(self):
        self.samples=np.array([9,2,6,1,10,3],dtype=np.int64)
        self.index=np.array([(0,4),(4,4),(8,1)],dtype=[('start','i8'),('m','i8')])
        self.raw=np.array([[0,1,2,3,4,5],[0,0,0,0,0,0],[3,3,3,3,3,3],[2,2,2,1,5,4],
                           [4,5,0,1,2,3],[0,1,0,1,0,1],[5,5,5,5,5,5],[2,2,0,0,1,1],[1,4,5,3,2,0]],dtype=np.uint8)
        self.calls=[]
    def read_frame(self,j):
        self.calls.append(j);start,m=self.index[j];raw=self.raw[start:start+m];off,idx,st,R,A=fixture(raw)
        return dict(start=int(start),m=int(m),n=6,offsets=off,sample_index=idx,state=st,reference_alleles=R,called_alleles=A)


def expected(container,variants,samples,minimum_mac):
    from fudan_wgs_toolkit.cache_runtime.sparse_decode import prepare,dosage_numpy
    from fudan_wgs_toolkit.genotype import SparseMinorBlock
    rows=np.array([np.flatnonzero(container.samples==s)[0] for s in samples],dtype=np.int64)
    p=prepare(*fixture(container.raw),6,np.asarray(variants,dtype=np.int64),rows,minimum_mac=minimum_mac)
    d=dosage_numpy(p).astype(np.float64);d[d==3]=np.nan
    r,c=np.nonzero((d!=0)|np.isnan(d));af,miss,mac,R,A=p['summaries']
    return SparseMinorBlock(r,c,d[r,c],np.asarray(samples),p['columns'],af,mac,miss,R,A)


class Contracts(unittest.TestCase):
    def setUp(self):self.reader=MetadataReader();self.container=Container();self.a=CachedGenotypeAdapter(self.reader,self.container)
    def test_crossframe_reverse_missing_half_subset_and_filter(self):
        for vv in [np.arange(8,-1,-1),np.array([8,0,5,3]),np.empty(0,dtype=np.int64)]:
            for ss in [self.container.samples,self.container.samples[::-1],np.array([3,9,1]),np.empty(0,dtype=np.int64)]:
                for mac in [None,2,20]:
                    actual=self.a.minor_block(vv,ss,minimum_mac=mac);control=expected(self.container,vv,ss,mac)
                    for k in ('row','col','value','sample_indices','variant_indices','union_ref_af','union_initial_mac','union_missing_rate','union_ref_ac','union_called_alleles'):
                        np.testing.assert_array_equal(getattr(actual,k),getattr(control,k))
                    for imp in ('mean','minor'):
                        left=actual.trait_dense(np.arange(len(ss)),imp,dtype=np.float32);right=control.trait_dense(np.arange(len(ss)),imp,dtype=np.float32)
                        for x,y in zip(left,right):np.testing.assert_array_equal(x,y)
    def test_original_metadata_and_cache_hit_boundaries(self):
        np.testing.assert_array_equal(self.a.sample_ids(),self.reader.sample_ids())
        self.assertEqual(self.a.read_field('position'),('position',None))
        self.a.minor_block(np.array([0,8]),self.container.samples)
        self.a.minor_block(np.array([8,0]),self.container.samples)
        self.assertEqual(self.container.calls,[0,2])
        self.assertEqual(self.a.reader_metadata['analysis_cache']['frame_cache_hits'],2)
        self.assertEqual(self.a.reader_metadata['analysis_cache']['genotype_sdk_fallback_count'],0)
    def test_lru_closed_duplicates_unsupported_and_incomplete(self):
        a=CachedGenotypeAdapter(MetadataReader(),Container(),compact_cache_bytes=120)
        a.minor_block(np.array([0,8]),self.container.samples)
        self.assertLessEqual(a._lru_bytes,120)
        with self.assertRaises(ValueError):self.a.minor_block(np.array([0,0]),self.container.samples)
        with self.assertRaises(RuntimeError):self.a.minor_block(np.array([0]),np.array([11]))
        with self.assertRaises(RuntimeError):self.a.read_genotype([],[])
        self.a.close()
        with self.assertRaises(RuntimeError):self.a.minor_block(np.array([0]),self.container.samples)
        c=Container();c.complete=False
        with self.assertRaises(RuntimeError):CachedGenotypeAdapter(MetadataReader(),c)
    def test_real_container_binding_checksum_fail_closed(self):
        import tempfile
        raw=self.container.raw
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache';binding={'source':'anonymous-test-binding'}
            writer=store.Writer(path,binding,self.container.samples,9,source_bytes=10**8)
            off,idx,st,R,A=fixture(raw)
            counts=sparse_codec_fast.integer_counts(off,idx,st,9,6)
            writer.append(raw,counts);writer.finish()
            c=store.open_container(path,expected_source_binding=binding,expected_samples=self.container.samples)
            a=CachedGenotypeAdapter(self.reader,c)
            a.minor_block(np.arange(9),self.container.samples)
            with self.assertRaises(ValueError):store.open_container(path,expected_source_binding={'source':'wrong'})
            # Corrupt compressed payload on a fresh reader, never reuse an LRU.
            with (path/'data.bin').open('r+b') as f:
                byte=f.read(1);f.seek(0);f.write(bytes([byte[0]^1]))
            with self.assertRaises(ValueError):CachedGenotypeAdapter(self.reader,c).minor_block(np.array([0]),self.container.samples)
    def test_sparse_host_budget_failure(self):
        p=dict(columns=np.array([0]),summaries=(np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]),np.array([0])),
               exception_col=np.empty(0,dtype=np.int64),exception_row=np.empty(0,dtype=np.int64),exception_state=np.empty(0,dtype=np.uint8))
        with self.assertRaises(MemoryError):CachedGenotypeAdapter._sparse(p,range(400_000_000),np.array([0]))

    def test_resident_seam_and_original_chunk_schedule(self):
        from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
        import torch
        def materialize(p,ss,vv,device):
            af,miss,mac,R,A=p['summaries']
            return DeviceMinorBlock(torch.from_numpy(self.a._fast.dosage_numpy(p)),ss,vv[p['columns']],af,mac,miss,R,A)
        vv=np.array([8,0,5,3,7]);ss=np.array([3,9,1])
        with patch.object(self.a._fast,'to_minor_block',side_effect=materialize) as call:
            blocks=list(self.a.iter_minor_blocks(vv,ss,block_size=2,resident=True,device='cuda:1'))
            self.assertEqual(call.call_count,3)
            np.testing.assert_array_equal(np.concatenate([b.variant_indices for b in blocks]),vv)
            for i,b in enumerate(blocks):
                c=expected(self.container,vv[2*i:2*i+2],ss,None)
                for imp in ('mean','minor'):
                    a=b.trait_dense(np.arange(3),imp,dtype=torch.float32);r=c.trait_dense(np.arange(3),imp,dtype=np.float32)
                    for left,right in zip(a,r):np.testing.assert_array_equal(left.numpy() if torch.is_tensor(left) else left,right)
            self.assertEqual(call.call_args.kwargs['device'],'cuda:1')
        # CPU mock verifies only adapter routing, never claims a real GPU contract.


class SampleBindingContracts(unittest.TestCase):
    def setUp(self):
        self.c=Container();self.a=CachedGenotypeAdapter(MetadataReader(),self.c)

    def compare(self,samples,variants=None):
        if variants is None:variants=np.arange(8,-1,-1,dtype=np.int64)
        for mac in (None,2,20):
            actual=self.a.minor_block(variants,samples,minimum_mac=mac)
            original=expected(self.c,variants,samples,mac)
            for key in ('row','col','value','sample_indices','variant_indices','union_ref_af','union_initial_mac','union_missing_rate','union_ref_ac','union_called_alleles'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(original,key))
            for imputation in ('mean','minor'):
                left=actual.trait_dense(np.arange(len(samples)),imputation,dtype=np.float32)
                right=original.trait_dense(np.arange(len(samples)),imputation,dtype=np.float32)
                for x,y in zip(left,right):np.testing.assert_array_equal(x,y)

    def test_full_reordered_subset_empty_and_different_dtype(self):
        for samples in (self.c.samples.copy(),self.c.samples[::-1],np.array([3,9,1]),
                        np.empty(0,dtype=np.int64),self.c.samples.astype(np.int32),
                        self.c.samples.astype(np.uint32)):
            self.compare(samples)
            self.compare(samples.copy())
            self.compare(samples,np.empty(0,dtype=np.int64))
        self.assertGreater(self.a._metrics['sample_bind_cache_hits'],0)

    def test_full_identity_route_only_after_validated_binding(self):
        original_prepare=self.a._fast.prepare_validated
        with patch.object(self.a._fast,'prepare_validated',wraps=original_prepare) as prepare:
            samples=self.c.samples.copy()
            self.a.minor_block(np.array([0]),samples)
            self.assertIsNone(prepare.call_args.args[2])
            self.a.minor_block(np.array([8]),samples.copy())
            self.assertIsNone(prepare.call_args.args[2])
            samples[[0,1]]=samples[[1,0]]
            self.a.minor_block(np.array([0]),samples)
            self.assertIsNotNone(prepare.call_args.args[2])
            self.a.minor_block(np.array([0]),samples[:3])
            self.assertIsNotNone(prepare.call_args.args[2])
            samples[0]=samples[1]
            before=prepare.call_count
            with self.assertRaises(ValueError):self.a.minor_block(np.array([0]),samples)
            self.assertEqual(prepare.call_count,before)

    def test_repeated_same_values_one_validation(self):
        samples=self.c.samples.copy()
        for _ in range(10):self.a._bind_samples(samples.copy())
        self.assertEqual(self.a._metrics['sample_bind_calls'],10)
        self.assertEqual(self.a._metrics['sample_bind_cache_hits'],9)
        self.assertEqual(self.a._metrics['sample_bind_validations'],1)
        for axis in self.a._sample_binding[:3]:
            with self.assertRaises(ValueError):axis.flags.writeable=True
        self.assertEqual(self.a._metrics['sample_bind_cache_bytes'],samples.nbytes*3)

    def test_mutation_valid_reorder_and_invalid_fail_closed(self):
        samples=self.c.samples.copy();self.compare(samples)
        samples[[0,1]]=samples[[1,0]];self.compare(samples)
        baseline=samples.copy()
        for replacement,error in ((samples[1],ValueError),(12,IndexError),(-1,IndexError),(11,RuntimeError)):
            samples[:]=baseline;samples[0]=replacement
            with self.assertRaises(error):self.a._bind_samples(samples)
        samples[:]=baseline;self.compare(samples)
        samples[0]=6
        with self.assertRaises(ValueError):self.a.minor_block(np.array([0]),samples)

    def test_dtype_geometry_and_values_are_validated(self):
        self.a._bind_samples(self.c.samples)
        before=self.a._metrics['sample_bind_validations']
        self.a._bind_samples(self.c.samples.astype(np.int32))
        self.assertEqual(self.a._metrics['sample_bind_validations'],before+1)
        for invalid in (self.c.samples.astype(np.float64),self.c.samples.reshape(2,3),np.array(1),['9','2']):
            with self.assertRaises(ValueError):self.a._bind_samples(invalid)
        self.compare([9,2,6,1,10,3])
        self.compare(np.array([3,10,1]))

    def test_metrics_forwarding_and_nested_timer_contract(self):
        self.compare(self.c.samples)
        metadata=self.a.reader_metadata
        self.assertEqual(metadata['native_reads'],{'calls':0})
        self.assertEqual(self.a.read_field('position'),('position',None))
        self.assertEqual(metadata['analysis_cache']['sample_bind_calls'],3)
        self.assertEqual(metadata['analysis_cache']['sample_bind_cache_hits'],2)
        self.assertGreaterEqual(metadata['analysis_cache']['sample_bind_seconds'],metadata['analysis_cache']['sample_validate_seconds'])
        self.a.close();self.assertIsNone(self.a._sample_binding)
        self.assertEqual(self.a._metrics['sample_bind_cache_bytes'],0)

if __name__=='__main__':unittest.main()
