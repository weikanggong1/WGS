import tempfile
import unittest
import gzip
import os
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from scipy.stats import chi2, rankdata, norm
from torchwgs.io import (BedReader, PackedBedBlock, Variant, write_bed, read_sample_ids, materialize_bed,
                         materialize_discovery_bed)
from torchwgs.mask_output import MaskWriter
from torchwgs.single import (create_test_context, score_genotypes,
                            iter_single_variant_results, SingleVariantConfig)
from torchwgs.output import RegenieWriter
from torchwgs.config import WGSConfig
from torchwgs.summary import select_loci
import pandas as pd


class SingleIOTests(unittest.TestCase):
    def test_packed_counts_are_integer_exact_across_tiles_and_ignore_padding(self):
        from torchwgs.single import _allele_counts, _counts_from_observed
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(4103)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(13)]
            rng=np.random.default_rng(135)
            values=rng.integers(0,4,size=(len(ids),len(variants))).astype(float)
            values[values==3]=np.nan
            values[:,0]=np.nan;values[:,1]=0.;values[:,2]=2.;values[:,3]=1.
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            # FAM ends before the fourth call in the last byte. Deliberately
            # populate the padding with a REF hardcall; it must never count.
            bed=Path(str(prefix)+'.bed');raw=bytearray(bed.read_bytes())
            stride=(len(ids)+3)//4
            for j in range(len(variants)):raw[3+j*stride+stride-1]|=0xc0
            bed.write_bytes(raw)
            reader=BedReader(prefix,sample_ids=[ids[i] for i in range(4102,-1,-1)])
            # More than two count tiles, arbitrary order, and a repeated row.
            rows=list(range(0,4099))+[4102,0]
            columns=[12,0,2,4,3,1,7]
            expected=reader.read_variants(columns)[rows]
            expected_counts=_allele_counts(expected)
            devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
            for device in devices:
                for dtype in (torch.float32,torch.float64):
                    block=reader.read_packed_block(columns,sample_rows=rows,
                                                  device=device,dtype=dtype)
                    # Counting is not allowed to call either float decoder.
                    with patch.object(PackedBedBlock,'decode',side_effect=AssertionError('full decode')):
                        counted=block.allele_counts()
                    self.assertTrue(all(v.dtype==torch.int64 for v in counted.values()))
                    self.assertTrue(torch.equal(counted['N'].cpu(),expected_counts['N']))
                    self.assertTrue(torch.equal(counted['AAC'].cpu(),torch.nan_to_num(expected).sum(0).long()))
                    for name,value in [('N_MISSING',torch.isnan(expected)),
                                       ('N_HOM_A1',expected==2),('N_HET',expected==1),
                                       ('N_HOM_A0',expected==0)]:
                        self.assertTrue(torch.equal(counted[name].cpu(),value.sum(0)))
                    self.assertTrue(torch.equal((counted['N']+counted['N_MISSING']).cpu(),
                                                torch.full((len(columns),),len(rows))))
                    observed=_counts_from_observed(counted['N'],counted['AAC'])
                    for name in expected_counts:
                        self.assertTrue(torch.equal(observed[name].cpu(),expected_counts[name]),name)
                    decoded=block.decode([6,2,0,2,1]).cpu()
                    oracle=expected[:,[6,2,0,2,1]].to(dtype).contiguous()
                    bits=torch.int32 if dtype==torch.float32 else torch.int64
                    self.assertTrue(torch.equal(decoded.view(bits),oracle.view(bits)))

    def test_packed_block_empty_selection_and_invalid_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'source'
            write_bed(prefix,torch.tensor([[0.,2.],[1.,float('nan')],[2.,0.]]),
                      [Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(2)],
                      [('1','1'),('2','2'),('3','3')])
            reader=BedReader(prefix)
            devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
            for device in devices:
                empty=reader.read_packed_block([],device=device)
                self.assertEqual(empty.decode().shape,(3,0))
                self.assertTrue(all(count.numel()==0 for count in empty.allele_counts().values()))
                no_samples=reader.read_packed_block([0,1],sample_rows=[],device=device)
                self.assertEqual(no_samples.decode().shape,(0,2))
                self.assertTrue(all(not count.any() for count in no_samples.allele_counts().values()))
                block=reader.read_packed_block([0,1],device=device)
                self.assertEqual(block.decode([]).shape,(3,0))
                for invalid in ([-1],[2],torch.tensor([0,2])):
                    with self.assertRaises(IndexError):block.decode(invalid)
                with self.assertRaises(ValueError):reader.read_packed_block([0],device=device,dtype='float16')
                with self.assertRaises(ValueError):list(reader.iter_packed_variant_blocks(0,device=device))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed for retained-column association')
    def test_packed_single_filters_before_any_decode_and_preserves_mac_boundary(self):
        from torchwgs.single import _score_counted_genotypes
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(41)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(7)]
            values=np.zeros((len(ids),len(variants)))
            values[:,1]=2.;values[:,2]=np.nan
            values[:2,3]=1.;values[:3,4]=1.;values[:4,5]=1.;values[9,5]=np.nan
            values[:,6]=2.;values[:4,6]=1.
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            rows=list(range(40,-1,-1))
            include=[6,4,3,4,5,0,2,1]
            for dtype in ('float32','float64'):
                context=create_test_context(np.random.default_rng(15).normal(size=len(rows)),
                    sample_ids=[ids[i] for i in rows],device='cuda',dtype=dtype,apply_rint=False)
                common=dict(min_mac=3,maf_min=0,block_size=3,device='cuda',dtype=dtype)
                expected=list(iter_single_variant_results(BedReader(prefix),context,
                    config=SingleVariantConfig(**common,genotype_reader='cpu'),variant_indices=include))
                decoder=PackedBedBlock.decode
                with patch.object(PackedBedBlock,'decode',autospec=True,side_effect=decoder) as decode, \
                     patch.object(BedReader,'read_variants',side_effect=AssertionError('CPU expanded genotypes')), \
                     patch.object(BedReader,'read_packed_variants',side_effect=AssertionError('all columns decode')), \
                     patch('torchwgs.single._allele_counts',side_effect=AssertionError('expanded counts')), \
                     patch('torchwgs.single._score_counted_genotypes',wraps=_score_counted_genotypes) as score:
                    observed=list(iter_single_variant_results(BedReader(prefix),context,
                        config=SingleVariantConfig(**common,genotype_reader='cuda_packed'),variant_indices=include))
                self.assertEqual(observed,expected)
                self.assertEqual([row['ID'] for row in observed],['v4','v5','v6'])
                self.assertEqual([call.args[1] for call in decode.call_args_list],[[1,2],[0]])
                self.assertEqual(sum(call.args[0].shape[1] for call in score.call_args_list),3)
                self.assertEqual(observed[0]['MAC'],3.)
                self.assertGreater(observed[2]['A1FREQ'],.5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed for device-local fallback')
    def test_packed_cuda_torch_fallback_matches_fused_counts_and_decode_bits(self):
        from torchwgs import _packed_gpu
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'source'
            ids=[(str(i),str(i)) for i in range(2051)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(5)]
            values=(np.arange(len(ids)*5).reshape(len(ids),5)%4).astype(float)
            values[values==3]=np.nan
            write_bed(prefix,values,variants,ids)
            reader=BedReader(prefix)
            for dtype in (torch.float32,torch.float64):
                block=reader.read_packed_block([4,1,0],sample_rows=list(range(2050,-1,-1)),
                                              device='cuda',dtype=dtype)
                expected_counts=block.allele_counts();expected=block.decode([2,0,2])
                with patch.object(_packed_gpu,'triton',None):
                    observed_counts=block.allele_counts();observed=block.decode([2,0,2])
                self.assertEqual(observed.device.type,'cuda')
                for name in expected_counts:self.assertTrue(torch.equal(observed_counts[name],expected_counts[name]))
                bits=torch.int32 if dtype==torch.float32 else torch.int64
                self.assertTrue(torch.equal(observed.view(bits),expected.view(bits)))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed for dynamic-width decoder')
    def test_packed_decode_variable_widths_preserve_calls_bits_and_duplicate_order(self):
        # Retained MAC widths vary between blocks, including Triton scalar 1
        # and aligned widths. Decode from the same packed calls at every width.
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(2049)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(67)]
            values=(np.arange(len(ids)*67).reshape(len(ids),67)%4).astype(float)
            values[values==3]=np.nan
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            reader=BedReader(prefix)
            rows=list(range(2048,-1,-1))
            for dtype in (torch.float32,torch.float64):
                block=reader.read_packed_block(range(67),sample_rows=rows,device='cuda',dtype=dtype)
                for width in (1,2,16,17,32,63,67,129,1):
                    columns=[(i*7)%67 for i in range(width)]
                    expected=reader.read_variants(columns)[rows].to(dtype).contiguous()
                    actual=block.decode(columns).cpu()
                    bits=torch.int32 if dtype==torch.float32 else torch.int64
                    self.assertTrue(torch.equal(actual.view(bits),expected.view(bits)))

    def test_packed_torch_decode_is_bit_exact_with_missing_and_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(11)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(9)]
            values=(np.arange(99).reshape(11,9)%4).astype(float)
            values[values==3]=np.nan
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            reader=BedReader(prefix,sample_ids=[ids[i] for i in [10,3,0,7,1]])
            indices=[8,1,0,5]
            rows=torch.tensor([4,0,2])
            expected=reader.read_variants(indices)[rows]
            devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
            for device in devices:
                for dtype in [torch.float32,torch.float64]:
                    observed=reader.read_packed_variants(indices,sample_rows=rows,
                                                        device=device,dtype=dtype).cpu()
                    np.testing.assert_array_equal(observed.numpy(),expected.to(dtype).numpy())
                    # Compare the actual floating-point bits, including NaN calls.
                    bits=torch.int32 if dtype==torch.float32 else torch.int64
                    self.assertTrue(torch.equal(observed.view(bits),expected.to(dtype).contiguous().view(bits)))
                chunks=list(reader.iter_variant_blocks(2,indices,genotype_reader='cuda_packed',
                                                       sample_rows=rows,device=device))
                observed=torch.cat([chunk for _,chunk in chunks],dim=1).cpu()
                expected_order=sorted(indices)
                np.testing.assert_array_equal(observed,reader.read_variants(expected_order)[rows])
            self.assertLessEqual(len(reader._packed_decoder_cache),4)
            self.assertEqual(reader.read_packed_variants([],device='cpu').shape,(reader.n_samples,0))
            with self.assertRaises(IndexError):reader.read_packed_variants([reader.n_variants],device='cpu')
            with self.assertRaises(IndexError):reader.read_packed_variants([0],sample_rows=[5],device='cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed for GPU decoder equivalence')
    def test_packed_single_reader_matches_cpu_decoder_all_result_values(self):
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(37)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(13)]
            rng=np.random.default_rng(65)
            values=rng.binomial(2,.2,size=(37,13)).astype(float)
            values[rng.random(values.shape)<.07]=np.nan
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            rows=list(range(35,0,-1))
            for dtype in ['float32','float64']:
                context=create_test_context(rng.normal(size=len(rows)),sample_ids=[ids[i] for i in rows],
                                           apply_rint=True,device='cuda',dtype=dtype)
                common=dict(maf_min=0,min_mac=1,block_size=4,device='cuda',dtype=dtype)
                cpu=list(iter_single_variant_results(BedReader(prefix),context,
                         config=SingleVariantConfig(**common,genotype_reader='cpu')))
                packed=list(iter_single_variant_results(BedReader(prefix),context,
                            config=SingleVariantConfig(**common,genotype_reader='cuda_packed')))
                self.assertEqual(cpu,packed)

    def test_streaming_discovery_bed_preserves_all_variants_calls_and_fam(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'source'
            ids=[(str(i),str(i)) for i in range(11)]
            variants=[Variant(i,'21',f'v{i}',i+10,'A','G') for i in range(9)]
            rng=np.random.default_rng(40)
            values=rng.integers(0,4,size=(11,9)).astype(float)
            values[values==3]=np.nan
            write_bed(prefix,values,variants,ids)
            fam=Path(str(prefix)+'.fam')
            fam.write_text(''.join(f'{fid} {iid} 10 20 {1+i%2} -9\n'
                                   for i,(fid,iid) in enumerate(ids)))
            bed=Path(str(prefix)+'.bed');original_bed=bed.read_bytes()
            original_fam=fam.read_bytes()
            keep={ids[i] for i in [0,2,3,5,7,9,10]};remove={ids[3]}
            selected=[i for i,sid in enumerate(ids) if sid in keep and sid not in remove]
            cached=materialize_discovery_bed(prefix,Path(directory)/'plain',keep=keep,
                                             remove=remove,block_variants=2)
            reader=BedReader(cached)
            self.assertEqual(reader.n_variants,len(variants))
            self.assertEqual(reader.sample_ids,[ids[i] for i in selected])
            self.assertEqual(reader.sample_sex,[1+i%2 for i in selected])
            np.testing.assert_allclose(reader.read_variants(range(9)),values[selected],equal_nan=True)
            self.assertEqual(Path(str(cached)+'.fam').read_text(),
                             ''.join(fam.read_text().splitlines(True)[i] for i in selected))
            target_bed=Path(str(cached)+'.bed');before=target_bed.stat().st_mtime_ns
            self.assertEqual(cached,materialize_discovery_bed(prefix,Path(directory)/'plain',
                             keep=keep,remove=remove,block_variants=3))
            self.assertEqual(before,target_bed.stat().st_mtime_ns)
            compressed=Path(str(prefix)+'.bed.gz')
            with gzip.open(compressed,'wb') as stream:stream.write(original_bed)
            bed.unlink()
            cached_gz=materialize_discovery_bed(prefix,Path(directory)/'gzip',keep=keep,
                                               remove=remove,block_variants=4)
            self.assertEqual(target_bed.read_bytes(),Path(str(cached_gz)+'.bed').read_bytes())
            requested=[ids[i] for i in selected[::-1]]
            reordered=materialize_discovery_bed(prefix,Path(directory)/'gzip',keep=keep,
                        remove=remove,sample_ids=requested,block_variants=2)
            np.testing.assert_allclose(BedReader(reordered).read_variants(range(9)),
                                       values[selected[::-1]],equal_nan=True)
            self.assertEqual(fam.read_bytes(),original_fam)
            self.assertFalse(bed.exists())

    def test_streaming_discovery_bed_rejects_truncated_and_corrupt_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'source'
            write_bed(prefix,torch.zeros((5,3)),
                      [Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(3)],
                      [(str(i),str(i)) for i in range(5)])
            bed=Path(str(prefix)+'.bed');original=bed.read_bytes()
            bed.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError,'Source BED'):
                materialize_discovery_bed(prefix,Path(directory)/'cache',block_variants=1)
            self.assertFalse(list(Path(directory).rglob('*.partial')))
            compressed=Path(str(prefix)+'.bed.gz')
            with gzip.open(compressed,'wb') as stream:stream.write(original)
            bed.unlink()
            raw=bytearray(compressed.read_bytes());raw[-8]^=1
            compressed.write_bytes(raw)
            with self.assertRaises(gzip.BadGzipFile):
                materialize_discovery_bed(prefix,Path(directory)/'cache',block_variants=1)
            self.assertFalse(list(Path(directory).rglob('*.partial')))

    def test_single_prefilter_keeps_full_score_values_in_reader_order(self):
        from torchwgs.single import _score_counted_genotypes
        with tempfile.TemporaryDirectory() as directory:
            ids=[(str(i),str(i)) for i in range(31)]
            variants=[Variant(i,'21',f'v{i}',i+1,'A','G') for i in range(5)]
            rng=np.random.default_rng(42)
            values=rng.binomial(2,.2,size=(31,5)).astype(float)
            values[:,0]=0.;values[:2,0]=1.
            values[:,2]=np.nan
            values[:,3]=2.
            values[4,4]=np.nan
            prefix=Path(directory)/'source';write_bed(prefix,values,variants,ids)
            # Context may be a reordered subset of the reader's FAM.
            rows=[i for i in range(30,-1,-1) if i!=5]
            context=create_test_context(rng.normal(size=len(rows)),sample_ids=[ids[i] for i in rows],
                                       apply_rint=False,device='cpu',dtype='float64')
            expected=score_genotypes(values[rows],context)
            config=SingleVariantConfig(maf_min=.05,min_mac=3,block_size=2,device='cpu',dtype='float64')
            with patch('torchwgs.single._score_counted_genotypes',wraps=_score_counted_genotypes) as score:
                observed=list(iter_single_variant_results(BedReader(prefix),context,config=config))
                self.assertEqual(sum(call.args[0].shape[1] for call in score.call_args_list),2)
            selected=((expected['MAC']>=3)&(expected['MAF']>.05)&expected['VALID']).nonzero().flatten().tolist()
            self.assertEqual([row['ID'] for row in observed],[variants[i].id for i in selected])
            for row,index in zip(observed,selected):
                for key in ['N','A1FREQ','MAF','MAC','BETA','SE','CHISQ','LOG10P']:
                    self.assertAlmostEqual(row[key],float(expected[key][index]),places=12)
            self.assertEqual(expected['A1FREQ'].dtype,torch.float64)

    def test_bim_cache_preserves_new_missing_ids_and_bim_order(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'genotype'
            variants=[Variant(i,'1',f'v{i}',10+i,'A','G') for i in range(3)]
            write_bed(prefix,torch.zeros((3,3)),variants,[('1','1'),('2','2'),('3','3')])
            reader=BedReader(prefix)
            with patch('builtins.open',wraps=open) as opened:
                first=reader.find_variants(['v1','missing'])
                self.assertEqual(list(first),['v1'])
                self.assertEqual(opened.call_count,1)
                self.assertEqual(reader.find_variants(['missing','v1']),first)
                self.assertEqual(opened.call_count,1)
                self.assertEqual(list(reader.find_variants(['v2'])),['v2'])
                self.assertEqual(opened.call_count,2)
                result=reader.find_variants(['v2','missing','v0','v1'])
                self.assertEqual(list(result),['v0','v1','v2'])
                self.assertEqual(opened.call_count,3)
                self.assertEqual(list(reader.find_variants(['v0','v2'])),['v0','v2'])
                self.assertEqual(opened.call_count,3)

    def test_bim_cache_invalidates_on_size_time_and_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'genotype'
            write_bed(prefix,torch.zeros((3,1)),[Variant(0,'1','v',10,'A','G')],
                      [('1','1'),('2','2'),('3','3')])
            reader=BedReader(prefix)
            path=Path(str(prefix)+'.bim')
            self.assertEqual(reader.find_variants(['v'])['v'].position,10)
            before=path.stat()
            path.write_text('1 v 0 100 A G\n')
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(reader.find_variants(['v'])['v'].position,100)
            # Same length, updated mtime, and a previously cached missing ID.
            self.assertEqual(reader.find_variants(['w']),{})
            path.write_text('1 w 0 200 A G\n')
            before=path.stat()
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000))
            self.assertEqual(reader.find_variants(['v']),{})
            self.assertEqual(reader.find_variants(['w'])['w'].position,200)
            # Atomic replacement can preserve both size and mtime; inode/ctime
            # must still invalidate positive and negative query results.
            before=path.stat()
            replacement=Path(str(prefix)+'.replacement')
            replacement.write_text('1 v 0 300 A G\n')
            os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(reader.find_variants(['v'])['v'].position,300)
            self.assertEqual(reader.find_variants(['w']),{})

    def test_bim_cache_has_bounded_positive_and_negative_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'genotype'
            variants=[Variant(i,'1',f'v{i}',i+1,'A','G') for i in range(4)]
            write_bed(prefix,torch.zeros((3,4)),variants,[('1','1'),('2','2'),('3','3')])
            reader=BedReader(prefix,metadata_cache_size=2)
            self.assertEqual(list(reader.find_variants(['v3','v2','v1','v0','absent'])),
                             ['v0','v1','v2','v3'])
            self.assertLessEqual(len(reader._variant_metadata),2)
            self.assertEqual(list(reader.find_variants(['v0','v3'])),['v0','v3'])
            self.assertLessEqual(len(reader._variant_metadata),2)

    def test_changed_bim_still_rejects_duplicate_requested_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'genotype'
            write_bed(prefix,torch.zeros((3,1)),[Variant(0,'1','v',1,'A','G')],
                      [('1','1'),('2','2'),('3','3')])
            reader=BedReader(prefix)
            self.assertEqual(list(reader.find_variants(['v'])),['v'])
            Path(str(prefix)+'.bim').write_text('1 v 0 1 A G\n1 v 0 2 A G\n')
            with self.assertRaisesRegex(ValueError,'Duplicate requested variant ID'):
                reader.find_variants(['v'])

    def test_bed_codes_padding_and_requested_sample_order(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/'genotype'
            variants=[Variant(0,'1','v1',10,'A','G'),Variant(1,'2','v2',20,'T','C')]
            ids=[(str(i),str(i)) for i in range(5)]
            g=torch.tensor([[2.,0.],[float('nan'),1.],[1.,2.],[0.,float('nan')],[2.,1.]])
            write_bed(prefix,g,variants,ids)
            raw=Path(str(prefix)+'.bed').read_bytes()
            self.assertEqual(raw[:4],b'\x6c\x1b\x01\xe4')
            reader=BedReader(prefix,sample_ids=[ids[4],ids[1],ids[0]])
            np.testing.assert_allclose(reader.read_variants([1,0]),g[[4,1,0]][:,[1,0]],equal_nan=True)
            self.assertEqual(list(reader.find_variants(['v1'])),['v1'])

    def test_original_score_chi_square_with_missing_genotypes(self):
        rng=np.random.default_rng(6)
        y=rng.normal(size=93);pred=rng.normal(size=93)*.1
        g=rng.binomial(2,.07,size=(93,7)).astype(float);g[4,2]=np.nan
        device='cuda' if torch.cuda.is_available() else 'cpu'
        context=create_test_context(y,pred,apply_rint=True,device=device,dtype='float64',tf32=False)
        observed=score_genotypes(g,context)
        y0=norm.ppf((rankdata(y)-.375)/(len(y)+.25));y0-=y0.mean()
        sy=np.linalg.norm(y0)/np.sqrt(len(y)-1)
        r=y0/sy-pred;sr=np.linalg.norm(r)/np.sqrt(len(y)-1);r/=sr
        imputed=np.where(np.isfinite(g),g,np.nanmean(g,axis=0));imputed-=imputed.mean(axis=0)
        u=imputed.T@r;v=(imputed**2).sum(0)
        np.testing.assert_allclose(observed['BETA'].cpu(),u*sy*sr/v,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(observed['SE'].cpu(),sy*sr/np.sqrt(v),rtol=1e-12)
        np.testing.assert_allclose(observed['LOG10P'].cpu(),-chi2.logsf(u*u/v,1)/np.log(10),rtol=1e-12)

    def test_rint_is_optional_at_all_stages(self):
        configuration=WGSConfig.paper(apply_rint=False)
        self.assertFalse(configuration.step1.apply_rint)
        self.assertFalse(configuration.single_variant.apply_rint)
        self.assertFalse(configuration.gene_based.apply_rint)
        context=create_test_context([0.,1.,2.,3.,20.],apply_rint=False,device='cpu',dtype='float64')
        expected=torch.tensor([0.,1.,2.,3.,20.],dtype=torch.float64);expected-=expected.mean()
        np.testing.assert_allclose(context.y,expected/expected.std(),atol=1e-14)

    def test_regenie_native_columns_precision_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/'association'
            row={'CHROM':'1','GENPOS':10,'ID':'v1','ALLELE0':'G','ALLELE1':'A','A1FREQ':.123456789,
                 'N':100,'TEST':'ADD','BETA':-.012345678,'SE':.023456789,'CHISQ':.276,'LOG10P':.223,'EXTRA':'NA'}
            with RegenieWriter(prefix,'trait',sample_ids=[('2','2'),('10','10')]) as writer:writer.write(row)
            expected='CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE CHISQ LOG10P EXTRA\n1 10 v1 G A 0.123457 100 ADD -0.0123457 0.0234568 0.276 0.223 NA\n'
            self.assertEqual((Path(d)/'association_trait.regenie').read_text(),expected)
            self.assertEqual((Path(d)/'association_trait.regenie.ids').read_text(),'10\t10\n2\t2\n')

    def test_paper_lead_distance_uses_selected_leads(self):
        frame=pd.DataFrame({'CHROM':[1,1,1,1],'GENPOS':[1000000,1450000,1900000,3100000],
                            'LOG10P':[20.,19.,18.,17.],'ID':['a','b','c','d']})
        loci=select_loci(frame)
        self.assertEqual(loci.ID.tolist(),['a','d'])
        self.assertEqual(loci.N_LEADS.tolist(),[2,1])

    def test_interrupted_output_preserves_completed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'association'
            with RegenieWriter(prefix,'trait',sample_ids=[('1','1')]) as writer:
                writer.write({'ID':'complete'})
            completed=writer.path.read_bytes()
            with self.assertRaises(RuntimeError):
                with RegenieWriter(prefix,'trait',sample_ids=[('2','2')]) as writer:
                    writer.write({'ID':'incomplete'})
                    raise RuntimeError('Interrupted computation')
            self.assertEqual(writer.path.read_bytes(),completed)
            self.assertEqual(writer.ids_path.read_text(),'1\t1\n')
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_mask_artifact_preserves_missing_calls_and_sample_sex(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'association'
            ids=[('1','1'),('2','2'),('3','3'),('4','4'),('5','5')]
            raw=torch.tensor([0.,1.,2.,float('nan'),0.])
            mask=SimpleNamespace(aaf_upper=.01,frequency='0.01',name='Mask1',
                base_name='Mask1',raw_burden=raw,burden=torch.nan_to_num(raw),variant_ids=('a','b'))
            artifacts=SimpleNamespace(gene=SimpleNamespace(gene='GENE',chrom='5',position=123),masks=[mask])
            with MaskWriter(prefix,ids,[1,2,1,2,0]) as writer:writer(artifacts)
            reader=BedReader(str(prefix)+'_masks')
            np.testing.assert_allclose(reader.read_variants([0]).flatten(),raw,equal_nan=True)
            self.assertEqual(reader.sample_sex,[1,2,1,2,0])
            self.assertEqual(Path(str(prefix)+'_masks.snplist').read_text(),'GENE.Mask1.0.01\ta,b\n')
            original=Path(str(prefix)+'_masks.bed').read_bytes()
            with self.assertRaises(RuntimeError):
                with MaskWriter(prefix,ids,[1,2,1,2,0]) as writer:
                    writer(artifacts)
                    raise RuntimeError('Interrupted mask construction')
            self.assertEqual(Path(str(prefix)+'_masks.bed').read_bytes(),original)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_compressed_input_uses_cache_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'source'
            write_bed(prefix,torch.tensor([[0.],[1.],[2.]]),[Variant(0,'1','v',1,'A','C')],
                      [('1','1'),('2','2'),('3','3')])
            bed=Path(str(prefix)+'.bed'); original=bed.read_bytes()
            compressed=Path(str(prefix)+'.bed.gz')
            with gzip.open(compressed,'wb') as stream:stream.write(original)
            bed.unlink();before=compressed.read_bytes()
            cached=materialize_bed(prefix,Path(directory)/'cache')
            self.assertNotEqual(str(cached),str(prefix))
            self.assertEqual(Path(str(cached)+'.bed').read_bytes(),original)
            self.assertEqual(compressed.read_bytes(),before)
            self.assertFalse(bed.exists())

if __name__=='__main__':unittest.main()
