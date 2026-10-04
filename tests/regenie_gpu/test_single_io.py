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
from torchwgs.io import BedReader, Variant, write_bed, read_sample_ids, materialize_bed
from torchwgs.mask_output import MaskWriter
from torchwgs.single import create_test_context, score_genotypes
from torchwgs.output import RegenieWriter
from torchwgs.config import WGSConfig
from torchwgs.summary import select_loci
import pandas as pd


class SingleIOTests(unittest.TestCase):
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
