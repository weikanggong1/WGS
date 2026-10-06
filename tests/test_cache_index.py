import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import numpy as np
from staar_phewas.annotation_index import CandidateAnnotationIndex
from staar_phewas.pipeline import PheWASPipeline
from staar_phewas.cache_runtime import index_cache as a

class Contracts(unittest.TestCase):
    def test_prepare_uses_original_pipeline_then_publishes(self):
        from unittest.mock import Mock
        index,binding=self.fixture()
        pipe=SimpleNamespace(prepare_annotation_index=Mock(return_value=index))
        intervals=[('21',1,10),('21',1,10),('21',12,20)]
        with tempfile.TemporaryDirectory() as tmp:
            prepared,report=a.prepare(pipe,Path(tmp)/'prepared.npz',binding,promoter_intervals=intervals)
            self.assertIs(prepared,index)
            self.assertGreater(report['file_bytes'],0)
            pipe.prepare_annotation_index.assert_called_once_with('21',
                categories=binding['categories'],include_ncrna=False,promoter_intervals=intervals)

    def fixture(self):
        index=CandidateAnnotationIndex('21','SNV')
        index._groups={'gene_b':{'ncRNA':np.array([2,9],np.int64),'upstream':np.array([],np.int64)},
                       'gene_a':{'upstream':np.array([1,2,17],np.int64),'promoter_CAGE':np.array([3],np.int64)}}
        index.prepared_categories={'upstream','ncRNA','promoter_CAGE'}
        index.promoter_signature=((1,10),(1,10),(12,20))
        binding=a.make_binding(gds_stat={'device':1,'inode':2,'size':3,'mtime_ns':4},n_variants=100,
            source_sha256={k:'a'*64 for k in ('annotation_index.py','pipeline.py','masks.py','gds.py')},
            annotation_catalog={'GENCODE.Info':'annotation/info/gene','CAGE':'annotation/info/signal'},qc_path='annotation/filter',
            promoter_manifest={'file_sha256':'b'*64,'normalized_signature':[[1,10],[1,10],[12,20]]},
            chromosome='21',variant_type='SNV',categories=index.prepared_categories)
        return index,binding
    def test_lookup_insertion_empty_promoter_roundtrip(self):
        index,binding=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.npz';a.save(path,index,binding)
            pipe=SimpleNamespace(_annotation_indexes={});restored=a.restore(pipe,path,binding)
            self.assertEqual(list(restored._groups),list(index._groups))
            self.assertEqual(restored.promoter_signature,index.promoter_signature)
            for gene,groups in index._groups.items():
                self.assertEqual(restored.categories_for(gene),tuple(groups))
                for category,rows in groups.items():
                    np.testing.assert_array_equal(restored.indices(gene,category),rows)
                    self.assertFalse(restored.indices(gene,category).flags.writeable)
            self.assertEqual(restored.indices('absent','ncRNA').dtype,np.int64)
            self.assertEqual(len(restored.indices('absent','ncRNA')),0)
            with self.assertRaises(ValueError):restored.indices('gene_a','unprepared')
    def test_restored_index_skips_original_full_metadata_scan(self):
        index,binding=self.fixture()
        def reject(*args,**kw):raise AssertionError('unexpected raw annotation read')
        pipe=SimpleNamespace(_annotation_indexes={},options=SimpleNamespace(variant_type='SNV'),_base_mask=reject,_annotation_categories=reject)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.npz';a.save(path,index,binding);loaded=a.restore(pipe,path,binding)
            actual=PheWASPipeline.prepare_annotation_index(pipe,'chr21',categories=['upstream','ncRNA','promoter_CAGE'],
                include_ncrna=False,promoter_intervals=[('chr21',1,10),('21',12,20),('chr21',1,10)])
            self.assertIs(actual,loaded)
            with self.assertRaisesRegex(ValueError,'promoter references'):
                PheWASPipeline.prepare_annotation_index(pipe,'21',categories=['promoter_CAGE'],include_ncrna=False,promoter_intervals=[('21',1,11)])
    def test_changed_binding_or_budget_and_live_index_fail_closed(self):
        index,binding=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.npz';a.save(path,index,binding)
            changed=dict(binding);changed['qc_path']='annotation/other'
            with self.assertRaisesRegex(ValueError,'binding'):a.restore(SimpleNamespace(_annotation_indexes={}),path,changed)
            with self.assertRaises(MemoryError):a.restore(SimpleNamespace(_annotation_indexes={}),path,binding,max_uncompressed_bytes=16)
            with self.assertRaisesRegex(ValueError,'live prepared'):a.restore(SimpleNamespace(_annotation_indexes={('21','SNV'):index}),path,binding)
            with self.assertRaises(FileExistsError):a.save(path,index,binding)
    def test_invalid_original_order_and_promoter_manifest_rejected(self):
        index,binding=self.fixture();index._groups['gene_a']['upstream']=np.array([2,1],np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):a.save(Path(tmp)/'bad',index,binding)
        index,binding=self.fixture();index.promoter_signature=((1,9),)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'promoter signature'):a.save(Path(tmp)/'bad',index,binding)
    def test_completely_empty_index_roundtrip_without_promoters(self):
        index,binding=self.fixture();index._groups={};index.promoter_signature=None;index.prepared_categories={'ncRNA'}
        binding['categories']=['ncRNA'];binding['promoter_manifest']=None
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty';a.save(path,index,binding);result=a.restore(SimpleNamespace(_annotation_indexes={}),path,binding)
            self.assertEqual(result.genes,());self.assertEqual(result.promoter_signature,None)
            self.assertEqual(result.indices('absent','ncRNA').size,0)

if __name__=='__main__':unittest.main()
