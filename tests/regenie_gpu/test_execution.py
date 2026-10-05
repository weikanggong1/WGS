"""Scheduler correctness fixtures; these are not performance benchmarks."""
import json
import gzip
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from torchwgs import (DiscoveryInputs, ExecutionConfig, GeneAnalysis, NullModel,
                      WGSConfig, run_discovery, study_gene_analyses)
from torchwgs.execution import GpuExecutor
from torchwgs.io import Variant, write_bed
from torchwgs.summary import summarize_results
from torchwgs.config import SignificanceConfig


class ExecutionTests(unittest.TestCase):
    def test_imported_loco_uses_known_tokens_and_allows_omitted_and_na_samples(self):
        from torchwgs.pipeline import _import_aligned_loco
        with TemporaryDirectory() as directory:
            root=Path(directory)
            # FID and IID both contain underscores; source order differs from FAM.
            ids=[('family_A',f'sample_{i}') for i in range(5)]
            tokens=[f'{fid}_{iid}' for fid,iid in (ids[3],ids[0],ids[4],ids[1])]
            text='FID_IID '+' '.join(tokens)+'\n21 3.5 0.5 NA 1.5\n'
            for compressed in (False,True):
                loco=root/('source.loco.gz' if compressed else 'source.loco')
                if compressed:
                    with gzip.open(loco,'wt') as stream:stream.write(text)
                else:loco.write_text(text)
                predictions=root/'source_pred.list'
                predictions.write_text(f'qt {loco.name}\n')
                for source in (loco,predictions):
                    result=_import_aligned_loco(source,'qt',ids)
                    self.assertEqual(result.sample_ids,[ids[0],ids[1],ids[3]])
                    torch.testing.assert_close(result.loco[:,0],torch.tensor([.5,1.5,3.5],dtype=torch.float64))
            with self.assertRaisesRegex(ValueError,'token collision'):
                _import_aligned_loco(loco,'qt',[('family_A','sample_0'),('family','A_sample_0')])
            with self.assertRaisesRegex(ValueError,'No array FAM sample'):
                _import_aligned_loco(loco,'qt',[('unrelated','sample')])

    def test_corrupt_cache_schema_and_unreadable_artifacts_are_cache_misses(self):
        from torchwgs.pipeline import _read_manifest, _stage_cache, _identities_match, _file_identity
        with TemporaryDirectory() as directory:
            root=Path(directory)
            prefix=root/'output'
            manifest=Path(str(prefix)+'.manifest.json')
            for invalid in ([],None,3,'text'):
                manifest.write_text(json.dumps(invalid))
                self.assertEqual(_read_manifest(manifest),{})
                self.assertIsNone(_stage_cache(prefix,'fixture'))
            result=root/'result.regenie'
            result.write_text('complete result')
            valid={'key':'fixture','result':str(result),'report':{'rows':1},
                   'artifacts':[_file_identity(result)]}
            manifest.write_text(json.dumps(valid))
            self.assertEqual(_stage_cache(prefix,'fixture'),valid)
            for invalid_result in (None,[],{},1,''):
                manifest.write_text(json.dumps({**valid,'result':invalid_result}))
                self.assertIsNone(_stage_cache(prefix,'fixture'))
            for invalid_artifacts in (None,{},[],[None],[{}],[{'path':None}],[{'path':'\u0000'}]):
                manifest.write_text(json.dumps({**valid,'artifacts':invalid_artifacts}))
                self.assertIsNone(_stage_cache(prefix,'fixture'))
            manifest.write_text(json.dumps({**valid,'report':None}))
            self.assertIsNone(_stage_cache(prefix,'fixture'))
            with patch('torchwgs.pipeline._file_identity',side_effect=OSError('artifact unavailable')):
                self.assertFalse(_identities_match(valid['artifacts']))

    def test_context_rejects_zero_finite_aligned_samples_before_projection(self):
        from torchwgs.single import create_test_context
        for phenotype,prediction in (([float('nan')]*3,[0.]*3),([1.,2.,3.],[float('nan')]*3),([],[])):
            with self.assertRaisesRegex(ValueError,'No finite aligned phenotype/LOCO samples'):
                create_test_context(phenotype,prediction,device='cpu')

    def test_serial_order_errors_and_gpu_only_contract(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, 'CUDA GPU'):
            GpuExecutor(ExecutionConfig(), 'cpu')
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU association fallback'):
                GpuExecutor(ExecutionConfig(), 'cuda:0')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device', return_value=nullcontext()), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=80*1024**3)), \
             patch('torch.cuda.set_per_process_memory_fraction') as limit:
            with GpuExecutor(ExecutionConfig(), 'cuda:0') as executor:
                self.assertEqual(executor.map(lambda i: i*i, [3, 1, 2]), [9, 1, 4])
                with self.assertRaisesRegex(ValueError, 'stage failure'):
                    executor.map(lambda i: (_ for _ in ()).throw(ValueError('stage failure')), [1, 2])
            self.assertEqual(limit.call_args.args[0], .25)
        for arguments in [('mask', 1, 20.), ('chromosome', 1, 20.), ('unknown', 1, 20.),
                          ('serial', 0, 20.), ('serial', 2, 20.), ('serial', 1.5, 20.),
                          ('serial', True, 20.), ('serial', False, 20.),
                          ('serial', 1, float('nan')), ('serial', 1, float('inf')),
                          ('serial', 1, -float('inf')), ('serial', 1, 0.),
                          ('serial', 1, -1.), ('serial', 1, True), ('serial', 1, '20')]:
            with self.assertRaises(ValueError):
                ExecutionConfig(*arguments)

    def test_complete_mask_inventory_and_configuration_roundtrip(self):
        analyses = study_gene_analyses('/fixture/annotations', chromosomes=[21], require_files=False)['21']
        self.assertEqual(len(analyses), 26)
        self.assertTrue({'Intergenic', 'Pseudo', 'RNA'}.issubset({a.name for a in analyses}))
        config = WGSConfig.paper(apply_rint=False)
        config.execution = ExecutionConfig('serial', 1, 16.)
        loaded = WGSConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(json.dumps(loaded.to_dict()), json.dumps(config.to_dict()))
        self.assertEqual(loaded.single_variant.maf_min, 0.)
        self.assertTrue(loaded.print_pheno_name)
        self.assertTrue(loaded.write_masks)
        self.assertFalse(WGSConfig().write_masks)
        partial=WGSConfig.from_dict({'single_variant':{'min_mac':30}})
        self.assertEqual(partial.single_variant.min_mac,30)
        self.assertEqual(partial.single_variant.genotype_reader,'cuda_packed')
        self.assertEqual(partial.gene_based.vc_storage,'sparse')
        self.assertEqual(partial.gene_based.genotype_reader,'cuda_packed')
        self.assertEqual(partial.significance.single_frequency_field,'maf')

    def test_source_frequency_filter_happens_after_raw_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'raw.regenie'
            frame = pd.DataFrame({'CHROM':[21]*3, 'GENPOS':[1, 2, 3],
                                  'LOG10P':[20.]*3, 'A1FREQ':[.0009, .002, .9995]})
            frame.to_csv(path, sep=' ', index=False)
            original = path.read_bytes()
            source = summarize_results([path], [], output_dir=Path(directory)/'source')
            minor = summarize_results([path], [], output_dir=Path(directory)/'minor',
                                      config=SignificanceConfig(single_frequency_field='maf'))
            self.assertEqual(source['single_significant'], 2)
            self.assertEqual(minor['single_significant'], 1)
            self.assertEqual(path.read_bytes(), original)

    def test_source_mhc_exclusion_is_closed_and_only_filters_locus_candidates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'raw.regenie'
            frame = pd.DataFrame({'CHROM':[6, 6, 6, 6, 21],
                                  'GENPOS':[24999999, 25000000, 34000000, 34000001, 25000000],
                                  'LOG10P':[20., 30., 29., 21., 22.], 'A1FREQ':[.002]*5})
            frame.to_csv(path, sep=' ', index=False)
            original = path.read_bytes()
            source = summarize_results([path], [], output_dir=Path(directory)/'source')
            unrestricted = summarize_results([path], [], output_dir=Path(directory)/'all',
                                              config=SignificanceConfig(excluded_locus_regions=()))
            self.assertEqual(source['single_significant'],5)
            self.assertEqual(source['loci'],3)
            self.assertEqual(unrestricted['loci'],3)
            selected=pd.read_csv(Path(directory)/'source/single_loci.tsv',sep='\t')
            self.assertEqual(set(selected.GENPOS),{24999999,34000001,25000000})
            self.assertEqual(selected.loc[selected.CHROM==6,'LOG10P'].max(),21.)
            loaded=WGSConfig.from_dict({'significance':{'excluded_locus_regions':[[6,25000000,34000000]]}})
            self.assertEqual(loaded.significance.excluded_locus_regions,((6,25000000,34000000),))
            for invalid in [((6,0,10),),((6,20,10),),((6,1.5,10),),((6,1),)]:
                with self.assertRaises(ValueError):
                    SignificanceConfig(excluded_locus_regions=invalid)
            self.assertEqual(path.read_bytes(),original)

    def test_pipeline_rejects_cpu_before_reading_private_inputs(self):
        with TemporaryDirectory() as directory:
            inputs=DiscoveryInputs('/unreadable/array', '/unreadable/trait', 'trait_01',
                                   '/unreadable/cohort', {})
            config=WGSConfig.paper()
            config.step1=replace(config.step1, device='cpu')
            with self.assertRaisesRegex(ValueError, 'CUDA for Step1 and Step2'):
                run_discovery(inputs, config=config, output_dir=Path(directory)/'result')
            self.assertFalse((Path(directory)/'result').exists())

    def test_sub_whitelist_intersects_user_filter_and_header_uses_complete_annotation(self):
        from torchwgs.chromosome import _gene_job
        from torchwgs.single import create_test_context
        with TemporaryDirectory() as directory:
            root=Path(directory)
            ids=[(str(i),str(i)) for i in range(1000)]
            values=np.zeros((1000,3),dtype=np.float32)
            values[:13,0]=1
            values[13:27,1]=1
            values[27:40,2]=1
            variants=[Variant(i,'21',f'v{i}',i+1,'G','A') for i in range(3)]
            prefix=root/'source'
            write_bed(prefix,values,variants,ids)
            annotation=root/'annotation.txt'
            annotation.write_text('v0 G A\nv1 G B\nv2 OutsideSet C\nnot_in_bim OutsideSet Unknown\n')
            setlist=root/'sets.txt'
            setlist.write_text('G 21 1 v0,v1\n')
            mask=root/'masks.txt'
            mask.write_text('MaskA A\nMaskB B\nMaskC C\nMaskUnknown Unknown\nMaskMixed C,A,Unknown\n')
            score=root/'score.txt'
            score.write_text('v0\nv1\nv2\nnot_in_bim\n')
            phenotype=torch.cos(torch.arange(1000,dtype=torch.float64)*.17)
            context=create_test_context(phenotype,torch.zeros_like(phenotype),sample_ids=ids,
                                        apply_rint=False,device='cpu',dtype='float64')
            inputs=DiscoveryInputs(str(prefix),str(root/'unused_trait.txt'),'qt',str(prefix)+'.fam',
                                   {'21':str(prefix)})
            config=WGSConfig.paper(apply_rint=False)
            config.single_variant.device='cpu'
            config.gene_based.genotype_reader='cpu'
            config.gene_based.extract_variants={'v0','v2'}
            config.gene_based.include_singletons=False
            analysis=GeneAnalysis('Sub',str(annotation),str(setlist),str(mask),str(score))
            (root/'result/Gene').mkdir(parents=True)
            _,result,report=_gene_job(analysis,prefix=prefix,context=context,inputs=inputs,
                                     config=config,destination=root/'result',key=('21','fixture'))
            text=Path(result).read_text()
            self.assertTrue(text.startswith('##MASKS=<MaskA="A";MaskC="C";MaskMixed="C,A">\n'))
            self.assertNotIn('G.MaskB',text)
            self.assertIn('G.MaskA',text)
            self.assertEqual(report['mask_definitions'],5)
            self.assertEqual(report['header_mask_definitions'],3)
            config.gene_based.extract_variants=set()
            (root/'empty/Gene').mkdir(parents=True)
            _,_,empty=_gene_job(analysis,prefix=prefix,context=context,inputs=inputs,
                               config=config,destination=root/'empty',key=('21','empty'))
            self.assertEqual(empty['rows'],0)
            self.assertEqual(empty['header_mask_definitions'],0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA full discovery requires a GPU')
    def test_cuda_serial_preserves_complete_pipeline_files_and_resume(self):
        self._pipeline_equivalence('cuda')

    def _pipeline_equivalence(self, device):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            n = 1200
            ids = [('family_A', f'sample_{i}') for i in range(n)]
            values = np.zeros((n, 3), dtype=np.float32)
            values[:13, 0] = 1
            values[13:28, 1] = 1
            values[28, 2] = 1
            values[50, :] = np.nan
            # The phenotype and LOCO samples are independently reordered.
            trait = np.sin(np.arange(n)*.17)
            trait[-1] = np.nan
            phenotypes = root/'phenotype.txt'
            phenotypes.write_text('FID IID qt\n'+''.join(
                f'{ids[i][0]} {ids[i][1]} {trait[i]}\n' for i in reversed(range(n))))
            keep = root/'keep.txt'
            keep.write_text(''.join(f'{fid} {iid}\n' for fid,iid in ids))
            null = NullModel(ids[:-1], torch.zeros((n-1, 2)), chromosomes=(21, 22))
            _, predictions = null.export_regenie(root/'null', phenotype_name='qt')
            prefixes, analyses = {}, {}
            for chromosome in (21, 22):
                prefix = root/f'c{chromosome}'
                variants = [Variant(j, str(chromosome), f'{chromosome}:v{j}', j+1, 'G', 'A') for j in range(3)]
                write_bed(prefix, values, variants, ids)
                annotation, setlist, mask = (root/f'{chromosome}.{suffix}' for suffix in ('anno', 'sets', 'masks'))
                annotation.write_text(''.join(f'{v.id} G A\n' for v in variants))
                setlist.write_text(f'G {chromosome} 1 '+','.join(v.id for v in variants)+'\n')
                mask.write_text('Mask1 A\n')
                prefixes[str(chromosome)] = str(prefix)
                analyses[str(chromosome)] = [GeneAnalysis(name, str(annotation), str(setlist), str(mask))
                                              for name in ('Coding', 'Noncoding')]
            inputs = DiscoveryInputs(prefixes['21'], str(phenotypes), 'qt', str(keep), prefixes,
                                     gene_analyses=analyses, imported_loco=str(predictions))
            results = []
            for level in ('serial',):
                config = WGSConfig.paper(apply_rint=False)
                config.step1 = replace(config.step1, device=device)
                config.single_variant.device = device
                config.single_variant.genotype_reader = 'cuda_packed' if device=='cuda' else 'cpu'
                config.gene_based.genotype_reader = 'cuda_packed' if device=='cuda' else 'cpu'
                config.single_variant.dtype = 'float64'
                config.single_variant.min_mac = 1
                config.gene_based.include_domains = False
                config.gene_based.sbat_qmc_samples = 128
                config.execution = ExecutionConfig('serial', 1)
                config.write_masks = True
                out = root/level
                report = run_discovery(inputs, config=config, output_dir=out, resume=False)
                artifacts = {str(p.relative_to(out)):p.read_bytes() for folder in ('Single', 'Gene')
                             for p in (out/folder).iterdir() if p.suffix in ('.regenie', '.ids', '.bed', '.bim', '.fam', '.snplist')}
                self.assertEqual(len(report['output_files']['gene']), 4)
                self.assertEqual(len(report['output_files']['single']), 2)
                self.assertTrue(report['stages']['step1']['imported'])
                single_stage=report['stages']['single_c21']
                self.assertEqual(single_stage['source_variants'],3)
                self.assertEqual(single_stage['variants_scanned'],3)
                self.assertEqual(single_stage['tests'],{'ADD':single_stage['rows']})
                single_log=(out/'Single/discovery_c21.log').read_text()
                gene_log=(out/'Gene/discovery_c21_Coding.log').read_text()
                step1_log=(out/'Step1/discovery.log').read_text()
                for text in (single_log,gene_log,step1_log,(out/'discovery.log').read_text()):
                    self.assertTrue(text.startswith('REGENIE-compatible analysis log\nEngine: torchwgs (PyTorch '))
                    self.assertIn('Options in effect (REGENIE-equivalent):',text)
                    self.assertIn('Elapsed time : ',text)
                    self.assertFalse(any(line.startswith('{') for line in text.splitlines()))
                self.assertIn('# variants scanned to EOF: 3',single_log)
                self.assertIn('# masks written:',gene_log)
                self.assertIn('mode: imported',step1_log)
                self.assertIn('no fitting performed',step1_log)
                self.assertNotIn('fit_seconds:',step1_log)
                self.assertFalse((out/'Step1/discovery_1.loco').exists())
                self.assertFalse((out/'Step1/discovery_pred.list').exists())
                self.assertEqual(set(report['output_files']['step1']),
                                 {str(predictions),str(root/'null_1.loco'),str(out/'Step1/discovery.log')})
                from torchwgs.pipeline import _identities_match
                step1_manifest=json.loads((out/'Step1/cache.json').read_text())
                self.assertEqual(step1_manifest['mode'],'imported')
                self.assertTrue(_identities_match(step1_manifest['artifacts']))
                events=[json.loads(line) for line in (out/'discovery.events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(e.get('stage')=='chromosome_completed' for e in events),2)
                if device=='cuda':
                    self.assertIn(f'cuda:{torch.cuda.current_device()}',report['peak_gpu_by_device_bytes'])
                    self.assertEqual(report['peak_gpu_bytes'],max(report['peak_gpu_by_device_bytes'].values()))
                else:
                    self.assertNotIn('peak_gpu_bytes',report)
                self.assertTrue(all(p.read_text().startswith('qt\tNA\n') for p in out.glob('Single/*.ids')))
                cached = run_discovery(inputs, config=config, output_dir=out, resume=True)
                self.assertTrue(all(v.get('cached', False) for k,v in cached['stages'].items()
                                    if k.startswith(('single_', 'gene_'))),
                                {'level':level, 'stages':cached['stages']})
                events=[json.loads(line) for line in (out/'discovery.events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(e.get('stage')=='chromosome_cached' for e in events),2)
                if level=='serial':
                    # A result is complete only with its native companion log.
                    for missing_log in (out/'Single/discovery_c21.log',out/'Gene/discovery_c21_Coding.log'):
                        missing_log.unlink()
                    restored=run_discovery(inputs,config=config,output_dir=out,resume=True)
                    self.assertFalse(restored['stages']['single_c21'].get('cached',False))
                    self.assertFalse(restored['stages']['gene_c21_Coding'].get('cached',False))
                    self.assertTrue(restored['stages']['gene_c21_Noncoding']['cached'])
                    self.assertTrue(restored['stages']['single_c22']['cached'])
                    self.assertTrue((out/'Single/discovery_c21.log').is_file())
                    self.assertTrue((out/'Gene/discovery_c21_Coding.log').is_file())
                    from torchwgs import pipeline as pipeline_module
                    actual_identity=pipeline_module._implementation_identity
                    def changed_pipeline_identity(names):
                        identities=actual_identity(names)
                        if 'pipeline' in identities:identities['pipeline']='simulated-new-pipeline-source'
                        return identities
                    with patch.object(pipeline_module,'_implementation_identity',changed_pipeline_identity):
                        changed=run_discovery(inputs,config=config,output_dir=out,resume=True)
                    self.assertTrue(all(not stage.get('cached',False) for name,stage in changed['stages'].items()
                                        if name.startswith(('single_','gene_'))))
                results.append(artifacts)
            self.assertEqual(len(results), 1)
            if device=='cuda':
                # Exercise a small fit and repeated cache loads in the same
                # orchestration fixture. This checks log updates do not invalidate
                # their own manifest or pretend that cache loading re-fits ridge.
                fitting_inputs=replace(inputs,wgs_prefixes={},gene_analyses={},imported_loco=None)
                fitting_out=root/'fitted_step1'
                fitted=run_discovery(fitting_inputs,config=config,output_dir=fitting_out,resume=False)
                self.assertEqual(fitted['stages']['step1']['mode'],'fitted')
                fitted_log=(fitting_out/'Step1/discovery.log').read_text()
                self.assertIn('mode: fitted',fitted_log)
                self.assertIn('fit_seconds:',fitted_log)
                self.assertIn('# input variants: 3',fitted_log)
                local_files={str(fitting_out/'Step1'/filename) for filename in
                             ('null_model.pt','null_model.json','discovery_1.loco',
                              'discovery_pred.list','discovery.log')}
                self.assertEqual(set(fitted['output_files']['step1']),local_files)
                from torchwgs.pipeline import _file_identity
                prediction_identities=[_file_identity(path) for path in local_files
                                       if not path.endswith('/discovery.log')]
                for _ in range(2):
                    cached=run_discovery(fitting_inputs,config=config,output_dir=fitting_out,resume=True)
                    self.assertTrue(cached['stages']['step1']['cached'])
                    self.assertEqual(cached['stages']['step1']['mode'],'cached')
                    cached_log=(fitting_out/'Step1/discovery.log').read_text()
                    self.assertIn('mode: cached',cached_log)
                    self.assertIn('original_fit_seconds:',cached_log)
                    self.assertIn('no fitting performed',cached_log)
                    manifest=json.loads((fitting_out/'Step1/cache.json').read_text())
                    self.assertEqual(manifest['mode'],'cached')
                    self.assertTrue(_identities_match(manifest['artifacts']))
                    self.assertTrue(_identities_match(prediction_identities))
                for companion in ('discovery_1.loco','discovery_pred.list','discovery.log'):
                    (fitting_out/'Step1'/companion).unlink()
                    repaired=run_discovery(fitting_inputs,config=config,output_dir=fitting_out,resume=True)
                    self.assertEqual(repaired['stages']['step1']['mode'],'fitted')
                    self.assertTrue((fitting_out/'Step1'/companion).is_file())


if __name__ == '__main__':
    unittest.main()
