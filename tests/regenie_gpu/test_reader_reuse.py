"""CPU checks of reader ownership and unchanged per-group output plumbing.

The numerical association iterator is replaced by a deterministic fixture;
these are correctness guards, not GPU accuracy or performance benchmarks.
"""
import builtins
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from torchwgs import chromosome
from torchwgs.config import WGSConfig
from torchwgs.io import BedReader, Variant, write_bed
from torchwgs.pipeline import DiscoveryInputs, GeneAnalysis


class ReaderReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prefix = self.root / 'source'
        self.ids = [('family', f'sample{i}') for i in range(6)]
        self.variants = [Variant(i, '21', f'v{i}', 10+i, 'A', 'G') for i in range(4)]
        self.values = torch.tensor([[0., 1., 2., 0.], [1., 2., 0., 1.],
                                    [2., 0., 1., 2.], [0., 2., 1., 0.],
                                    [1., 0., 2., 1.], [2., 1., 0., 2.]])
        write_bed(self.prefix, self.values, self.variants, self.ids)
        fam = Path(str(self.prefix)+'.fam')
        fam.write_text(''.join(f'{fid} {iid} 0 0 {1+i%2} -9\n'
                               for i, (fid, iid) in enumerate(self.ids)))
        self.keep = self.root / 'keep'
        self.keep.write_text('\n'.join(f'{fid} {iid}' for fid, iid in self.ids[1:])+'\n')
        self.remove = self.root / 'remove'
        self.remove.write_text('family sample2\n')
        pheno = self.root / 'phenotype'
        pheno.write_text('FID IID continuous_trait\n')
        self.inputs = DiscoveryInputs(str(self.prefix), str(pheno), 'continuous_trait',
                                      str(self.keep), {'21': str(self.prefix)},
                                      sample_remove=str(self.remove))
        self.config = WGSConfig.paper(apply_rint=False)
        # No execution helper or CUDA association code is invoked in this file.
        self.config.single_variant.device = 'cpu'
        self.config.gene_based.extract_variants = frozenset({'v0', 'v2', 'v3', 'absent'})
        self.full = self._analysis('Full',
            'v3 modeled C\nv0 modeled A\nv2 outside_set B\nabsent modeled C\nv1 modeled A\n',
            'A A\nB B\nCombined C,B,A\nNull NULL\n', 'modeled 21 10 v0,v3\n')
        whitelist = self.root / 'sub.score'
        whitelist.write_text('v1 1\nv2 2\n')
        self.sub = self._analysis('Sub',
            'v0 modeled A2\nv2 modeled B2\nv3 modeled C2\nv1 modeled B2\n',
            'A A2\nB B2\nCombined C2,B2,A2\nNull NULL\n',
            'modeled 21 10 v2\n', str(whitelist))
        reader = chromosome._chromosome_reader(self.prefix, self.inputs)
        self.context = SimpleNamespace(y=torch.tensor([.2, -.5]),
            sample_indices=torch.tensor([3, 0]),
            sample_ids=[reader.sample_ids[i] for i in [3, 0]])

    def _analysis(self, name, annotations, masks, setlist, whitelist=None):
        paths = []
        for suffix, text in [('anno', annotations), ('mask', masks), ('set', setlist)]:
            path = self.root / (name+'.'+suffix)
            path.write_text(text)
            paths.append(str(path))
        return GeneAnalysis(name, paths[0], paths[2], paths[1], whitelist)

    def _iterator(self, records):
        def iterator(reader, context, annotations, setlist, definitions, config,
                     *, artifact_callback, variant_lookup):
            records.append(dict(reader=reader, annotations=list(annotations),
                lookup=list(variant_lookup), extract=config.extract_variants,
                definitions=list(definitions)))
            for identifier, variant in variant_lookup.items():
                burden = reader.read_variants([variant.index])[context.sample_indices, 0]
                if artifact_callback is not None:
                    mask = SimpleNamespace(name='Fixture', base_name='Fixture',
                        aaf_upper=1., frequency='all', variant_ids=(identifier,),
                        burden=burden)
                    artifact_callback(SimpleNamespace(
                        gene=SimpleNamespace(gene=identifier, chrom=variant.chrom,
                                             position=variant.position), masks=[mask]))
                yield dict(CHROM=variant.chrom, GENPOS=variant.position,
                    ID=identifier+'.Fixture.all', ALLELE0='ref', ALLELE1='Fixture.all',
                    A1FREQ=float(burden.mean()/2), N=len(burden), TEST='ADD',
                    BETA=float(burden.sum()), SE=.25, CHISQ=1.23456789,
                    LOG10P=.123456789, EXTRA='NA')
        return iterator

    def _job(self, analysis, destination, reader=None):
        return chromosome._gene_job(analysis, prefix=str(self.prefix), context=self.context,
            inputs=self.inputs, config=self.config, destination=destination,
            key=('21', 'fixture-stage'), reader=reader)

    def test_shared_and_fresh_readers_write_identical_results_ids_and_all_mask_files(self):
        shared = chromosome._chromosome_reader(self.prefix, self.inputs)
        records = []
        with patch.object(chromosome, 'test_gene_based', side_effect=self._iterator(records)):
            for analysis in (self.full, self.sub):
                self._job(analysis, self.root/'fresh')
            with patch.object(chromosome, 'BedReader', side_effect=AssertionError('reopened BED')):
                for analysis in (self.full, self.sub):
                    self._job(analysis, self.root/'shared', shared)
        for analysis in (self.full, self.sub):
            stem = f'discovery_c21_{analysis.name}'
            for suffix in ('_continuous_trait.regenie', '_continuous_trait.regenie.ids',
                           '_masks.bed', '_masks.bim', '_masks.fam', '_masks.snplist'):
                self.assertEqual((self.root/'fresh/Gene'/(stem+suffix)).read_bytes(),
                                 (self.root/'shared/Gene'/(stem+suffix)).read_bytes())
        self.assertIs(records[2]['reader'], shared)
        self.assertIs(records[3]['reader'], shared)
        self.assertEqual(records[2]['lookup'], ['v0', 'v2', 'v3'])
        self.assertEqual(records[3]['lookup'], ['v2'])
        self.assertEqual(len(records[2]['annotations']), 5)
        self.assertEqual(len(records[3]['annotations']), 4)
        self.assertEqual(records[3]['extract'], frozenset({'v2'}))
        self.assertEqual(self.config.gene_based.extract_variants,
                         frozenset({'v0', 'v2', 'v3', 'absent'}))
        full_header = (self.root/'shared/Gene/discovery_c21_Full_continuous_trait.regenie').read_text().splitlines()[0]
        sub_header = (self.root/'shared/Gene/discovery_c21_Sub_continuous_trait.regenie').read_text().splitlines()[0]
        # Category B is registered despite its annotation gene being outside setlist.
        self.assertEqual(full_header, '##MASKS=<A="A";B="B";Combined="C,B,A";Null="NULL">')
        self.assertEqual(sub_header, '##MASKS=<B="B2";Combined="B2";Null="NULL">')

    def _run(self, analyses, *, resume=False, single=True):
        self.inputs.gene_analyses = {'21': analyses}
        null = SimpleNamespace(sample_ids=self.ids, chromosomes=[21],
            align=lambda ids: torch.zeros((len(ids), 1)))
        lookup = dict(zip(self.ids, [.1, .2, .3, .4, .5, .6]))
        return chromosome.run_chromosome(('21', self.prefix), inputs=self.inputs,
            config=self.config, destination=self.root/'pipeline', null=null, null_key='fixture',
            y_lookup=lookup, x_lookup=None, resume=resume, run_single=single, run_gene=True)

    def test_complete_26_group_serial_loop_and_single_stage_share_one_reader(self):
        analyses = [GeneAnalysis(f'Group{i:02}', self.full.annotation_file,
            self.full.setlist_file, self.full.mask_definition_file) for i in range(26)]
        readers, order = [], []
        def job(analysis, **kwargs):
            readers.append(kwargs['reader'])
            order.append(analysis.name)
            chromosome._validate_gene_reader(kwargs['reader'], kwargs['prefix'],
                                              kwargs['inputs'], kwargs['context'])
            return analysis.name, analysis.name+'.regenie', {'rows':1, 'numerics':{}}
        def single(reader, context, *, config):
            readers.append(reader)
            return iter(())
        with patch.object(chromosome, 'BedReader', wraps=BedReader) as constructor, \
             patch.object(chromosome, 'create_test_context', return_value=self.context), \
             patch.object(chromosome, '_gene_job', side_effect=job), \
             patch.object(chromosome, 'iter_single_variant_results', side_effect=single):
            result = self._run(analyses)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(len(readers), 27)
        self.assertTrue(all(reader is readers[0] for reader in readers))
        self.assertEqual(order, [analysis.name for analysis in analyses])
        self.assertEqual(result['gene_files'], [analysis.name+'.regenie' for analysis in analyses])
        self.assertEqual(readers[0].sample_ids, [self.ids[i] for i in (1,3,4,5)])

    def test_independent_optional_rint_context_still_uses_same_reader(self):
        self.config.gene_based.apply_rint = True
        contexts = [self.context, SimpleNamespace(**vars(self.context))]
        with patch.object(chromosome, 'create_test_context', side_effect=contexts) as factory, \
             patch.object(chromosome, '_gene_job', return_value=('Full', 'result.regenie',
                                                                {'rows':0, 'numerics':{}})) as job:
            self._run([self.full], single=False)
        self.assertEqual([call.kwargs['apply_rint'] for call in factory.call_args_list], [False, True])
        self.assertIs(job.call_args.kwargs['context'], contexts[1])
        self.assertEqual(job.call_args.kwargs['reader'].sample_ids, [self.ids[i] for i in (1,3,4,5)])

    def test_mixed_resume_keeps_group_order_and_only_pending_groups_use_reader(self):
        def cache(prefix, key):
            if str(prefix).endswith('discovery_c21_Full'):
                return {'result':'cached-full.regenie', 'report':{'rows':1, 'numerics':{}}}
            return None
        with patch('torchwgs.pipeline._stage_cache', side_effect=cache), \
             patch.object(chromosome, 'create_test_context', return_value=self.context), \
             patch.object(chromosome, '_gene_job', return_value=('Sub', 'fresh-sub.regenie',
                                                                {'rows':0, 'numerics':{}})) as job:
            result = self._run([self.full, self.sub], resume=True, single=False)
        self.assertEqual(job.call_count, 1)
        self.assertEqual(job.call_args.args[0].name, 'Sub')
        self.assertEqual(result['gene_files'], ['cached-full.regenie', 'fresh-sub.regenie'])

    def test_fully_resumed_chromosome_does_not_open_a_reader(self):
        with patch('torchwgs.pipeline._stage_cache', return_value={
                    'result':'cached.regenie', 'report':{'rows':1, 'numerics':{}}}), \
             patch.object(chromosome, 'BedReader', side_effect=AssertionError('unused reader')):
            result = self._run([self.full, self.sub], resume=True)
        self.assertTrue(result['cached'])

    def test_other_prefix_and_same_count_different_filters_are_rejected(self):
        reader = chromosome._chromosome_reader(self.prefix, self.inputs)
        other = self.root/'other_chromosome'
        write_bed(other, self.values, self.variants, self.ids)
        with self.assertRaisesRegex(ValueError, 'Shared gene reader'):
            chromosome._validate_gene_reader(reader, other, self.inputs, self.context)
        replacement = self.root/'replacement_keep'
        replacement.write_text('\n'.join(f'{fid} {iid}' for fid, iid in self.ids[:5])+'\n')
        self.inputs.discovery_samples = str(replacement)
        self.assertEqual(BedReader(self.prefix, keep=replacement, remove=self.remove).n_samples,
                         reader.n_samples)
        with self.assertRaisesRegex(ValueError, 'Shared gene reader'):
            chromosome._validate_gene_reader(reader, self.prefix, self.inputs, self.context)

    def test_changed_filter_content_or_fam_or_bed_cannot_reuse_mapping(self):
        for path in (self.keep, self.remove, Path(str(self.prefix)+'.fam'), Path(str(self.prefix)+'.bed')):
            with self.subTest(suffix=path.name):
                reader = chromosome._chromosome_reader(self.prefix, self.inputs)
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
                with self.assertRaisesRegex(ValueError, 'Shared gene reader'):
                    chromosome._validate_gene_reader(reader, self.prefix, self.inputs, self.context)

    def test_context_subset_order_sex_and_invalid_identity_are_checked(self):
        reader = chromosome._chromosome_reader(self.prefix, self.inputs)
        self.assertEqual(chromosome._validate_gene_reader(reader, self.prefix, self.inputs,
                                                         self.context), [2,2])
        invalid = [SimpleNamespace(**{**vars(self.context), 'sample_ids':self.context.sample_ids[::-1]}),
                   SimpleNamespace(**{**vars(self.context), 'sample_indices':torch.tensor([0])}),
                   SimpleNamespace(**{**vars(self.context), 'sample_indices':torch.tensor([3,4])})]
        for context in invalid:
            with self.subTest(context=context.sample_indices.tolist()):
                with self.assertRaisesRegex(ValueError, 'sample order'):
                    chromosome._validate_gene_reader(reader, self.prefix, self.inputs, context)

    def test_changed_sample_source_during_construction_is_rejected(self):
        def construct(*args, **kwargs):
            reader = BedReader(*args, **kwargs)
            with self.remove.open('a') as stream:
                stream.write('# changed\n')
            return reader
        with patch.object(chromosome, 'BedReader', side_effect=construct):
            with self.assertRaisesRegex(RuntimeError, 'changed while creating'):
                chromosome._chromosome_reader(self.prefix, self.inputs)

    def test_reused_metadata_invalidates_on_bim_change_and_rechecks_duplicates(self):
        reader = chromosome._chromosome_reader(self.prefix, self.inputs)
        self.assertEqual(list(reader.find_variants({'v3', 'v0', 'absent'})), ['v0','v3'])
        original_open = builtins.open
        def no_bim_rescan(path, *args, **kwargs):
            if str(path) == str(self.prefix)+'.bim':
                raise AssertionError('Repeated metadata query reopened BIM')
            return original_open(path, *args, **kwargs)
        with patch('builtins.open', side_effect=no_bim_rescan):
            self.assertEqual(list(reader.find_variants({'absent', 'v0', 'v3'})), ['v0','v3'])
        bim = Path(str(self.prefix)+'.bim')
        def replace_with_changed_identity(contents):
            # A default temporary filesystem can assign identical timestamps
            # to rapid same-size writes. This guard tests changed-stat inputs;
            # make that contract deterministic without manually clearing cache.
            previous = bim.stat()
            bim.write_text(contents)
            os.utime(bim, ns=(previous.st_atime_ns, previous.st_mtime_ns+2_000_000_000))
            current = bim.stat()
            identity = (current.st_dev, current.st_ino, current.st_size,
                        current.st_mtime_ns, current.st_ctime_ns)
            self.assertNotEqual(identity, reader._bim_identity)
        text = bim.read_text().replace('v3', 'renamed')
        replace_with_changed_identity(text)
        self.assertEqual(list(reader.find_variants({'v0', 'v3', 'renamed'})), ['v0','renamed'])
        replace_with_changed_identity(text.replace('v1', 'v0'))
        with self.assertRaisesRegex(ValueError, 'Duplicate requested variant ID'):
            reader.find_variants({'v0'})

    def test_reused_reader_retains_only_bounded_metadata_and_sample_map_caches(self):
        reader = chromosome._chromosome_reader(self.prefix, self.inputs)
        reader.metadata_cache_size = 2
        for i in range(4):
            reader.find_variants({f'v{i}', f'absent{i}'})
            self.assertLessEqual(len(reader._variant_metadata), 2)
        rows = self.context.sample_indices.tolist()
        first = reader.read_packed_block([0], sample_rows=rows, device='cpu', dtype=torch.float64)
        second = reader.read_packed_block([2,3], sample_rows=rows, device='cpu', dtype=torch.float64)
        self.assertIs(first.sample_bytes, second.sample_bytes)
        self.assertIs(first.sample_shifts, second.sample_shifts)
        self.assertIs(first.lookup, second.lookup)
        self.assertEqual(first.sample_bytes.tolist(), [1,0])
        self.assertEqual(first.sample_shifts.tolist(), [2,2])
        for selected in ([0], [1], [2], [3], [0,1], [1,0]):
            reader.read_packed_block([0], sample_rows=selected, device='cpu')
            self.assertLessEqual(len(reader._packed_decoder_cache), 4)


if __name__ == '__main__':
    unittest.main()
