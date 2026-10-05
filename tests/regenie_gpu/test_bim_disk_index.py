"""Independent scan/index CPU oracles and private cache lifecycle guards.

These small fixtures test metadata and output plumbing. They do not measure
whole-chromosome speed or replace a real GPU association benchmark.
"""
import builtins
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from torchwgs import chromosome
from torchwgs.config import WGSConfig
from torchwgs.io import BedReader, Variant, write_bed
from torchwgs.pipeline import DiscoveryInputs, GeneAnalysis


class BimDiskIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ids = [('family', f'sample{i}') for i in range(6)]
        self.prefix = self._source('source')
        self.index = self.root/'private cache'/'metadata.sqlite'

    def _source(self, name, *, lines=None, count=12):
        prefix = self.root/name
        variants = [Variant(i, '21', f'v{i}', 10+i, 'A', 'G') for i in range(count)]
        values = torch.arange(len(self.ids)*count, dtype=torch.float32).reshape(len(self.ids), count)%3
        values[0, 0] = float('nan')
        write_bed(prefix, values, variants, self.ids)
        if lines is not None:
            Path(str(prefix)+'.bim').write_text(''.join(line+'\n' for line in lines))
        return prefix

    def _reader(self, *, prefix=None, index=True, cache=0, query=3, **kwargs):
        return BedReader(prefix or self.prefix, metadata_cache_size=cache,
            bim_index_path=self.index if index else None, bim_index_query_size=query, **kwargs)

    def _assert_oracle(self, identifiers, *, prefix=None, query=3):
        scan = self._reader(prefix=prefix, index=False)
        indexed = self._reader(prefix=prefix, query=query)
        def outcome(reader):
            try:
                return ('values', list(reader.find_variants(identifiers).items()))
            except (ValueError, RuntimeError) as failure:
                return ('error', type(failure), str(failure))
        expected = outcome(scan)
        self.assertEqual(outcome(indexed), expected)
        return expected

    def _deny_bim_open(self, path, *args, **kwargs):
        if str(path) == str(self.prefix)+'.bim':
            raise AssertionError('Committed disk index reread BIM')
        return self.original_open(path, *args, **kwargs)

    def test_all_metadata_order_and_selected_genotypes_match_scan(self):
        scan = self._reader(index=False)
        indexed = self._reader(keep={self.ids[i] for i in (1,3,4,5)},
                               sample_ids=[self.ids[i] for i in (5,1,3)])
        requested = ['v10', 'v0', 'absent', 'v5', 'v0', 'v1', 'v9', 'v3']
        self.assertEqual(list(indexed.find_variants(requested).items()),
                         list(scan.find_variants(requested).items()))
        result = indexed.find_variants(requested)
        actual = indexed.read_variants([variant.index for variant in result.values()])
        expected = scan.read_variants([variant.index for variant in result.values()])[[5,1,3]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(list(result), ['v0', 'v1', 'v3', 'v5', 'v9', 'v10'])

    def test_private_index_build_and_persisted_reuse_never_rescan_bim(self):
        reader = self._reader()
        first = reader.prepare_bim_index()
        self.assertEqual({key:value for key,value in first.items() if key != 'seconds'},
                         dict(enabled=True, built=True, rows=12, indexed_rows=12, valid_bim=True))
        self.assertGreaterEqual(first['seconds'], 0.)
        self.assertEqual(self.index.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.index.parent.stat().st_mode & 0o777, 0o700)
        self.original_open = builtins.open
        with patch('builtins.open', side_effect=self._deny_bim_open):
            self.assertFalse(reader.prepare_bim_index()['built'])
            self.assertEqual(list(reader.find_variants({'v11', 'absent'})), ['v11'])
            other = self._reader()
            self.assertFalse(other.prepare_bim_index()['built'])
            self.assertEqual(list(other.find_variants({'v8', 'v2'})), ['v2', 'v8'])
        self.assertEqual(list(self.index.parent.glob('*.partial')), [])

    def test_sql_query_chunks_and_metadata_cache_are_bounded(self):
        prefix = self._source('many', count=23)
        reader = self._reader(prefix=prefix, cache=2, query=3)
        reader.prepare_bim_index()
        real_connection = reader._bim_index_connection
        lengths, fetch_sizes = [], []
        class Cursor:
            def __init__(self, cursor): self.cursor = cursor
            def fetchmany(self, size):
                fetch_sizes.append(size)
                return self.cursor.fetchmany(size)
            def close(self): self.cursor.close()
        class Connection:
            def __init__(self): self.connection = real_connection()
            def execute(self, statement, parameters=()):
                cursor = self.connection.execute(statement, parameters)
                if 'identifier IN' in statement:
                    lengths.append(len(parameters))
                    return Cursor(cursor)
                return cursor
            def close(self): self.connection.close()
        identifiers = {f'v{i}' for i in range(23)} | {'absent'}
        with patch.object(reader, '_bim_index_connection', side_effect=Connection):
            actual = reader.find_variants(identifiers)
        self.assertEqual(len(actual), 23)
        self.assertEqual(len(lengths), math.ceil(len(identifiers)/3))
        self.assertTrue(all(1 <= length <= 3 for length in lengths))
        self.assertTrue(all(size == 3 for size in fetch_sizes))
        self.assertLessEqual(len(reader._variant_metadata), 2)
        for identifier in ('v0', 'v2', 'absent2', 'v4', 'absent3'):
            reader.find_variants({identifier})
            self.assertLessEqual(len(reader._variant_metadata), 2)

    def test_unrequested_duplicates_and_bad_positions_do_not_change_lookup(self):
        prefix = self._source('unrequested', lines=[
            'chr21 chosen 0 18 T C', '21 ignored 0 broken G A',
            '21 duplicate 0 20 A C', '21 duplicate 0 22 C T'])
        self._assert_oracle({'chosen', 'absent'}, prefix=prefix)
        self._assert_oracle({'duplicate'}, prefix=prefix)
        self._assert_oracle({'ignored'}, prefix=prefix)

    def test_duplicate_request_error_precedes_later_malformed_row(self):
        prefix = self._source('duplicate-first', lines=[
            '21 duplicate 0 1 A G', '21 duplicate 0 2 G A', 'bad row'])
        expected = self._assert_oracle({'duplicate'}, prefix=prefix, query=1)
        self.assertIn('Duplicate requested variant ID', expected[2])

    def test_unrequested_malformed_row_still_fails_before_later_duplicate(self):
        prefix = self._source('malformed-first', lines=[
            '21 chosen 0 1 A G', 'bad row', '21 duplicate 0 2 G A', '21 duplicate 0 3 A G'])
        for identifiers in ({'chosen'}, {'duplicate'}, {'absent'}):
            expected = self._assert_oracle(identifiers, prefix=prefix, query=1)
            self.assertEqual(expected[2], 'Malformed BIM row 2')

    def test_invalid_requested_position_retains_scan_error_order_across_sql_chunks(self):
        prefix = self._source('invalid-first', lines=[
            '21 earliest 0 not-an-int A G', '21 duplicate 0 2 G A',
            '21 duplicate 0 3 A G', 'wrong'])
        expected = self._assert_oracle({'earliest', 'duplicate', 'absent'}, prefix=prefix, query=1)
        self.assertIn('invalid literal for int()', expected[2])
        prefix = self._source('duplicate-before-position', lines=[
            '21 earliest 0 1 A G', '21 earliest 0 broken G A', '21 later 0 broken A G'])
        expected = self._assert_oracle({'earliest', 'later'}, prefix=prefix, query=1)
        self.assertIn('Duplicate requested variant ID', expected[2])

    def test_empty_request_neither_builds_nor_changes_malformed_scan_semantics(self):
        prefix = self._source('malformed', lines=['bad'])
        self.assertEqual(self._reader(prefix=prefix).find_variants([]), {})
        self.assertFalse(self.index.exists())
        prepared = self._reader(prefix=prefix).prepare_bim_index()
        self.assertFalse(prepared['valid_bim'])
        self.assertEqual(prepared['indexed_rows'], 0)
        self.assertEqual(self._reader(prefix=prefix).find_variants([]), {})
        self._assert_oracle({'absent'}, prefix=prefix)

    def test_non_string_id_does_not_match_sqlite_text_affinity(self):
        prefix = self._source('numeric-id', lines=['21 12 0 1 A G'])
        self.assertEqual(self._assert_oracle({12}, prefix=prefix)[1], [])
        self._assert_oracle({'12'}, prefix=prefix)

    def test_source_mutation_invalidates_positive_and_negative_cache(self):
        reader = self._reader(cache=10)
        self.assertEqual(list(reader.find_variants({'v0', 'new'})), ['v0'])
        bim = Path(str(self.prefix)+'.bim')
        text = bim.read_text().replace('v0', 'new')
        bim.write_text(text)
        self.assertEqual(list(reader.find_variants({'v0', 'new'})), ['new'])
        bim.write_text(text.replace('v1', 'new'))
        with self.assertRaisesRegex(ValueError, 'Duplicate requested variant ID'):
            reader.find_variants({'new'})

    def test_same_size_same_mtime_change_rebuilds_using_ctime_identity(self):
        reader = self._reader(cache=10)
        reader.prepare_bim_index()
        bim = Path(str(self.prefix)+'.bim')
        before = bim.stat()
        bim.write_text(bim.read_text().replace('v0', 'x0'))
        os.utime(bim, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(bim.stat().st_size, before.st_size)
        self.assertNotEqual(bim.stat().st_ctime_ns, before.st_ctime_ns)
        self.assertTrue(reader.prepare_bim_index()['built'])
        self.assertEqual(list(reader.find_variants({'v0', 'x0'})), ['x0'])

    def test_replaced_source_inode_and_different_prefix_rebuild(self):
        reader = self._reader()
        reader.prepare_bim_index()
        bim = Path(str(self.prefix)+'.bim')
        replacement = self.root/'replacement.bim'
        replacement.write_text(bim.read_text().replace('v0', 'renamed'))
        os.replace(replacement, bim)
        self.assertTrue(reader.prepare_bim_index()['built'])
        self.assertEqual(list(reader.find_variants({'v0', 'renamed'})), ['renamed'])
        other = self._source('other', lines=['22 other-id 0 5 G C'])
        other_reader = self._reader(prefix=other)
        self.assertTrue(other_reader.prepare_bim_index()['built'])
        self.assertEqual(list(other_reader.find_variants({'other-id', 'renamed'})), ['other-id'])
        self.assertTrue(reader.prepare_bim_index()['built'])

    def test_corrupt_wrong_schema_and_incomplete_indexes_rebuild(self):
        reader = self._reader()
        reader.prepare_bim_index()
        self.index.write_bytes(b'not a sqlite database')
        self.assertTrue(reader.prepare_bim_index()['built'])
        with sqlite3.connect(self.index) as connection:
            connection.execute('PRAGMA user_version=123')
        self.assertTrue(reader.prepare_bim_index()['built'])
        with sqlite3.connect(self.index) as connection:
            connection.execute("UPDATE metadata SET value='0' WHERE name='complete'")
        self.assertTrue(reader.prepare_bim_index()['built'])
        self.assertEqual(list(reader.find_variants({'v4'})), ['v4'])

    def test_inplace_corruption_rebuilds_even_when_database_stat_is_unchanged(self):
        reader = self._reader()
        reader.prepare_bim_index()
        original_stat = Path.stat
        saved_stat = self.index.stat()
        def fixed_database_stat(path, *args, **kwargs):
            if path == self.index:
                return saved_stat
            return original_stat(path, *args, **kwargs)
        updates = ['PRAGMA user_version=123', 'PRAGMA application_id=0',
                   "UPDATE metadata SET value='0' WHERE name='complete'",
                   "UPDATE metadata SET value='wrong-source' WHERE name='source'"]
        with patch.object(Path, 'stat', new=fixed_database_stat):
            for statement in updates:
                with self.subTest(statement=statement):
                    with sqlite3.connect(self.index) as connection:
                        connection.execute(statement)
                    self.assertEqual(self.index.stat(), saved_stat)
                    self.assertTrue(reader.prepare_bim_index()['built'])
                    self.assertEqual(list(reader.find_variants({'v4'})), ['v4'])
            self.index.write_bytes(b'corrupt sqlite content')
            self.assertEqual(self.index.stat(), saved_stat)
            self.assertTrue(reader.prepare_bim_index()['built'])
            self.assertEqual(list(reader.find_variants({'v5'})), ['v5'])

    def test_new_request_validates_index_metadata_despite_unchanged_database_stat(self):
        reader = self._reader(cache=10)
        self.assertEqual(list(reader.find_variants({'v0'})), ['v0'])
        original_stat = Path.stat
        saved_stat = self.index.stat()
        def fixed_database_stat(path, *args, **kwargs):
            return saved_stat if path == self.index else original_stat(path, *args, **kwargs)
        with sqlite3.connect(self.index) as connection:
            connection.execute('PRAGMA user_version=123')
        with patch.object(Path, 'stat', new=fixed_database_stat), \
             patch.object(reader, '_build_bim_index', wraps=reader._build_bim_index) as build:
            self.assertEqual(list(reader.find_variants({'v0', 'v8'})), ['v0', 'v8'])
            self.assertEqual(build.call_count, 1)

    def test_atomic_replacement_between_prepare_and_query_rechecks_opened_schema(self):
        reader = self._reader(cache=10)
        reader.prepare_bim_index()
        original_connection = reader._bim_index_connection
        for statement in ('PRAGMA user_version=123', 'PRAGMA application_id=0'):
            with self.subTest(statement=statement):
                replacement = self.root/'replacement.sqlite'
                replacement.write_bytes(self.index.read_bytes())
                connection = sqlite3.connect(replacement)
                try:
                    connection.execute(statement)
                    connection.commit()
                finally:
                    connection.close()
                opens = 0
                def replace_before_query_open():
                    nonlocal opens
                    opens += 1
                    if opens == 2:
                        os.replace(replacement, self.index)
                    return original_connection()
                with patch.object(reader, '_bim_index_connection', side_effect=replace_before_query_open):
                    with self.assertRaisesRegex(RuntimeError, 'changed while opening requested'):
                        reader.find_variants({'v8'})
                self.assertEqual(opens, 2)
                self.assertEqual(len(reader._variant_metadata), 0)
                self.assertTrue(reader.prepare_bim_index()['built'])
                self.assertEqual(list(reader.find_variants({'v8'})), ['v8'])
                reader._variant_metadata.clear()

    def test_failed_atomic_publication_keeps_committed_index_and_removes_temp(self):
        reader = self._reader()
        reader.prepare_bim_index()
        old = self.index.read_bytes()
        bim = Path(str(self.prefix)+'.bim')
        bim.write_text(bim.read_text().replace('v0', 'renamed'))
        with patch('torchwgs.io.os.replace', side_effect=OSError('fixture publication failure')):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                reader.prepare_bim_index()
        self.assertEqual(self.index.read_bytes(), old)
        self.assertEqual(list(self.index.parent.iterdir()), [self.index])
        self.assertTrue(reader.prepare_bim_index()['built'])

    def test_source_change_during_build_cannot_publish_stale_index(self):
        reader = self._reader()
        reader.prepare_bim_index()
        old = self.index.read_bytes()
        bim = Path(str(self.prefix)+'.bim')
        bim.write_text(bim.read_text().replace('v0', 'renamed'))
        real_refresh = reader._refresh_bim_metadata
        calls = 0
        def mutate_before_check():
            nonlocal calls
            calls += 1
            if calls == 2:
                bim.write_text(bim.read_text().replace('v1', 'changed'))
            return real_refresh()
        with patch.object(reader, '_refresh_bim_metadata', side_effect=mutate_before_check):
            with self.assertRaisesRegex(RuntimeError, 'changed while building'):
                reader.prepare_bim_index()
        self.assertEqual(self.index.read_bytes(), old)
        self.assertEqual(list(self.index.parent.iterdir()), [self.index])
        self.assertTrue(reader.prepare_bim_index()['built'])

    def test_source_change_during_query_is_rejected_without_cached_results(self):
        reader = self._reader(cache=10)
        reader.prepare_bim_index()
        real_refresh = reader._refresh_bim_metadata
        calls = 0
        def mutate_before_check():
            nonlocal calls
            calls += 1
            if calls == 4:
                bim = Path(str(self.prefix)+'.bim')
                bim.write_text(bim.read_text().replace('v0', 'renamed'))
            return real_refresh()
        with patch.object(reader, '_refresh_bim_metadata', side_effect=mutate_before_check):
            with self.assertRaisesRegex(RuntimeError, 'changed while reading requested'):
                reader.find_variants({'v0'})
        self.assertEqual(len(reader._variant_metadata), 0)
        self.assertEqual(list(reader.find_variants({'v0', 'renamed'})), ['renamed'])

    def test_disabled_path_does_not_create_sqlite_and_keeps_old_scan(self):
        reader = self._reader(index=False)
        self.assertFalse(reader.prepare_bim_index()['enabled'])
        with patch('torchwgs.io.sqlite3.connect', side_effect=AssertionError('disabled sqlite')):
            self.assertEqual(list(reader.find_variants({'v3'})), ['v3'])
        self.assertFalse(self.index.exists())

    def test_query_size_validation_and_source_overwrite_protection(self):
        for query in (0, -1, 901, True, 2.5, '500'):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, 'bim_index_query_size'):
                self._reader(query=query)
        for suffix in ('.bed', '.bim', '.fam'):
            source = Path(str(self.prefix)+suffix)
            before = source.read_bytes()
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'must not overwrite'):
                BedReader(self.prefix, bim_index_path=source)
            self.assertEqual(source.read_bytes(), before)
        alias = self.root/'alias.sqlite'
        alias.symlink_to(Path(str(self.prefix)+'.bim'))
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            BedReader(self.prefix, bim_index_path=alias)


class ChromosomeBimIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prefix = self.root/'source'
        self.ids = [('family', f'sample{i}') for i in range(6)]
        write_bed(self.prefix, torch.arange(24, dtype=torch.float32).reshape(6,4)%3,
                  [Variant(i,'21',f'v{i}',10+i,'A','G') for i in range(4)], self.ids)
        pheno = self.root/'trait'
        pheno.write_text('FID IID continuous_trait\n')
        keep = self.root/'keep'
        keep.write_text(''.join(f'{fid} {iid}\n' for fid,iid in self.ids))
        self.inputs = DiscoveryInputs(str(self.prefix), str(pheno), 'continuous_trait',
                                      str(keep), {'21':str(self.prefix)})
        self.analysis = GeneAnalysis('Fixture', str(pheno), str(pheno), str(pheno))
        self.config = WGSConfig.paper(apply_rint=False)
        self.config.single_variant.device = 'cpu'
        self.context = SimpleNamespace(y=torch.tensor([.2,-.5]),
            sample_indices=torch.tensor([3,0]), sample_ids=[self.ids[i] for i in (3,0)])

    def _run(self, *, single=True, gene=True, resume=False):
        null = SimpleNamespace(sample_ids=self.ids, chromosomes=[21],
            align=lambda identifiers:torch.zeros((len(identifiers),1)))
        return chromosome.run_chromosome(('21',self.prefix), inputs=self.inputs,
            config=self.config, destination=self.root/'pipeline', null=null, null_key='fixture',
            y_lookup=dict(zip(self.ids,[.1,.2,.3,.4,.5,.6])), x_lookup=None,
            resume=resume, run_single=single, run_gene=gene)

    def test_26_groups_and_single_share_one_prepared_index_and_record_first_build(self):
        self.inputs.gene_analyses = {'21':[GeneAnalysis(f'Group{i:02}',
            self.analysis.annotation_file,self.analysis.setlist_file,self.analysis.mask_definition_file)
            for i in range(26)]}
        readers = []
        def single(reader, context, *, config):
            readers.append(reader)
            self.assertTrue(reader.bim_index_path.is_file())
            return iter(())
        def job(analysis, **kwargs):
            readers.append(kwargs['reader'])
            self.assertEqual(list(kwargs['reader'].find_variants({'v2'})), ['v2'])
            return analysis.name, analysis.name+'.regenie', {'rows':1,'numerics':{}}
        with patch.object(chromosome,'create_test_context',return_value=self.context), \
             patch.object(chromosome,'_gene_job',side_effect=job), \
             patch.object(chromosome,'iter_single_variant_results',side_effect=single):
            first = self._run()
            second = self._run()
        self.assertEqual(len(readers), 54)
        self.assertTrue(all(reader is readers[0] for reader in readers[:27]))
        self.assertTrue(all(reader is readers[27] for reader in readers[27:]))
        self.assertTrue(first['stages']['bim_index_c21']['built'])
        self.assertFalse(second['stages']['bim_index_c21']['built'])
        self.assertEqual(first['stages']['bim_index_c21']['rows'], 4)
        self.assertGreaterEqual(first['stages']['bim_index_c21']['seconds'], 0.)
        self.assertEqual(readers[0].bim_index_path, self.root/'pipeline/InputCache/bim_c21.sqlite')

    def test_python_switch_disables_preparation_for_fresh_gene_pipeline(self):
        self.inputs.gene_analyses = {'21':[self.analysis]}
        self.config.bim_index_enabled = False
        def job(analysis, **kwargs):
            self.assertIsNone(kwargs['reader'].bim_index_path)
            self.assertEqual(list(kwargs['reader'].find_variants({'v2'})), ['v2'])
            return analysis.name, 'result.regenie', {'rows':1,'numerics':{}}
        with patch.object(chromosome,'create_test_context',return_value=self.context), \
             patch.object(chromosome,'_gene_job',side_effect=job), \
             patch.object(BedReader,'prepare_bim_index',side_effect=AssertionError('disabled prep')):
            result = self._run(single=False)
        self.assertNotIn('bim_index_c21', result['stages'])
        self.assertFalse((self.root/'pipeline/InputCache/bim_c21.sqlite').exists())

    def test_single_only_and_fully_resumed_pipeline_never_prepare_gene_index(self):
        with patch.object(chromosome,'create_test_context',return_value=self.context), \
             patch.object(chromosome,'iter_single_variant_results',return_value=iter(())), \
             patch.object(BedReader,'prepare_bim_index',side_effect=AssertionError('unused prep')):
            result = self._run(gene=False)
        self.assertNotIn('bim_index_c21', result['stages'])
        self.inputs.gene_analyses = {'21':[self.analysis]}
        with patch('torchwgs.pipeline._stage_cache',return_value={
                    'result':'cached.regenie','report':{'rows':0,'numerics':{}}}), \
             patch.object(chromosome,'BedReader',side_effect=AssertionError('resumed reader')):
            result = self._run(resume=True)
        self.assertTrue(result['cached'])
        self.assertNotIn('bim_index_c21', result['stages'])


if __name__ == '__main__':
    unittest.main()
