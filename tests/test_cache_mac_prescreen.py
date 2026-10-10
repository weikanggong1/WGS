"""CPU contract comparisons against the independent original CSR decoder.

These anonymous fixtures validate reader semantics only, not performance or
scientific equivalence. No genotype, GPU, sample identifiers or site metadata.
"""
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast, sparse_decode, sparse_decode_fast
from fudan_wgs_toolkit.cache_runtime.adapter_fast import CachedGenotypeAdapter
from fudan_wgs_toolkit.genotype import _allele_frequency_summary


def payload(raw):
    off, idx, states = sparse_codec_fast.compact(raw)
    counts = sparse_codec_fast.integer_counts(off, idx, states, *raw.shape)
    return off, idx, states, counts['reference_alleles'], counts['called_alleles']


def assert_prepared_equal(case, actual, expected):
    case.assertEqual(set(actual), set(expected))
    for name in ('cache_variant_count', 'cache_sample_count', 'full_union_summaries'):
        case.assertEqual(actual[name], expected[name])
    for name in ('columns', 'samples', 'exception_col', 'exception_row', 'exception_state'):
        case.assertEqual(actual[name].dtype, expected[name].dtype)
        np.testing.assert_array_equal(actual[name], expected[name])
    for actual_summary, expected_summary in zip(actual['summaries'], expected['summaries']):
        case.assertEqual(actual_summary.dtype, expected_summary.dtype)
        np.testing.assert_array_equal(actual_summary, expected_summary)


class SafeMacContracts(unittest.TestCase):
    def test_all_states_missing_empty_permutations_and_fractional_cutoffs(self):
        raw = np.array([
            [0, 1, 2, 3, 4, 5], [3, 3, 3, 3, 3, 3],
            [4, 5, 4, 5, 4, 5], [0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2], [1, 1, 0, 0, 2, 3],
        ], dtype=np.uint8)
        args = payload(raw)
        source = sparse_decode_fast.validate_source(*args, raw.shape[1])
        axes = [None, np.arange(5, -1, -1), np.array([5, 0, 2]), np.empty(0, dtype=np.int64)]
        for samples in axes:
            binding = sparse_decode_fast.bind_samples(samples, raw.shape[1])
            canonical = None if binding.identity else binding.rows
            for columns in (None, np.array([5, 2, 0]), np.empty(0, dtype=np.int64)):
                for cutoff in (None, 0, 0.5, 1, 2, 3, 20):
                    with self.subTest(samples=samples, columns=columns, cutoff=cutoff):
                        expected = sparse_decode.prepare(*args, raw.shape[1], columns, samples, minimum_mac=cutoff)
                        unbound = sparse_decode_fast.prepare_validated(source, columns, samples, minimum_mac=cutoff)
                        bound = sparse_decode_fast.prepare_validated(source, columns, canonical,
                            minimum_mac=cutoff, sample_binding=binding)
                        assert_prepared_equal(self, unbound, expected)
                        assert_prepared_equal(self, bound, expected)
                        np.testing.assert_array_equal(sparse_decode.dosage_numpy(bound), sparse_decode.dosage_numpy(expected))

    def test_random_cohorts_bit_exact_against_original_loop(self):
        rng = np.random.default_rng(806)
        raw = rng.integers(0, 6, (48, 31), dtype=np.uint8)
        raw[:8] = 0
        args = payload(raw)
        source = sparse_decode_fast.validate_source(*args, raw.shape[1])
        for count in (1, 2, 11, 30, 31):
            rows = rng.choice(raw.shape[1], count, replace=False)
            columns = rng.permutation(raw.shape[0])[:37]
            binding = sparse_decode_fast.bind_samples(rows, raw.shape[1])
            for cutoff in (None, 0, 1, 5, 10, 19, 20, 20.25):
                with self.subTest(count=count, cutoff=cutoff):
                    expected = sparse_decode.prepare(*args, raw.shape[1], columns, rows, minimum_mac=cutoff)
                    actual = sparse_decode_fast.prepare_validated(source, columns, binding.rows,
                        minimum_mac=cutoff, sample_binding=binding)
                    assert_prepared_equal(self, actual, expected)

    def test_half_missing_raw_alt_19_must_survive_cutoff_20(self):
        # Sparse construction of the rounding boundary without a dense cohort.
        # 19 heterozygotes + missing diploid + half-called REF gives ALT=19.
        n = 345967
        offsets = np.array([0, 21, 21], dtype=np.uint32)
        indices = np.arange(21, dtype=np.uint32)
        states = np.array([1] * 19 + [3, 4], dtype=np.uint8)
        reference = np.array([2 * n - 22, 2 * n], dtype=np.int64)
        called = np.array([2 * n - 3, 2 * n], dtype=np.int64)
        args = offsets, indices, states, reference, called
        self.assertEqual(int(called[0] - reference[0]), 19)
        self.assertEqual(_allele_frequency_summary(reference, called, n)[2][0], 20)
        # Remove only default-REF samples; original floating rounding still20.
        rows = np.concatenate((np.arange(21), np.arange(6975, n))).astype(np.int64)
        self.assertEqual(len(rows), 339013)
        metrics = {}
        binding = sparse_decode_fast.bind_samples(rows, n, metrics=metrics)
        source = sparse_decode_fast.validate_source(*args, n)
        expected = sparse_decode.prepare(*args, n, None, rows, minimum_mac=20)
        actual = sparse_decode_fast.prepare_validated(source, None, binding.rows,
            minimum_mac=20, sample_binding=binding, metrics=metrics)
        np.testing.assert_array_equal(expected['columns'], [0])
        self.assertEqual(expected['summaries'][2][0], 20)
        assert_prepared_equal(self, actual, expected)
        self.assertEqual(metrics['safe_mac_prescreen_input_variants'], 2)
        self.assertEqual(metrics['safe_mac_prescreen_rejected_variants'], 1)
        self.assertEqual(metrics['safe_mac_prescreen_candidate_variants'], 1)

    def test_prescreen_precedes_exception_expansion_and_preserves_order(self):
        raw = np.zeros((6, 24), dtype=np.uint8)
        raw[1, :10] = 1
        raw[4, :15] = 1
        raw[5] = 2  # zero REF, cannot reach the threshold in a subset
        args = payload(raw)
        source = sparse_decode_fast.validate_source(*args, raw.shape[1])
        rows = np.arange(23, dtype=np.int64)
        columns = np.array([5, 4, 3, 1, 0], dtype=np.int64)
        metrics = {}
        binding = sparse_decode_fast.bind_samples(rows, raw.shape[1], metrics=metrics)
        with patch.object(sparse_decode_fast, '_exceptions', wraps=sparse_decode_fast._exceptions) as expand:
            actual = sparse_decode_fast.prepare_validated(source, columns, binding.rows,
                minimum_mac=5, sample_binding=binding, metrics=metrics)
            np.testing.assert_array_equal(expand.call_args.args[1], [4, 1])
        expected = sparse_decode.prepare(*args, raw.shape[1], columns, rows, minimum_mac=5)
        assert_prepared_equal(self, actual, expected)
        self.assertEqual(metrics['safe_mac_prescreen_rejected_variants'], 3)

    def test_high_uint32_rows_and_reordered_subset(self):
        n = 70003
        offsets = np.array([0, 3, 6], dtype=np.uint32)
        indices = np.array([0, 65536, 70002, 1, 65535, 70001], dtype=np.uint32)
        states = np.array([1, 4, 2, 3, 5, 1], dtype=np.uint8)
        counts = sparse_codec_fast.integer_counts(offsets, indices, states, 2, n)
        args = offsets, indices, states, counts['reference_alleles'], counts['called_alleles']
        rows = np.array([70002, 65536, 1, 65535, 0], dtype=np.uint32)
        source = sparse_decode_fast.validate_source(*args, n)
        binding = sparse_decode_fast.bind_samples(rows, n)
        for cutoff in (None, 0, 1, 2, 3):
            with self.subTest(cutoff=cutoff):
                actual = sparse_decode_fast.prepare_validated(source, np.array([1, 0]), binding.rows,
                    minimum_mac=cutoff, sample_binding=binding)
                expected = sparse_decode.prepare(*args, n, np.array([1, 0]), rows, minimum_mac=cutoff)
                assert_prepared_equal(self, actual, expected)
                np.testing.assert_array_equal(sparse_decode.dosage_numpy(actual), sparse_decode.dosage_numpy(expected))

    def test_zero_cache_axis_and_invalid_minimum_mac(self):
        raw = np.empty((2, 0), dtype=np.uint8)
        args = payload(raw)
        source = sparse_decode_fast.validate_source(*args, 0)
        binding = sparse_decode_fast.bind_samples(None, 0)
        for cutoff in (None, 0, 1):
            assert_prepared_equal(self,
                sparse_decode_fast.prepare_validated(source, minimum_mac=cutoff, sample_binding=binding),
                sparse_decode.prepare(*args, 0, minimum_mac=cutoff))
        for cutoff in (-1, np.inf, np.nan, [1]):
            with self.assertRaises(ValueError):
                sparse_decode_fast.prepare_validated(source, minimum_mac=cutoff, sample_binding=binding)


class SampleMapContracts(unittest.TestCase):
    def setUp(self):
        raw = np.array([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], dtype=np.uint8)
        self.args = payload(raw)
        self.source = sparse_decode_fast.validate_source(*self.args, raw.shape[1])

    def test_canonical_map_reused_without_repeated_axis_validation(self):
        rows = np.array([5, 1, 3])
        metrics = {}
        binding = sparse_decode_fast.bind_samples(rows, 6, metrics=metrics)
        with patch.object(sparse_decode_fast, '_indices', wraps=sparse_decode_fast._indices) as indices:
            for column in (0, 1, 0):
                sparse_decode_fast.prepare_validated(self.source, np.array([column]), binding.rows,
                    sample_binding=binding, metrics=metrics)
            self.assertEqual(indices.call_count, 3)  # each call validates columns only
        self.assertEqual(metrics['decoder_sample_axis_validations'], 1)
        self.assertEqual(metrics['decoder_sample_map_builds'], 1)
        self.assertEqual(metrics['decoder_sample_axis_cache_hits'], 3)
        self.assertEqual(metrics['decoder_sample_map_cache_bytes'], 6 * 8)

    def test_input_mutation_does_not_mutate_binding_and_foreign_input_is_checked(self):
        rows = np.array([5, 1, 3])
        binding = sparse_decode_fast.bind_samples(rows, 6)
        rows[:] = [0, 1, 2]
        np.testing.assert_array_equal(binding.rows, [5, 1, 3])
        for axis in (binding.rows, binding.rowmap):
            with self.assertRaises(ValueError):
                axis.flags.writeable = True
        with self.assertRaises(FrozenInstanceError):
            binding.n = 7
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=rows, sample_binding=binding)
        expected = sparse_decode.prepare(*self.args, 6, samples=np.array([5, 1, 3]))
        actual = sparse_decode_fast.prepare_validated(self.source, samples=np.array([5, 1, 3]), sample_binding=binding)
        assert_prepared_equal(self, actual, expected)
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=np.array([5, 5, 3]), sample_binding=binding)

    def test_unsealed_wrong_dimension_or_modified_array_metadata_fails(self):
        binding = sparse_decode_fast.bind_samples(np.array([1, 3]), 6)
        unsealed = sparse_decode_fast.SampleBinding(binding.rows, binding.rowmap, 6, False, False)
        for invalid in (unsealed, sparse_decode_fast.bind_samples(np.array([1, 3]), 7)):
            with self.assertRaises(ValueError):
                sparse_decode_fast.prepare_validated(self.source, samples=invalid.rows, sample_binding=invalid)
        binding.rows.shape = (1, 2)
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=binding.rows, sample_binding=binding)
        binding = sparse_decode_fast.bind_samples(np.array([1, 3]), 6)
        binding.rowmap.shape = (2, 3)
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=binding.rows, sample_binding=binding)
        binding = sparse_decode_fast.bind_samples(np.array([1, 3]), 6)
        # Read-only data does not prohibit writable ndarray metadata. A zero
        # stride would duplicate indices while retaining shape and dtype.
        binding.rows.strides = (0,)
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=binding.rows, sample_binding=binding)
        binding = sparse_decode_fast.bind_samples(np.array([1, 3]), 6)
        binding.rowmap.strides = (0,)
        with self.assertRaises(ValueError):
            sparse_decode_fast.prepare_validated(self.source, samples=binding.rows, sample_binding=binding)

    def test_full_permutation_and_identity_take_different_map_paths(self):
        metrics = {}
        identity = sparse_decode_fast.bind_samples(None, 6, metrics=metrics)
        self.assertIsNone(identity.rowmap)
        self.assertEqual(metrics.get('decoder_sample_map_builds', 0), 0)
        reordered = sparse_decode_fast.bind_samples(np.arange(5, -1, -1), 6, metrics=metrics)
        self.assertIsNotNone(reordered.rowmap)
        actual = sparse_decode_fast.prepare_validated(self.source, samples=reordered.rows, sample_binding=reordered)
        expected = sparse_decode.prepare(*self.args, 6, samples=np.arange(5, -1, -1))
        assert_prepared_equal(self, actual, expected)
        self.assertEqual(metrics['decoder_sample_map_builds'], 1)


class AdapterMapContracts(unittest.TestCase):
    def test_crossframe_adapter_reuses_one_map_then_rebinds_changed_values(self):
        class Reader:
            n_samples = 30
            n_variants = 6
            reader_metadata = {}
            def close(self):
                pass

        class Container:
            complete = True
            samples = np.arange(29, -1, -1, dtype=np.int64)
            index = np.array([(0, 3), (3, 3)], dtype=[('start', 'i8'), ('m', 'i8')])
            raw = np.zeros((6, 30), dtype=np.uint8)
            raw[:, :12] = 1
            def read_frame(self, frame):
                start, count = self.index[frame]
                off, idx, states, ref, called = payload(self.raw[start:start + count])
                return dict(start=int(start), m=int(count), n=30, offsets=off,
                    sample_index=idx, state=states, reference_alleles=ref, called_alleles=called)

        container = Container()
        adapter = CachedGenotypeAdapter(Reader(), container)
        samples = np.array([29, 0, 15, 22, 28], dtype=np.int64)
        columns = np.array([5, 0, 4, 2])
        for _ in range(3):
            actual, _, _ = adapter._prepare(columns, samples.copy(), 1)
            cache_rows = 29 - samples
            expected = sparse_decode.prepare(*payload(container.raw), 30, columns, cache_rows, minimum_mac=1)
            # Adapter normalizes both selected input axes; dosage/summaries are
            # still exact original values and the filtered positions match.
            np.testing.assert_array_equal(actual['columns'], np.flatnonzero(np.isin(columns, expected['columns'])))
            np.testing.assert_array_equal(sparse_decode.dosage_numpy(actual), sparse_decode.dosage_numpy(expected))
            for a, e in zip(actual['summaries'], expected['summaries']):
                np.testing.assert_array_equal(a, e)
        self.assertEqual(adapter._metrics['decoder_sample_axis_validations'], 1)
        self.assertEqual(adapter._metrics['decoder_sample_map_builds'], 1)
        self.assertEqual(adapter._metrics['decoder_sample_axis_cache_hits'], 6)
        first = adapter._decoder_binding
        samples[[0, 1]] = samples[[1, 0]]
        adapter._prepare(columns, samples, 1)
        self.assertIsNot(adapter._decoder_binding, first)
        self.assertEqual(adapter._metrics['decoder_sample_map_builds'], 2)
        adapter.close()
        self.assertIsNone(adapter._decoder_binding)
        self.assertEqual(adapter._metrics['decoder_sample_map_cache_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
