"""Large physical sample-axis reader contracts; no GPU benchmark claims."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from staar_phewas.cache_runtime import sparse_codec_fast as codec
from staar_phewas.cache_runtime import sparse_decode, sparse_decode_fast
from staar_phewas.cache_runtime import fast_container, store


REF = np.array([2, 1, 0, 0, 1, 0], dtype=np.int64)
CALLED = np.array([2, 2, 2, 0, 1, 1], dtype=np.int64)


def fixture(n):
    raw = np.zeros((6, n), dtype=np.uint8)
    high = np.array([0, 1, n // 2, n - 3, n - 2, n - 1])
    raw[1, high] = np.arange(6, dtype=np.uint8)
    raw[2] = 2  # A rare REF variant exercises the opposite minor direction.
    raw[2, high] = np.arange(6, dtype=np.uint8)
    raw[3] = 3  # Entirely uncalled; summaries and dosage must preserve NaNs.
    raw[4, high] = np.array([4, 5, 4, 5, 1, 2], dtype=np.uint8)
    raw[5, high] = 1
    arrays = codec.compact(raw)
    reference = REF[raw].sum(axis=1, dtype=np.int64)
    called = CALLED[raw].sum(axis=1, dtype=np.int64)
    return raw, (*arrays, reference, called)


def dense_expected(raw, columns, samples, minimum_mac):
    """Independent dense six-state count/orientation oracle."""
    selected = raw[np.ix_(columns, samples)]
    reference = REF[selected].sum(axis=1, dtype=np.int64).astype(np.float64)
    called = CALLED[selected].sum(axis=1, dtype=np.int64)
    reference[called == 0] = np.nan
    af = np.divide(reference, called, out=np.full(len(columns), np.nan), where=called > 0)
    n = len(samples)
    missing = (2 * n - called) / (2 * n) if n else np.full(len(columns), np.nan)
    alternate = 2 * np.rint(n * (1 - missing)) - reference
    mac = np.where(reference >= alternate, alternate, reference)
    eligible = np.arange(len(columns)) if minimum_mac is None else np.flatnonzero(mac >= minimum_mac)
    dosage = np.where(selected < 3, 2 - selected.astype(np.int64), 3)
    dosage = np.where((dosage != 3) & (af[:, None] >= .5), 2 - dosage, dosage)
    summaries = tuple(value[eligible] for value in (af, missing, mac, reference, called))
    return columns[eligible], dosage[eligible].T.astype(np.uint8), summaries


def small_frame_container(path, chunk=128):
    """Build a complete fixture from codec records, without changing Writer."""
    path.mkdir()
    samples = np.arange(9, dtype=np.int64)
    np.save(path / 'samples.npy', samples, allow_pickle=False)
    raw = (np.arange(257 * len(samples)).reshape(257, len(samples)) % 6).astype(np.uint8)
    streams = {name: bytearray() for name in ('data.bin', 'headers.bin', 'counts.bin')}
    index = np.empty((len(raw) + chunk - 1) // chunk, dtype=store.INDEX)
    for j, start in enumerate(range(0, len(raw), chunk)):
        block = raw[start:start + chunk]
        payload, meta = codec.encode(block, level=3)
        counts = codec.integer_counts(*codec.compact(block), *block.shape)
        count_raw = np.column_stack([counts[key] for key in (
            'reference_alleles', 'called_alleles', 'half_missing_samples')]).astype('<i8').tobytes()
        count_payload = codec.codec._zstd().ZstdCompressor(
            level=3, write_content_size=True).compress(count_raw)
        header = store.canonical(dict(
            csr_meta=meta, counts_raw_sha256=store.sha(count_raw), counts_raw_bytes=len(count_raw),
            counts_compressed_sha256=store.sha(count_payload), counts_compressed_bytes=len(count_payload)))
        row = dict(start=start, m=len(block), offset=len(streams['data.bin']), size=len(payload),
                   header_offset=len(streams['headers.bin']), header_size=len(header),
                   counts_offset=len(streams['counts.bin']), counts_size=len(count_payload),
                   payload_sha=store.sha(payload), counts_sha=store.sha(count_payload))
        index[j] = tuple(row[key] for key in store.INDEX.names)
        for name, data in (('data.bin', payload), ('headers.bin', header), ('counts.bin', count_payload)):
            streams[name].extend(data)
    for name, data in streams.items():
        (path / name).write_bytes(data)
    np.save(path / 'index.npy', index, allow_pickle=False)
    manifest = dict(format=store.FORMAT, state_encoding=codec.codec.STATE_ENCODING,
                    m=len(raw), n=len(samples), chunk=chunk, frames=len(index),
                    binding={'unit_test': 'smaller-committed-frames'},
                    sample_sha256=store.sha(samples.astype('<i8').tobytes()),
                    files_sha256={name: store.file_sha(path / name) for name in (
                        'data.bin', 'headers.bin', 'counts.bin', 'samples.npy', 'index.npy')})
    write_manifest(path, manifest)
    return raw, manifest


def write_manifest(path, manifest):
    data = store.canonical(manifest)
    (path / 'manifest.json').write_bytes(data)
    (path / 'COMPLETE').write_text(store.sha(data))


class SmallerFrameContainerContracts(unittest.TestCase):
    def test_completed_128_column_frames_keep_geometry_and_hash_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cache'
            raw, manifest = small_frame_container(path)
            original = store.Container(path, manifest['binding'], np.arange(9, dtype=np.int64))
            self.assertEqual(original.manifest['chunk'], 128)
            self.assertEqual(store.CHUNK, 1024)
            self.assertEqual(original.index['m'].tolist(), [128, 128, 1])
            with fast_container.Container(path, manifest['binding']) as fast:
                self.assertTrue(fast.verify_streams())
                for frame in range(3):
                    actual = fast.read_frame(frame)
                    control = original.read_frame(frame)
                    for name in actual:
                        np.testing.assert_array_equal(actual[name], control[name])
                    start = actual['start']
                    states = raw[start:start + actual['m']]
                    np.testing.assert_array_equal(actual['reference_alleles'], REF[states].sum(1))
                    np.testing.assert_array_equal(actual['called_alleles'], CALLED[states].sum(1))

    def test_authenticated_noncontinuous_or_invalid_frame_size_rejected(self):
        for field, value, message in (
                ('start', 129, 'Index physical order'),
                ('m', 127, 'Index physical order'),
                ('offset', 1, 'Index stream gap')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'cache'
                _, manifest = small_frame_container(path)
                index = np.load(path / 'index.npy', allow_pickle=False)
                if field == 'offset':
                    index[1][field] += value
                else:
                    index[1][field] = value
                np.save(path / 'index.npy', index, allow_pickle=False)
                # Re-sign the fixture so geometry rejection is tested after SHA validation.
                manifest['files_sha256']['index.npy'] = store.file_sha(path / 'index.npy')
                write_manifest(path, manifest)
                with self.assertRaisesRegex(ValueError, message):
                    store.Container(path)

    def test_chunk_is_bounded_integer_and_original_writer_is_unchanged(self):
        self.assertEqual(store.CHUNK, 1024)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cache'
            _, manifest = small_frame_container(path)
            for invalid in (0, -1, 1025, True, 128.0):
                with self.subTest(chunk=invalid):
                    manifest['chunk'] = invalid
                    write_manifest(path, manifest)
                    with self.assertRaisesRegex(ValueError, 'Container semantic mismatch'):
                        store.Container(path)


class LargeSampleReaderContracts(unittest.TestCase):
    def test_codec_width_boundary_and_large_axis_match_dense_oracle(self):
        for n in (65535, 65536, 341101):
            with self.subTest(n=n):
                raw, arrays = fixture(n)
                expected_dtype = np.dtype(np.uint16 if n < 65536 else np.uint32)
                self.assertEqual(arrays[1].dtype, expected_dtype)
                # Last physical sample must survive without uint16 wrapping.
                self.assertEqual(int(arrays[1].max()), n - 1)
                source = sparse_decode_fast.validate_source(*arrays, n)
                axes = (
                    np.arange(n, dtype=np.int64),
                    np.arange(n - 1, -1, -1, dtype=np.int64),
                    np.array([n - 1, 0, n - 2, n // 2, 1], dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )
                columns = np.array([5, 2, 0, 4, 1, 3], dtype=np.int64)
                for samples in axes:
                    for minimum_mac in (None, 2, 20):
                        expected_columns, expected_dosage, summaries = dense_expected(
                            raw, columns, samples, minimum_mac)
                        prepared = (
                            sparse_decode.prepare(*arrays, n, columns, samples,
                                                  minimum_mac=minimum_mac),
                            sparse_decode_fast.prepare_validated(
                                source, columns, samples, minimum_mac=minimum_mac),
                        )
                        for result in prepared:
                            np.testing.assert_array_equal(result['columns'], expected_columns)
                            np.testing.assert_array_equal(result['samples'], samples)
                            np.testing.assert_array_equal(sparse_decode.dosage_numpy(result), expected_dosage)
                            for actual, expected in zip(result['summaries'], summaries):
                                np.testing.assert_array_equal(actual, expected)

    def test_large_axis_empty_columns_and_empty_exceptions(self):
        n = 341101
        raw, arrays = fixture(n)
        source = sparse_decode_fast.validate_source(*arrays, n)
        for columns in (np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64)):
            result = sparse_decode_fast.prepare_validated(source, columns)
            self.assertEqual(result['exception_state'].size, 0)
            self.assertEqual(result['exception_row'].dtype, np.int64)
            self.assertEqual(sparse_decode.dosage_numpy(result).shape, (n, len(columns)))
        _, rows, _ = sparse_decode_fast._exceptions(source, np.array([0]))
        self.assertEqual(rows.dtype, np.uint32)

    def test_dtype_dimension_and_high_index_corruption_rejected(self):
        for n in (65535, 65536, 341101):
            raw, arrays = fixture(n)
            wrong_width = np.uint32 if n < 65536 else np.uint16
            wrong = (arrays[0], arrays[1].astype(wrong_width), *arrays[2:])
            for validate in (sparse_decode.prepare, sparse_decode_fast.validate_source):
                with self.assertRaises(ValueError):
                    validate(*wrong, n)
                outside = arrays[1].copy()
                outside[-1] = n
                with self.assertRaises(ValueError):
                    validate(arrays[0], outside, *arrays[2:], n)
        # Validate uint32's full index capacity without allocating its axis.
        n = 2**32
        offsets = np.array([0, 1], dtype=np.uint32)
        samples = np.array([n - 1], dtype=np.uint32)
        states = np.array([1], dtype=np.uint8)
        reference = np.array([2 * n - 1], dtype=np.int64)
        called = np.array([2 * n], dtype=np.int64)
        source = sparse_decode_fast.validate_source(offsets, samples, states, reference, called, n)
        self.assertEqual(int(source.sample_index[0]), n - 1)
        for validate in (sparse_decode.prepare, sparse_decode_fast.validate_source):
            with self.assertRaises(ValueError):
                validate(offsets, samples, states, reference, called, n + 1)
        duplicate = np.array([65536, 65536], dtype=np.uint32)
        for validate in (sparse_decode.prepare, sparse_decode_fast.validate_source):
            with self.assertRaises(ValueError):
                validate(np.array([0, 2], dtype=np.uint32), duplicate,
                         np.array([1, 2], dtype=np.uint8),
                         np.array([2 * 341101 - 3], dtype=np.int64),
                         np.array([2 * 341101], dtype=np.int64), 341101)

    def test_large_source_payload_remains_immutable(self):
        _, arrays = fixture(341101)
        source = sparse_decode_fast.validate_source(*arrays, 341101)
        for value in (source.offsets, source.sample_index, source.state, source.ref_ac, source.called):
            with self.assertRaises(ValueError):
                value.flags.writeable = True
        original = int(source.sample_index[0])
        arrays[1][0] = original + 1
        self.assertEqual(int(source.sample_index[0]), original)


if __name__ == '__main__':
    unittest.main()
