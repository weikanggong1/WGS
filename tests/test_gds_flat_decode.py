"""Synthetic edge-case units only; these are not real-data benchmarks."""
from types import SimpleNamespace
import numpy as np
import pytest

from staar_phewas.gds import SeqArrayGDS


def make_reader(sample_multiplier=1):
    steps = np.asarray([1, 2, 16, 0, 16, 15, 7, 8], dtype=np.uint8)
    genotype = np.asarray([
        [[0, 1], [2, -1], [-1, 1], [0, 0]],
        [[0, 3], [4, 14], [-1, 1], [0, 0]],
        [[0, 2**32 - 2], [2**31, 65535], [2**31 + 5, -1], [16, 32768]],
        [[-1, -1]] * 4,
        [[2**32 - 2, -1], [2**31, 1], [0, 0], [2**32 - 2, -1]],
        [[2**30 - 2, -1], [2**29, 1], [0, 0], [65535, 32768]],
        [[2**14 - 2, -1], [2**13, 1], [0, 0], [16, 3276]],
        [[2**16 - 2, -1], [2**15, 1], [0, 0], [65533, 32768]],
    ], dtype=np.int64)
    genotype = np.tile(genotype, (1, sample_multiplier, 1))
    layers = []
    for count, calls in zip(steps, genotype):
        encoded = np.where(calls < 0, (1 << (2 * int(count))) - 1, calls).astype(np.uint64)
        for layer in range(int(count)):
            layers.append(((encoded >> (2 * layer)) & 3).astype(np.uint8))
    node = SimpleNamespace(flat=np.asarray(layers).ravel(), calls=[])
    reader = SeqArrayGDS.__new__(SeqArrayGDS)
    reader.n_samples, reader.n_variants, reader.ploidy = genotype.shape[1], len(steps), 2
    reader._genotype_steps = steps
    reader._genotype_offsets = np.r_[0, np.cumsum(steps, dtype=np.int64)]
    reader._file = SimpleNamespace(fileid=42, index=lambda path: node)
    reader.genotype_raw_memory_bytes = 256
    reader.genotype_max_gap_layers = 0

    def read_flat(fileid, path, offset, count, dtype):
        assert fileid == 42 and path == "genotype/data"
        assert dtype == "uint8"
        assert 0 <= offset <= len(node.flat) and offset + count <= len(node.flat)
        node.calls.append((offset, count))
        return node.flat[offset:offset + count].copy()

    def read_selected(fileid, path, offset, raw_rows, row_width, selection, dtype):
        assert fileid == 42 and path == "genotype/data" and dtype == "uint8"
        assert row_width == reader.n_samples * reader.ploidy
        raw = node.flat[offset:offset + raw_rows * row_width].reshape(raw_rows, row_width)
        value = raw[:, selection].ravel().copy()
        node.calls.append((offset, value.size))
        return value
    reader._flat_reader = SimpleNamespace(read_flat_path=read_flat, read_selected_rows_path=read_selected)
    def readex(selection):
        raw_mask, sample_mask, _ = selection
        return node.flat.reshape(-1, reader.n_samples, 2)[raw_mask][:, sample_mask, :].copy()
    node.readex = readex
    return reader, genotype, read_flat, node


def test_all_bit_layers_signed_missing_high_codes_and_arbitrary_orders():
    reader, expected, read_flat, node = make_reader()
    variants = np.asarray([7, 2, 0, 3, 5, 1, 4, 6])
    samples = np.asarray([3, 0, 2, 1])
    actual = reader.read_genotype(variants, samples)
    np.testing.assert_array_equal(actual, expected[variants][:, samples])
    assert actual.dtype == np.int64
    assert np.any(actual == 2**32 - 2) and np.any(actual == 2**31)
    assert not np.any(actual == 2**32 - 1)
    assert np.all(actual[3] == -1)
    assert all(count <= 256 for _, count in node.calls)
    assert len(node.calls) > len(variants) - 1  # 16-layer sites split under budget.
    dosage = reader.read_ref_dosage(variants, samples)
    target = (expected[variants][:, samples] == 0).sum(axis=2).astype(float)
    target[(expected[variants][:, samples] < 0).any(axis=2)] = np.nan
    np.testing.assert_array_equal(dosage, target.T)
    assert dosage[3, 1] == 0  # Both high called codes remain ALT, not missing.


def test_raw_budget_rejection_and_empty_cases_do_not_read():
    reader, _, read_flat, node = make_reader()
    reader.genotype_raw_memory_bytes = 8
    with pytest.raises(MemoryError):
        reader.read_genotype(np.asarray([0]), np.arange(4))
    empty = np.asarray([], dtype=np.int64)
    assert reader.read_genotype(empty, np.arange(4)).shape == (0, 4, 2)
    assert reader.read_genotype(np.arange(8), empty).shape == (8, 0, 2)
    assert not node.calls


def test_optional_readex_fallback_retains_signed_16_layer_codes():
    reader, expected, _, _ = make_reader()
    reader._flat_reader = None
    variants, samples = np.asarray([4, 2, 3, 1]), np.asarray([2, 0, 3, 1])
    actual = reader.read_genotype(variants, samples)
    np.testing.assert_array_equal(actual, expected[variants][:, samples])
    assert actual.dtype == np.int64


def test_native_read_errors_are_not_silently_retried():
    reader, _, _, _ = make_reader()
    def broken(*args):
        raise RuntimeError("closed GDS handle")
    reader._flat_reader = SimpleNamespace(read_flat_path=broken)
    with pytest.raises(RuntimeError, match="closed GDS handle"):
        reader.read_genotype(np.asarray([0]), np.arange(4))


@pytest.mark.parametrize("samples,route", [(np.asarray([6, 1, 4]), "selected"),
                                            (np.asarray([7, 1, 5, 3]), "flat")])
def test_native_auto_two_sites_and_half_sample_boundary(samples, route):
    reader, expected, _, _ = make_reader(sample_multiplier=2)
    reader.genotype_raw_memory_bytes = 1024
    variants = np.asarray([1, 0])
    actual = reader.read_genotype(variants, samples)
    np.testing.assert_array_equal(actual, expected[variants][:, samples])
    assert reader._reader_io_counts[route]["calls"] == 1
    other = "flat" if route == "selected" else "selected"
    assert reader._reader_io_counts[other]["calls"] == 0
    width = len(samples) * 2 if route == "selected" else reader.n_samples * 2
    assert reader._reader_io_counts[route]["returned_bytes"] == 3 * width
