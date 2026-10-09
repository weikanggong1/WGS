"""Anonymous cache-only reader contracts; fixtures are not benchmarks."""
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from staar_phewas.cache_runtime import sparse_codec_fast, sparse_decode, store
from staar_phewas.cache_runtime.portable import (
    PortableCachedGDS, PortableMetadataReader, canonical_numeric_ids, export_metadata)


class NativeMetadataFixture:
    n_samples = 8
    n_variants = 6

    def __init__(self):
        self.calls = []
        self.fields = {
            "position": np.asarray([10, 20, 20, 40, 50, 60], dtype=np.int64),
            "chromosome": np.asarray(["1"] * 6),
            "variant.id": np.arange(11, 17, dtype=np.int32),
            "allele": np.asarray(["A,C", "C,T,G", "G,A", "AG,A", "T,TC", "C,G"]),
            "annotation/filter": np.asarray(["PASS", None, "PASS", "OTHER", "PASS", "PASS"], dtype=object),
            "annotation/info/Category": np.asarray(["coding", "", None, "untranslated", "coding", "intergenic"], dtype=object),
            "annotation/info/weight": np.asarray([.5, np.nan, 0., 2., 4., 1.]),
            "annotation/info/vector": np.arange(12, dtype=np.float64).reshape(6, 2),
            "annotation/info/ragged": [np.asarray(["a", "b"]), np.asarray([], dtype="U1"),
                np.asarray(["longer_name"]), np.asarray(["c", "d", "e"]),
                np.asarray([], dtype="U1"), np.asarray(["f"])],
            "annotation/info/bytes": np.asarray([b"A", b"BC", b"D", b"E", b"FG", b"H"]),
            "annotation/info/numeric_ragged": [np.asarray([1, 2], dtype=np.int32),
                np.asarray([], dtype=float), np.asarray([3], dtype=np.int32),
                np.asarray([], dtype=float), np.asarray([4, 5], dtype=np.int32),
                np.asarray([6], dtype=np.int32)],
        }

    def sample_ids(self):
        return np.asarray(["%d_%d" % (101 + i, 101 + i) for i in range(8)])

    def read_field(self, path, indices):
        self.calls.append((path, np.asarray(indices).copy()))
        values = self.fields[path]
        return [values[int(i)] for i in indices] if isinstance(values, list) else values[indices]


@pytest.fixture
def portable(tmp_path):
    native = NativeMetadataFixture()
    # Larger source population, smaller reordered cached population. Public
    # indices must become 0,1,2,3; no original physical rows leak into lookup.
    source_rows = np.asarray([7, 1, 5, 3], dtype=np.int64)
    raw = np.asarray([[0, 1, 2, 3], [4, 0, 5, 1], [0, 0, 0, 0],
                      [2, 1, 4, 0], [1, 5, 0, 3], [0, 1, 1, 0]], dtype=np.uint8)
    cache = tmp_path / "cache"
    writer = store.Writer(cache, {"source": "unavailable-source.gds"}, source_rows, 6, source_bytes=10**8)
    offsets, rows, states = sparse_codec_fast.compact(raw)
    counts = sparse_codec_fast.integer_counts(offsets, rows, states, *raw.shape)
    writer.append(raw, counts)
    writer.finish()
    metadata = cache / "metadata"
    manifest = export_metadata(native, cache, metadata, list(native.fields), block_size=2,
        annotation_catalog={"Category": "annotation/info/Category", "weight": "annotation/info/weight"},
        annotation_names=["weight"])
    return native, cache, metadata, manifest, raw, source_rows


def test_normalization_is_explicit_bijective_and_rejects_ambiguous_identifiers():
    np.testing.assert_array_equal(canonical_numeric_ids(["101", "102_102"]), [101, 102])
    for values in (["101_102"], ["00101"], ["101.0"], ["prefix101"], ["0"],
                   ["101", "101_101"], [str(2**63)]):
        with pytest.raises(ValueError):
            canonical_numeric_ids(values)


def test_metadata_chunked_roundtrip_fixed_ragged_missing_and_multiallelic(portable):
    native, cache, metadata, manifest, _, source_rows = portable
    assert manifest["n_samples"] == 4 and manifest["n_source_samples"] == 8
    assert manifest["preparation"]["genotype_read"] is False
    assert all(len(indices) <= 2 for _, indices in native.calls)
    with PortableMetadataReader(metadata, cache) as reader:
        assert reader.n_samples == 4 and reader.n_variants == 6
        np.testing.assert_array_equal(reader.sample_ids(), ["108", "102", "106", "104"])
        np.testing.assert_array_equal(reader.sample_indices(["106", "108"]), [2, 0])
        selected = np.asarray([5, 1, 3, 0], dtype=np.int64)
        for name, expected in native.fields.items():
            for indices in (None, selected, np.asarray([], dtype=np.int64)):
                observed = reader.read_field(name, indices)
                request = np.arange(6) if indices is None else indices
                if isinstance(expected, list):
                    assert isinstance(observed, list) and len(observed) == len(request)
                    for a, i in zip(observed, request):
                        np.testing.assert_array_equal(a, expected[i])
                else:
                    np.testing.assert_array_equal(observed, expected[request])
        assert isinstance(reader.read_field("position"), np.memmap)
        ref, alt = reader.read_ref_alt([1, 3])
        np.testing.assert_array_equal(ref, ["C", "AG"])
        np.testing.assert_array_equal(alt, ["T,G", "A"])
        np.testing.assert_array_equal(reader.read_field("$num_allele", [1, 3]), [3, 2])
        np.testing.assert_array_equal(reader._array(manifest["source_sample_rows"]), source_rows)
        assert reader.reader_metadata["original_gds_required"] is False
    with pytest.raises(ValueError, match="closed"):
        reader.read_field("position")


def test_cache_only_adapter_never_imports_sdk_or_stats_original_source(portable, monkeypatch):
    _, cache, _, _, raw, _ = portable
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == "pygds" or name.startswith("pygds."):
            raise AssertionError("cache-only runtime imported native SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    original_stat = Path.stat
    def checked_stat(path, *args, **kwargs):
        if path.suffix == ".gds":
            raise AssertionError("cache-only runtime statted original source")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", checked_stat)
    with PortableCachedGDS(cache, device="cpu") as reader:
        np.testing.assert_array_equal(reader.sample_indices(["106", "108", "102"]), [2, 0, 1])
        variants, samples = np.asarray([5, 1, 3, 0]), np.asarray([2, 0, 1])
        actual = reader.minor_block(variants, samples)
        offsets, rows, states = sparse_codec_fast.compact(raw)
        counts = sparse_codec_fast.integer_counts(offsets, rows, states, *raw.shape)
        control = sparse_decode.prepare(offsets, rows, states,
            counts["reference_alleles"], counts["called_alleles"], 4, variants, samples)
        expected = sparse_decode.dosage_numpy(control).astype(np.float64)
        expected[expected == 3] = np.nan
        observed, _, _, _, _ = actual.trait_dense(np.arange(3), "minor")
        np.testing.assert_array_equal(observed, np.nan_to_num(expected))
        np.testing.assert_array_equal(actual.variant_indices, variants)
        np.testing.assert_array_equal(actual.sample_indices, samples)
        assert hasattr(reader, "iter_effective_minor_blocks")
        assert reader.reader_metadata["analysis_cache"]["genotype_sdk_fallback_count"] == 0
        with pytest.raises(IndexError, match="outside"):
            reader.minor_block([0], [4])


def test_fail_closed_missing_samples_duplicates_incomplete_and_field_mutation(portable):
    _, cache, metadata, manifest, _, _ = portable
    reader = PortableMetadataReader(metadata, cache)
    for values in (["101"], ["108", "108"], [["108"]]):
        with pytest.raises(ValueError):
            reader.sample_indices(values)
    with pytest.raises(KeyError):
        reader.read_field("not_exported")
    with pytest.raises(ValueError, match="duplicate"):
        reader.read_field("position", [0, 0])
    field = metadata / manifest["fields"]["position"]["file"]
    reader.read_field("position")
    with field.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="changed"):
        reader.read_field("position")
    fresh = PortableMetadataReader(metadata, cache)
    with pytest.raises(ValueError, match="checksum"):
        fresh.read_field("position")
    (metadata / "COMPLETE").write_text("bad")
    with pytest.raises(ValueError, match="marker"):
        PortableMetadataReader(metadata, cache)


def test_exact_genotype_manifest_binding_no_silent_reuse(portable):
    native, cache, metadata, manifest, _, _ = portable
    with pytest.raises(FileExistsError):
        export_metadata(native, cache, metadata, list(native.fields))
    document = json.loads((metadata / "manifest.json").read_text())
    document["genotype_manifest_sha256"] = "0" * 64
    raw = json.dumps(document).encode()
    (metadata / "manifest.json").write_bytes(raw)
    (metadata / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="another"):
        PortableMetadataReader(metadata, cache)
