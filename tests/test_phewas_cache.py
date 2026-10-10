"""Legacy cache contracts on small anonymous fixtures, not a benchmark."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import phewas_cache
from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast, store
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader, export_metadata


class MetadataFixture:
    n_samples = 9
    n_variants = 3

    def sample_pairs(self):
        return np.asarray([["family", str(301 + index)] for index in range(self.n_samples)])

    def read_field(self, path, indices):
        fields = {"position": np.asarray([10, 20, 30]), "chromosome": np.asarray(["2"] * 3),
            "variant.id": np.asarray([1, 2, 3]), "allele": np.asarray(["A,C"] * 3),
            "annotation/filter": np.asarray(["PASS"] * 3),
            "annotation/info/weight": np.asarray([1., 2., 3.])}
        return fields[path][indices]


def write_manifest(directory, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(raw)
    (directory / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())


def file_binding(path):
    return dict(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def legacy_cache(tmp_path):
    root, cache = tmp_path / "population", tmp_path / "population/chr02"
    source_rows = np.asarray([8, 2, 6, 0], dtype=np.int64)
    states = np.asarray([[0, 1, 2, 3], [4, 0, 5, 1], [2, 1, 0, 0]], dtype=np.uint8)
    writer = store.Writer(cache, {"source": "anonymous-unavailable.genotype"}, source_rows, 3,
                          source_bytes=10**8)
    offsets, rows, compact_states = sparse_codec_fast.compact(states)
    writer.append(states, sparse_codec_fast.integer_counts(offsets, rows, compact_states, *states.shape))
    writer.finish()
    metadata = cache / "metadata"
    analysis = dict(annotation_catalog={"weight": "annotation/info/weight"},
                    annotation_names=["weight"], qc_path="annotation/filter")
    fields = ["position", "chromosome", "variant.id", "allele", "annotation/filter", "annotation/info/weight"]
    manifest = export_metadata(MetadataFixture(), cache, metadata, fields, **analysis)
    # Only this anonymous fixture is rewritten to reproduce the historical
    # completed export. Production compatibility never writes cache files.
    ids = np.asarray([309, 303, 307, 301], dtype=np.int64)
    np.save(metadata / "sample_ids.npy", ids, allow_pickle=False)
    (metadata / "sample_pairs.npy").unlink()
    manifest.pop("sample_pairs")
    manifest["files"].pop("sample_pairs.npy")
    manifest["sample_identifier_format"] = "positive_decimal_int64"
    manifest["files"]["sample_ids.npy"] = file_binding(metadata / "sample_ids.npy")
    write_manifest(metadata, manifest)
    np.save(root / "sample_ids.npy", ids, allow_pickle=False)
    dataset = dict(schema_version=1, sample_ids="sample_ids.npy", **analysis,
        chromosomes=[dict(name="chr2", cache_directory="chr02", metadata_directory="chr02/metadata")])
    (root / "cache_dataset.json").write_text(json.dumps(dataset))
    return root, cache, metadata, dataset, manifest, ids, source_rows


def test_legacy_dataset_keeps_exact_ids_order_and_existing_links(legacy_cache, tmp_path, monkeypatch):
    root, cache, metadata, _, _, ids, source_rows = legacy_cache
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    def no_frame(*args, **kwargs):
        raise AssertionError("dataset validation must not read genotype frames")
    monkeypatch.setattr(store.Container, "read_frame", no_frame)
    bound_root, path, manifest, entries, samples = phewas_cache._dataset(root, chromosomes="chr2")
    np.testing.assert_array_equal(samples, ids)
    assert samples.dtype == np.int64 and not samples.flags.writeable
    assert bound_root == root and path.name == "cache_dataset.json" and entries[0]["name"] == "2"
    with PortableMetadataReader(metadata, cache, legacy_numeric=True) as reader:
        np.testing.assert_array_equal(reader.sample_indices(["307", "309"]), [2, 0])
        np.testing.assert_array_equal(reader._array(reader.manifest["source_sample_rows"]), source_rows)
        assert reader.sample_axis_kind == "prepared_population"
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert after == before
    outside = tmp_path / "linked_chr"
    outside.symlink_to(cache, target_is_directory=True)
    (root / "reuse").symlink_to(outside, target_is_directory=True)
    assert phewas_cache._cache_path(root, "reuse/metadata").resolve() == metadata
    for value in ("../escape", "/absolute", "", None):
        with pytest.raises(ValueError):
            phewas_cache._cache_path(root, value)


def test_modern_reader_rejects_legacy_without_explicit_opt_in(legacy_cache):
    _, cache, metadata, _, _, _, _ = legacy_cache
    with pytest.raises(ValueError, match="FID/IID"):
        PortableMetadataReader(metadata, cache)


@pytest.mark.parametrize("problem", ["duplicate", "nonpositive", "float", "reordered"])
def test_legacy_dataset_rejects_wrong_sample_axis(legacy_cache, problem):
    root, _, _, _, _, ids, _ = legacy_cache
    altered = ids.copy()
    if problem == "duplicate":
        altered[1] = altered[0]
    elif problem == "nonpositive":
        altered[0] = 0
    elif problem == "float":
        altered = altered.astype(np.float64)
    else:
        altered = altered[::-1]
    np.save(root / "sample_ids.npy", altered, allow_pickle=False)
    with pytest.raises(ValueError, match="sample axis|sample axes"):
        phewas_cache._dataset(root)


@pytest.mark.parametrize("problem", ["qc", "catalog", "order", "duplicate_names", "missing_field"])
def test_legacy_dataset_rejects_annotation_drift(legacy_cache, problem):
    root, _, metadata, dataset, manifest, _, _ = legacy_cache
    if problem == "qc":
        dataset["qc_path"] = "annotation/wrong_qc"
    elif problem == "catalog":
        dataset["annotation_catalog"]["weight"] = "annotation/wrong_weight"
    elif problem == "order":
        dataset["annotation_names"] = []
    elif problem == "duplicate_names":
        dataset["annotation_names"] = ["weight", "weight"]
    else:
        manifest["fields"].pop("annotation/info/weight")
        write_manifest(metadata, manifest)
    (root / "cache_dataset.json").write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="annotation"):
        phewas_cache._dataset(root)


def test_portable_legacy_still_checks_hash_identity_and_original_rows(legacy_cache):
    root, cache, metadata, _, manifest, _, source_rows = legacy_cache
    np.save(metadata / "source_sample_rows.npy", source_rows[::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="checksum"):
        phewas_cache._dataset(root)
    manifest["files"]["source_sample_rows.npy"] = file_binding(metadata / "source_sample_rows.npy")
    write_manifest(metadata, manifest)
    with pytest.raises(ValueError, match="original physical sample axis"):
        PortableMetadataReader(metadata, cache, legacy_numeric=True)


def test_legacy_rejects_ambiguous_format_and_mixed_identity(legacy_cache):
    _, cache, metadata, _, manifest, _, _ = legacy_cache
    manifest["sample_identifier_format"] = "string"
    write_manifest(metadata, manifest)
    with pytest.raises(ValueError, match="FID/IID"):
        PortableMetadataReader(metadata, cache, legacy_numeric=True)
    manifest["sample_identifier_format"] = "positive_decimal_int64"
    manifest["sample_pairs"] = "sample_pairs.npy"
    write_manifest(metadata, manifest)
    with pytest.raises(ValueError, match="also declare"):
        PortableMetadataReader(metadata, cache, legacy_numeric=True)


def test_schema_two_is_not_invented_as_numeric_identity(tmp_path):
    root = tmp_path
    pairs = np.asarray([["0", "00101"], ["family", "101"]])
    np.save(root / "sample_pairs.npy", pairs)
    manifest = dict(schema_version=2, sample_pairs="sample_pairs.npy",
        annotation_catalog={}, annotation_names=[], qc_path="annotation/filter",
        chromosomes=[dict(name="1", container_directory="chr01", metadata_directory="chr01/metadata")])
    raw = json.dumps(manifest).encode()
    (root / "dataset.json").write_bytes(raw)
    (root / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="PheWAS numeric inputs"):
        phewas_cache._dataset(root)
    (root / "COMPLETE").write_text("wrong")
    with pytest.raises(ValueError, match="COMPLETE hash"):
        phewas_cache._dataset(root)


def test_legacy_chromosome_coverage_and_annotation_file(legacy_cache):
    root, _, _, dataset, _, _, _ = legacy_cache
    (root / "catalog.json").write_text(json.dumps(dataset["annotation_catalog"]))
    dataset["annotation_catalog"] = "catalog.json"
    (root / "cache_dataset.json").write_text(json.dumps(dataset))
    assert len(phewas_cache._dataset(root)[3]) == 1
    with pytest.raises(ValueError, match="absent"):
        phewas_cache._dataset(root, ["1", "2"])
    dataset["chromosomes"].append(dict(dataset["chromosomes"][0]))
    (root / "cache_dataset.json").write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="distinct"):
        phewas_cache._dataset(root)
