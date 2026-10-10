"""Lossless family/individual alignment checks, not production benchmarks."""
import json
import hashlib
import numpy as np
import pytest

from fudan_wgs_toolkit.identity import sample_keys, validate_sample_pairs, pairs_from_keys
from fudan_wgs_toolkit.run import read_csv_inputs, aligned_phenotype, _dataset


def test_full_pairs_preserve_zeroes_and_do_not_collide():
    pairs = np.asarray([["0", "001"], ["family", "001"], ["a_b", "c"], ["a", "b_c"], ["00", "1_1"]])
    keys = sample_keys(pairs)
    assert len(set(keys)) == len(pairs)
    np.testing.assert_array_equal(pairs_from_keys(keys), pairs)
    assert keys[0] == '["0","001"]'


@pytest.mark.parametrize("pairs", [[["0", ""]], [[" ", "1"]], [["f", "a b"]],
                                  [["f", "a\n"]], [["f", "1"], ["f", "1"]]])
def test_invalid_or_duplicate_pairs_are_rejected(pairs):
    with pytest.raises(ValueError):
        validate_sample_pairs(pairs)


def test_numeric_values_are_not_silently_normalized():
    with pytest.raises(ValueError, match="strings"):
        validate_sample_pairs(np.asarray([[0, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="canonical"):
        pairs_from_keys(['["0", "001"]'])


def csv_inputs(tmp_path):
    phenotype = tmp_path / "phenotypes.csv"
    covariate = tmp_path / "covariates.csv"
    phenotype.write_text("IID,FID,trait_a,trait_b\n001,f,7,NA\n001,0,3,9\n003,0,5,8\n004,0,NA,4\n005,0,11,2\n006,0,13,1\n")
    covariate.write_text("FID,IID,adjustment\n0,006,6\n0,005,5\nf,001,1\n0,004,4\n0,003,3\n0,001,2\n")
    return phenotype, covariate


def test_matching_uses_complete_pairs_and_trait_specific_missingness(tmp_path):
    inputs = read_csv_inputs(*csv_inputs(tmp_path))
    cached = np.asarray([["0", "005"], ["f", "001"], ["0", "003"], ["0", "001"], ["0", "004"], ["0", "006"]])
    first = aligned_phenotype(*inputs, cached, "trait_a")
    second = aligned_phenotype(*inputs, cached, "trait_b")
    assert pairs_from_keys(first[0]).tolist() == cached[[0, 1, 2, 3, 5]].tolist()
    assert pairs_from_keys(second[0]).tolist() == cached[[0, 2, 3, 4, 5]].tolist()
    np.testing.assert_array_equal(first[1], [11, 7, 5, 3, 13])
    np.testing.assert_array_equal(first[2][:, 1], [5, 1, 3, 2, 6])
    assert first[-1]["analysis_samples"] == second[-1]["analysis_samples"] == 5


def test_missing_covariate_excludes_each_affected_outcome(tmp_path):
    phenotype, covariate = csv_inputs(tmp_path)
    covariate.write_text(covariate.read_text().replace("0,005,5", "0,005,NaN"))
    inputs = read_csv_inputs(phenotype, covariate)
    cached = inputs[0].sample_pairs
    for name in ("trait_a", "trait_b"):
        selected = aligned_phenotype(*inputs, cached, name)
        assert selected[-1]["analysis_samples"] == 4
        assert ["0", "005"] not in pairs_from_keys(selected[0]).tolist()


def test_unused_covariate_missingness_does_not_exclude_other_outcomes(tmp_path):
    phenotype, covariate = csv_inputs(tmp_path)
    covariate.write_text("FID,IID,adjustment,optional\n0,006,6,1\n0,005,5,NaN\nf,001,1,2\n0,004,4,3\n0,003,3,4\n0,001,2,5\n")
    inputs = read_csv_inputs(phenotype, covariate)
    pairs = inputs[0].sample_pairs
    selected = aligned_phenotype(*inputs, pairs, "trait_a", covariate_names=["adjustment"])
    all_columns = aligned_phenotype(*inputs, pairs, "trait_a")
    assert selected[-1]["analysis_samples"] == 5
    assert all_columns[-1]["analysis_samples"] == 4
    assert selected[3] == ("Intercept", "adjustment")
    for columns in (["unknown"], ["adjustment", "adjustment"], "adjustment"):
        with pytest.raises(ValueError):
            aligned_phenotype(*inputs, pairs, "trait_a", covariate_names=columns)


def test_covariate_identity_only_adds_one_intercept(tmp_path):
    phenotype, covariate = csv_inputs(tmp_path)
    covariate.write_text("FID,IID\n0,001\nf,001\n0,003\n0,004\n0,005\n0,006\n")
    inputs = read_csv_inputs(phenotype, covariate)
    selected = aligned_phenotype(*inputs, inputs[0].sample_pairs, "trait_a")
    assert selected[2].shape == (5, 1)
    np.testing.assert_array_equal(selected[2], np.ones((5, 1)))


def test_duplicate_pair_rejected_but_iid_in_two_families_allowed(tmp_path):
    phenotype, covariate = csv_inputs(tmp_path)
    read_csv_inputs(phenotype, covariate)
    phenotype.write_text(phenotype.read_text() + "001,f,2,3\n")
    with pytest.raises(ValueError, match="duplicate FID/IID"):
        read_csv_inputs(phenotype, covariate)


def test_prepared_dataset_binds_pair_and_key_arrays(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    pairs = np.asarray([["0", "001"], ["f", "001"]])
    np.save(root / "sample_pairs.npy", pairs)
    np.save(root / "sample_ids.npy", sample_keys(pairs))
    raw=json.dumps(dict(schema_version=2,
        annotation_catalog={},annotation_names=[],qc_path="annotation/filter",
        sample_pairs="sample_pairs.npy", sample_ids="sample_ids.npy",
        chromosomes=[dict(name="21", container_directory="chr21", metadata_directory="chr21/metadata")])).encode()
    (root/"dataset.json").write_bytes(raw)
    (root/"COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    returned = _dataset(root, ["21"])
    np.testing.assert_array_equal(returned[-1], pairs)
    np.save(root / "sample_ids.npy", sample_keys(pairs[::-1]))
    with pytest.raises(ValueError, match="disagree"):
        _dataset(root)


@pytest.mark.parametrize("action", ["missing", "corrupt", "changed_dataset"])
def test_prepared_dataset_requires_bound_completion_hash(tmp_path,action):
    root=tmp_path/"prepared"
    root.mkdir()
    raw=json.dumps(dict(schema_version=2,annotation_catalog={},annotation_names=[],qc_path="annotation/filter")).encode()
    (root/"dataset.json").write_bytes(raw)
    if action != "missing":
        (root/"COMPLETE").write_text("0"*64 if action=="corrupt" else hashlib.sha256(raw).hexdigest())
    if action=="changed_dataset":
        (root/"dataset.json").write_bytes(raw+b"\n")
    with pytest.raises(ValueError,match="COMPLETE"):
        _dataset(root)


@pytest.mark.parametrize("missing", ["annotation_catalog", "annotation_names", "qc_path"])
def test_prepared_dataset_requires_explicit_scientific_mapping(tmp_path,missing):
    root=tmp_path/"prepared"
    root.mkdir()
    dataset=dict(schema_version=2,annotation_catalog={},annotation_names=[],qc_path="annotation/filter")
    del dataset[missing]
    raw=json.dumps(dataset).encode()
    (root/"dataset.json").write_bytes(raw)
    (root/"COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError,match="requires annotation_catalog"):
        _dataset(root)


def test_saved_null_model_preserves_complete_pair_identity(tmp_path):
    from fudan_wgs_toolkit.null_model import fit_gaussian_null
    from fudan_wgs_toolkit.io import load_null_model, save_null_model
    pairs = np.asarray([["0", "001"], ["f", "001"], ["0", "003"], ["0", "004"], ["0", "005"], ["0", "006"]])
    model = fit_gaussian_null([1., 3., 2., 6., 4., 7.], sample_ids=sample_keys(pairs), device="cpu")
    model.sample_pairs = pairs
    path = tmp_path / "model.npz"
    save_null_model(model, path)
    restored = load_null_model(path, device="cpu", matmul_mode="tf32")
    np.testing.assert_array_equal(restored.sample_pairs, pairs)
    np.testing.assert_array_equal(restored.sample_ids, sample_keys(pairs))
    model.sample_pairs = pairs[::-1]
    with pytest.raises(ValueError, match="disagree"):
        save_null_model(model, tmp_path / "changed.npz")
