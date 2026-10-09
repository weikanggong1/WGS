"""Actual CPU FP64 workflow on an anonymous eight-sample CSR cache.

These small fixtures verify interfaces, coverage and native output contracts.
They do not estimate production speed or original-software scientific accuracy.
"""
import builtins
import importlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from staar_phewas.cache_runtime import sparse_codec_fast, store
from staar_phewas.cache_runtime.portable import PortableCachedGDS, export_metadata
from staar_phewas.io import load_null_model
from staar_phewas.pipeline import AnalysisOptions, PheWASPipeline
from staar_phewas.r_output import write_association_output


workflow = importlib.import_module("staar_phewas.run")


class MetadataFixture:
    n_samples = 10
    n_variants = 12

    def __init__(self):
        self.catalog = {name: "annotation/info/" + name for name in (
            "GENCODE.Category", "GENCODE.EXONIC.Category", "GENCODE.Info", "MetaSVM",
            "GeneHancer", "CAGE", "DHS", "weight")}
        self.fields = {
            "position": np.arange(1, 13, dtype=np.int64) * 10,
            "chromosome": np.asarray(["1"] * 12),
            "variant.id": np.arange(1, 13, dtype=np.int32),
            "allele": np.asarray(["A,C", "C,G", "G,T", "T,A"] * 3),
            "annotation/filter": np.asarray(["FAIL"] + ["PASS"] * 11),
            self.catalog["GENCODE.Category"]: np.asarray(["exonic"] * 6 + ["upstream"] * 3 + ["ncRNA_exonic"] * 3),
            self.catalog["GENCODE.EXONIC.Category"]: np.asarray(["stopgain"] * 3 + ["nonsynonymous SNV"] * 3 + [""] * 6),
            self.catalog["GENCODE.Info"]: np.asarray(["GENE_A"] * 9 + ["RNA_A"] * 3),
            self.catalog["MetaSVM"]: np.asarray([""] * 3 + ["D"] * 3 + [""] * 6),
            self.catalog["GeneHancer"]: np.asarray([""] * 12),
            self.catalog["CAGE"]: np.asarray([""] * 12),
            self.catalog["DHS"]: np.asarray([""] * 12),
            self.catalog["weight"]: np.linspace(0., 20., 12),
        }

    def sample_ids(self):
        return np.asarray([f"{101 + i}_{101 + i}" for i in range(10)])

    def read_field(self, path, indices):
        return self.fields[path][indices]


def dataset(tmp_path):
    root = tmp_path / "population"
    cache = root / "chr01"
    native = MetadataFixture()
    # Retain a reordered eight-person subset of the larger native source.
    physical = np.asarray([8, 2, 7, 1, 6, 0, 5, 3], dtype=np.int64)
    raw = np.zeros((12, 8), dtype=np.uint8)
    for variant in range(12):
        raw[variant, variant % 8] = 1
        if variant % 3 == 0:
            raw[variant, (variant + 3) % 8] = 1
    offsets, rows, states = sparse_codec_fast.compact(raw)
    counts = sparse_codec_fast.integer_counts(offsets, rows, states, *raw.shape)
    writer = store.Writer(cache, {"source": "anonymous-unavailable.gds"}, physical, 12, source_bytes=10**8)
    writer.append(raw, counts)
    writer.finish()
    export_metadata(native, cache, cache / "metadata", list(native.fields), block_size=3,
                    annotation_catalog=native.catalog, annotation_names=["weight"])
    cache_ids = np.asarray([101 + int(i) for i in physical], dtype=np.int64)
    np.save(root / "sample_ids.npy", cache_ids)
    (cache / "metadata/genes.json").write_text(json.dumps([
        dict(kind="coding", gene_name="GENE_A", start=1, end=90),
        dict(kind="noncoding", gene_name="GENE_A"),
        dict(kind="ncrna", gene_name="RNA_A")]))
    (cache / "metadata/promoters.json").write_text("[]")
    (root / "cache_dataset.json").write_text(json.dumps(dict(schema_version=1, sample_ids="sample_ids.npy",
        annotation_catalog=native.catalog, annotation_names=["weight"], qc_path="annotation/filter",
        chromosomes=[dict(name="1", container_directory="chr01", metadata_directory="chr01/metadata",
            gene_catalog="chr01/metadata/genes.json", promoter_intervals="chr01/metadata/promoters.json")])))
    phenotype, covariate = tmp_path / "phenotypes.csv", tmp_path / "covariates.csv"
    y = [2., 8., 3., 7., 1., 9., 4., 6.]
    phenotype.write_text("eid,trait_a\n" + "".join(f"{identifier},{value}\n" for identifier, value in zip(cache_ids, y)))
    covariate.write_text("eid,adjustment\n" + "".join(f"{identifier},{index}\n" for index, identifier in enumerate(cache_ids)))
    return root, cache, phenotype, covariate


def no_native_sdk(monkeypatch):
    original_import = builtins.__import__
    def check_import(name, *args, **kwargs):
        if name == "pygds" or name.startswith("pygds."):
            raise AssertionError("public workflow imported native SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", check_import)
    original_stat = Path.stat
    def check_stat(path, *args, **kwargs):
        if path.suffix == ".gds":
            raise AssertionError("public workflow accessed original GDS")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", check_stat)


def assert_native_values(saved, expected):
    """Compare the complete parsed gene object, including labels and empty masks."""
    if isinstance(expected, dict):
        assert isinstance(saved, dict) and list(saved) == list(expected)
        for key in expected:
            assert_native_values(saved[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(saved) == len(expected)
        for left, right in zip(saved, expected):
            assert_native_values(left, right)
    elif isinstance(expected, np.ndarray):
        assert isinstance(saved, np.ndarray) and saved.shape == expected.shape
        for left, right in zip(saved.flat, expected.flat):
            assert_native_values(left, right)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(saved, expected, rtol=1e-12, atol=1e-12)
    else:
        assert saved == expected


def test_actual_worker_cache_only_single_coding_noncoding_ncrna_native_coverage(tmp_path, monkeypatch):
    import rdata
    root, cache, phenotype, covariate = dataset(tmp_path)
    no_native_sdk(monkeypatch)
    output = tmp_path / "run"
    plan_path = workflow.prepare_run(phenotype, covariate, root, output_directory=output,
        matmul_mode="fp64", null_fit_mode="fp64", analyses=["individual", "coding", "noncoding", "ncrna"],
        single_mac_cutoff=1, single_group_variants=2, single_output_groups=1,
        host_memory_reserve_gib=2.,
        analysis_options=dict(rare_maf_cutoff=.49, variant_type="variant"))
    plan = json.loads(plan_path.read_text())
    assert plan["task_count"] == 4
    workflow._run_worker(str(plan_path), "cpu")
    with sqlite3.connect(output / "jobs.sqlite") as connection:
        rows = connection.execute("SELECT status,payload,report,error FROM jobs ORDER BY id").fetchall()
    failures = [error for status, _, _, error in rows if status != "completed"]
    assert not failures, failures
    assert {json.loads(payload)["kind"] for _, payload, _, _ in rows} == {"individual", "coding", "noncoding", "ncrna"}
    for _, payload, raw_report, _ in rows:
        job, report = json.loads(payload), json.loads(raw_report)
        assert report["association_rows"] > 0
        for file in report["outputs"]:
            values = rdata.read_rda(file["path"])
            expected = {"individual": "results_individual_analysis", "coding": "results_coding",
                        "noncoding": "results_noncoding", "ncrna": "results_ncRNA"}[job["kind"]]
            assert list(values) == [expected]
    individual = json.loads(rows[0][2])
    assert individual["association_rows"] == 11  # One source FAIL variant is excluded.
    assert len(individual["outputs"]) > 1
    frames = [rdata.read_rda(item["path"])["results_individual_analysis"] for item in individual["outputs"]]
    with PortableCachedGDS(cache, device="cpu") as reader:
        model = load_null_model(plan["phenotypes"][0]["path"], device="cpu", matmul_mode="fp64")
        pipeline = PheWASPipeline(reader, [model],
            annotation_catalog=reader.manifest["analysis"]["annotation_catalog"], annotation_names=["weight"],
            options=AnalysisOptions(**plan["input_identity"]["analysis_options"]))
        expected = pipeline.individual("1", mac_cutoff=1, subset_variants_num=2)[0]
        # The public worker and direct mature pipeline agree on every native
        # gene cell, rather than only producing readable files with row counts.
        for _, raw_job, raw_report, _ in rows[1:]:
            job, report = json.loads(raw_job), json.loads(raw_report)
            arguments = dict(job["arguments"])
            if job["kind"] == "noncoding":
                arguments["promoter_intervals"] = []
            computed = getattr(pipeline, job["kind"])(chromosome="1", **arguments)
            reference = tmp_path / f"direct_{job['kind']}.Rdata"
            write_association_output(reference, computed, kind=job["kind"], layout="base")
            assert_native_values(rdata.read_rda(report["outputs"][0]["path"]), rdata.read_rda(reference))
    # Parts preserve original global row names and actual numerical values.
    observed_rows = [row for frame in frames for _, row in frame.iterrows()]
    observed_index = [str(index) for frame in frames for index in frame.index]
    assert observed_index == [str(index) for index in expected.row_names]
    assert len(observed_rows) == len(expected)
    for saved, computed in zip(observed_rows, expected):
        for key, value in computed.items():
            if key in ("REF", "ALT"):
                assert saved[key] == value
            else:
                assert float(saved[key]) == float(value)
    single_index = json.loads((output / "results/trait_0001/chr1/individual/job_000001/index.private.json").read_text())
    assert single_index["rows"] == 11
    assert single_index["final_factor_levels"] == expected.factor_levels
    for frame in frames:
        for field in ("REF", "ALT"):
            levels = frame[field].cat.categories.tolist()
            assert levels == expected.factor_levels[field][:len(levels)]
