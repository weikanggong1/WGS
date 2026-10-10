"""Actual CPU FP64 workflow on an anonymous eight-sample CSR cache.

These small fixtures verify interfaces, coverage and native output contracts.
They do not estimate production speed or original-software scientific accuracy.
"""
import builtins
import importlib
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast, store
from fudan_wgs_toolkit.cache_runtime.portable import PortableGenotypeReader, export_metadata
from fudan_wgs_toolkit.io import load_null_model
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
from fudan_wgs_toolkit.identity import sample_keys


workflow = importlib.import_module("fudan_wgs_toolkit.run")


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
        return sample_keys(self.sample_pairs())

    def sample_pairs(self):
        return np.asarray([["0", f"{101 + i}_{101 + i}"] for i in range(10)])

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
    writer = store.Writer(cache, {"source": "anonymous-unavailable.genotype"}, physical, 12, source_bytes=10**8)
    writer.append(raw, counts)
    writer.finish()
    export_metadata(native, cache, cache / "metadata", list(native.fields), block_size=3,
                    annotation_catalog=native.catalog, annotation_names=["weight"])
    pairs = native.sample_pairs()[physical]
    cache_ids = sample_keys(pairs)
    np.save(root / "sample_ids.npy", cache_ids)
    np.save(root / "sample_pairs.npy", pairs)
    (cache / "metadata/genes.json").write_text(json.dumps([
        dict(kind="coding", gene_name="GENE_A", start=1, end=90),
        dict(kind="noncoding", gene_name="GENE_A"),
        dict(kind="ncrna", gene_name="RNA_A")]))
    (cache / "metadata/promoters.json").write_text("[]")
    raw_dataset=json.dumps(dict(schema_version=2, sample_ids="sample_ids.npy", sample_pairs="sample_pairs.npy",
        annotation_catalog=native.catalog, annotation_names=["weight"], qc_path="annotation/filter",
        chromosomes=[dict(name="1", container_directory="chr01", metadata_directory="chr01/metadata",
            gene_catalog="chr01/metadata/genes.json", promoter_intervals="chr01/metadata/promoters.json")])).encode()
    (root/"dataset.json").write_bytes(raw_dataset)
    (root/"COMPLETE").write_text(hashlib.sha256(raw_dataset).hexdigest())
    phenotype, covariate = tmp_path / "phenotypes.csv", tmp_path / "covariates.csv"
    y = [2., 8., 3., 7., 1., 9., 4., 6.]
    phenotype.write_text("FID,IID,trait_a\n" + "".join(f"{family},{individual},{value}\n" for (family, individual), value in zip(pairs, y)))
    covariate.write_text("FID,IID,adjustment\n" + "".join(f"{family},{individual},{index}\n" for index, (family, individual) in enumerate(pairs)))
    return root, cache, phenotype, covariate
