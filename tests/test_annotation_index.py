"""Annotation identity and bounded-reader unit checks; not performance data."""
from types import SimpleNamespace
import numpy as np
import pytest

from fudan_wgs_toolkit.pipeline import PheWASPipeline, AnalysisOptions
from fudan_wgs_toolkit.masks import VariantAnnotations, noncoding_masks, ncRNA_mask, promoter_overlaps


class MetadataFile:
    def __init__(self):
        self.n_samples = 1
        self.n_variants = 9
        self.calls = []
        self.data = {
            "position": [10, 20, 30, 30, 50, 60, 70, 80, 90],
            "qc": ["PASS"]*8+["FAIL"],
            "chromosome": ["21", "chr21", "21", "21", "21", "21", "21", "22", "21"],
            "GENCODE.Category": ["upstream", "downstream", "UTR3", "ncRNA_exonic", "intergenic", "UTR5", "ncRNA_splicing", "UTR5", "UTR3"],
            "GENCODE.Info": ["A,B", "B,A", "A(detail)", "A,B,C,D(detail);rest", "A-rest", "A(detail)", "B(detail);rest", "A(detail)", "A(detail)"],
            "GeneHancer": ["", "", "a=b=c=A;detail", "", "a=b=c=B;detail", "", "", "", ""],
            "CAGE": ["", "yes", "yes", "", "yes", "yes", "", "yes", "yes"],
            "DHS": ["", "", "yes", "yes", "", "", "yes", "yes", "yes"],
            "ref": ["A"]*9, "alt": ["G"]*5+["G,T", "C", "G", "G"],
        }
    def sample_ids(self): return np.asarray(["sample"])
    def sample_indices(self, ids): return np.asarray([0])
    def read_field(self, path, indices=None):
        self.calls.append(path)
        values = np.asarray(self.data[path], dtype=object)
        return values if indices is None else values[indices]
    def read_ref_alt(self, indices):
        return self.read_field("ref", indices), self.read_field("alt", indices)


def make_pipeline():
    genotype = MetadataFile()
    model = SimpleNamespace(n=1, sample_ids=np.asarray(["sample"]), family="gaussian", n_pheno=1, use_spa=False)
    names = ("GENCODE.Category", "GENCODE.Info", "GeneHancer", "CAGE", "DHS")
    pipeline = PheWASPipeline(genotype, [model], qc_path="qc", annotation_catalog={name:name for name in names},
                             options=AnalysisOptions(annotation_block_size=3))
    return pipeline, genotype


def test_index_matches_selectors_and_utr_does_not_read_signal_fields():
    pipeline, genotype = make_pipeline()
    index = pipeline.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    assert index.indices("A", "UTR").tolist() == [2]
    assert not set(("CAGE", "DHS", "GeneHancer")) & set(genotype.calls)
    calls = len(genotype.calls)
    assert pipeline.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False) is index
    assert len(genotype.calls) == calls
    intervals = [("21", 15, 45), ("chr21", 40, 75)]
    index = pipeline.prepare_annotation_index("21", promoter_intervals=intervals)
    a = VariantAnnotations(genotype.data["position"], genotype.data["qc"],
                           {name:genotype.data[name] for name in pipeline.annotation_catalog},
                           ref=genotype.data["ref"], alt=genotype.data["alt"], chromosome=genotype.data["chromosome"])
    overlaps = promoter_overlaps(a.position, a.chromosome, intervals)
    for gene in ("A", "B", "C", "D"):
        expected = noncoding_masks(a, gene, promoter_overlap=overlaps, chromosome="21")
        expected["ncRNA"] = ncRNA_mask(a, gene, chromosome="21")
        for category, rows in expected.items():
            np.testing.assert_array_equal(index.indices(gene, category), rows)
    with pytest.raises(ValueError, match="different promoter"):
        pipeline.prepare_annotation_index("21", promoter_intervals=[("21", 1, 100)])
