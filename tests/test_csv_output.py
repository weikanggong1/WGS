import csv
import math

import numpy as np
import pytest

from fudan_wgs_toolkit.csv_output import GENE_COLUMNS, SINGLE_COLUMNS, write_association_batch


def gene(name, category, p=0.01):
    return {"Gene name": name, "Chr": 21, "Category": category, "#SNV": 3,
            "cMAC": 42.0, "WGS-O": p}


def read(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def test_gene_batches_retain_job_then_category_order(tmp_path):
    first = {"a": [[gene("gene_first", "a")]], "b": [[gene("gene_first", "b")]]}
    second = {"a": [[gene("gene_second", "a")]], "b": [[gene("gene_second", "b")]]}
    path = tmp_path / "coding.csv"
    proof = write_association_batch(path, [first, second], kind="coding")
    assert [(r[0], r[2]) for r in read(path)[1:]] == [
        ("gene_first", "a"), ("gene_first", "b"), ("gene_second", "a"), ("gene_second", "b")]
    assert proof["rows"] == 4 and proof["numeric_roundtrip_passed"]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("kind,expected", [("coding", GENE_COLUMNS), ("noncoding", GENE_COLUMNS),
    ("ncrna", GENE_COLUMNS), ("individual", SINGLE_COLUMNS)])
def test_empty_results_keep_full_header(tmp_path, kind, expected):
    result = {"empty": [[]]} if kind in ("coding", "noncoding") else [[]]
    path = tmp_path / f"{kind}.csv"
    proof = write_association_batch(path, [result], kind=kind)
    assert read(path) == [list(expected)]
    assert proof["rows"] == 0 and proof["empty_schema"] == "default_unannotated"


def test_single_tiny_p_and_signed_zero_roundtrip(tmp_path):
    row = dict(zip(SINGLE_COLUMNS, [21, 1234, "A", "C", .01, .01, 100000,
        1e-300, 300.0, -0.0, .00012345678901234567, .002, .003]))
    path = tmp_path / "single.csv"
    proof = write_association_batch(path, [[[row]]], kind="individual")
    values = dict(zip(read(path)[0], read(path)[1]))
    assert float(values["pvalue"]) == row["pvalue"]
    assert float(values["pvalue_log10"]) == 300.0
    assert math.copysign(1, float(values["Score"])) == -1
    assert proof["maximum_logp_error"] == 0 and proof["maximum_stored_logp_error"] == 0


def test_missense_extra_columns_union_and_empty_cells(tmp_path):
    first, second = gene("first", "other"), gene("second", "missense")
    second["SKAT(1,25)-Disruptive"] = np.float64(0.12345678901234567)
    path = tmp_path / "coding.csv"
    proof = write_association_batch(path, [{"a": [[first]], "b": [[second]]}], kind="coding")
    table = read(path)
    assert table[0][-1] == "SKAT(1,25)-Disruptive" and table[1][-1] == ""
    assert float(table[2][-1]) == second["SKAT(1,25)-Disruptive"]
    assert proof["rows"] == 2


def test_column_order_collision_does_not_replace_completed_output(tmp_path):
    path = tmp_path / "coding.csv"
    path.write_text("completed\n")
    with pytest.raises(ValueError, match="column orders"):
        write_association_batch(path, [{"a": [[{"x": 1, "y": 2}]],
            "b": [[{"y": 2, "x": 1}]]}], kind="coding")
    assert path.read_text() == "completed\n"


def test_noncoding_metadata_and_excluded_columns(tmp_path):
    row = gene("example", "UTR")
    row["unused_metadata"] = "private"
    path = tmp_path / "noncoding.csv"
    proof = write_association_batch(path, [[[row]]], kind="noncoding", exclude_columns=["unused_metadata"])
    assert "unused_metadata" not in read(path)[0]
    assert read(path)[1][1:4] == ["21", "UTR", "3"]
    assert proof["excluded_columns_absent"]


def test_single_jobs_require_separate_output_files(tmp_path):
    with pytest.raises(ValueError, match="separate CSV"):
        write_association_batch(tmp_path / "single.csv", [[[]], [[]]], kind="individual")
