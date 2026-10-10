"""CPU format tests; fixtures are not performance or scientific benchmarks."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from rdata.missing import R_FLOAT_NA

from staar_phewas.r_output import (
    RAttributed, RDataFrame, RFactor, RMatrix, write_association_batch,
    write_association_output, write_r_object,
)
from staar_phewas.results import TraitRows


_PATH = Path(__file__).parents[1] / "torchstaar_phewas" / "export_results.py"
_SPEC = importlib.util.spec_from_file_location("staged_export_results", _PATH)
exporter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exporter
_SPEC.loader.exec_module(exporter)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def save(path, value):
    write_r_object(path, value, object_name="association")
    return path


def manifest(*paths, name="trait example", kind="individual", **options):
    return [{"name": name, "native_files": [{"path": str(path), "kind": kind, **options} for path in paths]}]


def one_file(report):
    return report["traits"][0]["files"][0]


def single_rows():
    rows = []
    for index, probability in enumerate((0.0, float.fromhex("0x0.0000000000001p-1022"), 0.05)):
        rows.append(dict(zip(exporter._SINGLE_COLUMNS, (
            21.0, 300.0 - index, ("T", "A", "T")[index], ("A", "G", "C")[index],
            0.12345678901234567, 0.01, 1200, probability,
            (1000.0, -math.log10(probability), -math.log10(probability))[index] if index else 1000.0,
            -0.0, 0.12345678901234567, -1.2345678901234567, 2.345678901234567,
        ))))
    return TraitRows(rows, row_names=[8, 3, 5], factor_levels={"REF": ["A", "T"], "ALT": ["C", "G", "A"]})


def gene_row(label, *, disruptive=False):
    row = dict(zip(exporter._GENE_COLUMNS, (label, 21, "mask", 3, 12.0) + (0.01,) * 14))
    if disruptive:
        for name in ("SKAT(1,25)", "SKAT(1,1)", "Burden(1,25)", "Burden(1,1)", "ACAT-V(1,25)", "ACAT-V(1,1)"):
            row[name + "-Disruptive"] = 0.02
    return row


def test_mature_single_writer_preserves_all_columns_factors_order_and_tiny_values(tmp_path):
    source = tmp_path / "single.Rdata"
    rows = single_rows()
    write_association_output(source, [rows], kind="individual", layout="base")
    report = exporter.export_results(manifest(source), results_directory=tmp_path / "results")
    entry = one_file(report)
    csv_rows = read_csv(entry["csv_file"])
    assert csv_rows[0] == list(exporter._SINGLE_COLUMNS)
    assert [float(row[1]) for row in csv_rows[1:]] == [300, 299, 298]
    assert [row[2:4] for row in csv_rows[1:]] == [["T", "A"], ["A", "G"], ["T", "C"]]
    for original, row in zip(rows, csv_rows[1:]):
        for name in set(exporter._SINGLE_COLUMNS) - {"REF", "ALT"}:
            assert float(row[csv_rows[0].index(name)]) == original[name]
    assert math.copysign(1, float(csv_rows[1][9])) == -1
    assert csv_rows[1][7:9] == ["0", "1000"]
    assert sha(source) == sha(entry["native_file"]) == entry["source_sha256"] == entry["native_sha256"]
    assert report["association_computed"] is False
    assert entry["p_cells"] == entry["logp_cells"] == 3
    assert entry["maximum_logp_error"] == entry["maximum_stored_logp_error"] == 0
    assert entry["numeric_roundtrip_passed"] and entry["pvalue_log10_present"]


def test_repeated_named_category_slots_and_null_slots_survive_batch_writer(tmp_path):
    source = tmp_path / "coding.Rdata"
    jobs = [
        {"mask": [[gene_row("row A")]], "empty": [[]]},
        {"mask": [[gene_row("row B")]], "empty": [[]]},
    ]
    write_association_batch(source, jobs, kind="coding", layout="base")
    entry = one_file(exporter.export_results(manifest(source, kind="coding"), results_directory=tmp_path / "out"))
    assert [row[0] for row in read_csv(entry["csv_file"])[1:]] == ["row A", "row B"]
    assert [slot["names"] for slot in entry["slots"]] == [["mask"], ["empty"], ["mask"], ["empty"]]
    assert [slot["ordinal_path"] for slot in entry["slots"]] == [[1], [2], [3], [4]]
    assert entry["null_slots"] == 2 and entry["rows"] == 2
    assert entry["p_cells"] == 28


def test_heterogeneous_masks_keep_first_seen_headers_and_missing_extra_cells(tmp_path):
    source = tmp_path / "coding.rds"
    first, second = gene_row("row A"), gene_row("row B", disruptive=True)
    matrices = [RMatrix([list(row.values())], list(row), mode="list") for row in (first, second)]
    save(source, RAttributed(matrices, {"names": np.asarray(["mask", "mask"])}))
    entry = one_file(exporter.export_results(manifest(source, kind="coding"), results_directory=tmp_path / "out"))
    rows = read_csv(entry["csv_file"])
    assert len(rows[0]) == 25 and rows[0][:19] == list(first)
    assert rows[1][19:] == [""] * 6 and [float(value) for value in rows[2][19:]] == [0.02] * 6
    assert entry["rows"] == 2 and entry["checked_cells"] == 50


@pytest.mark.parametrize("mode,values,expected", [
    ("double", [[1.2345678901234567, -0.0], [np.inf, np.nan]], [["1.2345678901234567", "-0"], ["Inf", "NaN"]]),
    ("integer", [[1, 2], [3, 4]], [["1", "2"], ["3", "4"]]),
    ("character", [["甲,乙", '"quoted"'], ["newline\ncell", ""]], [["甲,乙", '"quoted"'], ["newline\ncell", ""]]),
    ("list", [[None, True], [np.array([0.01]), np.array([12])]], [["", "TRUE"], ["0.01", "12"]]),
])
def test_matrix_modes_scalar_cells_and_csv_quoting(tmp_path, mode, values, expected):
    source = save(tmp_path / "matrix.rds", RMatrix(values, ["first", "second"], ["r2", "r1"], mode=mode))
    entry = one_file(exporter.export_results(manifest(source, kind="ncrna"), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"]) == [["first", "second"]] + expected
    assert entry["roundtrip_passed"]


def test_duplicate_column_names_are_not_dropped(tmp_path):
    source = save(tmp_path / "columns.rds", RMatrix([[0.1, 0.2, 0.3]], ["x", "x", "pvalue"], mode="double"))
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"])[0] == ["x", "x", "pvalue"]
    assert [float(value) for value in read_csv(entry["csv_file"])[1]] == [0.1, 0.2, 0.3]


def test_character_probability_cells_are_audited_without_coercing_metadata(tmp_path):
    values = [["name", "0.0100", "2.0000"], ["other", "1e-300", "300"], ["third", "NaN", "Inf"],
              ["fourth", "NA", "NA"], ["last", "0", "1000"]]
    source = save(tmp_path / "chars.rds", RMatrix(values, ["metadata", "pvalue", "pvalue_log10"], mode="character"))
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"])[1:] == values
    assert entry["numeric_cells"] == 0 and entry["p_cells"] == entry["logp_cells"] == 4
    assert entry["invalid_p_cells"] == entry["nonfinite_logp_cells"] == 1
    assert entry["unparsed_probability_cells"] == 2
    assert entry["maximum_logp_error"] == entry["maximum_stored_logp_error"] == 0


def test_integer_probability_cells_are_audited(tmp_path):
    source = save(tmp_path / "integers.rds", RMatrix([[0, 1000], [1, 0]], ["pvalue", "pvalue_log10"], mode="integer"))
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert entry["p_cells"] == entry["logp_cells"] == 2
    assert entry["maximum_logp_error"] == entry["maximum_stored_logp_error"] == 0


def test_named_scalar_statistics_vector_remains_one_row(tmp_path):
    source = save(tmp_path / "vector.rds", RAttributed(np.array([0.01, 2.0]), {"names": np.array(["pvalue", "pvalue_log10"])}))
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"]) == [["pvalue", "pvalue_log10"], ["0.01", "2"]]


def test_zero_row_matrix_uses_its_actual_header(tmp_path):
    source = save(tmp_path / "zero.rds", RMatrix(np.empty((0, 2)), ["pvalue", "pvalue_log10"]))
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"]) == [["pvalue", "pvalue_log10"]]
    assert entry["rows"] == 0 and entry["empty_schema"] is None


def test_masked_factor_integer_and_float_missing_cells_roundtrip(tmp_path):
    factor = RAttributed(np.ma.array([1, 2], mask=[False, True], dtype=np.int32),
                        {"levels": np.array(["α", "β"]), "class": "factor"}, object_flag=True)
    columns = [factor, np.ma.array([7, 8], mask=[False, True], dtype=np.int32),
               np.array([0.01, R_FLOAT_NA], dtype=np.float64)]
    frame = RAttributed(columns, {"names": np.array(["text", "count", "pvalue"]),
                                "row.names": np.array([2, 1], dtype=np.int32), "class": "data.frame"}, object_flag=True)
    source = save(tmp_path / "masked.Rdata", frame)
    entry = one_file(exporter.export_results(manifest(source), results_directory=tmp_path / "out"))
    assert read_csv(entry["csv_file"]) == [["text", "count", "pvalue"], ["α", "7", "0.01"], ["", "", ""]]


@pytest.mark.parametrize("value", [None, [], RAttributed([None, None], {"names": np.array(["empty", "empty"])})])
@pytest.mark.parametrize("kind", ["individual", "coding", "noncoding", "ncrna"])
def test_null_only_output_has_readable_header_without_fake_rows(tmp_path, value, kind):
    source = save(tmp_path / "empty.Rdata", value)
    entry = one_file(exporter.export_results(manifest(source, kind=kind), results_directory=tmp_path / "out"))
    expected = exporter._SINGLE_COLUMNS if kind == "individual" else exporter._GENE_COLUMNS
    assert read_csv(entry["csv_file"]) == [list(expected)]
    assert entry["rows"] == entry["p_cells"] == 0 and entry["empty_schema"] == "default_unannotated"


def test_empty_schema_override_and_column_exclusion(tmp_path):
    source = save(tmp_path / "empty.Rdata", None)
    entry = one_file(exporter.export_results(manifest(source, empty_columns=["first", "omit"]),
                        results_directory=tmp_path / "out", exclude_columns=["omit"]))
    assert read_csv(entry["csv_file"]) == [["first"]] and entry["empty_schema"] == "provided"


def test_excludes_only_explicit_columns_without_appending_trait_metadata(tmp_path):
    columns = {"pvalue": np.array([0.01]), "extra_key_a": np.array(["secret"]),
               "pvalue_log10": np.array([2.0]), "extra_key_b": np.array(["other"])}
    source = save(tmp_path / "data.Rdata", RDataFrame(columns, [9]))
    traits = manifest(source)
    traits[0]["exclude_columns"] = ["extra_key_b", "already absent"]
    entry = one_file(exporter.export_results(traits, results_directory=tmp_path / "out", exclude_columns=["extra_key_a"]))
    assert read_csv(entry["csv_file"]) == [["pvalue", "pvalue_log10"], ["0.01", "2"]]
    assert entry["excluded_columns_absent"] and "name" not in entry["columns"]
    assert Path(entry["native_file"]).read_bytes() == source.read_bytes()


def test_all_eighteen_files_including_four_individual_segments_are_saved(tmp_path):
    entries = []
    for index, kind in enumerate(["individual"] * 4 + ["coding"] * 5 + ["noncoding"] * 8 + ["ncrna"]):
        path = save(tmp_path / f"segment_{index:02}.Rdata", None)
        entries.append({"path": str(path), "kind": kind})
    report = exporter.export_results([{"name": "表型 示例", "native_files": entries}], results_directory=tmp_path / "results")
    assert report["native_files"] == report["csv_files"] == 18
    destination = tmp_path / "results" / "表型 示例"
    assert len(list(destination.iterdir())) == 36
    assert len([entry for entry in report["traits"][0]["files"] if entry["kind"] == "individual"]) == 4
    assert all(Path(entry["csv_file"]).is_file() for entry in report["traits"][0]["files"])


def test_identical_repeat_reuses_hashes_and_preserves_mtime(tmp_path):
    source = save(tmp_path / "x.rds", RMatrix([[0.01]], ["pvalue"]))
    traits, output = manifest(source), tmp_path / "out"
    first = exporter.export_results(traits, results_directory=output)
    entry = one_file(first)
    before = {key: Path(entry[key]).stat().st_mtime_ns for key in ("native_file", "csv_file")}
    second = exporter.export_results(traits, results_directory=output)
    assert first["created_files"] == second["reused_files"] == 2
    assert second["created_files"] == 0
    assert before == {key: Path(entry[key]).stat().st_mtime_ns for key in before}


@pytest.mark.parametrize("tamper", ["native_file", "csv_file"])
def test_different_existing_results_are_never_overwritten(tmp_path, tamper):
    source = save(tmp_path / "x.rds", RMatrix([[0.01]], ["pvalue"]))
    traits, output = manifest(source), tmp_path / "out"
    entry = one_file(exporter.export_results(traits, results_directory=output))
    Path(entry[tamper]).write_bytes(b"different complete result\n")
    before = {key: Path(entry[key]).read_bytes() for key in ("native_file", "csv_file")}
    with pytest.raises(FileExistsError, match="no result was overwritten"):
        exporter.export_results(traits, results_directory=output)
    assert before == {key: Path(entry[key]).read_bytes() for key in before}
    assert not (output / ".torchstaar-export.lock").exists()


def test_partial_identical_export_can_finish(tmp_path):
    source = save(tmp_path / "x.rds", None)
    destination = tmp_path / "out" / "trait example"
    destination.mkdir(parents=True)
    (destination / source.name).write_bytes(source.read_bytes())
    report = exporter.export_results(manifest(source), results_directory=tmp_path / "out")
    assert report["created_files"] == report["reused_files"] == 1


def test_malformed_second_file_does_not_publish_first_file_of_trait(tmp_path):
    first = save(tmp_path / "first.rds", None)
    malformed = tmp_path / "second.rds"
    malformed.write_bytes(b"invalid serialization")
    with pytest.raises(Exception):
        exporter.export_results(manifest(first, malformed), results_directory=tmp_path / "out")
    assert not (tmp_path / "out" / "trait example").exists()
    assert list((tmp_path / "out").iterdir()) == []


def test_source_change_during_csv_conversion_is_detected(tmp_path, monkeypatch):
    source = save(tmp_path / "x.rds", None)
    original = exporter._write_csv
    def changing(*args, **kwargs):
        details = original(*args, **kwargs)
        save(source, RMatrix([[0.1]], ["pvalue"]))
        return details
    monkeypatch.setattr(exporter, "_write_csv", changing)
    with pytest.raises(RuntimeError, match="source changed"):
        exporter.export_results(manifest(source), results_directory=tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("name", ["", " ", ".", "..", "../outside", "/outside", "a/b", "a\\b", "C:drive", "control\nname"])
def test_unsafe_names_rejected_before_output_creation(tmp_path, name):
    source = save(tmp_path / "x.rds", None)
    with pytest.raises(ValueError, match="safe directory"):
        exporter.export_results(manifest(source, name=name), results_directory=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_case_and_unicode_canonical_directory_collisions_rejected(tmp_path):
    source = save(tmp_path / "x.rds", None)
    for names in (("Same", "same"), ("é", "e\u0301")):
        with pytest.raises(ValueError, match="directory collision"):
            exporter.export_results(manifest(source, name=names[0]) + manifest(source, name=names[1]), results_directory=tmp_path / "out")


def test_same_stem_source_collision_rejected(tmp_path):
    sources = [save(tmp_path / name, None) for name in ("x.rds", "x.Rdata")]
    with pytest.raises(ValueError, match="CSV stem collision"):
        exporter.export_results(manifest(*sources), results_directory=tmp_path / "out")


@pytest.mark.parametrize("level", ["directory", "file"])
def test_destination_symlink_rejected(tmp_path, level):
    source = save(tmp_path / "x.rds", None)
    output, other = tmp_path / "out", tmp_path / "other"
    output.mkdir(); other.mkdir()
    destination = output / "trait example"
    if level == "directory":
        destination.symlink_to(other, target_is_directory=True)
    else:
        destination.mkdir()
        (destination / "x.csv").symlink_to(other / "unknown.csv")
    with pytest.raises(ValueError):
        exporter.export_results(manifest(source), results_directory=output)
    assert list(other.iterdir()) == []


def test_existing_lock_rejected_without_touching_results(tmp_path):
    source = save(tmp_path / "x.rds", None)
    output = tmp_path / "out"
    output.mkdir()
    lock = output / ".torchstaar-export.lock"
    lock.write_text("999\n")
    with pytest.raises(RuntimeError, match="export lock"):
        exporter.export_results(manifest(source), results_directory=output)
    assert lock.read_text() == "999\n"


def test_select_object_name_requires_exact_native_workspace_name(tmp_path):
    source = save(tmp_path / "x.Rdata", RMatrix([[0.01]], ["pvalue"]))
    with pytest.raises(ValueError, match="exactly one"):
        exporter.export_results(manifest(source, object_name="absent"), results_directory=tmp_path / "out")
    report = exporter.export_results(manifest(source, object_name="association"), results_directory=tmp_path / "out")
    assert one_file(report)["rows"] == 1


def test_rds_is_direct_object_even_when_workspace_selector_supplied(tmp_path):
    source = save(tmp_path / "x.rds", RMatrix([[0.01]], ["pvalue"]))
    report = exporter.export_results(manifest(source, object_name="unused"), results_directory=tmp_path / "out")
    assert one_file(report)["rows"] == 1


def test_nonscalar_list_matrix_cell_rejected(tmp_path):
    values = np.empty((1, 1), dtype=object)
    values[0, 0] = np.array([0.1, 0.2])
    source = save(tmp_path / "x.rds", RMatrix(values, ["pvalue"], mode="list"))
    with pytest.raises(ValueError, match="not scalar"):
        exporter.export_results(manifest(source), results_directory=tmp_path / "out")
    assert not (tmp_path / "out" / "trait example").exists()


def test_incompatible_column_order_rejected(tmp_path):
    source = save(tmp_path / "x.rds", [RMatrix([[1, 2]], ["a", "b"]), RMatrix([[2, 1]], ["b", "a"])])
    with pytest.raises(ValueError, match="column orders"):
        exporter.export_results(manifest(source), results_directory=tmp_path / "out")


@pytest.mark.parametrize("unknown", [{"mystery": True}, {"native_files": []}])
def test_invalid_manifest_contract_rejected(tmp_path, unknown):
    source = save(tmp_path / "x.rds", None)
    traits = manifest(source)
    traits[0].update(unknown)
    with pytest.raises(ValueError):
        exporter.export_results(traits, results_directory=tmp_path / "out")


def test_cli_report_and_protected_path_preflight(tmp_path, capsys):
    source = save(tmp_path / "x.rds", None)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest(source)))
    output = tmp_path / "out"
    protected = output / "trait example" / "x.csv"
    with pytest.raises(ValueError, match="collides"):
        exporter.main([str(manifest_path), "--results-directory", str(output), "--report", str(protected)])
    assert not output.exists()
    report_path = tmp_path / "report.json"
    report = exporter.main([str(manifest_path), "--results-directory", str(output), "--report", str(report_path)])
    assert json.loads(report_path.read_text()) == report
    assert json.loads(capsys.readouterr().out)["native_files"] == 1
