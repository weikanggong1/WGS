"""The generated promoter catalog must be consumable by association jobs."""
import json
import pytest
from fudan_wgs_toolkit.cli import _read_promoter_intervals


def test_prepared_json_and_original_tsv_preserve_identical_intervals(tmp_path):
    expected = [("21", 10, 30), ("21", 25, 45)]
    prepared = tmp_path / "chr21.promoters.json"
    prepared.write_text(json.dumps(expected))
    original = tmp_path / "promoters.tsv"
    original.write_text("chromosome\tstart\tend\tstrand\tgene_id\n21\t10\t30\t+\tgene_a\n21\t25\t45\t-\tgene_b\n")
    assert _read_promoter_intervals(prepared) == expected
    assert _read_promoter_intervals(original) == expected


@pytest.mark.parametrize("interval", [["21", 0, 10], ["21", 11, 10],
                                     ["21", 1.5, 10], ["21", True, 10]])
def test_invalid_prepared_intervals_fail_before_mask_selection(tmp_path, interval):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([interval]))
    with pytest.raises(ValueError, match="coordinates"):
        _read_promoter_intervals(path)
