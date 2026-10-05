"""R topology regressions; scientific equivalence uses separate real-data oracles."""
import numpy as np
import pytest

from staar_phewas.results import TraitRows
from staar_phewas.r_output import (
    RAttributed, RDataFrame, RMatrix, association_object,
    association_batch_object, write_association_output,
)


def gene_row(category="plof"):
    return {"Gene name": "GENE", "Chr": 21, "Category": category,
            "#SNV": 2, "cMAC": 5.0, "STAAR-O": 0.5}


def test_base_has_category_matrices_and_null_without_trait_layer():
    obj = association_object({"plof": [[gene_row()]], "ptv": [[]]},
                             kind="coding", layout="base")
    assert isinstance(obj["plof"], RMatrix)
    assert obj["ptv"] is None
    assert isinstance(obj["plof"].values[0, 1], int)
    assert isinstance(obj["plof"].values[0, 3], int)


def test_base_noncoding_retains_original_mixed_cell_types():
    obj = association_object({k: [[gene_row(k)]] for k in ("UTR", "promoter_CAGE")},
                             kind="noncoding", layout="base")
    assert obj["UTR"].values[0, 1] == "21"
    assert obj["UTR"].values[0, 3] == "2"
    assert isinstance(obj["promoter_CAGE"].values[0, 1], int)
    assert isinstance(obj["promoter_CAGE"].values[0, 3], int)


def test_base_individual_retains_dataframe_metadata():
    rows = TraitRows([{"CHR": 21, "POS": 10, "REF": "A", "ALT": "G", "N": 42}],
                     row_names=[3], factor_levels={"REF": ["A"], "ALT": ["G"]})
    obj = association_object([rows], kind="individual", layout="base")
    assert isinstance(obj, RDataFrame)
    assert obj.row_names == [3]
    assert obj.columns["N"].dtype == np.dtype("int32")
    assert obj.columns["CHR"].dtype == np.dtype("float64")


def test_base_batch_keeps_duplicate_category_names_and_empty_entries():
    result = {"plof": [[gene_row()]], "ptv": [[]]}
    obj = association_batch_object([result, result], kind="coding", layout="base")
    assert isinstance(obj, RAttributed)
    assert list(obj.attributes["names"]) == ["plof", "ptv", "plof", "ptv"]
    assert obj.value[1] is obj.value[3] is None


def test_base_ncrna_rbind_and_saved_name(tmp_path):
    result = [[gene_row("ncRNA")]]
    obj = association_batch_object([result, [[]], result], kind="ncrna", layout="base")
    assert isinstance(obj, RMatrix)
    assert obj.values.shape == (2, 6)
    assert obj.row_names == ["results_temp", "results_temp"]
    import rdata
    path = tmp_path / "Phenotype_ncRNA_1.Rdata"
    write_association_output(path, result, kind="ncrna", layout="base")
    assert list(rdata.read_rda(path)) == ["results_ncRNA"]


def test_base_rejects_multiple_traits():
    with pytest.raises(ValueError, match="exactly one trait"):
        association_object([[gene_row()], [gene_row()]], kind="ncrna", layout="base")
