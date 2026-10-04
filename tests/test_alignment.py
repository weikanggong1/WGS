"""Identity-preserving shard alignment; small fixtures are unit checks only."""
from types import SimpleNamespace
import numpy as np
import pytest
from staar_phewas.cli import _bind_gds_samples
from staar_phewas.pipeline import PheWASPipeline


class Shard:
    def __init__(self,ids):
        self.ids=np.asarray(ids,dtype=str);self.n_samples=len(ids);self.n_variants=1
    def sample_ids(self):return self.ids.copy()
    def sample_indices(self,ids):
        lookup={value:j for j,value in enumerate(self.ids)}
        try:return np.asarray([lookup[value] for value in ids],dtype=np.int64)
        except KeyError:raise ValueError("missing identifier") from None
    def read_field(self,path):
        return np.asarray([1]) if path=="position" else np.asarray(["PASS"])


def model(ids):
    return SimpleNamespace(sample_ids=np.asarray(ids,dtype=str),n=len(ids),family="gaussian",n_pheno=1,use_spa=False)


def test_cached_identity_remaps_reordered_shard():
    fitted=model(["101","102"])
    first=_bind_gds_samples(Shard(["s_101","s_102"]),fitted,np.asarray([0,1]))
    second=_bind_gds_samples(Shard(["s_102","s_101"]),fitted,np.asarray([0,1]))
    assert first.tolist()==[0,1]
    assert second.tolist()==[1,0]
    assert fitted.gds_sample_ids.tolist()==["s_101","s_102"]


def test_prepared_rows_must_match_model_identity():
    with pytest.raises(ValueError,match="do not match"):
        _bind_gds_samples(Shard(["s_101","s_102"]),model(["101","102"]),np.asarray([1,0]))


def test_union_uses_one_row_per_physical_sample():
    models=[model(["101","102"]),model(["s_102","s_103"])]
    pipeline=PheWASPipeline(Shard(["s_101","s_102","s_103"]),models,
                            gds_sample_indices=[np.asarray([0,1]),np.asarray([1,2])])
    assert pipeline.union_rows.tolist()==[0,1,2]
    assert [rows.tolist() for rows in pipeline.trait_rows]==[[0,1],[1,2]]
