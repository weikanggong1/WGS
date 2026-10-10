"""Bounded dispatch retains real cached-CLI index validation on CPU."""
from pathlib import Path

import numpy as np
import pytest

from staar_phewas import cli
from staar_phewas.annotation_index import CandidateAnnotationIndex
from staar_phewas.cache_runtime import IndexCacheSpec, run_cached_configuration
from staar_phewas.cache_runtime import index_cache
from staar_phewas.cache_runtime import runtime as cached_runtime
from staar_phewas.pipeline import PheWASPipeline
from staar_phewas.phewas_runtime.mask_limit import LimitedMaskPipeline


class _ReachedPreparedIndex(RuntimeError):
    pass


def _index_fixture(tmp_path):
    binding = index_cache.make_binding(
        gds_stat=dict(device=0, inode=1, size=2, mtime_ns=3), n_variants=5,
        source_sha256={name: "a" * 64 for name in
            ("annotation_index.py", "pipeline.py", "masks.py", "gds.py")},
        annotation_catalog={}, qc_path="annotation/filter", promoter_manifest=None,
        chromosome="21", variant_type="variant", categories=["ncRNA"])
    index = CandidateAnnotationIndex("21", "variant")
    index._groups = {"fixture_gene": {"ncRNA": np.array([0, 3], dtype=np.int64)}}
    index.prepared_categories = {"ncRNA"}
    path = tmp_path / "candidate-index.npz"
    index_cache.save(path, index, binding)
    return path, binding


def _exercise_cached_dispatch(tmp_path, monkeypatch, *, capped, rejection=None):
    source = tmp_path / "synthetic.gds"
    path, binding = _index_fixture(tmp_path)
    if rejection == "binding":
        binding = dict(binding, qc_path="changed/qc")
    budget = 1 if rejection == "budget" else 512 * 2**20
    instances, readers, hooks = [], [], []

    class Reader:
        reader_metadata = {}

        def __init__(self, filename, **options):
            self._cache_source_path = str(Path(filename).resolve())
            self.closed = False
            readers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    class Model:
        n = 3
        n_pheno = 1
        family = "gaussian"
        use_spa = False
        matmul_mode = "fp64"

        def set_matmul_mode(self, mode):
            self.matmul_mode = mode

    def initialize(self, gds, models, **options):
        self.gds, self.models, self.options = gds, models, options["options"]
        self._annotation_indexes = {}
        if rejection == "live":
            self._annotation_indexes[("21", "variant")] = object()
        instances.append(self)

    class Wrapper(PheWASPipeline):
        def __init__(self, *args, **kwargs):
            hooks.append("before")
            super().__init__(*args, **kwargs)
            hooks.append("after")

    def prepared(self, chromosome, **kwargs):
        restored = self._annotation_indexes[("21", "variant")]
        np.testing.assert_array_equal(restored._groups["fixture_gene"]["ncRNA"], [0, 3])
        assert not restored._groups["fixture_gene"]["ncRNA"].flags.writeable
        assert restored.prepared_categories == {"ncRNA"}
        raise _ReachedPreparedIndex("restored index reached before association")

    monkeypatch.setattr(PheWASPipeline, "__init__", initialize)
    monkeypatch.setattr(PheWASPipeline, "prepare_annotation_index", prepared)
    monkeypatch.setattr(cli, "PheWASPipeline", Wrapper)
    monkeypatch.setattr(cli, "SeqArrayGDS", Reader)
    monkeypatch.setattr(cli, "GaussianNullModel", Model)
    monkeypatch.setattr(cli, "load_null_model", lambda *args, **kwargs: Model())
    monkeypatch.setattr(cli, "_bind_gds_samples", lambda *args: np.arange(3, dtype=np.int64))
    monkeypatch.setattr(cached_runtime, "make_reader_factory", lambda *args, **kwargs: Reader)
    # Exercise the actual CLI model/chromosome/constructor dispatch. Numerical
    # verification contexts are outside this constructor contract; no products
    # are requested, and the index hook ends the test before association.
    monkeypatch.setattr(cli, "run_configuration", cli._run_configuration)
    mode, device = ("fp64", "cpu") if capped else ("tf32", "cuda:0")
    if not capped:
        monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(cli.torch.cuda, "init", lambda: None)
        monkeypatch.setattr(cli.torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    config = dict(matmul_mode=mode, precision_control=capped, resident_genotypes=True,
        phenotypes=[dict(name="fixture_trait", model="synthetic-model.npz")],
        chromosomes=[dict(name="21", gds=str(source), annotation_index={}, jobs=[
            dict(kind="ncrna", output=str(tmp_path / "synthetic.Rdata"))])])
    if capped:
        config["maximum_mask_variants"] = 4
    spec = IndexCacheSpec(path, binding, max_uncompressed_bytes=budget)
    expected = {
        None: (_ReachedPreparedIndex, "restored index reached"),
        "binding": (ValueError, "source/input binding mismatch"),
        "budget": (MemoryError, "decompression budget exceeded"),
        "live": (ValueError, "replace a live prepared index"),
    }[rejection]
    with pytest.raises(expected[0], match=expected[1]):
        run_cached_configuration(config, cache_specs={}, device=device,
            index_caches={source: [spec]})
    assert len(instances) == 1 and len(readers) == 1 and readers[0].closed
    assert hooks == ["before", "after"]
    assert isinstance(instances[0], LimitedMaskPipeline)
    assert isinstance(instances[0], Wrapper)
    assert any(base.__name__ == "CachedPipeline" for base in type(instances[0]).__mro__)
    assert cli.PheWASPipeline is Wrapper and cli.SeqArrayGDS is Reader
    assert not cached_runtime._CLI_LOCK.locked()


@pytest.mark.parametrize("capped", [True, False])
def test_real_cached_wrapper_restores_index_for_both_bounded_dispatches(tmp_path, monkeypatch, capped):
    _exercise_cached_dispatch(tmp_path, monkeypatch, capped=capped)


@pytest.mark.parametrize("capped", [True, False])
@pytest.mark.parametrize("rejection", ["binding", "budget", "live"])
def test_real_cached_wrapper_rejects_invalid_index_before_analysis(tmp_path, monkeypatch, capped, rejection):
    _exercise_cached_dispatch(tmp_path, monkeypatch, capped=capped, rejection=rejection)


def test_bounded_type_returns_unwrapped_original_subclass(monkeypatch):
    monkeypatch.setattr(cli, "PheWASPipeline", PheWASPipeline)
    assert cli._bounded_pipeline_type() is LimitedMaskPipeline


def test_bounded_type_preserves_already_bounded_constructor(monkeypatch):
    calls = []

    class AlreadyBounded(LimitedMaskPipeline):
        def __init__(self):
            calls.append("hook")

    monkeypatch.setattr(cli, "PheWASPipeline", AlreadyBounded)
    candidate = cli._bounded_pipeline_type()
    assert candidate is AlreadyBounded
    candidate()
    assert calls == ["hook"]


@pytest.mark.parametrize("current", [object, object()])
def test_bounded_type_rejects_unrelated_or_nonclass_wrapper(monkeypatch, current):
    monkeypatch.setattr(cli, "PheWASPipeline", current)
    with pytest.raises(TypeError, match="must inherit PheWASPipeline"):
        cli._bounded_pipeline_type()
