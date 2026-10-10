"""Single owns upstream iterator cleanup; tiny CPU contracts, no benchmark."""
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype import SparseMinorBlock
from fudan_wgs_toolkit.null_model import GaussianNullModel
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline


class SourceFailure(RuntimeError):
    pass


class ScoreFailure(RuntimeError):
    pass


class ConsumerFailure(RuntimeError):
    pass


def tiny_block(indices):
    indices = np.asarray(indices, dtype=np.int64)
    width = len(indices)
    return SparseMinorBlock(
        np.tile(np.asarray([0, 1]), width), np.repeat(np.arange(width), 2),
        np.ones(2 * width), np.arange(4), indices,
        np.full(width, .75), union_initial_mac=np.full(width, 2),
        union_missing_rate=np.zeros(width))


class JoiningIterator:
    """Reader mock whose close synchronously reclaims a real producer thread."""

    def __init__(self, indices, events, *, fail_read=False):
        self.blocks = iter([tiny_block(indices[start:start + 3])
                            for start in range(0, len(indices), 3)])
        self.events, self.fail_read = events, fail_read
        self.close_calls = 0
        self.stop = Event()
        self.producer = Thread(target=self.stop.wait, daemon=True)
        self.producer.start()

    def __iter__(self):
        return self

    def __next__(self):
        assert not self.close_calls, "read after iterator was closed"
        if self.fail_read:
            raise SourceFailure("source failed")
        return next(self.blocks)

    def close(self):
        self.close_calls += 1
        self.events.append("source_close")
        self.stop.set()
        self.producer.join(timeout=2)
        assert not self.producer.is_alive(), "source producer was not joined"
        self.events.append("producer_join")


class ClosingReader:
    n_variants = 6

    def __init__(self, effective, *, fail_read=False):
        self.effective, self.fail_read = effective, fail_read
        self.iterators, self.events = [], []

    def _iterator(self, indices, route):
        assert route == ("effective" if self.effective else "ordinary")
        iterator = JoiningIterator(indices, self.events, fail_read=self.fail_read)
        self.iterators.append(iterator)
        return iterator

    def iter_minor_blocks(self, indices, rows, **kwargs):
        return self._iterator(indices, "ordinary")

    def iter_effective_minor_blocks(self, indices, rows, **kwargs):
        assert kwargs["effective_block_size"] == 1024
        return self._iterator(indices, "effective")

    def read_field(self, name, indices):
        assert name == "chromosome"
        return np.full(len(indices), "1")

    def read_ref_alt(self, indices):
        return np.full(len(indices), "A"), np.full(len(indices), "T")

    def close(self):
        # Closing the cache reader while any iterator still produces blocks
        # models the use-after-close race that this regression prevents.
        assert all(iterator.close_calls == 1 and not iterator.producer.is_alive()
                   for iterator in self.iterators)
        self.events.append("reader_close")


def analysis(reader, monkeypatch, *, fail_score=False):
    model = GaussianNullModel.__new__(GaussianNullModel)
    model.sample_ids = np.arange(4).astype(str)
    model.n_pheno, model.use_spa, model.matmul_mode = 1, False, "tf32"
    model.x = (SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32)
               if reader.effective else torch.ones((4, 1), dtype=torch.float32))

    def score_variance(genotype):
        if fail_score:
            raise ScoreFailure("association failed")
        return genotype.sum(0), (genotype * genotype).sum(0) + 1

    model.individual_score_variance = score_variance
    if reader.effective:
        # Exercise the effective-route compatibility check without requiring
        # CUDA. The source returns only tiny CPU genotypes for this contract.
        as_tensor = torch.as_tensor

        def on_cpu(value, *args, **kwargs):
            if str(kwargs.get("device", "cpu")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return as_tensor(value, *args, **kwargs)

        monkeypatch.setattr(torch, "as_tensor", on_cpu)

    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.models, pipeline.genotype = [model], reader
    pipeline.options = AnalysisOptions(wrapper_semantics="base", annotation_block_size=3)
    pipeline.trait_rows, pipeline.union_rows = [np.arange(4)], np.arange(4)
    pipeline.resident_genotypes, pipeline.single_batch_optimization = True, True
    pipeline.individual_effective_block_size = 1024
    pipeline.position = np.arange(reader.n_variants) * 10 + 100
    pipeline._base_mask = lambda *args: np.ones(reader.n_variants, dtype=bool)
    pipeline._limit = lambda *args, **kwargs: None
    pipeline.region_indices = lambda *args: np.arange(reader.n_variants)
    pipeline.annotations = lambda *args, **kwargs: None
    monkeypatch.setattr("fudan_wgs_toolkit.pipeline.variant_filter",
                        lambda *args: np.ones(reader.n_variants, dtype=bool))
    return pipeline


@pytest.mark.parametrize("effective", [False, True])
@pytest.mark.parametrize("region", [False, True])
@pytest.mark.parametrize("termination", ["close", "throw", "score", "source", "exhaust"])
def test_single_closes_upstream_and_joins_before_reader_release(
        effective, region, termination, monkeypatch):
    reader = ClosingReader(effective, fail_read=termination == "source")
    pipeline = analysis(reader, monkeypatch, fail_score=termination == "score")
    args = ("1", 100, 150) if region else ("1",)
    records = pipeline.iter_individual_records(*args, mac_cutoff=1)
    try:
        if termination == "score":
            with pytest.raises(ScoreFailure, match="association failed"):
                next(records)
        elif termination == "source":
            with pytest.raises(SourceFailure, match="source failed"):
                next(records)
        elif termination == "exhaust":
            rows = [row for _, part in records for row in part]
            assert [row["POS"] for row in rows] == list(pipeline.position)
        else:
            trait, rows = next(records)
            assert trait == 0 and len(rows) == 3
            if termination == "throw":
                with pytest.raises(ConsumerFailure, match="consumer failed"):
                    records.throw(ConsumerFailure("consumer failed"))
            else:
                records.close()
            assert len(reader.iterators) == 1  # no next request was started
        assert reader.iterators
        assert all(iterator.close_calls == 1 for iterator in reader.iterators)
        reader.close()
        assert reader.events == [event for _ in reader.iterators
                                 for event in ("source_close", "producer_join")] + ["reader_close"]
    finally:
        # A failing regression still releases mock threads for the test suite.
        records.close()
        for iterator in reader.iterators:
            if not iterator.close_calls:
                iterator.close()


@pytest.mark.parametrize("effective", [False, True])
def test_unstarted_single_close_does_not_open_reader(effective, monkeypatch):
    reader = ClosingReader(effective)
    pipeline = analysis(reader, monkeypatch)
    records = pipeline.iter_individual_records("1", mac_cutoff=1)
    records.close()
    reader.close()
    assert reader.iterators == [] and reader.events == ["reader_close"]
