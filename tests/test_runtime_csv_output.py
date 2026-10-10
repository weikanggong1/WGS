import csv

import numpy as np
import pytest

from fudan_wgs_toolkit import cli
from fudan_wgs_toolkit.phewas_runtime import runtime


def test_csv_outputs_wait_for_every_scheduled_batch(tmp_path):
    output = str(tmp_path / "trait_one" / "coding.csv")
    jobs = [{"kind": "coding", "output": output}] * 2
    sink = runtime._CSVOutputs({"chromosomes": [{"jobs": jobs}]})
    sink.append(jobs[0], {"a": [[{"Gene name": "first", "WGS-O": .001}]]})
    assert not (tmp_path / "trait_one" / "coding.csv").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        sink.check()
    sink.append(jobs[1], {"a": [[{"Gene name": "second", "WGS-O": .002}]]})
    sink.check()
    assert sink.files == 1 and sink.proofs[0]["rows"] == 2 and sink.seconds >= 0
    with open(output, newline="") as stream:
        assert [r[0] for r in list(csv.reader(stream))[1:]] == ["first", "second"]


def test_sample_binding_never_discards_family_id():
    class Reader:
        def sample_ids(self):
            return np.asarray(['["family_two","same"]', '["family_one","same"]'])
        def sample_indices(self, keys):
            lookup = {key: i for i, key in enumerate(self.sample_ids())}
            return np.asarray([lookup[key] for key in keys], dtype=np.int64)
    class Model:
        n = 2
        sample_ids = np.asarray(['["family_one","same"]', '["family_two","same"]'])
    model = Model()
    assert cli._bind_genotype_samples(Reader(), model).tolist() == [1, 0]
    with pytest.raises(ValueError, match="never truncated"):
        cli._bind_genotype_samples(Reader(), model, rule="auto")
    with pytest.raises(ValueError, match="complete model"):
        cli._bind_genotype_samples(Reader(), model, prepared_indices=np.asarray([0, 1]))


def test_cpu_threads_restore_after_runtime_failure(monkeypatch):
    import torch
    original = torch.get_num_threads()
    seen = []
    def fail(*args, **kwargs):
        seen.append(torch.get_num_threads())
        raise ValueError("controlled failure")
    monkeypatch.setattr(runtime, "_run", fail)
    with pytest.raises(ValueError, match="controlled failure"):
        runtime.run_configuration([], cache_specs={}, cpu_threads=1)
    assert seen == [1] and torch.get_num_threads() == original
    assert not runtime._LOCK.locked()


def test_prepared_reordered_population_binds_logical_rows_and_cohort_maf(tmp_path):
    """A FAM subset and distinct trait subsets must keep both identity axes.

    This composes the real preparation, portable metadata, sample binding and
    shared state broker. Direct BED-state counts provide the cohort oracle;
    this small contract check does not measure scientific accuracy or speed.
    """
    from types import SimpleNamespace
    from test_plink_prepare import source_fixture
    from fudan_wgs_toolkit.prepare import prepare_WGS_data
    from fudan_wgs_toolkit.identity import sample_keys
    from fudan_wgs_toolkit.cache_runtime.fast_container import Container
    from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
    from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker

    source, annotations, pairs, source_states = source_fixture(tmp_path)
    physical_rows = np.asarray([5, 1, 2], dtype=np.int64)
    prepared_pairs = pairs[physical_rows]
    output = tmp_path / "prepared"
    prepare_WGS_data(source, output, annotation_directory=annotations,
                     sample_pairs=prepared_pairs, chunk_size=2, cpu_threads=1)
    container = Container(output / "chr21")
    np.testing.assert_array_equal(container.samples, physical_rows)
    logical_states = source_states[:, physical_rows].T
    reference_alleles = np.asarray([2, 1, 0, 0], dtype=np.int64)
    called_alleles = np.asarray([2, 2, 2, 0], dtype=np.int64)
    frequencies = []
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as metadata:
        with SharedStateBroker(metadata, container, device="cpu") as broker:
            np.testing.assert_array_equal(broker.samples, np.arange(3))
            variant_rows = np.arange(metadata.n_variants, dtype=np.int64)
            raw = broker.read_states(variant_rows)
            np.testing.assert_array_equal(raw.states.numpy(), logical_states)
            for trait_rows in (np.asarray([2, 0]), np.asarray([1, 2])):
                model = SimpleNamespace(n=len(trait_rows), sample_ids=sample_keys(prepared_pairs[trait_rows]))
                bound_rows = cli._bind_genotype_samples(metadata, model)
                np.testing.assert_array_equal(bound_rows, trait_rows)
                with broker.reader_view(bound_rows) as view:
                    block = view.minor_block(variant_rows, bound_rows)
                states = logical_states[trait_rows]
                reference_count = reference_alleles[states].sum(axis=0)
                called_count = called_alleles[states].sum(axis=0)
                reference_frequency = reference_count / called_count
                np.testing.assert_array_equal(block.sample_indices, trait_rows)
                np.testing.assert_array_equal(block.union_ref_ac, reference_count)
                np.testing.assert_array_equal(block.union_called_alleles, called_count)
                np.testing.assert_array_equal(block.union_ref_af, reference_frequency)
                np.testing.assert_array_equal(block.union_missing_rate,
                                               1 - called_count / (2 * len(trait_rows)))
                expected_dosage = np.where(states < 3,
                    np.where(reference_frequency >= .5, states, 2 - states), 3)
                np.testing.assert_array_equal(block.dosage.numpy(), expected_dosage)
                frequencies.append(np.minimum(reference_frequency, 1-reference_frequency))
            # The same prepared variants have different MAF after trait-specific
            # missing phenotypes select different participants.
            assert frequencies[0][0] == .25
            assert frequencies[1][0] == .5
