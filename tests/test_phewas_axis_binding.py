"""Verified immutable row bindings avoid full-N scans; tiny contract fixtures."""
import numpy as np
import pytest
from types import SimpleNamespace

from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime, single
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker, _immutable
from test_phewas_shared_state import MetadataReader, make_cache, fixture_states
from test_phewas_single_batches import states_fixture
from test_phewas_batched_runtime import (cpu_product_oracle, model_for, Repository,
                                         descriptor)


class _SingleMetadata(MetadataReader):
    def read_ref_alt(self, indices=None):
        n = self.n_variants if indices is None else len(indices)
        return np.full(n, 'A'), np.full(n, 'C')


def test_immutable_descriptor_groups_are_owned_and_cannot_be_made_writable():
    original = np.arange(7, dtype=np.int64)
    _, groups = runtime._descriptor_groups(Repository([descriptor(0, original)]))
    rows = groups[0]['rows']
    original[:] = original[::-1]
    np.testing.assert_array_equal(rows, np.arange(7))
    with pytest.raises(ValueError):
        rows.flags.writeable = True


def test_proven_immutable_axis_alias_skips_digest_and_full_value_comparison(tmp_path, monkeypatch):
    raw = fixture_states(m=9, n=11)
    rows = _immutable([8, 1, 4, 2])
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        first = broker.reader_view(rows)
        before = broker.metrics['sample_axis_digest_calls']
        def forbidden(*args, **kwargs):
            raise AssertionError('proven immutable alias must not scan all row values')
        monkeypatch.setattr(np, 'array_equal', forbidden)
        second = broker.reader_view(rows)
        assert second._axis is first._axis
        assert broker.metrics['sample_axis_digest_calls'] == before
        assert broker.metrics['sample_axis_immutable_alias_hits'] == 1


@pytest.mark.parametrize('readonly_view', [False, True])
def test_unproven_mutable_storage_is_rehashed_after_value_mutation(tmp_path, readonly_view):
    raw = fixture_states(m=9, n=11)
    owner = np.asarray([8, 1, 4, 2], dtype=np.int64)
    rows = owner.view()
    if readonly_view:
        rows.flags.writeable = False
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        original = broker.reader_view(rows)._axis
        owner[:2] = owner[:2][::-1]
        changed = broker.reader_view(rows)._axis
        assert original is not changed
        np.testing.assert_array_equal(changed.samples, owner)
        assert broker.metrics['sample_axis_digest_calls'] == 2
        assert broker.metrics['sample_axis_immutable_alias_hits'] == 0


def test_readonly_mmap_alias_binds_source_file_identity(tmp_path):
    raw = fixture_states(m=9, n=11)
    filename = tmp_path/'rows.npy'
    np.save(filename, np.asarray([8, 1, 4, 2], dtype=np.int64))
    rows = np.load(filename, mmap_mode='r', allow_pickle=False)
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        original = broker.reader_view(rows)._axis
        assert broker.reader_view(rows)._axis is original
        np.save(filename, np.asarray([1, 8, 4, 2], dtype=np.int64))
        with pytest.raises(ValueError, match='source binding was altered'):
            broker.reader_view(rows)


def test_previously_proven_axis_rejects_geometry_mutation(tmp_path):
    raw = fixture_states(m=9, n=11)
    rows = _immutable([8, 1, 4, 2])
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        broker.reader_view(rows)
        rows.shape = (2, 2)
        with pytest.raises(ValueError):
            broker.reader_view(rows)


def test_all_cohort_identity_rows_share_one_population_sized_immutable_rowmap(tmp_path):
    raw = fixture_states(m=9, n=11)
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        first = broker.reader_view(_immutable([8, 1, 4, 2]))._single_axis_binding()
        second = broker.reader_view(_immutable([8, 1, 4]))._single_axis_binding()
        assert broker._identity_rows.nbytes == 11*8
        assert np.shares_memory(first.axis.identity_rows, broker._identity_rows)
        assert np.shares_memory(second.axis.identity_rows, broker._identity_rows)
        assert not first.axis.identity_rows.flags.writeable


def test_thin_pipeline_normalizes_equal_caller_rows_and_preserves_dimension_gate(tmp_path):
    raw = fixture_states(m=9, n=11)
    original = np.asarray([8, 1, 4, 2], dtype=np.int64)
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        view = broker.reader_view(original)
        model = model_for(original)
        pipeline = runtime._thin_pipeline(view, model, original,
            AnalysisOptions(wrapper_semantics='base'), {})
        assert pipeline.union_rows is view._axis.samples
        structural = runtime._thin_pipeline(view, SimpleNamespace(device=broker.device, use_spa=False),
            original, pipeline.options, {}, pipeline)
        assert structural.trait_rows[0] is pipeline.trait_rows[0]
        with pytest.raises(ValueError, match='reader axis'):
            runtime._thin_pipeline(view, model, original[::-1], pipeline.options, {})
        with pytest.raises(ValueError, match='binding was altered'):
            runtime._thin_pipeline(view, SimpleNamespace(n=3), original, pipeline.options, {})


def _prepared_binding(tmp_path):
    states = states_fixture()
    metadata = SharedMetadataReader(_SingleMetadata(256, 10))
    broker = SharedStateBroker(metadata, make_cache(tmp_path/'cache', states.T), device='cpu')
    rows = _immutable(np.arange(256))
    first_view = broker.reader_view(rows)
    models = [model_for(first_view._axis.samples, .8), model_for(first_view._axis.samples, 1.2)]
    verified = set()
    pipelines = []
    for index, model in enumerate(models):
        runtime._check_model(model, descriptor(index, rows), metadata, verified)
        pipelines.append(runtime._thin_pipeline(first_view, model, first_view._axis.samples,
            AnalysisOptions(wrapper_semantics='base'), {}, pipelines[0] if pipelines else None))
    block = broker.trait_block(broker.read_states(np.arange(10)), first_view._axis.samples, minimum_mac=1)
    state, ordinal = pipelines[0]._prepare_individual_block(block, 0, mac_cutoff=1)
    prepared = pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=1)
    prepared.pop('model')
    return broker, rows, pipelines, prepared, ordinal


def test_proven_single_binding_skips_full_n_array_equal_and_arange(tmp_path, monkeypatch, cpu_product_oracle):
    broker, rows, pipelines, prepared, ordinal = _prepared_binding(tmp_path)
    with broker:
        expected, _ = single.process_single_shared_genotype(pipelines, None, ordinal, '1',
            prepared_state=(prepared, ordinal))
        equal, arange = np.array_equal, np.arange
        def bounded_equal(left, right, *args, **kwargs):
            assert np.asarray(left).size < len(rows) and np.asarray(right).size < len(rows)
            return equal(left, right, *args, **kwargs)
        def bounded_arange(*args, **kwargs):
            value = arange(*args, **kwargs)
            assert value.size < len(rows)
            return value
        monkeypatch.setattr(np, 'array_equal', bounded_equal)
        monkeypatch.setattr(np, 'arange', bounded_arange)
        actual, _ = single.process_single_shared_genotype(pipelines, None, ordinal, '1',
            prepared_state=(prepared, ordinal))
        assert actual == expected
        second_view = broker.reader_view(rows)
        second = runtime._thin_pipeline(second_view, pipelines[0].models[0], second_view._axis.samples,
            pipelines[0].options, {}, pipelines[0])
        assert second.trait_rows[0] is pipelines[0].trait_rows[0]


@pytest.mark.parametrize('change', ['union_copy', 'identity_writable', 'identity_reverse', 'reader', 'token'])
def test_single_binding_rejects_unverified_replacement(tmp_path, change, cpu_product_oracle):
    broker, rows, pipelines, prepared, ordinal = _prepared_binding(tmp_path)
    with broker:
        pipeline = pipelines[0]
        if change == 'union_copy':
            pipeline.union_rows = pipeline.union_rows.copy()
        elif change == 'identity_writable':
            pipeline.trait_rows = [pipeline.trait_rows[0].copy()]
        elif change == 'identity_reverse':
            pipeline.trait_rows = [pipeline.trait_rows[0][::-1]]
        elif change == 'reader':
            pipeline.genotype = broker.reader_view(rows[::-1])
        else:
            pipeline._single_axis_binding = object()
        with pytest.raises(ValueError, match='binding'):
            single.process_single_shared_genotype(pipelines, None, ordinal, '1',
                prepared_state=(prepared, ordinal))
