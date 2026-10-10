"""PyTorch intra-op CPU control contracts; no cache workers or GPU are started."""
from copy import deepcopy
import json
import os
import numpy as np
import pytest
import torch

from staar_phewas.phewas_runtime import runtime
from torchstaar_phewas import cli


@pytest.mark.parametrize('requested', [1, 2, 4])
def test_cpu_threads_apply_during_actual_cpu_operation_and_restore(monkeypatch, requested):
    original = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    environment = dict(os.environ)
    observed = []
    def compute(*args, **kwargs):
        observed.append(torch.get_num_threads())
        # This executes a CPU tensor operation while the requested setting is active.
        value = torch.ones((17, 13)) @ torch.ones((13, 11))
        assert value.device.type == 'cpu' and torch.all(value == 13)
        return {'cpu_operation_completed': True}
    monkeypatch.setattr(runtime, '_run', compute)
    report = runtime.run_configuration([], cache_specs={}, cpu_threads=requested)
    assert observed == [requested] and torch.get_num_threads() == original
    assert report['cpu_thread_control'] == dict(requested=requested, effective=requested,
                                              previous=original, restored=True, scope='pytorch_intraop')
    assert torch.get_num_interop_threads() == interop and dict(os.environ) == environment
    assert not runtime._LOCK.locked()


def test_python_default_is_two_cpu_intraop_threads(monkeypatch):
    observed = []
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: observed.append(torch.get_num_threads()) or {})
    report = runtime.run_configuration([], cache_specs={})
    assert observed == [2] and report['cpu_thread_control']['requested'] == 2


@pytest.mark.parametrize('failure', [ValueError, RuntimeError, KeyboardInterrupt])
def test_failed_run_restores_cpu_threads_and_releases_lock(monkeypatch, failure):
    original = torch.get_num_threads()
    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 3
        raise failure('controlled failure')
    monkeypatch.setattr(runtime, '_run', fail)
    with pytest.raises(failure, match='controlled failure'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=3)
    assert torch.get_num_threads() == original and not runtime._LOCK.locked()


@pytest.mark.parametrize('invalid', [True, False, 0, -1, 2.0, '2', None, np.int64(2)])
def test_invalid_cpu_threads_fail_before_thread_mutation_or_run(monkeypatch, invalid):
    called = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: called.append(value))
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: called.append('run'))
    with pytest.raises(ValueError, match='positive integer'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=invalid)
    assert called == [] and not runtime._LOCK.locked()


def test_busy_context_does_not_change_cpu_threads(monkeypatch):
    called = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: called.append(value))
    assert runtime._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            runtime.run_configuration([], cache_specs={}, cpu_threads=4)
    finally:
        runtime._LOCK.release()
    assert called == []


def test_initial_thread_setup_failure_still_attempts_restore_and_unlocks(monkeypatch):
    state = {'threads': 7}
    calls = []
    monkeypatch.setattr(torch, 'get_num_threads', lambda: state['threads'])
    def set_threads(value):
        calls.append(value)
        state['threads'] = value
        if value == 2:
            raise RuntimeError('thread setup failed')
    monkeypatch.setattr(torch, 'set_num_threads', set_threads)
    with pytest.raises(RuntimeError, match='thread setup failed'):
        runtime.run_configuration([], cache_specs={})
    assert calls == [2, 7] and state['threads'] == 7 and not runtime._LOCK.locked()


def test_effective_setting_must_match_request(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 7)
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: calls.append(value))
    with pytest.raises(RuntimeError, match='was not applied'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=3)
    assert calls == [3, 7] and not runtime._LOCK.locked()


@pytest.mark.parametrize('config_threads,override,expected', [(None, None, None), (3, None, 3), (3, 4, 4), (None, 4, 4)])
def test_cli_cpu_threads_overrides_shared_option_without_mutating_input(tmp_path, monkeypatch, config_threads, override, expected):
    config = {'analyses': [], 'caches': [], 'shared_options': {'device_cache_bytes': 123}}
    if config_threads is not None:
        config['shared_options']['cpu_threads'] = config_threads
    original = deepcopy(config)
    path, report_path = tmp_path / 'configuration.json', tmp_path / 'report.json'
    path.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(cli, 'configuration_inputs', lambda value: (['plan'], {'cache': 'spec'}))
    monkeypatch.setattr(cli, 'run_configuration', lambda analyses, **options: calls.append((analyses, options)) or {'complete': True})
    arguments = [str(path), '--report', str(report_path)]
    if override is not None:
        arguments += ['--cpu-threads', str(override)]
    cli.main(arguments)
    assert calls[0][0] == ['plan'] and calls[0][1]['device_cache_bytes'] == 123
    if expected is None:
        assert 'cpu_threads' not in calls[0][1]  # public Python default is the authority
    else:
        assert calls[0][1]['cpu_threads'] == expected
    assert json.loads(path.read_text()) == original
    assert json.loads(report_path.read_text()) == {'complete': True}


@pytest.mark.parametrize('value', ['0', '-2', '1.5', 'True'])
def test_cli_rejects_nonpositive_or_noninteger_thread_flags_before_input_read(tmp_path, value):
    with pytest.raises(SystemExit) as error:
        cli.main([str(tmp_path / 'missing.json'), '--report', str(tmp_path / 'report.json'), '--cpu-threads', value])
    assert error.value.code == 2 and not (tmp_path / 'report.json').exists()


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '2', None])
def test_json_cpu_threads_use_strict_python_validation(tmp_path, monkeypatch, value):
    path, report = tmp_path / 'config.json', tmp_path / 'report.json'
    path.write_text(json.dumps({'analyses': [], 'caches': [], 'shared_options': {'cpu_threads': value}}))
    monkeypatch.setattr(cli, 'configuration_inputs', lambda config: ([], {}))
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: pytest.fail('invalid threads must not run'))
    with pytest.raises(ValueError, match='positive integer'):
        cli.main([str(path), '--report', str(report)])
    assert not report.exists()


def test_cpu_workers_remains_an_unknown_option(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'analyses': [], 'caches': [], 'shared_options': {'cpu_workers': 4}}))
    monkeypatch.setattr(cli, 'configuration_inputs', lambda config: ([], {}))
    with pytest.raises(ValueError, match='unknown shared options'):
        cli.main([str(path), '--report', str(tmp_path / 'report.json')])
