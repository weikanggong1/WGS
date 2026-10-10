"""Public namespace and private CLI configuration contracts on CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

import torchstaar_phewas
from torchstaar_phewas import cli
from staar_phewas import __version__
from staar_phewas.cache_runtime import CacheSpec
from staar_phewas.phewas_runtime.runtime import run_configuration


@pytest.fixture
def live_proof(tmp_path, monkeypatch):
    """Test source proof reads a current file rather than returning a constant."""
    source = tmp_path / "synthetic_source_identity.json"
    binding = {"synthetic_source_revision": 1}
    source.write_text(json.dumps(binding))
    module = ModuleType("synthetic_phewas_source_proof")
    def current_source():
        return json.loads(source.read_text())
    module.current_source = current_source
    module.noncallable = "synthetic descriptor"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return dict(binding=binding, source=source, function=current_source,
                reference=module.__name__ + ":current_source", module=module)


def configuration(tmp_path, proof):
    return dict(analyses=[{"phenotypes": [{"name": "trait_001", "model": "state.npz"}],
                           "chromosomes": []}],
                caches=[dict(gds=str(tmp_path / "source.gds"), directory=str(tmp_path / "cache"),
                             expected_binding=deepcopy(proof["binding"]), source_proof=proof["reference"])])


def test_public_namespace_exports_the_same_runtime_cache_spec_and_version():
    assert torchstaar_phewas.run_configuration is run_configuration
    assert torchstaar_phewas.CacheSpec is CacheSpec
    assert torchstaar_phewas.__version__ == __version__
    assert set(torchstaar_phewas.__all__) == {"run_configuration", "CacheSpec"}


def test_configuration_inputs_loads_json_analyses_bindings_rows_and_live_proof(tmp_path, live_proof):
    config = configuration(tmp_path, live_proof)
    analysis = config["analyses"][0]
    analysis_file, binding_file, sample_file = (tmp_path / name for name in
                                               ("analysis.json", "binding.json", "rows.npy"))
    analysis_file.write_text(json.dumps(analysis))
    binding_file.write_text(json.dumps(live_proof["binding"]))
    samples = np.asarray([3, 1, 0], dtype=np.uint32)
    np.save(sample_file, samples)
    config["analyses"] = [str(analysis_file), deepcopy(analysis)]
    config["caches"][0].update(expected_binding=str(binding_file),
        expected_samples_file=str(sample_file), compact_cache_bytes=8 * 2**20)
    before = deepcopy(config)
    analyses, specs = cli.configuration_inputs(config)
    assert config == before
    assert analyses == [analysis, analysis]
    assert list(specs) == [(tmp_path / "source.gds").resolve()]
    spec = next(iter(specs.values()))
    assert isinstance(spec, CacheSpec)
    assert spec.directory == tmp_path / "cache"
    assert spec.expected_binding == live_proof["binding"]
    assert spec.source_proof is live_proof["function"]
    np.testing.assert_array_equal(spec.expected_samples, samples)
    assert spec.expected_samples.dtype == np.uint32
    assert spec.compact_cache_bytes == 8 * 2**20
    assert spec.source_proof() == spec.expected_binding
    live_proof["source"].write_text(json.dumps({"synthetic_source_revision": 2}))
    assert spec.source_proof() != spec.expected_binding


def test_configuration_inputs_defaults_preserve_explicit_cache_contract(tmp_path, live_proof):
    analyses, specs = cli.configuration_inputs(configuration(tmp_path, live_proof))
    assert len(analyses) == 1
    spec = next(iter(specs.values()))
    assert spec.expected_samples is None
    assert spec.compact_cache_bytes == 64 * 2**20


@pytest.mark.parametrize("reference", [None, 1, "missing_colon", ":function", "module:", "module:fn:extra"])
def test_configuration_inputs_rejects_invalid_source_proof_reference(tmp_path, live_proof, reference):
    config = configuration(tmp_path, live_proof)
    config["caches"][0]["source_proof"] = reference
    with pytest.raises(ValueError, match="module:callable"):
        cli.configuration_inputs(config)


def test_configuration_inputs_rejects_noncallable_source_proof(tmp_path, live_proof):
    config = configuration(tmp_path, live_proof)
    config["caches"][0]["source_proof"] = live_proof["module"].__name__ + ":noncallable"
    with pytest.raises(ValueError, match="must be callable"):
        cli.configuration_inputs(config)


@pytest.mark.parametrize("alias", ["identical", "parent", "relative"])
def test_configuration_inputs_rejects_duplicate_canonical_gds_paths(tmp_path, live_proof, monkeypatch, alias):
    config = configuration(tmp_path, live_proof)
    duplicate = deepcopy(config["caches"][0])
    if alias == "parent":
        duplicate["gds"] = str(tmp_path / "unused_directory" / ".." / "source.gds")
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        duplicate["gds"] = "source.gds"
    config["caches"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate cache specification"):
        cli.configuration_inputs(config)


def test_configuration_inputs_rejects_duplicate_symlink_gds_alias(tmp_path, live_proof):
    original, alias = tmp_path / "source.gds", tmp_path / "source_alias.gds"
    original.write_bytes(b"synthetic path identity only")
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("symlink creation is unavailable on this filesystem")
    config = configuration(tmp_path, live_proof)
    duplicate = deepcopy(config["caches"][0])
    duplicate["gds"] = str(alias)
    config["caches"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate cache specification"):
        cli.configuration_inputs(config)


def test_configuration_inputs_refuses_pickled_sample_axes(tmp_path, live_proof):
    config = configuration(tmp_path, live_proof)
    samples = tmp_path / "unsafe_object_rows.npy"
    np.save(samples, np.asarray(["row_001"], dtype=object))
    config["caches"][0]["expected_samples_file"] = str(samples)
    with pytest.raises(ValueError, match="Object arrays"):
        cli.configuration_inputs(config)


@pytest.mark.parametrize("unknown", ["maximum_mask_variants", "device", "ignored_option"])
def test_cli_rejects_unknown_shared_options_before_running(tmp_path, live_proof, monkeypatch, unknown):
    config = configuration(tmp_path, live_proof)
    config["shared_options"] = {unknown: 1}
    source, report = tmp_path / "config.json", tmp_path / "execution.json"
    source.write_text(json.dumps(config))
    called = []
    monkeypatch.setattr(cli, "run_configuration", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="unknown shared options"):
        cli.main([str(source), "--report", str(report)])
    assert called == [] and not report.exists()


def test_cli_forwards_device_options_and_writes_the_returned_private_report(tmp_path, live_proof, monkeypatch):
    config = configuration(tmp_path, live_proof)
    shared_options = dict(device_cache_bytes=12 * 2**20, compact_cache_bytes=8 * 2**20,
                          metadata_cache_bytes=4 * 2**20)
    config["shared_options"] = shared_options
    source, report = tmp_path / "config.json", tmp_path / "reports" / "execution.json"
    source.write_text(json.dumps(config))
    calls = []
    expected_report = dict(schema_version=1, models=[dict(index=0, n=3)], verified=True)
    def execute(analyses, **kwargs):
        calls.append((analyses, kwargs))
        return expected_report
    monkeypatch.setattr(cli, "run_configuration", execute)
    actual = cli.main([str(source), "--device", "cuda:1", "--report", str(report)])
    assert actual is expected_report
    assert json.loads(report.read_text()) == expected_report
    assert report.read_text().endswith("\n")
    analyses, kwargs = calls[0]
    assert analyses == config["analyses"]
    assert kwargs["device"] == "cuda:1"
    for key, value in shared_options.items():
        assert kwargs[key] == value
    assert list(kwargs["cache_specs"]) == [(tmp_path / "source.gds").resolve()]


def test_cli_failed_run_preserves_an_existing_report(tmp_path, live_proof, monkeypatch):
    source, report = tmp_path / "config.json", tmp_path / "execution.json"
    source.write_text(json.dumps(configuration(tmp_path, live_proof)))
    previous = '{"previous_complete_run": true}\n'
    report.write_text(previous)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic verification failure")
    monkeypatch.setattr(cli, "run_configuration", fail)
    with pytest.raises(RuntimeError, match="verification failure"):
        cli.main([str(source), "--report", str(report)])
    assert report.read_text() == previous


def test_cli_requires_an_explicit_private_report_path(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main([str(tmp_path / "config.json")])
    assert error.value.code == 2
