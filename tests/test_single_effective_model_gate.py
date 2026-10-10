"""CPU-only contracts for the production Single effective-block gate.

Real model classes carry a device descriptor, without creating CUDA tensors.
An empty reader exercises the real iterator selection without association work.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from staar_phewas.binary_null import BinaryNullModel
from staar_phewas.null_model import GaussianNullModel
from staar_phewas.pipeline import AnalysisOptions, PheWASPipeline
from staar_phewas.profiling import StageProfiler


class EmptyReader:
    n_variants = 5

    def __init__(self):
        self.calls = []

    def _read(self, kind, variants, samples, **options):
        self.calls.append((kind, np.asarray(variants).copy(),
                           np.asarray(samples).copy(), options))
        return iter(())

    def iter_minor_blocks(self, variants, samples, **options):
        return self._read("ordinary", variants, samples, **options)


class EffectiveEmptyReader(EmptyReader):
    def iter_effective_minor_blocks(self, variants, samples, **options):
        return self._read("effective", variants, samples, **options)


def make_model(model_class, *, device="cuda:0", mode="tf32", spa=False,
               n_pheno=1):
    model = model_class.__new__(model_class)
    # The production gate reads x.device; this descriptor allocates no tensor.
    model.x = SimpleNamespace(device=device, shape=(3, 2))
    model.sample_ids = np.asarray(["sample_a", "sample_b", "sample_c"])
    model.n_pheno = n_pheno
    model.use_spa = spa
    model.matmul_mode = mode
    return model


def make_pipeline(model, *, optimized=True, resident=True,
                  effective_reader=True):
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.models = [model]
    pipeline.options = AnalysisOptions(genotype_block_size=7,
                                       annotation_block_size=3,
                                       wrapper_semantics="base")
    pipeline.union_rows = np.asarray([4, 1, 3], dtype=np.int64)
    pipeline.gds = EffectiveEmptyReader() if effective_reader else EmptyReader()
    pipeline.single_batch_optimization = optimized
    pipeline.resident_genotypes = resident
    pipeline.individual_effective_block_size = 11
    pipeline._base_mask = lambda *args: np.asarray([True, False, True,
                                                   True, False])
    pipeline.profiler = StageProfiler(device="cpu", enabled=False)
    pipeline._prepare_individual_block = Mock(
        side_effect=AssertionError("an empty reader must not compute a block"))
    return pipeline


class SingleEffectiveModelGateTests(unittest.TestCase):
    def assert_reader_gate(self, pipeline, *, effective):
        # Even a CUDA-tagged fake model must leave CUDA initialization untouched.
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError(
                "this gate contract must stay CPU-only")) as initialize:
            self.assertEqual(list(pipeline.iter_individual_records(
                "1", mac_cutoff=23)), [])
        initialize.assert_not_called()
        pipeline._prepare_individual_block.assert_not_called()
        self.assertEqual(len(pipeline.gds.calls), 2)
        expected_variants = ([0, 2], [3])
        device = next((model.device for model in pipeline.models
                       if str(model.device).startswith("cuda")), None)
        for call, variants in zip(pipeline.gds.calls, expected_variants):
            kind, selected, rows, options = call
            self.assertEqual(kind, "effective" if effective else "ordinary")
            np.testing.assert_array_equal(selected, variants)
            np.testing.assert_array_equal(rows, pipeline.union_rows)
            expected = {"block_size": 7}
            if device is not None:
                expected.update(device=device, minimum_mac=23,
                                resident=pipeline.resident_genotypes)
            if effective:
                expected["effective_block_size"] = getattr(
                    pipeline, "individual_effective_block_size", 1024)
            self.assertEqual(options, expected)

    def test_real_gaussian_and_non_spa_binary_select_effective_reader(self):
        for model_class in (GaussianNullModel, BinaryNullModel):
            with self.subTest(model=model_class.__name__):
                self.assert_reader_gate(make_pipeline(make_model(model_class)),
                                        effective=True)

    def test_model_modes_retain_ordinary_reader(self):
        cases = (
            {"spa": True},
            {"mode": "fp64"},
            {"n_pheno": 2},
            {"device": "cpu"},
        )
        for model_class in (GaussianNullModel, BinaryNullModel):
            for options in cases:
                with self.subTest(model=model_class.__name__, **options):
                    self.assert_reader_gate(make_pipeline(make_model(
                        model_class, **options)), effective=False)

    def test_reader_and_pipeline_capabilities_retain_ordinary_reader(self):
        cases = (
            {"optimized": False},
            {"resident": False},
            {"effective_reader": False},
        )
        for model_class in (GaussianNullModel, BinaryNullModel):
            for options in cases:
                with self.subTest(model=model_class.__name__, **options):
                    self.assert_reader_gate(make_pipeline(make_model(
                        model_class), **options), effective=False)

    def test_multiple_models_retain_ordinary_reader(self):
        for model_class in (GaussianNullModel, BinaryNullModel):
            with self.subTest(model=model_class.__name__):
                pipeline = make_pipeline(make_model(model_class))
                pipeline.models.append(make_model(model_class))
                self.assert_reader_gate(pipeline, effective=False)

    def test_missing_optimization_flag_retain_ordinary_reader(self):
        pipeline = make_pipeline(make_model(BinaryNullModel))
        del pipeline.single_batch_optimization
        self.assert_reader_gate(pipeline, effective=False)

    def test_missing_mode_defaults_to_ordinary_reader(self):
        pipeline = make_pipeline(make_model(BinaryNullModel))
        del pipeline.models[0].matmul_mode
        # The dataclass's FP64 class default applies after attribute deletion.
        self.assert_reader_gate(pipeline, effective=False)

    def test_unknown_model_type_retain_ordinary_reader(self):
        model = SimpleNamespace(device="cuda:0", n_pheno=1, use_spa=False,
                                matmul_mode="tf32")
        self.assert_reader_gate(make_pipeline(model), effective=False)

    def test_default_effective_block_size_remains_1024(self):
        pipeline = make_pipeline(make_model(BinaryNullModel))
        del pipeline.individual_effective_block_size
        self.assert_reader_gate(pipeline, effective=True)


if __name__ == "__main__":
    unittest.main()
