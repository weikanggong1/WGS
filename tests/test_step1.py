"""Focused Step1 correctness tests; fixtures are not performance benchmarks."""
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from torchwgs.step1 import (
    NullModel, Step1Config, contiguous_folds, covariate_basis,
    fit_null, inverse_normal_transform,
)


class ArrayReader:
    def __init__(self, genotypes, chromosomes, sample_ids):
        self.matrix = genotypes
        self.variant_chromosomes = chromosomes
        self.n_samples, self.n_variants = genotypes.shape
        self.sample_ids = sample_ids
        self.largest_read = 0

    def iter_blocks(self, block_size, indices=None):
        indices = np.arange(self.n_variants) if indices is None else np.asarray(indices)
        for begin in range(0, len(indices), block_size):
            selected = indices[begin:begin + block_size]
            self.largest_read = max(self.largest_read, len(selected))
            yield selected, self.matrix[:, selected]


def direct_ridge_reference(g, y, chromosome, block_size, h0, h1, folds):
    """CPU double reference using direct training-row solves, rather than Gram subtraction."""
    g = g.numpy().copy()
    y = y.numpy().copy()
    n, m = g.shape
    g = np.where(np.isnan(g), np.nanmean(g, axis=0), g)
    g -= g.mean(axis=0)
    g /= np.sqrt(np.sum(g ** 2, axis=0) / (n - 1))
    y -= y.mean()
    y /= np.sqrt(np.sum(y ** 2) / (n - 1))
    features = []
    feature_chromosomes = []
    for chrom in sorted(set(chromosome.tolist())):
        indices = np.flatnonzero(chromosome == chrom)
        for begin in range(0, len(indices), block_size):
            x = g[:, indices[begin:begin + block_size]]
            p = np.empty((n, len(h0)))
            for heldout in folds:
                train = np.ones(n, dtype=bool)
                train[heldout] = False
                for parameter, h in enumerate(h0):
                    beta = np.linalg.solve(x[train].T @ x[train] + m * (1 - h) / h * np.eye(x.shape[1]),
                                           x[train].T @ y[train])
                    p[heldout, parameter] = x[heldout] @ beta
            p -= p.mean(axis=0)
            p /= p.std(axis=0, ddof=1)
            features.append(p)
            feature_chromosomes.extend([chrom] * len(h0))
    x = np.column_stack(features)
    k = x.shape[1]
    predictions = np.empty((n, len(h1)))
    all_betas = []
    for heldout in folds:
        train = np.ones(n, dtype=bool)
        train[heldout] = False
        betas = np.column_stack([np.linalg.solve(x[train].T @ x[train] + k * (1 - h) / h * np.eye(k),
                                               x[train].T @ y[train]) for h in h1])
        predictions[heldout] = x[heldout] @ betas
        all_betas.append(betas)
    mse = np.mean((predictions - y[:, None]) ** 2, axis=0)
    selected = mse.argmin()
    loco = np.empty((n, 22))
    for heldout, betas in zip(folds, all_betas):
        beta = betas[:, selected]
        for chrom in range(1, 23):
            remaining = np.asarray(feature_chromosomes) != chrom
            loco[heldout, chrom - 1] = x[heldout][:, remaining] @ beta[remaining]
    return loco, mse, selected


class Step1Tests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1303)
        self.g = torch.randint(0, 3, (53, 17), generator=generator).double()
        self.g[2, 3] = float("nan")
        self.y = torch.randn(53, generator=generator).double() + 0.35 * self.g[:, 1]
        self.chrom = np.array([1] * 10 + [2] * 7)
        self.ids = [(str(100 + i), str(100 + i)) for i in range(53)]
        self.config = Step1Config(block_size=4, folds=5, apply_rint=False,
                                  device="cpu", dtype="float64", sample_chunk_size=7)

    def test_blom_average_ties_and_missing(self):
        actual = inverse_normal_transform(torch.tensor([4., 1., 1., 7., float("nan")]))
        ranks = [3., 1.5, 1.5, 4.]
        expected = torch.tensor([NormalDist().inv_cdf((rank - .375) / 4.25) for rank in ranks], dtype=torch.float64)
        torch.testing.assert_close(actual[:4], expected, rtol=0, atol=2e-15)
        self.assertTrue(torch.isnan(actual[4]))

    def test_contiguous_remainder_and_covariate_rank(self):
        self.assertEqual([(fold.start, fold.stop) for fold in contiguous_folds(13, 5)],
                         [(0, 2), (2, 4), (4, 6), (6, 8), (8, 13)])
        x = torch.arange(13, dtype=torch.float64)
        cov = torch.stack((x, x * 2, torch.ones_like(x)), dim=1)
        q = covariate_basis(cov)
        self.assertEqual(q.shape, (13, 2))
        torch.testing.assert_close(q.T @ q, torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-13)

    def test_double_matches_independent_direct_training_solve(self):
        model = fit_null(self.g, self.y, self.chrom, config=self.config)
        expected, mse, selected = direct_ridge_reference(self.g, self.y, self.chrom,
            self.config.block_size, self.config.ridge_l0, self.config.ridge_l1,
            contiguous_folds(len(self.y), self.config.folds))
        np.testing.assert_allclose(model.loco.numpy(), expected, rtol=5e-12, atol=3e-13)
        np.testing.assert_allclose(model.metadata["cv_mse"], mse, rtol=5e-12, atol=3e-13)
        self.assertEqual(model.metadata["selected_ridge_l1_index"], int(selected))
        self.assertFalse(model.metadata["final_l1_refit_full_samples"])

    def test_reader_memory_disk_and_missing_sample_alignment(self):
        y = self.y.clone()
        y[4] = float("nan")
        reader = ArrayReader(self.g, self.chrom, self.ids)
        with tempfile.TemporaryDirectory() as directory:
            disk = fit_null(reader, y, config=replace(self.config, keep_l0=True), output_dir=directory)
            memory = fit_null(self.g, y, self.chrom, config=replace(self.config, l0_storage="memory"), sample_ids=self.ids)
            torch.testing.assert_close(disk.loco, memory.loco, rtol=0, atol=0)
            self.assertEqual(reader.largest_read, 4)
            self.assertNotIn(self.ids[4], disk.sample_ids)
            self.assertEqual(len(disk.sample_indices), 52)
            self.assertTrue(Path(disk.metadata["level0_file"]).exists())
            loaded = NullModel.load(directory)
            torch.testing.assert_close(loaded.loco, disk.loco)

    def test_masked_rows_preserve_original_fold_boundaries(self):
        y = self.y.clone()
        y[4] = float("nan")
        original = fit_null(self.g, y, self.chrom, config=self.config)
        compact = fit_null(self.g, y, self.chrom,
                           config=replace(self.config, preserve_masked_rows=False))
        self.assertEqual(original.metadata["n_internal_rows"], 53)
        self.assertEqual(original.metadata["n_samples"], 52)
        self.assertEqual(original.metadata["fold_sizes"], [11, 10, 10, 10, 12])
        self.assertEqual(original.metadata["fold_active_sizes"], [10, 10, 10, 10, 12])
        self.assertEqual(original.sample_indices.tolist(), compact.sample_indices.tolist())
        # REGENIE's centered L0 features at zero-masked rows contribute to L1.
        self.assertGreater(float((original.loco-compact.loco).abs().max()), 1e-8)

    def test_regenie_text_layout_alignment_and_gzip(self):
        model = fit_null(self.g, self.y, self.chrom, config=self.config, sample_ids=self.ids)
        with tempfile.TemporaryDirectory() as directory:
            path, prediction_list = model.export_regenie(Path(directory) / "step1", phenotype_name="T1")
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 24)  # Header plus 23 chromosome rows.
            self.assertTrue(all(line.endswith(" ") for line in lines))
            self.assertEqual(prediction_list.name, "step1_pred.list")
            self.assertEqual(path.name, "step1_1.loco")
            imported = NullModel.from_regenie(prediction_list, phenotype_name="T1", sample_ids=list(reversed(self.ids)),
                                              chromosomes=range(1, 23))
            torch.testing.assert_close(imported.loco, model.loco.flip(0), rtol=5e-6, atol=5e-7)
            path, _ = model.export_regenie(Path(directory) / "compressed", compressed=True)
            imported_all = NullModel.from_regenie(path)
            torch.testing.assert_close(imported_all.loco[:, -1], model.prs, rtol=5e-6, atol=5e-7)

    def test_failed_save_and_export_preserve_completed_outputs(self):
        model = NullModel(self.ids[:2], torch.tensor([[1., 2.], [3., 4.]]),
                          chromosomes=(1, 2), prs=torch.tensor([3., 7.]),
                          metadata={"phenotype_name": "T1", "version": "old"})
        with tempfile.TemporaryDirectory() as directory:
            target = model.save(Path(directory) / "model.pt")
            loco, prediction_list = model.export_regenie(Path(directory) / "step1")
            completed = {path: path.read_bytes() for path in
                         (target, target.with_suffix(".json"), loco, prediction_list)}
            model.loco += 1
            model.metadata["version"] = "new"
            # JSON serialization fails after the replacement tensor is written.
            model.metadata["invalid_json"] = {1}
            with self.assertRaises(TypeError):
                model.save(target)
            del model.metadata["invalid_json"]
            original_replace = Path.replace
            for action, failing_target in ((lambda: model.save(target), target.with_suffix(".json")),
                                           (lambda: model.export_regenie(Path(directory) / "step1"), prediction_list)):
                def interrupted_replace(source, destination):
                    if Path(destination) == failing_target and source.name.endswith(".partial"):
                        raise OSError("Simulated second-file commit failure")
                    return original_replace(source, destination)
                with patch.object(Path, "replace", interrupted_replace), self.assertRaises(OSError):
                    action()
                for path, contents in completed.items():
                    self.assertEqual(path.read_bytes(), contents)
                self.assertFalse(list(Path(directory).glob("*.partial")))
                self.assertFalse(list(Path(directory).glob("*.backup")))
            torch.testing.assert_close(NullModel.load(target).loco,
                                       torch.tensor([[1., 2.], [3., 4.]]))

    def test_reject_low_precision_and_bad_variants(self):
        with self.assertRaises(ValueError):
            Step1Config(dtype="float16")
        with self.assertRaises(ValueError):
            fit_null(torch.ones_like(self.g), self.y, self.chrom, config=self.config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_float32_agrees_with_double_for_fixture(self):
        double = fit_null(self.g, self.y, self.chrom, config=self.config)
        gpu = fit_null(self.g, self.y, self.chrom,
                       config=replace(self.config, device="cuda", dtype="float32", tf32=True))
        torch.testing.assert_close(gpu.loco.double(), double.loco, rtol=5e-3, atol=5e-4)
        self.assertGreater(gpu.metadata["peak_gpu_allocated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
