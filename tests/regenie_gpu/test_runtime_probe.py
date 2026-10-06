"""CPU differential/guard tests for instrumentation, never runtime benchmarks."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torchwgs.io import BedReader, PackedBedBlock, Variant, write_bed
from torchwgs.mask_output import MaskWriter
from torchwgs.masks import Annotation, GeneConfig, GeneSet, MaskDefinition, build_gene_masks
from torchwgs.output import RegenieWriter
from torchwgs.single import SingleVariantConfig, create_test_context, iter_single_variant_results
from torchwgs.runtime_estimate import StageTimers, estimate_gene_strata
from torchwgs import runtime_probe as probe


def fixture(directory):
    ids = [("synthetic", str(i)) for i in range(37)]
    variants = [Variant(i, "21", "synthetic_v" + str(i), i + 1, "A", "G") for i in range(11)]
    rng = np.random.default_rng(911)
    genotype = rng.integers(0, 3, size=(37, 11)).astype(np.float64)
    genotype[:, 0] = 0
    genotype[:, 1] = np.nan
    genotype[:4, 2] = np.nan
    genotype[:, 3] = 0
    genotype[:3, 3] = 1
    prefix = Path(directory) / "source"
    write_bed(prefix, genotype, variants, ids)
    return prefix, ids, variants, rng


class InstrumentationDifferentialTests(unittest.TestCase):
    def test_production_single_rows_and_native_files_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, ids, _, rng = fixture(directory)
            reader = BedReader(prefix)
            context = create_test_context(rng.normal(size=len(ids)), sample_ids=ids[::-1],
                device="cpu", dtype="float64", apply_rint=False)
            config = SingleVariantConfig(device="cpu", dtype="float64", tf32=False,
                genotype_reader="cpu", min_mac=3, block_size=3)
            selected = [10, 2, 3, 0, 1, 6]
            expected = list(iter_single_variant_results(reader, context, config=config, variant_indices=selected))
            original_bed = reader._bed
            original_write = RegenieWriter.write
            timers = StageTimers(synchronize=lambda: None)
            paths = []
            for label, instrumented in (("plain", False), ("profile", True)):
                manager = (probe.RuntimeInstrumentation(timers, reader) if instrumented
                           else contextlib.nullcontext())
                with manager:
                    actual = list(iter_single_variant_results(reader, context, config=config, variant_indices=selected))
                    self.assertEqual(actual, expected)
                    with RegenieWriter(Path(directory)/label, "synthetic_trait", sample_ids=context.sample_ids,
                            write_samples=True, print_pheno_name=True) as writer:
                        for row in actual:
                            writer.write(row)
                    paths.append((writer.path, writer.ids_path))
            self.assertIs(reader._bed, original_bed)
            self.assertIs(RegenieWriter.write, original_write)
            for left, right in zip(*paths):
                self.assertEqual(left.read_bytes(), right.read_bytes())
            summary = timers.summary()
            self.assertGreater(summary["single_score"]["calls"], 0)
            self.assertEqual(summary["association_text_write"]["completed_calls"], len(expected))
            self.assertTrue(all(entry["failed_calls"] == 0 for entry in summary.values()))

    def test_packed_geometry_counts_and_decoder_bits_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _, _, _ = fixture(directory)
            reader = BedReader(prefix)
            before = reader.read_packed_block([9, 2, 0, 1], sample_rows=[36, 2, 0, 8],
                device="cpu", dtype=torch.float64)
            expected_counts = before.allele_counts()
            expected = before.decode([3, 1, 1]).view(torch.int64).clone()
            original_decode = PackedBedBlock.decode
            with probe.RuntimeInstrumentation(StageTimers(synchronize=lambda: None), reader) as hooks:
                block = reader.read_packed_block([9, 2, 0, 1], sample_rows=[36, 2, 0, 8],
                    device="cpu", dtype=torch.float64)
                actual_counts = block.allele_counts()
                actual = block.decode([3, 1, 1])
                self.assertEqual(hooks.geometry["decoded_columns"], 3)
                self.assertEqual(hooks.geometry["packed_bytes"], block.packed.numel())
            self.assertIs(PackedBedBlock.decode, original_decode)
            self.assertTrue(torch.equal(actual.view(torch.int64), expected))
            for field in expected_counts:
                self.assertTrue(torch.equal(actual_counts[field], expected_counts[field]))

    def test_mask_generator_and_special_method_writer_preserve_all_artifact_bytes(self):
        values = torch.zeros((37, 3), dtype=torch.float64)
        values[:5, 0] = 1; values[8:16, 1] = 1; values[17:26, 2] = 1
        values[1, :] = float("nan")
        identifiers = ["synthetic_v0", "synthetic_v1", "synthetic_v2"]
        annotations = [Annotation(v, "synthetic_gene", "A", "R") for v in identifiers]
        definitions = [MaskDefinition("M", frozenset(["A"]))]
        config = GeneConfig(aaf_bins=(1.,), vc_max_aaf=1., min_mac=1,
                            collapse_mac=0, include_singletons=False)
        original_call = MaskWriter.__call__
        with tempfile.TemporaryDirectory() as directory:
            prefixes = []
            for label in ("plain", "profile"):
                timers = StageTimers(synchronize=lambda: None)
                manager = probe.RuntimeInstrumentation(timers) if label == "profile" else contextlib.nullcontext()
                with manager as hooks:
                    masks = build_gene_masks(values, identifiers, annotations, definitions, config)
                    writer = MaskWriter(Path(directory)/label, [("synthetic", str(i)) for i in range(37)], [0]*37)
                    writer(SimpleNamespace(gene=GeneSet("synthetic_gene", "21", 1, tuple(identifiers)), masks=masks))
                    writer.close()
                    prefixes.append(writer.prefix)
                    if label == "profile":
                        self.assertEqual(hooks.geometry["masks"], len(masks))
                        self.assertGreater(hooks.geometry["max_vc_columns"], 0)
                        self.assertEqual(timers.summary()["mask_pack_transfer_write"]["completed_calls"], 1)
                        self.assertTrue(all(s["failed_calls"] == 0 for s in timers.summary().values()))
            self.assertIs(MaskWriter.__call__, original_call)
            for suffix in (".bed", ".bim", ".fam", ".snplist"):
                self.assertEqual(Path(prefixes[0]+suffix).read_bytes(), Path(prefixes[1]+suffix).read_bytes())

    def test_keyword_sbat_invocation_preserves_actual_result_and_geometry(self):
        from torchwgs import statistics as stats
        score = torch.tensor([.7, -.4], dtype=torch.float64)
        covariance = torch.tensor([[1., .2], [.2, 1.]], dtype=torch.float64)
        expected = stats.sbat_logp(score_vec=score, cov_mat=covariance)
        with probe.RuntimeInstrumentation(StageTimers(synchronize=lambda: None)) as hooks:
            actual = stats.sbat_logp(score_vec=score, cov_mat=covariance)
            self.assertEqual(hooks.geometry["max_sbat_columns"], 2)
        for field in expected:
            if isinstance(expected[field], torch.Tensor):
                torch.testing.assert_close(actual[field], expected[field], atol=0, rtol=0)
            else:
                self.assertEqual(actual[field], expected[field])

    def test_body_exception_restores_every_hook_and_reader(self):
        from torchwgs import single
        with tempfile.TemporaryDirectory() as directory:
            prefix, _, _, _ = fixture(directory)
            reader = BedReader(prefix)
            originals = (reader._bed, BedReader.read_packed_block, PackedBedBlock.decode,
                         single._score_counted_genotypes, MaskWriter.__call__, torch.linalg.eigh)
            with self.assertRaisesRegex(ArithmeticError, "synthetic failure"):
                with probe.RuntimeInstrumentation(StageTimers(synchronize=lambda: None), reader):
                    raise ArithmeticError("synthetic failure")
            restored = (reader._bed, BedReader.read_packed_block, PackedBedBlock.decode,
                        single._score_counted_genotypes, MaskWriter.__call__, torch.linalg.eigh)
            self.assertTrue(all(a is b for a, b in zip(originals, restored)))

    def test_partial_instrumentation_install_failure_restores_prior_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _, _, _ = fixture(directory)
            reader = BedReader(prefix)
            bed, read = reader._bed, BedReader.read_packed_block
            hooks = probe.RuntimeInstrumentation(StageTimers(synchronize=lambda: None), reader)
            original_wrap = hooks._wrap
            calls = []
            def fail_after_first_hook(*args, **kwargs):
                calls.append(None)
                if len(calls) == 2:
                    raise RuntimeError("synthetic installation failure")
                original_wrap(*args, **kwargs)
            with patch.object(hooks, "_wrap", side_effect=fail_after_first_hook), \
                    self.assertRaisesRegex(RuntimeError, "synthetic installation failure"):
                hooks.__enter__()
            self.assertIs(reader._bed, bed)
            self.assertIs(BedReader.read_packed_block, read)


class MetadataAndCensusTests(unittest.TestCase):
    def test_metadata_sampling_preserves_source_geometry_order_and_restores_method(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _, _, _ = fixture(directory)
            reader = BedReader(prefix)
            original = BedReader._iter_variant_metadata_blocks
            variants = list(reader.iter_variants())
            windows = [(1, 4), (9, 11)]
            sampled = probe._metadata_windows(reader, windows, StageTimers())
            self.assertEqual([[v.index for v in block] for block in sampled], [[1, 2, 3], [9, 10]])
            with probe._sample_metadata(reader, sampled[0]):
                blocks = list(reader._iter_variant_metadata_blocks(2, [1, 2, 3]))
                self.assertEqual([[v.index for v in b] for b in blocks], [[1, 2], [3]])
                self.assertTrue(all(v is variants[v.index] or v == variants[v.index] for b in blocks for v in b))
                self.assertEqual(reader.n_variants, 11)
                with self.assertRaises(ValueError):
                    list(reader._iter_variant_metadata_blocks(2, [3, 2, 1]))
                other = BedReader(prefix)
                with self.assertRaises(ValueError):
                    list(other._iter_variant_metadata_blocks(2, [1, 2, 3]))
            self.assertIs(BedReader._iter_variant_metadata_blocks, original)

    def test_census_strata_use_full_unique_genes_and_separate_analysis_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            setlist = Path(directory)/"setlist"
            sizes = [100, 101, 1000, 1001, 5000, 5001]
            setlist.write_text("# synthetic fixture\n" + "".join(
                "synthetic_gene%d chr21 %d %s\n" % (i, i+1, ",".join("v%d" % j for j in range(size)))
                for i, size in enumerate(sizes)))
            analyses = [SimpleNamespace(setlist_file=setlist), SimpleNamespace(setlist_file=setlist)]
            census, jobs = probe._gene_census(analyses, GeneConfig(), probe.ProbeConfig(genes_per_stratum=2))
            counts = {r["stratum"]: r["count"] for r in census}
            for ordinal in (0, 1):
                self.assertEqual(counts["a%d_0001_0100" % ordinal], 1)
                self.assertEqual(counts["a%d_0101_1000" % ordinal], 2)
                self.assertEqual(counts["a%d_1001_5000" % ordinal], 2)
                self.assertEqual(counts["a%d_5001_plus" % ordinal], 1)
            self.assertEqual(sum(counts.values()), 12)
            self.assertEqual(len(jobs), 12)
            self.assertEqual(max(len(gene.variant_ids) for _, (_, gene) in jobs), 5001)
            self.assertTrue(all(gene.chrom == "21" for _, (_, gene) in jobs))
            serialized = json.dumps(census)
            self.assertNotIn("synthetic_gene", serialized)
            self.assertNotIn(str(setlist), serialized)
            estimate_gene_strata(census, [])  # Labels must be valid anonymous estimator categories.

    def test_census_reservoir_is_seeded_and_duplicate_members_stay_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            setlist = Path(directory)/"setlist"
            setlist.write_text("".join("synthetic_gene%d 21 %d v2,v1,v2,v0\n" % (i, i+1) for i in range(20)))
            analyses = [SimpleNamespace(setlist_file=setlist)]
            config = probe.ProbeConfig(seed=73, genes_per_stratum=3)
            census, first = probe._gene_census(analyses, GeneConfig(), config)
            _, second = probe._gene_census(analyses, GeneConfig(), config)
            self.assertEqual(first, second)
            self.assertEqual(census, [{"stratum": "a0_0001_0100", "count": 20}])
            self.assertEqual(len(first), 3)
            self.assertTrue(all(g.variant_ids == ("v2", "v1", "v0") for _, (_, g) in first))
            selected = GeneConfig(extract_genes=frozenset(["synthetic_gene3"]))
            filtered, jobs = probe._gene_census(analyses, selected, config)
            self.assertEqual(filtered[0]["count"], 1)
            self.assertEqual(jobs[0][1][1].gene, "synthetic_gene3")


class ProbeGuardAndSupervisorTests(unittest.TestCase):
    def test_unknown_empty_census_is_not_zero_but_confirmed_empty_census_is(self):
        observations = {"gene_census": [], "gene_samples": []}
        unknown = probe._gene_estimate(dict(observations, gene_census_complete=False))
        self.assertIsNone(unknown["total_seconds"])
        self.assertIsNone(unknown["range_seconds"])
        confirmed = probe._gene_estimate(dict(observations, gene_census_complete=True))
        self.assertEqual(confirmed["total_seconds"], 0.)
        self.assertEqual(confirmed["range_seconds"], {"min": 0., "max": 0.})

    def test_cuda_only_guard_rejects_before_input_reads_or_association(self):
        from torchwgs.config import WGSConfig
        inputs = SimpleNamespace(wgs_prefixes={"21": "synthetic_not_read"}, imported_loco="synthetic_not_read")
        with patch.object(probe.torch.cuda, "is_available", return_value=False), \
                patch.object(probe, "_load_context") as load, self.assertRaisesRegex(ValueError, "requires CUDA"):
            probe.probe_discovery(inputs, "21", configuration=WGSConfig.paper())
        load.assert_not_called()

    def test_invalid_probe_budgets_cannot_start_work(self):
        for args in ({"sample_blocks": True}, {"genes_per_stratum": 0}, {"io_repeats": 1.5},
                     {"budget_seconds": float("nan")}, {"gene_seconds": -1},
                     {"max_gpu_gb": True}, {"run_gene": 1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                probe.ProbeConfig(**args)

    def test_supervisor_timeout_reaps_process_group_and_preserves_censored_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            calls = []
            child = SimpleNamespace(pid=987654)
            def wait(timeout=None):
                calls.append(timeout)
                if timeout is not None:
                    probe._atomic_report(output, {"status": "running", "complete": False,
                        "gene_samples": [{"stratum": "a0_large", "seconds": 15., "complete": False}]})
                    raise subprocess.TimeoutExpired("synthetic worker", timeout)
                return -9
            child.wait = wait
            argv = ["--inputs", str(Path(directory)/"private.json"), "--chromosome", "21",
                    "--out", str(output), "--budget-seconds", "1"]
            printed = io.StringIO()
            with patch.object(probe.subprocess, "Popen", return_value=child) as launch, \
                    patch.object(probe.os, "killpg") as kill, contextlib.redirect_stdout(printed):
                self.assertEqual(probe.main(argv), 2)
            kill.assert_called_once_with(child.pid, probe.signal.SIGKILL)
            self.assertEqual(calls, [1., None])
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            command = launch.call_args.args[0]
            self.assertEqual(command[1:3], ["-m", "torchwgs.runtime_probe"])
            self.assertEqual(command[-1], "--_worker")
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "hard_budget_exhausted")
            self.assertFalse(report["complete"])
            self.assertFalse(report["full_pipeline_measured"])
            self.assertIsNone(report["whole_pipeline_estimate_seconds"])
            self.assertFalse(report["gene_samples"][0]["complete"])
            self.assertNotIn("private.json", printed.getvalue())

    def test_existing_report_cannot_be_overwritten_or_launch_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            output.write_text("original evidence")
            with patch.object(probe.subprocess, "Popen") as launch, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.main(["--inputs", "synthetic.json", "--chromosome", "21", "--out", str(output)])
            launch.assert_not_called()
            self.assertEqual(output.read_text(), "original evidence")

    def test_worker_error_only_exports_exception_type_and_never_error_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            with patch.object(probe, "_load_inputs", return_value=SimpleNamespace()), \
                    patch.object(probe, "probe_discovery", side_effect=ValueError("/private/sensitive_input synthetic_gene")):
                code = probe.main(["--inputs", "synthetic.json", "--chromosome", "21", "--out", str(output), "--_worker"])
            self.assertEqual(code, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["error_type"], "ValueError")
            self.assertFalse(report["complete"])
            self.assertIsNone(report["whole_pipeline_estimate_seconds"])
            self.assertNotIn("/private/", output.read_text())
            self.assertNotIn("synthetic_gene", output.read_text())

    def test_worker_invalid_config_writes_anonymous_failure_before_association(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            config = Path(directory)/"synthetic_private_config.json"
            config.write_text("invalid /private/sensitive_config")
            with patch.object(probe, "probe_discovery") as association:
                code = probe.main(["--inputs", "synthetic.json", "--chromosome", "21", "--out", str(output),
                                   "--config", str(config), "--_worker"])
            association.assert_not_called()
            self.assertEqual(code, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["complete"])
            self.assertIsNone(report["whole_pipeline_estimate_seconds"])
            self.assertNotIn(str(config), output.read_text())
            self.assertNotIn("sensitive_config", output.read_text())

    def test_child_early_exit_without_report_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            child = SimpleNamespace(pid=987654, wait=lambda timeout=None: 17)
            with patch.object(probe.subprocess, "Popen", return_value=child), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(["--inputs", "synthetic.json", "--chromosome", "21", "--out", str(output)])
            self.assertNotEqual(code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["complete"])
            self.assertFalse(report["full_pipeline_measured"])
            self.assertIsNone(report["whole_pipeline_estimate_seconds"])


if __name__ == "__main__":
    unittest.main()
