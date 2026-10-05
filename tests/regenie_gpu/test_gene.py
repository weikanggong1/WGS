"""Mask and quantitative gene-test regression checks, runnable with unittest."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
import weakref
from unittest.mock import patch

import torch

from torchwgs.masks import (Annotation, FrequencyDomain, GeneConfig, GeneMaskBuilder, GeneSet,
                            MaskDefinition, PreparedMask, beta_maf_weights,
                            build_gene_masks, load_annotations, load_mask_definitions,
                            load_setlist, load_variant_whitelist, orient_alt)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Context:
    def __init__(self, y, scale=1., residual_scale=1.):
        self.y = y.to(DEVICE, torch.float64)
        self.y_scale, self.residual_scale = scale, residual_scale
        n = self.y.numel()
        self.covariates_q = torch.ones((n, 1), device=DEVICE, dtype=torch.float64) / n ** .5
        self.sample_ids = tuple((str(i), str(i)) for i in range(n))

    def residualize(self, matrix):
        x = matrix.to(DEVICE, torch.float64)
        valid = torch.isfinite(x) & (x >= 0)
        mean = torch.where(valid, x, 0.).sum(0) / valid.sum(0).clamp_min(1)
        x = torch.where(valid, x, mean)
        return x - self.covariates_q @ (self.covariates_q.T @ x)


class MaskTests(unittest.TestCase):
    def test_score_whitelist_candidate_filter_matches_complete_source_membership(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "score.extract"
            path.write_text("# original annotation score ids\nv0\nv1 .7\n\nv1 .7\nv2 .8\nv3\n")
            complete = load_variant_whitelist(path)
            candidates = {"v1", "v3", "not_in_source"}
            self.assertEqual(load_variant_whitelist(path, variant_ids=candidates),
                             complete.intersection(candidates))
            self.assertEqual(load_variant_whitelist(path, variant_ids=[]), frozenset())
            self.assertEqual(candidates, {"v1", "v3", "not_in_source"})

    def test_all_masks_share_regular_vc_storage_and_release_after_last_mask(self):
        genotype = torch.zeros((1000, 4), device=DEVICE, dtype=torch.float64)
        for column in range(4):
            genotype[column*12:(column+1)*12, column] = 1
        identifiers = [f"v{i}" for i in range(4)]
        annotations = [Annotation(identifier, "G", "A", "R") for identifier in identifiers]
        definitions = [MaskDefinition(f"Mask{i}", frozenset(["A"])) for i in range(10)]
        # Ten masks times domain/overall would exceed this limit if every VC
        # input were copied. One shared input plus one mask's working matrix fits.
        configuration = GeneConfig(include_singletons=False, max_matrix_bytes=110000)
        builder = GeneMaskBuilder(annotations, definitions, 1000, DEVICE,
                                  torch.float64, configuration)
        builder.update(identifiers, genotype)
        self.assertEqual(builder.retained_bytes, genotype.numel()*8)
        references = [state.vc_chunks[0].shared for state in builder.states]
        self.assertTrue(all(shared is references[0] for shared in references))
        masks = builder.finish_iter()
        count = 0
        for mask in masks:
            count += 1
            torch.testing.assert_close(mask.vc_genotypes, genotype, atol=0, rtol=0)
            self.assertEqual(mask.variant_ids, tuple(identifiers))
        self.assertEqual(count, 20)
        self.assertEqual(builder.retained_bytes, 0)
        self.assertIsNone(references[0].values)
        with self.assertRaises(RuntimeError):
            list(builder.finish_iter())

    def test_empty_and_low_mac_masks_release_shared_storage(self):
        genotype = torch.zeros((1000, 2), device=DEVICE, dtype=torch.float64)
        genotype[:12, 0] = 1
        genotype[12:24, 1] = 1
        definitions = [MaskDefinition("M1", frozenset(["A"])),
                       MaskDefinition("M2", frozenset(["B"]))]
        annotations = [Annotation("v1", "G", "A", "R"),
                       Annotation("v2", "G", "B", "UR")]
        builder = GeneMaskBuilder(annotations, definitions, 1000, DEVICE, torch.float64,
                                  GeneConfig(min_mac=30))
        builder.update(["v1", "v2"], genotype)
        self.assertEqual(list(builder.finish_iter()), [])
        self.assertEqual(builder.retained_bytes, 0)

    def test_json_configuration_roundtrip_builds_actual_masks(self):
        from torchwgs.config import WGSConfig
        original = WGSConfig.paper(apply_rint=False)
        loaded = WGSConfig.from_dict(json.loads(json.dumps(original.to_dict())))
        genotypes = torch.zeros((1000, 1), device=DEVICE, dtype=torch.float64)
        genotypes[0, 0] = 1
        result = build_gene_masks(genotypes, ["v"], [Annotation("v", "G", "A", "UR")],
                                  [MaskDefinition("Mask1", frozenset(["A"]))],
                                  loaded.gene_based)
        self.assertEqual({(mask.domain, mask.frequency) for mask in result},
                         {(None, "singleton"), (None, "0.01"),
                          ("UR", "singleton"), ("UR", "0.01")})
        self.assertFalse(loaded.gene_based.apply_rint)
        self.assertEqual(loaded.gene_based.skato_integral_backend, "adaptive_x")
        self.assertEqual(loaded.gene_based.skato_integral_epsabs, 1e-25)
        self.assertEqual(loaded.gene_based.skato_integral_epsrel, 2.**-13)
        self.assertEqual(loaded.gene_based.skato_integral_max_intervals, 1000)
        with self.assertRaises(ValueError):
            GeneConfig(skato_rhos=[float("nan")])
        for values in ({"skato_integral_backend": "unknown"}, {"skato_integral_epsabs": -1},
                       {"skato_integral_epsrel": 0}, {"skato_integral_max_intervals": 0}):
            with self.assertRaises(ValueError):
                GeneConfig(**values)

    def test_custom_beta_weights_use_gpu_beta_density(self):
        maf = torch.tensor([.001, .01, .5], device=DEVICE, dtype=torch.float64)
        torch.testing.assert_close(beta_maf_weights(maf, 2., 3.), 12*maf*(1-maf).square())
        with self.assertRaises(ValueError):
            GeneConfig(beta_a=0.)

    def test_text_resources_preserve_unknown_domain_labels(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "anno").write_text("1:2:C:T G C Missense\n1:3:G:A G UR Missense\n")
            (path / "sets").write_text("G 1 2 1:2:C:T,1:3:G:A\n")
            (path / "masks").write_text("Mask1 Missense,PTV\n")
            annotations = load_annotations(path / "anno")
            self.assertEqual([a.domain for a in annotations], ["C", "UR"])
            self.assertEqual(load_setlist(path / "sets")[0].variant_ids,
                             ("1:2:C:T", "1:3:G:A"))
            self.assertEqual(load_mask_definitions(path / "masks")[0].categories,
                             frozenset(["Missense", "PTV"]))
            self.assertEqual(load_mask_definitions(path / "masks")[0].category_order,
                             ("Missense", "PTV"))

    def test_max_before_missing_imputation_and_singleton_membership(self):
        g = torch.zeros((1000, 3), device=DEVICE, dtype=torch.float64)
        g[:12, 0] = 1
        g[12:24, 1] = 1
        g[24, 2] = 1
        g[25, :2] = float("nan")
        g[26, :] = float("nan")
        annotations = [Annotation(f"v{i}", "G", "Missense", "opaque") for i in range(3)]
        prepared = build_gene_masks(g, ["v0", "v1", "v2"], annotations,
                                    [MaskDefinition("M", frozenset(["Missense"]))])
        all_rare = next(m for m in prepared if m.domain is None and m.frequency == "0.01")
        expected = torch.zeros(1000, device=DEVICE, dtype=torch.float64)
        expected[:25] = 1
        expected[26] = 25 / 999
        # v2 is observed as zero in row 25, so max is observed zero, not imputed.
        torch.testing.assert_close(all_rare.burden, expected)
        self.assertTrue(bool(torch.isnan(all_rare.raw_burden[26])))
        self.assertEqual(float(all_rare.raw_burden[25]), 0.)
        self.assertEqual(all_rare.vc_genotypes.shape[1], 3)  # two regular + collapsed singleton
        singleton = next(m for m in prepared if m.domain is None and m.frequency == "singleton")
        self.assertEqual(singleton.variant_ids, ("v2",))
        self.assertEqual(singleton.mac, 1)

    def test_alt_burden_and_minor_allele_collapse_are_different(self):
        g = torch.zeros((100, 2), device=DEVICE, dtype=torch.float64)
        g[:, 0] = 2
        g[0, 0] = 1
        g[1, 1] = 1
        prepared = build_gene_masks(g, ["v0", "v1"],
                    [Annotation("v0", "G", "A"), Annotation("v1", "G", "A")],
                    [MaskDefinition("M", frozenset(["A"]))],
                    GeneConfig(aaf_bins=(1.,), vc_max_aaf=1.))
        mask = next(m for m in prepared if m.frequency == "1")
        torch.testing.assert_close(mask.burden, g[:, 0])
        self.assertEqual(mask.vc_genotypes.shape[1], 2)
        expected_minor = torch.zeros((100, 2), device=DEVICE, dtype=torch.float64)
        expected_minor[0, 0] = 1
        expected_minor[1, 1] = 1
        torch.testing.assert_close(mask.vc_genotypes, expected_minor)
        torch.testing.assert_close(mask.vc_mafs, torch.tensor([.005, .005], device=DEVICE, dtype=torch.float64))
        self.assertAlmostEqual(float(mask.vc_weights[0]), 25 * .995 ** 24, places=11)
        singleton = next(m for m in prepared if m.frequency == "singleton")
        self.assertEqual(set(singleton.variant_ids), {"v1"})

    def test_score_whitelist_and_explicit_domain_boundaries(self):
        g = torch.zeros((1000, 2), device=DEVICE, dtype=torch.float64)
        g[0, 0], g[:12, 1] = 1, 1
        annotations = [Annotation("v0", "G", "A"), Annotation("v1", "G", "A")]
        config = GeneConfig(domain_mapping={"custom_rare": FrequencyDomain(.001, .01, False, True)})
        masks = [MaskDefinition("M", frozenset(["A"]), frozenset(["v1"]), "user-score")]
        prepared = build_gene_masks(g, ["v0", "v1"], annotations, masks, config)
        self.assertTrue(prepared)
        self.assertTrue(all(m.variant_ids == ("v1",) for m in prepared))
        self.assertTrue(any(m.domain == "custom_rare" for m in prepared))
        self.assertTrue(all(m.score == "user-score" for m in prepared))

    def test_orientation_checks_bim_allele_against_cpra_alt(self):
        variants = [SimpleNamespace(id="1:1:A:G", allele1="G", allele0="A"),
                    SimpleNamespace(id="1:2:C:T", allele1="C", allele0="T")]
        g = torch.tensor([[1., 2.], [0., float("nan")]], device=DEVICE)
        actual = orient_alt(g, variants, "variant_id")
        torch.testing.assert_close(actual, torch.tensor([[1., 0.], [0., float("nan")]], device=DEVICE),
                                   equal_nan=True)


class GeneTests(unittest.TestCase):
    def test_failed_kernel_retains_acatv_and_uses_separate_joint_mask_counts(self):
        from torchwgs.gene import test_prepared_gene
        from torchwgs.statistics import acat_logp
        n = 1024
        y = torch.sin(torch.arange(n, device=DEVICE, dtype=torch.float64)*.37)
        y -= y.mean()
        y *= ((n-1)/y.square().sum()).sqrt()
        context = Context(y)
        first = torch.zeros(n, device=DEVICE, dtype=torch.float64)
        second = torch.zeros_like(first)
        first[:16], second[32:48] = 1., 1.
        duplicate = first[:, None].expand(n, 3).clone()
        masks = [PreparedMask("failed", "failed", None, "0.01", .01,
                    ("a", "b", "c"), first, .0078125, 16., n, duplicate,
                    first.new_full((3,), .0078125), beta_a=1., beta_b=1.),
                 PreparedMask("valid", "valid", None, "0.01", .01,
                    ("d",), second, .0078125, 16., n, second[:, None],
                    first.new_tensor([.0078125]), beta_a=1., beta_b=1.)]
        configuration = GeneConfig(run_sbat=False)
        gene = GeneSet("G", "21", 1, ("a", "b", "c", "d"))
        for selected, expected_components in ((masks[:1], 2), (masks, 3)):
            rows = test_prepared_gene(gene, selected, context, configuration)
            failed_tests = {row["TEST"] for row in rows
                            if row["ID"] == "G.failed.0.01"}
            self.assertEqual(failed_tests, {"ADD", "ADD-ACATV"})
            joint = {row["TEST"]: row for row in rows if row["ID"] == "G"}
            self.assertEqual(joint["ADD-ACATV-ACAT"]["EXTRA"], f"DF={len(selected)}")
            if len(selected) == 1:
                self.assertNotIn("ADD-SKATO-ACAT", joint)
            else:
                self.assertEqual(joint["ADD-SKATO-ACAT"]["EXTRA"], "DF=1")
            self.assertTrue(joint["GENE_P"]["EXTRA"].startswith(f"DF={expected_components}"))
            component_tests = ["ADD-BURDEN-ACAT", "ADD-ACATV-ACAT"]
            if len(selected) > 1:
                component_tests.append("ADD-SKATO-ACAT")
            expected = acat_logp(first.new_tensor([joint[test]["LOG10P"]
                                                  for test in component_tests]))
            self.assertAlmostEqual(joint["GENE_P"]["LOG10P"], float(expected), places=12)

    def test_packed_gene_reader_preserves_all_results_masks_missingness_and_sample_order(self):
        from torchwgs.gene import test_gene_based
        from torchwgs.io import BedReader, Variant, write_bed
        with TemporaryDirectory() as directory:
            identifiers = [(str(i), str(i)) for i in range(1004)]
            variants = [Variant(i, "21", f"v{i}", i+1, "A", "G") for i in range(4)]
            genotype = torch.zeros((1004, 4), dtype=torch.float64)
            genotype[5:18, 0] = 1
            genotype[20:36, 1] = 1
            genotype[40, 2] = 1
            genotype[45:58, 3] = 1
            genotype[800:803, :] = torch.nan
            prefix = Path(directory) / "source"
            write_bed(prefix, genotype, variants, identifiers)
            context = Context(torch.sin(torch.arange(1000, dtype=torch.float64)*.37))
            context.sample_ids = tuple(identifiers[1000:0:-1])
            annotations = [Annotation(f"v{i}", "G", "A", "UR" if i == 2 else "R")
                           for i in range(4)]
            sets = [GeneSet("G", "21", 1, ("v3", "v1", "v0", "v2"))]
            definitions = [MaskDefinition("Mask1", frozenset(["A"]))]
            cpu_artifacts, packed_artifacts = [], []
            expected = list(test_gene_based(BedReader(prefix), context, annotations, sets, definitions,
                GeneConfig(variant_block_size=2), artifact_callback=cpu_artifacts.append))
            reader = BedReader(prefix)
            with patch.object(reader, "read_variants", side_effect=AssertionError("expanded CPU decoding called")):
                observed = list(test_gene_based(reader, context, annotations, sets, definitions,
                    GeneConfig(variant_block_size=2, genotype_reader="cuda_packed"),
                    artifact_callback=packed_artifacts.append))
            self.assertEqual(observed, expected)
            self.assertEqual(len(cpu_artifacts[0].masks), len(packed_artifacts[0].masks))
            for actual, reference in zip(packed_artifacts[0].masks, cpu_artifacts[0].masks):
                self.assertEqual((actual.name, actual.frequency, actual.variant_ids, actual.n_observed),
                                 (reference.name, reference.frequency, reference.variant_ids, reference.n_observed))
                torch.testing.assert_close(actual.raw_burden, reference.raw_burden, atol=0, rtol=0,
                                           equal_nan=True)
        with self.assertRaises(ValueError):
            GeneConfig(genotype_reader="invalid")

    def test_burden_sample_count_excludes_missing_while_vc_uses_active_samples(self):
        from torchwgs.gene import test_prepared_gene
        genotype = torch.zeros(1000, device=DEVICE, dtype=torch.float64)
        genotype[:15] = 1
        genotype[-3:] = torch.nan
        mask = PreparedMask("M", "M", None, "0.01", .01, ("v",), genotype,
                            15/(2*997), 15., 997, genotype[:, None],
                            torch.tensor([15/(2*997)], device=DEVICE))
        context = Context(torch.sin(torch.arange(1000, dtype=torch.float64)*.37))
        rows = test_prepared_gene(GeneSet("G", "1", 1, ("v",)), [mask], context)
        self.assertEqual([row["N"] for row in rows if row["TEST"] == "ADD"], [997])
        self.assertTrue(all(row["N"] == 1000 for row in rows if row["TEST"] != "ADD"))

    def test_crossproduct_dense_and_sparse_scores_match_projected_formula_with_missing(self):
        from torchwgs.gene import _score_covariance, _vc_score_covariance
        from torchwgs.single import create_test_context
        rows = torch.arange(1000, dtype=torch.float64)
        covariates = torch.stack([torch.sin(rows*.02), rows/1000], 1)
        y = torch.sin(rows*.37) + rows*.001
        context = create_test_context(y, covariates=covariates, apply_rint=False,
                                      device="cpu", dtype="float32")
        # LOCO subtraction can leave a covariate component in y. The source's
        # U correction must be retained even though genotypes use raw products.
        context.y_float64 += context.covariates_q_float64[:, 1]*.13
        genotypes = torch.zeros((1000, 5), dtype=torch.float64)
        for column in range(4):
            genotypes[column*13:(column+1)*13, column] = 1
            genotypes[800+column, column] = torch.nan
        genotypes[:, 4] = torch.nan
        score, covariance, _ = _score_covariance(genotypes, context)
        for storage in (genotypes, genotypes.to_sparse_coo()):
            actual = _vc_score_covariance(storage, context, method="crossproduct", block_size=2)
            torch.testing.assert_close(actual[0], score, atol=2e-12, rtol=2e-12)
            torch.testing.assert_close(actual[1], covariance, atol=2e-11, rtol=2e-12)
        with self.assertRaises(ValueError):
            GeneConfig(vc_storage="sparse")

    def test_sparse_mask_storage_preserves_imputation_and_full_gene_results(self):
        from torchwgs.gene import test_prepared_gene
        genotype = torch.zeros((1000, 3), dtype=torch.float64)
        genotype[:12, 0] = 1
        genotype[12:28, 1] = 1
        genotype[28, 2] = 1
        genotype[900:903, :] = torch.nan
        identifiers = ["v0", "v1", "v2"]
        annotations = [Annotation("v0", "G", "A", "R"), Annotation("v1", "G", "A", "R"),
                       Annotation("v2", "G", "A", "UR")]
        definitions = [MaskDefinition("M", frozenset(["A"]))]
        dense_config = GeneConfig(variant_block_size=1)
        sparse_config = GeneConfig(variant_block_size=1, vc_storage="sparse",
                                   vc_score_method="crossproduct", vc_score_block_size=1)
        dense = build_gene_masks(genotype, identifiers, annotations, definitions, dense_config)
        sparse = build_gene_masks(genotype, identifiers, annotations, definitions, sparse_config)
        for actual, expected in zip(sparse, dense):
            self.assertEqual(actual.variant_ids, expected.variant_ids)
            torch.testing.assert_close(actual.burden, expected.burden, atol=0, rtol=0)
            if expected.vc_genotypes is not None:
                self.assertTrue(actual.vc_genotypes.is_sparse)
                torch.testing.assert_close(actual.vc_genotypes.to_dense(), expected.vc_genotypes,
                                           atol=0, rtol=0)
                torch.testing.assert_close(actual.vc_mafs, expected.vc_mafs, atol=0, rtol=0)
        y = torch.sin(torch.arange(1000, device=DEVICE, dtype=torch.float64)*.37)
        y -= y.mean()
        y *= (999/y.square().sum()).sqrt()
        gene = GeneSet("G", "1", 100, tuple(identifiers))
        original = test_prepared_gene(gene, dense, Context(y), dense_config)
        actual = test_prepared_gene(gene, sparse, Context(y), sparse_config)
        self.assertEqual([(row["ID"], row["TEST"], row["EXTRA"]) for row in actual],
                         [(row["ID"], row["TEST"], row["EXTRA"]) for row in original])
        for row, expected in zip(actual, original):
            self.assertAlmostEqual(row["LOG10P"], expected["LOG10P"], places=9)

    def test_streamed_gene_tests_release_earlier_mask_vc_matrices(self):
        from torchwgs.gene import test_prepared_gene
        y = torch.sin(torch.arange(1000, device=DEVICE, dtype=torch.float64)*.37)
        y -= y.mean()
        y *= (999/y.square().sum()).sqrt()
        context = Context(y)
        references = []
        def prepared():
            for index in range(3):
                if index == 2:
                    self.assertIsNone(references[0]())
                vc = torch.zeros((1000, 1), device=DEVICE, dtype=torch.float64)
                vc[index*12:(index+1)*12] = 1
                references.append(weakref.ref(vc))
                yield PreparedMask(f"M{index}", f"M{index}", None, "0.01", .01,
                                   (f"v{index}",), vc[:, 0].clone(), .006, 12., 1000,
                                   vc, vc.new_tensor([.006]))
        rows = test_prepared_gene(GeneSet("G", "1", 1, ()), prepared(), context)
        self.assertEqual(len([row for row in rows if row["TEST"] == "ADD-SKAT"]), 3)
        self.assertIn("GENE_P", {row["TEST"] for row in rows})
        self.assertTrue(all(reference() is None for reference in references))

    def test_global_whitelist_filters_io_and_preserves_bim_column_order(self):
        from torchwgs.gene import test_gene_based
        y = torch.sin(torch.arange(1000, dtype=torch.float64)*.37)
        ids = tuple((str(i), str(i)) for i in range(1000))
        context = SimpleNamespace(y=y, sample_ids=ids)
        variants = [SimpleNamespace(id=name, index=i, chrom="1", position=i+1,
                                    allele1="A", allele0="G")
                    for i, name in enumerate(("v2", "v1", "v0"))]
        raw = torch.zeros((1000, 3), dtype=torch.float64)
        raw[12:25, 0] = 1
        raw[:, 1] = 2
        raw[:12, 2] = 1
        class Reader:
            sample_ids, n_samples = ids, len(ids)
            def __init__(self):
                self.requests, self.columns = [], []
            def find_variants(self, needed):
                self.requests.append(set(needed))
                return {v.id: v for v in variants if v.id in needed}
            def read_variants(self, columns):
                self.columns.append(tuple(columns))
                return raw[:, columns]
        reader = Reader()
        annotations = [Annotation(name, "G", "A", "R")
                       for name in ("v0", "v1", "v2", "missing")]
        sets = [GeneSet("G", "1", 1, ("v0", "v1", "missing", "v2"))]
        definitions = [MaskDefinition("M", frozenset(["A"]))]
        config = GeneConfig(extract_variants={"v0", "v2", "missing"}, variant_block_size=1)
        artifacts = []
        materialized = []
        def consume_masks(gene, prepared, *args, **kwargs):
            materialized.append(list(prepared))
            return []
        with patch("torchwgs.gene.test_prepared_gene", side_effect=consume_masks):
            list(test_gene_based(reader, context, annotations, sets, definitions,
                                 config, artifact_callback=artifacts.append))
        self.assertEqual(reader.requests, [{"v0", "v2", "missing"}])
        self.assertEqual(reader.columns, [(0,), (2,)])
        # Compare to the original semantic path: constructing masks from every
        # input column and filtering the whitelist inside the mask builder.
        baseline = build_gene_masks(raw, [v.id for v in variants], annotations,
                                    definitions, config)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(artifacts[0].masks), 2)
        self.assertEqual(len(baseline), len(artifacts[0].masks))
        for actual, expected in zip(artifacts[0].masks, baseline):
            self.assertEqual(actual.variant_ids, ("v2", "v0"))
            self.assertEqual(actual.variant_ids, expected.variant_ids)
            torch.testing.assert_close(actual.burden, expected.burden, atol=0, rtol=0)
            self.assertIsNone(actual.vc_genotypes)
            self.assertEqual(actual.aaf, expected.aaf)
        for actual, expected in zip(materialized[0], baseline):
            torch.testing.assert_close(actual.vc_genotypes, expected.vc_genotypes, atol=0, rtol=0)
        matrix_artifacts = []
        with patch("torchwgs.gene.test_prepared_gene", side_effect=consume_masks):
            list(test_gene_based(raw, context, annotations, sets, definitions, config,
                                 variants=variants, artifact_callback=matrix_artifacts.append))
        self.assertEqual([m.variant_ids for m in matrix_artifacts[0].masks],
                         [m.variant_ids for m in baseline])

    def test_sbat_configuration_is_forwarded_to_statistical_backend(self):
        from torchwgs.gene import test_prepared_gene
        y = torch.sin(torch.arange(100, device=DEVICE, dtype=torch.float64)*.37)
        y -= y.mean()
        y *= (99 / y.square().sum()).sqrt()
        context = Context(y)
        first = torch.zeros_like(y)
        second = torch.zeros_like(y)
        first[:12] = 1
        second[12:28] = 1
        masks = [PreparedMask(name, name, None, "0.01", .01, (name,), raw,
                              float(raw.mean()/2), float(raw.sum()), len(y), None, None)
                 for name, raw in (("M1", first), ("M2", second))]
        configuration = GeneConfig(sbat_max_subsets=0, sbat_qmc_samples=64, sbat_seed=27,
                                   sbat_subset_sampling="with_replacement")
        fake = {name: y.new_tensor(.2) for name in ("SBAT", "SBAT_POS", "SBAT_NEG")}
        with patch("torchwgs.gene.stats.sbat_logp", return_value=fake) as backend:
            result = test_prepared_gene(GeneSet("G", "1", 100, ("M1", "M2")),
                                        masks, context, configuration)
        self.assertIn("ADD-BURDEN-SBAT", {row["TEST"] for row in result})
        backend.assert_called_once()
        self.assertEqual(backend.call_args.kwargs["max_subsets"], 0)
        self.assertEqual(backend.call_args.kwargs["qmc_samples"], 64)
        self.assertEqual(backend.call_args.kwargs["seed"], 27)
        self.assertEqual(backend.call_args.kwargs["subset_sampling"], "with_replacement")
        with self.assertRaises(ValueError):
            GeneConfig(sbat_qmc_samples=0)
        with self.assertRaises(ValueError):
            GeneConfig(sbat_max_subsets=-1)

    def test_constant_burden_is_ignored_after_covariate_projection(self):
        from torchwgs.gene import test_prepared_gene
        genotype = torch.ones(1000, device=DEVICE, dtype=torch.float64)
        y = torch.sin(torch.arange(1000, device=DEVICE, dtype=torch.float64))
        mask = PreparedMask("M", "M", None, "0.01", .01, ("v",), genotype,
                            .5, 1000., 1000, None, None)
        self.assertEqual(test_prepared_gene(GeneSet("G", "1", 100, ("v",)), [mask], Context(y)), [])

    def test_single_site_zero_score_keeps_native_zero_logp(self):
        from torchwgs.gene import test_prepared_gene
        genotype = torch.zeros(1000, device=DEVICE, dtype=torch.float64)
        genotype[:2] = 1
        y = torch.ones_like(genotype)
        y[::2] = -1
        mask = PreparedMask("M", "M", None, "0.01", .01, ("v",), genotype,
                            .001, 2., 1000, genotype[:, None],
                            torch.tensor([.001], device=DEVICE, dtype=torch.float64))
        rows = test_prepared_gene(GeneSet("G", "1", 100, ("v",)), [mask], Context(y))
        for row in rows:
            if row["TEST"] in {"ADD", "ADD-ACATO", "ADD-ACATV", "ADD-SKAT", "ADD-SKATO", "ADD-SKATO-ACAT"}:
                self.assertAlmostEqual(row["LOG10P"], 0., places=12)
        overall = next(row for row in rows if row["TEST"] == "GENE_P")
        self.assertNotIn("STRONGEST_MASK", overall["EXTRA"])

    def test_pivoted_qr_retains_independent_original_columns(self):
        from torchwgs.gene import _independent_columns
        matrix = torch.tensor([[3., 0., 3., 0.], [0., 1., 0., 1.],
                               [-3., 0., -3., 0.], [0., -1., 0., -1.]],
                              device=DEVICE, dtype=torch.float64)
        selected = _independent_columns(matrix, 1e-7)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], 0)
        basis = matrix[:, selected]
        coefficients = torch.linalg.solve(basis.T @ basis, basis.T @ matrix)
        torch.testing.assert_close(basis @ coefficients, matrix)

    def test_qr_ties_use_original_column_order_when_requested(self):
        from torchwgs.gene import _independent_columns
        # Equal signed-column norms can differ by floating-point reduction noise.
        # The optional tolerance fixes ties without changing the selected span.
        matrix = torch.tensor([[1., 0., 1.], [0., 1.+1e-14, 0.],
                               [-1., 0., -1.], [0., -1.-1e-14, 0.]],
                              device=DEVICE, dtype=torch.float64)
        self.assertEqual(_independent_columns(matrix, 1e-7, 1e-12), [0, 1])
        self.assertEqual(_independent_columns(matrix, 1e-7, 0.), [1, 0])

    def test_float32_context_keeps_double_gene_projection(self):
        from torchwgs.single import create_test_context
        from torchwgs.gene import _score_covariance
        y = torch.sin(torch.arange(1000, device=DEVICE, dtype=torch.float64)*.37)
        genotypes = torch.zeros((1000, 2), device=DEVICE, dtype=torch.float64)
        genotypes[:17, 0] = 1
        genotypes[17:33, 1] = 1
        context32 = create_test_context(y, apply_rint=False, device=str(DEVICE), dtype="float32")
        context64 = create_test_context(y, apply_rint=False, device=str(DEVICE), dtype="float64")
        actual = _score_covariance(genotypes, context32)
        expected = _score_covariance(genotypes, context64)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)

    def test_burden_score_matches_closed_form_and_original_columns(self):
        from torchwgs.gene import test_prepared_gene
        raw = torch.tensor([0., 1., 0., 2., 0., 1.], device=DEVICE, dtype=torch.float64)
        y = torch.tensor([-1., 1., -1., 2., -1., 0.], device=DEVICE, dtype=torch.float64)
        y -= y.mean()
        y *= ((len(y) - 1) / y.square().sum()).sqrt()
        context = Context(y, 2., 3.)
        mask = PreparedMask("M", "M", None, "0.01", .01, ("v",), raw,
                            1 / 3, 4., 6, None, None)
        rows = test_prepared_gene(GeneSet("G", "1", 100, ("v",)), [mask], context,
                                 GeneConfig(gene_p=False))
        self.assertEqual(len(rows), 1)
        centered = raw - raw.mean()
        u, v = centered @ y, centered.square().sum()
        self.assertAlmostEqual(rows[0]["BETA"], float(6 * u / v), places=12)
        self.assertAlmostEqual(rows[0]["SE"], float(6 / v.sqrt()), places=12)
        self.assertAlmostEqual(rows[0]["CHISQ"], float(u.square() / v), places=12)
        self.assertEqual(rows[0]["ID"], "G.M.0.01")
        self.assertEqual(rows[0]["ALLELE1"], "M.0.01")
        self.assertEqual(rows[0]["EXTRA"], "DF=1")
        self.assertEqual(set(rows[0]), set("CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE CHISQ LOG10P EXTRA".split()))

    def test_full_gene_tests_and_variant_order_invariance(self):
        from torchwgs.gene import test_gene_based
        g = torch.zeros((1000, 3), device=DEVICE, dtype=torch.float64)
        g[:12, 0] = 1
        g[12:28, 1] = 1
        g[28, 2] = 1
        # The deterministic phenotype has no dependency on RNG or variant order.
        y = torch.sin(torch.arange(1000, device=DEVICE, dtype=torch.float64) * .37)
        y[:12] += .3
        y -= y.mean()
        y *= (999 / y.square().sum()).sqrt()
        context = Context(y)
        variants = [SimpleNamespace(id=f"v{i}", index=i, chrom="1", position=100+i,
                                    allele1="T", allele0="C") for i in range(3)]
        annotations = [Annotation("v0", "G", "A", "R"),
                       Annotation("v1", "G", "A", "R"),
                       Annotation("v2", "G", "A", "UR")]
        sets = [GeneSet("G", "1", 100, ("v0", "v1", "v2"))]
        masks = [MaskDefinition("Mask1", frozenset(["A"]))]
        config = GeneConfig(genotype_orientation="alt")
        actual = list(test_gene_based(g, context, annotations, sets, masks, config,
                                      variants=variants))
        expected_tests = {"ADD", "ADD-SKAT", "ADD-SKATO", "ADD-SKATO-ACAT", "ADD-ACATO",
                          "ADD-ACATV", "ADD-ACATV-ACAT", "ADD-BURDEN-ACAT",
                          "ADD-BURDEN-SBAT", "ADD-BURDEN-SBAT_POS", "ADD-BURDEN-SBAT_NEG", "GENE_P"}
        self.assertEqual({r["TEST"] for r in actual}, expected_tests)
        self.assertTrue(all(r["LOG10P"] is not None and r["LOG10P"] >= 0 for r in actual))
        self.assertTrue(any(r["ID"] == "G.R.Mask1.0.01" for r in actual))
        self.assertEqual(next(r for r in actual if r["TEST"] == "GENE_P")["EXTRA"].split(";")[0], "DF=4")
        reordered = list(test_gene_based(g[:, [2, 0, 1]], context, annotations, sets, masks, config,
                                        variants=[variants[i] for i in [2, 0, 1]]))
        values = {(r["ID"], r["TEST"]): r["LOG10P"] for r in actual}
        for row in reordered:
            self.assertAlmostEqual(row["LOG10P"], values[(row["ID"], row["TEST"])], places=10)


if __name__ == "__main__":
    unittest.main()
