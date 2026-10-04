"""Mask and quantitative gene-test regression checks, runnable with unittest."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch

from torchwgs.masks import (Annotation, FrequencyDomain, GeneConfig, GeneSet,
                            MaskDefinition, PreparedMask, beta_maf_weights,
                            build_gene_masks, load_annotations, load_mask_definitions,
                            load_setlist, orient_alt)


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
        with self.assertRaises(ValueError):
            GeneConfig(skato_rhos=[float("nan")])

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
        with patch("torchwgs.gene.test_prepared_gene", return_value=[]):
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
            torch.testing.assert_close(actual.vc_genotypes, expected.vc_genotypes, atol=0, rtol=0)
            self.assertEqual(actual.aaf, expected.aaf)
        matrix_artifacts = []
        with patch("torchwgs.gene.test_prepared_gene", return_value=[]):
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
        configuration = GeneConfig(sbat_max_subsets=0, sbat_qmc_samples=64, sbat_seed=27)
        fake = {name: y.new_tensor(.2) for name in ("SBAT", "SBAT_POS", "SBAT_NEG")}
        with patch("torchwgs.gene.stats.sbat_logp", return_value=fake) as backend:
            result = test_prepared_gene(GeneSet("G", "1", 100, ("M1", "M2")),
                                        masks, context, configuration)
        self.assertIn("ADD-BURDEN-SBAT", {row["TEST"] for row in result})
        backend.assert_called_once()
        self.assertEqual(backend.call_args.kwargs["max_subsets"], 0)
        self.assertEqual(backend.call_args.kwargs["qmc_samples"], 64)
        self.assertEqual(backend.call_args.kwargs["seed"], 27)
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
