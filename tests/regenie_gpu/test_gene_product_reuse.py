"""Independent QT product oracles and bounded same-gene reuse guards."""
from dataclasses import replace
from types import SimpleNamespace
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from torchwgs import gene as gene_module
from torchwgs import statistics as stats
from torchwgs.gene import (_GeneProductCache, _mask_product_key,
                           _product_cache_limit, _vc_score_covariance,
                           test_gene_based, test_prepared_gene)
from torchwgs.masks import (Annotation, GeneConfig, GeneMaskBuilder, GeneSet,
                            MaskDefinition, _vc_tensor_guard, build_gene_masks)
from torchwgs.single import create_test_context


class GeneProductReuseTests(unittest.TestCase):
    @staticmethod
    def fixture(storage="dense", **changes):
        n = 96
        x = torch.zeros((n, 5), dtype=torch.float64)
        x[:8, 0] = 1
        x[10:23, 1] = 1
        x[25:32, 2] = 2
        x[40, 3] = 1
        x[41, 4] = 1
        x[80:83, :] = torch.nan
        ids = tuple("synthetic_site_" + str(i) for i in range(5))
        annotations = [Annotation(identifier, "SyntheticSet", "A") for identifier in ids]
        definitions = [MaskDefinition(name, frozenset({"A"})) for name in ("First", "Second")]
        config = GeneConfig(aaf_bins=(.5,), vc_max_aaf=.5, collapse_mac=2,
                            include_singletons=False, include_domains=False,
                            vc_score_method="crossproduct", vc_storage=storage,
                            vc_score_block_size=2, variant_block_size=2,
                            genotype_orientation="alt", gene_p=False,
                            skato_rhos=(0., 1.), **changes)
        row = torch.arange(n, dtype=torch.float64)
        covariates = torch.stack((row/n, torch.cos(row*.13)), dim=1)
        context = create_test_context(torch.sin(row*.31)+row*.01,
                                      covariates=covariates, apply_rint=False,
                                      device="cpu", dtype="float64")
        masks = build_gene_masks(x, ids, annotations, definitions, config)
        gene = GeneSet("SyntheticSet", "1", 1, ids)
        return x, ids, annotations, definitions, config, context, masks, gene

    @staticmethod
    def light_kernel(score, covariance, *, weights, rhos, **unused):
        # Independent scalar rayleigh statistic makes weight changes observable;
        # existing full-statistics tests cover the actual SKAT/SKAT-O algorithms.
        weighted_score = score * weights
        weighted_covariance = covariance * weights[:, None] * weights[None, :]
        statistic = weighted_score.square().sum() / weighted_covariance.diag().sum()
        lp = stats.chi2_logsf(statistic)
        return {"kernel_valid": True, "rhos": rhos,
                "rho_log10ps": torch.stack([lp for _ in rhos]),
                "SKAT": lp, "SKATO": lp, "SKATO-ACAT": lp}

    def test_dense_sparse_cached_products_match_independent_numpy_projection(self):
        for storage in ("dense", "sparse"):
            with self.subTest(storage=storage):
                _, _, _, _, config, context, masks, _ = self.fixture(storage)
                cache = _GeneProductCache(2**20)
                with patch.object(gene_module, "_vc_score_covariance",
                                  wraps=_vc_score_covariance) as compute:
                    actual = []
                    for mask in masks:
                        key = (_mask_product_key(mask, context, "vc"),
                               config.vc_score_method, config.vc_score_block_size)
                        actual.append(cache.get_or_compute("vc", key,
                            lambda: gene_module._vc_score_covariance(mask.vc_genotypes, context,
                                method=config.vc_score_method, block_size=2)))
                self.assertEqual(compute.call_count, 1)
                self.assertEqual(cache.hits["vc"], 1)
                self.assertIs(actual[0][0], actual[1][0])
                dense = masks[0].vc_genotypes
                g = (dense.to_dense() if dense.is_sparse else dense).numpy()
                finite = np.isfinite(g)
                means = np.sum(np.where(finite, g, 0.), axis=0) / np.maximum(finite.sum(0), 1)
                g = np.where(finite, g, means)
                q, y = context.covariates_q_float64.numpy(), context.y_float64.numpy()
                residual = g - q @ (q.T @ g)
                expected_score, expected_covariance = residual.T @ y, residual.T @ residual
                np.testing.assert_allclose(actual[0][0].numpy(), expected_score, atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(actual[0][1].numpy(), expected_covariance, atol=2e-12, rtol=2e-12)

    def test_cached_rows_are_exact_and_each_weighted_kernel_is_still_computed(self):
        _, _, _, _, config, context, masks, gene = self.fixture()
        # Equal VC inputs with two deliberately different MAF weight functions.
        masks[1] = replace(masks[1], beta_b=7.)
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel):
            expected = test_prepared_gene(gene, masks, context, config)
        cache = _GeneProductCache(2**20)
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel) as kernels, \
             patch.object(gene_module, "_vc_score_covariance", wraps=_vc_score_covariance) as vc, \
             patch.object(gene_module, "_score_covariance", wraps=gene_module._score_covariance) as burden:
            actual = test_prepared_gene(gene, masks, context, config, _product_cache=cache)
        self.assertEqual(actual, expected)
        self.assertEqual(vc.call_count, 1)
        self.assertEqual(burden.call_count, 1)
        self.assertEqual(kernels.call_count, 2)
        self.assertFalse(torch.equal(kernels.call_args_list[0].kwargs["weights"],
                                     kernels.call_args_list[1].kwargs["weights"]))
        self.assertEqual(cache.hits, {"vc": 1, "burden": 1})
        self.assertEqual(cache.bytes, 0)
        self.assertLessEqual(cache.peak_bytes, cache.max_bytes)

    def test_residual_method_reuses_original_double_projection_exactly(self):
        _, _, _, _, config, context, masks, _ = self.fixture()
        cache = _GeneProductCache(2**20)
        products = []
        with patch.object(gene_module, "_score_covariance", wraps=gene_module._score_covariance) as project:
            for mask in masks:
                key = (_mask_product_key(mask, context, "vc"), "residual", 2)
                products.append(cache.get_or_compute("vc", key,
                    lambda: _vc_score_covariance(mask.vc_genotypes, context, method="residual")))
        self.assertEqual(project.call_count, 1)
        self.assertIs(products[0][1], products[1][1])
        g, q = masks[0].vc_genotypes.numpy(), context.covariates_q_float64.numpy()
        residual = g - q @ (q.T @ g)
        np.testing.assert_allclose(products[0][0].numpy(),
                                   residual.T @ context.y_float64.numpy(), atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(products[0][1].numpy(), residual.T @ residual,
                                   atol=2e-12, rtol=2e-12)

    def test_streaming_entry_point_reduces_actual_products_and_reserves_budget(self):
        x, ids, annotations, definitions, config, context, _, gene = self.fixture()
        variants = [SimpleNamespace(id=identifier, index=i, allele0="A", allele1="C")
                    for i, identifier in enumerate(ids)]
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel):
            expected = list(test_gene_based(x, context, annotations, [gene], definitions,
                                            replace(config, product_cache_bytes=0), variants=variants))
        observed_budgets = []
        original_builder = gene_module.GeneMaskBuilder
        def builder(*args):
            observed_budgets.append(args[-1].max_matrix_bytes)
            return original_builder(*args)
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel), \
             patch.object(gene_module, "GeneMaskBuilder", side_effect=builder), \
             patch.object(gene_module, "_vc_score_covariance", wraps=_vc_score_covariance) as vc, \
             patch.object(gene_module, "_score_covariance", wraps=gene_module._score_covariance) as burden:
            actual = list(test_gene_based(x, context, annotations, [gene], definitions,
                                          config, variants=variants))
        self.assertEqual(actual, expected)
        self.assertEqual((vc.call_count, burden.call_count), (1, 1))
        self.assertEqual(observed_budgets,
                         [config.max_matrix_bytes-_product_cache_limit(config)])

    def test_materialized_public_call_does_not_allocate_unreserved_cache(self):
        _, _, _, _, config, context, masks, gene = self.fixture()
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel), \
             patch.object(gene_module, "_vc_score_covariance", wraps=_vc_score_covariance) as compute:
            test_prepared_gene(gene, masks, context, config)
        self.assertEqual(compute.call_count, 2)

    def test_collapsed_members_and_vc_column_order_cannot_share_products(self):
        x, ids, annotations, _, config, context, _, _ = self.fixture()
        definitions = [MaskDefinition("Left", frozenset({"A"}), frozenset(ids[:4])),
                       MaskDefinition("Right", frozenset({"A"}), frozenset(ids[:3]+ids[4:]))]
        masks = build_gene_masks(x, ids, annotations, definitions, config)
        self.assertNotEqual(masks[0].vc_reuse_key, masks[1].vc_reuse_key)
        self.assertFalse(torch.equal(masks[0].vc_genotypes[:, -1], masks[1].vc_genotypes[:, -1]))
        first = masks[0]
        reordered = replace(first, vc_genotypes=first.vc_genotypes.flip(1),
                             vc_mafs=first.vc_mafs.flip(0))
        self.assertIsNone(_mask_product_key(reordered, context, "vc"))

    def test_builder_scope_and_external_masks_cannot_alias_by_variant_ids(self):
        _, _, _, _, _, context, masks, _ = self.fixture()
        _, _, _, _, _, _, other, _ = self.fixture()
        self.assertEqual(masks[0].variant_ids, other[0].variant_ids)
        self.assertNotEqual(masks[0].vc_reuse_key, other[0].vc_reuse_key)
        external = replace(masks[0], vc_reuse_key=None, vc_reuse_tensor_guard=None)
        self.assertIsNone(_mask_product_key(external, context, "vc"))

    def test_dense_sparse_and_burden_mutations_invalidate_bound_guard(self):
        for storage in ("dense", "sparse"):
            _, _, _, _, _, context, masks, _ = self.fixture(storage)
            mask = masks[0]
            self.assertIsNotNone(_mask_product_key(mask, context, "vc"))
            if mask.vc_genotypes.is_sparse:
                mask.vc_genotypes.values()[0] += 1
            else:
                mask.vc_genotypes[0, 0] += 1
            self.assertIsNone(_mask_product_key(mask, context, "vc"))
            mask.burden[0] += .5
            self.assertIsNone(_mask_product_key(mask, context, "burden"))
        _, _, _, _, _, context, masks, _ = self.fixture("sparse")
        masks[0].vc_genotypes.indices()[0, 0] += 1
        self.assertIsNone(_mask_product_key(masks[0], context, "vc"))

    def test_context_mutation_invalidates_old_cache_key(self):
        _, _, _, _, _, context, masks, _ = self.fixture()
        first = _mask_product_key(masks[0], context, "vc")
        context.y_float64.add_(.01)
        second = _mask_product_key(masks[1], context, "vc")
        self.assertNotEqual(first, second)
        context.covariates_q_float64[0, 0] += .01
        self.assertNotEqual(second, _mask_product_key(masks[1], context, "vc"))

    def test_lru_budget_exact_boundary_and_oversized_products(self):
        cache = _GeneProductCache(40)
        calls = []
        def make(value, size=5):
            calls.append(value)
            return (torch.full((size,), value, dtype=torch.float64),)
        cache.get_or_compute("vc", "one", lambda: make(1.))
        self.assertEqual(cache.bytes, 40)
        cache.get_or_compute("vc", "one", lambda: make(1.))
        cache.get_or_compute("vc", "two", lambda: make(2.))
        cache.get_or_compute("vc", "one", lambda: make(1.))
        cache.get_or_compute("vc", "large", lambda: make(3., 6))
        cache.get_or_compute("vc", "large", lambda: make(3., 6))
        self.assertEqual(calls, [1., 2., 1., 3., 3.])
        self.assertEqual(cache.peak_bytes, 40)
        self.assertEqual(cache.bytes, 40)
        self.assertEqual(cache.hits["vc"], 1)

    def test_cache_does_not_keep_genotypes_and_clear_releases_products(self):
        _, _, _, _, config, context, masks, _ = self.fixture()
        mask = masks[0]
        cache = _GeneProductCache(2**20)
        score, covariance = cache.get_or_compute("vc", _mask_product_key(mask, context, "vc"),
            lambda: _vc_score_covariance(mask.vc_genotypes, context, method=config.vc_score_method))
        genotype_ref, score_ref = weakref.ref(mask.vc_genotypes), weakref.ref(score)
        mask.vc_genotypes = None
        del score, covariance
        gc.collect()
        self.assertIsNone(genotype_ref())
        self.assertIsNotNone(score_ref())
        cache.clear()
        gc.collect()
        self.assertIsNone(score_ref())

    def test_inference_tensors_conservatively_bypass_reuse(self):
        with torch.inference_mode():
            tensor = torch.ones((4, 2), dtype=torch.float64)
        self.assertIsNone(_vc_tensor_guard(tensor))

    def test_products_are_released_when_mask_generator_fails(self):
        _, _, _, _, config, context, masks, gene = self.fixture()
        cache = _GeneProductCache(2**20)
        def failing_masks():
            yield masks[0]
            self.assertGreater(cache.bytes, 0)
            raise RuntimeError("synthetic iterator failure")
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel):
            with self.assertRaisesRegex(RuntimeError, "synthetic iterator failure"):
                test_prepared_gene(gene, failing_masks(), context, config, _product_cache=cache)
        self.assertEqual(cache.bytes, 0)
        self.assertFalse(cache._entries)

    def test_streamed_cached_masks_release_each_previous_genotype(self):
        x, ids, annotations, definitions, config, context, _, gene = self.fixture()
        builder = GeneMaskBuilder(annotations, definitions, x.shape[0], "cpu", torch.float64,
                                  replace(config, max_matrix_bytes=config.max_matrix_bytes-2**20))
        builder.update(ids, x)
        references = []
        def prepared():
            for mask in builder.finish_iter():
                gc.collect()
                if references:
                    self.assertIsNone(references[-1]())
                references.append(weakref.ref(mask.vc_genotypes))
                yield mask
                mask = None
        cache = _GeneProductCache(2**20)
        with patch.object(stats, "skato_logp", side_effect=self.light_kernel):
            test_prepared_gene(gene, prepared(), context, config, _product_cache=cache)
        self.assertEqual(cache.hits["vc"], 1)
        self.assertTrue(all(reference() is None for reference in references))

    def test_insufficient_cache_budget_computes_original_products_and_records_counts(self):
        _, _, _, _, config, context, masks, gene = self.fixture()
        cache = _GeneProductCache(1)
        with stats.diagnostics_scope() as diagnostics, \
             patch.object(stats, "skato_logp", side_effect=self.light_kernel), \
             patch.object(gene_module, "_vc_score_covariance", wraps=_vc_score_covariance) as compute:
            actual = test_prepared_gene(gene, masks, context, config, _product_cache=cache)
            self.assertEqual(diagnostics["gene_vc_product_cache_hits"], 0)
            self.assertEqual(diagnostics["gene_vc_product_cache_misses"], 2)
            self.assertEqual(diagnostics["gene_burden_product_cache_misses"], 2)
        self.assertTrue(actual)
        self.assertEqual(compute.call_count, 2)
        self.assertEqual(cache.peak_bytes, 0)

    def test_private_cache_never_crosses_separate_gene_calls(self):
        _, _, _, _, config, context, masks, gene = self.fixture()
        cache = _GeneProductCache(2**20)
        with stats.diagnostics_scope() as diagnostics, \
             patch.object(stats, "skato_logp", side_effect=self.light_kernel), \
             patch.object(gene_module, "_vc_score_covariance", wraps=_vc_score_covariance) as compute:
            for _ in range(2):
                test_prepared_gene(gene, masks, context, config, _product_cache=cache)
            self.assertEqual(diagnostics["gene_vc_product_cache_hits"], 2)
            self.assertEqual(diagnostics["gene_vc_product_cache_misses"], 2)
        self.assertEqual(compute.call_count, 2)

    def test_full_gene_summary_and_sbat_keep_exact_output_and_order(self):
        x, ids, annotations, definitions, config, context, _, gene = self.fixture()
        variants = [SimpleNamespace(id=identifier, index=i, allele0="A", allele1="C")
                    for i, identifier in enumerate(ids)]
        config = replace(config, gene_p=True, run_sbat=True)
        # Here use the actual SKAT/O, SBAT and four-component GENE_P algorithms.
        expected = list(test_gene_based(x, context, annotations, [gene], definitions,
                                       replace(config, product_cache_bytes=0), variants=variants))
        actual = list(test_gene_based(x, context, annotations, [gene], definitions,
                                     config, variants=variants))
        self.assertEqual(actual, expected)
        self.assertIn("GENE_P", [row["TEST"] for row in actual])
        self.assertIn("ADD-BURDEN-SBAT", [row["TEST"] for row in actual])


if __name__ == "__main__":
    unittest.main()
