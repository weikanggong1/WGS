"""Exact mask membership and storage checks for batched metadata transfers."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from torchwgs import masks as mask_module
from torchwgs.masks import (Annotation, FrequencyDomain, GeneConfig,
                            GeneMaskBuilder, MaskDefinition, build_gene_masks)


def _legacy_variant_metadata(aaf, maf, mac, ac, counts, singletons, *, max_count):
    return [value.detach().cpu().tolist()
            for value in (aaf, maf, mac, ac, counts, singletons)]


def _legacy_frequency(ac, n):
    return [float((ac / (2 * max(n, 1))).item()),
            float(torch.minimum(ac, 2 * n - ac).item())]


class _MaskMetadataChecks:
    device = "cpu"

    def _build(self, genotypes, identifiers, annotations, definitions, config, *, legacy=False):
        with ExitStack() as stack:
            if legacy:
                stack.enter_context(patch.object(mask_module, "_copy_mask_variant_metadata",
                                                _legacy_variant_metadata))
                stack.enter_context(patch.object(mask_module, "_copy_mask_frequency",
                                                _legacy_frequency))
            return build_gene_masks(genotypes, identifiers, annotations, definitions, config)

    def _assert_masks_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for new, old in zip(actual, expected):
            for field in ("name", "base_name", "domain", "frequency", "aaf_upper",
                          "variant_ids", "aaf", "mac", "n_observed", "score",
                          "beta_a", "beta_b"):
                self.assertEqual(getattr(new, field), getattr(old, field), field)
            for field in ("burden", "raw_burden", "vc_genotypes", "vc_mafs",
                          "vc_weights", "acat_weights"):
                new_value, old_value = getattr(new, field), getattr(old, field)
                if old_value is None:
                    self.assertIsNone(new_value, field)
                    continue
                self.assertEqual(new_value.device, old_value.device, field)
                self.assertEqual(new_value.dtype, old_value.dtype, field)
                self.assertEqual(new_value.layout, old_value.layout, field)
                if new_value.is_sparse:
                    torch.testing.assert_close(new_value.indices(), old_value.indices(),
                                               rtol=0, atol=0)
                    new_value, old_value = new_value.values(), old_value.values()
                torch.testing.assert_close(new_value, old_value, rtol=0, atol=0,
                                           equal_nan=True, msg=field)

    def test_variant_metadata_copies_one_vector_and_preserves_scalar_types(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([64, 63, 0], device=self.device, dtype=torch.int64)
                ac = torch.tensor([2., 1.001, 0.], device=self.device, dtype=dtype)
                aaf = ac / (2 * counts.clamp_min(1))
                maf = torch.minimum(aaf, 1 - aaf)
                mac = torch.minimum(ac, 2 * counts - ac)
                singleton = (ac + .5).floor() == 1
                expected = _legacy_variant_metadata(aaf, maf, mac, ac, counts,
                                                    singleton, max_count=64)
                transfers = []
                original_cpu = torch.Tensor.cpu

                def tracked_cpu(tensor, *args, **kwargs):
                    transfers.append((tuple(tensor.shape), tensor.dtype))
                    return original_cpu(tensor, *args, **kwargs)

                with patch.object(torch.Tensor, "cpu", tracked_cpu):
                    actual = mask_module._copy_mask_variant_metadata(
                        aaf, maf, mac, ac, counts, singleton, max_count=64)
                self.assertEqual(actual, expected)
                self.assertEqual(transfers, [((6, 3), torch.float64)])
                self.assertTrue(all(type(value) is int for value in actual[4]))
                self.assertTrue(all(type(value) is bool for value in actual[5]))
                self.assertTrue(all(type(value) is float for row in actual[:4] for value in row))

    def test_frequency_copies_two_scalars_once_without_changing_division(self):
        for dtype in (torch.float32, torch.float64):
            for n, value in ((0, 0.), (63, 1.001), (999, 37.123456789), (64, 127.)):
                with self.subTest(dtype=dtype, n=n):
                    ac = torch.tensor(value, device=self.device, dtype=dtype)
                    expected = _legacy_frequency(ac, n)
                    transfers = []
                    original_cpu = torch.Tensor.cpu

                    def tracked_cpu(tensor, *args, **kwargs):
                        transfers.append(tuple(tensor.shape))
                        return original_cpu(tensor, *args, **kwargs)

                    with patch.object(torch.Tensor, "cpu", tracked_cpu):
                        actual = mask_module._copy_mask_frequency(ac, n)
                    self.assertEqual(actual, expected)
                    self.assertEqual(transfers, [(2,)])

    def test_huge_integer_count_keeps_int64_without_float64_rounding(self):
        counts = torch.tensor([2**53 + 1], device=self.device, dtype=torch.int64)
        value = torch.tensor([.25], device=self.device, dtype=torch.float64)
        singleton = torch.tensor([True], device=self.device)
        actual = mask_module._copy_mask_variant_metadata(
            value, value, value, value, counts, singleton, max_count=2**53 + 1)
        self.assertEqual(actual[4], [2**53 + 1])
        self.assertIs(type(actual[4][0]), int)
        self.assertIs(type(actual[5][0]), bool)

    def test_empty_block_has_no_members_or_retained_storage(self):
        empty = torch.empty(0, device=self.device)
        actual = mask_module._copy_mask_variant_metadata(
            empty, empty, empty, empty,
            empty.to(torch.int64), empty.to(torch.bool), max_count=64)
        self.assertEqual(actual, [[], [], [], [], [], []])
        builder = GeneMaskBuilder([], [MaskDefinition("M", frozenset(["A"]))],
                                  64, self.device, torch.float32)
        builder.update([], torch.empty((64, 0), device=self.device))
        self.assertEqual(builder.finish(), [])
        self.assertEqual(builder.retained_bytes, 0)

    def test_aaf_mac_domains_whitelists_and_collapse_match_legacy_exactly(self):
        identifiers = ["low", "at", "over", "half", "subhalf", "collapse_at",
                       "collapse_over", "missing", "forbidden", "other_mask", "invalid"]
        for dtype in (torch.float32, torch.float64):
            for storage in ("dense", "sparse"):
                with self.subTest(dtype=dtype, storage=storage):
                    g = torch.zeros((64, len(identifiers)), device=self.device, dtype=dtype)
                    g[0, 0] = 1
                    g[1:3, 1] = 1
                    g[3, 2], g[4, 2] = 2., .001
                    g[5, 3], g[6, 4] = .5, .49
                    g[7:17, 5] = 1
                    g[17:27, 6], g[27, 6] = 1., .001
                    g[28, 7] = 1
                    g[29:33, 7] = torch.tensor([float("nan"), -1., float("inf"), 3.],
                                                device=self.device, dtype=dtype)
                    g[34, 8], g[35, 9] = 1., 1.
                    g[:, 10] = float("nan")
                    annotations = [Annotation(v, "FixtureGene", "A", "R") for v in identifiers]
                    annotations[-2] = Annotation("other_mask", "FixtureGene", "B", "X")
                    annotations.append(Annotation("low", "FixtureGene", "B", "X"))
                    definitions = [
                        MaskDefinition("All", frozenset(["A", "B"]), category_order=("B", "A")),
                        MaskDefinition("Filtered", frozenset(["A"]),
                                       extract_variants=frozenset(["low", "at", "half", "missing", "collapse_at"]),
                                       score="score_filter")]
                    config = GeneConfig(
                        aaf_bins=(1 / 64, .1), vc_max_aaf=.1, min_mac=.5,
                        variant_block_size=3, vc_storage=storage,
                        vc_score_method="crossproduct" if storage == "sparse" else "residual",
                        extract_variants=[v for v in identifiers if v != "forbidden"],
                        domain_mapping={"Exact": FrequencyDomain(1 / 128, 1 / 128),
                                        "Open": FrequencyDomain(1 / 128, 1 / 64, False, True)})
                    actual = self._build(g, identifiers, annotations, definitions, config)
                    expected = self._build(g, identifiers, annotations, definitions, config, legacy=True)
                    self._assert_masks_equal(actual, expected)
                    by_key = {(m.base_name, m.domain, m.frequency): m for m in actual}
                    threshold = by_key[("All", None, "0.015625")]
                    self.assertEqual(threshold.variant_ids, ("low", "at", "half", "missing", "other_mask"))
                    singleton = by_key[("All", None, "singleton")]
                    self.assertEqual(singleton.variant_ids, ("low", "half", "missing", "other_mask"))
                    self.assertEqual(by_key[("All", "Exact", "0.015625")].variant_ids,
                                     ("low", "other_mask"))
                    self.assertEqual(by_key[("All", "Open", "0.015625")].variant_ids,
                                     ("at", "missing"))
                    self.assertEqual(by_key[("Filtered", None, "0.1")].variant_ids,
                                     ("low", "at", "half", "collapse_at", "missing"))
                    full = by_key[("All", None, "0.1")]
                    self.assertEqual(full.vc_genotypes.shape[1], 2)  # AAC > 10, then collapsed AAC <= 10.
                    self.assertEqual(full.raw_burden[5].item(), .5)  # MAC == .5 is eligible.
                    self.assertEqual(full.raw_burden[6].item(), 0.)  # MAC < .5 is excluded.
                    self.assertEqual(full.raw_burden[34].item(), 0.)  # Global whitelist.

    def test_singletons_use_alt_count_before_minor_allele_flip(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                g = torch.zeros((100, 2), device=self.device, dtype=dtype)
                g[:, 0], g[0, 0], g[1, 1] = 2., 1., 1.
                ids = ["high_alt", "one_alt"]
                annotations = [Annotation(v, "FixtureGene", "A") for v in ids]
                definitions = [MaskDefinition("M", frozenset(["A"]))]
                config = GeneConfig(aaf_bins=(1.,), vc_max_aaf=1., include_domains=False)
                actual = self._build(g, ids, annotations, definitions, config)
                self._assert_masks_equal(actual, self._build(
                    g, ids, annotations, definitions, config, legacy=True))
                singleton = next(m for m in actual if m.frequency == "singleton")
                self.assertEqual(singleton.variant_ids, ("one_alt",))
                full = next(m for m in actual if m.frequency == "1")
                torch.testing.assert_close(full.raw_burden, g[:, 0], rtol=0, atol=0)
                expected_vc = torch.stack((2 - g[:, 0], g[:, 1]), 1)
                torch.testing.assert_close(full.vc_genotypes, expected_vc, rtol=0, atol=0)

    def test_singleton_carrier_option_keeps_homozygote_and_excludes_two_carriers(self):
        g = torch.zeros((64, 2), device=self.device, dtype=torch.float64)
        g[0, 0], g[1:3, 1] = 2., 1.
        ids = ["one_carrier", "two_carriers"]
        annotations = [Annotation(v, "FixtureGene", "A") for v in ids]
        definitions = [MaskDefinition("M", frozenset(["A"]))]
        for carrier in (False, True):
            with self.subTest(singleton_carrier=carrier):
                config = GeneConfig(aaf_bins=(.1,), vc_max_aaf=.1,
                                    include_domains=False, singleton_carrier=carrier)
                actual = self._build(g, ids, annotations, definitions, config)
                self._assert_masks_equal(actual, self._build(
                    g, ids, annotations, definitions, config, legacy=True))
                singleton = [m for m in actual if m.frequency == "singleton"]
                self.assertEqual([m.variant_ids for m in singleton],
                                 [("one_carrier",)] if carrier else [])

    def test_missing_max_and_low_mac_release_keep_raw_burden_and_vc_exact(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                g = torch.zeros((1000, 3), device=self.device, dtype=dtype)
                g[:12, 0], g[12:24, 1], g[24, 2] = 1., 1., 1.
                g[25, :2], g[26, :] = float("nan"), float("nan")
                ids = ["regular_first", "regular_second", "singleton"]
                annotations = [Annotation(v, "FixtureGene", "A", "R") for v in ids]
                definitions = [MaskDefinition("M", frozenset(["A"]))]
                config = GeneConfig(variant_block_size=2)
                actual = self._build(g, ids, annotations, definitions, config)
                self._assert_masks_equal(actual, self._build(
                    g, ids, annotations, definitions, config, legacy=True))
                overall = next(m for m in actual if m.domain is None and m.frequency == "0.01")
                self.assertEqual(overall.n_observed, 999)
                self.assertTrue(torch.isnan(overall.raw_burden[26]).item())
                self.assertEqual(overall.raw_burden[25].item(), 0.)
                self.assertEqual(overall.vc_genotypes.shape[1], 3)
                # CUDA scalar division can round differently from Python
                # 25/999. Preserve the original device AF then scalar fill.
                expected_af = float((g.new_tensor(25.) / (2 * 999)).item())
                self.assertEqual(overall.burden[26].item(),
                                 g.new_tensor(2 * expected_af).item())
                builder = GeneMaskBuilder(annotations, definitions, 1000, self.device,
                                          dtype, GeneConfig(min_mac=30))
                builder.update(ids, g)
                references = [selection.shared for state in builder.states
                              for selection in state.vc_chunks]
                self.assertTrue(references)
                self.assertEqual(builder.finish(), [])
                self.assertEqual(builder.retained_bytes, 0)
                self.assertTrue(all(shared.values is None for shared in references))


class MaskMetadataCPU(_MaskMetadataChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class MaskMetadataCUDA(_MaskMetadataChecks, unittest.TestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
