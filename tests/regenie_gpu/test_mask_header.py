"""Native category-registration semantics; these fixtures are not benchmarks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from torchwgs.masks import Annotation, MaskDefinition, effective_mask_definitions
from torchwgs.output import RegenieWriter


class MaskHeaderTests(unittest.TestCase):
    def test_unknown_only_mask_is_removed_and_null_is_pre_registered(self):
        source = [MaskDefinition("Known", frozenset({"lncRNA"})),
                  MaskDefinition("Absent", frozenset({"ribozyme"})),
                  MaskDefinition("Default", frozenset({"NULL"}))]
        annotations = [Annotation("v1", "G1", "lncRNA")]
        effective = effective_mask_definitions(source, annotations, {"v1"})
        self.assertEqual([mask.name for mask in effective], ["Known", "Default"])
        self.assertEqual(effective[1].categories, frozenset({"NULL"}))
        self.assertEqual([mask.name for mask in source], ["Known", "Absent", "Default"])
        self.assertIsNot(effective[0], source[0])
        self.assertIsNone(source[0].category_order)

    def test_mixed_categories_keep_source_order_and_other_fields(self):
        source_order = ("Synonymous", "ribozyme", "Missense", "not_registered")
        source = MaskDefinition("Mixed", frozenset(source_order),
                                extract_variants=frozenset({"v1"}), score="REVEL50",
                                category_order=source_order)
        annotations = [Annotation("v1", "G1", "Synonymous"),
                       Annotation("v2", "G1", "Missense")]
        effective = effective_mask_definitions([source], annotations, {"v1", "v2"})
        self.assertEqual(effective[0].category_order, ("Synonymous", "Missense"))
        self.assertEqual(effective[0].categories, frozenset({"Synonymous", "Missense"}))
        self.assertEqual(effective[0].extract_variants, source.extract_variants)
        self.assertEqual(effective[0].score, "REVEL50")
        self.assertEqual(source.category_order, source_order)
        self.assertEqual(source.categories, frozenset(source_order))
        with TemporaryDirectory() as directory:
            with RegenieWriter(Path(directory)/"gene", "qt", masks=effective,
                               write_samples=False) as writer:
                pass
            self.assertEqual(writer.path.read_text().splitlines()[0],
                             '##MASKS=<Mixed="Synonymous,Missense">')

    def test_registry_keeps_available_annotation_outside_selected_setlist(self):
        # Only G_SELECTED belongs to the chosen setlist. Native registration
        # also sees an available variant annotated to the unselected gene.
        annotations = [Annotation("selected", "G_SELECTED", "PTV"),
                       Annotation("outside", "G_OUTSIDE_SETLIST", "lncRNA")]
        definitions = [MaskDefinition("Coding", frozenset({"PTV"})),
                       MaskDefinition("Noncoding", frozenset({"lncRNA"}))]
        available_lookup = {"selected": object(), "outside": object()}
        effective = effective_mask_definitions(definitions, annotations, available_lookup)
        self.assertEqual([mask.name for mask in effective], ["Coding", "Noncoding"])

    def test_variant_filter_intersection_removes_unique_excluded_categories(self):
        annotations = [Annotation("both", "G1", "PTV"),
                       Annotation("global_only", "G1", "Missense"),
                       Annotation("score_only", "G1", "Splice"),
                       Annotation("absent_from_bim", "G1", "lncRNA")]
        definitions = [MaskDefinition(category, frozenset({category}))
                       for category in ("PTV", "Missense", "Splice", "lncRNA")]
        bim_ids = {"both", "global_only", "score_only"}
        global_whitelist = {"both", "global_only", "absent_from_bim"}
        score_whitelist = {"both", "score_only", "absent_from_bim"}
        available = bim_ids.intersection(global_whitelist, score_whitelist)
        effective = effective_mask_definitions(definitions, annotations, available)
        self.assertEqual([mask.name for mask in effective], ["PTV"])
        self.assertEqual(len(definitions), 4)


if __name__ == "__main__":
    unittest.main()
