from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "build_hpo_cpo_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("hpo_cpo_dataset_builder", MODULE_PATH)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class HpoCpoDatasetBuilderTest(unittest.TestCase):
    def recovery_row(self) -> dict[str, str]:
        return {
            "relation_id": "recovery_1",
            "hpo_id": "HP:1",
            "hpo_label": "Recovered phenotype",
            "cl_id": "CL:1",
            "cl_label": "test cell",
            "go_id": "GO:1",
            "go_label": "test process",
            "direction": "decreased",
            "evidence_tier": "direct",
            "bridge_scopes": "direct_hpo",
            "relation_types": "causally_upstream_of",
            "species": "human; mouse",
            "claim_count": "2",
            "source_record_ids": "pmid_1; pmid_2",
            "source_urls": "https://one.example/; https://two.example/",
            "evidence_quotes": "first quote || second quote",
        }

    def test_recovery_mapping_uses_unified_relation_schema(self) -> None:
        mapped = BUILDER.map_recovery_relation(
            self.recovery_row(), "anatomical_structural"
        )

        self.assertEqual(tuple(mapped), BUILDER.RELATION_FIELDS)
        self.assertEqual(mapped["link_level"], "direct")
        self.assertEqual(mapped["cell_text"], "test cell")
        self.assertEqual(mapped["process_text"], "test process")
        self.assertEqual(mapped["article_count"], "2")
        self.assertEqual(mapped["representative_quote"], "first quote")
        self.assertEqual(mapped["record_id"], "pmid_1")
        self.assertEqual(mapped["source_url"], "https://one.example/")
        self.assertNotIn("source_batch", mapped)

    def test_exact_duplicate_is_removed_without_a_batch_dimension(self) -> None:
        status = {"HP:1": {"primary_class": "anatomical_structural"}}
        mapped = BUILDER.map_recovery_relation(
            self.recovery_row(), "anatomical_structural"
        )

        merged, duplicate_count = BUILDER.merge_relations(
            [mapped], [self.recovery_row()], status
        )

        self.assertEqual(merged, [mapped])
        self.assertEqual(duplicate_count, 1)

    def test_overview_recalculates_merged_relation_status(self) -> None:
        recovery = self.recovery_row()
        status = {
            "hpo_id": "HP:1",
            "hpo_label": "Recovered phenotype",
            "primary_class": "anatomical_structural",
            "retrieved_record_count": "20",
            "candidate_count": "10",
            "raw_claim_count": "2",
            "coverage_status": "direct_candidate",
        }
        relation = BUILDER.map_recovery_relation(
            recovery, "anatomical_structural"
        )

        overview = BUILDER.merge_overview([], [status], [recovery], [relation])

        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0]["link_status"], "direct")
        self.assertEqual(overview[0]["unique_relation_count"], 1)
        self.assertEqual(overview[0]["claim_count"], 2)
        self.assertEqual(overview[0]["direct_claim_count"], 2)
        self.assertIn("CL:1 test cell (1)", overview[0]["direct_cl_links"])


if __name__ == "__main__":
    unittest.main()
