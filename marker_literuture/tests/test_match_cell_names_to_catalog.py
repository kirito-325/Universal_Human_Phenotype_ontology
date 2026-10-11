from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import match_cell_names_to_catalog as matching  # noqa: E402


class MatchCellNamesTests(unittest.TestCase):
    def test_load_precomputed_candidates_preserves_rank_and_limit(self) -> None:
        catalog = [
            matching.CatalogEntry("CL:1", "T cell", "T cell", "", "Lymphocyte", "immune", "type"),
            matching.CatalogEntry("CL:2", "Tc2 cell", "Tc2 cell", "", "T cell", "immune", "type"),
        ]
        source = matching.SourceEntry(0, "CD8 T cell", {})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source_index",
                        "cell_name",
                        "candidate_rank",
                        "catalog_id",
                        "similarity_score",
                        "embedding_model",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "source_index": 0,
                            "cell_name": "CD8 T cell",
                            "candidate_rank": 2,
                            "catalog_id": "CL:1",
                            "similarity_score": 0.7,
                            "embedding_model": "sapbert",
                        },
                        {
                            "source_index": 0,
                            "cell_name": "CD8 T cell",
                            "candidate_rank": 1,
                            "catalog_id": "CL:2",
                            "similarity_score": 0.8,
                            "embedding_model": "sapbert",
                        },
                    ]
                )
            candidates, metadata = matching.load_precomputed_candidates(
                path, catalog, [source], 1
            )
        self.assertEqual(candidates[0][0][0].catalog_id, "CL:2")
        self.assertEqual(metadata["models"], ["sapbert"])

    def test_llm_payload_contains_context(self) -> None:
        items = [
            {
                "source_index": 3,
                "source_name": "DC",
                "paper_count": 4,
                "source_types": "statistics",
                "example_titles": ["A dendritic-cell atlas"],
                "candidates": [],
            }
        ]
        messages = matching.llm_messages(items)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["items"][0]["paper_count"], 4)
        self.assertEqual(payload["items"][0]["example_titles"], ["A dendritic-cell atlas"])
        self.assertEqual(
            payload["schema"]["matches"][0]["source_name"],
            "exact input source_name",
        )

    def test_normalization_distinguishes_signs_from_hyphenated_words(self) -> None:
        self.assertEqual(matching.normalize_name("CD8+ T-cell"), "cd8 positive t cell")
        self.assertEqual(
            matching.normalize_name("CD8-positive, alpha-beta T cell"),
            "cd8 positive alpha beta t cell",
        )

    def test_exact_alias_recall_handles_parenthetical_abbreviation(self) -> None:
        catalog = [
            matching.CatalogEntry(
                "CL:1",
                "conventional dendritic cell",
                "conventional dendritic cell",
                "",
                "Dendritic cell",
                "immune",
                "type",
            )
        ]
        index = matching.build_exact_catalog_index(catalog)
        match = matching.exact_catalog_match(
            "conventional dendritic cell (cDC)", catalog, index
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.catalog_id, "CL:1")

        memory_catalog = [
            matching.CatalogEntry(
                "CL:2",
                "effector memory CD8-positive, alpha-beta T cell",
                "effector memory CD8-positive, alpha-beta T cell",
                "",
                "memory T cell",
                "immune",
                "type",
            )
        ]
        memory_index = matching.build_exact_catalog_index(memory_catalog)
        memory_match = matching.exact_catalog_match(
            "CD8+ Tem T cell", memory_catalog, memory_index
        )
        self.assertIsNotNone(memory_match)
        self.assertEqual(memory_match.catalog_id, "CL:2")

    def test_strict_safety_flags_mixed_and_generalized_matches(self) -> None:
        self.assertIn(
            "mixed_population",
            matching.semantic_safety_flags(
                "Microglia/macrophage", "match", "best match is microglial cell"
            ),
        )
        self.assertIn(
            "admitted_generalization",
            matching.semantic_safety_flags(
                "CAF", "match", "the closest general match is stromal cell"
            ),
        )
        self.assertIn(
            "low_similarity_match",
            matching.semantic_safety_flags(
                "mMac1", "match", "meningeal macrophage is plausible", 0.42
            ),
        )
        self.assertIn(
            "contradictory_reason",
            matching.semantic_safety_flags(
                "mMac1", "match", "meningeal macrophage is plausible", 0.42
            ),
        )
        self.assertNotIn(
            "source_reason_mismatch",
            matching.semantic_safety_flags(
                "Enterocyte (cluster 2)",
                "match",
                "Source name 'Enterocyte' matches enterocyte.",
            ),
        )
        self.assertIn(
            "source_reason_mismatch",
            matching.semantic_safety_flags(
                "Gallbladder_Goblet cells",
                "match",
                "Source name 'Gallbladder_Macrophages' matches macrophage.",
            ),
        )

    def test_safety_check_preserves_review_audit_metadata(self) -> None:
        row = {
            "cell_name": "Microglia/macrophage",
            "decision": "review",
            "validation_flags": "mixed_population",
            "proposed_catalog_id": "CL:0000129",
            "proposed_catalog_label": "Microglial cell",
        }
        matching.apply_semantic_safety(row)
        self.assertEqual(row["validation_flags"], "mixed_population")
        self.assertEqual(row["proposed_catalog_id"], "CL:0000129")

    def test_model_result_requires_matching_source_name(self) -> None:
        items = [
            {
                "source_index": 3,
                "source_name": "Goblet cell",
                "candidates": [{"catalog_id": "CL:1"}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "source_name mismatch"):
            matching.validate_model_results(
                {
                    "matches": [
                        {
                            "source_index": 3,
                            "source_name": "Macrophage",
                            "decision": "match",
                            "catalog_id": "CL:1",
                        }
                    ]
                },
                items,
            )

    def test_load_reusable_mapping_keeps_only_final_consistent_decisions(self) -> None:
        catalog = [
            matching.CatalogEntry("CL:1", "T cell", "T cell", "", "Lymphocyte", "immune", "type"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cell_name", "decision", "confidence", "catalog_id", "reason"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "cell_name": "CD3 T cell",
                            "decision": "match",
                            "confidence": "high",
                            "catalog_id": "CL:1",
                            "reason": "confirmed",
                        },
                        {
                            "cell_name": "Microglia/macrophage",
                            "decision": "match",
                            "confidence": "high",
                            "catalog_id": "CL:1",
                            "reason": "combined label; choosing microglia",
                        },
                        {"cell_name": "artifact", "decision": "no_match", "catalog_id": ""},
                        {"cell_name": "pending", "decision": "pending_llm", "catalog_id": ""},
                        {"cell_name": "bad id", "decision": "match", "catalog_id": "CL:999"},
                    ]
                )
            reusable, summary = matching.load_reusable_mapping(path, catalog)

        self.assertEqual(reusable["cd3 t cell"]["catalog_id"], "CL:1")
        self.assertEqual(reusable["artifact"]["decision"], "no_match")
        self.assertNotIn("pending", reusable)
        self.assertNotIn("bad id", reusable)
        self.assertNotIn("microglia macrophage", reusable)
        self.assertEqual(summary["reusable_rows"], 2)
        self.assertEqual(summary["skipped_counts"]["unsafe_mixed_population"], 1)

    def test_review_reuse_keeps_completed_review_but_retries_llm_failure(self) -> None:
        catalog = [
            matching.CatalogEntry("CL:1", "T cell", "T cell", "", "", "immune", "type")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "cell_name",
                        "decision",
                        "confidence",
                        "catalog_id",
                        "reason",
                        "validation_flags",
                        "proposed_catalog_id",
                        "proposed_catalog_label",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "cell_name": "mixed population",
                            "decision": "review",
                            "confidence": "low",
                            "reason": "ambiguous source",
                            "validation_flags": "mixed_population",
                            "proposed_catalog_id": "CL:1",
                            "proposed_catalog_label": "T cell",
                        },
                        {
                            "cell_name": "failed population",
                            "decision": "review",
                            "confidence": "low",
                            "reason": "LLM error: timed out",
                        },
                    ]
                )
            reusable, summary = matching.load_reusable_mapping(
                path, catalog, reuse_reviews=True
            )

        self.assertIn("mixed population", reusable)
        self.assertNotIn("failed population", reusable)
        self.assertEqual(
            reusable["mixed population"]["validation_flags"], "mixed_population"
        )
        self.assertEqual(reusable["mixed population"]["proposed_catalog_id"], "CL:1")
        self.assertEqual(summary["skipped_counts"]["review_llm_failure"], 1)


if __name__ == "__main__":
    unittest.main()
