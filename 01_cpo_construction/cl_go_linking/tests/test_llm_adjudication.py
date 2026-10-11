from __future__ import annotations

import unittest
from pathlib import Path

from GO_link.apply_llm_adjudication import merge_classifications
from GO_link.llm_adjudicate_undetermined import (
    go_ids_from_error_batches,
    llm_messages,
    parse_model_json,
    validate_decisions,
)
from celltype_configure.common import CatalogNode, CellCatalog


def catalog() -> CellCatalog:
    return CellCatalog(
        [
            CatalogNode(
                "CL:0000000", "cell", "Cell", "", "CL:0000000", "Cell", "skeleton", "root", 0
            ),
            CatalogNode(
                "CL:0001000", "immune cell", "Immune cell", "CL:0000000", "CL:0001000", "Immune cell", "skeleton", "immune", 1
            ),
            CatalogNode(
                "CL:0001001", "T cell", "T cell", "CL:0001000", "CL:0001001", "T cell", "skeleton", "immune", 2
            ),
        ]
    )


def item() -> dict:
    return {
        "go_id": "GO:0000001",
        "name": "T cell activation",
        "candidates": [{"cl_id": "CL:0001001"}],
    }


class ModelOutputTests(unittest.TestCase):
    def test_parse_json_code_fence(self) -> None:
        self.assertEqual(parse_model_json('```json\n{"decisions": []}\n```'), {"decisions": []})

    def test_validate_rejects_cl_outside_candidates(self) -> None:
        output = {
            "decisions": [
                {
                    "go_id": "GO:0000001",
                    "classification": "cell_type_specific",
                    "constraint_cl_id": "CL:9999999",
                    "confidence": "high",
                    "reason": "not actually allowed",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "Invalid candidate selection"):
            validate_decisions(output, [item()])

    def test_validate_clears_cl_for_general_decision(self) -> None:
        output = {
            "decisions": [
                {
                    "go_id": "GO:0000001",
                    "classification": "cell_general",
                    "constraint_cl_id": "CL:0001001",
                    "confidence": "medium",
                    "reason": "broad process",
                }
            ]
        }
        decision = validate_decisions(output, [item()])["GO:0000001"]
        self.assertEqual(decision["llm_constraint_cl_id"], "")

    def test_repair_message_contains_validation_error(self) -> None:
        messages = llm_messages([item()], "invalid candidate CL:9999999")
        self.assertEqual(len(messages), 3)
        self.assertIn("CL:9999999", messages[-1]["content"])
        self.assertIn("supplied candidates", messages[-1]["content"])


class MergeTests(unittest.TestCase):
    def original(self) -> dict[str, str]:
        return {
            "go_id": "GO:0000001",
            "go_name": "T cell activation",
            "classification": "undetermined",
            "confidence": "low",
            "review_required": "true",
            "constraint_scope": "",
            "constraint_root_cl_ids": "",
            "constraint_root_labels": "",
            "decision_reason": "old",
        }

    def decision(self, confidence: str = "high") -> dict[str, str]:
        return {
            "go_id": "GO:0000001",
            "llm_classification": "cell_type_specific",
            "llm_constraint_cl_id": "CL:0001001",
            "llm_constraint_label": "T cell",
            "llm_confidence": confidence,
            "llm_reason": "intrinsic T-cell process",
            "llm_status": "success",
        }

    def test_medium_or_higher_specific_becomes_candidate(self) -> None:
        rows, counts = merge_classifications(
            [self.original()], [self.decision("medium")], catalog(), "medium"
        )
        self.assertEqual(rows[0]["classification"], "cell_type_specific_candidate")
        self.assertEqual(rows[0]["constraint_root_cl_ids"], "CL:0001001")
        self.assertEqual(counts["accepted_specific"], 1)

    def test_low_confidence_specific_stays_undetermined(self) -> None:
        rows, counts = merge_classifications(
            [self.original()], [self.decision("low")], catalog(), "medium"
        )
        self.assertEqual(rows[0]["classification"], "undetermined")
        self.assertEqual(rows[0]["constraint_root_cl_ids"], "")
        self.assertEqual(counts["below_confidence_threshold"], 1)


class RetrySelectionTests(unittest.TestCase):
    def test_missing_error_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            go_ids_from_error_batches(Path("/definitely/missing/error-batches"))


if __name__ == "__main__":
    unittest.main()
