from __future__ import annotations

import unittest

from celltype_configure.common import CatalogNode, CellCatalog
from GO_link.llm_adjudicate_direct_phenotypes import (
    SYSTEM_PROMPT,
    annotate_decision,
    build_result_tables,
    llm_messages,
    normalize_decision,
    select_targets,
    validate_decisions,
)


def catalog() -> CellCatalog:
    return CellCatalog(
        [
            CatalogNode(
                "CL:0000000",
                "cell",
                "Cell",
                "",
                "CL:0000000",
                "Cell",
                "skeleton",
                "root",
                0,
            ),
            CatalogNode(
                "CL:0001000",
                "leukocyte",
                "Leukocytes",
                "CL:0000000",
                "CL:0001000",
                "Leukocytes",
                "skeleton",
                "immune",
                1,
            ),
            CatalogNode(
                "CL:0002000",
                "neuron",
                "Neuron",
                "CL:0000000",
                "CL:0002000",
                "Neuron",
                "skeleton",
                "neural",
                1,
            ),
            CatalogNode(
                "CL:0001001",
                "monocyte",
                "Monocyte",
                "CL:0001000",
                "CL:0001001",
                "Monocyte",
                "skeleton",
                "immune",
                2,
            ),
        ]
    )


def llm_item() -> dict:
    return {
        "go_id": "GO:1",
        "name": "leukocyte differentiation",
        "candidates": [
            {"cl_id": "CL:0001000"},
            {"cl_id": "CL:0002000"},
        ],
    }


class TargetSelectionTests(unittest.TestCase):
    def test_union_of_undetermined_and_candidate_review_is_deduplicated(self) -> None:
        semantic = [
            {
                "go_id": "GO:1",
                "go_name": "one",
                "semantic_classification": "undetermined",
            },
            {
                "go_id": "GO:2",
                "go_name": "two",
                "semantic_classification": "cell_general_candidate",
            },
        ]
        assignments = [
            {"go_id": "GO:1", "assignment_status": "candidate_only"},
            {"go_id": "GO:2", "assignment_status": "candidate_only"},
        ]

        rows = select_targets(semantic, assignments)

        self.assertEqual([row["go_id"] for row in rows], ["GO:1", "GO:2"])
        self.assertEqual(
            rows[0]["task_types"],
            "resolve_undetermined|review_candidate_links",
        )
        self.assertEqual(rows[1]["task_types"], "review_candidate_links")


class DirectPromptValidationTests(unittest.TestCase):
    def test_prompt_uses_direct_phenotype_not_executor_only_semantics(self) -> None:
        self.assertIn("leukocyte differentiation", SYSTEM_PROMPT)
        self.assertIn("executor-only", SYSTEM_PROMPT)

    def test_validate_allows_multiple_supplied_cells(self) -> None:
        result = validate_decisions(
            {
                "decisions": [
                    {
                        "go_id": "GO:1",
                        "scope": "cell_type_specific",
                        "selected_cl_ids": ["CL:0001000", "CL:0002000"],
                        "confidence": "medium",
                        "reason": "both named lineages are intrinsic",
                    }
                ]
            },
            [llm_item()],
        )["GO:1"]

        self.assertEqual(
            result["llm_selected_cl_ids"], ["CL:0001000", "CL:0002000"]
        )

    def test_validate_rejects_cell_outside_fixed_recall(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid candidate selection"):
            validate_decisions(
                {
                    "decisions": [
                        {
                            "go_id": "GO:1",
                            "scope": "cell_type_specific",
                            "selected_cl_ids": ["CL:9999999"],
                            "confidence": "high",
                            "reason": "invented candidate",
                        }
                    ]
                },
                [llm_item()],
            )

    def test_repair_message_requires_empty_cells_for_general_scope(self) -> None:
        messages = llm_messages([llm_item()], "invalid selected cells")
        self.assertIn("it must be []", messages[-1]["content"])

    def test_prompt_does_not_use_candidate_definition_as_specificity_evidence(self) -> None:
        self.assertIn(
            "not evidence that the GO process is specific",
            " ".join(SYSTEM_PROMPT.split()),
        )

    def test_prompt_preserves_signal_source_and_responder_direction(self) -> None:
        prompt = " ".join(SYSTEM_PROMPT.split())
        self.assertIn("upstream stimulus", prompt)
        self.assertIn("responding cell, not the signal source", prompt)
        self.assertIn("General across subtypes of one lineage", prompt)

    def test_unresolved_medium_decision_is_not_accepted(self) -> None:
        row = annotate_decision(
            {
                "go_id": "GO:1",
                "go_name": "unclear process",
                "task_types": "resolve_undetermined",
            },
            {
                "llm_scope": "unresolved",
                "llm_selected_cl_ids": [],
                "llm_confidence": "medium",
                "llm_reason": "insufficient ontology context",
            },
            [],
            catalog(),
            "batch_1",
            {},
        )

        self.assertFalse(row["decision_accepted"])
        self.assertTrue(row["review_required"])

    def test_specific_reason_conflict_is_not_accepted(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:1",
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0001000"],
                "llm_confidence": "high",
                "llm_reason": "None of the supplied candidates fits, so this is unresolved.",
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                }
            ],
            catalog(),
        )

        self.assertFalse(row["decision_accepted"])
        self.assertIn("specific_scope_reason_conflict", row["quality_flags"])
        self.assertEqual(row["adjudicated_scope"], "unresolved")

    def test_specific_choice_among_many_other_cells_is_not_accepted(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:1",
                "go_name": "generic secretion",
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0001000"],
                "llm_confidence": "high",
                "llm_reason": (
                    "Many other cell types can perform this process, but "
                    "leukocytes are the canonical example."
                ),
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                }
            ],
            catalog(),
        )

        self.assertFalse(row["decision_accepted"])
        self.assertIn("specific_scope_reason_conflict", row["quality_flags"])

    def test_general_across_one_lineage_is_not_accepted_as_cell_general(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:1",
                "go_name": "epithelial organization",
                "llm_scope": "cell_general",
                "llm_selected_cl_ids": [],
                "llm_confidence": "high",
                "llm_reason": (
                    "This occurs in many types of epithelial cells but is not "
                    "specific to a particular subtype."
                ),
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                }
            ],
            catalog(),
        )

        self.assertFalse(row["decision_accepted"])
        self.assertIn("general_scope_lineage_conflict", row["quality_flags"])

    def test_selected_descendant_is_pruned_when_parent_is_also_selected(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:1",
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0001000", "CL:0001001"],
                "llm_confidence": "high",
                "llm_reason": "The GO explicitly names the leukocyte lineage.",
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                },
                {
                    "cl_id": "CL:0001001",
                    "candidate_rank": 2,
                    "recall_sources": "sapbert_catalog",
                },
            ],
            catalog(),
        )

        self.assertEqual(row["llm_selected_cl_ids"], ["CL:0001000"])
        self.assertIn("redundant_descendant_pruned", row["quality_flags"])

    def test_cell_name_inside_factor_name_is_not_accepted_as_specific(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:2",
                "go_name": "hepatocyte growth factor production",
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0001000"],
                "llm_confidence": "high",
                "llm_reason": "The GO name contains hepatocyte.",
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                }
            ],
            catalog(),
        )

        self.assertFalse(row["decision_accepted"])
        self.assertIn("cell_name_in_molecule_name", row["quality_flags"])

    def test_explicit_cell_process_survives_factor_name_guard(self) -> None:
        row = normalize_decision(
            {
                "go_id": "GO:3",
                "go_name": (
                    "leukocyte chemotaxis to vascular endothelial growth factor"
                ),
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0001000"],
                "llm_confidence": "high",
                "llm_reason": "Leukocyte chemotaxis explicitly names the responding cell.",
                "llm_status": "success",
            },
            [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                }
            ],
            catalog(),
        )

        self.assertTrue(row["decision_accepted"])
        self.assertNotIn("cell_name_in_molecule_name", row["quality_flags"])


class ResultMergeTests(unittest.TestCase):
    def test_general_resolution_removes_old_undetermined_supported_link(self) -> None:
        targets = [
            {
                "go_id": "GO:1",
                "go_name": "general process",
                "task_types": "resolve_undetermined",
            }
        ]
        decisions = [
            {
                "go_id": "GO:1",
                "go_name": "general process",
                "task_types": "resolve_undetermined",
                "llm_scope": "cell_general",
                "llm_selected_cl_ids": [],
                "llm_confidence": "high",
                "llm_reason": "widely cell autonomous",
                "decision_accepted": True,
                "review_required": False,
                "llm_status": "success",
            }
        ]
        accepted = [
            {
                "go_id": "GO:1",
                "go_name": "general process",
                "cl_id": "CL:0001000",
                "cl_name": "leukocyte",
                "catalog_label": "Leukocytes",
                "branch": "immune",
                "link_status": "supported",
                "confidence": "medium",
                "review_required": "false",
                "evidence_sources": "sapbert_catalog",
                "decision_reason": "old",
            }
        ]

        _, _, final_links, general = build_result_tables(
            targets,
            decisions,
            {"GO:1": []},
            accepted,
            [],
            catalog(),
        )

        self.assertEqual(final_links, [])
        self.assertEqual(len(general), 1)

    def test_review_can_reject_proposal_and_accept_recalled_alternative(self) -> None:
        targets = [
            {
                "go_id": "GO:2",
                "go_name": "neuron process",
                "task_types": "review_candidate_links",
            }
        ]
        decisions = [
            {
                "go_id": "GO:2",
                "go_name": "neuron process",
                "task_types": "review_candidate_links",
                "llm_scope": "cell_type_specific",
                "llm_selected_cl_ids": ["CL:0002000"],
                "llm_confidence": "medium",
                "llm_reason": "the process is neuronal",
                "decision_accepted": True,
                "review_required": False,
                "llm_status": "success",
            }
        ]
        candidates = {
            "GO:2": [
                {
                    "cl_id": "CL:0001000",
                    "candidate_rank": 1,
                    "recall_sources": "sapbert_catalog",
                },
                {
                    "cl_id": "CL:0002000",
                    "candidate_rank": 2,
                    "recall_sources": "lexical_char_tfidf",
                },
            ]
        }
        pending = [
            {
                "go_id": "GO:2",
                "go_name": "neuron process",
                "cl_id": "CL:0001000",
                "cl_name": "leukocyte",
                "catalog_label": "Leukocytes",
                "link_status": "candidate",
                "assignment_method": "sapbert_catalog",
                "evidence_sources": "sapbert_catalog",
            }
        ]

        specific, reviews, final_links, _ = build_result_tables(
            targets,
            decisions,
            candidates,
            [],
            pending,
            catalog(),
        )

        self.assertEqual(reviews[0]["review_verdict"], "rejected")
        self.assertEqual(specific[0]["link_origin"], "recalled_alternative")
        self.assertEqual(final_links[0]["cl_id"], "CL:0002000")


if __name__ == "__main__":
    unittest.main()
