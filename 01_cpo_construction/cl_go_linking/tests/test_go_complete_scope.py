from __future__ import annotations

import unittest

from GO_link.build_complete_scope import (
    build_scope_rows,
    exact_link_index,
    identify_targets,
)


class CompleteGoScopeTests(unittest.TestCase):
    def test_latest_llm_decision_replaces_all_old_links_for_go(self) -> None:
        existing = [
            {
                "go_id": "GO:1",
                "cl_id": "CL:old",
                "final_link_status": "llm_supported",
            }
        ]
        supplemental = [
            {
                "go_id": "GO:1",
                "go_name": "process",
                "cl_id": "CL:new",
                "cl_name": "new cell",
                "catalog_label": "New cell",
                "branch": "test",
                "llm_confidence": "high",
                "decision_accepted": "true",
                "review_required": "false",
                "recall_sources": "test",
                "link_origin": "recalled_alternative",
                "llm_reason": "Direct cell.",
            }
        ]
        decisions = {
            "GO:1": {
                "adjudicated_scope": "cell_type_specific",
                "llm_selected_cl_ids": "CL:new",
                "llm_confidence": "high",
                "decision_accepted": "true",
                "review_required": "false",
                "llm_status": "success",
            }
        }

        links, _ = exact_link_index(
            existing, supplemental, decisions, {"GO:1"}
        )

        self.assertEqual([row["cl_id"] for row in links], ["CL:new"])

    def test_latest_general_decision_removes_old_specific_links(self) -> None:
        existing = [{"go_id": "GO:1", "cl_id": "CL:old"}]
        decisions = {
            "GO:1": {
                "adjudicated_scope": "cell_general",
                "llm_selected_cl_ids": "",
                "llm_confidence": "high",
                "decision_accepted": "true",
                "review_required": "false",
                "llm_status": "success",
            }
        }

        links, _ = exact_link_index(existing, [], decisions, {"GO:1"})

        self.assertEqual(links, [])

    def test_scope_and_mapping_status_are_separate(self) -> None:
        semantic = [
            {"go_id": "GO:1", "go_name": "named cell process", "semantic_classification": "cell_type_specific"},
            {"go_id": "GO:2", "go_name": "generic process", "semantic_classification": "cell_general_candidate"},
            {"go_id": "GO:3", "go_name": "missing catalog cell process", "semantic_classification": "undetermined"},
            {"go_id": "GO:4", "go_name": "organism process", "semantic_classification": "undetermined"},
        ]
        decisions = {
            "GO:4": {
                "adjudicated_scope": "not_direct_human_cell_process",
                "llm_confidence": "high",
                "decision_accepted": "true",
                "review_required": "false",
                "llm_status": "success",
                "llm_reason": "Organism-level process.",
            }
        }
        links = {
            "GO:1": [
                {
                    "go_id": "GO:1",
                    "cl_id": "CL:1",
                    "catalog_label": "Cell one",
                    "final_link_status": "confirmed",
                    "confidence": "high",
                }
            ]
        }
        overrides = {
            "GO:3": {
                "go_id": "GO:3",
                "cell_scope": "cell_type_specific",
                "confidence": "high",
                "decision_source": "ontology_text_scope_completion",
                "decision_reason": "Exact catalog cell is absent.",
            }
        }

        rows = build_scope_rows(
            semantic,
            {},
            decisions,
            links,
            overrides,
            {"GO:3": ("CL:3",)},
            {"CL:3": "Missing cell"},
            set(),
            {"GO:2"},
            set(),
        )
        by_go = {row["go_id"]: row for row in rows}

        self.assertEqual(by_go["GO:1"]["cell_mapping_status"], "mapped")
        self.assertEqual(by_go["GO:2"]["cell_scope"], "cell_general")
        self.assertEqual(by_go["GO:3"]["cell_scope"], "cell_type_specific")
        self.assertEqual(
            by_go["GO:3"]["cell_mapping_status"], "unmapped_catalog_cell"
        )
        self.assertEqual(
            by_go["GO:4"]["cell_scope"], "not_direct_human_cell_process"
        )

    def test_overrides_and_general_defaults_are_not_llm_targets(self) -> None:
        semantic = [
            {"go_id": "GO:1", "go_name": "generic", "semantic_classification": "cell_general_candidate"},
            {"go_id": "GO:2", "go_name": "curated", "semantic_classification": "undetermined"},
            {"go_id": "GO:3", "go_name": "ambiguous", "semantic_classification": "undetermined"},
        ]

        targets = identify_targets(
            semantic,
            {},
            {},
            {},
            {"GO:2": {"cell_scope": "cell_general"}},
            {},
        )

        self.assertEqual([row["go_id"] for row in targets], ["GO:3"])


if __name__ == "__main__":
    unittest.main()
