from __future__ import annotations

import unittest

from cellphenorank_validation.discovery.candidate_builder import (
    assign_pareto_fronts,
    build_case_candidates,
)


class CandidateDiscoveryTests(unittest.TestCase):
    def test_only_improved_gold_with_function_path_is_retained(self) -> None:
        a4 = [
            {
                "disease_id": "D",
                "disease_name": "Disease",
                "gene_symbol": "GOLD",
                "base_rank": "5",
                "final_rank": "1",
                "supported_hpo_count": "1",
                "supported_hpo_fraction": "1",
                "supported_cpo_count": "1",
                "cpo_enrichment_pvalue": "0.01",
                "cpo_enrichment_fdr": "0.02",
            },
            {
                "disease_id": "D",
                "disease_name": "Disease",
                "gene_symbol": "OTHER",
                "base_rank": "2",
                "final_rank": "2",
            },
        ]
        a5 = [{"disease_id": "D", "gene_symbol": "GOLD", "final_rank": "5"}]
        evidence = [
            {
                "disease_id": "D",
                "gene_symbol": "GOLD",
                "hpo_id": "HP:1",
                "cpo_id": "C1",
                "cl_id": "CL:1",
                "cell_name": "cell",
                "go_id": "GO:1",
                "function_name": "function",
                "path_type": "function",
                "function_membership": "true",
                "marker_membership": "false",
                "hpo_cpo_scope": "cell_type_specific",
                "source_genes": "GOLD;SOURCE",
            }
        ]
        rows, paths, excluded = build_case_candidates(
            a4, a5, evidence, {"D": {"GOLD"}}, {"GO:1": {"GOLD", "X"}}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gene_symbol"], "GOLD")
        self.assertEqual(rows[0]["bridge_dependency_status"], "bridge_fully_dependent")
        self.assertEqual(len(paths), 1)
        self.assertEqual(excluded, [])

    def test_pareto_front_excludes_dominated_candidate(self) -> None:
        strong = {
            "candidate_case_id": "A",
            "disease_id": "D1",
            "gene_symbol": "A",
            "final_rank": 1,
            "rank_change": 5,
            "supported_hpo_count": 3,
            "cell_type_specific_path_count": 2,
            "cell_general_fraction": 0.2,
        }
        weak = {
            "candidate_case_id": "B",
            "disease_id": "D2",
            "gene_symbol": "B",
            "final_rank": 2,
            "rank_change": 2,
            "supported_hpo_count": 1,
            "cell_type_specific_path_count": 0,
            "cell_general_fraction": 1.0,
        }
        ordered = assign_pareto_fronts([strong, weak])
        by_id = {row["candidate_case_id"]: row for row in ordered}
        self.assertEqual(by_id["A"]["pareto_front"], 1)
        self.assertEqual(by_id["B"]["pareto_front"], 2)


if __name__ == "__main__":
    unittest.main()
