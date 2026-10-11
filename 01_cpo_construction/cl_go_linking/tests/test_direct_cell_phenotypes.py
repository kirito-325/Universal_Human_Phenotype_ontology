from __future__ import annotations

import unittest

import networkx as nx

from GO_link.assign_go_cell_phenotypes import (
    assign_direct_links,
    build_human_go_support,
)
from celltype_configure.common import CatalogNode, CellCatalog


def catalog() -> CellCatalog:
    return CellCatalog(
        [
            CatalogNode("CL:0000000", "cell", "Cell", "", "CL:0000000", "Cell", "skeleton", "root", 0),
            CatalogNode("CL:0001000", "leukocyte", "Leukocytes", "CL:0000000", "CL:0001000", "Leukocytes", "skeleton", "immune", 1),
            CatalogNode("CL:0001001", "myeloid leukocyte", "Myeloid leukocytes", "CL:0001000", "CL:0001001", "Myeloid leukocytes", "skeleton", "immune", 2),
            CatalogNode("CL:0001002", "monocyte", "Monocyte", "CL:0001001", "CL:0001002", "Monocyte", "skeleton", "immune", 3),
            CatalogNode("CL:0001003", "mature monocyte", "Mature monocyte", "CL:0001002", "CL:0001002", "Monocyte", "extension", "immune", 4),
            CatalogNode("CL:0002000", "neuron", "Neuron", "CL:0000000", "CL:0002000", "Neuron", "skeleton", "neural", 1),
        ]
    )


def semantic(go_id: str, name: str, explicit: str = "") -> dict[str, str]:
    return {
        "go_id": go_id,
        "go_name": name,
        "explicit_cl_ids": explicit,
        "explicit_catalog_cl_ids": explicit,
    }


def candidate(
    go_id: str, rank: int, cl_id: str, score: float
) -> dict[str, str]:
    return {
        "go_id": go_id,
        "candidate_kind": "catalog_node",
        "rank": str(rank),
        "cl_id": cl_id,
        "score": str(score),
    }


class DirectCellPhenotypeTests(unittest.TestCase):
    def test_explicit_cell_name_beats_over_specific_embedding_hit(self) -> None:
        semantics = [
            semantic("GO:1", "myeloid leukocyte differentiation")
        ]
        candidates = [
            candidate("GO:1", 1, "CL:0001002", 0.72),
            candidate("GO:1", 2, "CL:0001001", 0.69),
            candidate("GO:1", 3, "CL:0002000", 0.55),
        ]
        links, statuses, _, _ = assign_direct_links(
            semantics,
            candidates,
            [],
            catalog(),
            {},
            [{"go_id": "GO:1", "human_function_gene_count": "5"}],
        )

        self.assertEqual([row["cl_id"] for row in links], ["CL:0001001"])
        self.assertEqual(links[0]["link_status"], "supported")
        self.assertEqual(links[0]["name_matched_cl_name"], "myeloid leukocyte")
        self.assertEqual(links[0]["embedding_rank"], 2)
        self.assertEqual(statuses[0]["assignment_status"], "linked")

    def test_name_and_embedding_without_human_support_remain_candidate(self) -> None:
        semantics = [semantic("GO:5", "leukocyte double fertilization")]
        candidates = [
            candidate("GO:5", 1, "CL:0001000", 0.75),
            candidate("GO:5", 2, "CL:0002000", 0.60),
        ]
        links, statuses, _, _ = assign_direct_links(
            semantics,
            candidates,
            [],
            catalog(),
            {},
            [{"go_id": "GO:5", "human_function_gene_count": "0"}],
        )

        self.assertEqual(links[0]["link_status"], "candidate")
        self.assertEqual(statuses[0]["assignment_status"], "candidate_only")

    def test_regulation_term_uses_human_support_from_regulated_process(self) -> None:
        semantics = [semantic("GO:6", "regulation of leukocyte differentiation")]
        candidates = [
            candidate("GO:6", 1, "CL:0001000", 0.72),
            candidate("GO:6", 2, "CL:0002000", 0.60),
        ]
        graph = nx.MultiDiGraph()
        graph.add_edge("GO:6", "GO:BASE", key="regulates")
        links, _, _, _ = assign_direct_links(
            semantics,
            candidates,
            [],
            catalog(),
            {},
            [
                {"go_id": "GO:6", "human_function_gene_count": "0"},
                {"go_id": "GO:BASE", "human_function_gene_count": "4"},
            ],
            graph,
        )

        self.assertEqual(links[0]["link_status"], "supported")
        self.assertEqual(links[0]["human_go_support_kind"], "related_process")
        self.assertEqual(links[0]["human_go_support_go_ids"], "GO:BASE")

    def test_part_of_does_not_transfer_human_applicability(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("GO:PLANT", "GO:BASE", key="part_of")
        support = build_human_go_support(
            [
                {"go_id": "GO:PLANT", "human_function_gene_count": "0"},
                {"go_id": "GO:BASE", "human_function_gene_count": "4"},
            ],
            graph,
        )

        self.assertNotIn("GO:PLANT", support)

    def test_curator_reference_is_direct_and_does_not_expand_descendants(self) -> None:
        semantics = [semantic("GO:2", "immune activation", "CL:0001000")]
        candidates = [candidate("GO:2", 1, "CL:0001002", 0.80)]
        links, _, _, _ = assign_direct_links(
            semantics, candidates, [], catalog(), {}
        )

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["cl_id"], "CL:0001000")
        self.assertEqual(links[0]["link_status"], "confirmed")
        self.assertNotIn("CL:0001002", [row["cl_id"] for row in links])

    def test_embedding_only_assignment_is_a_reviewable_direct_candidate(self) -> None:
        semantics = [semantic("GO:3", "specialized immune process")]
        candidates = [
            candidate("GO:3", 1, "CL:0001002", 0.70),
            candidate("GO:3", 2, "CL:0002000", 0.60),
        ]
        links, statuses, _, _ = assign_direct_links(
            semantics, candidates, [], catalog(), {}
        )

        self.assertEqual(links[0]["cl_id"], "CL:0001002")
        self.assertEqual(links[0]["link_status"], "candidate")
        self.assertEqual(links[0]["review_required"], "true")
        self.assertEqual(statuses[0]["assignment_status"], "candidate_only")

    def test_repeated_database_enrichment_supports_exact_cell_link(self) -> None:
        semantics = [semantic("GO:4", "immune signaling")]
        candidates = [candidate("GO:4", 1, "CL:0002000", 0.50)]
        database = [
            {
                "go_id": "GO:4",
                "catalog_cl_id": "CL:0001002",
                "experiment_id": experiment,
                "cell_label": "monocyte",
                "is_significant": "true",
                "qvalue": "0.01",
                "odds_ratio": "3",
                "overlap_genes": "A|B|C",
            }
            for experiment in ("E-1", "E-2")
        ]
        links, _, _, _ = assign_direct_links(
            semantics, candidates, database, catalog(), {}
        )

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["cl_id"], "CL:0001002")
        self.assertEqual(links[0]["link_status"], "supported")
        self.assertEqual(links[0]["database_experiment_count"], 2)


if __name__ == "__main__":
    unittest.main()
