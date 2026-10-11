from __future__ import annotations

import unittest

import networkx as nx

from GO_link.classify_go_to_cl import aggregate_database, fuse_row
from celltype_configure.common import CatalogNode, CellCatalog, nearest_catalog_ancestor
from celltype_configure.database_depend.scea_enrichment import benjamini_hochberg
from celltype_configure.text_depend.semantic_mapper import semantic_label


def catalog() -> CellCatalog:
    return CellCatalog(
        [
            CatalogNode("CL:0000000", "cell", "Cell", "", "CL:0000000", "Cell", "skeleton", "root", 0),
            CatalogNode("CL:0001000", "immune cell", "Immune", "CL:0000000", "CL:0001000", "Immune", "skeleton", "immune", 1),
            CatalogNode("CL:0001001", "neutrophil", "Neutrophil", "CL:0001000", "CL:0001001", "Neutrophil", "skeleton", "immune", 2),
            CatalogNode("CL:0001002", "mature neutrophil", "Mature neutrophil", "CL:0001001", "CL:0001001", "Neutrophil", "extension", "immune", 3),
            CatalogNode("CL:0002000", "neuron", "Neuron", "CL:0000000", "CL:0002000", "Neuron", "skeleton", "neural", 1),
        ]
    )


class CatalogTests(unittest.TestCase):
    def test_closure_and_common_ancestor(self) -> None:
        value = catalog()
        self.assertEqual(
            value.descendants("CL:0001000", include_self=True),
            ["CL:0001000", "CL:0001001", "CL:0001002"],
        )
        self.assertEqual(
            value.common_ancestor(["CL:0001001", "CL:0001002"]), "CL:0001001"
        )
        self.assertEqual(value.common_ancestor(["CL:0001002", "CL:0002000"]), "")

    def test_nearest_catalog_ancestor_returns_id_and_distance(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("CL:9999999", "CL:0001002")
        self.assertEqual(
            nearest_catalog_ancestor(graph, catalog(), "CL:9999999"),
            ("CL:0001002", 1),
        )


class SemanticTests(unittest.TestCase):
    def test_three_state_thresholds(self) -> None:
        self.assertEqual(
            semantic_label(0.70, 0.08, 0.62, 0.04, 0.52),
            "cell_type_specific_candidate",
        )
        self.assertEqual(semantic_label(0.49, 0.01, 0.62, 0.04, 0.52), "cell_general_candidate")
        self.assertEqual(semantic_label(0.58, 0.03, 0.62, 0.04, 0.52), "undetermined")


class DatabaseTests(unittest.TestCase):
    def test_bh_is_monotonic_in_pvalue_order(self) -> None:
        pvalues = [0.04, 0.001, 0.02]
        adjusted = benjamini_hochberg(pvalues)
        ordered = sorted(zip(pvalues, adjusted))
        self.assertLessEqual(ordered[0][1], ordered[1][1])
        self.assertLessEqual(ordered[1][1], ordered[2][1])

    def test_repeated_concentrated_evidence_is_specific(self) -> None:
        rows = [
            {
                "go_id": "GO:1",
                "is_significant": "true",
                "experiment_id": experiment,
                "catalog_anchor_id": "CL:0001001",
                "catalog_anchor_label": "Neutrophil",
                "evidence_weight": "10",
            }
            for experiment in ("E-1", "E-2")
        ]
        result = aggregate_database(rows, {})["GO:1"]
        self.assertEqual(result["classification"], "cell_type_specific")

    def test_broad_repeated_evidence_is_general(self) -> None:
        rows = []
        for experiment in ("E-1", "E-2", "E-3"):
            for index in range(4):
                rows.append(
                    {
                        "go_id": "GO:2",
                        "is_significant": "true",
                        "experiment_id": experiment,
                        "catalog_anchor_id": f"CL:{index:07d}",
                        "catalog_anchor_label": f"Cell {index}",
                        "evidence_weight": "1",
                    }
                )
        result = aggregate_database(rows, {})["GO:2"]
        self.assertEqual(result["classification"], "cell_general")


class FusionTests(unittest.TestCase):
    def test_curator_reference_has_priority(self) -> None:
        semantic = {
            "go_id": "GO:1",
            "go_name": "neutrophil activation",
            "semantic_classification": "cell_type_specific",
            "semantic_constraint_cl_id": "CL:0001001",
            "semantic_top_score": "0.7",
            "semantic_margin": "0.1",
            "explicit_cl_ids": "CL:0001002",
            "explicit_catalog_cl_ids": "CL:0001002",
            "explicit_constraint_cl_id": "CL:0001002",
        }
        row, roots = fuse_row(semantic, {}, catalog())
        self.assertEqual(row["classification"], "cell_type_specific")
        self.assertEqual(row["confidence"], "high")
        self.assertEqual(roots, ["CL:0001002"])


if __name__ == "__main__":
    unittest.main()
