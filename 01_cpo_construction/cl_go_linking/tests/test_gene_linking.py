from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from GO_link.link_go_cell_genes import (
    HgncNormalizer,
    link_confidence,
    load_goa_associations,
)


class HgncNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = HgncNormalizer(
            [
                {
                    "status": "Approved",
                    "symbol": "CLDN7",
                    "alias_symbol": "CEPTRL2",
                    "prev_symbol": "OLDCLDN",
                },
                {
                    "status": "Approved",
                    "symbol": "GENE2",
                    "alias_symbol": "AMBIG",
                    "prev_symbol": "",
                },
                {
                    "status": "Approved",
                    "symbol": "GENE3",
                    "alias_symbol": "AMBIG",
                    "prev_symbol": "",
                },
            ]
        )

    def test_exact_and_casefold_approved_symbols(self) -> None:
        exact = self.normalizer.normalize("CLDN7")
        folded = self.normalizer.normalize("Cldn7")
        self.assertEqual((exact.approved_symbol, exact.method), ("CLDN7", "approved_exact"))
        self.assertEqual(
            (folded.approved_symbol, folded.method), ("CLDN7", "approved_casefold")
        )

    def test_unique_aliases_map_and_ambiguous_aliases_do_not(self) -> None:
        alias = self.normalizer.normalize("CEPTRL2")
        self.assertEqual((alias.approved_symbol, alias.method), ("CLDN7", "alias_exact"))
        self.assertIsNone(self.normalizer.normalize("AMBIG"))


class LinkConfidenceTests(unittest.TestCase):
    def test_confirmed_constraint_can_make_high_confidence_link(self) -> None:
        self.assertEqual(
            link_confidence("cell_type_specific", "descendant", 2, 1, 2), "high"
        )

    def test_model_candidate_and_ancestor_context_are_capped(self) -> None:
        self.assertEqual(
            link_confidence("cell_type_specific_candidate", "self", 2, 1, 1),
            "medium",
        )
        self.assertEqual(
            link_confidence("cell_type_specific_candidate", "ancestor", 8, 8, 8),
            "low",
        )


class GoaPropagationTests(unittest.TestCase):
    def test_direct_mode_does_not_promote_child_annotation_to_parent(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("GO:0000002", "GO:0000001")
        gaf_row = "\t".join(
            [
                "UniProtKB",
                "P1",
                "GENE1",
                "involved_in",
                "GO:0000002",
                "PMID:1",
                "EXP",
                "",
                "P",
                "gene one",
                "",
                "protein",
                "taxon:9606",
                "20260718",
                "TEST",
            ]
        )
        with TemporaryDirectory() as directory:
            gaf_path = Path(directory) / "test.gaf"
            gaf_path.write_text(gaf_row + "\n", encoding="utf-8")
            inherited, _ = load_goa_associations(
                gaf_path, graph, {"GO:0000001"}, {"GENE1"}, True
            )
            direct, _ = load_goa_associations(
                gaf_path, graph, {"GO:0000001"}, {"GENE1"}, False
            )
        self.assertIn(("GENE1", "GO:0000001"), inherited)
        self.assertNotIn(("GENE1", "GO:0000001"), direct)


if __name__ == "__main__":
    unittest.main()
