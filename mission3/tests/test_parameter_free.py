from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellphenorank.binary_cpo import BinaryCpoEdge, BinaryCpoGraph
from cellphenorank.hpo_redundancy import HpoOntology, reduce_hpo_profiles
from cellphenorank.parameter_free_reranking import (
    rerank_constrained,
    rerank_unconstrained,
    score_graph,
)
from cellphenorank.phenotype_stability import (
    calculate_exomiser_phenotype_stability,
    calculate_phenotype_stability,
)
from cellphenorank.schemas import DiseaseTerm, ModelPhenotype, MouseModel
from cellphenorank.similarity import SimilarityProvider


def edge(hpo: str, cpo: str, go: str, sources: tuple[str, ...] = ()) -> BinaryCpoEdge:
    return BinaryCpoEdge(
        hpo_id=hpo,
        hpo_name=hpo,
        cpo_id=cpo,
        cell_id="CL:1",
        cell_name="cell",
        function_id=go,
        function_name=go,
        scope="cell_type_specific",
        relation_type="binary_supported",
        review_status="supported",
        confidence="high",
        raw_confidence="0.9",
        source_genes=sources,
        source_records="",
    )


class HpoReductionTests(unittest.TestCase):
    def test_ancestor_removed_but_sibling_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ontology_path = Path(temporary) / "hp.obo"
            ontology_path.write_text(
                """format-version: 1.2
[Term]
id: HP:1
name: parent
[Term]
id: HP:2
name: child
is_a: HP:1 ! parent
[Term]
id: HP:3
name: sibling
is_a: HP:1 ! parent
""",
                encoding="utf-8",
            )
            diseases = {
                "D": [
                    DiseaseTerm("D", "D", "HP:1"),
                    DiseaseTerm("D", "D", "HP:2"),
                    DiseaseTerm("D", "D", "HP:3"),
                ]
            }
            reduced, audit = reduce_hpo_profiles(diseases, HpoOntology(ontology_path))
            self.assertEqual({term.hpo_id for term in reduced["D"]}, {"HP:2", "HP:3"})
            removed = [row for row in audit if row["status"] == "ancestor_removed"]
            self.assertEqual(removed[0]["original_hpo_id"], "HP:1")


class BinaryEnrichmentTests(unittest.TestCase):
    def test_hub_gene_is_penalized_and_hpo_support_is_deduplicated(self) -> None:
        edges = [
            edge("HP:1", "C1", "GO:1"),
            edge("HP:1", "C2", "GO:2"),
            edge("HP:2", "C3", "GO:A"),
            edge("HP:3", "C4", "GO:A"),
            edge("HP:4", "C5", "GO:A"),
            edge("HP:5", "C6", "GO:A"),
        ]
        graph = BinaryCpoGraph(
            "strict",
            "v1",
            tuple(edges),
            {
                "GO:1": {"HUB", "SPECIFIC"},
                "GO:2": {"HUB", "SPECIFIC"},
                "GO:A": {"HUB"},
            },
            {("CL:1", "MARKER_ONLY")},
        )
        diseases = {"D": [DiseaseTerm("D", "D", "HP:1")]}
        base = {
            ("D", gene): {
                "disease_name": "D",
                "base_rank": rank,
                "base_dense_rank": rank,
                "base_score": 1.0 / rank,
                "best_model_id": gene,
                "best_hpo_id": "HP:1",
                "best_mpo_id": "MP:1",
            }
            for rank, gene in enumerate(["HUB", "SPECIFIC", "MARKER_ONLY"], 1)
        }
        rows, _ = score_graph(
            diseases,
            {"HUB", "SPECIFIC", "MARKER_ONLY"},
            graph,
            {"HP:1", "HP:2", "HP:3", "HP:4", "HP:5"},
            base,
        )
        by_gene = {row["gene_symbol"]: row for row in rows}
        self.assertEqual(by_gene["SPECIFIC"]["supported_hpo_count"], 1)
        self.assertEqual(by_gene["SPECIFIC"]["raw_function_path_count"], 2)
        self.assertLess(
            by_gene["SPECIFIC"]["cpo_enrichment_pvalue"],
            by_gene["HUB"]["cpo_enrichment_pvalue"],
        )
        self.assertEqual(by_gene["MARKER_ONLY"]["supported_hpo_count"], 0)

    def test_bridge_loo_removes_only_uniquely_supported_edges(self) -> None:
        graph = BinaryCpoGraph(
            "strict",
            "v1",
            (
                edge("HP:1", "C1", "GO:1", ("TARGET",)),
                edge("HP:2", "C2", "GO:2", ("TARGET", "OTHER")),
                edge("HP:3", "C3", "GO:3", ()),
            ),
            {"GO:1": {"TARGET"}, "GO:2": {"TARGET"}, "GO:3": {"TARGET"}},
            set(),
        )
        filtered = graph.bridge_loo({"TARGET"})
        self.assertEqual({item.cpo_id for item in filtered.edges}, {"C2", "C3"})


class ConstrainedRerankingTests(unittest.TestCase):
    @staticmethod
    def rows(with_cpo: bool = True) -> list[dict[str, object]]:
        output = []
        for rank, gene in enumerate(["A", "B", "C"], 1):
            pvalue = {"A": 0.5, "B": 0.2, "C": 0.01}[gene] if with_cpo else 1.0
            support = 1 if with_cpo else 0
            output.append(
                {
                    "disease_id": "D",
                    "disease_name": "D",
                    "gene_symbol": gene,
                    "base_rank": rank,
                    "base_dense_rank": rank,
                    "base_score": 1.0 / rank,
                    "best_model_id": gene,
                    "best_hpo_id": "HP:1",
                    "best_mpo_id": "MP:1",
                    "cpo_graph": "strict",
                    "cpo_graph_version": "v1",
                    "background_hpo_count": 10,
                    "disease_hpo_count": 1,
                    "cpo_mappable_hpo_count": support,
                    "cpo_coverage": float(support),
                    "supported_hpo_count": support,
                    "supported_hpo_fraction": float(support),
                    "global_reachable_hpo_count": support,
                    "supported_cpo_count": support,
                    "supported_cell_count": support,
                    "supported_function_count": support,
                    "raw_function_path_count": support,
                    "raw_marker_path_count": 0,
                    "concordant_cpo_count": 0,
                    "cpo_enrichment_log_p": 0.0,
                    "cpo_enrichment_pvalue": pvalue,
                    "cpo_enrichment_fdr": pvalue,
                    "has_cpo_evidence": bool(support),
                    "predicted_cells": [],
                    "predicted_functions": [],
                    "top_cpo_paths": [],
                }
            )
        return output

    def test_stable_constraint_cannot_be_violated(self) -> None:
        constraints = [
            {"disease_id": "D", "higher_gene": "A", "lower_gene": "C"}
        ]
        rows, summary = rerank_constrained(
            self.rows(), constraints, "A3_strict_constrained"
        )
        ranks = {row["gene_symbol"]: row["final_rank"] for row in rows}
        self.assertLess(ranks["A"], ranks["C"])
        self.assertEqual(summary[0]["constraint_violations"], 0)

    def test_no_cpo_degrades_to_base_order(self) -> None:
        rows = rerank_unconstrained(self.rows(False), "enrichment", "A2")
        self.assertEqual([row["gene_symbol"] for row in rows], ["A", "B", "C"])


class DictProvider(SimilarityProvider):
    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores

    def get_score(self, hpo_id: str, mpo_id: str) -> float:
        return self.scores.get((hpo_id, mpo_id), 0.0)

    def get_evidence(self, hpo_id: str, mpo_id: str) -> dict[str, object]:
        return {"mapped": (hpo_id, mpo_id) in self.scores}


class PhenotypeStabilityTests(unittest.TestCase):
    def test_consistent_loo_dominance_creates_constraint(self) -> None:
        diseases = {
            "D": [
                DiseaseTerm("D", "D", "HP:1", weight=1.0),
                DiseaseTerm("D", "D", "HP:2", weight=1.0),
            ]
        }
        models = {
            "MA": MouseModel("MA", "A", "A", "A"),
            "MB": MouseModel("MB", "B", "B", "B"),
        }
        phenotypes = {
            "MA": [ModelPhenotype("MA", "MP:1")],
            "MB": [ModelPhenotype("MB", "MP:2")],
        }
        provider = DictProvider(
            {
                ("HP:1", "MP:1"): 0.9,
                ("HP:2", "MP:1"): 0.8,
                ("HP:1", "MP:2"): 0.5,
                ("HP:2", "MP:2"): 0.4,
            }
        )
        _, _, constraints, summary = calculate_phenotype_stability(
            diseases,
            models,
            phenotypes,
            provider,
            {"reverse_match_weight": 1.0, "bma_weight": 0.8, "max_pair_weight": 0.2},
        )
        self.assertTrue(
            any(
                row["higher_gene"] == "A" and row["lower_gene"] == "B"
                for row in constraints
            )
        )
        self.assertEqual(summary[0]["valid_loo_count"], 2)

    def test_exomiser_full_and_loo_stability(self) -> None:
        diseases = {
            "D": [
                DiseaseTerm("D", "D", "HP:1"),
                DiseaseTerm("D", "D", "HP:2"),
            ]
        }
        models = {
            "MA": MouseModel("MA", "A", "A", "A"),
            "MB": MouseModel("MB", "B", "B", "B"),
        }
        phenotypes = {
            "MA": [ModelPhenotype("MA", "MP:1")],
            "MB": [ModelPhenotype("MB", "MP:2")],
        }

        class Matrix:
            best_by_hpo = {"HP:1": (1.0, "MP:1"), "HP:2": (1.0, "MP:1")}
            values = {
                ("HP:1", "MP:1"): 1.0,
                ("HP:2", "MP:1"): 0.9,
                ("HP:1", "MP:2"): 0.5,
                ("HP:2", "MP:2"): 0.4,
            }

            def get(self, hp_id: str, mp_id: str) -> float:
                return self.values.get((hp_id, mp_id), 0.0)

        base, loo, constraints, summary = calculate_exomiser_phenotype_stability(
            diseases, models, phenotypes, Matrix()
        )
        ranks = {row["gene_symbol"]: row["base_rank"] for row in base}
        self.assertLess(ranks["A"], ranks["B"])
        self.assertEqual(len(loo), 4)
        self.assertTrue(
            any(row["higher_gene"] == "A" and row["lower_gene"] == "B" for row in constraints)
        )
        self.assertEqual(summary[0]["valid_loo_count"], 2)


if __name__ == "__main__":
    unittest.main()
