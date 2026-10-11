from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cellphenorank.baseline import score_model
from cellphenorank.rerank import rerank
from cellphenorank.schemas import DiseaseTerm, GeneSetResources, ModelPhenotype, MouseModel
from cellphenorank.similarity import EmbeddingSimilarityProvider, MatrixSimilarityProvider
from cellphenorank.single_cell import (
    _expression_match_pools,
    _module_scores,
    _sample_expression_matched_indices,
    map_prediction_to_label,
)


ROOT = Path(__file__).resolve().parents[1]


class SimilarityTests(unittest.TestCase):
    def test_matrix_provider_and_missing_pair(self) -> None:
        provider = MatrixSimilarityProvider(
            ROOT / "examples/hpo_mpo_similarity.tsv",
            {"HP:1"},
            {"MP:1", "MP:404"},
            {"score_columns": ["score"], "missing_pair_score": 0.0},
        )
        self.assertAlmostEqual(provider.get_score("HP:1", "MP:1"), 0.9)
        self.assertEqual(provider.get_score("HP:1", "MP:404"), 0.0)
        self.assertFalse(provider.get_evidence("HP:1", "MP:404")["mapped"])

    def test_embedding_provider_keeps_deterministic_top_k_cosines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "embeddings.npz"
            np.savez_compressed(
                artifact,
                hpo_ids=np.asarray(["HP:1"]),
                hpo_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
                mpo_ids=np.asarray(["MP:A", "MP:B", "MP:C"]),
                mpo_vectors=np.asarray(
                    [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
                ),
            )
            provider = EmbeddingSimilarityProvider(
                artifact,
                {"HP:1"},
                {"MP:A", "MP:B", "MP:C"},
                {"top_mpo_per_hpo": 2},
            )
            self.assertAlmostEqual(provider.get_score("HP:1", "MP:A"), 1.0)
            self.assertAlmostEqual(provider.get_score("HP:1", "MP:B"), 0.8)
            self.assertEqual(provider.get_score("HP:1", "MP:C"), 0.0)
            self.assertEqual(provider.pair_count, 2)
            self.assertEqual(
                provider.get_evidence("HP:1", "MP:B")["hpo_neighbor_rank"], 2
            )

    def test_bma_and_max_pair(self) -> None:
        provider = MatrixSimilarityProvider(
            ROOT / "examples/hpo_mpo_similarity.tsv",
            {"HP:1", "HP:2", "HP:3"},
            {"MP:1", "MP:2", "MP:3"},
            {"score_columns": ["score"]},
        )
        disease = [
            DiseaseTerm("D1", "Disease", f"HP:{index}", weight=1.0)
            for index in (1, 2, 3)
        ]
        model = MouseModel("MA", "A", "Genea", "GENEA")
        phenotypes = [
            ModelPhenotype("MA", f"MP:{index}", status="abnormal")
            for index in (1, 2, 3)
        ]
        row, evidence = score_model(
            disease,
            model,
            phenotypes,
            provider,
            {"reverse_match_weight": 1.0, "bma_weight": 0.8, "max_pair_weight": 0.2},
        )
        self.assertAlmostEqual(row["bma_score"], 0.8)
        self.assertAlmostEqual(row["max_pair_score"], 0.9)
        self.assertAlmostEqual(row["model_score"], 0.82)
        self.assertEqual(len(evidence), 3)


class RerankTests(unittest.TestCase):
    def test_missing_cpo_degrades_exactly_to_baseline(self) -> None:
        base = [
            {"disease_id": "D", "disease_name": "D", "gene_id": "A", "gene_symbol": "A", "base_score": 1.0, "base_rank": 1, "base_percentile": 1.0},
            {"disease_id": "D", "disease_name": "D", "gene_id": "B", "gene_symbol": "B", "base_score": 0.5, "base_rank": 2, "base_percentile": 0.0},
        ]
        diseases = {"D": [DiseaseTerm("D", "D", "HP:1", weight=1.0)]}
        result = rerank(
            base,
            [],
            diseases,
            {},
            GeneSetResources(),
            {"gene_set_scoring": {}, "rank_fusion": {"lambda": 1.0}, "report": {}},
        )
        self.assertEqual([row["gene_symbol"] for row in result["final_rankings"]], ["A", "B"])
        self.assertEqual([row["final_rank"] for row in result["final_rankings"]], [1, 2])


class SingleCellMappingTests(unittest.TestCase):
    def test_fine_cell_maps_to_dataset_parent(self) -> None:
        rows = [{"dataset_id": "study", "dataset_label": "neuron", "cell_id": "CL:parent"}]
        parents = {"CL:fine": {"CL:middle"}, "CL:middle": {"CL:parent"}}
        self.assertEqual(
            map_prediction_to_label("CL:fine", "study", rows, parents),
            ("neuron", "predicted_child_to_dataset_parent", 2),
        )

    def test_gene_zscore_module_and_matched_null(self) -> None:
        expression = np.asarray(
            [
                [1.0, 2.0, 0.0, 4.0, 8.0, 2.0],
                [2.0, 3.0, 0.0, 5.0, 7.0, 3.0],
                [4.0, 5.0, 0.0, 7.0, 5.0, 6.0],
                [5.0, 7.0, 0.0, 8.0, 4.0, 7.0],
            ]
        )
        scores = _module_scores(expression[:, [0, 1]], "gene_zscore")
        self.assertAlmostEqual(float(scores.mean()), 0.0)
        target = np.asarray([0, 1], dtype=int)
        pools, valid = _expression_match_pools(expression, target, bins=2)
        sampled = _sample_expression_matched_indices(
            pools, valid, target, np.random.default_rng(2026)
        )
        self.assertEqual(len(sampled), len(target))
        self.assertTrue(set(sampled).isdisjoint(set(target)))


if __name__ == "__main__":
    unittest.main()
