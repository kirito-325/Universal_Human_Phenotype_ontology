from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import MappingExample, MechanismEdge  # noqa: E402
from mpo_hpo_mapping.embedding import ProfileTensor, build_profile_tensor  # noqa: E402
from mpo_hpo_mapping.model import (  # noqa: E402
    DualHeadEncoder,
    balanced_parent_sample,
    fit_relation_calibrator,
    mask_pair_features,
    poincare_profile_score,
    train_dual_head,
)


def profile(ids: list[str], offset: float = 0.0) -> ProfileTensor:
    prototypes = np.zeros((len(ids), 2, 6), dtype=np.float32)
    masks = np.zeros((len(ids), 2), dtype=bool)
    semantic = np.zeros((len(ids), 4), dtype=np.float32)
    for index in range(len(ids)):
        value = 0.05 + 0.04 * index + offset
        prototypes[index, 0] = [value, 0.0, 0.0, value, 1.0, 0.0]
        masks[index, 0] = True
        semantic[index, index % 4] = 1.0
    return ProfileTensor(
        ids=ids,
        names=ids,
        prototypes=prototypes,
        masks=masks,
        edge_counts=np.ones(len(ids), dtype=np.int32),
        specific_counts=np.ones(len(ids), dtype=np.int32),
        semantic_vectors=semantic,
        aliases={term_id: {term_id.lower()} for term_id in ids},
        mechanism_ids={term_id: {f"M:{index}"} for index, term_id in enumerate(ids)},
        specific_mechanism_ids={
            term_id: {f"M:{index}"} for index, term_id in enumerate(ids)
        },
    )


class LateFusionTest(unittest.TestCase):
    def test_profile_keeps_structure_and_semantic_separate(self) -> None:
        edges = {
            "HP:1": [
                MechanismEdge(
                    "HP:1",
                    "term",
                    "CL:1",
                    "cell",
                    "GO:1",
                    "process",
                    "specific",
                    0.9,
                    1,
                    "high",
                    "GENE",
                )
            ]
        }
        result = build_profile_tensor(
            edges,
            {"CL:1": np.asarray([0.1, 0.0], dtype=np.float32)},
            {"GO:1": np.asarray([0.0, 0.1], dtype=np.float32)},
            max_prototypes=2,
            semantic_vectors={"HP:1": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
        )
        self.assertEqual(result.prototypes.shape, (1, 2, 6))
        self.assertEqual(result.semantic_vectors.shape, (1, 3))
        np.testing.assert_allclose(result.prototypes[0, 0, -2:], [1.0, 0.0])

    def test_poincare_score_prefers_identical_mechanisms(self) -> None:
        hp = profile(["HP:1", "HP:2"])
        identical = poincare_profile_score(hp, 0, hp, 0, hierarchy_dimension=2)
        different = poincare_profile_score(hp, 0, hp, 1, hierarchy_dimension=2)
        self.assertGreater(identical, different)

    def test_fusion_weights_are_non_negative(self) -> None:
        model = DualHeadEncoder(6, 4, hidden_dimension=8, output_dimension=4, hierarchy_dimension=2)
        for head in ("exact", "ancestor"):
            weights = model.fusion_weight_dict(head)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
            self.assertTrue(all(value >= 0 for value in weights.values()))

        semantic = DualHeadEncoder(
            6, 4, hidden_dimension=8, output_dimension=4, hierarchy_dimension=2,
            fusion_mode="semantic_only",
        )
        structure = DualHeadEncoder(
            6, 4, hidden_dimension=8, output_dimension=4, hierarchy_dimension=2,
            fusion_mode="structure_only",
        )
        self.assertEqual(semantic.fusion_weight_dict("exact"), {
            "semantic": 1.0, "structure": 0.0, "poincare": 0.0
        })
        self.assertEqual(structure.fusion_weight_dict("exact")["semantic"], 0.0)
        masked = mask_pair_features(np.ones(16, dtype=np.float64), "semantic_only")
        np.testing.assert_array_equal(masked[[3, 4, 7, 8, 11, 12, 13, 14, 15]], 0.0)

    def test_source_balancing_and_training(self) -> None:
        rows = [
            MappingExample("MP:1", "HP:1", "hp_broader", 1.0, "train", source)
            for source in ("expert_broad", "counterfactual_exact", "synthetic_hpo_cutout")
            for _ in range(10)
        ]
        selected = balanced_parent_sample(rows, 9, random.Random(3))
        self.assertEqual(Counter(row.source for row in selected), {
            "counterfactual_exact": 3,
            "expert_broad": 3,
            "synthetic_hpo_cutout": 3,
        })

        hp = profile(["HP:1", "HP:2", "HP:3"])
        mp = profile(["MP:1", "MP:2", "MP:3"], offset=0.01)
        exact = [
            MappingExample(f"MP:{index}", f"HP:{index}", "exact", 1.0, "train", "manual")
            for index in range(1, 4)
        ]
        parents = [
            MappingExample("MP:1", "HP:2", "hp_broader", 1.0, "train", "expert_broad"),
            MappingExample("MP:2", "HP:3", "hp_broader", 1.0, "train", "counterfactual_exact"),
            MappingExample(
                "synthetic:HP:1",
                "HP:2",
                "hp_broader",
                1.0,
                "train",
                "synthetic_hpo_cutout",
                query_profile_source="hp",
                exact_hp_id="HP:1",
            ),
        ]
        trained = train_dual_head(
            hp,
            mp,
            exact,
            parents,
            epochs=1,
            batch_size=3,
            hidden_dimension=8,
            output_dimension=4,
            hierarchy_dimension=2,
            seed=7,
        )
        self.assertEqual(len(trained.loss_history), 1)
        self.assertIn("exact_structure_loss", trained.loss_history[0])
        self.assertEqual(trained.loss_history[0]["structure_aux_effective_weight"], 0.35)
        calibrator = fit_relation_calibrator(
            trained.model, hp, mp, exact, parents, seed=7
        )
        totals = list(calibrator.source_weight_totals.values())
        self.assertEqual(set(calibrator.source_weight_totals), {
            "manual", "expert_broad", "counterfactual_exact", "synthetic_hpo_cutout"
        })
        self.assertLess(max(totals) - min(totals), 1e-8)


if __name__ == "__main__":
    unittest.main()
