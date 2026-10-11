from __future__ import annotations

import unittest

from cellphenorank.impc_dr20_embedding import (
    aggregate_profile_score,
    binary_ranking_metrics,
    bootstrap_binary_intervals,
)


class ImpcDr20EmbeddingTests(unittest.TestCase):
    def test_profile_aggregation_arms(self) -> None:
        self.assertEqual(aggregate_profile_score(0.5, 0.9, "symmetric_bma"), 0.5)
        self.assertEqual(
            aggregate_profile_score(0.5, 0.9, "phenodigm_equal_avg_max"), 0.7
        )
        self.assertAlmostEqual(
            aggregate_profile_score(0.5, 0.9, "project_80_bma_20_max"), 0.58
        )

    def test_binary_metrics_rank_positives_above_negatives(self) -> None:
        metrics = binary_ranking_metrics(
            [True, False, True, False], [0.9, 0.2, 0.8, 0.1]
        )
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["auprc"], 1.0)

    def test_bootstrap_intervals_are_deterministic(self) -> None:
        first = bootstrap_binary_intervals(
            [True, False, True, False], [0.9, 0.2, 0.8, 0.1], iterations=20, seed=7
        )
        second = bootstrap_binary_intervals(
            [True, False, True, False], [0.9, 0.2, 0.8, 0.1], iterations=20, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first["auroc_ci_low"], 1.0)


if __name__ == "__main__":
    unittest.main()
