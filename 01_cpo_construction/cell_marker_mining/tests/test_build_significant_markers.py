from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "marker_statistics"
sys.path.insert(0, str(SCRIPT_DIR))

import build_significant_markers as significant  # noqa: E402


class SignificantMarkerTests(unittest.TestCase):
    def marker_row(self, gene: str, score: int, rank: int) -> dict[str, str]:
        return {
            "catalog_order": "1",
            "catalog_id": "CL:1",
            "catalog_label": "Test cell",
            "catalog_cl_name": "test cell",
            "catalog_label_zh": "",
            "parent_id": "CL:0",
            "parent_label": "Cell",
            "branch": "test",
            "catalog_rank": "type",
            "marker_rank": str(rank),
            "gene": gene,
            "marker_score": str(score),
            "statistics_score": str(score),
            "signatures_score": "0",
        }

    def test_bh_adjustment_includes_unobserved_gene_hypotheses(self) -> None:
        adjusted = significant.bh_adjust_observed(np.asarray([0.001, 0.01]), 4)
        np.testing.assert_allclose(adjusted, np.asarray([0.004, 0.02]))

    def test_enrichment_retains_repeated_specific_gene_only(self) -> None:
        rows = [
            self.marker_row("SPECIFIC", 10, 1),
            self.marker_row("COMMON", 10, 2),
            self.marker_row("RARE", 2, 3),
        ]
        gene_totals = Counter({"SPECIFIC": 10, "COMMON": 100, "RARE": 2})

        selected, metrics = significant.analyze_cell_rows(
            rows,
            gene_totals,
            total_opportunities=200,
            cell_opportunities=20,
            q_threshold=0.05,
            min_score=3,
            min_odds_ratio=1.0,
        )

        self.assertEqual([row["gene"] for row in selected], ["SPECIFIC"])
        self.assertLess(selected[0]["q_value"], 0.05)
        self.assertGreater(selected[0]["odds_ratio"], 1.0)
        self.assertEqual(selected[0]["cell_prevalence"], 0.5)
        self.assertEqual(metrics["tested_observed_genes"], 3)
        self.assertEqual(metrics["significant_genes"], 1)

    def test_invalid_score_above_cell_opportunities_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            significant.analyze_cell_rows(
                [self.marker_row("G1", 6, 1)],
                Counter({"G1": 6}),
                total_opportunities=20,
                cell_opportunities=5,
                q_threshold=0.05,
                min_score=1,
                min_odds_ratio=1.0,
            )


if __name__ == "__main__":
    unittest.main()
