from __future__ import annotations

import unittest

from cellphenorank.phenodigm_official import (
    combined_percentage_score,
    rank_official_scores,
)
from cellphenorank.schemas import MouseModel


class OfficialPhenodigmTests(unittest.TestCase):
    def test_combined_score_is_mean_of_normalized_average_and_maximum(self) -> None:
        self.assertAlmostEqual(combined_percentage_score(40.0, 80.0), 60.0)

    def test_gene_uses_best_model_and_missing_records_score_zero(self) -> None:
        models = {
            "M1": MouseModel("M1", "model 1", "Genea", "GENEA"),
            "M2": MouseModel("M2", "model 2", "Genea", "GENEA"),
            "M3": MouseModel("M3", "model 3", "Geneb", "GENEB"),
        }
        docs = [
            {
                "model_id": "M1",
                "model_source": "MGI",
                "disease_model_avg_norm": 20.0,
                "disease_model_max_norm": 40.0,
                "association_curated": False,
            },
            {
                "model_id": "M2",
                "model_source": "MGI",
                "disease_model_avg_norm": 60.0,
                "disease_model_max_norm": 80.0,
                "association_curated": True,
            },
        ]
        model_rows, gene_rows = rank_official_scores("D", "Disease", models, docs)
        by_gene = {row["gene_symbol"]: row for row in gene_rows}
        self.assertEqual(by_gene["GENEA"]["best_model_id"], "M2")
        self.assertEqual(by_gene["GENEA"]["phenodigm_score"], 70.0)
        self.assertEqual(by_gene["GENEA"]["phenodigm_rank"], 1)
        self.assertEqual(by_gene["GENEB"]["phenodigm_score"], 0.0)
        self.assertFalse(by_gene["GENEB"]["official_model_score_available"])
        self.assertEqual(len(model_rows), 3)


if __name__ == "__main__":
    unittest.main()
