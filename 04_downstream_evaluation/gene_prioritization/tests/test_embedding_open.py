from __future__ import annotations

import unittest

from cellphenorank.embedding_open import _compact_rankings


class OpenRankingMergeTests(unittest.TestCase):
    def test_top_k_is_reranked_and_tail_keeps_base_rank(self) -> None:
        base = [
            {
                "disease_id": "D",
                "disease_name": "Disease",
                "gene_symbol": gene,
                "base_score": 1.0 / rank,
                "base_rank": rank,
                "best_model_id": gene,
                "best_hpo_id": "HP:1",
                "best_mpo_id": "MP:1",
                "hpo_mpo_coverage": 1.0,
            }
            for rank, gene in enumerate(("A", "B", "C"), 1)
        ]
        top = [
            {"disease_id": "D", "gene_symbol": "B", "final_rank": 1},
            {"disease_id": "D", "gene_symbol": "A", "final_rank": 2},
        ]
        rows = _compact_rankings(base, top, {"D": {"A", "B"}}, "test")
        by_gene = {row["gene_symbol"]: row for row in rows}
        self.assertEqual(by_gene["B"]["rank"], 1)
        self.assertEqual(by_gene["A"]["rank"], 2)
        self.assertEqual(by_gene["C"]["rank"], 3)
        self.assertFalse(by_gene["C"]["in_rerank_shortlist"])


if __name__ == "__main__":
    unittest.main()
