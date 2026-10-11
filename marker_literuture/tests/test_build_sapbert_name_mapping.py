from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_sapbert_name_mapping as mapping  # noqa: E402


class SapBertNameMappingTests(unittest.TestCase):
    def test_enriches_candidates_with_source_and_catalog_metadata(self) -> None:
        source = SimpleNamespace(
            source_index=0,
            name="B cell",
            row={"filtered_rank": "1", "original_rank": "2", "paper_count": "10", "source_types": "both"},
        )
        catalog = [
            SimpleNamespace(
                catalog_id="CL:0000236",
                label="B cell",
                cl_name="B cell",
                label_zh="B细胞",
                parent_label="Lymphocyte",
                branch="lymphoid",
                rank="type",
            )
        ]
        candidate = {
            "source_index": 0,
            "normalized_name": "b cell",
            "rank": 1,
            "catalog_id": "CL:0000236",
            "score": "0.987654",
        }
        row = mapping.enrich_candidates([candidate], [source], catalog)[0]
        self.assertEqual(row["filtered_rank"], "1")
        self.assertEqual(row["catalog_id"], "CL:0000236")
        self.assertEqual(row["catalog_label_zh"], "B细胞")

    def test_review_rows_preserve_top10_order_and_margin(self) -> None:
        rows = []
        for rank in range(1, 11):
            rows.append(
                {
                    "source_index": 0,
                    "filtered_rank": "1",
                    "original_rank": "1",
                    "cell_name": "B cell",
                    "paper_count": "10",
                    "source_types": "both",
                    "candidate_rank": rank,
                    "catalog_id": f"CL:{rank:07d}",
                    "catalog_label": f"Cell {rank}",
                    "catalog_cl_name": f"cell {rank}",
                    "catalog_label_zh": "",
                    "parent_label": "Cell",
                    "branch": "test",
                    "catalog_rank": "type",
                    "similarity_score": f"{1.0 - rank / 100:.6f}",
                }
            )
        review = mapping.review_rows(rows)[0]
        self.assertEqual(len(review["candidates"]), 10)
        self.assertEqual(review["candidates"][0]["rank"], 1)
        self.assertAlmostEqual(review["margin"], 0.01)

    def test_score_summary_reports_top1_and_margin(self) -> None:
        rows = [
            {"source_index": 0, "similarity_score": "0.9"},
            {"source_index": 0, "similarity_score": "0.8"},
            {"source_index": 1, "similarity_score": "0.7"},
            {"source_index": 1, "similarity_score": "0.69"},
        ]
        summary = mapping.score_summary(rows)
        self.assertEqual(summary["top1_score"]["max"], 0.9)
        self.assertEqual(summary["top1_top2_margin"]["min"], 0.01)


if __name__ == "__main__":
    unittest.main()
