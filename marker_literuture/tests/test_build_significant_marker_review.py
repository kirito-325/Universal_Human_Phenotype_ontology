from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "marker_statistics"
sys.path.insert(0, str(SCRIPT_DIR))

import build_significant_marker_review as review  # noqa: E402


class SignificantMarkerReviewTests(unittest.TestCase):
    def write_tsv(self, path: Path, fields: list[str], row: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)

    def test_build_data_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells_path = root / "cells.tsv"
            markers_path = root / "markers.tsv"
            cell = {
                "catalog_order": 1,
                "catalog_id": "CL:1",
                "catalog_label": "Test cell",
                "catalog_cl_name": "test cell",
                "catalog_label_zh": "测试细胞",
                "parent_label": "Cell",
                "branch": "test",
                "catalog_rank": "type",
                "cell_evidence_opportunities": 20,
                "tested_observed_genes": 100,
                "significant_genes": 1,
                "significant_fraction": 0.01,
                "max_marker_score": 10,
                "max_significance_score": 8,
            }
            self.write_tsv(cells_path, list(cell), cell)
            marker = {
                "catalog_id": "CL:1",
                "significant_rank": 1,
                "original_marker_rank": 2,
                "gene": "G1",
                "marker_score": 10,
                "statistics_score": 7,
                "signatures_score": 3,
                "cell_prevalence": 0.5,
                "background_prevalence": 0.01,
                "enrichment_fold": 50,
                "odds_ratio": 80,
                "p_value": "1e-10",
                "q_value": "1e-8",
                "significance_score": 8,
            }
            self.write_tsv(markers_path, list(marker), marker)

            data = review.build_data(cells_path, markers_path)

            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["markers"][0]["gene"], "G1")
            self.assertEqual(data[0]["markers"][0]["fold"], 50)
            output = root / "review.html"
            review.write_html(output, data, {"significant_marker_pairs": 1})
            document = output.read_text(encoding="utf-8")
            self.assertIn("Test cell", document)
            self.assertIn("G1", document)
            self.assertNotIn("__DATA__", document)


if __name__ == "__main__":
    unittest.main()
