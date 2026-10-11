from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "marker_statistics"
sys.path.insert(0, str(SCRIPT_DIR))

import build_marker_review as review  # noqa: E402


class MarkerReviewTests(unittest.TestCase):
    def write_tsv(self, path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_review_data_combines_markers_full_metrics_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            top = root / "top.tsv"
            full = root / "full.tsv"
            mapping = root / "mapping.tsv"
            marker_fields = [
                "catalog_order",
                "catalog_id",
                "catalog_label",
                "catalog_cl_name",
                "catalog_label_zh",
                "parent_id",
                "parent_label",
                "branch",
                "catalog_rank",
                "marker_rank",
                "gene",
                "marker_score",
                "statistics_score",
                "signatures_score",
            ]
            common = {
                "catalog_order": 2,
                "catalog_id": "CL:1",
                "catalog_label": "Test cell",
                "catalog_cl_name": "test cell",
                "catalog_label_zh": "测试细胞",
                "parent_id": "CL:0",
                "parent_label": "Cell",
                "branch": "test",
                "catalog_rank": "type",
            }
            self.write_tsv(
                top,
                marker_fields,
                [
                    {**common, "marker_rank": 1, "gene": "G1", "marker_score": 3, "statistics_score": 2, "signatures_score": 1},
                    {**common, "marker_rank": 2, "gene": "G2", "marker_score": 1, "statistics_score": 1, "signatures_score": 0},
                ],
            )
            self.write_tsv(
                full,
                marker_fields,
                [
                    {**common, "marker_rank": 1, "gene": "G1", "marker_score": 3, "statistics_score": 2, "signatures_score": 1},
                    {**common, "marker_rank": 2, "gene": "G2", "marker_score": 1, "statistics_score": 1, "signatures_score": 0},
                    {**common, "marker_rank": 3, "gene": "G3", "marker_score": 5, "statistics_score": 4, "signatures_score": 1},
                ],
            )
            self.write_tsv(
                mapping,
                ["cell_name", "catalog_id", "similarity_score", "paper_count", "source_types"],
                [
                    {"cell_name": "Rare alias", "catalog_id": "CL:1", "similarity_score": 0.9, "paper_count": 1, "source_types": "marker_statistics"},
                    {"cell_name": "Common alias", "catalog_id": "CL:1", "similarity_score": 0.8, "paper_count": 10, "source_types": "both"},
                ],
            )

            data = review.build_review_data(
                review.load_top_markers(top),
                review.load_full_metrics(full),
                review.load_aliases(mapping),
            )

            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["markerCount"], 3)
            self.assertEqual(data[0]["totalPoints"], 9)
            self.assertEqual(data[0]["statisticsPoints"], 7)
            self.assertEqual(data[0]["signaturePoints"], 2)
            self.assertEqual([item["gene"] for item in data[0]["markers"]], ["G1", "G2"])
            self.assertEqual([item["name"] for item in data[0]["aliases"]], ["Common alias", "Rare alias"])

            output = root / "review.html"
            review.write_review_html(
                output,
                data,
                {"scores": {"standard_cells_with_markers": 1, "marker_pairs": 3, "total_points": 9, "statistics_points": 7, "signatures_points": 2}},
            )
            document = output.read_text(encoding="utf-8")
            self.assertIn("Test cell", document)
            self.assertIn("Common alias", document)
            self.assertNotIn("__DATA__", document)
            self.assertNotIn("__META__", document)

    def test_aliases_accept_adjudicated_mapping_and_skip_unaccepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "mapping.tsv"
            self.write_tsv(
                mapping,
                [
                    "cell_name",
                    "catalog_id",
                    "top_score",
                    "paper_count",
                    "source_types",
                    "decision",
                ],
                [
                    {
                        "cell_name": "Accepted exact",
                        "catalog_id": "CL:1",
                        "top_score": 1.0,
                        "paper_count": 2,
                        "source_types": "both",
                        "decision": "exact_auto",
                    },
                    {
                        "cell_name": "Accepted semantic",
                        "catalog_id": "CL:1",
                        "top_score": 0.75,
                        "paper_count": 1,
                        "source_types": "marker_statistics",
                        "decision": "match",
                    },
                    {
                        "cell_name": "Needs review",
                        "catalog_id": "",
                        "top_score": 0.8,
                        "paper_count": 4,
                        "source_types": "both",
                        "decision": "review",
                    },
                ],
            )

            aliases = review.load_aliases(mapping)

        self.assertEqual(
            [item["name"] for item in aliases["CL:1"]],
            ["Accepted exact", "Accepted semantic"],
        )
        self.assertEqual(aliases["CL:1"][0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
