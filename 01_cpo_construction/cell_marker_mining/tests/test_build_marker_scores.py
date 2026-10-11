from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "marker_statistics"
sys.path.insert(0, str(SCRIPT_DIR))

import build_marker_scores as marker_scores  # noqa: E402


class MarkerScoreTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[list[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            delimiter = "\t" if path.suffix == ".tsv" else ","
            csv.writer(handle, delimiter=delimiter).writerows(rows)

    def make_paper(self, papers: Path, folder: str, paper_id: str) -> Path:
        record = papers / folder
        record.mkdir(parents=True)
        (record / "metadata.json").write_text(json.dumps({"id": paper_id}), encoding="utf-8")
        return record / "agent_standardized"

    def test_one_point_per_paper_table_after_standard_cell_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            papers = Path(temporary) / "papers"
            first = self.make_paper(papers, "paper_1", "doi:first")
            self.write_csv(
                first / "marker_statistics.csv",
                [
                    ["cell_cluster", "gene"],
                    ["Alias A", "RAW_ONLY"],
                ],
            )
            self.write_csv(
                first / marker_scores.STATISTICS_NAME,
                [
                    ["cell_cluster", "gene"],
                    ["Alias A", "G1"],
                    ["Alias A", "G1"],
                    ["Alias B", "G1"],
                    ["Alias A", "G2"],
                    ["Excluded name", "G9"],
                ],
            )
            self.write_csv(
                first / marker_scores.SIGNATURES_NAME,
                [
                    ["cell_cluster", "gene"],
                    ["Alias A", "G1"],
                    ["Alias B", "G1"],
                    ["Excluded name", "G9"],
                    ["Alias A", "G1"],
                    ["Alias A", "G3"],
                    ["Alias B", "G3"],
                ],
            )

            second = self.make_paper(papers, "paper_2", "doi:second")
            self.write_csv(
                second / marker_scores.STATISTICS_NAME,
                [
                    ["cell_cluster", "gene"],
                    ["Alias A", "G1"],
                ],
            )

            mappings = {"Alias A": "CL:1", "Alias B": "CL:1"}
            result = marker_scores.collect(papers, mappings)

            self.assertEqual(result.statistics_scores[("CL:1", "G1")], 2)
            self.assertEqual(result.signature_scores[("CL:1", "G1")], 1)
            self.assertEqual(result.statistics_scores[("CL:1", "G2")], 1)
            self.assertEqual(result.signature_scores[("CL:1", "G3")], 1)
            self.assertNotIn(("CL:1", "RAW_ONLY"), result.statistics_scores)
            self.assertEqual(result.unmapped_occurrences["statistics"]["Excluded name"], 1)
            self.assertEqual(result.unmapped_occurrences["signatures"]["Excluded name"], 1)
            self.assertEqual(result.table_metrics["statistics"]["awarded_points"], 3)
            self.assertEqual(result.table_metrics["signatures"]["awarded_points"], 2)
            self.assertEqual(result.cell_opportunities["statistics"]["CL:1"], 2)
            self.assertEqual(result.cell_opportunities["signatures"]["CL:1"], 1)

    def test_score_rows_add_two_table_scores_and_rank_within_cell(self) -> None:
        result = marker_scores.CollectionResult()
        result.statistics_scores.update({("CL:1", "G1"): 2, ("CL:1", "G2"): 1})
        result.signature_scores.update({("CL:1", "G1"): 1, ("CL:1", "G3"): 2})
        result.cell_opportunities["statistics"]["CL:1"] = 2
        result.cell_opportunities["signatures"]["CL:1"] = 1
        catalog = {
            "CL:1": marker_scores.CatalogCell(
                order=7,
                catalog_id="CL:1",
                cl_name="standard cell",
                label="Standard cell",
                label_zh="标准细胞",
                parent_id="CL:0",
                parent_label="Cell",
                branch="test",
                rank="type",
            )
        }

        rows = list(marker_scores.iter_score_rows(result, catalog))

        self.assertEqual([row["gene"] for row in rows], ["G1", "G3", "G2"])
        self.assertEqual([row["marker_score"] for row in rows], [3, 2, 1])
        self.assertEqual([row["marker_rank"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["statistics_score"], 2)
        self.assertEqual(rows[0]["signatures_score"], 1)

        with tempfile.TemporaryDirectory() as temporary:
            full_path = Path(temporary) / "full.tsv"
            top_path = Path(temporary) / "top.tsv"
            summary = marker_scores.write_score_table(full_path, top_path, 2, result, catalog)
            with top_path.open(encoding="utf-8", newline="") as handle:
                top_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(summary["marker_pairs"], 3)
            self.assertEqual(summary["top_marker_pairs"], 2)
            self.assertEqual([row["gene"] for row in top_rows], ["G1", "G3"])

            opportunities_path = Path(temporary) / "opportunities.tsv"
            opportunity_summary = marker_scores.write_opportunity_table(
                opportunities_path, result, catalog
            )
            with opportunities_path.open(encoding="utf-8", newline="") as handle:
                opportunity_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(opportunity_summary["total_cell_opportunities"], 3)
            self.assertEqual(opportunity_rows[0]["statistics_opportunities"], "2")
            self.assertEqual(opportunity_rows[0]["signatures_opportunities"], "1")
            self.assertEqual(opportunity_rows[0]["total_opportunities"], "3")

    def test_duplicate_paper_id_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            papers = Path(temporary) / "papers"
            for folder in ("copy_a", "copy_b"):
                output = self.make_paper(papers, folder, "doi:same")
                self.write_csv(
                    output / marker_scores.STATISTICS_NAME,
                    [["cell_cluster", "gene"], ["Cell", "G1"]],
                )

            result = marker_scores.collect(papers, {"Cell": "CL:1"})

            self.assertEqual(result.statistics_scores[("CL:1", "G1")], 1)
            self.assertEqual(result.papers_processed, 1)
            self.assertEqual(result.duplicate_paper_directories_skipped, 1)

    def test_adjudicated_mapping_accepts_only_confirmed_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.tsv"
            path.write_text(
                "cell_name\tcatalog_id\tdecision\n"
                "Exact cell\tCL:1\texact_auto\n"
                "Confirmed cell\tCL:1\tmatch\n"
                "Pending cell\tCL:1\tpending_llm\n"
                "Rejected cell\t\tno_match\n",
                encoding="utf-8",
            )
            catalog = {
                "CL:1": marker_scores.CatalogCell(
                    order=1,
                    catalog_id="CL:1",
                    cl_name="cell",
                    label="Cell",
                    label_zh="",
                    parent_id="",
                    parent_label="",
                    branch="test",
                    rank="type",
                )
            }
            mappings, summary = marker_scores.load_top1_mapping(
                path, catalog, require_adjudicated=True
            )

        self.assertEqual(mappings, {"Exact cell": "CL:1", "Confirmed cell": "CL:1"})
        self.assertTrue(summary["mapping_has_decisions"])
        self.assertEqual(summary["mapped_source_names"], 2)
        self.assertEqual(summary["mapping_decision_counts"]["pending_llm"], 1)

    def test_adjudication_can_be_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.tsv"
            path.write_text("cell_name\tcatalog_id\nCell\tCL:1\n", encoding="utf-8")
            catalog = {
                "CL:1": marker_scores.CatalogCell(
                    1, "CL:1", "cell", "Cell", "", "", "", "test", "type"
                )
            }
            with self.assertRaisesRegex(ValueError, "decision column is absent"):
                marker_scores.load_top1_mapping(
                    path, catalog, require_adjudicated=True
                )


if __name__ == "__main__":
    unittest.main()
