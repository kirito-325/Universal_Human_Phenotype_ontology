from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import filter_marker_statistics as marker_filter  # noqa: E402


class MarkerStatisticsFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = marker_filter.build_parser().parse_args([])

    def row(self, **updates: str) -> dict[str, str]:
        row = {
            "cell_cluster": "B cell",
            "gene": "CD79A",
            "p_val": "",
            "avg_logFC": "0.5",
            "pct.1": "0.7",
            "pct.2": "0.2",
            "p_val_adj": "0.01",
        }
        row.update(updates)
        return row

    def decide(self, mode: str = "pvalue", **updates: str) -> tuple[bool, str, str]:
        return marker_filter.decide_row(self.row(**updates), self.args, mode)

    def test_keeps_positive_significant_marker(self) -> None:
        self.assertEqual(self.decide(), (True, "kept", "adjusted_p"))

    def test_rejects_nonpositive_logfc(self) -> None:
        self.assertEqual(self.decide(avg_logFC="0")[1], "logfc_not_above_threshold")
        self.assertEqual(self.decide(avg_logFC="-0.2")[1], "logfc_not_above_threshold")

    def test_uses_raw_p_when_adjusted_p_is_missing(self) -> None:
        result = self.decide(p_val_adj="", p_val="0.03")
        self.assertEqual(result, (True, "kept", "raw_p"))
        self.assertEqual(self.decide(p_val_adj="", p_val="0.2")[1], "raw_p_above_threshold")

    def test_missing_significance_is_optional_by_default(self) -> None:
        self.assertEqual(self.decide(p_val_adj="", p_val=""), (True, "kept", "missing"))
        self.args.require_significance = True
        self.assertEqual(self.decide(p_val_adj="", p_val="")[1], "missing_significance")

    def test_checks_detection_fraction_direction(self) -> None:
        self.assertEqual(self.decide(**{"pct.1": "25", "pct.2": "10"})[0], True)
        self.assertEqual(
            self.decide(**{"pct.1": "0.1", "pct.2": "0.2"})[1],
            "pct1_not_above_pct2",
        )

    def test_neglog10_fdr_threshold_uses_opposite_direction(self) -> None:
        self.assertEqual(self.decide(mode="neglog10", p_val_adj="1.4")[0], True)
        self.assertEqual(
            self.decide(mode="neglog10", p_val_adj="1.2")[1],
            "neglog10_fdr_below_threshold",
        )

    def test_mixed_mode_falls_back_to_raw_p(self) -> None:
        self.assertEqual(
            self.decide(mode="mixed", p_val_adj="9.0", p_val="0.02"),
            (True, "kept", "raw_p"),
        )

    def test_detects_neglog10_fdr_from_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "marker_statistics.csv"
            source.touch()
            analysis = {
                "table_decisions": [
                    {
                        "reason": "Marker genes with -Log10(FDR)",
                        "evidence": ["Header: Gene, -Log10(FDR), Log2(FC)"],
                        "extractions": [{"columns": {"p_val_adj": 2}}],
                    }
                ]
            }
            (folder / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
            mode, evidence = marker_filter.detect_adjusted_p_mode(source)
            self.assertEqual(mode, "neglog10")
            self.assertEqual(evidence["neg_log10_fdr_tables"], 1)

    def test_process_file_writes_filtered_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "marker_statistics.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(marker_filter.REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerow(self.row())
                writer.writerow(self.row(gene="MS4A1", avg_logFC="-0.5"))
            result = marker_filter.process_file(source, self.args)
            self.assertEqual(result["kept_rows"], 1)
            output = folder / "marker_statistics_filtered.csv"
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["gene"] for row in rows], ["CD79A"])


if __name__ == "__main__":
    unittest.main()
