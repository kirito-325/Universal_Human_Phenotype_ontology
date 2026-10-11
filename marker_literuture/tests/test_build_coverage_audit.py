from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "marker_statistics"))

import build_coverage_audit as coverage  # noqa: E402


class CoverageAuditTests(unittest.TestCase):
    def test_all_catalog_nodes_are_retained_and_missing_evidence_is_zero(self) -> None:
        catalog = [
            {
                "order": "1",
                "cl_id": "CL:0",
                "cl_name": "cell",
                "catalog_label": "Cell",
                "branch": "root",
                "rank": "root",
                "source": "skeleton",
            },
            {
                "order": "2",
                "cl_id": "CL:1",
                "cl_name": "alpha cell",
                "catalog_label": "Alpha cell",
                "branch": "endocrine",
                "rank": "type",
                "source": "skeleton",
            },
            {
                "order": "3",
                "cl_id": "CL:2",
                "cl_name": "beta cell",
                "catalog_label": "Beta cell",
                "branch": "endocrine",
                "rank": "cl_extension",
                "source": "cl",
                "distance_to_anchor": "1",
            },
        ]
        opportunities = [
            {
                "catalog_id": "CL:1",
                "statistics_opportunities": "2",
                "signatures_opportunities": "1",
                "total_opportunities": "3",
            }
        ]
        significant = [{"catalog_id": "CL:1", "significant_genes": "4"}]
        report = coverage.audit(catalog, opportunities, significant)

        self.assertEqual(report["totals"]["catalog_nodes"], 3)
        self.assertEqual(report["totals"]["nodes_with_any_evidence"], 1)
        self.assertEqual(report["totals"]["nodes_with_at_least_three_opportunities"], 1)
        beta = next(cell for cell in report["cells"] if cell["catalog_id"] == "CL:2")
        self.assertEqual(beta["total_opportunities"], 0)
        self.assertEqual(beta["coverage_band"], "none")
        self.assertEqual(beta["opportunities_needed_for_three"], 3)

    def test_core_gaps_exclude_root_and_extensions(self) -> None:
        catalog = [
            {"order": "1", "cl_id": "CL:0", "rank": "root", "source": "skeleton"},
            {"order": "2", "cl_id": "CL:1", "rank": "type", "source": "skeleton"},
            {"order": "3", "cl_id": "CL:2", "rank": "cl_extension", "source": "cl"},
        ]
        report = coverage.audit(catalog, [], [])
        self.assertEqual(
            [cell["catalog_id"] for cell in report["core_skeleton_gaps"]], ["CL:1"]
        )
        self.assertEqual(
            [cell["catalog_id"] for cell in report["extension_gaps"]], ["CL:2"]
        )


if __name__ == "__main__":
    unittest.main()
