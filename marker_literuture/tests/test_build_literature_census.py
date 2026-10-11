import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_literature_census import error_category, record_flags


class LiteratureCensusTests(unittest.TestCase):
    def test_error_category_reports_publisher_access_failures(self):
        self.assertEqual(
            error_category(["publisher_html: HTTPError: HTTP Error 403: Forbidden"]),
            "publisher_http_403",
        )

    def test_record_flags_separate_tabular_candidates_from_verified_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            record_dir = Path(temporary)
            (record_dir / "assets").mkdir()
            local_table = record_dir / "assets" / "supplement.xlsx"
            local_table.write_bytes(b"placeholder")
            metadata = {
                "assets": [
                    {
                        "filename": "supplement.xlsx",
                        "local_path": "assets/supplement.xlsx",
                        "download_status": "downloaded",
                    },
                    {"filename": "archive.zip", "local_path": ""},
                ],
                "content_probe": {"errors": []},
            }
            agent_dir = record_dir / "agent_standardized"
            agent_dir.mkdir()
            (agent_dir / "agent_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "target_tables": 1,
                        "statistics_rows": 3,
                        "signature_cell_gene_pairs": 0,
                    }
                ),
                encoding="utf-8",
            )
            (agent_dir / "marker_statistics_filter_summary.json").write_text(
                json.dumps({"kept_rows": 2}), encoding="utf-8"
            )

            flags = record_flags(record_dir, metadata)

            self.assertEqual(flags["direct_tabular_supplement_papers"], 1)
            self.assertEqual(flags["direct_tabular_assets"], 1)
            self.assertEqual(flags["broad_tabular_candidate_assets"], 2)
            self.assertEqual(flags["downloaded_direct_tabular_papers"], 1)
            self.assertEqual(flags["complete_direct_tabular_papers"], 1)
            self.assertEqual(flags["partial_direct_tabular_papers"], 0)
            self.assertEqual(flags["downloaded_direct_tabular_assets"], 1)
            self.assertEqual(flags["failed_direct_tabular_assets"], 0)
            self.assertEqual(flags["pending_direct_tabular_assets"], 0)
            self.assertEqual(flags["agent_target_table_papers"], 1)
            self.assertEqual(flags["marker_positive_papers"], 1)
            self.assertEqual(flags["post_filter_marker_positive_papers"], 1)


if __name__ == "__main__":
    unittest.main()
