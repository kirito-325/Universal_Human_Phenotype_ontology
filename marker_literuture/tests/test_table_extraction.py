import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from literature_pipeline.assets import extract_tables_for_record
from literature_pipeline.cli import (
    has_complete_standardization,
    has_complete_table_extraction,
)
from literature_pipeline.utils import sha256_file


class TableExtractionTests(unittest.TestCase):
    def make_record(self, root: Path, filename: str, content: bytes) -> Path:
        record = root / "paper"
        assets = record / "assets"
        assets.mkdir(parents=True)
        path = assets / filename
        path.write_bytes(content)
        manifest = [
            {
                "filename": filename,
                "local_path": f"assets/{filename}",
                "download_status": "downloaded",
                "sha256": sha256_file(path),
            }
        ]
        (assets / "assets_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return record

    def test_extracts_tab_delimited_text_mislabeled_as_xls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = self.make_record(
                Path(temporary), "markers.xls", b"cell\tgene\nT cell\tCD3D\n"
            )
            stats = extract_tables_for_record(record)
            manifest = json.loads(
                (record / "extracted/tables/tables_manifest.json").read_text()
            )
            complete = has_complete_table_extraction(record)

        self.assertEqual(stats["tables"], 1)
        self.assertEqual(manifest[0]["table_type"], "mislabeled_delimited_text")
        self.assertTrue(complete)

    def test_extracts_csv_member_from_zip_wrapped_csv_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "wrapped.csv"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("SupplementaryTable.csv", "cell,gene\nB cell,CD79A\n")
            record = self.make_record(root, "wrapped.csv", archive_path.read_bytes())
            stats = extract_tables_for_record(record)
            manifest = json.loads(
                (record / "extracted/tables/tables_manifest.json").read_text()
            )

        self.assertEqual(stats["tables"], 1)
        self.assertEqual(manifest[0]["table_type"], "zip_delimited_text")
        self.assertEqual(manifest[0]["sheet_name"], "SupplementaryTable.csv")

    def test_extracts_xlsx_archive_mislabeled_as_csv(self) -> None:
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook_path = root / "source.xlsx"
            workbook = Workbook()
            workbook.active.append(["cell", "gene"])
            workbook.active.append(["astrocyte", "GFAP"])
            workbook.save(workbook_path)
            workbook.close()
            record = self.make_record(root, "markers.csv", workbook_path.read_bytes())
            stats = extract_tables_for_record(record)
            manifest = json.loads(
                (record / "extracted/tables/tables_manifest.json").read_text()
            )

        self.assertEqual(stats["tables"], 1)
        self.assertEqual(manifest[0]["table_type"], "xlsx_sheet")

    def test_complete_standardization_requires_both_declared_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary) / "paper"
            output_dir = record / "standardized"
            output_dir.mkdir(parents=True)
            summary = {
                "outputs": {
                    "statistics": "standardized/marker_statistics.csv",
                    "signatures": "standardized/marker_signatures.csv",
                }
            }
            (output_dir / "standardization_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (output_dir / "marker_statistics.csv").write_text("", encoding="utf-8")
            self.assertFalse(has_complete_standardization(record, "standardized"))
            (output_dir / "marker_signatures.csv").write_text("", encoding="utf-8")
            self.assertTrue(has_complete_standardization(record, "standardized"))


if __name__ == "__main__":
    unittest.main()
