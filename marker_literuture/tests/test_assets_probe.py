import io
import tempfile
import urllib.error
import unittest
from pathlib import Path

from literature_pipeline.assets import (
    download_assets_for_record,
    fetch_pmc_jats,
    load_manifest,
    validate_downloaded_asset,
    write_manifest,
)


class FallbackHttpClient:
    def __init__(self):
        self.urls = []

    def request_text(self, url):
        self.urls.append(url)
        if "oai.cgi" in url:
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO())
        return "<pmc-articleset><article><body>text</body></article></pmc-articleset>", url, "text/xml"


class DownloadHttpClient:
    def download(self, url, path, headers=None):
        path.write_bytes(b"PK\x03\x04test workbook")
        return url, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class AssetsProbeTests(unittest.TestCase):
    def test_pmc_oai_http_error_falls_back_to_efetch(self):
        http = FallbackHttpClient()

        article, status = fetch_pmc_jats(http, "PMC123")

        self.assertEqual(status, "ok_efetch_fallback")
        self.assertIn("<article>", article)
        self.assertEqual(len(http.urls), 2)
        self.assertIn("efetch.fcgi", http.urls[1])

    def test_direct_table_download_does_not_mutate_other_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            record_dir = Path(temporary)
            write_manifest(
                record_dir,
                [
                    {
                        "filename": "supplement.pdf",
                        "url": "https://example.org/supplement.pdf",
                        "download_status": "not_downloaded",
                    },
                    {
                        "filename": "markers.xlsx",
                        "url": "https://example.org/markers.xlsx",
                        "download_status": "not_downloaded",
                    },
                ],
            )

            stats = download_assets_for_record(
                record_dir, DownloadHttpClient(), extensions=[".xlsx"]
            )
            assets = load_manifest(record_dir)

        self.assertEqual(stats["candidate_count"], 1)
        self.assertEqual(stats["downloaded"], 1)
        self.assertEqual(stats["skipped_extension"], 1)
        self.assertEqual(assets[0]["download_status"], "not_downloaded")
        self.assertEqual(assets[1]["download_status"], "downloaded")

    def test_excel_validator_accepts_common_publisher_mislabels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_xlsx = root / "legacy.xlsx"
            legacy_xlsx.write_bytes(b"\xd0\xcf\x11\xe0" + b"\0" * 32)
            text_xls = root / "markers.xls"
            text_xls.write_text("gene\tlogFC\nCD3D\t2.1\n", encoding="utf-8")

            legacy_result = validate_downloaded_asset(
                {"filename": legacy_xlsx.name}, legacy_xlsx, "application/octet-stream"
            )
            text_result = validate_downloaded_asset(
                {"filename": text_xls.name}, text_xls, "text/plain"
            )

        self.assertTrue(legacy_result[0])
        self.assertTrue(text_result[0])

    def test_excel_validator_rejects_plain_text_without_tabular_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-a-table.xls"
            path.write_text("This is not a spreadsheet.\n", encoding="utf-8")

            result = validate_downloaded_asset(
                {"filename": path.name}, path, "text/plain"
            )

        self.assertFalse(result[0])
        self.assertEqual(result[1], "invalid_content")


if __name__ == "__main__":
    unittest.main()
