from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellphenorank.single_cell_discovery.savi_literature_case import load_signature


class SaviLiteratureCaseTests(unittest.TestCase):
    def test_upr_signature_parser_excludes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signature.csv"
            path.write_text("UPR\npmid 123\nDDIT3\nHSPA5\nDDIT3\n", encoding="utf-8")
            self.assertEqual(load_signature(path), ["DDIT3", "HSPA5"])


if __name__ == "__main__":
    unittest.main()
