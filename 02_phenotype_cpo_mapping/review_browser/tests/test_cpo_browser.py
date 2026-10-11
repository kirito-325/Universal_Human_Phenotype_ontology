from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_database.py"
SPEC = importlib.util.spec_from_file_location("cpo_browser_builder", MODULE_PATH)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class CpoBrowserBuilderTest(unittest.TestCase):
    def test_parse_obo_retains_tree_text_and_ignores_obsolete_terms(self) -> None:
        fixture = """format-version: 1.2
data-version: test/v1

[Term]
id: T:1
name: Root
def: \"Root definition\" []

[Term]
id: T:2
name: Child
def: \"A child with \\\"quoted\\\" text\" []
synonym: \"Child synonym\" EXACT []
is_a: T:1 ! Root

[Term]
id: T:3
name: Old child
is_obsolete: true
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.obo"
            path.write_text(fixture, encoding="utf-8")
            version, records = BUILDER.parse_obo(path)

        self.assertEqual(version, "test/v1")
        self.assertEqual(records["T:2"]["parents"], ["T:1"])
        self.assertEqual(records["T:2"]["synonyms"], ["Child synonym"])
        self.assertEqual(records["T:2"]["definition"], 'A child with "quoted" text')
        self.assertNotIn("T:3", records)


if __name__ == "__main__":
    unittest.main()
