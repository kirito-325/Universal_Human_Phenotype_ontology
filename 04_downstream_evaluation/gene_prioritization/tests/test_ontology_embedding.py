from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellphenorank.ontology_embedding import parse_obo_term_texts


class OntologyEmbeddingTests(unittest.TestCase):
    def test_parser_retains_exact_synonyms_and_resolves_alt_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terms.obo"
            path.write_text(
                """format-version: 1.2
[Term]
id: HP:1
name: Primary label
alt_id: HP:OLD
synonym: \"Exact alias\" EXACT []
synonym: \"Broad alias\" BROAD []
[Term]
id: HP:2
name: Obsolete label
is_obsolete: true
""",
                encoding="utf-8",
            )
            terms = parse_obo_term_texts(path)
            self.assertEqual(terms["HP:1"].exact_synonyms, ("Exact alias",))
            self.assertIs(terms["HP:1"], terms["HP:OLD"])
            self.assertNotIn("HP:2", terms)


if __name__ == "__main__":
    unittest.main()
