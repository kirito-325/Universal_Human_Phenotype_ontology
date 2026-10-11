from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from cpo_mapping.direction import (
    DirectionSignature,
    aggregate_signatures,
    compare_signatures,
    load_rules,
    normalize_direction,
)
from cpo_mapping.ontology import Ontology
from cpo_mapping.scoring import directional_bma, score_pair
from cpo_mapping.matrix import _completed_blocks


class OntologyTest(unittest.TestCase):
    def test_intrinsic_ic_lin_and_multipath(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.obo"
            path.write_text(
                """format-version: 1.2
data-version: test-v1

[Term]
id: T:0
name: root

[Term]
id: T:1
name: branch one
is_a: T:0 ! root

[Term]
id: T:2
name: branch two
is_a: T:0 ! root

[Term]
id: T:3
name: multipath leaf
is_a: T:1 ! branch one
is_a: T:2 ! branch two

[Term]
id: T:4
name: sibling
is_a: T:1 ! branch one
""",
                encoding="utf-8",
            )
            ontology = Ontology.from_obo(path, "T:0", "T:")
            self.assertEqual(len(ontology.nodes), 5)
            self.assertEqual(ontology.descendant_counts["T:0"], 4)
            self.assertEqual(ontology.descendant_counts["T:1"], 2)
            self.assertEqual(ontology.lin("T:3", "T:3"), 1.0)
            self.assertEqual(ontology.lin("T:1", "T:2"), 0.0)
            self.assertGreater(ontology.lin("T:3", "T:4"), 0.0)


class ScoringTest(unittest.TestCase):
    def test_fixed_bma_example(self) -> None:
        matrix = np.asarray([[1.0, 0.2], [0.4, 0.0]])
        forward, reverse, bma = directional_bma(matrix)
        self.assertAlmostEqual(forward, 0.7)
        self.assertAlmostEqual(reverse, 0.6)
        self.assertAlmostEqual(bma, 0.65)

    def test_shape_support_conflict_and_unknown(self) -> None:
        shape = np.asarray([[1.0]])
        agree = score_pair(shape, np.asarray([[1]], dtype=np.int8))
        self.assertEqual((agree.shape, agree.support, agree.conflict), (1.0, 1.0, 0.0))
        conflict = score_pair(shape, np.asarray([[-1]], dtype=np.int8))
        self.assertEqual(
            (conflict.shape, conflict.support, conflict.conflict), (1.0, 0.0, 1.0)
        )
        unknown = score_pair(shape, np.asarray([[0]], dtype=np.int8))
        self.assertTrue(np.isnan(unknown.support))
        self.assertTrue(np.isnan(unknown.conflict))

    def test_checkpoint_digest_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocks.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(value)
                    for value in (
                        {
                            "channel": "DD",
                            "row_start": 0,
                            "row_end": 64,
                            "status": "complete",
                            "input_digest": "current",
                            "config_sha256": "config",
                        },
                        {
                            "channel": "DM",
                            "row_start": 0,
                            "row_end": 64,
                            "status": "complete",
                            "input_digest": "stale",
                            "config_sha256": "config",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _completed_blocks(path, "current", "config"), {("DD", 0, 64)}
            )


class DirectionTest(unittest.TestCase):
    def test_explicit_process_rule_matches(self) -> None:
        rules = load_rules("config/direction_rules.tsv")
        value = normalize_direction(
            {
                "direction": "increased",
                "cell_text": "cardiac muscle cell",
                "process_text": "apoptosis",
                "quality_text": "",
            },
            rules,
        )
        self.assertEqual(value["direction_target"], "go_process")
        self.assertEqual(value["direction_property"], "activity_or_level")
        self.assertEqual(value["direction_resolution_status"], "resolved_explicit_text")

    def test_aggregate_and_compare(self) -> None:
        values = [
            {
                "direction_resolution_status": "resolved_explicit_text",
                "direction_target": "go_process",
                "direction_property": "activity_or_level",
                "direction_state": "increased",
            }
        ]
        signature = aggregate_signatures(values)
        self.assertEqual(signature.status, "unique")
        self.assertEqual(compare_signatures(signature, signature), "agree")
        opposite = DirectionSignature(
            "unique", "go_process", "activity_or_level", "decreased"
        )
        self.assertEqual(compare_signatures(signature, opposite), "conflict")
        mixed = aggregate_signatures(
            [values[0], {**values[0], "direction_state": "decreased"}]
        )
        self.assertEqual(mixed.status, "mixed_direction")


if __name__ == "__main__":
    unittest.main()
