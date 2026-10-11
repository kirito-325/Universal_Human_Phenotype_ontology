from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from semantic_cpo.fusion import fuse_array, fuse_scalar, midcdf
from semantic_cpo.ontology import Ontology
from semantic_cpo.profiles import balanced_ot, bma, mutual_best, one_to_one


class MathContractTests(unittest.TestCase):
    def test_ontology_ic_and_node_scores(self) -> None:
        text = """format-version: 1.2
data-version: fixture

[Term]
id: X:0
name: root

[Term]
id: X:1
name: broad
is_a: X:0 ! root

[Term]
id: X:2
name: leaf a
is_a: X:1 ! broad

[Term]
id: X:3
name: leaf b
is_a: X:1 ! broad
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.obo"
            path.write_text(text, encoding="utf-8")
            ontology = Ontology.from_obo(path, "X:0")
        self.assertEqual(ontology.ic["X:0"], 0.0)
        self.assertEqual(ontology.ic["X:2"] / ontology.log_n, 1.0)
        self.assertEqual(ontology.similarity("lin", "X:2", "X:2"), 1.0)
        self.assertLess(ontology.similarity("global_resnik", "X:1", "X:1"), 1.0)
        for method in ("exact", "lin", "global_resnik", "ic_ancestor_jaccard"):
            value = ontology.similarity(method, "X:2", "X:3")
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_profile_10_by_1_contract(self) -> None:
        values = np.ones((10, 1))
        ids = [f"a{i:02d}" for i in range(10)]
        self.assertEqual(bma(values), 1.0)
        self.assertEqual(balanced_ot(values), 1.0)
        self.assertAlmostEqual(mutual_best(values, ids, ["b"]), 0.55)
        self.assertAlmostEqual(one_to_one(values), 0.55)

    def test_equal_uniform_ot_matches_assignment(self) -> None:
        values = np.asarray([[1.0, 0.2], [0.1, 0.8]])
        self.assertAlmostEqual(balanced_ot(values), one_to_one(values))

    def test_fusion_bounds_and_fallback(self) -> None:
        methods = ("positive", "linear", "bounded_geometric")
        for method in methods:
            self.assertEqual(fuse_scalar(method, 0.8, None, 0.0), 0.8)
            value = fuse_scalar(method, 0.8, 0.2, 1.0)
            self.assertLessEqual(abs(value - 0.8), 0.05 + 1e-12)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(fuse_scalar("positive", 0.8, 0.2, 1.0), 0.802)
        self.assertAlmostEqual(fuse_scalar("linear", 0.8, 0.2, 1.0), 0.77)
        self.assertAlmostEqual(fuse_scalar("bounded_geometric", 0.8, 0.2, 1.0), 0.78)
        self.assertGreater(fuse_scalar("bounded_geometric", 0.8, 0.0, 1.0), 0.0)
        s = np.asarray([[0.8, 0.3]], dtype=np.float32)
        m = np.asarray([[np.nan, 0.2]], dtype=np.float32)
        q = np.asarray([[0.0, 1.0]], dtype=np.float32)
        output = fuse_array("linear", s, m, q)
        self.assertEqual(float(output[0, 0]), float(s[0, 0]))

    def test_midcdf_includes_zero_and_uses_midrank(self) -> None:
        self.assertEqual(midcdf(0.0, np.asarray([0.0, 0.0, 1.0])), 1.0 / 3.0)
        self.assertIsNone(midcdf(0.0, np.asarray([math.nan])))


if __name__ == "__main__":
    unittest.main()

