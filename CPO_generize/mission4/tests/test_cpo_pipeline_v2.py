from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from cpo_pipeline.term_matrix import _write_bma_view
from cpo_pipeline.mapping import _direction_code
from cpo_pipeline.validate import _aggregate_fixture
from cpo_pipeline.gene_rank import _rank_interval


class V2TermMatrixTest(unittest.TestCase):
    def test_incidence_bma_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "matrix.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("shape", shape=(1, 1), dtype="float32")
                handle.create_dataset("forward", shape=(1, 1), dtype="float32")
                handle.create_dataset("reverse", shape=(1, 1), dtype="float32")
                result = _write_bma_view(
                    handle,
                    "shape",
                    [np.asarray([0, 1], dtype=np.int32)],
                    [np.asarray([0, 1], dtype=np.int32)],
                    np.asarray([[1.0, 0.2], [0.4, 0.0]], dtype=np.float32),
                    root,
                    {"matrix": {"hpo_block_size": 1, "mpo_block_size": 1}},
                    "forward",
                    "reverse",
                )
                self.assertAlmostEqual(float(handle["forward"][0, 0]), 0.7)
                self.assertAlmostEqual(float(handle["reverse"][0, 0]), 0.6)
                self.assertAlmostEqual(float(handle["shape"][0, 0]), 0.65)
                self.assertEqual(result["finite_pair_count"], 1)

    def test_direction_is_separate_from_shape(self) -> None:
        base = {
            "structure_id": "cpo:1",
            "direction_signature_status": "unique",
            "direction_target": "GO:1",
            "direction_property": "activity",
        }
        increased = {**base, "direction_state": "increased"}
        decreased = {**base, "direction_state": "decreased"}
        unknown = {**base, "direction_signature_status": "unknown", "direction_state": ""}
        other_structure = {**increased, "structure_id": "cpo:2"}
        self.assertEqual(_direction_code(increased, increased), 1)
        self.assertEqual(_direction_code(increased, decreased), -1)
        self.assertEqual(_direction_code(increased, unknown), 0)
        self.assertEqual(_direction_code(increased, other_structure), 0)

    def test_coverage_adjusted_disease_fixture(self) -> None:
        result = _aggregate_fixture()
        self.assertEqual(result["fixture_score_observed"], 1.0)
        self.assertEqual(result["fixture_score_evidence"], 0.75)

    def test_gene_rank_interval_preserves_ties(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.8, 0.1, np.nan, 0.0])
        self.assertEqual(_rank_interval(scores, 0.8), (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
