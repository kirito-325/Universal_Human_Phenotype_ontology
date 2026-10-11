from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_cpo.provider import ArrayProvider


ROOT = Path(__file__).resolve().parents[1] / "results" / "semantic_cpo_training_free_v1"


class RealExecutionContractTests(unittest.TestCase):
    def test_frozen_independent_inputs_and_baseline(self) -> None:
        frozen = json.loads((ROOT / "frozen_inputs/input_manifest.json").read_text())
        baseline = json.loads((ROOT / "r0/baseline_reproduction.json").read_text())
        self.assertTrue(frozen["frozen"])
        self.assertTrue(frozen["middle_layer_independent_benchmark"])
        self.assertFalse(frozen["training_enabled"])
        self.assertEqual(baseline["status"], "pass")
        self.assertFalse(baseline["failures"])

    def test_selected_surface_invariants(self) -> None:
        selection = json.loads((ROOT / "r4_fusion/selection.json").read_text())
        finalist = selection["finalists"][0]
        self.assertEqual(finalist["method_id"], "node_exact__atom_A2__profile_bma")
        self.assertEqual(finalist["fusion"], "linear")
        self.assertLessEqual(float(finalist["maximum_absolute_delta"]), 0.05 + 1e-6)
        self.assertEqual(int(finalist["exact_fallback_pair_count"]), 7_598_656)

    def test_six_arm_and_null_outcome(self) -> None:
        metrics = pd.read_csv(ROOT / "r5_confirmation/six_arm_metrics.tsv", sep="\t")
        values = metrics.pivot(index="arm", columns="metric", values="value")
        self.assertAlmostEqual(values.loc["P0", "mrr"], 0.6631753247)
        self.assertAlmostEqual(values.loc["O0", "mrr"], 0.6636515152)
        self.assertAlmostEqual(values.loc["P1", "mrr"], values.loc["O1", "mrr"])
        controls = json.loads((ROOT / "r5_controls/controls_manifest.json").read_text())
        self.assertEqual(controls["mechanism_claim_status"], "exploratory")

    def test_pair_scores_preserve_unknown_and_zero(self) -> None:
        frame = pd.read_parquet(ROOT / "report/pair_scores.parquet")
        self.assertEqual(len(frame), 1_494_064)
        self.assertIn("no_qualifying_cpo", set(frame["mechanism_status"]))
        self.assertTrue(frame.loc[frame["mechanism_status"] == "no_qualifying_cpo", "M_used"].isna().all())
        fallback = frame[frame["mechanism_status"] == "no_qualifying_cpo"]
        self.assertTrue(np.allclose(fallback["F"], fallback["S_used"]))
        computed = frame[frame["mechanism_status"] == "computed"]
        self.assertTrue((computed["M_used"] == 0).any())

    def test_provider_accepts_term_ids_only(self) -> None:
        s_raw = np.asarray([[0.2, 0.4]])
        s = np.asarray([[0.2, 0.4]])
        m = np.asarray([[np.nan, 0.7]])
        q = np.asarray([[0.0, 1.0]])
        fused = np.asarray([[0.2, 0.415]])
        provider = ArrayProvider(
            ["HP:1"], ["MP:1", "MP:2"], s_raw, s, m, q, fused, "x", "n", "c"
        )
        result = provider.get_pairs(["HP:1"], ["MP:1"], "x")[0]
        self.assertIsNone(result.M_used)
        self.assertEqual(result.F, result.S_used)
        self.assertEqual(provider.get_block(["HP:1"], ["MP:2"], "x").shape, (1, 1))

    def test_freeze_recommends_pure_semantics(self) -> None:
        value = json.loads((ROOT / "frozen_method/frozen_method.json").read_text())
        self.assertTrue(value["frozen"])
        self.assertEqual(value["recommended_default"], "P0")
        self.assertEqual(value["exploratory_candidate"]["status"], "exploratory")
        self.assertFalse(value["external_test_data_seen"])


if __name__ == "__main__":
    unittest.main()

