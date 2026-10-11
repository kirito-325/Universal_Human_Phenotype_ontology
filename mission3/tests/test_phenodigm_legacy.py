from __future__ import annotations

import unittest

from cellphenorank.phenodigm_legacy import (
    combine_percentage_scores,
    score_exomiser_current_profile,
    score_phenodigm_profile,
)


class FakeMatrix:
    def __init__(self) -> None:
        self.scores = {
            "H1": {"M1": 4.0, "M2": 1.0},
            "H2": {"M1": 2.0, "M2": 3.0},
        }
        self.best_by_hpo = {"H1": (4.0, "M1"), "H2": (3.0, "M2")}
        self.present_mpo = {"M1", "M2"}

    def get(self, hp_id: str, mp_id: str) -> float:
        return self.scores.get(hp_id, {}).get(mp_id, 0.0)


class LegacyPhenodigmTests(unittest.TestCase):
    def test_original_combination_does_not_silently_clip(self) -> None:
        self.assertEqual(combine_percentage_scores(122.0, 100.0), 111.0)

    def test_symmetric_profile_and_ideal_normalization(self) -> None:
        score = score_phenodigm_profile(["H1", "H2"], ["M1"], FakeMatrix())
        self.assertAlmostEqual(score["raw_avg_score"], 10.0 / 3.0)
        self.assertAlmostEqual(score["raw_max_score"], 4.0)
        self.assertAlmostEqual(score["ideal_avg_score"], 3.5)
        self.assertAlmostEqual(score["ideal_max_score"], 4.0)
        self.assertAlmostEqual(score["avg_percentage_score"], 1000.0 / 10.5)
        self.assertAlmostEqual(score["max_percentage_score"], 100.0)

    def test_terms_absent_from_legacy_matrix_are_excluded(self) -> None:
        score = score_phenodigm_profile(
            ["H1", "H_MISSING"], ["M1", "M_MISSING"], FakeMatrix()
        )
        self.assertEqual(score["mapped_hpo_count"], 1)
        self.assertEqual(score["legacy_mpo_count"], 1)

    def test_current_exomiser_uses_only_query_matched_model_terms(self) -> None:
        score = score_exomiser_current_profile(
            ["H1", "H2"], ["M1", "M_MISSING"], FakeMatrix()
        )
        self.assertEqual(score["legacy_mpo_count"], 1)
        self.assertAlmostEqual(score["raw_avg_score"], 10.0 / 3.0)
        self.assertAlmostEqual(score["ideal_avg_score"], 3.5)
        self.assertLessEqual(score["phenodigm_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
