from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cellphenorank.io import sha256
from cellphenorank.evaluation import evaluate_open_rankings
from cellphenorank.pipeline import verify_freeze


class EvaluationManifestTests(unittest.TestCase):
    def _broad_fixture(self, root: Path) -> tuple[dict[str, str], Path]:
        files = {}
        inputs = {}
        for name, key in (
            ("diseases.tsv", "diseases"),
            ("models.tsv", "models"),
            ("model_phenotypes.tsv", "model_phenotypes"),
            ("gold_standard.tsv", "gold_standard"),
        ):
            path = root / name
            path.write_text(name + "\n", encoding="utf-8")
            files[name] = {"sha256": sha256(path)}
            inputs[key] = str(path)
        manifest_path = root / "source_snapshot_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source_labels_frozen": True,
                    "frozen": False,
                    "middle_layer_independent": True,
                    "publication_grade": False,
                    "review_policy": "source_hpo_mgi_unreviewed",
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        inputs["source_snapshot_manifest"] = str(manifest_path)
        return inputs, manifest_path

    def test_broad_source_snapshot_is_accepted_without_curated_freeze_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs, _ = self._broad_fixture(Path(temporary))
            result = verify_freeze(
                {
                    "project": {"evaluation_mode": "source_label_broad"},
                    "inputs": inputs,
                }
            )
            self.assertTrue(result["accepted"])
            self.assertEqual(result["manifest_type"], "source_label_snapshot")
            self.assertFalse(result["publication_grade"])

    def test_broad_source_snapshot_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs, _ = self._broad_fixture(Path(temporary))
            Path(inputs["gold_standard"]).write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold_standard.tsv"):
                verify_freeze(
                    {
                        "project": {"evaluation_mode": "source_label_broad"},
                        "inputs": inputs,
                    }
                )

    def test_open_evaluation_counts_missing_gold_as_failure(self) -> None:
        rows = [
            {"disease_id": "D1", "gene_symbol": "A", "rank": 1},
            {"disease_id": "D2", "gene_symbol": "B", "rank": 1},
        ]
        disease_rows, metrics = evaluate_open_rankings(
            rows,
            {"D1": {"A"}, "D2": {"MISSING"}},
            "rank",
            "open",
            100,
        )
        by_disease = {row["disease_id"]: row for row in disease_rows}
        self.assertEqual(by_disease["D2"]["best_known_gene_rank"], 101)
        self.assertEqual(by_disease["D2"]["reciprocal_rank"], 0.0)
        self.assertEqual(metrics["gold_gene_retrieval_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
