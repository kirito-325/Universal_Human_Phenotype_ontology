from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cellphenorank.single_cell_discovery.case_freeze import select_primary_case
from cellphenorank.single_cell_discovery.dataset_catalog import build_dataset_catalog
from cellphenorank.single_cell_discovery.metadata_preflight import donor_tier, preflight_rows
from cellphenorank.single_cell_discovery.xla_validation import _matched_null


class DatasetCatalogTests(unittest.TestCase):
    def test_catalog_deduplicates_accession_per_candidate(self) -> None:
        candidates = [{"candidate_case_id": "D__G", "pareto_front": "1", "priority_rank": "1", "gene_symbol": "G", "disease_name": "D"}]
        discovery = [{"candidate_case_id": "D__G", "study_accession": "GSE1", "publication_id": "P"}] * 2
        rows = build_dataset_catalog(candidates, discovery)
        self.assertEqual(len(rows), 1)

    def test_donor_tiers(self) -> None:
        self.assertEqual(donor_tier(4, 4, "cross-sectional"), "A")
        self.assertEqual(donor_tier(2, 2, "cross-sectional"), "B")
        self.assertEqual(donor_tier(1, 1, "cross-sectional"), "C")
        self.assertEqual(donor_tier(0, 0, "unknown"), "D")


class MetadataPreflightTests(unittest.TestCase):
    def test_feature_only_preflight_passes_without_expression_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "features.tsv"
            feature.write_text("id1\tG\nid2\tA\nid3\tB\nid4\tC\nid5\tD\nid6\tE\n", encoding="utf-8")
            candidate = {
                "candidate_case_id": "D__G", "gene_symbol": "G", "disease_id": "D",
                "disease_name": "Disease", "pareto_front": "1", "priority_rank": "1",
                "primary_predicted_function": "GO:1", "primary_predicted_function_name": "fn",
                "bridge_source_genes": "G", "primary_predicted_cell_names": "T cell",
            }
            dataset = {
                "candidate_case_id": "D__G", "study_accession": "GSE1", "retrieval_status": "public",
                "species": "Homo sapiens", "causal_gene_match": "exact", "patient_donor_count": "2",
                "control_donor_count": "2", "longitudinal_or_cross_sectional": "cross-sectional",
                "donor_metadata_available": "true", "processed_matrix_available": "true",
                "cell_labels_available": "true", "possible_tissue_match": "true",
                "possible_cell_match": "true", "access_restriction": "public",
                "matched_predicted_cl_ids": "CL:1", "matched_dataset_labels": "T cell",
                "feature_axis_paths": str(feature), "evidence_overlap": "none",
            }
            statuses, _, functions = preflight_rows(
                [candidate], [dataset], {"GO:1": {"G", "A", "B", "C", "D", "E"}}, root
            )
            self.assertEqual(statuses[0]["status"], "pass")
            self.assertTrue(statuses[0]["candidate_gene_on_feature_axis"])
            self.assertEqual(functions[0]["strict_feature_count"], 5)

    def test_primary_selection_prefers_front_before_data_tier(self) -> None:
        rows = [
            {"status": "pass", "pareto_front": "2", "data_tier": "A", "priority_rank": "2", "study_accession": "B"},
            {"status": "pass", "pareto_front": "1", "data_tier": "B", "priority_rank": "3", "study_accession": "A"},
        ]
        self.assertEqual(select_primary_case(rows)["study_accession"], "A")

    def test_expression_matched_null_preserves_target_size(self) -> None:
        rng = np.random.default_rng(7)
        expression = pd.DataFrame(
            rng.normal(size=(6, 120)),
            columns=[f"G{index}" for index in range(120)],
        )
        conditions = pd.Series(["HC"] * 3 + ["XLA"] * 3)
        observed, empirical_p, null, target_count = _matched_null(
            expression,
            conditions,
            [f"G{index}" for index in range(20)],
            {f"G{index}" for index in range(20)},
            iterations=50,
            bins=5,
            seed=2026,
        )
        self.assertEqual(target_count, 20)
        self.assertEqual(len(null), 50)
        self.assertTrue(np.isfinite(observed))
        self.assertGreaterEqual(empirical_p, 1 / 51)


if __name__ == "__main__":
    unittest.main()
