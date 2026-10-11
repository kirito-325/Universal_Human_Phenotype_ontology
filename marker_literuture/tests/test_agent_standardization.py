import unittest

from literature_pipeline.agent_standardization import normalize_analysis_for_evidence


class AgentStandardizationPolicyTests(unittest.TestCase):
    def test_standard_harness_rejects_genotype_contrast_as_cell_state(self) -> None:
        analysis = {
            "paper_context": {},
            "table_decisions": [
                {
                    "table_id": "T0001",
                    "classification": "marker_statistics",
                    "feature_type": "gene",
                    "reason": "genotype comparison",
                    "extractions": [
                        {
                            "structure_block_id": "",
                            "layout": "statistics_long",
                            "header_row": 1,
                            "data_start_row": 2,
                            "cell_cluster_source": "constant",
                            "cell_cluster_column": -1,
                            "cell_cluster_value": (
                                "NOTCH WT vs damaging mutant SCLC tumors and cell lines"
                            ),
                            "cell_cluster_map": {},
                            "columns": {
                                "gene": 0,
                                "p_val": -1,
                                "avg_logFC": 1,
                                "pct.1": -1,
                                "pct.2": -1,
                                "p_val_adj": -1,
                            },
                            "signature_columns": [],
                            "signature_cell_column": -1,
                            "signature_gene_column": -1,
                        }
                    ],
                }
            ],
            "excluded_table_ids": [],
        }
        evidence = {
            "harness_mode": "standard",
            "tables": [
                {
                    "table_id": "T0001",
                    "sheet_name": "Notch WT v Damaging Mutant",
                    "normalized_structure": {"blocks": []},
                }
            ],
        }

        normalized = normalize_analysis_for_evidence(analysis, evidence)

        decision = normalized["table_decisions"][0]
        self.assertEqual(decision["classification"], "review")
        self.assertEqual(decision["extractions"], [])
        self.assertEqual(
            decision["reason"], "experimental-condition contrast is not a cell name"
        )


if __name__ == "__main__":
    unittest.main()
