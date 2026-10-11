from __future__ import annotations

import unittest

from cellphenorank.impc_dr20 import (
    _model_life_stage,
    is_eligible_gene,
    split_disorder_ids,
)


class ImpcDr20Tests(unittest.TestCase):
    def test_eligible_gene_requires_both_phenotype_profiles(self) -> None:
        row = {
            "IMPC_pipeline": "yes",
            "IMPC_phenotypes": "yes",
            "HPO_phenotypes": "yes",
            "disorder_id": "OMIM:1",
        }
        self.assertTrue(is_eligible_gene(row))
        self.assertFalse(is_eligible_gene({**row, "HPO_phenotypes": "no"}))
        self.assertFalse(is_eligible_gene({**row, "disorder_id": "-"}))

    def test_disorder_ids_are_deduplicated_without_parsing_names(self) -> None:
        self.assertEqual(
            split_disorder_ids("OMIM:1|ORPHA:2|OMIM:1"),
            ("OMIM:1", "ORPHA:2"),
        )

    def test_life_stage_is_derived_from_pipeline_and_procedure(self) -> None:
        base = {
            "pipeline_name": "Center pipeline",
            "pipeline_stable_id": "CENTER_001",
            "procedure_name": "Procedure",
            "procedure_stable_id": "IMPC_XYZ_001",
        }
        self.assertEqual(_model_life_stage(base), "early")
        self.assertEqual(
            _model_life_stage({**base, "procedure_name": "Gross Morphology Embryo"}),
            "embryo",
        )
        self.assertEqual(
            _model_life_stage({**base, "pipeline_name": "Center late adult"}),
            "late",
        )
        self.assertEqual(
            _model_life_stage({**base, "pipeline_name": "Center Interval Pipeline"}),
            "middle",
        )


if __name__ == "__main__":
    unittest.main()
