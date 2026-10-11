from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_cell_name_catalog_matches as audit  # noqa: E402


class CellNameMappingAuditTests(unittest.TestCase):
    def test_mapping_risk_flags_include_persisted_safety_outcomes(self) -> None:
        flags = audit.mapping_risk_flags(
            {
                "cell_name": "CAF",
                "decision": "review",
                "reason": "strict validation: closest broad category",
                "top_score": "0.43",
                "validation_flags": "low_similarity_match;admitted_generalization",
            }
        )

        self.assertEqual(
            flags,
            ["low_similarity_match", "admitted_generalization"],
        )


if __name__ == "__main__":
    unittest.main()
