#!/usr/bin/env python3
"""Run task suites in separate processes to isolate duplicate test module names."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = [
    (".", ["tests"]),
    ("01_cpo_construction/cl_go_linking", ["tests"]),
    ("01_cpo_construction/cell_marker_mining", ["tests"]),
    ("02_phenotype_cpo_mapping/hpo_cpo/dataset_tools", ["tests"]),
    ("02_phenotype_cpo_mapping/mpo_cpo/dataset_tools", ["tests"]),
    ("02_phenotype_cpo_mapping/review_browser", ["tests"]),
    ("03_mpo_hpo_mapping/cpo_profile_similarity", ["tests"]),
    ("03_mpo_hpo_mapping/semantic_cpo_experiments", [
        "tests/test_math_contracts.py",
        "tests/test_real_execution_contracts.py::RealExecutionContractTests::test_provider_accepts_term_ids_only",
    ]),
    ("04_downstream_evaluation/gene_prioritization", ["tests"]),
    ("04_downstream_evaluation/mouse_model_prioritization", ["tests"]),
    ("04_downstream_evaluation/single_cell_validation", ["tests"]),
    ("04_downstream_evaluation/cpo_disease_gene_prediction", ["tests"]),
]
AGENTS = [(f"02_phenotype_cpo_mapping/{side}_cpo", ["tests"]) for side in ("hpo", "mpo")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["scientific", "agents", "all"], default="scientific")
    args = parser.parse_args()
    cases = (SCIENTIFIC if args.group != "agents" else []) + (AGENTS if args.group != "scientific" else [])
    failures = []
    for directory, targets in cases:
        print(f"\nTesting {directory}", flush=True)
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", *targets], cwd=ROOT / directory)
        if result.returncode:
            failures.append(directory)
    if failures:
        print("Failed task suites: " + ", ".join(failures))
        return 1
    print(f"All {len(cases)} task suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
