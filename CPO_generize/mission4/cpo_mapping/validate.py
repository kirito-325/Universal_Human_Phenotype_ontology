from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import output_root, project_path
from .io import read_tsv, write_json
from .matrix import CHANNELS, load_prepared_scoring, score_term_pair


REQUIRED_OUTPUTS = (
    "audit/input_audit.md",
    "audit/schema_inventory.json",
    "audit/field_mapping.yaml",
    "normalized/assertions.parquet",
    "normalized/quarantined_assertions.parquet",
    "normalized/cpo_structure_nodes.parquet",
    "normalized/cpo_node_assertions.parquet",
    "indices/hpo_index.tsv",
    "indices/mpo_index.tsv",
    "profiles/profile_summary.tsv",
    "ontology/cl_ic.tsv",
    "ontology/go_ic.tsv",
    "matrices/hpo_mpo_cpo.h5",
    "matrices/score_status_metadata.json",
    "candidates/hpo_to_mpo_topk.parquet",
    "candidates/mpo_to_hpo_topk.parquet",
    "candidates/auxiliary_topk.parquet",
    "candidates/branch_support_topk.parquet",
    "explanations/best_cpo_matches.parquet",
    "explanations/auxiliary_matches.parquet",
    "explanations/branch_matches.parquet",
    "evaluation/reference_manifest.tsv",
    "evaluation/coverage.tsv",
    "evaluation/ranking_metrics.tsv",
    "evaluation/review_cases.tsv",
    "reports/final_report.md",
    "matrices/hpo_mpo_all_abnormal_branch.h5",
    "matrices/all_abnormal_score_status_metadata.json",
    "indices/hpo_all_abnormal_index.tsv",
    "indices/mpo_abnormal_index.tsv",
    "candidates/hpo_all_to_mpo_all_topk.parquet",
    "candidates/mpo_all_to_hpo_all_topk.parquet",
    "explanations/all_abnormal_branch_matches.parquet",
    "evaluation/all_abnormal_coverage.tsv",
    "reports/all_abnormal_report.md",
)


def run_validation(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    randomizer = random.Random(int(config["runtime"]["seed"]))
    hpo_index = {term_id: index for index, term_id in enumerate(prepared.hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    maximum_error = 0.0
    formula_pairs_checked = 0

    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for channel, (h_class, m_class) in CHANNELS.items():
            h_values = [
                index
                for index, profile in enumerate(prepared.profiles[("HPO", h_class)])
                if profile.size
            ]
            m_values = [
                index
                for index, profile in enumerate(prepared.profiles[("MPO", m_class)])
                if profile.size
            ]
            for _ in range(100):
                h_index = randomizer.choice(h_values)
                m_index = randomizer.choice(m_values)
                reference = score_term_pair(prepared, h_index, m_index, channel)
                assert reference is not None
                for score_name, expected in (
                    ("shape", reference.shape),
                    ("support", reference.support),
                    ("conflict", reference.conflict),
                ):
                    actual = float(handle[channel][score_name][h_index, m_index])
                    if math.isnan(expected):
                        if not math.isnan(actual):
                            raise ValueError(
                                f"NA mismatch for {channel}/{score_name}: {h_index}, {m_index}"
                            )
                    else:
                        maximum_error = max(maximum_error, abs(actual - expected))
                formula_pairs_checked += 1

        forward = pd.read_parquet(root / "candidates/hpo_to_mpo_topk.parquet")
        candidate_sample = forward.sample(
            min(200, len(forward)), random_state=int(config["runtime"]["seed"])
        )
        for row in candidate_sample.itertuples(index=False):
            stored = float(
                handle[row.channel]["shape"][
                    hpo_index[str(row.query_id)], mpo_index[str(row.target_id)]
                ]
            )
            if abs(stored - float(row.shape)) > 1e-6:
                raise ValueError("candidate shape does not match matrix")

        _, provenance = read_tsv(
            project_path(config, config["inputs"]["inheritance_provenance"])
        )
        descendant_leaves: dict[str, set[str]] = {}
        for row in provenance:
            descendant_leaves.setdefault(row["target_mp_id"], set()).add(
                row["source_leaf_mp_id"]
            )
        branch = pd.read_parquet(root / "candidates/branch_support_topk.parquet")
        branch_sample = branch.sample(
            min(50, len(branch)), random_state=int(config["runtime"]["seed"])
        )
        for row in branch_sample.itertuples(index=False):
            m_class = CHANNELS[str(row.channel)][1]
            leaf_indices = sorted(
                mpo_index[leaf]
                for leaf in descendant_leaves[str(row.mpo_ancestor_id)]
                if prepared.profiles[("MPO", m_class)][mpo_index[leaf]].size
            )
            actual = float(
                np.nanmax(
                    handle[row.channel]["shape"][
                        hpo_index[str(row.hpo_id)], leaf_indices
                    ]
                )
            )
            if abs(actual - float(row.branch_score)) > 1e-6:
                raise ValueError("branch score is not the complete descendant-leaf maximum")

    assertions = pd.read_parquet(
        root / "normalized/assertions.parquet", columns=["assertion_id"]
    )
    valid_assertions = set(assertions["assertion_id"].astype(str))
    explanations = pd.read_parquet(
        root / "explanations/best_cpo_matches.parquet",
        columns=["status", "hpo_assertion_id", "mpo_assertion_id"],
    )
    explanation_sample = explanations.sample(
        min(1000, len(explanations)), random_state=int(config["runtime"]["seed"])
    )
    for row in explanation_sample.itertuples(index=False):
        if row.status == "OK" and (
            str(row.hpo_assertion_id) not in valid_assertions
            or str(row.mpo_assertion_id) not in valid_assertions
        ):
            raise ValueError("explanation references an unknown normalized assertion")

    missing = [relative for relative in REQUIRED_OUTPUTS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required output files are missing: {missing}")
    metadata = json.loads(
        (root / "matrices/score_status_metadata.json").read_text(encoding="utf-8")
    )
    if metadata["not_computed_block_count"] != 0:
        raise ValueError("matrix metadata reports uncomputed blocks")
    result = {
        "status": "PASS",
        "formula_pairs_checked": formula_pairs_checked,
        "maximum_float32_absolute_error": maximum_error,
        "candidate_rows_checked": len(candidate_sample),
        "branch_rows_checked": len(branch_sample),
        "explanation_rows_checked": len(explanation_sample),
        "missing_required_files": missing,
        "completed_matrix_blocks": metadata["completed_block_count"],
        "expected_matrix_blocks": metadata["expected_block_count"],
        "not_computed_matrix_blocks": metadata["not_computed_block_count"],
    }
    write_json(root / "logs/validation_summary.json", result)
    return result
