from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import output_root
from .io import read_tsv, write_json, write_tsv
from .matrix import CHANNELS, load_prepared_scoring


COVERAGE_FIELDS = (
    "channel",
    "hpo_complete_profile_terms",
    "mpo_complete_profile_terms",
    "finite_shape_pairs",
    "positive_shape_pairs",
    "finite_direction_pairs",
    "positive_support_pairs",
    "positive_conflict_pairs",
    "shape_pair_coverage_fraction",
    "direction_comparable_fraction_of_shape_pairs",
)


def _matrix_counts(dataset: h5py.Dataset, block_size: int = 64) -> tuple[int, int]:
    finite = 0
    positive = 0
    for start in range(0, dataset.shape[0], block_size):
        values = dataset[start : start + block_size, :]
        finite += int(np.isfinite(values).sum())
        positive += int(np.count_nonzero(np.isfinite(values) & (values > 0)))
    return finite, positive


def _build_review_cases(root: Path, limit: int = 40) -> int:
    forward = pd.read_parquet(root / "candidates/hpo_to_mpo_topk.parquet")
    branch = pd.read_parquet(root / "candidates/branch_support_topk.parquet")
    auxiliary = pd.read_parquet(root / "candidates/auxiliary_topk.parquet")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add(hpo_id: str, mpo_id: str, channel: str, category: str, score: float, note: str) -> None:
        key = (hpo_id, mpo_id)
        if key in seen or len(rows) >= limit:
            return
        seen.add(key)
        rows.append(
            {
                "hpo_id": hpo_id,
                "mpo_id": mpo_id,
                "channel": channel,
                "review_category": category,
                "score": score,
                "review_status": "PENDING_HUMAN_REVIEW",
                "reviewer_type": "agent_review_not_gold",
                "note": note,
            }
        )

    for channel in CHANNELS:
        subset = forward[
            (forward["channel"] == channel)
            & (forward["view"] == "shape_topk")
            & (forward["rank"] <= 3)
        ].sort_values(["rank", "shape"], ascending=[True, False])
        for row in subset.head(2).itertuples(index=False):
            add(str(row.query_id), str(row.target_id), channel, "high_shape", float(row.shape), "Top structural candidate")
    conflict = forward[
        np.isfinite(forward["conflict"]) & (forward["conflict"] > 0)
    ].sort_values("conflict", ascending=False)
    for row in conflict.head(8).itertuples(index=False):
        add(str(row.query_id), str(row.target_id), str(row.channel), "shape_with_direction_conflict", float(row.conflict), "Explicit comparable opposite direction")
    unknown = forward[
        (forward["view"] == "shape_topk")
        & (forward["direction_status"] == "NO_COMPARABLE_DIRECTION")
    ].sort_values("shape", ascending=False)
    for row in unknown.head(8).itertuples(index=False):
        add(str(row.query_id), str(row.target_id), str(row.channel), "high_shape_unknown_direction", float(row.shape), "Direction is unresolved or incomparable")
    for row in branch.sort_values(["rank", "branch_score"], ascending=[True, False]).head(8).itertuples(index=False):
        add(str(row.hpo_id), str(row.mpo_ancestor_id), str(row.channel), "branch_only", float(row.branch_score), f"Best descendant leaf {row.best_leaf_id}")
    for row in auxiliary.sort_values(["rank", "score"], ascending=[True, False]).head(8).itertuples(index=False):
        if row.query_side == "HPO":
            add(str(row.query_id), str(row.target_id), str(row.channel), f"{str(row.dimension).lower()}_auxiliary", float(row.score), "Single-dimension auxiliary candidate")
        else:
            add(str(row.target_id), str(row.query_id), str(row.channel), f"{str(row.dimension).lower()}_auxiliary", float(row.score), "Single-dimension auxiliary candidate")
    return write_tsv(
        root / "evaluation/review_cases.tsv",
        (
            "hpo_id",
            "mpo_id",
            "channel",
            "review_category",
            "score",
            "review_status",
            "reviewer_type",
            "note",
        ),
        rows,
    )


def run_evaluate(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    _, profile_rows = read_tsv(root / "profiles/profile_summary.tsv")
    profile_counts = Counter(
        (row["ontology_side"], row["evidence_class"])
        for row in profile_rows
        if int(row["complete_structure_count"]) > 0
    )
    coverage_rows: list[dict[str, object]] = []
    matrix_size = 12121 * 14385
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for channel, (h_class, m_class) in CHANNELS.items():
            finite_shape, positive_shape = _matrix_counts(handle[channel]["shape"])
            finite_support, positive_support = _matrix_counts(handle[channel]["support"])
            finite_conflict, positive_conflict = _matrix_counts(handle[channel]["conflict"])
            if finite_support != finite_conflict:
                raise ValueError(f"support/conflict finite masks differ for {channel}")
            coverage_rows.append(
                {
                    "channel": channel,
                    "hpo_complete_profile_terms": profile_counts[("HPO", h_class)],
                    "mpo_complete_profile_terms": profile_counts[("MPO", m_class)],
                    "finite_shape_pairs": finite_shape,
                    "positive_shape_pairs": positive_shape,
                    "finite_direction_pairs": finite_support,
                    "positive_support_pairs": positive_support,
                    "positive_conflict_pairs": positive_conflict,
                    "shape_pair_coverage_fraction": f"{finite_shape / matrix_size:.10g}",
                    "direction_comparable_fraction_of_shape_pairs": f"{finite_support / finite_shape:.10g}" if finite_shape else "",
                }
            )
    write_tsv(root / "evaluation/coverage.tsv", COVERAGE_FIELDS, coverage_rows)

    prepared = load_prepared_scoring(config)
    hpo_index = {term_id: index for index, term_id in enumerate(prepared.hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    forward_candidates = pd.read_parquet(root / "candidates/hpo_to_mpo_topk.parquet")
    control_rows: list[dict[str, object]] = []
    for channel, (h_class, m_class) in CHANNELS.items():
        subset = forward_candidates[
            (forward_candidates["channel"] == channel)
            & (forward_candidates["view"] == "shape_topk")
        ]
        jaccards: list[float] = []
        shape_values: list[float] = []
        approximate_only = 0
        exact_positive = 0
        for row in subset.itertuples(index=False):
            left = prepared.profiles[("HPO", h_class)][hpo_index[str(row.query_id)]]
            right = prepared.profiles[("MPO", m_class)][mpo_index[str(row.target_id)]]
            left_set = set(map(int, left.exact_codes))
            right_set = set(map(int, right.exact_codes))
            jaccard = len(left_set & right_set) / len(left_set | right_set)
            jaccards.append(jaccard)
            shape_values.append(float(row.shape))
            exact_positive += jaccard > 0
            approximate_only += jaccard == 0 and float(row.shape) > 0
        control_rows.append(
            {
                "channel": channel,
                "candidate_pair_count": len(jaccards),
                "mean_shape": f"{np.mean(shape_values):.10g}" if shape_values else "",
                "mean_exact_cpo_jaccard": f"{np.mean(jaccards):.10g}" if jaccards else "",
                "exact_jaccard_positive_count": exact_positive,
                "ontology_approximate_only_count": approximate_only,
                "interpretation": "Candidate-set diagnostic; not an independent accuracy metric.",
            }
        )
    write_tsv(
        root / "evaluation/exact_cpo_jaccard_control.tsv",
        (
            "channel",
            "candidate_pair_count",
            "mean_shape",
            "mean_exact_cpo_jaccard",
            "exact_jaccard_positive_count",
            "ontology_approximate_only_count",
            "interpretation",
        ),
        control_rows,
    )

    references = list(config.get("evaluation", {}).get("reference_mapping_files", []))
    if references:
        raise NotImplementedError(
            "reference mappings were configured but no audited adapter was selected"
        )
    write_tsv(
        root / "evaluation/reference_manifest.tsv",
        ("status", "source", "version", "relation_types", "note"),
        [
            {
                "status": "NOT_AVAILABLE",
                "source": "",
                "version": "",
                "relation_types": "",
                "note": "No independent reference mapping file was configured; no ranking metric was fabricated.",
            }
        ],
    )
    write_tsv(
        root / "evaluation/ranking_metrics.tsv",
        (
            "status",
            "channel",
            "view",
            "metric",
            "k",
            "value",
            "denominator",
            "note",
        ),
        [
            {
                "status": "NOT_AVAILABLE",
                "channel": "",
                "view": "",
                "metric": "",
                "k": "",
                "value": "",
                "denominator": "",
                "note": "Independent HPO-MPO reference mappings are not configured.",
            }
        ],
    )
    review_count = _build_review_cases(root)
    result = {
        "status": "complete_without_external_reference",
        "coverage": coverage_rows,
        "exact_cpo_jaccard_control": control_rows,
        "review_case_count": review_count,
        "reference_status": "NOT_AVAILABLE",
    }
    write_json(root / "logs/evaluation_summary.json", result)
    return result
