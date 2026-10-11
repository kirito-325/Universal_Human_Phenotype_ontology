from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpo_pipeline.config import load_config, output_root, resource_path


OUTPUT_COLUMNS = (
    "hpo_id",
    "hpo_label",
    "hpo_node_type",
    "mpo_id",
    "mpo_label",
    "mpo_node_type",
    "hpo_to_mpo_rank",
    "mpo_to_hpo_rank",
    "reciprocal_rank_max",
    "reciprocal_rank_sum",
    "reciprocal_tier",
    "score_shape",
    "hpo_to_mpo_score_F",
    "mpo_to_hpo_score_R",
    "F_R_absolute_difference",
    "score_direction_support",
    "score_direction_conflict",
    "direction_status",
    "n_hpo_cpo",
    "n_mpo_cpo",
    "positive_match_coverage_hpo",
    "positive_match_coverage_mpo",
    "source_origin_summary",
    "evidence_class_summary",
    "missing_required_self_flags",
    "source_leaf_counts",
    "heterogeneity_flags",
    "profile_identity_group",
    "best_hpo_structure_id",
    "best_mpo_structure_id",
    "best_hpo_cl_id",
    "best_mpo_cl_id",
    "best_hpo_go_id",
    "best_mpo_go_id",
    "best_cl_similarity",
    "best_go_similarity",
    "best_cpo_pair_score",
    "best_pair_direction_status",
    "hpo_source_leaf_id",
    "mpo_source_leaf_id",
    "hpo_inheritance_path",
    "mpo_inheritance_path",
    "hpo_publication_ids",
    "mpo_publication_ids",
    "mhmi_reference_match_type",
    "in_mhmi_reference",
    "review_flags",
)


def _labels(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, usecols=["term_id", "label"])
    return dict(zip(frame["term_id"], frame["label"]))


def _reference(config: dict[str, Any]) -> dict[tuple[str, str], str]:
    frame = pd.read_csv(
        resource_path(config, "hpo_mpo_reference"),
        sep="\t",
        dtype=str,
        usecols=["hp_id", "mp_id", "match_type"],
    )
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        grouped[(str(row.hp_id), str(row.mp_id))].add(str(row.match_type))
    return {key: "|".join(sorted(values)) for key, values in grouped.items()}


def _direction_status(support: float, conflict: float) -> str:
    has_support = np.isfinite(support) and support > 0
    has_conflict = np.isfinite(conflict) and conflict > 0
    if has_support and has_conflict:
        return "MIXED_SUPPORT_AND_CONFLICT"
    if has_conflict:
        return "CONFLICT"
    if has_support:
        return "SUPPORTED"
    return "NO_COMPARABLE_DIRECTION"


def _flags(row: pd.Series) -> str:
    flags: list[str] = []
    if np.isfinite(row["score_direction_conflict"]) and row["score_direction_conflict"] > 0:
        flags.append("DIRECTION_CONFLICT")
    if "true" in str(row["missing_required_self_flags"]).lower():
        flags.append("MISSING_REQUIRED_SELF")
    if row["F_R_absolute_difference"] >= 0.5:
        flags.append("F_R_ASYMMETRIC")
    if "inherited" in str(row["source_origin_summary"]):
        flags.append("INHERITED_EVIDENCE_INVOLVED")
    if max(int(row["n_hpo_cpo"]), int(row["n_mpo_cpo"])) >= 50:
        flags.append("BROAD_CPO_PROFILE")
    if row["direction_status"] == "NO_COMPARABLE_DIRECTION":
        flags.append("DIRECTION_UNKNOWN_OR_INCOMPARABLE")
    return "|".join(flags) if flags else "NONE"


def build_table(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    mapping_root = root / "mapping"
    hpo = pd.read_parquet(mapping_root / "hpo_to_mpo.parquet")
    mpo = pd.read_parquet(mapping_root / "mpo_to_hpo.parquet")
    hpo = hpo[(hpo["view"] == "topk_review") & (hpo["rank"] <= 5)].copy()
    mpo = mpo[(mpo["view"] == "topk_review") & (mpo["rank"] <= 5)][
        ["hpo_id", "mpo_id", "rank"]
    ].rename(columns={"rank": "mpo_to_hpo_rank"})
    table = hpo.merge(mpo, on=["hpo_id", "mpo_id"], how="inner", validate="one_to_one")
    table = table.rename(
        columns={
            "rank": "hpo_to_mpo_rank",
            "hpo_to_mpo": "hpo_to_mpo_score_F",
            "mpo_to_hpo": "mpo_to_hpo_score_R",
            "score_support": "score_direction_support",
            "score_conflict": "score_direction_conflict",
        }
    )
    table["hpo_label"] = table["hpo_id"].map(
        _labels(root / "indices/hpo_index.tsv")
    )
    table["mpo_label"] = table["mpo_id"].map(
        _labels(root / "indices/mpo_index.tsv")
    )
    table["reciprocal_rank_max"] = table[
        ["hpo_to_mpo_rank", "mpo_to_hpo_rank"]
    ].max(axis=1)
    table["reciprocal_rank_sum"] = table[
        ["hpo_to_mpo_rank", "mpo_to_hpo_rank"]
    ].sum(axis=1)
    table["reciprocal_tier"] = np.select(
        [table["reciprocal_rank_max"] == 1, table["reciprocal_rank_max"] <= 3],
        ["MUTUAL_TOP1", "MUTUAL_TOP3"],
        default="MUTUAL_TOP5",
    )
    table["F_R_absolute_difference"] = (
        table["hpo_to_mpo_score_F"] - table["mpo_to_hpo_score_R"]
    ).abs()
    table["direction_status"] = [
        _direction_status(float(support), float(conflict))
        for support, conflict in zip(
            table["score_direction_support"], table["score_direction_conflict"]
        )
    ]

    evidence = pd.read_parquet(root / "explanations/term_matches.parquet")
    evidence = evidence[evidence["score_type"] == "shape"][
        [
            "hpo_id",
            "mpo_id",
            "hpo_structure_id",
            "mpo_structure_id",
            "hpo_cl_id",
            "mpo_cl_id",
            "hpo_go_id",
            "mpo_go_id",
            "cl_similarity",
            "go_similarity",
            "cpo_pair_score",
            "direction_status",
            "hpo_source_leaf_id",
            "mpo_source_leaf_id",
            "hpo_inheritance_path",
            "mpo_inheritance_path",
            "hpo_publication_ids",
            "mpo_publication_ids",
        ]
    ].rename(
        columns={
            "hpo_structure_id": "best_hpo_structure_id",
            "mpo_structure_id": "best_mpo_structure_id",
            "hpo_cl_id": "best_hpo_cl_id",
            "mpo_cl_id": "best_mpo_cl_id",
            "hpo_go_id": "best_hpo_go_id",
            "mpo_go_id": "best_mpo_go_id",
            "cl_similarity": "best_cl_similarity",
            "go_similarity": "best_go_similarity",
            "cpo_pair_score": "best_cpo_pair_score",
            "direction_status": "best_pair_direction_status",
        }
    )
    table = table.merge(evidence, on=["hpo_id", "mpo_id"], how="left", validate="one_to_one")

    reference = _reference(config)
    table["mhmi_reference_match_type"] = [
        reference.get((hpo_id, mpo_id), "")
        for hpo_id, mpo_id in zip(table["hpo_id"], table["mpo_id"])
    ]
    table["in_mhmi_reference"] = np.where(
        table["mhmi_reference_match_type"] != "", "true", "false"
    )
    table["review_flags"] = table.apply(_flags, axis=1)
    table = table[list(OUTPUT_COLUMNS)].sort_values(
        [
            "reciprocal_rank_max",
            "reciprocal_rank_sum",
            "score_shape",
            "hpo_id",
            "mpo_id",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    if len(table) != 6505:
        raise AssertionError(f"unexpected mutual Top-5 row count: {len(table)}")
    if table.duplicated(["hpo_id", "mpo_id"]).any():
        raise AssertionError("duplicate HPO-MPO pairs in mutual Top-5 table")
    if not (
        (table["hpo_to_mpo_rank"] <= 5) & (table["mpo_to_hpo_rank"] <= 5)
    ).all():
        raise AssertionError("non-reciprocal Top-5 row reached output")

    tsv_path = mapping_root / "hpo_mpo_mutual_top5.tsv"
    parquet_path = mapping_root / "hpo_mpo_mutual_top5.parquet"
    summary_path = mapping_root / "hpo_mpo_mutual_top5_summary.json"
    table.to_csv(tsv_path, sep="\t", index=False)
    table.to_parquet(parquet_path, index=False)
    summary = {
        "definition": "rank<=5 in both HPO_to_MPO and MPO_to_HPO topk_review views",
        "row_count": len(table),
        "unique_hpo_count": int(table["hpo_id"].nunique()),
        "unique_mpo_count": int(table["mpo_id"].nunique()),
        "reciprocal_tier_counts": dict(Counter(table["reciprocal_tier"])),
        "direction_status_counts": dict(Counter(table["direction_status"])),
        "mhmi_reference_pair_count": int((table["in_mhmi_reference"] == "true").sum()),
        "missing_required_self_pair_count": int(
            table["missing_required_self_flags"].str.contains("true").sum()
        ),
        "direction_conflict_pair_count": int(
            table["score_direction_conflict"].fillna(0).gt(0).sum()
        ),
        "output_tsv": str(tsv_path.resolve()),
        "output_parquet": str(parquet_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reciprocal HPO-MPO Top-5 pairs")
    parser.add_argument("--config", default="config/cpo_v2.yaml")
    args = parser.parse_args()
    print(json.dumps(build_table(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
