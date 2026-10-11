from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa

from .candidates import ParquetRecordWriter, _rank, _schema
from .config import output_root, project_path
from .io import read_tsv, write_json
from .matrix import CHANNELS, load_prepared_scoring


BRANCH_FIELDS = (
    "hpo_id",
    "mpo_ancestor_id",
    "channel",
    "rank",
    "branch_score",
    "self_shape",
    "best_leaf_id",
    "n_best_leaf_ties",
    "n_descendant_leaves",
    "n_scoreable_leaves",
    "source_assertion_ids",
    "provenance_ids",
    "ancestor_node_class",
    "interpretation_type",
)
BRANCH_SCHEMA = _schema(
    BRANCH_FIELDS,
    {"branch_score", "self_shape"},
    {
        "rank",
        "n_best_leaf_ties",
        "n_descendant_leaves",
        "n_scoreable_leaves",
    },
)


def run_branch(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    _, provenance = read_tsv(
        project_path(config, config["inputs"]["inheritance_provenance"])
    )
    descendant_leaves: dict[int, set[int]] = defaultdict(set)
    relation_ids: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for row in provenance:
        target_index = mpo_index[row["target_mp_id"]]
        leaf_index = mpo_index[row["source_leaf_mp_id"]]
        descendant_leaves[target_index].add(leaf_index)
        relation_ids[(target_index, leaf_index, row["source_relation_scope"])].add(
            row["source_relation_id"]
        )
    assertions = pd.read_parquet(
        root / "normalized/assertions.parquet",
        columns=["assertion_id", "ontology_side", "origin", "source_record_id"],
    )
    assertions = assertions[
        (assertions["ontology_side"] == "MPO") & (assertions["origin"] == "self")
    ]
    assertion_to_raw = dict(
        zip(assertions["assertion_id"].astype(str), assertions["source_record_id"].astype(str))
    )
    links = pd.read_parquet(root / "normalized/cpo_node_assertions.parquet")
    node_assertions: dict[str, set[str]] = defaultdict(set)
    for row in links.itertuples(index=False):
        assertion_id = str(row.assertion_id)
        if assertion_id in assertion_to_raw:
            node_assertions[str(row.cpo_node_id)].add(assertion_id)

    candidate_targets = np.asarray(
        sorted(
            index
            for index in descendant_leaves
            if prepared.mpo_classes[index] == "middle"
        ),
        dtype=np.int32,
    )
    top_k = int(config["candidates"]["top_k"])
    writer = ParquetRecordWriter(
        root / "candidates/branch_support_topk.parquet", BRANCH_SCHEMA
    )

    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for channel, (h_class, m_class) in CHANNELS.items():
            h_profiles = prepared.profiles[("HPO", h_class)]
            m_profiles = prepared.profiles[("MPO", m_class)]
            scoreable_leaf = {
                index
                for index, profile in enumerate(m_profiles)
                if profile.size and prepared.mpo_classes[index] == "leaf"
            }
            grouped_leaves = [
                sorted(descendant_leaves[int(target)] & scoreable_leaf)
                for target in candidate_targets
            ]
            active_positions = np.asarray(
                [index for index, values in enumerate(grouped_leaves) if values],
                dtype=np.int32,
            )
            active_targets = candidate_targets[active_positions]
            active_groups = [grouped_leaves[index] for index in active_positions]
            if not active_groups:
                continue
            lengths = np.asarray([len(values) for values in active_groups], dtype=np.int32)
            offsets = np.concatenate((np.asarray([0]), np.cumsum(lengths)))
            flat_leaf_indices = np.asarray(
                [leaf for values in active_groups for leaf in values], dtype=np.int32
            )
            shape_dataset = handle[channel]["shape"]
            for h_index, h_profile in enumerate(h_profiles):
                if not h_profile.size:
                    continue
                row_scores = shape_dataset[h_index, :]
                branch_scores = np.fmax.reduceat(
                    row_scores[flat_leaf_indices], offsets[:-1]
                )
                full = np.full(len(prepared.mpo_ids), np.nan, dtype=np.float64)
                full[active_targets] = branch_scores
                ranked, cutoff_ties, cutoff_truncated = _rank(
                    full,
                    full,
                    prepared.mpo_ids,
                    np.asarray(
                        [kind == "middle" for kind in prepared.mpo_classes], dtype=bool
                    ),
                    max(top_k * 20, top_k),
                )
                selected_rows: list[tuple[int, int, int]] = []
                used_best_leaves: set[int] = set()
                for target_index in ranked:
                    leaves = sorted(descendant_leaves[target_index] & scoreable_leaf)
                    scores = row_scores[leaves]
                    best = float(np.nanmax(scores))
                    tied = [
                        leaf
                        for leaf, score in zip(leaves, scores)
                        if np.isfinite(score) and float(score) == best
                    ]
                    if not tied:
                        continue
                    best_leaf = min(tied, key=lambda index: prepared.mpo_ids[index])
                    if best_leaf in used_best_leaves:
                        continue
                    used_best_leaves.add(best_leaf)
                    selected_rows.append((target_index, best_leaf, len(tied)))
                    if len(selected_rows) == top_k:
                        break
                for rank, (target_index, best_leaf, tie_count) in enumerate(
                    selected_rows, start=1
                ):
                    scope = m_class
                    raw_scope = "direct" if scope == "D" else "mechanism"
                    participating_assertions = sorted(
                        {
                            assertion_id
                            for node_id in m_profiles[best_leaf].node_ids
                            for assertion_id in node_assertions.get(node_id, set())
                        }
                    )
                    participating_raw = {
                        assertion_to_raw[assertion_id]
                        for assertion_id in participating_assertions
                    }
                    sources = sorted(
                        relation_ids.get((target_index, best_leaf, raw_scope), set())
                        & participating_raw
                    )
                    writer.add(
                        {
                            "hpo_id": prepared.hpo_ids[h_index],
                            "mpo_ancestor_id": prepared.mpo_ids[target_index],
                            "channel": channel,
                            "rank": rank,
                            "branch_score": float(full[target_index]),
                            "self_shape": float(row_scores[target_index])
                            if np.isfinite(row_scores[target_index])
                            else float("nan"),
                            "best_leaf_id": prepared.mpo_ids[best_leaf],
                            "n_best_leaf_ties": tie_count,
                            "n_descendant_leaves": len(descendant_leaves[target_index]),
                            "n_scoreable_leaves": len(
                                descendant_leaves[target_index] & scoreable_leaf
                            ),
                            "source_assertion_ids": "|".join(participating_assertions),
                            "provenance_ids": "|".join(
                                f"{prepared.mpo_ids[target_index]}:{prepared.mpo_ids[best_leaf]}:{source}"
                                for source in sources
                            ),
                            "ancestor_node_class": prepared.mpo_classes[target_index],
                            "interpretation_type": "branch_supported_candidate",
                        }
                    )
    count = writer.close()
    result = {"branch_candidate_count": count}
    write_json(root / "logs/branch_summary.json", result)
    return result
