from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from cpo_mapping.candidates import ParquetRecordWriter, _rank, _schema
from cpo_mapping.io import read_tsv, write_json

from .config import output_root
from .prepare import stable_id
from .term_matrix import _cpo_similarity, load_term_data


MAPPING_FIELDS = (
    "query_side",
    "hpo_id",
    "mpo_id",
    "hpo_node_type",
    "mpo_node_type",
    "view",
    "rank",
    "score_shape",
    "hpo_to_mpo",
    "mpo_to_hpo",
    "score_support",
    "score_conflict",
    "n_hpo_cpo",
    "n_mpo_cpo",
    "positive_match_coverage_hpo",
    "positive_match_coverage_mpo",
    "direction_comparable_fraction_hpo",
    "direction_comparable_fraction_mpo",
    "source_origin_summary",
    "evidence_class_summary",
    "missing_required_self_flags",
    "source_leaf_counts",
    "heterogeneity_flags",
    "profile_identity_group",
    "status",
    "ties_at_cutoff",
    "ties_truncated",
)
MAPPING_SCHEMA = _schema(
    MAPPING_FIELDS,
    {
        "score_shape",
        "hpo_to_mpo",
        "mpo_to_hpo",
        "score_support",
        "score_conflict",
        "positive_match_coverage_hpo",
        "positive_match_coverage_mpo",
        "direction_comparable_fraction_hpo",
        "direction_comparable_fraction_mpo",
    },
    {
        "rank",
        "n_hpo_cpo",
        "n_mpo_cpo",
        "ties_at_cutoff",
        "ties_truncated",
    },
)

AUX_FIELDS = (
    "query_side",
    "hpo_id",
    "mpo_id",
    "dimension",
    "rank",
    "score",
    "hpo_node_type",
    "mpo_node_type",
    "view",
    "status",
    "ties_at_cutoff",
    "ties_truncated",
)
AUX_SCHEMA = _schema(
    AUX_FIELDS,
    {"score"},
    {"rank", "ties_at_cutoff", "ties_truncated"},
)


class MappingContext:
    def __init__(self, config: dict[str, Any]) -> None:
        self.root = output_root(config)
        self.data = load_term_data(config)
        _, self.hpo_policy = read_tsv(self.root / "indices/hpo_index.tsv")
        _, self.mpo_policy = read_tsv(self.root / "indices/mpo_index.tsv")
        self.hpo_ids = self.data.hpo_ids
        self.mpo_ids = self.data.mpo_ids
        self.hpo_index = {value: index for index, value in enumerate(self.hpo_ids)}
        self.mpo_index = {value: index for index, value in enumerate(self.mpo_ids)}
        h_structures, m_structures, similarity, _ = _cpo_similarity(self.data)
        self.h_structures = h_structures
        self.m_structures = m_structures
        self.h_structure_index = {value: index for index, value in enumerate(h_structures)}
        self.m_structure_index = {value: index for index, value in enumerate(m_structures)}
        self.similarity = similarity
        final = self.data.memberships[self.data.memberships["profile_view"] == "final"]
        self.h_profiles = self._profiles(final, "HPO", self.hpo_ids)
        self.m_profiles = self._profiles(final, "MPO", self.mpo_ids)

    @staticmethod
    def _profiles(
        frame: pd.DataFrame, side: str, ids: list[str]
    ) -> list[list[dict[str, object]]]:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in frame[frame["ontology_side"] == side].to_dict("records"):
            groups[str(row["term_id"])].append(row)
        return [
            sorted(groups.get(term_id, []), key=lambda row: str(row["structure_id"]))
            for term_id in ids
        ]


def _direction_code(left: dict[str, object], right: dict[str, object]) -> int:
    if (
        left["structure_id"] != right["structure_id"]
        or left["direction_signature_status"] != "unique"
        or right["direction_signature_status"] != "unique"
        or (left["direction_target"], left["direction_property"])
        != (right["direction_target"], right["direction_property"])
    ):
        return 0
    pair = (left["direction_state"], right["direction_state"])
    if pair[0] == pair[1] and pair[0] in {"increased", "decreased", "present", "absent"}:
        return 1
    if pair in {
        ("increased", "decreased"),
        ("decreased", "increased"),
        ("present", "absent"),
        ("absent", "present"),
    }:
        return -1
    return 0


def _origin_summary(values: list[dict[str, object]]) -> str:
    origins = {
        origin
        for value in values
        for origin in str(value["origin_set"]).split("|")
        if origin
    }
    if origins == {"self"}:
        return "self_only"
    if origins == {"inherited"}:
        return "inherited_only"
    return "both"


def _pair_details(context: MappingContext, h_index: int, m_index: int) -> dict[str, object]:
    h_profile = context.h_profiles[h_index]
    m_profile = context.m_profiles[m_index]
    h_components = np.asarray(
        [context.h_structure_index[str(value["structure_id"])] for value in h_profile],
        dtype=np.int32,
    )
    m_components = np.asarray(
        [context.m_structure_index[str(value["structure_id"])] for value in m_profile],
        dtype=np.int32,
    )
    pair = context.similarity[np.ix_(h_components, m_components)]
    positive = pair > 0
    directions = np.asarray(
        [[_direction_code(left, right) for right in m_profile] for left in h_profile],
        dtype=np.int8,
    )
    comparable = positive & np.isin(directions, (1, -1))
    origin_summary = f"HPO:{_origin_summary(h_profile)};MPO:{_origin_summary(m_profile)}"
    evidence = sorted(
        {
            item
            for value in (*h_profile, *m_profile)
            for item in str(value["evidence_class_set"]).split("|")
            if item
        }
    )
    heterogeneity = sorted(
        {
            str(value["direction_signature_status"])
            for value in (*h_profile, *m_profile)
            if value["direction_signature_status"] != "unique"
        }
    )
    h_policy = context.hpo_policy[h_index]
    m_policy = context.mpo_policy[m_index]
    return {
        "n_hpo_cpo": len(h_profile),
        "n_mpo_cpo": len(m_profile),
        "positive_match_coverage_hpo": float(np.any(positive, axis=1).mean()),
        "positive_match_coverage_mpo": float(np.any(positive, axis=0).mean()),
        "direction_comparable_fraction_hpo": float(np.any(comparable, axis=1).mean()),
        "direction_comparable_fraction_mpo": float(np.any(comparable, axis=0).mean()),
        "source_origin_summary": origin_summary,
        "evidence_class_summary": "|".join(evidence),
        "missing_required_self_flags": (
            f"HPO:{h_policy['missing_required_self_evidence']};"
            f"MPO:{m_policy['missing_required_self_evidence']}"
        ),
        "source_leaf_counts": (
            f"HPO:{max((int(value['source_leaf_count']) for value in h_profile), default=0)};"
            f"MPO:{max((int(value['source_leaf_count']) for value in m_profile), default=0)}"
        ),
        "heterogeneity_flags": "|".join(heterogeneity),
        "profile_identity_group": stable_id(
            "profile_pair",
            "|".join(str(value["structure_id"]) for value in h_profile),
            "|".join(str(value["structure_id"]) for value in m_profile),
        ),
        "status": "OK",
    }


def _candidate_row(
    context: MappingContext,
    query_side: str,
    h_index: int,
    m_index: int,
    view: str,
    rank: int,
    scores: dict[str, float],
    ties: int,
    truncated: int,
) -> dict[str, object]:
    return {
        "query_side": query_side,
        "hpo_id": context.hpo_ids[h_index],
        "mpo_id": context.mpo_ids[m_index],
        "hpo_node_type": context.hpo_policy[h_index]["node_type"],
        "mpo_node_type": context.mpo_policy[m_index]["node_type"],
        "view": view,
        "rank": rank,
        **scores,
        **_pair_details(context, h_index, m_index),
        "ties_at_cutoff": ties,
        "ties_truncated": truncated,
    }


def run_mapping(config: dict[str, Any]) -> dict[str, int]:
    context = MappingContext(config)
    root = context.root
    top_k = int(config["mapping"]["top_k"])
    h_writer = ParquetRecordWriter(root / "mapping/hpo_to_mpo.parquet", MAPPING_SCHEMA)
    m_writer = ParquetRecordWriter(root / "mapping/mpo_to_hpo.parquet", MAPPING_SCHEMA)
    review_h = np.asarray(
        [row["node_type"] != "ROOT_LAYER" for row in context.hpo_policy], dtype=bool
    )
    review_m = np.asarray(
        [row["node_type"] != "ROOT_LAYER" for row in context.mpo_policy], dtype=bool
    )
    all_h = np.ones(len(context.hpo_ids), dtype=bool)
    all_m = np.ones(len(context.mpo_ids), dtype=bool)
    h_block = int(config["matrix"]["hpo_block_size"])
    m_block = int(config["matrix"]["mpo_block_size"])
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        shape_ds = handle["final/shape"]
        f_ds = handle["final/hpo_to_mpo"]
        r_ds = handle["final/mpo_to_hpo"]
        support_ds = handle["final/direction_support"]
        conflict_ds = handle["final/direction_conflict"]
        for start in range(0, len(context.hpo_ids), h_block):
            end = min(len(context.hpo_ids), start + h_block)
            shape = shape_ds[start:end, :]
            forward = f_ds[start:end, :]
            reverse = r_ds[start:end, :]
            support = support_ds[start:end, :]
            conflict = conflict_ds[start:end, :]
            for offset, h_index in enumerate(range(start, end)):
                for view, primary, secondary, allowed in (
                    ("topk_all", shape[offset], shape[offset], all_m),
                    ("topk_review", shape[offset], shape[offset], review_m),
                    ("direction_supported_topk", support[offset], shape[offset], all_m),
                ):
                    selected, ties, truncated = _rank(
                        primary, secondary, context.mpo_ids, allowed, top_k
                    )
                    for rank, m_index in enumerate(selected, start=1):
                        h_writer.add(
                            _candidate_row(
                                context,
                                "HPO",
                                h_index,
                                m_index,
                                view,
                                rank,
                                {
                                    "score_shape": float(shape[offset, m_index]),
                                    "hpo_to_mpo": float(forward[offset, m_index]),
                                    "mpo_to_hpo": float(reverse[offset, m_index]),
                                    "score_support": float(support[offset, m_index]),
                                    "score_conflict": float(conflict[offset, m_index]),
                                },
                                ties,
                                truncated,
                            )
                        )
        for start in range(0, len(context.mpo_ids), m_block):
            end = min(len(context.mpo_ids), start + m_block)
            shape = shape_ds[:, start:end]
            forward = f_ds[:, start:end]
            reverse = r_ds[:, start:end]
            support = support_ds[:, start:end]
            conflict = conflict_ds[:, start:end]
            for offset, m_index in enumerate(range(start, end)):
                for view, primary, secondary, allowed in (
                    ("topk_all", shape[:, offset], shape[:, offset], all_h),
                    ("topk_review", shape[:, offset], shape[:, offset], review_h),
                    ("direction_supported_topk", support[:, offset], shape[:, offset], all_h),
                ):
                    selected, ties, truncated = _rank(
                        primary, secondary, context.hpo_ids, allowed, top_k
                    )
                    for rank, h_index in enumerate(selected, start=1):
                        m_writer.add(
                            _candidate_row(
                                context,
                                "MPO",
                                h_index,
                                m_index,
                                view,
                                rank,
                                {
                                    "score_shape": float(shape[h_index, offset]),
                                    "hpo_to_mpo": float(forward[h_index, offset]),
                                    "mpo_to_hpo": float(reverse[h_index, offset]),
                                    "score_support": float(support[h_index, offset]),
                                    "score_conflict": float(conflict[h_index, offset]),
                                },
                                ties,
                                truncated,
                            )
                        )
    counts = {
        "hpo_to_mpo_rows": h_writer.close(),
        "mpo_to_hpo_rows": m_writer.close(),
    }
    counts["auxiliary_rows"] = run_auxiliary_mapping(config, context)
    write_json(root / "mapping/mapping_manifest.json", counts)
    return counts


def run_auxiliary_mapping(config: dict[str, Any], context: MappingContext) -> int:
    root = context.root
    writer = ParquetRecordWriter(root / "mapping/auxiliary_topk.parquet", AUX_SCHEMA)
    top_k = int(config["mapping"]["top_k"])
    review_h = np.asarray(
        [row["node_type"] != "ROOT_LAYER" for row in context.hpo_policy], dtype=bool
    )
    review_m = np.asarray(
        [row["node_type"] != "ROOT_LAYER" for row in context.mpo_policy], dtype=bool
    )
    h_block = int(config["matrix"]["hpo_block_size"])
    m_block = int(config["matrix"]["mpo_block_size"])
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for dimension in ("cl", "go"):
            dataset = handle[f"aux/{dimension}"]
            for start in range(0, len(context.hpo_ids), h_block):
                end = min(len(context.hpo_ids), start + h_block)
                block = dataset[start:end, :]
                for offset, h_index in enumerate(range(start, end)):
                    values = block[offset]
                    selected, ties, truncated = _rank(
                        values, values, context.mpo_ids, review_m, top_k
                    )
                    for rank, m_index in enumerate(selected, start=1):
                        writer.add(
                            {
                                "query_side": "HPO",
                                "hpo_id": context.hpo_ids[h_index],
                                "mpo_id": context.mpo_ids[m_index],
                                "dimension": dimension.upper(),
                                "rank": rank,
                                "score": float(values[m_index]),
                                "hpo_node_type": context.hpo_policy[h_index]["node_type"],
                                "mpo_node_type": context.mpo_policy[m_index]["node_type"],
                                "view": "topk_review",
                                "status": "OK",
                                "ties_at_cutoff": ties,
                                "ties_truncated": truncated,
                            }
                        )
            for start in range(0, len(context.mpo_ids), m_block):
                end = min(len(context.mpo_ids), start + m_block)
                block = dataset[:, start:end]
                for offset, m_index in enumerate(range(start, end)):
                    values = block[:, offset]
                    selected, ties, truncated = _rank(
                        values, values, context.hpo_ids, review_h, top_k
                    )
                    for rank, h_index in enumerate(selected, start=1):
                        writer.add(
                            {
                                "query_side": "MPO",
                                "hpo_id": context.hpo_ids[h_index],
                                "mpo_id": context.mpo_ids[m_index],
                                "dimension": dimension.upper(),
                                "rank": rank,
                                "score": float(values[h_index]),
                                "hpo_node_type": context.hpo_policy[h_index]["node_type"],
                                "mpo_node_type": context.mpo_policy[m_index]["node_type"],
                                "view": "topk_review",
                                "status": "OK",
                                "ties_at_cutoff": ties,
                                "ties_truncated": truncated,
                            }
                        )
    return writer.close()
