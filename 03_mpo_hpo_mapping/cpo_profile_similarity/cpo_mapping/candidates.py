from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import output_root
from .io import read_tsv, write_json
from .matrix import (
    CHANNELS,
    PreparedScoring,
    TermProfile,
    direction_matrix,
    load_prepared_scoring,
    score_term_pair,
)
from .scoring import directional_bma


MAIN_FIELDS = (
    "query_side",
    "query_id",
    "target_id",
    "channel",
    "view",
    "rank",
    "shape",
    "support",
    "conflict",
    "shape_forward_mean",
    "shape_reverse_mean",
    "support_forward_mean",
    "support_reverse_mean",
    "conflict_forward_mean",
    "conflict_reverse_mean",
    "n_hpo_cpo",
    "n_mpo_cpo",
    "n_hpo_direction_comparable",
    "n_mpo_direction_comparable",
    "fraction_hpo_direction_comparable",
    "fraction_mpo_direction_comparable",
    "n_agree_pairs",
    "n_conflict_pairs",
    "n_mixed_direction_nodes",
    "context_flags",
    "shape_status",
    "direction_status",
    "interpretation_type",
    "target_node_class",
    "ties_at_cutoff",
    "ties_truncated",
)

AUX_FIELDS = (
    "query_side",
    "query_id",
    "target_id",
    "channel",
    "dimension",
    "rank",
    "score",
    "forward_mean",
    "reverse_mean",
    "n_hpo_components",
    "n_mpo_components",
    "target_node_class",
    "interpretation_type",
    "ties_at_cutoff",
    "ties_truncated",
)


def _schema(fields: Sequence[str], float_fields: set[str], int_fields: set[str]) -> pa.Schema:
    values = []
    for field in fields:
        if field in float_fields:
            dtype = pa.float64()
        elif field in int_fields:
            dtype = pa.int64()
        else:
            dtype = pa.string()
        values.append(pa.field(field, dtype))
    return pa.schema(values)


MAIN_SCHEMA = _schema(
    MAIN_FIELDS,
    {
        "shape",
        "support",
        "conflict",
        "shape_forward_mean",
        "shape_reverse_mean",
        "support_forward_mean",
        "support_reverse_mean",
        "conflict_forward_mean",
        "conflict_reverse_mean",
        "fraction_hpo_direction_comparable",
        "fraction_mpo_direction_comparable",
    },
    {
        "rank",
        "n_hpo_cpo",
        "n_mpo_cpo",
        "n_hpo_direction_comparable",
        "n_mpo_direction_comparable",
        "n_agree_pairs",
        "n_conflict_pairs",
        "n_mixed_direction_nodes",
        "ties_at_cutoff",
        "ties_truncated",
    },
)
AUX_SCHEMA = _schema(
    AUX_FIELDS,
    {"score", "forward_mean", "reverse_mean"},
    {
        "rank",
        "n_hpo_components",
        "n_mpo_components",
        "ties_at_cutoff",
        "ties_truncated",
    },
)


class ParquetRecordWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 10000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.buffer: list[dict[str, object]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

    def add(self, row: dict[str, object]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        columns = {
            field.name: [row.get(field.name) for row in self.buffer]
            for field in self.schema
        }
        self.writer.write_table(pa.Table.from_pydict(columns, schema=self.schema))
        self.count += len(self.buffer)
        self.buffer.clear()

    def close(self) -> int:
        self.flush()
        self.writer.close()
        self.path.chmod(0o644)
        return self.count


def _rank(
    primary: np.ndarray,
    secondary: np.ndarray,
    ids: Sequence[str],
    allowed: np.ndarray,
    top_k: int,
) -> tuple[list[int], int, int]:
    valid = np.flatnonzero(allowed & np.isfinite(primary) & (primary > 0))
    if not len(valid):
        return [], 0, 0
    id_values = np.asarray(ids, dtype=object)
    order = np.lexsort((id_values[valid], -secondary[valid], -primary[valid]))
    ranked = valid[order]
    selected = ranked[:top_k].tolist()
    if len(ranked) <= top_k:
        return selected, 0, 0
    cutoff = primary[selected[-1]]
    ties = int(np.count_nonzero(primary[valid] == cutoff))
    selected_ties = sum(primary[index] == cutoff for index in selected)
    return selected, ties, max(0, ties - selected_ties)


def _main_diagnostics(
    prepared: PreparedScoring,
    hpo_index: int,
    mpo_index: int,
    channel: str,
) -> dict[str, object]:
    h_class, m_class = CHANNELS[channel]
    left = prepared.profiles[("HPO", h_class)][hpo_index]
    right = prepared.profiles[("MPO", m_class)][mpo_index]
    result = score_term_pair(prepared, hpo_index, mpo_index, channel)
    if result is None:
        raise ValueError("candidate pair has an empty profile")
    shape_matrix = prepared.cpo_similarity[np.ix_(left.cpo_indices, right.cpo_indices)]
    directions = direction_matrix(left, right)
    comparable = (shape_matrix > 0) & np.isin(directions, (1, -1))
    h_comparable = int(np.any(comparable, axis=1).sum())
    m_comparable = int(np.any(comparable, axis=0).sum())
    return {
        "shape": result.shape,
        "support": result.support,
        "conflict": result.conflict,
        "shape_forward_mean": result.shape_forward,
        "shape_reverse_mean": result.shape_reverse,
        "support_forward_mean": result.support_forward,
        "support_reverse_mean": result.support_reverse,
        "conflict_forward_mean": result.conflict_forward,
        "conflict_reverse_mean": result.conflict_reverse,
        "n_hpo_cpo": left.size,
        "n_mpo_cpo": right.size,
        "n_hpo_direction_comparable": h_comparable,
        "n_mpo_direction_comparable": m_comparable,
        "fraction_hpo_direction_comparable": h_comparable / left.size,
        "fraction_mpo_direction_comparable": m_comparable / right.size,
        "n_agree_pairs": int(np.count_nonzero((shape_matrix > 0) & (directions == 1))),
        "n_conflict_pairs": int(np.count_nonzero((shape_matrix > 0) & (directions == -1))),
        "n_mixed_direction_nodes": int(np.count_nonzero(left.signature_codes == 2))
        + int(np.count_nonzero(right.signature_codes == 2)),
        "context_flags": "context_unverified" if result.comparable else "",
        "shape_status": "OK",
        "direction_status": "OK" if result.comparable else "NO_COMPARABLE_DIRECTION",
    }


def _interpretation(channel: str, view: str) -> str:
    if view == "direction_supported_topk":
        return "direction_supported_candidate"
    if "M" in channel:
        return "mechanism_related_candidate"
    return "structural_candidate"


def run_main_candidates(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    top_k = int(config["candidates"]["top_k"])
    prepared = load_prepared_scoring(config)
    allowed_mpo = np.asarray(
        [kind not in {"root", "top_level", "normal"} for kind in prepared.mpo_classes],
        dtype=bool,
    )
    allowed_hpo = np.ones(len(prepared.hpo_ids), dtype=bool)
    forward_writer = ParquetRecordWriter(
        root / "candidates/hpo_to_mpo_topk.parquet", MAIN_SCHEMA
    )
    reverse_writer = ParquetRecordWriter(
        root / "candidates/mpo_to_hpo_topk.parquet", MAIN_SCHEMA
    )

    def emit(
        writer: ParquetRecordWriter,
        query_side: str,
        h_index: int,
        m_index: int,
        channel: str,
        view: str,
        rank: int,
        ties: int,
        truncated: int,
    ) -> None:
        diagnostics = _main_diagnostics(prepared, h_index, m_index, channel)
        writer.add(
            {
                "query_side": query_side,
                "query_id": prepared.hpo_ids[h_index] if query_side == "HPO" else prepared.mpo_ids[m_index],
                "target_id": prepared.mpo_ids[m_index] if query_side == "HPO" else prepared.hpo_ids[h_index],
                "channel": channel,
                "view": view,
                "rank": rank,
                **diagnostics,
                "interpretation_type": _interpretation(channel, view),
                "target_node_class": prepared.mpo_classes[m_index] if query_side == "HPO" else "leaf",
                "ties_at_cutoff": ties,
                "ties_truncated": truncated,
            }
        )

    block_size = int(config["runtime"]["hpo_block_size"])
    mpo_block_size = int(config["runtime"].get("mpo_block_size", 512))
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for channel in CHANNELS:
            shape_dataset = handle[channel]["shape"]
            support_dataset = handle[channel]["support"]
            conflict_dataset = handle[channel]["conflict"]
            for start in range(0, len(prepared.hpo_ids), block_size):
                end = min(len(prepared.hpo_ids), start + block_size)
                shape_block = shape_dataset[start:end, :]
                support_block = support_dataset[start:end, :]
                for offset, h_index in enumerate(range(start, end)):
                    for view, primary, secondary in (
                        ("shape_topk", shape_block[offset], shape_block[offset]),
                        ("direction_supported_topk", support_block[offset], shape_block[offset]),
                    ):
                        selected, ties, truncated = _rank(
                            primary, secondary, prepared.mpo_ids, allowed_mpo, top_k
                        )
                        for rank, m_index in enumerate(selected, start=1):
                            emit(
                                forward_writer,
                                "HPO",
                                h_index,
                                m_index,
                                channel,
                                view,
                                rank,
                                ties,
                                truncated,
                            )

            for m_start in range(0, len(prepared.mpo_ids), mpo_block_size):
                m_end = min(len(prepared.mpo_ids), m_start + mpo_block_size)
                shape_block = shape_dataset[:, m_start:m_end]
                support_block = support_dataset[:, m_start:m_end]
                for offset, m_index in enumerate(range(m_start, m_end)):
                    for view, primary, secondary in (
                        ("shape_topk", shape_block[:, offset], shape_block[:, offset]),
                        ("direction_supported_topk", support_block[:, offset], shape_block[:, offset]),
                    ):
                        selected, ties, truncated = _rank(
                            primary, secondary, prepared.hpo_ids, allowed_hpo, top_k
                        )
                        for rank, h_index in enumerate(selected, start=1):
                            emit(
                                reverse_writer,
                                "MPO",
                                h_index,
                                m_index,
                                channel,
                                view,
                                rank,
                                ties,
                                truncated,
                            )
    counts = {
        "hpo_to_mpo_candidate_count": forward_writer.close(),
        "mpo_to_hpo_candidate_count": reverse_writer.close(),
    }
    write_json(root / "logs/main_candidates_summary.json", counts)
    return counts


def _projection_profiles(
    config: dict[str, Any],
    prepared: PreparedScoring,
    dimension: str,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    dict[tuple[str, str], list[np.ndarray]],
]:
    assertions = pd.read_parquet(output_root(config) / "normalized/assertions.parquet")
    assertions = assertions[
        (assertions["origin"] == "self")
        & (assertions["statement_status"] == "asserted")
        & (assertions["evidence_class"].isin(["D", "M"]))
    ]
    column = "cl_id" if dimension == "CL" else "go_id"
    root_id = prepared.cl.root_id if dimension == "CL" else prepared.go.root_id
    assertions = assertions[(assertions[column] != "") & (assertions[column] != root_id)]
    h_ids = sorted(set(assertions.loc[assertions["ontology_side"] == "HPO", column]))
    m_ids = sorted(set(assertions.loc[assertions["ontology_side"] == "MPO", column]))
    ontology = prepared.cl if dimension == "CL" else prepared.go
    similarity = np.empty((len(h_ids), len(m_ids)), dtype=np.float64)
    for left_index, left_id in enumerate(h_ids):
        for right_index, right_id in enumerate(m_ids):
            similarity[left_index, right_index] = ontology.lin(str(left_id), str(right_id))
    indices = {
        "HPO": {term_id: index for index, term_id in enumerate(h_ids)},
        "MPO": {term_id: index for index, term_id in enumerate(m_ids)},
    }
    term_indices = {
        "HPO": {term_id: index for index, term_id in enumerate(prepared.hpo_ids)},
        "MPO": {term_id: index for index, term_id in enumerate(prepared.mpo_ids)},
    }
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in assertions[["ontology_side", "term_id", "evidence_class", column]].itertuples(index=False):
        side = str(row.ontology_side)
        grouped[(side, str(row.evidence_class), str(row.term_id))].add(
            indices[side][str(getattr(row, column))]
        )
    profiles: dict[tuple[str, str], list[np.ndarray]] = {}
    for side, ids in (("HPO", prepared.hpo_ids), ("MPO", prepared.mpo_ids)):
        for evidence in ("D", "M"):
            profiles[(side, evidence)] = [
                np.asarray(sorted(grouped.get((side, evidence, term_id), set())), dtype=np.int32)
                for term_id in ids
            ]
    return h_ids, m_ids, similarity, profiles


def _aux_score_matrix(
    h_profiles: list[np.ndarray],
    m_profiles: list[np.ndarray],
    component_similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_indices = np.asarray([i for i, profile in enumerate(h_profiles) if len(profile)], dtype=np.int32)
    m_indices = np.asarray([i for i, profile in enumerate(m_profiles) if len(profile)], dtype=np.int32)
    lengths = np.asarray([len(m_profiles[i]) for i in m_indices], dtype=np.int32)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.concatenate([m_profiles[i] for i in m_indices])
    scores = np.empty((len(h_indices), len(m_indices)), dtype=np.float32)
    for output_index, h_index in enumerate(h_indices):
        pair = component_similarity[np.ix_(h_profiles[h_index], flat)]
        forward = np.maximum.reduceat(pair, starts[:-1], axis=1).mean(axis=0)
        reverse = np.add.reduceat(pair.max(axis=0), starts[:-1]) / lengths
        scores[output_index] = (forward + reverse) / 2
    return h_indices, m_indices, scores, lengths


def _aux_pair_detail(
    h_profile: np.ndarray, m_profile: np.ndarray, similarity: np.ndarray
) -> tuple[float, float, float]:
    matrix = similarity[np.ix_(h_profile, m_profile)]
    return directional_bma(matrix)


def run_auxiliary_candidates(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    top_k = int(config["candidates"]["top_k"])
    writer = ParquetRecordWriter(root / "candidates/auxiliary_topk.parquet", AUX_SCHEMA)
    allowed_mpo = np.asarray(
        [kind not in {"root", "top_level", "normal"} for kind in prepared.mpo_classes], dtype=bool
    )
    for dimension in ("CL", "GO"):
        _h_components, _m_components, similarity, profiles = _projection_profiles(
            config, prepared, dimension
        )
        for channel, (h_class, m_class) in CHANNELS.items():
            h_profiles = profiles[("HPO", h_class)]
            m_profiles = profiles[("MPO", m_class)]
            h_indices, m_indices, scores, _lengths = _aux_score_matrix(
                h_profiles, m_profiles, similarity
            )
            h_local = {int(index): local for local, index in enumerate(h_indices)}
            m_local = {int(index): local for local, index in enumerate(m_indices)}
            for h_index in h_indices:
                local_h = h_local[int(h_index)]
                full = np.full(len(prepared.mpo_ids), np.nan)
                full[m_indices] = scores[local_h]
                selected, ties, truncated = _rank(
                    full, full, prepared.mpo_ids, allowed_mpo, top_k
                )
                for rank, m_index in enumerate(selected, start=1):
                    forward, reverse, score = _aux_pair_detail(
                        h_profiles[int(h_index)], m_profiles[m_index], similarity
                    )
                    writer.add(
                        {
                            "query_side": "HPO",
                            "query_id": prepared.hpo_ids[int(h_index)],
                            "target_id": prepared.mpo_ids[m_index],
                            "channel": channel,
                            "dimension": dimension,
                            "rank": rank,
                            "score": score,
                            "forward_mean": forward,
                            "reverse_mean": reverse,
                            "n_hpo_components": len(h_profiles[int(h_index)]),
                            "n_mpo_components": len(m_profiles[m_index]),
                            "target_node_class": prepared.mpo_classes[m_index],
                            "interpretation_type": "cell_only_similarity" if dimension == "CL" else "function_only_similarity",
                            "ties_at_cutoff": ties,
                            "ties_truncated": truncated,
                        }
                    )
            for m_index in m_indices:
                local_m = m_local[int(m_index)]
                full = np.full(len(prepared.hpo_ids), np.nan)
                full[h_indices] = scores[:, local_m]
                selected, ties, truncated = _rank(
                    full, full, prepared.hpo_ids, np.ones(len(full), dtype=bool), top_k
                )
                for rank, h_index in enumerate(selected, start=1):
                    forward, reverse, score = _aux_pair_detail(
                        h_profiles[h_index], m_profiles[int(m_index)], similarity
                    )
                    writer.add(
                        {
                            "query_side": "MPO",
                            "query_id": prepared.mpo_ids[int(m_index)],
                            "target_id": prepared.hpo_ids[h_index],
                            "channel": channel,
                            "dimension": dimension,
                            "rank": rank,
                            "score": score,
                            "forward_mean": forward,
                            "reverse_mean": reverse,
                            "n_hpo_components": len(h_profiles[h_index]),
                            "n_mpo_components": len(m_profiles[int(m_index)]),
                            "target_node_class": "leaf",
                            "interpretation_type": "cell_only_similarity" if dimension == "CL" else "function_only_similarity",
                            "ties_at_cutoff": ties,
                            "ties_truncated": truncated,
                        }
                    )
    count = writer.close()
    result = {"auxiliary_candidate_count": count}
    write_json(root / "logs/auxiliary_candidates_summary.json", result)
    return result


def run_candidates(config: dict[str, Any]) -> dict[str, int]:
    result = {**run_main_candidates(config), **run_auxiliary_candidates(config)}
    write_json(output_root(config) / "logs/candidates_summary.json", result)
    return result
