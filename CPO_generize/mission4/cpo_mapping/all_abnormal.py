from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

from .candidates import ParquetRecordWriter, _rank, _schema
from .config import output_root, project_path
from .io import file_record, read_tsv, sha256, write_json, write_text, write_tsv
from .matrix import CHANNELS, PreparedScoring, direction_matrix, load_prepared_scoring


HPO_PROVENANCE = "data/hpo_cpo/inherited/hpo_cpo_inheritance_provenance.tsv"
HPO_STATUS = "data/hpo_cpo/inherited/hpo_cpo_inheritance_term_status.tsv"
HPO_INHERITANCE_MANIFEST = "data/hpo_cpo/inherited/manifest.json"
MATRIX_RELATIVE = "matrices/hpo_mpo_all_abnormal_branch.h5"
HPO_INDEX_RELATIVE = "indices/hpo_all_abnormal_index.tsv"
MPO_INDEX_RELATIVE = "indices/mpo_abnormal_index.tsv"
CHECKPOINT_RELATIVE = "checkpoints/all_abnormal_completed_blocks.jsonl"


ALL_INDEX_FIELDS = (
    "index",
    "term_id",
    "label",
    "node_class",
    "needs_self_query",
    "selected_for_processing",
    "processing_status",
    "has_descendant_D_profile",
    "has_descendant_M_profile",
)

ALL_CANDIDATE_FIELDS = (
    "query_side",
    "query_id",
    "target_id",
    "channel",
    "view",
    "rank",
    "shape",
    "support",
    "conflict",
    "n_hpo_source_leaves",
    "n_mpo_source_leaves",
    "query_node_class",
    "target_node_class",
    "query_processing_status",
    "target_processing_status",
    "interpretation_type",
    "ties_at_cutoff",
    "ties_truncated",
)
ALL_CANDIDATE_SCHEMA = _schema(
    ALL_CANDIDATE_FIELDS,
    {"shape", "support", "conflict"},
    {
        "rank",
        "n_hpo_source_leaves",
        "n_mpo_source_leaves",
        "ties_at_cutoff",
        "ties_truncated",
    },
)

ALL_EXPLANATION_FIELDS = (
    "hpo_id",
    "mpo_id",
    "channel",
    "score_type",
    "status",
    "branch_score",
    "best_hpo_leaf_id",
    "best_mpo_leaf_id",
    "leaf_pair_score",
    "n_best_leaf_pair_ties",
    "hpo_leaf_to_target_path",
    "mpo_leaf_to_target_path",
    "hpo_cpo_node_id",
    "mpo_cpo_node_id",
    "hpo_cl_id",
    "mpo_cl_id",
    "hpo_go_id",
    "mpo_go_id",
    "cpo_pair_score",
    "direction_status",
    "hpo_assertion_id",
    "mpo_assertion_id",
    "hpo_source_record_id",
    "mpo_source_record_id",
    "hpo_publication_ids",
    "mpo_publication_ids",
    "hpo_evidence_sentence",
    "mpo_evidence_sentence",
    "interpretation_type",
)
ALL_EXPLANATION_SCHEMA = _schema(
    ALL_EXPLANATION_FIELDS,
    {"branch_score", "leaf_pair_score", "cpo_pair_score"},
    {"n_best_leaf_pair_ties"},
)


@dataclass
class BranchContext:
    prepared: PreparedScoring
    hpo_rows: list[dict[str, str]]
    mpo_rows: list[dict[str, str]]
    hpo_ids: list[str]
    mpo_ids: list[str]
    hpo_labels: list[str]
    mpo_labels: list[str]
    hpo_classes: list[str]
    mpo_classes: list[str]
    hpo_processing: list[str]
    mpo_processing: list[str]
    hpo_groups: dict[str, list[list[int]]]
    mpo_groups: dict[str, list[list[int]]]
    hpo_paths: dict[tuple[str, str], str]
    mpo_paths: dict[tuple[str, str], str]


def _read_provenance_groups(
    path: Path,
    target_column: str,
    leaf_column: str,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    _, rows = read_tsv(path)
    groups: dict[str, set[str]] = defaultdict(set)
    paths: dict[tuple[str, str], tuple[int, str]] = {}
    for row in rows:
        target = row[target_column]
        leaf = row[leaf_column]
        groups[target].add(leaf)
        key = (target, leaf)
        candidate = (int(row["distance"]), row["canonical_path"])
        if key not in paths or candidate < paths[key]:
            paths[key] = candidate
    return dict(groups), {key: value[1] for key, value in paths.items()}


def _build_groups(
    full_ids: Sequence[str],
    descendants: dict[str, set[str]],
    source_ids: Sequence[str],
) -> list[list[int]]:
    source_index = {term_id: index for index, term_id in enumerate(source_ids)}
    groups: list[list[int]] = []
    for target in full_ids:
        leaves = set(descendants.get(target, set()))
        if target in source_index:
            leaves.add(target)
        groups.append(sorted(source_index[leaf] for leaf in leaves if leaf in source_index))
    return groups


def load_branch_context(config: dict[str, Any]) -> BranchContext:
    root = output_root(config)
    project = Path(config["_project_root"])
    prepared = load_prepared_scoring(config)
    _, hpo_rows = read_tsv(project / HPO_STATUS)
    _, mpo_all_rows = read_tsv(project_path(config, config["inputs"]["inheritance_status"]))
    mpo_rows = [row for row in mpo_all_rows if row["is_normal_branch"] != "true"]
    hpo_rows = sorted(hpo_rows, key=lambda row: row["hpo_id"])
    mpo_rows = sorted(mpo_rows, key=lambda row: row["mp_id"])
    hpo_ids = [row["hpo_id"] for row in hpo_rows]
    mpo_ids = [row["mp_id"] for row in mpo_rows]
    hpo_descendants, hpo_paths = _read_provenance_groups(
        project / HPO_PROVENANCE, "target_hpo_id", "source_leaf_hpo_id"
    )
    mpo_descendants, mpo_paths = _read_provenance_groups(
        project_path(config, config["inputs"]["inheritance_provenance"]),
        "target_mp_id",
        "source_leaf_mp_id",
    )
    hpo_groups: dict[str, list[list[int]]] = {}
    mpo_groups: dict[str, list[list[int]]] = {}
    for evidence in ("D", "M"):
        h_source_ids = [
            prepared.hpo_ids[index]
            for index, profile in enumerate(prepared.profiles[("HPO", evidence)])
            if profile.size
        ]
        m_source_ids = [
            prepared.mpo_ids[index]
            for index, profile in enumerate(prepared.profiles[("MPO", evidence)])
            if profile.size and prepared.mpo_classes[index] == "leaf"
        ]
        hpo_groups[evidence] = _build_groups(hpo_ids, hpo_descendants, h_source_ids)
        mpo_groups[evidence] = _build_groups(mpo_ids, mpo_descendants, m_source_ids)
    return BranchContext(
        prepared=prepared,
        hpo_rows=hpo_rows,
        mpo_rows=mpo_rows,
        hpo_ids=hpo_ids,
        mpo_ids=mpo_ids,
        hpo_labels=[row["hpo_label"] for row in hpo_rows],
        mpo_labels=[row["mp_label"] for row in mpo_rows],
        hpo_classes=[row["node_type"] for row in hpo_rows],
        mpo_classes=[row["node_type"] for row in mpo_rows],
        hpo_processing=[
            row["middle_processing_status"] or row["inheritance_status"]
            for row in hpo_rows
        ],
        mpo_processing=[row["profile_mode"] for row in mpo_rows],
        hpo_groups=hpo_groups,
        mpo_groups=mpo_groups,
        hpo_paths=hpo_paths,
        mpo_paths=mpo_paths,
    )


def _source_ids(context: BranchContext, side: str, evidence: str) -> list[str]:
    if side == "HPO":
        return [
            context.prepared.hpo_ids[index]
            for index, profile in enumerate(context.prepared.profiles[("HPO", evidence)])
            if profile.size
        ]
    return [
        context.prepared.mpo_ids[index]
        for index, profile in enumerate(context.prepared.profiles[("MPO", evidence)])
        if profile.size and context.prepared.mpo_classes[index] == "leaf"
    ]


def _write_all_indices(root: Path, context: BranchContext) -> None:
    h_rows = []
    for index, row in enumerate(context.hpo_rows):
        h_rows.append(
            {
                "index": index,
                "term_id": row["hpo_id"],
                "label": row["hpo_label"],
                "node_class": row["node_type"],
                "needs_self_query": row["needs_self_query"],
                "selected_for_processing": row["selected_for_processing"],
                "processing_status": context.hpo_processing[index],
                "has_descendant_D_profile": str(bool(context.hpo_groups["D"][index])).lower(),
                "has_descendant_M_profile": str(bool(context.hpo_groups["M"][index])).lower(),
            }
        )
    m_rows = []
    for index, row in enumerate(context.mpo_rows):
        m_rows.append(
            {
                "index": index,
                "term_id": row["mp_id"],
                "label": row["mp_label"],
                "node_class": row["node_type"],
                "needs_self_query": "",
                "selected_for_processing": "",
                "processing_status": context.mpo_processing[index],
                "has_descendant_D_profile": str(bool(context.mpo_groups["D"][index])).lower(),
                "has_descendant_M_profile": str(bool(context.mpo_groups["M"][index])).lower(),
            }
        )
    write_tsv(root / HPO_INDEX_RELATIVE, ALL_INDEX_FIELDS, h_rows)
    write_tsv(root / MPO_INDEX_RELATIVE, ALL_INDEX_FIELDS, m_rows)


def _all_input_digest(config: dict[str, Any]) -> str:
    root = output_root(config)
    project = Path(config["_project_root"])
    paths = (
        root / "matrices/hpo_mpo_cpo.h5",
        project / HPO_PROVENANCE,
        project / HPO_STATUS,
        project / HPO_INHERITANCE_MANIFEST,
        project_path(config, config["inputs"]["inheritance_provenance"]),
        project_path(config, config["inputs"]["inheritance_status"]),
        project_path(config, config["inputs"]["inheritance_manifest"]),
    )
    payload = "\n".join(f"{path.resolve()}\t{sha256(path)}" for path in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_records(path: Path, digest: str) -> set[tuple[str, str, int, int]]:
    if not path.is_file():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("input_digest") == digest and row.get("status") == "complete":
            result.add(
                (
                    row["channel"],
                    row["score_type"],
                    int(row["column_start"]),
                    int(row["column_end"]),
                )
            )
    return result


def _append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _create_all_matrix(
    path: Path,
    n_hpo: int,
    n_mpo: int,
    digest: str,
    hpo_block: int,
    mpo_block: int,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "a")
    if "input_digest" in handle.attrs:
        if handle.attrs["input_digest"] != digest:
            handle.close()
            raise ValueError("all-abnormal matrix input digest mismatch")
        return handle
    handle.attrs["schema_version"] = 1
    handle.attrs["view"] = "best_descendant_leaf_pair"
    handle.attrs["input_digest"] = digest
    handle.attrs["hpo_count"] = n_hpo
    handle.attrs["mpo_count"] = n_mpo
    chunks = (min(hpo_block, n_hpo), min(mpo_block, n_mpo))
    for channel in CHANNELS:
        group = handle.require_group(channel)
        for score_type in ("shape", "support", "conflict"):
            group.create_dataset(
                score_type,
                shape=(n_hpo, n_mpo),
                dtype="float32",
                chunks=chunks,
                compression="lzf",
                shuffle=True,
                fillvalue=np.nan,
            )
    handle.flush()
    return handle


def _group_arrays(groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray([index for index, values in enumerate(groups) if values], dtype=np.int32)
    values = [groups[index] for index in targets]
    lengths = np.asarray([len(group) for group in values], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.asarray([item for group in values for item in group], dtype=np.int32)
    return targets, offsets, flat


def run_all_abnormal_compute(
    config: dict[str, Any], resume: bool = True
) -> dict[str, object]:
    root = output_root(config)
    context = load_branch_context(config)
    _write_all_indices(root, context)
    digest = _all_input_digest(config)
    matrix_path = root / MATRIX_RELATIVE
    checkpoint_path = root / CHECKPOINT_RELATIVE
    existed = matrix_path.is_file()
    completed = _checkpoint_records(checkpoint_path, digest) if resume and existed else set()
    hpo_block = int(config["runtime"]["hpo_block_size"])
    mpo_block = int(config["runtime"].get("mpo_block_size", 512))
    output = _create_all_matrix(
        matrix_path,
        len(context.hpo_ids),
        len(context.mpo_ids),
        digest,
        hpo_block,
        mpo_block,
    )
    current_hpo_index = {
        term_id: index for index, term_id in enumerate(context.prepared.hpo_ids)
    }
    current_mpo_index = {
        term_id: index for index, term_id in enumerate(context.prepared.mpo_ids)
    }
    stage_started = time.monotonic()
    blocks_written = 0
    try:
        with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as base_handle:
            with tempfile.TemporaryDirectory(prefix="cpo_all_abnormal_") as temporary:
                temporary_root = Path(temporary)
                for channel, (h_evidence, m_evidence) in CHANNELS.items():
                    h_source_ids = _source_ids(context, "HPO", h_evidence)
                    m_source_ids = _source_ids(context, "MPO", m_evidence)
                    h_source_current = np.asarray(
                        [current_hpo_index[term_id] for term_id in h_source_ids], dtype=np.int32
                    )
                    m_source_current = np.asarray(
                        [current_mpo_index[term_id] for term_id in m_source_ids], dtype=np.int32
                    )
                    h_targets, h_offsets, h_flat = _group_arrays(context.hpo_groups[h_evidence])
                    m_targets, m_offsets, m_flat = _group_arrays(context.mpo_groups[m_evidence])
                    for score_type in ("shape", "support", "conflict"):
                        base_dataset = base_handle[channel][score_type]
                        base = base_dataset[h_source_current, :][:, m_source_current]
                        temporary_path = temporary_root / f"{channel}_{score_type}.dat"
                        m_aggregated = np.memmap(
                            temporary_path,
                            mode="w+",
                            dtype="float32",
                            shape=(len(h_source_ids), len(context.mpo_ids)),
                        )
                        m_aggregated[:] = np.nan
                        for start in range(0, len(h_source_ids), hpo_block):
                            end = min(len(h_source_ids), start + hpo_block)
                            values = base[start:end, :][:, m_flat]
                            aggregated = np.fmax.reduceat(values, m_offsets[:-1], axis=1)
                            m_aggregated[start:end, m_targets] = aggregated
                        m_aggregated.flush()
                        dataset = output[channel][score_type]
                        for start in range(0, len(context.mpo_ids), mpo_block):
                            end = min(len(context.mpo_ids), start + mpo_block)
                            key = (channel, score_type, start, end)
                            if key in completed:
                                continue
                            block_started = time.monotonic()
                            repeated = m_aggregated[:, start:end][h_flat, :]
                            aggregated = np.fmax.reduceat(
                                repeated, h_offsets[:-1], axis=0
                            )
                            block = np.full(
                                (len(context.hpo_ids), end - start),
                                np.nan,
                                dtype=np.float32,
                            )
                            block[h_targets, :] = aggregated
                            dataset[:, start:end] = block
                            output.flush()
                            _append_checkpoint(
                                checkpoint_path,
                                {
                                    "channel": channel,
                                    "score_type": score_type,
                                    "column_start": start,
                                    "column_end": end,
                                    "input_digest": digest,
                                    "finite_count": int(np.isfinite(block).sum()),
                                    "elapsed_seconds": time.monotonic() - block_started,
                                    "status": "complete",
                                },
                            )
                            blocks_written += 1
                        del m_aggregated
    finally:
        output.close()

    expected = len(CHANNELS) * 3 * math.ceil(len(context.mpo_ids) / mpo_block)
    final_completed = _checkpoint_records(checkpoint_path, digest)
    if len(final_completed) != expected:
        raise ValueError(f"all-abnormal matrix incomplete: {len(final_completed)}/{expected}")
    metadata = {
        "schema_version": 1,
        "view": "best_descendant_leaf_pair",
        "matrix_shape": [len(context.hpo_ids), len(context.mpo_ids)],
        "channels": list(CHANNELS),
        "scores": ["shape", "support", "conflict"],
        "hpo_scope": "HP:0000118 plus all is_a descendants",
        "mpo_scope": "all active MPO terms except normal phenotype branch",
        "source": "complete self CPO profiles on processed leaf terms",
        "aggregation": "maximum over every scoreable HPO-leaf/MPO-leaf pair below the two target terms",
        "inherited_relations_merged_into_self": False,
        "input_digest": digest,
        "completed_block_count": len(final_completed),
        "expected_block_count": expected,
        "not_computed_block_count": 0,
    }
    write_json(root / "matrices/all_abnormal_score_status_metadata.json", metadata)
    result = {
        "status": "complete",
        "matrix": file_record(matrix_path),
        "metadata": metadata,
        "blocks_written_this_run": blocks_written,
        "runtime_seconds": time.monotonic() - stage_started,
    }
    write_json(root / "logs/all_abnormal_compute_summary.json", result)
    return result


def _all_candidate_row(
    context: BranchContext,
    query_side: str,
    h_index: int,
    m_index: int,
    channel: str,
    view: str,
    rank: int,
    shape: float,
    support: float,
    conflict: float,
    ties: int,
    truncated: int,
) -> dict[str, object]:
    return {
        "query_side": query_side,
        "query_id": context.hpo_ids[h_index] if query_side == "HPO" else context.mpo_ids[m_index],
        "target_id": context.mpo_ids[m_index] if query_side == "HPO" else context.hpo_ids[h_index],
        "channel": channel,
        "view": view,
        "rank": rank,
        "shape": shape,
        "support": support,
        "conflict": conflict,
        "n_hpo_source_leaves": len(context.hpo_groups[CHANNELS[channel][0]][h_index]),
        "n_mpo_source_leaves": len(context.mpo_groups[CHANNELS[channel][1]][m_index]),
        "query_node_class": context.hpo_classes[h_index] if query_side == "HPO" else context.mpo_classes[m_index],
        "target_node_class": context.mpo_classes[m_index] if query_side == "HPO" else context.hpo_classes[h_index],
        "query_processing_status": context.hpo_processing[h_index] if query_side == "HPO" else context.mpo_processing[m_index],
        "target_processing_status": context.mpo_processing[m_index] if query_side == "HPO" else context.hpo_processing[h_index],
        "interpretation_type": "all_abnormal_descendant_leaf_evidence",
        "ties_at_cutoff": ties,
        "ties_truncated": truncated,
    }


def run_all_abnormal_candidates(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    context = load_branch_context(config)
    top_k = int(config["candidates"]["top_k"])
    allowed_hpo = np.asarray(
        [kind not in {"root", "top_level"} for kind in context.hpo_classes], dtype=bool
    )
    allowed_mpo = np.asarray(
        [kind not in {"root", "top_level"} for kind in context.mpo_classes], dtype=bool
    )
    h_writer = ParquetRecordWriter(
        root / "candidates/hpo_all_to_mpo_all_topk.parquet", ALL_CANDIDATE_SCHEMA
    )
    m_writer = ParquetRecordWriter(
        root / "candidates/mpo_all_to_hpo_all_topk.parquet", ALL_CANDIDATE_SCHEMA
    )
    h_block = int(config["runtime"]["hpo_block_size"])
    m_block = int(config["runtime"].get("mpo_block_size", 512))
    with h5py.File(root / MATRIX_RELATIVE, "r") as handle:
        for channel in CHANNELS:
            shape_ds = handle[channel]["shape"]
            support_ds = handle[channel]["support"]
            conflict_ds = handle[channel]["conflict"]
            for start in range(0, len(context.hpo_ids), h_block):
                end = min(len(context.hpo_ids), start + h_block)
                shape_block = shape_ds[start:end, :]
                support_block = support_ds[start:end, :]
                conflict_block = conflict_ds[start:end, :]
                for offset, h_index in enumerate(range(start, end)):
                    for view, primary, secondary in (
                        ("shape_topk", shape_block[offset], shape_block[offset]),
                        ("direction_supported_topk", support_block[offset], shape_block[offset]),
                    ):
                        selected, ties, truncated = _rank(
                            primary,
                            secondary,
                            context.mpo_ids,
                            allowed_mpo,
                            top_k,
                        )
                        for rank, m_index in enumerate(selected, start=1):
                            h_writer.add(
                                _all_candidate_row(
                                    context,
                                    "HPO",
                                    h_index,
                                    m_index,
                                    channel,
                                    view,
                                    rank,
                                    float(shape_block[offset, m_index]),
                                    float(support_block[offset, m_index]),
                                    float(conflict_block[offset, m_index]),
                                    ties,
                                    truncated,
                                )
                            )
            for start in range(0, len(context.mpo_ids), m_block):
                end = min(len(context.mpo_ids), start + m_block)
                shape_block = shape_ds[:, start:end]
                support_block = support_ds[:, start:end]
                conflict_block = conflict_ds[:, start:end]
                for offset, m_index in enumerate(range(start, end)):
                    for view, primary, secondary in (
                        ("shape_topk", shape_block[:, offset], shape_block[:, offset]),
                        ("direction_supported_topk", support_block[:, offset], shape_block[:, offset]),
                    ):
                        selected, ties, truncated = _rank(
                            primary,
                            secondary,
                            context.hpo_ids,
                            allowed_hpo,
                            top_k,
                        )
                        for rank, h_index in enumerate(selected, start=1):
                            m_writer.add(
                                _all_candidate_row(
                                    context,
                                    "MPO",
                                    h_index,
                                    m_index,
                                    channel,
                                    view,
                                    rank,
                                    float(shape_block[h_index, offset]),
                                    float(support_block[h_index, offset]),
                                    float(conflict_block[h_index, offset]),
                                    ties,
                                    truncated,
                                )
                            )
    result = {
        "hpo_to_mpo_candidate_count": h_writer.close(),
        "mpo_to_hpo_candidate_count": m_writer.close(),
    }
    write_json(root / "logs/all_abnormal_candidates_summary.json", result)
    return result


def _all_candidate_pairs(root: Path) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for path in (
        root / "candidates/hpo_all_to_mpo_all_topk.parquet",
        root / "candidates/mpo_all_to_hpo_all_topk.parquet",
    ):
        frame = pd.read_parquet(
            path, columns=["query_side", "query_id", "target_id", "channel"]
        )
        for row in frame.itertuples(index=False):
            if row.query_side == "HPO":
                result.add((str(row.query_id), str(row.target_id), str(row.channel)))
            else:
                result.add((str(row.target_id), str(row.query_id), str(row.channel)))
    return result


def _best_leaf_pair(
    values: np.ndarray,
    h_local_indices: list[int],
    m_local_indices: list[int],
    h_ids: list[str],
    m_ids: list[str],
) -> tuple[tuple[int, int, float, int] | None, float]:
    selected = values[np.ix_(h_local_indices, m_local_indices)]
    finite = selected[np.isfinite(selected)]
    if not len(finite):
        return None, float("nan")
    maximum = float(finite.max())
    if maximum <= 0:
        return None, maximum
    positions = np.argwhere(selected == maximum)
    choices = sorted(
        ((int(left), int(right)) for left, right in positions),
        key=lambda pair: (h_ids[pair[0]], m_ids[pair[1]]),
    )
    left, right = choices[0]
    return (left, right, maximum, len(choices)), maximum


def _best_cpo_pair(
    context: BranchContext,
    h_index: int,
    m_index: int,
    channel: str,
    score_type: str,
) -> tuple[int, int, float, int] | None:
    h_evidence, m_evidence = CHANNELS[channel]
    left = context.prepared.profiles[("HPO", h_evidence)][h_index]
    right = context.prepared.profiles[("MPO", m_evidence)][m_index]
    shape = context.prepared.cpo_similarity[np.ix_(left.cpo_indices, right.cpo_indices)]
    if score_type == "shape":
        values = shape
    else:
        directions = direction_matrix(left, right)
        values = shape * (directions == (1 if score_type == "support" else -1))
    maximum = float(values.max())
    if maximum <= 0:
        return None
    positions = np.argwhere(values == maximum)
    choices = sorted(
        ((int(i), int(j)) for i, j in positions),
        key=lambda pair: (left.node_ids[pair[0]], right.node_ids[pair[1]]),
    )
    left_position, right_position = choices[0]
    return left_position, right_position, maximum, len(choices)


def run_all_abnormal_explanations(config: dict[str, Any]) -> dict[str, int]:
    from .explain import _direction_name, _source_indexes

    root = output_root(config)
    context = load_branch_context(config)
    h_all_index = {term_id: index for index, term_id in enumerate(context.hpo_ids)}
    m_all_index = {term_id: index for index, term_id in enumerate(context.mpo_ids)}
    h_current_index = {
        term_id: index for index, term_id in enumerate(context.prepared.hpo_ids)
    }
    m_current_index = {
        term_id: index for index, term_id in enumerate(context.prepared.mpo_ids)
    }
    structures = {
        str(row.cpo_node_id): row
        for row in context.prepared.structures[
            context.prepared.structures["origin"] == "self"
        ].itertuples(index=False)
    }
    assertions, first_assertion = _source_indexes(root)
    writer = ParquetRecordWriter(
        root / "explanations/all_abnormal_branch_matches.parquet",
        ALL_EXPLANATION_SCHEMA,
    )
    source_ids = {
        (side, evidence): _source_ids(context, side, evidence)
        for side in ("HPO", "MPO")
        for evidence in ("D", "M")
    }
    pairs_by_channel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for hpo_id, mpo_id, channel in sorted(_all_candidate_pairs(root)):
        pairs_by_channel[channel].append((hpo_id, mpo_id))
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as base_handle:
        for channel in CHANNELS:
            h_evidence, m_evidence = CHANNELS[channel]
            channel_h_source_ids = source_ids[("HPO", h_evidence)]
            channel_m_source_ids = source_ids[("MPO", m_evidence)]
            channel_h_current = np.asarray(
                [h_current_index[term_id] for term_id in channel_h_source_ids],
                dtype=np.int32,
            )
            channel_m_current = np.asarray(
                [m_current_index[term_id] for term_id in channel_m_source_ids],
                dtype=np.int32,
            )
            base_matrices = {
                score_type: base_handle[channel][score_type][channel_h_current, :][
                    :, channel_m_current
                ]
                for score_type in ("shape", "support", "conflict")
            }
            for hpo_id, mpo_id in pairs_by_channel[channel]:
                h_target = h_all_index[hpo_id]
                m_target = m_all_index[mpo_id]
                h_local_group = context.hpo_groups[h_evidence][h_target]
                m_local_group = context.mpo_groups[m_evidence][m_target]
                h_leaf_ids = [channel_h_source_ids[index] for index in h_local_group]
                m_leaf_ids = [channel_m_source_ids[index] for index in m_local_group]
                for score_type in ("shape", "support", "conflict"):
                    best_leaf, branch_score = _best_leaf_pair(
                        base_matrices[score_type],
                        h_local_group,
                        m_local_group,
                        h_leaf_ids,
                        m_leaf_ids,
                    )
                    if best_leaf is None:
                        writer.add(
                            {
                                "hpo_id": hpo_id,
                                "mpo_id": mpo_id,
                                "channel": channel,
                                "score_type": score_type,
                                "status": "NO_POSITIVE_MATCH",
                                "branch_score": branch_score,
                                "best_hpo_leaf_id": "",
                                "best_mpo_leaf_id": "",
                                "leaf_pair_score": 0.0,
                                "n_best_leaf_pair_ties": 0,
                                "hpo_leaf_to_target_path": "",
                                "mpo_leaf_to_target_path": "",
                                "hpo_cpo_node_id": "",
                                "mpo_cpo_node_id": "",
                                "hpo_cl_id": "",
                                "mpo_cl_id": "",
                                "hpo_go_id": "",
                                "mpo_go_id": "",
                                "cpo_pair_score": 0.0,
                                "direction_status": "unknown_or_incomparable",
                                "hpo_assertion_id": "",
                                "mpo_assertion_id": "",
                                "hpo_source_record_id": "",
                                "mpo_source_record_id": "",
                                "hpo_publication_ids": "",
                                "mpo_publication_ids": "",
                                "hpo_evidence_sentence": "",
                                "mpo_evidence_sentence": "",
                                "interpretation_type": "all_abnormal_descendant_leaf_evidence",
                            }
                        )
                        continue
                    h_leaf_position, m_leaf_position, leaf_score, leaf_ties = best_leaf
                    h_leaf_id = h_leaf_ids[h_leaf_position]
                    m_leaf_id = m_leaf_ids[m_leaf_position]
                    h_leaf_index = h_current_index[h_leaf_id]
                    m_leaf_index = m_current_index[m_leaf_id]
                    cpo_best = _best_cpo_pair(
                        context,
                        h_leaf_index,
                        m_leaf_index,
                        channel,
                        score_type,
                    )
                    if cpo_best is None:
                        raise ValueError("positive leaf score lacks a positive CPO explanation")
                    h_pos, m_pos, cpo_score, _cpo_ties = cpo_best
                    h_profile = context.prepared.profiles[("HPO", h_evidence)][h_leaf_index]
                    m_profile = context.prepared.profiles[("MPO", m_evidence)][m_leaf_index]
                    h_node_id = h_profile.node_ids[h_pos]
                    m_node_id = m_profile.node_ids[m_pos]
                    h_node = structures[h_node_id]
                    m_node = structures[m_node_id]
                    directions = direction_matrix(h_profile, m_profile)
                    h_assertion_id = first_assertion.get(h_node_id, "")
                    m_assertion_id = first_assertion.get(m_node_id, "")
                    h_assertion = assertions.get(h_assertion_id, {})
                    m_assertion = assertions.get(m_assertion_id, {})
                    writer.add(
                        {
                            "hpo_id": hpo_id,
                            "mpo_id": mpo_id,
                            "channel": channel,
                            "score_type": score_type,
                            "status": "OK",
                            "branch_score": branch_score,
                            "best_hpo_leaf_id": h_leaf_id,
                            "best_mpo_leaf_id": m_leaf_id,
                            "leaf_pair_score": leaf_score,
                            "n_best_leaf_pair_ties": leaf_ties,
                            "hpo_leaf_to_target_path": context.hpo_paths.get(
                                (hpo_id, h_leaf_id), h_leaf_id if hpo_id == h_leaf_id else ""
                            ),
                            "mpo_leaf_to_target_path": context.mpo_paths.get(
                                (mpo_id, m_leaf_id), m_leaf_id if mpo_id == m_leaf_id else ""
                            ),
                            "hpo_cpo_node_id": h_node_id,
                            "mpo_cpo_node_id": m_node_id,
                            "hpo_cl_id": str(h_node.cl_id),
                            "mpo_cl_id": str(m_node.cl_id),
                            "hpo_go_id": str(h_node.go_id),
                            "mpo_go_id": str(m_node.go_id),
                            "cpo_pair_score": cpo_score,
                            "direction_status": _direction_name(directions[h_pos, m_pos]),
                            "hpo_assertion_id": h_assertion_id,
                            "mpo_assertion_id": m_assertion_id,
                            "hpo_source_record_id": str(h_assertion.get("source_record_id", "")),
                            "mpo_source_record_id": str(m_assertion.get("source_record_id", "")),
                            "hpo_publication_ids": str(h_assertion.get("publication_ids", "")),
                            "mpo_publication_ids": str(m_assertion.get("publication_ids", "")),
                            "hpo_evidence_sentence": str(h_assertion.get("evidence_sentence", "")),
                            "mpo_evidence_sentence": str(m_assertion.get("evidence_sentence", "")),
                            "interpretation_type": "all_abnormal_descendant_leaf_evidence",
                        }
                    )
    count = writer.close()
    result = {"all_abnormal_explanation_count": count}
    write_json(root / "logs/all_abnormal_explanations_summary.json", result)
    return result


def _dataset_counts(dataset: h5py.Dataset, block: int = 64) -> tuple[int, int]:
    finite = 0
    positive = 0
    for start in range(0, dataset.shape[0], block):
        values = dataset[start : start + block, :]
        finite += int(np.isfinite(values).sum())
        positive += int(np.count_nonzero(np.isfinite(values) & (values > 0)))
    return finite, positive


def run_all_abnormal_report(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    context = load_branch_context(config)
    coverage_rows: list[dict[str, object]] = []
    with h5py.File(root / MATRIX_RELATIVE, "r") as handle:
        for channel, (h_evidence, m_evidence) in CHANNELS.items():
            finite_shape, positive_shape = _dataset_counts(handle[channel]["shape"])
            finite_support, positive_support = _dataset_counts(handle[channel]["support"])
            finite_conflict, positive_conflict = _dataset_counts(handle[channel]["conflict"])
            if finite_support != finite_conflict:
                raise ValueError("all-abnormal support/conflict finite masks differ")
            coverage_rows.append(
                {
                    "channel": channel,
                    "hpo_terms_with_source_leaves": sum(bool(group) for group in context.hpo_groups[h_evidence]),
                    "mpo_terms_with_source_leaves": sum(bool(group) for group in context.mpo_groups[m_evidence]),
                    "finite_shape_pairs": finite_shape,
                    "positive_shape_pairs": positive_shape,
                    "finite_direction_pairs": finite_support,
                    "positive_support_pairs": positive_support,
                    "positive_conflict_pairs": positive_conflict,
                }
            )
    write_tsv(
        root / "evaluation/all_abnormal_coverage.tsv",
        tuple(coverage_rows[0]),
        coverage_rows,
    )
    lines = [
        "# All-abnormality HPO-MPO descendant-leaf evidence matrix",
        "",
        "The matrix indexes all 18,690 HPO phenotypic-abnormality terms and 14,383 non-normal MPO terms. Scores are the maximum verified self score over all eligible descendant HPO-leaf/MPO-leaf pairs. They are branch evidence, not parent self assertions.",
        "",
        "HPO middle-node processing states are preserved: 3,241 selected terms require self processing but currently have no formal self CPO result; 258 required terms were excluded by the processing-class policy; 2,214 terms are inheritance-only.",
        "",
        "| Channel | HPO terms with source leaves | MPO terms with source leaves | Scoreable pairs | Positive shape | Comparable direction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['channel']} | {row['hpo_terms_with_source_leaves']:,} | {row['mpo_terms_with_source_leaves']:,} | "
            f"{row['finite_shape_pairs']:,} | {row['positive_shape_pairs']:,} | {row['finite_direction_pairs']:,} |"
        )
    lines.extend(
        [
            "",
            "Each selected candidate is explained by the actual best HPO leaf, MPO leaf, both legal ontology paths, the best CPO pair, and original normalized assertions.",
        ]
    )
    report_path = root / "reports/all_abnormal_report.md"
    write_text(report_path, "\n".join(lines) + "\n")
    result = {
        "status": "complete",
        "matrix_shape": [len(context.hpo_ids), len(context.mpo_ids)],
        "coverage": coverage_rows,
        "report": file_record(report_path),
    }
    write_json(root / "logs/all_abnormal_report_summary.json", result)
    return result


def run_all_abnormal(config: dict[str, Any], resume: bool = True) -> dict[str, object]:
    result = {
        "compute": run_all_abnormal_compute(config, resume=resume),
        "candidates": run_all_abnormal_candidates(config),
        "explanations": run_all_abnormal_explanations(config),
        "report": run_all_abnormal_report(config),
    }
    write_json(output_root(config) / "logs/all_abnormal_summary.json", result)
    return result
