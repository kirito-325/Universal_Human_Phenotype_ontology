from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from .config import config_digest, output_root, project_path
from .io import file_record, read_tsv, sha256, write_json
from .ontology import Ontology
from .scoring import ScoreResult, score_pair


CHANNELS = {"DD": ("D", "D"), "DM": ("D", "M"), "MD": ("M", "D"), "MM": ("M", "M")}
STATE_CODE = {"increased": 1, "decreased": 2, "absent": 3, "present": 4}
SIGNATURE_CODE = {"unknown": 0, "unique": 1, "mixed_direction": 2, "multiple_axes": 3}


@dataclass(frozen=True)
class TermProfile:
    cpo_indices: np.ndarray
    exact_codes: np.ndarray
    axis_codes: np.ndarray
    state_codes: np.ndarray
    signature_codes: np.ndarray
    node_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.cpo_indices)


@dataclass
class PreparedScoring:
    hpo_ids: list[str]
    mpo_ids: list[str]
    hpo_labels: list[str]
    mpo_labels: list[str]
    mpo_classes: list[str]
    hpo_cpo_keys: list[tuple[str, str]]
    mpo_cpo_keys: list[tuple[str, str]]
    cpo_similarity: np.ndarray
    profiles: dict[tuple[str, str], list[TermProfile]]
    structures: pd.DataFrame
    cl: Ontology
    go: Ontology


def _empty_profile() -> TermProfile:
    return TermProfile(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.int8),
        (),
    )


def _pairwise_lin(
    ontology: Ontology, left_ids: list[str], right_ids: list[str]
) -> np.ndarray:
    result = np.empty((len(left_ids), len(right_ids)), dtype=np.float64)
    for left_index, left_id in enumerate(left_ids):
        for right_index, right_id in enumerate(right_ids):
            result[left_index, right_index] = ontology.lin(left_id, right_id)
    return result


def load_prepared_scoring(config: dict[str, Any]) -> PreparedScoring:
    root = output_root(config)
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    hpo_ids = [row["term_id"] for row in hpo_rows]
    mpo_ids = [row["term_id"] for row in mpo_rows]
    hpo_labels = [row["label"] for row in hpo_rows]
    mpo_labels = [row["label"] for row in mpo_rows]
    mpo_classes = [row["node_class"] for row in mpo_rows]
    hpo_index = {term_id: index for index, term_id in enumerate(hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(mpo_ids)}
    structures = pd.read_parquet(root / "normalized/cpo_structure_nodes.parquet")
    structures = structures[structures["origin"] == "self"].copy()

    cl = Ontology.from_obo(
        project_path(config, config["ontologies"]["cl"]), "CL:0000000", "CL:"
    )
    go = Ontology.from_obo(
        project_path(config, config["ontologies"]["go"]),
        "GO:0008150",
        "GO:",
        "biological_process",
    )
    hpo_cpo_keys = sorted(
        {
            (str(row.cl_id), str(row.go_id))
            for row in structures[structures["ontology_side"] == "HPO"].itertuples(index=False)
        }
    )
    mpo_cpo_keys = sorted(
        {
            (str(row.cl_id), str(row.go_id))
            for row in structures[structures["ontology_side"] == "MPO"].itertuples(index=False)
        }
    )
    hpo_cpo_index = {key: index for index, key in enumerate(hpo_cpo_keys)}
    mpo_cpo_index = {key: index for index, key in enumerate(mpo_cpo_keys)}
    exact_code_index = {
        key: index for index, key in enumerate(sorted(set(hpo_cpo_keys) | set(mpo_cpo_keys)))
    }
    hpo_cl_ids = sorted({key[0] for key in hpo_cpo_keys})
    mpo_cl_ids = sorted({key[0] for key in mpo_cpo_keys})
    hpo_go_ids = sorted({key[1] for key in hpo_cpo_keys})
    mpo_go_ids = sorted({key[1] for key in mpo_cpo_keys})
    cl_similarity = _pairwise_lin(cl, hpo_cl_ids, mpo_cl_ids)
    go_similarity = _pairwise_lin(go, hpo_go_ids, mpo_go_ids)
    hpo_cl_index = {term_id: index for index, term_id in enumerate(hpo_cl_ids)}
    mpo_cl_index = {term_id: index for index, term_id in enumerate(mpo_cl_ids)}
    hpo_go_index = {term_id: index for index, term_id in enumerate(hpo_go_ids)}
    mpo_go_index = {term_id: index for index, term_id in enumerate(mpo_go_ids)}
    h_cl = np.asarray([hpo_cl_index[key[0]] for key in hpo_cpo_keys], dtype=np.int32)
    m_cl = np.asarray([mpo_cl_index[key[0]] for key in mpo_cpo_keys], dtype=np.int32)
    h_go = np.asarray([hpo_go_index[key[1]] for key in hpo_cpo_keys], dtype=np.int32)
    m_go = np.asarray([mpo_go_index[key[1]] for key in mpo_cpo_keys], dtype=np.int32)
    cpo_similarity = (
        cl_similarity[h_cl[:, None], m_cl[None, :]]
        * go_similarity[h_go[:, None], m_go[None, :]]
    )

    axes = sorted(
        {
            (str(row.direction_target), str(row.direction_property))
            for row in structures.itertuples(index=False)
            if row.direction_signature_status == "unique"
        }
    )
    axis_index = {axis: index for index, axis in enumerate(axes)}
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in structures.itertuples(index=False):
        grouped[(str(row.ontology_side), str(row.evidence_class), str(row.term_id))].append(row)

    profiles: dict[tuple[str, str], list[TermProfile]] = {}
    for side, evidence in (("HPO", "D"), ("HPO", "M"), ("MPO", "D"), ("MPO", "M")):
        ids = hpo_ids if side == "HPO" else mpo_ids
        cpo_index = hpo_cpo_index if side == "HPO" else mpo_cpo_index
        values: list[TermProfile] = []
        for term_id in ids:
            nodes = sorted(grouped.get((side, evidence, term_id), ()), key=lambda row: row.cpo_node_id)
            if not nodes:
                values.append(_empty_profile())
                continue
            cpo_indices = np.asarray(
                [cpo_index[(str(row.cl_id), str(row.go_id))] for row in nodes],
                dtype=np.int32,
            )
            exact_codes = np.asarray(
                [exact_code_index[(str(row.cl_id), str(row.go_id))] for row in nodes],
                dtype=np.int32,
            )
            axis_codes = np.asarray(
                [
                    axis_index[(str(row.direction_target), str(row.direction_property))]
                    if row.direction_signature_status == "unique"
                    else -1
                    for row in nodes
                ],
                dtype=np.int32,
            )
            state_codes = np.asarray(
                [
                    STATE_CODE.get(str(row.direction_state), 0)
                    if row.direction_signature_status == "unique"
                    else 0
                    for row in nodes
                ],
                dtype=np.int8,
            )
            signature_codes = np.asarray(
                [SIGNATURE_CODE.get(str(row.direction_signature_status), 0) for row in nodes],
                dtype=np.int8,
            )
            values.append(
                TermProfile(
                    cpo_indices,
                    exact_codes,
                    axis_codes,
                    state_codes,
                    signature_codes,
                    tuple(str(row.cpo_node_id) for row in nodes),
                )
            )
        profiles[(side, evidence)] = values
    return PreparedScoring(
        hpo_ids,
        mpo_ids,
        hpo_labels,
        mpo_labels,
        mpo_classes,
        hpo_cpo_keys,
        mpo_cpo_keys,
        cpo_similarity,
        profiles,
        structures,
        cl,
        go,
    )


def direction_matrix(left: TermProfile, right: TermProfile) -> np.ndarray:
    same_cpo = left.exact_codes[:, None] == right.exact_codes[None, :]
    same_axis = (
        (left.axis_codes[:, None] >= 0)
        & (right.axis_codes[None, :] >= 0)
        & (left.axis_codes[:, None] == right.axis_codes[None, :])
    )
    left_state = left.state_codes[:, None]
    right_state = right.state_codes[None, :]
    agree = same_cpo & same_axis & (left_state == right_state) & (left_state > 0)
    conflict = same_cpo & same_axis & (
        ((left_state == 1) & (right_state == 2))
        | ((left_state == 2) & (right_state == 1))
        | ((left_state == 3) & (right_state == 4))
        | ((left_state == 4) & (right_state == 3))
    )
    return agree.astype(np.int8) - conflict.astype(np.int8)


def score_term_pair(
    prepared: PreparedScoring,
    hpo_index: int,
    mpo_index: int,
    channel: str,
) -> ScoreResult | None:
    h_class, m_class = CHANNELS[channel]
    left = prepared.profiles[("HPO", h_class)][hpo_index]
    right = prepared.profiles[("MPO", m_class)][mpo_index]
    if not left.size or not right.size:
        return None
    shape = prepared.cpo_similarity[np.ix_(left.cpo_indices, right.cpo_indices)]
    return score_pair(shape, direction_matrix(left, right))


def _flatten_profiles(
    profiles: list[TermProfile],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    target_indices = np.asarray(
        [index for index, profile in enumerate(profiles) if profile.size], dtype=np.int32
    )
    lengths = np.asarray([profiles[index].size for index in target_indices], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat_cpo = np.concatenate([profiles[index].cpo_indices for index in target_indices])
    flat_exact = np.concatenate([profiles[index].exact_codes for index in target_indices])
    flat_axis = np.concatenate([profiles[index].axis_codes for index in target_indices])
    flat_state = np.concatenate([profiles[index].state_codes for index in target_indices])
    return target_indices, lengths, offsets, flat_cpo, flat_exact, flat_axis, flat_state


def _optimized_row_scores(
    prepared: PreparedScoring,
    left: TermProfile,
    right_profiles: list[TermProfile],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_targets = len(right_profiles)
    shape_output = np.full(n_targets, np.nan, dtype=np.float64)
    support_output = np.full(n_targets, np.nan, dtype=np.float64)
    conflict_output = np.full(n_targets, np.nan, dtype=np.float64)
    if not left.size:
        return shape_output, support_output, conflict_output
    (
        target_indices,
        lengths,
        offsets,
        flat_cpo,
        flat_exact,
        flat_axis,
        flat_state,
    ) = _flatten_profiles(right_profiles)
    if not len(target_indices):
        return shape_output, support_output, conflict_output
    starts = offsets[:-1]
    pair = prepared.cpo_similarity[np.ix_(left.cpo_indices, flat_cpo)]
    forward = np.maximum.reduceat(pair, starts, axis=1).mean(axis=0)
    reverse = np.add.reduceat(pair.max(axis=0), starts) / lengths
    shape_output[target_indices] = (forward + reverse) / 2

    same_cpo = left.exact_codes[:, None] == flat_exact[None, :]
    same_axis = (
        (left.axis_codes[:, None] >= 0)
        & (flat_axis[None, :] >= 0)
        & (left.axis_codes[:, None] == flat_axis[None, :])
    )
    left_state = left.state_codes[:, None]
    right_state = flat_state[None, :]
    agree = same_cpo & same_axis & (left_state == right_state) & (left_state > 0)
    conflict = same_cpo & same_axis & (
        ((left_state == 1) & (right_state == 2))
        | ((left_state == 2) & (right_state == 1))
        | ((left_state == 3) & (right_state == 4))
        | ((left_state == 4) & (right_state == 3))
    )
    comparable_by_group = (
        np.maximum.reduceat(agree | conflict, starts, axis=1).any(axis=0)
    )
    agree_forward = np.maximum.reduceat(agree, starts, axis=1).mean(axis=0)
    agree_reverse = np.add.reduceat(agree.max(axis=0), starts) / lengths
    conflict_forward = np.maximum.reduceat(conflict, starts, axis=1).mean(axis=0)
    conflict_reverse = np.add.reduceat(conflict.max(axis=0), starts) / lengths
    comparable_targets = target_indices[comparable_by_group]
    support_output[comparable_targets] = (
        (agree_forward + agree_reverse) / 2
    )[comparable_by_group]
    conflict_output[comparable_targets] = (
        (conflict_forward + conflict_reverse) / 2
    )[comparable_by_group]
    return shape_output, support_output, conflict_output


def matrix_input_digest(config: dict[str, Any]) -> str:
    root = output_root(config)
    paths = [
        root / "normalized/cpo_structure_nodes.parquet",
        root / "indices/hpo_index.tsv",
        root / "indices/mpo_index.tsv",
        project_path(config, config["ontologies"]["cl"]),
        project_path(config, config["ontologies"]["go"]),
    ]
    payload = "\n".join(f"{path.resolve()}\t{sha256(path)}" for path in paths)
    return sha256_bytes(payload.encode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def run_smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    prepared = load_prepared_scoring(config)
    rng = np.random.default_rng(int(config["runtime"]["seed"]))
    comparisons = 0
    max_error = 0.0
    status_mismatches = 0
    examples: list[dict[str, object]] = []
    for channel, (h_class, m_class) in CHANNELS.items():
        h_nonempty = [
            index for index, value in enumerate(prepared.profiles[("HPO", h_class)]) if value.size
        ]
        m_nonempty = [
            index for index, value in enumerate(prepared.profiles[("MPO", m_class)]) if value.size
        ]
        h_sample = sorted(rng.choice(h_nonempty, min(25, len(h_nonempty)), replace=False))
        m_sample = sorted(rng.choice(m_nonempty, min(50, len(m_nonempty)), replace=False))
        right_subset = [prepared.profiles[("MPO", m_class)][index] for index in m_sample]
        for h_index in h_sample:
            left = prepared.profiles[("HPO", h_class)][h_index]
            shape, support, conflict = _optimized_row_scores(prepared, left, right_subset)
            for local_index, m_index in enumerate(m_sample):
                reference = score_term_pair(prepared, h_index, m_index, channel)
                assert reference is not None
                for actual, expected in (
                    (shape[local_index], reference.shape),
                    (support[local_index], reference.support),
                    (conflict[local_index], reference.conflict),
                ):
                    if math.isnan(expected):
                        status_mismatches += not math.isnan(actual)
                    else:
                        max_error = max(max_error, abs(float(actual) - expected))
                comparisons += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "channel": channel,
                            "hpo_id": prepared.hpo_ids[h_index],
                            "mpo_id": prepared.mpo_ids[m_index],
                            "shape": reference.shape,
                            "support": reference.support,
                            "conflict": reference.conflict,
                        }
                    )
    result = {
        "status": "PASS" if max_error <= 1e-12 and status_mismatches == 0 else "FAIL",
        "pair_comparison_count": comparisons,
        "maximum_absolute_error": max_error,
        "na_status_mismatch_count": status_mismatches,
        "runtime_seconds": time.monotonic() - started,
        "examples": examples,
    }
    write_json(output_root(config) / "logs/smoke_test.json", result)
    if result["status"] != "PASS":
        raise ValueError(f"optimized scorer failed smoke comparison: {result}")
    return result


def _create_matrix_file(
    path: Path,
    n_hpo: int,
    n_mpo: int,
    block_size: int,
    mpo_block_size: int,
    digest: str,
    config_sha: str,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "a")
    if "input_digest" in handle.attrs:
        if handle.attrs["input_digest"] != digest or handle.attrs["config_sha256"] != config_sha:
            handle.close()
            raise ValueError("existing matrix input/config digest does not match this run")
        return handle
    handle.attrs["schema_version"] = 1
    handle.attrs["input_digest"] = digest
    handle.attrs["config_sha256"] = config_sha
    handle.attrs["hpo_count"] = n_hpo
    handle.attrs["mpo_count"] = n_mpo
    chunks = (min(block_size, n_hpo), min(mpo_block_size, n_mpo))
    for channel in CHANNELS:
        group = handle.require_group(channel)
        for score_name in ("shape", "support", "conflict"):
            group.create_dataset(
                score_name,
                shape=(n_hpo, n_mpo),
                dtype="float32",
                chunks=chunks,
                compression="lzf",
                shuffle=True,
                fillvalue=np.nan,
            )
    handle.flush()
    return handle


def _completed_blocks(
    path: Path,
    input_digest: str,
    config_sha: str,
) -> set[tuple[str, int, int]]:
    if not path.is_file():
        return set()
    result = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if (
                    value.get("status") != "complete"
                    or value.get("input_digest") != input_digest
                    or value.get("config_sha256") != config_sha
                ):
                    continue
                result.add((value["channel"], int(value["row_start"]), int(value["row_end"])))
    return result


def _append_checkpoint(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_compute(config: dict[str, Any], resume: bool = True) -> dict[str, Any]:
    root = output_root(config)
    matrix_path = root / "matrices/hpo_mpo_cpo.h5"
    checkpoint_path = root / "checkpoints/completed_blocks.jsonl"
    input_digest = matrix_input_digest(config)
    config_sha = config_digest(config)
    prepared = load_prepared_scoring(config)
    n_hpo = len(prepared.hpo_ids)
    n_mpo = len(prepared.mpo_ids)
    block_size = int(config["runtime"]["hpo_block_size"])
    mpo_block_size = int(config["runtime"].get("mpo_block_size", 512))
    matrix_existed = matrix_path.is_file()
    completed = (
        _completed_blocks(checkpoint_path, input_digest, config_sha)
        if resume and matrix_existed
        else set()
    )
    handle = _create_matrix_file(
        matrix_path,
        n_hpo,
        n_mpo,
        block_size,
        mpo_block_size,
        input_digest,
        config_sha,
    )
    stage_started = time.monotonic()
    blocks_written = 0
    try:
        for channel, (h_class, m_class) in CHANNELS.items():
            h_profiles = prepared.profiles[("HPO", h_class)]
            m_profiles = prepared.profiles[("MPO", m_class)]
            for start in range(0, n_hpo, block_size):
                end = min(n_hpo, start + block_size)
                key = (channel, start, end)
                if key in completed:
                    continue
                block_started = time.monotonic()
                shape_block = np.full((end - start, n_mpo), np.nan, dtype=np.float32)
                support_block = np.full_like(shape_block, np.nan)
                conflict_block = np.full_like(shape_block, np.nan)
                for offset, h_index in enumerate(range(start, end)):
                    shape, support, conflict = _optimized_row_scores(
                        prepared, h_profiles[h_index], m_profiles
                    )
                    shape_block[offset] = shape
                    support_block[offset] = support
                    conflict_block[offset] = conflict
                handle[channel]["shape"][start:end, :] = shape_block
                handle[channel]["support"][start:end, :] = support_block
                handle[channel]["conflict"][start:end, :] = conflict_block
                handle.flush()
                record = {
                    "channel": channel,
                    "row_start": start,
                    "row_end": end,
                    "input_digest": input_digest,
                    "config_sha256": config_sha,
                    "finite_shape_count": int(np.isfinite(shape_block).sum()),
                    "finite_support_count": int(np.isfinite(support_block).sum()),
                    "elapsed_seconds": time.monotonic() - block_started,
                    "status": "complete",
                }
                _append_checkpoint(checkpoint_path, record)
                blocks_written += 1
    finally:
        handle.close()

    expected_blocks = len(CHANNELS) * math.ceil(n_hpo / block_size)
    final_completed = _completed_blocks(checkpoint_path, input_digest, config_sha)
    matching_completed = {
        value for value in final_completed if value[0] in CHANNELS and value[2] <= n_hpo
    }
    if len(matching_completed) != expected_blocks:
        raise ValueError(
            f"matrix has incomplete blocks: {len(matching_completed)}/{expected_blocks}"
        )
    metadata = {
        "schema_version": 1,
        "matrix_shape": [n_hpo, n_mpo],
        "channels": list(CHANNELS),
        "scores": ["shape", "support", "conflict"],
        "storage_dtype": "float32",
        "compute_dtype": "float64",
        "nan_semantics": {
            "shape": "one or both channel profiles are empty or target is excluded",
            "support_conflict": "empty channel profile or no comparable direction pair",
            "not_computed": "represented only by missing checkpoint blocks; final count must be zero",
        },
        "zero_semantics": {
            "shape": "both profiles exist but no positive structural match",
            "support": "comparable direction exists but no agreeing match contributes",
            "conflict": "comparable direction exists but no conflicting match contributes",
        },
        "input_digest": input_digest,
        "config_sha256": config_sha,
        "completed_block_count": len(matching_completed),
        "expected_block_count": expected_blocks,
        "not_computed_block_count": 0,
    }
    write_json(root / "matrices/score_status_metadata.json", metadata)
    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix": file_record(matrix_path),
        "metadata": metadata,
        "blocks_written_this_run": blocks_written,
        "runtime_seconds": time.monotonic() - stage_started,
    }
    write_json(root / "logs/compute_summary.json", result)
    return result
