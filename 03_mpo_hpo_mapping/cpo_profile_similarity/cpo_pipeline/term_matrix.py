from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from cpo_mapping.io import file_record, read_tsv, sha256, write_json
from cpo_mapping.ontology import Ontology

from .config import config_digest, output_root, resource_path


MATRIX_DATASETS = (
    "final/shape",
    "final/hpo_to_mpo",
    "final/mpo_to_hpo",
    "final/direction_support",
    "final/direction_conflict",
    "self_only/shape",
    "inherited_only/shape",
    "aux/cl",
    "aux/go",
)


@dataclass
class TermData:
    hpo_ids: list[str]
    mpo_ids: list[str]
    memberships: pd.DataFrame
    projections: pd.DataFrame
    cl: Ontology
    go: Ontology


def load_term_data(config: dict[str, Any]) -> TermData:
    root = output_root(config)
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    return TermData(
        hpo_ids=[row["term_id"] for row in hpo_rows],
        mpo_ids=[row["term_id"] for row in mpo_rows],
        memberships=pd.read_parquet(root / "profiles/structure_membership.parquet"),
        projections=pd.read_parquet(root / "profiles/projection_membership.parquet"),
        cl=Ontology.from_obo(resource_path(config, "cl_ontology"), "CL:0000000", "CL:"),
        go=Ontology.from_obo(
            resource_path(config, "go_ontology"),
            "GO:0008150",
            "GO:",
            "biological_process",
        ),
    )


def _lin_matrix(
    ontology: Ontology, left_ids: Sequence[str], right_ids: Sequence[str]
) -> np.ndarray:
    result = np.empty((len(left_ids), len(right_ids)), dtype=np.float32)
    for left_index, left_id in enumerate(left_ids):
        for right_index, right_id in enumerate(right_ids):
            result[left_index, right_index] = ontology.lin(left_id, right_id)
    return result


def _cpo_similarity(
    data: TermData,
) -> tuple[list[str], list[str], np.ndarray, dict[str, tuple[str, str]]]:
    structures = {
        str(row.structure_id): (str(row.cl_id), str(row.go_id))
        for row in data.memberships.itertuples(index=False)
    }
    h_structures = sorted(
        set(
            data.memberships.loc[
                data.memberships["ontology_side"] == "HPO", "structure_id"
            ].astype(str)
        )
    )
    m_structures = sorted(
        set(
            data.memberships.loc[
                data.memberships["ontology_side"] == "MPO", "structure_id"
            ].astype(str)
        )
    )
    h_cl_ids = sorted({structures[value][0] for value in h_structures})
    m_cl_ids = sorted({structures[value][0] for value in m_structures})
    h_go_ids = sorted({structures[value][1] for value in h_structures})
    m_go_ids = sorted({structures[value][1] for value in m_structures})
    cl_sim = _lin_matrix(data.cl, h_cl_ids, m_cl_ids)
    go_sim = _lin_matrix(data.go, h_go_ids, m_go_ids)
    h_cl = {value: index for index, value in enumerate(h_cl_ids)}
    m_cl = {value: index for index, value in enumerate(m_cl_ids)}
    h_go = {value: index for index, value in enumerate(h_go_ids)}
    m_go = {value: index for index, value in enumerate(m_go_ids)}
    h_cl_index = np.asarray([h_cl[structures[value][0]] for value in h_structures])
    m_cl_index = np.asarray([m_cl[structures[value][0]] for value in m_structures])
    h_go_index = np.asarray([h_go[structures[value][1]] for value in h_structures])
    m_go_index = np.asarray([m_go[structures[value][1]] for value in m_structures])
    similarity = (
        cl_sim[h_cl_index[:, None], m_cl_index[None, :]]
        * go_sim[h_go_index[:, None], m_go_index[None, :]]
    )
    return h_structures, m_structures, similarity, structures


def _profile_lists(
    frame: pd.DataFrame,
    view: str,
    hpo_ids: list[str],
    mpo_ids: list[str],
    h_components: list[str],
    m_components: list[str],
    component_field: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    selected = frame[frame["profile_view"] == view]
    h_index = {value: index for index, value in enumerate(h_components)}
    m_index = {value: index for index, value in enumerate(m_components)}
    grouped: dict[tuple[str, str], set[int]] = {}
    for row in selected.itertuples(index=False):
        side = str(row.ontology_side)
        term_id = str(row.term_id)
        component = str(getattr(row, component_field))
        index = h_index[component] if side == "HPO" else m_index[component]
        grouped.setdefault((side, term_id), set()).add(index)
    return (
        [
            np.asarray(sorted(grouped.get(("HPO", term_id), set())), dtype=np.int32)
            for term_id in hpo_ids
        ],
        [
            np.asarray(sorted(grouped.get(("MPO", term_id), set())), dtype=np.int32)
            for term_id in mpo_ids
        ],
    )


def _incidence(profiles: list[np.ndarray], component_count: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, profile in enumerate(profiles):
        if not len(profile):
            continue
        weight = 1.0 / len(profile)
        rows.extend([row_index] * len(profile))
        columns.extend(map(int, profile))
        values.extend([weight] * len(profile))
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(profiles), component_count), dtype=np.float64
    )


def _flat_profiles(
    profiles: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray([i for i, values in enumerate(profiles) if len(values)], dtype=np.int32)
    lengths = np.asarray([len(profiles[i]) for i in active], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.asarray([value for index in active for value in profiles[index]], dtype=np.int32)
    return active, offsets, flat


def _max_to_profiles(
    similarity: np.ndarray,
    target_profiles: list[np.ndarray],
    output_path: Path,
    row_block: int = 64,
) -> np.memmap:
    active, offsets, flat = _flat_profiles(target_profiles)
    result = np.memmap(
        output_path,
        mode="w+",
        dtype="float32",
        shape=(similarity.shape[0], len(target_profiles)),
    )
    result[:] = 0
    for start in range(0, similarity.shape[0], row_block):
        end = min(similarity.shape[0], start + row_block)
        values = similarity[start:end, :][:, flat]
        result[start:end, active] = np.maximum.reduceat(
            values, offsets[:-1], axis=1
        )
    result.flush()
    return result


def _create_matrix(path: Path, n_hpo: int, n_mpo: int, digest: str, config: dict[str, Any]) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "a")
    if "input_digest" in handle.attrs:
        if handle.attrs["input_digest"] != digest:
            handle.close()
            raise ValueError("v2 term matrix input digest mismatch")
        return handle
    handle.attrs["schema_version"] = 2
    handle.attrs["input_digest"] = digest
    handle.attrs["hpo_count"] = n_hpo
    handle.attrs["mpo_count"] = n_mpo
    chunks = (
        min(int(config["matrix"]["hpo_block_size"]), n_hpo),
        min(int(config["matrix"]["mpo_block_size"]), n_mpo),
    )
    for path_name in MATRIX_DATASETS:
        group_name, dataset_name = path_name.split("/")
        group = handle.require_group(group_name)
        group.create_dataset(
            dataset_name,
            shape=(n_hpo, n_mpo),
            dtype="float32",
            chunks=chunks,
            compression="lzf",
            shuffle=True,
            fillvalue=np.nan,
        )
    handle.flush()
    return handle


def _write_bma_view(
    handle: h5py.File,
    dataset_path: str,
    h_profiles: list[np.ndarray],
    m_profiles: list[np.ndarray],
    similarity: np.ndarray,
    temporary_root: Path,
    config: dict[str, Any],
    forward_path: str | None = None,
    reverse_path: str | None = None,
) -> dict[str, int]:
    n_hpo = len(h_profiles)
    n_mpo = len(m_profiles)
    h_active = np.asarray([i for i, values in enumerate(h_profiles) if len(values)], dtype=np.int32)
    m_active = np.asarray([i for i, values in enumerate(m_profiles) if len(values)], dtype=np.int32)
    h_incidence = _incidence(h_profiles, similarity.shape[0])
    m_incidence = _incidence(m_profiles, similarity.shape[1])
    r_m = _max_to_profiles(
        similarity,
        m_profiles,
        temporary_root / (dataset_path.replace("/", "_") + "_rm.dat"),
    )
    r_h = _max_to_profiles(
        similarity.T,
        h_profiles,
        temporary_root / (dataset_path.replace("/", "_") + "_rh.dat"),
    )
    block_size = int(config["matrix"]["hpo_block_size"])
    mpo_block = int(config["matrix"]["mpo_block_size"])
    forward_dataset = handle[forward_path] if forward_path else None
    reverse_dataset = handle[reverse_path] if reverse_path else None
    target = handle[dataset_path]
    with tempfile.NamedTemporaryFile(
        prefix="cpo_forward_", suffix=".dat", dir=temporary_root, delete=False
    ) as temporary:
        forward_file = Path(temporary.name)
    forward_memmap = np.memmap(
        forward_file, mode="w+", dtype="float32", shape=(n_hpo, n_mpo)
    )
    forward_memmap[:] = np.nan
    for start in range(0, n_mpo, mpo_block):
        end = min(n_mpo, start + mpo_block)
        raw = h_incidence @ np.asarray(r_m[:, start:end], dtype=np.float64)
        block = np.full((n_hpo, end - start), np.nan, dtype=np.float32)
        active_columns = m_active[(m_active >= start) & (m_active < end)] - start
        if len(active_columns):
            block[np.ix_(h_active, active_columns)] = raw[
                np.ix_(h_active, active_columns)
            ]
        forward_memmap[:, start:end] = block
        if forward_dataset is not None:
            forward_dataset[:, start:end] = block
    forward_memmap.flush()
    finite_count = 0
    positive_count = 0
    for start in range(0, n_hpo, block_size):
        end = min(n_hpo, start + block_size)
        raw_reverse = (m_incidence @ np.asarray(r_h[:, start:end], dtype=np.float64)).T
        reverse_block = np.full((end - start, n_mpo), np.nan, dtype=np.float32)
        active_rows = h_active[(h_active >= start) & (h_active < end)] - start
        if len(active_rows):
            reverse_block[np.ix_(active_rows, m_active)] = raw_reverse[
                np.ix_(active_rows, m_active)
            ]
        if reverse_dataset is not None:
            reverse_dataset[start:end, :] = reverse_block
        shape_block = (np.asarray(forward_memmap[start:end, :]) + reverse_block) / 2
        target[start:end, :] = shape_block
        finite_count += int(np.isfinite(shape_block).sum())
        positive_count += int(np.count_nonzero(np.isfinite(shape_block) & (shape_block > 0)))
    handle.flush()
    del forward_memmap, r_m, r_h
    forward_file.unlink(missing_ok=True)
    return {
        "hpo_profile_count": len(h_active),
        "mpo_profile_count": len(m_active),
        "finite_pair_count": finite_count,
        "positive_pair_count": positive_count,
    }


def _direction_matrices(
    handle: h5py.File,
    final_membership: pd.DataFrame,
    hpo_ids: list[str],
    mpo_ids: list[str],
    config: dict[str, Any],
) -> dict[str, int]:
    h_index = {value: index for index, value in enumerate(hpo_ids)}
    m_index = {value: index for index, value in enumerate(mpo_ids)}
    unique = final_membership[
        final_membership["direction_signature_status"] == "unique"
    ]
    tokens = sorted(
        {
            (
                str(row.structure_id),
                str(row.direction_target),
                str(row.direction_property),
                str(row.direction_state),
            )
            for row in unique.itertuples(index=False)
        }
    )
    token_index = {value: index for index, value in enumerate(tokens)}
    opposite = {
        "increased": "decreased",
        "decreased": "increased",
        "present": "absent",
        "absent": "present",
    }

    def incidence(side: str, conflict: bool = False) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        for row in unique[unique["ontology_side"] == side].itertuples(index=False):
            state = str(row.direction_state)
            if conflict:
                if state not in opposite:
                    continue
                state = opposite[state]
            token = (
                str(row.structure_id),
                str(row.direction_target),
                str(row.direction_property),
                state,
            )
            if token not in token_index:
                continue
            rows.append(
                h_index[str(row.term_id)] if side == "HPO" else m_index[str(row.term_id)]
            )
            columns.append(token_index[token])
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, columns)),
            shape=(len(hpo_ids) if side == "HPO" else len(mpo_ids), len(tokens)),
        )

    h = incidence("HPO")
    m = incidence("MPO")
    m_conflict = incidence("MPO", conflict=True)
    agree = (h @ m.T).tocsr()
    conflict = (h @ m_conflict.T).tocsr()
    h_sizes = final_membership[final_membership["ontology_side"] == "HPO"].groupby("term_id").size()
    m_sizes = final_membership[final_membership["ontology_side"] == "MPO"].groupby("term_id").size()
    h_count = np.asarray([int(h_sizes.get(value, 0)) for value in hpo_ids])
    m_count = np.asarray([int(m_sizes.get(value, 0)) for value in mpo_ids])
    block_size = int(config["matrix"]["hpo_block_size"])
    comparable_count = 0
    support_positive = 0
    conflict_positive = 0
    for start in range(0, len(hpo_ids), block_size):
        end = min(len(hpo_ids), start + block_size)
        agree_block = agree[start:end, :].toarray().astype(np.float64)
        conflict_block = conflict[start:end, :].toarray().astype(np.float64)
        comparable = (agree_block + conflict_block) > 0
        weights = 0.5 * (
            np.divide(
                1.0,
                h_count[start:end, None],
                out=np.zeros((end - start, 1)),
                where=h_count[start:end, None] > 0,
            )
            + np.divide(
                1.0,
                m_count[None, :],
                out=np.zeros((1, len(mpo_ids))),
                where=m_count[None, :] > 0,
            )
        )
        support = np.where(comparable, agree_block * weights, np.nan).astype(np.float32)
        conflicts = np.where(comparable, conflict_block * weights, np.nan).astype(np.float32)
        handle["final/direction_support"][start:end, :] = support
        handle["final/direction_conflict"][start:end, :] = conflicts
        comparable_count += int(comparable.sum())
        support_positive += int(np.count_nonzero(support > 0))
        conflict_positive += int(np.count_nonzero(conflicts > 0))
    handle.flush()
    return {
        "comparable_pair_count": comparable_count,
        "positive_support_pair_count": support_positive,
        "positive_conflict_pair_count": conflict_positive,
    }


def _input_digest(config: dict[str, Any]) -> str:
    root = output_root(config)
    paths = (
        root / "profiles/structure_membership.parquet",
        root / "profiles/projection_membership.parquet",
        root / "indices/hpo_index.tsv",
        root / "indices/mpo_index.tsv",
        resource_path(config, "cl_ontology"),
        resource_path(config, "go_ontology"),
    )
    payload = "\n".join(f"{path.resolve()}\t{sha256(path)}" for path in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_term_matrix(config: dict[str, Any], resume: bool = True) -> dict[str, object]:
    root = output_root(config)
    data = load_term_data(config)
    digest = _input_digest(config)
    matrix_path = root / "matrices/hpo_mpo_cpo.h5"
    manifest_path = root / "matrices/scoring_manifest.json"
    if resume and matrix_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_digest") == digest and manifest.get("status") == "complete":
            return {**manifest, "resumed_without_recompute": True}
    handle = _create_matrix(
        matrix_path, len(data.hpo_ids), len(data.mpo_ids), digest, config
    )
    started = time.monotonic()
    view_counts: dict[str, dict[str, int]] = {}
    h_structures, m_structures, cpo_similarity, _structure_map = _cpo_similarity(data)
    with tempfile.TemporaryDirectory(prefix="cpo_v2_matrix_") as temporary:
        temporary_root = Path(temporary)
        for view, dataset, forward, reverse in (
            (
                "final",
                "final/shape",
                "final/hpo_to_mpo",
                "final/mpo_to_hpo",
            ),
            ("self_only", "self_only/shape", None, None),
            ("inherited_only", "inherited_only/shape", None, None),
        ):
            h_profiles, m_profiles = _profile_lists(
                data.memberships,
                view,
                data.hpo_ids,
                data.mpo_ids,
                h_structures,
                m_structures,
                "structure_id",
            )
            view_counts[view] = _write_bma_view(
                handle,
                dataset,
                h_profiles,
                m_profiles,
                cpo_similarity,
                temporary_root,
                config,
                forward,
                reverse,
            )
        final_membership = data.memberships[
            data.memberships["profile_view"] == "final"
        ]
        direction_counts = _direction_matrices(
            handle, final_membership, data.hpo_ids, data.mpo_ids, config
        )
        for dimension, ontology, dataset in (
            ("CL", data.cl, "aux/cl"),
            ("GO", data.go, "aux/go"),
        ):
            projection = data.projections[data.projections["dimension"] == dimension]
            h_components = sorted(
                set(
                    projection.loc[
                        projection["ontology_side"] == "HPO", "component_id"
                    ].astype(str)
                )
            )
            m_components = sorted(
                set(
                    projection.loc[
                        projection["ontology_side"] == "MPO", "component_id"
                    ].astype(str)
                )
            )
            similarity = _lin_matrix(ontology, h_components, m_components)
            h_profiles, m_profiles = _profile_lists(
                projection,
                "final",
                data.hpo_ids,
                data.mpo_ids,
                h_components,
                m_components,
                "component_id",
            )
            view_counts[f"aux_{dimension.lower()}"] = _write_bma_view(
                handle,
                dataset,
                h_profiles,
                m_profiles,
                similarity,
                temporary_root,
                config,
            )
    handle.close()
    result = {
        "schema_version": 2,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_digest": digest,
        "config_sha256": config_digest(config),
        "matrix_shape": [len(data.hpo_ids), len(data.mpo_ids)],
        "datasets": list(MATRIX_DATASETS),
        "view_counts": view_counts,
        "direction_counts": direction_counts,
        "runtime_seconds": time.monotonic() - started,
        "matrix": file_record(matrix_path),
        "resumed_without_recompute": False,
    }
    write_json(manifest_path, result)
    write_json(
        root / "matrices/status_metadata.json",
        {
            "nan": "one or both selected profile views lack a complete structure, or no comparable direction for direction datasets",
            "zero": "both profiles exist but no positive match for the selected score",
            "not_computed": "scoring_manifest status is not complete",
            "missing_required_self": "separate node-policy flag; inherited final remains scoreable",
            "datasets": list(MATRIX_DATASETS),
        },
    )
    return result
