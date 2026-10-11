from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import config_hash, result_root
from .fusion import fuse_array
from .io import file_record, stable_digest, write_json, write_tsv
from .matrices import (
    WorkingProfiles,
    atom_kernel,
    bma_surface,
    load_working_profiles,
    node_kernel,
    q_matrix,
    semantic_surface,
    solver_surface,
    text_kernels,
)
from .mission3_adapter import evaluate_surface, task_hpo_ids
from .text import encode_all


def _require_frozen(resources: dict[str, Any]) -> None:
    frozen = result_root(resources) / "frozen_inputs" / "input_manifest.json"
    if not frozen.is_file():
        raise FileNotFoundError("freeze-inputs must complete before mechanism data are loaded")
    value = json.loads(frozen.read_text(encoding="utf-8"))
    if value.get("frozen") is not True or value.get("training_enabled") is not False:
        raise ValueError("invalid frozen input manifest")


def _append_log(root: Path, value: dict[str, object]) -> None:
    path = root / "method_selection_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _surface_cache(
    resources: dict[str, Any], hpo_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    cache = result_root(resources) / "cache"
    path = cache / "surface_S.npy"
    index_path = cache / "surface_index.json"
    if path.is_file() and index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["hpo_ids"] == hpo_ids:
            return np.load(path, mmap_mode="r"), index["mpo_ids"]
    surface, mpo_ids = semantic_surface(resources, hpo_ids)
    np.save(path, surface)
    write_json(
        index_path,
        {
            "hpo_ids": hpo_ids,
            "mpo_ids": mpo_ids,
            "score": "max(0, cosine)",
            "surface": file_record(path),
        },
    )
    return surface, mpo_ids


def _evaluate_candidate(
    resources: dict[str, Any],
    registry: dict[str, Any],
    root: Path,
    method_id: str,
    s: np.ndarray,
    m: np.ndarray,
    q: np.ndarray,
    hpo_ids: list[str],
    mpo_ids: list[str],
    fusion_methods: tuple[str, ...] = ("positive", "linear"),
    r_s: np.ndarray | None = None,
    r_m: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    epsilon = float(registry["fusion"]["epsilon"])
    for fusion in fusion_methods:
        started = time.monotonic()
        surface = fuse_array(fusion, s, m, q, epsilon, r_s, r_m)
        surface_id = f"{method_id}__{fusion}"
        evaluation = evaluate_surface(
            resources,
            surface_id,
            surface,
            hpo_ids,
            mpo_ids,
            root / method_id / fusion,
        )
        metrics = evaluation["metrics"]
        row = {
            "method_id": method_id,
            "fusion": fusion,
            "registered_order": _registered_order(registry, method_id, fusion),
            "mrr": metrics["mrr"],
            "ndcg_at_10": metrics["ndcg_at_10"],
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_5": metrics["recall_at_5"],
            "recall_at_10": metrics["recall_at_10"],
            "runtime_seconds": time.monotonic() - started,
            "computed_pair_count": int(np.isfinite(m).sum()),
            "q_zero_pair_count": int(np.count_nonzero(q == 0)),
            "exact_fallback_pair_count": int(np.count_nonzero(~np.isfinite(m) | (q == 0))),
            "mean_absolute_delta": float(np.abs(surface - s).mean()),
            "maximum_absolute_delta": float(np.abs(surface - s).max()),
        }
        if row["maximum_absolute_delta"] > epsilon + 1e-6:
            raise ValueError(f"fusion delta budget violated: {surface_id}")
        rows.append(row)
        _append_log(
            result_root(resources),
            {
                "event": "candidate_evaluated",
                "stage": root.name,
                "surface_id": surface_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **row,
            },
        )
    return rows


def _registered_order(
    registry: dict[str, Any], method_id: str, fusion: str
) -> int:
    node, atom, profile_method = _parse_method(method_id)
    return (
        registry["ontology"]["node_methods"].index(node) * 1000
        + registry["atoms"]["methods"].index(atom) * 100
        + registry["profiles"]["methods"].index(profile_method) * 10
        + registry["fusion"]["methods"].index(fusion)
    )


def _selection_key(row: dict[str, object]) -> tuple[float, float, float, int, float, str]:
    return (
        -float(row["mrr"]),
        -float(row["ndcg_at_10"]),
        -float(row["recall_at_1"]),
        int(row["registered_order"]),
        float(row["runtime_seconds"]),
        str(row["method_id"]),
    )


def _write_stage(
    output: Path,
    rows: list[dict[str, object]],
    finalists: list[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    fields = (
        "method_id",
        "fusion",
        "registered_order",
        "mrr",
        "ndcg_at_10",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "runtime_seconds",
        "computed_pair_count",
        "q_zero_pair_count",
        "exact_fallback_pair_count",
        "mean_absolute_delta",
        "maximum_absolute_delta",
    )
    write_tsv(output / "stage_metrics.tsv", rows, fields)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "candidate_count": len({row["method_id"] for row in rows}),
        "surface_count": len(rows),
        "finalists": finalists,
        **metadata,
    }
    write_json(output / "selection.json", manifest)
    return manifest


def _method_best(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_id"]), []).append(row)
    return [min(values, key=_selection_key) for values in grouped.values()]


def _save_mechanism(path: Path, value: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32))
    write_json(
        path.with_suffix(".json"),
        {
            **metadata,
            "shape": list(value.shape),
            "finite_count": int(np.isfinite(value).sum()),
            "zero_count": int(np.count_nonzero(value == 0)),
            "positive_count": int(np.count_nonzero(value > 0)),
            "output": file_record(path),
        },
    )


def _base_context(
    resources: dict[str, Any], registry: dict[str, Any]
) -> tuple[list[str], list[str], np.ndarray, WorkingProfiles, np.ndarray]:
    hpo_ids = task_hpo_ids(resources)
    s, mpo_ids = _surface_cache(resources, hpo_ids)
    profiles = load_working_profiles(resources, hpo_ids)
    if profiles.mpo_ids != mpo_ids:
        raise ValueError("semantic and mechanism MPO indices differ")
    q = q_matrix(profiles)
    cache = result_root(resources) / "cache"
    q_path = cache / "Q_availability.npy"
    if not q_path.is_file():
        np.save(q_path, q)
        write_json(
            q_path.with_suffix(".json"),
            {
                "quality_id": registry["quality"]["id"],
                "interpretation": registry["quality"]["interpretation"],
                "shape": list(q.shape),
                "positive_count": int(np.count_nonzero(q > 0)),
                "output": file_record(q_path),
            },
        )
    return hpo_ids, mpo_ids, np.asarray(s), profiles, q


def run_nodes(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources) / "r1_nodes"
    hpo_ids, mpo_ids, s, profiles, q = _base_context(resources, registry)
    p0 = evaluate_surface(resources, "P0", s, hpo_ids, mpo_ids, root / "P0")
    rows: list[dict[str, object]] = []
    for node_method in registry["ontology"]["node_methods"]:
        started = time.monotonic()
        kernel, _, _ = node_kernel(resources, profiles, node_method)
        m = bma_surface(resources, profiles, kernel)
        method_id = f"node_{node_method}__atom_A0__profile_bma"
        _save_mechanism(
            root / "mechanisms" / f"{method_id}.npy",
            m,
            {
                "node_method": node_method,
                "atom_method": "A0",
                "profile_method": "bma",
                "kernel_runtime_seconds": time.monotonic() - started,
            },
        )
        rows.extend(
            _evaluate_candidate(
                resources, registry, root, method_id, s, m, q, hpo_ids, mpo_ids
            )
        )
    best = sorted(_method_best(rows), key=_selection_key)
    return _write_stage(
        root,
        rows,
        [best[0]],
        {
            "baseline_P0": p0,
            "selection_rule": "max(MRR(F+), MRR(FL)); then NDCG@10, Recall@1, registered cost/method order, measured runtime",
            "config_hash": config_hash(resources, registry),
        },
    )


def _node_from_selection(root: Path) -> str:
    value = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    method_id = value["finalists"][0]["method_id"]
    return method_id.split("__")[0].removeprefix("node_")


def run_atoms(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    base = result_root(resources)
    r1 = base / "r1_nodes"
    if not (r1 / "selection.json").is_file():
        raise FileNotFoundError("nodes stage must complete first")
    node_method = _node_from_selection(r1)
    root = base / "r2_atoms"
    hpo_ids, mpo_ids, s, profiles, q = _base_context(resources, registry)
    ontology_shape, _, _ = node_kernel(resources, profiles, node_method)
    component_text, cpo_text = text_kernels(resources, profiles)
    rows: list[dict[str, object]] = []
    for atom_method in registry["atoms"]["methods"]:
        started = time.monotonic()
        kernel = atom_kernel(atom_method, ontology_shape, component_text, cpo_text)
        m = bma_surface(resources, profiles, kernel)
        method_id = f"node_{node_method}__atom_{atom_method}__profile_bma"
        _save_mechanism(
            root / "mechanisms" / f"{method_id}.npy",
            m,
            {
                "node_method": node_method,
                "atom_method": atom_method,
                "profile_method": "bma",
                "kernel_runtime_seconds": time.monotonic() - started,
            },
        )
        rows.extend(
            _evaluate_candidate(
                resources, registry, root, method_id, s, m, q, hpo_ids, mpo_ids
            )
        )
    best = sorted(_method_best(rows), key=_selection_key)
    finalists = best[: int(registry["selection"]["atom_finalists"])]
    return _write_stage(
        root,
        rows,
        finalists,
        {
            "selected_node_method": node_method,
            "selection_rule": "top two atom methods by max anchor MRR",
        },
    )


def _parse_method(method_id: str) -> tuple[str, str, str]:
    values = method_id.split("__")
    return (
        values[0].removeprefix("node_"),
        values[1].removeprefix("atom_"),
        values[2].removeprefix("profile_"),
    )


def _kernel_for(
    resources: dict[str, Any],
    profiles: WorkingProfiles,
    node_method: str,
    atom_method: str,
) -> np.ndarray:
    ontology_shape, _, _ = node_kernel(resources, profiles, node_method)
    if atom_method in {"A0", "A1"}:
        return atom_kernel(atom_method, ontology_shape)
    component, cpo = text_kernels(resources, profiles)
    return atom_kernel(atom_method, ontology_shape, component, cpo)


def _mechanism_for_method(
    resources: dict[str, Any], profiles: WorkingProfiles, method_id: str
) -> np.ndarray:
    node_method, atom_method, profile_method = _parse_method(method_id)
    kernel = _kernel_for(resources, profiles, node_method, atom_method)
    if profile_method == "bma":
        return bma_surface(resources, profiles, kernel)
    if profile_method == "ic_bma":
        return bma_surface(resources, profiles, kernel, weighted=True)
    return solver_surface(profiles, kernel, profile_method, resources)


def _panel_ids(
    ids: list[str], available: np.ndarray, maximum: int, seed: str
) -> list[str]:
    candidates = [value for value, keep in zip(ids, available) if keep]
    return sorted(
        candidates,
        key=lambda value: (hashlib.sha256(f"{seed}\t{value}".encode()).hexdigest(), value),
    )[:maximum]


def _directional_midcdf(
    query: np.ndarray,
    row_background: np.ndarray,
    column_background: np.ndarray,
) -> np.ndarray:
    output_row = np.full(query.shape, np.nan, dtype=np.float32)
    output_column = np.full(query.shape, np.nan, dtype=np.float32)
    for index in range(query.shape[0]):
        background = np.sort(row_background[index][np.isfinite(row_background[index])])
        finite = np.isfinite(query[index])
        if len(background) and np.any(finite):
            values = query[index, finite]
            left = np.searchsorted(background, values, side="left")
            right = np.searchsorted(background, values, side="right")
            output_row[index, finite] = (left + 0.5 * (right - left)) / len(background)
    for index in range(query.shape[1]):
        background = np.sort(column_background[:, index][np.isfinite(column_background[:, index])])
        finite = np.isfinite(query[:, index])
        if len(background) and np.any(finite):
            values = query[finite, index]
            left = np.searchsorted(background, values, side="left")
            right = np.searchsorted(background, values, side="right")
            output_column[finite, index] = (left + 0.5 * (right - left)) / len(background)
    result = np.full(query.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(output_row) & np.isfinite(output_column)
    result[valid] = (output_row[valid] + output_column[valid]) / 2.0
    return result


def _rank_references(
    resources: dict[str, Any],
    registry: dict[str, Any],
    task_profiles: WorkingProfiles,
    s: np.ndarray,
    m: np.ndarray,
    method_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.load(result_root(resources) / "cache" / "self_profiles.npz", allow_pickle=False)
    maximum = int(registry["rank_reference"]["max_terms_per_ontology"])
    seed = str(registry["rank_reference"]["hash_seed"])
    h_panel = _panel_ids(
        list(values["h_term_ids"]), values["h_qualified_counts"] > 0, maximum, seed + ":HPO"
    )
    m_panel = _panel_ids(
        list(values["m_term_ids"]), values["m_qualified_counts"] > 0, maximum, seed + ":MPO"
    )
    panel_profiles = load_working_profiles(resources, h_panel)
    panel_s, panel_mpo_ids = semantic_surface(resources, h_panel)
    if panel_mpo_ids != task_profiles.mpo_ids:
        raise ValueError("rank panel and task MPO indices differ")
    panel_m = _mechanism_for_method(resources, panel_profiles, method_id)
    m_index = {value: index for index, value in enumerate(task_profiles.mpo_ids)}
    panel_columns = [m_index[value] for value in m_panel]
    r_s = _directional_midcdf(s, s[:, panel_columns], panel_s)
    r_m = _directional_midcdf(m, m[:, panel_columns], panel_m)
    manifest = {
        "reference_panel_id": stable_digest({"hpo": h_panel, "mpo": m_panel, "seed": seed}),
        "mode": "fixed_hash_panel_percentile",
        "hpo_ids": h_panel,
        "mpo_ids": m_panel,
        "hpo_count": len(h_panel),
        "mpo_count": len(m_panel),
        "method_id": method_id,
        "s_rank_available": int(np.isfinite(r_s).sum()),
        "m_rank_available": int(np.isfinite(r_m).sum()),
    }
    return r_s, r_m, manifest


def run_profiles(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    base = result_root(resources)
    r2 = base / "r2_atoms"
    if not (r2 / "selection.json").is_file():
        raise FileNotFoundError("atoms stage must complete first")
    atom_selection = json.loads((r2 / "selection.json").read_text(encoding="utf-8"))
    root = base / "r3_profiles"
    hpo_ids, mpo_ids, s, profiles, q = _base_context(resources, registry)
    rows: list[dict[str, object]] = []
    preflight_rows: list[dict[str, object]] = []
    for finalist in atom_selection["finalists"]:
        node_method, atom_method, _ = _parse_method(finalist["method_id"])
        kernel = _kernel_for(resources, profiles, node_method, atom_method)
        for profile_method in registry["profiles"]["methods"]:
            started = time.monotonic()
            if profile_method == "bma":
                m = bma_surface(resources, profiles, kernel)
            elif profile_method == "ic_bma":
                m = bma_surface(resources, profiles, kernel, weighted=True)
            else:
                active_h = [value for value in profiles.h_profiles if len(value)][:25]
                active_m = [value for value in profiles.m_profiles if len(value)][:40]
                sample = WorkingProfiles(
                    hpo_ids=profiles.hpo_ids[: len(active_h)],
                    mpo_ids=profiles.mpo_ids[: len(active_m)],
                    h_atoms=profiles.h_atoms,
                    m_atoms=profiles.m_atoms,
                    h_profiles=active_h,
                    m_profiles=active_m,
                    h_q=profiles.h_q[: len(active_h)],
                    m_q=profiles.m_q[: len(active_m)],
                )
                pilot_started = time.monotonic()
                solver_surface(sample, kernel, profile_method, resources)
                preflight_rows.append(
                    {
                        "atom_method": atom_method,
                        "profile_method": profile_method,
                        "sample_pairs": len(active_h) * len(active_m),
                        "runtime_seconds": time.monotonic() - pilot_started,
                    }
                )
                m = solver_surface(profiles, kernel, profile_method, resources)
            method_id = f"node_{node_method}__atom_{atom_method}__profile_{profile_method}"
            _save_mechanism(
                root / "mechanisms" / f"{method_id}.npy",
                m,
                {
                    "node_method": node_method,
                    "atom_method": atom_method,
                    "profile_method": profile_method,
                    "runtime_seconds": time.monotonic() - started,
                },
            )
            rows.extend(
                _evaluate_candidate(
                    resources, registry, root, method_id, s, m, q, hpo_ids, mpo_ids
                )
            )
    write_tsv(
        root / "solver_preflight.tsv",
        preflight_rows,
        ("atom_method", "profile_method", "sample_pairs", "runtime_seconds"),
    )
    best = sorted(_method_best(rows), key=_selection_key)
    finalists = best[: int(registry["selection"]["mechanism_finalists"])]
    return _write_stage(
        root,
        rows,
        finalists,
        {
            "selection_rule": "top two complete mechanism methods by max anchor MRR",
            "solver_preflight": file_record(root / "solver_preflight.tsv"),
        },
    )


def run_fusion(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    base = result_root(resources)
    r3 = base / "r3_profiles"
    if not (r3 / "selection.json").is_file():
        raise FileNotFoundError("profiles stage must complete first")
    selection = json.loads((r3 / "selection.json").read_text(encoding="utf-8"))
    root = base / "r4_fusion"
    hpo_ids, mpo_ids, s, _profiles, q = _base_context(resources, registry)
    rows: list[dict[str, object]] = []
    for finalist in selection["finalists"]:
        method_id = finalist["method_id"]
        m = np.load(r3 / "mechanisms" / f"{method_id}.npy", mmap_mode="r")
        rows.extend(
            _evaluate_candidate(
                resources,
                registry,
                root,
                method_id,
                s,
                m,
                q,
                hpo_ids,
                mpo_ids,
                ("positive", "linear", "bounded_geometric"),
            )
        )
        r_s, r_m, panel = _rank_references(
            resources, registry, _profiles, s, np.asarray(m), method_id
        )
        panel_root = root / "rank_references" / method_id
        panel_root.mkdir(parents=True, exist_ok=True)
        np.save(panel_root / "R_S.npy", r_s)
        np.save(panel_root / "R_M.npy", r_m)
        panel["r_s"] = file_record(panel_root / "R_S.npy")
        panel["r_m"] = file_record(panel_root / "R_M.npy")
        write_json(panel_root / "panel_manifest.json", panel)
        rows.extend(
            _evaluate_candidate(
                resources,
                registry,
                root,
                method_id,
                s,
                m,
                q,
                hpo_ids,
                mpo_ids,
                ("amplitude_gated_rank",),
                r_s,
                r_m,
            )
        )
    best = sorted(rows, key=_selection_key)
    return _write_stage(
        root,
        rows,
        [best[0]],
        {
            "selection_rule": "best O0 development MRR; then registered tie breakers",
            "rank_fusion_status": "fixed_hash_panel_percentile_complete",
        },
    )


def run_command(
    command: str,
    resources: dict[str, Any],
    registry: dict[str, Any],
    adapter: dict[str, Any],
    stage: str | None = None,
) -> dict[str, object]:
    _require_frozen(resources)
    if command == "encode":
        return encode_all(resources, registry)
    if command == "run-stage" and stage == "nodes":
        return run_nodes(resources, registry)
    if command == "run-stage" and stage == "atoms":
        return run_atoms(resources, registry)
    if command == "run-stage" and stage == "profiles":
        return run_profiles(resources, registry)
    if command == "run-stage" and stage == "fusion":
        return run_fusion(resources, registry)
    if command == "run-controls":
        from .confirmation import run_confirmation
        from .controls import run_controls

        confirmation = result_root(resources) / "r5_confirmation/confirmation_manifest.json"
        if not confirmation.is_file():
            run_confirmation(resources, registry)
        return run_controls(resources, registry)
    if command == "report":
        from .reporting import build_report

        return build_report(resources, registry, adapter)
    if command == "freeze-method":
        from .reporting import freeze_method

        return freeze_method(resources, registry, adapter)
    raise NotImplementedError(f"{command} {stage or ''}".strip())


def external_test(resources: dict[str, Any], cohort: str | Path) -> dict[str, object]:
    method = result_root(resources) / "frozen_method" / "frozen_method.json"
    if not method.is_file():
        raise FileNotFoundError("freeze-method must complete before external-test")
    cohort_path = Path(cohort).resolve()
    if not cohort_path.is_file():
        raise FileNotFoundError(cohort_path)
    raise NotImplementedError(
        "A cohort was supplied, but its adapter is not registered; development selection was not rerun."
    )
