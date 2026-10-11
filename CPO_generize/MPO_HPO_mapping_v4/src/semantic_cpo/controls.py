from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve, result_root
from .evaluation import evaluate_rankings
from .fusion import fuse_array
from .io import file_record, write_json, write_tsv
from .matrices import (
    WorkingProfiles,
    bma_surface,
    direction_mask,
    load_working_profiles,
    q_matrix,
    text_kernels,
)
from .mission3_adapter import load_task, score_surface, task_hpo_ids


def _permute_exact_size(
    items: list[np.ndarray], q: np.ndarray, random: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray, int]:
    output = list(items)
    q_output = q.copy()
    movable = 0
    groups: dict[int, list[int]] = defaultdict(list)
    for index, profile in enumerate(items):
        groups[len(profile)].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        permutation = random.permutation(indices)
        movable += len(indices)
        for target, source in zip(indices, permutation):
            output[target] = items[int(source)]
            q_output[target] = q[int(source)]
    return output, q_output, movable


def _rewire(
    profiles: list[np.ndarray], random: np.random.Generator, multiplier: int = 10
) -> tuple[list[np.ndarray], int, int]:
    edges = [(term, int(atom)) for term, values in enumerate(profiles) for atom in values]
    edge_set = set(edges)
    requested = multiplier * len(edges)
    accepted = 0
    if len(edges) < 2:
        return list(profiles), requested, accepted
    for _ in range(requested):
        left, right = random.integers(0, len(edges), size=2)
        if left == right:
            continue
        term_a, atom_a = edges[int(left)]
        term_b, atom_b = edges[int(right)]
        if term_a == term_b or atom_a == atom_b:
            continue
        new_a = (term_a, atom_b)
        new_b = (term_b, atom_a)
        if new_a in edge_set or new_b in edge_set:
            continue
        edge_set.remove((term_a, atom_a))
        edge_set.remove((term_b, atom_b))
        edge_set.add(new_a)
        edge_set.add(new_b)
        edges[int(left)] = new_a
        edges[int(right)] = new_b
        accepted += 1
    grouped: dict[int, list[int]] = defaultdict(list)
    for term, atom in edges:
        grouped[term].append(atom)
    output = [np.asarray(sorted(grouped[index]), dtype=np.int32) for index in range(len(profiles))]
    return output, requested, accepted


def _direction_atoms(atoms: list[Any], random: np.random.Generator) -> tuple[list[Any], int]:
    known = {"increased", "decreased", "present", "absent"}
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        if atom.direction_state in known and atom.direction_target:
            groups[(atom.direction_target, atom.direction_property)].append(index)
    output = list(atoms)
    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        states = [atoms[index].direction_state for index in indices]
        random.shuffle(states)
        for index, state in zip(indices, states):
            changed += int(state != atoms[index].direction_state)
            output[index] = replace(atoms[index], direction_state=state)
    return output, changed


def _evaluate(
    task: Any,
    surface: np.ndarray,
    hpo_ids: list[str],
    mpo_ids: list[str],
) -> dict[str, float]:
    rankings = score_surface(task, surface, hpo_ids, mpo_ids)
    _, metrics = evaluate_rankings(rankings, task.gold)
    return metrics


def _summary(rows: list[dict[str, object]], observed: float) -> list[dict[str, object]]:
    result = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["control"])].append(float(row["delta_vs_P0_mrr"]))
    for control, values in sorted(grouped.items()):
        array = np.asarray(values)
        result.append(
            {
                "control": control,
                "repeats": len(array),
                "null_mean_delta_mrr": float(array.mean()),
                "null_ci_low": float(np.quantile(array, 0.025)),
                "null_ci_high": float(np.quantile(array, 0.975)),
                "observed_delta_mrr": observed,
                "null_at_least_observed": int(np.count_nonzero(array >= observed)),
                "empirical_p_upper": (int(np.count_nonzero(array >= observed)) + 1) / (len(array) + 1),
            }
        )
    return result


def _density_report(
    output: Path,
    s: np.ndarray,
    m: np.ndarray,
    o: np.ndarray,
    q: np.ndarray,
    candidate_columns: list[int],
) -> dict[str, object]:
    def stats(values: np.ndarray) -> dict[str, object]:
        finite = values[np.isfinite(values)]
        return {
            "total": values.size,
            "finite": len(finite),
            "missing": int(np.count_nonzero(~np.isfinite(values))),
            "zero": int(np.count_nonzero(finite == 0)),
            "positive": int(np.count_nonzero(finite > 0)),
            "quantiles": {
                str(key): float(np.quantile(finite, key))
                for key in (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
            },
        }

    candidate_s = s[:, candidate_columns]
    candidate_m = m[:, candidate_columns]
    candidate_o = o[:, candidate_columns]
    margins: dict[str, dict[str, float]] = {}
    for name, values in (("S", candidate_s), ("M", candidate_m), ("F", candidate_o)):
        top1_top2: list[float] = []
        top1_top10: list[float] = []
        variances: list[float] = []
        for row in values:
            finite = row[np.isfinite(row)]
            if not len(finite):
                continue
            ordered = np.sort(finite)[::-1]
            variances.append(float(np.var(finite)))
            if len(ordered) >= 2:
                top1_top2.append(float(ordered[0] - ordered[1]))
            if len(ordered) >= 10:
                top1_top10.append(float(ordered[0] - ordered[9]))
        margins[name] = {
            "mean_row_variance": float(np.mean(variances)) if variances else 0.0,
            "mean_top1_top2_margin": float(np.mean(top1_top2)) if top1_top2 else 0.0,
            "mean_top1_top10_margin": float(np.mean(top1_top10)) if top1_top10 else 0.0,
        }
    report = {
        "all_pairs": {"S": stats(s), "M": stats(m), "F": stats(o), "Q": stats(q)},
        "candidate_pairs": {"S": stats(candidate_s), "M": stats(candidate_m), "F": stats(candidate_o)},
        "row_diagnostics": margins,
    }
    write_json(output / "density_coverage_bias_report.json", report)
    lines = [
        "# Density, coverage and bias report",
        "",
        f"- S computable: {report['all_pairs']['S']['finite']:,}/{report['all_pairs']['S']['total']:,}.",
        f"- M computable: {report['all_pairs']['M']['finite']:,}/{report['all_pairs']['M']['total']:,}; missing mechanism is not zero.",
        f"- Q=0 pairs: {report['all_pairs']['Q']['zero']:,}.",
        f"- F exact fallback pairs: {int(np.count_nonzero(~np.isfinite(m) | (q == 0))):,}.",
        "",
        "All S/M/F quantiles and candidate-pair margins are in `density_coverage_bias_report.json`. Q is availability, not causal confidence.",
    ]
    (output / "density_coverage_bias_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_controls(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources)
    confirmation = json.loads((root / "r5_confirmation/confirmation_manifest.json").read_text(encoding="utf-8"))
    selected = confirmation["selected_surface"]
    method_id, fusion = selected.rsplit("__", 1)
    index = json.loads((root / "cache/surface_index.json").read_text(encoding="utf-8"))
    hpo_ids = index["hpo_ids"]
    mpo_ids = index["mpo_ids"]
    profiles = load_working_profiles(resources, hpo_ids)
    s = np.load(root / "cache/surface_S.npy", mmap_mode="r")
    m = np.load(root / "r3_profiles/mechanisms" / f"{method_id}.npy", mmap_mode="r")
    q = q_matrix(profiles)
    epsilon = float(registry["fusion"]["epsilon"])
    observed_surface = fuse_array(fusion, s, m, q, epsilon)
    task = load_task(resources)
    p_metrics = _evaluate(task, s, hpo_ids, mpo_ids)
    observed_metrics = _evaluate(task, observed_surface, hpo_ids, mpo_ids)
    observed_delta = observed_metrics["mrr"] - p_metrics["mrr"]
    _, component_unmasked = (None, None)
    component_unmasked, _ = text_kernels(resources, profiles, apply_direction=False)
    kernel = component_unmasked * direction_mask(profiles.h_atoms, profiles.m_atoms)
    seeds = json.loads(
        resolve(resources, f"{resources['module_root']}/configs/seeds.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    for seed in seeds["profile_permutations"]:
        random = np.random.default_rng(seed)
        h_values, h_q, movable_h = _permute_exact_size(profiles.h_profiles, profiles.h_q, random)
        m_values, m_q, movable_m = _permute_exact_size(profiles.m_profiles, profiles.m_q, random)
        null = WorkingProfiles(
            hpo_ids, mpo_ids, profiles.h_atoms, profiles.m_atoms, h_values, m_values, h_q, m_q
        )
        null_m = bma_surface(resources, null, kernel)
        null_q = q_matrix(null)
        metrics = _evaluate(task, fuse_array(fusion, s, null_m, null_q, epsilon), hpo_ids, mpo_ids)
        rows.append(
            {
                "control": "exact_size_profile_permutation",
                "seed": seed,
                "mrr": metrics["mrr"],
                "delta_vs_P0_mrr": metrics["mrr"] - p_metrics["mrr"],
                "movable_hpo_terms": movable_h,
                "movable_mpo_terms": movable_m,
                "requested_swaps": 0,
                "accepted_swaps": 0,
                "direction_states_changed": 0,
            }
        )
    for seed in seeds["degree_rewiring"]:
        random = np.random.default_rng(seed)
        h_values, h_requested, h_accepted = _rewire(profiles.h_profiles, random)
        m_values, m_requested, m_accepted = _rewire(profiles.m_profiles, random)
        null = WorkingProfiles(
            hpo_ids, mpo_ids, profiles.h_atoms, profiles.m_atoms, h_values, m_values, profiles.h_q, profiles.m_q
        )
        null_m = bma_surface(resources, null, kernel)
        metrics = _evaluate(task, fuse_array(fusion, s, null_m, q, epsilon), hpo_ids, mpo_ids)
        rows.append(
            {
                "control": "degree_preserving_rewire",
                "seed": seed,
                "mrr": metrics["mrr"],
                "delta_vs_P0_mrr": metrics["mrr"] - p_metrics["mrr"],
                "movable_hpo_terms": 0,
                "movable_mpo_terms": 0,
                "requested_swaps": h_requested + m_requested,
                "accepted_swaps": h_accepted + m_accepted,
                "direction_states_changed": 0,
            }
        )
    for seed in seeds["direction_permutations"]:
        random = np.random.default_rng(seed)
        h_atoms, h_changed = _direction_atoms(profiles.h_atoms, random)
        m_atoms, m_changed = _direction_atoms(profiles.m_atoms, random)
        null = WorkingProfiles(
            hpo_ids, mpo_ids, h_atoms, m_atoms, profiles.h_profiles, profiles.m_profiles, profiles.h_q, profiles.m_q
        )
        null_kernel = component_unmasked * direction_mask(h_atoms, m_atoms)
        null_m = bma_surface(resources, null, null_kernel)
        metrics = _evaluate(task, fuse_array(fusion, s, null_m, q, epsilon), hpo_ids, mpo_ids)
        rows.append(
            {
                "control": "direction_permutation",
                "seed": seed,
                "mrr": metrics["mrr"],
                "delta_vs_P0_mrr": metrics["mrr"] - p_metrics["mrr"],
                "movable_hpo_terms": 0,
                "movable_mpo_terms": 0,
                "requested_swaps": 0,
                "accepted_swaps": 0,
                "direction_states_changed": h_changed + m_changed,
            }
        )
    output = root / "r5_controls"
    fields = (
        "control",
        "seed",
        "mrr",
        "delta_vs_P0_mrr",
        "movable_hpo_terms",
        "movable_mpo_terms",
        "requested_swaps",
        "accepted_swaps",
        "direction_states_changed",
    )
    write_tsv(output / "permutation_controls.tsv", rows, fields)
    summary = _summary(rows, observed_delta)
    write_tsv(
        output / "permutation_summary.tsv",
        summary,
        (
            "control",
            "repeats",
            "null_mean_delta_mrr",
            "null_ci_low",
            "null_ci_high",
            "observed_delta_mrr",
            "null_at_least_observed",
            "empirical_p_upper",
        ),
    )
    candidate = sorted({value for model in task.models for value in model.mpo_ids})
    m_lookup = {value: index for index, value in enumerate(mpo_ids)}
    candidate_columns = [m_lookup[value] for value in candidate if value in m_lookup]
    density = _density_report(output, np.asarray(s), np.asarray(m), observed_surface, q, candidate_columns)
    source_rows = [
        {
            "analysis": "candidate_pool_source_gene_mask",
            "status": "source_gene_audit_unavailable",
            "reason": "HPO normalized assertions do not provide a complete source_gene_ids/retrieval_query derivation key.",
        },
        {
            "analysis": "two_channel_source_family_overlap",
            "status": "not_available",
            "reason": "The legacy Extended reranker and pair-level mission4 assertions do not share a complete source_family_id contract.",
        },
        {
            "analysis": "A_B_source_family_split",
            "status": "not_available",
            "reason": "A leakage-safe cross-channel source-family partition cannot be constructed from the available legacy fields.",
        },
        {
            "analysis": "source_bridge_LOO_v1",
            "status": "not_available",
            "reason": "Cross-channel derived copies cannot be removed consistently without a shared source-family graph.",
        },
    ]
    write_tsv(output / "source_mask_and_bridge_loo.tsv", source_rows, ("analysis", "status", "reason"))
    core = {
        "exact_size_profile_permutation",
        "degree_preserving_rewire",
    }
    exploratory = any(
        float(row["empirical_p_upper"]) > 0.05
        for row in summary
        if row["control"] in core
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "selected_surface": selected,
        "observed_delta_vs_P0_mrr": observed_delta,
        "permutation_repeats": len(rows),
        "summary": summary,
        "mechanism_claim_status": "exploratory" if exploratory else "supported_by_null_controls",
        "source_audit_status": "incomplete_source_contract",
        "outputs": {
            "controls": file_record(output / "permutation_controls.tsv"),
            "summary": file_record(output / "permutation_summary.tsv"),
            "density": file_record(output / "density_coverage_bias_report.json"),
            "source_audit": file_record(output / "source_mask_and_bridge_loo.tsv"),
        },
    }
    write_json(output / "controls_manifest.json", manifest)
    return manifest
