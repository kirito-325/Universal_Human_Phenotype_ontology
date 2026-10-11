from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cpo_mapping.io import read_tsv, write_json, write_text

from .config import output_root
from .mapping import MappingContext, _direction_code
from .term_matrix import MATRIX_DATASETS


def _assert_close(actual: float, expected: float, label: str, atol: float = 2e-6) -> None:
    if np.isnan(expected):
        if not np.isnan(actual):
            raise AssertionError(f"{label}: expected NA, observed {actual}")
        return
    if not np.isclose(actual, expected, atol=atol, rtol=atol):
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def _profile_identity_checks(root: Path) -> dict[str, int]:
    memberships = pd.read_parquet(root / "profiles/structure_membership.parquet")
    if memberships.duplicated(
        ["ontology_side", "term_id", "profile_view", "structure_id"]
    ).any():
        raise AssertionError("duplicate structure membership")
    _, policies = read_tsv(root / "profiles/node_policy.tsv")
    grouped = (
        memberships.groupby(["ontology_side", "term_id", "profile_view"])["structure_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    checked = 0
    for row in policies:
        key = (row["ontology_side"], row["term_id"])
        self_set = grouped.get((*key, "self_only"), set())
        inherited_set = grouped.get((*key, "inherited_only"), set())
        final_set = grouped.get((*key, "final"), set())
        strategy = row["final_strategy"]
        expected = (
            self_set
            if strategy == "self"
            else inherited_set
            if strategy == "inherited"
            else self_set | inherited_set
        )
        if final_set != expected:
            raise AssertionError(
                f"profile identity mismatch for {row['ontology_side']} {row['term_id']}"
            )
        checked += 1
    return {"profile_identity_terms_checked": checked}


def _selected_indices(profiles: list[list[dict[str, object]]], policies: list[dict[str, str]], count: int, seed: int) -> list[int]:
    active = [index for index, profile in enumerate(profiles) if profile]
    roots = [
        index for index in active if policies[index]["node_type"] == "ROOT_LAYER"
    ][:1]
    largest = sorted(active, key=lambda index: (-len(profiles[index]), index))[:2]
    remaining = [index for index in active if index not in set(roots + largest)]
    rng = np.random.default_rng(seed)
    sampled = (
        rng.choice(remaining, size=min(max(0, count - len(roots + largest)), len(remaining)), replace=False).tolist()
        if remaining
        else []
    )
    return sorted(dict.fromkeys([*roots, *largest, *map(int, sampled)]))


def _term_matrix_checks(config: dict[str, Any]) -> dict[str, int]:
    context = MappingContext(config)
    root = context.root
    path = root / "matrices/hpo_mpo_cpo.h5"
    h_selected = _selected_indices(context.h_profiles, context.hpo_policy, 20, 20260905)
    m_selected = _selected_indices(context.m_profiles, context.mpo_policy, 40, 20260906)
    checked = 0
    roots_checked = 0
    with h5py.File(path, "r") as handle:
        missing = [dataset for dataset in MATRIX_DATASETS if dataset not in handle]
        if missing:
            raise AssertionError(f"missing matrix datasets: {missing}")
        expected_shape = (len(context.hpo_ids), len(context.mpo_ids))
        for dataset in MATRIX_DATASETS:
            if handle[dataset].shape != expected_shape:
                raise AssertionError(f"wrong shape for {dataset}: {handle[dataset].shape}")
        for h_index in h_selected:
            h_profile = context.h_profiles[h_index]
            h_components = np.asarray(
                [context.h_structure_index[str(row["structure_id"])] for row in h_profile]
            )
            for m_index in m_selected:
                m_profile = context.m_profiles[m_index]
                m_components = np.asarray(
                    [context.m_structure_index[str(row["structure_id"])] for row in m_profile]
                )
                pair = context.similarity[np.ix_(h_components, m_components)]
                forward = float(np.max(pair, axis=1).mean())
                reverse = float(np.max(pair, axis=0).mean())
                shape = (forward + reverse) / 2
                _assert_close(float(handle["final/hpo_to_mpo"][h_index, m_index]), forward, "term F")
                _assert_close(float(handle["final/mpo_to_hpo"][h_index, m_index]), reverse, "term R")
                _assert_close(float(handle["final/shape"][h_index, m_index]), shape, "term BMA")
                h_by_structure = {
                    str(value["structure_id"]): value for value in h_profile
                }
                m_by_structure = {
                    str(value["structure_id"]): value for value in m_profile
                }
                direction_codes = [
                    _direction_code(h_by_structure[structure], m_by_structure[structure])
                    for structure in h_by_structure.keys() & m_by_structure.keys()
                ]
                agree = direction_codes.count(1)
                conflict = direction_codes.count(-1)
                if agree + conflict:
                    direction_weight = 0.5 * (1 / len(h_profile) + 1 / len(m_profile))
                    expected_support = agree * direction_weight
                    expected_conflict = conflict * direction_weight
                else:
                    expected_support = float("nan")
                    expected_conflict = float("nan")
                _assert_close(
                    float(handle["final/direction_support"][h_index, m_index]),
                    expected_support,
                    "term direction support",
                )
                _assert_close(
                    float(handle["final/direction_conflict"][h_index, m_index]),
                    expected_conflict,
                    "term direction conflict",
                )
                checked += 1
                if (
                    context.hpo_policy[h_index]["node_type"] == "ROOT_LAYER"
                    or context.mpo_policy[m_index]["node_type"] == "ROOT_LAYER"
                ):
                    roots_checked += 1
        for dataset in MATRIX_DATASETS:
            values = handle[dataset][h_selected, :][:, m_selected]
            finite = values[np.isfinite(values)]
            if len(finite) and (float(finite.min()) < -2e-6 or float(finite.max()) > 1 + 2e-6):
                raise AssertionError(f"out-of-range values in {dataset}")
    return {
        "term_pairs_recomputed": checked,
        "root_layer_pairs_recomputed": roots_checked,
        "matrix_dataset_count": len(MATRIX_DATASETS),
    }


def _aggregate_fixture() -> dict[str, float]:
    values = np.asarray([[1.0], [np.nan]])
    h_weights = np.asarray([1.0, 1.0])
    m_weights = np.asarray([1.0])
    h_computable = np.any(np.isfinite(values), axis=1)
    m_computable = np.any(np.isfinite(values), axis=0)
    a = np.nanmax(values[h_computable], axis=1)
    b = np.nanmax(values[:, m_computable], axis=0)
    score_a = float(np.average(a, weights=h_weights[h_computable]))
    score_b = float(np.average(b, weights=m_weights[m_computable]))
    q_h = float(h_weights[h_computable].sum() / h_weights.sum())
    q_m = float(m_weights[m_computable].sum() / m_weights.sum())
    observed = (score_a + score_b) / 2
    evidence = (q_h * score_a + q_m * score_b) / 2
    _assert_close(observed, 1.0, "disease fixture observed")
    _assert_close(evidence, 0.75, "disease fixture evidence")
    return {"fixture_score_observed": observed, "fixture_score_evidence": evidence}


def _disease_model_checks(root: Path) -> dict[str, int]:
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    h_index = {row["term_id"]: int(row["index"]) for row in hpo_rows}
    m_index = {row["term_id"]: int(row["index"]) for row in mpo_rows}
    disease_profiles = pd.read_parquet(root / "disease_model/disease_profiles.parquet")
    model_profiles = pd.read_parquet(root / "disease_model/model_profiles.parquet")
    disease_grouped = {
        key: list(zip(frame["term_id"].astype(str), frame["weight"].astype(float)))
        for key, frame in disease_profiles.groupby("profile_id")
    }
    model_grouped = {
        key: list(zip(frame["term_id"].astype(str), frame["weight"].astype(float)))
        for key, frame in model_profiles.groupby("profile_id")
    }
    rng = np.random.default_rng(20260907)
    pairs = list(
        zip(
            rng.choice(len(disease_rows), size=20, replace=False),
            rng.choice(len(model_rows), size=20, replace=False),
        )
    )
    checked = 0
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as term_handle, h5py.File(
        root / "disease_model/scores/disease_model_scores.h5", "r"
    ) as score_handle:
        for disease_index, model_index in pairs:
            disease = disease_rows[int(disease_index)]["disease_id"]
            model = model_rows[int(model_index)]["model_profile_id"]
            h_values = sorted(disease_grouped[disease], key=lambda value: h_index[value[0]])
            m_values = sorted(model_grouped[model], key=lambda value: m_index[value[0]])
            h_terms = [h_index[term] for term, _ in h_values]
            m_terms = [m_index[term] for term, _ in m_values]
            pair_scores = np.asarray(term_handle["final/shape"][h_terms, :][:, m_terms])
            h_weights = np.asarray([weight for _, weight in h_values])
            m_weights = np.asarray([weight for _, weight in m_values])
            h_computable = np.any(np.isfinite(pair_scores), axis=1)
            m_computable = np.any(np.isfinite(pair_scores), axis=0)
            a = (
                float(np.average(np.nanmax(pair_scores[h_computable], axis=1), weights=h_weights[h_computable]))
                if h_computable.any()
                else float("nan")
            )
            b = (
                float(np.average(np.nanmax(pair_scores[:, m_computable], axis=0), weights=m_weights[m_computable]))
                if m_computable.any()
                else float("nan")
            )
            q_h = float(h_weights[h_computable].sum() / h_weights.sum())
            q_m = float(m_weights[m_computable].sum() / m_weights.sum())
            positive_h = np.any(np.isfinite(pair_scores) & (pair_scores > 0), axis=1)
            positive_m = np.any(np.isfinite(pair_scores) & (pair_scores > 0), axis=0)
            pos_h = float(h_weights[positive_h].sum() / h_weights.sum())
            pos_m = float(m_weights[positive_m].sum() / m_weights.sum())
            observed = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else float("nan")
            evidence = (q_h * a + q_m * b) / 2 if np.isfinite(observed) else float("nan")
            for dataset, expected in (
                ("A", a), ("B", b), ("q_H", q_h), ("q_M", q_m),
                ("positive_match_coverage_H", pos_h),
                ("positive_match_coverage_M", pos_m),
                ("score_observed", observed), ("score_evidence", evidence),
            ):
                _assert_close(
                    float(score_handle[dataset][int(disease_index), int(model_index)]),
                    expected,
                    f"disease-model {dataset}",
                    atol=5e-6,
                )
            checked += 1
    return {"disease_model_pairs_recomputed": checked}


def _gene_checks(root: Path) -> dict[str, int]:
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, mouse_rows = read_tsv(root / "gene_prediction/mouse_gene_index.tsv")
    models_by_gene: dict[str, list[int]] = defaultdict(list)
    joint_model_ids = set()
    for index, row in enumerate(model_rows):
        if row["model_status"] == "single_gene_model":
            models_by_gene[row["perturbed_mouse_gene_ids"]].append(index)
        else:
            joint_model_ids.add(row["model_profile_id"])
    candidates = pd.read_parquet(
        root / "gene_prediction/mouse_gene_scores.parquet", columns=["best_model_id"]
    )
    if set(candidates["best_model_id"].astype(str)) & joint_model_ids:
        raise AssertionError("joint genotype was attributed as a single-gene best model")
    human_rows_checked = 0
    for filename in ("human_1to1.parquet", "human_expanded.parquet"):
        human = pd.read_parquet(
            root / f"gene_prediction/{filename}",
            columns=["best_model_id", "rank_min", "rank_max", "tie_count"],
        )
        if (human["best_model_id"] == "SEE_MOUSE_GENE_EVIDENCE").any():
            raise AssertionError(f"unresolved best-model placeholder in {filename}")
        if (human["rank_min"] > human["rank_max"]).any():
            raise AssertionError(f"invalid rank interval in {filename}")
        if (human["tie_count"] != human["rank_max"] - human["rank_min"] + 1).any():
            raise AssertionError(f"invalid tie count in {filename}")
        human_rows_checked += len(human)
    rng = np.random.default_rng(20260908)
    disease_indices = rng.choice(len(disease_rows), size=10, replace=False)
    gene_indices = rng.choice(len(mouse_rows), size=10, replace=False)
    checked = 0
    with h5py.File(root / "disease_model/scores/disease_model_scores.h5", "r") as model_handle, h5py.File(
        root / "gene_prediction/mouse_gene_scores.h5", "r"
    ) as gene_handle:
        for disease_index, gene_index in zip(disease_indices, gene_indices):
            gene = mouse_rows[int(gene_index)]["mouse_gene_id"]
            model_values = np.asarray(
                model_handle["score_evidence"][int(disease_index), models_by_gene[gene]]
            )
            expected = (
                float(np.nanmax(model_values))
                if np.isfinite(model_values).any()
                else float("nan")
            )
            actual = float(gene_handle["score"][int(disease_index), int(gene_index)])
            _assert_close(actual, expected, "mouse gene max-model aggregation", atol=5e-6)
            checked += 1
    return {
        "mouse_gene_aggregations_recomputed": checked,
        "joint_models_checked_not_attributed": len(joint_model_ids),
        "human_candidate_rows_checked": human_rows_checked,
    }


def run_validate(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    required = [
        root / "profiles/node_policy.tsv",
        root / "matrices/hpo_mpo_cpo.h5",
        root / "mapping/hpo_to_mpo.parquet",
        root / "mapping/mpo_to_hpo.parquet",
        root / "disease_model/scores/disease_model_scores.h5",
        root / "gene_prediction/human_1to1.parquet",
        root / "gene_prediction/human_expanded.parquet",
        root / "evaluation/ranking_metrics.tsv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation artifacts: {missing}")
    checks: dict[str, object] = {}
    checks.update(_profile_identity_checks(root))
    checks.update(_term_matrix_checks(config))
    checks.update(_aggregate_fixture())
    checks.update(_disease_model_checks(root))
    checks.update(_gene_checks(root))
    result = {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "required_artifact_count": len(required),
        "checks": checks,
    }
    validation_root = root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    write_json(validation_root / "validation_report.json", result)
    lines = [
        "# Validation report",
        "",
        f"Status: **{result['status']}**",
        "",
        "All profile-policy identities, matrix schemas, numerical fixtures, sampled term BMA values, sampled disease-model scores, and sampled max-model gene aggregations passed.",
        "",
        "| Check | Value |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in checks.items()],
        "",
    ]
    write_text(validation_root / "validation_report.md", "\n".join(lines))
    return result
