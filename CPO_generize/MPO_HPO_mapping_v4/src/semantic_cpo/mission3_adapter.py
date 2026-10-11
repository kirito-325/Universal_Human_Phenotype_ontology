from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve, result_root
from .evaluation import evaluate_rankings
from .io import read_tsv, write_json, write_tsv


@dataclass(frozen=True)
class Model:
    model_id: str
    model_name: str
    gene_symbol: str
    mpo_ids: tuple[str, ...]


@dataclass
class Task:
    diseases: dict[str, tuple[str, ...]]
    disease_names: dict[str, str]
    models: tuple[Model, ...]
    gold: dict[str, set[str]]


def load_task(resources: dict[str, Any]) -> Task:
    reduced = resolve(
        resources,
        "MPO_HPO_mapping_v4/results/r0_baseline_reproduction/exomiser_2602/hpo/nonredundant_disease_hpo.tsv",
    )
    disease_rows = read_tsv(reduced)
    diseases: dict[str, list[str]] = defaultdict(list)
    disease_names: dict[str, str] = {}
    for row in disease_rows:
        diseases[row["disease_id"]].append(row["hpo_id"])
        disease_names[row["disease_id"]] = row["disease_name"]
    model_rows = read_tsv(resolve(resources, resources["benchmark"]["models"]))
    eligible = {
        row["model_id"]: row
        for row in model_rows
        if row.get("human_gene_symbol") and ";" not in row["human_gene_symbol"]
    }
    phenotype_rows = read_tsv(
        resolve(resources, resources["benchmark"]["model_phenotypes"])
    )
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in phenotype_rows:
        if row["model_id"] in eligible and row.get("status") == "abnormal":
            by_model[row["model_id"]].add(row["mp_id"])
    models = tuple(
        Model(
            model_id=model_id,
            model_name=row["model_name"],
            gene_symbol=row["human_gene_symbol"],
            mpo_ids=tuple(sorted(by_model[model_id])),
        )
        for model_id, row in sorted(eligible.items())
        if by_model[model_id]
    )
    gold: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(resolve(resources, resources["benchmark"]["gold_models"])):
        gold[row["disease_id"]].add(row["gene_symbol"])
    return Task(
        diseases={key: tuple(dict.fromkeys(value)) for key, value in diseases.items()},
        disease_names=disease_names,
        models=models,
        gold=dict(gold),
    )


def score_surface(
    task: Task,
    surface: np.ndarray,
    hpo_ids: list[str],
    mpo_ids: list[str],
) -> list[dict[str, object]]:
    h_index = {value: index for index, value in enumerate(hpo_ids)}
    m_index = {value: index for index, value in enumerate(mpo_ids)}
    ideal = np.nanmax(surface, axis=1)
    rankings: list[dict[str, object]] = []
    for disease_id, terms in sorted(task.diseases.items()):
        h_positions = [h_index[value] for value in terms if value in h_index]
        by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
        for model in task.models:
            m_positions = [m_index[value] for value in model.mpo_ids if value in m_index]
            if not h_positions or not m_positions:
                score = 0.0
                best_hpo = ""
                best_mpo = ""
            else:
                pair = np.asarray(surface[np.ix_(h_positions, m_positions)], dtype=np.float64)
                pair = np.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)
                matched = np.any(pair > 0, axis=0)
                if not np.any(matched):
                    score = 0.0
                    best_hpo = ""
                    best_mpo = ""
                else:
                    pair = pair[:, matched]
                    selected_m = [m_positions[i] for i, value in enumerate(matched) if value]
                    forward = pair.max(axis=1)
                    reverse = pair.max(axis=0)
                    raw_avg = (float(forward.sum()) + float(reverse.sum())) / (
                        len(forward) + len(reverse)
                    )
                    raw_max = float(pair.max())
                    ideal_scores = ideal[h_positions]
                    ideal_avg = float(ideal_scores.mean())
                    ideal_max = float(ideal_scores.max())
                    avg_percentage = 100.0 * raw_avg / ideal_avg if ideal_avg else 0.0
                    max_percentage = 100.0 * raw_max / ideal_max if ideal_max else 0.0
                    score = min(100.0, (avg_percentage + max_percentage) / 2.0)
                    best_flat = int(np.argmax(pair))
                    best_row, best_column = np.unravel_index(best_flat, pair.shape)
                    best_hpo = hpo_ids[h_positions[best_row]]
                    best_mpo = mpo_ids[selected_m[best_column]]
            by_gene[model.gene_symbol].append(
                {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "score": score,
                    "best_hpo_id": best_hpo,
                    "best_mpo_id": best_mpo,
                }
            )
        gene_rows: list[dict[str, object]] = []
        for gene, rows in sorted(by_gene.items()):
            best = min(rows, key=lambda row: (-float(row["score"]), str(row["model_id"])))
            gene_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": task.disease_names[disease_id],
                    "gene_symbol": gene,
                    "score": best["score"],
                    "best_model_id": best["model_id"],
                    "best_hpo_id": best["best_hpo_id"],
                    "best_mpo_id": best["best_mpo_id"],
                    "candidate_model_count": len(rows),
                }
            )
        gene_rows.sort(key=lambda row: (-float(row["score"]), str(row["gene_symbol"])))
        for rank, row in enumerate(gene_rows, start=1):
            row["rank"] = rank
        rankings.extend(gene_rows)
    return rankings


RANKING_FIELDS = (
    "disease_id",
    "disease_name",
    "gene_symbol",
    "score",
    "rank",
    "best_model_id",
    "best_hpo_id",
    "best_mpo_id",
    "candidate_model_count",
)


def evaluate_surface(
    resources: dict[str, Any],
    method_id: str,
    surface: np.ndarray,
    hpo_ids: list[str],
    mpo_ids: list[str],
    output: Path,
) -> dict[str, object]:
    task = load_task(resources)
    rankings = score_surface(task, surface, hpo_ids, mpo_ids)
    cases, metrics = evaluate_rankings(rankings, task.gold)
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "gene_rankings.tsv", rankings, RANKING_FIELDS)
    write_tsv(
        output / "disease_metrics.tsv",
        cases,
        (
            "disease_id",
            "best_known_gene_rank",
            "reciprocal_rank",
            "ndcg_at_10",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
        ),
    )
    result = {"method_id": method_id, "metrics": metrics}
    write_json(output / "metrics.json", result)
    return result


def task_hpo_ids(resources: dict[str, Any]) -> list[str]:
    task = load_task(resources)
    return sorted({term for terms in task.diseases.values() for term in terms})

