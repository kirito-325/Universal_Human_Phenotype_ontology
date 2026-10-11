from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from .io import as_bool, read_tsv, value


def load_gold_genes(path: str) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(path):
        disease_id = value(row, "disease_id")
        gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
        if not disease_id or not gene:
            continue
        evaluation_flag = value(row, "is_evaluation_positive")
        if evaluation_flag and not as_bool(evaluation_flag):
            continue
        output[disease_id].add(gene)
    return dict(output)


def evaluate_rankings(
    ranking_rows: Sequence[Mapping[str, Any]],
    gold_genes: Mapping[str, set[str]],
    rank_key: str,
    experiment: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_disease: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        by_disease[str(row["disease_id"])].append(row)
    disease_rows: list[dict[str, Any]] = []
    for disease_id in sorted(gold_genes):
        candidates = by_disease.get(disease_id, [])
        ranks_by_gene = {
            str(row["gene_symbol"]).upper(): int(row[rank_key])
            for row in candidates
        }
        positives = sorted(gold_genes[disease_id])
        ranks = sorted(ranks_by_gene[gene] for gene in positives if gene in ranks_by_gene)
        candidate_count = len(candidates)
        best_rank = ranks[0] if ranks else candidate_count + 1
        mean_known_rank = mean(ranks) if ranks else candidate_count + 1
        reciprocal = 1.0 / best_rank if ranks else 0.0
        dcg = sum(1.0 / math.log2(rank + 1.0) for rank in ranks if rank <= 10)
        ideal_count = min(10, len(positives), candidate_count)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
        percentile = (
            (best_rank - 1) / (candidate_count - 1)
            if ranks and candidate_count > 1
            else 0.0 if ranks else 1.0
        )
        disease_rows.append(
            {
                "experiment": experiment,
                "disease_id": disease_id,
                "candidate_gene_count": candidate_count,
                "known_gene_count": len(positives),
                "known_genes": positives,
                "retrieved_known_genes": len(ranks),
                "best_known_gene_rank": best_rank,
                "mean_known_gene_rank": mean_known_rank,
                "best_rank_percentile": percentile,
                "reciprocal_rank": reciprocal,
                "ndcg_at_10": dcg / idcg if idcg else 0.0,
                "recall_at_1": bool(ranks and best_rank <= 1),
                "recall_at_5": bool(ranks and best_rank <= 5),
                "recall_at_10": bool(ranks and best_rank <= 10),
                "recall_at_50": bool(ranks and best_rank <= 50),
                "all_known_recalled": len(ranks) == len(positives),
            }
        )
    metrics = summarize_disease_metrics(disease_rows)
    return disease_rows, metrics


def evaluate_open_rankings(
    ranking_rows: Sequence[Mapping[str, Any]],
    gold_genes: Mapping[str, set[str]],
    rank_key: str,
    experiment: str,
    candidate_universe_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate positive-score rankings while counting absent gold genes as failures."""
    if candidate_universe_size <= 0:
        raise ValueError("candidate_universe_size must be positive")
    by_disease: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        by_disease[str(row["disease_id"])].append(row)
    disease_rows: list[dict[str, Any]] = []
    for disease_id in sorted(gold_genes):
        candidates = by_disease.get(disease_id, [])
        ranks_by_gene = {
            str(row["gene_symbol"]).upper(): int(row[rank_key]) for row in candidates
        }
        positives = sorted(gold_genes[disease_id])
        ranks = sorted(ranks_by_gene[gene] for gene in positives if gene in ranks_by_gene)
        best_rank = ranks[0] if ranks else candidate_universe_size + 1
        mean_known_rank = mean(ranks) if ranks else candidate_universe_size + 1
        reciprocal = 1.0 / best_rank if ranks else 0.0
        dcg = sum(1.0 / math.log2(rank + 1.0) for rank in ranks if rank <= 10)
        ideal_count = min(10, len(positives), candidate_universe_size)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
        percentile = (
            (best_rank - 1) / (candidate_universe_size - 1)
            if ranks and candidate_universe_size > 1
            else 0.0 if ranks else 1.0
        )
        disease_rows.append(
            {
                "experiment": experiment,
                "disease_id": disease_id,
                "candidate_gene_count": candidate_universe_size,
                "known_gene_count": len(positives),
                "known_genes": positives,
                "retrieved_known_genes": len(ranks),
                "best_known_gene_rank": best_rank,
                "mean_known_gene_rank": mean_known_rank,
                "best_rank_percentile": percentile,
                "reciprocal_rank": reciprocal,
                "ndcg_at_10": dcg / idcg if idcg else 0.0,
                "recall_at_1": bool(ranks and best_rank <= 1),
                "recall_at_5": bool(ranks and best_rank <= 5),
                "recall_at_10": bool(ranks and best_rank <= 10),
                "recall_at_50": bool(ranks and best_rank <= 50),
                "all_known_recalled": len(ranks) == len(positives),
            }
        )
    metrics = summarize_disease_metrics(disease_rows)
    metrics["gold_gene_retrieval_rate"] = mean(
        int(row["retrieved_known_genes"]) > 0 for row in disease_rows
    ) if disease_rows else 0.0
    return disease_rows, metrics


def summarize_disease_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "evaluated_diseases": 0.0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_50": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "median_rank": 0.0,
            "median_rank_percentile": 0.0,
        }
    return {
        "evaluated_diseases": float(len(rows)),
        "recall_at_1": mean(bool(row["recall_at_1"]) for row in rows),
        "recall_at_5": mean(bool(row["recall_at_5"]) for row in rows),
        "recall_at_10": mean(bool(row["recall_at_10"]) for row in rows),
        "recall_at_50": mean(bool(row["recall_at_50"]) for row in rows),
        "mrr": mean(float(row["reciprocal_rank"]) for row in rows),
        "ndcg_at_10": mean(float(row["ndcg_at_10"]) for row in rows),
        "median_rank": float(median(float(row["best_known_gene_rank"]) for row in rows)),
        "median_rank_percentile": float(median(float(row["best_rank_percentile"]) for row in rows)),
    }


def metrics_as_rows(experiment: str, metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    n = int(metrics.get("evaluated_diseases", 0))
    return [
        {"experiment": experiment, "metric": metric, "value": score, "evaluated_diseases": n}
        for metric, score in metrics.items()
        if metric != "evaluated_diseases"
    ]


def compare_disease_ranks(
    baseline_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    experiment: str,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = {str(row["disease_id"]): row for row in baseline_rows}
    final = {str(row["disease_id"]): row for row in final_rows}
    comparisons: list[dict[str, Any]] = []
    for disease_id in sorted(set(baseline).intersection(final)):
        base_rank = float(baseline[disease_id]["best_known_gene_rank"])
        reranked = float(final[disease_id]["best_known_gene_rank"])
        change = base_rank - reranked
        comparisons.append(
            {
                "experiment": experiment,
                "disease_id": disease_id,
                "base_best_known_rank": base_rank,
                "final_best_known_rank": reranked,
                "rank_change": change,
                "outcome": "win" if change > 0 else "loss" if change < 0 else "tie",
            }
        )
    changes = np.asarray([row["rank_change"] for row in comparisons], dtype=float)
    if len(changes) and np.any(changes != 0):
        test = wilcoxon(changes, alternative="two-sided", zero_method="wilcox")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0
    rng = np.random.default_rng(seed)
    if len(changes) and bootstrap_iterations > 0:
        samples = rng.choice(changes, size=(bootstrap_iterations, len(changes)), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    else:
        ci_low = ci_high = 0.0
    outcomes = Counter(row["outcome"] for row in comparisons)
    summary = {
        "experiment": experiment,
        "paired_diseases": len(comparisons),
        "mean_rank_change": float(changes.mean()) if len(changes) else 0.0,
        "median_rank_change": float(np.median(changes)) if len(changes) else 0.0,
        "mean_rank_change_ci_low": float(ci_low),
        "mean_rank_change_ci_high": float(ci_high),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "wins": outcomes.get("win", 0),
        "ties": outcomes.get("tie", 0),
        "losses": outcomes.get("loss", 0),
        "improved_fraction": outcomes.get("win", 0) / len(comparisons) if comparisons else 0.0,
    }
    return comparisons, summary


DISEASE_METRIC_FIELDS = [
    "experiment",
    "disease_id",
    "candidate_gene_count",
    "known_gene_count",
    "known_genes",
    "retrieved_known_genes",
    "best_known_gene_rank",
    "mean_known_gene_rank",
    "best_rank_percentile",
    "reciprocal_rank",
    "ndcg_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "all_known_recalled",
]

BENCHMARK_METRIC_FIELDS = ["experiment", "metric", "value", "evaluated_diseases"]

RANK_CHANGE_FIELDS = [
    "experiment",
    "disease_id",
    "base_best_known_rank",
    "final_best_known_rank",
    "rank_change",
    "outcome",
]

STATISTICAL_TEST_FIELDS = [
    "experiment",
    "paired_diseases",
    "mean_rank_change",
    "median_rank_change",
    "mean_rank_change_ci_low",
    "mean_rank_change_ci_high",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "wins",
    "ties",
    "losses",
    "improved_fraction",
]
