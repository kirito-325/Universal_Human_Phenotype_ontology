from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import evaluate_rankings
from .rerank import rerank
from .schemas import DiseaseTerm, GeneSetResources, HpoCpoEdge


def shuffle_gene_sets(
    resources: GeneSetResources,
    candidate_genes: Sequence[str],
    rng: np.random.Generator,
) -> GeneSetResources:
    genes = np.asarray(sorted({gene.upper() for gene in candidate_genes}), dtype=object)

    def shuffle(scores: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
        by_group: dict[str, list[float]] = defaultdict(list)
        for (group, _), weight in scores.items():
            by_group[group].append(float(weight))
        output: dict[tuple[str, str], float] = {}
        for group, weights in sorted(by_group.items()):
            if not weights or not len(genes):
                continue
            selected = rng.choice(genes, size=min(len(weights), len(genes)), replace=False)
            shuffled_weights = rng.permutation(np.asarray(weights, dtype=float))[: len(selected)]
            for gene, weight in zip(selected.tolist(), shuffled_weights.tolist()):
                output[(group, str(gene))] = float(weight)
        return output

    return GeneSetResources(
        marker_scores=shuffle(resources.marker_scores),
        function_scores=shuffle(resources.function_scores),
        cell_names=dict(resources.cell_names),
        function_names=dict(resources.function_names),
        marker_set_sizes=dict(resources.marker_set_sizes),
        function_set_sizes=dict(resources.function_set_sizes),
    )


def run_gene_set_null(
    base_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    hpo_cpo: Mapping[str, Sequence[HpoCpoEdge]],
    resources: GeneSetResources,
    config: Mapping[str, Any],
    gold_genes: Mapping[str, set[str]],
    observed_metrics: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config.get("null_models", {})
    iterations = int(settings.get("iterations", 500))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    rng = np.random.default_rng(seed)
    candidate_genes = sorted({str(row["gene_symbol"]).upper() for row in base_rows})
    iteration_rows: list[dict[str, Any]] = []
    metric_names = ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "mrr", "ndcg_at_10")
    for iteration in range(1, iterations + 1):
        shuffled = shuffle_gene_sets(resources, candidate_genes, rng)
        result = rerank(
            base_rows,
            evidence_rows,
            diseases,
            hpo_cpo,
            shuffled,
            config,
            experiment="null_gene_sets",
            emit_details=False,
        )
        _, metrics = evaluate_rankings(
            result["final_rankings"], gold_genes, "final_rank", f"null_{iteration}"
        )
        for metric in metric_names:
            iteration_rows.append(
                {"iteration": iteration, "null_model": "gene_set_membership", "metric": metric, "value": metrics[metric]}
            )
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        values = np.asarray(
            [float(row["value"]) for row in iteration_rows if row["metric"] == metric], dtype=float
        )
        observed = float(observed_metrics.get(metric, 0.0))
        summary_rows.append(
            {
                "null_model": "gene_set_membership",
                "metric": metric,
                "observed": observed,
                "null_mean": float(values.mean()) if len(values) else 0.0,
                "null_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "null_q025": float(np.quantile(values, 0.025)) if len(values) else 0.0,
                "null_q975": float(np.quantile(values, 0.975)) if len(values) else 0.0,
                "permutation_p_value": (
                    (1 + int(np.sum(values >= observed))) / (len(values) + 1) if len(values) else 1.0
                ),
                "iterations": len(values),
            }
        )
    return iteration_rows, summary_rows


NULL_ITERATION_FIELDS = ["iteration", "null_model", "metric", "value"]

NULL_SUMMARY_FIELDS = [
    "null_model",
    "metric",
    "observed",
    "null_mean",
    "null_sd",
    "null_q025",
    "null_q975",
    "permutation_p_value",
    "iterations",
]
