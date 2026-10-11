from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Iterable, Mapping, Sequence


def evaluate_rankings(
    rankings: Sequence[Mapping[str, object]], gold: Mapping[str, set[str]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rankings:
        grouped[str(row["disease_id"])].append(row)
    cases: list[dict[str, object]] = []
    for disease_id in sorted(gold):
        rows = sorted(grouped[disease_id], key=lambda row: int(row["rank"]))
        positions = [int(row["rank"]) for row in rows if str(row["gene_symbol"]) in gold[disease_id]]
        best = min(positions) if positions else len(rows) + 1
        cases.append(
            {
                "disease_id": disease_id,
                "best_known_gene_rank": best,
                "reciprocal_rank": 1.0 / best if positions else 0.0,
                "ndcg_at_10": 1.0 / math.log2(best + 1) if best <= 10 else 0.0,
                "recall_at_1": float(best <= 1),
                "recall_at_5": float(best <= 5),
                "recall_at_10": float(best <= 10),
            }
        )
    count = len(cases)
    metrics = {
        "evaluated_diseases": float(count),
        "recall_at_1": sum(float(row["recall_at_1"]) for row in cases) / count,
        "recall_at_5": sum(float(row["recall_at_5"]) for row in cases) / count,
        "recall_at_10": sum(float(row["recall_at_10"]) for row in cases) / count,
        "mrr": sum(float(row["reciprocal_rank"]) for row in cases) / count,
        "ndcg_at_10": sum(float(row["ndcg_at_10"]) for row in cases) / count,
        "median_rank": float(median(int(row["best_known_gene_rank"]) for row in cases)),
    }
    return cases, metrics


def metric_rows(method_id: str, metrics: Mapping[str, float]) -> Iterable[dict[str, object]]:
    for key, value in metrics.items():
        yield {"method_id": method_id, "metric": key, "value": value}

