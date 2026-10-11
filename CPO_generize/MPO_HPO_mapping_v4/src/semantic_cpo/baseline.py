from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve, result_root
from .io import file_record, read_tsv, write_json, write_tsv


ARMS = (
    "A0_surface",
    "A3_strict_constrained",
    "A4_extended_constrained",
    "A5_bridge_loo",
)
EXPECTED = {
    "A0_surface": {"recall_at_1": 0.42, "recall_at_5": 0.76, "recall_at_10": 0.92, "mrr": 0.5509484127, "ndcg_at_10": 0.6348819579},
    "A3_strict_constrained": {"recall_at_1": 0.48, "recall_at_5": 0.74, "recall_at_10": 0.92, "mrr": 0.5942896825, "ndcg_at_10": 0.6675399593},
    "A4_extended_constrained": {"recall_at_1": 0.56, "recall_at_5": 0.88, "recall_at_10": 0.92, "mrr": 0.6908356375, "ndcg_at_10": 0.7439892971},
    "A5_bridge_loo": {"recall_at_1": 0.46, "recall_at_5": 0.72, "recall_at_10": 0.92, "mrr": 0.5755119048, "ndcg_at_10": 0.6526840842},
}


def _metrics(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in read_tsv(path):
        result.setdefault(row["experiment"], {})[row["metric"]] = float(row["value"])
    return result


def _compare_tsv(left: Path, right: Path, keys: tuple[str, ...], fields: tuple[str, ...]) -> dict[str, object]:
    left_rows = {tuple(row[key] for key in keys): row for row in read_tsv(left)}
    right_rows = {tuple(row[key] for key in keys): row for row in read_tsv(right)}
    differing = 0
    maximum = 0.0
    for key in set(left_rows) & set(right_rows):
        for field in fields:
            a = float(left_rows[key][field])
            b = float(right_rows[key][field])
            delta = abs(a - b)
            maximum = max(maximum, delta)
            differing += int(delta > 1e-10)
    return {
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "shared_rows": len(set(left_rows) & set(right_rows)),
        "left_only": len(set(left_rows) - set(right_rows)),
        "right_only": len(set(right_rows) - set(left_rows)),
        "differing_values": differing,
        "maximum_absolute_difference": maximum,
    }


def reproduce_baselines(resources: dict[str, Any]) -> dict[str, object]:
    original = resolve(resources, resources["mission3"]["baseline_output_2602"])
    rerun = resolve(
        resources,
        "MPO_HPO_mapping_v4/results/r0_baseline_reproduction/exomiser_2602",
    )
    if not rerun.is_dir():
        raise FileNotFoundError(
            "run the original CellPhenoRank command into results/r0_baseline_reproduction first"
        )
    original_metrics = _metrics(original / "benchmark/benchmark_metrics.tsv")
    rerun_metrics = _metrics(rerun / "benchmark/benchmark_metrics.tsv")
    metric_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for arm in ARMS:
        for metric, expected in EXPECTED[arm].items():
            observed = rerun_metrics[arm][metric]
            source_value = original_metrics[arm][metric]
            delta_expected = observed - expected
            delta_source = observed - source_value
            if abs(delta_expected) > 1e-9 or abs(delta_source) > 1e-10:
                failures.append(f"{arm}:{metric}")
            metric_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "expected": expected,
                    "source_value": source_value,
                    "reproduced_value": observed,
                    "delta_expected": delta_expected,
                    "delta_source": delta_source,
                }
            )
    file_comparisons = {
        "base_gene_rankings": _compare_tsv(
            original / "baseline/base_gene_rankings.tsv",
            rerun / "baseline/base_gene_rankings.tsv",
            ("disease_id", "gene_symbol"),
            ("base_score", "base_rank"),
        ),
        "strict_rankings": _compare_tsv(
            original / "reranking/final_constrained_rankings.tsv",
            rerun / "reranking/final_constrained_rankings.tsv",
            ("disease_id", "gene_symbol"),
            ("base_rank", "final_rank"),
        ),
        "extended_rankings": _compare_tsv(
            original / "reranking/extended_constrained_rankings.tsv",
            rerun / "reranking/extended_constrained_rankings.tsv",
            ("disease_id", "gene_symbol"),
            ("base_rank", "final_rank"),
        ),
    }
    for name, comparison in file_comparisons.items():
        if any(comparison[key] for key in ("left_only", "right_only", "differing_values")):
            failures.append(name)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source_output": str(original),
        "reproduction_output": str(rerun),
        "source_manifest": file_record(original / "run_manifest.json"),
        "reproduction_manifest": file_record(rerun / "run_manifest.json"),
        "metric_comparisons": metric_rows,
        "ranking_comparisons": file_comparisons,
    }
    output = result_root(resources) / "r0"
    write_json(output / "baseline_reproduction.json", manifest)
    write_tsv(
        output / "baseline_reproduction.tsv",
        metric_rows,
        ("arm", "metric", "expected", "source_value", "reproduced_value", "delta_expected", "delta_source"),
    )
    contract = f"""# Baseline contract

- Status: `{manifest['status']}`
- Cohort: independently frozen benchmark v4, 50 diseases, 49 genes, 1,068 single-gene models.
- Pair provider: Exomiser 2602 current H2 HP-MP subset.
- Profile scoring: current Exomiser semi-symmetric raw average and maximum, normalized by each surface's own global HPO ideal.
- Gene aggregation: maximum model score per gene; model ID breaks model-score ties.
- Gene ranking: descending score; gene symbol breaks ties.
- HPO reduction: duplicate removal and ancestor removal from the original mission3 implementation.
- Strict/Extended reranking: original deterministic CellPhenoRank implementation, including surface-specific leave-one-HPO-out recomputation.

All reported development arms use the same frozen disease, model, phenotype and gold files. The evaluator is separate from the term-pair provider.
"""
    (output / "baseline_contract.md").write_text(contract, encoding="utf-8")
    return manifest
