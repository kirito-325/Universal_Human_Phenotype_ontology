#!/usr/bin/env python3
"""Load cellular mechanism data only after freeze and run a paired ablation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from common import input_record, read_json, read_tsv, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_BATCH = DEFAULT_ROOT / "mission3/data/benchmark/benchmark_v1"


def evaluate(
    rankings: dict[str, list[str]], gold: dict[str, set[str]], disease_ids: list[str]
) -> dict[str, Any]:
    best_ranks = []
    reciprocal = []
    ndcg = []
    for disease_id in disease_ids:
        ordered = rankings.get(disease_id, [])
        positives = gold.get(disease_id, set())
        ranks = [index for index, model_id in enumerate(ordered, 1) if model_id in positives]
        if not ranks:
            continue
        best = min(ranks)
        best_ranks.append(best)
        reciprocal.append(1.0 / best)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(10, len(positives)) + 1)
        )
        ndcg.append(dcg / ideal if ideal else 0.0)
    return {
        "evaluated_diseases": len(best_ranks),
        "recall_at_1": mean(rank <= 1 for rank in best_ranks) if best_ranks else 0.0,
        "recall_at_5": mean(rank <= 5 for rank in best_ranks) if best_ranks else 0.0,
        "recall_at_10": mean(rank <= 10 for rank in best_ranks) if best_ranks else 0.0,
        "mrr": mean(reciprocal) if reciprocal else 0.0,
        "ndcg_at_10": mean(ndcg) if ndcg else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BATCH / "evaluation/cpo_ablation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_ROOT / "mission3/config/provisional_cpo_diagnostic.json",
    )
    parser.add_argument(
        "--mp-hpo-mapping",
        type=Path,
        default=DEFAULT_ROOT / "UHPO/data/mpo_hpo_mapping/mpo_hpo_mapping.tsv",
    )
    parser.add_argument(
        "--hpo-mechanisms",
        type=Path,
        default=DEFAULT_ROOT / "UHPO/data/hpo_cpo_links/hpo_cpo_candidate_links.partial.tsv",
    )
    parser.add_argument(
        "--mp-mechanisms",
        type=Path,
        default=DEFAULT_ROOT
        / "03_MPO_link/results/mpo_cpo_links/api_reasoner_final_20260726/mpo_cell_mechanism_triples.tsv",
    )
    parser.add_argument(
        "--cell-catalog",
        type=Path,
        default=DEFAULT_ROOT / "UHPO/data/cpo/cell_catalog.tsv",
    )
    parser.add_argument(
        "--go-classification",
        type=Path,
        default=DEFAULT_ROOT / "UHPO/data/cpo/go_term_classification.tsv",
    )
    parser.add_argument(
        "--cl-obo",
        type=Path,
        default=DEFAULT_ROOT / "HPO_link_v6/data/ontologies/cl.obo",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=DEFAULT_ROOT / "HPO_link_v6/data/ontologies/go-basic.obo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    batch = args.batch_dir.resolve()
    frozen = batch / "frozen"
    freeze_manifest_path = frozen / "freeze_manifest.json"
    if not freeze_manifest_path.is_file():
        raise ValueError(
            "Refusing ablation: frozen/freeze_manifest.json does not exist; "
            "complete independent research and validation first."
        )
    freeze_manifest = read_json(freeze_manifest_path)
    if not freeze_manifest.get("frozen") or not freeze_manifest.get(
        "middle_layer_independent"
    ):
        raise ValueError("Refusing ablation: benchmark is not independently frozen")
    output = args.output_dir.resolve()
    pipeline_output = output / "pipeline"
    command = [
        "python3",
        "-B",
        str(root / "mission3/scripts/run_pipeline.py"),
        "run",
        "--diseases",
        str(frozen / "diseases.tsv"),
        "--models",
        str(frozen / "models.tsv"),
        "--model-phenotypes",
        str(frozen / "model_phenotypes.tsv"),
        "--mp-hpo-mapping",
        str(args.mp_hpo_mapping),
        "--hpo-mechanisms",
        str(args.hpo_mechanisms),
        "--mp-mechanisms",
        str(args.mp_mechanisms),
        "--cell-catalog",
        str(args.cell_catalog),
        "--go-classification",
        str(args.go_classification),
        "--cl-obo",
        str(args.cl_obo),
        "--go-obo",
        str(args.go_obo),
        "--config",
        str(args.config),
        "--output-dir",
        str(pipeline_output),
    ]
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    (output / "pipeline.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "pipeline.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Mission 3 pipeline failed with exit code {result.returncode}")

    config = read_json(args.config)
    cpo_weight = float(config.get("weights", {}).get("cpo", 0.25))
    rows_by_disease: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(pipeline_output / "model_rankings.tsv"):
        rows_by_disease[row["disease_id"]].append(row)
    gold: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(frozen / "gold_models.tsv"):
        gold[row["disease_id"]].add(row["model_id"])
    comparable_hpo: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(pipeline_output / "phenotype_evidence.tsv"):
        if row.get("cpo_score", "").strip():
            comparable_hpo[row["disease_id"]].add(row["hpo_id"])

    surface_rankings: dict[str, list[str]] = {}
    cpo_rankings: dict[str, list[str]] = {}
    delta_rows = []
    for disease_id, rows in sorted(rows_by_disease.items()):
        surface_order = sorted(
            rows, key=lambda row: (-float(row["surface_score"]), row["model_id"])
        )
        cpo_order = sorted(
            rows,
            key=lambda row: (
                -(
                    float(row["surface_score"])
                    + cpo_weight
                    * float(row.get("cpo_coverage") or 0.0)
                    * float(row.get("cpo_score") or 0.0)
                ),
                row["model_id"],
            ),
        )
        surface_rankings[disease_id] = [row["model_id"] for row in surface_order]
        cpo_rankings[disease_id] = [row["model_id"] for row in cpo_order]
        surface_gold_ranks = [
            index
            for index, model_id in enumerate(surface_rankings[disease_id], 1)
            if model_id in gold[disease_id]
        ]
        cpo_gold_ranks = [
            index
            for index, model_id in enumerate(cpo_rankings[disease_id], 1)
            if model_id in gold[disease_id]
        ]
        surface_best = min(surface_gold_ranks) if surface_gold_ranks else 0
        cpo_best = min(cpo_gold_ranks) if cpo_gold_ranks else 0
        delta = surface_best - cpo_best if surface_best and cpo_best else 0
        delta_rows.append(
            {
                "disease_id": disease_id,
                "surface_best_gold_rank": surface_best,
                "cpo_best_gold_rank": cpo_best,
                "rank_improvement": delta,
                "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                "comparable_hpo_count": len(comparable_hpo[disease_id]),
                "cpo_evaluable": len(comparable_hpo[disease_id]) >= 2,
            }
        )

    all_ids = sorted(rows_by_disease)
    evaluable_ids = sorted(
        disease_id for disease_id in all_ids if len(comparable_hpo[disease_id]) >= 2
    )
    all_surface = evaluate(surface_rankings, gold, all_ids)
    all_cpo = evaluate(cpo_rankings, gold, all_ids)
    evaluable_surface = evaluate(surface_rankings, gold, evaluable_ids)
    evaluable_cpo = evaluate(cpo_rankings, gold, evaluable_ids)

    def deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        return {
            metric: float(right[metric]) - float(left[metric])
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": input_record(frozen / "freeze_manifest.json", root),
        "middle_layer_loaded_after_freeze": True,
        "disease_count": len(all_ids),
        "cpo_evaluable_disease_count": len(evaluable_ids),
        "all_diseases": {
            "surface": all_surface,
            "surface_plus_cpo": all_cpo,
            "delta": deltas(all_surface, all_cpo),
        },
        "cpo_evaluable_subset": {
            "definition": "at least two distinct human phenotype terms with a comparable cellular mechanism score",
            "surface": evaluable_surface,
            "surface_plus_cpo": evaluable_cpo,
            "delta": deltas(evaluable_surface, evaluable_cpo),
        },
        "rank_outcomes": dict(Counter(row["outcome"] for row in delta_rows)),
        "mechanism_inputs": {
            "human_mechanisms": input_record(args.hpo_mechanisms, root),
            "mouse_mechanisms": input_record(args.mp_mechanisms, root),
        },
        "pipeline_command": command,
    }
    write_tsv(
        output / "per_disease_rank_delta.tsv",
        delta_rows,
        [
            "disease_id",
            "surface_best_gold_rank",
            "cpo_best_gold_rank",
            "rank_improvement",
            "outcome",
            "comparable_hpo_count",
            "cpo_evaluable",
        ],
    )
    write_json(output / "cpo_gain_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
