#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import InputPaths, write_json, write_tsv  # noqa: E402
from mpo_hpo_mapping.pipeline import RunConfig, run_pipeline  # noqa: E402
from run_embedding_link_experiments import (  # noqa: E402
    METRICS,
    TASK_LABELS,
    metric_values,
)
from run_structure_embedding_controls import (  # noqa: E402
    TASKS,
    read_cases,
    row_multiset,
    sha256,
)


TRAINING_SEEDS = (20260726, 20260727, 20260728)
PERMUTATION_SEEDS = (20270727, 20270737, 20270747)
CONDITION_LABELS = {
    "constant": "常量 CL/GO + 旧链接",
    "permuted": "随机置换 CL/GO + 旧链接",
    "poincare": "真实 Poincare CL/GO + 旧链接",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    condition: str
    training_seed: int
    permutation_seed: int
    path: Path
    reused: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-training-seed and multi-permutation structure controls."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "results/structure_embedding_robustness_20260803",
    )
    parser.add_argument(
        "--source-experiment",
        type=Path,
        default=(
            MODULE_DIR
            / "results/embedding_link_experiment_full_deterministic_20260803"
        ),
    )
    parser.add_argument(
        "--single-seed-control",
        type=Path,
        default=(
            MODULE_DIR
            / "results/structure_embedding_control_full_deterministic_v2_20260803"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    source_root = args.source_experiment / "runs/seed_20260726"
    reused_random = (
        args.single_seed_control
        / "runs/seed_20260726/permuted_structure_baseline_links"
    )
    specs: list[RunSpec] = []
    for training_seed in TRAINING_SEEDS:
        for condition in ("constant", "poincare"):
            reused = training_seed == TRAINING_SEEDS[0]
            if reused:
                arm_id = (
                    "constant_structure_baseline_links"
                    if condition == "constant"
                    else "poincare_structure_baseline_links"
                )
                path = source_root / arm_id
            else:
                path = (
                    args.output_dir
                    / "runs"
                    / f"training_seed_{training_seed}"
                    / condition
                )
            specs.append(
                RunSpec(
                    f"{condition}_train_{training_seed}",
                    condition,
                    training_seed,
                    0,
                    path,
                    reused,
                )
            )
        for permutation_seed in PERMUTATION_SEEDS:
            reused = (
                training_seed == TRAINING_SEEDS[0]
                and permutation_seed == PERMUTATION_SEEDS[0]
            )
            path = (
                reused_random
                if reused
                else args.output_dir
                / "runs"
                / f"training_seed_{training_seed}"
                / f"permuted_{permutation_seed}"
            )
            specs.append(
                RunSpec(
                    f"permuted_train_{training_seed}_perm_{permutation_seed}",
                    "permuted",
                    training_seed,
                    permutation_seed,
                    path,
                    reused,
                )
            )
    return specs


def make_config(
    spec: RunSpec,
    hierarchy_reference: Path,
) -> RunConfig:
    return RunConfig(
        output_dir=spec.path,
        general_top_k=8,
        general_weight=0.15,
        hierarchy_dimension=16,
        hierarchy_epochs=30,
        hierarchy_mode=spec.condition,
        hierarchy_reference=(
            "" if spec.condition == "constant" else str(hierarchy_reference)
        ),
        hierarchy_permutation_seed=spec.permutation_seed,
        hpo_link_randomization="none",
        max_prototypes=4,
        semantic_mode="auto",
        semantic_dimension=64,
        semantic_batch_size=64,
        semantic_weight=1.0,
        semantic_seed=TRAINING_SEEDS[0],
        use_term_aliases=True,
        hidden_dimension=64,
        output_dimension=32,
        training_epochs=80,
        batch_size=64,
        min_mapping_confidence=0.9,
        synthetic_limit=0,
        ivf_clusters=96,
        nprobe=8,
        retrieve_k=80,
        output_top_k=20,
        evaluation_only=True,
        seed=spec.training_seed,
        data_split_seed=TRAINING_SEEDS[0],
    )


def run_missing(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    hierarchy_reference: Path,
) -> None:
    for position, spec in enumerate(specs, 1):
        manifest = spec.path / "training_manifest.json"
        cases = spec.path / "evaluation_cases.tsv"
        if spec.reused or (args.resume and manifest.is_file() and cases.is_file()):
            print(f"[{position}/{len(specs)}] Reuse {spec.run_id}", flush=True)
            continue
        print(f"[{position}/{len(specs)}] Run {spec.run_id}", flush=True)
        run_pipeline(InputPaths(), make_config(spec, hierarchy_reference))


def validate_runs(
    specs: Sequence[RunSpec], hierarchy_reference: Path
) -> dict[str, object]:
    controlled_files = (
        "data/processed/expert_mapping_examples.tsv",
        "data/processed/parent_training_examples.tsv",
        "data/processed/term_mechanism_edges.tsv",
        "data/processed/semantic_embeddings.npz",
    )
    hashes = {
        name: {spec.run_id: sha256(spec.path / name) for spec in specs}
        for name in controlled_files
    }
    mismatched = {
        name: values for name, values in hashes.items() if len(set(values.values())) != 1
    }
    if mismatched:
        raise RuntimeError(f"Controlled data mismatch: {mismatched}")

    cases = {
        spec.run_id: read_cases(spec.path / "evaluation_cases.tsv") for spec in specs
    }
    if len({frozenset(rows) for rows in cases.values()}) != 1:
        raise RuntimeError("Evaluation cohorts differ across robustness runs")

    with np.load(hierarchy_reference, allow_pickle=False) as reference:
        reference_vectors = {
            prefix: reference[f"{prefix}_vectors"].astype(np.float32)
            for prefix in ("cl", "go")
        }
        reference_ids = {
            prefix: reference[f"{prefix}_ids"].astype(str)
            for prefix in ("cl", "go")
        }
    permutation_audit: dict[str, object] = {}
    for spec in specs:
        if spec.condition != "permuted":
            continue
        with np.load(
            spec.path / "data/processed/hierarchy_embeddings.npz",
            allow_pickle=False,
        ) as archive:
            per_ontology = {}
            for prefix in ("cl", "go"):
                ids_equal = np.array_equal(
                    reference_ids[prefix], archive[f"{prefix}_ids"].astype(str)
                )
                vectors = archive[f"{prefix}_vectors"].astype(np.float32)
                multiset_equal = row_multiset(reference_vectors[prefix]) == row_multiset(
                    vectors
                )
                changed = int(
                    np.sum(np.any(reference_vectors[prefix] != vectors, axis=1))
                )
                if not ids_equal or not multiset_equal or changed == 0:
                    raise RuntimeError(f"Invalid permutation vectors in {spec.run_id}")
                per_ontology[prefix] = {
                    "ids_equal": ids_equal,
                    "vector_multiset_equal": multiset_equal,
                    "assignments_changed": changed,
                }
            permutation_audit[spec.run_id] = per_ontology

    return {
        "controlled_file_hashes_identical": True,
        "evaluation_cohort_identical": True,
        "controlled_file_hashes": hashes,
        "permutation_runs_validated": len(permutation_audit),
        "permutation_audit": permutation_audit,
        "queries_by_task": {
            task: sum(key[0] == task for key in next(iter(cases.values())))
            for task in TASKS
        },
    }


def collect_metrics(specs: Sequence[RunSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        cases = read_cases(spec.path / "evaluation_cases.tsv")
        for task in TASKS:
            ranks = [rank for key, rank in cases.items() if key[0] == task]
            rows.append(
                {
                    "run_id": spec.run_id,
                    "condition": spec.condition,
                    "condition_label": CONDITION_LABELS[spec.condition],
                    "training_seed": spec.training_seed,
                    "permutation_seed": spec.permutation_seed,
                    "task": task,
                    "queries": len(ranks),
                    **metric_values(ranks),
                }
            )
    return rows


def summarize_metrics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for condition in ("constant", "permuted", "poincare"):
        for task in TASKS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition and row["task"] == task
            ]
            result.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "task": task,
                    "runs": len(selected),
                    "training_seeds": len(
                        {int(row["training_seed"]) for row in selected}
                    ),
                    "permutation_seeds": len(
                        {
                            int(row["permutation_seed"])
                            for row in selected
                            if int(row["permutation_seed"])
                        }
                    ),
                    "queries_per_run": int(selected[0]["queries"]),
                    **{
                        f"{metric}_{suffix}": float(
                            getattr(
                                np.asarray(
                                    [float(row[metric]) for row in selected],
                                    dtype=np.float64,
                                ),
                                suffix,
                            )(ddof=1)
                            if suffix == "std" and len(selected) > 1
                            else getattr(
                                np.asarray(
                                    [float(row[metric]) for row in selected],
                                    dtype=np.float64,
                                ),
                                suffix,
                            )()
                        )
                        for metric in METRICS
                        for suffix in ("mean", "std")
                    },
                }
            )
    return result


def paired_seed_deltas(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (
            str(row["condition"]),
            int(row["training_seed"]),
            int(row["permutation_seed"]),
            str(row["task"]),
        ): row
        for row in rows
    }
    result: list[dict[str, object]] = []
    for training_seed in TRAINING_SEEDS:
        for task in TASKS:
            real = lookup[("poincare", training_seed, 0, task)]
            constant = lookup[("constant", training_seed, 0, task)]
            random_rows = [
                lookup[("permuted", training_seed, permutation_seed, task)]
                for permutation_seed in PERMUTATION_SEEDS
            ]
            for metric in METRICS:
                random_mean = float(
                    np.mean([float(row[metric]) for row in random_rows])
                )
                result.extend(
                    [
                        {
                            "comparison": "real_minus_constant",
                            "training_seed": training_seed,
                            "task": task,
                            "metric": metric,
                            "treatment": float(real[metric]),
                            "control": float(constant[metric]),
                            "delta": float(real[metric]) - float(constant[metric]),
                        },
                        {
                            "comparison": "real_minus_random_mean",
                            "training_seed": training_seed,
                            "task": task,
                            "metric": metric,
                            "treatment": float(real[metric]),
                            "control": random_mean,
                            "delta": float(real[metric]) - random_mean,
                        },
                    ]
                )
    return result


def summarize_deltas(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for comparison in ("real_minus_constant", "real_minus_random_mean"):
        for task in TASKS:
            for metric in METRICS:
                selected = [
                    float(row["delta"])
                    for row in rows
                    if row["comparison"] == comparison
                    and row["task"] == task
                    and row["metric"] == metric
                ]
                result.append(
                    {
                        "comparison": comparison,
                        "task": task,
                        "metric": metric,
                        "training_seeds": len(selected),
                        "delta_mean": float(np.mean(selected)),
                        "delta_std": float(np.std(selected, ddof=1)),
                        "positive_seeds": sum(value > 0 for value in selected),
                        "negative_seeds": sum(value < 0 for value in selected),
                        "zero_seeds": sum(value == 0 for value in selected),
                    }
                )
    return result


def summarize_random_variability(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    random_rows = [row for row in rows if row["condition"] == "permuted"]
    result: list[dict[str, object]] = []
    for task in TASKS:
        for metric in METRICS:
            within_training: list[np.ndarray] = []
            for training_seed in TRAINING_SEEDS:
                within_training.append(
                    np.asarray(
                        [
                            float(row[metric])
                            for row in random_rows
                            if row["task"] == task
                            and int(row["training_seed"]) == training_seed
                        ],
                        dtype=np.float64,
                    )
                )
            within_permutation: list[np.ndarray] = []
            for permutation_seed in PERMUTATION_SEEDS:
                within_permutation.append(
                    np.asarray(
                        [
                            float(row[metric])
                            for row in random_rows
                            if row["task"] == task
                            and int(row["permutation_seed"]) == permutation_seed
                        ],
                        dtype=np.float64,
                    )
                )
            all_values = np.concatenate(within_training)
            result.append(
                {
                    "task": task,
                    "metric": metric,
                    "runs": int(all_values.size),
                    "permutation_effect_mean_std": float(
                        np.mean([values.std(ddof=1) for values in within_training])
                    ),
                    "permutation_effect_max_range": float(
                        np.max([values.max() - values.min() for values in within_training])
                    ),
                    "training_effect_mean_std": float(
                        np.mean([values.std(ddof=1) for values in within_permutation])
                    ),
                    "training_effect_max_range": float(
                        np.max(
                            [values.max() - values.min() for values in within_permutation]
                        )
                    ),
                    "overall_std": float(all_values.std(ddof=1)),
                    "overall_range": float(all_values.max() - all_values.min()),
                }
            )
    return result


def render_html(
    summary: Sequence[Mapping[str, object]],
    delta_summary: Sequence[Mapping[str, object]],
    random_variability: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> str:
    by_key = {
        (str(row["condition"]), str(row["task"])): row for row in summary
    }
    sections: list[str] = []
    for task in TASKS:
        body = "".join(
            "<tr>"
            f"<td>{html.escape(CONDITION_LABELS[condition])}</td>"
            f"<td>{int(by_key[(condition, task)]['runs'])}</td>"
            f"<td>{int(by_key[(condition, task)]['queries_per_run'])}</td>"
            + "".join(
                f"<td>{100 * float(by_key[(condition, task)][metric + '_mean']):.2f}% ± {100 * float(by_key[(condition, task)][metric + '_std']):.2f}%</td>"
                for metric in ("recall_at_1", "recall_at_10", "recall_at_50")
            )
            + "</tr>"
            for condition in ("constant", "permuted", "poincare")
        )
        sections.append(
            f"<section><h2>{html.escape(TASK_LABELS[task])}</h2>"
            '<div class="table-wrap"><table><thead><tr><th>实验组</th>'
            "<th>运行数</th><th>每次样本数</th><th>Recall@1</th>"
            f"<th>Recall@10</th><th>Recall@50</th></tr></thead><tbody>{body}"
            "</tbody></table></div></section>"
        )
    random_deltas = [
        row
        for row in delta_summary
        if row["comparison"] == "real_minus_random_mean"
        and row["metric"] in {"recall_at_1", "recall_at_10", "recall_at_50"}
    ]
    delta_body = "".join(
        "<tr>"
        f"<td>{html.escape(TASK_LABELS[str(row['task'])])}</td>"
        f"<td>{html.escape(str(row['metric']).replace('recall_at_', 'Recall@'))}</td>"
        f"<td>{100 * float(row['delta_mean']):+.2f}% ± {100 * float(row['delta_std']):.2f}%</td>"
        f"<td>{int(row['positive_seeds'])}/3</td></tr>"
        for row in random_deltas
    )
    variability_rows = [
        row
        for row in random_variability
        if row["metric"] in {"recall_at_1", "recall_at_10", "recall_at_50"}
    ]
    variability_body = "".join(
        "<tr>"
        f"<td>{html.escape(TASK_LABELS[str(row['task'])])}</td>"
        f"<td>{html.escape(str(row['metric']).replace('recall_at_', 'Recall@'))}</td>"
        f"<td>{100 * float(row['permutation_effect_mean_std']):.2f}%</td>"
        f"<td>{100 * float(row['permutation_effect_max_range']):.2f}%</td>"
        f"<td>{100 * float(row['training_effect_mean_std']):.2f}%</td>"
        f"<td>{100 * float(row['training_effect_max_range']):.2f}%</td>"
        f"<td>{100 * float(row['overall_std']):.2f}%</td></tr>"
        for row in variability_rows
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>结构嵌入多种子稳健性实验</title>
<style>:root{{--ink:#17201d;--muted:#66716d;--line:#d9dfdc;--wash:#f4f6f5}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}header{{padding:30px max(22px,calc((100vw - 1100px)/2));background:var(--wash);border-bottom:1px solid var(--line)}}h1{{margin:0 0 7px;font-size:26px;letter-spacing:0}}header p,.note{{margin:0;color:var(--muted)}}main{{max-width:1100px;margin:auto;padding:26px 22px 50px}}.summary{{display:flex;gap:28px;flex-wrap:wrap;padding-bottom:20px;margin-bottom:28px;border-bottom:1px solid var(--line)}}.summary strong{{display:block;font-size:20px}}.summary span{{color:var(--muted)}}section{{margin:0 0 30px}}h2{{margin:0 0 9px;font-size:17px;letter-spacing:0}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th{{background:var(--wash);font-size:12px}}th:first-child,td:first-child{{text-align:left}}tr:last-child td{{border-bottom:0}}</style></head><body>
<header><h1>结构嵌入多种子稳健性实验</h1><p>3 个下游训练 seed × 3 个随机置换 seed；旧链接和数据划分保持一致。</p></header><main>
<div class="summary"><div><strong>15</strong><span>完整训练运行</span></div><div><strong>3</strong><span>训练 seed</span></div><div><strong>3</strong><span>置换 seed</span></div><div><strong>{int(audit['permutation_runs_validated'])}</strong><span>已审计随机运行</span></div></div>
{''.join(sections)}
<section><h2>真实 Poincare - 同训练 seed 的随机置换均值</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>指标</th><th>平均差值 ± seed 标准差</th><th>正向 seed</th></tr></thead><tbody>{delta_body}</tbody></table></div></section>
<section><h2>随机组波动分解</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>指标</th><th>置换波动平均 SD</th><th>置换最大跨度</th><th>训练波动平均 SD</th><th>训练最大跨度</th><th>9 次总体 SD</th></tr></thead><tbody>{variability_body}</tbody></table></div></section>
<p class="note">随机组先在每个训练 seed 内对 3 个置换取均值，再与同 seed 的真实结构比较，避免伪重复。</p></main></body></html>"""


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source_experiment = args.source_experiment.resolve()
    args.single_seed_control = args.single_seed_control.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_reference = (
        args.source_experiment
        / "runs/seed_20260726/poincare_structure_baseline_links"
        / "data/processed/hierarchy_embeddings.npz"
    )
    specs = build_specs(args)
    run_missing(args, specs, hierarchy_reference)
    audit = validate_runs(specs, hierarchy_reference)
    metrics = collect_metrics(specs)
    summary = summarize_metrics(metrics)
    deltas = paired_seed_deltas(metrics)
    delta_summary = summarize_deltas(deltas)
    random_variability = summarize_random_variability(metrics)
    report = {
        "design": {
            "training_seeds": list(TRAINING_SEEDS),
            "permutation_seeds": list(PERMUTATION_SEEDS),
            "data_split_seed": TRAINING_SEEDS[0],
            "semantic_seed": TRAINING_SEEDS[0],
            "hierarchy_reference": str(hierarchy_reference),
            "runs": [
                {
                    "run_id": spec.run_id,
                    "condition": spec.condition,
                    "training_seed": spec.training_seed,
                    "permutation_seed": spec.permutation_seed,
                    "path": str(spec.path),
                    "reused": spec.reused,
                }
                for spec in specs
            ],
        },
        "audit": audit,
        "metric_summary": summary,
        "delta_summary": delta_summary,
        "random_variability": random_variability,
    }
    write_json(args.output_dir / "experiment_report.json", report)
    write_tsv(args.output_dir / "run_metrics.tsv", metrics, list(metrics[0]))
    write_tsv(args.output_dir / "metric_summary.tsv", summary, list(summary[0]))
    write_tsv(args.output_dir / "paired_seed_deltas.tsv", deltas, list(deltas[0]))
    write_tsv(
        args.output_dir / "delta_summary.tsv", delta_summary, list(delta_summary[0])
    )
    write_tsv(
        args.output_dir / "random_variability.tsv",
        random_variability,
        list(random_variability[0]),
    )
    (args.output_dir / "index.html").write_text(
        render_html(summary, delta_summary, random_variability, audit),
        encoding="utf-8",
    )
    print(f"Wrote robustness report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
