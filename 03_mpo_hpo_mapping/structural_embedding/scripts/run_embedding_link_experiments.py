#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import InputPaths, write_json, write_tsv  # noqa: E402
from mpo_hpo_mapping.pipeline import RunConfig, run_pipeline  # noqa: E402


TASKS = (
    "expert_exact_present",
    "expert_hp_broader",
    "counterfactual_exact_removed",
    "synthetic_hpo_cutout",
)
REPORT_TASKS = TASKS + ("exact_absent_combined",)
TASK_LABELS = {
    "expert_exact_present": "专家金标准：exact",
    "expert_hp_broader": "专家金标准：broader",
    "counterfactual_exact_removed": "构建金标准：exact 遮蔽",
    "synthetic_hpo_cutout": "构建金标准：HPO cutout",
    "exact_absent_combined": "构建金标准：合并",
}
TWO_TASK_LAYOUT = (
    (
        "exact_retrieval",
        "任务一：等价 HPO 召回",
        "当等价 HPO 存在时，检验能否召回正确的等价术语。",
        (("expert_exact_present", "专家 exact 金标准"),),
    ),
    (
        "ancestor_retrieval",
        "任务二：HPO 上位类召回",
        "当等价 HPO 不可用时，检验能否召回正确的 HPO 上位类。",
        (
            ("expert_hp_broader", "专家 broader 金标准"),
            ("counterfactual_exact_removed", "构建金标准：真实 exact 遮蔽"),
            ("synthetic_hpo_cutout", "构建金标准：HPO 子节点切除"),
        ),
    ),
)
CASE_TASK_TO_SUMMARY = {
    "exact": "expert_exact_present",
    "hp_broader": "expert_hp_broader",
    "counterfactual_exact": "counterfactual_exact_removed",
    "synthetic_hpo_cutout": "synthetic_hpo_cutout",
}
METRICS = ("mrr", "recall_at_1", "recall_at_10", "recall_at_50")


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    hierarchy_mode: str
    link_source: str
    randomization: str = "none"


ARMS = (
    Arm(
        "constant_structure_baseline_links",
        "常量结构 + 旧链接",
        "constant",
        "baseline",
    ),
    Arm(
        "poincare_structure_baseline_links",
        "Poincare 结构 + 旧链接",
        "poincare",
        "baseline",
    ),
    Arm(
        "poincare_structure_v4_links",
        "Poincare 结构 + HPO_link_v4",
        "poincare",
        "v4",
    ),
    Arm(
        "poincare_structure_v4_random_links",
        "Poincare 结构 + v4 随机链接",
        "poincare",
        "v4",
        "degree_preserving",
    ),
)

COMPARISONS = (
    (
        "structure_effect",
        "真实结构 - 常量结构",
        "poincare_structure_baseline_links",
        "constant_structure_baseline_links",
    ),
    (
        "v4_link_effect",
        "v4 链接 - 旧链接",
        "poincare_structure_v4_links",
        "poincare_structure_baseline_links",
    ),
    (
        "link_signal_vs_random",
        "v4 真实链接 - v4 随机链接",
        "poincare_structure_v4_links",
        "poincare_structure_v4_random_links",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled structure-embedding and HPO-link experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "results/embedding_link_experiment_20260803",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260726])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--semantic-mode",
        choices=("auto", "sapbert", "tfidf", "none"),
        default="auto",
    )
    parser.add_argument("--training-epochs", type=int, default=0)
    parser.add_argument("--hierarchy-epochs", type=int, default=0)
    parser.add_argument("--disable-term-aliases", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--randomization-swaps-per-edge", type=int, default=10)
    parser.add_argument("--baseline-hpo-specific", type=Path)
    parser.add_argument("--baseline-hpo-general", type=Path)
    parser.add_argument(
        "--v4-hpo-specific",
        type=Path,
        default=ROOT / "HPO_link_v4/results/full_20260731/hpo_cpo_specific_links.tsv",
    )
    parser.add_argument(
        "--v4-hpo-general",
        type=Path,
        default=ROOT / "HPO_link_v4/results/full_20260731/hpo_cpo_general_links.tsv",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace, arm: Arm) -> InputPaths:
    defaults = InputPaths()
    if arm.link_source == "v4":
        hpo_specific = args.v4_hpo_specific
        hpo_general = args.v4_hpo_general
    else:
        hpo_specific = args.baseline_hpo_specific or defaults.hpo_specific
        hpo_general = args.baseline_hpo_general or defaults.hpo_general
    return InputPaths(
        hpo_specific=hpo_specific,
        hpo_general=hpo_general,
        mpo_specific=defaults.mpo_specific,
        mpo_general=defaults.mpo_general,
        expert_mappings=defaults.expert_mappings,
        hpo_edges=defaults.hpo_edges,
        hpo_obo=defaults.hpo_obo,
        mpo_obo=defaults.mpo_obo,
        cl_catalog=defaults.cl_catalog,
        go_obo=defaults.go_obo,
    )


def make_config(
    args: argparse.Namespace, arm: Arm, seed: int, output_dir: Path
) -> RunConfig:
    smoke = args.smoke
    return RunConfig(
        output_dir=output_dir,
        general_top_k=2 if smoke else 8,
        general_weight=0.15,
        hierarchy_dimension=8 if smoke else 16,
        hierarchy_epochs=args.hierarchy_epochs or (5 if smoke else 30),
        hierarchy_mode=arm.hierarchy_mode,
        hpo_link_randomization=arm.randomization,
        randomization_swaps_per_edge=args.randomization_swaps_per_edge,
        max_prototypes=2 if smoke else 4,
        semantic_mode=(
            "tfidf" if smoke and args.semantic_mode == "auto" else args.semantic_mode
        ),
        semantic_dimension=32 if smoke else 64,
        semantic_batch_size=64,
        semantic_weight=1.0,
        use_term_aliases=not args.disable_term_aliases,
        hidden_dimension=32 if smoke else 64,
        output_dimension=16 if smoke else 32,
        training_epochs=args.training_epochs or (12 if smoke else 80),
        batch_size=64,
        min_mapping_confidence=0.9,
        synthetic_limit=500 if smoke else 0,
        ivf_clusters=32 if smoke else 96,
        nprobe=4 if smoke else 8,
        retrieve_k=30 if smoke else 80,
        output_top_k=5 if smoke else 20,
        max_output_mp=100 if smoke else 0,
        evaluation_only=True,
        seed=seed,
    )


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_arms(args: argparse.Namespace) -> dict[tuple[int, str], dict[str, object]]:
    manifests: dict[tuple[int, str], dict[str, object]] = {}
    for seed in args.seeds:
        for arm in ARMS:
            run_dir = args.output_dir / "runs" / f"seed_{seed}" / arm.arm_id
            manifest_path = run_dir / "training_manifest.json"
            cases_path = run_dir / "evaluation_cases.tsv"
            if args.resume and manifest_path.is_file() and cases_path.is_file():
                print(f"Reuse {arm.arm_id}, seed {seed}", flush=True)
                manifests[(seed, arm.arm_id)] = load_json(manifest_path)
                continue
            started = time.monotonic()
            print(f"Run {arm.arm_id}, seed {seed}", flush=True)
            manifest = run_pipeline(
                make_paths(args, arm), make_config(args, arm, seed, run_dir)
            )
            manifest["experiment_arm"] = {
                "arm_id": arm.arm_id,
                "label": arm.label,
                "link_source": arm.link_source,
            }
            write_json(manifest_path, manifest)
            manifests[(seed, arm.arm_id)] = manifest
            print(
                f"Completed {arm.arm_id}, seed {seed}, "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
            gc.collect()
    return manifests


CaseKey = tuple[str, str, str]


def read_cases(path: Path) -> dict[CaseKey, int]:
    rows: dict[CaseKey, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            task = CASE_TASK_TO_SUMMARY[row["task"]]
            key = (task, row["query_id"], row["targets"])
            rows[key] = int(row["best_rank"] or 0)
    return rows


def metric_values(ranks: Sequence[int]) -> dict[str, float]:
    count = max(1, len(ranks))
    positive = [rank for rank in ranks if rank > 0]
    return {
        "mrr": sum(1.0 / rank for rank in positive) / count,
        "recall_at_1": sum(rank <= 1 for rank in positive) / count,
        "recall_at_10": sum(rank <= 10 for rank in positive) / count,
        "recall_at_50": sum(rank <= 50 for rank in positive) / count,
    }


def observation_value(rank: int, metric: str) -> float:
    if metric == "mrr":
        return 1.0 / rank if rank > 0 else 0.0
    cutoff = int(metric.rsplit("_", 1)[1])
    return float(0 < rank <= cutoff)


def bootstrap_interval(
    observations: Sequence[tuple[int, int]],
    metric: str,
    samples: int,
    key: str,
) -> tuple[float, float]:
    if not observations or samples <= 0:
        return 0.0, 0.0
    differences = np.asarray(
        [
            observation_value(treatment, metric) - observation_value(control, metric)
            for control, treatment in observations
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256(key.encode()).digest()
    randomizer = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = randomizer.integers(
        0, len(differences), size=(samples, len(differences))
    )
    estimates = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_comparisons(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_run = {
        (seed, arm.arm_id): read_cases(
            args.output_dir
            / "runs"
            / f"seed_{seed}"
            / arm.arm_id
            / "evaluation_cases.tsv"
        )
        for seed in args.seeds
        for arm in ARMS
    }
    summary_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for comparison_id, label, treatment_id, control_id in COMPARISONS:
        observations_by_task: dict[str, list[tuple[int, int]]] = {
            task: [] for task in TASKS
        }
        for seed in args.seeds:
            treatment = by_run[(seed, treatment_id)]
            control = by_run[(seed, control_id)]
            common = sorted(set(treatment) & set(control))
            for task, query_id, targets in common:
                treatment_rank = treatment[(task, query_id, targets)]
                control_rank = control[(task, query_id, targets)]
                observations_by_task[task].append((control_rank, treatment_rank))
                delta_rr = observation_value(treatment_rank, "mrr") - observation_value(
                    control_rank, "mrr"
                )
                case_rows.append(
                    {
                        "comparison": comparison_id,
                        "seed": seed,
                        "task": task,
                        "query_id": query_id,
                        "targets": targets,
                        "control_rank": control_rank,
                        "treatment_rank": treatment_rank,
                        "delta_reciprocal_rank": f"{delta_rr:.8g}",
                        "outcome": "improved"
                        if delta_rr > 0
                        else "worse"
                        if delta_rr < 0
                        else "tied",
                    }
                )
        observations_by_task["exact_absent_combined"] = [
            item
            for task in ("counterfactual_exact_removed", "synthetic_hpo_cutout")
            for item in observations_by_task[task]
        ]
        for task in REPORT_TASKS:
            observations = observations_by_task[task]
            control_metrics = metric_values([row[0] for row in observations])
            treatment_metrics = metric_values([row[1] for row in observations])
            for metric in METRICS:
                low, high = bootstrap_interval(
                    observations,
                    metric,
                    args.bootstrap_samples,
                    f"{comparison_id}|{task}|{metric}|{args.seeds}",
                )
                summary_rows.append(
                    {
                        "comparison": comparison_id,
                        "comparison_label": label,
                        "task": task,
                        "paired_queries": len(observations),
                        "metric": metric,
                        "control": f"{control_metrics[metric]:.8g}",
                        "treatment": f"{treatment_metrics[metric]:.8g}",
                        "delta": f"{treatment_metrics[metric] - control_metrics[metric]:.8g}",
                        "ci_95_low": f"{low:.8g}",
                        "ci_95_high": f"{high:.8g}",
                    }
                )
    return summary_rows, case_rows


def common_cohort_metrics(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score every arm on the strict four-way intersection of gold cases."""
    rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    arm_by_id = {arm.arm_id: arm for arm in ARMS}
    for seed in args.seeds:
        by_arm = {
            arm.arm_id: read_cases(
                args.output_dir
                / "runs"
                / f"seed_{seed}"
                / arm.arm_id
                / "evaluation_cases.tsv"
            )
            for arm in ARMS
        }
        common = set.intersection(*(set(cases) for cases in by_arm.values()))
        by_task = {
            task: sorted(key for key in common if key[0] == task) for task in TASKS
        }
        for task in TASKS:
            for _, query_id, targets in by_task[task]:
                cohort_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "query_id": query_id,
                        "targets": targets,
                    }
                )
        for arm_id, cases in by_arm.items():
            task_ranks = {
                task: [cases[key] for key in by_task[task]] for task in TASKS
            }
            task_ranks["exact_absent_combined"] = [
                rank
                for task in (
                    "counterfactual_exact_removed",
                    "synthetic_hpo_cutout",
                )
                for rank in task_ranks[task]
            ]
            for task in REPORT_TASKS:
                ranks = task_ranks[task]
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm_id,
                        "arm_label": arm_by_id[arm_id].label,
                        "task": task,
                        "queries": len(ranks),
                        **metric_values(ranks),
                    }
                )
    return rows, cohort_rows


def arm_metric_rows(
    manifests: Mapping[tuple[int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    arm_by_id = {arm.arm_id: arm for arm in ARMS}
    for (seed, arm_id), manifest in sorted(manifests.items()):
        arm = arm_by_id[arm_id]
        data = manifest["data"]
        evaluation = manifest["evaluation"]
        task_metrics = {task: evaluation[task] for task in TASKS}
        constructed_parts = [
            task_metrics["counterfactual_exact_removed"],
            task_metrics["synthetic_hpo_cutout"],
        ]
        constructed_queries = sum(int(part["queries"]) for part in constructed_parts)
        task_metrics["exact_absent_combined"] = {
            "queries": constructed_queries,
            **{
                metric: sum(
                    float(part[metric]) * int(part["queries"])
                    for part in constructed_parts
                )
                / max(1, constructed_queries)
                for metric in METRICS
            },
        }
        for task in REPORT_TASKS:
            metrics = task_metrics[task]
            rows.append(
                {
                    "seed": seed,
                    "arm": arm_id,
                    "arm_label": arm.label,
                    "task": task,
                    "queries": metrics["queries"],
                    "mrr": metrics["mrr"],
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_10": metrics["recall_at_10"],
                    "recall_at_50": metrics["recall_at_50"],
                    "hpo_profile_terms": data["hpo_profile_terms"],
                    "term_mechanism_edges": data["term_mechanism_edges"],
                    "cl_unique_vectors": data["cl_unique_vectors"],
                    "go_unique_vectors": data["go_unique_vectors"],
                }
            )
    return rows


def mean_arm_metrics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), str(row["task"])), []).append(row)
    result: list[dict[str, object]] = []
    for (arm_id, task), members in sorted(grouped.items()):
        result.append(
            {
                "arm": arm_id,
                "arm_label": members[0]["arm_label"],
                "task": task,
                "seeds": len(members),
                "queries": float(
                    np.mean([float(row["queries"]) for row in members])
                ),
                **{
                    metric: float(np.mean([float(row[metric]) for row in members]))
                    for metric in METRICS
                },
                "hpo_profile_terms": float(
                    np.mean([float(row["hpo_profile_terms"]) for row in members])
                ),
                "term_mechanism_edges": float(
                    np.mean([float(row["term_mechanism_edges"]) for row in members])
                ),
            }
        )
    return result


def mean_recall_metrics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), str(row["task"])), []).append(row)
    result: list[dict[str, object]] = []
    for (arm_id, task), members in sorted(grouped.items()):
        result.append(
            {
                "arm": arm_id,
                "arm_label": members[0]["arm_label"],
                "task": task,
                "seeds": len(members),
                "queries": float(
                    np.mean([float(row["queries"]) for row in members])
                ),
                **{
                    metric: float(np.mean([float(row[metric]) for row in members]))
                    for metric in METRICS
                },
            }
        )
    return result


def format_metric(value: object) -> str:
    return f"{float(value):.4f}"


def render_html(
    args: argparse.Namespace,
    mean_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
) -> str:
    arm_labels = {arm.arm_id: arm.label for arm in ARMS}
    raw_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['arm_label']))}</td>"
        f"<td>{html.escape(TASK_LABELS.get(str(row['task']), str(row['task'])))}</td>"
        f"<td>{format_metric(row['mrr'])}</td>"
        f"<td>{format_metric(row['recall_at_1'])}</td>"
        f"<td>{format_metric(row['recall_at_10'])}</td>"
        f"<td>{format_metric(row['recall_at_50'])}</td>"
        f"<td>{int(float(row['hpo_profile_terms'])):,}</td>"
        "</tr>"
        for row in mean_rows
    )
    selected = [row for row in comparison_rows if row["metric"] in METRICS]
    comparison_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['comparison_label']))}</td>"
        f"<td>{html.escape(TASK_LABELS.get(str(row['task']), str(row['task'])))}</td>"
        f"<td>{html.escape(str(row['metric']))}</td>"
        f"<td>{int(row['paired_queries']):,}</td>"
        f"<td>{format_metric(row['control'])}</td>"
        f"<td>{format_metric(row['treatment'])}</td>"
        f"<td class=\"{'positive' if float(row['delta']) > 0 else 'negative' if float(row['delta']) < 0 else ''}\">{float(row['delta']):+.4f}</td>"
        f"<td>[{float(row['ci_95_low']):+.4f}, {float(row['ci_95_high']):+.4f}]</td>"
        "</tr>"
        for row in selected
    )
    arm_design = "".join(
        "<tr>"
        f"<td>{html.escape(arm.label)}</td>"
        f"<td>{html.escape(arm.hierarchy_mode)}</td>"
        f"<td>{html.escape(arm.link_source)}</td>"
        f"<td>{html.escape(arm.randomization)}</td>"
        "</tr>"
        for arm in ARMS
    )
    mode = "smoke" if args.smoke else "full"
    aliases = "disabled" if args.disable_term_aliases else "enabled"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>结构嵌入与 HPO 链接消融实验</title>
<style>
:root {{ color-scheme: light; --ink:#17201d; --muted:#66716d; --line:#d9dfdc; --paper:#fff; --wash:#f4f6f5; --green:#08745a; --red:#b33939; --gold:#9a6a00; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ border-bottom:1px solid var(--line); background:var(--wash); padding:34px max(24px,calc((100vw - 1280px)/2)); }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
header p {{ margin:0; color:var(--muted); }}
main {{ max-width:1280px; margin:0 auto; padding:28px 24px 56px; }}
section {{ margin:0 0 34px; }}
h2 {{ margin:0 0 12px; font-size:18px; letter-spacing:0; }}
.meta {{ display:flex; gap:28px; flex-wrap:wrap; padding:14px 0; border-bottom:1px solid var(--line); }}
.meta strong {{ display:block; font-size:20px; }}
.meta span {{ color:var(--muted); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right; }}
th {{ position:sticky; top:0; background:var(--wash); color:#3d4844; font-size:12px; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }}
.positive {{ color:var(--green); font-weight:700; }}
.negative {{ color:var(--red); font-weight:700; }}
.note {{ color:var(--muted); max-width:920px; }}
code {{ color:var(--gold); }}
</style>
</head>
<body>
<header><h1>结构嵌入与 HPO 链接消融实验</h1><p>沿用专家金标准与构建金标准；正值表示实验组优于对照组。</p></header>
<main>
<section class="meta"><div><strong>{mode}</strong><span>运行模式</span></div><div><strong>{html.escape(args.semantic_mode)}</strong><span>语义嵌入</span></div><div><strong>{aliases}</strong><span>术语别名</span></div><div><strong>{len(args.seeds)}</strong><span>随机种子</span></div><div><strong>{args.bootstrap_samples:,}</strong><span>bootstrap 次数</span></div></section>
<section><h2>实验设计</h2><div class="table-wrap"><table><thead><tr><th>实验组</th><th>结构</th><th>HPO 链接</th><th>随机化</th></tr></thead><tbody>{arm_design}</tbody></table></div></section>
<section><h2>金标准配对比较</h2><p class="note">专家金标准分别报告 exact 与 broader；构建金标准分别报告 exact 遮蔽、HPO cutout，并沿用旧口径给出二者合并结果。仅纳入两组都存在且目标集合完全相同的查询。</p><div class="table-wrap"><table><thead><tr><th>比较</th><th>金标准</th><th>指标</th><th>配对查询</th><th>对照</th><th>实验</th><th>差值</th><th>95% CI</th></tr></thead><tbody>{comparison_html}</tbody></table></div></section>
<section><h2>各组原始指标</h2><p class="note">跨种子取均值。不同链接版本的原始查询数可能不同，因此版本结论应优先看上面的配对比较。</p><div class="table-wrap"><table><thead><tr><th>实验组</th><th>任务</th><th>MRR</th><th>R@1</th><th>R@10</th><th>R@50</th><th>HPO 覆盖</th></tr></thead><tbody>{raw_rows}</tbody></table></div></section>
<section><p class="note">完整逐查询结果见 <code>paired_case_deltas.tsv</code>，机器可读结果见 <code>experiment_report.json</code>。</p></section>
</main>
</body></html>
"""


def render_recall_html(
    args: argparse.Namespace,
    mean_rows: Sequence[Mapping[str, object]],
    *,
    title: str = "四组实验金标准召回率",
    subtitle: str = "展示各实验组的实际召回率，不使用跨任务综合分数。",
    note: str = (
        "旧链接与 v4 的 HPO 覆盖不同，因此查询数可能不同。版本间因果比较"
        "请结合主报告中的相同 query 配对结果；本页用于查看每组在各自正式"
        "金标准上的绝对表现。"
    ),
) -> str:
    by_key = {
        (str(row["arm"]), str(row["task"])): row for row in mean_rows
    }
    sections: list[str] = []
    for task in REPORT_TASKS:
        rows = [by_key[(arm.arm_id, task)] for arm in ARMS]
        best = {
            metric: max(float(row[metric]) for row in rows) for metric in METRICS
        }
        body = "".join(
            "<tr>"
            f"<td>{html.escape(arm.label)}</td>"
            f"<td>{int(float(row['queries'])):,}</td>"
            + "".join(
                f"<td class=\"{'best' if abs(float(row[metric]) - best[metric]) < 1e-12 else ''}\">{100.0 * float(row[metric]):.2f}%</td>"
                for metric in ("recall_at_1", "recall_at_10", "recall_at_50")
            )
            + f"<td class=\"{'best' if abs(float(row['mrr']) - best['mrr']) < 1e-12 else ''}\">{float(row['mrr']):.4f}</td>"
            "</tr>"
            for arm, row in zip(ARMS, rows)
        )
        sections.append(
            f"<section><h2>{html.escape(TASK_LABELS[task])}</h2>"
            '<div class="table-wrap"><table><thead><tr><th>实验组</th>'
            "<th>查询数</th><th>Recall@1</th><th>Recall@10</th>"
            f"<th>Recall@50</th><th>MRR</th></tr></thead><tbody>{body}"
            "</tbody></table></div></section>"
        )
    mode = "smoke" if args.smoke else "full"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme:light; --ink:#17201d; --muted:#66716d; --line:#d9dfdc; --wash:#f4f6f5; --green:#08745a; --green-wash:#e8f4ef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ padding:30px max(22px,calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); background:var(--wash); }}
h1 {{ margin:0 0 7px; font-size:26px; letter-spacing:0; }}
header p,.note {{ margin:0; color:var(--muted); }}
main {{ max-width:1080px; margin:0 auto; padding:26px 22px 50px; }}
.summary {{ display:flex; gap:28px; padding-bottom:20px; margin-bottom:26px; border-bottom:1px solid var(--line); }}
.summary strong {{ display:block; font-size:20px; }}
.summary span {{ color:var(--muted); }}
section {{ margin:0 0 30px; }}
h2 {{ margin:0 0 10px; font-size:17px; letter-spacing:0; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; }}
th {{ background:var(--wash); color:#43504b; font-size:12px; }}
th:first-child,td:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }}
.best {{ color:var(--green); background:var(--green-wash); font-weight:700; }}
.note {{ padding-top:4px; max-width:900px; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>
<main>
<div class="summary"><div><strong>{mode}</strong><span>运行模式</span></div><div><strong>{len(args.seeds)}</strong><span>随机种子</span></div><div><strong>deterministic</strong><span>训练模式</span></div></div>
{''.join(sections)}
<p class="note">{html.escape(note)}</p>
</main>
</body></html>
"""


def two_task_metric_rows(
    mean_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key = {(str(row["arm"]), str(row["task"])): row for row in mean_rows}
    result: list[dict[str, object]] = []
    for task_id, task_label, _, gold_sets in TWO_TASK_LAYOUT:
        for metric_task, gold_label in gold_sets:
            for arm in ARMS:
                row = by_key[(arm.arm_id, metric_task)]
                result.append(
                    {
                        "evaluation_task": task_id,
                        "evaluation_task_label": task_label,
                        "gold_standard": gold_label,
                        **row,
                    }
                )
    return result


def render_two_task_recall_html(
    args: argparse.Namespace,
    mean_rows: Sequence[Mapping[str, object]],
) -> str:
    by_key = {(str(row["arm"]), str(row["task"])): row for row in mean_rows}
    task_sections: list[str] = []
    for _, task_label, task_description, gold_sets in TWO_TASK_LAYOUT:
        gold_sections: list[str] = []
        for metric_task, gold_label in gold_sets:
            rows = [by_key[(arm.arm_id, metric_task)] for arm in ARMS]
            best = {
                metric: max(float(row[metric]) for row in rows) for metric in METRICS
            }
            body = "".join(
                "<tr>"
                f"<td>{html.escape(arm.label)}</td>"
                f"<td>{int(float(row['queries'])):,}</td>"
                + "".join(
                    f"<td class=\"{'best' if abs(float(row[metric]) - best[metric]) < 1e-12 else ''}\">{100.0 * float(row[metric]):.2f}%</td>"
                    for metric in ("recall_at_1", "recall_at_10", "recall_at_50")
                )
                + f"<td class=\"{'best' if abs(float(row['mrr']) - best['mrr']) < 1e-12 else ''}\">{float(row['mrr']):.4f}</td>"
                "</tr>"
                for arm, row in zip(ARMS, rows)
            )
            gold_sections.append(
                f"<h3>{html.escape(gold_label)}</h3>"
                '<div class="table-wrap"><table><thead><tr><th>实验组</th>'
                "<th>查询数</th><th>Recall@1</th><th>Recall@10</th>"
                f"<th>Recall@50</th><th>MRR</th></tr></thead><tbody>{body}"
                "</tbody></table></div>"
            )
        task_sections.append(
            f"<section><h2>{html.escape(task_label)}</h2>"
            f"<p class=\"task-note\">{html.escape(task_description)}</p>"
            f"{''.join(gold_sections)}</section>"
        )
    mode = "smoke" if args.smoke else "full"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四组实验两类任务统一样本评估</title>
<style>
:root {{ color-scheme:light; --ink:#17201d; --muted:#66716d; --line:#d9dfdc; --wash:#f4f6f5; --green:#08745a; --green-wash:#e8f4ef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ padding:30px max(22px,calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); background:var(--wash); }}
h1 {{ margin:0 0 7px; font-size:26px; letter-spacing:0; }}
header p,.note,.task-note {{ margin:0; color:var(--muted); }}
main {{ max-width:1080px; margin:0 auto; padding:26px 22px 50px; }}
.summary {{ display:flex; gap:28px; padding-bottom:20px; margin-bottom:28px; border-bottom:1px solid var(--line); }}
.summary strong {{ display:block; font-size:20px; }}
.summary span {{ color:var(--muted); }}
section {{ margin:0 0 42px; }}
h2 {{ margin:0 0 4px; font-size:20px; letter-spacing:0; }}
h3 {{ margin:20px 0 9px; font-size:15px; letter-spacing:0; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; }}
th {{ background:var(--wash); color:#43504b; font-size:12px; }}
th:first-child,td:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }}
.best {{ color:var(--green); background:var(--green-wash); font-weight:700; }}
.note {{ padding-top:4px; max-width:900px; }}
</style>
</head>
<body>
<header><h1>四组实验两类任务统一样本评估</h1><p>按原任务定义分别报告等价 HPO 召回和 HPO 上位类召回。</p></header>
<main>
<div class="summary"><div><strong>{mode}</strong><span>运行模式</span></div><div><strong>{len(args.seeds)}</strong><span>随机种子</span></div><div><strong>strict intersection</strong><span>统一样本口径</span></div></div>
{''.join(task_sections)}
<p class="note">所有表均使用四组共同的 query 与目标集合。构建金标准的合并指标只保留在 common_cohort_recall.tsv 中，不作为两类任务的主结果。</p>
</main>
</body></html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = run_arms(args)
    arm_rows = arm_metric_rows(manifests)
    comparison_rows, case_rows = paired_comparisons(args)
    mean_rows = mean_arm_metrics(arm_rows)
    common_rows, common_cases = common_cohort_metrics(args)
    common_mean_rows = mean_recall_metrics(common_rows)
    two_task_rows = two_task_metric_rows(common_mean_rows)

    write_tsv(
        args.output_dir / "arm_metrics.tsv",
        arm_rows,
        list(arm_rows[0]) if arm_rows else [],
    )
    write_tsv(
        args.output_dir / "comparison_summary.tsv",
        comparison_rows,
        list(comparison_rows[0]) if comparison_rows else [],
    )
    write_tsv(
        args.output_dir / "paired_case_deltas.tsv",
        case_rows,
        list(case_rows[0]) if case_rows else [],
    )
    write_tsv(
        args.output_dir / "common_cohort_metrics.tsv",
        common_rows,
        list(common_rows[0]) if common_rows else [],
    )
    write_tsv(
        args.output_dir / "common_cohort_cases.tsv",
        common_cases,
        list(common_cases[0]) if common_cases else [],
    )
    report = {
        "design": {
            "mode": "smoke" if args.smoke else "full",
            "semantic_mode": args.semantic_mode,
            "term_aliases": not args.disable_term_aliases,
            "seeds": args.seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "arms": [arm.__dict__ for arm in ARMS],
            "comparisons": [
                {
                    "comparison_id": comparison_id,
                    "label": label,
                    "treatment": treatment,
                    "control": control,
                }
                for comparison_id, label, treatment, control in COMPARISONS
            ],
            "random_control": (
                "Channel-stratified bipartite double-edge swaps preserve every HPO "
                "degree and every CL-GO mechanism degree."
            ),
            "paired_cohort": (
                "Only cases with identical task, query ID, and target set in both arms."
            ),
            "evaluation_protocol": {
                "task_1_exact_retrieval": ["expert_exact_present"],
                "task_2_ancestor_retrieval": [
                    "expert_hp_broader",
                    "counterfactual_exact_removed",
                    "synthetic_hpo_cutout",
                ],
                "note": (
                    "The constructed combined metric is supplemental and is not "
                    "used as a replacement for the two retrieval tasks."
                ),
            },
        },
        "arm_metric_means": mean_rows,
        "common_cohort_metric_means": common_mean_rows,
        "two_task_metric_means": two_task_rows,
        "paired_comparisons": comparison_rows,
        "run_manifests": {
            f"{seed}:{arm_id}": manifest
            for (seed, arm_id), manifest in sorted(manifests.items())
        },
    }
    write_json(args.output_dir / "experiment_report.json", report)
    (args.output_dir / "index.html").write_text(
        render_html(args, mean_rows, comparison_rows), encoding="utf-8"
    )
    write_tsv(
        args.output_dir / "gold_standard_recall_unmatched.tsv",
        mean_rows,
        list(mean_rows[0]) if mean_rows else [],
    )
    (args.output_dir / "gold_standard_recall_unmatched.html").write_text(
        render_recall_html(args, mean_rows), encoding="utf-8"
    )
    write_tsv(
        args.output_dir / "common_cohort_recall.tsv",
        common_mean_rows,
        list(common_mean_rows[0]) if common_mean_rows else [],
    )
    (args.output_dir / "common_cohort_recall.html").write_text(
        render_recall_html(
            args,
            common_mean_rows,
            title="四组实验统一样本金标准召回率",
            subtitle=(
                "四组使用完全相同的 query 与金标准目标集合，直接比较实际召回率。"
            ),
            note=(
                "统一队列按任务、query ID 和目标集合的四组严格交集生成。"
                "因此每张表中四组查询数完全一致，旧链接与 v4 的覆盖差异不会"
                "改变测试样本。"
            ),
        ),
        encoding="utf-8",
    )
    write_tsv(
        args.output_dir / "strict_intersection_recall.tsv",
        two_task_rows,
        list(two_task_rows[0]) if two_task_rows else [],
    )
    (args.output_dir / "strict_intersection_recall.html").write_text(
        render_two_task_recall_html(args, common_mean_rows), encoding="utf-8"
    )
    if not (args.output_dir / "fixed_cohort_recall.tsv").is_file():
        write_tsv(
            args.output_dir / "gold_standard_recall.tsv",
            two_task_rows,
            list(two_task_rows[0]) if two_task_rows else [],
        )
        (args.output_dir / "gold_standard_recall.html").write_text(
            render_two_task_recall_html(args, common_mean_rows), encoding="utf-8"
        )
    write_tsv(
        args.output_dir / "two_task_recall.tsv",
        two_task_rows,
        list(two_task_rows[0]) if two_task_rows else [],
    )
    (args.output_dir / "two_task_recall.html").write_text(
        render_two_task_recall_html(args, common_mean_rows), encoding="utf-8"
    )
    print(f"Wrote experiment report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
