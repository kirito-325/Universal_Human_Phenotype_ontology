#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ARMS = (
    ("semantic_only", "纯语义"),
    ("structure_only", "纯结构/机制"),
    ("late_fusion_constant", "晚融合 + 常量结构"),
    ("late_fusion_poincare", "晚融合 + 真实 Poincare"),
    ("late_fusion_permuted", "晚融合 + 置换结构"),
)
TASKS = (
    ("expert_exact", "专家金标准：Exact", {"exact"}),
    ("expert_broader", "专家金标准：Broader", {"hp_broader"}),
    (
        "constructed_combined",
        "构建金标准：综合结果",
        {"counterfactual_exact", "synthetic_hpo_cutout"},
    ),
)
CONTROLS = ("semantic_only", "late_fusion_constant", "late_fusion_permuted")


def read_cases(path: Path) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            result[(row["task"], row["query_id"], row["targets"])] = int(
                row["best_rank"] or 0
            )
    return result


def value(rank: int, metric: str) -> float:
    if metric == "mrr":
        return 1.0 / rank if rank > 0 else 0.0
    cutoff = int(metric.removeprefix("recall_at_"))
    return float(0 < rank <= cutoff)


def metrics(ranks: Sequence[int]) -> dict[str, float]:
    return {
        metric: float(np.mean([value(rank, metric) for rank in ranks]))
        for metric in ("mrr", "recall_at_1", "recall_at_10", "recall_at_50")
    }


def bootstrap_interval(
    differences: np.ndarray, seed: int, replicates: int
) -> tuple[float, float, float]:
    randomizer = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        indices = randomizer.integers(0, len(differences), size=(size, len(differences)))
        samples.append(differences[indices].mean(axis=1))
    bootstrap = np.concatenate(samples)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(lower), float(upper), float(np.mean(bootstrap > 0.0))


def exact_mcnemar_p(real: np.ndarray, control: np.ndarray) -> float:
    real_only = int(np.sum((real == 1.0) & (control == 0.0)))
    control_only = int(np.sum((real == 0.0) & (control == 1.0)))
    discordant = real_only + control_only
    if discordant == 0:
        return 1.0
    tail = min(real_only, control_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_html(
    metric_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
) -> str:
    arm_order = {arm_id: index for index, (arm_id, _label) in enumerate(ARMS)}
    sections: list[str] = []
    for task_id, task_label, _sources in TASKS:
        selected = sorted(
            (row for row in metric_rows if row["task"] == task_id),
            key=lambda row: arm_order[str(row["arm_id"])],
        )
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td><td>{int(row['queries'])}</td>"
            f"<td>{100 * float(row['recall_at_1']):.2f}%</td>"
            f"<td>{100 * float(row['recall_at_10']):.2f}%</td>"
            f"<td>{100 * float(row['recall_at_50']):.2f}%</td>"
            f"<td>{float(row['mrr']):.4f}</td></tr>"
            for row in selected
        )
        sections.append(
            f"<section><h2>{html.escape(task_label)}</h2><div class='table-wrap'><table>"
            "<thead><tr><th>实验组</th><th>样本数</th><th>Recall@1</th>"
            "<th>Recall@10</th><th>Recall@50</th><th>MRR</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div></section>"
        )
    comparison_body = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['task_label']))}</td>"
        f"<td>{html.escape(str(row['control_label']))}</td>"
        f"<td>{html.escape(str(row['metric']))}</td>"
        f"<td>{100 * float(row['delta']):+.2f} pp</td>"
        f"<td>[{100 * float(row['ci_lower']):+.2f}, {100 * float(row['ci_upper']):+.2f}]</td>"
        f"<td>{float(row['paired_p_value']):.4f}</td></tr>"
        for row in comparison_rows
        if row["metric"] in {"recall_at_1", "recall_at_10", "recall_at_50"}
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPO-HPO v2 正式评估</title>
<style>:root{{--ink:#18201d;--muted:#64706b;--line:#d8dfdb;--wash:#f4f6f5}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}header{{padding:30px max(22px,calc((100vw - 1100px)/2));background:var(--wash);border-bottom:1px solid var(--line)}}h1{{margin:0 0 6px;font-size:25px;letter-spacing:0}}header p{{margin:0;color:var(--muted)}}main{{max-width:1100px;margin:auto;padding:26px 22px 50px}}section{{margin:0 0 30px}}h2{{font-size:17px;letter-spacing:0}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:var(--wash)}}tr:last-child td{{border-bottom:0}}</style></head><body>
<header><h1>MPO-HPO Mapping v2 正式全量评估</h1><p>80 epoch、SapBERT 64d、相同训练/测试划分；配对区间为查询级 bootstrap 95% CI。</p></header><main>{''.join(sections)}
<section><h2>真实 Poincare 相对对照的配对差值</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>对照组</th><th>指标</th><th>差值</th><th>95% CI</th><th>配对 p</th></tr></thead><tbody>{comparison_body}</tbody></table></div></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize formal v2 paired evaluation.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    cases = {
        arm_id: read_cases(args.result_dir / "runs" / arm_id / "evaluation_cases.tsv")
        for arm_id, _label in ARMS
    }
    keys = list(cases["late_fusion_poincare"])
    if any(set(rows) != set(keys) for rows in cases.values()):
        raise RuntimeError("Evaluation cohorts differ between arms")

    metric_rows: list[dict[str, object]] = []
    selected_keys: dict[str, list[tuple[str, str, str]]] = {}
    for task_id, task_label, sources in TASKS:
        task_keys = [key for key in keys if key[0] in sources]
        selected_keys[task_id] = task_keys
        for arm_id, label in ARMS:
            ranks = [cases[arm_id][key] for key in task_keys]
            metric_rows.append(
                {
                    "task": task_id,
                    "task_label": task_label,
                    "arm_id": arm_id,
                    "label": label,
                    "queries": len(ranks),
                    **metrics(ranks),
                }
            )

    arm_labels = dict(ARMS)
    comparison_rows: list[dict[str, object]] = []
    for task_index, (task_id, task_label, _sources) in enumerate(TASKS):
        task_keys = selected_keys[task_id]
        for control_index, control_id in enumerate(CONTROLS):
            for metric_index, metric in enumerate(
                ("mrr", "recall_at_1", "recall_at_10", "recall_at_50")
            ):
                real = np.asarray(
                    [value(cases["late_fusion_poincare"][key], metric) for key in task_keys]
                )
                control = np.asarray(
                    [value(cases[control_id][key], metric) for key in task_keys]
                )
                differences = real - control
                lower, upper, probability = bootstrap_interval(
                    differences,
                    args.seed + task_index * 100 + control_index * 10 + metric_index,
                    args.bootstrap_replicates,
                )
                comparison_rows.append(
                    {
                        "task": task_id,
                        "task_label": task_label,
                        "control_arm": control_id,
                        "control_label": arm_labels[control_id],
                        "metric": metric,
                        "queries": len(task_keys),
                        "delta": float(np.mean(differences)),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_probability_real_better": probability,
                        "paired_p_value": (
                            exact_mcnemar_p(real, control) if metric != "mrr" else ""
                        ),
                    }
                )

    write_tsv(args.result_dir / "formal_metrics.tsv", metric_rows)
    write_tsv(args.result_dir / "paired_comparisons.tsv", comparison_rows)
    with (args.result_dir / "formal_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metrics": metric_rows, "paired_comparisons": comparison_rows},
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    (args.result_dir / "formal_summary.html").write_text(
        render_html(metric_rows, comparison_rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
