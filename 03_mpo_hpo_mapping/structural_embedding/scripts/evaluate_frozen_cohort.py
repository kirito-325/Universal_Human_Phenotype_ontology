#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from analyze_intermediate_ablation import (  # noqa: E402
    load_saved_profile,
    read_mechanism_edges,
)
from mpo_hpo_mapping.data import (  # noqa: E402
    InputPaths,
    MappingExample,
    build_parent_examples,
    file_fingerprint,
    load_expert_examples,
    load_hpo_ontology,
    write_json,
    write_tsv,
)
from mpo_hpo_mapping.model import DualHeadEncoder  # noqa: E402
from mpo_hpo_mapping.pipeline import RunConfig, _best_rank, _retrieve_ranked  # noqa: E402
from mpo_hpo_mapping.retrieval import (  # noqa: E402
    AncestorRankWeights,
    DualHeadRetriever,
)
from mpo_hpo_mapping.semantic import load_obo_term_text  # noqa: E402
from run_embedding_link_experiments import ARMS, METRICS, metric_values  # noqa: E402


TASKS = (
    "expert_exact_present",
    "expert_hp_broader",
    "counterfactual_exact_removed",
    "synthetic_hpo_cutout",
)
TASK_LAYOUT = (
    (
        "任务一：等价 HPO 召回",
        "当等价 HPO 存在时，检验正确等价术语的排名。",
        (("expert_exact_present", "专家金标准：精确 HPO"),),
    ),
    (
        "任务二：HPO 上位类召回",
        "当等价 HPO 不可用时，检验正确 HPO 上位类的排名。",
        (
            ("expert_hp_broader", "专家金标准：更宽泛 HPO"),
            ("counterfactual_exact_removed", "构建金标准：真实 exact 遮蔽"),
            ("synthetic_hpo_cutout", "构建金标准：HPO 子节点切除"),
        ),
    ),
)


@dataclass(frozen=True)
class FrozenCase:
    task: str
    query_id: str
    query_source: str
    query_profile_id: str
    targets: tuple[str, ...]
    excluded: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all embedding/link arms on one frozen leakage-safe cohort."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=(
            MODULE_DIR
            / "results/embedding_link_experiment_full_deterministic_20260803"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def profile_ids(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as archive:
        return set(archive["ids"].astype(str).tolist())


def group_real_cases(
    examples: Sequence[MappingExample], relation: str, task: str
) -> list[FrozenCase]:
    targets: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        if row.split == "test" and row.relation == relation:
            targets[row.mp_id].add(row.hp_id)
    return [
        FrozenCase(task, mp_id, "mp", mp_id, tuple(sorted(hp_ids)))
        for mp_id, hp_ids in sorted(targets.items())
    ]


def group_masked_cases(
    examples: Sequence[MappingExample], source: str, task: str
) -> list[FrozenCase]:
    grouped: dict[tuple[str, str], list[MappingExample]] = defaultdict(list)
    for row in examples:
        if row.split == "test" and row.source == source:
            grouped[(row.mp_id, row.masked_hp_ids)].append(row)
    result: list[FrozenCase] = []
    for (query_id, masked), rows in sorted(grouped.items()):
        first = rows[0]
        query_source = "hp" if first.query_profile_source == "hp" else "mp"
        profile_id = first.exact_hp_id if query_source == "hp" else first.mp_id
        result.append(
            FrozenCase(
                task,
                query_id,
                query_source,
                profile_id,
                tuple(sorted({row.hp_id for row in rows})),
                tuple(sorted(filter(None, masked.split(";")))),
            )
        )
    return result


def build_frozen_cohort(
    experiment_dir: Path, seed: int
) -> tuple[list[FrozenCase], dict[str, object]]:
    run_root = experiment_dir / "runs" / f"seed_{seed}"
    hp_sets = {
        arm.arm_id: profile_ids(
            run_root / arm.arm_id / "data/processed/hpo_profiles.npz"
        )
        for arm in ARMS
    }
    mp_sets = {
        arm.arm_id: profile_ids(
            run_root / arm.arm_id / "data/processed/mpo_profiles.npz"
        )
        for arm in ARMS
    }
    hp_union = set.union(*hp_sets.values())
    hp_intersection = set.intersection(*hp_sets.values())
    mp_intersection = set.intersection(*mp_sets.values())
    paths = InputPaths()
    ontology = load_hpo_ontology(paths.hpo_edges)
    expert = load_expert_examples(
        paths.expert_mappings,
        mp_intersection,
        hp_union,
        min_confidence=0.9,
        seed=seed,
    )
    parent = build_parent_examples(expert, ontology, hp_union, seed=seed)
    cases = [
        *group_real_cases(expert, "exact", "expert_exact_present"),
        *group_real_cases(expert, "hp_broader", "expert_hp_broader"),
        *group_masked_cases(
            parent, "counterfactual_exact", "counterfactual_exact_removed"
        ),
        *group_masked_cases(
            parent, "synthetic_hpo_cutout", "synthetic_hpo_cutout"
        ),
    ]

    fitted_query_ids = {
        row.mp_id for row in [*expert, *parent] if row.split != "test"
    }
    test_query_ids = {case.query_id for case in cases}
    fitted_pairs = {
        (row.source, row.mp_id, row.hp_id)
        for row in [*expert, *parent]
        if row.split != "test"
    }
    test_pairs = {
        (row.source, row.mp_id, row.hp_id)
        for row in [*expert, *parent]
        if row.split == "test"
    }
    audit = {
        "cohort_policy": (
            "Gold labels are frozen once from expert test mappings and HPO DAG edges. "
            "Arm profile IDs contribute only to one symmetric union eligibility universe; "
            "arm-specific links, weights, and ranking outcomes never select gold targets. "
            "Missing queries or targets score rank 0 instead of being removed."
        ),
        "hp_union_terms": len(hp_union),
        "hp_intersection_terms": len(hp_intersection),
        "mp_intersection_terms": len(mp_intersection),
        "arm_hpo_terms": {arm_id: len(ids) for arm_id, ids in hp_sets.items()},
        "task_queries": {
            task: sum(case.task == task for case in cases) for task in TASKS
        },
        "fitted_test_query_overlap": len(fitted_query_ids & test_query_ids),
        "fitted_test_pair_overlap": len(fitted_pairs & test_pairs),
        "expert_source": file_fingerprint(paths.expert_mappings),
        "hpo_dag_source": file_fingerprint(paths.hpo_edges),
    }
    if audit["fitted_test_query_overlap"] or audit["fitted_test_pair_overlap"]:
        raise RuntimeError(f"Leakage audit failed: {audit}")
    return sorted(cases, key=lambda row: (row.task, row.query_id)), audit


def load_retriever(
    run_dir: Path,
) -> tuple[DualHeadRetriever, object, object, RunConfig, AncestorRankWeights]:
    manifest = json.loads((run_dir / "training_manifest.json").read_text())
    hp_edges, mp_edges = read_mechanism_edges(
        run_dir / "data/processed/term_mechanism_edges.tsv"
    )
    hp_ids = set(hp_edges)
    mp_ids = set(mp_edges)
    paths = InputPaths()
    texts = {
        **load_obo_term_text(paths.hpo_obo, hp_ids),
        **load_obo_term_text(paths.mpo_obo, mp_ids),
    }
    aliases = {term_id: text.normalized_aliases() for term_id, text in texts.items()}
    hp_profiles = load_saved_profile(
        run_dir / "data/processed/hpo_profiles.npz", aliases, hp_edges
    )
    mp_profiles = load_saved_profile(
        run_dir / "data/processed/mpo_profiles.npz", aliases, mp_edges
    )
    checkpoint = torch.load(
        run_dir / "model/dual_head_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = DualHeadEncoder(
        int(checkpoint["input_dimension"]),
        int(checkpoint["hidden_dimension"]),
        int(checkpoint["output_dimension"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    relation = joblib.load(run_dir / "model/relation_calibrator.joblib")
    gate = joblib.load(run_dir / "model/availability_gate.joblib")
    config_data = dict(manifest["config"])
    config_data["output_dir"] = Path(config_data["output_dir"])
    config = RunConfig(**config_data)
    retriever = DualHeadRetriever(
        model,
        hp_profiles,
        mp_profiles,
        relation,
        config.ivf_clusters,
        config.seed,
    )
    ancestor_weights = AncestorRankWeights(**manifest["training"]["ancestor_rank_weights"])
    ontology = load_hpo_ontology(paths.hpo_edges)
    return retriever, gate, ontology, config, ancestor_weights


def evaluate_arm(
    run_dir: Path,
    arm_id: str,
    arm_label: str,
    cases: Sequence[FrozenCase],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    retriever, gate, ontology, config, ancestor_weights = load_retriever(run_dir)
    cache: dict[tuple[str, str, tuple[str, ...]], tuple[list[dict[str, object]], float]] = {}
    case_rows: list[dict[str, object]] = []
    ranks_by_task: dict[str, list[int]] = {task: [] for task in TASKS}
    for case in cases:
        profile_index = (
            retriever.hp_index.get(case.query_profile_id)
            if case.query_source == "hp"
            else retriever.mp_index.get(case.query_profile_id)
        )
        query_available = profile_index is not None
        target_count = sum(target in retriever.hp_index for target in case.targets)
        probability = 0.0
        rank = 0
        if profile_index is not None:
            cache_key = (case.query_source, case.query_profile_id, case.excluded)
            if cache_key not in cache:
                cache[cache_key] = _retrieve_ranked(
                    retriever,
                    gate,
                    ontology,
                    config,
                    case.query_source,
                    profile_index,
                    set(case.excluded),
                    ancestor_weights,
                )
            candidates, probability = cache[cache_key]
            rank = _best_rank(candidates, set(case.targets))
        ranks_by_task[case.task].append(rank)
        case_rows.append(
            {
                "arm": arm_id,
                "arm_label": arm_label,
                "task": case.task,
                "query_id": case.query_id,
                "targets": ";".join(case.targets),
                "excluded": ";".join(case.excluded),
                "query_available": int(query_available),
                "available_targets": target_count,
                "best_rank": rank,
                "gate_probability": f"{probability:.8g}",
            }
        )
    metric_rows: list[dict[str, object]] = []
    for task in TASKS:
        task_cases = [row for row in case_rows if row["task"] == task]
        metrics = metric_values(ranks_by_task[task])
        metric_rows.append(
            {
                "arm": arm_id,
                "arm_label": arm_label,
                "task": task,
                "queries": len(task_cases),
                **metrics,
                "query_coverage": sum(int(row["query_available"]) for row in task_cases)
                / max(1, len(task_cases)),
                "target_coverage": sum(int(row["available_targets"]) > 0 for row in task_cases)
                / max(1, len(task_cases)),
            }
        )
    return metric_rows, case_rows


def render_html(rows: Sequence[Mapping[str, object]], audit: Mapping[str, object]) -> str:
    by_key = {(str(row["arm"]), str(row["task"])): row for row in rows}
    sections: list[str] = []
    for task_label, description, gold_sets in TASK_LAYOUT:
        tables: list[str] = []
        for task, gold_label in gold_sets:
            task_rows = [by_key[(arm.arm_id, task)] for arm in ARMS]
            best = {
                metric: max(float(row[metric]) for row in task_rows)
                for metric in METRICS
            }
            body = "".join(
                "<tr>"
                f"<td>{html.escape(arm.label)}</td>"
                f"<td>{int(row['queries']):,}</td>"
                + "".join(
                    f"<td class=\"{'best' if abs(float(row[metric]) - best[metric]) < 1e-12 else ''}\">{100 * float(row[metric]):.2f}%</td>"
                    for metric in ("recall_at_1", "recall_at_10", "recall_at_50")
                )
                + "</tr>"
                for arm, row in zip(ARMS, task_rows)
            )
            tables.append(
                f"<h3>{html.escape(gold_label)}</h3>"
                '<div class="table-wrap"><table><thead><tr><th>实验组</th>'
                "<th>样本数</th><th>Recall@1</th><th>Recall@10</th>"
                f"<th>Recall@50</th></tr></thead><tbody>{body}</tbody></table></div>"
            )
        sections.append(
            f"<section><h2>{html.escape(task_label)}</h2>"
            f"<p class=\"task-note\">{html.escape(description)}</p>{''.join(tables)}</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四组实验冻结测试集评估</title>
<style>
:root {{ color-scheme:light; --ink:#17201d; --muted:#66716d; --line:#d9dfdc; --wash:#f4f6f5; --green:#08745a; --green-wash:#e8f4ef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ padding:30px max(22px,calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); background:var(--wash); }}
h1 {{ margin:0 0 7px; font-size:26px; letter-spacing:0; }}
header p,.note,.task-note {{ margin:0; color:var(--muted); }}
main {{ max-width:1080px; margin:0 auto; padding:26px 22px 50px; }}
.summary {{ display:flex; gap:28px; flex-wrap:wrap; padding-bottom:20px; margin-bottom:28px; border-bottom:1px solid var(--line); }}
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
.note {{ max-width:920px; }}
</style>
</head>
<body>
<header><h1>四组实验冻结测试集评估</h1><p>四组使用同一批 query 和同一套答案；缺失 query 或答案记为未召回。</p></header>
<main>
<div class="summary"><div><strong>{sum(int(value) for value in audit['task_queries'].values()):,}</strong><span>冻结测试样本</span></div><div><strong>{int(audit['hp_union_terms']):,}</strong><span>HPO 并集候选</span></div><div><strong>0</strong><span>拟合侧/测试 query 重叠</span></div></div>
{''.join(sections)}
<p class="note">金标准关系仅来自专家测试划分和 HPO DAG。四组 profile ID 只用于构造一次对称的并集资格范围；任何实验组自己的链接边、权重和排序结果都不参与答案选择。缺失 query 或答案统一记为未召回。</p>
</main>
</body></html>
"""


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    torch.use_deterministic_algorithms(True)
    cases, audit = build_frozen_cohort(experiment_dir, args.seed)
    print(f"Frozen cohort sizes: {audit['task_queries']}", flush=True)
    write_tsv(
        experiment_dir / "frozen_cohort_cases.tsv",
        [
            {
                "task": case.task,
                "query_id": case.query_id,
                "query_source": case.query_source,
                "query_profile_id": case.query_profile_id,
                "targets": ";".join(case.targets),
                "excluded": ";".join(case.excluded),
            }
            for case in cases
        ],
        ["task", "query_id", "query_source", "query_profile_id", "targets", "excluded"],
    )
    metric_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    run_root = experiment_dir / "runs" / f"seed_{args.seed}"
    for arm in ARMS:
        print(f"Evaluate frozen cohort: {arm.label}", flush=True)
        arm_metrics, arm_cases = evaluate_arm(
            run_root / arm.arm_id, arm.arm_id, arm.label, cases
        )
        metric_rows.extend(arm_metrics)
        case_rows.extend(arm_cases)
    write_tsv(
        experiment_dir / "fixed_cohort_recall.tsv",
        metric_rows,
        list(metric_rows[0]),
    )
    write_tsv(
        experiment_dir / "fixed_cohort_arm_cases.tsv",
        case_rows,
        list(case_rows[0]),
    )
    report = {
        "design": audit,
        "metrics": metric_rows,
        "arms": [arm.__dict__ for arm in ARMS],
    }
    write_json(experiment_dir / "fixed_cohort_report.json", report)
    page = render_html(metric_rows, audit)
    (experiment_dir / "fixed_cohort_recall.html").write_text(page, encoding="utf-8")
    (experiment_dir / "gold_standard_recall.html").write_text(page, encoding="utf-8")
    write_tsv(
        experiment_dir / "gold_standard_recall.tsv",
        metric_rows,
        list(metric_rows[0]),
    )
    print(f"Wrote frozen-cohort evaluation to {experiment_dir}", flush=True)


if __name__ == "__main__":
    main()
