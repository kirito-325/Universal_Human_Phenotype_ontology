#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
    CASE_TASK_TO_SUMMARY,
    METRICS,
    TASK_LABELS,
    bootstrap_interval,
    metric_values,
)


TASKS = (
    "expert_exact_present",
    "expert_hp_broader",
    "counterfactual_exact_removed",
    "synthetic_hpo_cutout",
)


@dataclass(frozen=True)
class ControlArm:
    arm_id: str
    label: str
    hierarchy_mode: str
    source_arm_id: str | None = None


ARMS = (
    ControlArm(
        "constant_structure_baseline_links",
        "常量 CL/GO + 旧链接",
        "constant",
        "constant_structure_baseline_links",
    ),
    ControlArm(
        "permuted_structure_baseline_links",
        "随机置换 CL/GO + 旧链接",
        "permuted",
    ),
    ControlArm(
        "poincare_structure_baseline_links",
        "真实 Poincare CL/GO + 旧链接",
        "poincare",
        "poincare_structure_baseline_links",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run constant, permuted, and real CL/GO structure controls."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            MODULE_DIR
            / "results/structure_embedding_control_full_deterministic_v2_20260803"
        ),
    )
    parser.add_argument(
        "--source-experiment",
        type=Path,
        default=(
            MODULE_DIR
            / "results/embedding_link_experiment_full_deterministic_20260803"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_config(
    output_dir: Path,
    arm: ControlArm,
    seed: int,
    hierarchy_reference: Path | None = None,
) -> RunConfig:
    return RunConfig(
        output_dir=output_dir,
        general_top_k=8,
        general_weight=0.15,
        hierarchy_dimension=16,
        hierarchy_epochs=30,
        hierarchy_mode=arm.hierarchy_mode,
        hierarchy_reference=str(hierarchy_reference or ""),
        hpo_link_randomization="none",
        max_prototypes=4,
        semantic_mode="auto",
        semantic_dimension=64,
        semantic_batch_size=64,
        semantic_weight=1.0,
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
        seed=seed,
    )


def run_directories(args: argparse.Namespace) -> dict[str, Path]:
    source_root = args.source_experiment / "runs" / f"seed_{args.seed}"
    local_root = args.output_dir / "runs" / f"seed_{args.seed}"
    result: dict[str, Path] = {}
    for arm in ARMS:
        if arm.source_arm_id:
            result[arm.arm_id] = source_root / arm.source_arm_id
        else:
            result[arm.arm_id] = local_root / arm.arm_id
    return result


def read_cases(path: Path) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            task = CASE_TASK_TO_SUMMARY[row["task"]]
            result[(task, row["query_id"], row["targets"])] = int(
                row["best_rank"] or 0
            )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_multiset(vectors: np.ndarray) -> list[str]:
    return sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in vectors)


def validate_controls(
    run_dirs: Mapping[str, Path],
) -> dict[str, object]:
    manifests = {
        arm.arm_id: json.loads(
            (run_dirs[arm.arm_id] / "training_manifest.json").read_text()
        )
        for arm in ARMS
    }
    files = (
        "data/processed/expert_mapping_examples.tsv",
        "data/processed/parent_training_examples.tsv",
        "data/processed/term_mechanism_edges.tsv",
        "data/processed/semantic_embeddings.npz",
    )
    file_hashes = {
        name: {arm.arm_id: sha256(run_dirs[arm.arm_id] / name) for arm in ARMS}
        for name in files
    }
    for name, hashes in file_hashes.items():
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"Control input mismatch for {name}: {hashes}")

    cases = {
        arm.arm_id: read_cases(run_dirs[arm.arm_id] / "evaluation_cases.tsv")
        for arm in ARMS
    }
    cohort_sets = {arm_id: set(rows) for arm_id, rows in cases.items()}
    if len({frozenset(keys) for keys in cohort_sets.values()}) != 1:
        raise RuntimeError("Evaluation query/target cohorts differ between controls")

    real_dir = run_dirs["poincare_structure_baseline_links"]
    permuted_dir = run_dirs["permuted_structure_baseline_links"]
    with np.load(
        real_dir / "data/processed/hierarchy_embeddings.npz", allow_pickle=False
    ) as real, np.load(
        permuted_dir / "data/processed/hierarchy_embeddings.npz", allow_pickle=False
    ) as permuted:
        permutation_checks = {}
        for prefix in ("cl", "go"):
            real_ids = real[f"{prefix}_ids"].astype(str)
            permuted_ids = permuted[f"{prefix}_ids"].astype(str)
            real_vectors = real[f"{prefix}_vectors"].astype(np.float32)
            permuted_vectors = permuted[f"{prefix}_vectors"].astype(np.float32)
            ids_equal = np.array_equal(real_ids, permuted_ids)
            multiset_equal = row_multiset(real_vectors) == row_multiset(permuted_vectors)
            assignments_changed = int(
                np.sum(np.any(real_vectors != permuted_vectors, axis=1))
            )
            if not ids_equal or not multiset_equal or assignments_changed == 0:
                raise RuntimeError(f"Invalid {prefix.upper()} permutation control")
            permutation_checks[prefix] = {
                "nodes": len(real_ids),
                "ids_equal": ids_equal,
                "vector_multiset_equal": multiset_equal,
                "node_assignments_changed": assignments_changed,
            }

    controlled_config_keys = (
        "general_top_k",
        "general_weight",
        "hierarchy_dimension",
        "hierarchy_epochs",
        "max_prototypes",
        "semantic_mode",
        "semantic_dimension",
        "semantic_weight",
        "use_term_aliases",
        "hidden_dimension",
        "output_dimension",
        "training_epochs",
        "batch_size",
        "min_mapping_confidence",
        "ivf_clusters",
        "nprobe",
        "retrieve_k",
        "seed",
    )
    config_values = {
        key: {manifests[arm.arm_id]["config"][key] for arm in ARMS}
        for key in controlled_config_keys
    }
    mismatched = {key: values for key, values in config_values.items() if len(values) != 1}
    if mismatched:
        raise RuntimeError(f"Controlled hyperparameters differ: {mismatched}")

    return {
        "training_and_input_hashes_identical": True,
        "evaluation_cohort_identical": True,
        "queries_by_task": {
            task: sum(key[0] == task for key in next(iter(cases.values())))
            for task in TASKS
        },
        "file_hashes": file_hashes,
        "permutation_checks": permutation_checks,
        "controlled_hyperparameters": {
            key: next(iter(values)) for key, values in config_values.items()
        },
    }


def metric_rows(run_dirs: Mapping[str, Path]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for arm in ARMS:
        cases = read_cases(run_dirs[arm.arm_id] / "evaluation_cases.tsv")
        for task in TASKS:
            ranks = [rank for key, rank in cases.items() if key[0] == task]
            result.append(
                {
                    "arm": arm.arm_id,
                    "arm_label": arm.label,
                    "task": task,
                    "queries": len(ranks),
                    **metric_values(ranks),
                }
            )
    return result


def comparison_rows(
    rows: Sequence[Mapping[str, object]],
    run_dirs: Mapping[str, Path],
    bootstrap_samples: int = 2000,
) -> list[dict[str, object]]:
    by_key = {(str(row["arm"]), str(row["task"])): row for row in rows}
    cases = {
        arm.arm_id: read_cases(run_dirs[arm.arm_id] / "evaluation_cases.tsv")
        for arm in ARMS
    }
    comparisons = (
        (
            "real_minus_constant",
            "真实 Poincare - 常量",
            "poincare_structure_baseline_links",
            "constant_structure_baseline_links",
        ),
        (
            "real_minus_permuted",
            "真实 Poincare - 随机置换",
            "poincare_structure_baseline_links",
            "permuted_structure_baseline_links",
        ),
        (
            "permuted_minus_constant",
            "随机置换 - 常量",
            "permuted_structure_baseline_links",
            "constant_structure_baseline_links",
        ),
    )
    result: list[dict[str, object]] = []
    for comparison, label, treatment, control in comparisons:
        for task in TASKS:
            common = sorted(
                key
                for key in set(cases[treatment]) & set(cases[control])
                if key[0] == task
            )
            observations = [
                (cases[control][key], cases[treatment][key]) for key in common
            ]
            for metric in METRICS:
                left = float(by_key[(treatment, task)][metric])
                right = float(by_key[(control, task)][metric])
                low, high = bootstrap_interval(
                    observations,
                    metric,
                    bootstrap_samples,
                    f"{comparison}|{task}|{metric}|structure-controls",
                )
                result.append(
                    {
                        "comparison": comparison,
                        "comparison_label": label,
                        "task": task,
                        "metric": metric,
                        "paired_queries": len(observations),
                        "treatment": left,
                        "control": right,
                        "delta": left - right,
                        "ci_95_low": low,
                        "ci_95_high": high,
                    }
                )
    return result


def render_html(rows: Sequence[Mapping[str, object]], audit: Mapping[str, object]) -> str:
    by_key = {(str(row["arm"]), str(row["task"])): row for row in rows}
    sections: list[str] = []
    for task in TASKS:
        task_rows = [by_key[(arm.arm_id, task)] for arm in ARMS]
        best = {
            metric: max(float(row[metric]) for row in task_rows) for metric in METRICS
        }
        body = "".join(
            "<tr>"
            f"<td>{html.escape(arm.label)}</td><td>{int(row['queries']):,}</td>"
            + "".join(
                f"<td class=\"{'best' if abs(float(row[metric]) - best[metric]) < 1e-12 else ''}\">{100 * float(row[metric]):.2f}%</td>"
                for metric in ("recall_at_1", "recall_at_10", "recall_at_50")
            )
            + f"<td>{float(row['mrr']):.4f}</td></tr>"
            for arm, row in zip(ARMS, task_rows)
        )
        sections.append(
            f"<section><h2>{html.escape(TASK_LABELS[task])}</h2>"
            '<div class="table-wrap"><table><thead><tr><th>实验组</th>'
            "<th>样本数</th><th>Recall@1</th><th>Recall@10</th>"
            f"<th>Recall@50</th><th>MRR</th></tr></thead><tbody>{body}"
            "</tbody></table></div></section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CL/GO 结构嵌入严格对照实验</title>
<style>
:root {{ --ink:#17201d; --muted:#66716d; --line:#d9dfdc; --wash:#f4f6f5; --green:#08745a; --green-wash:#e8f4ef; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ padding:30px max(22px,calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); background:var(--wash); }} h1 {{ margin:0 0 7px; font-size:26px; letter-spacing:0; }} header p,.note {{ margin:0; color:var(--muted); }}
main {{ max-width:1080px; margin:0 auto; padding:26px 22px 50px; }} .summary {{ display:flex; gap:28px; flex-wrap:wrap; padding-bottom:20px; margin-bottom:28px; border-bottom:1px solid var(--line); }} .summary strong {{ display:block; font-size:20px; }} .summary span {{ color:var(--muted); }}
section {{ margin:0 0 30px; }} h2 {{ margin:0 0 9px; font-size:17px; letter-spacing:0; }} .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:6px; }} table {{ width:100%; border-collapse:collapse; white-space:nowrap; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; }} th {{ background:var(--wash); color:#43504b; font-size:12px; }} th:first-child,td:first-child {{ text-align:left; }} tr:last-child td {{ border-bottom:0; }} .best {{ color:var(--green); background:var(--green-wash); font-weight:700; }}
</style></head><body>
<header><h1>CL/GO 结构嵌入严格对照实验</h1><p>三组仅改变 CL/GO 结构表示；旧链接、训练集、测试集和下游训练配置完全一致。</p></header><main>
<div class="summary"><div><strong>3</strong><span>结构对照组</span></div><div><strong>{sum(int(value) for value in audit['queries_by_task'].values()):,}</strong><span>测试 episode</span></div><div><strong>identical</strong><span>训练与测试数据</span></div></div>
{''.join(sections)}
<p class="note">随机组将真实 Poincare 向量在节点间确定性置换，保留维度、数值分布和范数，只破坏向量与本体节点的对应关系。</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source_experiment = args.source_experiment.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = run_directories(args)
    random_arm = next(arm for arm in ARMS if arm.hierarchy_mode == "permuted")
    random_dir = run_dirs[random_arm.arm_id]
    if args.resume and (random_dir / "training_manifest.json").is_file():
        print(f"Reuse {random_arm.label}", flush=True)
    else:
        print(f"Run {random_arm.label}", flush=True)
        real_hierarchy = (
            run_dirs["poincare_structure_baseline_links"]
            / "data/processed/hierarchy_embeddings.npz"
        )
        run_pipeline(
            InputPaths(),
            make_config(random_dir, random_arm, args.seed, real_hierarchy),
        )
    audit = validate_controls(run_dirs)
    rows = metric_rows(run_dirs)
    comparisons = comparison_rows(rows, run_dirs)
    report = {
        "design": {
            "claim": (
                "Tests structural hierarchy value conditional on fixed baseline links; "
                "does not independently prove link value without a no-link/shuffled-link arm."
            ),
            "arms": [arm.__dict__ for arm in ARMS],
            "seed": args.seed,
            "bootstrap_samples": 2000,
            "run_directories": {
                arm_id: str(path) for arm_id, path in run_dirs.items()
            },
        },
        "audit": audit,
        "metrics": rows,
        "comparisons": comparisons,
    }
    write_json(args.output_dir / "experiment_report.json", report)
    write_tsv(args.output_dir / "arm_metrics.tsv", rows, list(rows[0]))
    write_tsv(
        args.output_dir / "comparison_summary.tsv",
        comparisons,
        list(comparisons[0]),
    )
    (args.output_dir / "index.html").write_text(
        render_html(rows, audit), encoding="utf-8"
    )
    print(f"Wrote structure-control report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
