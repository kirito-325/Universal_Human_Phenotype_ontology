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


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import InputPaths, write_json, write_tsv  # noqa: E402
from mpo_hpo_mapping.pipeline import RunConfig, run_pipeline  # noqa: E402


TASKS = (
    "expert_exact_present",
    "expert_hp_broader",
    "counterfactual_exact_removed",
    "synthetic_hpo_cutout",
)
TASK_LABELS = {
    "expert_exact_present": "专家金标准：Exact",
    "expert_hp_broader": "专家金标准：Broader",
    "counterfactual_exact_removed": "构建金标准：Exact 遮蔽",
    "synthetic_hpo_cutout": "构建金标准：HPO Cutout",
}
CASE_TASKS = {
    "exact": "expert_exact_present",
    "hp_broader": "expert_hp_broader",
    "counterfactual_exact": "counterfactual_exact_removed",
    "synthetic_hpo_cutout": "synthetic_hpo_cutout",
}
SEMANTIC_ONLY_ZERO_FEATURES = {
    "structure_exact", "structure_ancestor", "poincare_score", "raw_structure_score",
    "shared_mechanism_log", "shared_specific_log", "mechanism_jaccard",
    "query_mechanism_log", "hp_mechanism_log",
}
STRUCTURE_ONLY_ZERO_FEATURES = {
    "semantic_exact", "semantic_ancestor", "alias_hit", "alias_jaccard",
}


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    fusion_mode: str
    hierarchy_mode: str


ARMS = (
    Arm("late_fusion_poincare", "晚融合 + 真实 Poincare", "late_fusion", "poincare"),
    Arm("semantic_only", "纯语义", "semantic_only", "poincare"),
    Arm("structure_only", "纯结构/机制", "structure_only", "poincare"),
    Arm("late_fusion_constant", "晚融合 + 常量结构", "late_fusion", "constant"),
    Arm("late_fusion_permuted", "晚融合 + 置换结构", "late_fusion", "permuted"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled v2 late-fusion ablations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "results/late_fusion_ablation",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--permutation-seed", type=int, default=20270727)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_config(
    args: argparse.Namespace,
    arm: Arm,
    output: Path,
    hierarchy_reference: Path | None,
    semantic_reference: Path | None,
) -> RunConfig:
    smoke = bool(args.smoke)
    return RunConfig(
        output_dir=output,
        general_top_k=2 if smoke else 8,
        hierarchy_dimension=8 if smoke else 16,
        hierarchy_epochs=5 if smoke else 30,
        hierarchy_mode=arm.hierarchy_mode,
        hierarchy_reference=(
            str(hierarchy_reference)
            if hierarchy_reference is not None and arm.hierarchy_mode != "constant"
            else ""
        ),
        hierarchy_permutation_seed=args.permutation_seed,
        max_prototypes=2 if smoke else 4,
        semantic_mode="tfidf" if smoke else "auto",
        semantic_dimension=32 if smoke else 64,
        semantic_reference=str(semantic_reference) if semantic_reference else "",
        hidden_dimension=32 if smoke else 64,
        output_dimension=16 if smoke else 32,
        training_epochs=12 if smoke else 80,
        synthetic_limit=500 if smoke else 0,
        ivf_clusters=32 if smoke else 96,
        nprobe=4 if smoke else 8,
        retrieve_k=30 if smoke else 80,
        output_top_k=5 if smoke else 20,
        max_output_mp=100 if smoke else 0,
        evaluation_only=True,
        seed=args.seed,
        data_split_seed=args.seed,
        semantic_seed=args.seed,
        structure_aux_weight=0.35,
        semantic_dropout_probability=0.25,
        balance_parent_sources=True,
        fusion_mode=arm.fusion_mode,
    )


def read_cases(path: Path) -> dict[tuple[str, str, str], int]:
    rows: dict[tuple[str, str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(CASE_TASKS[row["task"]], row["query_id"], row["targets"])] = int(
                row["best_rank"] or 0
            )
    return rows


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def split_audit(run_dir: Path) -> dict[str, object]:
    query_splits: dict[str, set[str]] = {}
    train_pairs: set[tuple[str, str, str]] = set()
    test_pairs: set[tuple[str, str, str]] = set()
    for filename in (
        "expert_mapping_examples.tsv",
        "parent_training_examples.tsv",
    ):
        path = run_dir / "data/processed" / filename
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["query_profile_source"] == "hp":
                    query_key = f"hp:{row['exact_hp_id']}"
                else:
                    query_key = f"mp:{row['mp_id']}"
                split = row["split"]
                query_splits.setdefault(query_key, set()).add(split)
                pair = (row["source"], query_key, row["hp_id"])
                if split == "train":
                    train_pairs.add(pair)
                elif split == "test":
                    test_pairs.add(pair)
    mixed = sorted(key for key, splits in query_splits.items() if len(splits) > 1)
    pair_overlap = sorted(train_pairs & test_pairs)
    return {
        "query_entities": len(query_splits),
        "mixed_split_query_count": len(mixed),
        "train_test_pair_overlap_count": len(pair_overlap),
        "passed": not mixed and not pair_overlap,
    }


def modality_audit(run_dirs: Mapping[str, Path]) -> dict[str, object]:
    attributions = {
        arm_id: read_json(path / "model/fusion_attribution.json")
        for arm_id, path in run_dirs.items()
    }

    def coefficients_are_zero(arm_id: str, names: set[str]) -> bool:
        coefficients = attributions[arm_id]["relation_coefficients_standardized"]
        return all(
            abs(float(features[name])) <= 1e-12
            for features in coefficients.values()
            for name in names
        )

    semantic = attributions["semantic_only"]
    structure = attributions["structure_only"]
    semantic_fusion = all(
        weights == {"semantic": 1.0, "structure": 0.0, "poincare": 0.0}
        for weights in semantic["fusion_weights"].values()
    )
    structure_fusion = all(
        float(weights["semantic"]) == 0.0
        and abs(sum(float(value) for value in weights.values()) - 1.0) <= 1e-6
        for weights in structure["fusion_weights"].values()
    )
    semantic_rank = semantic["ancestor_rank_weights"]
    structure_rank = structure["ancestor_rank_weights"]
    source_balanced = all(
        max(float(value) for value in attribution["relation_source_weight_totals"].values())
        - min(float(value) for value in attribution["relation_source_weight_totals"].values())
        <= 1e-8
        for attribution in attributions.values()
    )
    checks = {
        "semantic_only_fusion_masked": semantic_fusion,
        "semantic_only_relation_features_masked": coefficients_are_zero(
            "semantic_only", SEMANTIC_ONLY_ZERO_FEATURES
        ),
        "semantic_only_ranker_masked": (
            float(semantic_rank["structure"]) == 0.0
            and float(semantic_rank["mechanism"]) == 0.0
        ),
        "structure_only_fusion_masked": structure_fusion,
        "structure_only_relation_features_masked": coefficients_are_zero(
            "structure_only", STRUCTURE_ONLY_ZERO_FEATURES
        ),
        "structure_only_ranker_masked": float(structure_rank["semantic"]) == 0.0,
        "relation_sources_balanced": source_balanced,
    }
    return {**checks, "passed": all(checks.values())}


def shared_config_audit(run_dirs: Mapping[str, Path]) -> bool:
    ignored = {
        "output_dir", "fusion_mode", "hierarchy_mode", "hierarchy_reference",
        "hierarchy_permutation_seed", "semantic_reference",
    }
    controlled = []
    for path in run_dirs.values():
        config = read_json(path / "training_manifest.json")["config"]
        controlled.append({key: value for key, value in config.items() if key not in ignored})
    return len({json.dumps(config, sort_keys=True) for config in controlled}) == 1


def metrics(ranks: Sequence[int]) -> dict[str, float]:
    count = max(1, len(ranks))
    return {
        "mrr": sum(1.0 / rank for rank in ranks if rank > 0) / count,
        "recall_at_1": sum(0 < rank <= 1 for rank in ranks) / count,
        "recall_at_10": sum(0 < rank <= 10 for rank in ranks) / count,
        "recall_at_50": sum(0 < rank <= 50 for rank in ranks) / count,
    }


def render_html(rows: Sequence[Mapping[str, object]], audit: Mapping[str, object]) -> str:
    sections: list[str] = []
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{int(row['queries'])}</td>"
            f"<td>{100 * float(row['recall_at_1']):.2f}%</td>"
            f"<td>{100 * float(row['recall_at_10']):.2f}%</td>"
            f"<td>{100 * float(row['recall_at_50']):.2f}%</td>"
            "</tr>"
            for row in selected
        )
        sections.append(
            f"<section><h2>{html.escape(TASK_LABELS[task])}</h2><div class='table-wrap'>"
            "<table><thead><tr><th>实验组</th><th>样本数</th><th>Recall@1</th>"
            f"<th>Recall@10</th><th>Recall@50</th></tr></thead><tbody>{body}"
            "</tbody></table></div></section>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPO-HPO v2 晚融合消融</title>
<style>:root{{--ink:#18201d;--muted:#64706b;--line:#d8dfdb;--wash:#f4f6f5}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}header{{padding:30px max(22px,calc((100vw - 1080px)/2));background:var(--wash);border-bottom:1px solid var(--line)}}h1{{margin:0 0 6px;font-size:25px;letter-spacing:0}}header p{{margin:0;color:var(--muted)}}main{{max-width:1080px;margin:auto;padding:26px 22px 50px}}section{{margin:0 0 28px}}h2{{font-size:17px;letter-spacing:0}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:var(--wash)}}tr:last-child td{{border-bottom:0}}</style></head><body>
<header><h1>MPO-HPO Mapping v2 晚融合消融</h1><p>同一数据划分、语义向量、训练预算和测试集；结构坐标、模态与排序归因均可审计。</p></header><main>
<p>受控文件哈希一致：<strong>{str(audit['controlled_hashes_identical']).lower()}</strong>；评估队列一致：<strong>{str(audit['evaluation_cohort_identical']).lower()}</strong>；训练/测试查询隔离：<strong>{str(audit['split_isolation_passed']).lower()}</strong>；模态屏蔽审计：<strong>{str(audit['modality_constraints_passed']).lower()}</strong></p>
{''.join(sections)}</main></body></html>"""


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    runs = root / "runs"
    root.mkdir(parents=True, exist_ok=True)
    reference: Path | None = None
    semantic_reference: Path | None = None
    run_dirs: dict[str, Path] = {}
    for position, arm in enumerate(ARMS, 1):
        run_dir = runs / arm.arm_id
        run_dirs[arm.arm_id] = run_dir
        complete = (run_dir / "training_manifest.json").is_file() and (
            run_dir / "evaluation_cases.tsv"
        ).is_file()
        if args.resume and complete:
            print(f"[{position}/{len(ARMS)}] Reuse {arm.arm_id}", flush=True)
        else:
            print(f"[{position}/{len(ARMS)}] Run {arm.arm_id}", flush=True)
            run_pipeline(
                InputPaths(),
                make_config(args, arm, run_dir, reference, semantic_reference),
            )
        if arm.arm_id == "late_fusion_poincare":
            reference = run_dir / "data/processed/hierarchy_embeddings.npz"
            semantic_reference = run_dir / "data/processed/semantic_embeddings.npz"

    controlled = (
        "data/processed/expert_mapping_examples.tsv",
        "data/processed/parent_training_examples.tsv",
        "data/processed/semantic_embeddings.npz",
        "data/processed/term_mechanism_edges.tsv",
    )
    controlled_hashes = {
        name: {arm_id: sha256(path / name) for arm_id, path in run_dirs.items()}
        for name in controlled
    }
    cohorts = {arm_id: read_cases(path / "evaluation_cases.tsv") for arm_id, path in run_dirs.items()}
    if any(len(set(values.values())) != 1 for values in controlled_hashes.values()):
        raise RuntimeError("Controlled input hashes differ across v2 ablation arms")
    if len({frozenset(rows) for rows in cohorts.values()}) != 1:
        raise RuntimeError("Evaluation cohorts differ across v2 ablation arms")
    split_audits = {arm_id: split_audit(path) for arm_id, path in run_dirs.items()}
    modality_constraints = modality_audit(run_dirs)
    shared_config_identical = shared_config_audit(run_dirs)
    if not all(row["passed"] for row in split_audits.values()):
        raise RuntimeError("A train/test query entity or labeled pair crosses splits")
    if not modality_constraints["passed"]:
        raise RuntimeError("A single-modality arm retained a disabled modality shortcut")
    if not shared_config_identical:
        raise RuntimeError("Non-experimental hyperparameters differ across ablation arms")

    metric_rows: list[dict[str, object]] = []
    for arm in ARMS:
        cases = cohorts[arm.arm_id]
        for task in TASKS:
            ranks = [rank for key, rank in cases.items() if key[0] == task]
            metric_rows.append(
                {
                    "arm_id": arm.arm_id,
                    "label": arm.label,
                    "fusion_mode": arm.fusion_mode,
                    "hierarchy_mode": arm.hierarchy_mode,
                    "task": task,
                    "queries": len(ranks),
                    **metrics(ranks),
                }
            )
    audit = {
        "controlled_hashes_identical": True,
        "evaluation_cohort_identical": True,
        "shared_non_experimental_config_identical": shared_config_identical,
        "split_isolation_passed": True,
        "modality_constraints_passed": True,
        "controlled_hashes": controlled_hashes,
        "split_audits": split_audits,
        "modality_constraints": modality_constraints,
    }
    report = {
        "design": {
            "seed": args.seed,
            "permutation_seed": args.permutation_seed,
            "smoke": bool(args.smoke),
            "arms": [arm.__dict__ for arm in ARMS],
        },
        "audit": audit,
        "metrics": metric_rows,
    }
    write_json(root / "experiment_report.json", report)
    write_tsv(root / "arm_metrics.tsv", metric_rows, list(metric_rows[0]))
    (root / "index.html").write_text(render_html(metric_rows, audit), encoding="utf-8")
    print(f"Wrote v2 ablation report to {root}", flush=True)


if __name__ == "__main__":
    main()
