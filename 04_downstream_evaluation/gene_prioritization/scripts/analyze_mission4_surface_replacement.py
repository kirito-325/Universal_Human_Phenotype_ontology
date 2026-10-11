#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from cellphenorank.evaluation import (
    BENCHMARK_METRIC_FIELDS,
    DISEASE_METRIC_FIELDS,
    RANK_CHANGE_FIELDS,
    STATISTICAL_TEST_FIELDS,
    compare_disease_ranks,
    evaluate_rankings,
    load_gold_genes,
    metrics_as_rows,
)
from cellphenorank.io import (
    input_record,
    load_diseases,
    load_model_phenotypes,
    load_models,
    read_tsv,
    write_json,
    write_tsv,
)
from cellphenorank.phenodigm_legacy import LegacyPhenodigmMatrix, _rank


def filtered_diseases(
    diseases: Mapping[str, Sequence[Any]], allowed_hpo: set[str]
) -> dict[str, list[Any]]:
    result = {
        disease_id: [term for term in terms if term.hpo_id in allowed_hpo]
        for disease_id, terms in diseases.items()
    }
    empty = sorted(disease_id for disease_id, terms in result.items() if not terms)
    if empty:
        raise ValueError(f"Common HPO support leaves empty disease profiles: {empty}")
    return result


def metrics_for_rows(
    rows: list[dict[str, Any]],
    gold: Mapping[str, set[str]],
    rank_key: str,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    return evaluate_rankings(rows, gold, rank_key, label)


def common_support_ranking(
    diseases: Mapping[str, Sequence[Any]],
    models: Mapping[str, Any],
    phenotypes: Mapping[str, Sequence[Any]],
    matrix: LegacyPhenodigmMatrix,
) -> list[dict[str, Any]]:
    _, rows = _rank(diseases, models, phenotypes, matrix, "exomiser_current")
    return rows


def metric_cell(value: float, metric: str) -> str:
    if metric == "median_rank":
        return f"{value:.1f}"
    return f"{value:.4f}"


def surface_density(matrix: LegacyPhenodigmMatrix) -> dict[str, object]:
    counts: list[int] = []
    positive_mpo: set[str] = set()
    positive_pairs = 0
    for scores in matrix.scores.values():
        current = {mp_id for mp_id, score in scores.items() if score > 0}
        if current:
            counts.append(len(current))
            positive_mpo.update(current)
            positive_pairs += len(current)
    denominator = len(counts) * len(positive_mpo)
    return {
        "hpo_with_positive_candidate_pair": len(counts),
        "candidate_mpo_with_positive_pair": len(positive_mpo),
        "positive_candidate_pairs": positive_pairs,
        "conditional_positive_density": (
            positive_pairs / denominator if denominator else 0.0
        ),
        "positive_pairs_per_hpo_quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
                np.quantile(counts, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)),
            )
        }
        if counts
        else {},
    }


def run_metrics(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in read_tsv(path):
        result.setdefault(str(row["experiment"]), {})[str(row["metric"])] = float(
            row["value"]
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gold = load_gold_genes(str(args.gold_standard))
    old_full_rows = list(read_tsv(args.old_rankings))
    new_full_rows = list(read_tsv(args.new_rankings))
    old_full_disease, old_full_metrics = metrics_for_rows(
        old_full_rows, gold, "base_rank", "Exomiser_2602_full"
    )
    new_full_disease, new_full_metrics = metrics_for_rows(
        new_full_rows, gold, "base_rank", "mission4_CPO_full"
    )

    models_all = load_models(args.models)
    models = {
        model_id: model
        for model_id, model in models_all.items()
        if ";" not in model.human_gene_symbol
    }
    phenotypes = load_model_phenotypes(args.model_phenotypes, set(models))
    candidate_mpo = {
        row.mpo_id
        for rows in phenotypes.values()
        for row in rows
        if row.status == "abnormal"
    }
    old_matrix = LegacyPhenodigmMatrix(
        args.old_matrix, args.old_presence, candidate_mpo
    )
    new_matrix = LegacyPhenodigmMatrix(
        args.new_matrix, args.new_presence, candidate_mpo
    )
    old_density = surface_density(old_matrix)
    new_density = surface_density(new_matrix)
    diseases = load_diseases(args.nonredundant_diseases)
    common_hpo = set(old_matrix.best_by_hpo).intersection(new_matrix.best_by_hpo)
    diseases_common = filtered_diseases(diseases, common_hpo)
    common_mpo = candidate_mpo.intersection(old_matrix.present_mpo, new_matrix.present_mpo)
    phenotypes_common = {
        model_id: [
            row
            for row in rows
            if row.status != "abnormal" or row.mpo_id in common_mpo
        ]
        for model_id, rows in phenotypes.items()
    }

    scenarios: list[dict[str, Any]] = []

    def add_scenario(
        scenario: str,
        old_rows: list[dict[str, Any]],
        new_rows: list[dict[str, Any]],
        rank_key: str,
    ) -> None:
        old_disease, old_metrics = metrics_for_rows(
            old_rows, gold, rank_key, f"Exomiser_2602_{scenario}"
        )
        new_disease, new_metrics = metrics_for_rows(
            new_rows, gold, rank_key, f"mission4_CPO_{scenario}"
        )
        changes, comparison = compare_disease_ranks(
            old_disease,
            new_disease,
            f"mission4_vs_2602_{scenario}",
            args.bootstrap_iterations,
            args.seed,
        )
        scenarios.append(
            {
                "scenario": scenario,
                "old_metrics": old_metrics,
                "new_metrics": new_metrics,
                "old_disease": old_disease,
                "new_disease": new_disease,
                "changes": changes,
                "comparison": comparison,
            }
        )

    add_scenario("full", old_full_rows, new_full_rows, "base_rank")
    old_common_hpo = common_support_ranking(
        diseases_common, models, phenotypes, old_matrix
    )
    new_common_hpo = common_support_ranking(
        diseases_common, models, phenotypes, new_matrix
    )
    add_scenario("common_hpo", old_common_hpo, new_common_hpo, "phenodigm_rank")
    old_common_both = common_support_ranking(
        diseases_common, models, phenotypes_common, old_matrix
    )
    new_common_both = common_support_ranking(
        diseases_common, models, phenotypes_common, new_matrix
    )
    add_scenario(
        "common_hpo_mpo", old_common_both, new_common_both, "phenodigm_rank"
    )

    disease_names = {
        str(row["disease_id"]): str(row.get("disease_name", ""))
        for row in new_full_rows
    }
    common_counts = {
        disease_id: sum(term.hpo_id in common_hpo for term in terms)
        for disease_id, terms in diseases.items()
    }
    profile_counts = {disease_id: len(terms) for disease_id, terms in diseases.items()}
    primary_rows = []
    for row in scenarios[0]["changes"]:
        disease_id = str(row["disease_id"])
        total = profile_counts[disease_id]
        mapped = common_counts[disease_id]
        primary_rows.append(
            {
                **row,
                "disease_name": disease_names.get(disease_id, ""),
                "gold_genes": sorted(gold.get(disease_id, set())),
                "nonredundant_hpo_count": total,
                "mission4_mapped_hpo_count": mapped,
                "mission4_hpo_coverage": mapped / total if total else 0.0,
            }
        )
    coverage_values = [float(row["mission4_hpo_coverage"]) for row in primary_rows]
    rank_changes = [float(row["rank_change"]) for row in primary_rows]
    correlation = spearmanr(coverage_values, rank_changes)

    old_run = args.old_rankings.resolve().parents[1]
    new_run = args.new_rankings.resolve().parents[1]
    old_arm_metrics = run_metrics(old_run / "benchmark/benchmark_metrics.tsv")
    new_arm_metrics = run_metrics(new_run / "benchmark/benchmark_metrics.tsv")
    old_arm_disease = list(read_tsv(old_run / "benchmark/disease_level_metrics.tsv"))
    new_arm_disease = list(read_tsv(new_run / "benchmark/disease_level_metrics.tsv"))
    arm_comparisons = []
    for arm in (
        "A0_surface",
        "A3_strict_constrained",
        "A4_extended_constrained",
        "A5_bridge_loo",
    ):
        _, comparison = compare_disease_ranks(
            [row for row in old_arm_disease if row["experiment"] == arm],
            [row for row in new_arm_disease if row["experiment"] == arm],
            f"mission4_vs_2602_{arm}",
            args.bootstrap_iterations,
            args.seed,
        )
        arm_comparisons.append(comparison)

    metric_rows: list[dict[str, Any]] = []
    disease_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        metric_rows.extend(
            metrics_as_rows(
                f"Exomiser_2602_{scenario['scenario']}", scenario["old_metrics"]
            )
        )
        metric_rows.extend(
            metrics_as_rows(
                f"mission4_CPO_{scenario['scenario']}", scenario["new_metrics"]
            )
        )
        disease_rows.extend(scenario["old_disease"])
        disease_rows.extend(scenario["new_disease"])
        change_rows.extend(scenario["changes"])
        test_rows.append(scenario["comparison"])
    write_tsv(output / "benchmark_metrics.tsv", metric_rows, BENCHMARK_METRIC_FIELDS)
    write_tsv(output / "disease_level_metrics.tsv", disease_rows, DISEASE_METRIC_FIELDS)
    write_tsv(output / "rank_changes.tsv", change_rows, RANK_CHANGE_FIELDS)
    write_tsv(output / "statistical_tests.tsv", test_rows, STATISTICAL_TEST_FIELDS)
    write_tsv(
        output / "cross_matrix_arm_comparisons.tsv",
        arm_comparisons,
        STATISTICAL_TEST_FIELDS,
    )
    write_tsv(
        output / "full_cohort_disease_changes.tsv",
        primary_rows,
        [
            *RANK_CHANGE_FIELDS,
            "disease_name",
            "gold_genes",
            "nonredundant_hpo_count",
            "mission4_mapped_hpo_count",
            "mission4_hpo_coverage",
        ],
    )

    extraction = json.loads(args.extraction_manifest.read_text(encoding="utf-8"))
    manifest = {
        "generated_from_frozen_benchmark": True,
        "primary_scenario": "full",
        "diagnostic_scenarios": ["common_hpo", "common_hpo_mpo"],
        "common_support": {
            "hpo": len(common_hpo),
            "mpo": len(common_mpo),
            "all_diseases_retain_at_least_one_hpo": True,
        },
        "coverage_rank_change_spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        },
        "surface_density": {
            "exomiser_2602": old_density,
            "mission4_cpo": new_density,
            "denominator_note": (
                "Density is conditional on HPO and candidate MPO having at least one "
                "positive pair in the corresponding surface."
            ),
        },
        "reranking_arms": {
            arm: {
                "exomiser_2602": old_arm_metrics[arm],
                "mission4_cpo": new_arm_metrics[arm],
                "paired_comparison": next(
                    row
                    for row in arm_comparisons
                    if row["experiment"] == f"mission4_vs_2602_{arm}"
                ),
            }
            for arm in (
                "A0_surface",
                "A3_strict_constrained",
                "A4_extended_constrained",
                "A5_bridge_loo",
            )
        },
        "surface_extraction": extraction,
        "scenarios": {
            scenario["scenario"]: {
                "exomiser_2602": scenario["old_metrics"],
                "mission4_cpo": scenario["new_metrics"],
                "paired_comparison": scenario["comparison"],
            }
            for scenario in scenarios
        },
        "inputs": {
            "old_rankings": input_record(args.old_rankings),
            "new_rankings": input_record(args.new_rankings),
            "old_matrix": input_record(args.old_matrix),
            "new_matrix": input_record(args.new_matrix),
            "gold_standard": input_record(args.gold_standard),
            "nonredundant_diseases": input_record(args.nonredundant_diseases),
        },
    }
    write_json(output / "comparison_manifest.json", manifest)

    lines = [
        "# mission4 CPO HPO-MPO surface replacement benchmark",
        "",
        "本实验固定 50 个疾病、49 个候选基因、1,068 个单基因小鼠模型、金标准与 Exomiser 当前 profile 公式，唯一主变量是 HPO-MPO 成对相似度矩阵。",
        "",
        "## 主要结果",
        "",
        "| 场景 | 矩阵 | Top1 | Top5 | Top10 | MRR | NDCG@10 | 中位排名 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        for label, key in (("Exomiser 2602", "old_metrics"), ("mission4 CPO", "new_metrics")):
            values = scenario[key]
            lines.append(
                f"| {scenario['scenario']} | {label} | "
                f"{metric_cell(values['recall_at_1'], 'recall_at_1')} | "
                f"{metric_cell(values['recall_at_5'], 'recall_at_5')} | "
                f"{metric_cell(values['recall_at_10'], 'recall_at_10')} | "
                f"{metric_cell(values['mrr'], 'mrr')} | "
                f"{metric_cell(values['ndcg_at_10'], 'ndcg_at_10')} | "
                f"{metric_cell(values['median_rank'], 'median_rank')} |"
            )
    full = scenarios[0]["comparison"]
    ordered_changes = sorted(
        primary_rows,
        key=lambda row: (-float(row["rank_change"]), str(row["disease_id"])),
    )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            f"完整输入下，mission4 相对 Exomiser 2602 为 {full['wins']} 胜 / {full['ties']} 平 / {full['losses']} 负；平均排名变化 {full['mean_rank_change']:.2f}，Wilcoxon p={full['wilcoxon_p_value']:.6g}。正值表示 mission4 改善。",
            "",
            "## 相同重排算法下的跨矩阵比较",
            "",
            "| 分析臂 | 矩阵 | Top1 | Top5 | Top10 | MRR | NDCG@10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in (
        "A0_surface",
        "A3_strict_constrained",
        "A4_extended_constrained",
        "A5_bridge_loo",
    ):
        for label, source in (
            ("Exomiser 2602", old_arm_metrics),
            ("mission4 CPO", new_arm_metrics),
        ):
            values = source[arm]
            lines.append(
                f"| {arm} | {label} | {values['recall_at_1']:.4f} | "
                f"{values['recall_at_5']:.4f} | {values['recall_at_10']:.4f} | "
                f"{values['mrr']:.4f} | {values['ndcg_at_10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 覆盖与诊断口径",
            "",
            f"- mission4 `final/shape` 覆盖冻结疾病 HPO {extraction['counts']['benchmark_hpo_with_final_profile']}/{extraction['counts']['benchmark_hpo']}（{extraction['coverage']['hpo_final_profile']:.2%}）。",
            f"- 候选 MPO 在 mission4 索引中的覆盖为 {extraction['counts']['candidate_mpo_in_mission4_index']}/{extraction['counts']['candidate_mpo']}（{extraction['coverage']['mpo_index']:.2%}）；与任一基准 HPO 有正分的 MPO 为 {extraction['counts']['candidate_mpo_with_positive_benchmark_pair']}/{extraction['counts']['candidate_mpo']}（{extraction['coverage']['mpo_positive_pair']:.2%}）。",
            f"- 共同支持诊断保留 {len(common_hpo)} 个 HPO 和 {len(common_mpo)} 个 MPO；它不是主结果，只用于判断覆盖损失能否解释性能差异。",
            f"- 条件正分密度：Exomiser 2602 为 {old_density['conditional_positive_density']:.2%}，mission4 为 {new_density['conditional_positive_density']:.2%}。mission4 每个已映射 HPO 的正分候选 MPO 中位数为 {new_density['positive_pairs_per_hpo_quantiles']['0.5']:.0f}。",
            f"- 疾病 HPO 覆盖率与排名变化的 Spearman rho={correlation.statistic:.3f}，p={correlation.pvalue:.4g}。",
            "",
            "## 代表性逐病变化",
            "",
            "| 方向 | 疾病 | 金标准基因 | 2602 排名 | mission4 排名 | 变化 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for direction, rows in (
        ("改善", ordered_changes[:3]),
        ("下降", list(reversed(ordered_changes[-3:]))),
    ):
        for row in rows:
            lines.append(
                f"| {direction} | {row['disease_id']} {row['disease_name']} | "
                f"{';'.join(row['gold_genes'])} | {float(row['base_best_known_rank']):.0f} | "
                f"{float(row['final_best_known_rank']):.0f} | {float(row['rank_change']):+.0f} |"
            )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "当前 mission4 `final/shape` 不适合作为 Exomiser 2602 pair surface 的直接替代。2602 的 pair score 为 `sqrt(simJ * IC)`；mission4 分数则是 CL 与 GO Lin 相似度乘积后的双向 BMA，缺少等价的 HPO-MPO 信息量项，同时正分面过密。下一版应先在不查看冻结测试结果的开发数据上定义稀疏化/显著性门槛和 IC 校准，再回到冻结基准做一次性验证。",
            "",
            "## 结论边界",
            "",
            "`full` 是实际替换效果；`common_hpo` 与 `common_hpo_mpo` 是事后共同支持敏感性分析。若共同支持下仍明显落后，则问题不能只归因于术语缺失，还涉及相似度的区分能力或 profile 聚合与 Exomiser 归一化的兼容性。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a mission4 CPO surface to Exomiser 2602.")
    parser.add_argument("--old-rankings", type=Path, required=True)
    parser.add_argument("--new-rankings", type=Path, required=True)
    parser.add_argument("--old-matrix", type=Path, required=True)
    parser.add_argument("--new-matrix", type=Path, required=True)
    parser.add_argument("--old-presence", type=Path, required=True)
    parser.add_argument("--new-presence", type=Path, required=True)
    parser.add_argument("--gold-standard", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-phenotypes", type=Path, required=True)
    parser.add_argument("--nonredundant-diseases", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
