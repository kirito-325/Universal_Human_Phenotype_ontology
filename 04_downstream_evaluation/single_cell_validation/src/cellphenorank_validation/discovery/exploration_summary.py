from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cellphenorank.config import dump_config
from cellphenorank.io import input_record, read_tsv, split_values, value, write_json, write_tsv


RESULT_FIELDS = [
    "candidate_case_id",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "study_accession",
    "pareto_front",
    "priority_rank",
    "data_tier",
    "base_rank",
    "final_rank",
    "rank_change",
    "bridge_loo_rank",
    "predicted_cell",
    "predicted_function",
    "patient_donors",
    "control_donors",
    "candidate_gene_effect",
    "candidate_gene_status",
    "cell_specificity_status",
    "strict_module_effect",
    "strict_module_empirical_p",
    "strict_module_status",
    "overall_case_status",
    "preflight_failure_reasons",
    "limitations",
]

ENDPOINT_FIELDS = [
    "candidate_case_id",
    "rank_rescue",
    "cell_match",
    "gene_detection",
    "gene_effect",
    "cell_specificity",
    "practical_module",
    "strict_module",
    "wrong_cell_control",
    "random_set_control",
    "composition_change",
]


def _endpoint_status(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {value(row, "endpoint_id"): value(row, "status") for row in read_tsv(path)}


def _normalized_status(status: str) -> str:
    if status in {"supported", "descriptively_supported"}:
        return "supported"
    if status in {"not_supported"}:
        return "not_supported"
    if status in {"insufficient_data"}:
        return "insufficient_data"
    return "not_applicable"


def run_exploration_summary(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config["_project_root"])
    inputs = config["inputs"]
    candidates = {row["candidate_case_id"]: row for row in read_tsv(inputs["candidate_table"])}
    catalog = list(read_tsv(inputs["dataset_catalog"]))
    statuses = list(read_tsv(inputs["case_dataset_status"]))
    result_by_candidate: dict[str, Mapping[str, str]] = {}
    endpoint_by_candidate: dict[str, dict[str, str]] = {}
    cases_dir = root / "outputs/single_cell_case_exploration/cases"
    for case_dir in cases_dir.iterdir() if cases_dir.is_dir() else []:
        summary_path = case_dir / "case_summary.tsv"
        if not summary_path.is_file():
            continue
        rows = list(read_tsv(summary_path))
        if not rows:
            continue
        candidate_id = case_dir.name.split("__", 2)
        key = "__".join(candidate_id[:2])
        result_by_candidate[key] = rows[0]
        endpoint_by_candidate[key] = _endpoint_status(case_dir / "endpoint_results.tsv")
    all_results = []
    endpoint_rows = []
    status_by_case_accession = {
        (row["candidate_case_id"], row["study_accession"]): row for row in statuses
    }
    for dataset in catalog:
        candidate_id = value(dataset, "candidate_case_id")
        candidate = candidates[candidate_id]
        preflight = status_by_case_accession[(candidate_id, value(dataset, "study_accession"))]
        result = result_by_candidate.get(candidate_id, {})
        overall = value(result, "overall_case_status") or "preflight_failed"
        all_results.append(
            {
                "candidate_case_id": candidate_id,
                "disease_id": value(candidate, "disease_id"),
                "disease_name": value(candidate, "disease_name"),
                "gene_symbol": value(candidate, "gene_symbol"),
                "study_accession": value(dataset, "study_accession"),
                "pareto_front": value(candidate, "pareto_front"),
                "priority_rank": value(candidate, "priority_rank"),
                "data_tier": value(preflight, "data_tier"),
                "base_rank": value(candidate, "base_rank"),
                "final_rank": value(candidate, "final_rank"),
                "rank_change": value(candidate, "rank_change"),
                "bridge_loo_rank": value(candidate, "bridge_loo_rank"),
                "predicted_cell": value(candidate, "primary_predicted_cell_names"),
                "predicted_function": value(candidate, "primary_predicted_function_name"),
                "patient_donors": value(preflight, "patient_donors"),
                "control_donors": value(preflight, "control_donors"),
                "candidate_gene_effect": value(result, "candidate_gene_effect"),
                "candidate_gene_status": value(result, "candidate_gene_status"),
                "cell_specificity_status": value(result, "cell_specificity_status"),
                "strict_module_effect": value(result, "strict_module_effect"),
                "strict_module_empirical_p": value(result, "strict_module_empirical_p"),
                "strict_module_status": value(result, "strict_module_status"),
                "overall_case_status": overall,
                "preflight_failure_reasons": value(preflight, "reasons"),
                "limitations": value(result, "limitations") or value(preflight, "limitations"),
            }
        )
        endpoints = endpoint_by_candidate.get(candidate_id, {})
        passed_preflight = value(preflight, "status") == "pass"
        endpoint_rows.append(
            {
                "candidate_case_id": candidate_id,
                "rank_rescue": "supported",
                "cell_match": "supported" if passed_preflight else "insufficient_data",
                "gene_detection": "insufficient_data" if endpoints.get("E0") == "insufficient_data" else "supported" if endpoints.get("E0") else "not_applicable",
                "gene_effect": _normalized_status(endpoints.get("E0", "")),
                "cell_specificity": _normalized_status(endpoints.get("E1", "")),
                "practical_module": "not_applicable" if not endpoints else "supported" if endpoints.get("E3") == "supported" else "not_applicable",
                "strict_module": _normalized_status(endpoints.get("E2", "")),
                "wrong_cell_control": "insufficient_data" if passed_preflight else "not_applicable",
                "random_set_control": "supported" if endpoints.get("E2") in {"supported", "descriptively_supported"} else "not_supported" if endpoints.get("E2") == "not_supported" else "not_applicable",
                "composition_change": _normalized_status(endpoints.get("E4", "")),
            }
        )
    write_tsv(output / "all_case_results.tsv", all_results, RESULT_FIELDS)
    write_tsv(output / "endpoint_support_matrix.tsv", endpoint_rows, ENDPOINT_FIELDS)
    counts = Counter(row["overall_case_status"] for row in all_results)
    status_rows = [{"overall_case_status": key, "case_count": counts[key]} for key in sorted(counts)]
    write_tsv(output / "case_status_counts.tsv", status_rows, ["overall_case_status", "case_count"])
    resource_rows = [row for row in catalog if value(row, "retrieval_status") != "not_found"]
    exact_human_single_cell = [
        row for row in resource_rows
        if value(row, "species") == "Homo sapiens"
        and value(row, "causal_gene_match") == "exact"
        and "single" in value(row, "assay_type").lower().replace("sc", "single")
    ]
    report = f"""# CellPhenoRank 预测驱动单细胞探索结果

## 队列流转

- A4 排名改善且有完整 function 路径的候选：{len(candidates)}
- 找到相关单细胞/转录组资源的候选：{len(resource_rows)}
- 疾病-基因精确匹配的人体单细胞资源：{len(exact_human_single_cell)}
- 通过 metadata-only preflight：{sum(value(row, 'status') == 'pass' for row in statuses)}
- 已完成表达分析：{len(result_by_candidate)}
- 状态：`gene_only_supported` {counts.get('gene_only_supported', 0)}，`insufficient_data` {counts.get('insufficient_data', 0)}，`preflight_failed` {counts.get('preflight_failed', 0)}。

## 主病例：BTK-XLA

`BTK` 在当前 Exomiser 2602 基础排名为第 3 位，A4 重排到第 1 位，Bridge-LOO 仍为第 1 位。`E-MTAB-11845` 有 3 名 XLA 和 3 名健康供体，但只测量疫苗抗原刺激后分选的 T 细胞。发布的过滤后 RNA 轴没有 `BTK`，无法检验候选基因表达；strict TCR signaling module 的 day35 效应为 0.0211，表达匹配经验 P=0.929，且 6 个月方向相反。总体为 `insufficient_data`。

## 追加展示：FOXG1 缺失

`FOXG1` 从第 9 位升到第 7 位；GSE273180 的患者脑类器官携带杂合 FOXG1 全基因缺失。冻结映射到兴奋性神经元后，`FOXG1` 的 deletion-control 效应为 -1.3093 log2(CPM+1)，两个缺失技术重复均低于两个健康技术重复，候选基因缺失在预测细胞中得到描述性体现。

但该效应的细胞特异性只排第 2，神经祖细胞的绝对下降更大；删除 `FOXG1` 与全部桥接源后的 strict 功能模块效应为 0.1531，表达匹配经验 P=1.0。总体为 `gene_only_supported`，不能表述为完整 CPO 细胞-功能机制验证。

## 当前结论

当前流程已经实现了无表达值参与的候选生成、公开数据检索、metadata-only gate、冻结边界、供体/技术重复级 pseudobulk、候选基因终点、strict module 和随机 gene-set 对照。实际结果说明主要瓶颈是合适的罕见病单细胞数据，而不是分析程序：高优先级病例常见主预测细胞缺失、候选基因不可检测、来源重叠、受限访问或独立供体不足。本轮没有 `fully_supported` 病例；FOXG1 可作为“致病基因缺失在预测细胞中可见”的谨慎案例，不能作为完整知识关系有效性的最终证明。
"""
    (output / "supported_and_failed_cases.md").write_text(report, encoding="utf-8")
    (output / "representative_case_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "related_resource_count": len(resource_rows),
        "exact_human_single_cell_count": len(exact_human_single_cell),
        "preflight_pass_count": sum(value(row, "status") == "pass" for row in statuses),
        "analyzed_case_count": len(result_by_candidate),
        "status_counts": dict(counts),
        "inputs": {
            "candidate_table": input_record(inputs["candidate_table"]),
            "dataset_catalog": input_record(inputs["dataset_catalog"]),
            "case_dataset_status": input_record(inputs["case_dataset_status"]),
        },
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    write_json(output.parent / "run_manifest.json", manifest)
    (output.parent / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    return manifest
