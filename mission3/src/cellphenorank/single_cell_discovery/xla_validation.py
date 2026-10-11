from __future__ import annotations

import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t
from scipy.stats import ttest_ind

from ..config import dump_config
from ..io import input_record, read_tsv, value, write_json, write_tsv


def _verify_manifest(path: Path) -> str:
    checksum_path = path.with_name("manifest.sha256")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != observed:
        raise RuntimeError("Frozen case manifest checksum mismatch")
    return observed


def _function_genes(path: str | Path, function_id: str) -> set[str]:
    genes = set()
    for row in read_tsv(path):
        if value(row, "go_id", "function_id") == function_id:
            gene = value(row, "human_gene_symbol", "gene_symbol").upper()
            if gene:
                genes.add(gene)
    return genes


def _effect(control: np.ndarray, disease: np.ndarray) -> dict[str, float]:
    control = np.asarray(control, dtype=float)
    disease = np.asarray(disease, dtype=float)
    difference = float(disease.mean() - control.mean())
    control_var = float(control.var(ddof=1)) if len(control) > 1 else 0.0
    disease_var = float(disease.var(ddof=1)) if len(disease) > 1 else 0.0
    se = math.sqrt(control_var / len(control) + disease_var / len(disease))
    numerator = (control_var / len(control) + disease_var / len(disease)) ** 2
    denominator = 0.0
    if len(control) > 1:
        denominator += (control_var / len(control)) ** 2 / (len(control) - 1)
    if len(disease) > 1:
        denominator += (disease_var / len(disease)) ** 2 / (len(disease) - 1)
    df = numerator / denominator if denominator else float(len(control) + len(disease) - 2)
    critical = float(student_t.ppf(0.975, df)) if se and df > 0 else 0.0
    pooled_den = len(control) + len(disease) - 2
    pooled = math.sqrt(
        ((len(control) - 1) * control_var + (len(disease) - 1) * disease_var)
        / pooled_den
    ) if pooled_den > 0 else 0.0
    correction = 1.0 - 3.0 / (4.0 * (len(control) + len(disease)) - 9.0)
    hedges_g = correction * difference / pooled if pooled else 0.0
    test = ttest_ind(disease, control, equal_var=False) if len(control) >= 2 and len(disease) >= 2 else None
    return {
        "effect": difference,
        "standard_error": se,
        "ci_low": difference - critical * se,
        "ci_high": difference + critical * se,
        "hedges_g": hedges_g,
        "p_value": float(test.pvalue) if test is not None and math.isfinite(float(test.pvalue)) else 1.0,
    }


def _module_scores(expression: pd.DataFrame, genes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    present = [gene for gene in genes if gene in expression.columns]
    if not present:
        return np.zeros(len(expression)), []
    matrix = expression[present].to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    variable = std > 0
    if not variable.any():
        return np.zeros(len(expression)), []
    z = (matrix[:, variable] - means[variable]) / std[variable]
    return z.mean(axis=1), [gene for gene, keep in zip(present, variable) if keep]


def _matched_null(
    expression: pd.DataFrame,
    conditions: pd.Series,
    target_genes: Sequence[str],
    excluded: set[str],
    iterations: int,
    bins: int,
    seed: int,
    control_value: str = "HC",
    disease_value: str = "XLA",
) -> tuple[float, float, np.ndarray, int]:
    all_genes = list(expression.columns)
    available = [gene for gene in all_genes if gene not in excluded]
    means = expression[all_genes].mean(axis=0)
    quantiles = pd.qcut(
        means.rank(method="first"), q=min(bins, len(means)), labels=False
    )
    bin_by_gene = {gene: int(quantiles.loc[gene]) for gene in all_genes}
    matrix = expression[all_genes].to_numpy(dtype=float)
    std = matrix.std(axis=0, ddof=1)
    variable = std > 0
    standardized = np.zeros_like(matrix)
    standardized[:, variable] = (
        matrix[:, variable] - matrix[:, variable].mean(axis=0)
    ) / std[variable]
    control = conditions.astype(str).eq(control_value).to_numpy()
    disease = conditions.astype(str).eq(disease_value).to_numpy()
    gene_effects = standardized[disease].mean(axis=0) - standardized[control].mean(axis=0)
    effect_by_gene = {
        gene: float(effect) for gene, effect, keep in zip(all_genes, gene_effects, variable) if keep
    }
    target_variable = [gene for gene in target_genes if gene in effect_by_gene]
    target_effect = float(np.mean([effect_by_gene[gene] for gene in target_variable]))
    target_bin_counts: dict[int, int] = {}
    for item in target_variable:
        target_bin = bin_by_gene[item]
        target_bin_counts[target_bin] = target_bin_counts.get(target_bin, 0) + 1
    pools_by_bin = {
        target_bin: np.asarray(
            [
                gene for gene in available
                if gene in effect_by_gene and bin_by_gene[gene] == target_bin
            ],
            dtype=object,
        )
        for target_bin in target_bin_counts
    }
    fallback_pool = np.asarray(
        [gene for gene in available if gene in effect_by_gene], dtype=object
    )
    rng = np.random.default_rng(seed)
    null_effects = []
    for _ in range(iterations):
        selected: list[str] = []
        used: set[str] = set()
        for target_bin, needed in sorted(target_bin_counts.items()):
            pool = pools_by_bin[target_bin]
            take = min(needed, len(pool))
            if take:
                chosen = rng.choice(pool, size=take, replace=False)
                selected.extend(str(item) for item in chosen)
                used.update(str(item) for item in chosen)
        if len(selected) < len(target_variable):
            remaining = np.asarray(
                [gene for gene in fallback_pool if str(gene) not in used], dtype=object
            )
            needed = min(len(target_variable) - len(selected), len(remaining))
            if needed:
                chosen = rng.choice(remaining, size=needed, replace=False)
                selected.extend(str(item) for item in chosen)
        null_effects.append(float(np.mean([effect_by_gene[gene] for gene in selected])))
    null = np.asarray(null_effects, dtype=float)
    empirical_p = (1.0 + float(np.sum(np.abs(null) >= abs(target_effect)))) / (len(null) + 1.0)
    percentile = float(np.mean(np.abs(null) <= abs(target_effect)))
    return target_effect, empirical_p, null, len(target_variable)


def _choose_primary_timepoint(metadata: pd.DataFrame) -> str:
    candidates = []
    for timepoint, rows in metadata.groupby("timepoint"):
        counts = rows.groupby("condition")["donor_id"].nunique()
        candidates.append((-
            min(int(counts.get("HC", 0)), int(counts.get("XLA", 0))),
            str(timepoint),
        ))
    return sorted(candidates)[0][1]


def _plot_donor_endpoints(
    table: pd.DataFrame, output: Path, primary_timepoint: str
) -> None:
    metrics = [
        ("BTK_log2_cpm", "BTK log2(CPM + 1)"),
        ("strict_module_score", "Strict TCR module score"),
        ("practical_module_score", "Practical TCR module score"),
    ]
    timepoints = [primary_timepoint] + [item for item in sorted(table["timepoint"].unique()) if item != primary_timepoint]
    fig, axes = plt.subplots(len(metrics), len(timepoints), figsize=(4.2 * len(timepoints), 8.8), squeeze=False)
    colors = {"HC": "#2878B5", "XLA": "#D9534F"}
    rng = np.random.default_rng(2026)
    for row_index, (metric, title) in enumerate(metrics):
        for column_index, timepoint in enumerate(timepoints):
            axis = axes[row_index, column_index]
            subset = table[table["timepoint"].eq(timepoint)]
            for x, condition in enumerate(["HC", "XLA"]):
                values = subset.loc[subset["condition"].eq(condition), metric].to_numpy(float)
                jitter = rng.uniform(-0.055, 0.055, size=len(values))
                axis.scatter(np.full(len(values), x) + jitter, values, s=38, color=colors[condition], edgecolor="white", linewidth=0.6, zorder=3)
                if len(values):
                    axis.plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="#222222", linewidth=1.5)
            axis.set_xticks([0, 1], ["HC", "XLA"])
            axis.set_title(f"{title}\n{timepoint}", fontsize=10)
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_null(null: np.ndarray, observed: float, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    axis.hist(null, bins=35, color="#8A9A9A", edgecolor="white", linewidth=0.4)
    axis.axvline(observed, color="#C23B22", linewidth=2, label=f"Observed = {observed:.3f}")
    axis.axvline(-abs(observed), color="#C23B22", linewidth=1, linestyle="--", alpha=0.7)
    axis.axvline(abs(observed), color="#C23B22", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_xlabel("XLA - HC module effect")
    axis.set_ylabel("Expression-matched random sets")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_xla_validation(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["pseudobulk", "gene", "function", "controls", "figures", "report"]:
        (output / name).mkdir(exist_ok=True)
    inputs = config["inputs"]
    settings = config.get("single_cell_validation", {})
    manifest_path = Path(inputs["freeze_manifest"])
    manifest_hash = _verify_manifest(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rds_path = Path(inputs["rds"])
    counts_path = output / "pseudobulk/pseudobulk_counts.tsv.gz"
    metadata_path = output / "pseudobulk/pseudobulk_profiles.tsv"
    script_path = Path(config["_project_root"]) / "scripts/export_seurat_pseudobulk.R"
    subprocess.run(
        ["Rscript", str(script_path), str(rds_path), str(counts_path), str(metadata_path)],
        check=True,
    )
    counts = pd.read_csv(counts_path, sep="\t", index_col=0).T
    counts.columns = counts.columns.astype(str).str.upper()
    counts = counts.loc[:, ~counts.columns.duplicated()]
    metadata = pd.read_csv(metadata_path, sep="\t").set_index("profile_id")
    metadata = metadata.loc[counts.index].copy()
    metadata.index.name = "profile_id"
    log_cpm = np.log2(counts.div(metadata["library_size"], axis=0) * 1_000_000.0 + 1.0)
    primary_cell = str(settings.get("primary_cell_label", "T_all"))
    primary_profiles = metadata[metadata["cell_label"].astype(str).eq(primary_cell)]
    primary_timepoint = _choose_primary_timepoint(primary_profiles)
    gene = str(manifest["case"]["candidate_gene"]).upper()
    function_id = str(manifest["knowledge_prediction"]["primary_function_id"])
    practical = _function_genes(inputs["go_functions"], function_id)
    bridge = {str(item).upper() for item in manifest["knowledge_prediction"]["bridge_source_genes"]}
    strict = practical - bridge - {gene}
    if len(strict) != int(manifest["function_sets"]["strict_gene_count"]):
        raise RuntimeError("Strict function set no longer matches the frozen count")
    profile_values = []
    gene_effects = []
    module_effects = []
    all_null = None
    primary_strict_effect = 0.0
    iterations = int(settings.get("random_iterations", 1000))
    bins = int(settings.get("random_gene_match_bins", 10))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for cell_label in [primary_cell, "CD4", "CD8"]:
        for timepoint in sorted(metadata["timepoint"].astype(str).unique()):
            selected_meta = metadata[
                metadata["cell_label"].astype(str).eq(cell_label)
                & metadata["timepoint"].astype(str).eq(timepoint)
            ]
            if selected_meta.empty:
                continue
            expression = log_cpm.loc[selected_meta.index]
            strict_scores, strict_variable = _module_scores(expression, sorted(strict))
            practical_scores, practical_variable = _module_scores(expression, sorted(practical))
            values = selected_meta.reset_index()[["profile_id", "donor_id", "condition", "timepoint", "cell_label", "cell_count", "library_size"]].copy()
            gene_in_processed_axis = gene in expression.columns
            values["BTK_log2_cpm"] = expression[gene].to_numpy(float) if gene_in_processed_axis else np.nan
            values["BTK_detected"] = counts.loc[selected_meta.index, gene].to_numpy(float) > 0 if gene_in_processed_axis else False
            values["strict_module_score"] = strict_scores
            values["practical_module_score"] = practical_scores
            profile_values.extend(values.to_dict("records"))
            control_mask = selected_meta["condition"].astype(str).eq("HC").to_numpy()
            disease_mask = selected_meta["condition"].astype(str).eq("XLA").to_numpy()
            if control_mask.sum() < 1 or disease_mask.sum() < 1:
                continue
            gene_values = values["BTK_log2_cpm"].to_numpy(float)
            effect = (
                _effect(gene_values[control_mask], gene_values[disease_mask])
                if gene_in_processed_axis
                else {"effect": math.nan, "ci_low": math.nan, "ci_high": math.nan, "hedges_g": math.nan, "p_value": math.nan}
            )
            gene_effects.append(
                {
                    "cell_label": cell_label,
                    "timepoint": timepoint,
                    "gene_in_processed_axis": gene_in_processed_axis,
                    "control_donors": int(control_mask.sum()),
                    "xla_donors": int(disease_mask.sum()),
                    "control_mean_log2_cpm": float(np.nanmean(gene_values[control_mask])) if gene_in_processed_axis else math.nan,
                    "xla_mean_log2_cpm": float(np.nanmean(gene_values[disease_mask])) if gene_in_processed_axis else math.nan,
                    "xla_minus_control": effect["effect"],
                    "ci_low": effect["ci_low"],
                    "ci_high": effect["ci_high"],
                    "hedges_g": effect["hedges_g"],
                    "p_value": effect["p_value"],
                    "control_detection_fraction": float(values.loc[control_mask, "BTK_detected"].mean()),
                    "xla_detection_fraction": float(values.loc[disease_mask, "BTK_detected"].mean()),
                }
            )
            for module_name, scores, variable_genes in [
                ("strict", strict_scores, strict_variable),
                ("practical", practical_scores, practical_variable),
            ]:
                module_effect = _effect(scores[control_mask], scores[disease_mask])
                empirical_p = ""
                percentile = ""
                if cell_label == primary_cell and timepoint == primary_timepoint:
                    observed, empirical, null, matched_gene_count = _matched_null(
                        expression,
                        selected_meta["condition"],
                        variable_genes,
                        practical | bridge | {gene},
                        iterations,
                        bins,
                        seed + (0 if module_name == "strict" else 1),
                    )
                    module_effect["effect"] = observed
                    empirical_p = empirical
                    percentile = float(np.mean(np.abs(null) <= abs(observed)))
                    if module_name == "strict":
                        all_null = null
                        primary_strict_effect = observed
                else:
                    matched_gene_count = len(variable_genes)
                module_effects.append(
                    {
                        "module": module_name,
                        "cell_label": cell_label,
                        "timepoint": timepoint,
                        "control_donors": int(control_mask.sum()),
                        "xla_donors": int(disease_mask.sum()),
                        "feature_genes": len([item for item in (strict if module_name == "strict" else practical) if item in expression]),
                        "variable_genes": len(variable_genes),
                        "xla_minus_control": module_effect["effect"],
                        "ci_low": module_effect["ci_low"],
                        "ci_high": module_effect["ci_high"],
                        "hedges_g": module_effect["hedges_g"],
                        "p_value": module_effect["p_value"],
                        "expression_matched_percentile": percentile,
                        "expression_matched_empirical_p": empirical_p,
                        "matched_gene_count": matched_gene_count,
                    }
                )
    profile_table = pd.DataFrame(profile_values)
    gene_table = pd.DataFrame(gene_effects)
    module_table = pd.DataFrame(module_effects)
    write_tsv(output / "gene/candidate_gene_effects.tsv", gene_effects, list(gene_table.columns))
    write_tsv(output / "function/module_effects.tsv", module_effects, list(module_table.columns))
    write_tsv(output / "pseudobulk/donor_endpoint_values.tsv", profile_values, list(profile_table.columns))
    primary_gene = gene_table[(gene_table["cell_label"] == primary_cell) & (gene_table["timepoint"] == primary_timepoint)].iloc[0]
    primary_module = module_table[(module_table["module"] == "strict") & (module_table["cell_label"] == primary_cell) & (module_table["timepoint"] == primary_timepoint)].iloc[0]
    other_timepoints = [item for item in gene_table["timepoint"].unique() if item != primary_timepoint]
    gene_replication = gene_table[(gene_table["cell_label"] == primary_cell) & gene_table["timepoint"].isin(other_timepoints)]
    module_replication = module_table[(module_table["module"] == "strict") & (module_table["cell_label"] == primary_cell) & module_table["timepoint"].isin(other_timepoints)]
    gene_evaluable = bool(primary_gene["gene_in_processed_axis"])
    gene_direction_consistent = bool(gene_evaluable and len(gene_replication) and np.all(np.sign(gene_replication["xla_minus_control"]) == np.sign(primary_gene["xla_minus_control"])))
    module_direction_consistent = bool(len(module_replication) and np.all(np.sign(module_replication["xla_minus_control"]) == np.sign(primary_module["xla_minus_control"])))
    gene_status = (
        "insufficient_data"
        if not gene_evaluable
        else "supported" if primary_gene["p_value"] <= 0.05 and gene_direction_consistent
        else "not_supported"
    )
    module_status = "supported" if float(primary_module["expression_matched_empirical_p"] or 1.0) <= 0.05 and module_direction_consistent else "not_supported"
    if gene_status == "supported" and module_status == "supported":
        overall = "partially_supported"
    elif module_status == "supported":
        overall = "mechanism_only_supported"
    elif gene_status == "supported":
        overall = "gene_only_supported"
    elif gene_status == "insufficient_data":
        overall = "insufficient_data"
    else:
        overall = "not_supported"
    endpoint_rows = [
        {"endpoint_id": "E0", "endpoint": "BTK abnormality in predicted T cells", "status": gene_status, "effect": float(primary_gene["xla_minus_control"]) if gene_evaluable else "", "p_or_empirical_p": float(primary_gene["p_value"]) if gene_evaluable else "", "reason": "BTK was removed from the released processed RNA axis because it was not detected in these sorted T cells." if not gene_evaluable else "Balanced donor-level primary timepoint plus directional replication."},
        {"endpoint_id": "E1", "endpoint": "Predicted-cell specificity", "status": "insufficient_data", "effect": "", "p_or_empirical_p": "", "reason": "All assayed cells were sorted T cells; no valid non-T reference cell exists."},
        {"endpoint_id": "E2", "endpoint": "Strict downstream TCR signaling module", "status": module_status, "effect": float(primary_module["xla_minus_control"]), "p_or_empirical_p": float(primary_module["expression_matched_empirical_p"] or 1.0), "reason": "Candidate and all bridge-source genes removed; expression-matched random-set control."},
        {"endpoint_id": "E3", "endpoint": "Practical TCR signaling module", "status": "secondary", "effect": float(module_table[(module_table["module"] == "practical") & (module_table["cell_label"] == primary_cell) & (module_table["timepoint"] == primary_timepoint)].iloc[0]["xla_minus_control"]), "p_or_empirical_p": float(module_table[(module_table["module"] == "practical") & (module_table["cell_label"] == primary_cell) & (module_table["timepoint"] == primary_timepoint)].iloc[0]["expression_matched_empirical_p"] or 1.0), "reason": "Secondary full function set."},
        {"endpoint_id": "E4", "endpoint": "Cell composition", "status": "not_applicable", "effect": "", "p_or_empirical_p": "", "reason": "Cells were antigen-stimulated and T-cell sorted before sequencing."},
    ]
    write_tsv(output / "endpoint_results.tsv", endpoint_rows, ["endpoint_id", "endpoint", "status", "effect", "p_or_empirical_p", "reason"])
    if all_null is not None:
        write_tsv(output / "controls/strict_module_expression_matched_null.tsv", [{"iteration": index + 1, "effect": effect} for index, effect in enumerate(all_null)], ["iteration", "effect"])
        _plot_null(all_null, primary_strict_effect, output / "figures/strict_module_matched_null.png")
    _plot_donor_endpoints(profile_table[profile_table["cell_label"] == primary_cell], output / "figures/donor_level_endpoints.png", primary_timepoint)
    limitations = [
        "Only three XLA and three healthy donors were available; the six-month healthy arm has two donors.",
        "Cells were spike-peptide stimulated and enriched for activated T cells, so effects are context-specific.",
        "The same publication is present in the marker-literature corpus; this is not a fully independent cell-relation validation.",
        "No non-T cell was sequenced, so predicted-cell specificity and composition cannot be tested.",
    ]
    summary_row = {
        "case_id": manifest["case"]["case_id"],
        "disease_id": manifest["case"]["disease_id"],
        "disease_name": manifest["case"]["disease_name"],
        "gene_symbol": gene,
        "study_accession": manifest["single_cell"]["study_accession"],
        "data_tier": manifest["single_cell"]["data_tier"],
        "base_rank": manifest["ranking"]["base_rank"],
        "final_rank": manifest["ranking"]["final_rank"],
        "bridge_loo_rank": manifest["ranking"]["bridge_loo_rank"],
        "primary_timepoint": primary_timepoint,
        "candidate_gene_effect": float(primary_gene["xla_minus_control"]) if gene_evaluable else "",
        "candidate_gene_p": float(primary_gene["p_value"]) if gene_evaluable else "",
        "candidate_gene_status": gene_status,
        "cell_specificity_status": "insufficient_data",
        "strict_module_effect": float(primary_module["xla_minus_control"]),
        "strict_module_empirical_p": float(primary_module["expression_matched_empirical_p"] or 1.0),
        "strict_module_status": module_status,
        "overall_case_status": overall,
        "limitations": limitations,
    }
    write_tsv(output / "case_summary.tsv", [summary_row], list(summary_row))
    report = f"""# BTK-XLA 单细胞探索性验证

## 冻结预测

- 疾病：{manifest['case']['disease_name']}（{manifest['case']['disease_id']}）
- 候选基因：`{gene}`
- 排名：A0 第 {manifest['ranking']['base_rank']} 位，A4 第 {manifest['ranking']['final_rank']} 位，A5 Bridge-LOO 第 {manifest['ranking']['bridge_loo_rank']} 位
- 预测细胞：T cell（`CL:0000084`）
- 预测功能：T cell receptor signaling pathway（`{function_id}`）
- 数据：`E-MTAB-11845`，3 名 XLA 与 3 名健康供体；平衡主时间点由纯元数据规则选为 `{primary_timepoint}`

## 结果

- E0 候选基因：状态 `{gene_status}`。`BTK` 虽存在于原始 10x feature 定义中，但在发布的过滤后 RNA 轴中不存在，所有 donor pseudobulk 均不可检测，因此不把补零值解释成无差异。
- E1 细胞特异性：`insufficient_data`。该实验只测了分选后的 T 细胞，没有合法的非 T 参考细胞。
- E2 strict TCR module：冻结时 87 个基因；RDS 中 {int(primary_module['feature_genes'])} 个可用、{int(primary_module['variable_genes'])} 个跨供体可变。XLA - HC = {primary_module['xla_minus_control']:.4f}，表达匹配随机集经验 P = {float(primary_module['expression_matched_empirical_p'] or 1.0):.4g}，状态 `{module_status}`。
- 总体状态：`{overall}`。

## 解释边界

这是 prediction-driven exploratory case，不是盲法确认性验证。结果只适用于疫苗后、spike peptide 刺激并分选的 T 细胞。该论文已存在于 marker 文献库，因此必须视为有来源重叠；strict module 删除了 `BTK` 和全部桥接源基因，但不能消除病例选择层面的重叠。供体是统计单位，未把单个细胞当作独立重复。
"""
    (output / "report/case_report.md").write_text(report, encoding="utf-8")
    access_log_path = Path(config["_project_root"]) / "outputs/single_cell_case_exploration/preflight/expression_access_log.tsv"
    existing = list(read_tsv(access_log_path)) if access_log_path.is_file() else []
    existing.append(
        {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "stage": "post_freeze_expression_analysis",
            "resource": str(rds_path),
            "access_type": "raw_counts_and_cell_metadata",
            "expression_matrix_values_accessed": True,
            "used_for_selection": False,
            "note": f"Access occurred after verified freeze manifest {manifest_hash}.",
        }
    )
    write_tsv(access_log_path, existing, ["event_time", "stage", "resource", "access_type", "expression_matrix_values_accessed", "used_for_selection", "note"])
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256_verified": manifest_hash,
        "primary_timepoint_selection_rule": "maximize minimum donor count per condition, then lexical timepoint",
        "primary_timepoint": primary_timepoint,
        "expression_access_after_freeze": True,
        "overall_case_status": overall,
        "inputs": {"rds": input_record(rds_path), "freeze_manifest": input_record(manifest_path), "go_functions": input_record(inputs["go_functions"])},
    }
    write_json(output / "run_manifest.json", run_manifest)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    return run_manifest
