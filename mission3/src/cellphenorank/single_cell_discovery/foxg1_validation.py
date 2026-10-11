from __future__ import annotations

import gzip
import hashlib
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmread

from ..config import dump_config
from ..io import input_record, read_tsv, value, write_json, write_tsv
from .xla_validation import _effect, _function_genes, _matched_null, _module_scores, _plot_null, _verify_manifest


def _read_lines(path: str | Path) -> list[list[str]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8-sig") as handle:
        return [line.rstrip("\n").split("\t") for line in handle if line.strip()]


def _load_samples(samples: list[Mapping[str, Any]]) -> ad.AnnData:
    objects = []
    for sample in samples:
        features = _read_lines(sample["features"])
        barcodes = [row[0] for row in _read_lines(sample["barcodes"])]
        matrix = mmread(sample["matrix"]).tocsr().T
        if matrix.shape != (len(barcodes), len(features)):
            raise ValueError(f"Matrix dimensions do not match features/barcodes: {sample['sample_id']}")
        genes = [row[1] if len(row) > 1 else row[0] for row in features]
        obs = pd.DataFrame(
            {
                "sample_id": str(sample["sample_id"]),
                "condition": str(sample["condition"]),
                "technical_replicate": str(sample["technical_replicate"]),
            },
            index=[f"{sample['sample_id']}::{barcode}" for barcode in barcodes],
        )
        gene_index = ad.utils.make_index_unique(pd.Index(genes, name="gene_symbol"))
        var = pd.DataFrame(index=gene_index)
        item = ad.AnnData(X=matrix, obs=obs, var=var)
        objects.append(item)
    return ad.concat(objects, axis=0, join="inner", merge="same")


def _highly_variable_indices(matrix: sparse.csr_matrix, n_top: int) -> np.ndarray:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_square = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_square - mean * mean, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    return np.argsort(dispersion)[-min(n_top, matrix.shape[1]):]


def _cluster_and_annotate(
    data: ad.AnnData,
    marker_sets: Mapping[str, list[str]],
    n_hvg: int,
    resolution: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellphenorank-numba")
    import scanpy as sc

    hvg_indices = _highly_variable_indices(data.X.tocsr(), n_hvg)
    analysis = data[:, hvg_indices].copy()
    sc.pp.normalize_total(analysis, target_sum=10_000)
    sc.pp.log1p(analysis)
    sc.pp.scale(analysis, zero_center=False, max_value=10)
    sc.tl.pca(analysis, n_comps=min(30, analysis.n_vars - 1), zero_center=False, random_state=seed)
    sc.pp.neighbors(analysis, n_neighbors=15, n_pcs=min(30, analysis.obsm["X_pca"].shape[1]), random_state=seed)
    sc.tl.leiden(analysis, resolution=resolution, random_state=seed, key_added="cluster")
    sc.tl.umap(analysis, random_state=seed)
    data.obs["cluster"] = analysis.obs["cluster"].astype(str)
    data.obsm["X_umap"] = analysis.obsm["X_umap"]

    requested = sorted({gene.upper() for genes in marker_sets.values() for gene in genes})
    lookup = {str(gene).upper(): index for index, gene in enumerate(data.var_names)}
    present = [gene for gene in requested if gene in lookup]
    indices = [lookup[gene] for gene in present]
    libraries = np.asarray(data.X.sum(axis=1)).ravel()
    marker_matrix = data.X[:, indices].toarray()
    marker_matrix = np.log1p(marker_matrix / np.maximum(libraries[:, None], 1.0) * 10_000.0)
    marker_frame = pd.DataFrame(marker_matrix, index=data.obs_names, columns=present)
    cluster_gene = marker_frame.groupby(data.obs["cluster"], observed=True).mean()
    standard = cluster_gene.std(axis=0, ddof=0).replace(0, 1.0)
    cluster_z = (cluster_gene - cluster_gene.mean(axis=0)) / standard
    score_frame = pd.DataFrame(index=cluster_gene.index)
    for cell_type, genes in marker_sets.items():
        available = [gene.upper() for gene in genes if gene.upper() in cluster_z.columns]
        score_frame[cell_type] = cluster_z[available].mean(axis=1) if available else -np.inf
    score_frame["assigned_cell_type"] = score_frame.idxmax(axis=1)
    data.obs["cell_type"] = data.obs["cluster"].map(score_frame["assigned_cell_type"]).astype(str)
    cluster_output = score_frame.reset_index()
    return data.obs.copy(), cluster_output


def _pseudobulk(data: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = data.obs["sample_id"].astype(str) + "__" + data.obs["cell_type"].astype(str)
    levels = sorted(keys.unique())
    groups = pd.Categorical(keys, categories=levels)
    design = sparse.csr_matrix(
        (np.ones(data.n_obs), (groups.codes, np.arange(data.n_obs))),
        shape=(len(levels), data.n_obs),
    )
    matrix = design @ data.X.tocsr()
    count_frame = pd.DataFrame.sparse.from_spmatrix(matrix, index=levels, columns=data.var_names)
    rows = []
    for index, profile_id in enumerate(levels):
        mask = groups.codes == index
        first = data.obs.iloc[np.flatnonzero(mask)[0]]
        row_counts = matrix.getrow(index)
        rows.append(
            {
                "profile_id": profile_id,
                "sample_id": first["sample_id"],
                "condition": first["condition"],
                "technical_replicate": first["technical_replicate"],
                "cell_type": first["cell_type"],
                "cell_count": int(mask.sum()),
                "library_size": float(row_counts.sum()),
                "detected_gene_count": int(row_counts.getnnz()),
                **({"batch": first["batch"]} if "batch" in data.obs.columns else {}),
            }
        )
    return count_frame, pd.DataFrame(rows).set_index("profile_id")


def _separated(control: np.ndarray, disease: np.ndarray, direction: float) -> bool:
    if direction < 0:
        return bool(np.max(disease) < np.min(control))
    if direction > 0:
        return bool(np.min(disease) > np.max(control))
    return False


def _plot_umap(data: ad.AnnData, output: Path) -> None:
    cell_types = sorted(data.obs["cell_type"].unique())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, cell_type in enumerate(cell_types):
        selected = data.obs["cell_type"].eq(cell_type).to_numpy()
        axes[0].scatter(data.obsm["X_umap"][selected, 0], data.obsm["X_umap"][selected, 1], s=2, alpha=0.55, color=colors(index % 10), label=cell_type)
    axes[0].legend(markerscale=4, fontsize=7, frameon=False, loc="best")
    axes[0].set_title("Marker-based cell classes")
    for condition, color in [("control", "#2878B5"), ("foxg1_del", "#D9534F")]:
        selected = data.obs["condition"].eq(condition).to_numpy()
        axes[1].scatter(data.obsm["X_umap"][selected, 0], data.obsm["X_umap"][selected, 1], s=2, alpha=0.45, color=color, label=condition)
    axes[1].legend(markerscale=4, frameon=False)
    axes[1].set_title("Accessible GSE273180 conditions")
    for axis in axes:
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoints(values: pd.DataFrame, output: Path) -> None:
    metrics = [("FOXG1_log2_cpm", "FOXG1"), ("strict_module_score", "Strict function"), ("practical_module_score", "Practical function")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    rng = np.random.default_rng(2026)
    for axis, (metric, title) in zip(axes, metrics):
        for x, condition in enumerate(["control", "foxg1_del"]):
            subset = values[values["condition"].eq(condition)]
            points = subset[metric].to_numpy(float)
            axis.scatter(np.full(len(points), x) + rng.uniform(-0.05, 0.05, len(points)), points, s=48, color="#2878B5" if condition == "control" else "#D9534F", edgecolor="white", linewidth=0.6)
            axis.plot([x - 0.18, x + 0.18], [points.mean(), points.mean()], color="#222222", linewidth=1.4)
        axis.set_xticks([0, 1], ["Control", "FOXG1 del"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_foxg1_validation(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    for name in ["qc", "pseudobulk", "gene", "function", "composition", "controls", "figures", "report"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    settings = config["single_cell_validation"]
    manifest_path = Path(inputs["freeze_manifest"])
    manifest_hash = _verify_manifest(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    data = _load_samples(settings["samples"])
    obs, cluster_scores = _cluster_and_annotate(
        data,
        settings["cell_annotation"]["markers"],
        int(settings["cell_annotation"].get("n_highly_variable_genes", 2500)),
        float(settings["cell_annotation"].get("leiden_resolution", 0.7)),
        seed,
    )
    obs.reset_index(names="cell_id").to_csv(output / "qc/cell_annotations.tsv.gz", sep="\t", index=False, compression="gzip")
    cluster_scores.to_csv(output / "qc/cluster_marker_scores.tsv", sep="\t", index=False)
    _plot_umap(data, output / "figures/umap_cell_annotation.png")
    counts, metadata = _pseudobulk(data)
    metadata.reset_index().to_csv(output / "pseudobulk/pseudobulk_profiles.tsv", sep="\t", index=False)
    sparse_frame = counts.sparse.to_coo().tocsr()
    sparse.save_npz(output / "pseudobulk/pseudobulk_counts.npz", sparse_frame)
    pd.Series(counts.columns, name="gene_symbol").to_csv(output / "pseudobulk/pseudobulk_genes.tsv", sep="\t", index=False)
    pd.Series(counts.index, name="profile_id").to_csv(output / "pseudobulk/pseudobulk_profile_ids.tsv", sep="\t", index=False)
    dense_counts = pd.DataFrame(sparse_frame.toarray(), index=counts.index, columns=counts.columns)
    log_cpm = np.log2(dense_counts.div(metadata["library_size"], axis=0) * 1_000_000.0 + 1.0)
    candidate_gene = str(manifest["case"]["candidate_gene"]).upper()
    function_id = str(manifest["knowledge_prediction"]["primary_function_id"])
    practical = _function_genes(inputs["go_functions"], function_id)
    bridge = {str(item).upper() for item in manifest["knowledge_prediction"]["bridge_source_genes"]}
    strict = practical - bridge - {candidate_gene}
    if len(strict) != int(manifest["function_sets"]["strict_gene_count"]):
        raise RuntimeError("Strict function set no longer matches the frozen manifest")
    min_cells = int(settings.get("min_cells_per_profile", 20))
    eligible_metadata = metadata[metadata["cell_count"] >= min_cells]
    gene_rows = []
    module_rows = []
    profile_values = []
    null_values = None
    primary_cell = str(settings["primary_cell_type"])
    primary_strict_effect = math.nan
    for cell_type in sorted(eligible_metadata["cell_type"].unique()):
        selected_meta = eligible_metadata[eligible_metadata["cell_type"].eq(cell_type)]
        if set(selected_meta["condition"]) != {"control", "foxg1_del"}:
            continue
        expression = log_cpm.loc[selected_meta.index]
        control_mask = selected_meta["condition"].eq("control").to_numpy()
        disease_mask = selected_meta["condition"].eq("foxg1_del").to_numpy()
        gene_values = expression[candidate_gene].to_numpy(float)
        gene_effect = _effect(gene_values[control_mask], gene_values[disease_mask])
        gene_rows.append(
            {
                "cell_type": cell_type,
                "control_technical_replicates": int(control_mask.sum()),
                "deletion_technical_replicates": int(disease_mask.sum()),
                "control_mean_log2_cpm": float(gene_values[control_mask].mean()),
                "deletion_mean_log2_cpm": float(gene_values[disease_mask].mean()),
                "deletion_minus_control": gene_effect["effect"],
                "all_deletion_below_all_controls": _separated(gene_values[control_mask], gene_values[disease_mask], -1.0),
                "descriptive_welch_p_not_biological_inference": gene_effect["p_value"],
            }
        )
        strict_scores, strict_variable = _module_scores(expression, sorted(strict))
        practical_scores, practical_variable = _module_scores(expression, sorted(practical))
        values = selected_meta.reset_index()[["profile_id", "sample_id", "condition", "technical_replicate", "cell_type", "cell_count", "library_size"]].copy()
        values["FOXG1_log2_cpm"] = gene_values
        values["strict_module_score"] = strict_scores
        values["practical_module_score"] = practical_scores
        profile_values.extend(values.to_dict("records"))
        for module_name, scores, variable_genes, target in [
            ("strict", strict_scores, strict_variable, strict),
            ("practical", practical_scores, practical_variable, practical),
        ]:
            effect = _effect(scores[control_mask], scores[disease_mask])
            empirical_p: float | str = ""
            percentile: float | str = ""
            if cell_type == primary_cell:
                observed, empirical_p, null, _ = _matched_null(
                    expression,
                    selected_meta["condition"],
                    variable_genes,
                    practical | bridge | {candidate_gene},
                    int(settings.get("random_iterations", 1000)),
                    int(settings.get("random_gene_match_bins", 10)),
                    seed + (0 if module_name == "strict" else 1),
                    control_value="control",
                    disease_value="foxg1_del",
                )
                effect["effect"] = observed
                percentile = float(np.mean(np.abs(null) <= abs(observed)))
                if module_name == "strict":
                    null_values = null
                    primary_strict_effect = observed
            module_rows.append(
                {
                    "module": module_name,
                    "cell_type": cell_type,
                    "feature_genes": len(target & set(expression.columns)),
                    "variable_genes": len(variable_genes),
                    "deletion_minus_control": effect["effect"],
                    "all_deletion_separated_from_controls": _separated(scores[control_mask], scores[disease_mask], effect["effect"]),
                    "descriptive_welch_p_not_biological_inference": effect["p_value"],
                    "expression_matched_percentile": percentile,
                    "expression_matched_empirical_p": empirical_p,
                }
            )
    gene_table = pd.DataFrame(gene_rows)
    module_table = pd.DataFrame(module_rows)
    profile_table = pd.DataFrame(profile_values)
    write_tsv(output / "gene/candidate_gene_effects.tsv", gene_rows, list(gene_table.columns))
    write_tsv(output / "function/module_effects.tsv", module_rows, list(module_table.columns))
    write_tsv(output / "pseudobulk/technical_replicate_endpoint_values.tsv", profile_values, list(profile_table.columns))
    composition = (
        data.obs.groupby(["sample_id", "condition", "cell_type"], observed=True).size().rename("cell_count").reset_index()
    )
    composition["cell_fraction"] = composition["cell_count"] / composition.groupby("sample_id")["cell_count"].transform("sum")
    composition.to_csv(output / "composition/cell_composition.tsv", sep="\t", index=False)
    primary_gene = gene_table[gene_table["cell_type"].eq(primary_cell)].iloc[0]
    primary_module = module_table[(module_table["cell_type"].eq(primary_cell)) & (module_table["module"].eq("strict"))].iloc[0]
    gene_status = "descriptively_supported" if bool(primary_gene["all_deletion_below_all_controls"]) and primary_gene["deletion_minus_control"] < 0 else "not_supported"
    reference_effects = gene_table[gene_table["cell_type"].isin(settings["reference_cell_types"])]
    specificity_rank = int((reference_effects["deletion_minus_control"].abs() > abs(primary_gene["deletion_minus_control"])).sum() + 1)
    specificity_status = "descriptively_supported" if specificity_rank == 1 else "not_supported"
    module_empirical = float(primary_module["expression_matched_empirical_p"] or 1.0)
    module_status = "descriptively_supported" if module_empirical <= 0.05 and bool(primary_module["all_deletion_separated_from_controls"]) else "not_supported"
    if gene_status == "descriptively_supported" and module_status == "descriptively_supported":
        overall = "partially_supported"
    elif gene_status == "descriptively_supported":
        overall = "gene_only_supported"
    elif module_status == "descriptively_supported":
        overall = "mechanism_only_supported"
    else:
        overall = "not_supported"
    endpoint_rows = [
        {"endpoint_id": "E0", "endpoint": "FOXG1 decrease in predicted excitatory neurons", "status": gene_status, "effect": primary_gene["deletion_minus_control"], "evidence": "technical-replicate ranges only"},
        {"endpoint_id": "E1", "endpoint": "Predicted-cell specificity", "status": specificity_status, "effect": specificity_rank, "evidence": "rank of absolute FOXG1 effect among frozen cell classes"},
        {"endpoint_id": "E2", "endpoint": "Strict downstream transcription-repression module", "status": module_status, "effect": primary_module["deletion_minus_control"], "evidence": f"expression-matched empirical P={module_empirical:.4g}"},
        {"endpoint_id": "E3", "endpoint": "Practical function module", "status": "secondary", "effect": module_table[(module_table["cell_type"].eq(primary_cell)) & (module_table["module"].eq("practical"))].iloc[0]["deletion_minus_control"], "evidence": "technical-replicate descriptive effect"},
        {"endpoint_id": "E4", "endpoint": "Cell composition", "status": "descriptive_only", "effect": "", "evidence": "one line per condition"},
    ]
    write_tsv(output / "endpoint_results.tsv", endpoint_rows, ["endpoint_id", "endpoint", "status", "effect", "evidence"])
    if null_values is not None:
        write_tsv(output / "controls/strict_module_expression_matched_null.tsv", [{"iteration": i + 1, "effect": item} for i, item in enumerate(null_values)], ["iteration", "effect"])
        _plot_null(null_values, primary_strict_effect, output / "figures/strict_module_matched_null.png")
    predicted_values = profile_table[profile_table["cell_type"].eq(primary_cell)]
    _plot_endpoints(predicted_values, output / "figures/predicted_cell_endpoints.png")
    summary = {
        "case_id": manifest["case"]["case_id"],
        "disease_id": manifest["case"]["disease_id"],
        "disease_name": manifest["case"]["disease_name"],
        "gene_symbol": candidate_gene,
        "study_accession": manifest["single_cell"]["study_accession"],
        "data_tier": manifest["single_cell"]["data_tier"],
        "evidence_level": "descriptive_single_patient_and_control_cell_line",
        "base_rank": manifest["ranking"]["base_rank"],
        "final_rank": manifest["ranking"]["final_rank"],
        "bridge_loo_rank": manifest["ranking"]["bridge_loo_rank"],
        "candidate_gene_effect": primary_gene["deletion_minus_control"],
        "candidate_gene_status": gene_status,
        "cell_specificity_rank": specificity_rank,
        "cell_specificity_status": specificity_status,
        "strict_module_effect": primary_module["deletion_minus_control"],
        "strict_module_empirical_p": module_empirical,
        "strict_module_status": module_status,
        "overall_case_status": overall,
        "limitations": "one biological cell line per condition; two technical replicates; outcome known from publication; regional cell mapping approximate",
    }
    write_tsv(output / "case_summary.tsv", [summary], list(summary))
    report = f"""# FOXG1 缺失脑类器官单细胞探索性验证

## 冻结预测

- 疾病：{manifest['case']['disease_name']}（{manifest['case']['disease_id']}）
- 候选基因：`FOXG1`
- 排名：A0 第 {manifest['ranking']['base_rank']} 位，A4 第 {manifest['ranking']['final_rank']} 位，A5 第 {manifest['ranking']['bridge_loo_rank']} 位
- 预测细胞：`CL:2000028`，在本数据中预先映射为 DL-EN/UL-EN，并在分析中合并为 `excitatory_neuron`
- 预测功能：negative regulation of DNA-templated transcription（`{function_id}`）

## 结果

- E0：兴奋性神经元中 `FOXG1` 的 deletion - control 效应为 {primary_gene['deletion_minus_control']:.4f} log2(CPM+1)，状态 `{gene_status}`。
- E1：该效应绝对值在预冻结细胞类中的排名为第 {specificity_rank}，状态 `{specificity_status}`。
- E2：strict module 删除 `FOXG1` 和全部桥接源后理论为 {len(strict)} 个基因；数据中 {int(primary_module['feature_genes'])} 个存在、{int(primary_module['variable_genes'])} 个跨技术重复可变。module 效应为 {primary_module['deletion_minus_control']:.4f}，表达匹配经验 P={module_empirical:.4g}，状态 `{module_status}`。
- 总体状态：`{overall}`，证据等级为单病例细胞系描述性证据。

## 解释边界

公开 GEO 仅包含一个健康细胞系和一个 FOXG1 缺失患者细胞系，各 2 个技术重复。技术重复用于检查方向稳定性，不能替代独立供体。病例来自第三 Pareto 层，是 XLA 主病例之后的新 exploratory follow-up；原论文已提示 FOXG1 下调，因此这不是盲法确认性验证。区域映射从 cerebellar glutamatergic subtype 到 cortical excitatory neuron 也是近似映射。
"""
    (output / "report/case_report.md").write_text(report, encoding="utf-8")
    access_log_path = Path(config["_project_root"]) / "outputs/single_cell_case_exploration/preflight/expression_access_log.tsv"
    access_rows = list(read_tsv(access_log_path))
    access_rows.append(
        {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "stage": "post_followup_freeze_expression_analysis",
            "resource": "GSE273180 filtered count matrices",
            "access_type": "raw_counts",
            "expression_matrix_values_accessed": True,
            "used_for_selection": False,
            "note": f"Access occurred after verified follow-up freeze manifest {manifest_hash}.",
        }
    )
    write_tsv(access_log_path, access_rows, ["event_time", "stage", "resource", "access_type", "expression_matrix_values_accessed", "used_for_selection", "note"])
    sample_inputs = {
        f"{sample['sample_id']}_{kind}": input_record(sample[kind])
        for sample in settings["samples"] for kind in ["matrix", "features", "barcodes"]
    }
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256_verified": manifest_hash,
        "expression_access_after_followup_freeze": True,
        "biological_replicates_per_condition": 1,
        "technical_replicates_per_condition": 2,
        "overall_case_status": overall,
        "inputs": {"freeze_manifest": input_record(manifest_path), "go_functions": input_record(inputs["go_functions"]), **sample_inputs},
    }
    write_json(output / "run_manifest.json", run_manifest)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    return run_manifest
