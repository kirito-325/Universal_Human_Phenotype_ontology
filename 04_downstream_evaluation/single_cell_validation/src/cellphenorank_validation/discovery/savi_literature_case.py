from __future__ import annotations

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
from scipy.stats import ttest_ind
from scipy.stats import t as student_t

from cellphenorank.config import dump_config
from cellphenorank.io import input_record, read_tsv, value, write_json, write_tsv
from .foxg1_validation import _cluster_and_annotate, _pseudobulk
from .xla_validation import _effect, _module_scores


def load_signature(path: str | Path) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8-sig").splitlines()
    genes = []
    for line in rows:
        item = line.split(",", 1)[0].strip().upper()
        if not item or item == "UPR" or item.startswith("PMID "):
            continue
        genes.append(item)
    return sorted(set(genes))


def _verify_freeze(path: Path) -> str:
    checksum_path = path.with_suffix(".sha256")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != observed:
        raise RuntimeError("Literature-first SAVI freeze manifest checksum mismatch")
    return observed


def _load_raw_samples(
    samples: list[Mapping[str, Any]], settings: Mapping[str, Any]
) -> tuple[ad.AnnData, list[dict[str, Any]]]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellphenorank-numba")
    import scanpy as sc

    objects = []
    qc_rows = []
    for sample in samples:
        source = Path(sample["file"])
        item = sc.read_10x_h5(source, gex_only=True)
        item.var_names_make_unique()
        matrix = item.X.tocsr()
        total_counts = np.asarray(matrix.sum(axis=1)).ravel()
        detected_genes = matrix.getnnz(axis=1)
        mt_indices = np.asarray(
            [str(gene).upper().startswith("MT-") for gene in item.var_names]
        )
        mt_counts = (
            np.asarray(matrix[:, mt_indices].sum(axis=1)).ravel()
            if mt_indices.any()
            else np.zeros(item.n_obs)
        )
        mt_fraction = mt_counts / np.maximum(total_counts, 1.0)
        keep = (
            (detected_genes >= int(settings["minimum_genes_per_cell"]))
            & (detected_genes <= int(settings["maximum_genes_per_cell"]))
            & (total_counts >= int(settings["minimum_counts_per_cell"]))
            & (mt_fraction <= float(settings["maximum_mitochondrial_fraction"]))
        )
        filtered = item[keep].copy()
        filtered.obs_names = [
            f"{sample['sample_id']}::{barcode}" for barcode in filtered.obs_names
        ]
        filtered.obs["sample_id"] = str(sample["sample_id"])
        filtered.obs["condition"] = str(sample["condition"])
        filtered.obs["technical_replicate"] = str(sample["sample_id"])
        filtered.obs["batch"] = str(sample["batch"])
        filtered.obs["total_counts"] = total_counts[keep]
        filtered.obs["detected_genes"] = detected_genes[keep]
        filtered.obs["mt_fraction"] = mt_fraction[keep]
        objects.append(filtered)
        qc_rows.append(
            {
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "input_barcodes": item.n_obs,
                "retained_cells": filtered.n_obs,
                "retained_fraction": filtered.n_obs / item.n_obs,
                "median_counts": float(np.median(total_counts[keep])),
                "median_detected_genes": float(np.median(detected_genes[keep])),
                "median_mt_fraction": float(np.median(mt_fraction[keep])),
            }
        )
    return ad.concat(objects, axis=0, join="inner", merge="same"), qc_rows


def _cell_signature_score(
    data: ad.AnnData, genes: list[str], cell_mask: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    lookup = {str(gene).upper(): index for index, gene in enumerate(data.var_names)}
    present = [gene for gene in genes if gene in lookup]
    indices = [lookup[gene] for gene in present]
    matrix = data.X[cell_mask][:, indices].toarray()
    libraries = np.asarray(data.X[cell_mask].sum(axis=1)).ravel()
    normalized = np.log1p(matrix / np.maximum(libraries[:, None], 1.0) * 10_000.0)
    return normalized.mean(axis=1), present


def _condition_effect(
    values: np.ndarray, conditions: pd.Series, control: str = "control", disease: str = "savi"
) -> dict[str, float]:
    control_mask = conditions.astype(str).eq(control).to_numpy()
    disease_mask = conditions.astype(str).eq(disease).to_numpy()
    return _effect(values[control_mask], values[disease_mask])


def _batch_adjusted_effect(
    values: np.ndarray, conditions: pd.Series, batches: pd.Series
) -> dict[str, float]:
    outcome = np.asarray(values, dtype=float)
    disease = conditions.astype(str).eq("savi").to_numpy(dtype=float)
    batch_dummies = pd.get_dummies(batches.astype(str), drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(outcome)), disease, batch_dummies.to_numpy(dtype=float)]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients
    degrees = len(outcome) - design.shape[1]
    covariance = np.linalg.pinv(design.T @ design)
    variance = float(residuals @ residuals / degrees) if degrees > 0 else 0.0
    standard_error = math.sqrt(max(0.0, variance * covariance[1, 1]))
    statistic = float(coefficients[1] / standard_error) if standard_error else 0.0
    p_value = float(2 * student_t.sf(abs(statistic), degrees)) if degrees > 0 else 1.0
    return {
        "effect": float(coefficients[1]),
        "standard_error": standard_error,
        "p_value": p_value,
    }


def _plot_umap(data: ad.AnnData, output: Path) -> None:
    cell_types = sorted(data.obs["cell_type"].unique())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, cell_type in enumerate(cell_types):
        selected = data.obs["cell_type"].eq(cell_type).to_numpy()
        axes[0].scatter(
            data.obsm["X_umap"][selected, 0],
            data.obsm["X_umap"][selected, 1],
            s=1.4,
            alpha=0.45,
            color=cmap(index % 10),
            label=cell_type,
        )
    axes[0].legend(markerscale=5, fontsize=7, frameon=False)
    axes[0].set_title("Recomputed major PBMC cell classes")
    for condition, color in [("control", "#2878B5"), ("savi", "#D9534F")]:
        selected = data.obs["condition"].eq(condition).to_numpy()
        axes[1].scatter(
            data.obsm["X_umap"][selected, 0],
            data.obsm["X_umap"][selected, 1],
            s=1.4,
            alpha=0.35,
            color=color,
            label=condition,
        )
    axes[1].legend(markerscale=5, frameon=False)
    axes[1].set_title("Untreated SAVI and healthy controls")
    for axis in axes:
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_validation(
    monocyte_values: pd.DataFrame,
    high_state: pd.DataFrame,
    composition: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    panels = [
        (axes[0, 0], monocyte_values, "upr_score", "Monocyte UPR score"),
        (axes[0, 1], high_state, "high_upr_fraction", "High-UPR monocyte fraction"),
        (axes[1, 0], monocyte_values, "inflammatory_score", "Monocyte CCL3/CCL4/IL6 score"),
        (axes[1, 1], composition, "effector_T_fraction", "Effector T fraction of PBMC"),
    ]
    rng = np.random.default_rng(2026)
    for axis, frame, column, title in panels:
        for x, condition in enumerate(["control", "savi"]):
            values = frame.loc[frame["condition"].eq(condition), column].to_numpy(float)
            axis.scatter(
                np.full(len(values), x) + rng.uniform(-0.06, 0.06, len(values)),
                values,
                s=44,
                color="#2878B5" if condition == "control" else "#D9534F",
                edgecolor="white",
                linewidth=0.6,
            )
            axis.plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="#222222", linewidth=1.5)
        axis.set_xticks([0, 1], ["Control", "SAVI"])
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ranking_rows(a4_path: str | Path, a5_path: str | Path, disease_id: str, gene: str) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    disease_rows = [row for row in read_tsv(a4_path) if value(row, "disease_id") == disease_id]
    target = next(row for row in disease_rows if value(row, "gene_symbol").upper() == gene)
    bridge = next(
        row for row in read_tsv(a5_path)
        if value(row, "disease_id") == disease_id and value(row, "gene_symbol").upper() == gene
    )
    by_gene = {value(row, "gene_symbol"): row for row in disease_rows}
    top_genes = set(
        value(row, "gene_symbol")
        for row in sorted(disease_rows, key=lambda row: int(float(value(row, "base_rank"))))[:10]
    ).union(
        value(row, "gene_symbol")
        for row in sorted(disease_rows, key=lambda row: int(float(value(row, "final_rank"))))[:10]
    )
    comparison = [
        {
            "gene_symbol": item,
            "base_rank": int(float(value(by_gene[item], "base_rank"))),
            "a4_rank": int(float(value(by_gene[item], "final_rank"))),
            "rank_change": int(float(value(by_gene[item], "base_rank"))) - int(float(value(by_gene[item], "final_rank"))),
            "is_causal_gene": item == gene,
        }
        for item in top_genes
    ]
    comparison.sort(key=lambda row: (row["a4_rank"], row["base_rank"], row["gene_symbol"]))
    return target, bridge, comparison


def _plot_rank_change(rows: list[dict[str, Any]], gene: str, output: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["base_rank"], row["gene_symbol"]))
    fig, axis = plt.subplots(figsize=(7.2, max(4.2, 0.35 * len(ordered))))
    for y, row in enumerate(ordered):
        color = "#C23B22" if row["gene_symbol"] == gene else "#7A8585"
        axis.plot([row["base_rank"], row["a4_rank"]], [y, y], color=color, linewidth=1.5, alpha=0.8)
        axis.scatter([row["base_rank"]], [y], color="#2878B5", s=28)
        axis.scatter([row["a4_rank"]], [y], color=color, s=36)
    axis.set_yticks(range(len(ordered)), [row["gene_symbol"] for row in ordered])
    axis.invert_yaxis()
    axis.invert_xaxis()
    axis.set_xlabel("Rank (smaller is better)")
    axis.set_title("SAVI candidate ranks: A0 versus A4")
    axis.scatter([], [], color="#2878B5", s=30, label="A0")
    axis.scatter([], [], color="#C23B22", s=36, label="A4 causal gene")
    axis.scatter([], [], color="#7A8585", s=36, label="A4 other genes")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_savi_literature_case(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    for name in ["discovery", "qc", "pseudobulk", "cellular", "ranking", "figures", "report"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    settings = config["single_cell_validation"]
    manifest_path = Path(inputs["literature_case_manifest"])
    freeze_hash = _verify_freeze(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    signature = load_signature(inputs["signature_file"])
    data, qc_rows = _load_raw_samples(settings["samples"], settings)
    write_tsv(output / "qc/sample_qc.tsv", qc_rows, list(qc_rows[0]))
    obs, cluster_scores = _cluster_and_annotate(
        data,
        settings["cell_annotation"]["markers"],
        int(settings["highly_variable_genes"]),
        float(settings["leiden_resolution"]),
        int(config["project"]["random_seed"]),
    )
    obs.reset_index(names="cell_id").to_csv(output / "qc/cell_annotations.tsv.gz", sep="\t", index=False, compression="gzip")
    cluster_scores.to_csv(output / "qc/cluster_marker_scores.tsv", sep="\t", index=False)
    _plot_umap(data, output / "figures/umap_major_cell_types.png")

    counts, pb_meta = _pseudobulk(data)
    pb_matrix = counts.sparse.to_coo().tocsr()
    from scipy import sparse as scipy_sparse
    scipy_sparse.save_npz(output / "pseudobulk/pseudobulk_counts.npz", pb_matrix)
    pd.Series(counts.columns, name="gene_symbol").to_csv(output / "pseudobulk/genes.tsv", sep="\t", index=False)
    pd.Series(counts.index, name="profile_id").to_csv(output / "pseudobulk/profile_ids.tsv", sep="\t", index=False)
    pb_meta.reset_index().to_csv(output / "pseudobulk/profiles.tsv", sep="\t", index=False)
    dense_counts = pd.DataFrame(pb_matrix.toarray(), index=counts.index, columns=counts.columns)
    log_cpm = np.log2(dense_counts.div(pb_meta["library_size"], axis=0) * 1_000_000.0 + 1.0)
    eligible_meta = pb_meta[pb_meta["cell_count"] >= int(settings["minimum_cells_per_donor_cell_type"])]

    cell_effect_rows = []
    donor_values = []
    for cell_type in sorted(eligible_meta["cell_type"].unique()):
        selected_meta = eligible_meta[eligible_meta["cell_type"].eq(cell_type)]
        if set(selected_meta["condition"]) != {"control", "savi"}:
            continue
        if selected_meta.groupby("condition")["sample_id"].nunique().min() < int(settings["minimum_donors_per_condition"]):
            continue
        expression = log_cpm.loc[selected_meta.index]
        upr_scores, upr_variable = _module_scores(expression, signature)
        inflammatory_scores, inflammatory_variable = _module_scores(
            expression, settings["endpoint_genes"]["inflammatory_module"]
        )
        upr_effect = _condition_effect(upr_scores, selected_meta["condition"])
        inflammatory_effect = _condition_effect(inflammatory_scores, selected_meta["condition"])
        marker = settings["endpoint_genes"]["isr_marker"]
        marker_values = expression[marker].to_numpy(float)
        marker_effect = _condition_effect(marker_values, selected_meta["condition"])
        upr_adjusted = _batch_adjusted_effect(
            upr_scores, selected_meta["condition"], selected_meta["batch"]
        )
        inflammatory_adjusted = _batch_adjusted_effect(
            inflammatory_scores, selected_meta["condition"], selected_meta["batch"]
        )
        marker_adjusted = _batch_adjusted_effect(
            marker_values, selected_meta["condition"], selected_meta["batch"]
        )
        cell_effect_rows.append(
            {
                "cell_type": cell_type,
                "control_donors": int((selected_meta["condition"] == "control").sum()),
                "savi_donors": int((selected_meta["condition"] == "savi").sum()),
                "upr_feature_genes": len([gene for gene in signature if gene in expression.columns]),
                "upr_variable_genes": len(upr_variable),
                "upr_effect": upr_effect["effect"],
                "upr_ci_low": upr_effect["ci_low"],
                "upr_ci_high": upr_effect["ci_high"],
                "upr_p_value": upr_effect["p_value"],
                "upr_batch_adjusted_effect": upr_adjusted["effect"],
                "upr_batch_adjusted_p_value": upr_adjusted["p_value"],
                "ppp1r15a_effect": marker_effect["effect"],
                "ppp1r15a_p_value": marker_effect["p_value"],
                "ppp1r15a_batch_adjusted_effect": marker_adjusted["effect"],
                "ppp1r15a_batch_adjusted_p_value": marker_adjusted["p_value"],
                "inflammatory_variable_genes": len(inflammatory_variable),
                "inflammatory_effect": inflammatory_effect["effect"],
                "inflammatory_p_value": inflammatory_effect["p_value"],
                "inflammatory_batch_adjusted_effect": inflammatory_adjusted["effect"],
                "inflammatory_batch_adjusted_p_value": inflammatory_adjusted["p_value"],
            }
        )
        frame = selected_meta.reset_index()[["profile_id", "sample_id", "condition", "batch", "cell_type", "cell_count", "library_size"]].copy()
        frame["upr_score"] = upr_scores
        frame["PPP1R15A_log2_cpm"] = marker_values
        frame["inflammatory_score"] = inflammatory_scores
        donor_values.extend(frame.to_dict("records"))
    cell_effect_table = pd.DataFrame(cell_effect_rows)
    donor_table = pd.DataFrame(donor_values)
    write_tsv(output / "cellular/cell_type_effects.tsv", cell_effect_rows, list(cell_effect_table.columns))
    write_tsv(output / "cellular/donor_pseudobulk_scores.tsv", donor_values, list(donor_table.columns))

    monocyte_mask = data.obs["cell_type"].eq(settings["primary_cell_type"]).to_numpy()
    monocyte_scores, cell_signature_genes = _cell_signature_score(data, signature, monocyte_mask)
    monocyte_obs = data.obs.loc[monocyte_mask, ["sample_id", "condition"]].copy()
    monocyte_obs["upr_cell_score"] = monocyte_scores
    control_threshold = float(
        monocyte_obs.loc[monocyte_obs["condition"].eq("control"), "upr_cell_score"].quantile(
            float(settings["high_upr_control_quantile"])
        )
    )
    monocyte_obs["high_upr"] = monocyte_obs["upr_cell_score"] > control_threshold
    high_state = monocyte_obs.groupby(["sample_id", "condition"], observed=True).agg(
        monocyte_cells=("high_upr", "size"), high_upr_cells=("high_upr", "sum")
    ).reset_index()
    high_state["high_upr_fraction"] = high_state["high_upr_cells"] / high_state["monocyte_cells"]
    high_effect = _condition_effect(high_state["high_upr_fraction"].to_numpy(float), high_state["condition"])
    batch_by_sample = data.obs.groupby("sample_id")["batch"].first()
    high_state["batch"] = high_state["sample_id"].map(batch_by_sample)
    high_adjusted = _batch_adjusted_effect(
        high_state["high_upr_fraction"].to_numpy(float),
        high_state["condition"],
        high_state["batch"],
    )
    high_state.to_csv(output / "cellular/high_upr_monocyte_fraction.tsv", sep="\t", index=False)

    composition = data.obs.groupby(["sample_id", "condition", "cell_type"], observed=True).size().rename("cell_count").reset_index()
    totals = composition.groupby("sample_id")["cell_count"].sum()
    effector = composition[composition["cell_type"].eq("effector_T")].set_index("sample_id")["cell_count"]
    composition_summary = pd.DataFrame(
        {
            "condition": data.obs.groupby("sample_id")["condition"].first(),
            "batch": data.obs.groupby("sample_id")["batch"].first(),
            "total_cells": totals,
            "effector_T_cells": effector.reindex(totals.index, fill_value=0),
        }
    ).reset_index()
    composition_summary["effector_T_fraction"] = composition_summary["effector_T_cells"] / composition_summary["total_cells"]
    effector_effect = _condition_effect(composition_summary["effector_T_fraction"].to_numpy(float), composition_summary["condition"])
    effector_adjusted = _batch_adjusted_effect(
        composition_summary["effector_T_fraction"].to_numpy(float),
        composition_summary["condition"],
        composition_summary["batch"],
    )
    composition.to_csv(output / "cellular/cell_composition.tsv", sep="\t", index=False)
    composition_summary.to_csv(output / "cellular/effector_t_fraction.tsv", sep="\t", index=False)

    primary = cell_effect_table[cell_effect_table["cell_type"].eq(settings["primary_cell_type"])].iloc[0]
    other = cell_effect_table[cell_effect_table["cell_type"].isin(settings["other_major_cell_types"])]
    specificity_did = float(primary["upr_effect"] - other["upr_effect"].median())
    e0_status = "supported" if primary["upr_effect"] > 0 and primary["upr_p_value"] <= 0.05 else "not_supported"
    e0b_status = "supported" if high_effect["effect"] > 0 else "not_supported"
    e1_status = "supported" if specificity_did > 0 else "not_supported"
    e2_status = "supported" if primary["ppp1r15a_effect"] > 0 and primary["ppp1r15a_p_value"] <= 0.05 else "not_supported"
    e3_status = "supported" if primary["inflammatory_effect"] > 0 and primary["inflammatory_p_value"] <= 0.05 else "not_supported"
    e4_status = "supported" if effector_effect["effect"] < 0 and effector_effect["p_value"] <= 0.05 else "not_supported"

    target_rank, bridge_rank, ranking_comparison = _ranking_rows(
        inputs["a4_rankings"], inputs["a5_rankings"], manifest["case"]["disease_id"], manifest["case"]["gene_symbol"]
    )
    base_rank = int(float(value(target_rank, "base_rank")))
    final_rank = int(float(value(target_rank, "final_rank")))
    a5_rank = int(float(value(bridge_rank, "final_rank")))
    rank_status = "supported" if final_rank < base_rank else "not_supported"
    write_tsv(output / "ranking/top_candidates_before_after.tsv", ranking_comparison, ["gene_symbol", "base_rank", "a4_rank", "rank_change", "is_causal_gene"])
    ranking_summary = {
        "disease_id": manifest["case"]["disease_id"],
        "gene_symbol": manifest["case"]["gene_symbol"],
        "a0_rank": base_rank,
        "a4_rank": final_rank,
        "a5_bridge_loo_rank": a5_rank,
        "rank_change": base_rank - final_rank,
        "a4_cpo_enrichment_pvalue": value(target_rank, "cpo_enrichment_pvalue"),
        "a4_cpo_enrichment_fdr": value(target_rank, "cpo_enrichment_fdr"),
        "a4_supported_hpo_count": value(target_rank, "supported_hpo_count"),
        "a4_supported_hpo_fraction": value(target_rank, "supported_hpo_fraction"),
        "a4_supported_cell_count": value(target_rank, "supported_cell_count"),
        "a4_supported_function_count": value(target_rank, "supported_function_count"),
        "bridge_dependency_status": "fully_dependent" if a5_rank >= base_rank else "stable_or_partial",
        "rank_improvement_status": rank_status,
    }
    write_tsv(output / "ranking/sting1_rank_change.tsv", [ranking_summary], list(ranking_summary))
    _plot_rank_change(ranking_comparison, manifest["case"]["gene_symbol"], output / "figures/rank_change.png")

    endpoint_rows = [
        {"endpoint_id": "E0", "endpoint": "monocyte UPR donor-pseudobulk score", "effect": primary["upr_effect"], "p_value": primary["upr_p_value"], "batch_adjusted_effect": primary["upr_batch_adjusted_effect"], "batch_adjusted_p_value": primary["upr_batch_adjusted_p_value"], "status": e0_status},
        {"endpoint_id": "E0b", "endpoint": "high-UPR monocyte donor fraction", "effect": high_effect["effect"], "p_value": high_effect["p_value"], "batch_adjusted_effect": high_adjusted["effect"], "batch_adjusted_p_value": high_adjusted["p_value"], "status": e0b_status},
        {"endpoint_id": "E1", "endpoint": "monocyte-specific UPR difference-in-differences", "effect": specificity_did, "p_value": "", "status": e1_status},
        {"endpoint_id": "E2", "endpoint": "monocyte PPP1R15A expression", "effect": primary["ppp1r15a_effect"], "p_value": primary["ppp1r15a_p_value"], "batch_adjusted_effect": primary["ppp1r15a_batch_adjusted_effect"], "batch_adjusted_p_value": primary["ppp1r15a_batch_adjusted_p_value"], "status": e2_status},
        {"endpoint_id": "E3", "endpoint": "monocyte CCL3/CCL4/IL6 module", "effect": primary["inflammatory_effect"], "p_value": primary["inflammatory_p_value"], "batch_adjusted_effect": primary["inflammatory_batch_adjusted_effect"], "batch_adjusted_p_value": primary["inflammatory_batch_adjusted_p_value"], "status": e3_status},
        {"endpoint_id": "E4", "endpoint": "effector T fraction of PBMC", "effect": effector_effect["effect"], "p_value": effector_effect["p_value"], "batch_adjusted_effect": effector_adjusted["effect"], "batch_adjusted_p_value": effector_adjusted["p_value"], "status": e4_status},
        {"endpoint_id": "R0", "endpoint": "STING1 A0 to A4 rank improvement", "effect": base_rank - final_rank, "p_value": value(target_rank, "cpo_enrichment_pvalue"), "batch_adjusted_effect": "", "batch_adjusted_p_value": "", "status": rank_status},
    ]
    write_tsv(output / "endpoint_results.tsv", endpoint_rows, ["endpoint_id", "endpoint", "effect", "p_value", "batch_adjusted_effect", "batch_adjusted_p_value", "status"])
    monocyte_values = donor_table[donor_table["cell_type"].eq(settings["primary_cell_type"])]
    _plot_validation(monocyte_values, high_state, composition_summary, output / "figures/cellular_validation.png")

    chain_rows = [
        {"step": 1, "from": "STING1 gain-of-function", "to": "constitutive STING activation", "evidence_type": "human genetics and functional studies", "source": "https://pubmed.ncbi.nlm.nih.gov/25029335/"},
        {"step": 2, "from": "constitutive STING activation", "to": "high-UPR/ISR inflammatory monocytes", "evidence_type": "published scRNA-seq plus current raw-count reproduction", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10772457/"},
        {"step": 3, "from": "abnormal monocytes and T-cell stress/death", "to": "lymphopenia and systemic inflammation", "evidence_type": "single-cell association and mechanistic inference", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10772457/"},
        {"step": 4, "from": "HP:0001888 and inflammatory HPO features", "to": "OMIM:615934 SAVI", "evidence_type": "frozen human disease profile", "source": "https://hpo.jax.org/"},
    ]
    write_tsv(output / "discovery/evidence_chain.tsv", chain_rows, ["step", "from", "to", "evidence_type", "source"])
    search_rows = [
        {"query": "rare disease STING1 SAVI single-cell RNA-seq patient control", "source": "PubMed/PMC", "selected": True, "reason": "exact causal gene, 5 untreated patients, 7 controls, public per-sample raw counts"},
        {"query": "GSE226598 raw 10x H5", "source": "NCBI GEO", "selected": True, "reason": "raw H5 and donor metadata publicly downloadable"},
    ]
    write_tsv(output / "discovery/search_audit.tsv", search_rows, ["query", "source", "selected", "reason"])

    reproduced = e0_status == "supported" or e0b_status == "supported"
    overall = "case_supported" if reproduced and rank_status == "supported" else "case_not_supported"
    report = f"""# SAVI-STING1：文献先行、原始数据复算案例

## 病例选择

本病例先按单细胞证据与数据可复现性选择，再检查当前排名。原始研究包含 5 名未治疗 SAVI 患者和 7 名健康对照，并公开每个样本的 10x H5 counts（GSE226598）。本次从 raw counts 重新执行 QC、固定 marker 细胞注释、donor pseudobulk 和终点统计，没有使用论文的 SAVI-vs-control DEG 表作为结果。

## 证据链

```text
STING1 gain-of-function
  -> constitutive STING activation
  -> disease-associated monocytes with elevated UPR/ISR and inflammation
  -> T-cell activation/senescence/apoptosis and reduced effector lymphocytes
  -> lymphopenia/systemic inflammation HPO features
  -> OMIM:615934 SAVI
```

其中“异常单核细胞导致临床表型”是单细胞关联与机制推断，不应写成已证明的单向人体因果链。

## 原始数据复算

- 输入：{sum(row['input_barcodes'] for row in qc_rows):,} 个 raw barcodes；QC 后 {data.n_obs:,} 个细胞。
- UPR signature：来源于论文补充材料引用的外部 UPR signature，共 {len(signature)} 个基因；本数据单核细胞中可用 {len(cell_signature_genes)} 个。
- E0 单核细胞 UPR：SAVI-control 效应 {primary['upr_effect']:.4f}，95% CI [{primary['upr_ci_low']:.4f}, {primary['upr_ci_high']:.4f}]，Welch P={primary['upr_p_value']:.4g}，`{e0_status}`。
- E0b high-UPR 单核细胞比例：健康对照第 95 百分位阈值 {control_threshold:.4f}；SAVI-control 供体比例效应 {high_effect['effect']:.4f}，P={high_effect['p_value']:.4g}；批次校正效应 {high_adjusted['effect']:.4f}，P={high_adjusted['p_value']:.4g}，`{e0b_status}`。
- E1 单核细胞特异性 DiD：{specificity_did:.4f}，`{e1_status}`。
- E2 `PPP1R15A`：效应 {primary['ppp1r15a_effect']:.4f}，P={primary['ppp1r15a_p_value']:.4g}，`{e2_status}`。
- E3 `CCL3/CCL4/IL6`：效应 {primary['inflammatory_effect']:.4f}，P={primary['inflammatory_p_value']:.4g}；批次校正效应 {primary['inflammatory_batch_adjusted_effect']:.4f}，P={primary['inflammatory_batch_adjusted_p_value']:.4g}，`{e3_status}`。
- E4 effector T fraction：SAVI-control 效应 {effector_effect['effect']:.4f}，P={effector_effect['p_value']:.4g}；批次校正效应 {effector_adjusted['effect']:.4f}，P={effector_adjusted['p_value']:.4g}，`{e4_status}`。

所有统计以 donor 为单位，没有把单个细胞作为独立重复。

## 当前预测排名

- A0 current Exomiser 2602 PhenoDigm：`STING1` 第 {base_rank} 位。
- A4 Extended CPO：`STING1` 第 {final_rank} 位，上升 {base_rank - final_rank} 位。
- A4 CPO enrichment：P={value(target_rank, 'cpo_enrichment_pvalue')}，FDR={value(target_rank, 'cpo_enrichment_fdr')}。
- A5 Bridge-LOO：第 {a5_rank} 位。

因此，A4 中的排名提升为 `{rank_status}`。但 A5 回到基础排名，说明这次提升依赖包含 `STING1` 在内的桥接证据，不能作为新基因泛化证据；它适合证明完整知识库能利用已知 gene-cell-function 关系，而不适合证明去除正确答案后仍能恢复该基因。

## 总结

总体状态：`{overall}`。这个案例满足“已有单细胞证据、公开 raw counts 可复算、对应疾病的致病基因在 A4 中得到提升”的展示目标。它是 literature-first demonstration，不是盲法外部验证，并且原论文已存在于 marker 文献库，需要明确披露来源重叠。
"""
    (output / "report/SAVI_STING1_literature_first_case.md").write_text(report, encoding="utf-8")
    summary = {
        "case_id": manifest["case"]["case_id"],
        "disease_id": manifest["case"]["disease_id"],
        "gene_symbol": manifest["case"]["gene_symbol"],
        "raw_barcode_count": sum(row["input_barcodes"] for row in qc_rows),
        "retained_cell_count": data.n_obs,
        "control_donors": 7,
        "savi_donors": 5,
        "monocyte_upr_effect": primary["upr_effect"],
        "monocyte_upr_p": primary["upr_p_value"],
        "high_upr_fraction_effect": high_effect["effect"],
        "high_upr_fraction_p": high_effect["p_value"],
        "high_upr_fraction_batch_adjusted_effect": high_adjusted["effect"],
        "high_upr_fraction_batch_adjusted_p": high_adjusted["p_value"],
        "monocyte_specificity_did": specificity_did,
        "effector_t_fraction_effect": effector_effect["effect"],
        "effector_t_fraction_p": effector_effect["p_value"],
        "effector_t_fraction_batch_adjusted_effect": effector_adjusted["effect"],
        "effector_t_fraction_batch_adjusted_p": effector_adjusted["p_value"],
        "a0_rank": base_rank,
        "a4_rank": final_rank,
        "a5_rank": a5_rank,
        "rank_change": base_rank - final_rank,
        "cellular_reproduction_status": "supported" if reproduced else "not_supported",
        "rank_improvement_status": rank_status,
        "overall_status": overall,
    }
    write_tsv(output / "case_summary.tsv", [summary], list(summary))
    input_records = {
        f"sample_{sample['sample_id']}": input_record(sample["file"])
        for sample in settings["samples"]
    }
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "literature_case_freeze_sha256_verified": freeze_hash,
        "raw_expression_values_accessed_after_freeze": True,
        "used_published_case_control_deg_tables": False,
        "overall_status": overall,
        "inputs": {
            "literature_case_manifest": input_record(manifest_path),
            "signature_file": input_record(inputs["signature_file"]),
            "a4_rankings": input_record(inputs["a4_rankings"]),
            "a5_rankings": input_record(inputs["a5_rankings"]),
            **input_records,
        },
    }
    write_json(output / "run_manifest.json", run_manifest)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    return run_manifest
