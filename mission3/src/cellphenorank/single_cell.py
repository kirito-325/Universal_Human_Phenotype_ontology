from __future__ import annotations

import math
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2, norm, ttest_1samp, ttest_ind

from .io import input_record, read_tsv, value, write_json, write_tsv
from .plotting import generate_single_cell_figures


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def load_obo_parents(path: str | Path) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    current = ""
    obsolete = False
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                current = ""
                obsolete = False
            elif line.startswith("id: "):
                current = line[4:].strip()
            elif line == "is_obsolete: true":
                obsolete = True
            elif current and line.startswith("is_a: ") and not obsolete:
                parents[current].add(line[6:].split(" ! ", 1)[0].strip())
    return dict(parents)


def ancestors(term_id: str, parents: Mapping[str, set[str]]) -> dict[str, int]:
    distances: dict[str, int] = {term_id: 0}
    queue: deque[str] = deque([term_id])
    while queue:
        current = queue.popleft()
        for parent in parents.get(current, set()):
            if parent not in distances:
                distances[parent] = distances[current] + 1
                queue.append(parent)
    return distances


def load_label_map(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for row in read_tsv(path):
        label = value(row, "dataset_label")
        cell_id = value(row, "cell_id", "cl_id")
        if label and cell_id:
            rows.append(
                {
                    "dataset_id": value(row, "dataset_id", default="*"),
                    "dataset_label": label,
                    "cell_id": cell_id,
                    "mapping_type": value(row, "mapping_type", default="manual"),
                    "confidence": value(row, "confidence", default="1.0"),
                    "note": value(row, "note"),
                }
            )
    return rows


def map_prediction_to_label(
    predicted_cell_id: str,
    dataset_id: str,
    label_rows: Sequence[Mapping[str, str]],
    parents: Mapping[str, set[str]],
) -> tuple[str, str, int] | None:
    candidates = [
        row for row in label_rows if row["dataset_id"] in {"*", dataset_id}
    ]
    exact = [row for row in candidates if row["cell_id"] == predicted_cell_id]
    if exact:
        chosen = sorted(exact, key=lambda row: row["dataset_label"])[0]
        return chosen["dataset_label"], "exact", 0
    predicted_ancestors = ancestors(predicted_cell_id, parents)
    broader = [
        (predicted_ancestors[row["cell_id"]], row)
        for row in candidates
        if row["cell_id"] in predicted_ancestors
    ]
    if broader:
        distance, chosen = sorted(broader, key=lambda item: (item[0], item[1]["dataset_label"]))[0]
        return chosen["dataset_label"], "predicted_child_to_dataset_parent", distance
    return None


def load_predictions(
    evidence_path: str | Path,
    disease_id: str,
    candidate_gene: str,
    top_functions: int,
) -> list[dict[str, str]]:
    rows = []
    for row in read_tsv(evidence_path):
        if value(row, "disease_id") != disease_id:
            continue
        if candidate_gene and value(row, "gene_symbol").upper() != candidate_gene.upper():
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(value(row, "path_contribution", default="0")),
            -float(value(row, "path_weight", default="0")),
            value(row, "cpo_id"),
        )
    )
    unique = []
    seen = set()
    for row in rows:
        key = (value(row, "cell_id"), value(row, "function_id"))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= top_functions:
            break
    if not unique:
        raise ValueError(f"No evidence paths found for {disease_id} / {candidate_gene or 'top candidate'}")
    return unique


def load_frozen_predictions(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not manifest.get("frozen"):
        raise ValueError("Single-cell case manifest is not frozen")
    if manifest.get("selection_used_expression_results"):
        raise ValueError("Case selection used expression results and is not independent")
    predictions = [dict(row) for row in manifest.get("predictions", [])]
    if not predictions:
        raise ValueError("Frozen case manifest contains no predictions")
    return predictions, manifest


def _module_scores(expression: np.ndarray, mode: str) -> np.ndarray:
    if expression.ndim != 2 or expression.shape[1] == 0:
        return np.zeros(expression.shape[0], dtype=float)
    if mode == "mean_log2cpm":
        return expression.mean(axis=1)
    if mode != "gene_zscore":
        raise ValueError(f"Unsupported module_score_mode: {mode}")
    means = expression.mean(axis=0)
    standard_deviations = expression.std(axis=0, ddof=1)
    usable = np.isfinite(standard_deviations) & (standard_deviations > 0)
    if not np.any(usable):
        return np.zeros(expression.shape[0], dtype=float)
    z = (expression[:, usable] - means[usable]) / standard_deviations[usable]
    return z.mean(axis=1)


def _expression_match_pools(
    expression: np.ndarray,
    target_indices: np.ndarray,
    bins: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    means = expression.mean(axis=0)
    standard_deviations = expression.std(axis=0, ddof=1)
    valid = np.flatnonzero(np.isfinite(standard_deviations) & (standard_deviations > 0))
    ordered = valid[np.argsort(means[valid], kind="stable")]
    bin_by_index = np.full(expression.shape[1], -1, dtype=int)
    bin_by_index[ordered] = np.minimum(
        bins - 1,
        np.floor(np.arange(len(ordered)) * bins / len(ordered)).astype(int),
    )
    target_set = set(int(item) for item in target_indices)
    pools = []
    for target in target_indices:
        target_bin = bin_by_index[int(target)]
        candidates = np.asarray([
            int(index)
            for index in valid
            if bin_by_index[int(index)] == target_bin
            and int(index) not in target_set
        ], dtype=int)
        pools.append(candidates if len(candidates) else valid)
    return pools, valid


def _sample_expression_matched_indices(
    pools: Sequence[np.ndarray],
    valid: np.ndarray,
    target_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    target_set = set(int(item) for item in target_indices)
    chosen: set[int] = set()
    for pool in pools:
        available = np.asarray(
            [int(item) for item in pool if int(item) not in target_set and int(item) not in chosen],
            dtype=int,
        )
        if len(available):
            chosen.add(int(rng.choice(available)))
    if len(chosen) < len(target_indices):
        available = np.asarray(
            [int(item) for item in valid if int(item) not in target_set and int(item) not in chosen],
            dtype=int,
        )
        needed = min(len(target_indices) - len(chosen), len(available))
        if needed:
            chosen.update(int(item) for item in rng.choice(available, size=needed, replace=False))
    return np.asarray(sorted(chosen), dtype=int)


def _load_marker_genes(
    path: str | Path,
    cell_ids: set[str],
    masked_genes: set[str],
    top_n: int,
) -> dict[str, list[str]]:
    by_cell: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in read_tsv(path):
        cell_id = value(row, "cell_id", "catalog_id", "cl_id")
        if cell_id not in cell_ids:
            continue
        gene = value(row, "gene_symbol", "gene", "gene_id").upper()
        if not gene or gene in masked_genes:
            continue
        score = float(value(row, "significance_score", "marker_weight", "marker_score", default="1"))
        by_cell[cell_id].append((score, gene))
    return {
        cell_id: [gene for _, gene in sorted(items, key=lambda item: (-item[0], item[1]))[:top_n]]
        for cell_id, items in by_cell.items()
    }


def _identity_control(
    dataset_id: str,
    expression: np.ndarray,
    obs: pd.DataFrame,
    genes: np.ndarray,
    predictions: Sequence[Mapping[str, Any]],
    prediction_mappings: Mapping[tuple[int, str], str],
    marker_genes: Mapping[str, list[str]],
    donor_column: str,
    cell_type_column: str,
    control_labels: set[str],
) -> list[dict[str, Any]]:
    gene_lookup = {str(gene).upper(): index for index, gene in enumerate(genes)}
    rows = []
    for fallback_rank, prediction in enumerate(predictions, 1):
        rank = int(prediction.get("prediction_rank", fallback_rank))
        if not bool(prediction.get("requires_cell_specificity", False)):
            continue
        predicted_label = prediction_mappings.get((rank, dataset_id))
        markers = marker_genes.get(value(prediction, "cell_id"), [])
        indices = np.asarray([gene_lookup[gene] for gene in markers if gene in gene_lookup], dtype=int)
        if not predicted_label or not len(indices):
            continue
        eligible_labels = {predicted_label}.union(control_labels)
        selected = obs[cell_type_column].astype(str).isin(eligible_labels).to_numpy()
        scores = _module_scores(expression[selected][:, indices], "gene_zscore")
        selected_obs = obs.loc[selected, [donor_column, cell_type_column]].copy()
        selected_obs["marker_score"] = scores
        pivot = selected_obs.pivot_table(
            index=donor_column,
            columns=cell_type_column,
            values="marker_score",
            aggfunc="mean",
        )
        available_controls = [label for label in control_labels if label in pivot.columns]
        if predicted_label not in pivot.columns or not available_controls:
            continue
        differences = (
            pivot[predicted_label] - pivot[available_controls].mean(axis=1)
        ).dropna().to_numpy(dtype=float)
        test = ttest_1samp(differences, popmean=0.0, alternative="greater") if len(differences) >= 2 else None
        rows.append(
            {
                "prediction_rank": rank,
                "endpoint_role": prediction.get("endpoint_role", "legacy"),
                "dataset_id": dataset_id,
                "predicted_cell_id": value(prediction, "cell_id"),
                "predicted_dataset_label": predicted_label,
                "control_dataset_labels": sorted(available_controls),
                "marker_gene_count": len(markers),
                "marker_gene_coverage": len(indices) / len(markers) if markers else 0.0,
                "paired_donors": len(differences),
                "mean_predicted_minus_control_score": float(differences.mean()) if len(differences) else 0.0,
                "positive_donor_fraction": float(np.mean(differences > 0)) if len(differences) else 0.0,
                "p_value_greater": float(test.pvalue) if test is not None else 1.0,
                "status": (
                    "pass"
                    if len(differences) >= 4 and float(np.mean(differences > 0)) >= 0.8
                    else "fail"
                ),
            }
        )
    return rows


def _causal_gene_validation(
    datasets: Mapping[str, tuple[np.ndarray, pd.DataFrame, np.ndarray]],
    predictions: Sequence[Mapping[str, Any]],
    prediction_mappings: Mapping[tuple[int, str], str],
    gene_symbol: str,
    prediction_role: str,
    reference_labels: set[str],
    donor_column: str,
    condition_column: str,
    cell_type_column: str,
    cell_count_column: str,
    disease_value: str,
    control_value: str,
    min_cells: int,
    min_detection_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_prediction = next(
        (
            (int(row.get("prediction_rank", index)), row)
            for index, row in enumerate(predictions, 1)
            if str(row.get("endpoint_role", "")) == prediction_role
        ),
        None,
    )
    if selected_prediction is None:
        return [], []
    prediction_rank, prediction = selected_prediction
    per_dataset = []
    for dataset_id, (expression, obs, genes) in datasets.items():
        target_label = prediction_mappings.get((prediction_rank, dataset_id))
        gene_lookup = {str(gene).upper(): index for index, gene in enumerate(genes)}
        gene_index = gene_lookup.get(gene_symbol.upper())
        if not target_label or gene_index is None:
            continue
        eligible = obs[cell_count_column].astype(int) >= min_cells
        target = obs[cell_type_column].astype(str).eq(target_label) & eligible
        target_obs = obs.loc[target, [donor_column, condition_column]].copy()
        target_obs["expression"] = expression[target.to_numpy(), gene_index]
        control = target_obs.loc[
            target_obs[condition_column].astype(str).eq(control_value), "expression"
        ].to_numpy(dtype=float)
        disease = target_obs.loc[
            target_obs[condition_column].astype(str).eq(disease_value), "expression"
        ].to_numpy(dtype=float)
        direct_effect, direct_se, direct_p = _welch_effect(control, disease)
        selected_labels = {target_label}.union(reference_labels)
        contrast_rows = obs[cell_type_column].astype(str).isin(selected_labels) & eligible
        contrast = obs.loc[
            contrast_rows, [donor_column, condition_column, cell_type_column]
        ].copy()
        contrast["expression"] = expression[contrast_rows.to_numpy(), gene_index]
        pivot = contrast.pivot_table(
            index=[donor_column, condition_column],
            columns=cell_type_column,
            values="expression",
            aggfunc="mean",
        )
        available_references = [label for label in reference_labels if label in pivot.columns]
        if target_label in pivot.columns and available_references:
            pivot["cell_specific_contrast"] = (
                pivot[target_label] - pivot[available_references].mean(axis=1)
            )
            control_contrast = pivot.xs(control_value, level=condition_column)[
                "cell_specific_contrast"
            ].dropna().to_numpy(dtype=float)
            disease_contrast = pivot.xs(disease_value, level=condition_column)[
                "cell_specific_contrast"
            ].dropna().to_numpy(dtype=float)
            contrast_effect, contrast_se, contrast_p = _welch_effect(
                control_contrast, disease_contrast
            )
        else:
            control_contrast = disease_contrast = np.asarray([], dtype=float)
            contrast_effect, contrast_se, contrast_p = 0.0, math.inf, 1.0
        detection_fraction = float(np.mean(target_obs["expression"].to_numpy() > 0))
        common = {
            "dataset_id": dataset_id,
            "disease_id": prediction.get("disease_id", ""),
            "gene_symbol": gene_symbol.upper(),
            "predicted_cell_id": value(prediction, "cell_id"),
            "predicted_dataset_label": target_label,
            "reference_dataset_labels": sorted(available_references),
            "control_donors": len(control),
            "disease_donors": len(disease),
            "gene_detection_fraction": detection_fraction,
            "minimum_detection_fraction": min_detection_fraction,
        }
        per_dataset.extend(
            [
                {
                    **common,
                    "endpoint": "predicted_cell_gene_expression",
                    "effect": direct_effect,
                    "standard_error": direct_se,
                    "p_value": direct_p,
                    "direction": "decreased" if direct_effect < 0 else "increased",
                },
                {
                    **common,
                    "endpoint": "predicted_cell_vs_reference_difference_in_differences",
                    "effect": contrast_effect,
                    "standard_error": contrast_se,
                    "p_value": contrast_p,
                    "direction": "decreased" if contrast_effect < 0 else "increased",
                    "control_contrast_donors": len(control_contrast),
                    "disease_contrast_donors": len(disease_contrast),
                },
            ]
        )
    summary = []
    for endpoint in (
        "predicted_cell_gene_expression",
        "predicted_cell_vs_reference_difference_in_differences",
    ):
        rows = [row for row in per_dataset if row["endpoint"] == endpoint]
        effect, standard_error, p_value = _meta_effect(rows)
        q, heterogeneity_p, i_squared = _meta_heterogeneity(rows, effect)
        summary.append(
            {
                "disease_id": prediction.get("disease_id", ""),
                "gene_symbol": gene_symbol.upper(),
                "predicted_cell_id": value(prediction, "cell_id"),
                "predicted_cell_name": value(prediction, "cell_name"),
                "endpoint": endpoint,
                "datasets_evaluated": len(rows),
                "meta_effect": effect,
                "meta_standard_error": standard_error,
                "meta_ci_low": effect - 1.96 * standard_error if math.isfinite(standard_error) else 0.0,
                "meta_ci_high": effect + 1.96 * standard_error if math.isfinite(standard_error) else 0.0,
                "meta_p_value": p_value,
                "heterogeneity_q": q,
                "heterogeneity_p_value": heterogeneity_p,
                "i_squared": i_squared,
                "direction_concordant": len(rows) > 1 and all(float(row["effect"]) < 0 for row in rows),
                "detection_gate_passed": bool(rows) and all(
                    float(row["gene_detection_fraction"]) >= min_detection_fraction
                    for row in rows
                ),
            }
        )
    adjusted = bh_adjust(
        np.asarray([float(row["meta_p_value"]) for row in summary], dtype=float)
    )
    for row, q_value in zip(summary, adjusted):
        row["meta_q_value"] = float(q_value)
        row["validation_status"] = (
            "supported"
            if row["datasets_evaluated"] == len(datasets)
            and row["meta_q_value"] < 0.05
            and row["direction_concordant"]
            and row["detection_gate_passed"]
            else "not_supported"
        )
    return per_dataset, summary


def _pseudobulk_matrix(
    data: ad.AnnData,
    donor_column: str,
    condition_column: str,
    cell_type_column: str,
    cell_count_column: str,
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray]:
    for column in (donor_column, condition_column, cell_type_column):
        if column not in data.obs:
            raise ValueError(f"AnnData obs is missing required column: {column}")
    obs = data.obs.copy()
    keys = [donor_column, condition_column, cell_type_column]
    matrix = sparse.csr_matrix(data.X) if sparse.issparse(data.X) else sparse.csr_matrix(np.asarray(data.X))
    if not obs.duplicated(keys).any():
        output_obs = obs[keys].reset_index(drop=True)
        output_obs["cell_count"] = (
            obs[cell_count_column].to_numpy(dtype=int)
            if cell_count_column in obs
            else np.ones(len(obs), dtype=int)
        )
        return matrix, output_obs, np.asarray(data.var_names.astype(str))
    groups = obs.groupby(keys, sort=True, observed=True).indices
    matrices = []
    rows = []
    for group_key, indices in groups.items():
        index_array = np.asarray(indices, dtype=int)
        matrices.append(sparse.csr_matrix(matrix[index_array].sum(axis=0)))
        rows.append(
            {
                donor_column: group_key[0],
                condition_column: group_key[1],
                cell_type_column: group_key[2],
                "cell_count": (
                    int(obs.iloc[index_array][cell_count_column].sum())
                    if cell_count_column in obs
                    else len(index_array)
                ),
            }
        )
    return sparse.vstack(matrices).tocsr(), pd.DataFrame(rows), np.asarray(data.var_names.astype(str))


def _log2_cpm(matrix: sparse.csr_matrix) -> np.ndarray:
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    scale = np.divide(1_000_000.0, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = sparse.diags(scale) @ matrix.astype(float)
    return np.log2(normalized.toarray() + 1.0)


def _welch_effect(control: np.ndarray, disease: np.ndarray) -> tuple[float, float, float]:
    if len(control) < 2 or len(disease) < 2:
        return 0.0, math.inf, 1.0
    effect = float(np.mean(disease) - np.mean(control))
    variance = float(np.var(disease, ddof=1) / len(disease) + np.var(control, ddof=1) / len(control))
    standard_error = math.sqrt(max(0.0, variance))
    test = ttest_ind(disease, control, equal_var=False, nan_policy="omit")
    return effect, standard_error, float(np.nan_to_num(test.pvalue, nan=1.0))


def _meta_effect(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    effects = np.asarray([float(row["effect"]) for row in rows], dtype=float)
    errors = np.asarray([float(row["standard_error"]) for row in rows], dtype=float)
    weights = np.divide(1.0, errors**2, out=np.zeros_like(errors), where=np.isfinite(errors) & (errors > 0))
    if weights.sum() == 0:
        effect = float(effects.mean()) if len(effects) else 0.0
        return effect, math.inf, 1.0
    effect = float(np.sum(weights * effects) / weights.sum())
    standard_error = float(1.0 / math.sqrt(weights.sum()))
    p_value = float(2.0 * norm.sf(abs(effect / standard_error)))
    return effect, standard_error, p_value


def _meta_heterogeneity(
    rows: Sequence[Mapping[str, Any]], meta_effect: float
) -> tuple[float, float, float]:
    if len(rows) < 2:
        return 0.0, 1.0, 0.0
    effects = np.asarray([float(row["effect"]) for row in rows], dtype=float)
    errors = np.asarray([float(row["standard_error"]) for row in rows], dtype=float)
    weights = np.divide(1.0, errors**2, out=np.zeros_like(errors), where=np.isfinite(errors) & (errors > 0))
    q = float(np.sum(weights * (effects - meta_effect) ** 2))
    degrees = len(rows) - 1
    p_value = float(chi2.sf(q, degrees))
    i_squared = max(0.0, (q - degrees) / q) if q > 0 else 0.0
    return q, p_value, i_squared


def validate_single_cell(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    settings = config.get("single_cell", {})
    inputs = config.get("inputs", {})
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_manifest_path = settings.get("case_manifest")
    case_manifest: dict[str, Any] = {}
    if case_manifest_path:
        predictions, case_manifest = load_frozen_predictions(case_manifest_path)
        disease_id = str(case_manifest["disease_id"])
    else:
        disease_id = str(settings["disease_id"])
        evidence_path = settings.get("evidence_paths") or inputs.get("evidence_paths")
        predictions = load_predictions(
            evidence_path,
            disease_id,
            str(settings.get("candidate_gene", "")),
            int(settings.get("top_functions", 5)),
        )
    candidate_gene = str(settings.get("candidate_gene", ""))
    evidence_path = settings.get("evidence_paths") or inputs.get("evidence_paths")
    masked_genes = {
        str(gene).upper()
        for gene in settings.get("masked_genes", case_manifest.get("masked_genes", []))
    }
    datasets = [Path(path) for path in settings.get("datasets", [])]
    dataset_ids = list(settings.get("dataset_ids", [path.stem for path in datasets]))
    if len(dataset_ids) != len(datasets):
        raise ValueError("single_cell.dataset_ids must match single_cell.datasets")
    for path in datasets:
        if not path.is_file():
            raise FileNotFoundError(path)
    label_rows = load_label_map(settings["label_map"])
    parents = load_obo_parents(settings["cl_obo"])
    mapping_rows = []
    prediction_mappings: dict[tuple[int, str], str] = {}
    for fallback_index, prediction in enumerate(predictions, 1):
        index = int(prediction.get("prediction_rank", fallback_index))
        for dataset_id in dataset_ids:
            mapped = map_prediction_to_label(value(prediction, "cell_id"), dataset_id, label_rows, parents)
            if mapped:
                label, mapping_type, distance = mapped
                prediction_mappings[(index, dataset_id)] = label
                mapping_rows.append(
                    {
                        "prediction_rank": index,
                        "endpoint_role": prediction.get("endpoint_role", "legacy"),
                        "dataset_id": dataset_id,
                        "predicted_cell_id": value(prediction, "cell_id"),
                        "predicted_cell_name": value(prediction, "cell_name"),
                        "dataset_label": label,
                        "mapping_type": mapping_type,
                        "ontology_distance": distance,
                    }
                )
    relevant_functions = {value(row, "function_id") for row in predictions}
    function_names: dict[str, str] = {}
    genes_by_function: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(inputs["go_functions"]):
        function_id = value(row, "function_id", "go_id", "cell_phenotype_id")
        if function_id in relevant_functions:
            gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
            if gene and gene not in masked_genes:
                genes_by_function[function_id].add(gene)
                function_names[function_id] = value(row, "function_name", "go_name", default=function_id)

    donor_column = str(settings.get("donor_column", "donor_id"))
    condition_column = str(settings.get("condition_column", "condition"))
    cell_type_column = str(settings.get("cell_type_column", "harmonized_cell_type"))
    cell_count_column = str(settings.get("cell_count_column", "cell_count"))
    disease_value = str(settings.get("disease_value", "disease"))
    control_value = str(settings.get("control_value", "control"))
    min_donors = int(settings.get("min_donors_per_condition", 3))
    min_cells = int(settings.get("min_cells_per_profile", 20))
    min_function_genes = int(settings.get("min_function_genes", 5))
    min_gene_coverage = float(settings.get("min_function_gene_coverage", 0.8))
    max_ontology_distance = int(settings.get("max_ontology_distance", 99))
    module_score_mode = str(settings.get("module_score_mode", "gene_zscore"))
    composition_exclude = {str(item) for item in settings.get("composition_exclude_labels", [])}
    specificity_exclude = {str(item) for item in settings.get("specificity_exclude_labels", [])}
    random_iterations = int(settings.get("random_iterations", 100))
    random_match_bins = int(settings.get("random_gene_match_bins", 10))
    rng = np.random.default_rng(int(config.get("project", {}).get("random_seed", 2026)))
    function_rows: list[dict[str, Any]] = []
    specificity_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    de_rows: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    loaded_data: dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray]] = {}
    marker_genes = (
        _load_marker_genes(
            inputs["cell_markers"],
            {value(row, "cell_id") for row in predictions},
            masked_genes,
            int(settings.get("identity_marker_top_n", 50)),
        )
        if inputs.get("cell_markers")
        else {}
    )
    identity_control_labels = {
        str(item) for item in settings.get(
            "identity_control_labels", ["Tconv_resting", "Tconv_activated"]
        )
    }
    for dataset_id, path in zip(dataset_ids, datasets):
        data = ad.read_h5ad(path)
        matrix, obs, genes = _pseudobulk_matrix(
            data, donor_column, condition_column, cell_type_column, cell_count_column
        )
        expression = _log2_cpm(matrix)
        loaded_data[dataset_id] = (expression, obs, genes)
        obs = obs.copy()
        composition_base = ~obs[cell_type_column].astype(str).isin(composition_exclude)
        donor_totals = (
            obs.loc[composition_base]
            .groupby(donor_column, observed=True)["cell_count"]
            .sum()
        )
        obs["cell_fraction"] = (
            obs["cell_count"].astype(float)
            / obs[donor_column].astype(str).map(donor_totals).astype(float)
        )
        identity_rows.extend(
            _identity_control(
                dataset_id,
                expression,
                obs,
                genes,
                predictions,
                prediction_mappings,
                marker_genes,
                donor_column,
                cell_type_column,
                identity_control_labels,
            )
        )
        for fallback_rank, prediction in enumerate(predictions, 1):
            prediction_rank = int(prediction.get("prediction_rank", fallback_rank))
            endpoint_role = str(prediction.get("endpoint_role", "legacy"))
            mapped_label = prediction_mappings.get((prediction_rank, dataset_id))
            if not mapped_label:
                preflight_rows.append(
                    {
                        "prediction_rank": prediction_rank,
                        "endpoint_role": endpoint_role,
                        "dataset_id": dataset_id,
                        "mapped_dataset_label": "",
                        "control_donors": 0,
                        "disease_donors": 0,
                        "minimum_cells_per_profile": min_cells,
                        "function_gene_count_after_masking": 0,
                        "function_gene_coverage": 0.0,
                        "ontology_distance": -1,
                        "status": "fail",
                        "reasons": "cell_mapping_missing",
                    }
                )
                continue
            function_id = value(prediction, "function_id")
            function_genes = genes_by_function.get(function_id, set())
            gene_indices = np.asarray([index for index, gene in enumerate(genes) if gene.upper() in function_genes], dtype=int)
            mapped_rows = (
                obs[cell_type_column].astype(str).eq(mapped_label)
                & (obs["cell_count"].astype(int) >= min_cells)
            )
            mapped_conditions = obs.loc[mapped_rows, condition_column].astype(str)
            control_donor_count = int(
                obs.loc[mapped_rows & obs[condition_column].astype(str).eq(control_value), donor_column].nunique()
            )
            disease_donor_count = int(
                obs.loc[mapped_rows & obs[condition_column].astype(str).eq(disease_value), donor_column].nunique()
            )
            coverage = len(gene_indices) / len(function_genes) if function_genes else 0.0
            mapping_distance = next(
                (
                    int(row["ontology_distance"])
                    for row in mapping_rows
                    if int(row["prediction_rank"]) == prediction_rank
                    and row["dataset_id"] == dataset_id
                ),
                -1,
            )
            reasons = []
            if control_donor_count < min_donors or disease_donor_count < min_donors:
                reasons.append("insufficient_donors")
            if len(function_genes) < min_function_genes:
                reasons.append("function_gene_set_too_small")
            if coverage < min_gene_coverage:
                reasons.append("function_gene_coverage_too_low")
            if mapping_distance < 0 or mapping_distance > max_ontology_distance:
                reasons.append("cell_mapping_too_broad")
            preflight_rows.append(
                {
                    "prediction_rank": prediction_rank,
                    "endpoint_role": endpoint_role,
                    "dataset_id": dataset_id,
                    "mapped_dataset_label": mapped_label,
                    "control_donors": control_donor_count,
                    "disease_donors": disease_donor_count,
                    "minimum_cells_per_profile": min_cells,
                    "function_gene_count_after_masking": len(function_genes),
                    "function_gene_coverage": coverage,
                    "ontology_distance": mapping_distance,
                    "status": "pass" if not reasons else "fail",
                    "reasons": reasons,
                }
            )
            if reasons:
                continue
            for label in sorted(obs[cell_type_column].astype(str).unique()):
                if label in specificity_exclude and label != mapped_label:
                    continue
                selected = (
                    obs[cell_type_column].astype(str).eq(label)
                    & (obs["cell_count"].astype(int) >= min_cells)
                )
                condition = obs.loc[selected, condition_column].astype(str).to_numpy()
                module = _module_scores(
                    expression[selected.to_numpy()][:, gene_indices], module_score_mode
                )
                control = module[condition == control_value]
                disease = module[condition == disease_value]
                if len(control) < min_donors or len(disease) < min_donors:
                    continue
                effect, standard_error, p_value = _welch_effect(control, disease)
                row = {
                    "prediction_rank": prediction_rank,
                    "endpoint_role": endpoint_role,
                    "dataset_id": dataset_id,
                    "predicted_cell_id": value(prediction, "cell_id"),
                    "mapped_dataset_label": mapped_label,
                    "tested_dataset_label": label,
                    "is_predicted_cell": label == mapped_label,
                    "function_id": function_id,
                    "function_name": function_names.get(function_id, value(prediction, "function_name")),
                    "function_gene_count": len(function_genes),
                    "gene_coverage": len(gene_indices) / len(function_genes) if function_genes else 0.0,
                    "control_donors": len(control),
                    "disease_donors": len(disease),
                    "effect": effect,
                    "standard_error": standard_error,
                    "p_value": p_value,
                    "module_score_mode": module_score_mode,
                }
                specificity_rows.append(row)
                if label == mapped_label:
                    selected_indices = np.flatnonzero(selected.to_numpy())
                    selected_expression = expression[selected_indices]
                    match_pools, valid_background = _expression_match_pools(
                        selected_expression, gene_indices, random_match_bins
                    )
                    null_effects = []
                    for _ in range(random_iterations):
                        random_indices = _sample_expression_matched_indices(
                            match_pools,
                            valid_background,
                            gene_indices,
                            rng,
                        )
                        random_module = _module_scores(
                            expression[selected.to_numpy()][:, random_indices],
                            module_score_mode,
                        )
                        null_effects.append(
                            abs(float(random_module[condition == disease_value].mean() - random_module[condition == control_value].mean()))
                        )
                    row = dict(row)
                    candidate_matches = (
                        np.flatnonzero(np.char.upper(genes.astype(str)) == candidate_gene.upper())
                        if candidate_gene
                        else np.asarray([], dtype=int)
                    )
                    row["candidate_gene_expression_coverage"] = (
                        float(np.mean(selected_expression[:, candidate_matches[0]] > 0))
                        if len(candidate_matches)
                        else 0.0
                    )
                    row["random_gene_set_percentile"] = (
                        (1 + sum(item <= abs(effect) for item in null_effects)) / (len(null_effects) + 1)
                        if null_effects
                        else 0.0
                    )
                    row["random_gene_set_mode"] = f"mean_expression_{random_match_bins}_bin_matched"
                    function_rows.append(row)
                    control_mask = condition == control_value
                    disease_mask = condition == disease_value
                    p_values = []
                    local_de = []
                    for gene_index in gene_indices:
                        gene_effect, _, gene_p = _welch_effect(
                            selected_expression[control_mask, gene_index],
                            selected_expression[disease_mask, gene_index],
                        )
                        p_values.append(gene_p)
                        local_de.append(
                            {
                                "prediction_rank": prediction_rank,
                                "endpoint_role": endpoint_role,
                                "dataset_id": dataset_id,
                                "dataset_label": label,
                                "function_id": function_id,
                                "gene_symbol": genes[gene_index],
                                "log2cpm_effect": gene_effect,
                                "p_value": gene_p,
                            }
                        )
                    adjusted = bh_adjust(np.asarray(p_values, dtype=float))
                    for local, q_value in zip(local_de, adjusted):
                        local["q_value_within_function"] = float(q_value)
                    de_rows.extend(local_de)
            if mapped_label in composition_exclude:
                continue
            selected_composition = obs[
                obs[cell_type_column].astype(str).eq(mapped_label)
                & (obs["cell_count"].astype(int) >= min_cells)
            ]
            control_fraction = selected_composition.loc[
                selected_composition[condition_column].astype(str).eq(control_value), "cell_fraction"
            ].to_numpy()
            disease_fraction = selected_composition.loc[
                selected_composition[condition_column].astype(str).eq(disease_value), "cell_fraction"
            ].to_numpy()
            effect, standard_error, p_value = _welch_effect(control_fraction, disease_fraction)
            composition_rows.append(
                {
                    "prediction_rank": prediction_rank,
                    "endpoint_role": endpoint_role,
                    "dataset_id": dataset_id,
                    "predicted_cell_id": value(prediction, "cell_id"),
                    "mapped_dataset_label": mapped_label,
                    "control_donors": len(control_fraction),
                    "disease_donors": len(disease_fraction),
                    "fraction_effect": effect,
                    "standard_error": standard_error,
                    "p_value": p_value,
                }
            )

    for key in {(row["prediction_rank"], row["dataset_id"], row["function_id"]) for row in specificity_rows}:
        subset = [
            row for row in specificity_rows
            if (row["prediction_rank"], row["dataset_id"], row["function_id"]) == key
        ]
        ordered = sorted(subset, key=lambda row: -abs(float(row["effect"])))
        for rank, row in enumerate(ordered, 1):
            row["absolute_effect_rank"] = rank

    causal_settings = settings.get("causal_gene_validation", {})
    causal_rows, causal_summary = (
        _causal_gene_validation(
            loaded_data,
            predictions,
            prediction_mappings,
            str(causal_settings.get("gene_symbol", candidate_gene)),
            str(causal_settings.get("prediction_role", "practical_positive_control")),
            {str(item) for item in causal_settings.get("reference_labels", ["Tconv_resting", "Tconv_activated"])},
            donor_column,
            condition_column,
            cell_type_column,
            "cell_count",
            disease_value,
            control_value,
            min_cells,
            float(causal_settings.get("min_detection_fraction", 0.8)),
        )
        if bool(causal_settings.get("enabled", False))
        else ([], [])
    )

    summary_rows = []
    for fallback_rank, prediction in enumerate(predictions, 1):
        prediction_rank = int(prediction.get("prediction_rank", fallback_rank))
        study_rows = [row for row in function_rows if row["prediction_rank"] == prediction_rank]
        effect, standard_error, p_value = _meta_effect(study_rows)
        heterogeneity_q, heterogeneity_p, i_squared = _meta_heterogeneity(
            study_rows, effect
        )
        cell_ranks = [
            row["absolute_effect_rank"]
            for row in specificity_rows
            if row["prediction_rank"] == prediction_rank and row["is_predicted_cell"]
        ]
        predicted_effects = [
            abs(float(row["effect"]))
            for row in specificity_rows
            if row["prediction_rank"] == prediction_rank and row["is_predicted_cell"]
        ]
        other_effects = [
            abs(float(row["effect"]))
            for row in specificity_rows
            if row["prediction_rank"] == prediction_rank and not row["is_predicted_cell"]
        ]
        study_effects = [float(row["effect"]) for row in study_rows]
        expected_direction = str(prediction.get("expected_direction", "unknown"))
        direction_aligned = (
            expected_direction == "unknown"
            or (expected_direction == "increased" and effect > 0)
            or (expected_direction == "decreased" and effect < 0)
        )
        random_percentiles = [
            float(row.get("random_gene_set_percentile", 0.0)) for row in study_rows
        ]
        identity_statuses = [
            row["status"]
            for row in identity_rows
            if int(row["prediction_rank"]) == prediction_rank
        ]
        requires_specificity = bool(prediction.get("requires_cell_specificity", False))
        summary_rows.append(
            {
                "disease_id": disease_id,
                "candidate_gene": candidate_gene,
                "prediction_rank": prediction_rank,
                "endpoint_role": prediction.get("endpoint_role", "legacy"),
                "predicted_cell_id": value(prediction, "cell_id"),
                "predicted_cell_name": value(prediction, "cell_name"),
                "function_id": value(prediction, "function_id"),
                "function_name": function_names.get(value(prediction, "function_id"), value(prediction, "function_name")),
                "studies_evaluated": len(study_rows),
                "meta_effect": effect,
                "meta_standard_error": standard_error,
                "meta_ci_low": effect - 1.96 * standard_error if math.isfinite(standard_error) else 0.0,
                "meta_ci_high": effect + 1.96 * standard_error if math.isfinite(standard_error) else 0.0,
                "meta_p_value": p_value,
                "heterogeneity_q": heterogeneity_q,
                "heterogeneity_p_value": heterogeneity_p,
                "i_squared": i_squared,
                "expected_direction": expected_direction,
                "direction_aligned": direction_aligned,
                "best_predicted_cell_rank": min(cell_ranks) if cell_ranks else 0,
                "worst_predicted_cell_rank": max(cell_ranks) if cell_ranks else 0,
                "predicted_vs_other_absolute_effect_difference": (
                    max(predicted_effects) - max(other_effects)
                    if predicted_effects and other_effects
                    else 0.0
                ),
                "direction_concordant_across_studies": (
                    len(study_effects) > 1
                    and all(np.sign(item) == np.sign(study_effects[0]) for item in study_effects[1:])
                ),
                "minimum_random_gene_set_percentile": min(random_percentiles) if random_percentiles else 0.0,
                "requires_cell_specificity": requires_specificity,
                "identity_control_passed_all_datasets": bool(identity_statuses) and all(item == "pass" for item in identity_statuses),
                "preflight_passed_all_datasets": all(
                    row["status"] == "pass"
                    for row in preflight_rows
                    if int(row["prediction_rank"]) == prediction_rank
                ),
            }
        )
    adjusted_meta = bh_adjust(
        np.asarray([float(row["meta_p_value"]) for row in summary_rows], dtype=float)
    )
    min_random_percentile = float(settings.get("min_random_gene_set_percentile", 0.95))
    for row, q_value in zip(summary_rows, adjusted_meta):
        row["meta_q_value"] = float(q_value)
        specificity_pass = (
            not row["requires_cell_specificity"]
            or (
                row["identity_control_passed_all_datasets"]
                and row["worst_predicted_cell_rank"] == 1
            )
        )
        row["cell_specificity_passed"] = specificity_pass
        supported = (
            row["studies_evaluated"] == len(datasets)
            and row["preflight_passed_all_datasets"]
            and row["meta_q_value"] < 0.05
            and row["direction_aligned"]
            and row["direction_concordant_across_studies"]
            and row["minimum_random_gene_set_percentile"] >= min_random_percentile
            and specificity_pass
        )
        row["validation_status"] = (
            "supported" if supported
            else "not_supported" if row["studies_evaluated"] == len(datasets)
            else "not_evaluable"
        )

    write_tsv(
        output / "cell_label_mapping.tsv",
        mapping_rows,
        ["prediction_rank", "endpoint_role", "dataset_id", "predicted_cell_id", "predicted_cell_name", "dataset_label", "mapping_type", "ontology_distance"],
    )
    write_tsv(
        output / "cell_composition.tsv",
        composition_rows,
        ["prediction_rank", "endpoint_role", "dataset_id", "predicted_cell_id", "mapped_dataset_label", "control_donors", "disease_donors", "fraction_effect", "standard_error", "p_value"],
    )
    common_function_fields = [
        "prediction_rank", "endpoint_role", "dataset_id", "predicted_cell_id", "mapped_dataset_label", "tested_dataset_label",
        "is_predicted_cell", "function_id", "function_name", "function_gene_count", "gene_coverage",
        "control_donors", "disease_donors", "effect", "standard_error", "p_value", "module_score_mode",
    ]
    function_fields = common_function_fields + [
        "random_gene_set_percentile",
        "random_gene_set_mode",
        "candidate_gene_expression_coverage",
    ]
    write_tsv(output / "function_enrichment.tsv", function_rows, function_fields)
    write_tsv(output / "cell_specificity.tsv", specificity_rows, common_function_fields + ["absolute_effect_rank"])
    write_tsv(
        output / "pseudobulk_de.tsv",
        de_rows,
        ["prediction_rank", "endpoint_role", "dataset_id", "dataset_label", "function_id", "gene_symbol", "log2cpm_effect", "p_value", "q_value_within_function"],
    )
    write_tsv(
        output / "analysis_preflight.tsv",
        preflight_rows,
        [
            "prediction_rank",
            "endpoint_role",
            "dataset_id",
            "mapped_dataset_label",
            "control_donors",
            "disease_donors",
            "minimum_cells_per_profile",
            "function_gene_count_after_masking",
            "function_gene_coverage",
            "ontology_distance",
            "status",
            "reasons",
        ],
    )
    write_tsv(
        output / "cell_identity_control.tsv",
        identity_rows,
        [
            "prediction_rank",
            "endpoint_role",
            "dataset_id",
            "predicted_cell_id",
            "predicted_dataset_label",
            "control_dataset_labels",
            "marker_gene_count",
            "marker_gene_coverage",
            "paired_donors",
            "mean_predicted_minus_control_score",
            "positive_donor_fraction",
            "p_value_greater",
            "status",
        ],
    )
    write_tsv(
        output / "case_summary.tsv",
        summary_rows,
        [
            "disease_id", "candidate_gene", "prediction_rank", "endpoint_role", "predicted_cell_id",
            "predicted_cell_name", "function_id", "function_name", "studies_evaluated", "meta_effect",
            "meta_standard_error", "meta_ci_low", "meta_ci_high", "meta_p_value", "meta_q_value",
            "heterogeneity_q", "heterogeneity_p_value", "i_squared", "expected_direction", "direction_aligned",
            "best_predicted_cell_rank", "worst_predicted_cell_rank",
            "predicted_vs_other_absolute_effect_difference", "direction_concordant_across_studies",
            "minimum_random_gene_set_percentile", "requires_cell_specificity",
            "identity_control_passed_all_datasets", "preflight_passed_all_datasets",
            "cell_specificity_passed", "validation_status"
        ],
    )
    write_tsv(
        output / "causal_gene_expression.tsv",
        causal_rows,
        [
            "dataset_id", "disease_id", "gene_symbol", "predicted_cell_id",
            "predicted_dataset_label", "reference_dataset_labels", "endpoint",
            "control_donors", "disease_donors", "control_contrast_donors",
            "disease_contrast_donors", "gene_detection_fraction",
            "minimum_detection_fraction", "effect", "standard_error", "p_value", "direction"
        ],
    )
    write_tsv(
        output / "causal_gene_summary.tsv",
        causal_summary,
        [
            "disease_id", "gene_symbol", "predicted_cell_id", "predicted_cell_name",
            "endpoint", "datasets_evaluated", "meta_effect", "meta_standard_error",
            "meta_ci_low", "meta_ci_high", "meta_p_value", "meta_q_value",
            "heterogeneity_q", "heterogeneity_p_value", "i_squared",
            "direction_concordant", "detection_gate_passed", "validation_status"
        ],
    )
    figure_paths = generate_single_cell_figures(
        output, specificity_rows, composition_rows, summary_rows
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disease_id": disease_id,
        "candidate_gene": candidate_gene,
        "replication_unit": "donor_by_cell_type_pseudobulk",
        "datasets": [input_record(path) for path in datasets],
        "prediction_source": (
            input_record(case_manifest_path)
            if case_manifest_path
            else input_record(evidence_path)
        ),
        "case_predictions_frozen_before_expression": bool(case_manifest_path),
        "masked_genes": sorted(masked_genes),
        "prediction_count": len(predictions),
        "minimum_donors_per_condition": min_donors,
        "random_gene_set_iterations": random_iterations,
        "random_gene_set_mode": f"mean_expression_{random_match_bins}_bin_matched",
        "module_score_mode": module_score_mode,
        "preflight_failures": sum(row["status"] != "pass" for row in preflight_rows),
        "causal_gene_validation_status": {
            row["endpoint"]: row["validation_status"] for row in causal_summary
        },
        "figure_count": len(figure_paths),
        "interpretation": "Mechanism validation only; not evidence of candidate-gene causality.",
        "status_counts": {
            str(key): int(count)
            for key, count in pd.Series([row["validation_status"] for row in summary_rows]).value_counts().items()
        },
    }
    write_json(output / "run_manifest.json", manifest)
    report_lines = [
        "# Single-cell case validation",
        "",
        f"Disease: {disease_id}",
        f"Candidate gene tested directly: {candidate_gene}",
        f"Candidate gene excluded from downstream function modules: {candidate_gene}",
        "",
        "The analysis unit is donor x cell type. Composition and function-state results are reported separately.",
        "",
        "## Causal-gene expression validation",
        "",
        "| Endpoint | Meta effect | 95% CI | Q value | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in causal_summary:
        report_lines.append(
            f"| {row['endpoint']} | {row['meta_effect']:.4f} | "
            f"[{row['meta_ci_low']:.4f}, {row['meta_ci_high']:.4f}] | "
            f"{row['meta_q_value']:.4g} | {row['validation_status']} |"
        )
    report_lines.extend([
        "",
        "## Downstream cell-function validation",
        "",
        "| Rank | Role | Predicted cell | Function | Studies | Meta effect | Q value | Cell rank | Status |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in summary_rows:
        report_lines.append(
            f"| {row['prediction_rank']} | {row['endpoint_role']} | {row['predicted_cell_name']} | {row['function_name']} | "
            f"{row['studies_evaluated']} | {row['meta_effect']:.4f} | {row['meta_q_value']:.4g} | "
            f"{row['best_predicted_cell_rank']} | {row['validation_status']} |"
        )
    report_lines.extend(["", "This result cannot by itself establish causality or exclusivity of the predicted cell type."])
    (output / "case_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return manifest
