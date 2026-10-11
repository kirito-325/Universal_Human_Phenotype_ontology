from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cellphenorank.io import input_record, load_diseases, read_tsv, value, write_json, write_tsv
from cellphenorank.rerank import disease_cpo_profiles
from cellphenorank.resources import load_hpo_cpo
from cellphenorank_validation.analysis import load_label_map, load_obo_parents, map_prediction_to_label
from cellphenorank.evaluation import load_gold_genes


CASE_CANDIDATE_FIELDS = [
    "disease_id",
    "cpo_id",
    "cell_id",
    "cell_name",
    "function_id",
    "function_name",
    "practical_weight",
    "strict_weight",
    "strict_retained_fraction",
    "function_gene_count_after_masking",
    "mapped_dataset_labels",
    "max_ontology_distance",
    "mapping_status",
    "strict_eligible",
    "practical_eligible",
    "rejection_reasons",
]

PREDICTION_FIELDS = [
    "prediction_rank",
    "endpoint_role",
    "disease_id",
    "cpo_id",
    "cell_id",
    "cell_name",
    "function_id",
    "function_name",
    "prediction_weight",
    "practical_weight",
    "strict_weight",
    "strict_retained_fraction",
    "function_gene_count_after_masking",
    "expected_direction",
    "requires_cell_specificity",
    "masked_genes",
]


def _function_genes(path: str | Path, function_ids: set[str]) -> dict[str, set[str]]:
    output = {function_id: set() for function_id in function_ids}
    for row in read_tsv(path):
        function_id = value(row, "function_id", "go_id", "cell_phenotype_id")
        if function_id not in output:
            continue
        gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
        if gene:
            output[function_id].add(gene)
    return output


def _dataset_preflight(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = pd.read_csv(settings["metadata"], sep="\t", low_memory=False)
    genotype_column = str(settings.get("genotype_column", "genotype"))
    donor_column = str(settings.get("donor_column", "donor"))
    cohort_column = str(settings.get("cohort_column", "cohort"))
    cell_column = str(settings.get("cell_label_column", "Cluster_Annotation_General"))
    disease_value = str(settings.get("disease_value", "IPEX"))
    control_value = str(settings.get("control_value", "HD"))
    min_donors = int(settings.get("min_donors_per_condition", 4))
    min_cells = int(settings.get("min_cells_per_donor_cell", 20))
    rows = []
    for dataset in settings.get("datasets", []):
        cohort = str(dataset["metadata_cohort"])
        dataset_id = str(dataset["dataset_id"])
        excluded = {str(item) for item in dataset.get("excluded_donors", [])}
        selected = metadata[
            metadata[cohort_column].astype(str).eq(cohort)
            & metadata[genotype_column].astype(str).isin([disease_value, control_value])
            & ~metadata[donor_column].astype(str).isin(excluded)
        ].copy()
        for label in sorted(set(selected[cell_column].astype(str)).union({"T_cell_all"})):
            if label == "T_cell_all":
                counts = selected.groupby([genotype_column, donor_column], observed=True).size()
            else:
                counts = selected[selected[cell_column].astype(str).eq(label)].groupby(
                    [genotype_column, donor_column], observed=True
                ).size()
            disease_counts = counts.loc[disease_value] if disease_value in counts.index.get_level_values(0) else pd.Series(dtype=float)
            control_counts = counts.loc[control_value] if control_value in counts.index.get_level_values(0) else pd.Series(dtype=float)
            disease_eligible = int((disease_counts >= min_cells).sum())
            control_eligible = int((control_counts >= min_cells).sum())
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "metadata_cohort": cohort,
                    "dataset_label": label,
                    "disease_donors": int(len(disease_counts)),
                    "control_donors": int(len(control_counts)),
                    "disease_donors_min_cells": disease_eligible,
                    "control_donors_min_cells": control_eligible,
                    "minimum_cells_per_profile": min_cells,
                    "minimum_donors_per_condition": min_donors,
                    "status": (
                        "pass"
                        if disease_eligible >= min_donors and control_eligible >= min_donors
                        else "fail"
                    ),
                    "excluded_donors": sorted(excluded),
                }
            )
    return rows


def freeze_case(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    inputs = config["inputs"]
    settings = config["case_freeze"]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    disease_id = str(settings["disease_id"])
    diseases = load_diseases(inputs["diseases"], {disease_id})
    if disease_id not in diseases:
        raise ValueError(f"Disease not found in configured profile: {disease_id}")
    hpo_ids = {term.hpo_id for term in diseases[disease_id]}
    hpo_cpo = load_hpo_cpo(inputs["hpo_cpo"], hpo_ids, config.get("hpo_cpo", {}))
    gold = load_gold_genes(inputs.get("gold_standard") or inputs["gold_models"])
    masked_genes = {gene.upper() for gene in gold.get(disease_id, set())}
    practical_rows, _ = disease_cpo_profiles(diseases, hpo_cpo)
    strict_rows, _ = disease_cpo_profiles(
        diseases, hpo_cpo, {disease_id: masked_genes}
    )
    practical = {str(row["cpo_id"]): row for row in practical_rows}
    strict = {str(row["cpo_id"]): row for row in strict_rows}
    function_ids = {
        str(row["function_id"]) for row in practical_rows
    }.union(str(row["function_id"]) for row in strict_rows)
    genes_by_function = _function_genes(inputs["go_functions"], function_ids)
    label_rows = load_label_map(settings["label_map"])
    parents = load_obo_parents(settings["cl_obo"])
    dataset_ids = [str(row["dataset_id"]) for row in settings.get("datasets", [])]
    preflight_rows = _dataset_preflight(settings)
    passing_labels = {
        (str(row["dataset_id"]), str(row["dataset_label"]))
        for row in preflight_rows
        if row["status"] == "pass"
    }
    min_genes = int(settings.get("min_function_genes", 5))
    max_distance = int(settings.get("max_ontology_distance", 0))
    candidate_rows = []
    all_cpo_ids = sorted(set(practical).union(strict))
    for cpo_id in all_cpo_ids:
        source = practical.get(cpo_id) or strict[cpo_id]
        practical_weight = float(practical.get(cpo_id, {}).get("disease_cpo_weight", 0.0))
        strict_weight = float(strict.get(cpo_id, {}).get("disease_cpo_weight", 0.0))
        function_id = str(source["function_id"])
        function_genes = genes_by_function.get(function_id, set()) - masked_genes
        mappings = []
        mapping_failures = []
        for dataset_id in dataset_ids:
            mapped = map_prediction_to_label(
                str(source["cell_id"]), dataset_id, label_rows, parents
            )
            if not mapped:
                mapping_failures.append(dataset_id)
                continue
            label, mapping_type, distance = mapped
            if (dataset_id, label) not in passing_labels:
                mapping_failures.append(f"{dataset_id}:{label}:insufficient_donors")
                continue
            mappings.append((dataset_id, label, mapping_type, distance))
        reasons = []
        if not mappings or mapping_failures:
            reasons.append("cell_not_evaluable_in_all_datasets")
        if mappings and max(item[3] for item in mappings) > max_distance:
            reasons.append("cell_mapping_too_broad")
        if len(function_genes) < min_genes:
            reasons.append("function_gene_set_too_small_after_masking")
        strict_eligible = strict_weight > 0 and not reasons
        practical_eligible = practical_weight > 0 and not reasons
        candidate_rows.append(
            {
                "disease_id": disease_id,
                "cpo_id": cpo_id,
                "cell_id": source["cell_id"],
                "cell_name": source["cell_name"],
                "function_id": function_id,
                "function_name": source["function_name"],
                "practical_weight": practical_weight,
                "strict_weight": strict_weight,
                "strict_retained_fraction": strict_weight / practical_weight if practical_weight else 0.0,
                "function_gene_count_after_masking": len(function_genes),
                "mapped_dataset_labels": [f"{item[0]}:{item[1]}" for item in mappings],
                "max_ontology_distance": max((item[3] for item in mappings), default=-1),
                "mapping_status": ";".join(item[2] for item in mappings),
                "strict_eligible": strict_eligible,
                "practical_eligible": practical_eligible,
                "rejection_reasons": reasons,
            }
        )
    strict_candidates = sorted(
        (row for row in candidate_rows if row["strict_eligible"]),
        key=lambda row: (-float(row["strict_weight"]), str(row["cpo_id"])),
    )
    practical_candidates = sorted(
        (row for row in candidate_rows if row["practical_eligible"]),
        key=lambda row: (-float(row["practical_weight"]), str(row["cpo_id"])),
    )
    if not strict_candidates:
        raise ValueError("No strict single-cell prediction passed the preflight gates")
    strict_primary = strict_candidates[0]
    practical_positive = next(
        (row for row in practical_candidates if row["cell_id"] != strict_primary["cell_id"]),
        practical_candidates[0] if practical_candidates else None,
    )
    selected = [("strict_primary", strict_primary)]
    if practical_positive and practical_positive["cpo_id"] != strict_primary["cpo_id"]:
        selected.append(("practical_positive_control", practical_positive))
    directions = settings.get("expected_directions", {})
    prediction_rows = []
    for rank, (role, row) in enumerate(selected, 1):
        prediction_rows.append(
            {
                "prediction_rank": rank,
                "endpoint_role": role,
                "disease_id": disease_id,
                "cpo_id": row["cpo_id"],
                "cell_id": row["cell_id"],
                "cell_name": row["cell_name"],
                "function_id": row["function_id"],
                "function_name": row["function_name"],
                "prediction_weight": row["strict_weight"] if role == "strict_primary" else row["practical_weight"],
                "practical_weight": row["practical_weight"],
                "strict_weight": row["strict_weight"],
                "strict_retained_fraction": row["strict_retained_fraction"],
                "function_gene_count_after_masking": row["function_gene_count_after_masking"],
                "expected_direction": directions.get(row["function_id"], "unknown"),
                "requires_cell_specificity": role != "strict_primary",
                "masked_genes": sorted(masked_genes),
            }
        )
    write_tsv(output / "case_candidates.tsv", candidate_rows, CASE_CANDIDATE_FIELDS)
    write_tsv(output / "predictions.tsv", prediction_rows, PREDICTION_FIELDS)
    write_tsv(
        output / "dataset_preflight.tsv",
        preflight_rows,
        [
            "dataset_id",
            "metadata_cohort",
            "dataset_label",
            "disease_donors",
            "control_donors",
            "disease_donors_min_cells",
            "control_donors_min_cells",
            "minimum_cells_per_profile",
            "minimum_donors_per_condition",
            "status",
            "excluded_donors",
        ],
    )
    manifest = {
        "schema_version": "1.0",
        "frozen": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "disease_id": disease_id,
        "masked_genes": sorted(masked_genes),
        "selection_used_expression_results": False,
        "expression_matrix_loaded_before_freeze": False,
        "selection_inputs": {
            "diseases": input_record(inputs["diseases"]),
            "hpo_cpo": input_record(inputs["hpo_cpo"]),
            "go_functions": input_record(inputs["go_functions"]),
            "gold_standard": input_record(inputs.get("gold_standard") or inputs["gold_models"]),
            "metadata": input_record(settings["metadata"]),
            "label_map": input_record(settings["label_map"]),
            "cl_obo": input_record(settings["cl_obo"]),
        },
        "predeclared_gates": {
            "min_function_genes_after_masking": min_genes,
            "max_ontology_distance": max_distance,
            "min_donors_per_condition": int(settings.get("min_donors_per_condition", 4)),
            "min_cells_per_donor_cell": int(settings.get("min_cells_per_donor_cell", 20)),
        },
        "predictions": prediction_rows,
        "interpretation": {
            "strict_primary": "Leakage-controlled primary endpoint.",
            "practical_positive_control": "Biological positive control; not evidence of new-gene generalization.",
        },
    }
    write_json(output / "case_manifest.json", manifest)
    return manifest
