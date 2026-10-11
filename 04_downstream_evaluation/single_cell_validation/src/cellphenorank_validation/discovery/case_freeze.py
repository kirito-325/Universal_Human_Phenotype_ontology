from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from cellphenorank.config import dump_config
from cellphenorank.io import input_record, read_tsv, split_values, value, write_json, write_tsv


REGISTRY_FIELDS = [
    "case_id",
    "candidate_case_id",
    "study_accession",
    "role",
    "pareto_front",
    "data_tier",
    "gene_symbol",
    "manifest_path",
    "manifest_sha256",
    "status",
]

ENDPOINT_FIELDS = ["endpoint_id", "endpoint_role", "target", "status_at_freeze"]


def _tier_order(tier: str) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(tier, 9)


def _int(row: Mapping[str, str], key: str) -> int:
    return int(float(value(row, key, default="0")))


def select_primary_case(rows: list[Mapping[str, str]]) -> Mapping[str, str] | None:
    passing = [row for row in rows if value(row, "status") == "pass"]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda row: (
            _int(row, "pareto_front"),
            _tier_order(value(row, "data_tier")),
            _int(row, "priority_rank"),
            value(row, "study_accession"),
        ),
    )[0]


def run_case_freeze(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    candidate_path = Path(inputs["candidate_table"])
    catalog_path = Path(inputs["dataset_catalog"])
    status_path = Path(inputs["case_dataset_status"])
    candidates = {row["candidate_case_id"]: row for row in read_tsv(candidate_path)}
    catalog = {
        (row["candidate_case_id"], row["study_accession"]): row
        for row in read_tsv(catalog_path)
    }
    statuses = list(read_tsv(status_path))
    primary = select_primary_case(statuses)
    if primary is None:
        raise ValueError("No metadata-preflight-passing case can be frozen")
    access_log = Path(config["_project_root"]) / "outputs/single_cell_case_exploration/preflight/expression_access_log.tsv"
    accessed_resources = [
        value(row, "resource").lower()
        for row in read_tsv(access_log)
        if value(row, "expression_matrix_values_accessed") == "true"
    ] if access_log.is_file() else []
    registry: list[dict[str, Any]] = []
    passing = [row for row in statuses if value(row, "status") == "pass"]
    for status in passing:
        candidate = candidates[value(status, "candidate_case_id")]
        dataset = catalog[(value(status, "candidate_case_id"), value(status, "study_accession"))]
        case_id = f"{value(status, 'candidate_case_id')}__{value(status, 'study_accession').replace('-', '_')}"
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = case_dir / "case_freeze_manifest.yaml"
        checksum_path = case_dir / "manifest.sha256"
        if manifest_path.is_file() and checksum_path.is_file():
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            expected = checksum_path.read_text(encoding="utf-8").split()[0]
            if digest != expected:
                raise RuntimeError(f"Existing frozen manifest checksum mismatch: {case_id}")
            existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            registry.append(
                {
                    "case_id": case_id,
                    "candidate_case_id": value(status, "candidate_case_id"),
                    "study_accession": value(status, "study_accession"),
                    "role": existing["case"]["role"],
                    "pareto_front": existing["ranking"]["pareto_front"],
                    "data_tier": existing["single_cell"]["data_tier"],
                    "gene_symbol": existing["case"]["candidate_gene"],
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": digest,
                    "status": "frozen_existing",
                }
            )
            continue
        accession = value(status, "study_accession").lower()
        if any(accession and accession in resource for resource in accessed_resources):
            raise RuntimeError(f"Expression values for {accession} were accessed before freeze")
        endpoint_rows = [
            {"endpoint_id": "E0", "endpoint_role": "primary", "target": "candidate_gene_abnormality_in_predicted_cell", "status_at_freeze": "frozen"},
            {"endpoint_id": "E1", "endpoint_role": "primary", "target": "predicted_cell_specificity", "status_at_freeze": "frozen"},
            {"endpoint_id": "E2", "endpoint_role": "primary", "target": "strict_downstream_function_module", "status_at_freeze": "frozen"},
            {"endpoint_id": "E3", "endpoint_role": "secondary", "target": "practical_function_module", "status_at_freeze": "frozen"},
            {"endpoint_id": "E4", "endpoint_role": "secondary", "target": "predicted_cell_composition", "status_at_freeze": "frozen"},
        ]
        write_tsv(case_dir / "frozen_endpoints.tsv", endpoint_rows, ENDPOINT_FIELDS)
        manifest = {
            "schema_version": "1.0",
            "frozen": True,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "selection_used_expression_values": False,
            "case": {
                "case_id": case_id,
                "candidate_case_id": value(status, "candidate_case_id"),
                "disease_id": value(status, "disease_id"),
                "disease_name": value(status, "disease_name"),
                "candidate_gene": value(status, "gene_symbol"),
                "case_mode": "exploratory",
                "role": "primary" if status is primary else "additional_preflight_pass",
            },
            "ranking": {
                "baseline_arm": "A0_current_Exomiser_2602_PhenoDigm",
                "reranked_arm": "A4_extended_CPO",
                "bridge_arm": "A5_bridge_LOO",
                "base_rank": _int(candidate, "base_rank"),
                "final_rank": _int(candidate, "final_rank"),
                "bridge_loo_rank": _int(candidate, "bridge_loo_rank"),
                "rank_change": _int(candidate, "rank_change"),
                "pareto_front": _int(candidate, "pareto_front"),
                "priority_rank": _int(candidate, "priority_rank"),
            },
            "knowledge_prediction": {
                "primary_hpo_ids": list(split_values(value(candidate, "supported_hpo_ids"))),
                "primary_cpo_ids": list(split_values(value(candidate, "cpo_ids"))),
                "predicted_cl_ids": list(split_values(value(status, "selected_predicted_cl_ids"))),
                "predicted_dataset_labels": list(split_values(value(status, "selected_dataset_labels"))),
                "primary_function_id": value(status, "primary_function_id"),
                "primary_function_name": value(status, "primary_function_name"),
                "bridge_source_genes": list(split_values(value(candidate, "bridge_source_genes"))),
                "edge_scope_summary": {
                    "cell_type_specific_paths": _int(candidate, "cell_type_specific_path_count"),
                    "cell_general_paths": _int(candidate, "cell_general_path_count"),
                },
            },
            "single_cell": {
                "study_accession": value(status, "study_accession"),
                "publication_id": value(dataset, "publication_id"),
                "source_url": value(dataset, "source_url"),
                "tissue": value(dataset, "tissue"),
                "assay_type": value(dataset, "assay_type"),
                "patient_donor_count": _int(status, "patient_donors"),
                "control_donor_count": _int(status, "control_donors"),
                "data_tier": value(status, "data_tier"),
                "reference_dataset_labels": list(split_values(value(dataset, "reference_dataset_labels"))),
                "evidence_overlap": value(status, "evidence_overlap"),
                "outcome_known_before_freeze": value(dataset, "outcome_known_before_freeze") == "true",
            },
            "function_sets": {
                "practical_gene_count": _int(status, "practical_function_gene_count"),
                "practical_feature_gene_count": _int(status, "practical_feature_gene_count"),
                "strict_gene_count": _int(status, "strict_function_gene_count"),
                "strict_feature_gene_count": _int(status, "strict_feature_gene_count"),
                "mask_rule": "remove candidate gene and all candidate-level HPO-CPO bridge source genes",
            },
            "endpoints": endpoint_rows,
            "controls": {
                "reference_cells": list(split_values(value(dataset, "reference_dataset_labels"))),
                "expression_matched_null_seed": int(config.get("project", {}).get("random_seed", 2026)),
                "random_gene_set_iterations": int(config.get("case_freeze", {}).get("random_gene_set_iterations", 1000)),
            },
            "limitations_frozen_before_expression": list(split_values(value(status, "limitations"))),
            "inputs": {
                "candidate_table": input_record(candidate_path),
                "dataset_catalog": input_record(catalog_path),
                "case_dataset_status": input_record(status_path),
            },
        }
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        checksum_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")
        registry.append(
            {
                "case_id": case_id,
                "candidate_case_id": value(status, "candidate_case_id"),
                "study_accession": value(status, "study_accession"),
                "role": "primary" if status is primary else "additional_preflight_pass",
                "pareto_front": value(status, "pareto_front"),
                "data_tier": value(status, "data_tier"),
                "gene_symbol": value(status, "gene_symbol"),
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest,
                "status": "frozen",
            }
        )
    write_tsv(output / "frozen_case_registry.tsv", registry, REGISTRY_FIELDS)
    summary = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "frozen_case_count": len(registry),
        "primary_case_id": next(row["case_id"] for row in registry if row["role"] == "primary"),
        "new_case_expression_values_accessed_before_freeze": False,
        "prior_frozen_case_expression_already_accessed": bool(accessed_resources),
    }
    write_json(output / "run_manifest.json", summary)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    return summary
