from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import dump_config
from ..io import (
    as_bool,
    input_record,
    read_tsv,
    split_values,
    value,
    write_json,
    write_tsv,
)


STATUS_FIELDS = [
    "candidate_case_id",
    "study_accession",
    "pareto_front",
    "priority_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "data_tier",
    "patient_donors",
    "control_donors",
    "selected_predicted_cl_ids",
    "selected_dataset_labels",
    "cell_mapping_type",
    "candidate_gene_on_feature_axis",
    "primary_function_id",
    "primary_function_name",
    "practical_function_gene_count",
    "practical_feature_gene_count",
    "practical_feature_coverage",
    "strict_function_gene_count",
    "strict_feature_gene_count",
    "strict_feature_coverage",
    "evidence_overlap",
    "access_restriction",
    "status",
    "reasons",
    "limitations",
]

CELL_FIELDS = [
    "candidate_case_id",
    "study_accession",
    "predicted_cl_id",
    "predicted_cell_name",
    "dataset_label",
    "mapping_type",
    "ontology_distance",
    "status",
]

FUNCTION_FIELDS = [
    "candidate_case_id",
    "study_accession",
    "function_id",
    "function_name",
    "candidate_gene",
    "bridge_source_gene_count",
    "practical_gene_count",
    "practical_feature_count",
    "practical_feature_coverage",
    "strict_gene_count",
    "strict_feature_count",
    "strict_feature_coverage",
    "status",
]


def _int(row: Mapping[str, str], name: str) -> int:
    try:
        return int(float(value(row, name, default="0")))
    except ValueError:
        return 0


def donor_tier(patient: int, control: int, design: str) -> str:
    if patient >= 4 and control >= 4:
        return "A"
    if patient >= 2 and control >= 2:
        return "B"
    if patient >= 1 and (control >= 1 or "longitudinal" in design.lower()):
        return "C"
    return "D"


def _feature_names(paths: tuple[str, ...], root: Path) -> tuple[set[str], list[Path]]:
    per_file: list[set[str]] = []
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        opener = gzip.open if path.suffix == ".gz" else open
        names: set[str] = set()
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if not fields:
                    continue
                names.add((fields[1] if len(fields) > 1 else fields[0]).upper())
        per_file.append(names)
        resolved.append(path)
    if not per_file:
        return set(), resolved
    common = set(per_file[0])
    for names in per_file[1:]:
        common.intersection_update(names)
    return common, resolved


def _function_genes(path: str | Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in read_tsv(path):
        function = value(row, "go_id", "function_id")
        gene = value(row, "human_gene_symbol", "gene_symbol").upper()
        if function and gene:
            result.setdefault(function, set()).add(gene)
    return result


def preflight_rows(
    candidates: list[Mapping[str, str]],
    catalog: list[Mapping[str, str]],
    functions: Mapping[str, set[str]],
    project_root: Path,
    min_practical_coverage: float = 0.8,
    min_strict_genes: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_id = {value(row, "candidate_case_id"): row for row in candidates}
    statuses: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    for dataset in catalog:
        case_id = value(dataset, "candidate_case_id")
        candidate = candidate_by_id[case_id]
        gene = value(candidate, "gene_symbol").upper()
        function_id = value(candidate, "primary_predicted_function")
        practical = set(functions.get(function_id, set()))
        bridge = {item.upper() for item in split_values(value(candidate, "bridge_source_genes"))}
        strict = practical - bridge - {gene}
        not_found = value(dataset, "retrieval_status") == "not_found"
        feature_paths = split_values(value(dataset, "feature_axis_paths"))
        features: set[str] = set()
        reasons: list[str] = []
        if feature_paths:
            try:
                features, _ = _feature_names(feature_paths, project_root)
            except FileNotFoundError:
                reasons.append("feature_axis_file_missing")
        elif not not_found:
            reasons.append("feature_axis_not_verified")
        practical_present = practical & features
        strict_present = strict & features
        practical_coverage = len(practical_present) / len(practical) if practical else 0.0
        strict_coverage = len(strict_present) / len(strict) if strict else 0.0
        gene_present = gene in features
        patient = _int(dataset, "patient_donor_count")
        control = _int(dataset, "control_donor_count")
        design = value(dataset, "longitudinal_or_cross_sectional")
        if not_found:
            reasons.append("no_public_single_cell_dataset_found")
        if not not_found and value(dataset, "species").lower() != "homo sapiens":
            reasons.append("not_human")
        if not not_found and value(dataset, "causal_gene_match") != "exact":
            reasons.append("disease_or_causal_gene_mismatch")
        if not not_found and (patient < 1 or (control < 1 and "longitudinal" not in design.lower())):
            reasons.append("no_interpretable_comparator")
        if not not_found and not as_bool(dataset.get("donor_metadata_available", "")):
            reasons.append("donor_metadata_unavailable")
        if not not_found and not (
            as_bool(dataset.get("raw_count_available", ""))
            or as_bool(dataset.get("processed_matrix_available", ""))
        ):
            reasons.append("count_matrix_unavailable")
        if not not_found and not as_bool(dataset.get("cell_labels_available", "")):
            reasons.append("cell_labels_unavailable")
        if not not_found and not as_bool(dataset.get("possible_tissue_match", "")):
            reasons.append("tissue_mismatch")
        if not not_found and not as_bool(dataset.get("possible_cell_match", "")):
            reasons.append("primary_predicted_cell_absent")
        if not not_found and value(dataset, "access_restriction") not in {"", "none", "public"}:
            reasons.append("access_restricted")
        if not not_found and feature_paths and not gene_present:
            reasons.append("candidate_gene_absent_from_feature_axis")
        if not not_found and practical and practical_coverage < min_practical_coverage:
            reasons.append("practical_function_feature_coverage_low")
        if not not_found and len(strict) < min_strict_genes:
            reasons.append("strict_function_gene_set_too_small")
        if not not_found and feature_paths and len(strict_present) < min_strict_genes:
            reasons.append("strict_function_feature_count_too_small")
        selected_cl = split_values(value(dataset, "matched_predicted_cl_ids"))
        selected_labels = split_values(value(dataset, "matched_dataset_labels"))
        limitations = list(split_values(value(dataset, "notes")))
        if value(dataset, "evidence_overlap") not in {"", "none"}:
            limitations.append("validation study overlaps marker-literature source corpus")
        if value(dataset, "cell_count_evidence") != "donor_label_counts_verified":
            limitations.append("donor-by-label cell counts require post-freeze verification")
        if as_bool(dataset.get("outcome_known_before_freeze", "")):
            limitations.append("publication-level outcome was known before follow-up freeze")
        status = "pass" if not reasons else "fail"
        statuses.append(
            {
                "candidate_case_id": case_id,
                "study_accession": value(dataset, "study_accession"),
                "pareto_front": value(candidate, "pareto_front"),
                "priority_rank": value(candidate, "priority_rank"),
                "disease_id": value(candidate, "disease_id"),
                "disease_name": value(candidate, "disease_name"),
                "gene_symbol": gene,
                "data_tier": donor_tier(patient, control, design),
                "patient_donors": patient,
                "control_donors": control,
                "selected_predicted_cl_ids": selected_cl,
                "selected_dataset_labels": selected_labels,
                "cell_mapping_type": value(dataset, "cell_mapping_type", default="curated_metadata_mapping"),
                "candidate_gene_on_feature_axis": gene_present,
                "primary_function_id": function_id,
                "primary_function_name": value(candidate, "primary_predicted_function_name"),
                "practical_function_gene_count": len(practical),
                "practical_feature_gene_count": len(practical_present),
                "practical_feature_coverage": practical_coverage,
                "strict_function_gene_count": len(strict),
                "strict_feature_gene_count": len(strict_present),
                "strict_feature_coverage": strict_coverage,
                "evidence_overlap": value(dataset, "evidence_overlap", default="none"),
                "access_restriction": value(dataset, "access_restriction"),
                "status": status,
                "reasons": sorted(set(reasons)),
                "limitations": sorted(set(limitations)),
            }
        )
        predicted_names = split_values(value(candidate, "primary_predicted_cell_names"))
        for index, cl_id in enumerate(selected_cl):
            cells.append(
                {
                    "candidate_case_id": case_id,
                    "study_accession": value(dataset, "study_accession"),
                    "predicted_cl_id": cl_id,
                    "predicted_cell_name": predicted_names[index] if index < len(predicted_names) else cl_id,
                    "dataset_label": selected_labels[index] if index < len(selected_labels) else "",
                    "mapping_type": value(dataset, "cell_mapping_type", default="curated_metadata_mapping"),
                    "ontology_distance": value(dataset, "ontology_distance"),
                    "status": "pass" if as_bool(dataset.get("possible_cell_match", "")) else "fail",
                }
            )
        function_rows.append(
            {
                "candidate_case_id": case_id,
                "study_accession": value(dataset, "study_accession"),
                "function_id": function_id,
                "function_name": value(candidate, "primary_predicted_function_name"),
                "candidate_gene": gene,
                "bridge_source_gene_count": len(bridge),
                "practical_gene_count": len(practical),
                "practical_feature_count": len(practical_present),
                "practical_feature_coverage": practical_coverage,
                "strict_gene_count": len(strict),
                "strict_feature_count": len(strict_present),
                "strict_feature_coverage": strict_coverage,
                "status": "pass" if practical_coverage >= min_practical_coverage and len(strict_present) >= min_strict_genes else "fail",
            }
        )
    statuses.sort(key=lambda row: (int(row["pareto_front"]), int(row["priority_rank"]), row["study_accession"]))
    return statuses, cells, function_rows


def run_metadata_preflight(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    settings = config.get("metadata_preflight", {})
    candidate_path = Path(inputs["candidate_table"])
    catalog_path = Path(inputs["dataset_catalog"])
    function_path = Path(inputs["go_functions"])
    statuses, cells, functions = preflight_rows(
        list(read_tsv(candidate_path)),
        list(read_tsv(catalog_path)),
        _function_genes(function_path),
        Path(config["_project_root"]),
        float(settings.get("min_practical_feature_coverage", 0.8)),
        int(settings.get("min_strict_function_genes", 5)),
    )
    write_tsv(output / "case_dataset_pair_status.tsv", statuses, STATUS_FIELDS)
    write_tsv(output / "dataset_metadata_preflight.tsv", statuses, STATUS_FIELDS)
    write_tsv(output / "cell_mapping_preflight.tsv", cells, CELL_FIELDS)
    write_tsv(output / "function_feature_coverage.tsv", functions, FUNCTION_FIELDS)
    failures = [row for row in statuses if row["status"] != "pass"]
    write_tsv(output / "preflight_failures.tsv", failures, STATUS_FIELDS)
    access_log_path = output / "expression_access_log.tsv"
    access_log = list(read_tsv(access_log_path)) if access_log_path.is_file() else []
    access_log.extend([
        {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "stage": "metadata_preflight",
            "resource": str(catalog_path),
            "access_type": "metadata_and_feature_names_only",
            "expression_matrix_values_accessed": False,
            "used_for_selection": True,
            "note": "Published labels, donor metadata, file availability and feature names only.",
        },
        {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "stage": "source_overlap_audit",
            "resource": "marker_literature source index and generic cluster-marker table header",
            "access_type": "derived_source_audit",
            "expression_matrix_values_accessed": False,
            "used_for_selection": False,
            "note": "No candidate-gene or case-control result was used; exact study overlap was recorded.",
        },
    ])
    write_tsv(
        access_log_path,
        access_log,
        [
            "event_time",
            "stage",
            "resource",
            "access_type",
            "expression_matrix_values_accessed",
            "used_for_selection",
            "note",
        ],
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expression_matrix_values_accessed": False,
        "pair_count": len(statuses),
        "passing_pair_count": sum(row["status"] == "pass" for row in statuses),
        "inputs": {
            "candidate_table": input_record(candidate_path),
            "dataset_catalog": input_record(catalog_path),
            "go_functions": input_record(function_path),
        },
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    return manifest
