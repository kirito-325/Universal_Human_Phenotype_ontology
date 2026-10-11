from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import dump_config
from ..io import input_record, read_tsv, value, write_json, write_tsv


CATALOG_FIELDS = [
    "candidate_case_id",
    "pareto_front",
    "priority_rank",
    "gene_symbol",
    "study_accession",
    "publication_id",
    "disease_name",
    "disease_synonym",
    "causal_gene_match",
    "tissue",
    "assay_type",
    "species",
    "patient_donor_count",
    "control_donor_count",
    "other_group_count",
    "longitudinal_or_cross_sectional",
    "raw_count_available",
    "processed_matrix_available",
    "donor_metadata_available",
    "cell_labels_available",
    "variant_metadata_available",
    "access_restriction",
    "possible_tissue_match",
    "possible_cell_match",
    "possible_independence",
    "evidence_overlap",
    "matched_predicted_cl_ids",
    "matched_dataset_labels",
    "reference_dataset_labels",
    "feature_axis_paths",
    "cell_mapping_type",
    "ontology_distance",
    "cell_count_evidence",
    "retrieval_status",
    "source_url",
    "notes",
    "outcome_known_before_freeze",
]


def build_dataset_catalog(
    candidates: list[Mapping[str, str]], discoveries: list[Mapping[str, str]]
) -> list[dict[str, Any]]:
    candidate_by_id = {value(row, "candidate_case_id"): row for row in candidates}
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for discovery in discoveries:
        case_id = value(discovery, "candidate_case_id")
        candidate = candidate_by_id.get(case_id)
        if candidate is None:
            raise ValueError(f"Dataset discovery references unknown candidate: {case_id}")
        accession = value(discovery, "study_accession")
        publication = value(discovery, "publication_id")
        key = (case_id, accession, publication)
        if key in seen:
            continue
        seen.add(key)
        merged = dict(discovery)
        merged.update(
            {
                "candidate_case_id": case_id,
                "pareto_front": value(candidate, "pareto_front"),
                "priority_rank": value(candidate, "priority_rank"),
                "gene_symbol": value(candidate, "gene_symbol"),
                "disease_name": value(
                    discovery, "disease_name", default=value(candidate, "disease_name")
                ),
            }
        )
        rows.append(merged)
    rows.sort(
        key=lambda row: (
            int(value(row, "pareto_front", default="999")),
            int(value(row, "priority_rank", default="999")),
            value(row, "study_accession") or "~",
            value(row, "publication_id"),
        )
    )
    return rows


def run_dataset_catalog(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    candidate_path = Path(inputs["candidate_table"])
    evidence_path = Path(inputs["dataset_evidence"])
    rows = build_dataset_catalog(
        list(read_tsv(candidate_path)), list(read_tsv(evidence_path))
    )
    write_tsv(output / "single_cell_dataset_catalog.tsv", rows, CATALOG_FIELDS)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_used_expression_values": False,
        "candidate_count": len({row["candidate_case_id"] for row in rows}),
        "catalog_row_count": len(rows),
        "public_or_restricted_dataset_rows": sum(
            value(row, "retrieval_status") != "not_found" for row in rows
        ),
        "inputs": {
            "candidate_table": input_record(candidate_path),
            "dataset_evidence": input_record(evidence_path),
        },
    }
    write_json(output / "dataset_catalog_manifest.json", manifest)
    (output / "dataset_catalog_resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    return manifest
