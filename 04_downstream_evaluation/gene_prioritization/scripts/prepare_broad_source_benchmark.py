from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / '04_downstream_evaluation/mouse_model_prioritization'


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def split_values(value: str) -> set[str]:
    return {
        token.strip().upper()
        for token in re.split(r"[|;,]", value)
        if token.strip()
    }


def model_gene_tokens(row: Mapping[str, str]) -> set[str]:
    return split_values(row.get("human_gene_symbol", "")) | split_values(
        row.get("mouse_gene_symbol", "")
    )


def build_snapshot(args: argparse.Namespace) -> dict[str, object]:
    eligible = read_tsv(args.eligible_candidates)
    primary = read_tsv(args.primary_candidates)
    primary_ids = {row["disease_id"] for row in primary}
    cohort = [row for row in eligible if row["disease_id"] not in primary_ids]
    if len(cohort) != args.expected_diseases:
        raise ValueError(
            f"Expected {args.expected_diseases} non-primary diseases, found {len(cohort)}"
        )

    cohort_by_id = {row["disease_id"]: row for row in cohort}
    if len(cohort_by_id) != len(cohort):
        raise ValueError("Broad cohort contains duplicate disease IDs")
    cohort_ids = set(cohort_by_id)
    cohort_genes = {row["gene_symbol"].upper() for row in cohort}

    disease_rows = [
        row for row in read_tsv(args.disease_profiles) if row["disease_id"] in cohort_ids
    ]
    profile_ids = {row["disease_id"] for row in disease_rows}
    missing_profiles = sorted(cohort_ids - profile_ids)
    if missing_profiles:
        raise ValueError(f"Missing disease profiles: {', '.join(missing_profiles)}")

    all_models = read_tsv(args.models)
    model_rows = [
        row
        for row in all_models
        if model_gene_tokens(row) & cohort_genes
    ]
    model_ids = {row["model_id"] for row in model_rows}
    phenotype_rows = [
        row
        for row in read_tsv(args.model_phenotypes)
        if row.get("model_id", "") in model_ids
    ]
    phenotype_model_ids = {row["model_id"] for row in phenotype_rows}

    models_by_disease: dict[str, list[dict[str, str]]] = defaultdict(list)
    for model in model_rows:
        for disease_id in split_values(model.get("known_disease_ids", "")):
            if disease_id in cohort_ids:
                models_by_disease[disease_id].append(model)

    gold_model_rows: list[dict[str, object]] = []
    gold_rows: list[dict[str, object]] = []
    unavailable_rows: list[dict[str, object]] = []
    source_gold_fields = [
        "eligible_rank",
        "disease_id",
        "disease_name",
        "gene_symbol",
        *list(model_rows[0]),
        "gold_source",
        "gold_status",
    ]
    source_gold_fields = list(dict.fromkeys(source_gold_fields))

    for candidate in sorted(cohort, key=lambda row: int(row["eligible_rank"])):
        disease_id = candidate["disease_id"]
        gene = candidate["gene_symbol"].upper()
        source_gold = [
            model
            for model in models_by_disease.get(disease_id, ())
            if gene in model_gene_tokens(model)
        ]
        if not source_gold:
            raise ValueError(f"No source MGI gold model for {disease_id} ({gene})")
        for model in source_gold:
            gold_model_rows.append(
                {
                    "eligible_rank": candidate["eligible_rank"],
                    "disease_id": disease_id,
                    "disease_name": candidate["disease_name"],
                    "gene_symbol": gene,
                    **model,
                    "gold_source": "MGI known_disease_ids",
                    "gold_status": "source_gold_unreviewed",
                }
            )

        phenotype_bearing_single_gene_models = [
            model
            for model in model_rows
            if model["model_id"] in phenotype_model_ids
            and model.get("human_gene_symbol", "").strip().upper() == gene
        ]
        evaluable = bool(phenotype_bearing_single_gene_models)
        status = (
            "source_label_evaluable"
            if evaluable
            else "not_evaluable_no_phenotype_bearing_single_gene_model"
        )
        gold_rows.append(
            {
                "eligible_rank": candidate["eligible_rank"],
                "disease_id": disease_id,
                "disease_name": candidate["disease_name"],
                "gene_symbol": gene,
                "is_evaluation_positive": str(evaluable).lower(),
                "evaluation_status": status,
                "source_gold_model_count": len(source_gold),
                "evidence_level": "HPO_disease_gene_plus_MGI_source_association_unreviewed",
            }
        )
        if not evaluable:
            unavailable_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": candidate["disease_name"],
                    "gene_symbol": gene,
                    "reason": status,
                }
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cohort_fields = list(eligible[0]) + ["broad_cohort_status"]
    write_tsv(
        output / "cohort.tsv",
        ({**row, "broad_cohort_status": "included_non_primary_eligible"} for row in cohort),
        cohort_fields,
    )
    write_tsv(output / "diseases.tsv", disease_rows, list(disease_rows[0]))
    write_tsv(output / "models.tsv", model_rows, list(model_rows[0]))
    write_tsv(
        output / "model_phenotypes.tsv", phenotype_rows, list(phenotype_rows[0])
    )
    write_tsv(output / "source_gold_models.tsv", gold_model_rows, source_gold_fields)
    write_tsv(
        output / "gold_standard.tsv",
        gold_rows,
        [
            "eligible_rank",
            "disease_id",
            "disease_name",
            "gene_symbol",
            "is_evaluation_positive",
            "evaluation_status",
            "source_gold_model_count",
            "evidence_level",
        ],
    )
    write_tsv(
        output / "non_evaluable_diseases.tsv",
        unavailable_rows,
        ["disease_id", "disease_name", "gene_symbol", "reason"],
    )

    generated_files = [
        "cohort.tsv",
        "diseases.tsv",
        "models.tsv",
        "model_phenotypes.tsv",
        "source_gold_models.tsv",
        "gold_standard.tsv",
        "non_evaluable_diseases.tsv",
    ]
    single_gene_models = [
        row
        for row in model_rows
        if row.get("human_gene_symbol", "").strip()
        and ";" not in row.get("human_gene_symbol", "")
        and row["model_id"] in phenotype_model_ids
    ]
    manifest = {
        "schema_version": "1.0",
        "dataset": "broad HPO-MGI source-label evaluation cohort excluding benchmark v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_labels_frozen": True,
        "frozen": False,
        "middle_layer_independent": True,
        "publication_grade": False,
        "review_policy": "source_hpo_mgi_unreviewed",
        "evaluation_mode": "source_label_broad",
        "disease_count": len(cohort),
        "evaluable_disease_count": sum(
            row["is_evaluation_positive"] == "true" for row in gold_rows
        ),
        "non_evaluable_disease_count": len(unavailable_rows),
        "source_gene_count": len(cohort_genes),
        "phenotype_bearing_single_gene_count": len(
            {row["human_gene_symbol"].upper() for row in single_gene_models}
        ),
        "candidate_model_count": len(model_rows),
        "phenotype_bearing_single_gene_model_count": len(single_gene_models),
        "source_gold_model_count": len(gold_model_rows),
        "selection_rule": (
            "Every entry in the benchmark_v4 eligible pool whose disease_id is not in "
            "the final 50-disease primary cohort. No CPO input or ranking result was used."
        ),
        "interpretation_boundary": (
            "Source-label broad evaluation only. Disease-gene labels and MGI model "
            "associations have not received per-disease literature review or human signoff."
        ),
        "inputs": {
            "eligible_candidates": file_record(args.eligible_candidates),
            "primary_candidates": file_record(args.primary_candidates),
            "disease_profiles": file_record(args.disease_profiles),
            "models": file_record(args.models),
            "model_phenotypes": file_record(args.model_phenotypes),
        },
        "files": {
            name: file_record(output / name) for name in generated_files
        },
    }
    manifest_path = output / "source_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the non-primary broad HPO-MGI source-label evaluation snapshot."
    )
    benchmark_v4 = SOURCE_ROOT / "data/benchmark/benchmark_v4"
    parser.add_argument(
        "--eligible-candidates",
        type=Path,
        default=benchmark_v4 / "eligible_candidates.tsv",
    )
    parser.add_argument(
        "--primary-candidates",
        type=Path,
        default=benchmark_v4 / "candidate_diseases.tsv",
    )
    parser.add_argument(
        "--disease-profiles",
        type=Path,
        default=SOURCE_ROOT / "data/processed/diseases/diseases.all.tsv",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=SOURCE_ROOT / "data/processed/mgi_v3/models.tsv",
    )
    parser.add_argument(
        "--model-phenotypes",
        type=Path,
        default=SOURCE_ROOT / "data/processed/mgi_v3/model_phenotypes.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/benchmark/broad_source_435_v1",
    )
    parser.add_argument("--expected-diseases", type=int, default=435)
    return parser.parse_args()


def main() -> None:
    manifest = build_snapshot(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
