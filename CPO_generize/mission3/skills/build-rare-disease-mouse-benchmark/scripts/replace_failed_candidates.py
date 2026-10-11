#!/usr/bin/env python3
"""Replace failed benchmark entries from deterministic same-stratum reserves."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import input_record, read_json, read_tsv, split_values, write_json, write_tsv
from select_candidates import CANDIDATE_FIELDS, GOLD_FIELDS, inheritance_class


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_SOURCE = DEFAULT_ROOT / "mission3/data/benchmark/benchmark_v1"
DEFAULT_OUTPUT = DEFAULT_ROOT / "mission3/data/benchmark/benchmark_v2"

REPLACEMENT_FIELDS = [
    "selection_rank",
    "excluded_disease_id",
    "excluded_disease_name",
    "excluded_gene_symbol",
    "exclusion_errors",
    "replacement_disease_id",
    "replacement_disease_name",
    "replacement_gene_symbol",
    "phenotype_system_id",
    "phenotype_system",
    "inheritance_class",
    "reserve_stratum_rank",
    "selection_score",
]

CORRECTION_FIELDS = [
    "selection_rank",
    "disease_id",
    "model_id",
    "field",
    "old_value",
    "new_value",
]

PACKET_MODEL_FIELDS = [
    "model_name",
    "mouse_gene_symbol",
    "human_gene_symbol",
    "allele_id",
    "allele_type",
    "zygosity",
    "conditional",
    "genetic_background",
    "known_disease_ids",
]


def parse_ranks(value: str) -> list[int]:
    ranks = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not ranks:
        raise ValueError("--invalid-ranks must contain at least one rank")
    return ranks


def gene_tokens(row: dict[str, str]) -> set[str]:
    return {
        token.upper()
        for field in ("human_gene_symbol", "mouse_gene_symbol")
        for token in split_values(row.get(field, ""))
    }


def choose_replacements(
    candidates: list[dict[str, str]],
    eligible_pool: list[dict[str, str]],
    invalid_ranks: list[int],
    validation_by_rank: dict[int, dict[str, str]],
    *,
    max_per_gene: int,
    excluded_disease_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    current_by_rank = {int(row["selection_rank"]): row for row in candidates}
    unknown = sorted(set(invalid_ranks).difference(current_by_rank))
    if unknown:
        raise ValueError(f"Unknown invalid ranks: {unknown}")

    invalid_set = set(invalid_ranks)
    retained = [
        dict(row) for row in candidates if int(row["selection_rank"]) not in invalid_set
    ]
    blocked_disease_ids = {
        row["disease_id"] for row in candidates
    } | set(excluded_disease_ids or set())
    gene_counts = Counter(row["gene_symbol"].upper() for row in retained)
    used_reserves: set[str] = set()
    replacements: list[dict[str, str]] = []

    for rank in invalid_ranks:
        excluded = current_by_rank[rank]
        stratum = (
            excluded["phenotype_system_id"],
            inheritance_class(excluded["inheritance"]),
        )
        choices = sorted(
            (
                row
                for row in eligible_pool
                if row["disease_id"] not in blocked_disease_ids
                and row["disease_id"] not in used_reserves
                and row["phenotype_system_id"] == stratum[0]
                and row["inheritance_class"] == stratum[1]
                and gene_counts[row["gene_symbol"].upper()] < max_per_gene
            ),
            key=lambda row: (
                int(row["stratum_rank"]),
                -float(row["selection_score"]),
                row["disease_id"],
            ),
        )
        if not choices:
            raise ValueError(
                "No same-stratum eligible reserve for "
                f"rank {rank} ({stratum[0]}, {stratum[1]})"
            )
        reserve = choices[0]
        retained.append(
            {
                **{field: reserve.get(field, "") for field in CANDIDATE_FIELDS},
                "selection_rank": str(rank),
                "selection_status": "replacement_after_research_exclusion",
            }
        )
        used_reserves.add(reserve["disease_id"])
        gene_counts[reserve["gene_symbol"].upper()] += 1
        validation = validation_by_rank.get(rank, {})
        replacements.append(
            {
                "selection_rank": str(rank),
                "excluded_disease_id": excluded["disease_id"],
                "excluded_disease_name": excluded["disease_name"],
                "excluded_gene_symbol": excluded["gene_symbol"],
                "exclusion_errors": validation.get("errors", "not_machine_valid"),
                "replacement_disease_id": reserve["disease_id"],
                "replacement_disease_name": reserve["disease_name"],
                "replacement_gene_symbol": reserve["gene_symbol"],
                "phenotype_system_id": stratum[0],
                "phenotype_system": reserve["phenotype_system"],
                "inheritance_class": stratum[1],
                "reserve_stratum_rank": reserve["stratum_rank"],
                "selection_score": reserve["selection_score"],
            }
        )
    return sorted(retained, key=lambda row: int(row["selection_rank"])), replacements


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    source = args.source_batch.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    invalid_ranks = parse_ranks(args.invalid_ranks)
    source_manifest = read_json(source / "selection_manifest.json")
    candidates = read_tsv(source / "candidate_diseases.tsv")
    eligible_pool = read_tsv(args.eligible_pool)
    validation_rows = read_tsv(source / "curation/review_status.tsv")
    validation_by_rank = {int(row["selection_rank"]): row for row in validation_rows}
    for rank in invalid_ranks:
        status = validation_by_rank.get(rank)
        if not status or status.get("valid", "").lower() == "true":
            raise ValueError(f"Rank {rank} is not recorded as an invalid reviewed entry")

    max_per_gene = int(
        source_manifest["selection_protocol"]["maximum_diseases_per_gene"]
    )
    prior_replacements = source_manifest.get(
        "replacement_history", source_manifest.get("replacements", [])
    )
    prior_excluded_disease_ids = {
        str(row["excluded_disease_id"])
        for row in prior_replacements
        if row.get("excluded_disease_id")
    }
    selected, replacements = choose_replacements(
        candidates,
        eligible_pool,
        invalid_ranks,
        validation_by_rank,
        max_per_gene=max_per_gene,
        excluded_disease_ids=prior_excluded_disease_ids,
    )

    output.mkdir(parents=True)
    write_tsv(output / "candidate_diseases.tsv", selected, CANDIDATE_FIELDS)
    shutil.copyfile(args.eligible_pool, output / "eligible_candidates.tsv")

    input_paths = source_manifest["inputs"]
    disease_path = root / input_paths["human_disease_profiles"]["path"]
    model_path = (
        args.models.resolve()
        if args.models
        else root / input_paths["mouse_models"]["path"]
    )
    phenotype_path = (
        args.model_phenotypes.resolve()
        if args.model_phenotypes
        else root / input_paths["mouse_model_phenotypes_for_runnability_only"]["path"]
    )
    disease_rows = read_tsv(disease_path)
    model_rows = read_tsv(model_path)
    selected_ids = {row["disease_id"] for row in selected}
    selected_genes = {row["gene_symbol"].upper() for row in selected}
    profile_rows = [row for row in disease_rows if row["disease_id"] in selected_ids]
    selected_models = [row for row in model_rows if gene_tokens(row) & selected_genes]
    selected_model_ids = {row["model_id"] for row in selected_models}

    candidate_by_id = {row["disease_id"]: row for row in selected}
    gold_rows = []
    for model in selected_models:
        known_ids = set(split_values(model.get("known_disease_ids", "")))
        for disease_id in sorted(selected_ids.intersection(known_ids)):
            candidate = candidate_by_id[disease_id]
            gold_rows.append(
                {
                    "selection_rank": candidate["selection_rank"],
                    "disease_id": disease_id,
                    "disease_name": candidate["disease_name"],
                    "gene_symbol": candidate["gene_symbol"],
                    **model,
                    "gold_source": "MGI known_disease_ids",
                    "gold_status": "source_gold_pending_codex_review",
                }
            )
    phenotype_rows = [
        row for row in read_tsv(phenotype_path) if row["model_id"] in selected_model_ids
    ]
    write_tsv(
        output / "human_disease_profiles.source.tsv",
        profile_rows,
        list(profile_rows[0]),
    )
    write_tsv(output / "gold_models.source.tsv", gold_rows, GOLD_FIELDS)
    write_tsv(output / "candidate_models.tsv", selected_models, list(selected_models[0]))
    write_tsv(
        output / "evaluation_inputs/model_phenotypes.tsv",
        phenotype_rows,
        list(phenotype_rows[0]),
    )

    old_gold = {
        (row["disease_id"], row["model_id"]): row
        for row in read_tsv(source / "gold_models.source.tsv")
        if row["disease_id"] in selected_ids
    }
    new_gold = {
        (row["disease_id"], row["model_id"]): row for row in gold_rows
    }
    corrections = []
    correction_ranks: set[int] = set()
    for disease_id, model_id in sorted(set(old_gold) | set(new_gold)):
        old_row = old_gold.get((disease_id, model_id), {})
        new_row = new_gold.get((disease_id, model_id), {})
        for field in PACKET_MODEL_FIELDS:
            old_value = str(old_row.get(field, ""))
            new_value = str(new_row.get(field, ""))
            if old_value == new_value:
                continue
            rank = int(candidate_by_id[disease_id]["selection_rank"])
            correction_ranks.add(rank)
            corrections.append(
                {
                    "selection_rank": rank,
                    "disease_id": disease_id,
                    "model_id": model_id,
                    "field": field,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )

    retained_ranks = {
        int(row["selection_rank"])
        for row in selected
        if int(row["selection_rank"]) not in set(invalid_ranks)
    }
    source_outputs = source / "raw_codex_outputs"
    target_outputs = output / "raw_codex_outputs"
    copied_review_ranks: set[int] = set()
    for path in sorted(source_outputs.iterdir()):
        if not path.is_dir():
            continue
        try:
            rank = int(path.name.split("_", 1)[0])
        except ValueError:
            continue
        status_path = path / "status.json"
        final_path = path / "final.json"
        completed = (
            status_path.is_file()
            and final_path.is_file()
            and read_json(status_path).get("status") == "completed"
        )
        if rank in retained_ranks and rank not in correction_ranks and completed:
            shutil.copytree(path, target_outputs / path.name)
            copied_review_ranks.add(rank)

    required_research_ranks = sorted(
        {int(row["selection_rank"]) for row in selected}.difference(copied_review_ranks)
    )

    audit_dir = output / "replacement_audit"
    write_tsv(audit_dir / "replacements.tsv", replacements, REPLACEMENT_FIELDS)
    write_tsv(
        audit_dir / "research_input_corrections.tsv",
        corrections,
        CORRECTION_FIELDS,
    )
    write_json(
        audit_dir / "replacement_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_batch": str(source.relative_to(root)),
            "invalid_ranks": invalid_ranks,
            "selection_rule": "next eligible disease in the same phenotype-system and inheritance stratum",
            "middle_layer_independent": True,
            "research_input_changed_ranks": sorted(correction_ranks),
            "copied_completed_review_ranks": sorted(copied_review_ranks),
            "required_research_ranks": required_research_ranks,
            "replacements": replacements,
            "prior_excluded_disease_ids": sorted(prior_excluded_disease_ids),
        },
    )

    manifest = {
        **source_manifest,
        "schema_version": "1.1",
        "benchmark_version": output.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_set_replaced_after_research_exclusion",
        "frozen": False,
        "selected_disease_count": len(selected),
        "selected_disease_ids": [row["disease_id"] for row in selected],
        "parent_selection_manifest": input_record(source / "selection_manifest.json", root),
        "eligible_candidate_pool": input_record(args.eligible_pool, root),
        "replacement_protocol": {
            "same_phenotype_system": True,
            "same_inheritance_class": True,
            "order": ["stratum_rank", "selection_score_desc", "disease_id"],
            "middle_layer_independent": True,
        },
        "replacements": replacements,
        "replacement_history": [*prior_replacements, *replacements],
        "research_input_changed_ranks": sorted(correction_ranks),
        "copied_completed_review_ranks": sorted(copied_review_ranks),
        "required_research_ranks": required_research_ranks,
        "inputs": {
            **source_manifest["inputs"],
            "mouse_models": input_record(model_path, root),
            "mouse_model_phenotypes_for_runnability_only": input_record(
                phenotype_path, root
            ),
        },
        "outputs": {},
        "phenotype_system_counts": dict(
            sorted(Counter(row["phenotype_system"] for row in selected).items())
        ),
        "inheritance_class_counts": dict(
            sorted(Counter(inheritance_class(row["inheritance"]) for row in selected).items())
        ),
    }
    write_json(output / "selection_manifest.json", manifest)
    manifest["outputs"] = {
        str(path.relative_to(output)): input_record(path, root)
        for path in sorted(output.rglob("*.tsv"))
    }
    write_json(output / "selection_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-batch", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eligible-pool", type=Path, required=True)
    parser.add_argument("--invalid-ranks", required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--model-phenotypes", type=Path)
    return parser.parse_args()


def main() -> None:
    manifest = prepare(parse_args())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_disease_count": manifest["selected_disease_count"],
                "replacement_count": len(manifest["replacements"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
