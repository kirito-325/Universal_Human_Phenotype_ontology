#!/usr/bin/env python3
"""Select a fixed rare-disease benchmark candidate set without middle-layer data."""

from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import input_record, read_tsv, split_values, truthy, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_OUTPUT = DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization') / "data" / "benchmark" / "benchmark_v1"

CANDIDATE_FIELDS = [
    "selection_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "inheritance",
    "phenotype_system_id",
    "phenotype_system",
    "hpo_count",
    "source_core_hpo_count",
    "source_core_hpo_ids",
    "surface_hpo_count",
    "surface_weighted_coverage",
    "candidate_model_count",
    "candidate_model_with_phenotype_count",
    "gold_model_count",
    "selection_score",
    "selection_status",
]

ELIGIBLE_FIELDS = [
    "eligible_rank",
    "stratum_rank",
    "selected_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "inheritance",
    "inheritance_class",
    "phenotype_system_id",
    "phenotype_system",
    "hpo_count",
    "source_core_hpo_count",
    "source_core_hpo_ids",
    "surface_hpo_count",
    "surface_weighted_coverage",
    "candidate_model_count",
    "candidate_model_with_phenotype_count",
    "gold_model_count",
    "selection_score",
    "selection_status",
]

GOLD_FIELDS = [
    "selection_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "model_id",
    "model_name",
    "mouse_gene_symbol",
    "human_gene_symbol",
    "allele_id",
    "allele_type",
    "zygosity",
    "conditional",
    "cre_driver",
    "genetic_background",
    "availability",
    "known_disease_ids",
    "gold_source",
    "gold_status",
]


def parse_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current_id = ""
    current_name = ""
    current_parents: set[str] = set()

    def flush() -> None:
        if current_id.startswith("HP:"):
            names[current_id] = current_name
            parents[current_id].update(current_parents)

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                flush()
                current_id = ""
                current_name = ""
                current_parents = set()
            elif line.startswith("id: HP:"):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("is_a: HP:"):
                current_parents.add(line[6:].split()[0])
    flush()
    return names, dict(parents)


def top_branch(
    hpo_id: str,
    *,
    root_id: str,
    parents: dict[str, set[str]],
    direct_children: set[str],
) -> str:
    queue: deque[str] = deque([hpo_id])
    seen = {hpo_id}
    while queue:
        current = queue.popleft()
        if current in direct_children:
            return current
        for parent in sorted(parents.get(current, set())):
            if parent not in seen and parent != root_id:
                seen.add(parent)
                queue.append(parent)
    return "HP:0000118"


def inheritance_class(value: str) -> str:
    text = value.lower()
    if "x-linked" in text:
        return "x_linked"
    if "autosomal recessive" in text:
        return "autosomal_recessive"
    if "autosomal dominant" in text:
        return "autosomal_dominant"
    return "other"


def gene_tokens(row: dict[str, str]) -> set[str]:
    return {
        token.upper()
        for field in ("human_gene_symbol", "mouse_gene_symbol")
        for token in split_values(row.get(field, ""))
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.size <= 50:
        raise ValueError("--size must be between 30 and 50")
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names, parents = parse_obo(args.hpo_obo)
    phenotype_root = "HP:0000118"
    direct_children = {
        term_id for term_id, term_parents in parents.items() if phenotype_root in term_parents
    }

    disease_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(args.diseases):
        disease_rows[row["disease_id"]].append(row)
    models = read_tsv(args.models)
    models_by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for model in models:
        for gene in gene_tokens(model):
            models_by_gene[gene].append(model)
    phenotype_model_ids = {
        row["model_id"] for row in read_tsv(args.model_phenotypes) if row.get("model_id")
    }
    surface_hpo_ids = {
        row.get("hp_id") or row.get("hpo_id", "")
        for row in read_tsv(args.mp_hpo_mapping)
    }

    eligible: list[dict[str, Any]] = []
    for disease_id, rows in disease_rows.items():
        first = rows[0]
        gene = first["gene_symbol"].upper()
        candidate_models = models_by_gene.get(gene, [])
        candidate_with_phenotype = [
            model for model in candidate_models if model["model_id"] in phenotype_model_ids
        ]
        gold_models = [
            model
            for model in candidate_models
            if disease_id in split_values(model.get("known_disease_ids", ""))
        ]
        core_rows = [row for row in rows if truthy(row.get("is_core", ""))]
        total_weight = sum(float(row["weight"]) for row in rows) or 1.0
        surface_rows = [row for row in rows if row["hpo_id"] in surface_hpo_ids]
        surface_coverage = sum(float(row["weight"]) for row in surface_rows) / total_weight
        branch_votes: Counter[str] = Counter()
        for row in core_rows or rows:
            branch_votes[
                top_branch(
                    row["hpo_id"],
                    root_id=phenotype_root,
                    parents=parents,
                    direct_children=direct_children,
                )
            ] += float(row["weight"])
        branch_id = branch_votes.most_common(1)[0][0] if branch_votes else phenotype_root
        failures = []
        if len(core_rows) < args.min_core_hpo:
            failures.append("insufficient_source_core_hpo")
        if len(candidate_models) < args.min_models:
            failures.append("insufficient_candidate_models")
        if len(candidate_with_phenotype) < args.min_models:
            failures.append("insufficient_models_with_mp")
        if len(gold_models) < args.min_gold_models:
            failures.append("missing_mgi_gold_model")
        if surface_coverage < args.min_surface_coverage:
            failures.append("insufficient_surface_coverage")
        if failures:
            continue
        score = (
            4.0 * min(1.0, surface_coverage)
            + min(2.0, len(gold_models))
            + min(2.0, len(candidate_with_phenotype) / 5.0)
            + min(1.0, len(core_rows) / 8.0)
        )
        eligible.append(
            {
                "disease_id": disease_id,
                "disease_name": first["disease_name"],
                "gene_symbol": gene,
                "inheritance": first["inheritance"],
                "inheritance_class": inheritance_class(first["inheritance"]),
                "phenotype_system_id": branch_id,
                "phenotype_system": names.get(branch_id, branch_id),
                "hpo_count": len(rows),
                "source_core_hpo_count": len(core_rows),
                "source_core_hpo_ids": sorted(row["hpo_id"] for row in core_rows),
                "surface_hpo_count": len(surface_rows),
                "surface_weighted_coverage": surface_coverage,
                "candidate_model_count": len(candidate_models),
                "candidate_model_with_phenotype_count": len(candidate_with_phenotype),
                "gold_model_count": len(gold_models),
                "selection_score": score,
                "candidate_models": candidate_models,
                "gold_models": gold_models,
            }
        )

    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        eligible,
        key=lambda item: (-item["selection_score"], item["disease_id"]),
    ):
        groups[(row["phenotype_system_id"], row["inheritance_class"])].append(row)
    selected: list[dict[str, Any]] = []
    gene_counts: Counter[str] = Counter()
    while len(selected) < args.size and any(groups.values()):
        group_order = sorted(
            groups,
            key=lambda key: (
                sum(
                    item["phenotype_system_id"] == key[0]
                    for item in selected
                ),
                sum(
                    item["phenotype_system_id"] == key[0]
                    and item["inheritance_class"] == key[1]
                    for item in selected
                ),
                key,
            ),
        )
        progressed = False
        for key in group_order:
            queue = groups[key]
            while queue and gene_counts[queue[0]["gene_symbol"]] >= args.max_per_gene:
                queue.popleft()
            if queue and len(selected) < args.size:
                item = queue.popleft()
                selected.append(item)
                gene_counts[item["gene_symbol"]] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) < args.size:
        raise ValueError(
            f"Only {len(selected)} diseases passed independent gates; requested {args.size}."
        )

    candidate_rows = []
    gold_rows = []
    selected_model_ids: set[str] = set()
    for rank, row in enumerate(selected, 1):
        candidate_rows.append(
            {
                **{field: row.get(field, "") for field in CANDIDATE_FIELDS},
                "selection_rank": rank,
                "selection_status": "selected_for_independent_research",
            }
        )
        for model in row["candidate_models"]:
            selected_model_ids.add(model["model_id"])
        for model in row["gold_models"]:
            gold_rows.append(
                {
                    "selection_rank": rank,
                    "disease_id": row["disease_id"],
                    "disease_name": row["disease_name"],
                    "gene_symbol": row["gene_symbol"],
                    **model,
                    "gold_source": "MGI known_disease_ids",
                    "gold_status": "source_gold_pending_codex_review",
                }
            )

    selected_ids = {row["disease_id"] for row in selected}
    selected_rank_by_id = {
        row["disease_id"]: rank for rank, row in enumerate(selected, 1)
    }
    stratum_counts: Counter[tuple[str, str]] = Counter()
    eligible_rows = []
    for eligible_rank, row in enumerate(
        sorted(
            eligible,
            key=lambda item: (
                item["phenotype_system_id"],
                item["inheritance_class"],
                -item["selection_score"],
                item["disease_id"],
            ),
        ),
        1,
    ):
        stratum = (row["phenotype_system_id"], row["inheritance_class"])
        stratum_counts[stratum] += 1
        eligible_rows.append(
            {
                **{field: row.get(field, "") for field in ELIGIBLE_FIELDS},
                "eligible_rank": eligible_rank,
                "stratum_rank": stratum_counts[stratum],
                "selected_rank": selected_rank_by_id.get(row["disease_id"], ""),
                "selection_status": (
                    "selected_for_independent_research"
                    if row["disease_id"] in selected_ids
                    else "eligible_reserve"
                ),
            }
        )
    write_tsv(output / "eligible_candidates.tsv", eligible_rows, ELIGIBLE_FIELDS)
    profile_rows = [
        row
        for disease_id in sorted(selected_ids)
        for row in disease_rows[disease_id]
    ]
    model_rows = [model for model in models if model["model_id"] in selected_model_ids]
    phenotype_rows = [
        row
        for row in read_tsv(args.model_phenotypes)
        if row["model_id"] in selected_model_ids
    ]
    write_tsv(output / "candidate_diseases.tsv", candidate_rows, CANDIDATE_FIELDS)
    write_tsv(
        output / "human_disease_profiles.source.tsv",
        profile_rows,
        list(profile_rows[0]),
    )
    write_tsv(output / "gold_models.source.tsv", gold_rows, GOLD_FIELDS)
    write_tsv(output / "candidate_models.tsv", model_rows, list(model_rows[0]))
    evaluation_inputs = output / "evaluation_inputs"
    write_tsv(
        evaluation_inputs / "model_phenotypes.tsv",
        phenotype_rows,
        list(phenotype_rows[0]),
    )
    manifest = {
        "schema_version": "1.0",
        "dataset": "independent rare-disease mouse-model benchmark candidates",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_set_selected",
        "frozen": False,
        "requested_size": args.size,
        "selected_disease_count": len(selected),
        "eligible_disease_count": len(eligible),
        "selection_protocol": {
            "middle_layer_independent": True,
            "minimum_core_hpo": args.min_core_hpo,
            "minimum_candidate_models": args.min_models,
            "minimum_gold_models": args.min_gold_models,
            "minimum_surface_coverage": args.min_surface_coverage,
            "maximum_diseases_per_gene": args.max_per_gene,
            "diversity_strata": ["HPO top-level phenotype system", "inheritance class"],
        },
        "prohibited_selection_inputs": [
            "human phenotype to cellular mechanism links",
            "mouse phenotype to cellular mechanism links",
            "cellular mechanism coverage",
            "cellular mechanism scores",
            "post-ranking outcomes",
        ],
        "inputs": {
            "human_disease_profiles": input_record(args.diseases, root),
            "mouse_models": input_record(args.models, root),
            "mouse_model_phenotypes_for_runnability_only": input_record(
                args.model_phenotypes, root
            ),
            "surface_mapping_for_runnability_only": input_record(args.mp_hpo_mapping, root),
            "hpo_ontology": input_record(args.hpo_obo, root),
        },
        "outputs": {
            path.name: input_record(path, root)
            for path in sorted(output.glob("*.tsv"))
        },
        "selected_disease_ids": [row["disease_id"] for row in selected],
        "phenotype_system_counts": dict(
            sorted(Counter(row["phenotype_system"] for row in selected).items())
        ),
        "inheritance_class_counts": dict(
            sorted(Counter(row["inheritance_class"] for row in selected).items())
        ),
    }
    write_json(output / "selection_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--min-core-hpo", type=int, default=3)
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument("--min-gold-models", type=int, default=1)
    parser.add_argument("--min-surface-coverage", type=float, default=0.70)
    parser.add_argument("--max-per-gene", type=int, default=2)
    parser.add_argument(
        "--diseases",
        type=Path,
        default=DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization/data/processed/diseases/diseases.all.tsv'),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization/data/processed/mgi/models.tsv'),
    )
    parser.add_argument(
        "--model-phenotypes",
        type=Path,
        default=DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization/data/processed/mgi/model_phenotypes.tsv'),
    )
    parser.add_argument(
        "--mp-hpo-mapping",
        type=Path,
        default=DEFAULT_ROOT / "UHPO/data/mpo_hpo_mapping/mpo_hpo_mapping.tsv",
    )
    parser.add_argument(
        "--hpo-obo",
        type=Path,
        default=DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization/data/raw/hpo/hp.obo'),
    )
    return parser.parse_args()


def main() -> None:
    manifest = select(parse_args())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_disease_count": manifest["selected_disease_count"],
                "eligible_disease_count": manifest["eligible_disease_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
