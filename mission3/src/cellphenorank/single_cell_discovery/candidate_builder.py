from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..config import dump_config
from ..evaluation import load_gold_genes
from ..io import as_bool, as_float, input_record, read_tsv, value, write_json, write_tsv


CANDIDATE_FIELDS = [
    "candidate_case_id",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "base_rank",
    "final_rank",
    "rank_change",
    "candidate_count",
    "base_rank_percentile",
    "final_rank_percentile",
    "supported_hpo_count",
    "supported_hpo_fraction",
    "supported_cpo_count",
    "raw_function_path_count",
    "cell_type_specific_path_count",
    "cell_general_path_count",
    "cell_general_fraction",
    "cpo_enrichment_pvalue",
    "cpo_enrichment_fdr",
    "bridge_loo_rank",
    "bridge_rank_change",
    "bridge_dependency_status",
    "predicted_cell_count",
    "predicted_function_count",
    "primary_predicted_cells",
    "primary_predicted_cell_names",
    "primary_predicted_function",
    "primary_predicted_function_name",
    "secondary_predicted_functions",
    "secondary_predicted_function_names",
    "supported_hpo_ids",
    "cpo_ids",
    "bridge_source_genes",
    "pareto_front",
    "priority_rank",
]

EVIDENCE_FIELDS = [
    "candidate_case_id",
    "disease_id",
    "gene_symbol",
    "hpo_id",
    "hpo_name",
    "cpo_id",
    "cl_id",
    "cl_name",
    "go_id",
    "go_name",
    "edge_scope",
    "relation_type",
    "source_genes",
    "source_records",
    "candidate_is_bridge_source",
    "candidate_is_function_member",
    "candidate_is_marker",
]

EXCLUSION_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "base_rank",
    "final_rank",
    "rank_change",
    "is_gold_gene",
    "complete_function_path_count",
    "status",
    "reasons",
]


def _split(item: object) -> set[str]:
    text = str(item or "").replace("|", ";").replace(",", ";")
    return {part.strip().upper() for part in text.split(";") if part.strip()}


def _percentile(rank: int, count: int) -> float:
    return 1.0 if count <= 1 else 1.0 - (rank - 1) / (count - 1)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = (
        -int(left["final_rank"]),
        int(left["rank_change"]),
        int(left["supported_hpo_count"]),
        int(left["cell_type_specific_path_count"]),
        -float(left["cell_general_fraction"]),
    )
    right_values = (
        -int(right["final_rank"]),
        int(right["rank_change"]),
        int(right["supported_hpo_count"]),
        int(right["cell_type_specific_path_count"]),
        -float(right["cell_general_fraction"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def assign_pareto_fronts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(rows)
    front = 1
    while remaining:
        current = [
            row
            for row in remaining
            if not any(
                other is not row and _dominates(other, row) for other in remaining
            )
        ]
        for row in current:
            row["pareto_front"] = front
        current_ids = {id(row) for row in current}
        remaining = [row for row in remaining if id(row) not in current_ids]
        front += 1
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["pareto_front"]),
            int(row["final_rank"]),
            -int(row["rank_change"]),
            -int(row["supported_hpo_count"]),
            -int(row["cell_type_specific_path_count"]),
            float(row["cell_general_fraction"]),
            str(row["disease_id"]),
            str(row["gene_symbol"]),
        ),
    )
    for priority, row in enumerate(ordered, 1):
        row["priority_rank"] = priority
    return ordered


def _function_genes(path: str | Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(path):
        function_id = value(row, "function_id", "go_id", "cell_phenotype_id")
        gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
        if function_id and gene:
            result[function_id].add(gene)
    return dict(result)


def _path_is_function(row: Mapping[str, str]) -> bool:
    return as_bool(row.get("function_membership", "")) or value(
        row, "path_type"
    ) in {"function", "function_and_marker"}


def _select_cells(paths: Sequence[Mapping[str, str]]) -> tuple[list[str], list[str]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"hpo": set(), "specific_hpo": set(), "name": ""}
    )
    for row in paths:
        cell = value(row, "cl_id", "cell_id")
        if not cell:
            continue
        stats[cell]["hpo"].add(value(row, "hpo_id"))
        stats[cell]["name"] = value(row, "cell_name", "cl_name", default=cell)
        if value(row, "hpo_cpo_scope", "edge_scope") == "cell_type_specific":
            stats[cell]["specific_hpo"].add(value(row, "hpo_id"))
    ordered = sorted(
        stats,
        key=lambda cell: (
            -len(stats[cell]["hpo"]),
            -len(stats[cell]["specific_hpo"]),
            cell,
        ),
    )
    if not ordered:
        return [], []
    best = ordered[0]
    best_key = (len(stats[best]["hpo"]), len(stats[best]["specific_hpo"]))
    selected = [
        cell
        for cell in ordered
        if (len(stats[cell]["hpo"]), len(stats[cell]["specific_hpo"])) == best_key
    ]
    return selected, [str(stats[cell]["name"]) for cell in selected]


def _select_functions(
    paths: Sequence[Mapping[str, str]],
    genes_by_function: Mapping[str, set[str]],
    candidate_gene: str,
    bridge_genes: set[str],
) -> tuple[str, str, list[str], list[str]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"hpo": set(), "specific_hpo": set(), "name": ""}
    )
    for row in paths:
        function = value(row, "go_id", "function_id")
        if not function:
            continue
        stats[function]["hpo"].add(value(row, "hpo_id"))
        stats[function]["name"] = value(
            row, "function_name", "go_name", default=function
        )
        if value(row, "hpo_cpo_scope", "edge_scope") == "cell_type_specific":
            stats[function]["specific_hpo"].add(value(row, "hpo_id"))
    ordered = sorted(
        stats,
        key=lambda function: (
            -len(stats[function]["hpo"]),
            -len(stats[function]["specific_hpo"]),
            -len(
                genes_by_function.get(function, set())
                - {candidate_gene.upper()}
                - bridge_genes
            ),
            function,
        ),
    )
    if not ordered:
        return "", "", [], []
    primary = ordered[0]
    secondary = ordered[1:3]
    return (
        primary,
        str(stats[primary]["name"]),
        secondary,
        [str(stats[item]["name"]) for item in secondary],
    )


def build_case_candidates(
    a4_rows: Sequence[Mapping[str, str]],
    a5_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
    gold_genes: Mapping[str, set[str]],
    genes_by_function: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    a5 = {
        (value(row, "disease_id"), value(row, "gene_symbol").upper()): row
        for row in a5_rows
    }
    paths_by_key: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        if _path_is_function(row):
            paths_by_key[
                (value(row, "disease_id"), value(row, "gene_symbol").upper())
            ].append(row)
    candidate_counts = defaultdict(int)
    for row in a4_rows:
        candidate_counts[value(row, "disease_id")] += 1
    candidates: list[dict[str, Any]] = []
    evidence_output: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for ranking in a4_rows:
        disease_id = value(ranking, "disease_id")
        gene = value(ranking, "gene_symbol").upper()
        base_rank = int(float(value(ranking, "base_rank", default="0")))
        final_rank = int(float(value(ranking, "final_rank", default="0")))
        rank_change = base_rank - final_rank
        is_gold = gene in gold_genes.get(disease_id, set())
        paths = paths_by_key.get((disease_id, gene), [])
        reasons = []
        if not is_gold:
            reasons.append("not_frozen_gold_gene")
        if rank_change <= 0:
            reasons.append("a4_did_not_improve_rank")
        if not paths:
            reasons.append("no_complete_function_path")
        if reasons:
            if is_gold:
                exclusions.append(
                    {
                        "disease_id": disease_id,
                        "disease_name": value(ranking, "disease_name"),
                        "gene_symbol": gene,
                        "base_rank": base_rank,
                        "final_rank": final_rank,
                        "rank_change": rank_change,
                        "is_gold_gene": is_gold,
                        "complete_function_path_count": len(paths),
                        "status": "excluded",
                        "reasons": reasons,
                    }
                )
            continue
        case_id = f"{disease_id.replace(':', '_')}__{gene}"
        bridge = a5.get((disease_id, gene))
        bridge_rank = (
            int(float(value(bridge, "final_rank", default=str(base_rank))))
            if bridge
            else base_rank
        )
        bridge_gain = base_rank - bridge_rank
        if bridge is None:
            bridge_status = "bridge_not_evaluable"
        elif bridge_gain >= rank_change:
            bridge_status = "bridge_stable"
        elif bridge_gain > 0:
            bridge_status = "bridge_partially_dependent"
        else:
            bridge_status = "bridge_fully_dependent"
        source_genes = set().union(*(_split(row.get("source_genes")) for row in paths))
        cells, cell_names = _select_cells(paths)
        primary_function, primary_name, secondary, secondary_names = _select_functions(
            paths, genes_by_function, gene, source_genes
        )
        specific_count = sum(
            value(row, "hpo_cpo_scope", "edge_scope") == "cell_type_specific"
            for row in paths
        )
        general_count = sum(
            value(row, "hpo_cpo_scope", "edge_scope") == "cell_general"
            for row in paths
        )
        scope_total = specific_count + general_count
        count = candidate_counts[disease_id]
        candidates.append(
            {
                "candidate_case_id": case_id,
                "disease_id": disease_id,
                "disease_name": value(ranking, "disease_name"),
                "gene_symbol": gene,
                "base_rank": base_rank,
                "final_rank": final_rank,
                "rank_change": rank_change,
                "candidate_count": count,
                "base_rank_percentile": _percentile(base_rank, count),
                "final_rank_percentile": _percentile(final_rank, count),
                "supported_hpo_count": int(
                    float(value(ranking, "supported_hpo_count", default="0"))
                ),
                "supported_hpo_fraction": as_float(
                    value(ranking, "supported_hpo_fraction"), 0.0
                ),
                "supported_cpo_count": int(
                    float(value(ranking, "supported_cpo_count", default="0"))
                ),
                "raw_function_path_count": len(paths),
                "cell_type_specific_path_count": specific_count,
                "cell_general_path_count": general_count,
                "cell_general_fraction": general_count / scope_total if scope_total else 0.0,
                "cpo_enrichment_pvalue": as_float(
                    value(ranking, "cpo_enrichment_pvalue"), 1.0
                ),
                "cpo_enrichment_fdr": as_float(
                    value(ranking, "cpo_enrichment_fdr"), 1.0
                ),
                "bridge_loo_rank": bridge_rank,
                "bridge_rank_change": bridge_gain,
                "bridge_dependency_status": bridge_status,
                "predicted_cell_count": len(
                    {value(row, "cl_id", "cell_id") for row in paths}
                ),
                "predicted_function_count": len(
                    {value(row, "go_id", "function_id") for row in paths}
                ),
                "primary_predicted_cells": cells,
                "primary_predicted_cell_names": cell_names,
                "primary_predicted_function": primary_function,
                "primary_predicted_function_name": primary_name,
                "secondary_predicted_functions": secondary,
                "secondary_predicted_function_names": secondary_names,
                "supported_hpo_ids": sorted(
                    {value(row, "hpo_id") for row in paths}
                ),
                "cpo_ids": sorted({value(row, "cpo_id") for row in paths}),
                "bridge_source_genes": sorted(source_genes),
            }
        )
        for row in sorted(
            paths,
            key=lambda item: (
                value(item, "hpo_id"),
                value(item, "cpo_id"),
                value(item, "cl_id", "cell_id"),
                value(item, "go_id", "function_id"),
            ),
        ):
            row_sources = _split(row.get("source_genes"))
            evidence_output.append(
                {
                    "candidate_case_id": case_id,
                    "disease_id": disease_id,
                    "gene_symbol": gene,
                    "hpo_id": value(row, "hpo_id"),
                    "hpo_name": value(row, "hpo_name"),
                    "cpo_id": value(row, "cpo_id"),
                    "cl_id": value(row, "cl_id", "cell_id"),
                    "cl_name": value(row, "cell_name", "cl_name"),
                    "go_id": value(row, "go_id", "function_id"),
                    "go_name": value(row, "function_name", "go_name"),
                    "edge_scope": value(row, "hpo_cpo_scope", "edge_scope"),
                    "relation_type": value(row, "hpo_cpo_relation", "relation_type"),
                    "source_genes": sorted(row_sources),
                    "source_records": value(row, "source_records"),
                    "candidate_is_bridge_source": gene in row_sources,
                    "candidate_is_function_member": as_bool(
                        row.get("function_membership", "")
                    ),
                    "candidate_is_marker": as_bool(row.get("marker_membership", "")),
                }
            )
    return assign_pareto_fronts(candidates), evidence_output, exclusions


def run_candidate_discovery(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    required = [
        "a4_rankings",
        "a5_rankings",
        "evidence_paths_extended",
        "gold_standard",
        "go_functions",
    ]
    input_summary = []
    for name in required:
        path = Path(inputs[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        input_summary.append({"input": name, **input_record(path)})
    a4_rows = list(read_tsv(inputs["a4_rankings"]))
    a5_rows = list(read_tsv(inputs["a5_rankings"]))
    evidence_rows = list(read_tsv(inputs["evidence_paths_extended"]))
    candidates, evidence, exclusions = build_case_candidates(
        a4_rows,
        a5_rows,
        evidence_rows,
        load_gold_genes(inputs["gold_standard"]),
        _function_genes(inputs["go_functions"]),
    )
    write_tsv(output / "case_candidate_table.tsv", candidates, CANDIDATE_FIELDS)
    write_tsv(output / "candidate_priority_order.tsv", candidates, CANDIDATE_FIELDS)
    write_tsv(output / "candidate_evidence_summary.tsv", evidence, EVIDENCE_FIELDS)
    write_tsv(output / "candidate_exclusions.tsv", exclusions, EXCLUSION_FIELDS)
    write_tsv(
        output / "pareto_fronts.tsv",
        [
            {
                "candidate_case_id": row["candidate_case_id"],
                "pareto_front": row["pareto_front"],
                "priority_rank": row["priority_rank"],
                "is_first_front": int(row["pareto_front"]) == 1,
            }
            for row in candidates
        ],
        ["candidate_case_id", "pareto_front", "priority_rank", "is_first_front"],
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_used_expression_values": False,
        "hardcoded_disease_or_gene": False,
        "selection_rule": "A4 rank improvement for frozen gold gene with complete function path",
        "pareto_dimensions": [
            "final_rank_min",
            "rank_change_max",
            "supported_hpo_count_max",
            "cell_type_specific_path_count_max",
            "cell_general_fraction_min",
        ],
        "counts": {
            "a4_rows": len(a4_rows),
            "improved_evidence_auditable_candidates": len(candidates),
            "pareto_front_1": sum(int(row["pareto_front"]) == 1 for row in candidates),
            "evidence_rows": len(evidence),
            "excluded_gold_pairs": len(exclusions),
        },
        "inputs": {row["input"]: row for row in input_summary},
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    return manifest
