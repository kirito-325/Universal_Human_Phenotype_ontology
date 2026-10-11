from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import hypergeom

from .binary_cpo import BinaryCpoGraph
from .schemas import DiseaseTerm


def bh_adjust(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return []
    order = np.argsort(array, kind="stable")
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def global_gene_hpo_reachability(
    graph: BinaryCpoGraph,
    candidate_genes: set[str],
    background_hpo: set[str],
) -> dict[str, set[str]]:
    output = {gene: set() for gene in candidate_genes}
    for edge in graph.edges:
        if edge.hpo_id not in background_hpo:
            continue
        for gene in graph.function_genes.get(edge.function_id, set()):
            if gene in output:
                output[gene].add(edge.hpo_id)
    return output


def score_graph(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    candidate_genes: set[str],
    graph: BinaryCpoGraph,
    background_hpo: set[str],
    base_by_disease_gene: Mapping[tuple[str, str], Mapping[str, Any]],
    emit_paths: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    marker_by_cell: dict[str, set[str]] = defaultdict(set)
    for cell_id, gene in graph.marker_memberships:
        if gene in candidate_genes:
            marker_by_cell[cell_id].add(gene)
    mapped_universe = background_hpo.intersection(graph.edges_by_hpo)
    universe_size = len(mapped_universe)
    reachability = global_gene_hpo_reachability(
        graph, candidate_genes, mapped_universe
    )
    rows: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        disease_hpo = {term.hpo_id for term in terms}
        test_hpo = disease_hpo.intersection(mapped_universe)
        stats = {
            gene: {
                "hpo": set(),
                "cpo": set(),
                "cell": set(),
                "function": set(),
                "raw_function": 0,
                "raw_marker": 0,
                "concordant": set(),
                "path_labels": [],
            }
            for gene in candidate_genes
        }
        for hpo_id in sorted(test_hpo):
            for edge in graph.edges_by_hpo.get(hpo_id, ()):
                function_members = graph.function_genes.get(edge.function_id, set())
                marker_members = marker_by_cell.get(edge.cell_id, set())
                path_genes = function_members.union(marker_members).intersection(
                    candidate_genes
                )
                for gene in function_members.intersection(candidate_genes):
                    item = stats[gene]
                    item["hpo"].add(hpo_id)
                    item["cpo"].add(edge.cpo_id)
                    item["cell"].add(edge.cell_id)
                    item["function"].add(edge.function_id)
                    item["raw_function"] += 1
                    item["path_labels"].append(
                        f"{hpo_id}>{edge.cpo_id}>{gene}"
                    )
                    if gene in marker_members:
                        item["concordant"].add(edge.cpo_id)
                for gene in marker_members.intersection(candidate_genes):
                    stats[gene]["raw_marker"] += 1
                if emit_paths:
                    for gene in sorted(path_genes):
                        paths.append(
                            {
                                "disease_id": disease_id,
                                "gene_symbol": gene,
                                "hpo_id": hpo_id,
                                "hpo_name": edge.hpo_name,
                                "cpo_id": edge.cpo_id,
                                "cl_id": edge.cell_id,
                                "cell_name": edge.cell_name,
                                "go_id": edge.function_id,
                                "function_name": edge.function_name,
                                "path_type": (
                                    "function_and_marker"
                                    if gene in function_members and gene in marker_members
                                    else "function"
                                    if gene in function_members
                                    else "marker_only"
                                ),
                                "function_membership": gene in function_members,
                                "marker_membership": gene in marker_members,
                                "hpo_cpo_relation": edge.relation_type,
                                "hpo_cpo_scope": edge.scope,
                                "hpo_cpo_confidence_audit_only": edge.raw_confidence,
                                "source_genes": edge.source_genes,
                                "source_records": edge.source_records,
                                "graph_version": graph.version,
                            }
                        )
        disease_rows = []
        for gene in sorted(candidate_genes):
            item = stats[gene]
            overlap = len(item["hpo"])
            global_degree = len(reachability.get(gene, set()))
            if universe_size and test_hpo and global_degree and overlap:
                p_value = float(
                    hypergeom.sf(
                        overlap - 1,
                        universe_size,
                        global_degree,
                        len(test_hpo),
                    )
                )
            else:
                p_value = 1.0
            base = base_by_disease_gene[(disease_id, gene)]
            disease_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": base["disease_name"],
                    "gene_symbol": gene,
                    "base_rank": base["base_rank"],
                    "base_dense_rank": base["base_dense_rank"],
                    "base_score": base["base_score"],
                    "best_model_id": base["best_model_id"],
                    "best_hpo_id": base["best_hpo_id"],
                    "best_mpo_id": base["best_mpo_id"],
                    "cpo_graph": graph.name,
                    "cpo_graph_version": graph.version,
                    "background_hpo_count": universe_size,
                    "disease_hpo_count": len(disease_hpo),
                    "cpo_mappable_hpo_count": len(test_hpo),
                    "cpo_coverage": len(test_hpo) / len(disease_hpo) if disease_hpo else 0.0,
                    "supported_hpo_count": overlap,
                    "supported_hpo_fraction": overlap / len(test_hpo) if test_hpo else 0.0,
                    "global_reachable_hpo_count": global_degree,
                    "supported_cpo_count": len(item["cpo"]),
                    "supported_cell_count": len(item["cell"]),
                    "supported_function_count": len(item["function"]),
                    "raw_function_path_count": item["raw_function"],
                    "raw_marker_path_count": item["raw_marker"],
                    "concordant_cpo_count": len(item["concordant"]),
                    "cpo_enrichment_log_p": -math.log10(max(p_value, np.finfo(float).tiny)),
                    "cpo_enrichment_pvalue": p_value,
                    "has_cpo_evidence": overlap > 0,
                    "predicted_cells": sorted(item["cell"]),
                    "predicted_functions": sorted(item["function"]),
                    "top_cpo_paths": sorted(item["path_labels"])[:10],
                }
            )
        adjusted = bh_adjust(
            [float(row["cpo_enrichment_pvalue"]) for row in disease_rows]
        )
        for row, fdr in zip(disease_rows, adjusted):
            row["cpo_enrichment_fdr"] = fdr
        rows.extend(disease_rows)
    return rows, paths


def _priority(row: Mapping[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "raw_path":
        return (
            -int(row["raw_function_path_count"]),
            -int(row["supported_hpo_count"]),
            -int(row["supported_cpo_count"]),
            int(row["base_rank"]),
            str(row["gene_symbol"]),
        )
    return (
        float(row["cpo_enrichment_pvalue"]),
        -int(row["supported_hpo_count"]),
        -int(row["supported_cpo_count"]),
        -int(row["concordant_cpo_count"]),
        -int(row["raw_function_path_count"]),
        int(row["base_rank"]),
        str(row["gene_symbol"]),
    )


def rerank_unconstrained(
    enrichment_rows: Sequence[Mapping[str, Any]],
    mode: str,
    experiment: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in enrichment_rows:
        grouped[str(row["disease_id"])].append(row)
    output = []
    for disease_id, rows in sorted(grouped.items()):
        if mode == "base":
            ordered = sorted(rows, key=lambda row: (int(row["base_rank"]), str(row["gene_symbol"])))
        else:
            ordered = sorted(rows, key=lambda row: _priority(row, mode))
        total = len(ordered)
        for rank, source in enumerate(ordered, 1):
            row = dict(source)
            row["final_rank"] = rank
            row["final_rank_score"] = 1.0 if total == 1 else 1.0 - (rank - 1) / (total - 1)
            row["rank_change"] = int(row["base_rank"]) - rank
            row["reranking_mode"] = experiment
            output.append(row)
    return output


def rerank_constrained(
    enrichment_rows: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
    experiment: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in enrichment_rows:
        grouped[str(row["disease_id"])][str(row["gene_symbol"])] = row
    constraints_by_disease: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in constraints:
        constraints_by_disease[str(row["disease_id"])].append(row)
    output = []
    summaries = []
    for disease_id, rows_by_gene in sorted(grouped.items()):
        adjacency = {gene: set() for gene in rows_by_gene}
        indegree = {gene: 0 for gene in rows_by_gene}
        for edge in constraints_by_disease.get(disease_id, ()):
            higher = str(edge["higher_gene"])
            lower = str(edge["lower_gene"])
            if higher not in adjacency or lower not in adjacency or lower in adjacency[higher]:
                continue
            adjacency[higher].add(lower)
            indegree[lower] += 1
        queue: list[tuple[tuple[Any, ...], str]] = []
        for gene, degree in indegree.items():
            if degree == 0:
                heapq.heappush(queue, (_priority(rows_by_gene[gene], "enrichment"), gene))
        ordered = []
        while queue:
            _, gene = heapq.heappop(queue)
            ordered.append(gene)
            for child in sorted(adjacency[gene]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(
                        queue, (_priority(rows_by_gene[child], "enrichment"), child)
                    )
        if len(ordered) != len(rows_by_gene):
            raise RuntimeError(f"Stable-order graph contains a cycle for {disease_id}")
        rank_by_gene = {gene: rank for rank, gene in enumerate(ordered, 1)}
        violations = sum(
            rank_by_gene[str(edge["higher_gene"])]
            >= rank_by_gene[str(edge["lower_gene"])]
            for edge in constraints_by_disease.get(disease_id, ())
        )
        if violations:
            raise RuntimeError(
                f"Constrained reranking violated {violations} phenotype edges for {disease_id}"
            )
        total = len(ordered)
        for gene in ordered:
            source = rows_by_gene[gene]
            rank = rank_by_gene[gene]
            row = dict(source)
            row["final_rank"] = rank
            row["final_rank_score"] = 1.0 if total == 1 else 1.0 - (rank - 1) / (total - 1)
            row["rank_change"] = int(row["base_rank"]) - rank
            row["reranking_mode"] = experiment
            output.append(row)
        summaries.append(
            {
                "disease_id": disease_id,
                "candidate_gene_count": len(ordered),
                "stable_constraint_count": sum(len(children) for children in adjacency.values()),
                "constraint_violations": violations,
                "reranking_mode": experiment,
            }
        )
    return output, summaries


ENRICHMENT_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "base_rank",
    "base_dense_rank",
    "base_score",
    "best_model_id",
    "best_hpo_id",
    "best_mpo_id",
    "cpo_graph",
    "cpo_graph_version",
    "background_hpo_count",
    "disease_hpo_count",
    "cpo_mappable_hpo_count",
    "cpo_coverage",
    "supported_hpo_count",
    "supported_hpo_fraction",
    "global_reachable_hpo_count",
    "supported_cpo_count",
    "supported_cell_count",
    "supported_function_count",
    "raw_function_path_count",
    "raw_marker_path_count",
    "concordant_cpo_count",
    "cpo_enrichment_log_p",
    "cpo_enrichment_pvalue",
    "cpo_enrichment_fdr",
    "has_cpo_evidence",
    "predicted_cells",
    "predicted_functions",
    "top_cpo_paths",
]

FINAL_PARAMETER_FREE_FIELDS = ENRICHMENT_FIELDS + [
    "final_rank",
    "final_rank_score",
    "rank_change",
    "reranking_mode",
]

RAW_PATH_FIELDS = [
    "disease_id",
    "gene_symbol",
    "hpo_id",
    "hpo_name",
    "cpo_id",
    "cl_id",
    "cell_name",
    "go_id",
    "function_name",
    "path_type",
    "function_membership",
    "marker_membership",
    "hpo_cpo_relation",
    "hpo_cpo_scope",
    "hpo_cpo_confidence_audit_only",
    "source_genes",
    "source_records",
    "graph_version",
]
