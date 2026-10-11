from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .baseline import assign_ordinal_ranks, percentile_scores
from .schemas import DiseaseTerm, GeneSetResources, HpoCpoEdge


ABLATION_ARMS = {
    "A0_surface": {"function_weight": 0.0, "marker_weight": 0.0, "interaction_weight": 0.0},
    "A1_marker": {"function_weight": 0.0, "marker_weight": 1.0, "interaction_weight": 0.0},
    "A2_function": {"function_weight": 1.0, "marker_weight": 0.0, "interaction_weight": 0.0},
}


def compatibility(
    gene: str,
    edge: HpoCpoEdge,
    resources: GeneSetResources,
    weights: Mapping[str, float],
    masked_genes: set[str],
) -> tuple[float, float, float, float]:
    direct_masked = gene.upper() in {item.upper() for item in masked_genes}
    marker = 0.0 if direct_masked else resources.marker_scores.get((edge.cell_id, gene.upper()), 0.0)
    function = 0.0 if direct_masked else resources.function_scores.get((edge.function_id, gene.upper()), 0.0)
    interaction = math.sqrt(max(0.0, marker * function))
    function_weight = float(weights.get("function_weight", 0.0))
    marker_weight = float(weights.get("marker_weight", 0.0))
    interaction_weight = float(weights.get("interaction_weight", 0.0))
    denominator = function_weight + marker_weight + interaction_weight
    score = (
        function_weight * function
        + marker_weight * marker
        + interaction_weight * interaction
    ) / denominator if denominator else 0.0
    return marker, function, interaction, score


def disease_cpo_profiles(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    hpo_cpo: Mapping[str, Sequence[HpoCpoEdge]],
    masked_by_disease: Mapping[str, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    profile_rows: list[dict[str, Any]] = []
    coverage: dict[str, float] = {}
    masks = masked_by_disease or {}
    for disease_id, terms in sorted(diseases.items()):
        total_weight = sum(term.weight for term in terms)
        mapped_weight = 0.0
        cpo_weights: dict[str, float] = defaultdict(float)
        edge_lookup: dict[str, HpoCpoEdge] = {}
        for term in terms:
            usable = []
            for edge in hpo_cpo.get(term.hpo_id, ()):
                confidence = edge.masked_confidence(masks.get(disease_id, set()))
                if confidence > 0:
                    usable.append((edge, confidence))
                    cpo_weights[edge.cpo_id] += term.weight * confidence
                    edge_lookup[edge.cpo_id] = edge
            if usable:
                mapped_weight += term.weight
        coverage[disease_id] = mapped_weight / total_weight if total_weight else 0.0
        for cpo_id, weight in sorted(cpo_weights.items(), key=lambda item: (-item[1], item[0])):
            edge = edge_lookup[cpo_id]
            profile_rows.append(
                {
                    "disease_id": disease_id,
                    "cpo_id": cpo_id,
                    "cell_id": edge.cell_id,
                    "cell_name": edge.cell_name,
                    "function_id": edge.function_id,
                    "function_name": edge.function_name,
                    "disease_cpo_weight": weight,
                    "disease_cpo_coverage": coverage[disease_id],
                }
            )
    return profile_rows, coverage


def _masking_audit(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    hpo_cpo: Mapping[str, Sequence[HpoCpoEdge]],
    resources: GeneSetResources,
    masked_by_disease: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for disease_id, terms in sorted(diseases.items()):
        for test_gene in sorted(masked_by_disease.get(disease_id, set())):
            for term in terms:
                for edge in hpo_cpo.get(term.hpo_id, ()):
                    source_hit = test_gene.upper() in {gene.upper() for gene in edge.source_genes}
                    marker_hit = (edge.cell_id, test_gene.upper()) in resources.marker_scores
                    function_hit = (edge.function_id, test_gene.upper()) in resources.function_scores
                    if not (source_hit or marker_hit or function_hit):
                        continue
                    remaining_source = sum(
                        gene.upper() != test_gene.upper() for gene in edge.source_genes
                    )
                    rows.append(
                        {
                            "disease_id": disease_id,
                            "test_gene": test_gene,
                            "hpo_id": term.hpo_id,
                            "cpo_id": edge.cpo_id,
                            "gene_used_in_hpo_cpo": source_hit,
                            "gene_in_marker_set": marker_hit,
                            "gene_in_function_set": function_hit,
                            "masked": True,
                            "remaining_evidence_count": remaining_source,
                            "warning": "direct_evidence_removed",
                        }
                    )
    return rows


def rerank(
    base_rows: Sequence[Mapping[str, Any]],
    hpo_mpo_evidence: Sequence[Mapping[str, Any]],
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    hpo_cpo: Mapping[str, Sequence[HpoCpoEdge]],
    resources: GeneSetResources,
    config: Mapping[str, Any],
    experiment: str = "A3_marker_function",
    component_weights: Mapping[str, float] | None = None,
    masked_by_disease: Mapping[str, set[str]] | None = None,
    emit_details: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    masks = masked_by_disease or {}
    weights = dict(component_weights or config.get("gene_set_scoring", {}))
    if experiment == "A0_surface":
        weights = ABLATION_ARMS[experiment]
    disease_profiles, disease_coverage = disease_cpo_profiles(diseases, hpo_cpo, masks)
    evidence_by_gene: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in hpo_mpo_evidence:
        evidence_by_gene[(str(row["disease_id"]), str(row["gene_symbol"]).upper())].append(row)

    preliminary: list[dict[str, Any]] = []
    all_paths: list[dict[str, Any]] = []
    all_candidate_profiles: list[dict[str, Any]] = []
    for base in base_rows:
        disease_id = str(base["disease_id"])
        gene = str(base["gene_symbol"]).upper()
        masked_genes = masks.get(disease_id, set())
        total_hpo_weight = sum(term.weight for term in diseases[disease_id])
        comparable_hpos: set[str] = set()
        profile: dict[str, dict[str, Any]] = {}
        paths: list[dict[str, Any]] = []
        denominator = 0.0
        marker_numerator = 0.0
        function_numerator = 0.0
        interaction_numerator = 0.0
        cellpath_numerator = 0.0
        for evidence in evidence_by_gene.get((disease_id, gene), ()):
            hpo_score = float(evidence.get("hpo_mpo_score", 0.0))
            if hpo_score <= 0:
                continue
            hpo_id = str(evidence["hpo_id"])
            usable_edge = False
            for edge in hpo_cpo.get(hpo_id, ()):
                confidence = edge.masked_confidence(masked_genes)
                if confidence <= 0:
                    continue
                usable_edge = True
                path_weight = float(evidence.get("hpo_weight", 1.0)) * hpo_score * confidence
                marker, function, interaction, agreement = compatibility(
                    gene, edge, resources, weights, masked_genes
                )
                contribution = path_weight * agreement
                denominator += path_weight
                marker_numerator += path_weight * marker
                function_numerator += path_weight * function
                interaction_numerator += path_weight * interaction
                cellpath_numerator += contribution
                current = profile.setdefault(
                    edge.cpo_id,
                    {
                        "disease_id": disease_id,
                        "gene_symbol": gene,
                        "cpo_id": edge.cpo_id,
                        "cell_id": edge.cell_id,
                        "cell_name": resources.cell_names.get(edge.cell_id, edge.cell_name),
                        "function_id": edge.function_id,
                        "function_name": resources.function_names.get(edge.function_id, edge.function_name),
                        "candidate_cpo_weight": 0.0,
                    },
                )
                current["candidate_cpo_weight"] += path_weight
                paths.append(
                    {
                        "disease_id": disease_id,
                        "gene_symbol": gene,
                        "model_id": evidence.get("model_id", base.get("best_model_id", "")),
                        "hpo_id": hpo_id,
                        "hpo_weight": evidence.get("hpo_weight", 1.0),
                        "matched_mpo_id": evidence.get("matched_mpo_id", ""),
                        "hpo_mpo_score": hpo_score,
                        "cpo_id": edge.cpo_id,
                        "hpo_cpo_confidence": confidence,
                        "cell_id": edge.cell_id,
                        "cell_name": resources.cell_names.get(edge.cell_id, edge.cell_name),
                        "function_id": edge.function_id,
                        "function_name": resources.function_names.get(edge.function_id, edge.function_name),
                        "marker_score": marker,
                        "function_score": function,
                        "interaction_score": interaction,
                        "path_weight": path_weight,
                        "path_contribution": contribution,
                        "source_record_ids": edge.source_record_ids,
                        "source_gene_ids": edge.source_genes,
                    }
                )
            if usable_edge:
                comparable_hpos.add(hpo_id)
        cellpath_score = cellpath_numerator / denominator if denominator else 0.0
        marker_score = marker_numerator / denominator if denominator else 0.0
        function_score = function_numerator / denominator if denominator else 0.0
        interaction_score = interaction_numerator / denominator if denominator else 0.0
        candidate_coverage = (
            sum(term.weight for term in diseases[disease_id] if term.hpo_id in comparable_hpos)
            / total_hpo_weight
            if total_hpo_weight
            else 0.0
        )
        ordered_paths = sorted(
            paths,
            key=lambda row: (-float(row["path_contribution"]), -float(row["path_weight"]), str(row["cpo_id"])),
        )
        top = ordered_paths[0] if ordered_paths else {}
        row = deepcopy(dict(base))
        row.update(
            {
                "cpo_coverage": disease_coverage.get(disease_id, 0.0),
                "candidate_cpo_coverage": candidate_coverage,
                "marker_score": marker_score,
                "function_score": function_score,
                "interaction_score": interaction_score,
                "cellpath_score": cellpath_score,
                "predicted_cell_id": top.get("cell_id", ""),
                "predicted_cell_name": top.get("cell_name", ""),
                "predicted_function_id": top.get("function_id", ""),
                "predicted_function_name": top.get("function_name", ""),
                "top_cpo_id": top.get("cpo_id", ""),
                "direct_gene_evidence_score": max(marker_score, function_score),
                "indirect_functional_support_score": 0.0,
                "evaluation_mode": "strict" if masks else "practical",
                "experiment": experiment,
            }
        )
        preliminary.append(row)
        if emit_details:
            limit = int(config.get("report", {}).get("top_paths_per_gene", 10))
            all_paths.extend(ordered_paths[:limit])
            all_candidate_profiles.extend(
                sorted(profile.values(), key=lambda item: (-float(item["candidate_cpo_weight"]), str(item["cpo_id"])))
            )

    final_rows: list[dict[str, Any]] = []
    by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preliminary:
        by_disease[str(row["disease_id"])].append(row)
    lambda_value = 0.0 if experiment == "A0_surface" else float(config.get("rank_fusion", {}).get("lambda", 0.2))
    coverage_mode = str(config.get("rank_fusion", {}).get("coverage_mode", "disease"))
    for disease_id, rows in sorted(by_disease.items()):
        positive = [row for row in rows if float(row["cellpath_score"]) > 0]
        cellpath_percentiles = percentile_scores(positive, "cellpath_score", "gene_symbol")
        for row in rows:
            row["cellpath_percentile"] = cellpath_percentiles.get(str(row["gene_symbol"]), 0.0)
            coverage = (
                float(row["candidate_cpo_coverage"])
                if coverage_mode == "candidate"
                else float(row["cpo_coverage"])
            )
            row["final_score"] = float(row["base_percentile"]) + lambda_value * coverage * float(row["cellpath_percentile"])
        ordered = assign_ordinal_ranks(rows, "final_score", "gene_symbol", "final_rank")
        for row in ordered:
            row["rank_change"] = int(row["base_rank"]) - int(row["final_rank"])
        final_rows.extend(ordered)
    return {
        "final_rankings": final_rows,
        "evidence_paths": all_paths,
        "candidate_cpo_profiles": all_candidate_profiles,
        "disease_cpo_profiles": disease_profiles if emit_details else [],
        "leakage_audit": _masking_audit(diseases, hpo_cpo, resources, masks) if masks and emit_details else [],
    }


FINAL_RANKING_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_id",
    "gene_symbol",
    "best_model_id",
    "base_score",
    "base_rank",
    "base_percentile",
    "bma_score",
    "max_pair_score",
    "hpo_mpo_coverage",
    "cpo_coverage",
    "candidate_cpo_coverage",
    "marker_score",
    "function_score",
    "interaction_score",
    "cellpath_score",
    "cellpath_percentile",
    "final_score",
    "final_rank",
    "rank_change",
    "predicted_cell_id",
    "predicted_cell_name",
    "predicted_function_id",
    "predicted_function_name",
    "top_hpo_id",
    "top_mpo_id",
    "top_cpo_id",
    "direct_gene_evidence_score",
    "indirect_functional_support_score",
    "is_known_gene",
    "evaluation_mode",
    "experiment",
]

EVIDENCE_PATH_FIELDS = [
    "disease_id",
    "gene_symbol",
    "model_id",
    "hpo_id",
    "hpo_weight",
    "matched_mpo_id",
    "hpo_mpo_score",
    "cpo_id",
    "hpo_cpo_confidence",
    "cell_id",
    "cell_name",
    "function_id",
    "function_name",
    "marker_score",
    "function_score",
    "interaction_score",
    "path_weight",
    "path_contribution",
    "source_record_ids",
    "source_gene_ids",
]

CANDIDATE_CPO_FIELDS = [
    "disease_id",
    "gene_symbol",
    "cpo_id",
    "cell_id",
    "cell_name",
    "function_id",
    "function_name",
    "candidate_cpo_weight",
]

DISEASE_CPO_FIELDS = [
    "disease_id",
    "cpo_id",
    "cell_id",
    "cell_name",
    "function_id",
    "function_name",
    "disease_cpo_weight",
    "disease_cpo_coverage",
]

LEAKAGE_AUDIT_FIELDS = [
    "disease_id",
    "test_gene",
    "hpo_id",
    "cpo_id",
    "gene_used_in_hpo_cpo",
    "gene_in_marker_set",
    "gene_in_function_set",
    "masked",
    "remaining_evidence_count",
    "warning",
]
