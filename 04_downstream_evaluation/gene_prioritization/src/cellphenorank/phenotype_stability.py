from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .schemas import DiseaseTerm, ModelPhenotype, MouseModel
from .similarity import SimilarityProvider


def _top_two(values: Sequence[float]) -> tuple[float, int, float]:
    top_value = -1.0
    top_index = -1
    second_value = 0.0
    for index, item in enumerate(values):
        value = float(item)
        if value > top_value:
            second_value = max(0.0, top_value)
            top_value = value
            top_index = index
        elif value > second_value:
            second_value = value
    return max(0.0, top_value), top_index, max(0.0, second_value)


def _dense_ranks(scores: Mapping[str, float]) -> dict[str, int]:
    unique = sorted({float(score) for score in scores.values()}, reverse=True)
    by_score = {score: rank for rank, score in enumerate(unique, 1)}
    return {gene: by_score[float(score)] for gene, score in scores.items()}


def _ordinal_ranks(scores: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda gene: (-float(scores[gene]), gene))
    return {gene: rank for rank, gene in enumerate(ordered, 1)}


def _model_score(
    forward_sum: float,
    total_hpo_weight: float,
    reverse_sum: float,
    mpo_count: int,
    max_pair: float,
    config: Mapping[str, Any],
) -> float:
    alpha = float(config.get("reverse_match_weight", 1.0))
    denominator = total_hpo_weight + alpha * mpo_count
    bma = (
        (forward_sum + alpha * reverse_sum) / denominator
        if denominator > 0
        else 0.0
    )
    bma_weight = float(config.get("bma_weight", 0.8))
    max_weight = float(config.get("max_pair_weight", 0.2))
    norm = bma_weight + max_weight
    return (bma_weight * bma + max_weight * max_pair) / norm if norm else bma


def calculate_phenotype_stability(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    models: Mapping[str, MouseModel],
    phenotypes: Mapping[str, Sequence[ModelPhenotype]],
    provider: SimilarityProvider,
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    eligible_models = [
        model
        for model_id, model in sorted(models.items())
        if any(row.status == "abnormal" for row in phenotypes.get(model_id, ()))
    ]
    candidate_genes = sorted({model.human_gene_symbol for model in eligible_models})
    base_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        variant_ids = ["full", *[term.hpo_id for term in terms]]
        best_scores = {
            variant: {gene: 0.0 for gene in candidate_genes}
            for variant in variant_ids
        }
        best_models = {
            variant: {gene: "" for gene in candidate_genes}
            for variant in variant_ids
        }
        full_details: dict[str, dict[str, Any]] = {
            gene: {
                "best_hpo_id": "",
                "best_mpo_id": "",
                "hpo_mpo_coverage": 0.0,
            }
            for gene in candidate_genes
        }
        total_weight = sum(term.weight for term in terms)
        for model in eligible_models:
            abnormal = [
                row for row in phenotypes.get(model.model_id, ()) if row.status == "abnormal"
            ]
            if not abnormal:
                continue
            matrix = [
                [provider.get_score(term.hpo_id, phenotype.mpo_id) for phenotype in abnormal]
                for term in terms
            ]
            hpo_best = [max(row, default=0.0) for row in matrix]
            forward_sum = sum(
                term.weight * hpo_best[index] for index, term in enumerate(terms)
            )
            reverse_top = []
            for mpo_index in range(len(abnormal)):
                reverse_top.append(_top_two([matrix[hpo_index][mpo_index] for hpo_index in range(len(terms))]))
            reverse_sum = sum(item[0] for item in reverse_top)
            max_pair, max_hpo_index, second_max_pair = _top_two(hpo_best)
            full_score = _model_score(
                forward_sum,
                total_weight,
                reverse_sum,
                len(abnormal),
                max_pair,
                config,
            )
            gene = model.human_gene_symbol
            current = best_scores["full"][gene]
            current_model = best_models["full"][gene]
            if full_score > current or (
                math.isclose(full_score, current)
                and (not current_model or model.model_id < current_model)
            ):
                best_scores["full"][gene] = full_score
                best_models["full"][gene] = model.model_id
                if max_hpo_index >= 0:
                    row = matrix[max_hpo_index]
                    best_mpo_index = max(
                        range(len(row)), key=lambda index: (row[index], -index)
                    )
                    best_hpo = terms[max_hpo_index].hpo_id
                    best_mpo = abnormal[best_mpo_index].mpo_id
                else:
                    best_hpo = best_mpo = ""
                full_details[gene] = {
                    "best_hpo_id": best_hpo,
                    "best_mpo_id": best_mpo,
                    "hpo_mpo_coverage": (
                        sum(term.weight for index, term in enumerate(terms) if hpo_best[index] > 0)
                        / total_weight
                        if total_weight
                        else 0.0
                    ),
                }
            if len(terms) <= 1:
                continue
            for removed_index, removed_term in enumerate(terms):
                variant = removed_term.hpo_id
                loo_forward = (
                    forward_sum - removed_term.weight * hpo_best[removed_index]
                )
                loo_reverse = sum(
                    item[2] if item[1] == removed_index else item[0]
                    for item in reverse_top
                )
                loo_max = (
                    second_max_pair if max_hpo_index == removed_index else max_pair
                )
                loo_score = _model_score(
                    loo_forward,
                    total_weight - removed_term.weight,
                    loo_reverse,
                    len(abnormal),
                    loo_max,
                    config,
                )
                current = best_scores[variant][gene]
                current_model = best_models[variant][gene]
                if loo_score > current or (
                    math.isclose(loo_score, current)
                    and (not current_model or model.model_id < current_model)
                ):
                    best_scores[variant][gene] = loo_score
                    best_models[variant][gene] = model.model_id
        dense_by_variant = {
            variant: _dense_ranks(scores) for variant, scores in best_scores.items()
        }
        ordinal_full = _ordinal_ranks(best_scores["full"])
        valid_variants = ["full"]
        for term in terms:
            variant = term.hpo_id
            valid = len(terms) > 1 and any(
                score > 0 for score in best_scores[variant].values()
            )
            if valid:
                valid_variants.append(variant)
            for gene in candidate_genes:
                loo_rows.append(
                    {
                        "disease_id": disease_id,
                        "removed_hpo_id": variant,
                        "gene_symbol": gene,
                        "phenotype_score": best_scores[variant][gene],
                        "dense_rank": dense_by_variant[variant][gene],
                        "best_model_id": best_models[variant][gene],
                        "valid_run": valid,
                    }
                )
        for gene in candidate_genes:
            detail = full_details[gene]
            base_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": terms[0].disease_name if terms else disease_id,
                    "gene_symbol": gene,
                    "base_score": best_scores["full"][gene],
                    "base_rank": ordinal_full[gene],
                    "base_dense_rank": dense_by_variant["full"][gene],
                    "best_model_id": best_models["full"][gene],
                    "best_hpo_id": detail["best_hpo_id"],
                    "best_mpo_id": detail["best_mpo_id"],
                    "hpo_mpo_coverage": detail["hpo_mpo_coverage"],
                }
            )
        edge_count = 0
        rank_maps = [dense_by_variant[variant] for variant in valid_variants]
        for left_index, left in enumerate(candidate_genes):
            for right in candidate_genes[left_index + 1 :]:
                left_over_right = all(ranks[left] < ranks[right] for ranks in rank_maps)
                right_over_left = all(ranks[right] < ranks[left] for ranks in rank_maps)
                if not left_over_right and not right_over_left:
                    continue
                higher, lower = (left, right) if left_over_right else (right, left)
                edge_count += 1
                constraint_rows.append(
                    {
                        "disease_id": disease_id,
                        "higher_gene": higher,
                        "lower_gene": lower,
                        "valid_ranking_count": len(valid_variants),
                        "full_rank_higher": dense_by_variant["full"][higher],
                        "full_rank_lower": dense_by_variant["full"][lower],
                        "stable_in_all_runs": True,
                    }
                )
        summary_rows.append(
            {
                "disease_id": disease_id,
                "nonredundant_hpo_count": len(terms),
                "valid_loo_count": max(0, len(valid_variants) - 1),
                "ranking_count_in_constraints": len(valid_variants),
                "candidate_gene_count": len(candidate_genes),
                "stable_constraint_count": edge_count,
                "stability_mode": (
                    "leave_one_hpo_out"
                    if len(valid_variants) > 1
                    else "full_order_fallback"
                ),
            }
        )
    return base_rows, loo_rows, constraint_rows, summary_rows


def _exomiser_score(
    forward_sum: float,
    query_count: int,
    reverse_best: Sequence[float],
    max_pair: float,
    ideal_sum: float,
    ideal_max: float,
) -> float:
    matched_reverse = [value for value in reverse_best if value > 0]
    denominator = query_count + len(matched_reverse)
    if denominator <= 0 or ideal_sum <= 0 or ideal_max <= 0:
        return 0.0
    raw_average = (forward_sum + sum(matched_reverse)) / denominator
    ideal_average = ideal_sum / query_count
    combined = 0.5 * (raw_average / ideal_average + max_pair / ideal_max)
    return 100.0 * min(1.0, combined)


def calculate_exomiser_phenotype_stability(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    models: Mapping[str, MouseModel],
    phenotypes: Mapping[str, Sequence[ModelPhenotype]],
    matrix_provider: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Calculate current Exomiser PhenoDigm full and exact LOO gene ranks."""
    eligible_models = [
        model
        for model_id, model in sorted(models.items())
        if any(row.status == "abnormal" for row in phenotypes.get(model_id, ()))
    ]
    candidate_genes = sorted({model.human_gene_symbol for model in eligible_models})
    base_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        variant_ids = ["full", *[term.hpo_id for term in terms]]
        best_scores = {
            variant: {gene: 0.0 for gene in candidate_genes}
            for variant in variant_ids
        }
        best_models = {
            variant: {gene: "" for gene in candidate_genes}
            for variant in variant_ids
        }
        full_details = {
            gene: {
                "best_hpo_id": "",
                "best_mpo_id": "",
                "hpo_mpo_coverage": 0.0,
            }
            for gene in candidate_genes
        }
        ideal_scores = [
            float(matrix_provider.best_by_hpo.get(term.hpo_id, (0.0, ""))[0])
            for term in terms
        ]
        mapped_flags = [term.hpo_id in matrix_provider.best_by_hpo for term in terms]
        query_count = sum(mapped_flags)
        ideal_sum = sum(ideal_scores)
        ideal_max, ideal_max_index, second_ideal_max = _top_two(ideal_scores)
        for model in eligible_models:
            abnormal = sorted(
                {
                    row.mpo_id
                    for row in phenotypes.get(model.model_id, ())
                    if row.status == "abnormal"
                }
            )
            if not abnormal:
                continue
            score_matrix = [
                [matrix_provider.get(term.hpo_id, mpo_id) for mpo_id in abnormal]
                for term in terms
            ]
            hpo_best = [max(row, default=0.0) for row in score_matrix]
            forward_sum = sum(hpo_best)
            reverse_top = [
                _top_two(
                    [score_matrix[hpo_index][mpo_index] for hpo_index in range(len(terms))]
                )
                for mpo_index in range(len(abnormal))
            ]
            reverse_best = [item[0] for item in reverse_top]
            max_pair, max_hpo_index, second_max_pair = _top_two(hpo_best)
            full_score = _exomiser_score(
                forward_sum,
                query_count,
                reverse_best,
                max_pair,
                ideal_sum,
                ideal_max,
            )
            gene = model.human_gene_symbol
            current = best_scores["full"][gene]
            current_model = best_models["full"][gene]
            if full_score > current or (
                math.isclose(full_score, current)
                and (not current_model or model.model_id < current_model)
            ):
                best_scores["full"][gene] = full_score
                best_models["full"][gene] = model.model_id
                if max_hpo_index >= 0 and abnormal:
                    best_mpo_index = max(
                        range(len(abnormal)),
                        key=lambda index: (
                            score_matrix[max_hpo_index][index],
                            -index,
                        ),
                    )
                    best_hpo = terms[max_hpo_index].hpo_id
                    best_mpo = abnormal[best_mpo_index]
                else:
                    best_hpo = best_mpo = ""
                full_details[gene] = {
                    "best_hpo_id": best_hpo,
                    "best_mpo_id": best_mpo,
                    "hpo_mpo_coverage": (
                        sum(value > 0 for value in hpo_best) / len(terms)
                        if terms
                        else 0.0
                    ),
                }
            if len(terms) <= 1:
                continue
            for removed_index, removed_term in enumerate(terms):
                variant = removed_term.hpo_id
                loo_query_count = query_count - int(mapped_flags[removed_index])
                loo_forward = forward_sum - hpo_best[removed_index]
                loo_reverse = [
                    item[2] if item[1] == removed_index else item[0]
                    for item in reverse_top
                ]
                loo_max = (
                    second_max_pair if max_hpo_index == removed_index else max_pair
                )
                loo_ideal_max = (
                    second_ideal_max
                    if ideal_max_index == removed_index
                    else ideal_max
                )
                loo_score = _exomiser_score(
                    loo_forward,
                    loo_query_count,
                    loo_reverse,
                    loo_max,
                    ideal_sum - ideal_scores[removed_index],
                    loo_ideal_max,
                )
                current = best_scores[variant][gene]
                current_model = best_models[variant][gene]
                if loo_score > current or (
                    math.isclose(loo_score, current)
                    and (not current_model or model.model_id < current_model)
                ):
                    best_scores[variant][gene] = loo_score
                    best_models[variant][gene] = model.model_id

        dense_by_variant = {
            variant: _dense_ranks(scores) for variant, scores in best_scores.items()
        }
        ordinal_full = _ordinal_ranks(best_scores["full"])
        valid_variants = ["full"]
        for term in terms:
            variant = term.hpo_id
            valid = len(terms) > 1 and any(
                score > 0 for score in best_scores[variant].values()
            )
            if valid:
                valid_variants.append(variant)
            for gene in candidate_genes:
                loo_rows.append(
                    {
                        "disease_id": disease_id,
                        "removed_hpo_id": variant,
                        "gene_symbol": gene,
                        "phenotype_score": best_scores[variant][gene],
                        "dense_rank": dense_by_variant[variant][gene],
                        "best_model_id": best_models[variant][gene],
                        "valid_run": valid,
                    }
                )
        for gene in candidate_genes:
            detail = full_details[gene]
            base_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": terms[0].disease_name if terms else disease_id,
                    "gene_symbol": gene,
                    "base_score": best_scores["full"][gene],
                    "base_rank": ordinal_full[gene],
                    "base_dense_rank": dense_by_variant["full"][gene],
                    "best_model_id": best_models["full"][gene],
                    "best_hpo_id": detail["best_hpo_id"],
                    "best_mpo_id": detail["best_mpo_id"],
                    "hpo_mpo_coverage": detail["hpo_mpo_coverage"],
                }
            )
        rank_maps = [dense_by_variant[variant] for variant in valid_variants]
        edge_count = 0
        for left_index, left in enumerate(candidate_genes):
            for right in candidate_genes[left_index + 1 :]:
                left_over_right = all(ranks[left] < ranks[right] for ranks in rank_maps)
                right_over_left = all(ranks[right] < ranks[left] for ranks in rank_maps)
                if not left_over_right and not right_over_left:
                    continue
                higher, lower = (left, right) if left_over_right else (right, left)
                edge_count += 1
                constraint_rows.append(
                    {
                        "disease_id": disease_id,
                        "higher_gene": higher,
                        "lower_gene": lower,
                        "valid_ranking_count": len(valid_variants),
                        "full_rank_higher": dense_by_variant["full"][higher],
                        "full_rank_lower": dense_by_variant["full"][lower],
                        "stable_in_all_runs": True,
                    }
                )
        summary_rows.append(
            {
                "disease_id": disease_id,
                "nonredundant_hpo_count": len(terms),
                "valid_loo_count": max(0, len(valid_variants) - 1),
                "ranking_count_in_constraints": len(valid_variants),
                "candidate_gene_count": len(candidate_genes),
                "stable_constraint_count": edge_count,
                "stability_mode": (
                    "leave_one_hpo_out"
                    if len(valid_variants) > 1
                    else "full_order_fallback"
                ),
            }
        )
    return base_rows, loo_rows, constraint_rows, summary_rows


BASE_PARAMETER_FREE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "base_score",
    "base_rank",
    "base_dense_rank",
    "best_model_id",
    "best_hpo_id",
    "best_mpo_id",
    "hpo_mpo_coverage",
]

LOO_FIELDS = [
    "disease_id",
    "removed_hpo_id",
    "gene_symbol",
    "phenotype_score",
    "dense_rank",
    "best_model_id",
    "valid_run",
]

CONSTRAINT_FIELDS = [
    "disease_id",
    "higher_gene",
    "lower_gene",
    "valid_ranking_count",
    "full_rank_higher",
    "full_rank_lower",
    "stable_in_all_runs",
]

STABILITY_SUMMARY_FIELDS = [
    "disease_id",
    "nonredundant_hpo_count",
    "valid_loo_count",
    "ranking_count_in_constraints",
    "candidate_gene_count",
    "stable_constraint_count",
    "stability_mode",
]
