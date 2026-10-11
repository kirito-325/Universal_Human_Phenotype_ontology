from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from .schemas import DiseaseTerm, ModelPhenotype, MouseModel
from .similarity import SimilarityProvider


def percentile_scores(rows: Sequence[Mapping[str, Any]], score_key: str, id_key: str) -> dict[str, float]:
    if not rows:
        return {}
    groups: dict[float, list[str]] = defaultdict(list)
    for row in rows:
        groups[float(row[score_key])].append(str(row[id_key]))
    ordered_scores = sorted(groups, reverse=True)
    n = len(rows)
    output: dict[str, float] = {}
    position = 1
    for score in ordered_scores:
        identifiers = sorted(groups[score])
        average_rank = position + (len(identifiers) - 1) / 2.0
        percentile = 1.0 if n == 1 else 1.0 - (average_rank - 1.0) / (n - 1.0)
        for identifier in identifiers:
            output[identifier] = percentile
        position += len(identifiers)
    return output


def assign_ordinal_ranks(
    rows: list[dict[str, Any]], score_key: str, id_key: str, rank_key: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), str(row[id_key])))
    for rank, row in enumerate(ordered, 1):
        row[rank_key] = rank
    return ordered


def score_model(
    disease_terms: Sequence[DiseaseTerm],
    model: MouseModel,
    phenotypes: Sequence[ModelPhenotype],
    provider: SimilarityProvider,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    abnormal = [row for row in phenotypes if row.status == "abnormal"]
    total_hpo_weight = sum(row.weight for row in disease_terms)
    evidence_rows: list[dict[str, Any]] = []
    forward_sum = 0.0
    explained_weight = 0.0
    best_pair_score = 0.0
    best_pair_hpo = ""
    best_pair_mpo = ""
    threshold = float(config.get("match_threshold", 0.0))
    for term in disease_terms:
        best_score = 0.0
        best_pheno: ModelPhenotype | None = None
        for phenotype in abnormal:
            candidate = provider.get_score(term.hpo_id, phenotype.mpo_id)
            if candidate > best_score or (
                math.isclose(candidate, best_score) and best_pheno and phenotype.mpo_id < best_pheno.mpo_id
            ):
                best_score = candidate
                best_pheno = phenotype
        weighted = term.weight * best_score
        forward_sum += weighted
        if best_score > threshold:
            explained_weight += term.weight
        if best_score > best_pair_score:
            best_pair_score = best_score
            best_pair_hpo = term.hpo_id
            best_pair_mpo = best_pheno.mpo_id if best_pheno else ""
        pair_evidence = provider.get_evidence(term.hpo_id, best_pheno.mpo_id) if best_pheno else {}
        evidence_rows.append(
            {
                "disease_id": term.disease_id,
                "disease_name": term.disease_name,
                "gene_symbol": model.human_gene_symbol,
                "model_id": model.model_id,
                "hpo_id": term.hpo_id,
                "hpo_name": term.hpo_name,
                "hpo_weight": term.weight,
                "matched_mpo_id": best_pheno.mpo_id if best_pheno else "",
                "matched_mpo_name": best_pheno.mpo_name if best_pheno else "",
                "hpo_mpo_score": best_score,
                "weighted_match_score": weighted,
                "mapping_relation": pair_evidence.get("relation", "not_mapped"),
                "mapping_source": pair_evidence.get("source", ""),
            }
        )
    reverse_sum = 0.0
    for phenotype in abnormal:
        reverse_sum += max((provider.get_score(term.hpo_id, phenotype.mpo_id) for term in disease_terms), default=0.0)
    forward_score = forward_sum / total_hpo_weight if total_hpo_weight else 0.0
    reverse_score = reverse_sum / len(abnormal) if abnormal else 0.0
    alpha = float(config.get("reverse_match_weight", 1.0))
    denominator = total_hpo_weight + alpha * len(abnormal)
    bma = (forward_sum + alpha * reverse_sum) / denominator if denominator else 0.0
    bma_weight = float(config.get("bma_weight", 0.8))
    max_pair_weight = float(config.get("max_pair_weight", 0.2))
    norm = bma_weight + max_pair_weight
    model_score = (bma_weight * bma + max_pair_weight * best_pair_score) / norm if norm else bma
    row = {
        "disease_id": disease_terms[0].disease_id,
        "disease_name": disease_terms[0].disease_name,
        "model_id": model.model_id,
        "model_name": model.model_name,
        "mouse_gene_symbol": model.mouse_gene_symbol,
        "human_gene_symbol": model.human_gene_symbol,
        "allele_id": model.allele_id,
        "allele_type": model.allele_type,
        "zygosity": model.zygosity,
        "conditional": model.conditional,
        "hpo_to_mpo_score": forward_score,
        "mpo_to_hpo_score": reverse_score,
        "bma_score": bma,
        "max_pair_score": best_pair_score,
        "model_score": model_score,
        "hpo_mpo_coverage": explained_weight / total_hpo_weight if total_hpo_weight else 0.0,
        "hpo_count": len(disease_terms),
        "abnormal_mpo_count": len(abnormal),
        "top_hpo_id": best_pair_hpo,
        "top_mpo_id": best_pair_mpo,
    }
    return row, evidence_rows


def run_baseline(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    models: Mapping[str, MouseModel],
    phenotypes: Mapping[str, Sequence[ModelPhenotype]],
    provider: SimilarityProvider,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    best_model_evidence: list[dict[str, Any]] = []
    eligible_models = [
        model for model_id, model in sorted(models.items()) if any(p.status == "abnormal" for p in phenotypes.get(model_id, ()))
    ]
    for disease_id, disease_terms in sorted(diseases.items()):
        per_gene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        best_evidence: dict[str, list[dict[str, Any]]] = {}
        best_key: dict[str, tuple[float, str]] = {}
        for model in eligible_models:
            model_row, evidence = score_model(
                disease_terms,
                model,
                phenotypes.get(model.model_id, ()),
                provider,
                config,
            )
            model_rows.append(model_row)
            gene = model.human_gene_symbol
            per_gene_rows[gene].append(model_row)
            key = (float(model_row["model_score"]), model.model_id)
            previous = best_key.get(gene)
            if previous is None or key[0] > previous[0] or (math.isclose(key[0], previous[0]) and key[1] < previous[1]):
                best_key[gene] = key
                best_evidence[gene] = evidence

        current_gene_rows: list[dict[str, Any]] = []
        known_genes = {term.gene_symbol for term in disease_terms if term.gene_symbol}
        for gene, rows in sorted(per_gene_rows.items()):
            ordered_models = sorted(rows, key=lambda row: (-float(row["model_score"]), str(row["model_id"])))
            best = ordered_models[0]
            scores = [float(row["model_score"]) for row in ordered_models]
            top_three = scores[:3]
            row = {
                "disease_id": disease_id,
                "disease_name": disease_terms[0].disease_name,
                "gene_id": gene,
                "gene_symbol": gene,
                "best_model_id": best["model_id"],
                "best_model_name": best["model_name"],
                "base_score": best["model_score"],
                "bma_score": best["bma_score"],
                "max_pair_score": best["max_pair_score"],
                "hpo_mpo_coverage": best["hpo_mpo_coverage"],
                "top3_model_mean": mean(top_three),
                "model_count": len(rows),
                "abnormal_mpo_count": best["abnormal_mpo_count"],
                "model_score_stability": pstdev(scores) if len(scores) > 1 else 0.0,
                "top_hpo_id": best["top_hpo_id"],
                "top_mpo_id": best["top_mpo_id"],
                "is_known_gene": gene in known_genes,
            }
            current_gene_rows.append(row)
            best_model_evidence.extend(best_evidence[gene])
        current_gene_rows = assign_ordinal_ranks(current_gene_rows, "base_score", "gene_symbol", "base_rank")
        percentiles = percentile_scores(current_gene_rows, "base_score", "gene_symbol")
        for row in current_gene_rows:
            row["base_percentile"] = percentiles[row["gene_symbol"]]
        gene_rows.extend(current_gene_rows)
    return model_rows, gene_rows, best_model_evidence


MODEL_SCORE_FIELDS = [
    "disease_id",
    "disease_name",
    "model_id",
    "model_name",
    "mouse_gene_symbol",
    "human_gene_symbol",
    "allele_id",
    "allele_type",
    "zygosity",
    "conditional",
    "hpo_to_mpo_score",
    "mpo_to_hpo_score",
    "bma_score",
    "max_pair_score",
    "model_score",
    "hpo_mpo_coverage",
    "hpo_count",
    "abnormal_mpo_count",
    "top_hpo_id",
    "top_mpo_id",
]

BASE_GENE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_id",
    "gene_symbol",
    "best_model_id",
    "best_model_name",
    "base_score",
    "base_rank",
    "base_percentile",
    "bma_score",
    "max_pair_score",
    "hpo_mpo_coverage",
    "top3_model_mean",
    "model_count",
    "abnormal_mpo_count",
    "model_score_stability",
    "top_hpo_id",
    "top_mpo_id",
    "is_known_gene",
]

HPO_MPO_EVIDENCE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "model_id",
    "hpo_id",
    "hpo_name",
    "hpo_weight",
    "matched_mpo_id",
    "matched_mpo_name",
    "hpo_mpo_score",
    "weighted_match_score",
    "mapping_relation",
    "mapping_source",
]
