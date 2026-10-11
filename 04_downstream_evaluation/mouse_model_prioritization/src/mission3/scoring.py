from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from .models import (
    DiseasePhenotype,
    MechanismEdge,
    ModelPhenotype,
    MouseModel,
    PhenotypeEvidence,
    TermMatch,
)
from .ontology import Ontology


def direction_compatibility(left: str, right: str) -> float | None:
    unknown = {"", "unknown", "abnormal", "present", "bidirectional"}
    if left in unknown or right in unknown:
        return None
    if left == right:
        return 1.0
    if {left, right} == {"decreased", "absent"}:
        return 0.8
    return -1.0


def score_mechanism_pair(
    hpo_edge: MechanismEdge,
    mp_edge: MechanismEdge,
    cl_ontology: Ontology,
    go_ontology: Ontology,
    config: Mapping[str, float],
) -> tuple[float, float | None]:
    cl_score = cl_ontology.similarity(hpo_edge.cl_id, mp_edge.cl_id)
    if hpo_edge.cl_id == mp_edge.cl_id:
        cl_score = 1.0
    go_score = go_ontology.similarity(hpo_edge.go_id, mp_edge.go_id)
    if hpo_edge.go_id == mp_edge.go_id:
        go_score = 1.0
    direction = direction_compatibility(hpo_edge.direction, mp_edge.direction)
    direction_component = 0.0 if direction is None else direction
    raw = (
        config.get("cl_weight", 0.4) * cl_score
        + config.get("go_weight", 0.4) * go_score
        + config.get("direction_weight", 0.2) * direction_component
    )
    evidence_factor = math.sqrt(hpo_edge.evidence_weight * mp_edge.evidence_weight)
    return max(-1.0, min(1.0, raw * evidence_factor)), direction


def best_cpo_match(
    hpo_edges: Sequence[MechanismEdge],
    mp_edges: Sequence[MechanismEdge],
    cl_ontology: Ontology,
    go_ontology: Ontology,
    config: Mapping[str, float],
) -> tuple[float | None, MechanismEdge | None, MechanismEdge | None, float | None]:
    if not hpo_edges or not mp_edges:
        return None, None, None, None
    best: tuple[float, MechanismEdge, MechanismEdge, float | None] | None = None
    for hpo_edge in hpo_edges:
        for mp_edge in mp_edges:
            score, direction = score_mechanism_pair(hpo_edge, mp_edge, cl_ontology, go_ontology, config)
            candidate = (score, hpo_edge, mp_edge, direction)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best


def genetic_compatibility(disease: DiseasePhenotype, model: MouseModel) -> tuple[float, str, str]:
    mechanism = disease.molecular_mechanism.lower().replace("-", "_").replace(" ", "_")
    inheritance = disease.inheritance.lower().replace("-", "_").replace(" ", "_")
    allele = model.allele_type.lower().replace("-", "_").replace(" ", "_")
    zygosity = model.zygosity.lower().replace("-", "_").replace(" ", "_")
    is_null = any(token in allele for token in ("null", "knockout", "loss_of_function", "deletion"))
    is_knockin = any(token in allele for token in ("knockin", "knock_in", "missense", "point_mutation", "humanized"))
    is_hetero = any(token in zygosity for token in ("heterozyg", "hetero", "het"))
    is_homo = any(token in zygosity for token in ("homozyg", "homo", "hom"))
    is_hemizygous = any(token in zygosity for token in ("hemizyg", "hemiz", "hemi"))

    if model.conditional and mechanism in {"tissue_specific", "cell_specific"}:
        return 1.0, "concordant", "Conditional perturbation matches a tissue/cell-restricted disease mechanism."
    if mechanism in {"haploinsufficiency", "haploinsufficient", "monoallelic_lof"}:
        if is_null and is_hetero:
            return 1.0, "concordant", "Heterozygous null matches haploinsufficiency."
        if is_null and is_homo:
            return -0.5, "conflicting", "Homozygous null is more severe than haploinsufficiency."
    if mechanism in {"hemizygous_loss_of_function", "hemizygous_lof", "x_linked_lof"}:
        if is_null and is_hemizygous:
            return 1.0, "concordant", "Hemizygous null matches an X-linked loss-of-function mechanism."
        if is_null and is_hetero:
            return 0.25, "plausible", "Heterozygous null may model a variably affected carrier state."
        if is_null and is_homo:
            return -0.5, "conflicting", "Homozygous null does not reproduce the canonical hemizygous state."
    if mechanism in {"loss_of_function", "lof", "biallelic_lof", "recessive_lof"} or (
        mechanism == "unknown" and any(token in inheritance for token in ("recessive", "biallelic", "ar"))
    ):
        if is_null and is_homo:
            return 1.0, "concordant", "Homozygous null matches a biallelic loss-of-function mechanism."
        if is_null and is_hetero:
            return -0.25, "conflicting", "Heterozygous null may not model a biallelic loss-of-function disease."
    if mechanism in {"gain_of_function", "gof", "activating"}:
        if is_knockin and any(token in allele for token in ("activating", "gain", "missense", "humanized")):
            return 1.0, "concordant", "Knock-in allele is compatible with gain of function."
        if is_null:
            return -1.0, "conflicting", "Null allele opposes a gain-of-function mechanism."
    if mechanism in {"dominant_negative", "dn"}:
        if is_knockin:
            return 0.8, "plausible", "Missense/knock-in allele is compatible with dominant negative action."
        if is_null:
            return -0.75, "conflicting", "Null allele does not reproduce a dominant-negative product."
    if model.conditional:
        return 0.25, "plausible", "Conditional model may add useful cell specificity; disease mechanism is not explicit."
    return 0.0, "unknown", "Human or mouse molecular mechanism is insufficiently structured."


def score_disease_model(
    disease_rows: Sequence[DiseasePhenotype],
    model: MouseModel,
    model_phenotypes: Sequence[ModelPhenotype],
    surface_matches: Mapping[str, Mapping[str, TermMatch]],
    hpo_mechanisms: Mapping[str, Sequence[MechanismEdge]],
    mp_mechanisms: Mapping[str, Sequence[MechanismEdge]],
    cl_ontology: Ontology,
    go_ontology: Ontology,
    config: Mapping[str, object],
) -> tuple[dict[str, object], list[PhenotypeEvidence]]:
    threshold = float(config.get("surface_match_threshold", 0.45))
    core_threshold = float(config.get("core_reproduction_threshold", 0.65))
    cpo_config = config.get("cpo", {})
    assert isinstance(cpo_config, Mapping)
    positive = [row for row in model_phenotypes if row.status == "abnormal"]
    tested_normal = [row for row in model_phenotypes if row.status == "tested_normal"]
    all_tested = positive + tested_normal
    evidence_rows: list[PhenotypeEvidence] = []
    weighted_surface = 0.0
    comparable_weight = 0.0
    total_weight = sum(row.weight for row in disease_rows) or 1.0
    weighted_cpo = 0.0
    cpo_weight = 0.0
    core_reproduced = 0
    core_total = sum(1 for row in disease_rows if row.is_core)
    tested_normal_count = 0
    untested_count = 0
    opposite_count = 0

    for disease_pheno in disease_rows:
        abnormal_candidates: list[tuple[float, ModelPhenotype, TermMatch]] = []
        normal_candidates: list[tuple[float, ModelPhenotype, TermMatch]] = []
        low_candidates: list[tuple[float, ModelPhenotype, TermMatch]] = []
        for mouse_pheno in all_tested:
            match = surface_matches.get(disease_pheno.hpo_id, {}).get(mouse_pheno.mp_id)
            if match is not None:
                candidate = (match.similarity, mouse_pheno, match)
                low_candidates.append(candidate)
                if mouse_pheno.status == "abnormal":
                    abnormal_candidates.append(candidate)
                elif mouse_pheno.status == "tested_normal":
                    normal_candidates.append(candidate)
        for candidates in (abnormal_candidates, normal_candidates, low_candidates):
            candidates.sort(key=lambda item: (-item[0], item[1].mp_id))
        if abnormal_candidates and abnormal_candidates[0][0] >= threshold:
            best = abnormal_candidates[0]
        elif normal_candidates and normal_candidates[0][0] >= threshold:
            best = normal_candidates[0]
        else:
            best = low_candidates[0] if low_candidates else None
        row = PhenotypeEvidence(
            disease_id=disease_pheno.disease_id,
            model_id=model.model_id,
            hpo_id=disease_pheno.hpo_id,
            hpo_name=disease_pheno.hpo_name,
            hpo_weight=disease_pheno.weight,
            hpo_onset=disease_pheno.onset,
            hpo_direction=disease_pheno.direction,
        )
        disease_mechanisms = hpo_mechanisms.get(disease_pheno.hpo_id, ())
        if disease_mechanisms:
            strongest = max(disease_mechanisms, key=lambda edge: (edge.evidence_weight, edge.cl_id, edge.go_id))
            row.best_hpo_cl_id = strongest.cl_id
            row.best_hpo_go_id = strongest.go_id
        if best is None:
            if surface_matches.get(disease_pheno.hpo_id):
                row.assay_state = "untested"
                untested_count += 1
            else:
                row.assay_state = "not_mappable"
        else:
            similarity, mouse_pheno, match = best
            row.best_mp_id = mouse_pheno.mp_id
            row.best_mp_name = mouse_pheno.mp_name
            row.surface_similarity = similarity
            row.mapping_relation = match.relation
            row.mp_direction = mouse_pheno.direction
            row.mp_sex = mouse_pheno.sex
            row.mp_life_stage = mouse_pheno.life_stage
            if similarity < threshold:
                row.assay_state = "untested"
                untested_count += 1
            elif mouse_pheno.status == "tested_normal":
                row.assay_state = "tested_normal"
                comparable_weight += disease_pheno.weight
                tested_normal_count += 1
            else:
                comparable_weight += disease_pheno.weight
                direction = direction_compatibility(disease_pheno.direction, mouse_pheno.direction)
                if direction == -1.0:
                    row.assay_state = "opposite_direction"
                    opposite_count += 1
                else:
                    row.assay_state = "reproduced"
                    weighted_surface += disease_pheno.weight * similarity
                    if disease_pheno.is_core and similarity >= core_threshold:
                        core_reproduced += 1

            if mouse_pheno.status == "abnormal" and similarity >= threshold:
                cpo_score, hpo_edge, mp_edge, cpo_direction = best_cpo_match(
                    disease_mechanisms,
                    mp_mechanisms.get(mouse_pheno.mp_id, ()),
                    cl_ontology,
                    go_ontology,
                    cpo_config,
                )
                if cpo_score is not None:
                    row.cpo_score = cpo_score
                    row.cpo_coverage = 1.0
                    row.direction_compatibility = cpo_direction
                    assert hpo_edge is not None and mp_edge is not None
                    row.best_hpo_cl_id = hpo_edge.cl_id
                    row.best_hpo_go_id = hpo_edge.go_id
                    row.best_mp_cl_id = mp_edge.cl_id
                    row.best_mp_go_id = mp_edge.go_id
                    weighted_cpo += disease_pheno.weight * cpo_score
                    cpo_weight += disease_pheno.weight
                    if hpo_edge.provisional:
                        row.evidence_notes.append("HPO mechanism is provisional and down-weighted.")
        evidence_rows.append(row)

    disease = disease_rows[0]
    genetic_score, genetic_class, genetic_reason = genetic_compatibility(disease, model)
    surface_score = weighted_surface / total_weight
    cpo_score = weighted_cpo / cpo_weight if cpo_weight else None
    cpo_coverage = cpo_weight / total_weight
    weights = config.get("weights", {})
    assert isinstance(weights, Mapping)
    total_score = float(weights.get("surface", 1.0)) * surface_score
    if cpo_score is not None:
        total_score += float(weights.get("cpo", 0.25)) * cpo_coverage * cpo_score
    total_score += float(weights.get("genetic", 0.2)) * genetic_score
    summary: dict[str, object] = {
        "surface_score": surface_score,
        "surface_coverage": comparable_weight / total_weight,
        "core_reproduced": core_reproduced,
        "core_total": core_total,
        "cpo_score": cpo_score,
        "cpo_coverage": cpo_coverage,
        "genetic_score": genetic_score,
        "genetic_class": genetic_class,
        "genetic_reason": genetic_reason,
        "tested_normal_count": tested_normal_count,
        "untested_count": untested_count,
        "opposite_direction_count": opposite_count,
        "total_score": total_score,
    }
    return summary, evidence_rows


def candidates_for_disease(
    disease_rows: Sequence[DiseasePhenotype],
    models: Mapping[str, MouseModel],
    mode: str,
) -> list[MouseModel]:
    if mode == "all_models":
        return sorted(models.values(), key=lambda model: model.model_id)
    genes = {row.gene_symbol.upper() for row in disease_rows if row.gene_symbol}
    return sorted(
        (
            model
            for model in models.values()
            if genes.intersection(_gene_tokens(model.human_gene_symbol))
            or (not model.human_gene_symbol and genes.intersection(_gene_tokens(model.mouse_gene_symbol)))
        ),
        key=lambda model: model.model_id,
    )


def _gene_tokens(value: str) -> set[str]:
    return {
        token.strip().upper()
        for token in value.replace("|", ";").replace(",", ";").split(";")
        if token.strip()
    }


def ndcg_at_k(labels: Sequence[int], k: int) -> float:
    gains = sum(label / math.log2(index + 2) for index, label in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)
    ideal_gains = sum(label / math.log2(index + 2) for index, label in enumerate(ideal[:k]))
    return gains / ideal_gains if ideal_gains else 0.0


def evaluate_rankings(rankings: Mapping[str, Sequence[object]]) -> dict[str, float | int]:
    reciprocal_ranks: list[float] = []
    recalls = {1: [], 5: [], 10: []}
    ndcgs: list[float] = []
    evaluated = 0
    for rows in rankings.values():
        labels = [1 if getattr(row, "known_model") else 0 for row in rows]
        if not any(labels):
            continue
        evaluated += 1
        first = labels.index(1) + 1
        reciprocal_ranks.append(1.0 / first)
        for k in recalls:
            recalls[k].append(float(any(labels[:k])))
        ndcgs.append(ndcg_at_k(labels, 10))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "evaluated_diseases": evaluated,
        "mrr": mean(reciprocal_ranks),
        "recall_at_1": mean(recalls[1]),
        "recall_at_5": mean(recalls[5]),
        "recall_at_10": mean(recalls[10]),
        "ndcg_at_10": mean(ndcgs),
    }
