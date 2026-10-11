from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .io import (
    load_cell_catalog,
    load_config,
    load_cre_drivers,
    load_diseases,
    load_go_scopes,
    load_mechanisms,
    load_model_phenotypes,
    load_models,
    load_surface_matches,
    sha256,
    write_tsv,
)
from .models import ModelScore, PhenotypeEvidence
from .ontology import Ontology
from .report import write_html_report, write_jsonl_reports
from .scoring import candidates_for_disease, evaluate_rankings, score_disease_model


RANKING_FIELDS = [
    "disease_id",
    "disease_name",
    "rank",
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
    "sex",
    "life_stage",
    "surface_score",
    "surface_coverage",
    "core_reproduced",
    "core_total",
    "cpo_score",
    "cpo_coverage",
    "genetic_score",
    "genetic_class",
    "genetic_reason",
    "tested_normal_count",
    "untested_count",
    "opposite_direction_count",
    "total_score",
    "known_model",
    "availability",
]

EVIDENCE_FIELDS = [
    "disease_id",
    "model_id",
    "hpo_id",
    "hpo_name",
    "hpo_weight",
    "hpo_onset",
    "best_mp_id",
    "best_mp_name",
    "surface_similarity",
    "mapping_relation",
    "assay_state",
    "hpo_direction",
    "mp_direction",
    "mp_sex",
    "mp_life_stage",
    "cpo_score",
    "cpo_coverage",
    "best_hpo_cl_id",
    "best_hpo_go_id",
    "best_mp_cl_id",
    "best_mp_go_id",
    "direction_compatibility",
    "suggested_cre_drivers",
    "evidence_notes",
]


def _score_dict(row: ModelScore) -> dict[str, object]:
    return asdict(row)


def _evidence_dict(row: PhenotypeEvidence) -> dict[str, object]:
    return asdict(row)


def _ablation_metrics(
    rankings: Mapping[str, Sequence[ModelScore]], config: Mapping[str, object]
) -> dict[str, dict[str, float | int]]:
    weights = config.get("weights", {})
    assert isinstance(weights, Mapping)
    cpo_weight = float(weights.get("cpo", 0.25))
    genetic_weight = float(weights.get("genetic", 0.2))
    score_functions = {
        "surface_only": lambda row: row.surface_score,
        "surface_plus_cpo": lambda row: row.surface_score
        + cpo_weight * row.cpo_coverage * (row.cpo_score or 0.0),
        "surface_plus_genetic": lambda row: row.surface_score + genetic_weight * row.genetic_score,
        "full": lambda row: row.total_score,
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, score_function in score_functions.items():
        ordered = {
            disease_id: sorted(rows, key=lambda row: (-score_function(row), row.model_id))
            for disease_id, rows in rankings.items()
        }
        result[name] = evaluate_rankings(ordered)
    return result


def run_pipeline(
    *,
    diseases_path: str | Path,
    models_path: str | Path,
    model_phenotypes_path: str | Path,
    mp_hpo_mapping_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    hpo_mechanisms_path: str | Path | None = None,
    mp_mechanisms_path: str | Path | None = None,
    cell_catalog_path: str | Path | None = None,
    go_classification_path: str | Path | None = None,
    cl_obo_path: str | Path | None = None,
    go_obo_path: str | Path | None = None,
    cre_drivers_path: str | Path | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    diseases = load_diseases(diseases_path)
    models = load_models(models_path)
    model_phenotypes = load_model_phenotypes(model_phenotypes_path)
    surface_matches = load_surface_matches(
        mp_hpo_mapping_path,
        score_column=str(config.get("surface_score_column", "semantic_score")),
    )
    if not diseases:
        raise ValueError("No valid disease-HPO rows were loaded.")
    if not models:
        raise ValueError("No valid model rows were loaded.")
    if not surface_matches:
        raise ValueError("No valid MP-HPO surface mappings were loaded.")
    cell_catalog = load_cell_catalog(cell_catalog_path)
    go_scopes = load_go_scopes(go_classification_path)
    cre_drivers = load_cre_drivers(cre_drivers_path)
    include_provisional = bool(config.get("include_provisional_mechanisms", False))
    hpo_mechanisms = load_mechanisms(
        hpo_mechanisms_path,
        "HPO",
        cell_catalog=cell_catalog,
        go_scopes=go_scopes,
        include_provisional=include_provisional,
    )
    mp_mechanisms = load_mechanisms(
        mp_mechanisms_path,
        "MPO",
        cell_catalog=cell_catalog,
        go_scopes=go_scopes,
        include_provisional=True,
    )
    cl_ontology = Ontology.from_obo(cl_obo_path)
    go_ontology = Ontology.from_obo(go_obo_path)
    candidate_mode = str(config.get("candidate_mode", "within_gene"))
    if candidate_mode not in {"within_gene", "all_models"}:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")
    runtime_config = dict(config)
    runtime_weights = dict(config.get("weights", {}))
    if candidate_mode == "all_models":
        runtime_weights["genetic"] = 0.0
    runtime_config["weights"] = runtime_weights

    rankings: dict[str, list[ModelScore]] = {}
    evidence_by_pair: dict[tuple[str, str], list[PhenotypeEvidence]] = {}
    warnings: list[str] = []
    if hpo_mechanisms_path and not hpo_mechanisms:
        warnings.append(
            "No eligible HPO-CPO mechanisms were loaded; current provisional claims may be excluded by config."
        )
    if mp_mechanisms_path and not mp_mechanisms:
        warnings.append("No eligible MP-CPO mechanisms were loaded.")
    unknown_model_ids = sorted(set(model_phenotypes) - set(models))
    if unknown_model_ids:
        warnings.append(f"{len(unknown_model_ids)} phenotype model IDs are absent from models.tsv.")
    if candidate_mode == "all_models":
        warnings.append("all_models mode disables genetic compatibility in the ranking score.")
    for disease_id, disease_rows in sorted(diseases.items()):
        candidates = candidates_for_disease(disease_rows, models, candidate_mode)
        if not candidates:
            warnings.append(f"{disease_id}: no candidate model for mode={candidate_mode}")
            rankings[disease_id] = []
            continue
        scored: list[ModelScore] = []
        for model in candidates:
            summary, evidence = score_disease_model(
                disease_rows,
                model,
                model_phenotypes.get(model.model_id, ()),
                surface_matches,
                hpo_mechanisms,
                mp_mechanisms,
                cl_ontology,
                go_ontology,
                runtime_config,
            )
            score = ModelScore(
                disease_id=disease_id,
                disease_name=disease_rows[0].disease_name,
                model_id=model.model_id,
                model_name=model.model_name,
                mouse_gene_symbol=model.mouse_gene_symbol,
                human_gene_symbol=model.human_gene_symbol,
                allele_id=model.allele_id,
                allele_type=model.allele_type,
                zygosity=model.zygosity,
                conditional=model.conditional,
                cre_driver=model.cre_driver,
                genetic_background=model.genetic_background,
                sex=model.sex,
                life_stage=model.life_stage,
                surface_score=float(summary["surface_score"]),
                surface_coverage=float(summary["surface_coverage"]),
                core_reproduced=int(summary["core_reproduced"]),
                core_total=int(summary["core_total"]),
                cpo_score=(float(summary["cpo_score"]) if summary["cpo_score"] is not None else None),
                cpo_coverage=float(summary["cpo_coverage"]),
                genetic_score=float(summary["genetic_score"]),
                genetic_class=str(summary["genetic_class"]),
                genetic_reason=str(summary["genetic_reason"]),
                tested_normal_count=int(summary["tested_normal_count"]),
                untested_count=int(summary["untested_count"]),
                opposite_direction_count=int(summary["opposite_direction_count"]),
                total_score=float(summary["total_score"]),
                known_model=disease_id in model.known_disease_ids,
                availability=model.availability,
            )
            scored.append(score)
            evidence_by_pair[(disease_id, model.model_id)] = evidence
            for phenotype_evidence in evidence:
                if phenotype_evidence.assay_state != "reproduced":
                    phenotype_evidence.suggested_cre_drivers = list(
                        cre_drivers.get(phenotype_evidence.best_hpo_cl_id, ())
                    )
        scored.sort(key=lambda row: (-row.total_score, -row.surface_score, row.model_id))
        for rank, row in enumerate(scored, 1):
            row.rank = rank
        rankings[disease_id] = scored

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranking_rows = [_score_dict(row) for rows in rankings.values() for row in rows]
    evidence_rows = [
        _evidence_dict(row)
        for key in sorted(evidence_by_pair)
        for row in evidence_by_pair[key]
    ]
    write_tsv(output / "model_rankings.tsv", ranking_rows, RANKING_FIELDS)
    write_tsv(output / "phenotype_evidence.tsv", evidence_rows, EVIDENCE_FIELDS)
    write_tsv(
        output / "gap_analysis.tsv",
        (
            {
                "disease_id": row.disease_id,
                "model_id": row.model_id,
                "hpo_id": row.hpo_id,
                "hpo_name": row.hpo_name,
                "gap_status": row.assay_state,
                "best_mp_id": row.best_mp_id,
                "surface_similarity": row.surface_similarity,
                "mechanism_available": row.cpo_score is not None,
                "mechanism_score": row.cpo_score,
                "suggested_cl_id": row.best_hpo_cl_id,
                "suggested_go_id": row.best_hpo_go_id,
                "suggested_cre_drivers": row.suggested_cre_drivers,
            }
            for rows in evidence_by_pair.values()
            for row in rows
            if row.assay_state != "reproduced"
        ),
        [
            "disease_id",
            "model_id",
            "hpo_id",
            "hpo_name",
            "gap_status",
            "best_mp_id",
            "surface_similarity",
            "mechanism_available",
            "mechanism_score",
            "suggested_cl_id",
            "suggested_go_id",
            "suggested_cre_drivers",
        ],
    )
    evaluation = {
        "candidate_mode": candidate_mode,
        "ablations": _ablation_metrics(rankings, runtime_config),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl_reports(output / "model_reports.jsonl", rankings, evidence_by_pair)
    write_html_report(output / "index.html", rankings, evidence_by_pair, evaluation, warnings)

    input_paths = {
        "diseases": diseases_path,
        "models": models_path,
        "model_phenotypes": model_phenotypes_path,
        "mp_hpo_mapping": mp_hpo_mapping_path,
        "config": config_path,
        "hpo_mechanisms": hpo_mechanisms_path,
        "mp_mechanisms": mp_mechanisms_path,
        "cell_catalog": cell_catalog_path,
        "go_classification": go_classification_path,
        "cl_obo": cl_obo_path,
        "go_obo": go_obo_path,
        "cre_drivers": cre_drivers_path,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_mode": candidate_mode,
        "counts": {
            "diseases": len(diseases),
            "models": len(models),
            "model_phenotype_rows": sum(len(rows) for rows in model_phenotypes.values()),
            "hpo_terms_with_surface_matches": len(surface_matches),
            "hpo_terms_with_mechanisms": len(hpo_mechanisms),
            "mp_terms_with_mechanisms": len(mp_mechanisms),
            "ranking_rows": len(ranking_rows),
            "phenotype_evidence_rows": len(evidence_rows),
        },
        "inputs": {
            key: {"path": str(value), "sha256": sha256(value)}
            for key, value in input_paths.items()
            if value is not None and Path(value).is_file()
        },
        "warnings": warnings,
        "interpretation": [
            "Missing CPO evidence is unknown and does not reduce a model score.",
            "Availability is reported but is not part of biological suitability scoring.",
            "Results are research prioritization outputs, not clinical assertions.",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
