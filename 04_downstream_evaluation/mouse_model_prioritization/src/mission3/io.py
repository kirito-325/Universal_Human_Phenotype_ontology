from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .models import DiseasePhenotype, MechanismEdge, ModelPhenotype, MouseModel, TermMatch


def read_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fields})


def _format(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def _value(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return default


def _float(value: str, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "conditional"}


def _split(value: str) -> tuple[str, ...]:
    normalized = value.replace("|", ";").replace(",", ";")
    return tuple(sorted({part.strip() for part in normalized.split(";") if part.strip()}))


def load_diseases(path: str | Path) -> dict[str, list[DiseasePhenotype]]:
    result: dict[str, list[DiseasePhenotype]] = defaultdict(list)
    for row in read_tsv(path):
        disease_id = _value(row, "disease_id", "database_id", "disease")
        hpo_id = _value(row, "hpo_id", "hp_id", "hpo")
        if not disease_id or not hpo_id:
            continue
        is_core = _bool(_value(row, "is_core", "core_phenotype"))
        weight = _float(_value(row, "weight", "hpo_weight"), 1.5 if is_core else 1.0)
        result[disease_id].append(
            DiseasePhenotype(
                disease_id=disease_id,
                disease_name=_value(row, "disease_name", "disease_label", default=disease_id),
                gene_symbol=_value(row, "gene_symbol", "human_gene_symbol", "gene").upper(),
                hpo_id=hpo_id,
                hpo_name=_value(row, "hpo_name", "hp_name", "phenotype_name"),
                weight=max(0.0, weight),
                is_core=is_core,
                frequency=_value(row, "frequency", "frequency_modifier"),
                onset=_value(row, "onset"),
                direction=normalize_direction(_value(row, "direction", "phenotype_direction")),
                inheritance=_value(row, "inheritance", "mode_of_inheritance", default="unknown"),
                molecular_mechanism=_value(
                    row, "molecular_mechanism", "mechanism", "mode_of_pathogenicity", default="unknown"
                ),
            )
        )
    return dict(result)


def load_models(path: str | Path) -> dict[str, MouseModel]:
    result: dict[str, MouseModel] = {}
    for row in read_tsv(path):
        model_id = _value(row, "model_id", "genotype_id", "genotype_accession_id")
        if not model_id:
            continue
        result[model_id] = MouseModel(
            model_id=model_id,
            model_name=_value(row, "model_name", "genotype_name", default=model_id),
            mouse_gene_symbol=_value(row, "mouse_gene_symbol", "mouse_gene", "marker_symbol"),
            human_gene_symbol=_value(row, "human_gene_symbol", "human_gene").upper(),
            allele_id=_value(row, "allele_id", "allele_accession_id"),
            allele_type=_value(row, "allele_type", "mutation_type", default="unknown"),
            zygosity=_value(row, "zygosity", default="unknown"),
            conditional=_bool(_value(row, "conditional", "is_conditional")),
            cre_driver=_value(row, "cre_driver", "recombinase_driver"),
            genetic_background=_value(row, "genetic_background", "strain_background"),
            sex=_value(row, "sex", default="not_specified"),
            life_stage=_value(row, "life_stage", "stage", "age", default="not_specified"),
            availability=_value(row, "availability", "repository_status", default="unknown"),
            known_disease_ids=_split(_value(row, "known_disease_ids", "disease_ids")),
        )
    return result


def load_model_phenotypes(path: str | Path) -> dict[str, list[ModelPhenotype]]:
    result: dict[str, list[ModelPhenotype]] = defaultdict(list)
    for row in read_tsv(path):
        model_id = _value(row, "model_id", "genotype_id", "genotype_accession_id")
        mp_id = _value(row, "mp_id", "mp_term_id", "phenotype_id")
        if not model_id or not mp_id:
            continue
        result[model_id].append(
            ModelPhenotype(
                model_id=model_id,
                mp_id=mp_id,
                mp_name=_value(row, "mp_name", "mp_term_name", "phenotype_name"),
                status=normalize_status(_value(row, "status", "phenotype_status", default="abnormal")),
                direction=normalize_direction(_value(row, "direction", "effect_direction")),
                sex=_value(row, "sex", default="not_specified"),
                life_stage=_value(row, "life_stage", "stage", "age", default="not_specified"),
                procedure_id=_value(row, "procedure_id"),
                parameter_id=_value(row, "parameter_id"),
                source=_value(row, "source", "evidence_source"),
            )
        )
    return dict(result)


def normalize_status(value: str) -> str:
    text = value.strip().lower().replace(" ", "_").replace("-", "_")
    if text in {"abnormal", "significant", "positive", "present", "observed"}:
        return "abnormal"
    if text in {"tested_normal", "normal", "not_significant", "non_significant", "no_abnormality"}:
        return "tested_normal"
    if text in {"untested", "not_tested", "unknown", "not_measured"}:
        return "untested"
    return text or "unknown"


def normalize_direction(value: str) -> str:
    text = value.strip().lower().replace(" ", "_").replace("-", "_")
    if text in {"increased", "increase", "high", "elevated", "gain", "up", "activated"}:
        return "increased"
    if text in {"decreased", "decrease", "low", "reduced", "loss", "down", "inhibited"}:
        return "decreased"
    if text in {"absent", "missing", "abolished"}:
        return "absent"
    if text in {"present", "abnormal", "bidirectional"}:
        return text
    return "unknown"


def load_surface_matches(
    path: str | Path,
    score_column: str = "semantic_score",
) -> dict[str, dict[str, TermMatch]]:
    result: dict[str, dict[str, TermMatch]] = defaultdict(dict)
    for row in read_tsv(path):
        mp_id = _value(row, "mp_id", "subject_id")
        hpo_id = _value(row, "hp_id", "hpo_id", "object_id")
        if not mp_id or not hpo_id:
            continue
        selected_column = next(
            (
                name
                for name in (score_column, "semantic_score", "bi_score", "final_score", "score")
                if row.get(name, "").strip()
            ),
            "",
        )
        score = _float(row.get(selected_column, ""), 0.0)
        if selected_column in {"final_score", "score"} and score > 1.0:
            score = score / (1.0 + score)
        score = max(0.0, min(1.0, score))
        relation = _value(row, "predicted_relation", "relation", "predicate_id")
        relation_factor = 1.0 if "exact" in relation.lower() else 0.85 if "broad" in relation.lower() else 0.7
        match = TermMatch(hpo_id=hpo_id, mp_id=mp_id, similarity=score * relation_factor, relation=relation)
        current = result[hpo_id].get(mp_id)
        if current is None or match.similarity > current.similarity:
            result[hpo_id][mp_id] = match
    return {hpo_id: dict(matches) for hpo_id, matches in result.items()}


def load_cell_catalog(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    return {
        _value(row, "cl_id", "cell_type_id")
        for row in read_tsv(path)
        if _value(row, "cl_id", "cell_type_id")
    }


def load_go_scopes(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    return {
        _value(row, "go_id", "cell_phenotype_id"): _value(row, "cell_scope", "phenotype_scope")
        for row in read_tsv(path)
        if _value(row, "go_id", "cell_phenotype_id")
    }


def load_cre_drivers(path: str | Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(path):
        cl_id = _value(row, "cl_id", "cell_type_id")
        driver = _value(row, "cre_driver", "driver", "driver_allele_symbol")
        detected = _value(row, "detected", "detection_status", default="true").lower()
        if cl_id and driver and detected not in {"false", "0", "no", "absent"}:
            result[cl_id].add(driver)
    return {cl_id: sorted(drivers) for cl_id, drivers in result.items()}


def load_mechanisms(
    path: str | Path | None,
    ontology: str,
    cell_catalog: set[str] | None = None,
    go_scopes: Mapping[str, str] | None = None,
    include_provisional: bool = False,
) -> dict[str, list[MechanismEdge]]:
    if not path:
        return {}
    catalog = cell_catalog or set()
    scopes = go_scopes or {}
    result: dict[str, list[MechanismEdge]] = defaultdict(list)
    accepted_relations = {
        "causally_upstream_of",
        "equivalent_cellular_phenotype",
        "has_constituent_cellular_phenotype",
    }
    for row in read_tsv(path):
        term_id = _value(row, "hpo_id", "hp_id") if ontology == "HPO" else _value(row, "mp_id")
        cl_id = _value(row, "selected_cl_id", "cell_type_id", "cl_id")
        go_id = _value(row, "selected_go_id", "cell_phenotype_id", "go_id")
        if not term_id or not cl_id or not go_id:
            continue
        relation = _value(row, "relation_type", default="associated_with")
        scope = _value(row, "bridge_scope", "phenotype_scope", "channel", default="unknown")
        status = _value(row, "candidate_status", "final_link_status", default="accepted").lower()
        provisional = status in {"unreviewed_extracted_claim", "candidate", "review_required"}
        if ontology == "HPO":
            if scope != "direct_hpo" or relation not in accepted_relations:
                continue
            if catalog and cl_id not in catalog:
                continue
            if scopes.get(go_id) == "not_direct_human_cell_process":
                continue
            if provisional and not include_provisional:
                continue
        confidence = _value(row, "confidence_grade", "confidence", default="unknown").lower()
        numeric = _float(_value(row, "rerank_score", "evidence_score", "final_score"), 0.7)
        confidence_factor = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(confidence, 0.7)
        scope_factor = 1.0 if scope in {"direct_hpo", "cell_type_specific", "specific"} else 0.35
        provisional_factor = 0.5 if provisional else 1.0
        weight = max(0.05, min(1.0, numeric)) * confidence_factor * scope_factor * provisional_factor
        result[term_id].append(
            MechanismEdge(
                term_id=term_id,
                cl_id=cl_id,
                go_id=go_id,
                direction=normalize_direction(_value(row, "direction")),
                relation_type=relation,
                evidence_scope=scope,
                confidence=confidence,
                evidence_weight=weight,
                source_id=_value(row, "supporting_record_ids", "source_id", "evidence_source"),
                provisional=provisional,
            )
        )
    return {term_id: edges for term_id, edges in result.items()}


def load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
