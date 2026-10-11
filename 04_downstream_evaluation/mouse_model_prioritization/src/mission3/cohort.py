from __future__ import annotations

import csv
import html
import json
from collections import defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .diseases import DISEASE_FIELDS
from .io import (
    load_cell_catalog,
    load_config,
    load_diseases,
    load_go_scopes,
    load_mechanisms,
    load_model_phenotypes,
    load_models,
    load_surface_matches,
    read_tsv,
    sha256,
    write_tsv,
)
from .mgi import MODEL_FIELDS, PHENOTYPE_FIELDS
from .models import DiseasePhenotype, ModelPhenotype, MouseModel


COVERAGE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "inheritance",
    "molecular_mechanism",
    "hpo_count",
    "core_hpo_count",
    "weighted_hpo_total",
    "surface_hpo_count",
    "surface_weighted_coverage",
    "hpo_cpo_count",
    "hpo_cpo_weighted_coverage",
    "candidate_model_count",
    "known_positive_model_count",
    "candidate_phenotype_rows",
    "candidate_mp_count",
    "candidate_mp_cpo_count",
    "candidate_mp_cpo_coverage",
    "comparable_cpo_hpo_count",
    "surface_ready",
    "cpo_ready",
    "genetic_ready",
    "pilot_eligible",
    "selection_score",
    "failure_reasons",
]

CURATION_FIELDS = [
    "disease_id",
    "gene_symbol",
    "inheritance",
    "molecular_mechanism",
    "core_hpo_ids",
    "exclude_hpo_ids",
    "hpo_directions",
    "include",
    "notes",
]

HPO_REVIEW_FIELDS = [
    "priority_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "hpo_id",
    "hpo_name",
    "weight",
    "is_core",
    "surface_mappable",
    "strict_mechanism_available",
    "provisional_candidate_available",
    "candidate_model_count",
    "known_positive_model_count",
    "review_action",
]


class DiseaseXrefs:
    def __init__(self) -> None:
        self._graph: dict[str, set[str]] = defaultdict(set)
        self._cache: dict[str, frozenset[str]] = {}

    def add(self, left: str, right: str) -> None:
        if not left or not right or left == right:
            return
        self._graph[left].add(right)
        self._graph[right].add(left)
        self._cache.clear()

    def expand(self, value: str) -> frozenset[str]:
        if not value:
            return frozenset()
        if value in self._cache:
            return self._cache[value]
        seen = {value}
        queue = deque([value])
        while queue:
            current = queue.popleft()
            for neighbor in self._graph.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        result = frozenset(seen)
        for item in seen:
            self._cache[item] = result
        return result


def load_disease_xrefs(path: str | Path | None) -> DiseaseXrefs:
    result = DiseaseXrefs()
    if not path:
        return result
    for row in read_tsv(path):
        predicate = next(
            (row.get(name, "").strip() for name in ("predicate_id", "predicate", "relation") if row.get(name, "").strip()),
            "",
        )
        if predicate and not any(
            token in predicate.lower() for token in ("exactmatch", "equivalent", "same_as")
        ):
            continue
        pairs = (
            ("subject_id", "object_id"),
            ("source_id", "target_id"),
            ("disease_id", "equivalent_id"),
            ("left_id", "right_id"),
        )
        for left_name, right_name in pairs:
            left = row.get(left_name, "").strip()
            right = row.get(right_name, "").strip()
            if left and right:
                result.add(left, right)
                break
    return result


def _expanded_model(model: MouseModel, xrefs: DiseaseXrefs) -> MouseModel:
    disease_ids = {
        alias
        for disease_id in model.known_disease_ids
        for alias in xrefs.expand(disease_id)
    }
    return replace(model, known_disease_ids=tuple(sorted(disease_ids)))


def _known_model(model: MouseModel, disease_id: str, xrefs: DiseaseXrefs) -> bool:
    aliases = xrefs.expand(disease_id)
    return bool(aliases.intersection(model.known_disease_ids))


def _known_text(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized not in {"", "unknown", "not_known", "unspecified", "na", "n/a"}


def _gene_tokens(value: str) -> set[str]:
    return {
        token.strip().upper()
        for token in value.replace("|", ";").replace(",", ";").split(";")
        if token.strip()
    }


def _round_robin_select(rows: Sequence[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    groups: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["selection_score"]),
            str(row["disease_id"]),
        ),
    )
    for row in ordered:
        groups[str(row["molecular_mechanism"]).strip().lower() or "unknown"].append(row)
    selected: list[dict[str, object]] = []
    group_names = sorted(groups)
    while len(selected) < maximum and any(groups.values()):
        group_names.sort(
            key=lambda name: (
                sum(str(item["molecular_mechanism"]).lower() == name for item in selected),
                name,
            )
        )
        for name in group_names:
            if groups[name] and len(selected) < maximum:
                selected.append(groups[name].popleft())
    return selected


def _write_html_report(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    selected_ids: set[str],
    curation_ids: set[str],
    manifest: Mapping[str, object],
) -> None:
    counts = manifest["counts"]
    thresholds = manifest["thresholds"]
    body_rows = []
    for row in rows:
        selected = str(row["disease_id"]) in selected_ids
        in_curation = str(row["disease_id"]) in curation_ids
        status = (
            "selected"
            if selected
            else "curation queue"
            if in_curation
            else "eligible"
            if row["pilot_eligible"]
            else "not ready"
        )
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['disease_id']))}</td>"
            f"<td>{html.escape(str(row['disease_name']))}</td>"
            f"<td>{html.escape(str(row['gene_symbol']))}</td>"
            f"<td>{html.escape(str(row['molecular_mechanism']))}</td>"
            f"<td>{row['surface_weighted_coverage']:.1%}</td>"
            f"<td>{row['hpo_cpo_count']}</td>"
            f"<td>{row['candidate_model_count']}</td>"
            f"<td>{row['known_positive_model_count']}</td>"
            f"<td>{row['candidate_mp_cpo_coverage']:.1%}</td>"
            f"<td class='{status.replace(' ', '-')}'>{status}</td>"
            f"<td>{html.escape(str(row['failure_reasons']))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 3 cohort readiness</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;color:#202124;background:#fff}}header{{padding:24px 32px;background:#17324d;color:white}}
main{{padding:24px 32px;overflow:auto}}h1{{font-size:24px;margin:0 0 8px}}p{{margin:6px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#f4f6f8}}
.selected{{color:#087f5b;font-weight:700}}.curation-queue,.eligible{{color:#7a5500;font-weight:700}}.not-ready{{color:#a61e2d;font-weight:700}}
code{{background:#f1f3f5;padding:2px 4px}}
</style></head><body><header><h1>Mission 3 cohort readiness</h1>
<p>{counts['input_diseases']} diseases audited; {counts['eligible_diseases']} eligible; {counts['selected_diseases']} selected. Pilot ready: <strong>{manifest['pilot_ready']}</strong>.</p>
</header><main><p>Thresholds: <code>{html.escape(json.dumps(thresholds, sort_keys=True))}</code></p>
<table><thead><tr><th>Disease</th><th>Name</th><th>Gene</th><th>Mechanism</th><th>Surface</th><th>HPO-CPO</th><th>Models</th><th>Gold</th><th>MP-CPO</th><th>Status</th><th>Failures</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def audit_and_select_cohort(
    *,
    diseases_path: str | Path,
    models_path: str | Path,
    model_phenotypes_path: str | Path,
    mp_hpo_mapping_path: str | Path,
    hpo_mechanisms_path: str | Path,
    mp_mechanisms_path: str | Path,
    cell_catalog_path: str | Path,
    go_classification_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    disease_xrefs_path: str | Path | None = None,
    min_diseases: int = 30,
    max_diseases: int = 50,
    min_models: int = 2,
    min_known_models: int = 1,
    min_core_hpo: int = 3,
    min_surface_coverage: float = 0.70,
    min_hpo_cpo: int = 2,
    min_comparable_cpo: int = 2,
    min_mp_cpo_coverage: float = 0.30,
) -> dict[str, object]:
    if min_diseases < 1 or max_diseases < min_diseases:
        raise ValueError("Require 1 <= min_diseases <= max_diseases")
    diseases = load_diseases(diseases_path)
    models = load_models(models_path)
    phenotypes = load_model_phenotypes(model_phenotypes_path)
    config = load_config(config_path)
    surface = load_surface_matches(
        mp_hpo_mapping_path,
        score_column=str(config.get("surface_score_column", "semantic_score")),
    )
    cell_catalog = load_cell_catalog(cell_catalog_path)
    go_scopes = load_go_scopes(go_classification_path)
    strict_hpo_mechanisms = load_mechanisms(
        hpo_mechanisms_path,
        "HPO",
        cell_catalog=cell_catalog,
        go_scopes=go_scopes,
        include_provisional=False,
    )
    provisional_hpo_mechanisms = load_mechanisms(
        hpo_mechanisms_path,
        "HPO",
        cell_catalog=cell_catalog,
        go_scopes=go_scopes,
        include_provisional=True,
    )
    hpo_mechanisms = (
        provisional_hpo_mechanisms
        if bool(config.get("include_provisional_mechanisms", False))
        else strict_hpo_mechanisms
    )
    mp_mechanisms = load_mechanisms(
        mp_mechanisms_path,
        "MPO",
        cell_catalog=cell_catalog,
        go_scopes=go_scopes,
        include_provisional=True,
    )
    if not diseases:
        raise ValueError("No valid disease profiles were loaded for cohort audit.")
    if not models:
        raise ValueError("No valid concrete mouse models were loaded for cohort audit.")
    xrefs = load_disease_xrefs(disease_xrefs_path)
    expanded_models = {model_id: _expanded_model(model, xrefs) for model_id, model in models.items()}
    models_by_gene: dict[str, list[MouseModel]] = defaultdict(list)
    for model in expanded_models.values():
        genes = _gene_tokens(model.human_gene_symbol)
        if not genes:
            genes = _gene_tokens(model.mouse_gene_symbol)
        for gene in genes:
            models_by_gene[gene].append(model)

    coverage_rows: list[dict[str, object]] = []
    candidate_ids_by_disease: dict[str, set[str]] = {}
    for disease_id, disease_rows in sorted(diseases.items()):
        total_weight = sum(row.weight for row in disease_rows) or 1.0
        surface_rows = [row for row in disease_rows if row.hpo_id in surface]
        hpo_cpo_rows = [row for row in disease_rows if row.hpo_id in hpo_mechanisms]
        disease_genes = {row.gene_symbol.upper() for row in disease_rows if row.gene_symbol}
        candidates = sorted(
            {
                model.model_id: model
                for gene in disease_genes
                for model in models_by_gene.get(gene, ())
            }.values(),
            key=lambda model: model.model_id,
        )
        candidate_ids = {model.model_id for model in candidates}
        candidate_ids_by_disease[disease_id] = candidate_ids
        candidate_phenotypes = [
            phenotype
            for model_id in candidate_ids
            for phenotype in phenotypes.get(model_id, ())
        ]
        candidate_mp_ids = {row.mp_id for row in candidate_phenotypes}
        candidate_mp_cpo_ids = candidate_mp_ids.intersection(mp_mechanisms)
        known_count = sum(_known_model(model, disease_id, xrefs) for model in candidates)
        comparable_hpo = {
            row.hpo_id
            for row in disease_rows
            if row.hpo_id in hpo_mechanisms
            and any(mp_id in candidate_mp_cpo_ids for mp_id in surface.get(row.hpo_id, {}))
        }
        surface_coverage = sum(row.weight for row in surface_rows) / total_weight
        hpo_cpo_coverage = sum(row.weight for row in hpo_cpo_rows) / total_weight
        mp_cpo_coverage = (
            len(candidate_mp_cpo_ids) / len(candidate_mp_ids) if candidate_mp_ids else 0.0
        )
        first = disease_rows[0]
        core_count = sum(row.is_core for row in disease_rows)
        failures: list[str] = []
        if core_count < min_core_hpo:
            failures.append(f"core_hpo<{min_core_hpo}")
        if len(candidates) < min_models:
            failures.append(f"models<{min_models}")
        if known_count < min_known_models:
            failures.append(f"known_models<{min_known_models}")
        if surface_coverage < min_surface_coverage:
            failures.append(f"surface_coverage<{min_surface_coverage:.2f}")
        if len(hpo_cpo_rows) < min_hpo_cpo:
            failures.append(f"hpo_cpo<{min_hpo_cpo}")
        if len(comparable_hpo) < min_comparable_cpo:
            failures.append(f"comparable_cpo<{min_comparable_cpo}")
        if mp_cpo_coverage < min_mp_cpo_coverage:
            failures.append(f"mp_cpo_coverage<{min_mp_cpo_coverage:.2f}")
        if not _known_text(first.inheritance):
            failures.append("inheritance_unknown")
        if not _known_text(first.molecular_mechanism):
            failures.append("molecular_mechanism_unknown")

        surface_ready = (
            core_count >= min_core_hpo
            and len(candidates) >= min_models
            and known_count >= min_known_models
            and surface_coverage >= min_surface_coverage
        )
        cpo_ready = (
            surface_ready
            and len(hpo_cpo_rows) >= min_hpo_cpo
            and len(comparable_hpo) >= min_comparable_cpo
            and mp_cpo_coverage >= min_mp_cpo_coverage
        )
        genetic_ready = _known_text(first.inheritance) and _known_text(first.molecular_mechanism)
        eligible = cpo_ready and genetic_ready
        score = (
            4.0 * min(1.0, surface_coverage)
            + 2.0 * min(1.0, hpo_cpo_coverage)
            + 2.0 * min(1.0, mp_cpo_coverage)
            + min(2.0, len(comparable_hpo) / max(1, min_comparable_cpo))
            + min(2.0, known_count)
        )
        coverage_rows.append(
            {
                "disease_id": disease_id,
                "disease_name": first.disease_name,
                "gene_symbol": first.gene_symbol,
                "inheritance": first.inheritance,
                "molecular_mechanism": first.molecular_mechanism,
                "hpo_count": len(disease_rows),
                "core_hpo_count": core_count,
                "weighted_hpo_total": total_weight,
                "surface_hpo_count": len(surface_rows),
                "surface_weighted_coverage": surface_coverage,
                "hpo_cpo_count": len(hpo_cpo_rows),
                "hpo_cpo_weighted_coverage": hpo_cpo_coverage,
                "candidate_model_count": len(candidates),
                "known_positive_model_count": known_count,
                "candidate_phenotype_rows": len(candidate_phenotypes),
                "candidate_mp_count": len(candidate_mp_ids),
                "candidate_mp_cpo_count": len(candidate_mp_cpo_ids),
                "candidate_mp_cpo_coverage": mp_cpo_coverage,
                "comparable_cpo_hpo_count": len(comparable_hpo),
                "surface_ready": surface_ready,
                "cpo_ready": cpo_ready,
                "genetic_ready": genetic_ready,
                "pilot_eligible": eligible,
                "selection_score": score,
                "failure_reasons": ";".join(failures),
            }
        )

    eligible_rows = [row for row in coverage_rows if row["pilot_eligible"]]
    selected_rows = _round_robin_select(eligible_rows, max_diseases)
    selected_ids = {str(row["disease_id"]) for row in selected_rows}
    eligible_ids = {str(row["disease_id"]) for row in eligible_rows}
    curation_rows = sorted(
        (row for row in coverage_rows if row["surface_ready"]),
        key=lambda row: (
            -int(row["hpo_cpo_count"]),
            -int(row["comparable_cpo_hpo_count"]),
            -float(row["candidate_mp_cpo_coverage"]),
            -float(row["selection_score"]),
            str(row["disease_id"]),
        ),
    )[:max_diseases]
    curation_ids = {str(row["disease_id"]) for row in curation_rows}
    selected_model_ids = {
        model_id
        for disease_id in selected_ids
        for model_id in candidate_ids_by_disease[disease_id]
    }
    surface_model_ids = {
        model_id
        for disease_id in curation_ids
        for model_id in candidate_ids_by_disease[disease_id]
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "disease_coverage.tsv", coverage_rows, COVERAGE_FIELDS)
    write_tsv(output / "curation_queue.tsv", curation_rows, COVERAGE_FIELDS)
    write_tsv(
        output / "disease_curation.todo.tsv",
        (
            {
                "disease_id": disease_id,
                "gene_symbol": diseases[disease_id][0].gene_symbol,
                "inheritance": diseases[disease_id][0].inheritance,
                "molecular_mechanism": "unknown",
                "core_hpo_ids": ";".join(
                    row.hpo_id
                    for row in sorted(
                        (row for row in diseases[disease_id] if row.is_core),
                        key=lambda row: (-row.weight, row.hpo_id),
                    )[:15]
                ),
                "exclude_hpo_ids": "",
                "hpo_directions": "",
                "include": "false",
                "notes": "Review inheritance, mechanism and the up-to-15 suggested core HPOs; add directions and set include=true when complete.",
            }
            for disease_id in sorted(curation_ids)
        ),
        CURATION_FIELDS,
    )
    hpo_review_rows: list[dict[str, object]] = []
    coverage_by_id = {str(row["disease_id"]): row for row in coverage_rows}
    for disease_id in curation_ids:
        coverage = coverage_by_id[disease_id]
        for disease_row in diseases[disease_id]:
            if disease_row.hpo_id in strict_hpo_mechanisms:
                continue
            provisional = disease_row.hpo_id in provisional_hpo_mechanisms
            hpo_review_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": disease_row.disease_name,
                    "gene_symbol": disease_row.gene_symbol,
                    "hpo_id": disease_row.hpo_id,
                    "hpo_name": disease_row.hpo_name,
                    "weight": disease_row.weight,
                    "is_core": disease_row.is_core,
                    "surface_mappable": disease_row.hpo_id in surface,
                    "strict_mechanism_available": False,
                    "provisional_candidate_available": provisional,
                    "candidate_model_count": coverage["candidate_model_count"],
                    "known_positive_model_count": coverage["known_positive_model_count"],
                    "review_action": (
                        "review_existing_candidate" if provisional else "targeted_literature_curation"
                    ),
                }
            )
    hpo_review_rows.sort(
        key=lambda row: (
            not bool(row["provisional_candidate_available"]),
            not bool(row["is_core"]),
            not bool(row["surface_mappable"]),
            -float(row["weight"]),
            str(row["disease_id"]),
            str(row["hpo_id"]),
        )
    )
    for rank, row in enumerate(hpo_review_rows, 1):
        row["priority_rank"] = rank
    write_tsv(output / "hpo_cpo_review_queue.tsv", hpo_review_rows, HPO_REVIEW_FIELDS)
    write_tsv(
        output / "eligible_diseases.tsv",
        (asdict(row) for disease_id in sorted(eligible_ids) for row in diseases[disease_id]),
        DISEASE_FIELDS,
    )
    write_tsv(
        output / "pilot_diseases.tsv",
        (asdict(row) for disease_id in sorted(selected_ids) for row in diseases[disease_id]),
        DISEASE_FIELDS,
    )
    write_tsv(
        output / "pilot_models.tsv",
        (asdict(expanded_models[model_id]) for model_id in sorted(selected_model_ids)),
        MODEL_FIELDS,
    )
    write_tsv(
        output / "pilot_model_phenotypes.tsv",
        (
            asdict(row)
            for model_id in sorted(selected_model_ids)
            for row in phenotypes.get(model_id, ())
        ),
        PHENOTYPE_FIELDS,
    )
    write_tsv(
        output / "surface_baseline_diseases.tsv",
        (asdict(row) for disease_id in sorted(curation_ids) for row in diseases[disease_id]),
        DISEASE_FIELDS,
    )
    write_tsv(
        output / "surface_baseline_models.tsv",
        (asdict(expanded_models[model_id]) for model_id in sorted(surface_model_ids)),
        MODEL_FIELDS,
    )
    write_tsv(
        output / "surface_baseline_model_phenotypes.tsv",
        (
            asdict(row)
            for model_id in sorted(surface_model_ids)
            for row in phenotypes.get(model_id, ())
        ),
        PHENOTYPE_FIELDS,
    )

    thresholds = {
        "min_diseases": min_diseases,
        "max_diseases": max_diseases,
        "min_models": min_models,
        "min_known_models": min_known_models,
        "min_core_hpo": min_core_hpo,
        "min_surface_coverage": min_surface_coverage,
        "min_hpo_cpo": min_hpo_cpo,
        "min_comparable_cpo": min_comparable_cpo,
        "min_mp_cpo_coverage": min_mp_cpo_coverage,
    }
    input_paths = {
        "diseases": diseases_path,
        "models": models_path,
        "model_phenotypes": model_phenotypes_path,
        "mp_hpo_mapping": mp_hpo_mapping_path,
        "hpo_mechanisms": hpo_mechanisms_path,
        "mp_mechanisms": mp_mechanisms_path,
        "cell_catalog": cell_catalog_path,
        "go_classification": go_classification_path,
        "config": config_path,
        "disease_xrefs": disease_xrefs_path,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_ready": len(selected_ids) >= min_diseases,
        "counts": {
            "input_diseases": len(diseases),
            "eligible_diseases": len(eligible_ids),
            "selected_diseases": len(selected_ids),
            "curation_queue_diseases": len(curation_ids),
            "hpo_cpo_review_queue_rows": len(hpo_review_rows),
            "selected_models": len(selected_model_ids),
            "selected_model_phenotypes": sum(
                len(phenotypes.get(model_id, ())) for model_id in selected_model_ids
            ),
            "surface_baseline_models": len(surface_model_ids),
            "surface_baseline_model_phenotypes": sum(
                len(phenotypes.get(model_id, ())) for model_id in surface_model_ids
            ),
            "hpo_terms_with_surface_matches": len(surface),
            "hpo_terms_with_loaded_mechanisms": len(hpo_mechanisms),
            "hpo_terms_with_strict_mechanisms": len(strict_hpo_mechanisms),
            "hpo_terms_with_provisional_candidates": len(provisional_hpo_mechanisms),
            "mp_terms_with_loaded_mechanisms": len(mp_mechanisms),
        },
        "thresholds": thresholds,
        "selected_disease_ids": sorted(selected_ids),
        "inputs": {
            key: {"path": str(value), "sha256": sha256(value)}
            for key, value in input_paths.items()
            if value is not None and Path(value).is_file()
        },
        "next_command": (
            "python3 -B mission3/scripts/run_pipeline.py run "
            f"--diseases {output / 'pilot_diseases.tsv'} "
            f"--models {output / 'pilot_models.tsv'} "
            f"--model-phenotypes {output / 'pilot_model_phenotypes.tsv'} "
            f"--config {config_path} --output-dir mission3/outputs/pilot"
        ),
        "surface_baseline_command": (
            "python3 -B mission3/scripts/run_pipeline.py run "
            f"--diseases {output / 'surface_baseline_diseases.tsv'} "
            f"--models {output / 'surface_baseline_models.tsv'} "
            f"--model-phenotypes {output / 'surface_baseline_model_phenotypes.tsv'} "
            "--config mission3/config/surface_baseline.json "
            "--output-dir mission3/outputs/surface_baseline"
        ),
        "interpretation": [
            "A missing HPO-CPO or MP-CPO edge is treated as unknown coverage, not negative biological evidence.",
            "pilot_ready is false unless the requested minimum cohort size passes every fixed gate.",
            "Disease xrefs expand only exact/equivalent relationships before known-model evaluation.",
            "Selection is deterministic and round-robin across molecular-mechanism groups.",
            "The curation queue is a preselection aid, not an eligible benchmark cohort.",
        ],
    }
    (output / "cohort_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_html_report(
        output / "coverage_report.html", coverage_rows, selected_ids, curation_ids, manifest
    )
    return manifest
