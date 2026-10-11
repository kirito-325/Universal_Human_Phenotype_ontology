from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from .io import sha256, write_tsv


DISEASE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "hpo_id",
    "hpo_name",
    "weight",
    "is_core",
    "frequency",
    "onset",
    "direction",
    "inheritance",
    "molecular_mechanism",
    "source_disease_id",
    "reference",
]

FREQUENCY_WEIGHTS: dict[str, tuple[float, bool]] = {
    "HP:0040280": (2.0, True),   # Obligate
    "HP:0040281": (1.75, True),  # Very frequent
    "HP:0040282": (1.3, False),  # Frequent
    "HP:0040283": (0.8, False),  # Occasional
    "HP:0040284": (0.5, False),  # Very rare
    "HP:0040285": (1.0, False),  # Variable
}


def _dict_rows(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        lines = (line for line in handle if line.strip() and not line.startswith("#"))
        yield from csv.DictReader(lines, delimiter="\t")


def _value(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return default


def _split(value: str) -> set[str]:
    return {part.strip() for part in re.split(r"[|;,]", value) if part.strip()}


def _bool(value: str, default: bool = True) -> bool:
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "include"}


def _frequency_weight(value: str) -> tuple[float, bool]:
    clean = value.strip()
    if clean in FREQUENCY_WEIGHTS:
        return FREQUENCY_WEIGHTS[clean]
    match = re.search(r"(?:(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*%)", clean)
    if match:
        if match.group(1) and match.group(2) and float(match.group(2)) > 0:
            proportion = float(match.group(1)) / float(match.group(2))
        else:
            proportion = float(match.group(3)) / 100.0
        proportion = max(0.0, min(1.0, proportion))
        return 0.5 + 1.5 * proportion, proportion >= 0.8
    return 1.0, False


def _load_curation(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in _dict_rows(path):
        disease_id = _value(row, "disease_id", "database_id")
        if disease_id:
            result[disease_id] = row
    return result


def _load_hpo_names(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    result: dict[str, str] = {}
    current_id = ""
    current_name = ""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            clean = line.rstrip("\n")
            if clean == "[Term]":
                if current_id.startswith("HP:") and current_name:
                    result[current_id] = current_name
                current_id = ""
                current_name = ""
            elif clean.startswith("id: "):
                current_id = clean[4:].strip()
            elif clean.startswith("name: "):
                current_name = clean[6:].strip()
    if current_id.startswith("HP:") and current_name:
        result[current_id] = current_name
    return result


def prepare_disease_profiles(
    *,
    hpoa_path: str | Path,
    gene_disease_path: str | Path,
    output_dir: str | Path,
    curation_path: str | Path | None = None,
    hpo_obo_path: str | Path | None = None,
) -> dict[str, object]:
    curation = _load_curation(curation_path)
    hpo_names = _load_hpo_names(hpo_obo_path)
    genes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in _dict_rows(gene_disease_path):
        disease_id = _value(row, "disease_id", "database_id")
        gene = _value(row, "gene_symbol", "symbol", "gene").upper()
        association = _value(row, "association_type", default="UNKNOWN").upper()
        if disease_id and gene and gene != "-":
            genes[disease_id][gene].add(association)

    annotations: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    inheritance_by_disease: dict[str, set[str]] = defaultdict(set)
    disease_names: dict[str, str] = {}
    skipped_negated = 0
    for row in _dict_rows(hpoa_path):
        disease_id = _value(row, "database_id", "disease_id")
        hpo_id = _value(row, "hpo_id", "HPO_ID", "hpo")
        qualifier = _value(row, "qualifier").upper()
        aspect = _value(row, "aspect")
        if not disease_id or not hpo_id.startswith("HP:"):
            continue
        if "NOT" in qualifier:
            skipped_negated += 1
            continue
        disease_names[disease_id] = _value(
            row, "disease_name", "disease_label", default=disease_id
        )
        if aspect == "I":
            inheritance_by_disease[disease_id].add(hpo_names.get(hpo_id, hpo_id))
            continue
        if aspect and aspect != "P":
            continue
        frequency = _value(row, "frequency")
        weight, inferred_core = _frequency_weight(frequency)
        candidate = {
            "hpo_id": hpo_id,
            "hpo_name": _value(
                row, "hpo_name", "phenotype_name", default=hpo_names.get(hpo_id, "")
            ),
            "frequency": frequency,
            "onset": _value(row, "onset"),
            "reference": _value(row, "reference"),
            "weight": str(weight),
            "is_core": "true" if inferred_core else "false",
        }
        current = annotations[disease_id].get(hpo_id)
        if current is None or float(candidate["weight"]) > float(current["weight"]):
            annotations[disease_id][hpo_id] = candidate

    output_rows: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    emitted_diseases = 0
    for disease_id in sorted(annotations):
        curated = curation.get(disease_id, {})
        if curated and not _bool(_value(curated, "include"), True):
            skipped[disease_id] = "excluded_by_curation"
            continue
        curated_gene = _value(curated, "gene_symbol", "gene").upper()
        mendelian = sorted(
            gene
            for gene, types in genes.get(disease_id, {}).items()
            if "MENDELIAN" in types
        )
        all_genes = sorted(genes.get(disease_id, {}))
        if curated_gene:
            gene = curated_gene
        elif len(mendelian) == 1:
            gene = mendelian[0]
        elif not mendelian and len(all_genes) == 1:
            gene = all_genes[0]
        else:
            skipped[disease_id] = "ambiguous_or_missing_gene;add_disease_curation"
            continue

        core_ids = _split(_value(curated, "core_hpo_ids"))
        excluded_ids = _split(_value(curated, "exclude_hpo_ids"))
        inheritance = _value(
            curated,
            "inheritance",
            default=";".join(sorted(inheritance_by_disease.get(disease_id, ()))) or "unknown",
        )
        mechanism = _value(curated, "molecular_mechanism", "mechanism", default="unknown")
        direction_by_hpo: dict[str, str] = {}
        for assignment in _split(_value(curated, "hpo_directions")):
            hpo_id, separator, direction = assignment.partition("=")
            if separator and hpo_id.startswith("HP:"):
                direction_by_hpo[hpo_id] = direction

        disease_rows = 0
        for hpo_id, annotation in sorted(annotations[disease_id].items()):
            if hpo_id in excluded_ids:
                continue
            is_core = hpo_id in core_ids if core_ids else annotation["is_core"] == "true"
            output_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": disease_names.get(disease_id, disease_id),
                    "gene_symbol": gene,
                    "hpo_id": hpo_id,
                    "hpo_name": annotation["hpo_name"],
                    "weight": max(float(annotation["weight"]), 1.5) if is_core else annotation["weight"],
                    "is_core": is_core,
                    "frequency": annotation["frequency"],
                    "onset": annotation["onset"],
                    "direction": direction_by_hpo.get(hpo_id, "unknown"),
                    "inheritance": inheritance,
                    "molecular_mechanism": mechanism,
                    "source_disease_id": disease_id,
                    "reference": annotation["reference"],
                }
            )
            disease_rows += 1
        if disease_rows:
            emitted_diseases += 1

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "diseases.all.tsv", output_rows, DISEASE_FIELDS)
    inputs = {
        "hpoa": hpoa_path,
        "gene_disease": gene_disease_path,
        "curation": curation_path,
        "hpo_obo": hpo_obo_path,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "hpoa_diseases": len(annotations),
            "gene_disease_entries": len(genes),
            "curated_diseases": len(curation),
            "hpo_term_names": len(hpo_names),
            "diseases_with_hpo_inheritance": sum(
                bool(inheritance_by_disease.get(disease_id)) for disease_id in annotations
            ),
            "emitted_diseases": emitted_diseases,
            "emitted_phenotype_rows": len(output_rows),
            "skipped_diseases": len(skipped),
            "skipped_negated_annotations": skipped_negated,
        },
        "skipped_diseases": skipped,
        "inputs": {
            key: {"path": str(value), "sha256": sha256(value)}
            for key, value in inputs.items()
            if value is not None and Path(value).is_file()
        },
        "interpretation": [
            "Negated disease-HPO annotations are excluded from positive disease profiles.",
            "Diseases with ambiguous genes require an explicit curation row.",
            "HPO aspect-I annotations initialize inheritance; reviewed curation overrides them.",
            "Unknown inheritance and molecular mechanism are retained for coverage audit but fail the genetic-readiness gate.",
        ],
    }
    (output / "disease_profile_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
