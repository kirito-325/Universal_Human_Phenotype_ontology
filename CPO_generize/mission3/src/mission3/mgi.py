from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .io import sha256, write_tsv


MODEL_FIELDS = [
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
    "availability",
    "known_disease_ids",
]

PHENOTYPE_FIELDS = [
    "model_id",
    "mp_id",
    "mp_name",
    "status",
    "direction",
    "sex",
    "life_stage",
    "procedure_id",
    "parameter_id",
    "source",
]


def _rows(path: str | Path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.reader(handle, delimiter="\t")


def _split(value: str) -> list[str]:
    return sorted({part.strip() for part in re.split(r"[|;,]", value) if part.strip()})


def _join(values) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def infer_zygosity(allelic_composition: str) -> str:
    text = allelic_composition.strip()
    lower = text.lower()
    if "homozyg" in lower:
        return "homozygous"
    if "heterozyg" in lower:
        return "heterozygous"
    loci = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    states: set[str] = set()

    def wild_type(value: str) -> bool:
        normalized = value.strip().lower()
        return normalized in {"+", "wild type", "wt"} or bool(
            re.search(r"<\+>$", normalized)
        )

    def alleles(value: str) -> tuple[str, str] | None:
        angle_depth = 0
        parenthesis_depth = 0
        seen_descriptor_end = False
        fallback_separators: list[int] = []
        for index, char in enumerate(value):
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
                seen_descriptor_end = seen_descriptor_end or angle_depth == 0
            elif char == "(" and angle_depth == 0:
                parenthesis_depth += 1
            elif char == ")" and angle_depth == 0:
                parenthesis_depth = max(0, parenthesis_depth - 1)
                seen_descriptor_end = seen_descriptor_end or parenthesis_depth == 0
            elif char == "/" and angle_depth == 0 and parenthesis_depth == 0:
                if seen_descriptor_end:
                    return value[:index].strip(), value[index + 1 :].strip()
                fallback_separators.append(index)
        if fallback_separators:
            index = fallback_separators[0]
            return value[:index].strip(), value[index + 1 :].strip()
        return None

    for locus in loci:
        pair = alleles(locus)
        if pair is None:
            continue
        left, right = pair
        if left.upper() == "Y" or right.upper() == "Y":
            states.add("hemizygous")
        elif wild_type(left) or wild_type(right):
            states.add("heterozygous")
        elif left == right:
            states.add("homozygous")
        else:
            states.add("compound_heterozygous")
    return states.pop() if len(states) == 1 else "complex" if states else "unknown"


def load_orthologs(path: str | Path | None) -> tuple[dict[str, str], dict[str, str]]:
    by_mgi: dict[str, str] = {}
    by_symbol: dict[str, str] = {}
    if not path:
        return by_mgi, by_symbol
    for row in _rows(path):
        if len(row) < 5 or not row[0].startswith("MGI:"):
            continue
        by_mgi[row[0]] = row[4].upper()
        by_symbol[row[1]] = row[4].upper()
    return by_mgi, by_symbol


def load_alleles(path: str | Path | None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if not path:
        return result
    for row in _rows(path):
        if len(row) < 8 or not row[0].startswith("MGI:"):
            continue
        result[row[1]] = {
            "allele_id": row[0],
            "allele_type": row[3] or "unknown",
            "allele_attribute": row[4],
            "marker_id": row[6],
            "marker_symbol": row[7],
        }
    return result


def load_disease_annotations(path: str | Path | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    if not path:
        return result
    for row in _rows(path):
        if len(row) < 10:
            continue
        genotype_id = row[9].strip()
        if not genotype_id.startswith("MGI:"):
            continue
        result[genotype_id].update(_split(row[7]))
        result[genotype_id].update(f"OMIM:{item}" if item.isdigit() else item for item in _split(row[8]))
    return result


def load_disease_mouse_models(
    path: str | Path | None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    diseases: dict[str, set[str]] = defaultdict(set)
    repositories: dict[str, set[str]] = defaultdict(set)
    if not path:
        return diseases, repositories
    for row in _rows(path):
        if len(row) < 13:
            continue
        not_model = row[2].strip().lower()
        if not_model in {"1", "true", "yes", "y", "not"}:
            continue
        disease_id = row[1].strip()
        genotype_ids = [item for item in _split(row[9]) if item.startswith("MGI:")]
        repository_ids = _split(row[8]) + _split(row[12])
        for genotype_id in genotype_ids:
            if disease_id:
                diseases[genotype_id].add(disease_id)
            repositories[genotype_id].update(repository_ids)
    return diseases, repositories


def prepare_mgi_inputs(
    *,
    pheno_geno_path: str | Path,
    output_dir: str | Path,
    orthologs_path: str | Path | None = None,
    phenotypic_alleles_path: str | Path | None = None,
    genotype_disease_path: str | Path | None = None,
    disease_mouse_models_path: str | Path | None = None,
    phenotype_sex_path: str | Path | None = None,
) -> dict[str, object]:
    ortholog_by_mgi, ortholog_by_symbol = load_orthologs(orthologs_path)
    alleles = load_alleles(phenotypic_alleles_path)
    disease_annotations = load_disease_annotations(genotype_disease_path)
    curated_diseases, model_repositories = load_disease_mouse_models(disease_mouse_models_path)
    for genotype_id, disease_ids in curated_diseases.items():
        disease_annotations[genotype_id].update(disease_ids)
    models: dict[str, dict[str, object]] = {}
    phenotype_rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    phenotype_sources: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    malformed = 0

    for row in _rows(pheno_geno_path):
        if len(row) < 7:
            malformed += 1
            continue
        composition, allele_symbols, background, mp_id, references, marker_ids, genotype_id = row[:7]
        if not genotype_id.startswith("MGI:") or not mp_id.startswith("MP:"):
            malformed += 1
            continue
        symbols = _split(allele_symbols)
        allele_records = [alleles[symbol] for symbol in symbols if symbol in alleles]
        mouse_genes = [record["marker_symbol"] for record in allele_records]
        marker_id_list = _split(marker_ids)
        human_genes = [ortholog_by_mgi[item] for item in marker_id_list if item in ortholog_by_mgi]
        human_genes.extend(
            ortholog_by_symbol[symbol] for symbol in mouse_genes if symbol in ortholog_by_symbol
        )
        allele_types = [record["allele_type"] for record in allele_records]
        attributes = [record["allele_attribute"] for record in allele_records]
        conditional = any(
            "conditional" in value.lower() or "recombinase" in value.lower()
            for value in allele_types + attributes
        )
        models[genotype_id] = {
            "model_id": genotype_id,
            "model_name": composition or allele_symbols or genotype_id,
            "mouse_gene_symbol": _join(mouse_genes),
            "human_gene_symbol": _join(human_genes),
            "allele_id": _join(record["allele_id"] for record in allele_records),
            "allele_type": _join(allele_types + attributes) or "unknown",
            "zygosity": infer_zygosity(composition),
            "conditional": conditional,
            "cre_driver": "",
            "genetic_background": background,
            "sex": "not_specified",
            "life_stage": "not_specified",
            "availability": "available" if model_repositories.get(genotype_id) else "unknown",
            "known_disease_ids": _join(disease_annotations.get(genotype_id, set())),
        }
        key = (genotype_id, mp_id, "abnormal", "not_specified")
        phenotype_rows[key] = {
            "model_id": genotype_id,
            "mp_id": mp_id,
            "mp_name": "",
            "status": "abnormal",
            "direction": "unknown",
            "sex": "not_specified",
            "life_stage": "not_specified",
            "procedure_id": "",
            "parameter_id": "",
            "source": "",
        }
        phenotype_sources[key].update(
            f"MGI_PhenoGenoMP:{reference}" for reference in _split(references)
        )

    if phenotype_sex_path:
        for row in _rows(phenotype_sex_path):
            if len(row) < 7 or not row[0].startswith("MGI:") or not row[2].startswith("MP:"):
                continue
            genotype_id, sex, mp_id, mp_name = row[:4]
            normal = row[6].strip().lower() in {"y", "yes", "true", "1"}
            status = "tested_normal" if normal else "abnormal"
            key = (genotype_id, mp_id, status, sex.lower())
            phenotype_rows[key] = {
                "model_id": genotype_id,
                "mp_id": mp_id,
                "mp_name": mp_name,
                "status": status,
                "direction": "unknown",
                "sex": sex.lower() or "not_specified",
                "life_stage": "not_specified",
                "procedure_id": "",
                "parameter_id": "",
                "source": "",
            }
            citation = row[7].strip() if len(row) > 7 else ""
            phenotype_sources[key].add(
                f"MGI_Pheno_Sex:{citation}" if citation else "MGI_Pheno_Sex"
            )

    for key, phenotype in phenotype_rows.items():
        phenotype["source"] = _join(phenotype_sources.get(key, set()))

    if not models:
        raise ValueError(
            "No genotype rows were parsed from MGI_PhenoGenoMP; verify the official seven-column format."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "models.tsv", models.values(), MODEL_FIELDS)
    write_tsv(output / "model_phenotypes.tsv", phenotype_rows.values(), PHENOTYPE_FIELDS)
    source_paths = {
        "pheno_geno": pheno_geno_path,
        "orthologs": orthologs_path,
        "phenotypic_alleles": phenotypic_alleles_path,
        "genotype_disease": genotype_disease_path,
        "disease_mouse_models": disease_mouse_models_path,
        "phenotype_sex": phenotype_sex_path,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "models": len(models),
            "model_phenotypes": len(phenotype_rows),
            "malformed_or_header_rows": malformed,
            "models_with_human_ortholog": sum(bool(row["human_gene_symbol"]) for row in models.values()),
            "models_with_allele_type": sum(row["allele_type"] != "unknown" for row in models.values()),
            "models_with_known_disease": sum(bool(row["known_disease_ids"]) for row in models.values()),
        },
        "inputs": {
            key: {"path": str(value), "sha256": sha256(value)}
            for key, value in source_paths.items()
            if value is not None and Path(value).is_file()
        },
        "limitations": [
            "MGI_PhenoGenoMP does not encode IMPC procedure/parameter coverage.",
            "Sex-specific normal annotations are imported only when MGI_Pheno_Sex is supplied.",
            "Allele type and functional attribute are combined when MGI_PhenotypicAllele is supplied.",
            "MGI_DiseaseMouseModel supplements known conditional disease models when supplied.",
            "Conditional status is inferred from allele type/attribute and should be audited.",
        ],
    }
    (output / "mgi_import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
