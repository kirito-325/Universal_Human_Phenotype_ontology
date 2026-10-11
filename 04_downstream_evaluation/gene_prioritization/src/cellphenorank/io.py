from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

from .schemas import DiseaseTerm, ModelPhenotype, MouseModel


csv.field_size_limit(64 * 1024 * 1024)


def open_text(path: str | Path) -> TextIO:
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8-sig", newline="")
    return source.open(encoding="utf-8-sig", newline="")


def read_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    with open_text(path) as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def tsv_columns(path: str | Path) -> list[str]:
    with open_text(path) as handle:
        return next(csv.reader(handle, delimiter="\t"), [])


def write_tsv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def format_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)}


def value(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        item = row.get(name, "")
        if item is not None and str(item).strip():
            return str(item).strip()
    return default


def as_float(item: object, default: float = 0.0) -> float:
    try:
        result = float(item)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_bool(item: object) -> bool:
    return str(item).strip().lower() in {"1", "true", "yes", "y", "core", "conditional"}


def split_values(item: str) -> tuple[str, ...]:
    normalized = item.replace("|", ";").replace(",", ";")
    return tuple(sorted({part.strip() for part in normalized.split(";") if part.strip()}))


def load_diseases(path: str | Path, disease_ids: set[str] | None = None) -> dict[str, list[DiseaseTerm]]:
    output: dict[str, list[DiseaseTerm]] = defaultdict(list)
    for row in read_tsv(path):
        disease_id = value(row, "disease_id", "database_id", "disease")
        hpo_id = value(row, "hpo_id", "hp_id", "hpo")
        if not disease_id or not hpo_id or (disease_ids and disease_id not in disease_ids):
            continue
        core = as_bool(value(row, "is_core", "core_phenotype"))
        weight = max(0.0, as_float(value(row, "weight", "hpo_weight"), 1.5 if core else 1.0))
        output[disease_id].append(
            DiseaseTerm(
                disease_id=disease_id,
                disease_name=value(row, "disease_name", "disease_label", default=disease_id),
                hpo_id=hpo_id,
                hpo_name=value(row, "hpo_name", "hp_name", "phenotype_name"),
                weight=weight,
                is_core=core,
                gene_symbol=value(row, "gene_symbol", "human_gene_symbol", "gene").upper(),
                direction=value(row, "direction", default="unknown").lower(),
            )
        )
    return dict(output)


def load_models(path: str | Path, gene_symbols: set[str] | None = None) -> dict[str, MouseModel]:
    output: dict[str, MouseModel] = {}
    normalized_genes = {gene.upper() for gene in gene_symbols or set()}
    for row in read_tsv(path):
        model_id = value(row, "model_id", "genotype_id", "genotype_accession_id")
        human_gene = value(row, "human_gene_symbol", "human_gene", "gene_symbol").upper()
        if not model_id or not human_gene or (normalized_genes and human_gene not in normalized_genes):
            continue
        output[model_id] = MouseModel(
            model_id=model_id,
            model_name=value(row, "model_name", "genotype_name", default=model_id),
            mouse_gene_symbol=value(row, "mouse_gene_symbol", "mouse_gene", "marker_symbol"),
            human_gene_symbol=human_gene,
            allele_id=value(row, "allele_id", "allele_accession_id"),
            allele_type=value(row, "allele_type", "mutation_type"),
            zygosity=value(row, "zygosity", default="unknown"),
            conditional=as_bool(value(row, "conditional", "is_conditional")),
        )
    return output


def load_model_phenotypes(path: str | Path, model_ids: set[str]) -> dict[str, list[ModelPhenotype]]:
    output: dict[str, list[ModelPhenotype]] = defaultdict(list)
    for row in read_tsv(path):
        model_id = value(row, "model_id", "genotype_id", "genotype_accession_id")
        mpo_id = value(row, "mpo_id", "mp_id", "mp_term_id", "phenotype_id")
        if model_id not in model_ids or not mpo_id:
            continue
        status = value(row, "status", "phenotype_status", default="abnormal").lower().replace(" ", "_")
        output[model_id].append(
            ModelPhenotype(
                model_id=model_id,
                mpo_id=mpo_id,
                mpo_name=value(row, "mpo_name", "mp_name", "phenotype_name"),
                status=status,
                direction=value(row, "direction", default="unknown").lower(),
            )
        )
    return dict(output)


def require_columns(path: str | Path, alternatives: Sequence[set[str]]) -> None:
    columns = set(tsv_columns(path))
    if not any(required.issubset(columns) for required in alternatives):
        rendered = " or ".join(str(sorted(item)) for item in alternatives)
        raise ValueError(f"{path} must contain {rendered}; found {sorted(columns)}")

