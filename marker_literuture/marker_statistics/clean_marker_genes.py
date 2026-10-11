#!/usr/bin/env python3
"""Validate marker genes as human symbols and humanize supported mouse genes.

The cleaner never modifies extraction or agent-standardization outputs. It writes
separate, auditable files that are the only marker inputs consumed downstream.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS = ROOT / "database" / "papers"
DEFAULT_REFERENCE_DIR = ROOT / "database" / "reference" / "genes"
DEFAULT_RUN_SUMMARY = ROOT / "database" / "marker_gene_cleaning_summary.json"
HGNC_NAME = "hgnc_complete_set.txt"
MGI_MARKERS_NAME = "MRK_List2.rpt.gz"
MGI_GENE_MODELS_NAME = "MGI_Gene_Model_Coord.rpt"
MGI_HOMOLOGY_NAME = "HOM_MouseHumanSequence.rpt"
STATISTICS_INPUT = "marker_statistics_filtered.csv"
SIGNATURES_INPUT = "marker_signatures.csv"
STATISTICS_OUTPUT = "marker_statistics_cleaned.csv"
SIGNATURES_OUTPUT = "marker_signatures_cleaned.tsv"
AUDIT_OUTPUT = "marker_gene_cleaning_audit.tsv"
SUMMARY_OUTPUT = "marker_gene_cleaning_summary.json"

REFERENCE_SOURCES = {
    HGNC_NAME: "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    MGI_MARKERS_NAME: "https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt.gz",
    MGI_GENE_MODELS_NAME: "https://www.informatics.jax.org/downloads/reports/MGI_Gene_Model_Coord.rpt",
    MGI_HOMOLOGY_NAME: "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
}

SPECIES_PATTERNS = {
    "human": re.compile(r"\b(?:human|humans|homo\s+sapiens|patient|patients)\b", re.I),
    "mouse": re.compile(r"\b(?:mouse|mice|murine|mus\s+musculus)\b", re.I),
    "other": re.compile(
        r"\b(?:rat|rats|zebrafish|danio\s+rerio|macaque|monkey|porcine|pig|"
        r"bovine|cow|drosophila|fruit\s+fly|xenopus|chicken|canine|dog)\b",
        re.I,
    ),
}
ENSEMBL_HUMAN_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.I)
ENSEMBL_MOUSE_RE = re.compile(r"^ENSMUSG\d+(?:\.\d+)?$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def split_pipe(value: Any) -> Iterator[str]:
    for item in clean_text(value).split("|"):
        item = item.strip()
        if item:
            yield item


def unique_values(values: dict[str, set[str]]) -> tuple[dict[str, str], set[str]]:
    unique: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, candidates in values.items():
        if len(candidates) == 1:
            unique[key] = next(iter(candidates))
        elif candidates:
            ambiguous.add(key)
    return unique, ambiguous


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OrthologTarget:
    human_symbol: str
    hgnc_id: str
    homology_class: str
    cardinality: str


@dataclass
class GeneReferences:
    human_ids: dict[str, str]
    human_folded: dict[str, str]
    human_aliases: dict[str, str]
    human_previous: dict[str, str]
    human_ensembl: dict[str, str]
    mouse_symbols: set[str]
    mouse_folded: dict[str, str]
    mouse_aliases: dict[str, str]
    mouse_ensembl: dict[str, str]
    mouse_mgi_ids: dict[str, str]
    mouse_to_human: dict[str, OrthologTarget]
    ambiguous_human_aliases: set[str]
    ambiguous_mouse_aliases: set[str]
    ambiguous_mouse_orthologs: set[str]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class GeneDecision:
    source_gene: str
    gene: str
    source_species: str
    status: str
    method: str
    reason: str
    reference: str

    @property
    def accepted(self) -> bool:
        return self.status in {"kept", "mapped"}


def load_hgnc(path: Path) -> tuple[
    dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], set[str]
]:
    human_ids: dict[str, str] = {}
    folded: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    previous: dict[str, set[str]] = defaultdict(set)
    ensembl: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"hgnc_id", "symbol", "status", "alias_symbol", "prev_symbol", "ensembl_gene_id"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid HGNC complete-set columns: {path}")
        for row in reader:
            if clean_text(row.get("status")) != "Approved":
                continue
            symbol = clean_text(row.get("symbol"))
            hgnc_id = clean_text(row.get("hgnc_id"))
            if not symbol or not hgnc_id:
                continue
            human_ids[symbol] = hgnc_id
            folded[symbol.casefold()].add(symbol)
            for alias in split_pipe(row.get("alias_symbol")):
                if alias != symbol:
                    aliases[alias].add(symbol)
            for old_symbol in split_pipe(row.get("prev_symbol")):
                if old_symbol != symbol:
                    previous[old_symbol].add(symbol)
            for identifier in split_pipe(row.get("ensembl_gene_id")):
                ensembl[identifier.split(".", 1)[0].upper()].add(symbol)
    human_folded, _ = unique_values(folded)
    human_aliases, ambiguous_aliases = unique_values(aliases)
    human_previous, ambiguous_previous = unique_values(previous)
    human_ensembl, _ = unique_values(ensembl)
    return (
        human_ids,
        human_folded,
        human_aliases,
        human_previous,
        human_ensembl,
        ambiguous_aliases | ambiguous_previous,
    )


def load_homology(path: Path, human_ids: dict[str, str]) -> tuple[
    dict[str, OrthologTarget], set[str]
]:
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"mouse": set(), "human": set()})
    human_group_ids: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"DB Class Key", "NCBI Taxon ID", "Symbol", "HGNC ID"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid MGI homology columns: {path}")
        for row in reader:
            class_id = clean_text(row.get("DB Class Key"))
            taxon_id = clean_text(row.get("NCBI Taxon ID"))
            symbol = clean_text(row.get("Symbol"))
            if not class_id or not symbol:
                continue
            if taxon_id == "10090":
                groups[class_id]["mouse"].add(symbol)
            elif taxon_id == "9606":
                groups[class_id]["human"].add(symbol)
                human_group_ids[(class_id, symbol)] = clean_text(row.get("HGNC ID"))

    candidates: dict[str, set[OrthologTarget]] = defaultdict(set)
    for class_id, species in groups.items():
        mice = species["mouse"]
        humans = species["human"]
        if not mice or len(humans) != 1:
            continue
        human = next(iter(humans))
        if human not in human_ids:
            continue
        cardinality = f"{len(mice)}m:1h"
        target = OrthologTarget(
            human_symbol=human,
            hgnc_id=human_group_ids.get((class_id, human)) or human_ids[human],
            homology_class=class_id,
            cardinality=cardinality,
        )
        for mouse in mice:
            candidates[mouse].add(target)

    mappings: dict[str, OrthologTarget] = {}
    ambiguous: set[str] = set()
    for mouse, targets in candidates.items():
        human_targets = {target.human_symbol for target in targets}
        if len(human_targets) == 1:
            mappings[mouse] = sorted(targets, key=lambda item: item.homology_class)[0]
        else:
            ambiguous.add(mouse)
    return mappings, ambiguous


def load_mouse_markers(
    path: Path,
    mouse_to_human: dict[str, OrthologTarget],
) -> tuple[set[str], dict[str, str], dict[str, str], set[str], dict[str, str]]:
    symbols: set[str] = set()
    folded: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    mgi_ids: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"MGI Accession ID", "Marker Symbol", "Marker Type", "Marker Synonyms (pipe-separated)"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid MGI marker columns: {path}")
        for row in reader:
            if clean_text(row.get("Marker Type")) != "Gene":
                continue
            symbol = clean_text(row.get("Marker Symbol"))
            if not symbol:
                continue
            symbols.add(symbol)
            folded[symbol.casefold()].add(symbol)
            mgi_id = clean_text(row.get("MGI Accession ID"))
            if mgi_id:
                mgi_ids[mgi_id.upper()] = symbol
            if symbol not in mouse_to_human:
                continue
            for alias in split_pipe(row.get("Marker Synonyms (pipe-separated)")):
                if alias != symbol:
                    aliases[alias].add(symbol)
    mouse_folded, _ = unique_values(folded)
    mouse_aliases, ambiguous_aliases = unique_values(aliases)
    return symbols, mouse_folded, mouse_aliases, ambiguous_aliases, mgi_ids


def load_mouse_ensembl(path: Path) -> dict[str, str]:
    values: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        if len(header) < 11 or not header[0].startswith("1. MGI accession"):
            raise ValueError(f"invalid MGI gene-model columns: {path}")
        for row in reader:
            if len(row) < 11 or clean_text(row[1]) != "Gene":
                continue
            symbol = clean_text(row[2])
            identifier = clean_text(row[10]).split(".", 1)[0].upper()
            if symbol and identifier:
                values[identifier].add(symbol)
    unique, _ = unique_values(values)
    return unique


def load_references(reference_dir: Path) -> GeneReferences:
    paths = {name: reference_dir / name for name in REFERENCE_SOURCES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("missing gene reference files: " + ", ".join(missing))
    (
        human_ids,
        human_folded,
        human_aliases,
        human_previous,
        human_ensembl,
        ambiguous_human_aliases,
    ) = load_hgnc(paths[HGNC_NAME])
    mouse_to_human, ambiguous_mouse_orthologs = load_homology(
        paths[MGI_HOMOLOGY_NAME], human_ids
    )
    (
        mouse_symbols,
        mouse_folded,
        mouse_aliases,
        ambiguous_mouse_aliases,
        mouse_mgi_ids,
    ) = load_mouse_markers(paths[MGI_MARKERS_NAME], mouse_to_human)
    mouse_ensembl = load_mouse_ensembl(paths[MGI_GENE_MODELS_NAME])
    manifest = {
        "loaded_at": utc_now(),
        "files": {
            name: {
                "path": str(path.resolve()),
                "url": REFERENCE_SOURCES[name],
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "counts": {
            "hgnc_approved_symbols": len(human_ids),
            "hgnc_unique_aliases": len(human_aliases),
            "hgnc_unique_previous_symbols": len(human_previous),
            "mgi_gene_symbols": len(mouse_symbols),
            "mgi_unique_aliases_with_human_target": len(mouse_aliases),
            "mouse_genes_with_unambiguous_human_target": len(mouse_to_human),
            "mouse_genes_with_ambiguous_human_target": len(ambiguous_mouse_orthologs),
        },
    }
    return GeneReferences(
        human_ids=human_ids,
        human_folded=human_folded,
        human_aliases=human_aliases,
        human_previous=human_previous,
        human_ensembl=human_ensembl,
        mouse_symbols=mouse_symbols,
        mouse_folded=mouse_folded,
        mouse_aliases=mouse_aliases,
        mouse_ensembl=mouse_ensembl,
        mouse_mgi_ids=mouse_mgi_ids,
        mouse_to_human=mouse_to_human,
        ambiguous_human_aliases=ambiguous_human_aliases,
        ambiguous_mouse_aliases=ambiguous_mouse_aliases,
        ambiguous_mouse_orthologs=ambiguous_mouse_orthologs,
        manifest=manifest,
    )


def species_tokens(value: str) -> set[str]:
    return {name for name, pattern in SPECIES_PATTERNS.items() if pattern.search(value)}


def infer_paper_species(record_dir: Path) -> tuple[str, str]:
    analysis = read_json(record_dir / "agent_standardized" / "analysis.json")
    context = analysis.get("paper_context") or {}
    raw_values: list[str] = []
    if isinstance(context, dict):
        direct = clean_text(context.get("species"))
        if direct:
            raw_values.append(direct)
        for batch in context.get("batch_contexts", []) or []:
            if isinstance(batch, dict):
                value = clean_text(batch.get("species"))
                if value:
                    raw_values.append(value)
    tokens: set[str] = set()
    for value in raw_values:
        tokens.update(species_tokens(value))

    evidence_source = "analysis.json"
    if not tokens:
        metadata = read_json(record_dir / "metadata.json")
        metadata_text = " ".join(
            clean_text(metadata.get(key)) for key in ("title", "abstract", "keywords")
        )
        tokens.update(species_tokens(metadata_text))
        evidence_source = "metadata_fallback"

    if tokens == {"human"}:
        species = "human"
    elif tokens == {"mouse"}:
        species = "mouse"
    elif not tokens:
        species = "unknown"
    elif tokens == {"other"}:
        species = "other"
    else:
        species = "mixed"
    evidence_values = " | ".join(dict.fromkeys(raw_values))
    evidence = f"{evidence_source}: {evidence_values}" if evidence_values else evidence_source
    return species, evidence[:1000]


def human_decision(
    source: str, symbol: str, species: str, method: str, references: GeneReferences
) -> GeneDecision:
    status = "kept" if source == symbol else "mapped"
    reason = "approved_human_symbol" if status == "kept" else "normalized_to_hgnc_approved_symbol"
    return GeneDecision(
        source, symbol, species, status, method, reason,
        references.human_ids.get(symbol, ""),
    )


def mouse_decision(
    source: str, mouse_symbol: str, species: str, method: str, references: GeneReferences
) -> GeneDecision:
    target = references.mouse_to_human.get(mouse_symbol)
    if target is None:
        reason = (
            "ambiguous_human_ortholog"
            if mouse_symbol in references.ambiguous_mouse_orthologs
            else "no_unambiguous_human_ortholog"
        )
        return GeneDecision(source, "", species, "rejected", method, reason, mouse_symbol)
    reference = (
        f"MGI-HOM:{target.homology_class};{target.cardinality};{target.hgnc_id}"
    )
    return GeneDecision(
        source,
        target.human_symbol,
        species,
        "mapped",
        method,
        "mouse_human_ortholog",
        reference,
    )


def resolve_human(source: str, references: GeneReferences, include_casefold: bool) -> tuple[str, str] | None:
    if source in references.human_ids:
        return source, "hgnc_approved_exact"
    identifier = source.split(".", 1)[0].upper()
    if ENSEMBL_HUMAN_RE.match(source) and identifier in references.human_ensembl:
        return references.human_ensembl[identifier], "hgnc_ensembl_id"
    if source in references.human_previous:
        return references.human_previous[source], "hgnc_previous_symbol"
    if source in references.human_aliases:
        return references.human_aliases[source], "hgnc_alias_symbol"
    if include_casefold:
        symbol = references.human_folded.get(source.casefold())
        if symbol:
            return symbol, "hgnc_case_normalized"
    return None


def resolve_mouse(source: str, references: GeneReferences, include_casefold: bool) -> tuple[str, str] | None:
    if source in references.mouse_symbols:
        return source, "mgi_symbol_exact"
    upper = source.upper()
    if upper.startswith("MGI:") and upper in references.mouse_mgi_ids:
        return references.mouse_mgi_ids[upper], "mgi_accession_id"
    identifier = source.split(".", 1)[0].upper()
    if ENSEMBL_MOUSE_RE.match(source) and identifier in references.mouse_ensembl:
        return references.mouse_ensembl[identifier], "mgi_ensembl_id"
    if source in references.mouse_aliases:
        return references.mouse_aliases[source], "mgi_alias_symbol"
    if include_casefold:
        symbol = references.mouse_folded.get(source.casefold())
        if symbol:
            return symbol, "mgi_case_normalized"
    return None


def decide_gene(source_gene: Any, species: str, references: GeneReferences) -> GeneDecision:
    source = clean_text(source_gene)
    if not source:
        return GeneDecision(source, "", species, "rejected", "none", "empty_gene", "")
    if any(character in source for character in ("\n", "\r", "\t")):
        return GeneDecision(source, "", species, "rejected", "none", "malformed_gene", "")
    if species == "other":
        return GeneDecision(
            source, "", species, "rejected", "none", "unsupported_nonhuman_species", ""
        )

    if species == "mouse":
        mouse = resolve_mouse(source, references, include_casefold=True)
        if mouse:
            return mouse_decision(source, mouse[0], species, mouse[1], references)
        if source in references.human_ids:
            return GeneDecision(
                source, "", species, "rejected", "hgnc_approved_exact",
                "human_symbol_in_mouse_paper_without_mouse_match",
                references.human_ids[source],
            )
        return GeneDecision(source, "", species, "rejected", "none", "unrecognized_gene", "")

    if species == "human":
        human = resolve_human(source, references, include_casefold=True)
        if human:
            return human_decision(source, human[0], species, human[1], references)
        mouse = resolve_mouse(source, references, include_casefold=False)
        if mouse:
            return mouse_decision(source, mouse[0], species, mouse[1], references)
        return GeneDecision(source, "", species, "rejected", "none", "unrecognized_gene", "")

    human = resolve_human(source, references, include_casefold=False)
    if human:
        return human_decision(source, human[0], species, human[1], references)
    mouse = resolve_mouse(source, references, include_casefold=False)
    if mouse:
        return mouse_decision(source, mouse[0], species, mouse[1], references)

    human_folded = references.human_folded.get(source.casefold())
    mouse_folded = references.mouse_folded.get(source.casefold())
    if human_folded and mouse_folded:
        target = references.mouse_to_human.get(mouse_folded)
        if target and target.human_symbol == human_folded:
            return human_decision(
                source, human_folded, species, "hgnc_mgi_case_consensus", references
            )
        return GeneDecision(
            source, "", species, "rejected", "case_insensitive_lookup",
            "ambiguous_species_or_gene", "",
        )
    return GeneDecision(source, "", species, "rejected", "none", "unrecognized_gene", "")


def open_atomic_delimited(
    path: Path, fieldnames: list[str], delimiter: str
) -> tuple[TextIO, csv.DictWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    writer = csv.DictWriter(
        handle, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    return handle, writer, Path(handle.name)


PROVENANCE_FIELDS = [
    "source_gene",
    "source_species",
    "species_evidence",
    "gene_mapping_status",
    "gene_mapping_method",
    "gene_mapping_reference",
]
SIGNATURE_FIELDS = [
    "cell_cluster",
    "gene",
    "source_gene",
    "source_species",
    "species_evidence",
    "gene_mapping_status",
    "gene_mapping_method",
    "gene_mapping_reference",
]
AUDIT_FIELDS = [
    "table_type",
    "source_gene",
    "gene",
    "source_species",
    "status",
    "mapping_method",
    "reason",
    "mapping_reference",
    "occurrences",
]


def decision_values(decision: GeneDecision, species_evidence: str) -> dict[str, str]:
    return {
        "gene": decision.gene,
        "source_gene": decision.source_gene,
        "source_species": decision.source_species,
        "species_evidence": species_evidence,
        "gene_mapping_status": decision.status,
        "gene_mapping_method": decision.method,
        "gene_mapping_reference": decision.reference,
    }


def add_audit(
    audit: Counter[tuple[str, ...]], table_type: str, decision: GeneDecision
) -> None:
    audit[
        (
            table_type,
            decision.source_gene,
            decision.gene,
            decision.source_species,
            decision.status,
            decision.method,
            decision.reason,
            decision.reference,
        )
    ] += 1


def process_statistics(
    source: Path,
    output: Path,
    species: str,
    species_evidence: str,
    references: GeneReferences,
    audit: Counter[tuple[str, ...]],
) -> Counter[str]:
    metrics: Counter[str] = Counter()
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = list(reader.fieldnames or [])
        if not {"cell_cluster", "gene"} <= set(fieldnames):
            raise ValueError(f"statistics input lacks cell_cluster/gene: {source}")
        output_fields = fieldnames + [field for field in PROVENANCE_FIELDS if field not in fieldnames]
        handle, writer, temporary = open_atomic_delimited(output, output_fields, ",")
        try:
            for row in reader:
                metrics["input_rows"] += 1
                decision = decide_gene(row.get("gene"), species, references)
                add_audit(audit, "statistics", decision)
                metrics[f"status_{decision.status}"] += 1
                metrics[f"method_{decision.method}"] += 1
                if not decision.accepted:
                    metrics[f"rejected_{decision.reason}"] += 1
                    continue
                writer.writerow({**row, **decision_values(decision, species_evidence)})
                metrics["output_rows"] += 1
        finally:
            handle.close()
        temporary.replace(output)
        output.chmod(0o644)
    return metrics


def process_signatures(
    source: Path,
    output: Path,
    species: str,
    species_evidence: str,
    references: GeneReferences,
    audit: Counter[tuple[str, ...]],
) -> Counter[str]:
    metrics: Counter[str] = Counter()
    handle, writer, temporary = open_atomic_delimited(output, SIGNATURE_FIELDS, "\t")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.reader(input_handle)
            header = next(reader, [])
            columns = [(index, clean_text(name)) for index, name in enumerate(header) if clean_text(name)]
            for row in reader:
                for index, cell_name in columns:
                    if index >= len(row) or not clean_text(row[index]):
                        continue
                    metrics["input_rows"] += 1
                    decision = decide_gene(row[index], species, references)
                    add_audit(audit, "signatures", decision)
                    metrics[f"status_{decision.status}"] += 1
                    metrics[f"method_{decision.method}"] += 1
                    if not decision.accepted:
                        metrics[f"rejected_{decision.reason}"] += 1
                        continue
                    writer.writerow(
                        {
                            "cell_cluster": cell_name,
                            **decision_values(decision, species_evidence),
                        }
                    )
                    metrics["output_rows"] += 1
    finally:
        handle.close()
    temporary.replace(output)
    output.chmod(0o644)
    return metrics


def write_audit(path: Path, audit: Counter[tuple[str, ...]]) -> None:
    handle, writer, temporary = open_atomic_delimited(path, AUDIT_FIELDS, "\t")
    try:
        for key in sorted(audit, key=lambda item: (item[0], item[1].casefold(), item)):
            writer.writerow(dict(zip(AUDIT_FIELDS[:-1], key)) | {"occurrences": audit[key]})
    finally:
        handle.close()
    temporary.replace(path)
    path.chmod(0o644)


def process_record(
    record_dir: Path,
    references: GeneReferences,
    skip_existing: bool = False,
) -> dict[str, Any]:
    output_dir = record_dir / "agent_standardized"
    summary_path = output_dir / SUMMARY_OUTPUT
    if skip_existing and summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("status") == "complete":
            return {**summary, "status": "skipped_existing"}

    species, species_evidence = infer_paper_species(record_dir)
    audit: Counter[tuple[str, ...]] = Counter()
    table_results: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    inputs = (
        (
            "statistics",
            output_dir / STATISTICS_INPUT,
            output_dir / STATISTICS_OUTPUT,
            process_statistics,
        ),
        (
            "signatures",
            output_dir / SIGNATURES_INPUT,
            output_dir / SIGNATURES_OUTPUT,
            process_signatures,
        ),
    )
    for table_type, source, output, processor in inputs:
        if not source.is_file():
            table_results[table_type] = {"missing_input": 1}
            continue
        try:
            metrics = processor(
                source, output, species, species_evidence, references, audit
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            table_results[table_type] = {"failed": 1}
            continue
        table_results[table_type] = dict(metrics)

    write_audit(output_dir / AUDIT_OUTPUT, audit)
    metadata = read_json(record_dir / "metadata.json")
    summary = {
        "generated_at": utc_now(),
        "status": "failed" if errors else "complete",
        "paper_id": clean_text(metadata.get("id")) or record_dir.name,
        "paper_species": species,
        "species_evidence": species_evidence,
        "tables": table_results,
        "audit_unique_decisions": len(audit),
        "outputs": {
            "statistics": str((output_dir / STATISTICS_OUTPUT).relative_to(record_dir)),
            "signatures": str((output_dir / SIGNATURES_OUTPUT).relative_to(record_dir)),
            "audit": str((output_dir / AUDIT_OUTPUT).relative_to(record_dir)),
        },
        "errors": errors,
    }
    write_json_atomic(summary_path, summary)
    return summary


def iter_record_dirs(papers_dir: Path) -> Iterator[Path]:
    for record_dir in sorted(papers_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        output_dir = record_dir / "agent_standardized"
        if (output_dir / STATISTICS_INPUT).is_file() or (output_dir / SIGNATURES_INPUT).is_file():
            yield record_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="process at most N papers; 0 means all")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.papers_dir.is_dir():
        raise SystemExit(f"papers directory does not exist: {args.papers_dir}")
    try:
        references = load_references(args.reference_dir.resolve())
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    records = list(iter_record_dirs(args.papers_dir.resolve()))
    if args.limit > 0:
        records = records[: args.limit]

    totals: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, record_dir in enumerate(records, 1):
        summary = process_record(record_dir, references, skip_existing=args.skip_existing)
        summaries.append(summary)
        species_counts[clean_text(summary.get("paper_species")) or "unknown"] += 1
        totals["papers"] += 1
        totals[f"status_{summary.get('status', 'unknown')}"] += 1
        for table_type, table in (summary.get("tables") or {}).items():
            for key, value in table.items():
                totals[f"{table_type}_{key}"] += int(value)
        if summary.get("errors"):
            failures.append(
                {"paper_id": summary.get("paper_id"), "errors": summary["errors"]}
            )
        if index % 25 == 0 or index == len(records):
            print(
                f"[{index}/{len(records)}] cleaned; accepted="
                f"{totals['statistics_output_rows'] + totals['signatures_output_rows']:,}; "
                f"failures={len(failures)}",
                flush=True,
            )

    run_summary = {
        "generated_at": utc_now(),
        "policy": {
            "target_species": "human",
            "human_symbols": "HGNC approved symbols",
            "mouse_conversion": (
                "MGI homology class accepted only when the class has one human target"
            ),
            "unsupported_nonhuman_species": "rejected",
            "original_outputs_modified": False,
            "statistics_input": STATISTICS_INPUT,
            "signatures_input": SIGNATURES_INPUT,
            "statistics_output": STATISTICS_OUTPUT,
            "signatures_output": SIGNATURES_OUTPUT,
        },
        "references": references.manifest,
        "totals": dict(totals),
        "paper_species": dict(species_counts.most_common()),
        "failure_count": len(failures),
        "failures": failures,
        "papers": summaries,
    }
    write_json_atomic(args.run_summary.resolve(), run_summary)
    print(json.dumps({"totals": dict(totals), "failures": len(failures)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
