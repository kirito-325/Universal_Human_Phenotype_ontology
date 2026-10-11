#!/usr/bin/env python3
"""Translate MGI mouse MP annotations to human genes using explicit MGI groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def load_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    in_term = False
    identifier = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if line == "[Term]":
            in_term = True
            identifier = ""
        elif in_term and line.startswith("id: MP:"):
            identifier = line.split(":", 1)[1].strip()
        elif in_term and line.startswith("name: ") and identifier:
            names[identifier] = line[6:].strip()
        elif in_term and not line:
            in_term = False
    return names


def load_marker_symbols(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mgi_id = row.get("MGI Accession ID", "").strip()
            symbol = row.get("Marker Symbol", "").strip()
            if mgi_id and symbol:
                mapping[mgi_id] = symbol
    return mapping


def load_homology(path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "DB Class Key",
            "Common Organism Name",
            "NCBI Taxon ID",
            "Symbol",
            "EntrezGene ID",
            "Mouse MGI ID",
            "HGNC ID",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"HOM report missing columns: {sorted(missing)}")
        for row in reader:
            taxon = row.get("NCBI Taxon ID", "").strip()
            if taxon in {"10090", "9606"}:
                groups[row["DB Class Key"].strip()].append(row)

    by_mouse: dict[str, list[dict[str, str]]] = defaultdict(list)
    stats = {"groups": len(groups), "mouse_rows": 0, "human_rows": 0}
    for rows in groups.values():
        mouse_rows = [row for row in rows if row.get("NCBI Taxon ID") == "10090"]
        human_rows = [row for row in rows if row.get("NCBI Taxon ID") == "9606"]
        stats["mouse_rows"] += len(mouse_rows)
        stats["human_rows"] += len(human_rows)
        for mouse in mouse_rows:
            mgi_id = mouse.get("Mouse MGI ID", "").strip()
            if mgi_id:
                for human in human_rows:
                    by_mouse[mgi_id].append(
                        {
                            "mouse_symbol": mouse.get("Symbol", "").strip(),
                            "human_symbol": human.get("Symbol", "").strip().upper(),
                            "human_entrez": human.get("EntrezGene ID", "").strip(),
                            "human_hgnc_id": human.get("HGNC ID", "").strip(),
                            "db_class_key": mouse.get("DB Class Key", "").strip(),
                        }
                    )
    for mgi_id, values in by_mouse.items():
        dedup: dict[tuple[str, str, str], dict[str, str]] = {}
        for value in values:
            dedup[(value["human_symbol"], value["human_entrez"], value["human_hgnc_id"])] = value
        by_mouse[mgi_id] = list(dedup.values())
    stats["mouse_mgi_with_human"] = len(by_mouse)
    stats["human_links"] = sum(len(values) for values in by_mouse.values())
    return by_mouse, stats


def mapping_type(value_count: int) -> str:
    return "one2one" if value_count == 1 else "one2many"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo-obo", type=Path, required=True)
    parser.add_argument("--gene-pheno", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--homology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    names = load_names(args.mpo_obo)
    marker_symbols = load_marker_symbols(args.markers)
    homology, homology_stats = load_homology(args.homology)
    annotation_fields = (
        "mp_id",
        "mp_name",
        "mouse_mgi_id",
        "mouse_symbol",
        "genotype",
        "genetic_background",
        "pmid",
        "human_ortholog_count",
        "human_symbols",
        "mapping_status",
        "evidence_source",
    )
    human_fields = (
        "mp_id",
        "mp_name",
        "mouse_mgi_id",
        "mouse_symbol",
        "human_symbol",
        "human_entrez",
        "human_hgnc_id",
        "mapping_type",
        "db_class_key",
        "genotype",
        "genetic_background",
        "pmid",
        "evidence_source",
    )
    evidence_fields = (
        "mp_id",
        "mp_name",
        "human_gene_symbol",
        "mouse_mgi_id",
        "mouse_symbol",
        "pmid",
        "genotype",
        "genetic_background",
        "mapping_type",
        "evidence_source",
        "evidence_weight",
    )
    annotation_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    seen_evidence: set[tuple[str, str, str, str, str]] = set()
    total_rows = 0
    malformed_rows = 0
    with args.gene_pheno.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            row = raw.rstrip("\n").split("\t")
            total_rows += 1
            if len(row) < 8 or not row[4].startswith("MP:") or not row[6].startswith("MGI:"):
                malformed_rows += 1
                continue
            mp_id, mgi_id = row[4].strip(), row[6].strip()
            genotype, background, pmid = row[0].strip(), row[3].strip(), row[5].strip()
            values = homology.get(mgi_id, [])
            mouse_symbol = marker_symbols.get(mgi_id, row[1].split("<", 1)[0].strip())
            if not values:
                excluded_ids.add(mgi_id)
            annotation_rows.append(
                {
                    "mp_id": mp_id,
                    "mp_name": names.get(mp_id, mp_id),
                    "mouse_mgi_id": mgi_id,
                    "mouse_symbol": mouse_symbol,
                    "genotype": genotype,
                    "genetic_background": background,
                    "pmid": pmid,
                    "human_ortholog_count": len(values),
                    "human_symbols": ";".join(sorted({v["human_symbol"] for v in values})),
                    "mapping_status": "mapped" if values else "mouse_only",
                    "evidence_source": "MGI_GenePheno.rpt",
                }
            )
            for value in values:
                key = (mp_id, mgi_id, value["human_symbol"], pmid, genotype)
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                one_weight = 1.0 / max(1, len(values))
                mapped = {
                    "mp_id": mp_id,
                    "mp_name": names.get(mp_id, mp_id),
                    "mouse_mgi_id": mgi_id,
                    "mouse_symbol": mouse_symbol,
                    "human_symbol": value["human_symbol"],
                    "human_entrez": value["human_entrez"],
                    "human_hgnc_id": value["human_hgnc_id"],
                    "mapping_type": mapping_type(len(values)),
                    "db_class_key": value["db_class_key"],
                    "genotype": genotype,
                    "genetic_background": background,
                    "pmid": pmid,
                    "evidence_source": "MGI_GenePheno.rpt+HOM_MouseHumanSequence.rpt",
                }
                human_rows.append(mapped)
                evidence_rows.append(
                    {
                        "mp_id": mp_id,
                        "mp_name": names.get(mp_id, mp_id),
                        "human_gene_symbol": value["human_symbol"],
                        "mouse_mgi_id": mgi_id,
                        "mouse_symbol": mouse_symbol,
                        "pmid": pmid,
                        "genotype": genotype,
                        "genetic_background": background,
                        "mapping_type": mapped["mapping_type"],
                        "evidence_source": mapped["evidence_source"],
                        "evidence_weight": f"{one_weight:.6f}",
                    }
                )

    annotation_path = args.output_dir / "mpo_mouse_gene_annotations.tsv"
    human_path = args.output_dir / "mpo_human_ortholog_annotations.tsv"
    evidence_path = args.output_dir / "mpo_human_gene_evidence.tsv"
    excluded_path = args.output_dir / "excluded_mouse_only_genes.tsv"
    write_tsv(annotation_path, annotation_rows, annotation_fields)
    write_tsv(human_path, human_rows, human_fields)
    write_tsv(evidence_path, evidence_rows, evidence_fields)
    excluded_rows = [
        {
            "mouse_mgi_id": mgi_id,
            "mouse_symbol": marker_symbols.get(mgi_id, ""),
            "exclusion_reason": "no human row in current MGI homology group",
        }
        for mgi_id in sorted(excluded_ids)
    ]
    write_tsv(excluded_path, excluded_rows, ("mouse_mgi_id", "mouse_symbol", "exclusion_reason"))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "mpo_obo": str(args.mpo_obo.resolve()),
            "mgi_gene_pheno": str(args.gene_pheno.resolve()),
            "mgi_markers": str(args.markers.resolve()),
            "mgi_homology": str(args.homology.resolve()),
        },
        "policy": {
            "anchor": "MGI marker accession ID, never symbol-only matching",
            "mouse_only_genes": "excluded",
            "one_to_many_orthology": "retain every human ortholog and divide row weight by ortholog count",
            "parent_child_mpo_inheritance": False,
        },
        "counts": {
            "gene_pheno_rows": total_rows,
            "malformed_rows": malformed_rows,
            "annotation_rows": len(annotation_rows),
            "human_ortholog_rows": len(human_rows),
            "unique_mouse_mgi_ids": len({row["mouse_mgi_id"] for row in annotation_rows}),
            "mapped_mouse_mgi_ids": len({row["mouse_mgi_id"] for row in human_rows}),
            "excluded_mouse_only_mgi_ids": len(excluded_ids),
            "unique_mpo_terms": len({row["mp_id"] for row in annotation_rows}),
            "unique_mpo_terms_with_human_genes": len({row["mp_id"] for row in human_rows}),
            "unique_human_genes": len({row["human_symbol"] for row in human_rows}),
            "deduplicated_evidence_rows": len(evidence_rows),
            "homology_report": homology_stats,
        },
        "outputs": {
            "mouse_annotations": str(annotation_path.resolve()),
            "human_annotations": str(human_path.resolve()),
            "evidence": str(evidence_path.resolve()),
            "excluded_mouse_only": str(excluded_path.resolve()),
        },
    }
    manifest_path = args.output_dir / "ortholog_mapping_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
