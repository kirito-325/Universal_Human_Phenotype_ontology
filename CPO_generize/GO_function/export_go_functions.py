from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celltype_configure.common import load_obo  # noqa: E402


EXPERIMENTAL_EVIDENCE_CODES = {
    "EXP",
    "HDA",
    "HEP",
    "HGI",
    "HMP",
    "HTP",
    "IDA",
    "IEP",
    "IGI",
    "IMP",
    "IPI",
}

GO_FUNCTION_GENE_FIELDS = [
    "go_id",
    "go_name",
    "human_gene_symbol",
    "annotation_count",
    "has_experimental_evidence",
    "evidence_codes",
    "annotation_relations",
    "db_object_ids",
    "references",
]

GO_FUNCTION_SUMMARY_FIELDS = [
    "go_id",
    "go_name",
    "human_function_gene_count",
    "experimental_evidence_gene_count",
    "direct_annotation_count",
    "evidence_codes",
]


def write_atomic_tsv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def active_biological_process_terms(go_obo: Path) -> dict[str, str]:
    graph = load_obo(go_obo)
    return {
        term_id: str(data.get("name", ""))
        for term_id, data in graph.nodes(data=True)
        if data.get("namespace") == "biological_process"
        and data.get("is_obsolete") != "true"
    }


def split_multi_value(value: str) -> list[str]:
    return ordered_unique(
        item for comma_part in value.split(",") for item in comma_part.split("|")
    )


def export_go_functions(
    gaf_path: Path,
    go_terms: Mapping[str, str],
    output_dir: Path,
) -> dict[str, int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    metadata: dict[str, str] = {}
    active_go_ids = set(go_terms)

    with tempfile.NamedTemporaryFile(
        prefix="go_function_", suffix=".sqlite3", dir=output_dir, delete=False
    ) as temporary_database:
        database_path = Path(temporary_database.name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE annotations (
                go_id TEXT NOT NULL,
                gene_symbol TEXT NOT NULL,
                db_object_id TEXT NOT NULL,
                evidence_code TEXT NOT NULL,
                reference TEXT NOT NULL,
                annotation_relation TEXT NOT NULL
            )
            """
        )
        batch: list[tuple[str, str, str, str, str, str]] = []
        with gaf_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("!"):
                    if line.startswith("!date-generated:"):
                        metadata["gaf_date_generated"] = line.split(":", 1)[1].strip()
                    elif line.startswith("!go-version:"):
                        metadata["gaf_go_version"] = line.split(":", 1)[1].strip()
                    continue
                counts["gaf_data_rows"] += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 15:
                    counts["malformed_gaf_rows"] += 1
                    continue
                if fields[8] != "P":
                    counts["non_bp_rows"] += 1
                    continue
                relations = split_multi_value(fields[3])
                if "NOT" in relations:
                    counts["not_qualifier_rows"] += 1
                    continue
                if "taxon:9606" not in split_multi_value(fields[12]):
                    counts["non_human_rows"] += 1
                    continue
                go_id = fields[4]
                if go_id not in active_go_ids:
                    counts["inactive_or_unknown_bp_rows"] += 1
                    continue
                gene_symbol = fields[2].strip()
                if not gene_symbol:
                    counts["empty_gene_rows"] += 1
                    continue
                batch.append(
                    (
                        go_id,
                        gene_symbol,
                        f"{fields[0]}:{fields[1]}",
                        fields[6],
                        fields[5],
                        fields[3],
                    )
                )
                counts["retained_direct_annotations"] += 1
                if len(batch) >= 10000:
                    connection.executemany(
                        "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?)", batch
            )
        connection.commit()
        connection.execute(
            "CREATE INDEX annotation_go_gene_idx ON annotations(go_id, gene_symbol)"
        )

        gene_output = output_dir / "go_function_genes.tsv"
        gene_temporary = gene_output.with_suffix(gene_output.suffix + ".part")
        summaries: dict[str, dict[str, object]] = {}
        unique_genes: set[str] = set()
        with gene_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=GO_FUNCTION_GENE_FIELDS, delimiter="\t"
            )
            writer.writeheader()
            query = connection.execute(
                """
                SELECT
                    go_id,
                    gene_symbol,
                    COUNT(*),
                    GROUP_CONCAT(DISTINCT evidence_code),
                    GROUP_CONCAT(DISTINCT annotation_relation),
                    GROUP_CONCAT(DISTINCT db_object_id),
                    GROUP_CONCAT(DISTINCT reference)
                FROM annotations
                GROUP BY go_id, gene_symbol
                ORDER BY go_id, gene_symbol
                """
            )
            for (
                go_id,
                gene_symbol,
                annotation_count,
                raw_evidence_codes,
                raw_relations,
                raw_db_object_ids,
                raw_references,
            ) in query:
                evidence_codes = sorted(split_multi_value(raw_evidence_codes or ""))
                relations = sorted(split_multi_value(raw_relations or ""))
                db_object_ids = sorted(split_multi_value(raw_db_object_ids or ""))
                references = sorted(split_multi_value(raw_references or ""))
                has_experimental = bool(
                    set(evidence_codes) & EXPERIMENTAL_EVIDENCE_CODES
                )
                writer.writerow(
                    {
                        "go_id": go_id,
                        "go_name": go_terms[go_id],
                        "human_gene_symbol": gene_symbol,
                        "annotation_count": annotation_count,
                        "has_experimental_evidence": str(has_experimental).lower(),
                        "evidence_codes": "|".join(evidence_codes),
                        "annotation_relations": "|".join(relations),
                        "db_object_ids": "|".join(db_object_ids),
                        "references": "|".join(references),
                    }
                )
                summary = summaries.setdefault(
                    go_id,
                    {
                        "gene_count": 0,
                        "experimental_gene_count": 0,
                        "annotation_count": 0,
                        "evidence_codes": set(),
                    },
                )
                summary["gene_count"] += 1
                summary["experimental_gene_count"] += int(has_experimental)
                summary["annotation_count"] += int(annotation_count)
                summary["evidence_codes"].update(evidence_codes)
                unique_genes.add(gene_symbol)
                counts["unique_go_function_gene_pairs"] += 1
        gene_temporary.replace(gene_output)

        summary_rows: list[dict[str, object]] = []
        for go_id in sorted(go_terms):
            summary = summaries.get(
                go_id,
                {
                    "gene_count": 0,
                    "experimental_gene_count": 0,
                    "annotation_count": 0,
                    "evidence_codes": set(),
                },
            )
            summary_rows.append(
                {
                    "go_id": go_id,
                    "go_name": go_terms[go_id],
                    "human_function_gene_count": summary["gene_count"],
                    "experimental_evidence_gene_count": summary[
                        "experimental_gene_count"
                    ],
                    "direct_annotation_count": summary["annotation_count"],
                    "evidence_codes": "|".join(sorted(summary["evidence_codes"])),
                }
            )
        summary_output = output_dir / "go_function_summary.tsv"
        write_atomic_tsv(summary_output, GO_FUNCTION_SUMMARY_FIELDS, summary_rows)
        counts["active_go_bp_terms"] = len(go_terms)
        counts["go_terms_with_function_genes"] = len(summaries)
        counts["go_terms_without_function_genes"] = len(go_terms) - len(summaries)
        counts["unique_human_function_genes"] = len(unique_genes)
        return {
            **dict(counts),
            **metadata,
            "gene_output": str(gene_output.resolve()),
            "summary_output": str(summary_output.resolve()),
        }
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    go_terms = active_biological_process_terms(args.go_obo)
    function_counts = export_go_functions(args.gaf, go_terms, args.output_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": {
            "definition": (
                "All active GO biological-process terms are retained. Human function genes "
                "come from direct positive human GOA annotations."
            ),
            "go_annotation_propagation": False,
            "intersection_performed": False,
            "not_annotations_retained": False,
        },
        "inputs": {
            "gaf": str(args.gaf.resolve()),
            "go_obo": str(args.go_obo.resolve()),
        },
        "go_function": function_counts,
    }
    write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export direct positive human GO biological-process gene annotations."
    )
    parser.add_argument(
        "--gaf",
        type=Path,
        default=Path("/path/to/workspace/UHPO（6-8）/data/go/goa_human.gaf"),
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=Path("/path/to/workspace/HPO_expand/CPO/data/go/go-basic.obo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_ROOT / "results",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MODULE_ROOT / "results/go_function_manifest.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
