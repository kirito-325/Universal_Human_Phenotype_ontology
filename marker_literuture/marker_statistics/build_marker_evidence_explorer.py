#!/usr/bin/env python3
"""Build a queryable SQLite index for the cell-marker evidence explorer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "expanded" / "expanded_cell_catalog.tsv"
DEFAULT_MAPPING = ROOT / "database" / "cell_name_catalog_matches.tsv"
DEFAULT_MARKERS = ROOT / "marker_statistics" / "expanded_marker_scores.tsv"
DEFAULT_OPPORTUNITIES = ROOT / "marker_statistics" / "expanded_cell_evidence_opportunities.tsv"
DEFAULT_SUMMARY = ROOT / "marker_statistics" / "summary.json"
DEFAULT_PAPERS = ROOT / "database" / "papers"
DEFAULT_OUTPUT = ROOT / "marker_statistics" / "evidence_explorer" / "marker_evidence.sqlite3"
ACCEPTED_DECISIONS = frozenset({"exact_auto", "match"})
STATISTICS_NAME = "marker_statistics_cleaned.csv"
SIGNATURES_NAME = "marker_signatures_cleaned.tsv"


SCHEMA = """
CREATE TABLE cells (
    catalog_id TEXT PRIMARY KEY,
    catalog_order INTEGER NOT NULL,
    level INTEGER NOT NULL,
    rank TEXT NOT NULL,
    branch TEXT NOT NULL,
    cl_name TEXT NOT NULL,
    label TEXT NOT NULL,
    label_zh TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    parent_label TEXT NOT NULL,
    marker_count INTEGER NOT NULL DEFAULT 0,
    total_points INTEGER NOT NULL DEFAULT 0,
    statistics_points INTEGER NOT NULL DEFAULT 0,
    signatures_points INTEGER NOT NULL DEFAULT 0,
    max_score INTEGER NOT NULL DEFAULT 0,
    alias_count INTEGER NOT NULL DEFAULT 0,
    statistics_opportunities INTEGER NOT NULL DEFAULT 0,
    signatures_opportunities INTEGER NOT NULL DEFAULT 0,
    total_opportunities INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE aliases (
    alias_id INTEGER PRIMARY KEY,
    catalog_id TEXT NOT NULL REFERENCES cells(catalog_id),
    cell_name TEXT NOT NULL UNIQUE,
    decision TEXT NOT NULL,
    confidence TEXT NOT NULL,
    top_score REAL NOT NULL,
    paper_count INTEGER NOT NULL,
    source_types TEXT NOT NULL
);
CREATE TABLE markers (
    catalog_id TEXT NOT NULL REFERENCES cells(catalog_id),
    gene TEXT NOT NULL,
    marker_rank INTEGER NOT NULL,
    marker_score INTEGER NOT NULL,
    statistics_score INTEGER NOT NULL,
    signatures_score INTEGER NOT NULL,
    PRIMARY KEY (catalog_id, gene)
) WITHOUT ROWID;
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    journal TEXT NOT NULL,
    publication_year INTEGER NOT NULL,
    doi TEXT NOT NULL,
    pmid TEXT NOT NULL,
    url TEXT NOT NULL,
    record_path TEXT NOT NULL
);
CREATE TABLE evidence (
    alias_id INTEGER NOT NULL REFERENCES aliases(alias_id),
    catalog_id TEXT NOT NULL REFERENCES cells(catalog_id),
    gene TEXT NOT NULL,
    source_gene TEXT NOT NULL,
    source_species TEXT NOT NULL,
    mapping_method TEXT NOT NULL,
    mapping_reference TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    table_type INTEGER NOT NULL,
    PRIMARY KEY (alias_id, gene, source_gene, paper_id, table_type, mapping_method)
) WITHOUT ROWID;
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def integer(value: Any) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def number(value: Any) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def load_cells(connection: sqlite3.Connection, path: Path) -> set[str]:
    rows: list[tuple[Any, ...]] = []
    for row in read_tsv(path):
        catalog_id = str(row.get("cl_id") or "").strip()
        if not catalog_id:
            continue
        rows.append(
            (
                catalog_id,
                integer(row.get("order")),
                integer(row.get("level")),
                str(row.get("rank") or "").strip(),
                str(row.get("branch") or "").strip(),
                str(row.get("cl_name") or "").strip(),
                str(row.get("catalog_label") or "").strip(),
                str(row.get("catalog_label_zh") or "").strip(),
                str(row.get("parent_id") or "").strip(),
                str(row.get("parent_label") or "").strip(),
            )
        )
    connection.executemany(
        """
        INSERT INTO cells (
            catalog_id, catalog_order, level, rank, branch, cl_name, label,
            label_zh, parent_id, parent_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return {str(row[0]) for row in rows}


def load_aliases(
    connection: sqlite3.Connection,
    path: Path,
    catalog_ids: set[str],
) -> dict[str, tuple[int, str]]:
    aliases: dict[str, tuple[int, str]] = {}
    rows: list[tuple[Any, ...]] = []
    for row in read_tsv(path):
        decision = str(row.get("decision") or "").strip().lower()
        if decision not in ACCEPTED_DECISIONS:
            continue
        cell_name = str(row.get("cell_name") or "").strip()
        catalog_id = str(row.get("catalog_id") or "").strip()
        if not cell_name or catalog_id not in catalog_ids:
            continue
        previous = aliases.get(cell_name)
        if previous is not None:
            if previous[1] != catalog_id:
                raise ValueError(
                    f"conflicting accepted mappings for {cell_name!r}: "
                    f"{previous[1]} vs {catalog_id}"
                )
            continue
        alias_id = len(aliases) + 1
        aliases[cell_name] = (alias_id, catalog_id)
        rows.append(
            (
                alias_id,
                catalog_id,
                cell_name,
                decision,
                str(row.get("confidence") or "").strip(),
                number(row.get("top_score")),
                integer(row.get("paper_count")),
                str(row.get("source_types") or "").strip(),
            )
        )
    connection.executemany(
        """
        INSERT INTO aliases (
            alias_id, catalog_id, cell_name, decision, confidence, top_score,
            paper_count, source_types
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.execute(
        """
        UPDATE cells
        SET alias_count = (
            SELECT COUNT(*) FROM aliases WHERE aliases.catalog_id = cells.catalog_id
        )
        """
    )
    return aliases


def load_markers(connection: sqlite3.Connection, path: Path) -> int:
    batch: list[tuple[Any, ...]] = []
    metrics: dict[str, Counter[str]] = {}
    total = 0
    for row in read_tsv(path):
        catalog_id = str(row.get("catalog_id") or "").strip()
        gene = str(row.get("gene") or "").strip()
        if not catalog_id or not gene:
            continue
        marker_score = integer(row.get("marker_score"))
        statistics_score = integer(row.get("statistics_score"))
        signatures_score = integer(row.get("signatures_score"))
        batch.append(
            (
                catalog_id,
                gene,
                integer(row.get("marker_rank")),
                marker_score,
                statistics_score,
                signatures_score,
            )
        )
        metric = metrics.setdefault(catalog_id, Counter())
        metric["marker_count"] += 1
        metric["total_points"] += marker_score
        metric["statistics_points"] += statistics_score
        metric["signatures_points"] += signatures_score
        metric["max_score"] = max(metric["max_score"], marker_score)
        total += 1
        if len(batch) >= 25000:
            connection.executemany(
                "INSERT INTO markers VALUES (?, ?, ?, ?, ?, ?)", batch
            )
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO markers VALUES (?, ?, ?, ?, ?, ?)", batch)
    connection.executemany(
        """
        UPDATE cells SET marker_count = ?, total_points = ?, statistics_points = ?,
            signatures_points = ?, max_score = ? WHERE catalog_id = ?
        """,
        [
            (
                values["marker_count"],
                values["total_points"],
                values["statistics_points"],
                values["signatures_points"],
                values["max_score"],
                catalog_id,
            )
            for catalog_id, values in metrics.items()
        ],
    )
    return total


def load_opportunities(connection: sqlite3.Connection, path: Path) -> None:
    rows = []
    for row in read_tsv(path):
        rows.append(
            (
                integer(row.get("statistics_opportunities")),
                integer(row.get("signatures_opportunities")),
                integer(row.get("total_opportunities")),
                str(row.get("catalog_id") or "").strip(),
            )
        )
    connection.executemany(
        """
        UPDATE cells SET statistics_opportunities = ?, signatures_opportunities = ?,
            total_opportunities = ? WHERE catalog_id = ?
        """,
        rows,
    )


def iter_record_dirs(path: Path) -> Iterator[Path]:
    for record_dir in sorted(path.iterdir()):
        output = record_dir / "agent_standardized"
        if record_dir.is_dir() and (
            (output / STATISTICS_NAME).is_file()
            or (output / SIGNATURES_NAME).is_file()
        ):
            yield record_dir


def paper_record(record_dir: Path, metadata: dict[str, Any]) -> tuple[Any, ...]:
    paper_id = str(metadata.get("id") or record_dir.name).strip() or record_dir.name
    try:
        record_path = str(record_dir.relative_to(ROOT))
    except ValueError:
        record_path = str(record_dir.resolve())
    return (
        paper_id,
        str(metadata.get("title") or "").strip(),
        str(metadata.get("journal") or "").strip(),
        integer(metadata.get("publication_year")),
        str(metadata.get("doi") or "").strip(),
        str(metadata.get("pmid") or "").strip(),
        str(metadata.get("url") or "").strip(),
        record_path,
    )


def statistics_evidence(
    path: Path,
    aliases: dict[str, tuple[int, str]],
) -> set[tuple[int, str, str, str, str, str, str]]:
    evidence: set[tuple[int, str, str, str, str, str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"cell_cluster", "gene"} <= set(reader.fieldnames):
            raise ValueError(f"missing cell_cluster/gene columns: {path}")
        for row in reader:
            mapped = aliases.get(str(row.get("cell_cluster") or "").strip())
            gene = str(row.get("gene") or "").strip()
            if mapped is not None and gene:
                evidence.add(
                    (
                        mapped[0],
                        mapped[1],
                        gene,
                        str(row.get("source_gene") or gene).strip(),
                        str(row.get("source_species") or "unknown").strip(),
                        str(row.get("gene_mapping_method") or "unknown").strip(),
                        str(row.get("gene_mapping_reference") or "").strip(),
                    )
                )
    return evidence


def signature_evidence(
    path: Path,
    aliases: dict[str, tuple[int, str]],
) -> set[tuple[int, str, str, str, str, str, str]]:
    evidence: set[tuple[int, str, str, str, str, str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_cluster", "gene"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"missing cell_cluster/gene columns: {path}")
        for row in reader:
            mapped = aliases.get(str(row.get("cell_cluster") or "").strip())
            gene = str(row.get("gene") or "").strip()
            if mapped is not None and gene:
                evidence.add(
                    (
                        mapped[0],
                        mapped[1],
                        gene,
                        str(row.get("source_gene") or gene).strip(),
                        str(row.get("source_species") or "unknown").strip(),
                        str(row.get("gene_mapping_method") or "unknown").strip(),
                        str(row.get("gene_mapping_reference") or "").strip(),
                    )
                )
    return evidence


def load_evidence(
    connection: sqlite3.Connection,
    papers_dir: Path,
    aliases: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    counts = Counter()
    errors: list[str] = []
    seen_papers: set[str] = set()
    for number, record_dir in enumerate(iter_record_dirs(papers_dir), 1):
        metadata = read_json(record_dir / "metadata.json")
        paper = paper_record(record_dir, metadata)
        paper_id = str(paper[0])
        if paper_id in seen_papers:
            counts["duplicate_papers"] += 1
            continue
        seen_papers.add(paper_id)
        connection.execute("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)", paper)
        output = record_dir / "agent_standardized"
        inputs = (
            (0, output / STATISTICS_NAME, statistics_evidence, "statistics_evidence"),
            (1, output / SIGNATURES_NAME, signature_evidence, "signatures_evidence"),
        )
        for table_type, path, reader, count_key in inputs:
            if not path.is_file():
                continue
            try:
                items = reader(path, aliases)
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            connection.executemany(
                "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        alias_id,
                        catalog_id,
                        gene,
                        source_gene,
                        source_species,
                        mapping_method,
                        mapping_reference,
                        paper_id,
                        table_type,
                    )
                    for (
                        alias_id,
                        catalog_id,
                        gene,
                        source_gene,
                        source_species,
                        mapping_method,
                        mapping_reference,
                    ) in items
                ),
            )
            counts[count_key] += len(items)
        counts["papers"] += 1
        if number % 25 == 0:
            connection.commit()
        if number % 100 == 0:
            print(
                f"[{number}] papers; evidence="
                f"{counts['statistics_evidence'] + counts['signatures_evidence']:,}",
                flush=True,
            )
    counts["errors"] = len(errors)
    return {**dict(counts), "error_messages": errors}


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX aliases_catalog_name ON aliases(catalog_id, cell_name COLLATE NOCASE);
        CREATE INDEX markers_catalog_score ON markers(
            catalog_id, marker_score DESC, statistics_score DESC,
            signatures_score DESC, gene COLLATE NOCASE
        );
        CREATE INDEX evidence_catalog_gene ON evidence(catalog_id, gene);
        CREATE INDEX evidence_alias_gene ON evidence(alias_id, gene);
        CREATE INDEX evidence_paper ON evidence(paper_id);
        ANALYZE;
        """
    )


def build_database(
    catalog_path: Path,
    mapping_path: Path,
    markers_path: Path,
    opportunities_path: Path,
    summary_path: Path,
    papers_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; "
            "PRAGMA temp_store=MEMORY; PRAGMA cache_size=-262144;"
        )
        connection.executescript(SCHEMA)
        catalog_ids = load_cells(connection, catalog_path)
        aliases = load_aliases(connection, mapping_path, catalog_ids)
        marker_pairs = load_markers(connection, markers_path)
        load_opportunities(connection, opportunities_path)
        evidence = load_evidence(connection, papers_dir, aliases)
        source_summary = read_json(summary_path)
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalog_nodes": len(catalog_ids),
            "accepted_aliases": len(aliases),
            "marker_pairs": marker_pairs,
            "cells_with_markers": connection.execute(
                "SELECT COUNT(*) FROM cells WHERE marker_count > 0"
            ).fetchone()[0],
            "papers": evidence.get("papers", 0),
            "evidence_rows": evidence.get("statistics_evidence", 0)
            + evidence.get("signatures_evidence", 0),
            "statistics_evidence_rows": evidence.get("statistics_evidence", 0),
            "signatures_evidence_rows": evidence.get("signatures_evidence", 0),
            "source_generated_at": source_summary.get("generated_at", ""),
            "errors": evidence.get("error_messages", []),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            ((key, json.dumps(value, ensure_ascii=False)) for key, value in result.items()),
        )
        create_indexes(connection)
        connection.commit()
        connection.close()
        os.replace(temporary, output_path)
        output_path.chmod(0o644)
        return {**result, "output": str(output_path.resolve())}
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--opportunities", type=Path, default=DEFAULT_OPPORTUNITIES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    for path in (
        args.catalog,
        args.mapping,
        args.markers,
        args.opportunities,
        args.summary,
    ):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    if not args.papers_dir.is_dir():
        raise SystemExit(f"papers directory does not exist: {args.papers_dir}")
    result = build_database(
        args.catalog.resolve(),
        args.mapping.resolve(),
        args.markers.resolve(),
        args.opportunities.resolve(),
        args.summary.resolve(),
        args.papers_dir.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
