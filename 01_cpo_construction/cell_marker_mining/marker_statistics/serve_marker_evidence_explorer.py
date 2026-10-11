#!/usr/bin/env python3
"""Serve the cell-marker evidence explorer and its SQLite-backed API."""

from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_DIR = ROOT / "marker_statistics" / "evidence_explorer"
DEFAULT_DATABASE = DEFAULT_APP_DIR / "marker_evidence.sqlite3"


def rows(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, parameters)]


def bounded_integer(value: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class ExplorerStore:
    def __init__(self, database: Path) -> None:
        self.database = database

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def bootstrap(self) -> dict[str, Any]:
        with self.connect() as connection:
            metadata = {
                row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            cells = rows(
                connection,
                """
                SELECT catalog_id, catalog_order, level, rank, branch, cl_name,
                    label, label_zh, parent_id, parent_label, marker_count,
                    total_points, statistics_points, signatures_points, max_score,
                    alias_count, statistics_opportunities, signatures_opportunities,
                    total_opportunities
                FROM cells ORDER BY catalog_order
                """,
            )
            aliases = rows(
                connection,
                """
                SELECT alias_id, catalog_id, cell_name, decision, confidence,
                    top_score, paper_count, source_types
                FROM aliases ORDER BY catalog_id, paper_count DESC, cell_name COLLATE NOCASE
                """,
            )
        return {"metadata": metadata, "cells": cells, "aliases": aliases}

    def cell(self, catalog_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            cell = connection.execute(
                "SELECT * FROM cells WHERE catalog_id = ?", (catalog_id,)
            ).fetchone()
            if cell is None:
                return None
            aliases = rows(
                connection,
                """
                SELECT alias_id, cell_name, decision, confidence, top_score,
                    paper_count, source_types
                FROM aliases WHERE catalog_id = ?
                ORDER BY paper_count DESC, top_score DESC, cell_name COLLATE NOCASE
                """,
                (catalog_id,),
            )
        return {"cell": dict(cell), "aliases": aliases}

    def markers(
        self,
        catalog_id: str,
        query: str,
        sort: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        sort_sql = {
            "score": "marker_score DESC, statistics_score DESC, signatures_score DESC, gene COLLATE NOCASE",
            "statistics": "statistics_score DESC, marker_score DESC, gene COLLATE NOCASE",
            "signatures": "signatures_score DESC, marker_score DESC, gene COLLATE NOCASE",
            "gene": "gene COLLATE NOCASE, gene",
        }.get(sort, "marker_score DESC, statistics_score DESC, signatures_score DESC, gene COLLATE NOCASE")
        where = "catalog_id = ?"
        parameters: list[Any] = [catalog_id]
        if query:
            where += " AND gene LIKE ? ESCAPE '\\'"
            parameters.append(like_pattern(query))
        with self.connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM markers WHERE {where}", tuple(parameters)
            ).fetchone()[0]
            marker_rows = rows(
                connection,
                f"""
                SELECT gene, marker_rank, marker_score, statistics_score, signatures_score
                FROM markers WHERE {where}
                ORDER BY {sort_sql} LIMIT ? OFFSET ?
                """,
                tuple([*parameters, limit, offset]),
            )
        return {"total": total, "offset": offset, "limit": limit, "markers": marker_rows}

    def alias(
        self,
        alias_id: int,
        query: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any] | None:
        where = "e.alias_id = ?"
        parameters: list[Any] = [alias_id]
        if query:
            where += " AND e.gene LIKE ? ESCAPE '\\'"
            parameters.append(like_pattern(query))
        with self.connect() as connection:
            alias = connection.execute(
                """
                SELECT a.*, c.label AS catalog_label, c.label_zh AS catalog_label_zh
                FROM aliases a JOIN cells c ON c.catalog_id = a.catalog_id
                WHERE a.alias_id = ?
                """,
                (alias_id,),
            ).fetchone()
            if alias is None:
                return None
            total = connection.execute(
                f"SELECT COUNT(DISTINCT e.gene) FROM evidence e WHERE {where}",
                tuple(parameters),
            ).fetchone()[0]
            marker_rows = rows(
                connection,
                f"""
                SELECT e.gene,
                    COUNT(DISTINCT e.paper_id) AS paper_count,
                    COUNT(DISTINCT CASE WHEN e.table_type = 0 THEN e.paper_id END) AS statistics_papers,
                    COUNT(DISTINCT CASE WHEN e.table_type = 1 THEN e.paper_id END) AS signatures_papers,
                    COALESCE(m.marker_score, 0) AS marker_score
                FROM evidence e
                LEFT JOIN markers m ON m.catalog_id = e.catalog_id AND m.gene = e.gene
                WHERE {where}
                GROUP BY e.gene
                ORDER BY paper_count DESC, marker_score DESC, e.gene COLLATE NOCASE
                LIMIT ? OFFSET ?
                """,
                tuple([*parameters, limit, offset]),
            )
        return {
            "alias": dict(alias),
            "total": total,
            "offset": offset,
            "limit": limit,
            "markers": marker_rows,
        }

    def evidence(
        self,
        catalog_id: str,
        gene: str,
        alias_id: int | None,
        limit: int,
    ) -> dict[str, Any]:
        where = "e.catalog_id = ? AND e.gene = ?"
        parameters: list[Any] = [catalog_id, gene]
        if alias_id is not None:
            where += " AND e.alias_id = ?"
            parameters.append(alias_id)
        with self.connect() as connection:
            marker = connection.execute(
                """
                SELECT gene, marker_rank, marker_score, statistics_score, signatures_score
                FROM markers WHERE catalog_id = ? AND gene = ?
                """,
                (catalog_id, gene),
            ).fetchone()
            evidence_rows = rows(
                connection,
                f"""
                SELECT a.alias_id, a.cell_name, p.paper_id, p.title, p.journal,
                    p.publication_year, p.doi, p.pmid, p.url,
                    GROUP_CONCAT(DISTINCT e.source_gene) AS source_genes,
                    GROUP_CONCAT(DISTINCT e.source_species) AS source_species,
                    GROUP_CONCAT(DISTINCT e.mapping_method) AS mapping_methods,
                    GROUP_CONCAT(DISTINCT e.mapping_reference) AS mapping_references,
                    MAX(CASE WHEN e.table_type = 0 THEN 1 ELSE 0 END) AS statistics,
                    MAX(CASE WHEN e.table_type = 1 THEN 1 ELSE 0 END) AS signatures
                FROM evidence e
                JOIN aliases a ON a.alias_id = e.alias_id
                JOIN papers p ON p.paper_id = e.paper_id
                WHERE {where}
                GROUP BY a.alias_id, p.paper_id
                ORDER BY p.publication_year DESC, p.title COLLATE NOCASE, a.cell_name COLLATE NOCASE
                LIMIT ?
                """,
                tuple([*parameters, limit]),
            )
        return {
            "catalog_id": catalog_id,
            "gene": gene,
            "marker": dict(marker) if marker is not None else None,
            "alias_id": alias_id,
            "evidence": evidence_rows,
        }


class ExplorerHandler(BaseHTTPRequestHandler):
    store: ExplorerStore
    app_dir: Path

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_payload(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept_encoding and len(content) > 1024:
            content = gzip.compress(content, compresslevel=5)
            encoding = "gzip"
        else:
            encoding = ""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        self.wfile.write(content)

    def send_static(self, name: str) -> None:
        allowed = {"": "index.html", "/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        filename = allowed.get(name)
        if filename is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.app_dir / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_static(parsed.path)
            return
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/bootstrap":
                self.send_payload(self.store.bootstrap())
            elif parsed.path == "/api/cell":
                result = self.store.cell(str(query.get("catalog_id", [""])[0]))
                self.send_payload(result or {"error": "cell not found"}, HTTPStatus.OK if result else HTTPStatus.NOT_FOUND)
            elif parsed.path == "/api/markers":
                self.send_payload(
                    self.store.markers(
                        str(query.get("catalog_id", [""])[0]),
                        str(query.get("q", [""])[0]).strip(),
                        str(query.get("sort", ["score"])[0]),
                        bounded_integer(query.get("limit", ["100"])[0], 100, 1, 500),
                        bounded_integer(query.get("offset", ["0"])[0], 0, 0, 10_000_000),
                    )
                )
            elif parsed.path == "/api/alias":
                result = self.store.alias(
                    bounded_integer(query.get("alias_id", ["0"])[0], 0, 1, 10_000_000),
                    str(query.get("q", [""])[0]).strip(),
                    bounded_integer(query.get("limit", ["100"])[0], 100, 1, 500),
                    bounded_integer(query.get("offset", ["0"])[0], 0, 0, 10_000_000),
                )
                self.send_payload(result or {"error": "alias not found"}, HTTPStatus.OK if result else HTTPStatus.NOT_FOUND)
            elif parsed.path == "/api/evidence":
                alias_value = str(query.get("alias_id", [""])[0]).strip()
                self.send_payload(
                    self.store.evidence(
                        str(query.get("catalog_id", [""])[0]),
                        str(query.get("gene", [""])[0]),
                        bounded_integer(alias_value, 0, 1, 10_000_000) if alias_value else None,
                        bounded_integer(query.get("limit", ["500"])[0], 500, 1, 1000),
                    )
                )
            else:
                self.send_payload({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (sqlite3.Error, ValueError) as exc:
            self.send_payload({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.database.is_file():
        raise SystemExit(f"evidence database does not exist: {args.database}")
    if not args.app_dir.is_dir():
        raise SystemExit(f"app directory does not exist: {args.app_dir}")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    ExplorerHandler.store = ExplorerStore(args.database.resolve())
    ExplorerHandler.app_dir = args.app_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), ExplorerHandler)
    print(f"Marker evidence explorer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
