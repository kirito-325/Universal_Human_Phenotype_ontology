#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
DEFAULT_DB = HERE / "data" / "cpo_browser.sqlite3"


def as_dict(cursor: sqlite3.Cursor, row: tuple) -> dict[str, object]:
    return {description[0]: row[index] for index, description in enumerate(cursor.description)}


class BrowserHandler(BaseHTTPRequestHandler):
    database = DEFAULT_DB

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def db(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)
        connection.row_factory = as_dict
        return connection

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                self.api(parsed.path, parse_qs(parsed.query))
            except (KeyError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except sqlite3.Error as error:
                self.send_json({"error": f"Database error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.static(parsed.path)

    def static(self, request_path: str) -> None:
        routes = {"/": "index.html", "/hpo": "hpo.html", "/mpo": "mpo.html"}
        name = routes.get(request_path, request_path.lstrip("/"))
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api(self, path: str, query: dict[str, list[str]]) -> None:
        parts = [unquote(part) for part in path.strip("/").split("/")]
        if parts == ["api", "meta"]:
            with self.db() as connection:
                meta = {row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata")}
            self.send_json(meta)
            return
        if len(parts) < 3 or parts[0] != "api" or parts[1] not in {"hpo", "mpo"}:
            raise ValueError("Unknown API route")
        side = parts[1]
        action = parts[2]
        if action == "children":
            self.children(side, query)
        elif action == "search":
            self.search(side, query)
        elif action == "path":
            self.path_to_root(side, query)
        elif action == "term" and len(parts) == 4:
            self.term(side, parts[3])
        elif action == "relations" and len(parts) == 4:
            self.relations(side, parts[3], query)
        elif action == "sources" and len(parts) == 4:
            self.sources(side, parts[3], query)
        else:
            raise ValueError("Unknown API route")

    def children(self, side: str, query: dict[str, list[str]]) -> None:
        parent = query.get("parent", [""])[0]
        with self.db() as connection:
            result = list(connection.execute(
                "SELECT t.term_id,t.label,t.node_type,t.depth,t.child_count,t.self_relation_count,t.inherited_assertion_count FROM edges e JOIN terms t ON t.side=e.side AND t.term_id=e.child_id WHERE e.side=? AND e.parent_id=? ORDER BY t.label COLLATE NOCASE,t.term_id",
                (side, parent),
            ))
        self.send_json({"items": result})

    def search(self, side: str, query: dict[str, list[str]]) -> None:
        text = query.get("q", [""])[0].strip()
        if len(text) < 2:
            self.send_json({"items": []})
            return
        pattern = f"%{text}%"
        with self.db() as connection:
            result = list(connection.execute(
                "SELECT term_id,label,node_type,depth,child_count,self_relation_count,inherited_assertion_count FROM terms WHERE side=? AND (term_id LIKE ? OR label LIKE ? OR synonyms LIKE ?) ORDER BY CASE WHEN term_id=? THEN 0 WHEN label LIKE ? THEN 1 ELSE 2 END,label COLLATE NOCASE LIMIT 50",
                (side, pattern, pattern, pattern, text, f"{text}%"),
            ))
        self.send_json({"items": result})

    def term(self, side: str, term_id: str) -> None:
        with self.db() as connection:
            item = connection.execute("SELECT * FROM terms WHERE side=? AND term_id=?", (side, term_id)).fetchone()
            if item is None:
                self.send_json({"error": "Term not found"}, HTTPStatus.NOT_FOUND)
                return
            item["status"] = json.loads(item.pop("status_json"))
            item["parents"] = list(connection.execute(
                "SELECT t.term_id,t.label,t.node_type,t.depth,t.child_count FROM edges e JOIN terms t ON t.side=e.side AND t.term_id=e.parent_id WHERE e.side=? AND e.child_id=? ORDER BY t.label COLLATE NOCASE",
                (side, term_id),
            ))
            item["children"] = list(connection.execute(
                "SELECT t.term_id,t.label,t.node_type,t.depth,t.child_count FROM edges e JOIN terms t ON t.side=e.side AND t.term_id=e.child_id WHERE e.side=? AND e.parent_id=? ORDER BY t.label COLLATE NOCASE",
                (side, term_id),
            ))
        self.send_json(item)

    def path_to_root(self, side: str, query: dict[str, list[str]]) -> None:
        term_id = query.get("id", [""])[0]
        with self.db() as connection:
            path = []
            seen = set()
            current = term_id
            while current and current not in seen:
                seen.add(current)
                row = connection.execute("SELECT term_id,label,depth FROM terms WHERE side=? AND term_id=?", (side, current)).fetchone()
                if row is None:
                    break
                path.append(row)
                parent = connection.execute(
                    "SELECT t.term_id FROM edges e JOIN terms t ON t.side=e.side AND t.term_id=e.parent_id WHERE e.side=? AND e.child_id=? ORDER BY t.depth DESC,t.label COLLATE NOCASE,t.term_id LIMIT 1",
                    (side, current),
                ).fetchone()
                current = parent["term_id"] if parent else ""
        self.send_json({"items": list(reversed(path))})

    def relations(self, side: str, term_id: str, query: dict[str, list[str]]) -> None:
        origin = query.get("origin", ["self"])[0]
        if origin not in {"self", "inherited"}:
            raise ValueError("origin must be self or inherited")
        table = "self_relations" if origin == "self" else "inherited_relations"
        page = max(1, int(query.get("page", ["1"])[0]))
        page_size = min(100, max(10, int(query.get("page_size", ["25"])[0])))
        clauses = ["side=?", "term_id=?"]
        params: list[object] = [side, term_id]
        direction = query.get("direction", [""])[0]
        complete = query.get("complete", [""])[0]
        text = query.get("q", [""])[0].strip()
        if direction:
            clauses.append("direction=?")
            params.append(direction)
        if complete in {"0", "1"}:
            clauses.append("complete_cpo=?")
            params.append(int(complete))
        if text:
            clauses.append("(cl_id LIKE ? OR cl_label LIKE ? OR go_id LIKE ? OR go_label LIKE ? OR cell_text LIKE ? OR process_text LIKE ?)")
            params.extend([f"%{text}%"] * 6)
        where = " AND ".join(clauses)
        with self.db() as connection:
            total = connection.execute(f"SELECT count(*) AS n FROM {table} WHERE {where}", params).fetchone()["n"]
            items = list(connection.execute(
                f"SELECT * FROM {table} WHERE {where} ORDER BY complete_cpo DESC,cl_label COLLATE NOCASE,go_label COLLATE NOCASE,relation_type,direction LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ))
        self.send_json({"items": items, "total": total, "page": page, "page_size": page_size, "pages": max(1, (total + page_size - 1) // page_size)})

    def sources(self, side: str, assertion_id: str, query: dict[str, list[str]]) -> None:
        page = max(1, int(query.get("page", ["1"])[0]))
        page_size = 50
        with self.db() as connection:
            total = connection.execute("SELECT count(*) AS n FROM provenance WHERE side=? AND assertion_id=?", (side, assertion_id)).fetchone()["n"]
            items = list(connection.execute(
                "SELECT p.source_leaf_id,t.label AS source_leaf_label,p.source_relation_id,p.source_scope,p.distance,p.shortest_path_count,p.canonical_path FROM provenance p LEFT JOIN terms t ON t.side=p.side AND t.term_id=p.source_leaf_id WHERE p.side=? AND p.assertion_id=? ORDER BY p.distance,p.source_leaf_id LIMIT ? OFFSET ?",
                (side, assertion_id, page_size, (page - 1) * page_size),
            ))
        self.send_json({"items": items, "total": total, "page": page, "pages": max(1, (total + page_size - 1) // page_size)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local HPO/MPO CPO browser")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    if not args.database.is_file():
        raise SystemExit(f"Database not found: {args.database}\nRun: python3 cpo_browser/build_database.py")
    BrowserHandler.database = args.database.resolve()
    server = ThreadingHTTPServer((args.host, args.port), BrowserHandler)
    print(f"CPO browser: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
