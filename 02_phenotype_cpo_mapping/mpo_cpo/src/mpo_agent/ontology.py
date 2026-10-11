"""Versioned local OBO catalogs; lexical recall proposes terms, never accepts a mapping."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .schemas import OntologyTerm


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quoted(value: str) -> str:
    match = re.search(r'"((?:\\.|[^"\\])*)"', value)
    if not match:
        return ""
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m[1], m[1]), match[1])


def read_obo(path: Path, prefix: str) -> Iterator[OntologyTerm]:
    stanza: dict[str, list[str]] = {}
    is_term = False

    def term():
        term_id = stanza.get("id", [""])[0]
        if not is_term or not term_id.startswith(prefix + ":"):
            return None
        return OntologyTerm(
            term_id=term_id, label=stanza.get("name", [term_id])[0],
            namespace=stanza.get("namespace", [""])[0],
            definition=quoted(stanza.get("def", [""])[0]),
            synonyms=[quoted(s) for s in stanza.get("synonym", [])],
            parents=[p.split()[0] for p in stanza.get("is_a", [])],
            alt_ids=stanza.get("alt_id", []),
            obsolete=stanza.get("is_obsolete", ["false"])[0] == "true",
            replaced_by=[p.split()[0] for p in stanza.get("replaced_by", [])],
        )

    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("["):
                if item := term():
                    yield item
                stanza = {}
                is_term = line == "[Term]"
            elif is_term and ": " in line:
                key, value = line.split(": ", 1)
                stanza.setdefault(key, []).append(value)
        if item := term():
            yield item


def build_index(cl_path: Path, go_path: Path, output: Path) -> dict:
    """Create a new index. Existing snapshots are not silently replaced."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=".ontology-")
    os.close(descriptor)
    try:
        metadata = _populate_index(cl_path, go_path, Path(temporary))
        os.link(temporary, output)
        return metadata
    finally:
        os.unlink(temporary)


def _populate_index(cl_path: Path, go_path: Path, output: Path) -> dict:
    metadata = {"format_version": "1", "sources": {
        "CL": {"path": str(cl_path.resolve()), "sha256": digest_file(cl_path)},
        "GO": {"path": str(go_path.resolve()), "sha256": digest_file(go_path)},
    }}
    metadata["sha256"] = hashlib.sha256(json.dumps(metadata["sources"], sort_keys=True).encode()).hexdigest()
    with closing(sqlite3.connect(output)) as db, db:
        db.executescript("""
            CREATE TABLE terms (id TEXT PRIMARY KEY, prefix TEXT, obsolete INTEGER, payload TEXT);
            CREATE TABLE aliases (alias TEXT PRIMARY KEY, id TEXT);
            CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, label, synonyms, definition);
            CREATE TABLE metadata (payload TEXT);
        """)
        counts = {}
        for prefix, path in (("CL", cl_path), ("GO", go_path)):
            count = 0
            for item in read_obo(path, prefix):
                db.execute("INSERT INTO terms VALUES (?, ?, ?, ?)",
                           (item.term_id, prefix, int(item.obsolete), item.model_dump_json()))
                for alias in item.alt_ids:
                    db.execute("INSERT INTO aliases VALUES (?, ?)", (alias, item.term_id))
                if not item.obsolete:
                    db.execute("INSERT INTO search VALUES (?, ?, ?, ?)", (
                        item.term_id, item.label, " | ".join(item.synonyms), item.definition,
                    ))
                count += 1
            if count == 0:
                raise ValueError(f"No {prefix} terms in {path}")
            counts[prefix] = count
        metadata["counts"] = counts
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
    for prefix, path in (("CL", cl_path), ("GO", go_path)):
        if digest_file(path) != metadata["sources"][prefix]["sha256"]:
            raise ValueError(f"{prefix} snapshot changed while indexing")
    return metadata


class OntologyIndex:
    def __init__(self, path: Path):
        self.path = path.resolve()
        with self.connect() as db:
            self.metadata = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
        self.sha256 = digest_file(self.path)
        self.metadata["index_sha256"] = self.sha256

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            yield db

    def get(self, term_id: str) -> OntologyTerm | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM terms WHERE id = ?", (term_id,)).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT payload FROM terms JOIN aliases ON terms.id = aliases.id WHERE alias = ?",
                    (term_id,),
                ).fetchone()
            return OntologyTerm.model_validate_json(row[0]) if row else None

    def search(self, query: str, prefix: str, limit: int = 8) -> list[OntologyTerm]:
        if prefix not in {"CL", "GO"} or not 1 <= limit <= 30:
            raise ValueError("prefix must be CL/GO; limit must be 1..30")
        exact = self.get(query.strip())
        if exact and exact.term_id.startswith(prefix + ":") and not exact.obsolete:
            return [exact]
        words = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower())))[:24]
        if not words:
            return []
        match_query = " OR ".join('"' + word + '"' for word in words)
        with self.connect() as db:
            # BM25 alone can put a generic exact label below many specialized terms
            # whose synonyms repeat that label. Preserve exact labels/synonyms first.
            exact_rows = db.execute("""
                SELECT terms.payload FROM search JOIN terms ON search.id = terms.id
                WHERE terms.prefix = ? AND terms.obsolete = 0 AND (
                    lower(search.label) = lower(?) OR
                    instr(' | ' || lower(search.synonyms) || ' | ', ' | ' || lower(?) || ' | ') > 0
                ) ORDER BY CASE WHEN lower(search.label) = lower(?) THEN 0 ELSE 1 END, terms.id
                LIMIT ?
            """, (prefix, query.strip(), query.strip(), query.strip(), limit)).fetchall()
            rows = db.execute("""
                SELECT terms.payload FROM search JOIN terms ON search.id = terms.id
                WHERE search MATCH ? AND terms.prefix = ? AND terms.obsolete = 0
                ORDER BY bm25(search, 0, 8, 5, 1), terms.id LIMIT ?
            """, (match_query, prefix, limit)).fetchall()
        unique = {}
        for row in [*exact_rows, *rows]:
            item = OntologyTerm.model_validate_json(row[0])
            unique.setdefault(item.term_id, item)
        return list(unique.values())[:limit]
