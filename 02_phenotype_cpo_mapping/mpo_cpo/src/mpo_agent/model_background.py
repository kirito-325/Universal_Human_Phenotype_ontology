"""Versioned MP terms and genotype-level MGI phenotype annotations."""

import csv
import json
import os
import re
import sqlite3
import tempfile
from collections import deque
from contextlib import closing
from pathlib import Path

from .ontology import digest_file, quoted
from .schemas import MouseModel


def mp_terms(path):
    stanza, active = {}, False

    def term():
        key = stanza.get("id", [""])[0]
        if not active or not re.fullmatch(r"MP:\d{7}", key):
            return None
        return {"term_id": key, "label": stanza.get("name", [key])[0],
                "definition": quoted(stanza.get("def", [""])[0]),
                "synonyms": [quoted(item) for item in stanza.get("synonym", [])],
                "parents": [item.split()[0] for item in stanza.get("is_a", [])],
                "alt_ids": stanza.get("alt_id", []),
                "obsolete": stanza.get("is_obsolete") == ["true"]}

    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("["):
                if value := term():
                    yield value
                stanza, active = {}, line == "[Term]"
            elif active and ": " in line:
                key, value = line.split(": ", 1)
                stanza.setdefault(key, []).append(value)
        if value := term():
            yield value


def split_values(value):
    return list(dict.fromkeys(item.strip() for item in value.split("|") if item.strip()))


def build_mp_context(mp, annotations, output):
    """Read the seven-column ALL-genotypes report, including conditional mutants."""
    mp, annotations, output = Path(mp), Path(annotations), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format_version": "1", "sources": {
        "MP": {"path": str(mp.resolve()), "sha256": digest_file(mp)},
        "MGI_PhenoGenoMP": {"path": str(annotations.resolve()), "sha256": digest_file(annotations),
            "url": "https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt"}}}
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".mp-context-")
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE terms (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE aliases (alias TEXT PRIMARY KEY, id TEXT NOT NULL);
                CREATE TABLE parents (child TEXT, parent TEXT);
                CREATE TABLE models (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE annotations (mp_id TEXT, model_id TEXT, payload TEXT NOT NULL);
                CREATE TABLE metadata (payload TEXT NOT NULL);
            """)
            term_count = 0
            for term in mp_terms(mp):
                db.execute("INSERT INTO terms VALUES (?,?)", (term["term_id"], json.dumps(term)))
                db.executemany("INSERT INTO aliases VALUES (?,?)", [(key, term["term_id"]) for key in term["alt_ids"]])
                db.executemany("INSERT INTO parents VALUES (?,?)", [(term["term_id"], key) for key in term["parents"]])
                term_count += 1
            annotation_count = 0
            with annotations.open(encoding="utf-8", newline="") as stream:
                for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
                    if not row or row[0].startswith("#"):
                        continue
                    if len(row) != 7 or not re.fullmatch(r"MP:\d{7}", row[3]):
                        raise ValueError(f"Unsupported MGI_PhenoGenoMP row {line}: expected seven columns")
                    composition, alleles, background, mp_id, references, markers, model_id = row
                    if not re.fullmatch(r"MGI:\d+", model_id):
                        raise ValueError(f"Invalid genotype accession at row {line}")
                    pmids = split_values(references)
                    if any(not re.fullmatch(r"\d+", key) for key in pmids):
                        raise ValueError(f"Non-PubMed reference at row {line}; resolve it before indexing")
                    model = {"model_id": model_id, "allelic_composition": composition,
                             "allele_symbols": split_values(alleles), "genetic_background": background,
                             "marker_ids": split_values(markers)}
                    previous = db.execute("SELECT payload FROM models WHERE id=?", (model_id,)).fetchone()
                    if previous and json.loads(previous[0]) != model:
                        raise ValueError(f"Conflicting genotype metadata for {model_id}")
                    db.execute("INSERT OR IGNORE INTO models VALUES (?,?)", (model_id, json.dumps(model)))
                    alias = db.execute("SELECT id FROM aliases WHERE alias=?", (mp_id,)).fetchone()
                    canonical = alias[0] if alias else mp_id
                    record = {"mp_id": canonical, "reported_mp_id": mp_id, "pmids": pmids, "source_row": line}
                    db.execute("INSERT INTO annotations VALUES (?,?,?)", (canonical, model_id, json.dumps(record)))
                    annotation_count += 1
            if not term_count or not annotation_count:
                raise ValueError("Empty MP or MGI annotation source")
            db.executescript("""
                CREATE INDEX annotations_by_mp ON annotations(mp_id,model_id);
                CREATE INDEX children_by_parent ON parents(parent);
            """)
            metadata["counts"] = {"terms": term_count, "annotations": annotation_count,
                                  "models": db.execute("SELECT COUNT(*) FROM models").fetchone()[0]}
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return metadata


class MPContextIndex:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with closing(self.connect()) as db:
            self.metadata = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
        self.sha256 = digest_file(self.path)

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def lookup(self, mp_id, *, include_descendants=False, offset=0, limit=20):
        if not re.fullmatch(r"MP:\d{7}", mp_id) or offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid MP identifier or pagination")
        with closing(self.connect()) as db:
            alias = db.execute("SELECT id FROM aliases WHERE alias=?", (mp_id,)).fetchone()
            canonical = alias[0] if alias else mp_id
            found = db.execute("SELECT payload FROM terms WHERE id=?", (canonical,)).fetchone()
            if not found:
                raise ValueError("unknown_mp_id")
            term = json.loads(found[0])
            if term["obsolete"]:
                raise ValueError("obsolete_mp_id")
            selected, queue = {canonical}, deque([canonical])
            if include_descendants:
                while queue:
                    for child, in db.execute("SELECT child FROM parents WHERE parent=?", (queue.popleft(),)):
                        if child not in selected:
                            selected.add(child)
                            queue.append(child)
            # A connection-local relation avoids SQLite parameter limits for broad MP terms.
            db.execute("CREATE TEMP TABLE selected_terms (id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO selected_terms VALUES (?)", [(key,) for key in sorted(selected)])
            ids = [key for key, in db.execute(
                "SELECT DISTINCT model_id FROM annotations JOIN selected_terms ON mp_id=id ORDER BY model_id")]
            models = []
            for key in ids[offset:offset + limit]:
                model = json.loads(db.execute("SELECT payload FROM models WHERE id=?", (key,)).fetchone()[0])
                records = [json.loads(raw) for raw, in db.execute(
                    "SELECT payload FROM annotations JOIN selected_terms ON mp_id=id WHERE model_id=? ORDER BY mp_id,payload", (key,))]
                for record in records:
                    record["match_scope"] = "exact" if record["mp_id"] == canonical else "descendant"
                model.update(annotations=records, annotation_pmids=sorted({pmid for record in records for pmid in record["pmids"]}))
                models.append(MouseModel.model_validate(model).model_dump())
            return {"requested_id": mp_id, "term": term, "models": models, "total_models": len(ids),
                    "include_descendants": include_descendants, "offset": offset,
                    "next_offset": offset + limit if offset + limit < len(ids) else None,
                    "source_metadata": self.metadata, "index_sha256": self.sha256,
                    "scope_note": "Model phenotype annotations establish an observed MP association, not a cellular mechanism. Descendant matches are explicit."}

    def all_models(self, mp_id, *, include_descendants=False):
        first = self.lookup(mp_id, include_descendants=include_descendants, limit=100)
        records, offset = list(first["models"]), first["next_offset"]
        while offset is not None:
            page = self.lookup(mp_id, include_descendants=include_descendants, offset=offset, limit=100)
            records.extend(page["models"])
            offset = page["next_offset"]
        return first["term"], [MouseModel.model_validate(row) for row in records]
