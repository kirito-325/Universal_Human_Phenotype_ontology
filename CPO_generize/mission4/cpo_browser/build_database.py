#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "cpo_browser.sqlite3"
DEFAULT_HPO_OBO = Path(
    "/path/to/workspace/HPO_expand/gene-to-phenotype/data/HPO/script/hp.obo"
)
DEFAULT_MPO_OBO = Path(
    "/path/to/workspace/CPO_generize/03_MPO_link/data/processed/mpo_clean/mpo_clean.obo"
)


SCHEMA = """
CREATE TABLE terms (
    side TEXT NOT NULL,
    term_id TEXT NOT NULL,
    label TEXT NOT NULL,
    definition TEXT NOT NULL,
    synonyms TEXT NOT NULL,
    node_type TEXT NOT NULL,
    depth INTEGER NOT NULL,
    is_normal_branch INTEGER NOT NULL DEFAULT 0,
    self_relation_count INTEGER NOT NULL DEFAULT 0,
    inherited_assertion_count INTEGER NOT NULL DEFAULT 0,
    inherited_complete_cpo_count INTEGER NOT NULL DEFAULT 0,
    profile_mode TEXT NOT NULL,
    inheritance_status TEXT NOT NULL,
    status_json TEXT NOT NULL,
    parent_count INTEGER NOT NULL DEFAULT 0,
    child_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (side, term_id)
);
CREATE TABLE edges (
    side TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    PRIMARY KEY (side, parent_id, child_id)
);
CREATE TABLE self_relations (
    side TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    term_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    complete_cpo INTEGER NOT NULL,
    cl_id TEXT NOT NULL,
    cl_label TEXT NOT NULL,
    cell_text TEXT NOT NULL,
    go_id TEXT NOT NULL,
    go_label TEXT NOT NULL,
    process_text TEXT NOT NULL,
    quality_text TEXT NOT NULL,
    evidence_tier TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    species TEXT NOT NULL,
    confidence TEXT NOT NULL,
    supporting_quote TEXT NOT NULL,
    section TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_batch TEXT NOT NULL,
    PRIMARY KEY (side, relation_id)
);
CREATE TABLE inherited_relations (
    side TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    term_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    complete_cpo INTEGER NOT NULL,
    cl_id TEXT NOT NULL,
    cl_label TEXT NOT NULL,
    cell_text TEXT NOT NULL,
    go_id TEXT NOT NULL,
    go_label TEXT NOT NULL,
    process_text TEXT NOT NULL,
    quality_text TEXT NOT NULL,
    support_leaf_count INTEGER NOT NULL,
    support_relation_count INTEGER NOT NULL,
    minimum_distance INTEGER NOT NULL,
    maximum_distance INTEGER NOT NULL,
    eligible_descendant_leaf_count INTEGER NOT NULL,
    linked_descendant_leaf_count INTEGER NOT NULL,
    support_ratio REAL NOT NULL,
    PRIMARY KEY (side, assertion_id)
);
CREATE TABLE provenance (
    side TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    target_term_id TEXT NOT NULL,
    source_leaf_id TEXT NOT NULL,
    source_relation_id TEXT NOT NULL,
    source_scope TEXT NOT NULL,
    distance INTEGER NOT NULL,
    shortest_path_count INTEGER NOT NULL,
    canonical_path TEXT NOT NULL
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def integer(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def decimal(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def truth(value: str | None) -> int:
    return int((value or "").strip().lower() == "true")


def batches(values: Iterable[tuple], size: int = 5000) -> Iterator[list[tuple]]:
    batch: list[tuple] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def insert_batches(connection: sqlite3.Connection, sql: str, values: Iterable[tuple]) -> None:
    for batch in batches(values):
        connection.executemany(sql, batch)


def parse_obo(path: Path) -> tuple[str, dict[str, dict[str, object]]]:
    version = ""
    records: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None

    def emit() -> None:
        if current and current.get("id") and not current.get("obsolete"):
            records[str(current["id"])] = current.copy()

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not version and line.startswith("data-version: "):
                version = line.removeprefix("data-version: ")
            if line.startswith("["):
                emit()
                current = {"parents": [], "synonyms": [], "definition": "", "obsolete": False} if line == "[Term]" else None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split()[0])
            elif line.startswith("def: "):
                match = re.match(r'def: "((?:[^"\\]|\\.)*)"', line)
                current["definition"] = match.group(1).replace('\\"', '"') if match else line[5:]
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: "((?:[^"\\]|\\.)*)"', line)
                if match:
                    current["synonyms"].append(match.group(1).replace('\\"', '"'))
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        emit()
    return version, records


def hpo_relation_id(row: dict[str, str]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "hpo_relation_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def status_values(side: str, path: Path) -> Iterator[tuple]:
    prefix = "hpo" if side == "hpo" else "mp"
    for row in rows(path):
        term_id = row[f"{prefix}_id"]
        yield (
            side,
            term_id,
            row[f"{prefix}_label"],
            "",
            "",
            row.get("node_type", ""),
            integer(row.get("depth")),
            truth(row.get("is_normal_branch")),
            integer(row.get("self_relation_count")),
            integer(row.get("inherited_assertion_count")),
            integer(row.get("inherited_complete_cpo_count")),
            row.get("profile_mode", ""),
            row.get("inheritance_status", ""),
            json.dumps(row, ensure_ascii=False, separators=(",", ":")),
        )


def self_values(side: str, path: Path) -> Iterator[tuple]:
    for row in rows(path):
        if side == "hpo":
            relation_id = hpo_relation_id(row)
            term_id = row["hpo_id"]
            quote = row.get("representative_quote", "")
            source_id = row.get("record_id", "")
            title = ""
            scope = row.get("link_level", "")
            quality = ""
            tier = ""
            confidence = ""
            batch = "hpo_literature_unified"
        else:
            relation_id = row["relation_id"]
            term_id = row["mp_id"]
            quote = row.get("supporting_quote", "")
            source_id = row.get("pmid", "")
            title = row.get("article_title", "")
            scope = row.get("relation_scope", "")
            quality = row.get("quality_text", "")
            tier = row.get("evidence_tier", "")
            confidence = row.get("confidence", "")
            batch = row.get("source_batch", "")
        cl_id = row.get("cl_id", "")
        go_id = row.get("go_id", "")
        yield (
            side, relation_id, term_id, scope, row.get("relation_type", ""),
            row.get("direction", ""), int(bool(cl_id and go_id)), cl_id,
            row.get("cl_label", ""), row.get("cell_text", ""), go_id,
            row.get("go_label", ""), row.get("process_text", ""), quality,
            tier, row.get("evidence_type", row.get("evidence_types", "")),
            row.get("species", ""), confidence, quote, row.get("section", ""),
            source_id, title, row.get("source_url", ""), batch,
        )


def inherited_values(side: str, path: Path) -> Iterator[tuple]:
    prefix = "hpo" if side == "hpo" else "mp"
    for row in rows(path):
        yield (
            side, row["assertion_id"], row[f"target_{prefix}_id"],
            row.get("link_level", row.get("relation_scope", "")),
            row.get("relation_type", ""), row.get("direction", ""),
            truth(row.get("complete_cpo")), row.get("cl_id", ""),
            row.get("cl_label", ""), row.get("representative_cell_text", ""),
            row.get("go_id", ""), row.get("go_label", ""),
            row.get("representative_process_text", ""),
            row.get("representative_quality_text", ""),
            integer(row.get("support_leaf_count")),
            integer(row.get("support_relation_count")),
            integer(row.get("minimum_distance")), integer(row.get("maximum_distance")),
            integer(row.get("eligible_descendant_leaf_count")),
            integer(row.get("linked_descendant_leaf_count")),
            decimal(row.get("support_ratio_among_eligible_leaves")),
        )


def provenance_values(side: str, path: Path) -> Iterator[tuple]:
    prefix = "hpo" if side == "hpo" else "mp"
    for row in rows(path):
        yield (
            side, row["assertion_id"], row[f"target_{prefix}_id"],
            row[f"source_leaf_{prefix}_id"], row["source_relation_id"],
            row.get("source_link_level", row.get("source_relation_scope", "")),
            integer(row.get("distance")), integer(row.get("shortest_path_count")),
            row.get("canonical_path", ""),
        )


def build(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="cpo_browser_", suffix=".sqlite3", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;" + SCHEMA)
        sides = {
            "hpo": {
                "obo": args.hpo_obo,
                "status": ROOT / "data/hpo_cpo/inherited/hpo_cpo_inheritance_term_status.tsv",
                "self": ROOT / "data/hpo_cpo/hpo_cpo_relations.tsv",
                "inherited": ROOT / "data/hpo_cpo/inherited/hpo_cpo_inherited_assertions.tsv",
                "provenance": ROOT / "data/hpo_cpo/inherited/hpo_cpo_inheritance_provenance.tsv",
                "root": "HP:0000118",
            },
            "mpo": {
                "obo": args.mpo_obo,
                "status": ROOT / "data/mpo_cpo/inherited/mpo_cpo_inheritance_term_status.tsv",
                "self": ROOT / "data/mpo_cpo/final/mpo_cpo_relations.tsv",
                "inherited": ROOT / "data/mpo_cpo/inherited/mpo_cpo_inherited_assertions.tsv",
                "provenance": ROOT / "data/mpo_cpo/inherited/mpo_cpo_inheritance_provenance.tsv",
                "root": "MP:0000001",
            },
        }
        for side, files in sides.items():
            print(f"[{side.upper()}] terms and ontology", flush=True)
            insert_batches(connection, "INSERT INTO terms(side,term_id,label,definition,synonyms,node_type,depth,is_normal_branch,self_relation_count,inherited_assertion_count,inherited_complete_cpo_count,profile_mode,inheritance_status,status_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", status_values(side, files["status"]))
            version, records = parse_obo(files["obo"])
            valid = {row[0] for row in connection.execute("SELECT term_id FROM terms WHERE side=?", (side,))}
            connection.executemany(
                "UPDATE terms SET definition=?, synonyms=? WHERE side=? AND term_id=?",
                ((str(record.get("definition", "")), " | ".join(record.get("synonyms", [])), side, term_id) for term_id, record in records.items() if term_id in valid),
            )
            edge_values = (
                (side, parent, term_id)
                for term_id, record in records.items() if term_id in valid
                for parent in record.get("parents", []) if parent in valid
            )
            insert_batches(connection, "INSERT INTO edges VALUES (?,?,?)", edge_values)
            connection.execute("INSERT INTO metadata VALUES (?,?)", (f"{side}_root", files["root"]))
            connection.execute("INSERT INTO metadata VALUES (?,?)", (f"{side}_ontology_version", version))

            print(f"[{side.upper()}] self CPO relations", flush=True)
            insert_batches(connection, "INSERT INTO self_relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self_values(side, files["self"]))
            print(f"[{side.upper()}] inherited CPO relations", flush=True)
            insert_batches(connection, "INSERT INTO inherited_relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", inherited_values(side, files["inherited"]))
            print(f"[{side.upper()}] inheritance provenance", flush=True)
            insert_batches(connection, "INSERT INTO provenance VALUES (?,?,?,?,?,?,?,?,?)", provenance_values(side, files["provenance"]))
            connection.commit()

        print("Building indexes", flush=True)
        connection.executescript("""
        CREATE INDEX edge_children ON edges(side, parent_id, child_id);
        CREATE INDEX edge_parents ON edges(side, child_id, parent_id);
        CREATE INDEX self_by_term ON self_relations(side, term_id, direction, complete_cpo);
        CREATE INDEX inherited_by_term ON inherited_relations(side, term_id, direction, complete_cpo);
        CREATE INDEX provenance_by_assertion ON provenance(side, assertion_id, source_leaf_id);
        UPDATE terms SET parent_count=(SELECT count(*) FROM edges WHERE edges.side=terms.side AND edges.child_id=terms.term_id), child_count=(SELECT count(*) FROM edges WHERE edges.side=terms.side AND edges.parent_id=terms.term_id);
        """)
        for side in ("hpo", "mpo"):
            term_count = connection.execute("SELECT count(*) FROM terms WHERE side=?", (side,)).fetchone()[0]
            linked_term_count = connection.execute(
                "SELECT count(*) FROM terms WHERE side=? AND self_relation_count>0",
                (side,),
            ).fetchone()[0]
            self_count = connection.execute("SELECT count(*) FROM self_relations WHERE side=?", (side,)).fetchone()[0]
            complete_self_count = connection.execute(
                "SELECT count(*) FROM self_relations WHERE side=? AND complete_cpo=1",
                (side,),
            ).fetchone()[0]
            inherited_count = connection.execute("SELECT count(*) FROM inherited_relations WHERE side=?", (side,)).fetchone()[0]
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                (f"{side}_term_count", str(term_count)),
                (f"{side}_linked_term_count", str(linked_term_count)),
                (f"{side}_self_count", str(self_count)),
                (f"{side}_complete_self_count", str(complete_self_count)),
                (f"{side}_inherited_count", str(inherited_count)),
            ])
        connection.execute("PRAGMA optimize")
        connection.commit()
        connection.close()
        temporary.replace(args.output)
        print(f"Created {args.output}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CPO ontology browser SQLite index")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hpo-obo", type=Path, default=DEFAULT_HPO_OBO)
    parser.add_argument("--mpo-obo", type=Path, default=DEFAULT_MPO_OBO)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
