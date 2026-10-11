"""Local, versioned HPO disease context. Annotation scope is never silently inherited.

Format reference: https://obophenotype.github.io/human-phenotype-ontology/annotations/phenotype_hpoa/
The index stores disease annotations, including negative qualifiers, without interpreting
them as cellular mechanisms. Ancestor matches are explicitly broader background only.
"""

import csv
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from .ontology import digest_file, quoted


def hpo_terms(path):
    stanza = {}
    active = False

    def convert():
        term_id = stanza.get("id", [""])[0]
        if not active or not term_id.startswith("HP:"):
            return None
        return {"term_id": term_id, "label": stanza.get("name", [term_id])[0],
                "definition": quoted(stanza.get("def", [""])[0]),
                "synonyms": [quoted(s) for s in stanza.get("synonym", [])],
                "parents": [s.split()[0] for s in stanza.get("is_a", [])],
                "alt_ids": stanza.get("alt_id", []),
                "obsolete": stanza.get("is_obsolete") == ["true"]}

    with Path(path).open() as stream:
        for raw in stream:
            line = raw.rstrip()
            if line.startswith("["):
                if term := convert():
                    yield term
                stanza, active = {}, line == "[Term]"
            elif active and ": " in line:
                key, value = line.split(": ", 1)
                stanza.setdefault(key, []).append(value)
        if term := convert():
            yield term


def excluded(row, excluded_frequency_ids=()):
    qualifier = re.split(r"[;,|\s]+", row.get("qualifier", "").upper())
    frequency = row.get("frequency", "").strip()
    return ("NOT" in qualifier or frequency in excluded_frequency_ids
            or bool(re.fullmatch(r"0+(?:\.0+)?%?", frequency)) or bool(re.fullmatch(r"0+/[1-9]\d*", frequency)))


def build_context_index(hpo: Path, annotations: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".hpo-context-")
    os.close(fd)
    metadata = {"format_version": "1", "hpo": {"path": str(hpo.resolve()), "sha256": digest_file(hpo)},
                "annotations": {"path": str(annotations.resolve()), "sha256": digest_file(annotations)}}
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE terms (id TEXT PRIMARY KEY, payload TEXT);
                CREATE TABLE aliases (alias TEXT PRIMARY KEY, id TEXT);
                CREATE TABLE annotations (hpo_id TEXT, disease_id TEXT, disease_name TEXT, excluded INTEGER, payload TEXT);
                CREATE TABLE metadata (payload TEXT);
            """)
            term_count = 0
            excluded_frequency_ids = set()
            for term in hpo_terms(hpo):
                db.execute("INSERT INTO terms VALUES (?,?)", (term["term_id"], json.dumps(term)))
                db.executemany("INSERT INTO aliases VALUES (?,?)", [(s, term["term_id"]) for s in term["alt_ids"]])
                term_count += 1
                if term["label"].casefold() in {"excluded", "excluded (0%)"}:
                    excluded_frequency_ids.update([term["term_id"], *term["alt_ids"]])
            annotation_count = 0
            with annotations.open() as stream:
                header = None
                for line in stream:
                    if line.startswith("#"):
                        if ":" in line:
                            key, value = line[1:].split(":", 1)
                            metadata["annotations"][key.strip()] = value.strip()
                        continue
                    header = line.rstrip("\n").split("\t")
                    break
                required = {"database_id", "disease_name", "qualifier", "hpo_id", "reference", "evidence"}
                if not header or not required <= set(header):
                    raise ValueError("unsupported phenotype.hpoa header")
                for row in csv.DictReader(stream, fieldnames=header, delimiter="\t"):
                    if not row.get("hpo_id", "").startswith("HP:"):
                        continue
                    db.execute("INSERT INTO annotations VALUES (?,?,?,?,?)",
                               (row["hpo_id"], row["database_id"], row["disease_name"],
                                int(excluded(row, excluded_frequency_ids)), json.dumps(row)))
                    annotation_count += 1
            if not term_count or not annotation_count:
                raise ValueError("empty HPO context source")
            db.executescript("CREATE INDEX by_hpo ON annotations(hpo_id); CREATE INDEX by_disease ON annotations(disease_id);")
            metadata["counts"] = {"terms": term_count, "annotations": annotation_count}
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return metadata


class HPOContextIndex:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with closing(self.connect()) as db:
            self.metadata = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
        self.sha256 = digest_file(self.path)

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def lookup(self, phenotype_id, *, ancestor_depth=3, disease_query="", offset=0, limit=10):
        if not re.fullmatch(r"HP:\d{7}", phenotype_id):
            raise ValueError("HPO lookup requires an HP identifier")
        if not 0 <= ancestor_depth <= 3 or not 0 <= offset or not 1 <= limit <= 100:
            raise ValueError("invalid context pagination/depth")
        with closing(self.connect()) as db:
            alias = db.execute("SELECT id FROM aliases WHERE alias=?", (phenotype_id,)).fetchone()
            canonical = alias[0] if alias else phenotype_id
            row = db.execute("SELECT payload FROM terms WHERE id=?", (canonical,)).fetchone()
            if row is None:
                raise ValueError("unknown_hpo_id")
            term = json.loads(row[0])
            if term["obsolete"]:
                raise ValueError("obsolete_hpo_id")
            relatives = {canonical: (0, term)}
            queue = [(canonical, 0)]
            for current, depth in queue:
                if depth >= ancestor_depth:
                    continue
                for parent in relatives[current][1]["parents"]:
                    if parent in relatives:
                        continue
                    found = db.execute("SELECT payload FROM terms WHERE id=?", (parent,)).fetchone()
                    if found:
                        relatives[parent] = (depth + 1, json.loads(found[0]))
                        queue.append((parent, depth + 1))
            positive, negative = [], []
            for hpo_id, (depth, _) in relatives.items():
                for raw, is_excluded in db.execute("SELECT payload, excluded FROM annotations WHERE hpo_id=?", (hpo_id,)):
                    annotation = json.loads(raw)
                    if disease_query and disease_query.casefold() not in (
                        annotation["database_id"] + " " + annotation["disease_name"]
                    ).casefold():
                        continue
                    annotation.update({"match_scope": "exact" if depth == 0 else "ancestor_background",
                                       "ancestor_distance": depth, "excluded": bool(is_excluded)})
                    (negative if is_excluded else positive).append(annotation)
            positive.sort(key=lambda a: (a["ancestor_distance"], a["database_id"], a["hpo_id"], a["reference"]))
            groups = {}
            for annotation in positive:
                group = groups.setdefault(annotation["database_id"], {
                    "disease_id": annotation["database_id"], "disease_name": annotation["disease_name"],
                    "closest_match": annotation["match_scope"], "annotations": [],
                })
                group["annotations"].append(annotation)
            selected = list(groups.values())[offset:offset + limit]
            for disease in selected:
                disease["other_phenotypes"] = []
                for raw, term_raw in db.execute(
                    "SELECT a.payload,t.payload FROM annotations a LEFT JOIN terms t ON t.id=a.hpo_id "
                    "WHERE a.disease_id=? AND a.excluded=0 ORDER BY a.hpo_id LIMIT 6", (disease["disease_id"],)
                ):
                    annotation = json.loads(raw)
                    disease["other_phenotypes"].append({
                        "hpo_id": annotation["hpo_id"], "label": json.loads(term_raw)["label"] if term_raw else "",
                        "frequency": annotation.get("frequency", ""), "onset": annotation.get("onset", ""),
                        "aspect": annotation.get("aspect", ""),
                    })
                disease["annotation_count"] = len(disease["annotations"])
                disease["annotations"] = disease["annotations"][:20]
            return {"requested_id": phenotype_id, "term": term,
                    "ancestors": [{"distance": depth, **node} for key, (depth, node) in relatives.items() if key != canonical],
                    "exact_positive_annotation_count": sum(a["match_scope"] == "exact" for a in positive),
                    "related_disease_count": len(groups), "diseases": selected,
                    "offset": offset, "next_offset": offset + limit if offset + limit < len(groups) else None,
                    "excluded_annotation_count": len(negative), "excluded_annotations": negative[:limit],
                    "source_metadata": self.metadata, "index_sha256": self.sha256,
                    "scope_note": "Ancestor disease matches are broader search context, not annotations of the exact target. "
                                  "Disease annotations are not proof of a cellular mechanism; excluded rows are not positive links."}
