"""Freeze selected pairs and isolated agent inputs from versioned local sources."""

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
INDEXES = {
    "HP": WORKSPACE / "02_phenotype_cpo_mapping/hpo_cpo/data/hpo_context.sqlite",
    "MP": WORKSPACE / "02_phenotype_cpo_mapping/mpo_cpo/data/mp_context.sqlite",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def jsonl_text(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def tsv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def term(db, term_id):
    row = db.execute("SELECT payload FROM terms WHERE id=?", (term_id,)).fetchone()
    if row is None:
        raise ValueError("unknown_primary_term: " + term_id)
    value = json.loads(row[0])
    if value["obsolete"]:
        raise ValueError("obsolete_term: " + term_id)
    return value


def build():
    spec = json.loads((ROOT / "selection_spec.json").read_text())
    literature = json.loads((ROOT / "reference/literature.json").read_text())
    literature_ids = {row["literature_id"] for row in literature}
    source = ROOT / spec["mapping_source"]["path"]
    if digest(source) != spec["mapping_source"]["sha256"]:
        raise ValueError("mapping_source_checksum_changed")
    rows = list(csv.DictReader((line for line in source.read_text().splitlines()
                               if not line.startswith("#")), delimiter="\t"))
    by_pair = collections.defaultdict(list)
    for row in rows:
        by_pair[row["object_id"], row["subject_id"]].append(row)
    if collections.Counter(row["group"] for row in spec["pairs"]) != {
        "curated_exact": 5, "curated_hierarchical": 3, "mechanistic_candidate": 2,
    }:
        raise ValueError("selection_must_have_5_exact_3_hierarchical_2_candidates")
    if len({row["pair_id"] for row in spec["pairs"]}) != 10:
        raise ValueError("duplicate_pair_id")
    if len({(row["hp_id"], row["mp_id"]) for row in spec["pairs"]}) != 10:
        raise ValueError("duplicate_term_pair")

    pairs, models, selected_mappings, terms, metadata = [], [], [], {}, {}
    with closing(sqlite3.connect(INDEXES["HP"].as_uri() + "?mode=ro", uri=True)) as hp_db, \
            closing(sqlite3.connect(INDEXES["MP"].as_uri() + "?mode=ro", uri=True)) as mp_db:
        for prefix, db in (("HP", hp_db), ("MP", mp_db)):
            metadata[prefix] = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
        for chosen in spec["pairs"]:
            row = dict(chosen)
            hp, mp = term(hp_db, row["hp_id"]), term(mp_db, row["mp_id"])
            terms[hp["term_id"]], terms[mp["term_id"]] = hp, mp
            row.update(hp_term=hp, mp_term=mp)
            matches = by_pair[row["hp_id"], row["mp_id"]]
            if row["group"] == "mechanistic_candidate":
                if matches or row["expected_predicate_mp_to_hp"] is not None or row["gold_status"] != "unconfirmed_candidate":
                    raise ValueError("candidate_must_not_be_presented_as_a_curated_positive")
                if not set(row["literature_ids"]) <= literature_ids:
                    raise ValueError("unknown_literature_id")
                row["mapping_evidence"] = None
            else:
                matches = [m for m in matches if m["predicate_id"] == row["expected_predicate_mp_to_hp"]
                           and m["mapping_justification"] == "semapv:ManualMappingCuration"]
                if len(matches) != 1 or float(matches[0]["confidence"]) < 0.9 or not matches[0]["author_id"]:
                    raise ValueError("missing_unambiguous_high_confidence_manual_mapping")
                if row["group"] == "curated_exact" and matches[0]["predicate_id"] != "skos:exactMatch":
                    raise ValueError("exact_group_requires_exactMatch")
                if row["group"] == "curated_hierarchical":
                    expected_direction = {"skos:narrowMatch": "HP_narrower_than_MP",
                                          "skos:broadMatch": "MP_narrower_than_HP"}
                    if row["hierarchy_direction"] != expected_direction[matches[0]["predicate_id"]]:
                        raise ValueError("incorrect_MP_to_HP_mapping_direction")
                row["mapping_evidence"] = dict(matches[0], source_url=spec["mapping_source"]["url"])
                selected_mappings.append(matches[0])
            if witness := row.get("ontology_witness"):
                db = hp_db if witness["ontology"] == "HP" else mp_db
                child, parent = term(db, witness["child"]), term(db, witness["parent"])
                if parent["term_id"] not in child["parents"]:
                    raise ValueError("ontology_witness_is_not_a_direct_parent")
                terms[parent["term_id"]] = parent
            model_ids = [key for key, in mp_db.execute(
                "SELECT DISTINCT model_id FROM annotations WHERE mp_id=? ORDER BY model_id", (row["mp_id"],))]
            if not model_ids:
                raise ValueError("selected_MP_has_no_exactly_annotated_model")
            row.update(mp_exact_model_count=len(model_ids), mp_model_ids=model_ids)
            for model_id in model_ids:
                model = json.loads(mp_db.execute("SELECT payload FROM models WHERE id=?", (model_id,)).fetchone()[0])
                annotations = [dict(json.loads(raw), match_scope="exact") for raw, in mp_db.execute(
                    "SELECT payload FROM annotations WHERE mp_id=? AND model_id=? ORDER BY payload",
                    (row["mp_id"], model_id))]
                model.update(annotations=annotations,
                             annotation_pmids=sorted({p for a in annotations for p in a["pmids"]}))
                models.append({"pair_id": row["pair_id"], "mp_id": row["mp_id"], "model": model})
            a, b = (set(re.findall(r"[a-z0-9]+", t["label"].casefold())) for t in (hp, mp))
            row["label_token_jaccard"] = round(len(a & b) / len(a | b), 4)
            row["lexical_note"] = "Descriptive label-token overlap only; no embedding/semantic baseline has been run."
            pairs.append(row)

    outputs = {
        "pairs.jsonl": jsonl_text(pairs),
        "reference/ontology_terms.jsonl": jsonl_text([terms[key] for key in sorted(terms)]),
        "reference/models.jsonl": jsonl_text(models),
        "reference/manual_mapping_rows.tsv": tsv_text(selected_mappings, list(rows[0])),
        "inputs/hpo_terms.jsonl": jsonl_text([
            {"phenotype_id": key, **{field: terms[key][field] for field in ("label", "definition", "synonyms")}}
            for key in sorted({row["hp_id"] for row in pairs})]),
        "inputs/mpo_terms.jsonl": jsonl_text([
            {"phenotype_id": key, "include_descendants": False}
            for key in sorted({row["mp_id"] for row in pairs})]),
    }
    flat = [{"pair_id": row["pair_id"], "group": row["group"], "hp_id": row["hp_id"],
             "hp_label": row["hp_term"]["label"], "mp_id": row["mp_id"], "mp_label": row["mp_term"]["label"],
             "expected_predicate_mp_to_hp": row["expected_predicate_mp_to_hp"] or "",
             "gold_status": row["gold_status"], "mp_exact_model_count": row["mp_exact_model_count"],
             "selection_reason": row["selection_reason"]} for row in pairs]
    outputs["pairs.tsv"] = tsv_text(flat, list(flat[0]))
    manifest = {
        "selection_id": spec["selection_id"], "created_on": spec["created_on"], "status": spec["status"],
        "counts": {"pairs": len(pairs), "groups": dict(collections.Counter(p["group"] for p in pairs)),
                   "unique_HP_terms": len({p["hp_id"] for p in pairs}), "unique_MP_terms": len({p["mp_id"] for p in pairs}),
                   "MP_model_term_records": len(models), "unique_MGI_models": len({m["model"]["model_id"] for m in models})},
        "mapping_source": spec["mapping_source"], "mp_scope": spec["mp_scope"],
        "input_index_snapshots": {key: {"path": str(path.relative_to(WORKSPACE)), "sha256": digest(path),
                                       "metadata": metadata[key]} for key, path in INDEXES.items()},
        "selection_spec_sha256": digest(ROOT / "selection_spec.json"),
        "literature_sha256": digest(ROOT / "reference/literature.json"),
        "generator_sha256": digest(Path(__file__)),
        "artifact_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in outputs.items()},
        "evaluation_note": "Eight curated phenotype mappings; two unconfirmed mechanism candidates. No live agent research or alignment scoring has been run. Inputs exclude pair labels, expected mechanisms and literature hints.",
    }
    outputs["manifest.json"] = json_text(manifest)
    return outputs, manifest["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify sources and derived files without rewriting them")
    args = parser.parse_args()
    outputs, counts = build()
    for name, content in outputs.items():
        path = ROOT / name
        if args.check:
            if not path.is_file() or path.read_bytes() != content.encode():
                raise ValueError("artifact_missing_or_changed: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(json_text({"status": "verified" if args.check else "prepared", **counts}), end="")


if __name__ == "__main__":
    main()
