#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping


SOURCE_CONFIG = (
    ("leaf", "leaf", "mgi_all_eligible_leaves_20260823"),
    ("middle", "internal", "mgi_middle_layer_direct_20260831"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", newline=""), Path(name)


def iter_tsv(
    path: Path,
    source_batch: str,
    node_type: str,
) -> tuple[tuple[str, ...], Iterator[dict[str, str]]]:
    handle = path.open(encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    fields = tuple(reader.fieldnames or ())

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {"source_batch": source_batch, "node_type": node_type, **row}
        finally:
            handle.close()

    return fields, rows()


def iter_jsonl(
    path: Path,
    source_batch: str,
    node_type: str,
) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield {"source_batch": source_batch, "node_type": node_type, **value}


def relation_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["mp_id"]), str(row["pmid"]), str(row["relation_id"])


def term_key(row: Mapping[str, object]) -> str:
    return str(row["mp_id"])


def build_dataset(root: Path) -> dict[str, object]:
    final_root = root / "final"
    relation_tsv_output = final_root / "mpo_cpo_relations.tsv"
    relation_jsonl_output = final_root / "mpo_cpo_relations.jsonl"
    term_output = final_root / "mpo_cpo_term_status.tsv"
    source_manifests: dict[str, dict[str, object]] = {}
    relation_tsv_sources: list[Iterator[dict[str, str]]] = []
    relation_json_sources: list[Iterator[dict[str, object]]] = []
    term_sources: list[Iterator[dict[str, str]]] = []
    relation_fields: tuple[str, ...] | None = None
    term_fields: tuple[str, ...] | None = None
    source_term_ids: dict[str, set[str]] = {}

    for source_name, node_type, batch_id in SOURCE_CONFIG:
        source_root = root / source_name
        manifest_path = source_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("error_count", -1)) != 0:
            raise ValueError(f"source contains pipeline errors: {source_name}")
        source_manifests[source_name] = {
            "batch_id": batch_id,
            "node_type": node_type,
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256(manifest_path),
            "sampled_term_count": int(manifest["sampled_term_count"]),
            "relation_count": int(manifest["relation_count"]),
            "direct_relation_count": int(manifest["direct_relation_count"]),
            "mechanism_relation_count": int(manifest["mechanism_relation_count"]),
            "status_counts": manifest["status_counts"],
        }

        source_terms_fields, source_terms = iter_tsv(
            source_root / "term_status.tsv", batch_id, node_type
        )
        if term_fields is None:
            term_fields = source_terms_fields
        elif source_terms_fields != term_fields:
            raise ValueError("source term-status TSV headers differ")
        materialized_terms = list(source_terms)
        term_ids = {row["mp_id"] for row in materialized_terms}
        if len(term_ids) != len(materialized_terms):
            raise ValueError(f"duplicate MP IDs in {source_name} term status")
        if len(term_ids) != int(manifest["sampled_term_count"]):
            raise ValueError(f"term count does not match manifest: {source_name}")
        if Counter(row["status"] for row in materialized_terms) != Counter(
            manifest["status_counts"]
        ):
            raise ValueError(f"term statuses do not match manifest: {source_name}")
        source_term_ids[source_name] = term_ids
        term_sources.append(iter(materialized_terms))

        fields, rows = iter_tsv(source_root / "relations.tsv", batch_id, node_type)
        if relation_fields is None:
            relation_fields = fields
        elif fields != relation_fields:
            raise ValueError("source relation TSV headers differ")
        relation_tsv_sources.append(rows)
        relation_json_sources.append(
            iter_jsonl(source_root / "relations.jsonl", batch_id, node_type)
        )

    if source_term_ids["leaf"] & source_term_ids["middle"]:
        raise ValueError("leaf and middle-layer term sets overlap")
    assert relation_fields is not None and term_fields is not None

    term_handle, term_temporary = atomic_text_writer(term_output)
    try:
        term_writer = csv.DictWriter(
            term_handle,
            fieldnames=("source_batch", "node_type", *term_fields),
            delimiter="\t",
            lineterminator="\n",
        )
        term_writer.writeheader()
        term_count = 0
        for row in heapq.merge(*term_sources, key=term_key):
            term_writer.writerow(row)
            term_count += 1
        term_handle.flush()
        os.fsync(term_handle.fileno())
        term_handle.close()
        term_temporary.replace(term_output)
    except Exception:
        term_handle.close()
        term_temporary.unlink(missing_ok=True)
        raise

    relation_ids: set[str] = set()
    relation_mp_ids: set[str] = set()
    scope_counts: Counter[str] = Counter()
    tsv_handle, tsv_temporary = atomic_text_writer(relation_tsv_output)
    try:
        writer = csv.DictWriter(
            tsv_handle,
            fieldnames=("source_batch", "node_type", *relation_fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        relation_count = 0
        for row in heapq.merge(*relation_tsv_sources, key=relation_key):
            relation_id = row["relation_id"]
            if relation_id in relation_ids:
                raise ValueError(f"duplicate relation ID: {relation_id}")
            relation_ids.add(relation_id)
            relation_mp_ids.add(row["mp_id"])
            scope_counts[row["relation_scope"]] += 1
            writer.writerow(row)
            relation_count += 1
        tsv_handle.flush()
        os.fsync(tsv_handle.fileno())
        tsv_handle.close()
        tsv_temporary.replace(relation_tsv_output)
    except Exception:
        tsv_handle.close()
        tsv_temporary.unlink(missing_ok=True)
        raise

    json_ids: set[str] = set()
    json_handle, json_temporary = atomic_text_writer(relation_jsonl_output)
    try:
        for row in heapq.merge(*relation_json_sources, key=relation_key):
            relation_id = str(row["relation_id"])
            if relation_id in json_ids:
                raise ValueError(f"duplicate JSONL relation ID: {relation_id}")
            json_ids.add(relation_id)
            json_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        json_handle.flush()
        os.fsync(json_handle.fileno())
        json_handle.close()
        json_temporary.replace(relation_jsonl_output)
    except Exception:
        json_handle.close()
        json_temporary.unlink(missing_ok=True)
        raise

    all_term_ids = source_term_ids["leaf"] | source_term_ids["middle"]
    if not relation_mp_ids.issubset(all_term_ids):
        raise ValueError("relations reference MP IDs absent from term status")
    if relation_ids != json_ids:
        raise ValueError("TSV and JSONL relation IDs differ")
    expected_relations = sum(
        int(value["relation_count"]) for value in source_manifests.values()
    )
    if relation_count != expected_relations:
        raise ValueError("combined relation count does not match source manifests")

    status_counts = Counter()
    for value in source_manifests.values():
        status_counts.update(value["status_counts"])
    manifest = {
        "schema_version": 1,
        "dataset_id": "mission4_mpo_cpo_leaf_middle_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Evidence-backed MPO-to-CPO relations for leaf and internal MPO terms.",
        "source_batches": source_manifests,
        "counts": {
            "term_count": term_count,
            "leaf_term_count": len(source_term_ids["leaf"]),
            "internal_term_count": len(source_term_ids["middle"]),
            "linked_term_count": status_counts["linked"],
            "all_rejected_term_count": status_counts["all_rejected"],
            "no_candidate_term_count": status_counts["no_candidate"],
            "error_term_count": status_counts["error"],
            "relation_count": relation_count,
            "direct_relation_count": scope_counts["direct"],
            "mechanism_relation_count": scope_counts["mechanism"],
        },
        "invariants": {
            "leaf_internal_term_overlap": 0,
            "duplicate_relation_id_count": 0,
            "tsv_jsonl_relation_ids_equal": True,
            "source_error_count": 0,
        },
        "added_fields": ["source_batch", "node_type"],
        "outputs": {
            "relations_tsv": relation_tsv_output.relative_to(root).as_posix(),
            "relations_jsonl": relation_jsonl_output.relative_to(root).as_posix(),
            "term_status_tsv": term_output.relative_to(root).as_posix(),
        },
    }
    manifest_path = final_root / "manifest.json"
    manifest["output_sha256"] = {
        "relations_tsv": sha256(relation_tsv_output),
        "relations_jsonl": sha256(relation_jsonl_output),
        "term_status_tsv": sha256(term_output),
    }
    handle, temporary = atomic_text_writer(manifest_path)
    try:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        temporary.replace(manifest_path)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mission4 MPO-CPO dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "mpo_cpo",
    )
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
