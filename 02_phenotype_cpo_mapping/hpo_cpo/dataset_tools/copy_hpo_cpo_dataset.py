#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "data/legacy_inputs"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "HPO_link_v6"
    / "work"
    / "paper_batches"
    / "class_aware_leaf_all_genetic_mechanism_12121_20260812"
)
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "data" / "hpo_cpo"

SOURCE_FILES = {
    "relations_tsv": "completed_leaf_linkage_relations.tsv",
    "overview_tsv": "completed_leaf_linkage_overview.tsv",
    "selection_manifest": "selection_manifest.json",
}
OUTPUT_FILES = {
    "relations_tsv": "hpo_cpo_relations.tsv",
    "overview_tsv": "hpo_cpo_overview.tsv",
    "selection_manifest": "selection_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.replace(target)
        target.chmod(source.stat().st_mode & 0o777)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields:
        raise ValueError(f"missing TSV header: {path}")
    for index, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"malformed TSV row at logical record {index}: {path}")
    return fields, rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def copy_dataset(source: Path, target: Path) -> dict[str, object]:
    source_paths = {key: source / name for key, name in SOURCE_FILES.items()}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {', '.join(missing)}")

    relation_fields, relations = read_tsv(source_paths["relations_tsv"])
    overview_fields, overview = read_tsv(source_paths["overview_tsv"])
    required_relation_fields = {
        "hpo_id",
        "link_level",
        "bridge_scope",
        "relation_type",
        "cl_id",
        "go_id",
        "representative_quote",
        "source_url",
    }
    required_overview_fields = {"hpo_id", "link_status", "data_error"}
    if not required_relation_fields.issubset(relation_fields):
        raise ValueError("source relation table does not match the expected schema")
    if not required_overview_fields.issubset(overview_fields):
        raise ValueError("source overview table does not match the expected schema")

    overview_ids = [row["hpo_id"] for row in overview]
    relation_hpo_ids = {row["hpo_id"] for row in relations}
    if len(set(overview_ids)) != len(overview_ids):
        raise ValueError("source overview contains duplicate HPO IDs")
    if not relation_hpo_ids.issubset(overview_ids):
        raise ValueError("relations reference HPO IDs absent from the source overview")

    selection = json.loads(source_paths["selection_manifest"].read_text(encoding="utf-8"))
    selected_term_count = int(selection["term_count"])
    if len(overview) > selected_term_count:
        raise ValueError("overview contains more terms than the selection manifest")

    source_hashes = {key: sha256(path) for key, path in source_paths.items()}
    output_paths = {key: target / name for key, name in OUTPUT_FILES.items()}
    for key, output_path in output_paths.items():
        copy_atomic(source_paths[key], output_path)
        if sha256(output_path) != source_hashes[key]:
            raise ValueError(f"copied file hash mismatch: {output_path}")

    link_levels = Counter(row["link_level"] for row in relations)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "mission4_hpo_cpo_literature_leaf_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "Exact copies of literature-mined HPO-to-CPO leaf relations and "
            "their source-batch overview."
        ),
        "source": {
            "batch_id": selection["batch_id"],
            "root": source.relative_to(PROJECT_ROOT).as_posix(),
            "files": {
                key: path.relative_to(PROJECT_ROOT).as_posix()
                for key, path in source_paths.items()
            },
            "sha256": source_hashes,
        },
        "counts": {
            "selected_term_count": selected_term_count,
            "overview_term_count": len(overview),
            "terms_absent_from_source_overview": selected_term_count - len(overview),
            "linked_hpo_count": len(relation_hpo_ids),
            "relation_count": len(relations),
            "direct_relation_count": link_levels["direct"],
            "mechanism_lead_count": link_levels["lead"],
            "relations_with_cl_id": sum(bool(row["cl_id"]) for row in relations),
            "relations_with_go_id": sum(bool(row["go_id"]) for row in relations),
            "relations_with_representative_quote": sum(
                bool(row["representative_quote"]) for row in relations
            ),
            "relations_with_literature_url": sum(
                bool(row["source_url"]) for row in relations
            ),
            "hpo_definition_representative_count": sum(
                row["record_id"].startswith("hpo_definition_") for row in relations
            ),
            "overview_status_counts": dict(
                sorted(Counter(row["link_status"] for row in overview).items())
            ),
            "relation_type_counts": dict(
                sorted(Counter(row["relation_type"] for row in relations).items())
            ),
            "bridge_scope_counts": dict(
                sorted(Counter(row["bridge_scope"] for row in relations).items())
            ),
        },
        "invariants": {
            "copies_match_source_sha256": True,
            "duplicate_overview_hpo_count": 0,
            "relation_hpo_ids_absent_from_overview": 0,
        },
        "notes": [
            "The source export contains both direct relations and mechanism leads.",
            "Rows without a literature URL use an HPO self-definition as their representative evidence; every row retains a representative quote and record ID.",
            "The source overview is smaller than the selection cohort; this transfer preserves that source state without filling or filtering rows.",
            "The TSV parser treats quoted embedded newlines as part of one logical record.",
        ],
        "outputs": {
            key: path.relative_to(target).as_posix()
            for key, path in output_paths.items()
        },
        "output_sha256": {key: sha256(path) for key, path in output_paths.items()},
    }
    write_json_atomic(target / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the frozen HPO-CPO literature relation export into mission4"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    print(json.dumps(copy_dataset(args.source, args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
