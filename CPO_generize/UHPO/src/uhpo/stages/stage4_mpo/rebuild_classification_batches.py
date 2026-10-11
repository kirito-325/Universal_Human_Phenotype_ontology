"""Restore deterministic Stage 4 batch caches from the audited decision table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .classify_terms import digest_terms, load_terms
from .paths import (
    CLASSIFICATION_BATCH_DIR,
    CLASSIFICATION_BATCH_MANIFEST,
    MPO_OBO,
    TERM_DECISIONS,
)


DECISION_FIELDS = (
    "human_relevant",
    "decision",
    "confidence",
    "mouse_specific_features",
    "reason",
    "decision_source",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Invalid human_relevant value: {value!r}")
    return normalized == "true"


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        term_id = row.get("mp_id", "").strip()
        if term_id in decisions:
            raise ValueError(f"Duplicate decision for {term_id}")
        decisions[term_id] = row
    return decisions


def rebuild(
    mpo_obo: Path,
    decisions_path: Path,
    batch_dir: Path,
    manifest_path: Path,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _, terms, _ = load_terms(mpo_obo)
    active = sorted(
        (term for term in terms.values() if not term.obsolete),
        key=lambda term: term.mp_id,
    )
    source_digest = digest_terms(active)
    decisions = load_decisions(decisions_path)
    active_ids = {term.mp_id for term in active}
    missing = active_ids - decisions.keys()
    if missing:
        raise ValueError(f"Decision table is missing {len(missing)} active MPO terms")

    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_records: list[dict[str, object]] = []
    for batch_number, start in enumerate(range(0, len(active), batch_size), start=1):
        batch = active[start : start + batch_size]
        batch_id = f"batch_{batch_number:05d}"
        batch_path = batch_dir / f"{batch_id}.json"
        if batch_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {batch_path}; pass --overwrite")
        term_ids = [term.mp_id for term in batch]
        rows = [decisions[term_id] for term_id in term_ids]
        wrong_batches = [row["mp_id"] for row in rows if row.get("batch_id") != batch_id]
        if wrong_batches:
            raise ValueError(
                f"{batch_id} does not match saved batch IDs for {wrong_batches[:5]}"
            )
        models = {row.get("model", "").strip() for row in rows}
        if len(models) != 1 or not next(iter(models)):
            raise ValueError(f"{batch_id} has inconsistent model metadata: {models}")
        model = next(iter(models))
        payload = {
            "status": "complete",
            "batch_id": batch_id,
            "term_ids": term_ids,
            "source_digest": source_digest,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "metadata": {"restored_from_audited_decision_table": True},
            "api_failure": "",
            "decisions": {
                row["mp_id"]: {
                    "human_relevant": parse_bool(row["human_relevant"]),
                    **{field: row.get(field, "") for field in DECISION_FIELDS[1:]},
                }
                for row in rows
            },
        }
        write_json(batch_path, payload)
        batch_records.append(
            {
                "batch_id": batch_id,
                "terms": len(term_ids),
                "file": batch_path.name,
                "sha256": sha256(batch_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Offline reconstruction of the audited MPO relevance decision table",
        "source_digest": source_digest,
        "decision_table_sha256": sha256(decisions_path),
        "active_terms": len(active),
        "batch_size": batch_size,
        "batch_count": len(batch_records),
        "batches": batch_records,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpo-obo", type=Path, default=MPO_OBO)
    parser.add_argument("--decisions", type=Path, default=TERM_DECISIONS)
    parser.add_argument("--batch-dir", type=Path, default=CLASSIFICATION_BATCH_DIR)
    parser.add_argument(
        "--manifest", type=Path, default=CLASSIFICATION_BATCH_MANIFEST
    )
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = rebuild(
        args.mpo_obo,
        args.decisions,
        args.batch_dir,
        args.manifest,
        args.batch_size,
        args.overwrite,
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "batches"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
