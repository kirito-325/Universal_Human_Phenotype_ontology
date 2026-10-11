#!/usr/bin/env python3
"""Export a stable, partial HPO-CPO candidate-link snapshot from HPO_link_v6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TSV_FIELDS = (
    "hpo_id",
    "hpo_label",
    "selected_cl_id",
    "selected_cl_label",
    "selected_go_id",
    "selected_go_label",
    "cell_text",
    "process_text",
    "quality_text",
    "direction",
    "relation_type",
    "bridge_scope",
    "evidence_type",
    "species",
    "taxon_id",
    "disease_ids",
    "claim_id",
    "evidence_span_ids",
    "supporting_record_ids",
    "gene_mentions_ignored",
    "extraction_method",
    "extractor_version",
    "source_evidence_run_id",
    "candidate_status",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _list_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return "" if value is None else str(value)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _list_value(row.get(field, ""))
                    for field in TSV_FIELDS
                }
            )


def export_snapshot(source_root: Path, batch_root: Path, output_dir: Path) -> None:
    status_paths = sorted((batch_root / "status").glob("*.json"))
    statuses = [_read_json(path) for path in status_paths]
    selection_manifest_path = batch_root / "selection_manifest.json"
    selection_manifest = _read_json(selection_manifest_path)

    evidence_state_counts: Counter[str] = Counter()
    completed: list[tuple[str, str]] = []
    for status in statuses:
        hpo_id = str(status.get("hpo_id", ""))
        evidence = status.get("evidence")
        if isinstance(evidence, dict):
            state = str(evidence.get("state", "unknown"))
            run_id = str(evidence.get("run_id", ""))
        else:
            state = str(evidence or "unknown")
            run_id = ""
        evidence_state_counts[state] += 1
        if state == "completed" and hpo_id and run_id:
            completed.append((hpo_id, run_id))

    claims: list[dict[str, Any]] = []
    completed_with_claims: set[str] = set()
    for hpo_id, run_id in completed:
        hpo_root = source_root / "literature" / hpo_id.replace(":", "_")
        claim_path = hpo_root / "evidence" / "claims" / f"{run_id}.jsonl"
        span_path = hpo_root / "evidence" / "spans" / f"{run_id}.jsonl"
        span_records = {
            str(span.get("evidence_span_id", "")): str(span.get("record_id", ""))
            for span in _read_jsonl(span_path)
        }
        for claim in _read_jsonl(claim_path):
            span_ids = claim.get("evidence_span_ids", [])
            record_ids = sorted(
                {
                    span_records.get(str(span_id), "")
                    for span_id in span_ids
                    if span_records.get(str(span_id), "")
                }
            )
            exported = dict(claim)
            exported["supporting_record_ids"] = record_ids
            exported["source_evidence_run_id"] = run_id
            exported["candidate_status"] = "unreviewed_extracted_claim"
            claims.append(exported)
            completed_with_claims.add(hpo_id)

    claims.sort(key=lambda row: (str(row.get("hpo_id", "")), str(row.get("claim_id", ""))))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "hpo_cpo_candidate_links.partial.jsonl"
    tsv_path = output_dir / "hpo_cpo_candidate_links.partial.tsv"
    _write_jsonl(jsonl_path, claims)
    _write_tsv(tsv_path, claims)

    claim_ids = [str(row.get("claim_id", "")) for row in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Duplicate claim IDs found in snapshot")

    manifest = {
        "dataset": "HPO-CPO candidate links (partial snapshot)",
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_batch_id": batch_root.name,
        "source_selection_manifest_sha256": _sha256(selection_manifest_path),
        "scope": {
            "selected_hpo_target_count": int(selection_manifest.get("term_count", 0)),
            "status_file_count": len(statuses),
            "evidence_state_counts": dict(sorted(evidence_state_counts.items())),
            "evidence_completed_hpo_count": len(completed),
            "completed_hpo_with_claims_count": len(completed_with_claims),
            "claim_count": len(claims),
            "claims_with_selected_cl_id": sum(
                bool(row.get("selected_cl_id")) for row in claims
            ),
            "claims_with_selected_go_id": sum(
                bool(row.get("selected_go_id")) for row in claims
            ),
            "direct_hpo_claim_count": sum(
                row.get("bridge_scope") == "direct_hpo" for row in claims
            ),
            "unique_hpo_count": len(
                {str(row.get("hpo_id", "")) for row in claims}
            ),
        },
        "status": "partial; evidence-extracted candidates pending final mapping and review",
        "files": {
            jsonl_path.name: {
                "bytes": jsonl_path.stat().st_size,
                "sha256": _sha256(jsonl_path),
            },
            tsv_path.name: {
                "bytes": tsv_path.stat().st_size,
                "sha256": _sha256(tsv_path),
            },
        },
    }
    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_snapshot(
        args.source_root.resolve(),
        args.batch_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
