#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MISSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MISSION_ROOT / "data/legacy_inputs"
DEFAULT_PRIOR_ROOT = (
    PROJECT_ROOT
    / "HPO_link_v6"
    / "work"
    / "paper_batches"
    / "class_aware_leaf_all_genetic_mechanism_12121_20260812"
)
DEFAULT_RECOVERY_ROOT = (
    PROJECT_ROOT
    / "HPO_link_v6"
    / "results"
    / "coverage_recovery_uncovered_leaf_all_7774_v8_ecnu_20260904"
)
DEFAULT_TARGET = MISSION_ROOT / "data" / "hpo_cpo"

PRIOR_RELATIONS = "completed_leaf_linkage_relations.tsv"
PRIOR_OVERVIEW = "completed_leaf_linkage_overview.tsv"
PRIOR_SELECTION = "selection_manifest.json"
RECOVERY_RELATIONS = "all_candidate_relations.tsv"
RECOVERY_STATUS = "hpo_status.tsv"
RECOVERY_MANIFEST = "manifest.json"

RELATION_FIELDS = (
    "hpo_id",
    "hpo_label",
    "primary_class",
    "link_level",
    "bridge_scope",
    "relation_type",
    "direction",
    "cl_id",
    "cl_label",
    "cell_text",
    "go_id",
    "go_label",
    "process_text",
    "species",
    "evidence_count",
    "article_count",
    "evidence_types",
    "representative_quote",
    "record_id",
    "section",
    "source_url",
)
OVERVIEW_FIELDS = (
    "index",
    "hpo_id",
    "hpo_label",
    "primary_class",
    "link_status",
    "link_status_label",
    "data_error",
    "retrieval_records",
    "full_text_records",
    "candidate_count",
    "rejected_evidence_count",
    "unlinked_article_count",
    "claim_count",
    "direct_claim_count",
    "mechanistic_lead_count",
    "unique_relation_count",
    "direct_cl_links",
    "mechanistic_cl_links",
    "go_links",
    "relation_types",
    "bridge_scopes",
    "representative_evidence",
)
LINK_STATUS_LABELS = {
    "direct": "直接关系",
    "mechanism_only": "仅机制关系",
    "candidate_rejected": "无完整 CPO 关系",
    "no_candidate": "无候选证据",
    "no_retrieval": "无召回文献",
    "data_error": "审计数据损坏",
}


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields:
        raise ValueError(f"missing TSV header: {path}")
    for index, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"malformed TSV row {index}: {path}")
    return fields, rows


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def write_json(path: Path, value: object) -> None:
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
        target.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: object) -> int:
    try:
        return int(str(value or 0))
    except ValueError:
        return 0


def split_values(value: str, separator: str = ";") -> list[str]:
    return [part.strip() for part in value.split(separator) if part.strip()]


def first_value(value: str, separator: str = ";") -> str:
    values = split_values(value, separator)
    return values[0] if values else ""


def map_recovery_relation(
    row: Mapping[str, str], primary_class: str
) -> dict[str, str]:
    record_ids = split_values(row.get("source_record_ids", ""))
    quote = first_value(row.get("evidence_quotes", ""), " || ")
    return {
        "hpo_id": row.get("hpo_id", ""),
        "hpo_label": row.get("hpo_label", ""),
        "primary_class": primary_class,
        "link_level": "direct" if row.get("evidence_tier") == "direct" else "lead",
        "bridge_scope": row.get("bridge_scopes", ""),
        "relation_type": row.get("relation_types", ""),
        "direction": row.get("direction", ""),
        "cl_id": row.get("cl_id", ""),
        "cl_label": row.get("cl_label", ""),
        "cell_text": row.get("cl_label", ""),
        "go_id": row.get("go_id", ""),
        "go_label": row.get("go_label", ""),
        "process_text": row.get("go_label", ""),
        "species": row.get("species", ""),
        "evidence_count": row.get("claim_count", ""),
        "article_count": str(len(record_ids)),
        "evidence_types": "",
        "representative_quote": quote,
        "record_id": record_ids[0] if record_ids else "",
        "section": "",
        "source_url": first_value(row.get("source_urls", "")),
    }


def relation_sort_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in RELATION_FIELDS)


def merge_relations(
    prior_rows: Sequence[Mapping[str, str]],
    recovery_rows: Sequence[Mapping[str, str]],
    recovery_status: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, str]], int]:
    mapped = [
        map_recovery_relation(
            row, recovery_status.get(row.get("hpo_id", ""), {}).get("primary_class", "")
        )
        for row in recovery_rows
    ]
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for row in [*prior_rows, *mapped]:
        normalized = {field: str(row.get(field, "")) for field in RELATION_FIELDS}
        key = relation_sort_key(normalized)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        merged.append(normalized)
    merged.sort(key=relation_sort_key)
    return merged, duplicate_count


def counted_axis_values(
    rows: Sequence[Mapping[str, str]],
    id_field: str,
    label_field: str,
    text_field: str,
) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        identifier = row.get(id_field, "").strip()
        if identifier:
            label = row.get(label_field, "").strip()
            value = f"{identifier} {label}".strip()
        else:
            value = row.get(text_field, "").strip()
        if not value:
            continue
        counts[value] += 1
    return "; ".join(
        f"{value} ({count})"
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def counted_multi_values(rows: Sequence[Mapping[str, str]], field: str) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(split_values(row.get(field, "")))
    return "; ".join(
        f"{value} ({count})"
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def merge_overview(
    prior_rows: Sequence[Mapping[str, str]],
    recovery_status_rows: Sequence[Mapping[str, str]],
    recovery_relation_rows: Sequence[Mapping[str, str]],
    merged_relations: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    prior = {row["hpo_id"]: row for row in prior_rows}
    recovery = {row["hpo_id"]: row for row in recovery_status_rows}
    relations: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    recovery_claim_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in merged_relations:
        relations[row["hpo_id"]].append(row)
    for row in recovery_relation_rows:
        tier = "direct" if row.get("evidence_tier") == "direct" else "lead"
        recovery_claim_counts[row["hpo_id"]][tier] += integer(
            row.get("claim_count")
        )

    result: list[dict[str, object]] = []
    for index, hpo_id in enumerate(sorted(set(prior) | set(recovery)), start=1):
        old = prior.get(hpo_id, {})
        new = recovery.get(hpo_id, {})
        term_relations = relations.get(hpo_id, [])
        direct = [row for row in term_relations if row["link_level"] == "direct"]
        leads = [row for row in term_relations if row["link_level"] == "lead"]
        if direct:
            link_status = "direct"
        elif leads:
            link_status = "mechanism_only"
        elif old.get("data_error"):
            link_status = "data_error"
        elif new:
            link_status = (
                "no_retrieval"
                if integer(new.get("retrieved_record_count")) == 0
                else "no_candidate"
                if integer(new.get("candidate_count")) == 0
                else "candidate_rejected"
            )
        else:
            link_status = old.get("link_status", "no_retrieval")

        new_direct_claims = recovery_claim_counts[hpo_id]["direct"]
        new_lead_claims = recovery_claim_counts[hpo_id]["lead"]
        representative = next(
            (
                row.get("representative_quote", "")
                for row in [*direct, *leads]
                if row.get("representative_quote", "")
            ),
            old.get("representative_evidence", ""),
        )
        result.append(
            {
                "index": index,
                "hpo_id": hpo_id,
                "hpo_label": old.get("hpo_label") or new.get("hpo_label", ""),
                "primary_class": old.get("primary_class") or new.get("primary_class", ""),
                "link_status": link_status,
                "link_status_label": LINK_STATUS_LABELS[link_status],
                "data_error": old.get("data_error", ""),
                "retrieval_records": max(
                    integer(old.get("retrieval_records")),
                    integer(new.get("retrieved_record_count")),
                ),
                "full_text_records": integer(old.get("full_text_records")),
                "candidate_count": max(
                    integer(old.get("candidate_count")),
                    integer(new.get("candidate_count")),
                ),
                "rejected_evidence_count": integer(old.get("rejected_evidence_count")),
                "unlinked_article_count": integer(old.get("unlinked_article_count")),
                "claim_count": integer(old.get("claim_count"))
                + integer(new.get("raw_claim_count")),
                "direct_claim_count": integer(old.get("direct_claim_count"))
                + new_direct_claims,
                "mechanistic_lead_count": integer(old.get("mechanistic_lead_count"))
                + new_lead_claims,
                "unique_relation_count": len(term_relations),
                "direct_cl_links": counted_axis_values(
                    direct, "cl_id", "cl_label", "cell_text"
                ),
                "mechanistic_cl_links": counted_axis_values(
                    leads, "cl_id", "cl_label", "cell_text"
                ),
                "go_links": counted_axis_values(
                    term_relations, "go_id", "go_label", "process_text"
                ),
                "relation_types": counted_multi_values(term_relations, "relation_type"),
                "bridge_scopes": counted_multi_values(term_relations, "bridge_scope"),
                "representative_evidence": representative,
            }
        )
    return result


def build_dataset(prior_root: Path, recovery_root: Path, target: Path) -> dict[str, object]:
    prior_paths = {
        "relations": prior_root / PRIOR_RELATIONS,
        "overview": prior_root / PRIOR_OVERVIEW,
        "selection": prior_root / PRIOR_SELECTION,
    }
    recovery_paths = {
        "relations": recovery_root / RECOVERY_RELATIONS,
        "status": recovery_root / RECOVERY_STATUS,
        "manifest": recovery_root / RECOVERY_MANIFEST,
    }
    missing = [
        str(path)
        for path in [*prior_paths.values(), *recovery_paths.values()]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing HPO-CPO inputs: " + ", ".join(missing))

    prior_fields, prior_relations = read_tsv(prior_paths["relations"])
    prior_overview_fields, prior_overview = read_tsv(prior_paths["overview"])
    recovery_fields, recovery_relations = read_tsv(recovery_paths["relations"])
    recovery_status_fields, recovery_status_rows = read_tsv(recovery_paths["status"])
    if tuple(prior_fields) != RELATION_FIELDS:
        raise ValueError("prior HPO relation schema changed")
    if tuple(prior_overview_fields) != OVERVIEW_FIELDS:
        raise ValueError("prior HPO overview schema changed")
    required_recovery_fields = {
        "hpo_id",
        "hpo_label",
        "cl_id",
        "cl_label",
        "go_id",
        "go_label",
        "direction",
        "evidence_tier",
        "bridge_scopes",
        "relation_types",
        "claim_count",
        "source_record_ids",
        "source_urls",
        "evidence_quotes",
    }
    if not required_recovery_fields.issubset(recovery_fields):
        raise ValueError("recovery HPO relation schema changed")
    if not {"hpo_id", "primary_class", "coverage_status"}.issubset(
        recovery_status_fields
    ):
        raise ValueError("recovery HPO status schema changed")

    recovery_status = {row["hpo_id"]: row for row in recovery_status_rows}
    if len(recovery_status) != len(recovery_status_rows):
        raise ValueError("recovery status contains duplicate HPO IDs")
    recovery_manifest = json.loads(recovery_paths["manifest"].read_text(encoding="utf-8"))
    if len(recovery_status_rows) != int(recovery_manifest["term_count"]):
        raise ValueError("recovery status count does not match its manifest")
    recovery_relation_ids = [row["relation_id"] for row in recovery_relations]
    if len(recovery_relation_ids) != len(set(recovery_relation_ids)):
        raise ValueError("recovery relation IDs are not unique")

    merged_relations, duplicate_count = merge_relations(
        prior_relations, recovery_relations, recovery_status
    )
    merged_overview = merge_overview(
        prior_overview,
        recovery_status_rows,
        recovery_relations,
        merged_relations,
    )
    relation_hpo_ids = {row["hpo_id"] for row in merged_relations}
    overview_hpo_ids = {str(row["hpo_id"]) for row in merged_overview}
    if not relation_hpo_ids.issubset(overview_hpo_ids):
        raise ValueError("merged relations reference HPO IDs absent from overview")
    if len(overview_hpo_ids) != len(merged_overview):
        raise ValueError("merged overview contains duplicate HPO IDs")

    selection = json.loads(prior_paths["selection"].read_text(encoding="utf-8"))
    selected_term_count = int(selection["term_count"])
    if len(merged_overview) > selected_term_count:
        raise ValueError("merged overview exceeds the selected HPO cohort")

    target.mkdir(parents=True, exist_ok=True)
    relation_path = target / "hpo_cpo_relations.tsv"
    overview_path = target / "hpo_cpo_overview.tsv"
    selection_path = target / "selection_manifest.json"
    relation_count = write_tsv(relation_path, RELATION_FIELDS, merged_relations)
    overview_count = write_tsv(overview_path, OVERVIEW_FIELDS, merged_overview)
    copy_atomic(prior_paths["selection"], selection_path)

    link_levels = Counter(row["link_level"] for row in merged_relations)
    prior_hpo_ids = {row["hpo_id"] for row in prior_relations}
    recovery_hpo_ids = {row["hpo_id"] for row in recovery_relations}
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset_id": "mission4_hpo_cpo_literature_leaf_unified_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "Unified HPO-to-CPO leaf relations from the original literature export "
            "and ECNU recovery output, without an exposed batch dimension."
        ),
        "merge_policy": {
            "batch_dimension_exposed": False,
            "relation_order": "global_lexical",
            "duplicate_policy": "exact_full_row_first_wins",
            "recovery_direct_tier": "direct",
            "recovery_non_direct_tier": "lead",
            "recovery_representative_evidence": "first_aggregated_quote_and_source",
            "overview_scope": "union_of_source_status_tables",
        },
        "inputs": {
            "prior": {key: file_record(path) for key, path in prior_paths.items()},
            "recovery": {key: file_record(path) for key, path in recovery_paths.items()},
        },
        "counts": {
            "selected_term_count": selected_term_count,
            "overview_term_count": overview_count,
            "terms_absent_from_merged_overview": selected_term_count - overview_count,
            "linked_hpo_count": len(relation_hpo_ids),
            "relation_count": relation_count,
            "prior_relation_count": len(prior_relations),
            "recovery_relation_count": len(recovery_relations),
            "exact_duplicate_relation_count_removed": duplicate_count,
            "prior_linked_hpo_count": len(prior_hpo_ids),
            "recovery_linked_hpo_count": len(recovery_hpo_ids),
            "overlapping_linked_hpo_count": len(prior_hpo_ids & recovery_hpo_ids),
            "newly_linked_hpo_count": len(recovery_hpo_ids - prior_hpo_ids),
            "direct_relation_count": link_levels["direct"],
            "mechanism_lead_count": link_levels["lead"],
            "complete_cpo_relation_count": sum(
                bool(row["cl_id"] and row["go_id"]) for row in merged_relations
            ),
            "relations_with_cl_id": sum(bool(row["cl_id"]) for row in merged_relations),
            "relations_with_go_id": sum(bool(row["go_id"]) for row in merged_relations),
            "relations_with_representative_quote": sum(
                bool(row["representative_quote"]) for row in merged_relations
            ),
            "relations_with_literature_url": sum(
                bool(row["source_url"]) for row in merged_relations
            ),
            "overview_status_counts": dict(
                sorted(Counter(str(row["link_status"]) for row in merged_overview).items())
            ),
        },
        "invariants": {
            "batch_dimension_absent_from_relation_schema": "source_batch"
            not in RELATION_FIELDS,
            "duplicate_relation_rows": 0,
            "duplicate_overview_hpo_ids": 0,
            "relation_hpo_ids_absent_from_overview": 0,
            "recovery_excluded_hpo_count": int(
                recovery_manifest.get("excluded_hpo_count", 0)
            ),
        },
        "outputs": {
            "relations": file_record(relation_path),
            "overview": file_record(overview_path),
            "selection_manifest": file_record(selection_path),
        },
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the unified mission4 HPO-CPO leaf dataset"
    )
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(args.prior_root, args.recovery_root, args.target),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
