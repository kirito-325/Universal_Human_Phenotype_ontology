from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from celltype_configure.common import CellCatalog, stable_digest, write_tsv


LLM_FIELDS = [
    "llm_classification",
    "llm_constraint_cl_id",
    "llm_constraint_label",
    "llm_confidence",
    "llm_reason",
    "selected_candidate_rank",
    "selected_recall_sources",
    "selected_embedding_score",
    "selected_lexical_score",
    "llm_status",
    "llm_batch_id",
    "llm_response_id",
    "llm_model",
]

CONFIDENCE_LEVEL = {"low": 1, "medium": 2, "high": 3}


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def split_ids(value: str) -> list[str]:
    return list(dict.fromkeys(item for item in value.split("|") if item))


def accepted(decision: Mapping[str, str], minimum_confidence: str) -> bool:
    return (
        decision.get("llm_status") == "success"
        and CONFIDENCE_LEVEL.get(decision.get("llm_confidence", ""), 0)
        >= CONFIDENCE_LEVEL[minimum_confidence]
    )


def merge_classifications(
    original_rows: Sequence[Mapping[str, str]],
    decision_rows: Sequence[Mapping[str, str]],
    catalog: CellCatalog,
    minimum_confidence: str,
) -> tuple[list[dict[str, str]], Counter]:
    decisions: dict[str, Mapping[str, str]] = {}
    for decision in decision_rows:
        go_id = decision.get("go_id", "")
        if not go_id or go_id in decisions:
            raise ValueError(f"Duplicate or empty GO ID in LLM decisions: {go_id!r}")
        decisions[go_id] = decision

    original_ids = {row["go_id"] for row in original_rows}
    unexpected = set(decisions) - original_ids
    if unexpected:
        raise ValueError(f"LLM decisions contain unknown GO IDs: {sorted(unexpected)[:20]}")

    merged: list[dict[str, str]] = []
    outcomes: Counter = Counter()
    for original in original_rows:
        row = dict(original)
        for field in LLM_FIELDS:
            row[field] = ""
        decision = decisions.get(row["go_id"])
        if not decision:
            merged.append(row)
            continue
        if original.get("classification") != "undetermined":
            raise ValueError(
                f"LLM decision targets a non-undetermined GO term: {row['go_id']}"
            )
        for field in LLM_FIELDS:
            row[field] = decision.get(field, "")

        classification = decision.get("llm_classification", "undetermined")
        is_accepted = accepted(decision, minimum_confidence)
        if classification == "cell_type_specific" and is_accepted:
            cl_id = decision.get("llm_constraint_cl_id", "")
            if cl_id not in catalog.nodes or cl_id == "CL:0000000":
                raise ValueError(f"Invalid accepted CL mapping for {row['go_id']}: {cl_id}")
            row.update(
                {
                    "classification": "cell_type_specific_candidate",
                    "confidence": decision["llm_confidence"],
                    "review_required": "true",
                    "constraint_scope": "self_and_descendants",
                    "constraint_root_cl_ids": cl_id,
                    "constraint_root_labels": catalog.nodes[cl_id].catalog_label,
                    "decision_reason": "LLM dual-recall candidate: "
                    + decision.get("llm_reason", ""),
                }
            )
            outcomes["accepted_specific"] += 1
        elif classification == "cell_general" and is_accepted:
            row.update(
                {
                    "classification": "cell_general_candidate",
                    "confidence": decision["llm_confidence"],
                    "review_required": "true",
                    "constraint_scope": "",
                    "constraint_root_cl_ids": "",
                    "constraint_root_labels": "",
                    "decision_reason": "LLM dual-recall general candidate: "
                    + decision.get("llm_reason", ""),
                }
            )
            outcomes["accepted_general"] += 1
        else:
            row.update(
                {
                    "classification": "undetermined",
                    "confidence": decision.get("llm_confidence", "low") or "low",
                    "review_required": "true",
                    "constraint_scope": "",
                    "constraint_root_cl_ids": "",
                    "constraint_root_labels": "",
                    "decision_reason": "LLM retained for review: "
                    + decision.get("llm_reason", ""),
                }
            )
            outcomes[
                "retained_undetermined"
                if classification == "undetermined"
                else "below_confidence_threshold"
            ] += 1
        merged.append(row)
    outcomes["decision_rows"] = len(decisions)
    return merged, outcomes


def write_constraints(
    rows: Iterable[Mapping[str, str]], catalog: CellCatalog, output_dir: Path
) -> tuple[int, int]:
    closure_fields = [
        "go_id",
        "mapping_status",
        "constraint_root_cl_id",
        "allowed_cl_id",
        "allowed_cl_label",
        "relation_to_root",
    ]
    context_fields = [
        "go_id",
        "mapping_status",
        "constraint_root_cl_id",
        "ancestor_cl_id",
        "ancestor_cl_label",
        "distance",
    ]
    closure_path = output_dir / "go_cl_constraint_closure.tsv"
    context_path = output_dir / "go_cl_constraint_context.tsv"
    closure_temporary = closure_path.with_suffix(".tsv.part")
    context_temporary = context_path.with_suffix(".tsv.part")
    closure_count = 0
    context_count = 0
    with closure_temporary.open(
        "w", encoding="utf-8", newline=""
    ) as closure_handle, context_temporary.open(
        "w", encoding="utf-8", newline=""
    ) as context_handle:
        closure_writer = csv.DictWriter(
            closure_handle, fieldnames=closure_fields, delimiter="\t"
        )
        context_writer = csv.DictWriter(
            context_handle, fieldnames=context_fields, delimiter="\t"
        )
        closure_writer.writeheader()
        context_writer.writeheader()
        for row in rows:
            if row.get("classification") not in {
                "cell_type_specific",
                "cell_type_specific_candidate",
            }:
                continue
            emitted_allowed: set[str] = set()
            for root in split_ids(row.get("constraint_root_cl_ids", "")):
                if root not in catalog.nodes:
                    raise ValueError(f"Unknown constraint root for {row['go_id']}: {root}")
                for allowed in catalog.descendants(root, include_self=True):
                    if allowed in emitted_allowed:
                        continue
                    emitted_allowed.add(allowed)
                    closure_writer.writerow(
                        {
                            "go_id": row["go_id"],
                            "mapping_status": row["classification"],
                            "constraint_root_cl_id": root,
                            "allowed_cl_id": allowed,
                            "allowed_cl_label": catalog.nodes[allowed].catalog_label,
                            "relation_to_root": "self" if allowed == root else "descendant",
                        }
                    )
                    closure_count += 1
                for distance, ancestor in enumerate(catalog.ancestors(root), start=1):
                    context_writer.writerow(
                        {
                            "go_id": row["go_id"],
                            "mapping_status": row["classification"],
                            "constraint_root_cl_id": root,
                            "ancestor_cl_id": ancestor,
                            "ancestor_cl_label": catalog.nodes[ancestor].catalog_label,
                            "distance": distance,
                        }
                    )
                    context_count += 1
    closure_temporary.replace(closure_path)
    context_temporary.replace(context_path)
    return closure_count, context_count


def run(args: argparse.Namespace) -> None:
    if args.minimum_confidence not in CONFIDENCE_LEVEL:
        raise ValueError(f"Unknown confidence threshold: {args.minimum_confidence}")
    original_rows = read_tsv(args.classification)
    decision_rows = read_tsv(args.llm_decisions)
    if not original_rows:
        raise ValueError("Original classification is empty")
    if not decision_rows:
        raise ValueError("LLM decisions are empty")
    catalog = CellCatalog.from_tsv(args.catalog)
    merged_rows, outcomes = merge_classifications(
        original_rows, decision_rows, catalog, args.minimum_confidence
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classification_path = output_dir / "go_classification.tsv"
    write_tsv(classification_path, list(merged_rows[0]), merged_rows)
    closure_count, context_count = write_constraints(merged_rows, catalog, output_dir)
    classification_counts = Counter(row["classification"] for row in merged_rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "go_terms": len(merged_rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "llm_application_counts": dict(sorted(outcomes.items())),
        "minimum_llm_confidence": args.minimum_confidence,
        "constraint_closure_rows": closure_count,
        "constraint_context_rows": context_count,
        "inputs": {
            "classification": str(Path(args.classification).resolve()),
            "llm_decisions": str(Path(args.llm_decisions).resolve()),
            "catalog": str(Path(args.catalog).resolve()),
        },
        "classification_digest": stable_digest(
            "\t".join(str(row.get(column, "")) for column in merged_rows[0])
            for row in merged_rows
        ),
    }
    manifest_path = output_dir / "classification_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Apply reviewed LLM decisions to the GO-to-CL classification."
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=root / "GO_link/work/legacy/final/go_classification.tsv",
    )
    parser.add_argument(
        "--llm-decisions",
        type=Path,
        default=root / "GO_link/work/legacy/llm/llm_decisions.tsv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "/path/to/workspace/marker_literuture/expanded/expanded_cell_catalog.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "GO_link/work/legacy/final_llm"
    )
    parser.add_argument(
        "--minimum-confidence", choices=sorted(CONFIDENCE_LEVEL), default="medium"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
