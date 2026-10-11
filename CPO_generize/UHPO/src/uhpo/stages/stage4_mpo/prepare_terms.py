#!/usr/bin/env python3
"""Build an auditable human-relevant MPO subset without damaging the source OBO."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MP_ID = re.compile(r"^MP:\d+$")


def parse_blocks(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] = []
    blocks: dict[str, list[str]] = {}
    current: list[str] = []
    in_term = False

    def consume(values: list[str]) -> None:
        if not values:
            return
        identifier = next((line.split(":", 1)[1].strip() for line in values if line.startswith("id: MP:")), "")
        if MP_ID.fullmatch(identifier):
            blocks[identifier] = list(values)

    for line in lines:
        if line == "[Term]":
            if in_term:
                consume(current)
            current = [line]
            in_term = True
        elif in_term:
            if line.startswith("[") and line.endswith("]"):
                consume(current)
                current = []
                in_term = False
                header.append(line)
            else:
                current.append(line)
        else:
            header.append(line)
    if in_term:
        consume(current)
    return header, blocks


def field(block: list[str], prefix: str) -> str:
    return next((line[len(prefix) :].strip() for line in block if line.startswith(prefix)), "")


def parents(block: list[str]) -> tuple[str, ...]:
    return tuple(
        line.split(":", 1)[1].split("!", 1)[0].strip()
        for line in block
        if line.startswith("is_a: MP:")
    )


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["mp_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo-obo", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--clean-obo", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    header, blocks = parse_blocks(args.mpo_obo)
    decisions = load_decisions(args.decisions)
    child_map: dict[str, set[str]] = defaultdict(set)
    direct_excluded: dict[str, str] = {}
    for identifier, block in blocks.items():
        for parent in parents(block):
            child_map[parent].add(identifier)
        row = decisions.get(identifier, {})
        if field(block, "is_obsolete: ") == "true":
            direct_excluded[identifier] = "obsolete"
        elif (
            row.get("decision", "review") == "exclude"
            and row.get("human_relevant", "true").lower() == "false"
            and row.get("confidence", "low") == "high"
        ):
            direct_excluded[identifier] = "high_confidence_llm_exclude"

    # Remove an entire single-parent excluded branch, but retain a multiply
    # inherited term if it still has a non-excluded parent.
    excluded = dict(direct_excluded)
    queue = deque(direct_excluded)
    while queue:
        parent = queue.popleft()
        for child in child_map.get(parent, set()):
            if child in excluded:
                continue
            child_parents = parents(blocks[child])
            if child_parents and all(parent_id in excluded for parent_id in child_parents):
                excluded[child] = f"descendant_of:{parent}"
                queue.append(child)

    kept = set(blocks) - set(excluded)
    output_lines = list(header)
    if output_lines and output_lines[-1] != "":
        output_lines.append("")
    excluded_rows: list[dict[str, Any]] = []
    for identifier in sorted(blocks):
        block = blocks[identifier]
        if identifier in excluded:
            row = decisions.get(identifier, {})
            excluded_rows.append(
                {
                    "mp_id": identifier,
                    "mp_name": field(block, "name: "),
                    "exclusion_reason": excluded[identifier],
                    "llm_decision": row.get("decision", ""),
                    "llm_confidence": row.get("confidence", ""),
                    "llm_reason": row.get("reason", ""),
                    "mouse_specific_features": row.get("mouse_specific_features", ""),
                }
            )
            continue
        filtered: list[str] = []
        for line in block:
            if line.startswith("is_a: MP:"):
                parent = line.split(":", 1)[1].split("!", 1)[0].strip()
                if parent not in kept:
                    continue
            if line.startswith("relationship: "):
                match = re.search(r"\b(MP:\d+)\b", line)
                if match and match.group(1) not in kept:
                    continue
            filtered.append(line)
        output_lines.extend(filtered)
        output_lines.append("")
    args.clean_obo.parent.mkdir(parents=True, exist_ok=True)
    args.clean_obo.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    args.excluded.parent.mkdir(parents=True, exist_ok=True)
    fields = ("mp_id", "mp_name", "exclusion_reason", "llm_decision", "llm_confidence", "llm_reason", "mouse_specific_features")
    with args.excluded.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_obo": str(args.mpo_obo.resolve()),
        "decision_file": str(args.decisions.resolve()),
        "source_term_count": len(blocks),
        "kept_term_count": len(kept),
        "excluded_term_count": len(excluded),
        "direct_exclusion_count": len(direct_excluded),
        "descendant_exclusion_count": len(excluded) - len(direct_excluded),
        "high_confidence_llm_exclusion_count": sum(v == "high_confidence_llm_exclude" for v in excluded.values()),
        "obsolete_exclusion_count": sum(v == "obsolete" for v in excluded.values()),
        "clean_obo": str(args.clean_obo.resolve()),
        "excluded_terms": str(args.excluded.resolve()),
        "policy": "Only high-confidence explicit exclusion removes an active term; review/low-confidence terms are retained.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
