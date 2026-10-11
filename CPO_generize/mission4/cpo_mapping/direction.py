from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


STATE_MAP = {
    "increased": "increased",
    "decreased": "decreased",
    "absent": "absent",
    "present": "present",
    "abnormal": "abnormal_unspecified",
    "unchanged": "unchanged",
    "unknown": "unknown",
    "": "unknown",
    "bidirectional": "unknown",
}


@dataclass(frozen=True)
class DirectionRule:
    rule_id: str
    priority: int
    field: str
    pattern: re.Pattern[str]
    direction_values: frozenset[str]
    target: str
    property_name: str
    status: str


@dataclass(frozen=True)
class DirectionSignature:
    status: str
    target: str
    property_name: str
    state: str


def load_rules(path: str | Path) -> tuple[DirectionRule, ...]:
    rules: list[DirectionRule] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rules.append(
                DirectionRule(
                    rule_id=row["rule_id"],
                    priority=int(row["priority"]),
                    field=row["field"],
                    pattern=re.compile(row["pattern"], re.IGNORECASE),
                    direction_values=frozenset(row["direction_values"].split("|")),
                    target=row["direction_target"],
                    property_name=row["direction_property"],
                    status=row["resolution_status"],
                )
            )
    return tuple(sorted(rules, key=lambda rule: (rule.priority, rule.rule_id)))


def normalize_direction(
    row: Mapping[str, str], rules: Iterable[DirectionRule]
) -> dict[str, str]:
    raw = row.get("direction", "").strip().casefold()
    state = STATE_MAP.get(raw, "unknown")
    if state in {"unknown", "abnormal_unspecified", "unchanged"}:
        return {
            "direction_raw": raw,
            "direction_state": state,
            "direction_target": "",
            "direction_property": "",
            "direction_resolution_status": "unresolved_state",
            "direction_rule_id": "",
        }
    for rule in rules:
        if raw not in rule.direction_values:
            continue
        text = (
            " ".join(
                row.get(field, "")
                for field in ("cell_text", "process_text", "quality_text")
            )
            if rule.field == "combined_text"
            else row.get(rule.field, "")
        )
        if rule.pattern.search(text):
            return {
                "direction_raw": raw,
                "direction_state": state,
                "direction_target": rule.target,
                "direction_property": rule.property_name,
                "direction_resolution_status": rule.status,
                "direction_rule_id": rule.rule_id,
            }
    return {
        "direction_raw": raw,
        "direction_state": state,
        "direction_target": "",
        "direction_property": "",
        "direction_resolution_status": "unresolved_raw_direction_only",
        "direction_rule_id": "",
    }


def aggregate_signatures(
    directions: Iterable[Mapping[str, str]],
) -> DirectionSignature:
    resolved = [
        value
        for value in directions
        if value.get("direction_resolution_status", "").startswith("resolved")
    ]
    if not resolved:
        return DirectionSignature("unknown", "", "", "unknown")
    axes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for value in resolved:
        axes[(value["direction_target"], value["direction_property"])].add(
            value["direction_state"]
        )
    if len(axes) > 1:
        return DirectionSignature("multiple_axes", "", "", "unknown")
    (target, property_name), states = next(iter(axes.items()))
    if len(states) > 1:
        return DirectionSignature("mixed_direction", target, property_name, "unknown")
    return DirectionSignature("unique", target, property_name, next(iter(states)))


def compare_signatures(
    left: DirectionSignature, right: DirectionSignature
) -> str:
    if left.status != "unique" or right.status != "unique":
        return "unknown_or_incomparable"
    if (left.target, left.property_name) != (right.target, right.property_name):
        return "unknown_or_incomparable"
    pair = (left.state, right.state)
    if pair in {
        ("increased", "increased"),
        ("decreased", "decreased"),
        ("absent", "absent"),
        ("present", "present"),
    }:
        return "agree"
    if pair in {
        ("increased", "decreased"),
        ("decreased", "increased"),
        ("present", "absent"),
        ("absent", "present"),
    }:
        return "conflict"
    if set(pair) == {"decreased", "absent"}:
        return "related_nonidentical"
    return "unknown_or_incomparable"
