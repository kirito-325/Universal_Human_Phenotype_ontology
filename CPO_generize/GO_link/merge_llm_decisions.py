from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from celltype_configure.common import stable_digest, write_tsv


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def merge_decisions(paths: Sequence[Path]) -> list[dict[str, str]]:
    order: list[str] = []
    selected: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_tsv(path):
            go_id = row.get("go_id", "")
            if not go_id:
                raise ValueError(f"Empty GO ID in {path}")
            if go_id not in selected:
                order.append(go_id)
            previous = selected.get(go_id)
            if previous and previous.get("llm_status") == "success":
                if row.get("llm_status") != "success":
                    continue
            selected[go_id] = row
    return [selected[go_id] for go_id in order]


def run(args: argparse.Namespace) -> None:
    rows = merge_decisions(args.input)
    if not rows:
        raise ValueError("No LLM decisions to merge")
    write_tsv(args.output, list(rows[0]), rows)
    statuses = Counter(row.get("llm_status", "") for row in rows)
    classifications = Counter(row.get("llm_classification", "") for row in rows)
    summary: Mapping[str, object] = {
        "output": str(args.output),
        "rows": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "classification_counts": dict(sorted(classifications.items())),
        "digest": stable_digest(
            "\t".join(row.get(column, "") for column in rows[0]) for row in rows
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Merge LLM decision files, preferring later successful retries."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "GO_link/work/legacy/llm/llm_decisions_all_success.tsv",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
