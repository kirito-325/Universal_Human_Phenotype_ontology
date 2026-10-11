from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty TSV: {path}") from exc
        return columns, [row for row in reader]


def grouped_rows(path: Path, expected_columns: list[str]) -> dict[str, list[list[str]]]:
    columns, rows = read_tsv(path)
    missing = set(expected_columns) - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    positions = [columns.index(column) for column in expected_columns]
    result: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        go_id = row[positions[0]]
        result[go_id].append([row[position] for position in positions[1:]])
    return dict(result)


def html_safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def run(args: argparse.Namespace) -> None:
    classification_path = Path(args.classification)
    closure_path = Path(args.closure)
    context_path = Path(args.context)
    manifest_path = Path(args.manifest)
    template_path = Path(args.template)
    output_path = Path(args.output)

    columns, rows = read_tsv(classification_path)
    if "go_id" not in columns or "classification" not in columns:
        raise ValueError("Classification TSV must contain go_id and classification")
    go_position = columns.index("go_id")
    go_ids = [row[go_position] for row in rows]
    if len(go_ids) != len(set(go_ids)):
        raise ValueError("Classification TSV contains duplicate GO IDs")

    closure = grouped_rows(
        closure_path,
        [
            "go_id",
            "constraint_root_cl_id",
            "allowed_cl_id",
            "allowed_cl_label",
            "relation_to_root",
        ],
    )
    context = grouped_rows(
        context_path,
        [
            "go_id",
            "constraint_root_cl_id",
            "ancestor_cl_id",
            "ancestor_cl_label",
            "distance",
        ],
    )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("go_terms") != len(rows):
        raise ValueError(
            f"Manifest reports {manifest.get('go_terms')} GO terms, TSV has {len(rows)}"
        )

    payload = {
        "columns": columns,
        "rows": rows,
        "closure": closure,
        "context": context,
        "manifest": manifest,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    template = template_path.read_text(encoding="utf-8")
    placeholder = "__EMBEDDED_RESULTS_JSON__"
    if template.count(placeholder) != 1:
        raise ValueError(f"Template must contain exactly one {placeholder} placeholder")
    output = template.replace(placeholder, html_safe_json(payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "go_terms": len(rows),
                "closure_go_terms": len(closure),
                "closure_rows": sum(map(len, closure.values())),
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the standalone GO-to-CL dashboard")
    parser.add_argument(
        "--classification",
        default=root / "GO_link/work/legacy/final/go_classification.tsv",
        type=Path,
    )
    parser.add_argument(
        "--closure",
        default=root / "GO_link/work/legacy/final/go_cl_constraint_closure.tsv",
        type=Path,
    )
    parser.add_argument(
        "--context",
        default=root / "GO_link/work/legacy/final/go_cl_constraint_context.tsv",
        type=Path,
    )
    parser.add_argument(
        "--manifest",
        default=root / "GO_link/work/legacy/final/classification_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--template", default=root / "web/dashboard_template.html", type=Path
    )
    parser.add_argument(
        "--output",
        default=root / "GO_link/work/legacy/go_cl_dashboard.html",
        type=Path,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
