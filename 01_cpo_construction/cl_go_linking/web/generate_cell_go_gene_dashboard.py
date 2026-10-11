from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty TSV: {path}") from exc
        return columns, list(reader)


def require_columns(path: Path, columns: list[str], required: set[str]) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def group_rows(
    columns: list[str], rows: list[list[str]], key_columns: list[str]
) -> dict[str, list[list[str]]]:
    positions = [columns.index(column) for column in key_columns]
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        key = "\u0001".join(row[position] for position in positions)
        grouped.setdefault(key, []).append(row)
    return grouped


def html_safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def run(args: argparse.Namespace) -> None:
    cells_columns, cells_rows = read_tsv(args.cells)
    process_columns, process_rows = read_tsv(args.processes)
    gene_columns, gene_rows = read_tsv(args.genes)
    require_columns(args.cells, cells_columns, {"cell_cl_id", "cell_label"})
    require_columns(
        args.processes,
        process_columns,
        {"cell_cl_id", "go_id", "go_name", "link_confidence", "supporting_genes"},
    )
    require_columns(
        args.genes,
        gene_columns,
        {"cell_cl_id", "go_id", "human_gene_symbol", "marker_q_value"},
    )
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    catalog_path = args.catalog
    if catalog_path is None:
        manifest_catalog = manifest.get("inputs", {}).get("catalog")
        if not manifest_catalog:
            raise ValueError(
                "Catalog path is absent from the manifest; pass it with --catalog"
            )
        catalog_path = Path(manifest_catalog)
    catalog_columns, catalog_rows = read_tsv(catalog_path)
    require_columns(
        catalog_path,
        catalog_columns,
        {
            "order",
            "level",
            "rank",
            "source",
            "branch",
            "cl_id",
            "cl_name",
            "catalog_label",
            "catalog_label_zh",
            "parent_id",
        },
    )
    counts = manifest.get("counts", {})
    expected = {
        "linked_cells": len(cells_rows),
        "cell_go_process_links": len(process_rows),
        "cell_go_gene_links": len(gene_rows),
    }
    for key, actual in expected.items():
        if counts.get(key) != actual:
            raise ValueError(f"Manifest {key}={counts.get(key)}, TSV has {actual}")

    payload = {
        "cell_columns": cells_columns,
        "cells": cells_rows,
        "process_columns": process_columns,
        "processes_by_cell": group_rows(
            process_columns, process_rows, ["cell_cl_id"]
        ),
        "gene_columns": gene_columns,
        "genes_by_link": group_rows(
            gene_columns, gene_rows, ["cell_cl_id", "go_id"]
        ),
        "catalog_columns": catalog_columns,
        "catalog": catalog_rows,
        "manifest": manifest,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    template = args.template.read_text(encoding="utf-8")
    placeholder = "__EMBEDDED_CELL_GO_GENE_JSON__"
    if template.count(placeholder) != 1:
        raise ValueError(f"Template must contain exactly one {placeholder} placeholder")
    output = template.replace(placeholder, html_safe_json(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "catalog_nodes": len(catalog_rows),
                **expected,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    data = root / "GO_link/work/legacy/cell_go_gene"
    parser = argparse.ArgumentParser(
        description="Build the standalone cell-to-GO-to-gene evidence dashboard."
    )
    parser.add_argument(
        "--cells", type=Path, default=data / "cell_process_summary.tsv"
    )
    parser.add_argument(
        "--processes", type=Path, default=data / "cell_go_process_summary.tsv"
    )
    parser.add_argument(
        "--genes", type=Path, default=data / "cell_go_gene_links.tsv"
    )
    parser.add_argument(
        "--manifest", type=Path, default=data / "cell_go_gene_manifest.json"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Cell catalog TSV; defaults to manifest.inputs.catalog",
    )
    parser.add_argument(
        "--template", type=Path, default=root / "web/cell_go_gene_template.html"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "GO_link/work/legacy/cell_go_gene_dashboard.html",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
