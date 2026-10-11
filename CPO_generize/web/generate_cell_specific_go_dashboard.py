from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celltype_configure.common import load_obo


CATALOG_FIELDS = [
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
    "parent_label",
]

LINK_FIELDS = [
    "go_id",
    "go_name",
    "cl_id",
    "cl_name",
    "catalog_label",
    "branch",
    "final_link_status",
    "confidence",
    "review_required",
    "decision_source",
    "evidence_sources",
    "link_origin",
    "decision_reason",
]

MARKER_KEY = "catalog_id"
MARKER_FIELDS = [
    "marker_rank",
    "gene",
    "marker_score",
    "statistics_score",
    "signatures_score",
    "gene_mapping_method",
]

FUNCTION_GENE_KEY = "go_id"
FUNCTION_GENE_FIELDS = [
    "human_gene_symbol",
    "annotation_count",
    "has_experimental_evidence",
    "evidence_codes",
    "annotation_relations",
]

INHERITED_FUNCTION_GENE_FIELDS = [
    "human_gene_symbol",
    "source_go_count",
    "source_go_ids",
    "min_distance",
    "annotation_count",
    "has_experimental_evidence",
    "evidence_codes",
]

PROPAGATION_RELATIONS = {"is_a", "part_of"}


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty TSV: {path}") from exc
        return columns, list(reader)


def select_columns(
    path: Path, columns: list[str], rows: list[list[str]], required: list[str]
) -> list[list[str]]:
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    positions = [columns.index(column) for column in required]
    return [[row[position] for position in positions] for row in rows]


def group_selected_rows(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
    key_column: str,
    fields: list[str],
    accepted_keys: set[str] | None,
) -> dict[str, list[list[str]]]:
    required = [key_column, *fields]
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    key_position = columns.index(key_column)
    positions = [columns.index(field) for field in fields]
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        key = row[key_position]
        if accepted_keys is None or key in accepted_keys:
            grouped[key].append([row[position] for position in positions])
    return dict(grouped)


def propagate_function_genes(
    go_obo_path: Path,
    direct_genes_by_go: dict[str, list[list[str]]],
    target_go_ids: set[str],
) -> tuple[
    dict[str, list[list[str]]],
    dict[str, str],
]:
    graph = load_obo(go_obo_path)
    missing_targets = sorted(target_go_ids - set(graph.nodes))
    if missing_targets:
        raise ValueError(
            f"Linked GO terms are absent from the GO ontology: {missing_targets[:20]}"
        )

    parents_by_child: dict[str, set[str]] = defaultdict(set)
    for child, parent, relation in graph.edges(keys=True):
        if relation in PROPAGATION_RELATIONS:
            parents_by_child[str(child)].add(str(parent))

    symbol_position = FUNCTION_GENE_FIELDS.index("human_gene_symbol")
    annotation_count_position = FUNCTION_GENE_FIELDS.index("annotation_count")
    experimental_position = FUNCTION_GENE_FIELDS.index(
        "has_experimental_evidence"
    )
    evidence_position = FUNCTION_GENE_FIELDS.index("evidence_codes")
    direct_symbols_by_target = {
        go_id: {row[symbol_position] for row in direct_genes_by_go.get(go_id, [])}
        for go_id in target_go_ids
    }

    def merge_source_rows(
        target_genes: dict[str, dict[str, object]],
        source_go_id: str,
        source_rows: list[list[str]],
        distance: int,
        excluded_symbols: set[str] | None = None,
    ) -> None:
        excluded = excluded_symbols or set()
        for row in source_rows:
            symbol = row[symbol_position]
            if symbol in excluded:
                continue
            gene = target_genes.setdefault(
                symbol,
                {
                    "source_go_ids": set(),
                    "min_distance": distance,
                    "annotation_count": 0,
                    "has_experimental_evidence": False,
                    "evidence_codes": set(),
                },
            )
            gene["source_go_ids"].add(source_go_id)
            gene["min_distance"] = min(int(gene["min_distance"]), distance)
            gene["annotation_count"] = int(gene["annotation_count"]) + int(
                row[annotation_count_position] or 0
            )
            gene["has_experimental_evidence"] = bool(
                gene["has_experimental_evidence"]
                or row[experimental_position] == "true"
            )
            gene["evidence_codes"].update(
                code for code in row[evidence_position].split("|") if code
            )

    inherited: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for source_go_id, source_rows in direct_genes_by_go.items():
        if source_go_id not in graph:
            continue
        distances = {source_go_id: 0}
        queue = deque((parent, 1) for parent in parents_by_child[source_go_id])
        while queue:
            current, distance = queue.popleft()
            known_distance = distances.get(current)
            if known_distance is not None and known_distance <= distance:
                continue
            distances[current] = distance
            if current in target_go_ids:
                merge_source_rows(
                    inherited[current],
                    source_go_id,
                    source_rows,
                    distance,
                    direct_symbols_by_target[current],
                )
            queue.extend(
                (parent, distance + 1)
                for parent in parents_by_child.get(current, set())
            )

    used_source_go_ids: set[str] = set()

    def serialize_groups(
        groups: dict[str, dict[str, dict[str, object]]],
    ) -> dict[str, list[list[str]]]:
        rows_by_go: dict[str, list[list[str]]] = {}
        for target_go_id, genes in groups.items():
            rows: list[list[str]] = []
            for symbol, gene in sorted(genes.items()):
                source_go_ids = sorted(gene["source_go_ids"])
                used_source_go_ids.update(source_go_ids)
                rows.append(
                    [
                        symbol,
                        str(len(source_go_ids)),
                        "|".join(source_go_ids),
                        str(gene["min_distance"]),
                        str(gene["annotation_count"]),
                        str(gene["has_experimental_evidence"]).lower(),
                        "|".join(sorted(gene["evidence_codes"])),
                    ]
                )
            if rows:
                rows_by_go[target_go_id] = rows
        return rows_by_go

    inherited_rows_by_go = serialize_groups(inherited)

    source_go_names = {
        go_id: str(graph.nodes[go_id].get("name", ""))
        for go_id in sorted(used_source_go_ids)
    }
    return inherited_rows_by_go, source_go_names


def html_safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_payload(
    links_path: Path,
    catalog_path: Path,
    manifest_path: Path,
    markers_path: Path | None = None,
    function_genes_path: Path | None = None,
    go_obo_path: Path | None = None,
) -> dict[str, object]:
    link_columns, raw_links = read_tsv(links_path)
    catalog_columns, raw_catalog = read_tsv(catalog_path)
    links = select_columns(links_path, link_columns, raw_links, LINK_FIELDS)
    catalog = select_columns(
        catalog_path, catalog_columns, raw_catalog, CATALOG_FIELDS
    )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    cl_position = LINK_FIELDS.index("cl_id")
    go_position = LINK_FIELDS.index("go_id")
    catalog_cl_position = CATALOG_FIELDS.index("cl_id")
    catalog_ids = {row[catalog_cl_position] for row in catalog}
    missing_cells = sorted(
        {row[cl_position] for row in links if row[cl_position] not in catalog_ids}
    )
    if missing_cells:
        raise ValueError(f"Links contain cells outside the catalog: {missing_cells[:20]}")
    link_keys = [(row[cl_position], row[go_position]) for row in links]
    if len(link_keys) != len(set(link_keys)):
        raise ValueError("Final links contain duplicate (CL, GO) pairs")
    expected_links = manifest.get("final_direct_links")
    if expected_links is None:
        expected_links = manifest.get("output", {}).get("specific_links")
    if expected_links != len(links):
        raise ValueError(
            f"Manifest final_direct_links={expected_links}, TSV has {len(links)}"
        )

    linked_cells = {row[cl_position] for row in links}
    linked_go = {row[go_position] for row in links}

    markers_by_cell: dict[str, list[list[str]]] = {}
    if markers_path is not None:
        marker_columns, marker_rows = read_tsv(markers_path)
        markers_by_cell = group_selected_rows(
            markers_path,
            marker_columns,
            marker_rows,
            MARKER_KEY,
            MARKER_FIELDS,
            linked_cells,
        )
        rank_position = MARKER_FIELDS.index("marker_rank")
        for rows in markers_by_cell.values():
            rows.sort(key=lambda row: int(row[rank_position] or 0))

    function_genes_by_go: dict[str, list[list[str]]] = {}
    inherited_function_genes_by_go: dict[str, list[list[str]]] = {}
    function_source_go_names: dict[str, str] = {}
    if function_genes_path is not None:
        function_columns, function_rows = read_tsv(function_genes_path)
        all_function_genes_by_go = group_selected_rows(
            function_genes_path,
            function_columns,
            function_rows,
            FUNCTION_GENE_KEY,
            FUNCTION_GENE_FIELDS,
            None,
        )
        function_genes_by_go = {
            go_id: all_function_genes_by_go[go_id]
            for go_id in linked_go
            if go_id in all_function_genes_by_go
        }
        symbol_position = FUNCTION_GENE_FIELDS.index("human_gene_symbol")
        for rows in function_genes_by_go.values():
            rows.sort(key=lambda row: row[symbol_position])
        if go_obo_path is not None:
            (
                inherited_function_genes_by_go,
                function_source_go_names,
            ) = propagate_function_genes(
                go_obo_path,
                all_function_genes_by_go,
                linked_go,
            )

    marker_symbol_position = MARKER_FIELDS.index("gene")
    function_symbol_position = FUNCTION_GENE_FIELDS.index("human_gene_symbol")
    marker_symbols_by_cell = {
        cl_id: {row[marker_symbol_position] for row in rows}
        for cl_id, rows in markers_by_cell.items()
    }
    function_symbols_by_go = {
        go_id: {row[function_symbol_position] for row in rows}
        for go_id, rows in function_genes_by_go.items()
    }
    inherited_symbol_position = INHERITED_FUNCTION_GENE_FIELDS.index(
        "human_gene_symbol"
    )
    for go_id, rows in inherited_function_genes_by_go.items():
        function_symbols_by_go.setdefault(go_id, set()).update(
            row[inherited_symbol_position] for row in rows
        )
    overlap_links = sum(
        bool(
            marker_symbols_by_cell.get(row[cl_position], set())
            & function_symbols_by_go.get(row[go_position], set())
        )
        for row in links
    )
    return {
        "catalog_columns": CATALOG_FIELDS,
        "catalog": catalog,
        "link_columns": LINK_FIELDS,
        "links": links,
        "marker_columns": MARKER_FIELDS,
        "markers_by_cell": markers_by_cell,
        "function_gene_columns": FUNCTION_GENE_FIELDS,
        "function_genes_by_go": function_genes_by_go,
        "inherited_function_gene_columns": INHERITED_FUNCTION_GENE_FIELDS,
        "inherited_function_genes_by_go": inherited_function_genes_by_go,
        "function_source_go_names": function_source_go_names,
        "summary": {
            "catalog_nodes": len(catalog),
            "linked_cells": len(linked_cells),
            "linked_go_terms": len(linked_go),
            "direct_links": len(links),
            "cell_marker_pairs": sum(len(rows) for rows in markers_by_cell.values()),
            "direct_go_function_gene_pairs": sum(
                len(rows) for rows in function_genes_by_go.values()
            ),
            "inherited_go_function_gene_pairs": sum(
                len(rows) for rows in inherited_function_genes_by_go.values()
            ),
            "go_function_gene_pairs": sum(
                len(rows) for rows in function_genes_by_go.values()
            )
            + sum(len(rows) for rows in inherited_function_genes_by_go.values()),
            "links_with_gene_overlap": overlap_links,
        },
        "manifest": {
            "generated_at": manifest.get("generated_at", ""),
            "adjudicated_scope_counts": manifest.get(
                "adjudicated_scope_counts",
                manifest.get("output", {}).get("scope_counts", {}),
            ),
            "quality_flag_counts": manifest.get("quality_flag_counts", {}),
        },
        "dashboard_generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def run(args: argparse.Namespace) -> None:
    payload = build_payload(
        args.links,
        args.catalog,
        args.manifest,
        args.markers,
        args.function_genes,
        args.go_obo,
    )
    template = args.template.read_text(encoding="utf-8")
    placeholder = "__EMBEDDED_CELL_SPECIFIC_GO_JSON__"
    if template.count(placeholder) != 1:
        raise ValueError(f"Template must contain exactly one {placeholder}")
    output = template.replace(placeholder, html_safe_json(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                **payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    data = root / "GO_link/results/full"
    parser = argparse.ArgumentParser(
        description="Build a standalone cell-to-specific-GO dashboard."
    )
    parser.add_argument(
        "--links", type=Path, default=data / "cell_specific_go_links.tsv"
    )
    parser.add_argument(
        "--manifest", type=Path, default=data / "scope_manifest.json"
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=root / "cell_marker/results/cell_marker_genes.tsv.gz",
    )
    parser.add_argument(
        "--function-genes",
        type=Path,
        default=root / "GO_function/results/go_function_genes.tsv",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=Path(
            "/path/to/workspace/HPO_expand/CPO/data/go/go-basic.obo"
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "/path/to/workspace/marker_literuture/expanded/"
            "expanded_cell_catalog.tsv"
        ),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "web/cell_specific_go_template.html",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data / "cell_specific_go_dashboard.html",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
