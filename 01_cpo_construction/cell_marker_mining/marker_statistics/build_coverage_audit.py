#!/usr/bin/env python3
"""Audit expanded-catalog coverage using paper/table evidence opportunities."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "expanded" / "expanded_cell_catalog.tsv"
DEFAULT_OPPORTUNITIES = ROOT / "marker_statistics" / "expanded_cell_evidence_opportunities.tsv"
DEFAULT_SIGNIFICANT = ROOT / "marker_statistics" / "significant_marker_cell_summary.tsv"
DEFAULT_JSON = ROOT / "marker_statistics" / "expanded_coverage_audit.json"
DEFAULT_MARKDOWN = ROOT / "marker_statistics" / "expanded_coverage_audit.md"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_int(value: Any) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def coverage_band(opportunities: int) -> str:
    if opportunities <= 0:
        return "none"
    if opportunities < 3:
        return "thin_1_2"
    if opportunities < 10:
        return "developing_3_9"
    return "established_10_plus"


def safe_fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def audit(
    catalog_rows: list[dict[str, str]],
    opportunity_rows: list[dict[str, str]],
    significant_rows: list[dict[str, str]],
) -> dict[str, Any]:
    opportunities = {
        row.get("catalog_id", ""): {
            "statistics": as_int(row.get("statistics_opportunities")),
            "signatures": as_int(row.get("signatures_opportunities")),
            "total": as_int(row.get("total_opportunities")),
        }
        for row in opportunity_rows
        if row.get("catalog_id")
    }
    significant = {
        row.get("catalog_id", ""): as_int(row.get("significant_genes"))
        for row in significant_rows
        if row.get("catalog_id")
    }

    cells: list[dict[str, Any]] = []
    for row in catalog_rows:
        catalog_id = str(row.get("cl_id") or "").strip()
        evidence = opportunities.get(
            catalog_id, {"statistics": 0, "signatures": 0, "total": 0}
        )
        total = evidence["total"]
        cells.append(
            {
                "catalog_order": as_int(row.get("order")),
                "catalog_id": catalog_id,
                "catalog_label": str(row.get("catalog_label") or ""),
                "catalog_cl_name": str(row.get("cl_name") or ""),
                "catalog_label_zh": str(row.get("catalog_label_zh") or ""),
                "parent_id": str(row.get("parent_id") or ""),
                "parent_label": str(row.get("parent_label") or ""),
                "branch": str(row.get("branch") or ""),
                "catalog_rank": str(row.get("rank") or ""),
                "catalog_source": str(row.get("source") or ""),
                "distance_to_anchor": as_int(row.get("distance_to_anchor")),
                "statistics_opportunities": evidence["statistics"],
                "signatures_opportunities": evidence["signatures"],
                "total_opportunities": total,
                "coverage_band": coverage_band(total),
                "opportunities_needed_for_three": max(0, 3 - total),
                "significant_genes": significant.get(catalog_id, 0),
            }
        )

    branch_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        branch_cells[cell["branch"]].append(cell)
    total_opportunities = sum(cell["total_opportunities"] for cell in cells)
    branch_metrics: list[dict[str, Any]] = []
    for branch, members in branch_cells.items():
        counts = Counter(cell["coverage_band"] for cell in members)
        any_evidence = sum(cell["total_opportunities"] > 0 for cell in members)
        repeated_evidence = sum(cell["total_opportunities"] >= 3 for cell in members)
        established = sum(cell["total_opportunities"] >= 10 for cell in members)
        significant_cells = sum(cell["significant_genes"] > 0 for cell in members)
        branch_opportunities = sum(cell["total_opportunities"] for cell in members)
        branch_metrics.append(
            {
                "branch": branch,
                "catalog_nodes": len(members),
                "nodes_with_any_evidence": any_evidence,
                "any_evidence_fraction": safe_fraction(any_evidence, len(members)),
                "nodes_with_at_least_three_opportunities": repeated_evidence,
                "at_least_three_fraction": safe_fraction(repeated_evidence, len(members)),
                "nodes_with_at_least_ten_opportunities": established,
                "nodes_with_significant_markers": significant_cells,
                "total_opportunities": branch_opportunities,
                "opportunity_share": safe_fraction(
                    branch_opportunities, total_opportunities
                ),
                "coverage_band_counts": dict(sorted(counts.items())),
            }
        )
    branch_metrics.sort(key=lambda item: (-item["catalog_nodes"], item["branch"]))

    target_cells = [cell for cell in cells if cell["catalog_rank"] != "root"]
    core_gaps = [
        cell
        for cell in target_cells
        if cell["catalog_source"] == "skeleton" and cell["total_opportunities"] < 3
    ]
    core_gaps.sort(
        key=lambda cell: (
            cell["total_opportunities"],
            cell["catalog_order"],
            cell["catalog_id"],
        )
    )
    extension_gaps = [
        cell
        for cell in target_cells
        if cell["catalog_source"] != "skeleton" and cell["total_opportunities"] < 3
    ]
    extension_gaps.sort(
        key=lambda cell: (
            cell["distance_to_anchor"],
            cell["total_opportunities"],
            cell["catalog_order"],
        )
    )

    totals = {
        "catalog_nodes": len(cells),
        "target_nodes_excluding_root": len(target_cells),
        "nodes_with_any_evidence": sum(cell["total_opportunities"] > 0 for cell in cells),
        "nodes_with_at_least_three_opportunities": sum(
            cell["total_opportunities"] >= 3 for cell in cells
        ),
        "nodes_with_at_least_ten_opportunities": sum(
            cell["total_opportunities"] >= 10 for cell in cells
        ),
        "nodes_with_significant_markers": sum(
            cell["significant_genes"] > 0 for cell in cells
        ),
        "total_opportunities": total_opportunities,
        "core_skeleton_gaps_below_three": len(core_gaps),
        "extension_gaps_below_three": len(extension_gaps),
    }
    totals["any_evidence_fraction"] = safe_fraction(
        totals["nodes_with_any_evidence"], totals["catalog_nodes"]
    )
    totals["at_least_three_fraction"] = safe_fraction(
        totals["nodes_with_at_least_three_opportunities"], totals["catalog_nodes"]
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "thin_upper_exclusive": 3,
            "established_lower_inclusive": 10,
            "interpretation": (
                "Evidence opportunities are paper x table-type observations. "
                "Three opportunities are necessary but not sufficient for a gene "
                "to reach the current minimum marker score."
            ),
        },
        "totals": totals,
        "branches": branch_metrics,
        "core_skeleton_gaps": core_gaps,
        "extension_gaps": extension_gaps,
        "cells": cells,
    }


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Expanded Catalog Coverage Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Evidence opportunity means one `paper x table type x catalog ID` observation. "
        "It measures available replication depth, not mapping correctness or marker specificity.",
        "",
        "## Overall",
        "",
        f"- Catalog nodes: {totals['catalog_nodes']}",
        f"- Nodes with any evidence: {totals['nodes_with_any_evidence']} "
        f"({percent(totals['any_evidence_fraction'])})",
        f"- Nodes with at least 3 opportunities: "
        f"{totals['nodes_with_at_least_three_opportunities']} "
        f"({percent(totals['at_least_three_fraction'])})",
        f"- Nodes with at least 10 opportunities: "
        f"{totals['nodes_with_at_least_ten_opportunities']}",
        f"- Nodes with significant markers: {totals['nodes_with_significant_markers']}",
        f"- Total evidence opportunities: {totals['total_opportunities']}",
        "",
        "## By Branch",
        "",
        "| Branch | Nodes | Any evidence | At least 3 | At least 10 | Significant | Opportunities | Share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch in report["branches"]:
        lines.append(
            f"| {branch['branch']} | {branch['catalog_nodes']} | "
            f"{branch['nodes_with_any_evidence']} ({percent(branch['any_evidence_fraction'])}) | "
            f"{branch['nodes_with_at_least_three_opportunities']} "
            f"({percent(branch['at_least_three_fraction'])}) | "
            f"{branch['nodes_with_at_least_ten_opportunities']} | "
            f"{branch['nodes_with_significant_markers']} | "
            f"{branch['total_opportunities']} | "
            f"{percent(branch['opportunity_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Core Skeleton Gaps",
            "",
            "These reviewed skeleton nodes have fewer than three evidence opportunities and "
            "should be addressed before chasing every fine Cell Ontology extension.",
            "",
            "| Catalog ID | Cell | Branch | Rank | Opportunities | Needed for 3 |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for cell in report["core_skeleton_gaps"]:
        lines.append(
            f"| {cell['catalog_id']} | {cell['catalog_label']} | {cell['branch']} | "
            f"{cell['catalog_rank']} | {cell['total_opportunities']} | "
            f"{cell['opportunities_needed_for_three']} |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "Regenerate after marker scoring:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_coverage_audit.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--opportunities", type=Path, default=DEFAULT_OPPORTUNITIES)
    parser.add_argument("--significant", type=Path, default=DEFAULT_SIGNIFICANT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    for label, path in (
        ("catalog", args.catalog),
        ("opportunities", args.opportunities),
        ("significant summary", args.significant),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    report = audit(
        read_tsv(args.catalog),
        read_tsv(args.opportunities),
        read_tsv(args.significant),
    )
    report["inputs"] = {
        "catalog": str(args.catalog.resolve()),
        "opportunities": str(args.opportunities.resolve()),
        "significant": str(args.significant.resolve()),
    }
    atomic_write(
        args.json_output.resolve(),
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write(args.markdown_output.resolve(), render_markdown(report))
    print(json.dumps(report["totals"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
