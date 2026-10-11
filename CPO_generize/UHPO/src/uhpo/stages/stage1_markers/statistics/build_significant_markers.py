#!/usr/bin/env python3
"""Retain statistically enriched marker genes for each expanded catalog cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

try:
    from scipy.stats import hypergeom
except ImportError:  # pragma: no cover - exercised only in lean environments
    hypergeom = None

from ..paths import MARKER_DIR

DEFAULT_SCORES = MARKER_DIR / "expanded_marker_scores.tsv"
DEFAULT_OPPORTUNITIES = MARKER_DIR / "expanded_cell_evidence_opportunities.tsv"
DEFAULT_OUTPUT = MARKER_DIR / "significant_marker_scores.tsv"
DEFAULT_CELL_SUMMARY = MARKER_DIR / "significant_marker_cell_summary.tsv"
DEFAULT_SUMMARY = MARKER_DIR / "significant_marker_summary.json"


OUTPUT_FIELDS = [
    "catalog_order",
    "catalog_id",
    "catalog_label",
    "catalog_cl_name",
    "catalog_label_zh",
    "parent_id",
    "parent_label",
    "branch",
    "catalog_rank",
    "significant_rank",
    "original_marker_rank",
    "gene",
    "marker_score",
    "statistics_score",
    "signatures_score",
    "cell_evidence_opportunities",
    "cell_prevalence",
    "background_evidence_opportunities",
    "background_gene_score",
    "background_prevalence",
    "enrichment_fold",
    "odds_ratio",
    "p_value",
    "q_value",
    "significance_score",
]

CELL_SUMMARY_FIELDS = [
    "catalog_order",
    "catalog_id",
    "catalog_label",
    "catalog_cl_name",
    "catalog_label_zh",
    "parent_id",
    "parent_label",
    "branch",
    "catalog_rank",
    "cell_evidence_opportunities",
    "tested_observed_genes",
    "significant_genes",
    "significant_fraction",
    "max_marker_score",
    "max_significance_score",
]


def integer(row: dict[str, str], field: str) -> int:
    return int(str(row.get(field) or "0"))


def load_opportunities(path: Path) -> tuple[dict[str, dict[str, str]], int]:
    rows: dict[str, dict[str, str]] = {}
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"catalog_id", "total_opportunities"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid opportunity table columns: {path}")
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            opportunities = integer(row, "total_opportunities")
            if not catalog_id or opportunities < 1:
                continue
            if catalog_id in rows:
                raise ValueError(f"duplicate opportunity row for {catalog_id}")
            rows[catalog_id] = row
            total += opportunities
    if not rows or total < 1:
        raise ValueError(f"opportunity table is empty: {path}")
    return rows, total


def load_gene_totals(path: Path) -> tuple[Counter[str], int]:
    totals: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene", "marker_score"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid marker score columns: {path}")
        for row in reader:
            gene = str(row.get("gene") or "").strip()
            score = integer(row, "marker_score")
            if not gene or score < 1:
                continue
            totals[gene] += score
            rows += 1
    if not totals:
        raise ValueError(f"marker score table is empty: {path}")
    return totals, rows


def iter_cell_groups(path: Path) -> Iterator[tuple[str, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "catalog_id" not in reader.fieldnames:
            raise ValueError(f"marker score table lacks catalog_id: {path}")
        current_id = ""
        current_rows: list[dict[str, str]] = []
        completed: set[str] = set()
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            if not catalog_id:
                continue
            if current_id and catalog_id != current_id:
                completed.add(current_id)
                yield current_id, current_rows
                current_rows = []
            if catalog_id in completed:
                raise ValueError(f"marker score table is not grouped by catalog ID: {catalog_id}")
            current_id = catalog_id
            current_rows.append(row)
        if current_id:
            yield current_id, current_rows


def bh_adjust_observed(p_values: np.ndarray, total_hypotheses: int) -> np.ndarray:
    """BH-adjust observed p-values while treating all unobserved genes as p=1."""
    if p_values.size == 0:
        return np.array([], dtype=float)
    if total_hypotheses < p_values.size:
        raise ValueError("total hypotheses cannot be smaller than observed p-values")
    order = np.argsort(p_values, kind="stable")
    ranks = np.arange(1, p_values.size + 1, dtype=float)
    adjusted_sorted = p_values[order] * float(total_hypotheses) / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf_scalar(k_minus_one: int, total: int, successes: int, draws: int) -> float:
    lower = max(0, k_minus_one + 1)
    upper = min(successes, draws)
    if lower > upper:
        return 0.0
    denominator = log_comb(total, draws)
    log_terms = [
        log_comb(successes, observed) + log_comb(total - successes, draws - observed) - denominator
        for observed in range(lower, upper + 1)
    ]
    peak = max(log_terms)
    return math.exp(peak) * sum(math.exp(term - peak) for term in log_terms)


def hypergeom_sf_array(
    k_minus_one: np.ndarray,
    total: int,
    successes: np.ndarray,
    draws: int,
) -> np.ndarray:
    if hypergeom is not None:
        return np.asarray(hypergeom.sf(k_minus_one, total, successes, draws), dtype=float)
    return np.asarray(
        [
            hypergeom_sf_scalar(int(k), total, int(success), draws)
            for k, success in zip(k_minus_one, successes, strict=True)
        ],
        dtype=float,
    )


def analyze_cell_rows(
    rows: list[dict[str, str]],
    gene_totals: Counter[str],
    total_opportunities: int,
    cell_opportunities: int,
    *,
    q_threshold: float,
    min_score: int,
    min_odds_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"tested_observed_genes": 0, "significant_genes": 0}
    if not 0 < cell_opportunities < total_opportunities:
        raise ValueError("cell opportunities must be between zero and total opportunities")

    cell_scores = np.asarray([integer(row, "marker_score") for row in rows], dtype=np.int64)
    global_scores = np.asarray([gene_totals[str(row.get("gene") or "")] for row in rows], dtype=np.int64)
    if np.any(cell_scores > cell_opportunities):
        raise ValueError("marker score exceeds the cell evidence opportunities")
    if np.any(global_scores > total_opportunities) or np.any(global_scores < cell_scores):
        raise ValueError("global gene score is inconsistent with cell marker score")

    p_values = hypergeom_sf_array(
        cell_scores - 1,
        total_opportunities,
        global_scores,
        cell_opportunities,
    )
    q_values = bh_adjust_observed(p_values, len(gene_totals))
    background_opportunities = total_opportunities - cell_opportunities
    background_scores = global_scores - cell_scores
    cell_prevalence = cell_scores / float(cell_opportunities)
    background_prevalence = background_scores / float(background_opportunities)
    enrichment_fold = np.divide(
        cell_prevalence,
        background_prevalence,
        out=np.full_like(cell_prevalence, np.inf, dtype=float),
        where=background_prevalence > 0,
    )
    odds_ratios = (
        (cell_scores + 0.5)
        * (background_opportunities - background_scores + 0.5)
        / ((cell_opportunities - cell_scores + 0.5) * (background_scores + 0.5))
    )

    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if (
            cell_scores[index] < min_score
            or q_values[index] > q_threshold
            or odds_ratios[index] <= min_odds_ratio
        ):
            continue
        q_value = float(q_values[index])
        significance_score = -math.log10(max(q_value, np.finfo(float).tiny))
        output = dict(row)
        output.update(
            {
                "original_marker_rank": integer(row, "marker_rank"),
                "cell_evidence_opportunities": cell_opportunities,
                "cell_prevalence": float(cell_prevalence[index]),
                "background_evidence_opportunities": background_opportunities,
                "background_gene_score": int(background_scores[index]),
                "background_prevalence": float(background_prevalence[index]),
                "enrichment_fold": float(enrichment_fold[index]),
                "odds_ratio": float(odds_ratios[index]),
                "p_value": float(p_values[index]),
                "q_value": q_value,
                "significance_score": significance_score,
            }
        )
        selected.append(output)

    selected.sort(
        key=lambda row: (
            float(row["q_value"]),
            -integer(row, "marker_score"),
            -float(row["odds_ratio"]),
            str(row.get("gene") or "").casefold(),
            str(row.get("gene") or ""),
        )
    )
    for rank, row in enumerate(selected, 1):
        row["significant_rank"] = rank

    return selected, {
        "tested_observed_genes": len(rows),
        "significant_genes": len(selected),
        "max_marker_score": max((integer(row, "marker_score") for row in selected), default=0),
        "max_significance_score": max(
            (float(row["significance_score"]) for row in selected), default=0.0
        ),
    }


def format_output_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for field in ("cell_prevalence", "background_prevalence", "enrichment_fold", "odds_ratio"):
        value = float(output[field])
        output[field] = "inf" if math.isinf(value) else f"{value:.8g}"
    for field in ("p_value", "q_value"):
        output[field] = f"{float(output[field]):.8g}"
    output["significance_score"] = f"{float(output['significance_score']):.6f}"
    return output


def open_atomic_tsv(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    writer = csv.DictWriter(
        handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    return handle, writer, Path(handle.name)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def run_analysis(
    scores_path: Path,
    opportunities_path: Path,
    output_path: Path,
    cell_summary_path: Path,
    *,
    q_threshold: float,
    min_score: int,
    min_odds_ratio: float,
) -> dict[str, Any]:
    opportunities, total_opportunities = load_opportunities(opportunities_path)
    gene_totals, marker_pairs = load_gene_totals(scores_path)
    output_handle, output_writer, output_temporary = open_atomic_tsv(output_path, OUTPUT_FIELDS)
    summary_handle, summary_writer, summary_temporary = open_atomic_tsv(
        cell_summary_path, CELL_SUMMARY_FIELDS
    )
    significant_rows = 0
    cells_with_significant = 0
    tested_cells = 0
    observed_rows = 0
    seen_cells: set[str] = set()
    try:
        for catalog_id, rows in iter_cell_groups(scores_path):
            if catalog_id not in opportunities:
                raise ValueError(f"marker cell lacks evidence opportunities: {catalog_id}")
            opportunity_row = opportunities[catalog_id]
            selected, cell_metrics = analyze_cell_rows(
                rows,
                gene_totals,
                total_opportunities,
                integer(opportunity_row, "total_opportunities"),
                q_threshold=q_threshold,
                min_score=min_score,
                min_odds_ratio=min_odds_ratio,
            )
            for row in selected:
                output_writer.writerow(format_output_row(row))
            first = rows[0]
            significant_count = len(selected)
            summary_writer.writerow(
                {
                    **first,
                    "cell_evidence_opportunities": opportunity_row["total_opportunities"],
                    **cell_metrics,
                    "significant_fraction": (
                        f"{significant_count / len(rows):.8g}" if rows else "0"
                    ),
                    "max_significance_score": f"{cell_metrics['max_significance_score']:.6f}",
                }
            )
            tested_cells += 1
            observed_rows += len(rows)
            significant_rows += significant_count
            cells_with_significant += int(significant_count > 0)
            seen_cells.add(catalog_id)
    finally:
        output_handle.close()
        summary_handle.close()
    if observed_rows != marker_pairs:
        raise ValueError(f"marker pair count changed between passes: {marker_pairs} vs {observed_rows}")
    if seen_cells != set(opportunities):
        missing = sorted(set(opportunities) - seen_cells)
        raise ValueError(f"opportunity cells absent from marker table: {missing[:5]}")
    output_temporary.replace(output_path)
    output_path.chmod(0o644)
    summary_temporary.replace(cell_summary_path)
    cell_summary_path.chmod(0o644)
    return {
        "total_cell_evidence_opportunities": total_opportunities,
        "gene_universe": len(gene_totals),
        "tested_marker_pairs": marker_pairs,
        "tested_standard_cells": tested_cells,
        "significant_marker_pairs": significant_rows,
        "standard_cells_with_significant_markers": cells_with_significant,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--opportunities", type=Path, default=DEFAULT_OPPORTUNITIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-summary", type=Path, default=DEFAULT_CELL_SUMMARY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--min-odds-ratio", type=float, default=1.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not 0 < args.q_threshold <= 1:
        raise SystemExit("--q-threshold must be in (0, 1]")
    if args.min_score < 1 or args.min_odds_ratio < 0:
        raise SystemExit("--min-score must be positive and --min-odds-ratio cannot be negative")
    for path in (args.scores, args.opportunities):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    metrics = run_analysis(
        args.scores.resolve(),
        args.opportunities.resolve(),
        args.output.resolve(),
        args.cell_summary.resolve(),
        q_threshold=args.q_threshold,
        min_score=args.min_score,
        min_odds_ratio=args.min_odds_ratio,
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "test": "one-sided hypergeometric enrichment",
            "observation_unit": "paper x table_type x expanded_catalog_id",
            "background": "all other expanded-cell evidence opportunities",
            "multiple_testing": "Benjamini-Hochberg within each cell over the global gene universe",
            "q_threshold": args.q_threshold,
            "minimum_marker_score": args.min_score,
            "minimum_odds_ratio_exclusive": args.min_odds_ratio,
            "significance_score": "-log10(q_value)",
        },
        "inputs": {
            "marker_scores": str(args.scores.resolve()),
            "cell_evidence_opportunities": str(args.opportunities.resolve()),
        },
        "outputs": {
            "significant_marker_scores": str(args.output.resolve()),
            "cell_summary": str(args.cell_summary.resolve()),
        },
        **metrics,
    }
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
