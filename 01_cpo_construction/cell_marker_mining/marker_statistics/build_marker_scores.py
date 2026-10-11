#!/usr/bin/env python3
"""Score markers for expanded-catalog cells with paper-level evidence caps.

For a (standard cell ID, human gene) pair, one paper contributes at most one
point from marker_statistics_cleaned.csv and at most one point from
marker_signatures_cleaned.tsv. Therefore each paper contributes at most two
points even when rows, source genes, or source cell-name aliases are repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS = ROOT / "database" / "papers"
DEFAULT_MAPPING = ROOT / "name_mapping" / "sapbert_top1.tsv"
DEFAULT_CATALOG = ROOT / "expanded" / "expanded_cell_catalog.tsv"
DEFAULT_OUTPUT = ROOT / "marker_statistics" / "expanded_marker_scores.tsv"
DEFAULT_TOP_OUTPUT = ROOT / "marker_statistics" / "expanded_marker_scores_top100.tsv"
DEFAULT_OPPORTUNITIES = ROOT / "marker_statistics" / "expanded_cell_evidence_opportunities.tsv"
DEFAULT_UNMAPPED = ROOT / "marker_statistics" / "unmapped_cell_names.tsv"
DEFAULT_SUMMARY = ROOT / "marker_statistics" / "summary.json"
STATISTICS_NAME = "marker_statistics_cleaned.csv"
SIGNATURES_NAME = "marker_signatures_cleaned.tsv"


@dataclass(frozen=True)
class CatalogCell:
    order: int
    catalog_id: str
    cl_name: str
    label: str
    label_zh: str
    parent_id: str
    parent_label: str
    branch: str
    rank: str


@dataclass
class FileEvidence:
    evidence: set[tuple[str, str]] = field(default_factory=set)
    input_occurrences: int = 0
    mapped_occurrences: int = 0
    unmapped_occurrences: Counter[str] = field(default_factory=Counter)
    empty_cell_or_gene: int = 0


@dataclass
class CollectionResult:
    statistics_scores: Counter[tuple[str, str]] = field(default_factory=Counter)
    signature_scores: Counter[tuple[str, str]] = field(default_factory=Counter)
    cell_opportunities: dict[str, Counter[str]] = field(
        default_factory=lambda: {"statistics": Counter(), "signatures": Counter()}
    )
    table_metrics: dict[str, Counter[str]] = field(
        default_factory=lambda: {"statistics": Counter(), "signatures": Counter()}
    )
    unmapped_occurrences: dict[str, Counter[str]] = field(
        default_factory=lambda: {"statistics": Counter(), "signatures": Counter()}
    )
    unmapped_papers: dict[str, Counter[str]] = field(
        default_factory=lambda: {"statistics": Counter(), "signatures": Counter()}
    )
    papers_processed: int = 0
    duplicate_paper_directories_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def normalize_gene(value: str) -> str:
    # Inputs have already been validated as HGNC-approved human symbols.
    return unicodedata.normalize("NFKC", value).strip()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_catalog(path: Path) -> dict[str, CatalogCell]:
    cells: dict[str, CatalogCell] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"cl_id", "catalog_label"} <= set(reader.fieldnames):
            raise ValueError(f"invalid expanded catalog columns: {path}")
        for fallback_order, row in enumerate(reader, 1):
            catalog_id = str(row.get("cl_id") or "").strip()
            if not catalog_id:
                continue
            try:
                order = int(str(row.get("order") or fallback_order))
            except ValueError:
                order = fallback_order
            cell = CatalogCell(
                order=order,
                catalog_id=catalog_id,
                cl_name=str(row.get("cl_name") or "").strip(),
                label=str(row.get("catalog_label") or "").strip(),
                label_zh=str(row.get("catalog_label_zh") or "").strip(),
                parent_id=str(row.get("parent_id") or "").strip(),
                parent_label=str(row.get("parent_label") or "").strip(),
                branch=str(row.get("branch") or "").strip(),
                rank=str(row.get("rank") or "").strip(),
            )
            if catalog_id in cells:
                raise ValueError(f"duplicate catalog ID {catalog_id}: {path}")
            cells[catalog_id] = cell
    if not cells:
        raise ValueError(f"expanded catalog is empty: {path}")
    return cells


def load_top1_mapping(
    path: Path,
    catalog: dict[str, CatalogCell],
    require_adjudicated: bool = False,
    accepted_decisions: frozenset[str] = frozenset({"match", "exact_auto"}),
) -> tuple[dict[str, str], dict[str, Any]]:
    mappings: dict[str, str] = {}
    models: set[str] = set()
    decision_counts: Counter[str] = Counter()
    input_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_name", "catalog_id"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid Top1 mapping columns: {path}")
        has_decisions = "decision" in reader.fieldnames
        if require_adjudicated and not has_decisions:
            raise ValueError(
                f"adjudicated mapping required but decision column is absent: {path}"
            )
        for row in reader:
            input_rows += 1
            decision = str(row.get("decision") or "").strip().lower()
            if has_decisions:
                decision_counts[decision or "blank"] += 1
                if decision not in accepted_decisions:
                    continue
            cell_name = str(row.get("cell_name") or "").strip()
            catalog_id = str(row.get("catalog_id") or "").strip()
            if not cell_name or not catalog_id:
                continue
            if catalog_id not in catalog:
                raise ValueError(f"mapping refers to catalog ID absent from expanded catalog: {catalog_id}")
            previous = mappings.get(cell_name)
            if previous is not None and previous != catalog_id:
                raise ValueError(f"conflicting Top1 mappings for {cell_name!r}: {previous}, {catalog_id}")
            mappings[cell_name] = catalog_id
            model = str(row.get("embedding_model") or "").strip()
            if model:
                models.add(model)
    if not mappings:
        raise ValueError(f"Top1 mapping is empty: {path}")
    return mappings, {
        "mapping_rows": input_rows,
        "mapped_source_names": len(mappings),
        "mapping_has_decisions": has_decisions,
        "require_adjudicated_mapping": require_adjudicated,
        "accepted_mapping_decisions": sorted(accepted_decisions) if has_decisions else [],
        "mapping_decision_counts": dict(sorted(decision_counts.items())),
        "embedding_models": sorted(models),
    }


def read_statistics(path: Path, mappings: dict[str, str]) -> FileEvidence:
    result = FileEvidence()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cell_cluster", "gene"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"missing cell_cluster/gene columns: {path}")
        for row in reader:
            result.input_occurrences += 1
            cell_name = str(row.get("cell_cluster") or "").strip()
            gene = normalize_gene(str(row.get("gene") or ""))
            if not cell_name or not gene:
                result.empty_cell_or_gene += 1
                continue
            catalog_id = mappings.get(cell_name)
            if catalog_id is None:
                result.unmapped_occurrences[cell_name] += 1
                continue
            result.mapped_occurrences += 1
            result.evidence.add((catalog_id, gene))
    return result


def read_signatures(path: Path, mappings: dict[str, str]) -> FileEvidence:
    result = FileEvidence()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_cluster", "gene"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"missing cell_cluster/gene columns: {path}")
        for row in reader:
            result.input_occurrences += 1
            cell_name = str(row.get("cell_cluster") or "").strip()
            gene = normalize_gene(str(row.get("gene") or ""))
            if not cell_name or not gene:
                result.empty_cell_or_gene += 1
                continue
            catalog_id = mappings.get(cell_name)
            if catalog_id is None:
                result.unmapped_occurrences[cell_name] += 1
                continue
            result.mapped_occurrences += 1
            result.evidence.add((catalog_id, gene))
    return result


def paper_id(record_dir: Path) -> str:
    metadata = read_json(record_dir / "metadata.json")
    return str(metadata.get("id") or record_dir.name).strip() or record_dir.name


def iter_record_dirs(papers_dir: Path) -> Iterator[Path]:
    for record_dir in sorted(papers_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        output_dir = record_dir / "agent_standardized"
        if (output_dir / STATISTICS_NAME).is_file() or (output_dir / SIGNATURES_NAME).is_file():
            yield record_dir


def add_file_result(
    result: CollectionResult,
    table: str,
    file_result: FileEvidence,
) -> None:
    scores = result.statistics_scores if table == "statistics" else result.signature_scores
    scores.update(file_result.evidence)
    result.cell_opportunities[table].update({catalog_id for catalog_id, _ in file_result.evidence})
    metrics = result.table_metrics[table]
    metrics["files_read"] += 1
    metrics["input_occurrences"] += file_result.input_occurrences
    metrics["mapped_occurrences"] += file_result.mapped_occurrences
    metrics["unmapped_occurrences"] += sum(file_result.unmapped_occurrences.values())
    metrics["empty_cell_or_gene"] += file_result.empty_cell_or_gene
    metrics["awarded_points"] += len(file_result.evidence)
    metrics["duplicate_occurrences_collapsed"] += (
        file_result.mapped_occurrences - len(file_result.evidence)
    )
    result.unmapped_occurrences[table].update(file_result.unmapped_occurrences)
    result.unmapped_papers[table].update(file_result.unmapped_occurrences.keys())


def collect(papers_dir: Path, mappings: dict[str, str]) -> CollectionResult:
    result = CollectionResult()
    seen_papers: set[str] = set()
    for record_dir in iter_record_dirs(papers_dir):
        current_paper = paper_id(record_dir)
        if current_paper in seen_papers:
            result.duplicate_paper_directories_skipped += 1
            result.errors.append(
                f"duplicate paper ID skipped to preserve the paper-level score cap: "
                f"{current_paper} ({record_dir})"
            )
            continue
        seen_papers.add(current_paper)
        result.papers_processed += 1
        output_dir = record_dir / "agent_standardized"
        inputs = (
            ("statistics", output_dir / STATISTICS_NAME, read_statistics),
            ("signatures", output_dir / SIGNATURES_NAME, read_signatures),
        )
        for table, path, reader in inputs:
            if not path.is_file():
                result.table_metrics[table]["missing_files"] += 1
                continue
            try:
                file_result = reader(path, mappings)
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                result.table_metrics[table]["failed_files"] += 1
                result.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            add_file_result(result, table, file_result)
    return result


SCORE_FIELDS = [
    "catalog_order",
    "catalog_id",
    "catalog_label",
    "catalog_cl_name",
    "catalog_label_zh",
    "parent_id",
    "parent_label",
    "branch",
    "catalog_rank",
    "marker_rank",
    "gene",
    "marker_score",
    "statistics_score",
    "signatures_score",
]


def ordered_score_keys(
    result: CollectionResult, catalog: dict[str, CatalogCell]
) -> list[tuple[str, str]]:
    keys = list(result.statistics_scores)
    keys.extend(key for key in result.signature_scores if key not in result.statistics_scores)

    def sort_key(key: tuple[str, str]) -> tuple[Any, ...]:
        catalog_id, gene = key
        statistics_score = result.statistics_scores[key]
        signatures_score = result.signature_scores[key]
        return (
            catalog[catalog_id].order,
            -(statistics_score + signatures_score),
            -statistics_score,
            -signatures_score,
            gene.casefold(),
            gene,
        )

    keys.sort(key=sort_key)
    return keys


def iter_score_rows(
    result: CollectionResult, catalog: dict[str, CatalogCell]
) -> Iterator[dict[str, Any]]:
    current_catalog_id = ""
    marker_rank = 0
    for catalog_id, gene in ordered_score_keys(result, catalog):
        if catalog_id != current_catalog_id:
            current_catalog_id = catalog_id
            marker_rank = 0
        marker_rank += 1
        cell = catalog[catalog_id]
        statistics_score = result.statistics_scores[(catalog_id, gene)]
        signatures_score = result.signature_scores[(catalog_id, gene)]
        yield {
            "catalog_order": cell.order,
            "catalog_id": catalog_id,
            "catalog_label": cell.label,
            "catalog_cl_name": cell.cl_name,
            "catalog_label_zh": cell.label_zh,
            "parent_id": cell.parent_id,
            "parent_label": cell.parent_label,
            "branch": cell.branch,
            "catalog_rank": cell.rank,
            "marker_rank": marker_rank,
            "gene": gene,
            "marker_score": statistics_score + signatures_score,
            "statistics_score": statistics_score,
            "signatures_score": signatures_score,
        }


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
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return handle, writer, Path(handle.name)


def write_score_table(
    path: Path,
    top_path: Path,
    top_per_cell: int,
    result: CollectionResult,
    catalog: dict[str, CatalogCell],
) -> dict[str, int]:
    handle, writer, temporary = open_atomic_tsv(path, SCORE_FIELDS)
    top_handle, top_writer, top_temporary = open_atomic_tsv(top_path, SCORE_FIELDS)
    rows = 0
    top_rows = 0
    standard_cells: set[str] = set()
    max_score = 0
    try:
        for row in iter_score_rows(result, catalog):
            writer.writerow(row)
            rows += 1
            if int(row["marker_rank"]) <= top_per_cell:
                top_writer.writerow(row)
                top_rows += 1
            standard_cells.add(str(row["catalog_id"]))
            max_score = max(max_score, int(row["marker_score"]))
    finally:
        handle.close()
        top_handle.close()
    temporary.replace(path)
    path.chmod(0o644)
    top_temporary.replace(top_path)
    top_path.chmod(0o644)
    return {
        "marker_pairs": rows,
        "top_marker_pairs": top_rows,
        "top_markers_per_cell": top_per_cell,
        "standard_cells_with_markers": len(standard_cells),
        "max_score": max_score,
    }


OPPORTUNITY_FIELDS = [
    "catalog_order",
    "catalog_id",
    "catalog_label",
    "catalog_cl_name",
    "catalog_label_zh",
    "parent_id",
    "parent_label",
    "branch",
    "catalog_rank",
    "statistics_opportunities",
    "signatures_opportunities",
    "total_opportunities",
]


def write_opportunity_table(
    path: Path, result: CollectionResult, catalog: dict[str, CatalogCell]
) -> dict[str, int]:
    statistics = result.cell_opportunities["statistics"]
    signatures = result.cell_opportunities["signatures"]
    catalog_ids = set(statistics) | set(signatures)
    ordered = sorted(catalog_ids, key=lambda catalog_id: (catalog[catalog_id].order, catalog_id))
    handle, writer, temporary = open_atomic_tsv(path, OPPORTUNITY_FIELDS)
    total = 0
    try:
        for catalog_id in ordered:
            cell = catalog[catalog_id]
            statistics_count = statistics[catalog_id]
            signatures_count = signatures[catalog_id]
            total_count = statistics_count + signatures_count
            total += total_count
            writer.writerow(
                {
                    "catalog_order": cell.order,
                    "catalog_id": catalog_id,
                    "catalog_label": cell.label,
                    "catalog_cl_name": cell.cl_name,
                    "catalog_label_zh": cell.label_zh,
                    "parent_id": cell.parent_id,
                    "parent_label": cell.parent_label,
                    "branch": cell.branch,
                    "catalog_rank": cell.rank,
                    "statistics_opportunities": statistics_count,
                    "signatures_opportunities": signatures_count,
                    "total_opportunities": total_count,
                }
            )
    finally:
        handle.close()
    temporary.replace(path)
    path.chmod(0o644)
    return {"standard_cells_with_opportunities": len(ordered), "total_cell_opportunities": total}


UNMAPPED_FIELDS = [
    "cell_name",
    "normalized_name",
    "total_occurrences",
    "statistics_occurrences",
    "signatures_occurrences",
    "statistics_papers",
    "signatures_papers",
    "reason",
]


def write_unmapped_table(path: Path, result: CollectionResult) -> int:
    statistics = result.unmapped_occurrences["statistics"]
    signatures = result.unmapped_occurrences["signatures"]
    names = set(statistics) | set(signatures)
    ordered = sorted(
        names,
        key=lambda name: (-(statistics[name] + signatures[name]), normalize_name(name), name),
    )
    handle, writer, temporary = open_atomic_tsv(path, UNMAPPED_FIELDS)
    try:
        for name in ordered:
            writer.writerow(
                {
                    "cell_name": name,
                    "normalized_name": normalize_name(name),
                    "total_occurrences": statistics[name] + signatures[name],
                    "statistics_occurrences": statistics[name],
                    "signatures_occurrences": signatures[name],
                    "statistics_papers": result.unmapped_papers["statistics"][name],
                    "signatures_papers": result.unmapped_papers["signatures"][name],
                    "reason": "not_in_filtered_cell_name_mapping",
                }
            )
    finally:
        handle.close()
    temporary.replace(path)
    path.chmod(0o644)
    return len(ordered)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--require-adjudicated-mapping",
        action="store_true",
        help=(
            "Require a decision column and accept only match/exact_auto rows. "
            "Use this for final marker statistics."
        ),
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-output", type=Path, default=DEFAULT_TOP_OUTPUT)
    parser.add_argument("--top-per-cell", type=int, default=100)
    parser.add_argument("--opportunities-output", type=Path, default=DEFAULT_OPPORTUNITIES)
    parser.add_argument("--unmapped-output", type=Path, default=DEFAULT_UNMAPPED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.top_per_cell < 1:
        raise SystemExit("--top-per-cell must be positive")
    for label, path in (
        ("papers directory", args.papers_dir),
        ("Top1 mapping", args.mapping),
        ("expanded catalog", args.catalog),
    ):
        exists = path.is_dir() if label == "papers directory" else path.is_file()
        if not exists:
            raise SystemExit(f"{label} does not exist: {path}")

    catalog = load_catalog(args.catalog.resolve())
    mappings, mapping_summary = load_top1_mapping(
        args.mapping.resolve(),
        catalog,
        require_adjudicated=args.require_adjudicated_mapping,
    )
    result = collect(args.papers_dir.resolve(), mappings)
    output_summary = write_score_table(
        args.output.resolve(),
        args.top_output.resolve(),
        args.top_per_cell,
        result,
        catalog,
    )
    opportunity_summary = write_opportunity_table(
        args.opportunities_output.resolve(), result, catalog
    )
    unmapped_names = write_unmapped_table(args.unmapped_output.resolve(), result)

    statistics_points = sum(result.statistics_scores.values())
    signature_points = sum(result.signature_scores.values())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score_rule": {
            "unit": "paper x table_type x expanded_catalog_id x gene",
            "statistics_points_per_paper_pair_max": 1,
            "signatures_points_per_paper_pair_max": 1,
            "total_points_per_paper_pair_max": 2,
            "statistics_input": STATISTICS_NAME,
            "signatures_input": SIGNATURES_NAME,
            "gene_policy": "HGNC-approved human symbols from auditable cleaned inputs",
        },
        "inputs": {
            "papers_dir": str(args.papers_dir.resolve()),
            "top1_mapping": str(args.mapping.resolve()),
            "expanded_catalog": str(args.catalog.resolve()),
            "catalog_nodes": len(catalog),
            **mapping_summary,
        },
        "papers_processed": result.papers_processed,
        "duplicate_paper_directories_skipped": result.duplicate_paper_directories_skipped,
        "table_metrics": {name: dict(metrics) for name, metrics in result.table_metrics.items()},
        "scores": {
            **output_summary,
            **opportunity_summary,
            "statistics_points": statistics_points,
            "signatures_points": signature_points,
            "total_points": statistics_points + signature_points,
        },
        "unmapped_unique_cell_names": unmapped_names,
        "outputs": {
            "marker_scores": str(args.output.resolve()),
            "top_marker_scores": str(args.top_output.resolve()),
            "cell_evidence_opportunities": str(args.opportunities_output.resolve()),
            "unmapped_cell_names": str(args.unmapped_output.resolve()),
        },
        "failure_count": len(result.errors),
        "errors": result.errors,
    }
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
