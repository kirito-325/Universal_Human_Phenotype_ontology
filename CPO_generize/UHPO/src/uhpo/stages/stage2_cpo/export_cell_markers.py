from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


from .paths import (
    CATALOG,
    CELL_MARKER_DIR,
    HGNC,
    MARKER_SCORES,
    MGI_MARKERS,
    MOUSE_HUMAN_ORTHOLOGS,
)


DEFAULT_HGNC_PATH = HGNC
DEFAULT_MOUSE_HUMAN_ORTHOLOG_PATH = MOUSE_HUMAN_ORTHOLOGS
DEFAULT_MGI_MARKER_PATH = MGI_MARKERS

CELL_MARKER_SUMMARY_FIELDS = [
    "catalog_order",
    "cl_id",
    "catalog_label",
    "cl_name",
    "catalog_label_zh",
    "parent_id",
    "parent_label",
    "branch",
    "catalog_rank",
    "marker_gene_count",
]

REJECTED_GENE_FIELDS = [
    "source_gene",
    "rejection_reason",
    "input_row_count",
]


@dataclass(frozen=True)
class GeneMapping:
    human_symbol: str
    method: str
    ortholog_mapping_type: str = ""


class GeneNormalizer:
    """Validate human symbols and translate mouse symbols into human genes."""

    def __init__(
        self,
        hgnc_path: Path,
        ortholog_path: Path,
        mouse_gene_path: Path | None = None,
    ) -> None:
        self.hgnc_path = Path(hgnc_path)
        self.ortholog_path = Path(ortholog_path)
        self.mouse_gene_path = Path(mouse_gene_path) if mouse_gene_path else None
        self.approved_exact: dict[str, str] = {}
        self.approved_folded: dict[str, set[str]] = defaultdict(set)
        self.previous_exact: dict[str, set[str]] = defaultdict(set)
        self.previous_folded: dict[str, set[str]] = defaultdict(set)
        self.alias_exact: dict[str, set[str]] = defaultdict(set)
        self.alias_folded: dict[str, set[str]] = defaultdict(set)
        self.mouse_exact: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.mouse_folded: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.mouse_alias_exact: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.mouse_alias_folded: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.ambiguous_mouse_alias_exact: set[str] = set()
        self.ambiguous_mouse_alias_folded: set[str] = set()
        self.orthologs_by_mgi: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.resource_counts: Counter[str] = Counter()
        self._load_hgnc()
        self._load_orthologs()
        if self.mouse_gene_path:
            self._load_mouse_aliases()

    @staticmethod
    def _split_symbols(value: str) -> list[str]:
        return [symbol.strip() for symbol in value.split("|") if symbol.strip()]

    def _load_hgnc(self) -> None:
        with self.hgnc_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require_columns(
                self.hgnc_path,
                reader.fieldnames,
                {"symbol", "status", "prev_symbol", "alias_symbol"},
            )
            approved_rows = [
                row
                for row in reader
                if row.get("status", "").strip() == "Approved"
                and row.get("symbol", "").strip()
            ]

        for row in approved_rows:
            symbol = row["symbol"].strip()
            self.approved_exact[symbol] = symbol
            self.approved_folded[symbol.casefold()].add(symbol)

        for row in approved_rows:
            symbol = row["symbol"].strip()
            for previous in self._split_symbols(row.get("prev_symbol", "")):
                self.previous_exact[previous].add(symbol)
                self.previous_folded[previous.casefold()].add(symbol)
            for alias in self._split_symbols(row.get("alias_symbol", "")):
                self.alias_exact[alias].add(symbol)
                self.alias_folded[alias.casefold()].add(symbol)

        self.resource_counts["hgnc_approved_symbols"] = len(self.approved_exact)

    @staticmethod
    def _single_mapping(
        mappings: Mapping[str, set[str]], key: str, method: str
    ) -> tuple[list[GeneMapping], str] | None:
        candidates = mappings.get(key)
        if not candidates:
            return None
        if len(candidates) != 1:
            return [], f"ambiguous_{method}"
        return [GeneMapping(next(iter(candidates)), method)], ""

    def _normalize_human_reference(self, symbol: str) -> str | None:
        if symbol in self.approved_exact:
            return self.approved_exact[symbol]
        approved = self._single_mapping(
            self.approved_folded, symbol.casefold(), "hgnc_approved_symbol"
        )
        if approved and approved[0]:
            return approved[0][0].human_symbol
        for exact, folded, method in (
            (self.previous_exact, self.previous_folded, "hgnc_previous_symbol"),
            (self.alias_exact, self.alias_folded, "hgnc_alias_symbol"),
        ):
            resolved = self._single_mapping(exact, symbol, method)
            if resolved and resolved[0]:
                return resolved[0][0].human_symbol
            resolved = self._single_mapping(folded, symbol.casefold(), method)
            if resolved and resolved[0]:
                return resolved[0][0].human_symbol
        return None

    def _load_orthologs(self) -> None:
        pairs: set[tuple[str, str]] = set()
        with self.ortholog_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require_columns(
                self.ortholog_path,
                reader.fieldnames,
                {"mouse_symbol", "human_symbol", "mapping_type"},
            )
            for row in reader:
                mouse_symbol = row.get("mouse_symbol", "").strip()
                human_symbol = self._normalize_human_reference(
                    row.get("human_symbol", "").strip()
                )
                mapping_type = row.get("mapping_type", "").strip() or "unspecified"
                mouse_mgi_id = row.get("mouse_mgi_id", "").strip()
                if not mouse_symbol or not human_symbol:
                    self.resource_counts["excluded_ortholog_rows"] += 1
                    continue
                self.mouse_exact[mouse_symbol][human_symbol].add(mapping_type)
                self.mouse_folded[mouse_symbol.casefold()][human_symbol].add(
                    mapping_type
                )
                if mouse_mgi_id:
                    self.orthologs_by_mgi[mouse_mgi_id][human_symbol].add(
                        mapping_type
                    )
                pairs.add((mouse_symbol, human_symbol))
        self.resource_counts["mouse_symbols_with_human_ortholog"] = len(
            self.mouse_exact
        )
        self.resource_counts["mouse_human_ortholog_pairs"] = len(pairs)

    def _load_mouse_aliases(self) -> None:
        aliases_exact: dict[str, set[str]] = defaultdict(set)
        aliases_folded: dict[str, set[str]] = defaultdict(set)
        with self.mouse_gene_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require_columns(
                self.mouse_gene_path,
                reader.fieldnames,
                {
                    "MGI Accession ID",
                    "Marker Symbol",
                    "Marker Type",
                    "Marker Synonyms (pipe-separated)",
                },
            )
            for row in reader:
                mgi_id = row.get("MGI Accession ID", "").strip()
                if (
                    row.get("Marker Type", "").strip() != "Gene"
                    or mgi_id not in self.orthologs_by_mgi
                ):
                    continue
                current_symbol = row.get("Marker Symbol", "").strip()
                for alias in self._split_symbols(
                    row.get("Marker Synonyms (pipe-separated)", "")
                ):
                    if alias == current_symbol:
                        continue
                    aliases_exact[alias].add(mgi_id)
                    aliases_folded[alias.casefold()].add(mgi_id)

        for alias, mgi_ids in aliases_exact.items():
            if len(mgi_ids) != 1:
                self.ambiguous_mouse_alias_exact.add(alias)
                continue
            self._add_mouse_alias_mapping(self.mouse_alias_exact, alias, mgi_ids)
        for alias, mgi_ids in aliases_folded.items():
            if len(mgi_ids) != 1:
                self.ambiguous_mouse_alias_folded.add(alias)
                continue
            self._add_mouse_alias_mapping(self.mouse_alias_folded, alias, mgi_ids)

        self.resource_counts["mouse_ortholog_aliases"] = len(
            self.mouse_alias_exact
        )
        self.resource_counts["ambiguous_mouse_ortholog_aliases"] = len(
            self.ambiguous_mouse_alias_exact
        )

    def _add_mouse_alias_mapping(
        self,
        destination: dict[str, dict[str, set[str]]],
        alias: str,
        mgi_ids: set[str],
    ) -> None:
        mgi_id = next(iter(mgi_ids))
        for human_symbol, mapping_types in self.orthologs_by_mgi[mgi_id].items():
            destination[alias][human_symbol].update(mapping_types)

    @staticmethod
    def _mouse_mappings(
        mappings: Mapping[str, Mapping[str, set[str]]], key: str, method: str
    ) -> tuple[list[GeneMapping], str] | None:
        candidates = mappings.get(key)
        if not candidates:
            return None
        return (
            [
                GeneMapping(
                    human_symbol=human_symbol,
                    method=method,
                    ortholog_mapping_type="|".join(sorted(mapping_types)),
                )
                for human_symbol, mapping_types in sorted(candidates.items())
            ],
            "",
        )

    def normalize(self, source_gene: str) -> tuple[list[GeneMapping], str]:
        source_gene = source_gene.strip()
        if not source_gene:
            return [], "empty_gene"

        if source_gene in self.approved_exact:
            return [
                GeneMapping(
                    self.approved_exact[source_gene], "hgnc_approved_symbol"
                )
            ], ""

        mouse = self._mouse_mappings(
            self.mouse_exact, source_gene, "mouse_ortholog"
        )
        if mouse:
            return mouse

        for mappings, method in (
            (self.previous_exact, "hgnc_previous_symbol"),
            (self.alias_exact, "hgnc_alias_symbol"),
        ):
            resolved = self._single_mapping(mappings, source_gene, method)
            if resolved:
                return resolved

        mouse = self._mouse_mappings(
            self.mouse_alias_exact, source_gene, "mouse_ortholog_alias"
        )
        if mouse:
            return mouse
        if source_gene in self.ambiguous_mouse_alias_exact:
            return [], "ambiguous_mouse_ortholog_alias"

        folded = source_gene.casefold()
        approved = self._single_mapping(
            self.approved_folded,
            folded,
            "hgnc_approved_symbol_case_normalized",
        )
        if approved:
            return approved

        mouse = self._mouse_mappings(
            self.mouse_folded, folded, "mouse_ortholog_case_normalized"
        )
        if mouse:
            return mouse

        for mappings, method in (
            (self.previous_folded, "hgnc_previous_symbol_case_normalized"),
            (self.alias_folded, "hgnc_alias_symbol_case_normalized"),
        ):
            resolved = self._single_mapping(mappings, folded, method)
            if resolved:
                return resolved

        mouse = self._mouse_mappings(
            self.mouse_alias_folded,
            folded,
            "mouse_ortholog_alias_case_normalized",
        )
        if mouse:
            return mouse
        if folded in self.ambiguous_mouse_alias_folded:
            return [], "ambiguous_mouse_ortholog_alias_case_normalized"

        return [], "not_hgnc_or_mouse_ortholog_gene"


def require_columns(
    path: Path, fieldnames: Sequence[str] | None, required: set[str]
) -> None:
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_columns(
            path,
            reader.fieldnames,
            {
                "order",
                "rank",
                "branch",
                "cl_id",
                "cl_name",
                "catalog_label",
                "catalog_label_zh",
                "parent_id",
                "parent_label",
            },
        )
        return list(reader)


def write_atomic_tsv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.suffix == ".gz":
        handle_context = gzip.open(
            temporary, "wt", encoding="utf-8", newline="", compresslevel=6
        )
    else:
        handle_context = temporary.open("w", encoding="utf-8", newline="")
    with handle_context as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_cell_markers(
    marker_path: Path,
    catalog_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    top_markers_per_cell: int = 50,
    human_gene_path: Path = DEFAULT_HGNC_PATH,
    ortholog_path: Path = DEFAULT_MOUSE_HUMAN_ORTHOLOG_PATH,
    mouse_gene_path: Path | None = DEFAULT_MGI_MARKER_PATH,
) -> dict[str, int | str]:
    if top_markers_per_cell < 1:
        raise ValueError("top_markers_per_cell must be at least 1")
    normalizer = GeneNormalizer(human_gene_path, ortholog_path, mouse_gene_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_output = output_dir / "cell_marker_genes.tsv.gz"
    gene_temporary = gene_output.with_suffix(gene_output.suffix + ".part")
    marker_groups: dict[str, tuple[int, Mapping[str, str]]] = {}
    seen_groups: set[str] = set()
    counts: Counter[str] = Counter()
    rejected_genes: Counter[tuple[str, str]] = Counter()

    with marker_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        require_columns(
            marker_path,
            reader.fieldnames,
            {
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
            },
        )
        marker_fields = list(reader.fieldnames or []) + [
            "source_marker_rank",
            "source_gene",
            "gene_mapping_method",
            "ortholog_mapping_type",
        ]
        with gzip.open(
            gene_temporary, "wt", encoding="utf-8", newline="", compresslevel=6
        ) as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=marker_fields,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            current_id = ""
            current_rows: list[tuple[int, int, Mapping[str, str]]] = []
            current_row: Mapping[str, str] = {}

            def flush_group() -> None:
                if not current_id:
                    return
                if current_id in seen_groups:
                    raise ValueError(
                        f"Marker input is not grouped by catalog_id: {current_id} reappeared"
                    )
                seen_groups.add(current_id)
                seen_human_genes: set[str] = set()
                candidates: list[tuple[int, dict[str, str]]] = []
                for source_rank, source_order, source_row in sorted(current_rows):
                    source_gene = source_row.get("gene", "").strip()
                    mappings, rejection_reason = normalizer.normalize(source_gene)
                    if not mappings:
                        if rejection_reason.startswith("ambiguous_"):
                            counts["excluded_ambiguous_gene_rows"] += 1
                        else:
                            counts["excluded_non_gene_rows"] += 1
                        rejected_genes[(source_gene, rejection_reason)] += 1
                        continue

                    counts[f"normalized_by_{mappings[0].method}_rows"] += 1
                    source_added = False
                    for mapping in mappings:
                        if mapping.human_symbol in seen_human_genes:
                            counts["excluded_duplicate_human_gene_candidates"] += 1
                            continue
                        seen_human_genes.add(mapping.human_symbol)
                        normalized = dict(source_row)
                        normalized.update(
                            {
                                "gene": mapping.human_symbol,
                                "source_marker_rank": str(source_rank),
                                "source_gene": source_gene,
                                "gene_mapping_method": mapping.method,
                                "ortholog_mapping_type": mapping.ortholog_mapping_type,
                            }
                        )
                        candidates.append((source_order, normalized))
                        source_added = True
                    if source_added:
                        counts["source_rows_with_clean_gene"] += 1

                selected = candidates[:top_markers_per_cell]
                selected_source_orders: set[int] = set()
                for clean_rank, (source_order, normalized) in enumerate(selected, 1):
                    normalized["marker_rank"] = str(clean_rank)
                    writer.writerow(normalized)
                    selected_source_orders.add(source_order)
                counts["selected_source_marker_rows"] += len(selected_source_orders)
                counts["selected_marker_rows"] += len(selected)
                counts["clean_marker_candidates"] += len(candidates)
                counts["excluded_after_top_n_candidates"] += max(
                    0, len(candidates) - top_markers_per_cell
                )
                marker_groups[current_id] = (len(selected), current_row)

            for source_order, row in enumerate(reader, 1):
                counts["input_marker_rows"] += 1
                cell_id = row.get("catalog_id", "").strip()
                gene = row.get("gene", "").strip()
                if not cell_id or not gene:
                    counts["incomplete_marker_rows"] += 1
                    rejected_genes[(gene, "missing_catalog_id_or_gene")] += 1
                    continue
                try:
                    marker_rank = int(row.get("marker_rank", ""))
                except ValueError as error:
                    raise ValueError(
                        f"{marker_path}:{reader.line_num} has invalid marker_rank"
                    ) from error
                if marker_rank < 1:
                    raise ValueError(
                        f"{marker_path}:{reader.line_num} has invalid marker_rank"
                    )
                if current_id and cell_id != current_id:
                    flush_group()
                    current_rows = []
                if cell_id != current_id:
                    current_id = cell_id
                    current_row = row
                current_rows.append((marker_rank, source_order, row))
            flush_group()
    gene_temporary.replace(gene_output)

    rejected_output = output_dir / "cell_marker_rejected_genes.tsv.gz"
    write_atomic_tsv(
        rejected_output,
        REJECTED_GENE_FIELDS,
        (
            {
                "source_gene": source_gene,
                "rejection_reason": reason,
                "input_row_count": row_count,
            }
            for (source_gene, reason), row_count in sorted(
                rejected_genes.items(),
                key=lambda item: (-item[1], item[0][1], item[0][0]),
            )
        ),
    )

    catalog_ids = {row["cl_id"] for row in catalog_rows}
    summary_rows: list[dict[str, object]] = []
    for catalog_row in catalog_rows:
        cell_id = catalog_row["cl_id"]
        marker_count, _ = marker_groups.get(cell_id, (0, {}))
        summary_rows.append(
            {
                "catalog_order": catalog_row["order"],
                "cl_id": cell_id,
                "catalog_label": catalog_row["catalog_label"],
                "cl_name": catalog_row["cl_name"],
                "catalog_label_zh": catalog_row["catalog_label_zh"],
                "parent_id": catalog_row["parent_id"],
                "parent_label": catalog_row["parent_label"],
                "branch": catalog_row["branch"],
                "catalog_rank": catalog_row["rank"],
                "marker_gene_count": marker_count,
            }
        )

    unknown_cell_ids = sorted(set(marker_groups) - catalog_ids)
    for cell_id in unknown_cell_ids:
        marker_count, marker_row = marker_groups[cell_id]
        summary_rows.append(
            {
                "catalog_order": marker_row.get("catalog_order", ""),
                "cl_id": cell_id,
                "catalog_label": marker_row.get("catalog_label", ""),
                "cl_name": marker_row.get("catalog_cl_name", ""),
                "catalog_label_zh": marker_row.get("catalog_label_zh", ""),
                "parent_id": marker_row.get("parent_id", ""),
                "parent_label": marker_row.get("parent_label", ""),
                "branch": marker_row.get("branch", ""),
                "catalog_rank": marker_row.get("catalog_rank", ""),
                "marker_gene_count": marker_count,
            }
        )

    summary_output = output_dir / "cell_marker_summary.tsv"
    write_atomic_tsv(summary_output, CELL_MARKER_SUMMARY_FIELDS, summary_rows)
    counts["catalog_cells"] = len(catalog_rows)
    counts["cells_with_marker_genes"] = sum(
        bool(marker_count) for marker_count, _ in marker_groups.values()
    )
    counts["catalog_cells_without_marker_genes"] = sum(
        not marker_groups.get(row["cl_id"], (0, {}))[0] for row in catalog_rows
    )
    counts["unknown_marker_cell_ids"] = len(unknown_cell_ids)
    counts["top_markers_per_cell"] = top_markers_per_cell
    counts["unique_cell_marker_pairs"] = sum(
        marker_count for marker_count, _ in marker_groups.values()
    )
    counts["excluded_marker_rows"] = (
        counts["input_marker_rows"] - counts["selected_source_marker_rows"]
    )
    return {
        **dict(normalizer.resource_counts),
        **dict(counts),
        "gene_output": str(gene_output.resolve()),
        "rejected_gene_output": str(rejected_output.resolve()),
        "summary_output": str(summary_output.resolve()),
    }


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    catalog_rows = read_catalog(args.catalog)
    marker_counts = export_cell_markers(
        args.markers,
        catalog_rows,
        args.output_dir,
        args.top_markers_per_cell,
        args.human_genes,
        args.mouse_human_orthologs,
        args.mouse_genes,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": {
            "definition": (
                "All catalog CL IDs are retained. Marker values are normalized to current "
                "HGNC human symbols, including mouse-to-human ortholog translation, before "
                "each cell keeps its top-ranked unique genes."
            ),
            "intersection_performed": False,
            "go_to_cl_constraints_applied": False,
            "marker_ranking": (
                "Source marker_rank from expanded_marker_scores.tsv is preserved in "
                "source_marker_rank. marker_rank is reassigned after gene normalization, "
                "non-gene rejection, and human-symbol deduplication."
            ),
            "gene_normalization": (
                "Accept current HGNC symbols; resolve unambiguous previous/alias symbols; "
                "translate current and unambiguous historical MGI mouse symbols through "
                "the mouse-human ortholog table; reject values that cannot be resolved; "
                "deduplicate by current human symbol."
            ),
            "marker_top_n": args.top_markers_per_cell,
        },
        "inputs": {
            "catalog": str(args.catalog.resolve()),
            "markers": str(args.markers.resolve()),
            "human_genes": str(args.human_genes.resolve()),
            "mouse_human_orthologs": str(args.mouse_human_orthologs.resolve()),
            "mouse_genes": str(args.mouse_genes.resolve()),
        },
        "cell_marker": marker_counts,
    }
    write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export each catalog cell's top-ranked literature marker genes."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG,
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=MARKER_SCORES,
        help="Ranked CL-marker annotations used to select each cell's top markers.",
    )
    parser.add_argument(
        "--top-markers-per-cell",
        type=int,
        default=50,
        help="Maximum ranked marker genes retained per cell (default: 50).",
    )
    parser.add_argument(
        "--human-genes",
        type=Path,
        default=DEFAULT_HGNC_PATH,
        help="HGNC complete-set TSV used to validate and canonicalize human symbols.",
    )
    parser.add_argument(
        "--mouse-human-orthologs",
        type=Path,
        default=DEFAULT_MOUSE_HUMAN_ORTHOLOG_PATH,
        help="TSV with mouse_symbol, human_symbol, and mapping_type columns.",
    )
    parser.add_argument(
        "--mouse-genes",
        type=Path,
        default=DEFAULT_MGI_MARKER_PATH,
        help="MGI marker report used to resolve historical mouse gene symbols.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CELL_MARKER_DIR,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CELL_MARKER_DIR / "cell_marker_manifest.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
