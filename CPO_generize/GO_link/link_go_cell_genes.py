from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celltype_configure.common import (  # noqa: E402
    CellCatalog,
    load_obo,
    ontology_ancestors,
    write_tsv,
)


SELECTED_MAPPING_STATUSES = {
    "cell_type_specific",
    "cell_type_specific_candidate",
}
EXPERIMENTAL_EVIDENCE_CODES = {
    "EXP",
    "IDA",
    "IEP",
    "IGI",
    "IMP",
    "IPI",
    "HDA",
    "HEP",
    "HGI",
    "HMP",
    "HTP",
}
RELATION_PRIORITY = {"self": 0, "descendant": 1, "ancestor": 2}
CONFIDENCE_PRIORITY = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class GeneMapping:
    approved_symbol: str
    method: str


@dataclass(frozen=True)
class CellGoRelation:
    relation: str
    distance: int
    root_ids: tuple[str, ...]


class HgncNormalizer:
    def __init__(self, rows: Iterable[Mapping[str, str]]) -> None:
        self.approved: set[str] = set()
        aliases: dict[str, set[str]] = defaultdict(set)
        previous: dict[str, set[str]] = defaultdict(set)
        folded_approved: dict[str, set[str]] = defaultdict(set)
        folded_aliases: dict[str, set[str]] = defaultdict(set)
        folded_previous: dict[str, set[str]] = defaultdict(set)

        for row in rows:
            if row.get("status") != "Approved":
                continue
            symbol = row.get("symbol", "").strip()
            if not symbol:
                continue
            self.approved.add(symbol)
            folded_approved[symbol.casefold()].add(symbol)
            for alias in split_pipe_values(row.get("alias_symbol", "")):
                aliases[alias].add(symbol)
                folded_aliases[alias.casefold()].add(symbol)
            for old_symbol in split_pipe_values(row.get("prev_symbol", "")):
                previous[old_symbol].add(symbol)
                folded_previous[old_symbol.casefold()].add(symbol)

        self.aliases = unique_values(aliases)
        self.previous = unique_values(previous)
        self.folded_approved = unique_values(folded_approved)
        self.folded_aliases = unique_values(folded_aliases)
        self.folded_previous = unique_values(folded_previous)

    @classmethod
    def from_tsv(cls, path: str | Path) -> "HgncNormalizer":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return cls(csv.DictReader(handle, delimiter="\t"))

    def normalize(self, raw_symbol: str) -> GeneMapping | None:
        symbol = raw_symbol.strip()
        if not symbol:
            return None
        if symbol in self.approved:
            return GeneMapping(symbol, "approved_exact")
        if symbol in self.aliases:
            return GeneMapping(self.aliases[symbol], "alias_exact")
        if symbol in self.previous:
            return GeneMapping(self.previous[symbol], "previous_exact")

        folded = symbol.casefold()
        if folded in self.folded_approved:
            return GeneMapping(self.folded_approved[folded], "approved_casefold")
        if folded in self.folded_aliases:
            return GeneMapping(self.folded_aliases[folded], "alias_casefold")
        if folded in self.folded_previous:
            return GeneMapping(self.folded_previous[folded], "previous_casefold")
        return None


def split_pipe_values(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def unique_values(values: Mapping[str, set[str]]) -> dict[str, str]:
    return {
        key: next(iter(candidates))
        for key, candidates in values.items()
        if len(candidates) == 1
    }


def read_dict_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_columns(rows: Sequence[Mapping[str, str]], columns: set[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} contains no data rows")
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def catalog_distance(catalog: CellCatalog, ancestor: str, descendant: str) -> int:
    if ancestor == descendant:
        return 0
    for distance, candidate in enumerate(catalog.ancestors(descendant), start=1):
        if candidate == ancestor:
            return distance
    return -1


def merge_relation(
    existing: CellGoRelation | None,
    relation: str,
    distance: int,
    root_id: str,
) -> CellGoRelation:
    if existing is None or RELATION_PRIORITY[relation] < RELATION_PRIORITY[existing.relation]:
        return CellGoRelation(relation, distance, (root_id,))
    if relation != existing.relation:
        return existing
    roots = tuple(sorted(set(existing.root_ids) | {root_id}))
    best_distance = min(value for value in (existing.distance, distance) if value >= 0)
    return CellGoRelation(relation, best_distance, roots)


def build_cell_go_relations(
    classification_rows: Sequence[Mapping[str, str]],
    closure_rows: Sequence[Mapping[str, str]],
    context_rows: Sequence[Mapping[str, str]],
    catalog: CellCatalog,
    include_ancestor_context: bool,
) -> tuple[dict[str, dict[str, CellGoRelation]], dict[str, Mapping[str, str]]]:
    classifications = {
        row["go_id"]: row
        for row in classification_rows
        if row.get("classification") in SELECTED_MAPPING_STATUSES
    }
    by_cell: dict[str, dict[str, CellGoRelation]] = defaultdict(dict)

    for row in closure_rows:
        go_id = row.get("go_id", "")
        if go_id not in classifications:
            continue
        cell_id = row.get("allowed_cl_id", "")
        root_id = row.get("constraint_root_cl_id", "")
        if cell_id not in catalog.nodes or root_id not in catalog.nodes:
            continue
        relation = "self" if cell_id == root_id else "descendant"
        distance = catalog_distance(catalog, root_id, cell_id)
        by_cell[cell_id][go_id] = merge_relation(
            by_cell[cell_id].get(go_id), relation, distance, root_id
        )

    if include_ancestor_context:
        for row in context_rows:
            go_id = row.get("go_id", "")
            if go_id not in classifications:
                continue
            cell_id = row.get("ancestor_cl_id", "")
            root_id = row.get("constraint_root_cl_id", "")
            if cell_id == "CL:0000000" or cell_id not in catalog.nodes:
                continue
            distance = int(row.get("distance", "0") or 0)
            by_cell[cell_id][go_id] = merge_relation(
                by_cell[cell_id].get(go_id), "ancestor", distance, root_id
            )
    return dict(by_cell), classifications


def marker_mapping_quality(method: str) -> int:
    if method == "approved_exact":
        return 0
    if method in {"alias_exact", "previous_exact"}:
        return 1
    return 2


def normalize_markers(
    marker_rows: Sequence[Mapping[str, str]],
    relevant_cells: set[str],
    normalizer: HgncNormalizer,
) -> tuple[list[dict[str, str]], Counter[str], int]:
    counts: Counter[str] = Counter()
    best: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_count = 0
    for row in marker_rows:
        cell_id = row.get("catalog_id", "")
        if cell_id not in relevant_cells:
            counts["outside_constraint_cells"] += 1
            continue
        mapping = normalizer.normalize(row.get("gene", ""))
        if mapping is None:
            counts["unmapped"] += 1
            continue
        counts[mapping.method] += 1
        normalized = dict(row)
        normalized["human_gene_symbol"] = mapping.approved_symbol
        normalized["gene_mapping_method"] = mapping.method
        key = (cell_id, mapping.approved_symbol)
        current = best.get(key)
        if current is None:
            best[key] = normalized
            continue
        duplicate_count += 1
        candidate_key = (
            marker_mapping_quality(mapping.method),
            -safe_float(normalized.get("significance_score", "")),
            -safe_float(normalized.get("marker_score", "")),
        )
        current_key = (
            marker_mapping_quality(current["gene_mapping_method"]),
            -safe_float(current.get("significance_score", "")),
            -safe_float(current.get("marker_score", "")),
        )
        if candidate_key < current_key:
            best[key] = normalized
    return list(best.values()), counts, duplicate_count


def load_goa_associations(
    gaf_path: str | Path,
    go_graph,
    selected_go_ids: set[str],
    marker_symbols: set[str],
    propagate_to_ancestors: bool = True,
) -> tuple[dict[tuple[str, str], dict[str, set[str]]], Counter[str]]:
    associations: dict[tuple[str, str], dict[str, set[str]]] = {}
    counts: Counter[str] = Counter()

    @lru_cache(maxsize=None)
    def selected_targets(direct_go_id: str) -> frozenset[str]:
        if not propagate_to_ancestors:
            return (
                frozenset({direct_go_id})
                if direct_go_id in selected_go_ids
                else frozenset()
            )
        if direct_go_id not in go_graph:
            return frozenset()
        return frozenset(
            ontology_ancestors(go_graph, direct_go_id, include_self=True)
            & selected_go_ids
        )

    with Path(gaf_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                counts["malformed"] += 1
                continue
            if fields[8] != "P":
                continue
            qualifiers = set(split_pipe_values(fields[3]))
            if "NOT" in qualifiers:
                counts["not_qualifier"] += 1
                continue
            gene = fields[2]
            if gene not in marker_symbols:
                continue
            counts["marker_gene_bp_annotations"] += 1
            direct_go_id = fields[4]
            matched_go_ids = selected_targets(direct_go_id)
            if not matched_go_ids:
                continue
            counts["annotations_reaching_selected_go"] += 1
            for go_id in matched_go_ids:
                key = (gene, go_id)
                evidence = associations.setdefault(
                    key,
                    {
                        "direct_go_ids": set(),
                        "evidence_codes": set(),
                        "db_object_ids": set(),
                        "references": set(),
                    },
                )
                evidence["direct_go_ids"].add(direct_go_id)
                evidence["evidence_codes"].add(fields[6])
                evidence["db_object_ids"].add(f"{fields[0]}:{fields[1]}")
                evidence["references"].update(split_pipe_values(fields[5]))
    counts["gene_go_associations"] = len(associations)
    return associations, counts


def gene_evidence_score(marker: Mapping[str, str], relation: str, mapping_status: str) -> float:
    significance = min(
        safe_float(
            marker.get("significance_score", "")
            or marker.get("marker_significance_score", "")
        ),
        20.0,
    )
    marker_score = min(safe_float(marker.get("marker_score", "")), 20.0) / 4.0
    fold = max(safe_float(marker.get("enrichment_fold", ""), 1.0), 1.0)
    fold_component = min(math.log2(fold), 5.0)
    relation_factor = {"self": 1.0, "descendant": 0.95, "ancestor": 0.70}[relation]
    mapping_factor = 1.0 if mapping_status == "cell_type_specific" else 0.75
    return (significance + marker_score + fold_component) * relation_factor * mapping_factor


def link_confidence(
    mapping_status: str,
    relation: str,
    supporting_gene_count: int,
    experimental_gene_count: int,
    exact_symbol_gene_count: int,
) -> str:
    if mapping_status == "cell_type_specific":
        if relation != "ancestor" and supporting_gene_count >= 2 and (
            experimental_gene_count >= 1 or supporting_gene_count >= 3
        ):
            return "high"
        return "medium"
    if (
        relation != "ancestor"
        and supporting_gene_count >= 2
        and experimental_gene_count >= 1
        and exact_symbol_gene_count >= 1
    ):
        return "medium"
    return "low"


def build_links(
    markers: Sequence[Mapping[str, str]],
    cell_go_relations: Mapping[str, Mapping[str, CellGoRelation]],
    classifications: Mapping[str, Mapping[str, str]],
    associations: Mapping[tuple[str, str], Mapping[str, set[str]]],
    catalog: CellCatalog,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_gene: dict[str, set[str]] = defaultdict(set)
    for gene, go_id in associations:
        by_gene[gene].add(go_id)

    gene_rows: list[dict] = []
    for marker in markers:
        cell_id = marker["catalog_id"]
        gene = marker["human_gene_symbol"]
        compatible_go_ids = by_gene.get(gene, set()) & set(cell_go_relations[cell_id])
        for go_id in compatible_go_ids:
            relation = cell_go_relations[cell_id][go_id]
            classification = classifications[go_id]
            goa = associations[(gene, go_id)]
            evidence_codes = sorted(goa["evidence_codes"])
            root_ids = relation.root_ids
            row = {
                "cell_cl_id": cell_id,
                "cell_label": marker.get("catalog_label", ""),
                "cell_cl_name": marker.get("catalog_cl_name", ""),
                "cell_label_zh": marker.get("catalog_label_zh", ""),
                "cell_branch": marker.get("branch", ""),
                "cell_catalog_order": marker.get("catalog_order", ""),
                "cell_catalog_rank": marker.get("catalog_rank", ""),
                "go_id": go_id,
                "go_name": classification.get("go_name", ""),
                "go_classification": classification.get("classification", ""),
                "go_constraint_confidence": classification.get("confidence", ""),
                "go_constraint_review_required": classification.get("review_required", ""),
                "constraint_root_cl_ids": "|".join(root_ids),
                "constraint_root_labels": "|".join(
                    catalog.nodes[root].catalog_label
                    for root in root_ids
                    if root in catalog.nodes
                ),
                "relation_to_constraint": relation.relation,
                "relation_distance": relation.distance,
                "marker_gene": marker.get("gene", ""),
                "human_gene_symbol": gene,
                "gene_mapping_method": marker.get("gene_mapping_method", ""),
                "marker_rank": marker.get("significant_rank", ""),
                "marker_score": marker.get("marker_score", ""),
                "statistics_score": marker.get("statistics_score", ""),
                "signatures_score": marker.get("signatures_score", ""),
                "cell_prevalence": marker.get("cell_prevalence", ""),
                "background_prevalence": marker.get("background_prevalence", ""),
                "enrichment_fold": marker.get("enrichment_fold", ""),
                "odds_ratio": marker.get("odds_ratio", ""),
                "marker_p_value": marker.get("p_value", ""),
                "marker_q_value": marker.get("q_value", ""),
                "marker_significance_score": marker.get("significance_score", ""),
                "goa_direct_go_ids": "|".join(sorted(goa["direct_go_ids"])),
                "goa_direct_match": str(go_id in goa["direct_go_ids"]).lower(),
                "goa_evidence_codes": "|".join(evidence_codes),
                "goa_has_experimental_evidence": str(
                    bool(set(evidence_codes) & EXPERIMENTAL_EVIDENCE_CODES)
                ).lower(),
                "goa_db_object_ids": "|".join(sorted(goa["db_object_ids"])),
                "goa_references": "|".join(sorted(goa["references"])),
            }
            row["gene_evidence_score"] = f"{gene_evidence_score(row, relation.relation, row['go_classification']):.6f}"
            gene_rows.append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in gene_rows:
        grouped[(row["cell_cl_id"], row["go_id"])].append(row)

    pair_rows: list[dict] = []
    for _, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                -safe_float(row["gene_evidence_score"]),
                row["human_gene_symbol"],
            ),
        )
        first = ordered[0]
        experimental_count = sum(
            row["goa_has_experimental_evidence"] == "true" for row in ordered
        )
        direct_count = sum(row["goa_direct_match"] == "true" for row in ordered)
        exact_count = sum(
            row["gene_mapping_method"] == "approved_exact" for row in ordered
        )
        confidence = link_confidence(
            first["go_classification"],
            first["relation_to_constraint"],
            len(ordered),
            experimental_count,
            exact_count,
        )
        review_required = (
            first["go_constraint_review_required"] == "true"
            or first["relation_to_constraint"] == "ancestor"
            or confidence == "low"
        )
        pair_rows.append(
            {
                "cell_cl_id": first["cell_cl_id"],
                "cell_label": first["cell_label"],
                "cell_cl_name": first["cell_cl_name"],
                "cell_label_zh": first["cell_label_zh"],
                "cell_branch": first["cell_branch"],
                "cell_catalog_order": first["cell_catalog_order"],
                "cell_catalog_rank": first["cell_catalog_rank"],
                "go_id": first["go_id"],
                "go_name": first["go_name"],
                "go_classification": first["go_classification"],
                "go_constraint_confidence": first["go_constraint_confidence"],
                "constraint_root_cl_ids": first["constraint_root_cl_ids"],
                "constraint_root_labels": first["constraint_root_labels"],
                "relation_to_constraint": first["relation_to_constraint"],
                "relation_distance": first["relation_distance"],
                "link_confidence": confidence,
                "review_required": str(review_required).lower(),
                "evidence_score": f"{sum(safe_float(row['gene_evidence_score']) for row in ordered):.6f}",
                "supporting_gene_count": len(ordered),
                "supporting_genes": "|".join(
                    row["human_gene_symbol"] for row in ordered
                ),
                "experimental_goa_gene_count": experimental_count,
                "direct_go_annotation_gene_count": direct_count,
                "exact_hgnc_symbol_gene_count": exact_count,
                "best_marker_q_value": min(
                    safe_float(row["marker_q_value"], 1.0) for row in ordered
                ),
                "max_marker_score": max(
                    safe_float(row["marker_score"]) for row in ordered
                ),
                "max_marker_significance_score": max(
                    safe_float(row["marker_significance_score"]) for row in ordered
                ),
                "max_enrichment_fold": max(
                    safe_float(row["enrichment_fold"]) for row in ordered
                ),
                "interpretation": (
                    "candidate process in this cell catalog node"
                    if first["relation_to_constraint"] != "ancestor"
                    else "candidate process in a constrained descendant of this broad cell node"
                ),
            }
        )

    pair_rows.sort(
        key=lambda row: (
            int(row["cell_catalog_order"])
            if str(row["cell_catalog_order"]).isdigit()
            else 999999,
            row["cell_label"],
            CONFIDENCE_PRIORITY[row["link_confidence"]],
            -safe_float(row["evidence_score"]),
            row["go_id"],
        )
    )
    gene_rows.sort(
        key=lambda row: (
            row["cell_cl_id"],
            row["go_id"],
            -safe_float(row["gene_evidence_score"]),
            row["human_gene_symbol"],
        )
    )

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        by_cell[row["cell_cl_id"]].append(row)
    cell_rows: list[dict] = []
    for rows in by_cell.values():
        first = rows[0]
        all_genes = {
            gene for row in rows for gene in split_pipe_values(row["supporting_genes"])
        }
        top = sorted(rows, key=lambda row: -safe_float(row["evidence_score"]))[:10]
        cell_rows.append(
            {
                "cell_cl_id": first["cell_cl_id"],
                "cell_label": first["cell_label"],
                "cell_cl_name": first["cell_cl_name"],
                "cell_label_zh": first["cell_label_zh"],
                "cell_branch": first["cell_branch"],
                "cell_catalog_order": first["cell_catalog_order"],
                "linked_go_process_count": len(rows),
                "high_confidence_count": sum(
                    row["link_confidence"] == "high" for row in rows
                ),
                "medium_confidence_count": sum(
                    row["link_confidence"] == "medium" for row in rows
                ),
                "low_confidence_count": sum(
                    row["link_confidence"] == "low" for row in rows
                ),
                "confirmed_constraint_count": sum(
                    row["go_classification"] == "cell_type_specific" for row in rows
                ),
                "candidate_constraint_count": sum(
                    row["go_classification"] == "cell_type_specific_candidate"
                    for row in rows
                ),
                "supporting_gene_count": len(all_genes),
                "supporting_genes": "|".join(sorted(all_genes)),
                "top_go_ids": "|".join(row["go_id"] for row in top),
                "top_go_names": "|".join(row["go_name"] for row in top),
            }
        )
    cell_rows.sort(
        key=lambda row: (-int(row["linked_go_process_count"]), row["cell_label"])
    )
    return gene_rows, pair_rows, cell_rows


def run(args: argparse.Namespace) -> None:
    classification_rows = read_dict_tsv(args.classification)
    closure_rows = read_dict_tsv(args.closure)
    context_rows = read_dict_tsv(args.context)
    marker_rows = read_dict_tsv(args.markers)
    require_columns(
        classification_rows,
        {"go_id", "go_name", "classification", "confidence"},
        "classification TSV",
    )
    require_columns(
        marker_rows,
        {
            "catalog_id",
            "catalog_label",
            "gene",
            "marker_score",
            "enrichment_fold",
            "q_value",
        },
        "marker TSV",
    )

    catalog = CellCatalog.from_tsv(args.catalog)
    cell_go_relations, classifications = build_cell_go_relations(
        classification_rows,
        closure_rows,
        context_rows,
        catalog,
        include_ancestor_context=not args.exclude_ancestor_context,
    )
    normalizer = HgncNormalizer.from_tsv(args.hgnc)
    markers, normalization_counts, duplicate_count = normalize_markers(
        marker_rows, set(cell_go_relations), normalizer
    )
    marker_symbols = {row["human_gene_symbol"] for row in markers}
    go_graph = load_obo(args.go_obo)
    associations, goa_counts = load_goa_associations(
        args.gaf,
        go_graph,
        set(classifications),
        marker_symbols,
        propagate_to_ancestors=not args.direct_goa_only,
    )
    gene_rows, pair_rows, cell_rows = build_links(
        markers, cell_go_relations, classifications, associations, catalog
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if gene_rows:
        write_tsv(output_dir / "cell_go_gene_links.tsv", list(gene_rows[0]), gene_rows)
    if pair_rows:
        write_tsv(output_dir / "cell_go_process_summary.tsv", list(pair_rows[0]), pair_rows)
    if cell_rows:
        write_tsv(output_dir / "cell_process_summary.tsv", list(cell_rows[0]), cell_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition": (
            "Candidate GO biological-process phenotypes supported jointly by a GO-to-CL "
            "constraint, a statistically significant cell marker, and human GOA."
        ),
        "inputs": {
            "classification": str(Path(args.classification).resolve()),
            "closure": str(Path(args.closure).resolve()),
            "context": str(Path(args.context).resolve()),
            "markers": str(Path(args.markers).resolve()),
            "hgnc": str(Path(args.hgnc).resolve()),
            "gaf": str(Path(args.gaf).resolve()),
            "go_obo": str(Path(args.go_obo).resolve()),
            "catalog": str(Path(args.catalog).resolve()),
        },
        "settings": {
            "mapping_statuses": sorted(SELECTED_MAPPING_STATUSES),
            "include_ancestor_context": not args.exclude_ancestor_context,
            "goa_annotation_mode": (
                "direct_only" if args.direct_goa_only else "self_and_go_ancestors"
            ),
            "ancestor_cell_interpretation": (
                "A link to an ancestor catalog node means a compatible constrained descendant "
                "may carry the process; it is not a claim about every cell in the broad node."
            ),
            "gene_support": (
                "The marker's HGNC-normalized human gene must have a positive human GOA BP "
                + (
                    "annotation directly to the selected GO term."
                    if args.direct_goa_only
                    else "annotation to the selected GO term or one of its GO descendants."
                )
            ),
            "evidence_score": (
                "Ranking score from capped marker -log10(q), marker score, enrichment fold, "
                "CL relation, and GO mapping status; it is not a probability."
            ),
        },
        "counts": {
            "input_marker_rows": len(marker_rows),
            "input_marker_cells": len({row["catalog_id"] for row in marker_rows}),
            "selected_constrained_go_terms": len(classifications),
            "constraint_compatible_catalog_cells": len(cell_go_relations),
            "normalized_eligible_marker_rows": len(markers),
            "normalized_marker_symbols": len(marker_symbols),
            "deduplicated_cell_gene_rows": duplicate_count,
            "goa_gene_go_associations": len(associations),
            "cell_go_gene_links": len(gene_rows),
            "cell_go_process_links": len(pair_rows),
            "linked_cells": len(cell_rows),
            "linked_go_terms": len({row["go_id"] for row in pair_rows}),
        },
        "marker_normalization_counts": dict(sorted(normalization_counts.items())),
        "goa_counts": dict(sorted(goa_counts.items())),
        "cell_go_confidence_counts": dict(
            sorted(Counter(row["link_confidence"] for row in pair_rows).items())
        ),
        "cell_go_mapping_status_counts": dict(
            sorted(Counter(row["go_classification"] for row in pair_rows).items())
        ),
        "cell_go_relation_counts": dict(
            sorted(Counter(row["relation_to_constraint"] for row in pair_rows).items())
        ),
    }
    manifest_path = output_dir / "cell_go_gene_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Link constrained GO BP terms to catalog cells and significant marker genes."
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=root / "GO_link/work/legacy/final/go_classification.tsv",
    )
    parser.add_argument(
        "--closure",
        type=Path,
        default=root / "GO_link/work/legacy/final/go_cl_constraint_closure.tsv",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=root / "GO_link/work/legacy/final/go_cl_constraint_context.tsv",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=Path(
            "/path/to/workspace/marker_literuture/marker_statistics/"
            "significant_marker_scores.tsv"
        ),
    )
    parser.add_argument(
        "--hgnc",
        type=Path,
        default=Path(
            "/path/to/workspace/UHPO（6-8）/marker-agent/data/hgnc/"
            "hgnc_complete_set.txt"
        ),
    )
    parser.add_argument(
        "--gaf",
        type=Path,
        default=Path("/path/to/workspace/UHPO（6-8）/data/go/goa_human.gaf"),
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=Path("/path/to/workspace/HPO_expand/CPO/data/go/go-basic.obo"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "/path/to/workspace/marker_literuture/expanded/expanded_cell_catalog.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "GO_link/work/legacy/cell_go_gene"
    )
    parser.add_argument(
        "--exclude-ancestor-context",
        action="store_true",
        help="Only join exact constraint roots and their descendants.",
    )
    parser.add_argument(
        "--direct-goa-only",
        action="store_true",
        help="Do not propagate gene annotations from GO child terms to selected GO ancestors.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
