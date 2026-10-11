from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from scipy.stats import hypergeom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celltype_configure.common import (  # noqa: E402
    CellCatalog,
    load_json,
    load_obo,
    nearest_catalog_ancestor,
    normalize_cl_id,
    ontology_ancestors,
    read_go_ids,
    write_tsv,
)


def read_gene_map(path: str | None, source_column: str, symbol_column: str) -> dict[str, str]:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        missing = {source_column, symbol_column} - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Gene ID map is missing columns: {sorted(missing)}")
        result: dict[str, str] = {}
        for row in rows:
            source = row[source_column].strip()
            symbol = row[symbol_column].strip()
            if source and symbol:
                result[source] = symbol
                result[source.split(".", 1)[0]] = symbol
        return result


def map_gene(value: str, gene_map: Mapping[str, str]) -> str:
    gene_id = value.strip()
    if not gene_id:
        return ""
    if not gene_map:
        return gene_id
    return gene_map.get(gene_id, gene_map.get(gene_id.split(".", 1)[0], ""))


def load_cell_labels(
    path: str,
    label_column: str,
    ontology_column: str,
    cl_graph,
    catalog: CellCatalog,
) -> tuple[dict[str, dict], list[str]]:
    labels: dict[str, dict] = {}
    skipped: list[str] = []
    raw_ids: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        missing = {label_column, ontology_column} - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Cell metadata is missing columns: {sorted(missing)}")
        for row in rows:
            label = row[label_column].strip()
            if not label:
                continue
            counts[label] += 1
            for value in re.split(r"[|;,]", row[ontology_column]):
                cl_id = normalize_cl_id(value)
                if cl_id:
                    raw_ids[label].add(cl_id)
    for label, source_ids in raw_ids.items():
        projected: list[str] = []
        for source_id in sorted(source_ids):
            cl_id, _ = nearest_catalog_ancestor(cl_graph, catalog, source_id)
            if cl_id and cl_id != "CL:0000000" and cl_id not in projected:
                projected.append(cl_id)
        if not projected:
            skipped.append(label)
            continue
        catalog_cl_id = projected[0] if len(projected) == 1 else catalog.common_ancestor(projected)
        if not catalog_cl_id:
            skipped.append(label)
            continue
        node = catalog.nodes[catalog_cl_id]
        labels[label] = {
            "source_cl_ids": "|".join(sorted(source_ids)),
            "catalog_cl_id": catalog_cl_id,
            "catalog_cl_label": node.catalog_label,
            "catalog_anchor_id": node.anchor_id,
            "catalog_anchor_label": node.anchor_label,
            "n_cells": counts[label],
        }
    return labels, skipped


def load_marker_sets(
    path: str,
    gene_map: Mapping[str, str],
    cluster_column: str,
    gene_column: str,
    rank_column: str,
    logfc_column: str,
    pvalue_column: str,
    top_n: int,
    max_marker_p: float,
    min_logfc: float,
) -> tuple[dict[str, set[str]], set[str], int]:
    marker_sets: dict[str, set[str]] = defaultdict(set)
    seen_source_genes: set[str] = set()
    mapped_count = 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        required = {cluster_column, gene_column, rank_column, logfc_column, pvalue_column}
        missing = required - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Marker table is missing columns: {sorted(missing)}")
        for row in rows:
            try:
                rank = int(float(row[rank_column]))
                logfc = float(row[logfc_column])
                pvalue = float(row[pvalue_column])
            except (TypeError, ValueError):
                continue
            if rank >= top_n or logfc < min_logfc or pvalue > max_marker_p:
                continue
            source_gene = row[gene_column].strip()
            seen_source_genes.add(source_gene)
            gene = map_gene(source_gene, gene_map)
            if gene:
                mapped_count += int(gene != source_gene or not gene_map)
                marker_sets[row[cluster_column].strip()].add(gene)
    return marker_sets, seen_source_genes, mapped_count


def read_background_genes(path: str | None, gene_map: Mapping[str, str]) -> set[str]:
    if not path:
        return set()
    genes: set[str] = set()
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            source_id = line.rstrip("\n").split("\t", 1)[0]
            gene = map_gene(source_id, gene_map)
            if gene:
                genes.add(gene)
    return genes


def build_go_gene_sets(
    gaf_path: str, selected_go_ids: set[str], go_graph
) -> tuple[dict[str, set[str]], int]:
    gene_sets: dict[str, set[str]] = defaultdict(set)
    propagated_cache: dict[str, set[str]] = {}
    annotation_count = 0
    with Path(gaf_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[8] != "P" or "NOT" in fields[3].split("|"):
                continue
            gene_symbol = fields[2].strip()
            direct_go_id = fields[4].strip()
            if not gene_symbol or direct_go_id not in go_graph:
                continue
            if direct_go_id not in propagated_cache:
                propagated_cache[direct_go_id] = selected_go_ids.intersection(
                    ontology_ancestors(go_graph, direct_go_id, include_self=True)
                )
            for go_id in propagated_cache[direct_go_id]:
                gene_sets[go_id].add(gene_symbol)
                annotation_count += 1
    return gene_sets, annotation_count


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = count - reverse_rank + 1
        running = min(running, pvalues[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def odds_ratio(overlap: int, marker_count: int, gene_set_count: int, universe: int) -> float:
    a = overlap + 0.5
    b = marker_count - overlap + 0.5
    c = gene_set_count - overlap + 0.5
    d = universe - marker_count - gene_set_count + overlap + 0.5
    return (a * d) / (b * c)


def enrichment_rows(
    go_ids: Sequence[str],
    go_graph,
    gene_sets: Mapping[str, set[str]],
    marker_sets: Mapping[str, set[str]],
    label_metadata: Mapping[str, dict],
    background: set[str],
    experiment_id: str,
    organism: str,
    min_gene_set_size: int,
    max_qvalue: float,
    min_overlap: int,
    min_odds_ratio: float,
) -> list[dict]:
    rows: list[dict] = []
    universe = len(background)
    if universe == 0:
        raise ValueError("The enrichment background is empty")
    for cluster, markers in sorted(marker_sets.items()):
        if cluster not in label_metadata:
            continue
        marker_background = markers & background
        for go_id in go_ids:
            go_genes = gene_sets.get(go_id, set()) & background
            if len(go_genes) < min_gene_set_size:
                continue
            overlap_genes = marker_background & go_genes
            overlap = len(overlap_genes)
            pvalue = float(
                hypergeom.sf(overlap - 1, universe, len(go_genes), len(marker_background))
            )
            ratio = odds_ratio(overlap, len(marker_background), len(go_genes), universe)
            metadata = label_metadata[cluster]
            rows.append(
                {
                    "go_id": go_id,
                    "go_name": go_graph.nodes[go_id].get("name", ""),
                    "experiment_id": experiment_id,
                    "organism": organism,
                    "cell_label": cluster,
                    **metadata,
                    "background_size": universe,
                    "marker_count": len(marker_background),
                    "go_gene_count": len(go_genes),
                    "overlap_count": overlap,
                    "overlap_genes": "|".join(sorted(overlap_genes)),
                    "odds_ratio": ratio,
                    "pvalue": pvalue,
                }
            )
    qvalues = benjamini_hochberg([float(row["pvalue"]) for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["qvalue"] = qvalue
        row["is_significant"] = str(
            qvalue <= max_qvalue
            and int(row["overlap_count"]) >= min_overlap
            and float(row["odds_ratio"]) >= min_odds_ratio
        ).lower()
        row["evidence_weight"] = (
            min(50.0, -math.log10(max(qvalue, 1e-300)))
            * max(0.0, math.log2(max(float(row["odds_ratio"]), 1.0)))
        )
        for field in ("odds_ratio", "pvalue", "qvalue", "evidence_weight"):
            row[field] = f"{float(row[field]):.8g}"
        row["analysis_kind"] = "scea_marker_overrepresentation"
        row["source_url"] = (
            "https://www.ebi.ac.uk/gxa/sc/experiments/" + experiment_id
        )
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_json(args.config) if args.config else {}
    database_config = config.get("database", {})
    top_n = int(args.top_n or database_config.get("top_markers_per_cell_type", 200))
    max_marker_p = float(database_config.get("max_marker_adjusted_p", 0.05))
    min_logfc = float(database_config.get("min_marker_logfc", 0.25))
    min_gene_set_size = int(database_config.get("min_go_gene_set_size", 5))
    max_qvalue = float(database_config.get("max_enrichment_q", 0.05))
    min_overlap = int(database_config.get("min_overlap", 3))
    min_odds = float(database_config.get("min_odds_ratio", 1.5))

    go_ids = read_go_ids(args.go_terms)
    go_graph = load_obo(args.go_obo)
    cl_graph = load_obo(args.cl_obo)
    catalog = CellCatalog.from_tsv(args.catalog)
    gene_map = read_gene_map(args.gene_id_map, args.map_source_column, args.map_symbol_column)
    labels, skipped_labels = load_cell_labels(
        args.cell_metadata,
        args.label_column,
        args.ontology_column,
        cl_graph,
        catalog,
    )
    marker_sets, source_genes, mapped_count = load_marker_sets(
        args.marker_table,
        gene_map,
        args.cluster_column,
        args.gene_column,
        args.rank_column,
        args.logfc_column,
        args.marker_pvalue_column,
        top_n,
        max_marker_p,
        min_logfc,
    )
    gene_sets, annotation_count = build_go_gene_sets(args.gaf, set(go_ids), go_graph)
    background = read_background_genes(args.background_genes, gene_map)
    background_method = "expressed_gene_file"
    if not background:
        background = set().union(*marker_sets.values()) if marker_sets else set()
        background_method = "union_of_filtered_markers"
    rows = enrichment_rows(
        go_ids,
        go_graph,
        gene_sets,
        marker_sets,
        labels,
        background,
        args.experiment_id,
        args.organism,
        min_gene_set_size,
        max_qvalue,
        min_overlap,
        min_odds,
    )
    if not rows:
        raise ValueError(
            "No enrichment tests were produced. Check GO gene-set sizes, gene ID mapping, and CL metadata."
        )
    fieldnames = list(rows[0])
    write_tsv(args.output, fieldnames, rows)
    manifest = {
        "experiment_id": args.experiment_id,
        "go_terms_requested": len(go_ids),
        "go_terms_with_gene_sets": sum(bool(gene_sets.get(go_id)) for go_id in go_ids),
        "propagated_gaf_annotations": annotation_count,
        "marker_clusters": len(marker_sets),
        "metadata_cell_labels": len(labels),
        "skipped_cell_labels": skipped_labels,
        "source_marker_genes": len(source_genes),
        "mapped_marker_rows": mapped_count,
        "background_size": len(background),
        "background_method": background_method,
        "tests": len(rows),
        "significant_tests": sum(row["is_significant"] == "true" for row in rows),
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test GO gene sets against ontology-labelled SCEA marker genes."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--marker-table", required=True)
    parser.add_argument("--cell-metadata", required=True)
    parser.add_argument("--gaf", required=True)
    parser.add_argument("--go-terms", required=True)
    parser.add_argument("--go-obo", required=True)
    parser.add_argument("--cl-obo", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gene-id-map")
    parser.add_argument("--background-genes")
    parser.add_argument("--config")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--label-column", default="inferred_cell_type_-_ontology_labels")
    parser.add_argument(
        "--ontology-column", default="inferred_cell_type_-_ontology_labels_ontology"
    )
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--gene-column", default="genes")
    parser.add_argument("--rank-column", default="rank")
    parser.add_argument("--logfc-column", default="logfoldchanges")
    parser.add_argument("--marker-pvalue-column", default="pvals_adj")
    parser.add_argument("--map-source-column", default="source_gene_id")
    parser.add_argument("--map-symbol-column", default="gene_symbol")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
