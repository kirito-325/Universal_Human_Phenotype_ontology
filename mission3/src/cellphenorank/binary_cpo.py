from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .io import read_tsv, sha256, split_values, value


@dataclass(frozen=True)
class BinaryCpoEdge:
    hpo_id: str
    hpo_name: str
    cpo_id: str
    cell_id: str
    cell_name: str
    function_id: str
    function_name: str
    scope: str
    relation_type: str
    review_status: str
    confidence: str
    raw_confidence: str
    source_genes: tuple[str, ...]
    source_records: str


@dataclass
class BinaryCpoGraph:
    name: str
    version: str
    edges: tuple[BinaryCpoEdge, ...]
    function_genes: dict[str, set[str]]
    marker_memberships: set[tuple[str, str]]

    def __post_init__(self) -> None:
        by_hpo: dict[str, list[BinaryCpoEdge]] = defaultdict(list)
        for edge in self.edges:
            by_hpo[edge.hpo_id].append(edge)
        self.edges_by_hpo = dict(by_hpo)

    def bridge_loo(self, masked_genes: set[str]) -> "BinaryCpoGraph":
        masked = {gene.upper() for gene in masked_genes}
        retained = []
        for edge in self.edges:
            sources = {gene.upper() for gene in edge.source_genes}
            if sources and sources.intersection(masked) and not sources.difference(masked):
                continue
            retained.append(edge)
        suffix = ",".join(sorted(masked)) or "none"
        return BinaryCpoGraph(
            name=f"{self.name}_bridge_loo",
            version=f"{self.version}:bridge_loo:{suffix}",
            edges=tuple(retained),
            function_genes=self.function_genes,
            marker_memberships=self.marker_memberships,
        )


def load_hpoa_background(path: str | Path) -> set[str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"
        )
        return {
            row["hpo_id"]
            for row in reader
            if row.get("hpo_id")
            and row.get("qualifier", "").upper() != "NOT"
            and row.get("aspect", "P") == "P"
        }


def load_function_memberships(
    path: str | Path, candidate_genes: set[str]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    wanted = {gene.upper() for gene in candidate_genes}
    memberships: dict[str, set[str]] = defaultdict(set)
    names: dict[str, str] = {}
    for row in read_tsv(path):
        function_id = value(row, "function_id", "go_id", "cell_phenotype_id")
        gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
        if not function_id or gene not in wanted:
            continue
        memberships[function_id].add(gene)
        names[function_id] = value(
            row, "function_name", "go_name", default=function_id
        )
    return dict(memberships), names


def load_marker_memberships(
    path: str | Path, candidate_genes: set[str]
) -> tuple[set[tuple[str, str]], dict[str, str]]:
    wanted = {gene.upper() for gene in candidate_genes}
    memberships: set[tuple[str, str]] = set()
    names: dict[str, str] = {}
    for row in read_tsv(path):
        cell_id = value(row, "cell_id", "catalog_id", "cl_id")
        gene = value(row, "gene_symbol", "gene", "gene_id").upper()
        if not cell_id or gene not in wanted:
            continue
        memberships.add((cell_id, gene))
        names[cell_id] = value(
            row, "cell_name", "catalog_cl_name", "catalog_label", default=cell_id
        )
    return memberships, names


def build_binary_graphs(
    hpo_cpo_path: str | Path,
    function_path: str | Path,
    marker_path: str | Path,
    candidate_genes: set[str],
    valid_hpo_ids: set[str],
    background_hpo_ids: set[str],
    config: Mapping[str, object],
) -> tuple[BinaryCpoGraph, BinaryCpoGraph, list[dict[str, object]]]:
    function_genes, function_names = load_function_memberships(
        function_path, candidate_genes
    )
    marker_memberships, cell_names = load_marker_memberships(
        marker_path, candidate_genes
    )
    accepted_status = set(
        config.get("strict", {}).get(
            "accepted_cl_go_status", ["supported", "llm_supported", "confirmed"]
        )
    )
    strict_scopes = set(
        config.get("strict", {}).get("accepted_scopes", ["cell_type_specific"])
    )
    extended_scopes = set(
        config.get("extended", {}).get(
            "accepted_scopes", ["cell_type_specific", "cell_general"]
        )
    )
    allowed_anatomy = set(
        config.get("strict", {}).get(
            "accepted_anatomy_status", ["allowed", "compatible"]
        )
    )
    universe = valid_hpo_ids.intersection(background_hpo_ids)
    strict: dict[tuple[str, str], BinaryCpoEdge] = {}
    extended: dict[tuple[str, str], BinaryCpoEdge] = {}
    qc = Counter()
    for row in read_tsv(hpo_cpo_path):
        qc["input_rows"] += 1
        hpo_id = value(row, "hpo_id", "hp_id")
        cell_id = value(row, "cell_id", "cl_id", "selected_cl_id")
        function_id = value(row, "function_id", "go_id", "selected_go_id")
        if hpo_id not in universe or not cell_id or not function_id:
            qc["invalid_or_outside_background"] += 1
            continue
        if function_id not in function_genes:
            qc["edges_without_candidate_function_gene"] += 1
        anatomy = value(row, "anatomy_status", default="allowed")
        if anatomy not in allowed_anatomy:
            qc["anatomy_rejected"] += 1
            continue
        scope = value(row, "cell_scope", "scope", default="unknown")
        status = value(row, "cl_go_status", "review_status", default="unknown")
        cpo_id = value(row, "cpo_id", "cpo_key", default=f"{cell_id}__{function_id}")
        edge = BinaryCpoEdge(
            hpo_id=hpo_id,
            hpo_name=value(row, "hpo_name", "hp_name"),
            cpo_id=cpo_id,
            cell_id=cell_id,
            cell_name=value(row, "cell_name", "cl_name", default=cell_names.get(cell_id, cell_id)),
            function_id=function_id,
            function_name=value(
                row, "function_name", "go_name", default=function_names.get(function_id, function_id)
            ),
            scope=scope,
            relation_type=value(row, "relation_type", default="binary_supported"),
            review_status=status,
            confidence=value(row, "cl_go_confidence", "confidence_grade"),
            raw_confidence=value(
                row,
                "geometric_consensus",
                "confidence",
                "three_way_evidence_percentile",
            ),
            source_genes=tuple(
                gene.upper()
                for gene in split_values(
                    value(row, "source_gene_ids", "three_way_genes", "supporting_genes")
                )
            ),
            source_records=value(row, "source_record_ids", "supporting_record_ids"),
        )
        key = (hpo_id, cpo_id)
        if scope in extended_scopes:
            if key in extended:
                qc["extended_duplicate"] += 1
            extended.setdefault(key, edge)
        if scope in strict_scopes and status in accepted_status:
            if key in strict:
                qc["strict_duplicate"] += 1
            strict.setdefault(key, edge)
    graph_hash = sha256(hpo_cpo_path)[:16]
    function_hash = sha256(function_path)[:16]
    marker_hash = sha256(marker_path)[:16]
    version = f"hpo_cpo={graph_hash};function={function_hash};marker={marker_hash}"
    strict_graph = BinaryCpoGraph(
        name="strict",
        version=f"strict:{version}",
        edges=tuple(sorted(strict.values(), key=lambda edge: (edge.hpo_id, edge.cpo_id))),
        function_genes=function_genes,
        marker_memberships=marker_memberships,
    )
    extended_graph = BinaryCpoGraph(
        name="extended",
        version=f"extended:{version}",
        edges=tuple(sorted(extended.values(), key=lambda edge: (edge.hpo_id, edge.cpo_id))),
        function_genes=function_genes,
        marker_memberships=marker_memberships,
    )
    qc.update(
        {
            "strict_edges": len(strict_graph.edges),
            "extended_edges": len(extended_graph.edges),
            "function_sets_with_candidate_genes": len(function_genes),
            "marker_memberships_in_candidate_pool": len(marker_memberships),
            "background_hpo_count": len(universe),
        }
    )
    qc_rows = [
        {"metric": metric, "value": count} for metric, count in sorted(qc.items())
    ]
    return strict_graph, extended_graph, qc_rows


def graph_edge_rows(graph: BinaryCpoGraph) -> list[dict[str, object]]:
    return [
        {
            "graph": graph.name,
            "graph_version": graph.version,
            "hpo_id": edge.hpo_id,
            "hpo_name": edge.hpo_name,
            "cpo_id": edge.cpo_id,
            "cl_id": edge.cell_id,
            "cell_name": edge.cell_name,
            "go_id": edge.function_id,
            "function_name": edge.function_name,
            "scope": edge.scope,
            "relation_type": edge.relation_type,
            "review_status": edge.review_status,
            "confidence_audit_only": edge.confidence,
            "raw_confidence_audit_only": edge.raw_confidence,
            "source_genes": edge.source_genes,
            "source_records": edge.source_records,
        }
        for edge in graph.edges
    ]


GRAPH_EDGE_FIELDS = [
    "graph",
    "graph_version",
    "hpo_id",
    "hpo_name",
    "cpo_id",
    "cl_id",
    "cell_name",
    "go_id",
    "function_name",
    "scope",
    "relation_type",
    "review_status",
    "confidence_audit_only",
    "raw_confidence_audit_only",
    "source_genes",
    "source_records",
]
