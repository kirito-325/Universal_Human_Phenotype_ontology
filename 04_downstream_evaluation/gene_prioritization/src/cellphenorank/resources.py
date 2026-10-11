from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import as_bool, as_float, read_tsv, split_values, value
from .schemas import GeneSetResources, HpoCpoEdge


def _gene_contributions(text: str) -> tuple[tuple[str, float], ...]:
    output = []
    for item in text.split("|"):
        if not item.strip():
            continue
        gene, separator, raw = item.partition(":")
        if separator:
            output.append((gene.strip().upper(), as_float(raw, 0.0)))
    return tuple(output)


def load_hpo_cpo(
    path: str | Path,
    hpo_ids: Iterable[str],
    config: Mapping[str, Any],
) -> dict[str, list[HpoCpoEdge]]:
    wanted = set(hpo_ids)
    by_hpo: dict[str, list[HpoCpoEdge]] = defaultdict(list)
    confidence_column = str(config.get("confidence_column", "confidence"))
    min_confidence = float(config.get("min_confidence", 0.0))
    specific_weight = float(config.get("specific_scope_weight", 1.0))
    general_weight = float(config.get("general_scope_weight", 0.5))
    require_anatomy = bool(config.get("require_anatomy_allowed", True))
    for row in read_tsv(path):
        hpo_id = value(row, "hpo_id", "hp_id")
        if hpo_id not in wanted:
            continue
        if require_anatomy and value(row, "anatomy_status") not in {"", "allowed", "compatible"}:
            continue
        cell_id = value(row, "cell_id", "cl_id", "selected_cl_id", "cell_type_id")
        function_id = value(row, "function_id", "go_id", "selected_go_id", "cell_phenotype_id")
        if not cell_id or not function_id:
            continue
        scope = value(row, "cell_scope", "scope", "bridge_scope", default="unknown")
        scope_factor = specific_weight if scope in {"cell_type_specific", "specific", "direct_hpo"} else general_weight
        raw_confidence = next(
            (
                as_float(row.get(name, ""), -1.0)
                for name in (
                    confidence_column,
                    "confidence",
                    "geometric_consensus",
                    "three_way_evidence_percentile",
                    "rerank_score",
                )
                if str(row.get(name, "")).strip()
            ),
            0.0,
        )
        textual_confidence = value(row, "cl_go_confidence", "confidence_grade").lower()
        confidence_factor = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(textual_confidence, 1.0)
        confidence = max(0.0, min(1.0, raw_confidence)) * scope_factor * confidence_factor
        if confidence < min_confidence:
            continue
        source_genes = split_values(value(row, "source_gene_ids", "three_way_genes", "supporting_genes"))
        contributions = _gene_contributions(
            value(row, "source_gene_contributions", "three_way_gene_contributions")
        )
        by_hpo[hpo_id].append(
            HpoCpoEdge(
                hpo_id=hpo_id,
                hpo_name=value(row, "hpo_name", "hp_name"),
                cpo_id=value(row, "cpo_id", "cpo_key", default=f"{cell_id}__{function_id}"),
                cell_id=cell_id,
                cell_name=value(row, "cell_name", "cl_name"),
                function_id=function_id,
                function_name=value(row, "function_name", "go_name"),
                confidence=confidence,
                direction=value(row, "direction", default="unknown"),
                relation_type=value(row, "relation_type", default="associated_with"),
                scope=scope,
                source_genes=tuple(gene.upper() for gene in source_genes),
                source_gene_contributions=contributions,
                source_record_ids=value(row, "source_record_ids", "supporting_record_ids"),
            )
        )
    top_k = int(config.get("top_k_per_hpo", 25))
    output = {}
    for hpo_id, edges in by_hpo.items():
        unique: dict[str, HpoCpoEdge] = {}
        for edge in edges:
            current = unique.get(edge.cpo_id)
            if current is None or edge.confidence > current.confidence:
                unique[edge.cpo_id] = edge
        output[hpo_id] = sorted(
            unique.values(), key=lambda edge: (-edge.confidence, edge.cpo_id)
        )[:top_k]
    return output


def _normalize_by_group(raw: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    maxima: dict[str, float] = defaultdict(float)
    for (group, _), score in raw.items():
        maxima[group] = max(maxima[group], score)
    return {
        key: (score / maxima[key[0]] if maxima[key[0]] > 0 else 0.0)
        for key, score in raw.items()
    }


def _apply_idf(
    scores: Mapping[tuple[str, str], float],
    occurrence: Mapping[str, int],
    group_count: int,
    enabled: bool,
) -> dict[tuple[str, str], float]:
    if not enabled or group_count <= 1:
        return dict(scores)
    denominator = math.log(group_count + 1.0)
    output = {}
    for (group, gene), score in scores.items():
        idf = math.log((group_count + 1.0) / (occurrence.get(gene, 0) + 1.0)) / denominator
        output[(group, gene)] = max(0.0, min(1.0, score * idf))
    return output


def load_markers(
    path: str | Path,
    relevant_cells: set[str],
    candidate_genes: set[str],
    use_idf: bool = True,
) -> tuple[dict[tuple[str, str], float], dict[str, str], dict[str, int]]:
    raw: dict[tuple[str, str], float] = {}
    all_cells: set[str] = set()
    gene_cells: dict[str, set[str]] = defaultdict(set)
    cell_names: dict[str, str] = {}
    for row in read_tsv(path):
        cell_id = value(row, "cell_id", "catalog_id", "cl_id")
        gene = value(row, "gene_symbol", "gene", "gene_id").upper()
        if not cell_id or not gene:
            continue
        all_cells.add(cell_id)
        if gene in candidate_genes:
            gene_cells[gene].add(cell_id)
        if cell_id not in relevant_cells or gene not in candidate_genes:
            continue
        marker_weight = as_float(value(row, "marker_weight", "marker_score"), 1.0)
        specificity = as_float(value(row, "specificity", "significance_score"), 1.0)
        reproducibility = as_float(value(row, "reproducibility"), 1.0)
        evidence = math.log1p(max(0.0, marker_weight)) * max(0.0, specificity) * max(0.0, reproducibility)
        key = (cell_id, gene)
        raw[key] = max(raw.get(key, 0.0), evidence)
        cell_names[cell_id] = value(row, "cell_name", "catalog_cl_name", "catalog_label", default=cell_id)
    normalized = _normalize_by_group(raw)
    occurrence = {gene: len(cells) for gene, cells in gene_cells.items()}
    scores = _apply_idf(normalized, occurrence, len(all_cells), use_idf)
    sizes = Counter(cell for cell, _ in scores)
    return scores, cell_names, dict(sizes)


def load_functions(
    path: str | Path,
    relevant_functions: set[str],
    candidate_genes: set[str],
    use_idf: bool = True,
) -> tuple[dict[tuple[str, str], float], dict[str, str], dict[str, int]]:
    raw: dict[tuple[str, str], float] = {}
    all_functions: set[str] = set()
    gene_functions: dict[str, set[str]] = defaultdict(set)
    function_names: dict[str, str] = {}
    for row in read_tsv(path):
        function_id = value(row, "function_id", "go_id", "cell_phenotype_id")
        gene = value(row, "gene_symbol", "human_gene_symbol", "gene_id").upper()
        if not function_id or not gene:
            continue
        all_functions.add(function_id)
        if gene in candidate_genes:
            gene_functions[gene].add(function_id)
        if function_id not in relevant_functions or gene not in candidate_genes:
            continue
        weight = as_float(value(row, "function_weight", "annotation_count"), 1.0)
        if "annotation_count" in row:
            weight = math.log1p(max(0.0, weight))
        experimental = value(row, "has_experimental_evidence")
        if experimental and not as_bool(experimental):
            weight *= 0.7
        key = (function_id, gene)
        raw[key] = max(raw.get(key, 0.0), weight)
        function_names[function_id] = value(row, "function_name", "go_name", default=function_id)
    normalized = _normalize_by_group(raw)
    occurrence = {gene: len(functions) for gene, functions in gene_functions.items()}
    scores = _apply_idf(normalized, occurrence, len(all_functions), use_idf)
    sizes = Counter(function for function, _ in scores)
    return scores, function_names, dict(sizes)


def load_gene_set_resources(
    marker_path: str | Path,
    function_path: str | Path,
    hpo_cpo: Mapping[str, list[HpoCpoEdge]],
    candidate_genes: Iterable[str],
    config: Mapping[str, Any],
) -> GeneSetResources:
    cells = {edge.cell_id for edges in hpo_cpo.values() for edge in edges}
    functions = {edge.function_id for edges in hpo_cpo.values() for edge in edges}
    genes = {gene.upper() for gene in candidate_genes}
    use_idf = bool(config.get("use_idf", True))
    marker_scores, cell_names, marker_sizes = load_markers(
        marker_path, cells, genes, use_idf=use_idf
    )
    function_scores, function_names, function_sizes = load_functions(
        function_path, functions, genes, use_idf=use_idf
    )
    for edges in hpo_cpo.values():
        for edge in edges:
            cell_names.setdefault(edge.cell_id, edge.cell_name or edge.cell_id)
            function_names.setdefault(edge.function_id, edge.function_name or edge.function_id)
    return GeneSetResources(
        marker_scores=marker_scores,
        function_scores=function_scores,
        cell_names=cell_names,
        function_names=function_names,
        marker_set_sizes=marker_sizes,
        function_set_sizes=function_sizes,
    )
