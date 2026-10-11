from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from cpo_mapping.io import read_tsv, write_json, write_tsv

from .config import output_root, resource_path


K_VALUES = (1, 5, 10, 20)


def _rank_bounds(scores: np.ndarray, target_indices: Iterable[int]) -> list[tuple[int, int]]:
    finite_positive = scores[np.isfinite(scores) & (scores > 0)]
    result = []
    for index in target_indices:
        value = float(scores[index])
        if not np.isfinite(value) or value <= 0:
            continue
        result.append(
            (
                1 + int(np.count_nonzero(finite_positive > value)),
                int(np.count_nonzero(finite_positive >= value)),
            )
        )
    return result


def _metrics(cases: list[dict[str, object]], prefix: dict[str, str]) -> list[dict[str, object]]:
    query_count = len(cases)
    output = []
    for bound_name, field in (("optimistic", "best_rank_min"), ("conservative", "best_rank_max")):
        ranks = [int(row[field]) for row in cases if int(row[field]) > 0]
        output.append(
            {
                **prefix,
                "tie_policy": bound_name,
                "metric": "MRR",
                "k": "",
                "value": (
                    sum(1.0 / rank for rank in ranks) / query_count
                    if query_count
                    else float("nan")
                ),
                "denominator": query_count,
            }
        )
        for k in K_VALUES:
            output.append(
                {
                    **prefix,
                    "tie_policy": bound_name,
                    "metric": "Hit",
                    "k": k,
                    "value": (
                        sum(int(row[field]) > 0 and int(row[field]) <= k for row in cases)
                        / query_count
                        if query_count
                        else float("nan")
                    ),
                    "denominator": query_count,
                }
            )
            output.append(
                {
                    **prefix,
                    "tie_policy": bound_name,
                    "metric": "Recall",
                    "k": k,
                    "value": (
                        sum(
                            sum(rank <= k for rank in row[f"all_{field}"])
                            / max(1, int(row["target_count"]))
                            for row in cases
                        )
                        / query_count
                        if query_count
                        else float("nan")
                    ),
                    "denominator": query_count,
                }
            )
    return output


def evaluate_term_mapping(config: dict[str, Any]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = output_root(config)
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    hpo_ids = [row["term_id"] for row in hpo_rows]
    mpo_ids = [row["term_id"] for row in mpo_rows]
    h_index = {value: index for index, value in enumerate(hpo_ids)}
    m_index = {value: index for index, value in enumerate(mpo_ids)}
    _, reference = read_tsv(resource_path(config, "hpo_mpo_reference"))
    h_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    m_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in reference:
        if row["hp_id"] in h_index and row["mp_id"] in m_index:
            relation = row["match_type"]
            h_targets[(row["hp_id"], relation)].add(m_index[row["mp_id"]])
            m_targets[(row["mp_id"], relation)].add(h_index[row["hp_id"]])
    datasets = {
        "final": "final/shape",
        "self_only": "self_only/shape",
        "inherited_only": "inherited_only/shape",
        "cl_aux": "aux/cl",
        "go_aux": "aux/go",
    }
    cases: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for view, dataset_path in datasets.items():
            dataset = handle[dataset_path]
            for direction, groups, query_index, axis, block_size in (
                ("HPO_to_MPO", h_targets, h_index, "row", 64),
                ("MPO_to_HPO", m_targets, m_index, "column", 512),
            ):
                relation_cases: dict[str, list[dict[str, object]]] = defaultdict(list)
                by_index: dict[int, list[tuple[str, str, set[int]]]] = defaultdict(list)
                for (query_id, relation), targets in groups.items():
                    by_index[query_index[query_id]].append((query_id, relation, targets))
                total = dataset.shape[0] if axis == "row" else dataset.shape[1]
                for start in range(0, total, block_size):
                    end = min(total, start + block_size)
                    relevant = [index for index in by_index if start <= index < end]
                    if not relevant:
                        continue
                    score_block = (
                        dataset[start:end, :]
                        if axis == "row"
                        else dataset[:, start:end]
                    )
                    for index in relevant:
                        scores = (
                            score_block[index - start, :]
                            if axis == "row"
                            else score_block[:, index - start]
                        )
                        for query_id, relation, targets in by_index[index]:
                            bounds = _rank_bounds(scores, targets)
                            rank_min = min((value[0] for value in bounds), default=0)
                            rank_max = min((value[1] for value in bounds), default=0)
                            case = {
                                "task": "term_mapping",
                                "view": view,
                                "direction": direction,
                                "relation_type": relation,
                                "split": "all",
                                "query_id": query_id,
                                "target_count": len(targets),
                                "targets_computable": sum(
                                    bool(np.isfinite(scores[target])) for target in targets
                                ),
                                "best_rank_min": rank_min,
                                "best_rank_max": rank_max,
                                "all_best_rank_min": "|".join(str(value[0]) for value in bounds),
                                "all_best_rank_max": "|".join(str(value[1]) for value in bounds),
                                "all_best_rank_min_values": [value[0] for value in bounds],
                                "all_best_rank_max_values": [value[1] for value in bounds],
                                "scoreable": str(bool(np.isfinite(scores).any())).lower(),
                            }
                            cases.append(case)
                            relation_cases[relation].append(case)
                for relation, values in relation_cases.items():
                    adapted = [
                        {
                            **row,
                            "all_best_rank_min": row["all_best_rank_min_values"],
                            "all_best_rank_max": row["all_best_rank_max_values"],
                        }
                        for row in values
                    ]
                    metrics.extend(
                        _metrics(
                            adapted,
                            {
                                "task": "term_mapping",
                                "view": view,
                                "direction": direction,
                                "relation_type": relation,
                                "split": "all",
                                "evaluation_scope": "end_to_end",
                            },
                        )
                    )
                    conditional = [
                        row
                        for row in adapted
                        if row["scoreable"] == "true" and int(row["targets_computable"]) > 0
                    ]
                    metrics.extend(
                        _metrics(
                            conditional,
                            {
                                "task": "term_mapping",
                                "view": view,
                                "direction": direction,
                                "relation_type": relation,
                                "split": "all",
                                "evaluation_scope": "computable_query_and_reference",
                            },
                        )
                    )
    for row in cases:
        row.pop("all_best_rank_min_values")
        row.pop("all_best_rank_max_values")
    return cases, metrics


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _split_gold(gold: list[dict[str, str]], fraction: float) -> dict[str, str]:
    by_gene: dict[str, list[str]] = defaultdict(list)
    for row in gold:
        by_gene[row["ncbi_gene_id"]].append(row["disease_id"])
    union = UnionFind()
    for diseases in by_gene.values():
        for disease in diseases[1:]:
            union.union(diseases[0], disease)
    diseases = {row["disease_id"] for row in gold}
    components: dict[str, list[str]] = defaultdict(list)
    for disease in diseases:
        components[union.find(disease)].append(disease)
    result = {}
    threshold = int(fraction * 10000)
    for members in components.values():
        component_id = min(members)
        bucket = int(hashlib.sha256(component_id.encode()).hexdigest()[:8], 16) % 10000
        split = "development" if bucket < threshold else "test"
        for disease in members:
            result[disease] = split
    return result


def _hgnc_entrez(config: dict[str, Any]) -> dict[str, str]:
    _, rows = read_tsv(resource_path(config, "mouse_human_orthology"))
    return {
        row["HGNC ID"]: row["EntrezGene ID"]
        for row in rows
        if row["NCBI Taxon ID"] == "9606" and row["HGNC ID"] and row["EntrezGene ID"]
    }


def evaluate_gene(config: dict[str, Any]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    root = output_root(config)
    _, diseases = read_tsv(root / "disease_model/disease_index.tsv")
    disease_index = {row["disease_id"]: int(row["index"]) for row in diseases}
    _, gold = read_tsv(resource_path(config, "disease_gene_gold"))
    gold = [row for row in gold if row["disease_id"] in disease_index]
    split = _split_gold(gold, float(config["evaluation"]["development_fraction"]))
    split_rows = [
        {"disease_id": disease, "split": value}
        for disease, value in sorted(split.items())
    ]
    targets: dict[str, set[str]] = defaultdict(set)
    for row in gold:
        targets[row["disease_id"]].add(row["ncbi_gene_id"])
    hgnc_entrez = _hgnc_entrez(config)
    cases: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    with h5py.File(root / "gene_prediction/human_gene_scores.h5", "r") as handle:
        for view, dataset_name, index_file in (
            ("one_to_one", "one_to_one", "human_1to1_index.tsv"),
            ("expanded", "expanded", "human_expanded_index.tsv"),
        ):
            _, gene_rows = read_tsv(root / f"gene_prediction/{index_file}")
            gene_entrez = [hgnc_entrez.get(row["human_gene_id"], "") for row in gene_rows]
            positions: dict[str, set[int]] = defaultdict(set)
            for index, entrez in enumerate(gene_entrez):
                if entrez:
                    positions[entrez].add(index)
            dataset = handle[dataset_name]
            split_cases: dict[str, list[dict[str, object]]] = defaultdict(list)
            disease_ids_by_index = [row["disease_id"] for row in diseases]
            for start in range(0, len(diseases), 32):
                end = min(len(diseases), start + 32)
                score_block = dataset[start:end, :]
                for offset, disease_id in enumerate(disease_ids_by_index[start:end]):
                    if disease_id not in targets:
                        continue
                    target_genes = targets[disease_id]
                    target_indices = {
                        index
                        for gene in target_genes
                        for index in positions.get(gene, set())
                    }
                    scores = score_block[offset]
                    bounds = _rank_bounds(scores, target_indices)
                    case = {
                        "task": "disease_gene",
                        "view": view,
                        "direction": "disease_to_human_gene",
                        "relation_type": "known_disease_gene",
                        "split": split.get(disease_id, "unassigned"),
                        "query_id": disease_id,
                        "target_count": len(target_genes),
                        "targets_in_universe": len(target_indices),
                        "best_rank_min": min((value[0] for value in bounds), default=0),
                        "best_rank_max": min((value[1] for value in bounds), default=0),
                        "all_best_rank_min": [value[0] for value in bounds],
                        "all_best_rank_max": [value[1] for value in bounds],
                        "scoreable": str(bool(np.isfinite(scores).any())).lower(),
                    }
                    cases.append(case)
                    split_cases[case["split"]].append(case)
                    split_cases["all"].append(case)
            for split_name, values in split_cases.items():
                metrics.extend(
                    _metrics(
                        values,
                        {
                            "task": "disease_gene",
                            "view": view,
                            "direction": "disease_to_human_gene",
                            "relation_type": "known_disease_gene",
                            "split": split_name,
                            "evaluation_scope": "end_to_end",
                        },
                    )
                )
                conditional = [
                    row
                    for row in values
                    if row["scoreable"] == "true" and int(row["targets_in_universe"]) > 0
                ]
                metrics.extend(
                    _metrics(
                        conditional,
                        {
                            "task": "disease_gene",
                            "view": view,
                            "direction": "disease_to_human_gene",
                            "relation_type": "known_disease_gene",
                            "split": split_name,
                            "evaluation_scope": "target_in_candidate_universe",
                        },
                    )
                )
    for row in cases:
        row["all_best_rank_min"] = "|".join(map(str, row["all_best_rank_min"]))
        row["all_best_rank_max"] = "|".join(map(str, row["all_best_rank_max"]))
    return cases, metrics, split_rows


def run_evaluate(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    evaluation_root = root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    term_cases, term_metrics = evaluate_term_mapping(config)
    gene_cases, gene_metrics, split_rows = evaluate_gene(config)
    case_fields = tuple(term_cases[0])
    write_tsv(evaluation_root / "term_mapping_cases.tsv", case_fields, term_cases)
    write_tsv(evaluation_root / "disease_gene_cases.tsv", tuple(gene_cases[0]), gene_cases)
    write_tsv(evaluation_root / "disease_split.tsv", ("disease_id", "split"), split_rows)
    all_metrics = [*term_metrics, *gene_metrics]
    write_tsv(evaluation_root / "ranking_metrics.tsv", tuple(all_metrics[0]), all_metrics)
    write_tsv(
        evaluation_root / "source_filtering_status.tsv",
        ("status", "reason", "evaluation_type"),
        [
            {
                "status": "NOT_AVAILABLE",
                "reason": "HPO CPO export records gene_model/supplied_disease bridge scopes but does not expose a complete disease/gene derivation key for deterministic label-derived removal.",
                "evaluation_type": "retrospective_retrieval",
            }
        ],
    )
    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "term_mapping_case_count": len(term_cases),
        "disease_gene_case_count": len(gene_cases),
        "split_counts": dict(
            sorted(
                (key, sum(row["split"] == key for row in split_rows))
                for key in {row["split"] for row in split_rows}
            )
        ),
        "metric_row_count": len(all_metrics),
        "evaluation_type": "retrospective_retrieval",
        "source_filtered_evaluation": "NOT_AVAILABLE",
    }
    write_json(evaluation_root / "evaluation_manifest.json", result)
    return result
