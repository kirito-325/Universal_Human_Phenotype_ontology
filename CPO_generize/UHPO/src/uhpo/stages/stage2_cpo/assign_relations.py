from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .common import (
    CellCatalog,
    load_json,
    load_obo,
    stable_digest,
    write_tsv,
)
from .paths import (
    ASSIGNMENT_DIR,
    CATALOG,
    CATALOG_CONFIG,
    GO_FUNCTION_DIR,
    GO_OBO,
    SEMANTIC_DIR,
)
HUMAN_SUPPORT_RELATIONSHIPS = {
    "regulates",
    "positively_regulates",
    "negatively_regulates",
}

LINK_FIELDS = [
    "catalog_order",
    "cl_id",
    "cl_name",
    "catalog_label",
    "branch",
    "go_id",
    "go_name",
    "link_status",
    "confidence",
    "review_required",
    "assignment_method",
    "evidence_sources",
    "link_score",
    "decision_reason",
    "explicit_source_cl_ids",
    "name_matched_cl_name",
    "embedding_rank",
    "embedding_score",
    "embedding_margin",
    "database_experiment_count",
    "database_significant_test_count",
    "database_best_qvalue",
    "database_max_odds_ratio",
    "database_overlap_genes",
    "human_go_support_kind",
    "human_go_support_gene_count",
    "human_go_support_go_ids",
]

GO_STATUS_FIELDS = [
    "go_id",
    "go_name",
    "assignment_status",
    "direct_cell_count",
    "confirmed_cell_count",
    "supported_cell_count",
    "candidate_cell_count",
    "direct_cl_ids",
    "direct_cell_labels",
]

CELL_SUMMARY_FIELDS = [
    "catalog_order",
    "cl_id",
    "cl_name",
    "catalog_label",
    "branch",
    "direct_phenotype_count",
    "confirmed_phenotype_count",
    "supported_phenotype_count",
    "candidate_phenotype_count",
]


@dataclass
class DirectEvidence:
    sources: set[str] = field(default_factory=set)
    explicit_source_cl_ids: set[str] = field(default_factory=set)
    name_matches: set[str] = field(default_factory=set)
    embedding_rank: int = 0
    embedding_score: float = 0.0
    embedding_margin: float = 0.0
    database_experiments: set[str] = field(default_factory=set)
    database_significant_tests: int = 0
    database_best_qvalue: float = 1.0
    database_max_odds_ratio: float = 0.0
    database_overlap_genes: set[str] = field(default_factory=set)
    human_go_support_kind: str = ""
    human_go_support_gene_count: int = 0
    human_go_support_go_ids: set[str] = field(default_factory=set)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_columns(
    label: str, rows: Sequence[Mapping[str, str]], required: set[str]
) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def split_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def normalized_words(value: str) -> str:
    characters = [character.lower() if character.isalnum() else " " for character in value]
    return " ".join("".join(characters).split())


def phrase_present(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def compatible(catalog: CellCatalog, left: str, right: str) -> bool:
    return (
        left == right
        or left in catalog.ancestors(right)
        or right in catalog.ancestors(left)
    )


def find_name_matches(
    go_name: str,
    catalog: CellCatalog,
    candidate_scores: Mapping[str, float] | None = None,
) -> list[str]:
    """Return maximal CL names explicitly mentioned in the GO name."""

    normalized_go = normalized_words(go_name)
    phrases: dict[str, list[str]] = defaultdict(list)
    for node in catalog.nodes.values():
        if node.cl_id == "CL:0000000":
            continue
        phrase = normalized_words(node.cl_name)
        if phrase and phrase_present(normalized_go, phrase):
            phrases[phrase].append(node.cl_id)

    maximal_phrases = [
        phrase
        for phrase in phrases
        if not any(
            phrase != other and phrase_present(other, phrase) for other in phrases
        )
    ]
    scores = candidate_scores or {}
    result: list[str] = []
    for phrase in sorted(maximal_phrases, key=lambda value: (-len(value), value)):
        choices = phrases[phrase]
        selected = max(
            choices,
            key=lambda cl_id: (
                scores.get(cl_id, -1.0),
                catalog.nodes[cl_id].source == "skeleton",
                catalog.nodes[cl_id].level,
                cl_id,
            ),
        )
        if selected not in result:
            result.append(selected)
    return result


def index_candidates(
    rows: Sequence[Mapping[str, str]], catalog: CellCatalog
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("candidate_kind") != "catalog_node":
            continue
        cl_id = row.get("cl_id", "")
        if cl_id not in catalog.nodes or cl_id == "CL:0000000":
            continue
        try:
            rank = int(row.get("rank", ""))
            score = float(row.get("score", ""))
        except ValueError:
            continue
        result[row.get("go_id", "")].append(
            {"cl_id": cl_id, "rank": rank, "score": score}
        )
    for values in result.values():
        values.sort(key=lambda row: (int(row["rank"]), -float(row["score"])))
    return dict(result)


def embedding_margin(
    candidates: Sequence[Mapping[str, object]], catalog: CellCatalog
) -> float:
    if not candidates:
        return 0.0
    top_id = str(candidates[0]["cl_id"])
    top_score = float(candidates[0]["score"])
    if len(candidates) == 1:
        return top_score
    competitor_score = float(candidates[1]["score"])
    for candidate in candidates[1:]:
        if not compatible(catalog, top_id, str(candidate["cl_id"])):
            competitor_score = float(candidate["score"])
            break
    return top_score - competitor_score


def add_embedding(
    evidence: DirectEvidence,
    candidate: Mapping[str, object],
    margin: float,
) -> None:
    evidence.sources.add("sapbert_catalog")
    evidence.embedding_rank = int(candidate["rank"])
    evidence.embedding_score = float(candidate["score"])
    evidence.embedding_margin = margin


def add_database_evidence(
    proposals: dict[tuple[str, str], DirectEvidence],
    rows: Iterable[Mapping[str, str]],
    catalog: CellCatalog,
) -> None:
    seen_tests: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row.get("is_significant", "").lower() != "true":
            continue
        go_id = row.get("go_id", "").strip()
        cl_id = (row.get("catalog_cl_id") or row.get("catalog_anchor_id") or "").strip()
        if not go_id or cl_id not in catalog.nodes or cl_id == "CL:0000000":
            continue
        evidence = proposals.setdefault((go_id, cl_id), DirectEvidence())
        evidence.sources.add("scea_enrichment")
        experiment_id = row.get("experiment_id", "").strip()
        if experiment_id:
            evidence.database_experiments.add(experiment_id)
        test_key = (
            go_id,
            cl_id,
            experiment_id,
            row.get("cell_label", "").strip(),
        )
        if test_key not in seen_tests:
            seen_tests.add(test_key)
            evidence.database_significant_tests += 1
        try:
            evidence.database_best_qvalue = min(
                evidence.database_best_qvalue, float(row.get("qvalue", "1") or 1)
            )
        except ValueError:
            pass
        try:
            evidence.database_max_odds_ratio = max(
                evidence.database_max_odds_ratio,
                float(row.get("odds_ratio", "0") or 0),
            )
        except ValueError:
            pass
        evidence.database_overlap_genes.update(split_ids(row.get("overlap_genes", "")))


def build_human_go_support(
    rows: Sequence[Mapping[str, str]],
    go_graph=None,
    relationship_depth: int = 2,
) -> dict[str, dict[str, object]]:
    if not rows:
        return {}
    require_columns(
        "human GO summary", rows, {"go_id", "human_function_gene_count"}
    )
    direct_counts: dict[str, int] = {}
    for row in rows:
        try:
            direct_counts[row["go_id"]] = int(row["human_function_gene_count"] or 0)
        except ValueError:
            direct_counts[row["go_id"]] = 0

    result: dict[str, dict[str, object]] = {}
    for go_id, direct_count in direct_counts.items():
        if direct_count > 0:
            result[go_id] = {
                "kind": "direct",
                "gene_count": direct_count,
                "support_go_ids": {go_id},
            }
            continue
        if go_graph is None or go_id not in go_graph or relationship_depth < 1:
            continue
        queue = [(go_id, 0)]
        visited = {go_id}
        supporting_ids: set[str] = set()
        support_count = 0
        while queue:
            current, depth = queue.pop(0)
            if depth >= relationship_depth:
                continue
            for _, target, relation in go_graph.out_edges(current, keys=True):
                if relation not in HUMAN_SUPPORT_RELATIONSHIPS or target in visited:
                    continue
                visited.add(target)
                target_count = direct_counts.get(target, 0)
                if target_count > 0:
                    supporting_ids.add(target)
                    support_count = max(support_count, target_count)
                queue.append((target, depth + 1))
        if supporting_ids:
            result[go_id] = {
                "kind": "related_process",
                "gene_count": support_count,
                "support_go_ids": supporting_ids,
            }
    return result


def add_human_go_support(
    proposals: Mapping[tuple[str, str], DirectEvidence],
    support: Mapping[str, Mapping[str, object]],
) -> None:
    for (go_id, _), evidence in proposals.items():
        item = support.get(go_id)
        if not item:
            continue
        kind = str(item["kind"])
        evidence.human_go_support_kind = kind
        evidence.human_go_support_gene_count = int(item["gene_count"])
        evidence.human_go_support_go_ids.update(item["support_go_ids"])
        evidence.sources.add(
            "human_goa_direct" if kind == "direct" else "human_goa_related_process"
        )


def classify_link(
    evidence: DirectEvidence, database_min_experiments: int
) -> tuple[str, str, bool, str, float, str]:
    sources = evidence.sources
    cell_evidence_sources = sources - {
        "human_goa_direct",
        "human_goa_related_process",
    }
    database_repeated = (
        len(evidence.database_experiments) >= database_min_experiments
    )
    human_applicable = bool(
        evidence.human_go_support_gene_count
        or evidence.database_experiments
        or "go_curator_cl_reference" in sources
    )
    if "go_curator_cl_reference" in sources:
        return (
            "confirmed",
            "high",
            False,
            "curator_reference",
            1.0,
            "GO ontology explicitly references this CL term.",
        )
    if "go_name_cell_mention" in sources:
        if human_applicable:
            score = min(
                0.98,
                0.86
                + (0.04 if "sapbert_catalog" in sources else 0.0)
                + (0.04 if evidence.human_go_support_gene_count else 0.0),
            )
            return (
                "supported",
                "high" if "sapbert_catalog" in sources else "medium",
                False,
                "go_name_cell_mention",
                score,
                "The GO name explicitly names this cell type and human GO evidence "
                "supports the process.",
            )
        return (
            "candidate",
            "medium",
            True,
            "go_name_cell_mention",
            0.68 if "sapbert_catalog" in sources else 0.60,
            "The GO name names this cell, but human applicability has not been established.",
        )
    if database_repeated:
        score = min(0.95, 0.80 + 0.03 * len(evidence.database_experiments))
        return (
            "supported",
            "high" if len(evidence.database_experiments) >= 3 else "medium",
            False,
            "repeated_scea_enrichment",
            score,
            "Significant GO enrichment recurs for this exact catalog cell across experiments.",
        )
    if len(cell_evidence_sources) >= 2:
        return (
            "supported",
            "medium",
            False,
            "multi_source_candidate",
            min(0.90, max(0.70, evidence.embedding_score + 0.08)),
            "Independent direct-assignment evidence sources agree on this cell.",
        )
    if "sapbert_catalog" in sources:
        return (
            "candidate",
            "medium",
            True,
            "sapbert_catalog",
            evidence.embedding_score,
            "SapBERT directly retrieves this catalog cell; confirmation is still required.",
        )
    return (
        "candidate",
        "low",
        True,
        "single_scea_enrichment",
        0.55,
        "One experiment supports this direct cell association; replication is required.",
    )


def assign_direct_links(
    semantic_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    database_rows: Sequence[Mapping[str, str]],
    catalog: CellCatalog,
    config: Mapping[str, object] | None = None,
    human_go_rows: Sequence[Mapping[str, str]] = (),
    go_graph=None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    require_columns(
        "semantic summary",
        semantic_rows,
        {"go_id", "go_name", "explicit_cl_ids", "explicit_catalog_cl_ids"},
    )
    require_columns(
        "semantic candidates",
        candidate_rows,
        {"go_id", "candidate_kind", "rank", "cl_id", "score"},
    )
    direct_config = dict((config or {}).get("direct_assignment", {}))
    semantic_config = dict((config or {}).get("semantic", {}))
    min_score = float(
        direct_config.get(
            "embedding_min_score", semantic_config.get("specific_min_score", 0.62)
        )
    )
    min_margin = float(
        direct_config.get(
            "embedding_min_margin", semantic_config.get("specific_min_margin", 0.04)
        )
    )
    database_min_experiments = int(
        direct_config.get("database_min_experiments", 2)
    )
    human_support_relationship_depth = int(
        direct_config.get("human_support_relationship_depth", 2)
    )
    indexed_candidates = index_candidates(candidate_rows, catalog)
    proposals: dict[tuple[str, str], DirectEvidence] = {}
    go_names = {row["go_id"]: row["go_name"] for row in semantic_rows}

    for semantic in semantic_rows:
        go_id = semantic["go_id"]
        go_name = semantic["go_name"]
        candidates = indexed_candidates.get(go_id, [])
        by_cl = {str(candidate["cl_id"]): candidate for candidate in candidates}
        score_by_cl = {
            str(candidate["cl_id"]): float(candidate["score"])
            for candidate in candidates
        }
        top_margin = embedding_margin(candidates, catalog)

        explicit_targets = [
            cl_id
            for cl_id in split_ids(semantic.get("explicit_catalog_cl_ids", ""))
            if cl_id in catalog.nodes and cl_id != "CL:0000000"
        ]
        explicit_sources = split_ids(semantic.get("explicit_cl_ids", ""))
        for cl_id in explicit_targets:
            evidence = proposals.setdefault((go_id, cl_id), DirectEvidence())
            evidence.sources.add("go_curator_cl_reference")
            evidence.explicit_source_cl_ids.update(explicit_sources)

        name_targets = find_name_matches(go_name, catalog, score_by_cl)
        if explicit_targets:
            name_targets = [cl_id for cl_id in name_targets if cl_id in explicit_targets]
        for cl_id in name_targets:
            evidence = proposals.setdefault((go_id, cl_id), DirectEvidence())
            evidence.sources.add("go_name_cell_mention")
            evidence.name_matches.add(catalog.nodes[cl_id].cl_name)

        direct_targets = set(explicit_targets) | set(name_targets)
        if direct_targets:
            for cl_id in direct_targets:
                candidate = by_cl.get(cl_id)
                if candidate:
                    add_embedding(evidence=proposals[(go_id, cl_id)], candidate=candidate, margin=top_margin)
        elif candidates:
            top = candidates[0]
            if float(top["score"]) >= min_score and top_margin >= min_margin:
                key = (go_id, str(top["cl_id"]))
                add_embedding(
                    evidence=proposals.setdefault(key, DirectEvidence()),
                    candidate=top,
                    margin=top_margin,
                )

    add_database_evidence(proposals, database_rows, catalog)
    human_support = build_human_go_support(
        human_go_rows, go_graph, human_support_relationship_depth
    )
    add_human_go_support(proposals, human_support)

    catalog_order = {cl_id: index for index, cl_id in enumerate(catalog.nodes, 1)}
    links: list[dict[str, object]] = []
    for (go_id, cl_id), evidence in proposals.items():
        if go_id not in go_names:
            continue
        status, confidence, review, method, link_score, reason = classify_link(
            evidence, database_min_experiments
        )
        node = catalog.nodes[cl_id]
        links.append(
            {
                "catalog_order": catalog_order[cl_id],
                "cl_id": cl_id,
                "cl_name": node.cl_name,
                "catalog_label": node.catalog_label,
                "branch": node.branch,
                "go_id": go_id,
                "go_name": go_names[go_id],
                "link_status": status,
                "confidence": confidence,
                "review_required": str(review).lower(),
                "assignment_method": method,
                "evidence_sources": "|".join(sorted(evidence.sources)),
                "link_score": f"{link_score:.6f}",
                "decision_reason": reason,
                "explicit_source_cl_ids": "|".join(
                    sorted(evidence.explicit_source_cl_ids)
                ),
                "name_matched_cl_name": "|".join(sorted(evidence.name_matches)),
                "embedding_rank": evidence.embedding_rank or "",
                "embedding_score": (
                    f"{evidence.embedding_score:.6f}"
                    if evidence.embedding_rank
                    else ""
                ),
                "embedding_margin": (
                    f"{evidence.embedding_margin:.6f}"
                    if evidence.embedding_rank
                    else ""
                ),
                "database_experiment_count": len(evidence.database_experiments),
                "database_significant_test_count": evidence.database_significant_tests,
                "database_best_qvalue": (
                    f"{evidence.database_best_qvalue:.8g}"
                    if evidence.database_significant_tests
                    else ""
                ),
                "database_max_odds_ratio": (
                    f"{evidence.database_max_odds_ratio:.8g}"
                    if evidence.database_significant_tests
                    else ""
                ),
                "database_overlap_genes": "|".join(
                    sorted(evidence.database_overlap_genes)
                ),
                "human_go_support_kind": evidence.human_go_support_kind,
                "human_go_support_gene_count": evidence.human_go_support_gene_count,
                "human_go_support_go_ids": "|".join(
                    sorted(evidence.human_go_support_go_ids)
                ),
            }
        )
    links.sort(key=lambda row: (int(row["catalog_order"]), str(row["go_id"])))

    links_by_go: dict[str, list[dict[str, object]]] = defaultdict(list)
    links_by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in links:
        links_by_go[str(row["go_id"])].append(row)
        links_by_cell[str(row["cl_id"])].append(row)

    statuses: list[dict[str, object]] = []
    for semantic in semantic_rows:
        go_links = links_by_go.get(semantic["go_id"], [])
        status_counts = Counter(str(row["link_status"]) for row in go_links)
        if status_counts["confirmed"] or status_counts["supported"]:
            assignment_status = "linked"
        elif status_counts["candidate"]:
            assignment_status = "candidate_only"
        else:
            assignment_status = "unassigned"
        statuses.append(
            {
                "go_id": semantic["go_id"],
                "go_name": semantic["go_name"],
                "assignment_status": assignment_status,
                "direct_cell_count": len(go_links),
                "confirmed_cell_count": status_counts["confirmed"],
                "supported_cell_count": status_counts["supported"],
                "candidate_cell_count": status_counts["candidate"],
                "direct_cl_ids": "|".join(str(row["cl_id"]) for row in go_links),
                "direct_cell_labels": "|".join(
                    str(row["catalog_label"]) for row in go_links
                ),
            }
        )

    cell_summaries: list[dict[str, object]] = []
    for cl_id, node in catalog.nodes.items():
        cell_links = links_by_cell.get(cl_id, [])
        status_counts = Counter(str(row["link_status"]) for row in cell_links)
        cell_summaries.append(
            {
                "catalog_order": catalog_order[cl_id],
                "cl_id": cl_id,
                "cl_name": node.cl_name,
                "catalog_label": node.catalog_label,
                "branch": node.branch,
                "direct_phenotype_count": sum(
                    row["link_status"] in {"confirmed", "supported"}
                    for row in cell_links
                ),
                "confirmed_phenotype_count": status_counts["confirmed"],
                "supported_phenotype_count": status_counts["supported"],
                "candidate_phenotype_count": status_counts["candidate"],
            }
        )

    counts = Counter(str(row["link_status"]) for row in links)
    accepted_links = [
        row for row in links if row["link_status"] in {"confirmed", "supported"}
    ]
    candidate_links = [row for row in links if row["link_status"] == "candidate"]
    counts.update(
        {
            "go_terms": len(semantic_rows),
            "all_assignments": len(links),
            "direct_links": len(accepted_links),
            "candidate_links": len(candidate_links),
            "cells_with_direct_phenotypes": len(
                {str(row["cl_id"]) for row in accepted_links}
            ),
            "cells_with_candidates": len(
                {str(row["cl_id"]) for row in candidate_links}
            ),
            "linked_go_terms": sum(
                row["assignment_status"] == "linked" for row in statuses
            ),
            "candidate_only_go_terms": sum(
                row["assignment_status"] == "candidate_only" for row in statuses
            ),
            "unassigned_go_terms": sum(
                row["assignment_status"] == "unassigned" for row in statuses
            ),
        }
    )
    return links, statuses, cell_summaries, dict(counts)


def run(args: argparse.Namespace) -> None:
    semantic_rows = read_tsv(args.semantic_summary)
    candidate_rows = read_tsv(args.semantic_candidates)
    database_rows: list[dict[str, str]] = []
    for path in args.database_evidence or []:
        database_rows.extend(read_tsv(path))
    human_go_rows = read_tsv(args.human_go_summary) if args.human_go_summary else []
    go_graph = load_obo(args.go_obo) if args.go_obo else None
    catalog = CellCatalog.from_tsv(args.catalog)
    config = load_json(args.config) if args.config else {}
    links, statuses, cell_summaries, counts = assign_direct_links(
        semantic_rows,
        candidate_rows,
        database_rows,
        catalog,
        config,
        human_go_rows,
        go_graph,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    links_path = output_dir / "cell_go_phenotype_links.tsv"
    candidates_path = output_dir / "cell_go_phenotype_candidates.tsv"
    status_path = output_dir / "go_assignment_status.tsv"
    summary_path = output_dir / "cell_phenotype_summary.tsv"
    accepted_links = [
        row for row in links if row["link_status"] in {"confirmed", "supported"}
    ]
    candidate_links = [row for row in links if row["link_status"] == "candidate"]
    write_tsv(links_path, LINK_FIELDS, accepted_links)
    write_tsv(candidates_path, LINK_FIELDS, candidate_links)
    write_tsv(status_path, GO_STATUS_FIELDS, statuses)
    write_tsv(summary_path, CELL_SUMMARY_FIELDS, cell_summaries)
    manifest = {
        "design": {
            "definition": (
                "Direct CL-to-GO biological-process phenotype assignments from curator "
                "references, explicit cell mentions, catalog-node embeddings, and optional "
                "SCEA enrichment. No CL closure or marker-GOA overlap is used."
            ),
            "relation": "cell_has_go_biological_process_phenotype",
            "constraint_closure_generated": False,
            "marker_overlap_required": False,
        },
        "inputs": {
            "semantic_summary": str(Path(args.semantic_summary).resolve()),
            "semantic_candidates": str(Path(args.semantic_candidates).resolve()),
            "database_evidence": [
                str(Path(path).resolve()) for path in args.database_evidence or []
            ],
            "human_go_summary": str(Path(args.human_go_summary).resolve())
            if args.human_go_summary
            else "",
            "go_obo": str(Path(args.go_obo).resolve()) if args.go_obo else "",
            "catalog": str(Path(args.catalog).resolve()),
            "config": str(Path(args.config).resolve()) if args.config else "",
        },
        "outputs": {
            "links": str(links_path.resolve()),
            "candidates": str(candidates_path.resolve()),
            "go_status": str(status_path.resolve()),
            "cell_summary": str(summary_path.resolve()),
        },
        "counts": counts,
        "link_digest": stable_digest(
            f"{row['cl_id']}\t{row['go_id']}\t{row['link_status']}"
            for row in accepted_links
        ),
        "candidate_digest": stable_digest(
            f"{row['cl_id']}\t{row['go_id']}" for row in candidate_links
        ),
    }
    manifest_path = output_dir / "direct_assignment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign GO biological-process phenotypes directly to catalog cells."
    )
    parser.add_argument(
        "--semantic-summary",
        type=Path,
        default=SEMANTIC_DIR / "semantic_summary.tsv",
    )
    parser.add_argument(
        "--semantic-candidates",
        type=Path,
        default=SEMANTIC_DIR / "semantic_candidates.tsv",
    )
    parser.add_argument("--database-evidence", type=Path, action="append")
    parser.add_argument(
        "--human-go-summary",
        type=Path,
        default=GO_FUNCTION_DIR / "go_function_summary.tsv",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=GO_OBO,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CATALOG_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSIGNMENT_DIR,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
