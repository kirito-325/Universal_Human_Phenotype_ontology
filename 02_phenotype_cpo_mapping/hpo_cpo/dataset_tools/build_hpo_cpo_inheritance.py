#!/usr/bin/env python3
from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpo_mapping.io import file_record, read_tsv, write_json, write_tsv
from cpo_mapping.ontology import Ontology


MISSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MISSION_ROOT / "data/legacy_inputs"
DEFAULT_HPO_OBO = REPO_ROOT / 'data/ontologies/hp.obo'
DEFAULT_HPO_RELATIONS = MISSION_ROOT / "data/hpo_cpo/hpo_cpo_relations.tsv"
DEFAULT_HPO_TARGETS = (
    PROJECT_ROOT / "HPO_link_v6/work/hpo_term_selection_v1/selected_hpo_terms.tsv"
)
DEFAULT_MIDDLE_DECISIONS = (
    PROJECT_ROOT
    / "HPO_link_v6/work/hpo_literature_need/decisions/batches/"
    "hpo_depth2_internal_ecnu_max_b4_20260807.jsonl"
)
DEFAULT_OUTPUT = MISSION_ROOT / "data/hpo_cpo/inherited"
ROOT_HPO_ID = "HP:0000118"


PROVENANCE_FIELDS = (
    "assertion_id",
    "target_hpo_id",
    "source_leaf_hpo_id",
    "source_relation_id",
    "source_logical_row",
    "source_link_level",
    "distance",
    "shortest_path_count",
    "canonical_path",
)
ASSERTION_FIELDS = (
    "assertion_id",
    "target_hpo_id",
    "target_hpo_label",
    "target_node_type",
    "target_depth",
    "link_level",
    "relation_type",
    "direction",
    "cl_id",
    "cl_label",
    "representative_cell_text",
    "go_id",
    "go_label",
    "representative_process_text",
    "complete_cpo",
    "support_leaf_count",
    "support_relation_count",
    "minimum_distance",
    "maximum_distance",
    "eligible_descendant_leaf_count",
    "linked_descendant_leaf_count",
    "support_ratio_among_eligible_leaves",
    "support_ratio_among_linked_leaves",
)
STATUS_FIELDS = (
    "hpo_id",
    "hpo_label",
    "node_type",
    "depth",
    "is_target_leaf",
    "needs_self_query",
    "selected_for_processing",
    "middle_processing_status",
    "self_relation_count",
    "self_direct_relation_count",
    "self_mechanism_relation_count",
    "eligible_descendant_leaf_count",
    "linked_descendant_leaf_count",
    "inherited_relation_count",
    "inherited_assertion_count",
    "inherited_complete_cpo_count",
    "profile_mode",
    "inheritance_status",
)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def source_relation_id(row: Mapping[str, str]) -> str:
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "hpo_relation_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def assertion_key(target_id: str, row: Mapping[str, str]) -> tuple[str, ...]:
    cl_id = row.get("cl_id", "")
    go_id = row.get("go_id", "")
    return (
        target_id,
        row.get("link_level", ""),
        row.get("relation_type", ""),
        row.get("direction", ""),
        f"id:{cl_id}" if cl_id else f"text:{normalized_text(row.get('cell_text', ''))}",
        f"id:{go_id}" if go_id else f"text:{normalized_text(row.get('process_text', ''))}",
    )


def inherited_assertion_id(key: tuple[str, ...]) -> str:
    return "inherited_" + hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:20]


def shortest_depths(ontology: Ontology) -> dict[str, int]:
    result = {ontology.root_id: 0}
    queue = deque([ontology.root_id])
    while queue:
        parent = queue.popleft()
        for child in sorted(ontology.children[parent]):
            candidate = result[parent] + 1
            if child not in result or candidate < result[child]:
                result[child] = candidate
                queue.append(child)
    return result


def shortest_ancestor_paths(
    source_id: str, parents: Mapping[str, set[str]]
) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    seen_distance = {source_id: 0}
    frontier: dict[str, tuple[int, tuple[str, ...]]] = {source_id: (1, (source_id,))}
    result: dict[str, tuple[int, int, tuple[str, ...]]] = {}
    distance = 0
    while frontier:
        next_frontier: dict[str, tuple[int, tuple[str, ...]]] = {}
        for node, (path_count, path) in sorted(frontier.items()):
            for parent in sorted(parents.get(node, set())):
                candidate_distance = distance + 1
                if seen_distance.get(parent, candidate_distance) < candidate_distance:
                    continue
                candidate_path = (*path, parent)
                if parent in next_frontier:
                    count, old_path = next_frontier[parent]
                    next_frontier[parent] = (
                        count + path_count,
                        min(old_path, candidate_path),
                    )
                else:
                    next_frontier[parent] = (path_count, candidate_path)
        distance += 1
        for ancestor, (path_count, path) in next_frontier.items():
            seen_distance[ancestor] = distance
            result[ancestor] = (distance, path_count, path)
        frontier = next_frontier
    return result


def build_hpo_inheritance(
    hpo_obo: Path,
    hpo_relations_path: Path,
    hpo_targets_path: Path,
    middle_decisions_path: Path,
    output_root: Path,
) -> dict[str, object]:
    ontology = Ontology.from_obo(hpo_obo, ROOT_HPO_ID, "HP:")
    depths = shortest_depths(ontology)
    if set(depths) != ontology.nodes:
        raise ValueError("HPO abnormality scope contains nodes unreachable from its root")

    def node_type(term_id: str) -> str:
        if term_id == ROOT_HPO_ID:
            return "root"
        if depths[term_id] == 1:
            return "top_level"
        if ontology.children[term_id]:
            return "middle"
        return "leaf"

    relation_fields, relations = read_tsv(hpo_relations_path)
    required = {
        "hpo_id",
        "hpo_label",
        "link_level",
        "relation_type",
        "direction",
        "cl_id",
        "cl_label",
        "cell_text",
        "go_id",
        "go_label",
        "process_text",
    }
    if not required.issubset(relation_fields):
        raise ValueError("HPO relation TSV does not match the required schema")
    relation_ids = [source_relation_id(row) for row in relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("HPO source relation IDs derived from full rows are not unique")
    linked_leaf_ids = {row["hpo_id"] for row in relations}
    invalid_sources = sorted(
        term_id
        for term_id in linked_leaf_ids
        if term_id not in ontology.nodes or node_type(term_id) != "leaf"
    )
    if invalid_sources:
        raise ValueError(f"HPO relation sources are not abnormality leaves: {invalid_sources[:10]}")

    _, target_rows = read_tsv(hpo_targets_path)
    target_leaf_ids = {
        row["hpo_id"] for row in target_rows if row.get("node_type") == "leaf"
    }
    if len(target_leaf_ids) != 12121:
        raise ValueError(f"expected 12,121 target HPO leaves, found {len(target_leaf_ids)}")
    if not linked_leaf_ids.issubset(target_leaf_ids):
        raise ValueError("linked HPO leaves are absent from the target leaf cohort")
    selected_internal_ids = {
        row["hpo_id"]
        for row in target_rows
        if row.get("node_type") == "internal"
        and row.get("needs_parent_literature") == "true"
    }
    decision_rows = [
        json.loads(line)
        for line in middle_decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = {str(row["hpo_id"]): row for row in decision_rows}
    if len(decisions) != len(decision_rows):
        raise ValueError("HPO middle decisions contain duplicate IDs")
    expected_middle = {
        term_id for term_id in ontology.nodes if node_type(term_id) == "middle"
    }
    if set(decisions) != expected_middle:
        raise ValueError("HPO middle decisions do not equal the structural middle layer")
    if not selected_internal_ids.issubset(expected_middle):
        raise ValueError("selected HPO internal terms fall outside the middle layer")

    linked_paths = {
        leaf_id: shortest_ancestor_paths(leaf_id, ontology.parents)
        for leaf_id in sorted(linked_leaf_ids)
    }
    target_paths = {
        leaf_id: shortest_ancestor_paths(leaf_id, ontology.parents)
        for leaf_id in sorted(target_leaf_ids)
    }
    eligible_descendants: dict[str, set[str]] = defaultdict(set)
    linked_descendants: dict[str, set[str]] = defaultdict(set)
    for leaf_id, paths in target_paths.items():
        for ancestor in paths:
            eligible_descendants[ancestor].add(leaf_id)
    for leaf_id, paths in linked_paths.items():
        for ancestor in paths:
            linked_descendants[ancestor].add(leaf_id)

    aggregates: dict[tuple[str, ...], dict[str, object]] = {}
    inherited_counts: Counter[str] = Counter()
    distance_counts: Counter[int] = Counter()

    def provenance_rows() -> Iterable[dict[str, object]]:
        for logical_row, row in sorted(
            enumerate(relations, start=2),
            key=lambda value: (value[1]["hpo_id"], source_relation_id(value[1])),
        ):
            relation_id = source_relation_id(row)
            for target_id, (distance, path_count, path) in sorted(
                linked_paths[row["hpo_id"]].items(),
                key=lambda value: (value[1][0], value[0]),
            ):
                key = assertion_key(target_id, row)
                assertion_id = inherited_assertion_id(key)
                aggregate = aggregates.setdefault(
                    key,
                    {
                        "assertion_id": assertion_id,
                        "target_hpo_id": target_id,
                        "target_hpo_label": ontology.records[target_id].name,
                        "target_node_type": node_type(target_id),
                        "target_depth": depths[target_id],
                        "link_level": row["link_level"],
                        "relation_type": row["relation_type"],
                        "direction": row["direction"],
                        "cl_id": row["cl_id"],
                        "cl_label": row["cl_label"],
                        "representative_cell_text": row["cell_text"],
                        "go_id": row["go_id"],
                        "go_label": row["go_label"],
                        "representative_process_text": row["process_text"],
                        "complete_cpo": str(bool(row["cl_id"] and row["go_id"])).lower(),
                        "support_leaf_ids": set(),
                        "support_relation_count": 0,
                        "minimum_distance": distance,
                        "maximum_distance": distance,
                    },
                )
                aggregate["support_leaf_ids"].add(row["hpo_id"])
                aggregate["support_relation_count"] += 1
                aggregate["minimum_distance"] = min(aggregate["minimum_distance"], distance)
                aggregate["maximum_distance"] = max(aggregate["maximum_distance"], distance)
                inherited_counts[target_id] += 1
                distance_counts[distance] += 1
                yield {
                    "assertion_id": assertion_id,
                    "target_hpo_id": target_id,
                    "source_leaf_hpo_id": row["hpo_id"],
                    "source_relation_id": relation_id,
                    "source_logical_row": logical_row,
                    "source_link_level": row["link_level"],
                    "distance": distance,
                    "shortest_path_count": path_count,
                    "canonical_path": "|".join(path),
                }

    provenance_path = output_root / "hpo_cpo_inheritance_provenance.tsv"
    provenance_count = write_tsv(provenance_path, PROVENANCE_FIELDS, provenance_rows())

    assertion_rows: list[dict[str, object]] = []
    assertion_counts: Counter[str] = Counter()
    complete_counts: Counter[str] = Counter()
    for _key, aggregate in sorted(
        aggregates.items(), key=lambda value: (value[0][0], value[1]["assertion_id"])
    ):
        target_id = aggregate["target_hpo_id"]
        leaf_ids = aggregate.pop("support_leaf_ids")
        eligible_count = len(eligible_descendants.get(target_id, set()))
        linked_count = len(linked_descendants.get(target_id, set()))
        support_count = len(leaf_ids)
        aggregate["support_leaf_count"] = support_count
        aggregate["eligible_descendant_leaf_count"] = eligible_count
        aggregate["linked_descendant_leaf_count"] = linked_count
        aggregate["support_ratio_among_eligible_leaves"] = (
            f"{support_count / eligible_count:.8g}" if eligible_count else ""
        )
        aggregate["support_ratio_among_linked_leaves"] = (
            f"{support_count / linked_count:.8g}" if linked_count else ""
        )
        assertion_rows.append(aggregate)
        assertion_counts[target_id] += 1
        complete_counts[target_id] += aggregate["complete_cpo"] == "true"
    assertions_path = output_root / "hpo_cpo_inherited_assertions.tsv"
    assertion_count = write_tsv(assertions_path, ASSERTION_FIELDS, assertion_rows)

    self_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in relations:
        self_counts[row["hpo_id"]]["all"] += 1
        self_counts[row["hpo_id"]][row["link_level"]] += 1
    status_rows: list[dict[str, object]] = []
    type_counts: Counter[str] = Counter()
    profile_modes: Counter[str] = Counter()
    for term_id in sorted(ontology.nodes):
        kind = node_type(term_id)
        own = self_counts[term_id]["all"]
        inherited = inherited_counts[term_id]
        mode = (
            "self_and_inherited"
            if own and inherited
            else "self_only"
            if own
            else "inherited_only"
            if inherited
            else "none"
        )
        status = (
            "leaf_source_linked"
            if kind == "leaf" and own
            else "leaf_without_cpo"
            if kind == "leaf"
            else "inherited_from_descendant_leaves"
            if inherited
            else "no_linked_descendant_leaf"
        )
        decision = decisions.get(term_id)
        needs_self = bool(decision and decision["needs_parent_literature"])
        selected_for_processing = term_id in selected_internal_ids or term_id in target_leaf_ids
        if kind == "middle":
            if needs_self and term_id in selected_internal_ids:
                middle_status = (
                    "required_self_available"
                    if own
                    else "required_self_not_available"
                )
            elif needs_self:
                middle_status = "required_but_excluded_from_processing"
            else:
                middle_status = "inheritance_only"
        else:
            middle_status = ""
        status_rows.append(
            {
                "hpo_id": term_id,
                "hpo_label": ontology.records[term_id].name,
                "node_type": kind,
                "depth": depths[term_id],
                "is_target_leaf": str(term_id in target_leaf_ids).lower(),
                "needs_self_query": (
                    str(needs_self).lower() if decision is not None else ""
                ),
                "selected_for_processing": str(selected_for_processing).lower(),
                "middle_processing_status": middle_status,
                "self_relation_count": own,
                "self_direct_relation_count": self_counts[term_id]["direct"],
                "self_mechanism_relation_count": self_counts[term_id]["lead"],
                "eligible_descendant_leaf_count": len(eligible_descendants.get(term_id, set())),
                "linked_descendant_leaf_count": len(linked_descendants.get(term_id, set())),
                "inherited_relation_count": inherited,
                "inherited_assertion_count": assertion_counts[term_id],
                "inherited_complete_cpo_count": complete_counts[term_id],
                "profile_mode": mode,
                "inheritance_status": status,
            }
        )
        type_counts[kind] += 1
        profile_modes[mode] += 1
    status_path = output_root / "hpo_cpo_inheritance_term_status.tsv"
    term_count = write_tsv(status_path, STATUS_FIELDS, status_rows)

    if term_count != len(ontology.nodes):
        raise ValueError("HPO inheritance status does not cover the abnormality scope")
    if inherited_counts[ROOT_HPO_ID] != len(relations):
        raise ValueError("HPO root did not receive every leaf relation exactly once")
    if provenance_count != sum(row["support_relation_count"] for row in assertion_rows):
        raise ValueError("HPO aggregate support does not equal provenance count")

    manifest = {
        "schema_version": 1,
        "dataset_id": "mission4_hpo_cpo_leaf_to_ancestor_inheritance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Lossless upward inheritance of HPO leaf CPO relations to every abnormality is_a ancestor.",
        "rules": {
            "root_id": ROOT_HPO_ID,
            "source_node_type": "leaf",
            "inheritance_direction": "leaf_to_all_is_a_ancestors",
            "distance_cutoff": None,
            "direct_and_mechanism_kept_separate": True,
        },
        "counts": {
            "hpo_abnormality_term_count": len(ontology.nodes),
            "node_type_counts": dict(sorted(type_counts.items())),
            "target_leaf_count": len(target_leaf_ids),
            "selected_internal_processing_count": len(selected_internal_ids),
            "middle_decision_count": len(decisions),
            "middle_needs_self_query_count": sum(
                bool(row["needs_parent_literature"]) for row in decision_rows
            ),
            "middle_inheritance_only_count": sum(
                not bool(row["needs_parent_literature"]) for row in decision_rows
            ),
            "linked_leaf_count": len(linked_leaf_ids),
            "source_relation_count": len(relations),
            "source_link_level_counts": dict(sorted(Counter(row["link_level"] for row in relations).items())),
            "inherited_target_count": len(inherited_counts),
            "inheritance_provenance_count": provenance_count,
            "inherited_assertion_count": assertion_count,
            "inherited_complete_cpo_assertion_count": sum(row["complete_cpo"] == "true" for row in assertion_rows),
            "distance_counts": {str(key): value for key, value in sorted(distance_counts.items())},
            "profile_mode_counts": dict(sorted(profile_modes.items())),
        },
        "invariants": {
            "all_abnormality_terms_in_status": True,
            "all_sources_are_target_leaves": True,
            "middle_decisions_equal_structural_middle_layer": True,
            "root_receives_every_leaf_relation_exactly_once": True,
            "aggregate_support_equals_provenance_count": True,
        },
        "inputs": {
            "hpo_obo": file_record(hpo_obo),
            "hpo_relations": file_record(hpo_relations_path),
            "hpo_targets": file_record(hpo_targets_path),
            "middle_decisions": file_record(middle_decisions_path),
        },
        "outputs": {
            "inherited_assertions": assertions_path.name,
            "inheritance_provenance": provenance_path.name,
            "term_status": status_path.name,
        },
    }
    manifest["output_files"] = {
        "inherited_assertions": file_record(assertions_path),
        "inheritance_provenance": file_record(provenance_path),
        "term_status": file_record(status_path),
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inherit HPO leaf CPO relations upward")
    parser.add_argument("--hpo-obo", type=Path, default=DEFAULT_HPO_OBO)
    parser.add_argument("--hpo-relations", type=Path, default=DEFAULT_HPO_RELATIONS)
    parser.add_argument("--hpo-targets", type=Path, default=DEFAULT_HPO_TARGETS)
    parser.add_argument(
        "--middle-decisions", type=Path, default=DEFAULT_MIDDLE_DECISIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_hpo_inheritance(
                args.hpo_obo,
                args.hpo_relations,
                args.hpo_targets,
                args.middle_decisions,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
