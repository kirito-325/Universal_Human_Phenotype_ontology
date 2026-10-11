#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


MISSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MISSION_ROOT.parent
DEFAULT_MPO_OBO = (
    PROJECT_ROOT / "03_MPO_link/data/processed/mpo_clean/mpo_clean.obo"
)
DEFAULT_LEAF_RELATIONS = MISSION_ROOT / "data/mpo_cpo/leaf/relations.tsv"
DEFAULT_LEAF_TERM_STATUS = MISSION_ROOT / "data/mpo_cpo/leaf/term_status.tsv"
DEFAULT_ALL_RELATIONS = MISSION_ROOT / "data/mpo_cpo/final/mpo_cpo_relations.tsv"
DEFAULT_MIDDLE_DECISIONS = (
    PROJECT_ROOT
    / "03_MPO_link/work/mpo_literature_batches/"
    "mgi_middle_layer_direct_20260831/final_middle_layer_terms.tsv"
)
DEFAULT_OUTPUT_ROOT = MISSION_ROOT / "data/mpo_cpo/inherited"

ROOT_MP_ID = "MP:0000001"
NORMAL_PHENOTYPE_ID = "MP:0002873"

PROVENANCE_FIELDS = (
    "assertion_id",
    "target_mp_id",
    "source_leaf_mp_id",
    "source_relation_id",
    "source_relation_scope",
    "distance",
    "shortest_path_count",
    "canonical_path",
)
ASSERTION_FIELDS = (
    "assertion_id",
    "target_mp_id",
    "target_mp_label",
    "target_node_type",
    "target_depth",
    "relation_scope",
    "relation_type",
    "direction",
    "cl_id",
    "cl_label",
    "representative_cell_text",
    "go_id",
    "go_label",
    "representative_process_text",
    "representative_quality_text",
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
TERM_STATUS_FIELDS = (
    "mp_id",
    "mp_label",
    "node_type",
    "depth",
    "is_normal_branch",
    "middle_query_status",
    "middle_selection_reason",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fields:
        raise ValueError(f"missing TSV header: {path}")
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"malformed TSV row at logical record {line_number}: {path}")
    return fields, rows


def write_tsv_atomic(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[Mapping[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_obo(
    path: Path,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    records: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None

    def emit() -> None:
        if current is not None and current.get("id"):
            records[str(current["id"])] = dict(current)

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("["):
                emit()
                current = {"parents": [], "obsolete": False} if line == "[Term]" else None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("is_a: "):
                parents = current["parents"]
                assert isinstance(parents, list)
                parents.append(line[6:].split()[0])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        emit()

    active = {
        term_id
        for term_id, record in records.items()
        if not bool(record.get("obsolete"))
    }
    labels = {
        term_id: str(records[term_id].get("name", term_id)) for term_id in active
    }
    parents = {
        term_id: {
            parent
            for parent in records[term_id].get("parents", [])
            if str(parent) in active
        }
        for term_id in active
    }
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[str(parent)].add(child)
    for term_id in active:
        children.setdefault(term_id, set())
    return labels, parents, dict(children)


def shortest_depths(children: Mapping[str, set[str]], root_id: str) -> dict[str, int]:
    if root_id not in children:
        raise ValueError(f"MPO root is missing: {root_id}")
    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        parent = queue.popleft()
        for child in sorted(children[parent]):
            candidate = depths[parent] + 1
            if child not in depths or candidate < depths[child]:
                depths[child] = candidate
                queue.append(child)
    return depths


def descendants(children: Mapping[str, set[str]], root_id: str) -> set[str]:
    result: set[str] = set()
    queue = deque([root_id])
    while queue:
        parent = queue.popleft()
        for child in sorted(children.get(parent, set())):
            if child not in result:
                result.add(child)
                queue.append(child)
    return result


def shortest_ancestor_paths(
    source_id: str,
    parents: Mapping[str, set[str]],
) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    """Return distance, shortest-path count, and canonical path for each ancestor."""
    seen_distance = {source_id: 0}
    frontier: dict[str, tuple[int, tuple[str, ...]]] = {
        source_id: (1, (source_id,))
    }
    result: dict[str, tuple[int, int, tuple[str, ...]]] = {}
    distance = 0
    while frontier:
        next_frontier: dict[str, tuple[int, tuple[str, ...]]] = {}
        for node, (path_count, path) in sorted(frontier.items()):
            for parent in sorted(parents.get(node, set())):
                candidate_distance = distance + 1
                prior_distance = seen_distance.get(parent)
                if prior_distance is not None and prior_distance < candidate_distance:
                    continue
                candidate_path = (*path, parent)
                if parent in next_frontier:
                    old_count, old_path = next_frontier[parent]
                    next_frontier[parent] = (
                        old_count + path_count,
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


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def assertion_key(target_id: str, row: Mapping[str, str]) -> tuple[str, ...]:
    cl_id = row.get("cl_id", "")
    go_id = row.get("go_id", "")
    cell_key = f"id:{cl_id}" if cl_id else f"text:{normalized_text(row.get('cell_text', ''))}"
    process_key = (
        f"id:{go_id}"
        if go_id
        else f"text:{normalized_text(row.get('process_text', ''))}"
    )
    return (
        target_id,
        row.get("relation_scope", ""),
        row.get("relation_type", ""),
        row.get("direction", ""),
        cell_key,
        process_key,
    )


def assertion_id(key: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:20]
    return f"inherited_{digest}"


def middle_query_status(row: Mapping[str, str]) -> str:
    needed = row.get("needs_parent_literature") == "true"
    selected = row.get("selected_for_relationship") == "true"
    if needed and selected:
        return "required_and_queried"
    if needed:
        return "required_but_unavailable"
    if selected:
        return "not_required_but_unique_resource_queried"
    return "not_required_not_queried"


def build_inheritance(
    mpo_obo: Path,
    leaf_relations_path: Path,
    leaf_term_status_path: Path,
    all_relations_path: Path,
    middle_decisions_path: Path,
    output_root: Path,
) -> dict[str, object]:
    labels, parents, children = parse_obo(mpo_obo)
    active_terms = set(labels)
    depths = shortest_depths(children, ROOT_MP_ID)
    if set(depths) != active_terms:
        missing = sorted(active_terms - set(depths))
        raise ValueError(f"active MPO terms are unreachable from root: {missing[:10]}")

    normal_branch = {NORMAL_PHENOTYPE_ID} | descendants(
        children, NORMAL_PHENOTYPE_ID
    )

    def node_type(term_id: str) -> str:
        if term_id in normal_branch:
            return "normal"
        if term_id == ROOT_MP_ID:
            return "root"
        if depths[term_id] == 1:
            return "top_level"
        if children[term_id]:
            return "middle"
        return "leaf"

    leaf_fields, leaf_relations = read_tsv(leaf_relations_path)
    required_leaf_fields = {
        "relation_id",
        "mp_id",
        "relation_scope",
        "relation_type",
        "direction",
        "cl_id",
        "cl_label",
        "cell_text",
        "go_id",
        "go_label",
        "process_text",
        "quality_text",
    }
    if not required_leaf_fields.issubset(leaf_fields):
        raise ValueError("leaf relation TSV does not match the expected schema")
    relation_ids = [row["relation_id"] for row in leaf_relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("leaf relation IDs are not unique")
    source_leaf_ids = {row["mp_id"] for row in leaf_relations}
    invalid_sources = sorted(
        term_id
        for term_id in source_leaf_ids
        if term_id not in active_terms or node_type(term_id) != "leaf"
    )
    if invalid_sources:
        raise ValueError(f"relation sources are not active abnormal leaves: {invalid_sources[:10]}")

    leaf_status_fields, leaf_status_rows = read_tsv(leaf_term_status_path)
    if "mp_id" not in leaf_status_fields:
        raise ValueError("leaf term-status TSV is missing mp_id")
    eligible_leaf_ids = {row["mp_id"] for row in leaf_status_rows}
    invalid_eligible = sorted(
        term_id
        for term_id in eligible_leaf_ids
        if term_id not in active_terms or node_type(term_id) != "leaf"
    )
    if invalid_eligible:
        raise ValueError(f"eligible source terms are not active abnormal leaves: {invalid_eligible[:10]}")
    if not source_leaf_ids.issubset(eligible_leaf_ids):
        raise ValueError("linked leaf relation sources are absent from leaf term status")

    all_relation_fields, all_relations = read_tsv(all_relations_path)
    required_all_fields = {"mp_id", "relation_scope"}
    if not required_all_fields.issubset(all_relation_fields):
        raise ValueError("combined relation TSV does not match the expected schema")
    self_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_relations:
        term_id = row["mp_id"]
        if term_id not in active_terms:
            raise ValueError(f"combined relation references unknown MPO term: {term_id}")
        self_counts[term_id]["all"] += 1
        self_counts[term_id][row["relation_scope"]] += 1

    middle_fields, middle_rows = read_tsv(middle_decisions_path)
    required_middle_fields = {
        "mp_id",
        "needs_parent_literature",
        "selected_for_relationship",
        "selection_reason",
    }
    if not required_middle_fields.issubset(middle_fields):
        raise ValueError("middle decision TSV does not match the expected schema")
    middle_decisions = {row["mp_id"]: row for row in middle_rows}
    if len(middle_decisions) != len(middle_rows):
        raise ValueError("middle decision TSV contains duplicate MPO IDs")
    expected_middle = {
        term_id for term_id in active_terms if node_type(term_id) == "middle"
    }
    if set(middle_decisions) != expected_middle:
        raise ValueError("middle decision IDs do not equal the structural middle layer")

    linked_ancestor_paths = {
        leaf_id: shortest_ancestor_paths(leaf_id, parents)
        for leaf_id in sorted(source_leaf_ids)
    }
    eligible_ancestor_paths = {
        leaf_id: shortest_ancestor_paths(leaf_id, parents)
        for leaf_id in sorted(eligible_leaf_ids)
    }
    eligible_descendant_leaves: dict[str, set[str]] = defaultdict(set)
    for leaf_id, ancestor_paths in eligible_ancestor_paths.items():
        for target_id in ancestor_paths:
            eligible_descendant_leaves[target_id].add(leaf_id)
    linked_descendant_leaves: dict[str, set[str]] = defaultdict(set)
    for leaf_id, ancestor_paths in linked_ancestor_paths.items():
        for target_id in ancestor_paths:
            linked_descendant_leaves[target_id].add(leaf_id)

    aggregates: dict[tuple[str, ...], dict[str, object]] = {}
    inherited_relation_counts: Counter[str] = Counter()
    inheritance_distance_counts: Counter[int] = Counter()
    provenance_path = output_root / "mpo_cpo_inheritance_provenance.tsv"

    def provenance_rows() -> Iterable[dict[str, object]]:
        for row in sorted(
            leaf_relations, key=lambda value: (value["mp_id"], value["relation_id"])
        ):
            source_id = row["mp_id"]
            for target_id, (distance, path_count, path) in sorted(
                linked_ancestor_paths[source_id].items(),
                key=lambda value: (value[1][0], value[0]),
            ):
                key = assertion_key(target_id, row)
                inherited_id = assertion_id(key)
                aggregate = aggregates.get(key)
                if aggregate is None:
                    aggregate = {
                        "assertion_id": inherited_id,
                        "target_mp_id": target_id,
                        "target_mp_label": labels[target_id],
                        "target_node_type": node_type(target_id),
                        "target_depth": depths[target_id],
                        "relation_scope": row["relation_scope"],
                        "relation_type": row["relation_type"],
                        "direction": row["direction"],
                        "cl_id": row["cl_id"],
                        "cl_label": row["cl_label"],
                        "representative_cell_text": row["cell_text"],
                        "go_id": row["go_id"],
                        "go_label": row["go_label"],
                        "representative_process_text": row["process_text"],
                        "representative_quality_text": row["quality_text"],
                        "complete_cpo": str(bool(row["cl_id"] and row["go_id"])).lower(),
                        "support_leaf_ids": set(),
                        "support_relation_count": 0,
                        "minimum_distance": distance,
                        "maximum_distance": distance,
                    }
                    aggregates[key] = aggregate
                support_leaf_ids = aggregate["support_leaf_ids"]
                assert isinstance(support_leaf_ids, set)
                support_leaf_ids.add(source_id)
                aggregate["support_relation_count"] = (
                    int(aggregate["support_relation_count"]) + 1
                )
                aggregate["minimum_distance"] = min(
                    int(aggregate["minimum_distance"]), distance
                )
                aggregate["maximum_distance"] = max(
                    int(aggregate["maximum_distance"]), distance
                )
                inherited_relation_counts[target_id] += 1
                inheritance_distance_counts[distance] += 1
                yield {
                    "assertion_id": inherited_id,
                    "target_mp_id": target_id,
                    "source_leaf_mp_id": source_id,
                    "source_relation_id": row["relation_id"],
                    "source_relation_scope": row["relation_scope"],
                    "distance": distance,
                    "shortest_path_count": path_count,
                    "canonical_path": "|".join(path),
                }

    provenance_count = write_tsv_atomic(
        provenance_path, PROVENANCE_FIELDS, provenance_rows()
    )

    assertion_rows: list[dict[str, object]] = []
    assertion_counts: Counter[str] = Counter()
    complete_assertion_counts: Counter[str] = Counter()
    for key, aggregate in sorted(
        aggregates.items(), key=lambda value: (value[0][0], value[1]["assertion_id"])
    ):
        target_id = str(aggregate["target_mp_id"])
        support_leaf_ids = aggregate.pop("support_leaf_ids")
        assert isinstance(support_leaf_ids, set)
        eligible_count = len(eligible_descendant_leaves.get(target_id, set()))
        linked_count = len(linked_descendant_leaves.get(target_id, set()))
        support_count = len(support_leaf_ids)
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
        if aggregate["complete_cpo"] == "true":
            complete_assertion_counts[target_id] += 1

    assertion_path = output_root / "mpo_cpo_inherited_assertions.tsv"
    assertion_count = write_tsv_atomic(
        assertion_path, ASSERTION_FIELDS, assertion_rows
    )

    status_rows: list[dict[str, object]] = []
    profile_mode_counts: Counter[str] = Counter()
    inheritance_status_counts: Counter[str] = Counter()
    node_type_counts: Counter[str] = Counter()
    for term_id in sorted(active_terms):
        kind = node_type(term_id)
        self_relation_count = self_counts[term_id]["all"]
        inherited_relation_count = inherited_relation_counts[term_id]
        if self_relation_count and inherited_relation_count:
            profile_mode = "self_and_inherited"
        elif self_relation_count:
            profile_mode = "self_only"
        elif inherited_relation_count:
            profile_mode = "inherited_only"
        else:
            profile_mode = "none"
        if term_id in normal_branch:
            inheritance_status = "not_applicable_normal_branch"
        elif kind == "leaf":
            inheritance_status = (
                "leaf_source_linked" if term_id in source_leaf_ids else "leaf_without_cpo"
            )
        elif inherited_relation_count:
            inheritance_status = "inherited_from_descendant_leaves"
        else:
            inheritance_status = "no_linked_descendant_leaf"
        decision = middle_decisions.get(term_id)
        status_rows.append(
            {
                "mp_id": term_id,
                "mp_label": labels[term_id],
                "node_type": kind,
                "depth": depths[term_id],
                "is_normal_branch": str(term_id in normal_branch).lower(),
                "middle_query_status": middle_query_status(decision) if decision else "",
                "middle_selection_reason": decision.get("selection_reason", "") if decision else "",
                "self_relation_count": self_relation_count,
                "self_direct_relation_count": self_counts[term_id]["direct"],
                "self_mechanism_relation_count": self_counts[term_id]["mechanism"],
                "eligible_descendant_leaf_count": len(
                    eligible_descendant_leaves.get(term_id, set())
                ),
                "linked_descendant_leaf_count": len(
                    linked_descendant_leaves.get(term_id, set())
                ),
                "inherited_relation_count": inherited_relation_count,
                "inherited_assertion_count": assertion_counts[term_id],
                "inherited_complete_cpo_count": complete_assertion_counts[term_id],
                "profile_mode": profile_mode,
                "inheritance_status": inheritance_status,
            }
        )
        profile_mode_counts[profile_mode] += 1
        inheritance_status_counts[inheritance_status] += 1
        node_type_counts[kind] += 1

    status_path = output_root / "mpo_cpo_inheritance_term_status.tsv"
    term_count = write_tsv_atomic(status_path, TERM_STATUS_FIELDS, status_rows)

    inherited_targets = set(inherited_relation_counts)
    inherited_assertion_ids = {
        str(row["assertion_id"]) for row in assertion_rows
    }
    if provenance_count != sum(inherited_relation_counts.values()):
        raise ValueError("provenance counts do not equal inherited relation counts")
    if provenance_count != sum(
        int(row["support_relation_count"]) for row in assertion_rows
    ):
        raise ValueError("aggregate support counts do not equal provenance count")
    if term_count != len(active_terms):
        raise ValueError("term status does not cover every active MPO term")
    if inherited_targets & normal_branch:
        raise ValueError("normal-phenotype terms received abnormal leaf inheritance")
    if ROOT_MP_ID not in inherited_targets:
        raise ValueError("MPO root did not receive inherited leaf relations")
    if inherited_relation_counts[ROOT_MP_ID] != len(leaf_relations):
        raise ValueError("MPO root did not receive every leaf relation exactly once")
    if len(inherited_assertion_ids) != assertion_count:
        raise ValueError("inherited assertion IDs are not unique")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "mission4_mpo_cpo_leaf_to_ancestor_inheritance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "Lossless upward inheritance of accepted leaf MPO-CPO relations to every "
            "is_a ancestor in the complete active MPO DAG."
        ),
        "rules": {
            "source_node_type": "leaf",
            "inheritance_direction": "leaf_to_all_is_a_ancestors",
            "distance_cutoff": None,
            "middle_self_relations_propagated": False,
            "direct_and_mechanism_scopes_kept_separate": True,
            "multi_parent_deduplication": (
                "one target/source-relation row with shortest distance, shortest-path "
                "count, and lexicographically first shortest path"
            ),
            "assertion_identity": (
                "target MPO + relation scope/type/direction + CL ID or normalized cell "
                "text + GO ID or normalized process text"
            ),
            "normal_phenotype_branch_inheritable": False,
        },
        "counts": {
            "active_mpo_term_count": len(active_terms),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "normal_branch_term_count": len(normal_branch),
            "eligible_leaf_term_count": len(eligible_leaf_ids),
            "linked_leaf_term_count": len(source_leaf_ids),
            "leaf_relation_count": len(leaf_relations),
            "leaf_relation_scope_counts": dict(
                sorted(Counter(row["relation_scope"] for row in leaf_relations).items())
            ),
            "middle_decision_count": len(middle_decisions),
            "middle_query_need_counts": dict(
                sorted(
                    Counter(
                        row["needs_parent_literature"] for row in middle_rows
                    ).items()
                )
            ),
            "inherited_target_count": len(inherited_targets),
            "inherited_target_node_type_counts": dict(
                sorted(Counter(node_type(term_id) for term_id in inherited_targets).items())
            ),
            "middle_term_count": len(expected_middle),
            "middle_term_with_inheritance_count": len(
                expected_middle & inherited_targets
            ),
            "inheritance_provenance_count": provenance_count,
            "inheritance_distance_counts": {
                str(distance): count
                for distance, count in sorted(inheritance_distance_counts.items())
            },
            "inherited_assertion_count": assertion_count,
            "inherited_complete_cpo_assertion_count": sum(
                row["complete_cpo"] == "true" for row in assertion_rows
            ),
            "inherited_assertion_scope_counts": dict(
                sorted(Counter(str(row["relation_scope"]) for row in assertion_rows).items())
            ),
            "profile_mode_counts": dict(sorted(profile_mode_counts.items())),
            "inheritance_status_counts": dict(
                sorted(inheritance_status_counts.items())
            ),
        },
        "invariants": {
            "all_active_terms_in_status": True,
            "leaf_source_relation_ids_unique": True,
            "all_relation_sources_are_active_abnormal_leaves": True,
            "middle_decisions_equal_structural_middle_layer": True,
            "normal_branch_inherited_relation_count": 0,
            "root_receives_leaf_inheritance": True,
            "root_receives_every_leaf_relation_exactly_once": True,
            "aggregate_support_equals_provenance_count": True,
            "inherited_assertion_ids_unique": True,
        },
        "inputs": {
            "mpo_obo": {"path": str(mpo_obo.resolve()), "sha256": sha256(mpo_obo)},
            "leaf_relations": {
                "path": str(leaf_relations_path.resolve()),
                "sha256": sha256(leaf_relations_path),
            },
            "leaf_term_status": {
                "path": str(leaf_term_status_path.resolve()),
                "sha256": sha256(leaf_term_status_path),
            },
            "all_relations": {
                "path": str(all_relations_path.resolve()),
                "sha256": sha256(all_relations_path),
            },
            "middle_decisions": {
                "path": str(middle_decisions_path.resolve()),
                "sha256": sha256(middle_decisions_path),
            },
        },
        "outputs": {
            "inherited_assertions": assertion_path.name,
            "inheritance_provenance": provenance_path.name,
            "term_status": status_path.name,
        },
    }
    manifest["output_sha256"] = {
        "inherited_assertions": sha256(assertion_path),
        "inheritance_provenance": sha256(provenance_path),
        "term_status": sha256(status_path),
    }
    manifest_path = output_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inherit leaf MPO-CPO relations upward through the complete MPO DAG"
    )
    parser.add_argument("--mpo-obo", type=Path, default=DEFAULT_MPO_OBO)
    parser.add_argument(
        "--leaf-relations", type=Path, default=DEFAULT_LEAF_RELATIONS
    )
    parser.add_argument(
        "--leaf-term-status", type=Path, default=DEFAULT_LEAF_TERM_STATUS
    )
    parser.add_argument("--all-relations", type=Path, default=DEFAULT_ALL_RELATIONS)
    parser.add_argument(
        "--middle-decisions", type=Path, default=DEFAULT_MIDDLE_DECISIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = build_inheritance(
        args.mpo_obo,
        args.leaf_relations,
        args.leaf_term_status,
        args.all_relations,
        args.middle_decisions,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
