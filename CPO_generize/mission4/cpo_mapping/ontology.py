from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TermRecord:
    term_id: str
    name: str
    namespace: str
    parents: tuple[str, ...]
    alt_ids: tuple[str, ...]
    obsolete: bool
    replaced_by: tuple[str, ...]
    consider: tuple[str, ...]


class Ontology:
    def __init__(
        self,
        records: dict[str, TermRecord],
        data_version: str,
        root_id: str,
        id_prefix: str,
        namespace: str | None = None,
    ) -> None:
        self.records = records
        self.data_version = data_version
        self.root_id = root_id
        self.id_prefix = id_prefix
        self.namespace = namespace
        self.alt_to_primary = {
            alt_id: term_id
            for term_id, record in records.items()
            for alt_id in record.alt_ids
        }
        self.replacement = {
            term_id: record.replaced_by[0]
            for term_id, record in records.items()
            if record.obsolete and len(record.replaced_by) == 1
        }
        self.active = {
            term_id
            for term_id, record in records.items()
            if not record.obsolete
            and term_id.startswith(id_prefix)
            and (namespace is None or record.namespace == namespace)
        }
        raw_parents = {
            term_id: {
                parent for parent in records[term_id].parents if parent in self.active
            }
            for term_id in self.active
        }
        raw_children: dict[str, set[str]] = defaultdict(set)
        for child, parents in raw_parents.items():
            for parent in parents:
                raw_children[parent].add(child)
        descendants = reachable(raw_children, root_id)
        self.nodes = {root_id, *descendants}
        self.parents = {
            term_id: raw_parents.get(term_id, set()) & self.nodes
            for term_id in self.nodes
        }
        self.children: dict[str, set[str]] = {
            term_id: raw_children.get(term_id, set()) & self.nodes
            for term_id in self.nodes
        }
        self.order = topological_order(self.nodes, self.parents, self.children)
        self.ancestors = build_ancestor_closure(self.order, self.parents)
        descendant_counts = {term_id: 0 for term_id in self.nodes}
        for ancestors in self.ancestors.values():
            for ancestor in ancestors:
                descendant_counts[ancestor] += 1
        count = len(self.nodes)
        self.descendant_counts = descendant_counts
        self.ic = {
            term_id: -math.log((1 + descendant_counts[term_id]) / count)
            for term_id in self.nodes
        }

    @classmethod
    def from_obo(
        cls,
        path: str | Path,
        root_id: str,
        id_prefix: str,
        namespace: str | None = None,
    ) -> "Ontology":
        records, data_version = parse_obo(path)
        return cls(records, data_version, root_id, id_prefix, namespace)

    def canonicalize(self, term_id: str) -> tuple[str | None, str]:
        if term_id in self.nodes:
            return term_id, "current"
        if term_id in self.alt_to_primary:
            canonical = self.alt_to_primary[term_id]
            return (canonical, "alt_id") if canonical in self.nodes else (None, "invalid_alt")
        if term_id in self.replacement:
            canonical = self.replacement[term_id]
            return (
                (canonical, "replaced_by")
                if canonical in self.nodes
                else (None, "invalid_replacement")
            )
        if term_id in self.records and self.records[term_id].obsolete:
            return None, "obsolete_unresolved"
        return None, "unknown_id"

    def mica(self, left: str, right: str) -> tuple[str | None, float, int]:
        common = ({left, *self.ancestors[left]}) & ({right, *self.ancestors[right]})
        if not common:
            return None, 0.0, 0
        best_ic = max(self.ic[node] for node in common)
        ties = sorted(node for node in common if self.ic[node] == best_ic)
        return ties[0], best_ic, len(ties)

    def lin(self, left: str, right: str) -> float:
        _mica, mica_ic, _ties = self.mica(left, right)
        denominator = self.ic[left] + self.ic[right]
        if denominator == 0:
            return 0.0
        score = 2 * mica_ic / denominator
        if score < -1e-12 or score > 1 + 1e-12:
            raise ValueError(f"Lin score outside [0,1]: {left}, {right}, {score}")
        return min(1.0, max(0.0, score))


def parse_obo(path: str | Path) -> tuple[dict[str, TermRecord], str]:
    records: dict[str, TermRecord] = {}
    data_version = ""
    current: dict[str, object] | None = None

    def emit() -> None:
        if current is None or not current.get("id"):
            return
        term_id = str(current["id"])
        records[term_id] = TermRecord(
            term_id=term_id,
            name=str(current.get("name", term_id)),
            namespace=str(current.get("namespace", "")),
            parents=tuple(current.get("parents", ())),
            alt_ids=tuple(current.get("alt_ids", ())),
            obsolete=bool(current.get("obsolete", False)),
            replaced_by=tuple(current.get("replaced_by", ())),
            consider=tuple(current.get("consider", ())),
        )

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if current is None and line.startswith("data-version: "):
                data_version = line[len("data-version: ") :]
            if line.startswith("["):
                emit()
                current = {
                    "parents": [],
                    "alt_ids": [],
                    "replaced_by": [],
                    "consider": [],
                    "obsolete": False,
                } if line == "[Term]" else None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:]
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split()[0])
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line[8:])
            elif line.startswith("replaced_by: "):
                current["replaced_by"].append(line[13:].split()[0])
            elif line.startswith("consider: "):
                current["consider"].append(line[10:].split()[0])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        emit()
    return records, data_version


def reachable(children: dict[str, set[str]], root_id: str) -> set[str]:
    result: set[str] = set()
    queue = deque([root_id])
    while queue:
        parent = queue.popleft()
        for child in sorted(children.get(parent, set())):
            if child not in result:
                result.add(child)
                queue.append(child)
    return result


def topological_order(
    nodes: Iterable[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> tuple[str, ...]:
    node_set = set(nodes)
    remaining = {node: len(parents.get(node, set())) for node in node_set}
    queue = deque(sorted(node for node, count in remaining.items() if count == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(children.get(node, set())):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if len(order) != len(node_set):
        cyclic = sorted(node for node, count in remaining.items() if count > 0)
        raise ValueError(f"ontology is not a DAG; cyclic nodes include {cyclic[:10]}")
    return tuple(order)


def build_ancestor_closure(
    order: Iterable[str], parents: dict[str, set[str]]
) -> dict[str, frozenset[str]]:
    closure: dict[str, frozenset[str]] = {}
    for node in order:
        values: set[str] = set()
        for parent in parents.get(node, set()):
            values.add(parent)
            values.update(closure[parent])
        closure[node] = frozenset(values)
    return closure
