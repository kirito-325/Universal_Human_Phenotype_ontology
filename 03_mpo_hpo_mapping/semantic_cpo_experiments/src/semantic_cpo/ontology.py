from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


QUOTED_RE = re.compile(r'^\w+:\s+"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Term:
    term_id: str
    label: str
    namespace: str
    definition: str
    synonyms: tuple[str, ...]
    parents: tuple[str, ...]
    obsolete: bool

    def document(self) -> str:
        synonyms = "; ".join(dict.fromkeys(self.synonyms))
        return (
            f"Name: {self.label}\n"
            f"Exact synonyms: {synonyms}\n"
            f"Definition: {self.definition}"
        )


def parse_obo(path: str | Path) -> tuple[dict[str, Term], str]:
    records: dict[str, Term] = {}
    version = ""
    current: dict[str, object] | None = None

    def emit() -> None:
        nonlocal current
        if current and current.get("id"):
            term_id = str(current["id"])
            records[term_id] = Term(
                term_id=term_id,
                label=str(current.get("label", term_id)),
                namespace=str(current.get("namespace", "")),
                definition=str(current.get("definition", "")),
                synonyms=tuple(current.get("synonyms", ())),
                parents=tuple(current.get("parents", ())),
                obsolete=bool(current.get("obsolete", False)),
            )
        current = None

    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if current is None and line.startswith("data-version: "):
                version = line[14:].strip()
            if line.startswith("["):
                emit()
                if line == "[Term]":
                    current = {"parents": [], "synonyms": [], "obsolete": False}
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["label"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split()[0])
            elif line.startswith("def: "):
                match = QUOTED_RE.match(line)
                if match:
                    current["definition"] = match.group(1).replace(r'\"', '"')
            elif line.startswith("synonym: ") and " EXACT " in line:
                match = QUOTED_RE.match(line)
                if match:
                    current["synonyms"].append(match.group(1).replace(r'\"', '"'))
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        emit()
    return records, version


class Ontology:
    def __init__(
        self,
        records: dict[str, Term],
        root_id: str,
        namespace: str | None = None,
    ) -> None:
        self.records = records
        active = {
            term_id
            for term_id, record in records.items()
            if not record.obsolete and (namespace is None or record.namespace == namespace)
        }
        raw_children: dict[str, set[str]] = defaultdict(set)
        for child in active:
            for parent in records[child].parents:
                if parent in active:
                    raw_children[parent].add(child)
        nodes = {root_id}
        queue = deque([root_id])
        while queue:
            parent = queue.popleft()
            for child in raw_children.get(parent, set()):
                if child not in nodes:
                    nodes.add(child)
                    queue.append(child)
        self.root_id = root_id
        self.nodes = nodes
        self.parents = {
            node: set(records[node].parents) & nodes if node in records else set()
            for node in nodes
        }
        self.children = {node: raw_children.get(node, set()) & nodes for node in nodes}
        indegree = {node: len(self.parents[node]) for node in nodes}
        ready = deque(sorted(node for node, value in indegree.items() if value == 0))
        order: list[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for child in sorted(self.children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(order) != len(nodes):
            raise ValueError("ontology is not an is_a DAG")
        self.ancestors: dict[str, frozenset[str]] = {}
        for node in order:
            values: set[str] = set()
            for parent in self.parents[node]:
                values.add(parent)
                values.update(self.ancestors[parent])
            self.ancestors[node] = frozenset(values)
        descendants = {node: 0 for node in nodes}
        for values in self.ancestors.values():
            for ancestor in values:
                descendants[ancestor] += 1
        self.descendant_counts = descendants
        self.n = len(nodes)
        self.log_n = math.log(self.n) if self.n > 1 else 0.0
        self.ic = {
            node: -math.log((descendants[node] + 1) / self.n) if self.n > 1 else 0.0
            for node in nodes
        }

    @classmethod
    def from_obo(
        cls, path: str | Path, root_id: str, namespace: str | None = None
    ) -> "Ontology":
        records, _ = parse_obo(path)
        return cls(records, root_id, namespace)

    def mica(self, left: str, right: str) -> tuple[str | None, float]:
        common = ({left, *self.ancestors[left]}) & ({right, *self.ancestors[right]})
        if not common:
            return None, 0.0
        best = max(self.ic[node] for node in common)
        return min(node for node in common if self.ic[node] == best), best

    def similarity(self, method: str, left: str, right: str) -> float:
        if left not in self.nodes or right not in self.nodes:
            return 0.0
        if method == "exact":
            return float(left == right and left != self.root_id)
        _, mica_ic = self.mica(left, right)
        if method == "lin":
            denominator = self.ic[left] + self.ic[right]
            return 0.0 if denominator == 0 else 2.0 * mica_ic / denominator
        if method == "global_resnik":
            return 0.0 if self.log_n == 0 else mica_ic / self.log_n
        if method == "ic_ancestor_jaccard":
            a = {left, *self.ancestors[left]}
            b = {right, *self.ancestors[right]}
            denominator = sum(self.ic[node] for node in a | b)
            return 0.0 if denominator == 0 else sum(self.ic[node] for node in a & b) / denominator
        raise ValueError(f"unknown ontology similarity: {method}")

