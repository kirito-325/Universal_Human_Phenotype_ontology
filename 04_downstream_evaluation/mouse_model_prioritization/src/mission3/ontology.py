from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path


class Ontology:
    def __init__(self) -> None:
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        self.labels: dict[str, str] = {}
        self._ancestors: dict[str, dict[str, int]] = {}
        self._descendant_counts: dict[str, int] = {}

    @classmethod
    def from_obo(cls, path: str | Path | None) -> "Ontology":
        ontology = cls()
        if not path:
            return ontology
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        current: dict[str, object] = {}

        def commit() -> None:
            term_id = str(current.get("id", ""))
            if not term_id or current.get("obsolete"):
                return
            ontology.labels[term_id] = str(current.get("name", term_id))
            for parent in current.get("parents", []):
                ontology.parents[term_id].add(str(parent))
                ontology.children[str(parent)].add(term_id)

        with source.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line == "[Term]":
                    commit()
                    current = {"parents": []}
                elif line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current["name"] = line[6:].strip()
                elif line.startswith("is_a: "):
                    current.setdefault("parents", []).append(line[6:].split()[0])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
            commit()
        return ontology

    @property
    def nodes(self) -> set[str]:
        return set(self.labels) | set(self.parents) | set(self.children)

    def ancestors(self, term_id: str) -> dict[str, int]:
        if term_id in self._ancestors:
            return self._ancestors[term_id]
        result = {term_id: 0}
        queue = deque((parent, 1) for parent in self.parents.get(term_id, ()))
        while queue:
            node, distance = queue.popleft()
            if node in result and result[node] <= distance:
                continue
            result[node] = distance
            queue.extend((parent, distance + 1) for parent in self.parents.get(node, ()))
        self._ancestors[term_id] = result
        return result

    def _descendant_count(self, term_id: str) -> int:
        if term_id in self._descendant_counts:
            return self._descendant_counts[term_id]
        seen = {term_id}
        queue = deque(self.children.get(term_id, ()))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(self.children.get(node, ()))
        count = len(seen)
        self._descendant_counts[term_id] = count
        return count

    def information_content(self, term_id: str) -> float:
        node_count = max(1, len(self.nodes))
        return -math.log(max(1, self._descendant_count(term_id)) / node_count)

    def similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if not self.nodes or left not in self.nodes or right not in self.nodes:
            return 0.0
        common = set(self.ancestors(left)) & set(self.ancestors(right))
        if not common:
            return 0.0
        mica_ic = max(self.information_content(node) for node in common)
        denominator = max(self.information_content(left), self.information_content(right), 1e-9)
        return min(1.0, mica_ic / denominator)

