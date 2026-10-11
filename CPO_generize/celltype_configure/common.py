from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import networkx as nx
import obonet


CL_ID_RE = re.compile(r"(?:CL[:_]|/CL_)(\d{7})")
GO_ID_RE = re.compile(r"GO:\d{7}")


@dataclass(frozen=True)
class CatalogNode:
    cl_id: str
    cl_name: str
    catalog_label: str
    parent_id: str
    anchor_id: str
    anchor_label: str
    source: str
    branch: str
    level: int


class CellCatalog:
    def __init__(self, nodes: Sequence[CatalogNode]) -> None:
        self.nodes = {node.cl_id: node for node in nodes}
        self.children: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node.parent_id:
                self.children[node.parent_id].append(node.cl_id)
        for values in self.children.values():
            values.sort()

    @classmethod
    def from_tsv(cls, path: str | Path) -> "CellCatalog":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            required = {
                "cl_id",
                "cl_name",
                "catalog_label",
                "parent_id",
                "anchor_id",
                "anchor_label",
                "source",
                "branch",
                "level",
            }
            missing = required - set(rows.fieldnames or [])
            if missing:
                raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
            nodes = [
                CatalogNode(
                    cl_id=row["cl_id"],
                    cl_name=row["cl_name"],
                    catalog_label=row["catalog_label"],
                    parent_id=row["parent_id"],
                    anchor_id=row["anchor_id"],
                    anchor_label=row["anchor_label"],
                    source=row["source"],
                    branch=row["branch"],
                    level=int(row["level"]),
                )
                for row in rows
            ]
        return cls(nodes)

    def ancestors(self, cl_id: str, include_self: bool = False) -> list[str]:
        result = [cl_id] if include_self and cl_id in self.nodes else []
        seen = set(result)
        current = cl_id
        while current in self.nodes:
            parent = self.nodes[current].parent_id
            if not parent or parent in seen:
                break
            result.append(parent)
            seen.add(parent)
            current = parent
        return result

    def descendants(self, cl_id: str, include_self: bool = False) -> list[str]:
        result: list[str] = [cl_id] if include_self and cl_id in self.nodes else []
        queue = deque(self.children.get(cl_id, []))
        seen = set(result)
        while queue:
            child = queue.popleft()
            if child in seen:
                continue
            seen.add(child)
            result.append(child)
            queue.extend(self.children.get(child, []))
        return result

    def common_ancestor(self, cl_ids: Sequence[str], allow_root: bool = False) -> str:
        valid = [cl_id for cl_id in cl_ids if cl_id in self.nodes]
        if not valid:
            return ""
        chains = [self.ancestors(cl_id, include_self=True) for cl_id in valid]
        shared = set(chains[0]).intersection(*map(set, chains[1:]))
        if not allow_root:
            shared.discard("CL:0000000")
        if not shared:
            return ""
        return max(shared, key=lambda value: self.nodes[value].level)


def load_obo(path: str | Path) -> nx.MultiDiGraph:
    return obonet.read_obo(str(path))


def ontology_parents(graph: nx.MultiDiGraph, term_id: str) -> Iterator[str]:
    if term_id not in graph:
        return
    for parent in graph.successors(term_id):
        if isinstance(parent, str):
            yield parent


def ontology_ancestors(
    graph: nx.MultiDiGraph, term_id: str, include_self: bool = False
) -> set[str]:
    result = {term_id} if include_self and term_id in graph else set()
    queue = deque([term_id])
    visited = {term_id}
    while queue:
        current = queue.popleft()
        for parent in ontology_parents(graph, current):
            if parent in visited:
                continue
            visited.add(parent)
            result.add(parent)
            queue.append(parent)
    return result


def nearest_catalog_ancestor(
    graph: nx.MultiDiGraph, catalog: CellCatalog, cl_id: str
) -> tuple[str, int]:
    if cl_id in catalog.nodes:
        return cl_id, 0
    queue = deque([(cl_id, 0)])
    visited = {cl_id}
    matches: list[tuple[str, int]] = []
    best_distance: int | None = None
    while queue:
        current, distance = queue.popleft()
        if best_distance is not None and distance > best_distance:
            break
        if current in catalog.nodes:
            matches.append((current, distance))
            best_distance = distance
            continue
        for parent in ontology_parents(graph, current):
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, distance + 1))
    if not matches:
        return "", -1
    # Multiple inheritance can yield ties. Prefer the deepest reviewed catalog node.
    best_match = max(matches, key=lambda item: catalog.nodes[item[0]].level)
    return best_match[0], best_match[1]


def normalize_cl_id(value: str) -> str:
    match = CL_ID_RE.search(value or "")
    return f"CL:{match.group(1)}" if match else ""


def clean_obo_text(value: str) -> str:
    if not value:
        return ""
    match = re.match(r'^"(.*)"(?:\s+\[.*\])?$', value)
    return match.group(1) if match else value


def term_synonyms(data: Mapping[str, object], limit: int = 6) -> list[str]:
    result: list[str] = []
    for raw in data.get("synonym", []) if isinstance(data.get("synonym"), list) else []:
        match = re.match(r'^"(.*?)"', str(raw))
        if match and match.group(1) not in result:
            result.append(match.group(1))
        if len(result) >= limit:
            break
    return result


def explicit_cl_references(data: Mapping[str, object]) -> list[str]:
    fields: list[str] = []
    for key in ("def", "comment", "synonym", "xref"):
        value = data.get(key, [])
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
        else:
            fields.append(str(value))
    return sorted({f"CL:{match}" for match in CL_ID_RE.findall("\n".join(fields))})


def read_go_ids(path: str | Path) -> list[str]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        if "\t" in first and "go_id" in first.split("\t"):
            values = [row["go_id"].strip() for row in csv.DictReader(handle, delimiter="\t")]
        else:
            values = [line.strip().split("\t", 1)[0] for line in handle]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        match = GO_ID_RE.search(value)
        if match and match.group(0) not in seen:
            seen.add(match.group(0))
            result.append(match.group(0))
    return result


def write_tsv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
