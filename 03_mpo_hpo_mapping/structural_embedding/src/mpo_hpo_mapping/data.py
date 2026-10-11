from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import csv
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


ROOT = REPO_ROOT / "data/legacy_structural_embedding"


@dataclass(frozen=True)
class InputPaths:
    hpo_specific: Path = ROOT / "HPO_link/results/all_hpo_default_20260725/hpo_cell_mechanism_triples.tsv"
    hpo_general: Path = ROOT / "HPO_link/results/all_hpo_default_20260725/hpo_cell_general_mechanism_triples.tsv"
    mpo_specific: Path = ROOT / "03_MPO_link/results/mpo_cpo_links/api_reasoner_final_20260726/mpo_cell_mechanism_triples.tsv"
    mpo_general: Path = ROOT / "03_MPO_link/results/mpo_cpo_links/api_reasoner_final_20260726/mpo_cell_general_mechanism_triples.tsv"
    expert_mappings: Path = REPO_ROOT / 'data/reference/mhmi_mp_to_hp_mappings.tsv'
    hpo_edges: Path = REPO_ROOT / 'data/reference/hpo_is_a_edges.tsv'
    hpo_obo: Path = REPO_ROOT / 'data/ontologies/hp.obo'
    mpo_obo: Path = ROOT / "03_MPO_link/data/processed/mpo_clean/mpo_clean.obo"
    cl_catalog: Path = REPO_ROOT / '01_cpo_construction/cell_marker_mining/expanded/expanded_cell_catalog.tsv'
    go_obo: Path = REPO_ROOT / 'data/ontologies/go-basic.obo'

    def validate(self) -> None:
        missing = [str(path) for path in asdict(self).values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


@dataclass(frozen=True)
class MechanismEdge:
    term_id: str
    term_name: str
    cl_id: str
    cl_name: str
    go_id: str
    go_name: str
    channel: str
    weight: float
    rank: int
    confidence: str
    supporting_genes: str

    @property
    def mechanism_id(self) -> str:
        return f"{self.cl_id}|{self.go_id}"


@dataclass(frozen=True)
class MappingExample:
    mp_id: str
    hp_id: str
    relation: str
    confidence: float
    split: str
    source: str
    query_profile_source: str = "mp"
    exact_hp_id: str = ""
    ancestor_distance: int = 0
    masked_hp_ids: str = ""


class Ontology:
    def __init__(
        self,
        nodes: Iterable[str],
        edges: Iterable[tuple[str, str]],
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.nodes = set(nodes)
        self.parents: dict[str, list[str]] = defaultdict(list)
        self.children: dict[str, list[str]] = defaultdict(list)
        self.labels = dict(labels or {})
        for child, parent in edges:
            if not child or not parent or child == parent:
                continue
            self.nodes.update((child, parent))
            self.parents[child].append(parent)
            self.children[parent].append(child)
        for index in (self.parents, self.children):
            for values in index.values():
                values[:] = sorted(set(values))

    def ancestors(self, term_id: str, include_self: bool = False) -> dict[str, int]:
        result = {term_id: 0} if include_self and term_id in self.nodes else {}
        queue = deque((parent, 1) for parent in self.parents.get(term_id, ()))
        while queue:
            node, distance = queue.popleft()
            if node in result and result[node] <= distance:
                continue
            result[node] = distance
            queue.extend((parent, distance + 1) for parent in self.parents.get(node, ()))
        return result

    def nearest_available_ancestors(
        self,
        term_id: str,
        available: set[str],
        excluded: set[str] | None = None,
        max_distance: int = 12,
    ) -> tuple[list[str], int]:
        excluded = excluded or set()
        queue = deque((parent, 1) for parent in self.parents.get(term_id, ()))
        seen = {term_id}
        matches: list[str] = []
        best_distance = 0
        while queue:
            node, distance = queue.popleft()
            if node in seen or distance > max_distance:
                continue
            seen.add(node)
            if best_distance and distance > best_distance:
                break
            if node in available and node not in excluded:
                matches.append(node)
                best_distance = distance
                continue
            queue.extend((parent, distance + 1) for parent in self.parents.get(node, ()))
        return sorted(set(matches)), best_distance

    def branch_key(self, term_id: str, root_id: str = "HP:0000118") -> str:
        if term_id == root_id:
            return root_id
        ancestors = self.ancestors(term_id, include_self=True)
        branches = [node for node in ancestors if root_id in self.parents.get(node, ())]
        if not branches:
            return min(ancestors, key=lambda node: (ancestors[node], node), default=term_id)
        return min(branches, key=lambda node: (ancestors[node], node))


def read_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _float(value: str, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_value(row: Mapping[str, str], *columns: str) -> str:
    for column in columns:
        value = row.get(column, "")
        if value:
            return value
    return ""


def _edge_from_row(
    row: Mapping[str, str], term_column: str, channel: str, general_weight: float
) -> MechanismEdge | None:
    term_id = row.get(term_column, "")
    cl_id = row.get("cell_type_id", "")
    go_id = _first_value(row, "cell_phenotype_id", "go_id")
    if not term_id or not cl_id or not go_id:
        return None
    anatomy_status = _first_value(
        row, "anatomy_constraint_status", "anatomy_status"
    ).lower()
    if anatomy_status in {"blocked", "conflict", "incompatible", "rejected"}:
        return None
    confidence = _first_value(row, "confidence_grade", "evidence_level").lower()
    confidence_weight = {
        "high": 1.0,
        "medium": 0.85,
        "low": 0.65,
        "exploratory": 0.65,
    }.get(confidence, 0.75)
    score = _float(
        _first_value(row, "rerank_score", "final_score", "evidence_score"), 0.1
    )
    channel_weight = 1.0 if channel == "specific" else general_weight
    weight = max(1e-4, min(1.0, score) * confidence_weight * channel_weight)
    return MechanismEdge(
        term_id=term_id,
        term_name=row.get("hpo_name", row.get("mp_name", "")),
        cl_id=cl_id,
        cl_name=row.get("cell_type", ""),
        go_id=go_id,
        go_name=_first_value(row, "cell_phenotype", "go_name"),
        channel=channel,
        weight=weight,
        rank=_int(_first_value(row, "final_rank", "hpo_link_rank"), 10**9),
        confidence=confidence,
        supporting_genes=_first_value(row, "supporting_genes", "mechanism_genes"),
    )


def load_term_profiles(
    specific_path: str | Path,
    general_path: str | Path,
    term_column: str,
    general_top_k: int = 8,
    general_weight: float = 0.15,
) -> dict[str, list[MechanismEdge]]:
    profiles: dict[str, dict[str, MechanismEdge]] = defaultdict(dict)
    for row in read_tsv(specific_path):
        edge = _edge_from_row(row, term_column, "specific", general_weight)
        if edge is None:
            continue
        current = profiles[edge.term_id].get(edge.mechanism_id)
        if current is None or edge.weight > current.weight:
            profiles[edge.term_id][edge.mechanism_id] = edge

    if general_top_k > 0:
        general_by_term: dict[str, dict[str, MechanismEdge]] = defaultdict(dict)
        for row in read_tsv(general_path):
            edge = _edge_from_row(row, term_column, "general", general_weight)
            if edge is None:
                continue
            current = general_by_term[edge.term_id].get(edge.mechanism_id)
            if current is None or (edge.rank, -edge.weight) < (current.rank, -current.weight):
                general_by_term[edge.term_id][edge.mechanism_id] = edge
        for term_id, candidates in general_by_term.items():
            ordered = sorted(candidates.values(), key=lambda edge: (edge.rank, -edge.weight, edge.mechanism_id))
            for edge in ordered[:general_top_k]:
                current = profiles[term_id].get(edge.mechanism_id)
                if current is None:
                    profiles[term_id][edge.mechanism_id] = edge

    return {
        term_id: sorted(edges.values(), key=lambda edge: (edge.channel != "specific", edge.rank, edge.mechanism_id))
        for term_id, edges in profiles.items()
    }


def randomize_term_profiles(
    profiles: Mapping[str, Sequence[MechanismEdge]],
    seed: int,
    swaps_per_edge: int = 10,
) -> tuple[dict[str, list[MechanismEdge]], dict[str, object]]:
    """Randomize term-mechanism links without changing either endpoint degree.

    Double-edge swaps are performed separately for specific and general channels.
    Slot-level weights, ranks, and confidence stay attached to their HPO term while
    the complete CL-GO mechanism payload moves between terms.
    """

    edges = [edge for term_id in sorted(profiles) for edge in profiles[term_id]]
    original_mechanisms = [edge.mechanism_id for edge in edges]
    membership = {
        term_id: {edge.mechanism_id for edge in term_edges}
        for term_id, term_edges in profiles.items()
    }
    indices_by_channel: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        indices_by_channel[edge.channel].append(index)

    randomizer = random.Random(seed)
    channel_stats: dict[str, dict[str, int]] = {}
    for channel, indices in sorted(indices_by_channel.items()):
        attempts = max(0, swaps_per_edge) * len(indices)
        successful = 0
        if len(indices) >= 2:
            for _ in range(attempts):
                left_position = randomizer.randrange(len(indices))
                right_position = randomizer.randrange(len(indices) - 1)
                if right_position >= left_position:
                    right_position += 1
                left_index = indices[left_position]
                right_index = indices[right_position]
                left = edges[left_index]
                right = edges[right_index]
                if left.term_id == right.term_id or left.mechanism_id == right.mechanism_id:
                    continue
                if right.mechanism_id in membership[left.term_id]:
                    continue
                if left.mechanism_id in membership[right.term_id]:
                    continue

                membership[left.term_id].remove(left.mechanism_id)
                membership[right.term_id].remove(right.mechanism_id)
                membership[left.term_id].add(right.mechanism_id)
                membership[right.term_id].add(left.mechanism_id)
                edges[left_index] = replace(
                    left,
                    cl_id=right.cl_id,
                    cl_name=right.cl_name,
                    go_id=right.go_id,
                    go_name=right.go_name,
                    supporting_genes=right.supporting_genes,
                )
                edges[right_index] = replace(
                    right,
                    cl_id=left.cl_id,
                    cl_name=left.cl_name,
                    go_id=left.go_id,
                    go_name=left.go_name,
                    supporting_genes=left.supporting_genes,
                )
                successful += 1
        channel_stats[channel] = {
            "edges": len(indices),
            "attempted_swaps": attempts,
            "successful_swaps": successful,
        }

    randomized: dict[str, list[MechanismEdge]] = defaultdict(list)
    for edge in edges:
        randomized[edge.term_id].append(edge)
    changed = sum(
        edge.mechanism_id != original
        for edge, original in zip(edges, original_mechanisms)
    )
    return (
        {
            term_id: sorted(
                term_edges,
                key=lambda edge: (
                    edge.channel != "specific",
                    edge.rank,
                    edge.mechanism_id,
                ),
            )
            for term_id, term_edges in sorted(randomized.items())
        },
        {
            "algorithm": "channel_stratified_bipartite_double_edge_swap",
            "seed": seed,
            "swaps_per_edge": swaps_per_edge,
            "total_edges": len(edges),
            "changed_edges": changed,
            "changed_fraction": changed / max(1, len(edges)),
            "term_degrees_preserved": True,
            "mechanism_degrees_preserved": True,
            "channels": channel_stats,
        },
    )


def load_hpo_ontology(path: str | Path) -> Ontology:
    edges: list[tuple[str, str]] = []
    labels: dict[str, str] = {}
    for row in read_tsv(path):
        child = row.get("child_id", "")
        parent = row.get("parent_id", "")
        if child and parent:
            edges.append((child, parent))
            labels[child] = row.get("child_name", "")
            labels[parent] = row.get("parent_name", "")
    return Ontology(labels, edges, labels)


def load_cl_ontology(path: str | Path) -> Ontology:
    edges: list[tuple[str, str]] = []
    labels: dict[str, str] = {}
    for row in read_tsv(path):
        node = row.get("cl_id", "")
        parent = row.get("parent_id", "")
        if not node:
            continue
        labels[node] = row.get("catalog_label", row.get("cl_name", ""))
        if parent:
            edges.append((node, parent))
    return Ontology(labels, edges, labels)


def load_go_bp_ontology(path: str | Path) -> Ontology:
    terms: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    in_term = False

    def commit() -> None:
        nonlocal current
        if current and current.get("id"):
            terms[str(current["id"])] = current
        current = None

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                commit()
                current = {"parents": []}
                in_term = True
                continue
            if line.startswith("["):
                commit()
                in_term = False
                continue
            if not in_term or current is None:
                continue
            if line.startswith("id: GO:"):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("is_a: GO:"):
                parent = line[6:].split()[0]
                current["parents"].append((parent, "is_a"))  # type: ignore[union-attr]
            elif line.startswith("relationship: part_of GO:"):
                parent = line[len("relationship: part_of ") :].split()[0]
                current["parents"].append((parent, "part_of"))  # type: ignore[union-attr]
    commit()

    bp = {
        term_id
        for term_id, data in terms.items()
        if data.get("namespace") == "biological_process" and not data.get("obsolete")
    }
    labels = {term_id: str(terms[term_id].get("name", "")) for term_id in bp}
    edges = [
        (term_id, parent)
        for term_id in bp
        for parent, _relation in terms[term_id].get("parents", [])  # type: ignore[assignment]
        if parent in bp
    ]
    return Ontology(bp, edges, labels)


def stable_split(key: str, seed: int = 20260726) -> str:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "validation"
    return "test"


RELATION_MAP = {
    "skos:exactMatch": "exact",
    "skos:broadMatch": "hp_broader",
    "skos:narrowMatch": "hp_narrower",
    "skos:relatedMatch": "related",
    "skos:closeMatch": "close",
}


def load_expert_examples(
    path: str | Path,
    mp_terms: set[str],
    hp_terms: set[str],
    min_confidence: float = 0.9,
    seed: int = 20260726,
) -> list[MappingExample]:
    priority = {"exact": 5, "hp_broader": 4, "hp_narrower": 3, "related": 2, "close": 1}
    selected: dict[tuple[str, str], MappingExample] = {}
    for row in read_tsv(path):
        mp_id = row.get("mp_id", "")
        hp_id = row.get("hp_id", "")
        confidence = _float(row.get("confidence", ""))
        relation = RELATION_MAP.get(row.get("predicate_id", ""), "")
        if not relation or confidence < min_confidence or mp_id not in mp_terms or hp_id not in hp_terms:
            continue
        example = MappingExample(
            mp_id=mp_id,
            hp_id=hp_id,
            relation=relation,
            confidence=confidence,
            split=stable_split(mp_id, seed),
            source="mhmi_manual",
        )
        key = (mp_id, hp_id)
        current = selected.get(key)
        if current is None or priority[example.relation] > priority[current.relation]:
            selected[key] = example
    return sorted(selected.values(), key=lambda row: (row.mp_id, row.hp_id))


def build_parent_examples(
    expert: Sequence[MappingExample],
    hpo: Ontology,
    hp_terms: set[str],
    synthetic_limit: int = 0,
    seed: int = 20260726,
) -> list[MappingExample]:
    examples: dict[tuple[str, str, str, str], MappingExample] = {}
    exact_by_mp: dict[str, set[str]] = defaultdict(set)
    for row in expert:
        if row.relation == "exact":
            exact_by_mp[row.mp_id].add(row.hp_id)

    for row in expert:
        if row.relation == "hp_broader":
            key = (row.mp_id, row.hp_id, "expert_broad", "mp")
            examples[key] = MappingExample(
                mp_id=row.mp_id,
                hp_id=row.hp_id,
                relation="hp_broader",
                confidence=row.confidence,
                split=row.split,
                source="expert_broad",
            )
        elif row.relation == "exact":
            masked_exact = exact_by_mp[row.mp_id]
            parents, distance = hpo.nearest_available_ancestors(
                row.hp_id, hp_terms, excluded=masked_exact
            )
            for parent in parents:
                key = (row.mp_id, parent, "counterfactual_exact", "mp")
                examples[key] = MappingExample(
                    mp_id=row.mp_id,
                    hp_id=parent,
                    relation="hp_broader",
                    confidence=row.confidence,
                    split=row.split,
                    source="counterfactual_exact",
                    exact_hp_id=row.hp_id,
                    ancestor_distance=distance,
                    masked_hp_ids=";".join(sorted(masked_exact)),
                )

    synthetic: list[MappingExample] = []
    for child in sorted(hp_terms):
        parents, distance = hpo.nearest_available_ancestors(child, hp_terms)
        if not parents:
            continue
        branch = hpo.branch_key(child)
        split = stable_split(branch, seed)
        for parent in parents:
            synthetic.append(
                MappingExample(
                    mp_id=f"SYNMP:{child}",
                    hp_id=parent,
                    relation="hp_broader",
                    confidence=1.0,
                    split=split,
                    source="synthetic_hpo_cutout",
                    query_profile_source="hp",
                    exact_hp_id=child,
                    ancestor_distance=distance,
                    masked_hp_ids=child,
                )
            )
    if synthetic_limit > 0:
        synthetic = sorted(
            synthetic,
            key=lambda row: hashlib.sha256(f"{seed}|{row.mp_id}|{row.hp_id}".encode()).digest(),
        )[:synthetic_limit]
    for row in synthetic:
        examples[(row.mp_id, row.hp_id, row.source, row.query_profile_source)] = row
    return sorted(examples.values(), key=lambda row: (row.source, row.mp_id, row.hp_id))


def file_fingerprint(path: str | Path) -> dict[str, object]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_tsv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
