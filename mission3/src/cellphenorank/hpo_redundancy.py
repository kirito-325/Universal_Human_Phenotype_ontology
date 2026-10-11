from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .schemas import DiseaseTerm


class HpoOntology:
    def __init__(self, path: str | Path) -> None:
        self.parents: dict[str, set[str]] = {}
        self.names: dict[str, str] = {}
        self.obsolete: set[str] = set()
        current = ""
        with Path(path).open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line == "[Term]":
                    current = ""
                elif line.startswith("id: "):
                    current = line[4:].strip()
                    self.parents.setdefault(current, set())
                elif current and line.startswith("name: "):
                    self.names[current] = line[6:].strip()
                elif current and line.startswith("is_a: "):
                    self.parents[current].add(line[6:].split(" ! ", 1)[0].strip())
                elif current and line == "is_obsolete: true":
                    self.obsolete.add(current)
        self._ancestor_cache: dict[str, dict[str, int]] = {}

    @property
    def valid_ids(self) -> set[str]:
        return set(self.parents).difference(self.obsolete)

    def ancestors(self, term_id: str) -> dict[str, int]:
        if term_id in self._ancestor_cache:
            return self._ancestor_cache[term_id]
        distances: dict[str, int] = {}
        frontier = [(term_id, 0)]
        while frontier:
            current, distance = frontier.pop(0)
            for parent in self.parents.get(current, set()):
                if parent not in distances or distance + 1 < distances[parent]:
                    distances[parent] = distance + 1
                    frontier.append((parent, distance + 1))
        self._ancestor_cache[term_id] = distances
        return distances


def reduce_hpo_profiles(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
    ontology: HpoOntology,
) -> tuple[dict[str, list[DiseaseTerm]], list[dict[str, object]]]:
    reduced: dict[str, list[DiseaseTerm]] = {}
    audit: list[dict[str, object]] = []
    valid_ids = ontology.valid_ids
    for disease_id, terms in sorted(diseases.items()):
        first_by_id: dict[str, DiseaseTerm] = {}
        duplicate_ids: set[str] = set()
        for term in terms:
            if term.hpo_id in first_by_id:
                duplicate_ids.add(term.hpo_id)
            else:
                first_by_id[term.hpo_id] = term
        present = set(first_by_id)
        retained: list[DiseaseTerm] = []
        for hpo_id, term in first_by_id.items():
            if hpo_id not in valid_ids:
                audit.append(
                    {
                        "disease_id": disease_id,
                        "original_hpo_id": hpo_id,
                        "retained_hpo_id": "",
                        "status": "invalid_hpo",
                        "reason": "not_active_in_hpo_ontology",
                        "ancestor_distance": "",
                    }
                )
                continue
            descendant_matches = [
                (candidate, ontology.ancestors(candidate)[hpo_id])
                for candidate in present
                if candidate != hpo_id and hpo_id in ontology.ancestors(candidate)
            ]
            if descendant_matches:
                replacement_id, distance = sorted(
                    descendant_matches, key=lambda item: (item[1], item[0])
                )[0]
                audit.append(
                    {
                        "disease_id": disease_id,
                        "original_hpo_id": hpo_id,
                        "retained_hpo_id": replacement_id,
                        "status": "ancestor_removed",
                        "reason": "more_specific_descendant_present",
                        "ancestor_distance": distance,
                    }
                )
                continue
            retained.append(replace(term, weight=1.0))
            audit.append(
                {
                    "disease_id": disease_id,
                    "original_hpo_id": hpo_id,
                    "retained_hpo_id": hpo_id,
                    "status": "retained",
                    "reason": "nonredundant_active_term",
                    "ancestor_distance": 0,
                }
            )
        for hpo_id in sorted(duplicate_ids):
            audit.append(
                {
                    "disease_id": disease_id,
                    "original_hpo_id": hpo_id,
                    "retained_hpo_id": hpo_id if hpo_id in {term.hpo_id for term in retained} else "",
                    "status": "duplicate_removed",
                    "reason": "duplicate_input_assertion",
                    "ancestor_distance": 0,
                }
            )
        reduced[disease_id] = sorted(retained, key=lambda term: term.hpo_id)
    return reduced, audit


HPO_REDUCTION_FIELDS = [
    "disease_id",
    "original_hpo_id",
    "retained_hpo_id",
    "status",
    "reason",
    "ancestor_distance",
]

NONREDUNDANT_HPO_FIELDS = [
    "disease_id",
    "disease_name",
    "hpo_id",
    "hpo_name",
    "weight",
    "is_core",
    "gene_symbol",
    "direction",
]


def disease_terms_as_rows(
    diseases: Mapping[str, Sequence[DiseaseTerm]],
) -> list[dict[str, object]]:
    return [
        {
            "disease_id": term.disease_id,
            "disease_name": term.disease_name,
            "hpo_id": term.hpo_id,
            "hpo_name": term.hpo_name,
            "weight": term.weight,
            "is_core": term.is_core,
            "gene_symbol": term.gene_symbol,
            "direction": term.direction,
        }
        for disease_id in sorted(diseases)
        for term in diseases[disease_id]
    ]
