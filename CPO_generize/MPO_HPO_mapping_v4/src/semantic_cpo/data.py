from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve, result_root
from .io import file_record, read_tsv, stable_digest, write_json, write_tsv


ATOM_FIELDS = (
    "atom_id",
    "cl_id",
    "go_id",
    "direction_state",
    "direction_target",
    "direction_property",
    "relation_role",
)


def _atom_key(row: Any, raw: bool = False) -> tuple[str, ...]:
    suffix = "_raw" if raw else ""
    return (
        str(getattr(row, f"cl_id{suffix}") or ""),
        str(getattr(row, f"go_id{suffix}") or ""),
        str(row.direction_state or "unknown"),
        str(row.direction_target or ""),
        str(row.direction_property or ""),
        str(row.relation_role or ""),
    )


def _atom_id(key: tuple[str, ...]) -> str:
    return "atom_" + hashlib.sha256("\t".join(key).encode("utf-8")).hexdigest()[:24]


def _profile_arrays(
    side: str,
    term_ids: list[str],
    frame: pd.DataFrame,
    accepted_status: set[str],
    accepted_roles: set[str],
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], list[dict[str, object]]]:
    selected = frame[(frame["ontology_side"] == side) & (frame["origin"] == "self")]
    raw_by_term: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    atoms_by_term: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    provenance: dict[tuple[str, tuple[str, ...]], dict[str, set[str]]] = defaultdict(
        lambda: {"assertions": set(), "publications": set(), "source_families": set()}
    )
    for row in selected.itertuples(index=False):
        term_id = str(row.target_term_id)
        raw_by_term[term_id].add(_atom_key(row, raw=True))
        qualified = (
            bool(row.cl_id)
            and bool(row.go_id)
            and str(row.statement_status) in accepted_status
            and str(row.relation_role) in accepted_roles
            and bool(row.evidence_sentence)
            and bool(row.publication_ids)
        )
        if not qualified:
            continue
        key = _atom_key(row)
        atoms_by_term[term_id].add(key)
        item = provenance[(term_id, key)]
        item["assertions"].add(str(row.assertion_id))
        item["publications"].update(
            value for value in str(row.publication_ids).split("|") if value
        )
        family = str(row.source_assertion_id or row.source_record_id or row.assertion_id)
        item["source_families"].add(family)

    atom_keys = sorted({key for values in atoms_by_term.values() for key in values})
    atoms = [
        dict(zip(ATOM_FIELDS, (_atom_id(key), *key)))
        for key in atom_keys
    ]
    atom_index = {tuple(row[field] for field in ATOM_FIELDS[1:]): i for i, row in enumerate(atoms)}
    offsets = np.zeros(len(term_ids) + 1, dtype=np.int64)
    flat: list[int] = []
    q = np.zeros(len(term_ids), dtype=np.float32)
    raw_counts = np.zeros(len(term_ids), dtype=np.int32)
    qualified_counts = np.zeros(len(term_ids), dtype=np.int32)
    term_index = {term_id: index for index, term_id in enumerate(term_ids)}
    for position, term_id in enumerate(term_ids):
        keys = sorted(atoms_by_term.get(term_id, set()))
        flat.extend(atom_index[key] for key in keys)
        offsets[position + 1] = len(flat)
        raw_counts[position] = len(raw_by_term.get(term_id, set()))
        qualified_counts[position] = len(keys)
        q[position] = len(keys) / raw_counts[position] if raw_counts[position] else 0.0
    provenance_rows: list[dict[str, object]] = []
    for (term_id, key), item in sorted(provenance.items()):
        provenance_rows.append(
            {
                "ontology_side": side,
                "term_id": term_id,
                "atom_id": _atom_id(key),
                "assertion_count": len(item["assertions"]),
                "publication_count": len(item["publications"]),
                "source_family_count": len(item["source_families"]),
                "assertion_ids": "|".join(sorted(item["assertions"])),
                "publication_ids": "|".join(sorted(item["publications"])),
                "source_family_ids": "|".join(sorted(item["source_families"])),
            }
        )
    arrays = {
        "term_ids": np.asarray(term_ids),
        "offsets": offsets,
        "atom_indices": np.asarray(flat, dtype=np.int32),
        "q_availability": q,
        "raw_counts": raw_counts,
        "qualified_counts": qualified_counts,
    }
    return atoms, arrays, provenance_rows


def build_profile_cache(resources: dict[str, Any], registry: dict[str, Any]) -> dict[str, object]:
    root = result_root(resources)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    h_rows = read_tsv(resolve(resources, resources["mission4"]["hpo_index"]))
    m_rows = read_tsv(resolve(resources, resources["mission4"]["mpo_index"]))
    h_ids = [row["term_id"] for row in h_rows]
    m_ids = [row["term_id"] for row in m_rows]
    assertions_path = resolve(resources, resources["mission4"]["assertions"])
    frame = pd.read_parquet(assertions_path).fillna("")
    accepted_status = set(registry["quality"]["accepted_statement_status"])
    accepted_roles = set(registry["quality"]["accepted_relation_roles"])
    h_atoms, h_arrays, h_provenance = _profile_arrays(
        "HPO", h_ids, frame, accepted_status, accepted_roles
    )
    m_atoms, m_arrays, m_provenance = _profile_arrays(
        "MPO", m_ids, frame, accepted_status, accepted_roles
    )
    np.savez_compressed(
        cache / "self_profiles.npz",
        **{f"h_{key}": value for key, value in h_arrays.items()},
        **{f"m_{key}": value for key, value in m_arrays.items()},
    )
    write_tsv(cache / "hpo_atoms.tsv", h_atoms, ATOM_FIELDS)
    write_tsv(cache / "mpo_atoms.tsv", m_atoms, ATOM_FIELDS)
    provenance_rows = [*h_provenance, *m_provenance]
    write_tsv(
        cache / "atom_provenance.tsv",
        provenance_rows,
        (
            "ontology_side",
            "term_id",
            "atom_id",
            "assertion_count",
            "publication_count",
            "source_family_count",
            "assertion_ids",
            "publication_ids",
            "source_family_ids",
        ),
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "view": "V_self",
        "quality_id": registry["quality"]["id"],
        "quality_interpretation": registry["quality"]["interpretation"],
        "input": file_record(assertions_path),
        "counts": {
            "hpo_terms": len(h_ids),
            "mpo_terms": len(m_ids),
            "hpo_terms_with_qualified_atoms": int(np.count_nonzero(h_arrays["qualified_counts"])),
            "mpo_terms_with_qualified_atoms": int(np.count_nonzero(m_arrays["qualified_counts"])),
            "hpo_unique_atoms": len(h_atoms),
            "mpo_unique_atoms": len(m_atoms),
            "hpo_profile_memberships": int(len(h_arrays["atom_indices"])),
            "mpo_profile_memberships": int(len(m_arrays["atom_indices"])),
            "hpo_raw_deduplicated_records": int(h_arrays["raw_counts"].sum()),
            "mpo_raw_deduplicated_records": int(m_arrays["raw_counts"].sum()),
            "hpo_qualified_deduplicated_records": int(h_arrays["qualified_counts"].sum()),
            "mpo_qualified_deduplicated_records": int(m_arrays["qualified_counts"].sum()),
        },
        "outputs": {
            name: file_record(cache / name)
            for name in ("self_profiles.npz", "hpo_atoms.tsv", "mpo_atoms.tsv", "atom_provenance.tsv")
        },
    }
    write_json(cache / "profile_manifest.json", manifest)
    return manifest


def profile(values: np.lib.npyio.NpzFile, side: str, term_index: int) -> np.ndarray:
    offsets = values[f"{side}_offsets"]
    flat = values[f"{side}_atom_indices"]
    return flat[offsets[term_index] : offsets[term_index + 1]]

