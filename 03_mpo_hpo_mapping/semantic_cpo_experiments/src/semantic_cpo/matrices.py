from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .config import resolve, result_root
from .data import profile
from .io import read_tsv
from .ontology import Ontology
from .profiles import Atom, aggregate


@dataclass
class WorkingProfiles:
    hpo_ids: list[str]
    mpo_ids: list[str]
    h_atoms: list[Atom]
    m_atoms: list[Atom]
    h_profiles: list[np.ndarray]
    m_profiles: list[np.ndarray]
    h_q: np.ndarray
    m_q: np.ndarray


def _atoms(path: Path) -> list[Atom]:
    return [
        Atom(
            atom_id=row["atom_id"],
            cl_id=row["cl_id"],
            go_id=row["go_id"],
            direction_state=row["direction_state"],
            direction_target=row["direction_target"],
            direction_property=row["direction_property"],
            relation_role=row["relation_role"],
        )
        for row in read_tsv(path)
    ]


def load_working_profiles(resources: dict[str, Any], hpo_ids: list[str]) -> WorkingProfiles:
    cache = result_root(resources) / "cache"
    values = np.load(cache / "self_profiles.npz", allow_pickle=False)
    all_h = list(values["h_term_ids"])
    all_m = list(values["m_term_ids"])
    h_lookup = {value: index for index, value in enumerate(all_h)}
    h_global_profiles = [profile(values, "h", h_lookup[value]) for value in hpo_ids]
    m_global_profiles = [profile(values, "m", index) for index in range(len(all_m))]
    used_h = sorted({int(value) for items in h_global_profiles for value in items})
    used_m = sorted({int(value) for items in m_global_profiles for value in items})
    h_reindex = {value: index for index, value in enumerate(used_h)}
    m_reindex = {value: index for index, value in enumerate(used_m)}
    all_h_atoms = _atoms(cache / "hpo_atoms.tsv")
    all_m_atoms = _atoms(cache / "mpo_atoms.tsv")
    return WorkingProfiles(
        hpo_ids=hpo_ids,
        mpo_ids=all_m,
        h_atoms=[all_h_atoms[index] for index in used_h],
        m_atoms=[all_m_atoms[index] for index in used_m],
        h_profiles=[np.asarray([h_reindex[int(value)] for value in items], dtype=np.int32) for items in h_global_profiles],
        m_profiles=[np.asarray([m_reindex[int(value)] for value in items], dtype=np.int32) for items in m_global_profiles],
        h_q=np.asarray([values["h_q_availability"][h_lookup[value]] for value in hpo_ids], dtype=np.float32),
        m_q=np.asarray(values["m_q_availability"], dtype=np.float32),
    )


def load_ontologies(resources: dict[str, Any]) -> tuple[Ontology, Ontology]:
    cl = Ontology.from_obo(resolve(resources, resources["ontologies"]["cl"]), "CL:0000000")
    go = Ontology.from_obo(
        resolve(resources, resources["ontologies"]["go"]),
        "GO:0008150",
        "biological_process",
    )
    return cl, go


def _node_matrix(
    ontology: Ontology, method: str, left: Sequence[str], right: Sequence[str]
) -> np.ndarray:
    unique_left = sorted(set(left))
    unique_right = sorted(set(right))
    left_index = {value: index for index, value in enumerate(unique_left)}
    right_index = {value: index for index, value in enumerate(unique_right)}
    values = np.empty((len(unique_left), len(unique_right)), dtype=np.float32)
    for i, left_id in enumerate(unique_left):
        for j, right_id in enumerate(unique_right):
            values[i, j] = ontology.similarity(method, left_id, right_id)
    return values[
        np.asarray([left_index[value] for value in left])[:, None],
        np.asarray([right_index[value] for value in right])[None, :],
    ]


def direction_mask(h_atoms: Sequence[Atom], m_atoms: Sequence[Atom]) -> np.ndarray:
    output = np.ones((len(h_atoms), len(m_atoms)), dtype=np.float32)
    opposite = {
        "increased": "decreased",
        "decreased": "increased",
        "present": "absent",
        "absent": "present",
    }
    m_groups: dict[tuple[str, ...], list[int]] = {}
    for index, atom in enumerate(m_atoms):
        key = (
            atom.cl_id,
            atom.go_id,
            atom.direction_target,
            atom.direction_property,
            atom.direction_state,
        )
        m_groups.setdefault(key, []).append(index)
    for i, atom in enumerate(h_atoms):
        if atom.direction_state not in opposite or not atom.direction_target:
            continue
        key = (
            atom.cl_id,
            atom.go_id,
            atom.direction_target,
            atom.direction_property,
            opposite[atom.direction_state],
        )
        output[i, m_groups.get(key, [])] = 0.0
    return output


def node_kernel(
    resources: dict[str, Any], profiles: WorkingProfiles, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cl, go = load_ontologies(resources)
    h_cl = [atom.cl_id for atom in profiles.h_atoms]
    m_cl = [atom.cl_id for atom in profiles.m_atoms]
    h_go = [atom.go_id for atom in profiles.h_atoms]
    m_go = [atom.go_id for atom in profiles.m_atoms]
    cl_values = _node_matrix(cl, method, h_cl, m_cl)
    go_values = _node_matrix(go, method, h_go, m_go)
    shape = cl_values * go_values
    return shape * direction_mask(profiles.h_atoms, profiles.m_atoms), cl_values, go_values


def text_kernels(
    resources: dict[str, Any], profiles: WorkingProfiles, apply_direction: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(result_root(resources) / "cache" / "semantic_embeddings.npz", allow_pickle=False)
    cl_index = {value: index for index, value in enumerate(values["cl_ids"])}
    go_index = {value: index for index, value in enumerate(values["go_ids"])}
    cpo_index = {value: index for index, value in enumerate(values["cpo_ids"])}
    h_cl = values["cl_vectors"][[cl_index[atom.cl_id] for atom in profiles.h_atoms]]
    m_cl = values["cl_vectors"][[cl_index[atom.cl_id] for atom in profiles.m_atoms]]
    h_go = values["go_vectors"][[go_index[atom.go_id] for atom in profiles.h_atoms]]
    m_go = values["go_vectors"][[go_index[atom.go_id] for atom in profiles.m_atoms]]
    h_cpo = values["cpo_vectors"][[cpo_index[f"{atom.cl_id}__{atom.go_id}"] for atom in profiles.h_atoms]]
    m_cpo = values["cpo_vectors"][[cpo_index[f"{atom.cl_id}__{atom.go_id}"] for atom in profiles.m_atoms]]
    component = np.clip(h_cl @ m_cl.T, 0.0, 1.0) * np.clip(h_go @ m_go.T, 0.0, 1.0)
    document = np.clip(h_cpo @ m_cpo.T, 0.0, 1.0)
    direction = (
        direction_mask(profiles.h_atoms, profiles.m_atoms)
        if apply_direction
        else np.ones((len(profiles.h_atoms), len(profiles.m_atoms)), dtype=np.float32)
    )
    return (component * direction).astype(np.float32), (document * direction).astype(np.float32)


def atom_kernel(method: str, ontology_shape: np.ndarray, component_text: np.ndarray | None = None, cpo_text: np.ndarray | None = None) -> np.ndarray:
    if method == "A0":
        return ontology_shape
    if method == "A1":
        return np.sqrt(ontology_shape).astype(np.float32)
    if method == "A2" and component_text is not None:
        return component_text
    if method == "A3" and cpo_text is not None:
        return cpo_text
    if method == "A4" and cpo_text is not None:
        return np.sqrt(ontology_shape * cpo_text).astype(np.float32)
    if method == "A5" and cpo_text is not None:
        return ((ontology_shape + cpo_text) / 2.0).astype(np.float32)
    raise ValueError(f"missing kernel input for {method}")


def _incidence(
    profiles: list[np.ndarray], count: int, atom_weights: np.ndarray | None = None
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row, items in enumerate(profiles):
        if not len(items):
            continue
        weights = (
            np.ones(len(items), dtype=np.float64)
            if atom_weights is None
            else np.asarray(atom_weights[items], dtype=np.float64)
        )
        if weights.sum() <= 0:
            continue
        weights /= weights.sum()
        rows.extend([row] * len(items))
        columns.extend(map(int, items))
        data.extend(map(float, weights))
    return sparse.csr_matrix((data, (rows, columns)), shape=(len(profiles), count))


def _max_to_profiles(kernel: np.ndarray, profiles: list[np.ndarray]) -> np.ndarray:
    active = np.asarray([i for i, items in enumerate(profiles) if len(items)], dtype=np.int32)
    output = np.zeros((kernel.shape[0], len(profiles)), dtype=np.float32)
    if not len(active):
        return output
    lengths = np.asarray([len(profiles[i]) for i in active], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    flat = np.asarray([item for i in active for item in profiles[i]], dtype=np.int32)
    output[:, active] = np.maximum.reduceat(kernel[:, flat], offsets[:-1], axis=1)
    return output


def atom_ic_weights(resources: dict[str, Any], atoms: Sequence[Atom]) -> np.ndarray:
    cl, go = load_ontologies(resources)
    values = []
    for atom in atoms:
        cl_ic = cl.ic.get(atom.cl_id, 0.0) / cl.log_n if cl.log_n else 0.0
        go_ic = go.ic.get(atom.go_id, 0.0) / go.log_n if go.log_n else 0.0
        values.append(math.sqrt(cl_ic * go_ic))
    return np.asarray(values, dtype=np.float64)


def bma_surface(
    resources: dict[str, Any],
    profiles: WorkingProfiles,
    kernel: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    h_weights = atom_ic_weights(resources, profiles.h_atoms) if weighted else None
    m_weights = atom_ic_weights(resources, profiles.m_atoms) if weighted else None
    h_incidence = _incidence(profiles.h_profiles, len(profiles.h_atoms), h_weights)
    m_incidence = _incidence(profiles.m_profiles, len(profiles.m_atoms), m_weights)
    forward_best = _max_to_profiles(kernel, profiles.m_profiles)
    reverse_best = _max_to_profiles(kernel.T, profiles.h_profiles)
    forward = np.asarray(h_incidence @ forward_best, dtype=np.float32)
    reverse = np.asarray((m_incidence @ reverse_best).T, dtype=np.float32)
    result = (forward + reverse) / 2.0
    h_active = np.asarray([len(items) > 0 for items in profiles.h_profiles])
    m_active = np.asarray([len(items) > 0 for items in profiles.m_profiles])
    result[~(h_active[:, None] & m_active[None, :])] = np.nan
    return result


def solver_surface(
    profiles: WorkingProfiles,
    kernel: np.ndarray,
    method: str,
    resources: dict[str, Any],
) -> np.ndarray:
    result = np.full((len(profiles.h_profiles), len(profiles.m_profiles)), np.nan, dtype=np.float32)
    h_weights = atom_ic_weights(resources, profiles.h_atoms)
    m_weights = atom_ic_weights(resources, profiles.m_atoms)
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for h_index, h_profile in enumerate(profiles.h_profiles):
        if not len(h_profile):
            continue
        for m_index, m_profile in enumerate(profiles.m_profiles):
            if not len(m_profile):
                continue
            key = (tuple(map(int, h_profile)), tuple(map(int, m_profile)))
            value = cache.get(key)
            if value is None:
                pair = kernel[np.ix_(h_profile, m_profile)]
                value = aggregate(
                    method,
                    pair,
                    [profiles.h_atoms[i].atom_id for i in h_profile],
                    [profiles.m_atoms[i].atom_id for i in m_profile],
                    h_weights[h_profile],
                    m_weights[m_profile],
                )
                cache[key] = value
            result[h_index, m_index] = value
    return result


def q_matrix(profiles: WorkingProfiles) -> np.ndarray:
    return np.sqrt(profiles.h_q[:, None] * profiles.m_q[None, :]).astype(np.float32)


def semantic_surface(resources: dict[str, Any], hpo_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.load(result_root(resources) / "cache" / "semantic_embeddings.npz", allow_pickle=False)
    h_index = {value: index for index, value in enumerate(values["hpo_ids"])}
    h_vectors = values["hpo_vectors"][[h_index[value] for value in hpo_ids]]
    surface = np.clip(h_vectors @ values["mpo_vectors"].T, 0.0, 1.0).astype(np.float32)
    return surface, list(values["mpo_ids"])
