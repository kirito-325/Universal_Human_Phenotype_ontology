from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cpo_mapping.candidates import ParquetRecordWriter, _schema
from cpo_mapping.io import read_tsv, write_json

from .config import output_root, resource_path
from .mapping import MappingContext, _direction_code
from .prepare import stable_id


TERM_FIELDS = (
    "hpo_id",
    "mpo_id",
    "score_type",
    "status",
    "term_score",
    "hpo_structure_id",
    "mpo_structure_id",
    "hpo_cl_id",
    "mpo_cl_id",
    "hpo_go_id",
    "mpo_go_id",
    "cl_similarity",
    "go_similarity",
    "cl_mica_id",
    "cl_mica_ic",
    "go_mica_id",
    "go_mica_ic",
    "cpo_pair_score",
    "direction_status",
    "n_best_ties",
    "hpo_origin",
    "mpo_origin",
    "hpo_source_leaf_id",
    "mpo_source_leaf_id",
    "hpo_inheritance_path",
    "mpo_inheritance_path",
    "hpo_assertion_id",
    "mpo_assertion_id",
    "hpo_publication_ids",
    "mpo_publication_ids",
    "hpo_evidence_sentence",
    "mpo_evidence_sentence",
)
TERM_SCHEMA = _schema(
    TERM_FIELDS,
    {
        "term_score",
        "cl_similarity",
        "go_similarity",
        "cl_mica_ic",
        "go_mica_ic",
        "cpo_pair_score",
    },
    {"n_best_ties"},
)

GENE_CHAIN_FIELDS = (
    "ranking_view",
    "disease_id",
    "human_gene_id",
    "human_gene_symbol",
    "human_gene_score",
    "human_gene_rank",
    "human_gene_rank_min",
    "human_gene_rank_max",
    "human_gene_tie_count",
    "mouse_gene_id",
    "model_profile_id",
    "model_score_evidence",
    "model_status",
    "model_genotype_id",
    "model_allelic_composition",
    "model_allele_ids",
    "model_strain_background",
    "model_mpo_reference_ids",
    "hpo_id",
    "mpo_id",
    "term_shape",
    "term_direction_support",
    "term_direction_conflict",
    "hpo_structure_id",
    "mpo_structure_id",
    "cl_pair",
    "go_pair",
    "hpo_source_leaf_id",
    "mpo_source_leaf_id",
    "hpo_origin",
    "mpo_origin",
    "hpo_inheritance_path",
    "mpo_inheritance_path",
    "hpo_assertion_id",
    "mpo_assertion_id",
    "hpo_publication_ids",
    "mpo_publication_ids",
    "hpo_evidence_sentence",
    "mpo_evidence_sentence",
    "orthology_group_id",
    "orthology_type",
    "orthology_provider",
    "evidence_attribution",
    "requires_descendant_interpretation",
)
GENE_CHAIN_SCHEMA = _schema(
    GENE_CHAIN_FIELDS,
    {
        "human_gene_score",
        "model_score_evidence",
        "term_shape",
        "term_direction_support",
        "term_direction_conflict",
    },
    {
        "human_gene_rank",
        "human_gene_rank_min",
        "human_gene_rank_max",
        "human_gene_tie_count",
    },
)

AUXILIARY_FIELDS = (
    "query_side",
    "hpo_id",
    "mpo_id",
    "dimension",
    "rank",
    "term_score",
    "hpo_component_id",
    "mpo_component_id",
    "component_similarity",
    "mica_id",
    "mica_ic",
    "n_best_ties",
    "hpo_assertion_ids",
    "mpo_assertion_ids",
    "hpo_source_leaf_ids",
    "mpo_source_leaf_ids",
    "status",
)
AUXILIARY_SCHEMA = _schema(
    AUXILIARY_FIELDS,
    {"term_score", "component_similarity", "mica_ic"},
    {"rank", "n_best_ties"},
)

DESCENDANT_FIELDS = (
    "request_id",
    "ontology_side",
    "ancestor_term_id",
    "descendant_leaf_id",
    "structure_id",
    "source_assertion_id",
    "inheritance_distance",
    "canonical_path",
    "status",
)
DESCENDANT_SCHEMA = _schema(DESCENDANT_FIELDS, set(), set())


def _candidate_pairs(root: Path) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for path in (root / "mapping/hpo_to_mpo.parquet", root / "mapping/mpo_to_hpo.parquet"):
        frame = pd.read_parquet(path, columns=["hpo_id", "mpo_id"])
        result.update(zip(frame["hpo_id"].astype(str), frame["mpo_id"].astype(str)))
    return result


def _best(values: np.ndarray, h_profile: list[dict], m_profile: list[dict]):
    maximum = float(values.max())
    if maximum <= 0:
        return None
    positions = np.argwhere(values == maximum)
    choices = sorted(
        ((int(i), int(j)) for i, j in positions),
        key=lambda pair: (
            str(h_profile[pair[0]]["structure_id"]),
            str(m_profile[pair[1]]["structure_id"]),
        ),
    )
    return choices[0][0], choices[0][1], maximum, len(choices)


def _source_indexes(root: Path, config: dict[str, Any]):
    assertions = pd.read_parquet(root / "normalized/assertions.parquet")
    self_assertions = assertions[assertions["origin"] == "self"]
    assertion_index = {
        str(row["assertion_id"]): row for row in self_assertions.to_dict("records")
    }
    provenance = pd.read_parquet(root / "profiles/structure_provenance.parquet")
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in provenance[provenance["profile_view"] == "final"].to_dict("records"):
        grouped[(str(row["ontology_side"]), str(row["term_id"]), str(row["structure_id"]))].append(row)

    source_relation_to_assertion: dict[tuple[str, str], str] = {}
    for row in self_assertions.itertuples(index=False):
        if row.ontology_side == "MPO":
            source_relation_to_assertion[("MPO", str(row.source_record_id))] = str(row.assertion_id)
    _, h_provenance = read_tsv(resource_path(config, "hpo_inheritance_provenance"))
    h_self_by_row = {
        int(row.source_row): str(row.assertion_id)
        for row in self_assertions.itertuples(index=False)
        if row.ontology_side == "HPO"
    }
    for row in h_provenance:
        source_row = int(row["source_logical_row"])
        if source_row in h_self_by_row:
            source_relation_to_assertion[("HPO", row["source_relation_id"])] = (
                h_self_by_row[source_row]
            )
    return assertion_index, grouped, source_relation_to_assertion


def _select_provenance(
    side: str,
    term_id: str,
    structure_id: str,
    grouped: dict,
    source_relation_to_assertion: dict,
) -> tuple[dict[str, object], dict[str, object]]:
    rows = sorted(
        grouped.get((side, term_id, structure_id), ()),
        key=lambda row: (
            0 if row["origin"] == "self" else 1,
            str(row.get("source_leaf_id", "")),
            str(row["assertion_id"]),
        ),
    )
    if not rows:
        return {}, {}
    selected = rows[0]
    assertion_id = str(selected["assertion_id"])
    if selected["origin"] == "inherited":
        assertion_id = source_relation_to_assertion.get(
            (side, str(selected.get("source_relation_id", ""))), ""
        )
    return selected, {"resolved_assertion_id": assertion_id}


def run_term_explanations(config: dict[str, Any]) -> dict[str, int]:
    context = MappingContext(config)
    root = context.root
    assertion_index, provenance, relation_lookup = _source_indexes(root, config)
    writer = ParquetRecordWriter(root / "explanations/term_matches.parquet", TERM_SCHEMA)
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for hpo_id, mpo_id in sorted(_candidate_pairs(root)):
            h_index = context.hpo_index[hpo_id]
            m_index = context.mpo_index[mpo_id]
            h_profile = context.h_profiles[h_index]
            m_profile = context.m_profiles[m_index]
            h_components = np.asarray(
                [context.h_structure_index[str(value["structure_id"])] for value in h_profile]
            )
            m_components = np.asarray(
                [context.m_structure_index[str(value["structure_id"])] for value in m_profile]
            )
            shape = context.similarity[np.ix_(h_components, m_components)]
            directions = np.asarray(
                [[_direction_code(left, right) for right in m_profile] for left in h_profile]
            )
            matrices = {
                "shape": shape,
                "direction_support": shape * (directions == 1),
                "direction_conflict": shape * (directions == -1),
            }
            term_scores = {
                "shape": float(handle["final/shape"][h_index, m_index]),
                "direction_support": float(handle["final/direction_support"][h_index, m_index]),
                "direction_conflict": float(handle["final/direction_conflict"][h_index, m_index]),
            }
            for score_type, values in matrices.items():
                best = _best(values, h_profile, m_profile)
                if best is None:
                    writer.add(empty_term_row(hpo_id, mpo_id, score_type, term_scores[score_type]))
                    continue
                h_pos, m_pos, pair_score, ties = best
                h_node, m_node = h_profile[h_pos], m_profile[m_pos]
                h_prov, h_resolved = _select_provenance(
                    "HPO", hpo_id, str(h_node["structure_id"]), provenance, relation_lookup
                )
                m_prov, m_resolved = _select_provenance(
                    "MPO", mpo_id, str(m_node["structure_id"]), provenance, relation_lookup
                )
                h_assertion_id = str(h_resolved.get("resolved_assertion_id", ""))
                m_assertion_id = str(m_resolved.get("resolved_assertion_id", ""))
                h_assertion = assertion_index.get(h_assertion_id, {})
                m_assertion = assertion_index.get(m_assertion_id, {})
                cl_mica, cl_ic, _ = context.data.cl.mica(str(h_node["cl_id"]), str(m_node["cl_id"]))
                go_mica, go_ic, _ = context.data.go.mica(str(h_node["go_id"]), str(m_node["go_id"]))
                writer.add(
                    {
                        "hpo_id": hpo_id,
                        "mpo_id": mpo_id,
                        "score_type": score_type,
                        "status": "OK",
                        "term_score": term_scores[score_type],
                        "hpo_structure_id": h_node["structure_id"],
                        "mpo_structure_id": m_node["structure_id"],
                        "hpo_cl_id": h_node["cl_id"],
                        "mpo_cl_id": m_node["cl_id"],
                        "hpo_go_id": h_node["go_id"],
                        "mpo_go_id": m_node["go_id"],
                        "cl_similarity": context.data.cl.lin(str(h_node["cl_id"]), str(m_node["cl_id"])),
                        "go_similarity": context.data.go.lin(str(h_node["go_id"]), str(m_node["go_id"])),
                        "cl_mica_id": cl_mica or "",
                        "cl_mica_ic": cl_ic,
                        "go_mica_id": go_mica or "",
                        "go_mica_ic": go_ic,
                        "cpo_pair_score": pair_score,
                        "direction_status": {1: "agree", -1: "conflict"}.get(int(directions[h_pos, m_pos]), "unknown_or_incomparable"),
                        "n_best_ties": ties,
                        "hpo_origin": h_prov.get("origin", ""),
                        "mpo_origin": m_prov.get("origin", ""),
                        "hpo_source_leaf_id": h_prov.get("source_leaf_id", ""),
                        "mpo_source_leaf_id": m_prov.get("source_leaf_id", ""),
                        "hpo_inheritance_path": h_prov.get("canonical_path", ""),
                        "mpo_inheritance_path": m_prov.get("canonical_path", ""),
                        "hpo_assertion_id": h_assertion_id,
                        "mpo_assertion_id": m_assertion_id,
                        "hpo_publication_ids": h_assertion.get("publication_ids", ""),
                        "mpo_publication_ids": m_assertion.get("publication_ids", ""),
                        "hpo_evidence_sentence": h_assertion.get("evidence_sentence", ""),
                        "mpo_evidence_sentence": m_assertion.get("evidence_sentence", ""),
                    }
                )
    count = writer.close()
    result = {"term_explanation_rows": count}
    write_json(root / "explanations/term_explanation_manifest.json", result)
    return result


def empty_term_row(hpo_id: str, mpo_id: str, score_type: str, score: float) -> dict[str, object]:
    return {
        "hpo_id": hpo_id,
        "mpo_id": mpo_id,
        "score_type": score_type,
        "status": "NO_POSITIVE_MATCH",
        "term_score": score,
        "hpo_structure_id": "",
        "mpo_structure_id": "",
        "hpo_cl_id": "",
        "mpo_cl_id": "",
        "hpo_go_id": "",
        "mpo_go_id": "",
        "cl_similarity": float("nan"),
        "go_similarity": float("nan"),
        "cl_mica_id": "",
        "cl_mica_ic": float("nan"),
        "go_mica_id": "",
        "go_mica_ic": float("nan"),
        "cpo_pair_score": 0.0,
        "direction_status": "unknown_or_incomparable",
        "n_best_ties": 0,
        "hpo_origin": "",
        "mpo_origin": "",
        "hpo_source_leaf_id": "",
        "mpo_source_leaf_id": "",
        "hpo_inheritance_path": "",
        "mpo_inheritance_path": "",
        "hpo_assertion_id": "",
        "mpo_assertion_id": "",
        "hpo_publication_ids": "",
        "mpo_publication_ids": "",
        "hpo_evidence_sentence": "",
        "mpo_evidence_sentence": "",
    }


def run_gene_evidence_chains(config: dict[str, Any]) -> dict[str, int]:
    context = MappingContext(config)
    root = context.root
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    _, mouse_gene_rows = read_tsv(root / "gene_prediction/mouse_gene_index.tsv")
    disease_index = {row["disease_id"]: int(row["index"]) for row in disease_rows}
    hpo_index = context.hpo_index
    mpo_index = context.mpo_index
    mouse_gene_index = {
        row["mouse_gene_id"]: int(row["index"]) for row in mouse_gene_rows
    }
    models_by_gene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(model_rows):
        if row["model_status"] == "single_gene_model":
            models_by_gene[row["perturbed_mouse_gene_ids"]].append(index)
    disease_profiles = pd.read_parquet(root / "disease_model/disease_profiles.parquet")
    model_profiles = pd.read_parquet(root / "disease_model/model_profiles.parquet")
    disease_terms: dict[str, list[str]] = defaultdict(list)
    model_terms: dict[str, list[str]] = defaultdict(list)
    for row in disease_profiles.itertuples(index=False):
        disease_terms[str(row.profile_id)].append(str(row.term_id))
    for row in model_profiles.itertuples(index=False):
        model_terms[str(row.profile_id)].append(str(row.term_id))
    model_term_references = {
        (str(profile_id), str(term_id)): "|".join(
            sorted(
                {
                    reference
                    for raw in frame["reference_ids"].astype(str)
                    for reference in raw.split("|")
                    if reference and reference != "nan"
                }
            )
        )
        for (profile_id, term_id), frame in model_profiles.groupby(
            ["profile_id", "term_id"], sort=False
        )
    }
    candidates = []
    for view, filename in (
        ("one_to_one", "human_1to1.parquet"),
        ("expanded", "human_expanded.parquet"),
    ):
        frame = pd.read_parquet(root / f"gene_prediction/{filename}")
        frame = frame[frame["rank"] <= 10].copy()
        frame["ranking_view"] = view
        candidates.append(frame)
    candidates_frame = pd.concat(candidates, ignore_index=True)
    by_disease = {
        disease: frame
        for disease, frame in candidates_frame.groupby("disease_id", sort=True)
    }
    assertion_index, provenance, relation_lookup = _source_indexes(root, config)
    writer = ParquetRecordWriter(
        root / "explanations/gene_evidence_chains.parquet", GENE_CHAIN_SCHEMA
    )
    with h5py.File(root / "gene_prediction/mouse_gene_scores.h5", "r") as mouse_handle:
        with h5py.File(
            root / "disease_model/scores/disease_model_scores.h5", "r"
        ) as model_handle:
            with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as term_handle:
                cached_start = -1
                mouse_score_block = np.empty((0, 0), dtype=np.float32)
                model_score_block = np.empty((0, 0), dtype=np.float32)
                term_dataset = term_handle["final/shape"]
                term_block_size = int(term_dataset.chunks[0])
                required_h_indices = sorted(
                    {
                        hpo_index[term]
                        for terms in disease_terms.values()
                        for term in terms
                        if context.h_profiles[hpo_index[term]]
                    }
                )
                required_h_lookup = {
                    index: position for position, index in enumerate(required_h_indices)
                }
                term_shape_cache = np.empty(
                    (len(required_h_indices), len(context.mpo_ids)), dtype=np.float32
                )
                cursor = 0
                for start in range(0, len(context.hpo_ids), term_block_size):
                    end = min(len(context.hpo_ids), start + term_block_size)
                    stop = cursor
                    while stop < len(required_h_indices) and required_h_indices[stop] < end:
                        stop += 1
                    if stop > cursor:
                        block = np.asarray(term_dataset[start:end, :])
                        for position in range(cursor, stop):
                            term_shape_cache[position] = block[
                                required_h_indices[position] - start
                            ]
                    cursor = stop
                nan_term_row = np.full(len(context.mpo_ids), np.nan, dtype=np.float32)
                term_evidence_cache: OrderedDict[
                    tuple[str, str], dict[str, object] | None
                ] = OrderedDict()

                def load_term_rows(indices: list[int]) -> np.ndarray:
                    return np.asarray(
                        [
                            term_shape_cache[required_h_lookup[index]]
                            if index in required_h_lookup
                            else nan_term_row
                            for index in indices
                        ]
                    )

                def calculate_direction(
                    h_profile: list[dict[str, object]],
                    m_profile: list[dict[str, object]],
                ) -> tuple[float, float]:
                    h_by_structure = {
                        str(value["structure_id"]): value for value in h_profile
                    }
                    m_by_structure = {
                        str(value["structure_id"]): value for value in m_profile
                    }
                    agree = 0
                    conflict = 0
                    for structure_id in h_by_structure.keys() & m_by_structure.keys():
                        code = _direction_code(
                            h_by_structure[structure_id], m_by_structure[structure_id]
                        )
                        agree += code == 1
                        conflict += code == -1
                    if agree + conflict == 0:
                        return float("nan"), float("nan")
                    weight = 0.5 * (1 / len(h_profile) + 1 / len(m_profile))
                    return agree * weight, conflict * weight

                def explain_term_pair(
                    hpo_id: str,
                    mpo_id: str,
                    h_term_index: int,
                    m_term_index: int,
                    term_shape: float,
                ) -> dict[str, object] | None:
                    key = (hpo_id, mpo_id)
                    if key in term_evidence_cache:
                        term_evidence_cache.move_to_end(key)
                        return term_evidence_cache[key]
                    h_profile = context.h_profiles[h_term_index]
                    m_profile = context.m_profiles[m_term_index]
                    h_components = np.asarray(
                        [
                            context.h_structure_index[str(value["structure_id"])]
                            for value in h_profile
                        ]
                    )
                    m_components = np.asarray(
                        [
                            context.m_structure_index[str(value["structure_id"])]
                            for value in m_profile
                        ]
                    )
                    cpo_values = context.similarity[np.ix_(h_components, m_components)]
                    best_cpo = _best(cpo_values, h_profile, m_profile)
                    if best_cpo is None:
                        result = None
                    else:
                        h_cpo, m_cpo, _cpo_score, _ties = best_cpo
                        h_node, m_node = h_profile[h_cpo], m_profile[m_cpo]
                        h_prov, h_resolved = _select_provenance(
                            "HPO",
                            hpo_id,
                            str(h_node["structure_id"]),
                            provenance,
                            relation_lookup,
                        )
                        m_prov, m_resolved = _select_provenance(
                            "MPO",
                            mpo_id,
                            str(m_node["structure_id"]),
                            provenance,
                            relation_lookup,
                        )
                        h_assertion_id = str(
                            h_resolved.get("resolved_assertion_id", "")
                        )
                        m_assertion_id = str(
                            m_resolved.get("resolved_assertion_id", "")
                        )
                        h_assertion = assertion_index.get(h_assertion_id, {})
                        m_assertion = assertion_index.get(m_assertion_id, {})
                        direction_support, direction_conflict = calculate_direction(
                            h_profile, m_profile
                        )
                        result = {
                            "hpo_id": hpo_id,
                            "mpo_id": mpo_id,
                            "term_shape": term_shape,
                            "term_direction_support": direction_support,
                            "term_direction_conflict": direction_conflict,
                            "hpo_structure_id": h_node["structure_id"],
                            "mpo_structure_id": m_node["structure_id"],
                            "cl_pair": f"{h_node['cl_id']}|{m_node['cl_id']}",
                            "go_pair": f"{h_node['go_id']}|{m_node['go_id']}",
                            "hpo_source_leaf_id": h_prov.get("source_leaf_id", ""),
                            "mpo_source_leaf_id": m_prov.get("source_leaf_id", ""),
                            "hpo_origin": h_prov.get("origin", ""),
                            "mpo_origin": m_prov.get("origin", ""),
                            "hpo_inheritance_path": h_prov.get("canonical_path", ""),
                            "mpo_inheritance_path": m_prov.get("canonical_path", ""),
                            "hpo_assertion_id": h_assertion_id,
                            "mpo_assertion_id": m_assertion_id,
                            "hpo_publication_ids": h_assertion.get("publication_ids", ""),
                            "mpo_publication_ids": m_assertion.get("publication_ids", ""),
                            "hpo_evidence_sentence": h_assertion.get("evidence_sentence", ""),
                            "mpo_evidence_sentence": m_assertion.get("evidence_sentence", ""),
                            "evidence_attribution": "model_phenotype_observed|mechanism_transferred_from_term_profile",
                            "requires_descendant_interpretation": str(
                                str(m_prov.get("origin", "")) == "inherited"
                            ).lower(),
                        }
                    term_evidence_cache[key] = result
                    if len(term_evidence_cache) > 100000:
                        term_evidence_cache.popitem(last=False)
                    return result

                for disease_id, frame in by_disease.items():
                    d_index = disease_index[disease_id]
                    block_start = (d_index // 32) * 32
                    if block_start != cached_start:
                        block_end = min(len(disease_rows), block_start + 32)
                        mouse_score_block = np.asarray(
                            mouse_handle["score"][block_start:block_end, :]
                        )
                        model_score_block = np.asarray(
                            model_handle["score_evidence"][block_start:block_end, :]
                        )
                        cached_start = block_start
                    mouse_scores = mouse_score_block[d_index - block_start]
                    model_scores = model_score_block[d_index - block_start]
                    h_terms = sorted(set(disease_terms[disease_id]))
                    h_indices = [hpo_index[term] for term in h_terms]
                    term_shape_rows = load_term_rows(h_indices)
                    chain_cache: dict[str, dict[str, object] | None] = {}
                    for candidate in frame.itertuples(index=False):
                        source_mouse = [
                            value
                            for value in str(candidate.source_mouse_gene_ids).split("|")
                            if value in mouse_gene_index
                        ]
                        if not source_mouse:
                            continue
                        finite_source_scores = {
                            value: float(mouse_scores[mouse_gene_index[value]])
                            for value in source_mouse
                            if np.isfinite(mouse_scores[mouse_gene_index[value]])
                        }
                        if not finite_source_scores:
                            continue
                        best_mouse_score = max(finite_source_scores.values())
                        best_mouse = min(
                            value
                            for value, score in finite_source_scores.items()
                            if score == best_mouse_score
                        )
                        if best_mouse not in chain_cache:
                            model_indices = models_by_gene[best_mouse]
                            gene_model_scores = model_scores[model_indices]
                            finite_models = np.isfinite(gene_model_scores)
                            if not finite_models.any():
                                chain_cache[best_mouse] = None
                            else:
                                best_model_score = float(np.nanmax(gene_model_scores))
                                best_model_indices = [
                                    index
                                    for index, value in zip(model_indices, gene_model_scores)
                                    if np.isfinite(value) and float(value) == best_model_score
                                ]
                                best_model_index = min(
                                    best_model_indices,
                                    key=lambda index: model_rows[index]["model_profile_id"],
                                )
                                model = model_rows[best_model_index]
                                m_terms = sorted(set(model_terms[model["model_profile_id"]]))
                                m_indices = [mpo_index[term] for term in m_terms]
                                term_pairs = term_shape_rows[:, m_indices]
                                if not np.isfinite(term_pairs).any():
                                    chain_cache[best_mouse] = None
                                else:
                                    maximum = float(np.nanmax(term_pairs))
                                    positions = np.argwhere(term_pairs == maximum)
                                    choices = sorted(
                                        ((int(i), int(j)) for i, j in positions),
                                        key=lambda pair: (h_terms[pair[0]], m_terms[pair[1]]),
                                    )
                                    h_position, m_position = choices[0]
                                    hpo_id = h_terms[h_position]
                                    mpo_id = m_terms[m_position]
                                    h_term_index = hpo_index[hpo_id]
                                    m_term_index = mpo_index[mpo_id]
                                    term_evidence = explain_term_pair(
                                        hpo_id,
                                        mpo_id,
                                        h_term_index,
                                        m_term_index,
                                        maximum,
                                    )
                                    if term_evidence is None:
                                        chain_cache[best_mouse] = None
                                    else:
                                        chain_cache[best_mouse] = {
                                            "model_profile_id": model["model_profile_id"],
                                            "model_score_evidence": best_model_score,
                                            "model_status": model["model_status"],
                                            "model_genotype_id": model["genotype_id"],
                                            "model_allelic_composition": model["allelic_composition"],
                                            "model_allele_ids": model["allele_ids"],
                                            "model_strain_background": model["strain_background"],
                                            "model_mpo_reference_ids": model_term_references.get(
                                                (model["model_profile_id"], mpo_id), ""
                                            ),
                                            **term_evidence,
                                        }
                        chain = chain_cache[best_mouse]
                        if chain is None:
                            continue
                        writer.add(
                            {
                                "ranking_view": str(candidate.ranking_view),
                                "disease_id": disease_id,
                                "human_gene_id": str(candidate.gene_id),
                                "human_gene_symbol": str(candidate.gene_symbol),
                                "human_gene_score": float(candidate.score),
                                "human_gene_rank": int(candidate.rank),
                                "human_gene_rank_min": int(candidate.rank_min),
                                "human_gene_rank_max": int(candidate.rank_max),
                                "human_gene_tie_count": int(candidate.tie_count),
                                "mouse_gene_id": best_mouse,
                                **chain,
                                "orthology_group_id": str(candidate.orthology_group_id),
                                "orthology_type": str(candidate.orthology_type),
                                "orthology_provider": "MGI_HOM_MouseHumanSequence",
                            }
                        )
    count = writer.close()
    result = {"gene_evidence_chain_rows": count}
    write_json(root / "explanations/gene_evidence_manifest.json", result)
    return result


def run_auxiliary_explanations(config: dict[str, Any]) -> dict[str, int]:
    context = MappingContext(config)
    root = context.root
    assertions = pd.read_parquet(root / "normalized/assertions.parquet")
    policy = pd.read_csv(root / "profiles/node_policy.tsv", sep="\t", dtype=str)
    strategies = {
        (str(row.ontology_side), str(row.term_id)): str(row.final_strategy)
        for row in policy.itertuples(index=False)
    }

    def accepted_origins(side: str, term_id: str) -> set[str]:
        strategy = strategies[(side, term_id)]
        return (
            {"self"}
            if strategy == "self"
            else {"inherited"}
            if strategy == "inherited"
            else {"self", "inherited"}
        )

    source: dict[tuple[str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"assertions": set(), "leaves": set()}
    )
    usable = assertions[
        (assertions["statement_status"] == "asserted")
        & ~assertions["qc_flags"].str.contains("noninformative_", regex=True)
    ]
    for row in usable.itertuples(index=False):
        side = str(row.ontology_side)
        term_id = str(row.target_term_id)
        if (side, term_id) not in strategies:
            continue
        if str(row.origin) not in accepted_origins(side, term_id):
            continue
        for dimension, component in (("CL", str(row.cl_id)), ("GO", str(row.go_id))):
            if not component or component in {"CL:0000000", "GO:0008150"}:
                continue
            item = source[(side, term_id, dimension, component)]
            item["assertions"].add(str(row.assertion_id))
            if row.source_leaf_id:
                item["leaves"].add(str(row.source_leaf_id))

    projection = context.data.projections
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in projection.itertuples(index=False):
        grouped[(str(row.ontology_side), str(row.term_id), str(row.dimension))].append(
            str(row.component_id)
        )
    for key in grouped:
        grouped[key] = sorted(set(grouped[key]))

    cache: dict[tuple[str, str, str], tuple[str, str, float, str, float, int] | None] = {}
    writer = ParquetRecordWriter(
        root / "explanations/auxiliary_matches.parquet", AUXILIARY_SCHEMA
    )
    candidates = pd.read_parquet(root / "mapping/auxiliary_topk.parquet")
    for row in candidates.itertuples(index=False):
        key = (str(row.hpo_id), str(row.mpo_id), str(row.dimension))
        best = cache.get(key)
        if key not in cache:
            left = grouped.get(("HPO", key[0], key[2]), [])
            right = grouped.get(("MPO", key[1], key[2]), [])
            ontology = context.data.cl if key[2] == "CL" else context.data.go
            scored = [
                (ontology.lin(h, m), h, m)
                for h in left
                for m in right
            ]
            positive = [value for value in scored if value[0] > 0]
            if not positive:
                best = None
            else:
                maximum = max(value[0] for value in positive)
                ties = sorted((h, m) for score, h, m in positive if score == maximum)
                h_component, m_component = ties[0]
                mica_id, mica_ic, _ = ontology.mica(h_component, m_component)
                best = (
                    h_component,
                    m_component,
                    float(maximum),
                    mica_id or "",
                    float(mica_ic),
                    len(ties),
                )
            cache[key] = best
        if best is None:
            values = {
                "hpo_component_id": "",
                "mpo_component_id": "",
                "component_similarity": float("nan"),
                "mica_id": "",
                "mica_ic": float("nan"),
                "n_best_ties": 0,
                "hpo_assertion_ids": "",
                "mpo_assertion_ids": "",
                "hpo_source_leaf_ids": "",
                "mpo_source_leaf_ids": "",
                "status": "NO_POSITIVE_MATCH",
            }
        else:
            h_component, m_component, similarity, mica_id, mica_ic, ties = best
            h_source = source[("HPO", key[0], key[2], h_component)]
            m_source = source[("MPO", key[1], key[2], m_component)]
            values = {
                "hpo_component_id": h_component,
                "mpo_component_id": m_component,
                "component_similarity": similarity,
                "mica_id": mica_id,
                "mica_ic": mica_ic,
                "n_best_ties": ties,
                "hpo_assertion_ids": "|".join(sorted(h_source["assertions"])),
                "mpo_assertion_ids": "|".join(sorted(m_source["assertions"])),
                "hpo_source_leaf_ids": "|".join(sorted(h_source["leaves"])),
                "mpo_source_leaf_ids": "|".join(sorted(m_source["leaves"])),
                "status": "OK",
            }
        writer.add(
            {
                "query_side": str(row.query_side),
                "hpo_id": key[0],
                "mpo_id": key[1],
                "dimension": key[2],
                "rank": int(row.rank),
                "term_score": float(row.score),
                **values,
            }
        )
    count = writer.close()
    result = {"auxiliary_explanation_rows": count}
    write_json(root / "explanations/auxiliary_explanation_manifest.json", result)
    return result


def initialize_descendant_support(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    writer = ParquetRecordWriter(
        root / "explanations/descendant_support.parquet", DESCENDANT_SCHEMA
    )
    count = writer.close()
    result = {
        "status": "READY_ON_DEMAND",
        "row_count": count,
        "reason": "No descendant-specific explanation request has been submitted.",
    }
    write_json(root / "explanations/descendant_support_manifest.json", result)
    return result


def run_explanations(config: dict[str, Any]) -> dict[str, int]:
    result = {
        **run_term_explanations(config),
        **run_auxiliary_explanations(config),
        **run_gene_evidence_chains(config),
        "descendant_support_rows": int(initialize_descendant_support(config)["row_count"]),
    }
    write_json(output_root(config) / "explanations/explanation_manifest.json", result)
    return result
