from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .candidates import (
    ParquetRecordWriter,
    _projection_profiles,
    _schema,
)
from .config import output_root, project_path
from .io import read_tsv, write_json
from .matrix import CHANNELS, direction_matrix, load_prepared_scoring


MAIN_EXPLANATION_FIELDS = (
    "hpo_id",
    "mpo_id",
    "channel",
    "score_type",
    "aggregation_direction",
    "status",
    "hpo_cpo_node_id",
    "mpo_cpo_node_id",
    "hpo_cl_id",
    "mpo_cl_id",
    "hpo_go_id",
    "mpo_go_id",
    "cl_similarity",
    "go_similarity",
    "cl_mica_id",
    "cl_mica_ic",
    "cl_mica_tie_count",
    "go_mica_id",
    "go_mica_ic",
    "go_mica_tie_count",
    "pair_shape",
    "direction_status",
    "is_positive_best_match",
    "n_best_ties",
    "hpo_assertion_id",
    "mpo_assertion_id",
    "hpo_source_record_id",
    "mpo_source_record_id",
    "hpo_publication_ids",
    "mpo_publication_ids",
    "hpo_evidence_sentence",
    "mpo_evidence_sentence",
    "hpo_evidence_species",
    "mpo_evidence_species",
    "context_flags",
)
MAIN_EXPLANATION_SCHEMA = _schema(
    MAIN_EXPLANATION_FIELDS,
    {"cl_similarity", "go_similarity", "cl_mica_ic", "go_mica_ic", "pair_shape"},
    {"cl_mica_tie_count", "go_mica_tie_count", "n_best_ties"},
)

AUX_EXPLANATION_FIELDS = (
    "hpo_id",
    "mpo_id",
    "channel",
    "dimension",
    "status",
    "hpo_component_id",
    "mpo_component_id",
    "component_similarity",
    "mica_id",
    "mica_ic",
    "mica_tie_count",
    "n_best_ties",
    "hpo_assertion_id",
    "mpo_assertion_id",
)
AUX_EXPLANATION_SCHEMA = _schema(
    AUX_EXPLANATION_FIELDS,
    {"component_similarity", "mica_ic"},
    {"mica_tie_count", "n_best_ties"},
)

BRANCH_EXPLANATION_FIELDS = (
    "hpo_id",
    "mpo_ancestor_id",
    "channel",
    "branch_score",
    "best_leaf_id",
    "hpo_cpo_node_id",
    "leaf_cpo_node_id",
    "pair_shape",
    "source_relation_ids",
    "canonical_paths",
    "path_count",
    "status",
)
BRANCH_EXPLANATION_SCHEMA = _schema(
    BRANCH_EXPLANATION_FIELDS,
    {"branch_score", "pair_shape"},
    {"path_count"},
)


def _candidate_pairs(root: Path) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    for path in (
        root / "candidates/hpo_to_mpo_topk.parquet",
        root / "candidates/mpo_to_hpo_topk.parquet",
    ):
        frame = pd.read_parquet(path, columns=["query_side", "query_id", "target_id", "channel"])
        for row in frame.itertuples(index=False):
            if row.query_side == "HPO":
                pairs.add((str(row.query_id), str(row.target_id), str(row.channel)))
            else:
                pairs.add((str(row.target_id), str(row.query_id), str(row.channel)))
    return pairs


def _source_indexes(root: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    assertions = pd.read_parquet(root / "normalized/assertions.parquet")
    assertions = assertions[assertions["origin"] == "self"]
    assertion_index = {
        str(row["assertion_id"]): row for row in assertions.to_dict("records")
    }
    links = pd.read_parquet(root / "normalized/cpo_node_assertions.parquet")
    first_assertion: dict[str, str] = {}
    for node_id, values in links.groupby("cpo_node_id"):
        self_ids = sorted(
            assertion_id
            for assertion_id in values["assertion_id"].astype(str)
            if assertion_id in assertion_index
        )
        if self_ids:
            first_assertion[str(node_id)] = self_ids[0]
    return assertion_index, first_assertion


def _direction_name(value: int) -> str:
    return {1: "agree", -1: "conflict"}.get(int(value), "unknown_or_incomparable")


def _best_pair(
    values: np.ndarray,
    left_node_ids: tuple[str, ...],
    right_node_ids: tuple[str, ...],
) -> tuple[int, int, float, int] | None:
    maximum = float(np.max(values))
    if maximum <= 0:
        return None
    positions = np.argwhere(values == maximum)
    ordered = sorted(
        ((int(i), int(j)) for i, j in positions),
        key=lambda pair: (left_node_ids[pair[0]], right_node_ids[pair[1]]),
    )
    left, right = ordered[0]
    return left, right, maximum, len(ordered)


def run_main_explanations(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    hpo_index = {term_id: index for index, term_id in enumerate(prepared.hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    structures = {
        str(row.cpo_node_id): row
        for row in prepared.structures[prepared.structures["origin"] == "self"].itertuples(index=False)
    }
    assertions, first_assertion = _source_indexes(root)
    writer = ParquetRecordWriter(
        root / "explanations/best_cpo_matches.parquet", MAIN_EXPLANATION_SCHEMA
    )
    for hpo_id, mpo_id, channel in sorted(_candidate_pairs(root)):
        h_index = hpo_index[hpo_id]
        m_index = mpo_index[mpo_id]
        h_class, m_class = CHANNELS[channel]
        left = prepared.profiles[("HPO", h_class)][h_index]
        right = prepared.profiles[("MPO", m_class)][m_index]
        pair_shape = prepared.cpo_similarity[np.ix_(left.cpo_indices, right.cpo_indices)]
        directions = direction_matrix(left, right)
        matrices = {
            "shape": pair_shape,
            "support": pair_shape * (directions == 1),
            "conflict": pair_shape * (directions == -1),
        }
        for score_type, values in matrices.items():
            best = _best_pair(values, left.node_ids, right.node_ids)
            for aggregation_direction in ("forward", "reverse"):
                if best is None:
                    writer.add(
                        {
                            "hpo_id": hpo_id,
                            "mpo_id": mpo_id,
                            "channel": channel,
                            "score_type": score_type,
                            "aggregation_direction": aggregation_direction,
                            "status": "NO_POSITIVE_MATCH",
                            "hpo_cpo_node_id": "",
                            "mpo_cpo_node_id": "",
                            "hpo_cl_id": "",
                            "mpo_cl_id": "",
                            "hpo_go_id": "",
                            "mpo_go_id": "",
                            "cl_similarity": float("nan"),
                            "go_similarity": float("nan"),
                            "cl_mica_id": "",
                            "cl_mica_ic": float("nan"),
                            "cl_mica_tie_count": 0,
                            "go_mica_id": "",
                            "go_mica_ic": float("nan"),
                            "go_mica_tie_count": 0,
                            "pair_shape": 0.0,
                            "direction_status": "unknown_or_incomparable",
                            "is_positive_best_match": "false",
                            "n_best_ties": 0,
                            "hpo_assertion_id": "",
                            "mpo_assertion_id": "",
                            "hpo_source_record_id": "",
                            "mpo_source_record_id": "",
                            "hpo_publication_ids": "",
                            "mpo_publication_ids": "",
                            "hpo_evidence_sentence": "",
                            "mpo_evidence_sentence": "",
                            "hpo_evidence_species": "",
                            "mpo_evidence_species": "",
                            "context_flags": "",
                        }
                    )
                    continue
                left_position, right_position, value, tie_count = best
                h_node_id = left.node_ids[left_position]
                m_node_id = right.node_ids[right_position]
                h_node = structures[h_node_id]
                m_node = structures[m_node_id]
                cl_mica, cl_ic, cl_ties = prepared.cl.mica(str(h_node.cl_id), str(m_node.cl_id))
                go_mica, go_ic, go_ties = prepared.go.mica(str(h_node.go_id), str(m_node.go_id))
                h_assertion_id = first_assertion.get(h_node_id, "")
                m_assertion_id = first_assertion.get(m_node_id, "")
                h_assertion = assertions.get(h_assertion_id, {})
                m_assertion = assertions.get(m_assertion_id, {})
                direction_status = _direction_name(directions[left_position, right_position])
                writer.add(
                    {
                        "hpo_id": hpo_id,
                        "mpo_id": mpo_id,
                        "channel": channel,
                        "score_type": score_type,
                        "aggregation_direction": aggregation_direction,
                        "status": "OK",
                        "hpo_cpo_node_id": h_node_id,
                        "mpo_cpo_node_id": m_node_id,
                        "hpo_cl_id": str(h_node.cl_id),
                        "mpo_cl_id": str(m_node.cl_id),
                        "hpo_go_id": str(h_node.go_id),
                        "mpo_go_id": str(m_node.go_id),
                        "cl_similarity": prepared.cl.lin(str(h_node.cl_id), str(m_node.cl_id)),
                        "go_similarity": prepared.go.lin(str(h_node.go_id), str(m_node.go_id)),
                        "cl_mica_id": cl_mica or "",
                        "cl_mica_ic": cl_ic,
                        "cl_mica_tie_count": cl_ties,
                        "go_mica_id": go_mica or "",
                        "go_mica_ic": go_ic,
                        "go_mica_tie_count": go_ties,
                        "pair_shape": value,
                        "direction_status": direction_status,
                        "is_positive_best_match": "true",
                        "n_best_ties": tie_count,
                        "hpo_assertion_id": h_assertion_id,
                        "mpo_assertion_id": m_assertion_id,
                        "hpo_source_record_id": str(h_assertion.get("source_record_id", "")),
                        "mpo_source_record_id": str(m_assertion.get("source_record_id", "")),
                        "hpo_publication_ids": str(h_assertion.get("publication_ids", "")),
                        "mpo_publication_ids": str(m_assertion.get("publication_ids", "")),
                        "hpo_evidence_sentence": str(h_assertion.get("evidence_sentence", "")),
                        "mpo_evidence_sentence": str(m_assertion.get("evidence_sentence", "")),
                        "hpo_evidence_species": str(h_assertion.get("evidence_species", "")),
                        "mpo_evidence_species": str(m_assertion.get("evidence_species", "")),
                        "context_flags": "context_unverified" if direction_status in {"agree", "conflict"} else "",
                    }
                )
    count = writer.close()
    result = {"main_explanation_count": count}
    write_json(root / "logs/main_explanations_summary.json", result)
    return result


def run_auxiliary_explanations(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    hpo_index = {term_id: index for index, term_id in enumerate(prepared.hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    assertions = pd.read_parquet(root / "normalized/assertions.parquet")
    assertions = assertions[assertions["origin"] == "self"]
    first_component_assertion: dict[tuple[str, str, str, str, str], str] = {}
    for row in assertions.sort_values("assertion_id").itertuples(index=False):
        for dimension, component in (("CL", str(row.cl_id)), ("GO", str(row.go_id))):
            if component:
                first_component_assertion.setdefault(
                    (str(row.ontology_side), str(row.term_id), str(row.evidence_class), dimension, component),
                    str(row.assertion_id),
                )
    candidates = pd.read_parquet(root / "candidates/auxiliary_topk.parquet")
    pairs = set()
    for row in candidates.itertuples(index=False):
        if row.query_side == "HPO":
            pairs.add((str(row.query_id), str(row.target_id), str(row.channel), str(row.dimension)))
        else:
            pairs.add((str(row.target_id), str(row.query_id), str(row.channel), str(row.dimension)))
    writer = ParquetRecordWriter(
        root / "explanations/auxiliary_matches.parquet", AUX_EXPLANATION_SCHEMA
    )
    cache: dict[str, tuple[list[str], list[str], np.ndarray, dict]] = {}
    for dimension in ("CL", "GO"):
        cache[dimension] = _projection_profiles(config, prepared, dimension)
    for hpo_id, mpo_id, channel, dimension in sorted(pairs):
        h_ids, m_ids, similarity, profiles = cache[dimension]
        h_class, m_class = CHANNELS[channel]
        h_profile = profiles[("HPO", h_class)][hpo_index[hpo_id]]
        m_profile = profiles[("MPO", m_class)][mpo_index[mpo_id]]
        values = similarity[np.ix_(h_profile, m_profile)]
        maximum = float(values.max())
        if maximum <= 0:
            continue
        positions = np.argwhere(values == maximum)
        choices = sorted(
            ((int(i), int(j)) for i, j in positions),
            key=lambda pair: (h_ids[h_profile[pair[0]]], m_ids[m_profile[pair[1]]]),
        )
        left, right = choices[0]
        h_component = h_ids[h_profile[left]]
        m_component = m_ids[m_profile[right]]
        ontology = prepared.cl if dimension == "CL" else prepared.go
        mica, mica_ic, mica_ties = ontology.mica(h_component, m_component)
        writer.add(
            {
                "hpo_id": hpo_id,
                "mpo_id": mpo_id,
                "channel": channel,
                "dimension": dimension,
                "status": "OK",
                "hpo_component_id": h_component,
                "mpo_component_id": m_component,
                "component_similarity": maximum,
                "mica_id": mica or "",
                "mica_ic": mica_ic,
                "mica_tie_count": mica_ties,
                "n_best_ties": len(choices),
                "hpo_assertion_id": first_component_assertion.get(
                    ("HPO", hpo_id, h_class, dimension, h_component), ""
                ),
                "mpo_assertion_id": first_component_assertion.get(
                    ("MPO", mpo_id, m_class, dimension, m_component), ""
                ),
            }
        )
    count = writer.close()
    result = {"auxiliary_explanation_count": count}
    write_json(root / "logs/auxiliary_explanations_summary.json", result)
    return result


def run_branch_explanations(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    prepared = load_prepared_scoring(config)
    hpo_index = {term_id: index for index, term_id in enumerate(prepared.hpo_ids)}
    mpo_index = {term_id: index for index, term_id in enumerate(prepared.mpo_ids)}
    _, provenance = read_tsv(project_path(config, config["inputs"]["inheritance_provenance"]))
    provenance_index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in provenance:
        provenance_index[
            (row["target_mp_id"], row["source_leaf_mp_id"], row["source_relation_scope"])
        ].append(row)
    assertions = pd.read_parquet(
        root / "normalized/assertions.parquet",
        columns=["assertion_id", "ontology_side", "origin", "source_record_id"],
    )
    assertions = assertions[
        (assertions["ontology_side"] == "MPO") & (assertions["origin"] == "self")
    ]
    assertion_to_raw = dict(
        zip(assertions["assertion_id"].astype(str), assertions["source_record_id"].astype(str))
    )
    links = pd.read_parquet(root / "normalized/cpo_node_assertions.parquet")
    node_raw_relations: dict[str, set[str]] = defaultdict(set)
    for link in links.itertuples(index=False):
        assertion_id = str(link.assertion_id)
        if assertion_id in assertion_to_raw:
            node_raw_relations[str(link.cpo_node_id)].add(assertion_to_raw[assertion_id])
    candidates = pd.read_parquet(root / "candidates/branch_support_topk.parquet")
    writer = ParquetRecordWriter(
        root / "explanations/branch_matches.parquet", BRANCH_EXPLANATION_SCHEMA
    )
    for row in candidates.itertuples(index=False):
        h_index = hpo_index[str(row.hpo_id)]
        leaf_index = mpo_index[str(row.best_leaf_id)]
        h_class, m_class = CHANNELS[str(row.channel)]
        left = prepared.profiles[("HPO", h_class)][h_index]
        right = prepared.profiles[("MPO", m_class)][leaf_index]
        pair = prepared.cpo_similarity[np.ix_(left.cpo_indices, right.cpo_indices)]
        best = _best_pair(pair, left.node_ids, right.node_ids)
        raw_scope = "direct" if m_class == "D" else "mechanism"
        source_rows = provenance_index.get(
            (str(row.mpo_ancestor_id), str(row.best_leaf_id), raw_scope), []
        )
        if best is None:
            continue
        left_position, right_position, value, _ties = best
        allowed_source_relations = node_raw_relations.get(
            right.node_ids[right_position], set()
        )
        source_rows = [
            source
            for source in source_rows
            if source["source_relation_id"] in allowed_source_relations
        ]
        writer.add(
            {
                "hpo_id": str(row.hpo_id),
                "mpo_ancestor_id": str(row.mpo_ancestor_id),
                "channel": str(row.channel),
                "branch_score": float(row.branch_score),
                "best_leaf_id": str(row.best_leaf_id),
                "hpo_cpo_node_id": left.node_ids[left_position],
                "leaf_cpo_node_id": right.node_ids[right_position],
                "pair_shape": value,
                "source_relation_ids": "|".join(sorted({value["source_relation_id"] for value in source_rows})),
                "canonical_paths": "||".join(sorted({value["canonical_path"] for value in source_rows})),
                "path_count": len(source_rows),
                "status": "OK",
            }
        )
    count = writer.close()
    result = {"branch_explanation_count": count}
    write_json(root / "logs/branch_explanations_summary.json", result)
    return result


def run_explanations(config: dict[str, Any]) -> dict[str, int]:
    result = {
        **run_main_explanations(config),
        **run_auxiliary_explanations(config),
        **run_branch_explanations(config),
    }
    write_json(output_root(config) / "logs/explanations_summary.json", result)
    return result
