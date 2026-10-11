from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_root, project_path
from .io import read_tsv, write_tsv
from .ontology import Ontology


INDEX_FIELDS = (
    "index",
    "term_id",
    "label",
    "ontology_version",
    "node_class",
    "processing_status",
    "has_self_relations",
    "has_complete_cpo",
    "has_partial_cpo",
    "scoring_policy_status",
)


def build_indices_and_profiles(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    indices_root = root / "indices"
    profiles_root = root / "profiles"
    indices_root.mkdir(parents=True, exist_ok=True)
    profiles_root.mkdir(parents=True, exist_ok=True)

    assertions = pd.read_parquet(root / "normalized/assertions.parquet")
    structures = pd.read_parquet(root / "normalized/cpo_structure_nodes.parquet")
    self_assertions = assertions[assertions["origin"] == "self"]
    self_structures = structures[structures["origin"] == "self"]

    hp = Ontology.from_obo(
        project_path(config, config["ontologies"]["hp"]), "HP:0000118", "HP:"
    )
    mp = Ontology.from_obo(
        project_path(config, config["ontologies"]["mp"]), "MP:0000001", "MP:"
    )

    _, hpo_source_rows = read_tsv(
        project_path(config, config["inputs"]["hpo_target_index"])
    )
    target_rows = [row for row in hpo_source_rows if row.get("node_type") == "leaf"]
    if len(target_rows) != 12121:
        raise ValueError(f"expected 12,121 HPO leaf targets, found {len(target_rows)}")
    _, overview_rows = read_tsv(project_path(config, config["inputs"]["hpo_overview"]))
    overview = {row["hpo_id"]: row for row in overview_rows}
    if len(overview) != len(overview_rows):
        raise ValueError("HPO overview contains duplicate IDs")

    _, mpo_source_rows = read_tsv(
        project_path(config, config["inputs"]["inheritance_status"])
    )
    if len(mpo_source_rows) != 14385:
        raise ValueError(f"expected 14,385 MPO terms, found {len(mpo_source_rows)}")

    relation_terms = {
        (row.ontology_side, row.term_id)
        for row in self_assertions[["ontology_side", "term_id"]].itertuples(index=False)
    }
    complete_terms = {
        (row.ontology_side, row.term_id)
        for row in self_structures[["ontology_side", "term_id"]].itertuples(index=False)
    }
    partial_terms = {
        (row.ontology_side, row.term_id)
        for row in self_assertions.itertuples(index=False)
        if bool(row.cl_id) != bool(row.go_id)
    }

    hpo_index_rows: list[dict[str, object]] = []
    canonical_hpo_ids: set[str] = set()
    for index, source in enumerate(sorted(target_rows, key=lambda row: row["hpo_id"])):
        canonical, status = hp.canonicalize(source["hpo_id"])
        if canonical is None:
            raise ValueError(f"HPO target is invalid in frozen ontology: {source['hpo_id']} ({status})")
        if canonical in canonical_hpo_ids:
            raise ValueError(f"canonical HPO target collision: {canonical}")
        canonical_hpo_ids.add(canonical)
        has_relations = ("HPO", canonical) in relation_terms
        has_complete = ("HPO", canonical) in complete_terms
        has_partial = ("HPO", canonical) in partial_terms
        if has_complete:
            scoring_status = "OK"
        elif has_relations:
            scoring_status = "NO_COMPLETE_CPO"
        else:
            scoring_status = "NO_RELATIONS"
        hpo_index_rows.append(
            {
                "index": index,
                "term_id": canonical,
                "label": source["hpo_label"],
                "ontology_version": hp.data_version,
                "node_class": "leaf",
                "processing_status": overview.get(canonical, {}).get(
                    "link_status", "absent_from_source_overview"
                ),
                "has_self_relations": str(has_relations).lower(),
                "has_complete_cpo": str(has_complete).lower(),
                "has_partial_cpo": str(has_partial).lower(),
                "scoring_policy_status": scoring_status,
            }
        )

    mpo_index_rows: list[dict[str, object]] = []
    for index, source in enumerate(sorted(mpo_source_rows, key=lambda row: row["mp_id"])):
        term_id = source["mp_id"]
        canonical, status = mp.canonicalize(term_id)
        if canonical != term_id:
            raise ValueError(f"MPO index is not canonical: {term_id} ({status})")
        has_relations = ("MPO", term_id) in relation_terms
        has_complete = ("MPO", term_id) in complete_terms
        has_partial = ("MPO", term_id) in partial_terms
        if source["node_type"] in {"root", "normal"}:
            scoring_status = "EXCLUDED_NODE"
        elif has_complete:
            scoring_status = "OK"
        elif has_relations:
            scoring_status = "NO_COMPLETE_CPO"
        else:
            scoring_status = "NO_RELATIONS"
        mpo_index_rows.append(
            {
                "index": index,
                "term_id": term_id,
                "label": source["mp_label"],
                "ontology_version": mp.data_version,
                "node_class": source["node_type"],
                "processing_status": source["profile_mode"],
                "has_self_relations": str(has_relations).lower(),
                "has_complete_cpo": str(has_complete).lower(),
                "has_partial_cpo": str(has_partial).lower(),
                "scoring_policy_status": scoring_status,
            }
        )

    write_tsv(indices_root / "hpo_index.tsv", INDEX_FIELDS, hpo_index_rows)
    write_tsv(indices_root / "mpo_index.tsv", INDEX_FIELDS, mpo_index_rows)

    assertion_groups = defaultdict(list)
    for row in self_assertions.to_dict("records"):
        assertion_groups[(row["ontology_side"], row["term_id"], row["evidence_class"])].append(row)
    structure_groups = defaultdict(list)
    for row in self_structures.to_dict("records"):
        structure_groups[(row["ontology_side"], row["term_id"], row["evidence_class"])].append(row)

    profile_rows: list[dict[str, object]] = []
    for side, index_rows in (("HPO", hpo_index_rows), ("MPO", mpo_index_rows)):
        for term in index_rows:
            for evidence_class in ("D", "M"):
                assertion_values = assertion_groups[(side, term["term_id"], evidence_class)]
                structure_values = structure_groups[(side, term["term_id"], evidence_class)]
                cl_projection = {row["cl_id"] for row in assertion_values if row["cl_id"]}
                go_projection = {row["go_id"] for row in assertion_values if row["go_id"]}
                direction_counts = Counter(
                    row["direction_signature_status"] for row in structure_values
                )
                profile_rows.append(
                    {
                        "ontology_side": side,
                        "term_id": term["term_id"],
                        "evidence_class": evidence_class,
                        "self_assertion_count": len(assertion_values),
                        "complete_structure_count": len(structure_values),
                        "cl_projection_count": len(cl_projection),
                        "go_projection_count": len(go_projection),
                        "unique_direction_structure_count": direction_counts["unique"],
                        "mixed_direction_structure_count": direction_counts["mixed_direction"],
                        "multiple_axes_structure_count": direction_counts["multiple_axes"],
                        "unknown_direction_structure_count": direction_counts["unknown"],
                    }
                )
    profile_fields = tuple(profile_rows[0])
    write_tsv(profiles_root / "profile_summary.tsv", profile_fields, profile_rows)
    return {
        "hpo_index_count": len(hpo_index_rows),
        "mpo_index_count": len(mpo_index_rows),
        "profile_summary_count": len(profile_rows),
    }
