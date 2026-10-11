from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import output_root, project_path
from .direction import aggregate_signatures, load_rules, normalize_direction
from .io import read_tsv, write_tsv
from .ontology import Ontology


ASSERTION_COLUMNS = [
    "assertion_id",
    "source_record_id",
    "source_file",
    "source_row",
    "ontology_side",
    "term_id",
    "term_label",
    "cl_id_raw",
    "cl_id",
    "go_id_raw",
    "go_id",
    "cl_label_raw",
    "go_label_raw",
    "cell_text",
    "process_text",
    "quality_text",
    "evidence_class",
    "evidence_class_raw",
    "relation_type_raw",
    "relation_role",
    "origin",
    "source_leaf_id",
    "source_assertion_id",
    "direction_raw",
    "direction_state",
    "direction_target",
    "direction_property",
    "direction_resolution_status",
    "direction_rule_id",
    "tissue",
    "stage",
    "condition",
    "model_id",
    "evidence_species",
    "statement_status",
    "publication_ids",
    "evidence_sentence",
    "source_url",
    "normalization_rule_id",
    "qc_flags",
]


RELATION_ROLES = {
    "equivalent_cellular_phenotype": "equivalent",
    "has_constituent_cellular_phenotype": "constituent",
    "associated_with": "associated",
    "causally_upstream_of": "causal_upstream",
    "mechanistically_compatible_with": "mechanistic_compatible",
}


def stable_id(prefix: str, *values: object) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _canonical_component(
    raw_id: str, ontology: Ontology, flags: list[str], component: str
) -> str:
    if not raw_id:
        flags.append(f"missing_{component}")
        return ""
    canonical, status = ontology.canonicalize(raw_id)
    if canonical is None:
        flags.append(f"invalid_{component}:{status}")
        return ""
    if status != "current":
        flags.append(f"{component}_{status}:{raw_id}")
    if canonical == ontology.root_id:
        flags.append(f"noninformative_{component}_root")
    return canonical


def _base_assertion(
    *,
    assertion_id: str,
    source_record_id: str,
    source_file: str,
    source_row: int,
    side: str,
    term_id: str,
    term_label: str,
    raw: Mapping[str, str],
    evidence_class: str,
    evidence_class_raw: str,
    origin: str,
    cl: Ontology,
    go: Ontology,
    rules: Iterable,
    publication_ids: str,
    evidence_sentence: str,
    source_url: str,
    evidence_species: str,
    model_id: str,
    source_leaf_id: str = "",
    source_assertion_id: str = "",
) -> dict[str, object]:
    flags: list[str] = []
    cl_raw = raw.get("cl_id", "")
    go_raw = raw.get("go_id", "")
    cl_id = _canonical_component(cl_raw, cl, flags, "cl")
    go_id = _canonical_component(go_raw, go, flags, "go")
    direction = normalize_direction(raw, rules)
    relation_type = raw.get("relation_type", "")
    relation_role = RELATION_ROLES.get(relation_type, "unresolved")
    if relation_role == "unresolved":
        flags.append("unresolved_relation_role")
    if evidence_class not in {"D", "M"}:
        flags.append("unresolved_evidence_class")
    return {
        "assertion_id": assertion_id,
        "source_record_id": source_record_id,
        "source_file": source_file,
        "source_row": source_row,
        "ontology_side": side,
        "term_id": term_id,
        "term_label": term_label,
        "cl_id_raw": cl_raw,
        "cl_id": cl_id,
        "go_id_raw": go_raw,
        "go_id": go_id,
        "cl_label_raw": raw.get("cl_label", ""),
        "go_label_raw": raw.get("go_label", ""),
        "cell_text": raw.get("cell_text", ""),
        "process_text": raw.get("process_text", ""),
        "quality_text": raw.get("quality_text", ""),
        "evidence_class": evidence_class,
        "evidence_class_raw": evidence_class_raw,
        "relation_type_raw": relation_type,
        "relation_role": relation_role,
        "origin": origin,
        "source_leaf_id": source_leaf_id,
        "source_assertion_id": source_assertion_id,
        **direction,
        "tissue": "",
        "stage": "",
        "condition": "",
        "model_id": model_id,
        "evidence_species": evidence_species,
        "statement_status": "asserted",
        "publication_ids": publication_ids,
        "evidence_sentence": evidence_sentence,
        "source_url": source_url,
        "normalization_rule_id": "cpo_mapping_v1",
        "qc_flags": "|".join(flags),
    }


def normalize_assertions(config: dict[str, Any]) -> dict[str, int]:
    root = output_root(config)
    normalized_root = root / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)
    cl = Ontology.from_obo(
        project_path(config, config["ontologies"]["cl"]), "CL:0000000", "CL:"
    )
    go = Ontology.from_obo(
        project_path(config, config["ontologies"]["go"]),
        "GO:0008150",
        "GO:",
        "biological_process",
    )
    rules = load_rules(project_path(config, config["direction"]["rules_file"]))
    hpo_path = project_path(config, config["inputs"]["hpo_relations"])
    mpo_path = project_path(config, config["inputs"]["mpo_relations"])
    provenance_path = project_path(config, config["inputs"]["inheritance_provenance"])
    _, hpo_rows = read_tsv(hpo_path)
    _, mpo_rows = read_tsv(mpo_path)
    _, provenance_rows = read_tsv(provenance_path)

    assertions: list[dict[str, object]] = []
    mpo_source_by_relation: dict[str, tuple[dict[str, str], str]] = {}
    for logical_row, row in enumerate(hpo_rows, start=2):
        assertion = _base_assertion(
            assertion_id=stable_id("assertion", "HPO", logical_row, row.get("record_id", "")),
            source_record_id=f"hpo:{logical_row}:{row.get('record_id', '')}",
            source_file=str(hpo_path),
            source_row=logical_row,
            side="HPO",
            term_id=row["hpo_id"],
            term_label=row["hpo_label"],
            raw={**row, "quality_text": ""},
            evidence_class={"direct": "D", "lead": "M"}.get(row["link_level"], "UNRESOLVED"),
            evidence_class_raw=row["link_level"],
            origin="self",
            cl=cl,
            go=go,
            rules=rules,
            publication_ids=row.get("record_id", ""),
            evidence_sentence=row.get("representative_quote", ""),
            source_url=row.get("source_url", ""),
            evidence_species=row.get("species", ""),
            model_id="",
        )
        assertions.append(assertion)

    for logical_row, row in enumerate(mpo_rows, start=2):
        normalized_id = stable_id("assertion", "MPO", row["relation_id"])
        assertion = _base_assertion(
            assertion_id=normalized_id,
            source_record_id=row["relation_id"],
            source_file=str(mpo_path),
            source_row=logical_row,
            side="MPO",
            term_id=row["mp_id"],
            term_label=row["mp_label"],
            raw=row,
            evidence_class={"direct": "D", "mechanism": "M"}.get(row["relation_scope"], "UNRESOLVED"),
            evidence_class_raw=row["relation_scope"],
            origin="self",
            cl=cl,
            go=go,
            rules=rules,
            publication_ids=row.get("pmid", ""),
            evidence_sentence=row.get("supporting_quote", ""),
            source_url=row.get("source_url", ""),
            evidence_species=row.get("species", ""),
            model_id=row.get("genotypes", ""),
        )
        assertions.append(assertion)
        mpo_source_by_relation[row["relation_id"]] = (row, normalized_id)

    for logical_row, provenance in enumerate(provenance_rows, start=2):
        source_relation_id = provenance["source_relation_id"]
        if source_relation_id not in mpo_source_by_relation:
            raise ValueError(f"missing inherited source relation: {source_relation_id}")
        source, source_assertion_id = mpo_source_by_relation[source_relation_id]
        assertions.append(
            _base_assertion(
                assertion_id=stable_id(
                    "assertion",
                    "MPO_INHERITED",
                    provenance["target_mp_id"],
                    source_relation_id,
                ),
                source_record_id=f"{provenance['target_mp_id']}:{source_relation_id}",
                source_file=str(provenance_path),
                source_row=logical_row,
                side="MPO",
                term_id=provenance["target_mp_id"],
                term_label="",
                raw=source,
                evidence_class={"direct": "D", "mechanism": "M"}.get(
                    provenance["source_relation_scope"], "UNRESOLVED"
                ),
                evidence_class_raw=provenance["source_relation_scope"],
                origin="inherited",
                cl=cl,
                go=go,
                rules=rules,
                publication_ids=source.get("pmid", ""),
                evidence_sentence="",
                source_url=source.get("source_url", ""),
                evidence_species=source.get("species", ""),
                model_id=source.get("genotypes", ""),
                source_leaf_id=provenance["source_leaf_mp_id"],
                source_assertion_id=source_assertion_id,
            )
        )

    frame = pd.DataFrame(assertions, columns=ASSERTION_COLUMNS)
    quarantined = frame[
        frame["qc_flags"].str.contains("invalid_|unresolved_evidence_class", regex=True)
    ].copy()
    usable = frame.drop(index=quarantined.index).copy()
    usable.to_parquet(normalized_root / "assertions.parquet", index=False)
    quarantined.to_parquet(
        normalized_root / "quarantined_assertions.parquet", index=False
    )
    write_tsv(
        normalized_root / "assertions_sample.tsv",
        ASSERTION_COLUMNS,
        usable.head(200).to_dict("records"),
    )

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in usable.to_dict("records"):
        if (
            row["statement_status"] != "asserted"
            or row["evidence_class"] not in {"D", "M"}
            or not row["cl_id"]
            or not row["go_id"]
            or row["cl_id"] == cl.root_id
            or row["go_id"] == go.root_id
        ):
            continue
        key = (
            str(row["ontology_side"]),
            str(row["term_id"]),
            str(row["evidence_class"]),
            str(row["origin"]),
            str(row["cl_id"]),
            str(row["go_id"]),
        )
        grouped[key].append(row)

    structure_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        side, term_id, evidence_class, origin, cl_id, go_id = key
        node_id = stable_id("cpo", *key)
        signature = aggregate_signatures(values)
        first = values[0]
        structure_rows.append(
            {
                "cpo_node_id": node_id,
                "ontology_side": side,
                "term_id": term_id,
                "term_label": first["term_label"],
                "evidence_class": evidence_class,
                "origin": origin,
                "cl_id": cl_id,
                "cl_label": first["cl_label_raw"],
                "go_id": go_id,
                "go_label": first["go_label_raw"],
                "assertion_count": len(values),
                "direction_signature_status": signature.status,
                "direction_target": signature.target,
                "direction_property": signature.property_name,
                "direction_state": signature.state,
                "n_unresolved_direction_assertions": sum(
                    not str(value["direction_resolution_status"]).startswith("resolved")
                    for value in values
                ),
            }
        )
        link_rows.extend(
            {"cpo_node_id": node_id, "assertion_id": value["assertion_id"]}
            for value in values
        )
    structures = pd.DataFrame(structure_rows)
    links = pd.DataFrame(link_rows, columns=["cpo_node_id", "assertion_id"])
    structures.to_parquet(normalized_root / "cpo_structure_nodes.parquet", index=False)
    links.to_parquet(normalized_root / "cpo_node_assertions.parquet", index=False)
    return {
        "assertion_count": len(usable),
        "quarantined_assertion_count": len(quarantined),
        "structure_node_count": len(structures),
        "structure_assertion_link_count": len(links),
    }
