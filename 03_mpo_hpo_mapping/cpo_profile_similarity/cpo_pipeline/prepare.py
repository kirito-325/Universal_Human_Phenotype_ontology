from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from cpo_mapping.direction import aggregate_signatures, load_rules, normalize_direction
from cpo_mapping.io import file_record, read_tsv, write_json, write_text, write_tsv
from cpo_mapping.ontology import Ontology

from .audit import run_audit
from .config import config_digest, output_root, path_for, public_config, resource_path


NORMALIZED_COLUMNS = [
    "assertion_id",
    "source_record_id",
    "source_file",
    "source_row",
    "ontology_side",
    "target_term_id",
    "source_term_id",
    "source_leaf_id",
    "origin",
    "cl_id_raw",
    "cl_id",
    "go_id_raw",
    "go_id",
    "cl_label_raw",
    "go_label_raw",
    "cell_text",
    "process_text",
    "quality_text",
    "relation_type_raw",
    "relation_role",
    "statement_status",
    "direction_raw",
    "direction_state",
    "direction_target",
    "direction_property",
    "direction_resolution_status",
    "direction_rule_id",
    "evidence_class_raw",
    "evidence_class",
    "confidence_raw",
    "evidence_species",
    "model_id",
    "genotype_id",
    "mouse_gene_ids",
    "publication_ids",
    "evidence_sentence",
    "source_dataset",
    "source_leaf_count_hint",
    "source_assertion_id",
    "inheritance_provenance_id",
    "normalization_rule_id",
    "qc_flags",
]

RELATION_ROLE = {
    "equivalent_cellular_phenotype": "equivalent",
    "has_constituent_cellular_phenotype": "constituent",
    "associated_with": "associated",
    "causally_upstream_of": "causal_upstream",
    "mechanistically_compatible_with": "mechanistic_compatible",
}


def stable_id(prefix: str, *values: object) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return prefix + "_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _component(raw: str, ontology: Ontology, name: str, flags: list[str]) -> str:
    if not raw:
        flags.append(f"missing_{name}")
        return ""
    value, status = ontology.canonicalize(raw)
    if value is None:
        flags.append(f"invalid_{name}:{status}")
        return ""
    if status != "current":
        flags.append(f"{name}_{status}:{raw}")
    if value == ontology.root_id:
        flags.append(f"noninformative_{name}_root")
    return value


def _normalize_row(
    *,
    side: str,
    target_id: str,
    source_id: str,
    source_leaf_id: str,
    origin: str,
    raw: Mapping[str, str],
    raw_evidence: str,
    relation_type: str,
    assertion_id: str,
    source_record_id: str,
    source_file: Path,
    source_row: int,
    cl: Ontology,
    go: Ontology,
    rules: Iterable,
    publication_ids: str = "",
    evidence_sentence: str = "",
    evidence_species: str = "",
    model_id: str = "",
    genotype_id: str = "",
    mouse_gene_ids: str = "",
    confidence_raw: str = "",
    source_assertion_id: str = "",
    inheritance_provenance_id: str = "",
    source_leaf_count_hint: int = 0,
) -> dict[str, object]:
    flags: list[str] = []
    cl_raw = raw.get("cl_id", "")
    go_raw = raw.get("go_id", "")
    cl_id = _component(cl_raw, cl, "cl", flags)
    go_id = _component(go_raw, go, "go", flags)
    evidence = {"direct": "D", "lead": "M", "mechanism": "M"}.get(
        raw_evidence, "unknown"
    )
    direction = normalize_direction(raw, rules)
    return {
        "assertion_id": assertion_id,
        "source_record_id": source_record_id,
        "source_file": str(source_file),
        "source_row": source_row,
        "ontology_side": side,
        "target_term_id": target_id,
        "source_term_id": source_id,
        "source_leaf_id": source_leaf_id,
        "origin": origin,
        "cl_id_raw": cl_raw,
        "cl_id": cl_id,
        "go_id_raw": go_raw,
        "go_id": go_id,
        "cl_label_raw": raw.get("cl_label", ""),
        "go_label_raw": raw.get("go_label", ""),
        "cell_text": raw.get("cell_text", ""),
        "process_text": raw.get("process_text", ""),
        "quality_text": raw.get("quality_text", ""),
        "relation_type_raw": relation_type,
        "relation_role": RELATION_ROLE.get(relation_type, "unknown"),
        "statement_status": "asserted",
        **direction,
        "evidence_class_raw": raw_evidence,
        "evidence_class": evidence,
        "confidence_raw": confidence_raw,
        "evidence_species": evidence_species,
        "model_id": model_id,
        "genotype_id": genotype_id,
        "mouse_gene_ids": mouse_gene_ids,
        "publication_ids": publication_ids,
        "evidence_sentence": evidence_sentence,
        "source_dataset": source_file.name,
        "source_leaf_count_hint": source_leaf_count_hint,
        "source_assertion_id": source_assertion_id,
        "inheritance_provenance_id": inheritance_provenance_id,
        "normalization_rule_id": "cpo_v2",
        "qc_flags": "|".join(flags),
    }


def normalize_assertions(
    config: dict[str, Any], cl: Ontology, go: Ontology
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = load_rules(path_for(config, config["direction"]["rules_file"]))
    values: list[dict[str, object]] = []
    hpo_path = resource_path(config, "hpo_self_cpo")
    mpo_path = resource_path(config, "mpo_self_cpo")
    _, hpo_rows = read_tsv(hpo_path)
    _, mpo_rows = read_tsv(mpo_path)
    for number, row in enumerate(hpo_rows, start=2):
        values.append(
            _normalize_row(
                side="HPO",
                target_id=row["hpo_id"],
                source_id=row["hpo_id"],
                source_leaf_id=row["hpo_id"],
                origin="self",
                raw={**row, "quality_text": ""},
                raw_evidence=row["link_level"],
                relation_type=row["relation_type"],
                assertion_id=stable_id("assertion", "HPO", number, row["record_id"]),
                source_record_id=f"hpo:{number}:{row['record_id']}",
                source_file=hpo_path,
                source_row=number,
                cl=cl,
                go=go,
                rules=rules,
                publication_ids=row["record_id"],
                evidence_sentence=row["representative_quote"],
                evidence_species=row["species"],
            )
        )
    for number, row in enumerate(mpo_rows, start=2):
        values.append(
            _normalize_row(
                side="MPO",
                target_id=row["mp_id"],
                source_id=row["mp_id"],
                source_leaf_id=row["mp_id"] if row["node_type"] == "leaf" else "",
                origin="self",
                raw=row,
                raw_evidence=row["relation_scope"],
                relation_type=row["relation_type"],
                assertion_id=stable_id("assertion", "MPO", row["relation_id"]),
                source_record_id=row["relation_id"],
                source_file=mpo_path,
                source_row=number,
                cl=cl,
                go=go,
                rules=rules,
                publication_ids=row["pmid"],
                evidence_sentence=row["supporting_quote"],
                evidence_species=row["species"],
                model_id=row["genotypes"],
                genotype_id=row["genotypes"],
                mouse_gene_ids=row["marker_ids"],
                confidence_raw=row["confidence"],
            )
        )
    for side in ("HPO", "MPO"):
        path = resource_path(config, f"{side.lower()}_inherited_cpo")
        _, inherited_rows = read_tsv(path)
        for number, row in enumerate(inherited_rows, start=2):
            if side == "HPO":
                target_id = row["target_hpo_id"]
                evidence = row["link_level"]
                relation_type = row["relation_type"]
                raw = {
                    "cl_id": row["cl_id"],
                    "cl_label": row["cl_label"],
                    "cell_text": row["representative_cell_text"],
                    "go_id": row["go_id"],
                    "go_label": row["go_label"],
                    "process_text": row["representative_process_text"],
                    "quality_text": "",
                    "direction": row["direction"],
                }
            else:
                target_id = row["target_mp_id"]
                evidence = row["relation_scope"]
                relation_type = row["relation_type"]
                raw = {
                    "cl_id": row["cl_id"],
                    "cl_label": row["cl_label"],
                    "cell_text": row["representative_cell_text"],
                    "go_id": row["go_id"],
                    "go_label": row["go_label"],
                    "process_text": row["representative_process_text"],
                    "quality_text": row["representative_quality_text"],
                    "direction": row["direction"],
                }
            values.append(
                _normalize_row(
                    side=side,
                    target_id=target_id,
                    source_id="",
                    source_leaf_id="",
                    origin="inherited",
                    raw=raw,
                    raw_evidence=evidence,
                    relation_type=relation_type,
                    assertion_id=f"{side.lower()}_{row['assertion_id']}",
                    source_record_id=row["assertion_id"],
                    source_file=path,
                    source_row=number,
                    cl=cl,
                    go=go,
                    rules=rules,
                    source_assertion_id="MULTIPLE_SEE_PROVENANCE",
                    inheritance_provenance_id=row["assertion_id"],
                    source_leaf_count_hint=int(row["support_leaf_count"]),
                )
            )
    frame = pd.DataFrame(values, columns=NORMALIZED_COLUMNS)
    quarantine = frame["qc_flags"].str.contains("invalid_cl|invalid_go", regex=True)
    return frame[~quarantine].copy(), frame[quarantine].copy()


def build_node_policy(
    config: dict[str, Any], assertions: pd.DataFrame
) -> tuple[list[dict[str, object]], dict[tuple[str, str], str]]:
    _, hpo_rows = read_tsv(resource_path(config, "hpo_node_policy"))
    _, mpo_all_rows = read_tsv(resource_path(config, "mpo_node_policy"))
    mpo_rows = [row for row in mpo_all_rows if row["is_normal_branch"] != "true"]
    complete = assertions[
        (assertions["cl_id"] != "")
        & (assertions["go_id"] != "")
        & ~assertions["qc_flags"].str.contains("noninformative_", regex=True)
    ]
    complete_terms = {
        (str(row.ontology_side), str(row.target_term_id), str(row.origin))
        for row in complete.itertuples(index=False)
    }
    policies: list[dict[str, object]] = []
    strategies: dict[tuple[str, str], str] = {}
    for side, source_rows in (("HPO", hpo_rows), ("MPO", mpo_rows)):
        id_key = "hpo_id" if side == "HPO" else "mp_id"
        label_key = "hpo_label" if side == "HPO" else "mp_label"
        side_index = 0
        for row in sorted(source_rows, key=lambda value: value[id_key]):
            term_id = row[id_key]
            structural_type = row["node_type"]
            if structural_type == "leaf":
                node_type = "LEAF"
                requires_self = True
                strategy = "self"
            elif structural_type in {"root", "top_level"}:
                node_type = "ROOT_LAYER"
                requires_self = False
                strategy = "inherited"
            elif side == "HPO":
                requires_self = row["needs_self_query"] == "true"
                node_type = (
                    "SELF_REQUIRED_INTERNAL"
                    if requires_self
                    else "INHERIT_ONLY_INTERNAL"
                )
                strategy = "self_union_inherited" if requires_self else "inherited"
            else:
                requires_self = row["middle_query_status"].startswith("required")
                node_type = (
                    "SELF_REQUIRED_INTERNAL"
                    if requires_self
                    else "INHERIT_ONLY_INTERNAL"
                )
                strategy = "self_union_inherited" if requires_self else "inherited"
            has_self = (side, term_id, "self") in complete_terms
            has_inherited = (side, term_id, "inherited") in complete_terms
            if side == "HPO":
                if structural_type == "middle" and requires_self:
                    processing_status = "not_started"
                elif structural_type == "leaf" and row["is_target_leaf"] == "true":
                    processing_status = (
                        "completed" if int(row["self_relation_count"]) > 0 else "unknown"
                    )
                else:
                    processing_status = "not_applicable"
            else:
                query_status = row["middle_query_status"]
                if structural_type == "middle" and "queried" in query_status:
                    processing_status = "completed"
                elif structural_type == "middle" and requires_self:
                    processing_status = "not_started"
                elif structural_type == "leaf" and int(row["self_relation_count"]) > 0:
                    processing_status = "completed"
                elif structural_type == "leaf":
                    processing_status = "unknown"
                else:
                    processing_status = "not_applicable"
            final_available = (
                has_self
                if strategy == "self"
                else has_inherited
                if strategy == "inherited"
                else has_self or has_inherited
            )
            strategies[(side, term_id)] = strategy
            policies.append(
                {
                    "ontology_side": side,
                    "index": side_index,
                    "term_id": term_id,
                    "label": row[label_key],
                    "node_type": node_type,
                    "structural_node_type": structural_type,
                    "requires_self": str(requires_self).lower(),
                    "self_processing_status": processing_status,
                    "has_self_assertions": str(int(row["self_relation_count"]) > 0).lower(),
                    "has_usable_self_cpo": str(has_self).lower(),
                    "has_usable_inherited_cpo": str(has_inherited).lower(),
                    "missing_required_self_evidence": str(requires_self and not has_self).lower(),
                    "self_task_incomplete": str(
                        requires_self and processing_status != "completed"
                    ).lower(),
                    "provenance_status": "verified_leaf_closure",
                    "final_strategy": strategy,
                    "final_profile_status": "OK" if final_available else "NO_COMPLETE_CPO",
                }
            )
            side_index += 1
    return policies, strategies


def _membership_record(
    side: str,
    term_id: str,
    view: str,
    structure_id: str,
    node: Mapping[str, object],
    origins: Iterable[str],
    values: list[dict[str, object]],
) -> dict[str, object]:
    signature = aggregate_signatures(values)
    explicit_leaves = {
        str(value["source_leaf_id"]) for value in values if value["source_leaf_id"]
    }
    return {
        "ontology_side": side,
        "term_id": term_id,
        "profile_view": view,
        "structure_id": structure_id,
        "cl_id": node["cl_id"],
        "go_id": node["go_id"],
        "origin_set": "|".join(sorted(set(origins))),
        "evidence_class_set": "|".join(
            sorted({str(value["evidence_class"]) for value in values})
        ),
        "direction_signature_status": signature.status,
        "direction_target": signature.target,
        "direction_property": signature.property_name,
        "direction_state": signature.state,
        "assertion_count": len(values),
        "source_leaf_count": max(
            len(explicit_leaves),
            max((int(value["source_leaf_count_hint"]) for value in values), default=0),
        ),
    }


def build_memberships(
    assertions: pd.DataFrame,
    policies: list[dict[str, object]],
    strategies: dict[tuple[str, str], str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = assertions[
        (assertions["statement_status"] == "asserted")
        & (assertions["cl_id"] != "")
        & (assertions["go_id"] != "")
        & ~assertions["qc_flags"].str.contains("noninformative_", regex=True)
    ]
    origin_nodes: dict[
        tuple[str, str, str], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in complete.to_dict("records"):
        side = str(row["ontology_side"])
        term_id = str(row["target_term_id"])
        origin = str(row["origin"])
        structure_id = stable_id("cpo", row["cl_id"], row["go_id"])
        node = origin_nodes[(side, term_id, origin)].setdefault(
            structure_id,
            {
                "structure_id": structure_id,
                "cl_id": row["cl_id"],
                "go_id": row["go_id"],
                "values": [],
            },
        )
        node["values"].append(row)

    memberships: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for policy in policies:
        side = str(policy["ontology_side"])
        term_id = str(policy["term_id"])
        for view, origins in (
            ("self_only", ("self",)),
            ("inherited_only", ("inherited",)),
        ):
            for structure_id, node in sorted(
                origin_nodes.get((side, term_id, origins[0]), {}).items()
            ):
                memberships.append(
                    _membership_record(
                        side, term_id, view, structure_id, node, origins, node["values"]
                    )
                )
                provenance.extend(
                    _provenance_record(side, term_id, view, structure_id, value)
                    for value in node["values"]
                )
        strategy = strategies[(side, term_id)]
        origins = (
            ("self",)
            if strategy == "self"
            else ("inherited",)
            if strategy == "inherited"
            else ("self", "inherited")
        )
        combined: dict[str, dict[str, object]] = {}
        for origin in origins:
            for structure_id, node in origin_nodes.get((side, term_id, origin), {}).items():
                target = combined.setdefault(
                    structure_id,
                    {
                        "structure_id": structure_id,
                        "cl_id": node["cl_id"],
                        "go_id": node["go_id"],
                        "values": [],
                        "origins": [],
                    },
                )
                target["values"].extend(node["values"])
                target["origins"].append(origin)
        for structure_id, node in sorted(combined.items()):
            memberships.append(
                _membership_record(
                    side,
                    term_id,
                    "final",
                    structure_id,
                    node,
                    node["origins"],
                    node["values"],
                )
            )
            provenance.extend(
                _provenance_record(side, term_id, "final", structure_id, value)
                for value in node["values"]
            )

    assertion_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in assertions.to_dict("records"):
        assertion_groups[
            (str(row["ontology_side"]), str(row["target_term_id"]), str(row["origin"]))
        ].append(row)
    projections: list[dict[str, object]] = []
    for policy in policies:
        side = str(policy["ontology_side"])
        term_id = str(policy["term_id"])
        strategy = strategies[(side, term_id)]
        origins = (
            ("self",)
            if strategy == "self"
            else ("inherited",)
            if strategy == "inherited"
            else ("self", "inherited")
        )
        for dimension, field, root_id in (
            ("CL", "cl_id", "CL:0000000"),
            ("GO", "go_id", "GO:0008150"),
        ):
            components = sorted(
                {
                    str(value[field])
                    for origin in origins
                    for value in assertion_groups.get((side, term_id, origin), ())
                    if value[field]
                    and value[field] != root_id
                    and value["statement_status"] == "asserted"
                }
            )
            projections.extend(
                {
                    "ontology_side": side,
                    "term_id": term_id,
                    "profile_view": "final",
                    "dimension": dimension,
                    "component_id": component,
                }
                for component in components
            )
    return pd.DataFrame(memberships), pd.DataFrame(provenance), pd.DataFrame(projections)


def _provenance_record(
    side: str,
    term_id: str,
    view: str,
    structure_id: str,
    value: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ontology_side": side,
        "term_id": term_id,
        "profile_view": view,
        "structure_id": structure_id,
        "assertion_id": value["assertion_id"],
        "origin": value["origin"],
        "source_leaf_id": value["source_leaf_id"],
        "source_assertion_id": value["source_assertion_id"],
        "inheritance_provenance_id": value["inheritance_provenance_id"],
        "source_relation_id": "",
        "inheritance_distance": "",
        "shortest_path_count": "",
        "canonical_path": "",
    }


def expand_inherited_provenance(
    config: dict[str, Any], provenance: pd.DataFrame
) -> pd.DataFrame:
    self_rows = provenance[provenance["origin"] == "self"].to_dict("records")
    inherited = provenance[provenance["origin"] == "inherited"]
    memberships: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in inherited.itertuples(index=False):
        memberships[
            (str(row.ontology_side), str(row.term_id), str(row.assertion_id))
        ].append((str(row.profile_view), str(row.structure_id)))
    expanded: list[dict[str, object]] = self_rows
    for side in ("HPO", "MPO"):
        path = resource_path(config, f"{side.lower()}_inheritance_provenance")
        _, rows = read_tsv(path)
        for number, row in enumerate(rows, start=2):
            if side == "HPO":
                target = row["target_hpo_id"]
                source_leaf = row["source_leaf_hpo_id"]
            else:
                target = row["target_mp_id"]
                source_leaf = row["source_leaf_mp_id"]
            assertion_id = f"{side.lower()}_{row['assertion_id']}"
            for view, structure_id in memberships.get(
                (side, target, assertion_id), ()
            ):
                expanded.append(
                    {
                        "ontology_side": side,
                        "term_id": target,
                        "profile_view": view,
                        "structure_id": structure_id,
                        "assertion_id": assertion_id,
                        "origin": "inherited",
                        "source_leaf_id": source_leaf,
                        "source_assertion_id": row["source_relation_id"],
                        "inheritance_provenance_id": f"{path.name}:{number}",
                        "source_relation_id": row["source_relation_id"],
                        "inheritance_distance": row["distance"],
                        "shortest_path_count": row["shortest_path_count"],
                        "canonical_path": row["canonical_path"],
                    }
                )
    return pd.DataFrame(expanded)


def run_prepare(config: dict[str, Any]) -> dict[str, object]:
    audit = run_audit(config)
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    cl = Ontology.from_obo(resource_path(config, "cl_ontology"), "CL:0000000", "CL:")
    go = Ontology.from_obo(
        resource_path(config, "go_ontology"),
        "GO:0008150",
        "GO:",
        "biological_process",
    )
    assertions, quarantine = normalize_assertions(config, cl, go)
    policies, strategies = build_node_policy(config, assertions)
    memberships, provenance, projections = build_memberships(
        assertions, policies, strategies
    )
    provenance = expand_inherited_provenance(config, provenance)

    normalized_root = root / "normalized"
    profiles_root = root / "profiles"
    indices_root = root / "indices"
    for path in (normalized_root, profiles_root, indices_root, root / "config"):
        path.mkdir(parents=True, exist_ok=True)
    assertions.to_parquet(normalized_root / "assertions.parquet", index=False)
    quarantine.to_parquet(normalized_root / "quarantine.parquet", index=False)
    memberships.to_parquet(profiles_root / "structure_membership.parquet", index=False)
    provenance.to_parquet(profiles_root / "structure_provenance.parquet", index=False)
    projections.to_parquet(profiles_root / "projection_membership.parquet", index=False)

    policy_fields = tuple(policies[0])
    write_tsv(profiles_root / "node_policy.tsv", policy_fields, policies)
    for side, filename in (("HPO", "hpo_index.tsv"), ("MPO", "mpo_index.tsv")):
        write_tsv(
            indices_root / filename,
            policy_fields,
            [row for row in policies if row["ontology_side"] == side],
        )
    membership_counts = Counter(
        (row.ontology_side, row.term_id, row.profile_view)
        for row in memberships.itertuples(index=False)
    )
    coverage: list[dict[str, object]] = []
    for policy in policies:
        side = str(policy["ontology_side"])
        term_id = str(policy["term_id"])
        coverage.append(
            {
                **policy,
                "self_structure_count": membership_counts[(side, term_id, "self_only")],
                "inherited_structure_count": membership_counts[(side, term_id, "inherited_only")],
                "final_structure_count": membership_counts[(side, term_id, "final")],
            }
        )
    write_tsv(profiles_root / "coverage.tsv", tuple(coverage[0]), coverage)
    write_tsv(
        root / "config/root_layer_ids.tsv",
        ("ontology_side", "term_id", "label"),
        [
            {
                "ontology_side": row["ontology_side"],
                "term_id": row["term_id"],
                "label": row["label"],
            }
            for row in policies
            if row["node_type"] == "ROOT_LAYER"
        ],
    )
    write_text(
        root / "config/cpo_v2.yaml",
        yaml.safe_dump(public_config(config), allow_unicode=True, sort_keys=False),
    )
    write_text(
        root / "config/direction_rules.tsv",
        path_for(config, config["direction"]["rules_file"]).read_text(encoding="utf-8"),
    )

    output_paths = (
        normalized_root / "assertions.parquet",
        normalized_root / "quarantine.parquet",
        profiles_root / "structure_membership.parquet",
        profiles_root / "structure_provenance.parquet",
        profiles_root / "projection_membership.parquet",
        profiles_root / "node_policy.tsv",
        profiles_root / "coverage.tsv",
        indices_root / "hpo_index.tsv",
        indices_root / "mpo_index.tsv",
    )
    result = {
        "schema_version": 2,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_digest(config),
        "audit_counts": audit["disease_model_gene"],
        "counts": {
            "normalized_assertions": len(assertions),
            "quarantined_assertions": len(quarantine),
            "structure_memberships": len(memberships),
            "structure_provenance_rows": len(provenance),
            "projection_memberships": len(projections),
            "hpo_terms": sum(row["ontology_side"] == "HPO" for row in policies),
            "mpo_terms": sum(row["ontology_side"] == "MPO" for row in policies),
            "final_profile_terms": sum(row["final_structure_count"] > 0 for row in coverage),
            "missing_required_self_terms": sum(
                row["missing_required_self_evidence"] == "true" for row in policies
            ),
        },
        "outputs": {
            str(path.relative_to(root)): file_record(path) for path in output_paths
        },
    }
    write_json(profiles_root / "profile_manifest.json", result)
    return result
