from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .config import output_root, project_path, public_config, validate_config
from .io import file_record, read_tsv, write_json, write_text
from .ontology import Ontology


EXPECTED_COUNTS = {
    "hpo_target_leaf_count": 12121,
    "hpo_overview_count": 11999,
    "hpo_relation_count": 47527,
    "mpo_term_count": 14385,
    "mpo_relation_count": 63206,
    "inheritance_provenance_count": 340335,
    "inherited_assertion_count": 284338,
}


ENUM_FIELDS = {
    "hpo_relations": ("link_level", "bridge_scope", "relation_type", "direction", "species"),
    "hpo_overview": ("link_status", "primary_class"),
    "mpo_relations": (
        "node_type",
        "relation_scope",
        "relation_type",
        "direction",
        "evidence_tier",
        "evidence_type",
        "species",
        "confidence",
    ),
    "mpo_status": ("node_type", "status"),
    "inherited_assertions": (
        "target_node_type",
        "relation_scope",
        "relation_type",
        "direction",
        "complete_cpo",
    ),
    "inheritance_status": (
        "node_type",
        "middle_query_status",
        "profile_mode",
        "inheritance_status",
    ),
}


FIELD_MAPPING: dict[str, Any] = {
    "HPO": {
        "term_id": "hpo_id",
        "term_label": "hpo_label",
        "evidence_class": {"source": "link_level", "direct": "D", "lead": "M"},
        "relation_type": "relation_type",
        "direction": "direction",
        "cl_id": "cl_id",
        "go_id": "go_id",
        "cell_text": "cell_text",
        "process_text": "process_text",
        "publication": "record_id/source_url",
        "evidence_sentence": "representative_quote",
        "origin": "self",
    },
    "MPO": {
        "term_id": "mp_id",
        "term_label": "mp_label",
        "evidence_class": {
            "source": "relation_scope",
            "direct": "D",
            "mechanism": "M",
        },
        "relation_type": "relation_type",
        "direction": "direction",
        "cl_id": "cl_id",
        "go_id": "go_id",
        "cell_text": "cell_text",
        "process_text": "process_text",
        "publication": "pmid/source_url",
        "evidence_sentence": "supporting_quote",
        "origin": "self",
    },
    "MPO_inherited": {
        "term_id": "target_mp_id",
        "origin": "inherited",
        "source_leaf_id": "source_leaf_mp_id",
        "source_assertion_id": "source_relation_id -> normalized MPO self assertion",
        "attributes": "joined from source leaf relation",
    },
}


def _schema_record(fields: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    missing = {
        field: sum(not row.get(field, "") for row in rows) for field in fields
    }
    return {
        "fields": fields,
        "logical_row_count": len(rows),
        "missing_value_counts": missing,
    }


def _component_audit(
    rows: list[dict[str, str]],
    cl: Ontology,
    go: Ontology,
) -> dict[str, Any]:
    presence = Counter()
    invalid_cl = Counter()
    invalid_go = Counter()
    root_cl = 0
    root_go = 0
    complete_terms: set[str] = set()
    for row in rows:
        cl_id = row.get("cl_id", "")
        go_id = row.get("go_id", "")
        presence[f"cl_{bool(cl_id)}_go_{bool(go_id)}"] += 1
        canonical_cl, cl_status = cl.canonicalize(cl_id) if cl_id else (None, "missing")
        canonical_go, go_status = go.canonicalize(go_id) if go_id else (None, "missing")
        if cl_id and canonical_cl is None:
            invalid_cl[cl_status] += 1
        if go_id and canonical_go is None:
            invalid_go[go_status] += 1
        root_cl += canonical_cl == cl.root_id
        root_go += canonical_go == go.root_id
        if canonical_cl and canonical_go and canonical_cl != cl.root_id and canonical_go != go.root_id:
            complete_terms.add(row.get("hpo_id") or row.get("mp_id") or row.get("target_mp_id", ""))
    return {
        "component_presence": dict(sorted(presence.items())),
        "invalid_cl": dict(sorted(invalid_cl.items())),
        "invalid_go": dict(sorted(invalid_go.items())),
        "cl_root_row_count": root_cl,
        "go_root_row_count": root_go,
        "term_with_informative_complete_cpo_count": len(complete_terms),
    }


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = output_root(config)
    audit_root = root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    paths = {
        name: project_path(config, value)
        for name, value in config["inputs"].items()
    }
    tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for name in (
        "hpo_relations",
        "hpo_overview",
        "mpo_relations",
        "mpo_status",
        "inherited_assertions",
        "inheritance_provenance",
        "inheritance_status",
        "hpo_target_index",
    ):
        tables[name] = read_tsv(paths[name])

    hp = Ontology.from_obo(
        project_path(config, config["ontologies"]["hp"]), "HP:0000118", "HP:"
    )
    mp = Ontology.from_obo(
        project_path(config, config["ontologies"]["mp"]), "MP:0000001", "MP:"
    )
    cl = Ontology.from_obo(
        project_path(config, config["ontologies"]["cl"]), "CL:0000000", "CL:"
    )
    go = Ontology.from_obo(
        project_path(config, config["ontologies"]["go"]),
        "GO:0008150",
        "GO:",
        "biological_process",
    )

    inventory: dict[str, Any] = {"tables": {}, "ontologies": {}}
    for name, (fields, rows) in tables.items():
        record = _schema_record(fields, rows)
        record["file"] = file_record(paths[name])
        record["enums"] = {
            field: dict(sorted(Counter(row.get(field, "") for row in rows).items()))
            for field in ENUM_FIELDS.get(name, ())
        }
        inventory["tables"][name] = record

    for name, ontology, path_key in (
        ("hp", hp, "hp"),
        ("mp", mp, "mp"),
        ("cl", cl, "cl"),
        ("go", go, "go"),
    ):
        path = project_path(config, config["ontologies"][path_key])
        inventory["ontologies"][name] = {
            "file": file_record(path),
            "data_version": ontology.data_version,
            "root_id": ontology.root_id,
            "active_scope_node_count": len(ontology.nodes),
            "is_a_edge_count": sum(len(values) for values in ontology.parents.values()),
            "dag_valid": True,
        }

    hpo_target_rows = [
        row for row in tables["hpo_target_index"][1] if row.get("node_type") == "leaf"
    ]
    hpo_relation_rows = tables["hpo_relations"][1]
    mpo_relation_rows = tables["mpo_relations"][1]
    provenance_rows = tables["inheritance_provenance"][1]
    inherited_rows = tables["inherited_assertions"][1]
    inheritance_status_rows = tables["inheritance_status"][1]
    actual_counts = {
        "hpo_target_leaf_count": len(hpo_target_rows),
        "hpo_overview_count": len(tables["hpo_overview"][1]),
        "hpo_relation_count": len(hpo_relation_rows),
        "mpo_term_count": len(inheritance_status_rows),
        "mpo_relation_count": len(mpo_relation_rows),
        "inheritance_provenance_count": len(provenance_rows),
        "inherited_assertion_count": len(inherited_rows),
    }
    differences = {
        key: actual_counts[key] - expected for key, expected in EXPECTED_COUNTS.items()
    }
    inventory["actual_counts"] = actual_counts
    inventory["expected_count_differences"] = differences
    inventory["components"] = {
        "hpo": _component_audit(hpo_relation_rows, cl, go),
        "mpo": _component_audit(mpo_relation_rows, cl, go),
        "inherited": _component_audit(inherited_rows, cl, go),
    }
    inventory["duplicates"] = {
        "hpo_full_row_duplicates": len(hpo_relation_rows)
        - len({tuple(row.get(field, "") for field in tables["hpo_relations"][0]) for row in hpo_relation_rows}),
        "mpo_relation_id_duplicates": len(mpo_relation_rows)
        - len({row["relation_id"] for row in mpo_relation_rows}),
        "hpo_target_id_duplicates": len(hpo_target_rows)
        - len({row["hpo_id"] for row in hpo_target_rows}),
        "mpo_index_id_duplicates": len(inheritance_status_rows)
        - len({row["mp_id"] for row in inheritance_status_rows}),
    }
    leaf_relation_ids = {
        row["relation_id"] for row in mpo_relation_rows if row.get("node_type") == "leaf"
    }
    inventory["provenance"] = {
        "source_relation_ids_missing_from_leaf_relations": len(
            {row["source_relation_id"] for row in provenance_rows} - leaf_relation_ids
        ),
        "normal_branch_target_count": sum(
            row.get("is_normal_branch") == "true" and int(row.get("inherited_relation_count", "0")) > 0
            for row in inheritance_status_rows
        ),
    }
    inventory["config"] = public_config(config)

    write_json(audit_root / "schema_inventory.json", inventory)
    mapping_text = yaml.safe_dump(
        FIELD_MAPPING, allow_unicode=True, sort_keys=False, width=100
    )
    write_text(audit_root / "field_mapping.yaml", mapping_text)

    lines = [
        "# CPO mapping input audit",
        "",
        "## Counts",
        "",
        "| Metric | Expected | Actual | Difference |",
        "|---|---:|---:|---:|",
    ]
    for key, expected in EXPECTED_COUNTS.items():
        lines.append(
            f"| `{key}` | {expected:,} | {actual_counts[key]:,} | {differences[key]:+,} |"
        )
    lines.extend(["", "## Ontologies", "", "| Ontology | Version | Root | Nodes | is_a edges |", "|---|---|---|---:|---:|"])
    for name in ("hp", "mp", "cl", "go"):
        value = inventory["ontologies"][name]
        lines.append(
            f"| {name.upper()} | `{value['data_version']}` | `{value['root_id']}` | "
            f"{value['active_scope_node_count']:,} | {value['is_a_edge_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Component validity",
            "",
            "All standard CL and GO identifiers are checked against the frozen ontology scopes. "
            "Missing components remain partial assertions; invalid or non-informative root "
            "components do not enter complete CPO profiles.",
            "",
            "## Provenance",
            "",
            f"- Missing inherited source relation IDs: {inventory['provenance']['source_relation_ids_missing_from_leaf_relations']}",
            f"- Normal-branch targets with inherited abnormal relations: {inventory['provenance']['normal_branch_target_count']}",
            "- HPO `direct` maps to D and `lead` maps to M.",
            "- MPO `direct` maps to D and `mechanism` maps to M.",
            "- Existing accepted relation exports are treated as asserted statements; no raw field explicitly encodes negation or hypothesis status.",
            "",
            "## Index sources",
            "",
            f"- HPO target index: `{paths['hpo_target_index']}`; only rows with `node_type=leaf` are selected.",
            f"- MPO index: `{paths['inheritance_status']}`; every active MPO term is retained.",
        ]
    )
    write_text(audit_root / "input_audit.md", "\n".join(lines) + "\n")
    return inventory
