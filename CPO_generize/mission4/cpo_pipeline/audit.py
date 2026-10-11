from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from cpo_mapping.io import file_record, read_tsv, write_json, write_text
from cpo_mapping.ontology import Ontology

from .config import output_root, public_config, resource_path, validate_config


def _raw_mgi_counts(path: Path) -> dict[str, object]:
    rows = 0
    malformed = 0
    genotypes: set[str] = set()
    terms: set[str] = set()
    markers: set[str] = set()
    genes_per_model: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 7:
                malformed += 1
                continue
            rows += 1
            terms.add(row[3])
            markers.add(row[5])
            genotypes.add(row[6])
            genes_per_model[row[6]].update(
                value.strip() for value in row[5].split("|") if value.strip()
            )
    return {
        "row_count": rows,
        "malformed_row_count": malformed,
        "model_profile_count": len(genotypes),
        "mpo_term_count": len(terms),
        "mouse_marker_count": len(markers),
        "perturbed_gene_count_distribution": dict(
            sorted(Counter(len(values) for values in genes_per_model.values()).items())
        ),
    }


def _table_record(path: Path, enum_fields: tuple[str, ...] = ()) -> dict[str, object]:
    fields, rows = read_tsv(path)
    return {
        "file": file_record(path),
        "fields": fields,
        "logical_row_count": len(rows),
        "duplicate_full_row_count": len(rows)
        - len({tuple(row.get(field, "") for field in fields) for row in rows}),
        "enums": {
            field: dict(sorted(Counter(row.get(field, "") for row in rows).items()))
            for field in enum_fields
        },
    }


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = output_root(config)
    audit_root = root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    locator = {
        key: file_record(resource_path(config, key))
        for key in sorted(config["resources"])
    }
    write_text(
        audit_root / "resource_locator.yaml",
        yaml.safe_dump(locator, allow_unicode=True, sort_keys=True),
    )

    hp = Ontology.from_obo(
        resource_path(config, "hpo_ontology"), config["scope"]["hpo_root"], "HP:"
    )
    mp_all = Ontology.from_obo(
        resource_path(config, "mpo_ontology"), config["scope"]["mpo_root"], "MP:"
    )
    cl = Ontology.from_obo(
        resource_path(config, "cl_ontology"), "CL:0000000", "CL:"
    )
    go = Ontology.from_obo(
        resource_path(config, "go_ontology"),
        "GO:0008150",
        "GO:",
        "biological_process",
    )
    normal_root = config["scope"]["mpo_normal_root"]
    normal = {normal_root, *descendants(mp_all, normal_root)}

    tables = {
        "hpo_self_cpo": _table_record(
            resource_path(config, "hpo_self_cpo"),
            ("link_level", "bridge_scope", "relation_type", "direction", "species"),
        ),
        "mpo_self_cpo": _table_record(
            resource_path(config, "mpo_self_cpo"),
            ("node_type", "relation_scope", "relation_type", "direction", "species"),
        ),
        "hpo_inherited_cpo": _table_record(
            resource_path(config, "hpo_inherited_cpo"),
            ("target_node_type", "link_level", "direction", "complete_cpo"),
        ),
        "mpo_inherited_cpo": _table_record(
            resource_path(config, "mpo_inherited_cpo"),
            ("target_node_type", "relation_scope", "direction", "complete_cpo"),
        ),
        "hpo_node_policy": _table_record(
            resource_path(config, "hpo_node_policy"),
            ("node_type", "middle_processing_status", "profile_mode"),
        ),
        "mpo_node_policy": _table_record(
            resource_path(config, "mpo_node_policy"),
            ("node_type", "middle_query_status", "profile_mode"),
        ),
        "disease_hpo": _table_record(
            resource_path(config, "disease_hpo"), ("qualifier", "aspect", "sex")
        ),
        "disease_gene_gold": _table_record(
            resource_path(config, "disease_gene_gold"), ("association_type",)
        ),
        "hpo_mpo_reference": _table_record(
            resource_path(config, "hpo_mpo_reference"), ("match_type", "predicate_id")
        ),
    }
    model_counts = _raw_mgi_counts(resource_path(config, "model_mpo"))
    orthology_fields, orthology_rows = read_tsv(
        resource_path(config, "mouse_human_orthology")
    )
    orthology_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"mouse": set(), "human": set()}
    )
    for row in orthology_rows:
        group = orthology_groups[row["DB Class Key"]]
        if row["NCBI Taxon ID"] == "10090" and row["Mouse MGI ID"]:
            group["mouse"].add(row["Mouse MGI ID"])
        elif row["NCBI Taxon ID"] == "9606" and row["HGNC ID"]:
            group["human"].add(row["HGNC ID"])
    orthology_types = Counter(
        (len(value["mouse"]), len(value["human"]))
        for value in orthology_groups.values()
    )
    _, disease_rows = read_tsv(resource_path(config, "disease_hpo"))
    _, gold_rows = read_tsv(resource_path(config, "disease_gene_gold"))
    _, h_policy = read_tsv(resource_path(config, "hpo_node_policy"))
    _, m_policy = read_tsv(resource_path(config, "mpo_node_policy"))
    result = {
        "schema_version": 2,
        "config": public_config(config),
        "resources": locator,
        "ontologies": {
            "hpo": ontology_record(hp),
            "mpo": {
                **ontology_record(mp_all),
                "normal_branch_count": len(normal),
                "non_normal_count": len(mp_all.nodes - normal),
                "normal_branch_ids": sorted(normal),
            },
            "cl": ontology_record(cl),
            "go": ontology_record(go),
        },
        "tables": tables,
        "node_policy": {
            "hpo_count": len(h_policy),
            "hpo_node_types": dict(sorted(Counter(row["node_type"] for row in h_policy).items())),
            "hpo_middle_status": dict(
                sorted(
                    Counter(
                        row["middle_processing_status"]
                        for row in h_policy
                        if row["node_type"] == "middle"
                    ).items()
                )
            ),
            "mpo_count": len(m_policy),
            "mpo_node_types": dict(sorted(Counter(row["node_type"] for row in m_policy).items())),
            "mpo_middle_status": dict(
                sorted(
                    Counter(
                        row["middle_query_status"]
                        for row in m_policy
                        if row["node_type"] == "middle"
                    ).items()
                )
            ),
        },
        "disease_model_gene": {
            "disease_hpo_row_count": len(disease_rows),
            "disease_count": len({row["database_id"] for row in disease_rows}),
            "negative_annotation_count": sum(row["qualifier"] == "NOT" for row in disease_rows),
            "model": model_counts,
            "orthology_file": file_record(resource_path(config, "mouse_human_orthology")),
            "orthology_fields": orthology_fields,
            "orthology_group_count": len(orthology_groups),
            "orthology_type_counts": {
                f"{mouse}x{human}": count
                for (mouse, human), count in sorted(orthology_types.items())
            },
            "disease_gene_gold_row_count": len(gold_rows),
            "disease_gene_gold_disease_count": len({row["disease_id"] for row in gold_rows}),
        },
        "source_risk": {
            "hpo_cpo_gene_or_disease_guided": True,
            "reason": "HPO bridge_scope includes gene_model and supplied_disease; retrospective gene evaluation requires label-derived/source-overlap stratification.",
            "evaluation_type": "retrospective_retrieval",
        },
    }
    write_json(audit_root / "schema_inventory.json", result)
    field_mapping = {
        "disease_hpo": {
            "canonical_disease_id": "database_id",
            "hpo_id": "hpo_id",
            "qualifier": "qualifier",
            "reference_ids": "reference",
            "frequency": "frequency",
            "sex": "sex",
        },
        "model_mpo": {
            "columns": [
                "allelic_composition",
                "allele_symbols",
                "genetic_background",
                "mpo_id",
                "pmid",
                "mouse_marker_ids",
                "genotype_id",
            ],
            "model_profile_id": "genotype_id",
        },
        "orthology": {
            "group": "DB Class Key",
            "mouse_id": "Mouse MGI ID",
            "human_id": "HGNC ID",
        },
    }
    write_text(
        audit_root / "field_mapping.yaml",
        yaml.safe_dump(field_mapping, allow_unicode=True, sort_keys=False),
    )
    lines = [
        "# CPO mapping and gene prediction v2 input audit",
        "",
        f"- HPO abnormality terms: {len(hp.nodes):,}",
        f"- MPO non-normal terms: {len(mp_all.nodes - normal):,}",
        f"- Diseases: {result['disease_model_gene']['disease_count']:,}",
        f"- Specific MGI genotype models: {model_counts['model_profile_count']:,}",
        f"- HPO-MPO reference rows: {tables['hpo_mpo_reference']['logical_row_count']:,}",
        f"- Disease-gene gold rows: {len(gold_rows):,}",
        "",
        "## Provenance risk",
        "",
        result["source_risk"]["reason"],
        "The gene task is reported as retrospective retrieval unless a label-derived source-filtered evaluation can be completed.",
    ]
    write_text(audit_root / "input_audit.md", "\n".join(lines) + "\n")
    write_json(audit_root / "resource_manifest.json", locator)
    return result


def descendants(ontology: Ontology, root: str) -> set[str]:
    result: set[str] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in ontology.children.get(parent, set()):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def ontology_record(ontology: Ontology) -> dict[str, object]:
    return {
        "data_version": ontology.data_version,
        "root_id": ontology.root_id,
        "node_count": len(ontology.nodes),
        "is_a_edge_count": sum(len(values) for values in ontology.parents.values()),
        "dag_valid": True,
    }
