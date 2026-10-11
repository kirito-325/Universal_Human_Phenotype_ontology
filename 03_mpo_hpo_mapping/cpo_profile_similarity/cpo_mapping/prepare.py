from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .audit import run_audit
from .config import config_digest, output_root, project_path, public_config
from .io import file_record, sha256, write_json, write_text, write_tsv
from .normalize import normalize_assertions
from .ontology import Ontology
from .profiles import build_indices_and_profiles


def _write_ic(path: Path, ontology: Ontology) -> int:
    rows = (
        {
            "term_id": term_id,
            "label": ontology.records[term_id].name,
            "descendant_count": ontology.descendant_counts[term_id],
            "ic": f"{ontology.ic[term_id]:.17g}",
        }
        for term_id in sorted(ontology.nodes)
    )
    return write_tsv(path, ("term_id", "label", "descendant_count", "ic"), rows)


def run_prepare(config: dict[str, Any], audit_first: bool = True) -> dict[str, Any]:
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    if audit_first:
        audit = run_audit(config)
    else:
        audit = json.loads((root / "audit/schema_inventory.json").read_text(encoding="utf-8"))

    config_root = root / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    write_text(
        config_root / "cpo_mapping_v1.yaml",
        yaml.safe_dump(public_config(config), allow_unicode=True, sort_keys=False),
    )
    rules_source = project_path(config, config["direction"]["rules_file"])
    write_text(
        config_root / "direction_rules.tsv", rules_source.read_text(encoding="utf-8")
    )

    normalization = normalize_assertions(config)
    indices = build_indices_and_profiles(config)
    ontology_root = root / "ontology"
    cl = Ontology.from_obo(
        project_path(config, config["ontologies"]["cl"]), "CL:0000000", "CL:"
    )
    go = Ontology.from_obo(
        project_path(config, config["ontologies"]["go"]),
        "GO:0008150",
        "GO:",
        "biological_process",
    )
    cl_count = _write_ic(ontology_root / "cl_ic.tsv", cl)
    go_count = _write_ic(ontology_root / "go_ic.tsv", go)

    outputs = [
        root / "audit/input_audit.md",
        root / "audit/schema_inventory.json",
        root / "audit/field_mapping.yaml",
        root / "normalized/assertions.parquet",
        root / "normalized/quarantined_assertions.parquet",
        root / "normalized/cpo_structure_nodes.parquet",
        root / "normalized/cpo_node_assertions.parquet",
        root / "indices/hpo_index.tsv",
        root / "indices/mpo_index.tsv",
        root / "profiles/profile_summary.tsv",
        ontology_root / "cl_ic.tsv",
        ontology_root / "go_ic.tsv",
    ]
    manifest = {
        "schema_version": 1,
        "stage": "prepare",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_digest(config),
        "counts": {
            **normalization,
            **indices,
            "cl_ic_term_count": cl_count,
            "go_ic_term_count": go_count,
        },
        "audit_actual_counts": audit["actual_counts"],
        "outputs": {str(path.relative_to(root)): file_record(path) for path in outputs},
    }
    write_json(root / "prepared_manifest.json", manifest)
    return manifest
