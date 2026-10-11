from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("v2 configuration must be a mapping")
    project_value = Path(str(value.get("project_root", ".")))
    project_root = (
        project_value.resolve()
        if project_value.is_absolute()
        else (config_path.parent.parent / project_value).resolve()
    )
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(project_root)
    return value


def path_for(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path


def resource_path(config: dict[str, Any], key: str) -> Path:
    return path_for(config, config["resources"][key])


def output_root(config: dict[str, Any]) -> Path:
    return path_for(config, config["output_dir"])


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "hpo_ontology",
        "mpo_ontology",
        "cl_ontology",
        "go_ontology",
        "hpo_self_cpo",
        "mpo_self_cpo",
        "hpo_inherited_cpo",
        "mpo_inherited_cpo",
        "hpo_inheritance_provenance",
        "mpo_inheritance_provenance",
        "hpo_node_policy",
        "mpo_node_policy",
        "disease_hpo",
        "model_mpo",
        "model_metadata",
        "mouse_human_orthology",
        "hpo_mpo_reference",
        "disease_gene_gold",
    }
    missing = required - set(config.get("resources", {}))
    if missing:
        raise ValueError(f"missing resource keys: {sorted(missing)}")
    for key in required:
        path = resource_path(config, key)
        if not path.is_file():
            raise FileNotFoundError(f"missing v2 resource {key}: {path}")
