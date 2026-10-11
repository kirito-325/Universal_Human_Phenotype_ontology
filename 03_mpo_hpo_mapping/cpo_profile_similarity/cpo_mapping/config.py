from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    project_value = Path(str(config.get("project_root", ".")))
    project_root = (
        project_value.resolve()
        if project_value.is_absolute()
        else (config_path.parent.parent / project_value).resolve()
    )
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def output_root(config: dict[str, Any]) -> Path:
    return project_path(config, str(config["output_dir"]))


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    required_inputs = {
        "hpo_relations",
        "hpo_overview",
        "hpo_manifest",
        "mpo_relations",
        "mpo_status",
        "mpo_manifest",
        "inherited_assertions",
        "inheritance_provenance",
        "inheritance_status",
        "inheritance_manifest",
        "hpo_target_index",
    }
    required_ontologies = {"hp", "mp", "cl", "go"}
    missing_inputs = required_inputs - set(config.get("inputs", {}))
    missing_ontologies = required_ontologies - set(config.get("ontologies", {}))
    if missing_inputs or missing_ontologies:
        raise ValueError(
            f"configuration is incomplete: inputs={sorted(missing_inputs)}, "
            f"ontologies={sorted(missing_ontologies)}"
        )
    channels = list(config.get("scoring", {}).get("channels", []))
    if channels != ["DD", "DM", "MD", "MM"]:
        raise ValueError("scoring.channels must be exactly DD, DM, MD, MM")
    for section, keys in (("inputs", required_inputs), ("ontologies", required_ontologies)):
        for key in keys:
            value = config[section][key]
            if value is None:
                raise ValueError(f"unresolved configuration path: {section}.{key}")
            path = project_path(config, str(value))
            if not path.is_file():
                raise FileNotFoundError(f"missing input {section}.{key}: {path}")
