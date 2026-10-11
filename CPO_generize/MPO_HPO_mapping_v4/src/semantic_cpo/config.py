from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .io import stable_digest


MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCES = MODULE_ROOT / "configs" / "resources.yaml"
DEFAULT_REGISTRY = MODULE_ROOT / "configs" / "registry.yaml"
DEFAULT_ADAPTER = MODULE_ROOT / "configs" / "mission3_adapter.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return value


def load_configs(
    resources_path: str | Path = DEFAULT_RESOURCES,
    registry_path: str | Path = DEFAULT_REGISTRY,
    adapter_path: str | Path = DEFAULT_ADAPTER,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return load_yaml(resources_path), load_yaml(registry_path), load_yaml(adapter_path)


def resolve(resources: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(resources["project_root"]) / path


def result_root(resources: dict[str, Any]) -> Path:
    return resolve(resources, resources["module_root"]) / "results" / "semantic_cpo_training_free_v1"


def config_hash(*configs: dict[str, Any]) -> str:
    return stable_digest(configs)


def dump_public_config(*configs: dict[str, Any]) -> str:
    return json.dumps(configs, ensure_ascii=False, sort_keys=True, indent=2)

