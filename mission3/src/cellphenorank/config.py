from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "project": {"name": "CellPhenoRank", "random_seed": 2026, "evaluation_mode": "practical"},
    "filters": {"disease_ids": [], "model_gene_symbols": []},
    "similarity": {
        "provider": "matrix",
        "score_columns": ["semantic_score", "bi_score", "final_score", "score"],
        "missing_pair_score": 0.0,
        "relation_factors": {
            "exact": 1.0,
            "hp_broader": 0.9,
            "broad": 0.9,
            "hp_narrower": 0.8,
            "narrow": 0.8,
            "uncertain": 0.7,
            "related": 0.7,
        },
    },
    "model_scoring": {
        "reverse_match_weight": 1.0,
        "bma_weight": 0.8,
        "max_pair_weight": 0.2,
        "match_threshold": 0.0,
    },
    "hpo_cpo": {
        "confidence_column": "geometric_consensus",
        "min_confidence": 0.0,
        "top_k_per_hpo": 25,
        "specific_scope_weight": 1.0,
        "general_scope_weight": 0.5,
        "require_anatomy_allowed": True,
    },
    "gene_set_scoring": {
        "function_weight": 0.65,
        "marker_weight": 0.25,
        "interaction_weight": 0.10,
        "use_idf": True,
    },
    "rank_fusion": {"lambda": 0.2, "coverage_mode": "disease"},
    "benchmark": {"bootstrap_iterations": 1000},
    "null_models": {"enabled": True, "iterations": 500},
    "plots": {"enabled": True},
    "report": {"top_genes_per_disease": 20, "top_paths_per_gene": 10},
}


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be an object: {source}")
    config = _deep_merge(DEFAULTS, loaded)
    config["_config_path"] = str(source)
    config["_project_root"] = str(source.parent.parent if source.parent.name == "configs" else source.parent)
    return resolve_paths(config)


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_project_root"])
    path_keys = {
        "diseases",
        "models",
        "model_phenotypes",
        "similarity",
        "hpo_cpo",
        "cell_markers",
        "go_functions",
        "cell_functions",
        "gold_standard",
        "gold_models",
        "freeze_manifest",
        "source_snapshot_manifest",
        "open_candidate_manifest",
        "official_freeze_manifest",
        "profile_manifest",
        "gene_units",
        "disease_gene_gold",
        "gene_level_truth",
        "embedding_label",
        "embedding_synonym",
        "cl_obo",
        "label_map",
        "rankings",
        "evidence_paths",
        "metadata",
        "case_manifest",
        "hpo_obo",
        "hpo_background",
        "legacy_output_dir",
        "current_baseline",
        "phenodigm_legacy_matrix",
        "phenodigm_legacy_mpo_presence",
        "official_phenodigm_rankings",
        "phenodigm_mpo_presence",
        "a4_rankings",
        "a5_rankings",
        "evidence_paths_extended",
        "candidate_table",
        "dataset_evidence",
        "dataset_catalog",
        "case_dataset_status",
        "rds",
        "pseudobulk_counts",
        "pseudobulk_metadata",
        "matrix",
        "features",
        "barcodes",
        "signature_file",
        "literature_case_manifest",
        "file",
    }

    def visit(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {item_key: visit(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            if key in {"datasets", "adata"} and all(isinstance(item, str) for item in value):
                return [str((root / item).resolve()) if not Path(item).is_absolute() else str(Path(item)) for item in value]
            return [visit(item) for item in value]
        if key in path_keys and isinstance(value, str) and value:
            candidate = Path(value)
            return str(candidate if candidate.is_absolute() else (root / candidate).resolve())
        return value

    return visit(config)


def dump_config(config: dict[str, Any]) -> str:
    visible = {key: value for key, value in config.items() if not key.startswith("_")}
    return yaml.safe_dump(visible, sort_keys=True, allow_unicode=True)
