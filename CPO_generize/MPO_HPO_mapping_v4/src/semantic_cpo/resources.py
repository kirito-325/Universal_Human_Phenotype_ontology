from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import config_hash, resolve, result_root
from .io import file_record, read_tsv, sha256, write_json, write_tsv


def _path(resources: dict[str, Any], section: str, key: str) -> Path:
    return resolve(resources, resources[section][key])


def _schema(path: Path) -> dict[str, object]:
    suffixes = path.suffixes
    if suffixes[-2:] == [".tsv", ".gz"]:
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle, delimiter="\t"))
        return {"format": "tsv.gz", "fields": fields}
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle, delimiter="\t"))
        return {"format": "tsv", "fields": fields}
    if path.suffix == ".parquet":
        schema = pd.read_parquet(path, columns=[]).columns.tolist()
        if not schema:
            import pyarrow.parquet as pq

            schema = pq.read_schema(path).names
        return {"format": "parquet", "fields": schema}
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return {"format": "json", "top_level": sorted(value) if isinstance(value, dict) else type(value).__name__}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            return {
                "format": "npz",
                "arrays": {
                    key: {"shape": list(values[key].shape), "dtype": str(values[key].dtype)}
                    for key in values.files
                },
            }
    if path.suffix in {".h5", ".hdf5"}:
        datasets: dict[str, object] = {}
        with h5py.File(path, "r") as handle:
            handle.visititems(
                lambda name, value: datasets.update(
                    {name: {"shape": list(value.shape), "dtype": str(value.dtype)}}
                )
                if isinstance(value, h5py.Dataset)
                else None
            )
        return {"format": "hdf5", "datasets": datasets}
    return {"format": path.suffix.lstrip(".") or "directory"}


def audit_resources(
    resources: dict[str, Any], registry: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, object]:
    output = result_root(resources) / "discovered"
    configured = {
        "benchmark_diseases": _path(resources, "benchmark", "diseases"),
        "benchmark_models": _path(resources, "benchmark", "models"),
        "benchmark_model_phenotypes": _path(resources, "benchmark", "model_phenotypes"),
        "benchmark_gold_models": _path(resources, "benchmark", "gold_models"),
        "benchmark_freeze_manifest": _path(resources, "benchmark", "freeze_manifest"),
        "mission3_config_2602": _path(resources, "mission3", "config_2602"),
        "mission3_exomiser_2602_matrix": _path(resources, "mission3", "exomiser_2602_matrix"),
        "mission3_exomiser_2602_presence": _path(resources, "mission3", "exomiser_2602_presence"),
        "mission4_hpo_index": _path(resources, "mission4", "hpo_index"),
        "mission4_mpo_index": _path(resources, "mission4", "mpo_index"),
        "mission4_assertions": _path(resources, "mission4", "assertions"),
        "mission4_structure_membership": _path(resources, "mission4", "structure_membership"),
        "mission4_structure_provenance": _path(resources, "mission4", "structure_provenance"),
        "mission4_legacy_matrix": _path(resources, "mission4", "legacy_matrix"),
        "mission4_run_manifest": _path(resources, "mission4", "run_manifest"),
        "hpo_ontology": _path(resources, "ontologies", "hpo"),
        "mpo_ontology": _path(resources, "ontologies", "mpo"),
        "cl_ontology": _path(resources, "ontologies", "cl"),
        "go_ontology": _path(resources, "ontologies", "go"),
        "legacy_semantic_vectors": _path(resources, "legacy_semantics", "vectors"),
        "legacy_semantic_manifest": _path(resources, "legacy_semantics", "manifest"),
    }
    records: dict[str, object] = {}
    missing: list[str] = []
    for name, path in configured.items():
        if not path.is_file():
            records[name] = {"path": str(path), "status": "missing"}
            missing.append(name)
            continue
        records[name] = {**file_record(path), "status": "found", "schema": _schema(path)}
    encoder = _path(resources, "encoder", "path")
    encoder_files = [encoder / name for name in ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt")]
    encoder_missing = [str(path) for path in encoder_files if not path.is_file()]
    records["encoder"] = {
        "path": str(encoder),
        "status": "missing" if encoder_missing else "found",
        "revision": resources["encoder"]["revision"],
        "files": {path.name: file_record(path) for path in encoder_files if path.is_file()},
        "missing": encoder_missing,
    }
    if encoder_missing:
        missing.append("encoder")

    freeze = json.loads(configured["benchmark_freeze_manifest"].read_text(encoding="utf-8"))
    if freeze.get("frozen") is not True or freeze.get("middle_layer_independent") is not True:
        raise ValueError("benchmark is not frozen independently of the middle layer")
    diseases = read_tsv(configured["benchmark_diseases"])
    models = read_tsv(configured["benchmark_models"])
    phenotypes = read_tsv(configured["benchmark_model_phenotypes"])
    gold = read_tsv(configured["benchmark_gold_models"])
    single_models = [row for row in models if ";" not in row.get("human_gene_symbol", "")]
    single_ids = {row["model_id"] for row in single_models}
    abnormal = [
        row
        for row in phenotypes
        if row["model_id"] in single_ids and row.get("status") == "abnormal"
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete" if not missing else "incomplete",
        "config_hash": config_hash(resources, registry, adapter),
        "resources": records,
        "missing": missing,
        "benchmark_contract": {
            "frozen": True,
            "middle_layer_independent": True,
            "review_policy": freeze.get("review_policy"),
            "disease_count": len({row["disease_id"] for row in diseases}),
            "candidate_gene_count": len({row["human_gene_symbol"] for row in single_models}),
            "single_gene_model_count": len(single_models),
            "abnormal_model_phenotype_rows": len(abnormal),
            "unique_abnormal_mpo": len({row["mp_id"] for row in abnormal}),
            "gold_model_rows": len(gold),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "audit_notes": [
            "The existing projected semantic cache cannot encode missing terms because its projection basis was not persisted; the registered run rebuilds all term vectors with the same frozen SapBERT checkpoint.",
            "Q is registered as Q_availability because the normalized assertion resource has traceability fields but no independent record-level reliability review shared by both ontologies.",
        ],
    }
    write_json(output / "resource_manifest.json", manifest)
    rows = [
        {
            "resource": name,
            "status": value.get("status", ""),
            "path": value.get("path", ""),
            "sha256": value.get("sha256", ""),
            "bytes": value.get("bytes", ""),
        }
        for name, value in records.items()
    ]
    write_tsv(output / "resource_manifest.tsv", rows, ("resource", "status", "path", "sha256", "bytes"))
    return manifest


def freeze_inputs(
    resources: dict[str, Any], registry: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources)
    resource_manifest = root / "discovered" / "resource_manifest.json"
    baseline = root / "r0" / "baseline_reproduction.json"
    if not resource_manifest.is_file() or not baseline.is_file():
        raise FileNotFoundError("audit-resources and reproduce-baselines must complete first")
    audit = json.loads(resource_manifest.read_text(encoding="utf-8"))
    reproduction = json.loads(baseline.read_text(encoding="utf-8"))
    if audit["status"] != "complete":
        raise ValueError("resource audit is incomplete")
    if reproduction["status"] != "pass":
        raise ValueError("baseline reproduction has not passed")
    manifest = {
        "schema_version": 1,
        "frozen": True,
        "middle_layer_independent_benchmark": True,
        "training_enabled": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash(resources, registry, adapter),
        "resource_manifest": file_record(resource_manifest),
        "baseline_reproduction": file_record(baseline),
        "registry": registry,
        "adapter": adapter,
    }
    write_json(root / "frozen_inputs" / "input_manifest.json", manifest)
    return manifest

