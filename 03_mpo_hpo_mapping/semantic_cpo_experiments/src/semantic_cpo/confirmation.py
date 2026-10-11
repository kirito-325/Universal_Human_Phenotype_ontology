from __future__ import annotations


import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import resolve, result_root
from .fusion import fuse_array
from .io import file_record, read_tsv, write_json, write_tsv
from .matrices import load_working_profiles, q_matrix
from .mission3_adapter import load_task, task_hpo_ids


def _surface(resources: dict[str, Any], registry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], str]:
    root = result_root(resources)
    selection = json.loads((root / "r4_fusion/selection.json").read_text(encoding="utf-8"))
    finalist = selection["finalists"][0]
    method_id = finalist["method_id"]
    fusion = finalist["fusion"]
    index = json.loads((root / "cache/surface_index.json").read_text(encoding="utf-8"))
    hpo_ids = index["hpo_ids"]
    mpo_ids = index["mpo_ids"]
    s = np.load(root / "cache/surface_S.npy", mmap_mode="r")
    m = np.load(root / "r3_profiles/mechanisms" / f"{method_id}.npy", mmap_mode="r")
    q = q_matrix(load_working_profiles(resources, hpo_ids))
    o = fuse_array(fusion, s, m, q, float(registry["fusion"]["epsilon"]))
    return np.asarray(s), o, hpo_ids, mpo_ids, f"{method_id}__{fusion}"


def _candidate_mpo(task: Any) -> set[str]:
    return {mpo_id for model in task.models for mpo_id in model.mpo_ids}


def export_surface(
    path: Path,
    surface: np.ndarray,
    hpo_ids: list[str],
    mpo_ids: list[str],
    candidate_mpo: set[str],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    m_index = {value: index for index, value in enumerate(mpo_ids)}
    candidate = sorted(candidate_mpo & set(mpo_ids))
    candidate_indices = [m_index[value] for value in candidate]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    row_count = 0
    ideal_added = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("hp_id", "mp_id", "score", "pair_scope"))
            for h_index, hpo_id in enumerate(hpo_ids):
                row = np.asarray(surface[h_index], dtype=np.float64)
                for mp_id, m_position in zip(candidate, candidate_indices):
                    writer.writerow((hpo_id, mp_id, f"{row[m_position]:.10g}", "candidate"))
                    row_count += 1
                ideal_index = int(np.nanargmax(row))
                ideal_id = mpo_ids[ideal_index]
                if ideal_id not in candidate_mpo:
                    writer.writerow((hpo_id, ideal_id, f"{row[ideal_index]:.10g}", "global_ideal"))
                    row_count += 1
                    ideal_added += 1
        temporary.replace(path)
        path.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "rows": row_count,
        "candidate_mpo": len(candidate),
        "global_ideal_rows_added": ideal_added,
        "file": file_record(path),
    }


def _presence(path: Path, candidate_mpo: set[str]) -> None:
    rows = [
        {
            "mp_id": value,
            "present_in_2602_mp_ontology": "true",
            "has_benchmark_hpo_pair": "true",
        }
        for value in sorted(candidate_mpo)
    ]
    write_tsv(
        path,
        rows,
        ("mp_id", "present_in_2602_mp_ontology", "has_benchmark_hpo_pair"),
    )


def _run_mission3(
    resources: dict[str, Any], config_path: Path, output: Path
) -> None:
    if (output / "run_manifest.json").is_file():
        return
    command = [
        sys.executable,
        "-B",
        "-m",
        "cellphenorank",
        "run-parameter-free",
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(resolve(resources, resources['mission3']["source_root"]))
    subprocess.run(
        command,
        cwd=resolve(resources, resources['mission3']["project_root"]),
        env=environment,
        check=True,
    )


def _config(
    resources: dict[str, Any], source: Path, surface: Path, presence: Path, output: Path
) -> Path:
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["project"]["name"] = f"Training-free confirmation {output.name}"
    value["inputs"]["similarity"] = str(surface)
    value["inputs"]["phenodigm_mpo_presence"] = str(presence)
    value["surface_baseline"]["matrix_release"] = f"semantic_cpo_{output.name}"
    destination = output.parent / f"{output.name}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return destination


def _metrics(path: Path, experiment: str) -> dict[str, float]:
    return {
        row["metric"]: float(row["value"])
        for row in read_tsv(path)
        if row["experiment"] == experiment
    }


def _disease(path: Path, experiment: str) -> dict[str, dict[str, float]]:
    return {
        row["disease_id"]: {
            "rank": float(row["best_known_gene_rank"]),
            "rr": float(row["reciprocal_rank"]),
        }
        for row in read_tsv(path)
        if row["experiment"] == experiment
    }


def _adapter_check(adapter_path: Path, mission3_path: Path) -> dict[str, object]:
    adapter = {(row["disease_id"], row["gene_symbol"]): row for row in read_tsv(adapter_path)}
    mission3 = {(row["disease_id"], row["gene_symbol"]): row for row in read_tsv(mission3_path)}
    score_delta = 0.0
    rank_mismatch = 0
    for key in set(adapter) & set(mission3):
        score_delta = max(score_delta, abs(float(adapter[key]["score"]) - float(mission3[key]["base_score"])))
        rank_mismatch += int(int(adapter[key]["rank"]) != int(mission3[key]["base_rank"]))
    return {
        "adapter_rows": len(adapter),
        "mission3_rows": len(mission3),
        "shared_rows": len(set(adapter) & set(mission3)),
        "rank_mismatch_count": rank_mismatch,
        "maximum_score_difference": score_delta,
    }


def _bootstrap(
    arms: dict[str, dict[str, dict[str, float]]], repeats: int, seed: int
) -> list[dict[str, object]]:
    comparisons = (("O0", "P0"), ("O0", "E0"), ("O1", "P1"), ("O1", "E1"))
    diseases = sorted(arms["P0"])
    random = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for left, right in comparisons:
        delta = np.asarray([arms[left][value]["rr"] - arms[right][value]["rr"] for value in diseases])
        samples = np.empty(repeats, dtype=np.float64)
        for index in range(repeats):
            samples[index] = delta[random.integers(0, len(delta), len(delta))].mean()
        rows.append(
            {
                "comparison": f"{left}-{right}",
                "mean_delta_mrr": float(delta.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "wins": int(np.count_nonzero(delta > 0)),
                "ties": int(np.count_nonzero(delta == 0)),
                "losses": int(np.count_nonzero(delta < 0)),
                "repeats": repeats,
                "seed": seed,
            }
        )
    return rows


def run_confirmation(
    resources: dict[str, Any], registry: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources)
    output = root / "r5_confirmation"
    s, o, hpo_ids, mpo_ids, selected_surface = _surface(resources, registry)
    task = load_task(resources)
    candidate_mpo = _candidate_mpo(task)
    exports = output / "surface_exports"
    p_path = exports / "P.tsv.gz"
    o_path = exports / "O.tsv.gz"
    p_export = export_surface(p_path, s, hpo_ids, mpo_ids, candidate_mpo)
    o_export = export_surface(o_path, o, hpo_ids, mpo_ids, candidate_mpo)
    presence = exports / "candidate_mpo_presence.tsv"
    _presence(presence, candidate_mpo)
    source_config = resolve(resources, resources['mission3']["config_2602"])
    p_output = output / "P"
    o_output = output / "O"
    p_config = _config(resources, source_config, p_path, presence, p_output)
    o_config = _config(resources, source_config, o_path, presence, o_output)
    _run_mission3(resources, p_config, p_output)
    _run_mission3(resources, o_config, o_output)

    original = resolve(resources, resources['mission3']["baseline_output_2602"])
    arm_sources = {
        "E0": (original, "A0_surface"),
        "E1": (original, "A4_extended_constrained"),
        "P0": (p_output, "A0_surface"),
        "P1": (p_output, "A4_extended_constrained"),
        "O0": (o_output, "A0_surface"),
        "O1": (o_output, "A4_extended_constrained"),
    }
    metric_rows: list[dict[str, object]] = []
    disease_arms: dict[str, dict[str, dict[str, float]]] = {}
    for arm, (source, experiment) in arm_sources.items():
        metrics = _metrics(source / "benchmark/benchmark_metrics.tsv", experiment)
        disease_arms[arm] = _disease(source / "benchmark/disease_level_metrics.tsv", experiment)
        for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "median_rank"):
            metric_rows.append({"arm": arm, "metric": metric, "value": metrics[metric]})
    write_tsv(output / "six_arm_metrics.tsv", metric_rows, ("arm", "metric", "value"))
    bootstrap = _bootstrap(
        disease_arms,
        int(registry["controls"]["bootstrap_repeats"]),
        int(json.loads(resolve(resources, f"{resources['module_root']}/configs/seeds.json").read_text())["bootstrap"]),
    )
    write_tsv(
        output / "paired_deltas_and_intervals.tsv",
        bootstrap,
        ("comparison", "mean_delta_mrr", "ci_low", "ci_high", "wins", "ties", "losses", "repeats", "seed"),
    )
    adapter_checks = {
        "P0": _adapter_check(
            root / "r1_nodes/P0/gene_rankings.tsv", p_output / "baseline/base_gene_rankings.tsv"
        ),
        "O0": _adapter_check(
            root / "r4_fusion" / selected_surface.rsplit("__", 1)[0] / selected_surface.rsplit("__", 1)[1] / "gene_rankings.tsv",
            o_output / "baseline/base_gene_rankings.tsv",
        ),
    }
    if any(value["rank_mismatch_count"] for value in adapter_checks.values()):
        raise ValueError("mission3 adapter rank mismatch")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "selected_surface": selected_surface,
        "exports": {"P": p_export, "O": o_export, "presence": file_record(presence)},
        "adapter_checks": adapter_checks,
        "six_arm_metrics": file_record(output / "six_arm_metrics.tsv"),
        "paired_intervals": file_record(output / "paired_deltas_and_intervals.tsv"),
    }
    write_json(output / "confirmation_manifest.json", manifest)
    return manifest

