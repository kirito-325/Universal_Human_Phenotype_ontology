#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def load_index(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    by_term: dict[str, int] = {}
    by_index: dict[int, str] = {}
    for row in read_tsv(path):
        term_id = row["term_id"].strip()
        index = int(row["index"])
        if term_id in by_term or index in by_index:
            raise ValueError(f"Duplicate term or index in {path}: {term_id}, {index}")
        by_term[term_id] = index
        by_index[index] = term_id
    return by_term, by_index


def benchmark_terms(
    diseases_path: Path,
    models_path: Path,
    phenotypes_path: Path,
) -> tuple[set[str], set[str], int]:
    hpo_ids = {
        row.get("hpo_id", row.get("hp_id", "")).strip()
        for row in read_tsv(diseases_path)
    }
    eligible_models = {
        row["model_id"].strip()
        for row in read_tsv(models_path)
        if row.get("human_gene_symbol", "").strip()
        and ";" not in row["human_gene_symbol"]
    }
    mpo_ids = {
        row.get("mpo_id", row.get("mp_id", "")).strip()
        for row in read_tsv(phenotypes_path)
        if row.get("model_id", "").strip() in eligible_models
        and row.get("status", "abnormal").strip().lower() == "abnormal"
    }
    return hpo_ids - {""}, mpo_ids - {""}, len(eligible_models)


def write_gzip_rows(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with compressed_text(compressed) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["hp_id", "mp_id", "score", "pair_scope"],
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    count += 1
    return count


class compressed_text:
    def __init__(self, compressed: gzip.GzipFile) -> None:
        import io

        self.handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.handle.flush()
        self.handle.detach()


def extract(args: argparse.Namespace) -> dict[str, object]:
    matrix_path = args.matrix.resolve()
    hpo_index_path = args.hpo_index.resolve()
    mpo_index_path = args.mpo_index.resolve()
    diseases_path = args.diseases.resolve()
    models_path = args.models.resolve()
    phenotypes_path = args.model_phenotypes.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hpo_index, _ = load_index(hpo_index_path)
    mpo_index, mpo_by_index = load_index(mpo_index_path)
    hpo_ids, candidate_mpo_ids, eligible_model_count = benchmark_terms(
        diseases_path, models_path, phenotypes_path
    )
    indexed_hpo = sorted(hpo_ids.intersection(hpo_index))
    indexed_candidate_mpo = sorted(candidate_mpo_ids.intersection(mpo_index))
    candidate_columns = np.asarray(
        [mpo_index[term_id] for term_id in indexed_candidate_mpo], dtype=np.int64
    )

    positive_mpo: set[str] = set()
    mapped_hpo: set[str] = set()
    positive_scores: list[float] = []

    def matrix_rows() -> Iterable[dict[str, object]]:
        with h5py.File(matrix_path, "r") as handle:
            if args.dataset not in handle:
                raise KeyError(f"Dataset {args.dataset!r} is absent from {matrix_path}")
            dataset = handle[args.dataset]
            expected_shape = (len(hpo_index), len(mpo_index))
            if tuple(dataset.shape) != expected_shape:
                raise ValueError(
                    f"Matrix/index shape mismatch: {dataset.shape} != {expected_shape}"
                )
            for hp_id in indexed_hpo:
                values = np.asarray(dataset[hpo_index[hp_id], :], dtype=np.float64)
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                best_score = float(np.max(values[finite]))
                best_indices = np.flatnonzero(finite & (values == best_score))
                best_mp_id = min(mpo_by_index[int(index)] for index in best_indices)
                candidate_values = values[candidate_columns]
                selected = {
                    mp_id: float(score)
                    for mp_id, score in zip(indexed_candidate_mpo, candidate_values)
                    if np.isfinite(score) and score > 0
                }
                mapped_hpo.add(hp_id)
                positive_mpo.update(selected)
                positive_scores.extend(selected.values())
                selected.setdefault(best_mp_id, best_score)
                for mp_id in sorted(selected):
                    scopes = []
                    if mp_id in candidate_mpo_ids and selected[mp_id] > 0:
                        scopes.append("candidate")
                    if mp_id == best_mp_id:
                        scopes.append("global_ideal")
                    yield {
                        "hp_id": hp_id,
                        "mp_id": mp_id,
                        "score": f"{selected[mp_id]:.9g}",
                        "pair_scope": "|".join(scopes),
                    }

    matrix_output = output_dir / "hp_mp_mappings_final_shape.tsv.gz"
    row_count = write_gzip_rows(matrix_output, matrix_rows())

    presence_output = output_dir / "candidate_mpo_presence.tsv"
    with presence_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mp_id",
                "present_in_legacy_matrix",
                "has_benchmark_hpo_pair",
                "present_in_mission4_index",
                "positive_pair_with_benchmark_hpo",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for mp_id in sorted(candidate_mpo_ids):
            has_pair = mp_id in positive_mpo
            writer.writerow(
                {
                    "mp_id": mp_id,
                    "present_in_legacy_matrix": str(has_pair).lower(),
                    "has_benchmark_hpo_pair": str(has_pair).lower(),
                    "present_in_mission4_index": str(mp_id in mpo_index).lower(),
                    "positive_pair_with_benchmark_hpo": str(has_pair).lower(),
                }
            )

    score_array = np.asarray(positive_scores, dtype=np.float64)
    quantiles = (
        {
            str(q): float(value)
            for q, value in zip(
                (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1),
                np.quantile(score_array, (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)),
            )
        }
        if len(score_array)
        else {}
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "surface": {
            "source": "mission4 CPO-mediated HPO-MPO similarity",
            "dataset": args.dataset,
            "score_range": [0.0, 1.0],
            "profile_aggregation": "bidirectional_equal_bma",
            "normalization_contract": (
                "Candidate pairs plus each benchmark HPO global MPO maximum are retained "
                "so the Exomiser current-profile ideal denominator is unchanged."
            ),
        },
        "counts": {
            "eligible_single_gene_models": eligible_model_count,
            "benchmark_hpo": len(hpo_ids),
            "benchmark_hpo_in_mission4_index": len(indexed_hpo),
            "benchmark_hpo_with_final_profile": len(mapped_hpo),
            "candidate_mpo": len(candidate_mpo_ids),
            "candidate_mpo_in_mission4_index": len(indexed_candidate_mpo),
            "candidate_mpo_with_positive_benchmark_pair": len(positive_mpo),
            "positive_candidate_pairs": len(positive_scores),
            "exported_rows_including_global_ideal": row_count,
        },
        "coverage": {
            "hpo_final_profile": len(mapped_hpo) / len(hpo_ids) if hpo_ids else 0.0,
            "mpo_index": (
                len(indexed_candidate_mpo) / len(candidate_mpo_ids)
                if candidate_mpo_ids
                else 0.0
            ),
            "mpo_positive_pair": (
                len(positive_mpo) / len(candidate_mpo_ids)
                if candidate_mpo_ids
                else 0.0
            ),
        },
        "positive_score_quantiles": quantiles,
        "inputs": {
            "matrix": file_record(matrix_path),
            "hpo_index": file_record(hpo_index_path),
            "mpo_index": file_record(mpo_index_path),
            "diseases": file_record(diseases_path),
            "models": file_record(models_path),
            "model_phenotypes": file_record(phenotypes_path),
        },
        "outputs": {
            "similarity": file_record(matrix_output),
            "mpo_presence": file_record(presence_output),
        },
    }
    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a frozen-benchmark surface from mission4 CPO HPO-MPO HDF5."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", default="final/shape")
    parser.add_argument("--hpo-index", type=Path, required=True)
    parser.add_argument("--mpo-index", type=Path, required=True)
    parser.add_argument("--diseases", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-phenotypes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(extract(build_parser().parse_args()), indent=2, sort_keys=True))
