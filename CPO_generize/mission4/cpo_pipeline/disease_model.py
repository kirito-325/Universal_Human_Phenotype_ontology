from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from cpo_mapping.io import file_record, read_tsv, write_json, write_tsv
from cpo_mapping.ontology import Ontology

from .config import output_root, resource_path


SCORE_DATASETS = (
    "score_evidence",
    "score_observed",
    "A",
    "B",
    "q_H",
    "q_M",
    "positive_match_coverage_H",
    "positive_match_coverage_M",
)


def _remove_redundant(terms: set[str], ontology: Ontology) -> tuple[set[str], set[str]]:
    redundant = {
        term
        for term in terms
        if any(term in ontology.ancestors[other] for other in terms if other != term)
    }
    return terms - redundant, redundant


def _weight(ontology: Ontology, term_id: str) -> float:
    return ontology.ic[term_id] / np.log(len(ontology.nodes))


def prepare_disease_model_profiles(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    output = root / "disease_model"
    output.mkdir(parents=True, exist_ok=True)
    _, hpo_index_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_index_rows = read_tsv(root / "indices/mpo_index.tsv")
    hpo_ids = {row["term_id"] for row in hpo_index_rows}
    mpo_ids = {row["term_id"] for row in mpo_index_rows}
    hp = Ontology.from_obo(
        resource_path(config, "hpo_ontology"), config["scope"]["hpo_root"], "HP:"
    )
    mp = Ontology.from_obo(
        resource_path(config, "mpo_ontology"), config["scope"]["mpo_root"], "MP:"
    )

    _, disease_rows = read_tsv(resource_path(config, "disease_hpo"))
    disease_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    negative_rows: list[dict[str, str]] = []
    excluded_disease_rows: list[dict[str, str]] = []
    disease_names: dict[str, str] = {}
    for row in disease_rows:
        disease_names[row["database_id"]] = row["disease_name"]
        if row["qualifier"] == "NOT":
            negative_rows.append(row)
        elif row["hpo_id"] not in hpo_ids:
            excluded_disease_rows.append({**row, "exclusion_reason": "outside_hpo_abnormality_index"})
        else:
            disease_grouped[row["database_id"]].append(row)
    disease_profiles: list[dict[str, object]] = []
    redundant_rows: list[dict[str, object]] = []
    disease_index_rows: list[dict[str, object]] = []
    for disease_id in sorted(disease_grouped):
        annotations = disease_grouped[disease_id]
        terms = {row["hpo_id"] for row in annotations}
        retained, redundant = _remove_redundant(terms, hp)
        retained = {term for term in retained if _weight(hp, term) > 0}
        if not retained:
            continue
        disease_index_rows.append(
            {
                "index": len(disease_index_rows),
                "disease_id": disease_id,
                "disease_name": disease_names[disease_id],
                "hpo_count": len(retained),
            }
        )
        by_term: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in annotations:
            by_term[row["hpo_id"]].append(row)
        for term in sorted(retained):
            rows = by_term[term]
            disease_profiles.append(
                {
                    "profile_type": "disease",
                    "profile_id": disease_id,
                    "term_id": term,
                    "weight": _weight(hp, term),
                    "reference_ids": "|".join(sorted({row["reference"] for row in rows if row["reference"]})),
                    "frequency": "|".join(sorted({row["frequency"] for row in rows if row["frequency"]})),
                    "sex": "|".join(sorted({row["sex"] for row in rows if row["sex"]})),
                    "source_version": "frozen_hpo_annotation_export",
                }
            )
        redundant_rows.extend(
            {
                "profile_type": "disease",
                "profile_id": disease_id,
                "term_id": term,
                "reason": "redundant_ancestor",
            }
            for term in sorted(redundant)
        )

    model_path = resource_path(config, "model_mpo")
    model_raw: dict[str, list[list[str]]] = defaultdict(list)
    with model_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 7:
                continue
            model_raw[row[6]].append(row)
    model_profiles: list[dict[str, object]] = []
    model_index_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    excluded_model_annotations = 0
    for genotype_id in sorted(model_raw):
        rows = model_raw[genotype_id]
        raw_terms = {row[3] for row in rows if row[3] in mpo_ids}
        excluded_model_annotations += sum(row[3] not in mpo_ids for row in rows)
        retained, redundant = _remove_redundant(raw_terms, mp)
        retained = {term for term in retained if _weight(mp, term) > 0}
        if not retained:
            continue
        marker_ids = sorted(
            {
                marker
                for row in rows
                for marker in row[5].split("|")
                if marker
            }
        )
        model_status = "single_gene_model" if len(marker_ids) == 1 else "joint_genotype"
        model_index_rows.append(
            {
                "index": len(model_index_rows),
                "model_profile_id": genotype_id,
                "genotype_id": genotype_id,
                "allelic_composition": "|".join(sorted({row[0] for row in rows})),
                "allele_ids": "|".join(sorted({row[1] for row in rows})),
                "strain_background": "|".join(sorted({row[2] for row in rows})),
                "perturbed_mouse_gene_ids": "|".join(marker_ids),
                "model_status": model_status,
                "mpo_count": len(retained),
            }
        )
        for term in sorted(retained):
            term_rows = [row for row in rows if row[3] == term]
            model_profiles.append(
                {
                    "profile_type": "model",
                    "profile_id": genotype_id,
                    "term_id": term,
                    "weight": _weight(mp, term),
                    "reference_ids": "|".join(sorted({row[4] for row in term_rows if row[4]})),
                    "frequency": "",
                    "sex": "unknown",
                    "source_version": "MGI_PhenoGenoMP.rpt",
                }
            )
        redundant_rows.extend(
            {
                "profile_type": "model",
                "profile_id": genotype_id,
                "term_id": term,
                "reason": "redundant_ancestor",
            }
            for term in sorted(redundant)
        )
        if model_status == "joint_genotype":
            joint_rows.append(model_index_rows[-1])

    profiles = pd.DataFrame([*disease_profiles, *model_profiles])
    profiles.to_parquet(output / "profiles.parquet", index=False)
    pd.DataFrame(disease_profiles).to_parquet(output / "disease_profiles.parquet", index=False)
    pd.DataFrame(model_profiles).to_parquet(output / "model_profiles.parquet", index=False)
    pd.DataFrame(redundant_rows).to_parquet(output / "redundant_annotations.parquet", index=False)
    pd.DataFrame(negative_rows).to_parquet(output / "negative_disease_annotations.parquet", index=False)
    pd.DataFrame(excluded_disease_rows).to_parquet(output / "excluded_disease_annotations.parquet", index=False)
    write_tsv(output / "disease_index.tsv", tuple(disease_index_rows[0]), disease_index_rows)
    write_tsv(output / "model_index.tsv", tuple(model_index_rows[0]), model_index_rows)
    write_tsv(
        root / "gene_prediction/joint_genotype_candidates.tsv",
        tuple(model_index_rows[0]),
        joint_rows,
    )
    pd.DataFrame(joint_rows, columns=tuple(model_index_rows[0])).to_parquet(
        root / "gene_prediction/joint_genotype_candidates.parquet", index=False
    )
    result = {
        "status": "complete",
        "disease_count": len(disease_index_rows),
        "model_count": len(model_index_rows),
        "single_gene_model_count": sum(row["model_status"] == "single_gene_model" for row in model_index_rows),
        "joint_model_count": len(joint_rows),
        "disease_profile_annotation_count": len(disease_profiles),
        "model_profile_annotation_count": len(model_profiles),
        "negative_disease_annotation_count": len(negative_rows),
        "excluded_disease_annotation_count": len(excluded_disease_rows),
        "excluded_model_annotation_count": excluded_model_annotations,
    }
    write_json(output / "profile_manifest.json", result)
    return result


def _profile_arrays(
    profiles: pd.DataFrame,
    index_rows: list[dict[str, str]],
    term_index: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    grouped_terms: dict[str, list[int]] = defaultdict(list)
    grouped_weights: dict[str, list[float]] = defaultdict(list)
    for row in profiles.itertuples(index=False):
        grouped_terms[str(row.profile_id)].append(term_index[str(row.term_id)])
        grouped_weights[str(row.profile_id)].append(float(row.weight))
    id_key = "disease_id" if "disease_id" in index_rows[0] else "model_profile_id"
    return (
        [np.asarray(grouped_terms[row[id_key]], dtype=np.int32) for row in index_rows],
        [np.asarray(grouped_weights[row[id_key]], dtype=np.float64) for row in index_rows],
    )


def _flat_profiles(profiles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray([i for i, values in enumerate(profiles) if len(values)], dtype=np.int32)
    lengths = np.asarray([len(profiles[i]) for i in active], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.asarray([value for i in active for value in profiles[i]], dtype=np.int32)
    return active, offsets, flat


def _weighted_incidence(
    profiles: list[np.ndarray], weights: list[np.ndarray], term_count: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    totals = np.zeros(len(profiles), dtype=np.float64)
    for index, (terms, term_weights) in enumerate(zip(profiles, weights)):
        rows.extend([index] * len(terms))
        columns.extend(map(int, terms))
        values.extend(map(float, term_weights))
        totals[index] = float(term_weights.sum())
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(profiles), term_count), dtype=np.float64
    )
    return matrix, totals


def _create_score_file(path: Path, diseases: int, models: int, config: dict[str, Any]) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.attrs["schema_version"] = 2
    handle.attrs["disease_count"] = diseases
    handle.attrs["model_count"] = models
    chunks = (
        min(int(config["disease_model"]["disease_block_size"]), diseases),
        min(int(config["disease_model"]["model_block_size"]), models),
    )
    for name in SCORE_DATASETS:
        handle.create_dataset(
            name,
            shape=(diseases, models),
            dtype="float32",
            chunks=chunks,
            compression="lzf",
            shuffle=True,
            fillvalue=np.nan,
        )
    return handle


def run_disease_model(config: dict[str, Any], resume: bool = True) -> dict[str, object]:
    root = output_root(config)
    profile_manifest = root / "disease_model/profile_manifest.json"
    if not profile_manifest.is_file():
        prepare_disease_model_profiles(config)
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    hpo_index = {row["term_id"]: int(row["index"]) for row in hpo_rows}
    mpo_index = {row["term_id"]: int(row["index"]) for row in mpo_rows}
    disease_profiles_df = pd.read_parquet(root / "disease_model/disease_profiles.parquet")
    model_profiles_df = pd.read_parquet(root / "disease_model/model_profiles.parquet")
    disease_profiles, disease_weights = _profile_arrays(
        disease_profiles_df, disease_rows, hpo_index
    )
    model_profiles, model_weights = _profile_arrays(
        model_profiles_df, model_rows, mpo_index
    )
    disease_incidence, disease_totals = _weighted_incidence(
        disease_profiles, disease_weights, len(hpo_rows)
    )
    model_incidence, model_totals = _weighted_incidence(
        model_profiles, model_weights, len(mpo_rows)
    )
    disease_active, disease_offsets, disease_flat = _flat_profiles(disease_profiles)
    model_active, model_offsets, model_flat = _flat_profiles(model_profiles)
    score_path = root / "disease_model/scores/disease_model_scores.h5"
    summary_path = root / "disease_model/scores/scoring_manifest.json"
    if resume and score_path.is_file() and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            return {**previous, "resumed_without_recompute": True}
    started = time.monotonic()
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as term_handle:
        term_scores = term_handle["final/shape"]
        with tempfile.TemporaryDirectory(prefix="disease_model_v2_") as temporary:
            temporary_root = Path(temporary)
            r_model = np.memmap(
                temporary_root / "r_model.dat",
                mode="w+",
                dtype="float32",
                shape=(len(hpo_rows), len(model_rows)),
            )
            r_model[:] = np.nan
            for start in range(0, len(hpo_rows), 16):
                end = min(len(hpo_rows), start + 16)
                term_block = term_scores[start:end, :]
                selected = term_block[:, model_flat]
                r_model[start:end, model_active] = np.fmax.reduceat(
                    selected, model_offsets[:-1], axis=1
                )
            r_model.flush()

            r_disease = np.memmap(
                temporary_root / "r_disease.dat",
                mode="w+",
                dtype="float32",
                shape=(len(mpo_rows), len(disease_rows)),
            )
            r_disease[:] = np.nan
            for start in range(0, len(mpo_rows), 16):
                end = min(len(mpo_rows), start + 16)
                term_block = term_scores[:, start:end]
                selected = term_block[disease_flat, :]
                r_disease[start:end, disease_active] = np.fmax.reduceat(
                    selected, disease_offsets[:-1], axis=0
                ).T
            r_disease.flush()

            output = _create_score_file(
                score_path, len(disease_rows), len(model_rows), config
            )
            model_block = int(config["disease_model"]["model_block_size"])
            for start in range(0, len(model_rows), model_block):
                end = min(len(model_rows), start + model_block)
                values = np.asarray(r_model[:, start:end], dtype=np.float64)
                finite = np.isfinite(values)
                positive = finite & (values > 0)
                numerator = disease_incidence @ np.nan_to_num(values, nan=0.0)
                denominator = disease_incidence @ finite.astype(np.float64)
                positive_weight = disease_incidence @ positive.astype(np.float64)
                a = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan),
                    where=denominator > 0,
                )
                q_h = np.divide(
                    denominator,
                    disease_totals[:, None],
                    out=np.zeros_like(denominator),
                    where=disease_totals[:, None] > 0,
                )
                pos_h = np.divide(
                    positive_weight,
                    disease_totals[:, None],
                    out=np.zeros_like(positive_weight),
                    where=disease_totals[:, None] > 0,
                )
                output["A"][:, start:end] = a
                output["q_H"][:, start:end] = q_h
                output["positive_match_coverage_H"][:, start:end] = pos_h

            disease_block = int(config["disease_model"]["disease_block_size"])
            for start in range(0, len(disease_rows), disease_block):
                end = min(len(disease_rows), start + disease_block)
                values = np.asarray(r_disease[:, start:end], dtype=np.float64)
                finite = np.isfinite(values)
                positive = finite & (values > 0)
                numerator = model_incidence @ np.nan_to_num(values, nan=0.0)
                denominator = model_incidence @ finite.astype(np.float64)
                positive_weight = model_incidence @ positive.astype(np.float64)
                b = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan),
                    where=denominator > 0,
                ).T
                q_m = np.divide(
                    denominator,
                    model_totals[:, None],
                    out=np.zeros_like(denominator),
                    where=model_totals[:, None] > 0,
                ).T
                pos_m = np.divide(
                    positive_weight,
                    model_totals[:, None],
                    out=np.zeros_like(positive_weight),
                    where=model_totals[:, None] > 0,
                ).T
                output["B"][start:end, :] = b
                output["q_M"][start:end, :] = q_m
                output["positive_match_coverage_M"][start:end, :] = pos_m
                a = output["A"][start:end, :]
                q_h = output["q_H"][start:end, :]
                valid = np.isfinite(a) & np.isfinite(b)
                observed = np.where(valid, (a + b) / 2, np.nan)
                evidence = np.where(valid, (q_h * a + q_m * b) / 2, np.nan)
                output["score_observed"][start:end, :] = observed
                output["score_evidence"][start:end, :] = evidence
            output.flush()
            output.close()
            del r_model, r_disease
    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disease_count": len(disease_rows),
        "model_count": len(model_rows),
        "datasets": list(SCORE_DATASETS),
        "runtime_seconds": time.monotonic() - started,
        "score_file": file_record(score_path),
        "direction_status": "NOT_COMPUTED_FULL_MATRIX; generated later for selected fixed structural best matches",
        "resumed_without_recompute": False,
    }
    write_json(summary_path, result)
    return result
