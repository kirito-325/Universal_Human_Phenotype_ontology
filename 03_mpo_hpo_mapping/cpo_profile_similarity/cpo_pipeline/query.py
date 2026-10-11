from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cpo_mapping.io import read_tsv

from .config import output_root
from .mapping import MappingContext, _pair_details


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def query_term(config: dict[str, Any], side: str, term_id: str, limit: int = 20) -> dict[str, Any]:
    root = output_root(config)
    side = side.upper()
    if side not in {"HPO", "MPO"}:
        raise ValueError("side must be HPO or MPO")
    _, policies = read_tsv(root / f"indices/{side.lower()}_index.tsv")
    policy = next((row for row in policies if row["term_id"] == term_id), None)
    if policy is None:
        raise KeyError(f"term is not in the {side} index: {term_id}")
    filename = "hpo_to_mpo.parquet" if side == "HPO" else "mpo_to_hpo.parquet"
    frame = pd.read_parquet(root / f"mapping/{filename}")
    column = "hpo_id" if side == "HPO" else "mpo_id"
    selected = (
        frame[frame[column] == term_id]
        .sort_values(["view", "rank"])
        .groupby("view", sort=True, group_keys=False)
        .head(limit)
    )
    provenance = pd.read_parquet(root / "profiles/structure_provenance.parquet")
    provenance = provenance[
        (provenance["ontology_side"] == side)
        & (provenance["term_id"] == term_id)
        & (provenance["profile_view"] == "final")
    ]
    return {
        "term": policy,
        "na_reason": None if policy["final_profile_status"] == "OK" else policy["final_profile_status"],
        "final_source_origins": sorted(set(provenance["origin"].astype(str))),
        "source_leaf_count": int(
            provenance.loc[provenance["source_leaf_id"].astype(bool), "source_leaf_id"].nunique()
        ),
        "candidates": [
            {key: _clean(value) for key, value in row.items()}
            for row in selected.to_dict("records")
        ],
    }


def explain_pair(config: dict[str, Any], hpo_id: str, mpo_id: str) -> dict[str, Any]:
    context = MappingContext(config)
    if hpo_id not in context.hpo_index:
        raise KeyError(f"HPO term is not indexed: {hpo_id}")
    if mpo_id not in context.mpo_index:
        raise KeyError(f"MPO term is not indexed: {mpo_id}")
    h_index = context.hpo_index[hpo_id]
    m_index = context.mpo_index[mpo_id]
    root = context.root
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        scores = {
            path.replace("/", "_"): _clean(handle[path][h_index, m_index])
            for path in (
                "final/shape",
                "final/hpo_to_mpo",
                "final/mpo_to_hpo",
                "final/direction_support",
                "final/direction_conflict",
                "self_only/shape",
                "inherited_only/shape",
                "aux/cl",
                "aux/go",
            )
        }
    details = (
        _pair_details(context, h_index, m_index)
        if context.h_profiles[h_index] and context.m_profiles[m_index]
        else {"status": "NO_COMPLETE_CPO"}
    )
    explanations = pd.read_parquet(root / "explanations/term_matches.parquet")
    explanations = explanations[
        (explanations["hpo_id"] == hpo_id) & (explanations["mpo_id"] == mpo_id)
    ]
    return {
        "hpo_id": hpo_id,
        "mpo_id": mpo_id,
        "scores": scores,
        "profile_details": {key: _clean(value) for key, value in details.items()},
        "selected_evidence": [
            {key: _clean(value) for key, value in row.items()}
            for row in explanations.to_dict("records")
        ],
        "evidence_status": "PRECOMPUTED" if len(explanations) else "PAIR_NOT_IN_PRECOMPUTED_TOPK",
    }


def query_disease(config: dict[str, Any], disease_id: str, view: str, limit: int = 20) -> dict[str, Any]:
    root = output_root(config)
    filename = "human_1to1.parquet" if view == "one_to_one" else "human_expanded.parquet"
    frame = pd.read_parquet(root / f"gene_prediction/{filename}")
    selected = frame[frame["disease_id"] == disease_id].sort_values("rank").head(limit)
    _, diseases = read_tsv(root / "disease_model/disease_index.tsv")
    metadata = next((row for row in diseases if row["disease_id"] == disease_id), None)
    if metadata is None:
        raise KeyError(f"disease is not indexed: {disease_id}")
    return {
        "disease": metadata,
        "ranking_view": view,
        "candidates": [
            {key: _clean(value) for key, value in row.items()}
            for row in selected.to_dict("records")
        ],
    }


def explain_gene(config: dict[str, Any], disease_id: str, gene_id: str) -> dict[str, Any]:
    root = output_root(config)
    frame = pd.read_parquet(root / "explanations/gene_evidence_chains.parquet")
    selected = frame[
        (frame["disease_id"] == disease_id) & (frame["human_gene_id"] == gene_id)
    ]
    return {
        "disease_id": disease_id,
        "human_gene_id": gene_id,
        "status": "OK" if len(selected) else "NOT_IN_PRECOMPUTED_TOP10",
        "evidence_chains": [
            {key: _clean(value) for key, value in row.items()}
            for row in selected.to_dict("records")
        ],
    }


def export_term_scores(
    config: dict[str, Any], side: str, term_id: str, dataset: str, destination: str | Path
) -> dict[str, Any]:
    root = output_root(config)
    _, h_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, m_rows = read_tsv(root / "indices/mpo_index.tsv")
    h_ids = [row["term_id"] for row in h_rows]
    m_ids = [row["term_id"] for row in m_rows]
    side = side.upper()
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        if dataset not in handle:
            raise KeyError(f"unknown dataset: {dataset}")
        if side == "HPO":
            index = h_ids.index(term_id)
            target_ids = m_ids
            values = np.asarray(handle[dataset][index, :])
            target_side = "MPO"
        elif side == "MPO":
            index = m_ids.index(term_id)
            target_ids = h_ids
            values = np.asarray(handle[dataset][:, index])
            target_side = "HPO"
        else:
            raise ValueError("side must be HPO or MPO")
    output = Path(destination)
    frame = pd.DataFrame(
        {
            "query_side": side,
            "query_id": term_id,
            "target_side": target_side,
            "target_id": target_ids,
            "dataset": dataset,
            "score": values,
            "status": np.where(np.isfinite(values), "OK", "NO_COMPLETE_CPO"),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, sep="\t", index=False)
    return {"path": str(output), "row_count": len(frame), "dataset": dataset}
