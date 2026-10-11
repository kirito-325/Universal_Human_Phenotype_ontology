from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from .config import output_root
from .io import read_tsv, write_tsv


def query_candidates(
    config: dict[str, Any],
    side: str,
    term_id: str,
    channel: str,
    view: str,
    limit: int = 20,
    dimension: str | None = None,
    scope: str = "self",
) -> list[dict[str, object]]:
    root = output_root(config)
    side = side.upper()
    if scope == "all-abnormal":
        path = (
            root / "candidates/hpo_all_to_mpo_all_topk.parquet"
            if side == "HPO"
            else root / "candidates/mpo_all_to_hpo_all_topk.parquet"
        )
        frame = pd.read_parquet(path)
        selected = frame[
            (frame["query_side"] == side)
            & (frame["query_id"] == term_id)
            & (frame["channel"] == channel)
            & (frame["view"] == view)
        ]
        return selected.sort_values("rank").head(limit).to_dict("records")
    if view == "branch":
        if side != "HPO":
            raise ValueError("branch queries are defined only for HPO queries")
        frame = pd.read_parquet(root / "candidates/branch_support_topk.parquet")
        selected = frame[(frame["hpo_id"] == term_id) & (frame["channel"] == channel)]
        return selected.sort_values("rank").head(limit).to_dict("records")
    if view in {"CL", "GO", "auxiliary"}:
        selected_dimension = dimension or (view if view in {"CL", "GO"} else None)
        if selected_dimension is None:
            raise ValueError("auxiliary query requires --dimension CL or GO")
        frame = pd.read_parquet(root / "candidates/auxiliary_topk.parquet")
        selected = frame[
            (frame["query_side"] == side)
            & (frame["query_id"] == term_id)
            & (frame["channel"] == channel)
            & (frame["dimension"] == selected_dimension.upper())
        ]
        return selected.sort_values("rank").head(limit).to_dict("records")
    path = (
        root / "candidates/hpo_to_mpo_topk.parquet"
        if side == "HPO"
        else root / "candidates/mpo_to_hpo_topk.parquet"
    )
    frame = pd.read_parquet(path)
    selected = frame[
        (frame["query_side"] == side)
        & (frame["query_id"] == term_id)
        & (frame["channel"] == channel)
        & (frame["view"] == view)
    ]
    return selected.sort_values("rank").head(limit).to_dict("records")


def explain_candidate(
    config: dict[str, Any],
    hpo_id: str,
    mpo_id: str,
    channel: str,
    view: str,
    score_type: str = "shape",
    dimension: str | None = None,
    scope: str = "self",
) -> list[dict[str, object]]:
    root = output_root(config)
    if scope == "all-abnormal":
        frame = pd.read_parquet(
            root / "explanations/all_abnormal_branch_matches.parquet"
        )
        selected = frame[
            (frame["hpo_id"] == hpo_id)
            & (frame["mpo_id"] == mpo_id)
            & (frame["channel"] == channel)
            & (frame["score_type"] == score_type)
        ]
        return selected.to_dict("records")
    if view == "branch":
        frame = pd.read_parquet(root / "explanations/branch_matches.parquet")
        selected = frame[
            (frame["hpo_id"] == hpo_id)
            & (frame["mpo_ancestor_id"] == mpo_id)
            & (frame["channel"] == channel)
        ]
    elif view in {"CL", "GO", "auxiliary"}:
        selected_dimension = dimension or (view if view in {"CL", "GO"} else None)
        if selected_dimension is None:
            raise ValueError("auxiliary explanation requires --dimension CL or GO")
        frame = pd.read_parquet(root / "explanations/auxiliary_matches.parquet")
        selected = frame[
            (frame["hpo_id"] == hpo_id)
            & (frame["mpo_id"] == mpo_id)
            & (frame["channel"] == channel)
            & (frame["dimension"] == selected_dimension.upper())
        ]
    else:
        frame = pd.read_parquet(root / "explanations/best_cpo_matches.parquet")
        selected = frame[
            (frame["hpo_id"] == hpo_id)
            & (frame["mpo_id"] == mpo_id)
            & (frame["channel"] == channel)
            & (frame["score_type"] == score_type)
        ]
    return selected.to_dict("records")


def export_matrix(
    config: dict[str, Any],
    side: str,
    term_id: str | None,
    channel: str,
    score_type: str,
    output: str | Path,
    pairs_file: str | Path | None = None,
    scope: str = "self",
) -> int:
    root = output_root(config)
    if scope == "all-abnormal":
        _, hpo_rows = read_tsv(root / "indices/hpo_all_abnormal_index.tsv")
        _, mpo_rows = read_tsv(root / "indices/mpo_abnormal_index.tsv")
        matrix_path = root / "matrices/hpo_mpo_all_abnormal_branch.h5"
    else:
        _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
        _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
        matrix_path = root / "matrices/hpo_mpo_cpo.h5"
    hpo_ids = [row["term_id"] for row in hpo_rows]
    mpo_ids = [row["term_id"] for row in mpo_rows]
    hpo_index = {term: index for index, term in enumerate(hpo_ids)}
    mpo_index = {term: index for index, term in enumerate(mpo_ids)}
    rows: list[dict[str, object]] = []
    with h5py.File(matrix_path, "r") as handle:
        dataset = handle[channel][score_type]
        if pairs_file:
            _, pairs = read_tsv(pairs_file)
            for pair in pairs:
                hpo_id = pair["hpo_id"]
                mpo_id = pair["mpo_id"]
                value = float(dataset[hpo_index[hpo_id], mpo_index[mpo_id]])
                rows.append(
                    {
                        "hpo_id": hpo_id,
                        "mpo_id": mpo_id,
                        "channel": channel,
                        "score_type": score_type,
                        "score": "NA" if math.isnan(value) else value,
                    }
                )
        elif side.upper() == "HPO":
            if term_id not in hpo_index:
                raise KeyError(f"HPO term is absent from index: {term_id}")
            values = dataset[hpo_index[str(term_id)], :]
            rows.extend(
                {
                    "hpo_id": term_id,
                    "mpo_id": mpo_id,
                    "channel": channel,
                    "score_type": score_type,
                    "score": "NA" if math.isnan(float(value)) else float(value),
                }
                for mpo_id, value in zip(mpo_ids, values)
            )
        else:
            if term_id not in mpo_index:
                raise KeyError(f"MPO term is absent from index: {term_id}")
            values = dataset[:, mpo_index[str(term_id)]]
            rows.extend(
                {
                    "hpo_id": hpo_id,
                    "mpo_id": term_id,
                    "channel": channel,
                    "score_type": score_type,
                    "score": "NA" if math.isnan(float(value)) else float(value),
                }
                for hpo_id, value in zip(hpo_ids, values)
            )
    return write_tsv(
        output,
        ("hpo_id", "mpo_id", "channel", "score_type", "score"),
        rows,
    )
