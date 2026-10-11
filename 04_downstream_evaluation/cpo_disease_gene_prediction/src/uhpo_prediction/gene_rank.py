from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cpo_mapping.candidates import ParquetRecordWriter, _rank, _schema
from cpo_mapping.io import file_record, read_tsv, write_json, write_tsv

from cpo_pipeline.config import output_root, resource_path


GENE_FIELDS = (
    "disease_id",
    "rank",
    "gene_id",
    "gene_symbol",
    "score",
    "best_model_id",
    "n_best_model_ties",
    "model_count",
    "orthology_type",
    "orthology_group_id",
    "source_mouse_gene_ids",
    "orthology_ambiguous",
    "rank_min",
    "rank_max",
    "tie_count",
)
GENE_SCHEMA = _schema(
    GENE_FIELDS,
    {"score"},
    {
        "rank",
        "n_best_model_ties",
        "model_count",
        "rank_min",
        "rank_max",
        "tie_count",
    },
)


def _rank_interval_from_sorted(finite: np.ndarray, value: float) -> tuple[int, int, int]:
    if not np.isfinite(value) or value <= 0 or not len(finite):
        return 0, 0, 0
    rank_min = 1 + len(finite) - int(np.searchsorted(finite, value, side="right"))
    rank_max = len(finite) - int(np.searchsorted(finite, value, side="left"))
    return rank_min, rank_max, rank_max - rank_min + 1


def _rank_interval(scores: np.ndarray, value: float) -> tuple[int, int, int]:
    finite = np.sort(scores[np.isfinite(scores) & (scores > 0)])
    return _rank_interval_from_sorted(finite, value)


def _groups_to_flat(groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(values) for values in groups], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.asarray([value for group in groups for value in group], dtype=np.int32)
    return offsets, flat


def _orthology(config: dict[str, Any]) -> tuple[dict[str, dict[str, str]], dict[str, set[str]], dict[str, str]]:
    _, rows = read_tsv(resource_path(config, "mouse_human_orthology"))
    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: {"mouse": {}, "human": {}}
    )
    for row in rows:
        group = groups[row["DB Class Key"]]
        if row["NCBI Taxon ID"] == "10090" and row["Mouse MGI ID"]:
            group["mouse"][row["Mouse MGI ID"]] = row["Symbol"]
        elif row["NCBI Taxon ID"] == "9606" and row["HGNC ID"]:
            group["human"][row["HGNC ID"]] = row["Symbol"]
    human_meta: dict[str, dict[str, str]] = {}
    expanded: dict[str, set[str]] = defaultdict(set)
    strict: dict[str, str] = {}
    for group_id, value in groups.items():
        mouse = value["mouse"]
        human = value["human"]
        relation_type = (
            "one_to_one"
            if len(mouse) == 1 and len(human) == 1
            else "one_to_many"
            if len(mouse) == 1 and len(human) > 1
            else "many_to_one"
            if len(mouse) > 1 and len(human) == 1
            else "many_to_many"
        )
        for human_id, symbol in human.items():
            human_meta[human_id] = {
                "symbol": symbol,
                "group_id": group_id,
                "orthology_type": relation_type,
            }
            expanded[human_id].update(mouse)
        if relation_type == "one_to_one":
            strict[next(iter(human))] = next(iter(mouse))
    return human_meta, dict(expanded), strict


def run_gene_rank(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    single_models = [
        (index, row)
        for index, row in enumerate(model_rows)
        if row["model_status"] == "single_gene_model"
    ]
    models_by_gene: dict[str, list[int]] = defaultdict(list)
    for index, row in single_models:
        models_by_gene[row["perturbed_mouse_gene_ids"]].append(index)
    mouse_genes = sorted(models_by_gene)
    mouse_gene_index = {value: index for index, value in enumerate(mouse_genes)}
    model_groups = [sorted(models_by_gene[gene]) for gene in mouse_genes]
    model_offsets, model_flat = _groups_to_flat(model_groups)

    output_root_gene = root / "gene_prediction"
    output_root_gene.mkdir(parents=True, exist_ok=True)
    mouse_score_path = output_root_gene / "mouse_gene_scores.h5"
    human_score_path = output_root_gene / "human_gene_scores.h5"
    top_k = int(config["mapping"]["top_k"])
    mouse_writer = ParquetRecordWriter(
        output_root_gene / "mouse_gene_scores.parquet", GENE_SCHEMA
    )
    human_one_writer = ParquetRecordWriter(
        output_root_gene / "human_1to1.parquet", GENE_SCHEMA
    )
    human_expanded_writer = ParquetRecordWriter(
        output_root_gene / "human_expanded.parquet", GENE_SCHEMA
    )
    started = time.monotonic()
    with h5py.File(
        root / "disease_model/scores/disease_model_scores.h5", "r"
    ) as model_handle:
        model_scores = model_handle["score_evidence"]
        with h5py.File(mouse_score_path, "w") as mouse_handle:
            mouse_dataset = mouse_handle.create_dataset(
                "score",
                shape=(len(disease_rows), len(mouse_genes)),
                dtype="float32",
                chunks=(min(32, len(disease_rows)), min(1024, len(mouse_genes))),
                compression="lzf",
                shuffle=True,
                fillvalue=np.nan,
            )
            for disease_index, disease in enumerate(disease_rows):
                scores = model_scores[disease_index, :]
                gene_scores = np.fmax.reduceat(scores[model_flat], model_offsets[:-1])
                mouse_dataset[disease_index, :] = gene_scores
                selected, ties, truncated = _rank(
                    gene_scores,
                    gene_scores,
                    mouse_genes,
                    np.ones(len(mouse_genes), dtype=bool),
                    top_k,
                )
                finite_gene_scores = np.sort(
                    gene_scores[np.isfinite(gene_scores) & (gene_scores > 0)]
                )
                for rank, gene_index in enumerate(selected, start=1):
                    models = model_groups[gene_index]
                    values = scores[models]
                    maximum = float(gene_scores[gene_index])
                    rank_min, rank_max, tie_count = _rank_interval_from_sorted(
                        finite_gene_scores, maximum
                    )
                    best_models = [
                        model
                        for model, value in zip(models, values)
                        if np.isfinite(value) and float(value) == maximum
                    ]
                    mouse_writer.add(
                        {
                            "disease_id": disease["disease_id"],
                            "rank": rank,
                            "gene_id": mouse_genes[gene_index],
                            "gene_symbol": "",
                            "score": maximum,
                            "best_model_id": model_rows[min(best_models)]["model_profile_id"],
                            "n_best_model_ties": len(best_models),
                            "model_count": len(models),
                            "orthology_type": "mouse",
                            "orthology_group_id": "",
                            "source_mouse_gene_ids": mouse_genes[gene_index],
                            "orthology_ambiguous": "false",
                            "rank_min": rank_min,
                            "rank_max": rank_max,
                            "tie_count": tie_count,
                        }
                    )
            mouse_handle.flush()

    human_meta, expanded, strict = _orthology(config)
    strict_humans = sorted(human for human, mouse in strict.items() if mouse in mouse_gene_index)
    expanded_humans = sorted(
        human
        for human, mouse_values in expanded.items()
        if set(mouse_values) & set(mouse_genes)
    )
    strict_groups = [[mouse_gene_index[strict[human]]] for human in strict_humans]
    expanded_groups = [
        sorted(mouse_gene_index[mouse] for mouse in expanded[human] if mouse in mouse_gene_index)
        for human in expanded_humans
    ]
    strict_offsets, strict_flat = _groups_to_flat(strict_groups)
    expanded_offsets, expanded_flat = _groups_to_flat(expanded_groups)
    with h5py.File(mouse_score_path, "r") as mouse_handle:
        mouse_scores = mouse_handle["score"]
        with h5py.File(human_score_path, "w") as human_handle:
            one_dataset = human_handle.create_dataset(
                "one_to_one",
                shape=(len(disease_rows), len(strict_humans)),
                dtype="float32",
                chunks=(min(32, len(disease_rows)), min(1024, len(strict_humans))),
                compression="lzf",
                shuffle=True,
                fillvalue=np.nan,
            )
            expanded_dataset = human_handle.create_dataset(
                "expanded",
                shape=(len(disease_rows), len(expanded_humans)),
                dtype="float32",
                chunks=(min(32, len(disease_rows)), min(1024, len(expanded_humans))),
                compression="lzf",
                shuffle=True,
                fillvalue=np.nan,
            )
            for disease_index, disease in enumerate(disease_rows):
                scores = mouse_scores[disease_index, :]
                one_scores = np.fmax.reduceat(scores[strict_flat], strict_offsets[:-1])
                expanded_scores = np.fmax.reduceat(
                    scores[expanded_flat], expanded_offsets[:-1]
                )
                one_dataset[disease_index, :] = one_scores
                expanded_dataset[disease_index, :] = expanded_scores
                _emit_human_candidates(
                    human_one_writer,
                    disease,
                    one_scores,
                    strict_humans,
                    strict_groups,
                    mouse_genes,
                    human_meta,
                    top_k,
                    "one_to_one",
                )
                _emit_human_candidates(
                    human_expanded_writer,
                    disease,
                    expanded_scores,
                    expanded_humans,
                    expanded_groups,
                    mouse_genes,
                    human_meta,
                    top_k,
                    "expanded",
                )
            human_handle.flush()

    write_tsv(
        output_root_gene / "mouse_gene_index.tsv",
        ("index", "mouse_gene_id", "model_count"),
        [
            {"index": index, "mouse_gene_id": gene, "model_count": len(model_groups[index])}
            for index, gene in enumerate(mouse_genes)
        ],
    )
    write_tsv(
        output_root_gene / "human_1to1_index.tsv",
        ("index", "human_gene_id", "symbol", "mouse_gene_id", "orthology_group_id"),
        [
            {
                "index": index,
                "human_gene_id": human,
                "symbol": human_meta[human]["symbol"],
                "mouse_gene_id": strict[human],
                "orthology_group_id": human_meta[human]["group_id"],
            }
            for index, human in enumerate(strict_humans)
        ],
    )
    write_tsv(
        output_root_gene / "human_expanded_index.tsv",
        ("index", "human_gene_id", "symbol", "mouse_gene_ids", "orthology_type", "orthology_group_id"),
        [
            {
                "index": index,
                "human_gene_id": human,
                "symbol": human_meta[human]["symbol"],
                "mouse_gene_ids": "|".join(
                    mouse_genes[value] for value in expanded_groups[index]
                ),
                "orthology_type": human_meta[human]["orthology_type"],
                "orthology_group_id": human_meta[human]["group_id"],
            }
            for index, human in enumerate(expanded_humans)
        ],
    )
    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disease_count": len(disease_rows),
        "single_gene_model_count": len(single_models),
        "mouse_gene_count": len(mouse_genes),
        "strict_human_gene_count": len(strict_humans),
        "expanded_human_gene_count": len(expanded_humans),
        "mouse_candidate_rows": mouse_writer.close(),
        "human_1to1_candidate_rows": human_one_writer.close(),
        "human_expanded_candidate_rows": human_expanded_writer.close(),
        "mouse_score_file": file_record(mouse_score_path),
        "human_score_file": file_record(human_score_path),
        "runtime_seconds": time.monotonic() - started,
    }
    result["candidate_finalization"] = finalize_candidate_tables(config)
    write_json(output_root_gene / "gene_ranking_manifest.json", result)
    return result


def _emit_human_candidates(
    writer: ParquetRecordWriter,
    disease: dict[str, str],
    scores: np.ndarray,
    humans: list[str],
    groups: list[list[int]],
    mouse_genes: list[str],
    human_meta: dict[str, dict[str, str]],
    top_k: int,
    view: str,
) -> None:
    selected, ties, truncated = _rank(
        scores,
        scores,
        humans,
        np.ones(len(humans), dtype=bool),
        top_k,
    )
    finite_scores = np.sort(scores[np.isfinite(scores) & (scores > 0)])
    for rank, human_index in enumerate(selected, start=1):
        human = humans[human_index]
        sources = [mouse_genes[index] for index in groups[human_index]]
        rank_min, rank_max, tie_count = _rank_interval_from_sorted(
            finite_scores, float(scores[human_index])
        )
        writer.add(
            {
                "disease_id": disease["disease_id"],
                "rank": rank,
                "gene_id": human,
                "gene_symbol": human_meta[human]["symbol"],
                "score": float(scores[human_index]),
                "best_model_id": "SEE_MOUSE_GENE_EVIDENCE",
                "n_best_model_ties": 0,
                "model_count": 0,
                "orthology_type": human_meta[human]["orthology_type"],
                "orthology_group_id": human_meta[human]["group_id"],
                "source_mouse_gene_ids": "|".join(sources),
                "orthology_ambiguous": str(view == "expanded" and len(sources) != 1).lower(),
                "rank_min": rank_min,
                "rank_max": rank_max,
                "tie_count": tie_count,
            }
        )


def finalize_candidate_tables(config: dict[str, Any]) -> dict[str, int]:
    """Backfill exact tie intervals and the actual best model without rescoring."""
    root = output_root(config)
    _, disease_rows = read_tsv(root / "disease_model/disease_index.tsv")
    _, model_rows = read_tsv(root / "disease_model/model_index.tsv")
    _, mouse_rows = read_tsv(root / "gene_prediction/mouse_gene_index.tsv")
    disease_index = {row["disease_id"]: int(row["index"]) for row in disease_rows}
    mouse_index = {row["mouse_gene_id"]: int(row["index"]) for row in mouse_rows}
    models_by_gene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(model_rows):
        if row["model_status"] == "single_gene_model":
            models_by_gene[row["perturbed_mouse_gene_ids"]].append(index)

    mouse_path = root / "gene_prediction/mouse_gene_scores.parquet"
    mouse_frame = pd.read_parquet(mouse_path)
    human_frames = {
        "one_to_one": pd.read_parquet(root / "gene_prediction/human_1to1.parquet"),
        "expanded": pd.read_parquet(root / "gene_prediction/human_expanded.parquet"),
    }
    with h5py.File(root / "gene_prediction/mouse_gene_scores.h5", "r") as mouse_handle, h5py.File(
        root / "gene_prediction/human_gene_scores.h5", "r"
    ) as human_handle, h5py.File(
        root / "disease_model/scores/disease_model_scores.h5", "r"
    ) as model_handle:
        mouse_groups = mouse_frame.groupby("disease_id", sort=False).groups
        human_groups = {
            view: frame.groupby("disease_id", sort=False).groups
            for view, frame in human_frames.items()
        }
        block_size = 32
        for start in range(0, len(disease_rows), block_size):
            end = min(len(disease_rows), start + block_size)
            mouse_block = np.asarray(mouse_handle["score"][start:end, :])
            model_block = np.asarray(model_handle["score_evidence"][start:end, :])
            human_blocks = {
                view: np.asarray(human_handle[view][start:end, :])
                for view in human_frames
            }
            for offset, disease in enumerate(disease_rows[start:end]):
                disease_id = disease["disease_id"]
                mouse_scores = mouse_block[offset]
                if disease_id in mouse_groups:
                    finite_scores = np.sort(
                        mouse_scores[np.isfinite(mouse_scores) & (mouse_scores > 0)]
                    )
                    for row_index in mouse_groups[disease_id]:
                        value = float(mouse_frame.at[row_index, "score"])
                        rank_min, rank_max, tie_count = _rank_interval_from_sorted(
                            finite_scores, value
                        )
                        mouse_frame.at[row_index, "rank_min"] = rank_min
                        mouse_frame.at[row_index, "rank_max"] = rank_max
                        mouse_frame.at[row_index, "tie_count"] = tie_count

                model_scores = model_block[offset]
                for view, frame in human_frames.items():
                    if disease_id not in human_groups[view]:
                        continue
                    human_scores = human_blocks[view][offset]
                    finite_human_scores = np.sort(
                        human_scores[np.isfinite(human_scores) & (human_scores > 0)]
                    )
                    for row_index in human_groups[view][disease_id]:
                        value = float(frame.at[row_index, "score"])
                        rank_min, rank_max, tie_count = _rank_interval_from_sorted(
                            finite_human_scores, value
                        )
                        sources = [
                            gene
                            for gene in str(frame.at[row_index, "source_mouse_gene_ids"]).split("|")
                            if gene in mouse_index and np.isfinite(mouse_scores[mouse_index[gene]])
                        ]
                        if sources:
                            candidate_models = sorted(
                                {model for gene in sources for model in models_by_gene[gene]}
                            )
                            finite_models = [
                                model for model in candidate_models if np.isfinite(model_scores[model])
                            ]
                            if finite_models:
                                best_model_score = max(
                                    float(model_scores[model]) for model in finite_models
                                )
                                best_models = [
                                    model
                                    for model in finite_models
                                    if float(model_scores[model]) == best_model_score
                                ]
                                frame.at[row_index, "best_model_id"] = min(
                                    model_rows[model]["model_profile_id"]
                                    for model in best_models
                                )
                                frame.at[row_index, "n_best_model_ties"] = len(best_models)
                                frame.at[row_index, "model_count"] = len(candidate_models)
                        frame.at[row_index, "rank_min"] = rank_min
                        frame.at[row_index, "rank_max"] = rank_max
                        frame.at[row_index, "tie_count"] = tie_count
                    human_frames[view] = frame

    mouse_frame.to_parquet(mouse_path, index=False)
    human_frames["one_to_one"].to_parquet(
        root / "gene_prediction/human_1to1.parquet", index=False
    )
    human_frames["expanded"].to_parquet(
        root / "gene_prediction/human_expanded.parquet", index=False
    )
    result = {
        "mouse_candidate_rows": len(mouse_frame),
        "human_1to1_candidate_rows": len(human_frames["one_to_one"]),
        "human_expanded_candidate_rows": len(human_frames["expanded"]),
    }
    write_json(root / "gene_prediction/candidate_finalization_manifest.json", result)
    return result
