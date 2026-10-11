"""SapBERT retrieval used by the Stage 1 cell-name mapping workflow."""

from __future__ import annotations

import resource
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from .match_cell_names_to_catalog import (
    CatalogEntry,
    SourceEntry,
    exact_catalog_match,
    normalize_name,
    source_embedding_text,
)


def select_device(torch_module: Any, requested: str) -> tuple[Any, str]:
    if requested == "cpu":
        return torch_module.device("cpu"), "cpu"
    if requested in {"auto", "cuda"} and torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda"
    if (
        requested in {"auto", "mps"}
        and hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return torch_module.device("mps"), "mps"
    if requested in {"cuda", "mps"}:
        raise RuntimeError(f"requested device {requested!r} is not available")
    return torch_module.device("cpu"), "cpu"


def torch_memory_bytes(torch_module: Any, device_name: str) -> int | None:
    if device_name == "cuda":
        return int(torch_module.cuda.max_memory_allocated())
    if device_name == "mps" and hasattr(torch_module.mps, "current_allocated_memory"):
        return int(torch_module.mps.current_allocated_memory())
    return None


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(value / (1024 * 1024), 2)
    return round(value / 1024, 2)


def sapbert_encode(
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    texts: list[str],
    device: Any,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch_module.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch).last_hidden_state[:, 0, :]
            output = torch_module.nn.functional.normalize(output, p=2, dim=1)
            chunks.append(output.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def run_sapbert(
    sources: list[SourceEntry],
    catalog: list[CatalogEntry],
    top_k: int,
    model_name: str,
    requested_device: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SapBERT retrieval requires the project torch and transformers dependencies"
        ) from exc

    device, device_name = select_device(torch, requested_device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    setup_seconds = time.perf_counter() - started
    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats()

    catalog_texts = [entry.text for entry in catalog]
    query_texts = [source_embedding_text(source.name) for source in sources]
    catalog_started = time.perf_counter()
    catalog_vectors = sapbert_encode(
        tokenizer, model, torch, catalog_texts, device, batch_size, max_length
    )
    catalog_seconds = time.perf_counter() - catalog_started
    query_started = time.perf_counter()
    query_vectors = sapbert_encode(
        tokenizer, model, torch, query_texts, device, batch_size, max_length
    )
    query_seconds = time.perf_counter() - query_started
    scores = query_vectors @ catalog_vectors.T

    rows: list[dict[str, Any]] = []
    for row_index, source in enumerate(sources):
        order = np.argsort(-scores[row_index], kind="stable")[:top_k]
        for rank, catalog_index in enumerate(order, 1):
            entry = catalog[int(catalog_index)]
            rows.append(
                {
                    "source_index": source.source_index,
                    "cell_name": source.name,
                    "normalized_name": normalize_name(source.name),
                    "rank": rank,
                    "catalog_id": entry.catalog_id,
                    "catalog_label": entry.label,
                    "catalog_cl_name": entry.cl_name,
                    "score": f"{float(scores[row_index, int(catalog_index)]):.6f}",
                    "backend": "sapbert",
                    "device": device_name,
                }
            )

    names_with_candidates = len({int(row["source_index"]) for row in rows})
    return rows, {
        "backend": "sapbert",
        "model": model_name,
        "device": device_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dimension": int(catalog_vectors.shape[1]) if catalog_vectors.ndim == 2 else 0,
        "setup_seconds": round(setup_seconds, 4),
        "catalog_embedding_seconds": round(catalog_seconds, 4),
        "query_embedding_seconds": round(query_seconds, 4),
        "queries": len(sources),
        "names_with_candidates": names_with_candidates,
        "candidate_coverage": round(names_with_candidates / len(sources), 4) if sources else None,
        "queries_per_second": round(len(sources) / query_seconds, 3) if query_seconds else None,
        "peak_device_memory_bytes": torch_memory_bytes(torch, device_name),
        "gpu_required": device_name in {"cuda", "mps"},
        "peak_process_rss_mb": peak_rss_mb(),
    }


def exact_recall(
    rows: list[dict[str, Any]],
    sources: list[SourceEntry],
    catalog: list[CatalogEntry],
    top_k: int,
) -> dict[str, Any]:
    by_source: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if int(row["rank"]) <= top_k:
            by_source[int(row["source_index"])].add(str(row["catalog_id"]))
    exact_sources = [source for source in sources if exact_catalog_match(source.name, catalog)]
    hits = sum(
        exact_catalog_match(source.name, catalog).catalog_id
        in by_source.get(source.source_index, set())
        for source in exact_sources
    )
    return {
        "known_exact_names": len(exact_sources),
        "known_exact_recall_at_k": round(hits / len(exact_sources), 4)
        if exact_sources
        else None,
        "known_exact_hits": hits,
    }
