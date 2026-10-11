#!/usr/bin/env python3
"""Compare Gilda and SapBERT retrieval for the expanded cell catalog."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.match_cell_names_to_catalog import (
    ABBREVIATION_EXPANSIONS,
    CatalogEntry,
    SourceEntry,
    catalog_aliases,
    build_exact_catalog_index,
    exact_catalog_match,
    load_catalog,
    load_sources,
    normalize_name,
    source_embedding_text,
)


DEFAULT_INPUT = ROOT / "database" / "agent_standardized_cell_names_filtered.tsv"
DEFAULT_CATALOG = ROOT / "expanded" / "expanded_cell_catalog.html"
DEFAULT_OUTPUT_DIR = ROOT / "database" / "cell_retrieval_benchmark"
DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_index", "cell_name", "normalized_name", "rank", "catalog_id",
        "catalog_label", "catalog_cl_name", "score", "backend", "device",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def make_gilda_terms(catalog: list[CatalogEntry]) -> list[Any]:
    try:
        from gilda import Term
        from gilda.process import normalize
    except ImportError as exc:
        raise RuntimeError("Gilda backend requires: python3 -m pip install gilda") from exc

    terms: list[Any] = []
    for entry in catalog:
        primary = entry.cl_name or entry.label
        aliases = {entry.label, entry.cl_name, primary, *catalog_aliases(entry.label, entry.cl_name)}
        for alias in aliases:
            alias = str(alias or "").strip()
            if not alias:
                continue
            terms.append(
                Term(
                    norm_text=normalize(alias),
                    text=alias,
                    db="CL",
                    id=entry.catalog_id,
                    entry_name=primary,
                    status="name" if alias in {entry.label, entry.cl_name} else "synonym",
                    source="expanded_cell_catalog",
                )
            )
    return terms


def run_gilda(
    sources: list[SourceEntry],
    catalog: list[CatalogEntry],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ.setdefault("PYSTOW_HOME", str(ROOT / "database" / ".pystow"))
    try:
        from gilda import make_grounder
    except ImportError as exc:
        raise RuntimeError("Gilda backend requires: python3 -m pip install gilda") from exc

    started = time.perf_counter()
    terms = make_gilda_terms(catalog)
    grounder = make_grounder(terms)
    setup_seconds = time.perf_counter() - started
    catalog_by_id = {entry.catalog_id: entry for entry in catalog}
    rows: list[dict[str, Any]] = []
    query_started = time.perf_counter()
    for source in sources:
        query_texts = [source.name]
        source_key = normalize_name(source.name)
        expansion = ABBREVIATION_EXPANSIONS.get(source_key)
        if not expansion and source_key.endswith(" cell"):
            expansion = ABBREVIATION_EXPANSIONS.get(source_key[:-5].rstrip())
        if expansion:
            query_texts.append(expansion)
        matches = []
        for query_text in query_texts:
            matches.extend(grounder.ground(query_text, namespaces=["CL"]))
        by_id: dict[str, float] = {}
        for scored in matches:
            catalog_id = str(scored.term.id)
            by_id[catalog_id] = max(by_id.get(catalog_id, 0.0), float(scored.score))
        ranked = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        for rank, (catalog_id, score) in enumerate(ranked, 1):
            entry = catalog_by_id[catalog_id]
            rows.append(
                {
                    "source_index": source.source_index,
                    "cell_name": source.name,
                    "normalized_name": normalize_name(source.name),
                    "rank": rank,
                    "catalog_id": catalog_id,
                    "catalog_label": entry.label,
                    "catalog_cl_name": entry.cl_name,
                    "score": f"{score:.6f}",
                    "backend": "gilda",
                    "device": "cpu",
                }
            )
    query_seconds = time.perf_counter() - query_started
    return rows, {
        "backend": "gilda",
        "device": "cpu",
        "catalog_terms": len(terms),
        "setup_seconds": round(setup_seconds, 4),
        "query_seconds": round(query_seconds, 4),
        "queries": len(sources),
        "names_with_candidates": len({int(row["source_index"]) for row in rows}),
        "candidate_coverage": round(len({int(row["source_index"]) for row in rows}) / len(sources), 4) if sources else None,
        "queries_per_second": round(len(sources) / query_seconds, 3) if query_seconds else None,
        "gpu_required": False,
        "peak_process_rss_mb": peak_rss_mb(),
    }


def select_device(torch_module: Any, requested: str) -> tuple[Any, str]:
    if requested == "cpu":
        return torch_module.device("cpu"), "cpu"
    if requested in {"auto", "cuda"} and torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda"
    if requested in {"auto", "mps"} and hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
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
            "SapBERT backend requires: python3 -m pip install torch transformers"
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
    catalog_vectors = sapbert_encode(tokenizer, model, torch, catalog_texts, device, batch_size, max_length)
    catalog_seconds = time.perf_counter() - catalog_started
    query_started = time.perf_counter()
    query_vectors = sapbert_encode(tokenizer, model, torch, query_texts, device, batch_size, max_length)
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
        "names_with_candidates": len({int(row["source_index"]) for row in rows}),
        "candidate_coverage": round(len({int(row["source_index"]) for row in rows}) / len(sources), 4) if sources else None,
        "queries_per_second": round(len(sources) / query_seconds, 3) if query_seconds else None,
        "peak_device_memory_bytes": torch_memory_bytes(torch, device_name),
        "gpu_required": device_name in {"cuda", "mps"},
        "peak_process_rss_mb": peak_rss_mb(),
    }


def exact_recall(rows: list[dict[str, Any]], sources: list[SourceEntry], catalog: list[CatalogEntry], top_k: int) -> dict[str, Any]:
    by_source: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if int(row["rank"]) <= top_k:
            by_source[int(row["source_index"])].add(row["catalog_id"])
    exact_index = build_exact_catalog_index(catalog)
    exact_matches = {
        source.source_index: exact_catalog_match(source.name, catalog, exact_index)
        for source in sources
    }
    exact_sources = [source for source in sources if exact_matches[source.source_index] is not None]
    hits = sum(
        exact_matches[source.source_index].catalog_id
        in by_source.get(source.source_index, set())
        for source in exact_sources
    )
    return {
        "known_exact_names": len(exact_sources),
        "known_exact_recall_at_k": round(hits / len(exact_sources), 4) if exact_sources else None,
        "known_exact_hits": hits,
    }


def compare_backends(
    gilda_rows: list[dict[str, Any]],
    sapbert_rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    def grouped(rows: list[dict[str, Any]]) -> dict[int, list[str]]:
        output: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            if int(row["rank"]) <= top_k:
                output[int(row["source_index"])].append(row["catalog_id"])
        return output

    left, right = grouped(gilda_rows), grouped(sapbert_rows)
    common = sorted(set(left) & set(right))
    overlaps = [len(set(left[index]) & set(right[index])) / top_k for index in common]
    top1_agreement = sum(left[index][0] == right[index][0] for index in common)
    return {
        "compared_names": len(common),
        "mean_top_k_overlap": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "top1_agreement": round(top1_agreement / len(common), 4) if common else None,
        "top1_agreement_count": top1_agreement,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=("gilda", "sapbert", "both"), default="both")
    parser.add_argument("--sapbert-model", default=DEFAULT_SAPBERT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--apply-filter", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.top_k < 1 or args.batch_size < 1 or args.max_length < 1 or args.limit < 1:
        raise SystemExit("--top-k, --batch-size, --max-length, and --limit must be >= 1")
    catalog = load_catalog(args.catalog.resolve())
    sources, _ = load_sources(args.input.resolve(), args.apply_filter)
    sources = sources[: args.limit]
    output_dir = args.output_dir.resolve()
    backend_summaries: dict[str, Any] = {}
    backend_rows: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}

    if args.backend in {"gilda", "both"}:
        try:
            rows, summary = run_gilda(sources, catalog, args.top_k)
            backend_rows["gilda"] = rows
            backend_summaries["gilda"] = {**summary, **exact_recall(rows, sources, catalog, args.top_k)}
            write_candidates(output_dir / "gilda_candidates.tsv", rows)
        except Exception as exc:
            errors["gilda"] = f"{type(exc).__name__}: {exc}"
    if args.backend in {"sapbert", "both"}:
        try:
            rows, summary = run_sapbert(
                sources, catalog, args.top_k, args.sapbert_model,
                args.device, args.batch_size, args.max_length,
            )
            backend_rows["sapbert"] = rows
            backend_summaries["sapbert"] = {**summary, **exact_recall(rows, sources, catalog, args.top_k)}
            write_candidates(output_dir / "sapbert_candidates.tsv", rows)
        except Exception as exc:
            errors["sapbert"] = f"{type(exc).__name__}: {exc}"
    comparison = None
    if "gilda" in backend_rows and "sapbert" in backend_rows:
        comparison = compare_backends(backend_rows["gilda"], backend_rows["sapbert"], args.top_k)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "catalog": str(args.catalog.resolve()),
        "catalog_nodes": len(catalog),
        "source_names": len(sources),
        "top_k": args.top_k,
        "backend_summaries": backend_summaries,
        "comparison": comparison,
        "errors": errors,
    }
    atomic_write(output_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if backend_summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
