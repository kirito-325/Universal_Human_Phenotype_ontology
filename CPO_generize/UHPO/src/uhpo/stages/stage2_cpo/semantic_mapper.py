from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .common import (
    CellCatalog,
    clean_obo_text,
    explicit_cl_references,
    load_json,
    load_obo,
    nearest_catalog_ancestor,
    ontology_ancestors,
    read_go_ids,
    stable_digest,
    term_synonyms,
    write_tsv,
)


DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ontology_context(graph, term_id: str, limit: int = 6) -> list[str]:
    values = []
    for ancestor in ontology_ancestors(graph, term_id):
        name = graph.nodes.get(ancestor, {}).get("name", "")
        if name and name not in values:
            values.append(name)
    return sorted(values)[:limit]


def go_document(graph, go_id: str) -> str:
    data = graph.nodes[go_id]
    pieces = [str(data.get("name", go_id))]
    definition = clean_obo_text(str(data.get("def", "")))
    if definition:
        pieces.append(definition)
    synonyms = term_synonyms(data, limit=4)
    if synonyms:
        pieces.append("Synonyms: " + "; ".join(synonyms))
    context = ontology_context(graph, go_id)
    if context:
        pieces.append("Broader biological processes: " + "; ".join(context))
    return ". ".join(pieces)


def cl_document(graph, catalog: CellCatalog, cl_id: str) -> str:
    node = catalog.nodes[cl_id]
    data = graph.nodes.get(cl_id, {})
    pieces = [node.catalog_label, node.cl_name]
    definition = clean_obo_text(str(data.get("def", "")))
    if definition:
        pieces.append(definition)
    synonyms = term_synonyms(data, limit=4)
    if synonyms:
        pieces.append("Synonyms: " + "; ".join(synonyms))
    parent_names = [catalog.nodes[value].catalog_label for value in catalog.ancestors(cl_id)[:4]]
    if parent_names:
        pieces.append("Broader cell classes: " + "; ".join(parent_names))
    return ". ".join(dict.fromkeys(piece for piece in pieces if piece))


class SapBertEncoder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.model_name = model_name
        self.device = choose_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, documents: Sequence[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                tokens = self.tokenizer(
                    list(documents[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                cls_vectors = self.model(**tokens).last_hidden_state[:, 0, :]
                normalized = torch.nn.functional.normalize(cls_vectors, p=2, dim=1)
                batches.append(normalized.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)


def cached_encode(
    encoder: SapBertEncoder,
    documents: Sequence[str],
    batch_size: int,
    cache_dir: Path | None,
    cache_name: str,
) -> np.ndarray:
    digest = stable_digest([encoder.model_name, str(encoder.max_length), *documents])
    cache_path = cache_dir / f"{cache_name}.{digest[:16]}.npz" if cache_dir else None
    if cache_path and cache_path.exists():
        with np.load(cache_path) as data:
            return data["embeddings"]
    embeddings = encoder.encode(documents, batch_size)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def semantic_label(
    top_score: float,
    margin: float,
    specific_min_score: float,
    specific_min_margin: float,
    general_max_score: float,
) -> str:
    if top_score >= specific_min_score and margin >= specific_min_margin:
        return "cell_type_specific_candidate"
    if top_score <= general_max_score:
        return "cell_general_candidate"
    return "undetermined"


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(values))
    if count == 0:
        return np.array([], dtype=int)
    selected = np.argpartition(values, -count)[-count:]
    return selected[np.argsort(values[selected])[::-1]]


def map_terms(args: argparse.Namespace) -> None:
    config = load_json(args.config) if args.config else {}
    semantic_config = config.get("semantic", {})
    model_name = args.model or semantic_config.get("model", DEFAULT_MODEL)
    max_length = args.max_length or int(semantic_config.get("max_length", 128))
    batch_size = args.batch_size or int(semantic_config.get("batch_size", 32))
    top_k = args.top_k or int(semantic_config.get("top_k", 10))
    specific_min_score = float(semantic_config.get("specific_min_score", 0.62))
    specific_min_margin = float(semantic_config.get("specific_min_margin", 0.04))
    general_max_score = float(semantic_config.get("general_max_score", 0.52))

    go_graph = load_obo(args.go_obo)
    cl_graph = load_obo(args.cl_obo)
    catalog = CellCatalog.from_tsv(args.catalog)

    if args.all_biological_process:
        go_ids = sorted(
            term_id
            for term_id, data in go_graph.nodes(data=True)
            if data.get("namespace") == "biological_process" and data.get("is_obsolete") != "true"
        )
    else:
        go_ids = read_go_ids(args.go_terms)
    if not go_ids:
        raise ValueError("No GO IDs were found in the input")
    invalid = [
        go_id
        for go_id in go_ids
        if go_id not in go_graph
        or go_graph.nodes[go_id].get("namespace") != "biological_process"
        or go_graph.nodes[go_id].get("is_obsolete") == "true"
    ]
    if invalid:
        raise ValueError(f"Not active biological_process terms: {invalid[:20]}")

    catalog_ids = list(catalog.nodes)
    catalog_documents = [cl_document(cl_graph, catalog, cl_id) for cl_id in catalog_ids]
    go_documents = [go_document(go_graph, go_id) for go_id in go_ids]
    encoder = SapBertEncoder(model_name, args.device, max_length)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cl_embeddings = cached_encode(
        encoder, catalog_documents, batch_size, cache_dir, "cl_catalog"
    )
    go_embeddings = cached_encode(encoder, go_documents, batch_size, cache_dir, "go_terms")
    skeleton_positions = np.array(
        [
            index
            for index, cl_id in enumerate(catalog_ids)
            if catalog.nodes[cl_id].source == "skeleton" and cl_id != "CL:0000000"
        ],
        dtype=int,
    )
    summary_rows: list[dict] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_fields = [
        "go_id",
        "go_name",
        "candidate_kind",
        "rank",
        "cl_id",
        "cl_name",
        "catalog_label",
        "score",
        "anchor_id",
        "anchor_label",
    ]
    candidate_path = output_dir / "semantic_candidates.tsv"
    candidate_temporary = candidate_path.with_suffix(".tsv.part")
    similarity_batch_size = int(semantic_config.get("similarity_batch_size", 512))
    with candidate_temporary.open("w", encoding="utf-8", newline="") as candidate_handle:
        candidate_writer = csv.DictWriter(
            candidate_handle, fieldnames=candidate_fields, delimiter="\t"
        )
        candidate_writer.writeheader()
        for block_start in range(0, len(go_ids), similarity_batch_size):
            block_end = min(len(go_ids), block_start + similarity_batch_size)
            block_similarities = (
                go_embeddings[block_start:block_end] @ cl_embeddings.T
            )
            for block_offset, scores in enumerate(block_similarities):
                go_index = block_start + block_offset
                go_id = go_ids[go_index]
                data = go_graph.nodes[go_id]
                raw_positions = top_indices(scores, top_k)
                skeleton_scores = scores[skeleton_positions]
                skeleton_rank_positions = top_indices(skeleton_scores, top_k)
                ranked_skeleton_positions = skeleton_positions[skeleton_rank_positions]

                for kind, positions in (
                    ("catalog_node", raw_positions),
                    ("skeleton_constraint", ranked_skeleton_positions),
                ):
                    for rank, position in enumerate(positions, start=1):
                        cl_id = catalog_ids[int(position)]
                        node = catalog.nodes[cl_id]
                        candidate_writer.writerow(
                            {
                                "go_id": go_id,
                                "go_name": data.get("name", ""),
                                "candidate_kind": kind,
                                "rank": rank,
                                "cl_id": cl_id,
                                "cl_name": node.cl_name,
                                "catalog_label": node.catalog_label,
                                "score": f"{float(scores[position]):.6f}",
                                "anchor_id": node.anchor_id,
                                "anchor_label": node.anchor_label,
                            }
                        )

                top_position = int(ranked_skeleton_positions[0])
                runner_up_position = (
                    int(ranked_skeleton_positions[1])
                    if len(ranked_skeleton_positions) > 1
                    else top_position
                )
                top_cl_id = catalog_ids[top_position]
                top_score = float(scores[top_position])
                competing_position = runner_up_position
                for position in ranked_skeleton_positions[1:]:
                    candidate_cl_id = catalog_ids[int(position)]
                    if (
                        candidate_cl_id not in catalog.ancestors(top_cl_id)
                        and top_cl_id not in catalog.ancestors(candidate_cl_id)
                    ):
                        competing_position = int(position)
                        break
                runner_up_score = float(scores[runner_up_position])
                competing_score = float(scores[competing_position])
                margin = top_score - competing_score
                label = semantic_label(
                    top_score,
                    margin,
                    specific_min_score,
                    specific_min_margin,
                    general_max_score,
                )

                direct_refs = explicit_cl_references(data)
                projected_refs: list[str] = []
                projection_distances: list[str] = []
                for cl_id in direct_refs:
                    projected, distance = nearest_catalog_ancestor(
                        cl_graph, catalog, cl_id
                    )
                    if (
                        projected
                        and projected != "CL:0000000"
                        and projected not in projected_refs
                    ):
                        projected_refs.append(projected)
                        projection_distances.append(f"{cl_id}>{projected}:{distance}")
                direct_constraint = catalog.common_ancestor(projected_refs)
                if len(projected_refs) == 1:
                    direct_constraint = projected_refs[0]
                if projected_refs:
                    label = "cell_type_specific"

                summary_rows.append(
                    {
                        "go_id": go_id,
                        "go_name": data.get("name", ""),
                        "semantic_classification": label,
                        "semantic_constraint_cl_id": top_cl_id,
                        "semantic_constraint_label": catalog.nodes[
                            top_cl_id
                        ].catalog_label,
                        "semantic_top_score": f"{top_score:.6f}",
                        "semantic_runner_up_score": f"{runner_up_score:.6f}",
                        "semantic_competing_score": f"{competing_score:.6f}",
                        "semantic_margin": f"{margin:.6f}",
                        "explicit_cl_ids": "|".join(direct_refs),
                        "explicit_catalog_cl_ids": "|".join(projected_refs),
                        "explicit_projection_paths": "|".join(
                            projection_distances
                        ),
                        "explicit_constraint_cl_id": direct_constraint,
                        "model": model_name,
                        "model_device": encoder.device,
                    }
                )
    candidate_temporary.replace(candidate_path)

    write_tsv(
        output_dir / "semantic_summary.tsv",
        list(summary_rows[0]),
        summary_rows,
    )
    manifest = {
        "go_terms": len(go_ids),
        "catalog_nodes": len(catalog_ids),
        "skeleton_constraints": len(skeleton_positions),
        "model": model_name,
        "device": encoder.device,
        "thresholds": {
            "specific_min_score": specific_min_score,
            "specific_min_margin": specific_min_margin,
            "general_max_score": general_max_score,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate curator-reference and SapBERT evidence for GO BP to CL constraints."
    )
    parser.add_argument("--go-obo", required=True)
    parser.add_argument("--cl-obo", required=True)
    parser.add_argument("--catalog", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--go-terms", help="TSV with a go_id column, or one GO ID per line")
    source.add_argument("--all-biological-process", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--cache-dir")
    return parser


if __name__ == "__main__":
    map_terms(build_parser().parse_args())
