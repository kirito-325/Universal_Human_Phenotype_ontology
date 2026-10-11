from __future__ import annotations


import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import resolve, result_root
from .io import file_record, read_tsv, stable_digest, write_json
from .ontology import Term, parse_obo


def _documents(ids: Sequence[str], records: dict[str, Term], labels: dict[str, str]) -> list[str]:
    result: list[str] = []
    for term_id in ids:
        record = records.get(term_id)
        if record is None:
            result.append(f"Name: {labels.get(term_id, term_id)}\nExact synonyms: \nDefinition: ")
        else:
            result.append(record.document())
    return result


def _encode(
    model: Any,
    tokenizer: Any,
    documents: Sequence[str],
    batch_size: int,
    max_length: int,
    label: str,
) -> tuple[np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    truncated = 0
    maximum_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            batch_documents = list(documents[start : start + batch_size])
            lengths = tokenizer(
                batch_documents,
                add_special_tokens=True,
                truncation=False,
                return_length=True,
            )["length"]
            truncated += sum(length > max_length for length in lengths)
            maximum_tokens = max(maximum_tokens, max(lengths, default=0))
            batch = tokenizer(
                batch_documents,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            rows.append(pooled.cpu().numpy().astype(np.float32))
            if start == 0 or (start // batch_size + 1) % 50 == 0:
                print(f"{label}: encoded {min(start + batch_size, len(documents))}/{len(documents)}", flush=True)
    vectors = np.concatenate(rows) if rows else np.empty((0, model.config.hidden_size), dtype=np.float32)
    return vectors, {
        "document_count": len(documents),
        "truncated_count": truncated,
        "maximum_untruncated_tokens": maximum_tokens,
    }


def encode_all(resources: dict[str, Any], registry: dict[str, Any]) -> dict[str, object]:
    root = result_root(resources)
    cache = root / "cache"
    profile_manifest = cache / "profile_manifest.json"
    if not profile_manifest.is_file():
        from .data import build_profile_cache

        build_profile_cache(resources, registry)
    h_index = read_tsv(resolve(resources, resources['mission4']["hpo_index"]))
    m_index = read_tsv(resolve(resources, resources['mission4']["mpo_index"]))
    h_ids = [row["term_id"] for row in h_index]
    m_ids = [row["term_id"] for row in m_index]
    h_labels = {row["term_id"]: row["label"] for row in h_index}
    m_labels = {row["term_id"]: row["label"] for row in m_index}
    h_records, h_version = parse_obo(resolve(resources, resources["ontologies"]["hpo"]))
    m_records, m_version = parse_obo(resolve(resources, resources["ontologies"]["mpo"]))
    cl_records, cl_version = parse_obo(resolve(resources, resources["ontologies"]["cl"]))
    go_records, go_version = parse_obo(resolve(resources, resources["ontologies"]["go"]))
    h_atoms = read_tsv(cache / "hpo_atoms.tsv")
    m_atoms = read_tsv(cache / "mpo_atoms.tsv")
    cl_ids = sorted({row["cl_id"] for row in (*h_atoms, *m_atoms)})
    go_ids = sorted({row["go_id"] for row in (*h_atoms, *m_atoms)})
    cpo_pairs = sorted({(row["cl_id"], row["go_id"]) for row in (*h_atoms, *m_atoms)})
    cpo_ids = [f"{cl_id}__{go_id}" for cl_id, go_id in cpo_pairs]
    cpo_documents = [
        (
            f"Cell type: {cl_records[cl_id].label}. {cl_records[cl_id].definition}.\n"
            f"Biological process: {go_records[go_id].label}. {go_records[go_id].definition}."
        )
        for cl_id, go_id in cpo_pairs
    ]
    encoder = resolve(resources, resources["encoder"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    model = AutoModel.from_pretrained(encoder, local_files_only=True)
    model.eval()
    settings = registry["semantics"]
    batch_size = int(settings["batch_size"])
    max_length = int(settings["max_length"])
    started = time.monotonic()
    groups = {
        "hpo": (h_ids, _documents(h_ids, h_records, h_labels)),
        "mpo": (m_ids, _documents(m_ids, m_records, m_labels)),
        "cl": (cl_ids, _documents(cl_ids, cl_records, {})),
        "go": (go_ids, _documents(go_ids, go_records, {})),
        "cpo": (cpo_ids, cpo_documents),
    }
    diagnostics: dict[str, object] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, (_ids, documents) in groups.items():
        vectors[name], diagnostics[name] = _encode(
            model, tokenizer, documents, batch_size, max_length, name
        )
    np.savez_compressed(
        cache / "semantic_embeddings.npz",
        hpo_ids=np.asarray(h_ids),
        hpo_vectors=vectors["hpo"],
        mpo_ids=np.asarray(m_ids),
        mpo_vectors=vectors["mpo"],
        cl_ids=np.asarray(cl_ids),
        cl_vectors=vectors["cl"],
        go_ids=np.asarray(go_ids),
        go_vectors=vectors["go"],
        cpo_ids=np.asarray(cpo_ids),
        cpo_vectors=vectors["cpo"],
    )
    document_hashes = {
        name: stable_digest(documents) for name, (_ids, documents) in groups.items()
    }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_enabled": False,
        "encoder": {
            "path": str(encoder),
            "revision": resources["encoder"]["revision"],
            "pooling": "attention_mask_mean",
            "normalization": "l2",
            "max_length": max_length,
            "template_id": settings["id"],
        },
        "ontology_versions": {
            "hpo": h_version,
            "mpo": m_version,
            "cl": cl_version,
            "go": go_version,
        },
        "document_hashes": document_hashes,
        "diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
        "output": file_record(cache / "semantic_embeddings.npz"),
    }
    write_json(cache / "semantic_embeddings_manifest.json", manifest)
    return manifest

