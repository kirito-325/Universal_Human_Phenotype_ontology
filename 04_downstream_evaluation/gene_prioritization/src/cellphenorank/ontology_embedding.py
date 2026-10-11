from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .io import sha256, write_json


_SYNONYM_RE = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+([A-Z]+)\b')


@dataclass(frozen=True)
class OntologyTermText:
    term_id: str
    name: str
    exact_synonyms: tuple[str, ...]
    alt_ids: tuple[str, ...]


def _unescape_obo_text(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def parse_obo_term_texts(path: str | Path) -> dict[str, OntologyTermText]:
    terms: dict[str, OntologyTermText] = {}
    block: dict[str, object] | None = None

    def commit() -> None:
        nonlocal block
        if not block or not block.get("id") or not block.get("name"):
            block = None
            return
        if block.get("obsolete"):
            block = None
            return
        term = OntologyTermText(
            term_id=str(block["id"]),
            name=str(block["name"]),
            exact_synonyms=tuple(dict.fromkeys(block.get("synonyms", []))),
            alt_ids=tuple(dict.fromkeys(block.get("alt_ids", []))),
        )
        terms[term.term_id] = term
        for alt_id in term.alt_ids:
            terms[alt_id] = term
        block = None

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                commit()
                block = {"synonyms": [], "alt_ids": [], "obsolete": False}
                continue
            if line.startswith("["):
                commit()
                continue
            if block is None:
                continue
            if line.startswith("id: "):
                block["id"] = line[4:].strip()
            elif line.startswith("name: "):
                block["name"] = line[6:].strip()
            elif line.startswith("alt_id: "):
                block["alt_ids"].append(line[8:].strip())
            elif line == "is_obsolete: true":
                block["obsolete"] = True
            else:
                match = _SYNONYM_RE.match(line)
                if match and match.group(2) == "EXACT":
                    block["synonyms"].append(_unescape_obo_text(match.group(1)))
    commit()
    return terms


def _batches(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode_sapbert_cls(
    texts: Sequence[str],
    model_path: str | Path,
    *,
    batch_size: int = 128,
    max_length: int = 64,
    device: str = "auto",
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Embedding generation requires the optional torch and transformers packages"
        ) from error

    if device == "auto":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    try:
        model.to(resolved_device)
    except RuntimeError:
        if device != "auto":
            raise
        resolved_device = "cpu"
        model.to(resolved_device)
    model.eval()
    output: list[np.ndarray] = []
    for batch in _batches(texts, batch_size):
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[:, 0, :]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        output.append(hidden.detach().cpu().numpy().astype(np.float32, copy=False))
    if not output:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    return np.concatenate(output, axis=0)


def _requested_terms(
    ontology: dict[str, OntologyTermText], requested_ids: Iterable[str]
) -> tuple[list[str], list[OntologyTermText]]:
    ids: list[str] = []
    terms: list[OntologyTermText] = []
    for term_id in sorted(set(requested_ids)):
        term = ontology.get(term_id)
        if term is None:
            continue
        ids.append(term_id)
        terms.append(term)
    return ids, terms


def _concept_vectors(
    terms: Sequence[OntologyTermText],
    model_path: str | Path,
    representation: str,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    if representation == "label_cls":
        return encode_sapbert_cls(
            [term.name for term in terms],
            model_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    if representation != "exact_synonym_centroid_cls":
        raise ValueError(f"Unsupported embedding representation: {representation}")

    aliases: list[str] = []
    spans: list[tuple[int, int]] = []
    for term in terms:
        values = list(dict.fromkeys([term.name, *term.exact_synonyms]))
        start = len(aliases)
        aliases.extend(values)
        spans.append((start, len(aliases)))
    alias_vectors = encode_sapbert_cls(
        aliases,
        model_path,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    vectors = np.vstack(
        [alias_vectors[start:end].mean(axis=0) for start, end in spans]
    ).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)


def build_embedding_artifact(
    *,
    hpo_obo: str | Path,
    mpo_obo: str | Path,
    hpo_ids: Iterable[str],
    mpo_ids: Iterable[str],
    model_path: str | Path,
    output_path: str | Path,
    representation: str,
    batch_size: int = 128,
    max_length: int = 64,
    device: str = "auto",
) -> dict[str, object]:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    hpo_source = Path(hpo_obo).resolve()
    mpo_source = Path(mpo_obo).resolve()
    model_source = Path(model_path).resolve()
    hpo_lookup = parse_obo_term_texts(hpo_source)
    mpo_lookup = parse_obo_term_texts(mpo_source)
    requested_hpo = set(hpo_ids)
    requested_mpo = set(mpo_ids)
    stored_hpo_ids, hpo_terms = _requested_terms(hpo_lookup, requested_hpo)
    stored_mpo_ids, mpo_terms = _requested_terms(mpo_lookup, requested_mpo)
    if not stored_hpo_ids or not stored_mpo_ids:
        raise ValueError("No requested HPO or MP terms were found in the supplied ontologies")

    hpo_vectors = _concept_vectors(
        hpo_terms,
        model_source,
        representation,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    mpo_vectors = _concept_vectors(
        mpo_terms,
        model_source,
        representation,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    np.savez_compressed(
        target,
        hpo_ids=np.asarray(stored_hpo_ids),
        hpo_vectors=hpo_vectors,
        mpo_ids=np.asarray(stored_mpo_ids),
        mpo_vectors=mpo_vectors,
    )
    weights = model_source / "model.safetensors"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "method": "SapBERT ontology-term embedding",
        "representation": representation,
        "pooling": "pre-pooler CLS token, L2 normalized",
        "model_path": str(model_source),
        "model_weights_sha256": sha256(weights) if weights.is_file() else "",
        "hpo_obo": {"path": str(hpo_source), "sha256": sha256(hpo_source)},
        "mpo_obo": {"path": str(mpo_source), "sha256": sha256(mpo_source)},
        "requested_hpo_count": len(requested_hpo),
        "stored_hpo_count": len(stored_hpo_ids),
        "missing_hpo_ids": sorted(requested_hpo.difference(stored_hpo_ids)),
        "requested_mpo_count": len(requested_mpo),
        "stored_mpo_count": len(stored_mpo_ids),
        "missing_mpo_ids": sorted(requested_mpo.difference(stored_mpo_ids)),
        "embedding_dimension": int(hpo_vectors.shape[1]),
        "artifact": {"path": str(target), "sha256": sha256(target)},
    }
    write_json(str(target) + ".manifest.json", manifest)
    return manifest
