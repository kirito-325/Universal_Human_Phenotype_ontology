from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


QUOTED_TEXT_RE = re.compile(r'^\w+:\s+"((?:[^"\\]|\\.)*)"')
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TermText:
    term_id: str
    label: str
    definition: str
    synonyms: tuple[str, ...]

    def document(self) -> str:
        components = [self.label]
        if self.synonyms:
            components.append("Synonyms: " + "; ".join(self.synonyms[:12]))
        if self.definition:
            components.append("Definition: " + self.definition)
        return ". ".join(component for component in components if component)

    def normalized_aliases(self) -> set[str]:
        return {
            normalized
            for value in (self.label, *self.synonyms)
            if (normalized := normalize_alias(value))
        }


@dataclass
class SemanticEmbedding:
    ids: list[str]
    vectors: np.ndarray
    method: str
    source: str

    def as_dict(self) -> dict[str, np.ndarray]:
        return {term_id: self.vectors[index] for index, term_id in enumerate(self.ids)}


def _unescape_obo(value: str) -> str:
    return value.replace(r'\"', '"').replace(r"\\", "\\")


def normalize_alias(value: str) -> str:
    return " ".join(NON_ALNUM_RE.sub(" ", value.lower()).split())


def load_obo_term_text(path: str | Path, accepted_ids: set[str]) -> dict[str, TermText]:
    records: dict[str, TermText] = {}
    current: dict[str, object] | None = None
    in_term = False

    def commit() -> None:
        nonlocal current
        if not current or current.get("obsolete") or current.get("id") not in accepted_ids:
            current = None
            return
        term_id = str(current["id"])
        records[term_id] = TermText(
            term_id=term_id,
            label=str(current.get("label", "")),
            definition=str(current.get("definition", "")),
            synonyms=tuple(dict.fromkeys(current.get("synonyms", []))),  # type: ignore[arg-type]
        )
        current = None

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                commit()
                current = {"synonyms": []}
                in_term = True
                continue
            if line.startswith("["):
                commit()
                in_term = False
                continue
            if not in_term or current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["label"] = line[6:].strip()
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("def: "):
                match = QUOTED_TEXT_RE.match(line)
                if match:
                    current["definition"] = _unescape_obo(match.group(1))
            elif line.startswith("synonym: "):
                match = QUOTED_TEXT_RE.match(line)
                if match:
                    current["synonyms"].append(_unescape_obo(match.group(1)))  # type: ignore[union-attr]
    commit()
    return records


def discover_sapbert_model() -> Path | None:
    configured = os.environ.get("SAPBERT_MODEL", "")
    if configured and Path(configured).is_dir():
        return Path(configured)
    root = Path.home() / ".cache/huggingface/hub/models--cambridgeltl--SapBERT-from-PubMedBERT-fulltext/snapshots"
    if root.is_dir():
        for candidate in sorted(root.iterdir(), reverse=True):
            if (candidate / "model.safetensors").is_file() and (candidate / "vocab.txt").is_file():
                return candidate
    return None


def _reduce(vectors: np.ndarray, dimension: int, seed: int) -> np.ndarray:
    if vectors.shape[1] <= dimension:
        reduced = vectors.astype(np.float32)
    else:
        reduced = TruncatedSVD(
            n_components=dimension, algorithm="randomized", n_iter=7, random_state=seed
        ).fit_transform(vectors).astype(np.float32)
    return reduced / np.clip(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12, None)


def _sapbert_vectors(
    documents: Sequence[str], model_path: Path, batch_size: int
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            batch = tokenizer(
                list(documents[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
            output = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            rows.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def build_semantic_embeddings(
    term_ids: Sequence[str],
    texts: Mapping[str, TermText],
    mode: str = "auto",
    dimension: int = 64,
    batch_size: int = 64,
    seed: int = 20260726,
) -> SemanticEmbedding:
    ids = sorted(term_ids)
    documents = [texts[term_id].document() if term_id in texts else term_id for term_id in ids]
    model_path = discover_sapbert_model()
    selected_mode = "sapbert" if mode == "auto" and model_path else mode
    if selected_mode == "sapbert":
        if model_path is None:
            raise FileNotFoundError(
                "SapBERT was requested but no local model was found; set SAPBERT_MODEL or use --semantic-mode tfidf"
            )
        raw = _sapbert_vectors(documents, model_path, batch_size)
        vectors = _reduce(raw, dimension, seed)
        source = str(model_path.resolve())
    elif selected_mode == "tfidf" or (selected_mode == "auto" and model_path is None):
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(documents)
        vectors = _reduce(matrix, min(dimension, max(2, matrix.shape[1] - 1)), seed)
        selected_mode = "tfidf"
        source = "ontology label + definition + synonyms"
    elif selected_mode == "none":
        vectors = np.zeros((len(ids), 0), dtype=np.float32)
        source = "disabled"
    else:
        raise ValueError(f"Unsupported semantic mode: {mode}")
    return SemanticEmbedding(ids, vectors, selected_mode, source)
