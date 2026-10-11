from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .io import as_float, read_tsv, value


class SimilarityProvider(ABC):
    @abstractmethod
    def get_score(self, hpo_id: str, mpo_id: str) -> float:
        """Return a normalized score in [0, 1]."""

    @abstractmethod
    def get_evidence(self, hpo_id: str, mpo_id: str) -> dict[str, Any]:
        """Return provenance and relation evidence for one pair."""


class MatrixSimilarityProvider(SimilarityProvider):
    def __init__(
        self,
        path: str | Path,
        hpo_ids: Iterable[str] | None = None,
        mpo_ids: Iterable[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        settings = dict(config or {})
        wanted_hpo = set(hpo_ids or ())
        wanted_mpo = set(mpo_ids or ())
        score_columns = list(settings.get("score_columns", ["score", "bi_score"]))
        relation_factors = settings.get("relation_factors", {})
        self.missing_score = float(settings.get("missing_pair_score", 0.0))
        self.path = str(Path(path).resolve())
        self._scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._evidence: dict[tuple[str, str], dict[str, Any]] = {}
        for row in read_tsv(path):
            hpo_id = value(row, "hpo_id", "hp_id", "object_id")
            mpo_id = value(row, "mpo_id", "mp_id", "subject_id")
            if not hpo_id or not mpo_id:
                continue
            if wanted_hpo and hpo_id not in wanted_hpo:
                continue
            if wanted_mpo and mpo_id not in wanted_mpo:
                continue
            score_column = next((name for name in score_columns if str(row.get(name, "")).strip()), "")
            score = as_float(row.get(score_column, ""), self.missing_score)
            if score > 1.0:
                score = score / (1.0 + score)
            relation = value(row, "predicted_relation", "relation", "predicate_id", default="unknown")
            factor = float(relation_factors.get(relation, relation_factors.get(relation.lower(), 1.0)))
            score = max(0.0, min(1.0, score * factor))
            current = self._scores[hpo_id].get(mpo_id)
            if current is None or score > current:
                self._scores[hpo_id][mpo_id] = score
                self._evidence[(hpo_id, mpo_id)] = {
                    "mapped": True,
                    "score_column": score_column,
                    "raw_score": as_float(row.get(score_column, ""), self.missing_score),
                    "relation": relation,
                    "source": value(row, "source", default=self.path),
                    "version": value(row, "version"),
                    "evidence": value(row, "evidence"),
                }

    @property
    def pair_count(self) -> int:
        return sum(len(items) for items in self._scores.values())

    def scored_mpo_ids(self, hpo_id: str) -> tuple[str, ...]:
        return tuple(self._scores.get(hpo_id, ()))

    def get_score(self, hpo_id: str, mpo_id: str) -> float:
        return self._scores.get(hpo_id, {}).get(mpo_id, self.missing_score)

    def get_evidence(self, hpo_id: str, mpo_id: str) -> dict[str, Any]:
        return self._evidence.get(
            (hpo_id, mpo_id),
            {"mapped": False, "relation": "not_mapped", "source": self.path},
        )


class EmbeddingSimilarityProvider(SimilarityProvider):
    """Sparse top-k HPO-MP cosine similarities from a frozen embedding artifact."""

    def __init__(
        self,
        path: str | Path,
        hpo_ids: Iterable[str] | None = None,
        mpo_ids: Iterable[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        settings = dict(config or {})
        self.path = str(Path(path).resolve())
        self.manifest_path = self.path + ".manifest.json"
        self.missing_score = float(settings.get("missing_pair_score", 0.0))
        self.top_mpo_per_hpo = int(settings.get("top_mpo_per_hpo", 256))
        self.min_cosine = float(settings.get("min_cosine", 0.0))
        if self.top_mpo_per_hpo <= 0:
            raise ValueError("similarity.top_mpo_per_hpo must be positive")
        with np.load(self.path, allow_pickle=False) as artifact:
            stored_hpo_ids = [str(item) for item in artifact["hpo_ids"].tolist()]
            stored_mpo_ids = [str(item) for item in artifact["mpo_ids"].tolist()]
            hpo_vectors = np.asarray(artifact["hpo_vectors"], dtype=np.float32)
            mpo_vectors = np.asarray(artifact["mpo_vectors"], dtype=np.float32)
        if hpo_vectors.ndim != 2 or mpo_vectors.ndim != 2:
            raise ValueError("Embedding vectors must be two-dimensional")
        if hpo_vectors.shape[1] != mpo_vectors.shape[1]:
            raise ValueError("HPO and MP embedding dimensions do not match")
        wanted_hpo = set(hpo_ids or stored_hpo_ids)
        wanted_mpo = set(mpo_ids or stored_mpo_ids)
        hpo_index = [index for index, item in enumerate(stored_hpo_ids) if item in wanted_hpo]
        mpo_index = [index for index, item in enumerate(stored_mpo_ids) if item in wanted_mpo]
        self.available_hpo_ids = {stored_hpo_ids[index] for index in hpo_index}
        self.available_mpo_ids = {stored_mpo_ids[index] for index in mpo_index}
        self.missing_hpo_ids = wanted_hpo.difference(self.available_hpo_ids)
        self.missing_mpo_ids = wanted_mpo.difference(self.available_mpo_ids)
        self._scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._ranks: dict[tuple[str, str], int] = {}
        if hpo_index and mpo_index:
            selected_hpo_ids = [stored_hpo_ids[index] for index in hpo_index]
            selected_mpo_ids = np.asarray([stored_mpo_ids[index] for index in mpo_index])
            hp = hpo_vectors[hpo_index]
            mp = mpo_vectors[mpo_index]
            hp = hp / np.maximum(
                np.linalg.norm(hp, axis=1, keepdims=True), np.finfo(np.float32).eps
            )
            mp = mp / np.maximum(
                np.linalg.norm(mp, axis=1, keepdims=True), np.finfo(np.float32).eps
            )
            similarities = hp @ mp.T
            limit = min(self.top_mpo_per_hpo, len(selected_mpo_ids))
            for row_index, hpo_id in enumerate(selected_hpo_ids):
                values = similarities[row_index]
                ordered = np.lexsort((selected_mpo_ids, -values))[:limit].tolist()
                rank = 0
                for index in ordered:
                    cosine = float(values[index])
                    if cosine < self.min_cosine:
                        continue
                    rank += 1
                    mpo_id = str(selected_mpo_ids[index])
                    self._scores[hpo_id][mpo_id] = max(0.0, min(1.0, cosine))
                    self._ranks[(hpo_id, mpo_id)] = rank

    @property
    def pair_count(self) -> int:
        return sum(len(items) for items in self._scores.values())

    def scored_mpo_ids(self, hpo_id: str) -> tuple[str, ...]:
        return tuple(self._scores.get(hpo_id, ()))

    def get_score(self, hpo_id: str, mpo_id: str) -> float:
        return self._scores.get(hpo_id, {}).get(mpo_id, self.missing_score)

    def get_evidence(self, hpo_id: str, mpo_id: str) -> dict[str, Any]:
        score = self._scores.get(hpo_id, {}).get(mpo_id)
        if score is None:
            return {
                "mapped": False,
                "relation": "outside_embedding_top_k",
                "source": self.path,
            }
        return {
            "mapped": True,
            "relation": "embedding_cosine",
            "source": self.path,
            "cosine_similarity": score,
            "hpo_neighbor_rank": self._ranks[(hpo_id, mpo_id)],
            "top_mpo_per_hpo": self.top_mpo_per_hpo,
        }


def build_similarity_provider(
    path: str | Path,
    hpo_ids: Iterable[str],
    mpo_ids: Iterable[str],
    config: Mapping[str, Any] | None = None,
) -> SimilarityProvider:
    settings = dict(config or {})
    provider = str(settings.get("provider", "matrix"))
    if provider == "matrix":
        return MatrixSimilarityProvider(path, hpo_ids, mpo_ids, settings)
    if provider == "embedding":
        return EmbeddingSimilarityProvider(path, hpo_ids, mpo_ids, settings)
    raise ValueError(f"Unsupported similarity.provider: {provider}")
