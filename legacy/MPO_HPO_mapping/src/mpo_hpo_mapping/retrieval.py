from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .data import MappingExample
from .embedding import IVFPrototypeIndex, ProfileTensor
from .model import (
    DualHeadEncoder,
    RelationCalibrator,
    encode_profiles,
    encode_semantics,
    flatten_valid_prototypes,
    mask_pair_features,
    poincare_profile_score,
)


def _max_similarity(
    left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray
) -> float:
    left = left[left_mask]
    right = right[right_mask]
    if not len(left) or not len(right):
        return -1.0
    return float(np.max(left @ right.T))


def _raw_score(
    query: ProfileTensor, query_index: int, target: ProfileTensor, target_index: int
) -> float:
    left = query.prototypes[query_index][query.masks[query_index]]
    right = target.prototypes[target_index][target.masks[target_index]]
    left /= np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
    right /= np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
    return float(np.max(left @ right.T))


@dataclass
class GateCalibrator:
    scaler: StandardScaler
    classifier: LogisticRegression
    threshold: float

    def probability(self, features: np.ndarray) -> float:
        transformed = self.scaler.transform(features.reshape(1, -1))
        return float(self.classifier.predict_proba(transformed)[0, 1])


@dataclass(frozen=True)
class AncestorRankWeights:
    semantic: float
    structure: float
    mechanism: float
    relation_probability: float
    validation_mrr: float
    validation_source_mrr: dict[str, float]

    def score(self, row: Mapping[str, object]) -> float:
        structure_score = 0.5 * (
            (float(row.get("ancestor_structure_score", -1.0)) + 1.0) / 2.0
        ) + 0.5 * ((float(row.get("poincare_score", -1.0)) + 1.0) / 2.0)
        return (
            self.semantic
            * ((float(row.get("ancestor_semantic_tower_score", -1.0)) + 1.0) / 2.0)
            + self.structure * structure_score
            + self.mechanism * float(row.get("mechanism_jaccard", 0.0))
            + self.relation_probability * float(row.get("p_hp_broader", 0.0))
        )


class DualHeadRetriever:
    def __init__(
        self,
        model: DualHeadEncoder,
        hp_profiles: ProfileTensor,
        mp_profiles: ProfileTensor,
        relation_calibrator: RelationCalibrator,
        ivf_clusters: int = 96,
        seed: int = 20260726,
    ) -> None:
        self.model = model
        self.hp_profiles = hp_profiles
        self.mp_profiles = mp_profiles
        self.hp_index = hp_profiles.index
        self.mp_index = mp_profiles.index
        self.relation = relation_calibrator

        self.hp_exact = encode_profiles(model, hp_profiles, "hp", "exact")
        self.hp_ancestor = encode_profiles(model, hp_profiles, "hp", "ancestor")
        self.mp_exact = encode_profiles(model, mp_profiles, "mp", "exact")
        self.mp_ancestor = encode_profiles(model, mp_profiles, "mp", "ancestor")
        self.synthetic_exact = encode_profiles(model, hp_profiles, "mp", "exact")
        self.synthetic_ancestor = encode_profiles(model, hp_profiles, "mp", "ancestor")
        self.hp_semantic_exact = encode_semantics(model, hp_profiles, "hp", "exact")
        self.hp_semantic_ancestor = encode_semantics(
            model, hp_profiles, "hp", "ancestor"
        )
        self.mp_semantic_exact = encode_semantics(model, mp_profiles, "mp", "exact")
        self.mp_semantic_ancestor = encode_semantics(
            model, mp_profiles, "mp", "ancestor"
        )
        self.synthetic_semantic_exact = encode_semantics(
            model, hp_profiles, "mp", "exact"
        )
        self.synthetic_semantic_ancestor = encode_semantics(
            model, hp_profiles, "mp", "ancestor"
        )
        self.exact_fusion_weights = model.fusion_weight_dict("exact")
        self.ancestor_fusion_weights = model.fusion_weight_dict("ancestor")

        exact_vectors, exact_terms = flatten_valid_prototypes(
            self.hp_exact, hp_profiles.masks
        )
        ancestor_vectors, ancestor_terms = flatten_valid_prototypes(
            self.hp_ancestor, hp_profiles.masks
        )
        raw_vectors, raw_terms = flatten_valid_prototypes(
            hp_profiles.prototypes, hp_profiles.masks
        )
        self.exact_index = IVFPrototypeIndex(
            exact_vectors, exact_terms, clusters=ivf_clusters, seed=seed
        )
        self.ancestor_index = IVFPrototypeIndex(
            ancestor_vectors, ancestor_terms, clusters=ivf_clusters, seed=seed + 1
        )
        self.raw_index = IVFPrototypeIndex(
            raw_vectors, raw_terms, clusters=ivf_clusters, seed=seed + 2
        )
        self.semantic_index = None
        if hp_profiles.semantic_vectors.shape[1]:
            self.semantic_index = IVFPrototypeIndex(
                hp_profiles.semantic_vectors,
                np.arange(len(hp_profiles.ids), dtype=np.int32),
                clusters=ivf_clusters,
                seed=seed + 3,
            )
        self.alias_index: dict[str, set[int]] = {}
        for hp_position, hp_id in enumerate(hp_profiles.ids):
            for alias in hp_profiles.aliases.get(hp_id, set()):
                self.alias_index.setdefault(alias, set()).add(hp_position)

    def _query(
        self, source: str, index: int
    ) -> tuple[ProfileTensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if source == "hp":
            return (
                self.hp_profiles,
                self.synthetic_exact[index],
                self.synthetic_ancestor[index],
                self.synthetic_semantic_exact[index],
                self.synthetic_semantic_ancestor[index],
            )
        return (
            self.mp_profiles,
            self.mp_exact[index],
            self.mp_ancestor[index],
            self.mp_semantic_exact[index],
            self.mp_semantic_ancestor[index],
        )

    def retrieve(
        self,
        source: str,
        query_index: int,
        retrieve_k: int = 80,
        nprobe: int = 8,
        excluded_hp_ids: set[str] | None = None,
        forced_hp_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        excluded_hp_ids = excluded_hp_ids or set()
        forced_hp_ids = forced_hp_ids or set()
        excluded_indices = {
            self.hp_index[term_id]
            for term_id in excluded_hp_ids
            if term_id in self.hp_index
        }
        (
            query_profiles,
            query_exact,
            query_ancestor,
            query_semantic_exact,
            query_semantic_ancestor,
        ) = self._query(source, query_index)
        query_mask = query_profiles.masks[query_index]
        exact_hits = self.exact_index.search(
            query_exact[query_mask], retrieve_k, nprobe, excluded_indices
        )
        ancestor_hits = self.ancestor_index.search(
            query_ancestor[query_mask], retrieve_k, nprobe, excluded_indices
        )
        raw_hits = self.raw_index.search(
            query_profiles.prototypes[query_index][query_mask],
            retrieve_k,
            nprobe,
            excluded_indices,
        )
        semantic_hits = (
            self.semantic_index.search(
                query_profiles.semantic_vectors[query_index][None, :],
                retrieve_k,
                nprobe,
                excluded_indices,
            )
            if self.semantic_index is not None
            else []
        )
        query_id = query_profiles.ids[query_index]
        alias_hits = sorted(
            {
                hp_position
                for alias in query_profiles.aliases.get(query_id, set())
                for hp_position in self.alias_index.get(alias, set())
                if hp_position not in excluded_indices
            }
        )
        if self.model.fusion_mode == "semantic_only":
            exact_hits = []
            ancestor_hits = []
            raw_hits = []
        elif self.model.fusion_mode == "structure_only":
            semantic_hits = []
            alias_hits = []
        candidate_indices = sorted(
            set(index for index, _score in exact_hits)
            | set(index for index, _score in ancestor_hits)
            | set(index for index, _score in raw_hits)
            | set(index for index, _score in semantic_hits)
            | set(alias_hits)
            | {
                self.hp_index[term_id]
                for term_id in forced_hp_ids
                if term_id in self.hp_index and self.hp_index[term_id] not in excluded_indices
            }
        )
        exact_rank = {index: rank for rank, (index, _score) in enumerate(exact_hits, 1)}
        ancestor_rank = {
            index: rank for rank, (index, _score) in enumerate(ancestor_hits, 1)
        }
        raw_rank = {index: rank for rank, (index, _score) in enumerate(raw_hits, 1)}
        semantic_rank = {
            index: rank for rank, (index, _score) in enumerate(semantic_hits, 1)
        }
        query_mechanisms = query_profiles.mechanism_ids[query_id]
        query_specific = query_profiles.specific_mechanism_ids[query_id]
        query_aliases = query_profiles.aliases.get(query_id, set())
        feature_rows: list[np.ndarray] = []
        rows: list[dict[str, object]] = []

        for hp_position in candidate_indices:
            hp_id = self.hp_profiles.ids[hp_position]
            hp_mechanisms = self.hp_profiles.mechanism_ids[hp_id]
            hp_specific = self.hp_profiles.specific_mechanism_ids[hp_id]
            exact_structure_score = _max_similarity(
                query_exact,
                query_mask,
                self.hp_exact[hp_position],
                self.hp_profiles.masks[hp_position],
            )
            ancestor_structure_score = _max_similarity(
                query_ancestor,
                query_mask,
                self.hp_ancestor[hp_position],
                self.hp_profiles.masks[hp_position],
            )
            exact_semantic_tower_score = float(
                query_semantic_exact @ self.hp_semantic_exact[hp_position]
            )
            ancestor_semantic_tower_score = float(
                query_semantic_ancestor @ self.hp_semantic_ancestor[hp_position]
            )
            poincare_score = poincare_profile_score(
                query_profiles,
                query_index,
                self.hp_profiles,
                hp_position,
                self.model.hierarchy_dimension,
            )
            exact_score = (
                self.exact_fusion_weights["semantic"] * exact_semantic_tower_score
                + self.exact_fusion_weights["structure"] * exact_structure_score
                + self.exact_fusion_weights["poincare"] * poincare_score
            )
            ancestor_score = (
                self.ancestor_fusion_weights["semantic"]
                * ancestor_semantic_tower_score
                + self.ancestor_fusion_weights["structure"]
                * ancestor_structure_score
                + self.ancestor_fusion_weights["poincare"] * poincare_score
            )
            raw_score = _raw_score(
                query_profiles, query_index, self.hp_profiles, hp_position
            )
            semantic_score = (
                float(
                    query_profiles.semantic_vectors[query_index]
                    @ self.hp_profiles.semantic_vectors[hp_position]
                )
                if query_profiles.semantic_vectors.shape[1]
                else 0.0
            )
            hp_aliases = self.hp_profiles.aliases.get(hp_id, set())
            shared_aliases = query_aliases & hp_aliases
            alias_union = query_aliases | hp_aliases
            alias_jaccard = len(shared_aliases) / max(1, len(alias_union))
            shared = query_mechanisms & hp_mechanisms
            shared_specific = query_specific & hp_specific
            union = query_mechanisms | hp_mechanisms
            mechanism_jaccard = len(shared) / max(1, len(union))
            if self.model.fusion_mode == "semantic_only":
                exact_structure_score = 0.0
                ancestor_structure_score = 0.0
                poincare_score = 0.0
                raw_score = 0.0
                mechanism_jaccard = 0.0
            elif self.model.fusion_mode == "structure_only":
                exact_semantic_tower_score = 0.0
                ancestor_semantic_tower_score = 0.0
                semantic_score = 0.0
                shared_aliases = set()
                alias_jaccard = 0.0
            features = mask_pair_features(np.asarray(
                [
                    exact_score,
                    ancestor_score,
                    exact_score - ancestor_score,
                    exact_structure_score,
                    ancestor_structure_score,
                    exact_semantic_tower_score,
                    ancestor_semantic_tower_score,
                    poincare_score,
                    raw_score,
                    float(bool(shared_aliases)),
                    alias_jaccard,
                    math.log1p(len(shared)),
                    math.log1p(len(shared_specific)),
                    mechanism_jaccard,
                    math.log1p(len(query_mechanisms)),
                    math.log1p(len(hp_mechanisms)),
                ],
                dtype=np.float64,
            ), self.model.fusion_mode)
            feature_rows.append(features)
            rows.append(
                {
                    "hp_index": hp_position,
                    "hp_id": hp_id,
                    "hp_name": self.hp_profiles.names[hp_position],
                    "fusion_mode": self.model.fusion_mode,
                    "exact_embedding_score": exact_score,
                    "ancestor_embedding_score": ancestor_score,
                    "exact_structure_score": exact_structure_score,
                    "ancestor_structure_score": ancestor_structure_score,
                    "exact_semantic_tower_score": exact_semantic_tower_score,
                    "ancestor_semantic_tower_score": ancestor_semantic_tower_score,
                    "poincare_score": poincare_score,
                    "raw_graph_score": raw_score,
                    "semantic_score": semantic_score,
                    "shared_alias_count": len(shared_aliases),
                    "shared_aliases": ";".join(sorted(shared_aliases)[:20]),
                    "alias_jaccard": alias_jaccard,
                    "exact_retrieval_rank": exact_rank.get(hp_position, 0),
                    "ancestor_retrieval_rank": ancestor_rank.get(hp_position, 0),
                    "raw_retrieval_rank": raw_rank.get(hp_position, 0),
                    "semantic_retrieval_rank": semantic_rank.get(hp_position, 0),
                    "alias_retrieval": int(hp_position in alias_hits),
                    "shared_mechanism_count": len(shared),
                    "shared_specific_mechanism_count": len(shared_specific),
                    "mechanism_jaccard": mechanism_jaccard,
                    "shared_mechanisms": ";".join(sorted(shared)[:20]),
                }
            )
        if not rows:
            return []
        probabilities = self.relation.predict_proba(np.stack(feature_rows))
        class_index = {label: index for index, label in enumerate(self.relation.classes)}
        for row, probability in zip(rows, probabilities):
            for label in ("exact", "hp_broader", "hp_narrower", "related"):
                row[f"p_{label}"] = (
                    float(probability[class_index[label]]) if label in class_index else 0.0
                )
        return rows


def gate_feature_vector(candidates: Sequence[Mapping[str, object]]) -> np.ndarray:
    if not candidates:
        return np.zeros(10, dtype=np.float64)
    exact_probabilities = sorted(
        (float(row.get("p_exact", 0.0)) for row in candidates), reverse=True
    )
    top_exact = exact_probabilities[0]
    margin = top_exact - (exact_probabilities[1] if len(exact_probabilities) > 1 else 0.0)
    alias_candidates = [
        row for row in candidates if int(row.get("shared_alias_count", 0)) > 0
    ]
    return np.asarray(
        [
            top_exact,
            margin,
            max(float(row.get("exact_embedding_score", -1.0)) for row in candidates),
            max(float(row.get("p_hp_broader", 0.0)) for row in candidates),
            max(float(row.get("ancestor_embedding_score", -1.0)) for row in candidates),
            float(sum(float(row.get("p_exact", 0.0)) >= 0.5 for row in candidates)),
            math.log1p(len(candidates)),
            max(
                (float(row.get("shared_alias_count", 0)) for row in alias_candidates),
                default=0.0,
            ),
            max(
                (float(row.get("alias_jaccard", 0.0)) for row in alias_candidates),
                default=0.0,
            ),
            max(
                (float(row.get("semantic_score", -1.0)) for row in alias_candidates),
                default=-1.0,
            ),
        ],
        dtype=np.float64,
    )


def fit_gate_calibrator(
    retriever: DualHeadRetriever,
    expert_examples: Sequence[MappingExample],
    parent_examples: Sequence[MappingExample],
    retrieve_k: int = 80,
    nprobe: int = 8,
    seed: int = 20260726,
) -> GateCalibrator:
    features: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    seen: set[tuple[str, str]] = set()

    for row in expert_examples:
        if row.relation != "exact" or row.mp_id not in retriever.mp_index:
            continue
        key = ("positive", row.mp_id)
        if key in seen:
            continue
        seen.add(key)
        candidates = retriever.retrieve(
            "mp", retriever.mp_index[row.mp_id], retrieve_k, nprobe
        )
        features.append(gate_feature_vector(candidates))
        labels.append(1)
        splits.append(row.split)

    for row in parent_examples:
        if row.source not in {"counterfactual_exact", "synthetic_hpo_cutout"}:
            continue
        if row.query_profile_source == "hp":
            if row.exact_hp_id not in retriever.hp_index:
                continue
            source = "hp"
            query_index = retriever.hp_index[row.exact_hp_id]
        else:
            if row.mp_id not in retriever.mp_index:
                continue
            source = "mp"
            query_index = retriever.mp_index[row.mp_id]
        key = ("negative", row.mp_id)
        if key in seen:
            continue
        seen.add(key)
        excluded = set(filter(None, row.masked_hp_ids.split(";")))
        candidates = retriever.retrieve(
            source, query_index, retrieve_k, nprobe, excluded
        )
        features.append(gate_feature_vector(candidates))
        labels.append(0)
        splits.append(row.split)

    matrix = np.stack(features)
    labels_array = np.asarray(labels, dtype=np.int32)
    train = np.asarray([split == "train" for split in splits])
    validation = np.asarray([split == "validation" for split in splits])
    if train.sum() < 4 or len(set(labels_array[train])) < 2:
        raise ValueError("Gate calibration requires positive and negative training episodes")
    scaler = StandardScaler().fit(matrix[train])
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    ).fit(scaler.transform(matrix[train]), labels_array[train])
    threshold = 0.5
    if validation.sum() >= 4 and len(set(labels_array[validation])) == 2:
        probabilities = classifier.predict_proba(scaler.transform(matrix[validation]))[:, 1]
        candidates = np.linspace(0.10, 0.90, 81)
        threshold = float(
            max(
                candidates,
                key=lambda value: balanced_accuracy_score(
                    labels_array[validation], probabilities >= value
                ),
            )
        )
    return GateCalibrator(scaler, classifier, threshold)


def fit_ancestor_rank_weights(
    retriever: DualHeadRetriever,
    parent_examples: Sequence[MappingExample],
    exact_mp_ids: set[str],
    retrieve_k: int = 80,
    nprobe: int = 8,
) -> AncestorRankWeights:
    grouped: dict[tuple[str, str, str], list[MappingExample]] = {}
    for row in parent_examples:
        if row.split != "validation":
            continue
        if row.source == "expert_broad" and row.mp_id in exact_mp_ids:
            continue
        key = (row.source, row.mp_id, row.masked_hp_ids)
        grouped.setdefault(key, []).append(row)

    episodes: list[tuple[str, list[dict[str, object]], set[str]]] = []
    for (source_name, _query_id, masked), rows in sorted(grouped.items()):
        first = rows[0]
        if first.query_profile_source == "hp":
            source = "hp"
            query_index = retriever.hp_index.get(first.exact_hp_id)
        else:
            source = "mp"
            query_index = retriever.mp_index.get(first.mp_id)
        if query_index is None:
            continue
        candidates = retriever.retrieve(
            source,
            query_index,
            retrieve_k,
            nprobe,
            set(filter(None, masked.split(";"))),
        )
        episodes.append((source_name, candidates, {row.hp_id for row in rows}))
    if not episodes:
        if retriever.model.fusion_mode == "semantic_only":
            return AncestorRankWeights(1.0, 0.0, 0.0, 0.0, 0.0, {})
        if retriever.model.fusion_mode == "structure_only":
            return AncestorRankWeights(0.0, 0.70, 0.30, 0.0, 0.0, {})
        return AncestorRankWeights(0.50, 0.30, 0.20, 0.0, 0.0, {})

    grid: list[tuple[float, float, float, float]] = []
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    for semantic in values:
        for structure in values:
            for mechanism in values:
                for relation in values:
                    if retriever.model.fusion_mode == "semantic_only" and (
                        structure or mechanism
                    ):
                        continue
                    if retriever.model.fusion_mode == "structure_only" and semantic:
                        continue
                    total = semantic + structure + mechanism + relation
                    if total:
                        grid.append(
                            (
                                semantic / total,
                                structure / total,
                                mechanism / total,
                                relation / total,
                            )
                        )

    def evaluate(
        weights: tuple[float, float, float, float]
    ) -> tuple[float, dict[str, float]]:
        source_rr: dict[str, list[float]] = {}
        provisional = AncestorRankWeights(*weights, 0.0, {})
        for source_name, candidates, targets in episodes:
            ranked = sorted(candidates, key=lambda row: -provisional.score(row))
            rank = next(
                (position for position, row in enumerate(ranked, 1) if row["hp_id"] in targets),
                0,
            )
            source_rr.setdefault(source_name, []).append(1.0 / rank if rank else 0.0)
        means = {source: float(np.mean(values)) for source, values in source_rr.items()}
        weighted = []
        for source, mean in means.items():
            repeat = 2 if source == "counterfactual_exact" else 1
            weighted.extend([mean] * repeat)
        return float(np.mean(weighted)), means

    best_weights = grid[0]
    best_score = -1.0
    best_sources: dict[str, float] = {}
    for weights in grid:
        score, sources = evaluate(weights)
        if (score, weights[1], weights[2], weights[0]) > (
            best_score,
            best_weights[1],
            best_weights[2],
            best_weights[0],
        ):
            best_weights = weights
            best_score = score
            best_sources = sources
    return AncestorRankWeights(*best_weights, best_score, best_sources)


def rank_candidates(
    candidates: list[dict[str, object]],
    gate_probability: float,
    gate_threshold: float = 0.5,
    ancestor_weights: AncestorRankWeights | None = None,
    exact_available: bool | None = None,
) -> list[dict[str, object]]:
    ancestor_weights = ancestor_weights or AncestorRankWeights(
        0.50, 0.30, 0.20, 0.0, 0.0, {}
    )
    exact_available = (
        gate_probability >= gate_threshold
        if exact_available is None
        else exact_available
    )
    has_evidence_override = any(
        bool(row.get("exact_evidence_override", 0)) for row in candidates
    )
    for row in candidates:
        p_exact = float(row.get("p_exact", 0.0))
        p_ancestor = float(row.get("p_hp_broader", 0.0))
        exact_candidate = p_exact >= 0.5 or bool(row.get("exact_evidence_override", 0))
        ancestor_score = ancestor_weights.score(row)
        row["ancestor_rank_score"] = ancestor_score
        row["final_score"] = (
            2.0 * p_exact
            + 0.5 * ((float(row.get("exact_embedding_score", -1.0)) + 1.0) / 2.0)
            + ancestor_score
            if exact_available
            else ancestor_score
        )
        row["predicted_relation"] = (
            "exact"
            if exact_available and exact_candidate
            else "hp_broader"
            if int(row.get("derived_ancestor_distance", 0)) > 0
            or p_ancestor >= p_exact
            else "uncertain"
        )
    candidates.sort(
        key=lambda row: (
            0
            if exact_available and (
                bool(row.get("exact_evidence_override", 0))
                or (
                    not has_evidence_override
                    and float(row.get("p_exact", 0.0)) >= 0.5
                )
            )
            else 1
            if exact_available
            and int(row.get("derived_ancestor_distance", 0)) > 0
            else 2
            if exact_available
            and float(row.get("p_exact", 0.0)) >= 0.5
            else 3,
            int(row.get("derived_ancestor_distance", 0))
            if int(row.get("derived_ancestor_distance", 0)) > 0
            else 0,
            -float(row["final_score"]),
            -float(row.get("p_exact", 0.0)),
            str(row["hp_id"]),
        )
    )
    for rank, row in enumerate(candidates, 1):
        row["final_rank"] = rank
        row["gate_probability"] = gate_probability
        row["exact_available_prediction"] = int(exact_available)
    return candidates
