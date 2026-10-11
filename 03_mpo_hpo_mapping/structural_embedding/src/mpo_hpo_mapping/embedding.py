from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .data import MechanismEdge, Ontology


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def poincare_distance(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    left_norm = torch.sum(left * left, dim=-1).clamp_max(1.0 - eps)
    right_norm = torch.sum(right * right, dim=-1).clamp_max(1.0 - eps)
    squared_distance = torch.sum((left - right) ** 2, dim=-1)
    argument = 1.0 + 2.0 * squared_distance / ((1.0 - left_norm) * (1.0 - right_norm)).clamp_min(eps)
    return torch.acosh(argument.clamp_min(1.0 + eps))


def project_ball_(weights: torch.Tensor, max_norm: float = 1.0 - 1e-4) -> None:
    with torch.no_grad():
        norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True).clamp_min(1e-12)
        weights.mul_(torch.where(norms > max_norm, max_norm / norms, torch.ones_like(norms)))


def logmap_zero(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.vector_norm(points, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.atanh(norms.clamp_max(1.0 - 1e-5)) / norms
    return points * scale


@dataclass
class HierarchyEmbedding:
    ids: list[str]
    vectors: np.ndarray
    loss_history: list[float]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {term_id: self.vectors[index] for index, term_id in enumerate(self.ids)}


def train_poincare_hierarchy(
    ontology: Ontology,
    dimensions: int = 16,
    epochs: int = 30,
    batch_size: int = 2048,
    negatives: int = 5,
    learning_rate: float = 0.02,
    radial_margin: float = 0.01,
    seed: int = 20260726,
) -> HierarchyEmbedding:
    seed_everything(seed)
    ids = sorted(ontology.nodes)
    index = {term_id: position for position, term_id in enumerate(ids)}
    edges = sorted(
        (index[child], index[parent])
        for child, parents in ontology.parents.items()
        for parent in parents
        if child in index and parent in index
    )
    if not ids:
        return HierarchyEmbedding([], np.empty((0, dimensions), dtype=np.float32), [])

    embedding = torch.nn.Embedding(len(ids), dimensions)
    torch.nn.init.uniform_(embedding.weight, -1e-3, 1e-3)
    optimizer = torch.optim.Adam(embedding.parameters(), lr=learning_rate)
    edge_tensor = torch.tensor(edges, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []

    if not edges or epochs <= 0:
        project_ball_(embedding.weight)
        vectors = embedding.weight.detach().cpu().numpy().astype(np.float32)
        return HierarchyEmbedding(ids, vectors, history)

    for _epoch in range(epochs):
        permutation = torch.randperm(len(edges), generator=generator)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(edges), batch_size):
            batch = edge_tensor[permutation[start : start + batch_size]]
            child_index = batch[:, 0]
            parent_index = batch[:, 1]
            child = embedding(child_index)
            parent = embedding(parent_index)
            negative_index = torch.randint(
                0, len(ids), (len(batch), negatives), generator=generator
            )
            negative = embedding(negative_index)
            positive_distance = poincare_distance(child, parent)
            negative_distance = poincare_distance(child[:, None, :], negative)
            link_loss = torch.nn.functional.softplus(positive_distance).mean()
            link_loss += torch.nn.functional.softplus(1.0 - negative_distance).mean()
            child_radius = torch.linalg.vector_norm(child, dim=1)
            parent_radius = torch.linalg.vector_norm(parent, dim=1)
            hierarchy_loss = torch.relu(parent_radius - child_radius + radial_margin).mean()
            loss = link_loss + 0.25 * hierarchy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            project_ball_(embedding.weight)
            epoch_loss += float(loss.detach())
            batch_count += 1
        history.append(epoch_loss / max(1, batch_count))

    # Keep coordinates in the Poincare ball. The v2 model consumes hyperbolic
    # distances directly instead of flattening the hierarchy into cosine space.
    vectors = embedding.weight.detach().cpu().numpy().astype(np.float32)
    return HierarchyEmbedding(ids, vectors, history)


def constant_hierarchy_embedding(
    ontology: Ontology,
    dimensions: int = 16,
    value: float = 0.0,
) -> HierarchyEmbedding:
    """Return equal vectors for every node as a no-structure control."""

    ids = sorted(ontology.nodes)
    vectors = np.full((len(ids), dimensions), value, dtype=np.float32)
    return HierarchyEmbedding(ids, vectors, [])


def permute_hierarchy_embedding(
    embedding: HierarchyEmbedding,
    seed: int = 20260726,
) -> HierarchyEmbedding:
    """Preserve the vector distribution while breaking node-structure alignment."""

    randomizer = np.random.default_rng(seed)
    permutation = randomizer.permutation(len(embedding.ids))
    vectors = embedding.vectors[permutation].copy()
    return HierarchyEmbedding(list(embedding.ids), vectors, list(embedding.loss_history))


@dataclass
class ProfileTensor:
    ids: list[str]
    names: list[str]
    prototypes: np.ndarray
    masks: np.ndarray
    edge_counts: np.ndarray
    specific_counts: np.ndarray
    semantic_vectors: np.ndarray
    aliases: dict[str, set[str]]
    mechanism_ids: dict[str, set[str]]
    specific_mechanism_ids: dict[str, set[str]]

    @property
    def index(self) -> dict[str, int]:
        return {term_id: position for position, term_id in enumerate(self.ids)}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _weighted_prototypes(
    vectors: np.ndarray,
    weights: np.ndarray,
    max_prototypes: int,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vectors) == 0:
        return np.zeros((max_prototypes, vectors.shape[1]), dtype=np.float32), np.zeros(max_prototypes, dtype=bool)
    vectors = np.stack([_normalize(vector) for vector in vectors]).astype(np.float32)
    count = min(max_prototypes, len(vectors))
    if len(vectors) <= count:
        output = np.zeros((max_prototypes, vectors.shape[1]), dtype=np.float32)
        output[: len(vectors)] = vectors
        mask = np.zeros(max_prototypes, dtype=bool)
        mask[: len(vectors)] = True
        return output, mask

    centers = [int(np.argmax(weights))]
    while len(centers) < count:
        similarities = vectors @ vectors[centers].T
        distances = 1.0 - similarities.max(axis=1)
        distances[centers] = -1.0
        centers.append(int(np.argmax(distances * weights)))
    centroids = vectors[centers].copy()
    for _ in range(iterations):
        assignment = np.argmax(vectors @ centroids.T, axis=1)
        for cluster in range(count):
            selected = assignment == cluster
            if not np.any(selected):
                continue
            centroid = np.average(vectors[selected], axis=0, weights=weights[selected])
            centroids[cluster] = _normalize(centroid)
    order = sorted(
        range(count),
        key=lambda cluster: (-float(weights[assignment == cluster].sum()), cluster),
    )
    output = np.zeros((max_prototypes, vectors.shape[1]), dtype=np.float32)
    output[:count] = centroids[order]
    mask = np.zeros(max_prototypes, dtype=bool)
    mask[:count] = True
    return output, mask


def _select_structure_prototypes(
    vectors: np.ndarray,
    weights: np.ndarray,
    max_prototypes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep high-confidence mechanism pairs without changing ball coordinates."""

    output = np.zeros((max_prototypes, vectors.shape[1]), dtype=np.float32)
    mask = np.zeros(max_prototypes, dtype=bool)
    if not len(vectors):
        return output, mask
    order = sorted(range(len(vectors)), key=lambda index: (-float(weights[index]), index))
    selected = order[:max_prototypes]
    output[: len(selected)] = vectors[selected]
    mask[: len(selected)] = True
    return output, mask


def build_profile_tensor(
    profiles: Mapping[str, Sequence[MechanismEdge]],
    cl_vectors: Mapping[str, np.ndarray],
    go_vectors: Mapping[str, np.ndarray],
    max_prototypes: int = 4,
    semantic_vectors: Mapping[str, np.ndarray] | None = None,
    semantic_weight: float = 1.0,
    aliases: Mapping[str, set[str]] | None = None,
) -> ProfileTensor:
    ids = sorted(profiles)
    cl_dimension = len(next(iter(cl_vectors.values()))) if cl_vectors else 0
    go_dimension = len(next(iter(go_vectors.values()))) if go_vectors else 0
    semantic_vectors = semantic_vectors or {}
    aliases = aliases or {}
    semantic_dimension = len(next(iter(semantic_vectors.values()))) if semantic_vectors else 0
    # Structure and semantics remain separate in v2. Each structure prototype
    # is an interpretable (CL, GO, channel) mechanism tuple.
    vector_dimension = cl_dimension + go_dimension + 2
    all_prototypes = np.zeros((len(ids), max_prototypes, vector_dimension), dtype=np.float32)
    all_masks = np.zeros((len(ids), max_prototypes), dtype=bool)
    all_semantic = np.zeros((len(ids), semantic_dimension), dtype=np.float32)
    names: list[str] = []
    edge_counts = np.zeros(len(ids), dtype=np.int32)
    specific_counts = np.zeros(len(ids), dtype=np.int32)
    mechanism_ids: dict[str, set[str]] = {}
    specific_ids: dict[str, set[str]] = {}

    for term_index, term_id in enumerate(ids):
        edges = profiles[term_id]
        semantic = semantic_vectors.get(term_id)
        if semantic is not None:
            all_semantic[term_index] = _normalize(semantic)
        names.append(edges[0].term_name if edges else "")
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        all_ids: set[str] = set()
        term_specific_ids: set[str] = set()
        for edge in edges:
            cl = cl_vectors.get(edge.cl_id)
            go = go_vectors.get(edge.go_id)
            if cl is None or go is None:
                continue
            channel = np.array(
                [1.0 if edge.channel == "specific" else 0.0, 1.0 if edge.channel == "general" else 0.0],
                dtype=np.float32,
            )
            vectors.append(np.concatenate((cl, go, channel)).astype(np.float32))
            weights.append(edge.weight)
            all_ids.add(edge.mechanism_id)
            if edge.channel == "specific":
                term_specific_ids.add(edge.mechanism_id)
        mechanism_ids[term_id] = all_ids
        specific_ids[term_id] = term_specific_ids
        edge_counts[term_index] = len(vectors)
        specific_counts[term_index] = len(term_specific_ids)
        if vectors:
            prototypes, mask = _select_structure_prototypes(
                np.asarray(vectors), np.asarray(weights, dtype=np.float32), max_prototypes
            )
            all_prototypes[term_index] = prototypes
            all_masks[term_index] = mask

    keep = np.flatnonzero(all_masks.any(axis=1))
    kept_ids = [ids[index] for index in keep]
    return ProfileTensor(
        ids=kept_ids,
        names=[names[index] for index in keep],
        prototypes=all_prototypes[keep],
        masks=all_masks[keep],
        edge_counts=edge_counts[keep],
        specific_counts=specific_counts[keep],
        semantic_vectors=all_semantic[keep],
        aliases={term_id: set(aliases.get(term_id, set())) for term_id in kept_ids},
        mechanism_ids={term_id: mechanism_ids[term_id] for term_id in kept_ids},
        specific_mechanism_ids={term_id: specific_ids[term_id] for term_id in kept_ids},
    )


class IVFPrototypeIndex:
    """A small dependency-free IVF index over term prototype vectors."""

    def __init__(
        self,
        vectors: np.ndarray,
        term_indices: np.ndarray,
        clusters: int = 96,
        seed: int = 20260726,
    ) -> None:
        if len(vectors) == 0:
            raise ValueError("Cannot build an IVF index without vectors")
        self.vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        self.term_indices = term_indices.astype(np.int32)
        cluster_count = max(1, min(clusters, len(vectors)))
        if cluster_count == 1:
            self.centers = np.mean(self.vectors, axis=0, keepdims=True)
            self.assignments = np.zeros(len(vectors), dtype=np.int32)
        else:
            model = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=seed,
                batch_size=min(4096, max(256, len(vectors))),
                n_init=3,
                max_iter=100,
            )
            self.assignments = model.fit_predict(self.vectors).astype(np.int32)
            self.centers = model.cluster_centers_.astype(np.float32)
        self.centers /= np.clip(np.linalg.norm(self.centers, axis=1, keepdims=True), 1e-12, None)
        self.buckets = {
            cluster: np.flatnonzero(self.assignments == cluster)
            for cluster in range(len(self.centers))
        }

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        nprobe: int = 6,
        excluded_terms: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        excluded_terms = excluded_terms or set()
        query_vectors = query_vectors / np.clip(
            np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None
        )
        candidates: set[int] = set()
        probes = min(max(1, nprobe), len(self.centers))
        for query in query_vectors:
            center_order = np.argsort(-(self.centers @ query))[:probes]
            for cluster in center_order:
                candidates.update(map(int, self.buckets[int(cluster)]))
        if not candidates:
            return []
        prototype_indices = np.fromiter(candidates, dtype=np.int32)
        prototype_vectors = self.vectors[prototype_indices]
        similarities = query_vectors @ prototype_vectors.T
        best = similarities.max(axis=0)
        by_term: dict[int, float] = {}
        for local_index, prototype_index in enumerate(prototype_indices):
            term_index = int(self.term_indices[prototype_index])
            if term_index in excluded_terms:
                continue
            score = float(best[local_index])
            if score > by_term.get(term_index, -math.inf):
                by_term[term_index] = score
        return sorted(by_term.items(), key=lambda item: (-item[1], item[0]))[:top_k]
