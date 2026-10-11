from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import MappingExample
from .embedding import ProfileTensor, seed_everything


RELATION_CLASSES = ["exact", "hp_broader", "hp_narrower", "related"]
PAIR_FEATURE_NAMES = [
    "fused_exact",
    "fused_ancestor",
    "fused_gap",
    "structure_exact",
    "structure_ancestor",
    "semantic_exact",
    "semantic_ancestor",
    "poincare_score",
    "raw_structure_score",
    "alias_hit",
    "alias_jaccard",
    "shared_mechanism_log",
    "shared_specific_log",
    "mechanism_jaccard",
    "query_mechanism_log",
    "hp_mechanism_log",
]


def mask_pair_features(values: np.ndarray, fusion_mode: str) -> np.ndarray:
    """Remove every downstream shortcut from single-modality controls."""

    output = values.copy()
    if fusion_mode == "semantic_only":
        # Structure tower, Poincare/raw structure, and mechanism-link features.
        output[[3, 4, 7, 8, 11, 12, 13, 14, 15]] = 0.0
    elif fusion_mode == "structure_only":
        # Semantic tower and text-alias features.
        output[[5, 6, 9, 10]] = 0.0
    return output


def _poincare_distance_tensor(
    left: torch.Tensor, right: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    left_norm = torch.sum(left * left, dim=-1).clamp(max=1.0 - eps)
    right_norm = torch.sum(right * right, dim=-1).clamp(max=1.0 - eps)
    squared = torch.sum((left - right) ** 2, dim=-1)
    denominator = ((1.0 - left_norm) * (1.0 - right_norm)).clamp_min(eps)
    return torch.acosh((1.0 + 2.0 * squared / denominator).clamp_min(1.0 + eps))


def _poincare_distance_numpy(
    left: np.ndarray, right: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    left_norm = np.minimum(np.sum(left * left, axis=-1), 1.0 - eps)
    right_norm = np.minimum(np.sum(right * right, axis=-1), 1.0 - eps)
    squared = np.sum((left - right) ** 2, axis=-1)
    denominator = np.maximum((1.0 - left_norm) * (1.0 - right_norm), eps)
    return np.arccosh(np.maximum(1.0 + 2.0 * squared / denominator, 1.0 + eps))


class DualHeadEncoder(torch.nn.Module):
    """Late-fusion semantic/structure encoder with explicit hyperbolic scoring."""

    def __init__(
        self,
        structure_dimension: int,
        semantic_dimension: int,
        hidden_dimension: int = 64,
        output_dimension: int = 32,
        hierarchy_dimension: int = 16,
        dropout: float = 0.10,
        fusion_mode: str = "late_fusion",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"late_fusion", "semantic_only", "structure_only"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.structure_dimension = structure_dimension
        self.semantic_dimension = semantic_dimension
        self.semantic_input_dimension = max(1, semantic_dimension)
        self.hierarchy_dimension = hierarchy_dimension
        self.fusion_mode = fusion_mode

        self.structure_inputs = torch.nn.ModuleDict(
            {
                side: torch.nn.Linear(structure_dimension, hidden_dimension)
                for side in ("hp", "mp")
            }
        )
        self.semantic_inputs = torch.nn.ModuleDict(
            {
                side: torch.nn.Linear(self.semantic_input_dimension, hidden_dimension)
                for side in ("hp", "mp")
            }
        )
        self.structure_shared = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.GELU(),
        )
        self.semantic_shared = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.GELU(),
        )
        self.structure_projections = torch.nn.ModuleDict(
            {
                f"{head}_{side}": torch.nn.Linear(
                    hidden_dimension, output_dimension, bias=False
                )
                for head in ("exact", "ancestor")
                for side in ("hp", "mp")
            }
        )
        self.semantic_projections = torch.nn.ModuleDict(
            {
                f"{head}_{side}": torch.nn.Linear(
                    hidden_dimension, output_dimension, bias=False
                )
                for head in ("exact", "ancestor")
                for side in ("hp", "mp")
            }
        )
        # Softmax makes every modality contribution non-negative and sums to one.
        self.fusion_logits = torch.nn.ParameterDict(
            {
                head: torch.nn.Parameter(torch.zeros(3))
                for head in ("exact", "ancestor")
            }
        )
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(0.08)))

    def fusion_weights(self, head: str) -> torch.Tensor:
        if self.fusion_mode == "semantic_only":
            return torch.tensor(
                [1.0, 0.0, 0.0], device=self.fusion_logits[head].device
            )
        if self.fusion_mode == "structure_only":
            structure = torch.softmax(self.fusion_logits[head][1:], dim=0)
            return torch.cat(
                (torch.zeros(1, device=structure.device), structure), dim=0
            )
        return torch.softmax(self.fusion_logits[head], dim=0)

    def fusion_weight_dict(self, head: str) -> dict[str, float]:
        values = self.fusion_weights(head).detach().cpu().tolist()
        return dict(zip(("semantic", "structure", "poincare"), values))

    def encode_structure(
        self,
        prototypes: torch.Tensor,
        masks: torch.Tensor,
        side: str,
        head: str,
    ) -> torch.Tensor:
        base = self.structure_inputs[side](prototypes)
        encoded = self.structure_projections[f"{head}_{side}"](
            self.structure_shared(base)
        )
        encoded = torch.nn.functional.normalize(encoded, dim=-1)
        return encoded * masks.unsqueeze(-1)

    def _semantic_input(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.shape[-1]:
            return semantic
        return torch.zeros(
            (*semantic.shape[:-1], 1), dtype=semantic.dtype, device=semantic.device
        )

    def encode_semantic(
        self, semantic: torch.Tensor, side: str, head: str
    ) -> torch.Tensor:
        base = self.semantic_inputs[side](self._semantic_input(semantic))
        encoded = self.semantic_projections[f"{head}_{side}"](
            self.semantic_shared(base)
        )
        return torch.nn.functional.normalize(encoded, dim=-1)

    def encode(
        self,
        prototypes: torch.Tensor,
        masks: torch.Tensor,
        side: str,
        head: str,
    ) -> torch.Tensor:
        """Compatibility alias used by structure prototype indexes."""

        return self.encode_structure(prototypes, masks, side, head)

    def _poincare_score_matrix(
        self,
        query: torch.Tensor,
        query_masks: torch.Tensor,
        target: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        dimension = self.hierarchy_dimension
        query_expanded = query[:, None, :, None, :]
        target_expanded = target[None, :, None, :, :]
        cl_distance = _poincare_distance_tensor(
            query_expanded[..., :dimension], target_expanded[..., :dimension]
        )
        go_distance = _poincare_distance_tensor(
            query_expanded[..., dimension : 2 * dimension],
            target_expanded[..., dimension : 2 * dimension],
        )
        channel_score = torch.sum(
            query_expanded[..., 2 * dimension :] * target_expanded[..., 2 * dimension :],
            dim=-1,
        )
        score = (
            0.4 * (2.0 * torch.exp(-cl_distance) - 1.0)
            + 0.4 * (2.0 * torch.exp(-go_distance) - 1.0)
            + 0.2 * channel_score
        )
        valid = query_masks[:, None, :, None] & target_masks[None, :, None, :]
        return score.masked_fill(~valid, -1e4).amax(dim=(-1, -2))

    def score_components(
        self,
        query_prototypes: torch.Tensor,
        query_masks: torch.Tensor,
        query_semantic: torch.Tensor,
        target_prototypes: torch.Tensor,
        target_masks: torch.Tensor,
        target_semantic: torch.Tensor,
        head: str,
        semantic_dropout_probability: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        query_structure = self.encode_structure(
            query_prototypes, query_masks, "mp", head
        )
        target_structure = self.encode_structure(
            target_prototypes, target_masks, "hp", head
        )
        pair_scores = torch.einsum(
            "bkd,cld->bckl", query_structure, target_structure
        )
        valid = query_masks[:, None, :, None] & target_masks[None, :, None, :]
        structure = pair_scores.masked_fill(~valid, -1e4).amax(dim=(-1, -2))

        query_semantic_encoded = self.encode_semantic(query_semantic, "mp", head)
        target_semantic_encoded = self.encode_semantic(target_semantic, "hp", head)
        semantic = query_semantic_encoded @ target_semantic_encoded.T
        if self.training and semantic_dropout_probability > 0:
            keep = (
                torch.rand((semantic.shape[0], 1), device=semantic.device)
                >= semantic_dropout_probability
            )
            semantic = torch.where(keep, semantic, torch.zeros_like(semantic))

        poincare = self._poincare_score_matrix(
            query_prototypes, query_masks, target_prototypes, target_masks
        )
        weights = self.fusion_weights(head)
        fused = weights[0] * semantic + weights[1] * structure + weights[2] * poincare
        return {
            "fused": fused,
            "semantic": semantic,
            "structure": structure,
            "poincare": poincare,
        }

    def paired_components(
        self,
        query_prototypes: torch.Tensor,
        query_masks: torch.Tensor,
        query_semantic: torch.Tensor,
        target_prototypes: torch.Tensor,
        target_masks: torch.Tensor,
        target_semantic: torch.Tensor,
        head: str,
    ) -> dict[str, torch.Tensor]:
        components = self.score_components(
            query_prototypes,
            query_masks,
            query_semantic,
            target_prototypes,
            target_masks,
            target_semantic,
            head,
        )
        return {name: torch.diagonal(value) for name, value in components.items()}

    def score_matrix(
        self,
        query_prototypes: torch.Tensor,
        query_masks: torch.Tensor,
        query_semantic: torch.Tensor,
        target_prototypes: torch.Tensor,
        target_masks: torch.Tensor,
        target_semantic: torch.Tensor,
        head: str,
        semantic_dropout_probability: float = 0.0,
    ) -> torch.Tensor:
        return self.score_components(
            query_prototypes,
            query_masks,
            query_semantic,
            target_prototypes,
            target_masks,
            target_semantic,
            head,
            semantic_dropout_probability,
        )["fused"]

    def paired_scores(
        self,
        query_prototypes: torch.Tensor,
        query_masks: torch.Tensor,
        query_semantic: torch.Tensor,
        target_prototypes: torch.Tensor,
        target_masks: torch.Tensor,
        target_semantic: torch.Tensor,
        head: str,
    ) -> torch.Tensor:
        return self.paired_components(
            query_prototypes,
            query_masks,
            query_semantic,
            target_prototypes,
            target_masks,
            target_semantic,
            head,
        )["fused"]


@dataclass
class TrainedModel:
    model: DualHeadEncoder
    loss_history: list[dict[str, float]]


def _rows(
    profile: ProfileTensor, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = np.asarray(indices, dtype=np.int64)
    return (
        torch.from_numpy(profile.prototypes[selected]).float(),
        torch.from_numpy(profile.masks[selected]).bool(),
        torch.from_numpy(profile.semantic_vectors[selected]).float(),
    )


def _query_index(
    example: MappingExample,
    hp_index: Mapping[str, int],
    mp_index: Mapping[str, int],
) -> tuple[str, int] | None:
    if example.query_profile_source == "hp":
        return (
            ("hp", hp_index[example.exact_hp_id])
            if example.exact_hp_id in hp_index
            else None
        )
    return ("mp", mp_index[example.mp_id]) if example.mp_id in mp_index else None


def _query_batch(
    examples: Sequence[MappingExample],
    hp_profiles: ProfileTensor,
    mp_profiles: ProfileTensor,
    hp_index: Mapping[str, int],
    mp_index: Mapping[str, int],
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    prototypes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    semantics: list[np.ndarray] = []
    query_ids: list[str] = []
    for example in examples:
        location = _query_index(example, hp_index, mp_index)
        if location is None:
            raise KeyError(f"Missing query profile for {example.mp_id}")
        source, index = location
        profile = hp_profiles if source == "hp" else mp_profiles
        vector = profile.prototypes[index].copy()
        mask = profile.masks[index].copy()
        if augment:
            valid = np.flatnonzero(mask)
            if len(valid) > 1:
                dropped = valid[np.random.random(len(valid)) < 0.15]
                mask[dropped] = False
                if not mask.any():
                    mask[valid[0]] = True
            # Preserve ball coordinates; noise is applied only to channel flags.
            vector[:, -2:] += (
                np.random.normal(0.0, 0.01, vector[:, -2:].shape).astype(np.float32)
                * mask[:, None]
            )
        prototypes.append(vector)
        masks.append(mask)
        semantics.append(profile.semantic_vectors[index])
        query_ids.append(example.mp_id)
    return (
        torch.from_numpy(np.stack(prototypes)).float(),
        torch.from_numpy(np.stack(masks)).bool(),
        torch.from_numpy(np.stack(semantics)).float(),
        query_ids,
    )


def _multi_positive_contrastive_loss(
    scores: torch.Tensor,
    query_ids: Sequence[str],
    target_ids: Sequence[str],
    temperature: torch.Tensor,
) -> torch.Tensor:
    logits = scores / temperature.clamp(0.03, 0.5)
    device = logits.device
    row_positive = torch.tensor(
        [[left == right for right in target_ids] for left in target_ids],
        dtype=torch.bool,
        device=device,
    )
    column_positive = torch.tensor(
        [[left == right for right in query_ids] for left in query_ids],
        dtype=torch.bool,
        device=device,
    )
    row_log_prob = torch.log_softmax(logits, dim=1)
    row_loss = -torch.logsumexp(
        row_log_prob.masked_fill(~row_positive, -1e4), dim=1
    ).mean()
    column_log_prob = torch.log_softmax(logits, dim=0)
    column_loss = -torch.logsumexp(
        column_log_prob.masked_fill(~column_positive, -1e4), dim=0
    ).mean()
    return 0.5 * (row_loss + column_loss)


def _batches(
    examples: Sequence[MappingExample], batch_size: int, randomizer: random.Random
) -> Iterable[list[MappingExample]]:
    order = list(examples)
    randomizer.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        if len(batch) > 1:
            yield batch


def balanced_parent_sample(
    examples: Sequence[MappingExample],
    total_budget: int,
    randomizer: random.Random,
) -> list[MappingExample]:
    """Sample each parent source equally and cap its dominance per epoch."""

    grouped: dict[str, list[MappingExample]] = defaultdict(list)
    for row in examples:
        grouped[row.source].append(row)
    if not grouped:
        return []
    quota = max(2, math.ceil(total_budget / len(grouped)))
    selected: list[MappingExample] = []
    for source in sorted(grouped):
        rows = list(grouped[source])
        randomizer.shuffle(rows)
        selected.extend(rows[:quota])
    randomizer.shuffle(selected)
    return selected[: max(2, total_budget)]


def train_dual_head(
    hp_profiles: ProfileTensor,
    mp_profiles: ProfileTensor,
    expert_examples: Sequence[MappingExample],
    parent_examples: Sequence[MappingExample],
    epochs: int = 80,
    batch_size: int = 64,
    hidden_dimension: int = 64,
    output_dimension: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 20260726,
    hierarchy_dimension: int = 16,
    structure_aux_weight: float = 0.35,
    semantic_dropout_probability: float = 0.25,
    balance_parent_sources: bool = True,
    fusion_mode: str = "late_fusion",
) -> TrainedModel:
    seed_everything(seed)
    hp_index = hp_profiles.index
    mp_index = mp_profiles.index
    exact = [
        row
        for row in expert_examples
        if row.split == "train"
        and row.relation == "exact"
        and row.mp_id in mp_index
        and row.hp_id in hp_index
    ]
    parents = [
        row
        for row in parent_examples
        if row.split == "train"
        and row.hp_id in hp_index
        and _query_index(row, hp_index, mp_index) is not None
    ]
    if len(exact) < 2 or len(parents) < 2:
        raise ValueError("Exact and ancestor training both require at least two examples")

    model = DualHeadEncoder(
        hp_profiles.prototypes.shape[-1],
        hp_profiles.semantic_vectors.shape[-1],
        hidden_dimension,
        output_dimension,
        hierarchy_dimension,
        fusion_mode=fusion_mode,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    randomizer = random.Random(seed)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        exact_losses: list[float] = []
        exact_structure_losses: list[float] = []
        ancestor_losses: list[float] = []
        ancestor_structure_losses: list[float] = []
        ordering_losses: list[float] = []

        for batch in _batches(exact, batch_size, randomizer):
            query, query_mask, query_semantic, query_ids = _query_batch(
                batch, hp_profiles, mp_profiles, hp_index, mp_index, augment=False
            )
            target, target_mask, target_semantic = _rows(
                hp_profiles, [hp_index[row.hp_id] for row in batch]
            )
            components = model.score_components(
                query,
                query_mask,
                query_semantic,
                target,
                target_mask,
                target_semantic,
                "exact",
                semantic_dropout_probability,
            )
            target_ids = [row.hp_id for row in batch]
            fused_loss = _multi_positive_contrastive_loss(
                components["fused"], query_ids, target_ids, torch.exp(model.log_temperature)
            )
            structure_loss = _multi_positive_contrastive_loss(
                components["structure"],
                query_ids,
                target_ids,
                torch.exp(model.log_temperature),
            )
            effective_structure_weight = (
                0.0 if fusion_mode == "semantic_only" else structure_aux_weight
            )
            loss = fused_loss + effective_structure_weight * structure_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            exact_losses.append(float(fused_loss.detach()))
            exact_structure_losses.append(float(structure_loss.detach()))

        parent_epoch = (
            balanced_parent_sample(parents, len(exact), randomizer)
            if balance_parent_sources
            else list(parents)
        )
        for batch in _batches(parent_epoch, batch_size, randomizer):
            query, query_mask, query_semantic, query_ids = _query_batch(
                batch, hp_profiles, mp_profiles, hp_index, mp_index, augment=True
            )
            target, target_mask, target_semantic = _rows(
                hp_profiles, [hp_index[row.hp_id] for row in batch]
            )
            components = model.score_components(
                query,
                query_mask,
                query_semantic,
                target,
                target_mask,
                target_semantic,
                "ancestor",
                semantic_dropout_probability,
            )
            target_ids = [row.hp_id for row in batch]
            fused_loss = _multi_positive_contrastive_loss(
                components["fused"], query_ids, target_ids, torch.exp(model.log_temperature)
            )
            structure_loss = _multi_positive_contrastive_loss(
                components["structure"],
                query_ids,
                target_ids,
                torch.exp(model.log_temperature),
            )
            effective_structure_weight = (
                0.0 if fusion_mode == "semantic_only" else structure_aux_weight
            )
            loss = fused_loss + effective_structure_weight * structure_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ancestor_losses.append(float(fused_loss.detach()))
            ancestor_structure_losses.append(float(structure_loss.detach()))

        ordering = [
            row
            for row in parent_epoch
            if row.source == "counterfactual_exact"
            and row.exact_hp_id in hp_index
            and row.mp_id in mp_index
        ]
        for batch in _batches(ordering, batch_size, randomizer):
            query, query_mask, query_semantic, _query_ids = _query_batch(
                batch, hp_profiles, mp_profiles, hp_index, mp_index, augment=False
            )
            exact_target, exact_mask, exact_semantic = _rows(
                hp_profiles, [hp_index[row.exact_hp_id] for row in batch]
            )
            parent_target, parent_mask, parent_semantic = _rows(
                hp_profiles, [hp_index[row.hp_id] for row in batch]
            )
            exact_score = model.paired_scores(
                query,
                query_mask,
                query_semantic,
                exact_target,
                exact_mask,
                exact_semantic,
                "exact",
            )
            parent_score = model.paired_scores(
                query,
                query_mask,
                query_semantic,
                parent_target,
                parent_mask,
                parent_semantic,
                "exact",
            )
            loss = torch.relu(0.12 - exact_score + parent_score).mean()
            optimizer.zero_grad()
            (0.5 * loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ordering_losses.append(float(loss.detach()))

        exact_weights = model.fusion_weight_dict("exact")
        ancestor_weights = model.fusion_weight_dict("ancestor")
        history.append(
            {
                "epoch": float(epoch + 1),
                "exact_loss": float(np.mean(exact_losses)) if exact_losses else 0.0,
                "exact_structure_loss": float(np.mean(exact_structure_losses))
                if exact_structure_losses
                else 0.0,
                "ancestor_loss": float(np.mean(ancestor_losses))
                if ancestor_losses
                else 0.0,
                "ancestor_structure_loss": float(np.mean(ancestor_structure_losses))
                if ancestor_structure_losses
                else 0.0,
                "ordering_loss": float(np.mean(ordering_losses))
                if ordering_losses
                else 0.0,
                "structure_aux_effective_weight": (
                    0.0 if fusion_mode == "semantic_only" else structure_aux_weight
                ),
                "temperature": float(torch.exp(model.log_temperature).detach()),
                "exact_semantic_weight": exact_weights["semantic"],
                "exact_structure_weight": exact_weights["structure"],
                "exact_poincare_weight": exact_weights["poincare"],
                "ancestor_semantic_weight": ancestor_weights["semantic"],
                "ancestor_structure_weight": ancestor_weights["structure"],
                "ancestor_poincare_weight": ancestor_weights["poincare"],
                "parent_examples_sampled": float(len(parent_epoch)),
            }
        )
    model.eval()
    return TrainedModel(model, history)


def encode_profiles(
    model: DualHeadEncoder,
    profiles: ProfileTensor,
    side: str,
    head: str,
    batch_size: int = 512,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(profiles.ids), batch_size):
            prototypes = torch.from_numpy(
                profiles.prototypes[start : start + batch_size]
            ).float()
            masks = torch.from_numpy(profiles.masks[start : start + batch_size]).bool()
            encoded.append(
                model.encode_structure(prototypes, masks, side, head).cpu().numpy()
            )
    return np.concatenate(encoded, axis=0).astype(np.float32)


def encode_semantics(
    model: DualHeadEncoder,
    profiles: ProfileTensor,
    side: str,
    head: str,
    batch_size: int = 512,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(profiles.ids), batch_size):
            semantic = torch.from_numpy(
                profiles.semantic_vectors[start : start + batch_size]
            ).float()
            encoded.append(model.encode_semantic(semantic, side, head).cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32)


def raw_profile_score(
    left: ProfileTensor,
    left_index: int,
    right: ProfileTensor,
    right_index: int,
) -> float:
    left_vectors = left.prototypes[left_index][left.masks[left_index]]
    right_vectors = right.prototypes[right_index][right.masks[right_index]]
    left_vectors = left_vectors / np.clip(
        np.linalg.norm(left_vectors, axis=1, keepdims=True), 1e-12, None
    )
    right_vectors = right_vectors / np.clip(
        np.linalg.norm(right_vectors, axis=1, keepdims=True), 1e-12, None
    )
    return float(np.max(left_vectors @ right_vectors.T))


def poincare_profile_score(
    left: ProfileTensor,
    left_index: int,
    right: ProfileTensor,
    right_index: int,
    hierarchy_dimension: int,
) -> float:
    left_vectors = left.prototypes[left_index][left.masks[left_index]]
    right_vectors = right.prototypes[right_index][right.masks[right_index]]
    if not len(left_vectors) or not len(right_vectors):
        return -1.0
    left_expanded = left_vectors[:, None, :]
    right_expanded = right_vectors[None, :, :]
    cl_distance = _poincare_distance_numpy(
        left_expanded[..., :hierarchy_dimension],
        right_expanded[..., :hierarchy_dimension],
    )
    go_distance = _poincare_distance_numpy(
        left_expanded[..., hierarchy_dimension : 2 * hierarchy_dimension],
        right_expanded[..., hierarchy_dimension : 2 * hierarchy_dimension],
    )
    channel = np.sum(
        left_expanded[..., 2 * hierarchy_dimension :]
        * right_expanded[..., 2 * hierarchy_dimension :],
        axis=-1,
    )
    scores = (
        0.4 * (2.0 * np.exp(-cl_distance) - 1.0)
        + 0.4 * (2.0 * np.exp(-go_distance) - 1.0)
        + 0.2 * channel
    )
    return float(np.max(scores))


def pair_feature_vector(
    model: DualHeadEncoder,
    query_profile: ProfileTensor,
    query_index: int,
    hp_profiles: ProfileTensor,
    hp_index: int,
) -> np.ndarray:
    query, query_mask, query_semantic = _rows(query_profile, [query_index])
    target, target_mask, target_semantic = _rows(hp_profiles, [hp_index])
    with torch.no_grad():
        exact = model.paired_components(
            query,
            query_mask,
            query_semantic,
            target,
            target_mask,
            target_semantic,
            "exact",
        )
        ancestor = model.paired_components(
            query,
            query_mask,
            query_semantic,
            target,
            target_mask,
            target_semantic,
            "ancestor",
        )
    raw = raw_profile_score(query_profile, query_index, hp_profiles, hp_index)
    query_aliases = query_profile.aliases.get(query_profile.ids[query_index], set())
    hp_aliases = hp_profiles.aliases.get(hp_profiles.ids[hp_index], set())
    shared_aliases = query_aliases & hp_aliases
    alias_union = query_aliases | hp_aliases
    query_id = query_profile.ids[query_index]
    hp_id = hp_profiles.ids[hp_index]
    query_mechanisms = query_profile.mechanism_ids[query_id]
    hp_mechanisms = hp_profiles.mechanism_ids[hp_id]
    query_specific = query_profile.specific_mechanism_ids[query_id]
    hp_specific = hp_profiles.specific_mechanism_ids[hp_id]
    shared_any = len(query_mechanisms & hp_mechanisms)
    shared_specific = len(query_specific & hp_specific)
    union_any = len(query_mechanisms | hp_mechanisms)
    values = np.asarray(
        [
            float(exact["fused"][0]),
            float(ancestor["fused"][0]),
            float(exact["fused"][0] - ancestor["fused"][0]),
            float(exact["structure"][0]),
            float(ancestor["structure"][0]),
            float(exact["semantic"][0]),
            float(ancestor["semantic"][0]),
            float(exact["poincare"][0]),
            raw,
            float(bool(shared_aliases)),
            len(shared_aliases) / max(1, len(alias_union)),
            math.log1p(shared_any),
            math.log1p(shared_specific),
            shared_any / max(1, union_any),
            math.log1p(len(query_mechanisms)),
            math.log1p(len(hp_mechanisms)),
        ],
        dtype=np.float64,
    )
    return mask_pair_features(values, model.fusion_mode)


@dataclass
class RelationCalibrator:
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_names: list[str]
    source_weight_totals: dict[str, float]

    @property
    def classes(self) -> list[str]:
        return list(self.classifier.classes_)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(features))


def fit_relation_calibrator(
    model: DualHeadEncoder,
    hp_profiles: ProfileTensor,
    mp_profiles: ProfileTensor,
    expert_examples: Sequence[MappingExample],
    parent_examples: Sequence[MappingExample],
    seed: int = 20260726,
) -> RelationCalibrator:
    hp_index = hp_profiles.index
    mp_index = mp_profiles.index
    features: list[np.ndarray] = []
    labels: list[str] = []
    base_weights: list[float] = []
    sources: list[str] = []
    combined = list(expert_examples) + list(parent_examples)
    seen: set[tuple[str, str, str, str]] = set()
    for row in combined:
        if row.split != "train" or row.relation not in RELATION_CLASSES:
            continue
        location = _query_index(row, hp_index, mp_index)
        if location is None or row.hp_id not in hp_index:
            continue
        key = (row.mp_id, row.hp_id, row.relation, row.source)
        if key in seen:
            continue
        seen.add(key)
        source, query_index = location
        profile = hp_profiles if source == "hp" else mp_profiles
        features.append(
            pair_feature_vector(model, profile, query_index, hp_profiles, hp_index[row.hp_id])
        )
        labels.append(row.relation)
        base_weights.append(
            row.confidence
            * {"synthetic_hpo_cutout": 0.25, "counterfactual_exact": 0.65}.get(
                row.source, 1.0
            )
        )
        sources.append(row.source)
    if len(set(labels)) < 2:
        raise ValueError("Relation calibration requires at least two relation classes")
    source_base_totals: dict[str, float] = defaultdict(float)
    for source, weight in zip(sources, base_weights):
        source_base_totals[source] += weight
    target_total = len(base_weights) / max(1, len(source_base_totals))
    weights = np.asarray(
        [
            weight * target_total / source_base_totals[source]
            for source, weight in zip(sources, base_weights)
        ],
        dtype=np.float64,
    )
    source_weight_totals = {
        source: float(weights[np.asarray(sources) == source].sum())
        for source in sorted(source_base_totals)
    }
    matrix = np.stack(features)
    scaler = StandardScaler().fit(matrix)
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    ).fit(scaler.transform(matrix), labels, sample_weight=weights)
    return RelationCalibrator(
        scaler, classifier, list(PAIR_FEATURE_NAMES), source_weight_totals
    )


def flatten_valid_prototypes(
    encoded: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    term_indices, prototype_indices = np.nonzero(masks)
    return encoded[term_indices, prototype_indices], term_indices.astype(np.int32)
