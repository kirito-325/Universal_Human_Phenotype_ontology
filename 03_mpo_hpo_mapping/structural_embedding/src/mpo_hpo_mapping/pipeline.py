from __future__ import annotations

import dataclasses
import json
import platform
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
import torch

from .data import (
    InputPaths,
    MappingExample,
    MechanismEdge,
    build_parent_examples,
    file_fingerprint,
    load_cl_ontology,
    load_expert_examples,
    load_go_bp_ontology,
    load_hpo_ontology,
    load_term_profiles,
    randomize_term_profiles,
    write_json,
    write_tsv,
)
from .embedding import (
    HierarchyEmbedding,
    build_profile_tensor,
    constant_hierarchy_embedding,
    permute_hierarchy_embedding,
    train_poincare_hierarchy,
)
from .model import (
    PAIR_FEATURE_NAMES,
    fit_relation_calibrator,
    pair_feature_vector,
    train_dual_head,
)
from .retrieval import (
    AncestorRankWeights,
    DualHeadRetriever,
    fit_ancestor_rank_weights,
    fit_gate_calibrator,
    gate_feature_vector,
    rank_candidates,
)
from .semantic import SemanticEmbedding, build_semantic_embeddings, load_obo_term_text


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    general_top_k: int = 8
    general_weight: float = 0.15
    hierarchy_dimension: int = 16
    hierarchy_epochs: int = 30
    hierarchy_mode: str = "poincare"
    hierarchy_reference: str = ""
    hierarchy_permutation_seed: int = 0
    hpo_link_randomization: str = "none"
    randomization_swaps_per_edge: int = 10
    max_prototypes: int = 4
    semantic_mode: str = "auto"
    semantic_dimension: int = 64
    semantic_batch_size: int = 64
    semantic_reference: str = ""
    semantic_weight: float = 1.0
    semantic_seed: int = 0
    use_term_aliases: bool = True
    hidden_dimension: int = 64
    output_dimension: int = 32
    training_epochs: int = 80
    batch_size: int = 64
    structure_aux_weight: float = 0.35
    semantic_dropout_probability: float = 0.25
    balance_parent_sources: bool = True
    fusion_mode: str = "late_fusion"
    min_mapping_confidence: float = 0.9
    synthetic_limit: int = 0
    ivf_clusters: int = 96
    nprobe: int = 8
    retrieve_k: int = 80
    output_top_k: int = 20
    max_output_mp: int = 0
    evaluation_only: bool = False
    seed: int = 20260726
    data_split_seed: int = 0


MAPPING_FIELDS = [field.name for field in dataclasses.fields(MappingExample)]


def _edge_rows(
    ontology: str, profiles: Mapping[str, Sequence[MechanismEdge]]
) -> Iterable[dict[str, object]]:
    for term_id in sorted(profiles):
        for edge in profiles[term_id]:
            yield {
                "ontology": ontology,
                "term_id": edge.term_id,
                "term_name": edge.term_name,
                "mechanism_id": edge.mechanism_id,
                "cl_id": edge.cl_id,
                "cl_name": edge.cl_name,
                "go_id": edge.go_id,
                "go_name": edge.go_name,
                "channel": edge.channel,
                "weight": f"{edge.weight:.8g}",
                "rank": edge.rank,
                "confidence": edge.confidence,
                "supporting_genes": edge.supporting_genes,
            }


def _mapping_rows(examples: Sequence[MappingExample]) -> Iterable[dict[str, object]]:
    for row in examples:
        yield asdict(row)


def _save_embeddings(
    path: Path,
    cl_ids: Sequence[str],
    cl_vectors: np.ndarray,
    go_ids: Sequence[str],
    go_vectors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cl_ids=np.asarray(cl_ids),
        cl_vectors=cl_vectors,
        go_ids=np.asarray(go_ids),
        go_vectors=go_vectors,
        coordinate_system=np.asarray(["poincare_ball"]),
    )


def _save_profiles(path: Path, prefix: str, profile: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(profile.ids),
        names=np.asarray(profile.names),
        prototypes=profile.prototypes,
        masks=profile.masks,
        edge_counts=profile.edge_counts,
        specific_counts=profile.specific_counts,
        semantic_vectors=profile.semantic_vectors,
        ontology=np.asarray([prefix]),
    )


def _score_metrics(ranks: Sequence[int], query_count: int) -> dict[str, object]:
    positive = [rank for rank in ranks if rank > 0]
    return {
        "queries": query_count,
        "retrieved_targets": len(positive),
        "recall_at_1": sum(rank <= 1 for rank in positive) / max(1, query_count),
        "recall_at_10": sum(rank <= 10 for rank in positive) / max(1, query_count),
        "recall_at_50": sum(rank <= 50 for rank in positive) / max(1, query_count),
        "mrr": sum(1.0 / rank for rank in positive) / max(1, query_count),
    }


def _best_rank(candidates: Sequence[Mapping[str, object]], targets: set[str]) -> int:
    return min(
        (int(row["final_rank"]) for row in candidates if str(row["hp_id"]) in targets),
        default=0,
    )


def _strong_exact_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    strong = [
        row
        for row in candidates
        if int(row.get("shared_alias_count", 0)) > 0
        and float(row.get("alias_jaccard", 0.0)) >= 0.20
        and float(row.get("semantic_score", -1.0)) >= 0.80
    ]
    return sorted(
        strong,
        key=lambda row: (
            -float(row.get("alias_jaccard", 0.0)),
            -float(row.get("semantic_score", -1.0)),
            -float(row.get("p_exact", 0.0)),
            str(row.get("hp_id", "")),
        ),
    )


def _retrieve_ranked(
    retriever: DualHeadRetriever,
    gate: object,
    hpo_ontology: object,
    config: RunConfig,
    source: str,
    query_index: int,
    excluded: set[str] | None = None,
    ancestor_weights: AncestorRankWeights | None = None,
) -> tuple[list[dict[str, object]], float]:
    excluded = excluded or set()
    candidates = retriever.retrieve(
        source, query_index, config.retrieve_k, config.nprobe, excluded
    )
    probability = gate.probability(gate_feature_vector(candidates))
    strong_exact = _strong_exact_candidates(candidates)
    exact_available = probability >= gate.threshold or bool(strong_exact)
    override_reason = (
        "strong_alias_semantic"
        if probability < gate.threshold and strong_exact
        else ""
    )
    strong_ids = {str(row["hp_id"]) for row in strong_exact}
    for row in candidates:
        row["exact_evidence_override"] = int(str(row["hp_id"]) in strong_ids)
    derived: dict[str, tuple[int, set[str]]] = {}
    if exact_available:
        parent_sources = (
            strong_exact
            if strong_exact
            else [
                row
                for row in candidates
                if float(row.get("p_exact", 0.0)) >= 0.5
            ]
        )
        exact_candidates = sorted(
            parent_sources,
            key=lambda row: (
                -int(row.get("exact_evidence_override", 0)),
                -float(row.get("p_exact", 0.0)),
            ),
        )[:4]
        for exact in exact_candidates:
            exact_id = str(exact["hp_id"])
            ancestors, distance = hpo_ontology.nearest_available_ancestors(
                exact_id,
                set(retriever.hp_profiles.ids),
                excluded=excluded | {exact_id},
            )
            for ancestor in ancestors:
                previous_distance, sources = derived.get(ancestor, (distance, set()))
                derived[ancestor] = (min(previous_distance, distance), sources | {exact_id})
        if derived:
            candidates = retriever.retrieve(
                source,
                query_index,
                config.retrieve_k,
                config.nprobe,
                excluded,
                set(derived),
            )
            probability = gate.probability(gate_feature_vector(candidates))
            strong_exact = _strong_exact_candidates(candidates)
            strong_ids = {str(row["hp_id"]) for row in strong_exact}
            for row in candidates:
                hp_id = str(row["hp_id"])
                row["exact_evidence_override"] = int(hp_id in strong_ids)
                if hp_id in derived:
                    distance, sources = derived[hp_id]
                    row["derived_ancestor_distance"] = distance
                    row["derived_from_exact_ids"] = ";".join(sorted(sources))
    ranked = rank_candidates(
        candidates,
        probability,
        gate.threshold,
        ancestor_weights,
        exact_available=exact_available,
    )
    for row in ranked:
        row["gate_override_reason"] = override_reason
    return ranked, probability


def _evaluate_real(
    retriever: DualHeadRetriever,
    gate: object,
    cache: dict[str, tuple[list[dict[str, object]], float]],
    examples: Sequence[MappingExample],
    relation: str,
    config: RunConfig,
    hpo_ontology: object,
    ancestor_weights: AncestorRankWeights,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        if row.split == "test" and row.relation == relation:
            targets[row.mp_id].add(row.hp_id)
    ranks: list[int] = []
    cases: list[dict[str, object]] = []
    for mp_id, hp_ids in sorted(targets.items()):
        if mp_id not in cache and mp_id in retriever.mp_index:
            candidates, probability = _retrieve_ranked(
                retriever,
                gate,
                hpo_ontology,
                config,
                "mp",
                retriever.mp_index[mp_id],
                ancestor_weights=ancestor_weights,
            )
            cache[mp_id] = (candidates, probability)
        candidates, gate_probability = cache.get(mp_id, ([], 0.0))
        rank = _best_rank(candidates, hp_ids)
        ranks.append(rank)
        cases.append(
            {
                "task": relation,
                "query_id": mp_id,
                "targets": ";".join(sorted(hp_ids)),
                "best_rank": rank,
                "gate_probability": f"{gate_probability:.8g}",
            }
        )
    return _score_metrics(ranks, len(targets)), cases


def _evaluate_masked(
    retriever: DualHeadRetriever,
    gate: object,
    examples: Sequence[MappingExample],
    config: RunConfig,
    source_name: str,
    hpo_ontology: object,
    ancestor_weights: AncestorRankWeights,
    max_queries: int = 0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[MappingExample]] = defaultdict(list)
    for row in examples:
        if row.split == "test" and row.source == source_name:
            grouped[(row.mp_id, row.masked_hp_ids)].append(row)
    items = sorted(grouped.items())
    if max_queries > 0:
        items = items[:max_queries]
    ranks: list[int] = []
    cases: list[dict[str, object]] = []
    gate_probabilities: list[float] = []
    for (query_id, masked), rows in items:
        first = rows[0]
        if first.query_profile_source == "hp":
            source = "hp"
            query_index = retriever.hp_index.get(first.exact_hp_id)
        else:
            source = "mp"
            query_index = retriever.mp_index.get(first.mp_id)
        if query_index is None:
            continue
        excluded = set(filter(None, masked.split(";")))
        candidates, probability = _retrieve_ranked(
            retriever,
            gate,
            hpo_ontology,
            config,
            source,
            query_index,
            excluded,
            ancestor_weights,
        )
        targets = {row.hp_id for row in rows}
        rank = _best_rank(candidates, targets)
        ranks.append(rank)
        gate_probabilities.append(probability)
        cases.append(
            {
                "task": source_name,
                "query_id": query_id,
                "targets": ";".join(sorted(targets)),
                "best_rank": rank,
                "gate_probability": f"{probability:.8g}",
            }
        )
    metrics = _score_metrics(ranks, len(ranks))
    metrics["mean_gate_probability"] = float(np.mean(gate_probabilities)) if gate_probabilities else 0.0
    return metrics, cases


def run_pipeline(paths: InputPaths, config: RunConfig) -> dict[str, object]:
    if config.hierarchy_mode not in {"poincare", "constant", "permuted"}:
        raise ValueError(f"Unsupported hierarchy mode: {config.hierarchy_mode}")
    if config.hpo_link_randomization not in {"none", "degree_preserving"}:
        raise ValueError(
            f"Unsupported HPO link randomization: {config.hpo_link_randomization}"
        )
    if config.fusion_mode not in {"late_fusion", "semantic_only", "structure_only"}:
        raise ValueError(f"Unsupported fusion mode: {config.fusion_mode}")
    paths.validate()
    output = config.output_dir
    processed = output / "data/processed"
    model_dir = output / "model"
    output.mkdir(parents=True, exist_ok=True)

    hp_raw = load_term_profiles(
        paths.hpo_specific,
        paths.hpo_general,
        "hpo_id",
        config.general_top_k,
        config.general_weight,
    )
    randomization_stats: dict[str, object] | None = None
    if config.hpo_link_randomization == "degree_preserving":
        hp_raw, randomization_stats = randomize_term_profiles(
            hp_raw,
            config.seed + 2,
            config.randomization_swaps_per_edge,
        )
    mp_raw = load_term_profiles(
        paths.mpo_specific,
        paths.mpo_general,
        "mp_id",
        config.general_top_k,
        config.general_weight,
    )
    hpo_ontology = load_hpo_ontology(paths.hpo_edges)
    split_seed = config.data_split_seed or config.seed
    expert = load_expert_examples(
        paths.expert_mappings,
        set(mp_raw),
        set(hp_raw),
        config.min_mapping_confidence,
        split_seed,
    )
    parents = build_parent_examples(
        expert,
        hpo_ontology,
        set(hp_raw),
        config.synthetic_limit,
        split_seed,
    )

    edge_fields = [
        "ontology",
        "term_id",
        "term_name",
        "mechanism_id",
        "cl_id",
        "cl_name",
        "go_id",
        "go_name",
        "channel",
        "weight",
        "rank",
        "confidence",
        "supporting_genes",
    ]
    edge_count = write_tsv(
        processed / "term_mechanism_edges.tsv",
        list(_edge_rows("HPO", hp_raw)) + list(_edge_rows("MPO", mp_raw)),
        edge_fields,
    )
    write_tsv(
        processed / "expert_mapping_examples.tsv", _mapping_rows(expert), MAPPING_FIELDS
    )
    write_tsv(
        processed / "parent_training_examples.tsv", _mapping_rows(parents), MAPPING_FIELDS
    )

    cl_ontology = load_cl_ontology(paths.cl_catalog)
    go_ontology = load_go_bp_ontology(paths.go_obo)
    profile_cl_ids = {
        edge.cl_id for profiles in (hp_raw, mp_raw) for rows in profiles.values() for edge in rows
    }
    profile_go_ids = {
        edge.go_id for profiles in (hp_raw, mp_raw) for rows in profiles.values() for edge in rows
    }
    cl_ontology.nodes.update(profile_cl_ids)
    go_ontology.nodes.update(profile_go_ids)

    hierarchy_reference_fingerprint: dict[str, object] | None = None
    if config.hierarchy_mode == "constant":
        cl_embedding = constant_hierarchy_embedding(
            cl_ontology, config.hierarchy_dimension
        )
        go_embedding = constant_hierarchy_embedding(
            go_ontology, config.hierarchy_dimension
        )
    elif config.hierarchy_mode in {"poincare", "permuted"} and config.hierarchy_reference:
        reference_path = Path(config.hierarchy_reference)
        hierarchy_reference_fingerprint = file_fingerprint(reference_path)
        with np.load(reference_path, allow_pickle=False) as reference:
            coordinate_system = (
                str(reference["coordinate_system"][0])
                if "coordinate_system" in reference.files
                else "unknown"
            )
            if coordinate_system != "poincare_ball":
                raise ValueError(
                    "v2 hierarchy references must contain Poincare-ball coordinates"
                )
            cl_embedding = HierarchyEmbedding(
                reference["cl_ids"].astype(str).tolist(),
                reference["cl_vectors"].astype(np.float32),
                [],
            )
            go_embedding = HierarchyEmbedding(
                reference["go_ids"].astype(str).tolist(),
                reference["go_vectors"].astype(np.float32),
                [],
            )
        if cl_embedding.ids != sorted(cl_ontology.nodes):
            raise ValueError("CL hierarchy reference node set does not match current ontology")
        if go_embedding.ids != sorted(go_ontology.nodes):
            raise ValueError("GO hierarchy reference node set does not match current ontology")
        if config.hierarchy_mode == "permuted":
            permutation_seed = config.hierarchy_permutation_seed or (
                config.seed + 10_001
            )
            cl_embedding = permute_hierarchy_embedding(
                cl_embedding, seed=permutation_seed
            )
            go_embedding = permute_hierarchy_embedding(
                go_embedding, seed=permutation_seed + 1
            )
    else:
        cl_embedding = train_poincare_hierarchy(
            cl_ontology,
            config.hierarchy_dimension,
            config.hierarchy_epochs,
            seed=config.seed,
        )
        go_embedding = train_poincare_hierarchy(
            go_ontology,
            config.hierarchy_dimension,
            config.hierarchy_epochs,
            seed=config.seed + 1,
        )
        if config.hierarchy_mode == "permuted":
            permutation_seed = config.hierarchy_permutation_seed or (
                config.seed + 10_001
            )
            cl_embedding = permute_hierarchy_embedding(
                cl_embedding, seed=permutation_seed
            )
            go_embedding = permute_hierarchy_embedding(
                go_embedding, seed=permutation_seed + 1
            )
    _save_embeddings(
        processed / "hierarchy_embeddings.npz",
        cl_embedding.ids,
        cl_embedding.vectors,
        go_embedding.ids,
        go_embedding.vectors,
    )

    term_ids = sorted(set(hp_raw) | set(mp_raw))
    semantic_texts = {
        **load_obo_term_text(paths.hpo_obo, set(hp_raw)),
        **load_obo_term_text(paths.mpo_obo, set(mp_raw)),
    }
    semantic_reference_fingerprint: dict[str, object] | None = None
    semantic_output = processed / "semantic_embeddings.npz"
    if config.semantic_reference:
        semantic_reference_path = Path(config.semantic_reference)
        semantic_reference_fingerprint = file_fingerprint(semantic_reference_path)
        with np.load(semantic_reference_path, allow_pickle=False) as reference:
            semantic = SemanticEmbedding(
                ids=reference["ids"].astype(str).tolist(),
                vectors=reference["vectors"].astype(np.float32),
                method=str(reference["method"][0]),
                source=str(reference["source"][0]),
            )
        if semantic.ids != term_ids:
            raise ValueError("Semantic reference term set does not match current profiles")
        if semantic.vectors.shape != (len(term_ids), config.semantic_dimension):
            raise ValueError(
                "Semantic reference shape does not match configured semantic dimension"
            )
        if not np.isfinite(semantic.vectors).all():
            raise ValueError("Semantic reference contains non-finite values")
        semantic_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(semantic_reference_path, semantic_output)
    else:
        semantic = build_semantic_embeddings(
            term_ids,
            semantic_texts,
            config.semantic_mode,
            config.semantic_dimension,
            config.semantic_batch_size,
            config.semantic_seed or config.seed,
        )
        np.savez_compressed(
            semantic_output,
            ids=np.asarray(semantic.ids),
            vectors=semantic.vectors,
            method=np.asarray([semantic.method]),
            source=np.asarray([semantic.source]),
        )
    semantic_vectors = semantic.as_dict()
    term_aliases = (
        {
            term_id: text.normalized_aliases()
            for term_id, text in semantic_texts.items()
        }
        if config.use_term_aliases
        else {}
    )
    hp_profiles = build_profile_tensor(
        hp_raw,
        cl_embedding.as_dict(),
        go_embedding.as_dict(),
        config.max_prototypes,
        semantic_vectors,
        config.semantic_weight,
        term_aliases,
    )
    mp_profiles = build_profile_tensor(
        mp_raw,
        cl_embedding.as_dict(),
        go_embedding.as_dict(),
        config.max_prototypes,
        semantic_vectors,
        config.semantic_weight,
        term_aliases,
    )
    _save_profiles(processed / "hpo_profiles.npz", "HPO", hp_profiles)
    _save_profiles(processed / "mpo_profiles.npz", "MPO", mp_profiles)

    trained = train_dual_head(
        hp_profiles,
        mp_profiles,
        expert,
        parents,
        config.training_epochs,
        config.batch_size,
        config.hidden_dimension,
        config.output_dimension,
        seed=config.seed,
        hierarchy_dimension=config.hierarchy_dimension,
        structure_aux_weight=config.structure_aux_weight,
        semantic_dropout_probability=config.semantic_dropout_probability,
        balance_parent_sources=config.balance_parent_sources,
        fusion_mode=config.fusion_mode,
    )
    relation = fit_relation_calibrator(
        trained.model, hp_profiles, mp_profiles, expert, parents, config.seed
    )
    retriever = DualHeadRetriever(
        trained.model,
        hp_profiles,
        mp_profiles,
        relation,
        config.ivf_clusters,
        config.seed,
    )
    gate = fit_gate_calibrator(
        retriever,
        expert,
        parents,
        config.retrieve_k,
        config.nprobe,
        config.seed,
    )
    exact_mp_ids = {row.mp_id for row in expert if row.relation == "exact"}
    ancestor_weights = fit_ancestor_rank_weights(
        retriever,
        parents,
        exact_mp_ids,
        config.retrieve_k,
        config.nprobe,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "structure_dimension": hp_profiles.prototypes.shape[-1],
            "semantic_dimension": hp_profiles.semantic_vectors.shape[-1],
            "hierarchy_dimension": config.hierarchy_dimension,
            "hidden_dimension": config.hidden_dimension,
            "output_dimension": config.output_dimension,
            "fusion_weights": {
                head: trained.model.fusion_weight_dict(head)
                for head in ("exact", "ancestor")
            },
            "fusion_mode": config.fusion_mode,
            "version": "0.2.0",
        },
        model_dir / "dual_head_encoder.pt",
    )
    joblib.dump(relation, model_dir / "relation_calibrator.joblib")
    joblib.dump(gate, model_dir / "availability_gate.joblib")
    write_tsv(
        model_dir / "training_loss.tsv",
        trained.loss_history,
        list(trained.loss_history[0]) if trained.loss_history else ["epoch"],
    )
    relation_coefficients = {
        label: {
            feature: float(coefficient)
            for feature, coefficient in zip(
                relation.feature_names, relation.classifier.coef_[class_index]
            )
        }
        for class_index, label in enumerate(relation.classes)
    }
    fusion_attribution = {
        "model_version": "0.2.0",
        "fusion_mode": config.fusion_mode,
        "score_definition": {
            "semantic": "learned semantic-tower cosine",
            "structure": "learned structure-tower max cosine",
            "poincare": "direct CL/GO Poincare distance plus channel agreement",
            "fusion": "non-negative softmax weights over semantic, structure, poincare",
        },
        "fusion_weights": {
            head: trained.model.fusion_weight_dict(head)
            for head in ("exact", "ancestor")
        },
        "relation_feature_names": list(PAIR_FEATURE_NAMES),
        "relation_coefficients_standardized": relation_coefficients,
        "relation_source_weight_totals": relation.source_weight_totals,
        "ancestor_rank_weights": asdict(ancestor_weights),
    }
    write_json(model_dir / "fusion_attribution.json", fusion_attribution)

    mapping_fields = [
        "mp_id",
        "mp_name",
        "final_rank",
        "hp_id",
        "hp_name",
        "predicted_relation",
        "final_score",
        "gate_probability",
        "gate_override_reason",
        "exact_available_prediction",
        "gate_threshold",
        "p_exact",
        "p_hp_broader",
        "p_hp_narrower",
        "p_related",
        "exact_embedding_score",
        "ancestor_embedding_score",
        "exact_structure_score",
        "ancestor_structure_score",
        "exact_semantic_tower_score",
        "ancestor_semantic_tower_score",
        "poincare_score",
        "raw_graph_score",
        "semantic_score",
        "ancestor_rank_score",
        "shared_alias_count",
        "shared_aliases",
        "alias_jaccard",
        "exact_evidence_override",
        "exact_retrieval_rank",
        "ancestor_retrieval_rank",
        "raw_retrieval_rank",
        "semantic_retrieval_rank",
        "alias_retrieval",
        "shared_mechanism_count",
        "shared_specific_mechanism_count",
        "mechanism_jaccard",
        "shared_mechanisms",
        "derived_ancestor_distance",
        "derived_from_exact_ids",
    ]
    mapping_rows: list[dict[str, object]] = []
    real_cache: dict[str, tuple[list[dict[str, object]], float]] = {}
    output_mp_ids = (
        []
        if config.evaluation_only
        else mp_profiles.ids[: config.max_output_mp or None]
    )
    for mp_id in output_mp_ids:
        mp_position = retriever.mp_index[mp_id]
        ranked, gate_probability = _retrieve_ranked(
            retriever,
            gate,
            hpo_ontology,
            config,
            "mp",
            mp_position,
            ancestor_weights=ancestor_weights,
        )
        real_cache[mp_id] = (ranked, gate_probability)
        for candidate in ranked[: config.output_top_k]:
            mapping_rows.append(
                {
                    "mp_id": mp_id,
                    "mp_name": mp_profiles.names[mp_position],
                    **candidate,
                    "gate_threshold": gate.threshold,
                }
            )
    mapping_count = write_tsv(
        output / "mpo_hpo_mapping_final.tsv", mapping_rows, mapping_fields
    )

    exact_metrics, exact_cases = _evaluate_real(
        retriever,
        gate,
        real_cache,
        expert,
        "exact",
        config,
        hpo_ontology,
        ancestor_weights,
    )
    broad_metrics, broad_cases = _evaluate_real(
        retriever,
        gate,
        real_cache,
        expert,
        "hp_broader",
        config,
        hpo_ontology,
        ancestor_weights,
    )
    counterfactual_metrics, counterfactual_cases = _evaluate_masked(
        retriever,
        gate,
        parents,
        config,
        "counterfactual_exact",
        hpo_ontology,
        ancestor_weights,
    )
    synthetic_metrics, synthetic_cases = _evaluate_masked(
        retriever,
        gate,
        parents,
        config,
        "synthetic_hpo_cutout",
        hpo_ontology,
        ancestor_weights,
        max_queries=500 if config.max_output_mp else 0,
    )
    evaluation_cases = exact_cases + broad_cases + counterfactual_cases + synthetic_cases
    write_tsv(
        output / "evaluation_cases.tsv",
        evaluation_cases,
        ["task", "query_id", "targets", "best_rank", "gate_probability"],
    )
    evaluation = {
        "expert_exact_present": exact_metrics,
        "expert_hp_broader": broad_metrics,
        "counterfactual_exact_removed": counterfactual_metrics,
        "synthetic_hpo_cutout": synthetic_metrics,
        "gate_threshold": gate.threshold,
    }
    write_json(output / "evaluation_summary.json", evaluation)

    expert_counts = Counter(row.relation for row in expert)
    parent_counts = Counter(row.source for row in parents)
    manifest = {
        "generated_by": "mpo_hpo_mapping.pipeline",
        "config": {**asdict(config), "output_dir": str(config.output_dir.resolve())},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "inputs": {name: file_fingerprint(path) for name, path in asdict(paths).items()},
        "data": {
            "hpo_profile_terms": len(hp_profiles.ids),
            "mpo_profile_terms": len(mp_profiles.ids),
            "term_mechanism_edges": edge_count,
            "expert_examples": dict(sorted(expert_counts.items())),
            "parent_examples": dict(sorted(parent_counts.items())),
            "cl_nodes": len(cl_embedding.ids),
            "go_bp_nodes": len(go_embedding.ids),
            "semantic_method": semantic.method,
            "semantic_source": semantic.source,
            "semantic_dimension": semantic.vectors.shape[1],
            "semantic_reference": semantic_reference_fingerprint,
            "structure_dimension": hp_profiles.prototypes.shape[-1],
            "hierarchy_coordinate_system": "poincare_ball",
            "hierarchy_mode": config.hierarchy_mode,
            "hierarchy_randomization": (
                "deterministic_node_permutation_preserving_vector_distribution"
                if config.hierarchy_mode == "permuted"
                else "none"
            ),
            "hierarchy_reference": hierarchy_reference_fingerprint,
            "hierarchy_permutation_seed": (
                config.hierarchy_permutation_seed
                if config.hierarchy_mode == "permuted"
                else 0
            ),
            "data_split_seed": split_seed,
            "semantic_seed": config.semantic_seed or config.seed,
            "cl_unique_vectors": int(np.unique(cl_embedding.vectors, axis=0).shape[0]),
            "go_unique_vectors": int(np.unique(go_embedding.vectors, axis=0).shape[0]),
            "hpo_link_randomization": randomization_stats,
        },
        "training": {
            "final_loss": trained.loss_history[-1] if trained.loss_history else {},
            "relation_classes": relation.classes,
            "gate_threshold": gate.threshold,
            "ancestor_rank_weights": asdict(ancestor_weights),
            "fusion_weights": fusion_attribution["fusion_weights"],
            "relation_feature_names": list(PAIR_FEATURE_NAMES),
            "relation_source_weight_totals": relation.source_weight_totals,
            "structure_aux_weight": config.structure_aux_weight,
            "semantic_dropout_probability": config.semantic_dropout_probability,
            "balance_parent_sources": config.balance_parent_sources,
            "fusion_mode": config.fusion_mode,
        },
        "output": {
            "mapping_rows": mapping_count,
            "mapped_mpo_terms": len(output_mp_ids),
        },
        "evaluation": evaluation,
    }
    write_json(output / "training_manifest.json", manifest)
    return manifest
