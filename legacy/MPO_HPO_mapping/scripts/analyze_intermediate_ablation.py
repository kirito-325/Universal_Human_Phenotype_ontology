#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import (  # noqa: E402
    InputPaths,
    MappingExample,
    MechanismEdge,
    load_hpo_ontology,
)
from mpo_hpo_mapping.embedding import ProfileTensor, build_profile_tensor  # noqa: E402
from mpo_hpo_mapping.model import fit_relation_calibrator, train_dual_head  # noqa: E402
from mpo_hpo_mapping.pipeline import (  # noqa: E402
    RunConfig,
    _evaluate_masked,
    _evaluate_real,
)
from mpo_hpo_mapping.retrieval import (  # noqa: E402
    DualHeadRetriever,
    fit_ancestor_rank_weights,
    fit_gate_calibrator,
)
from mpo_hpo_mapping.semantic import load_obo_term_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CL/GO intermediate-layer contribution with controlled ablations."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=MODULE_DIR / "results/full_sapbert_final_20260727",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODULE_DIR / "results/intermediate_layer_ablation_20260727.json",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("semantic_only", "intermediate_only"),
        default=("semantic_only", "intermediate_only"),
    )
    return parser.parse_args()


def read_mapping_examples(path: Path) -> list[MappingExample]:
    rows: list[MappingExample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                MappingExample(
                    mp_id=row["mp_id"],
                    hp_id=row["hp_id"],
                    relation=row["relation"],
                    confidence=float(row["confidence"]),
                    split=row["split"],
                    source=row["source"],
                    query_profile_source=row["query_profile_source"],
                    exact_hp_id=row["exact_hp_id"],
                    ancestor_distance=int(row["ancestor_distance"] or 0),
                    masked_hp_ids=row["masked_hp_ids"],
                )
            )
    return rows


def read_mechanism_edges(
    path: Path,
) -> tuple[dict[str, list[MechanismEdge]], dict[str, list[MechanismEdge]]]:
    profiles: dict[str, dict[str, list[MechanismEdge]]] = {
        "HPO": defaultdict(list),
        "MPO": defaultdict(list),
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            edge = MechanismEdge(
                term_id=row["term_id"],
                term_name=row["term_name"],
                cl_id=row["cl_id"],
                cl_name=row["cl_name"],
                go_id=row["go_id"],
                go_name=row["go_name"],
                channel=row["channel"],
                weight=float(row["weight"]),
                rank=int(row["rank"]),
                confidence=row["confidence"],
                supporting_genes=row["supporting_genes"],
            )
            profiles[row["ontology"]][edge.term_id].append(edge)
    return dict(profiles["HPO"]), dict(profiles["MPO"])


def load_saved_profile(
    path: Path,
    aliases: dict[str, set[str]],
    edges: dict[str, list[MechanismEdge]],
) -> ProfileTensor:
    archive = np.load(path, allow_pickle=False)
    ids = archive["ids"].astype(str).tolist()
    return ProfileTensor(
        ids=ids,
        names=archive["names"].astype(str).tolist(),
        prototypes=archive["prototypes"].astype(np.float32),
        masks=archive["masks"].astype(bool),
        edge_counts=archive["edge_counts"].astype(np.int32),
        specific_counts=archive["specific_counts"].astype(np.int32),
        semantic_vectors=archive["semantic_vectors"].astype(np.float32),
        aliases={term_id: aliases.get(term_id, set()) for term_id in ids},
        mechanism_ids={
            term_id: {edge.mechanism_id for edge in edges[term_id]} for term_id in ids
        },
        specific_mechanism_ids={
            term_id: {
                edge.mechanism_id for edge in edges[term_id] if edge.channel == "specific"
            }
            for term_id in ids
        },
    )


def semantic_only(profile: ProfileTensor) -> ProfileTensor:
    count = len(profile.ids)
    return ProfileTensor(
        ids=profile.ids,
        names=profile.names,
        prototypes=profile.semantic_vectors[:, None, :].copy(),
        masks=np.ones((count, 1), dtype=bool),
        edge_counts=np.zeros(count, dtype=np.int32),
        specific_counts=np.zeros(count, dtype=np.int32),
        semantic_vectors=profile.semantic_vectors.copy(),
        aliases=profile.aliases,
        mechanism_ids={term_id: set() for term_id in profile.ids},
        specific_mechanism_ids={term_id: set() for term_id in profile.ids},
    )


def score_cases(cases: Iterable[dict[str, object]]) -> dict[str, float | int]:
    rows = list(cases)
    ranks = [int(row["best_rank"] or 0) for row in rows]
    result: dict[str, float | int] = {
        "queries": len(rows),
        "retrieved_targets": sum(rank > 0 for rank in ranks),
        "mrr": sum((1.0 / rank if rank else 0.0) for rank in ranks) / max(1, len(ranks)),
    }
    for cutoff in (1, 3, 5, 10, 20, 50, 100):
        result[f"recall_at_{cutoff}"] = sum(
            0 < rank <= cutoff for rank in ranks
        ) / max(1, len(ranks))
    return result


def load_formal_evaluation(result_dir: Path) -> dict[str, dict[str, float | int]]:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    with (result_dir / "evaluation_cases.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            by_task[row["task"]].append(row)
    return summarize_cases(by_task)


def summarize_cases(
    by_task: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, float | int]]:
    summary = {
        "expert_exact_present": score_cases(by_task["exact"]),
        "expert_hp_broader": score_cases(by_task["hp_broader"]),
        "counterfactual_exact_removed": score_cases(by_task["counterfactual_exact"]),
        "synthetic_hpo_cutout": score_cases(by_task["synthetic_hpo_cutout"]),
    }
    summary["exact_absent_combined"] = score_cases(
        by_task["counterfactual_exact"] + by_task["synthetic_hpo_cutout"]
    )
    return summary


def run_model(
    mode: str,
    hp_profiles: ProfileTensor,
    mp_profiles: ProfileTensor,
    expert: list[MappingExample],
    parents: list[MappingExample],
    hpo_ontology: object,
    config: RunConfig,
) -> dict[str, object]:
    started = time.monotonic()
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

    real_cache: dict[str, tuple[list[dict[str, object]], float]] = {}
    _, exact_cases = _evaluate_real(
        retriever,
        gate,
        real_cache,
        expert,
        "exact",
        config,
        hpo_ontology,
        ancestor_weights,
    )
    _, broad_cases = _evaluate_real(
        retriever,
        gate,
        real_cache,
        expert,
        "hp_broader",
        config,
        hpo_ontology,
        ancestor_weights,
    )
    _, counterfactual_cases = _evaluate_masked(
        retriever,
        gate,
        parents,
        config,
        "counterfactual_exact",
        hpo_ontology,
        ancestor_weights,
    )
    _, synthetic_cases = _evaluate_masked(
        retriever,
        gate,
        parents,
        config,
        "synthetic_hpo_cutout",
        hpo_ontology,
        ancestor_weights,
    )
    cases = {
        "exact": exact_cases,
        "hp_broader": broad_cases,
        "counterfactual_exact": counterfactual_cases,
        "synthetic_hpo_cutout": synthetic_cases,
    }
    return {
        "mode": mode,
        "profile_dimension": int(hp_profiles.prototypes.shape[-1]),
        "max_prototypes": int(hp_profiles.prototypes.shape[1]),
        "gate_threshold": gate.threshold,
        "ancestor_rank_weights": {
            "ancestor_embedding": ancestor_weights.ancestor_embedding,
            "semantic": ancestor_weights.semantic,
            "relation_probability": ancestor_weights.relation_probability,
        },
        "runtime_seconds": time.monotonic() - started,
        "evaluation": summarize_cases(cases),
    }


def metric_delta(
    full: dict[str, dict[str, float | int]],
    ablation: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    keys = (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr",
    )
    return {
        task: {
            key: float(full[task][key]) - float(ablation[task][key]) for key in keys
        }
        for task in full
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    processed = result_dir / "data/processed"
    paths = InputPaths()

    hp_edges, mp_edges = read_mechanism_edges(processed / "term_mechanism_edges.tsv")
    hp_ids = set(hp_edges)
    mp_ids = set(mp_edges)
    texts = {
        **load_obo_term_text(paths.hpo_obo, hp_ids),
        **load_obo_term_text(paths.mpo_obo, mp_ids),
    }
    aliases = {term_id: text.normalized_aliases() for term_id, text in texts.items()}
    full_hp = load_saved_profile(processed / "hpo_profiles.npz", aliases, hp_edges)
    full_mp = load_saved_profile(processed / "mpo_profiles.npz", aliases, mp_edges)
    expert = read_mapping_examples(processed / "expert_mapping_examples.tsv")
    parents = read_mapping_examples(processed / "parent_training_examples.tsv")
    hpo_ontology = load_hpo_ontology(paths.hpo_edges)

    config = RunConfig(output_dir=args.output)
    formal = load_formal_evaluation(result_dir)
    report: dict[str, object] = {
        "formal_result": str(result_dir),
        "design": {
            "controlled_term_coverage": {
                "mpo": len(full_mp.ids),
                "hpo": len(full_hp.ids),
            },
            "same_split_seed": config.seed,
            "semantic_only": "SapBERT + normalized aliases; CL/GO and mechanism overlap removed",
            "intermediate_only": "CL + GO + channel + mechanism overlap; SapBERT and aliases removed",
            "note": "This measures ranking contribution conditional on terms covered by the intermediate layer.",
        },
        "full": {"evaluation": formal},
        "ablations": {},
    }

    node_embedding_path = processed / "fused_node_embeddings.npz"
    if not node_embedding_path.is_file():
        node_embedding_path = processed / "hierarchy_embeddings.npz"
    node_embeddings = np.load(node_embedding_path, allow_pickle=False)
    cl_vectors = dict(
        zip(node_embeddings["cl_ids"].astype(str), node_embeddings["cl_vectors"])
    )
    go_vectors = dict(
        zip(node_embeddings["go_ids"].astype(str), node_embeddings["go_vectors"])
    )
    report["design"]["intermediate_node_embeddings"] = str(  # type: ignore[index]
        node_embedding_path
    )

    for mode in args.modes:
        if mode == "semantic_only":
            hp_profiles = semantic_only(full_hp)
            mp_profiles = semantic_only(full_mp)
        else:
            hp_profiles = build_profile_tensor(
                hp_edges, cl_vectors, go_vectors, max_prototypes=4
            )
            mp_profiles = build_profile_tensor(
                mp_edges, cl_vectors, go_vectors, max_prototypes=4
            )
        result = run_model(
            mode,
            hp_profiles,
            mp_profiles,
            expert,
            parents,
            hpo_ontology,
            config,
        )
        result["delta_full_minus_ablation"] = metric_delta(
            formal, result["evaluation"]
        )
        report["ablations"][mode] = result  # type: ignore[index]
        print(f"Completed {mode} in {result['runtime_seconds']:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
