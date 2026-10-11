#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import InputPaths  # noqa: E402
from mpo_hpo_mapping.pipeline import RunConfig, run_pipeline  # noqa: E402


def parser() -> argparse.ArgumentParser:
    default_output = MODULE_DIR / "results/late_fusion_v2"
    command = argparse.ArgumentParser(
        description="Train late-fusion semantic/structure MPO-to-HPO retrieval v2."
    )
    command.add_argument("--output-dir", type=Path, default=default_output)
    command.add_argument("--hpo-specific", type=Path)
    command.add_argument("--hpo-general", type=Path)
    command.add_argument("--mpo-specific", type=Path)
    command.add_argument("--mpo-general", type=Path)
    command.add_argument("--expert-mappings", type=Path)
    command.add_argument("--hpo-edges", type=Path)
    command.add_argument("--hpo-obo", type=Path)
    command.add_argument("--mpo-obo", type=Path)
    command.add_argument("--cl-catalog", type=Path)
    command.add_argument("--go-obo", type=Path)
    command.add_argument("--general-top-k", type=int, default=8)
    command.add_argument("--general-weight", type=float, default=0.15)
    command.add_argument("--hierarchy-dimension", type=int, default=16)
    command.add_argument("--hierarchy-epochs", type=int, default=30)
    command.add_argument(
        "--hierarchy-mode",
        choices=("poincare", "constant", "permuted"),
        default="poincare",
    )
    command.add_argument("--hierarchy-reference", default="")
    command.add_argument("--hierarchy-permutation-seed", type=int, default=0)
    command.add_argument(
        "--hpo-link-randomization",
        choices=("none", "degree_preserving"),
        default="none",
    )
    command.add_argument("--randomization-swaps-per-edge", type=int, default=10)
    command.add_argument("--max-prototypes", type=int, default=4)
    command.add_argument(
        "--semantic-mode", choices=("auto", "sapbert", "tfidf", "none"), default="auto"
    )
    command.add_argument("--semantic-dimension", type=int, default=64)
    command.add_argument("--semantic-batch-size", type=int, default=64)
    command.add_argument("--semantic-reference", default="")
    command.add_argument("--semantic-weight", type=float, default=1.0)
    command.add_argument("--semantic-seed", type=int, default=0)
    command.add_argument(
        "--disable-term-aliases",
        action="store_true",
        help="Remove normalized term aliases from calibration and ranking.",
    )
    command.add_argument("--hidden-dimension", type=int, default=64)
    command.add_argument("--output-dimension", type=int, default=32)
    command.add_argument("--training-epochs", type=int, default=80)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--structure-aux-weight", type=float, default=0.35)
    command.add_argument(
        "--semantic-dropout-probability", type=float, default=0.25
    )
    command.add_argument(
        "--disable-source-balancing",
        action="store_true",
        help="Use all synthetic parent examples each epoch instead of source balancing.",
    )
    command.add_argument(
        "--fusion-mode",
        choices=("late_fusion", "semantic_only", "structure_only"),
        default="late_fusion",
    )
    command.add_argument("--min-mapping-confidence", type=float, default=0.9)
    command.add_argument("--synthetic-limit", type=int, default=0)
    command.add_argument("--ivf-clusters", type=int, default=96)
    command.add_argument("--nprobe", type=int, default=8)
    command.add_argument("--retrieve-k", type=int, default=80)
    command.add_argument("--output-top-k", type=int, default=20)
    command.add_argument("--max-output-mp", type=int, default=0)
    command.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Train and evaluate without writing production MPO-to-HPO mappings.",
    )
    command.add_argument("--seed", type=int, default=20260726)
    command.add_argument("--data-split-seed", type=int, default=0)
    command.add_argument(
        "--smoke",
        action="store_true",
        help="Use short training and write mappings for 100 MPO terms.",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    defaults = InputPaths()
    paths = InputPaths(
        hpo_specific=args.hpo_specific or defaults.hpo_specific,
        hpo_general=args.hpo_general or defaults.hpo_general,
        mpo_specific=args.mpo_specific or defaults.mpo_specific,
        mpo_general=args.mpo_general or defaults.mpo_general,
        expert_mappings=args.expert_mappings or defaults.expert_mappings,
        hpo_edges=args.hpo_edges or defaults.hpo_edges,
        hpo_obo=args.hpo_obo or defaults.hpo_obo,
        mpo_obo=args.mpo_obo or defaults.mpo_obo,
        cl_catalog=args.cl_catalog or defaults.cl_catalog,
        go_obo=args.go_obo or defaults.go_obo,
    )
    config = RunConfig(
        output_dir=args.output_dir,
        general_top_k=2 if args.smoke else args.general_top_k,
        general_weight=args.general_weight,
        hierarchy_dimension=8 if args.smoke else args.hierarchy_dimension,
        hierarchy_epochs=5 if args.smoke else args.hierarchy_epochs,
        hierarchy_mode=args.hierarchy_mode,
        hierarchy_reference=args.hierarchy_reference,
        hierarchy_permutation_seed=args.hierarchy_permutation_seed,
        hpo_link_randomization=args.hpo_link_randomization,
        randomization_swaps_per_edge=args.randomization_swaps_per_edge,
        max_prototypes=2 if args.smoke else args.max_prototypes,
        semantic_mode="tfidf" if args.smoke else args.semantic_mode,
        semantic_dimension=32 if args.smoke else args.semantic_dimension,
        semantic_batch_size=args.semantic_batch_size,
        semantic_reference=args.semantic_reference,
        semantic_weight=args.semantic_weight,
        semantic_seed=args.semantic_seed,
        use_term_aliases=not args.disable_term_aliases,
        hidden_dimension=32 if args.smoke else args.hidden_dimension,
        output_dimension=16 if args.smoke else args.output_dimension,
        training_epochs=12 if args.smoke else args.training_epochs,
        batch_size=args.batch_size,
        structure_aux_weight=args.structure_aux_weight,
        semantic_dropout_probability=args.semantic_dropout_probability,
        balance_parent_sources=not args.disable_source_balancing,
        fusion_mode=args.fusion_mode,
        min_mapping_confidence=args.min_mapping_confidence,
        synthetic_limit=500 if args.smoke else args.synthetic_limit,
        ivf_clusters=32 if args.smoke else args.ivf_clusters,
        nprobe=4 if args.smoke else args.nprobe,
        retrieve_k=30 if args.smoke else args.retrieve_k,
        output_top_k=5 if args.smoke else args.output_top_k,
        max_output_mp=100 if args.smoke else args.max_output_mp,
        evaluation_only=args.evaluation_only,
        seed=args.seed,
        data_split_seed=args.data_split_seed,
    )
    manifest = run_pipeline(paths, config)
    print(f"Wrote {manifest['output']['mapping_rows']} mappings to {config.output_dir}")


if __name__ == "__main__":
    main()
