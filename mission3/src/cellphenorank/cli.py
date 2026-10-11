from __future__ import annotations

import argparse
import json

from . import __version__
from .config import load_config
from .embedding_open import run_embedding_open_benchmark
from .impc_dr20_embedding import run_impc_dr20_embedding_test
from .case_freeze import freeze_case
from .pipeline import run_pipeline, validate_inputs
from .parameter_free_pipeline import run_parameter_free_pipeline
from .phenodigm_official import run_official_phenodigm_benchmark
from .phenodigm_legacy import run_legacy_phenodigm_benchmark
from .single_cell import validate_single_cell
from .single_cell_discovery.candidate_builder import run_candidate_discovery
from .single_cell_discovery.dataset_catalog import run_dataset_catalog
from .single_cell_discovery.metadata_preflight import run_metadata_preflight
from .single_cell_discovery.case_freeze import run_case_freeze
from .single_cell_discovery.xla_validation import run_xla_validation
from .single_cell_discovery.foxg1_validation import run_foxg1_validation
from .single_cell_discovery.exploration_summary import run_exploration_summary
from .single_cell_discovery.savi_literature_case import run_savi_literature_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellphenorank")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-inputs", help="Validate configured input contracts")
    validate.add_argument("--config", required=True)
    run = subparsers.add_parser("run", help="Run baseline, reranking, benchmark and null model")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    single = subparsers.add_parser("validate-single-cell", help="Validate predicted cell/function paths")
    single.add_argument("--config", required=True)
    single.add_argument("--output-dir", required=True)
    freeze = subparsers.add_parser(
        "freeze-single-cell-case",
        help="Freeze case predictions and preflight gates before expression analysis",
    )
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output-dir", required=True)
    parameter_free = subparsers.add_parser(
        "run-parameter-free",
        help="Run binary CPO enrichment with phenotype-stability constraints",
    )
    parameter_free.add_argument("--config", required=True)
    parameter_free.add_argument("--output-dir", required=True)
    phenodigm = subparsers.add_parser(
        "run-phenodigm-official",
        help="Benchmark live official IMPC PhenoDigm scores on the frozen candidate set",
    )
    phenodigm.add_argument("--config", required=True)
    phenodigm.add_argument("--output-dir", required=True)
    phenodigm.add_argument(
        "--refresh", action="store_true", help="Refresh cached IMPC API responses"
    )
    legacy_phenodigm = subparsers.add_parser(
        "run-phenodigm-legacy",
        help="Run published PhenoDigm scoring with the archived HP-MP matrix",
    )
    legacy_phenodigm.add_argument("--config", required=True)
    legacy_phenodigm.add_argument("--output-dir", required=True)
    current_phenodigm = subparsers.add_parser(
        "run-phenodigm-2602",
        help="Run current Exomiser PhenoDigm scoring with the 2602 HP-MP matrix",
    )
    current_phenodigm.add_argument("--config", required=True)
    current_phenodigm.add_argument("--output-dir", required=True)
    embedding_open = subparsers.add_parser(
        "run-embedding-open",
        help="Run SapBERT HPO-MP scoring on a frozen open mouse-gene universe",
    )
    embedding_open.add_argument("--config", required=True)
    embedding_open.add_argument("--output-dir", required=True)
    impc_dr20_embedding = subparsers.add_parser(
        "run-impc-dr20-embedding",
        help="Test HPO-MP embeddings against official IMPC DR20.1 outcomes",
    )
    impc_dr20_embedding.add_argument("--config", required=True)
    impc_dr20_embedding.add_argument("--output-dir", required=True)
    case_candidates = subparsers.add_parser(
        "build-single-cell-case-candidates",
        help="Build prediction-driven single-cell case candidates and Pareto fronts",
    )
    case_candidates.add_argument("--config", required=True)
    case_candidates.add_argument("--output-dir", required=True)
    datasets = subparsers.add_parser(
        "discover-single-cell-datasets",
        help="Merge audited public-dataset discoveries with case candidates",
    )
    datasets.add_argument("--config", required=True)
    datasets.add_argument("--output-dir", required=True)
    preflight = subparsers.add_parser(
        "preflight-single-cell-cases",
        help="Apply metadata-only dataset gates without reading expression values",
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output-dir", required=True)
    freeze_cases = subparsers.add_parser(
        "freeze-single-cell-cases",
        help="Freeze all metadata-preflight-passing single-cell cases",
    )
    freeze_cases.add_argument("--config", required=True)
    freeze_cases.add_argument("--output-dir", required=True)
    validate_xla = subparsers.add_parser(
        "validate-frozen-xla-case",
        help="Validate the frozen BTK-XLA case with donor-level pseudobulk",
    )
    validate_xla.add_argument("--config", required=True)
    validate_xla.add_argument("--output-dir", required=True)
    validate_foxg1 = subparsers.add_parser(
        "validate-frozen-foxg1-case",
        help="Validate the frozen FOXG1 organoid follow-up case",
    )
    validate_foxg1.add_argument("--config", required=True)
    validate_foxg1.add_argument("--output-dir", required=True)
    summarize_cases = subparsers.add_parser(
        "summarize-single-cell-case-exploration",
        help="Summarize discovery, preflight and frozen case results",
    )
    summarize_cases.add_argument("--config", required=True)
    summarize_cases.add_argument("--output-dir", required=True)
    savi_case = subparsers.add_parser(
        "validate-literature-first-savi-case",
        help="Reproduce a literature-first SAVI cellular phenotype and evaluate rank rescue",
    )
    savi_case.add_argument("--config", required=True)
    savi_case.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-inputs":
        print(json.dumps(validate_inputs(config), indent=2, ensure_ascii=False))
    elif args.command == "run":
        print(json.dumps(run_pipeline(config, args.output_dir), indent=2, ensure_ascii=False))
    elif args.command == "validate-single-cell":
        print(json.dumps(validate_single_cell(config, args.output_dir), indent=2, ensure_ascii=False))
    elif args.command == "freeze-single-cell-case":
        print(json.dumps(freeze_case(config, args.output_dir), indent=2, ensure_ascii=False))
    elif args.command == "run-parameter-free":
        print(json.dumps(run_parameter_free_pipeline(config, args.output_dir), indent=2, ensure_ascii=False))
    elif args.command == "run-phenodigm-official":
        print(
            json.dumps(
                run_official_phenodigm_benchmark(
                    config, args.output_dir, refresh=args.refresh
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command in {"run-phenodigm-legacy", "run-phenodigm-2602"}:
        print(
            json.dumps(
                run_legacy_phenodigm_benchmark(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "run-embedding-open":
        print(
            json.dumps(
                run_embedding_open_benchmark(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "run-impc-dr20-embedding":
        print(
            json.dumps(
                run_impc_dr20_embedding_test(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "build-single-cell-case-candidates":
        print(
            json.dumps(
                run_candidate_discovery(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "discover-single-cell-datasets":
        print(
            json.dumps(
                run_dataset_catalog(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "preflight-single-cell-cases":
        print(
            json.dumps(
                run_metadata_preflight(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "freeze-single-cell-cases":
        print(
            json.dumps(
                run_case_freeze(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "validate-frozen-xla-case":
        print(
            json.dumps(
                run_xla_validation(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "validate-frozen-foxg1-case":
        print(
            json.dumps(
                run_foxg1_validation(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "summarize-single-cell-case-exploration":
        print(
            json.dumps(
                run_exploration_summary(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "validate-literature-first-savi-case":
        print(
            json.dumps(
                run_savi_literature_case(config, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
