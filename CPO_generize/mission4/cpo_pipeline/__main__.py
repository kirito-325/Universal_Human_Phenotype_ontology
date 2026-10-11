from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from cpo_mapping.io import write_json

from .audit import run_audit
from .config import load_config, output_root, validate_config
from .disease_model import run_disease_model
from .evaluate import run_evaluate
from .explain import (
    initialize_descendant_support,
    run_auxiliary_explanations,
    run_explanations,
    run_gene_evidence_chains,
    run_term_explanations,
)
from .gene_rank import finalize_candidate_tables, run_gene_rank
from .mapping import run_mapping
from .prepare import run_prepare
from .query import explain_gene, explain_pair, export_term_scores, query_disease, query_term
from .term_matrix import run_term_matrix
from .validate import run_validate


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _mapping_resume(root: Path) -> dict[str, Any] | None:
    paths = {
        "hpo_to_mpo_rows": root / "mapping/hpo_to_mpo.parquet",
        "mpo_to_hpo_rows": root / "mapping/mpo_to_hpo.parquet",
        "auxiliary_rows": root / "mapping/auxiliary_topk.parquet",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    result = {key: _parquet_rows(path) for key, path in paths.items()}
    result["status"] = "complete"
    result["resumed_without_recompute"] = True
    write_json(root / "mapping/mapping_manifest.json", result)
    return result


def _explanation_resume(root: Path) -> dict[str, Any] | None:
    paths = {
        "term_explanation_rows": root / "explanations/term_matches.parquet",
        "auxiliary_explanation_rows": root / "explanations/auxiliary_matches.parquet",
        "gene_evidence_chain_rows": root / "explanations/gene_evidence_chains.parquet",
        "descendant_support_rows": root / "explanations/descendant_support.parquet",
    }
    manifests = (
        root / "explanations/term_explanation_manifest.json",
        root / "explanations/auxiliary_explanation_manifest.json",
        root / "explanations/gene_evidence_manifest.json",
        root / "explanations/descendant_support_manifest.json",
    )
    if not all(path.is_file() for path in paths.values()) or not all(
        path.is_file() for path in manifests
    ):
        return None
    result = {key: _parquet_rows(path) for key, path in paths.items()}
    result["status"] = "complete"
    result["resumed_without_recompute"] = True
    write_json(root / "explanations/explanation_manifest.json", result)
    return result


def run_all(config: dict[str, Any], resume: bool) -> dict[str, Any]:
    from .report import run_report

    root = output_root(config)
    results: dict[str, Any] = {}
    profile_manifest = root / "profiles/profile_manifest.json"
    if resume and profile_manifest.is_file():
        results["prepare"] = {"status": "complete", "resumed_without_recompute": True}
    else:
        results["prepare"] = run_prepare(config)
    results["term_matrix"] = run_term_matrix(config, resume=resume)
    mapping = _mapping_resume(root) if resume else None
    results["mapping"] = mapping or run_mapping(config)
    results["disease_model"] = run_disease_model(config, resume=resume)
    gene_manifest = root / "gene_prediction/gene_ranking_manifest.json"
    if resume and gene_manifest.is_file():
        finalization = (
            {"status": "complete", "resumed_without_recompute": True}
            if (root / "gene_prediction/candidate_finalization_manifest.json").is_file()
            else finalize_candidate_tables(config)
        )
        results["gene_rank"] = {
            "status": "complete",
            "resumed_without_recompute": True,
            "candidate_finalization": finalization,
        }
    else:
        results["gene_rank"] = run_gene_rank(config)
    explanations = _explanation_resume(root) if resume else None
    if explanations is not None:
        results["explain"] = explanations
    elif resume:
        explanation_results: dict[str, Any] = {}
        if (root / "explanations/term_explanation_manifest.json").is_file():
            explanation_results["term_explanation_rows"] = _parquet_rows(
                root / "explanations/term_matches.parquet"
            )
        else:
            explanation_results.update(run_term_explanations(config))
        if (root / "explanations/auxiliary_explanation_manifest.json").is_file():
            explanation_results["auxiliary_explanation_rows"] = _parquet_rows(
                root / "explanations/auxiliary_matches.parquet"
            )
        else:
            explanation_results.update(run_auxiliary_explanations(config))
        if (root / "explanations/gene_evidence_manifest.json").is_file():
            explanation_results["gene_evidence_chain_rows"] = _parquet_rows(
                root / "explanations/gene_evidence_chains.parquet"
            )
        else:
            explanation_results.update(run_gene_evidence_chains(config))
        if not (root / "explanations/descendant_support_manifest.json").is_file():
            initialize_descendant_support(config)
        explanation_results["descendant_support_rows"] = _parquet_rows(
            root / "explanations/descendant_support.parquet"
        )
        explanation_results["status"] = "complete"
        write_json(root / "explanations/explanation_manifest.json", explanation_results)
        results["explain"] = explanation_results
    else:
        results["explain"] = run_explanations(config)
    evaluation_manifest = root / "evaluation/evaluation_manifest.json"
    if resume and evaluation_manifest.is_file():
        results["evaluate"] = {"status": "complete", "resumed_without_recompute": True}
    else:
        results["evaluate"] = run_evaluate(config)
    results["validate"] = run_validate(config)
    results["report"] = run_report(config)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cpo_pipeline",
        description="All-term HPO-MPO CPO similarity and disease-gene prioritization v2",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stages = (
        "audit", "prepare", "term-matrix", "mapping", "disease-model",
        "gene-rank", "explain", "evaluate", "validate", "report", "run-all",
    )
    for command in stages:
        stage = subparsers.add_parser(command)
        stage.add_argument("--config", default="config/cpo_v2.yaml")
        if command in {"term-matrix", "disease-model", "run-all"}:
            stage.add_argument("--resume", action="store_true")

    query_term_parser = subparsers.add_parser("query-term")
    query_term_parser.add_argument("--config", default="config/cpo_v2.yaml")
    query_term_parser.add_argument("--side", choices=["HPO", "MPO"], required=True)
    query_term_parser.add_argument("--term-id", required=True)
    query_term_parser.add_argument("--limit", type=int, default=20)

    pair_parser = subparsers.add_parser("explain-pair")
    pair_parser.add_argument("--config", default="config/cpo_v2.yaml")
    pair_parser.add_argument("--hpo-id", required=True)
    pair_parser.add_argument("--mpo-id", required=True)

    disease_parser = subparsers.add_parser("query-disease")
    disease_parser.add_argument("--config", default="config/cpo_v2.yaml")
    disease_parser.add_argument("--disease-id", required=True)
    disease_parser.add_argument("--view", choices=["one_to_one", "expanded"], default="one_to_one")
    disease_parser.add_argument("--limit", type=int, default=20)

    gene_parser = subparsers.add_parser("explain-gene")
    gene_parser.add_argument("--config", default="config/cpo_v2.yaml")
    gene_parser.add_argument("--disease-id", required=True)
    gene_parser.add_argument("--gene-id", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--config", default="config/cpo_v2.yaml")
    export_parser.add_argument("--side", choices=["HPO", "MPO"], required=True)
    export_parser.add_argument("--term-id", required=True)
    export_parser.add_argument("--dataset", default="final/shape")
    export_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    validate_config(config)
    command = args.command
    if command == "audit":
        result = run_audit(config)
    elif command == "prepare":
        result = run_prepare(config)
    elif command == "term-matrix":
        result = run_term_matrix(config, resume=args.resume)
    elif command == "mapping":
        result = run_mapping(config)
    elif command == "disease-model":
        result = run_disease_model(config, resume=args.resume)
    elif command == "gene-rank":
        result = run_gene_rank(config)
    elif command == "explain":
        result = run_explanations(config)
    elif command == "evaluate":
        result = run_evaluate(config)
    elif command == "validate":
        result = run_validate(config)
    elif command == "report":
        from .report import run_report

        result = run_report(config)
    elif command == "run-all":
        result = run_all(config, resume=args.resume)
    elif command == "query-term":
        result = query_term(config, args.side, args.term_id, args.limit)
    elif command == "explain-pair":
        result = explain_pair(config, args.hpo_id, args.mpo_id)
    elif command == "query-disease":
        result = query_disease(config, args.disease_id, args.view, args.limit)
    elif command == "explain-gene":
        result = explain_gene(config, args.disease_id, args.gene_id)
    elif command == "export":
        result = export_term_scores(config, args.side, args.term_id, args.dataset, args.output)
    else:
        raise AssertionError(command)
    _print(result)


if __name__ == "__main__":
    main()
