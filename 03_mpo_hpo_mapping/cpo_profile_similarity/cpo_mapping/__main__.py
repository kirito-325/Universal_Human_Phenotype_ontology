from __future__ import annotations

from uhpo_common.paths import PROFILE_SIMILARITY

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import run_audit
from .all_abnormal import (
    run_all_abnormal,
    run_all_abnormal_candidates,
    run_all_abnormal_compute,
    run_all_abnormal_explanations,
    run_all_abnormal_report,
)
from .branch import run_branch
from .candidates import run_candidates
from .config import config_digest, load_config, output_root, validate_config
from .evaluate import run_evaluate
from .explain import run_explanations
from .interface import explain_candidate, export_matrix, query_candidates
from .io import software_versions, write_json
from .matrix import run_compute, run_smoke_test
from .prepare import run_prepare
from .validate import run_validation


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _manifest(config: dict[str, Any], status: str, stages: dict[str, object], error: str = "") -> None:
    write_json(
        output_root(config) / "run_manifest.json",
        {
            "schema_version": 1,
            "run_name": config["run_name"],
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_sha256": config_digest(config),
            "software_versions": software_versions(),
            "stages": stages,
            "error": error,
        },
    )


def run_all(config: dict[str, Any], resume: bool) -> dict[str, object]:
    from .report import run_report

    stages: dict[str, object] = {}
    _manifest(config, "running", stages)
    try:
        stages["prepare"] = run_prepare(config)
        _manifest(config, "running", stages)
        stages["smoke_test"] = run_smoke_test(config)
        _manifest(config, "running", stages)
        stages["compute"] = run_compute(config, resume=resume)
        _manifest(config, "running", stages)
        stages["candidates"] = run_candidates(config)
        _manifest(config, "running", stages)
        stages["branch"] = run_branch(config)
        _manifest(config, "running", stages)
        stages["explanations"] = run_explanations(config)
        _manifest(config, "running", stages)
        stages["evaluate"] = run_evaluate(config)
        _manifest(config, "running", stages)
        stages["validate"] = run_validation(config)
        _manifest(config, "running", stages)
        stages["report"] = run_report(config)
        _manifest(config, "complete", stages)
        return {"status": "complete", "stages": stages}
    except Exception as exc:
        _manifest(config, "failed", stages, f"{type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="CPO-based HPO-MPO similarity pipeline")
    value.add_argument("--config", default=PROFILE_SIMILARITY / "config/cpo_mapping_v1.yaml")
    subparsers = value.add_subparsers(dest="command", required=True)
    for command in (
        "audit",
        "prepare",
        "smoke-test",
        "candidates",
        "branch",
        "build-explanations",
        "evaluate",
        "report",
        "validate",
        "all-abnormal-candidates",
        "all-abnormal-explanations",
        "all-abnormal-report",
    ):
        subparsers.add_parser(command)
    compute = subparsers.add_parser("compute")
    compute.add_argument("--resume", action="store_true")
    all_parser = subparsers.add_parser("run-all")
    all_parser.add_argument("--resume", action="store_true")
    all_abnormal_compute = subparsers.add_parser("all-abnormal-compute")
    all_abnormal_compute.add_argument("--resume", action="store_true")
    all_abnormal_run = subparsers.add_parser("run-all-abnormal")
    all_abnormal_run.add_argument("--resume", action="store_true")
    query = subparsers.add_parser("query")
    query.add_argument("--side", choices=["HPO", "MPO"], required=True)
    query.add_argument("--term-id", required=True)
    query.add_argument("--channel", choices=["DD", "DM", "MD", "MM"], required=True)
    query.add_argument(
        "--view",
        choices=["shape_topk", "direction_supported_topk", "branch", "auxiliary", "CL", "GO"],
        required=True,
    )
    query.add_argument("--dimension", choices=["CL", "GO"])
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--scope", choices=["self", "all-abnormal"], default="self")
    explain = subparsers.add_parser("explain")
    explain.add_argument("--hpo-id", required=True)
    explain.add_argument("--mpo-id", required=True)
    explain.add_argument("--channel", choices=["DD", "DM", "MD", "MM"], required=True)
    explain.add_argument("--view", choices=["main", "branch", "auxiliary", "CL", "GO"], required=True)
    explain.add_argument("--score-type", choices=["shape", "support", "conflict"], default="shape")
    explain.add_argument("--dimension", choices=["CL", "GO"])
    explain.add_argument("--scope", choices=["self", "all-abnormal"], default="self")
    export = subparsers.add_parser("export")
    export.add_argument("--side", choices=["HPO", "MPO"], default="HPO")
    export.add_argument("--term-id")
    export.add_argument("--pairs-file")
    export.add_argument("--channel", choices=["DD", "DM", "MD", "MM"], required=True)
    export.add_argument("--score-type", choices=["shape", "support", "conflict"], required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--scope", choices=["self", "all-abnormal"], default="self")
    for subparser in subparsers.choices.values():
        subparser.add_argument("--config", default=argparse.SUPPRESS)
    return value


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    validate_config(config)
    command = args.command
    if command == "audit":
        result = run_audit(config)
    elif command == "prepare":
        result = run_prepare(config)
    elif command == "smoke-test":
        result = run_smoke_test(config)
    elif command == "compute":
        result = run_compute(config, resume=args.resume)
    elif command == "candidates":
        result = run_candidates(config)
    elif command == "branch":
        result = run_branch(config)
    elif command == "build-explanations":
        result = run_explanations(config)
    elif command == "evaluate":
        result = run_evaluate(config)
    elif command == "report":
        from .report import run_report

        result = run_report(config)
    elif command == "validate":
        result = run_validation(config)
    elif command == "all-abnormal-compute":
        result = run_all_abnormal_compute(config, resume=args.resume)
    elif command == "all-abnormal-candidates":
        result = run_all_abnormal_candidates(config)
    elif command == "all-abnormal-explanations":
        result = run_all_abnormal_explanations(config)
    elif command == "all-abnormal-report":
        result = run_all_abnormal_report(config)
    elif command == "run-all-abnormal":
        result = run_all_abnormal(config, resume=args.resume)
    elif command == "run-all":
        result = run_all(config, resume=args.resume)
    elif command == "query":
        result = query_candidates(
            config,
            args.side,
            args.term_id,
            args.channel,
            args.view,
            args.limit,
            args.dimension,
            args.scope,
        )
    elif command == "explain":
        result = explain_candidate(
            config,
            args.hpo_id,
            args.mpo_id,
            args.channel,
            args.view,
            args.score_type,
            args.dimension,
            args.scope,
        )
    elif command == "export":
        if not args.term_id and not args.pairs_file:
            raise ValueError("export requires --term-id or --pairs-file")
        result = {
            "exported_row_count": export_matrix(
                config,
                args.side,
                args.term_id,
                args.channel,
                args.score_type,
                args.output,
                args.pairs_file,
                args.scope,
            ),
            "output": str(Path(args.output).resolve()),
        }
    else:
        raise AssertionError(command)
    _print(result)


if __name__ == "__main__":
    main()
