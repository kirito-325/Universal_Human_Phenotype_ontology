#!/usr/bin/env python3
"""Run the standardized-name and marker-statistics downstream pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "database" / "cell_name_catalog_matches.tsv"
DEFAULT_CANDIDATES = ROOT / "name_mapping" / "sapbert_top10.tsv"

STAGES = (
    "filter-statistics",
    "clean-genes",
    "build-names",
    "filter-names",
    "build-candidates",
    "match-names",
    "marker-scores",
    "marker-review",
    "significance",
    "significance-review",
    "coverage",
    "evidence-explorer",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--through-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--candidate-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--min-logfc",
        type=float,
        default=0.25,
        help="Strict Statistics policy: require avg_logFC above this value. Default: 0.25.",
    )
    parser.add_argument(
        "--min-pct1",
        type=float,
        default=0.1,
        help="When pct.1/pct.2 exist, require pct.1 at least this fraction. Default: 0.1.",
    )
    parser.add_argument(
        "--allow-missing-significance",
        action="store_true",
        help="Keep positive rows without p/FDR values instead of using the strict default.",
    )
    parser.add_argument("--sapbert-device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--sapbert-batch-size", type=int, default=32)
    parser.add_argument("--semantic-batch-size", type=int, default=20)
    parser.add_argument("--semantic-workers", type=int, default=1)
    parser.add_argument("--semantic-timeout", type=int, default=180)
    parser.add_argument("--semantic-retries", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument(
        "--no-reuse-mapping",
        action="store_true",
        help="Do not reuse final match/no_match decisions from an existing mapping.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def stage_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    python = sys.executable
    mapping = args.mapping.resolve()
    candidates = args.candidate_file.resolve()
    commands = {
        "filter-statistics": [
            python,
            "-B",
            "scripts/filter_marker_statistics.py",
            "--min-logfc",
            str(args.min_logfc),
            "--min-pct1",
            str(args.min_pct1),
        ],
        "clean-genes": [python, "-B", "marker_statistics/clean_marker_genes.py"],
        "build-names": [python, "-B", "scripts/build_agent_standardized_cell_name_table.py"],
        "filter-names": [python, "-B", "scripts/filter_agent_standardized_cell_names.py"],
        "build-candidates": [
            python,
            "-B",
            "scripts/build_sapbert_name_mapping.py",
            "--device",
            args.sapbert_device,
            "--batch-size",
            str(args.sapbert_batch_size),
        ],
        "match-names": [
            python,
            "-B",
            "scripts/match_cell_names_to_catalog.py",
            "--candidate-file",
            str(candidates),
            "--candidates",
            "10",
            "--output",
            str(mapping),
            "--batch-size",
            str(args.semantic_batch_size),
            "--workers",
            str(args.semantic_workers),
            "--timeout",
            str(args.semantic_timeout),
            "--retries",
            str(args.semantic_retries),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--max-consecutive-errors",
            str(args.max_consecutive_errors),
        ],
        "marker-scores": [
            python,
            "-B",
            "marker_statistics/build_marker_scores.py",
            "--mapping",
            str(mapping),
            "--require-adjudicated-mapping",
        ],
        "marker-review": [
            python,
            "-B",
            "marker_statistics/build_marker_review.py",
            "--mapping",
            str(mapping),
        ],
        "significance": [python, "-B", "marker_statistics/build_significant_markers.py"],
        "significance-review": [
            python,
            "-B",
            "marker_statistics/build_significant_marker_review.py",
        ],
        "coverage": [python, "-B", "marker_statistics/build_coverage_audit.py"],
        "evidence-explorer": [
            python,
            "-B",
            "marker_statistics/build_marker_evidence_explorer.py",
        ],
    }
    if not args.allow_missing_significance:
        commands["filter-statistics"].append("--require-significance")
    if not args.no_reuse_mapping and mapping.is_file():
        commands["match-names"].extend(["--reuse-mapping", str(mapping)])
    return commands


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if (
        args.sapbert_batch_size < 1
        or args.semantic_batch_size < 1
        or args.semantic_workers < 1
        or args.semantic_timeout < 1
        or args.semantic_retries < 0
    ):
        raise SystemExit("batch sizes must be positive")
    if args.checkpoint_every < 0 or args.max_consecutive_errors < 1:
        raise SystemExit("invalid checkpoint or consecutive-error setting")
    if args.min_pct1 < 0 or args.min_pct1 > 1:
        raise SystemExit("--min-pct1 must be in [0, 1]")
    start = STAGES.index(args.from_stage)
    stop = STAGES.index(args.through_stage)
    if start > stop:
        raise SystemExit("--from-stage must not come after --through-stage")

    commands = stage_commands(args)
    for stage in STAGES[start : stop + 1]:
        command = commands[stage]
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            environment = os.environ.copy()
            if stage == "build-candidates":
                environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
