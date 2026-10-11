#!/usr/bin/env python3
"""Run extracted-table processing through the marker evidence website build."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("extract-tables", "agent-standardize", "downstream")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default="downstream",
        help=(
            "Default: downstream, which only consumes existing LLM-standardized papers. "
            "Choose an earlier stage explicitly to expand the cohort."
        ),
    )
    parser.add_argument("--through-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--semantic-workers", type=int, default=1)
    parser.add_argument("--sapbert-device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def stage_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    python = sys.executable
    return {
        "extract-tables": [
            python,
            "-B",
            "scripts/run_literature_pipeline.py",
            "extract-tables",
            "--limit",
            "0",
        ],
        "agent-standardize": [
            python,
            "-B",
            "scripts/run_literature_pipeline.py",
            "agent-standardize-marker-tables",
            "--require-complete-inputs",
            "--stop-on-api-exhaustion",
            "--workers",
            str(args.agent_workers),
        ],
        "downstream": [
            python,
            "-B",
            "scripts/run_downstream_analysis.py",
            "--semantic-workers",
            str(args.semantic_workers),
            "--sapbert-device",
            args.sapbert_device,
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.agent_workers < 1 or args.semantic_workers < 1:
        raise SystemExit("worker counts must be positive")
    start = STAGES.index(args.from_stage)
    stop = STAGES.index(args.through_stage)
    if start > stop:
        raise SystemExit("--from-stage must not come after --through-stage")

    environment = os.environ.copy()
    for stage in STAGES[start : stop + 1]:
        command = stage_commands(args)[stage]
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
