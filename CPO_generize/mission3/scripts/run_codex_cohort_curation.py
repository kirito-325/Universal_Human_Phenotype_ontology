#!/usr/bin/env python3
"""Run auditable, schema-constrained Codex reviews for a formal disease cohort."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MISSION3 = ROOT / "mission3"
DEFAULT_BATCH = MISSION3 / "data" / "curation" / "formal_cohort_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_ranks(value: str) -> set[int] | None:
    if not value.strip():
        return None
    ranks: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            ranks.update(range(int(start_text), int(end_text) + 1))
        elif token:
            ranks.add(int(token))
    return ranks


def run_one(
    entry: dict[str, Any],
    *,
    batch_dir: Path,
    prompt_template: str,
    schema_path: Path,
    codex_path: str,
    timeout_seconds: int,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    rank = int(entry["selection_rank"])
    disease_id = str(entry["disease_id"])
    slug = f"{rank:02d}_{disease_id.replace(':', '_')}"
    output_dir = batch_dir / "codex_outputs" / slug
    prompt_dir = batch_dir / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.json"
    status_path = output_dir / "status.json"
    if not force and status_path.is_file():
        previous = read_json(status_path)
        if previous.get("status") == "completed" and final_path.is_file():
            return previous

    packet_path = ROOT / entry["packet_path"]
    rendered_prompt = prompt_template.replace(
        "{{PACKET_PATH}}", str(packet_path.relative_to(ROOT))
    )
    prompt_path = prompt_dir / f"{slug}.txt"
    prompt_path.write_text(rendered_prompt, encoding="utf-8")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    return_code: int | None = None
    error = ""
    attempts_root = output_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    previous_attempts = [
        int(path.name)
        for path in attempts_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_attempt = max(previous_attempts, default=0) + 1
    attempts: list[dict[str, Any]] = []
    output_valid_json = False
    command: list[str] = []
    for retry_index in range(retries + 1):
        attempt_number = next_attempt + retry_index
        attempt_dir = attempts_root / f"{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_final_path = attempt_dir / "final.json"
        events_path = attempt_dir / "events.jsonl"
        stderr_path = attempt_dir / "stderr.log"
        command = [
            codex_path,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(attempt_final_path),
            "--cd",
            str(ROOT),
            "-",
        ]
        attempt_started = time.monotonic()
        attempt_error = ""
        with events_path.open("w", encoding="utf-8", newline="\n") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as stderr_handle:
            try:
                result = subprocess.run(
                    command,
                    input=rendered_prompt,
                    text=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=ROOT,
                    timeout=timeout_seconds,
                    check=False,
                )
                return_code = result.returncode
            except subprocess.TimeoutExpired:
                return_code = None
                attempt_error = f"Codex review exceeded {timeout_seconds} seconds"
            except OSError as exc:
                return_code = None
                attempt_error = str(exc)
        output_valid_json = False
        if attempt_final_path.is_file():
            try:
                output_valid_json = isinstance(
                    json.loads(attempt_final_path.read_text(encoding="utf-8")), dict
                )
            except json.JSONDecodeError as exc:
                attempt_error = f"Invalid final JSON: {exc}"
        event_text = events_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        transient = any(
            marker in event_text or marker in stderr_text
            for marker in (
                "429 Too Many Requests",
                "500 Internal Server Error",
                "502 Bad Gateway",
                "503 Service Unavailable",
                "504 Gateway Timeout",
                "Reconnecting...",
                "connection reset",
            )
        )
        if return_code != 0 and not attempt_error:
            attempt_error = (
                f"Codex returned {return_code}; final JSON present={attempt_final_path.is_file()}"
            )
        attempts.append(
            {
                "attempt": attempt_number,
                "return_code": return_code,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "output_valid_json": output_valid_json,
                "transient_failure": transient,
                "error": attempt_error,
                "events_path": str(events_path.relative_to(ROOT)),
                "stderr_path": str(stderr_path.relative_to(ROOT)),
                "final_path": str(attempt_final_path.relative_to(ROOT)),
            }
        )
        if return_code == 0 and output_valid_json:
            shutil.copyfile(attempt_final_path, final_path)
            error = ""
            break
        error = attempt_error
        if not transient or retry_index >= retries:
            break
        time.sleep(min(20, 5 * (retry_index + 1)))

    status = "completed" if return_code == 0 and output_valid_json else "failed"

    result_status = {
        "selection_rank": rank,
        "disease_id": disease_id,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "return_code": return_code,
        "output_valid_json": output_valid_json,
        "error": error,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "final_path": str(final_path.relative_to(ROOT)),
        "command": command[:-1] + ["<prompt-via-stdin>"],
        "attempts": attempts,
    }
    write_json(status_path, result_status)
    return result_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Additional attempts after a transient 429/5xx/connection failure.",
    )
    parser.add_argument(
        "--ranks",
        default="",
        help="Comma-separated ranks or ranges, for example 1-3,8. Default: all.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    batch_dir = args.batch_dir.resolve()
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = read_json(manifest_path)
    schema_path = (
        MISSION3 / "curation" / "formal_cohort_v1" / "disease_review.schema.json"
    ).resolve()
    prompt_path = (
        MISSION3 / "curation" / "formal_cohort_v1" / "disease_review_prompt.md"
    ).resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    codex_path = shutil.which("codex")
    if not codex_path:
        raise RuntimeError("codex executable was not found on PATH")
    selected_ranks = parse_ranks(args.ranks)
    entries = [
        entry
        for entry in manifest["packets"]
        if selected_ranks is None or int(entry["selection_rank"]) in selected_ranks
    ]
    if not entries:
        raise ValueError("No packet matched --ranks")

    version = subprocess.run(
        [codex_path, "--version"],
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    ).stdout.strip()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                entry,
                batch_dir=batch_dir,
                prompt_template=prompt_template,
                schema_path=schema_path,
                codex_path=codex_path,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                force=args.force,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "rank": result["selection_rank"],
                        "disease_id": result["disease_id"],
                        "status": result["status"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "error": result["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    all_statuses = []
    for entry in manifest["packets"]:
        slug = f"{int(entry['selection_rank']):02d}_{str(entry['disease_id']).replace(':', '_')}"
        path = batch_dir / "codex_outputs" / slug / "status.json"
        if path.is_file():
            all_statuses.append(read_json(path))
    completed = sum(row.get("status") == "completed" for row in all_statuses)
    failed = sum(row.get("status") == "failed" for row in all_statuses)
    run_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "codex_version": version,
        "requested_this_run": len(entries),
        "packet_count": len(manifest["packets"]),
        "completed_count": completed,
        "failed_count": failed,
        "pending_count": len(manifest["packets"]) - completed - failed,
        "statuses": sorted(all_statuses, key=lambda row: int(row["selection_rank"])),
    }
    write_json(batch_dir / "codex_run_summary.json", run_summary)
    manifest["codex_run"] = {
        key: run_summary[key]
        for key in (
            "generated_at",
            "codex_version",
            "packet_count",
            "completed_count",
            "failed_count",
            "pending_count",
        )
    }
    manifest["status"] = (
        "codex_completed"
        if completed == len(manifest["packets"])
        else "codex_partial"
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["codex_run"], ensure_ascii=False))


if __name__ == "__main__":
    main()
