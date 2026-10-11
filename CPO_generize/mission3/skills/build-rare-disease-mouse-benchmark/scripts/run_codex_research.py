#!/usr/bin/env python3
"""Run Codex disease research with local schema validation and full audit logs."""

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

from jsonschema import Draft202012Validator

from common import read_json, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_BATCH = DEFAULT_ROOT / "mission3/data/benchmark/benchmark_v1"


def parse_ranks(value: str) -> set[int] | None:
    if not value.strip():
        return None
    result: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        elif token:
            result.add(int(token))
    return result


def transient_failure(events: str, stderr: str) -> bool:
    return any(
        marker in events or marker in stderr
        for marker in (
            "429 Too Many Requests",
            "500 Internal Server Error",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "connection reset",
        )
    )


def parse_json_object(message: str) -> dict[str, Any]:
    """Parse a whole JSON object, allowing only one optional Markdown fence."""
    candidate = message.strip()
    lines = candidate.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        candidate = "\n".join(lines[1:-1]).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Final response must be one JSON object")
    return value


def schema_validation_errors(
    value: dict[str, Any], schema: dict[str, Any]
) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(value),
        key=lambda item: tuple(str(token) for token in item.absolute_path),
    )
    result = []
    for error in errors:
        path = "$"
        for token in error.absolute_path:
            path += f"[{token}]" if isinstance(token, int) else f".{token}"
        result.append(f"{path}: {error.message}")
    return result


def packet_validation_errors(
    value: dict[str, Any], packet: dict[str, Any]
) -> list[str]:
    """Validate fields whose allowed values are fixed by the research packet."""
    errors: list[str] = []
    for field in ("disease_id", "disease_name", "gene_symbol"):
        if field in packet and str(value.get(field, "")) != str(packet[field]):
            errors.append(
                f"$.{field}: must exactly equal packet value {packet[field]!r}"
            )
    if "allowed_hpo_ids" in packet:
        allowed_hpo = set(packet["allowed_hpo_ids"])
        for field in ("core_hpo_ids", "excluded_hpo_ids"):
            outside = sorted(set(value.get(field, [])).difference(allowed_hpo))
            if outside:
                errors.append(f"$.{field}: IDs outside packet: {outside}")
        overlap = sorted(
            set(value.get("core_hpo_ids", [])).intersection(
                value.get("excluded_hpo_ids", [])
            )
        )
        if overlap:
            errors.append(f"$.core_hpo_ids: also excluded: {overlap}")
    if "source_gold_models" in packet:
        expected_models = {
            str(row["model_id"]) for row in packet["source_gold_models"]
        }
        decided_models = {
            str(row.get("model_id", ""))
            for row in value.get("gold_model_decisions", [])
        }
        if decided_models != expected_models:
            errors.append(
                "$.gold_model_decisions: model IDs must exactly cover packet models"
            )
    return errors


def build_codex_command(
    *,
    codex: str,
    root: Path,
    last_message_path: Path,
    model: str,
    ignore_user_config: bool,
) -> list[str]:
    command = [codex, "exec"]
    if ignore_user_config:
        command.append("--ignore-user-config")
    if model:
        command.extend(["--model", model])
    command.extend(
        [
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--output-last-message",
            str(last_message_path),
            "--cd",
            str(root),
            "-",
        ]
    )
    return command


def validation_retry_prompt(errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors[:12])
    return (
        "\n\nThe previous response failed local JSON validation. Return a corrected "
        "complete JSON object and no other text. Fix every listed problem:\n"
        f"{details}\n"
    )


def run_one(
    entry: dict[str, Any],
    *,
    root: Path,
    batch: Path,
    prompt_template: str,
    schema_path: Path,
    schema: dict[str, Any],
    codex: str,
    model: str,
    ignore_user_config: bool,
    timeout: int,
    retries: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    rank = int(entry["selection_rank"])
    disease_id = str(entry["disease_id"])
    slug = f"{rank:02d}_{disease_id.replace(':', '_')}"
    output_dir = batch / "raw_codex_outputs" / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.txt"
    packet_path = root / entry["packet_path"]
    packet = read_json(packet_path)
    rendered_prompt = prompt_template.replace(
        "{{PACKET_PATH}}", str(packet_path.relative_to(root))
    ).replace(
        "{{SCHEMA_PATH}}", str(schema_path.relative_to(root))
    )
    prompt_path.write_text(rendered_prompt, encoding="utf-8")
    final_path = output_dir / "final.json"
    status_path = output_dir / "status.json"
    if not force and status_path.is_file():
        previous = read_json(status_path)
        if previous.get("status") == "completed" and final_path.is_file():
            return previous
    if dry_run:
        status = {
            "selection_rank": rank,
            "disease_id": disease_id,
            "status": "dry_run",
            "packet_path": str(packet_path.relative_to(root)),
            "prompt_path": str(prompt_path.relative_to(root)),
            "final_path": str(final_path.relative_to(root)),
            "attempts": [],
        }
        write_json(status_path, status)
        return status

    attempts_root = output_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    previous_attempts = [
        int(path.name)
        for path in attempts_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_attempt = max(previous_attempts, default=0) + 1
    started_at = datetime.now(timezone.utc)
    overall_started = time.monotonic()
    attempts = []
    final_status = "failed"
    final_error = ""
    retry_context = ""

    for retry_index in range(retries + 1):
        attempt_number = next_attempt + retry_index
        attempt_dir = attempts_root / f"{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_last_message = attempt_dir / "last_message.txt"
        attempt_final = attempt_dir / "final.json"
        events_path = attempt_dir / "events.jsonl"
        stderr_path = attempt_dir / "stderr.log"
        command = build_codex_command(
            codex=codex,
            root=root,
            last_message_path=attempt_last_message,
            model=model,
            ignore_user_config=ignore_user_config,
        )
        attempt_started = time.monotonic()
        error = ""
        return_code: int | None = None
        attempt_prompt = rendered_prompt + retry_context
        with events_path.open("w", encoding="utf-8", newline="\n") as events, stderr_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as stderr:
            try:
                result = subprocess.run(
                    command,
                    input=attempt_prompt,
                    text=True,
                    stdout=events,
                    stderr=stderr,
                    cwd=root,
                    timeout=timeout,
                    check=False,
                )
                return_code = result.returncode
            except subprocess.TimeoutExpired:
                error = f"Codex exceeded {timeout} seconds"
            except OSError as exc:
                error = str(exc)

        output_valid_json = False
        output_valid_schema = False
        validation_errors: list[str] = []
        if attempt_last_message.is_file():
            try:
                review = parse_json_object(
                    attempt_last_message.read_text(encoding="utf-8")
                )
                output_valid_json = True
            except (ValueError, json.JSONDecodeError) as exc:
                validation_errors = [f"Invalid final JSON: {exc}"]
            else:
                validation_errors = schema_validation_errors(review, schema)
                validation_errors.extend(packet_validation_errors(review, packet))
                if not validation_errors:
                    write_json(attempt_final, review)
                    output_valid_schema = True
        else:
            validation_errors = ["Codex did not write a final message"]

        event_text = events_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        transient = transient_failure(event_text, stderr_text)
        if return_code != 0 and not error:
            error = (
                f"Codex returned {return_code}; valid JSON={output_valid_json}; "
                f"valid schema={output_valid_schema}"
            )
        elif return_code == 0 and not output_valid_schema:
            error = "Local output validation failed: " + " | ".join(validation_errors)
        validation_failure = return_code == 0 and not output_valid_schema
        retryable = validation_failure or (return_code != 0 and transient)
        attempts.append(
            {
                "attempt": attempt_number,
                "return_code": return_code,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "output_valid_json": output_valid_json,
                "output_valid_schema": output_valid_schema,
                "validation_errors": validation_errors,
                "transient_failure": transient,
                "retryable_failure": retryable,
                "error": error,
                "events_path": str(events_path.relative_to(root)),
                "stderr_path": str(stderr_path.relative_to(root)),
                "last_message_path": str(attempt_last_message.relative_to(root)),
                "final_path": str(attempt_final.relative_to(root)),
                "command": command[:-1] + ["<prompt-via-stdin>"],
            }
        )
        if return_code == 0 and output_valid_schema:
            shutil.copyfile(attempt_final, final_path)
            final_status = "completed"
            final_error = ""
            break
        final_error = error
        if not retryable or retry_index == retries:
            break
        retry_context = (
            validation_retry_prompt(validation_errors) if validation_failure else ""
        )
        time.sleep(min(20, 5 * (retry_index + 1)))

    status = {
        "selection_rank": rank,
        "disease_id": disease_id,
        "status": final_status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - overall_started, 3),
        "error": final_error,
        "structured_output_mode": "local_jsonschema",
        "schema_path": str(schema_path.relative_to(root)),
        "packet_path": str(packet_path.relative_to(root)),
        "prompt_path": str(prompt_path.relative_to(root)),
        "final_path": str(final_path.relative_to(root)),
        "attempts": attempts,
    }
    write_json(status_path, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ranks", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ignore-user-config", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 0:
        raise ValueError("Require workers >= 1 and retries >= 0")
    root = args.project_root.resolve()
    batch = args.batch_dir.resolve()
    manifest = read_json(batch / "research_manifest.json")
    if not manifest.get("middle_layer_independent"):
        raise ValueError("Refusing a research manifest without independence assertion")
    ranks = parse_ranks(args.ranks)
    entries = [
        entry
        for entry in manifest["packets"]
        if ranks is None or int(entry["selection_rank"]) in ranks
    ]
    if not entries:
        raise ValueError("No research packets matched --ranks")
    codex = shutil.which("codex")
    if not codex and not args.dry_run:
        raise RuntimeError("codex executable was not found")
    codex = codex or "codex"
    prompt_template = (SKILL_DIR / "assets/research_prompt.md").read_text(
        encoding="utf-8"
    )
    schema_path = (SKILL_DIR / "assets/research_review.schema.json").resolve()
    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                entry,
                root=root,
                batch=batch,
                prompt_template=prompt_template,
                schema_path=schema_path,
                schema=schema,
                codex=codex,
                model=args.model,
                ignore_user_config=args.ignore_user_config,
                timeout=args.timeout_seconds,
                retries=args.retries,
                force=args.force,
                dry_run=args.dry_run,
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
                        "error": result.get("error", ""),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(entries),
        "completed_count": sum(row["status"] == "completed" for row in results),
        "failed_count": sum(row["status"] == "failed" for row in results),
        "dry_run_count": sum(row["status"] == "dry_run" for row in results),
        "model_override": args.model,
        "ignore_user_config": args.ignore_user_config,
        "results": sorted(results, key=lambda row: int(row["selection_rank"])),
    }
    write_json(batch / "codex_research_summary.json", summary)
    print(json.dumps({key: summary[key] for key in summary if key.endswith("_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
