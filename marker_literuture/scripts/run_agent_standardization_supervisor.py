#!/usr/bin/env python3
"""Continuously resume agent standardization across ECNU credit windows."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "database"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_API_KEY_ENVS = ["ECNU_API_KEY"] + [
    f"ECNU_API_KEY_SHARD_{index}" for index in range(1, 8)
]

QUOTA_MARKERS = (
    "ecnu http 429",
    "insufficient_quota",
    "quota exceeded",
    "usage limit",
    "credit balance",
    "credits limit",
    "余额不足",
    "额度不足",
    "配额",
    "用量已用完",
)
AUTH_MARKERS = (
    "ecnu http 401",
    "ecnu http 402",
    "ecnu http 403",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_to_iso(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def iso_to_timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dotenv_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() and value.strip():
            names.add(name.strip())
    return names


def configured_key_envs(names: Sequence[str], env_file: Path) -> List[str]:
    file_names = dotenv_names(env_file)
    configured = []
    for name in names:
        if os.environ.get(name, "").strip() or name in file_names:
            configured.append(name)
    return configured


def classify_api_error(message: object) -> str:
    normalized = str(message or "").casefold()
    if any(marker in normalized for marker in AUTH_MARKERS):
        return "auth"
    if any(marker in normalized for marker in QUOTA_MARKERS):
        return "quota"
    return "other"


def select_key(
    key_envs: Sequence[str],
    key_states: Mapping[str, Mapping[str, Any]],
    start_index: int,
    now: float,
) -> Tuple[Optional[str], int, Optional[float]]:
    if not key_envs:
        return None, -1, None
    earliest: Optional[float] = None
    for step in range(len(key_envs)):
        index = (start_index + step) % len(key_envs)
        name = key_envs[index]
        state = key_states.get(name, {})
        if state.get("disabled"):
            continue
        available_after = iso_to_timestamp(state.get("available_after"))
        if available_after <= now:
            return name, index, None
        earliest = available_after if earliest is None else min(earliest, available_after)
    return None, -1, earliest


def successful_papers(summary: Mapping[str, Any]) -> int:
    return sum(
        int(summary.get(field, 0) or 0)
        for field in ("complete", "no_tables", "no_candidate_tables", "dry_run")
    )


def build_pipeline_command(
    args: argparse.Namespace,
    api_key_env: str,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-B",
        str(ROOT / "scripts" / "run_literature_pipeline.py"),
        "--database",
        str(args.database),
        "agent-standardize-marker-tables",
        "--cohort",
        args.cohort,
        "--exclude-cohort",
        args.exclude_cohort,
        "--require-complete-inputs",
        "--api-key-env",
        api_key_env,
        "--env-file",
        str(args.env_file),
        "--model",
        args.model,
        "--output-subdir",
        args.output_subdir,
        "--summary-name",
        args.summary_name,
        "--harness-mode",
        "standard",
        "--tables-per-request",
        str(args.tables_per_request),
        "--workers",
        str(args.workers),
        "--timeout",
        "600",
        "--retries",
        "3",
        "--llm-attempts",
        "3",
        "--sleep",
        "0.5",
        "--stop-on-api-exhaustion",
        "--no-thinking",
    ]


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_pipeline_pid(
    summary: Mapping[str, Any], supervisor_pid: Optional[int] = None
) -> int:
    if summary.get("run_status") != "running":
        return 0
    try:
        pid = int(summary.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return 0
    if pid == (supervisor_pid if supervisor_pid is not None else os.getpid()):
        return 0
    return pid if process_is_running(pid) else 0


def start_sleep_guard(enabled: bool) -> Optional[subprocess.Popen]:
    executable = Path("/usr/bin/caffeinate")
    if not enabled or sys.platform != "darwin" or not executable.is_file():
        return None
    try:
        return subprocess.Popen(
            [str(executable), "-is", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        print(f"Could not start sleep guard: {error}", flush=True)
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume all-cell agent standardization, rotate configured ECNU keys, "
            "and wait automatically for five-hour credit windows."
        )
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--cohort", default="all_cell_single_cell")
    parser.add_argument("--exclude-cohort", default="immune_single_cell")
    parser.add_argument("--model", default="ecnu-max")
    parser.add_argument("--output-subdir", default="agent_standardized")
    parser.add_argument(
        "--summary-name", default="agent_standardization_allcell_summary.json"
    )
    parser.add_argument(
        "--state-name", default="agent_standardization_supervisor_state.json"
    )
    parser.add_argument(
        "--lock-name", default="agent_standardization_supervisor.lock"
    )
    parser.add_argument(
        "--api-key-envs",
        default=",".join(DEFAULT_API_KEY_ENVS),
        help="Comma-separated API key environment variable names.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tables-per-request", type=int, default=5)
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=18120,
        help="Per-key cooldown after a quota response. Default: 5 hours 2 minutes.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=int,
        default=300,
        help="Delay after a non-quota child-process failure. Default: 5 minutes.",
    )
    parser.add_argument(
        "--existing-poll-seconds",
        type=int,
        default=30,
        help="Polling interval while an existing pipeline is active.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after N child runs; zero means unlimited.",
    )
    parser.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="Do not start the macOS sleep-prevention guard.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.tables_per_request < 1:
        parser.error("--workers and --tables-per-request must be positive")
    if (
        args.cooldown_seconds < 0
        or args.retry_delay_seconds < 0
        or args.existing_poll_seconds < 1
    ):
        parser.error("delays must be non-negative and polling must be positive")
    args.database = args.database.resolve()
    args.env_file = args.env_file.resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    state_path = args.database / args.state_name
    summary_path = args.database / args.summary_name
    lock_path = args.database / args.lock_name
    lock_handle = acquire_lock(lock_path)
    if lock_handle is None:
        print(f"Supervisor already running; lock is held: {lock_path}", flush=True)
        return 2

    sleep_guard = start_sleep_guard(not args.no_caffeinate)

    def finish(exit_code: int) -> int:
        if sleep_guard is not None and sleep_guard.poll() is None:
            sleep_guard.terminate()
            try:
                sleep_guard.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sleep_guard.kill()
                sleep_guard.wait()
        lock_handle.close()
        return exit_code

    requested_keys = [
        value.strip() for value in args.api_key_envs.split(",") if value.strip()
    ]
    key_envs = configured_key_envs(requested_keys, args.env_file)
    if not key_envs:
        print("No configured ECNU API key environment variables were found.", flush=True)
        return finish(2)

    state = read_json(state_path)
    key_states = state.get("key_states", {})
    if not isinstance(key_states, dict):
        key_states = {}
    next_key_index = int(state.get("next_key_index", 0) or 0) % len(key_envs)
    cycle = int(state.get("cycle", 0) or 0)
    stop_event = threading.Event()
    child_holder: Dict[str, Optional[subprocess.Popen]] = {"process": None}

    def save_state(status: str, **updates: Any) -> None:
        payload = {
            "generated_at": utc_now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "cycle": cycle,
            "api_key_envs": key_envs,
            "next_key_index": next_key_index,
            "key_states": key_states,
            **updates,
        }
        write_json_atomic(state_path, payload)

    def stop(signum, _frame) -> None:
        print(f"Received signal {signum}; stopping supervisor.", flush=True)
        stop_event.set()
        child = child_holder["process"]
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    initial_summary = read_json(summary_path)
    if initial_summary.get("run_status") == "complete":
        save_state("complete", summary=str(summary_path))
        print("Agent standardization is already complete.", flush=True)
        return finish(0)

    print(
        f"Supervisor started with {len(key_envs)} configured key(s); "
        f"cooldown={args.cooldown_seconds}s.",
        flush=True,
    )
    monitored_pid = 0
    while not stop_event.is_set():
        summary = read_json(summary_path)
        existing_pid = active_pipeline_pid(summary)
        if existing_pid:
            if existing_pid != monitored_pid:
                print(
                    f"Existing pipeline PID {existing_pid} is active; monitoring it "
                    "before starting the next key.",
                    flush=True,
                )
            monitored_pid = existing_pid
            save_state(
                "monitoring_existing",
                child_pid=existing_pid,
                successful_papers=successful_papers(summary),
                summary=str(summary_path),
            )
            stop_event.wait(args.existing_poll_seconds)
            continue
        if monitored_pid:
            print(
                f"Existing pipeline PID {monitored_pid} ended; resuming key rotation.",
                flush=True,
            )
            monitored_pid = 0

        if args.max_cycles and cycle >= args.max_cycles:
            save_state("max_cycles_reached")
            return finish(0)

        now = time.time()
        api_key_env, key_index, wait_until = select_key(
            key_envs, key_states, next_key_index, now
        )
        if api_key_env is None:
            if wait_until is None:
                save_state("blocked", last_error="All configured keys are disabled.")
                print("All configured ECNU keys are disabled; stopping.", flush=True)
                return finish(2)
            wait_seconds = max(1, int(wait_until - now + 0.999))
            save_state("waiting", next_retry_at=timestamp_to_iso(wait_until))
            print(
                f"All keys are cooling down; next retry in {wait_seconds}s "
                f"at {timestamp_to_iso(wait_until)}.",
                flush=True,
            )
            stop_event.wait(wait_seconds)
            continue

        cycle += 1
        command = build_pipeline_command(args, api_key_env)
        save_state("running", current_key_env=api_key_env, child_pid=None)
        print(
            f"Cycle {cycle}: starting pipeline with {api_key_env}; "
            f"previous_success={successful_papers(read_json(summary_path))}.",
            flush=True,
        )
        environment = dict(os.environ)
        environment["PYTHONPYCACHEPREFIX"] = "/tmp/marker_literuture_pycache"
        child = subprocess.Popen(command, cwd=ROOT, env=environment)
        child_holder["process"] = child
        save_state("running", current_key_env=api_key_env, child_pid=child.pid)
        exit_code = child.wait()
        child_holder["process"] = None
        summary = read_json(summary_path)
        run_status = str(summary.get("run_status", ""))
        fatal_error = str(summary.get("fatal_error", ""))
        success_count = successful_papers(summary)

        if stop_event.is_set():
            save_state(
                "stopped",
                current_key_env=api_key_env,
                last_exit_code=exit_code,
                last_run_status=run_status,
            )
            return finish(130)
        if run_status == "complete":
            save_state(
                "complete",
                current_key_env=api_key_env,
                last_exit_code=exit_code,
                successful_papers=success_count,
                summary=str(summary_path),
            )
            print(
                f"Agent standardization complete: successful_papers={success_count}.",
                flush=True,
            )
            return finish(0)

        error_kind = classify_api_error(fatal_error)
        key_state = key_states.setdefault(api_key_env, {})
        key_state.update(
            {
                "last_used_at": utc_now(),
                "last_exit_code": exit_code,
                "last_run_status": run_status,
                "last_error": fatal_error,
                "successful_papers": success_count,
            }
        )
        if run_status == "stopped_api_exhaustion" and error_kind == "auth":
            key_state["disabled"] = True
            print(f"Disabling {api_key_env} after an authentication error.", flush=True)
            next_key_index = (key_index + 1) % len(key_envs)
            save_state("rotating", last_error=fatal_error)
            continue
        if run_status == "stopped_api_exhaustion":
            available_after = time.time() + args.cooldown_seconds
            key_state["available_after"] = timestamp_to_iso(available_after)
            next_key_index = (key_index + 1) % len(key_envs)
            print(
                f"{api_key_env} reached its credit limit; cooling down until "
                f"{timestamp_to_iso(available_after)} and rotating keys.",
                flush=True,
            )
            save_state("rotating", last_error=fatal_error)
            continue

        retry_at = time.time() + args.retry_delay_seconds
        next_key_index = key_index
        save_state(
            "retry_wait",
            current_key_env=api_key_env,
            last_exit_code=exit_code,
            last_run_status=run_status,
            last_error=fatal_error or f"pipeline exit code {exit_code}",
            next_retry_at=timestamp_to_iso(retry_at),
        )
        print(
            f"Pipeline stopped unexpectedly (exit={exit_code}, status={run_status!r}); "
            f"retrying in {args.retry_delay_seconds}s.",
            flush=True,
        )
        stop_event.wait(args.retry_delay_seconds)

    save_state("stopped")
    return finish(130)


if __name__ == "__main__":
    raise SystemExit(main())
