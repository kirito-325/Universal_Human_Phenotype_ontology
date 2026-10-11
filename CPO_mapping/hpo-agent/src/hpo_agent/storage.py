"""Atomic host-owned artifacts. Model responses cannot write arbitrary deliverables."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class AuditTrail(list):
    """Keep the full audit in memory and append small live events after graph steps."""

    def __init__(self, event_path: Path):
        super().__init__()
        self.event_path = event_path

    def append(self, value):
        super().append(value)
        if "event" not in value:
            return
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now(UTC).isoformat(), **value},
                                    ensure_ascii=False) + "\n")


def to_json(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def fingerprint(value) -> str:
    return hashlib.sha256(to_json(value).encode()).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path: Path, value) -> None:
    atomic_write(path, to_json(value) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = []
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        return rows


def save_jsonl(path: Path, values) -> None:
    atomic_write(path, "".join(
        json.dumps(v.model_dump(mode="json") if isinstance(v, BaseModel) else v,
                   ensure_ascii=False, sort_keys=True) + "\n"
        for v in values
    ))
