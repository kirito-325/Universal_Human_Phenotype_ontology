from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: str | Path, rows: Iterable[Mapping[str, Any]], fields: list[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ";".join(str(item) for item in value)
                    if isinstance((value := row.get(field, "")), (list, tuple, set))
                    else str(value).lower()
                    if isinstance(value, bool)
                    else value
                    for field in fields
                }
            )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_values(value: str) -> list[str]:
    return sorted(
        {
            token.strip()
            for token in value.replace("|", ";").replace(",", ";").split(";")
            if token.strip()
        }
    )


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "include"}


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def input_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": relative_or_absolute(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
