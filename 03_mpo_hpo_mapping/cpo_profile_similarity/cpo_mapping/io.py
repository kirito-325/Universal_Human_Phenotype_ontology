from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, object]:
    value = Path(path).resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def read_tsv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    value = Path(path)
    with value.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields:
        raise ValueError(f"missing TSV header: {value}")
    for number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"malformed TSV row {number}: {value}")
    return fields, rows


def write_tsv(
    path: str | Path,
    fields: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> int:
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{value.name}.", suffix=".part", dir=value.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(value)
        value.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def write_json(path: str | Path, payload: object) -> None:
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{value.name}.", suffix=".part", dir=value.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(value)
        value.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_text(path: str | Path, text: str) -> None:
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{value.name}.", suffix=".part", dir=value.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(value)
        value.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def software_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "pyarrow", "h5py", "PyYAML", "scipy"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result
