from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=False)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    Path(temp_name).replace(path)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str, max_len: int = 80) -> str:
    value = value.lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "untitled"
    return value[:max_len].rstrip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def today_iso() -> str:
    return date.today().isoformat()


def flatten_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        item = normalize_space(value)
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def lower_lookup(values: Iterable[str]) -> Dict[str, str]:
    return {value.lower(): value for value in values}


def deep_get(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
