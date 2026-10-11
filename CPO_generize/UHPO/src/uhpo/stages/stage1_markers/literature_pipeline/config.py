from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

from ..paths import LITERATURE_CONFIG


DEFAULT_CONFIG = LITERATURE_CONFIG


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_date_range(config: Dict[str, Any], start: str = "", end: str = "") -> Tuple[str, str]:
    settings = config.get("date_range", {})
    end_date = end or settings.get("end_date") or date.today().isoformat()
    start_date = start or settings.get("start_date")
    if start_date:
        return start_date, end_date

    years_back = int(settings.get("years_back", 10))
    end_obj = date.fromisoformat(end_date)
    try:
        start_obj = end_obj.replace(year=end_obj.year - years_back)
    except ValueError:
        start_obj = end_obj.replace(month=2, day=28, year=end_obj.year - years_back)
    return start_obj.isoformat(), end_date


def journal_family_map(config: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in config.get("journals", []):
        family = item.get("family", "")
        names = [item.get("name", "")]
        names.extend(item.get("aliases", []))
        for name in names:
            if name:
                mapping[normalize_journal_name(name)] = family
    return mapping


def journal_names(config: Dict[str, Any]):
    return [item["name"] for item in config.get("journals", []) if item.get("name")]


def normalize_journal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
