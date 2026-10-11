from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .utils import read_json, slugify, utc_now, write_json, write_text_atomic


def papers_dir(database: Path) -> Path:
    return database / "papers"


def record_dir_name(metadata: Dict[str, Any]) -> str:
    title_slug = metadata.get("slug") or slugify(metadata.get("title", ""), 48)
    doi = metadata.get("doi", "")
    pmid = metadata.get("pmid", "")
    if doi:
        prefix = "doi_" + slugify(doi, 70)
    elif pmid:
        prefix = "pmid_" + slugify(pmid, 32)
    else:
        prefix = "manual_" + title_slug[:40]
    name = f"{prefix}_{title_slug[:48].rstrip('_')}"
    return name[:140].rstrip("_")


def load_metadata(record_dir: Path) -> Dict[str, Any]:
    return read_json(record_dir / "metadata.json", default={}) or {}


def merge_metadata(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        merged = dict(incoming)
        merged["created_at"] = merged.get("created_at") or utc_now()
        merged["updated_at"] = utc_now()
        return merged

    merged = dict(existing)
    preserve_when_present = {"assets", "download", "created_at", "verified_status"}
    for key, value in incoming.items():
        if key in preserve_when_present and merged.get(key):
            continue
        if value not in ("", None, [], {}):
            merged[key] = value

    if "screening" in incoming:
        current = dict(existing.get("screening", {}))
        union_fields = {
            "query_matched",
            "single_cell_methods",
            "matched_topic_terms",
            "immune_cell_types",
            "discovery_cohorts",
            "disease_context",
            "dataset_accessions",
        }
        for key, value in incoming["screening"].items():
            if key in union_fields and isinstance(value, list):
                combined = list(current.get(key, [])) + value
                current[key] = list(dict.fromkeys(item for item in combined if item))
                continue
            if value not in ("", None, [], {}):
                current[key] = value
        merged["screening"] = current

    source = dict(existing.get("source", {}))
    incoming_source = incoming.get("source", {})
    for key in ("discovery", "metadata"):
        values = []
        values.extend(source.get(key, []))
        values.extend(incoming_source.get(key, []))
        source[key] = sorted(set(values))
    if incoming_source.get("publisher"):
        source["publisher"] = incoming_source["publisher"]
    merged["source"] = source
    merged["updated_at"] = utc_now()
    return merged


def find_existing_record(database: Path, metadata: Dict[str, Any]) -> Path:
    doi = metadata.get("doi", "").lower()
    pmid = metadata.get("pmid", "")
    for record_dir, current in iter_records(database):
        if doi and current.get("doi", "").lower() == doi:
            return record_dir
        if pmid and current.get("pmid", "") == pmid:
            return record_dir
    return papers_dir(database) / record_dir_name(metadata)


def record_lookup_keys(metadata: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    doi = str(metadata.get("doi") or "").strip().lower()
    pmid = str(metadata.get("pmid") or "").strip()
    if doi:
        keys.append(f"doi:{doi}")
    if pmid:
        keys.append(f"pmid:{pmid}")
    return keys


def build_record_lookup(database: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for record_dir, metadata in iter_records(database):
        for key in record_lookup_keys(metadata):
            lookup.setdefault(key, record_dir)
    return lookup


def write_record(
    database: Path,
    metadata: Dict[str, Any],
    record_lookup: Dict[str, Path] | None = None,
) -> Path:
    record_dir = None
    if record_lookup is not None:
        for key in record_lookup_keys(metadata):
            record_dir = record_lookup.get(key)
            if record_dir is not None:
                break
    if record_dir is None:
        record_dir = (
            find_existing_record(database, metadata)
            if record_lookup is None
            else papers_dir(database) / record_dir_name(metadata)
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "assets").mkdir(exist_ok=True)
    (record_dir / "extracted" / "tables").mkdir(parents=True, exist_ok=True)
    (record_dir / "extracted" / "text").mkdir(parents=True, exist_ok=True)
    existing = load_metadata(record_dir)
    merged = merge_metadata(existing, metadata)
    write_json(record_dir / "metadata.json", merged)
    manifest = record_dir / "assets" / "assets_manifest.json"
    if not manifest.exists():
        write_json(manifest, [])
    if record_lookup is not None:
        for key in record_lookup_keys(merged):
            record_lookup[key] = record_dir
    return record_dir


def iter_records(database: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    root = papers_dir(database)
    if not root.exists():
        return
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = read_json(metadata_path, default={}) or {}
        yield metadata_path.parent, metadata


def index_entry(record_dir: Path, metadata: Dict[str, Any], database: Path) -> Dict[str, Any]:
    screening = metadata.get("screening", {})
    return {
        "record_path": str(record_dir.relative_to(database)),
        "id": metadata.get("id", ""),
        "doi": metadata.get("doi", ""),
        "pmid": metadata.get("pmid", ""),
        "pmcid": metadata.get("pmcid", ""),
        "title": metadata.get("title", ""),
        "journal": metadata.get("journal", ""),
        "publication_date": metadata.get("publication_date", ""),
        "publication_year": metadata.get("publication_year"),
        "cns_family": metadata.get("cns_family", ""),
        "url": metadata.get("url", ""),
        "relevance_status": screening.get("relevance_status", ""),
        "discovery_cohorts": screening.get("discovery_cohorts", []),
        "single_cell_methods": screening.get("single_cell_methods", []),
        "matched_topic_terms": screening.get("matched_topic_terms", []),
        "immune_cell_types": screening.get("immune_cell_types", []),
        "asset_count": len(metadata.get("assets", [])),
        "updated_at": metadata.get("updated_at", ""),
    }


def rebuild_index(database: Path) -> int:
    lines: List[str] = []
    for record_dir, metadata in iter_records(database):
        lines.append(json.dumps(index_entry(record_dir, metadata, database), ensure_ascii=True))
    text = "\n".join(lines)
    if text:
        text += "\n"
    write_text_atomic(database / "index.jsonl", text)
    return len(lines)
