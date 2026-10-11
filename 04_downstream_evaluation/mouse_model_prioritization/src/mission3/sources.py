from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .io import sha256


MGI_BASE_URL = "https://www.informatics.jax.org/downloads/reports"
HPO_RELEASE_BASE_URL = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download"
)

MGI_REPORTS: tuple[str, ...] = (
    "MGI_PhenoGenoMP.rpt",
    "MGI_PhenotypicAllele.rpt",
    "HOM_ProteinCoding.rpt",
    "MGI_Geno_DiseaseDO.rpt",
    "MGI_DiseaseMouseModel.rpt",
    "MGI_Pheno_Sex.rpt",
    "MGI_Recombinase_Full.rpt",
)

HPO_REPORTS: tuple[str, ...] = (
    "phenotype.hpoa",
    "genes_to_disease.txt",
    "hp.obo",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _hpo_version(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith("#version:"):
                return line.partition(":")[2].strip()
            if not line.startswith("#"):
                break
    return ""


def _download(
    url: str,
    target: Path,
    *,
    force: bool,
    timeout: int,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0 and not force:
        prior = previous or {}
        return {
            "status": "reused",
            "url": url,
            "resolved_url": _stable_url(str(prior.get("resolved_url", url))),
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "retrieved_at": prior.get("retrieved_at"),
            "last_modified": prior.get("last_modified"),
            "etag": prior.get("etag"),
        }

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mission3-mouse-model-prioritization/0.2"},
    )
    partial = target.with_name(f"{target.name}.part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
            resolved_url = _stable_url(response.geturl())
            last_modified = response.headers.get("Last-Modified")
            etag = response.headers.get("ETag")
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        os.replace(partial, target)
        return {
            "status": "downloaded",
            "url": url,
            "resolved_url": resolved_url,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "retrieved_at": _utc_now(),
            "last_modified": last_modified,
            "etag": etag,
        }
    finally:
        if partial.exists():
            partial.unlink()


def download_reference_data(
    *,
    output_dir: str | Path,
    source: str = "all",
    force: bool = False,
    timeout: int = 120,
    mgi_base_url: str = MGI_BASE_URL,
    hpo_base_url: str = HPO_RELEASE_BASE_URL,
) -> dict[str, object]:
    if source not in {"all", "mgi", "hpo"}:
        raise ValueError(f"Unsupported source: {source}")
    root = Path(output_dir)
    prior_records: Mapping[str, Mapping[str, object]] = {}
    prior_manifest_path = root / "download_manifest.json"
    if prior_manifest_path.is_file():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            if isinstance(prior_manifest.get("records"), dict):
                prior_records = prior_manifest["records"]
        except (OSError, json.JSONDecodeError):
            prior_records = {}
    requested: dict[str, Mapping[str, str]] = {}
    if source in {"all", "mgi"}:
        requested["mgi"] = {
            filename: f"{mgi_base_url.rstrip('/')}/{filename}" for filename in MGI_REPORTS
        }
    if source in {"all", "hpo"}:
        requested["hpo"] = {
            filename: f"{hpo_base_url.rstrip('/')}/{filename}" for filename in HPO_REPORTS
        }

    records: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    for collection, files in requested.items():
        for filename, url in files.items():
            key = f"{collection}/{filename}"
            try:
                records[key] = _download(
                    url,
                    root / collection / filename,
                    force=force,
                    timeout=timeout,
                    previous=prior_records.get(key),
                )
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                failures[key] = f"{type(error).__name__}: {error}"

    manifest = {
        "generated_at": _utc_now(),
        "source_selection": source,
        "records": records,
        "failures": failures,
        "complete": not failures and len(records) == sum(len(files) for files in requested.values()),
        "source_notes": {
            "mgi": "Official MGI tab-delimited reports; each retrieval is hash-pinned here.",
            "hpo": "Official HPO latest-release assets; resolved URLs record the concrete release.",
        },
        "versions": {
            "hpo": _hpo_version(root / "hpo" / "phenotype.hpoa"),
            "mgi_last_modified": max(
                (
                    str(record.get("last_modified", ""))
                    for key, record in records.items()
                    if key.startswith("mgi/") and record.get("last_modified")
                ),
                default="",
            ),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        names = ", ".join(sorted(failures))
        raise RuntimeError(
            f"Reference-data download was incomplete ({names}); see {root / 'download_manifest.json'}"
        )
    return manifest
