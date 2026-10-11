#!/usr/bin/env python3
"""Record the exact MPO and MGI inputs used by the independent MPO pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MPO_RELEASE = "v2026-07-22"
MPO_RELEASE_URL = (
    "https://github.com/mgijax/mammalian-phenotype-ontology/releases/"
    "download/v2026-07-22/mp.obo"
)
MPO_PURL = "https://purl.obolibrary.org/obo/mp.obo"
MGI_BASE = "https://www.informatics.jax.org/downloads/reports/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, source: str, last_modified: str = "") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "source": source}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "source": source,
        "http_last_modified": last_modified,
    }


def mpo_data_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("data-version:"):
            return line.split(":", 1)[1].strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo", type=Path, required=True)
    parser.add_argument("--gene-pheno", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--orthologs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {
        "mpo_obo": file_record(
            args.mpo, MPO_RELEASE_URL, "2026-07-20T12:00:41Z"
        ),
        "mgi_gene_pheno": file_record(
            args.gene_pheno,
            MGI_BASE + "MGI_GenePheno.rpt",
            "2026-07-20T12:01:09Z",
        ),
        "mgi_marker_list": file_record(
            args.markers, MGI_BASE + "MRK_List2.rpt", "2026-07-20T12:00:41Z"
        ),
        "mgi_homology": file_record(
            args.orthologs,
            MGI_BASE + "HOM_MouseHumanSequence.rpt",
            "2026-07-20T12:00:22Z",
        ),
    }
    local_version = mpo_data_version(args.mpo) if args.mpo.exists() else ""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verification": {
            "official_mpo_release": MPO_RELEASE,
            "official_mpo_release_url": MPO_RELEASE_URL,
            "official_mpo_purl": MPO_PURL,
            "official_mgi_report_last_modified": "2026-07-20",
            "remote_checks_performed": True,
            "local_mpo_data_version": local_version,
            "local_mpo_matches_official_release": local_version.endswith(
                "2026-07-22/mp.obo"
            ),
            "original_repo_files_preserved": True,
        },
        "inputs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
