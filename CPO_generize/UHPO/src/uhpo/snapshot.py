"""Build or verify the canonical manuscript data snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from uhpo.paths import CONFIG_ROOT, REPO_ROOT
from uhpo.stages.stage1_markers.paths import MARKER_DIR, RAW_CL_OBO
from uhpo.stages.stage2_cpo.paths import (
    CATALOG,
    CELL_MARKER_DIR,
    GOA_HUMAN,
    GO_FUNCTION_DIR,
    GO_OBO,
    HGNC,
    MGI_MARKERS as CPO_MGI_MARKERS,
    MOUSE_HUMAN_ORTHOLOGS,
    RELATION_DIR,
)
from uhpo.stages.stage3_hpo.paths import (
    CL_ANATOMY,
    GENES_TO_PHENOTYPE,
    HPO_ANATOMY,
    HPO_OBO,
    HPOA,
    LINK_DIR as HPO_LINK_DIR,
    UBERON_ANCESTORS,
)
from uhpo.stages.stage4_mpo.paths import (
    CLASSIFICATION_BATCH_MANIFEST,
    CLEAN_MPO_OBO,
    HUMAN_GENE_EVIDENCE,
    LINK_DIR as MPO_LINK_DIR,
    MGI_GENE_PHENO,
    MGI_HOMOLOGY,
    MGI_MARKERS as MPO_MGI_MARKERS,
    MPO_ANATOMY,
    MPO_OBO,
    MPO_UBERON_ANCESTORS,
    TERM_DECISIONS,
    TERM_DECISION_MANIFEST,
    VERSION_MANIFEST,
)


MANIFEST_PATH = CONFIG_ROOT / "snapshot_manifest.json"
CANONICAL_FILES: tuple[Path, ...] = (
    RAW_CL_OBO,
    CATALOG,
    MARKER_DIR / "expanded_marker_scores.tsv",
    MARKER_DIR / "significant_marker_scores.tsv",
    GO_OBO,
    GOA_HUMAN,
    HGNC,
    MOUSE_HUMAN_ORTHOLOGS,
    CPO_MGI_MARKERS,
    CELL_MARKER_DIR / "cell_marker_genes.tsv.gz",
    GO_FUNCTION_DIR / "go_function_genes.tsv",
    RELATION_DIR / "cell_go_phenotype_links_final.tsv",
    RELATION_DIR / "llm_cell_general_go_terms.tsv",
    RELATION_DIR / "llm_direct_manifest.json",
    HPO_OBO,
    HPOA,
    GENES_TO_PHENOTYPE,
    HPO_ANATOMY,
    CL_ANATOMY,
    UBERON_ANCESTORS,
    HPO_LINK_DIR / "hpo_cell_mechanism_triples.tsv",
    HPO_LINK_DIR / "hpo_cell_general_mechanism_triples.tsv",
    HPO_LINK_DIR / "hpo_cell_mechanism_manifest.json",
    MPO_OBO,
    MGI_GENE_PHENO,
    MPO_MGI_MARKERS,
    MGI_HOMOLOGY,
    VERSION_MANIFEST,
    TERM_DECISIONS,
    TERM_DECISION_MANIFEST,
    CLASSIFICATION_BATCH_MANIFEST,
    CLEAN_MPO_OBO,
    HUMAN_GENE_EVIDENCE,
    MPO_ANATOMY,
    MPO_UBERON_ANCESTORS,
    MPO_LINK_DIR / "mpo_cell_mechanism_triples.tsv",
    MPO_LINK_DIR / "mpo_cell_general_mechanism_triples.tsv",
    MPO_LINK_DIR / "mpo_cell_mechanism_manifest.json",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    newline_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            if path.suffix == ".tsv":
                newline_count += block.count(b"\n")
    if path.name.endswith(".tsv.gz"):
        with gzip.open(path, "rb") as handle:
            newline_count = sum(block.count(b"\n") for block in iter(lambda: handle.read(4 * 1024 * 1024), b""))
    record: dict[str, object] = {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    if path.suffix == ".tsv" or path.name.endswith(".tsv.gz"):
        record["data_rows"] = max(0, newline_count - 1)
    return record


def build_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    sources = list(paths)
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing canonical file(s): " + ", ".join(map(str, missing)))
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(fingerprint, sources))


def write_manifest() -> int:
    records = build_records(CANONICAL_FILES)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Completed four-stage manuscript pipeline; MPO-HPO WIP excluded",
        "files": records,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} fingerprints to {relative(MANIFEST_PATH)}")
    return 0


def verify_manifest() -> int:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {record["path"]: record for record in payload["files"]}
    paths = [REPO_ROOT / name for name in expected]
    actual = {record["path"]: record for record in build_records(paths)}
    failures = 0
    for name, wanted in expected.items():
        found = actual[name]
        differences = {
            key: (wanted.get(key), found.get(key))
            for key in ("bytes", "data_rows", "sha256")
            if wanted.get(key) != found.get(key)
        }
        if differences:
            failures += 1
            print(f"[mismatch] {name}: {differences}")
        else:
            print(f"[ok] {name}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Replace the baseline with current canonical files.")
    args = parser.parse_args(argv)
    return write_manifest() if args.write else verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
