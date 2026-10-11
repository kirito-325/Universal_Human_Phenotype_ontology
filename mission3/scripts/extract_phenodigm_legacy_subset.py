from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CHUNK_SIZE = 4 * 1024 * 1024


def digest(path: Path) -> dict[str, object]:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            sha.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha.hexdigest(),
    }


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def load_targets(
    diseases_path: Path, models_path: Path, phenotypes_path: Path
) -> tuple[set[str], set[str], set[str]]:
    hpo_ids = {
        first(row, "hpo_id", "hp_id")
        for row in read_rows(diseases_path)
        if first(row, "hpo_id", "hp_id")
    }
    single_gene_models = {
        first(row, "model_id", "genotype_id")
        for row in read_rows(models_path)
        if first(row, "model_id", "genotype_id")
        and ";" not in first(row, "human_gene_symbol", "gene_symbol")
    }
    mpo_ids = {
        first(row, "mpo_id", "mp_id")
        for row in read_rows(phenotypes_path)
        if first(row, "model_id", "genotype_id") in single_gene_models
        and first(row, "status", "phenotype_status").lower().replace(" ", "_")
        == "abnormal"
        and first(row, "mpo_id", "mp_id")
    }
    return hpo_ids, mpo_ids, single_gene_models


def extract(
    matrix_path: Path,
    diseases_path: Path,
    models_path: Path,
    phenotypes_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hp_mp_mappings.tsv.gz"
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    hpo_ids, candidate_mpo_ids, single_gene_models = load_targets(
        diseases_path, models_path, phenotypes_path
    )
    source_rows = 0
    retained_rows = 0
    malformed_rows = 0
    retained_hpo: set[str] = set()
    retained_mp: set[str] = set()
    retained_candidate_mp: set[str] = set()
    source_candidate_mp: set[str] = set()
    fields = ["hp_id", "hp_name", "mp_id", "mp_name", "simj", "ic", "score"]
    try:
        with bz2.open(matrix_path, "rt", encoding="utf-8", newline="") as source:
            with partial.open("wb") as binary_target:
                compressed_target = gzip.GzipFile(
                    filename="", mode="wb", fileobj=binary_target, mtime=0
                )
                target = io.TextIOWrapper(
                    compressed_target, encoding="utf-8", newline=""
                )
                writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for line in source:
                    source_rows += 1
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) != 6:
                        malformed_rows += 1
                        continue
                    hp_id, hp_name, mp_id, mp_name, simj_raw, ic_raw = parts
                    if mp_id in candidate_mpo_ids:
                        source_candidate_mp.add(mp_id)
                    if source_rows % 5_000_000 == 0:
                        print(
                            f"scanned {source_rows:,} rows; retained {retained_rows:,}",
                            flush=True,
                        )
                    if hp_id not in hpo_ids:
                        continue
                    try:
                        simj = float(simj_raw)
                        ic = float(ic_raw)
                    except ValueError:
                        malformed_rows += 1
                        continue
                    if not math.isfinite(simj) or not math.isfinite(ic):
                        malformed_rows += 1
                        continue
                    writer.writerow(
                        {
                            "hp_id": hp_id,
                            "hp_name": hp_name,
                            "mp_id": mp_id,
                            "mp_name": mp_name,
                            "simj": f"{simj:.10g}",
                            "ic": f"{ic:.10g}",
                            "score": f"{math.sqrt(max(0.0, simj * ic)):.10g}",
                        }
                    )
                    retained_rows += 1
                    retained_hpo.add(hp_id)
                    retained_mp.add(mp_id)
                    if mp_id in candidate_mpo_ids:
                        retained_candidate_mp.add(mp_id)
                target.close()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(output_path)
    missing_hpo = sorted(hpo_ids - retained_hpo)
    missing_candidate_mp = sorted(candidate_mpo_ids - retained_candidate_mp)
    absent_source_candidate_mp = sorted(candidate_mpo_ids - source_candidate_mp)
    presence_path = output_dir / "candidate_mpo_presence.tsv"
    with presence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mp_id", "present_in_legacy_matrix", "has_benchmark_hpo_pair"],
            delimiter="\t",
        )
        writer.writeheader()
        for mp_id in sorted(candidate_mpo_ids):
            writer.writerow(
                {
                    "mp_id": mp_id,
                    "present_in_legacy_matrix": (
                        "true" if mp_id in source_candidate_mp else "false"
                    ),
                    "has_benchmark_hpo_pair": (
                        "true" if mp_id in retained_candidate_mp else "false"
                    ),
                }
            )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_matrix": digest(matrix_path),
        "inputs": {
            "diseases": digest(diseases_path),
            "models": digest(models_path),
            "model_phenotypes": digest(phenotypes_path),
        },
        "output": digest(output_path),
        "candidate_mpo_presence": digest(presence_path),
        "selection_rule": "all legacy PhenoDigm rows whose HP ID occurs in the frozen benchmark",
        "pair_score": "sqrt(simJ * IC)",
        "counts": {
            "source_rows": source_rows,
            "retained_rows": retained_rows,
            "malformed_rows": malformed_rows,
            "target_hpo": len(hpo_ids),
            "retained_hpo": len(retained_hpo),
            "missing_hpo": len(missing_hpo),
            "single_gene_models": len(single_gene_models),
            "candidate_mpo": len(candidate_mpo_ids),
            "candidate_mpo_present_in_source": len(source_candidate_mp),
            "candidate_mpo_absent_from_source": len(absent_source_candidate_mp),
            "candidate_mpo_with_any_retained_pair": len(retained_candidate_mp),
            "candidate_mpo_without_retained_pair": len(missing_candidate_mp),
            "all_mp_in_retained_rows": len(retained_mp),
        },
        "missing_hpo_ids": missing_hpo,
        "candidate_mpo_absent_from_source": absent_source_candidate_mp,
        "candidate_mpo_without_retained_pair": missing_candidate_mp,
    }
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the frozen benchmark HP subset from legacy PhenoDigm"
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/external/phenodigm_legacy/hp_mp_mappings.bz2"),
    )
    parser.add_argument("--diseases", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-phenotypes", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/phenodigm_legacy/benchmark_v4"),
    )
    args = parser.parse_args()
    result = extract(
        args.matrix,
        args.diseases,
        args.models,
        args.model_phenotypes,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
