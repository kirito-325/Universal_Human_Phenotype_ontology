from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from .linker import (
    AnatomyConstraintIndex,
    HPOEvidenceIndex,
    build_parser as build_inference_parser,
    infer_hpo_terms,
)
from .paths import BATCH_DIR, LINK_DIR


DEFAULT_OUTPUT_DIR = LINK_DIR
MERGED_FILES = (
    "hpo_seed_genes.tsv",
    "hpo_cell_type_candidates.tsv",
    "hpo_cell_phenotype_candidates.tsv",
    "hpo_cell_mechanism_triples.tsv",
    "hpo_cell_general_mechanism_triples.tsv",
)
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return {"path": str(source.resolve()), "exists": False}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "exists": True,
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def stable_digest(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def eligible_hpo_rows(
    hpo_index: HPOEvidenceIndex,
    anatomy_index: AnatomyConstraintIndex,
    limit: int = 0,
) -> list[dict[str, Any]]:
    ids = [
        hpo_id
        for hpo_id in sorted(hpo_index.term_gene_weights)
        if anatomy_index.specific_hpo_anatomy(hpo_id)
    ]
    if limit:
        ids = ids[:limit]
    rows: list[dict[str, Any]] = []
    for order, hpo_id in enumerate(ids, start=1):
        anatomy_ids = anatomy_index.specific_hpo_anatomy(hpo_id)
        rows.append(
            {
                "order": order,
                "hpo_id": hpo_id,
                "hpo_name": hpo_index.names.get(hpo_id, hpo_id),
                "seed_gene_count": len(hpo_index.seeds(hpo_id)),
                "specific_anatomy_ids": ";".join(anatomy_ids),
                "specific_anatomy_labels": ";".join(
                    anatomy_index.anatomy_labels_for(anatomy_ids)
                ),
            }
        )
    return rows


def ontology_coverage_rows(
    hpo_index: HPOEvidenceIndex,
    anatomy_index: AnatomyConstraintIndex,
    selected_hpo_ids: set[str],
) -> list[dict[str, Any]]:
    gene_ids = set(hpo_index.term_gene_weights)
    rows: list[dict[str, Any]] = []
    for order, hpo_id in enumerate(sorted(hpo_index.names), start=1):
        has_genes = hpo_id in gene_ids
        anatomy_ids = anatomy_index.specific_hpo_anatomy(hpo_id)
        has_anatomy = bool(anatomy_ids)
        eligible = has_genes and has_anatomy
        if not has_genes:
            reason = "no_direct_high_quality_gene_evidence"
        elif not has_anatomy:
            reason = "no_specific_uberon_anatomy"
        else:
            reason = ""
        rows.append(
            {
                "ontology_order": order,
                "hpo_id": hpo_id,
                "hpo_name": hpo_index.names[hpo_id],
                "has_direct_high_quality_gene_evidence": str(has_genes).lower(),
                "seed_gene_count": len(hpo_index.seeds(hpo_id)) if has_genes else 0,
                "has_specific_uberon_anatomy": str(has_anatomy).lower(),
                "specific_anatomy_ids": ";".join(anatomy_ids),
                "specific_anatomy_labels": ";".join(
                    anatomy_index.anatomy_labels_for(anatomy_ids)
                ),
                "eligible_for_default_inference": str(eligible).lower(),
                "selected_for_this_run": str(hpo_id in selected_hpo_ids).lower(),
                "exclusion_reason": reason,
            }
        )
    return rows


def write_eligible_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "order",
        "hpo_id",
        "hpo_name",
        "seed_gene_count",
        "specific_anatomy_ids",
        "specific_anatomy_labels",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_ontology_coverage(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    fields = (
        "ontology_order",
        "hpo_id",
        "hpo_name",
        "has_direct_high_quality_gene_evidence",
        "seed_gene_count",
        "has_specific_uberon_anatomy",
        "specific_anatomy_ids",
        "specific_anatomy_labels",
        "eligible_for_default_inference",
        "selected_for_this_run",
        "exclusion_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def inference_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "hpo_obo": args.hpo_obo,
        "hpoa": args.hpoa,
        "gene_associations": args.gene_associations,
        "markers_path": args.markers,
        "functions_path": args.functions,
        "constraints_path": args.constraints,
        "classification_path": args.classification,
        "output_dir": output_dir,
        "top_k": args.top_k,
        "top_cells": args.top_cells,
        "top_phenotypes": args.top_phenotypes,
        "min_overlap": args.min_overlap,
        "ppi_path": args.ppi,
        "permutations": args.permutations,
        "restart": args.restart,
        "random_seed": args.random_seed,
        "hpo_anatomy_path": args.hpo_anatomy,
        "cl_anatomy_path": args.cl_anatomy,
        "uberon_ancestors_path": args.uberon_ancestors,
        "network_q_threshold": args.network_q_threshold,
        "network_min_support_genes": args.network_min_support_genes,
        "network_support_z": args.network_support_z,
        "pair_overlap_weight": args.pair_overlap_weight,
        "pair_rank_weight": args.pair_rank_weight,
        "pair_three_way_weight": args.pair_three_way_weight,
        "general_phenotypes_path": args.general_phenotypes,
        "top_general_phenotypes": args.top_general_phenotypes,
        "top_general_k": args.top_general_k,
    }


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "hpo_obo": args.hpo_obo,
        "hpoa": args.hpoa,
        "gene_associations": args.gene_associations,
        "markers": args.markers,
        "functions": args.functions,
        "constraints": args.constraints,
        "classification": args.classification,
        "general_phenotypes": args.general_phenotypes,
        "hpo_anatomy": args.hpo_anatomy,
        "cl_anatomy": args.cl_anatomy,
        "uberon_ancestors": args.uberon_ancestors,
        "ppi": args.ppi,
    }
    return {
        "inputs": {
            name: file_fingerprint(Path(path) if path else None)
            for name, path in input_paths.items()
        },
        "parameters": {
            "top_k": args.top_k,
            "top_cells": args.top_cells,
            "top_phenotypes": args.top_phenotypes,
            "min_overlap": args.min_overlap,
            "pair_rank_weight": args.pair_rank_weight,
            "pair_three_way_weight": args.pair_three_way_weight,
            "pair_overlap_weight": args.pair_overlap_weight,
            "top_general_phenotypes": args.top_general_phenotypes,
            "top_general_k": args.top_general_k,
            "ppi": str(args.ppi) if args.ppi else "",
            "permutations": args.permutations,
            "restart": args.restart,
            "random_seed": args.random_seed,
            "network_q_threshold": args.network_q_threshold,
            "network_min_support_genes": args.network_min_support_genes,
            "network_support_z": args.network_support_z,
        },
        "eligibility": (
            "direct high-quality exact HPO gene evidence and at least one "
            "non-generic most-specific UBERON target"
        ),
        "hpo_parent_child_inheritance": False,
        "cell_go_parent_child_inheritance": False,
    }


def batch_is_resumable(
    batch_dir: Path,
    config_digest: str,
    hpo_ids: Sequence[str],
) -> bool:
    sentinel = batch_dir / "batch_run_manifest.json"
    if not sentinel.exists():
        return False
    try:
        value = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "complete":
        return False
    if value.get("config_digest") != config_digest:
        return False
    if value.get("hpo_ids") != list(hpo_ids):
        return False
    return all((batch_dir / filename).is_file() for filename in MERGED_FILES) and (
        batch_dir / "hpo_cell_mechanism_manifest.json"
    ).is_file()


def run_batches(
    args: argparse.Namespace,
    hpo_ids: Sequence[str],
    config_digest: str,
) -> list[Path]:
    batches_root = args.batches_dir
    batches_root.mkdir(parents=True, exist_ok=True)
    batches = list(chunked(hpo_ids, args.batch_size))
    batch_dirs: list[Path] = []
    for index, ids in enumerate(batches, start=1):
        batch_dir = batches_root / f"batch_{index:04d}"
        batch_dirs.append(batch_dir)
        if not args.overwrite and batch_is_resumable(batch_dir, config_digest, ids):
            print(
                json.dumps(
                    {
                        "event": "batch_resumed",
                        "batch": index,
                        "batch_count": len(batches),
                        "hpo_count": len(ids),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        started_at = utc_now()
        print(
            json.dumps(
                {
                    "event": "batch_start",
                    "batch": index,
                    "batch_count": len(batches),
                    "hpo_count": len(ids),
                    "first_hpo": ids[0],
                    "last_hpo": ids[-1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        manifest = infer_hpo_terms(ids, **inference_kwargs(args, batch_dir))
        sentinel = {
            "status": "complete",
            "batch": index,
            "batch_count": len(batches),
            "started_at": started_at,
            "completed_at": utc_now(),
            "config_digest": config_digest,
            "hpo_ids": list(ids),
            "output": manifest["output"],
        }
        atomic_json(batch_dir / "batch_run_manifest.json", sentinel)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "batch": index,
                    "batch_count": len(batches),
                    "specific_triples": manifest["output"]["triple_count"],
                    "general_triples": manifest["output"]["general_triple_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return batch_dirs


def merge_tsv(batch_dirs: Sequence[Path], filename: str, output_path: Path) -> int:
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    row_count = 0
    expected_header = ""
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        for batch_index, batch_dir in enumerate(batch_dirs):
            source_path = batch_dir / filename
            with source_path.open(encoding="utf-8", newline="") as source:
                header = source.readline()
                if batch_index == 0:
                    expected_header = header
                    destination.write(header)
                elif header != expected_header:
                    raise ValueError(f"Header mismatch while merging {source_path}")
                for line in source:
                    destination.write(line)
                    row_count += 1
    temporary.replace(output_path)
    return row_count


def triple_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            value = (
                f"{row['hpo_id']}\t{row['cell_type_id']}\t"
                f"{row['cell_phenotype_id']}"
            )
            if count:
                digest.update(b"\n")
            digest.update(value.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def merge_manifests(
    batch_dirs: Sequence[Path],
    output_dir: Path,
    config_digest: str,
    hpo_ids: Sequence[str],
    row_counts: Mapping[str, int],
    batch_size: int,
) -> dict[str, Any]:
    manifests = [
        json.loads(
            (batch_dir / "hpo_cell_mechanism_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for batch_dir in batch_dirs
    ]
    if not manifests:
        raise ValueError("No batch manifests to merge")
    base = dict(manifests[0])
    base["queries"] = [query for manifest in manifests for query in manifest["queries"]]
    if [query["hpo_id"] for query in base["queries"]] != list(hpo_ids):
        raise ValueError("Merged query order does not match the eligible HPO list")

    specific_count, specific_digest = triple_digest(
        output_dir / "hpo_cell_mechanism_triples.tsv"
    )
    general_count, general_digest = triple_digest(
        output_dir / "hpo_cell_general_mechanism_triples.tsv"
    )
    if specific_count != row_counts["hpo_cell_mechanism_triples.tsv"]:
        raise ValueError("Specific triple count changed during digest calculation")
    if general_count != row_counts["hpo_cell_general_mechanism_triples.tsv"]:
        raise ValueError("General triple count changed during digest calculation")
    base["output"] = {
        "triple_count": specific_count,
        "triple_digest": specific_digest,
        "general_triple_count": general_count,
        "general_triple_digest": general_digest,
        "seed_gene_rows": row_counts["hpo_seed_genes.tsv"],
        "cell_candidate_rows": row_counts["hpo_cell_type_candidates.tsv"],
        "phenotype_candidate_rows": row_counts[
            "hpo_cell_phenotype_candidates.tsv"
        ],
    }
    base["full_run"] = {
        "generated_at": utc_now(),
        "config_digest": config_digest,
        "eligibility": (
            "all HPO terms with direct high-quality exact gene evidence and at "
            "least one non-generic most-specific UBERON target"
        ),
        "eligible_hpo_count": len(hpo_ids),
        "batch_size": batch_size,
        "batch_count": len(batch_dirs),
        "batching_changes_algorithm": False,
        "ppi_default_enabled": False,
    }
    atomic_json(output_dir / "hpo_cell_mechanism_manifest.json", base)
    return base


def merge_outputs(
    batch_dirs: Sequence[Path],
    output_dir: Path,
    config_digest: str,
    hpo_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    row_counts: dict[str, int] = {}
    for filename in MERGED_FILES:
        row_counts[filename] = merge_tsv(
            batch_dirs, filename, output_dir / filename
        )
        print(
            json.dumps(
                {
                    "event": "file_merged",
                    "file": filename,
                    "rows": row_counts[filename],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    manifest = merge_manifests(
        batch_dirs,
        output_dir,
        config_digest,
        hpo_ids,
        row_counts,
        batch_size,
    )
    queries = manifest["queries"]
    hpo_with_both = sum(
        bool(query.get("triple_count"))
        and bool(query.get("general_triple_count"))
        for query in queries
    )
    hpo_specific_only = sum(
        bool(query.get("triple_count"))
        and not query.get("general_triple_count")
        for query in queries
    )
    hpo_general_only = sum(
        not query.get("triple_count")
        and bool(query.get("general_triple_count"))
        for query in queries
    )
    hpo_without_results = sum(
        not query.get("triple_count")
        and not query.get("general_triple_count")
        for query in queries
    )
    coverage_counts: dict[str, int] = {
        "eligible": 0,
        "no_direct_high_quality_gene_evidence": 0,
        "no_specific_uberon_anatomy": 0,
    }
    coverage_path = output_dir / "hpo_ontology_coverage.tsv"
    with coverage_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["eligible_for_default_inference"] == "true":
                coverage_counts["eligible"] += 1
            else:
                coverage_counts[row["exclusion_reason"]] += 1
    summary = {
        "generated_at": utc_now(),
        "output_dir": str(output_dir.resolve()),
        "ontology_hpo_count": sum(coverage_counts.values()),
        "eligible_hpo_count": len(hpo_ids),
        "ineligible_hpo_counts": {
            key: value for key, value in coverage_counts.items() if key != "eligible"
        },
        "hpo_with_specific_results": sum(
            bool(query.get("triple_count")) for query in queries
        ),
        "hpo_with_general_results": sum(
            bool(query.get("general_triple_count")) for query in queries
        ),
        "hpo_with_both_channels": hpo_with_both,
        "hpo_with_specific_only": hpo_specific_only,
        "hpo_with_general_only": hpo_general_only,
        "hpo_without_results": hpo_without_results,
        "specific_triple_count": manifest["output"]["triple_count"],
        "general_triple_count": manifest["output"]["general_triple_count"],
        "specific_three_way_overlap_triple_count": sum(
            int(query.get("three_way_overlap_triple_count", 0)) for query in queries
        ),
        "general_three_way_overlap_triple_count": sum(
            int(query.get("general_three_way_overlap_triple_count", 0))
            for query in queries
        ),
        "seed_gene_rows": manifest["output"]["seed_gene_rows"],
        "cell_candidate_rows": manifest["output"]["cell_candidate_rows"],
        "phenotype_candidate_rows": manifest["output"][
            "phenotype_candidate_rows"
        ],
        "network_enabled": bool(manifest["design"]["network_diffusion"]),
        "specific_triple_digest": manifest["output"]["triple_digest"],
        "general_triple_digest": manifest["output"]["general_triple_digest"],
    }
    atomic_json(output_dir / "full_run_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = config_payload(args)
    config_digest = stable_digest(config)
    atomic_json(
        args.output_dir / "full_run_config.json",
        {**config, "config_digest": config_digest},
    )

    hpo_index = HPOEvidenceIndex(args.hpo_obo, args.hpoa, args.gene_associations)
    anatomy_index = AnatomyConstraintIndex(
        args.hpo_anatomy, args.cl_anatomy, args.uberon_ancestors
    )
    rows = eligible_hpo_rows(hpo_index, anatomy_index, args.limit)
    write_eligible_rows(args.output_dir / "eligible_hpo_terms.tsv", rows)
    hpo_ids = [str(row["hpo_id"]) for row in rows]
    coverage = ontology_coverage_rows(hpo_index, anatomy_index, set(hpo_ids))
    write_ontology_coverage(
        args.output_dir / "hpo_ontology_coverage.tsv", coverage
    )
    print(
        json.dumps(
            {
                "event": "eligible_hpo_ready",
                "eligible_hpo_count": len(hpo_ids),
                "batch_size": args.batch_size,
                "batch_count": math.ceil(len(hpo_ids) / args.batch_size),
                "config_digest": config_digest,
                "network_enabled": bool(args.ppi),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return {"eligible_hpo_count": len(hpo_ids), "config_digest": config_digest}
    batch_dirs = run_batches(args, hpo_ids, config_digest)
    summary = merge_outputs(
        batch_dirs,
        args.output_dir,
        config_digest,
        hpo_ids,
        args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    defaults = build_inference_parser().parse_args([])
    parser = argparse.ArgumentParser(
        description=(
            "Run the current default HPO-to-cell-mechanism logic for every HPO "
            "with direct high-quality genes and specific anatomy."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batches-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hpo-obo", type=Path, default=defaults.hpo_obo)
    parser.add_argument("--hpoa", type=Path, default=defaults.hpoa)
    parser.add_argument(
        "--gene-associations", type=Path, default=defaults.gene_associations
    )
    parser.add_argument("--markers", type=Path, default=defaults.markers)
    parser.add_argument("--functions", type=Path, default=defaults.functions)
    parser.add_argument("--constraints", type=Path, default=defaults.constraints)
    parser.add_argument("--classification", type=Path, default=defaults.classification)
    parser.add_argument(
        "--general-phenotypes", type=Path, default=defaults.general_phenotypes
    )
    parser.add_argument("--hpo-anatomy", type=Path, default=defaults.hpo_anatomy)
    parser.add_argument("--cl-anatomy", type=Path, default=defaults.cl_anatomy)
    parser.add_argument(
        "--uberon-ancestors", type=Path, default=defaults.uberon_ancestors
    )
    parser.add_argument("--ppi", type=Path, default=defaults.ppi)
    parser.add_argument("--permutations", type=int, default=defaults.permutations)
    parser.add_argument("--restart", type=float, default=defaults.restart)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--network-q-threshold",
        type=float,
        default=defaults.network_q_threshold,
    )
    parser.add_argument(
        "--network-min-support-genes",
        type=int,
        default=defaults.network_min_support_genes,
    )
    parser.add_argument(
        "--network-support-z", type=float, default=defaults.network_support_z
    )
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--top-cells", type=int, default=defaults.top_cells)
    parser.add_argument(
        "--top-phenotypes", type=int, default=defaults.top_phenotypes
    )
    parser.add_argument("--min-overlap", type=int, default=defaults.min_overlap)
    parser.add_argument(
        "--pair-rank-weight", type=float, default=defaults.pair_rank_weight
    )
    parser.add_argument(
        "--pair-three-way-weight",
        type=float,
        default=defaults.pair_three_way_weight,
    )
    parser.add_argument(
        "--pair-overlap-weight", type=float, default=defaults.pair_overlap_weight
    )
    parser.add_argument(
        "--top-general-phenotypes",
        type=int,
        default=defaults.top_general_phenotypes,
    )
    parser.add_argument(
        "--top-general-k", type=int, default=defaults.top_general_k
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
