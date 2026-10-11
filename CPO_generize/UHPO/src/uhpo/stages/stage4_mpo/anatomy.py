#!/usr/bin/env python3
"""Materialize reasoner-inferred MPO-to-UBERON anatomy constraints."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


from uhpo.stages.stage3_hpo.anatomy import (
    DEFAULT_ENDPOINT,
    REDUNDANT_GRAPH,
    materialize_phenotype_anatomy,
    materialize_uberon_ancestors,
)


MAPPING_FIELDS = (
    "mp_id",
    "uberon_id",
    "uberon_label",
    "predicate",
    "source_graph",
)
ANCESTOR_FIELDS = ("uberon_id", "ancestor_uberon_id")


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8", newline="")
    )


def _read_mpo_ids(path: Path) -> list[str]:
    return sorted(
        {
            line.split(None, 1)[1].strip()
            for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.startswith("id: MP:")
        }
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required reasoner materialization not found: {path}")
    with _open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_gzip(
    path: Path, rows: Sequence[dict[str, str]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _validated_reasoner_mappings(
    rows: Sequence[dict[str, str]], source: Path | None = None
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        mp_id = row.get("mp_id", "").strip()
        uberon_id = row.get("uberon_id", "").strip()
        predicate = row.get("predicate", "").strip()
        source_graph = row.get("source_graph", "").strip()
        if (
            not mp_id.startswith("MP:")
            or not uberon_id.startswith("UBERON:")
            or predicate != "UPHENO:0000001"
            or source_graph != REDUNDANT_GRAPH
        ):
            location = f" in {source}" if source else ""
            raise ValueError(
                "MPO anatomy mappings"
                f"{location} are not exclusively derived from the Ubergraph "
                "redundant reasoner. Regenerate without --reuse-mappings."
            )
        normalized.append(
            {
                "mp_id": mp_id,
                "uberon_id": uberon_id,
                "uberon_label": row.get("uberon_label", "").strip(),
                "predicate": predicate,
                "source_graph": source_graph,
            }
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize MPO-to-UBERON and UBERON ancestor relationships from "
            "the same Ubergraph reasoner graph used by HPO_link."
        )
    )
    parser.add_argument("--mpo-obo", type=Path, required=True)
    parser.add_argument(
        "--existing-ancestors",
        type=Path,
        required=True,
        help=(
            "HPO_link reasoner-derived UBERON closure. It supplies CL anatomy "
            "ancestors and is merged with the MPO-specific reasoner closure."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--reuse-mappings",
        action="store_true",
        help=(
            "Reuse output-dir/mpo_uberon.tsv.gz and refresh only its reasoned "
            "UBERON ancestor closure."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    mpo_ids = _read_mpo_ids(args.mpo_obo)
    mapping_path = args.output_dir / "mpo_uberon.tsv.gz"
    if args.reuse_mappings:
        mapping_rows = _validated_reasoner_mappings(
            _read_rows(mapping_path), source=mapping_path
        )
    else:
        mapping_rows = _validated_reasoner_mappings(
            materialize_phenotype_anatomy(
                args.endpoint,
                mpo_ids,
                ontology_prefix="MP",
                subject_field="mp_id",
                batch_size=args.batch_size,
                retries=args.retries,
                timeout=args.timeout,
                progress_label="MPO",
            )
        )

    uberon_ids = sorted(
        {
            row.get("uberon_id", "").strip()
            for row in mapping_rows
            if row.get("uberon_id", "").strip().startswith("UBERON:")
        }
    )
    reasoned_ancestor_rows = materialize_uberon_ancestors(
        args.endpoint,
        uberon_ids,
        args.batch_size,
        args.retries,
        args.timeout,
    )

    existing_ancestor_rows = _read_rows(args.existing_ancestors)
    ancestor_pairs = {
        (row.get("uberon_id", "").strip(), row.get("ancestor_uberon_id", "").strip())
        for row in [*existing_ancestor_rows, *reasoned_ancestor_rows]
        if row.get("uberon_id", "").strip().startswith("UBERON:")
        and row.get("ancestor_uberon_id", "").strip().startswith("UBERON:")
    }
    ancestor_rows = [
        {"uberon_id": subject, "ancestor_uberon_id": ancestor}
        for subject, ancestor in sorted(ancestor_pairs)
    ]

    _write_gzip(mapping_path, mapping_rows, MAPPING_FIELDS)
    ancestor_path = args.output_dir / "uberon_ancestors_with_mpo.tsv.gz"
    _write_gzip(ancestor_path, ancestor_rows, ANCESTOR_FIELDS)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "ubergraph_redundant_reasoner",
        "endpoint": args.endpoint,
        "reasoner_graph": REDUNDANT_GRAPH,
        "phenotype_anatomy_predicate": "UPHENO:0000001",
        "inputs": {
            "mpo_obo": str(args.mpo_obo.resolve()),
            "existing_reasoned_ancestors": str(args.existing_ancestors.resolve()),
        },
        "reuse_mappings": args.reuse_mappings,
        "mpo_source_count": len(mpo_ids),
        "mpo_mapped_count": len(
            {
                row.get("mp_id", "").strip()
                for row in mapping_rows
                if row.get("mp_id", "").strip().startswith("MP:")
            }
        ),
        "mpo_uberon_mapping_count": len(mapping_rows),
        "uberon_source_count": len(uberon_ids),
        "reasoned_mpo_ancestor_count": len(reasoned_ancestor_rows),
        "existing_reasoned_ancestor_count": len(existing_ancestor_rows),
        "combined_ancestor_count": len(ancestor_rows),
        "design": {
            "mpo_anatomy": (
                "reasoner-inferred MP UPHENO:0000001 UBERON relationships; "
                "no MPO text-xref or MP-HPO bridge"
            ),
            "cl_anatomy": (
                "reuse HPO_link CL part_of/located_in/has_soma_location "
                "materialization from the same reasoner graph"
            ),
            "compatibility": (
                "most-specific MPO and CL UBERON locations must be equal or "
                "comparable in the reasoned UBERON subclass closure"
            ),
        },
        "outputs": {
            "mpo_uberon": str(mapping_path.resolve()),
            "ancestors": str(ancestor_path.resolve()),
        },
    }
    manifest_path = args.output_dir / "mpo_anatomy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
