#!/usr/bin/env python3
"""Build middle-layer-independent disease research packets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import input_record, read_json, read_tsv, truthy, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_BATCH = DEFAULT_ROOT / "mission3/data/benchmark/benchmark_v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    batch = args.batch_dir.resolve()
    selection_manifest = read_json(batch / "selection_manifest.json")
    if not selection_manifest.get("selection_protocol", {}).get(
        "middle_layer_independent"
    ):
        raise ValueError("Selection manifest does not assert middle-layer independence")
    candidates = read_tsv(batch / "candidate_diseases.tsv")
    profiles: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(batch / "human_disease_profiles.source.tsv"):
        profiles[row["disease_id"]].append(row)
    gold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(batch / "gold_models.source.tsv"):
        gold[row["disease_id"]].append(row)

    packets_dir = batch / "research_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_entries = []
    for candidate in candidates:
        rank = int(candidate["selection_rank"])
        disease_id = candidate["disease_id"]
        phenotype_rows = sorted(
            profiles[disease_id],
            key=lambda row: (-float(row["weight"]), row["hpo_id"]),
        )
        gold_rows = sorted(gold[disease_id], key=lambda row: row["model_id"])
        packet = {
            "packet_schema_version": "1.0",
            "selection_rank": rank,
            "disease_id": disease_id,
            "disease_name": candidate["disease_name"],
            "gene_symbol": candidate["gene_symbol"],
            "source_inheritance": candidate["inheritance"],
            "allowed_hpo_ids": sorted(row["hpo_id"] for row in phenotype_rows),
            "human_phenotype_profile": [
                {
                    "hpo_id": row["hpo_id"],
                    "hpo_name": row["hpo_name"],
                    "frequency": row.get("frequency", ""),
                    "onset": row.get("onset", ""),
                    "weight": float(row["weight"]),
                    "source_core": truthy(row.get("is_core", "")),
                    "source_reference": row.get("reference", ""),
                }
                for row in phenotype_rows
            ],
            "source_gold_models": [
                {
                    "model_id": row["model_id"],
                    "model_name": row["model_name"],
                    "mouse_gene_symbol": row["mouse_gene_symbol"],
                    "human_gene_symbol": row["human_gene_symbol"],
                    "allele_id": row["allele_id"],
                    "allele_type": row["allele_type"],
                    "zygosity": row["zygosity"],
                    "conditional": truthy(row["conditional"]),
                    "genetic_background": row["genetic_background"],
                    "known_disease_ids": row["known_disease_ids"],
                    "mgi_url": f"https://www.informatics.jax.org/allele/genoview/{row['model_id']}",
                    "source": row["gold_source"],
                }
                for row in gold_rows
            ],
            "authoritative_starting_points": {
                "hpo_disease_url": (
                    "https://hpo.jax.org/browse/disease/"
                    + disease_id.replace(":", ":")
                ),
                "mgi_search_url": "https://www.informatics.jax.org/",
                "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
            "benchmark_constraints": {
                "minimum_core_hpo": 3,
                "maximum_core_hpo": 12,
                "minimum_accepted_gold_models": 1,
                "minimum_independent_sources": 2,
            },
        }
        slug = f"{rank:02d}_{disease_id.replace(':', '_')}"
        packet_path = packets_dir / f"{slug}.json"
        write_json(packet_path, packet)
        packet_entries.append(
            {
                "selection_rank": rank,
                "disease_id": disease_id,
                "disease_name": candidate["disease_name"],
                "gene_symbol": candidate["gene_symbol"],
                "packet_path": str(packet_path.relative_to(root)),
                "packet": input_record(packet_path, root),
                "allowed_hpo_count": len(packet["allowed_hpo_ids"]),
                "source_gold_model_count": len(gold_rows),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_packets_built",
        "middle_layer_independent": True,
        "mouse_phenotype_terms_withheld": True,
        "packet_count": len(packet_entries),
        "packets": packet_entries,
        "inputs": {
            "selection_manifest": input_record(batch / "selection_manifest.json", root),
            "candidate_diseases": input_record(batch / "candidate_diseases.tsv", root),
            "human_disease_profiles": input_record(
                batch / "human_disease_profiles.source.tsv", root
            ),
            "source_gold_models": input_record(batch / "gold_models.source.tsv", root),
        },
        "schema": input_record(args.schema, root),
        "prompt_template": input_record(args.prompt_template, root),
    }
    write_json(batch / "research_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--schema",
        type=Path,
        default=SKILL_DIR / "assets/research_review.schema.json",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=SKILL_DIR / "assets/research_prompt.md",
    )
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(
        json.dumps(
            {"status": manifest["status"], "packet_count": manifest["packet_count"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
