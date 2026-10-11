#!/usr/bin/env python3
"""Build immutable per-disease evidence packets for Codex cohort curation."""

from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
MISSION3 = ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization')
DEFAULT_BATCH = MISSION3 / "data" / "curation" / "formal_cohort_v1"
ACCEPTED_RELATIONS = {
    "causally_upstream_of",
    "equivalent_cellular_phenotype",
    "has_constituent_cellular_phenotype",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _claim_priority(claim: dict[str, Any]) -> tuple[Any, ...]:
    structural = (
        bool(claim.get("selected_cl_id"))
        and bool(claim.get("selected_go_id"))
        and claim.get("bridge_scope") == "direct_hpo"
        and claim.get("relation_type") in ACCEPTED_RELATIONS
        and str(claim.get("species", "")).lower() == "human"
    )
    return (
        not structural,
        claim.get("bridge_scope") != "direct_hpo",
        not bool(claim.get("selected_cl_id")),
        not bool(claim.get("selected_go_id")),
        str(claim.get("claim_id", "")),
    )


class SpanReader:
    def __init__(self, literature_root: Path) -> None:
        self.literature_root = literature_root
        self.cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    def get(self, hpo_id: str, run_id: str, span_id: str) -> dict[str, Any] | None:
        key = (hpo_id, run_id)
        if key not in self.cache:
            path = (
                self.literature_root
                / hpo_id.replace(":", "_")
                / "evidence"
                / "spans"
                / f"{run_id}.jsonl"
            )
            rows: dict[str, dict[str, Any]] = {}
            if path.is_file():
                rows = {
                    str(row.get("evidence_span_id", "")): row
                    for row in read_jsonl(path)
                }
            self.cache[key] = rows
        return self.cache[key].get(span_id)


def _model_context(
    selected_genes: set[str],
    models_path: Path,
    phenotypes_path: Path,
    mp_cpo_path: Path,
    mapping_path: Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, set[str]],
    set[str],
    dict[str, set[str]],
]:
    models_by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_model_ids: set[str] = set()
    for row in read_tsv(models_path):
        genes = {
            token.strip().upper()
            for value in (row.get("human_gene_symbol", ""), row.get("mouse_gene_symbol", ""))
            for token in value.replace("|", ";").replace(",", ";").split(";")
            if token.strip()
        }
        for gene in genes.intersection(selected_genes):
            models_by_gene[gene].append(row)
            selected_model_ids.add(row["model_id"])

    mp_by_model: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(phenotypes_path):
        if row.get("model_id") in selected_model_ids and row.get("mp_id"):
            mp_by_model[row["model_id"]].add(row["mp_id"])

    mp_cpo_ids = {row["mp_id"] for row in read_tsv(mp_cpo_path) if row.get("mp_id")}
    hpo_to_mp: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(mapping_path):
        hpo_id = row.get("hp_id") or row.get("hpo_id", "")
        mp_id = row.get("mp_id", "")
        if hpo_id and mp_id:
            hpo_to_mp[hpo_id].add(mp_id)
    return models_by_gene, mp_by_model, mp_cpo_ids, hpo_to_mp


def build_batch(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.limit <= 50:
        raise ValueError("--limit must be between 30 and 50")
    batch_dir = args.batch_dir.resolve()
    packets_dir = batch_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)

    queue = read_tsv(args.queue)
    selected_queue = queue[: args.limit]
    selected_ids = [row["disease_id"] for row in selected_queue]
    selected_id_set = set(selected_ids)
    selected_genes = {row["gene_symbol"].upper() for row in selected_queue}

    disease_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(args.diseases):
        if row.get("disease_id") in selected_id_set:
            disease_rows[row["disease_id"]].append(row)
    todo = {row["disease_id"]: row for row in read_tsv(args.todo)}

    claims_by_hpo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_hpo_ids = {
        row["hpo_id"]
        for rows in disease_rows.values()
        for row in rows
        if row.get("hpo_id")
    }
    for claim in read_jsonl(args.cpo_claims):
        hpo_id = str(claim.get("hpo_id", ""))
        if hpo_id in selected_hpo_ids:
            claims_by_hpo[hpo_id].append(claim)

    models_by_gene, mp_by_model, mp_cpo_ids, hpo_to_mp = _model_context(
        selected_genes,
        args.models,
        args.model_phenotypes,
        args.mp_cpo,
        args.mp_hpo_mapping,
    )
    span_reader = SpanReader(args.literature_root)
    packet_entries: list[dict[str, Any]] = []

    for rank, queue_row in enumerate(selected_queue, 1):
        disease_id = queue_row["disease_id"]
        rows = disease_rows.get(disease_id, [])
        if not rows:
            raise ValueError(f"Disease profile missing for {disease_id}")
        gene = queue_row["gene_symbol"].upper()
        candidate_models = models_by_gene.get(gene, [])
        candidate_mp = {
            mp_id
            for model in candidate_models
            for mp_id in mp_by_model.get(model["model_id"], set())
        }
        candidate_mp_cpo = candidate_mp.intersection(mp_cpo_ids)
        suggested_core = split_values(todo[disease_id].get("core_hpo_ids", ""))
        phenotype_rows = []
        comparability: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: (-float(item["weight"]), item["hpo_id"])):
            mapped = hpo_to_mp.get(row["hpo_id"], set())
            comparable = sorted(mapped.intersection(candidate_mp_cpo))
            comparability[row["hpo_id"]] = {
                "surface_mapped_candidate_mp_count": len(mapped.intersection(candidate_mp)),
                "comparable_mp_cpo_ids": comparable[:20],
                "comparable_with_candidate_model_cpo": bool(comparable),
            }
            phenotype_rows.append(
                {
                    "hpo_id": row["hpo_id"],
                    "hpo_name": row["hpo_name"],
                    "weight": float(row["weight"]),
                    "is_core_in_source": truthy(row["is_core"]),
                    "frequency": row.get("frequency", ""),
                    "onset": row.get("onset", ""),
                    "source_direction": row.get("direction", "unknown"),
                    **comparability[row["hpo_id"]],
                }
            )

        cpo_claims: list[dict[str, Any]] = []
        for hpo_id in sorted(selected_hpo_ids.intersection({row["hpo_id"] for row in rows})):
            for claim in sorted(claims_by_hpo.get(hpo_id, []), key=_claim_priority)[
                : args.max_claims_per_hpo
            ]:
                run_id = str(claim.get("source_evidence_run_id", ""))
                spans = []
                for span_id in claim.get("evidence_span_ids", []):
                    span = span_reader.get(hpo_id, run_id, str(span_id))
                    if span:
                        spans.append(
                            {
                                "evidence_span_id": span.get("evidence_span_id", ""),
                                "quote": span.get("quote", ""),
                                "record_id": span.get("record_id", ""),
                                "section": span.get("section", ""),
                                "location": span.get("location", ""),
                            }
                        )
                record_ids = [str(item) for item in claim.get("supporting_record_ids", [])]
                structural = (
                    bool(claim.get("selected_cl_id"))
                    and bool(claim.get("selected_go_id"))
                    and claim.get("bridge_scope") == "direct_hpo"
                    and claim.get("relation_type") in ACCEPTED_RELATIONS
                    and str(claim.get("species", "")).lower() == "human"
                    and bool(spans)
                )
                cpo_claims.append(
                    {
                        "claim_id": claim.get("claim_id", ""),
                        "hpo_id": hpo_id,
                        "hpo_label": claim.get("hpo_label", ""),
                        "selected_cl_id": claim.get("selected_cl_id", ""),
                        "selected_cl_label": claim.get("selected_cl_label", ""),
                        "selected_go_id": claim.get("selected_go_id", ""),
                        "selected_go_label": claim.get("selected_go_label", ""),
                        "cell_text": claim.get("cell_text", ""),
                        "process_text": claim.get("process_text", ""),
                        "quality_text": claim.get("quality_text", ""),
                        "direction": claim.get("direction", "unknown"),
                        "relation_type": claim.get("relation_type", ""),
                        "bridge_scope": claim.get("bridge_scope", ""),
                        "evidence_type": claim.get("evidence_type", ""),
                        "species": claim.get("species", ""),
                        "disease_ids": claim.get("disease_ids", []),
                        "supporting_record_ids": record_ids,
                        "source_urls": [
                            f"https://pubmed.ncbi.nlm.nih.gov/{record[5:]}/"
                            for record in record_ids
                            if record.startswith("pmid_")
                        ],
                        "evidence_spans": spans,
                        "machine_structural_candidate": structural,
                        "source_evidence_run_id": run_id,
                    }
                )

        packet = {
            "packet_schema_version": "1.0",
            "selection_rank": rank,
            "disease_id": disease_id,
            "disease_name": queue_row["disease_name"],
            "gene_symbol": queue_row["gene_symbol"],
            "source_inheritance": queue_row["inheritance"],
            "source_molecular_mechanism": queue_row["molecular_mechanism"],
            "allowed_hpo_ids": sorted(row["hpo_id"] for row in rows),
            "suggested_core_hpo_ids": suggested_core,
            "phenotype_profile": phenotype_rows,
            "cpo_claims": cpo_claims,
            "mouse_model_summary": {
                "candidate_model_count": int(queue_row["candidate_model_count"]),
                "known_positive_model_count": int(queue_row["known_positive_model_count"]),
                "candidate_mp_count": len(candidate_mp),
                "candidate_mp_cpo_count": len(candidate_mp_cpo),
                "candidate_mp_cpo_coverage": float(queue_row["candidate_mp_cpo_coverage"]),
                "models": [
                    {
                        key: model.get(key, "")
                        for key in (
                            "model_id",
                            "model_name",
                            "allele_type",
                            "zygosity",
                            "conditional",
                            "cre_driver",
                            "known_disease_ids",
                        )
                    }
                    for model in candidate_models[:50]
                ],
            },
            "pre_curation_metrics": {
                key: queue_row[key]
                for key in (
                    "surface_weighted_coverage",
                    "hpo_cpo_count",
                    "hpo_cpo_weighted_coverage",
                    "comparable_cpo_hpo_count",
                    "candidate_mp_cpo_coverage",
                    "surface_ready",
                    "cpo_ready",
                    "genetic_ready",
                    "failure_reasons",
                )
            },
            "curation_constraints": {
                "minimum_core_hpo": 3,
                "minimum_distinct_core_hpo_with_accepted_cpo": 2,
                "only_local_claim_ids_may_be_adjudicated": True,
                "codex_acceptance_requires_human_signoff": True,
            },
        }
        slug = disease_id.replace(":", "_")
        packet_path = packets_dir / f"{rank:02d}_{slug}.json"
        write_json(packet_path, packet)
        packet_entries.append(
            {
                "selection_rank": rank,
                "disease_id": disease_id,
                "disease_name": queue_row["disease_name"],
                "gene_symbol": queue_row["gene_symbol"],
                "packet_path": str(packet_path.relative_to(ROOT)),
                "packet_sha256": sha256(packet_path),
                "candidate_cpo_claim_count": len(cpo_claims),
                "machine_structural_cpo_claim_count": sum(
                    bool(claim["machine_structural_candidate"]) for claim in cpo_claims
                ),
            }
        )

    input_paths = {
        "queue": args.queue,
        "todo": args.todo,
        "diseases": args.diseases,
        "models": args.models,
        "model_phenotypes": args.model_phenotypes,
        "mp_hpo_mapping": args.mp_hpo_mapping,
        "mp_cpo": args.mp_cpo,
        "cpo_claims": args.cpo_claims,
        "schema": args.schema,
        "prompt_template": args.prompt_template,
    }
    manifest = {
        "batch_schema_version": "1.0",
        "batch_id": batch_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "packets_built",
        "requested_disease_count": args.limit,
        "selected_disease_count": len(packet_entries),
        "selection_rule": "first N rows of the deterministic provisional curation queue",
        "human_signoff_required": True,
        "pilot_ready": False,
        "selected_disease_ids": selected_ids,
        "packets": packet_entries,
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
    }
    write_json(batch_dir / "batch_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-claims-per-hpo", type=int, default=8)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--queue",
        type=Path,
        default=MISSION3 / "data" / "processed" / "cohort_provisional" / "curation_queue.tsv",
    )
    parser.add_argument(
        "--todo",
        type=Path,
        default=MISSION3 / "data" / "processed" / "cohort_provisional" / "disease_curation.todo.tsv",
    )
    parser.add_argument(
        "--diseases",
        type=Path,
        default=MISSION3 / "data" / "processed" / "diseases" / "diseases.all.tsv",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=MISSION3 / "data" / "processed" / "mgi" / "models.tsv",
    )
    parser.add_argument(
        "--model-phenotypes",
        type=Path,
        default=MISSION3 / "data" / "processed" / "mgi" / "model_phenotypes.tsv",
    )
    parser.add_argument(
        "--mp-hpo-mapping",
        type=Path,
        default=ROOT / "UHPO" / "data" / "mpo_hpo_mapping" / "mpo_hpo_mapping.tsv",
    )
    parser.add_argument(
        "--mp-cpo",
        type=Path,
        default=ROOT
        / "03_MPO_link"
        / "results"
        / "mpo_cpo_links"
        / "api_reasoner_final_20260726"
        / "mpo_cell_mechanism_triples.tsv",
    )
    parser.add_argument(
        "--cpo-claims",
        type=Path,
        default=DEFAULT_BATCH / "source_snapshot" / "hpo_cpo_candidate_links.partial.jsonl",
    )
    parser.add_argument(
        "--literature-root", type=Path, default=ROOT / "HPO_link_v6" / "literature"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=MISSION3 / "curation" / "formal_cohort_v1" / "disease_review.schema.json",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=MISSION3 / "curation" / "formal_cohort_v1" / "disease_review_prompt.md",
    )
    return parser.parse_args()


def main() -> None:
    manifest = build_batch(parse_args())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_disease_count": manifest["selected_disease_count"],
                "batch_id": manifest["batch_id"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
