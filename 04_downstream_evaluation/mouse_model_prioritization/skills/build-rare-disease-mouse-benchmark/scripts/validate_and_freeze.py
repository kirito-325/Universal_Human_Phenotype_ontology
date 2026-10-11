#!/usr/bin/env python3
"""Validate Codex research and freeze an independent benchmark when complete."""

from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    input_record,
    read_json,
    read_tsv,
    split_values,
    write_json,
    write_jsonl,
    write_tsv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = SKILL_DIR.parents[2]
DEFAULT_BATCH = DEFAULT_ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization/data/benchmark/benchmark_v1')

DISEASE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "hpo_id",
    "hpo_name",
    "weight",
    "is_core",
    "frequency",
    "onset",
    "direction",
    "inheritance",
    "molecular_mechanism",
    "source_disease_id",
    "reference",
]


def key_names(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            result.append(str(key).lower())
            result.extend(key_names(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(key_names(nested))
    return result


def packet_independence_errors(packet: dict[str, Any]) -> list[str]:
    forbidden_key_fragments = (
        "cpo",
        "cellular_mechanism",
        "mechanism_coverage",
        "prior_ranking",
        "ranking_result",
        "mouse_phenotype",
        "mp_term",
    )
    errors = []
    for key in key_names(packet):
        if any(fragment in key for fragment in forbidden_key_fragments):
            errors.append(f"forbidden_packet_key:{key}")
    return sorted(set(errors))


def validate(
    review: dict[str, Any], packet: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    errors = packet_independence_errors(packet)
    for field in ("disease_id", "disease_name", "gene_symbol"):
        if str(review.get(field, "")) != str(packet.get(field, "")):
            errors.append(f"{field}_mismatch")
    allowed_hpo = set(packet["allowed_hpo_ids"])
    core = review.get("core_hpo_ids", [])
    excluded = review.get("excluded_hpo_ids", [])
    if not isinstance(core, list) or not set(core).issubset(allowed_hpo):
        errors.append("core_hpo_outside_allowed_set")
        core = []
    if not 3 <= len(core) <= 12:
        errors.append("core_hpo_count_not_3_to_12")
    if not isinstance(excluded, list) or not set(excluded).issubset(allowed_hpo):
        errors.append("excluded_hpo_outside_allowed_set")
        excluded = []
    if set(core).intersection(excluded):
        errors.append("core_and_excluded_overlap")

    sources = review.get("sources", [])
    source_by_id = {
        str(source.get("source_id", "")): source
        for source in sources
        if isinstance(source, dict) and source.get("source_id")
    }
    if len(source_by_id) != len(sources):
        errors.append("duplicate_or_missing_source_id")
    valid_urls = [
        source
        for source in source_by_id.values()
        if str(source.get("url", "")).startswith(("https://", "http://"))
    ]
    if len({str(source["url"]) for source in valid_urls}) < 2:
        errors.append("fewer_than_two_resolvable_sources")
    supported = {
        support
        for source in valid_urls
        for support in source.get("supports", [])
    }
    for required in (
        "disease_identity",
        "molecular_mechanism",
        "core_phenotypes",
        "gold_mouse_model",
    ):
        if required not in supported:
            errors.append(f"missing_source_support:{required}")

    source_gold = {row["model_id"] for row in packet["source_gold_models"]}
    decisions = review.get("gold_model_decisions", [])
    decision_ids = [str(row.get("model_id", "")) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        errors.append("duplicate_gold_model_decision")
    if set(decision_ids) != source_gold:
        errors.append("gold_decisions_do_not_cover_source_models")
    accepted_gold = []
    for decision in decisions:
        unknown_sources = set(decision.get("source_ids", [])).difference(source_by_id)
        if unknown_sources:
            errors.append(f"unknown_source_ids_for_model:{decision.get('model_id')}")
        if decision.get("decision") == "accept":
            accepted_gold.append(str(decision.get("model_id", "")))
    if not accepted_gold:
        errors.append("no_accepted_gold_model")

    if review.get("decision") != "include":
        errors.append("codex_decision_not_include")
    if review.get("identity_confidence") not in {"high", "medium"}:
        errors.append("identity_confidence_too_low")
    if review.get("mechanism_class") == "unknown":
        errors.append("molecular_mechanism_unknown")
    if review.get("mechanism_confidence") not in {"high", "medium"}:
        errors.append("mechanism_confidence_too_low")
    readiness = review.get("readiness", {})
    for field in (
        "identity_ready",
        "core_hpo_ready",
        "genetic_mechanism_ready",
        "gold_models_ready",
    ):
        if readiness.get(field) is not True:
            errors.append(f"readiness_false:{field}")
    if readiness.get("blockers"):
        errors.append("readiness_blockers_present")
    return sorted(set(errors)), sorted(core), sorted(set(accepted_gold))


def render_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    table = []
    for row in rows:
        status = "valid" if row["valid"] else "pending"
        table.append(
            "<tr>"
            f"<td>{row['selection_rank']}</td>"
            f"<td><code>{html.escape(row['disease_id'])}</code><br>{html.escape(row['disease_name'])}</td>"
            f"<td>{html.escape(row['gene_symbol'])}</td>"
            f"<td>{row['core_hpo_count']}</td>"
            f"<td>{row['accepted_gold_model_count']}</td>"
            f"<td class='{status}'>{status}</td>"
            f"<td>{html.escape(row['errors'])}</td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent benchmark curation</title><style>
body{{font-family:system-ui,sans-serif;margin:0;color:#202124}}header{{padding:24px 32px;background:#17324d;color:#fff}}main{{padding:24px 32px;overflow:auto}}h1{{font-size:24px;margin:0 0 8px}}.metrics{{display:flex;gap:28px;margin:18px 0}}.metrics strong{{font-size:22px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}th{{background:#f4f6f8;position:sticky;top:0}}.valid{{color:#087f5b;font-weight:700}}.pending{{color:#a61e2d;font-weight:700}}
</style></head><body><header><h1>独立罕见病小鼠模型测评集</h1><p>疾病真值冻结与细胞机制评测完全分离。</p></header><main><div class="metrics"><div><strong>{summary['candidate_count']}</strong>候选疾病</div><div><strong>{summary['completed_review_count']}</strong>Codex完成</div><div><strong>{summary['valid_review_count']}</strong>校验通过</div><div><strong>{str(summary['frozen']).lower()}</strong>Frozen</div></div><table><thead><tr><th>#</th><th>疾病</th><th>基因</th><th>核心HPO</th><th>Gold模型</th><th>状态</th><th>问题</th></tr></thead><tbody>{''.join(table)}</tbody></table></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--review-policy", choices=("codex", "human"), default="codex"
    )
    parser.add_argument("--human-signoff", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    batch = args.batch_dir.resolve()
    research_manifest = read_json(batch / "research_manifest.json")
    candidates = read_tsv(batch / "candidate_diseases.tsv")
    source_profiles = read_tsv(batch / "human_disease_profiles.source.tsv")
    profile_by_disease: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_profiles:
        profile_by_disease[row["disease_id"]].append(row)
    source_models = read_tsv(batch / "candidate_models.tsv")
    source_model_by_id = {row["model_id"]: row for row in source_models}
    source_phenotypes = read_tsv(batch / "evaluation_inputs/model_phenotypes.tsv")
    signoff = {}
    if args.human_signoff:
        signoff = {
            row["disease_id"]: row for row in read_tsv(args.human_signoff)
        }

    status_rows = []
    valid_reviews = []
    frozen_profiles = []
    frozen_gold = []
    evidence_rows = []
    accepted_candidate_model_ids: set[str] = set()
    signoff_todo = []
    pending_count = 0
    for entry in research_manifest["packets"]:
        rank = int(entry["selection_rank"])
        disease_id = entry["disease_id"]
        packet = read_json(root / entry["packet_path"])
        slug = f"{rank:02d}_{disease_id.replace(':', '_')}"
        final_path = batch / "raw_codex_outputs" / slug / "final.json"
        errors = packet_independence_errors(packet)
        core_ids: list[str] = []
        accepted_gold: list[str] = []
        review: dict[str, Any] | None = None
        if not final_path.is_file():
            errors.append("missing_codex_review")
            pending_count += 1
        else:
            try:
                review = read_json(final_path)
                errors, core_ids, accepted_gold = validate(review, packet)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid_codex_json:{exc}")
        if args.review_policy == "human":
            status = signoff.get(disease_id, {}).get("status", "pending").lower()
            if status != "approved":
                errors.append("human_signoff_pending")
        valid = not errors
        status_rows.append(
            {
                "selection_rank": rank,
                "disease_id": disease_id,
                "disease_name": entry["disease_name"],
                "gene_symbol": entry["gene_symbol"],
                "core_hpo_count": len(core_ids),
                "accepted_gold_model_count": len(accepted_gold),
                "valid": valid,
                "errors": ";".join(sorted(set(errors))),
            }
        )
        signoff_todo.append(
            {
                "selection_rank": rank,
                "disease_id": disease_id,
                "disease_name": entry["disease_name"],
                "gene_symbol": entry["gene_symbol"],
                "status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
        if not valid or review is None:
            continue
        valid_reviews.append(review)
        excluded = set(review["excluded_hpo_ids"])
        core = set(core_ids)
        source_ids = [source["source_id"] for source in review["sources"]]
        for source in review["sources"]:
            evidence_rows.append(
                {
                    "selection_rank": rank,
                    "disease_id": disease_id,
                    "gene_symbol": entry["gene_symbol"],
                    **source,
                }
            )
        for source_row in profile_by_disease[disease_id]:
            if source_row["hpo_id"] in excluded:
                continue
            frozen_profiles.append(
                {
                    **source_row,
                    "is_core": source_row["hpo_id"] in core,
                    "weight": max(float(source_row["weight"]), 1.5)
                    if source_row["hpo_id"] in core
                    else source_row["weight"],
                    "inheritance": review["inheritance"],
                    "molecular_mechanism": review["mechanism_class"],
                    "reference": ";".join(source_ids),
                }
            )
        for model_id in accepted_gold:
            frozen_gold.append(
                {
                    "selection_rank": rank,
                    "disease_id": disease_id,
                    "disease_name": entry["disease_name"],
                    "gene_symbol": entry["gene_symbol"],
                    **source_model_by_id[model_id],
                    "gold_status": "codex_validated"
                    if args.review_policy == "codex"
                    else "human_validated",
                }
            )
        gene = entry["gene_symbol"].upper()
        accepted_candidate_model_ids.update(
            model["model_id"]
            for model in source_models
            if gene in {token.upper() for token in split_values(model["human_gene_symbol"])}
        )

    all_valid = len(valid_reviews) == len(candidates) and not any(
        not row["valid"] for row in status_rows
    )
    frozen = all_valid
    if not frozen and not args.allow_incomplete:
        missing = len(candidates) - len(valid_reviews)
        raise ValueError(
            f"Benchmark cannot freeze: {missing} of {len(candidates)} disease reviews are not valid. Use --allow-incomplete to write an audit report."
        )
    aggregate = batch / "curation"
    write_tsv(
        aggregate / "review_status.tsv",
        status_rows,
        [
            "selection_rank",
            "disease_id",
            "disease_name",
            "gene_symbol",
            "core_hpo_count",
            "accepted_gold_model_count",
            "valid",
            "errors",
        ],
    )
    write_tsv(
        aggregate / "human_signoff.todo.tsv",
        signoff_todo,
        [
            "selection_rank",
            "disease_id",
            "disease_name",
            "gene_symbol",
            "status",
            "reviewer",
            "reviewed_at",
            "notes",
        ],
    )
    write_jsonl(aggregate / "validated_reviews.jsonl", valid_reviews)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_policy": args.review_policy,
        "candidate_count": len(candidates),
        "completed_review_count": len(candidates) - pending_count,
        "valid_review_count": len(valid_reviews),
        "invalid_or_pending_count": len(candidates) - len(valid_reviews),
        "frozen": frozen,
        "middle_layer_independent": True,
        "error_counts": dict(
            sorted(
                Counter(
                    error
                    for row in status_rows
                    for error in row["errors"].split(";")
                    if error
                ).items()
            )
        ),
    }
    write_json(aggregate / "validation_summary.json", summary)
    render_report(aggregate / "index.html", status_rows, summary)

    if frozen:
        frozen_dir = batch / "frozen"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        frozen_models = [
            model
            for model in source_models
            if model["model_id"] in accepted_candidate_model_ids
        ]
        frozen_model_ids = {model["model_id"] for model in frozen_models}
        frozen_phenotypes = [
            row for row in source_phenotypes if row["model_id"] in frozen_model_ids
        ]
        write_tsv(frozen_dir / "diseases.tsv", frozen_profiles, DISEASE_FIELDS)
        write_tsv(
            frozen_dir / "gold_models.tsv",
            frozen_gold,
            list(frozen_gold[0]),
        )
        write_tsv(frozen_dir / "models.tsv", frozen_models, list(frozen_models[0]))
        write_tsv(
            frozen_dir / "model_phenotypes.tsv",
            frozen_phenotypes,
            list(frozen_phenotypes[0]),
        )
        write_tsv(
            frozen_dir / "evidence_sources.tsv",
            evidence_rows,
            [
                "selection_rank",
                "disease_id",
                "gene_symbol",
                "source_id",
                "title",
                "url",
                "source_type",
                "supports",
            ],
        )
        freeze_manifest = {
            "schema_version": "1.0",
            "dataset": "frozen independent rare-disease mouse-model benchmark",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "frozen": True,
            "review_policy": args.review_policy,
            "middle_layer_independent": True,
            "disease_count": len(candidates),
            "gold_model_count": len(frozen_gold),
            "candidate_model_count": len(frozen_models),
            "inputs": {
                "selection_manifest": input_record(batch / "selection_manifest.json", root),
                "research_manifest": input_record(batch / "research_manifest.json", root),
                "validated_reviews": input_record(
                    aggregate / "validated_reviews.jsonl", root
                ),
            },
            "files": {
                path.name: input_record(path, root)
                for path in sorted(frozen_dir.glob("*.tsv"))
            },
            "evaluation_rule": "Load cellular mechanism data only after this manifest is frozen.",
        }
        write_json(frozen_dir / "freeze_manifest.json", freeze_manifest)
        summary["freeze_manifest"] = str(
            (frozen_dir / "freeze_manifest.json").relative_to(root)
        )
        write_json(aggregate / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
