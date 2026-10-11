#!/usr/bin/env python3
"""Validate Codex cohort reviews and apply non-negotiable benchmark gates."""

from __future__ import annotations

from uhpo_common.paths import REPO_ROOT

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
MISSION3 = ROOT / str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization')
DEFAULT_BATCH = MISSION3 / "data" / "curation" / "formal_cohort_v1"

COHORT_FIELDS = [
    "selection_rank",
    "disease_id",
    "disease_name",
    "gene_symbol",
    "codex_decision",
    "inheritance",
    "mechanism_class",
    "mechanism_confidence",
    "core_hpo_count",
    "core_hpo_ids",
    "valid_accepted_cpo_claim_count",
    "core_hpo_with_valid_cpo_count",
    "comparable_core_hpo_with_valid_cpo_count",
    "candidate_model_count",
    "known_positive_model_count",
    "surface_weighted_coverage",
    "candidate_mp_cpo_coverage",
    "disease_level_ready",
    "codex_cpo_ready",
    "automated_gate_pass",
    "human_signoff",
    "pilot_eligible",
    "blockers",
    "mechanism_summary",
]

ADJUDICATION_FIELDS = [
    "selection_rank",
    "disease_id",
    "gene_symbol",
    "hpo_id",
    "claim_id",
    "codex_decision",
    "valid_acceptance",
    "machine_structural_candidate",
    "selected_cl_id",
    "selected_cl_label",
    "selected_go_id",
    "selected_go_label",
    "direction",
    "relation_type",
    "bridge_scope",
    "species",
    "supporting_record_ids",
    "source_urls",
    "evidence_quote",
    "codex_reason",
    "human_review_status",
]

PENDING_EDGE_FIELDS = [
    "hpo_id",
    "hpo_label",
    "selected_cl_id",
    "selected_cl_label",
    "selected_go_id",
    "selected_go_label",
    "cell_text",
    "process_text",
    "quality_text",
    "direction",
    "relation_type",
    "bridge_scope",
    "evidence_type",
    "species",
    "disease_ids",
    "claim_id",
    "evidence_span_ids",
    "supporting_record_ids",
    "source_evidence_run_id",
    "candidate_status",
    "codex_decision",
    "human_review_status",
    "codex_reason",
]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ";".join(str(item) for item in value)
                    if isinstance((value := row.get(field, "")), list)
                    else str(value).lower()
                    if isinstance(value, bool)
                    else value
                    for field in fields
                }
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate_review(
    review: dict[str, Any], packet: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    errors: list[str] = []
    for key in ("disease_id", "disease_name", "gene_symbol"):
        if str(review.get(key, "")) != str(packet.get(key, "")):
            errors.append(f"{key}_does_not_match_packet")
    allowed_hpo = set(packet["allowed_hpo_ids"])
    core = review.get("core_hpo_ids", [])
    excluded = review.get("excluded_hpo_ids", [])
    if not isinstance(core, list) or not set(core).issubset(allowed_hpo):
        errors.append("core_hpo_not_subset_of_packet")
    if not isinstance(excluded, list) or not set(excluded).issubset(allowed_hpo):
        errors.append("excluded_hpo_not_subset_of_packet")
    if set(core).intersection(excluded):
        errors.append("core_and_excluded_hpo_overlap")

    local_claims = {claim["claim_id"]: claim for claim in packet["cpo_claims"]}
    decisions = review.get("cpo_claim_decisions", [])
    if not isinstance(decisions, list):
        decisions = []
        errors.append("cpo_claim_decisions_not_array")
    decision_ids = [str(row.get("claim_id", "")) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        errors.append("duplicate_cpo_claim_decision")
    if set(decision_ids) != set(local_claims):
        errors.append("cpo_claim_decisions_do_not_cover_packet_exactly")
    valid_decisions = [
        row
        for row in decisions
        if row.get("claim_id") in local_claims
        and row.get("hpo_id") == local_claims[row["claim_id"]].get("hpo_id")
    ]
    if len(valid_decisions) != len(decisions):
        errors.append("claim_hpo_mismatch_or_unknown_claim")
    return errors, local_claims, valid_decisions


def render_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    table_rows = []
    for row in rows:
        status = (
            "pass"
            if row["pilot_eligible"]
            else "automatic-pass"
            if row["automated_gate_pass"]
            else "disease-ready"
            if row["disease_level_ready"]
            else "blocked"
        )
        table_rows.append(
            "<tr>"
            f"<td>{row['selection_rank']}</td>"
            f"<td><code>{html.escape(row['disease_id'])}</code><br>{html.escape(row['disease_name'])}</td>"
            f"<td>{html.escape(row['gene_symbol'])}</td>"
            f"<td>{html.escape(row['mechanism_class'])}<br><small>{html.escape(row['mechanism_confidence'])}</small></td>"
            f"<td>{row['core_hpo_count']}</td>"
            f"<td>{row['core_hpo_with_valid_cpo_count']} / {row['comparable_core_hpo_with_valid_cpo_count']}</td>"
            f"<td>{float(row['surface_weighted_coverage']):.1%}</td>"
            f"<td>{float(row['candidate_mp_cpo_coverage']):.1%}</td>"
            f"<td class='{status}'>{status.replace('-', ' ')}</td>"
            f"<td>{html.escape(row['blockers'])}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 3 正式机制队列 v1</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#202124;background:#fff}}header{{background:#17324d;color:#fff;padding:24px 32px}}
main{{padding:24px 32px;overflow:auto}}h1{{font-size:24px;margin:0 0 8px}}p{{margin:7px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:#d9dee3;border:1px solid #d9dee3;margin:20px 0}}
.metric{{background:#fff;padding:14px}}.metric strong{{display:block;font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#f4f6f8}}code{{font-size:12px}}small{{color:#5f6368}}.pass,.automatic-pass{{color:#087f5b;font-weight:700}}.disease-ready{{color:#7a5500;font-weight:700}}.blocked{{color:#a61e2d;font-weight:700}}
</style></head><body><header><h1>Mission 3 正式机制队列 v1</h1><p>Codex 结构化审核 + 程序硬门槛；人工签字与 strict HPO-CPO 导入保持分离。</p></header><main>
<div class="metrics"><div class="metric"><strong>{summary['counts']['packet_count']}</strong>队列疾病</div><div class="metric"><strong>{summary['counts']['completed_review_count']}</strong>Codex 完成</div><div class="metric"><strong>{summary['counts']['disease_level_ready_count']}</strong>疾病层就绪</div><div class="metric"><strong>{summary['counts']['codex_cpo_ready_count']}</strong>CPO 就绪</div><div class="metric"><strong>{summary['counts']['automated_gate_pass_count']}</strong>自动门槛通过</div><div class="metric"><strong>{summary['counts']['pilot_eligible_count']}</strong>正式 pilot</div></div>
<p><strong>pilot_ready = {str(summary['pilot_ready']).lower()}</strong>。Codex 接受的边仍为 <code>review_required</code>，因此在人审前不会进入 strict 数据。</p>
<p>CPO 增益验证状态：<strong>{html.escape(summary['cpo_gain_validation']['status'])}</strong>。{html.escape(summary['cpo_gain_validation']['reason'])}</p>
<table><thead><tr><th>#</th><th>疾病</th><th>基因</th><th>机制</th><th>核心 HPO</th><th>CPO / 可比较</th><th>Surface</th><th>MP-CPO</th><th>状态</th><th>阻塞项</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    aggregate_dir = batch_dir / "aggregate"
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = read_json(manifest_path)
    cohort_rows: list[dict[str, Any]] = []
    adjudication_rows: list[dict[str, Any]] = []
    pending_edges: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    invalid_outputs: list[dict[str, Any]] = []
    targeted_rows: list[dict[str, Any]] = []
    curation_rows: list[dict[str, Any]] = []
    signoff_rows: list[dict[str, Any]] = []

    for entry in manifest["packets"]:
        rank = int(entry["selection_rank"])
        packet = read_json(ROOT / entry["packet_path"])
        slug = f"{rank:02d}_{str(entry['disease_id']).replace(':', '_')}"
        final_path = batch_dir / "codex_outputs" / slug / "final.json"
        if not final_path.is_file():
            invalid_outputs.append(
                {
                    "selection_rank": rank,
                    "disease_id": entry["disease_id"],
                    "errors": "missing_final_json",
                }
            )
            continue
        try:
            review = read_json(final_path)
        except (json.JSONDecodeError, ValueError) as exc:
            invalid_outputs.append(
                {
                    "selection_rank": rank,
                    "disease_id": entry["disease_id"],
                    "errors": f"invalid_json:{exc}",
                }
            )
            continue
        errors, local_claims, decisions = validate_review(review, packet)
        if errors:
            invalid_outputs.append(
                {
                    "selection_rank": rank,
                    "disease_id": entry["disease_id"],
                    "errors": ";".join(errors),
                }
            )
        review_records.append(review)
        core_hpo = set(review.get("core_hpo_ids", []))
        valid_accepted: list[dict[str, Any]] = []
        valid_hpo: set[str] = set()
        comparable_valid_hpo: set[str] = set()
        phenotype_by_hpo = {
            row["hpo_id"]: row for row in packet["phenotype_profile"]
        }
        for decision in decisions:
            claim = local_claims[decision["claim_id"]]
            valid_acceptance = (
                decision.get("decision") == "accept"
                and as_bool(decision.get("quote_supports_cell"))
                and as_bool(decision.get("quote_supports_process"))
                and as_bool(decision.get("direct_hpo"))
                and not as_bool(decision.get("disease_specificity_issue"))
                and as_bool(claim.get("machine_structural_candidate"))
                and claim.get("hpo_id") in core_hpo
            )
            if valid_acceptance:
                valid_accepted.append(claim)
                valid_hpo.add(claim["hpo_id"])
                if phenotype_by_hpo.get(claim["hpo_id"], {}).get(
                    "comparable_with_candidate_model_cpo"
                ):
                    comparable_valid_hpo.add(claim["hpo_id"])
                pending_edges.append(
                    {
                        "hpo_id": claim["hpo_id"],
                        "hpo_label": claim["hpo_label"],
                        "selected_cl_id": claim["selected_cl_id"],
                        "selected_cl_label": claim["selected_cl_label"],
                        "selected_go_id": claim["selected_go_id"],
                        "selected_go_label": claim["selected_go_label"],
                        "cell_text": claim["cell_text"],
                        "process_text": claim["process_text"],
                        "quality_text": claim["quality_text"],
                        "direction": claim["direction"],
                        "relation_type": claim["relation_type"],
                        "bridge_scope": claim["bridge_scope"],
                        "evidence_type": claim["evidence_type"],
                        "species": claim["species"],
                        "disease_ids": claim["disease_ids"],
                        "claim_id": claim["claim_id"],
                        "evidence_span_ids": [
                            span["evidence_span_id"] for span in claim["evidence_spans"]
                        ],
                        "supporting_record_ids": claim["supporting_record_ids"],
                        "source_evidence_run_id": claim["source_evidence_run_id"],
                        "candidate_status": "review_required",
                        "codex_decision": "accept",
                        "human_review_status": "pending",
                        "codex_reason": decision.get("reason", ""),
                    }
                )
            adjudication_rows.append(
                {
                    "selection_rank": rank,
                    "disease_id": packet["disease_id"],
                    "gene_symbol": packet["gene_symbol"],
                    "hpo_id": claim["hpo_id"],
                    "claim_id": claim["claim_id"],
                    "codex_decision": decision.get("decision", ""),
                    "valid_acceptance": valid_acceptance,
                    "machine_structural_candidate": claim["machine_structural_candidate"],
                    "selected_cl_id": claim["selected_cl_id"],
                    "selected_cl_label": claim["selected_cl_label"],
                    "selected_go_id": claim["selected_go_id"],
                    "selected_go_label": claim["selected_go_label"],
                    "direction": claim["direction"],
                    "relation_type": claim["relation_type"],
                    "bridge_scope": claim["bridge_scope"],
                    "species": claim["species"],
                    "supporting_record_ids": claim["supporting_record_ids"],
                    "source_urls": claim["source_urls"],
                    "evidence_quote": " | ".join(
                        span["quote"] for span in claim["evidence_spans"]
                    ),
                    "codex_reason": decision.get("reason", ""),
                    "human_review_status": "pending",
                }
            )

        source_urls = [
            source.get("url", "")
            for source in review.get("sources", [])
            if source.get("url", "").startswith(("https://", "http://"))
        ]
        disease_level_ready = (
            not errors
            and review.get("decision") == "include"
            and len(core_hpo) >= 3
            and review.get("mechanism_class") != "unknown"
            and review.get("mechanism_confidence") in {"high", "medium"}
            and bool(source_urls)
        )
        surface_ready = as_bool(packet["pre_curation_metrics"]["surface_ready"])
        mp_cpo_coverage = float(
            packet["mouse_model_summary"]["candidate_mp_cpo_coverage"]
        )
        codex_cpo_ready = (
            len(valid_hpo) >= 2
            and len(comparable_valid_hpo) >= 2
            and mp_cpo_coverage >= 0.30
        )
        automated_gate_pass = disease_level_ready and surface_ready and codex_cpo_ready
        blockers: list[str] = []
        if errors:
            blockers.append("invalid_codex_output")
        if not disease_level_ready:
            blockers.append("disease_level_review_not_ready")
        if not surface_ready:
            blockers.append("surface_gate_failed")
        if len(valid_hpo) < 2:
            blockers.append("accepted_core_hpo_cpo<2")
        if len(comparable_valid_hpo) < 2:
            blockers.append("comparable_core_hpo_cpo<2")
        if mp_cpo_coverage < 0.30:
            blockers.append("mp_cpo_coverage<0.30")
        if automated_gate_pass:
            blockers.append("human_signoff_pending")
        human_signoff = False
        pilot_eligible = automated_gate_pass and human_signoff
        cohort_rows.append(
            {
                "selection_rank": rank,
                "disease_id": packet["disease_id"],
                "disease_name": packet["disease_name"],
                "gene_symbol": packet["gene_symbol"],
                "codex_decision": review.get("decision", ""),
                "inheritance": review.get("inheritance", ""),
                "mechanism_class": review.get("mechanism_class", "unknown"),
                "mechanism_confidence": review.get("mechanism_confidence", "insufficient"),
                "core_hpo_count": len(core_hpo),
                "core_hpo_ids": sorted(core_hpo),
                "valid_accepted_cpo_claim_count": len(valid_accepted),
                "core_hpo_with_valid_cpo_count": len(valid_hpo),
                "comparable_core_hpo_with_valid_cpo_count": len(comparable_valid_hpo),
                "candidate_model_count": packet["mouse_model_summary"]["candidate_model_count"],
                "known_positive_model_count": packet["mouse_model_summary"]["known_positive_model_count"],
                "surface_weighted_coverage": packet["pre_curation_metrics"]["surface_weighted_coverage"],
                "candidate_mp_cpo_coverage": mp_cpo_coverage,
                "disease_level_ready": disease_level_ready,
                "codex_cpo_ready": codex_cpo_ready,
                "automated_gate_pass": automated_gate_pass,
                "human_signoff": human_signoff,
                "pilot_eligible": pilot_eligible,
                "blockers": ";".join(blockers),
                "mechanism_summary": review.get("molecular_mechanism_summary", ""),
            }
        )
        curation_rows.append(
            {
                "disease_id": packet["disease_id"],
                "gene_symbol": packet["gene_symbol"],
                "inheritance": review.get("inheritance", ""),
                "molecular_mechanism": review.get("mechanism_class", "unknown"),
                "core_hpo_ids": sorted(core_hpo),
                "exclude_hpo_ids": review.get("excluded_hpo_ids", []),
                "hpo_directions": [
                    f"{item.get('hpo_id')}={item.get('direction')}"
                    for item in review.get("hpo_directions", [])
                ],
                "include": False,
                "notes": "Codex-reviewed; human disease and CPO sign-off pending.",
            }
        )
        for hpo_id in sorted(core_hpo):
            if hpo_id not in valid_hpo:
                phenotype = phenotype_by_hpo.get(hpo_id, {})
                targeted_rows.append(
                    {
                        "selection_rank": rank,
                        "disease_id": packet["disease_id"],
                        "disease_name": packet["disease_name"],
                        "gene_symbol": packet["gene_symbol"],
                        "hpo_id": hpo_id,
                        "hpo_name": phenotype.get("hpo_name", ""),
                        "surface_mapped_candidate_mp_count": phenotype.get(
                            "surface_mapped_candidate_mp_count", 0
                        ),
                        "comparable_with_candidate_model_cpo": phenotype.get(
                            "comparable_with_candidate_model_cpo", False
                        ),
                        "local_claim_count": sum(
                            claim["hpo_id"] == hpo_id for claim in packet["cpo_claims"]
                        ),
                        "required_action": "targeted_literature_curation_or_claim_remapping",
                    }
                )
        signoff_rows.append(
            {
                "selection_rank": rank,
                "disease_id": packet["disease_id"],
                "disease_name": packet["disease_name"],
                "gene_symbol": packet["gene_symbol"],
                "disease_review_status": "pending",
                "core_hpo_review_status": "pending",
                "genetic_mechanism_review_status": "pending",
                "cpo_claim_review_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )

    cohort_rows.sort(key=lambda row: int(row["selection_rank"]))
    adjudication_rows.sort(
        key=lambda row: (int(row["selection_rank"]), row["hpo_id"], row["claim_id"])
    )
    pending_edge_by_claim = {row["claim_id"]: row for row in pending_edges}
    pending_edges = [pending_edge_by_claim[key] for key in sorted(pending_edge_by_claim)]
    write_jsonl(aggregate_dir / "codex_disease_decisions.jsonl", review_records)
    write_tsv(aggregate_dir / "formal_cohort.tsv", cohort_rows, COHORT_FIELDS)
    write_tsv(
        aggregate_dir / "disease_curation.codex.tsv",
        curation_rows,
        [
            "disease_id",
            "gene_symbol",
            "inheritance",
            "molecular_mechanism",
            "core_hpo_ids",
            "exclude_hpo_ids",
            "hpo_directions",
            "include",
            "notes",
        ],
    )
    write_tsv(
        aggregate_dir / "hpo_cpo_adjudications.tsv",
        adjudication_rows,
        ADJUDICATION_FIELDS,
    )
    write_tsv(
        aggregate_dir / "hpo_cpo_codex_accepted.pending_human.tsv",
        pending_edges,
        PENDING_EDGE_FIELDS,
    )
    write_tsv(
        aggregate_dir / "targeted_hpo_cpo_queue.tsv",
        targeted_rows,
        [
            "selection_rank",
            "disease_id",
            "disease_name",
            "gene_symbol",
            "hpo_id",
            "hpo_name",
            "surface_mapped_candidate_mp_count",
            "comparable_with_candidate_model_cpo",
            "local_claim_count",
            "required_action",
        ],
    )
    write_tsv(
        aggregate_dir / "human_signoff.todo.tsv",
        signoff_rows,
        [
            "selection_rank",
            "disease_id",
            "disease_name",
            "gene_symbol",
            "disease_review_status",
            "core_hpo_review_status",
            "genetic_mechanism_review_status",
            "cpo_claim_review_status",
            "reviewer",
            "reviewed_at",
            "notes",
        ],
    )
    write_tsv(
        aggregate_dir / "invalid_outputs.tsv",
        invalid_outputs,
        ["selection_rank", "disease_id", "errors"],
    )

    completed = len(cohort_rows)
    automated_pass = sum(row["automated_gate_pass"] for row in cohort_rows)
    pilot_count = sum(row["pilot_eligible"] for row in cohort_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": manifest["batch_id"],
        "pilot_ready": pilot_count >= 30,
        "counts": {
            "packet_count": len(manifest["packets"]),
            "completed_review_count": completed,
            "invalid_output_count": len(invalid_outputs),
            "codex_include_count": sum(
                row["codex_decision"] == "include" for row in cohort_rows
            ),
            "disease_level_ready_count": sum(
                row["disease_level_ready"] for row in cohort_rows
            ),
            "codex_cpo_ready_count": sum(row["codex_cpo_ready"] for row in cohort_rows),
            "automated_gate_pass_count": automated_pass,
            "pilot_eligible_count": pilot_count,
            "valid_codex_accepted_cpo_claim_count": len(pending_edges),
            "targeted_hpo_cpo_gap_count": len(targeted_rows),
        },
        "mechanism_class_counts": dict(
            sorted(Counter(row["mechanism_class"] for row in cohort_rows).items())
        ),
        "cpo_gain_validation": {
            "status": "ready_to_run" if pilot_count >= 30 else "not_evaluable",
            "reason": (
                "At least 30 human-signed-off diseases pass all gates."
                if pilot_count >= 30
                else "Fewer than 30 diseases pass all mechanism gates and human sign-off; a ranking delta would not be a valid cohort-level CPO evaluation."
            ),
            "required_comparisons": [
                "surface-only baseline",
                "surface plus Codex/human-reviewed HPO-CPO",
                "full surface plus CPO plus genetic model",
            ],
            "required_metrics": ["Recall@5", "Recall@10", "MRR", "NDCG", "rank_delta"],
        },
        "blocker_counts": dict(
            sorted(
                Counter(
                    blocker
                    for row in cohort_rows
                    for blocker in row["blockers"].split(";")
                    if blocker
                ).items()
            )
        ),
        "interpretation": [
            "Codex disease review is auditable evidence, not human sign-off.",
            "Codex-accepted HPO-CPO claims remain candidate_status=review_required.",
            "pilot_ready requires at least 30 diseases to pass every automated gate and human sign-off.",
            "CPO gain is not declared when the formal evaluation cohort is smaller than 30.",
        ],
    }
    write_json(aggregate_dir / "summary.json", summary)
    render_report(aggregate_dir / "index.html", cohort_rows, summary)
    output_files = [path for path in aggregate_dir.iterdir() if path.is_file()]
    summary["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_files)
    }
    write_json(aggregate_dir / "summary.json", summary)
    manifest["status"] = "aggregated"
    manifest["pilot_ready"] = summary["pilot_ready"]
    manifest["aggregate_summary"] = {
        "path": str((aggregate_dir / "summary.json").relative_to(ROOT)),
        "counts": summary["counts"],
        "cpo_gain_validation_status": summary["cpo_gain_validation"]["status"],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(f"pilot_ready={str(summary['pilot_ready']).lower()}")


if __name__ == "__main__":
    main()
