#!/usr/bin/env python3
"""Build a deterministic, stratified audit sample for cell-name normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import match_cell_names_to_catalog as matching  # noqa: E402


DATABASE = ROOT / "database"
DEFAULT_MAPPING = DATABASE / "cell_name_catalog_matches.tsv"
DEFAULT_EXCLUDED = DATABASE / "agent_standardized_cell_names_excluded.tsv"
DEFAULT_SAMPLE = DATABASE / "cell_name_mapping_audit_sample.tsv"
DEFAULT_SUMMARY = DATABASE / "cell_name_mapping_audit_summary.json"
DEFAULT_REPORT = DATABASE / "cell_name_mapping_error_analysis.md"

DIRECT_MATCH_REASON_RE = re.compile(
    r"\b(?:directly (?:matches|corresponds)|exact match|is equivalent to|matches the catalog)\b",
    re.I,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def mapping_risk_flags(row: dict[str, str]) -> list[str]:
    decision = str(row.get("decision") or "").strip().lower()
    reason = str(row.get("reason") or "").strip()
    flags = [
        flag.strip()
        for flag in str(row.get("validation_flags") or "").split(";")
        if flag.strip()
    ]
    flags.extend(matching.semantic_safety_flags(
        str(row.get("cell_name") or ""), decision, reason, row.get("top_score")
    ))
    try:
        score = float(row.get("top_score") or 0)
    except ValueError:
        score = 0.0
    if decision == "review" and DIRECT_MATCH_REASON_RE.search(reason):
        flags.append("review_reason_says_match")
    if decision == "no_match" and score >= 0.8:
        flags.append("high_similarity_no_match")
    if "LLM error:" in reason or "LLM omitted" in reason:
        flags.append("llm_output_failure")
    return list(dict.fromkeys(flags))


def mapping_stratum(row: dict[str, str], flags: list[str]) -> str:
    decision = str(row.get("decision") or "").strip().lower()
    confidence = str(row.get("confidence") or "").strip().lower()
    try:
        score = float(row.get("top_score") or 0)
    except ValueError:
        score = 0.0
    try:
        paper_count = int(row.get("paper_count") or 0)
    except ValueError:
        paper_count = 0
    if "llm_output_failure" in flags:
        return "mapping_llm_failure"
    if decision == "match" and flags:
        return "mapping_match_risky"
    if decision == "exact_auto":
        return "mapping_exact_auto"
    if decision == "match" and paper_count >= 3:
        return "mapping_match_high_evidence"
    if decision == "match" and confidence == "high":
        return "mapping_match_high_unflagged"
    if decision == "match":
        return "mapping_match_medium_low_unflagged"
    if decision == "review" and (score >= 0.75 or bool(flags)):
        return "mapping_review_high_signal"
    if decision == "review":
        return "mapping_review_other"
    if decision == "no_match" and score >= 0.7:
        return "mapping_no_match_high_signal"
    if decision == "no_match" and paper_count >= 3:
        return "mapping_no_match_high_evidence"
    return "mapping_no_match_other"


def excluded_stratum(row: dict[str, str]) -> str:
    reason = str(row.get("exclusion_reason") or "").strip()
    if reason == "gene_symbol_in_cell_cluster":
        try:
            papers = int(row.get("paper_count") or 0)
        except ValueError:
            papers = 0
        if papers >= 3 and row.get("source_types") == "both":
            return "filter_gene_symbol_high_evidence"
        return "filter_gene_symbol_collision"
    if reason == "high_cardinality_source":
        return "filter_high_cardinality_source"
    if reason in {"numeric_value", "cluster_or_sample_code"}:
        return "filter_numeric_or_cluster"
    return "filter_other_reason"


def stable_rank(seed: str, stratum: str, name: str, index: str) -> str:
    payload = f"{seed}\0{stratum}\0{name}\0{index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_sample(
    rows: list[dict[str, str]], quota: int, seed: str
) -> tuple[list[dict[str, str]], Counter[str]]:
    populations = Counter(row["stratum"] for row in rows)
    selected: list[dict[str, str]] = []
    by_stratum: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], []).append(row)
    for stratum in sorted(by_stratum):
        candidates = sorted(
            by_stratum[stratum],
            key=lambda row: stable_rank(
                seed,
                stratum,
                row.get("cell_name", ""),
                row.get("source_index", row.get("original_rank", "")),
            ),
        )
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: (row["stratum"], row.get("cell_name", "").casefold()))
    return selected, populations


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "audit_id",
        "stage",
        "stratum",
        "risk_flags",
        "cell_name",
        "current_decision",
        "current_confidence",
        "current_catalog_id",
        "current_catalog_label",
        "top_score",
        "paper_count",
        "source_types",
        "current_reason",
        "candidate_ids",
        "candidate_labels",
        "exclusion_reason",
        "audit_decision",
        "corrected_catalog_id",
        "error_category",
        "audit_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "audit_id": f"A{number:04d}",
                    "stage": row.get("stage", ""),
                    "stratum": row.get("stratum", ""),
                    "risk_flags": row.get("risk_flags", ""),
                    "cell_name": row.get("cell_name", ""),
                    "current_decision": row.get("decision", ""),
                    "current_confidence": row.get("confidence", ""),
                    "current_catalog_id": row.get("catalog_id", ""),
                    "current_catalog_label": row.get("catalog_label", ""),
                    "top_score": row.get("top_score", ""),
                    "paper_count": row.get("paper_count", ""),
                    "source_types": row.get("source_types", ""),
                    "current_reason": row.get("reason", ""),
                    "candidate_ids": row.get("candidate_ids", ""),
                    "candidate_labels": row.get("candidate_labels", ""),
                    "exclusion_reason": row.get("exclusion_reason", ""),
                    "audit_decision": "",
                    "corrected_catalog_id": "",
                    "error_category": "",
                    "audit_notes": "",
                }
            )
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def build_report(summary: dict[str, Any]) -> str:
    decisions = summary["mapping_decision_counts"]
    risks = summary["mapping_risk_flag_counts"]
    validation = summary["mapping_validation_flag_counts"]
    residual_risks = {key: value for key, value in risks.items() if key not in validation}
    lines = [
        "# Cell-name normalization error analysis",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        f"- Mapping rows: {summary['mapping_rows']:,}",
        f"- Excluded-name rows: {summary['excluded_rows']:,}",
        f"- Deterministic audit sample: {summary['sample_rows']:,}",
        f"- Seed: `{summary['seed']}`; quota per stratum: {summary['quota_per_stratum']}",
        "",
        "## Current mapping decisions",
        "",
    ]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(decisions.items()))
    lines.extend(["", "## Safety guard outcomes", ""])
    lines.append(
        f"- Flagged rows routed to `review`: {summary['mapping_validation_flagged_rows']:,}"
    )
    if validation:
        lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(validation.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "These counts describe risky matches intercepted by strict validation; one row can have",
            "multiple flags. The proposed catalog ID is retained only as an audit aid.",
            "",
            "## Additional review-priority signals",
            "",
        ]
    )
    if residual_risks:
        lines.extend(
            f"- `{key}`: {value:,}" for key, value in sorted(residual_risks.items())
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "These signals are not ground-truth error counts. The audit TSV",
            "contains blank adjudication columns for independent manual review.",
            "",
            "## Baseline findings (2026-08-10)",
            "",
            "The pre-fix mapping contained 26,122 rows: 829 `exact_auto`, 11,707 `match`,",
            "2,401 `review`, and 11,185 `no_match`. Automated inspection found 2,165 matches",
            "whose own explanation admitted a broad-parent or nearest-term mapping, 216 matches",
            "with Top-1 similarity below 0.50, 153 mixed-population matches, 376 review rows",
            "whose explanation asserted a direct match, 21 failed/omitted LLM outputs, and one",
            "confirmed cross-item explanation mismatch.",
            "",
            "The filter also excluded 2,347 compact labels because they collided with observed gene",
            "tokens. Catalog-aware protection restored 99 names, including `Endothelial`,",
            "`Macrophages`, `Pericytes`, `Monocytes`, `gdT`, `MSC`, and `Rod`, while `MKI67`,",
            "`S100A6`, and `POLE` remained excluded.",
            "",
            "Representative errors:",
            "",
            "| Source | Baseline result | Root cause | Resolution |",
            "|---|---|---|---|",
            "| `mMac1` | meningeal macrophage | low-score abbreviation inference | low-score/plausibility guard -> review |",
            "| `HSC/MPP` | hematopoietic stem cell | mixed population collapsed to one node | mixed-population guard -> review |",
            "| `Microglia/macrophage` | microglial cell | mixed population collapsed to one node | mixed-population guard -> review |",
            "| `CAF` | stromal cells | specific concept mapped to broad parent | generalization guard -> review |",
            "| `CD4 CM T cells` | review despite direct-match reason | correct node absent from embedding Top-10 | lexical alias insertion |",
            "| `Gallbladder_Goblet cells` | reason discussed macrophages | batch item misalignment | mandatory echoed source-name validation |",
            "| `Endothelial` | filtered as a gene token | gene/cell vocabulary collision | catalog-aware cell-label protection |",
            "",
            "## Implemented controls",
            "",
            "- Hyphens in terms such as `CD8-positive` and `alpha-beta` are no longer interpreted as negative signs.",
            "- Exact catalog aliases are inserted before embedding candidates; conservative abbreviations are expanded.",
            "- Mixed populations, low-similarity matches, admitted generalizations, contradictory reasons, and item mismatches are changed to `review`.",
            "- LLM results must echo `source_name`; missing or misaligned batches are retried once before failing.",
            "- Reuse ignores unsafe historical decisions and strips repeated `reused prior decision:` prefixes.",
            "",
            "## Root-cause taxonomy",
            "",
            "- `filter_false_exclusion`: a real cell label collides with a gene token or broad source rule.",
            "- `candidate_recall_failure`: the correct ontology node is absent from the supplied Top-K set.",
            "- `semantic_overmapping`: a mixed, more specific, or ambiguous source is forced onto one node.",
            "- `semantic_false_negative`: a supported ontology node is present but rejected or left in review.",
            "- `output_alignment_failure`: a batch omits an item or returns reasoning for another source name.",
            "- `correct`: the current decision and granularity satisfy the strict equivalence policy.",
            "",
            "## Review protocol",
            "",
            "For each sampled row, fill `audit_decision` with `correct`, `wrong_match`,",
            "`missed_match`, `should_review`, or `wrong_filter`. Add the corrected CL ID only when",
            "the source and catalog concepts are semantically equivalent; do not collapse mixed",
            "populations or specific states to a broad parent.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--quota", type=int, default=25)
    parser.add_argument("--seed", default="cell-name-audit-v1")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.quota < 1:
        raise SystemExit("--quota must be positive")

    mapping_rows = read_tsv(args.mapping.resolve())
    excluded_rows = read_tsv(args.excluded.resolve())
    audit_pool: list[dict[str, str]] = []
    risk_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    validation_flagged_rows = 0
    for row in mapping_rows:
        flags = mapping_risk_flags(row)
        risk_counts.update(flags)
        persisted_flags = [
            flag.strip()
            for flag in str(row.get("validation_flags") or "").split(";")
            if flag.strip()
        ]
        validation_counts.update(persisted_flags)
        validation_flagged_rows += int(bool(persisted_flags))
        item = dict(row)
        item.update(
            {
                "stage": "mapping",
                "stratum": mapping_stratum(row, flags),
                "risk_flags": ";".join(flags),
            }
        )
        audit_pool.append(item)
    for row in excluded_rows:
        item = dict(row)
        item.update(
            {
                "stage": "filter",
                "stratum": excluded_stratum(row),
                "risk_flags": "",
                "decision": "excluded",
                "reason": row.get("exclusion_reason", ""),
            }
        )
        audit_pool.append(item)

    sample, populations = select_sample(audit_pool, args.quota, args.seed)
    write_sample(args.sample.resolve(), sample)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "quota_per_stratum": args.quota,
        "mapping_rows": len(mapping_rows),
        "excluded_rows": len(excluded_rows),
        "sample_rows": len(sample),
        "mapping_decision_counts": dict(
            sorted(Counter(row.get("decision", "") for row in mapping_rows).items())
        ),
        "mapping_risk_flag_counts": dict(sorted(risk_counts.items())),
        "mapping_validation_flag_counts": dict(sorted(validation_counts.items())),
        "mapping_validation_flagged_rows": validation_flagged_rows,
        "population_stratum_counts": dict(sorted(populations.items())),
        "sample_stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in sample).items())
        ),
        "mapping": str(args.mapping.resolve()),
        "excluded": str(args.excluded.resolve()),
        "sample": str(args.sample.resolve()),
    }
    atomic_write(
        args.summary.resolve(),
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(args.report.resolve(), build_report(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
