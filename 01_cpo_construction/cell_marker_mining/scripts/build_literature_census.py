#!/usr/bin/env python3
"""Build a reproducible literature and supplementary-material census."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "database"
DEFAULT_JSON = DEFAULT_DATABASE / "literature_census_all_cells.json"
DEFAULT_MARKDOWN = DEFAULT_DATABASE / "literature_census_all_cells.md"
DIRECT_TABLE_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv"}
BROAD_TABLE_EXTENSIONS = DIRECT_TABLE_EXTENSIONS | {".docx", ".txt", ".zip"}


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def asset_extension(asset: dict[str, Any]) -> str:
    value = str(asset.get("filename") or asset.get("url") or "")
    return Path(value.split("?", 1)[0]).suffix.lower()


def record_flags(record_dir: Path, metadata: dict[str, Any]) -> dict[str, int]:
    manifest_assets = read_json(
        record_dir / "assets" / "assets_manifest.json", default=None
    )
    assets = (
        manifest_assets
        if isinstance(manifest_assets, list)
        else metadata.get("assets", [])
    )
    direct_assets = [a for a in assets if asset_extension(a) in DIRECT_TABLE_EXTENSIONS]
    broad_assets = [a for a in assets if asset_extension(a) in BROAD_TABLE_EXTENSIONS]
    probe = metadata.get("content_probe", {})
    probe_errors = probe.get("errors", []) if isinstance(probe, dict) else []
    elsevier_sync = metadata.get("elsevier_content_sync", {})
    tables = read_json(record_dir / "extracted" / "tables" / "tables_manifest.json", [])
    agent = read_json(record_dir / "agent_standardized" / "agent_summary.json", {})
    marker_filter = read_json(
        record_dir / "agent_standardized" / "marker_statistics_filter_summary.json", {}
    )
    downloaded_direct = any(
        asset.get("local_path")
        and (record_dir / str(asset["local_path"])).is_file()
        for asset in direct_assets
    )
    direct_downloaded_flags = [
        bool(
            asset.get("download_status") in {"downloaded", "exists"}
            and asset.get("local_path")
            and (record_dir / str(asset["local_path"])).is_file()
        )
        for asset in direct_assets
    ]
    direct_failed_flags = [
        asset.get("download_status")
        in {"failed", "invalid_content", "blocked_challenge"}
        for asset in direct_assets
    ]
    downloaded_broad = any(
        asset.get("local_path")
        and (record_dir / str(asset["local_path"])).is_file()
        for asset in broad_assets
    )
    marker_positive = (
        int(agent.get("statistics_rows", 0) or 0) > 0
        or int(agent.get("signature_cell_gene_pairs", 0) or 0) > 0
    )
    post_filter_positive = (
        int(marker_filter.get("kept_rows", 0) or 0) > 0
        or int(agent.get("signature_cell_gene_pairs", 0) or 0) > 0
    )
    return {
        "candidate_papers": 1,
        "probe_completed_papers": int(bool(probe)),
        "probe_clean_papers": int(bool(probe) and not probe_errors),
        "probe_error_papers": int(bool(probe_errors)),
        "elsevier_exact_sync_papers": int(elsevier_sync.get("status") == "ok"),
        "supplement_papers": int(bool(assets)),
        "direct_tabular_supplement_papers": int(bool(direct_assets)),
        "broad_tabular_candidate_papers": int(bool(broad_assets)),
        "downloaded_direct_tabular_papers": int(downloaded_direct),
        "complete_direct_tabular_papers": int(
            bool(direct_assets) and all(direct_downloaded_flags)
        ),
        "partial_direct_tabular_papers": int(
            any(direct_downloaded_flags) and not all(direct_downloaded_flags)
        ),
        "downloaded_broad_tabular_papers": int(downloaded_broad),
        "extracted_table_papers": int(bool(tables)),
        "agent_assessed_papers": int(bool(agent)),
        "agent_target_table_papers": int(int(agent.get("target_tables", 0) or 0) > 0),
        "marker_positive_papers": int(marker_positive),
        "post_filter_marker_positive_papers": int(post_filter_positive),
        "no_assets_clean_papers": int(bool(probe) and not probe_errors and not assets),
        "unresolved_no_assets_error_papers": int(bool(probe_errors) and not assets),
        "supplement_assets": len(assets),
        "direct_tabular_assets": len(direct_assets),
        "downloaded_direct_tabular_assets": sum(direct_downloaded_flags),
        "failed_direct_tabular_assets": sum(direct_failed_flags),
        "pending_direct_tabular_assets": len(direct_assets)
        - sum(direct_downloaded_flags),
        "broad_tabular_candidate_assets": len(broad_assets),
    }


def aggregate(
    records: list[tuple[Path, dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [(record_dir, metadata) for record_dir, metadata in records if predicate(metadata)]
    total = Counter()
    by_family: dict[str, Counter[str]] = {}
    for record_dir, metadata in selected:
        flags = record_flags(record_dir, metadata)
        total.update(flags)
        family = str(metadata.get("cns_family") or "Unknown")
        by_family.setdefault(family, Counter()).update(flags)
    return {
        "total": dict(total),
        "by_cns_family": {
            family: dict(counts) for family, counts in sorted(by_family.items())
        },
    }


def cohort_values(metadata: dict[str, Any]) -> set[str]:
    return set(metadata.get("screening", {}).get("discovery_cohorts", []))


def error_category(errors: list[str]) -> str:
    combined = " ".join(errors).lower()
    if "http error 403" in combined:
        return "publisher_http_403"
    if "http error 404" in combined:
        return "publisher_http_404"
    if "timed out" in combined or "timeout" in combined:
        return "timeout"
    return "other"


def build_census(database: Path) -> dict[str, Any]:
    records = []
    for metadata_path in sorted((database / "papers").glob("*/metadata.json")):
        metadata = read_json(metadata_path, {})
        if metadata:
            records.append((metadata_path.parent, metadata))

    scopes = {
        "all_cns_candidates": lambda _: True,
        "all_cell_single_cell": lambda m: "all_cell_single_cell" in cohort_values(m),
        "immune_single_cell": lambda m: "immune_single_cell" in cohort_values(m),
        "all_cell_only": lambda m: (
            "all_cell_single_cell" in cohort_values(m)
            and "immune_single_cell" not in cohort_values(m)
        ),
    }
    unresolved_records = []
    unresolved_summary = Counter()
    for record_dir, metadata in records:
        cohorts = cohort_values(metadata)
        probe = metadata.get("content_probe", {})
        errors = probe.get("errors", []) if isinstance(probe, dict) else []
        if (
            "all_cell_single_cell" not in cohorts
            or not errors
            or metadata.get("assets")
        ):
            continue
        family = str(metadata.get("cns_family") or "Unknown")
        category = error_category(errors)
        cohort_scope = (
            "immune_existing" if "immune_single_cell" in cohorts else "all_cell_only"
        )
        unresolved_summary[f"{cohort_scope}:{family}:{category}"] += 1
        unresolved_records.append(
            {
                "record": record_dir.name,
                "id": metadata.get("id", ""),
                "doi": metadata.get("doi", ""),
                "pmid": metadata.get("pmid", ""),
                "pmcid": metadata.get("pmcid", ""),
                "title": metadata.get("title", ""),
                "journal": metadata.get("journal", ""),
                "cns_family": family,
                "cohort_scope": cohort_scope,
                "discovery_cohorts": sorted(cohorts),
                "pmc_status": probe.get("pmc_status", ""),
                "publisher_status": probe.get("publisher_status", ""),
                "errors": errors,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(database),
        "definitions": {
            "direct_tabular_extensions": sorted(DIRECT_TABLE_EXTENSIONS),
            "broad_tabular_candidate_extensions": sorted(BROAD_TABLE_EXTENSIONS),
            "verified_marker_lower_bound": "post_filter_marker_positive_papers",
            "note": (
                "Supplement and tabular counts are retrieval candidates, not confirmed marker papers. "
                "A marker paper is verified only after table assessment and marker-row filtering."
            ),
        },
        "scopes": {
            name: aggregate(records, predicate) for name, predicate in scopes.items()
        },
        "unresolved_supplement_access": {
            "count": len(unresolved_records),
            "summary": dict(sorted(unresolved_summary.items())),
            "records": unresolved_records,
        },
    }


def markdown_table(scope: dict[str, Any]) -> str:
    metrics = [
        ("候选论文", "candidate_papers"),
        ("完成页面/PMC探测", "probe_completed_papers"),
        ("存在任意附录", "supplement_papers"),
        ("直接表格型附录", "direct_tabular_supplement_papers"),
        ("直接表格已完整下载", "complete_direct_tabular_papers"),
        ("直接表格部分下载", "partial_direct_tabular_papers"),
        ("直接表格已下载文件", "downloaded_direct_tabular_assets"),
        ("直接表格失败文件", "failed_direct_tabular_assets"),
        ("直接表格待处理文件", "pending_direct_tabular_assets"),
        ("宽松表格候选", "broad_tabular_candidate_papers"),
        ("已抽取表格", "extracted_table_papers"),
        ("已完成表格判定", "agent_assessed_papers"),
        ("判定含目标表", "agent_target_table_papers"),
        ("marker阳性", "marker_positive_papers"),
        ("过滤后marker阳性", "post_filter_marker_positive_papers"),
        ("探测错误且无附录", "unresolved_no_assets_error_papers"),
    ]
    families = sorted(scope["by_cns_family"])
    lines = [
        "| 指标 | 总计 | " + " | ".join(families) + " |",
        "|---|---:|" + "---:|" * len(families),
    ]
    for label, key in metrics:
        values = [str(scope["total"].get(key, 0))]
        values.extend(
            str(scope["by_cns_family"][family].get(key, 0)) for family in families
        )
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown(census: dict[str, Any]) -> str:
    labels = {
        "all_cns_candidates": "全部 CNS 候选",
        "all_cell_single_cell": "全细胞检索队列",
        "immune_single_cell": "原免疫队列",
        "all_cell_only": "全细胞新增队列",
    }
    lines = [
        "# 单细胞 marker 文献普查",
        "",
        f"生成时间（UTC）：{census['generated_at']}",
        "",
        "口径：直接表格型附件为 xlsx/xls/csv/tsv；宽松候选另含 docx/txt/zip。",
        "附录和表格型附件只代表可处理候选，不等同于包含 marker。可靠 marker 文献下界采用“过滤后 marker 阳性”。",
    ]
    for name, scope in census["scopes"].items():
        lines.extend(["", f"## {labels[name]}", "", markdown_table(scope)])
    unresolved = census["unresolved_supplement_access"]
    lines.extend(
        [
            "",
            "## 未解析的附录访问",
            "",
            f"全队列共 {unresolved['count']} 篇。这些论文不能计为“确认无附录”；逐篇记录见 JSON 报告。",
            "",
            "| 来源与错误 | 论文数 |",
            "|---|---:|",
        ]
    )
    for label, count in unresolved["summary"].items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    census = build_census(args.database)
    atomic_write(args.json_output, json.dumps(census, ensure_ascii=False, indent=2) + "\n")
    atomic_write(args.markdown_output, render_markdown(census))
    total = census["scopes"]["all_cns_candidates"]["total"]
    print(
        "Census written: candidates={candidates} supplements={supplements} "
        "direct_tabular={tabular} verified_marker_lower_bound={verified}".format(
            candidates=total.get("candidate_papers", 0),
            supplements=total.get("supplement_papers", 0),
            tabular=total.get("direct_tabular_supplement_papers", 0),
            verified=total.get("post_filter_marker_positive_papers", 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
