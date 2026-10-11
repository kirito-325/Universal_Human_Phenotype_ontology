#!/usr/bin/env python3
"""Map normal CellMarker 2.0 marker genes onto the expanded cell catalog.

The mapping is intentionally conservative about cancer context:
only CellMarker rows with cancer_type == "Normal" are used, and the outputs do
not include the cancer_type column.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
EXPANDED_DIR = ROOT / "cell_ontology" / "expanded"
CATALOG_TSV = EXPANDED_DIR / "expanded_cell_catalog.tsv"
SUMMARY_JSON = EXPANDED_DIR / "summary.json"
CELLMARKER_XLSX = PROJECT_ROOT / "CPO2" / "data" / "cellmarker" / "Cell_marker_Human.xlsx"
OUT_DIR = EXPANDED_DIR / "cellmarker_normal"
MAPPED_ROWS_TSV = OUT_DIR / "cellmarker_normal_name_search_rows.tsv"
NODE_TISSUE_TSV = OUT_DIR / "cellmarker_normal_node_tissue_summary.tsv"
NODE_SUMMARY_TSV = OUT_DIR / "cellmarker_normal_node_summary.tsv"
REPORT_MD = OUT_DIR / "cellmarker_normal_quality_report.md"
BROWSER_HTML = EXPANDED_DIR / "expanded_cell_catalog_cellmarker_normal.html"


SKIP_CONTAINS_QUERIES = {
    "cell",
    "cells",
    "normal cell",
    "normal cells",
    "activated cell",
    "activated cells",
    "effector cell",
    "effector cells",
}

CANCER_CONTEXT_RE = re.compile(
    r"\b("
    r"cancer|carcinoma|adenocarcinoma|sarcoma|melanoma|glioma|glioblastoma|"
    r"leukemia|leukaemia|lymphoma|myeloma|tumou?r|malignan\w*|neoplasm\w*|"
    r"metasta\w*|oncolog\w*|oncogen\w*|oncotarget"
    r")\b",
    re.IGNORECASE,
)


def clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def norm_text(value: object) -> str:
    text = clean(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def normalize_cl_id(value: object) -> str:
    text = clean(value).replace("CL_", "CL:")
    return text if text.startswith("CL:") else ""


def normalize_pmid(value: object) -> str:
    text = clean(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.isdigit() else ""


def has_cancer_context(row: pd.Series) -> bool:
    """Detect cancer/tumor context even when CellMarker marks cancer_type Normal."""
    fields = ["cell_type", "cell_name", "tissue_class", "tissue_type", "Title", "journal"]
    text = " ".join(clean(row.get(field, "")) for field in fields)
    return bool(CANCER_CONTEXT_RE.search(text))


def tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm_text(value))


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))


def singular_cell_variant(text: str) -> str:
    if text.endswith(" cells"):
        return text[:-1]
    return text


def without_parentheses(text: str) -> str:
    return " ".join(re.sub(r"\([^)]*\)", " ", text).split())


def catalog_query_terms(row: dict[str, str]) -> list[str]:
    raw_terms = [row.get("catalog_label", ""), row.get("cl_name", "")]
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        base = norm_text(raw)
        for item in (base, singular_cell_variant(base), without_parentheses(base)):
            item = norm_text(item)
            if item and item not in seen:
                seen.add(item)
                terms.append(item)
    return terms


def cellmarker_parenthetical_aliases(name: str) -> list[str]:
    aliases = []
    for match in re.findall(r"\(([^)]*)\)", name):
        for part in re.split(r"[/;,]", match):
            item = norm_text(part)
            if item:
                aliases.append(item)
    return aliases


def valid_contains_query(term: str) -> bool:
    if term in SKIP_CONTAINS_QUERIES:
        return False
    term_tokens = tokens(term)
    if not term_tokens:
        return False
    if term_tokens == ["cell"] or term_tokens == ["cells"]:
        return False
    if len(term_tokens) == 1:
        return len(term_tokens[0]) >= 5
    return True


def best_method(methods: Iterable[str]) -> str:
    priority = [
        "cl_id_exact",
        "name_exact",
        "name_parenthetical_exact",
        "name_token_contains",
    ]
    method_set = set(methods)
    for method in priority:
        if method in method_set:
            return method
    return sorted(method_set)[0]


def confidence_for(methods: Iterable[str]) -> str:
    method_set = set(methods)
    if method_set & {"cl_id_exact", "name_exact", "name_parenthetical_exact"}:
        return "high"
    return "medium"


def read_catalog() -> list[dict[str, str]]:
    with CATALOG_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_cellmarker() -> tuple[list[dict[str, str]], dict[str, int]]:
    df = pd.read_excel(CELLMARKER_XLSX, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    raw_rows = len(df)

    if "species" in df.columns:
        df = df[df["species"].map(norm_text) == "human"].copy()
    human_rows = len(df)

    df = df[df["cancer_type"].map(norm_text) == "normal"].copy()
    normal_rows = len(df)

    cancer_context_mask = df.apply(has_cancer_context, axis=1)
    cancer_context_excluded_rows = int(cancer_context_mask.sum())
    df = df[~cancer_context_mask].copy()
    normal_non_cancer_context_rows = len(df)

    df["gene_symbol"] = df["Symbol"].map(clean)
    df = df[df["gene_symbol"] != ""].copy()
    symbol_rows = len(df)

    rows: list[dict[str, str]] = []
    for original_index, row in df.iterrows():
        rows.append(
            {
                "cellmarker_row": str(original_index + 2),
                "cellmarker_cell_name": clean(row.get("cell_name", "")),
                "cellmarker_cell_id": normalize_cl_id(row.get("cellontology_id", "")),
                "tissue_class": clean(row.get("tissue_class", "")),
                "tissue_type": clean(row.get("tissue_type", "")),
                "uberon_id": clean(row.get("uberonongology_id", "")),
                "gene_symbol": clean(row.get("gene_symbol", "")),
                "raw_marker": clean(row.get("marker", "")),
                "gene_id": clean(row.get("GeneID", "")),
                "uniprot_id": clean(row.get("UNIPROTID", "")),
                "technology_seq": clean(row.get("technology_seq", "")),
                "marker_source": clean(row.get("marker_source", "")),
                "pmid": normalize_pmid(row.get("PMID", "")),
            }
        )

    dedup_keys = set()
    deduped = []
    for row in rows:
        key = (
            row["cellmarker_cell_name"],
            row["cellmarker_cell_id"],
            row["tissue_type"],
            row["uberon_id"],
            row["gene_symbol"],
            row["pmid"],
        )
        if key in dedup_keys:
            continue
        dedup_keys.add(key)
        deduped.append(row)

    stats = {
        "raw_rows": raw_rows,
        "human_rows": human_rows,
        "normal_rows": normal_rows,
        "normal_cancer_context_excluded_rows": cancer_context_excluded_rows,
        "normal_non_cancer_context_rows": normal_non_cancer_context_rows,
        "normal_rows_with_symbol": symbol_rows,
        "normal_deduplicated_rows": len(deduped),
    }
    return deduped, stats


def build_indexes(rows: list[dict[str, str]]) -> dict[str, object]:
    by_cl_id: dict[str, set[int]] = defaultdict(set)
    exact_full: dict[str, set[int]] = defaultdict(set)
    exact_alias: dict[str, set[int]] = defaultdict(set)
    by_name_norm: dict[str, set[int]] = defaultdict(set)
    unique_name_tokens: dict[str, list[str]] = {}

    for index, row in enumerate(rows):
        cl_id = row["cellmarker_cell_id"]
        if cl_id:
            by_cl_id[cl_id].add(index)
        name = row["cellmarker_cell_name"]
        name_norm = norm_text(name)
        if name_norm:
            exact_full[name_norm].add(index)
            by_name_norm[name_norm].add(index)
            unique_name_tokens.setdefault(name_norm, tokens(name))
            outside = without_parentheses(name_norm)
            if outside and outside != name_norm:
                exact_full[outside].add(index)
        for alias in cellmarker_parenthetical_aliases(name):
            exact_alias[alias].add(index)

    return {
        "by_cl_id": by_cl_id,
        "exact_full": exact_full,
        "exact_alias": exact_alias,
        "by_name_norm": by_name_norm,
        "unique_name_tokens": unique_name_tokens,
    }


def map_rows(
    catalog_rows: list[dict[str, str]],
    cellmarker_rows: list[dict[str, str]],
    indexes: dict[str, object],
) -> list[dict[str, str]]:
    by_cl_id = indexes["by_cl_id"]
    exact_full = indexes["exact_full"]
    exact_alias = indexes["exact_alias"]
    by_name_norm = indexes["by_name_norm"]
    unique_name_tokens = indexes["unique_name_tokens"]

    matches: dict[tuple[str, int], dict[str, set[str]]] = {}

    def add(catalog_id: str, row_index: int, method: str, query_term: str) -> None:
        key = (catalog_id, row_index)
        if key not in matches:
            matches[key] = {"methods": set(), "query_terms": set()}
        matches[key]["methods"].add(method)
        if query_term:
            matches[key]["query_terms"].add(query_term)

    for catalog in catalog_rows:
        catalog_id = catalog["cl_id"]
        if catalog_id == "CL:0000000":
            continue
        for row_index in by_cl_id.get(catalog_id, set()):
            add(catalog_id, row_index, "cl_id_exact", catalog_id)

        for query in catalog_query_terms(catalog):
            for row_index in exact_full.get(query, set()):
                add(catalog_id, row_index, "name_exact", query)
            for row_index in exact_alias.get(query, set()):
                add(catalog_id, row_index, "name_parenthetical_exact", query)

            if not valid_contains_query(query):
                continue
            query_tokens = tokens(query)
            for cell_name_norm, name_tokens in unique_name_tokens.items():
                if contains_sequence(name_tokens, query_tokens):
                    for row_index in by_name_norm[cell_name_norm]:
                        add(catalog_id, row_index, "name_token_contains", query)

    catalog_by_id = {row["cl_id"]: row for row in catalog_rows}
    out_rows: list[dict[str, str]] = []
    for (catalog_id, row_index), match in matches.items():
        catalog = catalog_by_id[catalog_id]
        cellmarker = cellmarker_rows[row_index]
        methods = sorted(match["methods"])
        final_method = best_method(methods)
        out_rows.append(
            {
                "catalog_cl_id": catalog["cl_id"],
                "catalog_label": catalog["catalog_label"],
                "catalog_cl_name": catalog["cl_name"],
                "catalog_branch": catalog["branch"],
                "catalog_rank": catalog["rank"],
                "catalog_source": catalog["source"],
                "match_confidence": confidence_for(methods),
                "best_match_method": final_method,
                "all_match_methods": "|".join(methods),
                "query_terms": "|".join(sorted(match["query_terms"])),
                **cellmarker,
            }
        )

    out_rows.sort(
        key=lambda row: (
            int(row.get("catalog_cl_id", "CL:9999999").split(":")[-1]),
            row["tissue_type"].lower(),
            row["cellmarker_cell_name"].lower(),
            row["gene_symbol"],
        )
    )
    return out_rows


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_values(values: Iterable[str], limit: int = 20) -> str:
    counter = Counter(value for value in values if value)
    return ";".join(value for value, _ in counter.most_common(limit))


def summarize(mapped_rows: list[dict[str, str]], catalog_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    catalog_by_id = {row["cl_id"]: row for row in catalog_rows}
    by_node: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_node_tissue: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in mapped_rows:
        by_node[row["catalog_cl_id"]].append(row)
        by_node_tissue[(row["catalog_cl_id"], row["tissue_type"])].append(row)

    node_summary = []
    for catalog_id, rows in sorted(by_node.items(), key=lambda item: catalog_by_id[item[0]]["order"]):
        catalog = catalog_by_id[catalog_id]
        high_rows = [row for row in rows if row["match_confidence"] == "high"]
        medium_rows = [row for row in rows if row["match_confidence"] == "medium"]
        node_summary.append(
            {
                "catalog_cl_id": catalog_id,
                "catalog_label": catalog["catalog_label"],
                "catalog_cl_name": catalog["cl_name"],
                "catalog_branch": catalog["branch"],
                "catalog_rank": catalog["rank"],
                "catalog_source": catalog["source"],
                "evidence_links": len(rows),
                "high_confidence_links": len(high_rows),
                "medium_confidence_links": len(medium_rows),
                "tissue_count": len({row["tissue_type"] for row in rows if row["tissue_type"]}),
                "gene_count": len({row["gene_symbol"] for row in rows if row["gene_symbol"]}),
                "pmid_count": len({row["pmid"] for row in rows if row["pmid"]}),
                "cellmarker_cell_count": len({row["cellmarker_cell_name"] for row in rows if row["cellmarker_cell_name"]}),
                "best_match_methods": top_values((row["best_match_method"] for row in rows), 8),
                "top_tissues": top_values((row["tissue_type"] for row in rows), 12),
                "top_genes": top_values((row["gene_symbol"] for row in rows), 30),
                "cellmarker_cell_names": top_values((row["cellmarker_cell_name"] for row in rows), 20),
            }
        )

    tissue_summary = []
    for (catalog_id, tissue), rows in sorted(
        by_node_tissue.items(),
        key=lambda item: (catalog_by_id[item[0][0]]["order"], item[0][1].lower()),
    ):
        catalog = catalog_by_id[catalog_id]
        tissue_summary.append(
            {
                "catalog_cl_id": catalog_id,
                "catalog_label": catalog["catalog_label"],
                "catalog_cl_name": catalog["cl_name"],
                "catalog_branch": catalog["branch"],
                "tissue_type": tissue,
                "tissue_class": top_values((row["tissue_class"] for row in rows), 3),
                "uberon_id": top_values((row["uberon_id"] for row in rows), 3),
                "evidence_links": len(rows),
                "high_confidence_links": sum(row["match_confidence"] == "high" for row in rows),
                "medium_confidence_links": sum(row["match_confidence"] == "medium" for row in rows),
                "gene_count": len({row["gene_symbol"] for row in rows if row["gene_symbol"]}),
                "pmid_count": len({row["pmid"] for row in rows if row["pmid"]}),
                "cellmarker_cell_count": len({row["cellmarker_cell_name"] for row in rows if row["cellmarker_cell_name"]}),
                "top_genes": top_values((row["gene_symbol"] for row in rows), 30),
                "cellmarker_cell_names": top_values((row["cellmarker_cell_name"] for row in rows), 20),
                "best_match_methods": top_values((row["best_match_method"] for row in rows), 8),
            }
        )

    return node_summary, tissue_summary


def markdown_table(rows: list[dict[str, object]], fields: list[str], limit: int = 20) -> str:
    if not rows:
        return "\n_None._\n"
    clipped = rows[:limit]
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in clipped:
        values = [str(row.get(field, "")).replace("|", "/") for field in fields]
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines) + "\n"


def write_report(
    catalog_rows: list[dict[str, str]],
    cellmarker_stats: dict[str, int],
    mapped_rows: list[dict[str, str]],
    node_summary: list[dict[str, object]],
) -> None:
    matched_nodes = {row["catalog_cl_id"] for row in mapped_rows}
    high_nodes = {row["catalog_cl_id"] for row in mapped_rows if row["match_confidence"] == "high"}
    source_rows = {row["cellmarker_row"] for row in mapped_rows}
    method_counts = Counter(row["best_match_method"] for row in mapped_rows)
    confidence_counts = Counter(row["match_confidence"] for row in mapped_rows)

    catalog_by_id = {row["cl_id"]: row for row in catalog_rows}
    branch_rows = []
    for branch in sorted({row["branch"] for row in catalog_rows}):
        branch_nodes = [row for row in catalog_rows if row["branch"] == branch]
        branch_matched = [row for row in branch_nodes if row["cl_id"] in matched_nodes]
        branch_rows.append(
            {
                "branch": branch,
                "nodes": len(branch_nodes),
                "matched": len(branch_matched),
                "coverage": f"{100 * len(branch_matched) / len(branch_nodes):.1f}%",
            }
        )

    broad_rows = [
        row
        for row in node_summary
        if int(row["medium_confidence_links"]) >= 50
        and int(row["medium_confidence_links"]) > max(10, int(row["high_confidence_links"]) * 5)
    ]
    broad_rows.sort(key=lambda row: int(row["medium_confidence_links"]), reverse=True)

    top_nodes = sorted(node_summary, key=lambda row: int(row["evidence_links"]), reverse=True)
    high_top_nodes = sorted(node_summary, key=lambda row: int(row["high_confidence_links"]), reverse=True)

    unmatched_skeleton = [
        {
            "catalog_cl_id": row["cl_id"],
            "catalog_label": row["catalog_label"],
            "branch": row["branch"],
            "rank": row["rank"],
        }
        for row in catalog_rows
        if row["source"] == "skeleton" and row["cl_id"] not in matched_nodes
    ]

    lines = [
        "# Normal CellMarker 2.0 Mapping Quality Report",
        "",
        "Only CellMarker rows with `cancer_type == Normal` were used. Rows whose cell, tissue, title, or journal text contains tumor/cancer context keywords were also excluded.",
        "",
        "## Inputs",
        "",
        f"- Catalog nodes: {len(catalog_rows)}",
        f"- Raw CellMarker rows: {cellmarker_stats['raw_rows']}",
        f"- Human rows: {cellmarker_stats['human_rows']}",
        f"- Normal rows: {cellmarker_stats['normal_rows']}",
        f"- Normal rows excluded by tumor/cancer context keywords: {cellmarker_stats['normal_cancer_context_excluded_rows']}",
        f"- Normal rows after context exclusion: {cellmarker_stats['normal_non_cancer_context_rows']}",
        f"- Normal rows with HGNC symbol: {cellmarker_stats['normal_rows_with_symbol']}",
        f"- Deduplicated normal marker rows: {cellmarker_stats['normal_deduplicated_rows']}",
        "",
        "## Mapping Summary",
        "",
        f"- Catalog nodes with any normal marker link: {len(matched_nodes)} / {len(catalog_rows)} ({100 * len(matched_nodes) / len(catalog_rows):.1f}%)",
        f"- Catalog nodes with high-confidence link: {len(high_nodes)} / {len(catalog_rows)} ({100 * len(high_nodes) / len(catalog_rows):.1f}%)",
        f"- Evidence links after mapping: {len(mapped_rows)}",
        f"- Unique CellMarker normal source rows linked: {len(source_rows)}",
        f"- Match confidence counts: {dict(sorted(confidence_counts.items()))}",
        f"- Best match method counts: {dict(sorted(method_counts.items()))}",
        "",
        "## Branch Coverage",
        "",
        markdown_table(branch_rows, ["branch", "nodes", "matched", "coverage"], limit=50),
        "",
        "## Top Nodes By All Evidence Links",
        "",
        markdown_table(
            top_nodes,
            [
                "catalog_cl_id",
                "catalog_label",
                "catalog_branch",
                "evidence_links",
                "high_confidence_links",
                "medium_confidence_links",
                "tissue_count",
                "gene_count",
                "top_tissues",
            ],
            limit=25,
        ),
        "",
        "## Top Nodes By High-Confidence Links",
        "",
        markdown_table(
            high_top_nodes,
            [
                "catalog_cl_id",
                "catalog_label",
                "catalog_branch",
                "high_confidence_links",
                "tissue_count",
                "gene_count",
                "top_tissues",
                "top_genes",
            ],
            limit=25,
        ),
        "",
        "## Broad Name-Search Candidates To Review",
        "",
        "These nodes are dominated by token-containing name hits, so they are useful for discovery but should not be treated as direct marker assertions without review.",
        "",
        markdown_table(
            broad_rows,
            [
                "catalog_cl_id",
                "catalog_label",
                "catalog_branch",
                "high_confidence_links",
                "medium_confidence_links",
                "cellmarker_cell_names",
            ],
            limit=30,
        ),
        "",
        "## Skeleton Nodes Without Normal CellMarker Hits",
        "",
        markdown_table(unmatched_skeleton, ["catalog_cl_id", "catalog_label", "branch", "rank"], limit=60),
    ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def marker_payload(node_summary: list[dict[str, object]], tissue_summary: list[dict[str, object]]) -> dict[str, object]:
    tissue_by_node: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in tissue_summary:
        tissue_by_node[str(row["catalog_cl_id"])].append(
            {
                "tissue": row["tissue_type"],
                "links": row["evidence_links"],
                "high": row["high_confidence_links"],
                "medium": row["medium_confidence_links"],
                "genes": str(row["top_genes"]).split(";")[:12],
                "geneCount": row["gene_count"],
                "pmidCount": row["pmid_count"],
                "cellNames": str(row["cellmarker_cell_names"]).split(";")[:6],
            }
        )

    payload = {}
    for row in node_summary:
        catalog_id = str(row["catalog_cl_id"])
        tissues = sorted(tissue_by_node[catalog_id], key=lambda item: (-int(item["links"]), str(item["tissue"])))[:15]
        payload[catalog_id] = {
            "links": row["evidence_links"],
            "high": row["high_confidence_links"],
            "medium": row["medium_confidence_links"],
            "tissueCount": row["tissue_count"],
            "geneCount": row["gene_count"],
            "pmidCount": row["pmid_count"],
            "cellCount": row["cellmarker_cell_count"],
            "methods": str(row["best_match_methods"]).split(";"),
            "topTissues": str(row["top_tissues"]).split(";")[:12],
            "topGenes": str(row["top_genes"]).split(";")[:25],
            "cellNames": str(row["cellmarker_cell_names"]).split(";")[:20],
            "tissues": tissues,
        }
    return payload


def render_browser(catalog_rows: list[dict[str, str]], node_summary: list[dict[str, object]], tissue_summary: list[dict[str, object]]) -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.exists() else {}
    nodes = []
    for row in catalog_rows:
        nodes.append(
            {
                "id": row["cl_id"],
                "parent": row["parent_id"],
                "label": row["catalog_label"],
                "labelZh": row["catalog_label_zh"],
                "clName": row["cl_name"],
                "rank": row["rank"],
                "source": row["source"],
                "branch": row["branch"],
                "anchor": row["anchor_id"],
                "distance": row["distance_to_anchor"],
                "level": int(row["level"]),
                "note": row["note"],
                "order": int(row["order"]),
            }
        )

    data_json = json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
    marker_json = json.dumps(marker_payload(node_summary, tissue_summary), ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    title = "Expanded Cell Catalog - Normal CellMarker Preview"

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fb; --panel:#fff; --text:#172033; --muted:#647085; --line:#d8dee8; --hover:#edf2f7; --selected:#e5f0ff; --accent:#2776a6; --warn:#8a5a00; --radius:8px; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; min-height:100vh; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:var(--panel); border-bottom:1px solid var(--line); padding:18px 22px 14px; }}
    h1 {{ margin:0 0 12px; font-size:23px; line-height:1.2; letter-spacing:0; }}
    .stats {{ display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px; }}
    .stats div,.metric {{ background:#fbfcfe; border:1px solid var(--line); border-radius:var(--radius); padding:9px 11px; min-width:0; }}
    .stats b,.metric b {{ display:block; font-size:18px; line-height:1.1; overflow-wrap:anywhere; }}
    .stats span,.metric span,.muted {{ color:var(--muted); font-size:12px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(220px,1fr) 170px 150px 150px 150px auto auto; gap:8px; align-items:center; padding:12px 22px; background:#edf1f6; border-bottom:1px solid var(--line); }}
    input,select,button {{ height:36px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--text); font:inherit; min-width:0; }}
    input,select {{ padding:0 10px; }}
    button {{ padding:0 12px; cursor:pointer; white-space:nowrap; }}
    button:hover {{ background:var(--hover); }}
    main {{ display:grid; grid-template-columns:minmax(520px,1fr) 460px; gap:14px; padding:14px 22px 22px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); min-width:0; }}
    .tree-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 14px; border-bottom:1px solid var(--line); }}
    .tree-head h2,.details h2 {{ margin:0; font-size:15px; letter-spacing:0; }}
    #tree {{ height:calc(100vh - 222px); min-height:500px; overflow:auto; padding:10px 8px 18px; }}
    .row {{ display:grid; grid-template-columns:auto 24px minmax(0,1fr) auto; align-items:center; min-height:36px; padding-right:8px; border-radius:6px; cursor:pointer; }}
    .row:hover {{ background:var(--hover); }}
    .row.selected {{ background:var(--selected); }}
    .indent {{ width:calc(var(--depth) * 18px); height:100%; }}
    .toggle {{ width:22px; height:22px; display:grid; place-items:center; border:0; background:transparent; color:var(--muted); padding:0; font-size:14px; line-height:1; }}
    .toggle.placeholder {{ visibility:hidden; }}
    .label {{ display:flex; align-items:center; gap:8px; min-width:0; }}
    .swatch {{ width:9px; height:24px; border-radius:4px; flex:0 0 auto; background:var(--branch-color); }}
    .name {{ font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .zh,.id {{ color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .zh {{ max-width:160px; }}
    .id {{ font-size:12px; }}
    .badges {{ display:flex; align-items:center; gap:5px; margin-left:8px; }}
    .badge {{ height:22px; min-width:42px; padding:0 8px; border:1px solid var(--line); background:#f8fafc; color:var(--muted); border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:12px; white-space:nowrap; }}
    .badge.marker {{ color:#215c44; background:#e7f5ee; border-color:#bfdccd; }}
    mark {{ background:#fff1a8; color:inherit; padding:0 1px; border-radius:2px; }}
    .details {{ padding:14px; position:sticky; top:12px; height:fit-content; max-height:calc(100vh - 24px); overflow:auto; }}
    .title {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
    .color {{ width:12px; height:30px; border-radius:5px; background:var(--branch-color); flex:0 0 auto; }}
    .detail-zh {{ color:var(--muted); margin-top:2px; }}
    .meta-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:12px 0; }}
    .field {{ border-top:1px solid var(--line); padding-top:11px; margin-top:11px; }}
    .field label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; color:#26364f; background:#f1f4f8; border-radius:4px; padding:2px 5px; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .chip {{ border:1px solid var(--line); border-radius:999px; padding:4px 8px; background:#fbfcfe; color:#344055; font-size:12px; }}
    .tissue {{ border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }}
    .tissue-head {{ display:flex; justify-content:space-between; gap:8px; font-weight:700; }}
    .empty {{ color:var(--muted); padding:22px; text-align:center; }}
    .warn {{ color:var(--warn); }}
    @media (max-width:1120px) {{ .toolbar {{ grid-template-columns:1fr 1fr 1fr; }} main {{ grid-template-columns:1fr; }} .details {{ position:static; max-height:none; }} #tree {{ height:62vh; }} }}
    @media (max-width:720px) {{ header,.toolbar,main {{ padding-left:12px; padding-right:12px; }} .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .toolbar {{ grid-template-columns:1fr; }} .badges {{ display:none; }} .id {{ display:none; }} .name,.zh {{ white-space:normal; max-width:none; }} .swatch {{ display:none; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search name, Chinese label, CL ID, marker gene, or tissue">
    <select id="branchFilter" aria-label="Filter by branch"></select>
    <select id="sourceFilter" aria-label="Filter by source"></select>
    <select id="rankFilter" aria-label="Filter by rank"></select>
    <select id="markerFilter" aria-label="Filter by marker coverage">
      <option value="all">All marker states</option>
      <option value="with">With normal markers</option>
      <option value="high">High-confidence markers</option>
      <option value="none">No normal markers</option>
    </select>
    <button id="expandAll" type="button">Expand</button>
    <button id="collapseAll" type="button">Collapse</button>
  </div>
  <main>
    <section class="panel tree-panel">
      <div class="tree-head"><h2>Tree</h2><div class="muted" id="treeCount"></div></div>
      <div id="tree"></div>
    </section>
    <aside class="panel details" id="details"></aside>
  </main>
  <script>
    const DATA = {data_json};
    const MARKERS = {marker_json};
    const SUMMARY = {summary_json};
    const COLORS = {{"root":"#4a5568","stem_progenitor":"#697b2f","developmental":"#766bb0","epithelial":"#278276","endothelial":"#2776a6","stromal_connective":"#9a6b2f","bone":"#7d6d36","muscle":"#b24d45","neural":"#6754b5","blood_immune":"#b15a25","germline":"#2f7f52","endocrine":"#a45a9c","pigment":"#4d5961"}};
    const byId = new Map(DATA.map(node => [node.id, node]));
    const children = new Map();
    DATA.forEach(node => {{ if (!children.has(node.parent)) children.set(node.parent, []); children.get(node.parent).push(node); }});
    for (const list of children.values()) list.sort((a, b) => a.order - b.order);
    const root = DATA.find(node => !node.parent) || DATA[0];
    let selectedId = root.id;
    let searchText = "";
    let branchFilter = "all";
    let sourceFilter = "all";
    let rankFilter = "all";
    let markerFilter = "all";
    let collapsed = new Set(DATA.filter(node => node.parent && node.level > 2).map(node => node.id));
    const markerNodeCount = Object.keys(MARKERS).length;
    const highNodeCount = Object.values(MARKERS).filter(item => Number(item.high) > 0).length;
    const linkedRows = Object.values(MARKERS).reduce((sum, item) => sum + Number(item.links || 0), 0);
    function branchColor(branch) {{ return COLORS[branch] || "#647085"; }}
    function markerOf(id) {{ return MARKERS[id] || null; }}
    function escapeHtml(value) {{ return String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }}
    function highlight(value) {{
      const safe = escapeHtml(value || "");
      if (!searchText) return safe;
      const escaped = searchText.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return safe.replace(new RegExp("(" + escaped + ")", "ig"), "<mark>$1</mark>");
    }}
    function ancestorIds(id) {{
      const ids = [];
      let node = byId.get(id);
      while (node && node.parent) {{ ids.push(node.parent); node = byId.get(node.parent); }}
      return ids;
    }}
    function descendantIds(id) {{
      const ids = [];
      const stack = [...(children.get(id) || [])];
      while (stack.length) {{ const node = stack.pop(); ids.push(node.id); stack.push(...(children.get(node.id) || [])); }}
      return ids;
    }}
    function markerSearchText(marker) {{
      if (!marker) return "";
      return [marker.topGenes, marker.topTissues, marker.cellNames, marker.methods].flat().join(" ").toLowerCase();
    }}
    function nodeMatches(node) {{
      const marker = markerOf(node.id);
      const text = [node.label, node.labelZh, node.clName, node.id, node.branch, node.rank, node.source, markerSearchText(marker)].join(" ").toLowerCase();
      const searchOk = !searchText || text.includes(searchText);
      const branchOk = branchFilter === "all" || node.branch === branchFilter;
      const sourceOk = sourceFilter === "all" || node.source === sourceFilter;
      const rankOk = rankFilter === "all" || node.rank === rankFilter;
      const markerOk = markerFilter === "all" || (markerFilter === "with" && marker) || (markerFilter === "high" && marker && Number(marker.high) > 0) || (markerFilter === "none" && !marker);
      return searchOk && branchOk && sourceOk && rankOk && markerOk;
    }}
    function visibleIds() {{
      const ids = new Set();
      DATA.forEach(node => {{ if (nodeMatches(node)) {{ ids.add(node.id); ancestorIds(node.id).forEach(parent => ids.add(parent)); }} }});
      if (!searchText && branchFilter === "all" && sourceFilter === "all" && rankFilter === "all" && markerFilter === "all") DATA.forEach(node => ids.add(node.id));
      return ids;
    }}
    function renderStats() {{
      const stats = [["Catalog nodes", DATA.length], ["With normal markers", markerNodeCount], ["High-confidence", highNodeCount], ["Evidence links", linkedRows], ["Normal only", "context filtered"], ["CL added", SUMMARY.cl_extension_nodes || ""]];
      document.getElementById("stats").innerHTML = stats.map(item => '<div><b>' + escapeHtml(item[1]) + '</b><span>' + escapeHtml(item[0]) + '</span></div>').join("");
    }}
    function renderFilters() {{
      const fill = (id, label, values) => {{ document.getElementById(id).innerHTML = '<option value="all">All ' + label + '</option>' + values.map(value => '<option value="' + escapeHtml(value) + '">' + escapeHtml(value) + '</option>').join(""); }};
      fill("branchFilter", "branches", [...new Set(DATA.map(node => node.branch))]);
      fill("sourceFilter", "sources", [...new Set(DATA.map(node => node.source))]);
      fill("rankFilter", "ranks", [...new Set(DATA.map(node => node.rank))]);
    }}
    function renderTree() {{
      const visible = visibleIds();
      if (searchText || branchFilter !== "all" || sourceFilter !== "all" || rankFilter !== "all" || markerFilter !== "all") visible.forEach(id => ancestorIds(id).forEach(parent => collapsed.delete(parent)));
      const tree = document.getElementById("tree");
      tree.innerHTML = "";
      let shown = 0;
      function addNode(node, depth) {{
        if (!visible.has(node.id)) return;
        shown += 1;
        const childList = (children.get(node.id) || []).filter(child => visible.has(child.id));
        const hasChildren = childList.length > 0;
        const isCollapsed = collapsed.has(node.id);
        const marker = markerOf(node.id);
        const row = document.createElement("div");
        row.className = "row" + (node.id === selectedId ? " selected" : "");
        row.style.setProperty("--branch-color", branchColor(node.branch));
        row.innerHTML = '<div class="indent" style="--depth:' + depth + '"></div>' +
          '<button class="toggle ' + (hasChildren ? "" : "placeholder") + '" type="button">' + (isCollapsed ? ">" : "v") + '</button>' +
          '<div class="label"><span class="swatch"></span><span class="name">' + highlight(node.label) + '</span><span class="zh">' + highlight(node.labelZh) + '</span><span class="id">' + highlight(node.id) + '</span></div>' +
          '<div class="badges">' + (marker ? '<span class="badge marker">' + escapeHtml(marker.geneCount) + ' genes</span>' : '') + '<span class="badge">' + escapeHtml(node.rank) + '</span></div>';
        row.addEventListener("click", event => {{ if (event.target.classList.contains("toggle")) return; selectedId = node.id; renderTree(); renderDetails(); }});
        row.querySelector(".toggle").addEventListener("click", () => {{ if (collapsed.has(node.id)) collapsed.delete(node.id); else collapsed.add(node.id); renderTree(); }});
        tree.appendChild(row);
        if (hasChildren && !isCollapsed) childList.forEach(child => addNode(child, depth + 1));
      }}
      addNode(root, 0);
      if (!shown) tree.innerHTML = '<div class="empty">No matching nodes</div>';
      document.getElementById("treeCount").textContent = shown + " / " + DATA.length + " nodes";
    }}
    function chipList(items) {{
      const clean = (items || []).filter(Boolean);
      if (!clean.length) return '<span class="muted">None</span>';
      return '<div class="chips">' + clean.map(item => '<span class="chip">' + escapeHtml(item) + '</span>').join("") + '</div>';
    }}
    function renderMarker(marker) {{
      if (!marker) return '<div class="field"><label>Normal CellMarker 2.0</label><div class="muted">No normal marker hits for this catalog node.</div></div>';
      const warning = Number(marker.medium) > Math.max(10, Number(marker.high) * 5) ? '<div class="field warn"><label>Review note</label>Name-search hits dominate this node; treat as discovery evidence until reviewed.</div>' : '';
      const tissues = (marker.tissues || []).map(item => '<div class="tissue"><div class="tissue-head"><span>' + escapeHtml(item.tissue) + '</span><span class="muted">' + escapeHtml(item.geneCount) + ' genes / ' + escapeHtml(item.links) + ' links</span></div>' + chipList(item.genes) + '<div class="muted">' + escapeHtml((item.cellNames || []).join("; ")) + '</div></div>').join("");
      return '<div class="field"><label>Normal CellMarker 2.0 summary</label><div class="meta-grid">' +
        '<div class="metric"><b>' + escapeHtml(marker.links) + '</b><span>Evidence links</span></div>' +
        '<div class="metric"><b>' + escapeHtml(marker.high) + '</b><span>High-confidence links</span></div>' +
        '<div class="metric"><b>' + escapeHtml(marker.tissueCount) + '</b><span>Tissues</span></div>' +
        '<div class="metric"><b>' + escapeHtml(marker.geneCount) + '</b><span>Genes</span></div>' +
        '</div></div>' + warning +
        '<div class="field"><label>Top genes</label>' + chipList(marker.topGenes) + '</div>' +
        '<div class="field"><label>CellMarker cell names</label>' + chipList(marker.cellNames) + '</div>' +
        '<div class="field"><label>Top tissue contexts</label>' + tissues + '</div>';
    }}
    function renderDetails() {{
      const node = byId.get(selectedId) || root;
      const parent = node.parent ? byId.get(node.parent) : null;
      const anchor = byId.get(node.anchor);
      const lineage = [...ancestorIds(node.id)].reverse().map(id => byId.get(id)).filter(Boolean).concat(node);
      const details = document.getElementById("details");
      details.style.setProperty("--branch-color", branchColor(node.branch));
      details.innerHTML = '<div class="title"><span class="color"></span><div><h2>' + escapeHtml(node.label) + '</h2><div class="detail-zh">' + escapeHtml(node.labelZh) + '</div></div></div>' +
        '<div class="field"><label>CL ID</label><code>' + escapeHtml(node.id) + '</code></div>' +
        '<div class="meta-grid"><div class="metric"><b>' + escapeHtml(node.source === "skeleton" ? "base" : "CL") + '</b><span>Source</span></div><div class="metric"><b>' + escapeHtml(node.rank) + '</b><span>Rank</span></div><div class="metric"><b>' + escapeHtml((children.get(node.id) || []).length) + '</b><span>Children</span></div><div class="metric"><b>' + escapeHtml(descendantIds(node.id).length) + '</b><span>Descendants</span></div></div>' +
        renderMarker(markerOf(node.id)) +
        '<div class="field"><label>CL name</label>' + escapeHtml(node.clName) + '</div>' +
        '<div class="field"><label>Branch</label>' + escapeHtml(node.branch) + '</div>' +
        '<div class="field"><label>Parent</label>' + (parent ? escapeHtml(parent.label) + ' <code>' + escapeHtml(parent.id) + '</code>' : "None") + '</div>' +
        '<div class="field"><label>Skeleton anchor</label>' + (anchor ? escapeHtml(anchor.label) + ' <code>' + escapeHtml(anchor.id) + '</code>' : "") + '</div>' +
        '<div class="field"><label>Lineage</label>' + escapeHtml(lineage.map(item => item.label).join(" / ")) + '</div>';
    }}
    document.getElementById("search").addEventListener("input", event => {{ searchText = event.target.value.trim().toLowerCase(); renderTree(); }});
    document.getElementById("branchFilter").addEventListener("change", event => {{ branchFilter = event.target.value; renderTree(); }});
    document.getElementById("sourceFilter").addEventListener("change", event => {{ sourceFilter = event.target.value; renderTree(); }});
    document.getElementById("rankFilter").addEventListener("change", event => {{ rankFilter = event.target.value; renderTree(); }});
    document.getElementById("markerFilter").addEventListener("change", event => {{ markerFilter = event.target.value; renderTree(); }});
    document.getElementById("expandAll").addEventListener("click", () => {{ collapsed = new Set(); renderTree(); }});
    document.getElementById("collapseAll").addEventListener("click", () => {{ collapsed = new Set(DATA.filter(node => node.id !== root.id).map(node => node.id)); renderTree(); }});
    renderStats(); renderFilters(); renderTree(); renderDetails();
  </script>
</body>
</html>
"""
    BROWSER_HTML.write_text(doc, encoding="utf-8")


def main() -> None:
    if not CATALOG_TSV.exists():
        raise FileNotFoundError(CATALOG_TSV)
    if not CELLMARKER_XLSX.exists():
        raise FileNotFoundError(CELLMARKER_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_rows = read_catalog()
    cellmarker_rows, cellmarker_stats = read_cellmarker()
    indexes = build_indexes(cellmarker_rows)
    mapped_rows = map_rows(catalog_rows, cellmarker_rows, indexes)
    node_summary, tissue_summary = summarize(mapped_rows, catalog_rows)

    mapped_fields = [
        "catalog_cl_id",
        "catalog_label",
        "catalog_cl_name",
        "catalog_branch",
        "catalog_rank",
        "catalog_source",
        "match_confidence",
        "best_match_method",
        "all_match_methods",
        "query_terms",
        "cellmarker_row",
        "cellmarker_cell_name",
        "cellmarker_cell_id",
        "tissue_class",
        "tissue_type",
        "uberon_id",
        "gene_symbol",
        "raw_marker",
        "gene_id",
        "uniprot_id",
        "technology_seq",
        "marker_source",
        "pmid",
    ]
    node_fields = [
        "catalog_cl_id",
        "catalog_label",
        "catalog_cl_name",
        "catalog_branch",
        "catalog_rank",
        "catalog_source",
        "evidence_links",
        "high_confidence_links",
        "medium_confidence_links",
        "tissue_count",
        "gene_count",
        "pmid_count",
        "cellmarker_cell_count",
        "best_match_methods",
        "top_tissues",
        "top_genes",
        "cellmarker_cell_names",
    ]
    tissue_fields = [
        "catalog_cl_id",
        "catalog_label",
        "catalog_cl_name",
        "catalog_branch",
        "tissue_type",
        "tissue_class",
        "uberon_id",
        "evidence_links",
        "high_confidence_links",
        "medium_confidence_links",
        "gene_count",
        "pmid_count",
        "cellmarker_cell_count",
        "top_genes",
        "cellmarker_cell_names",
        "best_match_methods",
    ]
    write_tsv(MAPPED_ROWS_TSV, mapped_rows, mapped_fields)
    write_tsv(NODE_SUMMARY_TSV, node_summary, node_fields)
    write_tsv(NODE_TISSUE_TSV, tissue_summary, tissue_fields)
    write_report(catalog_rows, cellmarker_stats, mapped_rows, node_summary)
    render_browser(catalog_rows, node_summary, tissue_summary)

    matched_nodes = {row["catalog_cl_id"] for row in mapped_rows}
    high_nodes = {row["catalog_cl_id"] for row in mapped_rows if row["match_confidence"] == "high"}
    print(
        json.dumps(
            {
                **cellmarker_stats,
                "catalog_nodes": len(catalog_rows),
                "mapped_evidence_links": len(mapped_rows),
                "matched_catalog_nodes": len(matched_nodes),
                "high_confidence_catalog_nodes": len(high_nodes),
                "outputs": {
                    "mapped_rows": str(MAPPED_ROWS_TSV),
                    "node_summary": str(NODE_SUMMARY_TSV),
                    "node_tissue_summary": str(NODE_TISSUE_TSV),
                    "quality_report": str(REPORT_MD),
                    "browser_html": str(BROWSER_HTML),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
