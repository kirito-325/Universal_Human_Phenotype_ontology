#!/usr/bin/env python3
"""Conservatively filter obvious non-cell-type agent-standardized names."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "database"
DEFAULT_INPUT = DATABASE / "agent_standardized_cell_names.tsv"
DEFAULT_PAPERS_DIR = DATABASE / "papers"
DEFAULT_FILTERED = DATABASE / "agent_standardized_cell_names_filtered.tsv"
DEFAULT_EXCLUDED = DATABASE / "agent_standardized_cell_names_excluded.tsv"
DEFAULT_HTML = DATABASE / "agent_standardized_cell_names_filtered.html"
DEFAULT_SUMMARY = DATABASE / "agent_standardized_cell_names_filter_summary.json"
DEFAULT_CATALOG = ROOT / "expanded" / "expanded_cell_catalog.tsv"

NUMERIC_RE = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$", re.I)
BARCODE_RE = re.compile(r"^[ACGTN]{12,}(?:-\d+)?$", re.I)
DATE_OR_NUMERIC_CODE_RE = re.compile(r"^[\d\s:./-]+$")
GENE_ACCESSION_RE = re.compile(
    r"^(?:ENS[A-Z]*\d+(?:\.\d+)?|[A-Z]{2}\d{5,}(?:[.,]\d+)?)$", re.I
)
GENE_SET_RE = re.compile(
    r"\b(?:module|program|pathway|regulon|signature|classifier|gene[ _-](?:set|module|program|signature|score))s?\b",
    re.I,
)
PROCESS_LABEL_RE = re.compile(
    r"\b(?:activation|apoptosis|lactation|metabolism|response|scavenging|signaling|uptake)$",
    re.I,
)
QUALITY_LABEL_RE = re.compile(
    r"(?:low[ _-]?quality|high[ _-]?(?:mito|ribo)|degraded cells?|small fraction|\(excluded\))",
    re.I,
)
TISSUE_PANEL_RE = re.compile(r"^(?:blood|brain|liver|lung|skin|bone[ _-]marrow)[_-]", re.I)
CITATION_RE = re.compile(r"(?:\bet\s+al\.?|,).*(?:19|20)\d{2}", re.I)
DIRECTIONAL_SET_RE = re.compile(
    r"(?:^|[_ -])(?:up|down)(?:regulated)?(?:$|[_ -])|_(?:up|down)$", re.I
)
SIGNATURE_COUNT_RE = re.compile(r"^[A-Z][A-Z0-9_ +.-]*\s\d{2,}$")
COMPARISON_RE = re.compile(r"(?:^|[\s(])(?:vs\.?|versus)(?:[\s)]|$)", re.I)
SEPARATED_COMPARISON_RE = re.compile(
    r"(?:^|[\s_()/.-])(?:vs\.?|versus)(?:$|[\s_()/.-])", re.I
)
CLUSTER_RE = re.compile(
    r"^(?:(?:seurat[ _-]?)?clusters?|clust)[ _.:#-]*\d+[a-z]?$", re.I
)
SIMPLE_CODE_RE = re.compile(r"^(?:c|p)\d+[a-z]*$", re.I)
SEPARATED_CODE_RE = re.compile(r"^[a-z][a-z0-9-]*[_-]\d+[a-z]*$", re.I)
LEADING_NUMBER_CODE_RE = re.compile(r"^\d+[_-][a-z][a-z0-9_.-]*$", re.I)
NUMBERED_STATE_RE = re.compile(r"^(?:epi|mfol|mol|moam|pvm)\d+$", re.I)
COMPACT_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
HIGH_CARDINALITY_SOURCE_THRESHOLD = 500
GENOMIC_REGION_RE = re.compile(
    r"^(?:intron|exon|promoter(?:[ _-]tss)?|tts|(?:3|5)[ _-]?utr|utr)\b", re.I
)
PATHWAY_PREFIX_RE = re.compile(
    r"^(?:go(?:bp|cc|mf)?|kegg|reactome|hallmark|biocarta|pid|wikipathways|wp)"
    r"(?:[ _.:/-]|$)",
    re.I,
)
IMMUNE_MODULE_RE = re.compile(r"^[A-Z0-9_() .+-]+\(M\d+(?:\.\d+)?\)$")
GENE_LIST_DESCRIPTION_RE = re.compile(
    r"\b(?:(?:marker|target|signature|differentially[ _-]?expressed|co[ _-]?expressed|"
    r"upregulated|downregulated|enriched) genes?|genes? (?:enriched|co[ _-]?expressed|"
    r"associated with))\b|^(?:proteins? with|autoantibody targets?)\b",
    re.I,
)
TABLE_OR_FIGURE_RE = re.compile(
    r"^(?:supplementary\s+)?(?:table|figure)\s+[A-Za-z0-9]|^fig(?:ure)?[ ._-]*[A-Za-z0-9]",
    re.I,
)
REVIEW_OR_UNRESOLVED_RE = re.compile(
    r"^review\s*:|\b(?:review needed|not explicitly (?:named|specified)|"
    r"cell type not (?:specified|named|resolved)|not a named cell type|"
    r"likely contamination|unnamed (?:cell type|cluster)|based on markers, but)\b",
    re.I,
)
SAMPLE_OR_TIMEPOINT_RE = re.compile(
    r"^(?:bulk\d+|patient[ _-]*[A-Za-z0-9.-]+$|tier\b|[<>]=?\s*\d|"
    r"\d{4,}\s+(?:aml|fl|mf|pv)\b|"
    r"\d+(?:\.\d+)?\s*(?:dpa|dpi|hpi|days?|weeks?|months?)\b)",
    re.I,
)
SYMBOL_ONLY_RE = re.compile(r"^[^A-Za-z0-9]+$")
SUPPLEMENTARY_DATA_RE = re.compile(r"^supplementary\s+(?:data|dataset|file)\s*\d*\b", re.I)
UNRESOLVED_CLUSTER_RE = re.compile(
    r"^(?:cluster(?:[_ -]?\d+)?|subcluster(?:[_ -]?\d+)?|cl\d+)(?:\s*\([^)]*\))?$",
    re.I,
)
DESCRIPTION_ONLY_RE = re.compile(
    r"\b(?:associated genes?|whole sample|cell lines?|score \(|score$|"
    r"cell type unknown|algorithm factor|only used in|without previous .* annotation)\b|"
    r"^(?:dna|rna) (?:biosynthesis|binding|replication|repair)\b",
    re.I,
)
COLOR_MODULE_LABELS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "green",
    "greenyellow",
    "grey",
    "gray",
    "lightcyan",
    "lightgreen",
    "lightyellow",
    "magenta",
    "midnightblue",
    "pink",
    "purple",
    "red",
    "royalblue",
    "salmon",
    "tan",
    "turquoise",
    "yellow",
}

CELL_TOKEN_WHITELIST = {
    "abc",
    "b",
    "bam",
    "bc",
    "bmdm",
    "caf",
    "cdc",
    "cdc1",
    "cdc2",
    "cdc3",
    "cd4",
    "cd8",
    "dc",
    "dc1",
    "dc2",
    "dc3",
    "endo",
    "hsc",
    "ilc",
    "ilc1",
    "ilc2",
    "ilc3",
    "mait",
    "m1",
    "m2",
    "mdc",
    "nk",
    "nkt",
    "neut",
    "oec",
    "opc",
    "pc",
    "pdc",
    "preb",
    "t",
    "tcm",
    "tem",
    "tfh",
    "th1",
    "th2",
    "th17",
    "tram1",
    "tram2",
    "treg",
    "trm",
    "vlmc1",
}

COMMON_CELL_LABEL_ALIASES = {
    "adipocyte",
    "alpha",
    "atrichoblast",
    "basal",
    "beta",
    "ciliated",
    "clp",
    "ctl",
    "dcs",
    "ec",
    "endoderm",
    "endodermis",
    "endothelial",
    "epithelial",
    "fdc",
    "fib",
    "gdt",
    "goblet",
    "hepatocytes",
    "immune",
    "lymphoid",
    "mac",
    "macrophages",
    "mesoderm",
    "mesenchyme",
    "mkp",
    "mono",
    "monocytes",
    "msc",
    "myeloid",
    "myofibroblasts",
    "neuron",
    "neutrophils",
    "npc",
    "opcs",
    "org",
    "pdc",
    "pdcs",
    "pericycle",
    "pericytes",
    "pgc",
    "phloem",
    "plasma",
    "rod",
    "rpe",
    "smc",
    "stromal",
    "trichoblast",
    "tuft",
    "xylem",
}

TECHNICAL_NAMES = {
    "avg_logfc",
    "avg_log2fc",
    "cell_cluster",
    "cluster",
    "feature",
    "features",
    "gene",
    "genes",
    "logfc",
    "log2fc",
    "label",
    "name",
    "names",
    "p_val",
    "p_val_adj",
    "padj",
    "pct.1",
    "pct.2",
    "pvalue",
    "p_value",
    "score",
    "scores",
}

PLACEHOLDER_NAMES = {
    "all",
    "doublet",
    "doublets",
    "high",
    "low",
    "mixed",
    "multiplet",
    "multiplets",
    "n/a",
    "na",
    "none",
    "not assigned",
    "other",
    "others",
    "rest",
    "review",
    "unassigned",
    "undefined",
    "unknown",
}

STANDALONE_NON_CELL_NAMES = {
    "anergy",
    "activated",
    "angiogenesis",
    "anti-inflammatory",
    "anti_inflammatory",
    "blood",
    "bone marrow",
    "brain",
    "case",
    "chronic activation",
    "control",
    "cycling",
    "cytotoxic",
    "cytotoxicity",
    "disease",
    "down",
    "effector",
    "emt",
    "exhaustion",
    "glycolysis",
    "granulopoiesis",
    "healthy",
    "hla",
    "homeostatic",
    "hypoxia",
    "iga",
    "igg",
    "interferon",
    "isg",
    "liver",
    "late",
    "lung",
    "memory",
    "mhc-ii",
    "mhcii",
    "mitotic",
    "naive",
    "normal",
    "phagocytosis",
    "pro-inflammatory",
    "pro_inflammatory",
    "proliferating",
    "proliferation",
    "skin",
    "scavenging",
    "spleen",
    "stemness",
    "treated",
    "tumor",
    "untreated",
    "up",
}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    path.chmod(0o644)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "cell_name" not in reader.fieldnames:
            raise ValueError(f"missing cell_name column: {path}")
        return list(reader.fieldnames), list(reader)


def collect_gene_vocabulary(papers_dir: Path) -> tuple[set[str], list[str]]:
    genes: set[str] = set()
    errors: list[str] = []
    for path in sorted(papers_dir.glob("*/agent_standardized/marker_statistics.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "gene" not in reader.fieldnames:
                    continue
                for row in reader:
                    gene = str(row.get("gene") or "").strip()
                    if gene:
                        genes.add(gene.casefold())
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    for path in sorted(papers_dir.glob("*/agent_standardized/marker_signatures.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, [])
                for row in reader:
                    for value in row:
                        gene = value.strip()
                        if gene:
                            genes.add(gene.casefold())
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    return genes, errors


def normalized_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def cell_label_variants(value: str) -> set[str]:
    label = normalized_label(value)
    if not label:
        return set()
    variants = {label}
    pending = [label]
    while pending:
        candidate = pending.pop()
        additions: list[str] = []
        if candidate.endswith(" cells"):
            additions.append(candidate[:-1])
        if candidate.endswith(" cell"):
            base = candidate[:-5].rstrip()
            if len(base) >= 4:
                additions.append(base)
        if candidate.endswith("s") and not candidate.endswith("ss"):
            additions.append(candidate[:-1])
        for addition in additions:
            if addition and addition not in variants:
                variants.add(addition)
                pending.append(addition)
    if " " not in label and len(label) >= 3:
        variants.add(f"{label}s")
        if label.endswith("y") and len(label) > 3:
            variants.add(f"{label[:-1]}ies")
    return variants


def collect_catalog_cell_vocabulary(path: Path) -> set[str]:
    protected = set(COMMON_CELL_LABEL_ALIASES) | set(CELL_TOKEN_WHITELIST)
    if not path.is_file():
        return protected
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            for field in ("catalog_label", "cl_name"):
                protected.update(cell_label_variants(str(row.get(field) or "")))
    return protected


def is_protected_cell_label(
    name: str, label: str, protected_cell_labels: set[str] | None
) -> bool:
    if label not in (protected_cell_labels or set()):
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", name)
    if compact.isupper() and len(compact) >= 3:
        return label in COMMON_CELL_LABEL_ALIASES or label in CELL_TOKEN_WHITELIST
    return True


def high_cardinality_sources(
    rows: list[dict[str, str]],
) -> tuple[set[str], dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for source in row.get("source_paths", "").split(";"):
            if source:
                counts[source] += 1
    suspicious = {
        source
        for source, count in counts.items()
        if count > HIGH_CARDINALITY_SOURCE_THRESHOLD
        and source.endswith(
            (
                "/agent_standardized/marker_statistics.csv",
                "/agent_standardized/marker_signatures.csv",
            )
        )
    }
    return suspicious, dict(counts)


def only_suspicious_sources(row: dict[str, str], suspicious_sources: set[str]) -> bool:
    sources = {value for value in row.get("source_paths", "").split(";") if value}
    return bool(sources and sources <= suspicious_sources)


def exclusion_reason(
    row: dict[str, str],
    genes: set[str],
    suspicious_sources: set[str] | None = None,
    protected_cell_labels: set[str] | None = None,
) -> str | None:
    name = str(row.get("cell_name") or "").strip()
    label = normalized_label(name)
    if not name:
        return "empty_name"
    if NUMERIC_RE.fullmatch(name) or label in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
        return "numeric_value"
    if SYMBOL_ONLY_RE.fullmatch(name):
        return "symbol_only_label"
    if only_suspicious_sources(row, suspicious_sources or set()):
        return "high_cardinality_source"
    if label in TECHNICAL_NAMES:
        return "technical_field_name"
    if label in PLACEHOLDER_NAMES:
        return "placeholder_or_qc_label"
    if COMPARISON_RE.search(name) or SEPARATED_COMPARISON_RE.search(name) or re.search(
        r"\b(?:differential(?:ly)? expressed genes?|degs?)\b", name, re.I
    ):
        return "comparison_or_deg_label"
    if GENOMIC_REGION_RE.search(name):
        return "genomic_region_label"
    if PATHWAY_PREFIX_RE.search(name) or IMMUNE_MODULE_RE.fullmatch(name):
        return "pathway_or_module_label"
    if GENE_LIST_DESCRIPTION_RE.search(name):
        return "gene_list_description"
    if TABLE_OR_FIGURE_RE.search(name):
        return "table_or_figure_description"
    if SUPPLEMENTARY_DATA_RE.search(name):
        return "table_or_figure_description"
    if REVIEW_OR_UNRESOLVED_RE.search(name):
        return "review_or_unresolved_label"
    if UNRESOLVED_CLUSTER_RE.fullmatch(name):
        return "review_or_unresolved_label"
    if DESCRIPTION_ONLY_RE.search(name):
        return "description_not_cell_name"
    if SAMPLE_OR_TIMEPOINT_RE.search(name):
        return "sample_or_timepoint_label"
    if label in COLOR_MODULE_LABELS or re.fullmatch(r"geneset[ _-]*[A-Za-z0-9]+", name, re.I):
        return "gene_set_program_or_signature"
    if BARCODE_RE.fullmatch(name):
        return "barcode_like"
    if DATE_OR_NUMERIC_CODE_RE.fullmatch(name):
        return "date_or_numeric_code"
    if (
        CLUSTER_RE.fullmatch(name)
        or SIMPLE_CODE_RE.fullmatch(name)
        or SEPARATED_CODE_RE.fullmatch(name)
        or LEADING_NUMBER_CODE_RE.fullmatch(name)
        or NUMBERED_STATE_RE.fullmatch(name)
    ):
        return "cluster_or_sample_code"
    if len(name) > 100:
        return "citation_or_description"
    if CITATION_RE.search(name):
        return "citation_or_description"
    if GENE_ACCESSION_RE.fullmatch(name):
        return "gene_accession_in_cell_cluster"
    if GENE_SET_RE.search(name) or (
        row.get("source_types") == "marker_signatures"
        and SIGNATURE_COUNT_RE.fullmatch(name)
    ):
        return "gene_set_program_or_signature"
    if QUALITY_LABEL_RE.search(name):
        return "qc_or_excluded_population"
    if TISSUE_PANEL_RE.search(name):
        return "tissue_or_anatomy_panel"
    if PROCESS_LABEL_RE.search(name) or DIRECTIONAL_SET_RE.search(name):
        return "process_or_directional_label"
    if label in STANDALONE_NON_CELL_NAMES:
        return "standalone_state_tissue_or_feature"
    if (
        COMPACT_TOKEN_RE.fullmatch(name)
        and label in genes
        and label not in CELL_TOKEN_WHITELIST
        and not is_protected_cell_label(name, label, protected_cell_labels)
    ):
        return "gene_symbol_in_cell_cluster"
    return None


def write_table(
    path: Path,
    input_fields: list[str],
    rows: list[dict[str, str]],
    *,
    rank_field: str,
    reason_field: str | None = None,
) -> None:
    fields = [rank_field, "original_rank"]
    if reason_field:
        fields.append(reason_field)
    fields.extend(field for field in input_fields if field != "rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            output = {rank_field: rank, "original_rank": row.get("rank", "")}
            if reason_field:
                output[reason_field] = row.get(reason_field, "")
            output.update({field: row.get(field, "") for field in input_fields if field != "rank"})
            writer.writerow(output)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    path.chmod(0o644)


def write_html(
    path: Path,
    filtered_path: Path,
    excluded_path: Path,
    rows: list[dict[str, str]],
    summary: dict[str, object],
) -> None:
    data = []
    for rank, row in enumerate(rows, 1):
        data.append(
            {
                "rank": rank,
                "name": row["cell_name"],
                "normalized": row.get("normalized_name", ""),
                "source": row.get("source_types", ""),
                "papers": int(row.get("paper_count") or 0),
                "files": int(row.get("standard_file_count") or 0),
                "rows": int(row.get("total_gene_rows") or 0),
                "paperIds": [value for value in row.get("paper_ids", "").split(";") if value],
                "examples": [
                    value for value in row.get("example_titles", "").split(" || ") if value
                ],
            }
        )
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filtered Agent Standardized Cell Names</title>
  <style>
    :root {{ color-scheme:light; --bg:#f3f5f8; --panel:#fff; --text:#172033; --muted:#667085; --line:#d7dde7; --accent:#245b9e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:18px 24px 14px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 12px; font-size:23px; letter-spacing:0; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:8px; }}
    .stat {{ padding:9px 11px; border:1px solid var(--line); border-radius:7px; background:#fafbfd; }}
    .stat b {{ display:block; font-size:19px; }} .stat span {{ color:var(--muted); font-size:12px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px,1fr) 190px 160px 210px auto auto; gap:8px; padding:11px 24px; background:#e8edf3; border-bottom:1px solid var(--line); }}
    input,select,a {{ height:36px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--text); font:inherit; }}
    input,select {{ padding:0 10px; min-width:0; }} a {{ display:flex; align-items:center; justify-content:center; padding:0 11px; color:var(--accent); text-decoration:none; white-space:nowrap; }}
    main {{ padding:14px 24px 26px; }} .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .table-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 13px; border-bottom:1px solid var(--line); }}
    .table-status {{ display:flex; align-items:center; gap:8px; color:var(--muted); }} button {{ height:30px; border:1px solid var(--line); border-radius:6px; background:var(--panel); cursor:pointer; }} button:disabled {{ color:#a1a7b0; cursor:not-allowed; }}
    .table-wrap {{ overflow:auto; max-height:calc(100vh - 246px); min-height:520px; }} table {{ width:100%; border-collapse:collapse; }}
    th {{ position:sticky; top:0; z-index:1; background:#f7f8fb; color:#465166; text-align:left; font-size:12px; padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
    td {{ padding:8px 10px; border-bottom:1px solid #edf0f4; vertical-align:top; }} tr:hover {{ background:#f3f7fd; }}
    .num {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }} .name {{ min-width:250px; font-weight:650; }} .examples {{ min-width:360px; max-width:620px; color:var(--muted); }}
    mark {{ background:#fff0a6; color:inherit; }} .empty {{ padding:40px; text-align:center; color:var(--muted); }}
    @media(max-width:1000px) {{ .toolbar {{ grid-template-columns:1fr 1fr; }} }}
    @media(max-width:640px) {{ header,.toolbar,main {{ padding-left:12px; padding-right:12px; }} .stats {{ grid-template-columns:1fr 1fr; }} .toolbar {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header><h1>Filtered Agent Standardized Cell Names</h1><div class="stats" id="stats"></div></header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="搜索细胞名称、论文标题或论文 ID">
    <select id="source"><option value="all">全部来源类型</option><option value="both">两类表均出现</option><option value="marker_statistics">marker_statistics</option><option value="marker_signatures">marker_signatures</option></select>
    <select id="minPapers"><option value="0">不限论文数</option><option value="2">至少 2 篇论文</option><option value="5">至少 5 篇论文</option><option value="10">至少 10 篇论文</option></select>
    <select id="sort"><option value="frequency">按论文数排序</option><option value="name">按名称排序</option><option value="rows">按基因记录数排序</option></select>
    <a href="{html.escape(filtered_path.name)}" download>保留 TSV</a><a href="{html.escape(excluded_path.name)}" download>排除 TSV</a>
  </div>
  <main><section class="panel"><div class="table-head"><b>高置信细胞类型候选</b><div class="table-status"><span id="shown"></span><button id="previous">上一页</button><span id="pageInfo"></span><button id="next">下一页</button></div></div><div class="table-wrap"><table><thead><tr><th>#</th><th>Cell name</th><th>Source</th><th>Papers</th><th>Files</th><th>Gene rows</th><th>Example papers</th></tr></thead><tbody id="body"></tbody></table></div></section></main>
  <script>
    const DATA={data_json}; const SUMMARY={summary_json}; const number=new Intl.NumberFormat("en-US"); const pageSize=250;
    let query="",source="all",minPapers=0,sortMode="frequency",page=1;
    function esc(value) {{ return String(value).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }}
    function rows() {{ const values=DATA.filter(row=>{{const text=[row.name,row.normalized,...row.paperIds,...row.examples].join(" ").toLowerCase();return(!query||text.includes(query))&&(source==="all"||row.source===source)&&row.papers>=minPapers;}}); if(sortMode==="name")values.sort((a,b)=>a.normalized.localeCompare(b.normalized)||a.name.localeCompare(b.name));else if(sortMode==="rows")values.sort((a,b)=>b.rows-a.rows||b.papers-a.papers);else values.sort((a,b)=>b.papers-a.papers||b.files-a.files||a.normalized.localeCompare(b.normalized));return values; }}
    function renderStats() {{ const values=[[SUMMARY.input_names,"Input names"],[SUMMARY.kept_names,"Kept"],[SUMMARY.excluded_names,"Excluded"],[SUMMARY.gene_vocabulary_size,"Gene vocabulary"]];document.getElementById("stats").innerHTML=values.map(v=>'<div class="stat"><b>'+number.format(v[0])+'</b><span>'+v[1]+'</span></div>').join(""); }}
    function render() {{ const values=rows(),pages=Math.max(1,Math.ceil(values.length/pageSize));page=Math.min(page,pages);const start=(page-1)*pageSize,visible=values.slice(start,start+pageSize);document.getElementById("shown").textContent=number.format(values.length)+" / "+number.format(DATA.length);document.getElementById("pageInfo").textContent=page+" / "+pages;document.getElementById("previous").disabled=page===1;document.getElementById("next").disabled=page===pages;document.getElementById("body").innerHTML=visible.length?visible.map((row,index)=>'<tr><td class="num">'+number.format(start+index+1)+'</td><td class="name">'+esc(row.name)+'</td><td>'+esc(row.source)+'</td><td class="num">'+number.format(row.papers)+'</td><td class="num">'+number.format(row.files)+'</td><td class="num">'+number.format(row.rows)+'</td><td class="examples">'+row.examples.map(esc).join("<br>")+'</td></tr>').join(""):'<tr><td class="empty" colspan="7">没有匹配名称</td></tr>'; }}
    document.getElementById("search").addEventListener("input",e=>{{query=e.target.value.trim().toLowerCase();page=1;render();}});document.getElementById("source").addEventListener("change",e=>{{source=e.target.value;page=1;render();}});document.getElementById("minPapers").addEventListener("change",e=>{{minPapers=Number(e.target.value);page=1;render();}});document.getElementById("sort").addEventListener("change",e=>{{sortMode=e.target.value;page=1;render();}});document.getElementById("previous").addEventListener("click",()=>{{if(page>1){{page--;render();}}}});document.getElementById("next").addEventListener("click",()=>{{page++;render();}});renderStats();render();
  </script>
</body></html>
"""
    atomic_text(path, document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--filtered", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()

    input_fields, rows = read_rows(args.input.resolve())
    genes, errors = collect_gene_vocabulary(args.papers_dir.resolve())
    protected_cell_labels = collect_catalog_cell_vocabulary(args.catalog.resolve())
    suspicious_sources, source_counts = high_cardinality_sources(rows)
    kept: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        reason = exclusion_reason(
            row, genes, suspicious_sources, protected_cell_labels
        )
        if reason:
            rejected = dict(row)
            rejected["exclusion_reason"] = reason
            excluded.append(rejected)
            reasons[reason] += 1
        else:
            kept.append(row)

    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input.resolve().relative_to(ROOT)),
        "input_names": len(rows),
        "kept_names": len(kept),
        "excluded_names": len(excluded),
        "kept_fraction": round(len(kept) / len(rows), 6) if rows else 0,
        "gene_vocabulary_size": len(genes),
        "protected_cell_vocabulary_size": len(protected_cell_labels),
        "catalog_path": str(args.catalog.resolve().relative_to(ROOT)),
        "high_cardinality_source_threshold": HIGH_CARDINALITY_SOURCE_THRESHOLD,
        "high_cardinality_source_files": [
            {"path": source, "unique_names": source_counts[source]}
            for source in sorted(
                suspicious_sources, key=lambda value: (-source_counts[value], value)
            )
        ],
        "exclusion_counts": dict(sorted(reasons.items())),
        "gene_scan_errors": errors,
    }
    write_table(args.filtered.resolve(), input_fields, kept, rank_field="filtered_rank")
    write_table(
        args.excluded.resolve(),
        input_fields,
        excluded,
        rank_field="excluded_rank",
        reason_field="exclusion_reason",
    )
    write_html(args.html.resolve(), args.filtered.resolve(), args.excluded.resolve(), kept, summary)
    atomic_text(
        args.summary.resolve(),
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
