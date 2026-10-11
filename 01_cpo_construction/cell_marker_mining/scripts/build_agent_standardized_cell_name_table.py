#!/usr/bin/env python3
"""Summarize cell names from all agent-standardized marker tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_DIR = ROOT / "database" / "papers"
DEFAULT_TSV = ROOT / "database" / "agent_standardized_cell_names.tsv"
DEFAULT_HTML = ROOT / "database" / "agent_standardized_cell_names.html"
DEFAULT_SUMMARY = ROOT / "database" / "agent_standardized_cell_names_summary.json"


@dataclass
class NameRecord:
    cell_name: str
    normalized_name: str
    statistics_gene_rows: int = 0
    signature_gene_rows: int = 0
    statistics_files: set[str] = field(default_factory=set)
    signature_files: set[str] = field(default_factory=set)
    papers: dict[str, str] = field(default_factory=dict)
    source_paths: set[str] = field(default_factory=set)

    @property
    def paper_count(self) -> int:
        return len(self.papers)

    @property
    def standard_file_count(self) -> int:
        return len(self.statistics_files | self.signature_files)

    @property
    def total_gene_rows(self) -> int:
        return self.statistics_gene_rows + self.signature_gene_rows

    @property
    def source_types(self) -> str:
        if self.statistics_files and self.signature_files:
            return "both"
        if self.statistics_files:
            return "marker_statistics"
        return "marker_signatures"


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def paper_context(record_dir: Path) -> tuple[str, str]:
    metadata = read_json(record_dir / "metadata.json")
    paper_id = str(metadata.get("id") or record_dir.name)
    title = str(metadata.get("title") or "")
    return paper_id, title


def record_for(records: dict[str, NameRecord], cell_name: str) -> NameRecord:
    if cell_name not in records:
        records[cell_name] = NameRecord(cell_name, normalize_name(cell_name))
    return records[cell_name]


def collect_statistics(
    path: Path,
    relative_path: str,
    paper_id: str,
    title: str,
    records: dict[str, NameRecord],
) -> tuple[int, set[str]]:
    gene_rows = 0
    names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "cell_cluster" not in reader.fieldnames:
            return 0, names
        for row in reader:
            cell_name = str(row.get("cell_cluster") or "").strip()
            gene = str(row.get("gene") or "").strip()
            if not cell_name:
                continue
            names.add(cell_name)
            record = record_for(records, cell_name)
            record.statistics_files.add(relative_path)
            record.source_paths.add(relative_path)
            record.papers[paper_id] = title
            if gene:
                record.statistics_gene_rows += 1
                gene_rows += 1
    return gene_rows, names


def collect_signatures(
    path: Path,
    relative_path: str,
    paper_id: str,
    title: str,
    records: dict[str, NameRecord],
) -> tuple[int, set[str]]:
    gene_rows = 0
    names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        columns = [(index, value.strip()) for index, value in enumerate(header) if value.strip()]
        if not columns:
            return 0, names
        for _, cell_name in columns:
            names.add(cell_name)
            record = record_for(records, cell_name)
            record.signature_files.add(relative_path)
            record.source_paths.add(relative_path)
            record.papers[paper_id] = title
        for row in reader:
            for index, cell_name in columns:
                if index >= len(row) or not row[index].strip():
                    continue
                record_for(records, cell_name).signature_gene_rows += 1
                gene_rows += 1
    return gene_rows, names


def iter_record_dirs(papers_dir: Path) -> Iterable[Path]:
    for record_dir in sorted(papers_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        output_dir = record_dir / "agent_standardized"
        if (output_dir / "marker_statistics.csv").is_file() or (
            output_dir / "marker_signatures.csv"
        ).is_file():
            yield record_dir


def collect(papers_dir: Path) -> tuple[list[NameRecord], dict[str, object]]:
    records: dict[str, NameRecord] = {}
    errors: list[str] = []
    paper_dirs = 0
    statistics_files = 0
    signature_files = 0
    nonempty_statistics_files = 0
    nonempty_signature_files = 0
    statistics_gene_rows = 0
    signature_gene_rows = 0

    for record_dir in iter_record_dirs(papers_dir):
        paper_dirs += 1
        paper_id, title = paper_context(record_dir)
        output_dir = record_dir / "agent_standardized"
        statistics_path = output_dir / "marker_statistics.csv"
        signatures_path = output_dir / "marker_signatures.csv"
        if statistics_path.is_file():
            statistics_files += 1
            relative_path = str(statistics_path.relative_to(ROOT))
            try:
                row_count, names = collect_statistics(
                    statistics_path, relative_path, paper_id, title, records
                )
                statistics_gene_rows += row_count
                if names:
                    nonempty_statistics_files += 1
            except (OSError, UnicodeError, csv.Error) as exc:
                errors.append(f"{relative_path}: {type(exc).__name__}: {exc}")
        if signatures_path.is_file():
            signature_files += 1
            relative_path = str(signatures_path.relative_to(ROOT))
            try:
                row_count, names = collect_signatures(
                    signatures_path, relative_path, paper_id, title, records
                )
                signature_gene_rows += row_count
                if names:
                    nonempty_signature_files += 1
            except (OSError, UnicodeError, csv.Error) as exc:
                errors.append(f"{relative_path}: {type(exc).__name__}: {exc}")

    ordered = sorted(
        records.values(),
        key=lambda item: (-item.paper_count, -item.standard_file_count, item.normalized_name, item.cell_name),
    )
    statistics_names = {name for name, record in records.items() if record.statistics_files}
    signature_names = {name for name, record in records.items() if record.signature_files}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "papers_with_agent_standardized_outputs": paper_dirs,
        "statistics_files": statistics_files,
        "signature_files": signature_files,
        "nonempty_statistics_files": nonempty_statistics_files,
        "nonempty_signature_files": nonempty_signature_files,
        "unique_cell_names": len(records),
        "statistics_unique_cell_names": len(statistics_names),
        "signature_unique_cell_names": len(signature_names),
        "names_in_both_table_types": len(statistics_names & signature_names),
        "statistics_gene_rows": statistics_gene_rows,
        "signature_gene_rows": signature_gene_rows,
        "total_gene_rows": statistics_gene_rows + signature_gene_rows,
        "errors": errors,
    }
    return ordered, summary


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    path.chmod(0o644)


def write_tsv(path: Path, records: list[NameRecord]) -> None:
    fields = [
        "rank",
        "cell_name",
        "normalized_name",
        "source_types",
        "paper_count",
        "standard_file_count",
        "statistics_file_count",
        "signature_file_count",
        "statistics_gene_rows",
        "signature_gene_rows",
        "total_gene_rows",
        "paper_ids",
        "example_titles",
        "source_paths",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            paper_items = sorted(record.papers.items())
            writer.writerow(
                {
                    "rank": rank,
                    "cell_name": record.cell_name,
                    "normalized_name": record.normalized_name,
                    "source_types": record.source_types,
                    "paper_count": record.paper_count,
                    "standard_file_count": record.standard_file_count,
                    "statistics_file_count": len(record.statistics_files),
                    "signature_file_count": len(record.signature_files),
                    "statistics_gene_rows": record.statistics_gene_rows,
                    "signature_gene_rows": record.signature_gene_rows,
                    "total_gene_rows": record.total_gene_rows,
                    "paper_ids": ";".join(paper_id for paper_id, _ in paper_items),
                    "example_titles": " || ".join(
                        title for _, title in paper_items[:5] if title
                    ),
                    "source_paths": ";".join(sorted(record.source_paths)),
                }
            )
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    path.chmod(0o644)


def html_data(records: list[NameRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, record in enumerate(records, 1):
        paper_items = sorted(record.papers.items())
        rows.append(
            {
                "rank": rank,
                "name": record.cell_name,
                "normalized": record.normalized_name,
                "source": record.source_types,
                "papers": record.paper_count,
                "files": record.standard_file_count,
                "statisticsFiles": len(record.statistics_files),
                "signatureFiles": len(record.signature_files),
                "statisticsRows": record.statistics_gene_rows,
                "signatureRows": record.signature_gene_rows,
                "rows": record.total_gene_rows,
                "paperIds": [paper_id for paper_id, _ in paper_items],
                "examples": [title for _, title in paper_items[:5] if title],
            }
        )
    return rows


def write_html(
    path: Path,
    tsv_path: Path,
    records: list[NameRecord],
    summary: dict[str, object],
) -> None:
    data_json = json.dumps(html_data(records), ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    download_name = html.escape(tsv_path.name)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Standardized Cell Names</title>
  <style>
    :root {{ color-scheme: light; --bg:#f4f6fa; --panel:#fff; --text:#172033; --muted:#667085; --line:#d8dee9; --accent:#315ea8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:20px 24px 16px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 12px; font-size:24px; }}
    .stats {{ display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:9px; }}
    .stat {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fbfcfe; }}
    .stat b {{ display:block; font-size:20px; }}
    .stat span {{ color:var(--muted); font-size:12px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(260px,1fr) 190px 160px 230px auto; gap:8px; padding:12px 24px; background:#e9edf4; border-bottom:1px solid var(--line); }}
    input,select,a.button {{ height:36px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--text); font:inherit; }}
    input,select {{ padding:0 10px; min-width:0; }}
    a.button {{ display:flex; align-items:center; justify-content:center; padding:0 12px; color:var(--accent); text-decoration:none; white-space:nowrap; }}
    main {{ padding:14px 24px 28px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; overflow:hidden; box-shadow:0 10px 28px rgba(20,28,42,.07); }}
    .table-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px 14px; border-bottom:1px solid var(--line); }}
    .table-head b {{ font-size:15px; }}
    .table-head span {{ color:var(--muted); }}
    .table-status {{ display:flex; align-items:center; gap:8px; }}
    .page-button {{ height:30px; border:1px solid var(--line); border-radius:6px; padding:0 10px; background:var(--panel); color:var(--text); cursor:pointer; }}
    .page-button:disabled {{ color:#a0a7b3; cursor:not-allowed; background:#f5f6f8; }}
    .table-wrap {{ overflow:auto; max-height:calc(100vh - 258px); min-height:520px; }}
    table {{ width:100%; border-collapse:collapse; }}
    thead th {{ position:sticky; top:0; z-index:1; background:#f6f8fb; color:#465166; text-align:left; font-size:12px; white-space:nowrap; border-bottom:1px solid var(--line); padding:9px 10px; }}
    tbody td {{ border-bottom:1px solid #edf0f4; padding:8px 10px; vertical-align:top; }}
    tbody tr:hover {{ background:#f4f8ff; }}
    td.name {{ font-weight:650; min-width:250px; }}
    td.number {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    td.examples {{ color:var(--muted); min-width:360px; max-width:620px; }}
    .badge {{ display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:2px 7px; color:var(--muted); background:#f8fafc; white-space:nowrap; font-size:12px; }}
    mark {{ background:#fff0a6; color:inherit; padding:0 1px; }}
    .empty {{ padding:40px; text-align:center; color:var(--muted); }}
    @media (max-width:1000px) {{ .stats {{ grid-template-columns:repeat(3,1fr); }} .toolbar {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:640px) {{ header,.toolbar,main {{ padding-left:12px; padding-right:12px; }} .stats {{ grid-template-columns:repeat(2,1fr); }} .toolbar {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Agent Standardized Cell Names</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="搜索细胞名称、论文标题或论文 ID">
    <select id="source"><option value="all">全部来源类型</option><option value="both">两类表均出现</option><option value="marker_statistics">marker_statistics</option><option value="marker_signatures">marker_signatures</option></select>
    <select id="minPapers"><option value="0">不限论文数</option><option value="2">至少 2 篇论文</option><option value="5">至少 5 篇论文</option><option value="10">至少 10 篇论文</option></select>
    <select id="sort"><option value="frequency">按论文数排序</option><option value="name">按名称排序</option><option value="rows">按基因记录数排序</option></select>
    <a class="button" href="{download_name}" download>下载 TSV</a>
  </div>
  <main>
    <section class="panel">
      <div class="table-head"><b>唯一原始细胞名称</b><div class="table-status"><span id="shown"></span><button class="page-button" id="previous" type="button">上一页</button><span id="pageInfo"></span><button class="page-button" id="next" type="button">下一页</button></div></div>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>Cell name</th><th>Source</th><th>Papers</th><th>Files</th><th>Statistics genes</th><th>Signature genes</th><th>Total genes</th><th>Example papers</th></tr></thead><tbody id="body"></tbody></table></div>
    </section>
  </main>
  <script>
    const DATA={data_json};
    const SUMMARY={summary_json};
    const number=new Intl.NumberFormat("en-US");
    let query="";
    let source="all";
    let minPapers=0;
    let sortMode="frequency";
    let page=1;
    const pageSize=250;
    function escapeHtml(value) {{ return String(value).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }}
    function highlight(value) {{
      const safe=escapeHtml(value||"");
      if(!query) return safe;
      const lower=safe.toLowerCase();
      let result="", cursor=0, index=lower.indexOf(query);
      while(index>=0) {{
        result+=safe.slice(cursor,index)+"<mark>"+safe.slice(index,index+query.length)+"</mark>";
        cursor=index+query.length;
        index=lower.indexOf(query,cursor);
      }}
      return result+safe.slice(cursor);
    }}
    function renderStats() {{ const values=[[SUMMARY.unique_cell_names,"Unique names"],[SUMMARY.papers_with_agent_standardized_outputs,"Papers scanned"],[SUMMARY.nonempty_statistics_files,"Statistics files"],[SUMMARY.nonempty_signature_files,"Signature files"],[SUMMARY.names_in_both_table_types,"Names in both"],[SUMMARY.total_gene_rows,"Gene rows"]]; document.getElementById("stats").innerHTML=values.map(item=>'<div class="stat"><b>'+number.format(item[0])+'</b><span>'+item[1]+'</span></div>').join(""); }}
    function filteredRows() {{ const rows=DATA.filter(row=>{{ const text=[row.name,row.normalized,...row.paperIds,...row.examples].join(" ").toLowerCase(); return (!query||text.includes(query))&&(source==="all"||row.source===source)&&row.papers>=minPapers; }}); if(sortMode==="name") rows.sort((a,b)=>a.normalized.localeCompare(b.normalized)||a.name.localeCompare(b.name)); else if(sortMode==="rows") rows.sort((a,b)=>b.rows-a.rows||b.papers-a.papers||a.normalized.localeCompare(b.normalized)); else rows.sort((a,b)=>b.papers-a.papers||b.files-a.files||a.normalized.localeCompare(b.normalized)); return rows; }}
    function render() {{ const rows=filteredRows(); const pageCount=Math.max(1,Math.ceil(rows.length/pageSize)); page=Math.min(page,pageCount); const start=(page-1)*pageSize; const visible=rows.slice(start,start+pageSize); document.getElementById("shown").textContent=number.format(rows.length)+" / "+number.format(DATA.length)+" names"; document.getElementById("pageInfo").textContent=number.format(page)+" / "+number.format(pageCount); document.getElementById("previous").disabled=page<=1; document.getElementById("next").disabled=page>=pageCount; const body=document.getElementById("body"); if(!visible.length) {{ body.innerHTML='<tr><td class="empty" colspan="9">没有匹配名称</td></tr>'; return; }} body.innerHTML=visible.map((row,index)=>'<tr><td class="number">'+number.format(start+index+1)+'</td><td class="name">'+highlight(row.name)+'</td><td><span class="badge">'+escapeHtml(row.source)+'</span></td><td class="number">'+number.format(row.papers)+'</td><td class="number">'+number.format(row.files)+'</td><td class="number">'+number.format(row.statisticsRows)+'</td><td class="number">'+number.format(row.signatureRows)+'</td><td class="number">'+number.format(row.rows)+'</td><td class="examples">'+row.examples.map(highlight).join("<br>")+'</td></tr>').join(""); }}
    document.getElementById("search").addEventListener("input",event=>{{ query=event.target.value.trim().toLowerCase(); page=1; render(); }});
    document.getElementById("source").addEventListener("change",event=>{{ source=event.target.value; page=1; render(); }});
    document.getElementById("minPapers").addEventListener("change",event=>{{ minPapers=Number(event.target.value); page=1; render(); }});
    document.getElementById("sort").addEventListener("change",event=>{{ sortMode=event.target.value; page=1; render(); }});
    document.getElementById("previous").addEventListener("click",()=>{{ if(page>1) {{ page-=1; render(); }} }});
    document.getElementById("next").addEventListener("click",()=>{{ page+=1; render(); }});
    renderStats(); render();
  </script>
</body>
</html>
"""
    atomic_text(path, document)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a searchable inventory of agent-standardized cell names."
    )
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    records, summary = collect(args.papers_dir.resolve())
    write_tsv(args.tsv.resolve(), records)
    write_html(args.html.resolve(), args.tsv.resolve(), records, summary)
    atomic_text(
        args.summary.resolve(),
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
