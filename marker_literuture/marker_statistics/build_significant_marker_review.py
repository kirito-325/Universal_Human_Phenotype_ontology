#!/usr/bin/env python3
"""Build an offline review page containing significant marker genes only."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MARKER_DIR = ROOT / "marker_statistics"
DEFAULT_MARKERS = MARKER_DIR / "significant_marker_scores.tsv"
DEFAULT_CELLS = MARKER_DIR / "significant_marker_cell_summary.tsv"
DEFAULT_SUMMARY = MARKER_DIR / "significant_marker_summary.json"
DEFAULT_OUTPUT = MARKER_DIR / "significant_marker_scores_review.html"


def integer(row: dict[str, str], field: str) -> int:
    return int(str(row.get(field) or "0"))


def number(row: dict[str, str], field: str) -> float:
    return float(str(row.get(field) or "0"))


def load_cells(path: Path) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"catalog_id", "significant_genes", "tested_observed_genes"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid significant-cell summary columns: {path}")
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            significant_genes = integer(row, "significant_genes")
            if not catalog_id or significant_genes < 1:
                continue
            cells[catalog_id] = {
                "order": integer(row, "catalog_order"),
                "id": catalog_id,
                "label": str(row.get("catalog_label") or ""),
                "clName": str(row.get("catalog_cl_name") or ""),
                "labelZh": str(row.get("catalog_label_zh") or ""),
                "parent": str(row.get("parent_label") or ""),
                "branch": str(row.get("branch") or ""),
                "catalogRank": str(row.get("catalog_rank") or ""),
                "opportunities": integer(row, "cell_evidence_opportunities"),
                "testedGenes": integer(row, "tested_observed_genes"),
                "significantGenes": significant_genes,
                "significantFraction": number(row, "significant_fraction"),
                "maxMarkerScore": integer(row, "max_marker_score"),
                "maxSignificance": number(row, "max_significance_score"),
                "markers": [],
            }
    return cells


def load_markers(path: Path, cells: dict[str, dict[str, Any]]) -> int:
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"catalog_id", "gene", "marker_score", "q_value", "significance_score"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid significant marker columns: {path}")
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            if catalog_id not in cells:
                raise ValueError(f"significant marker cell absent from cell summary: {catalog_id}")
            cells[catalog_id]["markers"].append(
                {
                    "rank": integer(row, "significant_rank"),
                    "originalRank": integer(row, "original_marker_rank"),
                    "gene": str(row.get("gene") or ""),
                    "score": integer(row, "marker_score"),
                    "statistics": integer(row, "statistics_score"),
                    "signatures": integer(row, "signatures_score"),
                    "cellPrevalence": number(row, "cell_prevalence"),
                    "backgroundPrevalence": number(row, "background_prevalence"),
                    "fold": number(row, "enrichment_fold"),
                    "odds": number(row, "odds_ratio"),
                    "p": number(row, "p_value"),
                    "q": number(row, "q_value"),
                    "significance": number(row, "significance_score"),
                }
            )
            rows += 1
    for cell in cells.values():
        cell["markers"].sort(key=lambda marker: marker["rank"])
        if len(cell["markers"]) != cell["significantGenes"]:
            raise ValueError(f"significant marker count mismatch for {cell['id']}")
    return rows


def build_data(cell_summary: Path, markers: Path) -> list[dict[str, Any]]:
    cells = load_cells(cell_summary)
    load_markers(markers, cells)
    return sorted(cells.values(), key=lambda cell: (cell["order"], cell["id"]))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Significant Expanded Cell Markers</title>
  <style>
    :root { color-scheme:light; --bg:#edf2f5; --surface:#fff; --surface2:#f6f8fa; --text:#17212b; --muted:#65717e; --line:#d3dce4; --green:#16805c; --green-soft:#e2f3ec; --amber:#d88919; --blue:#256ca8; --blue-soft:#dceaf5; --red:#ad4343; }
    * { box-sizing:border-box; } html,body { margin:0; min-height:100%; } body { background:var(--bg); color:var(--text); font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; } button,input,select { font:inherit; color:inherit; } button { cursor:pointer; }
    button:focus-visible,input:focus-visible,select:focus-visible { outline:2px solid var(--blue); outline-offset:1px; }
    header { padding:16px 20px 13px; background:var(--surface); border-bottom:1px solid var(--line); } .title-row { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; } h1 { margin:0 0 4px; font-size:21px; } .method { max-width:920px; color:var(--muted); } .links { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:6px; } .links a { display:inline-flex; height:32px; align-items:center; padding:0 10px; border:1px solid var(--line); border-radius:6px; color:var(--blue); background:#fff; text-decoration:none; white-space:nowrap; }
    .headline { display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); margin-top:12px; border:1px solid var(--line); background:var(--line); gap:1px; } .headline div { min-width:0; padding:8px 11px; background:var(--surface2); } .headline b { display:block; font-size:17px; overflow-wrap:anywhere; } .headline span { color:var(--muted); font-size:11px; }
    .toolbar { display:grid; grid-template-columns:minmax(260px,1fr) 190px 210px; gap:8px; padding:9px 20px; background:#e3e9ee; border-bottom:1px solid var(--line); } input,select { width:100%; height:35px; min-width:0; padding:0 10px; background:#fff; border:1px solid #bcc8d2; border-radius:6px; }
    .workspace { display:grid; grid-template-columns:330px minmax(0,1fr); width:100%; min-width:0; height:calc(100vh - 198px); min-height:600px; } aside { display:flex; flex-direction:column; min-width:0; overflow:hidden; background:var(--surface); border-right:1px solid var(--line); } .list-head { display:flex; justify-content:space-between; padding:9px 12px; color:var(--muted); border-bottom:1px solid var(--line); } #cellList { overflow:auto; }
    .cell-button { width:100%; padding:9px 11px 9px 9px; text-align:left; background:#fff; border:0; border-bottom:1px solid #e6ebef; border-left:4px solid transparent; } .cell-button:hover { background:#f5f8fa; } .cell-button.active { border-left-color:var(--green); background:var(--green-soft); } .cell-main { display:flex; justify-content:space-between; align-items:baseline; gap:7px; } .cell-label { min-width:0; font-weight:650; overflow-wrap:anywhere; } .cell-count { color:var(--green); white-space:nowrap; font-variant-numeric:tabular-nums; } .cell-meta { display:block; margin-top:2px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
    .detail { min-width:0; overflow:auto; background:var(--surface); } .detail-head { padding:14px 18px 11px; border-bottom:1px solid var(--line); } .detail-title { display:flex; flex-wrap:wrap; align-items:baseline; gap:7px; } h2 { margin:0; font-size:20px; } .cl-id { color:var(--blue); font:12px ui-monospace,SFMono-Regular,Menlo,monospace; } .detail-meta { margin-top:4px; color:var(--muted); }
    .cell-stats { display:grid; grid-template-columns:repeat(6,minmax(90px,1fr)); background:var(--surface2); border-bottom:1px solid var(--line); } .cell-stat { min-width:0; padding:8px 11px; border-right:1px solid var(--line); } .cell-stat:last-child { border-right:0; } .cell-stat b { display:block; font-size:16px; overflow-wrap:anywhere; } .cell-stat span { color:var(--muted); font-size:10px; }
    .marker-toolbar { display:grid; grid-template-columns:minmax(180px,1fr) 160px 160px 180px; gap:8px; padding:9px 18px; border-bottom:1px solid var(--line); } .marker-status { padding:7px 18px; color:var(--muted); border-bottom:1px solid #e8edf1; }
    .table-wrap { width:100%; max-width:100%; overflow:auto; } table { width:100%; min-width:1050px; border-collapse:collapse; table-layout:fixed; } th { position:sticky; top:0; z-index:2; padding:8px 8px; text-align:left; color:#4d5965; background:#f6f8fa; border-bottom:1px solid var(--line); } td { padding:7px 8px; border-bottom:1px solid #edf0f2; } tr:hover { background:#f8fbfa; }
    .rank { width:58px; text-align:right; color:var(--muted); } .gene { width:130px; font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; } .evidence { width:230px; } .prevalence { width:145px; } .num { width:86px; text-align:right; font-variant-numeric:tabular-nums; } .fdr { width:100px; text-align:right; font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }
    .bar-track { display:flex; width:100%; height:11px; overflow:hidden; background:#e7edf1; } .bar-stat { height:100%; background:var(--green); } .bar-sig { height:100%; background:var(--amber); } .sig-track { width:100%; height:6px; margin-top:3px; background:#e7edf1; } .sig-value { height:100%; background:var(--blue); } .sub { display:block; margin-top:2px; color:var(--muted); font-size:10px; } .empty { padding:35px; text-align:center; color:var(--muted); }
    @media(max-width:1050px) { .headline { grid-template-columns:repeat(3,1fr); } .toolbar { grid-template-columns:1fr 1fr; } .toolbar #globalSearch { grid-column:1/-1; } .workspace { height:auto; min-height:0; grid-template-columns:minmax(0,1fr); } aside { max-height:360px; border-right:0; border-bottom:1px solid var(--line); } .detail { overflow:visible; } .cell-stats { grid-template-columns:repeat(3,1fr); } }
    @media(max-width:620px) { header,.toolbar { padding-left:12px; padding-right:12px; } .title-row { display:block; } .links { justify-content:flex-start; margin-top:9px; } .headline { grid-template-columns:repeat(2,1fr); } .toolbar,.marker-toolbar { grid-template-columns:1fr; } .toolbar #globalSearch { grid-column:auto; } .cell-stats { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
  <header><div class="title-row"><div><h1>显著 Expanded Cell Marker</h1><div class="method">单侧超几何富集；每个标准细胞内进行 BH-FDR 校正。仅保留 FDR ≤ 0.05、marker 分数 ≥ 3 且优势比 &gt; 1 的基因。</div></div><nav class="links"><a href="expanded_marker_scores_review.html">全部 Marker</a><a href="significant_marker_scores.tsv" download>显著结果 TSV</a><a href="significant_marker_cell_summary.tsv" download>细胞摘要</a></nav></div><div class="headline" id="headline"></div></header>
  <div class="toolbar"><input id="globalSearch" type="search" placeholder="搜索标准细胞、CL ID 或显著 marker"><select id="branchFilter"><option value="">全部 expanded 分支</option></select><select id="cellSort"><option value="count">按显著 marker 数排序</option><option value="significance">按最高显著性排序</option><option value="opportunities">按证据机会数排序</option><option value="catalog">按 catalog 顺序</option></select></div>
  <main class="workspace"><aside><div class="list-head"><span>有显著 marker 的细胞</span><span id="cellCount"></span></div><div id="cellList"></div></aside><section class="detail" id="detail"></section></main>
  <script>
    const DATA=__DATA__; const META=__META__; const fmt=new Intl.NumberFormat("en-US");let query="",branch="",cellSort="count",selectedId="",geneQuery="",maxQ=.05,minScore=3,markerSort="significance";
    const esc=v=>String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");const sci=v=>Number(v).toExponential(2);const pct=v=>(Number(v)*100).toFixed(1)+"%";
    const index=new Map(DATA.map(c=>[c.id,[c.id,c.label,c.clName,c.labelZh,c.parent,c.branch,...c.markers.map(m=>m.gene)].join(" ").toLowerCase()]));
    function cells(){let values=DATA.filter(c=>(!branch||c.branch===branch)&&(!query||index.get(c.id).includes(query)));values.sort((a,b)=>{if(cellSort==="catalog")return a.order-b.order||a.id.localeCompare(b.id);if(cellSort==="significance")return b.maxSignificance-a.maxSignificance||b.significantGenes-a.significantGenes;if(cellSort==="opportunities")return b.opportunities-a.opportunities||b.significantGenes-a.significantGenes;return b.significantGenes-a.significantGenes||b.maxSignificance-a.maxSignificance;});return values;}
    function renderHeadline(){document.getElementById("headline").innerHTML=[[META.standard_cells_with_significant_markers,'标准细胞'],[META.significant_marker_pairs,'显著细胞-marker'],[META.tested_marker_pairs,'检验组合'],[META.gene_universe,'全局基因全集'],[META.total_cell_evidence_opportunities,'证据机会']].map(x=>'<div><b>'+fmt.format(x[0])+'</b><span>'+x[1]+'</span></div>').join("");}
    function renderBranches(){const select=document.getElementById("branchFilter");[...new Set(DATA.map(c=>c.branch).filter(Boolean))].sort().forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value;select.append(option);});}
    function renderList(){const values=cells();if(!values.some(c=>c.id===selectedId))selectedId=values[0]?.id||"";document.getElementById("cellCount").textContent=fmt.format(values.length)+" / "+fmt.format(DATA.length);document.getElementById("cellList").innerHTML=values.length?values.map(c=>'<button class="cell-button '+(c.id===selectedId?'active':'')+'" data-id="'+esc(c.id)+'"><span class="cell-main"><span class="cell-label">'+esc(c.label)+'</span><span class="cell-count">'+fmt.format(c.significantGenes)+'</span></span><span class="cell-meta">'+esc(c.id)+' · opportunities '+fmt.format(c.opportunities)+' · max −log10(FDR) '+c.maxSignificance.toFixed(1)+'</span></button>').join(""):'<div class="empty">没有匹配细胞</div>';document.querySelectorAll(".cell-button").forEach(button=>button.addEventListener("click",()=>{selectedId=button.dataset.id;geneQuery="";maxQ=.05;minScore=3;renderList();renderDetail();}));}
    function markerRows(c){let rows=c.markers.filter(m=>m.q<=maxQ&&m.score>=minScore&&(!geneQuery||m.gene.toLowerCase().includes(geneQuery)));rows.sort((a,b)=>{if(markerSort==="score")return b.score-a.score||a.q-b.q||a.gene.localeCompare(b.gene);if(markerSort==="fold")return b.fold-a.fold||a.q-b.q||a.gene.localeCompare(b.gene);return a.q-b.q||b.score-a.score||a.gene.localeCompare(b.gene);});return rows;}
    function renderDetail(){const c=DATA.find(value=>value.id===selectedId),detail=document.getElementById("detail");if(!c){detail.innerHTML='<div class="empty">请选择一个标准细胞</div>';return;}const markers=markerRows(c),maxEvidence=Math.max(1,...markers.map(m=>m.score)),maxSignificance=Math.max(1,...markers.map(m=>m.significance));const body=markers.map((m,i)=>'<tr><td class="rank">#'+(i+1)+'</td><td class="gene">'+esc(m.gene)+'<span class="sub">总分排名 #'+m.originalRank+'</span></td><td class="evidence"><div class="bar-track"><span class="bar-stat" style="width:'+(m.statistics/maxEvidence*100)+'%"></span><span class="bar-sig" style="width:'+(m.signatures/maxEvidence*100)+'%"></span></div><span class="sub">总分 '+m.score+' = Statistics '+m.statistics+' + Signatures '+m.signatures+'</span></td><td class="prevalence">'+pct(m.cellPrevalence)+'<span class="sub">背景 '+pct(m.backgroundPrevalence)+'</span></td><td class="num">'+m.fold.toFixed(2)+'×</td><td class="fdr">'+sci(m.q)+'</td><td class="num"><b>'+m.significance.toFixed(2)+'</b><div class="sig-track"><div class="sig-value" style="width:'+(m.significance/maxSignificance*100)+'%"></div></div></td></tr>').join("");detail.innerHTML='<div class="detail-head"><div class="detail-title"><h2>'+esc(c.label)+'</h2><span class="cl-id">'+esc(c.id)+'</span><span>'+esc(c.labelZh)+'</span></div><div class="detail-meta">CL name: '+esc(c.clName)+' · parent: '+esc(c.parent)+' · '+esc(c.branch)+' / '+esc(c.catalogRank)+'</div></div><div class="cell-stats">'+[['显著 marker',c.significantGenes],['证据机会',c.opportunities],['已观察基因',c.testedGenes],['显著比例',pct(c.significantFraction)],['最高 marker 分',c.maxMarkerScore],['最高 −log10(FDR)',c.maxSignificance.toFixed(2)]].map(x=>'<div class="cell-stat"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join("")+'</div><div class="marker-toolbar"><input id="geneSearch" type="search" value="'+esc(geneQuery)+'" placeholder="在该细胞的显著 marker 中搜索"><select id="maxQ"><option value="0.05">FDR ≤ 0.05</option><option value="0.01">FDR ≤ 0.01</option><option value="0.001">FDR ≤ 0.001</option><option value="0.000001">FDR ≤ 10⁻⁶</option></select><select id="minScore"><option value="3">marker 分 ≥ 3</option><option value="5">marker 分 ≥ 5</option><option value="10">marker 分 ≥ 10</option><option value="20">marker 分 ≥ 20</option></select><select id="markerSort"><option value="significance">按显著性排序</option><option value="score">按 marker 分排序</option><option value="fold">按富集倍数排序</option></select></div><div class="marker-status">显示 '+fmt.format(markers.length)+' / '+fmt.format(c.markers.length)+' 个显著 marker</div><div class="table-wrap"><table><thead><tr><th class="rank">排名</th><th class="gene">基因</th><th class="evidence">文献证据分</th><th class="prevalence">细胞内出现率</th><th class="num">富集倍数</th><th class="fdr">FDR</th><th class="num">−log10(FDR)</th></tr></thead><tbody>'+(body||'<tr><td class="empty" colspan="7">没有符合当前条件的显著 marker</td></tr>')+'</tbody></table></div>';const q=document.getElementById("maxQ"),score=document.getElementById("minScore"),sort=document.getElementById("markerSort");q.value=String(maxQ);score.value=String(minScore);sort.value=markerSort;document.getElementById("geneSearch").addEventListener("input",e=>{geneQuery=e.target.value.trim().toLowerCase();const cursor=e.target.selectionStart;renderDetail();const input=document.getElementById("geneSearch");input.focus();input.setSelectionRange(cursor,cursor);});q.addEventListener("change",e=>{maxQ=Number(e.target.value);renderDetail();});score.addEventListener("change",e=>{minScore=Number(e.target.value);renderDetail();});sort.addEventListener("change",e=>{markerSort=e.target.value;renderDetail();});}
    document.getElementById("globalSearch").addEventListener("input",e=>{query=e.target.value.trim().toLowerCase();renderList();renderDetail();});document.getElementById("branchFilter").addEventListener("change",e=>{branch=e.target.value;renderList();renderDetail();});document.getElementById("cellSort").addEventListener("change",e=>{cellSort=e.target.value;renderList();});renderHeadline();renderBranches();renderList();renderDetail();
  </script>
</body>
</html>
"""


def write_html(path: Path, data: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    encoded_data = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    encoded_meta = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    atomic_text(path, HTML_TEMPLATE.replace("__DATA__", encoded_data).replace("__META__", encoded_meta))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    for path in (args.markers, args.cells, args.summary):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    data = build_data(args.cells, args.markers)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    write_html(args.output.resolve(), data, summary)
    result = {
        "output": str(args.output.resolve()),
        "standard_cells": len(data),
        "significant_markers": sum(len(cell["markers"]) for cell in data),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
