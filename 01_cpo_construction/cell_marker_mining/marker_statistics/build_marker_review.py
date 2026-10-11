#!/usr/bin/env python3
"""Build an offline visual review page for expanded cell marker scores."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MARKER_DIR = ROOT / "marker_statistics"
DEFAULT_FULL = MARKER_DIR / "expanded_marker_scores.tsv"
DEFAULT_TOP = MARKER_DIR / "expanded_marker_scores_top100.tsv"
DEFAULT_MAPPING = ROOT / "name_mapping" / "sapbert_top1.tsv"
DEFAULT_SUMMARY = MARKER_DIR / "summary.json"
DEFAULT_OUTPUT = MARKER_DIR / "expanded_marker_scores_review.html"
ACCEPTED_MAPPING_DECISIONS = frozenset({"exact_auto", "match"})


def integer(row: dict[str, str], field: str) -> int:
    return int(str(row.get(field) or "0"))


def number(row: dict[str, str], field: str) -> float:
    return float(str(row.get(field) or "0"))


def load_top_markers(path: Path) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"catalog_id", "catalog_label", "marker_rank", "gene", "marker_score"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid marker Top100 columns: {path}")
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            if not catalog_id:
                continue
            cell = cells.setdefault(
                catalog_id,
                {
                    "order": integer(row, "catalog_order"),
                    "id": catalog_id,
                    "label": str(row.get("catalog_label") or ""),
                    "clName": str(row.get("catalog_cl_name") or ""),
                    "labelZh": str(row.get("catalog_label_zh") or ""),
                    "parentId": str(row.get("parent_id") or ""),
                    "parent": str(row.get("parent_label") or ""),
                    "branch": str(row.get("branch") or ""),
                    "catalogRank": str(row.get("catalog_rank") or ""),
                    "markers": [],
                },
            )
            cell["markers"].append(
                {
                    "rank": integer(row, "marker_rank"),
                    "gene": str(row.get("gene") or ""),
                    "score": integer(row, "marker_score"),
                    "statistics": integer(row, "statistics_score"),
                    "signatures": integer(row, "signatures_score"),
                }
            )
    for cell in cells.values():
        cell["markers"].sort(key=lambda marker: marker["rank"])
    return cells


def load_full_metrics(path: Path) -> dict[str, dict[str, int]]:
    metrics: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"catalog_id", "marker_score", "statistics_score", "signatures_score"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid full marker score columns: {path}")
        for row in reader:
            catalog_id = str(row.get("catalog_id") or "").strip()
            if not catalog_id:
                continue
            score = integer(row, "marker_score")
            current = metrics[catalog_id]
            current["markerCount"] += 1
            current["totalPoints"] += score
            current["statisticsPoints"] += integer(row, "statistics_score")
            current["signaturePoints"] += integer(row, "signatures_score")
            current["scoreAtLeast2"] += int(score >= 2)
            current["scoreAtLeast5"] += int(score >= 5)
            current["scoreAtLeast10"] += int(score >= 10)
            current["maxScore"] = max(current["maxScore"], score)
    return {catalog_id: dict(values) for catalog_id, values in metrics.items()}


def load_aliases(path: Path) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_name", "catalog_id"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"invalid cell-name mapping columns: {path}")
        score_field = (
            "similarity_score"
            if "similarity_score" in reader.fieldnames
            else "top_score" if "top_score" in reader.fieldnames else None
        )
        if score_field is None:
            raise ValueError(f"cell-name mapping has no similarity score column: {path}")
        has_decisions = "decision" in reader.fieldnames
        for row in reader:
            if has_decisions and str(row.get("decision") or "").strip().lower() not in ACCEPTED_MAPPING_DECISIONS:
                continue
            catalog_id = str(row.get("catalog_id") or "").strip()
            cell_name = str(row.get("cell_name") or "").strip()
            if not catalog_id or not cell_name:
                continue
            aliases[catalog_id].append(
                {
                    "name": cell_name,
                    "score": number(row, score_field),
                    "papers": integer(row, "paper_count"),
                    "source": str(row.get("source_types") or ""),
                }
            )
    for values in aliases.values():
        values.sort(key=lambda alias: (-alias["papers"], -alias["score"], alias["name"].casefold()))
    return dict(aliases)


def build_review_data(
    cells: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, int]],
    aliases: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for catalog_id, cell in cells.items():
        if catalog_id not in metrics:
            raise ValueError(f"Top100 cell absent from full marker table: {catalog_id}")
        item = dict(cell)
        item.update(metrics[catalog_id])
        item["aliases"] = aliases.get(catalog_id, [])
        item["aliasCount"] = len(item["aliases"])
        output.append(item)
    output.sort(key=lambda item: (item["order"], item["id"]))
    return output


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
  <title>Expanded Cell Marker Review</title>
  <style>
    :root { color-scheme:light; --bg:#eef2f5; --surface:#fff; --surface2:#f7f9fb; --text:#17212b; --muted:#66727f; --line:#d4dce4; --green:#16805c; --green-soft:#dff3ea; --amber:#d78918; --amber-soft:#fff0d2; --blue:#246aa5; --danger:#b44343; }
    * { box-sizing:border-box; } html,body { margin:0; min-height:100%; } body { background:var(--bg); color:var(--text); font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,select { font:inherit; color:inherit; } button { cursor:pointer; } button:focus-visible,input:focus-visible,select:focus-visible { outline:2px solid var(--blue); outline-offset:1px; }
    header { background:var(--surface); border-bottom:1px solid var(--line); padding:16px 20px 13px; }
    .title-row { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; } h1 { margin:0 0 4px; font-size:21px; font-weight:700; } .provenance { color:var(--muted); max-width:880px; }
    .downloads { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:6px; } .downloads a { height:32px; display:inline-flex; align-items:center; padding:0 10px; border:1px solid var(--line); border-radius:6px; color:var(--blue); text-decoration:none; background:#fff; white-space:nowrap; }
    .headline-stats { display:grid; grid-template-columns:repeat(5,minmax(125px,1fr)); gap:1px; margin-top:13px; border:1px solid var(--line); background:var(--line); }
    .headline-stat { min-width:0; background:var(--surface2); padding:8px 11px; } .headline-stat b { display:block; font-size:17px; overflow-wrap:anywhere; } .headline-stat span { color:var(--muted); font-size:11px; }
    .toolbar { display:grid; grid-template-columns:minmax(260px,1fr) 180px 190px auto; gap:8px; padding:9px 20px; background:#e4e9ee; border-bottom:1px solid var(--line); }
    input,select { width:100%; height:35px; border:1px solid #bdc8d3; border-radius:6px; background:#fff; padding:0 10px; min-width:0; }
    .segment { display:flex; height:35px; border:1px solid #bdc8d3; border-radius:6px; overflow:hidden; background:#fff; } .segment button { flex:1; min-width:68px; border:0; border-right:1px solid var(--line); background:#fff; padding:0 9px; } .segment button:last-child { border-right:0; } .segment button.active { background:#dceaf5; color:#174f7d; font-weight:650; }
    .workspace { display:grid; grid-template-columns:330px minmax(680px,1fr); width:100%; min-width:0; height:calc(100vh - 198px); min-height:600px; }
    aside { border-right:1px solid var(--line); background:var(--surface); overflow:hidden; display:flex; flex-direction:column; } .list-head { display:flex; justify-content:space-between; padding:9px 12px; border-bottom:1px solid var(--line); color:var(--muted); }
    #cellList { overflow:auto; } .cell-button { width:100%; border:0; border-bottom:1px solid #e7ebef; border-left:4px solid transparent; background:#fff; padding:9px 11px 9px 9px; text-align:left; } .cell-button:hover { background:#f5f8fa; } .cell-button.active { border-left-color:var(--green); background:#eaf6f1; }
    .cell-main { display:flex; gap:8px; justify-content:space-between; align-items:baseline; } .cell-label { min-width:0; font-weight:650; overflow-wrap:anywhere; } .cell-score { color:var(--green); font-variant-numeric:tabular-nums; white-space:nowrap; } .cell-meta { margin-top:2px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
    .detail { min-width:0; overflow:auto; background:var(--surface); } .detail-head { padding:14px 18px 11px; border-bottom:1px solid var(--line); } .detail-title { display:flex; align-items:baseline; flex-wrap:wrap; gap:7px; } h2 { margin:0; font-size:20px; } .cl-id { color:var(--blue); font:12px ui-monospace,SFMono-Regular,Menlo,monospace; } .detail-meta { margin-top:4px; color:var(--muted); }
    .cell-stats { display:grid; grid-template-columns:repeat(6,minmax(90px,1fr)); border-bottom:1px solid var(--line); background:var(--surface2); } .cell-stat { min-width:0; padding:8px 12px; border-right:1px solid var(--line); } .cell-stat:last-child { border-right:0; } .cell-stat b { display:block; font-size:16px; overflow-wrap:anywhere; } .cell-stat span { color:var(--muted); font-size:10px; }
    details.aliases { border-bottom:1px solid var(--line); } details.aliases summary { cursor:pointer; padding:9px 18px; font-weight:650; } .alias-wrap { max-height:220px; overflow:auto; border-top:1px solid #e9edf1; } .alias-table { width:100%; border-collapse:collapse; } .alias-table th,.alias-table td { padding:6px 10px; border-bottom:1px solid #edf0f2; text-align:left; } .alias-table th { position:sticky; top:0; background:#f7f9fb; color:var(--muted); font-weight:600; }
    .marker-toolbar { display:grid; grid-template-columns:minmax(180px,1fr) 170px auto; align-items:center; gap:8px; padding:9px 18px; border-bottom:1px solid var(--line); } .legend { display:flex; gap:12px; color:var(--muted); white-space:nowrap; } .legend i { display:inline-block; width:9px; height:9px; margin-right:4px; } .legend .statistics { background:var(--green); } .legend .signatures { background:var(--amber); }
    .marker-status { padding:7px 18px; color:var(--muted); border-bottom:1px solid #e9edf1; }
    .marker-table-wrap { width:100%; max-width:100%; overflow:auto; } .marker-table { width:100%; border-collapse:collapse; table-layout:fixed; } .marker-table th { position:sticky; top:0; z-index:2; background:#f7f9fb; color:#4d5a67; padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; } .marker-table td { padding:7px 9px; border-bottom:1px solid #edf0f2; } .marker-table tr:hover { background:#f8fbfa; }
    .rank { width:62px; color:var(--muted); text-align:right; } .gene { width:150px; font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; } .bar-cell { width:auto; } .score-col { width:82px; text-align:right; font-variant-numeric:tabular-nums; }
    .bar-track { display:flex; width:100%; height:12px; background:#e8edf1; overflow:hidden; } .bar-statistics { background:var(--green); height:100%; } .bar-signatures { background:var(--amber); height:100%; } .empty { padding:35px; text-align:center; color:var(--muted); }
    @media(max-width:1050px) { .headline-stats { grid-template-columns:repeat(3,1fr); } .toolbar { grid-template-columns:1fr 1fr; } .toolbar #globalSearch { grid-column:1/-1; } .workspace { height:auto; min-height:0; grid-template-columns:minmax(0,1fr); } aside { min-width:0; border-right:0; border-bottom:1px solid var(--line); max-height:360px; } .detail { min-width:0; overflow:visible; } .cell-stats { grid-template-columns:repeat(3,1fr); } .marker-table { min-width:760px; } }
    @media(max-width:620px) { header,.toolbar { padding-left:12px; padding-right:12px; } .title-row { display:block; } .downloads { justify-content:flex-start; margin-top:10px; } .headline-stats { grid-template-columns:repeat(2,1fr); } .toolbar { grid-template-columns:1fr; } .toolbar #globalSearch { grid-column:auto; } .cell-stats { grid-template-columns:repeat(2,1fr); } .marker-toolbar { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div class="title-row"><div><h1>Expanded 标准细胞 Marker 检查</h1><div class="provenance">marker_statistics 使用过滤后表；每篇文章在两类表中各最多贡献 1 分。标准 ID 来自名称映射表；若映射包含语义裁决，仅采用 exact_auto 与 match。</div></div><nav class="downloads"><a href="significant_marker_scores_review.html">显著 Marker</a><a href="expanded_marker_scores_top100.tsv" download>Top100 TSV</a><a href="expanded_marker_scores.tsv" download>全量 TSV</a><a href="unmapped_cell_names.tsv" download>未映射名称</a></nav></div>
    <div class="headline-stats" id="headlineStats"></div>
  </header>
  <div class="toolbar">
    <input id="globalSearch" type="search" placeholder="搜索标准细胞、CL ID、marker 或原始细胞名称">
    <select id="branchFilter"><option value="">全部 expanded 分支</option></select>
    <select id="cellSort"><option value="points">按总证据分排序</option><option value="score">按最高 marker 分排序</option><option value="markers">按 marker 数排序</option><option value="aliases">按原始名称数排序</option><option value="catalog">按 catalog 顺序</option></select>
    <div class="segment" id="sourceMode" aria-label="Marker 计分来源"><button data-mode="all" class="active">总分</button><button data-mode="statistics">Statistics</button><button data-mode="signatures">Signatures</button></div>
  </div>
  <main class="workspace">
    <aside><div class="list-head"><span>标准细胞</span><span id="cellCount"></span></div><div id="cellList"></div></aside>
    <section class="detail" id="detail"></section>
  </main>
  <script>
    const DATA=__DATA__; const META=__META__; const fmt=new Intl.NumberFormat("en-US");
    let query="",branch="",sortMode="points",sourceMode="all",selectedId="",markerQuery="",minScore=1;
    const searchIndex=new Map(DATA.map(c=>[c.id,[c.id,c.label,c.clName,c.labelZh,c.parent,c.branch,...c.aliases.map(a=>a.name),...c.markers.map(m=>m.gene)].join(" ").toLowerCase()]));
    const esc=v=>String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
    const activeScore=m=>sourceMode==="statistics"?m.statistics:sourceMode==="signatures"?m.signatures:m.score;
    const cellPoints=c=>sourceMode==="statistics"?c.statisticsPoints:sourceMode==="signatures"?c.signaturePoints:c.totalPoints;
    function filteredCells(){let values=DATA.filter(c=>(!branch||c.branch===branch)&&(!query||searchIndex.get(c.id).includes(query)));values.sort((a,b)=>{if(sortMode==="catalog")return a.order-b.order||a.id.localeCompare(b.id);if(sortMode==="score")return b.maxScore-a.maxScore||a.order-b.order;if(sortMode==="markers")return b.markerCount-a.markerCount||a.order-b.order;if(sortMode==="aliases")return b.aliasCount-a.aliasCount||a.order-b.order;return cellPoints(b)-cellPoints(a)||b.maxScore-a.maxScore||a.order-b.order;});return values;}
    function renderHeadline(){const s=META.scores;document.getElementById("headlineStats").innerHTML=[['标准细胞',fmt.format(s.standard_cells_with_markers)],['细胞-marker 组合',fmt.format(s.marker_pairs)],['总证据分',fmt.format(s.total_points)],['Statistics 分',fmt.format(s.statistics_points)],['Signatures 分',fmt.format(s.signatures_points)]].map(x=>'<div class="headline-stat"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join("");}
    function renderBranches(){const select=document.getElementById("branchFilter");[...new Set(DATA.map(c=>c.branch).filter(Boolean))].sort().forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value;select.append(option);});}
    function renderList(){const values=filteredCells();if(!values.some(c=>c.id===selectedId))selectedId=values[0]?.id||"";document.getElementById("cellCount").textContent=fmt.format(values.length)+" / "+fmt.format(DATA.length);document.getElementById("cellList").innerHTML=values.length?values.map(c=>'<button class="cell-button '+(c.id===selectedId?'active':'')+'" data-id="'+esc(c.id)+'"><span class="cell-main"><span class="cell-label">'+esc(c.label)+'</span><span class="cell-score">'+fmt.format(cellPoints(c))+' 分</span></span><span class="cell-meta">'+esc(c.id)+' · '+fmt.format(c.markerCount)+' markers · '+fmt.format(c.aliasCount)+' names</span></button>').join(""):'<div class="empty">没有匹配的标准细胞</div>';document.querySelectorAll(".cell-button").forEach(button=>button.addEventListener("click",()=>{selectedId=button.dataset.id;markerQuery="";minScore=1;renderList();renderDetail();}));}
    function aliasTable(c){if(!c.aliases.length)return '<div class="empty">没有原始名称映射</div>';return '<div class="alias-wrap"><table class="alias-table"><thead><tr><th>原始细胞名称</th><th>候选相似度</th><th>论文数</th><th>来源</th></tr></thead><tbody>'+c.aliases.map(a=>'<tr><td>'+esc(a.name)+'</td><td>'+a.score.toFixed(4)+'</td><td>'+fmt.format(a.papers)+'</td><td>'+esc(a.source)+'</td></tr>').join("")+'</tbody></table></div>';}
    function markerRows(c){let markers=c.markers.filter(m=>activeScore(m)>=minScore&&(!markerQuery||m.gene.toLowerCase().includes(markerQuery)));markers.sort((a,b)=>activeScore(b)-activeScore(a)||b.score-a.score||a.gene.localeCompare(b.gene));return markers;}
    function renderDetail(){const c=DATA.find(value=>value.id===selectedId),detail=document.getElementById("detail");if(!c){detail.innerHTML='<div class="empty">请选择一个标准细胞</div>';return;}const markers=markerRows(c),max=Math.max(1,...markers.map(activeScore));const rows=markers.map((m,index)=>{const statWidth=sourceMode==="signatures"?0:(m.statistics/max*100),sigWidth=sourceMode==="statistics"?0:(m.signatures/max*100);return '<tr><td class="rank">#'+(index+1)+'</td><td class="gene">'+esc(m.gene)+'</td><td class="bar-cell"><div class="bar-track"><span class="bar-statistics" style="width:'+statWidth+'%"></span><span class="bar-signatures" style="width:'+sigWidth+'%"></span></div></td><td class="score-col"><b>'+fmt.format(m.score)+'</b></td><td class="score-col">'+fmt.format(m.statistics)+'</td><td class="score-col">'+fmt.format(m.signatures)+'</td></tr>';}).join("");detail.innerHTML='<div class="detail-head"><div class="detail-title"><h2>'+esc(c.label)+'</h2><span class="cl-id">'+esc(c.id)+'</span><span>'+esc(c.labelZh)+'</span></div><div class="detail-meta">CL name: '+esc(c.clName)+' · parent: '+esc(c.parent)+' · '+esc(c.branch)+' / '+esc(c.catalogRank)+'</div></div><div class="cell-stats">'+[['Marker 总数',c.markerCount],['总证据分',c.totalPoints],['最高得分',c.maxScore],['得分 ≥ 2',c.scoreAtLeast2],['得分 ≥ 5',c.scoreAtLeast5],['原始名称',c.aliasCount]].map(x=>'<div class="cell-stat"><b>'+fmt.format(x[1])+'</b><span>'+x[0]+'</span></div>').join("")+'</div><details class="aliases"><summary>归并到该 ID 的原始细胞名称（'+fmt.format(c.aliasCount)+'）</summary>'+aliasTable(c)+'</details><div class="marker-toolbar"><input id="markerSearch" type="search" value="'+esc(markerQuery)+'" placeholder="在该细胞的 Top100 中搜索基因"><select id="minScore"><option value="1">最低来源分：1</option><option value="2">最低来源分：2</option><option value="3">最低来源分：3</option><option value="5">最低来源分：5</option><option value="10">最低来源分：10</option><option value="20">最低来源分：20</option></select><div class="legend"><span><i class="statistics"></i>Statistics</span><span><i class="signatures"></i>Signatures</span></div></div><div class="marker-status">显示 '+fmt.format(markers.length)+' / '+fmt.format(c.markers.length)+' 个 Top marker · 当前按'+(sourceMode==="all"?'总分':sourceMode)+ '排序</div><div class="marker-table-wrap"><table class="marker-table"><thead><tr><th class="rank">当前排名</th><th class="gene">基因</th><th>证据分布</th><th class="score-col">总分</th><th class="score-col">Statistics</th><th class="score-col">Signatures</th></tr></thead><tbody>'+(rows||'<tr><td colspan="6" class="empty">没有符合条件的 marker</td></tr>')+'</tbody></table></div>';const min=document.getElementById("minScore");min.value=String(minScore);document.getElementById("markerSearch").addEventListener("input",e=>{markerQuery=e.target.value.trim().toLowerCase();const cursor=e.target.selectionStart;renderDetail();const input=document.getElementById("markerSearch");input.focus();input.setSelectionRange(cursor,cursor);});min.addEventListener("change",e=>{minScore=Number(e.target.value);renderDetail();});}
    document.getElementById("globalSearch").addEventListener("input",e=>{query=e.target.value.trim().toLowerCase();renderList();renderDetail();});document.getElementById("branchFilter").addEventListener("change",e=>{branch=e.target.value;renderList();renderDetail();});document.getElementById("cellSort").addEventListener("change",e=>{sortMode=e.target.value;renderList();});document.querySelectorAll("#sourceMode button").forEach(button=>button.addEventListener("click",()=>{sourceMode=button.dataset.mode;document.querySelectorAll("#sourceMode button").forEach(value=>value.classList.toggle("active",value===button));minScore=1;renderList();renderDetail();}));
    renderHeadline();renderBranches();renderList();renderDetail();
  </script>
</body>
</html>
"""


def write_review_html(path: Path, data: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    encoded_data = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    encoded_meta = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = HTML_TEMPLATE.replace("__DATA__", encoded_data).replace("__META__", encoded_meta)
    atomic_text(path, document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    for path in (args.full, args.top, args.mapping, args.summary):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    cells = load_top_markers(args.top)
    metrics = load_full_metrics(args.full)
    aliases = load_aliases(args.mapping)
    data = build_review_data(cells, metrics, aliases)
    write_review_html(args.output.resolve(), data, summary)
    result = {
        "output": str(args.output.resolve()),
        "standard_cells": len(data),
        "top_markers": sum(len(cell["markers"]) for cell in data),
        "mapped_source_names": sum(len(cell["aliases"]) for cell in data),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
