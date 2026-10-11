#!/usr/bin/env python3
"""Build a searchable HTML review table from Gilda and SapBERT candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "database" / "agent_standardized_cell_names_filtered.tsv"
DEFAULT_GILDA = ROOT / "database" / "cell_retrieval_benchmark" / "gilda_candidates.tsv"
DEFAULT_SAPBERT = ROOT / "database" / "cell_retrieval_benchmark" / "sapbert_candidates.tsv"
DEFAULT_SUMMARY = ROOT / "database" / "cell_retrieval_benchmark" / "summary.json"
DEFAULT_OUTPUT = ROOT / "database" / "cell_retrieval_benchmark" / "retrieval_review.html"


def read_candidates(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grouped.setdefault(int(row["source_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return grouped


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def build_rows(input_path: Path, gilda_path: Path, sapbert_path: Path) -> list[dict[str, object]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    gilda = read_candidates(gilda_path)
    sapbert = read_candidates(sapbert_path)
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(source_rows):
        gilda_rows = gilda.get(source_index, [])
        sapbert_rows = sapbert.get(source_index, [])
        rows.append(
            {
                "sourceIndex": source_index,
                "originalRank": source.get("original_rank", source.get("rank", "")),
                "name": source.get("cell_name", ""),
                "paperCount": source.get("paper_count", ""),
                "gilda": [
                    {
                        "rank": int(item["rank"]),
                        "id": item["catalog_id"],
                        "label": item["catalog_label"],
                        "clName": item["catalog_cl_name"],
                        "score": float(item["score"]),
                    }
                    for item in gilda_rows
                ],
                "sapbert": [
                    {
                        "rank": int(item["rank"]),
                        "id": item["catalog_id"],
                        "label": item["catalog_label"],
                        "clName": item["catalog_cl_name"],
                        "score": float(item["score"]),
                    }
                    for item in sapbert_rows
                ],
            }
        )
    return rows


def render(
    path: Path,
    rows: list[dict[str, object]],
    gilda_path: Path,
    sapbert_path: Path,
    summary: dict[str, object],
) -> None:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    benchmark = json.dumps(summary.get("backend_summaries", {}), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape(path.name)
    gilda_name = html.escape(gilda_path.name)
    sapbert_name = html.escape(sapbert_path.name)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cell Retrieval Review</title>
  <style>
    :root {{ color-scheme:light; --bg:#f3f5f8; --panel:#fff; --text:#172033; --muted:#667085; --line:#d7dde7; --gilda:#8b5cf6; --sap:#087f5b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:18px 22px 14px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 6px; font-size:22px; }}
    .sub {{ color:var(--muted); }}
    .stats {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    .stat {{ border:1px solid var(--line); border-radius:7px; background:#fbfcfe; padding:7px 11px; min-width:130px; }}
    .stat b {{ display:block; font-size:17px; }} .stat span {{ color:var(--muted); font-size:11px; }}
    .benchmark {{ margin-top:14px; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    .benchmark table {{ width:100%; border-collapse:collapse; min-width:720px; }}
    .benchmark th, .benchmark td {{ border-bottom:1px solid #edf0f4; padding:7px 9px; text-align:left; white-space:nowrap; }}
    .benchmark th {{ background:#f6f8fb; color:#596579; font-size:11px; }}
    .benchmark td {{ font-variant-numeric:tabular-nums; }}
    .benchmark tr:last-child td {{ border-bottom:0; }} .benchmark-note {{ color:var(--muted); font-size:11px; margin-top:7px; }}
    .toolbar {{ position:sticky; top:0; z-index:3; display:flex; gap:8px; padding:10px 22px; background:#e9edf4; border-bottom:1px solid var(--line); }}
    input,select,a.button {{ height:34px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); font:inherit; }}
    input {{ flex:1; min-width:220px; padding:0 10px; }} select,a.button {{ padding:0 10px; }} a.button {{ display:flex; align-items:center; text-decoration:none; white-space:nowrap; }}
    main {{ padding:12px 22px 24px; }} .panel {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .status {{ padding:9px 12px; border-bottom:1px solid var(--line); color:var(--muted); }} .table-wrap {{ overflow:auto; max-height:calc(100vh - 190px); }}
    table {{ width:100%; border-collapse:collapse; table-layout:fixed; }} th {{ position:sticky; top:0; z-index:2; background:#f6f8fb; text-align:left; border-bottom:1px solid var(--line); padding:9px 10px; }}
    th:first-child {{ width:24%; }} th:nth-child(2),th:nth-child(3) {{ width:38%; }} td {{ vertical-align:top; border-bottom:1px solid #edf0f4; padding:9px 10px; }} tr:hover {{ background:#f7faff; }}
    .name {{ font-weight:650; font-size:14px; }} .meta {{ color:var(--muted); font-size:11px; margin-top:3px; }}
    .candidate {{ display:grid; grid-template-columns:28px 1fr auto; gap:6px; padding:4px 0; border-bottom:1px dotted #e4e8ef; }} .candidate:last-child {{ border-bottom:0; }}
    .rank {{ color:var(--muted); text-align:right; }} .label {{ font-weight:600; }} .id {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }} .cl {{ color:#596579; font-size:11px; }}
    .score {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }} .head-gilda {{ color:var(--gilda); }} .head-sapbert {{ color:var(--sap); }}
    .empty {{ color:var(--muted); font-style:italic; }} .hidden {{ display:none; }}
    @media(max-width:900px) {{ .toolbar {{ flex-wrap:wrap; }} input {{ flex-basis:100%; }} table {{ min-width:920px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Cell Retrieval Review</h1>
    <div class="sub">每行一个数据库细胞名称；候选来自 Gilda 自定义 catalog Grounder 与 SapBERT。可搜索名称、按覆盖情况筛选。</div>
    <div class="stats" id="stats"></div>
    <div class="benchmark" id="benchmark"></div>
    <div class="benchmark-note">“已知精确名称”指原始名称本来就能与 catalog 标签直接对应的名称，例如 B cell、Monocyte。两种方案的分数属于不同评分体系，不直接比较分数大小。</div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="搜索原始名称、CL ID 或候选标准名">
    <select id="filter"><option value="all">全部名称</option><option value="gilda">Gilda 有候选</option><option value="gilda_missing">Gilda 无候选</option><option value="disagree">两者 Top-1 不一致</option></select>
    <select id="pageSize"><option value="100">每页 100</option><option value="250" selected>每页 250</option><option value="500">每页 500</option></select>
    <a class="button" href="{gilda_name}" download>下载 Gilda TSV</a>
    <a class="button" href="{sapbert_name}" download>下载 SapBERT TSV</a>
  </div>
  <main><section class="panel"><div class="status" id="status"></div><div class="table-wrap"><table><thead><tr><th>原始名称</th><th class="head-gilda">Gilda 候选（Top-10）</th><th class="head-sapbert">SapBERT 候选（Top-10）</th></tr></thead><tbody id="body"></tbody></table></div></section></main>
  <script>
    const DATA={data};
    const BENCHMARK={benchmark};
    const number=new Intl.NumberFormat("en-US");
    let query="", filter="all", page=1, pageSize=250;
    function esc(value) {{ return String(value??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }}
    function candidateList(values) {{ if(!values.length) return '<div class="empty">无候选</div>'; return values.map(item=>'<div class="candidate"><span class="rank">#'+item.rank+'</span><span><span class="label">'+esc(item.label)+'</span><br><span class="id">'+esc(item.id)+'</span> <span class="cl">'+esc(item.clName)+'</span></span><span class="score">'+Number(item.score).toFixed(4)+'</span></div>').join(""); }}
    function topId(values) {{ return values.length?values[0].id:""; }}
    function matches(row) {{ const hay=[row.name,...row.gilda.map(x=>x.id+" "+x.label+" "+x.clName),...row.sapbert.map(x=>x.id+" "+x.label+" "+x.clName)].join(" ").toLowerCase(); if(query&&!hay.includes(query)) return false; if(filter==="gilda"&&!row.gilda.length) return false; if(filter==="gilda_missing"&&row.gilda.length) return false; if(filter==="disagree"&&topId(row.gilda)===topId(row.sapbert)) return false; return true; }}
    function filtered() {{ return DATA.filter(matches); }}
    function renderStats() {{ const g=DATA.filter(x=>x.gilda.length).length, d=DATA.filter(x=>topId(x.gilda)!==topId(x.sapbert)).length; document.getElementById("stats").innerHTML='<div class="stat"><b>'+number.format(DATA.length)+'</b><span>原始名称</span></div><div class="stat"><b>'+number.format(g)+'</b><span>Gilda 有候选</span></div><div class="stat"><b>'+number.format(DATA.length-g)+'</b><span>Gilda 无候选</span></div><div class="stat"><b>'+number.format(DATA.length-d)+'</b><span>Top-1 一致</span></div><div class="stat"><b>'+number.format(d)+'</b><span>Top-1 不一致</span></div>'; }}
    function percent(value) {{ return value==null?"—":(Number(value)*100).toFixed(2)+"%"; }}
    function integer(value) {{ return value==null?"—":number.format(Math.round(Number(value))); }}
    function memory(value) {{ return value==null?"—":Number(value).toFixed(0)+" MB"; }}
    function renderBenchmark() {{ const rows=[['Gilda 自定义 catalog',BENCHMARK.gilda],['SapBERT',BENCHMARK.sapbert]]; document.getElementById("benchmark").innerHTML='<table><thead><tr><th>方案</th><th>名称覆盖率</th><th>已知精确名称 Top-10 召回</th><th>速度</th><th>峰值内存</th></tr></thead><tbody>'+rows.map(([name,item])=>'<tr><td>'+esc(name)+'</td><td>'+percent(item&&item.candidate_coverage)+'</td><td>'+percent(item&&item.known_exact_recall_at_k)+'</td><td>'+integer(item&&item.queries_per_second)+' 名称/秒</td><td>'+memory(item&&item.peak_process_rss_mb)+'</td></tr>').join('')+'</tbody></table>'; }}
    function render() {{ const rows=filtered(), pages=Math.max(1,Math.ceil(rows.length/pageSize)); page=Math.min(page,pages); const start=(page-1)*pageSize, visible=rows.slice(start,start+pageSize); document.getElementById("status").textContent=number.format(rows.length)+" / "+number.format(DATA.length)+" 名称，第 "+page+" / "+pages+" 页"; document.getElementById("body").innerHTML=visible.map(row=>'<tr><td><div class="name">'+esc(row.name)+'</div><div class="meta">source_index='+row.sourceIndex+(row.originalRank?' · original_rank='+esc(row.originalRank):"")+(row.paperCount?' · papers='+esc(row.paperCount):"")+'</div></td><td>'+candidateList(row.gilda)+'</td><td>'+candidateList(row.sapbert)+'</td></tr>').join("")||'<tr><td colspan="3" class="empty">没有匹配名称</td></tr>'; }}
    document.getElementById("search").addEventListener("input",e=>{{query=e.target.value.trim().toLowerCase();page=1;render();}}); document.getElementById("filter").addEventListener("change",e=>{{filter=e.target.value;page=1;render();}}); document.getElementById("pageSize").addEventListener("change",e=>{{pageSize=Number(e.target.value);page=1;render();}}); renderStats(); renderBenchmark(); render();
  </script>
</body>
</html>"""
    atomic_write(path, document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gilda", type=Path, default=DEFAULT_GILDA)
    parser.add_argument("--sapbert", type=Path, default=DEFAULT_SAPBERT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_rows(args.input.resolve(), args.gilda.resolve(), args.sapbert.resolve())
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    render(args.output.resolve(), rows, args.gilda.resolve(), args.sapbert.resolve(), summary)
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
