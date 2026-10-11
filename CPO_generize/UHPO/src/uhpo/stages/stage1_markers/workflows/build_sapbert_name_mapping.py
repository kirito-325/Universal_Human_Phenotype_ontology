#!/usr/bin/env python3
"""Map the latest filtered cell names to catalog IDs with SapBERT retrieval."""

from __future__ import annotations

import argparse
import csv
import html
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..paths import CATALOG_DIR, DATABASE_DIR, NAME_MAPPING_DIR


DATABASE = DATABASE_DIR
NAME_MAPPING = NAME_MAPPING_DIR
EXPANDED = CATALOG_DIR
DEFAULT_INPUT = DATABASE / "agent_standardized_cell_names_filtered.tsv"
DEFAULT_CATALOG = EXPANDED / "expanded_cell_catalog.tsv"
DEFAULT_OUTPUT_DIR = NAME_MAPPING
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def enrich_candidates(
    candidate_rows: list[dict[str, Any]], sources: list[Any], catalog: list[Any]
) -> list[dict[str, Any]]:
    sources_by_index = {source.source_index: source for source in sources}
    catalog_by_id = {entry.catalog_id: entry for entry in catalog}
    output: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        source = sources_by_index[int(candidate["source_index"])]
        entry = catalog_by_id[str(candidate["catalog_id"])]
        source_row = source.row
        output.append(
            {
                "source_index": source.source_index,
                "filtered_rank": source_row.get("filtered_rank", ""),
                "original_rank": source_row.get("original_rank", source_row.get("rank", "")),
                "cell_name": source.name,
                "normalized_name": candidate["normalized_name"],
                "paper_count": source_row.get("paper_count", ""),
                "source_types": source_row.get("source_types", ""),
                "candidate_rank": int(candidate["rank"]),
                "catalog_id": entry.catalog_id,
                "catalog_label": entry.label,
                "catalog_cl_name": entry.cl_name,
                "catalog_label_zh": entry.label_zh,
                "parent_label": entry.parent_label,
                "branch": entry.branch,
                "catalog_rank": entry.rank,
                "similarity_score": candidate["score"],
                "embedding_model": DEFAULT_MODEL,
            }
        )
    return output


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_source[int(row["source_index"])].append(float(row["similarity_score"]))
    top1 = [values[0] for _, values in sorted(by_source.items())]
    margins = [values[0] - values[1] for _, values in sorted(by_source.items()) if len(values) > 1]

    def quantiles(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
        ordered = sorted(values)

        def at(fraction: float) -> float:
            return ordered[round((len(ordered) - 1) * fraction)]

        return {
            "min": round(ordered[0], 6),
            "p25": round(at(0.25), 6),
            "median": round(at(0.5), 6),
            "p75": round(at(0.75), 6),
            "max": round(ordered[-1], 6),
        }

    return {"top1_score": quantiles(top1), "top1_top2_margin": quantiles(margins)}


def review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(row)
    output: list[dict[str, Any]] = []
    for source_index in sorted(grouped):
        candidates = sorted(grouped[source_index], key=lambda item: int(item["candidate_rank"]))
        first = candidates[0]
        top1_score = float(first["similarity_score"])
        top2_score = float(candidates[1]["similarity_score"]) if len(candidates) > 1 else top1_score
        output.append(
            {
                "sourceIndex": source_index,
                "filteredRank": first["filtered_rank"],
                "originalRank": first["original_rank"],
                "name": first["cell_name"],
                "paperCount": int(first["paper_count"] or 0),
                "sourceTypes": first["source_types"],
                "top1Score": top1_score,
                "margin": top1_score - top2_score,
                "candidates": [
                    {
                        "rank": int(item["candidate_rank"]),
                        "id": item["catalog_id"],
                        "label": item["catalog_label"],
                        "clName": item["catalog_cl_name"],
                        "labelZh": item["catalog_label_zh"],
                        "parent": item["parent_label"],
                        "branch": item["branch"],
                        "catalogRank": item["catalog_rank"],
                        "score": float(item["similarity_score"]),
                    }
                    for item in candidates
                ],
            }
        )
    return output


def write_review_html(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    top1_path: Path,
    top10_path: Path,
) -> None:
    data = json.dumps(review_rows(rows), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    meta = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    top1_name = html.escape(top1_path.name)
    top10_name = html.escape(top10_path.name)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SapBERT Cell Name Mapping Review</title>
  <style>
    :root {{ color-scheme:light; --bg:#f2f5f8; --panel:#fff; --text:#172033; --muted:#667085; --line:#d6dde7; --accent:#087f5b; --soft:#e9f7f1; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--text); background:var(--bg); font:13px/1.42 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:var(--panel); border-bottom:1px solid var(--line); padding:18px 22px 14px; }} h1 {{ margin:0 0 6px; font-size:22px; }} .sub {{ color:var(--muted); }}
    .stats {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }} .stat {{ min-width:145px; padding:8px 11px; border:1px solid var(--line); border-radius:7px; background:#fbfcfd; }} .stat b {{ display:block; font-size:18px; }} .stat span {{ color:var(--muted); font-size:11px; }}
    .toolbar {{ position:sticky; top:0; z-index:4; display:grid; grid-template-columns:minmax(260px,1fr) 170px 170px 170px auto auto; gap:8px; padding:10px 22px; background:#e8edf3; border-bottom:1px solid var(--line); }}
    input,select,a {{ height:35px; min-width:0; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); font:inherit; }} input,select {{ padding:0 10px; }} a {{ display:flex; align-items:center; justify-content:center; padding:0 11px; color:var(--accent); text-decoration:none; white-space:nowrap; }}
    main {{ padding:13px 22px 25px; }} .panel {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }} .status {{ display:flex; justify-content:space-between; gap:12px; padding:9px 12px; border-bottom:1px solid var(--line); color:var(--muted); }}
    button {{ border:1px solid var(--line); border-radius:5px; background:#fff; height:28px; }} button:disabled {{ color:#a4a9b1; }} .table-wrap {{ overflow:auto; max-height:calc(100vh - 226px); min-height:520px; }} table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
    th {{ position:sticky; top:0; z-index:2; padding:9px 10px; background:#f6f8fb; border-bottom:1px solid var(--line); text-align:left; }} th:first-child {{ width:27%; }} th:last-child {{ width:73%; }} td {{ padding:9px 10px; vertical-align:top; border-bottom:1px solid #edf0f4; }} tr:hover {{ background:#f7fbf9; }}
    .source {{ font-weight:680; font-size:14px; }} .source-meta {{ margin-top:3px; color:var(--muted); font-size:11px; }} .candidate {{ display:grid; grid-template-columns:28px minmax(220px,1.4fr) minmax(150px,.9fr) 70px; gap:7px; align-items:start; padding:4px 5px; border-bottom:1px dotted #e1e6ed; }} .candidate:first-child {{ background:var(--soft); border-radius:5px; }} .candidate:last-child {{ border-bottom:0; }}
    .rank {{ color:var(--muted); text-align:right; }} .label {{ font-weight:620; }} .id,.details {{ color:var(--muted); font-size:11px; }} .id {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} .score {{ text-align:right; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} .empty {{ padding:35px; text-align:center; color:var(--muted); }}
    @media(max-width:1000px) {{ .toolbar {{ grid-template-columns:1fr 1fr; }} .toolbar input {{ grid-column:1/-1; }} table {{ min-width:980px; }} }}
  </style>
</head>
<body>
  <header><h1>SapBERT 细胞名称映射检查</h1><div class="sub">输入仅来自刚过滤后的细胞名称；每行展示 embedding 相似度最高的 10 个标准 catalog ID。Top-1 是召回结果，不代表已经完成人工或 LLM 语义确认。</div><div class="stats" id="stats"></div></header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="搜索原始细胞名称、CL ID 或候选标准名称">
    <select id="score"><option value="-1">不限 Top-1 分数</option><option value="0.9">Top-1 ≥ 0.90</option><option value="0.8">Top-1 ≥ 0.80</option><option value="0.7">Top-1 ≥ 0.70</option><option value="0.6">Top-1 ≥ 0.60</option></select>
    <select id="margin"><option value="-1">不限 Top1-Top2 差值</option><option value="0.1">差值 ≥ 0.10</option><option value="0.05">差值 ≥ 0.05</option><option value="0.02">差值 ≥ 0.02</option><option value="0">差值 ≥ 0</option></select>
    <select id="sort"><option value="input">按输入顺序</option><option value="score">按 Top-1 分数</option><option value="margin">按差值</option><option value="papers">按论文数</option></select>
    <a href="{top1_name}" download>下载 Top-1</a><a href="{top10_name}" download>下载 Top-10</a>
  </div>
  <main><section class="panel"><div class="status"><span id="shown"></span><span><button id="prev">上一页</button> <span id="pageInfo"></span> <button id="next">下一页</button></span></div><div class="table-wrap"><table><thead><tr><th>提取的细胞名称</th><th>SapBERT Top-10 标准候选</th></tr></thead><tbody id="body"></tbody></table></div></section></main>
  <script>
    const DATA={data}; const META={meta}; const fmt=new Intl.NumberFormat("en-US"); let query="",minScore=-1,minMargin=-1,sortMode="input",page=1; const pageSize=200;
    function esc(v){{return String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");}}
    function candidates(values){{return values.map(x=>'<div class="candidate"><span class="rank">#'+x.rank+'</span><span><span class="label">'+esc(x.label)+'</span> <span class="details">'+esc(x.labelZh)+'</span><br><span class="id">'+esc(x.id)+'</span> <span class="details">'+esc(x.clName)+'</span></span><span class="details">parent: '+esc(x.parent)+'<br>'+esc(x.branch)+' · '+esc(x.catalogRank)+'</span><span class="score">'+Number(x.score).toFixed(4)+'</span></div>').join("");}}
    function rows(){{let r=DATA.filter(x=>{{const text=[x.name,...x.candidates.flatMap(c=>[c.id,c.label,c.clName,c.labelZh,c.parent])].join(" ").toLowerCase();return(!query||text.includes(query))&&x.top1Score>=minScore&&x.margin>=minMargin;}});if(sortMode==="score")r.sort((a,b)=>b.top1Score-a.top1Score||a.sourceIndex-b.sourceIndex);else if(sortMode==="margin")r.sort((a,b)=>b.margin-a.margin||a.sourceIndex-b.sourceIndex);else if(sortMode==="papers")r.sort((a,b)=>b.paperCount-a.paperCount||a.sourceIndex-b.sourceIndex);else r.sort((a,b)=>a.sourceIndex-b.sourceIndex);return r;}}
    function renderStats(){{const low=DATA.filter(x=>x.top1Score<0.7).length,amb=DATA.filter(x=>x.margin<0.02).length;document.getElementById("stats").innerHTML='<div class="stat"><b>'+fmt.format(DATA.length)+'</b><span>过滤后输入名称</span></div><div class="stat"><b>'+fmt.format(META.catalog_nodes)+'</b><span>Catalog 标准 ID</span></div><div class="stat"><b>'+fmt.format(low)+'</b><span>Top-1 分数低于 0.70</span></div><div class="stat"><b>'+fmt.format(amb)+'</b><span>Top1-Top2 差值低于 0.02</span></div><div class="stat"><b>'+esc(META.model_short)+'</b><span>Embedding 模型</span></div>';}}
    function render(){{const r=rows(),pages=Math.max(1,Math.ceil(r.length/pageSize));page=Math.min(page,pages);const start=(page-1)*pageSize,visible=r.slice(start,start+pageSize);document.getElementById("shown").textContent=fmt.format(r.length)+" / "+fmt.format(DATA.length)+" 个名称";document.getElementById("pageInfo").textContent=page+" / "+pages;document.getElementById("prev").disabled=page<=1;document.getElementById("next").disabled=page>=pages;document.getElementById("body").innerHTML=visible.length?visible.map(x=>'<tr><td><div class="source">'+esc(x.name)+'</div><div class="source-meta">filtered_rank='+esc(x.filteredRank)+' · papers='+fmt.format(x.paperCount)+'<br>Top-1='+x.top1Score.toFixed(4)+' · margin='+x.margin.toFixed(4)+'</div></td><td>'+candidates(x.candidates)+'</td></tr>').join(""):'<tr><td class="empty" colspan="2">没有匹配名称</td></tr>';}}
    document.getElementById("search").addEventListener("input",e=>{{query=e.target.value.trim().toLowerCase();page=1;render();}});document.getElementById("score").addEventListener("change",e=>{{minScore=Number(e.target.value);page=1;render();}});document.getElementById("margin").addEventListener("change",e=>{{minMargin=Number(e.target.value);page=1;render();}});document.getElementById("sort").addEventListener("change",e=>{{sortMode=e.target.value;page=1;render();}});document.getElementById("prev").addEventListener("click",()=>{{if(page>1){{page--;render();}}}});document.getElementById("next").addEventListener("click",()=>{{page++;render();}});renderStats();render();
  </script>
</body></html>"""
    atomic_text(path, document)


TOP10_FIELDS = [
    "source_index",
    "filtered_rank",
    "original_rank",
    "cell_name",
    "normalized_name",
    "paper_count",
    "source_types",
    "candidate_rank",
    "catalog_id",
    "catalog_label",
    "catalog_cl_name",
    "catalog_label_zh",
    "parent_label",
    "branch",
    "catalog_rank",
    "similarity_score",
    "embedding_model",
]
TOP1_FIELDS = [field for field in TOP10_FIELDS if field != "candidate_rank"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 processes every filtered name")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size < 1 or args.max_length < 1 or args.limit < 0:
        raise SystemExit("--batch-size and --max-length must be positive; --limit cannot be negative")
    if not args.input.is_file():
        raise SystemExit(f"filtered cell-name input does not exist: {args.input}")
    if not args.catalog.is_file():
        raise SystemExit(f"catalog does not exist: {args.catalog}")

    # Import here so --help and validation do not initialize torch/transformers.
    from .sapbert_retrieval import exact_recall, run_sapbert
    from .match_cell_names_to_catalog import load_catalog, load_sources

    catalog = load_catalog(args.catalog.resolve())
    sources, excluded = load_sources(args.input.resolve(), apply_filter=False)
    if excluded:
        raise RuntimeError("the filtered input unexpectedly produced excluded rows")
    if args.limit:
        sources = sources[: args.limit]

    candidate_rows, backend_summary = run_sapbert(
        sources,
        catalog,
        top_k=10,
        model_name=args.model,
        requested_device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    enriched = enrich_candidates(candidate_rows, sources, catalog)
    # Preserve the requested model string if a non-default compatible checkpoint is used.
    for row in enriched:
        row["embedding_model"] = args.model
    top1 = [{key: value for key, value in row.items() if key != "candidate_rank"} for row in enriched if int(row["candidate_rank"]) == 1]

    output_dir = args.output_dir.resolve()
    top1_path = output_dir / "sapbert_top1.tsv"
    top10_path = output_dir / "sapbert_top10.tsv"
    html_path = output_dir / "sapbert_top10_review.html"
    summary_path = output_dir / "summary.json"
    write_tsv(top1_path, top1, TOP1_FIELDS)
    write_tsv(top10_path, enriched, TOP10_FIELDS)

    top1_recall = exact_recall(candidate_rows, sources, catalog, 1)
    top10_recall = exact_recall(candidate_rows, sources, catalog, 10)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_is_filtered_cell_names": args.input.resolve() == DEFAULT_INPUT.resolve(),
        "source_names": len(sources),
        "catalog": str(args.catalog.resolve()),
        "catalog_nodes": len(catalog),
        "embedding_model": args.model,
        "model_short": args.model.rsplit("/", 1)[-1],
        "top1_rows": len(top1),
        "top10_rows": len(enriched),
        "top1_output": str(top1_path),
        "top10_output": str(top10_path),
        "review_html": str(html_path),
        "backend": backend_summary,
        "known_exact_recall_at_1": top1_recall,
        "known_exact_recall_at_10": top10_recall,
        "scores": score_summary(enriched),
    }
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_review_html(html_path, enriched, summary, top1_path, top10_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
