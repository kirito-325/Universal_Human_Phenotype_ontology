#!/usr/bin/env python3
"""Render the simplified cell classification as a standalone HTML tree."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "cell_ontology" / "processed"
CLASSES_TSV = PROCESSED / "simplified_cell_classes.tsv"
SUMMARY_JSON = PROCESSED / "summary.json"
OUTPUT_HTML = PROCESSED / "simplified_cell_classification_tree.html"


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CLASSES_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            rows.append(
                {
                    "id": row["class_id"],
                    "name": row["class_name"],
                    "parent": row["parent_id"],
                    "parentName": row["parent_name"],
                    "group": row["group"],
                    "mapped": int(row["mapped_term_count"]),
                    "flagged": int(row["flagged_fine_term_count"]),
                    "note": row["note"],
                    "order": index,
                }
            )
    return rows


def read_summary() -> dict[str, object]:
    if not SUMMARY_JSON.exists():
        return {}
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def render_html(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    title = "Simplified Cell Classification Tree"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647085;
      --line: #d7dde7;
      --hover: #eef3f8;
      --selected: #e7f1ff;
      --shadow: 0 10px 30px rgba(20, 28, 42, 0.08);
      --radius: 8px;
      --root: #48556a;
      --broad: #4f72b8;
      --epithelial: #2f8f84;
      --endocrine: #a45a9c;
      --endothelial: #2877a8;
      --connective: #9a6b2f;
      --muscle: #b24d45;
      --neural: #6b58b8;
      --hematopoietic: #b15a25;
      --germ: #2f7f52;
      --precursor: #6f7f2f;
      --developmental: #7770aa;
      --pigment: #4d5961;
      --functional: #6c727a;
      --in_vitro: #536b8d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 22px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 12px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .stat b {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
      margin-bottom: 3px;
    }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px auto auto auto;
      gap: 8px;
      padding: 12px 22px;
      align-items: center;
      background: #eef2f7;
      border-bottom: 1px solid var(--line);
    }}
    input, select, button {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }}
    input, select {{ padding: 0 10px; min-width: 0; }}
    button {{
      padding: 0 12px;
      cursor: pointer;
    }}
    button:hover {{ background: var(--hover); }}
    main {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 360px;
      gap: 14px;
      padding: 14px 22px 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .tree-panel {{
      overflow: hidden;
    }}
    .tree-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      gap: 10px;
    }}
    .tree-head h2, .details h2 {{
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .tree-count {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    #tree {{
      height: calc(100vh - 214px);
      min-height: 460px;
      overflow: auto;
      padding: 10px 8px 18px;
    }}
    .node {{
      position: relative;
    }}
    .row {{
      display: grid;
      grid-template-columns: auto 22px minmax(0, 1fr) auto;
      align-items: center;
      min-height: 34px;
      border-radius: 6px;
      cursor: pointer;
      padding-right: 8px;
    }}
    .row:hover {{ background: var(--hover); }}
    .row.selected {{ background: var(--selected); }}
    .indent {{
      width: calc(var(--depth) * 18px);
      height: 100%;
    }}
    .toggle {{
      width: 20px;
      height: 20px;
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 0;
      display: grid;
      place-items: center;
      font-size: 14px;
    }}
    .toggle.placeholder {{ visibility: hidden; }}
    .label {{
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 9px;
      height: 22px;
      border-radius: 4px;
      flex: 0 0 auto;
      background: var(--group-color);
    }}
    .name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
    }}
    .id {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .badges {{
      display: flex;
      gap: 5px;
      align-items: center;
      margin-left: 8px;
    }}
    .badge {{
      height: 22px;
      min-width: 34px;
      padding: 0 7px;
      border-radius: 11px;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.warn {{
      color: #8a4a11;
      background: #fff5e6;
      border-color: #efd4a7;
    }}
    mark {{
      background: #fff1a8;
      color: inherit;
      padding: 0 1px;
      border-radius: 2px;
    }}
    .details {{
      padding: 14px;
      position: sticky;
      top: 12px;
      height: fit-content;
    }}
    .details .title {{
      display: flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 10px;
    }}
    .details .color {{
      width: 12px;
      height: 28px;
      border-radius: 5px;
      background: var(--group-color);
      flex: 0 0 auto;
    }}
    .details h2 {{
      font-size: 18px;
      line-height: 1.25;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }}
    .meta div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfe;
    }}
    .meta b {{
      display: block;
      font-size: 18px;
    }}
    .meta span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .field {{
      border-top: 1px solid var(--line);
      padding-top: 11px;
      margin-top: 11px;
    }}
    .field label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .field code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #26364f;
      background: #f1f4f8;
      border-radius: 4px;
      padding: 2px 5px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 13px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfe;
    }}
    .legend i {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--group-color);
    }}
    .empty {{
      color: var(--muted);
      padding: 22px;
      text-align: center;
    }}
    @media (max-width: 980px) {{
      .stats {{ grid-template-columns: repeat(3, minmax(120px, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      .details {{ position: static; }}
      #tree {{ height: 62vh; }}
    }}
    @media (max-width: 620px) {{
      header, .toolbar, main {{ padding-left: 12px; padding-right: 12px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .badges {{ display: none; }}
      .id {{ display: none; }}
      .name {{ white-space: normal; }}
      main {{ padding-top: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Simplified Cell Classification Tree</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search name or CL ID">
    <select id="groupFilter" aria-label="Filter by group"></select>
    <button id="expandAll" type="button">Expand</button>
    <button id="collapseAll" type="button">Collapse</button>
    <button id="reset" type="button">Reset</button>
  </div>
  <main>
    <section class="panel tree-panel">
      <div class="tree-head">
        <h2>Tree</h2>
        <div class="tree-count" id="treeCount"></div>
      </div>
      <div id="tree"></div>
    </section>
    <aside class="panel details" id="details"></aside>
  </main>
  <script>
    const DATA = {data_json};
    const SUMMARY = {summary_json};
    const GROUP_COLORS = {{
      root: "var(--root)",
      broad: "var(--broad)",
      epithelial: "var(--epithelial)",
      endocrine: "var(--endocrine)",
      endothelial: "var(--endothelial)",
      connective: "var(--connective)",
      muscle: "var(--muscle)",
      neural: "var(--neural)",
      hematopoietic: "var(--hematopoietic)",
      germ: "var(--germ)",
      precursor: "var(--precursor)",
      developmental: "var(--developmental)",
      pigment: "var(--pigment)",
      functional: "var(--functional)",
      in_vitro: "var(--in_vitro)"
    }};
    const byId = new Map(DATA.map(node => [node.id, node]));
    const children = new Map();
    DATA.forEach(node => {{
      if (!children.has(node.parent)) children.set(node.parent, []);
      children.get(node.parent).push(node);
    }});
    for (const list of children.values()) list.sort((a, b) => a.order - b.order);

    const root = DATA.find(node => !node.parent) || DATA[0];
    let selectedId = root.id;
    let collapsed = new Set(DATA.filter(node => node.parent && depthOf(node.id) > 1).map(node => node.id));
    let searchTerm = "";
    let groupFilter = "all";

    function depthOf(id) {{
      let depth = 0;
      let node = byId.get(id);
      while (node && node.parent) {{
        depth += 1;
        node = byId.get(node.parent);
      }}
      return depth;
    }}

    function groupColor(group) {{
      return GROUP_COLORS[group] || "var(--muted)";
    }}

    function ancestorIds(id) {{
      const ids = [];
      let node = byId.get(id);
      while (node && node.parent) {{
        ids.push(node.parent);
        node = byId.get(node.parent);
      }}
      return ids;
    }}

    function descendantIds(id) {{
      const ids = [];
      const stack = [...(children.get(id) || [])];
      while (stack.length) {{
        const node = stack.pop();
        ids.push(node.id);
        stack.push(...(children.get(node.id) || []));
      }}
      return ids;
    }}

    function nodeMatches(node) {{
      const text = `${{node.name}} ${{node.id}} ${{node.group}}`.toLowerCase();
      const searchOk = !searchTerm || text.includes(searchTerm);
      const groupOk = groupFilter === "all" || node.group === groupFilter;
      return searchOk && groupOk;
    }}

    function visibleIds() {{
      const visible = new Set();
      DATA.forEach(node => {{
        if (nodeMatches(node)) {{
          visible.add(node.id);
          ancestorIds(node.id).forEach(id => visible.add(id));
        }}
      }});
      if (!searchTerm && groupFilter === "all") DATA.forEach(node => visible.add(node.id));
      return visible;
    }}

    function highlight(text) {{
      const safe = escapeHtml(text);
      if (!searchTerm) return safe;
      const escaped = searchTerm.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return safe.replace(new RegExp(`(${{escaped}})`, "ig"), "<mark>$1</mark>");
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function renderStats() {{
      const stats = [
        ["Retained classes", DATA.length],
        ["Mapped terms", SUMMARY.mapped_terms ?? ""],
        ["Active CL terms", SUMMARY.active_cl_terms ?? ""],
        ["Fine flagged", sum(DATA, "flagged")],
        ["Multi-parent CL", SUMMARY.terms_with_multiple_is_a_parents ?? ""],
        ["Unmapped", SUMMARY.unmapped_terms ?? ""]
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) =>
        `<div class="stat"><b>${{escapeHtml(value)}}</b><span>${{escapeHtml(label)}}</span></div>`
      ).join("");
    }}

    function sum(items, key) {{
      return items.reduce((total, item) => total + Number(item[key] || 0), 0);
    }}

    function renderGroupFilter() {{
      const groups = [...new Set(DATA.map(node => node.group))];
      const select = document.getElementById("groupFilter");
      select.innerHTML = `<option value="all">All groups</option>` +
        groups.map(group => `<option value="${{escapeHtml(group)}}">${{escapeHtml(group)}}</option>`).join("");
    }}

    function renderTree() {{
      const visible = visibleIds();
      if (searchTerm || groupFilter !== "all") {{
        visible.forEach(id => ancestorIds(id).forEach(parent => collapsed.delete(parent)));
      }}
      let shown = 0;
      const tree = document.getElementById("tree");
      tree.innerHTML = "";

      function addNode(node, depth, container) {{
        if (!visible.has(node.id)) return;
        shown += 1;
        const childList = (children.get(node.id) || []).filter(child => visible.has(child.id));
        const hasChildren = childList.length > 0;
        const isCollapsed = collapsed.has(node.id);
        const wrapper = document.createElement("div");
        wrapper.className = "node";
        wrapper.style.setProperty("--group-color", groupColor(node.group));
        const row = document.createElement("div");
        row.className = "row" + (node.id === selectedId ? " selected" : "");
        row.innerHTML = `
          <div class="indent" style="--depth:${{depth}}"></div>
          <button class="toggle ${{hasChildren ? "" : "placeholder"}}" type="button" title="${{isCollapsed ? "Expand" : "Collapse"}}">${{isCollapsed ? "›" : "⌄"}}</button>
          <div class="label"><span class="swatch"></span><span class="name">${{highlight(node.name)}}</span><span class="id">${{highlight(node.id)}}</span></div>
          <div class="badges"><span class="badge">${{node.mapped}}</span>${{node.flagged ? `<span class="badge warn">${{node.flagged}}</span>` : ""}}</div>
        `;
        row.addEventListener("click", event => {{
          if (event.target.classList.contains("toggle")) return;
          selectedId = node.id;
          renderTree();
          renderDetails();
        }});
        row.querySelector(".toggle").addEventListener("click", () => {{
          if (collapsed.has(node.id)) collapsed.delete(node.id);
          else collapsed.add(node.id);
          renderTree();
        }});
        wrapper.appendChild(row);
        container.appendChild(wrapper);
        if (hasChildren && !isCollapsed) {{
          childList.forEach(child => addNode(child, depth + 1, container));
        }}
      }}

      addNode(root, 0, tree);
      if (!shown) tree.innerHTML = '<div class="empty">No matching nodes</div>';
      document.getElementById("treeCount").textContent = `${{shown}} / ${{DATA.length}} nodes`;
    }}

    function renderDetails() {{
      const node = byId.get(selectedId) || root;
      const parent = node.parent ? byId.get(node.parent) : null;
      const childCount = (children.get(node.id) || []).length;
      const descendantCount = descendantIds(node.id).length;
      const lineage = [...ancestorIds(node.id)].reverse().map(id => byId.get(id)).filter(Boolean).concat(node);
      const detail = document.getElementById("details");
      detail.style.setProperty("--group-color", groupColor(node.group));
      detail.innerHTML = `
        <div class="title"><span class="color"></span><h2>${{escapeHtml(node.name)}}</h2></div>
        <div class="field"><label>CL ID</label><code>${{escapeHtml(node.id)}}</code></div>
        <div class="meta">
          <div><b>${{node.mapped}}</b><span>Mapped terms</span></div>
          <div><b>${{node.flagged}}</b><span>Flagged fine terms</span></div>
          <div><b>${{childCount}}</b><span>Direct children</span></div>
          <div><b>${{descendantCount}}</b><span>Descendants</span></div>
        </div>
        <div class="field"><label>Group</label>${{escapeHtml(node.group)}}</div>
        <div class="field"><label>Parent</label>${{parent ? `${{escapeHtml(parent.name)}} <code>${{escapeHtml(parent.id)}}</code>` : "None"}}</div>
        <div class="field"><label>Lineage</label>${{lineage.map(item => escapeHtml(item.name)).join(" / ")}}</div>
        <div class="field"><label>Note</label>${{escapeHtml(node.note || "")}}</div>
        <div class="legend">${{renderLegend()}}</div>
      `;
    }}

    function renderLegend() {{
      const groups = [...new Set(DATA.map(node => node.group))];
      return groups.map(group =>
        `<span style="--group-color:${{groupColor(group)}}"><i></i>${{escapeHtml(group)}}</span>`
      ).join("");
    }}

    document.getElementById("search").addEventListener("input", event => {{
      searchTerm = event.target.value.trim().toLowerCase();
      renderTree();
    }});
    document.getElementById("groupFilter").addEventListener("change", event => {{
      groupFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("expandAll").addEventListener("click", () => {{
      collapsed = new Set();
      renderTree();
    }});
    document.getElementById("collapseAll").addEventListener("click", () => {{
      collapsed = new Set(DATA.filter(node => node.id !== root.id).map(node => node.id));
      renderTree();
    }});
    document.getElementById("reset").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.getElementById("groupFilter").value = "all";
      searchTerm = "";
      groupFilter = "all";
      selectedId = root.id;
      collapsed = new Set(DATA.filter(node => node.parent && depthOf(node.id) > 1).map(node => node.id));
      renderTree();
      renderDetails();
    }});

    renderStats();
    renderGroupFilter();
    renderTree();
    renderDetails();
  </script>
</body>
</html>
"""


def main() -> None:
    rows = read_rows()
    summary = read_summary()
    OUTPUT_HTML.write_text(render_html(rows, summary), encoding="utf-8")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
