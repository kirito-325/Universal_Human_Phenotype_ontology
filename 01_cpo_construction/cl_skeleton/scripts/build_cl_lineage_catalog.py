#!/usr/bin/env python3
"""Extract Cell Ontology develops_from relations as a lineage catalog."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from build_cell_catalog_skeleton import COLORS, OBO_PATH, SKELETON  # noqa: E402


OUT_DIR = ROOT / "cell_ontology" / "lineage"
LINEAGE_REL = "RO:0002202"
LINEAGE_REL_LABEL = "develops_from"
VIRTUAL_ROOT_ID = "__LINEAGE_ROOT__"

NON_HUMAN_RE = re.compile(
    r"\b("
    r"mouse|murine|rat|macaque|non-human primate|"
    r"Mmus|Nematoda|Protostomia|Arthropoda|invertebrate|insect|"
    r"Fungi|fungal|Caenorhabditis|Drosophila|zebrafish"
    r")\b",
    re.IGNORECASE,
)
NON_HUMAN_SENSU_RE = re.compile(r"\(sensu [^)]+\)", re.IGNORECASE)
EXPLICIT_HUMAN_RE = re.compile(r"(?<!non-)\bhuman\b", re.IGNORECASE)


@dataclass
class Term:
    id: str = ""
    name: str = ""
    is_a: list[str] = field(default_factory=list)
    lineage_sources: list[str] = field(default_factory=list)
    obsolete: bool = False
    line: int = 0


@dataclass
class LineageNode:
    id: str
    name: str
    primary_parent_id: str
    all_parent_ids: list[str]
    child_ids: list[str]
    root_id: str
    depth: int
    component_id: int
    branch: str
    anchor_id: str
    order: int


def non_human_reasons(term: Term) -> list[str]:
    """Return reasons for excluding explicitly non-human lineage terms."""
    name = term.name
    reasons: list[str] = []
    if term.id.startswith("CL:43"):
        reasons.append("non_human_project_namespace")
    sensu_match = NON_HUMAN_SENSU_RE.search(name)
    has_non_human_sensu = bool(
        sensu_match and not EXPLICIT_HUMAN_RE.search(sensu_match.group(0))
    )
    if NON_HUMAN_RE.search(name) or has_non_human_sensu:
        reasons.append("non_human_taxon_label")
    return reasons


def parse_obo() -> dict[str, Term]:
    terms: list[Term] = []
    current: Term | None = None
    in_term = False
    for line_no, raw in enumerate(OBO_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if raw == "[Term]":
            if current:
                terms.append(current)
            current = Term(line=line_no)
            in_term = True
            continue
        if raw.startswith("[") and raw != "[Term]":
            if current:
                terms.append(current)
            current = None
            in_term = False
            continue
        if not in_term or current is None:
            continue
        if raw.startswith("id: "):
            current.id = raw[4:].strip()
        elif raw.startswith("name: "):
            current.name = raw[6:].strip()
        elif raw.startswith("is_a: "):
            current.is_a.append(raw[6:].split()[0])
        elif raw.startswith("relationship: "):
            parts = raw[14:].split(" ! ", 1)[0].split()
            if len(parts) >= 2 and parts[0] == LINEAGE_REL:
                current.lineage_sources.append(parts[1])
        elif raw.startswith("is_obsolete: true"):
            current.obsolete = True
    if current:
        terms.append(current)
    return {
        term.id: term
        for term in terms
        if term.id.startswith("CL:") and term.name and not term.obsolete
    }


def build_edges(terms: dict[str, Term]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for term in terms.values():
        for source_id in term.lineage_sources:
            if source_id in terms:
                edges.append((source_id, term.id))
    edges = sorted(set(edges), key=lambda edge: (terms[edge[0]].name.lower(), terms[edge[1]].name.lower()))
    return edges


def filter_human_lineage_terms(
    terms: dict[str, Term],
) -> tuple[dict[str, Term], list[dict[str, str]]]:
    excluded = []
    kept = {}
    for term_id, term in terms.items():
        reasons = non_human_reasons(term)
        if reasons:
            excluded.append(
                {
                    "term_id": term_id,
                    "term_name": term.name,
                    "reason": ";".join(reasons),
                }
            )
            continue
        kept[term_id] = term
    return kept, sorted(excluded, key=lambda row: (row["term_name"].lower(), row["term_id"]))


def ancestor_distances(term_id: str, terms: dict[str, Term]) -> dict[str, int]:
    distances = {term_id: 0}
    queue = deque([term_id])
    while queue:
        current = queue.popleft()
        for parent in terms[current].is_a:
            if parent not in terms or parent in distances:
                continue
            distances[parent] = distances[current] + 1
            queue.append(parent)
    return distances


def infer_branch(term_id: str, terms: dict[str, Term]) -> tuple[str, str]:
    skeleton_ids = {node.cl_id for node in SKELETON}
    skeleton_order = {node.cl_id: index for index, node in enumerate(SKELETON)}
    skeleton_by_id = {node.cl_id: node for node in SKELETON}
    distances = ancestor_distances(term_id, terms)
    candidates = [
        (distance, skeleton_order[ancestor], ancestor)
        for ancestor, distance in distances.items()
        if ancestor in skeleton_ids
    ]
    if not candidates:
        return "unanchored", ""
    candidates.sort()
    anchor_id = candidates[0][2]
    return skeleton_by_id[anchor_id].branch, anchor_id


def primary_parent(
    node_id: str,
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
    terms: dict[str, Term],
) -> str:
    parents = incoming.get(node_id, [])
    if not parents:
        return VIRTUAL_ROOT_ID
    # Prefer sources that themselves have lineage context, then deterministic name order.
    ranked = sorted(
        parents,
        key=lambda parent: (
            0 if incoming.get(parent) or outgoing.get(parent) else 1,
            terms[parent].name.lower(),
            parent,
        ),
    )
    return ranked[0]


def assign_components(
    node_ids: set[str],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
    terms: dict[str, Term],
) -> dict[str, int]:
    undirected: dict[str, set[str]] = defaultdict(set)
    for source, targets in outgoing.items():
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    component_by_id: dict[str, int] = {}
    component = 0
    for node_id in sorted(node_ids, key=lambda item: terms[item].name.lower()):
        if node_id in component_by_id:
            continue
        component += 1
        queue = deque([node_id])
        component_by_id[node_id] = component
        while queue:
            current = queue.popleft()
            for neighbor in sorted(undirected[current], key=lambda item: terms[item].name.lower()):
                if neighbor in component_by_id:
                    continue
                component_by_id[neighbor] = component
                queue.append(neighbor)
    return component_by_id


def depths_and_roots(
    node_ids: set[str],
    primary_parent_by_id: dict[str, str],
    terms: dict[str, Term],
) -> tuple[dict[str, int], dict[str, str]]:
    depth_by_id: dict[str, int] = {}
    root_by_id: dict[str, str] = {}

    def resolve(node_id: str) -> tuple[int, str]:
        if node_id in depth_by_id:
            return depth_by_id[node_id], root_by_id[node_id]
        parent = primary_parent_by_id[node_id]
        if parent == VIRTUAL_ROOT_ID or parent not in node_ids:
            depth_by_id[node_id] = 1
            root_by_id[node_id] = node_id
            return depth_by_id[node_id], root_by_id[node_id]
        parent_depth, root_id = resolve(parent)
        depth_by_id[node_id] = parent_depth + 1
        root_by_id[node_id] = root_id
        return depth_by_id[node_id], root_by_id[node_id]

    for node_id in sorted(node_ids, key=lambda item: terms[item].name.lower()):
        resolve(node_id)
    return depth_by_id, root_by_id


def build_lineage_catalog() -> tuple[list[LineageNode], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    all_terms = parse_obo()
    terms, excluded_non_human = filter_human_lineage_terms(all_terms)
    edges = build_edges(terms)
    node_ids = {node_id for edge in edges for node_id in edge}

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source_id, target_id in edges:
        incoming[target_id].append(source_id)
        outgoing[source_id].append(target_id)
    for values in incoming.values():
        values.sort(key=lambda item: terms[item].name.lower())
    for values in outgoing.values():
        values.sort(key=lambda item: terms[item].name.lower())

    primary_parent_by_id = {
        node_id: primary_parent(node_id, incoming, outgoing, terms)
        for node_id in node_ids
    }
    depth_by_id, root_by_id = depths_and_roots(node_ids, primary_parent_by_id, terms)
    component_by_id = assign_components(node_ids, incoming, outgoing, terms)
    branch_anchor = {node_id: infer_branch(node_id, terms) for node_id in node_ids}

    nodes = [
        LineageNode(
            id=node_id,
            name=terms[node_id].name,
            primary_parent_id=primary_parent_by_id[node_id],
            all_parent_ids=incoming.get(node_id, []),
            child_ids=outgoing.get(node_id, []),
            root_id=root_by_id[node_id],
            depth=depth_by_id[node_id],
            component_id=component_by_id[node_id],
            branch=branch_anchor[node_id][0],
            anchor_id=branch_anchor[node_id][1],
            order=0,
        )
        for node_id in node_ids
    ]
    nodes.sort(
        key=lambda node: (
            terms[node.root_id].name.lower(),
            node.depth,
            node.name.lower(),
            node.id,
        )
    )
    for order, node in enumerate(nodes):
        node.order = order

    edge_rows = [
        {
            "source_id": source_id,
            "source_name": terms[source_id].name,
            "target_id": target_id,
            "target_name": terms[target_id].name,
            "relation_id": LINEAGE_REL,
            "relation_label": LINEAGE_REL_LABEL,
        }
        for source_id, target_id in edges
    ]
    roots = [node for node in nodes if node.primary_parent_id == VIRTUAL_ROOT_ID]
    summary = {
        "active_cl_terms": len(all_terms),
        "human_filtered_active_cl_terms": len(terms),
        "excluded_non_human_terms": len(excluded_non_human),
        "lineage_relation": LINEAGE_REL,
        "lineage_relation_label": LINEAGE_REL_LABEL,
        "lineage_edges": len(edges),
        "lineage_nodes": len(nodes),
        "lineage_roots": len(roots),
        "connected_components": len(set(node.component_id for node in nodes)),
        "nodes_with_multiple_lineage_sources": sum(1 for node in nodes if len(node.all_parent_ids) > 1),
        "source_nodes_with_multiple_children": sum(1 for node in nodes if len(node.child_ids) > 1),
        "max_primary_depth": max(node.depth for node in nodes) if nodes else 0,
        "branch_counts": dict(sorted(Counter(node.branch for node in nodes).items())),
    }
    return nodes, edge_rows, excluded_non_human, summary


def write_tsv(
    nodes: list[LineageNode],
    edge_rows: list[dict[str, str]],
    excluded_non_human: list[dict[str, str]],
    summary: dict[str, object],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {node.id: node for node in nodes}
    node_fields = [
        "order",
        "depth",
        "component_id",
        "root_id",
        "root_name",
        "cl_id",
        "cl_name",
        "primary_lineage_parent_id",
        "primary_lineage_parent_name",
        "all_lineage_parent_ids",
        "all_lineage_parent_names",
        "lineage_child_ids",
        "lineage_child_names",
        "branch",
        "anchor_id",
        "anchor_name",
    ]
    with (OUT_DIR / "lineage_nodes.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_fields, delimiter="\t")
        writer.writeheader()
        for node in sorted(nodes, key=lambda item: item.order):
            parent = by_id.get(node.primary_parent_id)
            root = by_id[node.root_id]
            anchor = by_id.get(node.anchor_id)
            writer.writerow(
                {
                    "order": node.order + 1,
                    "depth": node.depth,
                    "component_id": node.component_id,
                    "root_id": node.root_id,
                    "root_name": root.name,
                    "cl_id": node.id,
                    "cl_name": node.name,
                    "primary_lineage_parent_id": "" if node.primary_parent_id == VIRTUAL_ROOT_ID else node.primary_parent_id,
                    "primary_lineage_parent_name": parent.name if parent else "",
                    "all_lineage_parent_ids": ";".join(node.all_parent_ids),
                    "all_lineage_parent_names": "; ".join(by_id[parent_id].name for parent_id in node.all_parent_ids),
                    "lineage_child_ids": ";".join(node.child_ids),
                    "lineage_child_names": "; ".join(by_id[child_id].name for child_id in node.child_ids),
                    "branch": node.branch,
                    "anchor_id": node.anchor_id,
                    "anchor_name": anchor.name if anchor else "",
                }
            )

    edge_fields = list(edge_rows[0]) if edge_rows else [
        "source_id",
        "source_name",
        "target_id",
        "target_name",
        "relation_id",
        "relation_label",
    ]
    with (OUT_DIR / "lineage_edges.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(edge_rows)

    excluded_fields = ["term_id", "term_name", "reason"]
    with (OUT_DIR / "excluded_non_human_lineage_terms.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=excluded_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded_non_human)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown(nodes: list[LineageNode]) -> None:
    by_id = {node.id: node for node in nodes}
    children: dict[str, list[LineageNode]] = defaultdict(list)
    for node in nodes:
        children[node.primary_parent_id].append(node)
    for child_list in children.values():
        child_list.sort(key=lambda node: (by_id[node.root_id].name.lower(), node.name.lower(), node.id))

    lines = [
        "# CL Develops-From Lineage Catalog",
        "",
        "Direction: source cell -> derived cell, based on `relationship: RO:0002202`.",
        "Nodes with multiple lineage sources are shown once under a primary parent; all sources are preserved in TSV/HTML details.",
        "",
    ]

    def add(node: LineageNode, depth: int) -> None:
        multi = f" +{len(node.all_parent_ids) - 1} source" if len(node.all_parent_ids) > 1 else ""
        lines.append(f"{'  ' * depth}- {node.name} `{node.id}`{multi}")
        for child in children.get(node.id, []):
            add(child, depth + 1)

    for root in children[VIRTUAL_ROOT_ID]:
        add(root, 0)
    (OUT_DIR / "lineage_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(nodes: list[LineageNode], summary: dict[str, object]) -> None:
    by_id = {node.id: node for node in nodes}
    data = [
        {
            "id": node.id,
            "name": node.name,
            "parent": "" if node.primary_parent_id == VIRTUAL_ROOT_ID else node.primary_parent_id,
            "allParents": node.all_parent_ids,
            "children": node.child_ids,
            "root": node.root_id,
            "rootName": by_id[node.root_id].name,
            "depth": node.depth,
            "component": node.component_id,
            "branch": node.branch,
            "anchor": node.anchor_id,
            "anchorName": by_id[node.anchor_id].name if node.anchor_id in by_id else "",
            "order": node.order,
        }
        for node in sorted(nodes, key=lambda item: item.order)
    ]
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    colors_json = json.dumps(COLORS | {"unanchored": "#647085"}, ensure_ascii=False, separators=(",", ":"))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CL Develops-From Lineage Catalog</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647085;
      --line: #d8dee8;
      --hover: #edf2f7;
      --selected: #e5f0ff;
      --shadow: 0 10px 28px rgba(20, 28, 42, 0.08);
      --radius: 8px;
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
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 22px 14px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 23px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stats div {{
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 12px;
      min-width: 0;
    }}
    .stats b {{
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }}
    .stats span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 190px 170px auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 12px 22px;
      background: #edf1f6;
      border-bottom: 1px solid var(--line);
    }}
    input, select, button {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
      min-width: 0;
    }}
    input, select {{ padding: 0 10px; }}
    button {{
      padding: 0 12px;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{ background: var(--hover); }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 420px;
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
    .tree-panel {{ overflow: hidden; }}
    .tree-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
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
      height: calc(100vh - 216px);
      min-height: 500px;
      overflow: auto;
      padding: 10px 8px 18px;
    }}
    .row {{
      display: grid;
      grid-template-columns: auto 24px minmax(0, 1fr) auto;
      align-items: center;
      min-height: 36px;
      padding-right: 8px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .row:hover {{ background: var(--hover); }}
    .row.selected {{ background: var(--selected); }}
    .indent {{
      width: calc(var(--depth) * 18px);
      height: 100%;
    }}
    .toggle {{
      width: 22px;
      height: 22px;
      display: grid;
      place-items: center;
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 0;
      font-size: 14px;
      line-height: 1;
    }}
    .toggle.placeholder {{ visibility: hidden; }}
    .label {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .swatch {{
      width: 9px;
      height: 24px;
      border-radius: 4px;
      flex: 0 0 auto;
      background: var(--branch-color);
    }}
    .name {{
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .id {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .badges {{
      display: flex;
      align-items: center;
      gap: 5px;
      margin-left: 8px;
    }}
    .badge {{
      height: 22px;
      min-width: 42px;
      padding: 0 8px;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.multi {{
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
      gap: 10px;
      margin-bottom: 12px;
    }}
    .details .color {{
      width: 12px;
      height: 30px;
      border-radius: 5px;
      background: var(--branch-color);
      flex: 0 0 auto;
    }}
    .details h2 {{
      font-size: 18px;
      line-height: 1.25;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }}
    .meta-grid div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .meta-grid b {{
      display: block;
      font-size: 18px;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }}
    .meta-grid span {{
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
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #26364f;
      background: #f1f4f8;
      border-radius: 4px;
      padding: 2px 5px;
    }}
    .lineage-list {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .lineage-list button {{
      height: auto;
      min-height: 30px;
      text-align: left;
      padding: 5px 8px;
    }}
    .lineage {{
      color: #344055;
      overflow-wrap: anywhere;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 13px;
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
      background: var(--branch-color);
    }}
    .empty {{
      color: var(--muted);
      padding: 22px;
      text-align: center;
    }}
    @media (max-width: 1120px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      .details {{ position: static; }}
      #tree {{ height: 62vh; }}
    }}
    @media (max-width: 720px) {{
      header, .toolbar, main {{ padding-left: 12px; padding-right: 12px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .badges {{ display: none; }}
      .id {{ display: none; }}
      .name {{ white-space: normal; }}
      .swatch {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CL Develops-From Lineage Catalog</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search cell name or CL ID">
    <select id="branchFilter" aria-label="Filter by inferred branch"></select>
    <select id="rootFilter" aria-label="Filter by lineage root"></select>
    <button id="expandAll" type="button">Expand</button>
    <button id="collapseAll" type="button">Collapse</button>
    <button id="reset" type="button">Reset</button>
  </div>
  <main>
    <section class="panel tree-panel">
      <div class="tree-head">
        <h2>Lineage Tree</h2>
        <div class="tree-count" id="treeCount"></div>
      </div>
      <div id="tree"></div>
    </section>
    <aside class="panel details" id="details"></aside>
  </main>
  <script>
    const DATA = {data_json};
    const SUMMARY = {summary_json};
    const COLORS = {colors_json};
    const ROOT_ID = "{VIRTUAL_ROOT_ID}";
    const byId = new Map(DATA.map(node => [node.id, node]));
    const children = new Map();
    DATA.forEach(node => {{
      const parent = node.parent || ROOT_ID;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(node);
    }});
    for (const list of children.values()) list.sort((a, b) => a.order - b.order);

    let selectedId = DATA[0]?.id || "";
    let searchText = "";
    let branchFilter = "all";
    let rootFilter = "all";
    let collapsed = new Set(DATA.filter(node => node.depth > 2).map(node => node.id));

    function branchColor(branch) {{
      return COLORS[branch] || "#647085";
    }}

    function escapeHtml(value) {{
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function highlight(value) {{
      const safe = escapeHtml(value);
      if (!searchText) return safe;
      const escaped = searchText.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return safe.replace(new RegExp("(" + escaped + ")", "ig"), "<mark>$1</mark>");
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
      const text = [node.name, node.id, node.branch, node.rootName].join(" ").toLowerCase();
      const searchOk = !searchText || text.includes(searchText);
      const branchOk = branchFilter === "all" || node.branch === branchFilter;
      const rootOk = rootFilter === "all" || node.root === rootFilter;
      return searchOk && branchOk && rootOk;
    }}

    function visibleIds() {{
      const ids = new Set();
      DATA.forEach(node => {{
        if (nodeMatches(node)) {{
          ids.add(node.id);
          ancestorIds(node.id).forEach(parent => ids.add(parent));
        }}
      }});
      if (!searchText && branchFilter === "all" && rootFilter === "all") {{
        DATA.forEach(node => ids.add(node.id));
      }}
      return ids;
    }}

    function renderStats() {{
      const stats = [
        ["Nodes", SUMMARY.lineage_nodes],
        ["Edges", SUMMARY.lineage_edges],
        ["Roots", SUMMARY.lineage_roots],
        ["Components", SUMMARY.connected_components],
        ["Multi-source", SUMMARY.nodes_with_multiple_lineage_sources],
        ["Max depth", SUMMARY.max_primary_depth]
      ];
      document.getElementById("stats").innerHTML = stats.map(item =>
        '<div><b>' + escapeHtml(item[1]) + '</b><span>' + escapeHtml(item[0]) + '</span></div>'
      ).join("");
    }}

    function renderFilters() {{
      const branchSelect = document.getElementById("branchFilter");
      const rootSelect = document.getElementById("rootFilter");
      const branches = [...new Set(DATA.map(node => node.branch))].sort();
      const roots = [...new Map(DATA.map(node => [node.root, node.rootName])).entries()]
        .sort((a, b) => a[1].localeCompare(b[1]));
      branchSelect.innerHTML = '<option value="all">All inferred branches</option>' +
        branches.map(branch => '<option value="' + escapeHtml(branch) + '">' + escapeHtml(branch) + '</option>').join("");
      rootSelect.innerHTML = '<option value="all">All lineage roots</option>' +
        roots.map(entry => '<option value="' + escapeHtml(entry[0]) + '">' + escapeHtml(entry[1]) + '</option>').join("");
    }}

    function renderTree() {{
      const visible = visibleIds();
      if (searchText || branchFilter !== "all" || rootFilter !== "all") {{
        visible.forEach(id => ancestorIds(id).forEach(parent => collapsed.delete(parent)));
      }}
      const tree = document.getElementById("tree");
      tree.innerHTML = "";
      let shown = 0;

      function addNode(node, depth) {{
        if (!visible.has(node.id)) return;
        shown += 1;
        const childList = (children.get(node.id) || []).filter(child => visible.has(child.id));
        const hasChildren = childList.length > 0;
        const isCollapsed = collapsed.has(node.id);
        const row = document.createElement("div");
        row.className = "row" + (node.id === selectedId ? " selected" : "");
        row.style.setProperty("--branch-color", branchColor(node.branch));
        row.innerHTML =
          '<div class="indent" style="--depth:' + depth + '"></div>' +
          '<button class="toggle ' + (hasChildren ? "" : "placeholder") + '" type="button" title="' + (isCollapsed ? "Expand" : "Collapse") + '">' + (isCollapsed ? "›" : "⌄") + '</button>' +
          '<div class="label"><span class="swatch"></span><span class="name">' + highlight(node.name) + '</span><span class="id">' + highlight(node.id) + '</span></div>' +
          '<div class="badges"><span class="badge">D' + escapeHtml(node.depth) + '</span>' + (node.allParents.length > 1 ? '<span class="badge multi">' + escapeHtml(node.allParents.length) + ' src</span>' : '') + '</div>';
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
        tree.appendChild(row);
        if (hasChildren && !isCollapsed) childList.forEach(child => addNode(child, depth + 1));
      }}

      (children.get(ROOT_ID) || []).forEach(root => addNode(root, 0));
      if (!shown) tree.innerHTML = '<div class="empty">No matching nodes</div>';
      document.getElementById("treeCount").textContent = shown + " / " + DATA.length + " nodes";
    }}

    function gotoNode(id) {{
      if (!byId.has(id)) return;
      selectedId = id;
      ancestorIds(id).forEach(parent => collapsed.delete(parent));
      renderTree();
      renderDetails();
    }}

    function linkedList(ids) {{
      if (!ids.length) return '<span class="lineage">None</span>';
      return '<div class="lineage-list">' + ids.map(id => {{
        const node = byId.get(id);
        if (!node) return "";
        return '<button type="button" data-node="' + escapeHtml(id) + '">' + escapeHtml(node.name) + ' <code>' + escapeHtml(id) + '</code></button>';
      }}).join("") + '</div>';
    }}

    function renderDetails() {{
      const node = byId.get(selectedId) || DATA[0];
      if (!node) return;
      const primaryParent = node.parent ? byId.get(node.parent) : null;
      const childCount = node.children.length;
      const descendants = descendantIds(node.id).length;
      const lineage = [...ancestorIds(node.id)].reverse().map(id => byId.get(id)).filter(Boolean).concat(node);
      const details = document.getElementById("details");
      details.style.setProperty("--branch-color", branchColor(node.branch));
      details.innerHTML =
        '<div class="title"><span class="color"></span><div><h2>' + escapeHtml(node.name) + '</h2><code>' + escapeHtml(node.id) + '</code></div></div>' +
        '<div class="meta-grid">' +
          '<div><b>D' + escapeHtml(node.depth) + '</b><span>Primary depth</span></div>' +
          '<div><b>' + escapeHtml(node.component) + '</b><span>Component</span></div>' +
          '<div><b>' + escapeHtml(childCount) + '</b><span>Direct children</span></div>' +
          '<div><b>' + escapeHtml(descendants) + '</b><span>Descendants</span></div>' +
        '</div>' +
        '<div class="field"><label>Primary lineage parent</label>' + (primaryParent ? escapeHtml(primaryParent.name) + ' <code>' + escapeHtml(primaryParent.id) + '</code>' : "None") + '</div>' +
        '<div class="field"><label>All lineage sources</label>' + linkedList(node.allParents) + '</div>' +
        '<div class="field"><label>Lineage children</label>' + linkedList(node.children) + '</div>' +
        '<div class="field"><label>Root</label>' + escapeHtml(node.rootName) + ' <code>' + escapeHtml(node.root) + '</code></div>' +
        '<div class="field"><label>Inferred branch</label>' + escapeHtml(node.branch) + '</div>' +
        '<div class="field"><label>Nearest skeleton anchor by is_a</label>' + (node.anchor ? escapeHtml(node.anchorName) + ' <code>' + escapeHtml(node.anchor) + '</code>' : "None") + '</div>' +
        '<div class="field"><label>Primary lineage path</label><div class="lineage">' + lineage.map(item => escapeHtml(item.name)).join(" -> ") + '</div></div>' +
        '<div class="legend">' + renderLegend() + '</div>';
      details.querySelectorAll("button[data-node]").forEach(button => {{
        button.addEventListener("click", () => gotoNode(button.getAttribute("data-node")));
      }});
    }}

    function renderLegend() {{
      const branches = [...new Set(DATA.map(node => node.branch))].sort();
      return branches.map(branch =>
        '<span style="--branch-color:' + branchColor(branch) + '"><i></i>' + escapeHtml(branch) + '</span>'
      ).join("");
    }}

    document.getElementById("search").addEventListener("input", event => {{
      searchText = event.target.value.trim().toLowerCase();
      renderTree();
    }});
    document.getElementById("branchFilter").addEventListener("change", event => {{
      branchFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("rootFilter").addEventListener("change", event => {{
      rootFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("expandAll").addEventListener("click", () => {{
      collapsed = new Set();
      renderTree();
    }});
    document.getElementById("collapseAll").addEventListener("click", () => {{
      collapsed = new Set(DATA.map(node => node.id));
      renderTree();
    }});
    document.getElementById("reset").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.getElementById("branchFilter").value = "all";
      document.getElementById("rootFilter").value = "all";
      searchText = "";
      branchFilter = "all";
      rootFilter = "all";
      selectedId = DATA[0]?.id || "";
      collapsed = new Set(DATA.filter(node => node.depth > 2).map(node => node.id));
      renderTree();
      renderDetails();
    }});

    renderStats();
    renderFilters();
    renderTree();
    renderDetails();
  </script>
</body>
</html>
"""
    (OUT_DIR / "lineage_catalog.html").write_text(html_doc, encoding="utf-8")


def write_readme(summary: dict[str, object]) -> None:
    (OUT_DIR / "README.md").write_text(
        f"""# CL Develops-From Lineage Catalog

This directory extracts Cell Ontology lineage relations from `cl-basic.obo`.

Lineage relation:

- `relationship: RO:0002202`
- label: `develops_from`
- display direction: source cell -> derived cell

Files:

- `lineage_nodes.tsv`: unique CL terms participating in lineage relations.
- `lineage_edges.tsv`: direct `develops_from` edges.
- `excluded_non_human_lineage_terms.tsv`: explicitly non-human CL terms removed
  before lineage extraction.
- `lineage_catalog.md`: tree-like lineage catalog using one primary parent per node.
- `lineage_catalog.html`: interactive lineage visualization.
- `summary.json`: generation statistics.

Important modeling note:

The CL lineage graph is a DAG, not a strict tree. Nodes with multiple
`develops_from` sources are shown once under a deterministic primary parent;
all lineage sources remain available in `lineage_nodes.tsv`,
`lineage_edges.tsv`, and the HTML detail panel.

Human filtering note:

Generic CL terms without an explicit species/taxon label are retained. Terms
with explicit non-human labels such as mouse/Mmus, rat, Nematoda/Protostomia,
Arthropoda, invertebrate, insect, Fungi, Caenorhabditis, or Drosophila are
removed before lineage extraction. Parenthetical `sensu ...` taxon-scoped terms
are also removed unless they explicitly mention human. Explicit human terms are
retained.

Current summary:

- Lineage nodes: {summary["lineage_nodes"]}
- Lineage edges: {summary["lineage_edges"]}
- Excluded non-human CL terms: {summary["excluded_non_human_terms"]}
- Lineage roots: {summary["lineage_roots"]}
- Connected components: {summary["connected_components"]}
- Nodes with multiple lineage sources: {summary["nodes_with_multiple_lineage_sources"]}

Regenerate:

```bash
python3 scripts/build_cl_lineage_catalog.py
```
""",
        encoding="utf-8",
    )


def main() -> None:
    nodes, edge_rows, excluded_non_human, summary = build_lineage_catalog()
    write_tsv(nodes, edge_rows, excluded_non_human, summary)
    write_markdown(nodes)
    write_html(nodes, summary)
    write_readme(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
