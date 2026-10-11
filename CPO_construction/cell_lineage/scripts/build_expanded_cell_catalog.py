#!/usr/bin/env python3
"""Expand the reviewed catalog skeleton with filtered Cell Ontology descendants."""

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

from build_cell_catalog_skeleton import COLORS, OBO_PATH, SKELETON, depth_by_id  # noqa: E402


OUT_DIR = ROOT / "cell_ontology" / "expanded"
MAX_EXTENSION_DISTANCE = 6


@dataclass
class Term:
    id: str = ""
    name: str = ""
    parents: list[str] = field(default_factory=list)
    obsolete: bool = False
    line: int = 0


@dataclass
class CatalogNode:
    id: str
    parent_id: str
    label: str
    label_zh: str
    cl_name: str
    branch: str
    rank: str
    source: str
    anchor_id: str
    distance_to_anchor: int
    note: str
    order: int


NON_HUMAN_RE = re.compile(
    r"\((Mmus|Primate|Mus)\)|,\s*(mouse|murine|rat)$|"
    r"\b("
    r"mouse|murine|rat|macaque|non-human primate|"
    r"Mmus|Nematoda|Protostomia|Arthropoda|Endopterygota|"
    r"vertebrate|Vertebrata|invertebrate|insect|Teleostei|teleost|"
    r"Fungi|fungal|Caenorhabditis|Drosophila|zebrafish"
    r")\b",
    re.IGNORECASE,
)
NON_HUMAN_SENSU_RE = re.compile(r"\(sensu [^)]+\)", re.IGNORECASE)
EXPLICIT_HUMAN_RE = re.compile(r"\b(Homo sapiens|human)\b", re.IGNORECASE)
MARKER_RE = re.compile(
    r"\b(CD\d+[a-zA-Z-]*|CD\d+[+-]|positive|negative|high|low|dim|bright|"
    r"IgA|IgD|IgE|IgG|IgM|B220|CCR\d|CXCR\d|Sca1|Kit-positive)\b",
    re.IGNORECASE,
)
TRANSCRIPTOMIC_RE = re.compile(r"\bNN_\d+\b|\b[A-Z]{2,}[-_][A-Z0-9]")
CLUSTER_WORD_RE = re.compile(r"\b(cluster|subcluster)\b", re.IGNORECASE)
NON_CELL_RE = re.compile(
    r"(cytoplasm|nucleus|heterochromatin|junction nucleus|myotendinous junction)",
    re.IGNORECASE,
)
NON_TARGET_RE = re.compile(
    r"\b(fungal|spore|conidium|prokaryotic|heterocyst|photosynthetic|"
    r"nitrogen fixing|actinomycete|hyphal|Eumycetozoan|bacterial)\b",
    re.IGNORECASE,
)
RELATIONAL_ANATOMY_RE = re.compile(r"\b(of|in|from)\b", re.IGNORECASE)
ANATOMY_WORD_RE = re.compile(
    r"\b("
    r"lung|pulmonary|airway|respiratory|trachea|tracheal|tracheobronchial|"
    r"bronchus|bronchial|bronchiolar|bronchiole|alveolar|submucosal gland|"
    r"kidney|renal|perirenal|interrenal|liver|hepatic|biliary|"
    r"heart|cardiac valve|brain|central nervous system|peripheral nervous system|"
    r"CNS|PNS|forebrain|midbrain|hindbrain|brainstem|hypothalamus|"
    r"cortex|cortical|cerebellar|hypothalam|striatal|hippocampal|"
    r"amygdala|leptomeningeal|leptomeninx|dura mater|retina|retinal|"
    r"eye|compound eye|conjunctival|"
    r"skin|dermal|epidermal|hair follicle|mammary|pancreas|pancreatic|"
    r"stomach|gastric|gastrointestinal|intestinal|intestin|colon|colonic|"
    r"ileum|ileal|duodenum|duodenal|lamina propria|fallopian tube|"
    r"spleen|splenic|thymus|thymic|"
    r"lymph node|bone marrow|ovary|ovarian|testis|testicular|uterus|uterine|"
    r"placenta|placental|amnion|prostate|bladder|ureter|salivary|tooth|"
    r"dental|adrenal|pituitary|olfactory|substantia|pectoralis|insect"
    r")\b",
    re.IGNORECASE,
)


def parse_obo(path: Path) -> dict[str, Term]:
    terms: list[Term] = []
    current: Term | None = None
    in_term = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
            current.parents.append(raw[6:].split()[0])
        elif raw.startswith("is_obsolete: true"):
            current.obsolete = True
    if current:
        terms.append(current)
    return {
        term.id: term
        for term in terms
        if term.id.startswith("CL:") and term.name and not term.obsolete
    }


def ancestor_distances(term_id: str, terms: dict[str, Term]) -> dict[str, int]:
    distances = {term_id: 0}
    queue = deque([term_id])
    while queue:
        current = queue.popleft()
        for parent in terms[current].parents:
            if parent not in terms or parent in distances:
                continue
            distances[parent] = distances[current] + 1
            queue.append(parent)
    return distances


def filter_reasons(term: Term, skeleton_ids: set[str]) -> list[str]:
    if term.id in skeleton_ids:
        return []
    reasons: list[str] = []
    name = term.name
    sensu_match = NON_HUMAN_SENSU_RE.search(name)
    has_non_human_sensu = bool(
        sensu_match and not EXPLICIT_HUMAN_RE.search(sensu_match.group(0))
    )
    if NON_HUMAN_RE.search(name) or has_non_human_sensu or term.id.startswith("CL:43"):
        reasons.append("species_specific")
    if MARKER_RE.search(name):
        reasons.append("marker_specific")
    if TRANSCRIPTOMIC_RE.search(name) or CLUSTER_WORD_RE.search(name):
        reasons.append("transcriptomic_cluster_like")
    if NON_CELL_RE.search(name):
        reasons.append("non_cell_component")
    if NON_TARGET_RE.search(name):
        reasons.append("non_target_taxon")
    if RELATIONAL_ANATOMY_RE.search(name) or ANATOMY_WORD_RE.search(name):
        reasons.append("anatomy_or_tissue_specific")
    return reasons


def choose_anchor(
    term_id: str,
    terms: dict[str, Term],
    skeleton_ids: set[str],
    skeleton_order: dict[str, int],
    skeleton_depth: dict[str, int],
) -> tuple[str, int]:
    distances = ancestor_distances(term_id, terms)
    candidates = [
        (distance, -skeleton_depth[ancestor_id], skeleton_order[ancestor_id], ancestor_id)
        for ancestor_id, distance in distances.items()
        if ancestor_id in skeleton_ids and ancestor_id != "CL:0000000"
    ]
    if not candidates:
        return "", -1
    candidates.sort()
    distance, _, _, anchor_id = candidates[0]
    return anchor_id, distance


def nearest_included_parent(
    term_id: str,
    terms: dict[str, Term],
    included_ids: set[str],
    anchor_by_id: dict[str, str],
    anchor_id: str,
    skeleton_ids: set[str],
) -> str:
    queue = deque((parent, 1) for parent in terms[term_id].parents if parent in terms)
    seen: set[str] = set()
    candidates: list[tuple[int, int, str, bool]] = []
    while queue:
        current, distance = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if current in included_ids:
            same_anchor = anchor_by_id.get(current) == anchor_id or current == anchor_id
            skeleton_bonus = 0 if current in skeleton_ids else 1
            candidates.append((distance, skeleton_bonus, current, same_anchor))
            continue
        for parent in terms[current].parents:
            if parent in terms:
                queue.append((parent, distance + 1))
    if not candidates:
        return anchor_id
    same_anchor_candidates = [candidate for candidate in candidates if candidate[3]]
    parent_candidates = same_anchor_candidates or candidates
    parent_candidates.sort()
    return parent_candidates[0][2]


def build_catalog() -> tuple[list[CatalogNode], list[dict[str, object]], dict[str, object]]:
    terms = parse_obo(OBO_PATH)
    skeleton_ids = {node.cl_id for node in SKELETON}
    skeleton_order = {node.cl_id: index for index, node in enumerate(SKELETON)}
    skeleton_by_id = {node.cl_id: node for node in SKELETON}
    skeleton_depth = depth_by_id(SKELETON)

    included_ids = set(skeleton_ids)
    anchor_by_id = {node.cl_id: node.cl_id for node in SKELETON}
    distance_by_id = {node.cl_id: 0 for node in SKELETON}
    excluded: list[dict[str, object]] = []

    for term_id, term in terms.items():
        if term_id in skeleton_ids:
            continue
        anchor_id, distance = choose_anchor(
            term_id, terms, skeleton_ids, skeleton_order, skeleton_depth
        )
        if not anchor_id:
            excluded.append(
                {
                    "term_id": term_id,
                    "term_name": term.name,
                    "reason": "no_non_root_skeleton_anchor",
                    "anchor_id": "",
                    "anchor_label": "",
                    "distance_to_anchor": "",
                }
            )
            continue
        reasons = filter_reasons(term, skeleton_ids)
        if distance > MAX_EXTENSION_DISTANCE:
            reasons.append("too_deep_from_skeleton_anchor")
        if reasons:
            excluded.append(
                {
                    "term_id": term_id,
                    "term_name": term.name,
                    "reason": ";".join(reasons),
                    "anchor_id": anchor_id,
                    "anchor_label": skeleton_by_id[anchor_id].label,
                    "distance_to_anchor": distance,
                }
            )
            continue
        included_ids.add(term_id)
        anchor_by_id[term_id] = anchor_id
        distance_by_id[term_id] = distance

    catalog_nodes: list[CatalogNode] = []
    for order, node in enumerate(SKELETON):
        catalog_nodes.append(
            CatalogNode(
                id=node.cl_id,
                parent_id=node.parent_id,
                label=node.label,
                label_zh=node.label_zh,
                cl_name=terms[node.cl_id].name,
                branch=node.branch,
                rank=node.rank,
                source="skeleton",
                anchor_id=node.cl_id,
                distance_to_anchor=0,
                note=node.note,
                order=order,
            )
        )

    extension_ids = sorted(
        included_ids - skeleton_ids,
        key=lambda item: (
            skeleton_order[anchor_by_id[item]],
            distance_by_id[item],
            terms[item].name.lower(),
            item,
        ),
    )
    extension_order_start = len(catalog_nodes)
    for offset, term_id in enumerate(extension_ids):
        anchor_id = anchor_by_id[term_id]
        parent_id = nearest_included_parent(
            term_id, terms, included_ids, anchor_by_id, anchor_id, skeleton_ids
        )
        catalog_nodes.append(
            CatalogNode(
                id=term_id,
                parent_id=parent_id,
                label=terms[term_id].name,
                label_zh="",
                cl_name=terms[term_id].name,
                branch=skeleton_by_id[anchor_id].branch,
                rank="cl_extension",
                source="cl_is_a",
                anchor_id=anchor_id,
                distance_to_anchor=distance_by_id[term_id],
                note=f"Added from CL is_a descendants of {skeleton_by_id[anchor_id].label}.",
                order=extension_order_start + offset,
            )
        )

    summary = {
        "active_cl_terms": len(terms),
        "skeleton_nodes": len(SKELETON),
        "included_nodes": len(catalog_nodes),
        "cl_extension_nodes": len(extension_ids),
        "excluded_terms": len(excluded),
        "max_extension_distance": MAX_EXTENSION_DISTANCE,
        "excluded_reason_counts": dict(
            sorted(Counter(reason for row in excluded for reason in str(row["reason"]).split(";")).items())
        ),
    }
    return catalog_nodes, excluded, summary


def children_by_parent(nodes: list[CatalogNode]) -> dict[str, list[CatalogNode]]:
    children: dict[str, list[CatalogNode]] = defaultdict(list)
    for node in nodes:
        children[node.parent_id].append(node)
    order_by_id = {node.id: node.order for node in nodes}
    for child_list in children.values():
        child_list.sort(key=lambda node: (order_by_id.get(node.id, 10**9), node.label.lower()))
    return children


def depths_for_catalog(nodes: list[CatalogNode]) -> dict[str, int]:
    by_id = {node.id: node for node in nodes}
    depths: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in depths:
            return depths[node_id]
        parent = by_id[node_id].parent_id
        depths[node_id] = 0 if not parent else depth(parent) + 1
        return depths[node_id]

    for node in nodes:
        depth(node.id)
    return depths


def write_tsv(nodes: list[CatalogNode], excluded: list[dict[str, object]], summary: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    depths = depths_for_catalog(nodes)
    rows = []
    by_id = {node.id: node for node in nodes}
    for node in sorted(nodes, key=lambda item: item.order):
        parent = by_id.get(node.parent_id)
        rows.append(
            {
                "order": node.order + 1,
                "level": depths[node.id],
                "rank": node.rank,
                "source": node.source,
                "branch": node.branch,
                "cl_id": node.id,
                "cl_name": node.cl_name,
                "catalog_label": node.label,
                "catalog_label_zh": node.label_zh,
                "parent_id": node.parent_id,
                "parent_label": parent.label if parent else "",
                "anchor_id": node.anchor_id,
                "anchor_label": by_id[node.anchor_id].label,
                "distance_to_anchor": node.distance_to_anchor,
                "note": node.note,
            }
        )
    fields = list(rows[0])
    with (OUT_DIR / "expanded_cell_catalog.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    excluded_fields = [
        "term_id",
        "term_name",
        "reason",
        "anchor_id",
        "anchor_label",
        "distance_to_anchor",
    ]
    with (OUT_DIR / "excluded_cl_terms.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=excluded_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown(nodes: list[CatalogNode]) -> None:
    children = children_by_parent(nodes)
    lines = [
        "# Expanded Cell Catalog",
        "",
        "Skeleton nodes are expanded with filtered Cell Ontology `is_a` descendants.",
        "",
    ]

    def add(node: CatalogNode, depth: int) -> None:
        zh = f" / {node.label_zh}" if node.label_zh else ""
        source = "skeleton" if node.source == "skeleton" else "CL"
        lines.append(
            f"{'  ' * depth}- {node.label}{zh} `{node.id}` ({node.rank}, {source})"
        )
        for child in children.get(node.id, []):
            add(child, depth + 1)

    for root in children[""]:
        add(root, 0)
    (OUT_DIR / "expanded_cell_catalog.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_html(nodes: list[CatalogNode], summary: dict[str, object]) -> None:
    depths = depths_for_catalog(nodes)
    data = [
        {
            "id": node.id,
            "parent": node.parent_id,
            "label": node.label,
            "labelZh": node.label_zh,
            "clName": node.cl_name,
            "rank": node.rank,
            "source": node.source,
            "branch": node.branch,
            "anchor": node.anchor_id,
            "distance": node.distance_to_anchor,
            "level": depths[node.id],
            "note": node.note,
            "order": node.order,
        }
        for node in sorted(nodes, key=lambda item: item.order)
    ]
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    colors_json = json.dumps(COLORS, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Expanded Cell Catalog</title>
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
      grid-template-columns: repeat(5, minmax(120px, 1fr));
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
      grid-template-columns: minmax(240px, 1fr) 190px 150px 150px auto auto auto;
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
      grid-template-columns: minmax(520px, 1fr) 400px;
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
    .zh, .id {{
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .zh {{ max-width: 190px; }}
    .id {{ font-size: 12px; }}
    .badges {{
      display: flex;
      align-items: center;
      gap: 5px;
      margin-left: 8px;
    }}
    .badge {{
      height: 22px;
      min-width: 46px;
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
    .badge.source-skeleton {{
      color: #325d78;
      background: #e8f3fa;
      border-color: #c8ddec;
    }}
    .badge.source-cl_is_a {{
      color: #5d4c22;
      background: #fff5d8;
      border-color: #e8d59b;
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
    .detail-zh {{
      color: var(--muted);
      margin-top: 2px;
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
      .name, .zh {{ white-space: normal; max-width: none; }}
      .swatch {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Expanded Cell Catalog</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search name, Chinese label, or CL ID">
    <select id="branchFilter" aria-label="Filter by branch"></select>
    <select id="sourceFilter" aria-label="Filter by source"></select>
    <select id="rankFilter" aria-label="Filter by rank"></select>
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
    const COLORS = {colors_json};
    const SUMMARY = {summary_json};
    const byId = new Map(DATA.map(node => [node.id, node]));
    const children = new Map();
    DATA.forEach(node => {{
      if (!children.has(node.parent)) children.set(node.parent, []);
      children.get(node.parent).push(node);
    }});
    for (const list of children.values()) list.sort((a, b) => a.order - b.order);

    const root = DATA.find(node => !node.parent) || DATA[0];
    let selectedId = root.id;
    let searchText = "";
    let branchFilter = "all";
    let sourceFilter = "all";
    let rankFilter = "all";
    let collapsed = new Set(DATA.filter(node => node.parent && node.level > 2).map(node => node.id));

    function branchColor(branch) {{
      return COLORS[branch] || "#647085";
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function highlight(value) {{
      const safe = escapeHtml(value || "");
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
      const text = [
        node.label,
        node.labelZh,
        node.clName,
        node.id,
        node.branch,
        node.rank,
        node.source
      ].join(" ").toLowerCase();
      const searchOk = !searchText || text.includes(searchText);
      const branchOk = branchFilter === "all" || node.branch === branchFilter;
      const sourceOk = sourceFilter === "all" || node.source === sourceFilter;
      const rankOk = rankFilter === "all" || node.rank === rankFilter;
      return searchOk && branchOk && sourceOk && rankOk;
    }}

    function visibleIds() {{
      const ids = new Set();
      DATA.forEach(node => {{
        if (nodeMatches(node)) {{
          ids.add(node.id);
          ancestorIds(node.id).forEach(parent => ids.add(parent));
        }}
      }});
      if (!searchText && branchFilter === "all" && sourceFilter === "all" && rankFilter === "all") {{
        DATA.forEach(node => ids.add(node.id));
      }}
      return ids;
    }}

    function renderStats() {{
      const stats = [
        ["Included", SUMMARY.included_nodes],
        ["Skeleton", SUMMARY.skeleton_nodes],
        ["CL added", SUMMARY.cl_extension_nodes],
        ["Excluded", SUMMARY.excluded_terms],
        ["Max CL depth", SUMMARY.max_extension_distance]
      ];
      document.getElementById("stats").innerHTML = stats.map(item =>
        '<div><b>' + escapeHtml(item[1]) + '</b><span>' + escapeHtml(item[0]) + '</span></div>'
      ).join("");
    }}

    function renderFilters() {{
      const branchSelect = document.getElementById("branchFilter");
      const sourceSelect = document.getElementById("sourceFilter");
      const rankSelect = document.getElementById("rankFilter");
      const branches = [...new Set(DATA.map(node => node.branch))];
      const sources = [...new Set(DATA.map(node => node.source))];
      const ranks = [...new Set(DATA.map(node => node.rank))];
      branchSelect.innerHTML = '<option value="all">All branches</option>' +
        branches.map(branch => '<option value="' + escapeHtml(branch) + '">' + escapeHtml(branch) + '</option>').join("");
      sourceSelect.innerHTML = '<option value="all">All sources</option>' +
        sources.map(source => '<option value="' + escapeHtml(source) + '">' + escapeHtml(source) + '</option>').join("");
      rankSelect.innerHTML = '<option value="all">All ranks</option>' +
        ranks.map(rank => '<option value="' + escapeHtml(rank) + '">' + escapeHtml(rank) + '</option>').join("");
    }}

    function renderTree() {{
      const visible = visibleIds();
      if (searchText || branchFilter !== "all" || sourceFilter !== "all" || rankFilter !== "all") {{
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
          '<div class="label"><span class="swatch"></span><span class="name">' + highlight(node.label) + '</span><span class="zh">' + highlight(node.labelZh) + '</span><span class="id">' + highlight(node.id) + '</span></div>' +
          '<div class="badges"><span class="badge source-' + escapeHtml(node.source) + '">' + escapeHtml(node.source === "skeleton" ? "base" : "CL") + '</span><span class="badge">' + escapeHtml(node.rank) + '</span></div>';
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

      addNode(root, 0);
      if (!shown) tree.innerHTML = '<div class="empty">No matching nodes</div>';
      document.getElementById("treeCount").textContent = shown + " / " + DATA.length + " nodes";
    }}

    function renderDetails() {{
      const node = byId.get(selectedId) || root;
      const parent = node.parent ? byId.get(node.parent) : null;
      const anchor = byId.get(node.anchor);
      const childCount = (children.get(node.id) || []).length;
      const descendants = descendantIds(node.id).length;
      const lineage = [...ancestorIds(node.id)].reverse().map(id => byId.get(id)).filter(Boolean).concat(node);
      const details = document.getElementById("details");
      details.style.setProperty("--branch-color", branchColor(node.branch));
      details.innerHTML =
        '<div class="title"><span class="color"></span><div><h2>' + escapeHtml(node.label) + '</h2><div class="detail-zh">' + escapeHtml(node.labelZh) + '</div></div></div>' +
        '<div class="field"><label>CL ID</label><code>' + escapeHtml(node.id) + '</code></div>' +
        '<div class="meta-grid">' +
          '<div><b>' + escapeHtml(node.source === "skeleton" ? "base" : "CL") + '</b><span>Source</span></div>' +
          '<div><b>' + escapeHtml(node.rank) + '</b><span>Rank</span></div>' +
          '<div><b>' + escapeHtml(childCount) + '</b><span>Children</span></div>' +
          '<div><b>' + escapeHtml(descendants) + '</b><span>Descendants</span></div>' +
        '</div>' +
        '<div class="field"><label>CL name</label>' + escapeHtml(node.clName) + '</div>' +
        '<div class="field"><label>Branch</label>' + escapeHtml(node.branch) + '</div>' +
        '<div class="field"><label>Parent</label>' + (parent ? escapeHtml(parent.label) + ' <code>' + escapeHtml(parent.id) + '</code>' : "None") + '</div>' +
        '<div class="field"><label>Skeleton anchor</label>' + (anchor ? escapeHtml(anchor.label) + ' <code>' + escapeHtml(anchor.id) + '</code>' : "") + '</div>' +
        '<div class="field"><label>CL distance to anchor</label>' + escapeHtml(node.distance) + '</div>' +
        '<div class="field"><label>Lineage</label><div class="lineage">' + lineage.map(item => escapeHtml(item.label)).join(" / ") + '</div></div>' +
        '<div class="field"><label>Note</label>' + escapeHtml(node.note || "") + '</div>' +
        '<div class="legend">' + renderLegend() + '</div>';
    }}

    function renderLegend() {{
      const branches = [...new Set(DATA.map(node => node.branch))];
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
    document.getElementById("sourceFilter").addEventListener("change", event => {{
      sourceFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("rankFilter").addEventListener("change", event => {{
      rankFilter = event.target.value;
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
      document.getElementById("branchFilter").value = "all";
      document.getElementById("sourceFilter").value = "all";
      document.getElementById("rankFilter").value = "all";
      searchText = "";
      branchFilter = "all";
      sourceFilter = "all";
      rankFilter = "all";
      selectedId = root.id;
      collapsed = new Set(DATA.filter(node => node.parent && node.level > 2).map(node => node.id));
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
    (OUT_DIR / "expanded_cell_catalog.html").write_text(html_doc, encoding="utf-8")


def write_readme(summary: dict[str, object]) -> None:
    (OUT_DIR / "README.md").write_text(
        f"""# Expanded Cell Catalog

This directory extends the reviewed skeleton with filtered Cell Ontology `is_a`
descendants.

Files:

- `expanded_cell_catalog.tsv`: included skeleton and CL extension nodes.
- `expanded_cell_catalog.md`: text tree.
- `expanded_cell_catalog.html`: interactive tree visualization.
- `excluded_cl_terms.tsv`: CL terms excluded from the expansion and the reason.
- `summary.json`: generation statistics.

Rules:

- Skeleton nodes are always retained.
- CL descendants are inherited through `is_a` links.
- Each CL term is assigned to one unique parent in the generated tree.
- Terms are filtered out if they are explicitly non-human species/taxon-scoped,
  marker-specific, transcriptomic-cluster-like, non-cell components, non-target
  taxa, too deep from a skeleton anchor, or anatomy/tissue-specific.
- Explicit human / Homo sapiens terms are retained; generic CL terms without a
  species label are also retained.
- The tissue filter intentionally removes terms such as organ-specific
  `... of lung` / `lung ...` cell types unless they are already curated into
  the skeleton.

Current summary:

- Included nodes: {summary["included_nodes"]}
- Skeleton nodes: {summary["skeleton_nodes"]}
- CL extension nodes: {summary["cl_extension_nodes"]}
- Excluded CL terms: {summary["excluded_terms"]}

Regenerate:

```bash
python3 scripts/build_expanded_cell_catalog.py
```
""",
        encoding="utf-8",
    )


def main() -> None:
    nodes, excluded, summary = build_catalog()
    write_tsv(nodes, excluded, summary)
    write_markdown(nodes)
    write_html(nodes, summary)
    write_readme(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
