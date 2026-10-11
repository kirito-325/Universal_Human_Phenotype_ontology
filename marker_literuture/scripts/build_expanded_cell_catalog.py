#!/usr/bin/env python3
"""Build the expanded catalog with controlled functional T-cell granularity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "CELL_LINEAGE_SOURCE_ROOT",
        str(ROOT.parent / "CPO_construction" / "cell_lineage"),
    )
)

FUNCTIONAL_T_CELL_TERM_IDS = frozenset(
    {
        "CL:0000492",
        "CL:0000624",
        "CL:0000625",
        "CL:0000794",
        "CL:0000795",
        "CL:0000895",
        "CL:0000896",
        "CL:0000897",
        "CL:0000900",
        "CL:0000904",
        "CL:0000905",
        "CL:0000906",
        "CL:0000907",
        "CL:0000908",
        "CL:0000909",
        "CL:0000913",
        "CL:0000934",
        "CL:0001044",
        "CL:0001050",
        "CL:0001062",
        "CL:0001087",
        "CL:0020030",
        "CL:0020031",
        "CL:0020033",
        "CL:0020034",
    }
)

OVERFINE_T_CELL_TERM_IDS = frozenset(
    {
        "CL:0000792",
        "CL:0000919",
        "CL:0000920",
        "CL:0001041",
        "CL:0001047",
        "CL:0001051",
        "CL:0001052",
        "CL:0001203",
        "CL:0001204",
    }
)

EXPECTED_PARENTS = {
    "CL:0000624": "CL:0000791",
    "CL:0000625": "CL:0000791",
    "CL:0000492": "CL:0000624",
    "CL:0000545": "CL:0000492",
    "CL:0000908": "CL:0000625",
    "CL:0000917": "CL:0000908",
    "CL:0000904": "CL:0000897",
    "CL:0000907": "CL:0000909",
}


def load_source_builder(source_root: Path) -> ModuleType:
    builder_path = source_root / "scripts" / "build_expanded_cell_catalog.py"
    if not builder_path.is_file():
        raise FileNotFoundError(f"Source builder not found: {builder_path}")
    module_name = "_uhpo_expanded_cell_catalog_builder"
    spec = importlib.util.spec_from_file_location(module_name, builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load source builder: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_functional_t_cell_policy(builder: ModuleType) -> None:
    original_filter = builder.filter_reasons

    def filter_reasons(term: object, skeleton_ids: set[str]) -> list[str]:
        reasons = original_filter(term, skeleton_ids)
        if term.id in FUNCTIONAL_T_CELL_TERM_IDS:
            reasons = [
                reason
                for reason in reasons
                if reason not in {"marker_specific", "anatomy_or_tissue_specific"}
            ]
        return reasons

    builder.filter_reasons = filter_reasons


def read_obo_version(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("data-version: "):
                return line.removeprefix("data-version: ").strip()
            if line == "[Term]\n":
                break
    return ""


def validate_catalog(nodes: list[object], excluded: list[dict[str, object]]) -> None:
    by_id = {node.id: node for node in nodes}
    excluded_by_id = {str(row["term_id"]): row for row in excluded}
    missing = sorted(FUNCTIONAL_T_CELL_TERM_IDS - by_id.keys())
    if missing:
        raise RuntimeError(f"Functional T-cell terms were not retained: {missing}")
    leaked = sorted(OVERFINE_T_CELL_TERM_IDS & by_id.keys())
    if leaked:
        raise RuntimeError(f"Over-fine marker-defined T-cell terms were retained: {leaked}")
    for term_id in OVERFINE_T_CELL_TERM_IDS:
        reasons = str(excluded_by_id.get(term_id, {}).get("reason", ""))
        if "marker_specific" not in reasons:
            raise RuntimeError(f"Expected marker-specific exclusion for {term_id}: {reasons}")
    for term_id, parent_id in EXPECTED_PARENTS.items():
        actual_parent = by_id[term_id].parent_id
        if actual_parent != parent_id:
            raise RuntimeError(
                f"Unexpected parent for {term_id}: {actual_parent}; expected {parent_id}"
            )


def write_readme(output_dir: Path, summary: dict[str, object], source_root: Path) -> None:
    source_hint = source_root / "cell_ontology" / "cl-basic.obo"
    output_dir.joinpath("README.md").write_text(
        f"""# Expanded Cell Catalog

This directory extends the reviewed skeleton with filtered Cell Ontology `is_a`
descendants.

Files:

- `expanded_cell_catalog.tsv`: included skeleton and CL extension nodes.
- `expanded_cell_catalog.md`: text tree.
- `expanded_cell_catalog.html`: interactive tree visualization.
- `excluded_cl_terms.tsv`: CL terms excluded from the expansion and the reason.
- `summary.json`: generation statistics and ontology version.

Rules:

- Skeleton nodes are always retained.
- CL descendants are inherited through `is_a` links.
- Each CL term is assigned to one unique parent in the generated tree.
- Explicitly non-human, transcriptomic-cluster-like, non-cell, non-target-taxon,
  overly deep, and anatomy/tissue-specific terms remain excluded.
- Marker-defined terms remain excluded by default.
- CD4 and CD8 are treated as core mature alpha-beta T-cell lineage axes rather
  than optional markers.
- CD4/CD8 T-cell terms defined by helper, cytotoxic, regulatory, naive, memory,
  effector, activated, exhausted, or terminal differentiation functions/states
  are retained through a reviewed CL-ID allowlist.
- Terms adding CD25, CD28, CCR4, CCR6, CXCR3, CD45RO, or comparable extra marker
  combinations remain excluded.

Current summary:

- Included nodes: {summary["included_nodes"]}
- Skeleton nodes: {summary["skeleton_nodes"]}
- CL extension nodes: {summary["cl_extension_nodes"]}
- Functional CD4/CD8 T-cell exceptions: {summary["functional_t_cell_exception_nodes"]}
- Excluded CL terms: {summary["excluded_terms"]}
- Cell Ontology version: {summary["cl_data_version"]}

Regenerate:

```bash
python3 scripts/build_expanded_cell_catalog.py \\
  --source-root "{source_root}"
```

The current source ontology is `{source_hint}`.
""",
        encoding="utf-8",
    )


def build(source_root: Path, output_dir: Path) -> dict[str, object]:
    builder = load_source_builder(source_root)
    install_functional_t_cell_policy(builder)
    builder.OUT_DIR = output_dir
    nodes, excluded, summary = builder.build_catalog()
    validate_catalog(nodes, excluded)
    summary["functional_t_cell_exception_nodes"] = len(FUNCTIONAL_T_CELL_TERM_IDS)
    summary["functional_t_cell_exception_ids"] = sorted(FUNCTIONAL_T_CELL_TERM_IDS)
    summary["cl_data_version"] = read_obo_version(builder.OBO_PATH)
    builder.write_tsv(nodes, excluded, summary)
    builder.write_markdown(nodes)
    builder.write_html(nodes, summary)
    write_readme(output_dir, summary, source_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the expanded cell catalog with functional T-cell granularity."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "expanded")
    args = parser.parse_args()
    summary = build(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
