#!/usr/bin/env python3
"""Build a CL-official-slim based skeleton and expanded catalog."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import build_cell_catalog_skeleton as skeleton_builder  # noqa: E402
import build_expanded_cell_catalog as expanded_builder  # noqa: E402


SKELETON_OUT_DIR = ROOT / "cell_ontology" / "official_slim_skeleton"
EXPANDED_OUT_DIR = ROOT / "cell_ontology" / "official_slim_expanded"


def build_skeleton() -> None:
    skeleton_builder.SKELETON = skeleton_builder.rows_to_nodes(
        skeleton_builder.OFFICIAL_SLIM_ROWS
    )
    skeleton_builder.OUT_DIR = SKELETON_OUT_DIR
    skeleton_builder.main()


def build_expanded() -> dict[str, object]:
    expanded_builder = importlib.import_module("build_expanded_cell_catalog")
    expanded_builder.SKELETON = skeleton_builder.rows_to_nodes(
        skeleton_builder.OFFICIAL_SLIM_ROWS
    )
    expanded_builder.OUT_DIR = EXPANDED_OUT_DIR
    nodes, excluded, summary = expanded_builder.build_catalog()
    expanded_builder.write_tsv(nodes, excluded, summary)
    expanded_builder.write_markdown(nodes)
    expanded_builder.write_html(nodes, summary)
    expanded_builder.write_readme(summary)
    return summary


def write_readme(summary: dict[str, object]) -> None:
    (SKELETON_OUT_DIR / "README.md").write_text(
        """# Official Slim Cell Catalog Skeleton

This directory contains a CL official-slim based alternative skeleton.

The top-level branches start from `general_cell_types_upper_slim`, then selected
domain branches are supplemented with CL-maintained upper slims such as
`blood_and_immune_upper_slim`, plus broad HRA/CELLxGENE-supported categories for
neural, epithelial, stromal, germline, endocrine, and muscle branches.

This directory is generated separately from `cell_ontology/skeleton` so the
legacy AI-curated skeleton remains available for comparison.

Regenerate:

```bash
python3 scripts/build_official_slim_cell_catalog.py
```
""",
        encoding="utf-8",
    )
    (EXPANDED_OUT_DIR / "README.md").write_text(
        f"""# Official Slim Expanded Cell Catalog

This directory expands the CL official-slim based skeleton with filtered CL
`is_a` descendants.

It is generated separately from `cell_ontology/expanded` so the legacy expanded
catalog remains available for comparison.

Current summary:

- Included nodes: {summary["included_nodes"]}
- Skeleton nodes: {summary["skeleton_nodes"]}
- CL extension nodes: {summary["cl_extension_nodes"]}
- Excluded CL terms: {summary["excluded_terms"]}

Regenerate:

```bash
python3 scripts/build_official_slim_cell_catalog.py
```
""",
        encoding="utf-8",
    )


def main() -> None:
    build_skeleton()
    summary = build_expanded()
    write_readme(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Wrote official slim skeleton to {SKELETON_OUT_DIR}")
    print(f"Wrote official slim expanded catalog to {EXPANDED_OUT_DIR}")


if __name__ == "__main__":
    main()
