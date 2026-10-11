from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celltype_configure.common import CellCatalog, load_json, write_tsv  # noqa: E402


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalized_entropy(weights: Sequence[float]) -> float:
    positive = [value for value in weights if value > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probabilities = [value / total for value in positive]
    return -sum(value * math.log(value) for value in probabilities) / math.log(
        len(probabilities)
    )


def aggregate_database(rows: Iterable[Mapping[str, str]], config: Mapping) -> dict[str, dict]:
    by_go: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("is_significant", "").lower() == "true":
            by_go[row["go_id"]].append(row)

    result: dict[str, dict] = {}
    for go_id, go_rows in by_go.items():
        experiment_anchor_best: dict[tuple[str, str], float] = {}
        labels: dict[str, str] = {}
        for row in go_rows:
            anchor_id = row["catalog_anchor_id"]
            key = (row["experiment_id"], anchor_id)
            weight = float(row.get("evidence_weight", 1.0) or 1.0)
            experiment_anchor_best[key] = max(weight, experiment_anchor_best.get(key, 0.0))
            labels[anchor_id] = row.get("catalog_anchor_label", "")
        anchor_weights: dict[str, float] = defaultdict(float)
        anchor_experiments: dict[str, set[str]] = defaultdict(set)
        for (experiment_id, anchor_id), weight in experiment_anchor_best.items():
            anchor_weights[anchor_id] += weight
            anchor_experiments[anchor_id].add(experiment_id)
        top_anchor = max(anchor_weights, key=anchor_weights.get)
        total_weight = sum(anchor_weights.values())
        top_share = anchor_weights[top_anchor] / total_weight if total_weight else 0.0
        entropy = normalized_entropy(list(anchor_weights.values()))
        experiments = {row["experiment_id"] for row in go_rows}
        specific = (
            len(anchor_experiments[top_anchor])
            >= int(config.get("specific_min_experiments", 2))
            and top_share >= float(config.get("specific_min_top_anchor_share", 0.65))
            and entropy <= float(config.get("specific_max_entropy", 0.65))
        )
        general = (
            len(experiments) >= int(config.get("general_min_experiments", 3))
            and len(anchor_weights) >= int(config.get("general_min_anchors", 4))
            and entropy >= float(config.get("general_min_entropy", 0.75))
        )
        classification = (
            "cell_type_specific"
            if specific
            else "cell_general"
            if general
            else "insufficient"
        )
        result[go_id] = {
            "classification": classification,
            "top_anchor_id": top_anchor,
            "top_anchor_label": labels[top_anchor],
            "top_anchor_share": top_share,
            "entropy": entropy,
            "experiment_count": len(experiments),
            "top_anchor_experiment_count": len(anchor_experiments[top_anchor]),
            "anchor_count": len(anchor_weights),
            "significant_test_count": len(go_rows),
        }
    return result


def compatible(catalog: CellCatalog, left: str, right: str) -> bool:
    if not left or not right:
        return False
    return (
        left == right
        or left in catalog.ancestors(right)
        or right in catalog.ancestors(left)
    )


def deeper(catalog: CellCatalog, left: str, right: str) -> str:
    if left not in catalog.nodes:
        return right
    if right not in catalog.nodes:
        return left
    return left if catalog.nodes[left].level >= catalog.nodes[right].level else right


def split_ids(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def fuse_row(
    semantic: Mapping[str, str], database: Mapping[str, object], catalog: CellCatalog
) -> tuple[dict, list[str]]:
    semantic_class = semantic["semantic_classification"]
    semantic_root = semantic.get("semantic_constraint_cl_id", "")
    direct_roots = split_ids(semantic.get("explicit_catalog_cl_ids", ""))
    direct_common = semantic.get("explicit_constraint_cl_id", "")
    db_class = str(database.get("classification", "insufficient"))
    db_root = str(database.get("top_anchor_id", ""))
    review = False
    reason = ""
    confidence = "low"
    roots: list[str] = []

    if direct_roots:
        roots = [direct_common] if direct_common else direct_roots
        classification = "cell_type_specific"
        confidence = "high"
        reason = "GO curator CL reference"
        if len(roots) > 1:
            review = True
            confidence = "medium"
            reason += "; references span unrelated catalog branches"
        supporting_root = roots[0] if len(roots) == 1 else ""
        if db_class == "cell_type_specific" and supporting_root:
            if compatible(catalog, supporting_root, db_root):
                reason += "; SCEA enrichment agrees"
            else:
                review = True
                reason += "; SCEA enrichment conflicts"
    elif db_class == "cell_type_specific":
        classification = "cell_type_specific"
        roots = [db_root]
        confidence = "medium"
        reason = "repeated concentrated SCEA enrichment"
        if semantic_class in {"cell_type_specific", "cell_type_specific_candidate"}:
            if compatible(catalog, semantic_root, db_root):
                roots = [deeper(catalog, semantic_root, db_root)]
                confidence = "high"
                reason += "; SapBERT agrees"
            else:
                review = True
                reason += "; SapBERT conflicts"
    elif semantic_class in {"cell_type_specific", "cell_type_specific_candidate"}:
        classification = "cell_type_specific_candidate"
        roots = [semantic_root]
        confidence = "medium"
        review = True
        reason = "SapBERT-only candidate; database or curator confirmation required"
        if db_class == "cell_general":
            classification = "conflicting_evidence"
            reason = "SapBERT specific signal conflicts with broad SCEA enrichment"
    elif db_class == "cell_general":
        classification = "cell_general"
        confidence = "medium"
        reason = "broad reproducible SCEA enrichment"
    elif semantic_class == "cell_general_candidate":
        classification = "cell_general_candidate"
        review = True
        reason = "no strong semantic cell-type signal; positive ubiquity evidence absent"
    else:
        classification = "undetermined"
        review = True
        reason = "available evidence does not separate general from cell-type-specific"

    row = {
        "go_id": semantic["go_id"],
        "go_name": semantic["go_name"],
        "classification": classification,
        "confidence": confidence,
        "review_required": str(review).lower(),
        "constraint_scope": "self_and_descendants" if roots else "",
        "constraint_root_cl_ids": "|".join(roots),
        "constraint_root_labels": "|".join(
            catalog.nodes[root].catalog_label for root in roots if root in catalog.nodes
        ),
        "decision_reason": reason,
        "semantic_classification": semantic_class,
        "semantic_constraint_cl_id": semantic_root,
        "semantic_top_score": semantic.get("semantic_top_score", ""),
        "semantic_margin": semantic.get("semantic_margin", ""),
        "explicit_cl_ids": semantic.get("explicit_cl_ids", ""),
        "explicit_catalog_cl_ids": semantic.get("explicit_catalog_cl_ids", ""),
        "database_classification": db_class,
        "database_top_anchor_id": db_root,
        "database_top_anchor_share": f"{float(database.get('top_anchor_share', 0.0)):.6f}",
        "database_normalized_entropy": f"{float(database.get('entropy', 0.0)):.6f}",
        "database_experiment_count": int(database.get("experiment_count", 0)),
        "database_top_anchor_experiment_count": int(
            database.get("top_anchor_experiment_count", 0)
        ),
        "database_anchor_count": int(database.get("anchor_count", 0)),
        "database_significant_test_count": int(database.get("significant_test_count", 0)),
    }
    return row, roots


def run(args: argparse.Namespace) -> None:
    config = load_json(args.config) if args.config else {}
    fusion_config = config.get("fusion", {})
    catalog = CellCatalog.from_tsv(args.catalog)
    semantic_rows = read_tsv(args.semantic_summary)
    database_rows: list[dict[str, str]] = []
    for path in args.database_evidence or []:
        database_rows.extend(read_tsv(path))
    database_summary = aggregate_database(database_rows, fusion_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_fields = [
        "go_id",
        "mapping_status",
        "constraint_root_cl_id",
        "allowed_cl_id",
        "allowed_cl_label",
        "relation_to_root",
    ]
    context_fields = [
        "go_id",
        "mapping_status",
        "constraint_root_cl_id",
        "ancestor_cl_id",
        "ancestor_cl_label",
        "distance",
    ]
    closure_path = output_dir / "go_cl_constraint_closure.tsv"
    context_path = output_dir / "go_cl_constraint_context.tsv"
    closure_temporary = closure_path.with_suffix(".tsv.part")
    context_temporary = context_path.with_suffix(".tsv.part")
    classification_rows: list[dict] = []
    closure_count = 0
    context_count = 0
    with closure_temporary.open(
        "w", encoding="utf-8", newline=""
    ) as closure_handle, context_temporary.open(
        "w", encoding="utf-8", newline=""
    ) as context_handle:
        closure_writer = csv.DictWriter(
            closure_handle, fieldnames=closure_fields, delimiter="\t"
        )
        context_writer = csv.DictWriter(
            context_handle, fieldnames=context_fields, delimiter="\t"
        )
        closure_writer.writeheader()
        context_writer.writeheader()
        for semantic in semantic_rows:
            go_id = semantic["go_id"]
            row, roots = fuse_row(semantic, database_summary.get(go_id, {}), catalog)
            classification_rows.append(row)
            if row["classification"] not in {
                "cell_type_specific",
                "cell_type_specific_candidate",
            }:
                continue
            emitted_allowed: set[str] = set()
            for root in roots:
                if root not in catalog.nodes:
                    continue
                for allowed in catalog.descendants(root, include_self=True):
                    if allowed in emitted_allowed:
                        continue
                    emitted_allowed.add(allowed)
                    closure_writer.writerow(
                        {
                            "go_id": go_id,
                            "mapping_status": row["classification"],
                            "constraint_root_cl_id": root,
                            "allowed_cl_id": allowed,
                            "allowed_cl_label": catalog.nodes[allowed].catalog_label,
                            "relation_to_root": (
                                "self" if allowed == root else "descendant"
                            ),
                        }
                    )
                    closure_count += 1
                for distance, ancestor in enumerate(
                    catalog.ancestors(root), start=1
                ):
                    context_writer.writerow(
                        {
                            "go_id": go_id,
                            "mapping_status": row["classification"],
                            "constraint_root_cl_id": root,
                            "ancestor_cl_id": ancestor,
                            "ancestor_cl_label": catalog.nodes[
                                ancestor
                            ].catalog_label,
                            "distance": distance,
                        }
                    )
                    context_count += 1
    closure_temporary.replace(closure_path)
    context_temporary.replace(context_path)
    write_tsv(
        output_dir / "go_classification.tsv",
        list(classification_rows[0]),
        classification_rows,
    )
    manifest = {
        "go_terms": len(classification_rows),
        "classification_counts": dict(
            sorted(
                {
                    label: sum(row["classification"] == label for row in classification_rows)
                    for label in {row["classification"] for row in classification_rows}
                }.items()
            )
        ),
        "database_evidence_files": args.database_evidence or [],
        "constraint_closure_rows": closure_count,
        "constraint_context_rows": context_count,
    }
    (output_dir / "classification_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse GO curator references, SapBERT, and SCEA evidence into CL constraints."
    )
    parser.add_argument("--semantic-summary", required=True)
    parser.add_argument("--database-evidence", nargs="*")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
