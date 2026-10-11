#!/usr/bin/env python3
"""Prototype identity marker discovery on the hematopoietic catalog subtree.

For each parent in the expanded catalog, its direct children are treated as
sibling classes. Each child is scored one-vs-rest against the other supported
children under that parent. The positive class is the child's whole catalog
subtree intersected with CELLxGENE-covered CL ids.

The expression source is CELLxGENE WMG. Tissue-specificity is intentionally
collapsed away first: each exact CL id gets one cross-tissue gene profile.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
EXPANDED_DIR = ROOT / "cell_ontology" / "expanded"
CATALOG_TSV = EXPANDED_DIR / "expanded_cell_catalog.tsv"
EXPR_PARQUET = PROJECT_ROOT / "CPO2" / "data" / "cellxgene" / "expression_stats.parquet"
OUT_DIR = EXPANDED_DIR / "identity_marker_test" / "hematopoietic_cell"

ROOT_ID = "CL:0000988"
ROOT_LABEL = "hematopoietic cell"

ACTIVE_EFF = 0.5
ACTIVE_PCT = 0.05
MIN_POS_EFF = 1.0
MIN_POS_PCT = 0.08
MIN_FOLD_VS_NEG_MAX = 3.0
MIN_DELTA_PCT = 0.05
TOP_N_PER_CHILD = 50
PANEL_N_PER_CHILD = 10
EPS = 1e-6

GENE_SYMBOL_DENY_RE = re.compile(
    r"^(?:MT-|MTRNR|RP[SL][0-9]|MRP[SL][0-9]|RP[0-9]+-|RP11-|CT[ABCD]-|AC[0-9]|AL[0-9]|"
    r"LINC[0-9]|MIR[0-9]|SNORD|SNORA|RNU|RN7|ENSG)"
)
GENE_SYMBOL_DENY_EXACT = {
    "ACTB",
    "ACTG1",
    "B2M",
    "EEF1A1",
    "EEF1G",
    "EEF2",
    "GAPDH",
    "HSP90AA1",
    "HSP90AB1",
    "MALAT1",
    "NEAT1",
    "PFN1",
    "PPIA",
    "PPIB",
    "TMSB10",
    "TMSB4X",
    "TPT1",
}


def canonical_gene_symbol(value: object) -> str:
    symbol = str(value or "").strip()
    if not symbol or symbol.lower() == "nan":
        return ""
    if "_ENSG" in symbol:
        symbol = symbol.split("_ENSG", 1)[0]
    return symbol.strip()


def is_marker_like_gene(symbol: str) -> bool:
    if not symbol:
        return False
    upper = symbol.upper()
    if upper in GENE_SYMBOL_DENY_EXACT:
        return False
    if GENE_SYMBOL_DENY_RE.search(upper):
        return False
    if upper.endswith("P") and any(char.isdigit() for char in upper):
        return False
    return True


def read_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_TSV, sep="\t", dtype=str).fillna("")
    df["order_num"] = df["order"].astype(int)
    df["level_num"] = df["level"].astype(int)
    return df


def children_map(catalog: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for row in catalog.sort_values("order_num").itertuples(index=False):
        children[row.parent_id].append(row.cl_id)
    return children


def descendants(node_id: str, children: dict[str, list[str]]) -> set[str]:
    out = {node_id}
    stack = list(children.get(node_id, []))
    while stack:
        current = stack.pop()
        if current in out:
            continue
        out.add(current)
        stack.extend(children.get(current, []))
    return out


def label_map(catalog: pd.DataFrame) -> dict[str, str]:
    return dict(zip(catalog["cl_id"], catalog["catalog_label"]))


def load_immune_expression(subtree_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "tissue",
        "cell_type",
        "cell_type_ontology_term_id",
        "gene_symbol",
        "mean_expression",
        "pct_expressing",
    ]
    print(f"Loading CELLxGENE expression for {len(subtree_ids)} hematopoietic catalog ids...")
    expr = pd.read_parquet(EXPR_PARQUET, columns=columns)
    expr = expr[expr["cell_type_ontology_term_id"].isin(subtree_ids)].copy()
    expr = expr.rename(columns={"cell_type_ontology_term_id": "cl_id"})
    expr["gene_symbol"] = expr["gene_symbol"].map(canonical_gene_symbol)
    expr = expr[expr["gene_symbol"] != ""].copy()
    expr["mean_expression"] = expr["mean_expression"].astype("float32")
    expr["pct_expressing"] = expr["pct_expressing"].astype("float32")
    expr["eff"] = (
        expr["pct_expressing"].astype("float64")
        * np.expm1(np.clip(expr["mean_expression"].astype("float64"), 0.0, None))
    ).astype("float32")
    expr["active"] = (expr["eff"] >= ACTIVE_EFF) & (expr["pct_expressing"] >= ACTIVE_PCT)

    category = expr[["cl_id", "cell_type", "tissue"]].drop_duplicates()
    tissue_counts = category.groupby("cl_id")["tissue"].nunique().rename("n_tissues")

    grouped = expr.groupby(["cl_id", "gene_symbol"], sort=False).agg(
        sum_eff=("eff", "sum"),
        sum_pct=("pct_expressing", "sum"),
        max_eff=("eff", "max"),
        max_pct=("pct_expressing", "max"),
        active_tissues=("active", "sum"),
    ).reset_index()
    grouped = grouped.merge(tissue_counts.reset_index(), on="cl_id", how="left")
    grouped["eff_mean"] = grouped["sum_eff"] / grouped["n_tissues"]
    grouped["pct_mean"] = grouped["sum_pct"] / grouped["n_tissues"]
    grouped["tissue_breadth"] = grouped["active_tissues"] / grouped["n_tissues"]
    grouped["cl_active"] = (
        (grouped["eff_mean"] >= ACTIVE_EFF)
        & (grouped["pct_mean"] >= ACTIVE_PCT)
    )
    grouped["is_marker_like_gene"] = grouped["gene_symbol"].map(is_marker_like_gene)
    cl_summary = category.groupby(["cl_id", "cell_type"], as_index=False).agg(
        n_tissues=("tissue", "nunique"),
        tissues=("tissue", lambda values: ";".join(sorted(set(map(str, values)))[:20])),
    )
    print(
        f"  expression rows: {len(expr):,} | covered CL ids: {expr['cl_id'].nunique()} | "
        f"CL-gene rows: {len(grouped):,}"
    )
    return grouped, cl_summary


def aggregate_side(cl_gene: pd.DataFrame, cl_ids: set[str], prefix: str) -> pd.DataFrame:
    side = cl_gene[cl_gene["cl_id"].isin(cl_ids)].copy()
    n_cl = len(cl_ids)
    if side.empty:
        return pd.DataFrame(columns=["gene_symbol"])
    side["active_int"] = side["cl_active"].astype(int)
    agg = side.groupby("gene_symbol", sort=False).agg(
        eff_sum=("eff_mean", "sum"),
        pct_sum=("pct_mean", "sum"),
        eff_max=("eff_mean", "max"),
        pct_max=("pct_mean", "max"),
        eff_mean_observed=("eff_mean", "mean"),
        pct_mean_observed=("pct_mean", "mean"),
        cl_observed=("cl_id", "nunique"),
        cl_active_count=("active_int", "sum"),
        tissue_breadth_sum=("tissue_breadth", "sum"),
    ).reset_index()
    agg[f"{prefix}_eff_mean"] = agg["eff_sum"] / n_cl
    agg[f"{prefix}_pct_mean"] = agg["pct_sum"] / n_cl
    agg[f"{prefix}_eff_max"] = agg["eff_max"]
    agg[f"{prefix}_pct_max"] = agg["pct_max"]
    agg[f"{prefix}_cl_observed_fraction"] = agg["cl_observed"] / n_cl
    agg[f"{prefix}_cl_breadth"] = agg["cl_active_count"] / n_cl
    agg[f"{prefix}_tissue_breadth_mean"] = agg["tissue_breadth_sum"] / n_cl
    keep = [
        "gene_symbol",
        f"{prefix}_eff_mean",
        f"{prefix}_pct_mean",
        f"{prefix}_eff_max",
        f"{prefix}_pct_max",
        f"{prefix}_cl_observed_fraction",
        f"{prefix}_cl_breadth",
        f"{prefix}_tissue_breadth_mean",
    ]
    return agg[keep]


def min_breadth(n_pos: int) -> float:
    if n_pos <= 1:
        return 1.0
    if n_pos == 2:
        return 0.5
    return 0.25


def score_child(
    cl_gene: pd.DataFrame,
    parent_id: str,
    parent_label: str,
    child_id: str,
    child_label: str,
    sibling_ids: list[str],
    pos_ids: set[str],
    neg_ids: set[str],
) -> pd.DataFrame:
    pos = aggregate_side(cl_gene, pos_ids, "pos")
    neg = aggregate_side(cl_gene, neg_ids, "neg")
    if pos.empty or neg.empty:
        return pd.DataFrame()
    scores = pos.merge(neg, on="gene_symbol", how="left").fillna(0)
    scores["fold_vs_neg_max"] = (scores["pos_eff_mean"] + EPS) / (scores["neg_eff_max"] + EPS)
    scores["fold_vs_neg_mean"] = (scores["pos_eff_mean"] + EPS) / (scores["neg_eff_mean"] + EPS)
    scores["delta_pct_vs_neg_max"] = scores["pos_pct_mean"] - scores["neg_pct_max"]
    scores["delta_pct_vs_neg_mean"] = scores["pos_pct_mean"] - scores["neg_pct_mean"]
    scores["marker_score"] = (
        np.log2(scores["fold_vs_neg_max"].clip(lower=1.0))
        * scores["pos_eff_mean"]
        * scores["delta_pct_vs_neg_max"].clip(lower=0)
        * np.sqrt(scores["pos_cl_breadth"].clip(lower=0))
    )
    needed_breadth = min_breadth(len(pos_ids))
    scores["passes_rule"] = (
        (scores["pos_eff_mean"] >= MIN_POS_EFF)
        & (scores["pos_pct_mean"] >= MIN_POS_PCT)
        & (scores["pos_cl_breadth"] >= needed_breadth)
        & (scores["fold_vs_neg_max"] >= MIN_FOLD_VS_NEG_MAX)
        & (scores["delta_pct_vs_neg_max"] >= MIN_DELTA_PCT)
    )
    scores["is_marker_like_gene"] = scores["gene_symbol"].map(is_marker_like_gene)
    scores["passes_marker_rule"] = scores["passes_rule"] & scores["is_marker_like_gene"]
    scores["parent_id"] = parent_id
    scores["parent_label"] = parent_label
    scores["child_id"] = child_id
    scores["child_label"] = child_label
    scores["sibling_child_ids"] = ";".join(sibling_ids)
    scores["n_pos_cl"] = len(pos_ids)
    scores["n_neg_cl"] = len(neg_ids)
    scores["min_pos_cl_breadth"] = needed_breadth
    scores = scores.sort_values(
        [
            "passes_marker_rule",
            "passes_rule",
            "marker_score",
            "fold_vs_neg_max",
            "pos_eff_mean",
        ],
        ascending=[False, False, False, False, False],
    ).head(TOP_N_PER_CHILD)
    scores.insert(0, "rank", range(1, len(scores) + 1))
    return scores


def write_tsv(path: Path, rows: list[dict[str, object]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, sep="\t", index=False, float_format="%.6g")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    labels = label_map(catalog)
    children = children_map(catalog)
    subtree_ids = descendants(ROOT_ID, children)
    subtree = catalog[catalog["cl_id"].isin(subtree_ids)].copy()
    subtree = subtree.sort_values("order_num")

    cl_gene, cl_summary = load_immune_expression(subtree_ids)
    observed_ids = set(cl_summary["cl_id"])
    desc_cache = {node_id: descendants(node_id, children) for node_id in subtree["cl_id"]}
    observed_desc = {
        node_id: desc_cache[node_id] & observed_ids
        for node_id in subtree["cl_id"]
    }

    coverage_rows = []
    for row in subtree.itertuples(index=False):
        obs = sorted(observed_desc[row.cl_id])
        coverage_rows.append(
            {
                "cl_id": row.cl_id,
                "catalog_label": row.catalog_label,
                "parent_id": row.parent_id,
                "parent_label": row.parent_label,
                "level": row.level,
                "rank": row.rank,
                "source": row.source,
                "subtree_nodes": len(desc_cache[row.cl_id]),
                "observed_cl_count": len(obs),
                "observed_cl_ids": ";".join(obs),
                "observed_cl_labels": ";".join(labels.get(item, item) for item in obs[:40]),
            }
        )
    write_tsv(OUT_DIR / "hematopoietic_subtree_expression_coverage.tsv", coverage_rows)

    score_tables = []
    summary_rows = []
    for parent_id in subtree["cl_id"]:
        child_ids = [child for child in children.get(parent_id, []) if child in subtree_ids]
        if len(child_ids) < 2:
            continue
        supported = [child for child in child_ids if observed_desc.get(child)]
        if len(supported) < 2:
            continue
        parent_label = labels.get(parent_id, parent_id)
        for child_id in supported:
            sibling_ids = [child for child in supported if child != child_id]
            pos_ids = observed_desc[child_id]
            neg_ids = set().union(*(observed_desc[sib] for sib in sibling_ids))
            if not pos_ids or not neg_ids:
                continue
            scores = score_child(
                cl_gene,
                parent_id,
                parent_label,
                child_id,
                labels.get(child_id, child_id),
                sibling_ids,
                pos_ids,
                neg_ids,
            )
            if scores.empty:
                continue
            passed = scores[scores["passes_marker_rule"]]
            score_tables.append(scores)
            summary_rows.append(
                {
                    "parent_id": parent_id,
                    "parent_label": parent_label,
                    "child_id": child_id,
                    "child_label": labels.get(child_id, child_id),
                    "sibling_child_ids": ";".join(sibling_ids),
                    "sibling_child_labels": ";".join(labels.get(item, item) for item in sibling_ids),
                    "n_pos_cl": len(pos_ids),
                    "positive_cl_ids": ";".join(sorted(pos_ids)),
                    "positive_cl_labels": ";".join(labels.get(item, item) for item in sorted(pos_ids)[:60]),
                    "n_neg_cl": len(neg_ids),
                    "top_pass_genes": ";".join(passed["gene_symbol"].head(15).tolist()),
                    "n_pass_genes_in_top": int(passed.shape[0]),
                    "top_any_genes": ";".join(scores["gene_symbol"].head(15).tolist()),
                }
            )

    all_scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    panel_rows = []
    if not all_scores.empty:
        front = [
            "parent_id",
            "parent_label",
            "child_id",
            "child_label",
            "rank",
            "gene_symbol",
            "passes_rule",
            "is_marker_like_gene",
            "passes_marker_rule",
            "marker_score",
            "n_pos_cl",
            "n_neg_cl",
            "pos_eff_mean",
            "pos_pct_mean",
            "pos_cl_breadth",
            "pos_tissue_breadth_mean",
            "neg_eff_max",
            "neg_pct_max",
            "neg_eff_mean",
            "neg_pct_mean",
            "neg_cl_breadth",
            "fold_vs_neg_max",
            "delta_pct_vs_neg_max",
            "fold_vs_neg_mean",
            "delta_pct_vs_neg_mean",
            "sibling_child_ids",
            "min_pos_cl_breadth",
        ]
        all_scores = all_scores[front]
        write_tsv(OUT_DIR / "hematopoietic_sibling_identity_marker_scores.tsv", all_scores)
        write_tsv(
            OUT_DIR / "hematopoietic_sibling_identity_marker_candidates.tsv",
            all_scores[all_scores["passes_marker_rule"]].copy(),
        )
        for row in summary_rows:
            child_scores = all_scores[
                (all_scores["parent_id"] == row["parent_id"])
                & (all_scores["child_id"] == row["child_id"])
                & (all_scores["passes_marker_rule"])
            ].copy()
            panel = child_scores.head(PANEL_N_PER_CHILD)
            warnings = []
            if panel.empty:
                warnings.append("no_marker_candidate")
            if int(row["n_pos_cl"]) == 1:
                warnings.append("single_positive_cl")
            elif int(row["n_pos_cl"]) < 3:
                warnings.append("limited_positive_cl")
            panel_rows.append(
                {
                    "parent_id": row["parent_id"],
                    "parent_label": row["parent_label"],
                    "child_id": row["child_id"],
                    "child_label": row["child_label"],
                    "sibling_child_labels": row["sibling_child_labels"],
                    "n_pos_cl": row["n_pos_cl"],
                    "n_neg_cl": row["n_neg_cl"],
                    "n_panel_genes": int(panel.shape[0]),
                    "panel_genes": ";".join(panel["gene_symbol"].tolist()),
                    "panel_scores": ";".join(panel["marker_score"].map(lambda value: f"{float(value):.3g}").tolist()),
                    "panel_pos_pct": ";".join(panel["pos_pct_mean"].map(lambda value: f"{float(value):.3g}").tolist()),
                    "panel_fold_vs_neg_max": ";".join(panel["fold_vs_neg_max"].map(lambda value: f"{float(value):.3g}").tolist()),
                    "positive_cl_labels": row["positive_cl_labels"],
                    "warning": ";".join(warnings),
                }
            )
    write_tsv(OUT_DIR / "hematopoietic_sibling_comparison_summary.tsv", summary_rows)
    write_tsv(OUT_DIR / "hematopoietic_sibling_identity_marker_panel.tsv", panel_rows)

    report = {
        "root_id": ROOT_ID,
        "root_label": ROOT_LABEL,
        "subtree_nodes": len(subtree_ids),
        "observed_exact_cl_ids": len(observed_ids),
        "cl_gene_rows": len(cl_gene),
        "comparisons_scored": len(summary_rows),
        "score_rows_top_n": int(all_scores.shape[0]) if not all_scores.empty else 0,
        "candidate_rows": int(all_scores["passes_marker_rule"].sum()) if not all_scores.empty else 0,
        "thresholds": {
            "active_eff": ACTIVE_EFF,
            "active_pct": ACTIVE_PCT,
            "min_pos_eff": MIN_POS_EFF,
            "min_pos_pct": MIN_POS_PCT,
            "min_fold_vs_neg_max": MIN_FOLD_VS_NEG_MAX,
            "min_delta_pct": MIN_DELTA_PCT,
            "top_n_per_child": TOP_N_PER_CHILD,
            "panel_n_per_child": PANEL_N_PER_CHILD,
            "candidate_rule": "passes_rule AND marker-like gene symbol filter",
        },
        "outputs": {
            "coverage": str(OUT_DIR / "hematopoietic_subtree_expression_coverage.tsv"),
            "summary": str(OUT_DIR / "hematopoietic_sibling_comparison_summary.tsv"),
            "scores": str(OUT_DIR / "hematopoietic_sibling_identity_marker_scores.tsv"),
            "candidates": str(OUT_DIR / "hematopoietic_sibling_identity_marker_candidates.tsv"),
            "panel": str(OUT_DIR / "hematopoietic_sibling_identity_marker_panel.tsv"),
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
