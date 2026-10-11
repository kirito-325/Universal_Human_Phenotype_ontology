from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellphenorank_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cellphenorank_cache")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "A0_surface": "#4C566A",
    "A1_marker": "#2A9D8F",
    "A2_function": "#E9C46A",
    "A3_marker_function": "#C84C3A",
}


def _save(fig: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_benchmark_figures(
    output_dir: str | Path,
    disease_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    rank_changes: Sequence[Mapping[str, Any]],
    metrics_by_arm: Mapping[str, Mapping[str, float]],
    coverage_rows: Sequence[Mapping[str, Any]],
    null_iterations: Sequence[Mapping[str, Any]],
) -> list[str]:
    target = Path(output_dir) / "figures"
    created: list[str] = []
    baseline = {row["disease_id"]: float(row["best_known_gene_rank"]) for row in disease_by_arm["A0_surface"]}
    primary = {row["disease_id"]: float(row["best_known_gene_rank"]) for row in disease_by_arm["A3_marker_function"]}
    common = sorted(set(baseline).intersection(primary))
    if common:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        x = np.asarray([baseline[item] for item in common])
        y = np.asarray([primary[item] for item in common])
        limit = max(float(x.max()), float(y.max()), 2.0)
        ax.scatter(x, y, s=28, color="#C84C3A", alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.plot([1, limit], [1, limit], color="#626A70", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Known-gene rank: A0 surface")
        ax.set_ylabel("Known-gene rank: A3 reranked")
        ax.set_title(f"Known-gene ranks (n={len(common)})")
        path = target / "known_gene_rank_scatter.png"
        _save(fig, path)
        created.append(str(path))

    primary_changes = [row for row in rank_changes if row["experiment"] == "A3_marker_function"]
    if primary_changes:
        ordered = sorted(primary_changes, key=lambda row: float(row["rank_change"]))
        values = np.asarray([float(row["rank_change"]) for row in ordered])
        colors = np.where(values > 0, "#2A9D8F", np.where(values < 0, "#C84C3A", "#8A949B"))
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        ax.bar(np.arange(len(values)), values, color=colors, width=0.86)
        ax.axhline(0, color="#202124", linewidth=0.8)
        ax.set_xlabel("Disease ordered by rank change")
        ax.set_ylabel("Base rank - final rank")
        ax.set_title("A3 known-gene rank changes")
        path = target / "rank_change_waterfall.png"
        _save(fig, path)
        created.append(str(path))

    selected_metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    arms = [name for name in COLORS if name in metrics_by_arm]
    if arms:
        x = np.arange(len(selected_metrics))
        width = 0.18
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for index, arm in enumerate(arms):
            ax.bar(
                x + (index - (len(arms) - 1) / 2) * width,
                [float(metrics_by_arm[arm][metric]) for metric in selected_metrics],
                width=width,
                label=arm,
                color=COLORS[arm],
            )
        ax.set_xticks(x, [item.replace("_", " ") for item in selected_metrics])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Metric value")
        ax.set_title("Ablation comparison")
        ax.legend(frameon=False, fontsize=8)
        path = target / "ablation_metrics.png"
        _save(fig, path)
        created.append(str(path))

    coverage = {row["disease_id"]: float(row["cpo_coverage"]) for row in coverage_rows}
    if primary_changes:
        x = [coverage.get(row["disease_id"], 0.0) for row in primary_changes]
        y = [float(row["rank_change"]) for row in primary_changes]
        colors = ["#2A9D8F" if item > 0 else "#C84C3A" if item < 0 else "#8A949B" for item in y]
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.scatter(x, y, c=colors, s=30, alpha=0.8)
        ax.axhline(0, color="#626A70", linewidth=0.8)
        ax.set_xlabel("Disease HPO-CPO coverage")
        ax.set_ylabel("Known-gene rank change")
        ax.set_title("Coverage and reranking effect")
        path = target / "coverage_vs_rank_change.png"
        _save(fig, path)
        created.append(str(path))

    null_mrr = [float(row["value"]) for row in null_iterations if row["metric"] == "mrr"]
    if null_mrr:
        observed = float(metrics_by_arm["A3_marker_function"]["mrr"])
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.hist(null_mrr, bins=min(20, max(5, len(null_mrr) // 2)), color="#577590", alpha=0.85)
        ax.axvline(observed, color="#C84C3A", linewidth=2, label=f"Observed {observed:.3f}")
        ax.set_xlabel("MRR")
        ax.set_ylabel("Randomizations")
        ax.set_title("Observed vs size-preserving gene-set null")
        ax.legend(frameon=False)
        path = target / "observed_vs_null_mrr.png"
        _save(fig, path)
        created.append(str(path))
    return created


