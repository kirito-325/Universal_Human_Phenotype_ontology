from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MPL_CACHE = Path("/tmp/cpo_mapping_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import config_digest, output_root
from .io import file_record, read_tsv, software_versions, write_json, write_text
from .matrix import CHANNELS


def _figures(root: Path, coverage: pd.DataFrame) -> list[str]:
    figures_root = root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(coverage))
    axis.bar(x - 0.18, coverage["hpo_complete_profile_terms"], width=0.36, label="HPO")
    axis.bar(x + 0.18, coverage["mpo_complete_profile_terms"], width=0.36, label="MPO")
    axis.set_xticks(x, coverage["channel"])
    axis.set_ylabel("Terms with complete self CPO profile")
    axis.set_title("Complete profile coverage by evidence channel")
    axis.legend()
    fig.tight_layout()
    path = figures_root / "channel_profile_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        coverage["channel"],
        coverage["direction_comparable_fraction_of_shape_pairs"].astype(float),
        color="#287271",
    )
    axis.set_ylim(0, max(0.01, coverage["direction_comparable_fraction_of_shape_pairs"].astype(float).max() * 1.15))
    axis.set_ylabel("Fraction of scoreable term pairs")
    axis.set_title("Strict comparable-direction coverage")
    fig.tight_layout()
    path = figures_root / "direction_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))

    samples: list[np.ndarray] = []
    labels: list[str] = []
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for channel in CHANNELS:
            values = handle[channel]["shape"][:: max(1, handle[channel]["shape"].shape[0] // 500), :]
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive) > 50000:
                positive = positive[:: math.ceil(len(positive) / 50000)]
            samples.append(positive)
            labels.append(channel)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.hist(samples, bins=40, label=labels, histtype="step", linewidth=1.5)
    axis.set_xlabel("Shape score")
    axis.set_ylabel("Sampled positive pairs")
    axis.set_title("Positive structural score distributions")
    axis.legend()
    fig.tight_layout()
    path = figures_root / "shape_score_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path.relative_to(root)))
    return outputs


def run_report(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    coverage = pd.read_csv(root / "evaluation/coverage.tsv", sep="\t")
    forward = pd.read_parquet(root / "candidates/hpo_to_mpo_topk.parquet")
    _, hpo_rows = read_tsv(root / "indices/hpo_index.tsv")
    _, mpo_rows = read_tsv(root / "indices/mpo_index.tsv")
    hpo_labels = {row["term_id"]: row["label"] for row in hpo_rows}
    mpo_labels = {row["term_id"]: row["label"] for row in mpo_rows}
    examples = forward[
        (forward["view"] == "shape_topk") & (forward["rank"] == 1)
    ].sort_values(["channel", "shape"], ascending=[True, False]).groupby("channel").head(1)
    conflict_examples = forward[
        np.isfinite(forward["conflict"]) & (forward["conflict"] > 0)
    ].sort_values("conflict", ascending=False).head(1)
    branch = pd.read_parquet(root / "candidates/branch_support_topk.parquet")
    branch_examples = branch[
        ~np.isfinite(branch["self_shape"])
        | (branch["branch_score"] > branch["self_shape"])
    ].sort_values("branch_score", ascending=False).head(1)
    figure_paths = _figures(root, coverage)

    lines = [
        "# CPO-based HPO-MPO mapping v1",
        "",
        "## Result",
        "",
        "The run contains a fixed 12,121 HPO by 14,385 MPO index and 12 HDF5 datasets: four evidence channels (DD, DM, MD, MM) by shape, direction support, and direction conflict. Main profiles use complete self CPO structures only. Inherited MPO relations are used only for branch retrieval.",
        "",
        "## Coverage",
        "",
        "| Channel | HPO profiles | MPO profiles | Scoreable pairs | Positive shape | Comparable direction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.channel} | {int(row.hpo_complete_profile_terms):,} | {int(row.mpo_complete_profile_terms):,} | "
            f"{int(row.finite_shape_pairs):,} | {int(row.positive_shape_pairs):,} | {int(row.finite_direction_pairs):,} |"
        )
    lines.extend(["", "## Example top structural candidates", ""])
    for row in examples.itertuples(index=False):
        lines.append(
            f"- {row.channel}: `{row.query_id}` {hpo_labels.get(row.query_id, '')} -> "
            f"`{row.target_id}` {mpo_labels.get(row.target_id, '')}; shape={row.shape:.6f}, "
            f"direction={row.direction_status}."
        )
    lines.extend(["", "## Counterexamples", ""])
    for row in conflict_examples.itertuples(index=False):
        lines.append(
            f"- Direction conflict: `{row.query_id}` -> `{row.target_id}` ({row.channel}); "
            f"shape={row.shape:.6f}, support={row.support:.6f}, conflict={row.conflict:.6f}."
        )
    for row in branch_examples.itertuples(index=False):
        self_value = "NA" if not np.isfinite(row.self_shape) else f"{row.self_shape:.6f}"
        lines.append(
            f"- Branch-only/broader support: `{row.hpo_id}` -> `{row.mpo_ancestor_id}` "
            f"through leaf `{row.best_leaf_id}` ({row.channel}); branch={row.branch_score:.6f}, "
            f"ancestor self={self_value}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Scores represent similarity between collected cell/process descriptions. They are not mapping probabilities and do not assert ontology equivalence. Support and conflict use only exact CL+GO structures with a resolved, comparable direction axis. Partial CPO records appear only in CL/GO auxiliary retrieval.",
            "",
            "## External evaluation",
            "",
            "No independent reference mapping was configured. `evaluation/ranking_metrics.tsv` therefore records `NOT_AVAILABLE`; no unlabeled pairs were treated as negatives.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 -B -m cpo_mapping run-all --config config/cpo_mapping_v1.yaml --resume",
            "```",
        ]
    )
    write_text(root / "reports/final_report.md", "\n".join(lines) + "\n")
    write_text(
        root / "README.md",
        "# cpo_mapping_v1\n\n"
        "Main matrix: `matrices/hpo_mpo_cpo.h5`.\n\n"
        "Read row example:\n\n"
        "```bash\n"
        "python3 -B -m cpo_mapping query --config config/cpo_mapping_v1.yaml "
        "--side HPO --term-id HP:0000003 --channel MD --view shape_topk\n"
        "```\n\n"
        "See `reports/final_report.md` for coverage and interpretation.\n",
    )
    outputs = [root / "reports/final_report.md", root / "README.md"] + [root / value for value in figure_paths]
    result = {
        "status": "complete",
        "figures": figure_paths,
        "outputs": {str(path.relative_to(root)): file_record(path) for path in outputs},
    }
    write_json(root / "logs/report_summary.json", result)
    stage_logs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "logs").glob("*.json"))
        if path.name != "report_summary.json"
    }
    write_json(
        root / "run_manifest.json",
        {
            "schema_version": 1,
            "run_name": config["run_name"],
            "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_sha256": config_digest(config),
            "software_versions": software_versions(),
            "matrix": file_record(root / "matrices/hpo_mpo_cpo.h5"),
            "stages": stage_logs,
            "external_reference_status": "NOT_AVAILABLE",
        },
    )
    return result
