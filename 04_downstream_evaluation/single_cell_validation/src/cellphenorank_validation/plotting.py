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



def generate_single_cell_figures(
    output_dir: str | Path,
    specificity_rows: Sequence[Mapping[str, Any]],
    composition_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    target = Path(output_dir) / "figures"
    created: list[str] = []
    labels = sorted({str(row["tested_dataset_label"]) for row in specificity_rows})
    predictions = sorted({int(row["prediction_rank"]) for row in specificity_rows})
    if labels and predictions:
        matrix = np.full((len(predictions), len(labels)), np.nan)
        for row_index, prediction in enumerate(predictions):
            for column_index, label in enumerate(labels):
                values = [
                    float(row["effect"])
                    for row in specificity_rows
                    if int(row["prediction_rank"]) == prediction
                    and str(row["tested_dataset_label"]) == label
                ]
                if values:
                    matrix[row_index, column_index] = float(np.mean(values))
        limit = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 0.01)
        fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.8), max(3.5, len(predictions) * 0.55)))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(predictions)), [f"Prediction {item}" for item in predictions])
        ax.set_title("Predicted function effects across cell types")
        fig.colorbar(image, ax=ax, label="Disease - control module effect")
        path = target / "predicted_function_cell_heatmap.png"
        _save(fig, path)
        created.append(str(path))
    if summary_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ranks = [int(row["prediction_rank"]) for row in summary_rows]
        effects = [float(row["meta_effect"]) for row in summary_rows]
        colors = ["#C84C3A" if effect < 0 else "#2A9D8F" for effect in effects]
        ax.bar(ranks, effects, color=colors)
        ax.axhline(0, color="#626A70", linewidth=0.8)
        ax.set_xlabel("Prediction rank")
        ax.set_ylabel("Meta module effect")
        ax.set_title("Predicted-cell function effects")
        path = target / "predicted_function_meta_effects.png"
        _save(fig, path)
        created.append(str(path))
    if composition_rows:
        unique = {}
        for row in composition_rows:
            unique[(str(row["dataset_id"]), str(row["mapped_dataset_label"]))] = float(row["fraction_effect"])
        keys = sorted(unique)
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        values = [unique[key] for key in keys]
        colors = ["#C84C3A" if value < 0 else "#2A9D8F" for value in values]
        ax.bar(np.arange(len(keys)), values, color=colors)
        ax.set_xticks(np.arange(len(keys)), [f"{study}\n{label}" for study, label in keys])
        ax.axhline(0, color="#626A70", linewidth=0.8)
        ax.set_ylabel("Disease - control cell fraction")
        ax.set_title("Predicted-cell composition effects")
        path = target / "predicted_cell_composition.png"
        _save(fig, path)
        created.append(str(path))
    return created

