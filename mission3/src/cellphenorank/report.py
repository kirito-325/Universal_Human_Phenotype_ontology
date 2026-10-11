from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping, Sequence


def _metric_table(metrics: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, dict[str, float]] = {}
    for row in metrics:
        grouped.setdefault(str(row["experiment"]), {})[str(row["metric"])] = float(row["value"])
    columns = ["recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "mrr", "ndcg_at_10", "median_rank"]
    lines = ["| Experiment | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for experiment, values in grouped.items():
        rendered = [f"{values.get(column, 0.0):.4f}" for column in columns]
        lines.append(f"| {experiment} | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    benchmark_metrics: Sequence[Mapping[str, Any]],
    statistical_tests: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(output_dir) / "report"
    target.mkdir(parents=True, exist_ok=True)
    metric_table = _metric_table(benchmark_metrics)
    counts = summary.get("counts", {})
    freeze = summary.get("freeze", {})
    lines = [
        "# CellPhenoRank run report",
        "",
        "## Run scope",
        "",
        f"- Diseases: {counts.get('diseases', 0)}",
        f"- Candidate mouse models: {counts.get('models', 0)}",
        f"- Candidate human genes: {counts.get('genes', 0)}",
        f"- Loaded HPO-MPO pairs: {counts.get('similarity_pairs', 0)}",
        f"- HPO terms with CPO mappings: {counts.get('hpo_terms_with_cpo', 0)}",
        f"- Frozen benchmark accepted: {bool(freeze.get('accepted', False))}",
        "",
        "## Benchmark metrics",
        "",
        metric_table,
        "",
        "## Paired rank changes",
        "",
    ]
    if statistical_tests:
        lines.extend(
            [
                "| Experiment | Mean change | 95% CI | Wins | Ties | Losses | Wilcoxon p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in statistical_tests:
            lines.append(
                "| {experiment} | {mean_rank_change:.3f} | [{mean_rank_change_ci_low:.3f}, {mean_rank_change_ci_high:.3f}] | "
                "{wins} | {ties} | {losses} | {wilcoxon_p_value:.4g} |".format(**row)
            )
    else:
        lines.append("No paired comparison was available.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The surface matrix, cellular reranker, and single-cell validation are separate evidence layers. "
            "A single-cell result tests predicted cell/function abnormalities; it does not establish gene causality.",
            "",
            "Missing CPO, marker, or function evidence is neutral. Candidate genes are never removed because a gene set lacks them.",
        ]
    )
    warnings = list(summary.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    markdown = "\n".join(lines) + "\n"
    (target / "report.md").write_text(markdown, encoding="utf-8")
    page = "<!doctype html><html><head><meta charset='utf-8'><title>CellPhenoRank report</title>" \
        "<style>body{font:15px/1.55 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#202124}" \
        "table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccd1d5;padding:7px;text-align:right}" \
        "th:first-child,td:first-child{text-align:left}h1,h2{letter-spacing:0}</style></head><body>" \
        f"<pre style='white-space:pre-wrap;font:inherit'>{html.escape(markdown)}</pre></body></html>"
    (target / "report.html").write_text(page, encoding="utf-8")
