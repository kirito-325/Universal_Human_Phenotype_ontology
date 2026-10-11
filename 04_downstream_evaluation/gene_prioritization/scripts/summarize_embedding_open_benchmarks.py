from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellphenorank.io import input_record, write_json, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize comparable open-candidate embedding benchmark runs"
    )
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=MANIFEST")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs: list[tuple[str, Path, dict]] = []
    for value in args.run:
        if "=" not in value:
            raise ValueError("--run must use LABEL=MANIFEST")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        runs.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if len(runs) < 2:
        raise ValueError("At least two runs are required")

    reference = runs[0][2]
    fixed_keys = ("candidate_genes", "candidate_models", "diseases")
    for label, _, manifest in runs[1:]:
        for key in fixed_keys:
            if manifest["counts"][key] != reference["counts"][key]:
                raise ValueError(f"{label} has a different {key}")
        if (
            manifest["inputs"]["open_candidate_manifest"]["sha256"]
            != reference["inputs"]["open_candidate_manifest"]["sha256"]
        ):
            raise ValueError(f"{label} uses a different open candidate manifest")

    rows = []
    for label, _, manifest in runs:
        for arm, metrics in manifest["metrics"].items():
            rows.append(
                {
                    "variant": label,
                    "top_mpo_per_hpo": manifest["settings"]["top_mpo_per_hpo"],
                    "arm": arm,
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "recall_at_50": metrics["recall_at_50"],
                    "mrr": metrics["mrr"],
                    "ndcg_at_10": metrics["ndcg_at_10"],
                    "median_rank": metrics["median_rank"],
                    "gold_gene_retrieval_rate": metrics["gold_gene_retrieval_rate"],
                }
            )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output / "metrics.tsv", rows, list(rows[0]))
    summary = {
        "evaluation_mode": "open_candidate_benchmark",
        "candidate_genes": reference["counts"]["candidate_genes"],
        "candidate_models": reference["counts"]["candidate_models"],
        "diseases": reference["counts"]["diseases"],
        "open_candidate_manifest_sha256": reference["inputs"]["open_candidate_manifest"][
            "sha256"
        ],
        "runs": {label: input_record(path) for label, path, _ in runs},
    }
    write_json(output / "comparison_manifest.json", summary)

    selected_arms = {
        "E0_embedding_surface",
        "E3_strict_constrained",
        "E4_extended_constrained",
        "E5_bridge_loo",
    }
    lines = [
        "# Open-candidate SapBERT comparison",
        "",
        f"Fixed universe: {summary['candidate_genes']} genes, "
        f"{summary['candidate_models']} models, {summary['diseases']} diseases.",
        "",
        "| Variant | Arm | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | Median rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["arm"] not in selected_arms:
            continue
        lines.append(
            f"| {row['variant']} | {row['arm']} | {row['recall_at_1']:.3f} | "
            f"{row['recall_at_5']:.3f} | {row['recall_at_10']:.3f} | "
            f"{row['recall_at_50']:.3f} | {row['mrr']:.4f} | "
            f"{row['median_rank']:.1f} |"
        )
    lines.extend(
        [
            "",
            "These variants are prespecified sensitivity analyses. The table must not be "
            "used to select a winning configuration and then report that configuration as "
            "a confirmatory result on the same disease cohort.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
