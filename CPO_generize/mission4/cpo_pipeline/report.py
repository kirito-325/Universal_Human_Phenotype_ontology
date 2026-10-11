from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpo_mapping.io import file_record, read_tsv, write_json, write_text

from .config import config_digest, output_root, public_config


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _coverage_figure(root: Path) -> None:
    coverage = pd.read_csv(root / "profiles/coverage.tsv", sep="\t")
    grouped = (
        coverage.groupby(["ontology_side", "node_type"])[
            ["self_structure_count", "inherited_structure_count", "final_structure_count"]
        ]
        .apply(lambda frame: (frame > 0).sum())
        .reset_index()
    )
    labels = [f"{row.ontology_side}\n{row.node_type}" for row in grouped.itertuples()]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(12, 5))
    for offset, (column, label, color) in enumerate(
        (
            ("self_structure_count", "self", "#277da1"),
            ("inherited_structure_count", "inherited", "#f8961e"),
            ("final_structure_count", "final", "#43aa8b"),
        )
    ):
        plt.bar(x + (offset - 1) * width, grouped[column], width, label=label, color=color)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Terms with a non-empty profile")
    plt.title("CPO profile coverage by ontology side and node policy")
    plt.legend(frameon=False)
    _save_figure(root / "figures/profile_coverage.png")


def _matrix_figure(root: Path) -> None:
    rng = np.random.default_rng(20260905)
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        rows = np.sort(rng.choice(handle["final/shape"].shape[0], 256, replace=False))
        columns = np.sort(rng.choice(handle["final/shape"].shape[1], 512, replace=False))
        values = {
            "shape": np.asarray(handle["final/shape"][rows, :][:, columns]).ravel(),
            "F-R": (
                np.asarray(handle["final/hpo_to_mpo"][rows, :][:, columns])
                - np.asarray(handle["final/mpo_to_hpo"][rows, :][:, columns])
            ).ravel(),
            "support": np.asarray(handle["final/direction_support"][rows, :][:, columns]).ravel(),
            "conflict": np.asarray(handle["final/direction_conflict"][rows, :][:, columns]).ravel(),
        }
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (label, raw), color in zip(
        axes.flat,
        values.items(),
        ("#277da1", "#577590", "#43aa8b", "#f94144"),
    ):
        finite = raw[np.isfinite(raw)]
        axis.hist(finite, bins=40, color=color, alpha=0.9)
        axis.set_title(f"{label} (n={len(finite):,})")
        axis.set_xlabel("Score")
        axis.set_ylabel("Sample count")
    figure.suptitle("Term-matrix score distributions (fixed random sample)")
    _save_figure(root / "figures/term_score_distributions.png")


def _evaluation_figure(root: Path) -> None:
    metrics = pd.read_csv(root / "evaluation/ranking_metrics.tsv", sep="\t")
    term = metrics[
        (metrics["task"] == "term_mapping")
        & (metrics["relation_type"] == "exactMatch")
        & (metrics["tie_policy"] == "optimistic")
        & (metrics["metric"] == "Hit")
        & (metrics["direction"] == "HPO_to_MPO")
        & (metrics["evaluation_scope"] == "end_to_end")
    ]
    gene = metrics[
        (metrics["task"] == "disease_gene")
        & (metrics["split"] == "test")
        & (metrics["tie_policy"] == "optimistic")
        & (metrics["metric"] == "Hit")
        & (metrics["evaluation_scope"] == "end_to_end")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for view, frame in term.groupby("view"):
        axes[0].plot(frame["k"], frame["value"], marker="o", label=view)
    axes[0].set_title("HPO to MPO exact-match retrieval")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Hit@k")
    axes[0].legend(frameon=False, fontsize=8)
    for view, frame in gene.groupby("view"):
        axes[1].plot(frame["k"], frame["value"], marker="o", label=view)
    axes[1].set_title("Disease-gene retrospective retrieval (test)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Hit@k")
    axes[1].legend(frameon=False, fontsize=8)
    _save_figure(root / "figures/ranking_metrics.png")


def _disease_coverage_figure(root: Path) -> None:
    with h5py.File(root / "disease_model/scores/disease_model_scores.h5", "r") as handle:
        row_count, column_count = handle["score_evidence"].shape
        rows = np.linspace(0, row_count - 1, 128, dtype=int)
        columns = np.linspace(0, column_count - 1, 256, dtype=int)
        score = np.asarray(handle["score_evidence"][rows, :][:, columns]).ravel()
        coverage = (
            np.asarray(handle["q_H"][rows, :][:, columns])
            + np.asarray(handle["q_M"][rows, :][:, columns])
        ).ravel() / 2
    finite = np.isfinite(score) & np.isfinite(coverage)
    plt.figure(figsize=(7, 5))
    plt.hexbin(coverage[finite], score[finite], gridsize=45, mincnt=1, cmap="viridis")
    plt.colorbar(label="Sampled disease-model pairs")
    plt.xlabel("Mean computable coverage (q_H + q_M) / 2")
    plt.ylabel("Coverage-adjusted evidence score")
    plt.title("Disease-model score and resource coverage")
    _save_figure(root / "figures/disease_model_coverage.png")


def _orthology_figure(root: Path) -> None:
    frames = {
        "one-to-one": pd.read_parquet(
            root / "gene_prediction/human_1to1.parquet",
            columns=["disease_id", "tie_count", "orthology_ambiguous"],
        ),
        "expanded": pd.read_parquet(
            root / "gene_prediction/human_expanded.parquet",
            columns=["disease_id", "tie_count", "orthology_ambiguous"],
        ),
    }
    labels = list(frames)
    disease_counts = [frame["disease_id"].nunique() for frame in frames.values()]
    ambiguous = [
        int((frame["orthology_ambiguous"].astype(str) == "true").sum())
        for frame in frames.values()
    ]
    median_ties = [float(frame["tie_count"].median()) for frame in frames.values()]
    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].bar(labels, disease_counts, color=["#277da1", "#43aa8b"])
    axes[0].set_title("Diseases with candidates")
    axes[0].set_ylabel("Disease count")
    axes[1].bar(labels, ambiguous, color=["#277da1", "#f8961e"])
    axes[1].set_title("Ambiguous candidate rows")
    axes[1].set_ylabel("Candidate rows")
    axes[2].bar(labels, median_ties, color=["#277da1", "#577590"])
    axes[2].set_title("Median score-tie size")
    axes[2].set_ylabel("Genes")
    figure.suptitle("One-to-one and expanded orthology ranking coverage")
    _save_figure(root / "figures/orthology_coverage_ties.png")


def _metric_rows(root: Path) -> list[dict[str, object]]:
    metrics = pd.read_csv(root / "evaluation/ranking_metrics.tsv", sep="\t")
    selected = metrics[
        (
            (metrics["task"] == "term_mapping")
            & (metrics["view"] == "final")
            & (metrics["relation_type"] == "exactMatch")
            & (metrics["direction"] == "HPO_to_MPO")
            & (metrics["tie_policy"] == "optimistic")
            & (((metrics["metric"] == "MRR")) | ((metrics["metric"] == "Hit") & (metrics["k"] == 20)))
        )
        |
        (
            (metrics["task"] == "disease_gene")
            & (metrics["split"] == "test")
            & (metrics["tie_policy"] == "optimistic")
            & (((metrics["metric"] == "MRR")) | ((metrics["metric"] == "Hit") & (metrics["k"] == 20)))
        )
    ]
    return selected.to_dict("records")


def _term_examples(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "mapping/hpo_to_mpo.parquet")
    frame = frame[(frame["view"] == "topk_all") & (frame["rank"] == 1)].copy()
    categories = [
        ("highest_shape", frame.sort_values(["score_shape", "hpo_id"], ascending=[False, True])),
        ("inherited_support", frame[frame["source_origin_summary"].str.contains("inherited")].sort_values(["score_shape", "hpo_id"], ascending=[False, True])),
        ("missing_required_self", frame[frame["missing_required_self_flags"].str.contains("true")].sort_values(["score_shape", "hpo_id"], ascending=[False, True])),
        ("direction_conflict", frame[frame["score_conflict"].fillna(0) > 0].sort_values(["score_conflict", "hpo_id"], ascending=[False, True])),
        ("broad_profile", frame[(frame["n_hpo_cpo"] >= 20) | (frame["n_mpo_cpo"] >= 20)].sort_values(["score_shape", "hpo_id"], ascending=[False, True])),
    ]
    rows = []
    used = set()
    for category, candidates in categories:
        for row in candidates.itertuples(index=False):
            key = (str(row.hpo_id), str(row.mpo_id))
            if key in used:
                continue
            used.add(key)
            rows.append(
                {
                    "category": category,
                    "hpo_id": key[0],
                    "mpo_id": key[1],
                    "shape": float(row.score_shape),
                    "support": float(row.score_support),
                    "conflict": float(row.score_conflict),
                    "origins": str(row.source_origin_summary),
                }
            )
            break
    return pd.DataFrame(rows)


def _gene_examples(root: Path) -> pd.DataFrame:
    path = root / "explanations/gene_evidence_chains.parquet"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame[frame["ranking_view"] == "one_to_one"].sort_values(
        ["human_gene_rank", "disease_id", "human_gene_id"]
    )
    top = frame.drop_duplicates("disease_id").set_index("disease_id")
    cases = pd.read_csv(root / "evaluation/disease_gene_cases.tsv", sep="\t")
    cases = cases[(cases["view"] == "one_to_one") & (cases["split"] == "test")]
    success = cases[(cases["best_rank_min"] > 0) & (cases["best_rank_min"] <= 10)].sort_values(
        ["best_rank_min", "query_id"]
    )
    missed = cases[(cases["best_rank_min"] == 0) | (cases["best_rank_min"] > 20)].sort_values(
        ["query_id"]
    )
    rows = []
    for category, selected in (("known_gene_hit_top10", success), ("known_gene_missed_top20", missed)):
        for case in selected.itertuples(index=False):
            disease_id = str(case.query_id)
            if disease_id not in top.index:
                continue
            candidate = top.loc[disease_id]
            rows.append(
                {
                    "category": category,
                    "disease_id": disease_id,
                    "known_rank_min": int(case.best_rank_min),
                    "known_rank_max": int(case.best_rank_max),
                    "displayed_top_gene": str(candidate.human_gene_symbol),
                    "human_gene_id": str(candidate.human_gene_id),
                    "human_gene_score": float(candidate.human_gene_score),
                    "mouse_gene_id": str(candidate.mouse_gene_id),
                    "model_profile_id": str(candidate.model_profile_id),
                    "hpo_id": str(candidate.hpo_id),
                    "mpo_id": str(candidate.mpo_id),
                    "evidence_attribution": str(candidate.evidence_attribution),
                }
            )
            if sum(row["category"] == category for row in rows) == 5:
                break
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("NA" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_report(config: dict[str, Any]) -> dict[str, object]:
    root = output_root(config)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    _coverage_figure(root)
    _matrix_figure(root)
    _evaluation_figure(root)
    _disease_coverage_figure(root)
    _orthology_figure(root)

    profile = _json(root / "profiles/profile_manifest.json")
    matrix = _json(root / "matrices/scoring_manifest.json")
    disease = _json(root / "disease_model/profile_manifest.json")
    disease_scores = _json(root / "disease_model/scores/scoring_manifest.json")
    gene = _json(root / "gene_prediction/gene_ranking_manifest.json")
    evaluation = _json(root / "evaluation/evaluation_manifest.json")
    validation_path = root / "validation/validation_report.json"
    validation = _json(validation_path) if validation_path.is_file() else {"status": "NOT_RUN"}
    term_examples = _term_examples(root)
    gene_examples = _gene_examples(root)
    term_examples.to_csv(root / "reports/selected_term_examples.tsv", sep="\t", index=False)
    gene_examples.to_csv(root / "reports/selected_gene_examples.tsv", sep="\t", index=False)

    metric_table = pd.DataFrame(_metric_rows(root))[
        [
            "task",
            "view",
            "direction",
            "split",
            "evaluation_scope",
            "metric",
            "k",
            "value",
            "denominator",
        ]
    ]
    report = f"""# HPO-MPO CPO similarity and gene prioritization v2

Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## Completed scope

- Indexed all {profile['counts']['hpo_terms']:,} HPO phenotypic-abnormality terms and {profile['counts']['mpo_terms']:,} active MPO terms outside the verified normal branch.
- Applied the frozen node policy exactly: leaf=self; inherit-only internal/root=inherited; self-required internal=self union inherited. Only descendant leaf self evidence propagates upward.
- Built the nine-dataset IMPC-inspired term matrix with intrinsic ontology IC, CL/GO Lin similarity, product CPO similarity, and bidirectional equal BMA.
- Scored {disease['disease_count']:,} disease profiles against {disease['model_count']:,} distinct mouse genotype profiles, then aggregated {disease['single_gene_model_count']:,} single-gene models before frozen mouse-human orthology conversion.
- Produced strict one-to-one and ambiguity-preserving expanded human gene rankings and retrospective evaluations.

## Coverage

The final term matrix contains {matrix['view_counts']['final']['finite_pair_count']:,} computable pairs and {matrix['view_counts']['final']['positive_pair_count']:,} positive pairs. Final profiles are available for {matrix['view_counts']['final']['hpo_profile_count']:,} HPO terms and {matrix['view_counts']['final']['mpo_profile_count']:,} MPO terms. A total of {profile['counts']['missing_required_self_terms']:,} required-self terms currently lack a usable complete self CPO; inherited evidence remains scoreable and this condition is explicitly flagged.

## Evaluation snapshot

{_markdown_table(metric_table)}

These are retrieval metrics against existing MHMI mappings and known disease-gene relationships. Unlisted relationships are not treated as reliable negatives, so AUROC and precision are intentionally omitted.

## Fixed-rule term cases

{_markdown_table(term_examples)}

## Fixed-rule gene evidence-chain cases

{_markdown_table(gene_examples)}

Cases are selected deterministically from the locked test split: five diseases whose known gene is retrieved in the top 10 and five whose known gene is absent from the top 20. The displayed chain is the method's rank-1 candidate for that disease; in failure rows it is intentionally not presented as the known causal gene.

`model_phenotype_observed` means the selected model is annotated with the displayed MPO. `mechanism_transferred_from_term_profile` means the CPO mechanism belongs to the general MPO/HPO profile and must not be described as directly observed in that candidate model. Inherited MPO evidence additionally requires descendant interpretation.

## Figures

- `figures/profile_coverage.png`: self, inherited, and final profile availability by node policy.
- `figures/term_score_distributions.png`: fixed-sample shape, F-R asymmetry, support, and conflict distributions.
- `figures/ranking_metrics.png`: term-mapping and test disease-gene Hit@k.
- `figures/disease_model_coverage.png`: relationship between computable coverage and evidence-adjusted model score.
- `figures/orthology_coverage_ties.png`: one-to-one versus expanded ranking coverage, ambiguity, and ties.

## Validation and limitations

- Validation status: `{validation.get('status', 'NOT_RUN')}`.
- The disease-gene result is `retrospective_retrieval`, not proof of prospective novel-gene prediction. Source-filtered evaluation is `{evaluation['source_filtered_evaluation']}` because the HPO CPO export does not expose a complete disease/gene derivation key.
- Term-level direction support/conflict is complete. Full disease-model direction matrices were not materialized; `{disease_scores['direction_status']}`. Gene chains report direction for their selected structural term pair.
- The current evaluation includes final, self-only, inherited-only, CL-only, and GO-only views. Exact-structure Jaccard and direct-only/mechanism-only disease-gene reruns are not materialized in this run.
- Top-k tables contain positive candidates only. NA means no computable complete CPO comparison and is not rewritten as zero.
- Max-over-model gene aggregation can favor genes with more tested mouse models; results should be stratified by model count before biological interpretation.

## Reproduction

```bash
PYTHONPYCACHEPREFIX=/tmp/mission4_pycache python3 -B -m cpo_pipeline run-all --config config/cpo_v2.yaml --resume
PYTHONPYCACHEPREFIX=/tmp/mission4_pycache python3 -B -m cpo_pipeline validate --config config/cpo_v2.yaml
```
"""
    write_text(root / "reports/final_report.md", report)

    readme = """# CPO mapping and gene prioritization v2 results

This directory is the executable output of `HPO_MPO_CPO_CODEX_EXECUTION_PLAN (1).md`.

- `reports/final_report.md`: results, examples, figures, and limitations.
- `validation/validation_report.md`: numerical and policy checks.
- `matrices/hpo_mpo_cpo.h5`: the nine formal all-term matrices.
- `mapping/`: bidirectional term candidates and CL/GO auxiliary candidates.
- `disease_model/` and `gene_prediction/`: model scores and human gene rankings.
- `explanations/`: term, auxiliary, and candidate-gene provenance.

Use `python3 -B -m cpo_pipeline --help` for stage and query commands.
"""
    write_text(root / "README.md", readme)

    code_files = sorted(Path(config["_project_root"]).glob("cpo_pipeline/*.py"))
    run_manifest = {
        "schema_version": 2,
        "run_name": config["run_name"],
        "status": "complete_with_declared_limitations",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_digest(config),
        "config": public_config(config),
        "code": {path.name: file_record(path) for path in code_files},
        "indices": {
            "hpo": file_record(root / "indices/hpo_index.tsv"),
            "mpo": file_record(root / "indices/mpo_index.tsv"),
        },
        "scoring": matrix,
        "disease_model_scoring": disease_scores,
        "gene_ranking": gene,
        "evaluation": evaluation,
        "validation": validation,
        "declared_limitations": [
            "source_filtered_evaluation_not_available",
            "full_disease_model_direction_not_materialized",
            "exact_jaccard_and_direct_mechanism_gene_ablations_not_materialized",
        ],
    }
    write_json(root / "run_manifest.json", run_manifest)
    result = {
        "status": run_manifest["status"],
        "report": str(root / "reports/final_report.md"),
        "figure_count": 5,
        "term_example_count": len(term_examples),
        "gene_example_count": len(gene_examples),
    }
    write_json(root / "reports/report_manifest.json", result)
    return result
