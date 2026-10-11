from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import dump_config
from .evaluation import (
    BENCHMARK_METRIC_FIELDS,
    DISEASE_METRIC_FIELDS,
    RANK_CHANGE_FIELDS,
    STATISTICAL_TEST_FIELDS,
    compare_disease_ranks,
    evaluate_rankings,
    load_gold_genes,
    metrics_as_rows,
)
from .io import (
    as_bool,
    as_float,
    input_record,
    load_diseases,
    load_model_phenotypes,
    load_models,
    read_tsv,
    require_columns,
    write_json,
    write_tsv,
)
from .pipeline import verify_freeze


MODEL_FIELDS = [
    "disease_id",
    "disease_name",
    "model_id",
    "model_name",
    "gene_symbol",
    "phenodigm_score",
    "avg_percentage_score",
    "max_percentage_score",
    "raw_avg_score",
    "raw_max_score",
    "ideal_avg_score",
    "ideal_max_score",
    "input_hpo_count",
    "mapped_hpo_count",
    "input_mpo_count",
    "legacy_mpo_count",
]

GENE_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "phenodigm_score",
    "phenodigm_rank",
    "best_model_id",
    "best_model_name",
    "avg_percentage_score",
    "max_percentage_score",
    "candidate_model_count",
    "positive_model_count",
]

COVERAGE_FIELDS = [
    "disease_id",
    "disease_name",
    "input_hpo_count",
    "mapped_hpo_count",
    "hpo_coverage",
    "candidate_gene_count",
    "positive_candidate_genes",
    "gold_genes",
    "gold_genes_positive",
]


class LegacyPhenodigmMatrix:
    def __init__(
        self,
        matrix_path: str | Path,
        presence_path: str | Path,
        candidate_mpo_ids: set[str],
    ) -> None:
        self.path = str(Path(matrix_path).resolve())
        self.scores: dict[str, dict[str, float]] = defaultdict(dict)
        self.best_by_hpo: dict[str, tuple[float, str]] = {}
        self.present_mpo = {
            str(row["mp_id"])
            for row in read_tsv(presence_path)
            if as_bool(
                row.get("present_in_legacy_matrix", "")
                or row.get("present_in_2602_mp_ontology", "")
            )
        }
        self.rows = 0
        self.candidate_pairs = 0
        for row in read_tsv(matrix_path):
            hp_id = str(row.get("hp_id", ""))
            mp_id = str(row.get("mp_id", ""))
            score = as_float(row.get("score"), 0.0)
            if not hp_id or not mp_id or score < 0:
                continue
            self.rows += 1
            previous_best = self.best_by_hpo.get(hp_id)
            if previous_best is None or score > previous_best[0] or (
                math.isclose(score, previous_best[0]) and mp_id < previous_best[1]
            ):
                self.best_by_hpo[hp_id] = (score, mp_id)
            if mp_id in candidate_mpo_ids:
                current = self.scores[hp_id].get(mp_id)
                if current is None or score > current:
                    self.scores[hp_id][mp_id] = score
                self.candidate_pairs += 1

    def get(self, hp_id: str, mp_id: str) -> float:
        return self.scores.get(hp_id, {}).get(mp_id, 0.0)


def combine_percentage_scores(avg_percentage: float, max_percentage: float) -> float:
    return (float(avg_percentage) + float(max_percentage)) / 2.0


def score_phenodigm_profile(
    hpo_ids: Sequence[str], mpo_ids: Sequence[str], matrix: LegacyPhenodigmMatrix
) -> dict[str, float | int]:
    hp = list(dict.fromkeys(item for item in hpo_ids if item in matrix.best_by_hpo))
    mp = list(dict.fromkeys(item for item in mpo_ids if item in matrix.present_mpo))
    if not hp or not mp:
        return {
            "phenodigm_score": 0.0,
            "avg_percentage_score": 0.0,
            "max_percentage_score": 0.0,
            "raw_avg_score": 0.0,
            "raw_max_score": 0.0,
            "ideal_avg_score": 0.0,
            "ideal_max_score": 0.0,
            "mapped_hpo_count": len(hp),
            "legacy_mpo_count": len(mp),
        }

    forward_best = [max((matrix.get(h, m) for m in mp), default=0.0) for h in hp]
    reverse_best = [max((matrix.get(h, m) for h in hp), default=0.0) for m in mp]
    raw_max = max([*forward_best, *reverse_best], default=0.0)
    raw_avg = (sum(forward_best) + sum(reverse_best)) / (len(hp) + len(mp))

    ideal_mp = [matrix.best_by_hpo[h][1] for h in hp]
    ideal_forward = [matrix.best_by_hpo[h][0] for h in hp]
    ideal_reverse = [
        max((matrix.get(h, best_mp) for h in hp), default=0.0)
        for best_mp in ideal_mp
    ]
    ideal_max = max([*ideal_forward, *ideal_reverse], default=0.0)
    ideal_avg = (sum(ideal_forward) + sum(ideal_reverse)) / (2 * len(hp))
    avg_percentage = 100.0 * raw_avg / ideal_avg if ideal_avg else 0.0
    max_percentage = 100.0 * raw_max / ideal_max if ideal_max else 0.0
    combined = combine_percentage_scores(avg_percentage, max_percentage)
    return {
        "phenodigm_score": combined,
        "avg_percentage_score": avg_percentage,
        "max_percentage_score": max_percentage,
        "raw_avg_score": raw_avg,
        "raw_max_score": raw_max,
        "ideal_avg_score": ideal_avg,
        "ideal_max_score": ideal_max,
        "mapped_hpo_count": len(hp),
        "legacy_mpo_count": len(mp),
    }


def score_exomiser_current_profile(
    hpo_ids: Sequence[str], mpo_ids: Sequence[str], matrix: LegacyPhenodigmMatrix
) -> dict[str, float | int]:
    hp = list(dict.fromkeys(item for item in hpo_ids if item in matrix.best_by_hpo))
    model_mp = list(dict.fromkeys(mpo_ids))
    matched_mp = [
        mp_id
        for mp_id in model_mp
        if any(matrix.get(hp_id, mp_id) > 0 for hp_id in hp)
    ]
    if not hp or not matched_mp:
        return {
            "phenodigm_score": 0.0,
            "avg_percentage_score": 0.0,
            "max_percentage_score": 0.0,
            "raw_avg_score": 0.0,
            "raw_max_score": 0.0,
            "ideal_avg_score": 0.0,
            "ideal_max_score": 0.0,
            "mapped_hpo_count": len(hp),
            "legacy_mpo_count": len(matched_mp),
        }

    forward_best = [
        max((matrix.get(hp_id, mp_id) for mp_id in matched_mp), default=0.0)
        for hp_id in hp
    ]
    reverse_best = [
        max((matrix.get(hp_id, mp_id) for hp_id in hp), default=0.0)
        for mp_id in matched_mp
    ]
    raw_max = max([*forward_best, *reverse_best], default=0.0)
    raw_avg = (sum(forward_best) + sum(reverse_best)) / (
        len(hp) + len(matched_mp)
    )
    ideal_scores = [matrix.best_by_hpo[hp_id][0] for hp_id in hp]
    ideal_max = max(ideal_scores, default=0.0)
    ideal_avg = sum(ideal_scores) / len(hp)
    avg_percentage = 100.0 * raw_avg / ideal_avg if ideal_avg else 0.0
    max_percentage = 100.0 * raw_max / ideal_max if ideal_max else 0.0
    combined = min(100.0, combine_percentage_scores(avg_percentage, max_percentage))
    return {
        "phenodigm_score": combined,
        "avg_percentage_score": avg_percentage,
        "max_percentage_score": max_percentage,
        "raw_avg_score": raw_avg,
        "raw_max_score": raw_max,
        "ideal_avg_score": ideal_avg,
        "ideal_max_score": ideal_max,
        "mapped_hpo_count": len(hp),
        "legacy_mpo_count": len(matched_mp),
    }


def _validate_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = config.get("inputs", {})
    required = {
        "diseases": [{"disease_id", "hpo_id"}, {"disease_id", "hp_id"}],
        "models": [{"model_id", "human_gene_symbol"}],
        "model_phenotypes": [{"model_id", "mpo_id"}, {"model_id", "mp_id"}],
        "gold_standard": [{"disease_id", "gene_symbol"}],
        "current_baseline": [{"disease_id", "gene_symbol", "base_rank"}],
        "phenodigm_legacy_matrix": [{"hp_id", "mp_id", "simj", "ic", "score"}],
        "phenodigm_legacy_mpo_presence": [
            {"mp_id", "present_in_legacy_matrix", "has_benchmark_hpo_pair"},
            {"mp_id", "present_in_2602_mp_ontology", "has_benchmark_hpo_pair"},
        ],
    }
    result = []
    for name, alternatives in required.items():
        path = Path(str(inputs.get(name, "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        require_columns(path, alternatives)
        result.append({"input": name, **input_record(path), "status": "valid"})
    freeze_path = Path(str(inputs.get("freeze_manifest", "")))
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    result.append(
        {"input": "freeze_manifest", **input_record(freeze_path), "status": "valid"}
    )
    return result


def _rank(
    diseases: Mapping[str, Sequence[Any]],
    models: Mapping[str, Any],
    phenotypes: Mapping[str, Sequence[Any]],
    matrix: LegacyPhenodigmMatrix,
    profile_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scorers = {
        "published_symmetric": score_phenodigm_profile,
        "exomiser_current": score_exomiser_current_profile,
    }
    if profile_mode not in scorers:
        raise ValueError(f"Unsupported phenodigm_matrix.profile_mode: {profile_mode}")
    scorer = scorers[profile_mode]
    model_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        hpo_ids = [term.hpo_id for term in terms]
        by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for model_id, model in sorted(models.items()):
            mpo_ids = sorted(
                {
                    row.mpo_id
                    for row in phenotypes.get(model_id, ())
                    if row.status == "abnormal"
                }
            )
            score = scorer(hpo_ids, mpo_ids, matrix)
            row = {
                "disease_id": disease_id,
                "disease_name": terms[0].disease_name,
                "model_id": model_id,
                "model_name": model.model_name,
                "gene_symbol": model.human_gene_symbol,
                **score,
                "input_hpo_count": len(set(hpo_ids)),
                "input_mpo_count": len(mpo_ids),
            }
            model_rows.append(row)
            by_gene[model.human_gene_symbol].append(row)
        current = []
        for gene, rows in sorted(by_gene.items()):
            ordered = sorted(
                rows,
                key=lambda row: (-float(row["phenodigm_score"]), str(row["model_id"])),
            )
            best = ordered[0]
            current.append(
                {
                    "disease_id": disease_id,
                    "disease_name": terms[0].disease_name,
                    "gene_symbol": gene,
                    "phenodigm_score": best["phenodigm_score"],
                    "best_model_id": best["model_id"],
                    "best_model_name": best["model_name"],
                    "avg_percentage_score": best["avg_percentage_score"],
                    "max_percentage_score": best["max_percentage_score"],
                    "candidate_model_count": len(rows),
                    "positive_model_count": sum(
                        float(row["phenodigm_score"]) > 0 for row in rows
                    ),
                }
            )
        current.sort(key=lambda row: (-float(row["phenodigm_score"]), row["gene_symbol"]))
        for rank, row in enumerate(current, 1):
            row["phenodigm_rank"] = rank
        gene_rows.extend(current)
    return model_rows, gene_rows


def _write_report(
    output: Path,
    metrics: Mapping[str, Mapping[str, float]],
    comparisons: Sequence[Mapping[str, Any]],
    counts: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    lines = [
        f"# {settings['report_title']}",
        "",
        str(settings["report_description"]),
        "",
        "## Coverage",
        "",
        f"- Unique frozen HPO: {counts['input_hpo']}",
        f"- HPO represented in the matrix: {counts['mapped_hpo']}",
        f"- Unique candidate MPO: {counts['input_mpo']}",
        f"- Candidate MPO represented in the matrix: {counts['mapped_mpo']}",
        f"- Model scores above 100 after normalization: "
        f"{counts['model_scores_above_100']}",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['recall_at_1']:.4f} | {values['recall_at_5']:.4f} | "
            f"{values['recall_at_10']:.4f} | {values['mrr']:.4f} | "
            f"{values['ndcg_at_10']:.4f} | {values['median_rank']:.2f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for row in comparisons:
        lines.append(
            f"- {row['experiment']}: {row['wins']} wins / {row['ties']} ties / "
            f"{row['losses']} losses; Wilcoxon p={row['wilcoxon_p_value']:.6g}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(settings["interpretation_boundary"]),
        ]
    )
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_legacy_phenodigm_benchmark(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    raw_settings = config.get("phenodigm_matrix", {})
    profile_mode = str(raw_settings.get("profile_mode", "published_symmetric"))
    method_label = str(raw_settings.get("method_label", "legacy_PhenoDigm"))
    default_legacy = profile_mode == "published_symmetric"
    settings = {
        "profile_mode": profile_mode,
        "method_label": method_label,
        "report_title": raw_settings.get(
            "report_title",
            "Archived PhenoDigm matrix benchmark"
            if default_legacy
            else "Exomiser 2602 PhenoDigm matrix benchmark",
        ),
        "report_description": raw_settings.get(
            "report_description",
            "This run applies the published symmetric PhenoDigm profile formula to the "
            "frozen local HPO and mouse-model profiles using the archived OWLSim HP-MP matrix."
            if default_legacy
            else "This run applies the current Exomiser PhenoDigm scorer to frozen local "
            "HPO and mouse-model profiles using the Exomiser 2602 HP-MP matrix.",
        ),
        "interpretation_boundary": raw_settings.get(
            "interpretation_boundary",
            "The matrix is the historical PhenoDigm Google Code artifact, not the current "
            "IMPC ontology snapshot. Missing newer HPO and MP terms are excluded following "
            "the original method. This run uses the published symmetric formula and does "
            "not reproduce an apparent reverse-loop array-index typo in the archived Perl source."
            if default_legacy
            else "The pair matrix and profile scorer correspond to Exomiser 2602. Disease "
            "HPO and mouse genotype profiles remain fixed by the local benchmark, so this is "
            "a controlled local-input evaluation rather than a replay of IMPC live disease profiles.",
        ),
    }
    input_summary = _validate_inputs(config)
    freeze = verify_freeze(config)
    if not freeze.get("accepted"):
        raise ValueError("Legacy PhenoDigm benchmark requires the frozen cohort")
    write_tsv(
        output / "qc/input_summary.tsv",
        input_summary,
        ["input", "path", "bytes", "sha256", "status"],
    )
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )

    diseases = load_diseases(inputs["diseases"])
    all_models = load_models(inputs["models"])
    models = {
        model_id: model
        for model_id, model in all_models.items()
        if ";" not in model.human_gene_symbol
    }
    phenotypes = load_model_phenotypes(inputs["model_phenotypes"], set(models))
    models = {
        model_id: model
        for model_id, model in models.items()
        if any(row.status == "abnormal" for row in phenotypes.get(model_id, ()))
    }
    candidate_mpo = {
        row.mpo_id
        for model_id in models
        for row in phenotypes.get(model_id, ())
        if row.status == "abnormal"
    }
    matrix = LegacyPhenodigmMatrix(
        inputs["phenodigm_legacy_matrix"],
        inputs["phenodigm_legacy_mpo_presence"],
        candidate_mpo,
    )
    model_rows, gene_rows = _rank(
        diseases, models, phenotypes, matrix, profile_mode
    )
    write_tsv(output / "rankings/model_scores.tsv", model_rows, MODEL_FIELDS)
    write_tsv(output / "rankings/gene_rankings.tsv", gene_rows, GENE_FIELDS)

    gold_genes = load_gold_genes(inputs["gold_standard"])
    hpo_union = {term.hpo_id for terms in diseases.values() for term in terms}
    coverage_rows = []
    by_disease_gene = defaultdict(list)
    for row in gene_rows:
        by_disease_gene[str(row["disease_id"])].append(row)
    for disease_id, terms in sorted(diseases.items()):
        hp = {term.hpo_id for term in terms}
        positive = {
            str(row["gene_symbol"])
            for row in by_disease_gene[disease_id]
            if float(row["phenodigm_score"]) > 0
        }
        gold = gold_genes.get(disease_id, set())
        coverage_rows.append(
            {
                "disease_id": disease_id,
                "disease_name": terms[0].disease_name,
                "input_hpo_count": len(hp),
                "mapped_hpo_count": len(hp.intersection(matrix.best_by_hpo)),
                "hpo_coverage": len(hp.intersection(matrix.best_by_hpo)) / len(hp),
                "candidate_gene_count": len(by_disease_gene[disease_id]),
                "positive_candidate_genes": len(positive),
                "gold_genes": sorted(gold),
                "gold_genes_positive": sorted(gold.intersection(positive)),
            }
        )
    write_tsv(output / "qc/coverage_by_disease.tsv", coverage_rows, COVERAGE_FIELDS)

    methods: list[tuple[str, list[Mapping[str, Any]], str]] = [
        ("current_A0", list(read_tsv(inputs["current_baseline"])), "base_rank"),
        (method_label, gene_rows, "phenodigm_rank"),
    ]
    official_path = inputs.get("official_phenodigm_rankings")
    if official_path and Path(official_path).is_file():
        methods.append(
            ("official_live_PhenoDigm", list(read_tsv(official_path)), "phenodigm_rank")
        )
    metrics = {}
    disease_metrics = {}
    metric_rows = []
    all_disease_rows = []
    for name, rows, rank_key in methods:
        per_disease, summary = evaluate_rankings(rows, gold_genes, rank_key, name)
        metrics[name] = summary
        disease_metrics[name] = per_disease
        metric_rows.extend(metrics_as_rows(name, summary))
        all_disease_rows.extend(per_disease)
    write_tsv(
        output / "benchmark/benchmark_metrics.tsv", metric_rows, BENCHMARK_METRIC_FIELDS
    )
    write_tsv(
        output / "benchmark/disease_metrics.tsv",
        all_disease_rows,
        DISEASE_METRIC_FIELDS,
    )

    comparisons = []
    rank_changes = []
    iterations = int(config.get("benchmark", {}).get("bootstrap_iterations", 10_000))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for reference in [name for name in metrics if name != method_label]:
        changes, summary = compare_disease_ranks(
            disease_metrics[reference],
            disease_metrics[method_label],
            f"{method_label}_vs_{reference}",
            iterations,
            seed,
        )
        rank_changes.extend(changes)
        comparisons.append(summary)
    write_tsv(
        output / "benchmark/rank_changes.tsv", rank_changes, RANK_CHANGE_FIELDS
    )
    write_tsv(
        output / "benchmark/statistical_tests.tsv",
        comparisons,
        STATISTICAL_TEST_FIELDS,
    )

    counts = {
        "diseases": len(diseases),
        "candidate_genes": len({model.human_gene_symbol for model in models.values()}),
        "candidate_models": len(models),
        "input_hpo": len(hpo_union),
        "mapped_hpo": len(hpo_union.intersection(matrix.best_by_hpo)),
        "input_mpo": len(candidate_mpo),
        "mapped_mpo": len(candidate_mpo.intersection(matrix.present_mpo)),
        "matrix_subset_rows": matrix.rows,
        "candidate_pair_rows": matrix.candidate_pairs,
        "diseases_with_positive_gold_gene": sum(
            bool(row["gold_genes_positive"]) for row in coverage_rows
        ),
        "model_scores_above_100": sum(
            float(row["phenodigm_score"]) > 100 for row in model_rows
        ),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method_label,
        "profile_mode": profile_mode,
        "pair_score": "sqrt(simJ * IC)",
        "profile_score": (
            "mean(normalized symmetric best-match average, normalized maximum pair)"
            if default_legacy
            else "current Exomiser semi-symmetric match average plus normalized maximum pair"
        ),
        "gene_aggregation": "maximum model score per gene",
        "freeze": freeze,
        "inputs": {row["input"]: row for row in input_summary},
        "counts": counts,
        "metrics": metrics,
        "comparisons": comparisons,
        "limitations": list(raw_settings.get("limitations", [])) or (
            [
                "The archived matrix predates many HPO and MP terms in the frozen benchmark.",
                "Missing ontology terms are excluded as specified by the original PhenoDigm method.",
                "The published symmetric formula is used; an apparent archived Perl index typo is not reproduced.",
            ]
            if default_legacy
            else [
                "The 2602 matrix uses HPO 2026-01-08 and MP 2026-01-27; the frozen profiles were assembled later.",
                "The local frozen disease profiles differ from the live IMPC disease annotation profiles.",
            ]
        ),
    }
    write_json(output / "run_manifest.json", manifest)
    _write_report(output, metrics, comparisons, counts, settings)
    return manifest
