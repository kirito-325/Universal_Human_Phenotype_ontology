from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .baseline import (
    BASE_GENE_FIELDS,
    HPO_MPO_EVIDENCE_FIELDS,
    MODEL_SCORE_FIELDS,
    run_baseline,
)
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
    input_record,
    load_diseases,
    load_model_phenotypes,
    load_models,
    require_columns,
    sha256,
    write_json,
    write_tsv,
)
from .null_models import NULL_ITERATION_FIELDS, NULL_SUMMARY_FIELDS, run_gene_set_null
from .plotting import generate_benchmark_figures
from .report import build_report
from .rerank import (
    ABLATION_ARMS,
    CANDIDATE_CPO_FIELDS,
    DISEASE_CPO_FIELDS,
    EVIDENCE_PATH_FIELDS,
    FINAL_RANKING_FIELDS,
    LEAKAGE_AUDIT_FIELDS,
    rerank,
)
from .resources import load_gene_set_resources, load_hpo_cpo
from .similarity import build_similarity_provider


def validate_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = config.get("inputs", {})
    similarity_provider = str(config.get("similarity", {}).get("provider", "matrix"))
    required = {
        "diseases": [{"disease_id", "hpo_id"}, {"disease_id", "hp_id"}],
        "models": [{"model_id", "human_gene_symbol"}],
        "model_phenotypes": [{"model_id", "mpo_id"}, {"model_id", "mp_id"}],
        "hpo_cpo": [{"hpo_id", "cell_id", "function_id"}, {"hpo_id", "cl_id", "go_id"}],
        "cell_markers": [{"cell_id", "gene_symbol"}, {"catalog_id", "gene"}],
        "go_functions": [{"function_id", "gene_symbol"}, {"go_id", "human_gene_symbol"}],
        "gold_standard": [{"disease_id", "gene_symbol"}, {"disease_id", "human_gene_symbol"}],
    }
    rows = []
    for name, alternatives in required.items():
        raw_path = inputs.get(name) or (inputs.get("gold_models") if name == "gold_standard" else None)
        if not raw_path:
            raise ValueError(f"Missing inputs.{name} in configuration")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        require_columns(path, alternatives)
        rows.append({"input": name, **input_record(path), "status": "valid"})
    similarity_path = inputs.get("similarity")
    if not similarity_path:
        raise ValueError("Missing inputs.similarity in configuration")
    similarity = Path(similarity_path)
    if not similarity.is_file():
        raise FileNotFoundError(similarity)
    if similarity_provider == "matrix":
        require_columns(similarity, [{"hpo_id", "mpo_id"}, {"hp_id", "mp_id"}])
    elif similarity_provider == "embedding":
        manifest = Path(str(similarity) + ".manifest.json")
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows.append(
            {"input": "similarity_manifest", **input_record(manifest), "status": "valid"}
        )
    else:
        raise ValueError(f"Unsupported similarity.provider: {similarity_provider}")
    rows.append({"input": "similarity", **input_record(similarity), "status": "valid"})
    evaluation_mode = str(config.get("project", {}).get("evaluation_mode", "practical"))
    if evaluation_mode == "source_label_broad":
        manifest_key = "source_snapshot_manifest"
    elif evaluation_mode == "open_candidate_benchmark":
        manifest_key = "open_candidate_manifest"
    else:
        manifest_key = "freeze_manifest"
    manifest_path = inputs.get(manifest_key)
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if evaluation_mode == "source_label_broad":
            if not manifest.get("source_labels_frozen") or not manifest.get(
                "middle_layer_independent"
            ):
                raise ValueError(
                    "Refusing CPO evaluation: broad source labels are not independently fixed"
                )
        elif evaluation_mode == "open_candidate_benchmark" and not (
            manifest.get("frozen")
            and manifest.get("candidate_universe_frozen")
            and manifest.get("middle_layer_independent")
        ):
            raise ValueError("Open candidate universe is not independently frozen")
        elif evaluation_mode != "open_candidate_benchmark" and (
            not manifest.get("frozen") or not manifest.get("middle_layer_independent")
        ):
            raise ValueError("Refusing CPO evaluation: benchmark is not independently frozen")
        rows.append({"input": manifest_key, **input_record(path), "status": "valid"})
    elif evaluation_mode in {"source_label_broad", "open_candidate_benchmark"}:
        raise ValueError(f"Missing inputs.{manifest_key} in configuration")
    return rows


def verify_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config.get("inputs", {})
    evaluation_mode = str(config.get("project", {}).get("evaluation_mode", "practical"))
    broad = evaluation_mode == "source_label_broad"
    manifest_key = "source_snapshot_manifest" if broad else "freeze_manifest"
    manifest_path = inputs.get(manifest_key)
    if not manifest_path:
        return {"accepted": False, "reason": f"no_{manifest_key}_configured"}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if broad:
        if not manifest.get("source_labels_frozen") or not manifest.get(
            "middle_layer_independent"
        ):
            raise ValueError(
                "Broad source snapshot does not authorize middle-layer evaluation"
            )
    elif not manifest.get("frozen") or not manifest.get("middle_layer_independent"):
        raise ValueError("Benchmark freeze manifest does not authorize middle-layer evaluation")
    mismatches = []
    configured_by_name = {
        "diseases.tsv": inputs.get("diseases"),
        "models.tsv": inputs.get("models"),
        "model_phenotypes.tsv": inputs.get("model_phenotypes"),
    }
    if broad:
        configured_by_name["gold_standard.tsv"] = inputs.get("gold_standard")
    for name, configured in configured_by_name.items():
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if broad and (not expected or not configured):
            mismatches.append(name)
        elif expected and configured and sha256(configured) != expected:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            f"Configured inputs do not match evaluation snapshot hashes: {', '.join(mismatches)}"
        )
    return {
        "accepted": True,
        "evaluation_mode": evaluation_mode,
        "manifest_type": "source_label_snapshot" if broad else "curated_benchmark_freeze",
        "frozen_at": manifest.get("frozen_at") or manifest.get("generated_at"),
        "review_policy": manifest.get("review_policy"),
        "publication_grade": bool(manifest.get("publication_grade", not broad)),
        "evidence_level": (
            "source_hpo_mgi_unreviewed" if broad else "curated_gold_models"
        ),
        "middle_layer_independent": True,
        "manifest_sha256": sha256(manifest_path),
    }


def _write_gene_set_tables(output: Path, resources: Any) -> None:
    marker_rows = [
        {
            "cell_id": cell,
            "cell_name": resources.cell_names.get(cell, cell),
            "gene_symbol": gene,
            "marker_score": score,
            "candidate_set_size": resources.marker_set_sizes.get(cell, 0),
        }
        for (cell, gene), score in sorted(resources.marker_scores.items())
    ]
    function_rows = [
        {
            "function_id": function,
            "function_name": resources.function_names.get(function, function),
            "gene_symbol": gene,
            "function_score": score,
            "candidate_set_size": resources.function_set_sizes.get(function, 0),
        }
        for (function, gene), score in sorted(resources.function_scores.items())
    ]
    write_tsv(
        output / "gene_sets/marker_specificity.tsv",
        marker_rows,
        ["cell_id", "cell_name", "gene_symbol", "marker_score", "candidate_set_size"],
    )
    write_tsv(
        output / "gene_sets/function_specificity.tsv",
        function_rows,
        ["function_id", "function_name", "gene_symbol", "function_score", "candidate_set_size"],
    )


def run_pipeline(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_summary = validate_inputs(config)
    freeze_summary = verify_freeze(config)
    (output / "resolved_config.yaml").write_text(dump_config(dict(config)), encoding="utf-8")
    write_tsv(output / "qc/input_summary.tsv", input_summary, ["input", "path", "bytes", "sha256", "status"])
    inputs = config["inputs"]
    filters = config.get("filters", {})
    disease_filter = set(filters.get("disease_ids", []))
    gene_filter = set(filters.get("model_gene_symbols", []))
    diseases = load_diseases(inputs["diseases"], disease_filter or None)
    models = load_models(inputs["models"], gene_filter or None)
    phenotypes = load_model_phenotypes(inputs["model_phenotypes"], set(models))
    if not diseases:
        raise ValueError("No disease profiles passed the configured filters")
    if not models or not phenotypes:
        raise ValueError("No phenotype-bearing models passed the configured filters")
    hpo_ids = {term.hpo_id for terms in diseases.values() for term in terms}
    mpo_ids = {term.mpo_id for terms in phenotypes.values() for term in terms if term.status == "abnormal"}
    provider = build_similarity_provider(
        inputs["similarity"], hpo_ids, mpo_ids, config.get("similarity", {})
    )
    model_rows, base_rows, evidence_rows = run_baseline(
        diseases, models, phenotypes, provider, config.get("model_scoring", {})
    )
    write_tsv(output / "baseline/disease_model_scores.tsv", model_rows, MODEL_SCORE_FIELDS)
    write_tsv(output / "baseline/base_gene_rankings.tsv", base_rows, BASE_GENE_FIELDS)
    write_tsv(output / "baseline/hpo_mpo_evidence.tsv", evidence_rows, HPO_MPO_EVIDENCE_FIELDS)

    hpo_cpo = load_hpo_cpo(inputs["hpo_cpo"], hpo_ids, config.get("hpo_cpo", {}))
    candidate_genes = {str(row["gene_symbol"]).upper() for row in base_rows}
    resources = load_gene_set_resources(
        inputs["cell_markers"],
        inputs["go_functions"],
        hpo_cpo,
        candidate_genes,
        config.get("gene_set_scoring", {}),
    )
    _write_gene_set_tables(output, resources)
    arms = dict(ABLATION_ARMS)
    arms["A3_marker_function"] = config.get("gene_set_scoring", {})
    arm_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, weights in arms.items():
        arm_results[name] = rerank(
            base_rows,
            evidence_rows,
            diseases,
            hpo_cpo,
            resources,
            config,
            experiment=name,
            component_weights=weights,
            emit_details=name == "A3_marker_function",
        )
    primary = arm_results["A3_marker_function"]
    write_tsv(output / "cpo/disease_cpo_profiles.tsv", primary["disease_cpo_profiles"], DISEASE_CPO_FIELDS)
    write_tsv(output / "cpo/candidate_cpo_profiles.tsv", primary["candidate_cpo_profiles"], CANDIDATE_CPO_FIELDS)
    write_tsv(output / "reranking/evidence_paths.tsv", primary["evidence_paths"], EVIDENCE_PATH_FIELDS)
    write_tsv(output / "reranking/final_gene_rankings.tsv", primary["final_rankings"], FINAL_RANKING_FIELDS)
    write_tsv(output / "reranking/candidate_cellpath_scores.tsv", primary["final_rankings"], FINAL_RANKING_FIELDS)

    gold_path = inputs.get("gold_standard") or inputs.get("gold_models")
    gold_genes = load_gold_genes(gold_path)
    benchmark_metric_rows: list[dict[str, Any]] = []
    all_disease_metrics: list[dict[str, Any]] = []
    metrics_by_arm: dict[str, dict[str, float]] = {}
    disease_by_arm: dict[str, list[dict[str, Any]]] = {}
    for name, result in arm_results.items():
        disease_metrics, metrics = evaluate_rankings(
            result["final_rankings"], gold_genes, "final_rank", name
        )
        disease_by_arm[name] = disease_metrics
        metrics_by_arm[name] = metrics
        all_disease_metrics.extend(disease_metrics)
        benchmark_metric_rows.extend(metrics_as_rows(name, metrics))

    statistical_rows: list[dict[str, Any]] = []
    rank_change_rows: list[dict[str, Any]] = []
    iterations = int(config.get("benchmark", {}).get("bootstrap_iterations", 1000))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for name in ("A1_marker", "A2_function", "A3_marker_function"):
        changes, statistics = compare_disease_ranks(
            disease_by_arm["A0_surface"], disease_by_arm[name], name, iterations, seed
        )
        rank_change_rows.extend(changes)
        statistical_rows.append(statistics)

    masked_by_disease = {disease_id: set(genes) for disease_id, genes in gold_genes.items()}
    strict = rerank(
        base_rows,
        evidence_rows,
        diseases,
        hpo_cpo,
        resources,
        config,
        experiment="A3_strict_masked",
        component_weights=config.get("gene_set_scoring", {}),
        masked_by_disease=masked_by_disease,
        emit_details=True,
    )
    strict_disease, strict_metrics = evaluate_rankings(
        strict["final_rankings"], gold_genes, "final_rank", "A3_strict_masked"
    )
    all_disease_metrics.extend(strict_disease)
    benchmark_metric_rows.extend(metrics_as_rows("A3_strict_masked", strict_metrics))
    write_tsv(output / "leakage_control/masked_rankings.tsv", strict["final_rankings"], FINAL_RANKING_FIELDS)
    write_tsv(output / "leakage_control/evidence_leakage_audit.tsv", strict["leakage_audit"], LEAKAGE_AUDIT_FIELDS)

    strict_changes, strict_statistics = compare_disease_ranks(
        disease_by_arm["A0_surface"], strict_disease, "A3_strict_masked", iterations, seed
    )
    rank_change_rows.extend(strict_changes)
    statistical_rows.append(strict_statistics)

    comparable_hpo: dict[str, set[str]] = {}
    for row in evidence_rows:
        disease_id = str(row["disease_id"])
        hpo_id = str(row["hpo_id"])
        if float(row.get("hpo_mpo_score", 0.0)) > 0 and hpo_id in hpo_cpo:
            comparable_hpo.setdefault(disease_id, set()).add(hpo_id)
    evaluable_ids = {
        disease_id for disease_id, hpo_ids_for_disease in comparable_hpo.items()
        if len(hpo_ids_for_disease) >= 2
    }
    primary_by_disease = {
        str(row["disease_id"]): row for row in primary["final_rankings"]
    }
    high_coverage_ids = {
        disease_id for disease_id, row in primary_by_disease.items()
        if float(row["cpo_coverage"]) >= 0.2
    }
    strata = {
        "all_frozen": set(gold_genes),
        "cpo_evaluable_ge2_hpo": evaluable_ids,
        "cpo_coverage_ge0.2": high_coverage_ids,
    }
    stratified_rows: list[dict[str, Any]] = []
    all_arm_rankings = {name: result["final_rankings"] for name, result in arm_results.items()}
    all_arm_rankings["A3_strict_masked"] = strict["final_rankings"]
    for stratum, disease_ids in strata.items():
        stratum_gold = {disease_id: gold_genes[disease_id] for disease_id in sorted(disease_ids) if disease_id in gold_genes}
        for name, rankings in all_arm_rankings.items():
            _, stratum_metrics = evaluate_rankings(rankings, stratum_gold, "final_rank", name)
            for metric, metric_value in stratum_metrics.items():
                if metric == "evaluated_diseases":
                    continue
                stratified_rows.append(
                    {
                        "stratum": stratum,
                        "experiment": name,
                        "disease_count": len(stratum_gold),
                        "metric": metric,
                        "value": metric_value,
                    }
                )
    write_tsv(
        output / "benchmark/stratified_metrics.tsv",
        stratified_rows,
        ["stratum", "experiment", "disease_count", "metric", "value"],
    )
    if evaluable_ids:
        evaluable_gold = {disease_id: gold_genes[disease_id] for disease_id in evaluable_ids if disease_id in gold_genes}
        evaluable_base, _ = evaluate_rankings(
            arm_results["A0_surface"]["final_rankings"], evaluable_gold, "final_rank", "A0_surface"
        )
        evaluable_a3, _ = evaluate_rankings(
            primary["final_rankings"], evaluable_gold, "final_rank", "A3_marker_function"
        )
        _, evaluable_statistics = compare_disease_ranks(
            evaluable_base,
            evaluable_a3,
            "A3_marker_function[cpo_evaluable_ge2_hpo]",
            iterations,
            seed,
        )
        statistical_rows.append(evaluable_statistics)

    write_tsv(output / "benchmark/benchmark_metrics.tsv", benchmark_metric_rows, BENCHMARK_METRIC_FIELDS)
    write_tsv(output / "benchmark/ablation_metrics.tsv", benchmark_metric_rows, BENCHMARK_METRIC_FIELDS)
    write_tsv(output / "benchmark/disease_level_metrics.tsv", all_disease_metrics, DISEASE_METRIC_FIELDS)
    write_tsv(output / "benchmark/statistical_tests.tsv", statistical_rows, STATISTICAL_TEST_FIELDS)
    write_tsv(output / "reranking/rank_changes.tsv", rank_change_rows, RANK_CHANGE_FIELDS)

    null_settings = config.get("null_models", {})
    if bool(null_settings.get("enabled", True)) and int(null_settings.get("iterations", 0)) > 0:
        null_iterations, null_summary = run_gene_set_null(
            base_rows,
            evidence_rows,
            diseases,
            hpo_cpo,
            resources,
            config,
            gold_genes,
            metrics_by_arm["A3_marker_function"],
        )
    else:
        null_iterations, null_summary = [], []
    write_tsv(output / "null_models/permutation_metrics.tsv", null_iterations, NULL_ITERATION_FIELDS)
    write_tsv(output / "null_models/observed_vs_null.tsv", null_summary, NULL_SUMMARY_FIELDS)

    coverage_rows = []
    disease_profile_by_id = Counter(row["disease_id"] for row in primary["disease_cpo_profiles"])
    for disease_id, terms in sorted(diseases.items()):
        final_for_disease = [row for row in primary["final_rankings"] if row["disease_id"] == disease_id]
        coverage_rows.append(
            {
                "disease_id": disease_id,
                "hpo_count": len(terms),
                "candidate_gene_count": len(final_for_disease),
                "hpo_cpo_profile_count": disease_profile_by_id[disease_id],
                "comparable_hpo_count": len(comparable_hpo.get(disease_id, set())),
                "cpo_evaluable": disease_id in evaluable_ids,
                "cpo_coverage": final_for_disease[0]["cpo_coverage"] if final_for_disease else 0.0,
            }
        )
    write_tsv(
        output / "qc/coverage_summary.tsv",
        coverage_rows,
        ["disease_id", "hpo_count", "candidate_gene_count", "hpo_cpo_profile_count", "comparable_hpo_count", "cpo_evaluable", "cpo_coverage"],
    )
    figure_paths = (
        generate_benchmark_figures(
            output,
            disease_by_arm,
            rank_change_rows,
            metrics_by_arm,
            coverage_rows,
            null_iterations,
        )
        if bool(config.get("plots", {}).get("enabled", True))
        else []
    )
    practical_mrr = metrics_by_arm["A3_marker_function"]["mrr"]
    baseline_mrr = metrics_by_arm["A0_surface"]["mrr"]
    strict_mrr = strict_metrics["mrr"]
    warnings = [
        "Candidate-pool size must be reported with retrieval metrics.",
        "Single-cell validation does not prove candidate-gene causality.",
    ]
    if practical_mrr > baseline_mrr and strict_mrr < practical_mrr:
        warnings.append(
            "Practical reranking improves MRR but the gain does not survive strict direct-gene masking; "
            "do not claim new-gene generalization from the practical result."
        )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "diseases": len(diseases),
            "models": len(models),
            "genes": len(candidate_genes),
            "model_phenotype_rows": sum(len(rows) for rows in phenotypes.values()),
            "similarity_pairs": provider.pair_count,
            "hpo_terms_with_cpo": len(hpo_cpo),
            "marker_pairs": len(resources.marker_scores),
            "function_pairs": len(resources.function_scores),
            "cpo_evaluable_diseases": len(evaluable_ids),
            "leakage_audit_rows": len(strict["leakage_audit"]),
            "figures": len(figure_paths),
        },
        "freeze": freeze_summary,
        "primary_experiment": "A3_marker_function",
        "evaluation_mode": config.get("project", {}).get("evaluation_mode", "practical"),
        "primary_metrics": metrics_by_arm["A3_marker_function"],
        "baseline_metrics": metrics_by_arm["A0_surface"],
        "strict_metrics": strict_metrics,
        "inputs": {
            str(row["input"]): {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in input_summary
        },
        "warnings": warnings,
    }
    write_json(output / "run_manifest.json", summary)
    build_report(output, summary, benchmark_metric_rows, statistical_rows)
    return summary
