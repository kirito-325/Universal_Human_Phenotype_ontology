from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .binary_cpo import (
    GRAPH_EDGE_FIELDS,
    BinaryCpoGraph,
    build_binary_graphs,
    graph_edge_rows,
    load_hpoa_background,
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
from .hpo_redundancy import (
    HPO_REDUCTION_FIELDS,
    NONREDUNDANT_HPO_FIELDS,
    HpoOntology,
    disease_terms_as_rows,
    reduce_hpo_profiles,
)
from .io import (
    input_record,
    load_diseases,
    load_model_phenotypes,
    load_models,
    sha256,
    write_json,
    write_tsv,
)
from .parameter_free_reranking import (
    ENRICHMENT_FIELDS,
    FINAL_PARAMETER_FREE_FIELDS,
    RAW_PATH_FIELDS,
    global_gene_hpo_reachability,
    rerank_constrained,
    rerank_unconstrained,
    score_graph,
)
from .phenotype_stability import (
    BASE_PARAMETER_FREE_FIELDS,
    CONSTRAINT_FIELDS,
    LOO_FIELDS,
    STABILITY_SUMMARY_FIELDS,
    calculate_phenotype_stability,
    calculate_exomiser_phenotype_stability,
)
from .phenodigm_legacy import LegacyPhenodigmMatrix
from .pipeline import validate_inputs, verify_freeze
from .similarity import build_similarity_provider


def _legacy_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    legacy = Path(config["inputs"].get("legacy_output_dir", "outputs/real_frozen")).resolve()
    if legacy == output or legacy in output.parents:
        raise ValueError("Parameter-free output must not overwrite the legacy output directory")
    tracked = [
        legacy / "run_manifest.json",
        legacy / "reranking/final_gene_rankings.tsv",
        legacy / "benchmark/benchmark_metrics.tsv",
    ]
    records = {
        str(path.relative_to(legacy)): input_record(path)
        for path in tracked
        if path.is_file()
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legacy_output_dir": str(legacy),
        "legacy_files": records,
        "legacy_outputs_modified": False,
        "new_method": "cpo_enrichment_constrained",
        "new_output_dir": str(output),
    }
    write_json(output / "migration_manifest.json", manifest)
    return manifest


def _write_report(
    output: Path,
    metrics_by_arm: Mapping[str, Mapping[str, float]],
    statistics: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        "# CellPhenoRank parameter-free reranking report",
        "",
        "The legacy CellPath/percentile output is preserved and was not overwritten.",
        "",
        "## Scope",
        "",
        f"- Surface baseline: {manifest.get('surface_baseline_method', 'project_bma')}",
        f"- Evaluation mode: {manifest.get('evaluation_mode', 'frozen_benchmark')}",
        f"- Gold evidence level: {manifest.get('evidence_level', 'curated_gold_models')}",
        f"- Diseases: {manifest['counts']['diseases']}",
        f"- Candidate genes: {manifest['counts']['candidate_genes']}",
        f"- Strict HPO-CPO edges: {manifest['counts']['strict_edges']}",
        f"- Extended HPO-CPO edges: {manifest['counts']['extended_edges']}",
        f"- HPOA background terms: {manifest['counts']['background_hpo']}",
        f"- Strict hypergeometric universe: {manifest['counts']['strict_enrichment_universe']}",
        f"- Extended hypergeometric universe: {manifest['counts']['extended_enrichment_universe']}",
        "",
        "## Metrics",
        "",
        "| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in metrics_by_arm.items():
        lines.append(
            f"| {arm} | {metrics['recall_at_1']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_10']:.4f} | {metrics['median_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired changes vs A0",
            "",
            "| Arm | Mean rank change | 95% CI | Wins | Ties | Losses | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in statistics:
        lines.append(
            f"| {row['experiment']} | {row['mean_rank_change']:.3f} | "
            f"[{row['mean_rank_change_ci_low']:.3f}, {row['mean_rank_change_ci_high']:.3f}] | "
            f"{row['wins']} | {row['ties']} | {row['losses']} | {row['wilcoxon_p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The final rank is produced by deterministic graph constraints and a lexicographic CPO key. "
            "`final_rank_score` is display-only and is not a disease probability.",
            "",
            "Practical and Bridge-LOO results must be reported separately. No benchmark result may be used "
            "to change the declared priority-key order.",
        ]
    )
    if manifest.get("evidence_level") == "source_hpo_mgi_unreviewed":
        lines.extend(
            [
                "",
                "This is a broad source-label evaluation, not a curated frozen gold benchmark. "
                "HPO disease-gene labels and MGI disease-model associations were fixed before CPO "
                "evaluation but were not reviewed disease by disease.",
            ]
        )
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parameter_free_pipeline(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_summary = validate_inputs(config)
    freeze = verify_freeze(config)
    if not freeze.get("accepted"):
        raise ValueError(
            "Parameter-free benchmark requires an independently fixed evaluation cohort"
        )
    for name in ("hpo_obo", "hpo_background"):
        path = Path(config["inputs"][name])
        if not path.is_file():
            raise FileNotFoundError(path)
        input_summary.append({"input": name, **input_record(path), "status": "valid"})
    surface_method = str(
        config.get("surface_baseline", {}).get("method", "project_bma")
    )
    if surface_method == "exomiser_2602_current":
        presence_path = Path(
            str(config["inputs"].get("phenodigm_mpo_presence", ""))
        )
        if not presence_path.is_file():
            raise FileNotFoundError(presence_path)
        input_summary.append(
            {
                "input": "phenodigm_mpo_presence",
                **input_record(presence_path),
                "status": "valid",
            }
        )
    config_path = Path(str(config.get("_config_path", "")))
    if config_path.is_file():
        input_summary.append(
            {"input": "config", **input_record(config_path), "status": "valid"}
        )
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    migration = _legacy_manifest(config, output)

    inputs = config["inputs"]
    diseases_original = load_diseases(inputs["diseases"])
    ontology = HpoOntology(inputs["hpo_obo"])
    diseases, reduction_rows = reduce_hpo_profiles(diseases_original, ontology)
    if any(not terms for terms in diseases.values()):
        empty = sorted(disease_id for disease_id, terms in diseases.items() if not terms)
        raise ValueError(f"HPO reduction removed every term for: {empty}")
    write_tsv(
        output / "hpo/hpo_reduction.tsv", reduction_rows, HPO_REDUCTION_FIELDS
    )
    write_tsv(
        output / "hpo/nonredundant_disease_hpo.tsv",
        disease_terms_as_rows(diseases),
        NONREDUNDANT_HPO_FIELDS,
    )

    all_models = load_models(inputs["models"])
    attribution_policy = str(
        config.get("candidate_models", {}).get(
            "attribution_policy", "single_gene_only"
        )
    )
    if attribution_policy != "single_gene_only":
        raise ValueError(
            "Parameter-free primary analysis requires candidate_models.attribution_policy=single_gene_only"
        )
    excluded_multigene_models = {
        model_id: model
        for model_id, model in all_models.items()
        if ";" in model.human_gene_symbol
    }
    models = {
        model_id: model
        for model_id, model in all_models.items()
        if model_id not in excluded_multigene_models
    }
    write_tsv(
        output / "qc/excluded_multigene_models.tsv",
        [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "mouse_gene_symbol": model.mouse_gene_symbol,
                "human_gene_symbol": model.human_gene_symbol,
                "reason": "multi_gene_genotype_not_attributable_to_one_candidate_gene",
            }
            for model in sorted(
                excluded_multigene_models.values(), key=lambda item: item.model_id
            )
        ],
        [
            "model_id",
            "model_name",
            "mouse_gene_symbol",
            "human_gene_symbol",
            "reason",
        ],
    )
    phenotypes = load_model_phenotypes(inputs["model_phenotypes"], set(models))
    hpo_ids = {term.hpo_id for terms in diseases.values() for term in terms}
    mpo_ids = {
        term.mpo_id
        for terms in phenotypes.values()
        for term in terms
        if term.status == "abnormal"
    }
    if surface_method == "exomiser_2602_current":
        provider = LegacyPhenodigmMatrix(
            inputs["similarity"], inputs["phenodigm_mpo_presence"], mpo_ids
        )
        base_rows, loo_rows, constraints, stability_summary = (
            calculate_exomiser_phenotype_stability(
                diseases,
                models,
                phenotypes,
                provider,
            )
        )
        similarity_pair_count = provider.candidate_pairs
    elif surface_method in {"project_bma", "embedding_bma"}:
        provider = build_similarity_provider(
            inputs["similarity"], hpo_ids, mpo_ids, config.get("similarity", {})
        )
        base_rows, loo_rows, constraints, stability_summary = calculate_phenotype_stability(
            diseases,
            models,
            phenotypes,
            provider,
            config.get("model_scoring", {}),
        )
        similarity_pair_count = provider.pair_count
    else:
        raise ValueError(f"Unsupported surface_baseline.method: {surface_method}")
    write_tsv(
        output / "baseline/base_gene_rankings.tsv",
        base_rows,
        BASE_PARAMETER_FREE_FIELDS,
    )
    write_tsv(
        output / "stability/leave_one_hpo_out_rankings.tsv", loo_rows, LOO_FIELDS
    )
    full_rank_rows = [
        {
            "disease_id": row["disease_id"],
            "removed_hpo_id": "full",
            "gene_symbol": row["gene_symbol"],
            "phenotype_score": row["base_score"],
            "dense_rank": row["base_dense_rank"],
            "best_model_id": row["best_model_id"],
            "valid_run": True,
        }
        for row in base_rows
    ]
    write_tsv(
        output / "stability/phenotype_rank_matrix.tsv",
        [*full_rank_rows, *loo_rows],
        LOO_FIELDS,
    )
    write_tsv(
        output / "stability/stable_phenotype_constraints.tsv",
        constraints,
        CONSTRAINT_FIELDS,
    )
    write_tsv(
        output / "stability/stable_order_graph_summary.tsv",
        stability_summary,
        STABILITY_SUMMARY_FIELDS,
    )

    candidate_genes = {str(row["gene_symbol"]) for row in base_rows}
    hpoa_background = load_hpoa_background(inputs["hpo_background"])
    strict_graph, extended_graph, graph_qc = build_binary_graphs(
        inputs["hpo_cpo"],
        inputs["go_functions"],
        inputs["cell_markers"],
        candidate_genes,
        ontology.valid_ids,
        hpoa_background,
        config.get("cpo_graph", {}),
    )
    write_tsv(
        output / "cpo_graph/strict_hpo_cpo_edges.tsv",
        graph_edge_rows(strict_graph),
        GRAPH_EDGE_FIELDS,
    )
    write_tsv(
        output / "cpo_graph/extended_hpo_cpo_edges.tsv",
        graph_edge_rows(extended_graph),
        GRAPH_EDGE_FIELDS,
    )
    write_tsv(output / "cpo_graph/graph_qc_summary.tsv", graph_qc, ["metric", "value"])
    write_tsv(
        output / "cpo_graph/cpo_function_gene_edges.tsv",
        [
            {"go_id": go_id, "gene_symbol": gene, "function_set_scope": "go_global"}
            for go_id, genes in sorted(strict_graph.function_genes.items())
            for gene in sorted(genes)
        ],
        ["go_id", "gene_symbol", "function_set_scope"],
    )
    write_tsv(
        output / "cpo_graph/cpo_cell_marker_edges.tsv",
        [
            {"cl_id": cell_id, "gene_symbol": gene}
            for cell_id, gene in sorted(strict_graph.marker_memberships)
        ],
        ["cl_id", "gene_symbol"],
    )
    background = hpoa_background.intersection(ontology.valid_ids)
    base_by_key = {
        (str(row["disease_id"]), str(row["gene_symbol"])): row for row in base_rows
    }
    strict_enrichment, strict_paths = score_graph(
        diseases,
        candidate_genes,
        strict_graph,
        background,
        base_by_key,
        emit_paths=True,
    )
    extended_enrichment, extended_paths = score_graph(
        diseases,
        candidate_genes,
        extended_graph,
        background,
        base_by_key,
        emit_paths=True,
    )
    write_tsv(
        output / "cpo_enrichment/gene_cpo_enrichment_strict.tsv",
        strict_enrichment,
        ENRICHMENT_FIELDS,
    )
    write_tsv(
        output / "cpo_enrichment/gene_cpo_enrichment_extended.tsv",
        extended_enrichment,
        ENRICHMENT_FIELDS,
    )
    write_tsv(
        output / "cpo_enrichment/raw_evidence_paths.tsv",
        strict_paths,
        RAW_PATH_FIELDS,
    )
    write_tsv(
        output / "cpo_enrichment/raw_evidence_paths_extended.tsv",
        extended_paths,
        RAW_PATH_FIELDS,
    )
    strict_reachability = global_gene_hpo_reachability(
        strict_graph, candidate_genes, background
    )
    write_tsv(
        output / "cpo_enrichment/gene_hpo_reachability.tsv",
        [
            {"gene_symbol": gene, "hpo_id": hpo_id, "graph_version": strict_graph.version}
            for gene, hpo_ids_for_gene in sorted(strict_reachability.items())
            for hpo_id in sorted(hpo_ids_for_gene)
        ],
        ["gene_symbol", "hpo_id", "graph_version"],
    )

    gold = load_gold_genes(inputs.get("gold_standard") or inputs["gold_models"])
    bridge_rows = []
    bridge_qc = []
    bridge_cache: dict[tuple[str, ...], BinaryCpoGraph] = {}
    for disease_id, terms in sorted(diseases.items()):
        mask_key = tuple(sorted(gold.get(disease_id, set())))
        if mask_key not in bridge_cache:
            bridge_cache[mask_key] = strict_graph.bridge_loo(set(mask_key))
        graph = bridge_cache[mask_key]
        scored, _ = score_graph(
            {disease_id: terms},
            candidate_genes,
            graph,
            background,
            base_by_key,
            emit_paths=False,
        )
        bridge_rows.extend(scored)
        bridge_qc.append(
            {
                "disease_id": disease_id,
                "masked_genes": mask_key,
                "strict_edge_count": len(strict_graph.edges),
                "bridge_loo_edge_count": len(graph.edges),
                "removed_edge_count": len(strict_graph.edges) - len(graph.edges),
            }
        )
    write_tsv(
        output / "bridge_loo/bridge_loo_enrichment.tsv",
        bridge_rows,
        ENRICHMENT_FIELDS,
    )
    write_tsv(
        output / "bridge_loo/bridge_loo_graph_qc.tsv",
        bridge_qc,
        [
            "disease_id",
            "masked_genes",
            "strict_edge_count",
            "bridge_loo_edge_count",
            "removed_edge_count",
        ],
    )

    arms: dict[str, list[dict[str, Any]]] = {}
    arms["A0_surface"] = rerank_unconstrained(
        strict_enrichment, "base", "A0_surface"
    )
    arms["A1_raw_path"] = rerank_unconstrained(
        strict_enrichment, "raw_path", "A1_raw_path"
    )
    arms["A2_enrichment"] = rerank_unconstrained(
        strict_enrichment, "enrichment", "A2_enrichment"
    )
    arms["A3_strict_constrained"], strict_order_summary = rerank_constrained(
        strict_enrichment, constraints, "A3_strict_constrained"
    )
    arms["A4_extended_constrained"], extended_order_summary = rerank_constrained(
        extended_enrichment, constraints, "A4_extended_constrained"
    )
    arms["A5_bridge_loo"], bridge_order_summary = rerank_constrained(
        bridge_rows, constraints, "A5_bridge_loo"
    )
    write_tsv(
        output / "reranking/final_constrained_rankings.tsv",
        arms["A3_strict_constrained"],
        FINAL_PARAMETER_FREE_FIELDS,
    )
    write_tsv(
        output / "reranking/extended_constrained_rankings.tsv",
        arms["A4_extended_constrained"],
        FINAL_PARAMETER_FREE_FIELDS,
    )
    write_tsv(
        output / "bridge_loo/bridge_loo_rankings.tsv",
        arms["A5_bridge_loo"],
        FINAL_PARAMETER_FREE_FIELDS,
    )
    write_tsv(
        output / "reranking/stable_order_validation.tsv",
        [*strict_order_summary, *extended_order_summary, *bridge_order_summary],
        [
            "disease_id",
            "candidate_gene_count",
            "stable_constraint_count",
            "constraint_violations",
            "reranking_mode",
        ],
    )
    strict_index = {
        (row["disease_id"], row["gene_symbol"]): row
        for row in arms["A3_strict_constrained"]
    }
    extended_index = {
        (row["disease_id"], row["gene_symbol"]): row
        for row in arms["A4_extended_constrained"]
    }
    stability_rows = []
    for key, strict_row in sorted(strict_index.items()):
        extended_row = extended_index[key]
        strict_evidence = bool(strict_row["has_cpo_evidence"])
        extended_evidence = bool(extended_row["has_cpo_evidence"])
        if not strict_evidence and not extended_evidence:
            status = "no_cpo_support"
        elif strict_evidence and not extended_evidence:
            status = "strict_only"
        elif extended_evidence and not strict_evidence:
            status = "extended_only"
        elif strict_row["final_rank"] == extended_row["final_rank"]:
            status = "stable_high"
        else:
            status = "unstable"
        stability_rows.append(
            {
                "disease_id": key[0],
                "gene_symbol": key[1],
                "strict_rank": strict_row["final_rank"],
                "extended_rank": extended_row["final_rank"],
                "strict_pvalue": strict_row["cpo_enrichment_pvalue"],
                "extended_pvalue": extended_row["cpo_enrichment_pvalue"],
                "rank_difference": int(strict_row["final_rank"]) - int(extended_row["final_rank"]),
                "graph_stability_status": status,
            }
        )
    write_tsv(
        output / "reranking/strict_vs_extended_stability.tsv",
        stability_rows,
        [
            "disease_id",
            "gene_symbol",
            "strict_rank",
            "extended_rank",
            "strict_pvalue",
            "extended_pvalue",
            "rank_difference",
            "graph_stability_status",
        ],
    )

    metric_rows = []
    all_disease_metrics = []
    disease_metrics_by_arm = {}
    metrics_by_arm = {}
    for arm, rankings in arms.items():
        disease_metrics, metrics = evaluate_rankings(
            rankings, gold, "final_rank", arm
        )
        disease_metrics_by_arm[arm] = disease_metrics
        metrics_by_arm[arm] = metrics
        all_disease_metrics.extend(disease_metrics)
        metric_rows.extend(metrics_as_rows(arm, metrics))
    statistics = []
    rank_changes = []
    iterations = int(config.get("benchmark", {}).get("bootstrap_iterations", 1000))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for arm in arms:
        if arm == "A0_surface":
            continue
        changes, stats = compare_disease_ranks(
            disease_metrics_by_arm["A0_surface"],
            disease_metrics_by_arm[arm],
            arm,
            iterations,
            seed,
        )
        rank_changes.extend(changes)
        statistics.append(stats)
    write_tsv(
        output / "benchmark/benchmark_metrics.tsv",
        metric_rows,
        BENCHMARK_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/ablation_metrics.tsv",
        metric_rows,
        BENCHMARK_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/disease_level_metrics.tsv",
        all_disease_metrics,
        DISEASE_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/statistical_tests.tsv",
        statistics,
        STATISTICAL_TEST_FIELDS,
    )
    write_tsv(
        output / "benchmark/rank_changes.tsv", rank_changes, RANK_CHANGE_FIELDS
    )
    strict_first = {}
    for row in strict_enrichment:
        strict_first.setdefault(str(row["disease_id"]), row)
    evaluable = {
        disease_id
        for disease_id, row in strict_first.items()
        if int(row["cpo_mappable_hpo_count"]) >= 2
    }
    overall_stratum = (
        "all_source_label_evaluable"
        if freeze.get("manifest_type") == "source_label_snapshot"
        else "all_frozen"
    )
    strata = {overall_stratum: set(gold), "strict_cpo_evaluable_ge2_hpo": evaluable}
    stratified_rows = []
    for stratum, disease_ids in strata.items():
        stratum_gold = {
            disease_id: gold[disease_id]
            for disease_id in disease_ids
            if disease_id in gold
        }
        for arm, rankings in arms.items():
            _, metrics = evaluate_rankings(rankings, stratum_gold, "final_rank", arm)
            for metric, score in metrics.items():
                if metric == "evaluated_diseases":
                    continue
                stratified_rows.append(
                    {
                        "stratum": stratum,
                        "experiment": arm,
                        "disease_count": len(stratum_gold),
                        "metric": metric,
                        "value": score,
                    }
                )
    write_tsv(
        output / "benchmark/stratified_metrics.tsv",
        stratified_rows,
        ["stratum", "experiment", "disease_count", "metric", "value"],
    )
    write_tsv(
        output / "qc/input_summary.tsv",
        input_summary,
        ["input", "path", "bytes", "sha256", "status"],
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "cpo_enrichment_constrained",
        "surface_baseline_method": surface_method,
        "evaluation_mode": freeze.get("evaluation_mode"),
        "evidence_level": freeze.get("evidence_level"),
        "parameter_free_definition": (
            "No fitted or manually weighted CPO score; deterministic binary graph, "
            "hypergeometric enrichment, and phenotype-stability constraints."
        ),
        "freeze": freeze,
        "migration_manifest_sha256": sha256(output / "migration_manifest.json"),
        "counts": {
            "diseases": len(diseases),
            "candidate_genes": len(candidate_genes),
            "models": len(models),
            "excluded_multigene_models": len(excluded_multigene_models),
            "model_phenotype_rows": sum(len(rows) for rows in phenotypes.values()),
            "similarity_pairs": similarity_pair_count,
            "original_hpo_rows": sum(len(rows) for rows in diseases_original.values()),
            "nonredundant_hpo_rows": sum(len(rows) for rows in diseases.values()),
            "background_hpo": len(background),
            "strict_enrichment_universe": (
                int(strict_enrichment[0]["background_hpo_count"])
                if strict_enrichment
                else 0
            ),
            "extended_enrichment_universe": (
                int(extended_enrichment[0]["background_hpo_count"])
                if extended_enrichment
                else 0
            ),
            "strict_edges": len(strict_graph.edges),
            "extended_edges": len(extended_graph.edges),
            "stable_constraints": len(constraints),
            "strict_raw_evidence_paths": len(strict_paths),
            "extended_raw_evidence_paths": len(extended_paths),
            "strict_cpo_evaluable_diseases": len(evaluable),
        },
        "metrics": metrics_by_arm,
        "inputs": {
            str(row["input"]): {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in input_summary
        },
        "ignored_legacy_parameters": [
            "function_weight",
            "marker_weight",
            "interaction_weight",
            "lambda",
            "coverage_mode",
        ],
        "warnings": [
            "Hypergeometric P values are graph-enrichment statistics, not disease probabilities.",
            "A5 Bridge-LOO retains independent GO function and marker memberships.",
            "The current candidate pool is frozen and limited; it is not genome-wide.",
        ],
        "legacy": migration,
    }
    write_json(output / "run_manifest.json", manifest)
    _write_report(output, metrics_by_arm, statistics, manifest)
    return manifest
