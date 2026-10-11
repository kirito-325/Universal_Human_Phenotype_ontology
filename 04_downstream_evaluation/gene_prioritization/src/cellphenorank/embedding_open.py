from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .baseline import HPO_MPO_EVIDENCE_FIELDS, MODEL_SCORE_FIELDS, run_baseline
from .binary_cpo import (
    GRAPH_EDGE_FIELDS,
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
    evaluate_open_rankings,
    load_gold_genes,
    metrics_as_rows,
)
from .hpo_redundancy import HPO_REDUCTION_FIELDS, HpoOntology, reduce_hpo_profiles
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
    RAW_PATH_FIELDS,
    rerank_constrained,
    rerank_unconstrained,
    score_graph,
)
from .phenotype_stability import (
    CONSTRAINT_FIELDS,
    LOO_FIELDS,
    STABILITY_SUMMARY_FIELDS,
    calculate_phenotype_stability,
)
from .similarity import EmbeddingSimilarityProvider


OPEN_RANKING_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "embedding_score",
    "rank",
    "base_rank",
    "rank_change",
    "best_model_id",
    "best_hpo_id",
    "best_mpo_id",
    "hpo_mpo_coverage",
    "returned_by_embedding",
    "in_rerank_shortlist",
    "reranking_mode",
    "cpo_graph",
    "cpo_enrichment_pvalue",
    "cpo_enrichment_fdr",
    "supported_hpo_count",
    "supported_cpo_count",
    "supported_cell_count",
    "supported_function_count",
    "raw_function_path_count",
    "concordant_cpo_count",
    "has_cpo_evidence",
]


def _verify_open_manifest(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = config["inputs"]
    manifest_path = Path(inputs["open_candidate_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("frozen")
        and manifest.get("candidate_universe_frozen")
        and manifest.get("middle_layer_independent")
    ):
        raise ValueError("Open candidate universe is not independently frozen")
    configured = {
        "diseases.tsv": inputs["diseases"],
        "models.tsv": inputs["models"],
        "model_phenotypes.tsv": inputs["model_phenotypes"],
        "gold_standard.tsv": inputs["gold_standard"],
    }
    rows = []
    for name, path in configured.items():
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not expected or sha256(path) != expected:
            raise ValueError(f"Configured {name} does not match the open candidate manifest")
        rows.append({"input": name, **input_record(path), "status": "valid"})
    for name in (
        "similarity",
        "hpo_cpo",
        "cell_markers",
        "go_functions",
        "hpo_obo",
        "hpo_background",
    ):
        path = Path(inputs[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"input": name, **input_record(path), "status": "valid"})
    embedding_manifest = Path(str(inputs["similarity"]) + ".manifest.json")
    if not embedding_manifest.is_file():
        raise FileNotFoundError(embedding_manifest)
    embedding_metadata = json.loads(embedding_manifest.read_text(encoding="utf-8"))
    expected_artifact = embedding_metadata.get("artifact", {}).get("sha256")
    if not expected_artifact or sha256(inputs["similarity"]) != expected_artifact:
        raise ValueError("Embedding artifact does not match its manifest")
    configured_representation = str(
        config.get("surface_baseline", {}).get("representation", "")
    )
    if configured_representation and configured_representation != str(
        embedding_metadata.get("representation", "")
    ):
        raise ValueError("Configured embedding representation does not match its manifest")
    rows.extend(
        [
            {
                "input": "embedding_manifest",
                **input_record(embedding_manifest),
                "status": "valid",
            },
            {
                "input": "open_candidate_manifest",
                **input_record(manifest_path),
                "status": "valid",
            },
        ]
    )
    return manifest, rows


def _score_open_baseline(
    diseases: Mapping[str, Sequence[Any]],
    models: Mapping[str, Any],
    phenotypes: Mapping[str, Sequence[Any]],
    provider: EmbeddingSimilarityProvider,
    model_scoring: Mapping[str, Any],
    *,
    collect_model_rows: bool = True,
    collect_evidence: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models_by_mpo: dict[str, set[str]] = defaultdict(set)
    for model_id, rows in phenotypes.items():
        for row in rows:
            if row.status == "abnormal":
                models_by_mpo[row.mpo_id].add(model_id)
    model_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        relevant_models: set[str] = set()
        for term in terms:
            for mpo_id in provider.scored_mpo_ids(term.hpo_id):
                relevant_models.update(models_by_mpo.get(mpo_id, ()))
        selected_models = {
            model_id: models[model_id]
            for model_id in relevant_models
            if model_id in models
        }
        current_models, current_genes, current_evidence = run_baseline(
            {disease_id: terms},
            selected_models,
            phenotypes,
            provider,
            model_scoring,
        )
        positive_models = [
            row for row in current_models if float(row["model_score"]) > 0.0
        ]
        positive_genes = [
            row for row in current_genes if float(row["base_score"]) > 0.0
        ]
        positive_gene_ids = {str(row["gene_symbol"]) for row in positive_genes}
        if collect_model_rows:
            model_rows.extend(positive_models)
        base_rows.extend(
            {
                "disease_id": row["disease_id"],
                "disease_name": row["disease_name"],
                "gene_symbol": row["gene_symbol"],
                "base_score": row["base_score"],
                "base_rank": row["base_rank"],
                "base_dense_rank": row["base_rank"],
                "best_model_id": row["best_model_id"],
                "best_hpo_id": row["top_hpo_id"],
                "best_mpo_id": row["top_mpo_id"],
                "hpo_mpo_coverage": row["hpo_mpo_coverage"],
            }
            for row in positive_genes
        )
        if collect_evidence:
            evidence_rows.extend(
                row
                for row in current_evidence
                if str(row["gene_symbol"]) in positive_gene_ids
            )
    return model_rows, base_rows, evidence_rows


def _compact_rankings(
    base_rows: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    shortlist_by_disease: Mapping[str, set[str]],
    experiment: str,
) -> list[dict[str, Any]]:
    top_index = {
        (str(row["disease_id"]), str(row["gene_symbol"])): row for row in top_rows
    }
    output: list[dict[str, Any]] = []
    for base in base_rows:
        disease_id = str(base["disease_id"])
        gene = str(base["gene_symbol"])
        top = top_index.get((disease_id, gene))
        rank = int(top["final_rank"]) if top else int(base["base_rank"])
        output.append(
            {
                "disease_id": disease_id,
                "disease_name": base["disease_name"],
                "gene_symbol": gene,
                "embedding_score": base["base_score"],
                "rank": rank,
                "base_rank": base["base_rank"],
                "rank_change": int(base["base_rank"]) - rank,
                "best_model_id": base["best_model_id"],
                "best_hpo_id": base["best_hpo_id"],
                "best_mpo_id": base["best_mpo_id"],
                "hpo_mpo_coverage": base["hpo_mpo_coverage"],
                "returned_by_embedding": True,
                "in_rerank_shortlist": gene in shortlist_by_disease.get(disease_id, set()),
                "reranking_mode": experiment,
                "cpo_graph": top.get("cpo_graph", "") if top else "",
                "cpo_enrichment_pvalue": top.get("cpo_enrichment_pvalue", "") if top else "",
                "cpo_enrichment_fdr": top.get("cpo_enrichment_fdr", "") if top else "",
                "supported_hpo_count": top.get("supported_hpo_count", "") if top else "",
                "supported_cpo_count": top.get("supported_cpo_count", "") if top else "",
                "supported_cell_count": top.get("supported_cell_count", "") if top else "",
                "supported_function_count": top.get("supported_function_count", "") if top else "",
                "raw_function_path_count": top.get("raw_function_path_count", "") if top else "",
                "concordant_cpo_count": top.get("concordant_cpo_count", "") if top else "",
                "has_cpo_evidence": top.get("has_cpo_evidence", False) if top else False,
            }
        )
    return sorted(output, key=lambda row: (row["disease_id"], int(row["rank"])))


def _write_report(
    output: Path,
    metrics: Mapping[str, Mapping[str, float]],
    comparisons: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        "# Open-candidate embedding benchmark",
        "",
        "All phenotype-bearing single-gene mouse models define the candidate universe. "
        "CPO reranking is limited to the embedding baseline Top-K; the tail retains its "
        "baseline order and missing gold genes remain failures.",
        "",
        "## Scope",
        "",
        f"- Candidate genes: {manifest['counts']['candidate_genes']}",
        f"- Candidate models: {manifest['counts']['candidate_models']}",
        f"- Diseases: {manifest['counts']['diseases']}",
        f"- Embedding HPO-MP pairs retained: {manifest['counts']['similarity_pairs']}",
        f"- CPO reranking shortlist: Top-{manifest['settings']['rerank_top_k']}",
        "",
        "## Metrics",
        "",
        "| Arm | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | Gold retrieved | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in metrics.items():
        lines.append(
            f"| {arm} | {values['recall_at_1']:.4f} | {values['recall_at_5']:.4f} | "
            f"{values['recall_at_10']:.4f} | {values['recall_at_50']:.4f} | "
            f"{values['mrr']:.4f} | {values['gold_gene_retrieval_rate']:.4f} | "
            f"{values['median_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired changes versus embedding surface",
            "",
            "| Arm | Mean rank change | Wins | Ties | Losses | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['experiment']} | {row['mean_rank_change']:.3f} | "
            f"{row['wins']} | {row['ties']} | {row['losses']} | "
            f"{row['wilcoxon_p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a current-snapshot open retrieval benchmark, not a literal replay of "
            "IMPC DR5 or DR20.1. SapBERT is trained for biomedical entity alignment; cosine "
            "similarity is an experimental HPO-MP signal and not a calibrated probability.",
        ]
    )
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_embedding_open_benchmark(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    open_manifest, input_summary = _verify_open_manifest(config)
    if str(config.get("similarity", {}).get("provider", "")) != "embedding":
        raise ValueError("Open embedding benchmark requires similarity.provider=embedding")
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    write_tsv(
        output / "qc/input_summary.tsv",
        input_summary,
        ["input", "path", "bytes", "sha256", "status"],
    )

    inputs = config["inputs"]
    diseases_original = load_diseases(inputs["diseases"])
    ontology = HpoOntology(inputs["hpo_obo"])
    diseases, reduction_rows = reduce_hpo_profiles(diseases_original, ontology)
    if any(not terms for terms in diseases.values()):
        empty = sorted(disease_id for disease_id, terms in diseases.items() if not terms)
        raise ValueError(f"HPO reduction removed every term for: {empty}")

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
    phenotypes = {model_id: phenotypes[model_id] for model_id in models}
    candidate_genes = {model.human_gene_symbol for model in models.values()}
    if len(candidate_genes) != int(open_manifest["candidate_gene_count"]):
        raise ValueError("Runtime candidate genes do not match the frozen open manifest")
    if len(models) != int(open_manifest["candidate_model_count"]):
        raise ValueError("Runtime candidate models do not match the frozen open manifest")

    hpo_ids = {term.hpo_id for terms in diseases.values() for term in terms}
    mpo_ids = {
        row.mpo_id
        for rows in phenotypes.values()
        for row in rows
        if row.status == "abnormal"
    }
    provider = EmbeddingSimilarityProvider(
        inputs["similarity"],
        hpo_ids,
        mpo_ids,
        config.get("similarity", {}),
    )
    detail_settings = config.get("open_benchmark", {})
    write_model_details = bool(detail_settings.get("write_model_scores", True))
    write_pair_evidence = bool(detail_settings.get("write_pair_evidence", True))
    write_cpo_paths = bool(detail_settings.get("write_cpo_paths", True))
    model_rows, base_rows, evidence_rows = _score_open_baseline(
        diseases,
        models,
        phenotypes,
        provider,
        config.get("model_scoring", {}),
        collect_model_rows=write_model_details,
        collect_evidence=write_pair_evidence,
    )
    if write_model_details:
        write_tsv(
            output / "baseline/retrieved_model_scores.tsv",
            model_rows,
            MODEL_SCORE_FIELDS,
        )
    write_tsv(
        output / "baseline/retrieved_gene_rankings.tsv",
        base_rows,
        [
            "disease_id",
            "disease_name",
            "gene_symbol",
            "base_score",
            "base_rank",
            "base_dense_rank",
            "best_model_id",
            "best_hpo_id",
            "best_mpo_id",
            "hpo_mpo_coverage",
        ],
    )
    if write_pair_evidence:
        write_tsv(
            output / "baseline/hpo_mpo_evidence.tsv",
            evidence_rows,
            HPO_MPO_EVIDENCE_FIELDS,
        )
    write_tsv(
        output / "hpo/hpo_reduction.tsv",
        reduction_rows,
        HPO_REDUCTION_FIELDS,
    )

    rerank_top_k = int(config.get("open_benchmark", {}).get("rerank_top_k", 200))
    if rerank_top_k <= 0:
        raise ValueError("open_benchmark.rerank_top_k must be positive")
    base_by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        base_by_disease[str(row["disease_id"])].append(row)
    shortlist_by_disease = {
        disease_id: {
            str(row["gene_symbol"])
            for row in sorted(rows, key=lambda item: int(item["base_rank"]))[:rerank_top_k]
        }
        for disease_id, rows in base_by_disease.items()
    }
    shortlist_union = set().union(*shortlist_by_disease.values()) if shortlist_by_disease else set()

    loo_rows: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    stability_summary: list[dict[str, Any]] = []
    for disease_id, terms in sorted(diseases.items()):
        genes = shortlist_by_disease.get(disease_id, set())
        if not genes:
            continue
        disease_models = {
            model_id: model
            for model_id, model in models.items()
            if model.human_gene_symbol in genes
        }
        _, current_loo, current_constraints, current_summary = (
            calculate_phenotype_stability(
                {disease_id: terms},
                disease_models,
                phenotypes,
                provider,
                config.get("model_scoring", {}),
            )
        )
        loo_rows.extend(current_loo)
        constraints.extend(current_constraints)
        stability_summary.extend(current_summary)
    write_tsv(
        output / "stability/leave_one_hpo_out_rankings.tsv", loo_rows, LOO_FIELDS
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

    background = load_hpoa_background(inputs["hpo_background"]).intersection(
        ontology.valid_ids
    )
    strict_graph, extended_graph, graph_qc = build_binary_graphs(
        inputs["hpo_cpo"],
        inputs["go_functions"],
        inputs["cell_markers"],
        shortlist_union,
        ontology.valid_ids,
        background,
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
    write_tsv(
        output / "cpo_graph/graph_qc_summary.tsv", graph_qc, ["metric", "value"]
    )

    base_by_key = {
        (str(row["disease_id"]), str(row["gene_symbol"])): row for row in base_rows
    }
    strict_rows: list[dict[str, Any]] = []
    extended_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    strict_paths: list[dict[str, Any]] = []
    extended_paths: list[dict[str, Any]] = []
    gold = load_gold_genes(inputs["gold_standard"])
    for disease_id, terms in sorted(diseases.items()):
        genes = shortlist_by_disease.get(disease_id, set())
        if not genes:
            continue
        current_strict, current_strict_paths = score_graph(
            {disease_id: terms},
            genes,
            strict_graph,
            background,
            base_by_key,
            emit_paths=write_cpo_paths,
        )
        current_extended, current_extended_paths = score_graph(
            {disease_id: terms},
            genes,
            extended_graph,
            background,
            base_by_key,
            emit_paths=write_cpo_paths,
        )
        current_bridge, _ = score_graph(
            {disease_id: terms},
            genes,
            strict_graph.bridge_loo(gold.get(disease_id, set())),
            background,
            base_by_key,
            emit_paths=False,
        )
        strict_rows.extend(current_strict)
        extended_rows.extend(current_extended)
        bridge_rows.extend(current_bridge)
        strict_paths.extend(current_strict_paths)
        extended_paths.extend(current_extended_paths)
    write_tsv(
        output / "cpo_enrichment/gene_cpo_enrichment_strict.tsv",
        strict_rows,
        ENRICHMENT_FIELDS,
    )
    write_tsv(
        output / "cpo_enrichment/gene_cpo_enrichment_extended.tsv",
        extended_rows,
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

    top_arms: dict[str, list[dict[str, Any]]] = {
        "E0_embedding_surface": rerank_unconstrained(
            strict_rows, "base", "E0_embedding_surface"
        ),
        "E1_raw_path": rerank_unconstrained(strict_rows, "raw_path", "E1_raw_path"),
        "E2_enrichment": rerank_unconstrained(
            strict_rows, "enrichment", "E2_enrichment"
        ),
    }
    top_arms["E3_strict_constrained"], _ = rerank_constrained(
        strict_rows, constraints, "E3_strict_constrained"
    )
    top_arms["E4_extended_constrained"], _ = rerank_constrained(
        extended_rows, constraints, "E4_extended_constrained"
    )
    top_arms["E5_bridge_loo"], _ = rerank_constrained(
        bridge_rows, constraints, "E5_bridge_loo"
    )
    arms = {
        name: _compact_rankings(base_rows, rows, shortlist_by_disease, name)
        for name, rows in top_arms.items()
    }
    for name, rows in arms.items():
        write_tsv(output / f"rankings/{name}.tsv", rows, OPEN_RANKING_FIELDS)

    metrics_by_arm: dict[str, dict[str, float]] = {}
    disease_metrics_by_arm: dict[str, list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    disease_metric_rows: list[dict[str, Any]] = []
    for name, rows in arms.items():
        disease_metrics, metrics = evaluate_open_rankings(
            rows,
            gold,
            "rank",
            name,
            len(candidate_genes),
        )
        metrics_by_arm[name] = metrics
        disease_metrics_by_arm[name] = disease_metrics
        metric_rows.extend(metrics_as_rows(name, metrics))
        disease_metric_rows.extend(disease_metrics)
    comparisons: list[dict[str, Any]] = []
    rank_changes: list[dict[str, Any]] = []
    iterations = int(config.get("benchmark", {}).get("bootstrap_iterations", 1000))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for name in arms:
        if name == "E0_embedding_surface":
            continue
        changes, statistics = compare_disease_ranks(
            disease_metrics_by_arm["E0_embedding_surface"],
            disease_metrics_by_arm[name],
            name,
            iterations,
            seed,
        )
        rank_changes.extend(changes)
        comparisons.append(statistics)
    write_tsv(
        output / "benchmark/benchmark_metrics.tsv",
        metric_rows,
        BENCHMARK_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/disease_level_metrics.tsv",
        disease_metric_rows,
        DISEASE_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/statistical_tests.tsv",
        comparisons,
        STATISTICAL_TEST_FIELDS,
    )
    write_tsv(
        output / "benchmark/rank_changes.tsv", rank_changes, RANK_CHANGE_FIELDS
    )

    returned_counts = [len(rows) for rows in base_by_disease.values()]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "open_candidate_sapbert_hpo_mpo",
        "evaluation_mode": "open_candidate_benchmark",
        "candidate_rule": open_manifest["candidate_rule"],
        "counts": {
            "diseases": len(diseases),
            "candidate_genes": len(candidate_genes),
            "candidate_models": len(models),
            "retrieved_model_rows": len(model_rows) if write_model_details else None,
            "hpo_mpo_evidence_rows": len(evidence_rows) if write_pair_evidence else None,
            "retrieved_gene_rows": len(base_rows),
            "retrieved_genes_per_disease_mean": mean(returned_counts) if returned_counts else 0.0,
            "retrieved_genes_per_disease_median": median(returned_counts) if returned_counts else 0.0,
            "similarity_pairs": provider.pair_count,
            "embedding_hpo_missing": len(provider.missing_hpo_ids),
            "embedding_mpo_missing": len(provider.missing_mpo_ids),
            "shortlist_gene_union": len(shortlist_union),
            "stable_constraints": len(constraints),
        },
        "settings": {
            "top_mpo_per_hpo": provider.top_mpo_per_hpo,
            "min_cosine": provider.min_cosine,
            "rerank_top_k": rerank_top_k,
            "write_model_scores": write_model_details,
            "write_pair_evidence": write_pair_evidence,
            "write_cpo_paths": write_cpo_paths,
            "embedding_manifest": str(Path(provider.manifest_path).resolve()),
        },
        "metrics": metrics_by_arm,
        "comparisons": comparisons,
        "inputs": {
            str(row["input"]): {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in input_summary
        },
        "warnings": [
            "Missing gold genes are counted as retrieval failures with reciprocal rank zero.",
            "CPO reranking is restricted to the embedding baseline Top-K and cannot rescue an omitted gold gene.",
            "This current MGI snapshot is broader than an IMPC-only release and is not a literal DR5/DR20.1 replay.",
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    _write_report(output, metrics_by_arm, comparisons, manifest)
    return manifest
