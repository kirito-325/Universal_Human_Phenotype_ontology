from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
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
    sha256,
    value,
    write_json,
    write_tsv,
)
from .pipeline import verify_freeze


OFFICIAL_FIELDS = (
    "disease_id",
    "disease_term",
    "model_id",
    "model_source",
    "model_description",
    "marker_id",
    "marker_symbol",
    "disease_model_avg_raw",
    "disease_model_avg_norm",
    "disease_model_max_raw",
    "disease_model_max_norm",
    "association_curated",
)

MODEL_SCORE_FIELDS = [
    "disease_id",
    "disease_name",
    "model_id",
    "model_name",
    "gene_symbol",
    "model_source",
    "avg_percentage_score",
    "max_percentage_score",
    "phenodigm_score",
    "association_curated",
    "official_record_found",
]

GENE_RANKING_FIELDS = [
    "disease_id",
    "disease_name",
    "gene_symbol",
    "phenodigm_score",
    "phenodigm_rank",
    "best_model_id",
    "best_model_name",
    "avg_percentage_score",
    "max_percentage_score",
    "official_model_score_available",
    "candidate_model_count",
    "scored_model_count",
]

COVERAGE_FIELDS = [
    "disease_id",
    "disease_name",
    "official_records_returned",
    "candidate_model_count",
    "candidate_models_scored",
    "candidate_model_coverage",
    "candidate_models_positive",
    "candidate_model_positive_coverage",
    "candidate_gene_count",
    "candidate_genes_scored",
    "candidate_gene_coverage",
    "candidate_genes_positive",
    "candidate_gene_positive_coverage",
    "gold_genes",
    "gold_genes_scored",
    "gold_genes_positive",
    "gold_models",
    "gold_models_scored",
    "gold_models_positive",
]


def combined_percentage_score(avg_percentage: float, max_percentage: float) -> float:
    """Return the original PhenoDigm combined percentage score."""
    return (float(avg_percentage) + float(max_percentage)) / 2.0


def _fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CellPhenoRank/0.1 official-PhenoDigm benchmark"},
    )
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("IMPC PhenoDigm response is not a JSON object")
            return payload
        except Exception as error:  # pragma: no cover - exercised only on network failure
            last_error = error
            if attempt + 1 < max(1, retries):
                time.sleep(min(4.0, 1.0 + attempt))
    raise RuntimeError(f"Failed to fetch IMPC PhenoDigm response: {last_error}")


def _official_url(endpoint: str, disease_id: str, rows: int) -> str:
    query = f'type:disease_model_summary AND disease_id:"{disease_id}"'
    parameters = urllib.parse.urlencode(
        {
            "q": query,
            "rows": rows,
            "wt": "json",
            "fl": ",".join(OFFICIAL_FIELDS),
        }
    )
    return f"{endpoint.rstrip('?')}?{parameters}"


def fetch_official_disease_scores(
    endpoint: str,
    disease_id: str,
    cache_path: Path,
    rows: int = 100_000,
    timeout: int = 60,
    retries: int = 3,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], int, str]:
    url = _official_url(endpoint, disease_id, rows)
    if cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = _fetch_json(url, timeout, retries)
        write_json(cache_path, payload)
    response = payload.get("response", {})
    docs = response.get("docs", [])
    num_found = int(response.get("numFound", len(docs)))
    if num_found > rows or len(docs) != num_found:
        raise ValueError(
            f"Incomplete IMPC response for {disease_id}: "
            f"returned {len(docs)} of {num_found}; increase phenodigm.rows"
        )
    return [dict(row) for row in docs], num_found, url


def _validate_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = config.get("inputs", {})
    required = {
        "diseases": [{"disease_id", "hpo_id"}, {"disease_id", "hp_id"}],
        "models": [{"model_id", "human_gene_symbol"}],
        "model_phenotypes": [{"model_id", "mpo_id"}, {"model_id", "mp_id"}],
        "gold_standard": [
            {"disease_id", "gene_symbol"},
            {"disease_id", "human_gene_symbol"},
        ],
        "current_baseline": [{"disease_id", "gene_symbol", "base_rank"}],
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


def _gold_models(path: str | Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(path):
        disease_id = value(row, "disease_id")
        model_id = value(row, "model_id")
        evaluation_flag = value(row, "is_evaluation_positive")
        if disease_id and model_id and (not evaluation_flag or as_bool(evaluation_flag)):
            result[disease_id].add(model_id)
    return dict(result)


def rank_official_scores(
    disease_id: str,
    disease_name: str,
    candidate_models: Mapping[str, Any],
    official_docs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restrict official model scores to the frozen candidate models and rank genes."""
    best_official: dict[str, dict[str, Any]] = {}
    for doc in official_docs:
        model_id = str(doc.get("model_id", "")).strip()
        if model_id not in candidate_models:
            continue
        avg_score = as_float(doc.get("disease_model_avg_norm"), 0.0)
        max_score = as_float(doc.get("disease_model_max_norm"), 0.0)
        combined = combined_percentage_score(avg_score, max_score)
        row = {
            "avg_percentage_score": avg_score,
            "max_percentage_score": max_score,
            "phenodigm_score": combined,
            "model_source": str(doc.get("model_source", "")),
            "association_curated": bool(doc.get("association_curated", False)),
        }
        current = best_official.get(model_id)
        if current is None or combined > float(current["phenodigm_score"]):
            best_official[model_id] = row

    model_rows: list[dict[str, Any]] = []
    by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_id, model in sorted(candidate_models.items()):
        official = best_official.get(model_id)
        row = {
            "disease_id": disease_id,
            "disease_name": disease_name,
            "model_id": model_id,
            "model_name": model.model_name,
            "gene_symbol": model.human_gene_symbol,
            "model_source": official["model_source"] if official else "",
            "avg_percentage_score": (
                official["avg_percentage_score"] if official else 0.0
            ),
            "max_percentage_score": (
                official["max_percentage_score"] if official else 0.0
            ),
            "phenodigm_score": official["phenodigm_score"] if official else 0.0,
            "association_curated": (
                official["association_curated"] if official else False
            ),
            "official_record_found": official is not None,
        }
        model_rows.append(row)
        by_gene[model.human_gene_symbol].append(row)

    gene_rows = []
    for gene, rows in sorted(by_gene.items()):
        ordered = sorted(
            rows,
            key=lambda row: (-float(row["phenodigm_score"]), str(row["model_id"])),
        )
        best = ordered[0]
        scored = sum(bool(row["official_record_found"]) for row in rows)
        gene_rows.append(
            {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "gene_symbol": gene,
                "phenodigm_score": best["phenodigm_score"],
                "best_model_id": best["model_id"],
                "best_model_name": best["model_name"],
                "avg_percentage_score": best["avg_percentage_score"],
                "max_percentage_score": best["max_percentage_score"],
                "official_model_score_available": scored > 0,
                "candidate_model_count": len(rows),
                "scored_model_count": scored,
            }
        )
    gene_rows.sort(key=lambda row: (-float(row["phenodigm_score"]), row["gene_symbol"]))
    for rank, row in enumerate(gene_rows, 1):
        row["phenodigm_rank"] = rank
    return model_rows, gene_rows


def _write_report(
    output: Path,
    current_metrics: Mapping[str, float],
    official_metrics: Mapping[str, float],
    comparison: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> None:
    lines = [
        "# Official IMPC PhenoDigm benchmark",
        "",
        "This run ranks the frozen candidate genes with live, official IMPC PhenoDigm "
        "model scores. Existing CellPhenoRank outputs were not overwritten.",
        "",
        "## Fixed evaluation scope",
        "",
        f"- Frozen diseases: {counts['diseases']}",
        f"- Candidate genes: {counts['candidate_genes']}",
        f"- Single-gene phenotype-bearing candidate models: {counts['candidate_models']}",
        f"- Candidate model scores found in IMPC: {counts['scored_model_pairs']} / "
        f"{counts['possible_model_pairs']}",
        f"- Positive candidate model scores: {counts['positive_model_pairs']} / "
        f"{counts['possible_model_pairs']}",
        f"- Mean candidate-gene coverage per disease: "
        f"{counts['mean_candidate_gene_coverage']:.4f}",
        f"- Disease gold genes with an official score: {counts['gold_gene_diseases_scored']} / "
        f"{counts['diseases']}",
        f"- Gold models with an official score: {counts['gold_models_scored']} / "
        f"{counts['gold_models']}",
        f"- Gold models with a positive official score: "
        f"{counts['gold_models_positive']} / {counts['gold_models']}",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Current A0 | {current_metrics['recall_at_1']:.4f} | "
        f"{current_metrics['recall_at_5']:.4f} | {current_metrics['recall_at_10']:.4f} | "
        f"{current_metrics['mrr']:.4f} | {current_metrics['ndcg_at_10']:.4f} | "
        f"{current_metrics['median_rank']:.2f} |",
        f"| Official PhenoDigm | {official_metrics['recall_at_1']:.4f} | "
        f"{official_metrics['recall_at_5']:.4f} | {official_metrics['recall_at_10']:.4f} | "
        f"{official_metrics['mrr']:.4f} | {official_metrics['ndcg_at_10']:.4f} | "
        f"{official_metrics['median_rank']:.2f} |",
        "",
        "## Paired comparison",
        "",
        f"- Mean gold-gene rank change (current A0 minus PhenoDigm): "
        f"{comparison['mean_rank_change']:.3f}",
        f"- Wins / ties / losses: {comparison['wins']} / {comparison['ties']} / "
        f"{comparison['losses']}",
        f"- Two-sided paired Wilcoxon p: {comparison['wilcoxon_p_value']:.6g}",
        "",
        "## Interpretation boundary",
        "",
        "The candidate genes, candidate models and gold standard are frozen locally, but the "
        "official score uses the current IMPC disease-annotation and ontology snapshot. This "
        "therefore measures official PhenoDigm performance on the local evaluation universe; "
        "it is not a controlled same-HPO algorithm ablation.",
    ]
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_official_phenodigm_benchmark(
    config: Mapping[str, Any], output_dir: str | Path, refresh: bool = False
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_summary = _validate_inputs(config)
    freeze = verify_freeze(config)
    if not freeze.get("accepted"):
        raise ValueError("Official PhenoDigm benchmark requires the frozen cohort")
    write_tsv(
        output / "qc/input_summary.tsv",
        input_summary,
        ["input", "path", "bytes", "sha256", "status"],
    )
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )

    inputs = config["inputs"]
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
    if not diseases or not models:
        raise ValueError("No diseases or phenotype-bearing single-gene models are available")

    settings = config.get("phenodigm", {})
    endpoint = str(
        settings.get(
            "endpoint",
            "https://www.ebi.ac.uk/mi/impc/solr/phenodigm/select",
        )
    )
    rows = int(settings.get("rows", 100_000))
    timeout = int(settings.get("timeout_seconds", 60))
    retries = int(settings.get("retries", 3))

    gold_genes = load_gold_genes(inputs["gold_standard"])
    gold_models = _gold_models(inputs["gold_standard"])
    all_model_rows: list[dict[str, Any]] = []
    all_gene_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    queries = []
    raw_files = []
    raw_dir = output / "official/raw"
    for disease_id, terms in sorted(diseases.items()):
        cache_path = raw_dir / f"{disease_id.replace(':', '_')}.json"
        docs, num_found, url = fetch_official_disease_scores(
            endpoint,
            disease_id,
            cache_path,
            rows=rows,
            timeout=timeout,
            retries=retries,
            refresh=refresh,
        )
        model_rows, gene_rows = rank_official_scores(
            disease_id, terms[0].disease_name, models, docs
        )
        all_model_rows.extend(model_rows)
        all_gene_rows.extend(gene_rows)
        found_models = {
            str(row["model_id"])
            for row in model_rows
            if row["official_record_found"]
        }
        positive_models = {
            str(row["model_id"])
            for row in model_rows
            if float(row["phenodigm_score"]) > 0
        }
        scored_genes = {
            str(row["gene_symbol"])
            for row in gene_rows
            if row["official_model_score_available"]
        }
        positive_genes = {
            str(row["gene_symbol"])
            for row in gene_rows
            if float(row["phenodigm_score"]) > 0
        }
        disease_gold_genes = gold_genes.get(disease_id, set())
        disease_gold_models = gold_models.get(disease_id, set())
        coverage_rows.append(
            {
                "disease_id": disease_id,
                "disease_name": terms[0].disease_name,
                "official_records_returned": num_found,
                "candidate_model_count": len(models),
                "candidate_models_scored": len(found_models),
                "candidate_model_coverage": len(found_models) / len(models),
                "candidate_models_positive": len(positive_models),
                "candidate_model_positive_coverage": len(positive_models)
                / len(models),
                "candidate_gene_count": len({m.human_gene_symbol for m in models.values()}),
                "candidate_genes_scored": len(scored_genes),
                "candidate_gene_coverage": len(scored_genes)
                / len({m.human_gene_symbol for m in models.values()}),
                "candidate_genes_positive": len(positive_genes),
                "candidate_gene_positive_coverage": len(positive_genes)
                / len({m.human_gene_symbol for m in models.values()}),
                "gold_genes": sorted(disease_gold_genes),
                "gold_genes_scored": sorted(disease_gold_genes.intersection(scored_genes)),
                "gold_genes_positive": sorted(
                    disease_gold_genes.intersection(positive_genes)
                ),
                "gold_models": sorted(disease_gold_models),
                "gold_models_scored": sorted(disease_gold_models.intersection(found_models)),
                "gold_models_positive": sorted(
                    disease_gold_models.intersection(positive_models)
                ),
            }
        )
        queries.append({"disease_id": disease_id, "url": url, "num_found": num_found})
        raw_files.append(
            {
                "path": str(cache_path.relative_to(output)),
                "bytes": cache_path.stat().st_size,
                "sha256": sha256(cache_path),
            }
        )

    write_tsv(
        output / "official/phenodigm_model_scores.tsv",
        all_model_rows,
        MODEL_SCORE_FIELDS,
    )
    write_tsv(
        output / "official/phenodigm_gene_rankings.tsv",
        all_gene_rows,
        GENE_RANKING_FIELDS,
    )
    write_tsv(output / "qc/coverage_by_disease.tsv", coverage_rows, COVERAGE_FIELDS)

    current_rows = list(read_tsv(inputs["current_baseline"]))
    current_disease, current_metrics = evaluate_rankings(
        current_rows, gold_genes, "base_rank", "current_A0"
    )
    official_disease, official_metrics = evaluate_rankings(
        all_gene_rows, gold_genes, "phenodigm_rank", "official_PhenoDigm"
    )
    benchmark_rows = [
        *metrics_as_rows("current_A0", current_metrics),
        *metrics_as_rows("official_PhenoDigm", official_metrics),
    ]
    write_tsv(
        output / "benchmark/benchmark_metrics.tsv",
        benchmark_rows,
        BENCHMARK_METRIC_FIELDS,
    )
    write_tsv(
        output / "benchmark/disease_metrics.tsv",
        [*current_disease, *official_disease],
        DISEASE_METRIC_FIELDS,
    )
    changes, comparison = compare_disease_ranks(
        current_disease,
        official_disease,
        "official_PhenoDigm_vs_current_A0",
        int(config.get("benchmark", {}).get("bootstrap_iterations", 10_000)),
        int(config.get("project", {}).get("random_seed", 2026)),
    )
    write_tsv(
        output / "benchmark/rank_changes_vs_current_a0.tsv",
        changes,
        RANK_CHANGE_FIELDS,
    )
    write_tsv(
        output / "benchmark/statistical_tests.tsv",
        [comparison],
        STATISTICAL_TEST_FIELDS,
    )

    candidate_genes = {model.human_gene_symbol for model in models.values()}
    scored_pairs = sum(bool(row["official_record_found"]) for row in all_model_rows)
    positive_pairs = sum(float(row["phenodigm_score"]) > 0 for row in all_model_rows)
    gold_model_total = sum(len(items) for items in gold_models.values())
    gold_model_scored = sum(
        len(row["gold_models_scored"])
        for row in coverage_rows
    )
    gold_model_positive = sum(
        len(row["gold_models_positive"]) for row in coverage_rows
    )
    counts = {
        "diseases": len(diseases),
        "candidate_genes": len(candidate_genes),
        "candidate_models": len(models),
        "possible_model_pairs": len(diseases) * len(models),
        "scored_model_pairs": scored_pairs,
        "positive_model_pairs": positive_pairs,
        "official_records_returned": sum(
            int(row["official_records_returned"]) for row in coverage_rows
        ),
        "mean_candidate_model_coverage": sum(
            float(row["candidate_model_coverage"]) for row in coverage_rows
        )
        / len(coverage_rows),
        "mean_candidate_gene_coverage": sum(
            float(row["candidate_gene_coverage"]) for row in coverage_rows
        )
        / len(coverage_rows),
        "mean_candidate_gene_positive_coverage": sum(
            float(row["candidate_gene_positive_coverage"])
            for row in coverage_rows
        )
        / len(coverage_rows),
        "gold_gene_diseases_scored": sum(
            bool(row["gold_genes_scored"]) for row in coverage_rows
        ),
        "gold_models": gold_model_total,
        "gold_models_scored": gold_model_scored,
        "gold_models_positive": gold_model_positive,
    }
    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at": fetched_at,
        "method": "official_IMPC_PhenoDigm_precomputed_scores",
        "formula": "combinedPercentageScore=(disease_model_avg_norm+disease_model_max_norm)/2",
        "gene_aggregation": "maximum combinedPercentageScore over frozen candidate models for each gene",
        "candidate_policy": "phenotype-bearing single-gene frozen models only",
        "disease_profile_scope": "live IMPC disease annotations, not local frozen HPO profile",
        "endpoint": endpoint,
        "freeze": freeze,
        "counts": counts,
        "inputs": {row["input"]: row for row in input_summary},
        "queries": queries,
        "raw_response_files": raw_files,
        "metrics": {
            "current_A0": current_metrics,
            "official_PhenoDigm": official_metrics,
        },
        "paired_comparison": comparison,
        "limitations": [
            "The live IMPC disease HPO and ontology snapshot may differ from the frozen local profile.",
            "Official zero-score or absent model-disease rows are represented as score zero.",
            "This run cannot generate local leave-one-HPO-out scores from aggregate IMPC output.",
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    _write_report(output, current_metrics, official_metrics, comparison, counts)
    return manifest
