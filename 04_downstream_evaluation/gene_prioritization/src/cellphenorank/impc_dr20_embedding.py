from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from .config import dump_config
from .io import (
    as_bool,
    input_record,
    load_diseases,
    load_model_phenotypes,
    load_models,
    read_tsv,
    sha256,
    split_values,
    write_json,
    write_tsv,
)


FORMULAS = ("symmetric_bma", "phenodigm_equal_avg_max", "project_80_bma_20_max")


class DenseEmbeddingStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        manifest_path = Path(self.path + ".manifest.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(self.path) != self.manifest["artifact"]["sha256"]:
            raise ValueError(f"Embedding artifact hash mismatch: {self.path}")
        with np.load(self.path, allow_pickle=False) as artifact:
            hpo_ids = [str(item) for item in artifact["hpo_ids"].tolist()]
            mpo_ids = [str(item) for item in artifact["mpo_ids"].tolist()]
            self.hpo_vectors = np.asarray(artifact["hpo_vectors"], dtype=np.float32)
            self.mpo_vectors = np.asarray(artifact["mpo_vectors"], dtype=np.float32)
        self.hpo_index = {term_id: index for index, term_id in enumerate(hpo_ids)}
        self.mpo_index = {term_id: index for index, term_id in enumerate(mpo_ids)}

    def profile_components(
        self, hpo_ids: Sequence[str], mpo_ids: Sequence[str]
    ) -> tuple[float, float] | None:
        hpo_index = [self.hpo_index[item] for item in dict.fromkeys(hpo_ids) if item in self.hpo_index]
        mpo_index = [self.mpo_index[item] for item in dict.fromkeys(mpo_ids) if item in self.mpo_index]
        if not hpo_index or not mpo_index:
            return None
        scores = np.clip(
            self.hpo_vectors[hpo_index] @ self.mpo_vectors[mpo_index].T,
            0.0,
            1.0,
        )
        forward = scores.max(axis=1)
        reverse = scores.max(axis=0)
        bma = float((forward.sum() + reverse.sum()) / (len(forward) + len(reverse)))
        return bma, float(scores.max())


def aggregate_profile_score(bma: float, max_pair: float, formula: str) -> float:
    if formula == "symmetric_bma":
        return bma
    if formula == "phenodigm_equal_avg_max":
        return 0.5 * bma + 0.5 * max_pair
    if formula == "project_80_bma_20_max":
        return 0.8 * bma + 0.2 * max_pair
    raise ValueError(f"Unsupported score formula: {formula}")


def binary_ranking_metrics(labels: Sequence[bool], scores: Sequence[float]) -> dict[str, float]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels and scores must be non-empty and have equal length")
    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranks = rankdata(values, method="average")
    auc = (
        float(ranks[y].sum()) - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    order = np.argsort(-values, kind="stable")
    ordered_y = y[order]
    ordered_scores = values[order]
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores)) + 1,
        len(ordered_scores),
    ]
    cumulative = np.cumsum(ordered_y)
    true_positives = cumulative[threshold_ends - 1]
    predicted_positives = threshold_ends
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "n": float(len(y)),
        "positives": float(positives),
        "negatives": float(negatives),
        "auroc": auc,
        "auprc": average_precision,
        "positive_prevalence": positives / len(y),
    }


def bootstrap_binary_intervals(
    labels: Sequence[bool],
    scores: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=float)
    positive_index = np.flatnonzero(y)
    negative_index = np.flatnonzero(~y)
    if not len(positive_index) or not len(negative_index) or iterations <= 0:
        return {
            "auroc_ci_low": math.nan,
            "auroc_ci_high": math.nan,
            "auprc_ci_low": math.nan,
            "auprc_ci_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    auc: list[float] = []
    average_precision: list[float] = []
    for _ in range(iterations):
        sampled_positive = rng.choice(
            positive_index, size=len(positive_index), replace=True
        )
        sampled_negative = rng.choice(
            negative_index, size=len(negative_index), replace=True
        )
        sampled = np.concatenate((sampled_positive, sampled_negative))
        current = binary_ranking_metrics(y[sampled].tolist(), values[sampled].tolist())
        auc.append(current["auroc"])
        average_precision.append(current["auprc"])
    auc_low, auc_high = np.quantile(auc, (0.025, 0.975))
    ap_low, ap_high = np.quantile(average_precision, (0.025, 0.975))
    return {
        "auroc_ci_low": float(auc_low),
        "auroc_ci_high": float(auc_high),
        "auprc_ci_low": float(ap_low),
        "auprc_ci_high": float(ap_high),
    }


def _verify_profile_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = config["inputs"]
    profile_manifest_path = Path(inputs["profile_manifest"])
    profile_manifest = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
    if not (
        profile_manifest.get("frozen")
        and profile_manifest.get("middle_layer_independent")
        and profile_manifest.get("labels_withheld")
    ):
        raise ValueError("DR20.1 raw profile snapshot is not independently frozen")
    configured = {
        "diseases.tsv": inputs["diseases"],
        "models.tsv": inputs["models"],
        "model_phenotypes.tsv": inputs["model_phenotypes"],
        "gene_units.tsv": inputs["gene_units"],
        "disease_gene_gold.tsv": inputs["disease_gene_gold"],
    }
    rows = []
    official_freeze_path = Path(inputs["official_freeze_manifest"])
    official_freeze = json.loads(official_freeze_path.read_text(encoding="utf-8"))
    expected_truth = official_freeze["files"]["gene_level_truth.tsv"]["sha256"]
    if sha256(inputs["gene_level_truth"]) != expected_truth:
        raise ValueError("Gene-level truth does not match the official freeze manifest")
    for name, path in configured.items():
        expected = profile_manifest["files"][name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Configured {name} does not match the profile manifest")
        rows.append({"input": name, **input_record(path), "status": "valid"})
    truth_path = Path(inputs["gene_level_truth"])
    rows.append({"input": "gene_level_truth", **input_record(truth_path), "status": "valid"})
    rows.append(
        {
            "input": "profile_manifest",
            **input_record(profile_manifest_path),
            "status": "valid",
        }
    )
    rows.append(
        {
            "input": "official_freeze_manifest",
            **input_record(official_freeze_path),
            "status": "valid",
        }
    )
    for name in ("embedding_label", "embedding_synonym"):
        path = Path(inputs[name])
        if not path.is_file() or not Path(str(path) + ".manifest.json").is_file():
            raise FileNotFoundError(path)
        rows.append({"input": name, **input_record(path), "status": "valid"})
        rows.append(
            {
                "input": name + "_manifest",
                **input_record(str(path) + ".manifest.json"),
                "status": "valid",
            }
        )
    return rows


def _score_units(
    *,
    diseases: Mapping[str, Sequence[Any]],
    models_by_gene: Mapping[str, Sequence[Any]],
    phenotypes: Mapping[str, Sequence[Any]],
    units: Sequence[Mapping[str, str]],
    stores: Mapping[str, DenseEmbeddingStore],
) -> list[dict[str, Any]]:
    results = []
    for unit in units:
        gene = unit["gene_symbol"].upper()
        disease_ids = split_values(unit.get("covered_disorder_ids", ""))
        gene_models = models_by_gene.get(gene, ())
        best: dict[tuple[str, str], tuple[float, str, str, float, float]] = {}
        comparisons = 0
        for disease_id in disease_ids:
            terms = diseases.get(disease_id, ())
            hpo_ids = [term.hpo_id for term in terms]
            if not hpo_ids:
                continue
            for model in gene_models:
                mpo_ids = [
                    row.mpo_id
                    for row in phenotypes.get(model.model_id, ())
                    if row.status == "abnormal"
                ]
                if not mpo_ids:
                    continue
                comparisons += 1
                for representation, store in stores.items():
                    components = store.profile_components(hpo_ids, mpo_ids)
                    if components is None:
                        continue
                    bma, max_pair = components
                    for formula in FORMULAS:
                        score = aggregate_profile_score(bma, max_pair, formula)
                        key = (representation, formula)
                        current = best.get(key)
                        candidate = (score, disease_id, model.model_id, bma, max_pair)
                        if current is None or score > current[0] or (
                            math.isclose(score, current[0])
                            and (disease_id, model.model_id) < (current[1], current[2])
                        ):
                            best[key] = candidate
        for representation in stores:
            for formula in FORMULAS:
                value = best.get((representation, formula))
                results.append(
                    {
                        "gene_symbol": gene,
                        "mgi_id": unit["mgi_id"],
                        "representation": representation,
                        "formula": formula,
                        "experiment": f"{representation}__{formula}",
                        "raw_replay_ready": as_bool(unit["raw_replay_ready"]),
                        "covered_disorder_count": unit["covered_disorder_count"],
                        "candidate_model_count": len(gene_models),
                        "profile_comparison_count": comparisons,
                        "score": value[0] if value else "",
                        "best_disease_id": value[1] if value else "",
                        "best_model_id": value[2] if value else "",
                        "best_bma": value[3] if value else "",
                        "best_max_pair": value[4] if value else "",
                    }
                )
    return results


def run_impc_dr20_embedding_test(
    config: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_summary = _verify_profile_inputs(config)
    inputs = config["inputs"]
    diseases = load_diseases(inputs["diseases"])
    models = load_models(inputs["models"])
    phenotypes = load_model_phenotypes(inputs["model_phenotypes"], set(models))
    models_by_gene: dict[str, list[Any]] = defaultdict(list)
    for model in models.values():
        models_by_gene[model.human_gene_symbol].append(model)
    units = list(read_tsv(inputs["gene_units"]))
    stores = {
        "label_cls": DenseEmbeddingStore(inputs["embedding_label"]),
        "exact_synonym_centroid_cls": DenseEmbeddingStore(
            inputs["embedding_synonym"]
        ),
    }
    score_rows = _score_units(
        diseases=diseases,
        models_by_gene=models_by_gene,
        phenotypes=phenotypes,
        units=units,
        stores=stores,
    )
    score_fields = list(score_rows[0])
    write_tsv(output / "scores/embedding_gene_unit_scores.tsv", score_rows, score_fields)

    truth = {
        row["gene_symbol"].upper(): row for row in read_tsv(inputs["gene_level_truth"])
    }
    evaluation_rows = []
    metric_rows = []
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        official = truth[row["gene_symbol"]]
        item = {
            **row,
            "official_phenodigm_match": official["official_phenodigm_match"],
            "official_max_score": official["official_max_score"],
            "coverage_failure": row["score"] == "",
        }
        evaluation_rows.append(item)
        by_experiment[row["experiment"]].append(item)
    write_tsv(
        output / "evaluation/per_unit_results.tsv",
        evaluation_rows,
        list(evaluation_rows[0]),
    )

    metrics_by_experiment: dict[str, dict[str, dict[str, float]]] = {}
    bootstrap_iterations = int(config.get("benchmark", {}).get("bootstrap_iterations", 1000))
    seed = int(config.get("project", {}).get("random_seed", 2026))
    for experiment, rows in sorted(by_experiment.items()):
        metrics_by_experiment[experiment] = {}
        for stratum, selected in (
            ("all_official_units", rows),
            (
                "public_raw_replay_ready",
                [row for row in rows if row["raw_replay_ready"]],
            ),
        ):
            labels = [
                row["official_phenodigm_match"] == "yes" for row in selected
            ]
            scores = [
                float(row["score"]) if row["score"] != "" else 0.0
                for row in selected
            ]
            metrics = binary_ranking_metrics(labels, scores)
            metrics.update(
                bootstrap_binary_intervals(
                    labels,
                    scores,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
            )
            positive_pairs = [
                (float(row["score"]), float(row["official_max_score"]))
                for row in selected
                if row["official_phenodigm_match"] == "yes"
                and row["score"] != ""
                and row["official_max_score"] not in {"", "NA"}
            ]
            correlation = (
                spearmanr(
                    [item[0] for item in positive_pairs],
                    [item[1] for item in positive_pairs],
                ).statistic
                if len(positive_pairs) > 1
                else math.nan
            )
            metrics["official_positive_score_spearman"] = float(correlation)
            metrics["coverage_rate"] = sum(row["score"] != "" for row in selected) / len(
                selected
            )
            metrics_by_experiment[experiment][stratum] = metrics
            for metric, value in metrics.items():
                metric_rows.append(
                    {
                        "experiment": experiment,
                        "stratum": stratum,
                        "metric": metric,
                        "value": value,
                    }
                )
    write_tsv(
        output / "evaluation/metrics.tsv",
        metric_rows,
        ["experiment", "stratum", "metric", "value"],
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "IMPC_DR20.1_gene_level_embedding_agreement",
        "evaluation_target": (
            "Official gene-level PhenoDigm match/non-match labels; this measures "
            "agreement with PhenoDigm, not independent biological truth."
        ),
        "counts": {
            "official_test_units": len(units),
            "raw_replay_ready_units": sum(as_bool(row["raw_replay_ready"]) for row in units),
            "diseases_with_profiles": len(diseases),
            "models": len(models),
            "model_phenotypes": sum(len(rows) for rows in phenotypes.values()),
            "embedding_score_ready_units": sum(
                row["score"] != ""
                for row in by_experiment[next(iter(sorted(by_experiment)))]
            ),
        },
        "metrics": metrics_by_experiment,
        "inputs": {
            str(row["input"]): {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in input_summary
        },
        "warnings": [
            "The 16 units without a public historical HPO profile receive score zero only in the all-units stratum.",
            "Public model profiles reproduce 96.3% of official open-candidate model keys; stage inference is not an exact copy of the internal IMPC pipeline.",
            "No CPO resource is loaded in this IMPC-equivalent surface test.",
        ],
    }
    (output / "resolved_config.yaml").write_text(
        dump_config(dict(config)), encoding="utf-8"
    )
    write_tsv(
        output / "qc/input_summary.tsv",
        input_summary,
        ["input", "path", "bytes", "sha256", "status"],
    )
    write_json(output / "run_manifest.json", manifest)
    _write_report(output, manifest)
    return manifest


def _write_report(output: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# IMPC DR20.1 embedding test",
        "",
        "This test measures how well embedding-derived HPO-MP profile scores reproduce "
        "the official gene-level PhenoDigm match labels.",
        "",
        "| Representation | Formula | Stratum | AUROC | AUPRC | Spearman | Coverage |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for experiment, strata in manifest["metrics"].items():
        representation, formula = experiment.split("__", 1)
        for stratum, metrics in strata.items():
            lines.append(
                f"| {representation} | {formula} | {stratum} | "
                f"{metrics['auroc']:.4f} [{metrics['auroc_ci_low']:.4f}, {metrics['auroc_ci_high']:.4f}] | "
                f"{metrics['auprc']:.4f} [{metrics['auprc_ci_low']:.4f}, {metrics['auprc_ci_high']:.4f}] | "
                f"{metrics['official_positive_score_spearman']:.4f} | "
                f"{metrics['coverage_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Official PhenoDigm labels are the evaluation target, so these are agreement "
            "metrics rather than independent disease-gene discovery metrics. CPO is excluded "
            "because the published DR20.1 endpoint evaluates surface phenotype recapitulation.",
        ]
    )
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
