from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from .io import input_record, sha256, write_json, write_tsv


ZENODO_DOI = "10.5281/zenodo.15489406"
PAPER_DOI = "10.1242/dmm.050604"
EXPECTED_MD5 = {
    "gene_summary_dr20.1.txt": "7b4c2c8d89d64d27b1a1472ec55f14cc",
    "phenodigm_matches_dr20.1.txt": "ea2909dc4677f826b6a4903a2c1a255c",
    "phenodigm_other_dr20.1.txt.gz": "d26223b6e44d9325906abe1d1348abf5",
}


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_eligible_gene(row: Mapping[str, str]) -> bool:
    return (
        row.get("IMPC_pipeline") == "yes"
        and row.get("IMPC_phenotypes") == "yes"
        and row.get("HPO_phenotypes") == "yes"
        and row.get("disorder_id") not in {"", "-"}
    )


def split_disorder_ids(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split("|") if item.strip()))


def _check_source_files(source_dir: Path) -> list[dict[str, object]]:
    records = []
    for name, expected in EXPECTED_MD5.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = md5(path)
        if observed != expected:
            raise ValueError(f"Zenodo MD5 mismatch for {name}: {observed} != {expected}")
        records.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "zenodo_md5": expected,
                "sha256": sha256(path),
                "status": "valid",
            }
        )
    return records


def _positive_match_stats(rows: Iterable[Mapping[str, str]]) -> dict[str, object]:
    row_count = 0
    genes: set[str] = set()
    diseases: set[str] = set()
    disease_genes: set[tuple[str, str]] = set()
    models: set[str] = set()
    scores: list[float] = []
    for row in rows:
        row_count += 1
        gene = str(row["gene_symbol"])
        disease = str(row["disorder_id"])
        genes.add(gene)
        diseases.add(disease)
        disease_genes.add((disease, gene))
        models.add(str(row["description"]))
        scores.append(float(row["score"]))
    return {
        "rows": row_count,
        "genes": len(genes),
        "diseases": len(diseases),
        "disease_gene_pairs": len(disease_genes),
        "model_descriptions": len(models),
        "minimum_score": min(scores) if scores else None,
        "maximum_score": max(scores) if scores else None,
    }


def _open_prediction_stats(path: Path) -> dict[str, object]:
    rows = 0
    genes: set[str] = set()
    diseases: set[str] = set()
    minimum: float | None = None
    maximum: float | None = None
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows += 1
            genes.add(row["gene_symbol"])
            diseases.add(row["query"])
            score = float(row["score"])
            minimum = score if minimum is None else min(minimum, score)
            maximum = score if maximum is None else max(maximum, score)
    return {
        "rows": rows,
        "genes": len(genes),
        "diseases": len(diseases),
        "minimum_score": minimum,
        "maximum_score": maximum,
        "selection_boundary": "published non-disease-gene candidates with score > 40",
    }


def prepare_impc_dr20_reproduction(
    source_dir: str | Path, output_dir: str | Path
) -> dict[str, object]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_records = _check_source_files(source)
    summary_rows = read_tsv_rows(source / "gene_summary_dr20.1.txt")
    positive_rows = read_tsv_rows(source / "phenodigm_matches_dr20.1.txt")

    eligible = [row for row in summary_rows if is_eligible_gene(row)]
    matched = [row for row in eligible if row["PhenoDigm_match"] == "yes"]
    matched_genes = {row["gene_symbol"] for row in matched}
    if any(row["max_score"] == "NA" for row in matched):
        raise ValueError("Matched gene has no official max_score")
    if {row["gene_symbol"] for row in positive_rows} != matched_genes:
        raise ValueError("Positive model file and gene-level match labels disagree")

    pipeline_genes = sum(row["IMPC_pipeline"] == "yes" for row in summary_rows)
    disease_genes = sum(row["disorder_id"] != "-" for row in summary_rows)
    pipeline_disease_genes = sum(
        row["IMPC_pipeline"] == "yes" and row["disorder_id"] != "-"
        for row in summary_rows
    )
    unique_diseases = {
        disease_id
        for row in eligible
        for disease_id in split_disorder_ids(row["disorder_id"])
    }
    inputs = [
        {
            "gene_symbol": row["gene_symbol"],
            "hgnc_id": row["hgnc_id"],
            "mgi_id": row["mgi_id"],
            "disorder_ids": row["disorder_id"],
            "disorder_names": row["disorder_name"],
            "impc_pipeline": row["IMPC_pipeline"],
            "impc_phenotypes_available": row["IMPC_phenotypes"],
            "hpo_phenotypes_available": row["HPO_phenotypes"],
            "evaluation_unit": "one_to_one_ortholog_gene_knockout",
        }
        for row in eligible
    ]
    candidate_universe = [
        {
            "gene_symbol": row["gene_symbol"],
            "hgnc_id": row["hgnc_id"],
            "mgi_id": row["mgi_id"],
            "candidate_status": "one_to_one_ortholog_in_impc_pipeline",
        }
        for row in summary_rows
        if row["IMPC_pipeline"] == "yes"
    ]
    truth = [
        {
            "gene_symbol": row["gene_symbol"],
            "hgnc_id": row["hgnc_id"],
            "mgi_id": row["mgi_id"],
            "official_phenodigm_match": row["PhenoDigm_match"],
            "official_max_score": row["max_score"],
            "truth_source": f"Zenodo {ZENODO_DOI} gene_summary_dr20.1.txt",
        }
        for row in eligible
    ]
    input_path = output / "evaluation_inputs/eligible_gene_cohort.tsv"
    candidate_path = output / "evaluation_inputs/candidate_gene_universe.tsv"
    truth_path = output / "withheld_truth/gene_level_truth.tsv"
    write_tsv(input_path, inputs, list(inputs[0]))
    write_tsv(candidate_path, candidate_universe, list(candidate_universe[0]))
    write_tsv(truth_path, truth, list(truth[0]))

    observed = {
        "gene_summary_rows": len(summary_rows),
        "one_to_one_genes_in_impc_pipeline": pipeline_genes,
        "one_to_one_disease_genes": disease_genes,
        "one_to_one_disease_genes_in_impc_pipeline": pipeline_disease_genes,
        "eligible_gene_knockouts": len(eligible),
        "phenodigm_matched_gene_knockouts": len(matched),
        "phenodigm_match_rate": len(matched) / len(eligible),
        "unique_disorder_ids_in_eligible_genes": len(unique_diseases),
    }
    checks = [
        {
            "metric": "eligible_gene_knockouts",
            "expected": 2400,
            "observed": observed["eligible_gene_knockouts"],
            "status": "pass" if observed["eligible_gene_knockouts"] == 2400 else "fail",
        },
        {
            "metric": "phenodigm_matched_gene_knockouts",
            "expected": 1311,
            "observed": observed["phenodigm_matched_gene_knockouts"],
            "status": (
                "pass"
                if observed["phenodigm_matched_gene_knockouts"] == 1311
                else "fail"
            ),
        },
    ]
    if any(row["status"] != "pass" for row in checks):
        raise ValueError("Published DR20.1 reproduction counts did not match")
    write_tsv(
        output / "reproduction_checks.tsv",
        checks,
        ["metric", "expected", "observed", "status"],
    )

    positive_stats = _positive_match_stats(positive_rows)
    open_stats = _open_prediction_stats(source / "phenodigm_other_dr20.1.txt.gz")
    source_manifest = {
        "schema_version": "1.0",
        "dataset": "IMPC Disease Models DR20.1",
        "zenodo_doi": ZENODO_DOI,
        "paper_doi": PAPER_DOI,
        "license": "CC-BY-4.0",
        "access": "open",
        "source_files": source_records,
        "source_boundaries": {
            "gene_summary": "gene-level cohort and official match label",
            "phenodigm_matches": "positive known disease-model matches only",
            "phenodigm_other": "non-disease-gene predictions truncated to score > 40",
        },
    }
    write_json(output / "source_manifest.json", source_manifest)
    result = {
        "status": "reproduced",
        "published_claim": "1311 of 2400 eligible gene knockouts match",
        "observed": observed,
        "positive_match_file": positive_stats,
        "open_prediction_file": open_stats,
        "interpretation": (
            "This reproduces the published output counts from the official frozen dataset. "
            "It does not yet recompute PhenoDigm scores from raw HPO and MP inputs."
        ),
    }
    write_json(output / "reproduction_summary.json", result)
    freeze_manifest = {
        "schema_version": "1.0",
        "dataset": "IMPC DR20.1/2024 external reproduction benchmark",
        "frozen": True,
        "middle_layer_independent": True,
        "official_external_benchmark": True,
        "labels_withheld": True,
        "evaluation_unit": "one_to_one_ortholog_gene_knockout",
        "candidate_gene_count": pipeline_genes,
        "test_unit_count": len(eligible),
        "positive_unit_count": len(matched),
        "files": {
            "eligible_gene_cohort.tsv": input_record(input_path),
            "candidate_gene_universe.tsv": input_record(candidate_path),
            "gene_level_truth.tsv": input_record(truth_path),
            "source_manifest.json": input_record(output / "source_manifest.json"),
            "reproduction_summary.json": input_record(
                output / "reproduction_summary.json"
            ),
        },
        "use_policy": (
            "Do not expose withheld_truth or official PhenoDigm outcome fields to "
            "candidate generation, embedding construction, threshold selection, or CPO reranking."
        ),
    }
    write_json(output / "freeze_manifest.json", freeze_manifest)
    return result


def audit_raw_replay_inputs(
    *,
    eligible_cohort: str | Path,
    assertions: str | Path,
    hpoa: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    cohort = read_tsv_rows(eligible_cohort)
    assertion_ids: set[str] = set()
    with gzip.open(assertions, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            assertion_ids.add(row["marker_accession_id"])
    hpo_disease_ids: set[str] = set()
    with Path(hpoa).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"
        )
        for row in reader:
            if row.get("qualifier", "").upper() != "NOT":
                hpo_disease_ids.add(row["database_id"])

    audit_rows = []
    for row in cohort:
        disease_ids = split_disorder_ids(row["disorder_ids"])
        covered = tuple(item for item in disease_ids if item in hpo_disease_ids)
        mouse_ready = row["mgi_id"] in assertion_ids
        human_ready = bool(covered)
        audit_rows.append(
            {
                "gene_symbol": row["gene_symbol"],
                "mgi_id": row["mgi_id"],
                "disorder_ids": disease_ids,
                "hpoa_covered_disorder_ids": covered,
                "mouse_mp_available": mouse_ready,
                "human_hpo_available": human_ready,
                "raw_replay_ready": mouse_ready and human_ready,
                "coverage_status": (
                    "ready"
                    if mouse_ready and human_ready
                    else "missing_mouse_mp"
                    if not mouse_ready
                    else "missing_human_hpo"
                ),
            }
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "raw_replay_coverage.tsv"
    write_tsv(audit_path, audit_rows, list(audit_rows[0]))
    ready = sum(bool(row["raw_replay_ready"]) for row in audit_rows)
    mouse_ready = sum(bool(row["mouse_mp_available"]) for row in audit_rows)
    human_ready = sum(bool(row["human_hpo_available"]) for row in audit_rows)
    result = {
        "test_units": len(audit_rows),
        "mouse_mp_available": mouse_ready,
        "human_hpo_available": human_ready,
        "raw_replay_ready": ready,
        "raw_replay_coverage": ready / len(audit_rows) if audit_rows else 0.0,
        "policy": (
            "Retain all official test units. Report the full 2400-unit output "
            "reproduction and the post-freeze raw-replay-ready stratum separately."
        ),
        "inputs": {
            "eligible_cohort": input_record(eligible_cohort),
            "assertions": input_record(assertions),
            "hpoa": input_record(hpoa),
        },
        "output": input_record(audit_path),
    }
    write_json(output / "raw_replay_audit.json", result)
    return result


def _model_life_stage(row: Mapping[str, str]) -> str:
    pipeline = " ".join(
        (
            row.get("pipeline_name", ""),
            row.get("pipeline_stable_id", ""),
            row.get("procedure_name", ""),
            row.get("procedure_stable_id", ""),
        )
    ).lower()
    procedure_id = row.get("procedure_stable_id", "").upper()
    if "embryo" in pipeline or procedure_id.startswith(
        ("IMPC_EV", "IMPC_GE", "ESLIM_")
    ):
        return "embryo"
    if "late adult" in pipeline or "LA_" in pipeline.upper():
        return "late"
    if "interval" in pipeline or "IP_" in pipeline.upper():
        return "middle"
    return "early"


def _zygosity_abbreviation(value: str) -> str:
    return {
        "heterozygote": "het",
        "homozygote": "hom",
        "hemizygote": "hem",
    }.get(value, value)


def prepare_raw_replay_profiles(
    *,
    eligible_cohort: str | Path,
    candidate_universe: str | Path,
    assertions: str | Path,
    hpoa: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    eligible = read_tsv_rows(eligible_cohort)
    candidates = read_tsv_rows(candidate_universe)
    candidate_by_mgi = {row["mgi_id"]: row for row in candidates}
    required_disease_ids = {
        disease_id
        for row in eligible
        for disease_id in split_disorder_ids(row["disorder_ids"])
    }

    model_metadata: dict[str, dict[str, object]] = {}
    model_phenotypes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    excluded_non_mp_assertions = 0
    with gzip.open(assertions, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate = candidate_by_mgi.get(row["marker_accession_id"])
            if candidate is None or not row.get("allele_accession_id") or not row.get(
                "mp_term_id"
            ):
                continue
            if not row["mp_term_id"].startswith("MP:"):
                excluded_non_mp_assertions += 1
                continue
            stage = _model_life_stage(row)
            zygosity = row["zygosity"]
            zygosity_short = _zygosity_abbreviation(zygosity)
            model_id = f"{row['allele_accession_id']}#{zygosity_short}#{stage}"
            model_metadata.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "model_name": f"{row['allele_symbol']} {zygosity_short} {stage}",
                    "mouse_gene_symbol": row["marker_symbol"],
                    "human_gene_symbol": candidate["gene_symbol"],
                    "marker_accession_id": row["marker_accession_id"],
                    "allele_id": row["allele_accession_id"],
                    "allele_type": "",
                    "zygosity": zygosity,
                    "life_stage": stage,
                    "conditional": False,
                    "source": "IMPC_DR20.1_genotype-phenotype-assertions-ALL",
                },
            )
            model_phenotypes[model_id].add(
                (row["mp_term_id"], row.get("mp_term_name", ""))
            )

    hpo_rows: list[dict[str, object]] = []
    hpo_disease_ids: set[str] = set()
    with Path(hpoa).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"
        )
        for row in reader:
            disease_id = row["database_id"]
            if (
                disease_id not in required_disease_ids
                or row.get("qualifier", "").upper() == "NOT"
                or row.get("aspect", "P") != "P"
            ):
                continue
            hpo_disease_ids.add(disease_id)
            hpo_rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": row["disease_name"],
                    "gene_symbol": "",
                    "hpo_id": row["hpo_id"],
                    "hpo_name": "",
                    "weight": 1.0,
                    "is_core": False,
                    "frequency": row.get("frequency", ""),
                    "onset": row.get("onset", ""),
                    "direction": "abnormal",
                    "source": "HPOA_v2023-10-09",
                }
            )
    seen_hpo: set[tuple[str, str]] = set()
    hpo_rows = [
        row
        for row in hpo_rows
        if (str(row["disease_id"]), str(row["hpo_id"])) not in seen_hpo
        and not seen_hpo.add((str(row["disease_id"]), str(row["hpo_id"])))
    ]

    gold_rows = []
    unit_rows = []
    for row in eligible:
        covered_ids = [
            item
            for item in split_disorder_ids(row["disorder_ids"])
            if item in hpo_disease_ids
        ]
        unit_rows.append(
            {
                "gene_symbol": row["gene_symbol"],
                "mgi_id": row["mgi_id"],
                "covered_disorder_ids": covered_ids,
                "covered_disorder_count": len(covered_ids),
                "raw_replay_ready": bool(covered_ids),
            }
        )
        for disease_id in covered_ids:
            gold_rows.append(
                {
                    "disease_id": disease_id,
                    "gene_symbol": row["gene_symbol"],
                    "mgi_id": row["mgi_id"],
                    "is_evaluation_positive": True,
                    "gold_source": "IMPC_DR20.1_gene_summary_disease_association",
                }
            )
    gold_rows = list(
        {
            (row["disease_id"], row["gene_symbol"]): row for row in gold_rows
        }.values()
    )

    model_rows = [model_metadata[key] for key in sorted(model_metadata)]
    phenotype_rows = [
        {
            "model_id": model_id,
            "mp_id": mp_id,
            "mp_name": mp_name,
            "status": "abnormal",
            "direction": "abnormal",
            "source": "IMPC_DR20.1_genotype-phenotype-assertions-ALL",
        }
        for model_id in sorted(model_phenotypes)
        for mp_id, mp_name in sorted(model_phenotypes[model_id])
    ]
    output = Path(output_dir).resolve()
    evaluation_dir = output / "evaluation_inputs"
    truth_dir = output / "withheld_truth"
    disease_path = evaluation_dir / "diseases.tsv"
    model_path = evaluation_dir / "models.tsv"
    phenotype_path = evaluation_dir / "model_phenotypes.tsv"
    unit_path = evaluation_dir / "gene_units.tsv"
    gold_path = truth_dir / "disease_gene_gold.tsv"
    write_tsv(disease_path, hpo_rows, list(hpo_rows[0]))
    write_tsv(model_path, model_rows, list(model_rows[0]))
    write_tsv(phenotype_path, phenotype_rows, list(phenotype_rows[0]))
    write_tsv(unit_path, unit_rows, list(unit_rows[0]))
    write_tsv(gold_path, gold_rows, list(gold_rows[0]))

    candidate_genes_with_models = {row["human_gene_symbol"] for row in model_rows}
    test_genes_with_models = {
        row["gene_symbol"]
        for row in unit_rows
        if row["gene_symbol"] in candidate_genes_with_models
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": "IMPC DR20.1 public raw replay profiles",
        "frozen": True,
        "middle_layer_independent": True,
        "labels_withheld": True,
        "model_definition": "allele_accession_id + zygosity + inferred life stage",
        "life_stage_rules": {
            "embryo": "embryo procedure name or IMPC_EV/IMPC_GE/ESLIM procedure",
            "late": "late adult pipeline",
            "middle": "interval pipeline",
            "early": "all remaining phenotype procedures",
        },
        "counts": {
            "candidate_genes_declared": len(candidates),
            "candidate_genes_with_models": len(candidate_genes_with_models),
            "models": len(model_rows),
            "model_phenotypes": len(phenotype_rows),
            "excluded_non_mp_assertions": excluded_non_mp_assertions,
            "required_diseases": len(required_disease_ids),
            "hpo_covered_diseases": len(hpo_disease_ids),
            "disease_hpo_rows": len(hpo_rows),
            "gene_test_units": len(unit_rows),
            "raw_replay_ready_gene_units": sum(
                bool(row["raw_replay_ready"]) for row in unit_rows
            ),
            "test_genes_with_models": len(test_genes_with_models),
            "disease_gene_gold_pairs": len(gold_rows),
        },
        "inputs": {
            "eligible_cohort": input_record(eligible_cohort),
            "candidate_universe": input_record(candidate_universe),
            "assertions": input_record(assertions),
            "hpoa": input_record(hpoa),
        },
        "files": {
            "diseases.tsv": input_record(disease_path),
            "models.tsv": input_record(model_path),
            "model_phenotypes.tsv": input_record(phenotype_path),
            "gene_units.tsv": input_record(unit_path),
            "disease_gene_gold.tsv": input_record(gold_path),
        },
    }
    write_json(output / "profile_manifest.json", manifest)
    return manifest
