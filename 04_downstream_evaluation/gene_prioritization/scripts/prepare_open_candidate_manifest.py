from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellphenorank.io import (
    input_record,
    load_model_phenotypes,
    load_models,
    sha256,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an answer-independent open mouse-model candidate universe"
    )
    parser.add_argument("--source-freeze-manifest", required=True)
    parser.add_argument("--diseases", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-phenotypes", required=True)
    parser.add_argument("--gold-standard", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_freeze_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_manifest.get("frozen") or not source_manifest.get(
        "middle_layer_independent"
    ):
        raise ValueError("Source disease benchmark is not independently frozen")
    diseases = Path(args.diseases).resolve()
    gold = Path(args.gold_standard).resolve()
    expected_diseases = source_manifest.get("files", {}).get("diseases.tsv", {}).get(
        "sha256"
    )
    expected_gold = source_manifest.get("files", {}).get("gold_models.tsv", {}).get(
        "sha256"
    )
    if expected_diseases and sha256(diseases) != expected_diseases:
        raise ValueError("Disease input does not match the source freeze")
    if expected_gold and sha256(gold) != expected_gold:
        raise ValueError("Gold input does not match the source freeze")

    all_models = load_models(args.models)
    single_gene_models = {
        model_id: model
        for model_id, model in all_models.items()
        if ";" not in model.human_gene_symbol
    }
    phenotypes = load_model_phenotypes(
        args.model_phenotypes, set(single_gene_models)
    )
    phenotype_models = {
        model_id: model
        for model_id, model in single_gene_models.items()
        if any(row.status == "abnormal" for row in phenotypes.get(model_id, ()))
    }
    genes = {model.human_gene_symbol for model in phenotype_models.values()}
    files = {
        "diseases.tsv": input_record(diseases),
        "models.tsv": input_record(args.models),
        "model_phenotypes.tsv": input_record(args.model_phenotypes),
        "gold_standard.tsv": input_record(gold),
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": "open answer-independent MGI single-gene candidate benchmark",
        "frozen": True,
        "candidate_universe_frozen": True,
        "middle_layer_independent": True,
        "publication_grade": False,
        "review_policy": source_manifest.get("review_policy"),
        "evaluation_mode": "open_candidate_benchmark",
        "source_freeze_manifest": input_record(source_manifest_path),
        "candidate_rule": (
            "Every single-gene mouse genotype with at least one abnormal MP annotation; "
            "no disease-gene label, CPO resource, or ranking result is used."
        ),
        "candidate_model_count": len(phenotype_models),
        "candidate_gene_count": len(genes),
        "excluded_multigene_or_unattributed_model_count": len(all_models)
        - len(single_gene_models),
        "files": files,
    }
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
