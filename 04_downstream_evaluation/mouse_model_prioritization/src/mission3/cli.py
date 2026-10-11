from __future__ import annotations

from uhpo_common.paths import HPO_CPO, MPO_CPO, CL_SKELETON, MARKER_MINING, RESOURCES, CL_GO_LINKING

from uhpo_common.paths import REPO_ROOT

import argparse
from pathlib import Path

from .cohort import audit_and_select_cohort
from .diseases import prepare_disease_profiles
from .mgi import prepare_mgi_inputs
from .pipeline import run_pipeline
from .sources import download_reference_data


MISSION_DIR = Path(__file__).resolve().parents[2]
WORKSPACE = MISSION_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=str(REPO_ROOT / '04_downstream_evaluation/gene_prioritization'),
        description="Prioritize concrete mouse genotypes for rare diseases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run model prioritization and reporting.")
    run.add_argument("--diseases", required=True)
    run.add_argument("--models", required=True)
    run.add_argument("--model-phenotypes", required=True)
    run.add_argument(
        "--mp-hpo-mapping",
        default=RESOURCES / "mappings/mpo_hpo_mapping.tsv",
    )
    run.add_argument(
        "--hpo-mechanisms",
        default=HPO_CPO / "data/legacy_inputs/hpo_cpo_candidate_links.partial.tsv",
    )
    run.add_argument(
        "--mp-mechanisms",
        default=MPO_CPO / "data/legacy_inputs/mpo_cell_mechanism_triples.tsv",
    )
    run.add_argument("--cell-catalog", default=MARKER_MINING / "expanded/expanded_cell_catalog.tsv")
    run.add_argument(
        "--go-classification", default=CL_GO_LINKING / "GO_link/results/full/go_process_scope.tsv"
    )
    run.add_argument("--cl-obo", default=CL_SKELETON / "cell_ontology/cl-basic.obo")
    run.add_argument("--go-obo", default=RESOURCES / "ontologies/go-basic.obo")
    run.add_argument("--cre-drivers")
    run.add_argument("--config", default=MISSION_DIR / "config/default.json")
    run.add_argument("--output-dir", required=True)
    prepare = subparsers.add_parser("prepare-mgi", help="Normalize official headerless MGI reports.")
    prepare.add_argument("--pheno-geno", required=True)
    prepare.add_argument("--orthologs")
    prepare.add_argument("--phenotypic-alleles")
    prepare.add_argument("--genotype-disease")
    prepare.add_argument("--disease-mouse-models")
    prepare.add_argument("--phenotype-sex")
    prepare.add_argument("--output-dir", required=True)
    download = subparsers.add_parser(
        "download-data",
        help="Download and hash-pin official MGI and HPO source files.",
    )
    download.add_argument("--source", choices=("all", "mgi", "hpo"), default="all")
    download.add_argument("--output-dir", default=MISSION_DIR / "data/raw")
    download.add_argument("--force", action="store_true")
    download.add_argument("--timeout", type=int, default=120)
    diseases = subparsers.add_parser(
        "prepare-diseases",
        help="Build disease-HPO profiles from official HPO annotation files.",
    )
    diseases.add_argument("--hpoa", required=True)
    diseases.add_argument("--gene-disease", required=True)
    diseases.add_argument("--hpo-obo")
    diseases.add_argument("--curation")
    diseases.add_argument("--output-dir", required=True)
    audit = subparsers.add_parser(
        "audit-cohort",
        help="Audit data coverage and select a fixed 30-50 disease pilot cohort.",
    )
    audit.add_argument("--diseases", required=True)
    audit.add_argument("--models", required=True)
    audit.add_argument("--model-phenotypes", required=True)
    audit.add_argument("--disease-xrefs")
    audit.add_argument(
        "--mp-hpo-mapping",
        default=RESOURCES / "mappings/mpo_hpo_mapping.tsv",
    )
    audit.add_argument(
        "--hpo-mechanisms",
        default=HPO_CPO / "data/legacy_inputs/hpo_cpo_candidate_links.partial.tsv",
    )
    audit.add_argument(
        "--mp-mechanisms",
        default=MPO_CPO / "data/legacy_inputs/mpo_cell_mechanism_triples.tsv",
    )
    audit.add_argument("--cell-catalog", default=MARKER_MINING / "expanded/expanded_cell_catalog.tsv")
    audit.add_argument(
        "--go-classification", default=CL_GO_LINKING / "GO_link/results/full/go_process_scope.tsv"
    )
    audit.add_argument("--config", default=MISSION_DIR / "config/default.json")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--min-diseases", type=int, default=30)
    audit.add_argument("--max-diseases", type=int, default=50)
    audit.add_argument("--min-models", type=int, default=2)
    audit.add_argument("--min-known-models", type=int, default=1)
    audit.add_argument("--min-core-hpo", type=int, default=3)
    audit.add_argument("--min-surface-coverage", type=float, default=0.70)
    audit.add_argument("--min-hpo-cpo", type=int, default=2)
    audit.add_argument("--min-comparable-cpo", type=int, default=2)
    audit.add_argument("--min-mp-cpo-coverage", type=float, default=0.30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        manifest = run_pipeline(
            diseases_path=args.diseases,
            models_path=args.models,
            model_phenotypes_path=args.model_phenotypes,
            mp_hpo_mapping_path=args.mp_hpo_mapping,
            output_dir=args.output_dir,
            config_path=args.config,
            hpo_mechanisms_path=args.hpo_mechanisms,
            mp_mechanisms_path=args.mp_mechanisms,
            cell_catalog_path=args.cell_catalog,
            go_classification_path=args.go_classification,
            cl_obo_path=args.cl_obo,
            go_obo_path=args.go_obo,
            cre_drivers_path=args.cre_drivers,
        )
        print(f"Wrote {manifest['counts']['ranking_rows']} rankings to {args.output_dir}")
        return 0
    if args.command == "prepare-mgi":
        manifest = prepare_mgi_inputs(
            pheno_geno_path=args.pheno_geno,
            output_dir=args.output_dir,
            orthologs_path=args.orthologs,
            phenotypic_alleles_path=args.phenotypic_alleles,
            genotype_disease_path=args.genotype_disease,
            disease_mouse_models_path=args.disease_mouse_models,
            phenotype_sex_path=args.phenotype_sex,
        )
        print(f"Wrote {manifest['counts']['models']} normalized MGI models to {args.output_dir}")
        return 0
    if args.command == "download-data":
        manifest = download_reference_data(
            output_dir=args.output_dir,
            source=args.source,
            force=args.force,
            timeout=args.timeout,
        )
        print(f"Downloaded or reused {len(manifest['records'])} files in {args.output_dir}")
        return 0
    if args.command == "prepare-diseases":
        manifest = prepare_disease_profiles(
            hpoa_path=args.hpoa,
            gene_disease_path=args.gene_disease,
            curation_path=args.curation,
            hpo_obo_path=args.hpo_obo,
            output_dir=args.output_dir,
        )
        print(
            f"Wrote {manifest['counts']['emitted_diseases']} disease profiles to "
            f"{args.output_dir}"
        )
        return 0
    if args.command == "audit-cohort":
        manifest = audit_and_select_cohort(
            diseases_path=args.diseases,
            models_path=args.models,
            model_phenotypes_path=args.model_phenotypes,
            disease_xrefs_path=args.disease_xrefs,
            mp_hpo_mapping_path=args.mp_hpo_mapping,
            hpo_mechanisms_path=args.hpo_mechanisms,
            mp_mechanisms_path=args.mp_mechanisms,
            cell_catalog_path=args.cell_catalog,
            go_classification_path=args.go_classification,
            config_path=args.config,
            output_dir=args.output_dir,
            min_diseases=args.min_diseases,
            max_diseases=args.max_diseases,
            min_models=args.min_models,
            min_known_models=args.min_known_models,
            min_core_hpo=args.min_core_hpo,
            min_surface_coverage=args.min_surface_coverage,
            min_hpo_cpo=args.min_hpo_cpo,
            min_comparable_cpo=args.min_comparable_cpo,
            min_mp_cpo_coverage=args.min_mp_cpo_coverage,
        )
        print(
            f"Selected {manifest['counts']['selected_diseases']} diseases; "
            f"pilot_ready={manifest['pilot_ready']}; outputs in {args.output_dir}"
        )
        return 0
    return 2
