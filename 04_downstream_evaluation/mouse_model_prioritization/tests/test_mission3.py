from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


MISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION_DIR / "src"))

from mission3.cohort import audit_and_select_cohort  # noqa: E402
from mission3.diseases import prepare_disease_profiles  # noqa: E402
from mission3.io import load_diseases, load_models  # noqa: E402
from mission3.mgi import infer_zygosity, prepare_mgi_inputs  # noqa: E402
from mission3.models import DiseasePhenotype, MouseModel  # noqa: E402
from mission3.pipeline import run_pipeline  # noqa: E402
from mission3.scoring import direction_compatibility, genetic_compatibility  # noqa: E402
from mission3.sources import MGI_REPORTS, download_reference_data  # noqa: E402


class Mission3Tests(unittest.TestCase):
    def test_reference_downloader_hashes_and_reuses_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for filename in MGI_REPORTS:
                (source / filename).write_text(f"fixture:{filename}\n", encoding="utf-8")
            output = root / "raw"
            first = download_reference_data(
                output_dir=output,
                source="mgi",
                mgi_base_url=source.as_uri(),
            )
            second = download_reference_data(
                output_dir=output,
                source="mgi",
                mgi_base_url=source.as_uri(),
            )
            self.assertTrue(first["complete"])
            self.assertEqual(len(first["records"]), len(MGI_REPORTS))
            self.assertTrue(
                all(record["status"] == "reused" for record in second["records"].values())
            )

    def test_prepare_diseases_and_audit_cohort_with_exact_xref(self) -> None:
        example = MISSION_DIR / "examples"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hpoa = root / "phenotype.hpoa"
            genes = root / "genes_to_disease.txt"
            curation = root / "curation.tsv"
            xrefs = root / "xrefs.tsv"
            hpoa.write_text(
                "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
                "OMIM:1\tFixture disease\t\tHP:TEST0001\tPMID:1\tPCS\t\tHP:0040281\t\t\tP\ttest\n"
                "OMIM:1\tFixture disease\t\tHP:TEST0002\tPMID:2\tPCS\t\tHP:0040282\t\t\tP\ttest\n"
                "OMIM:1\tFixture disease\t\tHP:TEST0003\tPMID:3\tPCS\t\tHP:0040283\t\t\tP\ttest\n"
                "OMIM:1\tFixture disease\tNOT\tHP:SHOULD_SKIP\tPMID:4\tPCS\t\t\t\t\tP\ttest\n",
                encoding="utf-8",
            )
            genes.write_text(
                "ncbi_gene_id\tgene_symbol\tassociation_type\tdisease_id\tsource\n"
                "1\tGENE1\tMENDELIAN\tOMIM:1\tfixture\n",
                encoding="utf-8",
            )
            curation.write_text(
                "disease_id\tgene_symbol\tinheritance\tmolecular_mechanism\tcore_hpo_ids\tinclude\n"
                "OMIM:1\tGENE1\tautosomal dominant\thaploinsufficiency\tHP:TEST0001;HP:TEST0002;HP:TEST0003\ttrue\n",
                encoding="utf-8",
            )
            xrefs.write_text(
                "subject_id\tpredicate_id\tobject_id\n"
                "OMIM:1\tskos:exactMatch\tEXAMPLE:1\n",
                encoding="utf-8",
            )
            profiles = root / "profiles"
            profile_manifest = prepare_disease_profiles(
                hpoa_path=hpoa,
                gene_disease_path=genes,
                curation_path=curation,
                output_dir=profiles,
            )
            self.assertEqual(profile_manifest["counts"]["emitted_diseases"], 1)
            self.assertEqual(profile_manifest["counts"]["skipped_negated_annotations"], 1)

            cohort = root / "cohort"
            cohort_manifest = audit_and_select_cohort(
                diseases_path=profiles / "diseases.all.tsv",
                models_path=example / "models.tsv",
                model_phenotypes_path=example / "model_phenotypes.tsv",
                disease_xrefs_path=xrefs,
                mp_hpo_mapping_path=example / "mp_hpo_mapping.tsv",
                hpo_mechanisms_path=example / "hpo_mechanisms.tsv",
                mp_mechanisms_path=example / "mp_mechanisms.tsv",
                cell_catalog_path=example / "cell_catalog.tsv",
                go_classification_path=example / "go_classification.tsv",
                config_path=example / "config.json",
                output_dir=cohort,
                min_diseases=1,
                max_diseases=1,
                min_models=2,
                min_known_models=1,
                min_core_hpo=3,
                min_surface_coverage=0.7,
                min_hpo_cpo=2,
                min_comparable_cpo=2,
                min_mp_cpo_coverage=0.3,
            )
            self.assertTrue(cohort_manifest["pilot_ready"])
            self.assertEqual(cohort_manifest["selected_disease_ids"], ["OMIM:1"])
            selected_models = load_models(cohort / "pilot_models.tsv")
            self.assertIn("OMIM:1", selected_models["MGI:MODEL_B"].known_disease_ids)
            self.assertTrue((cohort / "coverage_report.html").is_file())

    def test_mgi_adapter_preserves_genotype_and_normal_assay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pheno = root / "MGI_PhenoGenoMP.rpt"
            orthologs = root / "HOM_ProteinCoding.rpt"
            alleles = root / "MGI_PhenotypicAllele.rpt"
            diseases = root / "MGI_Geno_DiseaseDO.rpt"
            disease_models = root / "MGI_DiseaseMouseModel.rpt"
            sex = root / "MGI_Pheno_Sex.rpt"
            pheno.write_text(
                "Gene1<tm1>/+\tGene1<tm1>\tC57BL/6J\tMP:0001\t123\tMGI:1\tMGI:GT1\n"
                "Gene1<tm1>/+\tGene1<tm1>\tC57BL/6J\tMP:0001\t456\tMGI:1\tMGI:GT1\n",
                encoding="utf-8",
            )
            orthologs.write_text("MGI:1\tGene1\t1\tHGNC:1\tGENE1\t1\n", encoding="utf-8")
            alleles.write_text(
                "MGI:A1\tGene1<tm1>\tallele\tknockout\tconditional ready\t123\tMGI:1\tGene1\t\t\t\t\t\n",
                encoding="utf-8",
            )
            diseases.write_text(
                "comp\tsymbol\tid\tbg\tMP:0001\t123\tMGI:1\tDOID:1\t100001\tMGI:GT1\n",
                encoding="utf-8",
            )
            disease_models.write_text(
                "disease\tDOID:2\t\tpairs\tC57BL/6J\tGene1<tm1>\tMGI:A1\t1\tJAX:1\tMGI:GT1\tGene1\tMGI:1\tJAX:2\n",
                encoding="utf-8",
            )
            sex.write_text(
                "MGI:GT1\tfemale\tMP:0002\tnormal term\tcomp\tC57BL/6J\tY\t123\n",
                encoding="utf-8",
            )
            output = root / "out"
            manifest = prepare_mgi_inputs(
                pheno_geno_path=pheno,
                output_dir=output,
                orthologs_path=orthologs,
                phenotypic_alleles_path=alleles,
                genotype_disease_path=diseases,
                disease_mouse_models_path=disease_models,
                phenotype_sex_path=sex,
            )
            models = load_models(output / "models.tsv")
            self.assertEqual(models["MGI:GT1"].human_gene_symbol, "GENE1")
            self.assertTrue(models["MGI:GT1"].conditional)
            self.assertIn("DOID:1", models["MGI:GT1"].known_disease_ids)
            self.assertIn("DOID:2", models["MGI:GT1"].known_disease_ids)
            self.assertEqual(models["MGI:GT1"].availability, "available")
            self.assertEqual(manifest["counts"]["models"], 1)
            with (output / "model_phenotypes.tsv").open(encoding="utf-8") as handle:
                phenotype_rows = list(csv.DictReader(handle, delimiter="\t"))
            normal = next(row for row in phenotype_rows if row["status"] == "tested_normal")
            abnormal = next(row for row in phenotype_rows if row["status"] == "abnormal")
            self.assertIn("MGI_Pheno_Sex:123", normal["source"])
            self.assertIn("MGI_PhenoGenoMP:123", abnormal["source"])
            self.assertIn("MGI_PhenoGenoMP:456", abnormal["source"])

    def test_mgi_zygosity_inference(self) -> None:
        self.assertEqual(infer_zygosity("Gene1<tm1>/+"), "heterozygous")
        self.assertEqual(infer_zygosity("Gene1<tm1>/Gene1<+>"), "heterozygous")
        self.assertEqual(infer_zygosity("Gene1<tm1>/Gene1<tm1>"), "homozygous")
        self.assertEqual(infer_zygosity("Gene1<tm1>/Y"), "hemizygous")
        self.assertEqual(infer_zygosity("Gene1<PWD/Ph>/Y"), "hemizygous")
        self.assertEqual(
            infer_zygosity("Gene1<PWD/Ph>/Gene1<PWD/Ph>"),
            "homozygous",
        )
        self.assertEqual(infer_zygosity("Gene1/Gene2<tm1>/Y"), "hemizygous")
        self.assertEqual(infer_zygosity("Tg(Mov1/Tyr)OVE876BPay/Y"), "hemizygous")
        self.assertEqual(
            infer_zygosity("Gene1<tm1>/Gene1<+>,Gene2<tm1>/Y"),
            "complex",
        )

    def test_direction_compatibility_distinguishes_unknown_and_conflict(self) -> None:
        self.assertIsNone(direction_compatibility("unknown", "increased"))
        self.assertEqual(direction_compatibility("increased", "increased"), 1.0)
        self.assertEqual(direction_compatibility("increased", "decreased"), -1.0)

    def test_genetic_mechanism_prefers_heterozygous_null_for_haploinsufficiency(self) -> None:
        disease = load_diseases(MISSION_DIR / "examples/diseases.tsv")["EXAMPLE:1"][0]
        models = load_models(MISSION_DIR / "examples/models.tsv")
        homo = genetic_compatibility(disease, models["MGI:MODEL_A"])
        hetero = genetic_compatibility(disease, models["MGI:MODEL_B"])
        self.assertEqual(homo[1], "conflicting")
        self.assertEqual(hetero[1], "concordant")
        self.assertGreater(hetero[0], homo[0])

    def test_genetic_mechanism_supports_hemizygous_loss_of_function(self) -> None:
        disease = DiseasePhenotype(
            disease_id="OMIM:1",
            disease_name="X-linked fixture",
            gene_symbol="GENE1",
            hpo_id="HP:TEST0001",
            inheritance="X-linked recessive",
            molecular_mechanism="hemizygous_loss_of_function",
        )
        model = MouseModel(
            model_id="MGI:1",
            model_name="Gene1 null/Y",
            mouse_gene_symbol="Gene1",
            allele_type="null/knockout",
            zygosity="hemizygous",
        )
        score = genetic_compatibility(disease, model)
        self.assertEqual(score[0], 1.0)
        self.assertEqual(score[1], "concordant")

    def test_example_pipeline_ranks_mechanism_concordant_model_first(self) -> None:
        example = MISSION_DIR / "examples"
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_pipeline(
                diseases_path=example / "diseases.tsv",
                models_path=example / "models.tsv",
                model_phenotypes_path=example / "model_phenotypes.tsv",
                mp_hpo_mapping_path=example / "mp_hpo_mapping.tsv",
                output_dir=directory,
                config_path=example / "config.json",
                hpo_mechanisms_path=example / "hpo_mechanisms.tsv",
                mp_mechanisms_path=example / "mp_mechanisms.tsv",
                cell_catalog_path=example / "cell_catalog.tsv",
                go_classification_path=example / "go_classification.tsv",
                cl_obo_path=example / "cl.obo",
                go_obo_path=example / "go.obo",
                cre_drivers_path=example / "cre_drivers.tsv",
            )
            with (Path(directory) / "model_rankings.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["model_id"], "MGI:MODEL_B")
            self.assertEqual(rows[0]["genetic_class"], "concordant")
            self.assertGreater(float(rows[0]["cpo_score"]), float(rows[1]["cpo_score"]))
            self.assertEqual(manifest["counts"]["ranking_rows"], 2)
            evaluation = json.loads((Path(directory) / "evaluation_summary.json").read_text())
            self.assertEqual(evaluation["ablations"]["surface_only"]["recall_at_1"], 0.0)
            self.assertEqual(evaluation["ablations"]["full"]["recall_at_1"], 1.0)

    def test_tested_normal_is_not_treated_as_missing(self) -> None:
        example = MISSION_DIR / "examples"
        with tempfile.TemporaryDirectory() as directory:
            run_pipeline(
                diseases_path=example / "diseases.tsv",
                models_path=example / "models.tsv",
                model_phenotypes_path=example / "model_phenotypes.tsv",
                mp_hpo_mapping_path=example / "mp_hpo_mapping.tsv",
                output_dir=directory,
                config_path=example / "config.json",
                hpo_mechanisms_path=example / "hpo_mechanisms.tsv",
                mp_mechanisms_path=example / "mp_mechanisms.tsv",
                cell_catalog_path=example / "cell_catalog.tsv",
                go_classification_path=example / "go_classification.tsv",
                cl_obo_path=example / "cl.obo",
                go_obo_path=example / "go.obo",
                cre_drivers_path=example / "cre_drivers.tsv",
            )
            with (Path(directory) / "phenotype_evidence.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            target = next(
                row for row in rows if row["model_id"] == "MGI:MODEL_A" and row["hpo_id"] == "HP:TEST0003"
            )
            self.assertEqual(target["assay_state"], "tested_normal")


if __name__ == "__main__":
    unittest.main()
