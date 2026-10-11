from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from uhpo_common.paths import REPO_ROOT


class RepositoryStructureTests(unittest.TestCase):
    def test_scientific_packages_live_under_their_tasks(self):
        expected = {
            "GO_link": "01_cpo_construction/cl_go_linking",
            "celltype_configure": "01_cpo_construction/cl_go_linking",
            "cell_marker": "01_cpo_construction/cell_marker_mining",
            "literature_pipeline": "01_cpo_construction/cell_marker_mining",
            "cpo_pipeline": "03_mpo_hpo_mapping/cpo_profile_similarity",
            "semantic_cpo": "03_mpo_hpo_mapping/semantic_cpo_experiments",
            "cellphenorank": "04_downstream_evaluation/gene_prioritization",
            "mission3": "04_downstream_evaluation/mouse_model_prioritization",
            "cellphenorank_validation": "04_downstream_evaluation/single_cell_validation",
            "uhpo_prediction": "04_downstream_evaluation/cpo_disease_gene_prediction",
            "llm_api": "shared",
        }
        for module, directory in expected.items():
            with self.subTest(module=module):
                path = Path(importlib.import_module(module).__file__).resolve()
                self.assertTrue(path.is_relative_to(REPO_ROOT / directory), str(path))

    def test_current_matrix_config_does_not_depend_on_cwd(self):
        from cpo_pipeline.config import load_config, output_root, resource_path
        from cpo_pipeline.__main__ import _parser
        config_file = REPO_ROOT / "03_mpo_hpo_mapping/cpo_profile_similarity/config/cpo_v2.yaml"
        self.assertEqual(_parser().parse_args(["audit"]).config, config_file)
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                config = load_config(config_file)
                self.assertEqual(Path(config["_project_root"]), REPO_ROOT)
                self.assertTrue(resource_path(config, "hpo_self_cpo").is_relative_to(REPO_ROOT / "02_phenotype_cpo_mapping/hpo_cpo"))
                self.assertTrue(output_root(config).is_relative_to(REPO_ROOT / "03_mpo_hpo_mapping/cpo_profile_similarity"))
        finally:
            os.chdir(before)

    def test_semantic_experiment_resolves_benchmark_and_gene_task(self):
        from semantic_cpo.config import load_configs, resolve
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                resources, _, _ = load_configs()
                self.assertEqual(Path(resources["project_root"]), REPO_ROOT)
                benchmark = resolve(resources, resources["benchmark"]["models"])
                self.assertTrue(benchmark.is_relative_to(REPO_ROOT / "04_downstream_evaluation/mouse_model_prioritization"))
                self.assertEqual(resolve(resources, resources["mission3"]["project_root"]), REPO_ROOT / "04_downstream_evaluation/gene_prioritization")
        finally:
            os.chdir(before)

    def test_cli_imports_work_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in ("cpo_pipeline", "uhpo_prediction", "cellphenorank", "mission3", "GO_link.build_complete_scope"):
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
