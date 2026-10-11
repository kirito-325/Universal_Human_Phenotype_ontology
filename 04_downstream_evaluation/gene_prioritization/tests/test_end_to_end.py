from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellphenorank.config import load_config
from cellphenorank.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    def test_synthetic_known_gene_moves_up(self) -> None:
        config = load_config(ROOT / "configs/example.yaml")
        config["null_models"]["enabled"] = False
        config["null_models"]["iterations"] = 0
        config["plots"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temporary:
            run_pipeline(config, temporary)
            rankings = Path(temporary) / "reranking/final_gene_rankings.tsv"
            rows = rankings.read_text(encoding="utf-8").splitlines()
            header = rows[0].split("\t")
            first = dict(zip(header, rows[1].split("\t")))
            self.assertEqual(first["gene_symbol"], "GENEB")
            self.assertEqual(first["base_rank"], "2")
            self.assertEqual(first["final_rank"], "1")
            self.assertTrue((Path(temporary) / "benchmark/benchmark_metrics.tsv").is_file())
            self.assertTrue((Path(temporary) / "report/report.html").is_file())


if __name__ == "__main__":
    unittest.main()
