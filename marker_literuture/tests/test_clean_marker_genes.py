from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "marker_statistics"))

import clean_marker_genes as cleaner  # noqa: E402


def references() -> cleaner.GeneReferences:
    return cleaner.GeneReferences(
        human_ids={"CD3D": "HGNC:1673", "LYZ": "HGNC:6740", "PRKN": "HGNC:8607"},
        human_folded={"cd3d": "CD3D", "lyz": "LYZ", "prkn": "PRKN"},
        human_aliases={},
        human_previous={"PARK2": "PRKN"},
        human_ensembl={"ENSG0001": "CD3D"},
        mouse_symbols={"Cd3d", "Lyz2", "NoHuman"},
        mouse_folded={"cd3d": "Cd3d", "lyz2": "Lyz2", "nohuman": "NoHuman"},
        mouse_aliases={"T3d": "Cd3d"},
        mouse_ensembl={"ENSMUSG0001": "Cd3d"},
        mouse_mgi_ids={"MGI:1": "Cd3d"},
        mouse_to_human={
            "Cd3d": cleaner.OrthologTarget("CD3D", "HGNC:1673", "10", "1m:1h"),
            "Lyz2": cleaner.OrthologTarget("LYZ", "HGNC:6740", "11", "2m:1h"),
        },
        ambiguous_human_aliases=set(),
        ambiguous_mouse_aliases=set(),
        ambiguous_mouse_orthologs=set(),
        manifest={},
    )


class CleanMarkerGeneTests(unittest.TestCase):
    def test_human_and_mouse_symbols_are_standardized_conservatively(self) -> None:
        refs = references()

        human = cleaner.decide_gene("cd3d", "human", refs)
        old = cleaner.decide_gene("PARK2", "human", refs)
        mouse = cleaner.decide_gene("Cd3d", "mouse", refs)
        upper_mouse = cleaner.decide_gene("LYZ2", "mouse", refs)
        unsupported = cleaner.decide_gene("cd3d", "other", refs)
        no_ortholog = cleaner.decide_gene("NoHuman", "mouse", refs)

        self.assertEqual((human.gene, human.method), ("CD3D", "hgnc_case_normalized"))
        self.assertEqual((old.gene, old.method), ("PRKN", "hgnc_previous_symbol"))
        self.assertEqual((mouse.gene, mouse.reason), ("CD3D", "mouse_human_ortholog"))
        self.assertEqual((upper_mouse.gene, upper_mouse.reference), ("LYZ", "MGI-HOM:11;2m:1h;HGNC:6740"))
        self.assertEqual(unsupported.reason, "unsupported_nonhuman_species")
        self.assertEqual(no_ortholog.reason, "no_unambiguous_human_ortholog")

    def test_unknown_species_accepts_only_case_consensus(self) -> None:
        refs = references()

        consensus = cleaner.decide_gene("cD3d", "unknown", refs)
        rejected = cleaner.decide_gene("prkn", "unknown", refs)

        self.assertTrue(consensus.accepted)
        self.assertEqual(consensus.gene, "CD3D")
        self.assertEqual(consensus.method, "hgnc_mgi_case_consensus")
        self.assertFalse(rejected.accepted)

    def test_record_outputs_keep_source_gene_and_audit_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary) / "paper"
            output = record / "agent_standardized"
            output.mkdir(parents=True)
            (record / "metadata.json").write_text(
                json.dumps({"id": "doi:test", "title": "mouse atlas"}), encoding="utf-8"
            )
            (output / "analysis.json").write_text(
                json.dumps(
                    {"paper_context": {"batch_contexts": [{"species": "mouse"}]}}
                ),
                encoding="utf-8",
            )
            with (output / cleaner.STATISTICS_INPUT).open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["cell_cluster", "gene", "p_val", "avg_logFC", "pct.1", "pct.2", "p_val_adj"]
                )
                writer.writerow(["T cell", "Cd3d", "0.01", "1", "0.5", "0.1", "0.02"])
                writer.writerow(["T cell", "NoHuman", "0.01", "1", "0.5", "0.1", "0.02"])
            (output / cleaner.SIGNATURES_INPUT).write_text(
                "T cell\nLyz2\n", encoding="utf-8"
            )

            summary = cleaner.process_record(record, references())

            with (output / cleaner.STATISTICS_OUTPUT).open(
                encoding="utf-8", newline=""
            ) as handle:
                statistics = list(csv.DictReader(handle))
            with (output / cleaner.SIGNATURES_OUTPUT).open(
                encoding="utf-8", newline=""
            ) as handle:
                signatures = list(csv.DictReader(handle, delimiter="\t"))
            audit = (output / cleaner.AUDIT_OUTPUT).read_text(encoding="utf-8")

            self.assertEqual(summary["paper_species"], "mouse")
            self.assertEqual(statistics[0]["gene"], "CD3D")
            self.assertEqual(statistics[0]["source_gene"], "Cd3d")
            self.assertEqual(signatures[0]["gene"], "LYZ")
            self.assertIn("no_unambiguous_human_ortholog", audit)


if __name__ == "__main__":
    unittest.main()
