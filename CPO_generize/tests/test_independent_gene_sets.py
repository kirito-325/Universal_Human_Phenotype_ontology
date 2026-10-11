from __future__ import annotations

import csv
import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cell_marker.export_cell_markers import export_cell_markers
from GO_function.export_go_functions import export_go_functions


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class CellMarkerExportTests(unittest.TestCase):
    @staticmethod
    def write_gene_resources(
        root: Path,
        hgnc_rows: list[tuple[str, str, str, str]],
        ortholog_rows: list[tuple[str, ...]],
    ) -> tuple[Path, Path]:
        hgnc_path = root / "hgnc.tsv"
        with hgnc_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["symbol", "status", "prev_symbol", "alias_symbol"])
            writer.writerows(hgnc_rows)
        ortholog_path = root / "orthologs.tsv"
        with ortholog_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                ["mouse_symbol", "mouse_mgi_id", "human_symbol", "mapping_type"]
            )
            for row in ortholog_rows:
                if len(row) == 3:
                    mouse_symbol, human_symbol, mapping_type = row
                    writer.writerow([mouse_symbol, "", human_symbol, mapping_type])
                else:
                    writer.writerow(row)
        return hgnc_path, ortholog_path

    def test_genes_are_normalized_and_cleaned_before_top_n(self) -> None:
        catalog_rows = [
            {
                "order": "1",
                "rank": "root",
                "branch": "root",
                "cl_id": "CL:0000000",
                "cl_name": "cell",
                "catalog_label": "Cell",
                "catalog_label_zh": "细胞",
                "parent_id": "",
                "parent_label": "",
            },
            {
                "order": "2",
                "rank": "type",
                "branch": "blood_immune",
                "cl_id": "CL:0000001",
                "cl_name": "test cell",
                "catalog_label": "Test cell",
                "catalog_label_zh": "测试细胞",
                "parent_id": "CL:0000000",
                "parent_label": "Cell",
            },
        ]
        marker_fields = [
            "catalog_order",
            "catalog_id",
            "catalog_label",
            "catalog_cl_name",
            "catalog_label_zh",
            "parent_id",
            "parent_label",
            "branch",
            "catalog_rank",
            "marker_rank",
            "gene",
            "marker_score",
            "statistics_score",
            "signatures_score",
        ]
        marker_rows = [
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "1",
                "GENE1",
                "5",
                "3",
                "2",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "2",
                "Gene2",
                "4",
                "2",
                "2",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "3",
                "2020-09-11 00:00:00",
                "4",
                "2",
                "2",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "4",
                "Gene1",
                "3",
                "2",
                "1",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "5",
                "OLD3",
                "2",
                "1",
                "1",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "6",
                "ALIAS4",
                "2",
                "1",
                "1",
            ],
            [
                "2",
                "CL:0000001",
                "Test cell",
                "test cell",
                "测试细胞",
                "CL:0000000",
                "Cell",
                "blood_immune",
                "type",
                "7",
                "GENE5",
                "1",
                "1",
                "0",
            ],
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path = root / "markers.tsv"
            with marker_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(marker_fields)
                writer.writerows(marker_rows)
            hgnc_path, ortholog_path = self.write_gene_resources(
                root,
                [
                    ("GENE1", "Approved", "", ""),
                    ("GENE2", "Approved", "", ""),
                    ("GENE3", "Approved", "OLD3", ""),
                    ("GENE4", "Approved", "", "ALIAS4"),
                    ("GENE5", "Approved", "", ""),
                ],
                [
                    ("Gene1", "GENE1", "one2one"),
                    ("Gene2", "GENE2", "one2one"),
                ],
            )
            counts = export_cell_markers(
                marker_path,
                catalog_rows,
                root / "output",
                top_markers_per_cell=4,
                human_gene_path=hgnc_path,
                ortholog_path=ortholog_path,
                mouse_gene_path=None,
            )
            summary = read_tsv(root / "output/cell_marker_summary.tsv")
            rejected = read_tsv(root / "output/cell_marker_rejected_genes.tsv.gz")
            with gzip.open(
                root / "output/cell_marker_genes.tsv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                genes = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(counts["catalog_cells"], 2)
        self.assertEqual(counts["input_marker_rows"], 7)
        self.assertEqual(counts["selected_marker_rows"], 4)
        self.assertEqual(counts["excluded_marker_rows"], 3)
        self.assertEqual(counts["excluded_non_gene_rows"], 1)
        self.assertEqual(counts["excluded_duplicate_human_gene_candidates"], 1)
        self.assertEqual(counts["excluded_after_top_n_candidates"], 1)
        self.assertEqual(counts["top_markers_per_cell"], 4)
        self.assertEqual(counts["unique_cell_marker_pairs"], 4)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["marker_gene_count"], "0")
        self.assertEqual(summary[1]["marker_gene_count"], "4")
        self.assertEqual(
            [row["gene"] for row in genes],
            ["GENE1", "GENE2", "GENE3", "GENE4"],
        )
        self.assertEqual(
            [row["marker_rank"] for row in genes], ["1", "2", "3", "4"]
        )
        self.assertEqual(
            [row["source_marker_rank"] for row in genes], ["1", "2", "5", "6"]
        )
        self.assertEqual(genes[1]["source_gene"], "Gene2")
        self.assertEqual(genes[1]["gene_mapping_method"], "mouse_ortholog")
        self.assertEqual(genes[1]["ortholog_mapping_type"], "one2one")
        self.assertEqual(genes[2]["gene_mapping_method"], "hgnc_previous_symbol")
        self.assertEqual(genes[3]["gene_mapping_method"], "hgnc_alias_symbol")
        self.assertEqual(rejected[0]["source_gene"], "2020-09-11 00:00:00")
        self.assertEqual(
            rejected[0]["rejection_reason"], "not_hgnc_or_mouse_ortholog_gene"
        )

    def test_one_to_many_mouse_orthologs_expand_and_ambiguous_aliases_reject(self) -> None:
        catalog_rows = [
            {
                "order": "1",
                "rank": "type",
                "branch": "test",
                "cl_id": "CL:1",
                "cl_name": "test cell",
                "catalog_label": "Test cell",
                "catalog_label_zh": "",
                "parent_id": "",
                "parent_label": "",
            }
        ]
        fields = [
            "catalog_order",
            "catalog_id",
            "catalog_label",
            "catalog_cl_name",
            "catalog_label_zh",
            "parent_id",
            "parent_label",
            "branch",
            "catalog_rank",
            "marker_rank",
            "gene",
        ]
        base = ["1", "CL:1", "Test cell", "test cell", "", "", "", "test", "type"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path = root / "markers.tsv"
            with marker_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerow(base + ["1", "OldMulti"])
                writer.writerow(base + ["2", "SHARED"])
            hgnc_path, ortholog_path = self.write_gene_resources(
                root,
                [
                    ("GENE1", "Approved", "", "SHARED"),
                    ("GENE2", "Approved", "", "SHARED"),
                    ("GENE3", "Approved", "", ""),
                    ("GENE4", "Approved", "", ""),
                ],
                [
                    ("GeneMulti", "MGI:1", "GENE3", "one2many"),
                    ("GeneMulti", "MGI:1", "GENE4", "one2many"),
                ],
            )
            mouse_gene_path = root / "mgi_markers.tsv"
            with mouse_gene_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "MGI Accession ID",
                        "Marker Symbol",
                        "Marker Type",
                        "Marker Synonyms (pipe-separated)",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "MGI Accession ID": "MGI:1",
                        "Marker Symbol": "GeneMulti",
                        "Marker Type": "Gene",
                        "Marker Synonyms (pipe-separated)": "OldMulti|OlderMulti",
                    }
                )
            counts = export_cell_markers(
                marker_path,
                catalog_rows,
                root / "output",
                human_gene_path=hgnc_path,
                ortholog_path=ortholog_path,
                mouse_gene_path=mouse_gene_path,
            )
            with gzip.open(
                root / "output/cell_marker_genes.tsv.gz", "rt", encoding="utf-8"
            ) as handle:
                genes = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual([row["gene"] for row in genes], ["GENE3", "GENE4"])
        self.assertTrue(
            all(row["gene_mapping_method"] == "mouse_ortholog_alias" for row in genes)
        )
        self.assertEqual(counts["excluded_ambiguous_gene_rows"], 1)
        self.assertEqual(counts["mouse_human_ortholog_pairs"], 2)
        self.assertEqual(counts["mouse_ortholog_aliases"], 2)


class GoFunctionExportTests(unittest.TestCase):
    @staticmethod
    def gaf_row(
        gene: str,
        relation: str,
        go_id: str,
        evidence: str,
        aspect: str = "P",
        taxon: str = "taxon:9606",
    ) -> str:
        return "\t".join(
            [
                "UniProtKB",
                f"P-{gene}",
                gene,
                relation,
                go_id,
                "PMID:1",
                evidence,
                "",
                aspect,
                gene,
                "",
                "protein",
                taxon,
                "20260722",
                "TEST",
            ]
        )

    def test_only_direct_positive_human_bp_annotations_are_retained(self) -> None:
        lines = [
            "!gaf-version: 2.2",
            "!date-generated: 2026-07-22",
            self.gaf_row("GENE1", "involved_in", "GO:0000001", "EXP"),
            self.gaf_row("GENE1", "involved_in", "GO:0000001", "IEA"),
            self.gaf_row("GENE2", "NOT|involved_in", "GO:0000001", "EXP"),
            self.gaf_row("GENE3", "enables", "GO:0000001", "EXP", aspect="F"),
            self.gaf_row(
                "GENE4",
                "involved_in",
                "GO:0000002",
                "EXP",
                taxon="taxon:10090",
            ),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gaf_path = root / "test.gaf"
            gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts = export_go_functions(
                gaf_path,
                {"GO:0000001": "process one", "GO:0000002": "process two"},
                root / "output",
            )
            gene_rows = read_tsv(root / "output/go_function_genes.tsv")
            summary_rows = read_tsv(root / "output/go_function_summary.tsv")

        self.assertEqual(counts["retained_direct_annotations"], 2)
        self.assertEqual(counts["unique_go_function_gene_pairs"], 1)
        self.assertEqual(len(gene_rows), 1)
        self.assertEqual(gene_rows[0]["annotation_count"], "2")
        self.assertEqual(gene_rows[0]["evidence_codes"], "EXP|IEA")
        self.assertEqual(gene_rows[0]["has_experimental_evidence"], "true")
        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(summary_rows[0]["human_function_gene_count"], "1")
        self.assertEqual(summary_rows[1]["human_function_gene_count"], "0")


if __name__ == "__main__":
    unittest.main()
