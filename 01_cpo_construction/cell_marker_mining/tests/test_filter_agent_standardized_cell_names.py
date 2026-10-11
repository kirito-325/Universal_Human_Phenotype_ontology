from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import filter_agent_standardized_cell_names as cell_filter  # noqa: E402


class CellNameFilterTests(unittest.TestCase):
    def row(self, name: str, sources: str = "paper/agent_standardized/marker_statistics.csv") -> dict[str, str]:
        return {
            "cell_name": name,
            "source_types": "marker_statistics",
            "source_paths": sources,
        }

    def reason(
        self, name: str, *, genes: set[str] | None = None, suspicious: set[str] | None = None, protected: set[str] | None = None, sources: str = "paper/agent_standardized/marker_statistics.csv"
    ) -> str | None:
        return cell_filter.exclusion_reason(
            self.row(name, sources),
            genes or set(),
            suspicious or set(),
            protected or set(),
        )

    def test_keeps_real_cell_names(self) -> None:
        self.assertIsNone(self.reason("CD8+ tissue-resident memory T cell"))
        self.assertIsNone(self.reason("Disease-associated microglia (DAM)"))
        self.assertIsNone(self.reason("Luminal epithelial cell from treated tissue"))

    def test_catalog_cell_alias_overrides_gene_vocabulary_collision(self) -> None:
        protected = cell_filter.cell_label_variants("endothelial cell")
        self.assertIsNone(
            self.reason("Endothelial", genes={"endothelial"}, protected=protected)
        )
        self.assertEqual(
            self.reason("MKI67", genes={"mki67"}, protected=protected),
            "gene_symbol_in_cell_cluster",
        )
        pole = cell_filter.cell_label_variants("pole cell")
        self.assertEqual(
            self.reason("POLE", genes={"pole"}, protected=pole),
            "gene_symbol_in_cell_cluster",
        )
        self.assertIsNone(
            self.reason("OPC", genes={"opc"}, protected={"opc"})
        )

    def test_excludes_genomic_regions_and_pathways(self) -> None:
        self.assertEqual(self.reason("Intron (NM_001234, intron 2 of 4)"), "genomic_region_label")
        self.assertEqual(self.reason("KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY"), "pathway_or_module_label")
        self.assertEqual(self.reason("HALLMARK_INTERFERON_GAMMA_RESPONSE"), "pathway_or_module_label")

    def test_excludes_gene_lists_and_table_descriptions(self) -> None:
        self.assertEqual(
            self.reason("Predicted CAF target genes from NicheNet"),
            "gene_list_description",
        )
        self.assertEqual(
            self.reason("Supplementary Table 5. Differential expression analysis"),
            "table_or_figure_description",
        )

    def test_excludes_unresolved_and_comparison_labels(self) -> None:
        self.assertEqual(
            self.reason("Cluster 5 (cell type not resolved from provided text)"),
            "review_or_unresolved_label",
        )
        self.assertEqual(self.reason("NDMM_vs_MGUS"), "comparison_or_deg_label")

    def test_excludes_names_found_only_in_high_cardinality_source(self) -> None:
        suspicious = {"bad/agent_standardized/marker_signatures.csv"}
        self.assertEqual(
            self.reason(
                "Chondrogenic precursor",
                suspicious=suspicious,
                sources="bad/agent_standardized/marker_signatures.csv",
            ),
            "high_cardinality_source",
        )

    def test_keeps_name_with_an_independent_trusted_source(self) -> None:
        bad = "bad/agent_standardized/marker_signatures.csv"
        good = "good/agent_standardized/marker_statistics.csv"
        self.assertIsNone(
            self.reason("Neutrophil", suspicious={bad}, sources=f"{bad};{good}")
        )

    def test_excludes_obvious_sample_and_symbol_labels(self) -> None:
        self.assertEqual(self.reason("145790 AML"), "sample_or_timepoint_label")
        self.assertEqual(self.reason("5 dpa"), "sample_or_timepoint_label")
        self.assertEqual(self.reason("Patient P1"), "sample_or_timepoint_label")
        self.assertEqual(self.reason("+"), "symbol_only_label")

    def test_excludes_modules_supplements_and_unresolved_clusters(self) -> None:
        self.assertEqual(self.reason("turquoise"), "gene_set_program_or_signature")
        self.assertEqual(self.reason("Geneset-B1"), "gene_set_program_or_signature")
        self.assertEqual(
            self.reason("Supplementary Data 2"), "table_or_figure_description"
        )
        self.assertEqual(self.reason("Cluster 0 (review)"), "review_or_unresolved_label")

    def test_excludes_descriptions_that_are_not_cell_names(self) -> None:
        self.assertEqual(
            self.reason("Effector-differentiation score (EDS) for T/NK cells"),
            "description_not_cell_name",
        )
        self.assertEqual(
            self.reason("UBA1-knockdown cell lines (U937, THP1)"),
            "description_not_cell_name",
        )


if __name__ == "__main__":
    unittest.main()
