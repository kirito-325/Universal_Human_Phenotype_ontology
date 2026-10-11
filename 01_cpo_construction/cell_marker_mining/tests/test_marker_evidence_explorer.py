from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "marker_statistics"))

import build_marker_evidence_explorer as builder  # noqa: E402
import serve_marker_evidence_explorer as server  # noqa: E402


class MarkerEvidenceExplorerTests(unittest.TestCase):
    def write_tsv(self, path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_index_tracks_alias_marker_and_paper_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.tsv"
            mapping = root / "mapping.tsv"
            markers = root / "markers.tsv"
            opportunities = root / "opportunities.tsv"
            summary = root / "summary.json"
            papers = root / "papers"
            output = root / "explorer.sqlite3"
            self.write_tsv(
                catalog,
                [
                    "order", "level", "rank", "branch", "cl_id", "cl_name",
                    "catalog_label", "catalog_label_zh", "parent_id", "parent_label",
                ],
                [
                    {
                        "order": 1, "level": 0, "rank": "root", "branch": "root",
                        "cl_id": "CL:0", "cl_name": "cell", "catalog_label": "Cell",
                        "catalog_label_zh": "细胞", "parent_id": "", "parent_label": "",
                    },
                    {
                        "order": 2, "level": 1, "rank": "type", "branch": "immune",
                        "cl_id": "CL:1", "cl_name": "macrophage", "catalog_label": "Macrophage",
                        "catalog_label_zh": "巨噬细胞", "parent_id": "CL:0", "parent_label": "Cell",
                    },
                ],
            )
            self.write_tsv(
                mapping,
                [
                    "cell_name", "decision", "catalog_id", "confidence", "top_score",
                    "paper_count", "source_types",
                ],
                [
                    {
                        "cell_name": "Mac", "decision": "match", "catalog_id": "CL:1",
                        "confidence": "high", "top_score": 0.9, "paper_count": 1,
                        "source_types": "both",
                    },
                    {
                        "cell_name": "Mixed", "decision": "review", "catalog_id": "",
                        "confidence": "low", "top_score": 0.8, "paper_count": 1,
                        "source_types": "both",
                    },
                ],
            )
            marker_fields = [
                "catalog_id", "gene", "marker_rank", "marker_score",
                "statistics_score", "signatures_score",
            ]
            self.write_tsv(
                markers,
                marker_fields,
                [
                    {"catalog_id": "CL:1", "gene": "G1", "marker_rank": 1, "marker_score": 2, "statistics_score": 1, "signatures_score": 1},
                    {"catalog_id": "CL:1", "gene": "G2", "marker_rank": 2, "marker_score": 1, "statistics_score": 1, "signatures_score": 0},
                ],
            )
            self.write_tsv(
                opportunities,
                [
                    "catalog_id", "statistics_opportunities", "signatures_opportunities",
                    "total_opportunities",
                ],
                [
                    {
                        "catalog_id": "CL:1", "statistics_opportunities": 1,
                        "signatures_opportunities": 1, "total_opportunities": 2,
                    }
                ],
            )
            summary.write_text("{}", encoding="utf-8")
            record = papers / "paper_one"
            standardized = record / "agent_standardized"
            standardized.mkdir(parents=True)
            (record / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": "doi:paper.one", "title": "Paper one", "journal": "Cell",
                        "publication_year": 2025, "doi": "paper.one",
                        "url": "https://doi.org/paper.one",
                    }
                ),
                encoding="utf-8",
            )
            (standardized / builder.STATISTICS_NAME).write_text(
                "cell_cluster,gene,source_gene,source_species,gene_mapping_method,gene_mapping_reference\n"
                "Mac,G1,G1,human,hgnc_approved_exact,HGNC:1\n"
                "Mac,G1,G1,human,hgnc_approved_exact,HGNC:1\n"
                "Mac,G2,G2,human,hgnc_approved_exact,HGNC:2\n"
                "Mixed,GX,Gx,mouse,mgi_symbol_exact,MGI-HOM:1\n",
                encoding="utf-8",
            )
            (standardized / builder.SIGNATURES_NAME).write_text(
                "cell_cluster\tgene\tsource_gene\tsource_species\tgene_mapping_method\tgene_mapping_reference\n"
                "Mac\tG1\tG1\thuman\thgnc_approved_exact\tHGNC:1\n"
                "Mixed\tGX\tGx\tmouse\tmgi_symbol_exact\tMGI-HOM:1\n",
                encoding="utf-8",
            )

            result = builder.build_database(
                catalog, mapping, markers, opportunities, summary, papers, output
            )
            store = server.ExplorerStore(output)
            bootstrap = store.bootstrap()
            marker_result = store.markers("CL:1", "", "score", 100, 0)
            alias_result = store.alias(1, "", 100, 0)
            evidence_result = store.evidence("CL:1", "G1", None, 100)

            self.assertEqual(result["accepted_aliases"], 1)
            self.assertEqual(result["evidence_rows"], 3)
            self.assertEqual(bootstrap["metadata"]["cells_with_markers"], 1)
            self.assertEqual(marker_result["total"], 2)
            self.assertEqual(alias_result["total"], 2)
            self.assertEqual(len(evidence_result["evidence"]), 1)
            self.assertEqual(evidence_result["evidence"][0]["statistics"], 1)
            self.assertEqual(evidence_result["evidence"][0]["signatures"], 1)
            self.assertEqual(evidence_result["evidence"][0]["source_genes"], "G1")
            self.assertEqual(
                evidence_result["evidence"][0]["mapping_methods"],
                "hgnc_approved_exact",
            )
            with sqlite3.connect(output) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM aliases").fetchone()[0], 1)

    def test_like_pattern_escapes_wildcards(self) -> None:
        self.assertEqual(server.like_pattern("A_B%"), "%A\\_B\\%%")


if __name__ == "__main__":
    unittest.main()
