from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from literature_pipeline.config import load_config
from literature_pipeline.cli import selected_records
from literature_pipeline.pubmed import build_pubmed_query
from literature_pipeline.records import build_record_lookup, merge_metadata, write_record


class PubMedDiscoveryTests(unittest.TestCase):
    def test_empty_topic_terms_omit_the_topic_clause(self) -> None:
        config = {
            "journals": [{"name": "Nature", "family": "Nature"}],
            "single_cell_terms": ["single-cell RNA-seq"],
            "immune_terms": ["immune"],
            "topic_terms": [],
            "exclude_publication_types": ["Review"],
        }
        query = build_pubmed_query(config, "2020-01-01", "2021-01-01")
        self.assertIn('"single-cell RNA-seq"[Title/Abstract]', query)
        self.assertNotIn('"immune"[Title/Abstract]', query)
        self.assertNotIn("()", query)

    def test_legacy_config_requires_immune_terms(self) -> None:
        config = {
            "journals": [{"name": "Nature", "family": "Nature"}],
            "single_cell_terms": ["single-cell RNA-seq"],
            "immune_terms": ["immune"],
        }
        query = build_pubmed_query(config, "2020-01-01", "2021-01-01")
        self.assertIn('"immune"[Title/Abstract]', query)

    def test_relative_config_inheritance_replaces_top_level_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.json").write_text(
                '{"journals": [{"name": "Cell"}], "topic_terms": ["immune"]}',
                encoding="utf-8",
            )
            (root / "child.json").write_text(
                '{"extends": "base.json", "topic_terms": []}', encoding="utf-8"
            )
            config = load_config(root / "child.json")
        self.assertEqual(config["journals"], [{"name": "Cell"}])
        self.assertEqual(config["topic_terms"], [])

    def test_screening_lists_are_merged_across_discovery_cohorts(self) -> None:
        existing = {
            "screening": {
                "query_matched": ["immune"],
                "immune_cell_types": ["T cell"],
                "discovery_cohorts": ["immune_single_cell"],
            }
        }
        incoming = {
            "screening": {
                "query_matched": ["single-cell atlas"],
                "immune_cell_types": [],
                "discovery_cohorts": ["all_cell_single_cell"],
            }
        }
        merged = merge_metadata(existing, incoming)
        screening = merged["screening"]
        self.assertEqual(screening["query_matched"], ["immune", "single-cell atlas"])
        self.assertEqual(screening["immune_cell_types"], ["T cell"])
        self.assertEqual(
            screening["discovery_cohorts"],
            ["immune_single_cell", "all_cell_single_cell"],
        )

    def test_record_lookup_deduplicates_incremental_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database"
            first = {
                "doi": "10.1/example",
                "pmid": "123",
                "title": "First title",
                "slug": "first_title",
                "screening": {"discovery_cohorts": ["immune_single_cell"]},
            }
            record_dir = write_record(database, first)
            lookup = build_record_lookup(database)
            second = {
                **first,
                "title": "Updated title",
                "screening": {"discovery_cohorts": ["all_cell_single_cell"]},
            }
            merged_dir = write_record(database, second, record_lookup=lookup)

            self.assertEqual(merged_dir, record_dir)
            self.assertEqual(len(list((database / "papers").iterdir())), 1)
            self.assertEqual(lookup["doi:10.1/example"], record_dir)

    def test_record_selection_filters_cohorts_before_sharding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database"
            for index, cohorts in enumerate(
                [
                    ["immune_single_cell", "all_cell_single_cell"],
                    ["all_cell_single_cell"],
                    ["all_cell_single_cell"],
                ]
            ):
                write_record(
                    database,
                    {
                        "pmid": str(index + 1),
                        "title": f"Paper {index}",
                        "slug": f"paper_{index}",
                        "screening": {"discovery_cohorts": cohorts},
                    },
                )
            args = Namespace(
                database=database,
                cohort="all_cell_single_cell",
                exclude_cohort="immune_single_cell",
                unprobed_only=True,
                stride=2,
                offset=0,
                limit=0,
            )
            records = selected_records(args)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1]["pmid"], "2")

    def test_record_selection_shards_before_dynamic_unprobed_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database"
            for index in range(4):
                write_record(
                    database,
                    {
                        "pmid": str(index + 1),
                        "title": f"Paper {index}",
                        "slug": f"paper_{index}",
                        "screening": {
                            "discovery_cohorts": ["all_cell_single_cell"]
                        },
                        "content_probe": {"pmc_status": "ok"} if index == 0 else {},
                    },
                )
            args = Namespace(
                database=database,
                cohort="all_cell_single_cell",
                exclude_cohort="",
                unprobed_only=True,
                stride=2,
                offset=0,
                limit=0,
            )
            records = selected_records(args)
        self.assertEqual([metadata["pmid"] for _, metadata in records], ["3"])


if __name__ == "__main__":
    unittest.main()
