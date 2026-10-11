from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MISSION_DIR = Path(__file__).resolve().parents[1]
RUNNER_DIR = (
    MISSION_DIR
    / "skills/build-rare-disease-mouse-benchmark/scripts"
)
sys.path.insert(0, str(RUNNER_DIR))

from run_codex_research import (  # noqa: E402
    build_codex_command,
    packet_validation_errors,
    parse_json_object,
    run_one,
    schema_validation_errors,
)
from replace_failed_candidates import choose_replacements  # noqa: E402
from select_candidates import top_branch  # noqa: E402


class BenchmarkRunnerTests(unittest.TestCase):
    def test_top_branch_is_deterministic_for_multiple_parents(self) -> None:
        self.assertEqual(
            top_branch(
                "HP:child",
                root_id="HP:root",
                parents={"HP:child": {"HP:branch_b", "HP:branch_a"}},
                direct_children={"HP:branch_a", "HP:branch_b"},
            ),
            "HP:branch_a",
        )

    def test_replacement_uses_next_same_stratum_reserve(self) -> None:
        candidates = [
            {
                "selection_rank": "1",
                "disease_id": "OMIM:1",
                "disease_name": "failed",
                "gene_symbol": "OLD",
                "inheritance": "Autosomal recessive inheritance",
                "phenotype_system_id": "HP:1",
                "phenotype_system": "system",
            },
            {
                "selection_rank": "2",
                "disease_id": "OMIM:2",
                "disease_name": "retained",
                "gene_symbol": "KEEP",
                "inheritance": "Autosomal dominant inheritance",
                "phenotype_system_id": "HP:2",
                "phenotype_system": "other",
            },
        ]
        base_reserve = {
            "disease_name": "reserve",
            "gene_symbol": "NEW",
            "inheritance": "Autosomal recessive inheritance",
            "inheritance_class": "autosomal_recessive",
            "phenotype_system_id": "HP:1",
            "phenotype_system": "system",
            "stratum_rank": "4",
            "selection_score": "7.5",
        }
        pool = [
            {**base_reserve, "disease_id": "OMIM:4", "stratum_rank": "5"},
            {**base_reserve, "disease_id": "OMIM:3"},
            {
                **base_reserve,
                "disease_id": "OMIM:9",
                "inheritance_class": "autosomal_dominant",
                "stratum_rank": "1",
            },
        ]
        selected, audit = choose_replacements(
            candidates,
            pool,
            [1],
            {1: {"errors": "molecular_mechanism_unknown"}},
            max_per_gene=2,
        )
        self.assertEqual(selected[0]["disease_id"], "OMIM:3")
        self.assertEqual(audit[0]["excluded_disease_id"], "OMIM:1")
        self.assertEqual(audit[0]["exclusion_errors"], "molecular_mechanism_unknown")

    def test_replacement_never_reintroduces_prior_exclusion(self) -> None:
        candidates = [
            {
                "selection_rank": "1",
                "disease_id": "OMIM:2",
                "disease_name": "failed replacement",
                "gene_symbol": "OLD2",
                "inheritance": "Autosomal dominant inheritance",
                "phenotype_system_id": "HP:1",
                "phenotype_system": "system",
            }
        ]
        reserve = {
            "disease_name": "reserve",
            "gene_symbol": "NEW",
            "inheritance": "Autosomal dominant inheritance",
            "inheritance_class": "autosomal_dominant",
            "phenotype_system_id": "HP:1",
            "phenotype_system": "system",
            "selection_score": "7.5",
        }
        selected, _ = choose_replacements(
            candidates,
            [
                {**reserve, "disease_id": "OMIM:1", "stratum_rank": "1"},
                {**reserve, "disease_id": "OMIM:3", "stratum_rank": "2"},
            ],
            [1],
            {1: {"errors": "failed"}},
            max_per_gene=2,
            excluded_disease_ids={"OMIM:1"},
        )
        self.assertEqual(selected[0]["disease_id"], "OMIM:3")

    def test_packet_validation_requires_exact_identity_and_model_coverage(self) -> None:
        packet = {
            "disease_id": "OMIM:1",
            "disease_name": "Exact Name",
            "gene_symbol": "GENE",
            "allowed_hpo_ids": ["HP:1", "HP:2"],
            "source_gold_models": [{"model_id": "MGI:1"}],
        }
        review = {
            "disease_id": "OMIM:1",
            "disease_name": "Exact name",
            "gene_symbol": "GENE",
            "core_hpo_ids": ["HP:3"],
            "excluded_hpo_ids": ["HP:3"],
            "gold_model_decisions": [],
        }
        errors = packet_validation_errors(review, packet)
        self.assertTrue(any("disease_name" in error for error in errors))
        self.assertTrue(any("outside packet" in error for error in errors))
        self.assertTrue(any("exactly cover" in error for error in errors))

    def test_research_schema_supports_hemizygous_loss_of_function(self) -> None:
        schema = json.loads(
            (
                MISSION_DIR
                / "skills/build-rare-disease-mouse-benchmark/assets/research_review.schema.json"
            ).read_text(encoding="utf-8")
        )
        mechanisms = schema["properties"]["mechanism_class"]["enum"]
        self.assertIn("hemizygous_loss_of_function", mechanisms)

    def test_parse_json_object_accepts_plain_and_single_fence(self) -> None:
        self.assertEqual(parse_json_object('{"value": "ok"}'), {"value": "ok"})
        self.assertEqual(
            parse_json_object('```json\n{"value": "ok"}\n```'),
            {"value": "ok"},
        )

    def test_parse_json_object_rejects_surrounding_prose(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_json_object('Result:\n{"value": "ok"}')

    def test_local_schema_validation_reports_json_path(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        }
        self.assertEqual(schema_validation_errors({"value": "ok"}, schema), [])
        errors = schema_validation_errors({"value": 3}, schema)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("$.value:"), errors)

    def test_codex_command_omits_provider_side_output_schema(self) -> None:
        command = build_codex_command(
            codex="codex",
            root=Path("/tmp/project"),
            last_message_path=Path("/tmp/last_message.txt"),
            model="",
            ignore_user_config=False,
        )
        self.assertNotIn("--output-schema", command)
        self.assertIn("--output-last-message", command)

    def test_run_one_preserves_raw_message_and_writes_validated_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            packet = root / "packet.json"
            packet.write_text("{}\n", encoding="utf-8")
            schema_path = root / "schema.json"
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            }
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            fake_codex = root / "fake-codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "path = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
                "path.write_text('{\\\"value\\\": \\\"ok\\\"}\\n', encoding='utf-8')\n"
                "print('{\\\"type\\\":\\\"error\\\",\\\"message\\\":\\\"502 Bad Gateway\\\"}')\n"
                "print('{\\\"type\\\":\\\"turn.completed\\\"}')\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)

            result = run_one(
                {
                    "selection_rank": 1,
                    "disease_id": "OMIM:1",
                    "packet_path": "packet.json",
                },
                root=root,
                batch=batch,
                prompt_template="Read {{PACKET_PATH}} and {{SCHEMA_PATH}}",
                schema_path=schema_path,
                schema=schema,
                codex=str(fake_codex),
                model="",
                ignore_user_config=False,
                timeout=30,
                retries=0,
                force=True,
                dry_run=False,
            )

            self.assertEqual(result["status"], "completed")
            output = batch / "raw_codex_outputs/01_OMIM_1"
            self.assertEqual(
                json.loads((output / "final.json").read_text(encoding="utf-8")),
                {"value": "ok"},
            )
            self.assertTrue((output / "attempts/01/last_message.txt").is_file())
            command = result["attempts"][0]["command"]
            self.assertNotIn("--output-schema", command)
            self.assertTrue(result["attempts"][0]["output_valid_schema"])
            self.assertTrue(result["attempts"][0]["transient_failure"])
            self.assertFalse(result["attempts"][0]["retryable_failure"])


if __name__ == "__main__":
    unittest.main()
