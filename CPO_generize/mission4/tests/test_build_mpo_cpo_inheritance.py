from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_mpo_cpo_inheritance import build_inheritance


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class BuildMpoCpoInheritanceTest(unittest.TestCase):
    def test_leaf_relation_reaches_every_ancestor_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obo = root / "mp.obo"
            obo.write_text(
                """format-version: 1.2

[Term]
id: MP:0000001
name: mammalian phenotype

[Term]
id: MP:1000001
name: top level
is_a: MP:0000001 ! mammalian phenotype

[Term]
id: MP:1000002
name: middle one
is_a: MP:1000001 ! top level

[Term]
id: MP:1000003
name: middle two
is_a: MP:1000001 ! top level

[Term]
id: MP:1000004
name: linked leaf
is_a: MP:1000002 ! middle one
is_a: MP:1000003 ! middle two

[Term]
id: MP:1000005
name: unlinked leaf
is_a: MP:1000002 ! middle one

[Term]
id: MP:1000006
name: second linked leaf
is_a: MP:1000002 ! middle one

[Term]
id: MP:0002873
name: normal phenotype
is_a: MP:0000001 ! mammalian phenotype

[Term]
id: MP:0002169
name: no abnormal phenotype detected
is_a: MP:0002873 ! normal phenotype
""",
                encoding="utf-8",
            )
            relation_fields = (
                "relation_id",
                "mp_id",
                "relation_scope",
                "relation_type",
                "direction",
                "cl_id",
                "cl_label",
                "cell_text",
                "go_id",
                "go_label",
                "process_text",
                "quality_text",
            )
            leaf_relations = root / "leaf_relations.tsv"
            relation = {
                "relation_id": "relation_1",
                "mp_id": "MP:1000004",
                "relation_scope": "direct",
                "relation_type": "has_constituent_cellular_phenotype",
                "direction": "decreased",
                "cl_id": "CL:0000001",
                "cl_label": "cell",
                "cell_text": "cells",
                "go_id": "GO:0000001",
                "go_label": "process",
                "process_text": "process activity",
                "quality_text": "reduced",
            }
            second_relation = {
                **relation,
                "relation_id": "relation_2",
                "mp_id": "MP:1000006",
                "cell_text": "same cell with alternate wording",
                "process_text": "same process with alternate wording",
            }
            write_tsv(
                leaf_relations, relation_fields, [relation, second_relation]
            )

            leaf_status = root / "leaf_status.tsv"
            write_tsv(
                leaf_status,
                ("mp_id", "status"),
                [
                    {"mp_id": "MP:1000004", "status": "linked"},
                    {"mp_id": "MP:1000005", "status": "no_candidate"},
                    {"mp_id": "MP:1000006", "status": "linked"},
                ],
            )

            all_relations = root / "all_relations.tsv"
            all_fields = ("mp_id", "relation_scope")
            write_tsv(
                all_relations,
                all_fields,
                [
                    {"mp_id": "MP:1000004", "relation_scope": "direct"},
                    {"mp_id": "MP:1000006", "relation_scope": "direct"},
                    {"mp_id": "MP:1000002", "relation_scope": "direct"},
                ],
            )

            middle_decisions = root / "middle.tsv"
            middle_fields = (
                "mp_id",
                "needs_parent_literature",
                "selected_for_relationship",
                "selection_reason",
            )
            write_tsv(
                middle_decisions,
                middle_fields,
                [
                    {
                        "mp_id": "MP:1000002",
                        "needs_parent_literature": "true",
                        "selected_for_relationship": "true",
                        "selection_reason": "semantic_parent_literature_needed",
                    },
                    {
                        "mp_id": "MP:1000003",
                        "needs_parent_literature": "false",
                        "selected_for_relationship": "false",
                        "selection_reason": "delegable_without_direct_mgi",
                    },
                ],
            )

            output = root / "output"
            manifest = build_inheritance(
                obo,
                leaf_relations,
                leaf_status,
                all_relations,
                middle_decisions,
                output,
            )

            provenance = read_tsv(output / "mpo_cpo_inheritance_provenance.tsv")
            first_relation = [
                row for row in provenance if row["source_relation_id"] == "relation_1"
            ]
            by_target = {row["target_mp_id"]: row for row in first_relation}
            self.assertEqual(
                set(by_target),
                {"MP:1000002", "MP:1000003", "MP:1000001", "MP:0000001"},
            )
            self.assertEqual(by_target["MP:1000001"]["distance"], "2")
            self.assertEqual(by_target["MP:1000001"]["shortest_path_count"], "2")
            self.assertEqual(by_target["MP:0000001"]["distance"], "3")

            assertions = read_tsv(output / "mpo_cpo_inherited_assertions.tsv")
            self.assertEqual(len(assertions), 4)
            root_assertion = next(
                row for row in assertions if row["target_mp_id"] == "MP:0000001"
            )
            self.assertEqual(root_assertion["support_leaf_count"], "2")
            self.assertEqual(root_assertion["support_relation_count"], "2")
            self.assertEqual(root_assertion["eligible_descendant_leaf_count"], "3")
            self.assertEqual(root_assertion["complete_cpo"], "true")

            statuses = {
                row["mp_id"]: row
                for row in read_tsv(output / "mpo_cpo_inheritance_term_status.tsv")
            }
            self.assertEqual(len(statuses), 9)
            self.assertEqual(statuses["MP:1000002"]["profile_mode"], "self_and_inherited")
            self.assertEqual(statuses["MP:1000003"]["profile_mode"], "inherited_only")
            self.assertEqual(
                statuses["MP:0002873"]["inheritance_status"],
                "not_applicable_normal_branch",
            )
            self.assertEqual(manifest["counts"]["inheritance_provenance_count"], 7)


if __name__ == "__main__":
    unittest.main()
