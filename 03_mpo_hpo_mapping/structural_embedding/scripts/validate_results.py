#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR / "src"))

from mpo_hpo_mapping.data import load_hpo_ontology  # noqa: E402


def _int(row: dict[str, str], field: str) -> int:
    try:
        return int(row.get(field, "0") or 0)
    except ValueError:
        return 0


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, "0") or 0.0)
    except ValueError:
        return 0.0


def _stage(row: dict[str, str], has_override: bool) -> int:
    if not _int(row, "exact_available_prediction"):
        return 0
    if _int(row, "exact_evidence_override"):
        return 0
    if _int(row, "derived_ancestor_distance"):
        return 1
    if not has_override and _float(row, "p_exact") >= 0.5:
        return 0
    if has_override and _float(row, "p_exact") >= 0.5:
        return 2
    return 3


def validate(result_dir: Path) -> dict[str, object]:
    manifest_path = result_dir / "training_manifest.json"
    mapping_path = result_dir / "mpo_hpo_mapping_final.tsv"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    hpo_edges = Path(manifest["inputs"]["hpo_edges"]["path"])
    ontology = load_hpo_ontology(hpo_edges)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "mp_id",
            "final_rank",
            "hp_id",
            "p_exact",
            "exact_available_prediction",
            "exact_evidence_override",
            "derived_ancestor_distance",
            "derived_from_exact_ids",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing mapping fields: {sorted(missing)}")
        for row in reader:
            groups[row["mp_id"]].append(row)

    errors: list[str] = []
    derived_rows = 0
    override_queries = 0
    exact_available_queries = 0
    for mp_id, rows in groups.items():
        rows.sort(key=lambda row: _int(row, "final_rank"))
        expected_ranks = list(range(1, len(rows) + 1))
        actual_ranks = [_int(row, "final_rank") for row in rows]
        if actual_ranks != expected_ranks:
            errors.append(f"{mp_id}: non-contiguous ranks {actual_ranks[:5]}")
        hp_ids = [row["hp_id"] for row in rows]
        if len(hp_ids) != len(set(hp_ids)):
            errors.append(f"{mp_id}: duplicate HPO candidates")

        has_override = any(_int(row, "exact_evidence_override") for row in rows)
        override_queries += int(has_override)
        exact_available_queries += int(
            any(_int(row, "exact_available_prediction") for row in rows)
        )
        stages = [_stage(row, has_override) for row in rows]
        if stages != sorted(stages):
            errors.append(f"{mp_id}: ranking stages are not monotonic: {stages}")

        for row in rows:
            distance = _int(row, "derived_ancestor_distance")
            if not distance:
                continue
            derived_rows += 1
            source_distances: list[int] = []
            for source in filter(None, row["derived_from_exact_ids"].split(";")):
                actual_distance = ontology.ancestors(source).get(row["hp_id"])
                if actual_distance is None:
                    errors.append(
                        f"{mp_id}: {row['hp_id']} is not an ancestor of {source}"
                    )
                else:
                    source_distances.append(actual_distance)
            if source_distances and min(source_distances) != distance:
                errors.append(
                    f"{mp_id}: minimum source distance for {row['hp_id']} is "
                    f"{min(source_distances)}, output says {distance}"
                )

    expected_terms = int(manifest["output"]["mapped_mpo_terms"])
    expected_rows = int(manifest["output"]["mapping_rows"])
    actual_rows = sum(len(rows) for rows in groups.values())
    if len(groups) != expected_terms:
        errors.append(f"mapped MPO count {len(groups)} != manifest {expected_terms}")
    if actual_rows != expected_rows:
        errors.append(f"mapping row count {actual_rows} != manifest {expected_rows}")

    return {
        "status": "passed" if not errors else "failed",
        "mapped_mpo_terms": len(groups),
        "mapping_rows": actual_rows,
        "exact_available_queries": exact_available_queries,
        "strong_evidence_override_queries": override_queries,
        "derived_ancestor_rows": derived_rows,
        "errors": errors[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an MPO-HPO result directory.")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    report = validate(args.result_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
