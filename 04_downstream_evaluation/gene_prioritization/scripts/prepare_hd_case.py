#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FREQUENCY_WEIGHTS = {
    "HP:0040280": 2.0,
    "HP:0040281": 1.75,
    "HP:0040282": 1.3,
    "HP:0040283": 0.8,
    "HP:0040284": 0.3,
    "HP:0040285": 0.1,
}


def hpo_names(path: Path) -> dict[str, str]:
    names = {}
    current = ""
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                current = ""
            elif line.startswith("id: "):
                current = line[4:]
            elif current and line.startswith("name: "):
                names[current] = line[6:]
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpoa", type=Path, required=True)
    parser.add_argument("--hpo-obo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disease-id", default="ORPHA:399")
    parser.add_argument("--gene", default="HTT")
    args = parser.parse_args()
    names = hpo_names(args.hpo_obo)
    rows = {}
    with args.hpoa.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in reader:
            if row.get("database_id") != args.disease_id or row.get("aspect") != "P" or row.get("qualifier") == "NOT":
                continue
            hpo_id = row["hpo_id"]
            frequency = row.get("frequency", "")
            weight = FREQUENCY_WEIGHTS.get(frequency, 1.0)
            current = rows.get(hpo_id)
            if current is None or weight > current["weight"]:
                rows[hpo_id] = {
                    "disease_id": args.disease_id,
                    "disease_name": row.get("disease_name", "Huntington disease"),
                    "hpo_id": hpo_id,
                    "hpo_name": names.get(hpo_id, ""),
                    "weight": weight,
                    "is_core": weight >= 1.3,
                    "frequency": frequency,
                    "gene_symbol": args.gene,
                    "source": row.get("evidence", "") + ":" + row.get("reference", ""),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["disease_id", "disease_name", "hpo_id", "hpo_name", "weight", "is_core", "frequency", "gene_symbol", "source"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: row["hpo_id"]))


if __name__ == "__main__":
    main()
