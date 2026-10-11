from __future__ import annotations

import argparse
from pathlib import Path

from cellphenorank.io import read_tsv, value
from cellphenorank.ontology_embedding import build_embedding_artifact


def collect_ids(path: str | Path, names: tuple[str, ...]) -> set[str]:
    return {
        identifier
        for row in read_tsv(path)
        if (identifier := value(row, *names))
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen HPO/MP SapBERT embedding artifact"
    )
    parser.add_argument("--diseases", required=True)
    parser.add_argument("--model-phenotypes", required=True)
    parser.add_argument("--hpo-obo", required=True)
    parser.add_argument("--mpo-obo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--representation",
        choices=("label_cls", "exact_synonym_centroid_cls"),
        default="label_cls",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    manifest = build_embedding_artifact(
        hpo_obo=args.hpo_obo,
        mpo_obo=args.mpo_obo,
        hpo_ids=collect_ids(args.diseases, ("hpo_id", "hp_id")),
        mpo_ids=collect_ids(args.model_phenotypes, ("mpo_id", "mp_id")),
        model_path=args.model_path,
        output_path=args.output,
        representation=args.representation,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(manifest)


if __name__ == "__main__":
    main()
