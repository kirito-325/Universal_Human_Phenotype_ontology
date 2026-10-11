#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    source = path.resolve()
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)}


def collapse_duplicate_genes(
    matrix: sparse.csr_matrix, genes: list[str]
) -> tuple[sparse.csr_matrix, list[str]]:
    unique = sorted(set(genes))
    if len(unique) == len(genes):
        return matrix, genes
    position = {gene: index for index, gene in enumerate(unique)}
    mapping = sparse.csr_matrix(
        (
            np.ones(len(genes), dtype=np.int8),
            (np.arange(len(genes), dtype=np.int32), np.asarray([position[gene] for gene in genes])),
        ),
        shape=(len(genes), len(unique)),
    )
    return (matrix @ mapping).tocsr(), unique


def aggregate_dataset(
    counts: sparse.csc_matrix,
    metadata: pd.DataFrame,
    genes: list[str],
    dataset_id: str,
    cohort: str,
    excluded_donors: set[str],
) -> ad.AnnData:
    eligible = (
        metadata["cohort"].astype(str).eq(cohort)
        & metadata["genotype"].astype(str).isin(["IPEX", "HD"])
        & ~metadata["donor"].astype(str).isin(excluded_donors)
    )
    selected = metadata.loc[eligible].copy()
    selected["matrix_index"] = np.flatnonzero(eligible.to_numpy())
    profiles = []
    obs_rows = []
    grouped = selected.groupby(
        ["genotype", "donor", "Cluster_Annotation_General"],
        sort=True,
        observed=True,
    )
    for (genotype, donor, cell_type), frame in grouped:
        indices = frame["matrix_index"].to_numpy(dtype=int)
        profiles.append(sparse.csr_matrix(counts[:, indices].sum(axis=1).T))
        obs_rows.append(
            {
                "dataset_id": dataset_id,
                "cohort": cohort,
                "donor_id": str(donor),
                "condition": "ipex" if genotype == "IPEX" else "control",
                "genotype": str(genotype),
                "cell_type": str(cell_type),
                "cell_count": len(indices),
                "profile_kind": "published_general_cell_type",
            }
        )
    for (genotype, donor), frame in selected.groupby(
        ["genotype", "donor"], sort=True, observed=True
    ):
        indices = frame["matrix_index"].to_numpy(dtype=int)
        profiles.append(sparse.csr_matrix(counts[:, indices].sum(axis=1).T))
        obs_rows.append(
            {
                "dataset_id": dataset_id,
                "cohort": cohort,
                "donor_id": str(donor),
                "condition": "ipex" if genotype == "IPEX" else "control",
                "genotype": str(genotype),
                "cell_type": "T_cell_all",
                "cell_count": len(indices),
                "profile_kind": "all_cd4_t_cells_aggregate",
            }
        )
    matrix = sparse.vstack(profiles).tocsr()
    matrix, unique_genes = collapse_duplicate_genes(matrix, genes)
    obs = pd.DataFrame(obs_rows)
    obs.index = [
        f"{row.dataset_id}::{row.donor_id}::{row.cell_type}"
        for row in obs.itertuples()
    ]
    var = pd.DataFrame(index=pd.Index(unique_genes, name="gene_symbol"))
    return ad.AnnData(X=matrix, obs=obs, var=var)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/single_cell/ipex_gse167976/raw",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=ROOT / "data/cases/ipex_case_frozen/case_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/single_cell/ipex_gse167976/processed",
    )
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    manifest_path = args.case_manifest.resolve()
    case_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not case_manifest.get("frozen"):
        raise ValueError("Refusing expression processing: case predictions are not frozen")
    if case_manifest.get("expression_matrix_loaded_before_freeze"):
        raise ValueError("Case manifest reports expression access before prediction freeze")
    paths = {
        "matrix": raw / "GSE167976_matrix.mtx.gz",
        "genes": raw / "GSE167976_genes.tsv.gz",
        "barcodes": raw / "GSE167976_barcodes.tsv.gz",
        "metadata": raw / "GSE167976_metadata.tsv.gz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = pd.read_csv(paths["metadata"], sep="\t", low_memory=False)
    genes = pd.read_csv(paths["genes"], sep="\t", header=None)[0].astype(str).tolist()
    barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None)[0].astype(str).tolist()
    if barcodes != metadata["cell_ID"].astype(str).tolist():
        raise ValueError("Barcode order does not exactly match metadata cell_ID order")
    with gzip.open(paths["matrix"], "rb") as handle:
        counts = scipy_io.mmread(handle).tocsc()
    if counts.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"Matrix shape {counts.shape} does not match genes/barcodes {(len(genes), len(barcodes))}"
        )
    datasets = [
        ("GSE167976_cohort1", "cohort_1", set()),
        ("GSE167976_cohort2_disjoint", "cohort_2", {"B4", "B5", "P4"}),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    summaries = []
    for dataset_id, cohort, excluded in datasets:
        data = aggregate_dataset(counts, metadata, genes, dataset_id, cohort, excluded)
        target = args.output_dir / f"{dataset_id}_donor_celltype_pseudobulk.h5ad"
        data.write_h5ad(target, compression="gzip")
        outputs[dataset_id] = record(target)
        for (condition, cell_type), frame in data.obs.groupby(
            ["condition", "cell_type"], observed=True
        ):
            summaries.append(
                {
                    "dataset_id": dataset_id,
                    "condition": condition,
                    "cell_type": cell_type,
                    "donors": int(frame["donor_id"].nunique()),
                    "minimum_cells": int(frame["cell_count"].min()),
                    "median_cells": float(frame["cell_count"].median()),
                    "maximum_cells": int(frame["cell_count"].max()),
                }
            )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "pseudobulk_summary.tsv", sep="\t", index=False
    )
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_manifest": record(manifest_path),
        "expression_loaded_after_prediction_freeze": True,
        "source": "NCBI_GEO_GSE167976",
        "inputs": {name: record(path) for name, path in paths.items()},
        "outputs": outputs,
        "matrix_shape_gene_by_cell": list(counts.shape),
        "matrix_nonzero_entries": int(counts.nnz),
        "donor_independence_rule": (
            "cohort_2 excludes repeated IPEX donors B4, B5, and P4; "
            "cohort_1 and cohort_2_disjoint share no donors"
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
