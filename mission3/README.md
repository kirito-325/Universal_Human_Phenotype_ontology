# CellPhenoRank

> Code-only snapshot: real datasets, frozen cohorts, matrices and outputs are not included. Reported results describe the original research runs. Local paths have been replaced with placeholders; see the repository [data guide](../docs/data.md) before running a full experiment. The new HPO/MPO agents have not been used to regenerate these results.

Current primary reranking specification:
[`CellPhenoRank_parameter_free_reranking_revision_plan.md`](CellPhenoRank_parameter_free_reranking_revision_plan.md).
Current consolidated results:
[`CellPhenoRank_current_results.md`](CellPhenoRank_current_results.md).
The former weighted CellPath implementation is retained as a legacy comparator in
[`CellPhenoRank_project_description_v2.md`](CellPhenoRank_project_description_v2.md).

CellPhenoRank implements an auditable rare-disease candidate-gene workflow:

```text
disease HPO -> mouse genotype MPO -> base gene ranking
            -> binary HPO-CPO-function-gene enrichment
            -> leave-one-HPO-out constrained reranking
            -> frozen-cohort evaluation and donor-level single-cell validation
```

The current MP-HPO matrix is a replaceable input. When the new cellular-phenotype
MP-HPO matrix is ready, update `inputs.similarity` and its score-column settings;
the baseline, reranker, benchmark, leakage controls, and single-cell code do not
change.

## Evidence boundary

The benchmark cohort must be frozen before cellular-mechanism resources are loaded.
`configs/real_frozen.yaml` points to benchmark v4, whose manifest has
`frozen=true` and `middle_layer_independent=true`. The pipeline verifies the
manifest and the SHA-256 hashes of the disease, model, and model-phenotype inputs.

The current HPO-CPO input is the v5 result and should be treated as the present
middle-layer snapshot, not as the unfinished replacement MP-HPO matrix. GO cell
scope comes from `CPO_generize/GO_link`, GO-function genes from its canonical
`GO_function` companion output, and significant cell markers from
`marker_literuture`.

## Commands

### IMPC DR20.1/2024 reproduction

The official CC BY 4.0 DR20.1 test set is frozen locally with outcome labels
separated from evaluation inputs. The importer exactly reproduces the published
1,311 matches among 2,400 eligible one-to-one ortholog gene knockouts.

See [IMPC_DR20_1_REPRODUCTION.md](IMPC_DR20_1_REPRODUCTION.md) for source
checksums, raw-input coverage and the distinction between output reproduction
and a from-scratch PhenoDigm score replay.

Run the DR20.1 SapBERT agreement test:

    PYTHONPATH=src python3 -B -m cellphenorank run-impc-dr20-embedding \
      --config configs/impc_dr20_1_embedding_test.yaml \
      --output-dir outputs/impc_dr20_1_embedding_test

### Open-candidate SapBERT HPO-MP experiment

The embedding experiment replaces the first-step HPO-MP matrix score with
SapBERT cosine similarity and evaluates against an answer-independent universe of
13,194 phenotype-bearing single-gene MGI candidates. CPO reranking is restricted to
the embedding baseline Top-200; missing gold genes remain failures.

Three prespecified sensitivity variants and their results are documented in
[EMBEDDING_OPEN_BENCHMARK.md](EMBEDDING_OPEN_BENCHMARK.md). Run one variant with:

    PYTHONPATH=src python3 -B -m cellphenorank run-embedding-open \
      --config configs/embedding_open_sapbert_label_k64.yaml \
      --output-dir outputs/embedding_open_sapbert_label_k64

Run the current parameter-free frozen benchmark:

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-parameter-free \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free
```

This is the primary reranking method. It uses no continuous CPO confidence weights,
marker/function mixture weights, CellPath percentile, or lambda fusion. Legacy commands
below remain available for controlled comparison.

### Official IMPC PhenoDigm comparison

Run the live official PhenoDigm scores against the same frozen 50 diseases, 49 candidate
genes, 1,068 phenotype-bearing single-gene models, and gold standard:

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-phenodigm-official \
  --config configs/phenodigm_official.yaml \
  --output-dir outputs/phenodigm_official
```

The first run caches one official IMPC response per disease. Later runs reuse those files;
pass `--refresh` only when intentionally taking a new live snapshot. This is an exact use
of the official precomputed normalized average and maximum scores, followed by the original
PhenoDigm combined-score and best-model-per-gene aggregation. It is not a controlled
same-HPO ablation because the live IMPC score uses its own current disease annotation and
ontology snapshot.

Current result on the frozen candidate universe:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Current A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 |
| Official PhenoDigm | 0.30 | 0.66 | 0.90 | 0.4821 | 0.5756 |

The paired comparison is 20 wins, 9 ties, and 21 losses for official PhenoDigm relative
to current A0 (`p=0.597`); there is no significant overall difference. All 50 disease
gold genes have a positive official score. Among 182 gold genotypes, 179 have an official
record and 177 have a positive model-disease score.

### Archived original PhenoDigm matrix

The historical OWLSim HP-MP matrix from the official PhenoDigm Google Code archive is
stored at `data/external/phenodigm_legacy/hp_mp_mappings.bz2`. It contains 52,022,155
pairs with HP/MP identifiers, simJ and IC. The original artifact, frozen-benchmark subset,
hashes, coverage audit and scoring details are documented in
[`PHENODIGM_LEGACY_MATRIX.md`](PHENODIGM_LEGACY_MATRIX.md).

Reproduce retrieval, subset extraction and the local benchmark:

```bash
python3 -B scripts/fetch_phenodigm_legacy_matrix.py

python3 -B scripts/extract_phenodigm_legacy_subset.py \
  --diseases /path/to/workspace/CPO_generize/mission3/data/benchmark/benchmark_v4/frozen/diseases.tsv \
  --models /path/to/workspace/CPO_generize/mission3/data/benchmark/benchmark_v4/frozen/models.tsv \
  --model-phenotypes /path/to/workspace/CPO_generize/mission3/data/benchmark/benchmark_v4/frozen/model_phenotypes.tsv

PYTHONPATH=src python3 -B -m cellphenorank run-phenodigm-legacy \
  --config configs/phenodigm_legacy.yaml \
  --output-dir outputs/phenodigm_legacy
```

On the same frozen 49-gene universe, the archived-matrix implementation obtains
Recall@1/5/10 of `0.42/0.72/0.82` and MRR `0.5538`. It improves 17 diseases, ties 13,
and worsens 20 relative to current A0 (`p=0.435`), so it is useful as a historical
baseline but does not show significant overall superiority.

### Current Exomiser 2602 matrix

The verified Exomiser 2602 H2 database is retained on the remote worker. Its complete
`HP_MP_MAPPINGS` table contains 95,051,352 rows. The 3,143,719-row frozen-benchmark
subset is stored locally under `data/external/exomiser_2602/current_matrix/` with full
provenance in [`EXOMISER_2602_PHENODIGM_MATRIX.md`](EXOMISER_2602_PHENODIGM_MATRIX.md).

Run the current Exomiser scorer on the frozen local profiles:

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-phenodigm-2602 \
  --config configs/phenodigm_2602.yaml \
  --output-dir outputs/phenodigm_2602
```

The 2602 result obtains Recall@1/5/10 of `0.42/0.78/0.92`, MRR `0.5548`, and
NDCG@10 `0.6382`. Relative to current A0 it has 22 wins, 11 ties, and 17 losses;
the paired difference is not significant (`p=0.872`).

Run parameter-free CPO reranking on the 2602 baseline:

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-parameter-free \
  --config configs/real_frozen_parameter_free_2602.yaml \
  --output-dir outputs/parameter_free_2602
```

Strict A3 changes Recall@1 from `0.42` to `0.48` and MRR from `0.5509` to
`0.5943`, but yields only 6 wins, 37 ties, and 7 losses (`p=0.541`). Bridge-LOO
does not retain a stable overall benefit. See
[`EXOMISER_2602_CPO_RERANKING.md`](EXOMISER_2602_CPO_RERANKING.md).

### Legacy weighted workflow

Run the deterministic synthetic regression example:

```bash
PYTHONPATH=src python3 -B -m cellphenorank run \
  --config configs/example.yaml \
  --output-dir outputs/example
```

Validate and run the frozen 50-disease benchmark:

```bash
PYTHONPATH=src python3 -B -m cellphenorank validate-inputs \
  --config configs/real_frozen.yaml

PYTHONPATH=src python3 -B -m cellphenorank run \
  --config configs/real_frozen.yaml \
  --output-dir outputs/real_frozen
```

`real_frozen_quick.yaml` uses 20 randomizations for engineering verification.
The formal configuration uses 500.

Prepare the independent official ORPHA:399 profile, rank the HD case, and run
donor-level single-cell validation:

```bash
python3 -B scripts/prepare_hd_case.py \
  --hpoa /path/to/workspace/CPO_generize/mission3/data/raw/hpo/phenotype.hpoa \
  --hpo-obo /path/to/workspace/CPO_generize/mission3/data/raw/hpo/hp.obo \
  --output data/cases/hd_disease_hpo.tsv

PYTHONPATH=src python3 -B -m cellphenorank run \
  --config configs/hd_case_rank.yaml \
  --output-dir outputs/hd_case

PYTHONPATH=src python3 -B -m cellphenorank validate-single-cell \
  --config configs/hd_single_cell.yaml \
  --output-dir outputs/hd_single_cell
```

The single-cell analysis uses donor x cell-type pseudobulk, separates composition
from function-state effects, compares the predicted cell against wrong cells and
random gene sets, and never interprets candidate-gene differential expression as
a required success criterion.

The current primary case is IPEX/GSE167976. Freeze predictions before loading
expression, prepare the public matrix, then validate:

```bash
PYTHONPATH=src python3 -B -m cellphenorank freeze-single-cell-case \
  --config configs/ipex_case_freeze.yaml \
  --output-dir data/cases/ipex_case_frozen

python3 -B scripts/prepare_gse167976.py

PYTHONPATH=src python3 -B -m cellphenorank validate-single-cell \
  --config configs/ipex_single_cell.yaml \
  --output-dir outputs/ipex_single_cell
```

The full selection, leakage-control, preflight, null-model and support criteria
are frozen in [SINGLE_CELL_VALIDATION_PROTOCOL.md](SINGLE_CELL_VALIDATION_PROTOCOL.md).

## Main outputs

Current parameter-free outputs:

- `outputs/parameter_free/hpo/`: nonredundant HPO profiles and audit.
- `outputs/parameter_free/cpo_graph/`: Strict and Extended binary graphs.
- `outputs/parameter_free/stability/`: LOO ranks and stable phenotype constraints.
- `outputs/parameter_free/cpo_enrichment/`: hypergeometric enrichment and evidence paths.
- `outputs/parameter_free/reranking/final_constrained_rankings.tsv`: primary A3 ranking.
- `outputs/parameter_free/bridge_loo/`: bridge leakage-control result.
- `outputs/parameter_free/benchmark/`: A0-A5 metrics and paired statistics.

Official PhenoDigm comparison outputs:

- `outputs/phenodigm_official/official/raw/`: frozen live IMPC response per disease.
- `outputs/phenodigm_official/official/phenodigm_model_scores.tsv`: official model scores
  restricted to the frozen model universe.
- `outputs/phenodigm_official/official/phenodigm_gene_rankings.tsv`: best-model gene ranks.
- `outputs/phenodigm_official/qc/coverage_by_disease.tsv`: model, gene and gold coverage.
- `outputs/phenodigm_official/benchmark/`: metrics and paired comparison with current A0.

Archived-matrix PhenoDigm outputs:

- `data/external/phenodigm_legacy/`: original matrix, source manifest and benchmark subset.
- `outputs/phenodigm_legacy/rankings/`: local model scores and gene rankings.
- `outputs/phenodigm_legacy/qc/`: HPO and candidate-gene coverage.
- `outputs/phenodigm_legacy/benchmark/`: comparison with current A0 and live official scores.

Exomiser 2602 outputs:

- `data/external/exomiser_2602/current_matrix/`: current matrix subset, coverage and hashes.
- `outputs/phenodigm_2602/rankings/`: current model scores and gene rankings.
- `outputs/phenodigm_2602/benchmark/`: comparison with current A0 and live official scores.
- `outputs/parameter_free_2602/`: A0-A5 CPO reranking on the 2602 baseline.

Legacy weighted outputs:

- `baseline/disease_model_scores.tsv`: genotype-level BMA and maximum-pair scores.
- `baseline/base_gene_rankings.tsv`: best-model gene aggregation.
- `reranking/final_gene_rankings.tsv`: legacy weighted A3 ranking.
- `reranking/evidence_paths.tsv`: HPO-MPO-CPO-gene score decomposition.
- `benchmark/benchmark_metrics.tsv`: Recall@K, MRR, NDCG@10 and rank summaries.
- `leakage_control/`: strict known-gene masking results and audit rows.
- `null_models/`: size-preserving gene-set permutations.
- `report/report.md` and `report/report.html`: run summary and interpretation limits.

Run tests with the standard library runner:

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```
