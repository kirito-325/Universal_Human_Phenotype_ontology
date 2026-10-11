# Mission 4

> Code-only snapshot: all counts below describe the original data run. `data/`, matrices, databases and result reports are not included; configure local resources before reproduction. The latest HPO/MPO agents have not yet been integrated into this workflow.

## HPO/MPO CPO browser

The local review interface is under `cpo_browser/`. It keeps the HPO and MPO
ontology trees on separate `/hpo` and `/mpo` pages and separates each term's
self CPO relations from descendant-inherited relations. See
`cpo_browser/README.md` for index-building and startup commands.

## Current v2 workflow

The current implementation is configured by `config/cpo_v2.yaml`; policy and
scoring rules are implemented in `cpo_pipeline/`.
It covers all 18,690 HPO phenotypic-abnormality terms and 14,383 active MPO
terms outside the verified normal branch. Both ontologies use the frozen node
policy and descendant-leaf inheritance rule before one unified CPO-profile
matrix is calculated.

Run or resume the workflow from this directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/mission4_pycache \
MPLCONFIGDIR=/tmp/mission4_matplotlib \
python3 -B -m cpo_pipeline run-all --config config/cpo_v2.yaml --resume
```

The principal outputs are under `results/cpo_mapping_gene_v2/`:

- `matrices/hpo_mpo_cpo.h5`: nine formal HPO x MPO datasets, including the
  final bidirectional BMA matrix, directional diagnostics, self/inherited
  controls, and CL/GO auxiliary matrices.
- `mapping/`: bidirectional term candidates.
- `disease_model/`: all eligible disease-model scores.
- `gene_prediction/`: mouse-gene, strict one-to-one human, and expanded human
  rankings.
- `explanations/`: term and gene provenance chains.
- `reports/final_report.md`: actual results, examples, and declared limits.
- `validation/validation_report.md`: policy and numerical acceptance checks.
- `mhmi_cpo_case_study/index.html`: 50 evenly stratified MHMI reference pairs
  for side-by-side inspection of final CPO profiles and provenance.

Useful queries:

```bash
python3 -B -m cpo_pipeline query-term --config config/cpo_v2.yaml --side HPO --term-id HP:0000002
python3 -B -m cpo_pipeline explain-pair --config config/cpo_v2.yaml --hpo-id HP:0000002 --mpo-id MP:0009900
python3 -B -m cpo_pipeline query-disease --config config/cpo_v2.yaml --disease-id DECIPHER:1
python3 -B -m cpo_pipeline explain-gene --config config/cpo_v2.yaml --disease-id DECIPHER:1 --gene-id HGNC:12796
```

Export the reciprocal `topk_review` Top-5 HPO-MPO pairs with labels, scores,
direction, provenance, MHMI reference status, and review flags:

```bash
python3 -B scripts/export_hpo_mpo_mutual_top5.py --config config/cpo_v2.yaml
```

The resulting TSV and Parquet files are written under
`results/cpo_mapping_gene_v2/mapping/`.

The sections below document source preparation and the previous v1 workflow.
The v1 all-abnormality file is a best-descendant auxiliary result and is not
the v2 unified CPO-profile matrix.

`data/mpo_cpo/` contains the frozen, evidence-backed MPO-to-CPO connection
dataset transferred from the completed MPO literature pipeline.

- `leaf/`: final 6,101-term leaf-node source result and provenance manifests.
- `middle/`: final 4,077-term redundancy-filtered internal-node source result
  and provenance manifests.
- `final/`: unified relation and term-status tables. Every row adds
  `source_batch` and `node_type` so leaf and internal provenance remains
  explicit.

The source directories are immutable copies; regenerate only `final/` with:

```bash
python3 -B mission4/scripts/build_mpo_cpo_dataset.py
```

The builder verifies source error counts, source manifest counts, leaf/internal
term disjointness, relation ID uniqueness, and equality between TSV and JSONL
relation IDs before atomically publishing the final files.

`data/hpo_cpo/` contains the unified HPO-to-CPO leaf relation dataset from the
original literature export and the ECNU recovery run. The published tables do
not expose a batch dimension; input provenance remains in the manifest.

- `hpo_cpo_relations.tsv`: all merged literature relations, preserving both
  direct links and mechanism leads.
- `hpo_cpo_overview.tsv`: the unified per-HPO linkage overview.
- `selection_manifest.json`: the original 12,121-term cohort definition.
- `manifest.json`: source paths and hashes, merge policy, logical record counts,
  relation breakdowns, and invariants.

Rebuild the unified data, inheritance layer, and browser index with:

```bash
python3 -B mission4/scripts/build_hpo_cpo_dataset.py
python3 -B mission4/scripts/build_hpo_cpo_inheritance.py
python3 -B mission4/cpo_browser/build_database.py
```

Recovery `direct` relations remain direct and all non-direct recovery tiers are
published as mechanism leads. Exact full-row duplicates are removed before
publishing.

## MPO CPO inheritance

`data/mpo_cpo/inherited/` materializes leaf-to-ancestor CPO inheritance across
the complete active MPO `is_a` DAG. Every accepted leaf relation is propagated
to every middle-layer, top-level, and root ancestor without a distance cutoff.
Direct and mechanism relations remain separate, and middle-layer self-mined
relations are counted for profile status but are not used as propagation
sources.

Build the frozen inheritance export with:

```bash
python3 -B mission4/scripts/build_mpo_cpo_inheritance.py
```

The output separates compact evidence provenance from deduplicated assertions:

- `mpo_cpo_inheritance_provenance.tsv`: one row per source leaf relation and
  target ancestor, including shortest distance, number of shortest paths, and a
  deterministic canonical path.
- `mpo_cpo_inherited_assertions.tsv`: one row per deduplicated ancestor CPO
  assertion with descendant-leaf support and coverage statistics.
- `mpo_cpo_inheritance_term_status.tsv`: every active MPO term, including
  structural class, middle-layer query decision, self evidence, inherited
  evidence, and unresolved status.
- `manifest.json`: rules, counts, invariants, input/output paths, and SHA-256
  hashes.

## HPO-MPO CPO similarity

`cpo_mapping/` implements the earlier deterministic workflow configured by
`config/cpo_mapping_v1.yaml`. The original formal run uses 12,121 HPO leaf
terms and all 14,385 active MPO terms. Four self-evidence channels (`DD`,
`DM`, `MD`, and `MM`) each contain independent shape, direction-support, and
direction-conflict matrices. Inherited MPO evidence remains separate and is
used only for best-descendant branch retrieval.

Run the complete workflow from this directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/mission4_pycache \
python3 -B -m cpo_mapping run-all \
  --config config/cpo_mapping_v1.yaml \
  --resume
```

Individual stages use the same interface:

```bash
python3 -B -m cpo_mapping audit --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping prepare --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping smoke-test --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping compute --config config/cpo_mapping_v1.yaml --resume
python3 -B -m cpo_mapping candidates --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping branch --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping build-explanations --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping evaluate --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping validate --config config/cpo_mapping_v1.yaml
python3 -B -m cpo_mapping report --config config/cpo_mapping_v1.yaml
```

Query a real indexed HPO term:

```bash
python3 -B -m cpo_mapping query \
  --config config/cpo_mapping_v1.yaml \
  --side HPO --term-id HP:0000003 --channel MD --view shape_topk
```

The full HDF5 matrix and downstream artifacts are under
`results/cpo_mapping_v1/`. See its `README.md` and
`reports/final_report.md` for entry points and interpretation constraints.

## All-abnormality branch comparison

HPO leaf relations are also propagated upward with the same root / self-query
middle / inheritance-only middle / leaf classification used on the MPO side:

```bash
python3 -B scripts/build_hpo_cpo_inheritance.py
python3 -B -m cpo_mapping run-all-abnormal \
  --config config/cpo_mapping_v1.yaml --resume
```

This additional view indexes all 18,690 HPO phenotypic-abnormality terms and
14,383 MPO terms outside the normal-phenotype branch. Its scores are best
verified descendant-leaf evidence and remain separate from parent self scores.
