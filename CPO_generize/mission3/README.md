# Mission 3: mechanism-aware mouse-model prioritization

> Code-only snapshot: real data, frozen cohorts and outputs referenced below are not included. Synthetic `examples/` are included. The newer candidate-gene reranker is in the repository-root [mission3](../../mission3/README.md); these are distinct tasks.

`mission3` ranks concrete mouse genotypes for a rare disease and produces an
auditable suitability report. The unit of analysis is an MGI genotype, not a
gene-to-MP association.

The current official-data run, its limitations and the next review queues are
summarized in [`REAL_DATA_STATUS.md`](REAL_DATA_STATUS.md).

The implementation supports two tasks:

1. `within_gene` (default): use the known human disease gene only to select the
   candidate set, then rank alleles/zygosities with phenotype, CPO and genetic
   mechanism evidence.
2. `all_models`: search all supplied models without using the disease gene.
   Genetic compatibility is automatically removed from the score in this mode.

## What is implemented

- weighted disease-HPO to model-MP phenotype coverage;
- HPO `(CL, GO, direction)` to MP `(CL, GO, direction)` comparison;
- explicit `tested_normal`, `untested`, `opposite_direction` and
  `not_mappable` states;
- LoF, haploinsufficiency, GoF, dominant-negative and conditional-model rules;
- model-level rankings, per-HPO evidence, gap analysis and cell suggestions;
- known-model Recall@1/5/10, MRR and NDCG@10;
- surface/CPO/genetic/full ablations;
- input hashes, counts, limitations and run warnings;
- static HTML and JSONL suitability reports.

Missing CPO evidence is always `unknown`: it never reduces a score. Model
availability is reported but is not part of biological suitability scoring.

## Important evidence boundary

The default HPO mechanism input is the current literature-derived snapshot:

```text
UHPO/data/hpo_cpo_links/hpo_cpo_candidate_links.partial.tsv
```

Only rows satisfying all of the following can enter the main CPO score:

- `bridge_scope=direct_hpo`;
- both CL and GO IDs are present;
- relation is causal, equivalent, or constituent;
- CL is in the reviewed catalog;
- GO is not classified as `not_direct_human_cell_process`.

The current snapshot is provisional, so the default configuration excludes its
`unreviewed_extracted_claim` rows. Set
`include_provisional_mechanisms=true` only for an explicitly labelled pilot;
those edges are then down-weighted by 50%.

For the present partial snapshot, use
`mission3/config/provisional_pilot.json`. The resulting run must be labelled
provisional; the strict publication configuration remains `config/default.json`.

The surface score must come from a semantic-only MP-HPO channel. Do not use a
CPO-fused score as `S_surface` and then add `S_CPO`, because that double-counts
the mechanism evidence. The loader prefers the configured `semantic_score`
column, followed by `bi_score`, `final_score`, and `score`.

## Quick start

Run the synthetic, non-biological example:

```bash
python3 -B mission3/scripts/run_pipeline.py run \
  --diseases mission3/examples/diseases.tsv \
  --models mission3/examples/models.tsv \
  --model-phenotypes mission3/examples/model_phenotypes.tsv \
  --mp-hpo-mapping mission3/examples/mp_hpo_mapping.tsv \
  --hpo-mechanisms mission3/examples/hpo_mechanisms.tsv \
  --mp-mechanisms mission3/examples/mp_mechanisms.tsv \
  --cell-catalog mission3/examples/cell_catalog.tsv \
  --go-classification mission3/examples/go_classification.tsv \
  --cl-obo mission3/examples/cl.obo \
  --go-obo mission3/examples/go.obo \
  --cre-drivers mission3/examples/cre_drivers.tsv \
  --config mission3/examples/config.json \
  --output-dir mission3/outputs/example
```

Open `mission3/outputs/example/index.html` or inspect
`model_rankings.tsv`. The example is designed so that a phenotype-similar
homozygous null is ranked below the mechanism-concordant heterozygous null.

## Standard input contracts

### `diseases.tsv`

One row per disease-HPO assertion:

```text
disease_id, disease_name, gene_symbol, hpo_id, hpo_name,
weight, is_core, frequency, onset, direction,
inheritance, molecular_mechanism
```

Supported mechanism values include `loss_of_function`, `biallelic_lof`,
`hemizygous_loss_of_function`, `haploinsufficiency`, `gain_of_function`, `dominant_negative`,
`tissue_specific`, and `unknown`.

### `models.tsv`

One row per concrete genotype:

```text
model_id, model_name, mouse_gene_symbol, human_gene_symbol,
allele_id, allele_type, zygosity, conditional, cre_driver,
genetic_background, sex, life_stage, availability, known_disease_ids
```

`known_disease_ids` is used only for evaluation. Multiple IDs or genes are
semicolon-delimited.

### `model_phenotypes.tsv`

One row per genotype-MP observation or completed normal assay:

```text
model_id, mp_id, mp_name, status, direction, sex, life_stage,
procedure_id, parameter_id, source
```

`status` must be one of:

- `abnormal`: the MP abnormality was observed;
- `tested_normal`: the relevant procedure/parameter was completed without the
  abnormality;
- `untested`: explicit lack of a comparable assay.

If a disease HPO has known MP mappings but a model has no comparable MP assay
row, the pipeline reports `untested`. If the HPO has no MP mapping at all, it
reports `not_mappable`.

### Mechanism tables

The HPO loader accepts the current literature-extraction columns
`selected_cl_id`, `selected_go_id`, `relation_type`, `bridge_scope`, `direction`,
`candidate_status`, and `supporting_record_ids`.

The MP loader accepts the existing `03_MPO_link` columns `cell_type_id`,
`cell_phenotype_id`, `phenotype_scope`, `confidence_grade`, and `rerank_score`.

An optional `cre_drivers.tsv` accepts `cre_driver`, `cl_id`, `detected`,
`availability`, and `allele_id`. Only explicit detected driver-CL relationships
are used. MGI's anatomical `Detected in` text must be mapped and reviewed before
it is represented as CL; the ranking code does not guess CL terms from names.

## Preparing official MGI reports

Download the current official MGI and HPO release assets and write a hash-pinned
manifest:

```bash
python3 -B mission3/scripts/run_pipeline.py download-data \
  --source all \
  --output-dir mission3/data/raw
```

Existing nonempty files are reused and re-hashed. Use `--force` only when
intentionally advancing the data snapshot. The HPO manifest records the final
concrete release URL reached through GitHub's `latest` redirect.

MGI reports are headerless. Normalize them with:

```bash
python3 -B mission3/scripts/run_pipeline.py prepare-mgi \
  --pheno-geno mission3/data/raw/mgi/MGI_PhenoGenoMP.rpt \
  --orthologs mission3/data/raw/mgi/HOM_ProteinCoding.rpt \
  --phenotypic-alleles mission3/data/raw/mgi/MGI_PhenotypicAllele.rpt \
  --genotype-disease mission3/data/raw/mgi/MGI_Geno_DiseaseDO.rpt \
  --disease-mouse-models mission3/data/raw/mgi/MGI_DiseaseMouseModel.rpt \
  --phenotype-sex mission3/data/raw/mgi/MGI_Pheno_Sex.rpt \
  --output-dir mission3/data/processed/mgi
```

This writes `models.tsv`, `model_phenotypes.tsv`, and an import manifest. The
adapter follows the official field order documented by the
[MGI reports index](https://www.informatics.jax.org/downloads/reports/index.html).

Use `MGI_PhenoGenoMP.rpt`, not `MGI_GenePheno.rpt`: the latter explicitly
excludes conditional mutations. Supplying `MGI_PhenotypicAllele.rpt` is required
for useful allele-type scoring. Conditional status inferred from allele
type/attribute remains an auditable heuristic.

MGI reports do not contain complete IMPC assay coverage. Add IMPC statistical
results to `model_phenotypes.tsv` after mapping each procedure/parameter to its
possible MP term:

- significant same-direction result -> `abnormal`;
- completed nonsignificant result -> `tested_normal`;
- absent procedure/parameter -> no row, interpreted as `untested` when a mapping
  exists.

Do not convert all missing MP annotations into `tested_normal`.

## Building and auditing the pilot cohort

First copy `templates/disease_curation.tsv` to a project-specific reviewed
table. This table supplies the disease gene, inheritance, molecular mechanism,
three or more core HPO terms and optional HPO directions. Then prepare all
unambiguous HPO disease profiles:

```bash
python3 -B mission3/scripts/run_pipeline.py prepare-diseases \
  --hpoa mission3/data/raw/hpo/phenotype.hpoa \
  --gene-disease mission3/data/raw/hpo/genes_to_disease.txt \
  --hpo-obo mission3/data/raw/hpo/hp.obo \
  --curation mission3/data/curation/disease_curation.tsv \
  --output-dir mission3/data/processed/diseases
```

Negated HPO annotations are excluded. HPO aspect-I annotations initialize the
inheritance field, while reviewed curation can override it. Multi-gene diseases
without an explicit curated gene are retained in the manifest as skipped
instead of being guessed.

Next, audit all disease/model pairs and select at most 50 eligible diseases:

```bash
python3 -B mission3/scripts/run_pipeline.py audit-cohort \
  --diseases mission3/data/processed/diseases/diseases.all.tsv \
  --models mission3/data/processed/mgi/models.tsv \
  --model-phenotypes mission3/data/processed/mgi/model_phenotypes.tsv \
  --disease-xrefs mission3/data/curation/disease_xrefs.tsv \
  --config mission3/config/default.json \
  --output-dir mission3/data/processed/cohort
```

The fixed default gates are:

- at least 3 core HPO terms, 2 concrete genotypes and 1 known positive model;
- at least 70% weighted HPO surface-mapping coverage;
- at least 2 loaded HPO-CPO terms and 2 HPOs comparable to candidate MP-CPO;
- at least 30% candidate-MP CPO coverage;
- reviewed inheritance and molecular mechanism.

The command never relaxes these gates to reach 30 diseases. It writes every
failure reason to `disease_coverage.tsv` and marks `pilot_ready=false` when the
minimum cohort size is not met. Selection among eligible diseases is
deterministic and round-robin across molecular-mechanism groups.

Even when the pilot is not ready, the command writes three actionable review
artifacts:

- `curation_queue.tsv`: up to 50 diseases that already pass the surface/model
  gate, prioritized by current CPO evidence;
- `disease_curation.todo.tsv`: prefilled inheritance and suggested core HPOs;
  reviewers fill mechanism/direction, check the core profile and set
  `include=true`;
- `hpo_cpo_review_queue.tsv`: disease-specific HPO terms lacking a strict
  mechanism edge, with existing provisional candidates placed first.

The same preselection is exported as `surface_baseline_*.tsv`. It can be run
immediately with `config/surface_baseline.json`, where both CPO and genetic
weights are fixed at zero. This produces a real-data baseline without claiming
that the cohort has passed the mechanism-readiness gate.

`config/provisional_cpo_diagnostic.json` enables only the down-weighted,
unreviewed CPO channel on that exact cohort. It is a coverage diagnostic, not a
benchmark result. Genetic weight remains zero so unknown disease mechanism and
conditional-allele heuristics cannot leak into the comparison.

During exploration only, `config/provisional_pilot.json` may be supplied to
measure the effect of the current unreviewed HPO-CPO snapshot. Such a cohort is
provisional and cannot be reported as the strict benchmark.

When `pilot_ready=true`, use the exact `next_command` recorded in
`cohort_manifest.json` to run the existing ranking and ablation pipeline.

## Real run with repository defaults

After preparing disease and model tables, the current repository CPO/MP inputs
are already the command defaults:

```bash
python3 -B mission3/scripts/run_pipeline.py run \
  --diseases mission3/data/processed/diseases.tsv \
  --models mission3/data/processed/mgi/models.tsv \
  --model-phenotypes mission3/data/processed/mgi/model_phenotypes.tsv \
  --output-dir mission3/outputs/pilot
```

The defaults point to:

- `UHPO/data/mpo_hpo_mapping/mpo_hpo_mapping.tsv`;
- `UHPO/data/hpo_cpo_links/hpo_cpo_candidate_links.partial.tsv`;
- `03_MPO_link/.../mpo_cell_mechanism_triples.tsv`;
- the reviewed CPO cell catalog and GO classification;
- the local CL and GO OBO snapshots.

## Scores

For each disease HPO, `S_surface` uses the best comparable abnormal MP term and
is normalized by the total disease-HPO weight. Tested-normal phenotypes are
comparable evidence but contribute no positive surface score.

For a pair of mechanism edges:

```text
S_pair = evidence_weight * (
    0.4 * similarity(CL_human, CL_mouse)
  + 0.4 * similarity(GO_human, GO_mouse)
  + 0.2 * direction_compatibility
)
```

CL/GO similarities are exact matches or normalized ontology information-content
similarities. Unknown direction contributes zero rather than a penalty.

The benchmark rank is:

```text
S_total = S_surface + lambda_cpo * C_cpo * S_cpo
                        + lambda_genetic * S_genetic
```

`C_cpo` is the weighted proportion of disease HPO terms for which both human and
mouse mechanism evidence can be compared. Weights are configuration values, not
probabilities, and must be selected using validation diseases only before a
publication-scale evaluation.

## Outputs

| File | Purpose |
|---|---|
| `model_rankings.tsv` | One row per disease-model candidate with all score dimensions. |
| `phenotype_evidence.tsv` | One row per disease-HPO/model comparison. |
| `gap_analysis.tsv` | Unreproduced HPO terms plus suggested CL/GO targets. |
| `model_reports.jsonl` | Complete machine-readable suitability reports. |
| `index.html` | Static evidence browser. |
| `evaluation_summary.json` | Known-model metrics and ablations. |
| `run_manifest.json` | Input hashes, counts, warnings and interpretation boundaries. |

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s mission3/tests -v
```

The included example and tests are software validation only. They are not a
biological benchmark or evidence that CPO improves real model retrieval.
