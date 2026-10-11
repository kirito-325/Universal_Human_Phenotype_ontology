# Mission 3 real-data status

Snapshot inspected and generated on 2026-08-15 (Asia/Shanghai).

## What is now real

- HPO release: `2026-06-23` (`phenotype.hpoa`, `genes_to_disease.txt`,
  `hp.obo`).
- MGI reports: official files last modified 2026-08-10.
- Download provenance: `data/raw/download_manifest.json` records source URLs,
  byte sizes, timestamps and SHA-256 hashes.
- Normalized MGI data: 76,611 concrete genotypes and 467,378 genotype-MP
  observations.
- Model metadata coverage: 61,510 genotypes have a human ortholog, 72,353 have
  an allele type/functional attribute, and 4,474 have a known disease label.
- HPO disease profiles: 8,208 unambiguous single-gene profiles with 181,293
  positive disease-HPO rows. Another 4,727 HPOA diseases are explicitly skipped
  because a gene is absent or ambiguous.

The MGI allele adapter combines mutation class and functional attribute. For
example, a targeted allele annotated as `Null/knockout` is visible to the
genetic-mechanism scorer instead of being represented only as `Targeted`.

## Readiness audit

The strict and provisional audits are in:

- `data/processed/cohort_strict/coverage_report.html`
- `data/processed/cohort_provisional/coverage_report.html`

Both audits use fixed gates and currently report `pilot_ready=false`.

| Measure | Current result |
|---|---:|
| Diseases audited | 8,208 |
| HPO terms with MP-HPO surface mappings | 5,225 |
| MP terms with loaded CPO mechanisms | 2,173 |
| Strict reviewed HPO-CPO terms | 0 |
| Provisional structurally eligible HPO-CPO terms | 114 |
| Diseases passing the surface/model/gold gate | 485 |
| Diseases passing the provisional CPO gate | 0 |
| Diseases passing all strict pilot gates | 0 |

The main failure is not missing MGI data anymore. The provisional audit finds
only four surface-ready diseases with at least two current HPO-CPO terms, and
none has two actually comparable disease-HPO/model-MP mechanism channels.
Molecular mechanism is also intentionally unknown until reviewed disease
curation is supplied.

## Real surface baseline

The audit exported 50 surface-ready diseases for an immediate within-gene
baseline. This cohort is deliberately enriched for diseases with known MGI
models and adequate surface mappings; it is not an unbiased estimate over all
rare diseases.

| Measure | Result |
|---|---:|
| Diseases | 50 |
| Unique concrete MGI genotypes | 997 |
| Disease-model ranking rows | 1,052 |
| Model-MP rows | 6,610 |
| Recall@1 | 0.70 |
| Recall@5 | 0.94 |
| Recall@10 | 0.98 |
| MRR | 0.8144 |
| NDCG@10 | 0.7879 |

Results and the evidence browser are in `outputs/surface_baseline/`.

## Provisional CPO diagnostic

The exact same cohort was rerun with down-weighted provisional CPO enabled and
genetic weight fixed at zero:

- 12 of 50 diseases received any comparable CPO evidence;
- 44 of 1,052 disease-model rows had nonzero CPO coverage;
- zero ranking rows changed position;
- all retrieval metrics were identical to the surface baseline.

This is a coverage diagnostic, not evidence that CPO has no value. The current
114-term provisional snapshot is too sparse to test that hypothesis.

## Action queues

The provisional audit generated:

- `curation_queue.tsv`: the 50 disease candidates to review first;
- `disease_curation.todo.tsv`: inherited HPO mode, up to 15 suggested core HPOs,
  and explicit placeholders for molecular mechanism and direction;
- `hpo_cpo_review_queue.tsv`: 1,694 disease-HPO review tasks, ordered with
  existing provisional claims first;
- `disease_coverage.tsv`: all 8,208 diseases and every failed gate.

Reviewers must verify each proposed core profile, fill molecular mechanism and
directions, and set `include=true`. Existing provisional HPO-CPO claims must be
accepted or rejected with source evidence; missing high-priority terms then
receive targeted literature curation.

The full 12,121-term HPO literature batch is still active. At the latest check,
6,459 status files existed, so the 114-term snapshot must not be treated as the
final HPO-CPO release.

After reviewed disease and HPO-CPO tables are available, rerun
`prepare-diseases`, then the strict `audit-cohort`. Only a
`cohort_manifest.json` with `pilot_ready=true` authorizes the full mechanism and
genetic ablation benchmark. IMPC procedure/parameter coverage remains a later
extension for distinguishing `tested_normal` from `untested` systematically.
