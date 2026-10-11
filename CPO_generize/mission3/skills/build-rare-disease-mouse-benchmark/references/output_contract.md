# Output Contract

## Candidate Stage

`candidate_diseases.tsv` contains the fixed independent cohort proposal.
`human_disease_profiles.source.tsv` contains human HPO inputs.
`gold_models.source.tsv` contains source MGI positives pending review.
`candidate_models.tsv` and `evaluation_inputs/model_phenotypes.tsv` support later ranking; the
phenotype table is withheld from Codex research.

`selection_manifest.json` records gates, hashes, diversity strata and selected IDs.
`eligible_candidates.tsv` records the deterministic within-stratum order of every disease that
passed the same pre-research gates. It is the only allowed reserve pool for later replacement.

## Research Stage

`research_packets/` contains immutable disease packets.
`raw_codex_outputs/<rank>_<disease>/` contains the exact prompt, attempts, event streams, errors,
raw `last_message.txt`, locally schema-validated `final.json` and status. The runner must not depend
on provider-side structured-output support; `assets/research_review.schema.json` is the validation
authority.

Use `hemizygous_loss_of_function` for canonical X-linked hemizygous loss-of-function disease,
distinct from biallelic recessive loss and heterozygous haploinsufficiency.

Do not treat a dry run or failed status as a review.

## Validation Stage

`curation/review_status.tsv` lists every disease and validation error.
`curation/validated_reviews.jsonl` contains only fully valid reviews.
`curation/human_signoff.todo.tsv` is the optional human release gate.

An `include` review is machine-valid only when all four readiness flags are true, `blockers` is
empty, and at least two distinct resolvable source URLs support the required evidence classes.

`frozen/` is created only when every selected disease passes the chosen review policy. Its
`freeze_manifest.json` is the authoritative release marker.

`review_policy=codex` produces a Codex-curated research benchmark. Use `review_policy=human` for
a publication-grade benchmark.

A failed disease remains visible in its original benchmark version. A replacement release must
have a new version directory, a parent-manifest hash, and
`replacement_audit/replacements.tsv` recording the same-stratum substitution and exclusion error.

## Evaluation Stage

`evaluation/cpo_ablation/cpo_gain_summary.json` reports paired metrics for all diseases and the
post-freeze evaluable subset. `per_disease_rank_delta.tsv` reports gold-rank wins, ties and losses.

The evaluation stage may read cellular-mechanism inputs. No earlier stage may read them.
