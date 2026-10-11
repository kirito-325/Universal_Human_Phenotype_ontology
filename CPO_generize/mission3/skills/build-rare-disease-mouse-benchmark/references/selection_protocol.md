# Selection Protocol

## Objective

Select a fixed 30-50 disease cohort before evaluating a cellular phenotype or mechanism layer.
Make the selection reproducible and independent of downstream ranking changes.

## Allowed Inputs

- Human disease-HPO annotations and inheritance annotations.
- HPO ontology hierarchy for phenotype-system stratification.
- Official MGI genotype-to-disease associations for source gold labels.
- Mouse genotype metadata and the existence of MP observations for runnability.
- HPO-MP surface mapping coverage for baseline runnability.

Do not expose mouse MP terms to the disease research prompt.

## Prohibited Inputs

- Human or mouse cellular-mechanism mappings.
- Cellular-mechanism coverage and similarity scores.
- Results from any surface-plus-mechanism ranking.
- Manual preference for diseases known to improve after adding the tested method.

## Default Gates

- 3 or more source core HPO terms.
- 2 or more same-gene candidate mouse genotypes with MP observations.
- 1 or more official MGI disease-associated gold genotypes.
- At least 70% weighted HPO surface-mapping coverage.
- At most 2 diseases per causal gene.

Balance selected diseases by HPO top-level phenotype system and inheritance class. Rank within
each stratum using only surface runnability, number of source gold models, candidate count and
source core-HPO count.

## Replacement

Do not shrink the cohort after research failures. Record an exclusion reason, select the next
eligible disease from the same phenotype/inheritance stratum, assign a new benchmark version,
and rebuild all affected manifests before ranking.

## Leakage Checks

Require the research packet to contain human HPO annotations and source MGI gold-model metadata
only. Reject packet keys referring to mouse phenotypes, cellular mechanisms, mechanism coverage
or previous ranking outputs.
