# Source Policy

## Disease Identity And Mechanism

Use at least two independent sources for an included disease. Prefer:

1. Primary genetic or functional studies establishing the disease-gene mechanism.
2. Expert-curated resources such as MGI, HPO, ClinGen or Orphanet.
3. Authoritative clinical reviews such as GeneReviews.

Do not treat a generic gene-function statement as evidence for the disease-specific allelic
mechanism. Record `unknown` or `review_required` when loss-of-function, gain-of-function,
dominant-negative or altered-function status is not established.

## Core HPO Profile

Select 3-12 HPO terms from the packet's allowed set. Prefer characteristic, penetrant and
diagnostically meaningful findings. Use human disease evidence only. Do not choose a phenotype
because a candidate mouse happens to reproduce it.

## Gold Mouse Models

Start from official MGI disease associations. Review every supplied model ID. Accept the model
for the exact disease definition only when the MGI association and supporting evidence are
consistent. Keep allele, zygosity and background metadata; never collapse labels to a gene-only
positive.

## Citation Requirements

- Store resolvable HTTP(S) URLs.
- State which claim each source supports.
- Keep MGI pages and primary papers distinct when both are available.
- Reject invented IDs, inaccessible placeholders and citations that do not support the claim.
- Preserve search queries and raw Codex events for audit.
