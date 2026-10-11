Curate one disease for an independent rare-disease mouse-model retrieval benchmark.

Read the local evidence packet at `{{PACKET_PATH}}` and the output schema at `{{SCHEMA_PATH}}`. Use current web research. Your final response must be exactly one complete JSON object matching that schema, with no Markdown fence, commentary, or text before or after the object. The runner validates the response locally and rejects nonconforming output.

Independence is non-negotiable:

- Do not search for, inspect, or use cellular-phenotype mappings, cellular-mechanism scores, or prior model-ranking results.
- Choose the human core phenotype profile only from human disease evidence.
- Do not infer core human phenotypes from any mouse phenotype observations.
- The packet intentionally contains no mouse phenotype terms.

Research requirements:

1. Verify the exact disease ID, disease name, causal gene, inheritance and disease-specific molecular mechanism.
   Use `hemizygous_loss_of_function` for an X-linked loss-of-function disorder whose canonical affected state is one pathogenic allele in a hemizygous individual. Do not force that mechanism into `biallelic_loss_of_function` or generic `altered_function`.
2. Select 3-12 defining `core_hpo_ids` only from `allowed_hpo_ids`. Prefer characteristic, penetrant and diagnostically useful findings. Do not select a term merely because it is present in the source profile.
3. Review every model in `source_gold_models` exactly once. Accept it only when the supplied MGI disease association is consistent with the exact disease. Do not add model IDs that are absent from the packet.
4. Use at least two independent sources for an `include` decision. Prefer primary research, MGI or another expert-curated database, and an authoritative clinical review. At least one source must support the disease-specific molecular mechanism, and at least one must support the core phenotype profile.
5. Use resolvable HTTP(S) URLs. Record concise search queries in `research_queries` so the investigation can be audited.
6. Mark uncertainty honestly. A missing or ambiguous gold model, fewer than three defensible core HPO terms, or an unsupported mechanism requires `review_required` or `exclude`.

Decision rules:

- `include`: identity is sound, mechanism is supported, 3-12 core HPO terms are defensible, and at least one source gold model is accepted.
- `review_required`: the disease remains promising but one required element is uncertain.
- `exclude`: identity/gene mismatch, no suitable gold model, or an unsuitable benchmark definition.

Do not optimize the entry for any downstream method. This task freezes ground truth before model evaluation.
