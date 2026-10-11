# Verification Record

Date: 2026-10-10.

- 97 tests passed, including six complete compiled Root/eval/task workflows with
  model-scoped hypothesis generation, supporting retrieval, assessment,
  optional targeted scientific completion, normalization, review and MP aggregation.
- Scenarios include an inconclusive model, matching mechanisms across models, and
  opposite process directions that remain separate.
- Source tests verify the union of host-owned MGI annotation PMIDs and searched
  supplementary references, unknown-source rejection, model isolation and MP-term
  isolation when the same genotype appears in multiple term inputs.
- Strict results reject a mechanism assigned to an unrelated model and reject
  declaring a completed term while model coverage gaps remain.
- Unresolved normalization is not converted into a claim that no mechanism exists.
- Paginated model views preserve the full registry and pending count.
- Canonical and alternate IDs cannot start duplicate paid research for the same MP.
- Ruff checks passed.
- Public schemas exported and the MPO-only CLI exercised.
- Support-only research completes without challenge searches. Negative evidence
  labels do not override a defensible scientific review; applicable necessary-premise
  defects and source misinterpretation still block release.
- Recorded negative study findings remain in the empirical evidence block, separately
  from the scientific release decision. Search budgets no longer reserve a challenge quota.

Real data validation:

- MP ontology: 15,357 terms.
- MGI ALL-genotypes annotations: 396,060 rows covering 76,719 genotype IDs.
- MP:0000003: 32 exactly annotated models; first model MGI:2175884 is
  Fgf10<tm1Ska>/Fgf10<tm1Ska>, annotation PMID 16513252.
- Source MP SHA-256:
  93bacc7a9731aec917377cf8d8f95e22b12e61d7665d180ce2d2a17d195728d3
- Source MGI report SHA-256:
  5c7265538a18910bbcde29295bd649626052495379874e5cf8e62892aafec935
- SQLite snapshot SHA-256:
  5de49d339f30e8ef2d8e33514caceca8ad69b4dfb66e249008f7c61b64cb9fd6

No new paid live model call was made for this policy change. Tests exercise actual
application graphs with scripted responses; they establish workflow and contract
behavior, not biological accuracy or live retrieval coverage.

Read-only replay of the prior MP:0020132 result in
runs/single_MP_0020132_20261010 used the stored claims, mappings and scientific reviews:
the Gbvq1 claim changes from rejected to conditional under the revised release gate;
both Cckar claims remain conditional. This is a gate replay, not a new scientific
review or a new research run. Original result files were not modified.
