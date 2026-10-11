# Model-Scoped MP Research

One compiled Deep Agent Root uses QuickJS eval and four existing task specialists:
hypothesis-generator, hypothesis-evaluator, ontology-normalizer, evidence-reviewer.
There is no nested autonomous research agent.

MPContextIndex reads canonical MP definitions and genotype-level MGI annotations from
an immutable SQLite snapshot. Exact annotations are the default; descendants are
opt-in and explicitly marked. All model records are registered before model calls.

ModelCoordinator owns model IDs, annotation references, global budgets, status,
paginated model views, and final aggregation. ModelSession reuses the literature-first
research session, source client, evidence compiler and review gates. Each session
has independent sources, searches, hypotheses, evidence catalogs and audit files.
Only the LLM concurrency limiter and literature retrieval cache are shared.

Per-model ordering:

~~~text
model background + annotation references
    -> discovery plan and supplementary retrieval
    -> actual reading and hypotheses (or an explained empty pool)
    -> supporting retrieval for every hypothesis (challenge is optional)
    -> scientific assessment / one targeted supplement / explicit reassessment
    -> normalization and review of retained claims
    -> source accounting and model completion
~~~

Annotation papers are phenotype-association evidence, not automatic cellular proof.
Hypotheses and final claim identities are scoped to the genotype/model. Same-gene
models remain separate. Model-level experimental scope is judged scientifically,
while reference IDs, source identity and metadata are verified locally.

An empty, justified pool requires actual source reading and permits inconclusive.
Unstudied or inaccessible-source models are deferred/needs_sources, never automatically
inconclusive. An unresolved ontology mapping is not reported as no cellular mechanism.
Every pending model must receive an explicit result or coverage-gap status before Root
completion. Coverage gaps yield a partial term result.

MPResult checks model membership, accepted review scope, mouse target species and exact
observation provenance. The MP-level mechanism output is a union of reviewed
model-scoped observations. CL/GO/direction groups retain every member; they express
ontology grouping, not scientific equivalence, necessity or universality.

Storage schema: 1.0. Scientific interface: mpo_compact_v1.
All scientific roles share the acceptance policy and mouse-model applicability policy.
Research seeks supported, coherent mechanisms with explicit assumptions, without an
obligatory counterevidence pass or exhaustive exclusion of alternatives. Evidence
scope/effect labels do not veto release; review judges whether a read finding actually
invalidates a necessary premise in the target context. Cross-model negative findings
can remain limitations. Experimental evidence status describes recorded findings,
not the truth or demonstrated causality of the complete inferred mechanism.
Host-owned catalog IDs, model IDs, genotype data, source provenance and counts are not
additional model submission fields.
