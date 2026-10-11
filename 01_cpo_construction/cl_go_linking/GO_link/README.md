# Complete GO Biological Process cell-scope assignment

> Code-only snapshot: counts below describe the original scope release. The referenced `results/` and `work/` artifacts are not included. Rebuilding requires the input evidence tables and local ontology resources described below.

`GO_link` is the only supported location for GO-to-cell scope data. The
published full result is under `GO_link/results/full/`; intermediate and audit
records are under `GO_link/work/`.

## Biological scope definitions

Every non-obsolete GO Biological Process term receives exactly one scope:

| Scope | Meaning | CPO use |
|---|---|---|
| `cell_type_specific` | Curated ontology evidence or reviewed biological evidence supports one or more cell classes as the direct process host | Link only to accepted exact catalog cells |
| `cell_general` | The cellular process spans unrelated cell classes or has no defensible unique exact host | Localize downstream with marker enrichment and anatomy |
| `not_direct_human_cell_process` | The term is organism/tissue-level, non-human, or otherwise not a direct human-cell process | Excluded from CPO linking |

Scope classification and cell mapping are separate. A clearly cell-specific GO
can have `cell_mapping_status=unmapped_catalog_cell` when its correct cell is
absent from the reviewed catalog; the pipeline never substitutes a nearby but
incorrect cell.

Molecule names are not cell evidence. For example, `hepatocyte growth factor`,
`vascular endothelial growth factor`, and `macrophage colony-stimulating
factor` do not alone make a GO term specific to hepatocytes, endothelial cells,
or macrophages. A separately named process such as `endothelial cell
chemotaxis` remains cell-specific.

A cell mentioned in a GO definition is not automatically the process host. The
definition can mention a signal source, responder, interaction partner,
precursor, product, or example. Definition text is supplied to adjudication, but
only GO-name exact matches and curator CL references create deterministic links.
Ambiguous directionality is reviewed explicitly.

## Canonical full result

The current universe contains 24,428 GO-BP terms with no duplicate or unresolved
scope decisions:

| Scope | GO terms |
|---|---:|
| `cell_general` | 12,793 |
| `cell_type_specific` | 4,195 |
| `not_direct_human_cell_process` | 7,440 |

Of the 4,195 cell-specific terms, 3,861 map to at least one exact reviewed cell:
3,851 are fully mapped and 10 are
partially mapped because only part of their intended CL set is in the catalog.
Another 334 remain explicitly cell-specific but unmapped because the current
reviewed catalog lacks the correct exact cell. The result contains 3,942 unique
exact GO-CL links across 390 reviewed cells.

| File | Grain | Purpose |
|---|---|---|
| `results/full/go_process_scope.tsv` | GO | Complete canonical scope table |
| `results/full/cell_specific_go_links.tsv` | exact `(CL, GO)` | Accepted cell-specific links used by HPO/CPO |
| `results/full/cell_general_go_terms.tsv` | GO | General cellular processes used by the independent general channel |
| `results/full/not_direct_go_terms.tsv` | GO | Excluded non-direct processes |
| `results/full/cell_specific_unmapped_go_terms.tsv` | GO | Specific terms whose exact cell is absent from the catalog |
| `results/full/cell_specific_partial_go_terms.tsv` | GO | Specific terms with only part of the intended exact-cell set in the catalog |
| `results/full/unresolved_go_terms.tsv` | GO | Must remain header-only in a complete release |
| `results/full/scope_manifest.json` | run | Inputs, definitions, counts, and digest |
| `results/full/cell_specific_go_dashboard.html` | standalone HTML | Searchable exact cell-GO result with cell marker, GO function-gene, and overlap annotations |

## Decision pipeline

1. Start from the complete semantic GO-BP universe in
   `work/source/semantic/semantic_summary.tsv`.
2. Accept curator and deterministic exact GO-CL evidence from
   `work/source/direct_cell_phenotypes/`.
3. Route only terms under GO `cellular process` (`GO:0009987`) to the default
   general channel; terms outside that branch require positive direct-cell
   evidence or remain non-direct. General means no defensible unique exact host,
   not ubiquitous expression.
4. Send ambiguous and cell-specific-without-link terms to
   `llm_adjudicate_direct_phenotypes.py` with a fixed reviewed-cell candidate
   set.
5. Accept only successful medium/high-confidence decisions and apply
   deterministic guards for self-conflicting rationales, molecule-name false
   friends, and redundant selected descendants.
6. Keep completion and strict revalidation runs in `work/scope_completion/`,
   `work/scope_completion_retry1/`, `work/scope_revalidation_v2/`, and
   `work/scope_revalidation_retry1/` for audit.
7. Validate exact full-CL names and EXACT synonyms against the complete CL OBO.
   Never replace a missing intended cell with a reviewed ancestor or a nearby
   embedding candidate.
8. Apply the small, explicit boundary curation table in
   `curation/scope_overrides.tsv` and require zero unresolved scopes.
9. Fail the build unless the scope IDs exactly equal both the 24,428 active GO-BP
   IDs in the GO OBO and the GO function-summary IDs. The manifest also records
   link uniqueness, catalog membership, intended-CL, and mapping-status checks.

## Rebuild

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m GO_link.build_complete_scope \
  --require-complete

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  web/generate_cell_specific_go_dashboard.py
```

The dashboard generator joins the canonical links with
`cell_marker/results/cell_marker_genes.tsv.gz` and
`GO_function/results/go_function_genes.tsv`. Only cells and GO terms present in
the canonical links are embedded in the standalone HTML. Direct human GOA genes
remain a separate evidence class; additional function genes are propagated from
descendant GO terms only along `is_a` and `part_of` edges. Each inherited gene
retains its source GO IDs, shortest ontology distance, evidence codes, and
experimental-evidence flag.

All four LLM audit runs are included by default, with each retry decision applied
after its corresponding first-pass row.

## Downstream contract

HPO pipelines read only:

- `GO_link/results/full/cell_specific_go_links.tsv`
- `GO_link/results/full/cell_general_go_terms.tsv`

Cell-specific links never inherit through CL parents or descendants. The general
channel does not create an exact GO-CL assertion; cells are ranked downstream
using marker evidence and constrained by anatomy.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  tests.test_go_complete_scope \
  tests.test_direct_phenotype_llm \
  tests.test_cell_specific_go_dashboard

PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tests -p 'test_hpo*.py' -v
```
