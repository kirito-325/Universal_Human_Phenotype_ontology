"""Canonical Stage 4 inputs, intermediate files, and products."""

from uhpo.paths import INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT, REPO_ROOT
from uhpo.stages.stage3_hpo.paths import (
    CELL_GO_LINKS,
    CELL_MARKERS,
    CL_ANATOMY,
    GENERAL_GO_TERMS,
    GO_FUNCTIONS,
    UBERON_ANCESTORS,
)


RAW_DIR = RAW_ROOT / "04_mpo"
INTERIM_DIR = INTERIM_ROOT / "04_mpo"
PROCESSED_DIR = PROCESSED_ROOT / "04_mpo_cpo"
CLASSIFICATION_DIR = INTERIM_DIR / "term_classification"
CLASSIFICATION_BATCH_DIR = CLASSIFICATION_DIR / "batches"
CLASSIFICATION_BATCH_MANIFEST = CLASSIFICATION_DIR / "batch_manifest.json"
ORTHOLOG_DIR = INTERIM_DIR / "human_orthologs"
ANATOMY_DIR = INTERIM_DIR / "anatomy"
LINK_DIR = PROCESSED_DIR / "links"

MPO_OBO = RAW_DIR / "mp.obo"
MGI_GENE_PHENO = RAW_DIR / "MGI_GenePheno.rpt"
MGI_MARKERS = RAW_DIR / "MRK_List2.rpt"
MGI_HOMOLOGY = RAW_DIR / "HOM_MouseHumanSequence.rpt"
VERSION_MANIFEST = RAW_DIR / "version_manifest.json"
TERM_DECISIONS = CLASSIFICATION_DIR / "mpo_term_llm_decisions.tsv"
TERM_DECISION_MANIFEST = CLASSIFICATION_DIR / "mpo_term_llm_decisions_manifest.json"
CLEAN_MPO_OBO = INTERIM_DIR / "mpo_clean.obo"
EXCLUDED_TERMS = INTERIM_DIR / "mpo_excluded_terms.tsv"
CLEAN_MANIFEST = INTERIM_DIR / "mpo_clean_manifest.json"
HUMAN_GENE_EVIDENCE = ORTHOLOG_DIR / "mpo_human_gene_evidence.tsv"
MPO_ANATOMY = ANATOMY_DIR / "mpo_uberon.tsv.gz"
MPO_UBERON_ANCESTORS = ANATOMY_DIR / "uberon_ancestors_with_mpo.tsv.gz"
ENV_FILE = REPO_ROOT / ".env"
