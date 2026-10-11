"""Canonical Stage 3 inputs, intermediate files, and products."""

from uhpo.paths import INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT
from uhpo.stages.stage2_cpo.paths import CELL_MARKER_DIR, GO_FUNCTION_DIR, RELATION_DIR


RAW_DIR = RAW_ROOT / "03_hpo"
INTERIM_DIR = INTERIM_ROOT / "03_hpo"
PROCESSED_DIR = PROCESSED_ROOT / "03_hpo_cpo"
ANATOMY_DIR = INTERIM_DIR / "anatomy"
BATCH_DIR = INTERIM_DIR / "batches"
LINK_DIR = PROCESSED_DIR / "links"

HPO_OBO = RAW_DIR / "hp.obo"
HPOA = RAW_DIR / "phenotype.hpoa"
GENES_TO_PHENOTYPE = RAW_DIR / "genes_to_phenotype.txt"
CELL_MARKERS = CELL_MARKER_DIR / "cell_marker_genes.tsv.gz"
GO_FUNCTIONS = GO_FUNCTION_DIR / "go_function_genes.tsv"
CELL_GO_LINKS = RELATION_DIR / "cell_go_phenotype_links_final.tsv"
GENERAL_GO_TERMS = RELATION_DIR / "llm_cell_general_go_terms.tsv"
HPO_ANATOMY = ANATOMY_DIR / "hpo_uberon.tsv.gz"
CL_ANATOMY = ANATOMY_DIR / "cl_uberon.tsv.gz"
UBERON_ANCESTORS = ANATOMY_DIR / "uberon_ancestors.tsv.gz"
ANATOMY_MANIFEST = ANATOMY_DIR / "anatomy_constraint_manifest.json"
