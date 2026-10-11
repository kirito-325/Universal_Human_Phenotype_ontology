"""Canonical Stage 2 inputs, intermediate files, and products."""

from uhpo.paths import CONFIG_ROOT, INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT, REPO_ROOT
from uhpo.stages.stage1_markers.paths import CATALOG_DIR, MARKER_DIR, RAW_CL_OBO


RAW_DIR = RAW_ROOT / "02_cpo"
INTERIM_DIR = INTERIM_ROOT / "02_cpo"
PROCESSED_DIR = PROCESSED_ROOT / "02_cpo"
SEMANTIC_DIR = INTERIM_DIR / "semantic_candidates"
ASSIGNMENT_DIR = INTERIM_DIR / "direct_assignments"
LLM_BATCH_DIR = INTERIM_DIR / "llm_batches"
CELL_MARKER_DIR = PROCESSED_DIR / "cell_markers"
GO_FUNCTION_DIR = PROCESSED_DIR / "go_functions"
RELATION_DIR = PROCESSED_DIR / "relations"

GO_OBO = RAW_DIR / "go-basic.obo"
GOA_HUMAN = RAW_DIR / "goa_human.gaf"
HGNC = RAW_DIR / "hgnc_complete_set.txt"
MOUSE_HUMAN_ORTHOLOGS = RAW_DIR / "mouse_human_orthologs.tsv"
MGI_MARKERS = RAW_DIR / "MRK_List2.rpt"
CATALOG = CATALOG_DIR / "expanded_cell_catalog.tsv"
MARKER_SCORES = MARKER_DIR / "expanded_marker_scores.tsv"
CATALOG_CONFIG = CONFIG_ROOT / "stages" / "02_cpo_catalog.json"
ENV_FILE = REPO_ROOT / ".env"
