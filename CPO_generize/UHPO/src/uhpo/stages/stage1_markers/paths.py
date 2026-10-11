"""Canonical Stage 1 inputs, intermediate files, and products."""

from uhpo.paths import CONFIG_ROOT, INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT, REPO_ROOT


RAW_DIR = RAW_ROOT / "01_markers"
INTERIM_DIR = INTERIM_ROOT / "01_markers"
PROCESSED_DIR = PROCESSED_ROOT / "01_markers"
DATABASE_DIR = INTERIM_DIR / "literature_database"
NAME_MAPPING_DIR = INTERIM_DIR / "name_mapping"
CATALOG_DIR = PROCESSED_DIR / "catalog"
MARKER_DIR = PROCESSED_DIR / "marker_statistics"
RAW_CL_OBO = RAW_DIR / "cl-basic.obo"
LITERATURE_CONFIG = CONFIG_ROOT / "stages" / "01_markers.json"
ENV_FILE = REPO_ROOT / ".env"
