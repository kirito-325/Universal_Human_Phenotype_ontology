from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTRUCTION = REPO_ROOT / "01_cpo_construction"
CL_SKELETON = CONSTRUCTION / "cl_skeleton"
CL_GO_LINKING = CONSTRUCTION / "cl_go_linking"
MARKER_MINING = CONSTRUCTION / "cell_marker_mining"
PHENOTYPE_MAPPING = REPO_ROOT / "02_phenotype_cpo_mapping"
HPO_CPO = PHENOTYPE_MAPPING / "hpo_cpo"
MPO_CPO = PHENOTYPE_MAPPING / "mpo_cpo"
MPO_HPO_MAPPING = REPO_ROOT / "03_mpo_hpo_mapping"
PROFILE_SIMILARITY = MPO_HPO_MAPPING / "cpo_profile_similarity"
DOWNSTREAM = REPO_ROOT / "04_downstream_evaluation"
GENE_PRIORITIZATION = DOWNSTREAM / "gene_prioritization"
MODEL_PRIORITIZATION = DOWNSTREAM / "mouse_model_prioritization"
SINGLE_CELL_VALIDATION = DOWNSTREAM / "single_cell_validation"
RESOURCES = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"
