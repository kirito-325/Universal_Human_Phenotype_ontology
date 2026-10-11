from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiseasePhenotype:
    disease_id: str
    disease_name: str
    gene_symbol: str
    hpo_id: str
    hpo_name: str = ""
    weight: float = 1.0
    is_core: bool = False
    frequency: str = ""
    onset: str = ""
    direction: str = "unknown"
    inheritance: str = "unknown"
    molecular_mechanism: str = "unknown"


@dataclass(frozen=True)
class MouseModel:
    model_id: str
    model_name: str
    mouse_gene_symbol: str
    human_gene_symbol: str = ""
    allele_id: str = ""
    allele_type: str = "unknown"
    zygosity: str = "unknown"
    conditional: bool = False
    cre_driver: str = ""
    genetic_background: str = ""
    sex: str = "not_specified"
    life_stage: str = "not_specified"
    availability: str = "unknown"
    known_disease_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelPhenotype:
    model_id: str
    mp_id: str
    mp_name: str = ""
    status: str = "abnormal"
    direction: str = "unknown"
    sex: str = "not_specified"
    life_stage: str = "not_specified"
    procedure_id: str = ""
    parameter_id: str = ""
    source: str = ""


@dataclass(frozen=True)
class MechanismEdge:
    term_id: str
    cl_id: str
    go_id: str
    direction: str = "unknown"
    relation_type: str = "associated_with"
    evidence_scope: str = "unknown"
    confidence: str = "unknown"
    evidence_weight: float = 0.5
    source_id: str = ""
    provisional: bool = False


@dataclass(frozen=True)
class TermMatch:
    hpo_id: str
    mp_id: str
    similarity: float
    relation: str = ""


@dataclass
class PhenotypeEvidence:
    disease_id: str
    model_id: str
    hpo_id: str
    hpo_name: str
    hpo_weight: float
    hpo_onset: str = ""
    best_mp_id: str = ""
    best_mp_name: str = ""
    surface_similarity: float = 0.0
    mapping_relation: str = ""
    assay_state: str = "not_mappable"
    hpo_direction: str = "unknown"
    mp_direction: str = "unknown"
    mp_sex: str = "not_specified"
    mp_life_stage: str = "not_specified"
    cpo_score: float | None = None
    cpo_coverage: float = 0.0
    best_hpo_cl_id: str = ""
    best_hpo_go_id: str = ""
    best_mp_cl_id: str = ""
    best_mp_go_id: str = ""
    direction_compatibility: float | None = None
    suggested_cre_drivers: list[str] = field(default_factory=list)
    evidence_notes: list[str] = field(default_factory=list)


@dataclass
class ModelScore:
    disease_id: str
    disease_name: str
    model_id: str
    model_name: str
    mouse_gene_symbol: str
    human_gene_symbol: str
    allele_id: str
    allele_type: str
    zygosity: str
    conditional: bool
    cre_driver: str
    genetic_background: str
    sex: str
    life_stage: str
    surface_score: float
    surface_coverage: float
    core_reproduced: int
    core_total: int
    cpo_score: float | None
    cpo_coverage: float
    genetic_score: float
    genetic_class: str
    genetic_reason: str
    tested_normal_count: int
    untested_count: int
    opposite_direction_count: int
    total_score: float
    known_model: bool
    availability: str
    rank: int = 0
