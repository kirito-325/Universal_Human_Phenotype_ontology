from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiseaseTerm:
    disease_id: str
    disease_name: str
    hpo_id: str
    hpo_name: str = ""
    weight: float = 1.0
    is_core: bool = False
    gene_symbol: str = ""
    direction: str = "unknown"


@dataclass(frozen=True)
class MouseModel:
    model_id: str
    model_name: str
    mouse_gene_symbol: str
    human_gene_symbol: str
    allele_id: str = ""
    allele_type: str = ""
    zygosity: str = ""
    conditional: bool = False


@dataclass(frozen=True)
class ModelPhenotype:
    model_id: str
    mpo_id: str
    mpo_name: str = ""
    status: str = "abnormal"
    direction: str = "unknown"


@dataclass(frozen=True)
class HpoCpoEdge:
    hpo_id: str
    hpo_name: str
    cpo_id: str
    cell_id: str
    cell_name: str
    function_id: str
    function_name: str
    confidence: float
    direction: str = "unknown"
    relation_type: str = "associated_with"
    scope: str = "unknown"
    source_genes: tuple[str, ...] = ()
    source_gene_contributions: tuple[tuple[str, float], ...] = ()
    source_record_ids: str = ""

    def masked_confidence(self, masked_genes: set[str]) -> float:
        if not masked_genes or not self.source_genes:
            return self.confidence
        masked = {gene.upper() for gene in masked_genes}
        if self.source_gene_contributions:
            total = sum(max(0.0, value) for _, value in self.source_gene_contributions)
            remaining = sum(
                max(0.0, value)
                for gene, value in self.source_gene_contributions
                if gene.upper() not in masked
            )
            return self.confidence * (remaining / total if total else 0.0)
        remaining_count = sum(gene.upper() not in masked for gene in self.source_genes)
        return self.confidence * remaining_count / len(self.source_genes)


@dataclass
class GeneSetResources:
    marker_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    function_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    cell_names: dict[str, str] = field(default_factory=dict)
    function_names: dict[str, str] = field(default_factory=dict)
    marker_set_sizes: dict[str, int] = field(default_factory=dict)
    function_set_sizes: dict[str, int] = field(default_factory=dict)

