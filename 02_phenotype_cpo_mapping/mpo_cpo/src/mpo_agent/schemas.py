"""Shared contracts for agents, validators, persisted records, and JSON Schema export."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .contracts import (
    DIRECTION_DESCRIPTION,
    QUOTE_DESCRIPTION,
    SOURCE_DOCUMENT_DESCRIPTION,
    contract_error,
)
from .normalization import normalize_assessment

Text = Annotated[str, Field(min_length=1)]
PhenotypeID = Annotated[str, Field(pattern=r"^MP:[0-9]{7}$")]
Direction = Literal["increased", "decreased", "abnormal", "absent", "present", "bidirectional", "unknown"]
Relation = Literal[
    "equivalent_cellular_phenotype", "has_constituent_cellular_phenotype",
    "causally_upstream_of", "associated_with", "mechanistically_compatible_with",
]
Bridge = Literal[
    "direct_phenotype", "same_mgi_model", "phenotype_mechanism_bridge",
    "supplied_disease", "gene_model", "mechanistically_compatible",
]
EvidenceMode = Literal["direct", "inferred"]
ReasoningBasis = Literal["phenotype_definition", "literature", "inference", "model_knowledge"]
KnowledgeSource = Literal["literature", "model_knowledge", "mixed"]
SCHEMA_VERSION = "1.0"
MODEL_CONTRACT = "mpo_compact_v1"


def canonical_reasoning_start(source: str, phenotype_id: str, label: str) -> str:
    def normalize(value):
        return " ".join(value.split()).casefold()
    accepted = {normalize(value) for value in (
        label, phenotype_id, "INPUT_PHENOTYPE", f"{label} ({phenotype_id})",
        f"{phenotype_id}: {label}", f"{phenotype_id} ({label})",
    )}
    if normalize(source) not in accepted:
        raise contract_error("claim.start", field=("reasoning_steps", 0, "source"), expected=label, actual=source)
    return label


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False, strict=True)


class MGIContext(Record):
    context_id: Text
    pmid: Annotated[str, Field(pattern=r"^[0-9]+$")]
    genotype: Text
    markers: list[str] = Field(default_factory=list)
    source_ref: Text


class MouseModel(Record):
    model_id: Annotated[str, Field(pattern=r"^MGI:[0-9]+$")]
    allelic_composition: Text
    allele_symbols: list[str]
    genetic_background: str
    marker_ids: list[str]
    annotation_pmids: list[Annotated[str, Field(pattern=r"^[0-9]+$")]]
    annotations: list[dict]


class MPInput(Record):
    phenotype_id: PhenotypeID
    include_descendants: bool = False


class Phenotype(Record):
    phenotype_id: PhenotypeID
    label: Text
    definition: str = ""
    synonyms: list[str] = Field(default_factory=list)
    disease_contexts: list[str] = Field(default_factory=list)
    seed_pmids: list[Annotated[str, Field(pattern=r"^[0-9]+$")]] = Field(default_factory=list)
    mgi_contexts: list[MGIContext] = Field(default_factory=list)
    model_context: MouseModel | None = None

    @model_validator(mode="after")
    def contexts(self):
        ids = [c.context_id for c in self.mgi_contexts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate MGI context IDs")
        return self

    @property
    def target_species(self) -> str:
        return "mouse"


class Section(Record):
    name: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Document(Record):
    document_id: Text
    title: str
    text: Text
    source_url: Text
    source_kind: Literal["abstract", "full_text", "local_text"]
    pmid: str = Field(default="", pattern=r"^[0-9]*$")
    pmcid: str = Field(default="", pattern=r"^(PMC[0-9]+)?$")
    doi: str = ""
    phenotype_ids: list[PhenotypeID] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    retracted: bool = False
    retrieval_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def offsets(self):
        if any(s.end <= s.start or s.end > len(self.text) for s in self.sections):
            raise ValueError("section offsets outside document")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class QuoteReference(Record):
    document_id: Text = Field(description=SOURCE_DOCUMENT_DESCRIPTION)
    quote: Text = Field(description=QUOTE_DESCRIPTION)
    span_start: int | None = Field(default=None, ge=0, description="Optional exact host-provided absolute offset; omit rather than estimate.")


class HypothesisSeedReference(Record):
    """The host locates seed quotations; legacy model offsets are discarded."""

    document_id: Text = Field(description=SOURCE_DOCUMENT_DESCRIPTION)
    quote: Text = Field(description=QUOTE_DESCRIPTION)

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_offset(cls, value):
        if isinstance(value, QuoteReference):
            value = value.model_dump()
        if isinstance(value, dict) and "span_start" in value:
            return {key: item for key, item in value.items() if key != "span_start"}
        return value


class ReasoningStep(Record):
    scope: Literal["phenotype_link", "cell_identity", "mechanism"] = Field(
        default="phenotype_link", description="Use cell_identity for the step justifying which cell population is implicated; inferred identity needs inference/model_knowledge.")
    source: Text
    target: Text
    relation: Text
    basis: ReasoningBasis = Field(description="literature: exact cited facts, no assumptions; inference: cited premises plus assumptions; model_knowledge: learned premise, no citations; phenotype_definition: first input-only step.")
    explanation: Text
    citations: list[QuoteReference] = Field(default_factory=list, description="Exact read-source quotations for this step, not source IDs alone. Empty for model_knowledge/phenotype_definition.")
    assumptions: list[Text] = Field(default_factory=list, description="Applicability assumptions, required for inference; empty for literature/phenotype_definition. Do not add assumptions_note/basis_notes.")

    @model_validator(mode="after")
    def support(self):
        if self.basis == "phenotype_definition":
            if self.citations or self.assumptions:
                raise contract_error("step.definition", field=("basis",))
        elif self.basis == "model_knowledge":
            if self.citations:
                raise contract_error("step.model_knowledge", field=("citations",))
        elif not self.citations:
            raise contract_error("step.citations", field=("citations",))
        if self.basis == "inference" and not self.assumptions:
            raise contract_error("step.inference_assumptions", field=("assumptions",))
        if self.basis not in {"inference", "model_knowledge"} and self.assumptions:
            raise contract_error("step.assumptions", field=("basis",))
        return self


class ExperimentalEvidenceDraft(Record):
    document_id: Text = Field(description=SOURCE_DOCUMENT_DESCRIPTION)
    quote: Text = Field(description=QUOTE_DESCRIPTION)
    scope: Literal["cell_identity", "cell_abnormality", "mechanism", "phenotype_link", "whole_claim"] = Field(
        description="What this actual study supports/tests. Whole_claim requires the complete claim and applicable species, model, anatomy and experimental conditions. Other-model findings usually have component scope. An unreported phenotype is not a demonstrated negative result.")
    study_type: Literal["observation", "pathology", "intervention"]
    species: Literal["human", "mouse", "rat", "other_model", "unknown"]
    experimental_context: Text
    finding: Text = Field(description="The actual study finding, separately from your inferred mechanism or extrapolation. Review speculation is not an experimental finding.")
    effect: Literal["supports", "contradicts"] = "supports"


class ClaimDraft(Record):
    hypothesis_id: str = Field(default="", description="Optional current hypothesis ID; omit for the host to assign. Do not invent another hypothesis ID.")
    evidence_mode: EvidenceMode
    cell_evidence_mode: EvidenceMode = "direct"
    link_evidence_mode: EvidenceMode | None = None
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    experimental_evidence: list[ExperimentalEvidenceDraft] = Field(default_factory=list,
        description="Actual read primary study findings used for support/contradiction, scoped to a component or whole claim; [] allowed. Put interpretation in reasoning_steps, not this evidence block.")
    supporting_quote: str = Field(default="", description=QUOTE_DESCRIPTION + " Optional main quotation; empty requires supporting_document_id=''.")
    supporting_document_id: str = Field(default="", description=SOURCE_DOCUMENT_DESCRIPTION + " Paired only with supporting_quote; empty when quote is empty.")
    cell_text: Text = Field(description="Actual implicated cell population; anatomy, matrix and released vesicles are context/products, not themselves a cell type.")
    process_text: str = Field(description="Named cellular process whose direction is assessed; empty for a quality-only abnormality. Do not substitute the macro phenotype's polarity.")
    quality_text: Text = Field(description="Cellular abnormality/quality consistent with process_text and direction. The reasoning endpoint is exactly cell_text + ': ' + quality_text.")
    direction: Direction = Field(description=DIRECTION_DESCRIPTION)
    relation_type: Relation
    bridge_scope: Bridge
    species: Literal["human", "mouse", "rat", "other_model", "unknown"] = Field(
        description="For inferred claims, intended conclusion species; for direct observations, actual study species. Experiment entries retain their own actual species.")
    taxon_id: Annotated[str, Field(pattern=r"^NCBITaxon:[0-9]+$",
        description="Taxon matching claim.species: human=NCBITaxon:9606, mouse=NCBITaxon:10090, rat=NCBITaxon:10116, unknown=NCBITaxon:0; other_model uses its actual taxon.")]
    disease_context: str = ""
    anatomy_context: str = ""
    experimental_context: str = ""
    assertion: Literal["own_result", "review_statement", "background_statement", "model_inference"]
    evidence_type: Literal["pathology", "observation", "intervention", "causal_statement", "inference"]
    gene_mentions_ignored: list[str]
    mgi_context_id: str = Field(default="", description="Supplied MGI context ID only for bridge_scope=same_mgi_model; otherwise empty. General animal-study evidence belongs in experimental_evidence.")
    model_evidence_quote: str = Field(default="", description="Exact model-identifying quote only for same_mgi_model, from a read window of the main source; otherwise empty.")
    rationale: Text

    @property
    def cellular_statement(self) -> str:
        return f"{self.cell_text}: {self.quality_text}"

    def bind_phenotype(self, phenotype_id: str, label: str) -> ClaimDraft:
        if not self.reasoning_steps:
            return self
        first = self.reasoning_steps[0]
        canonical = canonical_reasoning_start(first.source, phenotype_id, label)
        if first.source == canonical:
            return self
        return self.model_copy(update={"reasoning_steps": [first.model_copy(update={"source": canonical}),
                                                           *self.reasoning_steps[1:]]})

    @property
    def knowledge_source(self) -> str:
        model = any(s.basis == "model_knowledge" for s in self.reasoning_steps)
        literature = bool(self.supporting_quote or self.experimental_evidence) or any(s.citations for s in self.reasoning_steps)
        return "mixed" if model and literature else "model_knowledge" if model else "literature"

    @model_validator(mode="after")
    def evidence_contract(self):
        if self.bridge_scope != "same_mgi_model" and (self.mgi_context_id or self.model_evidence_quote):
            field = "mgi_context_id" if self.mgi_context_id else "model_evidence_quote"
            raise contract_error("claim.mgi", field=(field,), expected="", actual=getattr(self, field))
        if self.bridge_scope == "same_mgi_model" and not (
            self.supporting_quote.strip() and self.mgi_context_id.strip() and self.model_evidence_quote.strip()
        ):
            raise contract_error("claim.mgi_required", field=("bridge_scope",))
        if not self.supporting_quote:
            if self.supporting_document_id:
                raise contract_error("claim.quote_source", field=("supporting_document_id",), expected="", actual=self.supporting_document_id)
            if self.evidence_mode != "inferred" or self.cell_evidence_mode != "inferred":
                field = "evidence_mode" if self.evidence_mode != "inferred" else "cell_evidence_mode"
                raise contract_error("claim.unsourced_modes", field=(field,), expected="inferred", actual=getattr(self, field))
            if self.link_evidence_mode == "direct" or self.bridge_scope == "direct_phenotype":
                raise contract_error("claim.unsourced_link", field=("bridge_scope",))
            if self.assertion != "model_inference" or self.evidence_type != "inference":
                raise contract_error("claim.unsourced_provenance", field=("assertion",))
        if self.link_evidence_mode is None:
            self.link_evidence_mode = "direct" if self.bridge_scope == "direct_phenotype" else "inferred"
        expected = "inferred" if "inferred" in (self.cell_evidence_mode, self.link_evidence_mode) else "direct"
        if self.evidence_mode != expected:
            raise contract_error("claim.mode_summary", field=("evidence_mode",), expected=expected, actual=self.evidence_mode)
        if (self.link_evidence_mode == "direct") != (self.bridge_scope == "direct_phenotype"):
            raise contract_error("claim.link_scope", field=("link_evidence_mode",))
        if self.evidence_mode == "direct":
            if self.reasoning_steps or self.bridge_scope != "direct_phenotype":
                raise contract_error("claim.direct", field=("reasoning_steps",))
        else:
            if self.relation_type == "causally_upstream_of":
                raise contract_error("claim.inference_relation", field=("relation_type",), actual=self.relation_type)
            if not self.reasoning_steps or not any(s.basis in {"inference", "model_knowledge"} for s in self.reasoning_steps):
                raise contract_error("claim.inference", field=("reasoning_steps",))
            if self.cell_evidence_mode == "inferred" and not any(
                s.basis in {"inference", "model_knowledge"} and s.scope == "cell_identity" for s in self.reasoning_steps
            ):
                raise contract_error("claim.cell_inference", field=("reasoning_steps",))
            for position, (left, right) in enumerate(zip(self.reasoning_steps, self.reasoning_steps[1:], strict=False), 2):
                if left.target != right.source:
                    raise contract_error("claim.chain", field=("reasoning_steps", position - 1, "source"),
                                         expected=left.target, actual=right.source)
            if self.reasoning_steps[-1].target != self.cellular_statement:
                raise contract_error("claim.endpoint", field=("reasoning_steps", len(self.reasoning_steps) - 1, "target"),
                                     expected=self.cellular_statement, actual=self.reasoning_steps[-1].target)
        return self

    @model_validator(mode="after")
    def taxon(self):
        expected = {"human": "9606", "mouse": "10090", "rat": "10116", "unknown": "0"}
        if self.species in expected and self.taxon_id != "NCBITaxon:" + expected[self.species]:
            raise contract_error("claim.taxon", field=("taxon_id",), expected="NCBITaxon:" + expected[self.species], actual=self.taxon_id)
        if self.species == "other_model" and self.taxon_id in {
            "NCBITaxon:9606", "NCBITaxon:10090", "NCBITaxon:10116", "NCBITaxon:0"
        }:
            raise contract_error("claim.taxon", field=("taxon_id",))
        return self


class ResearchGap(Record):
    missing_link: Text
    suggested_queries: list[str] = Field(default_factory=list)
    suggested_document_ids: list[str] = Field(default_factory=list)


class Extraction(Record):
    claims: list[ClaimDraft]
    rejection_reason: str = ""
    research_gaps: list[ResearchGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def explain_empty_result(self):
        if not self.claims and not self.rejection_reason.strip():
            raise ValueError("empty claims require a rejection_reason")
        return self


class SearchDirection(Record):
    direction_id: Text
    label: Text
    kind: Literal["phenotype", "disease", "pathology", "development", "cellular", "molecular", "model", "counterevidence"]
    rationale: Text
    queries: list[Text] = Field(min_length=1)


class DiscoveryPlan(Record):
    directions: list[SearchDirection]
    rationale: Text


class HypothesisDraft(Record):
    title: Text
    direction_id: Text
    cell_text: Text
    process_text: Text
    direction: Direction = Field(description=DIRECTION_DESCRIPTION)
    mechanism: Text
    why_distinct: Text
    seed_evidence: list[HypothesisSeedReference] = Field(min_length=1)
    model_knowledge_basis: str = ""


class HypothesisBatch(Record):
    hypotheses: list[HypothesisDraft]
    diversity_rationale: Text
    limitation: str = ""


class HypothesisAssessment(Record):
    decision: Literal["retain", "reject", "inconclusive"] = Field(
        description="Retain a supported, scientifically defensible prediction with explicit assumptions even without exact-site experiments or a challenge search. Reject concrete defects in necessary premises or unjustifiable applicability, not negative evidence labels alone. Inconclusive when the cell/mechanism or applicability cannot reasonably be justified. Format failures are not scientific decisions.")
    rationale: Text
    supporting_document_ids: list[str] = Field(default_factory=list, description="Assessment-level IDs of actually read supporting documents, not unexamined search hits.")
    contradicting_document_ids: list[str] = Field(default_factory=list, description="Assessment-level IDs of actually read contradictory sources; missing studies and alternative causes are not contradictions.")
    search_limitations: list[str] = Field(default_factory=list)
    revision_note: str = ""
    claim: ClaimDraft | None = None

    @model_validator(mode="after")
    def retained_claim(self):
        if (self.decision == "retain") != (self.claim is not None):
            raise ValueError("retain requires a final claim; reject/inconclusive must not publish one")
        return self


class AssessmentFieldUpdate(Record):
    path: list[str | int] = Field(min_length=1, description="Path in previous_submission, e.g. ['claim','supporting_document_id']. Scientific decision/rationale/cell/process/quality/direction cannot be patched.")
    operation: Literal["set", "remove"] = "set"
    value: JsonValue = Field(default=None, description="Replacement JSON value for set; ignored for remove. Copy IDs/quotes from the same read window.")


class AssessmentRepair(Record):
    updates: list[AssessmentFieldUpdate] = Field(min_length=1, description="Only targeted evidence/provenance field updates to the failed assessment; the host retains all other fields and revalidates the complete result.")


class ScientificReason(Record):
    explanation: Text = Field(description="Concise scientific justification, not a transcript of deliberation. Separate your interpretation from the selected source facts.")
    evidence_refs: list[Text] = Field(default_factory=list, description="Select evidence_ref IDs returned in this task's read_document passages; no quotes, offsets or publication metadata.")


class CellReason(ScientificReason):
    cell_text: Text = Field(description="The actual implicated cell population; anatomy and extracellular vesicles are context/products, not cell types.")


class AbnormalityReason(ScientificReason):
    process_text: str = Field(description="Cellular process; empty for a quality-only abnormality.")
    quality_text: Text = Field(description="The proposed cellular abnormality, consistent with the named process and direction.")
    direction: Direction = Field(description=DIRECTION_DESCRIPTION)


class PhenotypeReason(ScientificReason):
    relation_type: Literal["associated_with", "mechanistically_compatible_with"] = "mechanistically_compatible_with"


class ExperimentSelection(Record):
    evidence_ref: Text = Field(description="A read passage containing this actual study finding. The host copies its exact text, source and offsets.")
    scope: Literal["cell_identity", "cell_abnormality", "mechanism", "phenotype_link", "whole_claim"] = Field(
        description="The study's actual tested scope. Whole_claim requires the complete prediction and applicable species/model/anatomy/conditions; other-model findings usually support only a component.")
    study_type: Literal["observation", "pathology", "intervention"]
    species: Literal["human", "mouse", "rat", "other_model", "unknown"] = Field(description="Actual study species, not the species of the prediction or the gene alone.")
    experimental_context: Text
    finding: Text = Field(description="Actual study result, separately from your extrapolation. Reviews/speculation are not experiments. Not reporting a phenotype is not evidence that it was measured and absent.")
    effect: Literal["supports", "contradicts"] = "supports"


class CellularProposal(Record):
    cell: CellReason
    abnormality: AbnormalityReason
    phenotype_link: PhenotypeReason
    assumptions: list[Text] = Field(min_length=1, description="Inside claim.assumptions: scientific applicability assumptions for this prediction, once. Do not invent certainty at the exact anatomical site.")
    disease_context: str = Field(default="", description="Optional disease/subgroup limitation of this hypothesis, not a diagnosis of the input. Tissue/development/species applicability belongs in assumptions.")
    experimental_evidence: list[ExperimentSelection] = Field(default_factory=list, description="Inside claim.experimental_evidence: actual used primary-study findings with component/whole-claim scope; [] is allowed and does not block a reasonable prediction.")


class HypothesisProposal(Record):
    title: Text
    direction_id: Text
    cell_text: Text
    process_text: Text
    direction: Direction = Field(description=DIRECTION_DESCRIPTION)
    mechanism: Text
    why_distinct: Text
    seed_evidence_refs: list[Text] = Field(min_length=1, description="Read evidence_ref IDs that inspired this candidate; the host creates source quotes and offsets.")
    model_knowledge_basis: str = ""


class HypothesisProposals(Record):
    hypotheses: list[HypothesisProposal]
    diversity_rationale: Text
    limitation: str = ""


class ScientificAssessment(Record):
    decision: Literal["retain", "reject", "inconclusive"] = Field(description="Retain a supported, defensible prediction with explicit assumptions; challenge searches and exclusion of alternatives are optional. Reject concrete defects in necessary premises, not negative evidence labels alone. Use inconclusive when the cell/mechanism/applicability cannot reasonably be justified. Missing experiments alone are not rejection grounds.")
    rationale: Text
    supporting_evidence_refs: list[Text] = Field(default_factory=list)
    contradicting_evidence_refs: list[Text] = Field(default_factory=list)
    search_limitations: list[str] = Field(default_factory=list)
    revision_note: str = ""
    claim: CellularProposal | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_structure(cls, value):
        return normalize_assessment(value)[0]

    @model_validator(mode="after")
    def retained_proposal(self):
        if self.decision == "retain" and self.claim is None:
            raise contract_error("assessment.claim_required", field=("claim",))
        if self.decision != "retain" and self.claim is not None:
            raise contract_error("assessment.decision_claim", field=("claim",))
        return self


class PartialCellReason(Record):
    cell_text: Text | None = None
    explanation: Text | None = None
    evidence_refs: list[Text] | None = None


class PartialAbnormalityReason(Record):
    process_text: str | None = None
    quality_text: Text | None = None
    direction: Direction | None = None
    explanation: Text | None = None
    evidence_refs: list[Text] | None = None


class PartialPhenotypeReason(Record):
    explanation: Text | None = None
    evidence_refs: list[Text] | None = None
    relation_type: Literal["associated_with", "mechanistically_compatible_with"] | None = None


class PartialCellularProposal(Record):
    cell: PartialCellReason | None = None
    abnormality: PartialAbnormalityReason | None = None
    phenotype_link: PartialPhenotypeReason | None = None
    assumptions: list[Text] | None = None
    disease_context: str | None = None
    experimental_evidence: list[ExperimentSelection] | None = None


class ScientificSupplement(Record):
    decision: Literal["retain", "reject", "inconclusive"] | None = None
    rationale: Text | None = None
    claim: PartialCellularProposal | None = None
    supporting_evidence_refs: list[Text] | None = None
    contradicting_evidence_refs: list[Text] | None = None
    search_limitations: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_structure(cls, value):
        return normalize_assessment(value)[0]


class OntologySelection(Record):
    cl_id: str
    go_id: str
    cell_confidence: Literal["high", "medium", "low", "unmapped"]
    process_confidence: Literal["high", "medium", "low", "unmapped"]
    rationale: Text


class ScientificJudgment(Record):
    status: Literal["defensible", "uncertain", "invalid"] = Field(description="uncertain means reasonable but unverified; invalid means a concrete factual/logical defect or unjustifiable applicability, not merely missing experiments.")
    reason: Text = Field(description="Explain this scientific assessment. For invalid, name the conflicting propositions and their source or biological basis.")


class ScientificReview(Record):
    cell_identity: ScientificJudgment
    cell_abnormality: ScientificJudgment
    phenotype_link: ScientificJudgment = Field(description="Judge mechanism, anatomical applicability and intended-species transfer together; exact-site experiments are optional.")
    ontology_mapping: ScientificJudgment
    experimental_evidence: ScientificJudgment | None = Field(default=None, description="Required when experimental entries are present; omit when none are recorded. Judge study interpretation/species/scope, not whether experiments are required.")
    research_gaps: list[ResearchGap] = Field(default_factory=list)


class ScientificCompletion(Record):
    stop_reason: Literal["sufficient_evidence", "budget_exhausted", "no_evidence", "needs_curation"]
    summary: Text = Field(description="Scientific synthesis and limitations, not a hand-counted workflow report. The host adds exact counts and current state_version.")
    limitations: list[str]


class EvidenceSpan(Record):
    document_id: Text
    source_url: Text
    pmid: str
    pmcid: str
    doi: str
    source_kind: str
    text_sha256: Text
    span_start: int
    span_end: int
    quote: Text
    section: str


class GroundedReasoningStep(Record):
    scope: Literal["phenotype_link", "cell_identity", "mechanism"] = "phenotype_link"
    source: Text
    target: Text
    relation: Text
    basis: ReasoningBasis
    explanation: Text
    assumptions: list[str]
    evidence: list[EvidenceSpan]


class ExperimentalEvidence(Record):
    scope: Literal["cell_identity", "cell_abnormality", "mechanism", "phenotype_link", "whole_claim"]
    study_type: Literal["observation", "pathology", "intervention"]
    species: Literal["human", "mouse", "rat", "other_model", "unknown"]
    experimental_context: Text
    finding: Text
    effect: Literal["supports", "contradicts"]
    evidence: EvidenceSpan


class ExperimentalValidation(Record):
    status: Literal["not_recorded", "component_support", "claim_support", "contradictory", "mixed"] = Field(
        default="not_recorded", description="Summary of recorded study findings and their selected scope, not the scientific verdict on the inferred prediction or proof of target-specific causality.")
    evidence: list[ExperimentalEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def support_scope(self):
        supports = [e for e in self.evidence if e.effect == "supports"]
        contradicts = any(e.effect == "contradicts" for e in self.evidence)
        expected = "mixed" if supports and contradicts else "contradictory" if contradicts else (
            "claim_support" if any(e.scope == "whole_claim" for e in supports)
            else "component_support" if supports else "not_recorded")
        if self.status != "not_recorded" and self.status != expected:
            raise ValueError("experimental validation status must match its recorded evidence scope")
        self.status = expected
        return self


class OntologyTerm(Record):
    term_id: Annotated[str, Field(pattern=r"^(CL|GO):[0-9]{7}$")]
    label: Text
    namespace: str = ""
    definition: str = ""
    synonyms: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    alt_ids: list[str] = Field(default_factory=list)
    obsolete: bool = False
    replaced_by: list[str] = Field(default_factory=list)


class MappingChoice(Record):
    cl_id: str
    go_id: str
    cl_label: str = ""
    go_label: str = ""
    cell_confidence: Literal["high", "medium", "low", "unmapped"]
    process_confidence: Literal["high", "medium", "low", "unmapped"]
    rationale: Text


class Mapping(Record):
    cl: OntologyTerm | None
    go: OntologyTerm | None
    cell_confidence: str
    process_confidence: str
    rationale: str
    method: str = "llm_term_selection_then_local_id_label_validation"
    ontology_sha256: str
    candidate_ids: list[str]


class ReviewIssue(Record):
    code: Literal["source_misinterpretation", "logical_inconsistency", "contradictory_evidence",
                  "inapplicable_scope", "cell_misidentification", "direction_mismatch"]
    explanation: Text = Field(description="Identify the concrete conflicting propositions and source/biological basis. Direction mismatch concerns the same named process/comparison, not opposite signs across upstream process and downstream phenotype. Missing experiments alone are not a blocker.")


class Review(Record):
    verdict: Literal["supported", "association_only", "partial", "contradictory", "insufficient_evidence"]
    cell_supported: bool
    process_supported: bool
    abnormality_supported: bool
    phenotype_bridge_supported: bool
    species_supported: bool
    mapping_supported: bool
    model_match_supported: bool
    causal_supported: bool
    inference_supported: bool = Field(default=False, description="Scientific coherence, not experimental confirmation.")
    scope_transfer_supported: bool = Field(default=False, description="Defensible applicability assumptions; exact-site experiments are optional.")
    cell_inference_supported: bool = False
    experimental_evidence_supported: bool = True
    blocking_issues: list[ReviewIssue] = Field(default_factory=list)
    reason: Text
    research_gaps: list[ResearchGap] = Field(default_factory=list)


class ResearchSelection(Record):
    pmids: list[Annotated[str, Field(pattern=r"^[0-9]+$")]]
    rationale: Text


class ResearchCompletion(Record):
    state_version: int = Field(ge=0)
    stop_reason: Literal["sufficient_evidence", "budget_exhausted", "no_evidence", "needs_curation"]
    summary: Text
    limitations: list[str]


class Observation(Record):
    claim_id: str
    phenotype_id: PhenotypeID
    phenotype_label: str
    claim: ClaimDraft
    evidence_mode: EvidenceMode
    cell_evidence_mode: EvidenceMode | None = None
    link_evidence_mode: EvidenceMode | None = None
    knowledge_source: KnowledgeSource | None = None
    evidence: EvidenceSpan | None
    experimental_validation: ExperimentalValidation = Field(default_factory=ExperimentalValidation)
    reasoning_steps: list[GroundedReasoningStep] = Field(default_factory=list)
    reasoning_chain: str = ""
    mapping: Mapping
    review: Review
    release_scope: Literal["primary", "conditional", "cross_species", "unresolved", "rejected"]
    gate_reasons: list[str]
    pair_id: str = ""
    cell_abnormality_id: str = ""
    edge_source: str = ""
    edge_target: str = ""

    @model_validator(mode="after")
    def identity(self):
        self.claim = self.claim.bind_phenotype(self.phenotype_id, self.phenotype_label)
        if self.reasoning_steps:
            first = self.reasoning_steps[0]
            canonical = canonical_reasoning_start(first.source, self.phenotype_id, self.phenotype_label)
            if first.source != canonical:
                self.reasoning_steps = [first.model_copy(update={"source": canonical}), *self.reasoning_steps[1:]]
        if self.knowledge_source is not None and self.knowledge_source != self.claim.knowledge_source:
            raise ValueError("knowledge_source does not match claim premises")
        self.knowledge_source = self.claim.knowledge_source
        if self.claim.supporting_quote and self.evidence is None:
            raise ValueError("source quotation requires an anchored evidence record")
        if not self.claim.supporting_quote and self.evidence is not None:
            raise ValueError("model-only claim cannot fabricate a source evidence record")
        for field in ("cell_evidence_mode", "link_evidence_mode"):
            expected = getattr(self.claim, field)
            if getattr(self, field) is not None and getattr(self, field) != expected:
                raise ValueError(f"{field} does not match claim")
            setattr(self, field, expected)
        if self.evidence_mode != self.claim.evidence_mode:
            raise ValueError("evidence_mode does not match claim")
        if len(self.experimental_validation.evidence) != len(self.claim.experimental_evidence):
            raise ValueError("experimental evidence must match the claim's recorded experiments")
        for draft, grounded in zip(self.claim.experimental_evidence, self.experimental_validation.evidence, strict=True):
            if (draft.model_dump(exclude={"document_id", "quote"}) != grounded.model_dump(exclude={"evidence"})
                    or draft.document_id != grounded.evidence.document_id or draft.quote != grounded.evidence.quote):
                raise ValueError("grounded experimental evidence differs from claim")
        if len(self.reasoning_steps) != len(self.claim.reasoning_steps):
            raise ValueError("grounded reasoning steps must match claim")
        for draft, grounded in zip(self.claim.reasoning_steps, self.reasoning_steps, strict=True):
            if draft.model_dump(exclude={"citations"}) != grounded.model_dump(exclude={"evidence"}):
                raise ValueError("grounded reasoning content differs from claim")
            if len(draft.citations) != len(grounded.evidence) or any(
                citation.document_id != span.document_id or citation.quote != span.quote
                or (citation.span_start is not None and citation.span_start != span.span_start)
                for citation, span in zip(draft.citations, grounded.evidence, strict=True)
            ):
                raise ValueError("grounded reasoning citations differ from claim")
        chain = ""
        if self.reasoning_steps:
            if self.reasoning_steps[0].source != self.phenotype_label:
                raise ValueError("reasoning chain must start at phenotype label")
            chain = " -> ".join([self.phenotype_label, *[s.target for s in self.reasoning_steps]])
        if self.reasoning_chain and self.reasoning_chain != chain:
            raise ValueError("reasoning_chain must match structured steps")
        self.reasoning_chain = chain
        pair = ""
        node_id = ""
        if self.mapping.cl and self.mapping.go:
            pair = f"{self.mapping.cl.term_id}|{self.mapping.go.term_id}"
            node_id = "cellabn_" + hashlib.sha256((pair + "|" + self.claim.direction).encode()).hexdigest()[:24]
        if self.pair_id and self.pair_id != pair:
            raise ValueError("pair_id does not match normalized CL/GO")
        if self.cell_abnormality_id and self.cell_abnormality_id != node_id:
            raise ValueError("cell_abnormality_id does not match CL/GO/direction")
        self.pair_id = pair
        self.cell_abnormality_id = node_id
        source, target = (self.phenotype_id, node_id) if node_id else ("", "")
        if self.claim.relation_type == "causally_upstream_of":
            source, target = target, source
        if self.edge_source and self.edge_source != source:
            raise ValueError("edge_source does not match relation orientation")
        if self.edge_target and self.edge_target != target:
            raise ValueError("edge_target does not match relation orientation")
        self.edge_source, self.edge_target = source, target
        return self


class TermResult(Record):
    schema_version: str = SCHEMA_VERSION
    run_fingerprint: str
    phenotype: Phenotype
    status: Literal["completed", "partial", "processing_failed"]
    outcome: Literal["projected", "conditional_only", "normalization_unresolved", "insufficient_evidence", "processing_failed"]
    observations: list[Observation]
    document_ids: list[str]
    search_log: list[dict]
    coverage: dict
    errors: list[dict]


class Settings(Record):
    workflow_mode: Literal["literature_first", "component_test"] = "literature_first"
    min_hypotheses: int = Field(default=1, ge=1)
    min_discovery_directions: int = Field(default=3, ge=3)
    min_discovery_papers: int = Field(default=1, ge=1)
    max_total_searches: int = Field(default=300, ge=1)
    max_total_subagent_calls: int = Field(default=240, ge=1)
    max_models: int | None = Field(default=None, ge=1)
    run_mode: Literal["research", "smoke_test"] = "research"
    max_searches: int = Field(default=30, ge=1, le=1000)
    results_per_search: int = Field(default=10, ge=1, le=100)
    max_papers: int = Field(default=40, ge=1, le=500)
    max_full_texts: int = Field(default=8, ge=0, le=100)
    chunk_chars: int = Field(default=16000, ge=1000)
    chunk_overlap: int = Field(default=800, ge=0)
    max_chunks_per_paper: int = Field(default=16, ge=1)
    term_workers: int = Field(default=1, ge=1, le=32)
    llm_workers: int = Field(default=3, ge=1, le=64)
    model_min_interval_seconds: float = Field(default=0, ge=0)
    model_rate_limit_retries: int = Field(default=4, ge=0, le=12)
    model_connection_retries: int = Field(default=4, ge=0, le=12)
    model_retry_base_seconds: float = Field(default=15, ge=0)
    model_request_timeout_seconds: float = Field(default=120, gt=0)
    max_normalization_tool_calls: int = Field(default=12, ge=1)
    timeout_seconds: float = Field(default=900, gt=0)
    recursion_limit: int = Field(default=100, ge=5)
    max_repeated_tool_results: int = Field(default=3, ge=2)
    max_redundant_reads: int = Field(default=8, ge=2)
    max_subagent_calls: int = Field(default=120, ge=1, le=2000)
    eval_timeout_seconds: float = Field(default=900, gt=0)
    max_ptc_calls: int = Field(default=256, ge=1)
    offline: bool = False

    @model_validator(mode="after")
    def overlap(self):
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be less than chunk_chars")
        if self.workflow_mode == "literature_first":
            if self.max_searches < self.min_discovery_directions + self.min_hypotheses:
                raise ValueError("max_searches must cover discovery plus supporting searches for every hypothesis")
            if self.max_papers < self.min_discovery_papers:
                raise ValueError("max_papers must cover the discovery corpus minimum")
        return self
