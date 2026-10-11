"""Deterministic evidence integrity and release gates; the LLM cannot bypass these."""

from __future__ import annotations

import hashlib

from pydantic_core import PydanticCustomError

from .contracts import COMMON_SUPPORT_FIELDS, INFERRED_SUPPORT_FIELDS, contract_error
from .ontology import OntologyIndex
from .schemas import (
    ClaimDraft,
    Document,
    EvidenceSpan,
    ExperimentalEvidence,
    GroundedReasoningStep,
    Mapping,
    MappingChoice,
    Phenotype,
    QuoteReference,
    Review,
)


def anchor_quote(reference: QuoteReference, document: Document, start: int, end: int) -> EvidenceSpan:
    if document.retracted:
        raise ValueError("retracted_source")
    offset = reference.span_start
    if offset is None:
        relative = document.text[start:end].find(reference.quote)
        if relative < 0:
            raise contract_error("citation.quote", field=("quote",))
        offset = start + relative
    stop = offset + len(reference.quote)
    if offset < start or stop > end or document.text[offset:stop] != reference.quote:
        relative = document.text[start:end].find(reference.quote)
        raise contract_error("citation.offset", field=("span_start",),
                             expected=start + relative if relative >= 0 else f"a matching offset in window {start}:{end}", actual=offset)
    section = next((s.name for s in document.sections if s.start <= offset < s.end), "unspecified")
    return EvidenceSpan(
        document_id=document.document_id, source_url=document.source_url,
        pmid=document.pmid, pmcid=document.pmcid, doi=document.doi,
        source_kind=document.source_kind, text_sha256=document.sha256,
        span_start=offset, span_end=stop, quote=document.text[offset:stop], section=section,
    )


def anchor_claim(claim: ClaimDraft, document: Document | None, start: int, end: int,
                 *, provided: list[dict] | None = None) -> EvidenceSpan | None:
    ClaimDraft.model_validate(claim.model_dump())
    if not claim.supporting_quote:
        return None
    if document is None:
        raise ValueError("quoted_evidence_requires_a_staged_document")
    window = document.text[start:end]
    relative = window.find(claim.supporting_quote)
    if relative < 0:
        raise ValueError("quote_not_in_supplied_window")
    if claim.cell_evidence_mode == "direct" and claim.cell_text not in claim.supporting_quote:
        raise ValueError("cell_not_in_quote")
    if document.retracted:
        raise ValueError("retracted_source")
    return anchor_quote(QuoteReference(document_id=document.document_id, quote=claim.supporting_quote),
                        document, start, end)


def matching_quote_windows(quote: str, documents: dict[str, Document], provided: list[dict]) -> list[dict]:
    """Replay contexts may contain ranges only; immutable stored text is authoritative."""
    return [window for window in provided
            if window["document_id"] in documents
            and quote in documents[window["document_id"]].text[window["start"]:window["end"]]]


def main_quote_window(claim: ClaimDraft, documents: dict[str, Document], provided: list[dict]) -> dict | None:
    """Diagnose source identity separately from exact text and read coverage."""
    if not claim.supporting_quote:
        return None
    selected, quote = claim.supporting_document_id, claim.supporting_quote
    read_ids = {window["document_id"] for window in provided}
    if selected and selected not in documents:
        raise contract_error("claim.source_unstaged", field=("supporting_document_id",), actual=selected)
    if selected and selected not in read_ids:
        raise contract_error("claim.source_unread", field=("supporting_document_id",), actual=selected)
    matches = matching_quote_windows(quote, documents, provided)
    matching_ids = sorted({window["document_id"] for window in matches})
    if selected:
        own_matches = [window for window in matches if window["document_id"] == selected]
        if own_matches:
            return own_matches[0]
        if quote in documents[selected].text:
            raise contract_error("claim.quote_unread", field=("supporting_quote",), actual=quote)
        if matching_ids:
            raise contract_error("claim.source_mismatch", field=("supporting_document_id",),
                                 expected=matching_ids, actual=selected,
                                 related_paths=[["claim", "reasoning_steps", i, "citations", j, "document_id"]
                                     for i, step in enumerate(claim.reasoning_steps) for j, citation in enumerate(step.citations)
                                     if citation.document_id == selected and citation.quote == quote])
        raise contract_error("claim.quote_verbatim", field=("supporting_quote",), actual=quote)
    if len(matching_ids) > 1:
        raise contract_error("claim.source_ambiguous", field=("supporting_document_id",),
                             expected=matching_ids, actual="")
    if not matches:
        raise contract_error("claim.quote_verbatim", field=("supporting_quote",), actual=quote)
    return matches[0]


def anchor_reasoning(
    phenotype: Phenotype, claim: ClaimDraft, evidence: EvidenceSpan | None,
    documents: dict[str, Document], provided: list[dict],
) -> list[GroundedReasoningStep]:
    claim = claim.bind_phenotype(phenotype.phenotype_id, phenotype.label)
    # Validate even callers that constructed a draft with model_copy(update=...).
    ClaimDraft.model_validate(claim.model_dump())
    if claim.evidence_mode == "direct":
        return []
    grounded = []
    for position, step in enumerate(claim.reasoning_steps):
        if step.basis == "phenotype_definition" and position != 0:
            raise ValueError("definition_step_must_start_from_supplied_phenotype")
        spans = []
        for citation_index, citation in enumerate(step.citations):
            field = ("reasoning_steps", position, "citations", citation_index)
            doc = documents.get(citation.document_id)
            if doc is None:
                raise contract_error("reasoning.source", field=(*field, "document_id"), actual=citation.document_id)
            if doc.retracted:
                raise ValueError("retracted_source")
            span = None
            errors = []
            for window in provided:
                if window["document_id"] != doc.document_id:
                    continue
                try:
                    span = anchor_quote(citation, doc, window["start"], window["end"])
                    break
                except ValueError as error:
                    errors.append(error)
            if span is None:
                matching_ids = sorted({window["document_id"] for window in matching_quote_windows(citation.quote, documents, provided)
                                       if window["document_id"] != citation.document_id})
                if matching_ids and citation.quote not in doc.text:
                    raise contract_error("citation.source_mismatch", field=(*field, "document_id"),
                                         expected=matching_ids, actual=citation.document_id)
                offset_error = next((e for e in errors if isinstance(e, PydanticCustomError) and e.type == "citation.offset"), None)
                if offset_error:
                    raise contract_error("citation.offset", field=(*field, "span_start"),
                                         expected=offset_error.context["expected"], actual=citation.span_start)
                raise contract_error("reasoning.quote", field=(*field, "quote"))
            spans.append(span)
        grounded.append(GroundedReasoningStep(**step.model_dump(exclude={"citations"}), evidence=spans))
    if evidence is not None and not any(evidence in step.evidence for step in grounded):
        raise ValueError("reasoning_chain_must_cite_cellular_abnormality_quote")
    return grounded


def resolve_mapping(choice: MappingChoice, index: OntologyIndex, seen: set[str]) -> Mapping:
    resolved = []
    for prefix, term_id, confidence, label in (
        ("CL", choice.cl_id, choice.cell_confidence, choice.cl_label),
        ("GO", choice.go_id, choice.process_confidence, choice.go_label),
    ):
        if term_id in {"", "NONE"}:
            if label:
                raise contract_error("mapping.label", field=(prefix.lower() + "_label",), expected="", actual=label)
            if confidence != "unmapped":
                raise contract_error("mapping.label", field=("cell_confidence" if prefix == "CL" else "process_confidence",), expected="unmapped", actual=confidence)
            resolved.append(None)
            continue
        item = index.get(term_id)
        if not item or item.obsolete or not item.term_id.startswith(prefix + ":"):
            raise contract_error("mapping.id", field=(prefix.lower() + "_id",), expected=prefix + " non-obsolete index ID", actual=term_id)
        if label and label != item.label:
            raise contract_error("mapping.canonical", field=(prefix.lower() + "_label",), expected=item.label, actual=label)
        if confidence == "unmapped":
            raise contract_error("mapping.confidence", field=("cell_confidence" if prefix == "CL" else "process_confidence",))
        resolved.append(item)
    return Mapping(
        cl=resolved[0], go=resolved[1], cell_confidence=choice.cell_confidence,
        process_confidence=choice.process_confidence, rationale=choice.rationale,
        ontology_sha256=index.sha256, candidate_ids=sorted(seen | {item.term_id for item in resolved if item}),
    )


def anchor_experiments(claim: ClaimDraft, documents: dict[str, Document], provided: list[dict]) -> list[ExperimentalEvidence]:
    experiments = []
    for position, draft in enumerate(claim.experimental_evidence, 1):
        if draft.scope == "whole_claim" and (draft.species == "unknown" or draft.species != claim.species):
            raise ValueError(f"experimental_evidence[{position}]: whole_claim_requires_matching_known_species; use a component scope")
        doc = documents.get(draft.document_id)
        if doc is None:
            raise ValueError(f"experimental_evidence[{position}]: document_not_staged")
        span = None
        errors = []
        for window in provided:
            if window["document_id"] != draft.document_id:
                continue
            try:
                span = anchor_quote(QuoteReference(document_id=draft.document_id, quote=draft.quote),
                                    doc, window["start"], window["end"])
                break
            except ValueError as error:
                errors.append(str(error))
        if span is None:
            matching_ids = sorted({window["document_id"] for window in matching_quote_windows(draft.quote, documents, provided)
                                   if window["document_id"] != draft.document_id})
            if matching_ids and draft.quote not in doc.text:
                raise contract_error("citation.source_mismatch", field=("experimental_evidence", position - 1, "document_id"),
                                     expected=matching_ids, actual=draft.document_id)
            raise ValueError(f"experimental_evidence[{position}]: " + (errors[-1] if errors else "source_not_read"))
        experiments.append(ExperimentalEvidence(**draft.model_dump(exclude={"document_id", "quote"}), evidence=span))
    return experiments


def release_gate(
    phenotype: Phenotype, claim: ClaimDraft, evidence: EvidenceSpan | None,
    mapping: Mapping, review: Review, reasoning_steps: list[GroundedReasoningStep] | None = None,
) -> tuple[str, list[str]]:
    reasons = []
    if review.blocking_issues:
        return "rejected", ["scientific_review_blocked", *[issue.code for issue in review.blocking_issues]]
    if claim.experimental_evidence and not review.experimental_evidence_supported:
        return "rejected", ["experimental_evidence_misinterpreted", review.reason]
    # Evidence labels describe findings; the scientific review decides applicability.
    if claim.evidence_mode == "direct" and (claim.bridge_scope != "direct_phenotype" or claim.reasoning_steps):
        return "rejected", ["indirect_bridge_requires_inferred_evidence_and_chain"]
    if claim.evidence_mode == "inferred":
        if not reasoning_steps or len(reasoning_steps) != len(claim.reasoning_steps):
            return "rejected", ["inference_chain_not_grounded"]
        if not all(getattr(review, field) for field in INFERRED_SUPPORT_FIELDS):
            return "rejected", ["inference_review_failed", review.reason]
        if claim.cell_evidence_mode == "inferred" and not review.cell_inference_supported:
            return "rejected", ["cell_identity_inference_not_supported", review.reason]
        reasons.append("literature_grounded_inference" if claim.knowledge_source == "literature"
                       else "model_knowledge_inference")
        # A plausible causal explanation is retained as mechanistic compatibility.
        # A demonstrated causal edge still requires direct evidence for this target.
        if claim.relation_type == "causally_upstream_of":
            return "rejected", ["inferred_link_requires_noncausal_relation_type"]
    supported = all(getattr(review, field) for field in COMMON_SUPPORT_FIELDS)
    if (not supported or review.verdict == "contradictory"
            or review.verdict == "insufficient_evidence" and claim.evidence_mode == "direct"):
        return "rejected", ["evidence_review_failed", review.reason]
    if claim.relation_type == "causally_upstream_of" and (
        not review.causal_supported or review.verdict == "association_only"
    ):
        return "rejected", ["causality_not_supported"]
    if claim.species == "unknown":
        return "unresolved", ["unknown_species"]
    if claim.bridge_scope == "supplied_disease" and claim.disease_context not in phenotype.disease_contexts:
        return "rejected", ["unsupplied_disease_context"]
    if mapping.cl is None or mapping.go is None or not review.mapping_supported:
        return "unresolved", ["normalization_unresolved"]
    if mapping.cell_confidence == "low" or mapping.process_confidence == "low":
        return "unresolved", ["low_confidence_mapping"]
    if review.verdict in {"partial", "insufficient_evidence"}:
        if claim.evidence_mode == "direct":
            return "unresolved", ["partial_review"]
        reasons.append("inference_review_uncertainty")
    if claim.species != phenotype.target_species:
        return "cross_species", [*reasons, "evidence_species_differs_from_target"]
    if claim.bridge_scope != "direct_phenotype":
        reasons.append("indirect_phenotype_bridge")
    if claim.assertion not in {"own_result", "model_inference"}:
        reasons.append("secondary_statement")
    if claim.direction == "unknown":
        reasons.append("unknown_abnormal_direction")
    if claim.relation_type == "mechanistically_compatible_with":
        reasons.append("mechanism_only")
    return ("conditional", reasons) if reasons else ("primary", [])


def claim_id(phenotype: Phenotype, draft: ClaimDraft, evidence: EvidenceSpan | None,
             reasoning_steps: list[GroundedReasoningStep] | None = None) -> str:
    parts = [phenotype.phenotype_id, evidence.text_sha256 if evidence else "model_knowledge",
             str(evidence.span_start) if evidence else "",
             draft.model_dump_json(), *[s.model_dump_json() for s in (reasoning_steps or [])]]
    return "claim_" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
