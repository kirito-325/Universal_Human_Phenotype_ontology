"""Assemble strict research records from small, scientific model submissions."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import ValidationError

from .contracts import contract_error
from .schemas import (
    ClaimDraft,
    Document,
    ExperimentalEvidenceDraft,
    HypothesisAssessment,
    HypothesisBatch,
    MappingChoice,
    Phenotype,
    QuoteReference,
    ReasoningStep,
    ResearchCompletion,
    Review,
    ReviewIssue,
    ScientificAssessment,
    ScientificCompletion,
    ScientificReview,
)
from .validation import anchor_quote


def read_passages(document: Document, start: int, end: int) -> list[dict]:
    """Give immutable paragraph slices stable identities; no model-computed offsets."""
    passages = []
    source_hash = document.sha256
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", document.text[start:end]):
        left, right = start + match.start(), start + match.end()
        for offset in range(left, right, 1800):
            stop = min(right, offset + 1800)
            text = document.text[offset:stop]
            if not text.strip():
                continue
            identity = f"{document.document_id}|{source_hash}|{offset}|{stop}"
            passages.append({"evidence_ref": "ev_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                             "start": offset, "end": stop, "text": text})
    return passages


def model_window(window: dict, catalog=None) -> dict:
    return {key: value for key, value in window.items() if key not in {"text", "passages"}} | {
        "passages": [{"evidence_ref": catalog.alias(passage["evidence_ref"]) if catalog else passage["evidence_ref"], "text": passage["text"]}
                     for passage in window["passages"]]}


def resolve_refs(refs, documents, provided, *, field, scalar=False):
    available = {}
    for window in provided:
        doc = documents[window["document_id"]]
        for passage in read_passages(doc, window["start"], window["end"]):
            available[passage["evidence_ref"]] = (doc, passage)
    spans = []
    for position, raw in enumerate(refs):
        catalog = getattr(provided, "catalog", None)
        ref = catalog.resolve(raw) if catalog else raw.strip().lower()
        if ref not in available:
            raise contract_error("evidence.ref", field=field if scalar else (*field, position),
                                 expected="an evidence_ref returned by this task's read_document passages", actual=raw)
        doc, passage = available[ref]
        span = anchor_quote(QuoteReference(document_id=doc.document_id, quote=passage["text"],
                                          span_start=passage["start"]), doc, passage["start"], passage["end"])
        if span not in spans:
            spans.append(span)
    return spans


def check_references(selections, documents, provided):
    """Report all unknown selections together so one supplement can address them."""
    errors = []
    for ref, path in selections:
        try:
            resolve_refs([ref], documents, provided, field=path, scalar=True)
        except ValueError as error:
            if getattr(error, "type", None) != "evidence.ref":
                raise
            errors.append({"type": error, "loc": (), "input": ref})
    if errors:
        raise ValidationError.from_exception_data("EvidenceReferences", errors)


def assemble_hypotheses(value, documents, provided):
    check_references([(ref, ("hypotheses", i, "seed_evidence_refs", j))
        for i, proposal in enumerate(value.hypotheses) for j, ref in enumerate(proposal.seed_evidence_refs)], documents, provided)
    hypotheses = []
    for position, proposal in enumerate(value.hypotheses):
        spans = resolve_refs(proposal.seed_evidence_refs, documents, provided,
                             field=("hypotheses", position, "seed_evidence_refs"))
        hypotheses.append({**proposal.model_dump(exclude={"seed_evidence_refs"}),
                           "seed_evidence": [{"document_id": span.document_id, "quote": span.quote} for span in spans]})
    return HypothesisBatch(hypotheses=hypotheses, diversity_rationale=value.diversity_rationale, limitation=value.limitation)


def assemble_assessment(value: ScientificAssessment, phenotype: Phenotype, hypothesis_id: str, documents, provided,
                        *, original=None, previous_assessment=None, reassessment_reason=""):
    selections = [(ref, (field, j)) for field in ("supporting_evidence_refs", "contradicting_evidence_refs")
                  for j, ref in enumerate(getattr(value, field))]
    if value.claim:
        selections.extend((ref, ("claim", block, "evidence_refs", j)) for block in ("cell", "abnormality", "phenotype_link")
                          for j, ref in enumerate(getattr(value.claim, block).evidence_refs))
        selections.extend((entry.evidence_ref, ("claim", "experimental_evidence", i, "evidence_ref"))
                          for i, entry in enumerate(value.claim.experimental_evidence))
    check_references(selections, documents, provided)
    supporting = resolve_refs(value.supporting_evidence_refs, documents, provided, field=("supporting_evidence_refs",))
    contradicting = resolve_refs(value.contradicting_evidence_refs, documents, provided, field=("contradicting_evidence_refs",))
    claim = None
    if value.claim:
        proposal = value.claim
        steps = []
        source = phenotype.label
        endpoint = f"{proposal.cell.cell_text}: {proposal.abnormality.quality_text}"
        groups = (
            ("cell", "cell_identity", proposal.cell, proposal.cell.cell_text, "candidate_cell_identity"),
            ("abnormality", "mechanism", proposal.abnormality,
             proposal.abnormality.process_text or proposal.abnormality.quality_text, "proposed_cell_abnormality"),
            ("phenotype_link", "phenotype_link", proposal.phenotype_link, endpoint, proposal.phenotype_link.relation_type),
        )
        for name, scope, reason, target, relation in groups:
            spans = resolve_refs(reason.evidence_refs, documents, provided, field=("claim", name, "evidence_refs"))
            supporting.extend(span for span in spans if span not in supporting)
            steps.append(ReasoningStep(scope=scope, source=source, target=target, relation=relation,
                basis="inference" if spans else "model_knowledge", explanation=reason.explanation,
                assumptions=proposal.assumptions,
                citations=[QuoteReference(document_id=span.document_id, quote=span.quote, span_start=span.span_start)
                           for span in spans]))
            source = target
        experiments = []
        for position, selection in enumerate(proposal.experimental_evidence):
            span, = resolve_refs([selection.evidence_ref], documents, provided,
                                 field=("claim", "experimental_evidence", position, "evidence_ref"), scalar=True)
            experiments.append(ExperimentalEvidenceDraft(**selection.model_dump(exclude={"evidence_ref"}),
                                                         document_id=span.document_id, quote=span.quote))
            target = contradicting if selection.effect == "contradicts" else supporting
            if span not in target:
                target.append(span)
        species = phenotype.target_species
        claim = ClaimDraft(hypothesis_id=hypothesis_id,
            cell_text=proposal.cell.cell_text, process_text=proposal.abnormality.process_text,
            quality_text=proposal.abnormality.quality_text, direction=proposal.abnormality.direction,
            rationale=value.rationale, disease_context=proposal.disease_context, anatomy_context=phenotype.label,
            evidence_mode="inferred", cell_evidence_mode="inferred", link_evidence_mode="inferred",
            relation_type=proposal.phenotype_link.relation_type, bridge_scope="mechanistically_compatible",
            species=species, taxon_id="NCBITaxon:" + ("9606" if species == "human" else "10090"),
            assertion="model_inference", evidence_type="inference", gene_mentions_ignored=[],
            reasoning_steps=steps, experimental_evidence=experiments)
    note = value.revision_note
    if not note.strip():
        previous = (previous_assessment or {}).get("claim") or original or {}
        changes = {name: {"before": previous.get(name), "after": getattr(claim, name)}
                   for name in ("cell_text", "process_text", "direction")
                   if claim is not None and name in previous and previous[name] != getattr(claim, name)}
        previous_decision = (previous_assessment or {}).get("decision")
        if previous_decision and previous_decision != value.decision:
            changes["decision"] = {"before": previous_decision, "after": value.decision}
        if changes:
            note = "Host-recorded changes: " + json.dumps(changes, ensure_ascii=False, sort_keys=True) + ". Scientific basis: " + value.rationale
        elif reassessment_reason:
            note = "Host-recorded reassessment: " + reassessment_reason + ". Scientific basis: " + value.rationale
    return HypothesisAssessment(decision=value.decision, rationale=value.rationale, claim=claim,
        supporting_document_ids=sorted({span.document_id for span in supporting}),
        contradicting_document_ids=sorted({span.document_id for span in contradicting}),
        search_limitations=value.search_limitations, revision_note=note)


def assemble_mapping(value, index, seen):
    def label(term_id):
        term = index.get(term_id) if term_id not in {"", "NONE"} else None
        return term.label if term else ""
    return MappingChoice(**value.model_dump(), cl_label=label(value.cl_id), go_label=label(value.go_id))


def assemble_review(value: ScientificReview, has_experiments: bool):
    if has_experiments and value.experimental_evidence is None:
        raise contract_error("review.experiments", field=("experimental_evidence",))
    checks = {name: getattr(value, name) for name in
              ("cell_identity", "cell_abnormality", "phenotype_link", "ontology_mapping")}
    if has_experiments:
        checks["experimental_evidence"] = value.experimental_evidence
    codes = {"cell_identity": "cell_misidentification", "cell_abnormality": "logical_inconsistency",
             "phenotype_link": "inapplicable_scope", "ontology_mapping": "logical_inconsistency",
             "experimental_evidence": "source_misinterpretation"}
    issues = [ReviewIssue(code=codes[name], explanation=judgment.reason) for name, judgment in checks.items()
              if judgment.status == "invalid"]
    ok = {name: judgment.status != "invalid" for name, judgment in checks.items()}
    uncertain = any(judgment.status == "uncertain" for judgment in checks.values())
    return Review(verdict="contradictory" if issues else "partial" if uncertain else "supported",
        cell_supported=ok["cell_identity"], process_supported=ok["cell_abnormality"],
        abnormality_supported=ok["cell_abnormality"], phenotype_bridge_supported=ok["phenotype_link"],
        species_supported=ok["phenotype_link"], mapping_supported=ok["ontology_mapping"],
        inference_supported=ok["cell_abnormality"] and ok["phenotype_link"],
        scope_transfer_supported=ok["phenotype_link"], cell_inference_supported=ok["cell_identity"],
        model_match_supported=False, causal_supported=False,
        experimental_evidence_supported=ok.get("experimental_evidence", True), blocking_issues=issues,
        reason="\n".join(f"{name}: {judgment.reason}" for name, judgment in checks.items()),
        research_gaps=value.research_gaps)


def assemble_completion(value: ScientificCompletion, session):
    counts = session.summary_counts()
    return ResearchCompletion(state_version=session.version, stop_reason=value.stop_reason,
        summary=value.summary + "\n\nHost workflow counts: " + json.dumps(counts, sort_keys=True),
        limitations=value.limitations)
