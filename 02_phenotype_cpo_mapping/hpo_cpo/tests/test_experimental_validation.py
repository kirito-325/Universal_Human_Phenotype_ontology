"""Optional empirical support must not become a gate against plausible inference."""

import json

import pytest
from pydantic import ValidationError
from test_model_knowledge_submission import knowledge_claim as knowledge_claim
from test_runtime import ScriptedModel, response

from hpo_agent.agents import AgentEngine
from hpo_agent.schemas import (
    ClaimDraft,
    ExperimentalValidation,
    Extraction,
    Observation,
    ReviewIssue,
    Settings,
)
from hpo_agent.session import ResearchSession
from hpo_agent.validation import (
    anchor_claim,
    anchor_experiments,
    anchor_reasoning,
    release_gate,
)


def infer_review(review, **updates):
    return review.model_copy(update={"inference_supported": True, "scope_transfer_supported": True,
                                     "cell_inference_supported": True, **updates})


def with_experiment(claim, document, **updates):
    experiment = {"document_id": document.document_id, "quote": document.text,
                  "scope": "cell_abnormality", "study_type": "observation", "species": "human",
                  "experimental_context": "Synthetic study, not biological evidence.",
                  "finding": "Synthetic recorded cellular abnormality.", "effect": "supports", **updates}
    return ClaimDraft.model_validate({**claim.model_dump(), "experimental_evidence": [experiment]})


@pytest.mark.parametrize("verdict", ["supported", "partial", "insufficient_evidence"])
def test_defensible_unconfirmed_inference_is_retained(phenotype, knowledge_claim, mapping, review, verdict):
    steps = anchor_reasoning(phenotype, knowledge_claim, None, {}, [])
    scope, _ = release_gate(phenotype, knowledge_claim, None, mapping, infer_review(review, verdict=verdict), steps)
    assert scope == "conditional"


def test_source_misinterpretation_is_a_blocker_even_with_all_plausibility_flags_true(
    phenotype, knowledge_claim, mapping, review,
):
    checked = infer_review(review, verdict="partial", blocking_issues=[ReviewIssue(
        code="source_misinterpretation", explanation="The proposal attributes one entity's result to a different entity.",
    )])
    steps = anchor_reasoning(phenotype, knowledge_claim, None, {}, [])
    scope, reasons = release_gate(phenotype, knowledge_claim, None, mapping, checked, steps)
    assert scope == "rejected" and "source_misinterpretation" in reasons
    with pytest.raises(ValidationError):
        ReviewIssue(code="missing_experimental_proof", explanation="Untested at the exact site.")


def test_partial_direct_evidence_is_not_promoted_to_confirmed(phenotype, draft, document, mapping, review):
    evidence = anchor_claim(draft, document, 0, len(document.text))
    assert release_gate(phenotype, draft, evidence, mapping, review.model_copy(update={"verdict": "partial"}))[0] == "unresolved"


@pytest.mark.parametrize("scope,status", [("mechanism", "component_support"), ("whole_claim", "claim_support")])
def test_empirical_support_is_grounded_and_separate_from_inference(
    phenotype, knowledge_claim, document, mapping, review, scope, status,
):
    claim = with_experiment(knowledge_claim, document, scope=scope)
    windows = [{"document_id": document.document_id, "start": 0, "end": len(document.text)}]
    experiments = anchor_experiments(claim, {document.document_id: document}, windows)
    validation = ExperimentalValidation(evidence=experiments)
    assert validation.status == status
    span = experiments[0].evidence
    assert span.pmid == document.pmid and document.text[span.span_start:span.span_end] == span.quote
    steps = anchor_reasoning(phenotype, claim, None, {document.document_id: document}, windows)
    checked = infer_review(review, verdict="partial")
    released, reasons = release_gate(phenotype, claim, None, mapping, checked, steps)
    observation = Observation(
        claim_id="synthetic", phenotype_id=phenotype.phenotype_id, phenotype_label=phenotype.label,
        claim=claim, evidence_mode="inferred", evidence=None, reasoning_steps=steps,
        experimental_validation=validation, mapping=mapping, review=checked,
        release_scope=released, gate_reasons=reasons,
    )
    assert observation.evidence_mode == "inferred" and observation.knowledge_source == "mixed"
    assert observation.release_scope == "conditional"
    assert Observation.model_validate_json(observation.model_dump_json()) == observation
    row = observation.model_dump()
    row["experimental_validation"]["evidence"][0]["evidence"]["quote"] = "Invented quote."
    with pytest.raises(ValidationError, match="experimental evidence differs"):
        Observation.model_validate(row)
    with pytest.raises(ValidationError, match="status must match"):
        ExperimentalValidation(status="claim_support", evidence=[])


@pytest.mark.parametrize("mutation", ["unstaged", "unread", "fabricated", "retracted", "wrong_species"])
def test_experimental_evidence_cannot_be_fabricated_or_promote_a_different_species(
    knowledge_claim, document, mutation,
):
    claim = with_experiment(knowledge_claim, document)
    documents = {document.document_id: document}
    windows = [{"document_id": document.document_id, "start": 0, "end": len(document.text)}]
    if mutation == "unstaged":
        documents = {}
    elif mutation == "unread":
        windows = []
    elif mutation == "fabricated":
        claim = with_experiment(knowledge_claim, document, quote="Invented experimental result.")
    elif mutation == "retracted":
        documents[document.document_id] = document.model_copy(update={"retracted": True})
    else:
        claim = with_experiment(knowledge_claim, document, scope="whole_claim", species="mouse")
    with pytest.raises(ValueError, match="experimental_evidence"):
        anchor_experiments(claim, documents, windows)


def test_different_species_component_support_is_preserved(knowledge_claim, document):
    claim = with_experiment(knowledge_claim, document, scope="mechanism", species="mouse")
    findings = anchor_experiments(claim, {document.document_id: document},
                                 [{"document_id": document.document_id, "start": 0, "end": len(document.text)}])
    validation = ExperimentalValidation(evidence=findings)
    assert validation.status == "component_support" and validation.evidence[0].species == "mouse"


def test_negative_evidence_label_does_not_override_a_defensible_review(
    phenotype, knowledge_claim, document, mapping, review,
):
    claim = with_experiment(knowledge_claim, document, scope="whole_claim", effect="contradicts")
    steps = anchor_reasoning(phenotype, claim, None, {}, [])
    scope, reasons = release_gate(phenotype, claim, None, mapping, infer_review(review), steps)
    assert scope == "conditional"
    assert "whole_claim_experiment_contradicts_prediction" not in reasons
    assert claim.experimental_evidence[0].effect == "contradicts"


def test_applicable_necessary_premise_defect_still_blocks_release(
    phenotype, knowledge_claim, document, mapping, review,
):
    claim = with_experiment(knowledge_claim, document, scope="whole_claim", effect="contradicts")
    checked = infer_review(review, blocking_issues=[ReviewIssue(
        code="contradictory_evidence", explanation="An applicable study explicitly invalidates a necessary premise.",
    )])
    steps = anchor_reasoning(phenotype, claim, None, {}, [])
    scope, reasons = release_gate(phenotype, claim, None, mapping, checked, steps)
    assert scope == "rejected" and "contradictory_evidence" in reasons


async def test_session_exposes_empirical_context_to_review_and_preserves_evidence_on_output(
    tmp_path, phenotype, knowledge_claim, document, mapping, review, index,
):
    claim = with_experiment(knowledge_claim, document)
    checked = infer_review(review, verdict="partial")
    model = ScriptedModel(responses=[response("Review", checked.model_dump())])
    settings = Settings(workflow_mode="component_test", offline=True)
    engine = AgentEngine(index, settings, models={"review": model})
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                              output=tmp_path, attempt=tmp_path / "attempt")
    await session.stage(document)
    window = session.read(document.document_id)
    result = await session.collect_evidence({"supporting_document_ids": [document.document_id]},
        evidence_mode="inferred", extraction_override=Extraction(claims=[claim]),
        provided_context=[window])
    cid = result["claims"][0]["claim_id"]
    session.claims[cid]["mapping"] = mapping
    reviewed = await session.review({"claim_id": cid})
    assert reviewed["release_scope"] == "conditional"
    output = session.observations[cid]
    assert output.experimental_validation.evidence[0].evidence.pmid == document.pmid
    review_input = next(a["input"] for a in session.audit if a.get("role") == "review" and "input" in a)
    assert review_input["experimental_validation"]["status"] == "component_support"
    assert review_input["reasoning_source_contexts"][0]["text"] == document.text
    await session.checkpoint()
    persisted = json.loads((tmp_path / "attempt/session.json").read_text())
    assert persisted["observations"][0]["experimental_validation"]["evidence"][0]["evidence"]["pmid"] == document.pmid


async def test_live_review_contract_requires_explicit_scientific_blockers_and_empirical_check(index, review):
    missing = review.model_dump(exclude={"blocking_issues", "experimental_evidence_supported"})
    model = ScriptedModel(responses=[response("Review", missing), response("Review", review.model_dump())])
    engine = AgentEngine(index, Settings(), models={"review": model})
    audit = []
    await engine.review({}, audit)
    responses = [a for a in audit if a.get("event") == "agent_message" and a.get("message_type") == "ai"]
    assert len(responses) == 2
    feedback = next(a["content"] for a in audit if a.get("name") == "Review" and a.get("message_type") == "tool")
    assert "blocking_issues" in feedback and "experimental_evidence_supported" in feedback
