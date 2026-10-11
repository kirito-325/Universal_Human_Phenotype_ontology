"""The advertised output contract and the actual submission validators must agree."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
from test_model_knowledge_submission import knowledge_claim as knowledge_claim
from test_runtime import ScriptedModel, response
from test_runtime_instructions import PromptRecordingModel

from hpo_agent import prompts
from hpo_agent.agents import AgentEngine
from hpo_agent.contracts import (
    RULES,
    SCIENCE_POLICY,
    SubmissionRepairError,
    output_contract,
    submission_error_feedback,
)
from hpo_agent.progress import EvidenceProgress
from hpo_agent.schemas import ClaimDraft, Observation, ReasoningStep, Review, Settings
from hpo_agent.session import ResearchSession
from hpo_agent.validation import anchor_reasoning


@pytest.mark.parametrize("basis,citations,assumptions,code", [
    ("phenotype_definition", [], [], None),
    ("model_knowledge", [], [], None),
    ("model_knowledge", [], ["Applicability assumption."], None),
    ("literature", [{"document_id": "doc", "quote": "Exact quote."}], [], None),
    ("inference", [{"document_id": "doc", "quote": "Exact quote."}], ["Applicability assumption."], None),
    ("phenotype_definition", [], ["Assumption."], "step.definition"),
    ("phenotype_definition", [{"document_id": "doc", "quote": "Quote."}], [], "step.definition"),
    ("model_knowledge", [{"document_id": "doc", "quote": "Quote."}], [], "step.model_knowledge"),
    ("literature", [], [], "step.citations"),
    ("inference", [{"document_id": "doc", "quote": "Quote."}], [], "step.inference_assumptions"),
    ("literature", [{"document_id": "doc", "quote": "Quote."}], ["Assumption."], "step.assumptions"),
])
def test_step_rules_are_identical_in_instructions_and_validation(basis, citations, assumptions, code):
    value = {"source": "A", "target": "B", "relation": "supports", "explanation": "Synthetic explanation.",
             "basis": basis, "citations": citations, "assumptions": assumptions}
    if code is None:
        ReasoningStep.model_validate(value)
    else:
        assert RULES[code] in output_contract("verify_hypothesis")
        with pytest.raises(ValidationError) as raised:
            ReasoningStep.model_validate(value)
        assert raised.value.errors()[0]["type"] == code


@pytest.mark.parametrize("updates,code", [
    ({"supporting_document_id": "doc"}, "claim.quote_source"),
    ({"evidence_mode": "direct"}, "claim.unsourced_modes"),
    ({"link_evidence_mode": "direct", "bridge_scope": "direct_phenotype"}, "claim.unsourced_link"),
    ({"assertion": "own_result"}, "claim.unsourced_provenance"),
    ({"taxon_id": "NCBITaxon:10090"}, "claim.taxon"),
    ({"relation_type": "causally_upstream_of"}, "claim.inference_relation"),
])
def test_claim_rules_are_identical_in_instructions_and_validation(knowledge_claim, updates, code):
    assert RULES[code] in output_contract("verify_hypothesis")
    with pytest.raises(ValidationError) as raised:
        ClaimDraft.model_validate({**knowledge_claim.model_dump(), **updates})
    assert raised.value.errors()[0]["type"] == code


def test_schema_types_are_not_silently_coerced(knowledge_claim, review):
    with pytest.raises(ValidationError):
        Review.model_validate({**review.model_dump(), "cell_supported": "true"})
    with pytest.raises(ValidationError):
        Settings.model_validate({"max_searches": "30"})
    with pytest.raises(ValidationError):
        ClaimDraft.model_validate({**knowledge_claim.model_dump(), "reasoning_steps": json.dumps(
            knowledge_claim.model_dump()["reasoning_steps"])})


def test_programming_error_is_not_presented_as_a_model_format_error():
    bug = TypeError("Synthetic validator programming error.")
    wrapper = ValueError("Synthetic parser wrapper.")
    wrapper.__cause__ = bug
    with pytest.raises(TypeError, match="programming error"):
        submission_error_feedback(wrapper)


@pytest.mark.parametrize("style", ["token", "id", "label_id", "id_label", "id_parenthesized", "case_whitespace"])
def test_documented_start_forms_are_canonical_in_stored_output(
    phenotype, knowledge_claim, mapping, review, style,
):
    starts = {"token": "INPUT_PHENOTYPE", "id": phenotype.phenotype_id,
              "label_id": f"{phenotype.label} ({phenotype.phenotype_id})",
              "id_label": f"{phenotype.phenotype_id}: {phenotype.label}",
              "id_parenthesized": f"{phenotype.phenotype_id} ({phenotype.label})",
              "case_whitespace": f"  {phenotype.label.upper()}  "}
    original = knowledge_claim.model_copy(update={"reasoning_steps": [
        knowledge_claim.reasoning_steps[0].model_copy(update={"source": starts[style]})]})
    grounded = anchor_reasoning(phenotype, original, None, {}, [])
    observation = Observation(claim_id="synthetic", phenotype_id=phenotype.phenotype_id,
        phenotype_label=phenotype.label, claim=original, evidence_mode="inferred", evidence=None,
        reasoning_steps=grounded, mapping=mapping, review=review, release_scope="conditional", gate_reasons=[])
    assert original.reasoning_steps[0].source == starts[style]
    assert observation.claim.reasoning_steps[0].source == phenotype.label
    assert observation.reasoning_steps[0].source == phenotype.label
    assert Observation.model_validate_json(observation.model_dump_json()) == observation


@pytest.mark.parametrize("source", ["Different phenotype", "Seizure (HP:9999999)", "Different phenotype (HP:0001250)"])
def test_unrelated_start_is_not_silently_bound(phenotype, knowledge_claim, source):
    original = knowledge_claim.model_copy(update={"reasoning_steps": [
        knowledge_claim.reasoning_steps[0].model_copy(update={"source": source})]})
    with pytest.raises(ValueError, match="start_at_phenotype_label"):
        anchor_reasoning(phenotype, original, None, {}, [])


def assessment(claim):
    return {"decision": "retain", "rationale": "Synthetic reasonable inference, not experimentally confirmed.",
            "supporting_document_ids": [], "contradicting_document_ids": [], "claim": claim.model_dump()}


async def test_actual_prompt_and_structured_error_report_the_same_start_rule(index, phenotype, knowledge_claim):
    bad = knowledge_claim.model_copy(update={"reasoning_steps": [
        knowledge_claim.reasoning_steps[0].model_copy(update={"source": "Wrong phenotype"})]})
    good = knowledge_claim.model_copy(update={"reasoning_steps": [
        knowledge_claim.reasoning_steps[0].model_copy(update={"source": "INPUT_PHENOTYPE"})]})
    model = PromptRecordingModel(responses=[response("SubmitAssessment", assessment(bad)),
                                            response("SubmitAssessment", assessment(good))])
    engine = AgentEngine(index, Settings(), models={"verify_hypothesis": model})
    audit = []
    value = await engine.verify_hypothesis({"phenotype": phenotype.model_dump()}, audit)
    assert value.claim.reasoning_steps[0].source == phenotype.label
    actual = model.received_system_messages[0]
    assert SCIENCE_POLICY in actual and RULES["claim.start"] in actual
    failure, = [e for e in audit if e.get("event") == "submission_validation_failed"]
    issue, = failure["issues"]
    assert issue["code"] == "claim.start"
    assert issue["path"] == ["claim", "reasoning_steps", 0, "source"]
    assert issue["expected"] == phenotype.label and issue["actual"] == "Wrong phenotype"


async def test_unresolved_field_error_is_bounded_despite_changes_to_unrelated_text(index, phenotype, knowledge_claim):
    bad = knowledge_claim.model_dump()
    bad["reasoning_steps"][0]["source"] = "Wrong phenotype"
    responses = [response("SubmitAssessment", assessment(ClaimDraft.model_validate(
        {**bad, "experimental_context": f"Unrelated wording change {i}"}))) for i in range(3)]
    engine = AgentEngine(index, Settings(), models={"verify_hypothesis": ScriptedModel(responses=responses)})
    audit = []
    with pytest.raises(SubmissionRepairError) as raised:
        await engine.verify_hypothesis({"phenotype": phenotype.model_dump()}, audit)
    assert raised.value.issues[0]["code"] == "claim.start"
    assert len([e for e in audit if e.get("event") == "submission_validation_failed"]) == 3


def test_resolved_validation_errors_do_not_accumulate_across_different_repairs():
    guard = EvidenceProgress({}, Settings(), submissions_only=True)
    for number, code in enumerate(("claim.chain", "claim.chain", "claim.endpoint", "claim.chain")):
        call_id = str(number)
        guard.observe(AIMessage("", tool_calls=[{"name": "SubmitAssessment", "id": call_id, "args": {}}]))
        guard.observe(ToolMessage(json.dumps({"status": "invalid_format", "issues": [
            {"code": code, "path": ["claim", "reasoning_steps"], "message": "Repair this field."}]}), tool_call_id=call_id))


def test_different_host_errors_are_not_counted_as_one_unresolved_field():
    guard = EvidenceProgress({}, Settings(), submissions_only=True)
    for number, message in enumerate(("Source not read", "Source not read", "Revision note required")):
        call_id = str(number)
        guard.observe(AIMessage("", tool_calls=[{"name": "SubmitAssessment", "id": call_id, "args": {}}]))
        guard.observe(ToolMessage(json.dumps({"status": "invalid_format", "issues": [
            {"code": "value_error", "path": [], "message": message}]}), tool_call_id=call_id))


async def test_format_failures_are_visible_to_root_without_becoming_scientific_rejections(tmp_path, phenotype, knowledge_claim):
    issues = [{"code": "claim.start", "path": ["claim", "reasoning_steps", 0, "source"], "message": "Wrong start."}]

    async def reason(payload, audit, tools=()):
        raise SubmissionRepairError("Extraction", issues, {"claims": [knowledge_claim.model_dump()]})

    session = ResearchSession(phenotype=phenotype, engine=SimpleNamespace(reason=reason),
        settings=Settings(workflow_mode="component_test"), output=tmp_path, attempt=tmp_path / "attempt")
    adapter = next(s["runnable"] for s in session.subagents() if s["name"] == "evidence-reasoner")
    result = (await adapter.ainvoke({"messages": [HumanMessage("{}")]}))["structured_response"]
    assert result["status"] == "invalid_format" and result["retryable"] is False
    assert result["failed_submission"]["claims"] and result["issues"] == issues
    assert not session.claims and not session.observations


def test_scientific_policy_is_shared_by_all_production_roles():
    for name in ("ROOT", "HYPOTHESIZE", "VERIFY_HYPOTHESIS", "NORMALIZE", "REVIEW"):
        assert SCIENCE_POLICY in getattr(prompts, name)
