"""Repair the real evaluator submission path without restarting scientific judgment."""

import copy
import json

import pytest
from langchain_core.messages import HumanMessage
from pydantic import Field
from test_literature_workflow import context_index as context_index
from test_literature_workflow import discovery as discovery
from test_literature_workflow import prepared_session
from test_model_knowledge_submission import knowledge_claim as knowledge_claim
from test_runtime import DISPATCH, ScriptedModel, finish, response
from test_runtime_instructions import PromptRecordingModel

from hpo_agent import legacy_prompts as prompts
from hpo_agent.agents import AgentEngine
from hpo_agent.contracts import (
    DIRECTION_DESCRIPTION,
    FIELD_PLACEMENT_GUIDE,
    SCIENCE_POLICY,
    SOURCE_DOCUMENT_DESCRIPTION,
    apply_assessment_repair,
)
from hpo_agent.schemas import (
    ClaimDraft,
    ExperimentalEvidenceDraft,
    HypothesisAssessment,
    HypothesisDraft,
    QuoteReference,
    Settings,
)
from hpo_agent.validation import anchor_experiments, anchor_reasoning, main_quote_window


class SchemaRecordingModel(PromptRecordingModel):
    bound_schemas: list[dict] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        for item in tools:
            if isinstance(item, dict):
                definition = item.get("function", item)
                self.bound_schemas.append({"name": definition.get("name"), "schema": definition.get("parameters", definition)})
            else:
                schema = item.args_schema
                self.bound_schemas.append({"name": item.name, "schema": schema if isinstance(schema, dict) else schema.model_json_schema()})
        return super().bind_tools(tools, **kwargs)


async def failed_quote_session(tmp_path, phenotype, context_index, discovery, index, draft):
    abstract = discovery[0][0]
    full = abstract.model_copy(update={"document_id": "PMC123", "source_kind": "full_text",
                                       "text": "Synthetic full text differs from the paper's abstract."})
    bad = HypothesisAssessment(decision="retain", rationale="Preserve this scientific conclusion.",
        supporting_document_ids=[abstract.document_id],
        claim=draft.model_copy(update={"supporting_document_id": full.document_id})).model_dump()
    model = ScriptedModel(responses=[response("read_document", {"document_id": full.document_id}),
                                     *[response("SubmitAssessment", bad) for _ in range(3)]])
    engine = AgentEngine(index, Settings(offline=True), models={"verify_hypothesis": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine)
    await session.stage(full)
    hypotheses = session.workflow.register_hypotheses(discovery[2], [session.read(d.document_id) for d in discovery[0]])
    hid = hypotheses[0]["hypothesis_id"]
    await session.search_local("route0", hypothesis_id=hid, intent="support")
    await session.search_local("counter route0", hypothesis_id=hid, intent="challenge")
    adapter = next(spec["runnable"] for spec in session.subagents() if spec["name"] == "hypothesis-evaluator")

    async def dispatch(request):
        return (await adapter.ainvoke({"messages": [HumanMessage(json.dumps(request))]}))["structured_response"]

    failed = await dispatch({"hypothesis_id": hid})
    return session, model, hid, dispatch, failed


async def test_quote_source_feedback_and_repair_preserve_the_failed_assessment(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, model, hid, dispatch, failed = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    assert failed["status"] == "invalid_format" and failed["next_action"] == "repair_submission"
    issue, = failed["issues"]
    assert issue["code"] == "claim.source_mismatch"
    assert issue["path"] == ["claim", "supporting_document_id"]
    assert issue["actual"] == "PMC123" and issue["expected"] == ["synthetic:0"]
    assert session.workflow.hypotheses[hid]["status"] == "proposed" and not session.claims
    snapshot = json.loads((tmp_path / "attempt/session.json").read_text())
    assert snapshot["failed_assessments"][hid]["draft"] == failed["failed_submission"]
    assert session.state()["assessment_submission_feedback"][0]["failure_id"] == failed["failure_id"]

    count = len([event for event in session.audit if event.get("event") == "agent_started"])
    blocked = await dispatch({"hypothesis_id": hid})
    assert blocked["error"] == "assessment_submission_requires_repair_mode"
    assert len([event for event in session.audit if event.get("event") == "agent_started"]) == count
    malformed = await dispatch({"hypothesis_id": hid, "mode": []})
    assert malformed["error"] == "assessment_mode_must_be_evaluate_repair_or_reevaluate"
    stale = await dispatch({"hypothesis_id": hid, "mode": "repair", "failure_id": "stale"})
    assert "current_failure_id" in stale["error"]

    model.responses = [response("SubmitAssessmentRepair", {"updates": [
        {"path": ["claim", "supporting_document_id"], "value": "synthetic:0"}]})]
    model.i = 0
    repaired = await dispatch({"hypothesis_id": hid, "mode": "repair", "failure_id": failed["failure_id"]})
    assert repaired["status"] == "retained"
    actual = session.workflow.hypotheses[hid]["assessment"]
    original = failed["failed_submission"]
    assert actual["decision"] == original["decision"] and actual["rationale"] == original["rationale"]
    for field in ("cell_text", "process_text", "quality_text", "direction", "relation_type", "rationale"):
        assert actual["claim"][field] == original["claim"][field]
    assert actual["claim"]["supporting_document_id"] == "synthetic:0"
    assert not session.failed_assessments
    assert all(error["resolved"] for error in session.errors)
    recovery, = session.coverage["assessment_recoveries"]
    assert recovery["mode"] == "repair" and recovery["failure_id"] == failed["failure_id"]
    repair_input = next(event["input"] for event in session.audit
                        if event.get("role") == "verify_hypothesis" and event.get("input", {}).get("mode") == "repair")
    assert repair_input["previous_submission"] == original
    assert "PMC123" in repair_input["restored_read_document_ids"]


async def test_failed_repair_is_bounded_and_keeps_a_full_draft(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, model, hid, dispatch, failed = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    model.responses = [response("SubmitAssessmentRepair", {"updates": [
        {"path": ["claim", "supporting_document_id"], "value": "PMC123"}]}) for _ in range(3)]
    model.i = 0
    request = {"hypothesis_id": hid, "mode": "repair", "failure_id": failed["failure_id"]}
    again = await dispatch(request)
    assert again["status"] == "invalid_format" and again["next_action"] == "stop_or_explicit_reevaluate"
    assert "decision" in again["failed_submission"] and "updates" not in again["failed_submission"]
    blocked = await dispatch({**request, "failure_id": again["failure_id"]})
    assert blocked["error"] == "assessment_repair_budget_exhausted"
    assert session.workflow.hypotheses[hid]["status"] == "proposed" and not session.claims


async def test_root_eval_uses_the_existing_task_adapter_for_repair(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, model, hid, _, _ = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    model.responses = [response("SubmitAssessmentRepair", {"updates": [
        {"path": ["claim", "supporting_document_id"], "value": "synthetic:0"}]})]
    model.i = 0
    root = ScriptedModel(responses=[response("eval", {"code": DISPATCH + """
var st = await tools.getResearchState({});
var failure = st.assessment_submission_feedback[0];
var repaired = await dispatch('hypothesis-evaluator', {
  hypothesis_id:failure.hypothesis_id,mode:'repair',failure_id:failure.failure_id
});
({status:repaired.status});
"""}), finish(0)])
    session.engine.models["research"] = root
    await session.engine.run(session)
    assert session.workflow.hypotheses[hid]["status"] == "retained"
    messages = [event for event in session.audit if event.get("event") == "agent_message"
                and event.get("role") == "research" and event.get("message_type") == "tool"]
    assert any('retained' in event["content"] for event in messages)


async def test_transport_failure_does_not_consume_the_repair_opportunity(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, _, hid, dispatch, failed = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    original = session.engine.verify_hypothesis

    async def timeout(*args, **kwargs):
        raise TimeoutError("Synthetic transport timeout.")

    session.engine.verify_hypothesis = timeout
    await dispatch({"hypothesis_id": hid, "mode": "repair", "failure_id": failed["failure_id"]})
    assert not session.failed_assessments[hid]["repair_attempted"]
    session.engine.verify_hypothesis = original


async def test_scientific_reconsideration_requires_explicit_reason_and_preserves_history(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, model, hid, dispatch, failed = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    denied = await dispatch({"hypothesis_id": hid, "mode": "reevaluate"})
    assert denied["error"] == "explicit_reevaluation_requires_reassessment_reason"
    model.responses = [response("SubmitAssessment", {"decision": "reject",
        "rationale": "Synthetic new interpretation contradicts the preserved conclusion.",
        "revision_note": "Explicitly reconsidered the previous revised candidate.", "claim": None})]
    model.i = 0
    reason = "Reconsider the specific biological interpretation, separately from quotation repair."
    result = await dispatch({"hypothesis_id": hid, "mode": "reevaluate", "reassessment_reason": reason})
    assert result["status"] == "rejected"
    recovery, = session.coverage["assessment_recoveries"]
    assert recovery["mode"] == "reevaluate" and recovery["reassessment_reason"] == reason
    assert recovery["previous_submission"] == failed["failed_submission"]
    assert recovery["assessment"]["decision"] == "reject"


@pytest.mark.parametrize("path,value", [
    (["decision"], "reject"), (["rationale"], "A different scientific judgment."),
    (["claim"], None), (["claim", "cell_text"], "A different cell"),
    (["claim", "process_text"], "A different process"), (["claim", "direction"], "decreased"),
    (["claim", "species"], "mouse"), (["revision_note"], "Replace the previous revision."),
])
def test_repair_cannot_change_scientific_fields(path, value, draft):
    previous = HypothesisAssessment(decision="retain", rationale="Keep this judgment.", claim=draft).model_dump()
    with pytest.raises(ValueError, match="scientific decision"):
        apply_assessment_repair(previous, [{"path": path, "value": value}])
    assert previous["decision"] == "retain" and previous["claim"]["direction"] == "increased"


def test_repair_removes_invalid_extra_fields_without_mutating_the_baseline(draft):
    previous = HypothesisAssessment(decision="retain", rationale="Keep judgment.", claim=draft).model_dump()
    previous["claim"]["reasoning_steps"] = [{"basis_notes": "Invalid extra field"}]
    original = copy.deepcopy(previous)
    fixed = apply_assessment_repair(previous, [{"path": ["claim", "reasoning_steps", 0, "basis_notes"], "operation": "remove"}])
    assert fixed["claim"]["reasoning_steps"] == [{}] and previous == original


@pytest.mark.parametrize("case,code,field", [
    ("unstaged", "claim.source_unstaged", "supporting_document_id"),
    ("unread_source", "claim.source_unread", "supporting_document_id"),
    ("unread_quote", "claim.quote_unread", "supporting_quote"),
    ("paraphrase", "claim.quote_verbatim", "supporting_quote"),
    ("ambiguous", "claim.source_ambiguous", "supporting_document_id"),
])
def test_source_diagnostics_distinguish_identity_text_and_read_coverage(document, draft, case, code, field):
    other = document.model_copy(update={"document_id": "other"})
    docs = {document.document_id: document, other.document_id: other}
    windows = [{"document_id": document.document_id, "start": 0, "end": len(document.text), "text": document.text}]
    value = draft.model_copy(update={"supporting_document_id": document.document_id})
    if case == "unstaged":
        value.supporting_document_id = "unknown"
    elif case == "unread_source":
        value.supporting_document_id = other.document_id
    elif case == "unread_quote":
        windows[0].update(text="Synthetic header.", end=17)
    elif case == "paraphrase":
        value.supporting_quote = "An invented paraphrase of a real result."
    else:
        value.supporting_document_id = ""
        windows.append({**windows[0], "document_id": other.document_id})
    with pytest.raises(ValueError) as raised:
        main_quote_window(value, docs, windows)
    assert raised.value.type == code and raised.value.context["field"] == [field]


def test_main_quote_feedback_locates_related_citation_mistakes(phenotype, document, knowledge_claim):
    full = document.model_copy(update={"document_id": "PMC123", "text": "Different full-text wording."})
    quote = document.text
    step = knowledge_claim.reasoning_steps[0].model_copy(update={"basis": "inference",
        "assumptions": ["Synthetic applicability assumption."],
        "citations": [QuoteReference(document_id=full.document_id, quote=quote)]})
    value = knowledge_claim.model_copy(update={"supporting_quote": quote,
        "supporting_document_id": full.document_id, "reasoning_steps": [step]})
    docs = {document.document_id: document, full.document_id: full}
    windows = [{"document_id": d.document_id, "start": 0, "end": len(d.text), "text": d.text} for d in docs.values()]
    with pytest.raises(ValueError) as raised:
        main_quote_window(value, docs, windows)
    assert raised.value.context["related_paths"] == [["claim", "reasoning_steps", 0, "citations", 0, "document_id"]]
    with pytest.raises(ValueError) as citation_error:
        anchor_reasoning(phenotype, value, None, docs, windows)
    assert citation_error.value.type == "citation.source_mismatch"
    assert citation_error.value.context["field"] == ["reasoning_steps", 0, "citations", 0, "document_id"]
    assert citation_error.value.context["expected"] == [document.document_id]


def test_experimental_quote_uses_the_same_source_identity_rule(document, draft):
    full = document.model_copy(update={"document_id": "PMC123", "text": "Different full-text wording."})
    entry = ExperimentalEvidenceDraft(document_id=full.document_id, quote=document.text, scope="mechanism",
        study_type="observation", species="human", experimental_context="Synthetic test.", finding="Synthetic finding.")
    value = draft.model_copy(update={"experimental_evidence": [entry]})
    docs = {document.document_id: document, full.document_id: full}
    windows = [{"document_id": d.document_id, "start": 0, "end": len(d.text), "text": d.text} for d in docs.values()]
    with pytest.raises(ValueError) as raised:
        anchor_experiments(value, docs, windows)
    assert raised.value.type == "citation.source_mismatch"
    assert raised.value.context["field"] == ["experimental_evidence", 0, "document_id"]


async def test_repair_tool_rejects_decision_patch_and_accepts_only_targeted_updates(index, phenotype, draft):
    previous = HypothesisAssessment(decision="retain", rationale="Keep this judgment.", claim=draft).model_dump()
    model = PromptRecordingModel(responses=[
        response("SubmitAssessmentRepair", {"updates": [{"path": ["decision"], "value": "reject"}]}),
        response("SubmitAssessmentRepair", {"updates": [{"path": ["claim", "supporting_document_id"], "value": "doc"}]}),
    ])
    engine = AgentEngine(index, Settings(), models={"verify_hypothesis": model})
    audit = []
    result = await engine.verify_hypothesis({"mode": "repair", "phenotype": phenotype.model_dump(),
                                            "previous_submission": previous}, audit)
    assert result.decision == "retain" and result.rationale == previous["rationale"]
    failure, = [event for event in audit if event.get("event") == "submission_validation_failed"]
    assert failure["issues"][0]["path"] == ["updates", 0, "path"]
    assert all(SCIENCE_POLICY in message and FIELD_PLACEMENT_GUIDE in message for message in model.received_system_messages)
    assert "SubmitAssessment" not in model.observed_tools


def test_upstream_schema_and_both_judges_define_the_same_direction_and_quote_semantics():
    schema = ClaimDraft.model_json_schema()
    assert schema["properties"]["direction"]["description"] == DIRECTION_DESCRIPTION
    assert HypothesisDraft.model_json_schema()["properties"]["direction"]["description"] == DIRECTION_DESCRIPTION
    assert SOURCE_DOCUMENT_DESCRIPTION in schema["properties"]["supporting_document_id"]["description"]
    assert schema["$defs"]["QuoteReference"]["properties"]["document_id"]["description"] == SOURCE_DOCUMENT_DESCRIPTION
    for prompt in (prompts.HYPOTHESIZE, prompts.VERIFY_HYPOTHESIS, prompts.REPAIR_ASSESSMENT, prompts.REVIEW):
        assert SCIENCE_POLICY in prompt
    assert "SAME read_document window" in prompts.VERIFY_HYPOTHESIS
    assert "record entries" in prompts.VERIFY_HYPOTHESIS


async def test_field_placement_reaches_the_actual_model_and_bound_submission_tool(index, phenotype, draft):
    submission = HypothesisAssessment(decision="retain", rationale="Synthetic defensible proposal.", claim=draft)
    model = SchemaRecordingModel(responses=[response("SubmitAssessment", submission.model_dump())])
    engine = AgentEngine(index, Settings(), models={"verify_hypothesis": model})
    await engine.verify_hypothesis({"phenotype": phenotype.model_dump()}, [])
    assert FIELD_PLACEMENT_GUIDE in model.received_system_messages[0]
    bound = next(item["schema"] for item in model.bound_schemas if item["name"] == "SubmitAssessment")
    assert bound["$defs"]["ClaimDraft"]["properties"]["direction"]["description"] == DIRECTION_DESCRIPTION
    assert SOURCE_DOCUMENT_DESCRIPTION in bound["$defs"]["ClaimDraft"]["properties"]["supporting_document_id"]["description"]
    assert "actually read" in bound["properties"]["supporting_document_ids"]["description"]


async def test_source_list_feedback_names_the_actual_assessment_field(
    tmp_path, phenotype, context_index, discovery, index, draft,
):
    session, model, hid, dispatch, failed = await failed_quote_session(
        tmp_path, phenotype, context_index, discovery, index, draft)
    previous = failed["failed_submission"]
    bad = {**previous, "contradicting_document_ids": ["not_read"], "revision_note": "Revisited the interpretation."}
    bad["claim"] = {**bad["claim"], "supporting_document_id": "synthetic:0"}
    good = {**bad, "contradicting_document_ids": []}
    model.responses = [response("SubmitAssessment", bad), response("SubmitAssessment", good)]
    model.i = 0
    result = await dispatch({"hypothesis_id": hid, "mode": "reevaluate", "reassessment_reason": "Check the source interpretation."})
    assert result["status"] == "retained"
    event = next(event for event in session.audit if event.get("event") == "submission_validation_failed"
                 and event["issues"][0]["code"] == "assessment.sources")
    assert event["issues"][0]["path"] == ["contradicting_document_ids"]
    assert event["issues"][0]["actual"] == ["not_read"]
