"""Deterministic preparation and bounded scientific completion at real tool boundaries."""

import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from test_compact_submissions import assessment_payload, window
from test_literature_workflow import context_index as context_index
from test_literature_workflow import discovery as discovery
from test_literature_workflow import prepared_session
from test_runtime import ScriptedModel, response

from hpo_agent.agents import AgentEngine
from hpo_agent.assembly import assemble_assessment, model_window, resolve_refs
from hpo_agent.contracts import submission_error_feedback
from hpo_agent.normalization import complete_scientific_fields, normalize_assessment
from hpo_agent.references import EvidenceWindows
from hpo_agent.schemas import ScientificAssessment, Settings


def test_search_budget_reserves_support_without_requiring_challenge():
    settings = Settings()
    minimum = settings.min_discovery_directions + settings.min_hypotheses
    assert Settings(max_searches=minimum).max_searches == minimum
    with pytest.raises(ValidationError, match="supporting searches"):
        Settings(max_searches=minimum - 1)


@pytest.mark.parametrize("duplicate", [False, True])
def test_lossless_misplaced_fields_are_moved_or_deduplicated(duplicate):
    raw = assessment_payload()
    original = copy.deepcopy(raw)
    for key in ("assumptions", "experimental_evidence"):
        raw[key] = copy.deepcopy(raw["claim"][key])
        if not duplicate:
            del raw["claim"][key]
    raw["claim"]["cell_text"] = raw["claim"]["cell"]["cell_text"]
    normalized, changes = normalize_assessment(raw)
    assert normalized == original
    assert len(changes) == 3
    assert "assumptions" in raw  # Caller input is not mutated.
    assert ScientificAssessment.model_validate(raw) == ScientificAssessment.model_validate(original)


@pytest.mark.parametrize("field", ["assumptions", "experimental_evidence", "cell_text"])
def test_conflicting_duplicate_values_never_choose_a_winner(field):
    raw = assessment_payload()
    if field == "cell_text":
        raw["claim"][field] = "different cell"
    else:
        raw[field] = ["different value"]
    with pytest.raises(ValidationError) as raised:
        ScientificAssessment.model_validate(raw)
    feedback = json.loads(submission_error_feedback(SimpleNamespace(tool_name="SubmitAssessment", source=raised.value), compact=True))
    assert feedback["status"] == "field_conflict"
    assert feedback["issues"][0]["path"] == (["claim", "cell", "cell_text"] if field == "cell_text" else ["claim", field])


def test_flat_cell_name_never_fabricates_the_missing_cell_explanation():
    raw = assessment_payload()
    raw["claim"]["cell_text"] = raw["claim"].pop("cell")["cell_text"]
    normalized, _ = normalize_assessment(raw)
    assert normalized["claim"]["cell"] == {"cell_text": "neurons"}
    with pytest.raises(ValidationError) as raised:
        ScientificAssessment.model_validate(normalized)
    feedback = json.loads(submission_error_feedback(SimpleNamespace(tool_name="SubmitAssessment", source=raised.value), compact=True))
    assert feedback["status"] == "needs_reasoning"
    assert feedback["issues"][0]["path"] == ["claim", "cell", "explanation"]


def test_compact_extra_field_error_keeps_its_actual_location():
    raw = assessment_payload()
    raw["claim"] = None
    raw["decision"] = "inconclusive"
    raw["experimental_evidence"] = []
    with pytest.raises(ValidationError) as raised:
        ScientificAssessment.model_validate(raw)
    feedback = json.loads(submission_error_feedback(SimpleNamespace(tool_name="SubmitAssessment", source=raised.value), compact=True))
    assert feedback["issues"][0]["path"] == ["experimental_evidence"]
    assert feedback["status"] == "invalid_format"


def test_host_records_actual_scientific_changes_using_the_submitted_rationale(phenotype):
    value = ScientificAssessment.model_validate(assessment_payload())
    original = {"cell_text": "old cells", "process_text": "apoptosis", "direction": "decreased"}
    result = assemble_assessment(value, phenotype, "hid", {}, [], original=original)
    assert '"before": "old cells"' in result.revision_note
    assert '"after": "increased"' in result.revision_note
    assert value.rationale in result.revision_note
    assert not value.revision_note  # Host metadata never mutates model science.


def test_scientific_supplement_can_only_change_named_missing_fields():
    raw = assessment_payload()
    del raw["claim"]["cell"]["explanation"]
    supplement = {"claim": {"cell": {"explanation": "A justified scientific supplement."}}}
    paths = [["claim", "cell", "explanation"]]
    merged = complete_scientific_fields(raw, supplement, paths)
    assert merged["claim"]["cell"]["explanation"] == supplement["claim"]["cell"]["explanation"]
    assert "explanation" not in raw["claim"]["cell"]
    supplement["claim"]["cell"]["cell_text"] = "different cell"
    with pytest.raises(ValueError, match="conflict"):
        complete_scientific_fields(raw, supplement, paths)


def test_aliases_restore_the_same_source_and_do_not_authorize_unread_passages(document):
    first = EvidenceWindows()
    supplied = window(document)
    first.append(supplied)
    shown = model_window(supplied, first.catalog)
    assert shown["passages"][0]["evidence_ref"] == "E1"
    alias = shown["passages"][-1]["evidence_ref"]
    saved = first.catalog.snapshot()
    restored = EvidenceWindows(saved)
    restored.append(supplied)
    other = document.model_copy(update={"document_id": "different"})
    new_window = window(other)
    restored.append(new_window)
    assert model_window(new_window, restored.catalog)["passages"][0]["evidence_ref"] != "E1"
    span, = resolve_refs([alias], {document.document_id: document, other.document_id: other}, restored, field=("refs",))
    assert span.document_id == document.document_id
    assert saved["aliases"] == {key: restored.catalog.aliases[key] for key in saved["aliases"]}
    assert restored.catalog.catalog_id == first.catalog.catalog_id
    unread = EvidenceWindows(saved)
    with pytest.raises(ValueError, match="Unknown or unread"):
        resolve_refs([alias], {document.document_id: document}, unread, field=("refs",))
    with pytest.raises(ValueError, match="Unknown or unread"):
        resolve_refs(["E999999"], {document.document_id: document, other.document_id: other}, restored, field=("refs",))
    independent = EvidenceWindows()
    assert independent.catalog.catalog_id != first.catalog.catalog_id
    assert independent.catalog.resolve(alias) == alias.lower()


async def evaluation_session(tmp_path, index, phenotype, context_index, discovery, responses, *, challenge=True):
    model = ScriptedModel(responses=responses)
    engine = AgentEngine(index, Settings(offline=True), models={"verify_hypothesis": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine, compact=True)
    records = session.workflow.register_hypotheses(discovery[2], [session.read(doc.document_id) for doc in discovery[0]])
    hid = records[0]["hypothesis_id"]
    await session.search_local("route0", hypothesis_id=hid, intent="support")
    if challenge:
        await session.search_local("route1", hypothesis_id=hid, intent="challenge")
    adapter = next(spec["runnable"] for spec in session.subagents() if spec["name"] == "hypothesis-evaluator")

    async def dispatch(**request):
        return (await adapter.ainvoke({"messages": [HumanMessage(json.dumps({"hypothesis_id": hid, **request}))]}))["structured_response"]

    return session, hid, model, dispatch


async def test_validation_bodies_are_provided_once_per_intent_before_retention(tmp_path, index, phenotype, context_index, discovery):
    session, hid, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", assessment_payload("E1"))])
    result = await dispatch()
    assert result["status"] == "retained"
    event = next(event for event in session.audit if event.get("role") == "verify_hypothesis" and "input" in event)
    payload = event["input"]
    assert {intent for source in payload["validation_sources"] for intent in source["intents"]} == {"support", "challenge"}
    assert len(payload["validation_sources"]) == 2
    assert all(source["passages"] for source in payload["seed_contexts"])
    assert set(session.workflow.hypotheses[hid]["assessment"]["read_document_ids"]) == {"synthetic:0", "synthetic:1"}
    assert not any(event.get("event") == "submission_validation_failed" for event in session.audit)


async def test_missing_validation_source_stops_before_any_model_call(tmp_path, index, phenotype, context_index, discovery):
    session, hid, model, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", assessment_payload())])
    query = session.workflow.validation_searches(hid)[0]
    query.update(pmids=["999999999"], document_ids=[], count=1)
    result = await dispatch()
    assert result["status"] == "needs_sources"
    assert result["missing"][0]["pmids"] == ["999999999"]
    assert model.i == 0 and not any(event.get("event") == "agent_started" for event in session.audit)
    assert session.workflow.hypotheses[hid]["status"] == "proposed"


@pytest.mark.parametrize("optional_challenge", [False, True])
@pytest.mark.parametrize("support_hits", [False, True])
async def test_support_alone_unlocks_evaluation_even_with_unretrieved_optional_challenge(
    tmp_path, index, phenotype, context_index, discovery, optional_challenge, support_hits,
):
    session, hid, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", assessment_payload("E1"))], challenge=False)
    if not support_hits:
        session.workflow.validation_searches(hid)[0].update(count=0, pmids=[], document_ids=[])
    if optional_challenge:
        session.search_log.append({"query_id": "optional_challenge", "hypothesis_id": hid,
            "search_number": session.searches + 1, "intent": "challenge", "retmax": 10,
            "count": 1, "pmids": ["999999999"], "document_ids": []})
    result = await dispatch()
    assert result["status"] == "retained"
    event = next(event for event in session.audit if event.get("role") == "verify_hypothesis" and "input" in event)
    intents = {intent for source in event["input"]["validation_sources"] for intent in source["intents"]}
    assert intents == ({"support"} if support_hits else set())


async def test_real_evaluator_completes_only_the_missing_reason_and_restores_aliases(tmp_path, index, phenotype, context_index, discovery):
    raw = assessment_payload("E1")
    del raw["claim"]["cell"]["explanation"]
    supplement = {"claim": {"cell": {"explanation": "Scientifically justified targeted completion.", "evidence_refs": ["E1"]}}}
    session, hid, model, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", raw), response("SubmitAssessment", supplement)])
    failed = await dispatch()
    assert failed["status"] == "needs_reasoning" and failed["next_action"] == "complete_reasoning"
    feedback, = session.state()["assessment_submission_feedback"]
    assert feedback["failure_id"] == failed["failure_id"]
    catalog = copy.deepcopy(session.failed_assessments[hid]["evidence_catalog"])
    completed = await dispatch(mode="complete_reasoning", failure_id=failed["failure_id"])
    assert completed["status"] == "retained"
    recovery, = session.coverage["assessment_recoveries"]
    assert recovery["mode"] == "complete_reasoning"
    assert recovery["assessment"]["claim"]["cell_text"] == raw["claim"]["cell"]["cell_text"]
    record = session.claims[completed["claim_id"]]
    assert record["reasoning_steps"][0].explanation == supplement["claim"]["cell"]["explanation"]
    assert record["reasoning_steps"][0].evidence[0].document_id == "synthetic:0"
    assert catalog["aliases"]["E1"].startswith("ev_")
    assert hid not in session.failed_assessments
    assert model.observed_tools.count("SubmitAssessment") == 2


async def test_completion_rejects_changed_core_and_cannot_repeat(tmp_path, index, phenotype, context_index, discovery):
    raw = assessment_payload()
    del raw["claim"]["cell"]["explanation"]
    supplement = {"claim": {"cell": {"cell_text": "changed cell", "explanation": "New rationale."}}}
    session, hid, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", raw), response("SubmitAssessment", supplement)])
    failed = await dispatch()
    conflict = await dispatch(mode="complete_reasoning", failure_id=failed["failure_id"])
    assert conflict["status"] == "field_conflict"
    assert conflict["failed_submission"]["claim"]["cell"]["cell_text"] == "neurons"
    assert conflict["next_action"] == "stop_or_explicit_reevaluate"
    blocked = await dispatch(mode="complete_reasoning", failure_id=conflict["failure_id"])
    assert blocked["status"] == "needs_reasoning" and not session.claims


async def test_empty_transport_response_preserves_the_scientific_completion_budget(tmp_path, index, phenotype, context_index, discovery, monkeypatch):
    raw = assessment_payload()
    del raw["claim"]["cell"]["explanation"]
    supplement = {"claim": {"cell": {"explanation": "Justified completion after transport recovery."}}}
    session, hid, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", raw), response("SubmitAssessment", supplement)])
    failed = await dispatch()
    operation = session.engine.verify_hypothesis

    async def empty_response(*args, **kwargs):
        raise ValueError("verify_hypothesis: missing structured_response")

    monkeypatch.setattr(session.engine, "verify_hypothesis", empty_response)
    transport = await dispatch(mode="complete_reasoning", failure_id=failed["failure_id"])
    assert transport["status"] == "transport_error"
    assert not session.failed_assessments[hid]["completion_attempted"]
    monkeypatch.setattr(session.engine, "verify_hypothesis", operation)
    completed = await dispatch(mode="complete_reasoning", failure_id=failed["failure_id"])
    assert completed["status"] == "retained"


async def test_normalization_audit_records_only_lossless_changes(tmp_path, index, phenotype, context_index, discovery):
    raw = assessment_payload()
    raw["assumptions"] = raw["claim"].pop("assumptions")
    raw["experimental_evidence"] = raw["claim"].pop("experimental_evidence")
    session, _, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", raw)])
    result = await dispatch()
    assert result["status"] == "retained"
    event, = [event for event in session.audit if event.get("event") == "submission_normalized"]
    assert event["raw_submission"] == raw
    assert event["normalized_submission"] == assessment_payload()
    assert len(event["changes"]) == 2
    assert event["evidence_catalog_id"]
    assert not any(event.get("event") == "submission_validation_failed" for event in session.audit)


async def test_invalid_alias_gets_one_targeted_selection_without_changing_science(tmp_path, index, phenotype, context_index, discovery):
    raw = assessment_payload("E9999")
    session, _, _, dispatch = await evaluation_session(tmp_path, index, phenotype, context_index, discovery,
        [response("SubmitAssessment", raw), response("SubmitAssessment", {"claim": {
            "cell": {"evidence_refs": ["E1"]}, "abnormality": {"evidence_refs": ["E1"]}}})])
    failed = await dispatch()
    assert failed["status"] == "invalid_reference"
    assert {tuple(issue["path"]) for issue in failed["issues"]} == {
        ("claim", "cell", "evidence_refs", 0), ("claim", "abnormality", "evidence_refs", 0)}
    completed = await dispatch(mode="complete_reasoning", failure_id=failed["failure_id"])
    assert completed["status"] == "retained"
    assert session.claims[completed["claim_id"]]["draft"].cell_text == raw["claim"]["cell"]["cell_text"]


def test_summary_counts_and_exports_share_active_observations(tmp_path, phenotype):
    from hpo_agent.session import ResearchSession
    session = ResearchSession(phenotype=phenotype, engine=None, settings=Settings(), output=tmp_path, attempt=tmp_path / "attempt")
    active = SimpleNamespace(release_scope="conditional", experimental_validation=SimpleNamespace(evidence=["one"]))
    retired = SimpleNamespace(release_scope="conditional", experimental_validation=SimpleNamespace(evidence=["old", "old"]))
    session.observations = {"active": active, "retired": retired}
    session.retired["retired"] = "Superseded"
    assert session.active_observations() == [active]
    assert session.summary_counts()["release_scopes"] == {"conditional": 1}
    assert session.summary_counts()["experimental_entries"] == 1
