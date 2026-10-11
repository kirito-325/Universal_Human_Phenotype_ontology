"""Compact model interfaces compile into complete, strictly validated research records."""

import json

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from test_assessment_repair import SchemaRecordingModel
from test_literature_workflow import context_index as context_index
from test_literature_workflow import discovery as discovery
from test_literature_workflow import prepared_session
from test_runtime import DISPATCH, ScriptedModel, response

from hpo_agent import prompts
from hpo_agent.agents import AgentEngine
from hpo_agent.assembly import (
    assemble_assessment,
    assemble_hypotheses,
    assemble_mapping,
    assemble_review,
    model_window,
    read_passages,
    resolve_refs,
)
from hpo_agent.contracts import SCIENCE_POLICY
from hpo_agent.pipeline import BatchRunner
from hpo_agent.schemas import (
    MODEL_CONTRACT,
    SCHEMA_VERSION,
    HypothesisProposals,
    Observation,
    OntologySelection,
    ScientificAssessment,
    ScientificCompletion,
    ScientificReview,
    Settings,
)
from hpo_agent.storage import read_jsonl
from hpo_agent.validation import (
    anchor_claim,
    anchor_experiments,
    anchor_reasoning,
    release_gate,
    resolve_mapping,
)


def window(document, start=0, end=None):
    end = len(document.text) if end is None else end
    return {"document_id": document.document_id, "start": start, "end": end,
            "text": document.text[start:end], "passages": read_passages(document, start, end)}


def assessment_payload(ref=None, *, experiment=False):
    refs = [ref] if ref else []
    return {"decision": "retain", "rationale": "Synthetic scientific inference for software verification.",
        "claim": {
            "cell": {"cell_text": "neurons", "explanation": "Synthetic cell identity justification.", "evidence_refs": refs},
            "abnormality": {"process_text": "apoptosis", "quality_text": "increased apoptosis", "direction": "increased",
                "explanation": "Synthetic cellular abnormality justification.", "evidence_refs": refs},
            "phenotype_link": {"explanation": "Synthetic phenotype applicability explanation.", "evidence_refs": []},
            "assumptions": ["Synthetic anatomical applicability assumption, not a biological assertion."],
            "experimental_evidence": ([{"evidence_ref": ref, "scope": "mechanism", "study_type": "observation",
                "species": "human", "experimental_context": "Synthetic study context.",
                "finding": "Synthetic observed cellular abnormality.", "effect": "supports"}] if experiment else []),
        }}


def judgment_payload(*, experiment=False):
    judgment = {"status": "uncertain", "reason": "Scientifically plausible in this synthetic test; unverified."}
    value = {name: dict(judgment) for name in ("cell_identity", "cell_abnormality", "phenotype_link", "ontology_mapping")}
    if experiment:
        value["experimental_evidence"] = {"status": "defensible", "reason": "Synthetic component study scope preserved."}
    return value


def proposals_payload(discovery):
    documents, _, batch = discovery
    return {"hypotheses": [{**hyp.model_dump(exclude={"seed_evidence"}),
                            "seed_evidence_refs": [read_passages(doc, 0, len(doc.text))[-1]["evidence_ref"]]}
                           for hyp, doc in zip(batch.hypotheses, documents, strict=True)],
            "diversity_rationale": batch.diversity_rationale}


def test_read_refs_are_stable_source_specific_and_anchored(document):
    passages = read_passages(document, 0, len(document.text))
    assert passages == read_passages(document, 0, len(document.text))
    assert all(document.text[p["start"]:p["end"]] == p["text"] for p in passages)
    other = document.model_copy(update={"document_id": "PMC123"})
    assert {p["evidence_ref"] for p in passages}.isdisjoint(
        {p["evidence_ref"] for p in read_passages(other, 0, len(other.text))})
    shown = model_window(window(document))
    assert "text" not in shown and all(set(p) == {"evidence_ref", "text"} for p in shown["passages"])


def test_refs_do_not_authorize_another_tasks_unread_source(document):
    other = document.model_copy(update={"document_id": "other"})
    unread = read_passages(other, 0, len(other.text))[-1]["evidence_ref"]
    docs = {document.document_id: document, other.document_id: other}
    with pytest.raises(ValueError) as raised:
        resolve_refs([unread], docs, [window(document)], field=("claim", "cell", "evidence_refs"))
    assert raised.value.type == "evidence.ref"
    assert raised.value.context["field"] == ["claim", "cell", "evidence_refs", 0]


def test_ref_spelling_is_normalized_without_changing_its_selected_source(document):
    supplied = window(document)
    ref = supplied["passages"][-1]["evidence_ref"]
    span, = resolve_refs(["  " + ref.upper() + "  "], {document.document_id: document}, [supplied], field=("refs",))
    assert span.document_id == document.document_id and span.quote == supplied["passages"][-1]["text"]


def test_changed_source_text_cannot_reuse_an_old_ref(document):
    supplied = window(document)
    ref = supplied["passages"][-1]["evidence_ref"]
    changed = document.model_copy(update={"text": "A different immutable source version."})
    with pytest.raises(ValueError, match="Unknown or unread"):
        resolve_refs([ref], {changed.document_id: changed}, [window(changed)], field=("refs",))


@pytest.mark.parametrize("with_reference,with_experiment", [(False, False), (True, False), (True, True)])
def test_host_builds_identity_provenance_chain_and_evidence_without_model_metadata(
    phenotype, document, mapping, with_reference, with_experiment,
):
    supplied = window(document)
    ref = supplied["passages"][-1]["evidence_ref"] if with_reference else None
    raw = assessment_payload(ref, experiment=with_experiment)
    value = ScientificAssessment.model_validate(raw)
    original = value.model_dump()
    docs, windows = {document.document_id: document}, [supplied]
    result = assemble_assessment(value, phenotype, "host_hypothesis", docs, windows)
    assert value.model_dump() == original
    claim = result.claim
    assert claim.hypothesis_id == "host_hypothesis" and claim.species == "human" and claim.taxon_id == "NCBITaxon:9606"
    assert claim.evidence_mode == claim.cell_evidence_mode == claim.link_evidence_mode == "inferred"
    assert claim.assertion == "model_inference" and claim.supporting_quote == ""
    assert [step.scope for step in claim.reasoning_steps] == ["cell_identity", "mechanism", "phenotype_link"]
    assert claim.reasoning_steps[0].source == phenotype.label
    assert claim.reasoning_steps[-1].target == claim.cellular_statement
    assert all(a.target == b.source for a, b in zip(claim.reasoning_steps, claim.reasoning_steps[1:], strict=False))
    assert result.supporting_document_ids == ([document.document_id] if with_reference else [])
    evidence = anchor_claim(claim, None, 0, 0)
    grounded = anchor_reasoning(phenotype, claim, evidence, docs, windows)
    experiments = anchor_experiments(claim, docs, windows)
    review = assemble_review(ScientificReview.model_validate(judgment_payload(experiment=with_experiment)), with_experiment)
    scope, reasons = release_gate(phenotype, claim, evidence, mapping, review, grounded)
    assert scope == "conditional" and not review.blocking_issues
    observation = Observation(claim_id="synthetic", phenotype_id=phenotype.phenotype_id, phenotype_label=phenotype.label,
        claim=claim, evidence=None, evidence_mode="inferred", reasoning_steps=grounded, mapping=mapping,
        experimental_validation={"evidence": [entry.model_dump() for entry in experiments]},
        review=review, release_scope=scope, gate_reasons=reasons)
    assert Observation.model_validate_json(observation.model_dump_json()) == observation


def test_model_does_not_fill_species_taxon_or_string_endpoints(phenotype, document):
    raw = assessment_payload()
    raw["claim"]["cell"]["cell_text"] = "SYNTHETIC Cell With Mixed Case"
    raw["claim"]["abnormality"]["quality_text"] = "decreased clearance, with preserved context"
    raw["claim"]["abnormality"]["direction"] = "decreased"
    value = ScientificAssessment.model_validate(raw)
    result = assemble_assessment(value, phenotype, "model_hypothesis", {}, [])
    assert result.claim.species == "human" and result.claim.taxon_id == "NCBITaxon:9606"
    assert result.claim.direction == "decreased"
    assert result.claim.reasoning_steps[-1].target == result.claim.cellular_statement
    assert result.claim.reasoning_steps[-1].target.startswith("SYNTHETIC Cell With Mixed Case:")


def test_scientific_explanations_cannot_be_fabricated_by_defaults():
    raw = assessment_payload()
    del raw["claim"]["cell"]["explanation"]
    with pytest.raises(ValidationError):
        ScientificAssessment.model_validate(raw)
    raw = assessment_payload()
    raw["claim"]["assumptions"] = []
    with pytest.raises(ValidationError):
        ScientificAssessment.model_validate(raw)


def test_unknown_experiment_ref_names_the_scalar_field(phenotype, document):
    supplied = window(document)
    raw = assessment_payload(supplied["passages"][-1]["evidence_ref"], experiment=True)
    raw["claim"]["experimental_evidence"][0]["evidence_ref"] = "ev_unknown"
    with pytest.raises(ValueError) as raised:
        assemble_assessment(ScientificAssessment.model_validate(raw), phenotype, "hid", {document.document_id: document}, [supplied])
    assert raised.value.errors()[0]["ctx"]["field"] == ["claim", "experimental_evidence", 0, "evidence_ref"]


def test_mapping_labels_are_looked_up_not_generated(index):
    value = OntologySelection(cl_id="CL:9000001", go_id="GO:0006915", cell_confidence="high",
        process_confidence="high", rationale="Synthetic term selection.")
    choice = assemble_mapping(value, index, set())
    mapping = resolve_mapping(choice, index, set())
    assert mapping.cl.term_id == "CL:0000540" and mapping.cl.label == "neuron"
    assert mapping.go.label == "apoptotic process"


@pytest.mark.parametrize("field", ["cell_identity", "cell_abnormality", "phenotype_link", "ontology_mapping", "experimental_evidence"])
def test_actual_scientific_defects_remain_blocking(field, phenotype, document, mapping):
    supplied = window(document)
    raw = assessment_payload(supplied["passages"][-1]["evidence_ref"], experiment=True)
    assessment = assemble_assessment(ScientificAssessment.model_validate(raw), phenotype, "hid", {document.document_id: document}, [supplied])
    judgments = judgment_payload(experiment=True)
    judgments[field] = {"status": "invalid", "reason": "SYNTHETIC concrete inconsistency, not missing experimental proof."}
    review = assemble_review(ScientificReview.model_validate(judgments), True)
    grounded = anchor_reasoning(phenotype, assessment.claim, None, {document.document_id: document}, [supplied])
    assert release_gate(phenotype, assessment.claim, None, mapping, review, grounded)[0] == "rejected"
    assert review.blocking_issues and review.verdict == "contradictory"


def test_recorded_experiments_require_a_judgment_but_absent_experiments_do_not():
    value = ScientificReview.model_validate(judgment_payload())
    assert assemble_review(value, False).experimental_evidence_supported
    with pytest.raises(ValueError, match="experimental entries require"):
        assemble_review(value, True)


def test_production_schemas_exclude_host_owned_fields():
    schema = ScientificAssessment.model_json_schema()
    claim_fields = schema["$defs"]["CellularProposal"]["properties"]
    assert set(claim_fields) == {"cell", "abnormality", "phenotype_link", "assumptions", "disease_context", "experimental_evidence"}
    for name in ("CellReason", "AbnormalityReason", "PhenotypeReason"):
        fields = schema["$defs"][name]["properties"]
        assert not ({"source", "target", "basis", "scope", "citations", "taxon_id", "quote"} & fields.keys())
    assert "cl_label" not in OntologySelection.model_json_schema()["properties"]
    assert "state_version" not in ScientificCompletion.model_json_schema()["properties"]
    assert not ({"verdict", "blocking_issues", "cell_supported"} & ScientificReview.model_json_schema()["properties"].keys())


async def test_actual_model_receives_only_the_compact_contract_and_submission_schema(index, phenotype, document):
    supplied = window(document)
    ref = supplied["passages"][-1]["evidence_ref"]
    model = SchemaRecordingModel(responses=[response("SubmitAssessment", assessment_payload(ref))])
    engine = AgentEngine(index, Settings(), models={"verify_hypothesis": model})
    audit = []
    result = await engine.verify_hypothesis({"phenotype": phenotype.model_dump(), "model_contract": MODEL_CONTRACT,
        "seed_contexts": [model_window(supplied)]}, audit,
        assembler=lambda value: assemble_assessment(value, phenotype, "hid", {document.document_id: document}, [supplied]))
    assert result.claim.hypothesis_id == "hid"
    actual = model.received_system_messages[0]
    assert prompts.VERIFY_HYPOTHESIS in actual and SCIENCE_POLICY in actual
    assert "COMPACT OUTPUT CONTRACT" in actual and "FORMAL OUTPUT CONTRACT" not in actual
    bound = next(item["schema"] for item in model.bound_schemas if item["name"] == "SubmitAssessment")
    assert "ScientificReason" not in bound.get("$defs", {}) or "source" not in bound["$defs"]["ScientificReason"]["properties"]
    assert "cell" in bound["$defs"]["CellularProposal"]["properties"]
    record = next(event for event in audit if event.get("model_output"))
    assert record["model_contract"] == MODEL_CONTRACT
    assert "reasoning_steps" not in record["model_output"]["claim"] and record["output"]["claim"]["reasoning_steps"]


async def test_fragment_lookup_reuses_read_refs_without_another_read(tmp_path, index, phenotype, context_index, discovery):
    engine = AgentEngine(index, Settings(), models={})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine, compact=True)
    provided, tools = session.research_workspace()
    read, find = tools
    before = await find.ainvoke({"document_id": "synthetic:0", "query": "apoptosis"})
    assert before["read_evidence_refs"] == []
    result = await read.ainvoke({"document_id": "synthetic:0"})
    assert "text" not in result and result["passages"]
    after = await find.ainvoke({"document_id": "synthetic:0", "query": "apoptosis"})
    assert after["read_evidence_refs"] and len(provided) == 1


async def test_compact_insufficient_pool_can_stop_without_inventing_candidates(tmp_path, index, phenotype, context_index, discovery):
    model = ScriptedModel(responses=[response("SubmitHypotheses", {"hypotheses": [],
        "diversity_rationale": "Synthetic incomplete exploration.", "limitation": "Cannot justify a diverse pool in this synthetic test."})])
    engine = AgentEngine(index, Settings(offline=True), models={"hypothesize": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine, compact=True)
    result = await session.generate_hypotheses({})
    assert result["status"] == "needs_more_evidence" and not session.workflow.hypotheses
    assert "generate_minimum_hypothesis_batch" in session.pending()["workflow"]
    assert not [event for event in session.audit if event.get("event") == "submission_validation_failed"]


async def test_compact_context_cannot_silently_use_the_legacy_model_schema(index, phenotype):
    engine = AgentEngine(index, Settings(), models={})
    with pytest.raises(ValueError, match="requires_host_assembler"):
        await engine.verify_hypothesis({"phenotype": phenotype.model_dump(), "model_contract": MODEL_CONTRACT}, [])


async def test_unknown_ref_preserves_science_and_production_has_no_patch_mode(
    tmp_path, index, phenotype, context_index, discovery,
):
    valid_ref = read_passages(discovery[0][0], 0, len(discovery[0][0].text))[-1]["evidence_ref"]
    bad = assessment_payload(valid_ref)
    bad["claim"]["cell"]["evidence_refs"] = ["ev_unknown"]
    model = ScriptedModel(responses=[response("SubmitAssessment", bad) for _ in range(3)])
    engine = AgentEngine(index, Settings(offline=True), models={"verify_hypothesis": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine, compact=True)
    batch = assemble_hypotheses(HypothesisProposals.model_validate(proposals_payload(discovery)), session.documents,
                               [session.read(doc.document_id) for doc in discovery[0]])
    hypothesis = session.workflow.register_hypotheses(batch, [session.read(doc.document_id) for doc in discovery[0]])[0]
    hid = hypothesis["hypothesis_id"]
    await session.search_local("route0", hypothesis_id=hid, intent="support")
    await session.search_local("counter route0", hypothesis_id=hid, intent="challenge")
    adapter = next(spec["runnable"] for spec in session.subagents() if spec["name"] == "hypothesis-evaluator")

    async def dispatch(request):
        return (await adapter.ainvoke({"messages": [HumanMessage(json.dumps(request))]}))["structured_response"]

    failed = await dispatch({"hypothesis_id": hid})
    assert failed["next_action"] == "complete_reasoning"
    assert failed["failed_submission"] == bad
    assert session.failed_assessments[hid]["model_contract"] == MODEL_CONTRACT
    assert failed["issues"][0]["path"] == ["claim", "cell", "evidence_refs", 0]
    blocked = await dispatch({"hypothesis_id": hid})
    assert blocked["error"] == "scientific_submission_requires_explicit_reassessment"
    patch = await dispatch({"hypothesis_id": hid, "mode": "repair", "failure_id": failed["failure_id"]})
    assert patch["error"] == "compact_assessment_mode_must_be_evaluate_complete_reasoning_or_reevaluate"
    assert not session.claims and hypothesis["status"] == "proposed"
    good = assessment_payload(valid_ref)
    good["revision_note"] = "Corrected the selected source reference; scientific conclusion unchanged."
    model.responses, model.i = [response("SubmitAssessment", good)], 0
    recovered = await dispatch({"hypothesis_id": hid, "mode": "reevaluate",
        "reassessment_reason": "Reconsider the source selection against the preserved scientific draft."})
    assert recovered["status"] == "retained"
    recovery, = session.coverage["assessment_recoveries"]
    assert recovery["previous_submission"] == bad and recovery["mode"] == "reevaluate"
    assert recovery["assessment"]["claim"]["cell_text"] == bad["claim"]["cell"]["cell_text"]
    assert "SubmitAssessmentRepair" not in model.observed_tools


@pytest.mark.parametrize("with_experiment", [False, True])
async def test_full_compact_root_eval_task_workflow_exports_complete_records(
    tmp_path, index, phenotype, context_index, discovery, with_experiment,
):
    documents, plan, _ = discovery
    root_code = DISPATCH + "\nvar plan = " + json.dumps(plan.model_dump()) + ";\n" + """
await tools.getPhenotypeBackground({});
await tools.registerDiscoveryPlan({plan});
for (var i=0; i<5; i++) await tools.searchLocalDocuments({query:'route'+i,direction_id:'d'+i});
var pool = await dispatch('hypothesis-generator', {});
var kept = [];
for (var i=0; i<pool.hypotheses.length; i++) {
  var hid = pool.hypotheses[i].hypothesis_id;
  await tools.searchLocalDocuments({query:'route'+i,hypothesis_id:hid,intent:'support'});
  var assessment = await dispatch('hypothesis-evaluator', {hypothesis_id:hid});
  if (assessment.claim_id) kept.push(assessment.claim_id);
}
for (var cid of kept) {
  await dispatch('ontology-normalizer', {claim_id:cid});
  await dispatch('evidence-reviewer', {claim_id:cid});
}
var state = await tools.getResearchState({});
({ready:state.ready_to_finish,counts:state.summary_counts});
"""
    evaluations = []
    for i, doc in enumerate(documents):
        ref = read_passages(doc, 0, len(doc.text))[-1]["evidence_ref"]
        payload = assessment_payload(ref, experiment=with_experiment) if i == 0 else {
            "decision": "inconclusive" if i == 4 else "reject", "rationale": "Synthetic alternative assessment.",
            "supporting_evidence_refs": [ref], "claim": None}
        evaluations.append(response("SubmitAssessment", payload))
    root = SchemaRecordingModel(responses=[response("eval", {"code": root_code}), response("ResearchCompletion", {
        "stop_reason": "sufficient_evidence", "summary": "Synthetic scientific synthesis.", "limitations": ["Software test only."]})])
    models = {"research": root,
        "hypothesize": ScriptedModel(responses=[*[response("read_document", {"document_id": doc.document_id}) for doc in documents],
            response("SubmitHypotheses", proposals_payload(discovery))]),
        "verify_hypothesis": ScriptedModel(responses=evaluations),
        "normalize": ScriptedModel(responses=[response("SubmitMapping", {"cl_id": "CL:0000540", "go_id": "GO:0006915",
            "cell_confidence": "high", "process_confidence": "high", "rationale": "Synthetic term selection."})]),
        "review": ScriptedModel(responses=[response("Review", judgment_payload(experiment=with_experiment))])}
    runner = BatchRunner(engine=AgentEngine(index, Settings(offline=True), models=models), index=index,
        settings=Settings(offline=True), output=tmp_path / "compact_run", local_documents=documents, hpo_context=context_index)
    result = (await runner.run([phenotype]))[0]
    assert result.status == "completed", result.errors
    assert result.schema_version == SCHEMA_VERSION and result.outcome == "conditional_only"
    assert len(result.observations) == 1 and result.observations[0].release_scope == "conditional"
    assert result.observations[0].claim.hypothesis_id
    assert len(result.observations[0].experimental_validation.evidence) == int(with_experiment)
    assert result.observations[0].mapping.cl.label == "neuron"
    assert not result.coverage["pending"]["workflow"] and not result.errors
    ledger = read_jsonl(tmp_path / "compact_run/hypotheses.jsonl")
    assert [entry["status"] for entry in ledger] == ["retained", "rejected", "rejected", "rejected", "inconclusive"]
    assert result.coverage["root_completion"]["state_version"] > 0
    assert "Host workflow counts:" in result.coverage["root_completion"]["summary"]
    audit = read_jsonl(tmp_path / "compact_run" / result.coverage["audit_path"])
    assert not [event for event in audit if event.get("event") == "submission_validation_failed"]
    outputs = [event for event in audit if "model_output" in event]
    assert {event["role"] for event in outputs} == {"research", "hypothesize", "verify_hypothesis", "normalize", "review"}
    assert all(event["model_contract"] == MODEL_CONTRACT for event in outputs)
    assert "state_version" not in next(event["model_output"] for event in outputs if event["role"] == "research")
    assert all(prompts.ROOT in text for text in root.received_system_messages)
    assert "SubmitAssessmentRepair" not in models["verify_hypothesis"].observed_tools
