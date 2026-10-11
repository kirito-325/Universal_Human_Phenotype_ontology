"""Scientific inference provenance and terminating dictionary-only submission."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_runtime import DISPATCH, MAPPING, ScriptedModel, finish, response

from hpo_agent import prompts
from hpo_agent.agents import AgentEngine
from hpo_agent.pipeline import BatchRunner
from hpo_agent.schemas import ClaimDraft, Extraction, Settings, TermResult
from hpo_agent.session import ResearchSession
from hpo_agent.storage import read_jsonl
from hpo_agent.validation import anchor_claim, anchor_reasoning


@pytest.fixture
def knowledge_claim(phenotype, draft):
    return ClaimDraft.model_validate({
        **draft.model_dump(), "evidence_mode": "inferred", "cell_evidence_mode": "inferred",
        "link_evidence_mode": "inferred", "bridge_scope": "mechanistically_compatible",
        "supporting_quote": "", "assertion": "model_inference", "evidence_type": "inference",
        "relation_type": "mechanistically_compatible_with",
        "reasoning_steps": [{"source": phenotype.label, "target": draft.cellular_statement,
                             "scope": "cell_identity", "basis": "model_knowledge",
                             "relation": "hypothesized_cellular_mechanism",
                             "explanation": "SYNTHETIC model-knowledge software fixture, not a biological assertion.",
                             "citations": [], "assumptions": ["Synthetic applicability assumption."]}],
    })


async def test_valid_ids_and_labels_submit_once_without_literature_review(index, knowledge_claim):
    # The second scripted answer would repeat the choice if the graph failed to end.
    model = ScriptedModel(responses=[response("SubmitMapping", MAPPING), response("SubmitMapping", MAPPING)])
    engine = AgentEngine(index, Settings(workflow_mode="component_test", ), models={"normalize": model})
    audit = []
    mapped = await engine.normalize(knowledge_claim, audit)
    assert mapped.cl.term_id == MAPPING["cl_id"] and mapped.go.label == MAPPING["go_label"]
    assert sum(e.get("event") == "agent_message" and e.get("message_type") == "ai" for e in audit) == 1
    assert "SubmitMapping" in model.observed_tools
    assert "validate_mapping" not in model.observed_tools


async def test_known_id_needs_no_prior_model_lookup_or_paper_support(index, knowledge_claim):
    # This valid term is outside the initial lexical candidates for apoptosis.
    choice = {**MAPPING, "go_id": "GO:9000001", "go_label": "test molecular function"}
    model = ScriptedModel(responses=[response("SubmitMapping", choice)])
    engine = AgentEngine(index, Settings(workflow_mode="component_test", ), models={"normalize": model})
    audit = []
    mapped = await engine.normalize(knowledge_claim, audit)
    assert mapped.go.term_id == "GO:9000001"
    assert mapped.go.namespace == "molecular_function"
    assert "GO:9000001" in mapped.candidate_ids
    assert sum(e.get("event") == "agent_message" and e.get("message_type") == "ai" for e in audit) == 1


@pytest.mark.parametrize("wrong", [{"cl_label": "wrong label"}, {"go_id": "GO:9999999"}])
async def test_wrong_id_or_label_can_be_corrected_then_submission_ends(index, knowledge_claim, wrong):
    model = ScriptedModel(responses=[response("SubmitMapping", {**MAPPING, **wrong}),
                                    response("SubmitMapping", MAPPING)])
    engine = AgentEngine(index, Settings(workflow_mode="component_test", ), models={"normalize": model})
    audit = []
    mapped = await engine.normalize(knowledge_claim, audit)
    assert mapped.cl.label == "neuron"
    assert sum(e.get("event") == "agent_message" and e.get("message_type") == "ai" for e in audit) == 2


async def test_rewording_invalid_submission_does_not_bypass_guard(index, knowledge_claim):
    model = ScriptedModel(responses=[response("SubmitMapping", {**MAPPING, "cl_label": "wrong label",
                                                               "rationale": f"Reworded rationale {i}"}) for i in range(3)])
    engine = AgentEngine(index, Settings(workflow_mode="component_test", recursion_limit=600), models={"normalize": model})
    audit = []
    with pytest.raises(ValueError, match="no_progress"):
        await engine.normalize(knowledge_claim, audit)
    assert sum(e.get("event") == "agent_message" and e.get("message_type") == "ai" for e in audit) == 3


async def test_real_root_loop_exports_model_knowledge_without_fake_evidence(
    tmp_path, index, phenotype, knowledge_claim, review,
):
    root = ScriptedModel(responses=[response("eval", {"code": DISPATCH + """
var proposed = await dispatch("evidence-reasoner", {});
var cid = proposed.claims[0].claim_id;
await dispatch("ontology-normalizer", {claim_id:cid});
await dispatch("evidence-reviewer", {claim_id:cid});
var state = await tools.getResearchState({});
({ready:state.ready_to_finish,version:state.state_version});
"""}), finish(3)])
    approved = {**review.model_dump(), "verdict": "partial", "inference_supported": True, "scope_transfer_supported": True,
                "cell_inference_supported": True}
    models = {
        "research": root,
        "reason": ScriptedModel(responses=[response("Extraction", {"claims": [knowledge_claim.model_dump()]})]),
        "normalize": ScriptedModel(responses=[response("SubmitMapping", MAPPING)]),
        "review": ScriptedModel(responses=[response("Review", approved)]),
    }
    settings = Settings(workflow_mode="component_test", offline=True)
    runner = BatchRunner(engine=AgentEngine(index, settings, models=models), index=index, settings=settings,
                         output=tmp_path / "run")
    result = (await runner.run([phenotype]))[0]
    assert result.status == "completed", result.errors
    assert result.outcome == "conditional_only" and result.schema_version == "2.0"
    assert not result.document_ids
    observation = result.observations[0]
    assert observation.knowledge_source == "model_knowledge" and observation.evidence is None
    assert observation.evidence_mode == "inferred" and " -> " in observation.reasoning_chain
    assert observation.experimental_validation.status == "not_recorded"
    assert observation.experimental_validation.evidence == []
    assert all(not step.evidence for step in observation.reasoning_steps)
    assert TermResult.model_validate_json(result.model_dump_json()).observations == result.observations
    assert read_jsonl(tmp_path / "run" / "inferred_edges.jsonl")[0]["evidence"] is None
    assert read_jsonl(tmp_path / "run" / "relations.jsonl")[0]["experimental_validation"] == {
        "status": "not_recorded", "evidence": [],
    }
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["accepted_knowledge_sources"]["model_knowledge"] == 1
    assert (await runner.run([phenotype]))[0].coverage["audit_path"] == result.coverage["audit_path"]


def test_model_knowledge_cannot_masquerade_as_reported_observation(knowledge_claim):
    with pytest.raises(ValidationError):
        ClaimDraft.model_validate({**knowledge_claim.model_dump(), "assertion": "own_result"})
    with pytest.raises(ValidationError):
        ClaimDraft.model_validate({**knowledge_claim.model_dump(), "evidence_mode": "direct"})
    row = knowledge_claim.model_dump()
    row["reasoning_steps"][0]["citations"] = [{"document_id": "invented", "quote": "Invented citation"}]
    with pytest.raises(ValidationError, match="model_knowledge"):
        ClaimDraft.model_validate(row)


def test_mixed_literature_and_uncited_model_knowledge_keep_distinct_provenance(
    phenotype, draft, document, knowledge_claim,
):
    row = knowledge_claim.model_dump()
    model_step = row["reasoning_steps"][0]
    model_step["source"] = "Reported synthetic cellular observation"
    row.update({"supporting_quote": draft.supporting_quote, "assertion": "own_result",
                "evidence_type": "observation", "reasoning_steps": [
                    {"source": phenotype.label, "target": model_step["source"], "relation": "reported finding",
                     "basis": "literature", "explanation": "Synthetic source observation.",
                     "citations": [{"document_id": document.document_id, "quote": draft.supporting_quote}]},
                    model_step,
                ]})
    mixed = ClaimDraft.model_validate(row)
    evidence = anchor_claim(mixed, document, 0, len(document.text))
    steps = anchor_reasoning(phenotype, mixed, evidence, {document.document_id: document},
                             [{"document_id": document.document_id, "start": 0, "end": len(document.text)}])
    assert mixed.knowledge_source == "mixed"
    assert steps[0].evidence and not steps[1].evidence


async def test_specialist_can_choose_main_quote_from_a_read_supporting_document(tmp_path, phenotype, document, draft):
    primary = document.model_copy(update={"document_id": "initial", "text": "Unrelated source context."})
    selected = draft.model_copy(update={"supporting_document_id": document.document_id})

    async def extract(payload, audit, tools=()):
        read = next(t for t in tools if t.name == "read_document")
        await read.ainvoke({"document_id": document.document_id})
        return Extraction(claims=[selected])

    session = ResearchSession(phenotype=phenotype, engine=SimpleNamespace(extract=extract), settings=Settings(workflow_mode="component_test", ),
                              output=tmp_path, attempt=tmp_path / "attempt")
    await session.stage(primary)
    await session.stage(document)
    result = await session.extract({"document_id": primary.document_id,
                                    "supporting_document_ids": [document.document_id]})
    assert len(result["claims"]) == 1 and not result["invalid_claims"]
    record = session.claims[result["claims"][0]["claim_id"]]
    assert record["evidence"].document_id == document.document_id
    assert session.evidence_payload(record)["document_id"] == document.document_id


async def test_model_proposal_does_not_require_fetching_hits_from_an_earlier_probe(tmp_path, phenotype, knowledge_claim):
    async def reason(payload, audit, tools=()):
        return Extraction(claims=[knowledge_claim])

    session = ResearchSession(phenotype=phenotype, engine=SimpleNamespace(reason=reason), settings=Settings(workflow_mode="component_test", ),
                              output=tmp_path, attempt=tmp_path / "attempt")
    session.search_log.append({"query": "synthetic query", "count": 12, "pmids": []})
    assert session.pending()["unfetched_search_hits"]
    await session.reason({})
    assert not session.pending()["unfetched_search_hits"]
    assert session.pending()["claims"]  # Still requires normalization and review.


def test_production_prompts_do_not_embed_previous_test_answers():
    combined = "\n".join(getattr(prompts, name) for name in ("ROOT", "EXTRACT", "REASON", "NORMALIZE", "REVIEW"))
    for answer_seed in ("HP:0009511", "CL:1000217", "GO:0030282", "growth-plate chondrocyte", "EBP"):
        assert answer_seed not in combined
