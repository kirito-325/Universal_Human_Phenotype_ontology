"""Synthetic evidence chains: software contracts only, not biological conclusions."""

import json

import pytest
from pydantic import ValidationError
from test_runtime import DISPATCH, MAPPING, ScriptedModel, finish, response

from hpo_agent.agents import AgentEngine
from hpo_agent.pipeline import BatchRunner
from hpo_agent.schemas import (
    ClaimDraft,
    Document,
    Extraction,
    Observation,
    Settings,
    TermResult,
)
from hpo_agent.session import ResearchSession
from hpo_agent.storage import read_jsonl
from hpo_agent.validation import anchor_claim, anchor_reasoning, claim_id, release_gate


@pytest.fixture
def chain_case(phenotype, document, draft, review):
    phenotype = phenotype.model_copy(update={
        "label": "SYNTHETIC regional episodes",
        "definition": "A regional subtype of synthetic network dysfunction.",
    })
    anchor = document.model_copy(update={
        "text": "SYNTHETIC human neuronal injury samples showed increased apoptosis in neurons.",
    })
    bridge = Document(
        document_id="synthetic:bridge", title="SYNTHETIC PREMISE, NOT A PUBLICATION",
        text="In a SYNTHETIC human cohort, network dysfunction was associated with neuronal injury.",
        source_url="urn:synthetic:bridge", source_kind="local_text", pmid="456",
        phenotype_ids=[phenotype.phenotype_id],
    )
    steps = [
        {"source": phenotype.label, "target": "Synthetic network dysfunction",
         "relation": "subtype_of", "basis": "phenotype_definition",
         "explanation": "The supplied definition identifies this broader pathology."},
        {"source": "Synthetic network dysfunction", "target": "Synthetic neuronal injury",
         "relation": "potentially_shares_associated_mechanism", "basis": "inference",
         "explanation": "Apply the general cohort association to this regional subtype conditionally.",
         "assumptions": ["The general association may apply to this unstudied regional subtype."],
         "citations": [{"document_id": bridge.document_id, "quote": bridge.text}]},
        {"source": "Synthetic neuronal injury", "target": draft.cellular_statement,
         "relation": "has_observed_cellular_abnormality", "basis": "literature",
         "explanation": "The anchor reports an abnormal process in explicitly named cells.",
         "citations": [{"document_id": anchor.document_id, "quote": anchor.text}]},
    ]
    claim = ClaimDraft.model_validate({
        **draft.model_dump(), "evidence_mode": "inferred", "reasoning_steps": steps,
        "link_evidence_mode": "inferred",
        "supporting_quote": anchor.text, "bridge_scope": "phenotype_mechanism_bridge",
        "relation_type": "mechanistically_compatible_with",
    })
    reviewed = review.model_copy(update={"inference_supported": True, "scope_transfer_supported": True})
    documents = {d.document_id: d for d in (anchor, bridge)}
    windows = [{"document_id": d.document_id, "start": 0, "end": len(d.text)} for d in documents.values()]
    evidence = anchor_claim(claim, anchor, 0, len(anchor.text))
    return phenotype, claim, evidence, documents, windows, reviewed


def test_grounded_inference_is_retained_without_exact_phenotype_in_any_paper(chain_case, mapping):
    phenotype, claim, evidence, documents, windows, review = chain_case
    assert all(phenotype.label not in d.text for d in documents.values())
    steps = anchor_reasoning(phenotype, claim, evidence, documents, windows)
    scope, reasons = release_gate(phenotype, claim, evidence, mapping, review, steps)
    assert scope == "conditional" and "literature_grounded_inference" in reasons
    observation = Observation(
        claim_id="synthetic", phenotype_id=phenotype.phenotype_id, phenotype_label=phenotype.label,
        claim=claim, evidence_mode="inferred", evidence=evidence, reasoning_steps=steps,
        mapping=mapping, review=review, release_scope=scope, gate_reasons=reasons,
    )
    assert observation.reasoning_chain == (
        "SYNTHETIC regional episodes -> Synthetic network dysfunction -> Synthetic neuronal injury"
        " -> neurons: increased apoptosis"
    )
    assert Observation.model_validate_json(observation.model_dump_json()) == observation
    for step in observation.reasoning_steps:
        for span in step.evidence:
            source = documents[span.document_id]
            assert source.text[span.span_start:span.span_end] == span.quote
            assert source.sha256 == span.text_sha256
    wrong_chain = {**observation.model_dump(), "reasoning_chain": "An invented arrow chain"}
    with pytest.raises(ValidationError, match="reasoning_chain"):
        Observation.model_validate(wrong_chain)


@pytest.mark.parametrize("field", ["inference_supported", "scope_transfer_supported", "abnormality_supported"])
def test_inference_does_not_bypass_review(chain_case, mapping, field):
    phenotype, claim, evidence, documents, windows, review = chain_case
    steps = anchor_reasoning(phenotype, claim, evidence, documents, windows)
    review = review.model_copy(update={field: False})
    assert release_gate(phenotype, claim, evidence, mapping, review, steps)[0] == "rejected"


def test_inference_preserves_species_and_does_not_assert_demonstrated_causality(chain_case, mapping):
    phenotype, claim, evidence, documents, windows, review = chain_case
    steps = anchor_reasoning(phenotype, claim, evidence, documents, windows)
    assert release_gate(phenotype, claim, evidence, mapping, review)[0] == "rejected"
    causal = claim.model_copy(update={"relation_type": "causally_upstream_of"})
    assert release_gate(phenotype, causal, evidence, mapping,
                        review.model_copy(update={"causal_supported": True}), steps)[0] == "rejected"
    mouse = claim.model_copy(update={"species": "mouse", "taxon_id": "NCBITaxon:10090"})
    scope, reasons = release_gate(phenotype, mouse, evidence, mapping, review, steps)
    assert scope == "cross_species" and "literature_grounded_inference" in reasons
    assert release_gate(phenotype, claim, evidence, mapping.model_copy(update={"go": None}),
                        review, steps)[0] == "unresolved"


@pytest.mark.parametrize("mutation", ["unread", "unstaged", "retracted", "fabricated", "wrong_offset"])
def test_each_supporting_quote_must_be_real_read_and_nonretracted(chain_case, mutation):
    phenotype, claim, evidence, documents, windows, _ = chain_case
    bridge_id = "synthetic:bridge"
    if mutation == "unread":
        windows = [w for w in windows if w["document_id"] != bridge_id]
    elif mutation == "unstaged":
        documents.pop(bridge_id)
    elif mutation == "retracted":
        documents[bridge_id] = documents[bridge_id].model_copy(update={"retracted": True})
    else:
        payload = claim.model_dump()
        citation = payload["reasoning_steps"][1]["citations"][0]
        citation.update({"quote": "Fabricated premise"} if mutation == "fabricated" else {"span_start": 4})
        claim = ClaimDraft.model_validate(payload)
    with pytest.raises(ValueError):
        anchor_reasoning(phenotype, claim, evidence, documents, windows)


@pytest.mark.parametrize("mutation", ["disconnected", "wrong_endpoint", "no_assumptions", "direct_label", "no_inference"])
def test_chain_contract_prevents_unsupported_or_mislabeled_paths(chain_case, mutation):
    _, claim, *_ = chain_case
    payload = claim.model_dump()
    steps = payload["reasoning_steps"]
    if mutation == "disconnected":
        steps[1]["source"] = "Unrelated condition"
    elif mutation == "wrong_endpoint":
        steps[-1]["target"] = "Some other cells"
    elif mutation == "no_assumptions":
        steps[1]["assumptions"] = []
    elif mutation == "direct_label":
        payload["evidence_mode"] = "direct"
    else:
        steps[1]["basis"], steps[1]["assumptions"] = "literature", []
    with pytest.raises(ValidationError):
        ClaimDraft.model_validate(payload)


def test_chain_must_reference_input_and_end_in_observed_cell_quote(chain_case):
    phenotype, claim, evidence, documents, windows, _ = chain_case
    payload = claim.model_dump()
    payload["reasoning_steps"][0]["source"] = "Different phenotype"
    with pytest.raises(ValueError, match="start_at_phenotype"):
        anchor_reasoning(phenotype, ClaimDraft.model_validate(payload), evidence, documents, windows)
    payload = claim.model_dump()
    payload["reasoning_steps"][-1]["citations"] = payload["reasoning_steps"][1]["citations"]
    with pytest.raises(ValueError, match="must_cite_cellular"):
        anchor_reasoning(phenotype, ClaimDraft.model_validate(payload), evidence, documents, windows)


def test_supporting_source_content_participates_in_claim_identity(chain_case):
    phenotype, claim, evidence, documents, windows, _ = chain_case
    before = anchor_reasoning(phenotype, claim, evidence, documents, windows)
    doc = documents["synthetic:bridge"]
    documents[doc.document_id] = doc.model_copy(update={"text": doc.text + "\nAdditional source context."})
    after = anchor_reasoning(phenotype, claim, evidence, documents, windows)
    assert claim_id(phenotype, claim, evidence, before) != claim_id(phenotype, claim, evidence, after)


async def test_real_root_eval_builds_and_exports_multidocument_chain(tmp_path, index, chain_case):
    phenotype, claim, _, documents, _, review = chain_case
    root = ScriptedModel(responses=[
        response("eval", {"code": DISPATCH + """
var state = await tools.getResearchState({});
var extracted = await dispatch("evidence-reasoner", {
  document_id:state.documents[0].document_id,
  supporting_document_ids:[state.documents[1].document_id]
});
var cid = extracted.claims[0].claim_id;
({id:cid, mode:extracted.claims[0].evidence_mode});
"""}),
        response("eval", {"code": """
var mapped = await dispatch("ontology-normalizer", {claim_id:cid});
var checked = await dispatch("evidence-reviewer", {claim_id:cid});
var state = await tools.getResearchState({});
({scope:checked.release_scope, mode:checked.evidence_mode, version:state.state_version});
"""}),
        finish(5),
    ])
    models = {
        "research": root,
        "reason": ScriptedModel(responses=[
            response("get_phenotype_context", {}),
            response("read_document", {"document_id": "synthetic:bridge"}),
            response("validate_reasoning", {"claims": [claim.model_dump()]}),
            response("Extraction", {"claims": [claim.model_dump()]}),
        ]),
        "normalize": ScriptedModel(responses=[response("read_evidence", {}), response("SubmitMapping", MAPPING)]),
        "review": ScriptedModel(responses=[response("Review", review.model_dump())]),
    }
    settings = Settings(workflow_mode="component_test", offline=True, llm_workers=3, term_workers=1)
    engine = AgentEngine(index, settings, models=models)
    runner = BatchRunner(engine=engine, index=index, settings=settings,
                         output=tmp_path / "run", local_documents=list(documents.values()))
    result = (await runner.run([phenotype]))[0]
    assert result.status == "completed", result.errors
    assert result.outcome == "conditional_only" and result.schema_version == "2.0"
    assert TermResult.model_validate_json(result.model_dump_json()).model_dump(mode="json") == result.model_dump(mode="json")
    inferred = read_jsonl(tmp_path / "run" / "inferred_edges.jsonl")
    assert len(inferred) == 1 and inferred[0]["evidence_mode"] == "inferred"
    assert inferred == read_jsonl(tmp_path / "run" / "conditional_edges.jsonl")
    assert not read_jsonl(tmp_path / "run" / "direct_edges.jsonl")
    assert not read_jsonl(tmp_path / "run" / "edges.jsonl")
    assert " -> " in inferred[0]["reasoning_chain"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["accepted_evidence_modes"] == {"direct": 0, "inferred": 1}
    audit = read_jsonl(tmp_path / "run" / result.coverage["audit_path"])
    reviewed = next(row for row in audit if row["role"] == "review" and "input" in row)
    assert {s["document_id"] for s in reviewed["input"]["reasoning_source_contexts"]} == set(documents)
    normalized = next(row for row in audit if row["role"] == "normalize" and "trace" in row)
    read_response = next(m for m in normalized["trace"] if m.get("name") == "read_evidence")
    assert "reasoning_source_contexts" in read_response["content"]
    root_trace = next(row["trace"] for row in audit if row["role"] == "research" and "trace" in row)
    assert documents["synthetic:bridge"].text not in json.dumps(root_trace)
    dispatches = [a["subagent"] for a in audit if a["role"] == "dispatch"]
    assert dispatches == ["evidence-reasoner", "ontology-normalizer", "evidence-reviewer"]
    assert {"get_phenotype_context", "read_document", "validate_reasoning"} <= set(models["reason"].observed_tools)
    resumed = (await runner.run([phenotype]))[0]
    assert resumed.coverage["audit_path"] == result.coverage["audit_path"]


@pytest.mark.parametrize("role", ["extract", "reason"])
async def test_evidence_specialists_cannot_publish_the_other_evidence_mode(
    tmp_path, chain_case, draft, document, role,
):
    phenotype, inferred, _, documents, _, _ = chain_case
    wrong_claim = inferred if role == "extract" else draft

    class WrongRoleEngine:
        async def extract(self, payload, audit, tools=()):
            return Extraction(claims=[wrong_claim])

        reason = extract

    session = ResearchSession(phenotype=phenotype, engine=WrongRoleEngine(), settings=Settings(workflow_mode="component_test", offline=True),
                              output=tmp_path, attempt=tmp_path / "attempt")
    for source in (documents.values() if role == "extract" else [document]):
        await session.stage(source)
    result = await getattr(session, role)({"document_id": document.document_id})
    assert not result["claims"] and not session.claims
    assert "this_subagent_requires_" in result["invalid_claims"][0]["error"]


async def test_missing_premise_returns_actionable_gap_to_orchestrator(tmp_path, chain_case):
    phenotype, _, _, documents, _, _ = chain_case
    gap = {"missing_link": "No source connects regional specificity to the general mechanism.",
           "suggested_queries": ["synthetic regional phenotype cellular injury"],
           "suggested_document_ids": []}

    class GapEngine:
        async def reason(self, payload, audit, tools=()):
            return Extraction(claims=[], rejection_reason="Missing supporting premise.", research_gaps=[gap])

    session = ResearchSession(phenotype=phenotype, engine=GapEngine(), settings=Settings(workflow_mode="component_test", offline=True),
                              output=tmp_path, attempt=tmp_path / "attempt")
    await session.stage(documents["synthetic:1"])
    result = await session.reason({"document_id": "synthetic:1"})
    assert result["research_gaps"] == [gap]
    assert session.state()["research_gaps"][0]["gaps"] == [gap]
    assert not session.claims
