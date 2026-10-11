"""Exercise the real root -> eval -> task -> leaf loops without paid model calls."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from hpo_agent.agents import AgentEngine
from hpo_agent.pipeline import BatchRunner
from hpo_agent.schemas import Extraction, ResearchCompletion, Settings
from hpo_agent.session import ResearchSession
from hpo_agent.storage import read_jsonl


class ScriptedModel(FakeMessagesListChatModel):
    model_name: str = "cell-agent-test"
    observed_tools: list = Field(default_factory=list)

    def _get_ls_params(self, **kwargs):
        return {"ls_provider": "openai", "ls_model_name": self.model_name, "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        self.observed_tools.extend(
            t.name if hasattr(t, "name") else t.get("function", t).get("name") for t in tools
        )
        return self


def response(name, args):
    return AIMessage("", tool_calls=[{"id": "call_" + name, "name": name, "args": args}])


def finish(version):
    return response("ResearchCompletion", {
        "state_version": version, "stop_reason": "sufficient_evidence",
        "summary": "Synthetic workflow finished.", "limitations": ["Software test only."],
    })


DISPATCH = """
var dispatch = async (type, payload) => {
  var result = await task({subagentType:type, description:JSON.stringify(payload)});
  return typeof result === "string" ? JSON.parse(result) : result;
};
"""

FIRST_PASS = DISPATCH + """
var state = await tools.getResearchState({});
var extracted = await dispatch("evidence-extractor", {document_id:state.documents[0].document_id});
var cid = extracted.claims[0].claim_id;
var normalized = await dispatch("ontology-normalizer", {claim_id:cid});
var reviewed = await dispatch("evidence-reviewer", {claim_id:cid});
({scope:reviewed.release_scope, reason:reviewed.review_reason});
"""

MAPPING = {"cl_id": "CL:0000540", "go_id": "GO:0006915", "cell_confidence": "high",
           "cl_label": "neuron", "go_label": "apoptotic process",
           "process_confidence": "high", "rationale": "Read local definitions."}


async def test_real_root_eval_iterates_after_unresolved_mapping_and_keeps_payloads_out_of_root(
    tmp_path, phenotype, document, draft, review, index,
):
    root = ScriptedModel(responses=[
        response("eval", {"code": FIRST_PASS}),
        response("eval", {"code": """
var corrected = await dispatch("ontology-normalizer", {claim_id:cid, focus:"Revisit GO using apoptosis"});
var afterCorrection = await tools.getResearchState({});
({pending:afterCorrection.pending.claims, scope:corrected.release_scope});
"""}),
        response("eval", {"code": """
var rereview = await dispatch("evidence-reviewer", {claim_id:cid});
var finalState = await tools.getResearchState({});
({scope:rereview.release_scope, state_version:finalState.state_version});
"""}),
        finish(6),
    ])
    unmapped = {**MAPPING, "go_id": "NONE", "go_label": "", "process_confidence": "unmapped"}
    models = {
        "research": root,
        "extract": ScriptedModel(responses=[
            response("validate_extraction", {"claims": [draft.model_dump()]}),
            response("Extraction", {"claims": [draft.model_dump()], "rejection_reason": ""}),
        ]),
        "normalize": ScriptedModel(responses=[
            response("read_evidence", {}),
            response("SubmitMapping", unmapped),
            response("get_ontology_term", {"term_id": "GO:0006915"}),
            response("SubmitMapping", MAPPING),
        ]),
        "review": ScriptedModel(responses=[
            response("Review", {**review.model_dump(), "mapping_supported": False}),
            response("Review", review.model_dump()),
        ]),
    }
    settings = Settings(workflow_mode="component_test", offline=True)
    engine = AgentEngine(index, settings, models=models)
    runner = BatchRunner(engine=engine, index=index, settings=settings,
                         output=tmp_path / "run", local_documents=[document])
    result = (await runner.run([phenotype]))[0]
    assert result.status == "completed", result.errors
    assert result.outcome == "projected" and len(result.observations) == 1
    assert result.observations[0].mapping.go.term_id == "GO:0006915"
    audit = read_jsonl(tmp_path / "run" / result.coverage["audit_path"])
    root_trace = next(a["trace"] for a in audit if a["role"] == "research" and "trace" in a)
    eval_outputs = [m["content"] for m in root_trace if m["type"] == "tool" and m["name"] == "eval"]
    assert len(eval_outputs) == 3
    assert "unresolved" in str(eval_outputs[0])
    assert result.observations[0].claim_id in str(eval_outputs[1])
    assert draft.supporting_quote not in json.dumps(root_trace, ensure_ascii=False)
    dispatches = [a for a in audit if a["role"] == "dispatch"]
    assert [a["subagent"] for a in dispatches] == [
        "evidence-extractor", "ontology-normalizer", "evidence-reviewer",
        "ontology-normalizer", "evidence-reviewer",
    ]
    assert dispatches[3]["result"]["mapping_revision"] == 2
    assert dispatches[3]["result"]["release_scope"] is None
    assert "eval" in root.observed_tools
    # Like upstream, tools remain bound; the actual research trajectory uses eval.
    assert not (set(root.observed_tools) & {"execute", "write_file", "edit_file"})
    assert {call["name"] for message in root_trace for call in message["tool_calls"]} == {
        "eval", "ResearchCompletion",
    }
    assert {"read_evidence", "get_ontology_term", "SubmitMapping"} <= set(models["normalize"].observed_tools)
    assert "validate_mapping" not in models["normalize"].observed_tools
    edges = read_jsonl(tmp_path / "run" / "edges.jsonl")
    assert edges == read_jsonl(tmp_path / "run" / "direct_edges.jsonl")
    assert edges[0]["evidence_mode"] == "direct" and edges[0]["reasoning_chain"] == ""
    assert not read_jsonl(tmp_path / "run" / "inferred_edges.jsonl")
    nodes = read_jsonl(tmp_path / "run" / "cell_abnormalities.jsonl")
    assert nodes[0]["cell_abnormality_id"] == edges[0]["cell_abnormality_id"]
    original_audit = result.coverage["audit_path"]
    assert (await runner.run([phenotype]))[0].coverage["audit_path"] == original_audit


async def test_real_eval_source_dispatch_and_subagent_budget(index, phenotype, document, draft, review, tmp_path):
    root = ScriptedModel(responses=[
        response("eval", {"code": DISPATCH + """
var search = await tools.pubmedSearch({term:"seizure[tiab]",retmax:1});
var fetched = await tools.fetchAbstracts({pmids:search.pmids});
var ex = await dispatch("evidence-extractor", {document_id:fetched.documents[0].document_id});
var nm = await dispatch("ontology-normalizer", {claim_id:ex.claims[0].claim_id});
var rv = await dispatch("evidence-reviewer", {claim_id:ex.claims[0].claim_id});
var budgetError = await dispatch("evidence-extractor", {document_id:fetched.documents[0].document_id});
var lastState = await tools.getResearchState({});
({scope:rv.release_scope, budget_error:budgetError.error, version:lastState.state_version});
"""}),
        finish(6),
    ])
    models = {
        "research": root,
        "extract": ScriptedModel(responses=[response("Extraction", {"claims": [draft.model_dump()], "rejection_reason": ""})]),
        "normalize": ScriptedModel(responses=[response("SubmitMapping", MAPPING)]),
        "review": ScriptedModel(responses=[response("Review", review.model_dump())]),
    }
    settings = Settings(workflow_mode="component_test", max_subagent_calls=3)
    engine = AgentEngine(index, settings, models=models)
    searches = []

    async def search(query, limit):
        searches.append(query)
        return {"query": query, "pmids": ["123"], "count": 1, "records": [{"pmid": "123"}]}

    async def fetch(pmids):
        return [document] if pmids else []

    session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                              output=tmp_path, attempt=tmp_path / "attempt",
                              source=SimpleNamespace(search=search, fetch_abstracts=fetch))
    await engine.run(session)
    assert searches == ["seizure[tiab]"]
    assert len(session.observations) == 1
    assert session.errors[-1]["error"] == "subagent_budget_exhausted"
    assert session.dispatches == 3


class StubEngine:
    config = {"models": {"all": "synthetic-test"}}

    def __init__(self, draft, mapping, review):
        self.draft, self.mapping, self.review_result = draft, mapping, review
        self.fail = False
        self.calls = 0

    async def run(self, session):
        self.calls += 1
        if self.fail:
            raise TimeoutError("synthetic")
        for key in list(session.documents):
            extracted = await session.extract({"document_id": key})
            for summary in extracted["claims"]:
                request = {"claim_id": summary["claim_id"]}
                await session.normalize(request)
                await session.review(request)
        return ResearchCompletion(state_version=session.version, stop_reason="sufficient_evidence",
                                  summary="Synthetic test.", limitations=[])

    async def extract(self, payload, audit, tools=()):
        return Extraction(claims=[self.draft], rejection_reason="")

    async def normalize(self, draft, audit, context=None):
        return self.mapping

    async def review(self, payload, audit):
        return self.review_result


async def test_failure_retry_input_change_and_corrupt_artifact_invalidate_resume(
    tmp_path, phenotype, document, draft, mapping, review, index,
):
    engine = StubEngine(draft, mapping, review)
    runner = BatchRunner(engine=engine, index=index, settings=Settings(workflow_mode="component_test", offline=True),
                         output=tmp_path / "run", local_documents=[document])
    engine.fail = True
    assert (await runner.run([phenotype]))[0].status == "processing_failed"
    engine.fail = False
    good = (await runner.run([phenotype]))[0]
    assert good.status == "completed" and engine.calls == 2
    (tmp_path / "run" / good.coverage["document_artifacts"][0]["path"]).write_text("corrupt")
    repaired = (await runner.run([phenotype]))[0]
    assert repaired.status == "completed" and engine.calls == 3
    changed = phenotype.model_copy(update={"definition": "Changed definition"})
    updated = (await runner.run([changed]))[0]
    assert updated.run_fingerprint != repaired.run_fingerprint and engine.calls == 4


async def test_quote_errors_and_unmapped_claims_remain_auditable(
    tmp_path, phenotype, document, draft, mapping, review, index,
):
    bad = draft.model_copy(update={"supporting_quote": "Invented claim with neurons"})
    engine = StubEngine(bad, mapping, review)
    runner = BatchRunner(engine=engine, index=index, settings=Settings(workflow_mode="component_test", offline=True),
                         output=tmp_path / "bad", local_documents=[document])
    result = (await runner.run([phenotype]))[0]
    assert result.coverage["invalid_claims"] and not result.observations
    engine = StubEngine(draft, mapping.model_copy(update={"go": None}), review)
    runner = BatchRunner(engine=engine, index=index, settings=Settings(workflow_mode="component_test", offline=True),
                         output=tmp_path / "unmapped", local_documents=[document])
    result = (await runner.run([phenotype]))[0]
    assert result.outcome == "normalization_unresolved"
    unresolved = read_jsonl(tmp_path / "unmapped" / "unresolved.jsonl")
    assert unresolved[0]["evidence"]["quote"] == draft.supporting_quote


async def test_root_cannot_finish_without_processing_documents(tmp_path, phenotype, document, index):
    model = ScriptedModel(responses=[finish(1)])
    engine = AgentEngine(index, Settings(workflow_mode="component_test", offline=True), models={r: model for r in
                                                              ("research", "extract", "normalize", "review")})
    runner = BatchRunner(engine=engine, index=index, settings=engine.settings,
                         output=tmp_path / "premature", local_documents=[document])
    result = (await runner.run([phenotype]))[0]
    assert result.status == "partial" and result.errors[0]["type"] == "IncompleteResearch"
    assert read_jsonl(tmp_path / "premature" / "edges.jsonl") == []


@pytest.mark.parametrize("operation", ["remap", "retire"])
async def test_current_mapping_requires_new_review_and_retirement_keeps_audit(
    operation, tmp_path, phenotype, document, draft, mapping, review,
):
    engine = StubEngine(draft, mapping, review)
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=Settings(workflow_mode="component_test", offline=True),
                              output=tmp_path, attempt=tmp_path / "attempt")
    await session.stage(document)
    await engine.run(session)
    cid = next(iter(session.claims))
    if operation == "remap":
        await session.normalize({"claim_id": cid})
        assert cid not in session.observations and cid in session.pending()["claims"]
        await session.review({"claim_id": cid})
        assert session.observations[cid].release_scope == "primary"
    else:
        retire = next(t for t in session.root_tools() if t.name == "retire_claim")
        await retire.ainvoke({"claim_id": cid, "reason": "Superseded by a corrected extraction."})
        assert cid in session.retired and cid in session.claims
        assert cid not in session.pending()["claims"]


async def test_no_documents_and_no_proposal_cannot_be_claimed_complete(tmp_path, phenotype, draft, mapping, review, index):
    runner = BatchRunner(engine=StubEngine(draft, mapping, review), index=index,
                         settings=Settings(workflow_mode="component_test", offline=True), output=tmp_path / "empty")
    result = (await runner.run([phenotype]))[0]
    assert result.status == "partial"
    assert not result.observations
