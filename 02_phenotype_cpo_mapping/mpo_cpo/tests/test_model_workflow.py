import copy
import json

import pytest
from helpers import ScriptedModel, assessment, judgments, response
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from test_model_background import model_index as model_index

from mpo_agent.agents import AgentEngine
from mpo_agent.coordinator import ModelCoordinator
from mpo_agent.pipeline import BatchRunner
from mpo_agent.results import MPResult
from mpo_agent.schemas import DiscoveryPlan, Document, MPInput, Settings
from mpo_agent.storage import read_jsonl

PLAN = {"rationale": "Synthetic model-scoped recall.",
        "directions": [{"direction_id": f"d{i}", "label": f"Synthetic route {i}", "kind": kind,
            "rationale": "Explore a distinct synthetic facet.", "queries": [f"route{i}"]}
            for i, kind in enumerate(("phenotype", "cellular", "model"))]}


def documents():
    def doc(key, pmid, text):
        return Document(document_id=key, pmid=pmid, title="SYNTHETIC SOFTWARE TEST",
            source_kind="local_text", source_url="urn:synthetic:" + key, phenotype_ids=["MP:9990001"], text=text)
    return {"MGI:1": [
        doc("annotation1", "101", "route0 route1 route2. Synthetic annotated model MGI:1 with a mouse phenotype."),
        doc("supplement1", "901", "support challenge. Synthetic supplementary model MGI:1 cell mechanism study.")],
        "MGI:2": [doc("annotation2", "201", "route0 route1 route2. Synthetic annotated model MGI:2 without a clear cellular explanation.")]}


def proposal(direction="increased"):
    return {"hypotheses": [{"title": "Synthetic model mechanism", "direction_id": "d0",
        "cell_text": "neurons", "process_text": "apoptosis", "direction": direction,
        "mechanism": "Synthetic literature-informed model mechanism.",
        "why_distinct": "One justified mechanism for this particular model.", "seed_evidence_refs": ["E1"]}],
        "diversity_rationale": "Model-specific candidate, no invented quota."}


ROOT_CODE = """
var dispatch = async (name,payload) => {
  var r=await task({subagentType:name,description:JSON.stringify(payload)});
  return typeof r==='string'?JSON.parse(r):r;
};
await tools.getPhenotypeBackground({});
var plan = PLAN_VALUE;
var results=[];
for (var mid of ['MGI:1','MGI:2']) {
  await tools.getModelBackground({model_id:mid});
  var registered=await tools.registerDiscoveryPlan({model_id:mid,plan});
  if (registered.error) throw new Error(JSON.stringify(registered));
  for (var i=0;i<3;i++) await tools.searchLocalDocuments({model_id:mid,query:'route'+i,direction_id:'d'+i});
  var pool=await dispatch('hypothesis-generator',{model_id:mid});
  if(pool.error) throw new Error(JSON.stringify(pool));
  for(var h of pool.hypotheses) {
    var hid=h.hypothesis_id;
    await tools.searchLocalDocuments({model_id:mid,query:mid==='MGI:1'?'support':'route0',hypothesis_id:hid,intent:'support'});
    var evaluated=await dispatch('hypothesis-evaluator',{model_id:mid,hypothesis_id:hid});
    if(evaluated.next_action==='complete_reasoning')
      evaluated=await dispatch('hypothesis-evaluator',{model_id:mid,hypothesis_id:hid,mode:'complete_reasoning',failure_id:evaluated.failure_id});
    if(evaluated.error) throw new Error(JSON.stringify(evaluated));
    if(evaluated.claim_id) {
      await dispatch('ontology-normalizer',{model_id:mid,claim_id:evaluated.claim_id});
      await dispatch('evidence-reviewer',{model_id:mid,claim_id:evaluated.claim_id});
    }
  }
  var state=await tools.getResearchState({model_id:mid});
  for(var id of state.research.pending.documents) await tools.deferDocument({model_id:mid,document_id:id,reason:'Unused synthetic source.'});
  var finished=await tools.finishModel({model_id:mid});
  if(finished.error) throw new Error(JSON.stringify(finished));
  results.push(finished);
}
({results,state:await tools.getResearchState({})});
""".replace("PLAN_VALUE", json.dumps(PLAN))


@pytest.mark.parametrize("second", ["inconclusive", "same", "opposite"])
@pytest.mark.parametrize("missing_reason", [False, True])
async def test_real_root_eval_task_model_research_and_aggregation(tmp_path, index, model_index, second, missing_reason):
    root = ScriptedModel(responses=[response("eval", {"code": ROOT_CODE}),
        response("ResearchCompletion", {"stop_reason": "sufficient_evidence", "summary": "Synthetic model union complete.",
                                        "limitations": ["Software test only."]})])
    generator_values = [
        response("read_document", {"document_id": "annotation1"}), response("SubmitHypotheses", proposal()),
        response("read_document", {"document_id": "annotation2"}),
        response("SubmitHypotheses", {"hypotheses": [], "diversity_rationale": "Sources read, no defensible mechanism.",
            "limitation": "Synthetic annotation describes the phenotype without an explainable mechanism."}
            if second == "inconclusive" else proposal("decreased" if second == "opposite" else "increased"))]
    first = assessment("E2")
    if missing_reason:
        del first["claim"]["cell"]["explanation"]
    evaluations = [response("SubmitAssessment", first)]
    if missing_reason:
        evaluations.append(response("SubmitAssessment", {"claim": {"cell": {"explanation": "Completed model-specific cell reasoning."}}}))
    other = assessment("E1")
    if second == "opposite":
        other["claim"]["abnormality"].update(direction="decreased", quality_text="decreased apoptosis")
    if second != "inconclusive":
        evaluations.append(response("SubmitAssessment", other))
    mapping = {"cl_id": "CL:0000540", "go_id": "GO:0006915", "cell_confidence": "high",
               "process_confidence": "high", "rationale": "Synthetic local ontology selection."}
    settings = Settings(offline=True, recursion_limit=100, timeout_seconds=30)
    engine = AgentEngine(index, settings, models={"research": root,
        "hypothesize": ScriptedModel(responses=generator_values),
        "verify_hypothesis": ScriptedModel(responses=evaluations),
        "normalize": ScriptedModel(responses=[response("SubmitMapping", mapping)]),
        "review": ScriptedModel(responses=[response("Review", judgments())])})
    output = tmp_path / "run"
    result, = await BatchRunner(engine=engine, index=index, context=model_index, settings=settings,
        output=output, local_documents=documents()).run([MPInput(phenotype_id="MP:9990001")])
    assert result.status == "completed", (result.errors, [(row.status, row.coverage.get("pending")) for row in result.models])
    assert result.models[0].status == "mechanisms_found"
    assert result.models[1].status == ("inconclusive" if second == "inconclusive" else "mechanisms_found")
    assert len(result.relations) == (1 if second == "inconclusive" else 2)
    assert all(row.observation.claim.species == "mouse" for row in result.relations)
    assert all(row.observation.release_scope == "conditional" for row in result.relations)
    assert len(result.mechanisms) == (2 if second == "opposite" else 1)
    assert result.mechanisms[0].model_ids == (["MGI:1", "MGI:2"] if second == "same" else ["MGI:1"])
    first_observation = result.relations[0].observation
    assert first_observation.reasoning_steps[0].evidence[0].pmid == "901"
    assert result.models[0].coverage["workflow"]["hypotheses"]
    if missing_reason:
        recovery, = result.models[0].coverage["assessment_recoveries"]
        assert recovery["mode"] == "complete_reasoning"
        assert all(error.get("resolved") for error in result.models[0].errors)
    assert len(read_jsonl(output / "models.jsonl")) == 2
    assert len(read_jsonl(output / "relations.jsonl")) == len(result.relations)
    assert MPResult.model_validate_json(result.model_dump_json()) == result
    corrupted = result.model_dump()
    corrupted["mechanisms"][0]["members"][0]["model_id"] = "MGI:unknown"
    with pytest.raises(ValidationError, match="mechanism_group_model_ids"):
        MPResult.model_validate(corrupted)
    traces = read_jsonl(output / result.models[0].coverage["audit_path"])
    evaluator = next(row for row in traces if row.get("role") == "verify_hypothesis" and "input" in row)
    assert evaluator["input"]["phenotype"]["model_context"]["model_id"] == "MGI:1"
    assert evaluator["input"]["validation_sources"]
    snapshot = json.loads((output / result.models[0].coverage["session_path"]).read_text())
    assert {query["intent"] for query in snapshot["search_log"]} == {"discovery", "support"}
    root_trace = read_jsonl(next(output.glob("terms/*/*/attempt_*/audit.jsonl")))
    assert any(row.get("role") == "model_dispatch" and row.get("model_id") == "MGI:2" for row in root_trace)


async def coordinator(tmp_path, index, model_index, **changes):
    settings = Settings(offline=True, **changes)
    generator = ScriptedModel(responses=[response("SubmitHypotheses", proposal())])
    engine = AgentEngine(index, settings, models={"hypothesize": generator, "verify_hypothesis": generator})
    session = ModelCoordinator(request=MPInput(phenotype_id="MP:9990001"), context=model_index,
        engine=engine, settings=settings, output=tmp_path, attempt=tmp_path / "attempt", local_documents=documents())
    return session, generator


async def test_background_and_discovery_block_paid_hypothesis_generation(tmp_path, index, model_index):
    session, generator = await coordinator(tmp_path, index, model_index)
    adapter = next(spec["runnable"] for spec in session.subagents() if spec["name"] == "hypothesis-generator")
    request = {"messages": [HumanMessage('{"model_id":"MGI:1"}')]}
    early = (await adapter.ainvoke(request))["structured_response"]
    assert early["status"] == "needs_sources" and generator.i == 0
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    early = (await adapter.ainvoke(request))["structured_response"]
    assert "complete_broad_literature_recall" in early["error"] and generator.i == 0


async def test_unstudied_models_are_coverage_gaps_not_inconclusive(tmp_path, index, model_index):
    session, _ = await coordinator(tmp_path, index, model_index, max_models=1)
    assert len(session.models) == 2 and session.records["MGI:2"]["status"] == "deferred"
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    blocked = await tools["finish_model"].ainvoke({"model_id": "MGI:1"})
    assert blocked["error"] == "model_research_incomplete"
    await tools["defer_model"].ainvoke({"model_id": "MGI:1", "reason": "No source access.", "needs_sources": True})
    assert session.state()["ready_to_finish"]
    assert session.summary_counts()["models"] == {"needs_sources": 1, "deferred": 1}
    assert session.observations() == []


async def test_unmapped_scientific_claim_is_not_misreported_as_no_mechanism(tmp_path, index, model_index):
    from types import SimpleNamespace
    session, _ = await coordinator(tmp_path, index, model_index)
    session.background_loaded = True
    session.records["MGI:1"]["status"] = "researching"
    session.sessions["MGI:1"] = SimpleNamespace(
        state=lambda: {"ready_to_finish": True}, errors=[],
        active_observations=lambda: [SimpleNamespace(claim_id="unmapped_claim", release_scope="unresolved")])
    result = await session.finish_model("MGI:1")
    assert result["error"] == "model_normalization_unresolved"
    assert session.records["MGI:1"]["status"] != "inconclusive"


async def test_model_task_and_source_ids_cannot_cross_model_boundaries(tmp_path, index, model_index):
    session, generator = await coordinator(tmp_path, index, model_index)
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    first, other = await session.model_session("MGI:1"), await session.model_session("MGI:2")
    provided, read_tools = other.research_workspace()
    wrong = await read_tools[0].ainvoke({"document_id": "annotation1"})
    assert "error" in wrong and not provided
    adapter = next(spec["runnable"] for spec in session.subagents() if spec["name"] == "hypothesis-evaluator")
    result = (await adapter.ainvoke({"messages": [HumanMessage('{"model_id":"MGI:2","hypothesis_id":"a_hypothesis_from_model_1"}')]}))["structured_response"]
    assert result["error"] == "unknown_hypothesis_id" and generator.i == 0
    assert first.documents.keys() != other.documents.keys()


async def test_a_shared_genotype_does_not_mix_local_documents_between_MP_terms(tmp_path, index, model_index):
    session, _ = await coordinator(tmp_path, index, model_index)
    foreign = documents()["MGI:1"][0].model_copy(update={
        "document_id": "another_term_source", "phenotype_ids": ["MP:9990002"]})
    session.local_documents["MGI:1"].append(foreign)
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    child = await session.model_session("MGI:1")
    assert "annotation1" in child.documents and "another_term_source" not in child.documents


async def test_annotation_pmids_and_searched_sources_form_the_model_source_pool(tmp_path, index, model_index):
    class Source:
        fetched = []

        async def fetch_abstracts(self, pmids):
            self.fetched.extend(pmids)
            return [Document(document_id="PMID:" + pmid, pmid=pmid, title="Synthetic source",
                source_kind="abstract", source_url="urn:synthetic:" + pmid,
                text="Synthetic mouse model source body.", phenotype_ids=["MP:9990001"]) for pmid in pmids]

        async def search(self, query, count):
            return {"query": query, "count": 1, "pmids": ["901"], "records": [{"pmid": "901", "title": "Synthetic supplement"}]}

    settings = Settings()
    engine = AgentEngine(index, settings, models={"hypothesize": ScriptedModel(responses=[response("SubmitHypotheses", proposal())])})
    source = Source()
    session = ModelCoordinator(request=MPInput(phenotype_id="MP:9990001"), context=model_index, engine=engine,
        settings=settings, output=tmp_path, attempt=tmp_path / "attempt", source=source)
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    annotated = await tools["fetch_model_references"].ainvoke({"model_id": "MGI:1"})
    assert source.fetched == ["101", "102", "103"]
    assert {row["pmid"] for row in annotated["documents"]} == {"101", "102", "103"}
    await tools["register_discovery_plan"].ainvoke({"model_id": "MGI:1", "plan": PLAN})
    await tools["pubmed_search"].ainvoke({"model_id": "MGI:1", "term": "supplement", "direction_id": "d0"})
    await tools["fetch_abstracts"].ainvoke({"model_id": "MGI:1", "pmids": ["901"]})
    child = session.sessions["MGI:1"]
    assert {doc.pmid for doc in child.documents.values()} == {"101", "102", "103", "901"}
    assert child.coverage["annotation_retrieval_attempts"][0]["pmids"] == ["101", "102", "103"]
    blocked = await tools["fetch_abstracts"].ainvoke({"model_id": "MGI:1", "pmids": ["201"]})
    assert blocked["error"] == "pmid_not_retrieved_or_supplied"
    assert "201" not in source.fetched


async def test_global_search_budget_failure_is_not_an_empty_search(tmp_path, index, model_index):
    session, _ = await coordinator(tmp_path, index, model_index, max_total_searches=1)
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    await tools["register_discovery_plan"].ainvoke({"model_id": "MGI:1", "plan": PLAN})
    first = await tools["search_local_documents"].ainvoke({"model_id": "MGI:1", "query": "route0", "direction_id": "d0"})
    assert first["count"] == 1
    blocked = await tools["search_local_documents"].ainvoke({"model_id": "MGI:1", "query": "route1", "direction_id": "d1"})
    assert blocked["status"] == "budget_exhausted" and "count" not in blocked


async def test_paginated_state_keeps_the_complete_model_registry(tmp_path, index, model_index):
    session, _ = await coordinator(tmp_path, index, model_index)
    model = session.models["MGI:1"]
    session.models = {f"MGI:{number}": model.model_copy(update={"model_id": f"MGI:{number}"}) for number in range(1, 26)}
    session.records = {key: {"status": "pending", "reason": ""} for key in session.models}
    session.background_loaded = True
    first = session.state(limit=5)
    assert len(first["models"]) == 5 and first["next_model_offset"] == 5
    assert first["pending"]["total_models"] == 25 and len(first["pending"]["models"]) == 5
    for key in list(session.models)[:5]:
        session.records[key] = {"status": "deferred", "reason": "Budget gap."}
    next_pending = session.state(limit=5)
    assert next_pending["pending"]["total_models"] == 20
    assert next_pending["pending"]["models"][0] == "MGI:6"
    assert not next_pending["ready_to_finish"]
    await session.checkpoint()
    snapshot = json.loads((session.attempt / "coordinator.json").read_text())
    assert len(snapshot["model_records"]) == 25


async def test_annotation_reading_is_required_and_empty_pool_requires_actual_research(tmp_path, index, model_index):
    session, _ = await coordinator(tmp_path, index, model_index)
    tools = {item.name: item for item in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    child = await session.model_session("MGI:1")
    child.workflow.register_plan(DiscoveryPlan.model_validate(PLAN))
    for i in range(3):
        await child.search_local("route" + str(i), direction_id="d" + str(i))
    from mpo_agent.schemas import HypothesisBatch
    empty = HypothesisBatch(hypotheses=[], diversity_rationale="Synthetic source analysis.", limitation="No justified mechanism.")
    with pytest.raises(ValueError, match="read_model_annotation"):
        child.workflow.register_hypotheses(empty, [child.read("supplement1")])
    assert not child.workflow.generation_complete
    child.workflow.register_hypotheses(empty, [child.read("annotation1")])
    assert child.workflow.generation_complete and not child.workflow.hypotheses


def test_completed_result_cannot_hide_missing_model_work(tmp_path, model_index):
    from mpo_agent.schemas import Phenotype
    term, models = model_index.all_models("MP:9990001")
    value = dict(run_fingerprint="test", phenotype=Phenotype(phenotype_id=term["term_id"], label=term["label"]),
        status="completed", models=[{"model": model, "status": "deferred", "reason": "Not studied.",
            "observations": [], "coverage": {}, "errors": []} for model in models],
        relations=[], mechanisms=[], completion=None, summary_counts={}, context_sha256=model_index.sha256)
    with pytest.raises(ValidationError, match="coverage_gaps"):
        MPResult.model_validate(value)
    changed = copy.deepcopy(value)
    changed["status"] = "partial"
    assert MPResult.model_validate(changed).models[0].status == "deferred"


async def test_alias_and_canonical_MP_cannot_start_duplicate_paid_research(tmp_path, index, model_index):
    from types import SimpleNamespace
    runner = BatchRunner(engine=SimpleNamespace(config={}), index=index, context=model_index,
                         settings=Settings(), output=tmp_path / "run")
    with pytest.raises(ValueError, match="unique_MP_terms"):
        await runner.run([MPInput(phenotype_id="MP:9990001"), MPInput(phenotype_id="MP:9990009")])
    assert not (tmp_path / "run").exists()
