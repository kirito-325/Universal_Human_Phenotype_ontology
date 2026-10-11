"""Strict literature-first integration; synthetic sources, real root/eval/leaf graphs."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from test_runtime import DISPATCH, MAPPING, ScriptedModel, finish, response

from hpo_agent.agents import AgentEngine
from hpo_agent.background import HPOContextIndex, build_context_index
from hpo_agent.pipeline import BatchRunner
from hpo_agent.schemas import (
    ClaimDraft,
    DiscoveryPlan,
    HypothesisAssessment,
    HypothesisBatch,
    Settings,
)
from hpo_agent.session import HypothesisGenerationFailure, ResearchSession
from hpo_agent.storage import read_jsonl
from hpo_agent.workflow import HypothesisCitationError


@pytest.fixture
def context_index(tmp_path):
    hpo = tmp_path / "hp.obo"
    hpo.write_text('''format-version: 1.2
[Term]
id: HP:0001250
name: Synthetic target
def: "Synthetic phenotype definition." []
alt_id: HP:9000001
is_a: HP:0000707 ! Synthetic parent

[Term]
id: HP:0000707
name: Synthetic parent

[Term]
id: HP:0040280
name: Obligate

[Term]
id: HP:0040285
name: Excluded
''')
    annotations = tmp_path / "phenotype.hpoa"
    annotations.write_text(
        "#version: synthetic-test\n"
        "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
        "OMIM:1\tSynthetic direct disease\t\tHP:0001250\tPMID:100\tPCS\t\t1/2\t\t\tP\ttest\n"
        "OMIM:2\tSynthetic excluded disease\tNOT\tHP:0001250\tPMID:101\tPCS\t\t\t\t\tP\ttest\n"
        "OMIM:3\tSynthetic zero-frequency disease\t\tHP:0001250\tPMID:102\tPCS\t\t0/20\t\t\tP\ttest\n"
        "OMIM:4\tSynthetic broader disease\t\tHP:0000707\tPMID:103\tPCS\t\t80%\t\t\tP\ttest\n"
        "OMIM:5\tSynthetic obligate disease\t\tHP:0001250\tPMID:104\tPCS\t\tHP:0040280\t\t\tP\ttest\n"
        "OMIM:6\tSynthetic HPO-excluded disease\t\tHP:0001250\tPMID:105\tPCS\t\tHP:0040285\t\t\tP\ttest\n"
    )
    output = tmp_path / "hpo_context.sqlite"
    build_context_index(hpo, annotations, output)
    return HPOContextIndex(output)


@pytest.fixture
def discovery(document):
    documents = [document.model_copy(update={"document_id": f"synthetic:{i}", "pmid": str(100 + i),
                                             "text": f"route{i}. " + document.text}) for i in range(5)]
    kinds = ["phenotype", "disease", "pathology", "cellular", "molecular"]
    plan = DiscoveryPlan.model_validate({
        "rationale": "Synthetic broad recall across distinct facets.",
        "directions": [{"direction_id": f"d{i}", "label": f"Synthetic direction {i}", "kind": kinds[i],
                        "rationale": f"Explore synthetic source family {i}.", "queries": [f"route{i}"]} for i in range(5)],
    })
    batch = HypothesisBatch.model_validate({
        "diversity_rationale": "Five synthetic mechanisms inspired by five read source windows.",
        "hypotheses": [{"title": f"Synthetic hypothesis {i}", "direction_id": f"d{i}",
                        "cell_text": "neurons" if i == 0 else f"synthetic cell {i}",
                        "process_text": "apoptosis" if i == 0 else f"synthetic process {i}",
                        "direction": "increased", "mechanism": f"Distinct synthetic mechanism {i}.",
                        "why_distinct": f"Synthetic alternative {i} differs in mechanism and source family.",
                        "seed_evidence": [{"document_id": d.document_id, "quote": d.text}]}
                       for i, d in enumerate(documents)],
    })
    return documents, plan, batch


def test_context_keeps_exact_ancestor_and_negative_annotations_separate(context_index):
    found = context_index.lookup("HP:9000001", ancestor_depth=1)
    assert found["term"]["term_id"] == "HP:0001250"
    assert found["exact_positive_annotation_count"] == 2
    assert found["related_disease_count"] == 3
    assert found["excluded_annotation_count"] == 3
    assert [d["closest_match"] for d in found["diseases"]] == ["exact", "exact", "ancestor_background"]
    assert {d["disease_id"] for d in found["diseases"]} == {"OMIM:1", "OMIM:4", "OMIM:5"}
    assert found["diseases"][0]["annotations"][0]["frequency"] == "1/2"
    assert context_index.lookup("HP:0001250", limit=1)["next_offset"] == 1
    assert context_index.lookup("HP:0001250", disease_query="broader")["diseases"][0]["closest_match"] == "ancestor_background"
    with pytest.raises(ValueError, match="HP identifier"):
        context_index.lookup("MP:0000001")


def test_resource_budget_cannot_make_required_stages_impossible():
    with pytest.raises(ValidationError, match="max_searches"):
        Settings(max_searches=6)
    with pytest.raises(ValidationError, match="max_papers"):
        Settings(max_papers=4)


async def prepared_session(tmp_path, phenotype, context_index, discovery, engine=None, *, compact=False):
    if engine is not None and hasattr(engine, "compact_submissions"):
        engine.compact_submissions = compact
    documents, plan, _ = discovery
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=Settings(offline=True),
                              output=tmp_path, attempt=tmp_path / "attempt", hpo_context=context_index)
    for doc in documents:
        await session.stage(doc)
    tools = {t.name: t for t in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    result = await tools["register_discovery_plan"].ainvoke({"plan": plan.model_dump()})
    assert result["registered"]
    for i in range(5):
        await session.search_local(f"route{i}", direction_id=f"d{i}")
    return session


async def test_early_hypotheses_and_normalization_are_blocked(tmp_path, phenotype, context_index, discovery):
    session = ResearchSession(phenotype=phenotype, engine=None, settings=Settings(),
                              output=tmp_path, attempt=tmp_path / "attempt", hpo_context=context_index)
    with pytest.raises(ValueError, match="background_before_literature"):
        await session.search("test", 10, direction_id="d0")
    with pytest.raises(ValueError, match="broad_literature_recall"):
        await session.generate_hypotheses({})
    with pytest.raises(ValueError, match="hypothesis_batch_before_normalization"):
        await session.normalize({"claim_id": "anything"})
    assert session.pending()["workflow"] and not session.state()["ready_to_finish"]
    tools = {t.name: t for t in session.root_tools()}
    await tools["get_phenotype_background"].ainvoke({})
    session.workflow.register_plan(discovery[1])
    # A registered plan alone, or count probes alone, cannot unlock generation.
    with pytest.raises(ValueError, match="discovery_search"):
        session.workflow.require_discovery()


async def test_count_diversity_and_actually_read_seed_quotes_are_required(tmp_path, phenotype, context_index, discovery):
    session = await prepared_session(tmp_path, phenotype, context_index, discovery)
    documents, _, batch = discovery
    windows = [session.read(d.document_id) for d in documents]
    with pytest.raises(ValueError, match="at_least_5"):
        session.workflow.register_hypotheses(batch.model_copy(update={"hypotheses": batch.hypotheses[:4]}), windows)
    repeated = batch.model_copy(update={"hypotheses": [batch.hypotheses[0]] * 5})
    with pytest.raises(ValueError, match="must_differ"):
        session.workflow.register_hypotheses(repeated, windows)
    with pytest.raises(ValueError, match="read_the_discovery_corpus"):
        session.workflow.register_hypotheses(batch, windows[:1])
    ids = [h["hypothesis_id"] for h in session.workflow.register_hypotheses(batch, windows)]
    with pytest.raises(ValueError, match="run_support_literature_search"):
        session.workflow.require_verification(ids[0])
    await session.search_local("counter route0", hypothesis_id=ids[0], intent="challenge")
    with pytest.raises(ValueError, match="run_support_literature_search"):
        session.workflow.require_verification(ids[0])
    await session.search_local("route0", hypothesis_id=ids[0], intent="support")
    session.workflow.require_verification(ids[0])
    session.workflow.hypotheses[ids[0]].update(status="retained", claim_id="retained")
    with pytest.raises(ValueError, match="finish_verifying_all"):
        session.workflow.require_normalization("retained")
    for hid in ids[1:]:
        session.workflow.hypotheses[hid]["status"] = "rejected"
    session.workflow.require_normalization("retained")
    with pytest.raises(ValueError, match="only_retained"):
        session.workflow.require_normalization("not_retained")
    extension = batch.hypotheses[0].model_copy(update={"mechanism": "An additional synthetic mechanism discovered during verification."})
    session.workflow.register_hypotheses(HypothesisBatch(hypotheses=[extension], diversity_rationale="New source insight."),
                                         windows[:1], extend=True)
    assert len(session.workflow.hypotheses) == 6
    with pytest.raises(ValueError, match="finish_verifying_all"):
        session.workflow.require_normalization("retained")


async def test_hpo_background_is_required_before_starting_paid_research(tmp_path, phenotype, index):
    runner = BatchRunner(engine=SimpleNamespace(config={}), index=index, settings=Settings(), output=tmp_path)
    with pytest.raises(ValueError, match="HPO context index"):
        await runner.run([phenotype])


async def test_insufficient_hypotheses_can_report_limitations_without_unlocking_normalization(
    tmp_path, phenotype, context_index, discovery,
):
    async def hypothesize(payload, audit, tools=(), validator=None):
        return discovery[2].model_copy(update={"hypotheses": discovery[2].hypotheses[:3],
                                              "limitation": "Only three distinguishable candidates in this synthetic corpus."})

    session = await prepared_session(tmp_path, phenotype, context_index, discovery,
                                      engine=SimpleNamespace(hypothesize=hypothesize))
    output = await session.generate_hypotheses({})
    assert output["status"] == "needs_more_evidence" and not output["hypotheses"]
    assert not session.workflow.hypotheses and session.workflow.pending()
    with pytest.raises(ValueError, match="hypothesis_batch"):
        await session.normalize({"claim_id": "premature"})


async def test_validation_hits_must_be_read_before_retention(tmp_path, phenotype, context_index, discovery, draft):
    async def verify_hypothesis(payload, audit, tools=(), validator=None):
        return HypothesisAssessment(decision="retain", rationale="Synthetic unchecked assertion.",
                                    claim=draft.model_copy(update={"supporting_document_id": "synthetic:0"}))

    session = await prepared_session(tmp_path, phenotype, context_index, discovery,
                                      engine=SimpleNamespace(verify_hypothesis=verify_hypothesis))
    windows = [session.read(d.document_id) for d in discovery[0]]
    hypothesis = session.workflow.register_hypotheses(discovery[2], windows)[0]
    hid = hypothesis["hypothesis_id"]
    await session.search_local("route4", hypothesis_id=hid, intent="support")
    await session.search_local("counter route4", hypothesis_id=hid, intent="challenge")
    # The evaluator sees source 0 as a seed but never reads source 4 returned by the targeted search.
    with pytest.raises(ValueError, match="read_validation_sources_before_retention"):
        await session.evaluate_hypothesis({"hypothesis_id": hid})
    assert not session.claims and hypothesis["status"] == "proposed"


async def test_seed_offsets_are_host_owned_and_legacy_offsets_do_not_reject_exact_quotes(
    tmp_path, phenotype, context_index, discovery,
):
    session = await prepared_session(tmp_path, phenotype, context_index, discovery)
    documents, _, original = discovery
    payload = original.model_dump()
    for h, doc in zip(payload["hypotheses"], documents, strict=True):
        h["seed_evidence"] = [{"document_id": doc.document_id, "quote": doc.text[8:], "span_start": 1}]
    batch = HypothesisBatch.model_validate(payload)
    schema = HypothesisBatch.model_json_schema()
    assert set(schema["$defs"]["HypothesisSeedReference"]["properties"]) == {"document_id", "quote"}
    assert all("span_start" not in h.seed_evidence[0].model_dump() for h in batch.hypotheses)
    registered = session.workflow.register_hypotheses(batch, [session.read(d.document_id) for d in documents])
    assert all(h["seed_evidence"][0]["span_start"] == 8 for h in registered)
    assert all(documents[i].text[8:] == h["seed_evidence"][0]["quote"] for i, h in enumerate(registered))


@pytest.mark.parametrize("error_code", ["source_not_read", "quote_not_verbatim", "quote_not_in_read_windows"])
async def test_seed_diagnostics_identify_the_citation_without_committing_invalid_candidates(
    tmp_path, phenotype, context_index, discovery, error_code,
):
    session = await prepared_session(tmp_path, phenotype, context_index, discovery)
    documents, _, original = discovery
    payload = original.model_dump()
    windows = [session.read(d.document_id) for d in documents]
    if error_code == "source_not_read":
        payload["hypotheses"][0]["seed_evidence"][0]["document_id"] = "unstaged"
    elif error_code == "quote_not_verbatim":
        payload["hypotheses"][0]["seed_evidence"][0]["quote"] = "Fabricated source quotation."
    else:
        windows[0] = session.read(documents[0].document_id, 0, 5)
    with pytest.raises(HypothesisCitationError) as raised:
        session.workflow.register_hypotheses(HypothesisBatch.model_validate(payload), windows)
    issue, = raised.value.diagnostics
    assert issue["hypothesis_index"] == 1 and issue["citation_index"] == 1
    assert issue["document_id"] == payload["hypotheses"][0]["seed_evidence"][0]["document_id"]
    assert issue["error_code"] == error_code
    assert not session.workflow.hypotheses


async def test_real_generator_can_reread_after_a_failed_submission_and_commit_a_repair(
    tmp_path, phenotype, context_index, discovery, index,
):
    documents, _, batch = discovery
    invalid = batch.model_dump()
    invalid["hypotheses"][0]["seed_evidence"][0]["quote"] = "Wrong quote."
    model = ScriptedModel(responses=[
        *[response("read_document", {"document_id": d.document_id}) for d in documents],
        response("read_document", {"document_id": documents[0].document_id, "limit": 10}),
        response("SubmitHypotheses", invalid),
        response("read_document", {"document_id": documents[0].document_id, "limit": 15}),
        response("SubmitHypotheses", batch.model_dump()),
    ])
    engine = AgentEngine(index, Settings(offline=True, max_redundant_reads=2), models={"hypothesize": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine)
    session.settings = engine.settings
    result = await session.generate_hypotheses({})
    assert len(result["hypotheses"]) == 5
    assert len(session.workflow.hypotheses) == 5
    assert not session.failed_hypothesis_contexts
    errors = [a for a in session.audit if a.get("name") == "SubmitHypotheses" and a.get("message_type") == "tool"]
    assert "quote_not_verbatim" in errors[0]["content"]
    assert "already_read" in next(a["content"] for a in session.audit
                                  if a.get("name") == "read_document" and "already_read" in a.get("content", ""))


@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_failed_draft_is_preserved_and_same_context_has_one_explicit_repair(
    tmp_path, phenotype, context_index, discovery, index, repair_succeeds,
):
    documents, _, batch = discovery
    invalid = batch.model_dump()
    invalid["hypotheses"][0]["seed_evidence"][0]["quote"] = "Wrong quote."
    model = ScriptedModel(responses=[
        *[response("read_document", {"document_id": d.document_id}) for d in documents],
        *[response("SubmitHypotheses", invalid) for _ in range(3)],
    ])
    engine = AgentEngine(index, Settings(offline=True), models={"hypothesize": model})
    session = await prepared_session(tmp_path, phenotype, context_index, discovery, engine)
    adapter = next(s["runnable"] for s in session.subagents() if s["name"] == "hypothesis-generator")

    async def dispatch(payload):
        return (await adapter.ainvoke({"messages": [HumanMessage(json.dumps(payload))]}))["structured_response"]

    failed = await dispatch({})
    assert failed["retryable"] is False and failed["next_action"] == "repair_submission"
    assert failed["draft"]["hypotheses"] and failed["diagnostics"][0]["error_code"] == "quote_not_verbatim"
    assert not session.workflow.hypotheses
    blocked = await dispatch({"focus": "try again"})
    assert "hypothesis_context_already_failed" in blocked["error"]
    assert len([a for a in session.audit if a.get("event") == "agent_started"]) == 1
    assert session.state()["hypothesis_generation_feedback"][0]["candidate_count"] == 5
    snapshot = json.loads((tmp_path / "attempt/session.json").read_text())
    assert snapshot["failed_hypothesis_contexts"][failed["context_key"]]["draft"] == failed["draft"]

    model.responses = [response("SubmitHypotheses", batch.model_dump() if repair_succeeds else invalid)]
    model.i = 0
    repaired = await dispatch({"repair_submission": True})
    if repair_succeeds:
        assert len(repaired["hypotheses"]) == 5
        assert not session.failed_hypothesis_contexts
        assert all(error["resolved"] for error in session.errors)
    else:
        assert repaired["next_action"] == "stop_or_retrieve_new_sources"
        blocked_again = await dispatch({"repair_submission": True})
        assert "hypothesis_context_already_failed" in blocked_again["error"]
        assert len([a for a in session.audit if a.get("event") == "agent_started"]) == 2
        assert not session.workflow.hypotheses


async def test_stalled_generation_without_a_draft_only_restarts_with_new_sources(
    tmp_path, phenotype, context_index, discovery,
):
    calls = []

    async def hypothesize(payload, audit, tools=(), validator=None):
        calls.append(payload)
        raise ValueError("no_progress: redundant reading")

    session = await prepared_session(tmp_path, phenotype, context_index, discovery,
                                      SimpleNamespace(hypothesize=hypothesize))
    with pytest.raises(HypothesisGenerationFailure):
        await session.generate_hypotheses({})
    assert session.state()["hypothesis_generation_feedback"][0]["candidate_count"] == 0
    for payload in ({"focus": "retry"}, {"repair_submission": True}):
        with pytest.raises(HypothesisGenerationFailure, match="context_already_failed"):
            await session.generate_hypotheses(payload)
    assert len(calls) == 1
    await session.stage(discovery[0][0].model_copy(update={"document_id": "new_source", "pmid": "999"}))
    with pytest.raises(HypothesisGenerationFailure, match="no_progress"):
        await session.generate_hypotheses({})
    assert len(calls) == 2


@pytest.mark.parametrize("inferred_without_experiment", [False, True])
async def test_full_root_loop_verifies_five_and_only_normalizes_retained(
    tmp_path, phenotype, context_index, discovery, draft, review, index, inferred_without_experiment,
):
    documents, plan, batch = discovery
    if inferred_without_experiment:
        draft = ClaimDraft.model_validate({
            **draft.model_dump(), "evidence_mode": "inferred", "cell_evidence_mode": "inferred",
            "link_evidence_mode": "inferred", "supporting_quote": "", "bridge_scope": "mechanistically_compatible",
            "assertion": "model_inference", "evidence_type": "inference", "relation_type": "mechanistically_compatible_with",
            "reasoning_steps": [{"source": phenotype.label, "target": draft.cellular_statement,
                "scope": "cell_identity", "basis": "model_knowledge", "relation": "plausible_synthetic_mechanism",
                "explanation": "Synthetic unconfirmed inference for software validation.",
                "assumptions": ["Synthetic subtype applicability assumption."], "citations": []}],
        })
        review = review.model_copy(update={"verdict": "partial", "inference_supported": True,
                                           "cell_inference_supported": True, "scope_transfer_supported": True})
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
  if (i===0) var early = await dispatch('ontology-normalizer', {claim_id:assessment.claim_id});
}
for (var cid of kept) {
  await dispatch('ontology-normalizer', {claim_id:cid});
  await dispatch('evidence-reviewer', {claim_id:cid});
}
var state = await tools.getResearchState({});
({ready:state.ready_to_finish,version:state.state_version,early_error:early.error});
"""
    reader_messages = [response("read_document", {"document_id": d.document_id}) for d in documents]
    evaluations = [response("SubmitAssessment", {
        "decision": "retain" if i == 0 else "inconclusive" if i == 4 else "reject",
        "rationale": "Synthetic assessment after supporting searches.",
        "supporting_document_ids": [documents[i].document_id], "search_limitations": ["Synthetic local corpus only."],
        "claim": {**draft.model_dump(), "supporting_document_id": documents[0].document_id if draft.supporting_quote else ""} if i == 0 else None,
    }) for i in range(5)]
    models = {
        "research": ScriptedModel(responses=[response("eval", {"code": root_code}), finish(26)]),
        "hypothesize": ScriptedModel(responses=[*reader_messages, response("SubmitHypotheses", batch.model_dump())]),
        "verify_hypothesis": ScriptedModel(responses=evaluations),
        "normalize": ScriptedModel(responses=[response("SubmitMapping", MAPPING)]),
        "review": ScriptedModel(responses=[response("Review", review.model_dump())]),
    }
    settings = Settings(offline=True)
    engine = AgentEngine(index, settings, models=models)
    # Rich-record replay compatibility; compact production has a separate full-loop test.
    engine.compact_submissions = False
    runner = BatchRunner(engine=engine, index=index, settings=settings, output=tmp_path / "run",
                         local_documents=documents, hpo_context=context_index)
    result = (await runner.run([phenotype]))[0]
    assert result.status == "completed", result.errors
    ledger = read_jsonl(tmp_path / "run" / "hypotheses.jsonl")
    assert len(ledger) == 5
    assert [h["status"] for h in ledger] == ["retained", "rejected", "rejected", "rejected", "inconclusive"]
    assert all(len(h["assessment"]["query_ids"]) == 1 for h in ledger)
    assert len(result.observations) == 1 and result.observations[0].claim.hypothesis_id == ledger[0]["hypothesis_id"]
    if inferred_without_experiment:
        assert result.observations[0].release_scope == "conditional"
        assert read_jsonl(tmp_path / "run" / "relations.jsonl")[0]["experimental_validation"] == {
            "status": "not_recorded", "evidence": [],
        }
    audit = read_jsonl(tmp_path / "run" / result.coverage["audit_path"])
    normalized = [a for a in audit if a.get("role") == "normalize" and "output" in a]
    assert len(normalized) == 1
    blocked = [a for a in audit if a.get("role") == "dispatch" and "finish_verifying_all" in a.get("result", {}).get("error", "")]
    assert len(blocked) == 1
    assert not result.coverage["pending"]["workflow"]
    # The generator's saved input does not acquire later validation queries through shared mutable state.
    generated = next(a for a in audit if a.get("role") == "hypothesize" and "input" in a)
    assert len(generated["input"]["search_history"]) == 5
    proposed = next(a for a in audit if a.get("role") == "dispatch" and a["subagent"] == "hypothesis-generator")
    assert all(h["status"] == "proposed" and h["assessment"] is None for h in proposed["result"]["hypotheses"])
    assert len(result.search_log) == 10
    assert {query["intent"] for query in result.search_log} == {"discovery", "support"}
