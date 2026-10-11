"""Verify the operating guide actually reaches the model through the assembled graph."""

from types import SimpleNamespace

from pydantic import Field
from test_runtime import ScriptedModel, finish, response

from hpo_agent import legacy_prompts as prompts
from hpo_agent.agents import AgentEngine
from hpo_agent.schemas import Settings
from hpo_agent.session import ResearchSession


class PromptRecordingModel(ScriptedModel):
    received_system_messages: list[str] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        texts = []
        for message in messages:
            if message.type != "system":
                continue
            if isinstance(message.content, str):
                texts.append(message.content)
            else:
                texts.extend(block if isinstance(block, str) else block.get("text", "") for block in message.content)
        self.received_system_messages.append("\n".join(texts))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def test_actual_root_system_message_contains_contract_and_active_limits(tmp_path, index, phenotype):
    model = PromptRecordingModel(responses=[finish(0)])
    settings = Settings(workflow_mode="component_test", min_hypotheses=7, min_discovery_directions=6,
                        min_discovery_papers=8, max_searches=40, offline=True)
    engine = AgentEngine(index, settings, models={"research": model})
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                              output=tmp_path, attempt=tmp_path / "attempt")
    await engine.run(session)
    actual = model.received_system_messages[0]
    assert prompts.ROOT in actual
    assert prompts.ROOT_INTERACTION in actual
    assert '"minimum_initial_hypotheses": 7' in actual
    assert '"minimum_discovery_directions": 6' in actual
    assert '"minimum_discovery_papers": 8' in actual
    assert '"model_concurrency_limit": 3' in actual
    assert '"offline": true' in actual
    assert "literature-scout" not in actual
    assert "ResearchSelection" not in model.observed_tools
    for step in ("STAGE 1: BACKGROUND", "STAGE 2: BROAD DISCOVERY", "STAGE 3: LITERATURE-INFORMED",
                 "STAGE 4: TARGETED SUPPORT", "STAGE 5: NORMALIZE"):
        assert step in actual
    assert "You own literature planning" in actual


async def test_root_eval_calls_literature_tools_without_a_scout_loop(tmp_path, index, phenotype, document):
    calls = []

    async def search(query, limit):
        calls.append(("search", query, limit))
        return {"query": query, "count": 1, "pmids": [document.pmid], "records": []}

    async def fetch_abstracts(pmids):
        calls.append(("abstracts", pmids))
        return [document.model_copy(update={"document_id": "PMID:" + document.pmid,
                                           "pmcid": "PMC123", "source_kind": "abstract"})]

    async def full_text(abstract):
        calls.append(("full_text", abstract.pmid))
        return abstract.model_copy(update={"document_id": "PMC123", "source_kind": "full_text"})

    model = PromptRecordingModel(responses=[
        response("eval", {"code": """
var found = await tools.pubmedSearch({term:'synthetic discovery query',retmax:10});
({pmids:found.pmids});
"""}),
        response("eval", {"code": """
var fetched = await tools.fetchAbstracts({pmids:found.pmids});
var full = await tools.fetchFullText({pmcids:[fetched.documents[0].pmcid]});
var state = await tools.getResearchState({});
({kind:full.documents[0].source_kind,document_count:state.documents.length});
"""}),
        finish(6),
    ])
    settings = Settings(workflow_mode="component_test")
    engine = AgentEngine(index, settings, models={"research": model})
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                              output=tmp_path, attempt=tmp_path / "attempt",
                              source=SimpleNamespace(search=search, fetch_abstracts=fetch_abstracts, full_text=full_text))
    completion = await engine.run(session)
    trace = next(a["trace"] for a in session.audit if a.get("role") == "research" and "trace" in a)
    returned = [m["content"] for m in trace if m.get("name") == "eval" and m["type"] == "tool"]
    assert len(returned) == 2
    assert "full_text" in returned[-1] and "document_count: 2" in returned[-1]
    assert calls == [("search", "synthetic discovery query", 10), ("abstracts", [document.pmid]),
                     ("abstracts", []), ("full_text", document.pmid)]
    assert len(model.received_system_messages) == 3
    assert all(prompts.ROOT in message for message in model.received_system_messages)
    assert session.dispatches == 0
    assert session.searches == 1
    assert completion.state_version == session.version


async def test_scout_is_not_an_available_task(tmp_path, index, phenotype):
    model = PromptRecordingModel(responses=[
        response("eval", {"code": """
var removed = await task({subagentType:'literature-scout',description:'{}'});
({removed});
"""}),
        finish(0),
    ])
    settings = Settings(workflow_mode="component_test", offline=True)
    engine = AgentEngine(index, settings, models={"research": model})
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                              output=tmp_path, attempt=tmp_path / "attempt")
    await engine.run(session)
    trace = next(a["trace"] for a in session.audit if a.get("role") == "research" and "trace" in a)
    returned = next(m["content"] for m in trace if m.get("name") == "eval" and m["type"] == "tool")
    assert "literature-scout" in returned
    assert "hypothesis-generator" in returned
    assert len(model.received_system_messages) == 2
