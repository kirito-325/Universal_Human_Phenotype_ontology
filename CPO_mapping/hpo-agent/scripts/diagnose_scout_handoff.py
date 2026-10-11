"""Offline diagnosis of a stopped scout: saved prerequisites and actual graph handoff.

Uses scripted model responses, never a model API or literature network connection.
The replay diagnoses framework behavior; it cannot recover unrecorded live scout messages.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from hpo_agent.agents import AgentEngine
from hpo_agent.ontology import OntologyIndex, digest_file
from hpo_agent.schemas import Document, Phenotype, Settings
from hpo_agent.session import ResearchSession
from hpo_agent.storage import read_jsonl, save_json


class DiagnosticModel(FakeMessagesListChatModel):
    model_name: str = "scout-offline-diagnostic"

    def _get_ls_params(self, **kwargs):
        return {"ls_provider": "openai", "ls_model_name": self.model_name, "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, number):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"diagnostic_{number}"}])


async def diagnose(args):
    run = args.run.resolve()
    paths = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one persisted attempt")
    path = paths[0]
    original_paths = [path, path.parent / "audit.jsonl", path.parent / "events.jsonl"]
    hashes = {str(p.relative_to(run)): digest_file(p) for p in original_paths}
    saved = json.loads(path.read_text())
    audit = read_jsonl(path.parent / "audit.jsonl")
    events = read_jsonl(path.parent / "events.jsonl")
    phenotype = Phenotype.model_validate(read_jsonl(args.input)[0])
    settings = Settings.model_validate_json(args.settings.read_text()).model_copy(update={"offline": True})
    documents = [Document.model_validate_json((run / x["path"]).read_text())
                 for x in saved["coverage"]["document_artifacts"]]
    pmids = sorted({d.pmid for d in documents if d.pmid})
    repeats = settings.max_repeated_tool_results + 1
    responses = [call("eval", {"code": """
var raw = await task({subagentType:"literature-scout", description:JSON.stringify({phase:"discovery"})});
var selected = typeof raw === "string" ? JSON.parse(raw) : raw;
({selection_is_json:Array.isArray(selected.pmids), count:selected.pmids.length});
"""}, 0)]
    responses += [call("get_research_state", {}, i + 1) for i in range(repeats)]
    responses += [
        call("ResearchSelection", {"pmids": pmids, "rationale": "Scripted diagnostic; not a scientific result."}, repeats + 1),
        call("ResearchCompletion", {
            "state_version": saved["state_version"], "stop_reason": "needs_curation",
            "summary": "Scripted handoff diagnostic only.", "limitations": ["Offline model double."],
        }, repeats + 2),
    ]
    model = DiagnosticModel(responses=responses)
    engine = AgentEngine(OntologyIndex(args.index), settings, models={"research": model})
    with tempfile.TemporaryDirectory(prefix="scout-handoff-diagnostic-") as temporary:
        session = ResearchSession(phenotype=phenotype, engine=engine, settings=settings,
                                  output=Path(temporary), attempt=Path(temporary) / "attempt")
        session.documents = {d.document_id: d for d in documents}
        session.analyzed = {d.document_id: [] for d in documents}
        session.read_windows = {d.document_id: set() for d in documents}
        session.search_log = copy.deepcopy(saved["search_log"])
        session.searches = len(session.search_log)
        session.version = saved["state_version"]
        session.coverage = copy.deepcopy(saved["coverage"])
        session.workflow.background = copy.deepcopy(saved["workflow"]["background"])
        session.workflow.plan = copy.deepcopy(saved["workflow"]["discovery_plan"])
        gaps = session.workflow.discovery_gaps()
        pending = session.workflow.pending()
        if not gaps:
            session.workflow.require_discovery()
        await asyncio.wait_for(engine.run(session), timeout=30)
        trace = next(x["trace"] for x in session.audit if x.get("role") == "research" and "trace" in x)
        returned = next(x["content"] for x in trace if x.get("name") == "eval" and x["type"] == "tool")
        if "selection_is_json: true" not in returned:
            raise ValueError(f"Native structured handoff failed: {returned}")
        native_events = [x for x in session.audit if x.get("role") == "literature-scout"]

    fetches = [x for x in audit if x.get("role") == "source" and x.get("operation") == "fetch"]
    queries = saved["search_log"]
    last_ai = next(x for x in reversed(events) if x.get("event") == "agent_message" and x.get("message_type") == "ai")
    value = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_attempt": str(path.parent.relative_to(run)),
        "first_event_utc": events[0]["time"], "last_root_event_utc": events[-1]["time"],
        "last_checkpoint_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "queries": len(queries), "filtered_queries": sum(q.get("filters") == ["hasabstract"] for q in queries),
        "retrieved_unique_pmids": len({p for q in queries for p in q.get("pmids", [])}),
        "source_kinds": dict(Counter(d.source_kind for d in documents)), "distinct_staged_pmids": len(pmids),
        "fetch_calls": len(fetches), "empty_fetch_calls": sum(not x["documents"] for x in fetches),
        "missing_abstract_pmids": saved["coverage"]["missing_abstract_pmids"],
        "last_completed_fetch": fetches[-1] if fetches else None,
        "last_root_tool_calls": last_ai.get("tool_calls"),
        "audit_records": len(audit), "root_event_records": len(events),
        "hypotheses": len(saved["workflow"]["hypotheses"]), "claims": len(saved["claims"]),
        "errors": saved["errors"], "discovery_gaps_replayed": gaps, "workflow_pending_replayed": pending,
        "offline_graph_probe": {
            "network_used": False, "model_api_used": False,
            "same_state_reads": repeats, "configured_repeat_limit": settings.max_repeated_tool_results,
            "native_no_progress_guard_triggered": False,
            "native_scout_handoff_returned": True, "root_eval_result": returned,
            "persisted_native_scout_events": len(native_events),
        },
        "conclusion": "Discovery prerequisites pass and the native structured handoff works in a scripted replay. "
                      "Native scout repeated state reads exceed the configured repeat limit without a guard; "
                      "its internal messages are not recorded. Live internal blocking cause is not recoverable from these logs.",
        "uncertainty": "The scripted repetition demonstrates a protection/observability gap, not proof that the live model repeated that action. "
                       "No complete live scout model/tool trace or request timing exists to distinguish waiting, validation retries or other repeated tools.",
        "original_sha256": hashes,
        "source_unchanged": all(digest_file(run / name) == value for name, value in hashes.items()),
    }
    if not value["source_unchanged"]:
        raise ValueError("Original run changed during diagnosis")
    save_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in {
        "last_completed_fetch", "last_root_tool_calls", "original_sha256"
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "input", "settings", "index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    asyncio.run(diagnose(parser.parse_args()))
