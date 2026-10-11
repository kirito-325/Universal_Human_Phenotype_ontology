"""Index a saved research run for human review; no model or network calls."""

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hpo_agent.storage import read_jsonl, save_json, save_jsonl


def local_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Shanghai")).isoformat() if value else None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def build(run, output):
    paths = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(paths) != 1:
        raise ValueError("expected exactly one saved attempt")
    path = paths[0]
    state = json.loads(path.read_text())
    audit = read_jsonl(path.parent / "audit.jsonl")
    events = read_jsonl(path.parent / "events.jsonl")
    result = json.loads((path.parent.parent / "result.json").read_text())
    event_index = defaultdict(deque)
    root_actions = []
    for line, event in enumerate(events, 1):
        event_index[canonical({k: v for k, v in event.items() if k != "time"})].append((line, event["time"]))
        if event.get("event") == "agent_message" and event.get("role") == "research":
            for call in event.get("tool_calls", []):
                if call["name"] == "eval":
                    root_actions.append({"number": len(root_actions) + 1, "time": local_time(event["time"]),
                                         "events_line": line, "code": call["args"]["code"]})
    dispatches = []
    last_time = None
    for line, row in enumerate(audit, 1):
        matches = event_index.get(canonical(row))
        if matches:
            _, last_time = matches.popleft()
        if row.get("role") == "dispatch":
            outcome = row["result"]
            dispatches.append({
                "number": len(dispatches) + 1, "time_near_return": local_time(last_time),
                "time_note": "Nearest preceding timestamped audit event, not an exact dispatch start.",
                "audit_line": line, "subagent": row["subagent"], "request": row["request"],
                "error": outcome.get("error"), "claims": [c["claim_id"] for c in outcome.get("claims", [])],
                "rejection_reason": outcome.get("rejection_reason"), "research_gaps": outcome.get("research_gaps", []),
                "cl_id": outcome.get("cl_id"), "go_id": outcome.get("go_id"),
                "release_scope": outcome.get("release_scope"),
            })
    queries = []
    retrieved = {}
    for number, row in enumerate(state["search_log"], 1):
        action = next((a for a in root_actions if row["query"] in a["code"]), None)
        retmax = None
        if action:
            tail = action["code"].split(row["query"], 1)[1]
            match = re.search(r"retmax\s*:\s*(\d+)", tail)
            retmax = int(match[1]) if match else None
        queries.append({"number": number, "query": row["query"], "count": row.get("count"),
                        "returned_pmids": row.get("pmids", []), "retmax": retmax,
                        "eval_start_time": action["time"] if action else None,
                        "root_action": action["number"] if action else None,
                        "warnings": row.get("warnings", []), "query_translation": row.get("query_translation")})
        retrieved.update({record["pmid"]: record for record in row.get("records", [])})
    papers = []
    for artifact in state["coverage"]["document_artifacts"]:
        doc = json.loads((run / artifact["path"]).read_text())
        analyses = [{"audit_line": i, "rejection_reason": a["output"].get("rejection_reason"),
                     "claim_count": len(a["output"]["claims"]), "research_gaps": a["output"].get("research_gaps", [])}
                    for i, a in enumerate(audit, 1) if a.get("role") == "extract" and "output" in a
                    and a["input"].get("document_id") == doc["document_id"]]
        papers.append({"document_id": doc["document_id"], "pmid": doc["pmid"],
                       "title": retrieved.get(doc["pmid"], {}).get("title", doc["title"]),
                       "source_kind": doc["source_kind"], "chars": len(doc["text"]),
                       "source_path": str(run / artifact["path"]),
                       "extraction_outputs": analyses,
                       "dispatch_numbers": [d["number"] for d in dispatches
                                             if d["request"].get("document_id") == doc["document_id"]],
                       "deferred_reason": state["deferred_documents"].get(doc["document_id"])})
    fetched = {p["pmid"] for p in papers}
    save_jsonl(output / "root_actions.jsonl", root_actions)
    save_jsonl(output / "searches.jsonl", queries)
    save_jsonl(output / "subagent_dispatches.jsonl", dispatches)
    save_jsonl(output / "paper_decisions.jsonl", papers)
    summary = {
        "run": str(run), "run_fingerprint": result["run_fingerprint"],
        "phenotype": result["phenotype"], "status": result["status"],
        "start": local_time(events[0]["time"]), "end": local_time(events[-1]["time"]),
        "root_evals": len(root_actions), "subagent_dispatches": len(dispatches),
        "model_subagent_invocations": sum(e.get("event") == "agent_started" and e["role"] != "research" for e in events),
        "model_responses": sum(e.get("event") == "agent_message" and e.get("message_type") == "ai" for e in events),
        "searches": len(queries), "retrieved_unique_pmids": len(retrieved), "fetched_papers": len(papers),
        "retrieved_not_fetched": [record for pmid, record in retrieved.items() if pmid not in fetched],
        "source_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in (path, path.parent / "events.jsonl", path.parent / "audit.jsonl",
                                   path.parent.parent / "result.json")],
        "notes": ["Retrospective index of recorded actions and submitted scientific justifications, not new research.",
                  "Query times refer to the enclosing eval dispatch; batch queries may finish several seconds later.",
                  "Source papers and archived original run were not altered."],
    }
    save_json(output / "source_index.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.run.resolve(), args.output.resolve())
