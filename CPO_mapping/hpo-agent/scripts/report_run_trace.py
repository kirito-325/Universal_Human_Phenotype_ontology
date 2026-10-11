"""Describe actual live model/tool progress, including failed subtask responses."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from hpo_agent.storage import read_jsonl, save_json


def report(run):
    paths = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(paths) != 1:
        raise ValueError("expected one persisted attempt")
    path = paths[0]
    state = json.loads(path.read_text())
    events = read_jsonl(path.parent / "events.jsonl")
    audit = read_jsonl(path.parent / "audit.jsonl")
    invocations = {}
    usage = Counter()
    for event in events:
        invocation = event.get("invocation_id")
        if not invocation:
            continue
        row = invocations.setdefault(invocation, {"invocation_id": invocation, "role": event["role"],
                                                  "model_responses": 0, "tool_calls": Counter()})
        if event["event"] == "agent_started":
            row["started"] = event["time"]
        if event["event"] in {"agent_finished", "agent_failed"}:
            row["finished"] = event["time"]
            row["status"] = event["event"]
        if event["event"] != "agent_message" or event.get("message_type") != "ai":
            continue
        row["model_responses"] += 1
        row["tool_calls"].update(c["name"] for c in event.get("tool_calls", []))
        for key, value in (event.get("usage_metadata") or {}).items():
            if key in {"input_tokens", "output_tokens", "total_tokens"}:
                usage[key] += value
    for row in invocations.values():
        if row.get("started") and row.get("finished"):
            row["seconds"] = (datetime.fromisoformat(row["finished"]) - datetime.fromisoformat(row["started"])).total_seconds()
    result_path = path.parent.parent / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    active = [o for o in state["observations"] if o["claim_id"] not in state.get("retired_claims", {})]
    value = {
        "status": result["status"] if result else "unfinished",
        "first_event_utc": events[0]["time"], "last_event_utc": events[-1]["time"],
        "elapsed_seconds": (datetime.fromisoformat(events[-1]["time"]) - datetime.fromisoformat(events[0]["time"])).total_seconds(),
        "provider_reported_usage_all_recorded_responses": dict(usage),
        "usage_note": "Includes recorded failed subtask responses; unreturned requests/native scout calls may be absent. Not a bill.",
        "invocations": list(invocations.values()),
        "dispatch_counts": dict(Counter(a["subagent"] for a in audit if a.get("role") == "dispatch")),
        "knowledge_sources": dict(Counter(o.get("knowledge_source", "legacy") for o in active)),
        "reviewed_observations": len(active),
        "historical_session_observations": len(state["observations"]),
        "root_completion": (result or {}).get("coverage", {}).get("root_completion"),
    }
    save_json(run / "diagnostics" / "trace_report.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    report(parser.parse_args().run.resolve())
