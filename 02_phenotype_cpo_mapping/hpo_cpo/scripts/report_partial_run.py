"""Summarize persisted partial research and uncommitted mappings; never publish edges."""

import argparse
import json
from collections import Counter
from pathlib import Path

from hpo_agent.ontology import OntologyIndex
from hpo_agent.storage import read_jsonl, save_json, save_jsonl


def report(run, index_path):
    paths = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(paths) != 1:
        raise ValueError("expected one persisted attempt")
    path = paths[0]
    state = json.loads(path.read_text())
    events = read_jsonl(path.parent / "events.jsonl")
    documents = [json.loads((run / a["path"]).read_text()) for a in state["coverage"]["document_artifacts"]]
    by_invocation = {}
    proposals = []
    usage = Counter()
    for event in events:
        if event.get("event") != "agent_message":
            continue
        if event.get("message_type") == "ai":
            for key, value in (event.get("usage_metadata") or {}).items():
                if key in {"input_tokens", "output_tokens", "total_tokens"}:
                    usage[key] += value
        if event.get("role") != "normalize":
            continue
        for call in event.get("tool_calls", []):
            if call["name"] == "validate_mapping" and isinstance(call["args"].get("choice"), dict):
                proposal = {"time": event["time"], "invocation_id": event["invocation_id"],
                            "choice": call["args"]["choice"], "committed": False, "reviewed": False}
                proposals.append(proposal)
                by_invocation[event["invocation_id"]] = proposal
    index = OntologyIndex(index_path)
    for proposal in by_invocation.values():
        proposal["local_terms"] = [item.model_dump() for name in ("cl_id", "go_id")
                                   if (item := index.get(proposal["choice"][name]))]
    summary = {
        "status": json.loads((run / "run_control.json").read_text())["status"],
        "pipeline_completed": False,
        "first_event_utc": events[0]["time"], "last_event_utc": events[-1]["time"],
        "searches": len(state["search_log"]), "documents": len(documents),
        "unique_pmids": len({d["pmid"] for d in documents if d["pmid"]}),
        "source_kinds": dict(Counter(d["source_kind"] for d in documents)),
        "invocations": dict(Counter(e["role"] for e in events if e.get("event") == "agent_started")),
        "model_responses": dict(Counter(e["role"] for e in events if e.get("event") == "agent_message"
                                          and e.get("message_type") == "ai")),
        "candidate_claims": len(state["claims"]),
        "committed_mappings": sum(c["mapping"] is not None for c in state["claims"].values()),
        "reviewed_observations": len(state["observations"]),
        "mapping_pair_validation_counts": [{"cl_id": cl, "go_id": go, "count": n}
                                            for (cl, go), n in Counter((p["choice"]["cl_id"], p["choice"]["go_id"])
                                                                      for p in proposals).most_common()],
        "provider_reported_usage": dict(usage),
        "usage_note": "Recorded AI responses only; not a billing statement; failed transport/native scout calls may be absent.",
        "errors": state["errors"],
    }
    output = run / "diagnostics"
    save_json(output / "partial_report.json", summary)
    save_jsonl(output / "candidate_claims.jsonl", [{"claim_id": cid, **record} for cid, record in state["claims"].items()])
    save_jsonl(output / "mapping_proposals.jsonl", proposals)
    save_jsonl(output / "latest_uncommitted_mappings.jsonl", list(by_invocation.values()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.index)
