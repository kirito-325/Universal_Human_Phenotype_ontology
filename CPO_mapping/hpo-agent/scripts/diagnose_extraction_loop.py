"""Inspect a persisted failed extraction trace without network or model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from hpo_agent.schemas import ClaimDraft, Document
from hpo_agent.storage import read_jsonl, save_json, save_jsonl
from hpo_agent.validation import anchor_claim


def diagnose(run: Path):
    sessions = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(sessions) != 1:
        raise ValueError("expected exactly one persisted attempt")
    path = sessions[0]
    state = json.loads(path.read_text())
    audit = read_jsonl(path.parent / "audit.jsonl")
    events = read_jsonl(path.parent / "events.jsonl")
    failed = [r for r in audit if r.get("role") == "extract" and "failed_trace" in r]
    if len(failed) != 1:
        raise ValueError("expected exactly one fully persisted failed extraction trace")
    trace = failed[0]["failed_trace"]
    documents = {d.document_id: d for artifact in state["coverage"]["document_artifacts"]
                 for d in [Document.model_validate_json((run / artifact["path"]).read_text())]}
    starts = [e for e in events if e.get("event") == "agent_started" and e.get("role") == "extract"]
    start = starts[0]
    end = next(e for e in events if e.get("event") == "agent_failed"
               and e.get("invocation_id") == start["invocation_id"])
    document = documents[start["document_id"]]

    calls, errors, validations, pending_calls = [], [], [], []
    for position, message in enumerate(trace):
        if message["type"] == "ai":
            pending_calls = []
            for call in message.get("tool_calls", []):
                row = {"trace_index": position, **call}
                calls.append(row)
                pending_calls.append(row)
        elif message["type"] == "tool" and pending_calls:
            # Failed traces omit ToolMessage IDs. This association follows message
            # order; validation requests in this run each contain only one call.
            call = pending_calls.pop(0)
            try:
                response = json.loads(message["content"])
            except (ValueError, TypeError):
                response = {"raw": message["content"]}
            call["response"] = response
            if "error" in response:
                errors.append(response["error"])
            for result in response.get("results", []):
                if "error" in result:
                    errors.append(result["error"])
            if call["name"] == "validate_extraction":
                for raw in call["args"]["claims"]:
                    draft = ClaimDraft.model_validate(raw)
                    try:
                        anchor_claim(draft, document, 0, len(document.text))
                        replayed = None
                    except ValueError as error:
                        replayed = str(error)
                    validations.append({
                        "trace_index": call["trace_index"], "cell_text": draft.cell_text,
                        "quality_text": draft.quality_text, "supporting_quote": draft.supporting_quote,
                        "quote_in_document": draft.supporting_quote in document.text,
                        "cell_in_quote": draft.cell_text in draft.supporting_quote,
                        "quote_start": document.text.find(draft.supporting_quote),
                        "replayed_exact_error": replayed, "recorded_response": response,
                    })

    repeated = Counter((c["name"], json.dumps(c["args"], sort_keys=True)) for c in calls)
    read_success = [c["response"] for c in calls if c["name"] == "read_document"
                    and "text" in c.get("response", {})]
    settings = json.loads(Path(state_path_settings(run)).read_text())
    summary = {
        "status": json.loads((run / "run_control.json").read_text())["status"],
        "audit_path": str(path.parent / "audit.jsonl"), "source_document_id": document.document_id,
        "source_title": document.title, "source_characters": len(document.text),
        "initial_window_contains_entire_document": settings["chunk_chars"] >= len(document.text),
        "first_extraction_started": start["time"], "first_extraction_failed": end["time"],
        "first_extraction_duration_seconds": (datetime.fromisoformat(end["time"])
                                               - datetime.fromisoformat(start["time"])).total_seconds(),
        "first_extraction_model_responses": sum(m["type"] == "ai" for m in trace),
        "first_extraction_tool_calls": dict(Counter(c["name"] for c in calls)),
        "tool_error_counts": dict(Counter(errors)),
        "empty_validation_calls": sum(c["name"] == "validate_extraction" and not c["args"]["claims"] for c in calls),
        "literal_validation_successes": sum(v["replayed_exact_error"] is None for v in validations),
        "replayed_specific_errors": dict(Counter(v["replayed_exact_error"] for v in validations
                                                 if v["replayed_exact_error"])),
        "distinct_successful_reread_windows": len({(r["start"], r["end"]) for r in read_success}),
        "most_repeated_calls": [{"tool": tool, "args": json.loads(args), "count": count}
                                for (tool, args), count in repeated.most_common(8)],
        "extraction_invocations": [{**s, "model_responses_observed": sum(
            e.get("invocation_id") == s["invocation_id"] and e.get("message_type") == "ai" for e in events
        )} for s in starts],
        "completed_dispatches": [e for e in audit if e.get("role") == "dispatch"],
        "searches": len(state["search_log"]), "documents": len(documents),
        "unique_pmids": len({d.pmid for d in documents.values()}),
        "retained_claims": len(state["claims"]), "observations": len(state["observations"]),
        "reasoning_invocations": sum(e.get("event") == "agent_started" and e.get("role") == "reason" for e in events),
        "normalization_invocations": sum(e.get("event") == "agent_started" and e.get("role") == "normalize" for e in events),
        "limits": [
            "First failed extraction has complete saved AI/tool messages; root and second extraction were interrupted.",
            "Second invocation tool arguments and full root decisions are not recoverable from saved live events.",
            "Offline replay verifies literal anchoring only, not scientific support or model counterfactual behavior.",
            "Actual billed token totals are unavailable for these unfinished/failed invocations.",
        ],
    }
    output = run / "diagnostics"
    save_json(output / "loop_analysis.json", summary)
    save_jsonl(output / "first_extraction_tools.jsonl", calls)
    save_jsonl(output / "validation_replay.jsonl", validations)
    print(json.dumps({"diagnostics": str(output), "summary": summary}, ensure_ascii=False, indent=2))


def state_path_settings(run):
    control = json.loads((run / "run_control.json").read_text())
    return Path(__file__).resolve().parents[1] / control["settings"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    diagnose(parser.parse_args().run.resolve())
