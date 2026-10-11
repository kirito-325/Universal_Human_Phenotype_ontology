"""Replay recorded failed submissions without a model, network or changing old runs."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from hpo_agent.assembly import assemble_assessment, assemble_hypotheses, read_passages
from hpo_agent.contracts import ResearchActionRequired, submission_error_feedback
from hpo_agent.normalization import normalize_assessment
from hpo_agent.schemas import (
    Document,
    HypothesisProposals,
    Phenotype,
    ScientificAssessment,
    Settings,
)
from hpo_agent.session import ResearchSession
from hpo_agent.storage import read_jsonl, save_json


def replay(run, output):
    attempts = list(run.glob("terms/*/*/attempt_*/session.json"))
    if len(attempts) != 1:
        raise ValueError("expected one persisted attempt")
    path = attempts[0]
    state = json.loads(path.read_text())
    audit = read_jsonl(path.with_name("audit.jsonl"))
    docs = {doc.document_id: doc for source in path.parent.glob("documents/*.json")
            if (doc := Document.model_validate_json(source.read_text()))}
    phenotype = Phenotype.model_validate(read_jsonl(run / "input/terms.jsonl")[0])
    settings = Settings.model_validate(json.loads((run / "input/settings.json").read_text()))
    invocations, staged, queries, unbound = {}, set(), [], {}
    for i, event in enumerate(audit):
        if event.get("role") == "source":
            if event.get("operation") == "search":
                queries.append(event["result"])
            staged.update(doc["document_id"] for doc in event.get("documents", []) if "document_id" in doc)
        invocation = event.get("invocation_id")
        if event.get("event") == "agent_started":
            invocations[invocation] = {"role": event["role"], "documents": set(staged), "queries": copy.deepcopy(queries)}
        if "input" in event:
            finished = next((later for later in audit[i + 1:] if later.get("event") == "agent_finished"
                             and later.get("role") == event["role"]), None)
            if finished:
                invocations[finished["invocation_id"]]["payload"] = event["input"]
        if event.get("event") == "agent_failed":
            unbound.setdefault(event["role"], []).append(invocation)
        if event.get("role") == "dispatch" and event.get("subagent") == "hypothesis-evaluator":
            pending = unbound.get("verify_hypothesis", [])
            if pending and "error" in event["result"]:
                invocations[pending.pop()]["hypothesis_id"] = event["request"]["hypothesis_id"]
            elif "error" in event["result"]:
                # Old empty-response exits were raised after streaming, without agent_failed.
                candidates = [item for item in invocations.values() if item["role"] == "verify_hypothesis"
                              and "payload" not in item and "hypothesis_id" not in item]
                if candidates:
                    candidates[-1]["hypothesis_id"] = event["request"]["hypothesis_id"]

    windows, calls, feedback_calls, rows = {}, {}, {}, []

    def window(key, start, end):
        doc = docs[key]
        return {"document_id": key, "start": start, "end": end, "text": doc.text[start:end],
                "passages": read_passages(doc, start, end)}

    for event in audit:
        invocation = event.get("invocation_id")
        if event.get("event") == "agent_message":
            if event.get("message_type") == "ai":
                for call in event.get("tool_calls", []):
                    calls[(invocation, call["id"])] = call
            elif event.get("message_type") == "tool":
                try:
                    content = json.loads(event["content"])
                except (ValueError, TypeError):
                    continue
                if not isinstance(content, dict):
                    continue
                if event.get("name") == "read_document" and "document_id" in content and "error" not in content:
                    windows.setdefault(invocation, []).append(window(content["document_id"], content["start"], content["end"]))
                if content.get("status") == "invalid_format":
                    feedback_calls[invocation] = event["tool_call_id"]
        if event.get("event") != "submission_validation_failed":
            continue
        call = calls[(invocation, feedback_calls[invocation])]
        meta = invocations[invocation]
        payload = meta.get("payload", {})
        hid = meta.get("hypothesis_id") or payload.get("hypothesis", {}).get("hypothesis_id")
        hypothesis = state["workflow"]["hypotheses"].get(hid)
        provided = list(windows.get(invocation, []))
        seeds = payload.get("seed_contexts", [])
        if seeds:
            provided.extend(window(w["document_id"], w["start"], w["end"]) for w in seeds)
        elif hypothesis:
            for seed in hypothesis["seed_evidence"]:
                start = max(0, seed["span_start"] - 400)
                end = min(len(docs[seed["document_id"]].text), start + min(settings.chunk_chars, seed["span_end"] - seed["span_start"] + 800))
                provided.append(window(seed["document_id"], start, end))
        raw = call["args"]
        row = {"submission": len(rows) + 1, "invocation_id": invocation, "tool_call_id": call["id"],
               "tool": call["name"], "hypothesis_id": hid, "original_issues": event["issues"],
               "normalizations": [], "raw_submission": raw}
        try:
            try:
                normalized, changes = normalize_assessment(raw) if call["name"] == "SubmitAssessment" else (raw, [])
            except ValueError as error:
                if not getattr(error, "type", None):
                    raise
                raise ValidationError.from_exception_data(call["name"], [{"type": error, "loc": (), "input": raw}]) from error
            row["normalizations"] = changes
            if call["name"] == "SubmitHypotheses":
                assemble_hypotheses(HypothesisProposals.model_validate(normalized), docs, provided)
                row["status"] = "locally_valid"
            else:
                if not hypothesis:
                    raise ValueError(f"could_not_bind_historical_hypothesis: {invocation} / {hid}")
                # Use only sources and searches available when this invocation started.
                session = ResearchSession(phenotype=phenotype, engine=SimpleNamespace(compact_submissions=True),
                    settings=settings, output=output.parent, attempt=output.parent / "replay_unused")
                session.documents = {key: docs[key] for key in meta["documents"]}
                session.search_log = meta["queries"]
                session.workflow.hypotheses[hid] = hypothesis
                selected = session.validation_context_sources(hid)
                for key in selected:
                    provided.append(window(key, 0, min(len(docs[key].text), settings.chunk_chars)))
                value = ScientificAssessment.model_validate(normalized)
                assessment = assemble_assessment(value, phenotype, hid, docs, provided,
                    original=hypothesis["draft"], reassessment_reason=payload.get("reassessment_reason", ""))
                row.update(status="locally_valid", prepared_validation_sources=selected,
                           host_revision_note=assessment.revision_note)
        except ResearchActionRequired as error:
            row.update(status=error.status, details=error.details)
        except ValidationError as error:
            row.update(json.loads(submission_error_feedback(SimpleNamespace(tool_name=call["name"], source=error), compact=True)))
        rows.append(row)
    if len(rows) != len(state["submission_validation_failures"]):
        raise ValueError("not all recorded failures were replayed")
    report = {"historical_run": str(run), "count": len(rows), "statuses": dict(Counter(row["status"] for row in rows)),
        "note": "Local structural/source preparation replay only; no new model inference or scientific reevaluation. Historical run untouched.",
        "submissions": rows}
    save_json(output, report)
    print(json.dumps({"count": report["count"], "statuses": report["statuses"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.run.resolve(), args.output.resolve())
