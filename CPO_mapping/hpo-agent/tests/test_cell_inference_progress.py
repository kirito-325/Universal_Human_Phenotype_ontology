"""Cell identity inference and bounded repeated actions, without paid model calls."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from hpo_agent.progress import EvidenceProgress
from hpo_agent.schemas import ClaimDraft, Settings
from hpo_agent.session import ResearchSession
from hpo_agent.validation import anchor_claim, anchor_reasoning, release_gate


def test_inferred_cell_need_not_appear_in_main_quote(phenotype, draft, document, mapping, review):
    quote = "SYNTHETIC tissue samples showed increased apoptosis."
    cell_quote = "SYNTHETIC tissue contains neurons with the measured pathway."
    doc = document.model_copy(update={"text": quote + "\n" + cell_quote})
    claim = ClaimDraft.model_validate({
        **draft.model_dump(), "evidence_mode": "inferred", "cell_evidence_mode": "inferred",
        "link_evidence_mode": "direct", "supporting_quote": quote,
        "relation_type": "mechanistically_compatible_with",
        "reasoning_steps": [{"source": phenotype.label, "target": draft.cellular_statement,
                             "scope": "cell_identity", "basis": "inference",
                             "relation": "candidate_cell_population", "explanation": "Synthetic tissue-to-cell inference.",
                             "assumptions": ["The measured abnormal tissue process may involve this cell population."],
                             "citations": [{"document_id": doc.document_id, "quote": quote},
                                           {"document_id": doc.document_id, "quote": cell_quote}]}],
    })
    evidence = anchor_claim(claim, doc, 0, len(doc.text))
    steps = anchor_reasoning(phenotype, claim, evidence, {doc.document_id: doc},
                             [{"document_id": doc.document_id, "start": 0, "end": len(doc.text)}])
    approved = review.model_copy(update={"inference_supported": True, "scope_transfer_supported": True,
                                         "cell_inference_supported": True})
    assert release_gate(phenotype, claim, evidence, mapping, approved, steps)[0] == "conditional"
    assert release_gate(phenotype, claim, evidence, mapping,
                        approved.model_copy(update={"cell_inference_supported": False}), steps)[0] == "rejected"
    with pytest.raises(ValueError, match="cell_not_in_quote"):
        anchor_claim(claim.model_copy(update={"cell_evidence_mode": "direct", "evidence_mode": "direct",
                                              "reasoning_steps": []}), doc, 0, len(doc.text))
    payload = claim.model_dump()
    payload["reasoning_steps"][0]["scope"] = "phenotype_link"
    with pytest.raises(ValidationError, match="cell_identity"):
        ClaimDraft.model_validate(payload)


def observe(guard, name, args, result, count):
    call_id = f"call{count}"
    guard.observe(AIMessage("", tool_calls=[{"id": call_id, "name": name, "args": args}]))
    guard.observe(ToolMessage(json.dumps(result), tool_call_id=call_id, name=name))


def test_repeated_invalid_candidate_stops_even_with_large_global_budget():
    guard = EvidenceProgress({}, Settings(workflow_mode="component_test", recursion_limit=600))
    for i in range(2):
        observe(guard, "validate_extraction", {"claims": []}, {"results": []}, i)
    with pytest.raises(ValueError, match="no_progress"):
        observe(guard, "validate_extraction", {"claims": []}, {"results": []}, 2)


def test_small_window_variations_do_not_bypass_progress_guard():
    guard = EvidenceProgress({"document_id": "doc", "window_start": 0, "window_end": 100},
                             Settings(workflow_mode="component_test", max_redundant_reads=3))
    for i in range(2):
        observe(guard, "read_document", {"start": i},
                {"document_id": "doc", "start": i, "end": i + 10, "text": "text"}, i)
    with pytest.raises(ValueError, match="previously supplied text"):
        observe(guard, "read_document", {"start": 3},
                {"document_id": "doc", "start": 3, "end": 13, "text": "text"}, 3)


def test_new_evidence_remains_available_without_a_small_global_tool_cap():
    guard = EvidenceProgress({}, Settings(workflow_mode="component_test", max_redundant_reads=3))
    for i in range(30):
        observe(guard, "read_document", {"start": i * 20},
                {"document_id": "doc", "start": i * 20, "end": i * 20 + 10, "text": "new"}, i)


def test_new_submission_allows_citation_repair_but_unchanged_submissions_still_stop():
    guard = EvidenceProgress({"document_id": "doc", "window_start": 0, "window_end": 100},
                             Settings(max_redundant_reads=2))
    result = {"document_id": "doc", "start": 0, "end": 10, "text": "quote"}
    observe(guard, "read_document", {"start": 0}, result, 0)
    proposal = {"hypotheses": [{"seed_evidence": [{"document_id": "doc", "quote": "wrong"}]}]}
    observe(guard, "SubmitHypotheses", proposal, {"error": "quote_not_verbatim"}, 1)
    observe(guard, "read_document", {"start": 0}, result, 2)
    assert guard.redundant_reads == 1
    observe(guard, "SubmitHypotheses", {**proposal, "focus": "retry"}, {"error": "quote_not_verbatim"}, 3)
    assert guard.redundant_reads == 1
    with pytest.raises(ValueError, match="identical SubmitHypotheses"):
        observe(guard, "SubmitHypotheses", {**proposal, "focus": "retry again"}, {"error": "quote_not_verbatim"}, 4)


def test_read_revisited_between_new_sources_is_not_an_identical_result_loop():
    guard = EvidenceProgress({}, Settings())
    for i in range(4):
        observe(guard, "read_document", {"document_id": "anchor"},
                {"document_id": "anchor", "start": 0, "end": 10, "text": "anchor"}, i * 2)
        observe(guard, "read_document", {"document_id": f"new{i}"},
                {"document_id": f"new{i}", "start": 0, "end": 10, "text": "new"}, i * 2 + 1)


def test_identical_consecutive_read_results_are_still_bounded():
    guard = EvidenceProgress({}, Settings())
    result = {"document_id": "doc", "start": 0, "end": 10, "text": "quote"}
    for i in range(2):
        observe(guard, "read_document", {"document_id": "doc"}, result, i)
    with pytest.raises(ValueError, match="identical read_document"):
        observe(guard, "read_document", {"document_id": "doc"}, result, 2)


async def test_same_failed_context_cannot_restart_by_changing_focus(tmp_path, phenotype, document):
    class StuckEngine:
        calls = 0

        async def extract(self, payload, audit, tools=()):
            self.calls += 1
            raise ValueError("no_progress: repeated tools")

    engine = StuckEngine()
    session = ResearchSession(phenotype=phenotype, engine=engine, settings=Settings(workflow_mode="component_test", offline=True),
                              output=tmp_path, attempt=tmp_path / "attempt")
    await session.stage(document)
    graph = next(s["runnable"] for s in session.subagents() if s["name"] == "evidence-extractor")
    for focus in ("first", "try again"):
        result = await graph.ainvoke({"messages": [AIMessage(json.dumps({"document_id": document.document_id,
                                                                       "focus": focus}))]})
        assert result["structured_response"]["retryable"] is False
    assert engine.calls == 1
    assert "already_failed" in result["structured_response"]["error"]
    another = document.model_copy(update={"document_id": "other"})
    await session.stage(another)
    await graph.ainvoke({"messages": [AIMessage(json.dumps({"document_id": document.document_id,
                                                          "supporting_document_ids": ["other"]}))]})
    assert engine.calls == 2
