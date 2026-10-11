"""Deep Agents research coordinator plus isolated extraction, normalization, review agents.

All acceptance decisions belong to the host pipeline. No model has shell access,
host filesystem access, a sandbox, or a tool that publishes final edge records.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from typing import Literal

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError
from pydantic import Field, create_model, model_validator

from . import legacy_prompts, prompts
from .assembly import assemble_completion, assemble_mapping, assemble_review
from .contracts import (
    SubmissionRepairError,
    apply_assessment_repair,
    contract_error,
    output_contract,
    submission_error_feedback,
)
from .normalization import normalize_assessment
from .ontology import OntologyIndex
from .progress import EvidenceProgress
from .schemas import (
    MODEL_CONTRACT,
    AssessmentRepair,
    Extraction,
    HypothesisAssessment,
    HypothesisBatch,
    HypothesisProposals,
    MappingChoice,
    OntologySelection,
    ResearchCompletion,
    Review,
    ReviewIssue,
    ScientificAssessment,
    ScientificCompletion,
    ScientificJudgment,
    ScientificReview,
    ScientificSupplement,
    Settings,
)
from .storage import to_json
from .validation import resolve_mapping


class ModelLimit(AgentMiddleware):
    def __init__(self, semaphore, *, min_interval=0, retries=4, connection_retries=4, retry_base=15):
        self.semaphore = semaphore
        self.min_interval = min_interval
        self.retries, self.retry_base = retries, retry_base
        self.connection_retries = connection_retries
        self.start_lock = asyncio.Lock()
        self.next_start = 0.0

    async def pace(self):
        async with self.start_lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.min_interval

    async def awrap_model_call(self, request, handler):
        rate_attempts = connection_attempts = 0
        while True:
            async with self.semaphore:
                await self.pace()
                try:
                    return await handler(request)
                except Exception as error:
                    status = getattr(error, "status_code", None)
                    connection_error = isinstance(error, (APIConnectionError, APITimeoutError))
                    if connection_error or status in {408, 500, 502, 503, 504}:
                        if connection_attempts >= self.connection_retries:
                            raise
                        connection_attempts += 1
                        attempt, budget = connection_attempts, self.connection_retries
                    elif status == 429:
                        if rate_attempts >= self.retries:
                            raise
                        rate_attempts += 1
                        attempt, budget = rate_attempts, self.retries
                    else:
                        raise
                    response = getattr(error, "response", None)
                    raw = response.headers.get("retry-after", "") if response is not None else ""
                    try:
                        retry_after = max(0, float(raw))
                    except (ValueError, TypeError):
                        retry_after = 0
                    delay = max(retry_after, min(120, self.retry_base * 2 ** (attempt - 1)))
                    async with self.start_lock:
                        self.next_start = max(self.next_start, time.monotonic() + delay)
                    logging.getLogger(__name__).warning(
                        "Model request %s; retry %s/%s after %.1fs",
                        status or type(error).__name__, attempt, budget, delay,
                    )


def model_configuration(*, include_components=False) -> dict:
    default = os.getenv("LLM_MODEL") or os.getenv("ROOT_MODEL", "")
    legacy_roles = {"research": "ROOT_MODEL", "extract": "SUBAGENT_MODEL", "reason": "SUBAGENT_MODEL",
                    "hypothesize": "SUBAGENT_MODEL", "verify_hypothesis": "JUDGE_MODEL",
                    "normalize": "SUBAGENT_MODEL", "review": "JUDGE_MODEL"}
    roles = ["research", "hypothesize", "verify_hypothesis", "normalize", "review"]
    if include_components:
        roles.extend(["extract", "reason"])
    return {
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "models": {role: os.getenv(f"LLM_{role.upper()}_MODEL") or os.getenv("LLM_MODEL")
                   or os.getenv(legacy_roles[role]) or default
                   for role in roles},
    }


class AgentEngine:
    def __init__(self, index: OntologyIndex, settings: Settings, *, models: dict | None = None):
        self.index = index
        self.settings = settings
        self.compact_submissions = settings.workflow_mode == "literature_first"
        self.config = model_configuration(include_components=settings.workflow_mode == "component_test")
        self.limit = ModelLimit(
            asyncio.Semaphore(settings.llm_workers), min_interval=settings.model_min_interval_seconds,
            retries=settings.model_rate_limit_retries, retry_base=settings.model_retry_base_seconds,
            connection_retries=settings.model_connection_retries,
        )
        if models is None:
            if not os.getenv("LLM_API_KEY") or not all(self.config["models"].values()):
                raise ValueError("Set LLM_API_KEY and LLM_MODEL (or all per-role models)")
            models = {
                role: ChatOpenAI(
                    model=name, base_url=self.config["base_url"], api_key=os.environ["LLM_API_KEY"],
                    use_responses_api=False, timeout=settings.model_request_timeout_seconds,
                    max_retries=0,
                ) for role, name in self.config["models"].items()
            }
        self.models = models
        # Same harness mechanism as deep-life-sci. Scope registrations to these model names.
        profile = HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        for model in models.values():
            name = getattr(model, "model_name", None)
            if name:
                register_harness_profile("openai:" + name, profile)

    async def invoke(self, graph, payload, schema, *, role: str, audit: list, assembler=None):
        payload = copy.deepcopy(payload)
        invocation_id = uuid.uuid4().hex[:12]
        result = {}
        progress = EvidenceProgress(payload, self.settings, submissions_only=role == "research")
        seen_messages = 0
        audit.append({"event": "agent_started", "role": role, "invocation_id": invocation_id,
                      "document_id": payload.get("document_id")})

        async def execute():
            nonlocal result, seen_messages
            async for state in graph.astream(
                {"messages": [{"role": "user", "content": to_json(payload)}]},
                config={"recursion_limit": self.settings.recursion_limit}, stream_mode="values",
            ):
                result = state
                messages = state.get("messages", [])
                for message in messages[seen_messages:]:
                    if message.type in {"ai", "tool"}:
                        audit.append({"event": "agent_message", "role": role, "invocation_id": invocation_id,
                                      "message_type": message.type, "content": message.content,
                                      "name": getattr(message, "name", None),
                                      "tool_call_id": getattr(message, "tool_call_id", None),
                                      "tool_calls": getattr(message, "tool_calls", []),
                                      "status": getattr(message, "status", None),
                                      "usage_metadata": getattr(message, "usage_metadata", None)})
                        if message.type == "tool" and isinstance(message.content, str):
                            try:
                                feedback = json.loads(message.content)
                            except ValueError:
                                feedback = {}
                            if isinstance(feedback, dict) and feedback.get("status") in {
                                "invalid_format", "invalid_reference", "needs_reasoning", "field_conflict", "scientific_conflict"
                            }:
                                audit.append({"event": "submission_validation_failed", "role": role,
                                              "invocation_id": invocation_id, **feedback})
                    if progress:
                        progress.observe(message)
                seen_messages = len(messages)
                last = messages[-1] if messages else None
                audit.append({"event": "graph_step", "role": role, "invocation_id": invocation_id,
                              "message_type": getattr(last, "type", None),
                              "tool_calls": [t["name"] for t in getattr(last, "tool_calls", [])]})

        try:
            await asyncio.wait_for(execute(), timeout=self.settings.timeout_seconds)
        except Exception as error:
            failed_message = getattr(error, "ai_message", None)
            if failed_message is not None:
                audit.append({"event": "agent_message", "role": role, "invocation_id": invocation_id,
                    "message_type": "ai", "content": failed_message.content, "name": failed_message.name,
                    "tool_calls": failed_message.tool_calls, "usage_metadata": failed_message.usage_metadata})
                audit.append({"event": "submission_validation_failed", "role": role, "invocation_id": invocation_id,
                    "status": error.status, "tool": error.tool, "issues": error.issues,
                    "next_action": {"needs_reasoning": "complete_missing_scientific_content",
                        "invalid_reference": "select_a_read_reference", "field_conflict": "resolve_conflicting_values",
                        "scientific_conflict": "explicit_scientific_reevaluation"}[error.status]})
            audit.append({"event": "agent_failed", "role": role, "invocation_id": invocation_id,
                          "error_type": type(error).__name__})
            audit.append({"role": role, "failed_trace": [
                {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", [])}
                for m in result.get("messages", []) if m.type in {"ai", "tool"}
            ]})
            raise
        structured = result.get("structured_response")
        trace = [
            {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", []),
             "name": getattr(m, "name", None)}
            for m in result.get("messages", []) if m.type in {"ai", "tool"}
        ]
        audit.append({"role": role, "trace": trace})
        if structured is None:
            raise ValueError(f"{role}: missing structured_response")
        submitted = schema.model_validate(structured)
        value = assembler(submitted) if assembler else submitted
        usage = [m.usage_metadata for m in result.get("messages", [])
                 if getattr(m, "usage_metadata", None)]
        audit.append({"role": role, "invocation_id": invocation_id, "input": payload, "output": value.model_dump(mode="json"),
                      **({"model_output": submitted.model_dump(mode="json"), "model_contract": MODEL_CONTRACT} if assembler else {}),
                      "usage": usage, "model": self.config["models"].get(role, "injected")})
        audit.append({"event": "agent_finished", "role": role, "invocation_id": invocation_id})
        return value

    def leaf(self, role, prompt, schema, tools=(), *, compact=False):
        def feedback(error):
            message = submission_error_feedback(error, compact=compact)
            data = json.loads(message)
            if compact and role == "verify_hypothesis" and data["status"] in {"needs_reasoning", "field_conflict", "invalid_reference", "scientific_conflict"}:
                ai = error.ai_message
                call = next(call for call in reversed(ai.tool_calls) if call["name"] == error.tool_name)
                failure = SubmissionRepairError(error.tool_name, data["issues"], copy.deepcopy(call["args"]), status=data["status"])
                failure.ai_message = ai
                raise failure from error
            return message
        return create_agent(
            model=self.models[role], tools=list(tools), system_prompt=prompt + output_contract(role, compact=compact),
            response_format=ToolStrategy(schema, handle_errors=feedback), middleware=[self.limit], name=role,
        )

    async def scientific_leaf(self, role, prompt, name, schema, payload, audit, tools=(), *, assembler, validator=None):
        payload = {**payload, "model_contract": MODEL_CONTRACT}
        normalized = set()
        def normalize_submission(cls, value):
            if issubclass(schema, (ScientificAssessment, ScientificSupplement)):
                result, changes = normalize_assessment(value)
                if changes:
                    signature = to_json(value)
                    if signature not in normalized:
                        normalized.add(signature)
                        audit.append({"event": "submission_normalized", "role": role, "tool": name,
                            "evidence_catalog_id": payload.get("evidence_catalog_id"),
                            "changes": changes, "raw_submission": copy.deepcopy(value), "normalized_submission": copy.deepcopy(result)})
                return result
            return value
        def valid_submission(self):
            value = assembler(self)
            if validator:
                validator(value)
            return self

        submission = create_model(name, __base__=schema,
            __validators__={"normalize_submission": model_validator(mode="before")(classmethod(normalize_submission)),
                            "valid_submission": model_validator(mode="after")(valid_submission)})
        return await self.invoke(self.leaf(role, prompt, submission, tools, compact=True),
                                 payload, submission, role=role, audit=audit, assembler=assembler)

    async def run(self, session):
        from .agent import build_agent

        task = "Run the literature-first cellular phenotype research workflow, iterating on feedback."
        if self.settings.run_mode == "smoke_test":
            task += (
                " This is a single-term end-to-end integration test, not exhaustive phenotype curation."
                " Preserve the full required sequence: HPO background, diverse discovery recall,"
                " read papers and generate at least the configured minimum number of distinct hypotheses,"
                " supporting literature searches and evaluation for every hypothesis, then normalize retained claims."
                " Smoke-test mode does NOT waive hypothesis count, breadth or verification prerequisites."
                " If no relation survives, report that honestly; preserve the distinction between model knowledge and sources."
            )
        compact = self.compact_submissions and session.workflow.enabled
        return await self.invoke(
            build_agent(session, self),
            {"task": task,
             "phenotype": session.phenotype.model_dump(),
             "model_contract": MODEL_CONTRACT if compact else "legacy_component",
             "initial_state": session.state()},
            ScientificCompletion if compact else ResearchCompletion, role="research", audit=session.audit,
            assembler=(lambda value: assemble_completion(value, session)) if compact else None,
        )

    async def extract(self, payload, audit, tools=()):
        return await self.invoke(self.leaf("extract", prompts.EXTRACT, Extraction, tools), payload,
                                 Extraction, role="extract", audit=audit)

    async def reason(self, payload, audit, tools=()):
        return await self.invoke(self.leaf("reason", prompts.REASON, Extraction, tools), payload,
                                 Extraction, role="reason", audit=audit)

    async def hypothesize(self, payload, audit, tools=(), validator=None, assembler=None):
        if payload.get("model_contract") == MODEL_CONTRACT and assembler is None:
            raise ValueError("compact_model_contract_requires_host_assembler")
        if assembler:
            def check_compact_pool(value):
                if len(value.hypotheses) < payload["minimum_hypotheses"] and value.limitation.strip():
                    return
                if validator:
                    validator(value)
            return await self.scientific_leaf("hypothesize", prompts.HYPOTHESIZE, "SubmitHypotheses",
                HypothesisProposals, payload, audit, tools, assembler=assembler, validator=check_compact_pool)
        class SubmitHypotheses(HypothesisBatch):
            """Validate and submit the complete diverse hypothesis pool; success ends this subtask."""

            @model_validator(mode="before")
            @classmethod
            def strict_seed_output(cls, value):
                if isinstance(value, dict) and isinstance(value.get("hypotheses"), list):
                    for hypothesis_index, hypothesis in enumerate(value["hypotheses"]):
                        seeds = hypothesis.get("seed_evidence", []) if isinstance(hypothesis, dict) else []
                        if not isinstance(seeds, list):
                            continue
                        for citation_index, seed in enumerate(seeds):
                            if isinstance(seed, dict) and "span_start" in seed:
                                raise contract_error("hypothesis.seed_fields",
                                    field=("hypotheses", hypothesis_index, "seed_evidence", citation_index, "span_start"))
                return value

            @model_validator(mode="after")
            def valid_pool(self):
                if len(self.hypotheses) < payload["minimum_hypotheses"] and self.limitation.strip():
                    return self
                if validator:
                    validator(self)
                return self

        return await self.invoke(self.leaf("hypothesize", legacy_prompts.HYPOTHESIZE, SubmitHypotheses, tools), payload,
                                 SubmitHypotheses, role="hypothesize", audit=audit)

    async def verify_hypothesis(self, payload, audit, tools=(), validator=None, assembler=None):
        if payload.get("model_contract") == MODEL_CONTRACT and assembler is None:
            raise ValueError("compact_model_contract_requires_host_assembler")
        if assembler:
            if payload.get("mode") == "complete_reasoning":
                return await self.scientific_leaf("verify_hypothesis", prompts.COMPLETE_REASONING, "SubmitAssessment",
                    ScientificSupplement, payload, audit, tools, assembler=assembler, validator=validator)
            return await self.scientific_leaf("verify_hypothesis", prompts.VERIFY_HYPOTHESIS, "SubmitAssessment",
                ScientificAssessment, payload, audit, tools, assembler=assembler, validator=validator)
        if payload.get("mode") == "repair":
            previous = payload["previous_submission"]
            last_draft = copy.deepcopy(previous)

            class SubmitAssessmentRepair(AssessmentRepair):
                """Submit field updates to the preserved assessment; the host revalidates it without regenerating its scientific decision."""

                @model_validator(mode="after")
                def valid_repair(self):
                    nonlocal last_draft
                    last_draft = apply_assessment_repair(previous, [update.model_dump() for update in self.updates])
                    value = HypothesisAssessment.model_validate(last_draft)
                    if validator:
                        validator(value)
                    return self

            try:
                await self.invoke(self.leaf("verify_hypothesis", legacy_prompts.REPAIR_ASSESSMENT, SubmitAssessmentRepair, tools),
                                  payload, SubmitAssessmentRepair, role="verify_hypothesis", audit=audit)
            except SubmissionRepairError as error:
                error.arguments = copy.deepcopy(last_draft)
                raise
            value = HypothesisAssessment.model_validate(last_draft)
            if validator:
                validator(value)
            return value

        class SubmitAssessment(HypothesisAssessment):
            """Submit retain/reject/inconclusive plus evidence assessment; only retain includes a final claim."""

            @model_validator(mode="after")
            def valid_assessment(self):
                if self.claim is not None and payload.get("phenotype"):
                    phenotype = payload["phenotype"]
                    self.claim = self.claim.bind_phenotype(phenotype["phenotype_id"], phenotype["label"])
                if validator:
                    validator(self)
                return self

        return await self.invoke(self.leaf("verify_hypothesis", legacy_prompts.VERIFY_HYPOTHESIS, SubmitAssessment, tools),
                                 payload, SubmitAssessment, role="verify_hypothesis", audit=audit)

    async def normalize(self, draft, audit, context=None):
        seen: set[str] = set()
        calls = 0

        def expose(items):
            seen.update(item.term_id for item in items)
            return [item.model_dump(mode="json") for item in items]

        @tool
        def search_ontology(query: str, ontology: Literal["CL", "GO"]) -> dict:
            """Recall non-obsolete CL/GO terms from local labels, synonyms, definitions."""
            nonlocal calls
            calls += 1
            if calls > self.settings.max_normalization_tool_calls:
                return {"error": "normalization_tool_budget_exhausted"}
            return {"candidates": expose(self.index.search(query, ontology))}

        @tool
        def get_ontology_term(term_id: str) -> dict:
            """Read a local term and its is_a parents; alternate IDs resolve to canonical IDs."""
            nonlocal calls
            calls += 1
            if calls > self.settings.max_normalization_tool_calls:
                return {"error": "normalization_tool_budget_exhausted"}
            item = self.index.get(term_id)
            if item is None:
                return {"error": "unknown_ontology_id"}
            parents = [p for key in item.parents if (p := self.index.get(key))]
            return {"terms": expose([item, *parents])}

        candidates = await asyncio.to_thread(lambda: {
            "CL": expose(self.index.search(draft.cell_text, "CL")),
            "GO": expose(self.index.search(draft.process_text or draft.quality_text, "GO")),
        })
        @tool
        def read_evidence() -> dict:
            """Read the proposal and optional source/review context; literature is not required for normalization."""
            return context or {"quote": draft.supporting_quote}

        index = self.index

        class SubmitMapping(MappingChoice):
            """Submit final CL/GO IDs and canonical labels. Valid submission ends normalization immediately."""

            cl_label: str = Field(..., description="Exact canonical CL label; empty only for NONE.")
            go_label: str = Field(..., description="Exact canonical GO label; empty only for NONE.")

            @model_validator(mode="after")
            def legal_identifiers_and_labels(self):
                for term_id, label in ((self.cl_id, self.cl_label), (self.go_id, self.go_label)):
                    if term_id not in {"", "NONE"} and not label:
                        raise contract_error("mapping.label", field=("cl_label" if term_id == self.cl_id else "go_label",))
                resolve_mapping(self, index, seen)
                return self

        payload = {"claim": draft.model_dump(), "initial_candidates": candidates,
                   "feedback": (context or {}).get("previous_attempt"),
                   "focus": (context or {}).get("focus", "")}
        compact = (context or {}).get("model_contract") == MODEL_CONTRACT
        if compact:
            payload["model_contract"] = MODEL_CONTRACT
            choice = await self.scientific_leaf("normalize", prompts.NORMALIZE, "SubmitMapping", OntologySelection,
                payload, audit, [search_ontology, get_ontology_term, read_evidence],
                assembler=lambda value: assemble_mapping(value, self.index, seen),
                validator=lambda value: resolve_mapping(value, self.index, seen))
            return resolve_mapping(choice, self.index, seen).model_copy(update={"method": "llm_id_selection_host_canonical_labels"})
        choice = await self.invoke(
            self.leaf("normalize", legacy_prompts.NORMALIZE, SubmitMapping,
                      [search_ontology, get_ontology_term, read_evidence]),
            payload, SubmitMapping, role="normalize", audit=audit,
        )
        return resolve_mapping(choice, self.index, seen)

    async def review(self, payload, audit):
        if payload.get("model_contract") == MODEL_CONTRACT:
            has_experiments = bool(payload.get("experimental_validation", {}).get("evidence"))
            schema = (create_model("ExperimentalScientificReview", __base__=ScientificReview,
                experimental_evidence=(ScientificJudgment, Field(...))) if has_experiments else ScientificReview)
            return await self.scientific_leaf("review", prompts.REVIEW, "Review", schema, payload, audit,
                assembler=lambda value: assemble_review(value, has_experiments))
        submission = create_model("Review", __base__=Review,
            blocking_issues=(list[ReviewIssue], Field(..., description="Actual factual/logical defects; [] for missing experimental proof alone.")),
            experimental_evidence_supported=(bool, Field(..., description="Recorded experiment scope/interpretation is faithful; true when no experiment is recorded.")))
        value = await self.invoke(self.leaf("review", legacy_prompts.REVIEW, submission), payload,
                                  Review, role="review", audit=audit)
        return Review.model_validate(value.model_dump())
