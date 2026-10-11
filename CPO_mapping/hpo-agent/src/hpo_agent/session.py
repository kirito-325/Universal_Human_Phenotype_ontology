"""Host state and capabilities for a root-controlled research loop.

This module never chooses the next research step. Tools stage immutable sources;
registered subagent adapters hydrate trusted context, run their own model loops,
validate responses, and return compact manifests to the root's JavaScript heap.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from .assembly import assemble_assessment, assemble_hypotheses, model_window, read_passages
from .contracts import ResearchActionRequired, SubmissionRepairError, contract_error
from .literature import SourceError
from .normalization import complete_scientific_fields, normalize_assessment
from .references import EvidenceWindows
from .schemas import (
    MODEL_CONTRACT,
    SCHEMA_VERSION,
    ClaimDraft,
    DiscoveryPlan,
    Document,
    ExperimentalValidation,
    Extraction,
    Observation,
)
from .storage import fingerprint, save_json, save_jsonl, to_json
from .validation import (
    anchor_claim,
    anchor_experiments,
    anchor_reasoning,
    claim_id,
    main_quote_window,
    release_gate,
)
from .workflow import ResearchWorkflow


class HypothesisGenerationFailure(ValueError):
    def __init__(self, message, details):
        self.details = details
        super().__init__(message)


class AssessmentSubmissionFailure(ValueError):
    def __init__(self, message, details):
        self.details = details
        super().__init__(message)


class ResearchSession:
    def __init__(self, *, phenotype, engine, settings, attempt: Path, output: Path,
                 source=None, audit=None, hpo_context=None):
        self.phenotype, self.engine, self.settings = phenotype, engine, settings
        self.attempt, self.output, self.source = attempt, output, source
        self.audit = audit if audit is not None else []
        self.documents: dict[str, Document] = {}
        self.claims: dict[str, dict] = {}
        self.observations: dict[str, Observation] = {}
        self.search_log, self.errors = [], []
        self.version = self.searches = self.full_texts = self.dispatches = 0
        self.known_pmids = set(phenotype.seed_pmids)
        self.lock = asyncio.Lock()
        self.checkpoint_lock = asyncio.Lock()
        self.claim_locks: dict[str, asyncio.Lock] = {}
        self.read_windows: dict[str, set[tuple[int, int]]] = {}
        self.analyzed: dict[str, list[tuple[int, int]]] = {}
        self.deferred: dict[str, str] = {}
        self.retired: dict[str, str] = {}
        self.failed_evidence_contexts: set[str] = set()
        self.failed_hypothesis_contexts: dict[str, dict] = {}
        self.failed_assessments: dict[str, dict] = {}
        self.hpo_context = hpo_context
        self.workflow = ResearchWorkflow(self)
        self.hypothesis_generation_lock = asyncio.Lock()
        self.hypothesis_locks: dict[str, asyncio.Lock] = {}
        self.coverage = {
            "document_artifacts": [], "invalid_claims": [], "no_claim_windows": [],
            "missing_abstract_pmids": [], "paper_limit_omitted": [],
            "research_gaps": [],
        }

    def touch(self):
        self.version += 1

    @property
    def compact_submissions(self):
        return self.workflow.enabled and bool(getattr(self.engine, "compact_submissions", False))

    def summary_counts(self):
        observations = self.active_observations()
        return {"searches": len(self.search_log), "documents": len(self.documents),
                "hypotheses": dict(Counter(h["status"] for h in self.workflow.hypotheses.values())),
                "release_scopes": dict(Counter(o.release_scope for o in observations)),
                "experimental_entries": sum(len(o.experimental_validation.evidence) for o in observations)}

    def active_observations(self):
        return [value for key, value in self.observations.items() if key not in self.retired]

    def document_summary(self, document_id):
        doc = self.documents[document_id]
        return {"document_id": document_id, "title": doc.title, "pmid": doc.pmid, "pmcid": doc.pmcid,
                "source_kind": doc.source_kind, "chars": len(doc.text), "retracted": doc.retracted,
                "sections": [s.model_dump() for s in doc.sections],
                "retrieval_notes": doc.retrieval_notes}

    async def stage(self, doc: Document):
        async with self.lock:
            existing = self.documents.get(doc.document_id)
            if existing and existing.sha256 == doc.sha256:
                return self.document_summary(doc.document_id)
            paper_key = "PMID:" + doc.pmid if doc.pmid else doc.document_id
            paper_keys = {"PMID:" + d.pmid if d.pmid else d.document_id for d in self.documents.values()}
            if paper_key not in paper_keys and len(paper_keys) >= self.settings.max_papers:
                self.coverage["paper_limit_omitted"].append(doc.document_id)
                self.touch()
                return {"error": "paper_budget_exhausted", "document_id": doc.document_id}
            locator = doc.document_id
            if existing:
                locator += "@" + doc.sha256[:12]
                doc = doc.model_copy(update={"document_id": locator})
            self.documents[locator] = doc
            self.read_windows[locator] = set()
            self.analyzed[locator] = []
            path = self.attempt / "documents" / (fingerprint(doc.model_dump()) + ".json")
            await asyncio.to_thread(save_json, path, doc)
            self.coverage["document_artifacts"].append({
                "document_id": locator, "text_sha256": doc.sha256,
                "path": str(path.relative_to(self.output)),
            })
            self.touch()
            return self.document_summary(locator)

    async def checkpoint(self):
        async with self.checkpoint_lock:
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
                "workflow": copy.deepcopy(self.workflow.snapshot()),
                "failed_evidence_contexts": sorted(self.failed_evidence_contexts),
                "failed_hypothesis_contexts": copy.deepcopy(self.failed_hypothesis_contexts),
                "failed_assessments": copy.deepcopy(self.failed_assessments),
                "state_version": self.version,
                "claims": {key: {"draft": value["draft"].model_dump(),
                                 "evidence": value["evidence"].model_dump() if value["evidence"] else None,
                                 "reasoning_steps": [s.model_dump() for s in value["reasoning_steps"]],
                                 "experimental_evidence": [e.model_dump() for e in value.get("experimental_evidence", [])],
                                 "window": [value["start"], value["end"]],
                                 "mapping": value["mapping"].model_dump() if value["mapping"] else None,
                                 "mapping_revision": value["revision"]}
                           for key, value in self.claims.items()},
                "observations": [o.model_dump() for o in self.observations.values()],
                "retired_claims": dict(self.retired), "deferred_documents": dict(self.deferred),
                "coverage": copy.deepcopy(self.coverage), "search_log": copy.deepcopy(self.search_log),
                "errors": copy.deepcopy(self.errors),
                "submission_validation_failures": [copy.deepcopy(e) for e in self.audit
                                                   if e.get("event") == "submission_validation_failed"],
            }
            await asyncio.to_thread(save_json, self.attempt / "session.json", snapshot)
            await asyncio.to_thread(save_jsonl, self.attempt / "audit.jsonl", copy.deepcopy(list(self.audit)))

    async def search(self, query, limit=None, *, direction_id="", hypothesis_id="", intent="discovery"):
        self.workflow.check_search(direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent)
        if self.settings.offline or self.source is None:
            return {"error": "literature_network_disabled"}
        async with self.lock:
            if self.searches >= self.settings.max_searches:
                return {"error": "search_budget_exhausted"}
            self.searches += 1
            number = self.searches
            self.touch()
        try:
            count = self.settings.results_per_search if limit is None else max(0, min(limit, self.settings.results_per_search))
            result = await self.source.search(query, count)
            self.known_pmids.update(result["pmids"])
        except SourceError as error:
            result = {"query": query, "error": str(error)}
        result.update({"query_id": f"query_{number}", "search_number": number, "retmax": count,
                       "direction_id": direction_id, "hypothesis_id": hypothesis_id, "intent": intent})
        self.search_log.append(result)
        self.audit.append({"role": "source", "operation": "search", "result": result})
        self.touch()
        await self.checkpoint()
        return result

    async def fetch(self, pmids, full_text=False):
        if self.workflow.enabled and self.workflow.background is None:
            return {"error": "retrieve_phenotype_background_before_fetching_literature"}
        if self.settings.offline or self.source is None:
            return {"error": "literature_network_disabled"}
        if any(p not in self.known_pmids for p in pmids):
            return {"error": "pmid_not_retrieved_or_supplied"}
        documents = []
        try:
            unique = list(dict.fromkeys(pmids))
            by_pmid = {d.pmid: d for d in self.documents.values() if d.pmid}
            missing = [p for p in unique if p not in by_pmid]
            by_pmid.update({d.pmid: d for d in await self.source.fetch_abstracts(missing)})
            for pmid in unique:
                doc = by_pmid.get(pmid)
                if doc is None:
                    self.coverage["missing_abstract_pmids"].append(pmid)
                    continue
                if full_text and doc.pmcid and doc.source_kind != "full_text":
                    async with self.lock:
                        if self.full_texts >= self.settings.max_full_texts:
                            documents.append({"error": "full_text_budget_exhausted", "pmid": pmid})
                            continue
                        self.full_texts += 1
                    doc = await self.source.full_text(doc)
                documents.append(await self.stage(doc))
        except SourceError as error:
            return {"error": str(error)}
        self.touch()
        self.audit.append({"role": "source", "operation": "fetch", "pmids": pmids,
                           "full_text": full_text, "documents": documents})
        await self.checkpoint()
        return {"documents": documents}

    async def search_local(self, query, limit=10, *, direction_id="", hypothesis_id="", intent="discovery"):
        self.workflow.check_search(direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent)
        if not query.strip() or limit < 1:
            raise ValueError("local_search_requires_a_literal_phrase_and_positive_limit")
        async with self.lock:
            if self.searches >= self.settings.max_searches:
                return {"error": "search_budget_exhausted"}
            self.searches += 1
            number = self.searches
        matches = [d for d in self.documents.values() if not d.retracted
                   and query.casefold() in (d.title + "\n" + d.text).casefold()]
        selected = matches[:min(limit, self.settings.results_per_search)]
        self.known_pmids.update(d.pmid for d in selected if d.pmid)
        result = {"query": query, "query_id": f"query_{number}", "search_number": number,
                  "count": len(matches), "retmax": limit, "pmids": [d.pmid for d in selected if d.pmid],
                  "document_ids": [d.document_id for d in selected],
                  "records": [self.document_summary(d.document_id) for d in selected],
                  "direction_id": direction_id, "hypothesis_id": hypothesis_id, "intent": intent,
                  "source": "staged_corpus_literal_phrase_search"}
        self.search_log.append(result)
        self.audit.append({"role": "source", "operation": "local_search", "result": result})
        self.touch()
        await self.checkpoint()
        return result

    def read(self, document_id, start=0, limit=None):
        doc = self.documents.get(document_id)
        if doc is None:
            raise ValueError("unknown_document_id")
        if doc.retracted:
            raise ValueError("retracted_source")
        if start < 0 or start >= len(doc.text):
            raise ValueError("document_start_out_of_range")
        limit = min(limit or self.settings.chunk_chars, self.settings.chunk_chars)
        if limit <= 0:
            raise ValueError("document_limit_must_be_positive")
        end = min(len(doc.text), start + limit)
        windows = self.read_windows[document_id]
        if (start, end) not in windows and len(windows) >= self.settings.max_chunks_per_paper:
            raise ValueError("document_window_budget_exhausted")
        windows.add((start, end))
        result = {"document_id": document_id, "start": start, "end": end,
                  "source_kind": doc.source_kind, "total_chars": len(doc.text), "text": doc.text[start:end]}
        if self.compact_submissions:
            result["passages"] = read_passages(doc, start, end)
        return result

    def claim_summary(self, key):
        record = self.claims[key]
        draft, mapping = record["draft"], record["mapping"]
        observation = self.observations.get(key)
        return {
            "claim_id": key, "document_id": record["evidence"].document_id if record["evidence"] else None,
            "hypothesis_id": draft.hypothesis_id,
            "cell_text": draft.cell_text, "process_text": draft.process_text,
            "evidence_mode": draft.evidence_mode,
            "cell_evidence_mode": draft.cell_evidence_mode,
            "link_evidence_mode": draft.link_evidence_mode,
            "knowledge_source": draft.knowledge_source,
            "experimental_support": ExperimentalValidation(evidence=record.get("experimental_evidence", [])).status,
            "experimental_evidence_count": len(record.get("experimental_evidence", [])),
            "reasoning_chain": " -> ".join([self.phenotype.label, *[s.target for s in draft.reasoning_steps]])
                               if draft.reasoning_steps else "",
            "direction": draft.direction, "species": draft.species,
            "cl_id": mapping.cl.term_id if mapping and mapping.cl else None,
            "cl_label": mapping.cl.label if mapping and mapping.cl else None,
            "go_id": mapping.go.term_id if mapping and mapping.go else None,
            "go_label": mapping.go.label if mapping and mapping.go else None,
            "mapping_revision": record["revision"],
            "mapping_rationale": mapping.rationale if mapping else None,
            "release_scope": observation.release_scope if observation else None,
            "review_reason": observation.review.reason if observation else None,
            "gate_reasons": observation.gate_reasons if observation else [],
            "research_gaps": [g.model_dump() for g in observation.review.research_gaps] if observation else [],
            "blocking_issues": [issue.model_dump() for issue in observation.review.blocking_issues] if observation else [],
            "retired": key in self.retired,
        }

    def pending(self):
        return {"workflow": self.workflow.pending(),
                "claims": [key for key in self.claims if key not in self.retired and key not in self.observations],
                "documents": [key for key, doc in self.documents.items()
                              if not self.analyzed[key] and key not in self.deferred and not doc.retracted],
                "unfetched_search_hits": [entry["query"] for entry in self.search_log
                                          if entry.get("count", 0) > 0 and "error" not in entry]
                                         if not self.documents and not self.claims else []}

    def state(self):
        pending = self.pending()
        return {
            "state_version": self.version, "phenotype_id": self.phenotype.phenotype_id,
            "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
            "summary_counts": self.summary_counts(),
            "workflow": self.workflow.snapshot(), "offline": self.settings.offline,
            "query_summaries": [{key: query.get(key) for key in
                                 ("query_id", "search_number", "retmax", "query", "effective_query", "filters", "intent", "direction_id",
                                  "hypothesis_id", "count", "pmids", "error")}
                                for query in self.search_log],
            "documents": [self.document_summary(key) | {"analyzed_windows": self.analyzed[key],
                                                       "deferred_reason": self.deferred.get(key)}
                          for key in self.documents],
            "claims": [self.claim_summary(key) for key in self.claims], "pending": pending,
            "research_gaps": self.coverage["research_gaps"],
            "submission_validation_failures": [e for e in self.audit if e.get("event") == "submission_validation_failed"][-10:],
            "hypothesis_generation_feedback": [
                {"context_key": key, "candidate_count": len((value.get("draft") or {}).get("hypotheses", [])),
                 "diagnostics": value["diagnostics"], "repair_attempted": value["repair_attempted"]}
                for key, value in self.failed_hypothesis_contexts.items()],
            "assessment_submission_feedback": [
                {"hypothesis_id": key, **{field: value[field] for field in
                 ("failure_id", "issues", "repair_attempted", "status", "completion_attempted") if field in value},
                 "next_action": ("complete_reasoning" if self.compact_submissions
                     and value.get("status") in {"needs_reasoning", "invalid_reference"}
                     and not value.get("completion_attempted") else
                     "stop_or_explicit_reevaluate" if value["repair_attempted"] else "repair_submission")}
                for key, value in self.failed_assessments.items()],
            "ready_to_finish": not any(pending.values()) and bool(
                self.documents or self.claims or any("error" not in entry for entry in self.search_log)),
            "budgets": {"searches_remaining": self.settings.max_searches - self.searches,
                        "full_texts_remaining": self.settings.max_full_texts - self.full_texts,
                        "subagent_calls_remaining": self.settings.max_subagent_calls - self.dispatches,
                        "chunk_chars": self.settings.chunk_chars,
                        "suggested_overlap": self.settings.chunk_overlap,
                        "max_windows_per_document": self.settings.max_chunks_per_paper},
        }

    def root_tools(self):
        @tool
        async def pubmed_search(term: str, retmax: int = 10, direction_id: str = "",
                                hypothesis_id: str = "", intent: str = "discovery") -> dict:
            """Search PubMed with hasabstract enforced. Discovery needs direction_id; verification needs hypothesis_id + support intent. Challenge is optional."""
            try:
                return await self.search(term, retmax, direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent)
            except ValueError as error:
                return {"error": str(error)}

        @tool
        async def fetch_abstracts(pmids: list[str]) -> dict:
            """Use upstream batch retrieval, save abstracts and return document manifests."""
            return await self.fetch(pmids, False)

        @tool
        async def fetch_full_text(pmcids: list[str]) -> dict:
            """Use upstream PMC full-text retrieval for staged papers; return stored document manifests."""
            by_pmc = {d.pmcid: d.pmid for d in self.documents.values() if d.pmcid and d.pmid}
            if any(p not in by_pmc for p in pmcids):
                return {"error": "fetch_abstracts_first_for_these_pmcids"}
            return await self.fetch([by_pmc[p] for p in pmcids], True)

        @tool
        async def get_research_state() -> dict:
            """Inspect trusted document/claim manifests, review feedback, budgets and pending work."""
            return self.state()

        @tool
        async def defer_document(document_id: str, reason: str) -> dict:
            """Record an explicit reason for not analyzing a staged paper."""
            if document_id not in self.documents or not reason.strip():
                return {"error": "known_document_and_reason_required"}
            self.deferred[document_id] = reason
            self.touch()
            await self.checkpoint()
            return {"deferred": document_id, "reason": reason}

        @tool
        async def retire_claim(claim_id: str, reason: str) -> dict:
            """Retire a superseded claim; retain its audit, exclude its edge from release."""
            if claim_id not in self.claims or not reason.strip():
                return {"error": "known_claim_and_reason_required"}
            if self.workflow.enabled and any(h["status"] == "retained" and h["claim_id"] == claim_id
                                             for h in self.workflow.hypotheses.values()):
                return {"error": "reassess_the_retained_hypothesis_before_retiring_its_current_claim"}
            async with self.claim_locks[claim_id]:
                self.retired[claim_id] = reason
                self.touch()
            await self.checkpoint()
            return {"retired": claim_id, "reason": reason}

        @tool
        async def get_phenotype_background(ancestor_depth: int = 3, disease_query: str = "",
                                           offset: int = 0, limit: int = 10) -> dict:
            """Read HPO term/ancestor metadata and scoped disease annotations; negatives never become positive links."""
            if self.hpo_context is None:
                return {"error": "hpo_context_index_not_configured"}
            else:
                try:
                    result = await asyncio.to_thread(self.hpo_context.lookup, self.phenotype.phenotype_id,
                                                      ancestor_depth=ancestor_depth, disease_query=disease_query,
                                                      offset=offset, limit=limit)
                except ValueError as error:
                    return {"error": str(error)}
                for disease in result["diseases"]:
                    for annotation in disease["annotations"]:
                        self.known_pmids.update(re.findall(r"PMID:(\d+)", annotation.get("reference", "")))
            if self.workflow.background is None:
                self.workflow.background = {"input": self.phenotype.model_dump(), "lookups": []}
            self.workflow.background["lookups"].append(result)
            self.audit.append({"role": "background", "result": result})
            self.touch()
            await self.checkpoint()
            return result

        @tool
        async def register_discovery_plan(plan: DiscoveryPlan) -> dict:
            """Register a diverse literature search plan before querying; does not propose cellular answers."""
            try:
                self.workflow.register_plan(plan)
            except ValueError as error:
                return {"error": str(error)}
            self.touch()
            await self.checkpoint()
            return {"registered": True, "directions": self.workflow.plan["directions"]}

        @tool
        async def search_local_documents(query: str, limit: int = 10, direction_id: str = "",
                                         hypothesis_id: str = "", intent: str = "discovery") -> dict:
            """Literal phrase search over staged sources (no PubMed syntax); useful in offline research."""
            try:
                return await self.search_local(query, limit, direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent)
            except ValueError as error:
                return {"error": str(error)}

        return [pubmed_search, fetch_abstracts, fetch_full_text,
                get_research_state, defer_document, retire_claim, get_phenotype_background, register_discovery_plan,
                search_local_documents]

    def research_workspace(self, saved_catalog=None):
        """Each specialist gets its own observed windows, never root-supplied quotations."""
        provided = EvidenceWindows(saved_catalog)

        @tool
        def read_document(document_id: str, start: int = 0, limit: int = 16000) -> dict:
            """Read a staged source. Select returned evidence_ref passages; the host fills text/source/offset metadata. Abstract and full text are distinct artifacts."""
            try:
                result = self.read(document_id, start, limit)
                result["already_read"] = any(w["document_id"] == document_id
                    and w["start"] <= result["start"] and result["end"] <= w["end"] for w in provided)
                if result["already_read"]:
                    result["next_action"] = ("reuse the existing evidence_ref passages or submit" if self.compact_submissions
                                             else "reuse the exact quote with this window's document_id or submit; omit optional citation offsets rather than estimating")
                provided.append(result)
                return model_window(result, provided.catalog) if self.compact_submissions else result
            except ValueError as error:
                return {"error": str(error)}

        if not self.compact_submissions:
            read_document.description = "Read a staged source by character offsets; copy quotation and document_id from the same returned window."

        @tool
        def find_in_document(document_id: str, query: str) -> dict:
            """Locate a phrase in a staged source; read the matching window before citing it."""
            doc = self.documents.get(document_id)
            if doc is None or doc.retracted:
                return {"error": "unknown_or_retracted_document"}
            if len(query.strip()) < 2:
                return {"error": "query_too_short"}
            matches = list(re.finditer(re.escape(query), doc.text, re.IGNORECASE))[:20]
            result = {"matches": [{"start": m.start(), "suggested_read_start": max(0, m.start() - 300)} for m in matches]}
            if self.compact_submissions:
                result["read_evidence_refs"] = sorted({provided.catalog.alias(p["evidence_ref"]) for window in provided
                    if window["document_id"] == document_id for p in window["passages"]
                    if query.casefold() in p["text"].casefold()})
                result["instruction"] = "Reuse already-read refs; locations without read refs require read_document."
            return result

        return provided, [read_document, find_in_document]

    async def generate_hypotheses(self, request):
        async with self.hypothesis_generation_lock:
            return await self._generate_hypotheses(request)

    async def _generate_hypotheses(self, request):
        self.workflow.require_discovery()
        if self.compact_submissions and request.get("repair_submission"):
            raise ValueError("compact_submissions_have_no_format_repair_mode; select valid read evidence or report the scientific gap")
        extend = request.get("extend_pool", False) is True
        if self.workflow.hypotheses and not extend:
            raise ValueError("hypothesis_batch_already_registered; refine candidates in hypothesis-evaluator")
        context_key = fingerprint({"documents": {key: doc.sha256 for key, doc in self.documents.items()},
                                   "plan": self.workflow.plan, "hypotheses": self.workflow.hypotheses,
                                   "extend_pool": extend})
        previous = self.failed_hypothesis_contexts.get(context_key)
        repair = request.get("repair_submission", False) is True
        if previous and (not repair or not previous["draft"] or previous["repair_attempted"]):
            raise HypothesisGenerationFailure("hypothesis_context_already_failed: changing focus alone cannot restart generation", {
                "context_key": context_key, "draft": previous["draft"], "diagnostics": previous["diagnostics"],
                "next_action": "repair_submission" if previous["draft"] and not previous["repair_attempted"]
                               else "stop_or_retrieve_new_sources",
            })
        provided, tools = self.research_workspace(previous.get("evidence_catalog") if previous else None)
        seed_contexts = []
        if previous:
            previous["repair_attempted"] = True
            cited_ids = {ref["document_id"] for h in previous["draft"]["hypotheses"] for ref in h["seed_evidence"]}
            for window in previous["read_windows"]:
                restored = self.read(window["document_id"], window["start"], window["end"] - window["start"])
                provided.append(restored)
                if window["document_id"] in cited_ids and not any(
                    w["document_id"] == restored["document_id"] and w["start"] <= restored["start"]
                    and restored["end"] <= w["end"] for w in seed_contexts
                ):
                    seed_contexts.append(restored)
        last_submission = {"draft": previous["draft"] if previous else None,
                           "diagnostics": previous["diagnostics"] if previous else []}

        def check_batch(batch):
            last_submission["draft"] = batch.model_dump()
            try:
                result = self.workflow.register_hypotheses(batch, provided, commit=False, extend=extend)
            except ValueError as error:
                last_submission["diagnostics"] = getattr(error, "diagnostics", [{"error_code": str(error)}])
                raise
            last_submission["diagnostics"] = []
            return result

        payload = {
            "phenotype": self.phenotype.model_dump(), "background": self.workflow.background,
            "discovery_plan": self.workflow.plan,
            "minimum_hypotheses": 1 if self.workflow.hypotheses else self.settings.min_hypotheses,
            "minimum_papers_to_read": 1 if self.workflow.hypotheses else self.settings.min_discovery_papers,
            "existing_hypotheses": list(self.workflow.hypotheses.values()),
            "documents": [self.document_summary(key) for key in self.documents],
            "search_history": self.search_log, "focus": request.get("focus", ""),
            "previous_submission": previous["draft"] if previous else None,
            "evidence_catalog_id": provided.catalog.catalog_id,
            "submission_diagnostics": previous["diagnostics"] if previous else [],
            "seed_contexts": [model_window(w, provided.catalog) for w in seed_contexts] if self.compact_submissions else seed_contexts,
            "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
        }
        try:
            batch = await self.engine.hypothesize(payload, self.audit, tools=tools, validator=check_batch,
                **({"assembler": lambda value: assemble_hypotheses(value, self.documents, provided)}
                   if self.compact_submissions else {}))
        except (ValueError, GraphRecursionError) as error:
            if not isinstance(error, GraphRecursionError) and not str(error).startswith("no_progress:"):
                raise
            if self.compact_submissions and isinstance(error, SubmissionRepairError):
                last_submission = {"draft": copy.deepcopy(error.arguments), "diagnostics": copy.deepcopy(error.issues)}
            failure = {**last_submission, "repair_attempted": bool(previous) or self.compact_submissions,
                       "evidence_catalog": provided.catalog.snapshot(),
                       "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
                       "read_windows": list({(w["document_id"], w["start"], w["end"]):
                           {k: w[k] for k in ("document_id", "start", "end")} for w in provided}.values())}
            self.failed_hypothesis_contexts[context_key] = failure
            self.coverage.setdefault("hypothesis_generation_failures", []).append(copy.deepcopy(last_submission))
            self.touch()
            raise HypothesisGenerationFailure(str(error), {
                "context_key": context_key, **last_submission,
                "next_action": "repair_submission" if failure["draft"] and not failure["repair_attempted"]
                               else "stop_or_retrieve_new_sources",
            }) from error
        finally:
            if self.compact_submissions:
                self.audit.append({"event": "evidence_catalog", "role": "hypothesize", **provided.catalog.snapshot()})
        try:
            hypotheses = self.workflow.register_hypotheses(batch, provided, extend=extend)
        except ValueError as error:
            self.coverage.setdefault("hypothesis_generation_failures", []).append(batch.model_dump())
            minimum = 1 if self.workflow.hypotheses else self.settings.min_hypotheses
            if len(batch.hypotheses) < minimum and batch.limitation.strip():
                self.touch()
                return {"hypotheses": [], "status": "needs_more_evidence", "limitation": batch.limitation,
                        "unregistered_candidate_count": len(batch.hypotheses), "requirement": str(error),
                        "next_action": "broaden literature recall or finish honestly with incomplete workflow"}
            raise
        for window in provided:
            self.analyzed[window["document_id"]].append((window["start"], window["end"]))
        self.failed_hypothesis_contexts.clear()
        self.touch()
        return {"hypotheses": hypotheses, "next_action": "search supporting literature for EVERY hypothesis"}

    async def evaluate_hypothesis(self, request):
        hid = request["hypothesis_id"]
        async with self.hypothesis_locks.setdefault(hid, asyncio.Lock()):
            return await self._evaluate_hypothesis(request)

    def validation_context_sources(self, hid):
        """Select bounded, already staged source bodies before spending a model call."""
        selected, missing = {}, []
        queries = self.workflow.validation_searches(hid)
        for intent in ("support", "challenge"):
            relevant = [query for query in queries if query.get("intent") == intent and query.get("retmax", 0) > 0]
            pmids = list(dict.fromkeys(pmid for query in relevant for pmid in query.get("pmids", [])))
            ids = list(dict.fromkeys(key for query in relevant for key in query.get("document_ids", [])))
            if not pmids and not ids:
                if intent == "support" and any(query.get("count", 0) > 0 for query in relevant):
                    missing.append({"intent": intent, "pmids": [], "document_ids": [],
                                    "query_ids": [query["query_id"] for query in relevant]})
                continue
            candidates = [doc for doc in self.documents.values() if not doc.retracted and (doc.pmid in pmids or doc.document_id in ids)]
            if not candidates:
                if intent == "support":
                    missing.append({"intent": intent, "pmids": pmids[:3], "document_ids": ids[:3],
                                    "query_ids": [query["query_id"] for query in relevant]})
                continue
            candidates.sort(key=lambda doc: (doc.source_kind != "abstract",
                pmids.index(doc.pmid) if doc.pmid in pmids else len(pmids)))
            doc = candidates[0]
            selected.setdefault(doc.document_id, []).append(intent)
        if missing:
            raise ResearchActionRequired("needs_sources", "validation_sources_not_staged", missing=missing,
                                         next_action="fetch_validation_sources_or_revise_queries")
        return selected

    async def _evaluate_hypothesis(self, request):
        hid = request["hypothesis_id"]
        try:
            self.workflow.require_verification(hid)
        except ValueError as error:
            if self.compact_submissions and str(error).startswith("run_support_literature_search"):
                raise ResearchActionRequired("needs_sources", str(error), hypothesis_id=hid,
                                             next_action="run_missing_support_searches") from error
            raise
        hypothesis = self.workflow.hypotheses[hid]
        mode = request.get("mode", "evaluate")
        modes = {"evaluate", "reevaluate", "complete_reasoning"} if self.compact_submissions else {"evaluate", "repair", "reevaluate"}
        if not isinstance(mode, str) or mode not in modes:
            raise ValueError("compact_assessment_mode_must_be_evaluate_complete_reasoning_or_reevaluate" if self.compact_submissions
                             else "assessment_mode_must_be_evaluate_repair_or_reevaluate")
        previous = self.failed_assessments.get(hid)
        validation_sources = self.validation_context_sources(hid) if self.compact_submissions else {}
        if mode == "complete_reasoning":
            if (not previous or request.get("failure_id") != previous["failure_id"]
                    or previous.get("status") not in {"needs_reasoning", "invalid_reference"}):
                raise ResearchActionRequired("needs_reasoning", "completion_requires_current_scientific_failure", next_action="inspect_assessment_submission_feedback")
            if previous.get("completion_attempted"):
                raise ResearchActionRequired("needs_reasoning", "scientific_completion_budget_exhausted", next_action="stop_or_explicit_reevaluate")
            previous["completion_attempted"] = True
        elif mode == "repair":
            if not previous or request.get("failure_id") != previous["failure_id"]:
                raise ValueError("repair_requires_current_failure_id_from_assessment_submission_feedback")
            if previous["repair_attempted"]:
                raise AssessmentSubmissionFailure("assessment_repair_budget_exhausted", {
                    "status": "invalid_format", "failure_id": previous["failure_id"], "issues": previous["issues"],
                    "next_action": "stop_or_explicit_reevaluate", "retryable": False,
                })
        elif mode == "reevaluate":
            if not isinstance(request.get("reassessment_reason"), str) or not request["reassessment_reason"].strip():
                raise ValueError("explicit_reevaluation_requires_reassessment_reason")
        elif previous:
            raise AssessmentSubmissionFailure("scientific_submission_requires_explicit_reassessment" if self.compact_submissions
                                              else "assessment_submission_requires_repair_mode", {
                "status": previous.get("status", "invalid_format"), "failure_id": previous["failure_id"], "issues": previous["issues"],
                "next_action": ("complete_reasoning" if self.compact_submissions
                    and previous.get("status") in {"needs_reasoning", "invalid_reference"}
                    and not previous.get("completion_attempted") else
                    "stop_or_explicit_reevaluate" if previous["repair_attempted"] else "repair_submission"),
                "retryable": False,
            })
        if hypothesis["assessment"]:
            hypothesis["status"] = "proposed"
            if hypothesis["claim_id"]:
                self.retired[hypothesis["claim_id"]] = "Pending hypothesis re-verification"
            self.touch()
        provided, tools = self.research_workspace(previous.get("evidence_catalog") if previous else None)
        initial_contexts = []
        if previous and self.compact_submissions:
            for saved in previous["read_windows"]:
                provided.append(self.read(saved["document_id"], saved["start"], saved["end"] - saved["start"]))
        if mode == "repair":
            previous["repair_attempted"] = True
            for window in previous["read_windows"]:
                restored = self.read(window["document_id"], window["start"], window["end"] - window["start"])
                provided.append(restored)
                quote = (previous["draft"].get("claim") or {}).get("supporting_quote", "")
                if quote and quote in restored["text"] and not any(
                    w["document_id"] == restored["document_id"] for w in initial_contexts
                ):
                    initial_contexts.append(restored)
        for seed in hypothesis["seed_evidence"]:
            window = self.read(seed["document_id"], max(0, seed["span_start"] - 400),
                               min(self.settings.chunk_chars, seed["span_end"] - seed["span_start"] + 800))
            provided.append(window)
            initial_contexts.append(window)
        for key in validation_sources:
            window = self.read(key, 0, self.settings.chunk_chars)
            if not any(w["document_id"] == key and w["start"] <= window["start"] and window["end"] <= w["end"] for w in provided):
                provided.append(window)
                initial_contexts.append(window)

        def check_assessment(value):
            if mode == "reevaluate" and (previous or hypothesis["assessment"]) and not value.revision_note.strip():
                raise contract_error("assessment.reassessment_note", field=("revision_note",))
            read_ids = {w["document_id"] for w in provided}
            for field in ("supporting_document_ids", "contradicting_document_ids"):
                missing = sorted(set(getattr(value, field)) - read_ids)
                if missing:
                    raise contract_error("assessment.sources", field=(field,), expected=sorted(read_ids), actual=missing)
            if value.claim is None:
                return
            supporting_searches = [q for q in self.workflow.validation_searches(hid)
                                  if q.get("intent") == "support" and q.get("retmax", 0) > 0]
            validation_pmids = {pmid for q in supporting_searches for pmid in q.get("pmids", [])}
            read_pmids = {self.documents[key].pmid for key in read_ids if self.documents[key].pmid}
            validation_ids = {key for q in supporting_searches for key in q.get("document_ids", [])}
            if (validation_pmids or validation_ids) and not (validation_pmids & read_pmids or validation_ids & read_ids):
                if self.compact_submissions:
                    raise ResearchActionRequired("needs_reading", "validation_source_body_not_provided", hypothesis_id=hid,
                        document_ids=list(validation_sources), next_action="prepare_validation_context")
                raise ValueError("read_validation_sources_before_retention; metadata-only hits are not examined evidence")
            draft = value.claim.bind_phenotype(self.phenotype.phenotype_id, self.phenotype.label)
            value.claim = draft
            if draft.hypothesis_id and draft.hypothesis_id != hid:
                raise ValueError("assessment_hypothesis_id_mismatch")
            original = hypothesis["draft"]
            if (draft.cell_text != original["cell_text"] or draft.process_text != original["process_text"]
                    or draft.direction != original["direction"]) and not value.revision_note.strip():
                raise ValueError("changed_hypothesis_requires_revision_note")
            window = main_quote_window(draft, self.documents, provided)
            if window:
                evidence = anchor_claim(draft, self.documents[window["document_id"]], window["start"], window["end"], provided=provided)
            else:
                evidence = anchor_claim(draft, None, 0, 0)
            anchor_reasoning(self.phenotype, draft, evidence, self.documents, provided)
            anchor_experiments(draft, self.documents, provided)

        allowed_fields = []
        if mode == "complete_reasoning":
            for issue in previous["issues"]:
                path = list(issue["path"])
                if issue["code"] == "evidence.ref" and path and isinstance(path[-1], int):
                    path = path[:-1]
                allowed_fields.append(path)
                if path and path[-1] == "explanation":
                    allowed_fields.append([*path[:-1], "evidence_refs"])

        payload = {
            "phenotype": self.phenotype.model_dump(), "background": self.workflow.background,
            "hypothesis": hypothesis,
            "seed_contexts": [model_window(w, provided.catalog) for w in initial_contexts] if self.compact_submissions else initial_contexts,
            "verification_searches": self.workflow.validation_searches(hid),
            "validation_sources": [{"document_id": key, "intents": intents} for key, intents in validation_sources.items()],
            "documents": [self.document_summary(key) for key in self.documents],
            "competing_hypotheses": [{"hypothesis_id": key, **value["draft"]}
                                     for key, value in self.workflow.hypotheses.items() if key != hid],
            "reasoning_start": self.phenotype.label, "reasoning_start_id": self.phenotype.phenotype_id,
            "reasoning_start_token": "INPUT_PHENOTYPE",
            "focus": request.get("focus", ""),
            "mode": mode,
            "previous_submission": copy.deepcopy(previous["draft"]) if previous else None,
            "submission_diagnostics": copy.deepcopy(previous["issues"]) if previous else [],
            "completion_fields": allowed_fields,
            "evidence_catalog_id": provided.catalog.catalog_id,
            "restored_read_document_ids": sorted({w["document_id"] for w in provided}) if mode == "repair" else [],
            "previous_assessment": copy.deepcopy(hypothesis["assessment"]),
            "reassessment_reason": request.get("reassessment_reason", ""),
            "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
        }
        if previous and self.compact_submissions:
            def references(value):
                if isinstance(value, dict):
                    return set().union(*(references(item) for item in value.values()))
                if isinstance(value, list):
                    return set().union(*(references(item) for item in value))
                return {provided.catalog.resolve(value)} if isinstance(value, str) else set()
            selected = references(previous["draft"])
            payload["previous_evidence"] = list({p["evidence_ref"]: {"evidence_ref": provided.catalog.alias(p["evidence_ref"]),
                "document_id": window["document_id"], "text": p["text"]} for window in provided
                for p in window["passages"] if p["evidence_ref"] in selected}.values())
        last_completed = {"draft": copy.deepcopy(previous["draft"]) if previous else None}

        def assemble_submission(value):
            if mode == "complete_reasoning":
                merged = complete_scientific_fields(previous["draft"], value.model_dump(exclude_unset=True), allowed_fields)
                last_completed["draft"] = copy.deepcopy(merged)
                from .schemas import ScientificAssessment
                value = ScientificAssessment.model_validate(merged)
            return assemble_assessment(value, self.phenotype, hid, self.documents, provided,
                original=hypothesis["draft"], previous_assessment=hypothesis["assessment"],
                reassessment_reason=request.get("reassessment_reason", ""))
        try:
            assessment = await self.engine.verify_hypothesis(payload, self.audit, tools=tools, validator=check_assessment,
                **({"assembler": assemble_submission}
                   if self.compact_submissions else {}))
        except SubmissionRepairError as error:
            failed_draft = last_completed["draft"] if mode == "complete_reasoning" else error.arguments
            if self.compact_submissions and error.status != "field_conflict":
                failed_draft = normalize_assessment(failed_draft)[0]
            failure_id = "assessment_failure_" + fingerprint({"hypothesis_id": hid, "draft": failed_draft})[:24]
            failure = {"failure_id": failure_id, "draft": copy.deepcopy(failed_draft), "status": error.status,
                       "completion_attempted": mode == "complete_reasoning", "evidence_catalog": provided.catalog.snapshot(),
                       "issues": copy.deepcopy(error.issues), "repair_attempted": mode == "repair" or self.compact_submissions,
                       "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
                       "read_windows": list({(w["document_id"], w["start"], w["end"]):
                           {k: w[k] for k in ("document_id", "start", "end")} for w in provided}.values())}
            self.failed_assessments[hid] = failure
            self.touch()
            raise AssessmentSubmissionFailure(str(error), {
                "status": error.status, "failure_id": failure_id, "issues": error.issues,
                "failed_submission": copy.deepcopy(failed_draft), "retryable": False,
                "next_action": ("complete_reasoning" if self.compact_submissions and mode != "complete_reasoning"
                                and error.status in {"needs_reasoning", "invalid_reference"} else
                                "stop_or_explicit_reevaluate" if mode in {"repair", "complete_reasoning"} or self.compact_submissions else "repair_submission"),
            }) from error
        except (APIConnectionError, APIStatusError, APITimeoutError, TimeoutError):
            if mode == "repair":
                previous["repair_attempted"] = False
            elif mode == "complete_reasoning":
                previous["completion_attempted"] = False
            raise
        except ValueError as error:
            if "missing structured_response" in str(error):
                if mode == "complete_reasoning":
                    previous["completion_attempted"] = False
                elif mode == "repair":
                    previous["repair_attempted"] = False
            raise
        finally:
            if self.compact_submissions:
                self.audit.append({"event": "evidence_catalog", "role": "verify_hypothesis", "hypothesis_id": hid,
                                   **provided.catalog.snapshot()})
        check_assessment(assessment)
        read_ids = {w["document_id"] for w in provided}
        cid = None
        if assessment.claim:
            draft = assessment.claim.model_copy(update={"hypothesis_id": hid})
            original = hypothesis["draft"]
            if (draft.cell_text != original["cell_text"] or draft.process_text != original["process_text"]
                    or draft.direction != original["direction"]) and not assessment.revision_note.strip():
                raise ValueError("changed_hypothesis_requires_revision_note")
            document_id = draft.supporting_document_id
            if draft.supporting_quote and not document_id:
                document_id = main_quote_window(draft, self.documents, provided)["document_id"]
            result = await self.collect_evidence({"document_id": document_id, "supporting_document_ids": sorted(read_ids)},
                evidence_mode=draft.evidence_mode, extraction_override=Extraction(claims=[draft]), provided_context=provided)
            if not result["claims"]:
                raise ValueError("retained_claim_failed_integrity_checks: " + json.dumps(result["invalid_claims"]))
            cid = result["claims"][0]["claim_id"]
        old_cid = hypothesis["claim_id"]
        if old_cid and old_cid != cid:
            self.retired[old_cid] = "Superseded by hypothesis re-verification"
        if hypothesis["assessment"]:
            hypothesis["assessment_history"].append(hypothesis["assessment"])
        if cid:
            self.retired.pop(cid, None)
            self.observations.pop(cid, None)
        assessment_payload = assessment.model_dump()
        if cid:
            assessment_payload["claim"] = self.claims[cid]["draft"].model_dump()
        hypothesis.update({"status": {"retain": "retained", "reject": "rejected", "inconclusive": "inconclusive"}[assessment.decision],
                           "assessment": {**assessment_payload,
                                          "query_ids": [q["query_id"] for q in self.workflow.validation_searches(hid)],
                                          "read_document_ids": sorted(read_ids)}, "claim_id": cid})
        if mode in {"repair", "reevaluate", "complete_reasoning"}:
            self.coverage.setdefault("assessment_recoveries", []).append({
                "hypothesis_id": hid, "mode": mode, "failure_id": previous["failure_id"] if previous else None,
                "previous_submission": payload["previous_submission"],
                "previous_assessment": payload["previous_assessment"], "assessment": assessment.model_dump(),
                "reassessment_reason": payload["reassessment_reason"],
            })
        self.failed_assessments.pop(hid, None)
        for window in provided:
            self.analyzed[window["document_id"]].append((window["start"], window["end"]))
        self.touch()
        return {"hypothesis_id": hid, "status": hypothesis["status"], "claim_id": cid,
                "rationale": assessment.rationale, "revision_note": assessment.revision_note,
                "remaining_hypotheses": [key for key, item in self.workflow.hypotheses.items() if item["status"] == "proposed"]}

    async def extract(self, request):
        return await self.collect_evidence(request, evidence_mode="direct")

    async def reason(self, request):
        return await self.collect_evidence(request, evidence_mode="inferred")

    async def collect_evidence(self, request, *, evidence_mode, extraction_override=None, provided_context=None):
        if self.workflow.enabled and extraction_override is None:
            raise ValueError("use_hypothesis_generator_then_hypothesis_evaluator_before_creating_final_claims")
        key = request.get("document_id", "")
        if not key and evidence_mode == "direct":
            raise ValueError("direct_extraction_requires_document_id")
        supporting_ids = request.get("supporting_document_ids", [])
        if not isinstance(supporting_ids, list) or not all(isinstance(s, str) for s in supporting_ids):
            raise ValueError("supporting_document_ids_must_be_string_array")
        allowed = list(dict.fromkeys([s for s in [key, *supporting_ids] if s]))
        if any(doc_id not in self.documents for doc_id in allowed):
            raise ValueError("unknown_document_id")
        window = self.read(key, request.get("start", 0), request.get("limit")) if key else {
            "document_id": "", "start": 0, "end": 0, "text": "",
        }
        doc = self.documents.get(key)
        provided = [window] if key else []
        provided.extend(provided_context or [])

        @tool
        def read_document(start: int = 0, limit: int = 16000, document_id: str = "") -> dict:
            """Read anchor/supporting paper by CHARACTER offsets; empty document_id selects the anchor paper."""
            try:
                selected = document_id or key
                if selected not in allowed:
                    raise ValueError("document_not_in_extraction_context")
                result = self.read(selected, start, limit)
                provided.append(result)
                return result
            except ValueError as error:
                return {"error": str(error)}

        @tool
        def find_in_document(query: str, document_id: str = "") -> dict:
            """Locate a phrase in an anchor/supporting paper; read_document is required before citing it."""
            selected = document_id or key
            if selected not in allowed:
                return {"error": "document_not_in_extraction_context"}
            search_doc = self.documents[selected]
            if search_doc.retracted:
                return {"error": "retracted_source"}
            if len(query.strip()) < 2:
                return {"error": "query_must_have_at_least_two_characters"}
            matches = []
            for match in re.finditer(re.escape(query), search_doc.text, flags=re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(search_doc.text), match.end() + 200)
                matches.append({"start": match.start(), "suggested_read_start": start,
                                "preview": search_doc.text[start:end]})
                if len(matches) == 20:
                    break
            return {"matches": matches, "limit": 20,
                    "instruction": "Read the relevant window before extracting a claim from it."}

        def anchor(draft):
            if draft.evidence_mode != evidence_mode:
                raise ValueError(f"this_subagent_requires_{evidence_mode}_evidence")
            experiments = anchor_experiments(draft, self.documents, provided)
            if not draft.supporting_quote:
                evidence = anchor_claim(draft, None, 0, 0)
                reasoning = anchor_reasoning(self.phenotype, draft, evidence, self.documents, provided)
                return evidence, {"start": 0, "end": 0}, reasoning, experiments
            selected = draft.supporting_document_id or key
            if selected not in allowed:
                raise ValueError("supporting_document_id_must_identify_an_available_source")
            selected_doc = self.documents[selected]
            errors = []
            for supplied in provided:
                if supplied["document_id"] != selected:
                    continue
                try:
                    evidence = anchor_claim(draft, selected_doc, supplied["start"], supplied["end"], provided=provided)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                reasoning = anchor_reasoning(self.phenotype, draft, evidence, self.documents, provided)
                return evidence, supplied, reasoning, experiments
            if "cell_not_in_quote" in errors:
                raise ValueError("cell_not_in_quote: direct cell identity must occur in the quote; "
                                 "re-reading cannot fix it. Use a valid quote or return a research gap "
                                 "for evidence-reasoner to assess inferred cell identity.")
            raise ValueError(errors[-1] if errors else "quote_not_in_supplied_window")

        @tool("validate_reasoning" if evidence_mode == "inferred" else "validate_extraction")
        def validate_candidates(claims: list[ClaimDraft]) -> dict:
            """Check quotes against actually read windows before finalizing; repair any failures."""
            results = []
            for draft in claims:
                try:
                    anchor(draft)
                    results.append({"valid": True})
                except ValueError as error:
                    results.append({"valid": False, "error": str(error)})
            return {"results": results, "validation_scope": "literal_quotes_and_structure_only",
                    "next_action": "Submit Extraction with claims, or claims=[] with rejection_reason and research_gaps. "
                                   "This validation tool does not finish the task."}

        @tool
        def get_phenotype_context() -> dict:
            """Read trusted input label, definition, synonyms and supplied model/disease contexts."""
            return self.phenotype.model_dump()

        payload = {"phenotype": self.phenotype.model_dump(), "document_id": key,
                   "title": doc.title if doc else "", "pmid": doc.pmid if doc else "",
                   "source_kind": doc.source_kind if doc else "model_knowledge",
                   "window_start": window["start"], "window_end": window["end"],
                   "evidence_context": window["text"], "focus": request.get("focus", ""),
                   "supporting_documents": [self.document_summary(s) for s in allowed if s != key],
                   "reasoning_start": self.phenotype.label,
                   "reasoning_end_format": "cell_text + ': ' + quality_text",
                   "sections": [section.model_dump() for section in doc.sections] if doc else []}
        if extraction_override is not None:
            extraction = extraction_override
        else:
            operation = self.engine.reason if evidence_mode == "inferred" else self.engine.extract
            extraction = await operation(payload, self.audit, tools=[
                read_document, find_in_document, get_phenotype_context, validate_candidates,
            ])
        ids, invalid = [], []
        for draft in extraction.claims:
            try:
                draft = draft.bind_phenotype(self.phenotype.phenotype_id, self.phenotype.label)
                evidence, supplied, reasoning, experiments = anchor(draft)
            except ValueError as error:
                rejected = {"document_id": key, "draft": draft.model_dump(), "error": str(error)}
                self.coverage["invalid_claims"].append(rejected)
                invalid.append({"cell_text": draft.cell_text, "error": str(error)})
                continue
            # A re-read with a different explanatory rationale does not duplicate an observation.
            signature = fingerprint({"document": evidence.text_sha256 if evidence else None,
                                     "start": evidence.span_start if evidence else None,
                                     "reasoning": [s.model_dump() for s in reasoning],
                                     "draft": draft.model_dump(exclude={"rationale"})})
            cid = next((cid for cid, record in self.claims.items() if record["signature"] == signature), None)
            if cid is None:
                cid = claim_id(self.phenotype, draft, evidence, reasoning)
                self.claims[cid] = {"draft": draft, "evidence": evidence, "start": supplied["start"],
                                    "end": supplied["end"], "mapping": None, "revision": 0,
                                    "reasoning_steps": reasoning,
                                    "experimental_evidence": experiments,
                                    "signature": signature}
                self.claim_locks[cid] = asyncio.Lock()
            ids.append(cid)
        for supplied in provided:
            self.analyzed[supplied["document_id"]].append((supplied["start"], supplied["end"]))
        if not extraction.claims:
            self.coverage["no_claim_windows"].append({"document_id": key, "start": window["start"],
                                                     "reason": extraction.rejection_reason})
        if extraction.research_gaps:
            self.coverage["research_gaps"].append({
                "document_id": key, "role": "reason" if evidence_mode == "inferred" else "extract",
                "gaps": [g.model_dump() for g in extraction.research_gaps],
            })
        self.touch()
        return {"claims": [self.claim_summary(cid) for cid in ids], "invalid_claims": invalid,
                "rejection_reason": extraction.rejection_reason, "document_id": key,
                "research_gaps": [g.model_dump() for g in extraction.research_gaps],
                "read_windows": [{k: w[k] for k in ("document_id", "start", "end")} for w in provided]}

    def evidence_payload(self, record):
        doc = self.documents[record["evidence"].document_id] if record["evidence"] else None
        contexts = {}
        spans = [span for step in record["reasoning_steps"] for span in step.evidence]
        spans.extend(experiment.evidence for experiment in record.get("experimental_evidence", []))
        for span in spans:
            source = self.documents[span.document_id]
            start, end = max(0, span.span_start - 800), min(len(source.text), span.span_end + 800)
            contexts[(span.document_id, start, end)] = {
                "document_id": span.document_id, "title": source.title, "pmid": source.pmid,
                "source_kind": source.source_kind, "start": start, "end": end,
                "text_sha256": source.sha256, "text": source.text[start:end],
            }
        return {"document_id": doc.document_id if doc else None, "title": doc.title if doc else "",
                "model_contract": MODEL_CONTRACT if self.compact_submissions else "legacy_component",
                "pmid": doc.pmid if doc else "", "source_kind": doc.source_kind if doc else "model_knowledge",
                "evidence_context": doc.text[record["start"]:record["end"]] if doc else "",
                "claim": record["draft"].model_dump(),
                "evidence": record["evidence"].model_dump() if record["evidence"] else None,
                "knowledge_source": record["draft"].knowledge_source,
                "hypothesis_assessment": copy.deepcopy(self.workflow.hypotheses.get(record["draft"].hypothesis_id, {}).get("assessment")),
                "reasoning_steps": [s.model_dump() for s in record["reasoning_steps"]],
                "experimental_validation": ExperimentalValidation(evidence=record.get("experimental_evidence", [])).model_dump(),
                "reasoning_source_contexts": list(contexts.values())}

    async def normalize(self, request):
        cid = request["claim_id"]
        self.workflow.require_normalization(cid)
        async with self.claim_locks[cid]:
            if cid in self.retired:
                raise ValueError("claim_is_retired")
            record = self.claims[cid]
            previous = self.claim_summary(cid)
            mapping = await self.engine.normalize(record["draft"], self.audit, context={
                **self.evidence_payload(record), "previous_attempt": previous,
                "focus": request.get("focus", ""),
            })
            record["mapping"] = mapping
            record["revision"] += 1
            # Every remapping requires a fresh review. Never retain a verdict for an older mapping.
            self.observations.pop(cid, None)
            self.touch()
            return self.claim_summary(cid)

    async def review(self, request):
        cid = request["claim_id"]
        self.workflow.require_normalization(cid)
        async with self.claim_locks[cid]:
            if cid in self.retired:
                raise ValueError("claim_is_retired")
            record = self.claims[cid]
            if record["mapping"] is None:
                raise ValueError("normalize_claim_before_review")
            review = await self.engine.review({
                **self.evidence_payload(record), "phenotype": self.phenotype.model_dump(),
                "mapping": record["mapping"].model_dump(), "focus": request.get("focus", ""),
            }, self.audit)
            scope, reasons = release_gate(self.phenotype, record["draft"], record["evidence"],
                                          record["mapping"], review, record["reasoning_steps"])
            self.observations[cid] = Observation(
                claim_id=cid, phenotype_id=self.phenotype.phenotype_id, phenotype_label=self.phenotype.label,
                claim=record["draft"], evidence=record["evidence"], mapping=record["mapping"],
                evidence_mode=record["draft"].evidence_mode, reasoning_steps=record["reasoning_steps"],
                experimental_validation=ExperimentalValidation(evidence=record.get("experimental_evidence", [])),
                review=review, release_scope=scope, gate_reasons=reasons,
            )
            if review.research_gaps:
                self.coverage["research_gaps"].append({"claim_id": cid, "role": "review",
                                                       "gaps": [g.model_dump() for g in review.research_gaps]})
            self.touch()
            return self.claim_summary(cid)

    def subagents(self):
        specs = []
        legacy = (
            ("evidence-extractor", self.extract,
             'Extract directly reported phenotype/cell evidence. JSON: '
             '{"document_id":..., "supporting_document_ids":[], "start":0, "focus":"..."}.'),
            ("evidence-reasoner", self.reason,
             'Infer from model knowledge and optionally literature; no answer seeds. JSON: '
             '{"document_id":"optional source ID", "supporting_document_ids":[], "focus":"..."}. '
             'Empty object uses supplied phenotype and model knowledge.'),
        ) if not self.workflow.enabled else ()
        for name, operation, description in (*legacy,
            ("hypothesis-generator", self.generate_hypotheses,
             'Read literature and propose >=5 initial hypotheses or add new alternatives. JSON: {"focus":"optional","extend_pool":false}.'),
            ("hypothesis-evaluator", self.evaluate_hypothesis,
             ('Evaluate: {"hypothesis_id":...}. The host supplies staged validation sources and assembles scientific submissions. On needs_sources fetch listed sources/search first. On needs_reasoning or invalid_reference with next_action complete_reasoning use {"hypothesis_id":...,"mode":"complete_reasoning","failure_id":...} once for specified fields. Reconsideration requires {"hypothesis_id":...,"mode":"reevaluate","reassessment_reason":"specific reason"}.'
              if self.compact_submissions else
              'Evaluate: {"hypothesis_id":...}. After invalid_format use {"hypothesis_id":...,"mode":"repair","failure_id":...} to patch the preserved draft. Scientific changes require {"hypothesis_id":...,"mode":"reevaluate","reassessment_reason":"specific reason"}.' )),
            ("ontology-normalizer", self.normalize,
             'Multi-turn CL/GO normalization with ontology tools. JSON description: {"claim_id":..., "focus":"..."}.'),
            ("evidence-reviewer", self.review,
             'Review current mapping against source evidence. JSON description: {"claim_id":..., "focus":"..."}.'),
        ):
            def adapter(operation, name):
                async def run(state):
                    request = {}
                    context_key = None
                    try:
                        request = json.loads(state["messages"][-1].content)
                        if not isinstance(request, dict):
                            raise ValueError("task_description_must_be_json_object")
                        if name in {"evidence-extractor", "evidence-reasoner"}:
                            ids = [request.get("document_id"), *request.get("supporting_document_ids", [])]
                            context_key = fingerprint({"subagent": name, "documents": {
                                key: self.documents[key].sha256 for key in ids if key in self.documents
                            }})
                            if context_key in self.failed_evidence_contexts:
                                raise ValueError("evidence_context_already_failed: fetch additional supporting sources "
                                                 "or switch evidence specialist; changing focus alone is not new evidence")
                        async with self.lock:
                            if self.dispatches >= self.settings.max_subagent_calls:
                                raise ValueError("subagent_budget_exhausted")
                            self.dispatches += 1
                        result = copy.deepcopy(await operation(request))
                        target_key = request.get("hypothesis_id", request.get("claim_id", request.get("document_id")))
                        for error in self.errors:
                            previous_request = error["request"]
                            if (error["stage"] == name and isinstance(previous_request, dict)
                                    and previous_request.get("hypothesis_id", previous_request.get("claim_id", previous_request.get("document_id"))) == target_key):
                                error["resolved"] = True
                    except (ValueError, KeyError, TimeoutError, APIConnectionError,
                            APIStatusError, APITimeoutError, GraphRecursionError) as error:
                        detail = str(error) if isinstance(error, ValueError) else type(error).__name__
                        retryable = "budget_exhausted" not in detail and not isinstance(error, GraphRecursionError)
                        if detail.startswith(("no_progress:", "evidence_context_already_failed:")):
                            retryable = False
                        if isinstance(error, HypothesisGenerationFailure):
                            retryable = False
                        if not retryable and context_key:
                            self.failed_evidence_contexts.add(context_key)
                        if isinstance(error, APIStatusError) and error.status_code in {400, 401, 403}:
                            retryable = False
                        result = {"error": detail, "retryable": retryable, "subagent": name}
                        if isinstance(error, ResearchActionRequired):
                            result.update({"status": error.status, "retryable": False, **error.details})
                        elif isinstance(error, (APIConnectionError, APIStatusError, APITimeoutError, TimeoutError)) or "missing structured_response" in detail:
                            result.update({"status": "transport_error", "next_action": "retry_request_preserving_state"})
                        if isinstance(error, SubmissionRepairError):
                            result.update({"status": error.status, "issues": error.issues,
                                           "failed_submission": copy.deepcopy(error.arguments),
                                           "next_action": "correct_fields_before_redispatch"})
                        if isinstance(error, HypothesisGenerationFailure):
                            result.update(error.details)
                        if isinstance(error, AssessmentSubmissionFailure):
                            result.update(error.details)
                        self.errors.append({"stage": name, "request": request, **result})
                    self.audit.append({"role": "dispatch", "subagent": name,
                                       "request": request, "result": result})
                    await self.checkpoint()
                    return {"messages": [AIMessage(to_json(result))], "structured_response": result}
                return run

            specs.append({"name": name, "description": description,
                          "runnable": RunnableLambda(adapter(operation, name))})
        return specs

    def final_coverage(self):
        return {
            **self.coverage,
            "workflow": self.workflow.snapshot(),
            "analyzed_windows": {key: sorted(set(windows)) for key, windows in self.analyzed.items()},
            "deferred_documents": self.deferred, "retired_claims": self.retired,
            "pending": self.pending(), "state_version": self.version,
            "read_windows": {key: sorted(windows) for key, windows in self.read_windows.items()},
            "abstract_only_documents": [key for key, doc in self.documents.items() if doc.source_kind == "abstract"],
            "retracted_documents": [key for key, doc in self.documents.items() if doc.retracted],
            "unreviewed_claims": [
                {"claim_id": key, "claim": value["draft"].model_dump(),
                 "evidence": value["evidence"].model_dump() if value["evidence"] else None,
                 "reasoning_steps": [s.model_dump() for s in value["reasoning_steps"]],
                 "mapping": value["mapping"].model_dump() if value["mapping"] else None}
                for key, value in self.claims.items() if key not in self.retired and key not in self.observations
            ],
        }
