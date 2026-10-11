"""One Root schedules model-scoped research through the reused eval/task interface."""

import asyncio
import copy
import json
from collections import Counter
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from .contracts import ResearchActionRequired
from .schemas import DiscoveryPlan, Phenotype
from .session import ResearchSession
from .storage import AuditTrail, fingerprint, save_json, to_json

ACCEPTED = {"primary", "conditional", "cross_species"}
FINISHED = {"mechanisms_found", "inconclusive"}
TERMINAL = FINISHED | {"deferred", "needs_sources", "processing_failed"}


class ModelSession(ResearchSession):
    async def fetch(self, pmids, full_text=False):
        model = self.phenotype.model_context
        annotated = sorted(set(pmids) & set(model.annotation_pmids))
        result = await super().fetch(pmids, full_text)
        if annotated:
            self.coverage.setdefault("annotation_retrieval_attempts", []).append({
                "pmids": annotated, "error": result.get("error"),
                "staged_ids": [doc["document_id"] for doc in result.get("documents", []) if "document_id" in doc]})
        return result


class ModelCoordinator:
    is_model_coordinator = True

    def __init__(self, *, request, context, engine, settings, output, attempt, source=None,
                 audit=None, local_documents=None):
        self.request, self.context, self.engine, self.settings = request, context, engine, settings
        self.output, self.attempt, self.source = output, attempt, source
        self.audit = audit if audit is not None else []
        term, models = context.all_models(request.phenotype_id, include_descendants=request.include_descendants)
        self.phenotype = Phenotype(phenotype_id=term["term_id"], label=term["label"],
                                  definition=term["definition"], synonyms=term["synonyms"])
        self.models = {model.model_id: model for model in models}
        self.records = {key: {"status": "pending", "reason": ""} for key in self.models}
        if settings.max_models is not None:
            for key in list(self.models)[settings.max_models:]:
                self.records[key] = {"status": "deferred", "reason": "Configured max_models limit; not researched."}
        self.sessions, self.local_documents = {}, local_documents or {}
        self.workflow = SimpleNamespace(enabled=True)
        self.background_loaded = False
        self.version = self.searches = self.dispatches = 0
        self.lock = asyncio.Lock()
        self.checkpoint_lock = asyncio.Lock()
        self.guard_events = []

    def touch(self):
        self.version += 1

    async def model_session(self, model_id):
        if not self.background_loaded:
            raise ResearchActionRequired("needs_sources", "read_mp_background_before_model_research",
                                         next_action="getPhenotypeBackground")
        if model_id not in self.models:
            raise ResearchActionRequired("invalid_reference", "unknown_model_id", next_action="getPhenotypeBackground")
        if self.records[model_id]["status"] in TERMINAL:
            raise ValueError("model_already_terminal; inspect its recorded result")
        async with self.lock:
            if model_id not in self.sessions:
                model = self.models[model_id]
                phenotype = self.phenotype.model_copy(update={"model_context": model, "seed_pmids": model.annotation_pmids})
                model_attempt = self.attempt / "models" / model_id.replace(":", "_")
                session = ModelSession(phenotype=phenotype, engine=self.engine, settings=self.settings,
                    output=self.output, attempt=model_attempt,
                    source=self.source, audit=AuditTrail(model_attempt / "events.jsonl"))
                session.workflow.background = {"input": phenotype.model_dump(), "model": model.model_dump(),
                    "index_sha256": self.context.sha256,
                    "scope_note": "MGI links this genotype to the MP; it does not establish a cellular mechanism."}
                for document in self.local_documents.get(model_id, []):
                    if phenotype.phenotype_id in document.phenotype_ids:
                        await session.stage(document)
                self.sessions[model_id] = session
                self.records[model_id]["status"] = "researching"
                self.touch()
            return self.sessions[model_id]

    def observations(self):
        return [(key, observation) for key, session in self.sessions.items()
                if self.records[key]["status"] in FINISHED for observation in session.active_observations()]

    def summary_counts(self):
        observations = self.observations()
        accepted = [item for _, item in observations if item.release_scope in ACCEPTED]
        return {"models": dict(Counter(record["status"] for record in self.records.values())),
                "total_annotated_models": len(self.models), "searches": self.searches,
                "model_document_records": sum(len(session.documents) for session in self.sessions.values()),
                "distinct_source_versions": len({(doc.document_id, doc.sha256) for session in self.sessions.values()
                                                for doc in session.documents.values()}),
                "release_scopes": dict(Counter(item.release_scope for _, item in observations)),
                "accepted_relations": len(accepted),
                "experimental_entries": sum(len(item.experimental_validation.evidence) for item in accepted)}

    def state(self, offset=0, limit=20):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid_model_state_pagination")
        pending = [key for key, record in self.records.items() if record["status"] not in TERMINAL]
        keys = list(self.records)[offset:offset + limit]
        return {"state_version": self.version, "phenotype": self.phenotype.model_dump(),
                "summary_counts": self.summary_counts(), "background_loaded": self.background_loaded,
                "models": [{"model_id": key, **self.records[key]} for key in keys],
                "model_offset": offset,
                "next_model_offset": offset + limit if offset + limit < len(self.records) else None,
                "pending": {"background": not self.background_loaded, "models": pending[:limit], "total_models": len(pending)},
                "ready_to_finish": self.background_loaded and not pending,
                "budgets": {"searches_remaining": self.settings.max_total_searches - self.searches,
                            "subagent_calls_remaining": self.settings.max_total_subagent_calls - self.dispatches}}

    async def checkpoint(self):
        async with self.checkpoint_lock:
            snapshot = copy.deepcopy({**self.state(), "model_records": self.records,
                                      "guard_events": self.guard_events, "context_sha256": self.context.sha256})
            await asyncio.to_thread(save_json, self.attempt / "coordinator.json", snapshot)

    async def forward(self, model_id, name, arguments):
        try:
            session = await self.model_session(model_id)
            if name in {"pubmed_search", "search_local_documents"}:
                async with self.lock:
                    if self.searches >= self.settings.max_total_searches:
                        raise ResearchActionRequired("budget_exhausted", "global_search_budget_exhausted")
                    self.searches += 1
            function = next(item for item in session.root_tools() if item.name == name)
            result = await function.ainvoke(arguments)
            self.touch()
            await self.checkpoint()
            return {"model_id": model_id, **result}
        except ValueError as error:
            result = {"error": str(error), "retryable": False, "model_id": model_id}
            if isinstance(error, ResearchActionRequired):
                result.update(status=error.status, **error.details)
            self.guard_events.append(result)
            return result

    async def finish_model(self, model_id):
        session = await self.model_session(model_id)
        state = session.state()
        if not state["ready_to_finish"]:
            return {"error": "model_research_incomplete", "model_id": model_id, "pending": state["pending"]}
        unresolved = [error for error in session.errors if not error.get("resolved")]
        if unresolved:
            return {"error": "resolve_model_task_errors_before_completion", "model_id": model_id, "errors": unresolved}
        reviewed = session.active_observations()
        unmapped = [item.claim_id for item in reviewed if item.release_scope == "unresolved"]
        if unmapped:
            return {"error": "model_normalization_unresolved", "model_id": model_id, "claim_ids": unmapped,
                    "next_action": "revisit_mapping_or_defer_with_an_explicit_ontology_gap"}
        accepted = [item for item in reviewed if item.release_scope in ACCEPTED]
        self.records[model_id] = {"status": "mechanisms_found" if accepted else "inconclusive",
            "reason": "Reviewed mechanisms retained." if accepted else "Research completed without a defensible retained mechanism."}
        self.touch()
        await self.checkpoint()
        return {"model_id": model_id, **self.records[model_id], "accepted_relations": len(accepted)}

    def root_tools(self):
        @tool
        async def get_phenotype_background(offset: int = 0, limit: int = 20) -> dict:
            """Read canonical MP metadata and paginated genotype-level annotations and PMIDs."""
            try:
                result = self.context.lookup(self.request.phenotype_id, include_descendants=self.request.include_descendants,
                                             offset=offset, limit=limit)
                self.background_loaded = True
                self.touch()
                await self.checkpoint()
                return result
            except ValueError as error:
                return {"error": str(error)}

        @tool
        async def get_model_background(model_id: str) -> dict:
            """Read one exact annotated model and its annotation papers before planning."""
            try:
                session = await self.model_session(model_id)
                return {"model": self.models[model_id].model_dump(), "phenotype": session.phenotype.model_dump()}
            except ValueError as error:
                return {"error": str(error)}

        @tool
        async def get_research_state(model_id: str = "", offset: int = 0, limit: int = 20) -> dict:
            """Read all model progress, or one model's hypotheses, documents and diagnostics."""
            if not model_id:
                try:
                    return self.state(offset, limit)
                except ValueError as error:
                    return {"error": str(error)}
            if model_id not in self.models:
                return {"error": "unknown_model_id"}
            session = self.sessions.get(model_id)
            return {"model_id": model_id, **self.records[model_id],
                    "research": session.state() if session else None}

        @tool
        async def register_discovery_plan(model_id: str, plan: DiscoveryPlan) -> dict:
            """Register a model-scoped, diverse literature plan BEFORE proposing mechanisms."""
            return await self.forward(model_id, "register_discovery_plan", {"plan": plan.model_dump()})

        @tool
        async def fetch_model_references(model_id: str) -> dict:
            """Stage annotation papers using host-owned PMIDs; the model does not retype IDs."""
            if model_id not in self.models:
                return {"error": "unknown_model_id"}
            return await self.forward(model_id, "fetch_abstracts", {"pmids": self.models[model_id].annotation_pmids})

        @tool
        async def pubmed_search(model_id: str, term: str, retmax: int = 10, direction_id: str = "",
                                hypothesis_id: str = "", intent: str = "discovery") -> dict:
            """Search model-specific literature; discovery precedes hypotheses, supporting searches follow. Challenge is optional."""
            return await self.forward(model_id, "pubmed_search", dict(term=term, retmax=retmax,
                direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent))

        @tool
        async def search_local_documents(model_id: str, query: str, limit: int = 10, direction_id: str = "",
                                         hypothesis_id: str = "", intent: str = "discovery") -> dict:
            """Search only this model's staged sources; offline alternative to PubMed."""
            return await self.forward(model_id, "search_local_documents", dict(query=query, limit=limit,
                direction_id=direction_id, hypothesis_id=hypothesis_id, intent=intent))

        @tool
        async def fetch_abstracts(model_id: str, pmids: list[str]) -> dict:
            """Stage supplementary sources from returned search PMIDs; cannot guess IDs."""
            return await self.forward(model_id, "fetch_abstracts", {"pmids": pmids})

        @tool
        async def fetch_full_text(model_id: str, pmcids: list[str]) -> dict:
            """Retrieve available full text of staged papers for model identity, methods and results."""
            return await self.forward(model_id, "fetch_full_text", {"pmcids": pmcids})

        @tool
        async def defer_document(model_id: str, document_id: str, reason: str) -> dict:
            """Account for an unused source within one model; does not assert scientific absence."""
            return await self.forward(model_id, "defer_document", {"document_id": document_id, "reason": reason})

        @tool
        async def retire_claim(model_id: str, claim_id: str, reason: str) -> dict:
            """Retire a superseded model-scoped claim using the existing reassessment rules."""
            return await self.forward(model_id, "retire_claim", {"claim_id": claim_id, "reason": reason})

        @tool
        async def finish_model(model_id: str) -> dict:
            """Complete one model after research, hypothesis assessment, mapping, review and source accounting."""
            try:
                return await self.finish_model(model_id)
            except ValueError as error:
                return {"error": str(error), "model_id": model_id}

        @tool
        async def defer_model(model_id: str, reason: str, needs_sources: bool = False) -> dict:
            """Record inaccessible sources or unresearched budget-limited models. These are coverage gaps."""
            if model_id not in self.models or not reason.strip():
                return {"error": "known_model_and_nonempty_reason_required"}
            if self.records[model_id]["status"] in FINISHED:
                return {"error": "completed_model_cannot_be_deferred"}
            self.records[model_id] = {"status": "needs_sources" if needs_sources else "deferred", "reason": reason}
            self.touch()
            await self.checkpoint()
            return {"model_id": model_id, **self.records[model_id]}

        return [get_phenotype_background, get_model_background, get_research_state, register_discovery_plan,
                fetch_model_references, pubmed_search, search_local_documents, fetch_abstracts, fetch_full_text,
                defer_document, retire_claim, finish_model, defer_model]

    def subagents(self):
        specs = []
        for name in ("hypothesis-generator", "hypothesis-evaluator", "ontology-normalizer", "evidence-reviewer"):
            def adapter(role):
                async def run(state):
                    request, model_id = {}, ""
                    try:
                        request = json.loads(state["messages"][-1].content)
                        if not isinstance(request, dict):
                            raise ValueError("task_description_must_be_json_object")
                        model_id = request.pop("model_id", "")
                        session = await self.model_session(model_id)
                        async with self.lock:
                            if self.dispatches >= self.settings.max_total_subagent_calls:
                                raise ResearchActionRequired("budget_exhausted", "global_subagent_budget_exhausted")
                            self.dispatches += 1
                        leaf = next(spec["runnable"] for spec in session.subagents() if spec["name"] == role)
                        response = await leaf.ainvoke({"messages": [HumanMessage(to_json(request))]})
                        result = {"model_id": model_id, **response["structured_response"]}
                    except (ValueError, KeyError) as error:
                        result = {"model_id": model_id, "error": str(error), "retryable": False}
                        if isinstance(error, ResearchActionRequired):
                            result.update(status=error.status, **error.details)
                    self.touch()
                    self.audit.append({"role": "model_dispatch", "subagent": role, "model_id": model_id,
                                       "request": copy.deepcopy(request), "result": result})
                    await self.checkpoint()
                    return {"messages": [AIMessage(to_json(result))], "structured_response": result}
                return run
            specs.append({"name": name, "description": (
                f"{name} for ONE registered mouse model. JSON description MUST contain model_id. "
                "Other fields follow the existing specialist contract: generator focus?/extend_pool?; "
                "evaluator hypothesis_id, mode?/failure_id?/reassessment_reason?; normalizer/reviewer claim_id. "
                "Read sources before hypotheses; validate every hypothesis before normalization."),
                "runnable": RunnableLambda(adapter(name))})
        return specs

    def model_results(self):
        rows = []
        for key, model in self.models.items():
            session = self.sessions.get(key)
            coverage = json.loads(to_json(session.final_coverage())) if session else {}
            if session:
                coverage.update(session_path=str((session.attempt / "session.json").relative_to(self.output)),
                                audit_path=str((session.attempt / "audit.jsonl").relative_to(self.output)))
            rows.append({"model": model.model_dump(), **self.records[key],
                "observations": [item.model_dump() for item in session.active_observations()] if session else [],
                "coverage": coverage,
                "errors": session.errors if session else []})
        return rows

    def mechanism_groups(self):
        groups = {}
        for model_id, observation in self.observations():
            if observation.release_scope not in ACCEPTED:
                continue
            key = (observation.mapping.cl.term_id, observation.mapping.go.term_id, observation.claim.direction)
            group = groups.setdefault(key, {"mechanism_group_id": "mpmech_" + fingerprint(key)[:24],
                "cl_id": key[0], "go_id": key[1], "direction": key[2], "model_ids": [], "members": [],
                "scope_note": "Ontology grouping only. Individual mechanisms and applicability remain model-specific; not universal to all models."})
            if model_id not in group["model_ids"]:
                group["model_ids"].append(model_id)
            group["members"].append({"model_id": model_id, "claim_id": observation.claim_id,
                                    "claim": observation.claim.model_dump()})
        return list(groups.values())
