"""Batch lifecycle and exports around the upstream-derived root agent loop."""

from __future__ import annotations

import asyncio
import fcntl
import os
import uuid
from pathlib import Path

from . import legacy_prompts, prompts
from .schemas import (
    MODEL_CONTRACT,
    SCHEMA_VERSION,
    Document,
    Observation,
    Phenotype,
    Settings,
    TermResult,
)
from .storage import AuditTrail, fingerprint, save_json, save_jsonl


def safe_error(error: Exception) -> dict:
    # Transport/provider exception strings may contain request URLs or credentials.
    detail = str(error)[:1000] if isinstance(error, ValueError) else type(error).__name__
    result = {"type": type(error).__name__, "detail": detail}
    status = getattr(error, "status_code", None)
    if status is not None:
        result["http_status"] = status
        body = getattr(error, "body", {})
        if isinstance(body, dict):
            body = body.get("error", body)
            if isinstance(body, dict):
                message = str(body.get("message", ""))
                for name, value in os.environ.items():
                    if value and len(value) >= 8 and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET")):
                        message = message.replace(value, "[REDACTED]")
                result["provider_code"] = body.get("code")
                result["provider_message"] = message[:1000]
        response = getattr(error, "response", None)
        if response is not None:
            result["retry_after"] = response.headers.get("retry-after")
    return result


def outcome(observations: list[Observation]) -> str:
    scopes = {o.release_scope for o in observations}
    if "primary" in scopes:
        return "projected"
    if scopes & {"conditional", "cross_species"}:
        return "conditional_only"
    if "unresolved" in scopes:
        return "normalization_unresolved"
    return "insufficient_evidence"


class BatchRunner:
    def __init__(self, *, engine, index, settings: Settings, output: Path,
                 source=None, local_documents: list[Document] | None = None, hpo_context=None):
        self.engine = engine
        self.index = index
        self.settings = settings
        self.output = output.resolve()
        self.source = source
        self.hpo_context = hpo_context
        self.local_documents = local_documents or []
        self.term_limit = asyncio.Semaphore(settings.term_workers)
        compact = settings.workflow_mode == "literature_first" and bool(getattr(engine, "compact_submissions", False))
        active_prompts = prompts if compact else legacy_prompts
        self.provenance = {
            "model_contract": MODEL_CONTRACT if compact else "legacy_component",
            "settings": settings.model_dump(), "ontology": index.metadata,
            "models": engine.config,
            "hpo_context": {"metadata": hpo_context.metadata, "sha256": hpo_context.sha256} if hpo_context else None,
            "code_sha256": fingerprint({str(p.relative_to(Path(__file__).parent)): p.read_text()
                                        for p in sorted(Path(__file__).parent.rglob("*.py"))}),
            "prompt_sha256": fingerprint({k: active_prompts.build_root_prompt(settings) if k == "ROOT" else getattr(active_prompts, k) for k in
                                          ("COMMON", "ROOT", "EXTRACT", "REASON", "HYPOTHESIZE",
                                           "VERIFY_HYPOTHESIS", "NORMALIZE", "REVIEW")}),
        }

    async def run(self, terms: list[Phenotype]) -> list[TermResult]:
        ids = [p.phenotype_id for p in terms]
        if not terms or len(ids) != len(set(ids)):
            raise ValueError("input requires unique phenotype IDs and at least one term")
        if self.settings.workflow_mode == "literature_first" and self.hpo_context is None:
            raise ValueError("HPO context index is required before starting a literature-first HPO run")
        self.output.mkdir(parents=True, exist_ok=True)
        # One writer per output root; a crash releases the OS lock automatically.
        with (self.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another batch is writing this output directory") from error
            results = await asyncio.gather(*(self.run_term(p) for p in terms))
            await asyncio.to_thread(self.export, results)
            return results

    def export(self, results: list[TermResult]):
        all_observations = [o for result in results for o in result.observations]
        save_jsonl(self.output / "terms.jsonl", results)
        save_jsonl(self.output / "hypotheses.jsonl", [
            {"phenotype_id": result.phenotype.phenotype_id, **hypothesis}
            for result in results for hypothesis in result.coverage.get("workflow", {}).get("hypotheses", {}).values()
        ])
        nodes = {}
        for observation in all_observations:
            if observation.release_scope not in {"primary", "conditional", "cross_species"}:
                continue
            node = nodes.setdefault(observation.cell_abnormality_id, {
                "cell_abnormality_id": observation.cell_abnormality_id,
                "pair_id": observation.pair_id,
                "cl_id": observation.mapping.cl.term_id, "cl_label": observation.mapping.cl.label,
                "go_id": observation.mapping.go.term_id, "go_label": observation.mapping.go.label,
                "go_namespace": observation.mapping.go.namespace, "direction": observation.claim.direction,
                "claim_ids": [], "phenotype_ids": [], "evidence_species": [], "release_scopes": [],
                "evidence_modes": [],
            })
            node["claim_ids"].append(observation.claim_id)
            for key, value in (("phenotype_ids", observation.phenotype_id),
                               ("evidence_species", observation.claim.species),
                               ("evidence_modes", observation.evidence_mode),
                               ("release_scopes", observation.release_scope)):
                if value not in node[key]:
                    node[key].append(value)
        save_jsonl(self.output / "cell_abnormalities.jsonl", [nodes[key] for key in sorted(nodes)])
        for scope, filename in (
            ("primary", "edges.jsonl"), ("conditional", "conditional_edges.jsonl"),
            ("cross_species", "cross_species_edges.jsonl"), ("unresolved", "unresolved.jsonl"),
            ("rejected", "rejected.jsonl"),
        ):
            save_jsonl(self.output / filename, [o for o in all_observations if o.release_scope == scope])
        accepted = [o for o in all_observations if o.release_scope in {"primary", "conditional", "cross_species"}]
        save_jsonl(self.output / "relations.jsonl", accepted)
        for mode in ("direct", "inferred"):
            save_jsonl(self.output / f"{mode}_edges.jsonl", [o for o in accepted if o.evidence_mode == mode])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **self.provenance,
            "terms": [{"phenotype_id": r.phenotype.phenotype_id, "status": r.status,
                       "outcome": r.outcome, "run_fingerprint": r.run_fingerprint} for r in results],
            "counts": {scope: sum(o.release_scope == scope for o in all_observations)
                       for scope in ("primary", "conditional", "cross_species", "unresolved", "rejected")},
            "accepted_evidence_modes": {mode: sum(o.evidence_mode == mode for o in accepted)
                                        for mode in ("direct", "inferred")},
            "accepted_knowledge_sources": {source: sum(o.knowledge_source == source for o in accepted)
                                           for source in ("literature", "model_knowledge", "mixed")},
            "accepted_experimental_support": {status: sum(o.experimental_validation.status == status for o in accepted)
                                              for status in ("not_recorded", "component_support", "claim_support", "contradictory", "mixed")},
        }
        save_json(self.output / "manifest.json", manifest)

    def resume(self, path: Path, run_key: str) -> TermResult | None:
        if not path.is_file():
            return None
        try:
            result = TermResult.model_validate_json(path.read_text())
            if result.status != "completed" or result.run_fingerprint != run_key:
                return None
            for artifact in result.coverage.get("document_artifacts", []):
                doc = Document.model_validate_json((self.output / artifact["path"]).read_text())
                if doc.sha256 != artifact["text_sha256"]:
                    return None
            audit = self.output / result.coverage["audit_path"]
            if fingerprint(audit.read_text()) != result.coverage["audit_sha256"]:
                return None
            return result
        except (ValueError, OSError, KeyError):
            return None

    async def run_term(self, phenotype: Phenotype) -> TermResult:
        async with self.term_limit:
            local = [d for d in self.local_documents if phenotype.phenotype_id in d.phenotype_ids]
            run_key = fingerprint({**self.provenance, "phenotype": phenotype.model_dump(),
                                   "local_corpus": [d.model_dump() for d in local]})
            term_dir = self.output / "terms" / phenotype.phenotype_id.replace(":", "_") / run_key
            result_path = term_dir / "result.json"
            cached = await asyncio.to_thread(self.resume, result_path, run_key)
            if cached:
                return cached
            attempt = term_dir / ("attempt_" + uuid.uuid4().hex[:12])
            from .session import ResearchSession

            audit, errors = AuditTrail(attempt / "events.jsonl"), []
            session = ResearchSession(phenotype=phenotype, engine=self.engine, settings=self.settings,
                                      attempt=attempt, output=self.output, source=self.source, audit=audit,
                                      hpo_context=self.hpo_context)
            completion = None
            try:
                for document in local:
                    await session.stage(document)
                await session.checkpoint()
                completion = await self.engine.run(session)
                state = session.state()
                if completion.state_version != session.version or not state["ready_to_finish"]:
                    errors.append({"stage": "completion", "type": "IncompleteResearch",
                                   "detail": "Root stopped with pending work or a stale state version",
                                   "pending": state["pending"]})
                errors.extend(e for e in session.errors if not e.get("resolved"))
                status = "partial" if errors else "completed"
            except Exception as error:
                errors.append({"stage": "root", **safe_error(error)})
                status = "partial" if session.observations else "processing_failed"
            observations = session.active_observations()
            term_outcome = outcome(observations) if status != "processing_failed" else "processing_failed"
            await session.checkpoint()
            audit_path = attempt / "audit.jsonl"
            coverage = session.final_coverage()
            coverage["root_completion"] = completion.model_dump() if completion else None
            coverage["attempt_errors"] = session.errors
            coverage["session_path"] = str((attempt / "session.json").relative_to(self.output))
            coverage["audit_path"] = str(audit_path.relative_to(self.output))
            coverage["audit_sha256"] = await asyncio.to_thread(lambda: fingerprint(audit_path.read_text()))
            result = TermResult(
                run_fingerprint=run_key, phenotype=phenotype, status=status, outcome=term_outcome,
                observations=observations, document_ids=list(session.documents),
                search_log=session.search_log, coverage=coverage, errors=errors,
            )
            await asyncio.to_thread(save_json, result_path, result)
            return result
