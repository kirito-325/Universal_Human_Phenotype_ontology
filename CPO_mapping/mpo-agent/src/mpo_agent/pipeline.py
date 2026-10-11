"""MP batch lifecycle, model provenance, and reviewed mechanism aggregation."""

import asyncio
import fcntl
import os
import uuid
from pathlib import Path

from .coordinator import ACCEPTED, FINISHED, ModelCoordinator
from .results import MPResult, ScopedObservation
from .schemas import MODEL_CONTRACT, SCHEMA_VERSION
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



class BatchRunner:
    def __init__(self, *, engine, index, context, settings, output, source=None, local_documents=None):
        self.engine, self.index, self.context, self.settings = engine, index, context, settings
        self.output, self.source = Path(output).resolve(), source
        self.local_documents = local_documents or {}
        self.provenance = {"schema_version": SCHEMA_VERSION, "model_contract": MODEL_CONTRACT,
            "settings": settings.model_dump(), "ontology_sha256": index.sha256,
            "mp_context": {"sha256": context.sha256, "metadata": context.metadata},
            "models": engine.config, "code_sha256": fingerprint({
                str(path.relative_to(Path(__file__).parent)): path.read_text()
                for path in sorted(Path(__file__).parent.rglob("*.py"))})}

    async def run(self, terms):
        ids = [self.context.lookup(term.phenotype_id, limit=1)["term"]["term_id"] for term in terms]
        if not terms or len(set(ids)) != len(ids):
            raise ValueError("unique_MP_terms_required")
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another_batch_is_writing_this_output") from error
            semaphore = asyncio.Semaphore(self.settings.term_workers)

            async def one(term):
                async with semaphore:
                    return await self.run_term(term)
            results = await asyncio.gather(*(one(term) for term in terms))
            await asyncio.to_thread(self.export, results)
            return results

    async def run_term(self, request):
        corpus = {key: [doc.model_dump() for doc in docs] for key, docs in self.local_documents.items()}
        key = fingerprint({**self.provenance, "request": request.model_dump(), "local_corpus": corpus})
        directory = self.output / "terms" / request.phenotype_id.replace(":", "_") / key
        result_path = directory / "result.json"
        if result_path.is_file():
            try:
                cached = MPResult.model_validate_json(result_path.read_text())
                if cached.status == "completed" and cached.run_fingerprint == key:
                    return cached
            except ValueError:
                pass
        attempt = directory / ("attempt_" + uuid.uuid4().hex[:12])
        audit = AuditTrail(attempt / "events.jsonl")
        session = ModelCoordinator(request=request, context=self.context, engine=self.engine, settings=self.settings,
            output=self.output, attempt=attempt, source=self.source, audit=audit, local_documents=self.local_documents)
        completion, errors = None, []
        try:
            await session.checkpoint()
            completion = await self.engine.run(session)
            if not session.state()["ready_to_finish"] or completion.state_version != session.version:
                errors.append({"stage": "completion", "type": "IncompleteResearch", "pending": session.state()["pending"]})
        except Exception as error:
            errors.append({"stage": "root", **safe_error(error)})
        for child in session.sessions.values():
            await child.checkpoint()
        await session.checkpoint()
        save_jsonl(attempt / "audit.jsonl", audit)
        gaps = any(row["status"] not in FINISHED for row in session.records.values())
        status = ("partial" if errors or gaps else "completed") if completion else "processing_failed"
        result = MPResult(run_fingerprint=key, phenotype=session.phenotype, status=status,
            models=session.model_results(),
            relations=[ScopedObservation(model_id=model_id, observation=item)
                       for model_id, item in session.observations() if item.release_scope in ACCEPTED],
            mechanisms=session.mechanism_groups(), completion=completion, summary_counts=session.summary_counts(),
            context_sha256=self.context.sha256, errors=errors)
        save_json(result_path, result)
        return result

    def export(self, results):
        save_jsonl(self.output / "terms.jsonl", results)
        save_jsonl(self.output / "models.jsonl", [
            {"phenotype_id": result.phenotype.phenotype_id, **model.model_dump()}
            for result in results for model in result.models])
        save_jsonl(self.output / "relations.jsonl", [
            {"model_id": relation.model_id, **relation.observation.model_dump()}
            for result in results for relation in result.relations])
        save_jsonl(self.output / "mechanisms.jsonl", [
            {"phenotype_id": result.phenotype.phenotype_id, **group.model_dump()} for result in results for group in result.mechanisms])
        save_jsonl(self.output / "hypotheses.jsonl", [
            {"phenotype_id": result.phenotype.phenotype_id, "model_id": model.model.model_id, **hypothesis}
            for result in results for model in result.models
            for hypothesis in model.coverage.get("workflow", {}).get("hypotheses", {}).values()])
        save_json(self.output / "manifest.json", {**self.provenance,
            "terms": [{"phenotype_id": result.phenotype.phenotype_id, "status": result.status,
                       "run_fingerprint": result.run_fingerprint, "counts": result.summary_counts} for result in results],
            "coverage_note": "All database models are registered. Deferred/unavailable models are explicit coverage gaps. Ontology groups preserve model-scoped members."})
