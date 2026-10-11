import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from hpo_agent.agents import ModelLimit
from hpo_agent.pipeline import safe_error
from hpo_agent.schemas import Settings
from hpo_agent.session import ResearchSession


class RateLimited(Exception):
    status_code = 429
    response = SimpleNamespace(headers={"retry-after": "0"})


async def test_model_requests_share_three_slots_and_retry_429():
    limiter = ModelLimit(asyncio.Semaphore(3), retries=2, retry_base=0)
    active = maximum = 0
    attempts = {}

    async def handler(request):
        nonlocal active, maximum
        attempts[request] = attempts.get(request, 0) + 1
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            if attempts[request] == 1:
                raise RateLimited()
            return request
        finally:
            active -= 1

    result = await asyncio.gather(*(limiter.awrap_model_call(i, handler) for i in range(10)))
    assert result == list(range(10))
    assert maximum == 3
    assert set(attempts.values()) == {2}


async def test_non_rate_errors_are_not_hidden_or_retried():
    limiter = ModelLimit(asyncio.Semaphore(3))
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        raise ValueError("programming error")

    with pytest.raises(ValueError, match="programming"):
        await limiter.awrap_model_call(None, handler)
    assert calls == 1


async def test_model_connection_failure_retries_without_losing_request():
    limiter = ModelLimit(asyncio.Semaphore(3), connection_retries=2, retry_base=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return request

    assert await limiter.awrap_model_call("same input", handler) == "same input"
    assert calls == 3


def test_provider_error_diagnostics_redact_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret-api-key")
    error = RateLimited("request failed")
    error.body = {"error": {"code": "concurrency", "message": "limit for test-secret-api-key"}}
    result = safe_error(error)
    assert result["http_status"] == 429
    assert result["provider_code"] == "concurrency"
    assert "test-secret-api-key" not in str(result)


def test_positive_probe_without_papers_is_incomplete(phenotype, tmp_path):
    session = ResearchSession(phenotype=phenotype, engine=None, settings=Settings(workflow_mode="component_test", ),
                              attempt=tmp_path / "attempt", output=tmp_path)
    session.search_log.append({"query": "test", "count": 13, "pmids": []})
    assert not session.state()["ready_to_finish"]
    assert session.pending()["unfetched_search_hits"] == ["test"]
    session.search_log[0]["count"] = 0
    assert session.state()["ready_to_finish"]
