"""Rate limiting and retry shared by `pubmed.py` and `ctgov.py`.

Both talk to a metered public API over httpx, and both had grown their own copy of the
same three pieces: a process-wide pacer, jittered exponential backoff, and a chunker for
batch id lists. The copies had drifted only in constants, so they are parameters here.

**Each source keeps its own `Throttle`.** The pacing state is per instance precisely so
the two buckets stay separate — NCBI and ClinicalTrials.gov meter independently, and a
shared `_last_call` would make a PubMed search delay a registry fetch for no reason.

`_request` itself stays in each module. The two differ in ways that are not incidental:
PubMed needs a POST branch for long id lists, ClinicalTrials.gov surfaces 4xx bodies
verbatim, and they raise different exception types with different messages.

`pmc.py` uses none of this — S3 is not metered the same way, so it caps concurrency with
a semaphore instead.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterator

import httpx

# Retryable everywhere: 429 is the rate limiter, 5xx is the gateway or the origin having
# a moment. Anything else is the caller's problem and must surface.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class Throttle:
    """Serialises requests process-wide to at most one per `min_interval()` seconds.

    The interval is a callable, not a number, because NCBI's depends on whether
    `NCBI_API_KEY` is set — which is read from the environment after import.

    This paces *this process only*. A metered API counts per key or per IP, so a 429
    still arrives whenever something else shares the quota; that is what the retry
    ladder in each caller's `_request` is for.
    """

    def __init__(self, min_interval: Callable[[], float]) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._min_interval() - (time.monotonic() - self._last_call)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()


def retry_after(resp: httpx.Response) -> float | None:
    """Seconds from a `Retry-After` header, if it carries a usable delta-seconds value.

    Returns None for the legal HTTP-date form too — not worth parsing for a delay we
    already have a sane default for. Callers whose API never sends the header (see
    `ctgov.py`) simply don't pass a response.
    """
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def backoff_delay(
    attempt: int,
    *,
    base: float,
    maximum: float,
    resp: httpx.Response | None = None,
) -> float:
    """Seconds to wait before retry `attempt` + 1. A server's `Retry-After` wins.

    Jittered, because the callers that trip a 429 are concurrent by construction: a
    fan-out's requests would otherwise back off in lockstep and collide again.
    """
    if resp is not None:
        after = retry_after(resp)
        if after is not None:
            return min(after, maximum)
    ceiling = min(base * 2 ** (attempt - 1), maximum)
    return random.uniform(ceiling / 2, ceiling)


def chunks(items: list[str], size: int) -> Iterator[list[str]]:
    """Split a batch id list into request-sized pieces."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
