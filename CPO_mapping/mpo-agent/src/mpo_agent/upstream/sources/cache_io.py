"""Host-side cache reads and writes, kept off the event loop, with an idle lifetime.

The PubMed and PMC caches were written for `agent.py`, which runs one question and
exits. Blocking the loop there costs nothing, because nothing else is on it.

Under `langgraph dev` the same coroutines run inside an ASGI server, and that changes
the rules twice over. In development, blockbuster raises `BlockingError` on any
blocking syscall made from a coroutine, so a cached `read_text()` doesn't slow the run
down — it kills it. In production nothing raises, which is worse: a synchronous read
stalls every other run in the process, and the health check with them.

So every blocking call goes through `asyncio.to_thread`. The functions here are
deliberately forgiving — a cache is an optimisation, so an unreadable or corrupt entry
reports a miss and the caller refetches, rather than taking down a run over a bad file.

Note the distinction between `MISSING` and `None`: a resolved-to-nothing PMCID is
cached as literal `null`, and that is a real answer meaning "this article has no
objects". Collapsing the two would re-resolve every absent paper on every turn.

## Lifetime

Entries expire on an **idle** TTL — `paths.IDLE_TTL_SECONDS`, the same window that
reaps a thread's sandbox. A hit refreshes the entry's mtime, so a thread that keeps
working keeps its corpus warm for as long as it runs; a thread resumed tomorrow, or a
new thread on a cold machine, refetches. The cache is a within-run optimisation, not a
corpus. That is a deliberate narrowing from what this used to be — permanent until
manually deleted, which meant `data/` grew without bound and a "cached" result could be
arbitrarily old with nothing recording how old.

Expiry is global rather than per-surface. Every surface here has the same shape — a local
copy of something NCBI or S3 will hand back on request — so a per-surface TTL would be a
knob with no question behind it.

Two things follow that are worth knowing:

* **Freshness is checked before the read, not after.** A PMC figure runs to megabytes,
  and one `stat` is much cheaper than reading a file we are about to discard.
* **Only a *successful* read refreshes the entry.** A corrupt file reports a miss and
  keeps its old mtime, so the sweep collects it instead of the touch keeping it alive
  forever.

`sweep()` is what bounds the disk: expiry alone only stops stale entries being *used*,
not stored. It is cheap enough not to think about — a full `stat` walk of 1822 files
measured 10.6ms — so the entry points call `sweep_if_due()` at run start and it
self-gates to once per TTL window.

Set `DEEP_LIFE_SCI_CACHE_TTL` to override: a number of seconds, or `off` to disable
expiry entirely and get the old permanent-cache behaviour. `evals/run.py` sets `off`,
because a sweep whose timing depends on wall clock would add variance to exactly the
numbers evals exist to hold steady.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from mpo_agent.upstream.paths import CACHE_ROOTS, IDLE_TTL_SECONDS

logger = logging.getLogger(__name__)

# Sentinel for "no usable cache entry", distinct from a cached JSON `null`.
MISSING: Any = object()

TTL_ENV = "DEEP_LIFE_SCI_CACHE_TTL"
_DISABLED = frozenset({"off", "never", "none", "0", "-1", ""})

# Last completed sweep, as a monotonic timestamp. Process-local on purpose: the sweep is
# idempotent and cheap, so a second process re-running it wastes ~10ms, while persisting
# the timestamp would mean another file to keep consistent for no gain.
_last_sweep: float | None = None
_sweep_lock = asyncio.Lock()


def ttl_seconds() -> float | None:
    """Idle lifetime for a cache entry, or None when expiry is disabled.

    Read per call rather than at import. `cli.py` and `evals/run.py` both call
    `load_dotenv(override=True)` after this module can already be imported, and
    `paths.py` documents what an import-time env read costs when that
    ordering slips — a value baked in before `.env` was applied, with nothing to show
    it happened.
    """
    raw = os.environ.get(TTL_ENV)
    if raw is None:
        return float(IDLE_TTL_SECONDS)
    if raw.strip().lower() in _DISABLED:
        return None
    try:
        value = float(raw)
    except ValueError:
        # Loud, because the failure is otherwise invisible: the cache keeps working at a
        # TTL the operator didn't ask for and nothing in a trace says which one.
        logger.warning(
            "%s=%r is not a number or 'off' — falling back to %ss",
            TTL_ENV,
            raw,
            IDLE_TTL_SECONDS,
        )
        return float(IDLE_TTL_SECONDS)
    return value if value > 0 else None


def is_fresh(path: Path, ttl: float | None) -> bool:
    """Whether `path` is inside the TTL. A missing file is not fresh, it's a miss."""
    if ttl is None:
        return True
    try:
        return time.time() - path.stat().st_mtime < ttl
    except OSError:
        return False


def touch(path: Path) -> None:
    """Refresh an entry's mtime after a hit — what makes the TTL idle rather than absolute."""
    try:
        os.utime(path)
    except OSError:
        # A cache entry we can't touch is still one we just read successfully. Losing the
        # refresh costs a refetch after the TTL; raising here would cost the whole run.
        logger.debug("could not touch cache entry %s", path, exc_info=True)


def _read_json(path: Path, ttl: float | None) -> Any:
    if not is_fresh(path, ttl):
        return MISSING
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return MISSING
    touch(path)
    return data


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _read_bytes(path: Path, ttl: float | None) -> bytes | None:
    if not is_fresh(path, ttl):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    touch(path)
    return data


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def aread_json(path: Path) -> Any:
    """Fresh cached JSON, or `MISSING` if absent, stale or corrupt. A cached `null` returns None."""
    return await asyncio.to_thread(_read_json, path, ttl_seconds())


async def awrite_json(path: Path, obj: Any) -> None:
    """Write JSON, creating parent directories."""
    return await asyncio.to_thread(_write_json, path, obj)


async def aread_bytes(path: Path) -> bytes | None:
    """Fresh cached bytes, or None if absent, stale or unreadable."""
    return await asyncio.to_thread(_read_bytes, path, ttl_seconds())


async def awrite_bytes(path: Path, data: bytes) -> None:
    """Write bytes, creating parent directories."""
    return await asyncio.to_thread(_write_bytes, path, data)


def sweep(ttl: float | None = None) -> int:
    """Delete cache entries past the TTL. Returns the count. Blocking; call via `to_thread`.

    Scoped to `paths.CACHE_ROOTS`, never to `DATA_DIR` itself — that directory is
    operator-configurable and a sweep that recurses into whatever else lives there would
    delete files this module does not own.

    Empty directories go too, because the PMC cache mirrors the S3 layout one directory
    per versioned package (`data/pmc/PMC5904197.1/`) and 125 empty husks would make the
    thing illegible for exactly the reason that layout was chosen.
    """
    ttl = ttl_seconds() if ttl is None else ttl
    if ttl is None:
        return 0

    removed = 0
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if path.is_dir():
                    # Bottom-up by path depth, so a package directory is tried only after
                    # its contents. Non-empty is the normal case and not an error.
                    path.rmdir()
                elif not is_fresh(path, ttl):
                    path.unlink()
                    removed += 1
            except OSError:
                continue  # raced, non-empty, or not ours to delete — all fine to skip
    if removed:
        logger.info("cache sweep removed %d entries older than %ss", removed, ttl)
    return removed


async def sweep_if_due() -> int:
    """Sweep at most once per TTL window, off the event loop.

    Called by each entry point at run start rather than from the read path: the cost is
    trivial but it is still filesystem I/O, and burying it inside a cache lookup would
    put it in the middle of a fan-out where the latency is least welcome and least
    expected.
    """
    global _last_sweep

    ttl = ttl_seconds()
    if ttl is None:
        return 0

    async with _sweep_lock:
        now = time.monotonic()
        if _last_sweep is not None and now - _last_sweep < ttl:
            return 0
        removed = await asyncio.to_thread(sweep, ttl)
        _last_sweep = time.monotonic()
        return removed
