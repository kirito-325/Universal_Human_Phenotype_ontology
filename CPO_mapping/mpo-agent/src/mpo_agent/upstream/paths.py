"""Adapt upstream cache paths to context-local batch storage; no sandbox paths needed."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "source_cache"
_cache: ContextVar[Path] = ContextVar("cell_agent_source_cache", default=DEFAULT_CACHE)
IDLE_TTL_SECONDS = 600
CACHE_ROOTS = (DEFAULT_CACHE / "abstracts", DEFAULT_CACHE / "pmc")


def abstract_cache() -> Path:
    return _cache.get() / "abstracts"


def pmc_cache() -> Path:
    return _cache.get() / "pmc"


@contextmanager
def cache_scope(path: Path):
    token = _cache.set(path)
    try:
        yield
    finally:
        _cache.reset(token)
