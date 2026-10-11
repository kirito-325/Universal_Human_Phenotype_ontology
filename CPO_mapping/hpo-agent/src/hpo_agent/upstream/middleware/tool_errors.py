"""Source failures reach the model as a value instead of ending the run.

A PTC tool that raises does not fail one call — it fails the whole run. The exception
leaves `langchain_quickjs`'s host-function bridge, propagates out through `eval` and every
`awrap_tool_call` above it, and errors the graph. Measured on 2026-09-03: the model wrote
`AREA[StudyType]INTERVENTAL` for `INTERVENTIONAL`, one character off in one of a dozen
calls, and a 15-minute run that had already fetched its corpus ended with nothing. Its own
probe call a moment earlier had spelled it correctly.

Catching it in JS is not the fix. The bridge reports a Python exception to QuickJS as the
bare string `"Host function failed"`, so a `try/catch` in the model's code gets a caught
error carrying none of the 400 body — and the body is the entire value here, since these
APIs name the offending token. Verified both halves against the interpreter directly.

So a source failure comes back as a return value, which stays in the JS heap with its
message intact and costs the model nothing until it returns it. `sources/_errors.py` says
why `SourceError` exists to be caught here.

**The payload is `{"error": ...}` and nothing else** — no `count: 0`, no `records: []`.
A failed search that answers in the shape of an empty one is the exact failure this repo
guards hardest against everywhere else (`sources/CLAUDE.md`): it reads as "no such trials
exist" and gets reported as a finding. If the model ignores the `error` key, its next line
throws an ordinary JS `TypeError` on the missing field, which surfaces as a legible eval
error it can fix — a worse outcome than checking, and a far better one than a confident
wrong answer.

Only `SourceError` and transport failures are caught. A `TypeError` in our own code keeps
killing the run loudly rather than reaching the model as a string it will try to route
around.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.tools import BaseTool

from hpo_agent.upstream.sources._errors import SourceError


def with_error_capture(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Copies of `tools` that return `{"error": ...}` rather than raising a source failure.

    Same names, schemas and success-path return values, which is what PTC inspects to
    build the JS signature block.
    """
    return [_wrap(tool) for tool in tools]


def _wrap(tool: BaseTool) -> BaseTool:
    inner = getattr(tool, "coroutine", None)
    if inner is None:  # a sync tool; we have none, and PTC would call it the same way
        return tool

    @functools.wraps(inner)
    async def contained(*args: Any, **kwargs: Any) -> Any:
        try:
            return await inner(*args, **kwargs)
        except SourceError as exc:
            return {"error": f"{tool.name} failed: {exc}"}
        except httpx.HTTPError as exc:
            # Below `_request`'s retry ladder, so this is a transport failure that already
            # survived its backoff — a dead connection, not a 429. Same containment: one
            # unreachable host must not take the fan-out around it with it.
            return {"error": f"{tool.name} failed: {type(exc).__name__}: {exc}"}

    # A copy rather than a subclass, for the same reason `progress.py` uses one: the name,
    # description and args schema are what PTC renders into JS and inspects for injected
    # arguments, so the wrapper must be the same tool in every respect but this one.
    return tool.model_copy(update={"coroutine": contained})
