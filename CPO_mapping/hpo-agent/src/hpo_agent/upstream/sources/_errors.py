"""The base class for every source-level failure, and the reason there is one.

Each source raises its own type (`PubMedError`, `PMCError`, `ClinicalTrialsError`) with
its own message, and that stays true — the messages are the useful part. What they now
share is a base, so `middleware/tool_errors.py` can catch *a source failing* without
importing all three and going stale the day a fourth source is added.

Why this exists at all: a PTC tool that raises kills the entire run. The exception leaves
the host-function bridge, propagates out through `eval` and the middleware stack, and ends
the graph — measured on 2026-09-03, when one typo'd enum value (`AREA[StudyType]INTERVENTAL`)
ended a 15-minute run that had already fetched its corpus. Catching it in JS is not a fix
either: the bridge reports a Python exception to QuickJS as the string "Host function
failed", so a `try/catch` around `await tools.ctgovSearch(...)` gets a caught error with
none of the 400 body in it.

That makes raising the wrong shape for anything the model can fix by rewriting its call —
which is most of what these APIs reject. A failure has to come back as a *value* to stay
in the JS heap with its message intact, which is what the wrapper does.

Programming errors are deliberately not part of this hierarchy. A `TypeError` in our own
code should keep killing the run loudly rather than arriving at the model as a string it
will try to work around.
"""

from __future__ import annotations


class SourceError(RuntimeError):
    """A data source refused or failed a request.

    Subclassed, never raised directly. Carries the API's own message wherever there is
    one — these bodies name the offending token, and that is what the model needs to
    repair the call.
    """
