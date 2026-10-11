"""Bound repeated evidence-tool work without limiting productive research rounds."""

import json
from collections import Counter
from copy import deepcopy

from .contracts import SubmissionRepairError

SUBMISSION_TOOLS = {"SubmitMapping", "SubmitHypotheses", "SubmitAssessment", "SubmitAssessmentRepair", "Review", "ResearchCompletion"}


def semantic_arguments(value):
    if isinstance(value, dict):
        return {key: semantic_arguments(item) for key, item in value.items()
                if key not in {"rationale", "explanation", "focus"}}
    if isinstance(value, list):
        return [semantic_arguments(item) for item in value]
    return value


class EvidenceProgress:
    def __init__(self, payload, settings, *, submissions_only=False):
        self.settings = settings
        self.calls = {}
        self.results = Counter()
        self.redundant_reads = 0
        self.ranges = {}
        self.submissions = set()
        self.last_read_signature = None
        self.repeated_reads = 0
        self.validation_failures = {}
        self.submissions_only = submissions_only
        if payload.get("document_id") and "window_start" in payload:
            self.ranges[payload["document_id"]] = [(payload["window_start"], payload["window_end"])]
        for window in payload.get("seed_contexts", []):
            self.ranges.setdefault(window["document_id"], []).append((window["start"], window["end"]))

    def observe(self, message):
        if message.type == "ai":
            for call in getattr(message, "tool_calls", []):
                self.calls[call["id"]] = call
            return
        if message.type != "tool":
            return
        call = self.calls.get(getattr(message, "tool_call_id", ""))
        if not call:
            return
        try:
            feedback_object = json.loads(message.content)
        except (ValueError, TypeError):
            feedback_object = {}
        if isinstance(feedback_object, dict) and feedback_object.get("status") in {
            "invalid_format", "invalid_reference", "needs_reasoning", "field_conflict", "scientific_conflict"
        }:
            previous = self.validation_failures.get(call["name"], {})
            current = {}
            for issue in feedback_object["issues"]:
                detail = issue["message"] if issue["code"] in {"value_error", "submission_validation_failed"} else None
                signature = json.dumps([issue["code"], issue["path"], detail], sort_keys=True)
                current[signature] = previous.get(signature, 0) + 1
            self.validation_failures[call["name"]] = current
            if any(count >= self.settings.max_repeated_tool_results for count in current.values()):
                raise SubmissionRepairError(call["name"], feedback_object["issues"], deepcopy(call["args"]),
                                            status=feedback_object["status"])
        if self.submissions_only:
            return
        # Narration changes do not make a repeated scientific proposal new work.
        arguments = json.dumps(semantic_arguments(call["args"]), sort_keys=True)
        if call["name"] in SUBMISSION_TOOLS:
            proposal = (call["name"], arguments)
            if proposal not in self.submissions:
                self.submissions.add(proposal)
                # A new submission earns a bounded citation-repair reading round.
                self.redundant_reads = 0
                self.last_read_signature = None
                self.repeated_reads = 0
        feedback = getattr(message, "status", None) if call["name"] in SUBMISSION_TOOLS else message.content
        signature = json.dumps([call["name"], semantic_arguments(call["args"]), feedback], sort_keys=True)
        if call["name"] == "read_document":
            self.repeated_reads = self.repeated_reads + 1 if signature == self.last_read_signature else 1
            self.last_read_signature = signature
            repeats = self.repeated_reads
        else:
            self.results[signature] += 1
            repeats = self.results[signature]
        if repeats >= self.settings.max_repeated_tool_results:
            raise ValueError(f"no_progress: identical {call['name']} arguments and result repeated "
                             f"{repeats} times; return a gap or obtain new evidence")
        if call["name"] != "read_document":
            return
        try:
            result = json.loads(message.content)
        except (ValueError, TypeError):
            return
        if "text" not in result and "passages" not in result:
            return
        intervals = self.ranges.setdefault(result["document_id"], [])
        start, end = result["start"], result["end"]
        if any(left <= start and end <= right for left, right in intervals):
            self.redundant_reads += 1
        else:
            self.redundant_reads = 0
            intervals.append((start, end))
        if self.redundant_reads >= self.settings.max_redundant_reads:
            raise ValueError("no_progress: repeated read windows contain only previously supplied text; "
                             "obtain another source or return a research gap")
