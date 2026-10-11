#!/usr/bin/env python3
"""Classify whether MPO terms describe human-present phenotypes.

The API output is deliberately conservative: only an explicit, high-confidence
``exclude`` decision can remove a term. ``review`` and low-confidence responses
are retained for downstream linkage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


from uhpo.common.llm_client import ECNUClient, api_keys_from_environment

from .paths import ENV_FILE


TERM_ID = re.compile(r"^MP:\d+$")
MOUSE_HINTS = (
    re.compile(r"\btail\b", re.I),
    re.compile(r"\bcaudal vertebra", re.I),
    re.compile(r"\bvibriss", re.I),
    re.compile(r"\bwhisker", re.I),
    re.compile(r"\bbarbering\b", re.I),
    re.compile(r"\bpinna reflex\b", re.I),
)
FIELDS = (
    "mp_id",
    "mp_name",
    "human_relevant",
    "decision",
    "confidence",
    "mouse_specific_features",
    "reason",
    "decision_source",
    "model",
    "batch_id",
    "prefilter_hint",
)


@dataclass(frozen=True)
class Term:
    mp_id: str
    name: str
    definition: str
    synonyms: tuple[str, ...]
    parents: tuple[str, ...]
    obsolete: bool

    @property
    def text(self) -> str:
        return " ".join((self.name, self.definition, *self.synonyms))


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        value = value[1:]
        if '" [' in value:
            value = value.split('" [', 1)[0]
        elif '"' in value:
            value = value.split('"', 1)[0]
    return value.replace("\\\"", '"')


def load_terms(path: Path) -> tuple[list[str], dict[str, Term], dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] = []
    terms: dict[str, Term] = {}
    block: list[str] = []

    def consume(values: list[str]) -> None:
        if not values:
            return
        mp_id = next((x.split(":", 1)[1].strip() for x in values if x.startswith("id: MP:")), "")
        if not TERM_ID.fullmatch(mp_id):
            return
        name = next((x[6:].strip() for x in values if x.startswith("name: ")), mp_id)
        definition = next((x[5:].strip() for x in values if x.startswith("def: ")), "")
        synonyms = tuple(_unquote(x.split(": ", 1)[1]) for x in values if x.startswith("synonym: "))
        parents = tuple(
            x.split(":", 1)[1].split("!", 1)[0].strip()
            for x in values
            if x.startswith("is_a: MP:")
        )
        terms[mp_id] = Term(
            mp_id=mp_id,
            name=name,
            definition=definition,
            synonyms=synonyms,
            parents=parents,
            obsolete=any(x.startswith("is_obsolete: true") for x in values),
        )

    in_term = False
    for line in lines:
        if line == "[Term]":
            if in_term:
                consume(block)
            block = [line]
            in_term = True
        elif in_term:
            if line.startswith("[") and line.endswith("]"):
                consume(block)
                block = []
                in_term = False
                header.append(line)
            else:
                block.append(line)
        else:
            header.append(line)
    if in_term:
        consume(block)
    return header, terms, {term.mp_id: term.name for term in terms.values()}


def hint(term: Term) -> str:
    return ";".join(
        sorted(
            {
                pattern.pattern
                for pattern in MOUSE_HINTS
                # Definitions often mention a developmental tail region while
                # describing a conserved human structure. Use labels/synonyms
                # for the deterministic hint and leave definitions to the API.
                if pattern.search(" ".join((term.name, *term.synonyms)))
            }
        )
    )


def digest_terms(terms: Iterable[Term]) -> str:
    payload = "\n".join(f"{term.mp_id}\t{term.name}\t{term.definition}" for term in terms)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_for(batch: list[Term]) -> list[dict[str, str]]:
    records = [
        {
            "mp_id": term.mp_id,
            "name": term.name,
            "definition": term.definition[:1600],
            "synonyms": "; ".join(term.synonyms[:8]),
            "prefilter_hint": hint(term),
        }
        for term in batch
    ]
    return records


SYSTEM_PROMPT = """You are an ontology curator supporting a mouse-to-human phenotype linkage.
For each Mammalian Phenotype Ontology (MPO) term, decide whether the biological
structure or phenotype itself exists in humans. This is NOT a decision about whether
the experiment was performed in mice and NOT a clinical equivalence judgment.

Keep human-relevant conserved anatomy, physiology, behavior, disease phenotypes,
developmental processes, and cellular/molecular processes. Do not exclude a term
just because it mentions a mouse model, strain, allele, sex, age, or developmental
stage. Exclude only a phenotype whose defining structure or behavior is inherently
absent in humans (for example a mouse tail, vibrissae/whiskers, or a mouse-only
barbering assay). A term that is ambiguous, broad, or plausibly human-relevant must
be retained with decision=review or keep. Be conservative and return JSON only.

Return an array with exactly one object per input item and these keys:
mp_id, human_relevant (boolean), decision (keep|exclude|review),
confidence (high|medium|low), mouse_specific_features (array of strings), reason.
"""


def parse_response(content: str, expected: set[str]) -> dict[str, dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain JSON")
        parsed = json.loads(text[start : end + 1])
        values = parsed.get("decisions", parsed.get("items", [])) if isinstance(parsed, dict) else []
    else:
        values = json.loads(text[start : end + 1])
    if not isinstance(values, list):
        raise ValueError("decision payload is not an array")
    output: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        mp_id = str(item.get("mp_id", "")).strip().upper()
        if mp_id not in expected:
            continue
        decision = str(item.get("decision", "review")).strip().lower()
        confidence = str(item.get("confidence", "low")).strip().lower()
        if decision not in {"keep", "exclude", "review"}:
            decision = "review"
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        output[mp_id] = {
            "human_relevant": bool(item.get("human_relevant", decision != "exclude")),
            "decision": decision,
            "confidence": confidence,
            "mouse_specific_features": ";".join(
                str(x).strip() for x in item.get("mouse_specific_features", []) if str(x).strip()
            ),
            "reason": str(item.get("reason", "")).strip(),
        }
    missing = expected - set(output)
    if missing:
        raise ValueError(f"response omitted {len(missing)} MPO IDs")
    return output


def fallback(term: Term, source: str = "structured_fallback") -> dict[str, Any]:
    matched = hint(term)
    if matched:
        return {
            "human_relevant": False,
            "decision": "exclude",
            "confidence": "high",
            "mouse_specific_features": matched,
            "reason": "The term names a structure or assay that is inherently mouse-specific.",
            "decision_source": source,
        }
    return {
        "human_relevant": True,
        "decision": "review",
        "confidence": "low",
        "mouse_specific_features": "",
        "reason": "No conservative structural mouse-only signal was found; retain pending review.",
        "decision_source": source,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo-obo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://chat.ecnu.edu.cn/open/api/v1")
    parser.add_argument("--model", default="ecnu-max")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Use only conservative structured fallback.")
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild the decision table from complete saved batches without an API key.",
    )
    parser.add_argument(
        "--fallback-on-error",
        action="store_true",
        help="After an API failure, finish remaining batches with conservative fallback.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    _, terms, _ = load_terms(args.mpo_obo)
    active = [term for term in terms.values() if not term.obsolete]
    active.sort(key=lambda term: term.mp_id)
    if args.limit:
        active = active[: args.limit]
    source_digest = digest_terms(active)
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    keys = [] if args.offline or args.aggregate_only else api_keys_from_environment(env_file=ENV_FILE)
    client = None
    if not args.offline and not args.aggregate_only and keys:
        client = ECNUClient(
            keys[0][1],
            endpoint=args.endpoint.rstrip("/") + "/chat/completions",
            model=args.model,
            timeout=600,
            retries=4,
            temperature=0.0,
            max_tokens=8192,
            insecure_tls=True,
        )
    elif not args.offline and not args.aggregate_only:
        raise SystemExit("ECNU_API_KEY is not set; use --offline only for a conservative smoke run.")

    decisions: dict[str, dict[str, Any]] = {}
    term_to_batch: dict[str, str] = {}
    api_successes = 0
    api_failures = 0
    api_failure_kind = ""
    batch_count = (len(active) + args.batch_size - 1) // args.batch_size
    for batch_number, start in enumerate(range(0, len(active), args.batch_size), start=1):
        batch = active[start : start + args.batch_size]
        batch_id = f"batch_{batch_number:05d}"
        batch_path = args.batch_dir / f"{batch_id}.json"
        expected = {term.mp_id for term in batch}
        term_to_batch.update({term_id: batch_id for term_id in expected})
        cached: dict[str, Any] | None = None
        if batch_path.exists() and not args.overwrite:
            try:
                value = json.loads(batch_path.read_text(encoding="utf-8"))
                if value.get("status") == "complete" and value.get("term_ids") == sorted(expected) and value.get("source_digest") == source_digest:
                    cached = value
            except (OSError, json.JSONDecodeError):
                cached = None
        if cached:
            values = cached.get("decisions", {})
            cached_model = str(cached.get("model", args.model))
            decisions.update(
                {
                    term_id: {**value, "_model": cached_model}
                    for term_id, value in values.items()
                }
            )
            # Include resumed batches in the audit totals. Without this, a
            # restart reports only the batches called during the latest
            # process even though cached API responses are part of the final
            # decision table.
            if str(cached.get("model", "")).strip() == "structured-fallback":
                api_failures += 1 if cached.get("api_failure") else 0
            else:
                api_successes += 1
            print(json.dumps({"event": "resumed", "batch": batch_id}, ensure_ascii=True), flush=True)
            continue
        if args.aggregate_only:
            raise SystemExit(
                f"Saved batch {batch_path} is missing, incomplete, or does not match the MPO source digest."
            )
        batch_api_error = ""
        if client:
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(prompt_for(batch), ensure_ascii=False, indent=2),
                    },
                ]
                content, metadata = client.complete(messages)
                parsed = parse_response(content, expected)
                model = str(metadata.get("model", args.model))
                decision_source = "ecnu_api"
                api_successes += 1
            except Exception as exc:
                if not args.fallback_on_error:
                    raise
                api_failures += 1
                api_failure_kind = type(exc).__name__
                batch_api_error = "api_request_or_response_failed"
                # A malformed response or transient request failure should
                # downgrade only this batch. Keep the client alive so later
                # batches get another chance to use the API; disabling it here
                # would silently turn the rest of a large run into fallback.
                parsed = {term.mp_id: fallback(term) for term in batch}
                metadata = {"model": "structured-fallback", "api_error": batch_api_error}
                model = "structured-fallback"
                decision_source = "structured_fallback_after_api_error"
        else:
            parsed = {term.mp_id: fallback(term) for term in batch}
            metadata = {"model": "structured-fallback"}
            model = "structured-fallback"
            decision_source = "structured_fallback"
        for value in parsed.values():
            value["decision_source"] = decision_source
        record = {
            "status": "complete",
            "batch_id": batch_id,
            "term_ids": sorted(expected),
            "source_digest": source_digest,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "metadata": {k: v for k, v in metadata.items() if k != "usage"},
            "api_failure": batch_api_error,
            "decisions": parsed,
        }
        batch_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        decisions.update(
            {term_id: {**value, "_model": model} for term_id, value in parsed.items()}
        )
        print(json.dumps({"event": "complete", "batch": batch_id, "terms": len(batch)}, ensure_ascii=True), flush=True)
        if client:
            time.sleep(0.1)

    rows: list[dict[str, Any]] = []
    for term in sorted(terms.values(), key=lambda value: value.mp_id):
        if term.obsolete:
            value = {
                "human_relevant": False,
                "decision": "exclude",
                "confidence": "high",
                "mouse_specific_features": "obsolete",
                "reason": "Obsolete MPO term.",
                "decision_source": "ontology_obsolete",
            }
        else:
            value = decisions.get(term.mp_id) or fallback(term)
        rows.append(
            {
                "mp_id": term.mp_id,
                "mp_name": term.name,
                **{field: value.get(field, "") for field in FIELDS[2:8]},
                "model": (
                    "ontology"
                    if term.obsolete
                    else str(value.get("_model", args.model if client else "structured-fallback"))
                ),
                "batch_id": "" if term.obsolete else term_to_batch.get(term.mp_id, ""),
                "prefilter_hint": hint(term),
            }
        )
    write_tsv(args.output, rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "aggregate_saved_batches" if args.aggregate_only else ("offline" if args.offline else "api"),
        "source": str(args.mpo_obo.resolve()),
        "source_digest": source_digest,
        "term_count": len(rows),
        "active_term_count": len(active),
        "api_enabled": api_successes > 0,
        "api_called_this_run": client is not None,
        "api_evidence_present": api_successes > 0,
        "api_successful_batches": api_successes,
        "api_failed_batches": api_failures,
        "api_failure_kind": api_failure_kind,
        "decision_counts": {key: sum(row["decision"] == key for row in rows) for key in ("keep", "review", "exclude")},
        "high_confidence_excludes": sum(row["decision"] == "exclude" and row["confidence"] == "high" for row in rows),
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_name("mpo_term_llm_decisions_manifest.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
