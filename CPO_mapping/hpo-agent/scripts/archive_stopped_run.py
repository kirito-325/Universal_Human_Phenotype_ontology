"""Create a verified local optimization baseline from a stopped or finished research run.

Copies persisted evidence, audit, source code and ontology snapshots. It never
calls models, reads credentials, resumes the run or invents a final agent result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from hpo_agent.ontology import OntologyIndex, digest_file
from hpo_agent.schemas import Document
from hpo_agent.storage import read_jsonl, save_json, save_jsonl


def archive(args):
    project = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    output = args.output.resolve()
    if run.parent != project / "runs" or not run.is_dir():
        raise ValueError("run must be an existing immediate child of this project's runs/")
    if output.exists() or output.is_relative_to(run):
        raise ValueError("archive output must be new and outside the run directory")
    control = json.loads((run / "run_control.json").read_text())
    stopped = control.get("status") in {"stopped_by_user", "stopped_by_agent_no_progress"}
    finished = control.get("status") in {"completed", "partial", "processing_failed"} and control.get("pipeline_finished") is True
    if not stopped and not finished:
        raise ValueError("stop or finish and mark the run before archiving")
    sessions = list((run / "terms").glob("*/*/attempt_*/session.json"))
    if len(sessions) != 1:
        raise ValueError("this archive command expects one term and one attempt")
    session_path = sessions[0]
    session = json.loads(session_path.read_text())
    audit = read_jsonl(session_path.parent / "audit.jsonl")
    events = read_jsonl(session_path.parent / "events.jsonl")

    output.mkdir(parents=True)
    original_files = {str(p.relative_to(run)): digest_file(p) for p in run.rglob("*") if p.is_file()}
    shutil.copytree(run, output / "run")
    code = output / "code"
    code.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")
    for directory in ("src", "tests", "scripts"):
        shutil.copytree(project / directory, code / directory, ignore=ignore)
    for name in ("pyproject.toml", "README.md", "ARCHITECTURE.md", "UPSTREAM.md", "THIRD_PARTY_NOTICES.md", ".env.example"):
        shutil.copy2(project / name, code / name)
    (output / "inputs").mkdir()
    shutil.copy2(args.input, output / "inputs" / "terms.jsonl")
    shutil.copy2(args.settings, output / "inputs" / "settings.json")
    (output / "ontologies").mkdir()
    shutil.copy2(args.index, output / "ontologies" / "ontology.sqlite")
    if args.hpo:
        shutil.copy2(args.hpo, output / "ontologies" / "hp.obo")
    index = OntologyIndex(output / "ontologies" / "ontology.sqlite")

    findings = output / "analysis"
    findings.mkdir()
    save_jsonl(findings / "search_history.jsonl", session["search_log"])
    save_jsonl(findings / "claims.jsonl", [{"claim_id": cid, **record} for cid, record in session["claims"].items()])
    save_jsonl(findings / "reviews.jsonl", session["observations"])
    save_jsonl(findings / "dispatches.jsonl", [row for row in audit if row.get("role") == "dispatch"])
    save_json(findings / "errors.json", session["errors"])
    save_json(findings / "retired_claims.json", session["retired_claims"])
    save_json(findings / "deferred_documents.json", session["deferred_documents"])

    documents, problems = {}, []
    for artifact in session["coverage"]["document_artifacts"]:
        document = Document.model_validate_json((output / "run" / artifact["path"]).read_text())
        documents[document.document_id] = document
        if document.sha256 != artifact["text_sha256"]:
            problems.append({"document_id": document.document_id, "error": "document_hash_mismatch"})
    mapping_count = 0
    for cid, record in session["claims"].items():
        spans = ([record["evidence"]] if record["evidence"] else []) + [
            span for step in record.get("reasoning_steps", []) for span in step["evidence"]
        ]
        for evidence in spans:
            document = documents.get(evidence["document_id"])
            if (document is None or document.retracted or document.sha256 != evidence["text_sha256"]
                    or document.text[evidence["span_start"]:evidence["span_end"]] != evidence["quote"]):
                problems.append({"claim_id": cid, "error": "quote_or_source_hash_mismatch"})
        if record["mapping"]:
            mapping_count += 1
            for mapped in (record["mapping"]["cl"], record["mapping"]["go"]):
                if mapped:
                    term = index.get(mapped["term_id"])
                    if term is None or term.obsolete or term.label != mapped["label"]:
                        problems.append({"claim_id": cid, "error": "ontology_term_mismatch"})
    for relative, digest in original_files.items():
        if digest_file(output / "run" / relative) != digest or digest_file(run / relative) != digest:
            problems.append({"path": relative, "error": "copy_mismatch_or_source_changed"})
    all_source_now = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()}
    if all_source_now != set(original_files):
        problems.append({"error": "source_files_changed_during_archive"})

    summary = {
        "status": control["status"], "pipeline_completed": control.get("pipeline_completed", False),
        "pipeline_finished": control.get("pipeline_finished", False),
        "created_at_utc": datetime.now(UTC).isoformat(), "stop_control": control,
        "original_run": str(run), "attempt": str(session_path.parent.relative_to(run)),
        "run_fingerprint": session_path.parent.parent.name,
        "last_checkpoint_utc": datetime.fromtimestamp(session_path.stat().st_mtime, UTC).isoformat(),
        "first_event_utc": events[0]["time"], "last_event": events[-1],
        "searches": len(session["search_log"]), "documents": len(documents),
        "unique_pmids": len({d.pmid for d in documents.values() if d.pmid}),
        "source_kinds": dict(Counter(d.source_kind for d in documents.values())),
        "claims": len(session["claims"]), "normalized_claims": mapping_count,
        "reviewed_claims": len(session["observations"]),
        "review_scopes": dict(Counter(o["release_scope"] for o in session["observations"])),
        "retired_claims": len(session["retired_claims"]),
        "active_unreviewed_claims": [cid for cid in session["claims"]
                                    if cid not in session["retired_claims"]
                                    and cid not in {o["claim_id"] for o in session["observations"]}],
        "dispatch_counts": dict(Counter(e["subagent"] for e in audit if e.get("role") == "dispatch")),
        "audit_records": len(audit), "live_events": len(events),
        "integrity_problems": problems, "copied_run_files": len(original_files),
        "ontology_index_sha256": digest_file(output / "ontologies" / "ontology.sqlite"),
        "archive_limits": [
            "Only persisted data is captured; no final result or completed status is fabricated.",
            "If a root invocation was interrupted, its final message/eval-code summary may be absent from audit.",
            "Unreturned model responses and unlogged tool-call details cannot be reconstructed.",
            "Live message detail depends on the archived code version; nested native scout usage may be absent.",
            "Use run/console.log when present for captured stdout/stderr.",
            "This supports offline review and replay of saved evidence, not instruction-level process resume.",
        ],
    }
    save_json(findings / "summary.json", summary)
    names = ("deepagents", "langchain", "langchain-core", "langgraph", "langchain-openai",
             "langchain-quickjs", "openai", "httpx", "pydantic", "python-dotenv")
    save_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "note": "No .env, API credentials or process environment dump is copied.",
    })
    if problems:
        raise ValueError(f"archive integrity check failed: {problems}")
    print(json.dumps({"archive": str(output), "summary": summary}, ensure_ascii=False, indent=2))


def seal(output: Path):
    """Seal the finished human report and data together; verify every copied byte."""
    checksums = {str(p.relative_to(output)): digest_file(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "CHECKSUMS.json"}
    save_json(output / "CHECKSUMS.json", checksums)
    bundle = output.with_name(output.name + ".tar.gz")
    if bundle.exists():
        raise FileExistsError(bundle)
    with tarfile.open(bundle, "w:gz") as stream:
        stream.add(output, arcname=output.name)
    verified = 0
    with tarfile.open(bundle, "r:gz") as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            relative = str(Path(member.name).relative_to(output.name))
            if relative == "CHECKSUMS.json":
                expected = digest_file(output / relative)
            else:
                expected = checksums[relative]
            source = stream.extractfile(member)
            with source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"compressed archive checksum mismatch: {relative}")
            verified += 1
    save_json(bundle.with_name(bundle.name + ".sha256.json"), {
        "file": bundle.name, "sha256": digest_file(bundle), "verified_files": verified,
    })
    print(json.dumps({"bundle": str(bundle), "sha256": digest_file(bundle),
                      "verified_files": verified}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "seal"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--hpo", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        if any(getattr(args, name) is None for name in ("run", "input", "settings", "index")):
            parser.error("capture requires --run, --input, --settings and --index")
        archive(args)
    else:
        seal(args.output.resolve())


if __name__ == "__main__":
    main()
