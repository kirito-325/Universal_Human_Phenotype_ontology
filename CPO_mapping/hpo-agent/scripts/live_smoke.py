"""Prepare and inspect an external-only 10-HPO integration test.

No existing literature, disease associations, gene lists, CL/GO seeds or curated
PMIDs are read. Only HPO ontology metadata becomes the input to the research agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from hpo_agent.agents import model_configuration
from hpo_agent.literature import LiteratureClient
from hpo_agent.ontology import OntologyIndex, digest_file, quoted
from hpo_agent.schemas import Document, Phenotype, Settings, TermResult
from hpo_agent.storage import read_jsonl, save_json, save_jsonl

HPO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"
SELECTION = {
    "HP:0001873": "hematology: platelet count",
    "HP:0001875": "hematology: neutrophil count",
    "HP:0001250": "neurology",
    "HP:0000407": "hearing",
    "HP:0001252": "muscle function",
    "HP:0001644": "cardiac function",
    "HP:0002206": "pulmonary pathology",
    "HP:0000093": "renal function",
    "HP:0000819": "metabolism",
    "HP:0000939": "skeletal phenotype",
}


def prepare(output: Path):
    if output.exists():
        raise ValueError("Use a new preparation directory")
    response = httpx.get(HPO_URL, follow_redirects=True, timeout=90)
    response.raise_for_status()
    raw = response.text
    terms = {}
    for block in raw.split("[Term]\n")[1:]:
        fields: dict[str, list[str]] = {}
        for line in block.splitlines():
            if line.startswith("["):
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                fields.setdefault(key, []).append(value)
        term_id = fields.get("id", [""])[0]
        if term_id not in SELECTION:
            continue
        if fields.get("is_obsolete") == ["true"]:
            raise ValueError(f"Selected HPO is obsolete: {term_id}")
        terms[term_id] = Phenotype(
            phenotype_id=term_id, label=fields["name"][0],
            definition=quoted(fields.get("def", [""])[0]),
            synonyms=[quoted(s) for s in fields.get("synonym", [])],
        )
    if set(terms) != set(SELECTION):
        raise ValueError("Selected IDs missing from HPO snapshot")
    output.mkdir(parents=True)
    # This is a downloaded source artifact, not a hand-edited implementation file.
    (output / "hp.obo").write_bytes(response.content)
    ordered = [terms[term_id] for term_id in SELECTION]
    save_jsonl(output / "terms.jsonl", ordered)
    settings = Settings(
        max_searches=30, results_per_search=10, max_papers=40, max_full_texts=8,
        max_chunks_per_paper=4, term_workers=1, llm_workers=3,
        timeout_seconds=1800, recursion_limit=100, max_subagent_calls=48,
        eval_timeout_seconds=900, offline=False,
    )
    save_json(output / "settings.json", settings)
    save_json(output / "selection.json", {
        "created_at": datetime.now(UTC).isoformat(), "hpo_url": HPO_URL,
        "hpo_sha256": digest_file(output / "hp.obo"),
        "selection_method": "Purposive coverage of organ systems for integration smoke testing; not a representative benchmark.",
        "terms": [{"phenotype_id": p.phenotype_id, "label": p.label,
                   "selection_reason": SELECTION[p.phenotype_id]} for p in ordered],
        "existing_literature_used": False, "seed_pmids_used": False,
        "disease_contexts_supplied": False, "gene_contexts_supplied": False,
        "ontology_answer_seeds_supplied": False,
    })
    print(json.dumps({"prepared": str(output), "terms": [
        {"id": p.phenotype_id, "label": p.label} for p in ordered]}, ensure_ascii=False))


async def preflight(env_file: Path, output: Path):
    load_dotenv(env_file, override=False)
    os.environ["LANGSMITH_TRACING"] = "false"
    config = model_configuration()
    results = {"model": config["models"]["research"]}
    async with LiteratureClient(output / "source_cache") as source:
        probe = await source.search("thrombocytopenia[tiab]", 0)
        results["pubmed"] = {"ok": True, "count": probe["count"], "warnings": probe["warnings"]}

    class ProbeResult(BaseModel):
        token: str

    calls = []

    @tool
    def read_probe_token() -> str:
        """Read the test token, which must be returned in the structured response."""
        calls.append("read_probe_token")
        return "cell-agent-tool-probe-ok"

    model = ChatOpenAI(model=config["models"]["research"], base_url=config["base_url"],
                       api_key=os.environ["LLM_API_KEY"], use_responses_api=False,
                       timeout=90, max_retries=0)
    graph = create_agent(model=model, tools=[read_probe_token],
                         response_format=ToolStrategy(ProbeResult))
    state = await graph.ainvoke({"messages": [{"role": "user", "content":
        "Call read_probe_token, then return its exact token using ProbeResult. This is an integration test."}]},
        config={"recursion_limit": 10})
    value = state["structured_response"]
    if not calls or value.token != "cell-agent-tool-probe-ok":
        raise ValueError("Model failed real tool-calling/structured-output probe")
    results["model_tool_call"] = {"ok": True, "tool_calls": len(calls)}
    save_json(output / "preflight.json", results)
    print(json.dumps(results, ensure_ascii=False))


def report(output: Path, index_path: Path | None):
    index = OntologyIndex(index_path) if index_path else None
    results = []
    for path in sorted((output / "terms").glob("*/*/result.json")):
        record = TermResult.model_validate_json(path.read_text())
        results.append(record)
    summaries, problems = [], []
    for result in results:
        documents = {}
        for artifact in result.coverage.get("document_artifacts", []):
            doc = Document.model_validate_json((output / artifact["path"]).read_text())
            documents[doc.document_id] = doc
            if artifact["text_sha256"] != doc.sha256:
                problems.append({"term": result.phenotype.phenotype_id, "error": "document_hash"})
        audit_path = result.coverage.get("audit_path")
        audit = read_jsonl(output / audit_path) if audit_path else []
        dispatches = Counter(a["subagent"] for a in audit if a.get("role") == "dispatch")
        usage = Counter()
        for event in audit:
            for entry in event.get("usage", []):
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[k] += entry.get(k, 0)
        root_calls = [call["name"] for event in audit if event.get("role") == "research"
                      for message in event.get("trace", []) for call in message.get("tool_calls", [])]
        for observation in result.observations:
            spans = ([("anchor", observation.evidence)] if observation.evidence else []) + [
                (f"reasoning_step_{i}", evidence)
                for i, step in enumerate(observation.reasoning_steps, 1) for evidence in step.evidence
            ]
            spans.extend((f"experimental_evidence_{i}", item.evidence)
                         for i, item in enumerate(observation.experimental_validation.evidence, 1))
            for location, evidence in spans:
                doc = documents.get(evidence.document_id)
                if (doc is None or doc.retracted or doc.sha256 != evidence.text_sha256
                        or doc.text[evidence.span_start:evidence.span_end] != evidence.quote):
                    problems.append({"claim_id": observation.claim_id, "location": location,
                                     "error": "quote_hash_or_retraction"})
            if index:
                for mapped in (observation.mapping.cl, observation.mapping.go):
                    if mapped:
                        actual = index.get(mapped.term_id)
                        if actual is None or actual.obsolete or actual.label != mapped.label:
                            problems.append({"claim_id": observation.claim_id, "error": "ontology_mapping"})
        summaries.append({
            "phenotype_id": result.phenotype.phenotype_id, "label": result.phenotype.label,
            "status": result.status, "outcome": result.outcome,
            "searches": len(result.search_log),
            "successful_searches": sum("error" not in entry for entry in result.search_log),
            "documents": len(documents),
            "full_text_documents": sum(d.source_kind == "full_text" for d in documents.values()),
            "subagent_dispatches": dict(dispatches), "root_eval_calls": root_calls.count("eval"),
            "hypotheses": dict(Counter(h["status"] for h in result.coverage.get("workflow", {}).get("hypotheses", {}).values())),
            "scopes": dict(Counter(o.release_scope for o in result.observations)),
            "evidence_modes": dict(Counter(o.evidence_mode for o in result.observations)),
            "accepted_evidence_modes": dict(Counter(
                o.evidence_mode for o in result.observations
                if o.release_scope in {"primary", "conditional", "cross_species"}
            )),
            "reasoning_steps": sum(len(o.reasoning_steps) for o in result.observations),
            "reasoning_citations": sum(len(s.evidence) for o in result.observations for s in o.reasoning_steps),
            "experimental_validation": dict(Counter(o.experimental_validation.status for o in result.observations)),
            "distinct_primary_cell_abnormalities": len({
                o.cell_abnormality_id for o in result.observations if o.release_scope == "primary"
            }),
            "distinct_primary_cl_go_pairs": len({
                o.pair_id for o in result.observations if o.release_scope == "primary"
            }),
            "primary_supporting_pmids": sorted({
                o.evidence.pmid for o in result.observations
                if o.release_scope == "primary" and o.evidence and o.evidence.pmid
            }),
            "invalid_claims": len(result.coverage.get("invalid_claims", [])),
            "unreviewed_claims": len(result.coverage.get("unreviewed_claims", [])),
            "usage_reported_by_provider": dict(usage), "errors": result.errors,
            "stop_reason": (result.coverage.get("root_completion") or {}).get("stop_reason"),
        })
    report_value = {"terms_finished": len(results), "terms": summaries,
                    "integrity_problems": problems, "validated_with_ontology_index": bool(index),
                    "usage_note": "Returned agent traces only; does not include unreported failed calls or scout usage."}
    if index:
        save_json(output / "smoke_report.json", report_value)
    print(json.dumps(report_value, ensure_ascii=False, indent=2))


def status(output: Path):
    entries = []
    control_path = output / "run_control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    stop_status = control.get("status", "")
    stopped = stop_status.startswith("stopped_")
    for term_dir in sorted((output / "terms").glob("*")):
        results = sorted(term_dir.glob("*/result.json"), key=lambda p: p.stat().st_mtime)
        if results:
            result = TermResult.model_validate_json(results[-1].read_text())
            entries.append({"term": result.phenotype.phenotype_id, "status": result.status,
                            "outcome": result.outcome, "observations": len(result.observations),
                            "errors": result.errors})
            continue
        snapshots = sorted(term_dir.glob("*/attempt_*/session.json"), key=lambda p: p.stat().st_mtime)
        if not snapshots:
            entries.append({"term": term_dir.name, "status": stop_status if stopped else "waiting_for_first_checkpoint"})
            continue
        data = json.loads(snapshots[-1].read_text())
        events = read_jsonl(snapshots[-1].parent / "audit.jsonl")
        last = events[-1] if events else {}
        live_path = snapshots[-1].parent / "events.jsonl"
        live = None
        validation_failures = []
        if live_path.exists():
            for line in live_path.read_text().splitlines():
                try:
                    live = json.loads(line)
                    if live.get("event") == "submission_validation_failed":
                        validation_failures.append(live)
                except json.JSONDecodeError:
                    continue
        entries.append({
            "term": term_dir.name, "status": stop_status if stopped else "running", "searches": len(data["search_log"]),
            "documents": len(data["coverage"]["document_artifacts"]), "claims": len(data["claims"]),
            "extraction_calls": sum(e.get("role") == "extract" and "output" in e for e in events),
            "reasoning_calls": sum(e.get("role") == "reason" and "output" in e for e in events),
            "hypothesis_generation_calls": sum(e.get("role") == "hypothesize" and "output" in e for e in events),
            "hypothesis_verification_calls": sum(e.get("role") == "verify_hypothesis" and "output" in e for e in events),
            "background_ready": bool(data.get("workflow", {}).get("background")),
            "hypotheses": dict(Counter(h["status"] for h in data.get("workflow", {}).get("hypotheses", {}).values())),
            "no_claim_windows": len(data["coverage"].get("no_claim_windows", [])),
            "mapped": sum(bool(c["mapping"]) for c in data["claims"].values()),
            "observations": len(data["observations"]),
            "last_event": last.get("subagent", last.get("role")),
            "latest_live_step": live,
            "errors": data["errors"],
            "submission_validation_failure_count": len(validation_failures),
            "last_submission_validation_failure": validation_failures[-1] if validation_failures else None,
        })
    print(json.dumps({"output": str(output), "progress": entries}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "report", "status"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("../deep-life-sci/.env"))
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "preflight":
        asyncio.run(preflight(args.env_file, args.output))
    elif args.command == "status":
        status(args.output)
    else:
        report(args.output, args.index)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        from hpo_agent.pipeline import safe_error

        print(json.dumps({"error": safe_error(error)}, ensure_ascii=False))
        raise SystemExit(2) from None
