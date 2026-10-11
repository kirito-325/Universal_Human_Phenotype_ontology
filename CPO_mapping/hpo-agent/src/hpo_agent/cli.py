"""CLI emits JSON status; research deliverables are schema-validated JSONL files."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .adapters import import_v6, phenotype_from_legacy
from .background import HPOContextIndex, build_context_index
from .ontology import OntologyIndex, build_index
from .schemas import (
    MODEL_CONTRACT,
    SCHEMA_VERSION,
    AssessmentRepair,
    DiscoveryPlan,
    Document,
    HypothesisAssessment,
    HypothesisBatch,
    HypothesisProposals,
    Observation,
    OntologySelection,
    Phenotype,
    ResearchCompletion,
    ScientificAssessment,
    ScientificCompletion,
    ScientificReview,
    ScientificSupplement,
    Settings,
    TermResult,
)
from .storage import read_jsonl, save_json, save_jsonl


def parser():
    root = argparse.ArgumentParser(description="Human HPO to CL + GO cellular abnormalities")
    commands = root.add_subparsers(dest="command", required=True)
    index = commands.add_parser("build-index", help="Create a versioned CL/GO SQLite index from OBO files")
    index.add_argument("--cl", type=Path, required=True)
    index.add_argument("--go", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    context = commands.add_parser("build-hpo-context", help="Index versioned HPO ontology and disease annotations")
    context.add_argument("--hpo", type=Path, required=True)
    context.add_argument("--annotations", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    lookup = commands.add_parser("hpo-context", help="Inspect scoped HPO disease background without an LLM")
    lookup.add_argument("--index", type=Path, default=Path("data/hpo_context.sqlite"))
    lookup.add_argument("--phenotype-id", required=True)
    lookup.add_argument("--ancestor-depth", type=int, default=3)
    lookup.add_argument("--limit", type=int, default=10)
    search = commands.add_parser("ontology-search", help="Inspect local ontology candidates without an LLM")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--ontology", choices=["CL", "GO"], required=True)
    search.add_argument("--query", required=True)
    schema = commands.add_parser("schemas", help="Export versioned input/output JSON Schemas")
    schema.add_argument("--output", type=Path, default=Path("schemas"))
    convert = commands.add_parser("prepare-terms", help="Convert HPO_link JSONL profiles to current inputs")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    legacy = commands.add_parser("import-v6", help="Import one stored HPO_link_v6 retrieval run")
    legacy.add_argument("--literature-root", type=Path, required=True)
    legacy.add_argument("--hpo-id", required=True)
    legacy.add_argument("--retrieval-run-id", required=True)
    legacy.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run or resume a structured phenotype batch")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--index", type=Path, required=True)
    run.add_argument("--hpo-context-index", type=Path, default=Path("data/hpo_context.sqlite"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--documents", type=Path, help="Local document JSONL with explicit phenotype_ids")
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--settings", type=Path, help="Optional Settings JSON file")
    run.add_argument("--offline", action="store_true", help="Disable literature network retrieval; LLM calls remain enabled")
    run.add_argument("--workers", type=int, help="Concurrent phenotype workers")
    run.add_argument("--max-papers", type=int)
    return root


async def run_batch(args):
    from .agents import AgentEngine
    from .literature import LiteratureClient
    from .pipeline import BatchRunner

    load_dotenv(args.env_file, override=False)
    settings_data = json.loads(args.settings.read_text()) if args.settings else {}
    if args.offline:
        settings_data["offline"] = True
    if args.workers is not None:
        settings_data["term_workers"] = args.workers
    if args.max_papers is not None:
        settings_data["max_papers"] = args.max_papers
    settings = Settings.model_validate(settings_data)
    if settings.workflow_mode != "literature_first":
        raise ValueError("component_test mode is reserved for isolated Python tests, not CLI research runs")
    terms = [Phenotype.model_validate(row) for row in read_jsonl(args.input)]
    documents = [Document.model_validate(row) for row in read_jsonl(args.documents)] if args.documents else []
    ids = {p.phenotype_id for p in terms}
    for document in documents:
        if not document.phenotype_ids or not set(document.phenotype_ids) <= ids:
            raise ValueError(f"{document.document_id}: phenotype_ids must reference this input batch")
    index = OntologyIndex(args.index)
    engine = AgentEngine(index, settings)
    hpo_context = HPOContextIndex(args.hpo_context_index)
    kwargs = dict(engine=engine, index=index, settings=settings, output=args.output, local_documents=documents,
                  hpo_context=hpo_context)
    if settings.offline:
        results = await BatchRunner(**kwargs).run(terms)
    else:
        async with LiteratureClient(args.output / "cache" / "literature") as source:
            results = await BatchRunner(**kwargs, source=source).run(terms)
    print(json.dumps({"output": str(args.output.resolve()), "terms": len(results),
                      "completed": sum(r.status == "completed" for r in results),
                      "partial": sum(r.status == "partial" for r in results),
                      "failed": sum(r.status == "processing_failed" for r in results)}, ensure_ascii=False))
    return 0 if all(r.status == "completed" for r in results) else 2


def main():
    args = parser().parse_args()
    try:
        if args.command == "build-index":
            result = build_index(args.cl, args.go, args.output)
        elif args.command == "build-hpo-context":
            result = build_context_index(args.hpo, args.annotations, args.output)
        elif args.command == "hpo-context":
            result = HPOContextIndex(args.index).lookup(args.phenotype_id, ancestor_depth=args.ancestor_depth, limit=args.limit)
        elif args.command == "ontology-search":
            result = [t.model_dump() for t in OntologyIndex(args.index).search(args.query, args.ontology)]
        elif args.command == "schemas":
            for name, model in (("phenotype", Phenotype), ("document", Document),
                                ("observation", Observation), ("term_result", TermResult),
                                ("discovery_plan", DiscoveryPlan), ("hypothesis_batch", HypothesisBatch),
                                ("hypothesis_assessment", HypothesisAssessment),
                                ("assessment_repair", AssessmentRepair),
                                ("model_hypotheses", HypothesisProposals),
                                ("model_assessment", ScientificAssessment),
                                ("model_scientific_supplement", ScientificSupplement),
                                ("model_mapping", OntologySelection),
                                ("model_review", ScientificReview),
                                ("model_completion", ScientificCompletion),
                                ("research_completion", ResearchCompletion), ("settings", Settings)):
                save_json(args.output / (name + ".schema.json"), model.model_json_schema())
            save_json(args.output / "contracts.json", {
                "schema_version": SCHEMA_VERSION, "model_contract": MODEL_CONTRACT,
                "production_submissions": {"SubmitHypotheses": "model_hypotheses.schema.json",
                    "SubmitAssessment": "model_assessment.schema.json", "SubmitMapping": "model_mapping.schema.json",
                    "Review": "model_review.schema.json", "ResearchCompletion": "model_completion.schema.json"},
                "scientific_completion_submission": {"SubmitAssessment": "model_scientific_supplement.schema.json",
                    "mode": "complete_reasoning", "maximum_attempts": 1},
                "host_records": ["hypothesis_batch.schema.json", "hypothesis_assessment.schema.json",
                                 "observation.schema.json", "term_result.schema.json", "research_completion.schema.json"],
                "legacy_component_only": ["assessment_repair.schema.json"],
            })
            result = {"schemas": str(args.output.resolve())}
        elif args.command == "prepare-terms":
            rows = [phenotype_from_legacy(row) for row in read_jsonl(args.input)]
            save_jsonl(args.output, rows)
            result = {"terms": len(rows), "output": str(args.output)}
        elif args.command == "import-v6":
            result = import_v6(args.literature_root, args.hpo_id, args.retrieval_run_id, args.output)
        else:
            raise SystemExit(asyncio.run(run_batch(args)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from error
