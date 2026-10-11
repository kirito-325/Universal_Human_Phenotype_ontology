"""Public MPO-only CLI: database preparation, inspection, schemas and research."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .model_background import MPContextIndex, build_mp_context
from .ontology import OntologyIndex, build_index
from .results import MPResult
from .schemas import (
    MODEL_CONTRACT,
    SCHEMA_VERSION,
    Document,
    HypothesisProposals,
    MouseModel,
    MPInput,
    Observation,
    OntologySelection,
    ScientificAssessment,
    ScientificCompletion,
    ScientificReview,
    ScientificSupplement,
    Settings,
)
from .storage import read_jsonl, save_json


def parser():
    root = argparse.ArgumentParser(description="Mouse MP -> MGI models -> cellular mechanisms")
    commands = root.add_subparsers(dest="command", required=True)
    context = commands.add_parser("build-mp-context", help="Index MP OBO and ALL genotype-phenotype annotations")
    context.add_argument("--mp", type=Path, required=True)
    context.add_argument("--annotations", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    lookup = commands.add_parser("mp-context", help="Inspect model annotations without an LLM")
    lookup.add_argument("--index", type=Path, default=Path("data/mp_context.sqlite"))
    lookup.add_argument("--phenotype-id", required=True)
    lookup.add_argument("--include-descendants", action="store_true")
    lookup.add_argument("--offset", type=int, default=0)
    lookup.add_argument("--limit", type=int, default=20)
    index = commands.add_parser("build-index")
    index.add_argument("--cl", type=Path, required=True)
    index.add_argument("--go", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    schemas = commands.add_parser("schemas")
    schemas.add_argument("--output", type=Path, default=Path("schemas"))
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--index", type=Path, default=Path("data/ontology.sqlite"))
    run.add_argument("--mp-context-index", type=Path, default=Path("data/mp_context.sqlite"))
    run.add_argument("--settings", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--offline", action="store_true")
    run.add_argument("--documents", type=Path, help="JSONL {model_ids:[MGI:...],document:{...}}; explicit per-model sources")
    return root


async def run_batch(args):
    from .agents import AgentEngine
    from .literature import LiteratureClient
    from .pipeline import BatchRunner

    load_dotenv(args.env_file, override=False)
    data = json.loads(args.settings.read_text()) if args.settings else {}
    if args.offline:
        data["offline"] = True
    settings = Settings.model_validate(data)
    if settings.workflow_mode != "literature_first":
        raise ValueError("component_test is not a public research mode")
    requests = [MPInput.model_validate(row) for row in read_jsonl(args.input)]
    index, context = OntologyIndex(args.index), MPContextIndex(args.mp_context_index)
    documents = {}
    if args.documents:
        target_ids = {context.lookup(request.phenotype_id, limit=1)["term"]["term_id"] for request in requests}
        known = {model.model_id for request in requests for model in
                 context.all_models(request.phenotype_id, include_descendants=request.include_descendants)[1]}
        for row in read_jsonl(args.documents):
            if set(row) != {"model_ids", "document"} or not row["model_ids"]:
                raise ValueError("local documents require explicit nonempty model_ids and document")
            document = Document.model_validate(row["document"])
            canonical_ids = [context.lookup(key, limit=1)["term"]["term_id"] for key in document.phenotype_ids]
            if not canonical_ids or not set(canonical_ids) <= target_ids:
                raise ValueError("local_document_phenotype_ids_must_reference_the_input_batch")
            document = document.model_copy(update={"phenotype_ids": list(dict.fromkeys(canonical_ids))})
            for model_id in row["model_ids"]:
                if model_id not in known:
                    raise ValueError("local_document_unknown_model")
                documents.setdefault(model_id, []).append(document)
    engine = AgentEngine(index, settings)
    kwargs = dict(engine=engine, index=index, context=context, settings=settings, output=args.output,
                  local_documents=documents)
    if settings.offline:
        results = await BatchRunner(**kwargs).run(requests)
    else:
        async with LiteratureClient(args.output / "cache/literature") as source:
            results = await BatchRunner(**kwargs, source=source).run(requests)
    print(json.dumps({"output": str(args.output.resolve()), "statuses": [result.status for result in results]}))
    return 0 if all(result.status == "completed" for result in results) else 2


def export_schemas(output):
    for name, model in (("mp_input", MPInput), ("mouse_model", MouseModel), ("mp_result", MPResult),
                        ("observation", Observation), ("settings", Settings),
                        ("model_hypotheses", HypothesisProposals), ("model_assessment", ScientificAssessment),
                        ("model_scientific_supplement", ScientificSupplement), ("model_mapping", OntologySelection),
                        ("model_review", ScientificReview), ("model_completion", ScientificCompletion)):
        save_json(output / (name + ".schema.json"), model.model_json_schema())
    save_json(output / "contracts.json", {"schema_version": SCHEMA_VERSION, "model_contract": MODEL_CONTRACT,
        "production_submissions": {"SubmitHypotheses": "model_hypotheses.schema.json",
            "SubmitAssessment": "model_assessment.schema.json", "SubmitMapping": "model_mapping.schema.json",
            "Review": "model_review.schema.json", "ResearchCompletion": "model_completion.schema.json"},
        "scientific_completion_submission": {"SubmitAssessment": "model_scientific_supplement.schema.json",
            "mode": "complete_reasoning", "maximum_attempts": 1},
        "host_records": ["mouse_model.schema.json", "mp_result.schema.json", "observation.schema.json"]})


def main():
    args = parser().parse_args()
    if args.command == "build-mp-context":
        result = build_mp_context(args.mp, args.annotations, args.output)
    elif args.command == "mp-context":
        result = MPContextIndex(args.index).lookup(args.phenotype_id, offset=args.offset, limit=args.limit,
                                                   include_descendants=args.include_descendants)
    elif args.command == "build-index":
        result = build_index(args.cl, args.go, args.output)
    elif args.command == "schemas":
        export_schemas(args.output)
        result = {"schemas": str(args.output.resolve())}
    else:
        raise SystemExit(asyncio.run(run_batch(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
