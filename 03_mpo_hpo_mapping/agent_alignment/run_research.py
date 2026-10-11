"""Run each selected term once, with isolated artifacts and persistent timings."""

import argparse
import asyncio
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
AGENT_ROOT = WORKSPACE / "02_phenotype_cpo_mapping"
sys.path[:0] = [str(AGENT_ROOT / name / "src") for name in ("hpo_cpo", "mpo_cpo")]

from dotenv import load_dotenv
from hpo_agent.agents import AgentEngine as HPEngine
from hpo_agent.background import HPOContextIndex
from hpo_agent.literature import LiteratureClient as HPLiterature
from hpo_agent.ontology import OntologyIndex as HPIndex
from hpo_agent.pipeline import BatchRunner as HPRunner
from hpo_agent.schemas import Phenotype
from hpo_agent.schemas import Settings as HPSettings
from hpo_agent.storage import atomic_write, read_jsonl, save_json, save_jsonl
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mpo_agent.agents import AgentEngine as MPEngine
from mpo_agent.literature import LiteratureClient as MPLiterature
from mpo_agent.model_background import MPContextIndex
from mpo_agent.ontology import OntologyIndex as MPIndex
from mpo_agent.pipeline import BatchRunner as MPRunner
from mpo_agent.pipeline import safe_error
from mpo_agent.schemas import MPInput
from mpo_agent.schemas import Settings as MPSettings
from pydantic import BaseModel


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_tables(output, control):
    control["updated_at_utc"] = now()
    control["counts"] = dict(Counter(row["status"] for row in control["terms"]))
    save_json(output / "run_control.json", control)
    save_jsonl(output / "timings.jsonl", control["terms"])
    fields = ["pair_id", "agent", "phenotype_id", "label", "status", "started_at_utc",
              "finished_at_utc", "elapsed_seconds", "elapsed_minutes", "registered_models",
              "accepted_relations", "result_path", "error_count"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(control["terms"])
    atomic_write(output / "timings.tsv", stream.getvalue())


def prepare(output):
    if output.exists():
        raise ValueError("output_already_exists; choose a fresh batch directory")
    hp_inputs, mp_inputs = read_jsonl(ROOT / "inputs/hpo_terms.jsonl"), read_jsonl(ROOT / "inputs/mpo_terms.jsonl")
    hp = {value.phenotype_id: value for value in map(Phenotype.model_validate, hp_inputs)}
    mp = {value.phenotype_id: value for value in map(MPInput.model_validate, mp_inputs)}
    if len(hp) != 10 or len(mp) != 10:
        raise ValueError("expected_ten_unique_terms_per_agent")
    hpo_settings = json.loads((AGENT_ROOT / "hpo_cpo/examples/settings.json").read_text())
    hpo_settings.update(run_mode="research")
    mpo_settings = json.loads((AGENT_ROOT / "mpo_cpo/examples/settings.json").read_text())
    mpo_settings.update(run_mode="research", llm_workers=3, model_request_timeout_seconds=600,
                        model_connection_retries=4, model_rate_limit_retries=8)
    settings = {"hpo": HPSettings.model_validate(hpo_settings), "mpo": MPSettings.model_validate(mpo_settings)}
    pairs = read_jsonl(ROOT / "pairs.jsonl")
    context = MPContextIndex(AGENT_ROOT / "mpo_cpo/data/mp_context.sqlite")
    jobs = []
    for pair in sorted(pairs, key=lambda row: (row["mp_exact_model_count"], row["pair_id"])):
        for kind, value in (("hpo", hp[pair["hp_id"]]), ("mpo", mp[pair["mp_id"]])):
            model_count = context.lookup(value.phenotype_id, limit=1)["total_models"] if kind == "mpo" else 0
            if kind == "mpo" and model_count != pair["mp_exact_model_count"]:
                raise ValueError("model_background_changed_since_pair_selection")
            term_dir = output / kind / value.phenotype_id.replace(":", "_")
            save_jsonl(term_dir / "input.jsonl", [value])
            jobs.append({"pair_id": pair["pair_id"], "agent": kind, "phenotype_id": value.phenotype_id,
                         "label": pair["hp_term" if kind == "hpo" else "mp_term"]["label"],
                         "status": "queued", "started_at_utc": "", "finished_at_utc": "",
                         "elapsed_seconds": None, "elapsed_minutes": None, "registered_models": model_count,
                         "accepted_relations": None, "result_path": "", "error_count": 0,
                         "output_directory": str(term_dir.relative_to(output))})
    for kind, value in settings.items():
        save_json(output / "settings" / (kind + ".json"), value)
    input_paths = [ROOT / "inputs/hpo_terms.jsonl", ROOT / "inputs/mpo_terms.jsonl"]
    indexes = [AGENT_ROOT / name / "data" / filename for name, filename in (
        ("hpo_cpo", "hpo_context.sqlite"), ("hpo_cpo", "ontology.sqlite"),
        ("mpo_cpo", "mp_context.sqlite"), ("mpo_cpo", "ontology.sqlite"))]
    control = {"status": "prepared", "phase": "prepared", "created_at_utc": now(), "pid": None,
               "term_concurrency": 2, "global_model_concurrency": 3, "whole_term_retries": 0,
               "timing_definition": "Monotonic wall time from term start through persisted exports; includes retrieval, model latency and in-run throttling, excludes pre-start queue and batch preflight.",
               "reference_answers_supplied": False, "prior_research_imported": False,
               "source_checksums": {str(path.relative_to(WORKSPACE)): digest(path) for path in [*input_paths, *indexes]},
               "terms": jobs}
    write_tables(output, control)
    return control


async def preflight(output, config):
    result = {"started_at_utc": now(), "models": config["models"]}
    async with HPLiterature(output / "preflight_cache") as source:
        probe = await source.search("cell[tiab]", 0)
        result["pubmed"] = {"ok": True, "count": probe["count"]}

    class ProbeResult(BaseModel):
        token: str

    calls = []

    @tool
    def read_probe_token() -> str:
        """Read the integration token, then return it through ProbeResult."""
        calls.append(True)
        return "alignment-pilot-integration-ok"

    model = ChatOpenAI(model=config["models"]["research"], base_url=config["base_url"],
                       api_key=os.environ["LLM_API_KEY"], use_responses_api=False, timeout=90, max_retries=0)
    graph = create_agent(model=model, tools=[read_probe_token], response_format=ToolStrategy(ProbeResult))
    response = await graph.ainvoke({"messages": [{"role": "user", "content":
        "Call read_probe_token and return its exact token through ProbeResult. This is an integration test."}]},
        config={"recursion_limit": 10})
    if not calls or response["structured_response"].token != "alignment-pilot-integration-ok":
        raise ValueError("model_tool_call_probe_failed")
    result.update(model_tool_call={"ok": True, "calls": len(calls)}, finished_at_utc=now())
    save_json(output / "preflight.json", result)


async def run(output, control):
    for relative, expected in control["source_checksums"].items():
        if digest(WORKSPACE / relative) != expected:
            raise ValueError("prepared_source_changed: " + relative)
    os.environ["LANGSMITH_TRACING"] = "false"
    load_dotenv(WORKSPACE / ".env", override=False)
    hp_settings = HPSettings.model_validate_json((output / "settings/hpo.json").read_text())
    mp_settings = MPSettings.model_validate_json((output / "settings/mpo.json").read_text())
    hp_index, mp_index = HPIndex(AGENT_ROOT / "hpo_cpo/data/ontology.sqlite"), MPIndex(AGENT_ROOT / "mpo_cpo/data/ontology.sqlite")
    hp_context = HPOContextIndex(AGENT_ROOT / "hpo_cpo/data/hpo_context.sqlite")
    mp_context = MPContextIndex(AGENT_ROOT / "mpo_cpo/data/mp_context.sqlite")
    hp_engine, mp_engine = HPEngine(hp_index, hp_settings), MPEngine(mp_index, mp_settings)
    # One provider budget across both engines; scientific sessions remain isolated.
    mp_engine.limit = hp_engine.limit
    control.update(status="running", phase="preflight", pid=os.getpid(), started_at_utc=now(),
                   models=hp_engine.config["models"])
    write_tables(output, control)
    await preflight(output, hp_engine.config)
    control["phase"] = "research"
    write_tables(output, control)
    term_limit, starts = asyncio.Semaphore(control["term_concurrency"]), {}

    async def heartbeat():
        while True:
            for row in control["terms"]:
                if row["status"] == "running":
                    row["elapsed_seconds"] = round(time.perf_counter() - starts[row["phenotype_id"]], 3)
                    row["elapsed_minutes"] = round(row["elapsed_seconds"] / 60, 3)
            write_tables(output, control)
            await asyncio.sleep(15)

    async def one(row):
        async with term_limit:
            term_dir = output / row["output_directory"]
            row.update(status="running", started_at_utc=now())
            starts[row["phenotype_id"]] = time.perf_counter()
            write_tables(output, control)
            print(json.dumps({"event": "term_started", "phenotype_id": row["phenotype_id"], "at": row["started_at_utc"]}), flush=True)
            try:
                inputs = read_jsonl(term_dir / "input.jsonl")
                if row["agent"] == "hpo":
                    async with HPLiterature(term_dir / "cache/literature") as source:
                        result, = await HPRunner(engine=hp_engine, index=hp_index, settings=hp_settings,
                            hpo_context=hp_context, output=term_dir, source=source).run([Phenotype.model_validate(inputs[0])])
                    row["accepted_relations"] = sum(o.release_scope in {"primary", "conditional", "cross_species"} for o in result.observations)
                    row["outcome"] = result.outcome
                else:
                    async with MPLiterature(term_dir / "cache/literature") as source:
                        result, = await MPRunner(engine=mp_engine, index=mp_index, settings=mp_settings,
                            context=mp_context, output=term_dir, source=source).run([MPInput.model_validate(inputs[0])])
                    row["accepted_relations"] = len(result.relations)
                    row["model_status_counts"] = dict(Counter(model.status for model in result.models))
                result_path = term_dir / "terms" / row["phenotype_id"].replace(":", "_") / result.run_fingerprint / "result.json"
                row.update(status=result.status, result_path=str(result_path.relative_to(output)),
                           error_count=len(result.errors), errors=result.errors)
            except asyncio.CancelledError:
                row["status"] = "interrupted"
                raise
            except Exception as error:  # noqa: BLE001 -- Preserve one failed term without cancelling the batch.
                row.update(status="failed", error_count=1, errors=[safe_error(error)])
            finally:
                row["finished_at_utc"] = now()
                row["elapsed_seconds"] = round(time.perf_counter() - starts[row["phenotype_id"]], 3)
                row["elapsed_minutes"] = round(row["elapsed_seconds"] / 60, 3)
                save_json(term_dir / "timing.json", row)
                write_tables(output, control)
                print(json.dumps({"event": "term_finished", "phenotype_id": row["phenotype_id"],
                                  "status": row["status"], "elapsed_minutes": row["elapsed_minutes"]}), flush=True)

    pulse = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(one(row) for row in control["terms"]))
    finally:
        pulse.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse
    control.update(status="completed" if all(row["status"] == "completed" for row in control["terms"])
                   else "finished_with_incomplete_terms", phase="finished", finished_at_utc=now())
    write_tables(output, control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    control = prepare(output) if not output.exists() else json.loads((output / "run_control.json").read_text())
    if args.prepare_only:
        print(json.dumps({"output": str(output), "status": control["status"], "terms": len(control["terms"])}))
        return 0
    with (output / ".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(row["status"] != "queued" for row in control["terms"]):
            raise ValueError("batch_already_started; whole-term retries are disabled")
        with (output / "research.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                def interrupt(_signum, _frame):
                    raise KeyboardInterrupt

                signal.signal(signal.SIGTERM, interrupt)
                asyncio.run(run(output, control))
            except KeyboardInterrupt:
                control.update(status="interrupted", phase="stopped", finished_at_utc=now())
                write_tables(output, control)
            except Exception as error:  # noqa: BLE001 -- Persist setup/provider failures before exiting.
                control.update(status="failed", error=safe_error(error), finished_at_utc=now())
                write_tables(output, control)
    print(json.dumps({"output": str(output), "status": control["status"], "counts": control["counts"]}))
    return 0 if control["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
