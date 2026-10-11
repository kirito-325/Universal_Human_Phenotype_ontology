"""Summarize completed term runs and verify artifacts without new model calls."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import run_research  # Loads the same cached dependencies and local packages.
from hpo_agent.schemas import TermResult
from mpo_agent.results import MPResult


def summarize(output):
    control = json.loads((output / "run_control.json").read_text())
    problems, rows = [], []
    for timing in control["terms"]:
        row = dict(timing)
        if timing["status"] in {"queued", "running"}:
            rows.append(row)
            continue
        if not timing["started_at_utc"] or not timing["finished_at_utc"]:
            problems.append({"term": timing["phenotype_id"], "error": "missing_timestamps"})
        else:
            elapsed = (datetime.fromisoformat(timing["finished_at_utc"]) - datetime.fromisoformat(timing["started_at_utc"])).total_seconds()
            if timing["elapsed_seconds"] < 0 or abs(elapsed - timing["elapsed_seconds"]) > 5:
                problems.append({"term": timing["phenotype_id"], "error": "timing_mismatch"})
        term_dir = output / timing["output_directory"]
        attempts = list((term_dir / "terms").glob("*/*/attempt_*"))
        row["attempt_count"] = len(attempts)
        if len(attempts) != 1:
            problems.append({"term": timing["phenotype_id"], "error": "expected_one_attempt", "actual": len(attempts)})
        if not timing["result_path"]:
            row["verification"] = "no_result_artifact"
            rows.append(row)
            continue
        model = TermResult if timing["agent"] == "hpo" else MPResult
        result = model.model_validate_json((output / timing["result_path"]).read_text())
        if result.phenotype.phenotype_id != timing["phenotype_id"] or result.status != timing["status"]:
            problems.append({"term": timing["phenotype_id"], "error": "result_identity_or_status_mismatch"})
        if timing["agent"] == "hpo":
            coverage = [result.coverage]
            row["search_requests"] = len(result.search_log)
        else:
            coverage = [item.coverage for item in result.models]
            row["search_requests"] = result.summary_counts.get("searches", 0)
            row["model_error_count"] = sum(len(item.errors) for item in result.models)
            if len(result.models) != timing["registered_models"]:
                problems.append({"term": timing["phenotype_id"], "error": "model_coverage_count_mismatch"})
        row["hypothesis_count"] = sum(len(item.get("workflow", {}).get("hypotheses", {})) for item in coverage)
        kinds = Counter()
        for item in coverage:
            for artifact in item.get("document_artifacts", []):
                doc = json.loads((term_dir / artifact["path"]).read_text())
                kinds[doc["source_kind"]] += 1
        row["source_kinds"] = dict(kinds)
        row["verification"] = "schema_and_timing_checked"
        rows.append(row)

    finished = [row for row in rows if row["status"] not in {"queued", "running"}]
    report = {"batch_status": control["status"], "term_count": len(rows), "finished_terms": len(finished),
              "status_counts": dict(Counter(row["status"] for row in rows)), "integrity_problems": problems,
              "timing_definition": control["timing_definition"], "whole_term_retries": control["whole_term_retries"],
              "alignment_scoring_performed": False, "terms": rows}
    run_research.save_json(output / "verification.json", report)
    run_research.save_jsonl(output / "term_summaries.jsonl", rows)
    lines = ["# 20 个术语调研与计时", "", f"批次状态：`{control['status']}`；已结束 {len(finished)}/20 项。", "",
             "| 配对 | 术语 | 名称 | 状态 | 耗时（分钟） | 接受关系数 | MP 模型数 |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda value: (value["pair_id"], value["agent"])):
        target = row["result_path"] or row["output_directory"]
        duration = f"{row['elapsed_minutes']:.2f}" if row["elapsed_minutes"] is not None else ""
        lines.append(f"| {row['pair_id']} | [{row['phenotype_id']}]({target}) | {row['label']} | {row['status']} | {duration} | {row['accepted_relations'] if row['accepted_relations'] is not None else ''} | {row['registered_models'] if row['agent'] == 'mpo' else ''} |")
    lines.extend(["", "完整开始/结束时间及秒数见 [timings.tsv](timings.tsv)。时间包含该术语运行中的检索、推理、重试和共享限流等待，不含启动前排队及批次预检。", "",
                  "每个术语只启动一轮；不完整结果、科学上无可保留机制与执行失败应分别解读。MP 的部分模型未完成不能解释为该术语没有细胞机制。", "",
                  f"结果结构、计时及单次 attempt 检查发现 {len(problems)} 个完整性问题，详见 [verification.json](verification.json)。本报告未进行 HPO–MPO 对接评分。", ""])
    run_research.atomic_write(output / "REPORT.md", "\n".join(lines))
    print(json.dumps({key: report[key] for key in ("batch_status", "finished_terms", "status_counts", "integrity_problems")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output.resolve())
