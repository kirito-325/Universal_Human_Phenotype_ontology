#!/usr/bin/env python3
"""Filter standardized marker statistics into sibling CSV files.

The default balanced policy keeps positive markers (avg_logFC > 0), applies an
adjusted/raw p-value threshold when that statistic is available, and checks that
pct.1 > pct.2 when both detection fractions are present. Missing optional
statistics do not reject a row unless --require-significance is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..paths import DATABASE_DIR

DATABASE = DATABASE_DIR
DEFAULT_INPUT = DATABASE / "papers"
DEFAULT_PATTERN = "*/agent_standardized/marker_statistics.csv"
DEFAULT_RUN_SUMMARY = DATABASE / "marker_statistics_filter_summary.json"
REQUIRED_COLUMNS = (
    "cell_cluster",
    "gene",
    "p_val",
    "avg_logFC",
    "pct.1",
    "pct.2",
    "p_val_adj",
)
NEG_LOG_FDR_RE = re.compile(
    r"(?:-|minus\s*)log\s*10?\s*\(?\s*(?:adjusted\s*p|adj\.?\s*p|fdr|q[ _-]?value)",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def mapped_adjusted_p(extraction: dict[str, Any]) -> bool:
    columns = extraction.get("columns") or {}
    try:
        return int(columns.get("p_val_adj", -1)) >= 0
    except (TypeError, ValueError):
        return False


def detect_adjusted_p_mode(source: Path) -> tuple[str, dict[str, int]]:
    """Use analysis provenance to distinguish p/FDR from -log10(FDR)."""
    analysis = load_json(source.with_name("analysis.json"))
    mapped_tables = 0
    neg_log_tables = 0
    for decision in analysis.get("table_decisions", []):
        if not isinstance(decision, dict):
            continue
        extractions = [
            item
            for item in decision.get("extractions", [])
            if isinstance(item, dict) and mapped_adjusted_p(item)
        ]
        if not extractions:
            continue
        mapped_tables += 1
        evidence = decision.get("evidence") or []
        evidence_text = " ".join(str(item) for item in evidence)
        context = f"{decision.get('reason', '')} {evidence_text}"
        if NEG_LOG_FDR_RE.search(context):
            neg_log_tables += 1

    counts = {
        "mapped_adjusted_p_tables": mapped_tables,
        "neg_log10_fdr_tables": neg_log_tables,
    }
    if mapped_tables and neg_log_tables == mapped_tables:
        return "neglog10", counts
    if neg_log_tables:
        # Rows no longer carry their source table, so mixed transforms cannot be
        # filtered safely at row level.
        return "mixed", counts
    return "pvalue", counts


def parse_number(value: Any) -> tuple[str, float | None]:
    text = str(value or "").strip()
    if not text:
        return "missing", None
    try:
        number = float(text)
    except ValueError:
        return "invalid", None
    if not math.isfinite(number):
        return "invalid", None
    return "valid", number


def decide_row(row: dict[str, Any], args: argparse.Namespace, mode: str) -> tuple[bool, str, str]:
    cluster = str(row.get("cell_cluster") or "").strip()
    gene = str(row.get("gene") or "").strip()
    if not cluster or not gene:
        return False, "missing_cell_or_gene", "none"
    if None in row:
        return False, "malformed_extra_columns", "none"

    logfc_state, logfc = parse_number(row.get("avg_logFC"))
    if logfc_state != "valid" or logfc is None:
        return False, "missing_or_invalid_logfc", "none"
    if logfc <= args.min_logfc:
        return False, "logfc_not_above_threshold", "none"

    significance = "missing"
    adjusted_state, adjusted = parse_number(row.get("p_val_adj"))
    raw_state, raw = parse_number(row.get("p_val"))
    adjusted_is_usable = adjusted_state == "valid" and mode in {"pvalue", "neglog10"}
    if adjusted_is_usable and adjusted is not None:
        if mode == "pvalue":
            if not 0.0 <= adjusted <= 1.0:
                return False, "adjusted_p_out_of_range", "none"
            if adjusted > args.adjusted_p_threshold:
                return False, "adjusted_p_above_threshold", "none"
            significance = "adjusted_p"
        elif mode == "neglog10":
            if adjusted < 0.0:
                return False, "neglog10_fdr_out_of_range", "none"
            cutoff = -math.log10(args.adjusted_p_threshold)
            if adjusted < cutoff:
                return False, "neglog10_fdr_below_threshold", "none"
            significance = "neglog10_fdr"
    elif adjusted_state == "invalid":
        return False, "invalid_adjusted_p", "none"
    elif raw_state == "valid" and raw is not None:
        if not 0.0 <= raw <= 1.0:
            return False, "raw_p_out_of_range", "none"
        if raw > args.raw_p_threshold:
            return False, "raw_p_above_threshold", "none"
        significance = "raw_p"
    elif raw_state == "invalid":
        return False, "invalid_raw_p", "none"
    elif args.require_significance:
        return False, "missing_significance", "none"
    elif adjusted_state == "valid":
        significance = "adjusted_p_ignored" if mode == "ignore" else "mixed_adjusted_p_ignored"

    pct1_state, pct1 = parse_number(row.get("pct.1"))
    pct2_state, pct2 = parse_number(row.get("pct.2"))
    for state, value in ((pct1_state, pct1), (pct2_state, pct2)):
        if state == "invalid":
            return False, "invalid_detection_fraction", significance
        if state == "valid" and value is not None and not 0.0 <= value <= 100.0:
            return False, "detection_fraction_out_of_range", significance

    if pct1_state == pct2_state == "valid" and pct1 is not None and pct2 is not None:
        # pct.1 > pct.2 is invariant between 0-1 and 0-100 scales.
        if not args.ignore_pct_direction and pct1 <= pct2:
            return False, "pct1_not_above_pct2", significance
        normalized_pct1 = pct1 / 100.0 if args.pct_scale == "percent" else pct1
        if args.min_pct1 is not None and normalized_pct1 < args.min_pct1:
            return False, "pct1_below_threshold", significance
    elif args.require_pct_pair:
        return False, "missing_pct_pair", significance

    return True, "kept", significance


def open_atomic_csv(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return handle, writer, temporary


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def process_file(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    output = source.with_name(args.output_name)
    rejected_output = source.with_name(args.rejected_name)
    summary_output = source.with_name(args.file_summary_name)
    if output.resolve() == source.resolve():
        raise ValueError("output name must differ from marker_statistics.csv")

    detected_mode, mode_evidence = detect_adjusted_p_mode(source)
    mode = detected_mode if args.adjusted_p_mode == "auto" else args.adjusted_p_mode
    if args.skip_existing and output.exists():
        return {
            "source": str(source),
            "output": str(output),
            "status": "skipped_existing",
            "adjusted_p_mode": mode,
        }

    input_rows = 0
    kept_rows = 0
    rejected_reasons: Counter[str] = Counter()
    significance_sources: Counter[str] = Counter()
    output_handle = rejected_handle = None
    output_temp = rejected_temp = None
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            fieldnames = list(reader.fieldnames or [])
            missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing:
                raise ValueError(f"missing required columns: {', '.join(missing)}")

            if not args.dry_run:
                output_handle, output_writer, output_temp = open_atomic_csv(output, fieldnames)
                if args.write_rejected:
                    rejected_fields = fieldnames + ["filter_reason"]
                    rejected_handle, rejected_writer, rejected_temp = open_atomic_csv(
                        rejected_output, rejected_fields
                    )

            for row in reader:
                input_rows += 1
                keep, reason, significance = decide_row(row, args, mode)
                if keep:
                    kept_rows += 1
                    significance_sources[significance] += 1
                    if not args.dry_run:
                        output_writer.writerow(row)
                else:
                    rejected_reasons[reason] += 1
                    if not args.dry_run and args.write_rejected:
                        rejected_writer.writerow({**row, "filter_reason": reason})

        if output_handle is not None:
            output_handle.close()
            output_temp.replace(output)
            output.chmod(0o644)
        if rejected_handle is not None:
            rejected_handle.close()
            rejected_temp.replace(rejected_output)
            rejected_output.chmod(0o644)
    except Exception:
        if output_handle is not None and not output_handle.closed:
            output_handle.close()
        if rejected_handle is not None and not rejected_handle.closed:
            rejected_handle.close()
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if rejected_temp is not None:
            rejected_temp.unlink(missing_ok=True)
        raise

    result = {
        "source": str(source),
        "output": str(output),
        "status": "dry_run" if args.dry_run else "written",
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "rejected_rows": input_rows - kept_rows,
        "kept_fraction": round(kept_rows / input_rows, 6) if input_rows else 0.0,
        "adjusted_p_mode": mode,
        "adjusted_p_mode_detected": detected_mode,
        "mode_evidence": mode_evidence,
        "kept_by_significance": dict(significance_sources.most_common()),
        "rejected_by_reason": dict(rejected_reasons.most_common()),
    }
    if detected_mode == "mixed" and args.adjusted_p_mode == "auto":
        result["warning"] = (
            "mixed adjusted-p transforms cannot be assigned to output rows; "
            "adjusted values were not thresholded"
        )
    if not args.dry_run:
        write_json_atomic(summary_output, {"generated_at": utc_now(), "parameters": parameter_summary(args), **result})
    return result


def parameter_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_logfc_exclusive": args.min_logfc,
        "adjusted_p_threshold": args.adjusted_p_threshold,
        "raw_p_threshold": args.raw_p_threshold,
        "adjusted_p_mode_requested": args.adjusted_p_mode,
        "require_significance": args.require_significance,
        "require_pct_pair": args.require_pct_pair,
        "require_pct1_above_pct2": not args.ignore_pct_direction,
        "min_pct1": args.min_pct1,
        "pct_scale": args.pct_scale,
    }


def discover_sources(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    direct = input_path / "marker_statistics.csv"
    if direct.is_file():
        return [direct]
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def positive_probability(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be numeric") from exc
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in (0, 1]")
    return number


def optional_fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fraction must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("fraction must be in [0, 1]")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source CSV or papers directory")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="glob used when --input is a directory")
    parser.add_argument("--output-name", default="marker_statistics_filtered.csv")
    parser.add_argument("--file-summary-name", default="marker_statistics_filter_summary.json")
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--min-logfc", type=float, default=0.0, help="require avg_logFC > this value")
    parser.add_argument("--adjusted-p-threshold", type=positive_probability, default=0.05)
    parser.add_argument("--raw-p-threshold", type=positive_probability, default=0.05)
    parser.add_argument(
        "--adjusted-p-mode",
        choices=("auto", "pvalue", "neglog10", "ignore"),
        default="auto",
        help="auto uses analysis.json to recognize -log10(FDR)",
    )
    parser.add_argument("--require-significance", action="store_true", help="reject rows lacking both p fields")
    parser.add_argument("--require-pct-pair", action="store_true", help="reject rows without both pct fields")
    parser.add_argument("--ignore-pct-direction", action="store_true", help="do not require pct.1 > pct.2")
    parser.add_argument("--min-pct1", type=optional_fraction, help="optional minimum pct.1 as a 0-1 fraction")
    parser.add_argument(
        "--pct-scale",
        choices=("fraction", "percent"),
        default="fraction",
        help="scale for pct columns when --min-pct1 is set; direction checks are scale-independent",
    )
    parser.add_argument("--write-rejected", action="store_true", help="also write rejected rows with a reason")
    parser.add_argument("--rejected-name", default="marker_statistics_rejected.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="process at most N files; 0 means all")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input does not exist: {args.input}")
    sources = discover_sources(args.input, args.pattern)
    if args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"no marker_statistics.csv files matched {args.input} / {args.pattern}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    totals: Counter[str] = Counter()
    for index, source in enumerate(sources, start=1):
        try:
            result = process_file(source, args)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            failures.append({"source": str(source), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(sources)}] ERROR {source}: {exc}")
            continue
        results.append(result)
        totals["files"] += 1
        totals["input_rows"] += int(result.get("input_rows", 0))
        totals["kept_rows"] += int(result.get("kept_rows", 0))
        totals["rejected_rows"] += int(result.get("rejected_rows", 0))
        print(
            f"[{index}/{len(sources)}] {result['status']} {source}: "
            f"kept {result.get('kept_rows', 0)}/{result.get('input_rows', 0)} "
            f"(adjusted-p mode: {result.get('adjusted_p_mode', 'unknown')})"
        )

    run_summary = {
        "generated_at": utc_now(),
        "input": str(args.input),
        "pattern": args.pattern,
        "parameters": parameter_summary(args),
        "totals": dict(totals),
        "failure_count": len(failures),
        "failures": failures,
        "files": results,
    }
    if not args.dry_run:
        write_json_atomic(args.run_summary, run_summary)
    print(json.dumps({"totals": dict(totals), "failures": len(failures)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
