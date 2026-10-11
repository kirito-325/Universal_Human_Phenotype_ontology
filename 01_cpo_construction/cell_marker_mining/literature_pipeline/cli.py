from __future__ import annotations

from uhpo_common.paths import ENV_FILE

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from .assets import (
    download_assets_for_record,
    extract_tables_for_record,
    is_download_candidate,
    load_manifest,
    probe_record,
)
from .agent_standardization import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    AgentOptions,
    ECNUClient,
    api_key_from_environment,
    run_agent_standardization,
    select_agent_records,
)
from .config import DEFAULT_CONFIG, load_config, resolve_date_range
from .fulltext import (
    has_article_text,
    has_extracted_tables,
    recover_article_text,
    sync_elsevier_content,
)
from .http_client import HttpClient
from .pubmed import PubMedClient, build_pubmed_query, parse_pubmed_article
from .records import build_record_lookup, iter_records, rebuild_index, write_record
from .table_standardization import standardize_records
from .utils import read_json, utc_now, write_json


def cmd_discover(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    start_date, end_date = resolve_date_range(config, args.start_date, args.end_date)
    query = args.query or build_pubmed_query(config, start_date, end_date)

    print(f"PubMed query date range: {start_date} to {end_date}", flush=True)
    print(f"PubMed query:\n{query}\n", flush=True)

    if args.query_out:
        args.query_out.parent.mkdir(parents=True, exist_ok=True)
        args.query_out.write_text(query + "\n", encoding="utf-8")

    if args.dry_run:
        print("Dry run only; no network request was made.", flush=True)
        return 0

    http = HttpClient(email=args.email, sleep=args.sleep, retries=args.retries)
    pubmed = PubMedClient(http=http, api_key=args.api_key)

    pmids = pubmed.search(query, retmax=args.max_results)
    print(f"Found {len(pmids)} PubMed record(s).", flush=True)
    if not pmids:
        rebuild_index(args.database)
        return 0

    written = 0
    record_lookup = build_record_lookup(args.database)
    total_batches = (len(pmids) + args.batch_size - 1) // args.batch_size
    for batch_index, start in enumerate(range(0, len(pmids), args.batch_size), start=1):
        batch = pmids[start : start + args.batch_size]
        print(
            f"Fetching PubMed XML batch {batch_index}/{total_batches} "
            f"({len(batch)} record(s))...",
            flush=True,
        )
        root = pubmed.fetch_xml(batch)
        for article in root.findall(".//PubmedArticle"):
            metadata = parse_pubmed_article(article, config)
            write_record(args.database, metadata, record_lookup=record_lookup)
            written += 1

    indexed = rebuild_index(args.database)
    print(f"Wrote {written} paper record(s).", flush=True)
    print(
        f"Rebuilt index with {indexed} total record(s): {args.database / 'index.jsonl'}",
        flush=True,
    )
    return 0


def cmd_rebuild_index(args: argparse.Namespace) -> int:
    count = rebuild_index(args.database)
    print(f"Rebuilt index with {count} record(s): {args.database / 'index.jsonl'}", flush=True)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    records = list(iter_records(args.database))
    family_counts = Counter()
    year_counts = Counter()
    journal_counts = Counter()
    missing_doi = 0
    missing_pmcid = 0

    for _, metadata in records:
        family_counts[metadata.get("cns_family") or "unknown"] += 1
        year_counts[str(metadata.get("publication_year") or "unknown")] += 1
        journal_counts[metadata.get("journal") or "unknown"] += 1
        if not metadata.get("doi"):
            missing_doi += 1
        if not metadata.get("pmcid"):
            missing_pmcid += 1

    print(f"Total records: {len(records)}")
    print(f"Missing DOI: {missing_doi}")
    print(f"Missing PMCID: {missing_pmcid}")

    print("\nBy CNS family:")
    for key, value in sorted(family_counts.items()):
        print(f"  {key}: {value}")

    print("\nBy publication year:")
    for key, value in sorted(year_counts.items()):
        print(f"  {key}: {value}")

    print(f"\nTop {args.top_journals} journals:")
    for journal, value in journal_counts.most_common(args.top_journals):
        print(f"  {journal}: {value}")
    return 0


def cmd_probe_assets(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    http = HttpClient(
        email=args.email,
        sleep=args.sleep,
        retries=args.retries,
        timeout=args.timeout,
    )
    records = selected_records(args)

    totals = Counter()
    print(f"Probing {len(records)} record(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        title = metadata.get("title", "")
        show_progress = (
            args.progress_every <= 1
            or index == 1
            or index == len(records)
            or index % args.progress_every == 0
        )
        if show_progress:
            print(f"[{index}/{len(records)}] {title}", flush=True)
        probe = probe_record(
            record_dir=record_dir,
            metadata=metadata,
            http=http,
            config=config,
            fetch_publisher=not args.skip_publisher,
            save_text=not args.links_only,
            probe_elsevier_cdn=not args.skip_elsevier_cdn,
            publisher_if_no_assets=args.publisher_if_no_assets,
            fetch_elsevier_publisher=not args.skip_elsevier_publisher,
        )
        write_json(record_dir / "metadata.json", metadata)
        totals["records"] += 1
        totals["records_with_assets"] += int(probe.get("asset_count", 0) > 0)
        totals["assets"] += int(probe.get("asset_count", 0))
        totals["new_assets"] += int(probe.get("new_asset_count", 0))
        totals["records_with_errors"] += int(bool(probe.get("errors")))
        if show_progress:
            print(
                "  pmc={pmc} publisher={publisher} text_chars={chars} assets={assets} errors={errors}".format(
                    pmc=probe.get("pmc_status", ""),
                    publisher=probe.get("publisher_status", ""),
                    chars=probe.get("article_text", {}).get("text_characters", 0),
                    assets=probe.get("asset_count", 0),
                    errors=len(probe.get("errors", [])),
                ),
                flush=True,
            )

    indexed = rebuild_index(args.database)
    print(f"Probe totals: {dict(totals)}", flush=True)
    print(f"Rebuilt index with {indexed} total record(s): {args.database / 'index.jsonl'}", flush=True)
    return 0


def selected_records(args: argparse.Namespace, apply_limit: bool = True):
    records = list(iter_records(args.database))
    cohort = str(getattr(args, "cohort", "") or "").strip()
    exclude_cohort = str(getattr(args, "exclude_cohort", "") or "").strip()
    cns_family = str(getattr(args, "cns_family", "") or "").strip()
    records = filter_records_by_cohort(records, cohort, exclude_cohort)
    if cns_family:
        records = [item for item in records if item[1].get("cns_family") == cns_family]
    stride = getattr(args, "stride", 1) or 1
    offset = getattr(args, "offset", 0) or 0
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    if stride != 1 or offset:
        records = records[offset::stride]
    if getattr(args, "unprobed_only", False):
        records = [item for item in records if not item[1].get("content_probe")]
    if getattr(args, "probe_errors_only", False):
        records = [
            item
            for item in records
            if item[1].get("content_probe", {}).get("errors")
        ]
    if apply_limit and getattr(args, "limit", 0):
        records = records[: args.limit]
    return records


def filter_records_by_cohort(records, cohort: str = "", exclude_cohort: str = ""):
    if cohort:
        records = [
            item
            for item in records
            if cohort in item[1].get("screening", {}).get("discovery_cohorts", [])
        ]
    if exclude_cohort:
        records = [
            item
            for item in records
            if exclude_cohort
            not in item[1].get("screening", {}).get("discovery_cohorts", [])
        ]
    return records


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Process records starting at this zero-based shard offset. Default: 0.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth record for sharded runs. Default: 1.",
    )


def cmd_download_assets(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    extensions = args.extensions or config.get("asset_extensions", [])
    http = HttpClient(
        email=args.email,
        sleep=args.sleep,
        retries=args.retries,
        timeout=args.timeout,
    )
    elsevier_api_key = api_key_from_environment(
        args.elsevier_api_key_env,
        env_file=args.env_file,
    )
    elsevier_inst_token = api_key_from_environment(
        args.elsevier_inst_token_env,
        env_file=args.env_file,
    )
    records = selected_records(args, apply_limit=not args.pending_only)
    if args.pending_only and not args.overwrite:
        records = [
            (record_dir, metadata)
            for record_dir, metadata in records
            if any(
                is_download_candidate(asset, extensions)
                and (
                    asset.get("download_status") not in {"downloaded", "exists"}
                    or not (
                        asset.get("local_path")
                        and (record_dir / str(asset["local_path"])).is_file()
                    )
                )
                for asset in load_manifest(record_dir)
            )
        ]
        if args.limit:
            records = records[: args.limit]

    total_downloaded = 0
    total_failed = 0
    paper_results = []
    print(f"Downloading assets for {len(records)} record(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        show_progress = (
            args.progress_every <= 1
            or index == 1
            or index == len(records)
            or index % args.progress_every == 0
        )
        if show_progress:
            print(f"[{index}/{len(records)}] {metadata.get('title', '')}", flush=True)
        stats = download_assets_for_record(
            record_dir=record_dir,
            http=http,
            extensions=extensions,
            overwrite=args.overwrite,
            elsevier_api_key=elsevier_api_key,
            elsevier_inst_token=elsevier_inst_token,
        )
        metadata["assets"] = load_manifest(record_dir)
        metadata["updated_at"] = utc_now()
        write_json(record_dir / "metadata.json", metadata)
        total_downloaded += stats["downloaded"]
        total_failed += stats["failed"]
        paper_results.append(
            {
                "record": record_dir.name,
                "doi": metadata.get("doi", ""),
                "title": metadata.get("title", ""),
                **stats,
            }
        )
        if show_progress:
            print(
                "  candidates={candidate_count} downloaded={downloaded} skipped={skipped} "
                "failed={failed}".format(**stats),
                flush=True,
            )

    indexed = rebuild_index(args.database)
    summary_name = args.summary_name or "asset_download_summary.json"
    if Path(summary_name).name != summary_name:
        raise ValueError("--summary-name must be a filename, not a path")
    write_json(
        args.database / summary_name,
        {
            "generated_at": utc_now(),
            "papers": len(records),
            "extensions": sorted(set(extensions)),
            "downloaded": total_downloaded,
            "failed": total_failed,
            "paper_results": paper_results,
        },
    )
    print(
        f"Downloaded {total_downloaded} asset file(s), failed {total_failed}. "
        f"Rebuilt index with {indexed} total record(s).",
        flush=True,
    )
    return 0


def cmd_sync_elsevier_content(args: argparse.Namespace) -> int:
    api_key = api_key_from_environment(
        args.elsevier_api_key_env,
        env_file=args.env_file,
    )
    if not api_key:
        print(
            f"Missing API key: set environment variable {args.elsevier_api_key_env}.",
            file=sys.stderr,
        )
        return 2
    inst_token = api_key_from_environment(
        args.elsevier_inst_token_env,
        env_file=args.env_file,
    )
    records = [
        (record_dir, metadata)
        for record_dir, metadata in iter_records(args.database)
        if str(metadata.get("doi", "")).lower().startswith("10.1016/")
        and record_matches_paper_id(record_dir, metadata, args.paper_id)
    ]
    if args.cohort:
        records = [
            item
            for item in records
            if args.cohort in item[1].get("screening", {}).get("discovery_cohorts", [])
        ]
    if args.exclude_cohort:
        records = [
            item
            for item in records
            if args.exclude_cohort
            not in item[1].get("screening", {}).get("discovery_cohorts", [])
        ]
    stride = args.stride or 1
    offset = args.offset or 0
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    records = records[offset::stride]
    if args.unsynced_only:
        records = [item for item in records if not item[1].get("elsevier_content_sync")]
    if args.limit:
        records = records[: args.limit]

    http = HttpClient(
        email=args.email,
        sleep=args.sleep,
        retries=args.retries,
        timeout=args.timeout,
    )
    totals = Counter()
    paper_results = []
    print(f"Synchronizing Elsevier content for {len(records)} paper(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        show_progress = (
            args.progress_every <= 1
            or index == 1
            or index == len(records)
            or index % args.progress_every == 0
        )
        if show_progress:
            print(f"[{index}/{len(records)}] {metadata.get('doi', '')}", flush=True)
        try:
            result = sync_elsevier_content(
                record_dir=record_dir,
                metadata=metadata,
                http=http,
                api_key=api_key,
                inst_token=inst_token,
                refresh_xml=args.refresh_xml,
                overwrite_text=args.overwrite_text,
                assets_only=args.assets_only,
            )
            attachments = result["attachments"]
            totals["complete"] += 1
            totals["xml_fetched"] += int(result["xml_fetched"])
            totals["text_recovered"] += int(
                result["text_status"] == "recovered_elsevier_xml"
            )
            for key in ("discovered", "new", "updated", "already_local"):
                totals[f"attachments_{key}"] += int(attachments[key])
            metadata["assets"] = load_manifest(record_dir)
            metadata["elsevier_content_sync"] = result
            text_result = result.get("text_result", {})
            if text_result:
                metadata["fulltext_recovery"] = text_result
                metadata.setdefault("content_probe", {})["article_text"] = {
                    key: text_result[key]
                    for key in ("text_path", "text_characters", "text_sha256")
                    if text_result.get(key) not in (None, "")
                }
            metadata["updated_at"] = utc_now()
            write_json(record_dir / "metadata.json", metadata)
            paper_results.append(
                {
                    "record": record_dir.name,
                    "doi": metadata.get("doi", ""),
                    "status": "complete",
                    "text_status": result["text_status"],
                    "attachments": attachments,
                }
            )
            if show_progress:
                print(
                    "  text={text} attachments={found} new={new} updated={updated} local={local}".format(
                        text=result["text_status"],
                        found=attachments["discovered"],
                        new=attachments["new"],
                        updated=attachments["updated"],
                        local=attachments["already_local"],
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - continue per paper.
            totals["failed"] += 1
            error = f"{type(exc).__name__}: {exc}"
            paper_results.append(
                {
                    "record": record_dir.name,
                    "doi": metadata.get("doi", ""),
                    "status": "failed",
                    "error": error,
                }
            )
            print(f"  failed: {error}", flush=True)

    summary = {
        "generated_at": utc_now(),
        "papers": len(records),
        **dict(totals),
        "paper_results": paper_results,
    }
    summary_name = args.summary_name or "elsevier_content_sync_summary.json"
    if Path(summary_name).name != summary_name:
        raise ValueError("--summary-name must be a filename, not a path")
    write_json(args.database / summary_name, summary)
    indexed = rebuild_index(args.database)
    print(
        "Elsevier sync complete: papers={papers} complete={complete} failed={failed} "
        "text_recovered={text} attachments_discovered={found} attachments_new={new} "
        "attachments_updated={updated} index_records={indexed}.".format(
            papers=len(records),
            complete=totals["complete"],
            failed=totals["failed"],
            text=totals["text_recovered"],
            found=totals["attachments_discovered"],
            new=totals["attachments_new"],
            updated=totals["attachments_updated"],
            indexed=indexed,
        ),
        flush=True,
    )
    return 0


def cmd_extract_tables(args: argparse.Namespace) -> int:
    records = selected_records(
        args, apply_limit=not args.pending_only and not args.paper_id
    )
    if args.paper_id:
        records = [
            item
            for item in records
            if record_matches_paper_id(item[0], item[1], args.paper_id)
        ]
    if args.pending_only:
        records = [
            item for item in records if not has_complete_table_extraction(item[0])
        ]
        if args.limit:
            records = records[: args.limit]
    elif args.paper_id and args.limit:
        records = records[: args.limit]
    total_tables = 0
    total_failed = 0
    print(f"Extracting tables for {len(records)} record(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {metadata.get('title', '')}", flush=True)
        stats = extract_tables_for_record(record_dir)
        metadata["assets"] = load_manifest(record_dir)
        metadata["updated_at"] = utc_now()
        write_json(record_dir / "metadata.json", metadata)
        total_tables += stats["tables"]
        total_failed += stats["failed"]
        print(
            "  assets={asset_count} parsed_assets={parsed_assets} tables={tables} skipped={skipped} failed={failed}".format(
                **stats
            ),
            flush=True,
        )

    indexed = rebuild_index(args.database)
    print(
        f"Extracted {total_tables} table entry(s), failed {total_failed}. "
        f"Rebuilt index with {indexed} total record(s).",
        flush=True,
    )
    return 0


def has_complete_table_extraction(record_dir: Path) -> bool:
    manifest = read_json(
        record_dir / "extracted" / "tables" / "tables_manifest.json",
        default=None,
    )
    if not isinstance(manifest, list):
        return False
    return all(
        bool(entry.get("output_path"))
        and (record_dir / entry["output_path"]).is_file()
        for entry in manifest
        if isinstance(entry, dict)
    )


def record_matches_paper_id(record_dir: Path, metadata: dict, paper_id: str) -> bool:
    if not paper_id:
        return True
    needle = paper_id.strip().lower()
    values = [
        record_dir.name,
        metadata.get("id", ""),
        metadata.get("doi", ""),
        metadata.get("pmid", ""),
        metadata.get("pmcid", ""),
    ]
    return any(needle == str(value).strip().lower() or needle in str(value).lower() for value in values)


def cmd_recover_article_text(args: argparse.Namespace) -> int:
    invalid_arguments = []
    if args.limit < 0:
        invalid_arguments.append("--limit must be >= 0")
    if args.skip < 0:
        invalid_arguments.append("--skip must be >= 0")
    if args.sleep < 0:
        invalid_arguments.append("--sleep must be >= 0")
    if args.retries < 0:
        invalid_arguments.append("--retries must be >= 0")
    if args.timeout < 1:
        invalid_arguments.append("--timeout must be >= 1")
    if args.max_candidates < 1:
        invalid_arguments.append("--max-candidates must be >= 1")
    summary_name = Path(args.summary_name)
    if (
        not args.summary_name.strip()
        or summary_name.is_absolute()
        or len(summary_name.parts) != 1
        or summary_name.suffix.lower() != ".json"
    ):
        invalid_arguments.append("--summary-name must be one JSON filename")
    if invalid_arguments:
        for message in invalid_arguments:
            print(message, file=sys.stderr)
        return 2

    records = [
        (record_dir, metadata)
        for record_dir, metadata in iter_records(args.database)
        if record_matches_paper_id(record_dir, metadata, args.paper_id)
        and (
            not args.doi_prefix
            or str(metadata.get("doi", "")).lower().startswith(args.doi_prefix.lower())
        )
        and (args.include_without_tables or has_extracted_tables(record_dir))
        and (args.include_existing or args.overwrite or not has_article_text(record_dir))
    ]
    records = filter_records_by_cohort(
        records,
        str(args.cohort or "").strip(),
        str(args.exclude_cohort or "").strip(),
    )
    stride = args.stride or 1
    offset = args.offset or 0
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    records = records[offset::stride]
    if args.skip:
        records = records[args.skip :]
    if args.limit:
        records = records[: args.limit]
    if args.paper_id and not records:
        print(f"No table-bearing paper matched --paper-id {args.paper_id!r}.", file=sys.stderr)
        return 2

    http = HttpClient(
        email=args.email,
        sleep=args.sleep,
        retries=args.retries,
        timeout=args.timeout,
    )
    elsevier_api_key = api_key_from_environment(
        args.elsevier_api_key_env,
        env_file=args.env_file,
    )
    elsevier_inst_token = api_key_from_environment(
        args.elsevier_inst_token_env,
        env_file=args.env_file,
    )
    counts = Counter()
    record_results = []
    print(f"Recovering article text for {len(records)} paper(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {metadata.get('title', '')}", flush=True)
        result = recover_article_text(
            record_dir=record_dir,
            metadata=metadata,
            http=http,
            overwrite=args.overwrite,
            keep_pdf=not args.discard_pdf,
            max_candidates=args.max_candidates,
            elsevier_api_key=elsevier_api_key,
            elsevier_inst_token=elsevier_inst_token,
        )
        status = result.get("status", "unknown")
        counts[status] += 1
        if status != "existing" or not metadata.get("fulltext_recovery"):
            metadata["fulltext_recovery"] = result
        if status in {
            "recovered_pdf",
            "recovered_pmc",
            "recovered_elsevier_xml",
            "existing",
        }:
            metadata.setdefault("content_probe", {})["article_text"] = {
                key: result[key]
                for key in ("text_path", "text_characters", "text_sha256")
                if result.get(key) not in (None, "")
            }
        metadata["updated_at"] = utc_now()
        write_json(record_dir / "metadata.json", metadata)
        record_results.append(
            {
                "record": record_dir.name,
                "doi": metadata.get("doi", ""),
                "status": status,
                "source": result.get("source", ""),
                "text_characters": result.get("text_characters", 0),
                "reason": result.get("reason", ""),
                "error_count": len(result.get("errors", [])),
            }
        )
        print(
            f"  status={status} source={result.get('source', '') or '-'} "
            f"text_chars={result.get('text_characters', 0)} "
            f"errors={len(result.get('errors', []))}",
            flush=True,
        )

    summary = {
        "generated_at": utc_now(),
        "papers": len(records),
        "statuses": dict(sorted(counts.items())),
        "records": record_results,
    }
    write_json(args.database / args.summary_name, summary)
    indexed = rebuild_index(args.database)
    print(
        f"Article-text recovery complete: papers={len(records)} "
        f"recovered_pdf={counts['recovered_pdf']} recovered_pmc={counts['recovered_pmc']} "
        f"recovered_elsevier_xml={counts['recovered_elsevier_xml']} "
        f"existing={counts['existing']} unavailable={counts['unavailable']} "
        f"index_records={indexed}.",
        flush=True,
    )
    print(f"Summary: {args.database / args.summary_name}", flush=True)
    return 0


def cmd_standardize_marker_tables(args: argparse.Namespace) -> int:
    records = selected_records(args, apply_limit=not args.pending_only)
    if args.pending_only:
        records = [
            item
            for item in records
            if not has_complete_standardization(item[0], args.output_subdir)
        ]
        if args.limit:
            records = records[: args.limit]
    print(f"Standardizing marker tables for {len(records)} paper(s).", flush=True)
    summary = standardize_records(
        database=args.database,
        records=records,
        output_subdir=args.output_subdir,
        summary_name=args.summary_name,
    )
    print(
        "Marker-table standardization complete: "
        f"papers={summary['papers']} input_tables={summary['input_tables']} "
        f"statistics_sources={summary['statistics_source_tables']} "
        f"signature_sources={summary['signature_source_tables']} "
        f"statistics_rows={summary['statistics_rows']} "
        f"signature_pairs={summary['signature_cell_gene_pairs']} "
        f"errors={summary['errors']}.",
        flush=True,
    )
    print(f"Summary: {args.database / args.summary_name}", flush=True)
    return 0


def has_complete_standardization(record_dir: Path, output_subdir: str) -> bool:
    summary = read_json(
        record_dir / output_subdir / "standardization_summary.json",
        default=None,
    )
    if not isinstance(summary, dict):
        return False
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        return False
    return all(
        bool(outputs.get(name)) and (record_dir / outputs[name]).is_file()
        for name in ("statistics", "signatures")
    )


def cmd_agent_standardize_marker_tables(args: argparse.Namespace) -> int:
    invalid_arguments = []
    if args.limit < 0:
        invalid_arguments.append("--limit must be >= 0")
    if args.skip < 0:
        invalid_arguments.append("--skip must be >= 0")
    if args.preview_rows < 1:
        invalid_arguments.append("--preview-rows must be >= 1")
    if args.preview_columns < 1:
        invalid_arguments.append("--preview-columns must be >= 1")
    if args.cell_characters < 1:
        invalid_arguments.append("--cell-characters must be >= 1")
    if args.article_characters < 0:
        invalid_arguments.append("--article-characters must be >= 0")
    if args.context_blocks < 1:
        invalid_arguments.append("--context-blocks must be >= 1")
    if args.context_characters < 1:
        invalid_arguments.append("--context-characters must be >= 1")
    if args.tables_per_request < 1:
        invalid_arguments.append("--tables-per-request must be >= 1")
    if args.max_tokens < 1:
        invalid_arguments.append("--max-tokens must be >= 1")
    if not 0 <= args.temperature <= 2:
        invalid_arguments.append("--temperature must be between 0 and 2")
    if args.timeout < 1:
        invalid_arguments.append("--timeout must be >= 1")
    if args.retries < 0:
        invalid_arguments.append("--retries must be >= 0")
    if args.llm_attempts < 1:
        invalid_arguments.append("--llm-attempts must be >= 1")
    if args.sleep < 0:
        invalid_arguments.append("--sleep must be >= 0")
    if args.workers < 1:
        invalid_arguments.append("--workers must be >= 1")
    output_subdir = Path(args.output_subdir)
    if (
        not args.output_subdir.strip()
        or output_subdir == Path(".")
        or output_subdir.is_absolute()
        or ".." in output_subdir.parts
    ):
        invalid_arguments.append("--output-subdir must stay within each paper directory")
    summary_name = Path(args.summary_name)
    if (
        not args.summary_name.strip()
        or summary_name.is_absolute()
        or len(summary_name.parts) != 1
        or summary_name.suffix.lower() != ".json"
    ):
        invalid_arguments.append("--summary-name must be one JSON filename")
    if invalid_arguments:
        for message in invalid_arguments:
            print(message, file=sys.stderr)
        return 2

    records = select_agent_records(args.database, paper_id=args.paper_id)
    records = filter_records_by_cohort(
        records,
        str(args.cohort or "").strip(),
        str(args.exclude_cohort or "").strip(),
    )
    if args.quality_status:
        records = [
            (record_dir, metadata)
            for record_dir, metadata in records
            if (
                str(
                    read_json(
                        record_dir
                        / args.output_subdir
                        / "quality_flags.json",
                        default={},
                    ).get("status", "")
                )
                == args.quality_status
            )
        ]
    if args.require_complete_inputs:
        records = [
            (record_dir, metadata)
            for record_dir, metadata in records
            if has_article_text(record_dir) and has_extracted_tables(record_dir)
        ]
    stride = args.stride or 1
    offset = args.offset or 0
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    records = records[offset::stride]
    if args.skip:
        records = records[args.skip :]
    if args.limit:
        records = records[: args.limit]
    if args.paper_id and not records:
        print(f"No paper matched --paper-id {args.paper_id!r}.", file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        api_key = api_key_from_environment(args.api_key_env, env_file=args.env_file)
        if not api_key:
            print(
                f"Missing API key: set environment variable {args.api_key_env}.",
                file=sys.stderr,
            )
            return 2
        client = ECNUClient(
            api_key=api_key,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        )

    options = AgentOptions(
        output_subdir=args.output_subdir,
        preview_rows=args.preview_rows,
        preview_columns=args.preview_columns,
        cell_characters=args.cell_characters,
        article_characters=args.article_characters,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
        harness_mode=args.harness_mode,
        context_blocks=args.context_blocks,
        context_characters=args.context_characters,
        tables_per_request=args.tables_per_request,
        llm_attempts=args.llm_attempts,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        save_evidence=not args.no_save_evidence,
    )
    print(
        f"Agent-standardizing {len(records)} paper(s) with {args.model} "
        f"using {args.workers} worker(s), up to "
        f"{args.tables_per_request} table(s) per request.",
        flush=True,
    )
    summary = run_agent_standardization(
        database=args.database,
        records=records,
        client=client,
        options=options,
        sleep_seconds=args.sleep,
        fail_fast=args.fail_fast,
        workers=args.workers,
        stop_on_api_exhaustion=args.stop_on_api_exhaustion,
        summary_path=args.database / args.summary_name,
    )
    print(
        "Agent standardization complete: "
        f"papers={summary['papers']} complete={summary['complete']} "
        f"no_tables={summary['no_tables']} no_candidates={summary['no_candidate_tables']} "
        f"local_excluded={summary['locally_excluded_tables']} dry_run={summary['dry_run']} "
        f"failed={summary['failed']} resumed={summary['resumed']} "
        f"statistics_rows={summary['statistics_rows']} "
        f"signature_pairs={summary['signature_cell_gene_pairs']} "
        f"review_tables={summary['review_tables']}.",
        flush=True,
    )
    print(
        f"Summary: {args.database / args.summary_name}",
        flush=True,
    )
    return 1 if summary["failed"] else 0


def cmd_run_fulltext_flow(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    http = HttpClient(email=args.email, sleep=args.sleep, retries=args.retries)
    extensions = args.extensions or config.get("asset_extensions", [])
    records = selected_records(args)
    elsevier_api_key = api_key_from_environment(
        args.elsevier_api_key_env,
        env_file=args.env_file,
    )
    elsevier_inst_token = api_key_from_environment(
        args.elsevier_inst_token_env,
        env_file=args.env_file,
    )
    total_assets = 0
    total_downloaded = 0
    total_tables = 0

    print(f"Running full content flow for {len(records)} record(s).", flush=True)
    for index, (record_dir, metadata) in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {metadata.get('title', '')}", flush=True)
        probe = probe_record(
            record_dir=record_dir,
            metadata=metadata,
            http=http,
            config=config,
            fetch_publisher=not args.skip_publisher,
        )
        download_stats = download_assets_for_record(
            record_dir=record_dir,
            http=http,
            extensions=extensions,
            overwrite=args.overwrite,
            elsevier_api_key=elsevier_api_key,
            elsevier_inst_token=elsevier_inst_token,
        )
        extract_stats = extract_tables_for_record(record_dir)
        metadata["assets"] = load_manifest(record_dir)
        metadata["updated_at"] = utc_now()
        write_json(record_dir / "metadata.json", metadata)
        total_assets += probe.get("asset_count", 0)
        total_downloaded += download_stats["downloaded"]
        total_tables += extract_stats["tables"]
        print(
            "  pmc={pmc} text_chars={chars} assets={assets} downloaded={downloaded} tables={tables} errors={errors}".format(
                pmc=probe.get("pmc_status", ""),
                chars=probe.get("article_text", {}).get("text_characters", 0),
                assets=probe.get("asset_count", 0),
                downloaded=download_stats["downloaded"],
                tables=extract_stats["tables"],
                errors=len(probe.get("errors", [])) + download_stats["failed"] + extract_stats["failed"],
            ),
            flush=True,
        )

    indexed = rebuild_index(args.database)
    print(
        f"Flow complete: manifest_assets_seen={total_assets}, "
        f"downloaded_now={total_downloaded}, extracted_tables={total_tables}, "
        f"index_records={indexed}.",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local literature database for CNS-family single-cell "
            "sequencing papers."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("database"),
        help="Database root directory. Default: database",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Pipeline config file. Default: {DEFAULT_CONFIG}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover",
        help="Search PubMed and create one local metadata record per candidate paper.",
    )
    discover.add_argument("--start-date", default="", help="Publication start date, YYYY-MM-DD.")
    discover.add_argument("--end-date", default="", help="Publication end date, YYYY-MM-DD.")
    discover.add_argument(
        "--max-results",
        type=int,
        default=10000,
        help="Maximum number of PubMed records to fetch. Default: 10000.",
    )
    discover.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="PubMed efetch batch size. Default: 100.",
    )
    discover.add_argument(
        "--email",
        default="",
        help="Contact email for NCBI API courtesy. Strongly recommended.",
    )
    discover.add_argument("--api-key", default="", help="Optional NCBI API key.")
    discover.add_argument(
        "--sleep",
        type=float,
        default=0.34,
        help="Seconds to sleep between requests. Default: 0.34.",
    )
    discover.add_argument(
        "--retries",
        type=int,
        default=4,
        help="HTTP retry count. Default: 4.",
    )
    discover.add_argument(
        "--query",
        default="",
        help="Override the generated PubMed query.",
    )
    discover.add_argument(
        "--query-out",
        type=Path,
        default=None,
        help="Optional path to write the final PubMed query.",
    )
    discover.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated query without calling PubMed.",
    )
    discover.set_defaults(func=cmd_discover)

    index = subparsers.add_parser(
        "rebuild-index",
        help="Rebuild database/index.jsonl from existing metadata files.",
    )
    index.set_defaults(func=cmd_rebuild_index)

    summary = subparsers.add_parser(
        "summary",
        help="Print database-level metadata counts.",
    )
    summary.add_argument(
        "--top-journals",
        type=int,
        default=20,
        help="Number of journals to show. Default: 20.",
    )
    summary.set_defaults(func=cmd_summary)

    probe = subparsers.add_parser(
        "probe-assets",
        help="Probe PMC/publisher pages for article text and supplementary asset links.",
    )
    probe.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to probe from the current index order. Default: 10.",
    )
    add_shard_args(probe)
    probe.add_argument(
        "--email",
        default="",
        help="Contact email for NCBI/publisher API courtesy.",
    )
    probe.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests. Default: 0.5.",
    )
    probe.add_argument(
        "--retries",
        type=int,
        default=4,
        help="HTTP retry count. Default: 4.",
    )
    probe.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Per-request timeout in seconds. Default: 45.",
    )
    probe.add_argument(
        "--skip-publisher",
        action="store_true",
        help="Only probe PMC XML and skip DOI/publisher HTML pages.",
    )
    probe.add_argument(
        "--cohort",
        default="",
        help="Only probe records with this screening discovery cohort.",
    )
    probe.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    probe.add_argument(
        "--cns-family",
        choices=("Cell", "Nature", "Science"),
        default="",
        help="Only probe one CNS journal family.",
    )
    probe.add_argument(
        "--unprobed-only",
        action="store_true",
        help="Only probe records without an existing content_probe result.",
    )
    probe.add_argument(
        "--probe-errors-only",
        action="store_true",
        help="Only retry records whose existing content_probe contains errors.",
    )
    probe.add_argument(
        "--links-only",
        action="store_true",
        help="Discover supplementary links without saving PMC article XML or text.",
    )
    probe.add_argument(
        "--skip-elsevier-cdn",
        action="store_true",
        help="Skip slow guessed Elsevier CDN attachment URLs; use sync-elsevier-content instead.",
    )
    probe.add_argument(
        "--skip-elsevier-publisher",
        action="store_true",
        help="Skip Elsevier DOI pages; exact attachments can be registered with sync-elsevier-content.",
    )
    probe.add_argument(
        "--publisher-if-no-assets",
        action="store_true",
        help="Skip the publisher page when PMC already exposes supplementary assets.",
    )
    probe.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one detailed progress record every N papers. Default: 1.",
    )
    probe.set_defaults(func=cmd_probe_assets)

    download = subparsers.add_parser(
        "download-assets",
        help="Download supplementary assets already listed in each manifest.",
    )
    download.add_argument("--limit", type=int, default=10, help="Number of records to process. Default: 10.")
    add_shard_args(download)
    download.add_argument("--email", default="", help="Contact email for NCBI/publisher courtesy.")
    download.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests. Default: 0.5.")
    download.add_argument("--retries", type=int, default=4, help="HTTP retry count. Default: 4.")
    download.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Per-request timeout in seconds. Default: 90.",
    )
    download.add_argument(
        "--cohort",
        default="",
        help="Only download records with this screening discovery cohort.",
    )
    download.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    download.add_argument(
        "--cns-family",
        choices=("Cell", "Nature", "Science"),
        default="",
        help="Only download one CNS journal family.",
    )
    download.add_argument(
        "--env-file",
        type=Path,
        default=ENV_FILE,
        help="Dotenv file containing publisher API credentials. Default: project .env.",
    )
    download.add_argument(
        "--elsevier-api-key-env",
        default="ELSEVIER_API_KEY",
        help="Environment variable containing the Elsevier API key.",
    )
    download.add_argument(
        "--elsevier-inst-token-env",
        default="ELSEVIER_INST_TOKEN",
        help="Optional environment variable containing an Elsevier institutional token.",
    )
    download.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    download.add_argument(
        "--pending-only",
        action="store_true",
        help="Only process records with a download candidate missing locally.",
    )
    download.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Asset extensions to download. Default comes from config.",
    )
    download.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one detailed progress record every N papers. Default: 1.",
    )
    download.add_argument(
        "--summary-name",
        default="",
        help="Database-level summary filename; use a distinct name for each shard.",
    )
    download.set_defaults(func=cmd_download_assets)

    extract = subparsers.add_parser(
        "extract-tables",
        help="Extract CSV tables from downloaded xlsx/csv/tsv/docx supplementary assets.",
    )
    extract.add_argument("--limit", type=int, default=10, help="Number of records to process. Default: 10.")
    add_shard_args(extract)
    extract.add_argument(
        "--paper-id",
        default="",
        help="Only process records matching this DOI, PMID, PMCID, or record name.",
    )
    extract.add_argument(
        "--cohort",
        default="",
        help="Only process records with this screening discovery cohort.",
    )
    extract.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    extract.add_argument(
        "--pending-only",
        action="store_true",
        help="Only process records without a complete tables manifest.",
    )
    extract.set_defaults(func=cmd_extract_tables)

    recover_text = subparsers.add_parser(
        "recover-article-text",
        help="Recover missing article text from PMC or verified open-access repositories.",
    )
    recover_text.add_argument(
        "--paper-id",
        default="",
        help="Only process records matching this DOI, PMID, PMCID, or record name.",
    )
    recover_text.add_argument(
        "--doi-prefix",
        default="",
        help="Only process DOI values beginning with this prefix, for example 10.1016/.",
    )
    recover_text.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of matching paper records to process. Default: all matches.",
    )
    recover_text.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many records after filtering and sharding. Default: 0.",
    )
    add_shard_args(recover_text)
    recover_text.add_argument(
        "--cohort",
        default="",
        help="Only process records with this screening discovery cohort.",
    )
    recover_text.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    recover_text.add_argument(
        "--include-without-tables",
        action="store_true",
        help="Also process papers that currently have no extracted CSV tables.",
    )
    recover_text.add_argument(
        "--include-existing",
        action="store_true",
        help="Also inspect papers that already have a valid article_text.txt.",
    )
    recover_text.add_argument(
        "--email",
        default="",
        help="Contact email for NCBI and repository API courtesy.",
    )
    recover_text.add_argument(
        "--env-file",
        type=Path,
        default=ENV_FILE,
        help="Dotenv file containing publisher API credentials. Default: project .env.",
    )
    recover_text.add_argument(
        "--elsevier-api-key-env",
        default="ELSEVIER_API_KEY",
        help="Environment variable containing the Elsevier API key.",
    )
    recover_text.add_argument(
        "--elsevier-inst-token-env",
        default="ELSEVIER_INST_TOKEN",
        help="Optional environment variable containing an Elsevier institutional token.",
    )
    recover_text.add_argument(
        "--sleep",
        type=float,
        default=0.34,
        help="Seconds to sleep between requests. Default: 0.34.",
    )
    recover_text.add_argument(
        "--retries",
        type=int,
        default=4,
        help="HTTP retry count. Default: 4.",
    )
    recover_text.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds. Default: 120.",
    )
    recover_text.add_argument(
        "--max-candidates",
        type=int,
        default=12,
        help="Maximum verified OA PDF candidates tried per paper. Default: 12.",
    )
    recover_text.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing valid article_text.txt.",
    )
    recover_text.add_argument(
        "--discard-pdf",
        action="store_true",
        help="Delete the source PDF after validated text extraction.",
    )
    recover_text.add_argument(
        "--summary-name",
        default="fulltext_recovery_summary.json",
        help="Database-level summary filename; use a distinct name for each shard.",
    )
    recover_text.set_defaults(func=cmd_recover_article_text)

    sync_elsevier = subparsers.add_parser(
        "sync-elsevier-content",
        help="Fetch Elsevier XML, recover missing text, and register exact mmc attachments.",
    )
    sync_elsevier.add_argument(
        "--paper-id",
        default="",
        help="Only process records matching this DOI, PMID, PMCID, or record name.",
    )
    sync_elsevier.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of Elsevier records to process. Default: all records.",
    )
    add_shard_args(sync_elsevier)
    sync_elsevier.add_argument(
        "--cohort",
        default="",
        help="Only process records with this screening discovery cohort.",
    )
    sync_elsevier.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    sync_elsevier.add_argument(
        "--unsynced-only",
        action="store_true",
        help="Only process records without an existing elsevier_content_sync result.",
    )
    sync_elsevier.add_argument(
        "--email",
        default="",
        help="Contact email for API courtesy.",
    )
    sync_elsevier.add_argument(
        "--env-file",
        type=Path,
        default=ENV_FILE,
        help="Dotenv file containing Elsevier credentials. Default: project .env.",
    )
    sync_elsevier.add_argument(
        "--elsevier-api-key-env",
        default="ELSEVIER_API_KEY",
        help="Environment variable containing the Elsevier API key.",
    )
    sync_elsevier.add_argument(
        "--elsevier-inst-token-env",
        default="ELSEVIER_INST_TOKEN",
        help="Optional environment variable containing an Elsevier institutional token.",
    )
    sync_elsevier.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API requests. Default: 0.1.",
    )
    sync_elsevier.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry count. Default: 3.",
    )
    sync_elsevier.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds. Default: 180.",
    )
    sync_elsevier.add_argument(
        "--refresh-xml",
        action="store_true",
        help="Re-fetch Elsevier XML even when a cached copy exists.",
    )
    sync_elsevier.add_argument(
        "--overwrite-text",
        action="store_true",
        help="Replace an existing article_text.txt with Elsevier XML text.",
    )
    sync_elsevier.add_argument(
        "--assets-only",
        action="store_true",
        help="Register exact supplementary attachments without saving article XML or text.",
    )
    sync_elsevier.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one detailed progress record every N papers. Default: 1.",
    )
    sync_elsevier.add_argument(
        "--summary-name",
        default="",
        help="Database-level summary filename; use a distinct name for each shard.",
    )
    sync_elsevier.set_defaults(func=cmd_sync_elsevier_content)

    standardize = subparsers.add_parser(
        "standardize-marker-tables",
        help="Aggregate structurally compatible marker statistics and signature tables per paper.",
    )
    standardize.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of paper records to process. Default: all records.",
    )
    add_shard_args(standardize)
    standardize.add_argument(
        "--cohort",
        default="",
        help="Only process records with this screening discovery cohort.",
    )
    standardize.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    standardize.add_argument(
        "--output-subdir",
        default="standardized",
        help="Per-paper output directory name. Default: standardized.",
    )
    standardize.add_argument(
        "--pending-only",
        action="store_true",
        help="Only process records without complete standardization outputs.",
    )
    standardize.add_argument(
        "--summary-name",
        default="standardization_summary.json",
        help="Database-level summary filename; use a distinct name for each shard.",
    )
    standardize.set_defaults(func=cmd_standardize_marker_tables)

    agent_standardize = subparsers.add_parser(
        "agent-standardize-marker-tables",
        help="Use ECNU ecnu-max plus paper context to standardize marker tables.",
    )
    agent_standardize.add_argument(
        "--paper-id",
        default="",
        help="Only process records matching this DOI, PMID, PMCID, or record name.",
    )
    agent_standardize.add_argument(
        "--quality-status",
        choices=("error", "warning"),
        default="",
        help=(
            "Only process papers whose existing output quality_flags.json has "
            "this status. Default: no quality-status filter."
        ),
    )
    agent_standardize.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of matching paper records to process. Default: all matches.",
    )
    agent_standardize.add_argument(
        "--skip",
        type=int,
        default=0,
        help=(
            "Skip this many records after quality filtering and sharding. "
            "Useful when resuming a stopped shard. Default: 0."
        ),
    )
    add_shard_args(agent_standardize)
    agent_standardize.add_argument(
        "--cohort",
        default="",
        help="Only process records with this screening discovery cohort.",
    )
    agent_standardize.add_argument(
        "--exclude-cohort",
        default="",
        help="Exclude records with this screening discovery cohort.",
    )
    agent_standardize.add_argument(
        "--api-key-env",
        default="ECNU_API_KEY",
        help="Environment variable containing the ECNU API key. Default: ECNU_API_KEY.",
    )
    agent_standardize.add_argument(
        "--env-file",
        type=Path,
        default=ENV_FILE,
        help="Fallback dotenv file when the API key is not in the environment.",
    )
    agent_standardize.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Chat completions endpoint. Default: {DEFAULT_ENDPOINT}",
    )
    agent_standardize.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ECNU model name. Default: {DEFAULT_MODEL}.",
    )
    agent_standardize.add_argument(
        "--output-subdir",
        default="agent_standardized",
        help="Per-paper output directory name. Default: agent_standardized.",
    )
    agent_standardize.add_argument(
        "--summary-name",
        default="agent_standardization_summary.json",
        help="Database-level checkpoint filename. Default: agent_standardization_summary.json.",
    )
    agent_standardize.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="CSV rows supplied to the agent per table. Default: 8.",
    )
    agent_standardize.add_argument(
        "--preview-columns",
        type=int,
        default=64,
        help="CSV columns supplied to the agent per table. Default: 64.",
    )
    agent_standardize.add_argument(
        "--cell-characters",
        type=int,
        default=120,
        help="Maximum characters supplied per CSV cell. Default: 120.",
    )
    agent_standardize.add_argument(
        "--article-characters",
        type=int,
        default=30000,
        help="Maximum relevant article-text characters. Default: 30000.",
    )
    agent_standardize.add_argument(
        "--harness-mode",
        choices=("standard", "compact"),
        default="standard",
        help="Evidence architecture to use. Default: standard.",
    )
    agent_standardize.add_argument(
        "--context-blocks",
        type=int,
        default=10,
        help="Maximum structured article blocks in compact mode. Default: 10.",
    )
    agent_standardize.add_argument(
        "--context-characters",
        type=int,
        default=12000,
        help="Maximum article-context characters in compact mode. Default: 12000.",
    )
    agent_standardize.add_argument(
        "--tables-per-request",
        type=int,
        default=5,
        help="Maximum candidate tables in one model request. Default: 5.",
    )
    agent_standardize.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum generated tokens per model call. Default: 32768.",
    )
    agent_standardize.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature. Default: 0.1.",
    )
    agent_standardize.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds. Default: 600.",
    )
    agent_standardize.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient HTTP errors. Default: 3.",
    )
    agent_standardize.add_argument(
        "--llm-attempts",
        type=int,
        default=3,
        help="Attempts to obtain locally valid JSON per paper. Default: 3.",
    )
    agent_standardize.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds each worker sleeps after a paper. Default: 0.5.",
    )
    agent_standardize.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of papers processed concurrently. Default: 1.",
    )
    agent_standardize.add_argument(
        "--require-complete-inputs",
        action="store_true",
        help="Only process papers with valid article text and extracted CSV tables.",
    )
    agent_standardize.add_argument(
        "--stop-on-api-exhaustion",
        action="store_true",
        help="Stop scheduling API work after a persistent quota or authentication error.",
    )
    thinking_group = agent_standardize.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable model thinking mode. Disabled by default.",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Keep model thinking mode disabled (compatibility flag).",
    )
    agent_standardize.set_defaults(thinking=False)
    agent_standardize.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess papers whose agent output is already complete.",
    )
    agent_standardize.add_argument(
        "--dry-run",
        action="store_true",
        help="Build evidence and summaries without calling ECNU.",
    )
    agent_standardize.add_argument(
        "--no-save-evidence",
        action="store_true",
        help="Do not retain the evidence JSON after an API run.",
    )
    agent_standardize.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first paper-level failure.",
    )
    agent_standardize.set_defaults(func=cmd_agent_standardize_marker_tables)

    full = subparsers.add_parser(
        "run-fulltext-flow",
        help="Probe article content, download listed supplementary assets, and extract tables.",
    )
    full.add_argument("--limit", type=int, default=10, help="Number of records to process. Default: 10.")
    add_shard_args(full)
    full.add_argument("--email", default="", help="Contact email for NCBI/publisher courtesy.")
    full.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests. Default: 0.5.")
    full.add_argument("--retries", type=int, default=4, help="HTTP retry count. Default: 4.")
    full.add_argument(
        "--env-file",
        type=Path,
        default=ENV_FILE,
        help="Dotenv file containing publisher API credentials. Default: project .env.",
    )
    full.add_argument(
        "--elsevier-api-key-env",
        default="ELSEVIER_API_KEY",
        help="Environment variable containing the Elsevier API key.",
    )
    full.add_argument(
        "--elsevier-inst-token-env",
        default="ELSEVIER_INST_TOKEN",
        help="Optional environment variable containing an Elsevier institutional token.",
    )
    full.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    full.add_argument("--skip-publisher", action="store_true", help="Skip DOI/publisher HTML pages.")
    full.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Asset extensions to download. Default comes from config.",
    )
    full.set_defaults(func=cmd_run_fulltext_flow)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
