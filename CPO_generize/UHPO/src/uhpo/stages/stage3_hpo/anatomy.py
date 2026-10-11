from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ENDPOINT = "https://ubergraph.apps.renci.org/sparql"
REDUNDANT_GRAPH = "http://reasoner.renci.org/redundant"
OBO_BASE = "http://purl.obolibrary.org/obo/"
UPHENO_AFFECTS = f"{OBO_BASE}UPHENO_0000001"
CL_ANATOMY_PREDICATES = (
    f"{OBO_BASE}BFO_0000050",  # part of
    f"{OBO_BASE}RO_0001025",  # located in
    f"{OBO_BASE}RO_0002100",  # has soma location
)


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8", newline="")
    )


def _read_hpo_ids(path: Path) -> list[str]:
    ids: set[str] = set()
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("id: HP:"):
                ids.add(line.split(None, 1)[1].strip())
    return sorted(ids)


def _read_cl_ids(path: Path) -> list[str]:
    ids: set[str] = set()
    with _open_text(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            cl_id = (row.get("catalog_id") or row.get("cl_id") or "").strip()
            if cl_id.startswith("CL:"):
                ids.add(cl_id)
    return sorted(ids)


def _obo_uri(identifier: str) -> str:
    return f"<{OBO_BASE}{identifier.replace(':', '_')}>"


def _curie(value: str, prefix: str) -> str:
    uri = value.strip().strip("<>")
    expected = f"{OBO_BASE}{prefix}_"
    if not uri.startswith(expected):
        return ""
    return f"{prefix}:{uri.removeprefix(expected)}"


def _literal(value: str) -> str:
    value = value.strip()
    if not value.startswith('"'):
        return value.split("^^<", 1)[0]
    match = re.match(r'^("(?:[^"\\]|\\.)*")', value)
    if not match:
        return value.strip('"')
    try:
        return str(json.loads(match.group(1)))
    except json.JSONDecodeError:
        return match.group(1).strip('"')


def _query_tsv(endpoint: str, query: str, retries: int, timeout: int) -> list[dict[str, str]]:
    body = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "text/tab-separated-values",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "CPO-generize-anatomy-materializer/1.0",
        },
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
            reader = csv.DictReader(
                text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE
            )
            return [
                {(key or "").lstrip("?"): value for key, value in row.items()}
                for row in reader
            ]
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Ubergraph query failed after {retries + 1} attempts") from exc
            time.sleep(min(30, 2**attempt))
    return []


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _materialize_hpo(
    endpoint: str, ids: Sequence[str], batch_size: int, retries: int, timeout: int
) -> list[dict[str, str]]:
    output: set[tuple[str, str, str]] = set()
    for index, batch in enumerate(_batches(ids, batch_size), start=1):
        values = " ".join(_obo_uri(identifier) for identifier in batch)
        query = f"""
SELECT DISTINCT ?subject ?anatomy ?anatomy_label WHERE {{
  VALUES ?subject {{ {values} }}
  GRAPH <{REDUNDANT_GRAPH}> {{
    ?subject <{UPHENO_AFFECTS}> ?anatomy .
  }}
  FILTER(STRSTARTS(STR(?anatomy), "{OBO_BASE}UBERON_"))
  OPTIONAL {{ ?anatomy <http://www.w3.org/2000/01/rdf-schema#label> ?anatomy_label . }}
}}
"""
        for row in _query_tsv(endpoint, query, retries, timeout):
            hpo_id = _curie(row.get("subject", ""), "HP")
            uberon_id = _curie(row.get("anatomy", ""), "UBERON")
            if hpo_id and uberon_id:
                output.add((hpo_id, uberon_id, _literal(row.get("anatomy_label", ""))))
        print(f"HPO batches: {index}", flush=True)
    return [
        {
            "hpo_id": hpo_id,
            "uberon_id": uberon_id,
            "uberon_label": label,
            "predicate": "UPHENO:0000001",
            "source_graph": REDUNDANT_GRAPH,
        }
        for hpo_id, uberon_id, label in sorted(output)
    ]


def _materialize_cl(
    endpoint: str, ids: Sequence[str], batch_size: int, retries: int, timeout: int
) -> list[dict[str, str]]:
    output: set[tuple[str, str, str, str]] = set()
    predicates = " ".join(f"<{predicate}>" for predicate in CL_ANATOMY_PREDICATES)
    for index, batch in enumerate(_batches(ids, batch_size), start=1):
        values = " ".join(_obo_uri(identifier) for identifier in batch)
        query = f"""
SELECT DISTINCT ?subject ?predicate ?anatomy ?anatomy_label WHERE {{
  VALUES ?subject {{ {values} }}
  VALUES ?predicate {{ {predicates} }}
  GRAPH <{REDUNDANT_GRAPH}> {{
    ?subject ?predicate ?anatomy .
  }}
  FILTER(STRSTARTS(STR(?anatomy), "{OBO_BASE}UBERON_"))
  OPTIONAL {{ ?anatomy <http://www.w3.org/2000/01/rdf-schema#label> ?anatomy_label . }}
}}
"""
        for row in _query_tsv(endpoint, query, retries, timeout):
            cl_id = _curie(row.get("subject", ""), "CL")
            uberon_id = _curie(row.get("anatomy", ""), "UBERON")
            predicate_uri = row.get("predicate", "").strip().strip("<>")
            predicate_id = _curie(f"<{predicate_uri}>", "BFO") or _curie(
                f"<{predicate_uri}>", "RO"
            )
            if cl_id and uberon_id and predicate_id:
                output.add(
                    (cl_id, uberon_id, _literal(row.get("anatomy_label", "")), predicate_id)
                )
        print(f"CL batches: {index}", flush=True)
    return [
        {
            "cl_id": cl_id,
            "uberon_id": uberon_id,
            "uberon_label": label,
            "predicate": predicate,
            "source_graph": REDUNDANT_GRAPH,
        }
        for cl_id, uberon_id, label, predicate in sorted(output)
    ]


def _materialize_uberon_ancestors(
    endpoint: str, ids: Sequence[str], batch_size: int, retries: int, timeout: int
) -> list[dict[str, str]]:
    output: set[tuple[str, str]] = set()
    for index, batch in enumerate(_batches(ids, batch_size), start=1):
        values = " ".join(_obo_uri(identifier) for identifier in batch)
        query = f"""
SELECT DISTINCT ?subject ?ancestor WHERE {{
  VALUES ?subject {{ {values} }}
  GRAPH <{REDUNDANT_GRAPH}> {{
    ?subject <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?ancestor .
  }}
  FILTER(STRSTARTS(STR(?ancestor), "{OBO_BASE}UBERON_"))
}}
"""
        for row in _query_tsv(endpoint, query, retries, timeout):
            uberon_id = _curie(row.get("subject", ""), "UBERON")
            ancestor_id = _curie(row.get("ancestor", ""), "UBERON")
            if uberon_id and ancestor_id:
                output.add((uberon_id, ancestor_id))
        print(f"UBERON batches: {index}", flush=True)
    return [
        {"uberon_id": uberon_id, "ancestor_uberon_id": ancestor_id}
        for uberon_id, ancestor_id in sorted(output)
    ]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot reuse missing mapping file: {path}")
    with _open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Sequence[dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize inferred HPO/CL-to-UBERON closure from Ubergraph."
    )
    from .paths import (
        ANATOMY_MANIFEST,
        CELL_MARKERS,
        CL_ANATOMY,
        HPO_ANATOMY,
        HPO_OBO,
        UBERON_ANCESTORS,
    )

    parser.add_argument("--hpo-obo", type=Path, default=HPO_OBO)
    parser.add_argument(
        "--markers", type=Path, default=CELL_MARKERS
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--hpo-output", type=Path, default=HPO_ANATOMY
    )
    parser.add_argument(
        "--cl-output", type=Path, default=CL_ANATOMY
    )
    parser.add_argument(
        "--uberon-output",
        type=Path,
        default=UBERON_ANCESTORS,
    )
    parser.add_argument(
        "--reuse-mappings",
        action="store_true",
        help="Reuse existing HPO/CL mapping outputs and only refresh UBERON closure.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=ANATOMY_MANIFEST
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hpo_ids = _read_hpo_ids(args.hpo_obo)
    cl_ids = _read_cl_ids(args.markers)
    if args.reuse_mappings:
        hpo_rows = _read_rows(args.hpo_output)
        cl_rows = _read_rows(args.cl_output)
    else:
        hpo_rows = _materialize_hpo(
            args.endpoint, hpo_ids, args.batch_size, args.retries, args.timeout
        )
        cl_rows = _materialize_cl(
            args.endpoint, cl_ids, args.batch_size, args.retries, args.timeout
        )
    uberon_ids = sorted(
        {
            row["uberon_id"]
            for row in [*hpo_rows, *cl_rows]
            if row.get("uberon_id", "").startswith("UBERON:")
        }
    )
    uberon_rows = _materialize_uberon_ancestors(
        args.endpoint, uberon_ids, args.batch_size, args.retries, args.timeout
    )
    _write_tsv(
        args.hpo_output,
        hpo_rows,
        ("hpo_id", "uberon_id", "uberon_label", "predicate", "source_graph"),
    )
    _write_tsv(
        args.cl_output,
        cl_rows,
        ("cl_id", "uberon_id", "uberon_label", "predicate", "source_graph"),
    )
    _write_tsv(
        args.uberon_output,
        uberon_rows,
        ("uberon_id", "ancestor_uberon_id"),
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "reasoner_graph": REDUNDANT_GRAPH,
        "hpo_source_count": len(hpo_ids),
        "cl_source_count": len(cl_ids),
        "hpo_mapping_count": len(hpo_rows),
        "cl_mapping_count": len(cl_rows),
        "hpo_mapped_count": len({row["hpo_id"] for row in hpo_rows}),
        "cl_mapped_count": len({row["cl_id"] for row in cl_rows}),
        "uberon_source_count": len(uberon_ids),
        "uberon_ancestor_count": len(uberon_rows),
        "hpo_output": str(args.hpo_output),
        "cl_output": str(args.cl_output),
        "uberon_output": str(args.uberon_output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
