#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = MODULE_DIR / "results/smoke_mapping_v2"

INTEGER_FIELDS = {
    "final_rank",
    "exact_available_prediction",
    "shared_alias_count",
    "exact_evidence_override",
    "exact_retrieval_rank",
    "ancestor_retrieval_rank",
    "raw_retrieval_rank",
    "semantic_retrieval_rank",
    "alias_retrieval",
    "shared_mechanism_count",
    "shared_specific_mechanism_count",
    "derived_ancestor_distance",
}
FLOAT_FIELDS = {
    "final_score",
    "gate_probability",
    "gate_threshold",
    "p_exact",
    "p_hp_broader",
    "p_hp_narrower",
    "p_related",
    "exact_embedding_score",
    "ancestor_embedding_score",
    "exact_structure_score",
    "ancestor_structure_score",
    "exact_semantic_tower_score",
    "ancestor_semantic_tower_score",
    "poincare_score",
    "raw_graph_score",
    "semantic_score",
    "ancestor_rank_score",
    "alias_jaccard",
    "mechanism_jaccard",
}


def _typed_row(row: dict[str, str]) -> dict[str, object]:
    typed: dict[str, object] = dict(row)
    for field in INTEGER_FIELDS:
        try:
            typed[field] = int(row.get(field, "0") or 0)
        except ValueError:
            typed[field] = 0
    for field in FLOAT_FIELDS:
        try:
            typed[field] = float(row.get(field, "0") or 0.0)
        except ValueError:
            typed[field] = 0.0
    return typed


def _immune_cell_ids(catalog_path: Path) -> set[str]:
    parents: dict[str, str] = {}
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            cl_id = row.get("cl_id", "")
            if cl_id:
                parents[cl_id] = row.get("parent_id", "")
    roots = {"CL:0000738", "CL:0000129"}  # leukocyte and microglial cell
    immune: set[str] = set()
    for cl_id in parents:
        current = cl_id
        seen: set[str] = set()
        while current and current not in seen:
            if current in roots:
                immune.add(cl_id)
                break
            seen.add(current)
            current = parents.get(current, "")
    return immune | roots


def _case_metrics(cases: list[dict[str, object]]) -> dict[str, object]:
    ranks = [int(row["best_rank"]) for row in cases]
    positive = [rank for rank in ranks if rank > 0]
    return {
        "queries": len(cases),
        "retrieved_targets": len(positive),
        "recall_at_1": sum(rank <= 1 for rank in positive) / max(1, len(cases)),
        "recall_at_10": sum(rank <= 10 for rank in positive) / max(1, len(cases)),
        "recall_at_50": sum(rank <= 50 for rank in positive) / max(1, len(cases)),
        "mrr": sum(1.0 / rank for rank in positive) / max(1, len(cases)),
        "mean_gate_probability": sum(float(row["gate_probability"]) for row in cases)
        / max(1, len(cases)),
    }


def _cohort_evaluation(
    cases: list[dict[str, object]], gate_threshold: float
) -> dict[str, object]:
    task_keys = {
        "exact": "expert_exact_present",
        "hp_broader": "expert_hp_broader",
        "counterfactual_exact": "counterfactual_exact_removed",
        "synthetic_hpo_cutout": "synthetic_hpo_cutout",
    }
    result = {
        output_key: _case_metrics([row for row in cases if row["task"] == task])
        for task, output_key in task_keys.items()
    }
    result["gate_threshold"] = gate_threshold
    return result


def _mapping_cohort_metrics(
    summaries: list[dict[str, object]],
    by_mp: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    count = len(summaries)
    candidate_groups = [by_mp[str(row["mp_id"])] for row in summaries]
    return {
        "terms": count,
        "exact_available_rate": sum(bool(row["exact_available"]) for row in summaries)
        / max(1, count),
        "top1_exact_rate": sum(row["top_relation"] == "exact" for row in summaries)
        / max(1, count),
        "mean_gate_probability": sum(float(row["gate_probability"]) for row in summaries)
        / max(1, count),
        "top1_specific_overlap_rate": sum(
            int(rows[0]["shared_specific_mechanism_count"]) > 0
            for rows in candidate_groups
        )
        / max(1, count),
        "top20_specific_overlap_rate": sum(
            any(int(row["shared_specific_mechanism_count"]) > 0 for row in rows)
            for rows in candidate_groups
        )
        / max(1, count),
        "top20_mechanism_overlap_rate": sum(
            any(int(row["shared_mechanism_count"]) > 0 for row in rows)
            for rows in candidate_groups
        )
        / max(1, count),
        "derived_parent_rate": sum(int(row["derived_count"]) > 0 for row in summaries)
        / max(1, count),
    }


class ResultStore:
    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir.resolve()
        self.mapping_path = self.result_dir / "mpo_hpo_mapping_final.tsv"
        self.evaluation_path = self.result_dir / "evaluation_cases.tsv"
        with (self.result_dir / "training_manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        with (self.result_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            self.evaluation = json.load(handle)

        self.by_mp: dict[str, list[dict[str, object]]] = defaultdict(list)
        relation_counts: Counter[str] = Counter()
        with self.mapping_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            self.mapping_fields = list(reader.fieldnames or ())
            for raw in reader:
                row = _typed_row(raw)
                self.by_mp[str(row["mp_id"])].append(row)
                relation_counts[str(row["predicted_relation"])] += 1

        catalog_path = Path(self.manifest["inputs"]["cl_catalog"]["path"])
        immune_cells = _immune_cell_ids(catalog_path)
        immune_specific_terms: dict[str, set[str]] = {
            "MPO": set(),
            "HPO": set(),
        }
        immune_any_terms: dict[str, set[str]] = {"MPO": set(), "HPO": set()}
        immune_evidence: dict[str, dict[str, dict[str, object]]] = {
            "MPO": defaultdict(lambda: {"specific": 0, "general": 0, "cells": Counter()}),
            "HPO": defaultdict(lambda: {"specific": 0, "general": 0, "cells": Counter()}),
        }
        edge_path = self.result_dir / "data/processed/term_mechanism_edges.tsv"
        with edge_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ontology = row.get("ontology", "")
                cl_id = row.get("cl_id", "")
                if ontology not in immune_evidence or cl_id not in immune_cells:
                    continue
                term_id = row.get("term_id", "")
                channel = row.get("channel", "general")
                immune_any_terms[ontology].add(term_id)
                evidence = immune_evidence[ontology][term_id]
                evidence[channel] = int(evidence[channel]) + 1
                cells = evidence["cells"]
                assert isinstance(cells, Counter)
                cells[f"{cl_id} {row.get('cl_name', '')}"] += 1
                if channel == "specific":
                    immune_specific_terms[ontology].add(term_id)
        self.immune_specific_terms = immune_specific_terms
        self.immune_any_terms = immune_any_terms

        self.mpos: list[dict[str, object]] = []
        derived_rows = 0
        exact_available = 0
        strong_evidence = 0
        gate_overrides = 0
        for mp_id, rows in sorted(self.by_mp.items()):
            rows.sort(key=lambda item: int(item["final_rank"]))
            top = rows[0]
            derived_count = sum(int(row["derived_ancestor_distance"]) > 0 for row in rows)
            derived_rows += derived_count
            exact_flag = bool(top["exact_available_prediction"])
            strong_flag = any(bool(row["exact_evidence_override"]) for row in rows)
            override_flag = any(bool(row["gate_override_reason"]) for row in rows)
            exact_available += int(exact_flag)
            strong_evidence += int(strong_flag)
            gate_overrides += int(override_flag)
            immune = immune_evidence["MPO"].get(mp_id)
            immune_specific_count = int(immune["specific"]) if immune else 0
            immune_general_count = int(immune["general"]) if immune else 0
            immune_cell_labels = (
                [name for name, _count in immune["cells"].most_common(6)]
                if immune and isinstance(immune["cells"], Counter)
                else []
            )
            summary = {
                "mp_id": mp_id,
                "mp_name": top["mp_name"],
                "gate_probability": top["gate_probability"],
                "gate_threshold": top["gate_threshold"],
                "exact_available": exact_flag,
                "strong_evidence": strong_flag,
                "gate_override": override_flag,
                "derived_count": derived_count,
                "exact_count": sum(row["predicted_relation"] == "exact" for row in rows),
                "mechanism_count": max(
                    int(row["shared_mechanism_count"]) for row in rows
                ),
                "immune_specific": immune_specific_count > 0,
                "immune_any": immune_specific_count + immune_general_count > 0,
                "immune_specific_count": immune_specific_count,
                "immune_general_count": immune_general_count,
                "immune_cells": immune_cell_labels,
                "top_hp_id": top["hp_id"],
                "top_hp_name": top["hp_name"],
                "top_relation": top["predicted_relation"],
                "top_score": top["final_score"],
                "search_text": (
                    f"{mp_id} {top['mp_name']} {top['hp_id']} {top['hp_name']}"
                ).lower(),
            }
            self.mpos.append(summary)
        self.mp_summary = {str(row["mp_id"]): row for row in self.mpos}
        immune_summaries = [row for row in self.mpos if row["immune_specific"]]
        nonimmune_summaries = [row for row in self.mpos if not row["immune_specific"]]
        self.cohort_quality = {
            "immune_specific": _mapping_cohort_metrics(
                immune_summaries, self.by_mp
            ),
            "nonimmune": _mapping_cohort_metrics(nonimmune_summaries, self.by_mp),
        }

        self.cases: list[dict[str, object]] = []
        with self.evaluation_path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter="\t"):
                query_id = raw.get("query_id", "")
                ontology = "MPO" if query_id.startswith("MP:") else "HPO"
                self.cases.append(
                    {
                        **raw,
                        "best_rank": int(raw.get("best_rank", "0") or 0),
                        "gate_probability": float(
                            raw.get("gate_probability", "0") or 0.0
                        ),
                        "inspectable": raw.get("query_id", "") in self.by_mp,
                        "immune_related": query_id
                        in self.immune_specific_terms[ontology],
                    }
                )

        gate_threshold = float(self.evaluation["gate_threshold"])
        self.immune_evaluation = _cohort_evaluation(
            [row for row in self.cases if row["immune_related"]], gate_threshold
        )
        self.nonimmune_evaluation = _cohort_evaluation(
            [row for row in self.cases if not row["immune_related"]], gate_threshold
        )

        self.overview = {
            "mapped_mpo_terms": len(self.by_mp),
            "mapping_rows": sum(len(rows) for rows in self.by_mp.values()),
            "output_top_k": max((len(rows) for rows in self.by_mp.values()), default=0),
            "hpo_profile_terms": self.manifest["data"]["hpo_profile_terms"],
            "term_mechanism_edges": self.manifest["data"]["term_mechanism_edges"],
            "exact_available_queries": exact_available,
            "strong_evidence_queries": strong_evidence,
            "gate_override_queries": gate_overrides,
            "derived_ancestor_rows": derived_rows,
            "relation_counts": dict(relation_counts),
            "semantic_method": self.manifest["data"]["semantic_method"],
            "fusion_mode": self.manifest["training"]["fusion_mode"],
            "fusion_weights": self.manifest["training"]["fusion_weights"],
            "immune_specific_mpo_terms": len(self.immune_specific_terms["MPO"] & set(self.by_mp)),
            "immune_any_mpo_terms": len(self.immune_any_terms["MPO"] & set(self.by_mp)),
            "immune_cell_nodes": len(immune_cells),
        }

    def summary(self) -> dict[str, object]:
        default_mpo = "MP:0000017" if "MP:0000017" in self.by_mp else self.mpos[0]["mp_id"]
        immune_defaults = [
            row
            for row in self.mpos
            if row["immune_specific"] and row["top_relation"] == "exact"
        ]
        default_immune_mpo = (
            max(immune_defaults, key=lambda row: (float(row["top_score"]), row["mp_id"]))["mp_id"]
            if immune_defaults
            else next(
                (row["mp_id"] for row in self.mpos if row["immune_specific"]),
                default_mpo,
            )
        )
        return {
            "result_name": self.result_dir.name,
            "overview": self.overview,
            "evaluation": self.evaluation,
            "immune_evaluation": self.immune_evaluation,
            "nonimmune_evaluation": self.nonimmune_evaluation,
            "cohort_quality": self.cohort_quality,
            "training": self.manifest["training"],
            "default_mpo": default_mpo,
            "default_immune_mpo": default_immune_mpo,
            "immune_definition": {
                "roots": ["CL:0000738 leukocyte", "CL:0000129 microglial cell"],
                "required_channel": "specific",
            },
        }

    def list_mpos(self, query: dict[str, list[str]]) -> dict[str, object]:
        text = query.get("q", [""])[0].strip().lower()
        exact = query.get("exact", ["all"])[0]
        evidence = query.get("evidence", ["all"])[0]
        lineage = query.get("lineage", ["all"])[0]
        sort = query.get("sort", ["id"])[0]
        try:
            page = max(1, int(query.get("page", ["1"])[0]))
            page_size = min(100, max(10, int(query.get("page_size", ["30"])[0])))
        except ValueError:
            page, page_size = 1, 30

        rows = self.mpos
        if text:
            rows = [row for row in rows if text in str(row["search_text"])]
        if exact == "yes":
            rows = [row for row in rows if row["exact_available"]]
        elif exact == "no":
            rows = [row for row in rows if not row["exact_available"]]
        if evidence == "parent":
            rows = [row for row in rows if int(row["derived_count"]) > 0]
        elif evidence == "alias":
            rows = [row for row in rows if row["strong_evidence"]]
        elif evidence == "mechanism":
            rows = [row for row in rows if int(row["mechanism_count"]) > 0]
        if lineage == "immune_specific":
            rows = [row for row in rows if row["immune_specific"]]
        elif lineage == "immune_any":
            rows = [row for row in rows if row["immune_any"]]
        elif lineage == "nonimmune":
            rows = [row for row in rows if not row["immune_specific"]]

        if sort == "gate_desc":
            rows = sorted(rows, key=lambda row: (-float(row["gate_probability"]), row["mp_id"]))
        elif sort == "name":
            rows = sorted(rows, key=lambda row: (str(row["mp_name"]).lower(), row["mp_id"]))
        else:
            rows = sorted(rows, key=lambda row: str(row["mp_id"]))
        total = len(rows)
        start = (page - 1) * page_size
        public_rows = [
            {key: value for key, value in row.items() if key != "search_text"}
            for row in rows[start : start + page_size]
        ]
        return {
            "items": public_rows,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": max(1, (total + page_size - 1) // page_size),
        }

    def mpo(self, mp_id: str) -> dict[str, object] | None:
        if mp_id not in self.by_mp:
            return None
        return {
            "summary": {
                key: value
                for key, value in self.mp_summary[mp_id].items()
                if key != "search_text"
            },
            "candidates": self.by_mp[mp_id],
        }

    def evaluation_cases(self, query: dict[str, list[str]]) -> dict[str, object]:
        task = query.get("task", ["all"])[0]
        rank_filter = query.get("rank", ["all"])[0]
        sort = query.get("sort", ["worst"])[0]
        immune = query.get("immune", ["all"])[0]
        try:
            page = max(1, int(query.get("page", ["1"])[0]))
            page_size = min(100, max(10, int(query.get("page_size", ["30"])[0])))
        except ValueError:
            page, page_size = 1, 30
        rows = self.cases
        if task != "all":
            rows = [row for row in rows if row["task"] == task]
        if immune == "yes":
            rows = [row for row in rows if row["immune_related"]]
        elif immune == "no":
            rows = [row for row in rows if not row["immune_related"]]
        if rank_filter == "top1":
            rows = [row for row in rows if int(row["best_rank"]) == 1]
        elif rank_filter == "top10":
            rows = [row for row in rows if 0 < int(row["best_rank"]) <= 10]
        elif rank_filter == "over10":
            rows = [row for row in rows if int(row["best_rank"]) > 10]
        elif rank_filter == "missed":
            rows = [row for row in rows if int(row["best_rank"]) == 0]
        if sort == "best":
            rows = sorted(rows, key=lambda row: (int(row["best_rank"]) or 10**9, row["query_id"]))
        elif sort == "id":
            rows = sorted(rows, key=lambda row: (row["query_id"], row["task"]))
        else:
            rows = sorted(
                rows,
                key=lambda row: (
                    int(row["best_rank"]) == 0,
                    int(row["best_rank"]),
                    row["query_id"],
                ),
                reverse=True,
            )
        total = len(rows)
        start = (page - 1) * page_size
        return {
            "items": rows[start : start + page_size],
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": max(1, (total + page_size - 1) // page_size),
        }

    def mpo_tsv(self, mp_id: str) -> bytes | None:
        rows = self.by_mp.get(mp_id)
        if not rows:
            return None
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=self.mapping_fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")


def make_handler(store: ResultStore, static_dir: Path) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            sys.stdout.write(f"[web] {self.address_string()} {format % args}\n")

        def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _download(self, body: bytes, filename: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/tab-separated-values; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={quote(filename)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/health":
                self._json({"status": "ok", "result": store.result_dir.name})
                return
            if parsed.path == "/api/summary":
                self._json(store.summary())
                return
            if parsed.path == "/api/mpos":
                self._json(store.list_mpos(query))
                return
            if parsed.path.startswith("/api/mpo/"):
                mp_id = unquote(parsed.path.removeprefix("/api/mpo/"))
                payload = store.mpo(mp_id)
                self._json(payload or {"error": "MPO not found"}, HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND)
                return
            if parsed.path == "/api/evaluation-cases":
                self._json(store.evaluation_cases(query))
                return
            if parsed.path == "/download/mapping":
                self._download(store.mapping_path.read_bytes(), store.mapping_path.name)
                return
            if parsed.path == "/download/mpo":
                mp_id = query.get("id", [""])[0]
                body = store.mpo_tsv(mp_id)
                if body is None:
                    self._json({"error": "MPO not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._download(body, f"{mp_id.replace(':', '_')}_top20.tsv")
                return
            super().do_GET()

    return Handler


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Serve the MPO-HPO result inspector.")
    command.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8765)
    return command


def main() -> None:
    args = parser().parse_args()
    store = ResultStore(args.result_dir)
    handler = make_handler(store, Path(__file__).resolve().parent)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Loaded {store.overview['mapped_mpo_terms']:,} MPO and "
        f"{store.overview['mapping_rows']:,} mappings from {store.result_dir}"
    )
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
