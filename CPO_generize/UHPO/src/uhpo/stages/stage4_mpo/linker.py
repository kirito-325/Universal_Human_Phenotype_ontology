#!/usr/bin/env python3
"""Run the current HPO CL/GO linker with cleaned MPO evidence.

This adapter reuses the tested enrichment, exact CL-GO constraint, UBERON
compatibility, and marker/function reranking implementation. The only species
conversion is performed before this module, at the MGI mouse-gene -> human-gene
anchor step.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


from uhpo.stages.stage3_hpo.linker import (
    AnatomyConstraintIndex,
    CellMarkerIndex,
    ConstraintIndex,
    FunctionIndex,
    GeneralPhenotypeIndex,
    HpoTerm,
    NetworkGeneSetIndex,
    PPINetwork,
    SeedEvidence,
    TRIPLE_FIELDS,
    _stable_random_seed,
    infer_one,
)
from .paths import (
    CELL_GO_LINKS,
    CELL_MARKERS,
    CL_ANATOMY,
    GENERAL_GO_TERMS,
    GO_FUNCTIONS,
    HUMAN_GENE_EVIDENCE,
    LINK_DIR,
    MPO_ANATOMY,
    MPO_UBERON_ANCESTORS,
    CLEAN_MPO_OBO,
)


MP_ID = re.compile(r"^MP:\d+$")
MPO_SEED_FIELDS = [
    "mp_id", "mp_name", "gene", "seed_weight", "disease_count", "source_count", "supporting_evidence",
]
MPO_CELL_FIELDS = [
    field.replace("hpo_id", "mp_id").replace("hpo_name", "mp_name")
    for field in [
        "hpo_id", "hpo_name", "cell_rank", "cell_type_id", "cell_type", "overlap_genes",
        "overlap_count", "p_value", "q_value", "direct_p_value", "direct_q_value",
        "network_p_value", "network_q_value", "network_empirical_p_value",
        "network_mapped_gene_count", "support_mode", "weighted_support", "network_support",
        "network_support_genes", "score",
    ]
]
MPO_PHENOTYPE_FIELDS = [
    field.replace("hpo_id", "mp_id").replace("hpo_name", "mp_name")
    for field in [
        "hpo_id", "hpo_name", "phenotype_rank", "cell_phenotype_id", "cell_phenotype",
        "overlap_genes", "overlap_count", "p_value", "q_value", "direct_p_value", "direct_q_value",
        "network_p_value", "network_q_value", "network_empirical_p_value", "network_mapped_gene_count",
        "support_mode", "weighted_support", "network_support", "network_support_genes", "score",
        "go_constraint_cell_count", "go_constraint_status",
    ]
]
MPO_TRIPLE_FIELDS = [
    field.replace("hpo_id", "mp_id").replace("hpo_name", "mp_name") for field in TRIPLE_FIELDS
]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rename_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        if "hpo_id" in row:
            row["mp_id"] = row.pop("hpo_id")
        if "hpo_name" in row:
            row["mp_name"] = row.pop("hpo_name")
        if "supporting_diseases" in row:
            row["supporting_evidence"] = row.pop("supporting_diseases")
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = value.replace("HPO seed", "MPO seed").replace("HPO→", "MPO→")
        output.append(row)
    return output


@dataclass(frozen=True)
class MPOTerm:
    mp_id: str
    name: str


class MPOEvidenceIndex:
    """Exact MPO term -> human ortholog evidence, with no parent inheritance."""

    def __init__(self, mpo_obo: Path, evidence_path: Path) -> None:
        self.names: dict[str, str] = {}
        self.parents: dict[str, list[str]] = defaultdict(list)
        self.children: dict[str, list[str]] = defaultdict(list)
        self.term_gene_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.term_gene_evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.term_gene_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.term_gene_specificity: dict[str, float] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self._load_obo(mpo_obo)
        self._load_evidence(evidence_path)

    def _load_obo(self, path: Path) -> None:
        current = ""
        obsolete = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if line == "[Term]":
                current, obsolete = "", False
            elif line.startswith("id: MP:"):
                current = line.split(":", 1)[1].strip()
            elif current and line.startswith("name: "):
                self.names[current] = line[6:].strip()
            elif current and line.startswith("is_obsolete: true"):
                obsolete = True
            elif current and line.startswith("is_a: MP:"):
                parent = line.split(":", 1)[1].split("!", 1)[0].strip()
                self.parents[current].append(parent)
                self.children[parent].append(current)
            elif current and not line:
                if obsolete:
                    self.names.pop(current, None)
                    self.parents.pop(current, None)

    def _load_evidence(self, path: Path) -> None:
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"mp_id", "human_gene_symbol", "mouse_mgi_id", "evidence_weight"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"MPO evidence missing columns: {sorted(missing)}")
            seen: set[tuple[str, str, str, str, str]] = set()
            for row in reader:
                self.counts["evidence_rows"] += 1
                mp_id = row.get("mp_id", "").strip().upper()
                gene = row.get("human_gene_symbol", "").strip().upper()
                mgi_id = row.get("mouse_mgi_id", "").strip()
                if not MP_ID.fullmatch(mp_id) or mp_id not in self.names or not gene or not mgi_id:
                    self.counts["excluded_unknown_term_or_gene"] += 1
                    continue
                pmid = row.get("pmid", "").strip()
                genotype = row.get("genotype", "").strip()
                key = (mp_id, gene, mgi_id, pmid, genotype)
                if key in seen:
                    self.counts["excluded_duplicate_evidence"] += 1
                    continue
                seen.add(key)
                try:
                    weight = max(0.0, float(row.get("evidence_weight", "1")))
                except ValueError:
                    weight = 1.0
                if weight <= 0:
                    continue
                self.term_gene_weights[mp_id][gene] += weight
                evidence = "MGI:" + mgi_id.removeprefix("MGI:")
                if pmid:
                    evidence += "|PMID:" + pmid
                if genotype:
                    evidence += "|GENOTYPE:" + genotype
                self.term_gene_evidence[mp_id][gene].add(evidence)
                self.term_gene_sources[mp_id][gene].add(row.get("evidence_source", "MGI").strip() or "MGI")
                self.counts["retained_gene_phenotype_links"] += 1
        gene_terms: dict[str, set[str]] = defaultdict(set)
        for term, genes in self.term_gene_weights.items():
            for gene in genes:
                gene_terms[gene].add(term)
        n_terms = max(1, len(self.term_gene_weights))
        for gene, terms in gene_terms.items():
            self.term_gene_specificity[gene] = math.log((n_terms + 1) / (len(terms) + 1))
        self.counts["indexed_terms"] = len(self.term_gene_weights)
        self.counts["indexed_genes"] = len(gene_terms)

    def term(self, mp_id: str) -> HpoTerm:
        if mp_id not in self.names:
            raise KeyError(mp_id)
        return HpoTerm(mp_id, self.names[mp_id])

    def descendants(self, mp_id: str, include_self: bool = True) -> set[str]:
        values = {mp_id} if include_self else set()
        queue = deque([mp_id])
        while queue:
            current = queue.popleft()
            for child in self.children.get(current, []):
                if child not in values:
                    values.add(child)
                    queue.append(child)
        return values

    def seeds(self, mp_id: str) -> list[SeedEvidence]:
        rows: list[SeedEvidence] = []
        for gene, raw_weight in self.term_gene_weights.get(mp_id, {}).items():
            specificity = max(0.25, self.term_gene_specificity.get(gene, 0.25))
            source_count = len(self.term_gene_sources[mp_id][gene])
            rows.append(
                SeedEvidence(
                    gene=gene,
                    weight=raw_weight * specificity * min(1.30, 1.0 + 0.10 * max(0, source_count - 1)),
                    disease_count=len(self.term_gene_evidence[mp_id][gene]),
                    supporting_diseases=tuple(sorted(self.term_gene_evidence[mp_id][gene])),
                    source_count=source_count,
                )
            )
        return sorted(rows, key=lambda row: (-row.weight, row.gene))


class MPOAnatomyConstraintIndex(AnatomyConstraintIndex):
    """AnatomyConstraintIndex with MP IDs as the phenotype subject field."""

    def __init__(self, mpo_path: Path, cl_path: Path, ancestor_path: Path) -> None:
        self.mpo_path = Path(mpo_path)
        self.cl_path = Path(cl_path)
        self.ancestor_path = Path(ancestor_path)
        self.hpo_anatomy, hpo_labels = self._load(self.mpo_path, ("mp_id", "hpo_id", "subject_id"))
        self.cl_anatomy, cl_labels = self._load(self.cl_path, ("cl_id", "subject_id"))
        self.anatomy_labels = {**hpo_labels, **cl_labels}
        self.ancestors = self._load_ancestors(self.ancestor_path)
        self._hpo_specific = {}
        self._cl_specific = {}


def _manifest_input(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo-obo", type=Path, default=CLEAN_MPO_OBO)
    parser.add_argument("--evidence", type=Path, default=HUMAN_GENE_EVIDENCE)
    parser.add_argument("--markers", type=Path, default=CELL_MARKERS)
    parser.add_argument("--functions", type=Path, default=GO_FUNCTIONS)
    parser.add_argument("--constraints", type=Path, default=CELL_GO_LINKS)
    parser.add_argument("--general-phenotypes", type=Path, default=GENERAL_GO_TERMS)
    parser.add_argument("--mpo-anatomy", type=Path, default=MPO_ANATOMY)
    parser.add_argument("--cl-anatomy", type=Path, default=CL_ANATOMY)
    parser.add_argument("--uberon-ancestors", type=Path, default=MPO_UBERON_ANCESTORS)
    parser.add_argument("--output-dir", type=Path, default=LINK_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-cells", type=int, default=50)
    parser.add_argument("--top-phenotypes", type=int, default=250)
    parser.add_argument("--min-overlap", type=int, default=1)
    parser.add_argument("--top-general-phenotypes", type=int, default=100)
    parser.add_argument("--top-general-k", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpo = MPOEvidenceIndex(args.mpo_obo, args.evidence)
    markers = CellMarkerIndex(args.markers)
    functions = FunctionIndex(args.functions)
    constraints = ConstraintIndex(args.constraints)
    general = GeneralPhenotypeIndex(args.general_phenotypes)
    anatomy = MPOAnatomyConstraintIndex(args.mpo_anatomy, args.cl_anatomy, args.uberon_ancestors)
    eligible = [
        mp_id
        for mp_id in sorted(mpo.term_gene_weights)
        if anatomy.specific_hpo_anatomy(mp_id)
    ]
    if args.limit:
        eligible = eligible[: args.limit]
    all_seeds: list[dict[str, object]] = []
    all_cells: list[dict[str, object]] = []
    all_phenotypes: list[dict[str, object]] = []
    all_triples: list[dict[str, object]] = []
    all_general: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for index, mp_id in enumerate(eligible, start=1):
        seeds, cells, phenotypes, output = infer_one(
            hpo_index=mpo,  # infer_one only relies on the shared evidence-index protocol.
            hpo_id=mp_id,
            markers=markers,
            functions=functions,
            constraints=constraints,
            anatomy_constraints=anatomy,
            top_k=args.top_k,
            top_cells=args.top_cells,
            top_phenotypes=args.top_phenotypes,
            min_overlap=args.min_overlap,
            general_phenotypes=general,
            top_general_phenotypes=args.top_general_phenotypes,
            top_general_k=args.top_general_k,
        )
        all_seeds.extend(_rename_rows(seeds))
        all_cells.extend(_rename_rows(cells))
        all_phenotypes.extend(_rename_rows(phenotypes))
        all_triples.extend(_rename_rows(output["triples"]))
        all_general.extend(_rename_rows(output["general_triples"]))
        audit = dict(output["audit"])
        audit["mp_id"] = audit.pop("hpo_id")
        audit["mp_name"] = audit.pop("hpo_name")
        audits.append(audit)
        if index == 1 or index % 100 == 0 or index == len(eligible):
            print(json.dumps({"event": "processed", "current": index, "total": len(eligible), "mp_id": mp_id}, ensure_ascii=True), flush=True)

    _write_tsv(args.output_dir / "mpo_seed_genes.tsv", all_seeds, MPO_SEED_FIELDS)
    _write_tsv(args.output_dir / "mpo_cell_type_candidates.tsv", all_cells, MPO_CELL_FIELDS)
    _write_tsv(args.output_dir / "mpo_cell_phenotype_candidates.tsv", all_phenotypes, MPO_PHENOTYPE_FIELDS)
    _write_tsv(args.output_dir / "mpo_cell_mechanism_triples.tsv", all_triples, MPO_TRIPLE_FIELDS)
    _write_tsv(args.output_dir / "mpo_cell_general_mechanism_triples.tsv", all_general, MPO_TRIPLE_FIELDS)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": {
            "source_ontology": "MPO",
            "species_bridge": "mouse MP evidence -> MGI mouse MGI ID -> explicit MGI mouse-human homology -> human gene",
            "term_gene_evidence": "exact cleaned MPO term only; no MPO parent/child gene inheritance",
            "cell_and_go_method": "shared Stage 3 enrichment and exact-constraint implementation",
            "cl_go_constraint": "exact accepted direct CL-GO pair; no CL parent/child expansion",
            "anatomy_constraint": (
                "shared with the HPO-CPO linker: Ubergraph redundant reasoner MP/CL-to-UBERON "
                "materialization plus reasoned UBERON ancestor comparability"
            ),
            "anatomy_materialization_required": (
                "UPHENO:0000001 from reasoner graph; no MPO text-xref or "
                "MP-HPO bridge"
            ),
            "ppi": False,
            "specific_and_general_channels": True,
        },
        "inputs": {name: _manifest_input(path) for name, path in {
            "mpo_obo": args.mpo_obo,
            "evidence": args.evidence,
            "markers": args.markers,
            "functions": args.functions,
            "constraints": args.constraints,
            "general_phenotypes": args.general_phenotypes,
            "mpo_anatomy": args.mpo_anatomy,
            "cl_anatomy": args.cl_anatomy,
            "uberon_ancestors": args.uberon_ancestors,
        }.items()},
        "eligibility": {
            "terms_with_human_gene_evidence": len(mpo.term_gene_weights),
            "terms_with_specific_mpo_uberon": sum(bool(anatomy.specific_hpo_anatomy(mp_id)) for mp_id in mpo.term_gene_weights),
            "selected_term_count": len(eligible),
            "limit": args.limit,
        },
        "evidence_index": dict(mpo.counts),
        "cell_go_constraints": {"accepted_exact_rows": constraints.rows, "go_count": len(constraints.by_go)},
        "general_go": {"accepted_count": len(general.ids)},
        "output": {
            "seed_gene_rows": len(all_seeds),
            "cell_candidate_rows": len(all_cells),
            "phenotype_candidate_rows": len(all_phenotypes),
            "specific_triple_count": len(all_triples),
            "general_triple_count": len(all_general),
            "specific_mp_count": len({row["mp_id"] for row in all_triples}),
            "general_mp_count": len({row["mp_id"] for row in all_general}),
        },
        "queries": audits,
    }
    (args.output_dir / "mpo_cell_mechanism_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest["output"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
