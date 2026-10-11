from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.stats import hypergeom


from uhpo.stages.stage2_cpo.common import load_obo

from .paths import (
    CELL_GO_LINKS,
    CELL_MARKERS,
    CL_ANATOMY,
    GENERAL_GO_TERMS,
    GENES_TO_PHENOTYPE,
    GO_FUNCTIONS,
    HPO_ANATOMY,
    HPO_OBO,
    HPOA,
    LINK_DIR,
    UBERON_ANCESTORS,
)


FREQUENCY_WEIGHTS = {
    "HP:0040280": 1.00,  # obligate
    "HP:0040281": 0.90,  # very frequent
    "HP:0040282": 0.60,  # frequent
    "HP:0040283": 0.20,  # occasional
    "HP:0040284": 0.05,  # very rare
    "HP:0040285": 0.00,  # excluded
}
EVIDENCE_WEIGHTS = {"PCS": 1.00, "TAS": 0.85}
HPO_ID_RE = re.compile(r"HP:\d{7}")
DEFAULT_HPO_ANATOMY_PATH = HPO_ANATOMY
DEFAULT_CL_ANATOMY_PATH = CL_ANATOMY
DEFAULT_UBERON_ANCESTORS_PATH = UBERON_ANCESTORS
DEFAULT_GENERAL_PHENOTYPES_PATH = GENERAL_GO_TERMS

# Generic roots and non-locational groupings are not positive tissue evidence.
GENERIC_ANATOMY_IDS = frozenset(
    {
        "UBERON:0000061",  # anatomical structure
        "UBERON:0000062",  # organ
        "UBERON:0000064",  # organ part
        "UBERON:0000465",  # material anatomical entity
        "UBERON:0000466",  # immaterial anatomical entity
        "UBERON:0000467",  # anatomical system
        "UBERON:0000468",  # multicellular organism
        "UBERON:0000475",  # organism subdivision
        "UBERON:0000479",  # tissue
        "UBERON:0000481",  # multi-tissue structure
        "UBERON:0001062",  # anatomical entity
        "UBERON:0004120",  # mesoderm-derived structure
        "UBERON:0004121",  # ectoderm-derived structure
        "UBERON:0010000",  # multicellular anatomical structure
        "UBERON:0010314",  # structure with developmental contribution from neural crest
        "UBERON:0011216",  # organ system subdivision
        "UBERON:0011676",  # subdivision of organism along main body axis
        "UBERON:0013701",  # main body axis
        "UBERON:0013702",  # body proper
        "UBERON:0015203",  # non-connected functional system
        "UBERON:0015212",  # lateral structure
        "UBERON:0034923",  # disconnected anatomical group
        "UBERON:0034925",  # anatomical collection
    }
)

SEED_FIELDS = [
    "hpo_id", "hpo_name", "gene", "seed_weight", "disease_count",
    "source_count", "supporting_diseases",
]
CELL_FIELDS = [
    "hpo_id", "hpo_name", "cell_rank", "cell_type_id", "cell_type", "overlap_genes",
    "overlap_count", "p_value", "q_value", "direct_p_value", "direct_q_value",
    "network_p_value", "network_q_value", "network_empirical_p_value",
    "network_mapped_gene_count", "support_mode", "weighted_support",
    "network_support", "network_support_genes", "score",
]
PHENOTYPE_FIELDS = [
    "hpo_id", "hpo_name", "phenotype_rank", "cell_phenotype_id", "cell_phenotype", "overlap_genes",
    "overlap_count", "p_value", "q_value", "direct_p_value", "direct_q_value",
    "network_p_value", "network_q_value", "network_empirical_p_value",
    "network_mapped_gene_count", "support_mode", "weighted_support",
    "network_support", "network_support_genes", "score", "go_constraint_cell_count",
    "go_constraint_status",
]
TRIPLE_FIELDS = [
    "hpo_id", "hpo_name", "final_rank", "enrichment_pair_rank", "cell_rank",
    "phenotype_rank", "combined_rank_sum", "enrichment_rank_score",
    "three_way_overlap_score", "marker_function_overlap_score", "rerank_score",
    "rerank_route", "cell_type_id", "cell_type",
    "cell_phenotype_id", "cell_phenotype", "phenotype_scope", "scope_reason",
    "posterior_confidence",
    "confidence_grade", "evidence_score",
    "supporting_genes", "direct_support_genes", "three_way_overlap_count",
    "marker_support_genes",
    "function_support_genes", "marker_function_overlap_genes",
    "marker_function_overlap_count", "cell_marker_gene_count", "go_function_gene_count",
    "network_support_genes", "cell_q_value", "phenotype_q_value",
    "cell_direct_q_value", "phenotype_direct_q_value", "cell_network_q_value",
    "phenotype_network_q_value", "cell_support_mode", "phenotype_support_mode",
    "cell_overlap", "phenotype_overlap", "cell_score", "phenotype_score", "go_cl_relation",
    "go_constraint_status", "go_constraint_confidence",
    "go_constraint_decision_source", "anatomy_constraint_status", "compatible_anatomy_ids",
    "compatible_anatomy_labels", "evidence_summary",
]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _split(value: object, separators: str = "|,") -> list[str]:
    text = str(value or "")
    for separator in separators[1:]:
        text = text.replace(separator, separators[0])
    return [item.strip() for item in text.split(separators[0]) if item.strip()]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH adjusted values in the original order."""
    if not p_values:
        return []
    values = np.asarray([min(1.0, max(0.0, float(p))) for p in p_values])
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    n = len(values)
    for rank in range(n, 0, -1):
        index = int(order[rank - 1])
        running = min(running, values[index] * n / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _stable_random_seed(base_seed: int, *parts: str) -> int:
    payload = "\t".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _frequency_weight(value: str) -> float:
    value = value.strip()
    if not value:
        # Missing frequency is retained, but is deliberately weaker than an
        # obligate/very-frequent assertion.
        return 0.50
    if value in FREQUENCY_WEIGHTS:
        return FREQUENCY_WEIGHTS[value]
    if value.startswith("HP:"):
        return 0.50
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if match:
        denominator = float(match.group(2))
        return min(1.0, max(0.0, float(match.group(1)) / denominator)) if denominator else 0.5
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", value)
    if match:
        return min(1.0, max(0.0, float(match.group(1)) / 100.0))
    return 0.50


@dataclass(frozen=True)
class SeedEvidence:
    gene: str
    weight: float
    disease_count: int
    supporting_diseases: tuple[str, ...]
    source_count: int


@dataclass(frozen=True)
class SetResult:
    item_id: str
    label: str
    overlap_genes: tuple[str, ...]
    p_value: float
    q_value: float
    weighted_support: float
    network_support: float
    score: float
    direct_p_value: float = 1.0
    direct_q_value: float = 1.0
    network_p_value: float = 1.0
    network_q_value: float = 1.0
    network_empirical_p_value: float = 1.0
    network_mapped_gene_count: int = 0
    network_support_genes: tuple[str, ...] = ()
    support_mode: str = "direct"


@dataclass(frozen=True)
class HpoTerm:
    hpo_id: str
    name: str


class HPOEvidenceIndex:
    """Index high-quality HPOA rows joined to disease-gene records.

    HPOA itself stores disease-level annotations. The companion
    ``genes_to_phenotype.txt`` is used only for the disease-to-gene join; all
    phenotype evidence, frequency, aspect and evidence filtering comes from
    HPOA.
    """

    def __init__(self, hpo_obo: Path, hpoa: Path, gene_associations: Path) -> None:
        hpo_obo, hpoa, gene_associations = Path(hpo_obo), Path(hpoa), Path(gene_associations)
        graph = load_obo(hpo_obo)
        self.names: dict[str, str] = {
            term_id: str(data.get("name", ""))
            for term_id, data in graph.nodes(data=True)
            if HPO_ID_RE.fullmatch(str(term_id))
        }
        self.parents: dict[str, list[str]] = defaultdict(list)
        self.children: dict[str, list[str]] = defaultdict(list)
        for child, data in graph.nodes(data=True):
            child = str(child)
            for parent in data.get("is_a", []) if isinstance(data.get("is_a"), list) else []:
                parent = str(parent).split(" ! ", 1)[0]
                if HPO_ID_RE.fullmatch(parent):
                    self.parents[child].append(parent)
                    self.children[parent].append(child)
        for values in self.parents.values():
            values.sort()
        for values in self.children.values():
            values.sort()

        disease_genes: dict[str, set[str]] = defaultdict(set)
        with _open_text(gene_associations) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"gene_symbol", "disease_id"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Gene association file is missing columns: {sorted(required)}")
            for row in reader:
                gene = row.get("gene_symbol", "").strip().upper()
                disease = row.get("disease_id", "").strip()
                if gene and disease:
                    disease_genes[disease].add(gene)

        # term -> gene -> accumulated quality weight
        self.term_gene_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.term_gene_diseases: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.term_gene_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.counts: Counter[str] = Counter()
        with _open_text(hpoa) as handle:
            reader = csv.reader(
                (line for line in handle if not line.startswith("#")), delimiter="\t"
            )
            header = next(reader, [])
            positions = {name: index for index, name in enumerate(header)}
            needed = {"database_id", "hpo_id", "evidence", "frequency", "aspect", "qualifier"}
            missing = needed - set(positions)
            if missing:
                raise ValueError(f"HPOA is missing columns: {sorted(missing)}")
            for row in reader:
                self.counts["hpoa_rows"] += 1
                if len(row) <= max(positions.values()):
                    self.counts["malformed_hpoa_rows"] += 1
                    continue
                if row[positions["aspect"]].strip() != "P":
                    self.counts["excluded_non_phenotype_aspect"] += 1
                    continue
                if "NOT" in _split(row[positions["qualifier"]], separators="|,"):
                    self.counts["excluded_negative_annotations"] += 1
                    continue
                evidence = row[positions["evidence"]].strip()
                if evidence not in EVIDENCE_WEIGHTS:
                    self.counts["excluded_low_quality_evidence"] += 1
                    continue
                hpo_id = row[positions["hpo_id"]].strip()
                disease = row[positions["database_id"]].strip()
                if hpo_id not in self.names or not disease_genes.get(disease):
                    self.counts["excluded_unknown_term_or_gene"] += 1
                    continue
                freq = _frequency_weight(row[positions["frequency"]])
                if freq <= 0:
                    self.counts["excluded_zero_frequency"] += 1
                    continue
                source = disease.split(":", 1)[0]
                row_weight = freq * EVIDENCE_WEIGHTS[evidence]
                for gene in disease_genes[disease]:
                    self.term_gene_weights[hpo_id][gene] += row_weight
                    self.term_gene_diseases[hpo_id][gene].add(disease)
                    self.term_gene_sources[hpo_id][gene].add(source)
                    self.counts["retained_gene_phenotype_links"] += 1

        self.term_gene_specificity: dict[str, float] = {}
        self.term_information_content: dict[str, float] = {}
        gene_terms: dict[str, set[str]] = defaultdict(set)
        for term, genes in self.term_gene_weights.items():
            for gene in genes:
                gene_terms[gene].add(term)
        n_terms = max(1, len(self.term_gene_weights))
        n_genes = max(1, len(gene_terms))
        raw_ic: dict[str, float] = {}
        for term, genes in self.term_gene_weights.items():
            raw_ic[term] = -math.log((len(genes) + 1) / (n_genes + 1))
        max_ic = max(raw_ic.values(), default=1.0) or 1.0
        self.term_information_content = {
            term: 0.5 + 0.5 * value / max_ic for term, value in raw_ic.items()
        }
        for gene, terms in gene_terms.items():
            self.term_gene_specificity[gene] = math.log((n_terms + 1) / (len(terms) + 1))
        self.counts["indexed_terms"] = len(self.term_gene_weights)
        self.counts["indexed_genes"] = len(gene_terms)

    def descendants(self, term_id: str, include_self: bool = True) -> set[str]:
        if term_id not in self.names:
            return set()
        result = {term_id} if include_self else set()
        queue = deque([term_id])
        while queue:
            current = queue.popleft()
            for child in self.children.get(current, []):
                if child not in result:
                    result.add(child)
                    queue.append(child)
        return result

    def term(self, term_id: str) -> HpoTerm:
        if term_id not in self.names:
            raise KeyError(f"Unknown HPO term: {term_id}")
        return HpoTerm(term_id, self.names[term_id])

    def seeds(self, term_id: str, include_descendants: bool = False) -> list[SeedEvidence]:
        if term_id not in self.names:
            raise KeyError(f"Unknown HPO term: {term_id}")
        terms = self.descendants(term_id) if include_descendants else {term_id}
        aggregate: dict[str, float] = defaultdict(float)
        diseases: dict[str, set[str]] = defaultdict(set)
        sources: dict[str, set[str]] = defaultdict(set)
        for source_term in terms:
            for gene, weight in self.term_gene_weights.get(source_term, {}).items():
                aggregate[gene] += weight * self.term_information_content.get(source_term, 1.0)
                diseases[gene].update(self.term_gene_diseases[source_term][gene])
                sources[gene].update(self.term_gene_sources[source_term][gene])
        result = [
            SeedEvidence(
                gene=gene,
                weight=(
                    weight
                    * max(0.25, self.term_gene_specificity.get(gene, 0.25))
                    * min(1.30, 1.0 + 0.10 * max(0, len(sources[gene]) - 1))
                ),
                disease_count=len(diseases[gene]),
                supporting_diseases=tuple(sorted(diseases[gene])),
                source_count=len(sources[gene]),
            )
            for gene, weight in aggregate.items()
        ]
        return sorted(result, key=lambda item: (-item.weight, item.gene))


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")


class CellMarkerIndex:
    def __init__(self, path: Path) -> None:
        path = Path(path)
        self.genes: dict[str, dict[str, float]] = defaultdict(dict)
        self.labels: dict[str, str] = {}
        self.rows = 0
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                cell = (row.get("catalog_id") or row.get("cl_id") or "").strip()
                gene = (row.get("gene") or row.get("marker_gene") or "").strip().upper()
                if not cell or not gene:
                    continue
                self.rows += 1
                self.labels[cell] = (row.get("catalog_label") or row.get("catalog_cl_name") or cell).strip()
                rank = float(row.get("marker_rank") or row.get("catalog_rank") or 50)
                score = float(row.get("marker_score") or 1.0)
                # Rank is more stable across experiments than an uncalibrated score.
                self.genes[cell][gene] = max(self.genes[cell].get(gene, 0.0), score / math.log2(rank + 1.0))

    @property
    def universe(self) -> set[str]:
        return set(gene for genes in self.genes.values() for gene in genes)


class FunctionIndex:
    def __init__(self, path: Path, summary_path: Path | None = None) -> None:
        path = Path(path)
        summary_path = Path(summary_path) if summary_path else None
        self.genes: dict[str, set[str]] = defaultdict(set)
        self.labels: dict[str, str] = {}
        self.rows = 0
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                go_id = row.get("go_id", "").strip()
                gene = (row.get("human_gene_symbol") or row.get("gene_symbol") or "").strip().upper()
                if not go_id or not gene:
                    continue
                self.genes[go_id].add(gene)
                self.labels[go_id] = (row.get("go_name") or go_id).strip()
                self.rows += 1
        if summary_path and summary_path.exists():
            with summary_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    go_id = row.get("go_id", "").strip()
                    if go_id and row.get("go_name"):
                        self.labels[go_id] = row["go_name"].strip()

    @property
    def universe(self) -> set[str]:
        return set(gene for genes in self.genes.values() for gene in genes)


class ConstraintIndex:
    """Exact accepted cell-GO links; no CL parent/child expansion is permitted."""

    ACCEPTED_STATUSES = frozenset({"confirmed", "supported", "llm_supported"})

    def __init__(
        self,
        links_path: Path | None,
        classification_path: Path | None = None,
    ) -> None:
        del classification_path  # Legacy argument retained for API compatibility.
        links_path = Path(links_path) if links_path else None
        self.by_go: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.status: dict[str, str] = {}
        self.confidence: dict[str, str] = {}
        self.link_status: dict[tuple[str, str], str] = {}
        self.link_confidence: dict[tuple[str, str], str] = {}
        self.decision_source: dict[tuple[str, str], str] = {}
        self.decision_reason: dict[tuple[str, str], str] = {}
        self.rows = 0
        if not links_path or not links_path.exists():
            return
        with links_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"go_id", "cl_id", "final_link_status", "confidence"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{links_path} is not a final direct cell-GO link table; "
                    f"missing columns: {sorted(missing)}"
                )
            seen: set[tuple[str, str]] = set()
            statuses_by_go: dict[str, set[str]] = defaultdict(set)
            confidence_by_go: dict[str, set[str]] = defaultdict(set)
            for row in reader:
                go_id = row.get("go_id", "").strip()
                cell = row.get("cl_id", "").strip()
                status = row.get("final_link_status", "").strip()
                confidence = row.get("confidence", "").strip()
                if not go_id or not cell or status not in self.ACCEPTED_STATUSES:
                    continue
                if _as_bool(row.get("review_required", "false")):
                    continue
                key = (go_id, cell)
                if key in seen:
                    raise ValueError(
                        f"Final direct cell-GO links contain duplicate pair: {go_id}, {cell}"
                    )
                seen.add(key)
                label = (
                    row.get("catalog_label")
                    or row.get("cl_name")
                    or cell
                ).strip()
                self.by_go[go_id].append((cell, label, "direct"))
                self.link_status[key] = status
                self.link_confidence[key] = confidence
                self.decision_source[key] = row.get("decision_source", "").strip()
                self.decision_reason[key] = row.get("decision_reason", "").strip()
                statuses_by_go[go_id].add(status)
                confidence_by_go[go_id].add(confidence)
                self.rows += 1
        for go_id, values in self.by_go.items():
            self.by_go[go_id] = sorted(values, key=lambda value: value[0])
            self.status[go_id] = "|".join(sorted(statuses_by_go[go_id]))
            self.confidence[go_id] = "|".join(sorted(confidence_by_go[go_id]))


class GeneralPhenotypeIndex:
    """Accepted cell-general GO terms kept separate from exact cell-GO links."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Required cell-general GO table not found: {self.path}"
            )
        self.labels: dict[str, str] = {}
        self.confidence: dict[str, str] = {}
        self.reason: dict[str, str] = {}
        self.model: dict[str, str] = {}
        with self.path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "go_id",
                "go_name",
                "adjudicated_scope",
                "llm_confidence",
                "decision_accepted",
                "review_required",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{self.path} is not an accepted cell-general GO table; "
                    f"missing columns: {sorted(missing)}"
                )
            for row in reader:
                go_id = row.get("go_id", "").strip()
                if (
                    not go_id
                    or row.get("adjudicated_scope", "").strip() != "cell_general"
                    or not _as_bool(row.get("decision_accepted", ""))
                    or _as_bool(row.get("review_required", ""))
                    or row.get("llm_confidence", "").strip()
                    not in {"high", "medium"}
                ):
                    continue
                if go_id in self.labels:
                    raise ValueError(
                        f"Cell-general GO table contains duplicate GO: {go_id}"
                    )
                self.labels[go_id] = row.get("go_name", "").strip() or go_id
                self.confidence[go_id] = row.get("llm_confidence", "").strip()
                self.reason[go_id] = row.get("llm_reason", "").strip()
                self.model[go_id] = row.get("llm_model", "").strip()

    @property
    def ids(self) -> set[str]:
        return set(self.labels)


class AnatomyConstraintIndex:
    """Strict HPO-to-CL tissue compatibility from materialized UBERON closure."""

    def __init__(self, hpo_path: Path, cl_path: Path, ancestor_path: Path) -> None:
        self.hpo_path = Path(hpo_path)
        self.cl_path = Path(cl_path)
        self.ancestor_path = Path(ancestor_path)
        self.hpo_anatomy, hpo_labels = self._load(
            self.hpo_path, ("hpo_id", "subject_id")
        )
        self.cl_anatomy, cl_labels = self._load(
            self.cl_path, ("cl_id", "subject_id")
        )
        self.anatomy_labels = {**hpo_labels, **cl_labels}
        self.ancestors = self._load_ancestors(self.ancestor_path)
        self._hpo_specific: dict[str, tuple[str, ...]] = {}
        self._cl_specific: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _load(
        path: Path, subject_fields: Sequence[str]
    ) -> tuple[dict[str, set[str]], dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Required anatomy constraint file not found: {path}")
        mappings: dict[str, set[str]] = defaultdict(set)
        labels: dict[str, str] = {}
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            subject_field = next((field for field in subject_fields if field in fields), None)
            anatomy_field = next(
                (field for field in ("uberon_id", "anatomy_id") if field in fields), None
            )
            if not subject_field or not anatomy_field:
                raise ValueError(
                    f"{path} must contain one of {subject_fields} and uberon_id/anatomy_id"
                )
            for row in reader:
                subject = row.get(subject_field, "").strip()
                anatomy = row.get(anatomy_field, "").strip()
                if not subject or not anatomy.startswith("UBERON:"):
                    continue
                mappings[subject].add(anatomy)
                label = (row.get("uberon_label") or row.get("anatomy_label") or "").strip()
                if label:
                    labels[anatomy] = label
        return dict(mappings), labels

    @staticmethod
    def _load_ancestors(path: Path) -> dict[str, set[str]]:
        if not path.exists():
            raise FileNotFoundError(f"Required UBERON ancestor file not found: {path}")
        ancestors: dict[str, set[str]] = defaultdict(set)
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            subject_field = next(
                (field for field in ("uberon_id", "subject_id") if field in fields), None
            )
            ancestor_field = next(
                (
                    field
                    for field in ("ancestor_uberon_id", "ancestor_id")
                    if field in fields
                ),
                None,
            )
            if not subject_field or not ancestor_field:
                raise ValueError(
                    f"{path} must contain uberon_id and ancestor_uberon_id"
                )
            for row in reader:
                subject = row.get(subject_field, "").strip()
                ancestor = row.get(ancestor_field, "").strip()
                if subject.startswith("UBERON:") and ancestor.startswith("UBERON:"):
                    ancestors[subject].add(ancestor)
        return dict(ancestors)

    def _most_specific(
        self,
        subject_id: str,
        mappings: Mapping[str, set[str]],
        cache: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        if subject_id in cache:
            return cache[subject_id]
        values = {
            anatomy_id
            for anatomy_id in mappings.get(subject_id, set()) - GENERIC_ANATOMY_IDS
            if anatomy_id in self.ancestors
        }
        specific = tuple(
            sorted(
                candidate
                for candidate in values
                if not any(
                    candidate != other and candidate in self.ancestors.get(other, set())
                    for other in values
                )
            )
        )
        cache[subject_id] = specific
        return specific

    def compatible_anatomy(self, hpo_id: str, cl_id: str) -> tuple[str, ...]:
        hpo_specific = self._most_specific(
            hpo_id, self.hpo_anatomy, self._hpo_specific
        )
        cl_specific = self._most_specific(cl_id, self.cl_anatomy, self._cl_specific)
        compatible: set[str] = set()
        for hpo_anatomy in hpo_specific:
            for cl_anatomy in cl_specific:
                if hpo_anatomy == cl_anatomy:
                    compatible.add(hpo_anatomy)
                elif hpo_anatomy in self.ancestors.get(cl_anatomy, set()):
                    compatible.add(cl_anatomy)
                elif cl_anatomy in self.ancestors.get(hpo_anatomy, set()):
                    compatible.add(hpo_anatomy)
        return tuple(sorted(compatible))

    def specific_hpo_anatomy(self, hpo_id: str) -> tuple[str, ...]:
        """Return non-generic, most-specific anatomy targets for an HPO term."""
        return self._most_specific(hpo_id, self.hpo_anatomy, self._hpo_specific)

    def anatomy_labels_for(self, anatomy_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(self.anatomy_labels.get(anatomy_id, anatomy_id) for anatomy_id in anatomy_ids)


@dataclass
class NetworkResult:
    scores: dict[str, float]
    z_scores: dict[str, float]
    null_permutations: int
    iterations: int
    seed_genes: tuple[str, ...]
    input_seed_count: int
    mapped_seed_count: int
    seed_weight_coverage: float
    score_vector: np.ndarray
    z_vector: np.ndarray


@dataclass(frozen=True)
class NetworkSetEvidence:
    p_value: float
    q_value: float
    score: float
    mapped_gene_count: int
    support_genes: tuple[str, ...]


@dataclass
class PPINetwork:
    path: Path
    nodes: tuple[str, ...]
    index: dict[str, int]
    adjacency: csc_matrix
    degrees: np.ndarray
    degree_bins: tuple[np.ndarray, ...]
    edge_count: int
    input_edge_count: int
    duplicate_edge_count: int

    @classmethod
    def from_path(cls, path: Path) -> "PPINetwork":
        path = Path(path)
        edge_weights: dict[tuple[str, str], float] = {}
        input_edge_count = 0
        duplicate_edge_count = 0
        with _open_text(path) as handle:
            reader = csv.reader(
                (line for line in handle if line.strip() and not line.startswith("#")),
                delimiter="\t",
            )
            for row in reader:
                if len(row) < 2:
                    continue
                if row[0].strip().lower() in {"gene_a", "source", "protein_a"}:
                    continue
                a, b = row[0].strip().upper(), row[1].strip().upper()
                if not a or not b or a == b:
                    continue
                try:
                    weight = float(row[2]) if len(row) > 2 and row[2] else 1.0
                except ValueError:
                    weight = 1.0
                if not math.isfinite(weight) or weight <= 0:
                    continue
                input_edge_count += 1
                edge = tuple(sorted((a, b)))
                if edge in edge_weights:
                    duplicate_edge_count += 1
                    edge_weights[edge] = max(edge_weights[edge], weight)
                else:
                    edge_weights[edge] = weight
        nodes = tuple(sorted({gene for edge in edge_weights for gene in edge}))
        if not nodes:
            raise ValueError(f"No usable PPI edges found in {path}")
        index = {gene: position for position, gene in enumerate(nodes)}
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        degrees = np.zeros(len(nodes), dtype=np.int32)
        for (a, b), weight in edge_weights.items():
            ia, ib = index[a], index[b]
            rows.extend((ia, ib))
            columns.extend((ib, ia))
            values.extend((weight, weight))
            degrees[ia] += 1
            degrees[ib] += 1
        adjacency = csc_matrix(
            (values, (rows, columns)), shape=(len(nodes), len(nodes)), dtype=float
        )
        column_sums = np.asarray(adjacency.sum(axis=0)).ravel()
        inverse = np.zeros_like(column_sums)
        inverse[column_sums > 0] = 1.0 / column_sums[column_sums > 0]
        adjacency = adjacency @ diags(inverse, format="csc")
        bins: dict[int, list[int]] = defaultdict(list)
        for node_index, degree in enumerate(degrees):
            bins[int(math.log2(int(degree) + 1))].append(node_index)
        degree_bins = tuple(
            np.asarray(indices, dtype=np.int32)
            for _, indices in sorted(bins.items())
            if indices
        )
        return cls(
            path=path,
            nodes=nodes,
            index=index,
            adjacency=adjacency,
            degrees=degrees,
            degree_bins=degree_bins,
            edge_count=len(edge_weights),
            input_edge_count=input_edge_count,
            duplicate_edge_count=duplicate_edge_count,
        )

    def propagate(
        self,
        seed_weights: Mapping[str, float],
        permutations: int = 1000,
        restart: float = 0.65,
        iterations: int = 100,
        tolerance: float = 1e-10,
    ) -> NetworkResult:
        if not 0.0 < restart <= 1.0:
            raise ValueError("restart must be in (0, 1]")
        positive_weights = {
            gene.upper(): float(weight)
            for gene, weight in seed_weights.items()
            if math.isfinite(float(weight)) and float(weight) > 0
        }
        total_weight = sum(positive_weights.values())
        mapped = {
            gene: weight for gene, weight in positive_weights.items() if gene in self.index
        }
        mapped_weight = sum(mapped.values())
        empty = np.zeros(len(self.nodes), dtype=float)
        if mapped_weight <= 0:
            return NetworkResult(
                {}, {}, max(0, int(permutations)), 0, (), len(positive_weights), 0,
                0.0, empty, empty.copy(),
            )
        p0 = np.zeros(len(self.nodes), dtype=float)
        for gene, weight in mapped.items():
            p0[self.index[gene]] = weight
        p0 /= mapped_weight
        observed, actual_iterations = _rwr(
            self.adjacency, p0, restart, iterations, tolerance
        )
        degree_z = np.zeros(len(self.nodes), dtype=float)
        for indices in self.degree_bins:
            values = observed[indices]
            standard_deviation = float(values.std())
            if standard_deviation > 0:
                degree_z[indices] = (values - float(values.mean())) / standard_deviation
        seed_indices = np.asarray([self.index[gene] for gene in mapped], dtype=np.int32)
        degree_z[seed_indices] = 0.0
        return NetworkResult(
            scores={gene: float(observed[index]) for index, gene in enumerate(self.nodes)},
            z_scores={
                gene: float(degree_z[index])
                for index, gene in enumerate(self.nodes)
                if degree_z[index] != 0.0
            },
            null_permutations=max(0, int(permutations)),
            iterations=actual_iterations,
            seed_genes=tuple(sorted(mapped)),
            input_seed_count=len(positive_weights),
            mapped_seed_count=len(mapped),
            seed_weight_coverage=mapped_weight / total_weight if total_weight else 0.0,
            score_vector=observed,
            z_vector=degree_z,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "nodes": len(self.nodes),
            "edges": self.edge_count,
            "input_edges": self.input_edge_count,
            "duplicate_edges_collapsed": self.duplicate_edge_count,
            "degree_bins": len(self.degree_bins),
        }


@dataclass(frozen=True)
class NetworkGeneSetIndex:
    item_ids: tuple[str, ...]
    matrix: csr_matrix
    mapped_counts: np.ndarray
    genes_by_item: tuple[tuple[str, ...], ...]

    @classmethod
    def build(
        cls,
        network: PPINetwork,
        sets: Mapping[str, Iterable[str]],
        allowed_ids: set[str] | None = None,
    ) -> "NetworkGeneSetIndex":
        item_ids: list[str] = []
        rows: list[int] = []
        columns: list[int] = []
        genes_by_item: list[tuple[str, ...]] = []
        for item_id in sorted(sets):
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            genes = tuple(sorted({gene for gene in sets[item_id] if gene in network.index}))
            if not genes:
                continue
            row_index = len(item_ids)
            item_ids.append(item_id)
            genes_by_item.append(genes)
            rows.extend([row_index] * len(genes))
            columns.extend(network.index[gene] for gene in genes)
        matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(item_ids), len(network.nodes)),
        )
        mapped_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int32)
        return cls(tuple(item_ids), matrix, mapped_counts, tuple(genes_by_item))

    def score(
        self,
        network: PPINetwork,
        result: NetworkResult,
        permutations: int,
        random_seed: int,
        support_z: float = 1.645,
    ) -> dict[str, NetworkSetEvidence]:
        permutations = max(0, int(permutations))
        if not self.item_ids or result.mapped_seed_count == 0 or permutations < 2:
            return {}
        signal = np.maximum(result.z_vector, 0.0).astype(np.float32, copy=False)
        denominators = np.sqrt(np.maximum(1, self.mapped_counts)).astype(float)
        observed = np.asarray(self.matrix @ signal).ravel() / denominators
        null_scores = np.empty(
            (permutations, len(self.item_ids)), dtype=np.float32
        )
        rng = np.random.default_rng(random_seed)
        shuffled = np.empty_like(signal)
        for permutation_index in range(permutations):
            for indices in network.degree_bins:
                shuffled[indices] = signal[rng.permutation(indices)]
            null_scores[permutation_index] = (
                np.asarray(self.matrix @ shuffled).ravel() / denominators
            )

        # GSEA-style normalization and pooling avoids the 1/(B+1) resolution
        # ceiling of separate permutation P values when thousands of gene sets
        # are tested. Pooling is restricted by log2 mapped-set-size whenever a
        # stratum has enough sets to define a stable empirical tail.
        null_means = null_scores.mean(axis=0, dtype=np.float64)
        null_standard_deviations = null_scores.std(axis=0, ddof=1, dtype=np.float64)
        valid = null_standard_deviations > 1e-12
        observed_standardized = np.zeros(len(self.item_ids), dtype=float)
        observed_standardized[valid] = (
            observed[valid] - null_means[valid]
        ) / null_standard_deviations[valid]
        null_standardized = np.zeros_like(null_scores)
        null_standardized[:, valid] = (
            null_scores[:, valid] - null_means[valid]
        ) / null_standard_deviations[valid]
        p_values = np.ones(len(self.item_ids), dtype=float)
        size_bins = np.floor(np.log2(np.maximum(1, self.mapped_counts))).astype(int)
        all_valid_null = np.sort(null_standardized[:, valid].ravel())
        for size_bin in np.unique(size_bins):
            member_indices = np.flatnonzero((size_bins == size_bin) & valid)
            if not len(member_indices):
                continue
            if len(member_indices) >= 20:
                pooled_null = np.sort(null_standardized[:, member_indices].ravel())
            else:
                pooled_null = all_valid_null
            if not len(pooled_null):
                continue
            insertion_points = np.searchsorted(
                pooled_null, observed_standardized[member_indices], side="left"
            )
            exceedances = len(pooled_null) - insertion_points
            p_values[member_indices] = (exceedances + 1.0) / (
                len(pooled_null) + 1.0
            )
        q_values = benjamini_hochberg(p_values.tolist())
        seed_genes = set(result.seed_genes)
        evidence: dict[str, NetworkSetEvidence] = {}
        for index, item_id in enumerate(self.item_ids):
            ranked = sorted(
                (
                    (gene, float(result.z_scores.get(gene, 0.0)))
                    for gene in self.genes_by_item[index]
                    if gene not in seed_genes
                ),
                key=lambda item: (-item[1], item[0]),
            )
            support_genes = tuple(
                gene for gene, z_score in ranked[:10] if z_score >= support_z
            )
            evidence[item_id] = NetworkSetEvidence(
                p_value=float(p_values[index]),
                q_value=float(q_values[index]),
                score=float(observed_standardized[index]),
                mapped_gene_count=int(self.mapped_counts[index]),
                support_genes=support_genes,
            )
        return evidence


def _rwr(
    adjacency: csc_matrix,
    p0: np.ndarray,
    restart: float,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    p = p0.copy()
    actual_iterations = 0
    for actual_iterations in range(1, iterations + 1):
        updated = (1.0 - restart) * (adjacency @ p) + restart * p0
        if float(np.abs(updated - p).sum()) <= tolerance:
            p = updated
            break
        p = updated
    return p, actual_iterations


def network_diffusion(
    ppi_path: Path | PPINetwork,
    seed_weights: Mapping[str, float],
    permutations: int = 1000,
    restart: float = 0.65,
    iterations: int = 100,
    tolerance: float = 1e-10,
    random_seed: int = 13,
) -> NetworkResult:
    """Run weighted RWR; gene-set permutations are applied during enrichment."""
    _ = random_seed  # Retained for compatibility with the former public helper.
    network = ppi_path if isinstance(ppi_path, PPINetwork) else PPINetwork.from_path(ppi_path)
    return network.propagate(
        seed_weights,
        permutations=permutations,
        restart=restart,
        iterations=iterations,
        tolerance=tolerance,
    )


def _enrichment(
    sets: Mapping[str, set[str]],
    labels: Mapping[str, str],
    seed_weights: Mapping[str, float],
    universe: set[str],
    network_evidence: Mapping[str, NetworkSetEvidence] | None = None,
    min_overlap: int = 1,
    network_q_threshold: float = 0.05,
    network_min_support_genes: int = 2,
) -> list[SetResult]:
    seed_genes = set(seed_weights) & universe
    if not universe:
        return []
    n = len(seed_genes)
    direct_raw: list[tuple[str, str, tuple[str, ...], float, float]] = []
    for item_id, item_genes in sets.items():
        members = set(item_genes) & universe
        overlap = tuple(sorted(seed_genes & members))
        if len(overlap) < min_overlap or not members:
            continue
        direct_p = float(hypergeom.sf(len(overlap) - 1, len(universe), len(members), n)) if overlap else 1.0
        weighted = sum(seed_weights[gene] for gene in overlap) / (sum(seed_weights.values()) or 1.0)
        direct_raw.append((item_id, labels.get(item_id, item_id), overlap, direct_p, weighted))
    adjusted = benjamini_hochberg([row[3] for row in direct_raw])
    direct_by_id = {
        row[0]: (row, q_value) for row, q_value in zip(direct_raw, adjusted)
    }
    candidate_ids = set(direct_by_id)
    for item_id, evidence in (network_evidence or {}).items():
        if (
            evidence.q_value <= network_q_threshold
            and len(evidence.support_genes) >= network_min_support_genes
        ):
            candidate_ids.add(item_id)
    results: list[SetResult] = []
    for item_id in candidate_ids:
        direct = direct_by_id.get(item_id)
        evidence = (network_evidence or {}).get(item_id)
        network_pass = bool(
            evidence
            and evidence.q_value <= network_q_threshold
            and len(evidence.support_genes) >= network_min_support_genes
        )
        if direct:
            row, direct_q = direct
            _, label, overlap, direct_p, weighted = row
            score = -math.log10(max(direct_q, 1e-300)) + 2.0 * weighted
            p_value, q_value = direct_p, direct_q
            support_mode = "direct_and_network" if network_pass else "direct"
        else:
            label = labels.get(item_id, item_id)
            overlap = ()
            direct_p = direct_q = 1.0
            weighted = 0.0
            p_value = evidence.p_value if evidence else 1.0
            q_value = evidence.q_value if evidence else 1.0
            score = -math.log10(max(q_value, 1e-300))
            support_mode = "network_only"
        results.append(
            SetResult(
                item_id=item_id,
                label=label,
                overlap_genes=overlap,
                p_value=p_value,
                q_value=q_value,
                weighted_support=weighted,
                network_support=evidence.score if network_pass and evidence else 0.0,
                score=score,
                direct_p_value=direct_p,
                direct_q_value=direct_q,
                network_p_value=evidence.p_value if evidence else 1.0,
                network_q_value=evidence.q_value if evidence else 1.0,
                network_empirical_p_value=evidence.p_value if evidence else 1.0,
                network_mapped_gene_count=evidence.mapped_gene_count if evidence else 0,
                network_support_genes=evidence.support_genes if network_pass and evidence else (),
                support_mode=support_mode,
            )
        )
    return sorted(results, key=lambda row: (-row.score, row.q_value, row.item_id))


def _enrichment_rank_score(
    cell_rank: int,
    phenotype_rank: int,
    cell_count: int,
    phenotype_count: int,
) -> float:
    """Normalize the sum of two independent one-based ranks to [0, 1]."""
    worst_sum = max(2, cell_count + phenotype_count)
    if worst_sum == 2:
        return 1.0
    return min(
        1.0,
        max(0.0, 1.0 - ((cell_rank + phenotype_rank) - 2) / (worst_sum - 2)),
    )


def _rank_pairs(
    rows: list[dict[str, object]],
    cell_count: int,
    phenotype_count: int,
    overlap_weight: float = 0.15,
    rank_weight: float = 0.55,
    three_way_weight: float = 0.30,
) -> list[dict[str, object]]:
    """Assign the enrichment-only rank, then rerank with gene consistency."""
    initial = sorted(
        rows,
        key=lambda row: (
            int(row["combined_rank_sum"]),
            max(int(row["cell_rank"]), int(row["phenotype_rank"])),
            int(row["cell_rank"]),
            int(row["phenotype_rank"]),
            str(row["cell_phenotype_id"]),
            str(row["cell_type_id"]),
        ),
    )
    for rank, row in enumerate(initial, start=1):
        enrichment_score = _enrichment_rank_score(
            int(row["cell_rank"]),
            int(row["phenotype_rank"]),
            cell_count,
            phenotype_count,
        )
        three_way_score = float(row.get("three_way_overlap_score") or 0.0)
        overlap_score = float(row["marker_function_overlap_score"])
        rerank_score = (
            rank_weight * enrichment_score
            + three_way_weight * three_way_score
            + overlap_weight * overlap_score
        )
        row["enrichment_pair_rank"] = rank
        row["enrichment_rank_score"] = enrichment_score
        row["rerank_score"] = rerank_score
        # Preserve the established output column as the final sortable score.
        row["evidence_score"] = rerank_score

    final = sorted(
        initial,
        key=lambda row: (
            -float(row["rerank_score"]),
            int(row["combined_rank_sum"]),
            -int(row["marker_function_overlap_count"]),
            str(row["cell_phenotype_id"]),
            str(row["cell_type_id"]),
        ),
    )
    for rank, row in enumerate(final, start=1):
        row["final_rank"] = rank
        for field in (
            "enrichment_rank_score",
            "three_way_overlap_score",
            "marker_function_overlap_score",
            "rerank_score",
            "evidence_score",
        ):
            row[field] = f"{float(row[field]):.6f}"
    return final


def _three_way_overlap_score(
    marker_support: Iterable[str],
    function_support: Iterable[str],
    three_way_genes: Iterable[str],
) -> float:
    """Score HPO-supported agreement across the marker and function axes."""
    marker_support = set(marker_support)
    function_support = set(function_support)
    three_way_genes = set(three_way_genes)
    denominator = min(len(marker_support), len(function_support))
    if not denominator or not three_way_genes:
        return 0.0
    concordance = len(three_way_genes) / denominator
    evidence_strength = min(1.0, len(three_way_genes) / 2.0)
    return min(1.0, concordance * evidence_strength)


def _grade(
    direct_count: int,
    marker_count: int,
    function_count: int,
    cell_q: float,
    phenotype_q: float,
) -> tuple[str, float]:
    if direct_count >= 2 and marker_count >= 1 and function_count >= 1 and cell_q <= 0.05 and phenotype_q <= 0.05:
        return "high", 0.90
    if marker_count >= 1 and function_count >= 1:
        return "medium", 0.60
    return "needs_experimental_validation", 0.35


def infer_one(
    hpo_index: HPOEvidenceIndex,
    hpo_id: str,
    markers: CellMarkerIndex,
    functions: FunctionIndex,
    constraints: ConstraintIndex,
    anatomy_constraints: AnatomyConstraintIndex,
    top_k: int = 100,
    top_cells: int = 50,
    top_phenotypes: int = 250,
    min_overlap: int = 1,
    network: NetworkResult | None = None,
    phenotype_network_evidence: Mapping[str, NetworkSetEvidence] | None = None,
    network_q_threshold: float = 0.05,
    network_min_support_genes: int = 2,
    pair_overlap_weight: float = 0.15,
    pair_rank_weight: float = 0.55,
    pair_three_way_weight: float = 0.30,
    general_phenotypes: GeneralPhenotypeIndex | None = None,
    general_network_evidence: Mapping[str, NetworkSetEvidence] | None = None,
    top_general_phenotypes: int = 100,
    top_general_k: int = 100,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    term = hpo_index.term(hpo_id)
    seed_rows = hpo_index.seeds(hpo_id)
    seed_weights = {row.gene: row.weight for row in seed_rows}
    cell_results = _enrichment(
        markers.genes,
        markers.labels,
        seed_weights,
        markers.universe,
        min_overlap=min_overlap,
    )
    phenotype_results = _enrichment(
        functions.genes,
        functions.labels,
        seed_weights,
        functions.universe,
        network_evidence=phenotype_network_evidence,
        min_overlap=min_overlap,
        network_q_threshold=network_q_threshold,
        network_min_support_genes=network_min_support_genes,
    )
    general_results: list[SetResult] = []
    if general_phenotypes:
        general_sets = {
            go_id: functions.genes[go_id]
            for go_id in general_phenotypes.ids
            if go_id in functions.genes
        }
        general_labels = {
            go_id: functions.labels.get(
                go_id, general_phenotypes.labels.get(go_id, go_id)
            )
            for go_id in general_sets
        }
        general_results = _enrichment(
            general_sets,
            general_labels,
            seed_weights,
            functions.universe,
            network_evidence=general_network_evidence,
            min_overlap=min_overlap,
            network_q_threshold=network_q_threshold,
            network_min_support_genes=network_min_support_genes,
        )[:top_general_phenotypes]
    cell_results = cell_results[:top_cells]
    phenotype_results = phenotype_results[:top_phenotypes]
    cell_map = {row.item_id: row for row in cell_results}
    cell_ranks = {row.item_id: rank for rank, row in enumerate(cell_results, 1)}
    phenotype_ranks = {
        row.item_id: rank for rank, row in enumerate(phenotype_results, 1)
    }
    pair_rows: list[dict[str, object]] = []
    candidate_cells = {row.item_id for row in cell_results}
    candidate_pair_count = len(candidate_cells) * len(phenotype_results)
    go_cl_allowed_pair_count = 0
    anatomy_allowed_pair_count = 0
    for phenotype in phenotype_results:
        allowed = constraints.by_go.get(phenotype.item_id, [])
        allowed_cells = [
            (cell, label, relation)
            for cell, label, relation in allowed
            if cell in candidate_cells
        ]
        go_cl_allowed_pair_count += len(allowed_cells)
        for cell_id, cell_label, relation in allowed_cells:
            compatible_anatomy = anatomy_constraints.compatible_anatomy(hpo_id, cell_id)
            if not compatible_anatomy:
                continue
            anatomy_allowed_pair_count += 1
            marker_genes = set(markers.genes.get(cell_id, {}))
            function_genes = set(functions.genes.get(phenotype.item_id, set()))
            direct = tuple(sorted(set(seed_weights) & marker_genes & function_genes))
            marker_support = tuple(sorted(set(seed_weights) & marker_genes))
            function_support = tuple(sorted(set(seed_weights) & function_genes))
            marker_function_overlap = tuple(sorted(marker_genes & function_genes))
            marker_function_denominator = min(len(marker_genes), len(function_genes))
            marker_function_overlap_score = (
                len(marker_function_overlap) / marker_function_denominator
                if marker_function_denominator
                else 0.0
            )
            three_way_overlap_score = _three_way_overlap_score(
                marker_support, function_support, direct
            )
            cell = cell_map[cell_id]
            network_support = tuple(
                sorted(phenotype.network_support_genes)
            )
            if not marker_support and not function_support and not network_support:
                continue
            link_key = (phenotype.item_id, cell_id)
            status = constraints.link_status.get(link_key, "")
            c_q = cell.q_value
            p_q = phenotype.q_value
            grade, posterior = _grade(
                len(direct), len(marker_support), len(function_support), c_q, p_q
            )
            cell_rank = cell_ranks[cell_id]
            phenotype_rank = phenotype_ranks[phenotype.item_id]
            summary_bits = [
                f"{len(function_support)} HPO seed gene(s) support GO function",
                f"{len(marker_support)} support {cell_label} marker(s)",
            ]
            if direct:
                summary_bits.append(f"{len(direct)} gene(s) support both axes")
            if network_support:
                summary_bits.append(f"{len(network_support)} network-proximal gene(s)")
            summary_bits.append("exact accepted cell-GO constraint")
            if marker_function_overlap:
                summary_bits.append(
                    f"{len(marker_function_overlap)} marker-function overlap gene(s)"
                )
            compatible_labels = anatomy_constraints.anatomy_labels_for(compatible_anatomy)
            summary_bits.append(
                f"HPO→UBERON←CL tissue constraint: {', '.join(compatible_labels)}"
            )
            pair_rows.append(
                {
                    "hpo_id": hpo_id,
                    "hpo_name": term.name,
                    "cell_rank": cell_rank,
                    "phenotype_rank": phenotype_rank,
                    "combined_rank_sum": cell_rank + phenotype_rank,
                    "three_way_overlap_score": three_way_overlap_score,
                    "marker_function_overlap_score": marker_function_overlap_score,
                    "rerank_route": (
                        "ppi_network_only"
                        if phenotype.support_mode == "network_only"
                        else "direct_three_way"
                    ),
                    "cell_type_id": cell_id,
                    "cell_type": cell_label or markers.labels.get(cell_id, cell_id),
                    "cell_phenotype_id": phenotype.item_id,
                    "cell_phenotype": phenotype.label,
                    "phenotype_scope": "cell_type_specific",
                    "scope_reason": constraints.decision_reason.get(link_key, ""),
                    "posterior_confidence": f"{posterior:.4f}",
                    "confidence_grade": grade,
                    "evidence_score": 0.0,
                    "supporting_genes": ";".join(sorted(set(function_support) | set(marker_support))),
                    "direct_support_genes": ";".join(direct),
                    "three_way_overlap_count": len(direct),
                    "marker_support_genes": ";".join(marker_support),
                    "function_support_genes": ";".join(function_support),
                    "marker_function_overlap_genes": ";".join(
                        marker_function_overlap
                    ),
                    "marker_function_overlap_count": len(
                        marker_function_overlap
                    ),
                    "cell_marker_gene_count": len(marker_genes),
                    "go_function_gene_count": len(function_genes),
                    "network_support_genes": ";".join(network_support),
                    "cell_q_value": f"{c_q:.6g}",
                    "phenotype_q_value": f"{p_q:.6g}",
                    "cell_direct_q_value": f"{cell.direct_q_value:.6g}",
                    "phenotype_direct_q_value": f"{phenotype.direct_q_value:.6g}",
                    "cell_network_q_value": f"{cell.network_q_value:.6g}",
                    "phenotype_network_q_value": f"{phenotype.network_q_value:.6g}",
                    "cell_support_mode": cell.support_mode,
                    "phenotype_support_mode": phenotype.support_mode,
                    "cell_overlap": len(cell.overlap_genes),
                    "phenotype_overlap": len(phenotype.overlap_genes),
                    "cell_score": f"{cell.score:.6f}",
                    "phenotype_score": f"{phenotype.score:.6f}",
                    "go_cl_relation": relation,
                    "go_constraint_status": status or "missing",
                    "go_constraint_confidence": constraints.link_confidence.get(
                        link_key, ""
                    ),
                    "go_constraint_decision_source": constraints.decision_source.get(
                        link_key, ""
                    ),
                    "anatomy_constraint_status": "allowed",
                    "compatible_anatomy_ids": ";".join(compatible_anatomy),
                    "compatible_anatomy_labels": ";".join(compatible_labels),
                    "evidence_summary": "; ".join(summary_bits),
                }
            )
    pair_rows = _rank_pairs(
        pair_rows,
        len(cell_results),
        len(phenotype_results),
        pair_overlap_weight,
        pair_rank_weight,
        pair_three_way_weight,
    )
    eligible_pair_count = len(pair_rows)
    pair_rows = pair_rows[:top_k]

    general_pair_rows: list[dict[str, object]] = []
    general_anatomy_allowed_cell_count = 0
    general_ranks = {
        row.item_id: rank for rank, row in enumerate(general_results, start=1)
    }
    if general_phenotypes and general_results:
        for cell in cell_results:
            cell_id = cell.item_id
            compatible_anatomy = anatomy_constraints.compatible_anatomy(
                hpo_id, cell_id
            )
            if not compatible_anatomy:
                continue
            general_anatomy_allowed_cell_count += 1
            compatible_labels = anatomy_constraints.anatomy_labels_for(
                compatible_anatomy
            )
            marker_genes = set(markers.genes.get(cell_id, {}))
            marker_support = tuple(sorted(set(seed_weights) & marker_genes))
            for phenotype in general_results:
                function_genes = set(
                    functions.genes.get(phenotype.item_id, set())
                )
                function_support = tuple(
                    sorted(set(seed_weights) & function_genes)
                )
                direct = tuple(
                    sorted(set(seed_weights) & marker_genes & function_genes)
                )
                marker_function_overlap = tuple(
                    sorted(marker_genes & function_genes)
                )
                denominator = min(len(marker_genes), len(function_genes))
                overlap_score = (
                    len(marker_function_overlap) / denominator
                    if denominator
                    else 0.0
                )
                three_way_overlap_score = _three_way_overlap_score(
                    marker_support, function_support, direct
                )
                network_support = tuple(
                    sorted(phenotype.network_support_genes)
                )
                grade, posterior = _grade(
                    len(direct),
                    len(marker_support),
                    len(function_support),
                    cell.q_value,
                    phenotype.q_value,
                )
                cell_rank = cell_ranks[cell_id]
                phenotype_rank = general_ranks[phenotype.item_id]
                summary_bits = [
                    f"{len(function_support)} HPO seed gene(s) support general GO function",
                    f"{len(marker_support)} support {markers.labels.get(cell_id, cell.label)} marker(s)",
                    "cell-general GO is allowed for every tissue-compatible cell",
                ]
                if direct:
                    summary_bits.append(
                        f"{len(direct)} gene(s) support both axes"
                    )
                if marker_function_overlap:
                    summary_bits.append(
                        f"{len(marker_function_overlap)} marker-function overlap gene(s)"
                    )
                if network_support:
                    summary_bits.append(
                        f"{len(network_support)} network-proximal gene(s)"
                    )
                summary_bits.append(
                    f"HPO→UBERON←CL tissue constraint: {', '.join(compatible_labels)}"
                )
                general_pair_rows.append(
                    {
                        "hpo_id": hpo_id,
                        "hpo_name": term.name,
                        "cell_rank": cell_rank,
                        "phenotype_rank": phenotype_rank,
                        "combined_rank_sum": cell_rank + phenotype_rank,
                        "three_way_overlap_score": three_way_overlap_score,
                        "marker_function_overlap_score": overlap_score,
                        "rerank_route": (
                            "ppi_network_only"
                            if phenotype.support_mode == "network_only"
                            else "direct_three_way"
                        ),
                        "cell_type_id": cell_id,
                        "cell_type": markers.labels.get(cell_id, cell.label),
                        "cell_phenotype_id": phenotype.item_id,
                        "cell_phenotype": phenotype.label,
                        "phenotype_scope": "cell_general",
                        "scope_reason": general_phenotypes.reason.get(
                            phenotype.item_id, ""
                        ),
                        "posterior_confidence": f"{posterior:.4f}",
                        "confidence_grade": grade,
                        "evidence_score": 0.0,
                        "supporting_genes": ";".join(
                            sorted(set(function_support) | set(marker_support))
                        ),
                        "direct_support_genes": ";".join(direct),
                        "three_way_overlap_count": len(direct),
                        "marker_support_genes": ";".join(marker_support),
                        "function_support_genes": ";".join(function_support),
                        "marker_function_overlap_genes": ";".join(
                            marker_function_overlap
                        ),
                        "marker_function_overlap_count": len(
                            marker_function_overlap
                        ),
                        "cell_marker_gene_count": len(marker_genes),
                        "go_function_gene_count": len(function_genes),
                        "network_support_genes": ";".join(network_support),
                        "cell_q_value": f"{cell.q_value:.6g}",
                        "phenotype_q_value": f"{phenotype.q_value:.6g}",
                        "cell_direct_q_value": f"{cell.direct_q_value:.6g}",
                        "phenotype_direct_q_value": f"{phenotype.direct_q_value:.6g}",
                        "cell_network_q_value": f"{cell.network_q_value:.6g}",
                        "phenotype_network_q_value": f"{phenotype.network_q_value:.6g}",
                        "cell_support_mode": cell.support_mode,
                        "phenotype_support_mode": phenotype.support_mode,
                        "cell_overlap": len(cell.overlap_genes),
                        "phenotype_overlap": len(phenotype.overlap_genes),
                        "cell_score": f"{cell.score:.6f}",
                        "phenotype_score": f"{phenotype.score:.6f}",
                        "go_cl_relation": "cell_general",
                        "go_constraint_status": "cell_general",
                        "go_constraint_confidence": general_phenotypes.confidence.get(
                            phenotype.item_id, ""
                        ),
                        "go_constraint_decision_source": general_phenotypes.model.get(
                            phenotype.item_id, ""
                        ),
                        "anatomy_constraint_status": "allowed",
                        "compatible_anatomy_ids": ";".join(
                            compatible_anatomy
                        ),
                        "compatible_anatomy_labels": ";".join(
                            compatible_labels
                        ),
                        "evidence_summary": "; ".join(summary_bits),
                    }
                )
    general_pair_rows = _rank_pairs(
        general_pair_rows,
        len(cell_results),
        len(general_results),
        pair_overlap_weight,
        pair_rank_weight,
        pair_three_way_weight,
    )
    general_eligible_pair_count = len(general_pair_rows)
    general_pair_rows = general_pair_rows[:top_general_k]
    seeds_output = [
        {
            "hpo_id": hpo_id,
            "hpo_name": term.name,
            "gene": row.gene,
            "seed_weight": f"{row.weight:.6f}",
            "disease_count": row.disease_count,
            "source_count": row.source_count,
            "supporting_diseases": ";".join(row.supporting_diseases),
        }
        for row in seed_rows
    ]
    cell_output = [
        {
            "hpo_id": hpo_id,
            "hpo_name": term.name,
            "cell_rank": rank,
            "cell_type_id": row.item_id,
            "cell_type": markers.labels.get(row.item_id, row.label),
            "overlap_genes": ";".join(row.overlap_genes),
            "overlap_count": len(row.overlap_genes),
            "p_value": f"{row.p_value:.6g}",
            "q_value": f"{row.q_value:.6g}",
            "direct_p_value": f"{row.direct_p_value:.6g}",
            "direct_q_value": f"{row.direct_q_value:.6g}",
            "network_p_value": f"{row.network_p_value:.6g}",
            "network_q_value": f"{row.network_q_value:.6g}",
            "network_empirical_p_value": f"{row.network_empirical_p_value:.6g}",
            "network_mapped_gene_count": row.network_mapped_gene_count,
            "support_mode": row.support_mode,
            "weighted_support": f"{row.weighted_support:.6g}",
            "network_support": f"{row.network_support:.6g}",
            "network_support_genes": ";".join(row.network_support_genes),
            "score": f"{row.score:.6f}",
        }
        for rank, row in enumerate(cell_results, start=1)
    ]
    phenotype_output = [
        {
            "hpo_id": hpo_id,
            "hpo_name": term.name,
            "phenotype_rank": rank,
            "cell_phenotype_id": row.item_id,
            "cell_phenotype": row.label,
            "overlap_genes": ";".join(row.overlap_genes),
            "overlap_count": len(row.overlap_genes),
            "p_value": f"{row.p_value:.6g}",
            "q_value": f"{row.q_value:.6g}",
            "direct_p_value": f"{row.direct_p_value:.6g}",
            "direct_q_value": f"{row.direct_q_value:.6g}",
            "network_p_value": f"{row.network_p_value:.6g}",
            "network_q_value": f"{row.network_q_value:.6g}",
            "network_empirical_p_value": f"{row.network_empirical_p_value:.6g}",
            "network_mapped_gene_count": row.network_mapped_gene_count,
            "support_mode": row.support_mode,
            "weighted_support": f"{row.weighted_support:.6g}",
            "network_support": f"{row.network_support:.6g}",
            "network_support_genes": ";".join(row.network_support_genes),
            "score": f"{row.score:.6f}",
            "go_constraint_cell_count": len(constraints.by_go.get(row.item_id, [])),
            "go_constraint_status": constraints.status.get(row.item_id, "unconstrained"),
        }
        for rank, row in enumerate(phenotype_results, start=1)
    ]
    audit = {
        "hpo_id": hpo_id,
        "hpo_name": term.name,
        "descendant_term_count": len(hpo_index.descendants(hpo_id)),
        "hpo_seed_scope": "exact_term_only",
        "inherited_hpo_term_count": 0,
        "seed_gene_count": len(seed_rows),
        "cell_candidate_count": len(cell_results),
        "phenotype_candidate_count": len(phenotype_results),
        "candidate_pair_count": candidate_pair_count,
        "go_cl_allowed_pair_count": go_cl_allowed_pair_count,
        "go_cl_filtered_pair_count": candidate_pair_count - go_cl_allowed_pair_count,
        "anatomy_allowed_pair_count": anatomy_allowed_pair_count,
        "anatomy_filtered_pair_count": go_cl_allowed_pair_count - anatomy_allowed_pair_count,
        "eligible_pair_count_before_top_k": eligible_pair_count,
        "triple_count": len(pair_rows),
        "marker_function_overlap_triple_count": sum(
            int(row["marker_function_overlap_count"]) > 0 for row in pair_rows
        ),
        "three_way_overlap_triple_count": sum(
            int(row["three_way_overlap_count"]) > 0 for row in pair_rows
        ),
        "general_phenotype_candidate_count": len(general_results),
        "general_anatomy_allowed_cell_count": general_anatomy_allowed_cell_count,
        "general_candidate_pair_count": (
            general_anatomy_allowed_cell_count * len(general_results)
        ),
        "general_eligible_pair_count_before_top_k": general_eligible_pair_count,
        "general_triple_count": len(general_pair_rows),
        "general_marker_function_overlap_triple_count": sum(
            int(row["marker_function_overlap_count"]) > 0
            for row in general_pair_rows
        ),
        "general_three_way_overlap_triple_count": sum(
            int(row["three_way_overlap_count"]) > 0
            for row in general_pair_rows
        ),
        "pair_rank_weight": pair_rank_weight,
        "pair_three_way_weight": pair_three_way_weight,
        "pair_overlap_weight": pair_overlap_weight,
        "network_enabled": network is not None,
        "network_null_permutations": network.null_permutations if network else 0,
        "network_gene_set_permutations": network.null_permutations if network else 0,
        "network_input_seed_count": network.input_seed_count if network else 0,
        "network_mapped_seed_count": network.mapped_seed_count if network else 0,
        "network_seed_weight_coverage": (
            network.seed_weight_coverage if network else 0.0
        ),
        "network_rwr_iterations": network.iterations if network else 0,
        "direct_cell_candidate_count": sum(
            row.support_mode != "network_only" for row in cell_results
        ),
        "network_only_cell_candidate_count": sum(
            row.support_mode == "network_only" for row in cell_results
        ),
        "direct_phenotype_candidate_count": sum(
            row.support_mode != "network_only" for row in phenotype_results
        ),
        "network_only_phenotype_candidate_count": sum(
            row.support_mode == "network_only" for row in phenotype_results
        ),
        "network_supported_triple_count": sum(
            bool(row["network_support_genes"]) for row in pair_rows
        ),
    }
    return seeds_output, cell_output, phenotype_output, {
        "triples": pair_rows,
        "general_triples": general_pair_rows,
        "audit": audit,
    }


def _write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_hpo_terms(
    hpo_ids: Sequence[str],
    hpo_obo: Path,
    hpoa: Path,
    gene_associations: Path,
    markers_path: Path,
    functions_path: Path,
    constraints_path: Path | None = None,
    classification_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 100,
    top_cells: int = 50,
    top_phenotypes: int = 250,
    min_overlap: int = 1,
    ppi_path: Path | None = None,
    permutations: int = 1000,
    restart: float = 0.65,
    random_seed: int = 13,
    hpo_anatomy_path: Path | None = None,
    cl_anatomy_path: Path | None = None,
    uberon_ancestors_path: Path | None = None,
    network_q_threshold: float = 0.05,
    network_min_support_genes: int = 2,
    network_support_z: float = 1.645,
    pair_overlap_weight: float = 0.15,
    pair_rank_weight: float = 0.55,
    pair_three_way_weight: float = 0.30,
    general_phenotypes_path: Path | None = None,
    top_general_phenotypes: int = 100,
    top_general_k: int = 100,
) -> dict[str, object]:
    hpo_obo = Path(hpo_obo)
    hpoa = Path(hpoa)
    gene_associations = Path(gene_associations)
    markers_path = Path(markers_path)
    functions_path = Path(functions_path)
    constraints_path = Path(constraints_path) if constraints_path else None
    classification_path = Path(classification_path) if classification_path else None
    general_phenotypes_path = (
        Path(general_phenotypes_path) if general_phenotypes_path else None
    )
    hpo_anatomy_path = Path(hpo_anatomy_path or DEFAULT_HPO_ANATOMY_PATH)
    cl_anatomy_path = Path(cl_anatomy_path or DEFAULT_CL_ANATOMY_PATH)
    uberon_ancestors_path = Path(
        uberon_ancestors_path or DEFAULT_UBERON_ANCESTORS_PATH
    )
    output_dir = Path(output_dir) if output_dir else None
    ppi_path = Path(ppi_path) if ppi_path else None
    if not 0.0 <= network_q_threshold <= 1.0:
        raise ValueError("network_q_threshold must be in [0, 1]")
    if network_min_support_genes < 1:
        raise ValueError("network_min_support_genes must be at least 1")
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    pair_weights = (
        pair_rank_weight,
        pair_three_way_weight,
        pair_overlap_weight,
    )
    if any(weight < 0.0 for weight in pair_weights):
        raise ValueError("pair reranking weights must be non-negative")
    if not math.isclose(sum(pair_weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("pair reranking weights must sum to 1")
    if top_general_phenotypes < 1 or top_general_k < 1:
        raise ValueError("top_general_phenotypes and top_general_k must be positive")
    output_dir = output_dir or LINK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    hpo_index = HPOEvidenceIndex(hpo_obo, hpoa, gene_associations)
    markers = CellMarkerIndex(markers_path)
    functions = FunctionIndex(functions_path)
    if not constraints_path or not constraints_path.exists():
        raise FileNotFoundError(
            "Strict GO-to-CL filtering requires the final direct cell-GO link table"
        )
    constraints = ConstraintIndex(constraints_path, classification_path)
    if not constraints.rows:
        raise ValueError(
            "The final direct cell-GO link table contains no accepted exact pairs"
        )
    general_phenotypes = (
        GeneralPhenotypeIndex(general_phenotypes_path)
        if general_phenotypes_path
        else None
    )
    anatomy_constraints = AnatomyConstraintIndex(
        hpo_anatomy_path, cl_anatomy_path, uberon_ancestors_path
    )
    ppi_network = PPINetwork.from_path(ppi_path) if ppi_path else None
    phenotype_network_index = (
        NetworkGeneSetIndex.build(
            ppi_network,
            functions.genes,
            allowed_ids=set(constraints.by_go),
        )
        if ppi_network
        else None
    )
    general_network_index = (
        NetworkGeneSetIndex.build(
            ppi_network,
            functions.genes,
            allowed_ids=general_phenotypes.ids,
        )
        if ppi_network and general_phenotypes
        else None
    )
    all_triples: list[dict[str, object]] = []
    all_general_triples: list[dict[str, object]] = []
    all_seeds: list[dict[str, object]] = []
    all_cells: list[dict[str, object]] = []
    all_phenotypes: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for hpo_id in hpo_ids:
        normalized = hpo_id.strip().upper()
        seed_evidence = hpo_index.seeds(normalized)
        network = None
        phenotype_network_evidence: dict[str, NetworkSetEvidence] = {}
        general_network_evidence: dict[str, NetworkSetEvidence] = {}
        if ppi_network:
            network = ppi_network.propagate(
                {row.gene: row.weight for row in seed_evidence},
                permutations=permutations,
                restart=restart,
            )
            phenotype_network_evidence = phenotype_network_index.score(
                ppi_network,
                network,
                permutations=permutations,
                random_seed=_stable_random_seed(random_seed, normalized, "GO"),
                support_z=network_support_z,
            )
            if general_network_index:
                general_network_evidence = general_network_index.score(
                    ppi_network,
                    network,
                    permutations=permutations,
                    random_seed=_stable_random_seed(
                        random_seed, normalized, "GO_GENERAL"
                    ),
                    support_z=network_support_z,
                )
        seeds, cells, phenotypes, output = infer_one(
            hpo_index=hpo_index,
            hpo_id=normalized,
            markers=markers,
            functions=functions,
            constraints=constraints,
            anatomy_constraints=anatomy_constraints,
            top_k=top_k,
            top_cells=top_cells,
            top_phenotypes=top_phenotypes,
            min_overlap=min_overlap,
            network=network,
            phenotype_network_evidence=phenotype_network_evidence,
            network_q_threshold=network_q_threshold,
            network_min_support_genes=network_min_support_genes,
            pair_overlap_weight=pair_overlap_weight,
            pair_rank_weight=pair_rank_weight,
            pair_three_way_weight=pair_three_way_weight,
            general_phenotypes=general_phenotypes,
            general_network_evidence=general_network_evidence,
            top_general_phenotypes=top_general_phenotypes,
            top_general_k=top_general_k,
        )
        all_seeds.extend(seeds)
        all_cells.extend(cells)
        all_phenotypes.extend(phenotypes)
        all_triples.extend(output["triples"])
        all_general_triples.extend(output["general_triples"])
        audits.append(output["audit"])
    _write_tsv(output_dir / "hpo_seed_genes.tsv", all_seeds, SEED_FIELDS)
    _write_tsv(output_dir / "hpo_cell_type_candidates.tsv", all_cells, CELL_FIELDS)
    _write_tsv(output_dir / "hpo_cell_phenotype_candidates.tsv", all_phenotypes, PHENOTYPE_FIELDS)
    _write_tsv(output_dir / "hpo_cell_mechanism_triples.tsv", all_triples, TRIPLE_FIELDS)
    _write_tsv(
        output_dir / "hpo_cell_general_mechanism_triples.tsv",
        all_general_triples,
        TRIPLE_FIELDS,
    )
    digest = hashlib.sha256("\n".join(f"{row['hpo_id']}\t{row['cell_type_id']}\t{row['cell_phenotype_id']}" for row in all_triples).encode()).hexdigest()
    general_digest = hashlib.sha256(
        "\n".join(
            f"{row['hpo_id']}\t{row['cell_type_id']}\t{row['cell_phenotype_id']}"
            for row in all_general_triples
        ).encode()
    ).hexdigest()
    manifest = {
        "design": {
            "definition": "Exact HPO gene evidence is enriched independently against CL marker and direct GO function genes; accepted exact cell-GO links and HPO-to-CL tissue compatibility are strict post-enrichment filters; valid pairs are ranked by the two enrichment ranks and reranked by HPO-marker-function consistency plus marker-function overlap.",
            "hpo_aspect": "P",
            "hpo_evidence": sorted(EVIDENCE_WEIGHTS),
            "frequency_weights": FREQUENCY_WEIGHTS,
            "include_descendants": False,
            "term_granularity": "exact queried HPO term only; no HPO parent/child gene inheritance",
            "network_diffusion": bool(ppi_path),
            "network_method": "one weighted random walk with restart per HPO; positive non-seed degree-bin z scores are aggregated only for GO function gene sets",
            "network_null": "GSEA-style standardized gene-set empirical null from within-degree-bin node-score permutations, pooled by mapped-set-size stratum when at least 20 sets are available",
            "network_gene_set_permutations": permutations if ppi_path else 0,
            "network_q_threshold": network_q_threshold,
            "network_min_nonseed_support_genes": network_min_support_genes,
            "network_support_z_threshold": network_support_z,
            "network_go_test_scope": "GO sets with at least one accepted exact cell-GO link",
            "network_cell_marker_scope": "disabled; CL candidates require direct HPO seed-marker overlap",
            "multiple_testing": "direct hypergeometric and network empirical P values receive separate Benjamini-Hochberg correction within each HPO query and evidence axis",
            "direct_network_combination": "kept separate; no minimum-P or weighted combination",
            "gene_identifier_handling": "case-normalized exact symbols only; alias and historical-symbol normalization intentionally deferred",
            "go_cl_constraint_mode": "exact_direct_pair_strict_filter",
            "cl_parent_child_inheritance": False,
            "anatomy_constraint_mode": "strict_filter",
            "pair_ranking": "ascending cell_rank + phenotype_rank, normalized to enrichment_rank_score",
            "pair_reranking": "weighted enrichment rank, HPO-marker-function three-way consistency, and marker-function overlap coefficient",
            "pair_rank_weight": pair_rank_weight,
            "pair_three_way_weight": pair_three_way_weight,
            "pair_overlap_weight": pair_overlap_weight,
            "cell_general_channel": bool(general_phenotypes),
            "cell_general_pairing": "every enriched cell with compatible HPO-to-CL tissue may pair with an independently enriched accepted cell-general GO",
            "cell_general_ranking_independence": "separate GO enrichment, PPI FDR, candidate truncation, pair ranking, and output; specific ranks are unchanged",
            "generic_anatomy_ids_excluded": sorted(GENERIC_ANATOMY_IDS),
        },
        "inputs": {
            "hpo_obo": str(hpo_obo),
            "hpoa": str(hpoa),
            "gene_associations": str(gene_associations),
            "markers": str(markers_path),
            "functions": str(functions_path),
            "constraints": str(constraints_path) if constraints_path else "",
            "cell_go_links": str(constraints_path) if constraints_path else "",
            "classification": str(classification_path) if classification_path else "",
            "general_phenotypes": (
                str(general_phenotypes_path) if general_phenotypes_path else ""
            ),
            "hpo_anatomy": str(hpo_anatomy_path),
            "cl_anatomy": str(cl_anatomy_path),
            "uberon_ancestors": str(uberon_ancestors_path),
            "ppi": str(ppi_path) if ppi_path else "",
        },
        "hpo_index": dict(hpo_index.counts),
        "ppi_network": ppi_network.manifest() if ppi_network else None,
        "network_gene_sets": {
            "cell_sets_tested": 0,
            "go_sets_tested": (
                len(phenotype_network_index.item_ids)
                if phenotype_network_index
                else 0
            ),
            "general_go_sets_tested": (
                len(general_network_index.item_ids)
                if general_network_index
                else 0
            ),
        },
        "cell_go_constraints": {
            "exact_link_count": constraints.rows,
            "go_count": len(constraints.by_go),
            "inheritance_enabled": False,
        },
        "cell_general_constraints": {
            "enabled": bool(general_phenotypes),
            "accepted_go_count": (
                len(general_phenotypes.ids) if general_phenotypes else 0
            ),
            "cell_semantic_constraint_required": False,
            "anatomy_constraint_required": True,
            "ranking_isolated_from_specific_channel": True,
        },
        "queries": audits,
        "output": {
            "triple_count": len(all_triples),
            "triple_digest": digest,
            "general_triple_count": len(all_general_triples),
            "general_triple_digest": general_digest,
            "seed_gene_rows": len(all_seeds),
            "cell_candidate_rows": len(all_cells),
            "phenotype_candidate_rows": len(all_phenotypes),
        },
    }
    (output_dir / "hpo_cell_mechanism_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer cell types and cellular phenotypes from HPO terms and gene evidence.")
    parser.add_argument("--hpo-id", action="append", help="HPO ID; repeat this option for multiple terms.")
    parser.add_argument("--hpo-file", type=Path, help="File containing HPO IDs, one per line or a TSV first column.")
    parser.add_argument("--hpo-obo", type=Path, default=HPO_OBO)
    parser.add_argument("--hpoa", type=Path, default=HPOA)
    parser.add_argument("--gene-associations", type=Path, default=GENES_TO_PHENOTYPE)
    parser.add_argument("--markers", type=Path, default=CELL_MARKERS)
    parser.add_argument("--functions", type=Path, default=GO_FUNCTIONS)
    parser.add_argument(
        "--cell-go-links",
        "--constraints",
        dest="constraints",
        type=Path,
        default=CELL_GO_LINKS,
        help="Final accepted exact cell-GO links; --constraints is a legacy alias.",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        help="Legacy compatibility option; ignored by exact-link mode.",
    )
    parser.add_argument(
        "--general-phenotypes",
        type=Path,
        default=DEFAULT_GENERAL_PHENOTYPES_PATH,
        help="Accepted cell-general GO terms for the independent general channel.",
    )
    parser.add_argument("--hpo-anatomy", type=Path, default=DEFAULT_HPO_ANATOMY_PATH, help="Materialized HPO-to-UBERON closure TSV[.gz].")
    parser.add_argument("--cl-anatomy", type=Path, default=DEFAULT_CL_ANATOMY_PATH, help="Materialized CL-to-UBERON closure TSV[.gz].")
    parser.add_argument("--uberon-ancestors", type=Path, default=DEFAULT_UBERON_ANCESTORS_PATH, help="Materialized UBERON subclass closure TSV[.gz].")
    parser.add_argument("--ppi", type=Path, help="Optional tab-separated PPI edges: gene_a, gene_b, [confidence].")
    parser.add_argument("--permutations", type=int, default=1000, help="Within-degree-bin gene-set null permutations (default: 1000).")
    parser.add_argument("--restart", type=float, default=0.65, help="RWR restart probability (default: 0.65).")
    parser.add_argument("--network-q-threshold", type=float, default=0.05, help="Maximum network BH q value for network-supported candidates (default: 0.05).")
    parser.add_argument("--network-min-support-genes", type=int, default=2, help="Minimum non-seed network-proximal genes required (default: 2).")
    parser.add_argument("--network-support-z", type=float, default=1.645, help="Minimum degree-bin z score for a network support gene (default: 1.645).")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-cells", type=int, default=50)
    parser.add_argument("--top-phenotypes", type=int, default=250)
    parser.add_argument("--min-overlap", type=int, default=1)
    parser.add_argument(
        "--pair-overlap-weight",
        type=float,
        default=0.15,
        help="Final reranking weight for the marker-function overlap coefficient (default: 0.15).",
    )
    parser.add_argument(
        "--pair-rank-weight",
        type=float,
        default=0.55,
        help="Final reranking weight for the independent enrichment rank (default: 0.55).",
    )
    parser.add_argument(
        "--pair-three-way-weight",
        type=float,
        default=0.30,
        help="Final reranking weight for HPO-marker-function consistency (default: 0.30).",
    )
    parser.add_argument("--top-general-phenotypes", type=int, default=100)
    parser.add_argument("--top-general-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=LINK_DIR)
    parser.add_argument("--random-seed", type=int, default=13)
    return parser


def _read_hpo_ids(args: argparse.Namespace) -> list[str]:
    values = list(args.hpo_id or [])
    if args.hpo_file:
        for line in args.hpo_file.read_text(encoding="utf-8").splitlines():
            match = HPO_ID_RE.search(line.upper())
            if match:
                values.append(match.group(0))
    result: list[str] = []
    for value in values:
        for candidate in re.split(r"[,\s]+", value.upper()):
            if HPO_ID_RE.fullmatch(candidate) and candidate not in result:
                result.append(candidate)
    if not result:
        raise SystemExit("Provide at least one --hpo-id or --hpo-file.")
    return result


if __name__ == "__main__":
    args = build_parser().parse_args()
    manifest = infer_hpo_terms(
        _read_hpo_ids(args),
        hpo_obo=args.hpo_obo,
        hpoa=args.hpoa,
        gene_associations=args.gene_associations,
        markers_path=args.markers,
        functions_path=args.functions,
        constraints_path=args.constraints if args.constraints.exists() else None,
        classification_path=(
            args.classification
            if args.classification and args.classification.exists()
            else None
        ),
        output_dir=args.output_dir,
        top_k=args.top_k,
        top_cells=args.top_cells,
        top_phenotypes=args.top_phenotypes,
        min_overlap=args.min_overlap,
        ppi_path=args.ppi,
        permutations=args.permutations,
        restart=args.restart,
        random_seed=args.random_seed,
        hpo_anatomy_path=args.hpo_anatomy,
        cl_anatomy_path=args.cl_anatomy,
        uberon_ancestors_path=args.uberon_ancestors,
        network_q_threshold=args.network_q_threshold,
        network_min_support_genes=args.network_min_support_genes,
        network_support_z=args.network_support_z,
        pair_overlap_weight=args.pair_overlap_weight,
        pair_rank_weight=args.pair_rank_weight,
        pair_three_way_weight=args.pair_three_way_weight,
        general_phenotypes_path=args.general_phenotypes,
        top_general_phenotypes=args.top_general_phenotypes,
        top_general_k=args.top_general_k,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
