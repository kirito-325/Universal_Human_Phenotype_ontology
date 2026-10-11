from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from celltype_configure.common import CellCatalog, load_obo, write_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
ACCEPTED_CONFIDENCES = {"high", "medium"}
SCOPES = {
    "cell_type_specific",
    "cell_general",
    "not_direct_human_cell_process",
    "unresolved",
}
SCOPE_FIELDS = [
    "go_id",
    "go_name",
    "cell_scope",
    "confidence",
    "decision_accepted",
    "review_required",
    "decision_source",
    "decision_reason",
    "cell_mapping_status",
    "intended_cl_ids",
    "intended_cell_labels",
    "human_function_gene_count",
    "experimental_evidence_gene_count",
    "direct_annotation_count",
    "specific_cell_count",
    "specific_cl_ids",
    "specific_cell_labels",
    "llm_model",
]
GENERAL_FIELDS = SCOPE_FIELDS
NOT_DIRECT_FIELDS = SCOPE_FIELDS
UNRESOLVED_FIELDS = SCOPE_FIELDS
TARGET_FIELDS = [
    "go_id",
    "go_name",
    "semantic_classification",
    "target_reason",
    "human_function_gene_count",
    "experimental_evidence_gene_count",
]
SPECIFIC_LINK_FIELDS = [
    "go_id",
    "go_name",
    "cl_id",
    "cl_name",
    "catalog_label",
    "branch",
    "final_link_status",
    "confidence",
    "review_required",
    "decision_source",
    "evidence_sources",
    "link_origin",
    "decision_reason",
]

MOLECULE_SUFFIXES = (
    ("growth", "factor"),
    ("colony", "stimulating", "factor"),
    ("stimulating", "hormone"),
    ("activating", "factor"),
    ("migration", "inhibitory", "factor"),
    ("macrophage", "colony", "stimulating", "factor"),
)
RISKY_SUBSTITUTE_PATTERNS = (
    re.compile(r"\bnot in the candidate list\b", re.I),
    re.compile(r"\bno candidate is\b", re.I),
    re.compile(r"\bnone of (?:the )?(?:supplied )?candidates\b", re.I),
    re.compile(r"\bclosest (?:available|broad|liver|match|cell)\b", re.I),
    re.compile(r"\bcandidate list (?:does not|lacks)\b", re.I),
    re.compile(
        r"\bno (?:appropriate|exact|suitable|specific|direct|matching|correct) "
        r"candidate\b",
        re.I,
    ),
    re.compile(
        r"\bnot (?:an? )?(?:ideal|correct|appropriate) "
        r"(?:fit|match|candidate|choice)\b",
        re.I,
    ),
)
NONHUMAN_GO_NAME_PATTERNS = (
    re.compile(r"\bplant(?:-type)?\b", re.I),
    re.compile(r"\b(?:pollen|gametophyte|sporophyte|meristem|xylem|phloem|stomatal?)\b", re.I),
    re.compile(r"\b(?:ascospore|conidium|hyphal?|pseudohyphal|prospore)\b", re.I),
    re.compile(r"\b(?:mating projection|cell plate|phragmoplast|bud site)\b", re.I),
    re.compile(r"\bcell growth mode switching\b", re.I),
    re.compile(r"\b(?:compound eye|ommatid|imaginal disc|wing disc|wing hair)\b", re.I),
    re.compile(r"\b(?:hemocyte|plasmatocyte|lamellocyte|Malpighian tubule)\b", re.I),
    re.compile(r"\b(?:nurse cell|chorion-containing eggshell|eggshell chorion)\b", re.I),
    re.compile(r"\b(?:lateral line|fin development|fin morphogenesis)\b", re.I),
    re.compile(r"\bR[1-8](?:/R[1-8])? cell\b", re.I),
)
NONHUMAN_REASON_PATTERNS = (
    re.compile(
        r"\b(?:the|this) (?:GO )?(?:term|process).{0,120}"
        r"(?:not applicable|not relevant|not for human|does not occur in human|"
        r"species-specific|specific to non-mammalian)",
        re.I,
    ),
    re.compile(r"\bthe GO is species-specific\b", re.I),
    re.compile(r"\bscope should be not_direct_human_cell_process\b", re.I),
)


def normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.lower()))


def split_ids(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def risky_substitute_reason(value: str) -> bool:
    return any(pattern.search(value) for pattern in RISKY_SUBSTITUTE_PATTERNS)


def nonhuman_go_name(value: str) -> bool:
    return any(pattern.search(value) for pattern in NONHUMAN_GO_NAME_PATTERNS)


def nonhuman_decision_reason(value: str) -> bool:
    return any(pattern.search(value) for pattern in NONHUMAN_REASON_PATTERNS)


def ontology_descendants(graph, root_id: str) -> set[str]:
    result = {root_id} if root_id in graph else set()
    stack = [root_id]
    while stack:
        parent = stack.pop()
        for child in graph.predecessors(parent):
            child = str(child)
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


class ExplicitCellNameIndex:
    """Exact full-CL names mentioned in a GO name, excluding molecule-name uses."""

    def __init__(self, cl_obo: Path) -> None:
        graph = load_obo(cl_obo)
        self.labels: dict[str, str] = {}
        phrases: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for cl_id, data in graph.nodes(data=True):
            cl_id = str(cl_id)
            if not cl_id.startswith("CL:") or cl_id == "CL:0000000":
                continue
            name = str(data.get("name", "")).strip()
            phrase = normalized_tokens(name)
            if not phrase or phrase == ("cell",):
                continue
            self.labels[cl_id] = name
            phrases[phrase].append(cl_id)
            synonyms = data.get("synonym", [])
            if isinstance(synonyms, str):
                synonyms = [synonyms]
            for synonym in synonyms:
                match = re.match(r'^"([^"]+)"\s+EXACT\b', str(synonym))
                if not match:
                    continue
                synonym_phrase = normalized_tokens(match.group(1))
                if synonym_phrase and synonym_phrase != ("cell",):
                    phrases[synonym_phrase].append(cl_id)
        self.phrases = dict(phrases)
        self.lengths = tuple(sorted({len(phrase) for phrase in phrases}, reverse=True))

    @staticmethod
    def _is_molecule_use(tokens: tuple[str, ...], end: int) -> bool:
        remainder = tokens[end:]
        return any(remainder[: len(suffix)] == suffix for suffix in MOLECULE_SUFFIXES)

    def match(self, go_name: str) -> tuple[str, ...]:
        tokens = normalized_tokens(go_name)
        matches: list[tuple[int, int, tuple[str, ...]]] = []
        for length in self.lengths:
            if length > len(tokens):
                continue
            for start in range(len(tokens) - length + 1):
                phrase = tokens[start : start + length]
                ids = self.phrases.get(phrase)
                if not ids or self._is_molecule_use(tokens, start + length):
                    continue
                matches.append((start, start + length, tuple(ids)))
        maximal = [
            match
            for match in matches
            if not any(
                match is not other
                and other[0] <= match[0]
                and match[1] <= other[1]
                and (other[1] - other[0]) > (match[1] - match[0])
                for other in matches
            )
        ]
        return tuple(sorted({cl_id for _, _, ids in maximal for cl_id in ids}))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_paths(value: Path | Sequence[Path]) -> list[Path]:
    return [value] if isinstance(value, Path) else list(value)


def read_tsv_many(paths: Path | Sequence[Path]) -> list[dict[str, str]]:
    return [row for path in as_paths(paths) for row in read_tsv(path)]


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def accepted_decision(row: Mapping[str, str]) -> bool:
    return (
        row.get("adjudicated_scope") in SCOPES - {"unresolved"}
        and row.get("llm_confidence") in ACCEPTED_CONFIDENCES
        and as_bool(row.get("decision_accepted"))
        and not as_bool(row.get("review_required"))
        and row.get("llm_status") == "success"
        and not (
            row.get("llm_scope") == "cell_type_specific"
            and risky_substitute_reason(row.get("llm_reason", ""))
        )
    )


def decision_index(
    existing: Sequence[Mapping[str, str]],
    supplemental: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    result = {row["go_id"]: dict(row) for row in existing}
    for row in supplemental:
        result[row["go_id"]] = dict(row)
    return result


def exact_link_index(
    existing: Sequence[Mapping[str, str]],
    supplemental: Sequence[Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
    superseded_go_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in existing:
        if row["go_id"] in superseded_go_ids:
            continue
        by_key[(row["go_id"], row["cl_id"])] = dict(row)
    for row in supplemental:
        decision = decisions.get(row["go_id"], {})
        selected_ids = set(split_ids(decision.get("llm_selected_cl_ids", "")))
        if (
            row["go_id"] not in superseded_go_ids
            or not accepted_decision(decision)
            or decision.get("adjudicated_scope") != "cell_type_specific"
            or row["cl_id"] not in selected_ids
        ):
            continue
        if not as_bool(row.get("decision_accepted")) or as_bool(
            row.get("review_required")
        ):
            continue
        normalized = {
            "go_id": row["go_id"],
            "go_name": row["go_name"],
            "cl_id": row["cl_id"],
            "cl_name": row["cl_name"],
            "catalog_label": row["catalog_label"],
            "branch": row.get("branch", ""),
            "final_link_status": "llm_supported",
            "confidence": row.get("llm_confidence", ""),
            "review_required": "false",
            "decision_source": "scope_completion_llm",
            "evidence_sources": row.get("recall_sources", ""),
            "link_origin": row.get("link_origin", ""),
            "decision_reason": row.get("llm_reason", ""),
        }
        key = (normalized["go_id"], normalized["cl_id"])
        if by_key.get(key, {}).get("final_link_status") != "confirmed":
            by_key[key] = normalized
    links = sorted(by_key.values(), key=lambda row: (row["cl_id"], row["go_id"]))
    by_go: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in links:
        by_go[row["go_id"]].append(row)
    return links, dict(by_go)


def intended_cell_index(
    semantic_rows: Sequence[Mapping[str, str]],
    direct_rule_rows: Sequence[Mapping[str, str]],
    name_index: ExplicitCellNameIndex,
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    intended: dict[str, set[str]] = defaultdict(set)
    curator_go: set[str] = set()
    for row in semantic_rows:
        intended[row["go_id"]].update(name_index.match(row.get("go_name", "")))
    for row in direct_rule_rows:
        if row.get("assignment_method") != "curator_reference":
            continue
        go_id = row.get("go_id", "")
        if not go_id:
            continue
        source_ids = split_ids(row.get("explicit_source_cl_ids", ""))
        if source_ids:
            intended[go_id].update(source_ids)
            curator_go.add(go_id)
    return (
        {go_id: tuple(sorted(values)) for go_id, values in intended.items() if values},
        curator_go,
    )


def enforce_exact_intended_cells(
    links: Sequence[Mapping[str, str]],
    intended_by_go: Mapping[str, Sequence[str]],
    curator_go: set[str],
    semantic_by_go: Mapping[str, Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
    catalog: CellCatalog,
    name_index: ExplicitCellNameIndex,
    nonhuman_go_ids: set[str],
    overrides: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw in links:
        row = dict(raw)
        if row["go_id"] in nonhuman_go_ids:
            continue
        intended = set(intended_by_go.get(row["go_id"], ()))
        if intended and row["cl_id"] not in intended:
            continue
        if row.get("final_link_status") == "llm_supported" and risky_substitute_reason(
            row.get("decision_reason", "")
        ):
            continue
        by_key[(row["go_id"], row["cl_id"])] = row

    for go_id, intended_ids in intended_by_go.items():
        if go_id in nonhuman_go_ids:
            continue
        decision = decisions.get(go_id, {})
        if (
            go_id not in curator_go
            and accepted_decision(decision)
            and decision.get("adjudicated_scope") == "not_direct_human_cell_process"
        ):
            continue
        semantic = semantic_by_go[go_id]
        for cl_id in intended_ids:
            if cl_id not in catalog.nodes:
                continue
            node = catalog.nodes[cl_id]
            curator = go_id in curator_go
            override = overrides.get(go_id)
            override_specific = bool(
                override and override.get("cell_scope") == "cell_type_specific"
            )
            by_key[(go_id, cl_id)] = {
                "go_id": go_id,
                "go_name": semantic.get("go_name", ""),
                "cl_id": cl_id,
                "cl_name": node.cl_name,
                "catalog_label": node.catalog_label,
                "branch": node.branch,
                "final_link_status": "confirmed" if curator else "supported",
                "confidence": "high",
                "review_required": "false",
                "decision_source": (
                    "scope_override_exact_cl"
                    if override_specific
                    else "go_curator_exact_cl"
                    if curator
                    else "go_name_exact_full_cl"
                ),
                "evidence_sources": (
                    "scope_boundary_review"
                    if override_specific
                    else "go_curator_cl_reference"
                    if curator
                    else "go_name_cell_mention"
                ),
                "link_origin": "canonical_exact_cell_validation",
                "decision_reason": (
                    override.get("decision_reason", "Curated exact-cell decision")
                    if override_specific
                    else "GO curator references this exact reviewed CL"
                    if curator
                    else "GO name explicitly contains this exact full-CL cell name"
                ),
            }
    result = sorted(by_key.values(), key=lambda row: (row["cl_id"], row["go_id"]))
    by_go: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in result:
        by_go[row["go_id"]].append(row)
    return result, dict(by_go)


def identify_targets(
    semantic_rows: Sequence[Mapping[str, str]],
    function_by_go: Mapping[str, Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
    exact_by_go: Mapping[str, Sequence[Mapping[str, str]]],
    overrides: Mapping[str, Mapping[str, str]],
    intended_by_go: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in semantic_rows:
        go_id = row["go_id"]
        if go_id in overrides or intended_by_go.get(go_id):
            continue
        decision = decisions.get(go_id)
        if exact_by_go.get(go_id) or (decision and accepted_decision(decision)):
            continue
        semantic_class = row.get("semantic_classification", "")
        if semantic_class == "cell_general_candidate":
            continue
        if decision and decision.get("adjudicated_scope") == "unresolved":
            reason = "existing_decision_unresolved"
        elif semantic_class in {
            "cell_type_specific_candidate",
            "cell_type_specific",
        }:
            reason = "specific_candidate_without_accepted_cell"
        else:
            reason = "scope_not_resolved"
        function = function_by_go.get(go_id, {})
        rows.append(
            {
                "go_id": go_id,
                "go_name": row.get("go_name", ""),
                "semantic_classification": semantic_class,
                "target_reason": reason,
                "human_function_gene_count": function.get(
                    "human_function_gene_count", "0"
                ),
                "experimental_evidence_gene_count": function.get(
                    "experimental_evidence_gene_count", "0"
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["go_id"]))


def _decision_scope_row(
    semantic: Mapping[str, str],
    function: Mapping[str, str],
    decision: Mapping[str, str],
    cells: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    accepted = accepted_decision(decision)
    scope = decision.get("adjudicated_scope", "unresolved") if accepted else "unresolved"
    return {
        "go_id": semantic["go_id"],
        "go_name": semantic.get("go_name", ""),
        "cell_scope": scope,
        "confidence": decision.get("llm_confidence", ""),
        "decision_accepted": str(accepted).lower(),
        "review_required": str(not accepted).lower(),
        "decision_source": "direct_scope_llm",
        "decision_reason": decision.get("llm_reason", ""),
        "cell_mapping_status": (
            "mapped"
            if scope == "cell_type_specific" and cells
            else "unmapped_catalog_cell"
            if scope == "cell_type_specific"
            else "not_applicable"
        ),
        "intended_cl_ids": ";".join(row["cl_id"] for row in cells)
        if scope == "cell_type_specific"
        else "",
        "intended_cell_labels": ";".join(row["catalog_label"] for row in cells)
        if scope == "cell_type_specific"
        else "",
        "human_function_gene_count": function.get("human_function_gene_count", "0"),
        "experimental_evidence_gene_count": function.get(
            "experimental_evidence_gene_count", "0"
        ),
        "direct_annotation_count": function.get("direct_annotation_count", "0"),
        "specific_cell_count": len(cells) if scope == "cell_type_specific" else 0,
        "specific_cl_ids": ";".join(row["cl_id"] for row in cells)
        if scope == "cell_type_specific"
        else "",
        "specific_cell_labels": ";".join(row["catalog_label"] for row in cells)
        if scope == "cell_type_specific"
        else "",
        "llm_model": decision.get("llm_model", ""),
    }


def build_scope_rows(
    semantic_rows: Sequence[Mapping[str, str]],
    function_by_go: Mapping[str, Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
    exact_by_go: Mapping[str, Sequence[Mapping[str, str]]],
    overrides: Mapping[str, Mapping[str, str]],
    intended_by_go: Mapping[str, Sequence[str]],
    intended_labels: Mapping[str, str],
    curator_go: set[str],
    cellular_go_ids: set[str],
    nonhuman_go_ids: set[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for semantic in semantic_rows:
        go_id = semantic["go_id"]
        function = function_by_go.get(go_id, {})
        cells = exact_by_go.get(go_id, [])
        confirmed = [row for row in cells if row.get("final_link_status") == "confirmed"]
        decision = decisions.get(go_id)
        override = overrides.get(go_id)
        intended_ids = tuple(
            sorted(
                set(intended_by_go.get(go_id, ()))
                | set(split_ids(override.get("intended_cl_ids", "")) if override else ())
            )
        )
        intended_cell_labels = tuple(
            intended_labels.get(cl_id, cl_id) for cl_id in intended_ids
        )
        if go_id in nonhuman_go_ids:
            source = "nonhuman_scope_guard"
            confidence = "high"
            reason = (
                "The GO name or adjudication evidence explicitly identifies a "
                "plant, fungal, arthropod, or other non-human biological process"
            )
            accepted = True
            scope = "not_direct_human_cell_process"
            cells = []
            intended_ids = ()
            intended_cell_labels = ()
        elif override:
            source = override.get("decision_source", "scope_quality_override")
            confidence = override.get("confidence", "high")
            reason = override.get("decision_reason", "Curated scope quality decision")
            accepted = True
            scope = override["cell_scope"]
            if scope != "cell_type_specific":
                cells = []
                intended_ids = ()
                intended_cell_labels = ()
        elif (
            decision
            and accepted_decision(decision)
            and decision.get("adjudicated_scope") == "not_direct_human_cell_process"
            and go_id not in curator_go
        ):
            result.append(_decision_scope_row(semantic, function, decision, []))
            continue
        elif intended_ids:
            source = (
                "go_curator_exact_cell_validation"
                if go_id in curator_go
                else "go_name_exact_full_cl_validation"
            )
            confidence = "high"
            reason = (
                "Exact full-CL cell identity validated from GO curator references"
                if go_id in curator_go
                else "Exact full-CL cell identity validated from the GO name"
            )
            accepted = True
            scope = "cell_type_specific"
        elif confirmed:
            source = "go_curator_cl_reference"
            confidence = "high"
            reason = "Confirmed exact GO-CL relation"
            accepted = True
            scope = "cell_type_specific"
        elif decision and accepted_decision(decision):
            result.append(_decision_scope_row(semantic, function, decision, cells))
            continue
        elif cells:
            source = "accepted_direct_go_cl"
            confidence = max(
                (row.get("confidence", "") for row in cells),
                key=lambda value: {"high": 2, "medium": 1}.get(value, 0),
            )
            reason = "Accepted exact GO-CL relation"
            accepted = True
            scope = "cell_type_specific"
        elif semantic.get("semantic_classification") == "cell_general_candidate":
            cellular = go_id in cellular_go_ids
            source = (
                "semantic_non_specific_cellular_default"
                if cellular
                else "go_ontology_noncellular_guard"
            )
            confidence = "medium" if cellular else "high"
            reason = (
                "No intrinsic exact cell class was identified within the GO cellular-process branch; route through the cell-general channel and localize downstream with marker and anatomy"
                if cellular
                else "The GO term is outside the cellular-process branch and does not have accepted evidence for a direct exact-cell phenotype"
            )
            accepted = True
            scope = "cell_general" if cellular else "not_direct_human_cell_process"
        else:
            source = "unresolved_after_scope_completion"
            confidence = decision.get("llm_confidence", "") if decision else ""
            reason = (
                decision.get("llm_reason", "")
                if decision
                else "No accepted general or exact-cell decision"
            )
            accepted = False
            scope = "unresolved"
        result.append(
            {
                "go_id": go_id,
                "go_name": semantic.get("go_name", ""),
                "cell_scope": scope,
                "confidence": confidence,
                "decision_accepted": str(accepted).lower(),
                "review_required": str(not accepted).lower(),
                "decision_source": source,
                "decision_reason": reason,
                "cell_mapping_status": (
                    "mapped"
                    if scope == "cell_type_specific" and cells
                    and (not intended_ids or len(cells) == len(intended_ids))
                    else "partially_mapped_catalog_cells"
                    if scope == "cell_type_specific" and cells
                    else "unmapped_catalog_cell"
                    if scope == "cell_type_specific"
                    else "unresolved"
                    if scope == "unresolved"
                    else "not_applicable"
                ),
                "intended_cl_ids": ";".join(intended_ids)
                if scope == "cell_type_specific"
                else "",
                "intended_cell_labels": ";".join(intended_cell_labels)
                if scope == "cell_type_specific"
                else "",
                "human_function_gene_count": function.get(
                    "human_function_gene_count", "0"
                ),
                "experimental_evidence_gene_count": function.get(
                    "experimental_evidence_gene_count", "0"
                ),
                "direct_annotation_count": function.get("direct_annotation_count", "0"),
                "specific_cell_count": len(cells) if scope == "cell_type_specific" else 0,
                "specific_cl_ids": ";".join(row["cl_id"] for row in cells)
                if scope == "cell_type_specific"
                else "",
                "specific_cell_labels": ";".join(
                    row["catalog_label"] for row in cells
                )
                if scope == "cell_type_specific"
                else "",
                "llm_model": decision.get("llm_model", "") if decision else "",
            }
        )
    return result


def digest_rows(rows: Iterable[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{row['go_id']}\t{row['cell_scope']}\t{row['decision_source']}"
            for row in rows
        ).encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    semantic_rows = read_tsv(args.semantic_summary)
    function_rows = read_tsv(args.function_summary)
    existing_decisions = read_tsv(args.existing_decisions)
    supplemental_decisions = read_tsv_many(args.supplemental_decisions)
    existing_links = read_tsv(args.existing_specific_links)
    supplemental_links = read_tsv_many(args.supplemental_specific_links)
    override_rows = read_tsv(args.scope_overrides)
    direct_rule_rows = read_tsv(args.direct_rule_links)
    if not semantic_rows:
        raise ValueError("GO semantic universe is empty")
    if len({row["go_id"] for row in semantic_rows}) != len(semantic_rows):
        raise ValueError("GO semantic universe contains duplicate IDs")
    function_by_go = {row["go_id"]: row for row in function_rows}
    semantic_by_go = {row["go_id"]: row for row in semantic_rows}
    semantic_ids = set(semantic_by_go)
    if len(function_by_go) != len(function_rows):
        raise ValueError("GO function summary contains duplicate IDs")
    function_ids = set(function_by_go)
    if function_ids != semantic_ids:
        raise ValueError(
            "GO function and semantic universes differ: "
            f"missing_function={len(semantic_ids - function_ids)}, "
            f"extra_function={len(function_ids - semantic_ids)}"
        )
    catalog = CellCatalog.from_tsv(args.catalog)
    name_index = ExplicitCellNameIndex(args.cl_obo)
    go_graph = load_obo(args.go_obo)
    active_bp_ids = {
        str(go_id)
        for go_id, data in go_graph.nodes(data=True)
        if data.get("namespace") == "biological_process"
        and str(data.get("is_obsolete", "false")).lower() != "true"
    }
    if active_bp_ids != semantic_ids:
        raise ValueError(
            "Semantic universe is not the complete active GO-BP universe: "
            f"missing_semantic={len(active_bp_ids - semantic_ids)}, "
            f"extra_semantic={len(semantic_ids - active_bp_ids)}"
        )
    cellular_go_ids = ontology_descendants(go_graph, "GO:0009987")
    nonhuman_go_ids = {
        row["go_id"]
        for row in semantic_rows
        if nonhuman_go_name(row.get("go_name", ""))
    }
    intended_by_go, curator_go = intended_cell_index(
        semantic_rows, direct_rule_rows, name_index
    )
    intended_sets = {
        go_id: set(values) for go_id, values in intended_by_go.items()
    }
    for row in override_rows:
        if row.get("cell_scope") == "cell_type_specific":
            intended_sets[row["go_id"]] = set(
                split_ids(row.get("intended_cl_ids", ""))
            )
    intended_by_go = {
        go_id: tuple(sorted(values))
        for go_id, values in intended_sets.items()
        if values
    }
    overrides = {row["go_id"]: row for row in override_rows}
    unknown_overrides = sorted(set(overrides) - semantic_ids)
    if unknown_overrides:
        raise ValueError(f"Scope overrides contain unknown GO IDs: {unknown_overrides}")
    invalid_override_scopes = sorted(
        {row.get("cell_scope", "") for row in override_rows} - (SCOPES - {"unresolved"})
    )
    if invalid_override_scopes:
        raise ValueError(f"Invalid override scopes: {invalid_override_scopes}")
    decisions = decision_index(existing_decisions, supplemental_decisions)
    superseded_go_ids = {row["go_id"] for row in supplemental_decisions}
    nonhuman_go_ids.update(
        go_id
        for go_id, row in decisions.items()
        if nonhuman_decision_reason(row.get("llm_reason", ""))
    )
    specific_links, exact_by_go = exact_link_index(
        existing_links,
        supplemental_links,
        decisions,
        superseded_go_ids,
    )
    specific_links, exact_by_go = enforce_exact_intended_cells(
        specific_links,
        intended_by_go,
        curator_go,
        semantic_by_go,
        decisions,
        catalog,
        name_index,
        nonhuman_go_ids,
        overrides,
    )
    targets = identify_targets(
        semantic_rows,
        function_by_go,
        decisions,
        exact_by_go,
        overrides,
        intended_by_go,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "ambiguous_go_targets.tsv", TARGET_FIELDS, targets)
    (output_dir / "ambiguous_go_ids.txt").write_text(
        "".join(f"{row['go_id']}\n" for row in targets), encoding="utf-8"
    )
    if args.targets_only:
        manifest = {
            "mode": "targets_only",
            "go_process_terms": len(semantic_rows),
            "ambiguous_go_terms": len(targets),
        }
        (output_dir / "scope_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    scope_rows = build_scope_rows(
        semantic_rows,
        function_by_go,
        decisions,
        exact_by_go,
        overrides,
        intended_by_go,
        name_index.labels,
        curator_go,
        cellular_go_ids,
        nonhuman_go_ids,
    )
    if len(scope_rows) != len(semantic_rows):
        raise AssertionError("Scope output does not cover the full GO universe")
    unresolved = [row for row in scope_rows if row["cell_scope"] == "unresolved"]
    if args.require_complete and unresolved:
        raise ValueError(
            f"{len(unresolved)} GO terms remain unresolved; rerun their LLM targets"
        )
    accepted_specific_go = {
        str(row["go_id"])
        for row in scope_rows
        if row["cell_scope"] == "cell_type_specific"
        and row["decision_accepted"] == "true"
    }
    specific_links = [
        row for row in specific_links if row["go_id"] in accepted_specific_go
    ]
    general = [row for row in scope_rows if row["cell_scope"] == "cell_general"]
    specific_unmapped = [
        row
        for row in scope_rows
        if row["cell_scope"] == "cell_type_specific"
        and row["cell_mapping_status"] == "unmapped_catalog_cell"
    ]
    specific_partial = [
        row
        for row in scope_rows
        if row["cell_scope"] == "cell_type_specific"
        and row["cell_mapping_status"] == "partially_mapped_catalog_cells"
    ]
    not_direct = [
        row
        for row in scope_rows
        if row["cell_scope"] == "not_direct_human_cell_process"
    ]
    scope_by_go = {str(row["go_id"]): row for row in scope_rows}
    link_keys = [(row["go_id"], row["cl_id"]) for row in specific_links]
    if len(link_keys) != len(set(link_keys)):
        raise AssertionError("Canonical specific links contain duplicate GO-CL pairs")
    if any(row["cl_id"] not in catalog.nodes for row in specific_links):
        raise AssertionError("Canonical specific links contain cells outside the catalog")
    if any(
        scope_by_go[row["go_id"]]["cell_scope"] != "cell_type_specific"
        for row in specific_links
    ):
        raise AssertionError("A specific link belongs to a non-specific GO scope")
    links_by_go: dict[str, set[str]] = defaultdict(set)
    for row in specific_links:
        links_by_go[row["go_id"]].add(row["cl_id"])
    for row in scope_rows:
        go_id = str(row["go_id"])
        linked = links_by_go.get(go_id, set())
        intended = {
            item for item in str(row["intended_cl_ids"]).split(";") if item
        }
        if intended and not linked.issubset(intended):
            raise AssertionError(f"{go_id} links outside its intended exact CL set")
        if int(row["specific_cell_count"]) != len(linked):
            raise AssertionError(f"{go_id} specific-cell count disagrees with links")
        if row["cell_scope"] != "cell_type_specific" and linked:
            raise AssertionError(f"{go_id} has links outside cell-specific scope")
        status = str(row["cell_mapping_status"])
        if status == "mapped" and (not linked or (intended and linked != intended)):
            raise AssertionError(f"{go_id} has inconsistent mapped status")
        if status == "partially_mapped_catalog_cells" and not (
            linked and intended and linked < intended
        ):
            raise AssertionError(f"{go_id} has inconsistent partial mapping status")
        if status == "unmapped_catalog_cell" and linked:
            raise AssertionError(f"{go_id} has links despite unmapped status")
    partition_ids = [
        {str(row["go_id"]) for row in general},
        accepted_specific_go,
        {str(row["go_id"]) for row in not_direct},
        {str(row["go_id"]) for row in unresolved},
    ]
    if any(
        left & right
        for i, left in enumerate(partition_ids)
        for right in partition_ids[i + 1 :]
    ):
        raise AssertionError("Canonical GO scope partitions overlap")
    if set().union(*partition_ids) != semantic_ids:
        raise AssertionError("Canonical GO scope partitions are incomplete")
    write_tsv(output_dir / "go_process_scope.tsv", SCOPE_FIELDS, scope_rows)
    write_tsv(
        output_dir / "cell_specific_go_links.tsv",
        SPECIFIC_LINK_FIELDS,
        specific_links,
    )
    write_tsv(output_dir / "cell_general_go_terms.tsv", GENERAL_FIELDS, general)
    write_tsv(
        output_dir / "cell_specific_unmapped_go_terms.tsv",
        SCOPE_FIELDS,
        specific_unmapped,
    )
    write_tsv(
        output_dir / "cell_specific_partial_go_terms.tsv",
        SCOPE_FIELDS,
        specific_partial,
    )
    write_tsv(output_dir / "not_direct_go_terms.tsv", NOT_DIRECT_FIELDS, not_direct)
    write_tsv(output_dir / "unresolved_go_terms.tsv", UNRESOLVED_FIELDS, unresolved)
    counts = Counter(str(row["cell_scope"]) for row in scope_rows)
    sources = Counter(str(row["decision_source"]) for row in scope_rows)
    manifest = {
        "pipeline_version": "complete-go-process-scope-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition": {
            "cell_type_specific": "Curated ontology or reviewed biological evidence supports one or more cell classes as the direct process host; mapping status records exact catalog coverage",
            "cell_general": "The cellular process spans unrelated cell classes or has no defensible unique exact host; localize downstream with marker and anatomy",
            "not_direct_human_cell_process": "Organism, tissue, non-human, or non-cell-autonomous process",
            "unresolved": "Evidence remains insufficient after adjudication",
        },
        "inputs": {
            "semantic_summary": str(args.semantic_summary.resolve()),
            "function_summary": str(args.function_summary.resolve()),
            "scope_overrides": str(args.scope_overrides.resolve()),
            "catalog": str(args.catalog.resolve()),
            "cl_obo": str(args.cl_obo.resolve()),
            "direct_rule_links": str(args.direct_rule_links.resolve()),
            "go_obo": str(args.go_obo.resolve()),
            "existing_decisions": str(args.existing_decisions.resolve()),
            "supplemental_decisions": [
                str(path.resolve()) for path in as_paths(args.supplemental_decisions)
            ],
            "existing_specific_links": str(args.existing_specific_links.resolve()),
            "supplemental_specific_links": [
                str(path.resolve())
                for path in as_paths(args.supplemental_specific_links)
            ],
        },
        "output": {
            "go_process_terms": len(scope_rows),
            "scope_counts": dict(sorted(counts.items())),
            "decision_source_counts": dict(sorted(sources.items())),
            "specific_links": len(specific_links),
            "specific_go_terms": len(accepted_specific_go),
            "specific_unmapped_go_terms": len(specific_unmapped),
            "specific_partial_go_terms": len(specific_partial),
            "explicit_full_cl_go_terms": len(intended_by_go),
            "cellular_process_branch_terms": len(cellular_go_ids),
            "nonhuman_scope_guard_terms": len(nonhuman_go_ids),
            "general_go_terms": len(general),
            "not_direct_go_terms": len(not_direct),
            "unresolved_go_terms": len(unresolved),
            "ambiguous_go_terms": len(targets),
            "scope_digest": digest_rows(scope_rows),
            "integrity_checks": {
                "active_go_bp_equals_scope_universe": True,
                "function_summary_equals_scope_universe": True,
                "unique_go_scope_rows": True,
                "unique_specific_go_cl_pairs": True,
                "specific_links_within_catalog": True,
                "specific_links_within_intended_exact_cl": True,
                "scope_mapping_status_consistent": True,
            },
        },
    }
    (output_dir / "scope_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    source = MODULE_ROOT / "work/source"
    supplemental = MODULE_ROOT / "work/scope_completion"
    parser = argparse.ArgumentParser(
        description="Build one canonical cell-scope decision for every GO BP term."
    )
    parser.add_argument(
        "--semantic-summary",
        type=Path,
        default=source / "semantic/semantic_summary.tsv",
    )
    parser.add_argument(
        "--function-summary",
        type=Path,
        default=PROJECT_ROOT / "GO_function/results/go_function_summary.tsv",
    )
    parser.add_argument(
        "--scope-overrides",
        type=Path,
        default=MODULE_ROOT / "curation/scope_overrides.tsv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "/path/to/workspace/marker_literuture/expanded/"
            "expanded_cell_catalog.tsv"
        ),
    )
    parser.add_argument(
        "--cl-obo",
        type=Path,
        default=Path(
            "/path/to/workspace/UHPO（6-8）/cell_lineage/"
            "cell_ontology/cl-basic.obo"
        ),
    )
    parser.add_argument(
        "--direct-rule-links",
        type=Path,
        default=source / "direct_cell_phenotypes/cell_go_phenotype_links.tsv",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=Path(
            "/path/to/workspace/HPO_expand/CPO/data/go/go-basic.obo"
        ),
    )
    parser.add_argument(
        "--existing-decisions",
        type=Path,
        default=source / "direct_cell_phenotypes_llm/llm_go_decisions.tsv",
    )
    parser.add_argument(
        "--supplemental-decisions",
        type=Path,
        action="append",
        default=[
            supplemental / "llm_go_decisions.tsv",
            MODULE_ROOT / "work/scope_completion_retry1/llm_go_decisions.tsv",
            MODULE_ROOT / "work/scope_revalidation_v2/llm_go_decisions.tsv",
            MODULE_ROOT / "work/scope_revalidation_retry1/llm_go_decisions.tsv",
        ],
    )
    parser.add_argument(
        "--existing-specific-links",
        type=Path,
        default=source
        / "direct_cell_phenotypes_llm/cell_go_phenotype_links_final.tsv",
    )
    parser.add_argument(
        "--supplemental-specific-links",
        type=Path,
        action="append",
        default=[
            supplemental / "llm_cell_specific_links.tsv",
            MODULE_ROOT / "work/scope_completion_retry1/llm_cell_specific_links.tsv",
            MODULE_ROOT / "work/scope_revalidation_v2/llm_cell_specific_links.tsv",
            MODULE_ROOT
            / "work/scope_revalidation_retry1/llm_cell_specific_links.tsv",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODULE_ROOT / "results/full"
    )
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
