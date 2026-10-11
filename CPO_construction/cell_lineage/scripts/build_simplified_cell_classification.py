#!/usr/bin/env python3
"""Build a compact, single-parent cell classification from Cell Ontology OBO."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_OBO = ROOT / "cell_ontology" / "cl-basic.obo"
OUTPUT_DIR = ROOT / "cell_ontology" / "processed"


@dataclass
class Term:
    id: str
    name: str = ""
    parents: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str]] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    obsolete: bool = False
    line: int = 0


@dataclass(frozen=True)
class SlimNode:
    id: str
    parent_id: str
    group: str
    note: str


@dataclass(frozen=True)
class NameRule:
    pattern: re.Pattern[str]
    class_id: str
    note: str


# This is the editable taxonomy "slim". It intentionally keeps broad,
# annotation-friendly classes and omits marker/species/anatomy-specific leaves.
SLIM_NODES = [
    SlimNode("CL:0000000", "", "root", "Cell Ontology root."),
    SlimNode("CL:0000255", "CL:0000000", "broad", "Broad eukaryotic cell branch."),
    SlimNode("CL:0001034", "CL:0000000", "in_vitro", "In vitro/culture branch."),
    SlimNode("CL:0000010", "CL:0001034", "in_vitro", "Cultured cell."),
    SlimNode("CL:0011115", "CL:0000000", "precursor", "Generic precursor branch."),
    SlimNode("CL:0000034", "CL:0011115", "precursor", "Generic stem cell."),
    SlimNode("CL:0011026", "CL:0011115", "precursor", "Generic progenitor cell."),
    SlimNode("CL:0009010", "CL:0011026", "precursor", "Transit amplifying cell."),
    SlimNode("CL:0002322", "CL:0000034", "precursor", "Embryonic stem cell."),
    SlimNode("CL:0000134", "CL:0000034", "precursor", "Mesenchymal stem cell."),
    SlimNode("CL:0002321", "CL:0000000", "developmental", "Embryonic cell branch."),
    SlimNode("CL:0000349", "CL:0002321", "developmental", "Extraembryonic cell."),
    SlimNode("CL:0010017", "CL:0002321", "developmental", "Zygote."),
    SlimNode("CL:0000066", "CL:0000255", "epithelial", "Epithelial root."),
    SlimNode("CL:0000067", "CL:0000066", "epithelial", "Ciliated epithelial cell."),
    SlimNode("CL:1100001", "CL:0000066", "epithelial", "Secretory epithelial cell."),
    SlimNode("CL:0000160", "CL:1100001", "epithelial", "Goblet cell."),
    SlimNode("CL:0000158", "CL:1100001", "epithelial", "Club cell."),
    SlimNode("CL:0000646", "CL:0000066", "epithelial", "Basal epithelial-like cell."),
    SlimNode("CL:0000312", "CL:0000066", "epithelial", "Keratinocyte."),
    SlimNode("CL:0002563", "CL:0000066", "epithelial", "Intestinal epithelial cell."),
    SlimNode("CL:0000584", "CL:0002563", "epithelial", "Enterocyte."),
    SlimNode("CL:0000182", "CL:0000066", "epithelial", "Hepatocyte."),
    SlimNode("CL:1000488", "CL:0000066", "epithelial", "Cholangiocyte."),
    SlimNode("CL:0002518", "CL:0000066", "epithelial", "Kidney epithelial cell."),
    SlimNode("CL:0000653", "CL:0002518", "epithelial", "Podocyte."),
    SlimNode("CL:0005009", "CL:0002518", "epithelial", "Renal principal cell."),
    SlimNode("CL:0005010", "CL:0002518", "epithelial", "Renal intercalated cell."),
    SlimNode("CL:0002062", "CL:0000066", "epithelial", "Pulmonary alveolar type 1 cell."),
    SlimNode("CL:0002063", "CL:0000066", "epithelial", "Pulmonary alveolar type 2 cell."),
    SlimNode("CL:0000163", "CL:0000255", "endocrine", "Endocrine cell."),
    SlimNode("CL:0008024", "CL:0000163", "endocrine", "Pancreatic endocrine cell."),
    SlimNode("CL:0000171", "CL:0008024", "endocrine", "Pancreatic alpha/A cell."),
    SlimNode("CL:0000169", "CL:0008024", "endocrine", "Pancreatic beta/B cell."),
    SlimNode("CL:0000173", "CL:0008024", "endocrine", "Pancreatic delta/D cell."),
    SlimNode("CL:0000115", "CL:0000255", "endothelial", "Endothelial root."),
    SlimNode("CL:0002139", "CL:0000115", "endothelial", "Vascular endothelial cell."),
    SlimNode("CL:0000071", "CL:0002139", "endothelial", "Blood vessel endothelial cell."),
    SlimNode("CL:2000008", "CL:0002139", "endothelial", "Microvascular endothelial cell."),
    SlimNode("CL:0002138", "CL:0000115", "endothelial", "Lymphatic vessel endothelial cell."),
    SlimNode("CL:0002320", "CL:0000255", "connective", "Connective tissue root."),
    SlimNode("CL:0008019", "CL:0002320", "connective", "Mesenchymal cell."),
    SlimNode("CL:0000499", "CL:0002320", "connective", "Stromal cell."),
    SlimNode("CL:0000057", "CL:0000499", "connective", "Fibroblast."),
    SlimNode("CL:4033054", "CL:0002320", "connective", "Perivascular cell."),
    SlimNode("CL:0008034", "CL:4033054", "connective", "Mural cell."),
    SlimNode("CL:0000669", "CL:0008034", "connective", "Pericyte."),
    SlimNode("CL:0000136", "CL:0002320", "connective", "Adipocyte."),
    SlimNode("CL:0000650", "CL:0002320", "connective", "Mesangial cell."),
    SlimNode("CL:0007001", "CL:0002320", "connective", "Skeletogenic cell."),
    SlimNode("CL:0001035", "CL:0002320", "connective", "Bone cell."),
    SlimNode("CL:0000062", "CL:0001035", "connective", "Osteoblast."),
    SlimNode("CL:0000137", "CL:0001035", "connective", "Osteocyte."),
    SlimNode("CL:0000092", "CL:0001035", "connective", "Osteoclast; assigned to bone branch for compactness."),
    SlimNode("CL:0000138", "CL:0002320", "connective", "Chondrocyte."),
    SlimNode("CL:0000058", "CL:0002320", "connective", "Chondroblast."),
    SlimNode("CL:0000187", "CL:0000255", "muscle", "Muscle root."),
    SlimNode("CL:0000188", "CL:0000187", "muscle", "Cell of skeletal muscle."),
    SlimNode("CL:0000192", "CL:0000187", "muscle", "Smooth muscle cell."),
    SlimNode("CL:0000359", "CL:0000192", "muscle", "Vascular smooth muscle cell."),
    SlimNode("CL:0008002", "CL:0000187", "muscle", "Skeletal muscle fiber."),
    SlimNode("CL:0002494", "CL:0000187", "muscle", "Cardiocyte."),
    SlimNode("CL:0000746", "CL:0002494", "muscle", "Cardiac muscle cell."),
    SlimNode("CL:0000056", "CL:0000187", "muscle", "Myoblast."),
    SlimNode("CL:0002372", "CL:0000187", "muscle", "Myotube."),
    SlimNode("CL:0002319", "CL:0000255", "neural", "Neural root."),
    SlimNode("CL:0011020", "CL:0002319", "neural", "Neural progenitor cell."),
    SlimNode("CL:0000540", "CL:0002319", "neural", "Neuron."),
    SlimNode("CL:0000101", "CL:0000540", "neural", "Sensory neuron."),
    SlimNode("CL:0000100", "CL:0000540", "neural", "Motor neuron."),
    SlimNode("CL:0000099", "CL:0000540", "neural", "Interneuron."),
    SlimNode("CL:0000617", "CL:0000540", "neural", "GABAergic neuron."),
    SlimNode("CL:0000679", "CL:0000540", "neural", "Glutamatergic neuron."),
    SlimNode("CL:0000700", "CL:0000540", "neural", "Dopaminergic neuron."),
    SlimNode("CL:0000108", "CL:0000540", "neural", "Cholinergic neuron."),
    SlimNode("CL:0000850", "CL:0000540", "neural", "Serotonergic neuron."),
    SlimNode("CL:0000210", "CL:0000540", "neural", "Photoreceptor cell."),
    SlimNode("CL:0000740", "CL:0000540", "neural", "Retinal ganglion cell."),
    SlimNode("CL:0000125", "CL:0002319", "neural", "Glial cell."),
    SlimNode("CL:0000127", "CL:0000125", "neural", "Astrocyte."),
    SlimNode("CL:0000128", "CL:0000125", "neural", "Oligodendrocyte."),
    SlimNode("CL:0002453", "CL:0000125", "neural", "Oligodendrocyte precursor cell."),
    SlimNode("CL:0000129", "CL:0000125", "neural", "Microglial cell."),
    SlimNode("CL:0002573", "CL:0000125", "neural", "Schwann cell."),
    SlimNode("CL:0000681", "CL:0000125", "neural", "Radial glial cell."),
    SlimNode("CL:0000065", "CL:0000125", "neural", "Ependymal cell."),
    SlimNode("CL:0000988", "CL:0000000", "hematopoietic", "Hematopoietic root."),
    SlimNode("CL:0008001", "CL:0000988", "hematopoietic", "Hematopoietic precursor cell."),
    SlimNode("CL:0000037", "CL:0008001", "hematopoietic", "Hematopoietic stem cell."),
    SlimNode("CL:0000049", "CL:0008001", "hematopoietic", "Common myeloid progenitor."),
    SlimNode("CL:0000051", "CL:0008001", "hematopoietic", "Common lymphoid progenitor."),
    SlimNode("CL:0000050", "CL:0008001", "hematopoietic", "Megakaryocyte-erythroid progenitor."),
    SlimNode("CL:0000081", "CL:0000988", "hematopoietic", "Blood cell."),
    SlimNode("CL:0000764", "CL:0000081", "hematopoietic", "Erythroid lineage cell."),
    SlimNode("CL:0000765", "CL:0000764", "hematopoietic", "Erythroblast."),
    SlimNode("CL:0000558", "CL:0000764", "hematopoietic", "Reticulocyte."),
    SlimNode("CL:0000232", "CL:0000764", "hematopoietic", "Erythrocyte."),
    SlimNode("CL:0000556", "CL:0000081", "hematopoietic", "Megakaryocyte."),
    SlimNode("CL:0000233", "CL:0000081", "hematopoietic", "Platelet."),
    SlimNode("CL:0000738", "CL:0000988", "hematopoietic", "Leukocyte."),
    SlimNode("CL:0009002", "CL:0000738", "hematopoietic", "Inflammatory cell."),
    SlimNode("CL:0000542", "CL:0000738", "hematopoietic", "Lymphocyte."),
    SlimNode("CL:0000084", "CL:0000542", "hematopoietic", "T cell."),
    SlimNode("CL:0000789", "CL:0000084", "hematopoietic", "Alpha-beta T cell."),
    SlimNode("CL:0000798", "CL:0000084", "hematopoietic", "Gamma-delta T cell."),
    SlimNode("CL:0000624", "CL:0000084", "hematopoietic", "CD4-positive alpha-beta T cell."),
    SlimNode("CL:0000625", "CL:0000084", "hematopoietic", "CD8-positive alpha-beta T cell."),
    SlimNode("CL:0000815", "CL:0000084", "hematopoietic", "Regulatory T cell."),
    SlimNode("CL:0000813", "CL:0000084", "hematopoietic", "Memory T cell."),
    SlimNode("CL:0000898", "CL:0000084", "hematopoietic", "Naive T cell."),
    SlimNode("CL:0000911", "CL:0000084", "hematopoietic", "Effector T cell."),
    SlimNode("CL:0000236", "CL:0000542", "hematopoietic", "B cell."),
    SlimNode("CL:0000788", "CL:0000236", "hematopoietic", "Naive B cell."),
    SlimNode("CL:0000787", "CL:0000236", "hematopoietic", "Memory B cell."),
    SlimNode("CL:0000786", "CL:0000236", "hematopoietic", "Plasma cell."),
    SlimNode("CL:0000980", "CL:0000236", "hematopoietic", "Plasmablast."),
    SlimNode("CL:0000623", "CL:0000542", "hematopoietic", "Natural killer cell."),
    SlimNode("CL:0001065", "CL:0000542", "hematopoietic", "Innate lymphoid cell."),
    SlimNode("CL:0000766", "CL:0000738", "hematopoietic", "Myeloid leukocyte."),
    SlimNode("CL:0000234", "CL:0000766", "hematopoietic", "Phagocyte."),
    SlimNode("CL:0000094", "CL:0000766", "hematopoietic", "Granulocyte."),
    SlimNode("CL:0000775", "CL:0000094", "hematopoietic", "Neutrophil."),
    SlimNode("CL:0000771", "CL:0000094", "hematopoietic", "Eosinophil."),
    SlimNode("CL:0000767", "CL:0000094", "hematopoietic", "Basophil."),
    SlimNode("CL:0000113", "CL:0000234", "hematopoietic", "Mononuclear phagocyte."),
    SlimNode("CL:0000576", "CL:0000113", "hematopoietic", "Monocyte."),
    SlimNode("CL:0000235", "CL:0000113", "hematopoietic", "Macrophage."),
    SlimNode("CL:0000864", "CL:0000235", "hematopoietic", "Tissue-resident macrophage."),
    SlimNode("CL:0000451", "CL:0000113", "hematopoietic", "Dendritic cell."),
    SlimNode("CL:0000990", "CL:0000451", "hematopoietic", "Conventional dendritic cell."),
    SlimNode("CL:0000784", "CL:0000451", "hematopoietic", "Plasmacytoid dendritic cell."),
    SlimNode("CL:0000097", "CL:0000766", "hematopoietic", "Mast cell."),
    SlimNode("CL:0000039", "CL:0000000", "germ", "Germ line root."),
    SlimNode("CL:0000586", "CL:0000039", "germ", "Germ cell."),
    SlimNode("CL:0000015", "CL:0000586", "germ", "Male germ cell."),
    SlimNode("CL:0000021", "CL:0000586", "germ", "Female germ cell."),
    SlimNode("CL:0000300", "CL:0000586", "germ", "Gamete."),
    SlimNode("CL:0000023", "CL:0000021", "germ", "Oocyte."),
    SlimNode("CL:0000025", "CL:0000021", "germ", "Egg cell."),
    SlimNode("CL:0000019", "CL:0000015", "germ", "Sperm."),
    SlimNode("CL:0000020", "CL:0000015", "germ", "Spermatogonium."),
    SlimNode("CL:0000017", "CL:0000015", "germ", "Spermatocyte."),
    SlimNode("CL:0000018", "CL:0000015", "germ", "Spermatid."),
    SlimNode("CL:0000147", "CL:0000255", "pigment", "Pigment cell."),
    SlimNode("CL:0000148", "CL:0000147", "pigment", "Melanocyte."),
    SlimNode("CL:0000151", "CL:0000000", "functional", "Functional fallback: secretory cell."),
    SlimNode("CL:0000064", "CL:0000255", "functional", "Functional fallback: ciliated cell."),
    SlimNode("CL:0000197", "CL:0000000", "functional", "Functional fallback: sensory receptor cell."),
    SlimNode("CL:0000630", "CL:0000000", "functional", "Functional fallback: supporting cell."),
]


GROUP_PENALTY = {
    "root": 50,
    "broad": 12,
    "functional": 8,
    "precursor": 3,
    "developmental": 2,
}

NON_CELL_RE = re.compile(
    r"(cytoplasm|nucleus|heterochromatin|junction nucleus|myotendinous junction)",
    re.IGNORECASE,
)
NON_TARGET_RE = re.compile(
    r"\b("
    r"fungal|spore|conidium|prokaryotic|heterocyst|photosynthetic|"
    r"nitrogen fixing|actinomycete|hyphal|Eumycetozoan"
    r")\b",
    re.IGNORECASE,
)
SPECIES_SUFFIX_RE = re.compile(r"\s+\(([A-Z][A-Za-z]+|sensu [^)]+)\)$")
TRANSCRIPTOMIC_RE = re.compile(r"\bNN_\d+\b|\b[A-Z]{2,}[-_][A-Z0-9]", re.IGNORECASE)
MARKER_RE = re.compile(r"\bCD\d|positive|negative|expressing|marker", re.IGNORECASE)

BROAD_OR_FALLBACK_CLASS_IDS = {
    "",
    "CL:0000000",
    "CL:0000255",
    "CL:0000151",
    "CL:0000064",
    "CL:0000197",
    "CL:0000630",
}

NAME_RULES = [
    NameRule(
        re.compile(r"^macrophage derived foam cell$", re.IGNORECASE),
        "CL:0000235",
        "name contains a specific immune lineage absent from its is_a parents",
    ),
    NameRule(
        re.compile(r"^smooth muscle cell derived foam cell$", re.IGNORECASE),
        "CL:0000192",
        "name contains a specific muscle lineage absent from its is_a parents",
    ),
    NameRule(
        re.compile(
            r"\b(cardiocyte|his-purkinje|atrioventricular bundle|cardiac septum|"
            r"cardiac valve)\b",
            re.IGNORECASE,
        ),
        "CL:0002494",
        "cardiac name rule used because CL parentage is very broad",
    ),
    NameRule(
        re.compile(r"\bOPC-Oligo\b|\boligodendrocyte precursor\b", re.IGNORECASE),
        "CL:0002453",
        "oligodendrocyte precursor name rule",
    ),
    NameRule(
        re.compile(r"^insulin secreting cell$", re.IGNORECASE),
        "CL:0000169",
        "endocrine hormone name rule",
    ),
    NameRule(
        re.compile(r"^glucagon secreting cell$", re.IGNORECASE),
        "CL:0000171",
        "endocrine hormone name rule",
    ),
    NameRule(
        re.compile(r"^somatostatin secreting cell$", re.IGNORECASE),
        "CL:0000173",
        "endocrine hormone name rule",
    ),
    NameRule(
        re.compile(r"\binterstitial cell\b", re.IGNORECASE),
        "CL:0000499",
        "interstitial/stromal name rule",
    ),
]


def parse_obo(path: Path) -> list[Term]:
    terms: list[Term] = []
    current: Term | None = None
    in_term = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if line == "[Term]":
            if current:
                terms.append(current)
            current = Term(id="", line=line_no)
            in_term = True
            continue
        if line.startswith("[") and line != "[Term]":
            if current:
                terms.append(current)
            current = None
            in_term = False
            continue
        if not in_term or current is None:
            continue
        if line.startswith("id: "):
            current.id = line[4:].strip()
        elif line.startswith("name: "):
            current.name = line[6:].strip()
        elif line.startswith("is_a: "):
            current.parents.append(line[6:].split()[0])
        elif line.startswith("relationship: "):
            parts = line[14:].split(" ! ", 1)[0].split()
            if len(parts) >= 2:
                current.relationships.append((parts[0], parts[1]))
        elif line.startswith("subset: "):
            current.subsets.append(line[8:].split()[0])
        elif line.startswith("is_obsolete: true"):
            current.obsolete = True
    if current:
        terms.append(current)
    return terms


def flags_for(term: Term) -> list[str]:
    flags: list[str] = []
    if SPECIES_SUFFIX_RE.search(term.name):
        flags.append("species_or_sensu_specific")
    if TRANSCRIPTOMIC_RE.search(term.name):
        flags.append("transcriptomic_cluster_like")
    if MARKER_RE.search(term.name):
        flags.append("marker_specific")
    if " of " in term.name or " in " in term.name:
        flags.append("anatomy_specific")
    if term.id.startswith("CL:43"):
        flags.append("species_project_namespace")
    if NON_CELL_RE.search(term.name) and not term.parents:
        flags.append("non_cell_like")
    if NON_TARGET_RE.search(term.name):
        flags.append("non_metazoan_or_microbe_like")
    return flags


def normalize_name(name: str) -> str:
    normalized = name.lower()
    normalized = SPECIES_SUFFIX_RE.sub("", normalized)
    normalized = re.sub(r"\s+\(sensu [^)]+\)", "", normalized)
    normalized = re.sub(r"\bNN_\d+\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def ancestor_distances(term_id: str, by_id: dict[str, Term]) -> dict[str, int]:
    distances: dict[str, int] = {term_id: 0}
    queue = deque([term_id])
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        for parent in by_id[current].parents:
            if parent not in by_id or parent in distances:
                continue
            distances[parent] = current_distance + 1
            queue.append(parent)
    return distances


def best_slim_mapping(
    term: Term,
    by_id: dict[str, Term],
    slim_by_id: dict[str, SlimNode],
    slim_order: dict[str, int],
) -> tuple[str, int]:
    if "non_cell_like" in flags_for(term):
        return "", -1
    distances = ancestor_distances(term.id, by_id)
    candidates = []
    for slim_id, distance in distances.items():
        if slim_id not in slim_by_id:
            continue
        group = slim_by_id[slim_id].group
        penalty = GROUP_PENALTY.get(group, 0)
        # Keep exact retained classes exact, even if they are broad/fallback.
        if distance == 0:
            penalty = 0
        candidates.append((distance + penalty, distance, slim_order[slim_id], slim_id))
    if not candidates:
        return "", -1
    candidates.sort()
    best = candidates[0]
    return best[3], best[1]


def apply_name_rule(term: Term, class_id: str, distance: int) -> tuple[str, int, str, str]:
    if class_id not in BROAD_OR_FALLBACK_CLASS_IDS:
        return class_id, distance, "ancestor", ""
    for rule in NAME_RULES:
        if rule.pattern.search(term.name):
            return rule.class_id, -1, "name_rule", rule.note
    return class_id, distance, "ancestor", ""


def build_curated_children(slim_nodes: list[SlimNode]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for node in slim_nodes:
        if node.parent_id:
            children[node.parent_id].append(node.id)
    return children


def write_tree_markdown(
    path: Path,
    slim_nodes: list[SlimNode],
    terms: dict[str, Term],
    mapped_counts: Counter[str],
    fine_counts: Counter[str],
    summary: dict[str, int | str],
) -> None:
    children = build_curated_children(slim_nodes)
    node_order = {node.id: i for i, node in enumerate(slim_nodes)}
    for child_ids in children.values():
        child_ids.sort(key=lambda item: node_order[item])

    lines = [
        "# Simplified Cell Classification",
        "",
        f"Source: `{INPUT_OBO.relative_to(ROOT)}`",
        "",
        "Rules:",
        "- obsolete CL terms are excluded from mapping outputs;",
        "- only `is_a` links are used for the main mapping;",
        "- `part_of` and `develops_from` relationships are not used as tree parents;",
        "- each retained class has exactly one curated parent;",
        "- species, marker, anatomy, and transcriptomic-cluster terms are mapped to broader retained classes.",
        "",
        "Summary:",
    ]
    for key in sorted(summary):
        lines.append(f"- {key}: {summary[key]}")
    lines.extend(["", "## Tree", ""])

    def add_node(node_id: str, depth: int) -> None:
        term = terms[node_id]
        count = mapped_counts.get(node_id, 0)
        fine_count = fine_counts.get(node_id, 0)
        suffix = f"mapped={count}"
        if fine_count:
            suffix += f", flagged_fine={fine_count}"
        lines.append(f"{'  ' * depth}- {term.name} `{node_id}` ({suffix})")
        for child in children.get(node_id, []):
            add_node(child, depth + 1)

    add_node("CL:0000000", 0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_terms = parse_obo(INPUT_OBO)
    active_terms = [
        term
        for term in all_terms
        if term.id.startswith("CL:") and not term.obsolete and term.name
    ]
    by_id = {term.id: term for term in active_terms}
    slim_by_id = {node.id: node for node in SLIM_NODES}
    slim_order = {node.id: i for i, node in enumerate(SLIM_NODES)}

    missing = [node.id for node in SLIM_NODES if node.id not in by_id]
    missing_parents = [
        (node.id, node.parent_id)
        for node in SLIM_NODES
        if node.parent_id and node.parent_id not in slim_by_id
    ]
    duplicate_slim = [
        item for item, count in Counter(node.id for node in SLIM_NODES).items() if count > 1
    ]
    if missing or missing_parents or duplicate_slim:
        raise SystemExit(
            "Invalid slim configuration: "
            f"missing={missing}; missing_parents={missing_parents}; "
            f"duplicate_slim={duplicate_slim}"
        )

    mappings: list[dict[str, object]] = []
    mapped_counts: Counter[str] = Counter()
    fine_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    unmapped_count = 0
    name_rule_count = 0
    relationship_counts = Counter(rel for term in active_terms for rel, _ in term.relationships)

    for term in sorted(active_terms, key=lambda item: item.id):
        flags = flags_for(term)
        for flag in flags:
            flag_counts[flag] += 1
        class_id, distance = best_slim_mapping(term, by_id, slim_by_id, slim_order)
        if term.id not in slim_by_id:
            class_id, distance, mapping_method, mapping_note = apply_name_rule(
                term, class_id, distance
            )
        else:
            mapping_method = "retained_class"
            mapping_note = ""
        class_name = by_id[class_id].name if class_id else ""
        if class_id:
            mapped_counts[class_id] += 1
            if flags:
                fine_counts[class_id] += 1
            if mapping_method == "name_rule":
                name_rule_count += 1
        else:
            unmapped_count += 1
        parent_names = [
            f"{parent} {by_id[parent].name}"
            for parent in term.parents
            if parent in by_id
        ]
        mappings.append(
            {
                "term_id": term.id,
                "term_name": term.name,
                "class_id": class_id,
                "class_name": class_name,
                "ancestor_distance": distance,
                "mapping_method": mapping_method,
                "mapping_note": mapping_note,
                "is_retained_class": "yes" if term.id in slim_by_id else "no",
                "flags": ";".join(flags),
                "active_is_a_parents": "; ".join(parent_names),
            }
        )

    node_rows: list[dict[str, object]] = []
    for node in SLIM_NODES:
        term = by_id[node.id]
        parent_name = by_id[node.parent_id].name if node.parent_id else ""
        node_rows.append(
            {
                "class_id": node.id,
                "class_name": term.name,
                "parent_id": node.parent_id,
                "parent_name": parent_name,
                "group": node.group,
                "mapped_term_count": mapped_counts.get(node.id, 0),
                "flagged_fine_term_count": fine_counts.get(node.id, 0),
                "note": node.note,
            }
        )

    review_rows = []
    for row in mappings:
        class_id = str(row["class_id"])
        needs_review = (
            bool(row["flags"])
            or not class_id
            or class_id in BROAD_OR_FALLBACK_CLASS_IDS
            or row["mapping_method"] == "name_rule"
        )
        if needs_review:
            review_rows.append(row)
    review_rows.sort(
        key=lambda row: (
            0 if not row["class_id"] else 1,
            str(row["class_name"]),
            str(row["term_name"]),
        )
    )

    normalized_groups: dict[str, list[Term]] = defaultdict(list)
    for term in active_terms:
        normalized_groups[normalize_name(term.name)].append(term)
    duplicate_rows = []
    for normalized, group_terms in sorted(normalized_groups.items()):
        if len(group_terms) < 2:
            continue
        duplicate_rows.append(
            {
                "normalized_name": normalized,
                "term_count": len(group_terms),
                "term_ids": ";".join(term.id for term in group_terms),
                "term_names": "; ".join(term.name for term in group_terms),
            }
        )

    summary: dict[str, int | str] = {
        "source_data_version": source_data_version(INPUT_OBO),
        "all_obo_terms": len(all_terms),
        "active_cl_terms": len(active_terms),
        "obsolete_cl_terms": sum(
            1 for term in all_terms if term.id.startswith("CL:") and term.obsolete
        ),
        "retained_classes": len(SLIM_NODES),
        "mapped_terms": len(active_terms) - unmapped_count,
        "name_rule_mapped_terms": name_rule_count,
        "unmapped_terms": unmapped_count,
        "terms_with_multiple_is_a_parents": sum(
            1 for term in active_terms if len([p for p in term.parents if p in by_id]) > 1
        ),
        "develops_from_relationships_ignored": relationship_counts.get("RO:0002202", 0),
        "part_of_relationships_ignored": relationship_counts.get("BFO:0000050", 0),
        "duplicate_normalized_name_groups": len(duplicate_rows),
    }
    for flag, count in sorted(flag_counts.items()):
        summary[f"flag_{flag}"] = count

    write_tsv(
        OUTPUT_DIR / "simplified_cell_classes.tsv",
        node_rows,
        [
            "class_id",
            "class_name",
            "parent_id",
            "parent_name",
            "group",
            "mapped_term_count",
            "flagged_fine_term_count",
            "note",
        ],
    )
    write_tsv(
        OUTPUT_DIR / "term_to_simplified_class.tsv",
        mappings,
        [
            "term_id",
            "term_name",
            "class_id",
            "class_name",
            "ancestor_distance",
            "mapping_method",
            "mapping_note",
            "is_retained_class",
            "flags",
            "active_is_a_parents",
        ],
    )
    write_tsv(
        OUTPUT_DIR / "review_candidates.tsv",
        review_rows,
        [
            "term_id",
            "term_name",
            "class_id",
            "class_name",
            "ancestor_distance",
            "mapping_method",
            "mapping_note",
            "is_retained_class",
            "flags",
            "active_is_a_parents",
        ],
    )
    write_tsv(
        OUTPUT_DIR / "duplicate_name_candidates.tsv",
        duplicate_rows,
        ["normalized_name", "term_count", "term_ids", "term_names"],
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tree_markdown(
        OUTPUT_DIR / "simplified_cell_classification_tree.md",
        SLIM_NODES,
        by_id,
        mapped_counts,
        fine_counts,
        summary,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


def source_data_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("data-version: "):
            return line.split(": ", 1)[1]
    return ""


if __name__ == "__main__":
    main()
