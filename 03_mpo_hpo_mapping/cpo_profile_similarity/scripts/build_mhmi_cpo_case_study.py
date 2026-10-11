#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpo_pipeline.config import load_config, output_root, resource_path
from cpo_pipeline.explain import _select_provenance, _source_indexes
from cpo_pipeline.mapping import MappingContext, _direction_code


RELATION_ORDER = ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")
RELATION_QUOTAS = {
    "exactMatch": 11,
    "closeMatch": 6,
    "broadMatch": 11,
    "narrowMatch": 11,
    "relatedMatch": 11,
}
MATRIX_PATHS = (
    "final/shape",
    "final/hpo_to_mpo",
    "final/mpo_to_hpo",
    "final/direction_support",
    "final/direction_conflict",
    "self_only/shape",
    "inherited_only/shape",
    "aux/cl",
    "aux/go",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _stable_hash(row: pd.Series) -> str:
    return hashlib.sha256(f"{row['hp_id']}|{row['mp_id']}".encode()).hexdigest()


def _even_positions(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    positions = np.linspace(0, length - 1, count)
    selected: list[int] = []
    for value in positions:
        candidate = int(round(float(value)))
        while candidate in selected and candidate + 1 < length:
            candidate += 1
        while candidate in selected and candidate - 1 >= 0:
            candidate -= 1
        selected.append(candidate)
    return sorted(selected)


def _round_robin(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if not len(frame) or count <= 0:
        return frame.iloc[:0]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        key = "|".join(
            (
                str(row["hpo_node_type"]),
                str(row["mpo_node_type"]),
                str(row["hpo_profile_available"]),
                str(row["mpo_profile_available"]),
            )
        )
        groups[key].append(index)
    for key in groups:
        groups[key].sort(key=lambda index: frame.loc[index, "stable_hash"])
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
    return frame.loc[selected]


def _sample_reference(reference: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for relation in RELATION_ORDER:
        group = reference[reference["match_type"] == relation].copy()
        quota = RELATION_QUOTAS[relation]
        positive_count = 3 if relation == "closeMatch" else 5
        zero_count = 0 if relation == "closeMatch" else 1
        na_count = quota - positive_count - zero_count

        positive = group[group["score_status"] == "POSITIVE"].sort_values(
            ["score_shape", "stable_hash"], kind="stable"
        )
        positive_rows = positive.iloc[_even_positions(len(positive), positive_count)].copy()
        positive_rows["sample_stratum"] = [
            f"POSITIVE_Q{index + 1}_OF_{positive_count}" for index in range(len(positive_rows))
        ]

        zero = group[group["score_status"] == "ZERO"].sort_values("stable_hash")
        zero_rows = zero.head(zero_count).copy()
        zero_rows["sample_stratum"] = "ZERO"

        missing = group[group["score_status"] == "NA"].copy()
        missing_rows = _round_robin(missing, na_count).copy()
        missing_rows["sample_stratum"] = "NA_PROFILE_COVERAGE"
        selected.extend((positive_rows, zero_rows, missing_rows))
    result = pd.concat(selected, ignore_index=True)
    relation_rank = {value: index for index, value in enumerate(RELATION_ORDER)}
    result["relation_order"] = result["match_type"].map(relation_rank)
    result = result.sort_values(
        ["relation_order", "score_status", "score_shape", "stable_hash"],
        na_position="last",
        kind="stable",
    ).drop(columns="relation_order")
    if len(result) != 50:
        raise AssertionError(f"expected 50 sampled mappings, observed {len(result)}")
    if result.duplicated(["hp_id", "mp_id"]).any():
        raise AssertionError("duplicate MHMI mapping in sampled cases")
    return result.reset_index(drop=True)


def _read_reference(config: dict[str, Any], context: MappingContext) -> pd.DataFrame:
    reference = pd.read_csv(resource_path(config, "hpo_mpo_reference"), sep="\t", dtype=str)
    reference = reference[
        reference["hp_id"].isin(context.hpo_index)
        & reference["mp_id"].isin(context.mpo_index)
    ].copy()
    h_policy = pd.DataFrame(context.hpo_policy).set_index("term_id")
    m_policy = pd.DataFrame(context.mpo_policy).set_index("term_id")
    reference["hpo_node_type"] = reference["hp_id"].map(h_policy["node_type"])
    reference["mpo_node_type"] = reference["mp_id"].map(m_policy["node_type"])
    reference["hpo_profile_available"] = reference["hp_id"].map(
        lambda value: bool(context.h_profiles[context.hpo_index[value]])
    )
    reference["mpo_profile_available"] = reference["mp_id"].map(
        lambda value: bool(context.m_profiles[context.mpo_index[value]])
    )
    with h5py.File(context.root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        for path in MATRIX_PATHS:
            field = path.replace("/", "_")
            reference[field] = [
                float(
                    handle[path][
                        context.hpo_index[row.hp_id], context.mpo_index[row.mp_id]
                    ]
                )
                for row in reference.itertuples(index=False)
            ]
    reference = reference.rename(
        columns={
            "final_shape": "score_shape",
            "final_hpo_to_mpo": "score_F",
            "final_mpo_to_hpo": "score_R",
            "final_direction_support": "score_support",
            "final_direction_conflict": "score_conflict",
            "self_only_shape": "score_self_only",
            "inherited_only_shape": "score_inherited_only",
            "aux_cl": "score_cl_aux",
            "aux_go": "score_go_aux",
        }
    )
    reference["score_status"] = np.where(
        reference["score_shape"].isna(),
        "NA",
        np.where(reference["score_shape"] > 0, "POSITIVE", "ZERO"),
    )
    reference["stable_hash"] = reference.apply(_stable_hash, axis=1)
    return reference


def _profile_summary(profile: list[dict[str, object]]) -> dict[str, Any]:
    origins = Counter()
    evidence = Counter()
    direction = Counter()
    cl_ids = set()
    go_ids = set()
    source_leaves = 0
    for node in profile:
        for value in str(node["origin_set"]).split("|"):
            if value:
                origins[value] += 1
        for value in str(node["evidence_class_set"]).split("|"):
            if value:
                evidence[value] += 1
        direction[str(node["direction_signature_status"])] += 1
        cl_ids.add(str(node["cl_id"]))
        go_ids.add(str(node["go_id"]))
        source_leaves += int(node["source_leaf_count"])
    return {
        "structure_count": len(profile),
        "unique_cl_count": len(cl_ids),
        "unique_go_count": len(go_ids),
        "origin_node_counts": dict(sorted(origins.items())),
        "evidence_class_node_counts": dict(sorted(evidence.items())),
        "direction_signature_counts": dict(sorted(direction.items())),
        "summed_source_leaf_count": source_leaves,
    }


def _node_payload(
    context: MappingContext,
    side: str,
    term_id: str,
    node: dict[str, object],
    provenance: dict,
    relation_lookup: dict,
    assertion_index: dict,
) -> dict[str, Any]:
    selected, resolved = _select_provenance(
        side,
        term_id,
        str(node["structure_id"]),
        provenance,
        relation_lookup,
    )
    assertion_id = str(resolved.get("resolved_assertion_id", ""))
    assertion = assertion_index.get(assertion_id, {})
    cl_id = str(node["cl_id"])
    go_id = str(node["go_id"])
    return {
        "structure_id": str(node["structure_id"]),
        "cl_id": cl_id,
        "cl_label": context.data.cl.records[cl_id].name,
        "go_id": go_id,
        "go_label": context.data.go.records[go_id].name,
        "origin_set": str(node["origin_set"]),
        "evidence_class_set": str(node["evidence_class_set"]),
        "direction_signature_status": str(node["direction_signature_status"]),
        "direction_target": str(node["direction_target"]),
        "direction_property": str(node["direction_property"]),
        "direction_state": str(node["direction_state"]),
        "assertion_count": int(node["assertion_count"]),
        "source_leaf_count": int(node["source_leaf_count"]),
        "selected_origin": _text(selected.get("origin", "")),
        "source_leaf_id": _text(selected.get("source_leaf_id", "")),
        "inheritance_path": _text(selected.get("canonical_path", "")),
        "assertion_id": assertion_id,
        "publication_ids": _text(assertion.get("publication_ids", "")),
        "evidence_sentence": _text(assertion.get("evidence_sentence", "")),
        "source_dataset": _text(assertion.get("source_dataset", "")),
    }


def _pair_details(
    context: MappingContext,
    hpo_id: str,
    mpo_id: str,
    provenance: dict,
    relation_lookup: dict,
    assertion_index: dict,
) -> dict[str, Any]:
    h_profile = context.h_profiles[context.hpo_index[hpo_id]]
    m_profile = context.m_profiles[context.mpo_index[mpo_id]]
    h_nodes = [
        _node_payload(
            context, "HPO", hpo_id, node, provenance, relation_lookup, assertion_index
        )
        for node in h_profile
    ]
    m_nodes = [
        _node_payload(
            context, "MPO", mpo_id, node, provenance, relation_lookup, assertion_index
        )
        for node in m_profile
    ]
    if not h_profile or not m_profile:
        return {
            "hpo_summary": _profile_summary(h_profile),
            "mpo_summary": _profile_summary(m_profile),
            "hpo_nodes": h_nodes,
            "mpo_nodes": m_nodes,
            "top_cpo_matches": [],
            "coverage_hpo": None,
            "coverage_mpo": None,
            "exact_structure_count": 0,
            "pair_count": 0,
            "positive_pair_count": 0,
            "high_pair_count_0_5": 0,
            "high_pair_count_0_8": 0,
        }

    h_components = np.asarray(
        [context.h_structure_index[str(node["structure_id"])] for node in h_profile]
    )
    m_components = np.asarray(
        [context.m_structure_index[str(node["structure_id"])] for node in m_profile]
    )
    similarities = context.similarity[np.ix_(h_components, m_components)]
    h_best = np.max(similarities, axis=1)
    m_best = np.max(similarities, axis=0)
    h_argmax = np.argmax(similarities, axis=1)
    m_argmax = np.argmax(similarities, axis=0)
    for index, node in enumerate(h_nodes):
        node["best_opposite_structure_id"] = m_nodes[int(h_argmax[index])]["structure_id"]
        node["best_opposite_score"] = float(h_best[index])
    for index, node in enumerate(m_nodes):
        node["best_opposite_structure_id"] = h_nodes[int(m_argmax[index])]["structure_id"]
        node["best_opposite_score"] = float(m_best[index])

    flat = similarities.ravel()
    top_count = min(12, len(flat))
    if top_count < len(flat):
        cutoff = float(np.partition(flat, len(flat) - top_count)[len(flat) - top_count])
        candidates = np.flatnonzero(flat >= cutoff)
    else:
        candidates = np.arange(len(flat))
    positions = sorted(
        ((int(value // len(m_profile)), int(value % len(m_profile))) for value in candidates),
        key=lambda pair: (
            -float(similarities[pair]),
            h_nodes[pair[0]]["structure_id"],
            m_nodes[pair[1]]["structure_id"],
        ),
    )[:10]
    matches = []
    for h_index, m_index in positions:
        h_node = h_nodes[h_index]
        m_node = m_nodes[m_index]
        cl_mica, cl_ic, _ = context.data.cl.mica(h_node["cl_id"], m_node["cl_id"])
        go_mica, go_ic, _ = context.data.go.mica(h_node["go_id"], m_node["go_id"])
        matches.append(
            {
                "hpo_structure_id": h_node["structure_id"],
                "mpo_structure_id": m_node["structure_id"],
                "pair_score": float(similarities[h_index, m_index]),
                "cl_similarity": context.data.cl.lin(h_node["cl_id"], m_node["cl_id"]),
                "go_similarity": context.data.go.lin(h_node["go_id"], m_node["go_id"]),
                "cl_mica_id": cl_mica or "",
                "cl_mica_label": context.data.cl.records[cl_mica].name if cl_mica else "",
                "cl_mica_ic": float(cl_ic),
                "go_mica_id": go_mica or "",
                "go_mica_label": context.data.go.records[go_mica].name if go_mica else "",
                "go_mica_ic": float(go_ic),
                "direction_code": _direction_code(h_profile[h_index], m_profile[m_index]),
            }
        )
    exact = set(node["structure_id"] for node in h_nodes) & set(
        node["structure_id"] for node in m_nodes
    )
    return {
        "hpo_summary": _profile_summary(h_profile),
        "mpo_summary": _profile_summary(m_profile),
        "hpo_nodes": h_nodes,
        "mpo_nodes": m_nodes,
        "top_cpo_matches": matches,
        "coverage_hpo": float(np.mean(h_best > 0)),
        "coverage_mpo": float(np.mean(m_best > 0)),
        "exact_structure_count": len(exact),
        "pair_count": int(similarities.size),
        "positive_pair_count": int(np.count_nonzero(similarities > 0)),
        "high_pair_count_0_5": int(np.count_nonzero(similarities >= 0.5)),
        "high_pair_count_0_8": int(np.count_nonzero(similarities >= 0.8)),
    }


def _observations(case: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    h_count = case["cpo"]["hpo_summary"]["structure_count"]
    m_count = case["cpo"]["mpo_summary"]["structure_count"]
    if not h_count or not m_count:
        missing = "HPO" if not h_count else "MPO"
        if not h_count and not m_count:
            missing = "HPO 和 MPO"
        notes.append(f"{missing} final profile 为空，因此标准映射对没有主 CPO 分数。")
        return notes
    if case["cpo"]["exact_structure_count"]:
        notes.append(
            f"两侧共享 {case['cpo']['exact_structure_count']} 个完全相同的 (CL, GO) 结构。"
        )
    if case["scores"]["shape"] == 0:
        notes.append("两侧均有完整 CPO，但所有 CPO 配对的语义相似度均为 0。")
    if case["scores"]["F"] is not None and case["scores"]["R"] is not None:
        difference = abs(case["scores"]["F"] - case["scores"]["R"])
        if difference >= 0.3:
            notes.append(f"F/R 相差 {difference:.3f}，匹配主要是单侧覆盖，不是对称一致。")
    inherited = (
        case["cpo"]["hpo_summary"]["origin_node_counts"].get("inherited", 0)
        + case["cpo"]["mpo_summary"]["origin_node_counts"].get("inherited", 0)
    )
    if inherited:
        notes.append("最终 profile 涉及 inherited 结构；该分数可代表分支机制，不代表父术语自身直接观察。")
    if case["scores"]["shape"] is not None and case["scores"]["shape"] < 0.1:
        notes.append("这是人工参考映射，但主 CPO 相似度低于 0.1，提示当前中间层未捕获其本体语义关系。")
    if case["scores"]["conflict"] is not None and case["scores"]["conflict"] > 0:
        notes.append("存在明确方向冲突；结构接近不能解释为同方向异常。")
    return notes


def _figure(sample: pd.DataFrame, destination: Path) -> None:
    counts = pd.crosstab(sample["match_type"], sample["score_status"]).reindex(
        RELATION_ORDER, fill_value=0
    )
    for status in ("POSITIVE", "ZERO", "NA"):
        if status not in counts:
            counts[status] = 0
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bottom = np.zeros(len(counts))
    colors = {"POSITIVE": "#198754", "ZERO": "#6c757d", "NA": "#d9dde3"}
    for status in ("POSITIVE", "ZERO", "NA"):
        axes[0].bar(counts.index, counts[status], bottom=bottom, label=status, color=colors[status])
        bottom += counts[status].to_numpy()
    axes[0].set_title("Stratified 50-case sample")
    axes[0].set_ylabel("Cases")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].legend(frameon=False)
    finite = sample.dropna(subset=["score_shape"])
    for relation in RELATION_ORDER:
        values = finite.loc[finite["match_type"] == relation, "score_shape"]
        axes[1].scatter(
            [relation] * len(values),
            values,
            s=32,
            alpha=0.8,
            label=relation,
        )
    axes[1].set_title("CPO shape scores in sampled standards")
    axes[1].set_ylabel("S")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].tick_params(axis="x", rotation=28)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def build(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    context = MappingContext(config)
    root = output_root(config)
    output = root / "mhmi_cpo_case_study"
    data_root = output / "data"
    assets_root = output / "assets"
    data_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    reference = _read_reference(config, context)
    sample = _sample_reference(reference)
    assertion_index, provenance, relation_lookup = _source_indexes(root, config)
    cases = []
    with h5py.File(root / "matrices/hpo_mpo_cpo.h5", "r") as handle:
        matrix_digest = str(handle.attrs["input_digest"])
    for index, row in sample.iterrows():
        hpo_id = str(row["hp_id"])
        mpo_id = str(row["mp_id"])
        cpo = _pair_details(
            context,
            hpo_id,
            mpo_id,
            provenance,
            relation_lookup,
            assertion_index,
        )
        case = {
            "case_id": f"MHMI-CPO-{index + 1:03d}",
            "sample_stratum": str(row["sample_stratum"]),
            "reference": {
                "hpo_id": hpo_id,
                "hpo_label": str(row["hp_label"]),
                "hpo_node_type": str(row["hpo_node_type"]),
                "mpo_id": mpo_id,
                "mpo_label": str(row["mp_label"]),
                "mpo_node_type": str(row["mpo_node_type"]),
                "predicate_id": str(row["predicate_id"]),
                "match_type": str(row["match_type"]),
                "confidence": _text(row["confidence"]),
                "mapping_justification": _text(row["mapping_justification"]),
                "author_id": _text(row["author_id"]),
                "mapping_date": _text(row["mapping_date"]),
                "source_project": _text(row["source_project"]),
            },
            "scores": {
                "status": str(row["score_status"]),
                "shape": _finite(row["score_shape"]),
                "F": _finite(row["score_F"]),
                "R": _finite(row["score_R"]),
                "support": _finite(row["score_support"]),
                "conflict": _finite(row["score_conflict"]),
                "self_only": _finite(row["score_self_only"]),
                "inherited_only": _finite(row["score_inherited_only"]),
                "cl_aux": _finite(row["score_cl_aux"]),
                "go_aux": _finite(row["score_go_aux"]),
            },
            "cpo": cpo,
        }
        case["observations"] = _observations(case)
        cases.append(case)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampling_method": (
            "Five MHMI relation strata: exact/broad/narrow/related n=11 each and all six "
            "indexed closeMatch pairs. Main strata use five positive score quantiles, one zero, "
            "and five NA profile-coverage cases; closeMatch uses three positive and three NA."
        ),
        "source_reference": str(resource_path(config, "hpo_mpo_reference")),
        "matrix_input_digest": matrix_digest,
        "case_count": len(cases),
        "relation_counts": dict(Counter(case["reference"]["match_type"] for case in cases)),
        "score_status_counts": dict(Counter(case["scores"]["status"] for case in cases)),
        "node_pair_counts": dict(
            Counter(
                f"{case['reference']['hpo_node_type']}|{case['reference']['mpo_node_type']}"
                for case in cases
            )
        ),
    }
    payload = {"summary": summary, "cases": cases}
    json_path = data_root / "cases.json"
    js_path = data_root / "cases.js"
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    js_path.write_text("window.CASE_STUDY_DATA = " + json_text + ";\n", encoding="utf-8")

    sample_output = sample[
        [
            "hp_id",
            "hp_label",
            "mp_id",
            "mp_label",
            "match_type",
            "sample_stratum",
            "score_status",
            "score_shape",
            "score_F",
            "score_R",
            "score_support",
            "score_conflict",
            "hpo_node_type",
            "mpo_node_type",
            "hpo_profile_available",
            "mpo_profile_available",
            "mapping_justification",
            "source_project",
        ]
    ]
    sample_output.to_csv(data_root / "sampled_50_cases.tsv", sep="\t", index=False)
    _figure(sample, assets_root / "sample_overview.png")
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 50 stratified MHMI CPO case studies")
    parser.add_argument("--config", default="config/cpo_v2.yaml")
    args = parser.parse_args()
    print(json.dumps(build(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
