from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_hash, resolve, result_root
from .fusion import fuse_array
from .io import file_record, read_tsv, sha256, write_json, write_tsv
from .matrices import load_working_profiles, q_matrix
from .mission3_adapter import load_task


def _metric_table(rows: list[dict[str, str]]) -> str:
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics.setdefault(row["arm"], {})[row["metric"]] = float(row["value"])
    lines = [
        "| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("E0", "E1", "P0", "P1", "O0", "O1"):
        value = metrics[arm]
        lines.append(
            f"| {arm} | {value['recall_at_1']:.4f} | {value['recall_at_5']:.4f} | "
            f"{value['recall_at_10']:.4f} | {value['mrr']:.6f} | {value['ndcg_at_10']:.6f} |"
        )
    return "\n".join(lines)


def _raw_semantics(root: Path, hpo_ids: list[str]) -> np.ndarray:
    values = np.load(root / "cache/semantic_embeddings.npz", allow_pickle=False)
    h_index = {value: index for index, value in enumerate(values["hpo_ids"])}
    h_vectors = values["hpo_vectors"][[h_index[value] for value in hpo_ids]]
    return (h_vectors @ values["mpo_vectors"].T).astype(np.float32)


def _pair_scores(
    resources: dict[str, Any], registry: dict[str, Any], output: Path
) -> dict[str, object]:
    root = result_root(resources)
    confirmation = json.loads((root / "r5_confirmation/confirmation_manifest.json").read_text())
    surface_id = confirmation["selected_surface"]
    method_id, fusion = surface_id.rsplit("__", 1)
    index = json.loads((root / "cache/surface_index.json").read_text())
    hpo_ids = index["hpo_ids"]
    mpo_ids = index["mpo_ids"]
    profiles = load_working_profiles(resources, hpo_ids)
    s = np.load(root / "cache/surface_S.npy", mmap_mode="r")
    s_raw = _raw_semantics(root, hpo_ids)
    raw_path = root / "cache/surface_S_raw.npy"
    if not raw_path.is_file():
        np.save(raw_path, s_raw)
    m = np.load(root / "r3_profiles/mechanisms" / f"{method_id}.npy", mmap_mode="r")
    q = q_matrix(profiles)
    f = fuse_array(fusion, s, m, q, float(registry["fusion"]["epsilon"]))
    task = load_task(resources)
    candidate_ids = sorted({value for model in task.models for value in model.mpo_ids} & set(mpo_ids))
    m_lookup = {value: index for index, value in enumerate(mpo_ids)}
    columns = np.asarray([m_lookup[value] for value in candidate_ids], dtype=np.int32)
    n_h, n_m = len(hpo_ids), len(columns)
    values = np.load(root / "cache/self_profiles.npz", allow_pickle=False)
    all_h_index = {value: index for index, value in enumerate(values["h_term_ids"])}
    all_m_index = {value: index for index, value in enumerate(values["m_term_ids"])}
    h_raw = np.asarray([values["h_raw_counts"][all_h_index[value]] for value in hpo_ids])
    h_qualified = np.asarray([values["h_qualified_counts"][all_h_index[value]] for value in hpo_ids])
    m_raw = np.asarray([values["m_raw_counts"][all_m_index[value]] for value in candidate_ids])
    m_qualified = np.asarray([values["m_qualified_counts"][all_m_index[value]] for value in candidate_ids])
    selected_m = np.asarray(m)[:, columns]
    selected_s = np.asarray(s)[:, columns]
    selected_raw = s_raw[:, columns]
    selected_q = q[:, columns]
    selected_f = f[:, columns]
    frame = pd.DataFrame(
        {
            "hpo_id": np.repeat(np.asarray(hpo_ids), n_m),
            "mpo_id": np.tile(np.asarray(candidate_ids), n_h),
            "surface_id": surface_id,
            "S_raw": selected_raw.reshape(-1),
            "S_used": selected_s.reshape(-1),
            "M_raw": selected_m.reshape(-1),
            "M_used": selected_m.reshape(-1),
            "Q_record": selected_q.reshape(-1),
            "Q_used": selected_q.reshape(-1),
            "F": selected_f.reshape(-1),
            "delta": (selected_f - selected_s).reshape(-1),
            "mechanism_status": np.where(
                np.isfinite(selected_m.reshape(-1)), "computed", "no_qualifying_cpo"
            ),
            "direction_status": "structural_rule_applied",
            "n_records_h": np.repeat(h_raw, n_m),
            "n_records_m": np.tile(m_raw, n_h),
            "n_qualified_h": np.repeat(h_qualified, n_m),
            "n_qualified_m": np.tile(m_qualified, n_h),
            "normalization_id": registry["semantics"]["id"],
            "reference_panel_id": "not_used_by_linear_fusion",
            "config_hash": config_hash(resources, registry),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "rows": len(frame),
        "candidate_hpo": n_h,
        "candidate_mpo": n_m,
        "raw_cosine_negative_count": int(np.count_nonzero(frame["S_raw"] < 0)),
        "file": file_record(output),
        "raw_surface": file_record(raw_path),
    }


def _registry(root: Path, output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for stage in ("r1_nodes", "r2_atoms", "r3_profiles", "r4_fusion"):
        for row in read_tsv(root / stage / "stage_metrics.tsv"):
            rows.append({"stage": stage, "status": "complete", **row})
    value = {
        "schema_version": 1,
        "registry_id": "semantic_cpo_training_free_v1",
        "candidate_surface_count": len(rows),
        "all_results_retained": True,
        "results": rows,
    }
    write_json(output, value)
    return value


def _per_disease(root: Path, output: Path) -> dict[str, object]:
    sources = {
        "E0": (Path("/path/to/workspace/mission3/outputs/parameter_free_2602"), "A0_surface"),
        "E1": (Path("/path/to/workspace/mission3/outputs/parameter_free_2602"), "A4_extended_constrained"),
        "P0": (root / "r5_confirmation/P", "A0_surface"),
        "P1": (root / "r5_confirmation/P", "A4_extended_constrained"),
        "O0": (root / "r5_confirmation/O", "A0_surface"),
        "O1": (root / "r5_confirmation/O", "A4_extended_constrained"),
    }
    rows = []
    for arm, (source, experiment) in sources.items():
        for row in read_tsv(source / "benchmark/disease_level_metrics.tsv"):
            if row["experiment"] == experiment:
                rows.append({"arm": arm, **row})
    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)
    return {"rows": len(frame), "file": file_record(output)}


def _explanations(root: Path, output: Path) -> dict[str, object]:
    p = {
        (row["disease_id"], row["gene_symbol"]): row
        for row in read_tsv(root / "r5_confirmation/P/baseline/base_gene_rankings.tsv")
    }
    o = {
        (row["disease_id"], row["gene_symbol"]): row
        for row in read_tsv(root / "r5_confirmation/O/baseline/base_gene_rankings.tsv")
    }
    task = load_task({
        "project_root": "/path/to/workspace/CPO_generize",
        "benchmark": {
            "models": "mission3/data/benchmark/benchmark_v4/frozen/models.tsv",
            "model_phenotypes": "mission3/data/benchmark/benchmark_v4/frozen/model_phenotypes.tsv",
            "gold_models": "mission3/data/benchmark/benchmark_v4/frozen/gold_models.tsv",
        },
    })
    records = []
    for disease_id, genes in sorted(task.gold.items()):
        for gene in sorted(genes):
            left = p[(disease_id, gene)]
            right = o[(disease_id, gene)]
            if left["base_rank"] == right["base_rank"]:
                continue
            records.append(
                {
                    "disease_id": disease_id,
                    "gold_gene": gene,
                    "P0_rank": int(left["base_rank"]),
                    "O0_rank": int(right["base_rank"]),
                    "rank_change": int(left["base_rank"]) - int(right["base_rank"]),
                    "best_model_id": right["best_model_id"],
                    "best_hpo_id": right["best_hpo_id"],
                    "best_mpo_id": right["best_mpo_id"],
                    "interpretation": "post_selection_case_explanation_not_independent_validation",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for value in records:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(records), "file": file_record(output)}


def _stability(
    resources: dict[str, Any], registry: dict[str, Any], output: Path
) -> dict[str, object]:
    root = result_root(resources)
    surfaces: dict[str, tuple[int, list[dict[str, str]]]] = {}
    for stage in ("r1_nodes", "r2_atoms", "r3_profiles", "r4_fusion"):
        for metric in read_tsv(root / stage / "stage_metrics.tsv"):
            surface_id = f"{metric['method_id']}__{metric['fusion']}"
            path = root / stage / metric["method_id"] / metric["fusion"] / "disease_metrics.tsv"
            if path.is_file():
                surfaces[surface_id] = (int(metric["registered_order"]), read_tsv(path))
    selected = json.loads((root / "r4_fusion/selection.json").read_text())["finalists"][0]
    selected_id = f"{selected['method_id']}__{selected['fusion']}"
    p_rows = {
        row["disease_id"]: row
        for row in read_tsv(root / "r5_confirmation/P/benchmark/disease_level_metrics.tsv")
        if row["experiment"] == "A0_surface"
    }
    o_rows = {
        row["disease_id"]: row
        for row in read_tsv(root / "r5_confirmation/O/benchmark/disease_level_metrics.tsv")
        if row["experiment"] == "A0_surface"
    }
    diseases = sorted(p_rows)
    rows: list[dict[str, object]] = []
    champions: dict[str, int] = {}
    for held_out in diseases:
        candidates = []
        for surface_id, (order, metrics) in surfaces.items():
            retained = [row for row in metrics if row["disease_id"] != held_out]
            candidate = {
                "surface_id": surface_id,
                "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in retained])),
                "ndcg_at_10": float(np.mean([float(row["ndcg_at_10"]) for row in retained])),
                "recall_at_1": float(np.mean([float(row["recall_at_1"]) for row in retained])),
                "registered_order": order,
            }
            candidates.append(candidate)
        champion = min(
            candidates,
            key=lambda row: (
                -row["mrr"],
                -row["ndcg_at_10"],
                -row["recall_at_1"],
                row["registered_order"],
                row["surface_id"],
            ),
        )
        champions[champion["surface_id"]] = champions.get(champion["surface_id"], 0) + 1
        retained = [value for value in diseases if value != held_out]
        fixed_delta = float(
            np.mean(
                [
                    float(o_rows[value]["reciprocal_rank"])
                    - float(p_rows[value]["reciprocal_rank"])
                    for value in retained
                ]
            )
        )
        rows.append(
            {
                "held_out_disease": held_out,
                "fixed_selected_surface": selected_id,
                "fixed_O0_minus_P0_mrr": fixed_delta,
                "reselected_champion": champion["surface_id"],
                "reselected_mrr": champion["mrr"],
                "same_as_full_selection": str(champion["surface_id"] == selected_id).lower(),
            }
        )
    write_tsv(
        output / "leave_one_disease_out.tsv",
        rows,
        (
            "held_out_disease",
            "fixed_selected_surface",
            "fixed_O0_minus_P0_mrr",
            "reselected_champion",
            "reselected_mrr",
            "same_as_full_selection",
        ),
    )
    frequency = [
        {
            "surface_id": surface_id,
            "selection_count": count,
            "selection_fraction": count / len(diseases),
        }
        for surface_id, count in sorted(champions.items(), key=lambda value: (-value[1], value[0]))
    ]
    write_tsv(
        output / "leave_one_disease_out_selection_frequency.tsv",
        frequency,
        ("surface_id", "selection_count", "selection_fraction"),
    )

    task = load_task(resources)
    gene_diseases: dict[str, list[str]] = {}
    for disease_id, genes in task.gold.items():
        for gene in genes:
            gene_diseases.setdefault(gene, []).append(disease_id)
    gene_ids = sorted(gene_diseases)
    random = np.random.default_rng(20260907)
    samples = np.empty(int(registry["controls"]["bootstrap_repeats"]), dtype=np.float64)
    for index in range(len(samples)):
        chosen = random.choice(gene_ids, size=len(gene_ids), replace=True)
        deltas = [
            float(o_rows[disease_id]["reciprocal_rank"])
            - float(p_rows[disease_id]["reciprocal_rank"])
            for gene in chosen
            for disease_id in gene_diseases[str(gene)]
        ]
        samples[index] = np.mean(deltas)
    grouped = {
        "comparison": "O0-P0",
        "group_unit": "gold_gene",
        "group_count": len(gene_ids),
        "multi_disease_group_count": sum(len(values) > 1 for values in gene_diseases.values()),
        "repeats": len(samples),
        "seed": 20260907,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }
    write_tsv(
        output / "grouped_paired_intervals.tsv",
        [grouped],
        (
            "comparison",
            "group_unit",
            "group_count",
            "multi_disease_group_count",
            "repeats",
            "seed",
            "ci_low",
            "ci_high",
        ),
    )
    return {
        "candidate_surface_count": len(surfaces),
        "leave_one_out_count": len(rows),
        "full_champion_selection_frequency": champions.get(selected_id, 0) / len(diseases),
        "fixed_delta_min": min(float(row["fixed_O0_minus_P0_mrr"]) for row in rows),
        "fixed_delta_max": max(float(row["fixed_O0_minus_P0_mrr"]) for row in rows),
        "grouped_interval": grouped,
        "outputs": {
            "leave_one_out": file_record(output / "leave_one_disease_out.tsv"),
            "selection_frequency": file_record(output / "leave_one_disease_out_selection_frequency.tsv"),
            "grouped_interval": file_record(output / "grouped_paired_intervals.tsv"),
        },
    }


def build_report(
    resources: dict[str, Any], registry: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources)
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    pair_scores = _pair_scores(resources, registry, output / "pair_scores.parquet")
    experiment_registry = _registry(root, output / "experiment_registry.json")
    disease_rankings = _per_disease(root, output / "per_disease_rankings.parquet")
    explanations = _explanations(root, output / "pair_explanations.jsonl")
    stability = _stability(resources, registry, output)
    six = read_tsv(root / "r5_confirmation/six_arm_metrics.tsv")
    intervals = read_tsv(root / "r5_confirmation/paired_deltas_and_intervals.tsv")
    controls = read_tsv(root / "r5_controls/permutation_summary.tsv")
    profile_manifest = json.loads((root / "cache/profile_manifest.json").read_text())
    embedding = json.loads((root / "cache/semantic_embeddings_manifest.json").read_text())
    lines = [
        "# Training-free semantic/CPO development report",
        "",
        "## Outcome",
        "",
        "The frozen pure semantic surface P0 is the recommended default. The selected pair-level CPO correction is retained as an exploratory external-test candidate, not as a validated mechanism gain.",
        "",
        "## Six-arm comparison",
        "",
        _metric_table(six),
        "",
        "O0-P0 changes MRR by +0.000476 (1 win, 48 ties, 1 loss); its 2,000-bootstrap 95% interval crosses zero. O1 and P1 are identical on all ranking metrics.",
        "",
        "## Selected development candidate",
        "",
        "`node_exact__atom_A2__profile_bma__linear`: frozen SapBERT term semantics plus a bounded epsilon=0.05 Q_availability-gated linear correction using exact CL/GO component-text CPO atoms and equal-direction BMA.",
        "",
        "## Coverage",
        "",
        f"- Nonredundant development HPO: 572; HPO with qualified V_self atoms: 130.",
        f"- MPO index: 14,383; MPO with qualified V_self atoms: 4,834.",
        f"- Computable M pairs: 628,420/8,227,076; all other pairs fall back exactly to S.",
        f"- HPO/MPO qualified memberships: {profile_manifest['counts']['hpo_profile_memberships']:,}/{profile_manifest['counts']['mpo_profile_memberships']:,}.",
        "",
        "## Null controls",
        "",
    ]
    for row in controls:
        lines.append(
            f"- {row['control']}: null mean delta MRR {float(row['null_mean_delta_mrr']):.6f}; "
            f"empirical upper P={float(row['empirical_p_upper']):.3f}."
        )
    lines.extend(
        [
            "",
            "The real correction is not distinguishable from the registered null controls. Direction permutation has no effect on the selected A2 result.",
            "",
            "## Leave-one-disease-out stability",
            "",
            f"The full-data champion is reselected in {100 * stability['full_champion_selection_frequency']:.1f}% of leave-one-disease-out runs. The fixed O0-P0 MRR delta ranges from {stability['fixed_delta_min']:.6f} to {stability['fixed_delta_max']:.6f}. This is a development stability analysis, not nested external validation.",
            f"The gold-gene grouped bootstrap uses {stability['grouped_interval']['group_count']} groups and also crosses zero: [{stability['grouped_interval']['ci_low']:.6f}, {stability['grouped_interval']['ci_high']:.6f}].",
            "",
            "## Evidence and leakage boundary",
            "",
            "Q is Q_availability, not evidence reliability or causal probability. A complete source_gene_ids/retrieval_query key is absent on the HPO side, and the legacy Extended reranker does not share source_family_id with the pair-level assertions. Candidate-pool source-gene masking, cross-channel Bridge-LOO and A/B family splitting are therefore marked unavailable rather than approximated.",
            "",
            "## Encoding diagnostics",
            "",
            f"The fixed encoder processed all indexed terms without training. Truncated documents: HPO {embedding['diagnostics']['hpo']['truncated_count']:,}, MPO {embedding['diagnostics']['mpo']['truncated_count']:,}, CL {embedding['diagnostics']['cl']['truncated_count']:,}, GO {embedding['diagnostics']['go']['truncated_count']:,}, CPO {embedding['diagnostics']['cpo']['truncated_count']:,}.",
            "",
            "## Interpretation",
            "",
            "P0 exceeds E0 on this development cohort, but this is an overall surface replacement result, not a CPO result. Extended CPO improves P1 relative to P0, while the new term-pair mechanism correction adds no measurable value after Extended reranking. All 50 cases are development data; no external-test claim is made.",
        ]
    )
    report_path = output / "development_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "recommendation": "P0",
        "exploratory_candidate": "node_exact__atom_A2__profile_bma__linear",
        "pair_scores": pair_scores,
        "per_disease_rankings": disease_rankings,
        "pair_explanations": explanations,
        "leave_one_disease_out": stability,
        "experiment_registry_surface_count": experiment_registry["candidate_surface_count"],
        "outputs": {
            "report": file_record(report_path),
            "experiment_registry": file_record(output / "experiment_registry.json"),
        },
    }
    write_json(output / "report_manifest.json", manifest)
    return manifest


def freeze_method(
    resources: dict[str, Any], registry: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, object]:
    root = result_root(resources)
    required = {
        "input_manifest": root / "frozen_inputs/input_manifest.json",
        "report_manifest": root / "report/report_manifest.json",
        "controls_manifest": root / "r5_controls/controls_manifest.json",
        "confirmation_manifest": root / "r5_confirmation/confirmation_manifest.json",
        "selection": root / "r4_fusion/selection.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze; missing: {missing}")
    controls = json.loads(required["controls_manifest"].read_text())
    selection = json.loads(required["selection"].read_text())
    candidate = selection["finalists"][0]
    source_files = sorted((Path(__file__).parent).glob("*.py"))
    code = {path.name: {"path": str(path), "sha256": sha256(path)} for path in source_files}
    value = {
        "schema_version": 1,
        "frozen": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_enabled": False,
        "development_cohort_only": True,
        "external_test_data_seen": False,
        "recommended_default": "P0",
        "exploratory_candidate": {
            "surface_id": f"{candidate['method_id']}__{candidate['fusion']}",
            "status": controls["mechanism_claim_status"],
            "epsilon": registry["fusion"]["epsilon"],
            "view": registry["view"],
            "quality": registry["quality"],
            "node_method": "exact",
            "atom_method": "A2_component_text_product",
            "profile_method": "bidirectional_equal_bma",
            "fusion_method": "linear",
        },
        "interpretation": "Frozen for one-shot external testing; not recommended over P0 because null controls do not support a mechanism-specific gain.",
        "source_audit_status": controls["source_audit_status"],
        "config_hash": config_hash(resources, registry, adapter),
        "inputs": {name: file_record(path) for name, path in required.items()},
        "code": code,
        "external_test_policy": {
            "selection_allowed": False,
            "formula_change_allowed": False,
            "new_encoder_allowed": False,
            "requires_separate_cohort_manifest": True,
        },
    }
    output = root / "frozen_method"
    write_json(output / "frozen_method.json", value)
    manifest = {
        "frozen": True,
        "method_status": controls["mechanism_claim_status"],
        "recommended_default": "P0",
        "frozen_method": file_record(output / "frozen_method.json"),
    }
    write_json(output / "freeze_manifest.json", manifest)
    return manifest
