"""Run and validate the four completed UHPO manuscript stages."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from uhpo.paths import REPO_ROOT
from uhpo.stages.stage1_markers.paths import (
    CATALOG_DIR,
    DATABASE_DIR,
    MARKER_DIR,
    NAME_MAPPING_DIR,
    RAW_CL_OBO,
)
from uhpo.stages.stage2_cpo.paths import (
    ASSIGNMENT_DIR,
    CATALOG,
    CATALOG_CONFIG,
    CELL_MARKER_DIR,
    GOA_HUMAN,
    GO_FUNCTION_DIR,
    GO_OBO,
    HGNC,
    MGI_MARKERS as CPO_MGI_MARKERS,
    MOUSE_HUMAN_ORTHOLOGS,
    LLM_BATCH_DIR,
    RELATION_DIR,
    SEMANTIC_DIR,
)
from uhpo.stages.stage3_hpo.paths import (
    ANATOMY_DIR as HPO_ANATOMY_DIR,
    CELL_GO_LINKS,
    CELL_MARKERS,
    CL_ANATOMY,
    GENERAL_GO_TERMS,
    GENES_TO_PHENOTYPE,
    GO_FUNCTIONS,
    HPO_ANATOMY,
    HPO_OBO,
    HPOA,
    LINK_DIR as HPO_LINK_DIR,
    UBERON_ANCESTORS,
)
from uhpo.stages.stage4_mpo.paths import (
    ANATOMY_DIR as MPO_ANATOMY_DIR,
    CLASSIFICATION_BATCH_DIR,
    CLASSIFICATION_BATCH_MANIFEST,
    CLEAN_MANIFEST,
    CLEAN_MPO_OBO,
    EXCLUDED_TERMS,
    HUMAN_GENE_EVIDENCE,
    LINK_DIR as MPO_LINK_DIR,
    MGI_GENE_PHENO,
    MGI_HOMOLOGY,
    MGI_MARKERS as MPO_MGI_MARKERS,
    MPO_ANATOMY,
    MPO_OBO,
    MPO_UBERON_ANCESTORS,
    ORTHOLOG_DIR,
    TERM_DECISIONS,
    TERM_DECISION_MANIFEST,
    VERSION_MANIFEST,
)


@dataclass(frozen=True)
class Step:
    key: str
    stage: int
    title: str
    module: str
    arguments: tuple[str, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    network: bool = False
    api: bool = False

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.arguments]


def values(*items: object) -> tuple[str, ...]:
    return tuple(str(item) for item in items)


STEPS: tuple[Step, ...] = (
    Step(
        "1.01_catalog", 1, "构建扩展 CL 目录",
        "uhpo.stages.stage1_markers.workflows.build_expanded_cell_catalog", (),
        (RAW_CL_OBO,), (CATALOG,),
    ),
    Step(
        "1.02_discover", 1, "检索单细胞 Marker 文献",
        "uhpo.stages.stage1_markers.literature_pipeline", ("discover",),
        (), (DATABASE_DIR / "index.jsonl",), network=True,
    ),
    Step(
        "1.03_extract", 1, "提取补充表格",
        "uhpo.stages.stage1_markers.literature_pipeline", ("extract-tables", "--limit", "0"),
        (DATABASE_DIR / "index.jsonl",), (DATABASE_DIR / "standardization_summary.json",),
    ),
    Step(
        "1.04_standardize", 1, "规则标准化 Marker 表",
        "uhpo.stages.stage1_markers.literature_pipeline", ("standardize-marker-tables",),
        (DATABASE_DIR / "index.jsonl",), (DATABASE_DIR / "standardization_summary.json",),
    ),
    Step(
        "1.05_agent_standardize", 1, "按论文语境标准化 Marker 表",
        "uhpo.stages.stage1_markers.literature_pipeline", ("agent-standardize-marker-tables",),
        (DATABASE_DIR / "index.jsonl",), (DATABASE_DIR / "agent_standardization_summary.json",), api=True,
    ),
    Step(
        "1.06_filter_statistics", 1, "筛选正向 Marker 统计",
        "uhpo.stages.stage1_markers.workflows.filter_marker_statistics", (),
        (DATABASE_DIR / "papers",), (DATABASE_DIR / "marker_statistics_filter_summary.json",),
    ),
    Step(
        "1.07_collect_names", 1, "汇总标准化细胞名称",
        "uhpo.stages.stage1_markers.workflows.build_agent_standardized_cell_name_table", (),
        (DATABASE_DIR / "papers",), (DATABASE_DIR / "agent_standardized_cell_names.tsv",),
    ),
    Step(
        "1.08_filter_names", 1, "过滤非细胞标签",
        "uhpo.stages.stage1_markers.workflows.filter_agent_standardized_cell_names", (),
        (DATABASE_DIR / "agent_standardized_cell_names.tsv",),
        (DATABASE_DIR / "agent_standardized_cell_names_filtered.tsv",),
    ),
    Step(
        "1.09_sapbert", 1, "SapBERT 召回目录候选",
        "uhpo.stages.stage1_markers.workflows.build_sapbert_name_mapping", (),
        (DATABASE_DIR / "agent_standardized_cell_names_filtered.tsv", CATALOG),
        (NAME_MAPPING_DIR / "sapbert_top1.tsv", NAME_MAPPING_DIR / "sapbert_top10.tsv"),
    ),
    Step(
        "1.10_match_names", 1, "解析细胞名称到目录节点",
        "uhpo.stages.stage1_markers.workflows.match_cell_names_to_catalog", (),
        (DATABASE_DIR / "agent_standardized_cell_names_filtered.tsv", CATALOG),
        (DATABASE_DIR / "cell_name_catalog_matches.tsv",), api=True,
    ),
    Step(
        "1.11_marker_scores", 1, "计算细胞 Marker 候选分数",
        "uhpo.stages.stage1_markers.statistics.build_marker_scores", (),
        (DATABASE_DIR / "papers", NAME_MAPPING_DIR / "sapbert_top1.tsv", CATALOG),
        (MARKER_DIR / "expanded_marker_scores.tsv",),
    ),
    Step(
        "1.12_significant_markers", 1, "生成显著 Marker 候选",
        "uhpo.stages.stage1_markers.statistics.build_significant_markers", (),
        (MARKER_DIR / "expanded_marker_scores.tsv",),
        (MARKER_DIR / "significant_marker_scores.tsv",),
    ),
    Step(
        "2.01_cell_markers", 2, "导出建链用细胞 Marker 集",
        "uhpo.stages.stage2_cpo.export_cell_markers", (),
        (CATALOG, MARKER_DIR / "expanded_marker_scores.tsv", HGNC, MOUSE_HUMAN_ORTHOLOGS, CPO_MGI_MARKERS),
        (CELL_MARKER_DIR / "cell_marker_genes.tsv.gz", CELL_MARKER_DIR / "cell_marker_manifest.json"),
    ),
    Step(
        "2.02_go_functions", 2, "导出 GO BP 直接功能基因集",
        "uhpo.stages.stage2_cpo.export_go_functions", (),
        (GO_OBO, GOA_HUMAN),
        (GO_FUNCTION_DIR / "go_function_genes.tsv", GO_FUNCTION_DIR / "go_function_manifest.json"),
    ),
    Step(
        "2.03_semantic_recall", 2, "召回 GO--CL 直接关系候选",
        "uhpo.stages.stage2_cpo.semantic_mapper",
        values("--go-obo", GO_OBO, "--cl-obo", RAW_CL_OBO, "--catalog", CATALOG,
               "--all-biological-process", "--output-dir", SEMANTIC_DIR, "--config", CATALOG_CONFIG),
        (GO_OBO, RAW_CL_OBO, CATALOG),
        (SEMANTIC_DIR / "semantic_summary.tsv", SEMANTIC_DIR / "semantic_candidates.tsv"),
    ),
    Step(
        "2.04_rule_assignment", 2, "规则判定精确 CL--GO 关系",
        "uhpo.stages.stage2_cpo.assign_relations", (),
        (SEMANTIC_DIR / "semantic_summary.tsv", SEMANTIC_DIR / "semantic_candidates.tsv", GO_OBO, CATALOG),
        (ASSIGNMENT_DIR / "cell_go_phenotype_links.tsv", ASSIGNMENT_DIR / "go_assignment_status.tsv"),
    ),
    Step(
        "2.05_llm_adjudication", 2, "受约束模型判定未决关系",
        "uhpo.stages.stage2_cpo.adjudicate_relations", ("--aggregate-only",),
        (ASSIGNMENT_DIR / "cell_go_phenotype_links.tsv", ASSIGNMENT_DIR / "go_assignment_status.tsv", GO_OBO, RAW_CL_OBO, CATALOG,
         LLM_BATCH_DIR / "main", LLM_BATCH_DIR / "repair"),
        (CELL_GO_LINKS, GENERAL_GO_TERMS, RELATION_DIR / "llm_direct_manifest.json"),
    ),
    Step(
        "3.01_anatomy", 3, "物化 HPO/CL--Uberon 解剖约束",
        "uhpo.stages.stage3_hpo.anatomy", (),
        (HPO_OBO, CELL_MARKERS),
        (HPO_ANATOMY, CL_ANATOMY, UBERON_ANCESTORS, HPO_ANATOMY_DIR / "anatomy_constraint_manifest.json"), network=True,
    ),
    Step(
        "3.02_hpo_cpo_links", 3, "构建全部合格 HPO--CPO links",
        "uhpo.stages.stage3_hpo.run_all", (),
        (HPO_OBO, HPOA, GENES_TO_PHENOTYPE, CELL_MARKERS, GO_FUNCTIONS, CELL_GO_LINKS,
         GENERAL_GO_TERMS, HPO_ANATOMY, CL_ANATOMY, UBERON_ANCESTORS),
        (HPO_LINK_DIR / "hpo_cell_mechanism_triples.tsv", HPO_LINK_DIR / "hpo_cell_mechanism_manifest.json"),
    ),
    Step(
        "4.01_versions", 4, "记录 MPO/MGI 输入版本",
        "uhpo.stages.stage4_mpo.versions",
        values("--mpo", MPO_OBO, "--gene-pheno", MGI_GENE_PHENO, "--markers", MPO_MGI_MARKERS,
               "--orthologs", MGI_HOMOLOGY, "--output", VERSION_MANIFEST),
        (MPO_OBO, MGI_GENE_PHENO, MPO_MGI_MARKERS, MGI_HOMOLOGY), (VERSION_MANIFEST,),
    ),
    Step(
        "4.02_classify_terms", 4, "保守判定 MPO 人类相关性",
        "uhpo.stages.stage4_mpo.classify_terms",
        values("--mpo-obo", MPO_OBO, "--output", TERM_DECISIONS, "--batch-dir", CLASSIFICATION_BATCH_DIR,
               "--aggregate-only"),
        (MPO_OBO, CLASSIFICATION_BATCH_MANIFEST), (TERM_DECISIONS, TERM_DECISION_MANIFEST),
    ),
    Step(
        "4.03_prepare_terms", 4, "生成清洗后的 MPO 子集",
        "uhpo.stages.stage4_mpo.prepare_terms",
        values("--mpo-obo", MPO_OBO, "--decisions", TERM_DECISIONS, "--clean-obo", CLEAN_MPO_OBO,
               "--excluded", EXCLUDED_TERMS, "--summary", CLEAN_MANIFEST),
        (MPO_OBO, TERM_DECISIONS), (CLEAN_MPO_OBO, EXCLUDED_TERMS, CLEAN_MANIFEST),
    ),
    Step(
        "4.04_orthologs", 4, "按 MGI ID 转换人鼠同源证据",
        "uhpo.stages.stage4_mpo.orthologs",
        values("--mpo-obo", CLEAN_MPO_OBO, "--gene-pheno", MGI_GENE_PHENO, "--markers", MPO_MGI_MARKERS,
               "--homology", MGI_HOMOLOGY, "--output-dir", ORTHOLOG_DIR),
        (CLEAN_MPO_OBO, MGI_GENE_PHENO, MPO_MGI_MARKERS, MGI_HOMOLOGY),
        (HUMAN_GENE_EVIDENCE, ORTHOLOG_DIR / "ortholog_mapping_manifest.json"),
    ),
    Step(
        "4.05_anatomy", 4, "物化 MPO--Uberon 解剖约束",
        "uhpo.stages.stage4_mpo.anatomy",
        values("--mpo-obo", CLEAN_MPO_OBO, "--existing-ancestors", UBERON_ANCESTORS,
               "--output-dir", MPO_ANATOMY_DIR),
        (CLEAN_MPO_OBO, UBERON_ANCESTORS), (MPO_ANATOMY, MPO_UBERON_ANCESTORS), network=True,
    ),
    Step(
        "4.06_mpo_cpo_links", 4, "构建 MPO--CPO links",
        "uhpo.stages.stage4_mpo.linker", (),
        (CLEAN_MPO_OBO, HUMAN_GENE_EVIDENCE, CELL_MARKERS, GO_FUNCTIONS, CELL_GO_LINKS,
         GENERAL_GO_TERMS, MPO_ANATOMY, CL_ANATOMY, MPO_UBERON_ANCESTORS),
        (MPO_LINK_DIR / "mpo_cell_mechanism_triples.tsv", MPO_LINK_DIR / "mpo_cell_mechanism_manifest.json"),
    ),
)


def selected_steps(stage: str, start_at: str | None, stop_after: str | None) -> list[Step]:
    selected = [step for step in STEPS if stage == "all" or step.stage == int(stage)]
    keys = [step.key for step in selected]
    if start_at:
        if start_at not in keys:
            raise SystemExit(f"--start-at {start_at!r} is not in the selected stage")
        selected = selected[keys.index(start_at):]
    if stop_after:
        current = [step.key for step in selected]
        if stop_after not in current:
            raise SystemExit(f"--stop-after {stop_after!r} is not after --start-at")
        selected = selected[: current.index(stop_after) + 1]
    return selected


def describe(steps: Iterable[Step]) -> None:
    for step in steps:
        flags = ", ".join(label for enabled, label in ((step.network, "network"), (step.api, "API")) if enabled)
        suffix = f" [{flags}]" if flags else ""
        print(f"{step.key}  {step.title}{suffix}")
        print("    " + " ".join(step.command))


def missing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists() or (path.is_file() and path.stat().st_size == 0)]


def validate(steps: Sequence[Step]) -> int:
    failures = 0
    for step in steps:
        missing_inputs = missing(step.inputs)
        missing_outputs = missing(step.outputs)
        status = "ok" if not missing_inputs and not missing_outputs else "incomplete"
        print(f"[{status}] {step.key} {step.title}")
        for path in missing_inputs:
            print(f"  missing input:  {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
        for path in missing_outputs:
            print(f"  missing output: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
        failures += bool(missing_inputs or missing_outputs)
    return 1 if failures else 0


def run(steps: Sequence[Step], dry_run: bool) -> int:
    environment = os.environ.copy()
    source_root = str(REPO_ROOT / "src")
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source_root + (os.pathsep + current_pythonpath if current_pythonpath else "")
    for step in steps:
        missing_inputs = missing(step.inputs)
        if missing_inputs:
            print(f"[blocked] {step.key}: missing input", file=sys.stderr)
            for path in missing_inputs:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"[run] {step.key} {step.title}", flush=True)
        if dry_run:
            print("  " + " ".join(step.command))
            continue
        subprocess.run(step.command, cwd=REPO_ROOT, env=environment, check=True)
        missing_outputs = missing(step.outputs)
        if missing_outputs:
            print(f"[failed] {step.key}: expected output was not created", file=sys.stderr)
            for path in missing_outputs:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"[done] {step.key}", flush=True)
    return 0


def build_parser(fixed_stage: int | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_stage is None:
        parser.add_argument("--stage", choices=("all", "1", "2", "3", "4"), default="all")
        available_steps = STEPS
    else:
        available_steps = tuple(step for step in STEPS if step.stage == fixed_stage)
    parser.add_argument("--start-at", choices=tuple(step.key for step in available_steps))
    parser.add_argument("--stop-after", choices=tuple(step.key for step in available_steps))
    parser.add_argument("--list", action="store_true", help="List the selected commands without running them.")
    parser.add_argument("--validate-only", action="store_true", help="Check canonical inputs and outputs without changing data.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands after checking their inputs.")
    return parser


def main(argv: Sequence[str] | None = None, *, fixed_stage: int | None = None) -> int:
    args = build_parser(fixed_stage).parse_args(argv)
    stage = args.stage if fixed_stage is None else str(fixed_stage)
    steps = selected_steps(stage, args.start_at, args.stop_after)
    if args.list:
        describe(steps)
        return 0
    if args.validate_only:
        return validate(steps)
    return run(steps, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
