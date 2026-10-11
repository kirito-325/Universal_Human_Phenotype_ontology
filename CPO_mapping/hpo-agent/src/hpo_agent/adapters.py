"""Read existing HPO_link_v6 retrieval artifacts without importing its runtime."""

from __future__ import annotations

import json
from pathlib import Path

from .ontology import digest_file
from .schemas import Document, Phenotype, Section
from .storage import read_jsonl, save_json, save_jsonl


def phenotype_from_legacy(row: dict) -> Phenotype:
    if "phenotype_id" in row:
        return Phenotype.model_validate(row)
    phenotype_id = row.get("hpo_id")
    diseases = row.get("disease_contexts", [])
    return Phenotype(
        phenotype_id=phenotype_id, label=row.get("label") or row.get("hpo_label"),
        definition=row.get("definition", ""), synonyms=row.get("synonyms", []),
        disease_contexts=[d["label"] if isinstance(d, dict) else d for d in diseases],
        seed_pmids=row.get("curated_reference_pmids", row.get("pmids", [])),
    )


def import_v6(literature_root: Path, hpo_id: str, run_id: str, output: Path) -> dict:
    # Validate user-supplied identifiers before using them as path components.
    Phenotype(phenotype_id=hpo_id, label="validation")
    if not hpo_id.startswith("HP:") or not run_id or Path(run_id).name != run_id:
        raise ValueError("import-v6 requires an HP ID and a simple retrieval run ID")
    root = (literature_root / hpo_id.replace(":", "_")).resolve()
    manifest_path = root / "manifests" / (run_id + ".json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["hpo_id"] != hpo_id:
        raise ValueError("manifest phenotype mismatch")
    profile = dict(manifest.get("run_details", {}).get("profile", {}))
    profile["hpo_id"] = hpo_id
    phenotype = phenotype_from_legacy(profile)
    documents = []
    skipped = []

    def located(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("content path outside phenotype literature root")
        return path

    for row in read_jsonl(root / "metadata" / "batches" / (run_id + ".jsonl")):
        if row.get("hpo_id") != hpo_id:
            raise ValueError("metadata phenotype mismatch")
        content = row.get("content", {})
        text_path = located(content["text_path"])
        structured_path = located(content["structured_path"])
        for path, key in ((text_path, "content_sha256"), (structured_path, "structured_sha256")):
            if content.get(key) and digest_file(path) != content[key]:
                raise ValueError(f"stored content hash mismatch: {path}")
        structured = json.loads(structured_path.read_text())
        # v6 can store a title plus a failure/status string; that is not article evidence.
        if not structured.get("abstract", "").strip() and not structured.get("full_text", "").strip():
            skipped.append({"record_id": row["record_id"], "reason": "no_article_text"})
            continue
        text = text_path.read_text(encoding="utf-8")
        sections = []
        blocks = [{"section": "abstract", "text": structured.get("abstract", "")},
                  *structured.get("sections", [])]
        for block in blocks:
            fragment = block.get("text", "").strip()
            start = text.find(fragment) if fragment else -1
            if start >= 0:
                sections.append(Section(name=block.get("section", "body"), start=start, end=start + len(fragment)))
        url = structured.get("source_url") or row.get("landing_url")
        if not url and row.get("pmid"):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/"
        if not url:
            raise ValueError(f"missing source locator for {row['record_id']}")
        documents.append(Document(
            document_id=row["record_id"], phenotype_ids=[hpo_id], title=row["title"],
            text=text, source_url=url,
            source_kind="full_text" if structured.get("full_text", "").strip() else "abstract",
            pmid=row.get("pmid") or "", pmcid=row.get("pmcid") or "", doi=row.get("doi") or "",
            sections=sections, retracted="Retracted Publication" in row.get("publication_types", []),
            retrieval_notes=[f"imported_v6_manifest:{manifest_path}"],
        ))
    save_jsonl(output / "terms.jsonl", [phenotype])
    save_jsonl(output / "documents.jsonl", documents)
    report = {"source_manifest": str(manifest_path), "source_manifest_sha256": digest_file(manifest_path),
              "documents": len(documents), "skipped": skipped,
              "note": "CL/GO seeds and prior adjudications were not imported into evidence extraction."}
    save_json(output / "import_report.json", report)
    return report
