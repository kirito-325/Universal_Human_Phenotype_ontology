import hashlib

import pytest

from hpo_agent.adapters import import_v6, phenotype_from_legacy
from hpo_agent.storage import read_jsonl, save_json, save_jsonl


def test_legacy_profiles_do_not_seed_extraction_with_expected_cl_go():
    term = phenotype_from_legacy({"hpo_id": "HP:0001250", "label": "Seizure",
                                  "candidate_cells": [{"source_id": "CL:0000540", "label": "neuron"}],
                                  "disease_contexts": [{"label": "test disease", "source_id": "TEST:1"}]})
    assert "candidate_cells" not in term.model_dump()
    assert term.disease_contexts == ["test disease"]


def test_v6_import_verifies_hash_and_keeps_text(tmp_path):
    root = tmp_path / "library" / "HP_0001250"
    save_json(root / "manifests" / "test.json", {
        "hpo_id": "HP:0001250", "run_details": {"profile": {"label": "Seizure"}},
    })
    text = "Title\n\nAbstract\n\nSynthetic abstract.\n"
    text_path = root / "content" / "test.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(text)
    structured = root / "content" / "test.json"
    save_json(structured, {"abstract": "Synthetic abstract.", "full_text": "", "source_url": "urn:test"})
    save_jsonl(root / "metadata" / "batches" / "test.jsonl", [{
        "hpo_id": "HP:0001250", "record_id": "test", "title": "Title", "pmid": "123",
        "content": {"text_path": "content/test.txt", "structured_path": "content/test.json",
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest()},
    }])
    output = tmp_path / "imported"
    report = import_v6(tmp_path / "library", "HP:0001250", "test", output)
    assert report["documents"] == 1
    assert read_jsonl(output / "documents.jsonl")[0]["text"] == text
    text_path.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        import_v6(tmp_path / "library", "HP:0001250", "test", output)
