"""Negative evidence labels do not replace target-aware scientific review."""

import pytest
from helpers import judgments
from pydantic import ValidationError

from mpo_agent.assembly import assemble_review
from mpo_agent.schemas import ClaimDraft, ExperimentalValidation, ScientificReview, Settings
from mpo_agent.validation import anchor_experiments, anchor_reasoning, release_gate


@pytest.fixture
def mouse_study(document):
    return document.model_copy(update={"text": "Synthetic study of a component process in another mouse model."})


@pytest.fixture
def prediction(phenotype, draft, mouse_study):
    return ClaimDraft.model_validate({
        **draft.model_dump(), "evidence_mode": "inferred", "cell_evidence_mode": "inferred",
        "link_evidence_mode": "inferred", "bridge_scope": "mechanistically_compatible",
        "species": "mouse", "taxon_id": "NCBITaxon:10090",
        "supporting_quote": "", "supporting_document_id": "", "assertion": "model_inference",
        "evidence_type": "inference", "relation_type": "mechanistically_compatible_with",
        "reasoning_steps": [{"source": phenotype.label, "target": draft.cellular_statement,
            "scope": "cell_identity", "basis": "model_knowledge", "relation": "compatible_with",
            "explanation": "Synthetic target-model inference with an explicit transfer assumption.",
            "citations": [], "assumptions": ["The component mechanism applies to the target model."]}],
        "experimental_evidence": [{"document_id": mouse_study.document_id, "quote": mouse_study.text,
            "scope": "whole_claim", "effect": "contradicts", "study_type": "observation",
            "species": "mouse", "experimental_context": "A different mouse model, synthetic test only.",
            "finding": "A component study in another model does not establish the target-model mechanism."}],
    })


@pytest.mark.parametrize("dimension,status,expected", [
    ("experimental_evidence", "uncertain", "conditional"),
    ("phenotype_link", "invalid", "rejected"),
    ("experimental_evidence", "invalid", "rejected"),
])
def test_review_decides_applicability_without_label_veto(
    phenotype, prediction, mapping, dimension, status, expected,
):
    value = judgments()
    value["experimental_evidence"] = {"status": "uncertain",
        "reason": "Another model's negative finding does not invalidate the target prediction."}
    value[dimension] = {"status": status, "reason": (
        "An applicable read result explicitly invalidates a necessary premise."
        if status == "invalid" else "Faithful finding, limited target-model applicability.")}
    reviewed = assemble_review(ScientificReview.model_validate(value), has_experiments=True)
    steps = anchor_reasoning(phenotype, prediction, None, {}, [])
    released, reasons = release_gate(phenotype, prediction, None, mapping, reviewed, steps)
    assert released == expected
    assert ("scientific_review_blocked" in reasons) == (expected == "rejected")
    assert prediction.experimental_evidence[0].effect == "contradicts"


def test_negative_findings_remain_in_empirical_record(prediction, mouse_study):
    entries = anchor_experiments(prediction, {mouse_study.document_id: mouse_study},
        [{"document_id": mouse_study.document_id, "start": 0, "end": len(mouse_study.text)}])
    recorded = ExperimentalValidation(evidence=entries)
    assert recorded.status == "contradictory"
    assert recorded.evidence[0].evidence.quote == mouse_study.text
    assert recorded.evidence[0].experimental_context == prediction.experimental_evidence[0].experimental_context


def test_search_budget_reserves_support_without_requiring_challenge():
    assert Settings(max_searches=4).max_searches == 4
    with pytest.raises(ValidationError, match="supporting searches"):
        Settings(max_searches=3)
