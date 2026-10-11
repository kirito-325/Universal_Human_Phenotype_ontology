import pytest
from pydantic import ValidationError

from hpo_agent.ontology import OntologyIndex, build_index
from hpo_agent.schemas import (
    Extraction,
    MappingChoice,
    Observation,
    Phenotype,
)
from hpo_agent.validation import (
    anchor_claim,
    release_gate,
    resolve_mapping,
)


def test_ontology_synonyms_alt_ids_and_obsolete(index):
    assert index.search("neurons", "CL")[0].term_id == "CL:0000540"
    assert index.get("CL:9000001").term_id == "CL:0000540"
    assert index.search("apoptosis", "GO")[0].namespace == "biological_process"
    assert index.get("CL:9000002").obsolete
    assert not index.search("obsolete", "CL")
    assert index.metadata["sources"]["CL"]["sha256"]


def test_index_failed_build_does_not_leave_partial_output(tmp_path):
    missing = tmp_path / "missing.obo"
    output = tmp_path / "failed.sqlite"
    with pytest.raises(FileNotFoundError):
        build_index(missing, missing, output)
    assert not output.exists()


def test_exact_label_beats_specialized_terms_with_repeated_synonyms(tmp_path):
    cl, go = tmp_path / "cl.obo", tmp_path / "go.obo"
    cl.write_text('''[Term]
id: CL:0000556
name: megakaryocyte
def: "A very long definition with many unrelated words for an exact matching label." []

[Term]
id: CL:4033018
name: lung megakaryocyte
synonym: "lung megakaryocyte megakaryocyte megakaryocyte" EXACT []
''')
    go.write_text('[Term]\nid: GO:0030220\nname: platelet formation\n')
    path = tmp_path / "rank.sqlite"
    build_index(cl, go, path)
    assert OntologyIndex(path).search("megakaryocyte", "CL", 1)[0].term_id == "CL:0000556"


@pytest.mark.parametrize("cl_id,seen", [
    ("CL:9999999", {"CL:9999999"}), ("CL:9000002", {"CL:9000002"}),
    ("GO:0006915", {"GO:0006915"}),
])
def test_mapping_cannot_invent_or_use_wrong_ontology(index, cl_id, seen):
    choice = MappingChoice(cl_id=cl_id, go_id="NONE", cell_confidence="high",
                           process_confidence="unmapped", rationale="test")
    with pytest.raises(ValueError):
        resolve_mapping(choice, index, seen)


def test_quote_offsets_reproduce_text_and_cell_is_verbatim(draft, document):
    evidence = anchor_claim(draft, document, 5, len(document.text))
    assert document.text[evidence.span_start:evidence.span_end] == draft.supporting_quote
    assert evidence.text_sha256 == document.sha256
    with pytest.raises(ValueError, match="quote_not"):
        anchor_claim(draft.model_copy(update={"supporting_quote": "Invented quotation"}), document, 0, len(document.text))
    with pytest.raises(ValueError, match="cell_not"):
        anchor_claim(draft.model_copy(update={"cell_text": "motor neurons"}), document, 0, len(document.text))
    with pytest.raises(ValueError, match="quote_not"):
        anchor_claim(draft, document, 0, 15)


def test_retraction_blocks_claim(draft, document):
    with pytest.raises(ValueError, match="retracted"):
        anchor_claim(draft, document.model_copy(update={"retracted": True}), 0, len(document.text))


def test_gate_preserves_causality_and_species(phenotype, draft, document, mapping, review):
    evidence = anchor_claim(draft, document, 0, len(document.text))
    assert release_gate(phenotype, draft, evidence, mapping, review)[0] == "primary"
    causal = draft.model_copy(update={"relation_type": "causally_upstream_of"})
    assert release_gate(phenotype, causal, evidence, mapping, review)[0] == "rejected"
    mouse = draft.model_copy(update={"species": "mouse", "taxon_id": "NCBITaxon:10090"})
    assert release_gate(phenotype, mouse, evidence, mapping, review)[0] == "cross_species"
    negated = review.model_copy(update={"abnormality_supported": False})
    assert release_gate(phenotype, draft, evidence, mapping, negated)[0] == "rejected"


def test_unmapped_and_indirect_are_not_primary(phenotype, draft, document, mapping, review):
    evidence = anchor_claim(draft, document, 0, len(document.text))
    assert release_gate(phenotype, draft, evidence, mapping.model_copy(update={"go": None}), review)[0] == "unresolved"
    bridge = draft.model_copy(update={"bridge_scope": "phenotype_mechanism_bridge"})
    assert release_gate(phenotype, bridge, evidence, mapping, review)[0] == "rejected"
    background = draft.model_copy(update={"assertion": "background_statement"})
    assert release_gate(phenotype, background, evidence, mapping, review)[0] == "conditional"


def test_hpo_input_rejects_mouse_phenotype_and_nonempty_model_context():
    with pytest.raises(ValidationError):
        Phenotype(phenotype_id="MP:0000003", label="mouse phenotype")
    with pytest.raises(ValidationError):
        Phenotype(phenotype_id="HP:0001250", label="human phenotype", mgi_contexts=[{"context_id": "model"}])
    assert "mgi_contexts" not in Phenotype.model_json_schema()["properties"]


def test_identifier_species_constraints(phenotype, draft):
    with pytest.raises(ValidationError):
        Phenotype(phenotype_id="MPO:0000003", label="invalid prefix")
    with pytest.raises(ValidationError):
        type(draft).model_validate({**draft.model_dump(), "taxon_id": "NCBITaxon:10090"})


def test_omitted_empty_metadata_does_not_invalidate_supported_claim(draft):
    row = draft.model_dump(exclude={"anatomy_context", "disease_context", "mgi_context_id", "model_evidence_quote"})
    parsed = Extraction.model_validate({"claims": [row]})
    assert parsed.claims[0].supporting_quote == draft.supporting_quote
    assert parsed.claims[0].anatomy_context == ""
    with pytest.raises(ValidationError, match="rejection_reason"):
        Extraction.model_validate({"claims": []})


def test_causal_edge_is_oriented_from_cell_abnormality_to_phenotype(phenotype, draft, document, mapping, review):
    def observation(direction, relation):
        claim = draft.model_copy(update={"direction": direction, "relation_type": relation})
        return Observation(
            claim_id="test", phenotype_id=phenotype.phenotype_id, phenotype_label=phenotype.label,
            claim=claim, evidence=anchor_claim(claim, document, 0, len(document.text)),
            evidence_mode=claim.evidence_mode,
            mapping=mapping, review=review, release_scope="primary", gate_reasons=[],
        )
    associated = observation("increased", "associated_with")
    causal = observation("increased", "causally_upstream_of")
    decreased = observation("decreased", "associated_with")
    assert associated.edge_source == phenotype.phenotype_id
    assert causal.edge_source == causal.cell_abnormality_id
    assert causal.edge_target == phenotype.phenotype_id
    assert associated.cell_abnormality_id == causal.cell_abnormality_id
    assert decreased.cell_abnormality_id != associated.cell_abnormality_id
    assert Observation.model_validate(causal.model_dump()) == causal
