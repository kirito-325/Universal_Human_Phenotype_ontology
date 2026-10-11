"""Synthetic fixtures exercise software behavior; they are not literature evidence."""

import pytest

from hpo_agent.ontology import OntologyIndex, build_index
from hpo_agent.schemas import ClaimDraft, Document, MappingChoice, Phenotype, Review
from hpo_agent.validation import resolve_mapping

CL_OBO = '''format-version: 1.2
[Term]
id: CL:0000540
name: neuron
namespace: cell
def: "An electrically responsive cell." []
synonym: "neurons" EXACT []
alt_id: CL:9000001

[Term]
id: CL:9000002
name: obsolete test cell
is_obsolete: true
replaced_by: CL:0000540
'''
GO_OBO = '''format-version: 1.2
[Term]
id: GO:0006915
name: apoptotic process
namespace: biological_process
def: "A programmed cell death process." []
synonym: "apoptosis" EXACT []

[Term]
id: GO:9000001
name: test molecular function
namespace: molecular_function
def: "Synthetic test only." []
'''


@pytest.fixture
def index(tmp_path):
    cl, go = tmp_path / "cl.obo", tmp_path / "go.obo"
    cl.write_text(CL_OBO)
    go.write_text(GO_OBO)
    db = tmp_path / "ontology.sqlite"
    build_index(cl, go, db)
    return OntologyIndex(db)


@pytest.fixture
def phenotype():
    return Phenotype(phenotype_id="HP:0001250", label="Seizure")


@pytest.fixture
def document(phenotype):
    return Document(
        document_id="synthetic:1", title="SYNTHETIC SOFTWARE TEST, NOT A PUBLICATION",
        text="Synthetic header.\n\nHuman patients with seizures had increased apoptosis in neurons.",
        source_url="urn:synthetic:software-test", source_kind="local_text",
        phenotype_ids=[phenotype.phenotype_id], pmid="123",
    )


@pytest.fixture
def draft():
    return ClaimDraft(
        evidence_mode="direct",
        supporting_quote="Human patients with seizures had increased apoptosis in neurons.",
        cell_text="neurons", process_text="apoptosis", quality_text="increased apoptosis",
        direction="increased", relation_type="associated_with", bridge_scope="direct_phenotype",
        species="human", taxon_id="NCBITaxon:9606", disease_context="", anatomy_context="",
        experimental_context="synthetic test", assertion="own_result", evidence_type="observation",
        gene_mentions_ignored=[], mgi_context_id="", model_evidence_quote="", rationale="Synthetic fixture.",
    )


@pytest.fixture
def mapping(index):
    return resolve_mapping(MappingChoice(
        cl_id="CL:0000540", go_id="GO:0006915", cell_confidence="high", process_confidence="high",
        rationale="Synthetic fixture mapping.",
    ), index, {"CL:0000540", "GO:0006915"})


@pytest.fixture
def review():
    return Review(
        verdict="supported", cell_supported=True, process_supported=True,
        abnormality_supported=True, phenotype_bridge_supported=True, species_supported=True,
        mapping_supported=True, model_match_supported=False, causal_supported=False,
        reason="Synthetic fixture review.",
    )
