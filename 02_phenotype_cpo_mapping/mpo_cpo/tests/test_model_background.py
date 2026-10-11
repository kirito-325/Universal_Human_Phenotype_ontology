import pytest
from pydantic import ValidationError

from mpo_agent.model_background import MPContextIndex, build_mp_context
from mpo_agent.schemas import MPInput


@pytest.fixture
def model_index(tmp_path):
    mp, rows = tmp_path / "mp.obo", tmp_path / "mgi.rpt"
    mp.write_text('''format-version: 1.2
[Term]
id: MP:9990001
name: Synthetic mouse target
def: "Synthetic phenotype for software testing." []
alt_id: MP:9990009

[Term]
id: MP:9990002
name: Synthetic descendant
is_a: MP:9990001 ! Synthetic mouse target

[Term]
id: MP:9990003
name: obsolete synthetic term
is_obsolete: true
''')
    rows.write_text(
        "Gene<tm1>/Gene<tm1>\tGene<tm1>\tB6\tMP:9990001\t101|102\tMGI:100\tMGI:1\n"
        "Gene<tm1>/Gene<tm1>\tGene<tm1>\tB6\tMP:9990001\t102|103\tMGI:100\tMGI:1\n"
        "Gene<tm2>/Gene<tm2>\tGene<tm2>\t129\tMP:9990001\t201\tMGI:100\tMGI:2\n"
        "Gene<tm1>/Gene<tm1>\tGene<tm1>\tB6\tMP:9990002\t104\tMGI:100\tMGI:1\n"
        "Other<tm1>/Other<tm1>\tOther<tm1>\tB6\tMP:9990002\t301\tMGI:200\tMGI:3\n")
    output = tmp_path / "models.sqlite"
    build_mp_context(mp, rows, output)
    return MPContextIndex(output)


def test_background_preserves_genotypes_and_annotation_specific_references(model_index):
    page = model_index.lookup("MP:9990009", limit=1)
    assert page["term"]["term_id"] == "MP:9990001" and page["total_models"] == 2
    model, = page["models"]
    assert model["model_id"] == "MGI:1" and model["genetic_background"] == "B6"
    assert model["annotation_pmids"] == ["101", "102", "103"]
    assert page["next_offset"] == 1
    second = model_index.lookup("MP:9990001", offset=1)
    assert second["models"][0]["model_id"] == "MGI:2"
    assert second["models"][0]["marker_ids"] == model["marker_ids"]
    assert all(row["match_scope"] == "exact" for row in model["annotations"])


def test_descendants_are_opt_in_and_provenance_is_not_silently_inherited(model_index):
    exact = model_index.lookup("MP:9990001")
    broader = model_index.lookup("MP:9990001", include_descendants=True)
    assert exact["total_models"] == 2 and broader["total_models"] == 3
    first = broader["models"][0]
    assert first["annotation_pmids"] == ["101", "102", "103", "104"]
    assert {row["match_scope"] for row in first["annotations"]} == {"exact", "descendant"}
    assert broader["index_sha256"] == model_index.sha256


def test_invalid_or_obsolete_terms_and_human_inputs_are_rejected(model_index):
    for term in ("HP:0001250", "MP:9999999", "MP:9990003"):
        with pytest.raises(ValueError):
            model_index.lookup(term)
    with pytest.raises(ValidationError):
        MPInput(phenotype_id="HP:0001250")


def test_conflicting_genotype_metadata_never_overwrites_a_model(tmp_path):
    mp, rows, output = tmp_path / "mp.obo", tmp_path / "rows.rpt", tmp_path / "bad.sqlite"
    mp.write_text("[Term]\nid: MP:9990001\nname: synthetic\n")
    rows.write_text("a/a\ta\tB6\tMP:9990001\t101\tMGI:100\tMGI:1\n"
                    "b/b\tb\t129\tMP:9990001\t102\tMGI:100\tMGI:1\n")
    with pytest.raises(ValueError, match="Conflicting genotype"):
        build_mp_context(mp, rows, output)
    assert not output.exists()


def test_missing_index_sources_leave_no_partial_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_mp_context(tmp_path / "missing.obo", tmp_path / "missing.rpt", tmp_path / "failed.sqlite")
    assert not list(tmp_path.glob(".mp-context-*"))
