"""The retained upstream tools run against stubbed HTTP responses, not a new retriever."""

import asyncio

import httpx
import pytest

from mpo_agent.literature import LiteratureClient, SourceError
from mpo_agent.upstream.sources import pmc, pubmed

XML = """<PubmedArticleSet><PubmedArticle>
<MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Test</ArticleTitle>
<Abstract><AbstractText Label="RESULTS">First β sentence.</AbstractText>
<AbstractText>Second sentence.</AbstractText></Abstract></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC123</ArticleId>
<ArticleId IdType="doi">10.test/own</ArticleId></ArticleIdList>
<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.test/cited</ArticleId>
</ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle></PubmedArticleSet>"""


@pytest.fixture
def ncbi(monkeypatch):
    calls = []

    async def request(util, **params):
        calls.append((util, params))
        if util == "esearch":
            return httpx.Response(200, json={"esearchresult": {
                "count": "1", "idlist": ["123"], "querytranslation": "translated",
            }})
        if util == "esummary":
            return httpx.Response(200, json={"result": {"uids": ["123"], "123": {
                "title": "Test", "articleids": [{"idtype": "pmc", "value": "PMC123"}],
            }}})
        return httpx.Response(200, text=XML)

    monkeypatch.setattr(pubmed, "_request", request)
    return calls


async def test_upstream_abstract_ids_sections_and_cache(tmp_path, ncbi):
    async with LiteratureClient(tmp_path) as source:
        doc = (await source.fetch_abstracts(["123"]))[0]
        assert doc.doi == "10.test/own" and doc.pmcid == "PMC123"
        assert "First β sentence.\n\nSecond sentence." in doc.text
        assert doc.text[doc.sections[1].start:doc.sections[1].end] == "Second sentence."
        assert (await source.fetch_abstracts(["123"]))[0] == doc
        assert len(ncbi) == 1
        with pytest.raises(ValueError):
            await source.fetch_abstracts(["123.5"])
    assert (tmp_path / "abstracts" / "123.json").is_file()


async def test_upstream_search_supports_count_probes_and_unknown_tag_warnings(tmp_path, ncbi):
    async with LiteratureClient(tmp_path) as source:
        probe = await source.search("test[tiab]", 0)
        assert probe["count"] == 1 and not probe["pmids"]
        assert [u for u, _ in ncbi] == ["esearch"]
        assert ncbi[0][1]["term"] == "(test[tiab]) AND hasabstract"
        # The constraint must cover both OR branches, even with a bad field tag.
        query = "test[fakefield] OR alternative[tiab]"
        result = await source.search(query, 1)
        assert result["pmids"] == ["123"]
        assert result["warnings"] and result["query_translation"] == "translated"
        assert [u for u, _ in ncbi] == ["esearch", "esearch", "esummary"]
        assert ncbi[1][1]["term"] == "(test[fakefield] OR alternative[tiab]) AND hasabstract"
        assert result["query"] == query
        assert result["effective_query"] == ncbi[1][1]["term"]
        assert result["filters"] == ["hasabstract"]


async def test_context_local_caches_do_not_cross_batch_directories(tmp_path, ncbi):
    a, b = LiteratureClient(tmp_path / "a"), LiteratureClient(tmp_path / "b")
    await asyncio.gather(a.fetch_abstracts(["123"]), b.fetch_abstracts(["123"]))
    assert (tmp_path / "a" / "abstracts" / "123.json").is_file()
    assert (tmp_path / "b" / "abstracts" / "123.json").is_file()


async def test_upstream_pmc_parsing_is_used_by_adapter(tmp_path, ncbi, monkeypatch):
    raw = b"<article><body><sec><title>Results</title><p>First.</p><p>Second.</p></sec></body></article>"

    async def load(client, pmcid):
        return {"objects": {}}, {"pmid": 123, "license_code": "CC0"}, pmc.parse_jats(raw, {})

    monkeypatch.setattr(pmc, "_load", load)
    async with LiteratureClient(tmp_path) as source:
        abstract = (await source.fetch_abstracts(["123"]))[0]
        doc = await source.full_text(abstract)
    assert doc.source_kind == "full_text"
    assert "First.\n\nSecond." in doc.text
    assert doc.sections[0].name == "Results"


async def test_pmc_missing_body_preserves_abstract(tmp_path, ncbi, monkeypatch):
    async def load(client, pmcid):
        return None

    monkeypatch.setattr(pmc, "_load", load)
    async with LiteratureClient(tmp_path) as source:
        abstract = (await source.fetch_abstracts(["123"]))[0]
        result = await source.full_text(abstract)
    assert result.source_kind == "abstract"
    assert result.retrieval_notes == ["full_text_unavailable:PMC_open_data"]


async def test_upstream_fatal_search_error_is_not_no_hits(tmp_path, monkeypatch):
    async def request(util, **params):
        return httpx.Response(200, json={"error": "temporarily unavailable"})

    monkeypatch.setattr(pubmed, "_request", request)
    async with LiteratureClient(tmp_path) as source:
        with pytest.raises(SourceError):
            await source.search("test", 1)
