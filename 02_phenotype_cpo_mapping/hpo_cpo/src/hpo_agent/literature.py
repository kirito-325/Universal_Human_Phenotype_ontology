"""Adapter around the retained deep-life-sci PubMed/PMC tools.

Retrieval, batching, rate limiting, S3 version discovery and XML parsing remain in
upstream/sources/. This adapter restricts searches to records with abstracts and
converts upstream records into our evidence schema.
"""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import Document, Section
from .upstream.paths import cache_scope
from .upstream.sources import pmc, pubmed
from .upstream.sources._errors import SourceError as SourceError


class LiteratureClient:
    def __init__(self, cache: Path):
        self.cache = cache

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass  # Upstream owns each HTTP client's lifetime.

    async def search(self, query: str, limit: int) -> dict:
        if not query.strip() or not 0 <= limit <= 100:
            raise ValueError("query required; limit must be 0..100")
        # Group the entire original query so OR branches cannot escape the filter.
        effective_query = f"({query.strip()}) AND hasabstract"
        with cache_scope(self.cache):
            result = await pubmed.pubmed_search.ainvoke({"term": effective_query, "retmax": limit})
        return {**result, "query": query, "effective_query": effective_query,
                "filters": ["hasabstract"], "pmids": [r["pmid"] for r in result["records"]]}

    async def fetch_abstracts(self, pmids: list[str]) -> list[Document]:
        if any(not re.fullmatch(r"[0-9]+", p) for p in pmids):
            raise ValueError("PMIDs must contain digits only")
        with cache_scope(self.cache):
            result = await pubmed.fetch_abstracts.ainvoke({"pmids": pmids})
        documents = []
        for pmid, record in result["records"].items():
            if pmid not in pmids:
                raise ValueError("unrequested PMID returned")
            text = record.get("abstract")
            if not text:
                continue
            sections = []
            cursor = 0
            for section in record.get("sections", []):
                fragment = section.get("text", "")
                start = text.find(fragment, cursor) if fragment else -1
                if start >= 0:
                    sections.append(Section(name=section.get("label") or "abstract",
                                            start=start, end=start + len(fragment)))
                    cursor = start + len(fragment)
            documents.append(Document(
                document_id="PMID:" + pmid, title=record.get("title") or "", text=text,
                pmid=pmid, pmcid=record.get("pmcid") or "", doi=record.get("doi") or "",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", source_kind="abstract",
                sections=sections, retracted=record.get("retracted", False),
            ))
        return documents

    async def full_text(self, abstract: Document) -> Document:
        if not abstract.pmcid:
            return abstract
        with cache_scope(self.cache):
            result = await pmc.fetch_full_text.ainvoke({"pmcids": [abstract.pmcid]})
        record = result["records"].get(abstract.pmcid)
        if not record or not record.get("text"):
            return abstract.model_copy(update={
                "retrieval_notes": [*abstract.retrieval_notes, "full_text_unavailable:PMC_open_data"],
            })
        if record.get("pmid") and str(record["pmid"]) != abstract.pmid:
            raise ValueError("PMC full text PMID mismatch")
        text = record["text"]
        starts = list(re.finditer(r"(?m)^## (.+)$", text))
        sections = [Section(name=m[1], start=m.start(),
                            end=starts[i + 1].start() if i + 1 < len(starts) else len(text))
                    for i, m in enumerate(starts)]
        return abstract.model_copy(update={
            "document_id": abstract.pmcid, "text": text, "source_kind": "full_text",
            "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{abstract.pmcid}/",
            "sections": sections, "retracted": abstract.retracted or record.get("retracted", False),
            "retrieval_notes": [*abstract.retrieval_notes, "source:deep-life-sci/PMC_open_data",
                                "license:" + str(record.get("license", ""))],
        })
