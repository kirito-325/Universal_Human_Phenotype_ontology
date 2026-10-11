from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional

from .config import journal_family_map, journal_names, normalize_journal_name
from .http_client import HttpClient
from .utils import flatten_unique, normalize_space, slugify


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def pubmed_phrase(value: str, field: str = "Title/Abstract") -> str:
    escaped = value.replace('"', "")
    return f'"{escaped}"[{field}]'


def build_or_query(values: Iterable[str], field: str) -> str:
    return "(" + " OR ".join(pubmed_phrase(value, field) for value in values) + ")"


def build_pubmed_query(
    config: Dict[str, Any], start_date: str, end_date: str
) -> str:
    journal_query = build_or_query(journal_names(config), "Journal")
    single_cell_query = build_or_query(config.get("single_cell_terms", []), "Title/Abstract")
    immune_query = build_or_query(config.get("immune_terms", []), "Title/Abstract")
    date_query = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'

    exclude_terms = []
    for publication_type in config.get("exclude_publication_types", []):
        escaped = publication_type.replace('"', "")
        exclude_terms.append(f'"{escaped}"[Publication Type]')
    exclude_query = ""
    if exclude_terms:
        exclude_query = " NOT (" + " OR ".join(exclude_terms) + ")"

    return f"{journal_query} AND {single_cell_query} AND {immune_query} AND {date_query}{exclude_query}"


class PubMedClient:
    def __init__(self, http: HttpClient, api_key: str = "", tool: str = "marker-literature-pipeline") -> None:
        self.http = http
        self.api_key = api_key
        self.tool = tool

    def _common(self) -> Dict[str, str]:
        data = {"tool": self.tool}
        if self.http.email:
            data["email"] = self.http.email
        if self.api_key:
            data["api_key"] = self.api_key
        return data

    def search(self, term: str, retmax: int = 10000) -> List[str]:
        ids: List[str] = []
        retstart = 0
        count: Optional[int] = None
        while count is None or retstart < count:
            batch_size = min(10000, max(1, retmax - len(ids)))
            data = {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retstart": str(retstart),
                "retmax": str(batch_size),
                **self._common(),
            }
            response, _, _ = self.http.request_json(f"{EUTILS}/esearch.fcgi", data=data)
            result = response.get("esearchresult", {})
            count = int(result.get("count", "0"))
            batch = result.get("idlist", [])
            ids.extend(batch)
            if len(ids) >= retmax or not batch:
                break
            retstart += len(batch)
        return ids[:retmax]

    def fetch_xml(self, pmids: List[str]) -> ET.Element:
        data = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            **self._common(),
        }
        text, _, _ = self.http.request_text(f"{EUTILS}/efetch.fcgi", data=data)
        return ET.fromstring(text)

    def fetch_records(self, pmids: List[str], batch_size: int = 100) -> List[ET.Element]:
        records: List[ET.Element] = []
        for start in range(0, len(pmids), batch_size):
            root = self.fetch_xml(pmids[start : start + batch_size])
            records.extend(root.findall(".//PubmedArticle"))
        return records


def elem_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return normalize_space("".join(elem.itertext()))


def parse_article_ids(article: ET.Element) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    for item in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = (item.attrib.get("IdType") or "").lower()
        value = elem_text(item)
        if id_type and value:
            ids[id_type] = value
    return ids


def parse_date_from_parts(parent: Optional[ET.Element]) -> str:
    if parent is None:
        return ""
    year = elem_text(parent.find("Year"))
    month = elem_text(parent.find("Month"))
    day = elem_text(parent.find("Day"))
    medline = elem_text(parent.find("MedlineDate"))
    if not year and medline:
        match = re.search(r"(19|20)\d{2}", medline)
        if match:
            year = match.group(0)
    if not year:
        return ""
    if month:
        key = month[:4].lower() if month.lower().startswith("sept") else month[:3].lower()
        month = MONTHS.get(key, month)
        if month.isdigit():
            month = month.zfill(2)
        else:
            month = "01"
    else:
        month = "01"
    day = day.zfill(2) if day.isdigit() else "01"
    return f"{year}-{month}-{day}"


def parse_publication_date(article: ET.Element) -> str:
    article_date = article.find(".//Article/ArticleDate")
    parsed = parse_date_from_parts(article_date)
    if parsed:
        return parsed
    return parse_date_from_parts(article.find(".//JournalIssue/PubDate"))


def parse_authors(article: ET.Element) -> List[str]:
    authors = []
    for author in article.findall(".//AuthorList/Author"):
        collective = elem_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = elem_text(author.find("LastName"))
        fore = elem_text(author.find("ForeName")) or elem_text(author.find("Initials"))
        name = normalize_space(f"{fore} {last}")
        if name:
            authors.append(name)
    return authors


def parse_abstract(article: ET.Element) -> str:
    parts = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label") or abstract_text.attrib.get("NlmCategory")
        text = elem_text(abstract_text)
        if text and label:
            parts.append(f"{label}: {text}")
        elif text:
            parts.append(text)
    return normalize_space(" ".join(parts))


def parse_keywords(article: ET.Element) -> List[str]:
    terms = []
    for keyword in article.findall(".//KeywordList/Keyword"):
        terms.append(elem_text(keyword))
    for mesh in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
        terms.append(elem_text(mesh))
    return flatten_unique(terms)


def matched_terms(text: str, terms: Iterable[str]) -> List[str]:
    lower_text = text.lower()
    return [term for term in terms if term.lower() in lower_text]


def parse_pubmed_article(article: ET.Element, config: Dict[str, Any]) -> Dict[str, Any]:
    pmid = elem_text(article.find(".//PMID"))
    ids = parse_article_ids(article)
    doi = ids.get("doi", "")
    pmcid = ids.get("pmc", "")
    title = elem_text(article.find(".//ArticleTitle"))
    abstract = parse_abstract(article)
    journal = elem_text(article.find(".//Journal/Title"))
    publication_date = parse_publication_date(article)
    publication_year = None
    if publication_date[:4].isdigit():
        publication_year = int(publication_date[:4])
    article_types = [
        elem_text(item) for item in article.findall(".//PublicationTypeList/PublicationType")
    ]
    keywords = parse_keywords(article)
    text_for_screening = " ".join([title, abstract, " ".join(keywords)])
    families = journal_family_map(config)
    source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    if doi:
        source_url = f"https://doi.org/{doi}"

    return {
        "id": f"doi:{doi}" if doi else f"pmid:{pmid}",
        "id_type": "doi" if doi else "pmid",
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title,
        "slug": slugify(title, 72),
        "journal": journal,
        "publisher": "",
        "publication_date": publication_date,
        "publication_year": publication_year,
        "article_type": "; ".join(article_types) or "research-article",
        "cns_family": families.get(normalize_journal_name(journal), ""),
        "url": source_url,
        "pdf_url": "",
        "source": {
            "discovery": ["pubmed"],
            "metadata": ["pubmed-efetch"],
            "publisher": "",
        },
        "license": "",
        "is_open_access": None,
        "download": {
            "paper_pdf": {
                "local_path": "paper.pdf",
                "sha256": "",
                "downloaded_at": "",
                "status": "not_downloaded",
            }
        },
        "authors": parse_authors(article),
        "abstract": abstract,
        "keywords": keywords,
        "screening": {
            "query_matched": flatten_unique(
                matched_terms(text_for_screening, config.get("single_cell_terms", []))
                + matched_terms(text_for_screening, config.get("immune_terms", []))
            ),
            "single_cell_methods": matched_terms(
                text_for_screening, config.get("single_cell_terms", [])
            ),
            "immune_cell_types": matched_terms(
                text_for_screening, config.get("immune_terms", [])
            ),
            "disease_context": [],
            "dataset_accessions": extract_accessions(text_for_screening),
            "relevance_status": "candidate",
            "notes": "",
        },
        "assets": [],
        "created_at": "",
        "updated_at": "",
        "verified_status": "pending",
    }


def extract_accessions(text: str) -> List[str]:
    patterns = [
        r"\bGSE\d+\b",
        r"\bGSM\d+\b",
        r"\bSRP\d+\b",
        r"\bSRA\d+\b",
        r"\bE-MTAB-\d+\b",
        r"\bEGAS\d+\b",
        r"\bPRJNA\d+\b",
        r"\bPRJEB\d+\b",
        r"\bHRA\d+\b",
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return flatten_unique(values)
