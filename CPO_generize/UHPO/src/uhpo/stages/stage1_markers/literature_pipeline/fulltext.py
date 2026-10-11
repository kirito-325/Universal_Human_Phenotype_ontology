from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .assets import (
    fetch_pmc_jats,
    load_manifest,
    normalize_elsevier_pii,
    save_article_text,
    write_manifest,
)
from .http_client import HttpClient
from .utils import read_json, sha256_file, utc_now, write_json


OPENALEX_WORKS_BASE = "https://api.openalex.org/works"
ELSEVIER_ARTICLE_BASE = "https://api.elsevier.com/content/article/doi"
MIN_ARTICLE_CHARACTERS = 2000
PDF_HINTS = (".pdf", "/bitstream/", "/retrieve/", "/download")
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}


class RepositoryLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.urls: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            name = (values.get("name") or values.get("property") or "").lower()
            if name in {
                "citation_pdf_url",
                "dc.identifier",
                "eprints.document_url",
            }:
                self._add(values.get("content", ""))
        elif tag.lower() == "a":
            href = values.get("href", "")
            link_type = values.get("type", "").lower()
            if "pdf" in link_type or looks_like_pdf_url(href):
                self._add(href)

    def _add(self, url: str) -> None:
        if not url:
            return
        self.urls.append(urllib.parse.urljoin(self.base_url, html.unescape(url)))


def article_text_path(record_dir: Path) -> Path:
    return record_dir / "extracted" / "text" / "article_text.txt"


def has_article_text(record_dir: Path, minimum_characters: int = MIN_ARTICLE_CHARACTERS) -> bool:
    path = article_text_path(record_dir)
    if not path.exists():
        return False
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").strip()) >= minimum_characters
    except OSError:
        return False


def has_extracted_tables(record_dir: Path) -> bool:
    tables_dir = record_dir / "extracted" / "tables"
    if not tables_dir.exists():
        return False
    return any(path.is_file() for path in tables_dir.glob("*.csv"))


def looks_like_pdf_url(url: str) -> bool:
    lowered = urllib.parse.unquote(url or "").lower()
    return any(hint in lowered for hint in PDF_HINTS)


def openalex_work_url(doi: str) -> str:
    normalized = doi.strip()
    if normalized.lower().startswith("https://doi.org/"):
        normalized = normalized[16:]
    return f"{OPENALEX_WORKS_BASE}/https://doi.org/{urllib.parse.quote(normalized, safe='/')}"


def unique_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        url = html.unescape(str(candidate.get("url", "")).strip())
        if not url or url in seen:
            continue
        seen.add(url)
        copied = dict(candidate)
        copied["url"] = url
        output.append(copied)
    return output


def direct_openalex_candidates(work: Dict[str, Any]) -> List[Dict[str, Any]]:
    locations: List[Dict[str, Any]] = []
    for key in ("best_oa_location", "primary_location"):
        location = work.get(key)
        if isinstance(location, dict):
            locations.append(location)
    locations.extend(
        location for location in work.get("locations", []) if isinstance(location, dict)
    )

    candidates: List[Dict[str, Any]] = []
    for location in locations:
        if location.get("is_oa") is not True:
            continue
        source = location.get("source") or {}
        common = {
            "source": source.get("display_name", "") or "OpenAlex OA location",
            "source_type": source.get("type", ""),
            "license": location.get("license", "") or "",
            "version": location.get("version", "") or "",
            "location_id": location.get("id", "") or "",
            "discovered_by": "openalex",
        }
        pdf_url = location.get("pdf_url", "") or ""
        if pdf_url:
            candidates.append({**common, "url": pdf_url, "kind": "pdf"})
        landing_url = location.get("landing_page_url", "") or ""
        if landing_url and looks_like_pdf_url(landing_url):
            candidates.append({**common, "url": landing_url, "kind": "pdf"})
    return unique_candidates(candidates)


def repository_locations(work: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for location in work.get("locations", []):
        if not isinstance(location, dict) or location.get("is_oa") is not True:
            continue
        source = location.get("source") or {}
        if source.get("type") == "repository":
            output.append(location)
    return output


def oai_endpoint_for_location(location: Dict[str, Any]) -> str:
    source = location.get("source") or {}
    name = str(source.get("display_name", "")).lower()
    landing_url = str(location.get("landing_page_url", "")).lower()
    if "lirias" in name or "lirias.kuleuven.be" in landing_url:
        return "https://lirias.kuleuven.be/oai/request"
    if "leiden repository" in name or "universiteitleiden.nl" in landing_url:
        return "https://scholarlypublications.universiteitleiden.nl/oai2"
    return ""


def oai_identifier(location: Dict[str, Any]) -> str:
    location_id = str(location.get("id", ""))
    return location_id[4:] if location_id.startswith("pmh:") else ""


def parse_oai_candidates(xml_text: str, location: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    source = location.get("source") or {}
    identifiers: List[str] = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1].lower() != "identifier":
            continue
        value = (elem.text or "").strip()
        if value.startswith(("http://", "https://")) and looks_like_pdf_url(value):
            identifiers.append(value)
    return unique_candidates(
        {
            "url": url,
            "kind": "pdf",
            "source": source.get("display_name", "") or "OAI repository",
            "source_type": "repository",
            "license": location.get("license", "") or "",
            "version": location.get("version", "") or "",
            "location_id": location.get("id", "") or "",
            "discovered_by": "repository_oai",
        }
        for url in identifiers
    )


def refreshed_oai_candidates(
    http: HttpClient,
    work: Dict[str, Any],
    errors: List[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for location in repository_locations(work):
        endpoint = oai_endpoint_for_location(location)
        identifier = oai_identifier(location)
        if not endpoint or not identifier:
            continue
        query = urllib.parse.urlencode(
            {
                "verb": "GetRecord",
                "metadataPrefix": "oai_dc",
                "identifier": identifier,
            }
        )
        try:
            xml_text, _, _ = http.request_text(f"{endpoint}?{query}")
            candidates.extend(parse_oai_candidates(xml_text, location))
        except Exception as exc:  # noqa: BLE001 - one repository must not stop recovery.
            errors.append(
                f"repository_oai {source_name(location)}: {type(exc).__name__}: {exc}"
            )
    return unique_candidates(candidates)


def source_name(location: Dict[str, Any]) -> str:
    source = location.get("source") or {}
    return str(source.get("display_name", "") or location.get("landing_page_url", ""))


def repository_page_candidates(
    http: HttpClient,
    work: Dict[str, Any],
    errors: List[str],
    maximum_pages: int = 4,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    visited = set()
    for location in repository_locations(work):
        url = str(location.get("landing_page_url", ""))
        if not url or url in visited or looks_like_pdf_url(url):
            continue
        visited.add(url)
        if len(visited) > maximum_pages:
            break
        try:
            page, final_url, content_type = http.request_text(url)
            if "html" not in content_type.lower() and "<html" not in page[:1000].lower():
                continue
            parser = RepositoryLinkParser(final_url)
            parser.feed(page)
            source = location.get("source") or {}
            candidates.extend(
                {
                    "url": candidate_url,
                    "kind": "pdf",
                    "source": source.get("display_name", "") or "OA repository",
                    "source_type": "repository",
                    "license": location.get("license", "") or "",
                    "version": location.get("version", "") or "",
                    "location_id": location.get("id", "") or "",
                    "discovered_by": "repository_page",
                }
                for candidate_url in parser.urls
            )
        except Exception as exc:  # noqa: BLE001 - inaccessible repositories are expected.
            errors.append(
                f"repository_page {source_name(location)}: {type(exc).__name__}: {exc}"
            )
    return unique_candidates(candidates)


def title_tokens(value: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", value.lower())
    return [token for token in tokens if len(token) >= 4 and token not in TITLE_STOPWORDS]


def title_match_score(title: str, text: str) -> float:
    expected = set(title_tokens(title))
    if not expected:
        return 1.0
    observed = set(title_tokens(text[:50000]))
    return len(expected & observed) / len(expected)


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def xml_local_name(elem: ET.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1].lower()


def xml_element_text(elem: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def xml_values(root: ET.Element, local_names: set[str]) -> List[str]:
    values = []
    seen = set()
    for elem in root.iter():
        if xml_local_name(elem) not in local_names:
            continue
        value = xml_element_text(elem)
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def fetch_elsevier_article_xml(
    http: HttpClient,
    doi: str,
    api_key: str,
    inst_token: str = "",
    require_body: bool = True,
) -> Tuple[str, str]:
    url = f"{ELSEVIER_ARTICLE_BASE}/{urllib.parse.quote(doi, safe='/')}?httpAccept=text/xml"
    headers = {"Accept": "text/xml", "X-ELS-APIKey": api_key}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token
    xml_text, final_url, _ = http.request_text(url, headers=headers)
    root = ET.fromstring(xml_text)
    status_text = xml_values(root, {"statustext"})
    if status_text:
        raise PermissionError(status_text[0])
    if require_body and not any(xml_local_name(elem) == "body" for elem in root.iter()):
        raise ValueError("Elsevier response did not include the article body")
    return xml_text, final_url


def save_elsevier_article_text(
    record_dir: Path,
    metadata: Dict[str, Any],
    article_xml: str,
    source_url: str,
) -> Dict[str, Any]:
    root = ET.fromstring(article_xml)
    abstracts = xml_values(root, {"abstract"})
    bodies = [elem for elem in root.iter() if xml_local_name(elem) == "body"]
    if not bodies:
        raise ValueError("Elsevier XML has no body element")
    body = max(bodies, key=lambda elem: len(xml_element_text(elem)))
    body_parts = []
    for elem in body.iter():
        if xml_local_name(elem) not in {"section-title", "para"}:
            continue
        value = xml_element_text(elem)
        if value:
            body_parts.append(value)
    body_text = "\n\n".join(body_parts)
    if len(body_text) < MIN_ARTICLE_CHARACTERS:
        body_text = xml_element_text(body)

    title = str(metadata.get("title", "")).strip()
    text = normalize_pdf_text("\n\n".join([title, *abstracts, body_text]))
    if len(text) < MIN_ARTICLE_CHARACTERS:
        raise ValueError(f"Elsevier XML yielded only {len(text)} text characters")
    match_score = title_match_score(title, text)
    if match_score < 0.35:
        raise ValueError(f"Elsevier XML title match score too low ({match_score:.2f})")

    text_dir = record_dir / "extracted" / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    xml_path = text_dir / "elsevier_article.xml"
    text_path = article_text_path(record_dir)
    xml_path.write_text(article_xml, encoding="utf-8")
    text_path.write_text(text + "\n", encoding="utf-8")

    licenses = xml_values(root, {"oa-user-license", "sa-user-license"})
    access_values = xml_values(root, {"oa-article-status", "open-access"})
    return {
        "status": "recovered_elsevier_xml",
        "source": "Elsevier Article Retrieval API",
        "source_type": "publisher_api",
        "source_url": source_url,
        "license": licenses[0] if licenses else "",
        "open_access_status": access_values[0] if access_values else "",
        "version": "publishedVersion",
        "discovered_by": "elsevier_article_retrieval_api",
        "xml_path": str(xml_path.relative_to(record_dir)),
        "xml_sha256": sha256_file(xml_path),
        "text_path": str(text_path.relative_to(record_dir)),
        "text_characters": len(text),
        "text_sha256": sha256_file(text_path),
        "title_match_score": round(match_score, 4),
        "recovered_at": utc_now(),
    }


def parse_elsevier_supplementary_assets(article_xml: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(article_xml)
    captions = {}
    for elem in root.iter():
        if xml_local_name(elem) != "e-component":
            continue
        ref = str(elem.attrib.get("id", "")).lower()
        if ref.startswith("mmc"):
            captions[ref] = xml_element_text(elem)

    assets = []
    for elem in root.iter():
        if xml_local_name(elem) != "object":
            continue
        ref = str(elem.attrib.get("ref", "")).lower()
        if not ref.startswith("mmc") or elem.attrib.get("type", "").upper() != "APPLICATION":
            continue
        url = (elem.text or "").strip()
        if not url:
            continue
        object_path_name = Path(urllib.parse.urlparse(url).path).name
        pii = normalize_elsevier_pii(url)
        extension = Path(object_path_name).suffix.lower()
        if not pii or not extension:
            continue
        object_name = f"{ref}{extension}"
        filename = f"elsevier_{pii}_{object_name}"
        content_length = str(elem.attrib.get("size", ""))
        assets.append(
            {
                "filename": filename,
                "url": url,
                "type": "supplementary_table"
                if extension in {".xlsx", ".xls", ".csv", ".tsv", ".txt", ".docx"}
                else "supplementary_material",
                "label": ref,
                "caption": captions.get(ref, f"Elsevier supplementary material {ref}"),
                "source": "elsevier_article_api",
                "api_auth": "elsevier",
                "local_path": "",
                "sha256": "",
                "downloaded_at": "",
                "download_status": "not_downloaded",
                "parse_status": "pending",
                "content_type": elem.attrib.get("mimetype", ""),
                "content_length": int(content_length) if content_length.isdigit() else 0,
                "multimedia_type": elem.attrib.get("multimediatype", ""),
                "pii": pii,
                "object_ref": ref,
            }
        )
    return unique_candidates(assets)


def register_elsevier_supplementary_assets(
    record_dir: Path,
    article_xml: str,
) -> Dict[str, int]:
    existing = load_manifest(record_dir)
    discovered = parse_elsevier_supplementary_assets(article_xml)
    new_assets = 0
    updated_assets = 0
    already_local = 0

    for incoming in discovered:
        incoming_ref = incoming["object_ref"]
        incoming_extension = Path(incoming["filename"]).suffix.lower()
        candidates = []
        for current in existing:
            same_filename = current.get("filename", "") == incoming["filename"]
            current_text = " ".join(
                str(current.get(key, ""))
                for key in (
                    "object_ref",
                    "label",
                    "filename",
                    "url",
                    "url_final",
                    "url_elsevier_cdn",
                    "original_url",
                )
            )
            ref_match = re.search(
                r"(?:mmc|supplement(?:ary)?)[-_ ]*0*(\d+)",
                current_text,
                flags=re.IGNORECASE,
            )
            current_ref = f"mmc{int(ref_match.group(1))}" if ref_match else ""
            current_extension = Path(str(current.get("filename", ""))).suffix.lower()
            same_object = current_ref == incoming_ref and current_extension == incoming_extension
            if same_filename or same_object:
                candidates.append(current)

        if not candidates and incoming.get("content_length"):
            size_matches = []
            for current in existing:
                current_extension = Path(str(current.get("filename", ""))).suffix.lower()
                local_path = record_dir / str(current.get("local_path", ""))
                if (
                    current_extension == incoming_extension
                    and current.get("local_path")
                    and local_path.is_file()
                    and local_path.stat().st_size == incoming["content_length"]
                ):
                    size_matches.append(current)
            if len(size_matches) == 1:
                candidates = size_matches

        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("local_path")
                and (record_dir / str(candidate["local_path"])).is_file()
            ),
            candidates[0] if candidates else None,
        )
        if match is None:
            existing.append(incoming)
            new_assets += 1
            continue

        previous = {
            key: match.get(key)
            for key in (
                "url",
                "caption",
                "source",
                "api_auth",
                "content_type",
                "content_length",
            )
        }
        if match.get("url") and match.get("url") != incoming["url"]:
            match.setdefault("original_url", match["url"])
        for key in (
            "url",
            "type",
            "label",
            "caption",
            "source",
            "api_auth",
            "content_type",
            "content_length",
            "multimedia_type",
            "pii",
            "object_ref",
        ):
            match[key] = incoming[key]
        local_path = record_dir / str(match.get("local_path", ""))
        if match.get("local_path") and local_path.is_file():
            already_local += 1
        else:
            match["local_path"] = ""
            match["sha256"] = ""
            match["download_status"] = "not_downloaded"
            match["parse_status"] = "pending"
            match.pop("download_error", None)
        current = {key: match.get(key) for key in previous}
        if current != previous:
            updated_assets += 1

        for duplicate in candidates:
            if duplicate is not match and duplicate in existing:
                existing.remove(duplicate)

    write_manifest(record_dir, existing)
    return {
        "discovered": len(discovered),
        "new": new_assets,
        "updated": updated_assets,
        "already_local": already_local,
        "manifest_assets": len(existing),
    }


def sync_elsevier_content(
    record_dir: Path,
    metadata: Dict[str, Any],
    http: HttpClient,
    api_key: str,
    inst_token: str = "",
    refresh_xml: bool = False,
    overwrite_text: bool = False,
) -> Dict[str, Any]:
    doi = str(metadata.get("doi", "")).strip()
    if not doi.lower().startswith("10.1016/"):
        raise ValueError("sync_elsevier_content only supports 10.1016 DOI records")

    text_dir = record_dir / "extracted" / "text"
    xml_path = text_dir / "elsevier_article.xml"
    source_url = (
        f"{ELSEVIER_ARTICLE_BASE}/{urllib.parse.quote(doi, safe='/')}?httpAccept=text/xml"
    )
    fetched = False
    if xml_path.is_file() and not refresh_xml:
        article_xml = xml_path.read_text(encoding="utf-8", errors="replace")
    else:
        article_xml, source_url = fetch_elsevier_article_xml(
            http=http,
            doi=doi,
            api_key=api_key,
            inst_token=inst_token,
            require_body=False,
        )
        text_dir.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(article_xml, encoding="utf-8")
        fetched = True

    root = ET.fromstring(article_xml)
    has_body = any(xml_local_name(elem) == "body" for elem in root.iter())
    text_result: Dict[str, Any] = {}
    if has_body and (overwrite_text or not has_article_text(record_dir)):
        text_result = save_elsevier_article_text(
            record_dir=record_dir,
            metadata=metadata,
            article_xml=article_xml,
            source_url=source_url,
        )
        write_json(text_dir / "article_text_source.json", text_result)

    attachment_result = register_elsevier_supplementary_assets(record_dir, article_xml)
    return {
        "status": "ok",
        "doi": doi,
        "xml_fetched": fetched,
        "xml_path": str(xml_path.relative_to(record_dir)),
        "has_body": has_body,
        "text_status": text_result.get("status", "existing" if has_article_text(record_dir) else "no_body"),
        "text_result": text_result,
        "attachments": attachment_result,
        "synced_at": utc_now(),
    }


def pdf_to_text(pdf_path: Path, output_path: Path) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("pdftotext is required to extract downloaded PDFs")
    completed = subprocess.run(
        [executable, "-enc", "UTF-8", "-layout", str(pdf_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"pdftotext failed: {message[:500]}")
    return normalize_pdf_text(output_path.read_text(encoding="utf-8", errors="replace"))


def try_pdf_candidate(
    record_dir: Path,
    metadata: Dict[str, Any],
    http: HttpClient,
    candidate: Dict[str, Any],
    keep_pdf: bool,
) -> Dict[str, Any]:
    body, final_url, content_type = http.request_bytes(
        candidate["url"],
        headers={"Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1"},
    )
    pdf_start = body.find(b"%PDF-")
    if pdf_start < 0 or pdf_start > 1024:
        raise ValueError(f"response is not a PDF ({content_type or 'unknown content type'})")
    body = body[pdf_start:]

    text_dir = record_dir / "extracted" / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = text_dir / "article.pdf"
    temporary_text: Optional[Path] = None
    try:
        pdf_path.write_bytes(body)
        with tempfile.NamedTemporaryFile(
            prefix="article_text_",
            suffix=".txt",
            dir=text_dir,
            delete=False,
        ) as handle:
            temporary_text = Path(handle.name)
        text = pdf_to_text(pdf_path, temporary_text)
        if len(text) < MIN_ARTICLE_CHARACTERS:
            raise ValueError(f"PDF yielded only {len(text)} text characters")
        match_score = title_match_score(metadata.get("title", ""), text)
        if match_score < 0.35:
            raise ValueError(f"PDF title match score too low ({match_score:.2f})")
        output_path = article_text_path(record_dir)
        output_path.write_text(text + "\n", encoding="utf-8")
        result = {
            "status": "recovered_pdf",
            "source": candidate.get("source", ""),
            "source_type": candidate.get("source_type", ""),
            "source_url": candidate.get("url", ""),
            "final_url": final_url,
            "license": candidate.get("license", ""),
            "version": candidate.get("version", ""),
            "discovered_by": candidate.get("discovered_by", ""),
            "text_path": str(output_path.relative_to(record_dir)),
            "text_characters": len(text),
            "text_sha256": sha256_file(output_path),
            "pdf_bytes": len(body),
            "pdf_sha256": sha256_file(pdf_path),
            "title_match_score": round(match_score, 4),
            "recovered_at": utc_now(),
        }
        if keep_pdf:
            result["pdf_path"] = str(pdf_path.relative_to(record_dir))
        else:
            pdf_path.unlink(missing_ok=True)
        return result
    except Exception:
        pdf_path.unlink(missing_ok=True)
        raise
    finally:
        if temporary_text:
            temporary_text.unlink(missing_ok=True)


def existing_text_result(record_dir: Path) -> Dict[str, Any]:
    path = article_text_path(record_dir)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    provenance = read_json(
        record_dir / "extracted" / "text" / "article_text_source.json",
        default={},
    )
    result = dict(provenance) if isinstance(provenance, dict) else {}
    source_status = result.get("status", "")
    result.update({
        "status": "existing",
        "text_path": str(path.relative_to(record_dir)),
        "text_characters": len(text),
        "text_sha256": sha256_file(path),
        "checked_at": utc_now(),
    })
    if source_status:
        result["source_status"] = source_status
    return result


def recover_article_text(
    record_dir: Path,
    metadata: Dict[str, Any],
    http: HttpClient,
    overwrite: bool = False,
    keep_pdf: bool = True,
    max_candidates: int = 12,
    elsevier_api_key: str = "",
    elsevier_inst_token: str = "",
) -> Dict[str, Any]:
    result_path = record_dir / "extracted" / "text" / "article_text_source.json"
    if has_article_text(record_dir) and not overwrite:
        return existing_text_result(record_dir)

    errors: List[str] = []
    cached_xml = record_dir / "extracted" / "text" / "pmc_article.xml"
    article_xml = ""
    if cached_xml.exists():
        article_xml = cached_xml.read_text(encoding="utf-8", errors="replace")
    elif metadata.get("pmcid"):
        try:
            fetched_xml, status = fetch_pmc_jats(http, metadata.get("pmcid", ""))
            article_xml = fetched_xml or ""
            if not article_xml:
                errors.append(f"pmc: {status}")
        except Exception as exc:  # noqa: BLE001 - OA fallbacks may still recover text.
            errors.append(f"pmc: {type(exc).__name__}: {exc}")
    if article_xml:
        try:
            saved = save_article_text(record_dir, article_xml)
            result = {
                "status": "recovered_pmc",
                "source": "PubMed Central",
                "source_type": "repository",
                "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{metadata.get('pmcid', '')}/",
                "license": "",
                "version": "publishedVersion",
                "recovered_at": utc_now(),
                **saved,
                "errors": errors,
            }
            write_json(result_path, result)
            return result
        except Exception as exc:  # noqa: BLE001 - continue to OpenAlex OA locations.
            errors.append(f"pmc_parse: {type(exc).__name__}: {exc}")

    doi = str(metadata.get("doi", "")).strip()
    if not doi:
        result = {
            "status": "unavailable",
            "reason": "missing_doi_and_pmc_text",
            "errors": errors,
            "checked_at": utc_now(),
        }
        write_json(result_path, result)
        return result

    if doi.lower().startswith("10.1016/") and elsevier_api_key:
        try:
            article_xml, source_url = fetch_elsevier_article_xml(
                http=http,
                doi=doi,
                api_key=elsevier_api_key,
                inst_token=elsevier_inst_token,
            )
            result = save_elsevier_article_text(
                record_dir=record_dir,
                metadata=metadata,
                article_xml=article_xml,
                source_url=source_url,
            )
            result["errors"] = errors
            write_json(result_path, result)
            return result
        except Exception as exc:  # noqa: BLE001 - OA repository paths may still work.
            errors.append(f"elsevier_api: {type(exc).__name__}: {exc}")

    work: Dict[str, Any] = {}
    discovery_url = openalex_work_url(doi)
    try:
        work, _, _ = http.request_json(discovery_url, headers={"Accept": "application/json"})
    except Exception as exc:  # noqa: BLE001 - report unavailable without damaging the record.
        errors.append(f"openalex: {type(exc).__name__}: {exc}")
        result = {
            "status": "unavailable",
            "reason": "openalex_lookup_failed",
            "discovery_url": discovery_url,
            "errors": errors,
            "checked_at": utc_now(),
        }
        write_json(result_path, result)
        return result

    candidates = refreshed_oai_candidates(http, work, errors)
    candidates.extend(direct_openalex_candidates(work))
    candidates = unique_candidates(candidates)[:max_candidates]
    attempted_urls = set()

    for candidate in candidates:
        attempted_urls.add(candidate["url"])
        try:
            result = try_pdf_candidate(
                record_dir=record_dir,
                metadata=metadata,
                http=http,
                candidate=candidate,
                keep_pdf=keep_pdf,
            )
            result["discovery_url"] = discovery_url
            result["openalex_work_id"] = work.get("id", "")
            result["candidate_count"] = len(candidates)
            result["errors"] = errors
            write_json(result_path, result)
            return result
        except Exception as exc:  # noqa: BLE001 - try the next independently verified OA URL.
            errors.append(
                f"pdf {candidate.get('source', '')} {candidate.get('url', '')}: "
                f"{type(exc).__name__}: {exc}"
            )

    remaining = max(0, max_candidates - len(attempted_urls))
    page_candidates = repository_page_candidates(http, work, errors)
    page_candidates = [
        candidate for candidate in page_candidates if candidate["url"] not in attempted_urls
    ][:remaining]
    candidates.extend(page_candidates)
    for candidate in page_candidates:
        attempted_urls.add(candidate["url"])
        try:
            result = try_pdf_candidate(
                record_dir=record_dir,
                metadata=metadata,
                http=http,
                candidate=candidate,
                keep_pdf=keep_pdf,
            )
            result["discovery_url"] = discovery_url
            result["openalex_work_id"] = work.get("id", "")
            result["candidate_count"] = len(attempted_urls)
            result["errors"] = errors
            write_json(result_path, result)
            return result
        except Exception as exc:  # noqa: BLE001 - try all verified repository links.
            errors.append(
                f"pdf {candidate.get('source', '')} {candidate.get('url', '')}: "
                f"{type(exc).__name__}: {exc}"
            )

    result = {
        "status": "unavailable",
        "reason": "no_valid_open_pdf",
        "discovery_url": discovery_url,
        "openalex_work_id": work.get("id", ""),
        "candidate_count": len(attempted_urls),
        "errors": errors,
        "checked_at": utc_now(),
    }
    write_json(result_path, result)
    return result
