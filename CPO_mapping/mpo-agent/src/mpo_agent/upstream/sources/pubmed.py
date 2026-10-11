"""PubMed E-utilities client and the two agent tools.

The defensive guards in here are not generic paranoia — each one corresponds to a
verified failure mode of the live API. The ones that produce *wrong answers rather than
errors*:

1. efetch tokenizes malformed PMIDs and returns unrelated papers ('42.9' -> PMIDs 42
   and 9), so ids are validated against ^\\d+$ before any request goes out.
2. esearch silently rewrites broken queries ('cancer[nosuchfield]' -> 5.7M hits), so
   `query_translation` and `warnings` are always returned to the caller.
3. esummary's 500-UID cap returns HTTP 200 with no `result` key, so the error key is
   checked before the payload is read.
4. an efetch record's reference list carries an `ArticleIdList` per reference (58 of them
   in one measured record), so `.//ArticleIdList/ArticleId` returns a *cited* paper's
   ids; id extraction is scoped to `PubmedData/ArticleIdList`. See `_article_ids`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from langchain_core.tools import tool

# Anchored to the repo root, not to this file — see the note in paths.py about why
# getting that wrong starts a second empty cache instead of failing.
from mpo_agent.upstream.paths import abstract_cache
from mpo_agent.upstream.sources import cache_io
from mpo_agent.upstream.sources._errors import SourceError
from mpo_agent.upstream.sources._http import (
    RETRY_STATUSES,
    Throttle,
    backoff_delay,
    chunks,
)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# GET breaks at ~361 ids (HTTP 414); POST above this and there is no measured downside.
POST_THRESHOLD = 200
# esummary rejects >500 UIDs with an HTTP 200 error body. Chunk well under it.
# Raising it toward the cap is a pessimisation, not an optimisation: per-request latency
# scales with UID count, so 9 requests of 500 measured *slower* than 23 of 200 (44.9s vs
# 39.6s) for the same 4484 ids. Concurrency is the lever here, not batch size.
SUMMARY_CHUNK = 200
# How many esummary chunks may be in flight at once. This bounds *sockets*, not rate --
# `_throttle` still paces every dispatch, so NCBI sees the same requests/sec whatever
# this is set to, and raising it cannot breach the usage policy.
#
# 5 is where the throttle overtakes the cap as the binding constraint at the keyless
# 0.34s interval (~= latency / interval, 1.70 / 0.34). Measured over a stub transport,
# 4484 ids / 23 requests: sequential 39.4s, cap 3 14.1s, cap 5 9.3s, and caps 6 / 8 / 12
# all identical to cap 5. With NCBI_API_KEY set the interval drops to 0.11s and a bigger
# cap would keep paying (12 -> 4.6s), but 5 still gives 8.8s and is the one value that is
# safe and near-optimal in both configurations.
SUMMARY_CONCURRENCY = 5
FETCH_CHUNK = 200
# In flight at once for efetch, and the same argument as `SUMMARY_CONCURRENCY`: the cap
# bounds sockets, `_throttle` bounds the rate. Kept identical to it deliberately -- the
# two are the same shape of request against the same bucket, and one number is easier to
# reason about than two. efetch is heavier per request (2.6s and ~7 MB of XML for 200
# ids, against 1.7s for esummary), so the cap saturates later here, not sooner: a larger
# value would still pay at the keyless interval. 5 stays because it is safe with a key
# too, where the interval drops to 0.11s.
FETCH_CONCURRENCY = 5
# esearch silently clamps retmax to this; clamp explicitly so the caller knows.
MAX_RETMAX = 9999

PMID_RE = re.compile(r"^\d+$")
FIELD_TAG_RE = re.compile(r"\[([^\[\]]+)\]")

# PubMed's documented search field tags, plus the canonical names einfo reports and the
# long forms PubMed also accepts. An unrecognised tag is NOT rejected by the API — it is
# silently dropped and the search runs across all fields, so 'cancer[nosuchfield]'
# returns 5.7 million hits and reports no error anywhere in the response. Checking
# locally is the only way to catch it, and catching it before the request is better than
# detecting it after. A tag missing from this set produces a warning, not a failure, so
# a newly-added PubMed tag degrades to noise rather than a broken search.
# Two halves, both explicit. An earlier version generated the long forms by splitting a
# `|`-separated string on whitespace, which shredded every multi-word name into its
# component words and let `[date]`, `[terms]`, `[of]` and `[type]` through as valid.
# Nothing here may be built by splitting a string that contains a multi-word name.
# (The SIM905 suppression keeps this table wrapped; ruff's rewrite is 76 quoted items
# on one line. Splitting is safe here precisely because every token is one word.)
_SHORT_TAGS = frozenset(
    """
    1au ad affl aid all au auid book cdat cn cntys cois coln crdt dcom dp ecno ed edat
    epdt fau filt fir full grnt gr invr ip isbn iss issn jid jour la lang lastau lid lr
    majr mesh mh mhda nm ot otitle pa page papx pdat pg pid pl pmc pmcid pmid ppdt ps pt
    ptyp pubn rn sb sh si so subh subs ta ti tiab titl tt uid vi vol word
    """.split()  # noqa: SIM905
)

_LONG_TAGS = frozenset(
    {
        "affiliation", "all fields", "article identifier", "author",
        "author identifier", "book", "completion date",
        "conflict of interest statement", "corporate author", "create date",
        "date - publication", "ec/rn number", "editor", "entry date", "filter",
        "first author name", "full author name", "full investigator name",
        "grants and funding", "investigator", "isbn", "issue", "journal", "language",
        "last author name", "location id", "mesh date", "mesh major topic",
        "mesh subheadings", "mesh terms", "modification date", "nlm unique id",
        "other term", "pagination", "personal name as subject",
        "pharmacological action", "place of publication", "pmid", "publication date",
        "publication type", "publisher", "secondary source id", "subset",
        "supplementary concept", "text words", "title", "title/abstract",
        "transliterated title", "volume",
    }
)

FIELD_TAGS = _SHORT_TAGS | _LONG_TAGS


def check_field_tags(term: str) -> list[str]:
    """Warn about field tags PubMed will silently ignore.

    Date ranges like '2024:2025[dp]' and the tags inside them are handled by the same
    bracket match, so only the tag name is checked.
    """
    warnings = []
    for raw in FIELD_TAG_RE.findall(term):
        tag = raw.strip().lower()
        if tag and tag not in FIELD_TAGS:
            warnings.append(
                f"'[{raw}]' is not a recognised PubMed field tag. PubMed does not "
                f"reject unknown tags — it drops them and searches every field "
                f"instead, so these results are almost certainly far broader than "
                f"intended. Fix the tag and search again."
            )
    return warnings


class PubMedError(SourceError):
    """An E-utilities call failed in a way the caller needs to know about."""


# --------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------

def _common_params() -> dict[str, str]:
    params = {
        # `or`, not a get default: setup writes a per-install name here, and a .env
        # carrying an empty `NCBI_TOOL=` would otherwise drop the identifier entirely.
        "tool": os.environ.get("NCBI_TOOL") or "deep_life_sci",
        "email": os.environ.get("NCBI_EMAIL", ""),
    }
    if key := os.environ.get("NCBI_API_KEY"):
        params["api_key"] = key
    return {k: v for k, v in params.items() if v}


def _min_interval() -> float:
    """3 req/sec without an API key, 10 with one. Leave a little headroom."""
    return 0.11 if os.environ.get("NCBI_API_KEY") else 0.34


# The throttle paces *this* process; NCBI counts per api_key/IP, so a 429 still arrives
# whenever something else is sharing the quota. One fixed 1.5s retry was not enough — a
# 429 mid-fan-out propagated out of the PTC bridge and killed a whole run.
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# NCBI's own bucket, separate from the registry's in ctgov.py.
_throttle = Throttle(_min_interval)


# Transport-level failures, as opposed to HTTP statuses. `RETRY_STATUSES` cannot see
# these: the request never gets far enough to have a status. Retried on the same ladder
# because they are the same kind of event -- a connection NCBI dropped or never
# accepted. Scoped deliberately: `LocalProtocolError` and `UnsupportedProtocol` are
# caller bugs and retrying them just delays the traceback.
RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


async def _request(util: str, **params: Any) -> httpx.Response:
    """Call an E-utility, retrying 429/5xx and dropped connections. Raises on non-200.

    Retries are safe because every E-utility call here is a read; the POST branch below
    is only a workaround for URL length, not a mutation.

    Both failure modes go through one ladder. An earlier version keyed retries purely on
    `resp.status_code`, so a `ReadError` propagated out on the first occurrence without
    consuming an attempt -- and because `_esummary` gathers its chunks, one such error
    takes down the entire search rather than one request of twenty-three.
    """
    payload = {**_common_params(), **{k: v for k, v in params.items() if v is not None}}
    url = f"{BASE_URL}{util}.fcgi"
    # Long id lists must go in the body — GET dies at ~3.3k chars of URL with a 414
    # whose body is not parseable XML/JSON.
    use_post = len(str(payload.get("id", ""))) > POST_THRESHOLD * 9

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            await _throttle.wait()
            try:
                if use_post:
                    resp = await client.post(url, data=payload)
                else:
                    resp = await client.get(url, params=payload)
            except RETRY_EXCEPTIONS as exc:
                if attempt == RETRY_ATTEMPTS:
                    raise PubMedError(
                        f"{util}: {type(exc).__name__} on all {attempt} attempts"
                    ) from exc
                # No response, so no `Retry-After` to honour -- jittered backoff only.
                await asyncio.sleep(
                    backoff_delay(attempt, base=RETRY_BASE_DELAY, maximum=RETRY_MAX_DELAY)
                )
                continue
            if resp.status_code == 200:
                return resp
            retryable = resp.status_code in RETRY_STATUSES
            if retryable and attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(
                    backoff_delay(
                        attempt, base=RETRY_BASE_DELAY, maximum=RETRY_MAX_DELAY, resp=resp
                    )
                )
                continue
            raise PubMedError(
                f"{util} returned HTTP {resp.status_code}"
                + (f" on all {attempt} attempts" if retryable else "")
                + (" (id list too long for GET)" if resp.status_code == 414 else "")
            )
    raise PubMedError(f"{util} exhausted {RETRY_ATTEMPTS} attempts")


def validate_pmids(pmids: list[str]) -> tuple[list[str], list[str]]:
    """Split into (valid, invalid). Never send an unvalidated id to efetch."""
    valid, invalid = [], []
    for p in pmids:
        s = str(p).strip()
        (valid if PMID_RE.match(s) else invalid).append(s)
    # preserve order, drop duplicates (efetch dedupes server-side anyway)
    return list(dict.fromkeys(valid)), invalid


def normalize_pmcid(raw: Any) -> str | None:
    """'5904197' | 'pmc5904197' | 'PMC5904197.1' -> 'PMC5904197'. None if unusable.

    Lives here rather than in `pmc.py` for the same reason `validate_pmids` does: an id
    that gets coerced instead of rejected fetches a real, unrelated paper. Strict about
    the digits for that reason.
    """
    if raw is None:
        return None
    s = str(raw).strip().upper().removeprefix("PMC")
    # A trailing version suffix is common in copy-paste; the version is resolved from
    # the bucket listing, never taken on faith from the caller.
    s = s.split(".", 1)[0]
    return f"PMC{s}" if s.isdigit() else None


# --------------------------------------------------------------------------------
# esearch + esummary
# --------------------------------------------------------------------------------

def _collect_warnings(result: dict) -> list[str]:
    """Everything PubMed did to the query that the caller did not ask for.

    This is the only signal that a query was silently rewritten — a bad field tag
    returns millions of hits with an empty `fieldsnotfound`.
    """
    warnings: list[str] = []
    errorlist = result.get("errorlist") or {}
    warninglist = result.get("warninglist") or {}

    for phrase in errorlist.get("phrasesnotfound", []):
        warnings.append(f"phrase not found in index: {phrase!r}")
    for field in errorlist.get("fieldsnotfound", []):
        warnings.append(f"field tag not recognised: {field!r}")
    for phrase in warninglist.get("quotedphrasesnotfound", []):
        warnings.append(
            f"quoted phrase {phrase!r} not found — PubMed fell back to an unquoted "
            "search, so these results may be unrelated to the intended phrase"
        )
    for phrase in warninglist.get("phrasesignored", []):
        warnings.append(f"phrase ignored: {phrase!r}")
    for msg in warninglist.get("outputmessages", []):
        if msg != "No items found.":
            warnings.append(f"PubMed modified the query: {msg!r}")
    if result.get("ERROR"):
        warnings.append(f"esearch error: {result['ERROR']!r}")
    return warnings


async def _esummary_chunk(chunk: list[str], sem: asyncio.Semaphore) -> dict[str, dict]:
    """One esummary request. The semaphore is held only across the round trip."""
    async with sem:
        resp = await _request("esummary", db="pubmed", id=",".join(chunk), retmode="json")
    payload = resp.json()
    # Over-cap and malformed requests come back as HTTP 200 with an `error` key
    # and no `result` at all — reading result.uids first would look like 0 hits.
    if "error" in payload:
        raise PubMedError(f"esummary: {payload['error']}")
    result = payload.get("result")
    if result is None:
        raise PubMedError("esummary returned no `result` block")
    records: dict[str, dict] = {}
    for uid in result.get("uids", []):
        rec = result.get(uid)
        if isinstance(rec, dict) and not rec.get("error"):
            records[uid] = rec
    return records


async def _esummary(pmids: list[str]) -> dict[str, dict]:
    """Fetch citation metadata, chunked under the 500-UID cap, chunks concurrent.

    Sequential, this was the dominant cost of any large search: a 4484-hit query is 23
    round trips, and one at a time made a single `pubmed_search` 40.6s of a 44.0s `eval`.

    The semaphore is per call rather than module-global — unlike `pmc.py`'s, which is
    shared. Two reasons: the rate ceiling is `_throttle`'s job and is already global, so
    a shared socket cap buys nothing; and a module-global `asyncio.Semaphore` caches the
    first loop it is awaited on and raises if a later call arrives on a different one,
    which is exactly what a per-example `asyncio.run` would do.
    """
    sem = asyncio.Semaphore(SUMMARY_CONCURRENCY)
    parts = await asyncio.gather(
        *(_esummary_chunk(chunk, sem) for chunk in chunks(pmids, SUMMARY_CHUNK))
    )
    records: dict[str, dict] = {}
    for part in parts:
        records.update(part)
    return records


def _summary_to_record(uid: str, rec: dict) -> dict:
    articleids = {a.get("idtype"): a.get("value") for a in rec.get("articleids", [])}
    authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
    return {
        "pmid": uid,
        "title": rec.get("title") or None,
        "first_author": authors[0] if authors else None,
        # The senior author / PI — usually who a biologist means by "lead author".
        "last_author": rec.get("lastauthor") or None,
        # pubdate has 30+ formats ('2024', '2025 Oct-Dec', '2026 Jan-Jun'); only the
        # leading year is reliable, so don't pretend to parse a full date.
        "year": (rec.get("pubdate") or "")[:4] or None,
        # Book records (StatPearls, GeneReviews) leave `source` empty.
        "journal": rec.get("source") or rec.get("booktitle") or None,
        "doi": articleids.get("doi"),
        # Free: esummary already carries this, so knowing whether a paper has PMC full
        # text costs no extra request. **`pmc` first, and not interchangeable with
        # `pmcid`**: `pmc` is the id (`PMC3030664`), while `pmcid` is a deposit receipt
        # containing it (`pmc-id: PMC3030664;manuscript-id: NIHMS262124;`). The receipt
        # is truthy, so reading it first short-circuits the `or`, and `normalize_pmcid`
        # then rightly rejects it — silently, since None is the legal "abstract-only"
        # value. That ordering returned null for 166/166 full-text papers in
        # `data/abstracts/`. Same operand order as `_parse_article`.
        "pmcid": normalize_pmcid(articleids.get("pmc") or articleids.get("pmcid")),
    }


@tool
async def pubmed_search(
    term: str,
    retmax: int = 50,
    sort: str = "relevance",
    mindate: str | None = None,
    maxdate: str | None = None,
) -> dict:
    """Search PubMed with a boolean query and return matching papers with metadata.

    Supports full PubMed boolean syntax: AND/OR/NOT, field tags like [ti], [au],
    [mesh], [pt], and quoted phrases. Example:
    '(CRISPR OR "base editing") AND liver AND 2023:2025[dp] NOT review[pt]'

    ALWAYS read `query_translation` and `warnings` in the response before trusting the
    results. PubMed silently repairs malformed queries rather than rejecting them — an
    unrecognised field tag is dropped and the search runs across all fields, which can
    return millions of irrelevant hits that look like a successful search.

    Use `retmax=0` to probe: it returns `count`, `query_translation` and `warnings`
    without fetching any records, so it is cheap and fast. Probe first, refine the query
    until `count` is at or under 200, and only then pull the records.

    Args:
        term: The boolean query.
        retmax: Max papers to return (capped at 9999). 0 = count-only probe.
        sort: 'relevance', 'pub_date', or 'Author'. Invalid values are ignored by
            PubMed without error, so stick to these three.
        mindate: Earliest publication date, 'YYYY' or 'YYYY/MM/DD'.
        maxdate: Latest publication date, same format.

    Returns:
        count: total matches in PubMed (may be far larger than the records returned)
        returned: how many records are in this response
        query_translation: what PubMed ACTUALLY searched, with MeSH expansion
        warnings: list of ways PubMed altered the query. Empty means the query ran as written
        records: list of {pmid, title, first_author, last_author, year, journal, doi}
    """
    probe_only = int(retmax) == 0
    retmax = 0 if probe_only else max(1, min(int(retmax), MAX_RETMAX))
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": sort,
    }
    if mindate or maxdate:
        params.update(datetype="pdat", mindate=mindate, maxdate=maxdate)

    resp = await _request("esearch", **params)
    payload = resp.json()
    result = payload.get("esearchresult")
    if payload.get("error") or not isinstance(result, dict) or result.get("ERROR"):
        raise PubMedError("esearch returned an error or no result block")
    # Local tag check first: an unknown field tag leaves no trace anywhere in the
    # response, so _collect_warnings alone cannot see it.
    warnings = check_field_tags(term) + _collect_warnings(result)
    count = int(result.get("count", 0) or 0)

    pmids = result.get("idlist", [])
    records = []
    if pmids and not probe_only:
        summaries = await _esummary(pmids)
        # Keep esearch's ordering (relevance/date), which the summary dict loses.
        records = [
            _summary_to_record(p, summaries[p]) for p in pmids if p in summaries
        ]

    return {
        "count": count,
        "returned": len(records),
        "query_translation": result.get("querytranslation") or "",
        "warnings": warnings,
        # Every record is returned, not a truncated head. These land in the interpreter's
        # JS heap, not the model's context — only what `eval` returns reaches the model —
        # so the agent can filter and slice all of them in code. Truncating here would
        # hide records from that code for no benefit.
        "records": records,
    }


# --------------------------------------------------------------------------------
# efetch
# --------------------------------------------------------------------------------

def _text_of(node: ET.Element) -> str:
    """Full text of a node including inline markup.

    Abstracts contain <b>/<i>/<sub>/<sup>; node.text stops at the first child and
    silently drops everything after it.
    """
    return "".join(node.itertext()).strip()


def _article_ids(art: ET.Element) -> dict[str, str]:
    """The article's OWN ids, from the scoped ArticleIdList only.

    **Never** reach for `.//ArticleIdList/ArticleId` here. Every entry in a record's
    reference list carries its own `ArticleIdList`, so `.//` matches all of them —
    PMID 29695998 has **58** — and a last-wins dict comprehension over that returns a
    *cited paper's* ids. Measured: it reported `pmc: PMC5379068` for a paper whose real
    PMCID is PMC5904197, which would then fetch the wrong full text with no error
    anywhere. Same silent-wrong-answer class as the traps in the module docstring.
    """
    for path in ("PubmedData/ArticleIdList", "PubmedBookData/ArticleIdList"):
        node = art.find(path)
        if node is not None:
            return {
                a.get("IdType"): (a.text or "").strip()
                for a in node.findall("ArticleId")
                if a.get("IdType")
            }
    return {}


def _parse_article(art: ET.Element) -> dict | None:
    # Scoped, not `.//PMID`: a record's reference list and CommentsCorrections carry
    # PMIDs too. `findtext` returns the first in document order, which happens to be
    # the right one — but only by accident of ordering. See `_article_ids` below.
    pmid = (
        art.findtext("MedlineCitation/PMID")
        or art.findtext("BookDocument/PMID")
        or art.findtext(".//PMID")
    )
    if not pmid:
        return None

    sections = []
    for node in art.findall(".//Abstract/AbstractText"):
        text = _text_of(node)
        if text:
            sections.append({"label": node.get("Label") or node.get("NlmCategory"), "text": text})

    if any(s["label"] for s in sections):
        abstract = "\n\n".join(
            f"{s['label']}: {s['text']}" if s["label"] else s["text"] for s in sections
        )
    else:
        abstract = "\n\n".join(s["text"] for s in sections)

    pubtypes = [p.text for p in art.findall(".//PublicationType") if p.text]
    reftypes = {c.get("RefType") for c in art.findall(".//CommentsCorrections")}
    retracted = "Retracted Publication" in pubtypes or "RetractionIn" in reftypes

    year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""

    ids = _article_ids(art)

    return {
        "pmid": pmid,
        "title": art.findtext(".//ArticleTitle") or art.findtext(".//BookTitle"),
        "abstract": abstract or None,
        "sections": sections,
        "journal": art.findtext(".//Journal/Title") or art.findtext(".//BookTitle"),
        "year": year[:4] or None,
        "retracted": retracted,
        "publication_types": pubtypes,
        "doi": ids.get("doi"),
        # The handoff to PMC: non-null means full text may be retrievable.
        "pmcid": normalize_pmcid(ids.get("pmc") or ids.get("pmcid")),
    }


def _scan_cache(valid: list[str]) -> tuple[dict[str, dict], list[str], list[str]]:
    """Partition PMIDs into cached records and ones still needing a fetch.

    Blocking, and called via a single `asyncio.to_thread` rather than one hop per
    PMID: a 200-paper corpus is 200 cache reads, and paying thread-handoff overhead
    on each would cost more than the reads themselves.

    Returns `(records, from_cache, to_fetch)`.
    """
    abstract_cache().mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    from_cache: list[str] = []
    to_fetch: list[str] = []

    # One TTL lookup for the whole batch rather than per PMID, for the same reason this
    # function takes a list at all: 200 identical env reads would be 200 more than needed.
    ttl = cache_io.ttl_seconds()

    for pmid in valid:
        path = abstract_cache() / f"{pmid}.json"
        # Reading these directly rather than through `cache_io.aread_json` is what the
        # single-hop batching above buys; the freshness and touch semantics have to match
        # it exactly, so they come from the same helpers.
        if cache_io.is_fresh(path, ttl):
            try:
                cached = json.loads(path.read_text())
                # Schema migration: entries written before PMC support predate the
                # `pmcid` key. Checking for the key (not its value) is what makes this
                # a one-time refetch — None is a legitimate answer for 39% of papers,
                # so a truthiness check would refetch those forever.
                if "pmcid" in cached:
                    records[pmid] = cached
                    from_cache.append(pmid)
                    # Only a usable entry is kept alive: a stale-schema or corrupt file
                    # keeps its mtime so the next sweep collects it.
                    cache_io.touch(path)
                    continue
            except (OSError, json.JSONDecodeError):
                pass  # corrupt cache entry, refetch
        to_fetch.append(pmid)

    return records, from_cache, to_fetch


def _write_cache(parsed: dict[str, dict]) -> None:
    """Persist a chunk of freshly fetched records. Blocking; call via `to_thread`."""
    for pmid, rec in parsed.items():
        (abstract_cache() / f"{pmid}.json").write_text(json.dumps(rec, indent=2))


def _parse_efetch(xml_text: str) -> dict[str, dict]:
    root = ET.fromstring(xml_text)
    records = {}
    # Book records (StatPearls, GeneReviews) are PubmedBookArticle, not PubmedArticle,
    # and they do have abstracts — querying only the latter silently drops them.
    for tag in ("PubmedArticle", "PubmedBookArticle"):
        for art in root.findall(f".//{tag}"):
            rec = _parse_article(art)
            if rec:
                records[rec["pmid"]] = rec
    return records


async def _efetch_chunk(chunk: list[str], sem: asyncio.Semaphore) -> dict[str, dict]:
    """One efetch request, parsed and written to the cache.

    The semaphore is held only across the round trip, not across the parse or the cache
    write -- those are handed to a thread, and holding a slot through them would idle a
    socket the next chunk could be using.

    Parsing is on the event loop on purpose. 200 records is ~7 MB of XML and measures
    40-60ms, which is not worth a thread hop; `_write_cache` is the blocking part and
    that one does get `to_thread`.
    """
    async with sem:
        # retmode=xml, never the text mode: text concatenates every abstract behind a
        # positional counter that renumbers when records drop, so it can't be mapped
        # back to the requested ids. Never pass retmax — it truncates silently.
        resp = await _request("efetch", db="pubmed", id=",".join(chunk), retmode="xml")
    parsed = _parse_efetch(resp.text)
    await asyncio.to_thread(_write_cache, parsed)
    return parsed


@tool
async def fetch_abstracts(pmids: list[str]) -> dict:
    """Retrieve abstracts for a list of PMIDs, batched and cached locally.

    Pass every PMID you need in one call — batching is what keeps the whole workflow
    inside NCBI's rate limit. Do NOT call this once per paper in a loop, and do not
    have subagents call it; fetch here, then pass the text into subagent prompts.

    Already-cached abstracts are served from disk without an HTTP request, so calling
    this again with overlapping PMIDs is cheap. Batching is also what makes the fetch
    concurrent: chunks of one call go out together, where separate calls serialise.

    Args:
        pmids: PMIDs as strings. Anything that isn't all digits is rejected rather
            than sent — PubMed would otherwise return an unrelated paper for it.

    Returns:
        records: {pmid: {title, abstract, sections, journal, year, retracted,
            publication_types, doi, pmcid}}. `abstract` is None for errata and editorials,
            which have citation metadata but no body. `sections` preserves
            structured-abstract labels (BACKGROUND / METHODS / FINDINGS / ...) when the
            journal uses them. `retracted` is True for retracted papers — always surface
            that to the user. `pmcid` is non-null for roughly 61% of papers and is what
            you pass to `pmc_locate` to check for full text; null means abstract-only.
        missing: requested PMIDs that PubMed returned nothing for
        invalid: inputs rejected as malformed
        from_cache: PMIDs served from the local cache
    """
    valid, invalid = validate_pmids([str(p) for p in pmids])

    # The cache lives on the host filesystem, so reading it is blocking I/O. This
    # coroutine runs on the server's event loop under `langgraph dev`, which forbids
    # that outright — see cache_io for why it matters in production too.
    records, from_cache, to_fetch = await asyncio.to_thread(_scan_cache, valid)

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    parts = await asyncio.gather(
        *(_efetch_chunk(chunk, sem) for chunk in chunks(to_fetch, FETCH_CHUNK))
    )
    for part in parts:
        records.update(part)

    return {
        "records": records,
        "missing": [p for p in valid if p not in records],
        "invalid": invalid,
        "from_cache": from_cache,
    }
