"""PubMed Central full text, figures, and supplementary files.

`pubmed.py` gets you abstracts. This gets you papers.

**Why the S3 bucket and not `efetch db=pmc`.** The obvious move is a second E-utilities
call. It's the wrong one. The PMC Cloud Service open-data bucket is unauthenticated, has
no NCBI rate limit (88 concurrent LIST calls measured at 0.4s, against E-utilities' 3/sec),
and each per-article prefix already contains everything we need: a 924-byte metadata
sidecar with the licence and retraction flags, the JATS XML, a plain-text rendering PMC
did for us, and — the part nothing else provides — the figure images. `efetch db=pmc`
gives none of that and makes you download 11.5 MB to learn what a LIST call answers for
free.

The guards here, like the ones in `pubmed.py`, each correspond to a verified failure mode:

1. `efetch db=pmc` reports a closed article as HTTP 200 with a complete `<front>` — title,
   authors, abstract — and no `<body>`, with no error anywhere. We sidestep it entirely:
   in this bucket, an article with no objects is treated as an article with no full text.
2. The object key carries a version suffix that CANNOT be guessed. Observed versions in a
   145-paper corpus: 1 (x73), 2 (x2), and 319 (x2). Hardcoding `.1` silently 404s.
3. A figure named in the JATS may not be retrievable at all. Across 409 figures in 77
   papers, 40 (10%) have no image object in the bucket — PMC never deposited it, most
   often for author manuscripts — and 21 of the remaining 369 (5%) are over the 500 KB
   MAX_BINARY_BYTES ceiling, above which `read_file` errors instead of returning an
   image. Both are detected before staging, and reported apart, because the remedies
   differ: one is gone for good, the other is a real image the sandbox can't carry.
4. The legacy paths (`oa.fcgi`, `oa_package/`, `oa_comm/`, `deprecated/`) are withdrawn.
   Some of them still answer 200, which makes them a trap: code written against one passes
   testing and breaks later. Only the flat `PMC{id}.{version}/` layout is used here.
5. `"".join(node.itertext())` — correct for an abstract — silently fuses every paragraph
   in a body, because itertext inserts nothing between elements. See `_block_text`.

One thing this module deliberately does NOT do is keep its output small. `pmc_locate`
returns ~2,000 tokens per paper (mostly figure captions) and `fetch_full_text` returns
~10,000, which is more than the interpreter's whole result budget for a single paper.
That is the intended shape: results live in the JS heap, get projected down to a summary
or handed to a subagent, and never land in the root model's context wholesale. The
prompt, not the tool, enforces it.
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from langchain_core.tools import tool

# `normalize_pmcid` lives in pubmed.py alongside `validate_pmids`, because that's where
# the "never coerce an identifier" rule is established and it's what produces PMCIDs.
from hpo_agent.upstream.paths import pmc_cache
from hpo_agent.upstream.sources import cache_io
from hpo_agent.upstream.sources._errors import SourceError
from hpo_agent.upstream.sources.pubmed import normalize_pmcid

# The current PMC Cloud Service layout. Flat, one prefix per article *version*.
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"

# Resolved listings live outside the versioned dirs because the version is what they
# resolve — we don't know which directory to look in until after this lookup.


# S3 imposes no NCBI-style limit, but unbounded fan-out on a 200-paper corpus buys
# nothing — the wall clock is dominated by the largest object, not by queue depth.
S3_CONCURRENCY = 16

# deepagents' MAX_BINARY_BYTES. A binary file over this returns an ERROR from read_file
# rather than an image block, so staging a larger figure would hand a subagent a dead
# path. Mirrored, not imported: it's a private constant of the backend module.
SANDBOX_READ_CAP = 500 * 1024

# Supplementary files reach 38 MB in the measured corpus. Uploading one into the sandbox
# to run pandas over it is fine; uploading forty is not. Per-file cap.
MAX_SUPPLEMENTARY_BYTES = 10 * 1024 * 1024

# Where staged files land in the sandbox. `figures/` is deliberately NOT `out/`:
# `out/` is swept by ArtifactMiddleware and shown to the user, so an agent reading its
# own working figures out of `out/` would spam them. See the prompt.
SANDBOX_FIGURE_DIR = "/workspace/figures"
SANDBOX_SUPPLEMENTARY_DIR = "/workspace/supplementary"

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

DATA_SUFFIXES = (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".docx", ".pdf", ".zip", ".pptx")


class PMCError(SourceError):
    """A PMC lookup failed in a way the caller needs to know about."""


# --------------------------------------------------------------------------------
# S3 layer
# --------------------------------------------------------------------------------

_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    """Lazily created so the semaphore binds to the running loop, not import time."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(S3_CONCURRENCY)
    return _sem


async def _s3_list(client: httpx.AsyncClient, prefix: str) -> list[tuple[str, int]]:
    """List objects under `prefix`, following continuation tokens."""
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        async with _semaphore():
            resp = await client.get(BUCKET, params=params)
        if resp.status_code != 200:
            raise PMCError(f"S3 list failed for {prefix!r}: HTTP {resp.status_code}")
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise PMCError(f"S3 list returned invalid XML for {prefix!r}") from exc
        # S3 XML is namespaced; parsing also decodes entities in keys and tokens.
        for entry in root.findall("{*}Contents"):
            key = entry.findtext("{*}Key")
            size = entry.findtext("{*}Size")
            if key is None or size is None or not size.isdigit():
                raise PMCError(f"S3 list returned an invalid object for {prefix!r}")
            out.append((key, int(size)))
        if root.findtext("{*}IsTruncated") != "true":
            return out
        next_token = root.findtext("{*}NextContinuationToken")
        if not next_token or next_token == token:
            raise PMCError(
                f"S3 list returned a missing or repeated continuation token for {prefix!r}"
            )
        token = next_token


async def _s3_get(client: httpx.AsyncClient, key: str) -> bytes | None:
    """Fetch one object. None on 404 (the article simply doesn't have that file)."""
    async with _semaphore():
        resp = await client.get(f"{BUCKET}/{key}")
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise PMCError(f"S3 get failed for {key!r}: HTTP {resp.status_code}")
    return resp.content


# --------------------------------------------------------------------------------
# Package resolution — version, availability, and object inventory in one call
# --------------------------------------------------------------------------------

async def _resolve(client: httpx.AsyncClient, pmcid: str) -> dict | None:
    """Resolve a PMCID to its versioned prefix and object inventory.

    Returns None when the article has no objects, which is exactly the set of articles
    with no retrievable full text — verified against `efetch db=pmc`, which produced a
    `<body>` for precisely the 77 of 88 PMCIDs that have objects here.

    The trailing dot on the prefix matters: `PMC594` without it also matches PMC5941234.
    """
    cached = pmc_cache() / "_resolved" / f"{pmcid}.json"
    # MISSING means absent or corrupt; a cached `null` is a real answer meaning "no
    # objects", so the two cannot be collapsed or every absent paper re-resolves.
    hit = await cache_io.aread_json(cached)
    if hit is not cache_io.MISSING:
        return hit or None

    objects = await _s3_list(client, f"{pmcid}.")

    if not objects:
        # Cache the negative too. A miss is a stable fact for the length of a demo, and
        # re-listing every absent paper on every turn is the common case.
        await cache_io.awrite_json(cached, None)
        return None

    # Keys look like 'PMC5904197.1/PMC5904197.1.xml'. Take the highest version present,
    # never a hardcoded .1 — versions 2 and 319 both occur in a 145-paper corpus.
    versions: dict[int, dict[str, int]] = {}
    for key, size in objects:
        head, _, name = key.partition("/")
        if not name:
            continue
        _, _, ver = head.partition(".")
        if ver.isdigit():
            versions.setdefault(int(ver), {})[name] = size
    if not versions:
        await cache_io.awrite_json(cached, None)
        return None

    version = max(versions)
    package = {
        "pmcid": pmcid,
        "version": version,
        "prefix": f"{pmcid}.{version}",
        "objects": versions[version],
    }
    await cache_io.awrite_json(cached, package)
    return package


async def _object_bytes(
    client: httpx.AsyncClient, package: dict, name: str
) -> bytes | None:
    """Fetch one object from a resolved package, caching it host-side.

    The cache mirrors the S3 layout exactly (`data/pmc/PMC5904197.1/…`) so it stays
    legible when you go looking. Like the abstract cache it is host-side and invisible to
    the agent: it exists to avoid refetching, not to be read by anything but this module.
    """
    if name not in package["objects"]:
        return None
    # `name` is a bucket key's tail, and it lands in a host path below. NCBI's bucket has
    # never served a key that would escape it, but nothing here would notice if it did.
    if "/" in name or name.startswith("."):
        return None

    local = pmc_cache() / package["prefix"] / name
    # Figures and PDFs run to megabytes; reading one inline would block the loop for
    # every other run in the process, not just this one.
    hit = await cache_io.aread_bytes(local)
    if hit is not None:
        return hit

    data = await _s3_get(client, f"{package['prefix']}/{name}")
    if data is None:
        return None
    await cache_io.awrite_bytes(local, data)
    return data


async def _sidecar(client: httpx.AsyncClient, package: dict) -> dict:
    """The 924-byte metadata JSON: licence, retraction, OA/manuscript flags."""
    raw = await _object_bytes(client, package, f"{package['prefix']}.json")
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------------
# JATS parsing
# --------------------------------------------------------------------------------

def _inline_text(node: ET.Element) -> str:
    """One line of text, inline markup included.

    Same trap as PubMed abstracts: JATS is full of <italic>/<sup>/<xref>, and `node.text`
    stops at the first child. In a 25-record abstract sample that silently dropped
    content from 9 nodes. Correct for labels and table cells; NOT for section bodies —
    see `_block_text`.
    """
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


# Elements that must be followed by a paragraph break when flattened to text.
_BLOCK_TAGS = frozenset({
    "p", "title", "sec", "abstract", "list", "list-item", "disp-quote", "disp-formula",
    "statement", "boxed-text", "verse-group", "def-item", "speech", "caption", "label",
})

# Rendered separately with their own structure, so they're pulled out of the running
# text rather than flattened into it. Inline, a <table-wrap> becomes an unreadable
# run of undelimited cells, and a duplicated <fig> caption is pure waste.
_LIFTED_TAGS = frozenset({"table-wrap", "fig", "supplementary-material", "table-wrap-group"})


def _block_text(node: ET.Element, skip_first_title: bool = False) -> str:
    """Flatten an element to text with paragraph structure preserved.

    **`"".join(node.itertext())` is wrong here**, even though it's right for an abstract.
    itertext inserts nothing between elements, so consecutive <p> children come back
    fused — a body renders as '...regulates ferroptosis.The P47S polymorphism...' with no
    break anywhere. Every sentence boundary at a paragraph edge disappears, which is
    exactly the kind of damage a model reads straight past.
    """
    parts: list[str] = []
    skipped = not skip_first_title

    def walk(el: ET.Element) -> None:
        nonlocal skipped
        if el.tag in _LIFTED_TAGS:
            return
        if not skipped and el.tag == "title":
            skipped = True
            if el.tail:
                parts.append(el.tail)
            return
        block = el.tag in _BLOCK_TAGS
        if block:
            parts.append("\n\n")
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append("\n\n")

    walk(node)
    text = re.sub(r"[ \t]+", " ", "".join(parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _caption_text(parent: ET.Element) -> str:
    """A <caption>/<label> pair flattened to a single line."""
    caption = parent.find("./caption")
    return re.sub(r"\s+", " ", _block_text(caption)).strip() if caption is not None else ""


# Strips '1.', '3.2)', 'IV.' from a heading. The separator is REQUIRED, which is what
# keeps the roman-numeral branch from eating the 'I' in 'Introduction' and leaving
# 'ntroduction'.
_SEC_NUM_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[IVX]{1,5}[.)]\s*)")

# Canonical buckets, checked in order. Substring matching against the normalised title,
# because journals write 'Materials and methods', 'METHODS', 'Methods and Materials',
# '2. Experimental section' and mean the same thing.
_SECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("methods", ("method", "material", "experimental", "procedure", "protocol")),
    ("results", ("result", "finding")),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "summary", "perspective", "outlook")),
    ("intro", ("introduction", "background")),
)


# Checked before the alias table. 'Supplementary Material' contains 'material' and so
# bucketed as `methods`, which on a Science article is the *only* section — so
# `sections=['methods']` picked an empty back-matter stub and returned nothing, without
# falling back. Matching is substring-based by design; this is the exclusion list that
# keeps back matter out of the buckets.
_NOT_CANONICAL = ("supplementary", "supporting information")


def canonical_section(title: str | None) -> str | None:
    """Map a section heading to a canonical bucket, or None if it isn't one."""
    if not title:
        return None
    t = _SEC_NUM_RE.sub("", title).strip().lower()
    if any(p in t for p in _NOT_CANONICAL):
        return None
    for name, patterns in _SECTION_ALIASES:
        if any(p in t for p in patterns):
            return name
    return None


def _render_table(wrap: ET.Element) -> str:
    """Flatten a JATS table to pipe-delimited rows.

    Worth the twenty lines: quantitative results live in tables, and a table rendered as
    undelimited itertext is unreadable to a model — every cell runs into the next.
    """
    lines = []
    for row in wrap.findall(".//tr"):
        # Cells are inline by nature; a newline inside one would break the row.
        cells = [_inline_text(c) for c in row.findall("./th") + row.findall("./td")]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _untitled_section(nodes: list[ET.Element], lead: str | None) -> dict | None:
    """Flatten a run of section-less body children into one untitled section.

    The nodes are re-parented into a throwaway <sec> purely so `_block_text` has a
    single element to walk; ElementTree copies no state on append, so the real tree
    is untouched.
    """
    holder = ET.Element("sec")
    if lead and lead.strip():
        holder.text = lead
    holder.extend(nodes)
    text = _block_text(holder)
    return {"title": None, "canonical": None, "chars": len(text), "text": text} if text else None


def _parse_sections(body: ET.Element) -> list[dict]:
    """Top-level sections, in document order.

    **Body content outside a <sec> is content**, and keying the fallback on "no <sec>
    children at all" is not enough to catch it. Science research articles are deposited
    as a run of bare <p> children followed by a single 'Supplementary Material' <sec>,
    which is one <sec> — so the old `if not secs` fallback never fired and the whole
    article body was dropped. Measured: PMC3030664 and PMC7164637 both reported
    `body_chars: 0` against 12,791 and 10,339 characters of real body text, and nothing
    in the response said so (`fell_back` stays False, because a section *was* returned).
    Same silent-wrong-answer class as the efetch traps in pubmed.py.

    So loose runs are emitted as untitled sections interleaved in document order, rather
    than as a whole-body fallback. A body with no <sec> at all still collapses to the
    single untitled section it always did.
    """
    out: list[dict] = []
    loose: list[ET.Element] = []
    # Text before the first child element — rare, but it is body text like any other.
    lead: str | None = body.text

    def flush() -> None:
        nonlocal loose, lead
        if (section := _untitled_section(loose, lead)) is not None:
            out.append(section)
        loose, lead = [], None

    for child in body:
        if child.tag != "sec":
            loose.append(child)
            continue
        flush()
        title = _inline_text(child.find("title")) if child.find("title") is not None else None
        # skip_first_title: the title is emitted as a heading by the caller, so leaving
        # it in the body too gives '## Introduction / IntroductionThe tumor...'.
        text = _block_text(child, skip_first_title=True)
        out.append(
            {
                "title": title or None,
                "canonical": canonical_section(title),
                "chars": len(text),
                "text": text,
            }
        )
    flush()
    return out


def _resolve_asset(href: str | None, objects: dict[str, int]) -> tuple[str | None, int]:
    """Match a JATS href to a real object name. Exact first, then by stem.

    Measured exact-match rate is 9/9 on a nine-figure paper, but hrefs occasionally omit
    the extension, so the stem fallback covers that without guessing a suffix.
    """
    if not href:
        return None, 0
    if href in objects:
        return href, objects[href]
    stem = href.rsplit(".", 1)[0].lower()
    for name, size in objects.items():
        if name.rsplit(".", 1)[0].lower() == stem:
            return name, size
    return None, 0


def parse_jats(xml_bytes: bytes, objects: dict[str, int]) -> dict:
    """Parse a JATS article into sections, figures, tables and supplementary material."""
    root = ET.fromstring(xml_bytes)
    art = root if root.tag == "article" else root.find(".//article")
    if art is None:
        raise PMCError("no <article> element in JATS payload")

    body = art.find("./body")
    sections = _parse_sections(body) if body is not None else []

    figures = []
    for fig in art.findall(".//fig"):
        graphic = fig.find(".//graphic")
        href = graphic.get(XLINK_HREF) if graphic is not None else None
        name, size = _resolve_asset(href, objects)
        # Two distinct failures, kept distinct because the remedies differ: a figure PMC
        # never deposited is gone for good (fall back to the caption), while one that is
        # merely too large is a real image the sandbox just can't hand to a model.
        # Measured over 409 figures in 77 papers: 40 (10%) not deposited, 21 (5%) oversize.
        if not name:
            reason = (
                "PMC hosts no image file for this figure — the JATS references "
                f"{href!r} but it was never deposited. Use the caption instead."
            )
        elif size > SANDBOX_READ_CAP:
            reason = (
                f"image is {size:,} bytes, over the {SANDBOX_READ_CAP:,}-byte limit for "
                "reading a file as an image. Use the caption instead."
            )
        else:
            reason = None
        figures.append(
            {
                "fig_id": fig.get("id"),
                "label": fig.findtext("./label"),
                "caption": _caption_text(fig),
                "file": name,
                "bytes": size,
                # Precomputed so the agent knows which figures a subagent can actually
                # read before it stages anything.
                "readable_in_sandbox": reason is None,
                "unavailable_reason": reason,
            }
        )

    tables = []
    for wrap in art.findall(".//table-wrap"):
        tables.append(
            {
                "table_id": wrap.get("id"),
                "label": wrap.findtext("./label"),
                "caption": _caption_text(wrap),
                "rows": _render_table(wrap),
            }
        )

    supplementary = []
    for sup in art.findall(".//supplementary-material"):
        media = sup.find(".//media")
        href = (media.get(XLINK_HREF) if media is not None else None) or sup.get(XLINK_HREF)
        name, size = _resolve_asset(href, objects)
        supplementary.append(
            {
                "label": sup.findtext("./label"),
                "caption": _caption_text(sup),
                "file": name,
                "bytes": size,
            }
        )

    return {
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "supplementary": supplementary,
    }


async def _load(client: httpx.AsyncClient, pmcid: str) -> tuple[dict, dict, dict] | None:
    """Resolve + sidecar + parsed JATS for one article. None when unavailable."""
    package = await _resolve(client, pmcid)
    if package is None:
        return None
    sidecar = await _sidecar(client, package)
    raw = await _object_bytes(client, package, f"{package['prefix']}.xml")
    if raw is None:
        return None
    try:
        parsed = parse_jats(raw, package["objects"])
    except (ET.ParseError, PMCError) as exc:
        raise PMCError(f"{pmcid}: could not parse JATS ({exc})") from exc
    return package, sidecar, parsed


def _provenance(pmcid: str, sidecar: dict) -> dict:
    """The fields that must travel with every answer drawn from this paper."""
    return {
        "pmcid": pmcid,
        "pmid": str(sidecar["pmid"]) if sidecar.get("pmid") else None,
        "title": sidecar.get("title"),
        "citation": sidecar.get("citation"),
        "license": sidecar.get("license_code"),
        "retracted": bool(sidecar.get("is_retracted")),
        "is_manuscript": bool(sidecar.get("is_manuscript")),
        # TDM and the ND licences permit analysis but not redistribution, so a figure
        # from one must not be copied into the user's download folder. The gate is in
        # the prompt; this is the flag it reads.
        "redistributable": (sidecar.get("license_code") or "").upper().startswith("CC BY")
        and "ND" not in (sidecar.get("license_code") or "").upper(),
    }


async def _gather(pmcids: list[str], fn) -> tuple[dict, list[str], list[str]]:
    """Run `fn(client, pmcid)` across a corpus, splitting misses from bad input."""
    normalized, invalid = [], []
    for raw in pmcids:
        n = normalize_pmcid(raw)
        (normalized if n else invalid).append(n or str(raw))
    normalized = list(dict.fromkeys(normalized))

    records: dict[str, Any] = {}
    unavailable: list[str] = []
    async with httpx.AsyncClient(
        timeout=180.0, follow_redirects=True, headers={"User-Agent": "deep_life_sci/0.1"}
    ) as client:
        results = await asyncio.gather(
            *(fn(client, p) for p in normalized), return_exceptions=True
        )
    for pmcid, result in zip(normalized, results, strict=True):
        if isinstance(result, Exception):
            unavailable.append(f"{pmcid} (error: {result})")
        elif result is None:
            unavailable.append(pmcid)
        else:
            records[pmcid] = result
    return records, unavailable, invalid


# --------------------------------------------------------------------------------
# Tool 1 — triage
# --------------------------------------------------------------------------------

@tool
async def pmc_locate(pmcids: list[str]) -> dict:
    """Check which papers have full text in PMC and what's in them, WITHOUT the text.

    Call this before `fetch_full_text`, always. It is the cheap triage step: it tells you
    what each paper contains and how many characters it would cost, so you can decide
    where to spend context. It returns section titles, figure labels and captions, table
    captions, and supplementary file names — but no body text.

    Roughly 53% of PubMed papers have retrievable PMC full text. A paper missing from
    `available` is not an error and not a failure of this tool; report it to the user as
    "no full text available" and fall back to the abstract.

    Args:
        pmcids: PMCIDs, e.g. ['PMC5904197']. Bare digits and lowercase are accepted.
            Get these from `pubmed_search` or `fetch_abstracts`, which both return a
            `pmcid` field.

    Returns:
        available: {pmcid: {
            pmid, title, citation, license, retracted, is_manuscript, redistributable,
            body_chars: total characters of body text,
            sections: [{title, canonical, chars}] — `canonical` is one of
                intro/methods/results/discussion/conclusion, or null. Pass a canonical
                name to `fetch_full_text(sections=...)` to fetch only that part.
            figures: [{fig_id, label, caption, file, bytes, readable_in_sandbox,
                unavailable_reason}] — captions are FULL TEXT here, so a caption-level
                question needs no further fetching. Only stage figures with
                `readable_in_sandbox: true`; for the rest (~15% — either never deposited
                by PMC or over the 500 KB read limit) `unavailable_reason` says which,
                and the caption is your fallback.
            tables: [{table_id, label, caption}] — row data comes from `fetch_full_text`.
            supplementary: [{label, caption, file, bytes}]
        }}
        unavailable: PMCIDs with no full text in PMC
        invalid: inputs that were not usable PMCIDs
    """

    async def one(client, pmcid):
        loaded = await _load(client, pmcid)
        if loaded is None:
            return None
        _, sidecar, parsed = loaded
        return {
            **_provenance(pmcid, sidecar),
            "body_chars": sum(s["chars"] for s in parsed["sections"]),
            "sections": [
                {k: s[k] for k in ("title", "canonical", "chars")}
                for s in parsed["sections"]
            ],
            "figures": parsed["figures"],
            "tables": [
                {k: t[k] for k in ("table_id", "label", "caption")}
                for t in parsed["tables"]
            ],
            "supplementary": parsed["supplementary"],
        }

    records, unavailable, invalid = await _gather(pmcids, one)
    return {"available": records, "unavailable": unavailable, "invalid": invalid}


# --------------------------------------------------------------------------------
# Tool 2 — the text
# --------------------------------------------------------------------------------

@tool
async def fetch_full_text(
    pmcids: list[str],
    sections: list[str] | None = None,
    include_captions: bool = True,
    include_tables: bool = True,
) -> dict:
    """Retrieve full text for papers, optionally only certain sections.

    **Pass this into subagents, do not read it yourself.** The median paper is ~40,000
    characters (~10,000 tokens) and a whole corpus will not fit in your context — nor
    in an `eval` result, which is capped at 40,000 characters. Fan out one
    `full-text-analyst` per paper with the text in its prompt, exactly as you do for
    abstracts, and synthesize the structured answers that come back.

    Use `sections` to cut the cost when the question only needs part of a paper: methods
    is ~3,400 tokens against ~10,000 for the whole body.

    Args:
        pmcids: PMCIDs to fetch. Batch them all in one call.
        sections: Canonical names ('intro', 'methods', 'results', 'discussion',
            'conclusion') and/or substrings of a literal section title from
            `pmc_locate`. None or empty = the whole body. If none of the requested
            sections exist in a paper — about 1 in 4 have no methods section, mostly
            reviews — the whole body is returned instead and `fell_back` is set, so
            check it before assuming you got only what you asked for.
        include_captions: Append figure captions. Cheap (~1,400 tokens for a whole
            paper's worth) and often answers a figure question without an image.
        include_tables: Append table captions and pipe-delimited rows. This is where
            quantitative results usually live.

    Returns:
        records: {pmcid: {pmid, title, citation, license, retracted, is_manuscript,
            redistributable, text, chars, sections_returned, fell_back}}
        unavailable: PMCIDs with no full text in PMC
        invalid: inputs that were not usable PMCIDs
    """
    wanted = [s.strip().lower() for s in (sections or []) if s and s.strip()]

    def select(parsed: dict) -> tuple[list[dict], bool]:
        if not wanted:
            return parsed["sections"], False
        picked = [
            s
            for s in parsed["sections"]
            if (s["canonical"] and s["canonical"] in wanted)
            or any(w in (s["title"] or "").lower() for w in wanted)
        ]
        # Degrade to the whole paper rather than returning nothing — but say so. Keyed on
        # the character count, not on `picked` being empty: back-matter stubs match a
        # title and carry no text, so a match is not the same thing as an answer.
        if not any(s["chars"] for s in picked):
            return parsed["sections"], True
        return picked, False

    async def one(client, pmcid):
        loaded = await _load(client, pmcid)
        if loaded is None:
            return None
        _, sidecar, parsed = loaded
        picked, fell_back = select(parsed)

        parts = []
        for s in picked:
            parts.append(f"## {s['title']}\n\n{s['text']}" if s["title"] else s["text"])

        if include_tables and parsed["tables"]:
            for t in parsed["tables"]:
                head = " ".join(x for x in (t["label"], t["caption"]) if x)
                block = f"## {head}".rstrip()
                if t["rows"]:
                    block += f"\n\n{t['rows']}"
                parts.append(block)

        if include_captions and parsed["figures"]:
            caps = "\n".join(
                " ".join(x for x in (f["label"], f["caption"]) if x)
                for f in parsed["figures"]
                if f["label"] or f["caption"]
            )
            if caps:
                parts.append(f"## Figure captions\n\n{caps}")

        text = "\n\n".join(p for p in parts if p.strip())
        return {
            **_provenance(pmcid, sidecar),
            "text": text,
            "chars": len(text),
            "sections_returned": [s["title"] for s in picked],
            "fell_back": fell_back,
        }

    records, unavailable, invalid = await _gather(pmcids, one)
    return {"records": records, "unavailable": unavailable, "invalid": invalid}


# --------------------------------------------------------------------------------
# Tools 3 and 4 — files into the sandbox
# --------------------------------------------------------------------------------
#
# These two need the sandbox backend, so they're built by a factory rather than being
# module-level like the rest. That is the whole reason for the split: a figure has to
# arrive in the sandbox as real bytes on a real path, because the only way a subagent can
# *see* an image is to `read_file` it. Marshalling the bytes through the interpreter
# instead would turn a JPEG into a base64 string, and a base64 string written to disk is
# a text file, not an image.

# Sandbox figure/supplementary staging removed in the cellular-phenotype fork.
