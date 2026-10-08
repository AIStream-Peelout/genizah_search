"""Server-rendered public ``/fragment/<doc_id>`` pages for crawlers and visitors.

Each catalogue record gets a lightweight HTML page (Jinja2, autoescaped) with a
canonical URL, Open Graph tags and ``schema.org/Manuscript`` JSON-LD, so search
engines can index the records without running the React app.  Humans get a
prominent "Open in Cairo Genizah AI" button that opens the record in the app.

Data rules (see ``docs/seo_plan.md``):

* The record is looked up by **exact** ``doc_id`` in the catalogue index only
  (``term`` query).  ``search_service.get_document_by_id`` is deliberately not
  used: it falls back to fuzzy shelfmark matching and swallows errors.
* Everything shown comes from the public :class:`DocumentMetadata` built by
  ``search_service._extract_metadata`` (which drops provider URLs, contributor
  strings and non-allowlisted bibliography sources), plus a short allowlist of
  descriptive scalars it does not copy (:data:`EXTRA_PUBLIC_FIELDS`: dates and
  document type).  ``sources_present`` is only used internally, mapped through
  ``public_bibliography_source`` so that just ``pgp`` / ``ktiv`` credits survive.
* Free text from the catalogue goes through :func:`scrub_provider_mentions`:
  the unattributed provider is never named or linked.
* Machine readings (agreed consensus lines only) sit in a ``data-nosnippet``
  section and are clearly labelled as unchecked machine output.  They never
  reach the head metadata (meta description, Open Graph, JSON-LD), where
  ``data-nosnippet`` has no effect, and are left out entirely when
  ``AI_TRANSCRIPTIONS_ENABLED`` is off.

Rendered pages are kept in a small in-process TTL cache so crawler bursts do
not turn into Elasticsearch load.  Elasticsearch failures become a 503 page
with ``Retry-After`` (never a 404, which would get the URL dropped).
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

from elasticsearch import ApiError, TransportError
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.backend.ai_transcriptions import feature_enabled as ai_feature_enabled
from src.backend.consensus_reads import (
    PLACEHOLDER_DESCRIPTIONS,
    ConsensusRead,
    consensus_terms,
    is_thin_description,
    is_uncatalogued,
    read_url,
    to_catalogue_doc_id,
)
from src.backend.seo_urls import CANONICAL_BASE, fragment_path, fragment_url

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.backend.consensus_reads import ConsensusService
    from src.backend.search_service import DocumentMetadata, ElasticsearchService

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
SITE_NAME = "Cairo Genizah AI"
MACHINE_READING_LABEL = "Machine reading — two machine readers agree; not checked by a person"
META_DESCRIPTION_MAX = 160
JSON_LD_DESCRIPTION_MAX = 500
MAX_PAGE_IMAGES = 24
MAX_DOC_ID_CHARS = 512

# Heavy or private fields never fetched for a page (the embedding is ~22 KB;
# full_text_content duplicates description + transcription; tei_metadata only
# carries provider bookkeeping).
CATALOGUE_SOURCE_EXCLUDES: List[str] = ["embedding_vector", "full_text_content", "tei_metadata"]
# Descriptive scalars of the merged index that ``_extract_metadata`` does not
# copy into DocumentMetadata.  Dates go into ``date_info``, the type into
# ``document_type``; nothing provider-related is allowed here.
# Consensus search terms added to JSON-LD keywords of uncatalogued records.
KEYWORD_TERMS_MAX = 15

EXTRA_PUBLIC_FIELDS: Tuple[str, ...] = ("date_display", "date_start", "date_end", "century", "document_type")
_DATE_FIELDS: Tuple[str, ...] = ("date_display", "date_start", "date_end", "century")

# Hosts on which pages may be indexed; anything else (api.cairogenizah.ai, www,
# raw IPs) gets ``X-Robots-Tag: noindex``.
INDEXABLE_HOSTS = frozenset({"cairogenizah.ai", "localhost", "127.0.0.1", "::1"})

PAGE_CACHE_CONTROL = "public, max-age=3600"
NOT_FOUND_CACHE_CONTROL = "public, max-age=600"
RETRY_AFTER_SECONDS = 30
CACHE_MAX_ENTRIES = 2048
CACHE_MAX_BYTES = 64 * 1024 * 1024
CACHE_TTL_SECONDS = 3600

# Public attribution of the two catalogues whose labels survive
# ``public_bibliography_source`` (display order).
CREDITS: Dict[str, Dict[str, str]] = {
    "pgp": {
        "label": "PGP",
        "name": "Princeton Geniza Project",
        "phrase": "the Princeton Geniza Project (PGP)",
        "url": "https://geniza.princeton.edu/en/",
        "licence": "CC BY-NC 4.0",
        "licence_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    },
    "ktiv": {
        "label": "KTIV",
        "name": "KTIV, National Library of Israel",
        "phrase": "KTIV, the National Library of Israel's collection of digitised Hebrew manuscripts",
        "url": "https://www.nli.org.il/",
        "licence": "",
        "licence_url": "",
    },
}
_BIBLIOGRAPHY_GROUPS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("ktiv", "KTIV", "KTIV, National Library of Israel"),
    ("pgp", "PGP", "Princeton Geniza Project"),
    ("other", None, "Further references"),
)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# URLs or bare host names on the unattributed provider's domains.
_BLOCKED_URL_RE = re.compile(
    r"(?:https?://)?(?:[\w-]+\.)*(?<![\w-])(?:genizah|jewishmanuscripts)\.org\b"
    r"(?:[^\s<>\"')\]]*[^\s<>\"')\].,;:!?])?",
    re.IGNORECASE,
)
# The provider acronym, and an attribution phrase built on it ("Information
# from FGP", "Data in part per the FJP", "with information from FGP").
_ACR = r"(?:the\s+)?F[GJ]P\b"
_ATTR = (
    r"\b(?:with\s+)?(?:Information|Info|Data|Description)(?:\s+in\s+part)?"
    r"\s+(?:from|per|via|according\s+to)\s+" + _ACR
)
# A mention the replacement wording must not swallow ("on FGP images").
_NOT_A_PLACE = r"(?!\s+(?:join|image|photo|edition|bibliograph)\w*)"
# Ordered: whole attribution constructions first, so they go as a unit; the
# bare acronyms last.  Removing just the acronym would leave broken sentences
# ("Information.", "edition on.", "CUDL and.").
_PROVIDER_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # "See FGP for further information." -> the whole pointer sentence is dropped
    # (case-sensitive: a sentence-initial pointer; "…: see FGP." is reworded below).
    (re.compile(r"[ \t]*\bSee\s+(?:the\s+)?F[GJ]P\b[^.!?\n]*[.!?]?"), ""),
    # "FGP and CUDL" / "CUDL and FGP" / "PGP, FGP" -> the other source only
    (re.compile(r"\bF[GJ]P\s+(?:and|&)\s+", re.I), ""),
    (re.compile(r",?\s+(?:and|&)\s+" + _ACR, re.I), ""),
    (re.compile(r"\s*,\s*" + _ACR, re.I), ""),
    # "(Information in part from FGP)", "(with information from FGP)" -> dropped
    (re.compile(r"\s*\(\s*" + _ATTR + r"[^()]*\)", re.I), ""),
    # "Information from FGP." / "Information from FGP by X." -> dropped
    (re.compile(r"\s*" + _ATTR + r"(?:\s+by\s+[^.()\n]*)?\.?", re.I), ""),
    # "(not on FGP)", "(as catalogued on FGP)", "(according to FGP)" -> dropped
    (re.compile(r"\s*\([^()]*\b(?:on|in|at|see|to)\s+" + _ACR + r"[^()]*\)", re.I), ""),
    # "(FGP)", "(per FGP)", "(from the FJP)"
    (re.compile(r"\s*\(\s*(?:(?:per|from|via|see|according\s+to)\s+)?(?:the\s+)?F[GJ]P\s*\)", re.I), ""),
    # ", per FGP", " according (to) the FJP", " with information from FGP"
    (
        re.compile(
            r",?\s*\b(?:(?:with\s+)?information\s+)?(?:per|from|via|according(?:\s+to)?)\s+" + _ACR, re.I
        ),
        "",
    ),
    # "edition on FGP", "listed as verso on FGP", "joins: see FGP" -> neutral wording
    (re.compile(r"\b(?:on|in|at)\s+" + _ACR + _NOT_A_PLACE, re.I), "in another catalogue"),
    (re.compile(r"\b(see)\s+" + _ACR + _NOT_A_PLACE, re.I), r"\1 another catalogue"),
    # "The Friedberg Genizah Project and the Ben-Zvi Institute" -> "the Ben-Zvi Institute"
    (
        re.compile(
            r"\b(?:the\s+)?Friedberg\s+(?:Genizah|Jewish\s+Manuscripts?)\s+(?:Project|Society)"
            r"(?:\s*\(\s*F[GJ]P\s*\))?(?:\s+and\s+|\s*,\s*)?",
            re.I,
        ),
        "",
    ),
    # "the Friedberg/NLI metadata" -> "the NLI metadata"
    (re.compile(r"\bFriedberg\s*/\s*", re.I), ""),
    (re.compile(r"\s*/\s*Friedberg\b", re.I), ""),
    (re.compile(r"\bF[GJ]P\b", re.I), ""),
)
# Applied only to text a provider pattern changed, so untouched catalogue text
# keeps its exact spacing and punctuation.
_TIDY_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]+([.,;:!?)])"), r"\1"),
    (re.compile(r"\([ \t]+"), "("),
    (re.compile(r"([,;])\1+"), r"\1"),
    # The doubled period left after a removed "(…)." (ellipses untouched).
    (re.compile(r"(?<![.])([.!?])\s*\.(?![.])"), r"\1"),
    # Punctuation left at the start of a line ("." of a removed "(…).").
    (re.compile(r"(^|\n)[ \t]*(?:[,;:]|\.(?!\.))[ \t]*"), r"\1"),
)
# Formatting markup found in catalogue descriptions and citations.  Narrow on
# purpose: editors use "<and send it>" for supplied text, which must survive.
_MARKUP_TAG_RE = re.compile(
    r"<(?:/?(?:b|i|em|strong|u|sup|sub|small|span|p|div|ol|ul|li)\b[^<>]*|br\s*/?|a\s+href\s*=[^<>]*|/a\s*)>",
    re.IGNORECASE,
)
_EM_SPLIT_RE = re.compile(r"(</?(?:em|i)>)", re.IGNORECASE)
_HEBREW_RE = re.compile("[א-ת]")
_ARABIC_RE = re.compile("[ؠ-ي]")
_LATIN_RE = re.compile("[A-Za-z]")
_ISO_DATE_RE = re.compile(r"^-?\d{3,4}(?:-\d\d(?:-\d\d)?)?(?:/-?\d{3,4}(?:-\d\d(?:-\d\d)?)?)?$")


def scrub_provider_mentions(text: Optional[str]) -> str:
    """Remove names of, and links to, the unattributed catalogue provider.

    Drops URLs on ``genizah.org`` / ``jewishmanuscripts.org`` (any subdomain),
    the acronyms FGP/FJP together with the construction around them (an
    attribution such as "Information from FGP.", "per FGP", "(not on FGP)",
    "See FGP for …", a conjunction such as "CUDL and FGP"), rewording a
    remaining place mention ("edition on FGP") to "in another catalogue", and
    the project names "Friedberg Genizah Project" / "Friedberg Jewish
    Manuscript Society".
    "Friedberg" as a manuscript collection ("Friedberg MSS 9-004/22", held in
    Toronto) or as an author's surname is kept: it identifies an object or a
    person, not the provider.

    :param text: Catalogue free text (description, citation, title).
    :return: The text without provider mentions, whitespace and punctuation tidied.
    :rtype: str
    """
    original = text or ""
    out = _BLOCKED_URL_RE.sub("", original)
    for pattern, repl in _PROVIDER_PATTERNS:
        out = pattern.sub(repl, out)
    if out != original:
        for pattern, repl in _TIDY_PATTERNS:
            out = pattern.sub(repl, out)
    return out.strip()


def strip_markup(text: Optional[str]) -> str:
    """Remove formatting tags and decode entities (no other change).

    :param text: Catalogue text that may carry ``<em>``, ``<a href>`` etc.
    :return: Plain text.
    :rtype: str
    """
    return html.unescape(_MARKUP_TAG_RE.sub("", text or ""))


def clean_block_text(text: Optional[str]) -> str:
    """Public form of a multi-paragraph catalogue text (e.g. a description).

    :param text: Raw text.
    :return: Markup-free, provider-scrubbed text with ``\\n`` line breaks and at
        most one blank line in a row.
    :rtype: str
    """
    plain = strip_markup(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [scrub_provider_mentions(line) for line in plain.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_inline_text(text: Optional[str]) -> str:
    """Public one-line form of a catalogue text (titles, meta tags, JSON-LD).

    :param text: Raw text.
    :return: Markup-free, provider-scrubbed text on one line.
    :rtype: str
    """
    return " ".join(scrub_provider_mentions(" ".join(strip_markup(text).split())).split())


def shorten(text: str, max_chars: int) -> str:
    """Cut text on a word boundary, adding an ellipsis when cut.

    :param text: Text on one line.
    :param max_chars: Maximum length of the result, ellipsis included.
    :return: The text, or its cut form ending in ``"…"``.
    :rtype: str
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[: max(1, max_chars - 1)]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip(" .,;:-–—/") + "…"


def text_direction(text: Optional[str]) -> Tuple[str, Optional[str]]:
    """Writing direction and language tag for a block of manuscript text.

    :param text: Transcription or translation text.
    :return: ``("rtl", "he")`` when Hebrew script dominates (Hebrew, Aramaic,
        Judaeo-Arabic), ``("rtl", "ar")`` for Arabic script, else ``("auto", None)``.
    :rtype: Tuple[str, Optional[str]]
    """
    sample = text or ""
    hebrew = len(_HEBREW_RE.findall(sample))
    arabic = len(_ARABIC_RE.findall(sample))
    latin = len(_LATIN_RE.findall(sample))
    if hebrew and hebrew >= arabic and hebrew >= latin:
        return "rtl", "he"
    if arabic and arabic > latin:
        return "rtl", "ar"
    return "auto", None


def citation_segments(text: Optional[str]) -> List[Dict[str, Any]]:
    """Split a citation into plain and italic runs (``<em>``/``<i>`` honoured).

    :param text: Citation string as stored in the index.
    :return: ``[{"text": str, "em": bool}, …]`` with markup removed and the
        provider scrubbed from every run; empty runs dropped.
    :rtype: List[Dict[str, Any]]
    """
    segments: List[Dict[str, Any]] = []
    italic = False
    for part in _EM_SPLIT_RE.split(text or ""):
        tag = part.lower()
        if tag in ("<em>", "<i>"):
            italic = True
            continue
        if tag in ("</em>", "</i>"):
            italic = False
            continue
        clean = scrub_provider_mentions(" ".join(strip_markup(part).split()))
        if clean:
            segments.append({"text": clean, "em": italic})
    return segments


def json_ld_script_body(data: Dict[str, Any]) -> str:
    """Serialise JSON-LD so it cannot break out of its ``<script>`` element.

    :param data: JSON-LD object.
    :return: JSON with ``<``, ``>`` and ``&`` written as ``\\u003c``, ``\\u003e``,
        ``\\u0026`` (still valid JSON, decoded back by any parser).
    :rtype: str
    """
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def is_indexable_host(host: Optional[str]) -> bool:
    """Whether pages served under this ``Host`` header may be indexed.

    :param host: Raw ``Host`` header (port allowed, IPv6 in brackets).
    :return: ``True`` only for the public site and local development hosts.
    :rtype: bool
    """
    name = (host or "").strip().lower()
    if name.startswith("["):
        name = name[1:].split("]", 1)[0]
    elif name.count(":") == 1:
        name = name.split(":", 1)[0]
    return name in INDEXABLE_HOSTS


def _as_list(value: Any) -> List[Any]:
    """Treat a keyword value (scalar, list or missing) as a list.

    :param value: Field value.
    :return: The values as a list.
    :rtype: List[Any]
    """
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _unique(items: Iterable[str]) -> List[str]:
    """Drop empty strings and case-insensitive duplicates, keeping order.

    :param items: Strings.
    :return: Unique non-empty strings.
    :rtype: List[str]
    """
    seen = set()
    out: List[str] = []
    for item in items:
        text = (item or "").strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Public record view
# ---------------------------------------------------------------------------


def public_metadata(
    metadata_source: "ElasticsearchService", source: Dict[str, Any], index_name: str
) -> "DocumentMetadata":
    """Public metadata of a catalogue record, plus the dates and document type.

    :param metadata_source: The search service (its ``_extract_metadata``
        applies the provider filters).
    :param source: Raw catalogue ``_source``.
    :param index_name: Catalogue index the record came from.
    :return: :class:`DocumentMetadata` with ``document_type`` and ``date_info``
        filled from :data:`EXTRA_PUBLIC_FIELDS` when ``_extract_metadata`` left
        them empty.
    :rtype: DocumentMetadata
    """
    meta = metadata_source._extract_metadata(source, index_name=index_name)
    extras: Dict[str, Any] = {}
    doc_type = source.get("document_type")
    if not meta.document_type and isinstance(doc_type, str) and doc_type.strip():
        extras["document_type"] = doc_type.strip()
    if not meta.date_info:
        dates = {
            key: source[key].strip()
            for key in _DATE_FIELDS
            if isinstance(source.get(key), str) and source[key].strip()
        }
        if dates:
            extras["date_info"] = dates
    return meta.model_copy(update=extras) if extras else meta


def credited_sources(
    metadata_source: "ElasticsearchService", meta: "DocumentMetadata", source: Dict[str, Any]
) -> List[str]:
    """Catalogues to credit on the page: ``pgp`` and/or ``ktiv``.

    Both the bibliography labels and the record's ``sources_present`` go
    through ``public_bibliography_source``, the single allowlist of providers
    that may be named; ``sources_present`` itself is never shown.

    :param metadata_source: The search service.
    :param meta: Public metadata of the record.
    :param source: Raw catalogue ``_source``.
    :return: Credited source keys in :data:`CREDITS` order.
    :rtype: List[str]
    """
    labels = {entry.get("source") for entry in meta.bibliography_entries or []}
    labels |= {metadata_source.public_bibliography_source(str(s)) for s in _as_list(source.get("sources_present"))}
    return [key for key in CREDITS if key in labels]


def _public_url(metadata_source: "ElasticsearchService", url: Any) -> Optional[str]:
    """An http(s) URL that is not on a blocked provider host.

    :param metadata_source: The search service (owner of the host blocklist).
    :param url: Candidate URL (image ``srcset`` width descriptors are dropped).
    :return: The URL, or ``None``.
    :rtype: Optional[str]
    """
    if not isinstance(url, str) or not url.strip():
        return None
    return metadata_source._public_bibliography_url(url.strip().split()[0])


def _date_view(date_info: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Visible date and ``temporalCoverage`` value.

    :param date_info: ``DocumentMetadata.date_info``.
    :return: ``(display text, schema.org temporalCoverage)``; either may be ``None``.
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    info = date_info or {}
    display = str(info.get("date_display") or info.get("display") or "").strip()
    century = str(info.get("century") or "").strip()
    if display and _ISO_DATE_RE.match(display):
        return display.replace("/", " – "), display
    if display:
        return display, century or display
    if century:
        return century, century
    return None, None


def _languages(meta: "DocumentMetadata") -> List[str]:
    """Language names of the record ("Other" dropped).

    :param meta: Public metadata.
    :return: Unique language names.
    :rtype: List[str]
    """
    names = re.split(r"[,;]", meta.language or "") + [str(x) for x in _as_list(meta.other_languages)]
    return [n for n in _unique(names) if n.casefold() not in {"other", "unknown"}]


def _document_types(meta: "DocumentMetadata") -> List[str]:
    """Document type labels.

    :param meta: Public metadata.
    :return: Capitalised, unique types.
    :rtype: List[str]
    """
    raw = [meta.document_type or ""] + [str(x) for x in _as_list(meta.document_types)]
    return [t[:1].upper() + t[1:] for t in _unique(raw)]


def _bibliography_view(
    metadata_source: "ElasticsearchService", entries: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Bibliography grouped like the document modal, plus JSON-LD citation strings.

    :param metadata_source: The search service (URL filter).
    :param entries: ``DocumentMetadata.bibliography_entries``.
    :return: ``(groups, citations)``; each group has ``key``, ``label``,
        ``name``, ``credit`` (a :data:`CREDITS` entry or ``None``) and ``entries``
        (``authors``, ``title``, ``year``, ``pages``, ``segments``, ``relations``, ``url``).
    :rtype: Tuple[List[Dict[str, Any]], List[str]]
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    citations: List[str] = []
    for entry in entries:
        title = clean_inline_text(entry.get("title"))
        raw_citation = str(entry.get("citation") or "")
        # A dict repr means the index had no citation text: it could carry raw fields.
        segments = [] if raw_citation.lstrip().startswith("{") else citation_segments(raw_citation)
        if not title and not segments:
            continue
        location = clean_inline_text(entry.get("location"))
        if location and not re.match(r"^\s*(p{1,2}\.|pages?\b|עמ)", location, re.I):
            location = f"p. {location}"
        view = {
            "authors": "; ".join(_unique(clean_inline_text(a) for a in _as_list(entry.get("authors")))),
            "title": title,
            "year": clean_inline_text(str(entry.get("year") or "")),
            "pages": location,
            "segments": segments,
            "relations": _unique(clean_inline_text(str(r)) for r in _as_list(entry.get("relations"))),
            "url": _public_url(metadata_source, entry.get("url")),
        }
        key = entry.get("source") if entry.get("source") in CREDITS else "other"
        grouped.setdefault(key, []).append(view)
        citations.append(title or " ".join(s["text"] for s in segments))
    groups = [
        {"key": key, "label": label, "name": name, "entries": grouped[key], "credit": CREDITS.get(key)}
        for key, label, name in _BIBLIOGRAPHY_GROUPS
        if key in grouped
    ]
    return groups, _unique(citations)


def _joins_view(joins: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Shelfmarks of the record's join (already reduced by ``_public_joins_data``).

    :param joins: ``DocumentMetadata.joins_data``.
    :return: ``{"main": str | None, "joined": [str]}`` or ``None``.
    :rtype: Optional[Dict[str, Any]]
    """
    if not joins:
        return None
    joined = _unique(str(j.get("shelfmark") or "") for j in joins.get("joinedManuscripts") or [])
    main = str(joins.get("mainShelfmark") or "").strip() or None
    return {"main": main, "joined": joined} if (main or joined) else None


def _reads_view(
    reads: Sequence[ConsensusRead], doc_id: str, image_urls: Sequence[str]
) -> List[Dict[str, Any]]:
    """Machine-reading blocks, one per image.

    :param reads: Visible consensus reads (cleaned agreed lines only).
    :param doc_id: Catalogue id of the page.
    :param image_urls: The record's public image URLs, for "Image N" labels.
    :return: Dicts with ``label``, ``lines`` (``number``, ``text``), ``n_agreed``,
        ``n_lines`` and ``read_url``.
    :rtype: List[Dict[str, Any]]
    """
    blocks: List[Dict[str, Any]] = []
    for read in reads:
        if read.image_url and read.image_url in image_urls:
            label = f"Image {image_urls.index(read.image_url) + 1}"
        elif read.ai_doc_id == doc_id:
            label = f"Image {read.image_index + 1}"
        else:
            label = "Image"
        blocks.append(
            {
                "label": label,
                "lines": [{"number": line.index + 1, "text": line.text} for line in read.lines],
                "n_agreed": read.n_agreed,
                "n_lines": read.n_lines,
                "read_url": read_url(read.ai_doc_id, read.image_index, vlm_model=read.vlm_model or None),
            }
        )
    return blocks


def build_meta_description(shelf_mark: str, place: str, description: str, has_reads: bool = False) -> str:
    """The ``<meta name="description">`` text (at most 160 characters), from catalogue facts only.

    Records with a real description are described by that description; a
    short real note gets a sentence around it; empty or placeholder
    descriptions get a sentence built from the shelfmark and holding place.
    Machine-reading text is never quoted here: head metadata is outside the
    ``data-nosnippet`` section and would become the search snippet.  When
    readings exist, the fallback sentence only says that one is on the page.

    :param shelf_mark: Display shelfmark.
    :param place: "Institution, collection" (may be empty).
    :param description: Cleaned one-line description (may be empty).
    :param has_reads: Whether the page shows a machine reading.
    :return: Meta description.
    :rtype: str
    """
    if description and not is_thin_description(description):
        return shorten(description, META_DESCRIPTION_MAX)
    held = f" held at {place}" if place else ""
    if description and not is_placeholder_description(description):
        # A short but real note ("Responsum", "Marriage contract") still says something.
        return shorten(
            f"{shelf_mark}: {description.rstrip('. ')}. A Cairo Genizah manuscript fragment{held}.",
            META_DESCRIPTION_MAX,
        )
    offer = (
        "Images, catalogue record and an unchecked machine reading."
        if has_reads
        else f"Images and catalogue record on {SITE_NAME}."
    )
    return shorten(f"{shelf_mark}: a Cairo Genizah manuscript fragment{held}. {offer}", META_DESCRIPTION_MAX)


def is_placeholder_description(description: str) -> bool:
    """Whether a description is one of the catalogue's stock placeholders.

    :param description: Cleaned description.
    :return: ``True`` for e.g. "קטע גניזה", "Unidentified", "A few words. Blank verso."
    :rtype: bool
    """
    return _normalise_phrase(description) in _PLACEHOLDERS


def _normalise_phrase(text: str) -> str:
    """Case-fold, turn punctuation into spaces and collapse whitespace.

    :param text: Description or placeholder.
    :return: Normalised text.
    :rtype: str
    """
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


_PLACEHOLDERS = frozenset(_normalise_phrase(p) for p in PLACEHOLDER_DESCRIPTIONS)


def build_page_context(
    metadata_source: "ElasticsearchService",
    source: Dict[str, Any],
    index_name: str,
    reads: Sequence[ConsensusRead],
) -> Dict[str, Any]:
    """Template context of one fragment page.

    :param metadata_source: The search service.
    :param source: Raw catalogue ``_source`` (looked up by exact ``doc_id``).
    :param index_name: Catalogue index name.
    :param reads: Visible consensus reads of the fragment.
    :return: Everything ``fragment.html`` renders, already public and cleaned.
    :rtype: Dict[str, Any]
    """
    meta = public_metadata(metadata_source, source, index_name)
    doc_id = meta.doc_id
    shelf_mark = " ".join(str(meta.shelf_mark or meta.shelfmark or doc_id).split())
    institution = clean_inline_text(meta.institution or meta.repository)
    collection = clean_inline_text(meta.collection)
    place = ", ".join(_unique([institution, collection]))
    description = clean_block_text(meta.description)
    description_line = " ".join(description.split())
    images = [
        url
        for url in _unique(_public_url(metadata_source, u) or "" for u in _as_list(meta.image_urls))
    ][:MAX_PAGE_IMAGES]
    if not images:
        images = [u for u in [_public_url(metadata_source, meta.actual_image_url)] if u]
    date_text, temporal = _date_view(meta.date_info)
    languages = _languages(meta)
    doc_types = _document_types(meta)
    bib_groups, citations = _bibliography_view(metadata_source, meta.bibliography_entries or [])
    credits = [CREDITS[key] for key in credited_sources(metadata_source, meta, source)]
    transcription = (meta.transcription_full_text or "").strip()
    translation = (meta.translation_full_text or "").strip()
    meta_description = build_meta_description(shelf_mark, place, description_line, has_reads=bool(reads))
    canonical = fragment_url(doc_id)

    # The meta description stays catalogue-only (search engines show it as the
    # snippet, and machine text is data-nosnippet). JSON-LD keywords are never
    # shown, so uncatalogued records also carry their consensus search terms.
    keywords = list(doc_types)
    if reads and is_uncatalogued(source):
        keywords += consensus_terms(list(reads), limit=KEYWORD_TERMS_MAX)
    ld: Dict[str, Any] = {
        "@context": "https://schema.org",
        # ArchiveComponent makes holdingArchive valid; Manuscript keeps the type.
        "@type": ["Manuscript", "ArchiveComponent"],
        "name": shelf_mark,
        "identifier": shelf_mark,
        "url": canonical,
        "description": shorten(description_line, JSON_LD_DESCRIPTION_MAX) if description_line else meta_description,
        "isPartOf": {"@type": "Collection", "name": "Cairo Genizah"},
    }
    if images:
        ld["image"] = {"@type": "ImageObject", "contentUrl": images[0]}
    if institution:
        ld["holdingArchive"] = {"@type": "ArchiveOrganization", "name": institution}
    if languages:
        ld["inLanguage"] = languages if len(languages) > 1 else languages[0]
    if temporal:
        ld["temporalCoverage"] = temporal
    if keywords:
        ld["keywords"] = _unique(keywords)
    if citations:
        ld["citation"] = citations[:50]

    return {
        "doc_id": doc_id,
        "shelf_mark": shelf_mark,
        "page_title": f"{shelf_mark} · Cairo Genizah fragment · {SITE_NAME}",
        "meta_description": meta_description,
        "canonical_url": canonical,
        "open_in_app_url": "/?doc=" + quote(doc_id, safe=""),
        "institution": institution,
        "collection": collection,
        "date_text": date_text,
        "languages": languages,
        "document_types": doc_types,
        "dimensions": clean_inline_text(meta.dimensions),
        "description": description,
        "images": images,
        "transcription": transcription,
        "transcription_dir": text_direction(transcription),
        "translation": translation,
        "translation_dir": text_direction(translation),
        "bibliography_groups": bib_groups,
        "joins": _joins_view(meta.joins_data),
        "reads": _reads_view(reads, doc_id, images),
        "machine_reading_label": MACHINE_READING_LABEL,
        "credits": credits,
        "json_ld": json_ld_script_body(ld),
        "site_name": SITE_NAME,
        "canonical_base": CANONICAL_BASE,
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class FragmentResponse:
    """A rendered page with its status and response headers.

    :param status_code: HTTP status.
    :param body: HTML (empty for redirects).
    :param headers: Response headers (``Content-Type`` is set by the route).
    """

    status_code: int
    body: str
    headers: Dict[str, str] = field(default_factory=dict)


class PageCache:
    """Small thread-safe LRU cache with a TTL and a total size budget.

    :param max_entries: Maximum number of pages kept.
    :param max_bytes: Maximum total size of the cached bodies (characters).
    :param ttl_seconds: Lifetime of an entry.
    :param clock: Monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        max_bytes: int = CACHE_MAX_BYTES,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty cache.

        :param max_entries: Maximum number of pages kept.
        :param max_bytes: Maximum total body size.
        :param ttl_seconds: Entry lifetime in seconds.
        :param clock: Monotonic time source.
        """
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Tuple[float, FragmentResponse]]" = OrderedDict()
        self._size = 0

    def get(self, key: str) -> Optional[FragmentResponse]:
        """Cached response for a key, if present and fresh.

        :param key: Requested document id.
        :return: The response, or ``None``.
        :rtype: Optional[FragmentResponse]
        """
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            if self._clock() >= item[0]:
                self._remove(key)
                return None
            self._entries.move_to_end(key)
            return item[1]

    def put(self, key: str, response: FragmentResponse) -> None:
        """Store a response, evicting the least recently used ones over budget.

        :param key: Requested document id.
        :param response: Response to keep.
        """
        size = len(response.body)
        if size > self.max_bytes or self.max_entries <= 0:
            return
        with self._lock:
            if key in self._entries:
                self._remove(key)
            self._entries[key] = (self._clock() + self.ttl_seconds, response)
            self._size += size
            while len(self._entries) > self.max_entries or self._size > self.max_bytes:
                self._remove(next(iter(self._entries)))

    def __len__(self) -> int:
        """Number of cached entries.

        :return: Entry count.
        :rtype: int
        """
        return len(self._entries)

    def _remove(self, key: str) -> None:
        """Drop one entry (caller holds the lock).

        :param key: Key to drop.
        """
        _, response = self._entries.pop(key)
        self._size -= len(response.body)


class FragmentPageService:
    """Looks up a catalogue record by exact id and renders its public page.

    :param es: Elasticsearch client (read-only ``search`` calls).
    :param index_name: Catalogue index (``ELASTICSEARCH_INDEX``); fixed, never
        taken from the request.
    :param metadata_source: The search service, whose ``_extract_metadata``
        builds the public metadata.
    :param consensus_service: Source of machine readings, or ``None`` to omit them.
    :param cache: Rendered-page cache (a default one is created when omitted).
    """

    def __init__(
        self,
        es: Any,
        index_name: str,
        metadata_source: "ElasticsearchService",
        consensus_service: Optional["ConsensusService"] = None,
        cache: Optional[PageCache] = None,
    ) -> None:
        """Set up the Jinja environment and the cache (no ES call here).

        :param es: Elasticsearch client.
        :param index_name: Catalogue index name.
        :param metadata_source: The search service.
        :param consensus_service: Machine-reading source, or ``None``.
        :param cache: Page cache, or ``None`` for the default.
        """
        self.es = es
        self.index_name = index_name
        self.metadata_source = metadata_source
        self.consensus_service = consensus_service
        self.cache = cache if cache is not None else PageCache()
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, doc_id: str, host: Optional[str] = None) -> FragmentResponse:
        """The response for ``/fragment/<doc_id>``.

        :param doc_id: Requested id, already percent-decoded.
        :param host: Request ``Host`` header (decides ``X-Robots-Tag``).
        :return: 200 page, 301 redirect (old Oxford id), 404 or 503 page.
        :rtype: FragmentResponse
        """
        response = self.cache.get(doc_id)
        if response is None:
            try:
                response, cacheable = self._build(doc_id)
            except (ApiError, TransportError) as exc:
                logger.warning("Fragment page %r: Elasticsearch unavailable (%s)", doc_id, exc)
                response, cacheable = self._unavailable(doc_id), False
            if cacheable:
                self.cache.put(doc_id, response)
        headers = dict(response.headers)
        if not is_indexable_host(host):
            headers["X-Robots-Tag"] = "noindex"
        return FragmentResponse(response.status_code, response.body, headers)

    def lookup(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Catalogue ``_source`` for an exact ``doc_id`` (no fuzzy fallback).

        :param doc_id: Catalogue id.
        :return: The record without heavy fields, or ``None`` when absent.
        :rtype: Optional[Dict[str, Any]]
        :raises ApiError: On Elasticsearch errors (including a missing index).
        :raises TransportError: When Elasticsearch cannot be reached.
        """
        if not doc_id or len(doc_id) > MAX_DOC_ID_CHARS:
            return None
        response = self.es.search(
            index=self.index_name,
            query={"term": {"doc_id": doc_id}},
            size=1,
            source_excludes=CATALOGUE_SOURCE_EXCLUDES,
        )
        hits = response["hits"]["hits"]
        if not hits or hits[0].get("_source", {}).get("doc_id") != doc_id:
            return None
        return hits[0]["_source"]

    def _build(self, doc_id: str) -> Tuple[FragmentResponse, bool]:
        """Render the page (or redirect / 404) for one id.

        :param doc_id: Requested id.
        :return: ``(response, cacheable)``; a page rendered without its machine
            readings because the AI index failed is not cached.
        :rtype: Tuple[FragmentResponse, bool]
        """
        source = self.lookup(doc_id)
        if source is None:
            mapped = to_catalogue_doc_id(doc_id)
            if mapped != doc_id and self.lookup(mapped) is not None:
                return (
                    FragmentResponse(301, "", {"Location": fragment_path(mapped), "Cache-Control": PAGE_CACHE_CONTROL}),
                    True,
                )
            return self._not_found(doc_id), True
        reads, reads_ok = self._reads(doc_id)
        context = build_page_context(self.metadata_source, source, self.index_name, reads)
        body = self.env.get_template("fragment.html").render(**context)
        return FragmentResponse(200, body, {"Cache-Control": PAGE_CACHE_CONTROL}), reads_ok

    def _reads(self, doc_id: str) -> Tuple[List[ConsensusRead], bool]:
        """Machine readings of the fragment; an AI-index failure omits them.

        Readings are also omitted when no consensus service is wired or the
        ``AI_TRANSCRIPTIONS_ENABLED`` master switch is off (the ``/read``
        viewer its links point to is then switched off too).

        :param doc_id: Catalogue id.
        :return: ``(reads, ok)``; ``ok`` is ``False`` when the lookup failed.
        :rtype: Tuple[List[ConsensusRead], bool]
        """
        if self.consensus_service is None or not ai_feature_enabled():
            return [], True
        try:
            return self.consensus_service.reads_for_document(doc_id), True
        except (ApiError, TransportError) as exc:
            logger.warning("Fragment page %r: machine readings unavailable (%s)", doc_id, exc)
            return [], False

    def _not_found(self, doc_id: str) -> FragmentResponse:
        """404 page with links back into the site.

        :param doc_id: Requested id (echoed, autoescaped).
        :return: 404 response.
        :rtype: FragmentResponse
        """
        body = self.env.get_template("fragment_404.html").render(
            doc_id=doc_id, site_name=SITE_NAME, page_title=f"Fragment not found · {SITE_NAME}"
        )
        return FragmentResponse(404, body, {"Cache-Control": NOT_FOUND_CACHE_CONTROL, "X-Robots-Tag": "noindex"})

    def _unavailable(self, doc_id: str) -> FragmentResponse:
        """503 page shown while Elasticsearch is unreachable.

        :param doc_id: Requested id.
        :return: 503 response with ``Retry-After``.
        :rtype: FragmentResponse
        """
        body = self.env.get_template("fragment_503.html").render(
            doc_id=doc_id, site_name=SITE_NAME, page_title=f"Temporarily unavailable · {SITE_NAME}"
        )
        return FragmentResponse(
            503,
            body,
            {"Cache-Control": "no-store", "Retry-After": str(RETRY_AFTER_SECONDS), "X-Robots-Tag": "noindex"},
        )
