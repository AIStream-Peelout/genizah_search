"""Direct "View on <institution>" links for a catalogue record (pure, stdlib only).

One place builds the links that the result cards, the document modal and the
server-rendered ``/fragment/<doc_id>`` page show, so the three never drift
(``docs/planned_features/source_links/2026-10-05_source_links_handoff.md``).
Every link is derived from fields already in the merged index:

* CUDL item pages from the display shelfmark of Cambridge-held records (the
  handoff's ``cudl_id_rule.py``, ported verbatim below);
* the Bodleian catalogue from ``bodleian_catalogue_url``;
* a Manchester Digital Collections (Luna) search from the shelfmark;
* OPenn pages for Penn "Halper N" shelfmarks;
* KTIV from the system number in ``ktiv_iiif_manifest_url`` (switch:
  :data:`KTIV_LINKS_ENABLED`; the item-page URL shape was confirmed by hand in a
  browser on 2026-10-12);
* one Princeton Geniza Project page per ``tei_metadata.pgpids`` entry.

When the merge pipeline starts indexing a ``source_links`` list, that list is
used instead of the derivation.  Either way, the blocked-host filter runs last:
the unattributed provider (:data:`BLOCKED_HOSTS`) is never linked.  This module
also owns the host blocklist used by the bibliography URL filter and the
``/book-info`` link labels, so each lives in one place.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

# Hosts (and their subdomains) that are never linked or named on the public site,
# e.g. the old fjms.genizah.org and fgp.genizah.org. NLI/KTIV pages are fine.
BLOCKED_HOSTS: Tuple[str, ...] = ("genizah.org", "jewishmanuscripts.org")

# KTIV item pages: Isaac confirmed the itempage URL shape in a browser on
# 2026-10-12 (T-S 8J41.4 opens its NLI record). Set False to hide KTIV links.
KTIV_LINKS_ENABLED = True

LINK_KINDS: Tuple[str, ...] = ("item", "catalogue", "search")
# Public label of each link source (also used for indexed links without a label).
SOURCE_LABELS: Dict[str, str] = {
    "cudl": "View on Cambridge Digital Library",
    "bodleian": "View in the Bodleian catalogue",
    "manchester": "Search Manchester Digital Collections",
    "penn": "View on OPenn (Penn Libraries)",
    "ktiv": "View on KTIV (National Library of Israel)",
    "pgp": "View on Princeton Geniza Project",
}
# Labels of known hosts for the ``/book-info`` direct link (bibliography works).
BOOK_LINK_LABELS: Dict[str, str] = {
    "jstor.org": "View on JSTOR",
    "doi.org": "View via DOI",
    "academia.edu": "View on Academia.edu",
    "persee.fr": "View on Persée",
    "lib.cam.ac.uk": "View at Cambridge Library",
    "nli.org.il": "View at National Library of Israel",
    "catalog.princeton.edu": "View in Princeton catalog",
}

CUDL_VIEW_URL = "https://cudl.lib.cam.ac.uk/view/{}"
PGP_DOCUMENT_URL = "https://geniza.princeton.edu/en/documents/{}/"
OPENN_HALPER_URL = "https://openn.library.upenn.edu/Data/0002/html/h{}.html"
KTIV_ITEM_URL = (
    "https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage"
    "?vid=KTIV&scope=KTIV&docId=PNX_MANUSCRIPTS{}"
)
LUNA_SEARCH_URL = (
    "https://luna.manchester.ac.uk/luna/servlet/view/search"
    "?q={}+LIMIT%3AManchesterDev~95~&sort=reference_number%2Cdate_created"
)

_PGPID_RE = re.compile(r"^\d+$")
_KTIV_SYSNUM_RE = re.compile(r"DOCID/(?:PNX_MANUSCRIPTS)?(\d{15,})")
_LUNA_RE = re.compile(
    r"^(?:Manchester:\s*|JRL\s+(?:Genizah\s+)?|The University of Manchester Library, Manchester, England Ms\.\s*)"
    r"(A|B|C|L|P|G|AF|Ar\.)\s*(\d+[a-z]?)$"
)
_HALPER_RE = re.compile(r"(?:^|CAJS:\s*|Ms\.\s*)Halper\s+(\d+)$")

# ---------------------------------------------------------------------------
# URL safety and labels
# ---------------------------------------------------------------------------


def is_blocked_host(host: Optional[str]) -> bool:
    """Whether a host is one of :data:`BLOCKED_HOSTS` or a subdomain of one.

    :param host: Host name (any case).
    :return: ``True`` for e.g. ``genizah.org``, ``fgp.genizah.org``; ``False``
        for ``genizah.bodleian.ox.ac.uk`` or ``cairogenizah.org``.
    :rtype: bool
    """
    name = (host or "").strip().lower().rstrip(".")
    return any(name == blocked or name.endswith("." + blocked) for blocked in BLOCKED_HOSTS)


def public_url(url: Any) -> Optional[str]:
    """Return ``url`` only when it is safe to publish.

    :param url: Candidate URL.
    :return: The stripped URL, or ``None`` when empty, not http(s), without a
        host, or on a blocked host.
    :rtype: Optional[str]
    """
    if not url or not isinstance(url, str):
        return None
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host or is_blocked_host(host):
        return None
    return url.strip()


def _loose_host(url: str) -> str:
    """Host of a URL that may lack a scheme ("www.jstor.org/stable/1").

    :param url: URL or scheme-less address.
    :return: Lower-cased host without a leading ``www.`` (port kept).
    :rtype: str
    """
    return re.split(r"[/?#]", url.split("//")[-1], maxsplit=1)[0].lower().removeprefix("www.")


def is_blocked_url(url: Optional[str]) -> bool:
    """Whether a URL (with or without scheme) points at a blocked host.

    :param url: Candidate URL.
    :return: ``True`` when its host is one of :data:`BLOCKED_HOSTS` or a subdomain.
    :rtype: bool
    """
    return bool(url) and is_blocked_host(_loose_host(url).split(":")[0])


def link_label(url: Optional[str]) -> Optional[str]:
    """Label of a ``/book-info`` direct link ("View on JSTOR", "View at <host>").

    :param url: Direct URL of a cited work.
    :return: The label, or ``None`` when there is no URL or it is on a blocked host.
    :rtype: Optional[str]
    """
    if not url or is_blocked_url(url):
        return None
    host = _loose_host(url)
    return BOOK_LINK_LABELS.get(host, f"View at {host}")


# ---------------------------------------------------------------------------
# CUDL shelfmark rule (cudl_id_rule.py, ported verbatim)
# ---------------------------------------------------------------------------

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
_WORDS = {"NS": "NS", "AS": "AS", "AR": "AR", "MISC": "MISC"}
_CUDL_PREFIX = re.compile(
    r"^(Cambridge CUL:\s*|Cambridge University Library, Cambridge, England Ms\.\s*|CUL\s+|"
    r"Cambridge Lewis-Gibson:\s*|Cambridge University Library, Cambridge, England Ms\.\s*|"
    r"Mosseri, Jacques, Paris, France Ms\.\s*)",
    re.I,
)
_LG_CATEGORIES = {"Ar": "ARABIC", "Misc": "MISC", "Glass": "GLASS", "Bib": "BIBLE", "Lit": "LITURGY", "Talm": "TALMUD"}
# Known rule exceptions (handoff, "CUDL rule details"; confirmed against the
# CUDL URLs in PGP's fragments.csv).  T-S Ar.30.184.N items have "-P-" ids
# (T-S Ar.30.184.14 -> MS-TS-AR-00030-00184-P-00014); any other Ar.30.184.
# shelfmark (suffixed, ranged) gets no link rather than a guessed id.
_CUDL_AR_30_184_ITEM = re.compile(r"^\s*Ar\.?\s*30\.184\.(\d+)\s*$")
_CUDL_AR_30_184 = re.compile(r"\bAr\.?\s*30\.184\.")
CUDL_AR_30_184_ID = "MS-TS-AR-00030-00184-P-{}"
# Third-level sub-items of these shelfmarks live on their parent's page
# (T-S 8J4.3.1 -> the page of T-S 8J4.3); PGP links every one it has that
# way.  Other third-level shelfmarks (T-S K25.240.40, T-S Misc.27.4.11) have
# their own page: the plain rule.
_CUDL_PARENT_LINKED = re.compile(
    r"^\s*(8J4\.3|10J27\.3|13J4\.15|Misc\.?\s*28\.79|Ar\.?\s*40\.142|Ar\.?\s*54\.20)\.\d+[a-z]?\s*$",
    re.I,
)


def _cudl_series(shelf_mark: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a Cambridge shelfmark into its CUDL series and the rest.

    :param shelf_mark: Display shelfmark.
    :return: ``(series, rest)`` such as ``("TS", " 10J2.11")``, or ``(None, None)``.
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    s = _CUDL_PREFIX.sub("", shelf_mark.strip())
    if re.match(r"T-S\b", s):
        return "TS", s[3:]
    if re.match(r"Or\.?\s*\d", s):
        return "OR", re.sub(r"^Or\.?", "", s)
    if re.match(r"Add\.?\s*\d", s):
        return "ADD", re.sub(r"^Add\.?", "", s)
    if re.match(r"Moss(\.|eri)?\b", s, re.I):
        return "MOSSERI", re.sub(r"^Moss(\.|eri)?", "", s, flags=re.I)
    if re.match(r"L-G\b", s):
        return "LG", s[3:]
    if re.match(r"^(VI|VII|VIII|IX|X|I|II|III|IV|V)[a-z]?\s", s) and "Mosseri" in shelf_mark:
        return "MOSSERI", s
    return None, None


def cudl_from_shelfmark(shelf_mark: str) -> Optional[str]:
    """CUDL item id of a Cambridge shelfmark (the handoff's rule, unchanged).

    :param shelf_mark: Display shelfmark (not the canonical id, which drops "(n)").
    :return: E.g. ``"MS-TS-00010-J-00002-00011"`` for "T-S 10J2.11", or ``None``
        for joins, ranges and shelfmarks the rule does not cover.
    :rtype: Optional[str]
    """
    ser, rest = _cudl_series(shelf_mark)
    if not ser:
        return None
    if re.search(r"[–\-]\s*\d|\s(and|&)\s|;|,\s*\d+\s*-|\+", rest.replace("T-S", "")) and ser != "MOSSERI":
        return None  # ranges/joins
    raw = re.findall(r"\d+|[A-Za-z]+", rest)
    out: List[Optional[str]] = []
    if ser == "MOSSERI":
        if not raw:
            return None
        out.append(raw[0].upper())
        raw = raw[1:]
        if any(not x.isdigit() and len(x) > 1 for x in raw):
            return None
        out += [x.zfill(5) if x.isdigit() else x.upper() for x in raw]
        return "MS-MOSSERI-" + "-".join(out)
    if ser == "LG":
        cat = _LG_CATEGORIES.get(raw[0]) if raw else None
        if not cat:
            return None
        for x in raw[1:]:
            out.append(str(_ROMAN[x]).zfill(5) if x in _ROMAN else (x.zfill(5) if x.isdigit() else None))
        if None in out:
            return None
        return "MS-LG-" + cat + "".join("-" + o for o in out)
    for x in raw:
        if x.isdigit():
            out.append(x.zfill(5))
        elif x.upper() in _WORDS:
            out.append(_WORDS[x.upper()])
        elif x == "Ka":
            out.append("KA")
        elif len(x) <= 2:
            out += list(x.upper())
        else:
            return None
    return "MS-" + ser + "-" + "-".join(out)


def cudl_item_id(shelf_mark: str) -> Optional[str]:
    """CUDL id for a link: the rule, with its known exceptions applied.

    :param shelf_mark: Display shelfmark of a Cambridge-held record.
    :return: CUDL item id ("-P-" id for T-S Ar.30.184.N, the parent's id for
        sub-items of the parent-linked boxes), or ``None`` when there is no
        safe id.
    :rtype: Optional[str]
    """
    ser, rest = _cudl_series(shelf_mark)
    if ser == "TS":
        ar_item = _CUDL_AR_30_184_ITEM.match(rest or "")
        if ar_item:
            return CUDL_AR_30_184_ID.format(ar_item.group(1).zfill(5))
        if _CUDL_AR_30_184.search(rest or ""):
            return None
        parent = _CUDL_PARENT_LINKED.match(rest or "")
        if parent:
            return cudl_from_shelfmark("T-S " + parent.group(1))
    return cudl_from_shelfmark(shelf_mark)


# ---------------------------------------------------------------------------
# Link derivation
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    """Treat a field value (scalar, list or missing) as a list.

    :param value: Field value.
    :return: The values as a list.
    :rtype: List[Any]
    """
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _first_text(source: Dict[str, Any], field: str) -> str:
    """A URL field read at the top level, else under ``tei_metadata``.

    :param source: Catalogue ``_source``.
    :param field: Field name.
    :return: The stripped value, or ``""``.
    :rtype: str
    """
    tei = source.get("tei_metadata") if isinstance(source.get("tei_metadata"), dict) else {}
    for value in (source.get(field), tei.get(field)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _link(source: str, url: str) -> Dict[str, str]:
    """One link record with the source's public label and kind.

    :param source: Link source key (a :data:`SOURCE_LABELS` key).
    :param url: Target URL.
    :return: ``{"source", "label", "url", "kind"}``.
    :rtype: Dict[str, str]
    """
    kind = {"bodleian": "catalogue", "manchester": "search"}.get(source, "item")
    return {"source": source, "label": SOURCE_LABELS[source], "url": url, "kind": kind}


def ktiv_sysnum(source: Dict[str, Any]) -> Optional[str]:
    """NLI system number of the record's KTIV manifest.

    :param source: Catalogue ``_source``.
    :return: The system number (15+ digits), or ``None``.
    :rtype: Optional[str]
    """
    match = _KTIV_SYSNUM_RE.search(_first_text(source, "ktiv_iiif_manifest_url"))
    return match.group(1) if match else None


def _holder_links(source: Dict[str, Any], shelf_mark: str) -> List[Dict[str, str]]:
    """Holding-institution links (CUDL, Bodleian, Manchester, Penn).

    :param source: Catalogue ``_source``.
    :param shelf_mark: Display shelfmark.
    :return: Links in display order: CUDL, Bodleian, Manchester, Penn.
    :rtype: List[Dict[str, str]]
    """
    links: List[Dict[str, str]] = []
    # The holder is the canonical id prefix: ``institution`` is free text.
    holder = str(source.get("canonical_id") or source.get("doc_id") or "").split("_")[0]
    cudl = cudl_item_id(shelf_mark) if holder == "Cambridge" and shelf_mark else None
    if cudl:
        links.append(_link("cudl", CUDL_VIEW_URL.format(cudl)))
    bodleian = _first_text(source, "bodleian_catalogue_url")
    if bodleian:
        links.append(_link("bodleian", bodleian))
    luna = _LUNA_RE.match(shelf_mark)
    if luna:
        query = quote_plus('"%s %s"' % (luna.group(1), luna.group(2))).replace("%2B", "+")
        links.append(_link("manchester", LUNA_SEARCH_URL.format(query)))
    halper = _HALPER_RE.search(shelf_mark)
    if halper:
        links.append(_link("penn", OPENN_HALPER_URL.format(halper.group(1))))
    return links


def _derived_links(source: Dict[str, Any]) -> List[Dict[str, str]]:
    """Links derived from the record's own fields: holding institution, KTIV, PGP.

    :param source: Catalogue ``_source``.
    :return: Links in display order (unfiltered).
    :rtype: List[Dict[str, str]]
    """
    shelf_mark = str(source.get("shelf_mark") or "").strip()
    links = _holder_links(source, shelf_mark)
    sysnum = ktiv_sysnum(source)
    if sysnum:
        links.append(_link("ktiv", KTIV_ITEM_URL.format(sysnum)))
    tei = source.get("tei_metadata") if isinstance(source.get("tei_metadata"), dict) else {}
    for pgpid in _as_list(tei.get("pgpids")):
        pgpid = str(pgpid).strip()
        if _PGPID_RE.match(pgpid):
            links.append(_link("pgp", PGP_DOCUMENT_URL.format(pgpid)))
    return links


def _indexed_links(entries: Iterable[Any]) -> List[Dict[str, str]]:
    """Normalise a pipeline-indexed ``source_links`` list (labels and kinds filled).

    :param entries: Raw ``source_links`` entries.
    :return: Links with ``source``, ``label``, ``url`` and ``kind`` (unfiltered).
    :rtype: List[Dict[str, str]]
    """
    links: List[Dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            continue
        url = entry["url"].strip()
        source = str(entry.get("source") or "other").strip().lower()
        label = str(entry.get("label") or "").strip() or SOURCE_LABELS.get(source) or link_label(url) or ""
        kind = entry.get("kind") if entry.get("kind") in LINK_KINDS else "item"
        links.append({"source": source, "label": label, "url": url, "kind": kind})
    return links


def _public_links(links: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop disabled KTIV links, duplicate URLs and unsafe or blocked URLs.

    :param links: Candidate links in display order.
    :return: The links that may be published, order kept.
    :rtype: List[Dict[str, str]]
    """
    seen = set()
    out: List[Dict[str, str]] = []
    for link in links:
        if link["source"] == "ktiv" and not KTIV_LINKS_ENABLED:
            continue
        url = public_url(link["url"])
        if url is None or url in seen:
            continue
        seen.add(url)
        out.append({**link, "url": url})
    return out


def build_source_links(source: Dict[str, Any]) -> List[Dict[str, str]]:
    """Public "View at the source" links of a catalogue record.

    :param source: Catalogue ``_source`` (needs ``shelf_mark``, ``canonical_id``
        or ``doc_id``, ``tei_metadata.pgpids``, ``bodleian_catalogue_url`` and
        ``ktiv_iiif_manifest_url`` where present).
    :return: ``[{"source", "label", "url", "kind"}]`` with ``kind`` one of
        ``item`` / ``catalogue`` / ``search``: holding institution first, then
        KTIV, then PGP (one per pgpid); de-duplicated by URL.  An indexed
        ``source_links`` list (merge pipeline) replaces the derivation.  No URL
        on a :data:`BLOCKED_HOSTS` host ever survives.
    :rtype: List[Dict[str, str]]
    """
    indexed = source.get("source_links")
    links = _indexed_links(indexed) if isinstance(indexed, list) else _derived_links(source)
    return _public_links(links)
