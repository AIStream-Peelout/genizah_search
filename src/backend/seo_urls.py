"""Canonical public URLs for crawlable pages (fragment pages and sitemaps).

The public site is always ``https://cairogenizah.ai``.  The backend is also
reachable on ``api.cairogenizah.ai`` and uvicorn ignores ``X-Forwarded-*``
headers, so canonical links are never derived from the request: they are
built from :data:`CANONICAL_BASE` here.

Document ids come straight from the catalogue and some contain characters
that are unsafe in a URL path (``'°'``, ``'='``, ``'*'``, ``'?'``, zero-width
spaces...).  Every id is therefore percent-encoded with nothing kept "safe"
(``/`` included), and ids that cannot round-trip reliably through crawlers
are kept out of the sitemaps by :func:`is_sitemap_eligible`.
"""

from __future__ import annotations

import unicodedata
from urllib.parse import quote

CANONICAL_BASE = "https://cairogenizah.ai"
FRAGMENT_PREFIX = "/fragment/"
SITEMAP_INDEX_PATH = "/sitemap-fragments.xml"

# '?' starts a query string and '*' is a glob in most crawler tooling; both
# appear only in malformed catalogue ids (e.g. 'Heidelberg_Cod_Heid_Or_??').
_SITEMAP_FORBIDDEN_CHARS = frozenset("?*")
# Cc = control, Cf = format (zero-width space/joiners, bidi marks, BOM, soft
# hyphen), Cs = lone surrogates, Z* = space/line/paragraph separators.
_SITEMAP_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zs", "Zl", "Zp"})


def fragment_path(doc_id: str) -> str:
    """Site-relative path of a fragment page.

    :param doc_id: Catalogue document id (``doc_id`` / ``_id`` in the merged index).
    :return: ``/fragment/`` followed by the fully percent-encoded id.
    :rtype: str
    """
    return FRAGMENT_PREFIX + quote(doc_id, safe="")


def fragment_url(doc_id: str) -> str:
    """Absolute canonical URL of a fragment page.

    :param doc_id: Catalogue document id.
    :return: :data:`CANONICAL_BASE` plus :func:`fragment_path`.
    :rtype: str
    """
    return CANONICAL_BASE + fragment_path(doc_id)


def sitemap_page_path(page: int) -> str:
    """Site-relative path of one fragment sitemap file.

    :param page: 1-based sitemap file number.
    :return: Path such as ``/sitemap-fragments-1.xml``.
    :rtype: str
    """
    return f"/sitemap-fragments-{page}.xml"


def is_sitemap_eligible(doc_id: str) -> bool:
    """Whether a document id may be listed in a sitemap.

    Excludes empty ids and ids containing ``'?'``, ``'*'``, control or
    invisible formatting characters (zero-width space, bidi marks, BOM) or
    any whitespace.  Such ids still have pages; they are just not advertised.

    :param doc_id: Catalogue document id.
    :return: ``True`` when the id is safe to publish as a sitemap URL.
    :rtype: bool
    """
    if not doc_id:
        return False
    for ch in doc_id:
        if ch in _SITEMAP_FORBIDDEN_CHARS or ch.isspace():
            return False
        if unicodedata.category(ch) in _SITEMAP_FORBIDDEN_CATEGORIES:
            return False
    return True
