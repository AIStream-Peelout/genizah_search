"""Read-only lookup of one cited work: its publication details plus every
Genizah fragment in the index whose nested ``bibliography`` array cites it.

Powers ``GET /bibliography/work`` (src/backend/app.py) and the document
modal's "work detail" card (src/frontend/src/core_results/BibliographyDetail.jsx).
All provider attribution is delegated to
:meth:`~src.backend.search_service.ElasticsearchService.public_bibliography_source`
so this module never re-implements, and can never accidentally loosen, the
public-attribution allowlist (only ``ktiv``/``pgp`` may ever be named).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - import-time only, avoids a cycle
    from src.backend.search_service import ElasticsearchService

logger = logging.getLogger(__name__)

# Title/description snippet shown per fragment row.
SNIPPET_MAX_LEN = 160
# ``bibliography.title.keyword`` is ``ignore_above: 512`` in the ES mapping,
# so a keyword ``term`` query can never match a longer stored title.
TITLE_KEYWORD_IGNORE_ABOVE = 512

INNER_HITS_NAME = "bib_match"
_INNER_HITS_FIELDS = ("authors", "year", "location", "relations", "source")
_INNER_HITS_SOURCE = [f"bibliography.{field}" for field in _INNER_HITS_FIELDS]
_DOC_SOURCE_FIELDS = ["doc_id", "shelf_mark", "shelfmark", "classmark", "title", "description"]


class BibliographyWorkFragment(BaseModel):
    """One Genizah fragment whose bibliography cites the looked-up work."""

    doc_id: str
    shelf_mark: Optional[str] = None
    snippet: Optional[str] = Field(None, description="Title or description, truncated to 160 chars")
    location: Optional[str] = Field(None, description="Pages, from the matching bibliography entry")
    relations: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(None, description='"ktiv" | "pgp" | null for an unattributed provider')


class BibliographyWorkResponse(BaseModel):
    """Response of ``GET /bibliography/work``."""

    title: str
    authors: List[str] = Field(default_factory=list, description="Union across matching entries, deduplicated, max 10")
    years: List[str] = Field(default_factory=list, description="Distinct years across matching entries")
    total_fragments: int
    fragments: List[BibliographyWorkFragment] = Field(default_factory=list)
    index_name: str
    limit: int
    offset: int


def _nested_query(title: str, authors: Optional[List[str]], use_match_phrase: bool) -> Dict[str, Any]:
    """Nested ``bibliography`` query for one work title, scoped to one entry.

    Scoping the ``nested`` query's own ``bool`` to the title (and, when
    given, the authors) clause means ``inner_hits`` returns the matching
    citation entry itself rather than an arbitrary entry of the fragment's
    bibliography.

    :param title: Work title to match.
    :param authors: Optional author names the matching entry must include
        (``bibliography.authors`` is a keyword field; matched with ``terms``).
    :param use_match_phrase: Use ``match_phrase`` on the analyzed
        ``bibliography.title`` text field instead of an exact
        ``bibliography.title.keyword`` term — the fallback for titles longer
        than the keyword field's 512-char ``ignore_above``.
    :return: A top-level Elasticsearch ``query`` dict (a ``nested`` clause).
    """
    title_clause = (
        {"match_phrase": {"bibliography.title": title}}
        if use_match_phrase
        else {"term": {"bibliography.title.keyword": title}}
    )
    must: List[Dict[str, Any]] = [title_clause]
    if authors:
        must.append({"terms": {"bibliography.authors": authors}})
    return {
        "nested": {
            "path": "bibliography",
            "query": {"bool": {"must": must}},
            "inner_hits": {
                "name": INNER_HITS_NAME,
                "size": 3,
                "_source": _INNER_HITS_SOURCE,
            },
        }
    }


def _entry_source(entry: Dict[str, Any]) -> Dict[str, Any]:
    """``_source`` of one nested ``inner_hits`` entry.

    :param entry: One entry of ``hit["inner_hits"][INNER_HITS_NAME]["hits"]["hits"]``.
    :return: Its ``_source`` dict (empty when absent).
    """
    return entry.get("_source", {})


def _as_list(value: Any) -> List[str]:
    """Coerce a keyword field's value (string or list, as ES stores it) to a list.

    :param value: Raw field value from an ``_source`` dict.
    :return: ``[]`` for empty/``None``, ``[value]`` for a bare string, else ``list(value)``.
    """
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def _truncate_snippet(text: Optional[str]) -> Optional[str]:
    """Truncate a fragment's title/description to :data:`SNIPPET_MAX_LEN` chars.

    :param text: Raw ``title`` or ``description`` field value.
    :return: The text (with a trailing ellipsis if cut), or ``None`` when empty.
    """
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None
    return text[:SNIPPET_MAX_LEN] + "…" if len(text) > SNIPPET_MAX_LEN else text


def get_bibliography_work(
    service: "ElasticsearchService",
    title: str,
    authors: Optional[List[str]] = None,
    index_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> BibliographyWorkResponse:
    """Look up one cited work and every fragment whose bibliography cites it.

    Tries an exact ``bibliography.title.keyword`` term first; when that
    returns nothing (including when the title is too long to have been
    indexed as a keyword, see :data:`TITLE_KEYWORD_IGNORE_ABOVE`), retries
    with a nested ``match_phrase`` on the analyzed ``bibliography.title``
    text field. Author/year unions are built only from the fragments
    actually fetched on this page (bounded by ``limit``), which is fine in
    practice since a given title is cited with materially the same authors
    everywhere.

    :param service: Search service providing the Elasticsearch client, the
        default index, and :meth:`ElasticsearchService.public_bibliography_source`
        for attribution sanitising. Never re-implement that sanitising here.
    :param title: Exact work title, as stored in ``bibliography.title``.
    :param authors: Optional author names/surnames to narrow the match to
        citation entries whose ``bibliography.authors`` includes them.
    :param index_name: Elasticsearch index to search; defaults to ``service.index_name``.
    :param limit: Maximum fragments to return (caller is expected to have
        already bounded this to a sane maximum).
    :param offset: Pagination offset into the fragment list.
    :return: Work summary plus the citing fragments.
    :raises HTTPException: 404 when nothing cites the title, 500 on an
        Elasticsearch failure.
    """
    target_index = index_name or service.index_name
    es = service.es

    try:
        response = es.search(
            index=target_index,
            query=_nested_query(title, authors, use_match_phrase=False),
            size=limit,
            from_=offset,
            _source=_DOC_SOURCE_FIELDS,
        )
        if int(response.get("hits", {}).get("total", {}).get("value", 0)) == 0:
            response = es.search(
                index=target_index,
                query=_nested_query(title, authors, use_match_phrase=True),
                size=limit,
                from_=offset,
                _source=_DOC_SOURCE_FIELDS,
            )
    except Exception as exc:
        logger.error("Bibliography work search failed for %r: %s", title, exc)
        raise HTTPException(status_code=500, detail=f"Bibliography work search failed: {exc}")

    total = int(response.get("hits", {}).get("total", {}).get("value", 0))
    if total == 0:
        raise HTTPException(status_code=404, detail=f"No fragments cite {title!r}")

    fragments: List[BibliographyWorkFragment] = []
    authors_union: Dict[str, None] = {}
    years_seen: Dict[str, None] = {}

    for hit in response["hits"]["hits"]:
        source = hit.get("_source", {})
        doc_id = source.get("doc_id") or hit.get("_id")
        shelf_mark = source.get("shelf_mark") or source.get("shelfmark") or source.get("classmark")
        snippet = _truncate_snippet(source.get("description")) or _truncate_snippet(source.get("title"))

        inner_hits = hit.get("inner_hits", {}).get(INNER_HITS_NAME, {}).get("hits", {}).get("hits", [])
        matched_entry = _entry_source(inner_hits[0]) if inner_hits else {}

        fragments.append(
            BibliographyWorkFragment(
                doc_id=doc_id,
                shelf_mark=shelf_mark,
                snippet=snippet,
                location=matched_entry.get("location") or None,
                relations=_as_list(matched_entry.get("relations")),
                source=service.public_bibliography_source(matched_entry.get("source")),
            )
        )

        for entry in inner_hits:
            entry_source = _entry_source(entry)
            for author in _as_list(entry_source.get("authors")):
                authors_union.setdefault(author, None)
            year = entry_source.get("year")
            if year:
                years_seen.setdefault(year, None)

    return BibliographyWorkResponse(
        title=title,
        authors=list(authors_union)[:10],
        years=list(years_seen),
        total_fragments=total,
        fragments=fragments,
        index_name=target_index,
        limit=limit,
        offset=offset,
    )
