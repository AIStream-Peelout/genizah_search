"""XML sitemaps listing one public ``/fragment/<id>`` page per catalogue record.

Served by ``app.py`` as ``/sitemap-fragments.xml`` (a ``<sitemapindex>``) and
``/sitemap-fragments-<n>.xml`` (``<urlset>`` files, 1-based).  The id list is
read from the catalogue index with ``search_after`` paging on the ``doc_id``
keyword (no ``_source``, doc values only), filtered with
:func:`src.backend.seo_urls.is_sitemap_eligible`, and kept in memory for
``ttl_seconds``.  The files are rendered once per refresh, so serving a
sitemap returns an already-built string.

No ``<lastmod>`` is emitted: every record in the merged index shares one
``indexed_at`` date, which would tell crawlers nothing.

Limits follow sitemaps.org: at most 50,000 URLs and 50 MiB (uncompressed)
per file.  The default of 45,000 URLs per file gives two files for the
~73.5k records of ``genizah_merged_v8``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from elasticsearch import ApiError, TransportError

from src.backend.seo_urls import (
    CANONICAL_BASE,
    fragment_path,
    is_sitemap_eligible,
    sitemap_page_path,
)

logger = logging.getLogger(__name__)

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
MAX_URLS_PER_SITEMAP = 50_000
MAX_SITEMAP_BYTES = 52_428_800
MAX_SCAN_BATCH = 10_000
RETRY_AFTER_FAILURE_S = 300

_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>\n'
_URLSET_OPEN = f'{_XML_DECL}<urlset xmlns="{SITEMAP_NS}">\n'
_URLSET_CLOSE = "</urlset>\n"
_INDEX_OPEN = f'{_XML_DECL}<sitemapindex xmlns="{SITEMAP_NS}">\n'
_INDEX_CLOSE = "</sitemapindex>\n"


@dataclass(frozen=True)
class SitemapSnapshot:
    """One immutable refresh of the fragment id list, already rendered.

    :param pages: Complete ``<urlset>`` documents, page 1 first.
    :param url_count: Number of URLs across all pages.
    :param excluded_doc_ids: Ids left out by :func:`is_sitemap_eligible`, sorted.
    :param expires_at: Clock value after which the snapshot is refreshed.
    """

    pages: Tuple[str, ...]
    url_count: int
    excluded_doc_ids: Tuple[str, ...] = field(default_factory=tuple)
    expires_at: float = 0.0


def url_entry(loc: str) -> str:
    """One ``<url>`` element, XML-escaped.

    :param loc: Absolute, already percent-encoded URL.
    :return: ``<url><loc>...</loc></url>`` followed by a newline.
    :rtype: str
    """
    return f"<url><loc>{escape(loc)}</loc></url>\n"


def chunk_entries(entries: Sequence[str], max_urls: int, max_bytes: int) -> List[List[str]]:
    """Split ``<url>`` entries into files under both the count and size limits.

    :param entries: Rendered ``<url>`` elements in output order.
    :param max_urls: Maximum entries per file.
    :param max_bytes: Maximum UTF-8 size of a whole ``<urlset>`` file.
    :return: Consecutive groups of entries, none empty.
    :rtype: List[List[str]]
    :raises ValueError: If one entry alone cannot fit in a file.
    """
    budget = max_bytes - len((_URLSET_OPEN + _URLSET_CLOSE).encode("utf-8"))
    pages: List[List[str]] = []
    current: List[str] = []
    size = 0
    for entry in entries:
        n = len(entry.encode("utf-8"))
        if n > budget:
            raise ValueError(f"sitemap entry of {n} bytes exceeds the per-file budget")
        if current and (len(current) >= max_urls or size + n > budget):
            pages.append(current)
            current, size = [], 0
        current.append(entry)
        size += n
    if current:
        pages.append(current)
    return pages


def render_urlset(entries: Sequence[str]) -> str:
    """Wrap rendered ``<url>`` entries in a ``<urlset>`` document.

    :param entries: Output of :func:`url_entry`.
    :return: Complete sitemap XML.
    :rtype: str
    """
    return _URLSET_OPEN + "".join(entries) + _URLSET_CLOSE


def render_sitemap_index(locs: Sequence[str]) -> str:
    """A ``<sitemapindex>`` document listing sitemap files.

    :param locs: Absolute URLs of the sitemap files.
    :return: Complete sitemap index XML.
    :rtype: str
    """
    body = "".join(f"<sitemap><loc>{escape(loc)}</loc></sitemap>\n" for loc in locs)
    return _INDEX_OPEN + body + _INDEX_CLOSE


class FragmentSitemaps:
    """Cached fragment sitemaps for one catalogue index.

    Safe to call from FastAPI's sync-endpoint thread pool: a refresh is done
    by one thread under a lock while the others wait for its result, and the
    published snapshot is immutable.

    :param es: Elasticsearch client (only ``search`` is used, read-only).
    :param index_name: Catalogue index, e.g. ``genizah_merged_v8``.
    :param base_url: Site origin for every ``<loc>``; a trailing slash is ignored.
    :param urls_per_file: URLs per ``<urlset>`` file (1..50,000).
    :param ttl_seconds: How long one scan of the index is reused.
    :param batch_size: Hits per ``search_after`` request (1..10,000).
    :param clock: Monotonic time source, injectable for tests.
    """

    def __init__(
        self,
        es: Any,
        index_name: str,
        base_url: str = CANONICAL_BASE,
        urls_per_file: int = 45000,
        ttl_seconds: int = 86400,
        batch_size: int = MAX_SCAN_BATCH,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Validate limits and set up an empty cache (no ES call here).

        :param es: Elasticsearch client.
        :param index_name: Catalogue index name.
        :param base_url: Site origin for the URLs.
        :param urls_per_file: URLs per sitemap file.
        :param ttl_seconds: Cache lifetime in seconds.
        :param batch_size: Hits per scan request.
        :param clock: Monotonic time source.
        :raises ValueError: If a limit is outside the sitemap protocol / ES bounds.
        """
        if not 1 <= urls_per_file <= MAX_URLS_PER_SITEMAP:
            raise ValueError(f"urls_per_file must be 1..{MAX_URLS_PER_SITEMAP}")
        if not 1 <= batch_size <= MAX_SCAN_BATCH:
            raise ValueError(f"batch_size must be 1..{MAX_SCAN_BATCH}")
        self.es = es
        self.index_name = index_name
        self.base_url = base_url.rstrip("/")
        self.urls_per_file = urls_per_file
        self.ttl_seconds = ttl_seconds
        self.batch_size = batch_size
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: Optional[SitemapSnapshot] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_xml(self) -> str:
        """The ``<sitemapindex>`` listing every fragment sitemap file.

        :return: XML for ``/sitemap-fragments.xml``.
        :rtype: str
        """
        pages = len(self.snapshot().pages)
        return render_sitemap_index(
            [self.base_url + sitemap_page_path(n) for n in range(1, pages + 1)]
        )

    def page_xml(self, page: int) -> Optional[str]:
        """One ``<urlset>`` file.

        :param page: 1-based file number.
        :return: XML for ``/sitemap-fragments-<page>.xml``, or ``None`` if out of range.
        :rtype: Optional[str]
        """
        pages = self.snapshot().pages
        if page < 1 or page > len(pages):
            return None
        return pages[page - 1]

    def page_count(self) -> int:
        """Number of ``<urlset>`` files currently published.

        :return: File count (0 when the index has no eligible ids).
        :rtype: int
        """
        return len(self.snapshot().pages)

    def snapshot(self) -> SitemapSnapshot:
        """Current rendered snapshot, refreshing it when the TTL has passed.

        If a refresh fails while an older snapshot exists, the older one keeps
        being served and the next attempt is made after
        :data:`RETRY_AFTER_FAILURE_S`.  With no snapshot yet, the error is raised.

        :return: The live snapshot.
        :rtype: SitemapSnapshot
        """
        snap = self._snapshot
        if snap is not None and self._clock() < snap.expires_at:
            return snap
        with self._lock:
            snap = self._snapshot
            now = self._clock()
            if snap is not None and now < snap.expires_at:
                return snap
            try:
                doc_ids = self.scan_doc_ids()
            except (ApiError, TransportError):
                if snap is None:
                    raise
                logger.warning(
                    "Fragment sitemap refresh failed; serving the previous list", exc_info=True
                )
                snap = SitemapSnapshot(
                    pages=snap.pages,
                    url_count=snap.url_count,
                    excluded_doc_ids=snap.excluded_doc_ids,
                    expires_at=now + RETRY_AFTER_FAILURE_S,
                )
                self._snapshot = snap
                return snap
            snap = self.build_snapshot(doc_ids, expires_at=now + self.ttl_seconds)
            self._snapshot = snap
            logger.info(
                "Fragment sitemaps: %d URLs in %d file(s), %d id(s) excluded",
                snap.url_count, len(snap.pages), len(snap.excluded_doc_ids),
            )
            return snap

    def build_snapshot(self, doc_ids: Sequence[str], expires_at: float) -> SitemapSnapshot:
        """Filter, sort and render a list of document ids.

        :param doc_ids: Raw ids (duplicates allowed).
        :param expires_at: Clock value at which the snapshot goes stale.
        :return: Rendered snapshot.
        :rtype: SitemapSnapshot
        """
        unique = sorted(set(doc_ids))
        eligible = [d for d in unique if is_sitemap_eligible(d)]
        excluded = tuple(d for d in unique if not is_sitemap_eligible(d))
        entries = [url_entry(self.base_url + fragment_path(d)) for d in eligible]
        groups = chunk_entries(entries, self.urls_per_file, MAX_SITEMAP_BYTES)
        return SitemapSnapshot(
            pages=tuple(render_urlset(g) for g in groups),
            url_count=len(eligible),
            excluded_doc_ids=excluded,
            expires_at=expires_at,
        )

    def scan_doc_ids(self) -> List[str]:
        """Every ``doc_id`` in the catalogue index, via ``search_after``.

        Sorted on the ``doc_id`` keyword with ``_source`` disabled and the id
        read from doc values, so each request moves only ids.

        :return: Ids in index sort order (unfiltered).
        :rtype: List[str]
        """
        ids: List[str] = []
        after: Optional[List[Any]] = None
        while True:
            kwargs: Dict[str, Any] = dict(
                index=self.index_name,
                query={"exists": {"field": "doc_id"}},
                sort=[{"doc_id": "asc"}],
                size=self.batch_size,
                source=False,
                docvalue_fields=["doc_id"],
                track_total_hits=False,
            )
            if after is not None:
                kwargs["search_after"] = after
            hits = self.es.search(**kwargs)["hits"]["hits"]
            ids.extend(hit["fields"]["doc_id"][0] for hit in hits)
            if len(hits) < self.batch_size:
                return ids
            after = hits[-1]["sort"]
