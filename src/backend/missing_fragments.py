"""Track shelf marks cited in scholarship but absent from the primary index.

When the RAG pipeline fails to resolve a cited shelf mark to a real fragment
document (either no hit at all, or only a near-miss fuzzy neighbor), the mark
is recorded here with the works and queries that wanted it. The resulting
index is a demand-ranked worklist for prioritizing which fragments to scrape
and ingest next.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

from src.backend.shelfmark_normalizer import ShelfmarkNormalizer

logger = logging.getLogger(__name__)

MAX_TRACKED_CITATIONS = 20
MAX_TRACKED_QUERIES = 20


class MissingFragmentTracker:
    """Records unresolved shelf-mark citations in a small Elasticsearch index."""

    def __init__(self) -> None:
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "elastic.cairogenizah.ai")
        self.es_port = os.getenv("ELASTICSEARCH_PORT", "443")
        self.index_name = os.getenv(
            "ELASTICSEARCH_MISSING_FRAGMENTS_INDEX", "genizah_missing_fragments_v1"
        )
        self.es = Elasticsearch(
            [f"https://{self.es_host}:{self.es_port}"],
            basic_auth=(
                os.getenv("ELASTICSEARCH_USER", "cairo_user"),
                os.getenv("ELASTICSEARCH_PASSWORD"),
            ),
            verify_certs=False,
            retry_on_status=[429, 502, 503, 504],
            max_retries=2,
            retry_on_timeout=True,
            request_timeout=15,
        )

    def record(
        self,
        shelf_mark: str,
        origin: str,
        citations: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        nearest_match: Optional[str] = None,
    ) -> None:
        """Upsert one unresolved shelf-mark observation.

        Never raises: telemetry must not break the answer pipeline.

        :param shelf_mark: The cited shelf mark exactly as observed.
        :param origin: Where the citation came from
            (``bibliography_mention`` or ``answer_mention``).
        :param citations: Scholarly works citing this mark, when known.
        :param user_query: The user query that surfaced the citation.
        :param nearest_match: Shelf mark of a rejected near-miss hit, if any.
        """
        canonical = ShelfmarkNormalizer.to_canonical_id(shelf_mark or "").lower()
        if not canonical:
            return
        now = datetime.now(timezone.utc).isoformat()
        new_citations = [c for c in (citations or []) if c][:MAX_TRACKED_CITATIONS]
        new_queries = [q for q in ([user_query] if user_query else []) if q]
        try:
            self.es.update(
                index=self.index_name,
                id=canonical,
                retry_on_conflict=3,
                script={
                    "source": """
                        ctx._source.occurrence_count += 1;
                        ctx._source.last_seen = params.now;
                        if (params.nearest_match != null) {
                            ctx._source.nearest_match = params.nearest_match;
                        }
                        for (item in params.citations) {
                            if (!ctx._source.citations.contains(item)
                                && ctx._source.citations.size() < params.max_citations) {
                                ctx._source.citations.add(item);
                            }
                        }
                        for (item in params.queries) {
                            if (!ctx._source.queries.contains(item)
                                && ctx._source.queries.size() < params.max_queries) {
                                ctx._source.queries.add(item);
                            }
                        }
                    """,
                    "params": {
                        "now": now,
                        "citations": new_citations,
                        "queries": new_queries,
                        "nearest_match": nearest_match,
                        "max_citations": MAX_TRACKED_CITATIONS,
                        "max_queries": MAX_TRACKED_QUERIES,
                    },
                },
                upsert={
                    "shelf_mark": shelf_mark,
                    "canonical_id": canonical,
                    "origin": origin,
                    "occurrence_count": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "citations": new_citations,
                    "queries": new_queries,
                    **({"nearest_match": nearest_match} if nearest_match else {}),
                },
            )
            logger.info(
                "Recorded missing fragment %r (origin=%s, nearest=%s)",
                shelf_mark, origin, nearest_match,
            )
        except Exception as exc:
            logger.warning("Could not record missing fragment %r: %s", shelf_mark, exc)

    def list_missing(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return unresolved shelf marks ranked by citation demand.

        :param limit: Maximum entries to return.
        :returns: Missing-fragment records, most-cited first.
        :rtype: List[Dict[str, Any]]
        """
        try:
            response = self.es.search(
                index=self.index_name,
                size=limit,
                sort=[{"occurrence_count": {"order": "desc"}}, {"last_seen": {"order": "desc"}}],
                query={"match_all": {}},
            )
        except Exception as exc:
            logger.warning("Could not list missing fragments: %s", exc)
            return []
        return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]


# Global instance
missing_fragment_tracker = MissingFragmentTracker()
