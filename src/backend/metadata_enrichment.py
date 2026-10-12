"""Display fields of a catalogue record read from a richer index by exact doc id.

The catalogue index behind ``/document`` and the ``/fragment/<doc_id>`` pages
(``ELASTICSEARCH_INDEX``, genizah_merged_v8) has no ``subjects``, ``genres`` or
``scholarly_frames`` and fewer ``alt_titles`` than genizah_merged_v10.  Until
the catalogue index moves on, those two single-record views read the four
fields from :data:`DEFAULT_ENRICHMENT_INDEX` with one extra GET by doc id.
Search result cards never do this: they show what their own hit carries (the
document modal opened from a card reads ``/document`` once for them).
Internal lookups such as the IIIF manifest skip it
(``get_document_by_id(enrich=False)``, the default).

``METADATA_ENRICHMENT_INDEX`` (env) names the index; an empty value switches
the lookup off.  A missing index or document, or any Elasticsearch error,
leaves the record as it is (one log line, never an error).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from elasticsearch import ApiError, NotFoundError, TransportError

logger = logging.getLogger(__name__)

DEFAULT_ENRICHMENT_INDEX = "genizah_merged_v10"
ENRICHMENT_FIELDS: Tuple[str, ...] = ("alt_titles", "subjects", "genres", "scholarly_frames")


def enrichment_index() -> str:
    """The index the display fields are read from (``""`` when switched off).

    :return: ``METADATA_ENRICHMENT_INDEX``, default :data:`DEFAULT_ENRICHMENT_INDEX`.
    :rtype: str
    """
    return os.getenv("METADATA_ENRICHMENT_INDEX", DEFAULT_ENRICHMENT_INDEX).strip()


def fetch_enrichment(es: Any, doc_id: Optional[str], record_index: str) -> Dict[str, Any]:
    """The :data:`ENRICHMENT_FIELDS` the enrichment index holds for one doc id.

    :param es: Elasticsearch client (one ``get`` call, source filtered).
    :param doc_id: Catalogue id of the record (the indices share ``_id == doc_id``).
    :param record_index: Index the record itself came from; no lookup when it
        is the enrichment index.
    :return: The non-empty fields of the enrichment document, or ``{}`` when the
        lookup is off, not needed, or finds nothing.
    :rtype: Dict[str, Any]
    """
    index = enrichment_index()
    if not index or not doc_id or index == record_index:
        return {}
    try:
        response = es.get(index=index, id=doc_id, source_includes=["doc_id", *ENRICHMENT_FIELDS])
    except NotFoundError:
        logger.info("Metadata enrichment: %r not found in %s", doc_id, index)
        return {}
    except (ApiError, TransportError) as exc:
        logger.warning("Metadata enrichment: %s unavailable for %r (%s)", index, doc_id, exc)
        return {}
    source = response.get("_source") or {}
    if source.get("doc_id") != doc_id:
        return {}
    return {field: source[field] for field in ENRICHMENT_FIELDS if source.get(field)}


def enrich_source(es: Any, source: Dict[str, Any], record_index: str) -> Dict[str, Any]:
    """A catalogue ``_source`` with the enrichment index's display fields merged in.

    :param es: Elasticsearch client.
    :param source: Catalogue ``_source`` (not modified).
    :param record_index: Index the record came from.
    :return: ``source`` with each field the enrichment index has replacing its own,
        or ``source`` itself when there is nothing to add.
    :rtype: Dict[str, Any]
    """
    extra = fetch_enrichment(es, source.get("doc_id"), record_index)
    return {**source, **extra} if extra else source
