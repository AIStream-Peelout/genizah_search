"""Readings sidecars and expert line corrections (advanced "Readers" view).

The consensus read the ``/read`` page shows (``src/backend/ai_transcriptions.py``)
is the *final* output of the offline pipeline in the sibling
``historical-document-analysis`` project.  For every page image the pipeline
also keeps a **readings sidecar** (schema ``readings-v1``, contract in that
project's ``docs/readings_sidecar.md``): every reader's lines with their boxes
(the whole-page VLM read, each decode-policy unit read such as column crops or
rescue halves, Kraken's line fragments) plus the decode policy and the merged
``final`` lines.  The Studio pushes one Elasticsearch document per sidecar into
:data:`DEFAULT_READINGS_INDEX`; this module only reads it.

Scholars can suggest a corrected reading of a line.  Suggestions go to
:data:`DEFAULT_CORRECTIONS_INDEX` and the Studio pulls them back as human
ground truth through ``GET /admin/corrections.jsonl``.  A suggestion names the
sidecar's ``decoded_at`` the panel had loaded; one made on a decode the server
no longer holds is refused (409), so it cannot land on a different line.

Access: the site has no accounts.  Every route here needs the shared
``X-Advanced-Token`` header matching ``ADVANCED_ACCESS_TOKEN``; when that
variable is unset the routes answer 404, so the feature is off by default and
ordinary visitors never see it.  The export needs its own
``READINGS_EXPORT_TOKEN`` so the token handed to scholars cannot download every
contributor's name.  Suggestions are rate-limited in memory, per client IP and
in total.

Coordinates: every box is ``[x1, y1, x2, y2]`` in 0-1000 of the oriented page,
the frame the consensus overlay already uses.  Reader lines carry ``box``,
final lines ``bbox``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from elasticsearch import BadRequestError, Elasticsearch, NotFoundError, helpers
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.backend.ai_transcriptions import record_id
from src.backend.consensus_reads import to_ai_doc_ids

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_READINGS_INDEX = "genizah_ai_readings_v1"
DEFAULT_CORRECTIONS_INDEX = "genizah_line_corrections_v1"
SCHEMA = "readings-v1"
#: ``reader_id`` of the merged consensus lines (``final.lines``).
FINAL_READER_ID = "final"

#: How long a fetched sidecar is served from memory (the Studio rebuilds
#: sidecars every half hour, so an hour-old copy is at most one cycle behind).
DEFAULT_CACHE_TTL_S = 3600
#: The browser revalidates every readings response (the hour-long cache is the
#: server's): a browser-held copy could hide a second model's sidecar pushed
#: since, or pair a suggestion with a decode the server no longer serves. The
#: ETag makes an unchanged answer a cheap 304.
REVALIDATE = "private, no-cache"
DEFAULT_CACHE_ENTRIES = 512
#: Suggestions accepted per client IP, and from everyone together, per hour.
DEFAULT_CORRECTIONS_PER_IP_HOUR = 30
DEFAULT_CORRECTIONS_TOTAL_HOUR = 300

MAX_TEXT_CHARS = 2000
MAX_NOTE_CHARS = 1000
MAX_CONTRIBUTOR_CHARS = 100
MAX_ID_CHARS = 300

#: One document per sidecar, ``_id`` = ``<doc_id>__<image_index>__<vlm_model>``.
#: The sidecar itself is stored, not indexed, under ``readings``; the few
#: top-level fields exist to look a page up and to audit what was pushed.
#: ``rule_version`` and ``final_source`` are optional.
READINGS_INDEX_MAPPING: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": {"type": "keyword"},
            "image_index": {"type": "integer"},
            "vlm_model": {"type": "keyword"},
            "image_sha256": {"type": "keyword"},
            "decoded_at": {"type": "date"},
            "rule_version": {"type": "keyword"},
            "policy_action": {"type": "keyword"},
            "final_source": {"type": "keyword"},
            "n_readers": {"type": "integer"},
            "readings": {"type": "object", "enabled": False},
        },
    },
}

#: One document per suggestion.  ``page_hash`` (the image's sha256),
#: ``vlm_model``, ``rule_version`` and ``decoded_at`` pin the exact read that was
#: corrected, so a later re-read of the page cannot silently re-target it.
#: IP addresses are never stored.
CORRECTIONS_INDEX_MAPPING: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": {"type": "keyword"},
            "image_index": {"type": "integer"},
            "line_index": {"type": "integer"},
            "reader_id": {"type": "keyword"},
            "source_reader": {"type": "keyword"},
            "bbox": {"type": "integer"},
            "original_text": {"type": "text"},
            "corrected_text": {"type": "text"},
            "note": {"type": "text"},
            "contributor": {"type": "keyword"},
            "created_at": {"type": "date"},
            "page_hash": {"type": "keyword"},
            "vlm_model": {"type": "keyword"},
            "rule_version": {"type": "keyword"},
            "decoded_at": {"type": "date"},
        },
    },
}


def readings_index_name() -> str:
    """Name of the index holding readings sidecars.

    :return: ``AI_READINGS_INDEX`` or :data:`DEFAULT_READINGS_INDEX`.
    :rtype: str
    """
    return os.getenv("AI_READINGS_INDEX", DEFAULT_READINGS_INDEX)


def corrections_index_name() -> str:
    """Name of the index holding line corrections.

    :return: ``LINE_CORRECTIONS_INDEX`` or :data:`DEFAULT_CORRECTIONS_INDEX`.
    :rtype: str
    """
    return os.getenv("LINE_CORRECTIONS_INDEX", DEFAULT_CORRECTIONS_INDEX)


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def readings_document_id(sidecar: Dict[str, Any]) -> str:
    """Elasticsearch ``_id`` of a sidecar, the same key the consensus records use.

    :param sidecar: A ``readings-v1`` sidecar.
    :return: ``"<doc_id>__<image_index>__<vlm_model>"``.
    :rtype: str
    """
    return record_id(sidecar["doc_id"], int(sidecar["image_index"]), sidecar["vlm_model"])


def readings_document(sidecar: Dict[str, Any]) -> Dict[str, Any]:
    """The stored document for one sidecar (what the Studio push writes).

    :param sidecar: A ``readings-v1`` sidecar, stored verbatim under ``readings``.
    :return: Indexed lookup fields plus the sidecar.
    :rtype: Dict[str, Any]
    """
    policy = sidecar.get("policy") or {}
    final = sidecar.get("final") or {}
    return {
        "doc_id": sidecar["doc_id"],
        "image_index": int(sidecar["image_index"]),
        "vlm_model": sidecar["vlm_model"],
        "image_sha256": sidecar.get("image_sha256"),
        "decoded_at": sidecar.get("decoded_at"),
        "rule_version": sidecar.get("rule_version"),
        "policy_action": policy.get("action"),
        "final_source": final.get("source"),
        "n_readers": len(sidecar.get("readers") or []),
        "readings": sidecar,
    }


def _is_box(value: Any) -> bool:
    """Whether ``value`` is a ``[x1, y1, x2, y2]`` box of numbers.

    :param value: Candidate box.
    :return: ``True`` for a list of four numbers.
    :rtype: bool
    """
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def validate_sidecar(sidecar: Any) -> List[str]:
    """Structural problems of a sidecar that would break the viewer.

    Only what the site relies on is checked; unknown keys (new reader kinds, a
    future ``vote`` block) are allowed.

    :param sidecar: Parsed JSON.
    :return: Human-readable problems; empty when the sidecar is usable.
    :rtype: List[str]
    """
    if not isinstance(sidecar, dict):
        return ["sidecar is not a JSON object"]
    problems = [f"missing {key}" for key in ("doc_id", "vlm_model") if not sidecar.get(key)]
    if not isinstance(sidecar.get("image_index"), int) or isinstance(sidecar.get("image_index"), bool):
        problems.append("image_index is not an integer")
    readers = sidecar.get("readers")
    if not isinstance(readers, list):
        problems.append("readers is not a list")
        readers = []
    for position, reader in enumerate(readers):
        if not isinstance(reader, dict) or not reader.get("id"):
            problems.append(f"readers[{position}] has no id")
            continue
        lines = reader.get("lines")
        if lines is None:
            continue
        if not isinstance(lines, list):
            problems.append(f"reader {reader['id']}: lines is neither a list nor null")
            continue
        bad = sum(1 for line in lines if not isinstance(line, dict) or not _is_box(line.get("box")))
        if bad:
            problems.append(f"reader {reader['id']}: {bad} line(s) without a 4-number box")
    final = sidecar.get("final")
    if not isinstance(final, dict) or not isinstance(final.get("lines"), list):
        problems.append("final.lines is not a list")
    else:
        bad = sum(1 for line in final["lines"] if not isinstance(line, dict) or not _is_box(line.get("bbox")))
        if bad:
            problems.append(f"final: {bad} line(s) without a 4-number bbox")
    return problems


def resolve_line(sidecar: Dict[str, Any], reader_id: str, line_index: int) -> Optional[Dict[str, Any]]:
    """Find one line of one reader (or of the final consensus) in a sidecar.

    :param sidecar: A ``readings-v1`` sidecar.
    :param reader_id: A ``readers[].id`` or :data:`FINAL_READER_ID`.
    :param line_index: The line's ``index`` field.
    :return: ``{"text", "box", "source_reader"}``, or ``None`` when the reader
        or the line does not exist.  ``source_reader`` is the reader id, or for
        the final lines ``final:<final.source>`` (e.g. ``final:assembled``).
    :rtype: Optional[Dict[str, Any]]
    """
    if reader_id == FINAL_READER_ID:
        final = sidecar.get("final") or {}
        lines, box_key = final.get("lines") or [], "bbox"
        source_reader = f"{FINAL_READER_ID}:{final.get('source') or 'unknown'}"
    else:
        reader = next((r for r in sidecar.get("readers") or [] if r.get("id") == reader_id), None)
        if reader is None:
            return None
        lines, box_key, source_reader = reader.get("lines") or [], "box", reader_id
    line = next((l for l in lines if isinstance(l, dict) and l.get("index") == line_index), None)
    if line is None:
        return None
    return {"text": line.get("text") or "", "box": line.get(box_key), "source_reader": source_reader}


def readings_etag(sidecars: Iterable[Dict[str, Any]]) -> str:
    """Weak ETag naming exactly which decodes (and builds) a response holds.

    :param sidecars: The sidecar(s) in the response.
    :return: ``W/"<digest>"``; it changes when a model is added or a page is re-decoded or rebuilt.
    :rtype: str
    """
    keys = sorted(
        "|".join(str(part) for part in (
            sidecar.get("vlm_model"), sidecar.get("decoded_at"), sidecar.get("image_sha256"),
            (sidecar.get("provenance") or {}).get("built_at"),
        ))
        for sidecar in sidecars
    )
    return 'W/"' + hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()[:20] + '"'


def _not_modified(if_none_match: Optional[str], etag: str) -> bool:
    """Whether the browser already holds this exact answer.

    :param if_none_match: ``If-None-Match`` request header, if any.
    :param etag: ETag of the answer about to be sent.
    :return: ``True`` when one of the listed tags equals ``etag`` (or is ``*``).
    :rtype: bool
    """
    if not if_none_match:
        return False
    tags = {tag.strip() for tag in if_none_match.split(",")}
    return "*" in tags or etag in tags


class _TtlCache:
    """Small thread-safe LRU cache whose entries expire after a fixed time.

    :param ttl_s: Lifetime of an entry in seconds.
    :param max_entries: Oldest entries are evicted beyond this size.
    :param clock: Monotonic clock (injectable for tests).
    """

    def __init__(self, ttl_s: float, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._data: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Return a live entry, or ``None``.

        :param key: Cache key.
        :return: The cached value, or ``None`` when missing or expired.
        :rtype: Optional[Any]
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry[0] <= self._clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry[1]

    def put(self, key: Any, value: Any) -> None:
        """Store a value.

        :param key: Cache key.
        :param value: Value to keep for :attr:`ttl_s` seconds.
        """
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def discard(self, key: Any) -> None:
        """Drop an entry if there is one.

        :param key: Cache key.
        """
        with self._lock:
            self._data.pop(key, None)


class ReadingsStore:
    """Read-only access to the readings index.

    Hits are cached for an hour; empty results are not, so a page pushed by
    the Studio shows up on the next request.  A cached page can still lack a
    second model's sidecar or a re-decode pushed after it was cached; callers
    that need one the cached set lacks read again with ``fresh=True``.

    :param es: Elasticsearch client shared with the search service.
    :param index: Index name (defaults to :func:`readings_index_name`).
    :param cache_ttl_s: Lifetime of a cached page in seconds.
    :param clock: Monotonic clock for the cache (injectable for tests).
    """

    def __init__(
        self,
        es: Elasticsearch,
        index: Optional[str] = None,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.es = es
        self.index = index or readings_index_name()
        self._cache = _TtlCache(cache_ttl_s, DEFAULT_CACHE_ENTRIES, clock)

    def pages(self, doc_id: str) -> List[Dict[str, Any]]:
        """Which images of a fragment have a sidecar, under which model.

        :param doc_id: Fragment id (either Oxford form works).
        :return: Lookup fields of every sidecar, by image then newest first.
        :rtype: List[Dict[str, Any]]
        """
        key = ("pages", doc_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            response = self.es.search(
                index=self.index,
                query={"terms": {"doc_id": to_ai_doc_ids(doc_id)}},
                source_excludes=["readings"],
                sort=[{"image_index": {"order": "asc"}}, {"decoded_at": {"order": "desc", "missing": "_last"}}],
                size=200,
            )
        except NotFoundError:
            logger.info("Readings index %s does not exist yet", self.index)
            return []
        pages = [
            {
                "image_index": source.get("image_index"),
                "vlm_model": source.get("vlm_model"),
                "policy_action": source.get("policy_action"),
                "n_readers": source.get("n_readers"),
                "decoded_at": source.get("decoded_at"),
                "rule_version": source.get("rule_version"),
                "final_source": source.get("final_source"),
            }
            for source in (hit["_source"] for hit in response["hits"]["hits"])
        ]
        if pages:
            self._cache.put(key, pages)
        return pages

    def get(self, doc_id: str, image_index: int, *, fresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Every sidecar of one page image, keyed by ``vlm_model``.

        A page normally has one; a page read by two checkpoints has two.

        :param doc_id: Fragment id (either Oxford form works).
        :param image_index: Which image of the fragment.
        :param fresh: Skip the cache and read Elasticsearch; the result
            replaces the cached entry (or drops it when empty).
        :return: ``{vlm_model: sidecar}``; empty when the page has none.
        :rtype: Dict[str, Dict[str, Any]]
        """
        key = ("page", doc_id, image_index)
        if not fresh:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        try:
            response = self.es.search(
                index=self.index,
                query={
                    "bool": {
                        "filter": [
                            {"terms": {"doc_id": to_ai_doc_ids(doc_id)}},
                            {"term": {"image_index": image_index}},
                        ]
                    }
                },
                sort=[{"decoded_at": {"order": "desc", "missing": "_last"}}],
                size=10,
            )
        except NotFoundError:
            logger.info("Readings index %s does not exist yet", self.index)
            self._cache.discard(key)
            return {}
        by_model: Dict[str, Dict[str, Any]] = {}
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            # Newest first: an old-Oxford-id duplicate of the same model loses.
            by_model.setdefault(source["vlm_model"], source["readings"])
        if by_model:
            self._cache.put(key, by_model)
        else:
            # A fresh read that finds nothing must not leave the old entry behind.
            self._cache.discard(key)
        return by_model


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


class CorrectionRequest(BaseModel):
    """Body of ``POST /documents/{doc_id}/readings/corrections``.

    ``bbox`` and ``original_text`` are accepted because the panel sends them,
    but the stored values come from the sidecar so a client cannot attach a
    suggestion to text the readers never produced (``original_text`` is not
    length-limited, so a suggestion on a very long line is not refused over a
    field nobody reads).  ``decoded_at`` is the sidecar's ``decoded_at`` the
    panel had loaded; a suggestion made on another decode is refused.
    """

    image_index: int = Field(..., ge=0, description="Which image of the fragment.")
    line_index: int = Field(..., ge=0, description="The corrected line's ``index``.")
    reader_id: str = Field(
        FINAL_READER_ID,
        min_length=1,
        max_length=MAX_ID_CHARS,
        description="``final`` for the consensus line, else a ``readers[].id``.",
    )
    corrected_text: str = Field(..., max_length=MAX_TEXT_CHARS, description="The suggested reading.")
    contributor: str = Field(..., max_length=MAX_CONTRIBUTOR_CHARS, description="Name the contributor typed.")
    note: Optional[str] = Field(None, max_length=MAX_NOTE_CHARS, description="Optional remark.")
    vlm_model: Optional[str] = Field(None, max_length=MAX_ID_CHARS, description="Sidecar the line belongs to.")
    decoded_at: Optional[str] = Field(
        None, max_length=64, description="The sidecar's ``decoded_at`` the panel had loaded."
    )
    bbox: Optional[List[float]] = Field(None, description="Ignored; the sidecar's box is stored.")
    original_text: Optional[str] = Field(None, description="Ignored; the sidecar's text is stored.")

    @field_validator("corrected_text", "contributor")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Strip and reject empty text.

        :param value: Submitted text.
        :return: The stripped text.
        :rtype: str
        :raises ValueError: When nothing but whitespace was submitted.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("note")
    @classmethod
    def _blank_note_is_none(cls, value: Optional[str]) -> Optional[str]:
        """Store an empty note as ``None``.

        :param value: Submitted note.
        :return: The stripped note, or ``None``.
        :rtype: Optional[str]
        """
        value = (value or "").strip()
        return value or None


def _instant(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp (``Z`` or an offset; no offset means UTC).

    :param value: Timestamp text.
    :return: An aware datetime, or ``None`` when the text is not ISO 8601.
    :rtype: Optional[datetime]
    """
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def same_decode(supplied: str, stored: Any) -> bool:
    """Whether the ``decoded_at`` the panel loaded names the decode the server holds.

    The two sides may spell the same instant differently (``Z`` or
    ``+00:00``), so timestamps are compared as instants; text that does not
    parse is compared as it is.

    :param supplied: ``decoded_at`` sent with the suggestion.
    :param stored: The server sidecar's ``decoded_at`` (may be missing).
    :return: ``True`` when both name the same decode.
    :rtype: bool
    """
    if not isinstance(stored, str):
        return False
    supplied_at, stored_at = _instant(supplied), _instant(stored)
    if supplied_at is None or stored_at is None:
        return supplied.strip() == stored.strip()
    return supplied_at == stored_at


def build_correction(
    doc_id: str, request: CorrectionRequest, sidecar: Dict[str, Any], line: Dict[str, Any]
) -> Dict[str, Any]:
    """The stored document for one suggestion.

    :param doc_id: Fragment id from the URL.
    :param request: Validated body.
    :param sidecar: The sidecar the line was resolved in.
    :param line: :func:`resolve_line` result.
    :return: Document to index (no IP address, no token).
    :rtype: Dict[str, Any]
    """
    return {
        "doc_id": doc_id,
        "image_index": request.image_index,
        "line_index": request.line_index,
        "reader_id": request.reader_id,
        "source_reader": line["source_reader"],
        "bbox": [int(round(v)) for v in line["box"]] if _is_box(line["box"]) else None,
        "original_text": line["text"],
        "corrected_text": request.corrected_text,
        "note": request.note,
        "contributor": request.contributor,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_hash": sidecar.get("image_sha256"),
        "vlm_model": sidecar.get("vlm_model"),
        "rule_version": sidecar.get("rule_version"),
        "decoded_at": sidecar.get("decoded_at"),
    }


class CorrectionStore:
    """Write and read line corrections.

    :param es: Elasticsearch client shared with the search service.
    :param index: Index name (defaults to :func:`corrections_index_name`).
    """

    def __init__(self, es: Elasticsearch, index: Optional[str] = None) -> None:
        self.es = es
        self.index = index or corrections_index_name()
        self._index_checked = False

    def ensure_index(self) -> bool:
        """Create the corrections index with its mapping if it does not exist.

        :return: ``True`` when this call created the index.
        :rtype: bool
        """
        if self._index_checked:
            return False
        if self.es.indices.exists(index=self.index):
            self._index_checked = True
            return False
        try:
            self.es.indices.create(index=self.index, **CORRECTIONS_INDEX_MAPPING)
        except BadRequestError as exc:
            # Another worker created it between the check and the create.
            if exc.error != "resource_already_exists_exception":
                raise
            self._index_checked = True
            return False
        self._index_checked = True
        logger.info("Created line corrections index %s", self.index)
        return True

    def record(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Store one suggestion, visible to the next read.

        :param document: :func:`build_correction` result.
        :return: The stored suggestion with its ``id``.
        :rtype: Dict[str, Any]
        """
        self.ensure_index()
        correction_id = uuid.uuid4().hex
        self.es.index(index=self.index, id=correction_id, document=document, refresh="wait_for")
        return {"id": correction_id, **document}

    def list_for_page(self, doc_id: str, image_index: int) -> List[Dict[str, Any]]:
        """Suggestions made on one page image, newest first.

        :param doc_id: Fragment id (either Oxford form works).
        :param image_index: Which image of the fragment.
        :return: Stored suggestions with their ``id``.
        :rtype: List[Dict[str, Any]]
        """
        try:
            response = self.es.search(
                index=self.index,
                query={
                    "bool": {
                        "filter": [
                            {"terms": {"doc_id": to_ai_doc_ids(doc_id)}},
                            {"term": {"image_index": image_index}},
                        ]
                    }
                },
                sort=[{"created_at": {"order": "desc"}}],
                size=500,
            )
        except NotFoundError:
            return []
        return [{"id": hit["_id"], **hit["_source"]} for hit in response["hits"]["hits"]]

    def export(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Every suggestion (optionally only newer ones), oldest first.

        :param since: Only suggestions created at or after this time.
        :return: Stored suggestions with their ``id``.
        :rtype: List[Dict[str, Any]]
        """
        query: Dict[str, Any] = {"match_all": {}}
        if since is not None:
            query = {"range": {"created_at": {"gte": since.isoformat()}}}
        try:
            rows = [
                {"id": hit["_id"], **hit["_source"]}
                for hit in helpers.scan(self.es, index=self.index, query={"query": query})
            ]
        except NotFoundError:
            return []
        return sorted(rows, key=lambda row: (row.get("created_at") or "", row["id"]))


class RateLimiter:
    """Sliding-window limit on events per key (in memory, per process).

    :param limit: Events allowed per window and key.
    :param window_s: Window length in seconds.
    :param clock: Monotonic clock (injectable for tests).
    """

    def __init__(self, limit: int, window_s: float = 3600.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> Tuple[bool, int]:
        """Count one event for ``key`` if the limit allows it.

        :param key: Who is acting (an IP, or a constant for a global limit).
        :return: ``(allowed, retry_after_seconds)``; ``retry_after`` is 0 when allowed.
        :rtype: Tuple[bool, int]
        """
        now = self._clock()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= now - self.window_s:
                events.popleft()
            if len(events) >= self.limit:
                return False, max(1, int(events[0] + self.window_s - now) + 1)
            events.append(now)
            return True, 0


def client_ip(request: Request) -> str:
    """Best guess at the visitor's IP behind the Cloudflare tunnel.

    :param request: Inbound request.
    :return: ``CF-Connecting-IP``, else the first ``X-Forwarded-For`` hop, else the peer.
    :rtype: str
    """
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip()
    if ip:
        return ip
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _token_matches(supplied: Optional[str], expected: str) -> bool:
    """Constant-time token comparison.

    :param supplied: Header value, if any.
    :param expected: Configured token (non-empty).
    :return: ``True`` when they are equal.
    :rtype: bool
    """
    return bool(supplied) and secrets.compare_digest(supplied.encode(), expected.encode())


def _correction_sidecar(by_model: Dict[str, Dict[str, Any]], body: CorrectionRequest, doc_id: str) -> Dict[str, Any]:
    """The sidecar a suggestion is made on.

    :param by_model: :meth:`ReadingsStore.get` result for the page.
    :param body: Validated suggestion (its ``vlm_model`` picks the sidecar).
    :param doc_id: Fragment id from the URL (for the error message).
    :return: The chosen sidecar.
    :rtype: Dict[str, Any]
    :raises HTTPException: 422 when the page has several reads and none is
        named, 404 when there is no matching sidecar.
    """
    if not body.vlm_model and len(by_model) > 1:
        raise HTTPException(status_code=422, detail="This page has several reads; pass vlm_model")
    if body.vlm_model:
        sidecar = by_model.get(body.vlm_model)
    else:
        sidecar = next(iter(by_model.values()), None)
    if sidecar is None:
        raise HTTPException(status_code=404, detail=f"No readings for {doc_id} image {body.image_index}")
    return sidecar


def create_router(
    readings: ReadingsStore,
    corrections: CorrectionStore,
    access_token: str = "",
    export_token: str = "",
    per_ip_limiter: Optional[RateLimiter] = None,
    total_limiter: Optional[RateLimiter] = None,
) -> APIRouter:
    """Routes of the advanced Readers view.

    :param readings: Sidecar store.
    :param corrections: Correction store.
    :param access_token: Shared advanced-access token; empty disables every
        route (404).
    :param export_token: Token for ``/admin/corrections.jsonl``; empty
        disables the export (404).
    :param per_ip_limiter: Limit on suggestions per client IP (default
        :data:`DEFAULT_CORRECTIONS_PER_IP_HOUR` per hour).
    :param total_limiter: Limit on suggestions from everyone together
        (default :data:`DEFAULT_CORRECTIONS_TOTAL_HOUR` per hour).
    :return: Router to include in the app.
    :rtype: APIRouter
    """
    router = APIRouter()
    per_ip = per_ip_limiter or RateLimiter(DEFAULT_CORRECTIONS_PER_IP_HOUR)
    total = total_limiter or RateLimiter(DEFAULT_CORRECTIONS_TOTAL_HOUR)

    def require_access(x_advanced_token: Optional[str] = Header(default=None)) -> None:
        """Reject requests without the advanced-access token.

        :param x_advanced_token: Value of the ``X-Advanced-Token`` header.
        :raises HTTPException: 404 when the feature is off, 401 when the token is wrong.
        """
        if not access_token:
            raise HTTPException(status_code=404, detail="Not Found")
        if not _token_matches(x_advanced_token, access_token):
            raise HTTPException(status_code=401, detail="Advanced access token required")

    def require_export(x_export_token: Optional[str] = Header(default=None)) -> None:
        """Reject export requests without the export token.

        :param x_export_token: Value of the ``X-Export-Token`` header.
        :raises HTTPException: 404 when the export is off, 401 when the token is wrong.
        """
        if not export_token:
            raise HTTPException(status_code=404, detail="Not Found")
        if not _token_matches(x_export_token, export_token):
            raise HTTPException(status_code=401, detail="Export token required")

    @router.get("/readings/access", dependencies=[Depends(require_access)], include_in_schema=False)
    def check_access() -> Dict[str, bool]:
        """Confirm an advanced-access token (the panel calls this once per page load).

        :return: ``{"ok": True}``.
        :rtype: Dict[str, bool]
        """
        return {"ok": True}

    @router.get("/documents/{doc_id}/readings/pages", dependencies=[Depends(require_access)], include_in_schema=False)
    def list_reading_pages(doc_id: str, if_none_match: Optional[str] = Header(default=None)) -> Response:
        """Which images of a fragment have a readings sidecar.

        :param doc_id: Fragment id.
        :param if_none_match: ``If-None-Match`` header (revalidation).
        :return: ``{"doc_id", "pages": [...]}``, or 304 when unchanged.
        :rtype: Response
        """
        pages = readings.pages(doc_id)
        body = {"doc_id": doc_id, "pages": pages}
        etag = 'W/"' + hashlib.sha1(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:20] + '"'
        headers = {"Cache-Control": REVALIDATE, "ETag": etag}
        if _not_modified(if_none_match, etag):
            return Response(status_code=304, headers=headers)
        return JSONResponse(content=body, headers=headers)

    @router.get("/documents/{doc_id}/readings", dependencies=[Depends(require_access)], include_in_schema=False)
    def get_readings(
        doc_id: str,
        image_index: int = Query(..., ge=0),
        vlm_model: Optional[str] = Query(None, max_length=MAX_ID_CHARS),
        if_none_match: Optional[str] = Header(default=None),
    ) -> Response:
        """The stored readings sidecar of one page image.

        :param doc_id: Fragment id.
        :param image_index: Which image of the fragment.
        :param vlm_model: The checkpoint whose read the page view shows.
        :param if_none_match: ``If-None-Match`` header (revalidation).
        :return: The sidecar (304 when the browser's copy is current); 409
            with every sidecar keyed by model when the page has several and
            ``vlm_model`` does not pick one.
        :rtype: Response
        :raises HTTPException: 404 when there is no matching sidecar.
        """
        by_model = readings.get(doc_id, image_index)
        if vlm_model and vlm_model not in by_model:
            # The cached set may predate this model's sidecar; look once more before a 404.
            by_model = readings.get(doc_id, image_index, fresh=True)
        if vlm_model:
            if vlm_model not in by_model:
                raise HTTPException(
                    status_code=404,
                    detail=f"No readings for {doc_id} image {image_index} under {vlm_model}",
                )
            chosen = by_model[vlm_model]
        elif len(by_model) == 1:
            chosen = next(iter(by_model.values()))
        elif not by_model:
            raise HTTPException(status_code=404, detail=f"No readings for {doc_id} image {image_index}")
        else:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": f"{doc_id} image {image_index} has readings under {len(by_model)} models; pass vlm_model",
                    "readings": by_model,
                },
                headers={"Cache-Control": "no-store"},
            )
        etag = readings_etag([chosen])
        headers = {"Cache-Control": REVALIDATE, "ETag": etag}
        if _not_modified(if_none_match, etag):
            return Response(status_code=304, headers=headers)
        return JSONResponse(content=chosen, headers=headers)

    @router.get(
        "/documents/{doc_id}/readings/corrections", dependencies=[Depends(require_access)], include_in_schema=False
    )
    def list_corrections(doc_id: str, response: Response, image_index: int = Query(..., ge=0)) -> Dict[str, Any]:
        """Suggestions made on one page image, newest first.

        :param doc_id: Fragment id.
        :param response: Outgoing response (marked uncacheable).
        :param image_index: Which image of the fragment.
        :return: ``{"doc_id", "image_index", "corrections": [...]}``.
        :rtype: Dict[str, Any]
        """
        response.headers["Cache-Control"] = "no-store"
        return {
            "doc_id": doc_id,
            "image_index": image_index,
            "corrections": corrections.list_for_page(doc_id, image_index),
        }

    @router.post(
        "/documents/{doc_id}/readings/corrections",
        status_code=201,
        dependencies=[Depends(require_access)],
        include_in_schema=False,
    )
    def submit_correction(doc_id: str, body: CorrectionRequest, request: Request) -> Dict[str, Any]:
        """Store a suggested reading of one line.

        :param doc_id: Fragment id.
        :param body: Validated suggestion.
        :param request: Inbound request (for the client IP of the rate limit).
        :return: ``{"ok": True, "correction": {...}}``.
        :rtype: Dict[str, Any]
        :raises HTTPException: 429 when rate-limited, 404 without a sidecar,
            409 when the page was re-decoded since the panel loaded it, 422 when
            the reader or line does not exist.
        """
        allowed, retry_after = per_ip.allow(client_ip(request))
        if allowed:
            allowed, retry_after = total.allow("*")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many suggestions; try again later",
                headers={"Retry-After": str(retry_after)},
            )
        by_model = readings.get(doc_id, body.image_index)
        fresh = False
        if body.vlm_model and body.vlm_model not in by_model:
            # The cached set may predate this model's sidecar; look once more before a 404.
            by_model = readings.get(doc_id, body.image_index, fresh=True)
            fresh = True
        sidecar = _correction_sidecar(by_model, body, doc_id)
        if body.decoded_at and not same_decode(body.decoded_at, sidecar.get("decoded_at")):
            # Either side may hold the older decode; the server's copy is checked
            # against Elasticsearch once before the suggestion is refused.
            if not fresh:
                sidecar = _correction_sidecar(readings.get(doc_id, body.image_index, fresh=True), body, doc_id)
            if not same_decode(body.decoded_at, sidecar.get("decoded_at")):
                raise HTTPException(
                    status_code=409,
                    detail="This page was re-decoded since you loaded it; reload the readings and pin the line again.",
                )
        line = resolve_line(sidecar, body.reader_id, body.line_index)
        if line is None:
            raise HTTPException(
                status_code=422, detail=f"Line {body.line_index} of reader {body.reader_id} does not exist"
            )
        stored = corrections.record(build_correction(sidecar.get("doc_id") or doc_id, body, sidecar, line))
        logger.info(
            "Line correction %s on %s image %s line %s (%s)",
            stored["id"], doc_id, body.image_index, body.line_index, body.reader_id,
        )
        return {"ok": True, "correction": stored}

    @router.get("/admin/corrections.jsonl", dependencies=[Depends(require_export)], include_in_schema=False)
    def export_corrections(since: Optional[datetime] = Query(None)) -> StreamingResponse:
        """Every suggestion as JSON lines, oldest first, for the Studio to pull.

        :param since: Only suggestions created at or after this ISO time.
        :return: ``application/x-ndjson`` body, one suggestion per line.
        :rtype: StreamingResponse
        """
        rows = corrections.export(since)

        def lines() -> Iterator[str]:
            """Serialise one row per line.

            :return: Iterator of NDJSON lines.
            :rtype: Iterator[str]
            """
            for row in rows:
                yield json.dumps(row, ensure_ascii=False) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    return router
