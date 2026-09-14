"""AI reads: schema, surfacing rule and the read-only serving layer.

Reads are computed **offline** in the sibling ``historical-document-analysis``
pipeline: the site's Kraken microservice segments and reads lines, the
fine-tuned Qwen3-VL checkpoint reads the page with grounded line boxes, and
the two are matched line by line.  A line whose two readings agree
(letters-only similarity >= :data:`AGREED_MIN`) is *agreed*; everything else
is *unconfirmed*.  Nothing in this module calls a model: the website only
reads finished records, so a visitor never ties up LM Studio.

The pipeline writes one sidecar per image (``ai_read``); the loader wraps it
in an envelope naming the document and image and derives the surfacing flag.
Records are keyed by ``(doc_id, image_index, ai_read.vlm_model)`` so
recto/verso images and successive checkpoints coexist.

Coordinate convention: every ``bbox`` is ``[x1, y1, x2, y2]`` in the 0-1000
space normalised to the width/height of the image the readers received, i.e.
the file at ``image_url`` with EXIF orientation applied.  Since rule v2 the
box is the *evidence box*: the union of the VLM's line box and the Kraken
fragments assigned to that line (the VLM box alone when there are none).  The
VLM's own boxes are a layout prior (one x-range stepped down the page on most
pages), so the fragments are what make the box meaningful.  The viewer scales
boxes by whatever size it displays the same file at.

Evidence and rationale: ``docs/planned_features/ai-transcriptions.md``
(probe of 2026-09-08, rules ``lines-v1-20260908`` and ``lines-v2-20260909``).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from elasticsearch import Elasticsearch, NotFoundError
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Short-lived cache of surfaced doc ids per index (the set grows while a pilot
# batch loads, so a long TTL would hide freshly published fragments).
_SURFACED_TTL_S = 120
_surfaced_cache: Dict[str, Tuple[float, List[str]]] = {}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INDEX = "genizah_ai_transcriptions_v1"

LineStatus = Literal["agreed", "unconfirmed"]

# Agreement threshold tau from the 2026-09-08 probe (24 KTIV pages, 294
# scorable lines): at 0.8 nothing with CER > 0.5 leaks, 84% of agreed lines
# are at CER <= 0.10 and the box sits on the right line 96% of the time.
# The same rule on v1.9a gives the same precision, so it is not checkpoint
# specific.  Change ONLY together with a new rule_version in the pipeline.
RULE_VERSION = "lines-v2-20260909"
# Every rule version the site accepts.  All of them share AGREED_MIN and the
# same sidecar schema; v2 (2026-09-09) widens each line box to the union of the
# VLM box and its Kraken fragments and assigns fragments more permissively.
ACCEPTED_RULE_VERSIONS = frozenset({"lines-v1-20260908", RULE_VERSION})
AGREED_MIN = 0.8

# Surfacing rule: a read is shown to visitors only when the JSON parsed and it
# has at least this many agreed lines, or this share of its lines agreed.
DEFAULT_MIN_AGREED_LINES = 3
DEFAULT_MIN_AGREED_SHARE = 0.25


def index_name() -> str:
    """Name of the Elasticsearch side index holding AI reads.

    :return: Index name from ``AI_TRANSCRIPTIONS_INDEX`` or the default.
    :rtype: str
    """
    return os.getenv("AI_TRANSCRIPTIONS_INDEX", DEFAULT_INDEX)


def feature_enabled() -> bool:
    """Whether the AI read endpoints are switched on.

    :return: ``False`` only when ``AI_TRANSCRIPTIONS_ENABLED`` is a falsy string.
    :rtype: bool
    """
    return os.getenv("AI_TRANSCRIPTIONS_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def surfacing_thresholds() -> Dict[str, float]:
    """Thresholds of the surfacing rule.

    :return: ``{"min_agreed_lines": int, "min_agreed_share": float}``.
    :rtype: Dict[str, float]
    """
    return {
        "min_agreed_lines": int(os.getenv("AI_READ_MIN_AGREED_LINES", DEFAULT_MIN_AGREED_LINES)),
        "min_agreed_share": float(os.getenv("AI_READ_MIN_AGREED_SHARE", DEFAULT_MIN_AGREED_SHARE)),
    }


def line_status(agreement: Optional[float], htr_text: Optional[str]) -> LineStatus:
    """Derive a line's status from its cross-reader agreement.

    :param agreement: ``1 - Levenshtein / max(len)`` on Hebrew letters only
        between the VLM and Kraken readings, or ``None`` when Kraken produced
        no fragment for the line.
    :param htr_text: Kraken's reading (``None``/empty when absent).
    :return: ``"agreed"`` when both readers concur, else ``"unconfirmed"``.
    :rtype: LineStatus
    """
    if agreement is None or not (htr_text or "").strip():
        return "unconfirmed"
    return "agreed" if agreement >= AGREED_MIN else "unconfirmed"


def is_surfaced(
    parsed: bool, n_lines: int, n_agreed: int, thresholds: Optional[Dict[str, float]] = None
) -> bool:
    """Apply the surfacing rule.

    :param parsed: Whether the VLM reply was a parseable JSON array.
    :param n_lines: Number of lines in the read.
    :param n_agreed: Number of agreed lines.
    :param thresholds: Override for :func:`surfacing_thresholds`.
    :return: ``True`` when visitors may see the read.
    :rtype: bool
    """
    thr = thresholds or surfacing_thresholds()
    if not parsed or n_lines <= 0 or n_agreed <= 0:
        return False
    return n_agreed >= thr["min_agreed_lines"] or n_agreed >= thr["min_agreed_share"] * n_lines


def record_id(doc_id: str, image_index: int, vlm_model: str) -> str:
    """Deterministic Elasticsearch ``_id`` for a record.

    :param doc_id: Elasticsearch ``_id`` of the fragment in the merged index.
    :param image_index: Position of the image in the fragment's ``image_urls``.
    :param vlm_model: LM Studio key of the VLM checkpoint used.
    :return: ``"<doc_id>__<image_index>__<vlm_model>"``.
    :rtype: str
    """
    return f"{doc_id}__{image_index}__{vlm_model}"


# ---------------------------------------------------------------------------
# Schema (mirrors the pipeline sidecar)
# ---------------------------------------------------------------------------


def clamp_bbox(bbox: Sequence[float]) -> List[int]:
    """Clamp a 0-1000 box to range and order its corners.

    :param bbox: ``[x1, y1, x2, y2]`` in the 0-1000 normalised space.
    :return: Integer box with ``x1 <= x2`` and ``y1 <= y2``.
    :rtype: List[int]
    :raises ValueError: If the box does not have exactly four values.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    x1, y1, x2, y2 = (int(round(max(0.0, min(1000.0, float(v))))) for v in bbox)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


class AiLine(BaseModel):
    """One VLM line with its box and the matching Kraken evidence."""

    index: int = Field(..., ge=0, description="0-based position in reading order")
    text: str = Field(..., description="VLM reading of the line")
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2], 0-1000 normalised")
    agreement: Optional[float] = Field(
        None, ge=0, le=1, description="1 - Levenshtein/max(len), Hebrew letters only"
    )
    status: LineStatus = Field("unconfirmed")
    htr_text: Optional[str] = Field(None, description="Kraken fragments joined right-to-left")
    htr_fragments: List[List[int]] = Field(
        default_factory=list, description="Kraken fragment boxes assigned to this line"
    )

    @field_validator("bbox")
    @classmethod
    def _clamp(cls, value: Sequence[float]) -> List[int]:
        return clamp_bbox(value)

    @field_validator("htr_fragments")
    @classmethod
    def _clamp_fragments(cls, value: Sequence[Sequence[float]]) -> List[List[int]]:
        return [clamp_bbox(box) for box in value]

    @model_validator(mode="after")
    def _status_matches_rule(self) -> "AiLine":
        """Reject lines whose status disagrees with the agreement rule."""
        expected = line_status(self.agreement, self.htr_text)
        if self.status != expected:
            raise ValueError(
                f"line {self.index}: status {self.status!r} but rule {RULE_VERSION} gives {expected!r}"
            )
        return self


class AiRead(BaseModel):
    """The pipeline sidecar for one image, stored verbatim."""

    vlm_model: str = Field(..., description="LM Studio key of the VLM checkpoint")
    vlm_revision: Optional[str] = Field(None, description="Git revision of the checkpoint recipe")
    htr_model: str = Field(..., description="Kraken model name")
    rule_version: str = Field(..., description="Matching/agreement rule id")
    decoded_at: datetime
    parsed: bool = Field(..., description="VLM reply was a parseable JSON array")
    n_lines: int = Field(0, ge=0)
    n_agreed: int = Field(0, ge=0)
    lines: List[AiLine] = Field(default_factory=list)

    @field_validator("rule_version")
    @classmethod
    def _known_rule(cls, value: str) -> str:
        if value not in ACCEPTED_RULE_VERSIONS:
            raise ValueError(
                f"rule_version {value!r} is not accepted (known: {sorted(ACCEPTED_RULE_VERSIONS)})"
            )
        return value

    @model_validator(mode="after")
    def _counts(self) -> "AiRead":
        """Fill ``n_lines``/``n_agreed`` from ``lines`` and check they agree."""
        agreed = sum(1 for line in self.lines if line.status == "agreed")
        if self.n_lines == 0 and self.n_agreed == 0:
            self.n_lines = len(self.lines)
            self.n_agreed = agreed
        elif (self.n_lines, self.n_agreed) != (len(self.lines), agreed):
            raise ValueError(
                f"n_lines/n_agreed {self.n_lines}/{self.n_agreed} do not match lines {len(self.lines)}/{agreed}"
            )
        indices = [line.index for line in self.lines]
        if indices != list(range(len(indices))):
            raise ValueError("line indices must be 0..n-1 in reading order")
        return self


class AiTranscriptionRecord(BaseModel):
    """Envelope + sidecar for one image of one fragment."""

    doc_id: str = Field(..., description="ES _id of the fragment in the merged index")
    source_index: str = Field(..., description="Merged index the doc_id belongs to")
    image_index: int = Field(..., ge=0)
    image_url: str = Field(..., description="Exact URL of the rendition that was read")
    image_width: int = Field(..., gt=0, description="Pixel width after EXIF orientation")
    image_height: int = Field(..., gt=0, description="Pixel height after EXIF orientation")
    image_sha256: Optional[str] = Field(None, description="Hash of the bytes that were read")
    ai_read: AiRead
    text: str = Field("", description="Lines joined with newlines; NOT in default site search")
    surfaced: bool = Field(False, description="Passes the surfacing rule (derived)")
    published: bool = Field(True, description="Maintainer switch; visible only if also surfaced")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive(self) -> "AiTranscriptionRecord":
        """Derive ``text`` and ``surfaced`` from the sidecar."""
        if not self.text:
            self.text = "\n".join(line.text for line in self.ai_read.lines)
        self.surfaced = is_surfaced(
            self.ai_read.parsed, self.ai_read.n_lines, self.ai_read.n_agreed
        )
        return self

    def es_id(self) -> str:
        """Return the deterministic ``_id`` for this record.

        :return: See :func:`record_id`.
        :rtype: str
        """
        return record_id(self.doc_id, self.image_index, self.ai_read.vlm_model)


class AiReadSummary(BaseModel):
    """Sidecar header without the lines."""

    vlm_model: str
    vlm_revision: Optional[str] = None
    htr_model: str
    rule_version: str
    decoded_at: datetime
    parsed: bool
    n_lines: int
    n_agreed: int


class AiTranscriptionSummary(BaseModel):
    """Lightweight view of a record for the availability endpoint."""

    doc_id: str
    image_index: int
    image_url: str
    ai_read: AiReadSummary
    created_at: datetime


class AiTranscriptionStatus(BaseModel):
    """Response of ``GET /ai-transcriptions/{doc_id}``."""

    doc_id: str
    enabled: bool
    available: bool
    items: List[AiTranscriptionSummary] = Field(default_factory=list)


def build_lines(raw_lines: Sequence[Dict[str, Any]]) -> List[AiLine]:
    """Turn loose pipeline line dicts into :class:`AiLine` objects.

    ``index`` and ``status`` are filled in when missing; when present they
    are validated against the rule.

    :param raw_lines: Line dictionaries in reading order.
    :return: Validated lines.
    :rtype: List[AiLine]
    """
    lines: List[AiLine] = []
    for i, raw in enumerate(raw_lines):
        payload = dict(raw)
        payload.setdefault("index", i)
        payload.setdefault("status", line_status(raw.get("agreement"), raw.get("htr_text")))
        lines.append(AiLine(**payload))
    return lines


# ---------------------------------------------------------------------------
# Elasticsearch mapping
# ---------------------------------------------------------------------------

INDEX_MAPPING: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": {"type": "keyword"},
            "source_index": {"type": "keyword"},
            "image_index": {"type": "integer"},
            "image_url": {"type": "keyword"},
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "image_sha256": {"type": "keyword"},
            "ai_read": {
                "properties": {
                    "vlm_model": {"type": "keyword"},
                    "vlm_revision": {"type": "keyword"},
                    "htr_model": {"type": "keyword"},
                    "rule_version": {"type": "keyword"},
                    "decoded_at": {"type": "date"},
                    "parsed": {"type": "boolean"},
                    "n_lines": {"type": "integer"},
                    "n_agreed": {"type": "integer"},
                    # Stored verbatim for the viewer; switch to ``nested`` if
                    # per-line search is ever wanted.
                    "lines": {"type": "object", "enabled": False},
                }
            },
            # Indexed but deliberately NOT part of the site's default search:
            # unconfirmed lines can be hallucinated.
            "text": {"type": "text"},
            "surfaced": {"type": "boolean"},
            "published": {"type": "boolean"},
            "created_at": {"type": "date"},
        },
    },
}


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


class AiTranscriptionService:
    """Read-only access to surfaced, published AI reads.

    :param es: Elasticsearch client shared with the search service.
    :param index: Side index name (defaults to :func:`index_name`).
    """

    def __init__(self, es: Elasticsearch, index: Optional[str] = None) -> None:
        self.es = es
        self.index = index or index_name()

    @staticmethod
    def _visible_query(doc_id: str) -> Dict[str, Any]:
        """Query matching records the public viewer may show.

        :param doc_id: Fragment id.
        :return: Elasticsearch ``query`` clause.
        :rtype: Dict[str, Any]
        """
        return {
            "bool": {
                "filter": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"published": True}},
                    {"term": {"surfaced": True}},
                ]
            }
        }

    def list_for_document(self, doc_id: str) -> List[AiTranscriptionSummary]:
        """Summaries of every visible record for a fragment, newest first.

        A missing index means the feature has no data yet and is reported as
        "nothing available" rather than as an error.

        :param doc_id: Fragment id.
        :return: Summaries ordered by ``created_at`` descending.
        :rtype: List[AiTranscriptionSummary]
        """
        try:
            response = self.es.search(
                index=self.index,
                query=self._visible_query(doc_id),
                sort=[{"created_at": {"order": "desc"}}],
                source_excludes=["ai_read.lines", "text"],
                size=50,
            )
        except NotFoundError:
            logger.info("AI transcription index %s does not exist yet", self.index)
            return []
        return [
            AiTranscriptionSummary(**hit["_source"]) for hit in response["hits"]["hits"]
        ]

    def get(
        self, doc_id: str, image_index: int, vlm_model: Optional[str] = None
    ) -> Optional[AiTranscriptionRecord]:
        """Fetch one full record.

        :param doc_id: Fragment id.
        :param image_index: Which image of the fragment.
        :param vlm_model: Specific checkpoint; newest visible record when omitted.
        :return: The record, or ``None`` when nothing visible matches.
        :rtype: Optional[AiTranscriptionRecord]
        """
        query = self._visible_query(doc_id)
        query["bool"]["filter"].append({"term": {"image_index": image_index}})
        if vlm_model:
            query["bool"]["filter"].append({"term": {"ai_read.vlm_model": vlm_model}})
        try:
            response = self.es.search(
                index=self.index,
                query=query,
                sort=[{"created_at": {"order": "desc"}}],
                size=1,
            )
        except NotFoundError:
            return None
        hits = response["hits"]["hits"]
        if not hits:
            return None
        return AiTranscriptionRecord(**hits[0]["_source"])

    def status(self, doc_id: str) -> AiTranscriptionStatus:
        """Availability payload for the document page's button.

        :param doc_id: Fragment id.
        :return: Whether the feature is on and which images have reads.
        :rtype: AiTranscriptionStatus
        """
        if not feature_enabled():
            return AiTranscriptionStatus(doc_id=doc_id, enabled=False, available=False)
        items = self.list_for_document(doc_id)
        return AiTranscriptionStatus(
            doc_id=doc_id, enabled=True, available=bool(items), items=items
        )

    def surfaced_doc_ids(self, force: bool = False) -> List[str]:
        """Fragment ids that have at least one visible AI read.

        Used by the search filter to select documents with an AI transcription.
        Cached briefly (``_SURFACED_TTL_S``) because the pilot set grows while a
        batch is loading; a missing index yields an empty list, not an error.

        :param force: Ignore the cache and re-query.
        :return: Sorted unique fragment ids.
        :rtype: List[str]
        """
        now = time.time()
        cached = _surfaced_cache.get(self.index)
        if not force and cached and now - cached[0] < _SURFACED_TTL_S:
            return cached[1]
        ids: List[str] = []
        query = {
            "bool": {"filter": [{"term": {"published": True}}, {"term": {"surfaced": True}}]}
        }
        after: Optional[List[Any]] = None
        try:
            while True:
                kwargs: Dict[str, Any] = dict(
                    index=self.index, query=query, _source=["doc_id"],
                    sort=[{"doc_id": "asc"}], size=2000,
                )
                if after:
                    kwargs["search_after"] = after
                hits = self.es.search(**kwargs)["hits"]["hits"]
                if not hits:
                    break
                ids.extend(h["_source"]["doc_id"] for h in hits)
                after = hits[-1]["sort"]
        except NotFoundError:
            ids = []
        ids = sorted(set(ids))
        _surfaced_cache[self.index] = (now, ids)
        return ids
