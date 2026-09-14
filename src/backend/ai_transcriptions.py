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

Search (index v2, ``docs/ai_transcription_search.md``): ``ai_read.lines`` is
a ``nested`` field so a query resolves to the line that matched, and two
derived doc-level fields (``text_agreed``, ``text_all``) carry the joined
text.  All of them use a Hebrew folding analyzer (points stripped, final
forms folded) with a ``.confusable`` subfield that additionally folds the
letter pairs both readers confuse.  None of this is wired into the site's
default catalogue search; it is a separate, opt-in endpoint.

Evidence and rationale: ``docs/planned_features/ai-transcriptions.md``
(probe of 2026-09-08, rules ``lines-v1-20260908`` and ``lines-v2-20260909``).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from elasticsearch import BadRequestError, Elasticsearch, NotFoundError
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Short-lived cache of surfaced doc ids per index (the set grows while a pilot
# batch loads, so a long TTL would hide freshly published fragments).
_SURFACED_TTL_S = 120
_surfaced_cache: Dict[str, Tuple[float, List[str]]] = {}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Serving default until the cutover (``AI_TRANSCRIPTIONS_INDEX`` overrides).
DEFAULT_INDEX = "genizah_ai_transcriptions_v1"
# Index the loader writes by default: the searchable v2 mapping (nested lines,
# Hebrew folding analyzer).  v1 stays live until AI_TRANSCRIPTIONS_INDEX is
# pointed here and the backend is rebuilt; see docs/ai_transcription_search.md.
SEARCH_INDEX = "genizah_ai_transcriptions_v2"

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


def catalogue_index_name() -> Optional[str]:
    """Name of the merged catalogue index the side index's ``doc_id`` points into.

    :return: ``ELASTICSEARCH_INDEX`` or ``None`` when unset.
    :rtype: Optional[str]
    """
    return os.getenv("ELASTICSEARCH_INDEX") or None


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


def read_key(doc_id: str, image_index: int) -> str:
    """Key shared by every checkpoint's read of the same image.

    Search collapses on it so a visitor sees one card per image even when
    two VLM checkpoints have read it.

    :param doc_id: Fragment id.
    :param image_index: Which image of the fragment.
    :return: ``"<doc_id>__<image_index>"``.
    :rtype: str
    """
    return f"{doc_id}__{image_index}"


# ---------------------------------------------------------------------------
# Hebrew folding (shared by the ES analyzer definition and its Python mirror)
# ---------------------------------------------------------------------------

# Hebrew points, cantillation and other combining marks: U+0591..U+05C7.
POINTS_PATTERN = "[\\u0591-\\u05C7]"
# Final forms fold to their medial letter so a query typed either way matches.
FINAL_FORMS: Dict[str, str] = {"\u05da": "\u05db", "\u05dd": "\u05de", "\u05df": "\u05e0", "\u05e3": "\u05e4", "\u05e5": "\u05e6"}
# Letter pairs both readers confuse (probe of 2026-09-08: resh/dalet,
# kaf/bet, samekh/final-mem).  Applied AFTER the final-form fold, so samekh
# folds to medial mem, where final mem already landed.
CONFUSABLES: Dict[str, str] = {"\u05e8": "\u05d3", "\u05db": "\u05d1", "\u05e1": "\u05de"}

_POINTS_RE = __import__("re").compile("[\u0591-\u05c7]")


def fold_hebrew(text: str, confusable: bool = False) -> str:
    """Python mirror of the index's ``hebrew_fold`` / ``hebrew_confusable`` analyzers.

    Used by tests and by anything that needs to compare a query with a
    stored line the way Elasticsearch does: points and cantillation are
    stripped, final forms are folded and, optionally, the shared reader
    confusions are folded too.  Tokenisation is not mirrored.

    :param text: Raw Hebrew (or mixed) text.
    :param confusable: Also apply :data:`CONFUSABLES`.
    :return: Folded, lower-cased text.
    :rtype: str
    """
    out = _POINTS_RE.sub("", text)
    out = "".join(FINAL_FORMS.get(ch, ch) for ch in out)
    if confusable:
        out = "".join(CONFUSABLES.get(ch, ch) for ch in out)
    return out.lower()


def _mapping_rules(table: Dict[str, str]) -> List[str]:
    """Render a fold table as ES ``mapping`` char_filter rules.

    :param table: ``{from_char: to_char}``.
    :return: ``["a=>b", ...]``.
    :rtype: List[str]
    """
    return [f"{src}=>{dst}" for src, dst in table.items()]


ANALYSIS: Dict[str, Any] = {
    "char_filter": {
        "hebrew_strip_points": {
            "type": "pattern_replace",
            "pattern": POINTS_PATTERN,
            "replacement": "",
        },
        "hebrew_final_forms": {"type": "mapping", "mappings": _mapping_rules(FINAL_FORMS)},
        "hebrew_confusables": {"type": "mapping", "mappings": _mapping_rules(CONFUSABLES)},
    },
    "analyzer": {
        "hebrew_fold": {
            "type": "custom",
            "tokenizer": "standard",
            "char_filter": ["hebrew_strip_points", "hebrew_final_forms"],
            "filter": ["lowercase"],
        },
        "hebrew_confusable": {
            "type": "custom",
            "tokenizer": "standard",
            "char_filter": ["hebrew_strip_points", "hebrew_final_forms", "hebrew_confusables"],
            "filter": ["lowercase"],
        },
    },
}

# A searchable Hebrew text field: folded by default, with a lower-precision
# ``.confusable`` subfield used only as a low-boost recall fallback.
HEBREW_TEXT_FIELD: Dict[str, Any] = {
    "type": "text",
    "analyzer": "hebrew_fold",
    "fields": {"confusable": {"type": "text", "analyzer": "hebrew_confusable"}},
}


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
    text_all: str = Field("", description="All lines joined with newlines (derived); NOT in default site search")
    text_agreed: str = Field(
        "", description="Agreed lines only, joined with newlines (derived); what AI-transcription search matches by default"
    )
    read_key: str = Field("", description="<doc_id>__<image_index> (derived); search collapses on it")
    surfaced: bool = Field(False, description="Passes the surfacing rule (derived)")
    published: bool = Field(True, description="Maintainer switch; visible only if also surfaced")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive(self) -> "AiTranscriptionRecord":
        """Derive the joined text fields, ``read_key`` and ``surfaced`` from the sidecar.

        The text fields are always recomputed so a stored value can never
        drift from the lines (v1 records carry a ``text`` field instead,
        which pydantic ignores).
        """
        lines = self.ai_read.lines
        self.text_all = "\n".join(line.text for line in lines)
        self.text_agreed = "\n".join(line.text for line in lines if line.status == "agreed")
        self.read_key = read_key(self.doc_id, self.image_index)
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


class AiLineHit(BaseModel):
    """One matched line of a search hit (from the nested ``inner_hits``)."""

    index: int
    text: str
    highlight: Optional[str] = Field(None, description="Line text with <em> around the matched terms")
    agreement: Optional[float] = None
    status: LineStatus
    bbox: List[int]


class AiTranscriptionSearchHit(BaseModel):
    """One image of one fragment whose AI read matched, with its matched lines."""

    doc_id: str
    source_index: str
    image_index: int
    image_url: str
    vlm_model: str
    n_lines: int
    n_agreed: int
    score: float
    shelf_mark: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    lines: List[AiLineHit] = Field(default_factory=list)


class AiModelFacet(BaseModel):
    """Record count per VLM checkpoint among the matching records."""

    vlm_model: str
    count: int


class AiTranscriptionSearchResponse(BaseModel):
    """Response of ``POST /search-ai-transcriptions``.

    ``enabled``/``available`` mirror :class:`AiTranscriptionStatus` so the
    disabled shape is the same one the status endpoint returns.
    """

    enabled: bool
    available: bool
    query: str = ""
    include_unconfirmed: bool = False
    vlm_model: Optional[str] = None
    total: int = Field(0, description="Matching images (one card each)")
    total_records: int = Field(0, description="Matching records before collapsing checkpoints")
    limit: int = 10
    offset: int = 0
    has_more: bool = False
    index_name: Optional[str] = None
    models: List[AiModelFacet] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None
    message: Optional[str] = None
    results: List[AiTranscriptionSearchHit] = Field(default_factory=list)


LINES_PATH = "ai_read.lines"
LINE_TEXT = f"{LINES_PATH}.text"
LINE_TEXT_CONFUSABLE = f"{LINE_TEXT}.confusable"
INNER_HITS_NAME = "lines"
MAX_LINES_PER_HIT = 3
# At most one edit per term, and only for terms of 4+ letters: a 3-letter
# Hebrew token has too many 1-edit neighbours that are real words.
FUZZINESS = "AUTO:4,99"


def build_search_query(
    query: str,
    include_unconfirmed: bool = False,
    vlm_model: Optional[str] = None,
    lines_per_hit: int = MAX_LINES_PER_HIT,
) -> Dict[str, Any]:
    """Elasticsearch ``query`` for a full-text search over AI line reads.

    Every hit must contain a matching *line* (nested clause, so the response
    can point at the line's box).  By default only ``agreed`` lines are
    searched, and the doc-level ``text_agreed`` phrase clause only adds
    score; ``include_unconfirmed`` widens both to all lines / ``text_all``.
    Within a line: exact phrase (boost 3) > all terms with at most one edit
    per term, and none for tokens under four letters (Hebrew tokens are
    short, so plain ``AUTO`` is too loose) > all terms on the
    confusable-folded subfield at a low boost, for recall only.

    :param query: Visitor's query, Hebrew expected.
    :param include_unconfirmed: Search unconfirmed lines too.
    :param vlm_model: Restrict to one VLM checkpoint.
    :param lines_per_hit: Inner hits returned per record.
    :return: Query clause including the nested ``inner_hits`` with highlighting.
    :rtype: Dict[str, Any]
    """
    line_should: List[Dict[str, Any]] = [
        {"match_phrase": {LINE_TEXT: {"query": query, "boost": 3.0}}},
        {"match": {LINE_TEXT: {"query": query, "operator": "and", "fuzziness": FUZZINESS, "prefix_length": 1}}},
        {"match": {LINE_TEXT_CONFUSABLE: {"query": query, "operator": "and", "boost": 0.25}}},
    ]
    line_bool: Dict[str, Any] = {"should": line_should, "minimum_should_match": 1}
    if not include_unconfirmed:
        line_bool["filter"] = [{"term": {f"{LINES_PATH}.status": "agreed"}}]
    nested = {
        "nested": {
            "path": LINES_PATH,
            "score_mode": "max",
            "query": {"bool": line_bool},
            "inner_hits": {
                "name": INNER_HITS_NAME,
                "size": lines_per_hit,
                "_source": [f"{LINES_PATH}.{f}" for f in ("index", "text", "agreement", "status", "bbox")],
                "highlight": {
                    "fields": {LINE_TEXT: {}, LINE_TEXT_CONFUSABLE: {}},
                    "number_of_fragments": 0,
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                },
            },
        }
    }
    filters: List[Dict[str, Any]] = [{"term": {"surfaced": True}}, {"term": {"published": True}}]
    if vlm_model:
        filters.append({"term": {"ai_read.vlm_model": vlm_model}})
    doc_field = "text_all" if include_unconfirmed else "text_agreed"
    return {
        "bool": {
            "filter": filters,
            "must": [nested],
            "should": [{"match_phrase": {doc_field: {"query": query, "boost": 2.0}}}],
        }
    }


def parse_line_hits(hit: Dict[str, Any]) -> List[AiLineHit]:
    """Turn a hit's nested ``inner_hits`` into :class:`AiLineHit` objects.

    The highlight on the folded field wins; the confusable subfield's
    highlight is the fallback when only that clause matched.

    :param hit: One entry of ``hits.hits`` from :func:`build_search_query`.
    :return: Matched lines in score order.
    :rtype: List[AiLineHit]
    """
    inner = hit.get("inner_hits", {}).get(INNER_HITS_NAME, {}).get("hits", {}).get("hits", [])
    lines: List[AiLineHit] = []
    for entry in inner:
        source = entry["_source"]
        highlights = entry.get("highlight", {})
        marked = highlights.get(LINE_TEXT) or highlights.get(LINE_TEXT_CONFUSABLE) or []
        lines.append(
            AiLineHit(
                index=source["index"],
                text=source["text"],
                highlight=marked[0] if marked else None,
                agreement=source.get("agreement"),
                status=source["status"],
                bbox=source["bbox"],
            )
        )
    return lines


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

# v2 (2026-09-14): ``ai_read.lines`` is nested and searchable, the joined
# text is split into ``text_agreed`` / ``text_all`` (v1's ``text`` is gone),
# and ``read_key`` lets search collapse to one card per image.  v1's mapping
# was ``lines: {type: object, enabled: false}`` with a plain ``text`` field;
# the strict mapping means the loader cannot accidentally write v2 records
# into v1.  Rationale: docs/ai_transcription_search.md.
INDEX_MAPPING: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": ANALYSIS},
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
                    # Nested so a hit resolves to the line that matched
                    # (inner_hits) and ``status`` can be filtered per line.
                    "lines": {
                        "type": "nested",
                        "properties": {
                            "index": {"type": "integer"},
                            "text": HEBREW_TEXT_FIELD,
                            "status": {"type": "keyword"},
                            "agreement": {"type": "float"},
                            "bbox": {"type": "integer"},
                            # Kraken evidence: kept in _source for the viewer,
                            # never searched.
                            "htr_text": {"type": "keyword", "index": False, "doc_values": False},
                            "htr_fragments": {"type": "integer", "index": False, "doc_values": False},
                        },
                    },
                }
            },
            # Doc-level joined text.  Neither field is part of the site's
            # default catalogue search: unconfirmed lines can be hallucinated
            # and even agreed lines are machine reads.
            "text_all": HEBREW_TEXT_FIELD,
            "text_agreed": HEBREW_TEXT_FIELD,
            "read_key": {"type": "keyword"},
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
    :param catalogue_index: Merged index used to label search hits with
        shelf mark / title / description (defaults to
        :func:`catalogue_index_name`; ``None`` skips the join).
    """

    def __init__(
        self, es: Elasticsearch, index: Optional[str] = None, catalogue_index: Optional[str] = None
    ) -> None:
        self.es = es
        self.index = index or index_name()
        self.catalogue_index = catalogue_index or catalogue_index_name()

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
                source_excludes=["ai_read.lines", "text", "text_all", "text_agreed"],
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

    # ------------------------------------------------------------------
    # Full-text search over line reads (index v2)
    # ------------------------------------------------------------------

    def catalogue_metadata(self, doc_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Shelf mark, title and description for hits, in one ``mget``.

        :param doc_ids: Fragment ids (Elasticsearch ``_id`` in the merged index).
        :return: ``{doc_id: {"shelf_mark", "title", "description"}}`` for the
            ids that were found; empty when there is no catalogue index.
        :rtype: Dict[str, Dict[str, Any]]
        """
        ids = list(dict.fromkeys(doc_ids))
        if not ids or not self.catalogue_index:
            return {}
        try:
            response = self.es.mget(
                index=self.catalogue_index,
                ids=ids,
                source_includes=["shelf_mark", "classmark", "title", "description"],
            )
        except NotFoundError:
            logger.info("Catalogue index %s not found; hits will be unlabelled", self.catalogue_index)
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for doc in response.get("docs", []):
            if not doc.get("found"):
                continue
            source = doc.get("_source", {})
            out[doc["_id"]] = {
                "shelf_mark": source.get("shelf_mark") or source.get("classmark"),
                "title": source.get("title"),
                "description": source.get("description"),
            }
        return out

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        include_unconfirmed: bool = False,
        vlm_model: Optional[str] = None,
    ) -> AiTranscriptionSearchResponse:
        """Full-text search over the line reads, one card per image.

        Records of the same image by different checkpoints are collapsed on
        ``read_key`` (the best-scoring record is shown; ``vlm_model`` pins
        one).  A missing index, or a v1 index without the nested mapping,
        yields ``available: false`` rather than an error, so the endpoint
        is safe to deploy before the v2 cutover.

        :param query: Visitor's query.
        :param limit: Page size.
        :param offset: Cards to skip.
        :param include_unconfirmed: Search unconfirmed lines too.
        :param vlm_model: Restrict to one VLM checkpoint.
        :return: Cards with their matched lines and catalogue labels.
        :rtype: AiTranscriptionSearchResponse
        """
        base = dict(
            query=query, include_unconfirmed=include_unconfirmed, vlm_model=vlm_model,
            limit=limit, offset=offset, index_name=self.index,
        )
        if not feature_enabled():
            return AiTranscriptionSearchResponse(enabled=False, available=False, **base)
        started = time.time()
        try:
            response = self.es.search(
                index=self.index,
                query=build_search_query(query, include_unconfirmed, vlm_model),
                collapse={"field": "read_key"},
                aggs={
                    "images": {"cardinality": {"field": "read_key", "precision_threshold": 10000}},
                    "models": {"terms": {"field": "ai_read.vlm_model", "size": 20}},
                },
                sort=[{"_score": "desc"}, {"ai_read.decoded_at": "desc"}],
                source_includes=[
                    "doc_id", "source_index", "image_index", "image_url",
                    "ai_read.vlm_model", "ai_read.n_lines", "ai_read.n_agreed",
                ],
                from_=offset,
                size=limit,
                track_total_hits=True,
            )
        except NotFoundError:
            return AiTranscriptionSearchResponse(
                enabled=True, available=False, message=f"index {self.index} does not exist", **base
            )
        except BadRequestError as exc:
            # v1 mapping (lines not nested / no text_agreed): searchable only after the cutover.
            logger.warning("AI transcription search unavailable on %s: %s", self.index, exc)
            return AiTranscriptionSearchResponse(
                enabled=True, available=False, message=f"index {self.index} is not searchable", **base
            )
        hits = response["hits"]["hits"]
        labels = self.catalogue_metadata([h["_source"]["doc_id"] for h in hits])
        results: List[AiTranscriptionSearchHit] = []
        for hit in hits:
            source = hit["_source"]
            results.append(
                AiTranscriptionSearchHit(
                    doc_id=source["doc_id"],
                    source_index=source["source_index"],
                    image_index=source["image_index"],
                    image_url=source["image_url"],
                    vlm_model=source["ai_read"]["vlm_model"],
                    n_lines=source["ai_read"]["n_lines"],
                    n_agreed=source["ai_read"]["n_agreed"],
                    score=hit.get("_score") or 0.0,
                    lines=parse_line_hits(hit),
                    **labels.get(source["doc_id"], {}),
                )
            )
        aggs = response.get("aggregations", {})
        total = int(aggs.get("images", {}).get("value", len(results)))
        models = [
            AiModelFacet(vlm_model=b["key"], count=b["doc_count"])
            for b in aggs.get("models", {}).get("buckets", [])
        ]
        return AiTranscriptionSearchResponse(
            enabled=True,
            available=True,
            total=total,
            total_records=int(response["hits"]["total"]["value"]),
            has_more=offset + len(results) < total,
            models=models,
            processing_time_ms=int((time.time() - started) * 1000),
            results=results,
            **base,
        )
