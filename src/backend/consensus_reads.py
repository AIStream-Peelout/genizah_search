"""Consensus lines: the agreed part of the AI reads, for public pages and search.

An AI read (``src/backend/ai_transcriptions.py``) marks a line *agreed* when
two independent machine readers (Kraken and the fine-tuned Qwen3-VL) produced
the same text.  This module turns raw records of the AI side index into the
small, cleaned view that public ``/fragment`` pages and the site-search
fallback show:

* **Visibility is recomputed from the data**, never trusted from stored
  counters: the record must be ``published``, carry the stored ``surfaced``
  flag the ``/read`` viewer filters on, have parsed, pass
  :func:`~src.backend.ai_transcriptions.is_surfaced` on counts recomputed from
  the lines (``status == "agreed"``), and keep at least one agreed line after
  :func:`clean_agreed_line`.  (72 production records carry ``surfaced: true``
  with zero agreed lines; this rejects them.)
* **Only agreed lines** are exposed, cleaned of reader junk (Latin, CJK or
  Cyrillic fragments the VLM sometimes emits inside Hebrew text).
* **Old Oxford ids** (``Oxford_Bodleian_MS_heb_*``, loaded from merged v5/v6)
  are mapped to the catalogue's ``Oxford_Bodleian_Bodl_MS_heb_*`` and back.
* **One read per image**: several checkpoints read the same image; the one
  with the most agreed lines wins, then the newest ``ai_read.decoded_at``
  (``created_at`` is reset on every load and says nothing).

The search fallback (:meth:`ConsensusService.search_uncatalogued`) is the
"second pass" of ``docs/ai_transcription_search.md``: agreed lines only
(``text_agreed`` / nested agreed lines, never ``text_all``), restricted to
fragments whose catalogue record is *uncatalogued* (:func:`is_uncatalogued`).
The caller appends its hits strictly after every catalogue result.  Nothing
here writes to Elasticsearch.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from elasticsearch import ApiError, BadRequestError, Elasticsearch, NotFoundError, TransportError

from src.backend.ai_transcriptions import (
    AiTranscriptionService,
    build_search_query,
    feature_enabled,
    fold_hebrew,
    is_surfaced,
    parse_line_hits,
    read_key as make_read_key,
    record_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document ids
# ---------------------------------------------------------------------------

# Records loaded from merged v5/v6 use the old Bodleian ids; merged v8 renamed
# them.  The image lists differ between the two (the v6 image 0 can be the v8
# image 2), so a record keeps its own ``image_index`` and ``ai_doc_id``.
OXFORD_OLD_PREFIX = "Oxford_Bodleian_MS_heb_"
OXFORD_V8_PREFIX = "Oxford_Bodleian_Bodl_MS_heb_"


def to_catalogue_doc_id(ai_doc_id: str) -> str:
    """Map a ``doc_id`` of the AI side index to the catalogue (merged v8) id.

    Only the old Oxford form is rewritten; everything else is already a
    catalogue id (``Oxford_Bodleian_Ms_Heb_d_66_137`` deliberately stays
    unresolved: it has no v8 record).

    :param ai_doc_id: ``doc_id`` as stored in the AI index.
    :return: The corresponding catalogue ``doc_id``.
    :rtype: str
    """
    if ai_doc_id.startswith(OXFORD_OLD_PREFIX) and len(ai_doc_id) > len(OXFORD_OLD_PREFIX):
        return OXFORD_V8_PREFIX + ai_doc_id[len(OXFORD_OLD_PREFIX):]
    return ai_doc_id


def to_ai_doc_ids(catalogue_doc_id: str) -> List[str]:
    """Every ``doc_id`` form under which the AI index may hold reads of a fragment.

    :param catalogue_doc_id: Catalogue ``doc_id`` (an old Oxford id is
        normalised first, so either form works).
    :return: The catalogue id, plus the old Oxford form when applicable.
    :rtype: List[str]
    """
    canonical = to_catalogue_doc_id(catalogue_doc_id)
    forms = [canonical]
    if canonical.startswith(OXFORD_V8_PREFIX) and len(canonical) > len(OXFORD_V8_PREFIX):
        forms.append(OXFORD_OLD_PREFIX + canonical[len(OXFORD_V8_PREFIX):])
    return forms


def read_url(
    ai_doc_id: str, image_index: int, line_index: Optional[int] = None, vlm_model: Optional[str] = None
) -> str:
    """Relative link to the ``/read`` viewer for one image (and optionally one line).

    ``model`` pins the checkpoint: line indices belong to one record, and
    without it the viewer opens whichever checkpoint it lists first.

    :param ai_doc_id: ``doc_id`` as stored in the AI index (the viewer looks
        the record up under that id, so the old Oxford form is kept).
    :param image_index: The record's own image index.
    :param line_index: 0-based line to zoom to.
    :param vlm_model: Checkpoint whose line numbering ``line_index`` uses.
    :return: ``/read?doc=…&image=…[&line=…][&model=…]``, every value percent-encoded.
    :rtype: str
    """
    url = f"/read?doc={quote(ai_doc_id, safe='')}&image={int(image_index)}"
    if line_index is not None:
        url += f"&line={int(line_index)}"
    if vlm_model:
        url += f"&model={quote(vlm_model, safe='')}"
    return url


# ---------------------------------------------------------------------------
# Cleaning agreed lines
# ---------------------------------------------------------------------------

# Characters a kept token may contain: Hebrew letters, points, maqaf,
# geresh/gershayim and presentation forms; Arabic script (Judaeo-Arabic reads
# sometimes switch script mid-word, and that is still a reading); the combining
# dot above used for Judaeo-Arabic diacritic dots (e.g. "אפ̇ט̇ל"); digits; and
# the punctuation editors and the readers use ("[...]", ר"ת, ג'רי, "עולם.").
_ALLOWED_CHARS = (
    "֑-ׇא-תׯ-״יִ-ﭏ"
    "؀-ۿ"
    "̇"
    "0-9"
    ".,:;'\"()\\[\\]\\-–—/?!…·"
)
_DISALLOWED_RE = re.compile(f"[^{_ALLOWED_CHARS}]")
_HEBREW_LETTER_RE = re.compile("[א-תיִ-ﭏ]")
_HEBREW_LETTER_ONLY_RE = re.compile("[^א-ת]")


def _strip_format_chars(text: str) -> str:
    """Remove invisible format characters (RLM/LRM, zero-width spaces, BOM).

    :param text: Raw text.
    :return: Text without Unicode category ``Cf`` characters.
    :rtype: str
    """
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def clean_agreed_line(text: str) -> Optional[str]:
    """Clean an agreed line for public display.

    The text is split on whitespace and every token holding a character
    outside the allowed set (Hebrew script, Arabic script, digits, common
    punctuation) is **dropped whole**: ``"לalmולא"`` disappears rather than
    becoming ``"למולא"``, because gluing the Hebrew halves together would
    print a word neither reader produced.  Invisible format characters are
    removed first so they never cost a token.

    The line is rejected (``None``) when no Hebrew letter is left, or when
    more letter-bearing tokens were dropped than kept (a line that is mostly
    junk is not a trustworthy agreed reading).

    :param text: Agreed line text as stored in ``ai_read.lines[].text``.
    :return: Cleaned line with single spaces, or ``None`` when nothing usable remains.
    :rtype: Optional[str]
    """
    kept: List[str] = []
    dropped_words = 0
    for token in _strip_format_chars(text or "").split():
        if _DISALLOWED_RE.search(token):
            if any(ch.isalpha() for ch in token):
                dropped_words += 1
            continue
        kept.append(token)
    cleaned = " ".join(kept)
    if not _HEBREW_LETTER_RE.search(cleaned):
        return None
    kept_words = sum(1 for token in kept if any(ch.isalpha() for ch in token))
    if dropped_words > kept_words:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Consensus reads
# ---------------------------------------------------------------------------


@dataclass
class ConsensusLine:
    """One cleaned agreed line."""

    index: int
    text: str


@dataclass
class ConsensusRead:
    """The visible, cleaned agreed lines of one image's AI read.

    ``n_agreed``/``n_lines`` are recomputed from the record's lines (the same
    counts the ``/read`` viewer shows); ``lines`` holds the agreed lines that
    survived :func:`clean_agreed_line`, so ``len(lines) <= n_agreed``.
    """

    doc_id: str
    ai_doc_id: str
    image_index: int
    read_key: str
    vlm_model: str
    n_agreed: int
    n_lines: int
    decoded_at: Optional[str]
    image_url: Optional[str]
    lines: List[ConsensusLine] = field(default_factory=list)


def consensus_read(record: Dict[str, Any]) -> Optional[ConsensusRead]:
    """Build the public view of one raw AI-index record, if it is visible.

    :param record: ``_source`` of an AI-index record (needs ``doc_id``,
        ``image_index``, ``published``, ``surfaced``, ``ai_read.parsed`` and
        ``ai_read.lines[].index/text/status``).
    :return: The read, or ``None`` when it is unpublished, not surfaced
        (stored flag or recomputed rule) or has no usable agreed line.
    :rtype: Optional[ConsensusRead]
    """
    ai_read = record.get("ai_read") or {}
    if record.get("published") is not True or record.get("surfaced") is not True:
        return None
    if ai_read.get("parsed") is not True:
        return None
    raw_lines = sorted(
        enumerate(ai_read.get("lines") or []), key=lambda pair: pair[1].get("index", pair[0])
    )
    agreed = [(pos, line) for pos, line in raw_lines if line.get("status") == "agreed"]
    if not is_surfaced(True, len(raw_lines), len(agreed)):
        return None
    lines: List[ConsensusLine] = []
    for pos, line in agreed:
        text = clean_agreed_line(line.get("text") or "")
        if text is not None:
            lines.append(ConsensusLine(index=int(line.get("index", pos)), text=text))
    if not lines:
        return None
    ai_doc_id = record["doc_id"]
    image_index = int(record["image_index"])
    return ConsensusRead(
        doc_id=to_catalogue_doc_id(ai_doc_id),
        ai_doc_id=ai_doc_id,
        image_index=image_index,
        read_key=record.get("read_key") or make_read_key(ai_doc_id, image_index),
        vlm_model=ai_read.get("vlm_model") or "",
        n_agreed=len(agreed),
        n_lines=len(raw_lines),
        decoded_at=ai_read.get("decoded_at"),
        image_url=record.get("image_url"),
        lines=lines,
    )


def _decoded_at_key(decoded_at: Optional[str]) -> datetime:
    """Sortable form of ``ai_read.decoded_at`` (mixed naive / ``Z`` ISO strings).

    :param decoded_at: Stored timestamp or ``None``.
    :return: Aware UTC datetime; the minimum datetime when missing or unparsable.
    :rtype: datetime
    """
    floor = datetime.min.replace(tzinfo=timezone.utc)
    if not decoded_at:
        return floor
    try:
        parsed = datetime.fromisoformat(decoded_at.replace("Z", "+00:00"))
    except ValueError:
        return floor
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _preference(read: ConsensusRead) -> Tuple[int, datetime, str]:
    """Ordering key among competing reads of one image (larger is better).

    :param read: Candidate read.
    :return: ``(cleaned agreed lines, decoded_at, vlm_model)``.
    :rtype: Tuple[int, datetime, str]
    """
    return (len(read.lines), _decoded_at_key(read.decoded_at), read.vlm_model)


def best_reads(records: Iterable[Dict[str, Any]]) -> List[ConsensusRead]:
    """Visible reads of a fragment, one per image, in image order.

    Duplicates are resolved twice with the same preference (most cleaned
    agreed lines, then newest ``decoded_at``, then checkpoint name for
    determinism): first per ``read_key`` (several checkpoints of one image),
    then per physical image (``image_sha256``, else ``image_url``), which
    catches an Oxford image loaded under both its old and its v8 id with
    different image indices.

    :param records: Raw AI-index ``_source`` dicts, typically every record of
        one fragment.
    :return: Reads sorted by ``(image_index, ai_doc_id)``.
    :rtype: List[ConsensusRead]
    """
    per_key: Dict[str, Tuple[ConsensusRead, str]] = {}
    for record in records:
        read = consensus_read(record)
        if read is None:
            continue
        image_identity = record.get("image_sha256") or record.get("image_url") or read.read_key
        current = per_key.get(read.read_key)
        if current is None or _preference(read) > _preference(current[0]):
            per_key[read.read_key] = (read, image_identity)
    per_image: Dict[str, ConsensusRead] = {}
    for read, image_identity in per_key.values():
        current = per_image.get(image_identity)
        if current is None or _preference(read) > _preference(current):
            per_image[image_identity] = read
    return sorted(per_image.values(), key=lambda r: (r.image_index, r.ai_doc_id))


# ---------------------------------------------------------------------------
# "Uncatalogued" (decision 1)
# ---------------------------------------------------------------------------

MIN_DESCRIPTION_CHARS = 50
PLACEHOLDER_DESCRIPTIONS: Tuple[str, ...] = (
    "Newly treated and encapsulated, must be examined",
    "קטע גניזה",
    "Unidentified",
    "תחום לא מזוהה",
    "A few words",
    "Blank verso",
    "A few words Blank verso",
)
# A record from any of these sources has a scholarly catalogue entry.
CATALOGUED_SOURCES = frozenset({"pgp", "ktiv", "bodleian"})


def _normalise_phrase(text: str) -> str:
    """Case-fold, turn punctuation into spaces and collapse whitespace.

    :param text: Description or placeholder.
    :return: Normalised text.
    :rtype: str
    """
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


_PLACEHOLDER_ONLY_RE = re.compile(
    "^(?:(?:"
    + "|".join(
        re.escape(p) for p in sorted({_normalise_phrase(p) for p in PLACEHOLDER_DESCRIPTIONS}, key=len, reverse=True)
    )
    + r")(?: |$))+$"
)


def is_thin_description(description: Optional[str]) -> bool:
    """Whether a catalogue description says (almost) nothing.

    Thin = empty after trimming, made only of stock placeholders
    (:data:`PLACEHOLDER_DESCRIPTIONS`, matched case- and punctuation-
    insensitively, alone or combined, e.g. "A few words. Blank verso."), or
    shorter than :data:`MIN_DESCRIPTION_CHARS` characters.

    :param description: The catalogue ``description`` field.
    :return: ``True`` when the description is thin.
    :rtype: bool
    """
    text = (description or "").strip()
    if not text or len(text) < MIN_DESCRIPTION_CHARS:
        return True
    return bool(_PLACEHOLDER_ONLY_RE.match(_normalise_phrase(text)))


def _as_list(value: Any) -> List[Any]:
    """Treat an Elasticsearch keyword value (scalar, list or missing) as a list.

    :param value: Field value from ``_source``.
    :return: The values as a list.
    :rtype: List[Any]
    """
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def is_uncatalogued(source: Dict[str, Any]) -> bool:
    """Decision 1: does the catalogue record leave the fragment effectively uncatalogued?

    True when (the description is thin **and** there is no scholarly
    transcription) **or** the record has no PGP, KTIV or Bodleian entry
    (``sources_present`` holds none of them, e.g. an FJP-only record).

    :param source: Catalogue ``_source`` with ``description``,
        ``has_transcriptions`` and ``sources_present``.
    :return: ``True`` when the fragment counts as uncatalogued.
    :rtype: bool
    """
    sources = {str(s).strip().lower() for s in _as_list(source.get("sources_present"))}
    if not sources & CATALOGUED_SOURCES:
        return True
    return is_thin_description(source.get("description")) and not bool(source.get("has_transcriptions"))


# ---------------------------------------------------------------------------
# Terms and incipit (page text)
# ---------------------------------------------------------------------------

# Frequent Hebrew / Aramaic / Judaeo-Arabic function words and letter
# formulas of 3+ letters (shorter words are excluded by length anyway).
# Compared after final-form folding.
_STOPWORDS_RAW = """
אשר הוא היא הם הן זאת אתה אתם אני אנו אנחנו אנכי להם להן לנו לכם לכן עליו עליה עליהם
עליכם עלינו אליו אליה אליהם אליכם אלינו אותו אותה אותם אותנו ממנו ממני ממנה מהם בהם
ולא וכל ועל ואת ואם ואין כאשר אבל אחר אחרי לפני אין איננו היה היתה יהיה יהי הזה הזאת
האלה אלה אלו כמו שלא שהוא כבר מאד מאוד וגם שלו שלה שלהם עמו עמה עמהם בכל לכל מכל והוא
והיא ואני ועד ובין בין תחת מעל אצל ככה איך למה מדוע כדי הנה עוד
הוה דהוה ליה ביה להון בהון הכי היכי מאי האי ההוא נמי אית לית דלא דכל אנא אנן הדין
דהוא דיליה עמיה מניה
עלי אלי דלך כאן כאנת אלא אלדי אלתי אלדין הדא הדה הדי ומא מנה מנהא מנהם פיה פיהא וקד
אנה אנהא אנהם ואן והו והי ומן ענד ענדה ענדי גיר יכון מני לנא ואנא להא למא ואל וכאן לאן
כמא אליך במא חתי ועלי פאן ואמא עליך מעה ענה ופי עליהא כנת אדא ממא באן אנני אלינא אנת
בלא בדלך פלמא פלא ליס נחן איצא תלך ולם ואמ ואנ
אללה תעאלי אטאל אדאם ואדאם
"""
STOPWORDS = frozenset(fold_hebrew(w) for w in _STOPWORDS_RAW.split())

_TERM_RE = re.compile("^[א-ת]+(?:['׳״\"][א-ת]+)*$")
_MARKS_RE = re.compile("[֑-ֽֿ-ׇ̇]")
_EDGE_PUNCT = ".,:;'\"()[]-–—/?!…·׳״־"


def _words(text: str) -> Iterator[str]:
    """Hebrew-script words of a cleaned line, unpointed, edge punctuation removed.

    :param text: Cleaned agreed line.
    :return: Words made only of Hebrew letters (internal geresh/quote allowed).
    :rtype: Iterator[str]
    """
    for token in _MARKS_RE.sub("", text).replace("־", " ").split():
        word = token.strip(_EDGE_PUNCT)
        if word and _TERM_RE.match(word):
            yield word


def consensus_terms(reads: Sequence[ConsensusRead], limit: int = 20) -> List[str]:
    """Distinctive Hebrew-script words of the agreed lines, most frequent first.

    Words need 3+ Hebrew letters and must not be in :data:`STOPWORDS`; words
    that differ only in final forms or points count as one and are shown in
    their most frequent spelling.  Ties keep first-occurrence order.

    :param reads: Reads of one fragment (e.g. from :func:`best_reads`).
    :param limit: Maximum number of terms.
    :return: Up to ``limit`` unique words.
    :rtype: List[str]
    """
    counts: Counter = Counter()
    surfaces: Dict[str, Counter] = {}
    first_seen: Dict[str, int] = {}
    for read in reads:
        for line in read.lines:
            for word in _words(line.text):
                if len(_HEBREW_LETTER_ONLY_RE.sub("", word)) < 3:
                    continue
                key = fold_hebrew(word)
                if key in STOPWORDS:
                    continue
                counts[key] += 1
                surfaces.setdefault(key, Counter())[word] += 1
                first_seen.setdefault(key, len(first_seen))
    ranked = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))
    return [surfaces[key].most_common(1)[0][0] for key in ranked[: max(0, limit)]]


def consensus_incipit(reads: Sequence[ConsensusRead], max_chars: int = 140) -> str:
    """Opening agreed lines joined with ``" / "``, cut on a word boundary.

    :param reads: Reads in display order.
    :param max_chars: Maximum length of the result, ellipsis included.
    :return: The incipit, ending in ``"…"`` when it was cut; ``""`` without lines.
    :rtype: str
    """
    max_chars = max(2, max_chars)
    parts: List[str] = []
    for line in (line for read in reads for line in read.lines):
        parts.append(line.text)
        if len(" / ".join(parts)) > max_chars:
            break
    text = " / ".join(parts)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    if not text[max_chars - 1].isspace():
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
    cut = cut.rstrip(" /.,:;-–—־")
    return (cut or text[: max_chars - 1]) + "…"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# Fields of an AI record needed to rebuild visibility and the public view.
READ_SOURCE_FIELDS: List[str] = [
    "doc_id", "image_index", "image_url", "image_sha256", "read_key", "published", "surfaced",
    "ai_read.vlm_model", "ai_read.decoded_at", "ai_read.parsed",
    "ai_read.lines.index", "ai_read.lines.text", "ai_read.lines.status",
]
CATALOGUE_SOURCE_FIELDS: List[str] = ["description", "has_transcriptions", "sources_present"]
# Ids per mget / terms batch, and records per search_after page.
BATCH_SIZE = 1000
PAGE_SIZE = 1000
_HEBREW_QUERY_RE = re.compile("[א-ת]")


def search_enabled() -> bool:
    """Kill switch of the machine-reading second pass in site search.

    ``AI_TRANSCRIPTIONS_ENABLED`` is the master switch of every AI-read
    consumer; ``CONSENSUS_SEARCH_ENABLED`` switches off only this search pass.

    :return: ``False`` when ``AI_TRANSCRIPTIONS_ENABLED`` or
        ``CONSENSUS_SEARCH_ENABLED`` is a falsy string.
    :rtype: bool
    """
    return feature_enabled() and os.getenv("CONSENSUS_SEARCH_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _batches(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Split a sequence into consecutive chunks.

    :param items: Items to split.
    :param size: Maximum chunk length.
    :return: Chunks in order.
    :rtype: Iterator[Sequence[str]]
    """
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _matched_line(hit: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    """Best-scoring matched agreed line of a search hit that survives cleaning.

    :param hit: One entry of ``hits.hits`` from a :func:`build_search_query` search.
    :return: ``(line index, cleaned text)``, or ``None`` when no matched line is usable.
    :rtype: Optional[Tuple[int, str]]
    """
    for line in parse_line_hits(hit):
        if line.status != "agreed":
            continue
        text = clean_agreed_line(line.text)
        if text is not None:
            return line.index, text
    return None


@dataclass
class _UncataloguedSet:
    """Cached result of :meth:`ConsensusService.uncatalogued_ai_doc_ids`."""

    built_at: float
    ai_doc_ids: List[str]
    record_ids: List[str]


class ConsensusService:
    """Read-only access to consensus reads and the uncatalogued search pass.

    :param es: Elasticsearch client shared with the search service.
    :param catalogue_index: Merged catalogue index (``ELASTICSEARCH_INDEX``).
    :param ai_index: AI side index (``AI_TRANSCRIPTIONS_INDEX``, v2 mapping).
    :param cache_ttl_seconds: Lifetime of the uncatalogued id set.
    """

    def __init__(
        self, es: Elasticsearch, catalogue_index: str, ai_index: str, cache_ttl_seconds: int = 3600
    ) -> None:
        self.es = es
        self.catalogue_index = catalogue_index
        self.ai_index = ai_index
        self.cache_ttl_seconds = cache_ttl_seconds
        # Held by whoever builds the uncatalogued set (first build or refresh).
        self._lock = threading.Lock()
        self._uncatalogued: Optional[_UncataloguedSet] = None
        self._refresh_thread: Optional[threading.Thread] = None

    @staticmethod
    def _visible_filter(ai_doc_ids: Sequence[str]) -> Dict[str, Any]:
        """Query for the published, surfaced records of some fragments.

        :param ai_doc_ids: ``doc_id`` values as stored in the AI index.
        :return: Elasticsearch ``query`` clause.
        :rtype: Dict[str, Any]
        """
        return {
            "bool": {
                "filter": [
                    {"terms": {"doc_id": list(ai_doc_ids)}},
                    {"term": {"published": True}},
                    {"term": {"surfaced": True}},
                ]
            }
        }

    def _scan_records(self, ai_doc_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Every published, surfaced record of the given fragments (light ``_source``).

        :param ai_doc_ids: ``doc_id`` values as stored in the AI index.
        :return: Raw hits (``_id`` and ``_source``); empty when the index is missing.
        :rtype: List[Dict[str, Any]]
        """
        hits_out: List[Dict[str, Any]] = []
        try:
            for batch in _batches(list(dict.fromkeys(ai_doc_ids)), BATCH_SIZE):
                after: Optional[List[Any]] = None
                while True:
                    kwargs: Dict[str, Any] = dict(
                        index=self.ai_index,
                        query=self._visible_filter(batch),
                        source_includes=READ_SOURCE_FIELDS,
                        sort=[{"read_key": "asc"}, {"ai_read.vlm_model": "asc"}],
                        size=PAGE_SIZE,
                    )
                    if after:
                        kwargs["search_after"] = after
                    hits = self.es.search(**kwargs)["hits"]["hits"]
                    hits_out.extend(hits)
                    if len(hits) < PAGE_SIZE:
                        break
                    after = hits[-1]["sort"]
        except NotFoundError:
            logger.info("AI transcription index %s does not exist yet", self.ai_index)
            return []
        return hits_out

    def reads_for_document(self, catalogue_doc_id: str) -> List[ConsensusRead]:
        """Visible consensus reads of one catalogue fragment (both Oxford id forms).

        :param catalogue_doc_id: Catalogue ``doc_id``.
        :return: See :func:`best_reads`.
        :rtype: List[ConsensusRead]
        """
        hits = self._scan_records(to_ai_doc_ids(catalogue_doc_id))
        return best_reads(hit["_source"] for hit in hits)

    def _catalogue_sources(self, catalogue_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fields of :func:`is_uncatalogued` for catalogue records, in ``mget`` batches.

        :param catalogue_ids: Catalogue ``doc_id`` values (``_id`` in the merged index).
        :return: ``{doc_id: _source}`` for the records that exist.
        :rtype: Dict[str, Dict[str, Any]]
        """
        out: Dict[str, Dict[str, Any]] = {}
        for batch in _batches(list(dict.fromkeys(catalogue_ids)), BATCH_SIZE):
            response = self.es.mget(
                index=self.catalogue_index, ids=list(batch), source_includes=CATALOGUE_SOURCE_FIELDS
            )
            for doc in response.get("docs", []):
                if doc.get("found"):
                    out[doc["_id"]] = doc.get("_source") or {}
        return out

    def _build_uncatalogued(self) -> _UncataloguedSet:
        """Scan surfaced AI ids, keep uncatalogued fragments, verify visible reads.

        A fragment whose catalogue record does not exist is left out: neither
        the ``/fragment`` page nor the search result can be built without it.

        :return: Fresh set of AI doc ids and their visible record ``_id`` values.
        :rtype: _UncataloguedSet
        """
        surfaced = AiTranscriptionService(
            self.es, index=self.ai_index, catalogue_index=self.catalogue_index
        ).surfaced_doc_ids(force=True)
        catalogue = self._catalogue_sources([to_catalogue_doc_id(i) for i in surfaced])
        candidates = [
            i for i in surfaced
            if to_catalogue_doc_id(i) in catalogue and is_uncatalogued(catalogue[to_catalogue_doc_id(i)])
        ]
        doc_ids: Set[str] = set()
        record_ids: Set[str] = set()
        for hit in self._scan_records(candidates):
            source = hit["_source"]
            if consensus_read(source) is None:
                continue
            doc_ids.add(source["doc_id"])
            record_ids.add(
                hit.get("_id")
                or record_id(source["doc_id"], int(source["image_index"]), source["ai_read"]["vlm_model"])
            )
        return _UncataloguedSet(time.monotonic(), sorted(doc_ids), sorted(record_ids))

    def _uncatalogued_set(self) -> _UncataloguedSet:
        """The cached uncatalogued set; an expired one is refreshed in the background.

        Only the very first build blocks the caller (and anyone arriving while
        it runs, e.g. during the startup warm-up). Once a set exists, an
        expired one is still returned at once while a single background
        thread rebuilds it (:meth:`_refresh_locked`); a failed refresh keeps
        serving the stale set.

        :return: Current (possibly stale) set.
        :rtype: _UncataloguedSet
        :raises ApiError: When the first build fails on Elasticsearch.
        :raises TransportError: When Elasticsearch cannot be reached for the first build.
        """
        cached = self._uncatalogued
        if cached is not None:
            expired = time.monotonic() - cached.built_at >= self.cache_ttl_seconds
            if expired and self._lock.acquire(blocking=False):
                if self._uncatalogued is not cached:  # another refresh finished meanwhile
                    self._lock.release()
                    return self._uncatalogued
                self._refresh_thread = threading.Thread(
                    target=self._refresh_locked, name="uncatalogued-refresh", daemon=True
                )
                self._refresh_thread.start()
            return cached
        with self._lock:  # first build only
            if self._uncatalogued is None:
                self._uncatalogued = self._build_uncatalogued()
            return self._uncatalogued

    def _refresh_locked(self) -> None:
        """Rebuild the set and swap it in; the caller already holds ``self._lock``.

        Runs in a background thread and releases the lock when done. An
        Elasticsearch failure is logged and the stale set stays in use (the
        next call after expiry tries again).

        :return: None
        :rtype: None
        """
        try:
            self._uncatalogued = self._build_uncatalogued()
        except (ApiError, TransportError) as exc:  # I/O boundary: keep serving the stale set
            logger.warning("Uncatalogued set refresh failed, serving the stale set: %s", exc)
        finally:
            self._lock.release()

    def uncatalogued_ai_doc_ids(self) -> List[str]:
        """AI-index doc ids with a visible read whose catalogue record is uncatalogued.

        :return: Sorted ``doc_id`` values in their AI-index form (old Oxford ids
            unchanged; map with :func:`to_catalogue_doc_id`).
        :rtype: List[str]
        """
        return list(self._uncatalogued_set().ai_doc_ids)

    def search_uncatalogued(self, query: str, size: int, offset: int) -> Tuple[int, List[Dict[str, Any]]]:
        """Agreed-lines search restricted to uncatalogued fragments, one hit per image.

        Uses :func:`~src.backend.ai_transcriptions.build_search_query` with
        ``include_unconfirmed=False`` (nested agreed lines and ``text_agreed``,
        never ``text_all``), filtered to the visible records of
        :meth:`uncatalogued_ai_doc_ids` and collapsed on ``read_key``.  The
        snippet is the best-scoring matched line that survives
        :func:`clean_agreed_line`; a hit with none is skipped.

        :param query: Visitor's query; without a Hebrew letter nothing is searched.
        :param size: Page size.
        :param offset: Hits to skip.
        :return: ``(distinct matching reads, hits)``; each hit has ``doc_id``
            (catalogue id), ``ai_doc_id``, ``image_index``, ``line_index``,
            ``snippet``, ``read_url``, plus ``vlm_model`` and ``score``.
        :rtype: Tuple[int, List[Dict[str, Any]]]
        """
        if not search_enabled() or not _HEBREW_QUERY_RE.search(query or "") or size <= 0:
            return 0, []
        record_ids = self._uncatalogued_set().record_ids
        if not record_ids:
            return 0, []
        es_query = build_search_query(query, include_unconfirmed=False)
        es_query["bool"]["filter"].append({"ids": {"values": list(record_ids)}})
        try:
            response = self.es.search(
                index=self.ai_index,
                query=es_query,
                collapse={"field": "read_key"},
                aggs={"reads": {"cardinality": {"field": "read_key", "precision_threshold": 10000}}},
                sort=[{"_score": "desc"}, {"ai_read.decoded_at": "desc"}],
                source_includes=["doc_id", "image_index", "ai_read.vlm_model"],
                from_=max(0, offset),
                size=size,
                track_total_hits=True,
            )
        except (NotFoundError, BadRequestError) as exc:
            logger.warning("Consensus search unavailable on %s: %s", self.ai_index, exc)
            return 0, []
        hits: List[Dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            matched = _matched_line(hit)
            if matched is None:
                continue
            ai_doc_id = source["doc_id"]
            image_index = int(source["image_index"])
            vlm_model = (source.get("ai_read") or {}).get("vlm_model") or ""
            hits.append(
                {
                    "doc_id": to_catalogue_doc_id(ai_doc_id),
                    "ai_doc_id": ai_doc_id,
                    "image_index": image_index,
                    "line_index": matched[0],
                    "snippet": matched[1],
                    "read_url": read_url(ai_doc_id, image_index, matched[0], vlm_model),
                    "vlm_model": vlm_model,
                    "score": hit.get("_score") or 0.0,
                }
            )
        total = int(response.get("aggregations", {}).get("reads", {}).get("value", len(hits)))
        return total, hits
