"""Deterministic quotation and shelf-mark check for drafted chat answers.

Pure module: no I/O and no LLM calls. A quotation is verified when its wording
occurs in a source's quoteable page text after normalisation that ignores
punctuation, case, whitespace and diacritics (Hebrew nikud included). Two
orthographic tolerances follow from the same "small things off" rule: ו/י
after the first letter of a Hebrew word are ignored (plene vs defective
spelling, והגהתי = והגהותי; a quoted segment's own leading ו/י too, as it may
start behind a source prefix letter), and a quotation may leave out a
standalone one- or two-digit footnote marker of the source text (superscript
markers included). Transliteration marks for ayin and hamza (ʿ ʾ) count as
punctuation. It is never fuzzy otherwise: any other changed letter, and any
added or changed digit, fails. Sentences whose quotation or shelf mark cannot
be verified are deleted by character offset.

Imports are limited to the standard library and ``shelfmark_normalizer``:
``lms_agentic_search`` imports this module, so this module must never import
``lms_agentic_search``.

Known coverage limits:

- Single-quoted quotations and quotations of fewer than three words are
  treated as terms and stay unchecked.
- Digitless shelf-mark captures are skipped here and left to the 35B
  verifier (``SHELFMARK_PATTERN`` captures "L-G Arabic 1.99" as "L-G
  Arabic").
- A number truncated exactly at a quotation or ellipsis edge (a quote ending
  "…numbered 700" against "7000") still verifies, because spaces are removed
  before matching; a digit-boundary check was rejected because it falsely
  deleted faithful quotes.
- A quoted segment's leading ו/י may stand for a source prefix letter
  (ושהכלבמ, Aramaic ד, Judeo-Arabic פ), so ימלך still verifies against המלך.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.backend.shelfmark_normalizer import (
    SHELFMARK_PATTERN,
    detect_shelfmarks,
    find_shelfmark_source,
    shelfmarks_equivalent,
)

# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------

_HEBREW_FINAL_LETTERS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
# Editorial brackets are deleted without a space so that a restored letter
# joins its word: במעשי[ו] → במעשיו.
_EDITORIAL_BRACKETS = frozenset("[]⟨⟩")
_WHITESPACE_RUN = re.compile(r"\s+")
# Superscript digits (footnote markers such as India¹⁹) become a standalone
# token; NFKD alone would glue them to the word ("india19").
_SUPERSCRIPT_DIGIT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _is_quote_punctuation(char: str) -> bool:
    """Return whether a character counts as punctuation for quote matching.

    Spacing modifier letters (ʿ ʾ ʼ ʻ, U+02B0-U+02FF) are transliteration marks
    for ayin and hamza that sources and models swap freely with ' and ‘, so
    they count as punctuation too: Saʿadya = Sa'adya = Saadya.

    :param char: One character after NFKC and casefolding.
    :returns: ``True`` for Unicode punctuation, symbols and spacing modifiers.
    :rtype: bool
    """
    return unicodedata.category(char)[0] in "PS" or "\u02b0" <= char <= "\u02ff"


def normalize_for_quote_match(text: str) -> str:
    """Normalise text so that only letters, digits and word breaks remain.

    Steps, in order: superscript digit runs become standalone ASCII tokens;
    NFKD; drop combining marks (Hebrew nikud and cantillation, Latin
    diacritics); NFKC; casefold; delete editorial brackets without a space;
    replace every other punctuation, symbol or spacing-modifier character
    with a space; fold Hebrew final letters; collapse whitespace.

    :param text: Draft or source text.
    :returns: Normalised text with single spaces between words.
    :rtype: str
    """
    detached = _SUPERSCRIPT_DIGIT_RUN.sub(
        lambda match: f" {match.group(0).translate(_SUPERSCRIPT_TO_ASCII)} ", text or ""
    )
    decomposed = unicodedata.normalize("NFKD", detached)
    unmarked = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    folded = unicodedata.normalize("NFKC", unmarked).casefold()
    characters: List[str] = []
    for char in folded:
        if char in _EDITORIAL_BRACKETS:
            continue
        characters.append(" " if _is_quote_punctuation(char) else char)
    joined = "".join(characters).translate(_HEBREW_FINAL_LETTERS)
    return _WHITESPACE_RUN.sub(" ", joined).strip()


_HEBREW_WORD_REGEX = re.compile(r"[\u05d0-\u05ea]+")
# A standalone one- or two-digit token in normalised source text: an inline
# footnote marker in OCR'd book pages ("a qințir (?) 19 arrived", "laymen.4").
_FOOTNOTE_NUMERAL_REGEX = re.compile(r"(?<!\S)\d{1,2}(?!\S)")


def _fold_hebrew_vowel_letters(normalized: str) -> str:
    """Drop ו and י after the first letter of every Hebrew word.

    Hebrew writes the same word with or without these vowel letters (plene
    and defective spelling: והגהותי / והגהתי), and models regenerating a quote
    drift toward the more common spelling. The first letter is kept so a
    word-initial ו ("and") or a consonantal initial י/ו still counts.

    :param normalized: Output of :func:`normalize_for_quote_match`.
    :returns: The same text with the vowel letters folded away.
    :rtype: str
    """
    return _HEBREW_WORD_REGEX.sub(
        lambda match: match.group(0)[0] + match.group(0)[1:].replace("ו", "").replace("י", ""),
        normalized,
    )


def _match_key(normalized: str) -> str:
    """Turn normalised text into its spaceless, vowel-letter-folded match key.

    :param normalized: Output of :func:`normalize_for_quote_match`.
    :returns: Match key compared by substring.
    :rtype: str
    """
    return _fold_hebrew_vowel_letters(normalized).replace(" ", "")


def build_haystack(text: str) -> str:
    """Build a spaceless match haystack (or needle) from text.

    Removing all spaces absorbs OCR line-break hyphenation (``Ara- bic``),
    maqaf and abbreviation marks; Hebrew vowel letters are folded (see
    :func:`_fold_hebrew_vowel_letters`). Any other changed letter or digit
    still fails.

    :param text: Source, citation, question or quotation text.
    :returns: Normalised match key with every space removed.
    :rtype: str
    """
    return _match_key(normalize_for_quote_match(text))


def build_source_haystacks(text: str) -> Tuple[str, ...]:
    """Build the haystacks a quotation is matched against for one source text.

    The first is :func:`build_haystack`. When the source contains standalone
    one- or two-digit tokens (inline footnote markers in OCR'd pages), a
    second haystack without them lets a faithful quote that omits the marker
    verify. Only the source side loses numbers, so a quote that adds or
    changes a number still fails.

    :param text: A source's quoteable page text.
    :returns: One or two spaceless haystacks, plain first.
    :rtype: Tuple[str, ...]
    """
    normalized = normalize_for_quote_match(text)
    plain = _match_key(normalized)
    without_footnotes = _match_key(_FOOTNOTE_NUMERAL_REGEX.sub(" ", normalized))
    return (plain,) if without_footnotes == plain else (plain, without_footnotes)


def build_quote_haystacks(sources: List[Dict[str, Any]]) -> List[Tuple[int, Tuple[str, ...]]]:
    """Build the quote haystacks of every bibliography source.

    Graph sources have no ``quoteable_text`` key and can never verify a quote.

    :param sources: Output of ``build_verification_sources``.
    :returns: ``(source_number, haystacks)`` per source with quoteable text.
    :rtype: List[Tuple[int, Tuple[str, ...]]]
    """
    return [
        (int(source["source_number"]), build_source_haystacks(str(source.get("quoteable_text") or "")))
        for source in sources
        if "quoteable_text" in source
    ]


def find_quote_source_number(body: str, quote_haystacks: List[Tuple[int, Tuple[str, ...]]]) -> Optional[int]:
    """Return the first source whose quoteable text contains a quotation.

    A plain match in any source wins over a footnote-tolerant one, so the
    reported source is the most literal match.

    :param body: Quotation body from the draft.
    :param quote_haystacks: Output of :func:`build_quote_haystacks`.
    :returns: The verifying source number, or ``None``.
    :rtype: Optional[int]
    """
    for variant in range(2):
        for number, haystacks in quote_haystacks:
            if variant < len(haystacks) and quote_found(body, haystacks[variant]):
                return number
    return None


def find_supporting_quote_source(
    quote: str,
    quote_haystacks: List[Tuple[int, Tuple[str, ...]]],
) -> Optional[int]:
    """Verify a supporting quote offered to keep a claim the verifier rejected.

    Same matcher as a drafted quotation (:func:`find_quote_source_number`),
    but the quote must be one contiguous passage, since no model re-reads a
    supporting quote before it is shown as the source's wording: an ellipsis
    inside it fails it (the matcher would skip a one-word segment), while
    leading or trailing ellipses skip nothing and are dropped. Editorial
    brackets are matched with their content kept (``במעשי[ו]``), never as a
    gap. It also needs ``QUOTATION_MIN_WORDS`` words of two or more letters,
    so a lone term or shelf mark ("as T-S 12.243") cannot vouch for a claim.

    :param quote: The supporting quote as the model returned it.
    :param quote_haystacks: Output of :func:`build_quote_haystacks`.
    :returns: The number of the source containing the quote, or ``None`` when
        it is not contiguous, too short or found in no source.
    :rtype: Optional[int]
    """
    quote = _EDGE_ELLIPSIS_REGEX.sub("", quote.strip())
    if _ELLIPSIS_REGEX.search(quote):
        return None
    quote = "".join(char for char in quote if char not in _EDITORIAL_BRACKETS)
    words = [word for word in normalize_for_quote_match(quote).split() if sum(char.isalpha() for char in word) >= 2]
    if len(words) < QUOTATION_MIN_WORDS:
        return None
    return find_quote_source_number(quote, quote_haystacks)


# ----------------------------------------------------------------------------
# Quotation extraction
# ----------------------------------------------------------------------------

QUOTATION_MIN_WORDS = 3
SEGMENT_MIN_WORDS = 2

_STRAIGHT_MARKS = frozenset('"״')
_CURLY_PAIRS = {"“": "”", "«": "»"}
_CURLY_CLOSERS = frozenset(_CURLY_PAIRS.values())
_OPEN_BEFORE = frozenset("([{—–-:/*_«„“")
_CLOSE_AFTER = frozenset(".,;:!?)]}—–-…'’»")
_HEBREW_PREFIX_LETTERS = frozenset("ושהכלבמ")
# Letters a quoted segment may start behind inside a source word: the Hebrew
# prefixes plus the Aramaic relative ד and the Judeo-Arabic conjunction פ.
_SEGMENT_PREFIX_LETTERS = _HEBREW_PREFIX_LETTERS | frozenset("דפ")
_MAX_HEBREW_PREFIX_LETTERS = 3
_ELLIPSIS_REGEX = re.compile(r"\.\s?\.\s?\.|…")
_EDGE_ELLIPSIS_REGEX = re.compile(r"^(?:\.\s?\.\s?\.|…)\s*|\s*(?:\.\s?\.\s?\.|…)$")


def _is_word_start(text: str, index: int) -> bool:
    """Return whether ``index`` starts a word that a quotation could open at.

    :param text: Full text being scanned.
    :param index: Position of the first character of the word.
    :returns: ``True`` at the start of text, after whitespace, or after an
        opening-context character.
    :rtype: bool
    """
    return index == 0 or text[index - 1].isspace() or text[index - 1] in _OPEN_BEFORE


def _opens_hebrew_prefix_quote(text: str, index: int) -> bool:
    """Return whether a word-internal mark opens a Hebrew prefix quotation.

    The mark must follow 1-3 prefix letters (ושהכלבמ) at word start and precede
    a word of at least two letters, as in ש"נשתמרו …". Acronym marks such as
    כ"י, ע"ו, רמב"ם and ור"ד are followed by exactly one letter, so they never
    qualify.

    :param text: Full text being scanned.
    :param index: Position of the straight or gershayim mark.
    :returns: ``True`` when the mark opens a quotation.
    :rtype: bool
    """
    following = text[index + 1:index + 3]
    if len(following) < 2 or not following.isalpha():
        return False
    start = index
    while (
        start > 0
        and index - start < _MAX_HEBREW_PREFIX_LETTERS
        and text[start - 1] in _HEBREW_PREFIX_LETTERS
    ):
        start -= 1
    return start < index and _is_word_start(text, start)


def _neighbour_beyond_emphasis(text: str, index: int, step: int) -> str:
    """Return the first character next to ``index`` that is not ``*`` or ``_``.

    :param text: Full text being scanned.
    :param index: Position of the mark.
    :param step: ``-1`` to look before the mark, ``1`` to look after it.
    :returns: That character, or ``""`` at either end of the text.
    :rtype: str
    """
    position = index + step
    while 0 <= position < len(text) and text[position] in "*_":
        position += step
    return text[position] if 0 <= position < len(text) else ""


def _straight_mark_roles(text: str, index: int) -> Tuple[bool, bool]:
    """Decide whether a straight or gershayim mark can open and/or close.

    Each side is judged by its raw neighbour or, when bold or italic markup
    touches the mark, by the character beyond that markup; either may give
    the context. So ``"**quoted words**"`` pairs, and so do quotations glued
    to emphasis outside them, ``**Goitein**"quoted words"`` and
    ``"quoted words"**Goitein**`` (a mark between a letter and emphasis
    closes).

    :param text: Full text being scanned.
    :param index: Position of the mark.
    :returns: ``(can_open, can_close)``. A mark inside a word is inert unless
        it opens a Hebrew prefix quotation.
    :rtype: Tuple[bool, bool]
    """
    raw_before = text[index - 1] if index > 0 else ""
    raw_after = text[index + 1] if index + 1 < len(text) else ""
    if raw_before.isalnum() and raw_after.isalnum():
        return _opens_hebrew_prefix_quote(text, index), False
    befores = (raw_before, _neighbour_beyond_emphasis(text, index, -1))
    afters = (raw_after, _neighbour_beyond_emphasis(text, index, 1))
    can_open = any(not char or char.isspace() or char in _OPEN_BEFORE for char in befores)
    can_close = any(not char or char.isspace() or char in _CLOSE_AFTER for char in afters) or (
        raw_after in ("*", "_") and raw_before.isalnum()
    )
    if can_open and can_close and _ELLIPSIS_REGEX.match(text, index + 1):
        # An opening context followed by an ellipsis starts a quotation that
        # begins mid-sentence ("…the court met"); otherwise it would never open.
        can_close = False
    return can_open, can_close


def extract_quotations(text: str, min_words: int = QUOTATION_MIN_WORDS) -> List[Tuple[int, int, str]]:
    """Pair quotation marks by position and return the quotations.

    Straight (``"``) and gershayim (``״``) marks open only at a word start and
    close only before a word end; curly quotes and guillemets pair directly.
    Single quotes are never delimiters, and pairing resets at a newline. Spans
    with fewer than ``min_words`` words are terms: they are not returned, but
    their marks are consumed so they cannot mis-pair with later marks.

    :param text: Draft answer text.
    :param min_words: Minimum whitespace-separated words for a quotation.
    :returns: ``(start, end, body)`` triples where ``start`` is the opening
        mark's offset and ``end`` is one past the closing mark.
    :rtype: List[Tuple[int, int, str]]
    """
    pairs: List[Tuple[int, int]] = []
    open_at: Optional[int] = None
    open_char = ""
    for index, char in enumerate(text):
        if char == "\n":
            open_at = None
        elif char in _STRAIGHT_MARKS:
            can_open, can_close = _straight_mark_roles(text, index)
            if open_at is not None and open_char == char and can_close:
                pairs.append((open_at, index))
                open_at = None
            elif can_open and not can_close and (open_at is None or open_char == char):
                # A fresh unambiguous opener supersedes an unclosed one of the
                # same kind, so a stray mark cannot swallow later prose.
                open_at, open_char = index, char
        elif char in _CURLY_PAIRS:
            open_at, open_char = index, char
        elif char in _CURLY_CLOSERS and open_at is not None and _CURLY_PAIRS.get(open_char) == char:
            pairs.append((open_at, index))
            open_at = None
    return [
        (opening, closing + 1, text[opening + 1:closing])
        for opening, closing in pairs
        if len(text[opening + 1:closing].split()) >= min_words
    ]


# ----------------------------------------------------------------------------
# Quotation matching
# ----------------------------------------------------------------------------

_EDITORIAL_INTERPOLATION_REGEX = re.compile(r"\[[^\]\n]*\]")


def _find_segment(needle: str, haystack_nospace: str, position: int) -> int:
    """Find one segment's match key in the haystack at or after ``position``.

    The fold keeps a word's first letter, but a segment may start inside a
    source word behind a prefix letter (ש"ישבו… against שישבו, or a dropped
    conjunction ו), where the source copy of that letter was folded away. So
    when the full needle is missing and starts with ו/י, the rest of it is
    accepted right after a prefix letter of the source (ושהכלבמ, Aramaic ד,
    Judeo-Arabic פ) that lies at or after ``position``. Any other letter there fails (יכתוב against
    תכתוב). Accepted cost: a ו/י that replaces a prefix letter still verifies
    (ילך against הלך).

    :param needle: Spaceless match key of the segment.
    :param haystack_nospace: Spaceless normalised source text.
    :param position: Offset where the previous segment's match ended.
    :returns: The offset one past the match, or ``-1``.
    :rtype: int
    """
    found = haystack_nospace.find(needle, position)
    if found != -1:
        return found + len(needle)
    if needle[:1] not in ("ו", "י"):
        return -1
    rest = needle[1:]
    found = haystack_nospace.find(rest, position + 1)
    while found != -1 and haystack_nospace[found - 1] not in _SEGMENT_PREFIX_LETTERS:
        found = haystack_nospace.find(rest, found + 1)
    return -1 if found == -1 else found + len(rest)


def _segments_found_in_order(body: str, haystack_nospace: str) -> bool:
    """Check that the ellipsis-separated segments of a body occur in order.

    Segments of fewer than two words are ignored (a lone "the" between
    ellipses proves nothing). When no segment of two or more words remains,
    the check fails closed. A segment's leading ו/י may stand for a source
    prefix letter (see :func:`_find_segment`).

    :param body: Quotation body, possibly containing ellipses.
    :param haystack_nospace: Spaceless normalised source text.
    :returns: ``True`` when every substantive segment is found in order.
    :rtype: bool
    """
    position = 0
    matched_any = False
    for segment in _ELLIPSIS_REGEX.split(body):
        normalized = normalize_for_quote_match(segment)
        if len(normalized.split()) < SEGMENT_MIN_WORDS:
            continue
        position = _find_segment(_match_key(normalized), haystack_nospace, position)
        if position == -1:
            return False
        matched_any = True
    return matched_any


def quote_found(body: str, haystack_nospace: str) -> bool:
    """Return whether a quotation body occurs in a spaceless source haystack.

    Attempt 1 matches the ellipsis-separated segments as written. Attempt 2,
    only when the body contains ``[``, treats each draft-side ``[...]`` as an
    omission gap, which accepts an editorial interpolation such as
    "the [Geniza] documents of …" against "the documents of …".

    :param body: Quotation body from the draft.
    :param haystack_nospace: Output of :func:`build_haystack` for a source.
    :returns: ``True`` when either attempt succeeds.
    :rtype: bool
    """
    if not haystack_nospace:
        return False
    if _segments_found_in_order(body, haystack_nospace):
        return True
    if "[" not in body:
        return False
    return _segments_found_in_order(_EDITORIAL_INTERPOLATION_REGEX.sub("…", body), haystack_nospace)


# ----------------------------------------------------------------------------
# Draft check
# ----------------------------------------------------------------------------

QUOTE_CHECK_COUNT_KEYS: Tuple[str, ...] = (
    "quotations",
    "verified",
    "exempt_title",
    "exempt_question",
    "terms_skipped",
    "unverified",
    "shelf_marks_unverified",
)


@dataclass
class QuoteCheckResult:
    """Outcome of :func:`check_quotations` for one draft.

    :param unverified_spans: ``(start, end, text, kind)`` per unverified
        quotation or shelf-mark occurrence, ``kind`` being ``quote`` or
        ``shelf_mark``, ordered by ``start``.
    :param verified: Claim dictionaries shaped like the verify node's
        deterministic claims (``type``, ``text``, ``supported``,
        ``source_number``, ``reasoning``) plus the ``start``/``end`` offsets
        of their first occurrence; de-duplicated per normalised text.
    :param counts: Per-occurrence metrics keyed by
        :data:`QUOTE_CHECK_COUNT_KEYS`. All keys but ``shelf_marks_unverified``
        count quotations only.
    """

    unverified_spans: List[Tuple[int, int, str, str]] = field(default_factory=list)
    verified: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(QUOTE_CHECK_COUNT_KEYS, 0))

    def unverified_offsets(self) -> List[Tuple[int, int]]:
        """Return the unverified spans as offsets for :func:`delete_sentences_at`.

        :returns: ``(start, end)`` pairs in draft order.
        :rtype: List[Tuple[int, int]]
        """
        return [(start, end) for start, end, _text, _kind in self.unverified_spans]


def _verified_claim(claim_type: str, text: str, source_number: int, start: int, end: int) -> Dict[str, Any]:
    """Build a supported deterministic claim dictionary with its span.

    :param claim_type: ``quote`` or ``shelf_mark``.
    :param text: Quotation body or shelf mark as written in the draft.
    :param source_number: Number of the source that contains it.
    :param start: Offset of the first occurrence in the draft.
    :param end: Offset one past the first occurrence.
    :returns: Claim dictionary for the verify node.
    :rtype: Dict[str, Any]
    """
    reasoning = (
        f"Quoted wording found in SOURCE {source_number}"
        if claim_type == "quote"
        else f"Equivalent shelf mark found in SOURCE {source_number}"
    )
    return {
        "type": claim_type,
        "text": text,
        "supported": True,
        "source_number": source_number,
        "reasoning": reasoning,
        "start": start,
        "end": end,
    }


def _classify_shelf_mark(
    mark: str,
    sources: List[Dict[str, Any]],
    question_marks: List[str],
) -> Tuple[str, Optional[int]]:
    """Classify one shelf mark against the sources and the question.

    :param mark: Shelf mark as written in the draft.
    :param sources: Verification source dictionaries.
    :param question_marks: Shelf marks detected in the user's question.
    :returns: ``(verdict, source_number)`` with verdict ``verified``,
        ``exempt_question`` or ``unverified``.
    :rtype: Tuple[str, Optional[int]]
    """
    source_number = find_shelfmark_source(mark, sources)
    if source_number is not None:
        return "verified", source_number
    if any(shelfmarks_equivalent(mark, question_mark) for question_mark in question_marks):
        return "exempt_question", None
    return "unverified", None


# Halper catalogue numbers have at most three digits; "Halper 1924" is an
# author-year citation, not a shelf mark.
_HALPER_YEAR_REGEX = re.compile(r"halper\s+\d{4,}", flags=re.IGNORECASE)


def _is_checkable_shelf_mark(mark: str) -> bool:
    """Return whether a ``SHELFMARK_PATTERN`` match can be a real shelf mark.

    Some pattern alternatives accept any word as the tail, so prose such as
    "the Mosseri collection" or "moss. The" matches. Every real shelf mark
    carries an item number, so a match without a digit is prose, as is an
    author-year "Halper 1924".

    :param mark: Matched text, stripped of trailing punctuation.
    :returns: ``True`` when the match should be checked against the sources.
    :rtype: bool
    """
    return any(char.isdigit() for char in mark) and not _HALPER_YEAR_REGEX.fullmatch(mark)


def check_quotations(draft: str, sources: List[Dict[str, Any]], question: str) -> QuoteCheckResult:
    """Classify every quotation and shelf mark of a draft.

    ``sources`` is the output of ``build_verification_sources``. Quotations are
    tested in order against: the quoteable text of bibliography sources
    (verified); citation lines plus the evidence text of graph sources, which
    have no ``quoteable_text`` key (exempt_title); the user's question
    (exempt_question). Anything else is unverified. Shelf marks are checked at
    every occurrence, with one verdict per lowercased mark; pattern matches
    that cannot be a shelf mark (no item number) are left to the verifier.

    :param draft: Drafted answer text.
    :param sources: Verification source dictionaries.
    :param question: The user's question for this turn only (never assistant
        turns, which would let an unverified quote launder itself).
    :returns: Unverified spans, verified claims and counts.
    :rtype: QuoteCheckResult
    """
    result = QuoteCheckResult()
    counts = result.counts
    quote_haystacks = build_quote_haystacks(sources)
    title_haystacks = [build_haystack(str(source.get("citation") or "")) for source in sources]
    title_haystacks.extend(
        build_haystack(str(source.get("evidence_text") or ""))
        for source in sources
        if "quoteable_text" not in source
    )
    question_haystack = build_haystack(question)
    seen: Set[Tuple[str, str]] = set()

    for start, end, body in extract_quotations(draft, min_words=1):
        body_nospace = build_haystack(body)
        if len(body.split()) < QUOTATION_MIN_WORDS or not body_nospace:
            counts["terms_skipped"] += 1
            continue
        counts["quotations"] += 1
        source_number = find_quote_source_number(body, quote_haystacks)
        if source_number is not None:
            counts["verified"] += 1
            if ("quote", body_nospace) not in seen:
                seen.add(("quote", body_nospace))
                result.verified.append(_verified_claim("quote", body.strip(), source_number, start, end))
        elif any(body_nospace in haystack for haystack in title_haystacks):
            counts["exempt_title"] += 1
        elif body_nospace in question_haystack:
            counts["exempt_question"] += 1
        else:
            counts["unverified"] += 1
            result.unverified_spans.append((start, end, body.strip(), "quote"))

    question_marks = detect_shelfmarks(question)
    verdicts: Dict[str, Tuple[str, Optional[int]]] = {}
    for match in SHELFMARK_PATTERN.finditer(draft):
        raw = match.group(1)
        mark = raw.strip().rstrip(".,")
        if not _is_checkable_shelf_mark(mark):
            continue
        start = match.start(1) + (len(raw) - len(raw.lstrip()))
        end = start + len(mark)
        key = mark.lower()
        if key not in verdicts:
            verdicts[key] = _classify_shelf_mark(mark, sources, question_marks)
        verdict, source_number = verdicts[key]
        if verdict == "verified" and ("shelf_mark", key) not in seen:
            seen.add(("shelf_mark", key))
            result.verified.append(_verified_claim("shelf_mark", mark, source_number, start, end))
        elif verdict == "unverified":
            counts["shelf_marks_unverified"] += 1
            result.unverified_spans.append((start, end, mark, "shelf_mark"))

    result.unverified_spans.sort(key=lambda span: span[0])
    return result


# ----------------------------------------------------------------------------
# Sentence splitting (shared with flag anchoring in lms_agentic_search)
# ----------------------------------------------------------------------------

# Splits after . ! ? (optionally followed by a closing quote, bracket or
# bold/italic marker) when the next sentence starts with a capital (Latin
# extended included), digit, Hebrew letter or ayin/hamza mark, optionally
# behind bold/italic markup or an opening quote or bracket.
_SENTENCE_BOUNDARY_REGEX = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"”’״)\]*_])|(?<=[.!?]\*\*)|(?<=[.!?][\"”’״]\*\*))"
    r"\s+(?=(?:\*\*|[*_])?[\"“״(\[‘]?[A-Z0-9א-תÀ-ÖØ-ÞḀ-ỿʿʾ])"
)

# Words that commonly start a sentence, and titles that precede a name.
# Both decide whether a capital letter with a period is an initial (see
# :func:`_is_initial`).
_SENTENCE_STARTERS = frozenset(
    "The This That These Those There Then Thus Here He She It They We I In On At By For "
    "From With As An And But Or So His Her Its Their Our However Moreover Furthermore "
    "Both Each Such Some Many Most All No Not One When While Although After Before "
    "Since If What Which Who Where Why How See Also Other Another Later Yet Still".split()
)
_NAME_TITLES = frozenset(
    "Professor Prof Rabbi Rav Rabbenu Mar Dr Sir Sheikh Judge Saint St Abu Ibn Ben Bar".split()
)
# A capital letter with a period that is followed by another initial or by a
# capitalised word: a candidate initial ("S. D. Goitein", "As M. Cohen").
_CAPITAL_INITIAL_CANDIDATE_REGEX = re.compile(r"(?<![\w'’])[A-Z]\.(?=\s+(?:[A-Z]\.|[A-Z][a-z]))")
_NEXT_WORD_REGEX = re.compile(r"\s+([A-Z]\.|[A-Za-z]+)")

# Abbreviations whose period does not end a sentence. Without these, "Levin,
# p. 45" splits after "p.", "S. D. Goitein" after "S." and "CUL Or. 1080"
# after "Or.", so a flag or a deletion would cut mid-sentence.
# - Citation, honorific and shelf-mark words; closing emphasis may follow the
#   period (*et al.* / *ibid.*).
# - "no." / "nos." only before a number, so a bare "No." still ends a sentence.
# - Lowercase citation and name letters: p. n. r. v. f. (folio) l. (line),
#   b. (ben), d. (died), c. (circa).
# - Capital initials are decided separately by :func:`_is_initial`.
# - Hebrew single-letter initials only in a chain (ש. ד. גויטיין): Hebrew has
#   no capitals, so "סעיף א. המכתב …" must split, and a letter closing an
#   acronym or numeral (רש"י. / תתק"ב.) is never an initial.
_ABBREVIATION_REGEX = re.compile(
    r"(?i:\b(?:pp|vol|vols|ed|eds|trans|cf|ch|chap|fig|figs|ff|esp|et al|e\.g|i\.e"
    r"|etc|st|mt|mr|mrs|ms|dr|prof|fol|fols|mss|ca|fl|or|ar|misc|bodl|moss|heb|col|ibid))"
    r"\.[*_]*\s"
    r"|(?i:\bnos?)\.\s(?=\d)"
    r"|(?<![\w'’])[pnrvfldbc]\.\s"
    r"|(?<!\w)(?<![א-ת][\"״׳”])[א-ת]\.\s(?=[א-ת]\.\s)"
    r"|(?<=(?<!\w)[א-ת]\.\s)[א-ת]\.\s"
)
_ABBREVIATION_SENTINEL = "\x00"


def _is_initial(text: str, start: int, end: int) -> bool:
    """Decide whether a capital letter with a period is a name initial.

    An initial is followed by another initial, or by a capitalised word that
    is not a common sentence starter and comes after a lowercase word, a
    sentence starter, a title, an abbreviation or the start of the text
    ("S. D. Goitein", "As M. Cohen", "by M. Gil", "Professor M. Cohen"). A
    letter after another capitalised word is a label or numeral that ends
    its sentence ("Appendix A. Goitein …", "Baldwin I. He …").

    :param text: Paragraph being split.
    :param start: Offset of the capital letter.
    :param end: Offset one past its period.
    :returns: ``True`` when the period must not end the sentence.
    :rtype: bool
    """
    following = _NEXT_WORD_REGEX.match(text, end)
    next_word = following.group(1) if following else ""
    if next_word.endswith("."):
        return True
    if next_word in _SENTENCE_STARTERS:
        return False
    preceding = text[:start].split()
    previous = preceding[-1].strip("*_\"“‘(") if preceding else ""
    return (
        not previous
        or previous.endswith((".", _ABBREVIATION_SENTINEL))
        or not previous[0].isupper()
        or previous in _SENTENCE_STARTERS
        or previous.rstrip(",") in _NAME_TITLES
    )


def sentence_spans(paragraph: str) -> List[Tuple[int, int]]:
    """Return the offsets of each sentence of a paragraph.

    Abbreviation and initial periods are masked (length-preserving) before
    the boundary rule runs, so citations such as "Levin, p. 45" and names
    such as "S. D. Goitein" stay intact.

    :param paragraph: Single paragraph or line of answer text.
    :returns: ``(start, end)`` offsets into ``paragraph``, in order.
    :rtype: List[Tuple[int, int]]
    """
    stripped = paragraph.strip()
    if not stripped:
        return []
    offset = len(paragraph) - len(paragraph.lstrip())
    masked = _ABBREVIATION_REGEX.sub(
        lambda match: match.group(0).replace(".", _ABBREVIATION_SENTINEL),
        stripped,
    )
    masked = _CAPITAL_INITIAL_CANDIDATE_REGEX.sub(
        lambda match: (
            match.group(0)[0] + _ABBREVIATION_SENTINEL
            if _is_initial(masked, match.start(), match.end())
            else match.group(0)
        ),
        masked,
    )
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for boundary in _SENTENCE_BOUNDARY_REGEX.finditer(masked):
        spans.append((offset + cursor, offset + boundary.start()))
        cursor = boundary.end()
    spans.append((offset + cursor, offset + len(masked)))
    return [(start, end) for start, end in spans if paragraph[start:end].strip()]


def split_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences with a conservative boundary rule.

    The returned sentences are verbatim substrings of the input, which callers
    rely on for locating and wrapping spans.

    :param paragraph: Single paragraph of answer text (no blank lines).
    :returns: Sentences in order; the original text is recoverable modulo
        inter-sentence whitespace.
    :rtype: List[str]
    """
    return [paragraph[start:end] for start, end in sentence_spans(paragraph)]


# ----------------------------------------------------------------------------
# Sentence deletion
# ----------------------------------------------------------------------------

# Leading indent plus an optional list, heading or blockquote marker; kept
# aside so a line that loses a sentence keeps its markup.
_LINE_PREFIX_REGEX = re.compile(r"^\s*(?:(?:[-*+]|\d+\.|#{1,6}|>)\s+)?")
_HEADING_REGEX = re.compile(r"^\s{0,3}(#{1,6})\s")


# Lines that carry no answer on their own: headings, horizontal rules, table
# separator rows, bare list or blockquote markers, emphasis-only labels
# ("**The court**", "1. **Elections.**"), lead-ins ending in a colon and
# bare work citations ("**Goitein**, *Letters*, p. 7.").
_HORIZONTAL_RULE_REGEX = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_SEPARATOR_REGEX = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_EMPHASIS_LABEL_REGEX = re.compile(r"^(\*\*|__|\*|_)[^*_]+\1[.:]?$")
_BARE_CITATION_REGEX = re.compile(
    r"^\*\*[^*]+\*\*,\s*\*[^*]+\*(?:,\s*(?:vol\.\s*[\w]+,\s*)?(?:pp?\.)\s*[\d–-]+)?\.?$"
)


def _is_label_line(line: str, table_has_body: bool) -> bool:
    """Return whether a draft line carries no answer on its own.

    :param line: One line of the draft (no newline).
    :param table_has_body: Whether a table separator row came before this
        line, so a ``|`` row is a data row rather than a header.
    :returns: ``True`` for structure, labels and lead-ins.
    :rtype: bool
    """
    if _HEADING_REGEX.match(line) or _HORIZONTAL_RULE_REGEX.match(line) or _TABLE_SEPARATOR_REGEX.match(line):
        return True
    if line.lstrip().startswith("|"):
        return not table_has_body
    text = _LINE_PREFIX_REGEX.sub("", line, count=1).strip()
    return (
        not text
        or text in ("-", "*", "+", ">")
        or text.endswith(":")
        or bool(_EMPHASIS_LABEL_REGEX.match(text))
        or bool(_BARE_CITATION_REGEX.match(text))
    )


def has_answer_prose(draft: str) -> bool:
    """Return whether a draft still holds some answer after deletions.

    A short answer is fine (Isaac, 2026-10-11), but a remainder made only of
    headings, labels, lead-ins ending in a colon, bare citations or empty
    table headers is not an answer and must not ship as one.

    :param draft: Draft text after sentence deletion.
    :returns: ``True`` when at least one line carries answer text.
    :rtype: bool
    """
    table_has_body = False
    for line in draft.splitlines():
        if not line.strip():
            table_has_body = False
            continue
        if _TABLE_SEPARATOR_REGEX.match(line):
            table_has_body = True
        if not _is_label_line(line, table_has_body):
            return True
    return False


def _line_sentences(line: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Split one line into its markup prefix and sentence offsets.

    :param line: One line of the draft (no newline).
    :returns: The prefix and sentence ``(start, end)`` offsets into ``line``.
    :rtype: Tuple[str, List[Tuple[int, int]]]
    """
    prefix = _LINE_PREFIX_REGEX.match(line).group(0)
    rest_offset = len(prefix)
    return prefix, [
        (rest_offset + start, rest_offset + end)
        for start, end in sentence_spans(line[rest_offset:])
    ]


def _overlaps(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    """Return whether ``[start, end)`` overlaps any of the spans.

    :param start: Sentence start offset.
    :param end: Sentence end offset.
    :param spans: ``(start, end)`` offsets to test against.
    :returns: ``True`` when some span intersects the sentence.
    :rtype: bool
    """
    return any(span_start < end and span_end > start for span_start, span_end in spans)


def _paragraph_sentence_spans(paragraph: str) -> List[Tuple[int, int]]:
    """Return sentence offsets for every line of a paragraph.

    :param paragraph: Paragraph text (may hold several lines, e.g. bullets).
    :returns: ``(start, end)`` offsets into ``paragraph``; list markers are
        left outside the sentences.
    :rtype: List[Tuple[int, int]]
    """
    spans: List[Tuple[int, int]] = []
    line_start = 0
    for line in paragraph.split("\n"):
        _prefix, sentences = _line_sentences(line)
        spans.extend((line_start + start, line_start + end) for start, end in sentences)
        line_start += len(line) + 1
    return spans


def sentences_overlapping(draft: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return the offsets of the sentences that contain any of the spans.

    This is the set of sentences :func:`delete_sentences_at` removes, so its
    length is the number of sentences deleted.

    :param draft: Drafted answer text.
    :param spans: ``(start, end)`` offsets of unverified items in ``draft``.
    :returns: Sentence ``(start, end)`` offsets into ``draft``, in order.
    :rtype: List[Tuple[int, int]]
    """
    return [(start, end) for start, end in _paragraph_sentence_spans(draft) if _overlaps(start, end, spans)]


def _drop_emptied_headings(lines: List[str], kept: List[Optional[str]]) -> None:
    """Remove headings whose section lost all of its body to deletion.

    A heading is dropped when its section (up to the next surviving heading of
    the same or a higher level, or the end) had a line removed and no
    non-blank line remains; a surviving subsection heading counts as body.
    Headings that had no body to begin with are left alone. Works bottom-up
    so that a parent heading whose only subsection vanished is dropped too.

    :param lines: Original draft lines.
    :param kept: Per-line surviving text, ``None`` for a removed line; updated
        in place.
    :returns: Nothing.
    :rtype: None
    """
    for index in range(len(lines) - 1, -1, -1):
        heading = kept[index]
        heading_match = _HEADING_REGEX.match(heading) if heading is not None else None
        if heading_match is None:
            continue
        level = len(heading_match.group(1))
        lost_line = False
        has_body = False
        for following in range(index + 1, len(lines)):
            text = kept[following]
            if text is None:
                lost_line = True
                continue
            following_heading = _HEADING_REGEX.match(text)
            if following_heading is not None and len(following_heading.group(1)) <= level:
                break
            if following_heading is not None or text.strip():
                has_body = True
                break
        if lost_line and not has_body:
            kept[index] = None


def _rebuild_line(line: str, line_start: int, spans: List[Tuple[int, int]]) -> Tuple[Optional[str], List[int]]:
    """Remove the sentences of one line that overlap any span.

    :param line: One line of the draft (no newline).
    :param line_start: Offset of the line in the draft.
    :param spans: ``(start, end)`` draft offsets of unverified items.
    :returns: The surviving line (``line`` itself when nothing was removed,
        ``None`` when nothing survives) and the offsets into it where removed
        sentences used to be: the start of the survivor that followed a
        removal, or the end of the line when the removal came last.
    :rtype: Tuple[Optional[str], List[int]]
    """
    prefix, sentences = _line_sentences(line)
    rebuilt = prefix
    gaps: List[int] = []
    survivors = 0
    pending_gap = False
    for start, end in sentences:
        if _overlaps(line_start + start, line_start + end, spans):
            pending_gap = True
            continue
        if survivors:
            rebuilt += " "
        if pending_gap:
            gaps.append(len(rebuilt))
            pending_gap = False
        rebuilt += line[start:end]
        survivors += 1
    if survivors == len(sentences):
        return line, []
    if not survivors:
        return None, []
    if pending_gap:
        gaps.append(len(rebuilt))
    return rebuilt, gaps


def delete_sentences_with_gaps(
    draft: str,
    spans: List[Tuple[int, int]],
) -> Tuple[str, Dict[int, List[int]]]:
    """Delete every sentence that overlaps a span and report where gaps remain.

    Same deletion as :func:`delete_sentences_at`, which wraps this function.
    The gap offsets tell a later smoothing step where each removal happened, so
    it can mark the spot (see :func:`insert_gap_markers`) without the marker
    ever entering the draft.

    :param draft: Drafted answer text.
    :param spans: ``(start, end)`` offsets of unverified items in ``draft``.
    :returns: The new draft and, for each paragraph (index into
        ``new_draft.split("\\n\\n")``) that lost a sentence but still has text,
        the sorted offsets into that paragraph's text where removed sentences
        or lines used to be. A removal inside a line is marked at the start of
        the surviving sentence after it, or at the end of the line when it came
        last; a removed whole line at the end of the line before it, or, at the
        top of the paragraph, after the next line's list marker. Keys are in
        paragraph order.
    :rtype: Tuple[str, Dict[int, List[int]]]
    """
    if not sentences_overlapping(draft, spans):
        return draft, {}
    lines = draft.split("\n")
    kept: List[Optional[str]] = []
    line_gaps: List[List[int]] = []
    line_start = 0
    for line in lines:
        rebuilt, gaps = _rebuild_line(line, line_start, spans)
        kept.append(rebuilt)
        line_gaps.append(gaps)
        line_start += len(line) + 1
    _drop_emptied_headings(lines, kept)

    paragraphs: List[str] = []
    gaps_by_paragraph: Dict[int, List[int]] = {}
    current: List[str] = []
    current_gaps: Set[int] = set()
    current_changed = False
    pending_line_gap = False
    for original, text, gaps in zip(lines + [""], kept + [""], line_gaps + [[]]):
        if not original.strip():
            if current:
                if pending_line_gap:
                    current_gaps.add(len("\n".join(current)))
                if current_changed:
                    gaps_by_paragraph[len(paragraphs)] = sorted(current_gaps)
                paragraphs.append("\n".join(current))
            current, current_gaps, current_changed, pending_line_gap = [], set(), False, False
            continue
        current_changed = current_changed or text != original
        if text is None:
            pending_line_gap = True
            continue
        line_offset = len("\n".join(current)) + 1 if current else 0
        if pending_line_gap:
            # A removed line is marked at the end of the line before it, or,
            # at the top of a paragraph, after this line's list marker.
            current_gaps.add(line_offset - 1 if current else len(_LINE_PREFIX_REGEX.match(text).group(0)))
            pending_line_gap = False
        current_gaps.update(line_offset + gap for gap in gaps)
        current.append(text)
    return "\n\n".join(paragraphs), gaps_by_paragraph


def delete_sentences_at(draft: str, spans: List[Tuple[int, int]]) -> Tuple[str, List[int]]:
    """Delete every sentence that overlaps an unverified span.

    Deletion is by character offset, never by fuzzy lookup. Per line, a list,
    heading or blockquote marker (or indent) is kept aside and the surviving
    sentences are rejoined with single spaces behind it. Lines left empty are
    dropped, as are headings left with no body; blank-line runs collapse to
    one paragraph break.

    :param draft: Drafted answer text.
    :param spans: ``(start, end)`` offsets of unverified items in ``draft``.
    :returns: The new draft and the indices (in ``new_draft.split("\\n\\n")``)
        of paragraphs that lost a sentence but still have text.
    :rtype: Tuple[str, List[int]]
    """
    new_draft, gaps_by_paragraph = delete_sentences_with_gaps(draft, spans)
    return new_draft, list(gaps_by_paragraph)


GAP_MARKER = "⟦gap⟧"


def insert_gap_markers(paragraph: str, offsets: List[int], marker: str = GAP_MARKER) -> str:
    """Insert a marker at each gap offset reported by :func:`delete_sentences_with_gaps`.

    A space separates the marker from adjacent non-space text, so a gap
    between two sentences reads ``First. ⟦gap⟧ Next.``.

    :param paragraph: One paragraph of the post-deletion draft.
    :param offsets: Gap offsets into ``paragraph``.
    :param marker: Marker text to insert.
    :returns: The paragraph with one marker per distinct offset.
    :rtype: str
    """
    marked = paragraph
    for offset in sorted(set(offsets), reverse=True):
        before = " " if offset > 0 and not marked[offset - 1].isspace() else ""
        after = " " if offset < len(marked) and not marked[offset].isspace() else ""
        marked = marked[:offset] + before + marker + after + marked[offset:]
    return marked


def remove_gap_markers(paragraph_with_gap: str, marker: str = GAP_MARKER) -> Tuple[str, List[int]]:
    """Undo :func:`insert_gap_markers`: drop each marker and report where it was.

    Each marker goes together with one adjacent space (the one after it when
    it starts a line or follows whitespace, otherwise the one before it), so
    ``First. ⟦gap⟧ Next.`` becomes ``First. Next.`` with the gap at ``Next.``.

    :param paragraph_with_gap: Paragraph text containing gap markers.
    :param marker: Marker text to remove.
    :returns: The paragraph without markers and the gap offsets into it.
    :rtype: Tuple[str, List[int]]
    """
    pieces: List[str] = []
    offsets: List[int] = []
    length = 0
    cursor = 0
    index = paragraph_with_gap.find(marker)
    while index != -1:
        start, end = index, index + len(marker)
        if (start == 0 or paragraph_with_gap[start - 1].isspace()) and paragraph_with_gap.startswith(" ", end):
            end += 1
        elif start > cursor and paragraph_with_gap[start - 1] == " ":
            start -= 1
        pieces.append(paragraph_with_gap[cursor:start])
        length += start - cursor
        offsets.append(length)
        cursor = end
        index = paragraph_with_gap.find(marker, cursor)
    pieces.append(paragraph_with_gap[cursor:])
    return "".join(pieces), offsets


# ----------------------------------------------------------------------------
# Smoothing guard (the one guarded rewrite after a deletion)
# ----------------------------------------------------------------------------

SMOOTHING_MAX_NEW_WORDS_PER_GAP = 8
# A new Hebrew word costs two units of the per-gap budget, so a gap allows at
# most four (prefixes glue more meaning into one Hebrew word).
SMOOTHING_HEBREW_WORD_COST = 2
SMOOTHING_MIN_LENGTH_RATIO = 0.6
SMOOTHING_MAX_ADDED_CHARS = 160
# Markup the rewrite may never add, with the name used in its reject reason.
_SMOOTHING_MARKUP: Tuple[Tuple[str, str], ...] = (
    ("**", "bold"),
    ("[", "bracket"),
    ("(", "parenthesis"),
    ("doc:", "doc_link"),
    ("`", "backtick"),
)
# Ordinary words that may start a bridging sentence capitalised without
# counting as a new proper noun (they still count as new words).
_COMMON_CAPITALISED_WORDS = frozenset(
    "a an the this that these those such his her its their he she it they there here "
    "however moreover furthermore additionally also meanwhile nevertheless nonetheless "
    "still yet thus therefore hence consequently accordingly instead indeed similarly "
    "likewise later then next afterwards subsequently finally overall in on at by for "
    "with from as after before during while when although though because since until "
    "but and or so not no both each other another many some most all one only even "
    "what which who whose where why how if notably ultimately eventually importantly "
    "interestingly earlier initially today elsewhere specifically further besides rather "
    "throughout beyond despite crucially significantly again following over unlike among "
    "through within together".split()
)
_HEBREW_LEADING_PREFIXES = "וה"
_LATIN_WORD_REGEX = re.compile(r"[^\W\d_]+")
_DIGIT_RUN_REGEX = re.compile(r"\d+")


def _is_hebrew(token: str) -> bool:
    """Return whether a token contains a Hebrew letter.

    :param token: One word.
    :returns: ``True`` when any character is in the Hebrew block.
    :rtype: bool
    """
    return any("\u0590" <= char <= "\u05ff" for char in token)


def _word_key(token: str) -> str:
    """Return the vocabulary key of a normalised word.

    Hebrew words lose one or two leading ו/ה (keeping at least two letters),
    so ובנוסף and בנוסף share a key.

    :param token: One word from :func:`normalize_for_quote_match` output.
    :returns: The key used for "is this word new" lookups.
    :rtype: str
    """
    stripped = 0
    while (
        stripped < 2
        and _is_hebrew(token)
        and len(token) > 2
        and token[0] in _HEBREW_LEADING_PREFIXES
    ):
        token = token[1:]
        stripped += 1
    return token


def _vocabulary(text: str) -> Set[str]:
    """Return the vocabulary keys of every word in a text.

    :param text: Paragraph text.
    :returns: Casefolded, diacritic-free word keys.
    :rtype: Set[str]
    """
    return {_word_key(token) for token in normalize_for_quote_match(text).split()}


def _new_word_cost(text: str, vocabulary: Set[str]) -> int:
    """Count the words of a text that the paragraph did not already contain.

    :param text: Rewritten region of the paragraph.
    :param vocabulary: Keys of the input paragraph's words.
    :returns: The number of new word occurrences, a Hebrew word counting
        :data:`SMOOTHING_HEBREW_WORD_COST`; numbers are ignored (they are
        rejected separately).
    :rtype: int
    """
    cost = 0
    for token in normalize_for_quote_match(text).split():
        if token.isdigit() or _word_key(token) in vocabulary:
            continue
        cost += SMOOTHING_HEBREW_WORD_COST if _is_hebrew(token) else 1
    return cost


def _has_new_proper_noun(text: str, vocabulary: Set[str]) -> bool:
    """Return whether a text has a capitalised Latin word new to the paragraph.

    :param text: Rewritten region of the paragraph.
    :param vocabulary: Keys of the input paragraph's words.
    :returns: ``True`` for a capitalised Latin-script word whose key is
        neither in the paragraph nor a common sentence-starting word.
    :rtype: bool
    """
    for match in _LATIN_WORD_REGEX.finditer(text):
        word = match.group(0)
        if not (word[0].isupper() and unicodedata.name(word[0], "").startswith("LATIN")):
            continue
        key = build_haystack(word)
        if key not in vocabulary and key not in _COMMON_CAPITALISED_WORDS:
            return True
    return False


@dataclass
class _SmoothingRegion:
    """A run of the paragraph the smoothing call may rewrite.

    :param gaps: Number of gaps inside the run.
    :param sentences: Number of input sentences in the run (each adjacent to
        a gap).
    """

    gaps: int
    sentences: int


def _smoothing_layout(paragraph: str, gaps: List[int]) -> List[Union[str, _SmoothingRegion]]:
    """Split a paragraph into fixed text and rewritable regions.

    Only the sentence before and the sentence after each gap may change;
    every other sentence is fixed. A fixed run keeps the text between its
    sentences, and a run at either end of the paragraph extends to that end.

    :param paragraph: Stripped paragraph text without markers.
    :param gaps: Gap offsets into ``paragraph``.
    :returns: Alternating parts in paragraph order: a ``str`` for each fixed
        run and a :class:`_SmoothingRegion` for each rewritable region.
    :rtype: List[Union[str, _SmoothingRegion]]
    """
    spans = _paragraph_sentence_spans(paragraph)
    adjacent: Set[int] = set()
    for gap in gaps:
        before = [index for index, (_start, end) in enumerate(spans) if end <= gap]
        after = [index for index, (start, _end) in enumerate(spans) if start >= gap]
        inside = [index for index, (start, end) in enumerate(spans) if start < gap < end]
        adjacent.update(before[-1:] + after[:1] + inside)

    groups: List[Tuple[bool, int, int]] = []
    for index in range(len(spans)):
        fixed = index not in adjacent
        if groups and groups[-1][0] == fixed:
            groups[-1] = (fixed, groups[-1][1], index)
        else:
            groups.append((fixed, index, index))

    parts: List[Union[str, _SmoothingRegion]] = []
    for fixed, first, last in groups:
        start = 0 if first == 0 else spans[first][0]
        end = len(paragraph) if last == len(spans) - 1 else spans[last][1]
        if fixed:
            parts.append(paragraph[start:end])
        else:
            parts.append(_SmoothingRegion(
                gaps=sum(1 for gap in gaps if start <= gap <= end),
                sentences=last - first + 1,
            ))
    return parts


def _whitespace_bounded(text: str, start: int, end: int) -> bool:
    """Return whether ``text[start:end]`` is bounded by whitespace or the ends.

    :param text: Text containing the span.
    :param start: Span start offset.
    :param end: Span end offset.
    :returns: ``True`` when the span neither starts nor ends mid-word.
    :rtype: bool
    """
    return (start == 0 or text[start - 1].isspace()) and (end == len(text) or text[end].isspace())


def _find_bounded(text: str, needle: str, position: int) -> int:
    """Find the first whitespace-bounded occurrence of ``needle`` from ``position``.

    :param text: Text to search.
    :param needle: Text to find.
    :param position: Offset to start searching from.
    :returns: The offset of the occurrence, or ``-1``.
    :rtype: int
    """
    found = text.find(needle, position)
    while found != -1 and not _whitespace_bounded(text, found, found + len(needle)):
        found = text.find(needle, found + 1)
    return found


def _match_layout(
    output: str,
    parts: List[Union[str, _SmoothingRegion]],
) -> Optional[List[Tuple[_SmoothingRegion, str]]]:
    """Locate a layout's fixed runs in the output, in order.

    Each fixed run must appear byte-identical and whitespace-bounded; a run
    that starts the paragraph must start the output and one that ends the
    paragraph must end it.

    :param output: Stripped rewritten paragraph.
    :param parts: Layout from :func:`_smoothing_layout`.
    :returns: Each rewritable region paired with the output text that
        replaced it, or ``None`` when a fixed run is missing or altered.
    :rtype: Optional[List[Tuple[_SmoothingRegion, str]]]
    """
    matched: List[Tuple[_SmoothingRegion, str]] = []
    position = 0
    pending: Optional[_SmoothingRegion] = None
    for number, part in enumerate(parts):
        if isinstance(part, _SmoothingRegion):
            pending = part
            continue
        if number == len(parts) - 1:
            found = len(output) - len(part)
            if found < position or not output.endswith(part) or (number == 0 and found != 0):
                return None
        elif number == 0:
            found = 0 if output.startswith(part) else -1
        else:
            found = _find_bounded(output, part, position)
        if found == -1 or not _whitespace_bounded(output, found, found + len(part)):
            return None
        if pending is not None:
            matched.append((pending, output[position:found]))
            pending = None
        position = found + len(part)
    if pending is not None:
        matched.append((pending, output[position:]))
    return matched


def smoothing_violation(paragraph_with_gap: str, output: str) -> Optional[str]:
    """Check one smoothed paragraph against the deterministic guard.

    The guard keeps the smoothing call from adding facts. Only the sentence
    before and the sentence after each ``⟦gap⟧`` may change, plus at most one
    inserted sentence per gap; every other sentence must be byte-identical.
    The rewrite may add no digits, no capitalised Latin-script word absent
    from the paragraph (a new proper noun), no quotation, ``**``, ``[``,
    ``(``, ``doc:`` or backtick, and no shelf mark, and must keep every input
    quotation verbatim. At most 8 new words per gap (a Hebrew word counts
    double, so at most 4), and ``0.6 * len(in) <= len(out) <= len(in) + 160``
    with ``len(in)`` measured without markers. No marker may remain.

    :param paragraph_with_gap: The paragraph as sent to the model, with a gap
        marker wherever a sentence was removed.
    :param output: The model's rewritten paragraph.
    :returns: ``None`` when the rewrite is acceptable, otherwise the reason:
        ``marker_left``, ``too_short``, ``too_long``, ``removed_quotation``,
        ``new_quotation``, ``new_bold``, ``new_bracket``,
        ``new_parenthesis``, ``new_doc_link``, ``new_backtick``,
        ``new_shelf_mark``, ``new_number``, ``altered_sentence``,
        ``too_many_sentences``, ``new_proper_noun`` or ``too_many_new_words``.
    :rtype: Optional[str]
    """
    unmarked, gaps = remove_gap_markers(paragraph_with_gap)
    leading = len(unmarked) - len(unmarked.lstrip())
    paragraph = unmarked.strip()
    gaps = [min(max(gap - leading, 0), len(paragraph)) for gap in gaps]
    output = output.strip()

    if "⟦" in output:
        return "marker_left"
    if len(output) < SMOOTHING_MIN_LENGTH_RATIO * len(paragraph):
        return "too_short"
    if len(output) > len(paragraph) + SMOOTHING_MAX_ADDED_CHARS:
        return "too_long"

    input_quotes = Counter(paragraph[start:end] for start, end, _body in extract_quotations(paragraph, min_words=1))
    output_quotes = Counter(output[start:end] for start, end, _body in extract_quotations(output, min_words=1))
    if any(quote not in output for quote in input_quotes):
        return "removed_quotation"
    if output_quotes - input_quotes:
        return "new_quotation"
    for token, name in _SMOOTHING_MARKUP:
        if output.count(token) > paragraph.count(token):
            return f"new_{name}"
    input_marks = detect_shelfmarks(paragraph)
    if any(
        not any(shelfmarks_equivalent(mark, known) for known in input_marks)
        for mark in detect_shelfmarks(output)
    ):
        return "new_shelf_mark"
    if Counter(_DIGIT_RUN_REGEX.findall(output)) - Counter(_DIGIT_RUN_REGEX.findall(paragraph)):
        return "new_number"

    regions = _match_layout(output, _smoothing_layout(paragraph, gaps))
    if regions is None:
        return "altered_sentence"
    vocabulary = _vocabulary(paragraph)
    for region, text in regions:
        if len(_paragraph_sentence_spans(text)) > region.sentences + region.gaps:
            return "too_many_sentences"
    for _region, text in regions:
        if _has_new_proper_noun(text, vocabulary):
            return "new_proper_noun"
    for region, text in regions:
        if _new_word_cost(text, vocabulary) > SMOOTHING_MAX_NEW_WORDS_PER_GAP * max(region.gaps, 1):
            return "too_many_new_words"
    return None
