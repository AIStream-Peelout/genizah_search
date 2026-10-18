"""Reference prototype (NOT production code) for the deterministic quote rule.

Feasibility check for the survey: position-paired quote extraction, uniform
normalisation, ellipsis segments, title / question-echo exemption, run over the
P2 replay drafts whose full verifier prompts (sources + draft) are on disk.

Usage (read-only; no network, no LM Studio)::

    python3 -I proto_quote_check.py \
        /Users/isaac/Documents/GitHub/genizah_search/evals/results/rag_audit_20261007/p2_raw
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- normalise
_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_DROP_NO_SPACE = set("[]⟨⟩")  # editorial brackets are word-internal in editions: במעשי[ו]


def normalize_for_quote_match(text: str) -> str:
    """Normalise text identically for quote and source before substring matching.

    NFKD -> drop combining marks (Hebrew nikud/cantillation, Latin diacritics)
    -> NFKC -> casefold -> editorial brackets removed -> every other Unicode
    punctuation/symbol char (quotes of all kinds, geresh/gershayim, maqaf,
    dashes, guillemets) -> space -> fold Hebrew finals -> collapse whitespace.

    :param text: Raw text.
    :returns: Normalised text.
    :rtype: str
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    out: List[str] = []
    for ch in folded:
        if ch in _DROP_NO_SPACE:
            continue
        category = unicodedata.category(ch)
        out.append(" " if category[0] in "PS" else ch)
    return re.sub(r"\s+", " ", "".join(out).translate(_FINALS)).strip()


# ---------------------------------------------------------------- extract
_OPEN_BEFORE = set(" \t([{—–-:/*_«„“")
_CLOSE_AFTER = set(" \t.,;:!?)]}—–-*_…'’»")


def extract_quotations(text: str, min_words: int = 3) -> List[Tuple[int, int, str]]:
    """Pair double-quote marks by position and return quotation spans.

    A straight ``"`` or gershayim ``״`` opens only after start/space/bracket/
    dash/colon and closes only before end/space/punctuation; a mark with a
    word character on both sides (ע"ו, כ״י) is an abbreviation and ignored.
    Curly “ ” and guillemets « » pair directly. Spans under ``min_words``
    words are terms, not quotations, but their marks are still consumed so
    they cannot mis-pair with later marks. Pairing never crosses a newline.

    :param text: Draft answer.
    :param min_words: Minimum words for a span to count as a quotation.
    :returns: (start, end, body) for each quotation, in order.
    :rtype: List[Tuple[int, int, str]]
    """
    spans: List[Tuple[int, int, str]] = []
    open_at: Optional[int] = None
    open_char = ""
    closers = {'"': '"', "״": "״", "“": "”", "«": "»"}
    for index, ch in enumerate(text):
        if ch == "\n":
            open_at = None
            continue
        before = text[index - 1] if index > 0 else " "
        after = text[index + 1] if index + 1 < len(text) else " "
        if ch in ('"', "״"):
            if before.isalnum() and after.isalnum():
                if _is_abbreviation_mark(text, index):
                    continue  # ע"ו, כ"י, ור"ד, רמב"ם: mark before the token's last letter
                if not _follows_hebrew_prefix(text, index):
                    continue
                can_open, can_close = True, False  # ש"נשתמרו ...": prefix letter + opening mark
            else:
                can_open = before in _OPEN_BEFORE or before == "\n" or index == 0
                can_close = after in _CLOSE_AFTER or after in "\n" or index + 1 == len(text)
            if open_at is not None and open_char == ch and can_close:
                body = text[open_at + 1:index]
                if len(body.split()) >= min_words:
                    spans.append((open_at, index + 1, body))
                open_at = None
            elif open_at is None and can_open and not can_close:
                open_at, open_char = index, ch
        elif ch in ("“", "«"):
            open_at, open_char = index, ch
        elif ch in ("”", "»") and open_at is not None and closers.get(open_char) == ch:
            body = text[open_at + 1:index]
            if len(body.split()) >= min_words:
                spans.append((open_at, index + 1, body))
            open_at = None
    return spans


_HEBREW_PREFIXES = re.compile(r"(?:^|(?<=[\s(\[—–:]))[ושהכלבמ]{1,3}$")


def _is_abbreviation_mark(text: str, index: int) -> bool:
    """Return whether a word-internal mark is a Hebrew abbreviation (gershayim before the last letter).

    :param text: Draft text.
    :param index: Position of the mark.
    :returns: True for ע"ו-style abbreviations.
    :rtype: bool
    """
    after = text[index + 1:index + 3]
    return len(after) >= 1 and after[0].isalnum() and (len(after) == 1 or not after[1].isalnum())


def _follows_hebrew_prefix(text: str, index: int) -> bool:
    """Return whether the mark follows 1-3 Hebrew prefix letters at a word start (ש"..., וכ"...).

    :param text: Draft text.
    :param index: Position of the mark.
    :returns: True when the mark opens a quotation after a prefix.
    :rtype: bool
    """
    return bool(_HEBREW_PREFIXES.search(text[max(0, index - 4):index]))


_ELLIPSIS = re.compile(r"\.\s?\.\s?\.|…")


def quote_found(body: str, haystack_norm: str) -> bool:
    """Return whether every ellipsis segment of a quote occurs, in order.

    :param body: Quotation body.
    :param haystack_norm: Already-normalised source text.
    :returns: True when verified.
    :rtype: bool
    """
    position = 0
    segments = [normalize_for_quote_match(s).replace(" ", "") for s in _ELLIPSIS.split(body)]
    segments = [s for s in segments if s]
    if not segments:
        return True
    for segment in segments:
        found = haystack_norm.find(segment, position)
        if found == -1:
            return False
        position = found + len(segment)
    return True


# ---------------------------------------------------------------- replay
def parse_verifier_prompt(prompt: str) -> Tuple[List[Dict[str, str]], str]:
    """Split a logged verifier prompt into sources and the draft.

    :param prompt: Full verifier prompt text.
    :returns: (sources with citation/quoteable/evidence, draft).
    :rtype: Tuple[List[Dict[str, str]], str]
    """
    chunks = prompt.split("SOURCE CHUNKS:\n", 1)[1].split("\n\nDRAFT ANSWER:\n", 1)
    sources_text, rest = chunks
    draft = rest.split("\n\nThe type must be", 1)[0]
    sources = []
    for block in re.split(r"\n\n(?=\[SOURCE \d+\])", sources_text.strip()):
        head, _, body = block.partition("\n")
        quoteable = ""
        match = re.search(r"Original page text \(quoteable\): (.*?)(?:\nShelf marks cited in this source:|\Z)", body, re.S)
        if match:
            quoteable = match.group(1)
        sources.append({"citation": re.sub(r"^\[SOURCE \d+\] ", "", head), "quoteable": quoteable, "evidence": body})
    return sources, draft


def classify(draft: str, sources: List[Dict[str, str]], question: str) -> List[Tuple[str, str]]:
    """Classify each quotation of a draft.

    :param draft: Draft text.
    :param sources: Parsed sources.
    :param question: User question.
    :returns: (verdict, body) pairs.
    :rtype: List[Tuple[str, str]]
    """
    # Spaceless comparison: OCR line-break hyphenation ("where- abouts", "Ara- bic"),
    # maqaf and abbreviation marks stop mattering; any changed letter still fails.
    quote_hay = [normalize_for_quote_match(s["quoteable"]).replace(" ", "") for s in sources]
    title_hay = normalize_for_quote_match(" | ".join(s["citation"] for s in sources)
                                          + " | " + " | ".join(s["evidence"] for s in sources if not s["quoteable"])).replace(" ", "")
    question_norm = normalize_for_quote_match(question).replace(" ", "")
    results = []
    for _start, _end, body in extract_quotations(draft):
        norm = normalize_for_quote_match(body).replace(" ", "")
        if any(quote_found(body, hay) for hay in quote_hay):
            results.append(("verified", body))
        elif norm and norm in title_hay:
            results.append(("title", body))
        elif norm and norm in question_norm:
            results.append(("question", body))
        else:
            results.append(("UNVERIFIED", body))
    return results


def main(raw_dir: str) -> None:
    """Replay every P2 draft through the prototype and print verdicts.

    :param raw_dir: ``p2_raw`` directory.
    """
    groups = {
        "p18": ("Who is S.D. Goitein", ["E1a_27b_low", "E1b_27b_low_repeat", "E3a_27b_none",
                                        "E2a_35b_synth_low", "E2b_35b_synth_none"]),
        "p07": ("מה ידוע מן הגניזה על תשובות הגאונים?", ["E1c_27b_low_P07_cold", "E2c_35b_synth_low_P07",
                                                         "E2d_35b_synth_none_P07"]),
    }
    old_re = re.compile(r'"([^"\n]{8,})"|“([^”\n]{8,})”')
    for probe, (question, tags) in groups.items():
        prompt = json.load(open(os.path.join(raw_dir, f"{probe}_verify_request.json")))["messages"][0]["content"]
        sources, prod_draft = parse_verifier_prompt(prompt)
        drafts = [("prod_" + probe, prod_draft)]
        for tag in tags:
            drafts.append((tag, json.load(open(os.path.join(raw_dir, tag + ".json")))["content"]))
        for tag, draft in drafts:
            verdicts = classify(draft, sources, question)
            old = [a or b for a, b in old_re.findall(draft)]
            print(f"== {tag}: old-extractor spans={len(old)} new quotations={len(verdicts)}")
            for verdict, body in verdicts:
                if verdict != "verified":
                    print(f"   {verdict:10s} {body[:110]!r}")


if __name__ == "__main__":
    main(sys.argv[1])
