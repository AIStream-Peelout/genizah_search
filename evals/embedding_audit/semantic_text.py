"""Semantic view of the corpus: production embedding text minus record-identity lines.

The production text (``create_text_representation``) starts with ``Document ID:`` and ``Shelf Mark:`` lines.
Those are numbering schemes plus the collection a fragment was shipped to, so a model trained on them learns
literal id/shelf-mark matching instead of subjects. This module strips them (and the ``[Editor: ...]`` /
PGP ``Editor: Surname, Given`` attributions), masks the shelf marks and PGPIDs that remain inside other lines
(the record's own mark restated in a description, joins citing other fragments) as ``[shelfmark]``, flags records
with nothing semantic left, and writes ``v3/corpus_semantic.jsonl``::

    {doc_id, text, eligible, text_hash, series, topics, frames, framed, sukkot_gold}

A record is *ineligible* when the stripped text is empty, shorter than :data:`MIN_CHARS`, or carries nothing
beyond a boilerplate catalogue description (one shared by >= ``min_count`` records, e.g. "Piyyut",
"Newly treated and encapsulated, must be examined"), metadata lines (:data:`META_PREFIXES`) and an
``Alternate titles:`` line whose every ``;`` part is itself boilerplate ("Unidentified", "פיוט",
"אין בידנו לעת עתה פרטים על תוכנו של הקטע"). Ineligible records are never training positives and are left out
of the vector eval pool. The ``eligible`` flag in the output file is the source of truth for later steps.

Run (json only, no model)::

    python3 semantic_text.py
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import takewhile
from pathlib import Path
from typing import Iterable, Set, Tuple

from embed_utils import AUDIT_ROOT

ID_PREFIXES = ("Document ID:", "Shelf Mark:")
# lines that describe the record's form, not its subject: they never make a record eligible on their own
META_PREFIXES = ("Language:", "Document Type:", "Date:")
ALT_PREFIX = "Alternate titles:"
# "[Editor: KTIV Academic transcription (Frag. 001r)]"; also the truncated form at a cut-off line end ("[Editor: KTIV Acad...")
EDITOR_RE = re.compile(r"\[Editors?:[^\]\n]{0,200}(?:\]|$)\s*")
# PGP transcriptions open with unbracketed attributions: "Editor: Gil, Moshe Editor: Goitein, S. D. ",
# "Editors: Goitein, S. D.; Friedman, Mordechai Akiva ", "Editors: [unknown source]". A name is a capitalised surname,
# optionally ", Given names/initials"; the lookahead stops a name from swallowing the next "Editor:".
_EDITOR_NAME = r"(?:\[unknown source\]‎?|[A-Z][\w'’\-]*(?:\.\.\.)?(?:,(?: (?!Editors?:)[A-Z][\w'’\-]*\.?)+)?)"
PGP_EDITOR_RE = re.compile(rf"(?<!\w)Editors?: {_EDITOR_NAME}(?:; {_EDITOR_NAME})*\s*")
EMPTY_FIELD_RE = re.compile(r"^[A-Z][\w /()-]{0,40}:\s*$")  # a field label left with no value, e.g. "Transcription:"
MIN_CHARS = 40

# Shelf-mark detector copied from origin/prod-mbp:src/backend/shelfmark_normalizer.py (SHELFMARK_PATTERN; read-only
# copy: the audit never imports production code). Moved here from build_training_mix_v3 so the semantic
# text itself is masked (training, the frozen eval and the text production would embed all agree).
PROD_SHELFMARK_RE = re.compile(
    r'\b('
    r'T-S\s+AS\s+\d+[\w.]*|T-S\s+NS\s+\d+[\w.]*|T-S\s+K\s+\d+[\w.]*|T-S\s+Ar\.?\s*\d+[\w.]*|T-S\s+Misc\.?\s*\d+[\w.]*'
    r'|TS\s+AS\s+\d+[\w.]*|TS\s+NS\s+\d+[\w.]*|TS\s+Misc\.?\s*\d+[\w.]*'
    r'|T-S\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*|T-S\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    r'|TS\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*|TS\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    r'|ENA\s+NS\s+\d+[\w.]*|ENA\s+\d+[\w.]*'
    r'|L-G\s+[A-Za-z0-9]+[\w.]*'
    r'|Mosseri\s+[A-Za-z0-9]+[\w.]*|Moss\.\s+[A-Za-z0-9]+[\w.]*'
    r'|CUL\s+Or\.?\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?[A-Za-z]?\d+(?:\.\d+)?)?|CUL\s+Add\s+\d+[\w.]*'
    r'|Or\.\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?[A-Za-z]?\d+(?:\.\d+)?)?'
    r'|Bodl\.?\s+MS\.?\s+[A-Za-z]+\.?\s*[A-Za-z]?\.?\s*\d+[\w.]*|MS\.?\s+heb\.?\s*[A-Za-z]\.?\s*\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+'
    r'|Manchester\s*[:\-]?\s*Gaster\s+Printed\s+Series\s+\d+[\w.]*|Manchester\s*[:\-]?\s*Gaster\s+\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Glass\s+\d+[\w.]*|Manchester\s*[:\-]?\s*[ABCGLP]\s+\d+[\w.]*'
    r'|Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+|Gaster\s+[A-Z]\s+\d+[\w.]*'
    r'|BL\s+Or\s+\d+[\w.]*|BL\s+Add\s+\d+[\w.]*|Halper\s+\d+|AIU\s+[IVX]+\.[A-Z]\.\d+'
    r')', re.IGNORECASE)
# record-identity strings the production pattern misses, as cited inside PGP / KTIV descriptions: PGP ids, the
# St Petersburg collections, "T-S NS J401.21", "T-S Loan 137", "CUL Add.3339a", "Bodl. Heb. f. 26/1-6",
# "ENA NS I.2" / "ENA L43", "AIU XII.118", "JRL Gaster heb. ms 1636/14", "DK 238.5 alt. XVI" (Kaufmann),
# "Stras. 4038/27", "UPenn E 16510", "Freer F 1908.44H", "Rylands B 8725", "Pococke 400", "Firk. I: 1587",
# "Westminster Frag. Cairens. 25", "NLI#$FL188116337", "NLI 577.4/27", "F 1908.44i", "Sassoon 713", "JRL B 5758",
# "Mosseri VII 155"
EXTRA_ID_RE = re.compile(
    r"\bPGPID\s*:?\s*#?\s*\d+"
    r"|\bAntonin\s+\d+[\w.]*"
    r"|\bYevr\.?(?:[-\s]Arab\.?)?\s+[IVX]+(?:\s*[A-Z]\b)?(?:\s*\d+[\w./]*)?"
    r"|\bT-?S\s+NS\s+J\.?\s?\d+[\w.]*"
    r"|\bT-?S\s+(?:Loan|Arabic|Misc\.?|[A-Z]{1,2}\.?)\s?\d+[\w.]*"
    r"|\bCUL\s+(?:Add|Or)\.\s*\d+[\w.]*(?:[–-][a-z]\b)?"
    r"|\bBodl\.?\s+(?:MS\.?\s+)?(?:Heb\.?\s*)?[a-z]\.?\s*\d+[\w./-]*"
    r"|\bJTS:?\s+(?:MS\s+|Geniza\s+Misc\.?\s+|ENA\s+)?\d+[\w.]*"
    r"|\bENA\s+(?:NS\s+|Misc\.?\s+)?(?:[IVX]+\.\d+|[A-Z]\d+|\d+)[\w.]*"
    r"|\bAIU\s+(?:[IVX]+\.?\s*)?[A-Z]?\.?\s*\d+[\w.]*"
    r"|\b(?:JRL\s+)?Gaster\s+(?:hebr?|ar)\.?\s*(?:ms\.?\s*)?\d+[\w./]*"
    r"|\bDK\s*\d+[\w./]*(?:\s+alt\.?\s*[IVXLC]+)?"
    r"|\bStras(?:bourg)?\.?\s*\d+[\w./]*"
    r"|\bU?Penn\s+E\s*\d+[\w.]*"
    r"|\bFreer:?\s+F\s*\d{4}\.\d+[\w.]*"
    r"|\bRylands\s+[A-Z]\s*\d+[\w./]*"
    r"|\bPococke\s+\d+[\w.]*"
    r"|\bFirk\.?\s*[IV]+\s*:?\s*\d+[\w.]*"
    r"|\bWestminster\s+Frag\.?\s+Cairens\.?\s*\d+[\w.]*"
    r"|\bNLI\s*#?\$?FL\d+"
    r"|\bNLI\s+\d+(?:\.\d+)*(?:/\d+)?"
    r"|\bF\s+1908\.\d+[\w.]*"
    r"|\bSassoon\s+\d+[\w.]*"
    r"|\bJRL\s+[A-Z]\s*\d+[\w./]*",
    re.IGNORECASE)
# applied BEFORE the production pattern, which would stop "Mosseri VII 155" after "Mosseri VII"
PRE_ID_RE = re.compile(r"\bMosseri\s+[IVX]+[a-z]?\s*[.,]?\s*\d+[\w.]*")
MASK = "[shelfmark]"
OUT_DIR = AUDIT_ROOT / "v3"


def mask_shelfmarks(text: str, own_mark: str = "") -> Tuple[str, int]:
    """Replace the record's own shelf mark, cited shelf marks and PGPIDs with :data:`MASK`.

    :param text: Text (a stripped record text or one of its lines).
    :param own_mark: The record's ``Shelf Mark:`` value (masked literally when longer than 4 characters).
    :returns: (masked text, number of spans masked).
    :rtype: Tuple[str, int]
    """
    n = 0
    if len(own_mark) > 4 and own_mark in text:
        # whole-token match only: "T-S 12.34" must not turn "T-S 12.345" into "[shelfmark]5"
        text, n = re.subn(rf"(?<!\w){re.escape(own_mark)}(?!\w)", MASK, text)
    text, k0 = PRE_ID_RE.subn(MASK, text)
    text, k1 = PROD_SHELFMARK_RE.subn(MASK, text)
    text, k2 = EXTRA_ID_RE.subn(MASK, text)
    return text, n + k0 + k1 + k2


def own_shelfmark(text: str) -> str:
    """The ``Shelf Mark:`` value of a production text (empty when absent).

    :param text: Production embedding text.
    :returns: Shelf mark.
    :rtype: str
    """
    return next((ln[len("Shelf Mark:"):].strip() for ln in text.split("\n") if ln.startswith("Shelf Mark:")), "")


def semantic_text(text: str, mask: bool = True) -> str:
    """Drop ``Document ID:`` / ``Shelf Mark:`` lines and editor attributions; mask other record identifiers.

    Removes ``[Editor: ...]`` tags and the PGP form ``Editor: Surname, Given`` (both are who edited the transcription,
    not what the record is about), drops field lines left without a value, and (``mask``) replaces the record's own
    shelf mark, cited shelf marks of other fragments and PGPIDs anywhere in the remaining lines with :data:`MASK`
    (descriptions and join notes cite them: numbering schemes + collection, never the subject).

    :param text: Production embedding text of one record.
    :param mask: Mask identifiers (:func:`mask_shelfmarks`).
    :returns: Stripped text (may be empty).
    :rtype: str
    """
    lines = [PGP_EDITOR_RE.sub("", EDITOR_RE.sub("", ln)).rstrip() for ln in text.split("\n")
             if not ln.startswith(ID_PREFIXES)]
    out = "\n".join(ln for ln in lines if ln.strip() and not EMPTY_FIELD_RE.match(ln)).strip()
    if mask:
        out = mask_shelfmarks(out, own_shelfmark(text))[0]
    return out


def description_value(text: str) -> str:
    """Return the value of the first ``Description:`` line (empty string if none).

    :param text: Record text (raw or stripped).
    :returns: Description value.
    :rtype: str
    """
    for ln in text.split("\n"):
        if ln.startswith("Description:"):
            return ln[len("Description:"):].strip()
    return ""


def iter_corpus(corpus_path: Path) -> Iterable[dict]:
    """Yield corpus rows.

    :param corpus_path: ``corpus_v1.jsonl`` path.
    :yields: Row dicts.
    """
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def alt_title_parts(text: str) -> Set[str]:
    """The ``;``-separated parts of a record's ``Alternate titles:`` line(s).

    :param text: Record text (raw or stripped).
    :returns: Set of non-empty parts.
    :rtype: Set[str]
    """
    parts: Set[str] = set()
    for ln in text.split("\n"):
        if ln.startswith(ALT_PREFIX):
            parts.update(p.strip() for p in ln[len(ALT_PREFIX):].split(";") if p.strip())
    return parts


def boilerplate_descriptions(corpus_path: Path, min_count: int = 50, include_alt_titles: bool = True) -> Set[str]:
    """Description values shared by at least ``min_count`` records (compared after :func:`mask_shelfmarks`).

    :param corpus_path: ``corpus_v1.jsonl`` path.
    :param min_count: Minimum number of records sharing the exact description.
    :param include_alt_titles: Also return ``Alternate titles`` parts shared by ``min_count`` records (the
        catalogue's alternative labels; e.g. "Unidentified" or "Piyyut" restated as an alternate title).
    :returns: Set of boilerplate label strings.
    :rtype: Set[str]
    """
    counts, alt_counts = Counter(), Counter()
    for row in iter_corpus(corpus_path):
        own = own_shelfmark(row["text"])  # counted on masked values: "Join with [shelfmark]" is one label
        counts[mask_shelfmarks(description_value(row["text"]), own)[0]] += 1
        if include_alt_titles:
            alt_counts.update({mask_shelfmarks(p, own)[0] for p in alt_title_parts(row["text"])})
    counts.pop("", None)
    return {d for d, c in counts.items() if c >= min_count} | {d for d, c in alt_counts.items() if c >= min_count}


def content_lines(stripped: str, boiler: Set[str]) -> list:
    """Lines of a stripped text that say something beyond boilerplate description and metadata.

    :param stripped: Output of :func:`semantic_text`.
    :param boiler: Boilerplate descriptions.
    :returns: Remaining content lines.
    :rtype: list
    """
    out = []
    for ln in stripped.split("\n"):
        if not ln.strip() or ln.startswith(META_PREFIXES):
            continue
        if ln.startswith("Description:") and ln[len("Description:"):].strip() in boiler:
            continue
        if ln.startswith(ALT_PREFIX) and alt_title_parts(ln) <= boiler:
            continue
        out.append(ln)
    return out


def is_eligible(stripped: str, boiler: Set[str]) -> bool:
    """Whether a record can serve as a training positive / vector-eval candidate.

    :param stripped: Output of :func:`semantic_text`.
    :param boiler: Boilerplate descriptions (:func:`boilerplate_descriptions`).
    :returns: False when empty, shorter than :data:`MIN_CHARS`, or boilerplate + metadata only.
    :rtype: bool
    """
    if len(stripped) < MIN_CHARS:
        return False
    return bool(content_lines(stripped, boiler))


def ineligible_reason(stripped: str, boiler: Set[str]) -> str:
    """Name the reason a record is ineligible (for the summary counts).

    :param stripped: Output of :func:`semantic_text`.
    :param boiler: Boilerplate descriptions.
    :returns: ``empty``, ``short``, ``boilerplate_only`` or ``eligible``.
    :rtype: str
    """
    if not stripped:
        return "empty"
    if len(stripped) < MIN_CHARS:
        return "short"
    return "eligible" if content_lines(stripped, boiler) else "boilerplate_only"


def series_of(doc_id: str) -> str:
    """Collection series of a record: ``doc_id`` tokens before the first token containing a digit.

    :param doc_id: Canonical id, e.g. ``Cambridge_CUL_T_S_24_33``.
    :returns: Series, e.g. ``Cambridge_CUL_T_S`` (empty when the id starts with a number).
    :rtype: str
    """
    return "_".join(takewhile(lambda tok: not any(ch.isdigit() for ch in tok), doc_id.split("_")))


def main() -> None:
    """Write ``v3/corpus_semantic.jsonl`` and print eligibility / duplicate counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--out", default=str(OUT_DIR / "corpus_semantic.jsonl"))
    parser.add_argument("--min-count", type=int, default=50, help="records sharing a description to call it boilerplate")
    parser.add_argument("--no-alt-titles", action="store_true", help="only Description values count as boilerplate")
    args = parser.parse_args()
    corpus = Path(args.corpus)
    boiler = boilerplate_descriptions(corpus, args.min_count, include_alt_titles=not args.no_alt_titles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    hashes: Counter = Counter()
    eligible_hashes: Counter = Counter()
    boiler_only_desc: Counter = Counter()  # which boilerplate labels make records ineligible (for review)
    with open(args.out, "w", encoding="utf-8") as out:
        for row in iter_corpus(corpus):
            unmasked = semantic_text(row["text"], mask=False)
            stripped = semantic_text(row["text"])
            reason = ineligible_reason(stripped, boiler)
            eligible = reason == "eligible"
            text_hash = hashlib.md5(stripped.encode("utf-8")).hexdigest()
            hashes[text_hash] += 1
            eligible_hashes[text_hash] += eligible
            shelf = own_shelfmark(row["text"])
            stats["records"] += 1
            stats[reason] += 1
            stats["editor_attribution_removed"] += bool(EDITOR_RE.search(row["text"]) or PGP_EDITOR_RE.search(row["text"]))
            if reason == "boilerplate_only":
                boiler_only_desc[description_value(stripped) or "<no description>"] += 1
            stats["description_boilerplate"] += description_value(stripped) in boiler
            stats["own_shelfmark_before_masking"] += bool(len(shelf) > 4 and shelf in unmasked)
            stats["own_shelfmark_still_in_text"] += bool(len(shelf) > 4 and shelf in stripped)
            n_masked = mask_shelfmarks(unmasked, shelf)[1]
            stats["shelfmark_masked_records"] += bool(n_masked)
            stats["shelfmark_masked_records_eligible"] += bool(n_masked) and eligible
            stats["shelfmark_masked_spans"] += n_masked
            out.write(json.dumps({
                "doc_id": row["doc_id"], "text": stripped, "eligible": eligible, "text_hash": text_hash,
                "series": series_of(row["doc_id"]), "topics": row.get("topics") or [], "frames": row.get("frames") or [],
                "framed": bool(row.get("framed")), "sukkot_gold": bool(row.get("sukkot_gold")),
            }, ensure_ascii=False) + "\n")
    dup_groups = {h: n for h, n in hashes.items() if n > 1}
    eligible_dups = {h: n for h, n in eligible_hashes.items() if n > 1}
    summary = {
        **dict(stats),
        "ineligible": stats["records"] - stats["eligible"],
        "boilerplate_descriptions": len(boiler),
        "duplicate_text_groups": len(dup_groups),
        "records_in_duplicate_groups": sum(dup_groups.values()),
        "eligible_duplicate_text_groups": len(eligible_dups),
        "eligible_records_in_duplicate_groups": sum(eligible_dups.values()),
        "largest_duplicate_groups": sorted(dup_groups.values(), reverse=True)[:8],
        "boilerplate_only_top_descriptions": dict(boiler_only_desc.most_common(25)),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("boilerplate descriptions:", json.dumps(sorted(boiler), ensure_ascii=False))
    stats_path = Path(args.out).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({**summary, "boilerplate_list": sorted(boiler), "min_count": args.min_count,
                                      "min_chars": MIN_CHARS}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
