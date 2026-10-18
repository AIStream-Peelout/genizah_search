"""Semantic view of the corpus: production embedding text minus record-identity lines (``semantic_v3``), and the
``semantic_v4`` layout over it.

The production text (``create_text_representation``) starts with ``Document ID:`` and ``Shelf Mark:`` lines.
Those are numbering schemes plus the collection a fragment was shipped to, so a model trained on them learns
literal id/shelf-mark matching instead of subjects. ``semantic_v3`` strips them (and the ``[Editor: ...]`` /
PGP ``Editor: Surname, Given`` attributions), masks the shelf marks and PGPIDs that remain inside other lines
(the record's own mark restated in a description, joins citing other fragments) as ``[shelfmark]`` and flags records
with nothing semantic left.

``semantic_v4`` (the v10 embedder round, ``docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md`` B1-B3, B7) is a pure
layout over the ``semantic_v3`` text, so the masking and editor stripping are exactly v3's:

* continuation lines (a description value that contained a newline) are folded into their field line;
* the ``Subjects:`` line (catalogue keywords = training-label names) and the ``Document Type:`` line (an English
  keyword guess, present on 73% of FJP-only and 0% of KTIV-only records) are dropped;
* the ``Alternate titles:`` parts are folded into the ``Description:`` line, a part whose normalised key already
  occurs as whole words being left out, the appended parts capped at :data:`MAX_ALT_CHARS` (whole parts). No
  ``Alternate titles:`` label is left, so the layout no longer tells KTIV records from the rest.

KTIV scholarly frames (and specific work titles) are a separate HDA document attribute that
``create_text_representation(include_scholarly_frames=...)`` lists at the end of the ``Alternate titles:`` line or
leaves out. That render switch is the frames switch: ``build_corpus.py`` writes both renderings (``text`` with frames,
``text_no_frames`` where it differs) and ``--scholarly-frames off`` embeds the second, as text version
``semantic_v4_nf`` (:func:`v4_text_version`). The v4 layout itself has no frames logic.

Eligibility v4 is decided on the ``semantic_v3`` text without its ``Subjects:`` lines, with the frozen 64-label
boilerplate list (:data:`BOILERPLATE_SHA256`) and the v3 rule (alternate-title parts compared without square
brackets), plus one admission: a ``short`` text whose content names a festival or a topic (frozen patterns in
``eligibility_admit_v4.json``, :data:`ADMIT_SHA256`, written by ``build_admit_v4.py``) is eligible. On a text v3
judged, it never removes a record (``--v3-corpus`` asserts it).

Everything between the ``semantic_v4 begin`` / ``semantic_v4 end`` markers is copied byte for byte into
historical-document-analysis ``src/embeddings/semantic_text.py`` (``parity_v4.py`` checks the block and every
record's output; ``test_semantic_v4.py`` checks the golden fixture ``tests_data/semantic_v4_golden.jsonl``).

Writes ``corpus_semantic.jsonl``::

    {doc_id, text, text_v3, text_version, eligible, reason, text_hash, text_sha256, series, topics, frames, framed,
     sukkot_gold}

``text`` is the embedded text (``semantic_v4`` / ``semantic_v4_nf`` with ``--text-version semantic_v4``, else
``semantic_v3``); ``text_v3`` is always the ``semantic_v3`` text of the same rendering (readers that parse fields use
it after :func:`fold_field_lines`); ``text_hash`` is the md5 of ``text`` (the leak checks recompute it),
``text_sha256`` its sha256 (the index field ``embedded_text_sha256``). Ineligible records are never training
positives and are left out of the vector eval pool. The ``eligible`` flag in the output file is the source of truth
for later steps.

Run (json only, no model)::

    python3 semantic_text.py                                 # semantic_v3, labels recomputed (reproduces v9 inputs)
    python3 semantic_text.py --corpus $V4/corpus_v1.jsonl --text-version semantic_v4 \\
        --out $V4/corpus_semantic.jsonl --v3-corpus $AR/v3/corpus_semantic.jsonl
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from itertools import takewhile
from pathlib import Path
from typing import AbstractSet, Dict, FrozenSet, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

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

TEXT_VERSION_SEMANTIC_V3 = "semantic_v3"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Frozen inputs of the v4 text function. The HDA copy points the same names at src/embeddings/data/ (same bytes).
BOILERPLATE_PATH = REPO_ROOT / "docs" / "handoffs" / "v9_embedder" / "boilerplate_descriptions_v3.json"
ADMIT_PATH = REPO_ROOT / "docs" / "handoffs" / "v10_embedder" / "eligibility_admit_v4.json"
V4_BLOCK_BEGIN = "# semantic_v4 begin"
V4_BLOCK_END = "# semantic_v4 end"
# corpus field (build_corpus.py) holding the production text rendered without KTIV scholarly frames, where it differs
TEXT_NO_FRAMES_FIELD = "text_no_frames"
# Residual record-identity tokens in v4 texts (plan H5): long digit runs (NLI system numbers) and manuscript sigla.
RESIDUAL_ID_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)|\b(?:Ms|MS|Cod|Codex)\.?\s+[A-Za-z]*\.?\s*\d+")


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

    Only the legacy ``semantic_v3`` run recomputes labels; ``semantic_v4`` reads the frozen list
    (:func:`load_boilerplate_labels`).

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


def content_lines(stripped: str, boiler: AbstractSet[str]) -> list:
    """Lines of a stripped text that say something beyond boilerplate description and metadata (v3 rule).

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


def is_eligible(stripped: str, boiler: AbstractSet[str]) -> bool:
    """Whether a record can serve as a training positive / vector-eval candidate (v3 rule).

    :param stripped: Output of :func:`semantic_text`.
    :param boiler: Boilerplate descriptions (:func:`boilerplate_descriptions`).
    :returns: False when empty, shorter than :data:`MIN_CHARS`, or boilerplate + metadata only.
    :rtype: bool
    """
    if len(stripped) < MIN_CHARS:
        return False
    return bool(content_lines(stripped, boiler))


def ineligible_reason(stripped: str, boiler: AbstractSet[str]) -> str:
    """Name the reason a record is ineligible (v3 rule, for the summary counts).

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


@lru_cache(maxsize=None)
def load_boilerplate_labels(path: Path = BOILERPLATE_PATH) -> FrozenSet[str]:
    """Load the frozen boilerplate labels once (same function as the HDA copy's).

    :param path: JSON file with a ``boilerplate`` list and the ``min_chars`` it was frozen with.
    :returns: The labels.
    :rtype: FrozenSet[str]
    :raises ValueError: When the file's ``min_chars`` disagrees with :data:`MIN_CHARS`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("min_chars") != MIN_CHARS:
        raise ValueError(f"{path}: min_chars={data.get('min_chars')} but this text version uses {MIN_CHARS}")
    return frozenset(data["boilerplate"])


# =====================================================================================================================
# semantic_v4 begin. Everything from this line to the "semantic_v4 end" line is copied byte for byte into
# historical-document-analysis src/embeddings/semantic_text.py; parity_v4.py compares the two blocks and every record's
# output. The block reads only names both copies define above it (ALT_PREFIX, META_PREFIXES, MIN_CHARS,
# alt_title_parts, load_boilerplate_labels, semantic_text) plus the repo-specific BOILERPLATE_PATH / ADMIT_PATH.
# Changing it changes the embedded text: that is a new text version and a full re-embed, never an edit of one copy.
# =====================================================================================================================
TEXT_VERSION_SEMANTIC_V4 = "semantic_v4"
# the same layout over the production text rendered without the KTIV scholarly frames (the H0 "no frames" arm)
TEXT_VERSION_SEMANTIC_V4_NF = "semantic_v4_nf"
DESC_PREFIX = "Description:"
SUBJECTS_PREFIX = "Subjects:"
# Every field label GenizahDocument.create_text_representation() writes; a line starting with none of them continues
# the value of the line before it (a description that contained a newline).
FIELD_PREFIXES = ("Document ID:", "Shelf Mark:", DESC_PREFIX, ALT_PREFIX, SUBJECTS_PREFIX, "Title:", "Summary:",
                  "Language:", "Document Type:", "Date:", "Transcription:", "Translation:", "Related People:",
                  "Related Places:", "Physical:", "Additional Info:")
# never embedded by v4: catalogue keywords (the training labels' names) and the English keyword guess at the type
V4_DROP_PREFIXES = (SUBJECTS_PREFIX, "Document Type:")
TITLE_SEP = "; "
MAX_ALT_CHARS = 1000
TITLE_KEY_RE = re.compile(r"[\W_]+")
BOILERPLATE_SHA256 = "d57204f7278cc38bc713d54c05533fe87a40c79e3110cbbd51e42cb10e372677"
ADMIT_SHA256 = "2922b7747033c4d1d9b9845fee01b7c8fc0228b532d45c0e359db5786637ec01"


def text_sha256(text: str) -> str:
    """sha256 hex digest of the exact UTF-8 bytes handed to the embedder (index field ``embedded_text_sha256``).

    :param text: Embedded text.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file's bytes.

    :param path: File path.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str) -> Path:
    """Refuse a frozen data file whose bytes are not the pinned ones.

    :param path: Data file.
    :param expected_sha256: Pinned sha256 hex digest.
    :returns: ``path`` as a Path.
    :rtype: Path
    :raises ValueError: When the file's sha256 differs from the pin.
    """
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: sha256 {actual} is not the pinned {expected_sha256} (frozen text-function input)")
    return Path(path)


def v4_text_version(scholarly_frames: bool) -> str:
    """Text version name of a v4 variant.

    :param scholarly_frames: Whether the production text was rendered with the KTIV scholarly frames.
    :returns: :data:`TEXT_VERSION_SEMANTIC_V4` or :data:`TEXT_VERSION_SEMANTIC_V4_NF`.
    :rtype: str
    """
    return TEXT_VERSION_SEMANTIC_V4 if scholarly_frames else TEXT_VERSION_SEMANTIC_V4_NF


def fold_field_lines(text: str) -> str:
    """The text with every line that starts no field label appended (one space) to the line before it.

    Blank continuation lines are dropped; a first line that starts no label is kept as it is.

    :param text: A production or semantic text.
    :returns: The text with one line per field.
    :rtype: str
    """
    out: List[str] = []
    for line in text.split("\n"):
        if out and not line.startswith(FIELD_PREFIXES):
            if line.strip():
                out[-1] = f"{out[-1].rstrip()} {line.strip()}"
        else:
            out.append(line)
    return "\n".join(out)


def title_key(text: str) -> str:
    """Comparison key of a title: case-folded, every run of punctuation / brackets / underscores one space.

    :param text: A title, an alternate-title part or a description.
    :returns: Key (words separated by single spaces).
    :rtype: str
    """
    return " ".join(TITLE_KEY_RE.sub(" ", text.casefold()).split())


def title_parts(lines: Sequence[str]) -> List[str]:
    """The non-empty ``;`` parts of every ``Alternate titles:`` line, in text order.

    :param lines: Lines of a folded text (:func:`fold_field_lines`).
    :returns: Parts, stripped, duplicates kept.
    :rtype: List[str]
    """
    parts: List[str] = []
    for line in lines:
        if line.startswith(ALT_PREFIX):
            parts.extend(p.strip() for p in line[len(ALT_PREFIX):].split(";") if p.strip())
    return parts


def merge_title_parts(description: str, parts: Sequence[str]) -> Tuple[List[str], int, int]:
    """Choose the alternate-title parts appended to a description.

    A part is left out when its :func:`title_key` occurs, as whole words, in the description's key or in an earlier
    kept part's key (or is empty); appending stops before the first part that would take the appended characters
    past :data:`MAX_ALT_CHARS` (the first kept part is always kept whole).

    :param description: The ``Description:`` value (may be empty).
    :param parts: Candidate parts in order (:func:`title_parts`).
    :returns: (kept parts in order, parts left out as duplicates, parts left out by the cap).
    :rtype: Tuple[List[str], int, int]
    """
    keys = [f" {title_key(description)} "] if description else []
    kept: List[str] = []
    added = duplicates = 0
    for i, part in enumerate(parts):
        key = title_key(part)
        if not key or any(f" {key} " in seen for seen in keys):
            duplicates += 1
            continue
        if kept and added + len(part) > MAX_ALT_CHARS:
            return kept, duplicates, len(parts) - i
        kept.append(part)
        added += len(part) + len(TITLE_SEP)
        keys.append(f" {key} ")
    return kept, duplicates, 0


def merge_title_lines(lines: List[str]) -> List[str]:
    """Fold the ``Alternate titles:`` parts into the ``Description:`` line (one catalogue line under one label).

    The merged line takes the description's place, or the first ``Alternate titles:`` line's place when the record
    has no description; every other line keeps its order.

    :param lines: Lines of a folded text.
    :returns: Lines with no ``Alternate titles:`` line.
    :rtype: List[str]
    """
    alt_at = {i for i, line in enumerate(lines) if line.startswith(ALT_PREFIX)}
    if not alt_at:
        return list(lines)
    desc_i = next((i for i, line in enumerate(lines) if line.startswith(DESC_PREFIX)), None)
    description = lines[desc_i][len(DESC_PREFIX):].strip() if desc_i is not None else ""
    kept = merge_title_parts(description, title_parts(lines))[0]
    value = TITLE_SEP.join(([description] if description else []) + kept)
    merged_at = desc_i if desc_i is not None else min(alt_at)
    out: List[str] = []
    for i, line in enumerate(lines):
        if i == merged_at:
            if value:
                out.append(f"{DESC_PREFIX} {value}")
        elif i not in alt_at:
            out.append(line)
    return out


def semantic_v4_text(v3_text: str) -> str:
    """``semantic_v4`` text of a record from its ``semantic_v3`` text.

    Folds continuation lines, drops the :data:`V4_DROP_PREFIXES` lines, then folds the alternate titles into the
    description (:func:`merge_title_lines`).

    :param v3_text: Output of ``semantic_text()``.
    :returns: Embedded text (may be empty).
    :rtype: str
    """
    lines = [line for line in fold_field_lines(v3_text).split("\n") if not line.startswith(V4_DROP_PREFIXES)]
    return "\n".join(merge_title_lines(lines)).strip()


@lru_cache(maxsize=None)
def admit_pattern(path: Path = ADMIT_PATH) -> Pattern[str]:
    """Frozen festival / topic patterns that admit a short record (``eligibility_admit_v4.json``).

    :param path: Admission file: ``{"festival_terms": {topic: regex}, "topic_rules": {topic: regex},
        "flags": {"festival_terms": "i", "topic_rules": ""}, ...}``.
    :returns: One compiled alternation (each block with its own case flag).
    :rtype: Pattern[str]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: List[str] = []
    for block in ("festival_terms", "topic_rules"):
        opener = "(?i:" if "i" in data["flags"][block] else "(?:"
        groups.extend(f"{opener}{pattern})" for _, pattern in sorted(data[block].items()))
    return re.compile("|".join(groups))


@lru_cache(maxsize=None)
def frozen_v4_labels() -> FrozenSet[str]:
    """The frozen boilerplate labels, after checking the file against :data:`BOILERPLATE_SHA256`.

    :returns: The 64 labels.
    :rtype: FrozenSet[str]
    """
    return load_boilerplate_labels(verify_frozen_file(BOILERPLATE_PATH, BOILERPLATE_SHA256))


@lru_cache(maxsize=None)
def frozen_admit_pattern() -> Pattern[str]:
    """The frozen admission pattern, after checking the file against :data:`ADMIT_SHA256`.

    :returns: Compiled pattern (:func:`admit_pattern`).
    :rtype: Pattern[str]
    """
    return admit_pattern(verify_frozen_file(ADMIT_PATH, ADMIT_SHA256))


def v4_measured_text(v3_text: str) -> str:
    """The part of a ``semantic_v3`` text that decides v4 eligibility: every line but the ``Subjects:`` lines (and
    their continuation lines), otherwise unchanged.

    :param v3_text: Output of ``semantic_text()``.
    :returns: The measured text, stripped (equal to ``v3_text`` when it has no ``Subjects:`` line).
    :rtype: str
    """
    out: List[str] = []
    skip = False
    for line in v3_text.split("\n"):
        if line.startswith(FIELD_PREFIXES):
            skip = line.startswith(SUBJECTS_PREFIX)
        if not skip:
            out.append(line)
    return "\n".join(out).strip()


def unbracketed(part: str) -> str:
    """An alternate-title part without square brackets, for the boilerplate comparison (a KTIV frame brackets its
    work, ``[טוביה בן אליעזר, לקח טוב]: ...``; the frozen list holds the same identification without them).

    :param part: One ``;`` part.
    :returns: The part with ``[`` and ``]`` removed, stripped.
    :rtype: str
    """
    return part.replace("[", "").replace("]", "").strip()


def content_lines_v4(measured: str, boiler: AbstractSet[str]) -> List[str]:
    """Lines that say something beyond boilerplate labels and metadata: the v3 rule, ``Subjects:`` never counts and
    alternate-title parts are compared without square brackets.

    :param measured: Output of :func:`v4_measured_text`.
    :param boiler: Frozen boilerplate labels.
    :returns: Content lines.
    :rtype: List[str]
    """
    out: List[str] = []
    for line in measured.split("\n"):
        if not line.strip() or line.startswith(META_PREFIXES + (SUBJECTS_PREFIX,)):
            continue
        if line.startswith(DESC_PREFIX) and line[len(DESC_PREFIX):].strip() in boiler:
            continue
        if line.startswith(ALT_PREFIX) and {unbracketed(p) for p in alt_title_parts(line)} <= boiler:
            continue
        out.append(line)
    return out


def admission_value(content: Sequence[str]) -> str:
    """The values of content lines (labels removed), joined, for the admission test.

    :param content: Output of :func:`content_lines_v4`.
    :returns: ``" ; "``-joined values.
    :rtype: str
    """
    return " ; ".join(line.split(":", 1)[1].strip() if line.startswith(FIELD_PREFIXES) else line.strip()
                      for line in content)


def semantic_v4_ineligible_reason(v3_text: str, labels: Optional[AbstractSet[str]] = None,
                                  admit: Optional[Pattern[str]] = None) -> str:
    """``empty`` / ``short`` / ``boilerplate_only`` / ``eligible`` for a record's ``semantic_v3`` text.

    The v3 rule on :func:`v4_measured_text` (:data:`MIN_CHARS`, then at least one content line), plus one admission:
    a short text whose content names a festival or a topic (:func:`admit_pattern`) is eligible.

    :param v3_text: Output of ``semantic_text()``.
    :param labels: Boilerplate labels (``None``: the frozen list, :func:`frozen_v4_labels`).
    :param admit: Admission pattern (``None``: the frozen one, :func:`frozen_admit_pattern`).
    :returns: Reason.
    :rtype: str
    """
    boiler = frozen_v4_labels() if labels is None else labels
    pattern = frozen_admit_pattern() if admit is None else admit
    measured = v4_measured_text(v3_text)
    if not measured:
        return "empty"
    content = content_lines_v4(measured, boiler)
    if len(measured) < MIN_CHARS:
        value = admission_value(content)
        return "eligible" if value and pattern.search(value) else "short"
    return "eligible" if content else "boilerplate_only"


def semantic_v4_record(v8_text: str) -> Tuple[str, str, str]:
    """The v4 text of one record from its production (``v8``) text, its reason, and its v3 text.

    :param v8_text: ``GenizahDocument.create_text_representation()`` output (with or without scholarly frames).
    :returns: (v4 text, :func:`semantic_v4_ineligible_reason` with the frozen inputs, v3 text).
    :rtype: Tuple[str, str, str]
    """
    v3_text = semantic_text(v8_text)
    return semantic_v4_text(v3_text), semantic_v4_ineligible_reason(v3_text), v3_text
# semantic_v4 end
# =====================================================================================================================


def text_md5(text: str) -> str:
    """md5 hex digest of a text: the corpus ``text_hash`` that the leak checks recompute.

    :param text: Embedded text.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def v4_block_source(path: Path) -> str:
    """The text of a module's ``semantic_v4`` block, from the begin marker line through the end marker line.

    :param path: A ``semantic_text.py`` copy.
    :returns: The block (empty when the file has no block).
    :rtype: str
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    begin = next((i for i, ln in enumerate(lines) if ln.startswith(V4_BLOCK_BEGIN)), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith(V4_BLOCK_END)), None)
    if begin is None or end is None or end < begin:
        return ""
    return "\n".join(lines[begin:end + 1])


def series_of(doc_id: str) -> str:
    """Collection series of a record: ``doc_id`` tokens before the first token containing a digit.

    :param doc_id: Canonical id, e.g. ``Cambridge_CUL_T_S_24_33``.
    :returns: Series, e.g. ``Cambridge_CUL_T_S`` (empty when the id starts with a number).
    :rtype: str
    """
    return "_".join(takewhile(lambda tok: not any(ch.isdigit() for ch in tok), doc_id.split("_")))


def production_text(row: dict, scholarly_frames: bool) -> str:
    """The production (``v8``) text of a corpus row in the requested frames rendering.

    :param row: ``corpus_v1.jsonl`` row (``text``, and ``text_no_frames`` where the no-frames rendering differs).
    :param scholarly_frames: Rendering with the KTIV scholarly frames.
    :returns: Production text.
    :rtype: str
    """
    return row["text"] if scholarly_frames else row.get(TEXT_NO_FRAMES_FIELD, row["text"])


def has_bracketed_alt_part(v3_text: str) -> bool:
    """Whether an ``Alternate titles:`` part opens with a bracketed work (``[Talmud Bavli]: ...``): the shape of a KTIV
    scholarly frame, which v8 texts before the frames attribute never carried (a masked ``[shelfmark]`` does not count).

    :param v3_text: ``semantic_v3`` text.
    :returns: True when such a part exists.
    :rtype: bool
    """
    return any(p.startswith("[") and not p.startswith(MASK) for p in alt_title_parts(v3_text))


def title_merge_counts(v3_text: str) -> Tuple[int, int, int, int]:
    """Diagnostics of the title fold for one record (the same steps as :func:`semantic_v4_text`).

    :param v3_text: ``semantic_v3`` text.
    :returns: (alternate-title parts, parts kept, parts left out as duplicates, parts left out by the cap).
    :rtype: Tuple[int, int, int, int]
    """
    lines = [ln for ln in fold_field_lines(v3_text).split("\n") if not ln.startswith(V4_DROP_PREFIXES)]
    parts = title_parts(lines)
    if not parts:
        return 0, 0, 0, 0
    desc = next((ln[len(DESC_PREFIX):].strip() for ln in lines if ln.startswith(DESC_PREFIX)), "")
    kept, dup, capped = merge_title_parts(desc, parts)
    return len(parts), len(kept), dup, capped


def quantiles(values: Sequence[int]) -> Dict[str, int]:
    """n/p50/p90/p95/p99/max of a list of lengths.

    :param values: Lengths.
    :returns: Quantile dict (empty when there are no values).
    :rtype: Dict[str, int]
    """
    vals = sorted(values)
    if not vals:
        return {}

    def pick(p: float) -> int:
        """Value at quantile ``p`` (nearest rank, rounding down).

        :param p: Quantile in [0, 1].
        :returns: Value.
        :rtype: int
        """
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    return {"n": len(vals), "p50": pick(0.5), "p90": pick(0.9), "p95": pick(0.95), "p99": pick(0.99), "max": vals[-1]}


def duplicate_summary(hashes: Counter) -> Dict[str, object]:
    """Duplicate-text summary of a hash -> count table.

    :param hashes: Text hash -> number of records.
    :returns: Unique texts, groups, records in groups, redundant vectors, group-size buckets, largest groups.
    :rtype: Dict[str, object]
    """
    groups = [n for n in hashes.values() if n > 1]
    return {
        "unique_texts": len(hashes), "groups": len(groups), "records_in_groups": sum(groups),
        "redundant": sum(n - 1 for n in groups),
        "buckets": {"2": sum(n == 2 for n in groups), "3-5": sum(3 <= n <= 5 for n in groups),
                    "6-20": sum(6 <= n <= 20 for n in groups), "21-100": sum(21 <= n <= 100 for n in groups),
                    ">100": sum(n > 100 for n in groups)},
        "largest": sorted(groups, reverse=True)[:8],
    }


def token_lengths(texts: List[str], tokenizer_path: str, batch: int = 2048) -> Dict[str, object]:
    """Token-count quantiles of texts with a local ``tokenizer.json`` (truncation and padding off; counts include
    special tokens). Needs the ``tokenizers`` package (the HDA venv has it); never loads a model.

    :param texts: Texts to count.
    :param tokenizer_path: ``tokenizer.json`` path.
    :param batch: Encode batch size.
    :returns: Quantiles plus over-window counts, or ``{"skipped": reason}``.
    :rtype: Dict[str, object]
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {"skipped": "tokenizers not installed in this interpreter (run with the HDA venv python)"}
    tok = Tokenizer.from_file(tokenizer_path)
    tok.no_truncation()
    tok.no_padding()
    lens: List[int] = []
    for i in range(0, len(texts), batch):
        lens.extend(len(e.ids) for e in tok.encode_batch(texts[i:i + batch], add_special_tokens=True))
    return {**quantiles(lens), **{f">{t}": sum(v > t for v in lens) for t in (384, 512, 768, 1024, 2048)},
            "tokenizer": tokenizer_path}


def load_v3_eligibility(path: Path) -> Dict[str, Tuple[bool, str]]:
    """``doc_id -> (eligible, md5 of the v3 text)`` of an earlier ``corpus_semantic.jsonl``.

    :param path: Earlier corpus_semantic.jsonl (the v9 text, ``semantic_v3``).
    :returns: Eligibility and v3-text md5 per doc_id.
    :rtype: Dict[str, Tuple[bool, str]]
    """
    return {row["doc_id"]: (bool(row["eligible"]), text_md5(row.get("text_v3") or row["text"]))
            for row in iter_corpus(path)}


def compare_with_v3(before: Dict[str, Tuple[bool, str]], eligibility: Dict[str, bool],
                    measured_md5: Dict[str, str]) -> Dict[str, object]:
    """Eligibility changes against an earlier v3 corpus, removals split by cause.

    A removal whose v3 text (without ``Subjects:`` lines) is unchanged would be the text function's fault (the v4 rule
    admits a superset of v3 on one text, so there must be none); a removal whose text changed comes from the source
    (merged-file churn, HDA document-model changes) and needs a human decision.

    :param before: :func:`load_v3_eligibility` output.
    :param eligibility: This run's eligibility per doc_id.
    :param measured_md5: md5 of this run's :func:`v4_measured_text` (or v3 text) per doc_id.
    :returns: Summary with ``removed_text_unchanged`` and ``removed_source_changed`` id lists.
    :rtype: Dict[str, object]
    """
    shared = before.keys() & eligibility.keys()
    removed = sorted(d for d in shared if before[d][0] and not eligibility[d])
    new_ids = eligibility.keys() - before.keys()
    return {
        "shared_ids": len(shared), "v3_eligible_shared": sum(before[d][0] for d in shared),
        "eligible_shared": sum(eligibility[d] for d in shared), "shared_text_unchanged": sum(
            before[d][1] == measured_md5[d] for d in shared),
        "added": sum(eligibility[d] and not before[d][0] for d in shared), "removed": len(removed),
        "removed_text_unchanged": [d for d in removed if before[d][1] == measured_md5[d]],
        "removed_source_changed": [d for d in removed if before[d][1] != measured_md5[d]],
        "ids_only_here": len(new_ids), "eligible_only_here": sum(eligibility[d] for d in new_ids),
        "ids_only_in_v3": len(before.keys() - eligibility.keys()), "new_ids": new_ids,
    }


def corpus_provenance(corpus: Path) -> Dict[str, object]:
    """Input provenance: the corpus file's sha256 and the ``provenance`` block ``build_corpus.py`` wrote beside it.

    :param corpus: corpus_v1.jsonl path.
    :returns: ``{corpus, corpus_sha256, corpus_build_provenance}`` (the last ``None`` for older corpora).
    :rtype: Dict[str, object]
    """
    stats_path = corpus.with_suffix(".stats.json")
    build = json.loads(stats_path.read_text(encoding="utf-8")).get("provenance") if stats_path.exists() else None
    return {"corpus": str(corpus), "corpus_sha256": file_sha256(corpus), "corpus_build_provenance": build}


def check_frames_rendering(build: Optional[dict], scholarly_frames: bool) -> None:
    """Refuse a frames switch the corpus cannot serve (judged from the ``build_corpus.py`` provenance).

    :param build: ``corpus_build_provenance`` (``None`` for corpora built before it existed: checked per row).
    :param scholarly_frames: Requested rendering.
    :raises SystemExit: Frames requested from a corpus rendered without them, or no-frames requested from a corpus
        rendered with frames that holds no ``text_no_frames`` rendering.
    """
    if not build:
        return
    rendered = build.get("scholarly_frames")
    if scholarly_frames and rendered is False:
        raise SystemExit("this corpus was rendered without scholarly frames: use --scholarly-frames off")
    if not scholarly_frames and rendered and not build.get("text_no_frames_written"):
        raise SystemExit("this corpus was rendered with scholarly frames and holds no text_no_frames rendering; "
                         "rebuild it with the current build_corpus.py")


def resolve_v4_inputs(args: argparse.Namespace) -> Tuple[FrozenSet[str], Pattern[str], Dict[str, object]]:
    """Load and pin-check the frozen boilerplate and admission files of a v4 run.

    :param args: Parsed CLI arguments.
    :returns: (labels, admission pattern, text-function metadata).
    :rtype: Tuple[FrozenSet[str], Pattern[str], Dict[str, object]]
    :raises SystemExit: When a file's sha differs from its pin and ``--allow-unpinned`` is not given.
    """
    boiler_path = Path(args.boilerplate_file or BOILERPLATE_PATH)
    admit_path = Path(args.admit_file or ADMIT_PATH)
    shas = {"boilerplate_sha256": file_sha256(boiler_path), "admit_sha256": file_sha256(admit_path)}
    pins = {"boilerplate_sha256": BOILERPLATE_SHA256, "admit_sha256": ADMIT_SHA256}
    off = [f"{k}: file {shas[k]} != pinned {pins[k]}" for k in shas if shas[k] != pins[k]]
    if off and not args.allow_unpinned:
        raise SystemExit("frozen text-function input differs from its pin (regenerate deliberately, update the "
                         "pin in both semantic_text.py copies and the golden fixture, or pass --allow-unpinned "
                         "for a dry run):\n  " + "\n  ".join(off))
    meta = {"boilerplate_file": str(boiler_path), "admit_file": str(admit_path), **shas, "pinned": not off,
            "pin_mismatches": off, "semantic_text_py_sha256": file_sha256(Path(__file__)),
            "v4_block_sha256": text_sha256(v4_block_source(Path(__file__)))}
    return load_boilerplate_labels(boiler_path), admit_pattern(admit_path), meta


def v4_row_stats(stats: Counter, stripped: str, text: str) -> None:
    """Layout diagnostics of one v4 row.

    :param stats: Counter updated in place.
    :param stripped: The row's ``semantic_v3`` text.
    :param text: Its ``semantic_v4`` text.
    """
    stats["continuation_lines_folded"] += len(stripped.split("\n")) - len(fold_field_lines(stripped).split("\n"))
    stats["records_with_subjects_line"] += f"\n{SUBJECTS_PREFIX}" in f"\n{stripped}"
    stats["records_with_doctype_line"] += "\nDocument Type:" in f"\n{stripped}"
    stats["records_with_bracketed_alt_part"] += has_bracketed_alt_part(stripped)
    n_parts, n_kept, n_dup, n_cap = title_merge_counts(stripped)
    stats["records_with_alt_titles"] += bool(n_parts)
    stats["alt_title_parts"] += n_parts
    stats["alt_title_parts_kept"] += n_kept
    stats["alt_title_parts_duplicate"] += n_dup
    stats["alt_title_parts_capped"] += n_cap
    stats["records_alt_titles_capped"] += bool(n_cap)
    stats["v4_text_with_dropped_label"] += any(ln.startswith((ALT_PREFIX,) + V4_DROP_PREFIXES) for ln in text.split("\n"))


def main() -> None:
    """Write ``corpus_semantic.jsonl`` plus ``.stats.json`` and print eligibility / duplicate / length counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"), help="build_corpus.py output (v8 text)")
    parser.add_argument("--out", default=str(OUT_DIR / "corpus_semantic.jsonl"))
    parser.add_argument("--text-version", choices=[TEXT_VERSION_SEMANTIC_V3, TEXT_VERSION_SEMANTIC_V4],
                        default=TEXT_VERSION_SEMANTIC_V3, help="embedded text (default reproduces the v9 inputs)")
    parser.add_argument("--scholarly-frames", choices=["on", "off"], default="on",
                        help="production rendering with / without KTIV scholarly frames (off: semantic_v4_nf, "
                             "read from the corpus' text_no_frames)")
    parser.add_argument("--boilerplate-file", default=None,
                        help=f"frozen labels (semantic_v4 default {BOILERPLATE_PATH}; semantic_v3 recomputes "
                             "them unless this is given)")
    parser.add_argument("--admit-file", default=None, help=f"semantic_v4 admission patterns (default {ADMIT_PATH})")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="accept boilerplate / admission files whose sha256 differs from the pins (dry runs)")
    parser.add_argument("--v3-corpus", default=None,
                        help="an earlier corpus_semantic.jsonl: report eligibility changes, exit 1 on any removal")
    parser.add_argument("--accept-source-removals", action="store_true",
                        help="--v3-corpus: do not fail on removals whose v3 text changed (merged-file churn; recorded)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json for token-length stats (needs tokenizers)")
    parser.add_argument("--min-count", type=int, default=50, help="semantic_v3 recompute: records sharing a label")
    parser.add_argument("--no-alt-titles", action="store_true", help="semantic_v3 recompute: descriptions only")
    args = parser.parse_args()
    corpus = Path(args.corpus)
    v4 = args.text_version == TEXT_VERSION_SEMANTIC_V4
    frames = args.scholarly_frames == "on"
    provenance = corpus_provenance(corpus)
    check_frames_rendering(provenance["corpus_build_provenance"], frames)
    if v4:
        boiler, admit, text_function = resolve_v4_inputs(args)
        version = v4_text_version(frames)
    else:
        if args.boilerplate_file:
            boiler = load_boilerplate_labels(Path(args.boilerplate_file))
        else:
            boiler = frozenset(boilerplate_descriptions(corpus, args.min_count, include_alt_titles=not args.no_alt_titles))
        admit, version = None, TEXT_VERSION_SEMANTIC_V3
        text_function = {"boilerplate_file": args.boilerplate_file or "recomputed",
                         "boilerplate_sha256": file_sha256(Path(args.boilerplate_file)) if args.boilerplate_file else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    transitions: Counter = Counter()
    hashes: Counter = Counter()
    eligible_hashes: Counter = Counter()
    boiler_only_desc: Counter = Counter()  # which boilerplate labels make records ineligible (for review)
    eligible_chars: List[int] = []
    eligible_texts: List[str] = []
    eligibility: Dict[str, bool] = {}
    measured_md5: Dict[str, str] = {}
    residual: Dict[str, List[str]] = {"ids": [], "examples": []}
    with open(args.out, "w", encoding="utf-8") as out:
        for row in iter_corpus(corpus):
            v8 = production_text(row, frames)
            unmasked = semantic_text(v8, mask=False)
            stripped = semantic_text(v8)
            reason_v3 = ineligible_reason(v4_measured_text(stripped) if v4 else stripped, boiler)
            if v4:
                text = semantic_v4_text(stripped)
                reason = semantic_v4_ineligible_reason(stripped, boiler, admit)
            else:
                text, reason = stripped, reason_v3
            eligible = reason == "eligible"
            if eligible and not text:
                raise AssertionError(f"{row['doc_id']}: eligible with an empty embedded text")
            eligibility[row["doc_id"]] = eligible
            measured_md5[row["doc_id"]] = text_md5(v4_measured_text(stripped) if v4 else stripped)
            text_hash = text_md5(text)
            hashes[text_hash] += 1
            eligible_hashes[text_hash] += eligible
            shelf = own_shelfmark(v8)
            stats["records"] += 1
            stats[reason] += 1
            transitions[f"{reason_v3}->{reason}"] += 1
            stats["editor_attribution_removed"] += bool(EDITOR_RE.search(v8) or PGP_EDITOR_RE.search(v8))
            if reason == "boilerplate_only":
                boiler_only_desc[description_value(stripped) or "<no description>"] += 1
            stats["description_boilerplate"] += description_value(stripped) in boiler
            stats["own_shelfmark_before_masking"] += bool(len(shelf) > 4 and shelf in unmasked)
            stats["own_shelfmark_still_in_text"] += bool(len(shelf) > 4 and shelf in stripped)
            n_masked = mask_shelfmarks(unmasked, shelf)[1]
            stats["shelfmark_masked_records"] += bool(n_masked)
            stats["shelfmark_masked_records_eligible"] += bool(n_masked) and eligible
            stats["shelfmark_masked_spans"] += n_masked
            if v4:
                stats["admitted_short"] += reason_v3 == "short" and eligible
                v4_row_stats(stats, stripped, text)
                if TEXT_NO_FRAMES_FIELD in row:
                    other = semantic_v4_ineligible_reason(semantic_text(production_text(row, not frames)), boiler, admit)
                    stats["records_frames_rendering_differs"] += 1
                    stats["eligible_only_in_this_rendering"] += eligible and other != "eligible"
                    stats["eligible_only_in_other_rendering"] += other == "eligible" and not eligible
                elif not frames:
                    stats["no_frames_rows_without_nf_rendering"] += has_bracketed_alt_part(stripped)
            if eligible:
                eligible_chars.append(len(text))
                if args.tokenizer:
                    eligible_texts.append(text)
                hit = RESIDUAL_ID_RE.search(text)
                if hit:
                    residual["ids"].append(row["doc_id"])
                    if len(residual["examples"]) < 12:
                        residual["examples"].append(
                            f"{row['doc_id']}: ...{text[max(0, hit.start() - 40):hit.end() + 20]}...")
            out.write(json.dumps({
                "doc_id": row["doc_id"], "text": text, "text_v3": stripped, "text_version": version,
                "eligible": eligible, "reason": reason, "text_hash": text_hash, "text_sha256": text_sha256(text),
                "series": series_of(row["doc_id"]), "topics": row.get("topics") or [], "frames": row.get("frames") or [],
                "framed": bool(row.get("framed")), "sukkot_gold": bool(row.get("sukkot_gold")),
            }, ensure_ascii=False) + "\n")
    if stats["no_frames_rows_without_nf_rendering"]:
        raise SystemExit(f"--scholarly-frames off: {stats['no_frames_rows_without_nf_rendering']} rows carry "
                         "bracketed alternate-title parts (frames) but no text_no_frames rendering; rebuild the "
                         "corpus with the current build_corpus.py")
    dup_groups = {h: n for h, n in hashes.items() if n > 1}
    eligible_dups = {h: n for h, n in eligible_hashes.items() if n > 1}
    summary: Dict[str, object] = {
        **dict(stats),
        "text_version": version,
        "scholarly_frames": frames if v4 else None,
        "ineligible": stats["records"] - stats["eligible"],
        "boilerplate_descriptions": len(boiler),
        "duplicate_text_groups": len(dup_groups),
        "records_in_duplicate_groups": sum(dup_groups.values()),
        "eligible_duplicate_text_groups": len(eligible_dups),
        "eligible_records_in_duplicate_groups": sum(eligible_dups.values()),
        "largest_duplicate_groups": sorted(dup_groups.values(), reverse=True)[:8],
        "eligible_duplicates": duplicate_summary(Counter({h: n for h, n in eligible_hashes.items() if n})),
        "eligible_text_chars": quantiles(eligible_chars),
        "reason_transitions_v3_to_this": dict(transitions),
        "residual_id_tokens_eligible": {"records": len(residual["ids"]), "examples": residual["examples"]},
        "boilerplate_only_top_descriptions": dict(boiler_only_desc.most_common(25)),
    }
    if args.tokenizer:
        summary["eligible_text_tokens"] = token_lengths(eligible_texts, args.tokenizer)
    failing: List[str] = []
    if args.v3_corpus:
        vs = compare_with_v3(load_v3_eligibility(Path(args.v3_corpus)), eligibility, measured_md5)
        new_ids = vs.pop("new_ids")
        summary["vs_v3_corpus"] = {"v3_corpus": args.v3_corpus, **vs,
                                   "residual_id_tokens_new_ids": len(set(residual["ids"]) & new_ids)}
        failing = vs["removed_text_unchanged"] + ([] if args.accept_source_removals else vs["removed_source_changed"])
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    stats_path = Path(args.out).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({
        **summary, "boilerplate_list": sorted(boiler), "min_count": args.min_count, "min_chars": MIN_CHARS,
        "text_function": {"version": version, **text_function}, "input": provenance,
    }, ensure_ascii=False, indent=1))
    if failing:
        raise SystemExit(f"{len(failing)} records eligible in {args.v3_corpus} are ineligible here, e.g. {failing[:5]} "
                         "(see vs_v3_corpus: removed_text_unchanged must be empty; removed_source_changed comes from "
                         "the merged file / HDA document model and passes only with --accept-source-removals)")


if __name__ == "__main__":
    main()
