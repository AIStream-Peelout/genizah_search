"""Training mixture v3 for the semantic text embedder (text track step 3 of 4).

Every pair teaches *subject* similarity or cross-lingual naming, never record identity. Document texts are the v3
semantic text (``v3/corpus_semantic.jsonl``: no ``Document ID:`` / ``Shelf Mark:`` lines, no editor credits, and the
shelf marks / PGPIDs that descriptions cite already masked as ``[shelfmark]`` by ``semantic_text.py``, so training,
the frozen eval and production embed the same text; :func:`mask_ids` re-applies the same masking as a no-op guard). Nothing that
``build_eval_v3.py`` holds out (records, subjects, linked subjects, eval queries) appears on either side of a pair.

Families (``family`` -> ``kind`` / ``mask_level``):

* ``synthetic_<type>`` (q2d) -- LLM-written queries (``synthetic_queries/{r1,r2}/{out,out_he}``) for train-pool
  records -> the record. implicit / topical / he_implicit / he_topical describe the subject (``subject`` masking);
  specific / hebrew / translit name the record itself (``doc`` masking).
* ``subject`` (q2d, subject) -- an English or Hebrew name / Sefaria alternate title of a *seen* subject -> a train
  record carrying it (strong label). At most :data:`MAX_SUBJECT_PAIRS_PER_RECORD` per record; per-subject quota
  ``ceil(k * sqrt(n))`` so big subjects do not dominate. Sub-topic names (Sefaria slugs merged into a topic, e.g.
  "Shofar" under Rosh Hashanah; raw PGP tags merged into a PGP subject, e.g. "Muslim marriage contract" under
  marriage) only pair with records actually about them (frame -> Sefaria refs / the record's own raw tag). Names
  that literally appear in the record are not used (label title copies); a subject whose every name is literal is
  skipped for that record.
* ``t2t_xling`` / ``t2t_alias`` / ``t2t_parent`` (t2t, subject) -- Sefaria concept graph, no records: English <->
  Hebrew name of one topic, alternate title -> primary name, child -> parent along participates-in / member-of /
  temporally-contained-in / is-a (coarse parents such as religion / middot dropped). Nodes within 2 hops of a
  held-out subject are excluded (see :func:`excluded_nodes` for the hop rule).
* ``content`` (q2d, doc) -- a 6-12 word Hebrew-script span of a train record's transcription -> the record's text
  without its Transcription / Translation lines (skipped when that text is boilerplate, or when a 4-word run of the
  span is copied in it).
* ``d2d_cross_genre`` (d2d, subject) -- two train records sharing a topic / PGP / Sefaria subject but from different
  works and genres (KTIV work + domain, PGP document type), token Jaccard < :data:`D2D_MAX_JACCARD`.

Dropped vs v2: label title copies, scholarship page pairs (they match on shelf marks), Sefaria passage pairs.

Row schema::

    {anchor, positive, family, kind, mask_level, doc_id, mask_ids, anchor_mask_ids, anchor_lang, ...}

``mask_ids`` = every subject id of the positive record (strong and weak), ``"M:<sid>"`` for every subject its
Description / Alternate titles name without carrying it (``subjects_v1`` ``mentions``; never a held-out / linked id;
off with ``--no-mention-masks``), plus ``"D:<doc_id>"`` (empty for t2t); ``anchor_mask_ids`` = the same for the anchor
record of a d2d row (empty otherwise). Trainer contract: in-batch candidate j is not a negative for row i when their
ids overlap -- subject ids only for ``mask_level == "subject"`` rows (``M:`` ids are broad: they mask only through the
row's own ``anchor_mention_id`` / ``anchor_topic_ids``), ``D:`` ids (+ identical text) only for ``mask_level ==
"doc"`` rows. ``mask_level`` is ``doc`` for :data:`DOC_LEVEL` families, except ``synthetic_hebrew`` /
``synthetic_translit`` rows whose anchor has <= :data:`SHORT_ANCHOR_MAX_TOKENS` normalised tokens (topical queries
such as "כתובה קראית": ``subject``; ``--legacy-mask-level`` keeps them ``doc``). Extra fields: ``anchor_subject`` and
``anchor_mention_id`` (``"M:<anchor_subject>"``; subject family), ``concept_ids`` (t2t: the Sefaria nodes of the pair),
``grade`` / ``round`` (synthetic), ``anchor_doc_id`` (d2d), ``text_hash`` (semantic-text hash of the positive record),
``anchor_topic_ids`` (every q2d row except ``content``: the festival topics the anchor names, e.g. "Passover
Haggadah" -> ``["M:topic:pesach", "topic:pesach"]``; the trainer adds them to the row's relevance so other records of
that festival are never its negatives. Not set when the anchor is a canon / folio reference ("Talmud Pesahim 5a") or
the positive carries the festival only as canon or not at all; off with ``--no-festival-guard``).

Festival / canon rules (v10, plan A6 / A7 / C2 / C4): the subject family keys carriers by ``(subject, class)``: for
``topic:`` / ``sefaria:`` subjects, ``canon`` carriers (the subject is in the record's ``canon`` list, including weak
festival canon; every ``sefaria:`` label is canon by construction unless ``--no-sefaria-canon``) take only law names
(:func:`build_subjects.is_law_name`: "laws of Pesach", "הלכות סוכה"), ``content`` carriers (strong, not canon) only
the other names; work / PGP / domain subjects take every name (class ``all``). Each key has its own
``ceil(k * sqrt(n))`` quota (off with ``--no-name-form-gate``: strong carriers, every name, one quota). A bare
festival name (:func:`bare_festivals`: names a festival, not a law name, not a canon / folio reference) pairs only
with a strong ``topic:<festival>`` carrier of every festival it names; t2t pairs of a bare festival name and a canon
name ("Purim" <-> "Esther") and q2d rows whose bare festival anchor has a positive carrying that festival only as
canon are dropped (off with ``--no-festival-canon-drops``). :data:`build_subjects.NAME_STOPLIST` names are never
anchors (off with ``--no-name-stoplist``).

Leak filter (anchor and positive, niqqud stripped, case / punctuation normalised), against every eval query (probe,
llm_t2, known-item; ``--probe-scope all`` also counts the 91 concept_probe_v1 queries of seen topics) and every
held-out subject name: exact match; containment either way when the contained string has >= ``--min-contain``
tokens (held-out names at any length, contained in the text); content-token Jaccard >= 0.5 with >= ``--min-shared``
shared tokens; the held-out concept regexes (``held_out_subjects.json["concept_patterns"]``). Linked-subject name
queries are blocked as exact anchors. A record whose semantic text fails the filter is excluded from every family.
Acceptance-gate queries marked ``blocked`` (``--gate-dir GATE/queries.jsonl``, or ``--gate-spec gate_v1_spec.json``
before the gate dir is built) are eval strings for ANCHORS and short t2t terms only (source ``gate``; exact,
containment of a >= 2-token query inside an anchor, Jaccard; one-token queries exact only): no blocked gate query is a
training anchor, inside one, or a near-copy of one, while records whose text contains one stay trainable (plan H11
checks anchors only).

Every row is re-scanned before the file is written (``Document ID:`` / ``Shelf Mark:`` / shelf-mark regexes, held-out
doc ids and texts, held-out / linked subject ids, leak filter); the file is not written if anything is found.

Final run (eval dir from ``build_eval_v3.py --final``): no subject is held out, so nothing is gated by subject, no
concept regex applies and the Sefaria graph has no 2-hop exclusion; the six ``focus`` subjects train like seen
subjects (``subject`` / ``d2d`` families included). Their eval queries (probe, llm_t2 and, by default, their name
queries: ``--focus-names block``) join the leak filter as eval strings (exact / containment of >= ``--min-contain``
tokens / Jaccard, not the any-length held-out-name rule): no focus query is a training anchor or a near-copy of one,
but a ONE-word query ("Haggadah", "סוכות") still occurs inside longer anchors and record texts, because the subjects
are trained on. With the concept regexes gone, containment also matches Hebrew-prefixed first words
(``--he-prefix-contain auto`` = on for final-run eval dirs: "ושמחת בית השואבה" contains the Sukkot probe "שמחת בית
השואבה"), and ``--final-strict auto`` (on for final-run eval dirs, off for v3) adds three tightenings: a record whose
text equals a held-out record's up to niqqud / case / punctuation / spacing is not trained on (v3 only excluded
byte-identical text; 290 such near-twins, e.g. "Hilkhot ha-Rif: Pesahim 1 – 2" / "Pesahim 1; 2"); a focus
probe / llm_t2 query of two tokens is blocked inside an anchor ("סוגיית בדיקת חמץ בתלמוד" contains the probe
"בדיקת חמץ"); exact matching ignores spacing / punctuation inside words ("Hosh'ana Rabbah" = "Hoshana Rabbah").
The stats gain ``focus_report``:
rows touching the six subjects / their v3 linked subjects / v3-excluded Sefaria nodes / records v3 held out, and pair
counts by family vs the v3 mixture::

    python3 build_training_mix_v3.py --eval-dir AUDIT_ROOT/v3/eval_final --out AUDIT_ROOT/v3/train_mix_v3final.jsonl

Run (json only, no model; ~1.5-6 min depending on NAS cache, ~0.4 GB)::

    python3 build_training_mix_v3.py --self-test
    python3 build_training_mix_v3.py
    python3 build_training_mix_v3.py --verify AUDIT_ROOT/v3/train_mix_v3.jsonl   # re-scan any mixture file
"""

import argparse
import glob
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from build_eval_v3 import SHELFMARK_RE as QUERY_SHELFMARK_RE
from build_eval_v3 import merged_pgp_info, synthetic_pools
from build_subjects import (FESTIVAL_CANON_RE, LITURGY_CUE_RE, MERGED, NAME_STOPLIST, PGP_SUBJECT_SEFARIA,
                            TOPIC_EXTRA_PHRASES, TOPIC_SEFARIA, festival_mentions, is_law_name, iter_jsonl, parse_ref,
                            sefaria_names, title_index)
from build_train_pairs import TOPIC_PHRASES, content_spans, split_text
from embed_utils import AUDIT_ROOT
from semantic_text import EXTRA_ID_RE, MASK, MIN_CHARS, PROD_SHELFMARK_RE, content_lines
from semantic_text import mask_shelfmarks as mask_ids

HERE = Path(__file__).resolve().parent
V3 = AUDIT_ROOT / "v3"
PROBE = HERE / "concept_probe_v1.json"

# ---------------------------------------------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------------------------------------------
SYNTHETIC_TYPES = ("implicit", "topical", "hebrew", "specific", "he_implicit", "he_topical", "translit")
DOC_LEVEL = {"synthetic_specific", "synthetic_hebrew", "synthetic_translit", "content"}  # known-item families
# topical in practice (46-47 % of these anchors are <= 3 tokens, "כתובה קראית" x 8): short anchors mask by subject
SHORT_ANCHOR_FAMILIES = {"synthetic_hebrew", "synthetic_translit"}
SHORT_ANCHOR_MAX_TOKENS = 3
KIND = {"t2t_xling": "t2t", "t2t_alias": "t2t", "t2t_parent": "t2t", "d2d_cross_genre": "d2d"}  # default q2d
MAX_SUBJECT_PAIRS_PER_RECORD = 2
D2D_MAX_JACCARD = 0.2
D2D_MAX_USES = 2  # one record anchors / is the positive of at most this many d2d rows
D2D_PASSES = 3
SUBJECT_KINDS_D2D = ("topic:", "pgp:", "sefaria:")  # a shared work / domain is not "different genre"
TRAINABLE_STATUSES = ("seen", "focus")  # focus = final run: trained on, scored over held-out records only
TOPIC_ALL_SUB = {"tefillin_mezuzah"}  # topics whose Sefaria slugs are all sub-topics (no parent slug)
NAME_FORM_KINDS = ("topic:", "sefaria:")  # subjects whose carriers split into canon (law names) / content (bare names)
MENTION_PREFIX = "M:"  # "M:<sid>": the record's catalogue text names the subject without carrying it (broad id)
# folio / daf references make an anchor a canon known-item query ("Pesahim 5a", "דף ה", 'ע"א')
FOLIO_RE = re.compile(r"\b\d+[ab]\b|(?<![א-ת])[בהול]?דף(?![א-ת])|ע[\"״][אב]")
# Mix-side additions to build_subjects.FESTIVAL_CANON_RE for query anchors: LLM-written known-item queries cite the
# Talmud the way readers do ("Bavli Sukkah chapters 1-5", "Pesachim daf 6", "גמרא ראש השנה", "כותח הבבלי"); without
# these such rows are dropped as bare-festival / canon pairs and tagged with every festival record (finding 18).
ANCHOR_CANON_RE = re.compile(
    r"\b(?:bavli|yerushalmi|gemar[ao]|gemor[ao]|talmudic|mishnaic|daf|amud|mas+e[kc]h?(?:et|es|ta|tos|tot)"
    r"|pesa(?:c?h|ḥ)im)\b"
    r"|(?<![א-ת])[בהוכלמשד]{0,2}(?:בבלי|ירושלמי|גמרא|סוגיי?ת|סוגיא)(?![א-ת])"
    r"|(?<![א-ת])גמ['׳]", re.I)

# ---------------------------------------------------------------------------------------------------------------
# t2t (Sefaria concept graph)
# ---------------------------------------------------------------------------------------------------------------
T2T_LINKS_UP = ("participates-in", "member-of", "temporally-contained-in", "is-a")
COARSE_PARENTS = {"religion", "middot", "emotions", "history", "life", "philosophy", "numbers", "letters", "days",
                  "peoples", "avodat-hashem", "values", "traits", "quality", "concepts", "people", "torah",
                  "jewish-people", "god", "halakhah", "laws", "texts", "words", "phrase"}
COARSE_FAN_IN = 50  # a parent with more whitelisted children than this is a category, not a concept
# 2-hop exclusion: UI display links and gender links are not concept proximity, and a hop never continues through a
# category hub (degree > HOP_HUB_DEGREE: "acts", "male", "biblical-figures", "role-of-person" ...). The literal rule
# (every link, through hubs) removes 64 % of the graph, e.g. every biblical figure via "male".
HOP_SKIP_LINKS = {"displays-under", "displays-above", "gender-of", "has-gender"}
HOP_HUB_DEGREE = 40
T2T_ALIAS_PER_NODE = 2
T2T_PARENTS_PER_NODE = 2

# ---------------------------------------------------------------------------------------------------------------
# normalisation / leak filter
# ---------------------------------------------------------------------------------------------------------------
NIQQUD_RE = re.compile(r"[֑-ׇ]")
HEB_LETTER = re.compile(r"[א-ת]")
LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
STOPWORDS = {"a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "from", "by", "with", "about", "as",
             "is", "are", "was", "be", "its", "his", "her", "their", "that", "this", "which", "who",
             "של", "את", "על", "עם", "מן", "אל", "או", "כי", "לא", "הוא", "היא", "זה", "זו", "אשר", "גם", "כל"}
# single-word held-out names that are ordinary words inside transcriptions ("עבד" also spells Abd- names, "כשר" =
# valid): checked in anchors, but in record texts only through the concept regexes, which leave them out on purpose
AMBIGUOUS_HELD_NAMES = {"עבד", "כשר"}
HE_PREFIX_LETTERS = "בהוכלמש"  # Hebrew one-letter prefixes (and, the, in, as, to, from, that)

ID_LINES = ("Document ID:", "Shelf Mark:")


def strip_niqqud(text: str) -> str:
    """Remove Hebrew niqqud and cantillation marks (U+0591-U+05C7).

    :param text: Any text.
    :returns: Text without the marks.
    :rtype: str
    """
    return NIQQUD_RE.sub("", text)


def clean_anchor(text: str) -> str:
    """Niqqud-stripped, whitespace-collapsed anchor / term.

    :param text: Raw anchor.
    :returns: Cleaned anchor.
    :rtype: str
    """
    return " ".join(strip_niqqud(text or "").split())


def clean_name(text: str) -> str:
    """A subject / concept name as an anchor: :func:`clean_anchor` without editorial brackets ("[Rabbi] Shimon").

    :param text: Raw name.
    :returns: Cleaned name.
    :rtype: str
    """
    return clean_anchor(re.sub(r"[\[\]]", "", text or ""))


def norm_tokens(text: str) -> List[str]:
    """Leak-check tokens: niqqud stripped, lower-cased, punctuation (incl. geresh / gershayim and ``_``) as spaces.

    :param text: Any text.
    :returns: Token list.
    :rtype: List[str]
    """
    return re.sub(r"[^\w\s]|_", " ", strip_niqqud(text).lower()).split()


def content_tokens(tokens: Iterable[str]) -> Set[str]:
    """Tokens that carry content: stopwords and one-character tokens (split-off prefixes, initials) removed.

    :param tokens: Output of :func:`norm_tokens`.
    :returns: Token set.
    :rtype: Set[str]
    """
    return {t for t in tokens if len(t) > 1 and t not in STOPWORDS}


def norm_key(text: str) -> str:
    """Normalised comparison key of a whole text: :func:`norm_tokens` joined by single spaces.

    Two texts with the same key differ only in niqqud, case, punctuation or spacing ("Page from a Hebrew medical
    treatise" / "... treatise."; "Pesahim 1 – 2" / "Pesahim 1; 2"). Used by the normalised-text hold-out gate.

    :param text: Any text.
    :returns: Key.
    :rtype: str
    """
    return " ".join(norm_tokens(text))


def held_norm_keys(rows: Dict[str, dict], held: Set[str]) -> Set[str]:
    """Normalised-text keys (:func:`norm_key`) of the eligible held-out records.

    :param rows: Semantic rows (:func:`load_semantic`).
    :param held: Held-out doc ids.
    :returns: Keys.
    :rtype: Set[str]
    """
    return {norm_key(r["text"]) for d, r in rows.items() if d in held and r["eligible"]}


def contains_seq(tokens: Sequence[str], seq: Sequence[str]) -> bool:
    """Whether ``seq`` occurs as a contiguous run of ``tokens``.

    :param tokens: Haystack tokens.
    :param seq: Needle tokens (non-empty).
    :returns: True on a match.
    :rtype: bool
    """
    n, first = len(seq), seq[0]
    return any(tokens[i] == first and list(tokens[i:i + n]) == list(seq) for i in range(len(tokens) - n + 1))


def script_of(text: str) -> str:
    """Dominant script of an anchor.

    :param text: Anchor.
    :returns: ``he`` (more Hebrew than Latin letters), ``latin`` or ``other``.
    :rtype: str
    """
    he, la = len(HEB_LETTER.findall(text)), len(LATIN_LETTER.findall(text))
    if he == la == 0:
        return "other"
    return "he" if he > la else "latin"


def id_hits(text: str) -> List[str]:
    """Record-identity strings in a text: id lines, production shelf-mark pattern, PGPIDs.

    :param text: Anchor or document text.
    :returns: Matched strings (empty when clean).
    :rtype: List[str]
    """
    hits = [p for p in ID_LINES if p in text]
    hits += [m.group(0) for m in PROD_SHELFMARK_RE.finditer(text)]
    hits += [m.group(0) for m in EXTRA_ID_RE.finditer(text)]
    return hits


def own_mark_in(anchor: str, own_mark: str) -> bool:
    """Whether an anchor spells the record's own shelf mark (alphanumerics only, as ``build_eval_v3.known_items``).

    :param anchor: Anchor text.
    :param own_mark: The record's shelf mark.
    :returns: True when the mark (>= 6 alphanumerics) is inside the anchor.
    :rtype: bool
    """
    sm = re.sub(r"\W", "", own_mark.lower())
    return len(sm) >= 6 and sm in re.sub(r"\W", "", anchor.lower())


# ---------------------------------------------------------------------------------------------------------------
# festival / canon / mask-level rules (shared by the builder and verify)
# ---------------------------------------------------------------------------------------------------------------
_STOPLIST_FOLDED = {n.casefold() for n in NAME_STOPLIST}


class SubjectLabels(NamedTuple):
    """Per-record labels from ``subjects_v1.jsonl`` (missing keys of older files read as empty)."""

    subjects: Dict[str, Set[str]]  # every subject id, strong and weak
    weak: Dict[str, Set[str]]
    weak_reason: Dict[str, Dict[str, str]]  # weak id -> "canon" | "incidental" | "title"
    canon: Dict[str, Set[str]]  # topic ids matched on a canon frame / description (festival ones are weak)
    mentions: Dict[str, Set[str]]  # subjects the catalogue text names that the record does not carry strongly
    gold_override: Set[str]  # records whose strong label is a documented override (sukkot gold)


def mask_level_of(family: str, anchor: str, short_anchor_subject: bool = True) -> str:
    """Mask level of a row (plan C6).

    :param family: Row family.
    :param anchor: Cleaned anchor text.
    :param short_anchor_subject: ``synthetic_hebrew`` / ``synthetic_translit`` anchors of at most
        :data:`SHORT_ANCHOR_MAX_TOKENS` normalised tokens are topical and mask by subject; False reproduces v9 (always
        ``doc`` for :data:`DOC_LEVEL` families).
    :returns: ``doc`` or ``subject``.
    :rtype: str
    """
    if family not in DOC_LEVEL:
        return "subject"
    if short_anchor_subject and family in SHORT_ANCHOR_FAMILIES and len(norm_tokens(anchor)) <= SHORT_ANCHOR_MAX_TOKENS:
        return "subject"
    return "doc"


def canon_reference(text: str) -> bool:
    """Whether a query / name refers to a canon text (scripture, tractate, law code) or a folio of one.

    A folio / daf pattern always counts; :data:`build_subjects.FESTIVAL_CANON_RE` or :data:`ANCHOR_CANON_RE` counts
    unless a liturgy cue also matches (:data:`build_subjects.LITURGY_CUE_RE`: "piyyut for Purim on the Book of
    Esther" is liturgy).

    :param text: Cleaned anchor or name.
    :returns: True for a canon reference.
    :rtype: bool
    """
    if FOLIO_RE.search(text):
        return True
    canon = FESTIVAL_CANON_RE.search(text) or ANCHOR_CANON_RE.search(text)
    return bool(canon) and not LITURGY_CUE_RE.search(text)


def bare_festivals(text: str) -> Set[str]:
    """Festivals a query / name names as the festival itself (not as a law name or a canon reference).

    :param text: Cleaned anchor or name.
    :returns: Festival topic names (``festival_mentions``), empty for law names ("laws of Pesach") and canon
        references ("Talmud Bavli Sukkah", "Pesahim 5a").
    :rtype: Set[str]
    """
    found = festival_mentions(text)
    if not found or is_law_name(text) or canon_reference(text):
        return set()
    return found


def festival_canon_pair(a: str, b: str) -> bool:
    """Whether a term pair links a bare festival name to a canon name ("Purim" <-> "Esther", "פורים" <-> "מגילה").

    :param a: One term.
    :param b: The other term.
    :returns: True when one side is a bare festival name and the other a canon reference.
    :rtype: bool
    """
    return bool((bare_festivals(a) and canon_reference(b)) or (bare_festivals(b) and canon_reference(a)))


def is_stoplisted(name: str) -> bool:
    """Whether a name is in :data:`build_subjects.NAME_STOPLIST` (case-insensitive exact match).

    :param name: Cleaned name.
    :returns: True for an ambiguous name never used as an anchor ("אבל", "מזון", "Get").
    :rtype: bool
    """
    return name.casefold() in _STOPLIST_FOLDED


def festival_carry(labels: Optional[SubjectLabels], doc_id: str, topic_id: str) -> str:
    """How a record carries a festival topic.

    :param labels: Subject labels (None: nothing is carried).
    :param doc_id: Record id.
    :param topic_id: ``topic:<festival>``.
    :returns: ``strong`` (strong label), ``canon`` (weak, reason canon), ``weak`` (weak, incidental / title / unknown
        reason), ``mention`` (catalogue text names it, no label) or ``none``.
    :rtype: str
    """
    if labels is None:
        return "none"
    subj, weak = labels.subjects.get(doc_id, set()), labels.weak.get(doc_id, set())
    if topic_id in subj and topic_id not in weak:
        return "strong"
    if topic_id in weak:
        reason = labels.weak_reason.get(doc_id, {}).get(topic_id)
        canon_weak = reason == "canon" or (reason is None and topic_id in labels.canon.get(doc_id, set()))
        return "canon" if canon_weak else "weak"
    if topic_id in labels.mentions.get(doc_id, set()):
        return "mention"
    return "none"


def canon_only_festivals(anchor: str, doc_id: str, labels: Optional[SubjectLabels]) -> List[str]:
    """Festivals a bare-festival anchor names that its positive carries only as canon (plan A7 drop / assert).

    :param anchor: Cleaned anchor.
    :param doc_id: Positive record.
    :param labels: Subject labels.
    :returns: Sorted festival names (empty = the row is fine).
    :rtype: List[str]
    """
    return sorted(f for f in bare_festivals(anchor) if festival_carry(labels, doc_id, f"topic:{f}") == "canon")


class LeakFilter:
    """Eval-query / held-out-concept leak check for anchors and positives (see the module docstring)."""

    def __init__(self, eval_strings: Dict[str, str], held_names: Iterable[str], anchor_exact: Iterable[str],
                 patterns: Dict[str, str], min_contain: int = 3, min_shared: int = 2, he_prefixes: bool = False,
                 anchor_contain: Iterable[Tuple[str, str]] = (), glued: bool = False,
                 anchor_strings: Optional[Dict[str, str]] = None) -> None:
        """Index the eval strings.

        :param eval_strings: Eval query text -> source (``probe``, ``llm_t2``, ``known_item``).
        :param held_names: Held-out subject names (exact / contained at any length / Jaccard).
        :param anchor_exact: Strings blocked as exact anchors only (linked-subject name queries).
        :param patterns: Held-out subject id -> concept regex (case-insensitive).
        :param min_contain: Minimum tokens of the contained string for a containment hit (eval queries).
        :param min_shared: Minimum shared content tokens for a Jaccard hit.
        :param he_prefixes: Containment of an eval query (>= ``min_contain`` tokens) also matches when the text's
            first matching word carries up to two Hebrew prefix letters (:data:`HE_PREFIX_LETTERS`: "ושמחת בית
            השואבה" contains the probe "שמחת בית השואבה"). Off reproduces v3, where the held-out concepts were
            covered by the concept regexes instead; on for final-run eval dirs (``--he-prefix-contain auto``).
        :param anchor_contain: ``(query, source)`` pairs (final run: the focus subjects' probe / llm_t2 queries) whose
            two-token forms (below ``min_contain``) also count as contained in an ANCHOR ("סוגיית בדיקת חמץ בתלמוד"
            contains the probe "בדיקת חמץ"); record texts are not checked for them (the subjects are trained on).
            Prefix-aware like the eval queries when ``he_prefixes`` is on ("להושענא רבה").
        :param glued: Exact matching also ignores spacing / punctuation inside words ("Hosh'ana Rabbah" equals the
            probe "Hoshana Rabbah"); off reproduces v3.
        :param anchor_strings: Eval strings checked in ANCHORS only (text -> source; the acceptance gate's blocked
            queries): exact (and glued), contains (strings of >= 2 tokens -- a blocked query is never contained in
            an anchor, plan H11 --, prefix-aware like the eval queries), contained_in (>= ``min_contain``) and
            Jaccard; record texts are never tested against them. One-token strings are exact matches only (a blocked
            "Purim" must not remove every anchor naming Purim). A key that is also an ``eval_strings`` / held-out
            key keeps that entry's (record-level) source.
        """
        self.min_contain, self.min_shared, self.he_prefixes = min_contain, min_shared, he_prefixes
        self.patterns = {sid: re.compile(p, re.I) for sid, p in patterns.items()}
        entries: List[Tuple[Tuple[str, ...], str]] = []
        for text, src in eval_strings.items():
            entries.append((tuple(norm_tokens(text)), src))
        for name in held_names:
            entries.append((tuple(norm_tokens(name)), "held_out_name"))
        entries = [(t, s) for t, s in entries if t]
        self.exact = {" ".join(t): s for t, s in entries}
        self.anchor_exact = {" ".join(norm_tokens(x)) for x in anchor_exact} - {""}
        anchor_only = [(t, s) for t, s in ((tuple(norm_tokens(x)), src) for x, src in (anchor_strings or {}).items())
                       if t and " ".join(t) not in self.exact]
        self.anchor_only_exact = {" ".join(t): s for t, s in anchor_only}
        # "eval string inside the text": queries with >= min_contain tokens, held-out names at any length
        self.by_first: Dict[str, List[Tuple[Tuple[str, ...], str, bool]]] = defaultdict(list)
        for toks, src in entries:
            if src == "held_out_name":
                self.by_first[toks[0]].append((toks, src, " ".join(toks) in AMBIGUOUS_HELD_NAMES))
            elif len(toks) >= min_contain:
                self.by_first[toks[0]].append((toks, src, False))
        for text, src in anchor_contain:  # anchor-only entries (third field True = skipped for record texts)
            toks = tuple(norm_tokens(text))
            if 2 <= len(toks) < min_contain:
                self.by_first[toks[0]].append((toks, src, True))
        for toks, src in anchor_only:
            if len(toks) >= 2:
                self.by_first[toks[0]].append((toks, src, True))
        self.glued = {"".join(t): s for t, s in entries} if glued else {}
        self.anchor_only_glued = ({"".join(t): s for t, s in anchor_only if "".join(t) not in self.glued}
                                  if glued else {})
        # "text inside an eval string": every run of >= min_contain tokens of every eval string
        self.ngrams: Dict[Tuple[str, ...], str] = {}
        for toks, src in entries:
            for n in range(min_contain, len(toks) + 1):
                for i in range(len(toks) - n + 1):
                    self.ngrams.setdefault(toks[i:i + n], src)
        self.anchor_only_ngrams: Dict[Tuple[str, ...], str] = {}
        for toks, src in anchor_only:
            for n in range(min_contain, len(toks) + 1):
                for i in range(len(toks) - n + 1):
                    if toks[i:i + n] not in self.ngrams:
                        self.anchor_only_ngrams.setdefault(toks[i:i + n], src)
        self.max_len = max((len(t) for t, _ in entries + anchor_only), default=0)
        # third field: anchor-only entry (skipped for record texts)
        self.q_content: List[Tuple[Set[str], str, bool]] = [(content_tokens(t), s, False) for t, s in entries]
        self.q_content += [(content_tokens(t), s, True) for t, s in anchor_only]
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for i, (ct, _, _) in enumerate(self.q_content):
            for tok in ct:
                self.postings[tok].append(i)
        self.max_content = max((len(ct) for ct, _, _ in self.q_content), default=0)

    def _first_forms(self, tok: str) -> List[str]:
        """The token itself, plus (``he_prefixes``) its forms without one or two leading Hebrew prefix letters.

        :param tok: Normalised token.
        :returns: Candidate first tokens of a contained eval string.
        :rtype: List[str]
        """
        forms = [tok]
        if self.he_prefixes:
            for k in (1, 2):
                if len(tok) - k >= 2 and all(c in HE_PREFIX_LETTERS for c in tok[:k]):
                    forms.append(tok[k:])
        return forms

    def check(self, text: str, anchor: bool, min_shared: Optional[int] = None) -> Optional[str]:
        """Return the first leak reason for a text, or None when clean.

        :param text: Anchor or positive text.
        :param anchor: True for anchors (linked-name exact block; ambiguous held-out names also count).
        :param min_shared: Override of the Jaccard minimum overlap (diagnostics).
        :returns: ``"<test>:<source>"`` (test = exact / exact_glued / concept / contains / contained_in / jaccard) or
            None.
        :rtype: Optional[str]
        """
        toks = norm_tokens(text)
        if not toks:
            return None
        key = " ".join(toks)
        if key in self.exact:
            return f"exact:{self.exact[key]}"
        if self.glued and "".join(toks) in self.glued:
            return f"exact_glued:{self.glued[''.join(toks)]}"
        if anchor and key in self.anchor_exact:
            return "exact:linked_name"
        if anchor and key in self.anchor_only_exact:
            return f"exact:{self.anchor_only_exact[key]}"
        if anchor and self.anchor_only_glued and "".join(toks) in self.anchor_only_glued:
            return f"exact_glued:{self.anchor_only_glued[''.join(toks)]}"
        plain = strip_niqqud(text)
        for sid, rx in self.patterns.items():
            if rx.search(plain):
                return f"concept:{sid}"
        for i, tok in enumerate(toks):
            for first in self._first_forms(tok):
                for seq, src, ambiguous in self.by_first.get(first, ()):
                    if ambiguous and not anchor:
                        continue  # ambiguous held-out names and two-token anchor-only queries: anchors only
                    if first != tok and src == "held_out_name":
                        continue  # prefixed forms only for eval queries, not for short held-out names
                    if tuple(toks[i + 1:i + len(seq)]) == seq[1:]:
                        return f"contains:{src}"
        if self.min_contain <= len(toks) <= self.max_len:
            if tuple(toks) in self.ngrams:
                return f"contained_in:{self.ngrams[tuple(toks)]}"
            if anchor and tuple(toks) in self.anchor_only_ngrams:
                return f"contained_in:{self.anchor_only_ngrams[tuple(toks)]}"
        ct = content_tokens(toks)
        need = self.min_shared if min_shared is None else min_shared
        if ct and len(ct) <= 2 * self.max_content:
            shared: Counter = Counter()
            for tok in ct:
                for qi in self.postings.get(tok, ()):
                    shared[qi] += 1
            for qi, c in shared.items():
                q, src, only_anchor = self.q_content[qi]
                if only_anchor and not anchor:
                    continue
                if c >= need and c / (len(ct) + len(q) - c) >= 0.5:
                    return f"jaccard:{src}"
        return None


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def load_eval(eval_dir: Path) -> dict:
    """Frozen eval files needed for the hold-out rules.

    :param eval_dir: ``AUDIT_ROOT/v3/eval`` (or ``eval_final``).
    :returns: Dict with ``held`` (ids), ``held_subjects``, ``linked``, ``status``, ``patterns``, ``held_names``,
        ``subject_queries``, ``known_item``, ``held_subject_rows``, ``focus`` (final run: trained subjects scored over
        held-out records), ``focus_doc`` (``focus_subjects.json`` or None).
    :rtype: dict
    """
    hs = json.loads((eval_dir / "held_out_subjects.json").read_text())
    sq = json.loads((eval_dir / "subject_queries.json").read_text())["subjects"]
    held = set(json.loads((eval_dir / "held_out_records.json").read_text())["held_out"])
    held_names = [n for s in hs["subjects"] for n in s["names_en"] + s["names_he"]]
    held_names += [q["text"] for v in sq.values() if v["status"] == "held_out" for q in v["queries"]
                   if q["source"] == "name"]
    focus_path = eval_dir / "focus_subjects.json"
    return {"held": held, "held_subjects": {s["id"] for s in hs["subjects"]}, "linked": set(hs["linked"]),
            "held_subject_rows": hs["subjects"], "status": {sid: v["status"] for sid, v in sq.items()},
            "patterns": hs["concept_patterns"], "held_names": sorted(set(held_names)), "subject_queries": sq,
            "known_item": list(iter_jsonl(eval_dir / "known_item.jsonl")),
            "focus": {sid for sid, v in sq.items() if v["status"] == "focus"},
            "focus_doc": json.loads(focus_path.read_text()) if focus_path.exists() else None}


def gate_strings_from_rows(rows: Iterable[dict]) -> Dict[str, str]:
    """Blocked query texts of a built gate dir (``build_gate_v1.py`` ``queries.jsonl`` rows).

    :param rows: Rows ``{qid, text, family, concept, form, blocked}``.
    :returns: ``text -> "gate"`` for every row with ``blocked`` true.
    :rtype: Dict[str, str]
    """
    return {r["text"]: "gate" for r in rows if r.get("blocked") and (r.get("text") or "").strip()}


def gate_strings_from_spec(spec: dict) -> Dict[str, str]:
    """Blocked query texts of the gate spec (``gate_v1_spec.json``), for runs before the gate dir is built.

    :param spec: Parsed spec ``{"concepts": {cid: {"queries": [{text, form, blocked}], ...}}, ...}``.
    :returns: ``text -> "gate"`` for every concept query with ``blocked`` true.
    :rtype: Dict[str, str]
    """
    return {q["text"]: "gate" for c in (spec.get("concepts") or {}).values() for q in c.get("queries") or []
            if q.get("blocked") and (q.get("text") or "").strip()}


def load_gate_strings(gate_dir: Optional[Path], gate_spec: Optional[Path]) -> Dict[str, str]:
    """Blocked acceptance-gate queries (plan C9) from the gate dir and / or the gate spec (union).

    :param gate_dir: ``eval_final/gate_v1`` (reads ``queries.jsonl``), or None.
    :param gate_spec: ``gate_v1_spec.json``, or None.
    :returns: ``text -> "gate"`` (empty when neither is given).
    :rtype: Dict[str, str]
    :raises SystemExit: When a given path does not exist (a silently empty gate would let gate queries into training).
    """
    out: Dict[str, str] = {}
    if gate_dir is not None:
        path = gate_dir / "queries.jsonl"
        if not path.exists():
            raise SystemExit(f"--gate-dir {gate_dir}: no queries.jsonl (build it, or pass --gate-spec instead)")
        out.update(gate_strings_from_rows(iter_jsonl(path)))
    if gate_spec is not None:
        if not gate_spec.exists():
            raise SystemExit(f"--gate-spec {gate_spec} does not exist")
        out.update(gate_strings_from_spec(json.loads(gate_spec.read_text())))
    return out


def build_leak_filter(ev: dict, probe_scope: str, min_contain: int, min_shared: int,
                      focus_names: str = "block", he_prefixes: bool = False,
                      strict: bool = False, gate_strings: Optional[Dict[str, str]] = None) -> Tuple[LeakFilter, Counter]:
    """Leak filter over the eval queries of ``ev``.

    :param ev: :func:`load_eval` output.
    :param probe_scope: ``all`` (every concept_probe_v1 query) or ``v3`` (only the held-out subjects' probe queries).
    :param min_contain: See :class:`LeakFilter`.
    :param min_shared: See :class:`LeakFilter`.
    :param focus_names: ``block`` (focus subjects' name queries are eval strings, kept out of training like the
        probe / llm_t2 queries) or ``label`` (usable as training labels, like seen-subject names). No effect without
        focus subjects.
    :param he_prefixes: Hebrew-prefix-aware containment (see :class:`LeakFilter`).
    :param strict: Final-run tightenings of the filter (``--final-strict``): the focus subjects' two-token probe /
        llm_t2 queries are blocked inside anchors (``anchor_contain``), and exact matches ignore spacing /
        punctuation inside words (``glued``). Off reproduces v3.
    :param gate_strings: Blocked acceptance-gate queries (:func:`load_gate_strings`), checked in anchors only
        (``LeakFilter(anchor_strings=...)``); None / empty = no gate.
    :returns: (filter, count of eval strings per source).
    :rtype: Tuple[LeakFilter, Counter]
    """
    if focus_names not in ("block", "label"):
        raise ValueError(f"focus_names must be 'block' or 'label', not {focus_names!r}")
    strings: Dict[str, str] = {}
    for v in ev["subject_queries"].values():
        if v["status"] == "held_out":
            for q in v["queries"]:
                if q["source"] in ("probe", "llm_t2"):
                    strings.setdefault(q["text"], q["source"])
        elif v["status"] == "focus":  # final run: the focus group's eval queries stay out of training text
            for q in v["queries"]:
                if q["source"] in ("probe", "llm_t2") or (q["source"] == "name" and focus_names == "block"):
                    strings.setdefault(q["text"], "focus_name" if q["source"] == "name" else q["source"])
    if probe_scope == "all":
        for spec in json.loads(PROBE.read_text())["concepts"].values():
            for q in spec["queries"]:
                strings.setdefault(q, "probe")
    for q in ev["known_item"]:
        strings.setdefault(q["text"], "known_item")
    linked_names = [q["text"] for v in ev["subject_queries"].values() if v["status"] == "linked"
                    for q in v["queries"]]
    anchor_contain = [(q["text"], q["source"]) for v in ev["subject_queries"].values() if strict
                      and v["status"] == "focus" for q in v["queries"] if q["source"] in ("probe", "llm_t2")]
    counts = Counter(strings.values())
    counts["held_out_name"] = len(ev["held_names"])
    counts["linked_name"] = len(linked_names)
    if strict:
        counts["focus_two_token_anchor_only"] = sum(2 <= len(norm_tokens(t)) < min_contain for t, _ in anchor_contain)
    if gate_strings:
        counts["gate_anchor_only"] = len(gate_strings)
    return LeakFilter(strings, ev["held_names"], linked_names, ev["patterns"], min_contain, min_shared,
                      he_prefixes, anchor_contain, glued=strict, anchor_strings=gate_strings), counts


def load_semantic(path: Path) -> Tuple[Dict[str, dict], List[str]]:
    """Semantic corpus rows (text kept only for eligible records) and the file order.

    :param path: ``v3/corpus_semantic.jsonl``.
    :returns: (``doc_id -> {eligible, text_hash, text}``, doc ids in file order).
    :rtype: Tuple[Dict[str, dict], List[str]]
    """
    rows, order = {}, []
    for r in iter_jsonl(path):
        order.append(r["doc_id"])
        rows[r["doc_id"]] = {"eligible": bool(r["eligible"]), "text_hash": r["text_hash"],
                             "text": r["text"] if r["eligible"] else ""}
    return rows, order


def load_subject_labels(path: Path) -> SubjectLabels:
    """Every label field of ``subjects_v1.jsonl`` (v10 schema; older files lack the new keys, read as empty).

    Row schema: ``{doc_id, subjects, weak, weak_reason, canon, mentions, gold_override}``.

    :param path: ``subjects_v1.jsonl``.
    :returns: :class:`SubjectLabels`.
    :rtype: SubjectLabels
    """
    labels = SubjectLabels({}, {}, {}, {}, {}, set())
    for r in iter_jsonl(path):
        d = r["doc_id"]
        labels.subjects[d] = set(r["subjects"])
        labels.weak[d] = set(r["weak"])
        if r.get("weak_reason"):
            labels.weak_reason[d] = dict(r["weak_reason"])
        if r.get("canon"):
            labels.canon[d] = set(r["canon"])
        if r.get("mentions"):
            labels.mentions[d] = set(r["mentions"])
        if r.get("gold_override"):
            labels.gold_override.add(d)
    return labels


def load_subjects(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Subject ids (strong + weak) and weak ids per record (the v9 view; see :func:`load_subject_labels`).

    :param path: ``v3/subjects_v1.jsonl``.
    :returns: (``doc_id -> subjects``, ``doc_id -> weak subjects``).
    :rtype: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
    """
    labels = load_subject_labels(path)
    return labels.subjects, labels.weak


def own_shelfmarks(corpus_v1: Path) -> Dict[str, str]:
    """``Shelf Mark:`` value of every record (production text).

    :param corpus_v1: ``corpus_v1.jsonl``.
    :returns: ``doc_id -> shelf mark``.
    :rtype: Dict[str, str]
    """
    out = {}
    for r in iter_jsonl(corpus_v1):
        out[r["doc_id"]] = next((ln[len("Shelf Mark:"):].strip() for ln in r["text"].split("\n")
                                 if ln.startswith("Shelf Mark:")), "")
    return out


def pgp_tags_types(order: List[str], merged: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Raw PGP tags and PGP document types per record, matched by id (``build_eval_v3.merged_pgp_info``: the
    merged file is re-merged in place and no longer in corpus order).

    :param order: Corpus doc ids.
    :param merged: ``merged_shelfmarks.jsonl``.
    :returns: (``doc_id -> lower-cased raw tags``, ``doc_id -> PGP types``), records without PGP data omitted.
    :rtype: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
    """
    info, missing = merged_pgp_info(order, merged)
    print(f"merged PGP info: {len(info) - missing} corpus records matched by id, {missing} absent (no PGP data)",
          flush=True)
    tags, types = {}, {}
    for doc_id in order:
        t, ty = info[doc_id]
        if t:
            tags[doc_id] = {x.lower() for x in t}
        if ty:
            types[doc_id] = set(ty)
    return tags, types


def ref_slugs(frame_refs: Path, curated: Path, wanted: Set[str]) -> Dict[str, Set[str]]:
    """Sefaria topics (any, not only the subject whitelist) whose curated refs overlap a record's frame refs.

    :param frame_refs: ``frame_refs.jsonl``.
    :param curated: ``sefaria/derived/curated_refs.jsonl``.
    :param wanted: Topic slugs to resolve.
    :returns: ``doc_id -> slugs``.
    :rtype: Dict[str, Set[str]]
    """
    by_book: Dict[str, List[Tuple[tuple, tuple, str]]] = defaultdict(list)
    for row in iter_jsonl(frame_refs):
        parsed = parse_ref(row["ref"])
        if parsed:
            by_book[parsed[0]].append((parsed[1], parsed[2], row["doc_id"]))
    out: Dict[str, Set[str]] = defaultdict(set)
    for row in iter_jsonl(curated):
        if row["topic"] not in wanted:
            continue
        parsed = parse_ref(row["ref"])
        if not parsed:
            continue
        book, low, high = parsed
        for dlow, dhigh, doc_id in by_book.get(book, []):
            if low <= dhigh and dlow <= high:
                out[doc_id].add(row["topic"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# subject names
# ---------------------------------------------------------------------------------------------------------------
def _key(text: str) -> str:
    """Comparison key for names (leak-check normalisation).

    :param text: Name.
    :returns: Normalised key.
    :rtype: str
    """
    return " ".join(norm_tokens(text))


def subject_name_table(sid: str, meta: dict, graph: dict) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    """Split a subject's names into parent names and conditional sub-topic names.

    Topic subjects borrow names from several Sefaria slugs (``TOPIC_SEFARIA``): names of the first (parent) slug
    and the hand phrases are parent names; names only found under the other slugs are sub-topics, usable when the
    record's frame refs hit that slug. PGP subjects merge raw tags: a raw tag other than the display name is usable
    only for records carrying that tag.

    :param sid: Subject id.
    :param meta: ``subject_vocab.json`` entry.
    :param graph: Sefaria concept graph.
    :returns: (parent names, ``name -> ("slug"|"tag", value)`` sub-topic conditions).
    :rtype: Tuple[List[str], Dict[str, Tuple[str, str]]]
    """
    kind, slug = sid.split(":", 1)
    names = [clean_name(n) for n in (meta.get("names_en") or []) + (meta.get("names_he") or [])]
    names = list(dict.fromkeys(n for n in names if len(n) >= 2 and not n.startswith("#")))
    sub: Dict[str, Tuple[str, str]] = {}
    if kind == "topic":
        slugs = TOPIC_SEFARIA.get(slug, [])
        parent_slugs = [] if slug in TOPIC_ALL_SUB else slugs[:1]
        parent_keys = {_key(p) for p in TOPIC_PHRASES.get(slug, []) + TOPIC_EXTRA_PHRASES.get(slug, [])}
        en, he = sefaria_names(parent_slugs, graph)
        parent_keys |= {_key(n) for n in en + he}
        for s in slugs:
            if s in parent_slugs:
                continue
            en, he = sefaria_names([s], graph)
            for n in en + he:
                if _key(n) not in parent_keys:
                    sub.setdefault(_key(n), ("slug", s))
    elif kind == "pgp":
        display = _key((meta.get("names_en") or [""])[0])
        for tag in meta.get("raw_tags") or []:
            if _key(tag) != display:
                sub.setdefault(_key(tag), ("tag", tag.lower()))
    parent = [n for n in names if _key(n) not in sub]
    conditional = {n: sub[_key(n)] for n in names if _key(n) in sub}
    return parent, conditional


# ---------------------------------------------------------------------------------------------------------------
# t2t
# ---------------------------------------------------------------------------------------------------------------
def node_names(node: dict) -> Tuple[List[str], List[str]]:
    """Clean English and Hebrew names of a Sefaria node (primary first).

    :param node: Concept-graph node.
    :returns: (English names, Hebrew names).
    :rtype: Tuple[List[str], List[str]]
    """
    raw = [node.get("en") or "", node.get("he") or ""] + list(node.get("titles") or [])
    seen, en, he = set(), [], []
    for t in raw:
        t = clean_name(t)
        k = _key(t)
        if len(t) < 2 or t.startswith("#") or not k or k in seen or not (HEB_LETTER.search(t) or LATIN_LETTER.search(t)):
            continue
        seen.add(k)
        (he if script_of(t) == "he" else en).append(t)
    return en, he


def held_out_seed_nodes(ev: dict, graph: dict) -> Set[str]:
    """Sefaria nodes that *are* a held-out subject: mapped slugs, linked Sefaria subjects, concept-regex matches.

    :param ev: :func:`load_eval` output.
    :param graph: Concept graph.
    :returns: Seed slugs.
    :rtype: Set[str]
    """
    titles = title_index(graph)
    seeds: Set[str] = set()
    for sid in ev["held_subjects"]:
        kind, slug = sid.split(":", 1)
        if kind == "topic":
            seeds |= set(TOPIC_SEFARIA.get(slug, []))
        else:
            mapped = PGP_SUBJECT_SEFARIA.get(slug) or titles.get(slug.replace("-", " "))
            if mapped:
                seeds.add(mapped)
    seeds |= {sid.split(":", 1)[1] for sid in ev["linked"] if sid.startswith("sefaria:")}
    rx = [re.compile(p, re.I) for p in ev["patterns"].values()]
    for slug, node in graph.items():
        label = strip_niqqud(" | ".join([slug, node.get("en") or "", node.get("he") or ""] + list(node.get("titles") or [])))
        if any(r.search(label) for r in rx):
            seeds.add(slug)
    return seeds


def excluded_nodes(seeds: Set[str], graph: dict, hops: int = 2) -> Tuple[Set[str], int]:
    """Nodes within ``hops`` of a seed (undirected; display / gender links skipped; no hop through a category hub).

    :param seeds: Seed slugs.
    :param graph: Concept graph.
    :param hops: Hop radius.
    :returns: (excluded slugs, size of the literal all-links exclusion within the graph, for the report).
    :rtype: Tuple[Set[str], int]
    """
    def adjacency(skip: Set[str]) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for s, node in graph.items():
            for kind, targets in (node.get("links") or {}).items():
                if kind not in skip:
                    for t in targets:
                        adj[s].add(t)
                        adj[t].add(s)
        return adj

    def reach(adj: Dict[str, Set[str]], hub: int) -> Set[str]:
        out, frontier = set(seeds), set(seeds)
        for _ in range(hops):
            nxt = set()
            for s in frontier:
                for t in adj.get(s, ()):
                    if t not in out:
                        out.add(t)
                        if len(adj[t]) <= hub:
                            nxt.add(t)
            frontier = nxt
        return out

    literal = reach(adjacency(set()), 10 ** 9) & set(graph)
    return reach(adjacency(HOP_SKIP_LINKS), HOP_HUB_DEGREE), len(literal)


def t2t_pairs(graph: dict, excluded: Set[str], rng: random.Random) -> Tuple[List[dict], Counter]:
    """Term pairs from the concept graph (before the leak filter).

    :param graph: Concept graph.
    :param excluded: Slugs never used on either side.
    :param rng: RNG.
    :returns: (rows ``{anchor, positive, family, concept_ids}``, stats).
    :rtype: Tuple[List[dict], Counter]
    """
    stats: Counter = Counter()
    fan_in: Counter = Counter()
    for node in graph.values():
        for kind in T2T_LINKS_UP:
            fan_in.update(set((node.get("links") or {}).get(kind) or []))
    coarse = COARSE_PARENTS | {p for p, n in fan_in.items() if n > COARSE_FAN_IN}
    names = {slug: node_names(node) for slug, node in graph.items()}
    out = []

    def pick(slug: str, lang: Optional[str] = None) -> Optional[str]:
        en, he = names[slug]
        if lang == "en" or (lang is None and en and (not he or rng.random() < 0.5)):
            return en[0] if en else None
        return he[0] if he else None

    for slug in sorted(graph):
        if slug in excluded:
            stats["node_excluded_2hop"] += 1
            continue
        en, he = names[slug]
        if not (en or he):
            stats["node_no_names"] += 1
            continue
        ids = [f"sefaria:{slug}"]
        if en and he:
            a, b = (en[0], he[0]) if rng.random() < 0.5 else (he[0], en[0])
            out.append({"anchor": a, "positive": b, "family": "t2t_xling", "concept_ids": ids})
        alts = [(n, "en") for n in en[1:]] + [(n, "he") for n in he[1:]]
        rng.shuffle(alts)
        for alt, lang in alts[:T2T_ALIAS_PER_NODE]:
            target_lang = lang if not (en and he) or rng.random() < 0.5 else ("he" if lang == "en" else "en")
            target = pick(slug, target_lang)
            if target and _key(target) != _key(alt):
                out.append({"anchor": alt, "positive": target, "family": "t2t_alias", "concept_ids": ids})
        parents = []
        for kind in T2T_LINKS_UP:
            for p in (graph[slug].get("links") or {}).get(kind) or []:
                if p not in graph or p in parents:
                    continue
                if p in coarse:
                    stats["parent_coarse"] += 1
                elif p in excluded:
                    stats["parent_excluded_2hop"] += 1
                else:
                    parents.append(p)
        rng.shuffle(parents)
        for p in parents[:T2T_PARENTS_PER_NODE]:
            child, parent = pick(slug), pick(p)
            if child and parent and _key(child) != _key(parent):
                out.append({"anchor": child, "positive": parent, "family": "t2t_parent",
                            "concept_ids": ids + [f"sefaria:{p}"]})
    stats["coarse_parents"] = len(coarse)
    return out, stats


# ---------------------------------------------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------------------------------------------
class MixBuilder:
    """Holds the gated train records and emits family rows through one leak / identity gate."""

    def __init__(self, ev: dict, leak: LeakFilter, rows: Dict[str, dict], subj: Dict[str, Set[str]],
                 weak: Dict[str, Set[str]], marks: Dict[str, str], rng: random.Random,
                 norm_gate: bool = False, festival_guard: bool = True, labels: Optional[SubjectLabels] = None,
                 mention_masks: bool = True, name_form_gate: bool = True, festival_canon_drops: bool = True,
                 name_stoplist: bool = True, sefaria_canon: bool = True, short_anchor_subject: bool = True) -> None:
        """Gate every eligible record (see :meth:`gate_records`).

        The record gate depends only on ``ev`` / ``leak`` / ``rows`` / ``subj`` / ``marks`` / ``norm_gate`` (the
        packager rebuilds the train pool with the first eight arguments); the other flags shape the rows.

        :param ev: :func:`load_eval` output.
        :param leak: Leak filter.
        :param rows: Semantic rows.
        :param subj: ``doc_id -> subjects``.
        :param weak: ``doc_id -> weak subjects``.
        :param marks: ``doc_id -> own shelf mark``.
        :param rng: RNG.
        :param norm_gate: Also exclude records whose text equals a held-out record's up to niqqud / case /
            punctuation / spacing (:func:`norm_key`; final run, ``--final-strict``). Off reproduces v3.
        :param festival_guard: Tag q2d rows (not ``content``) with ``anchor_topic_ids`` (:meth:`anchor_festivals`).
        :param labels: Full labels (:func:`load_subject_labels`: weak reasons, canon, mentions); None = none of
            them (v9 behaviour of every rule that reads them).
        :param mention_masks: Add ``M:<sid>`` mention ids to mask ids, ``anchor_mention_id`` to subject rows and the
            ``M:`` twins to ``anchor_topic_ids`` (plan C3); off also ignores mentions in the festival-carry test.
        :param name_form_gate: Subject family carriers keyed ``(subject, class)`` with class-specific names and
            quotas (plan C2; see the module docstring). Off: strong carriers, every name, one quota (v9).
        :param festival_canon_drops: Bare festival names pair only with strong festival carriers; drop t2t festival
            <-> canon pairs and q2d rows whose bare festival anchor has a canon-only positive (plan A7).
        :param name_stoplist: Never use :data:`build_subjects.NAME_STOPLIST` names as subject anchors (plan C4).
        :param sefaria_canon: Name-form gate: every ``sefaria:`` label is a canon carrier (Sefaria labels come from
            frame refs into canon texts: 1,415 of 1,428 labelled records are canon).
        :param short_anchor_subject: :func:`mask_level_of` rule for short hebrew / translit anchors (plan C6).
        """
        self.ev, self.leak, self.rows, self.subj, self.weak, self.marks, self.rng = ev, leak, rows, subj, weak, marks, rng
        self.norm_gate = norm_gate
        self.festival_guard = festival_guard
        self.labels = labels
        self.mention_masks = mention_masks
        self.name_form_gate = name_form_gate
        self.festival_canon_drops = festival_canon_drops
        self.name_stoplist = name_stoplist
        self.sefaria_canon = sefaria_canon
        self.short_anchor_subject = short_anchor_subject
        # the festival-carry test sees mentions only when mention masks are on (ablation = as if no mentions)
        self.carry_labels = labels if labels is None or mention_masks else labels._replace(mentions={})
        self.blocked_subjects = ev["held_subjects"] | ev["linked"]
        self.out: List[dict] = []
        self.drop: Counter = Counter()
        self.detail: Counter = Counter()
        self.guard: Counter = Counter()  # festival-guard decisions (pre-dedupe)
        self.examples: Dict[str, List[list]] = defaultdict(list)  # removal examples for the stats (first 20 each)
        self._anchor_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self.train: Dict[str, str] = {}
        self.rep: Dict[str, str] = {}
        self.gate_stats = self.gate_records()

    def gate_records(self) -> Counter:
        """Fill ``self.train`` (doc_id -> masked semantic text) with eligible records that may be trained on.

        Excluded: held-out records, records carrying a held-out / linked subject, text identical to a held-out
        record (with ``norm_gate``: also identical up to niqqud / case / punctuation / spacing), and texts failing
        the leak filter (concept regex, held-out name, contained eval query).

        :returns: Gate counts.
        :rtype: Counter
        """
        ev, stats = self.ev, Counter()
        held_hashes = {r["text_hash"] for d, r in self.rows.items() if d in ev["held"] and r["eligible"]}
        self.held_hashes = held_hashes
        self.held_norms = held_norm_keys(self.rows, ev["held"]) if self.norm_gate else set()
        blocked = ev["held_subjects"] | ev["linked"]
        for d, r in self.rows.items():
            if not r["eligible"]:
                continue
            stats["eligible"] += 1
            if d in ev["held"]:
                stats["held_out_record"] += 1
                continue
            if self.subj.get(d, set()) & blocked:
                stats["held_or_linked_subject"] += 1
                continue
            if r["text_hash"] in held_hashes:
                stats["same_text_as_held_out"] += 1
                continue
            if self.norm_gate and norm_key(r["text"]) in self.held_norms:
                stats["norm_text_as_held_out"] += 1
                continue
            why = self.leak.check(r["text"], anchor=False)
            if why:
                stats["text_leak:" + why.split(":")[0]] += 1
                self.detail["record_" + why] += 1
                continue
            text, n = mask_ids(r["text"], self.marks.get(d, ""))
            stats["masked_records"] += bool(n)
            stats["masked_spans"] += n
            self.train[d] = text
        stats["train_records"] = len(self.train)
        for d in sorted(self.train):
            self.rep.setdefault(self.rows[d]["text_hash"], d)
        stats["train_text_groups"] = len(self.rep)
        return stats

    def is_rep(self, doc_id: str) -> bool:
        """Whether a train record represents its identical-text group (smallest doc id).

        :param doc_id: Record id.
        :returns: True for the representative.
        :rtype: bool
        """
        return self.rep.get(self.rows[doc_id]["text_hash"]) == doc_id

    def anchor_reject(self, anchor: str, doc_id: Optional[str] = None) -> Optional[str]:
        """Why an anchor (or short positive term) cannot be used, or None.

        :param anchor: Cleaned anchor.
        :param doc_id: Positive record (for its own shelf mark), if any.
        :returns: Reason or None.
        :rtype: Optional[str]
        """
        if len(anchor) < 2 or not (HEB_LETTER.search(anchor) or LATIN_LETTER.search(anchor)):
            return "anchor_empty"
        if doc_id and own_mark_in(anchor, self.marks.get(doc_id, "")):
            return "anchor_own_shelfmark"
        key = (anchor, "a")
        if key not in self._anchor_cache:
            if id_hits(anchor) or QUERY_SHELFMARK_RE.search(anchor):
                self._anchor_cache[key] = "anchor_shelfmark"
            else:
                why = self.leak.check(anchor, anchor=True)
                self._anchor_cache[key] = f"leak_{why}" if why else None
        return self._anchor_cache[key]

    def mention_ids(self, doc_id: str) -> List[str]:
        """``M:<sid>`` ids of the subjects a record's catalogue text names without carrying them.

        Held-out / linked subjects are left out (such records rarely pass the leak filter anyway).

        :param doc_id: Record id.
        :returns: Sorted ids (empty with ``mention_masks`` off or without labels).
        :rtype: List[str]
        """
        if not (self.mention_masks and self.labels):
            return []
        return sorted(MENTION_PREFIX + s for s in self.labels.mentions.get(doc_id, set())
                      if s not in self.blocked_subjects)

    def subject_ids(self, doc_id: str) -> List[str]:
        """A record's subject ids (strong and weak) plus its ``M:`` mention ids (the train pool's ``subjects``).

        :param doc_id: Record id.
        :returns: Sorted ids.
        :rtype: List[str]
        """
        return sorted(self.subj.get(doc_id, set()) | set(self.mention_ids(doc_id)))

    def record_ids(self, doc_id: str) -> List[str]:
        """Mask ids of a record: all its subject ids, its ``M:`` mention ids, plus ``D:<doc_id>``.

        :param doc_id: Record id.
        :returns: Sorted ids, ``D:`` last.
        :rtype: List[str]
        """
        return self.subject_ids(doc_id) + [f"D:{doc_id}"]

    def anchor_festivals(self, anchor: str, doc_id: Optional[str] = None) -> List[str]:
        """Trainable festival topic ids (and their ``M:`` twins) a query anchor names, for ``anchor_topic_ids``.

        A row ("Passover Haggadah" -> one Haggadah) otherwise masks only its own record (doc level) or its
        positive's subjects (subject level), so every other Passover record could be its mined or in-batch negative;
        the trainer adds these ids to the row's relevance. Nothing is returned for a canon / folio anchor
        (:func:`canon_reference`: "Talmud Pesahim", "Sukkah 45a" -- known-item queries whose within-work negatives
        must stay), and a festival is left out when the positive carries it only as canon or not at all
        (:func:`festival_carry`; a mention counts as carrying when mention masks are on).

        :param anchor: Cleaned anchor.
        :param doc_id: Positive record (None: no positive test).
        :returns: Sorted ``topic:<festival>`` ids whose subject is trainable (never held-out ids), each with its
            ``M:topic:<festival>`` twin when mention masks are on.
        :rtype: List[str]
        """
        found = festival_mentions(anchor)
        if not found:
            return []
        if FOLIO_RE.search(anchor):
            self.guard["skip_folio_anchor"] += 1
            return []
        if canon_reference(anchor):
            self.guard["skip_canon_anchor"] += 1
            return []
        status, out = self.ev["status"], []
        for t in sorted(found):
            sid = f"topic:{t}"
            if status.get(sid) not in TRAINABLE_STATUSES:
                self.guard[f"skip_not_trainable:{t}"] += 1
                continue
            if doc_id is not None:
                carry = festival_carry(self.carry_labels, doc_id, sid)
                if carry in ("canon", "none"):
                    self.guard[f"skip_positive_{carry}:{t}"] += 1
                    continue
            out.append(sid)
            if self.mention_masks:
                out.append(MENTION_PREFIX + sid)
        return sorted(out)

    def canon_carrier_ids(self, doc_id: str) -> Set[str]:
        """Subjects a record carries as canon for the name-form gate (``topic:`` / ``sefaria:`` only).

        :param doc_id: Record id.
        :returns: The record's ``canon`` list plus weak ids whose reason is canon, plus every ``sefaria:`` subject
            (``sefaria_canon``), restricted to :data:`NAME_FORM_KINDS` ids the record carries.
        :rtype: Set[str]
        """
        subj = self.subj.get(doc_id, set())
        canon: Set[str] = set()
        if self.labels:
            canon |= self.labels.canon.get(doc_id, set())
            canon |= {s for s, why in self.labels.weak_reason.get(doc_id, {}).items() if why == "canon"}
        if self.sefaria_canon:
            canon |= {s for s in subj if s.startswith("sefaria:")}
        return {s for s in canon & subj if s.startswith(NAME_FORM_KINDS)}

    def carrier_classes(self, doc_id: str) -> List[Tuple[str, str]]:
        """``(subject, class)`` carrier keys of a record for the subject family.

        :param doc_id: Record id.
        :returns: Name-form gate on: canon ids -> ``canon`` (weak festival canon included), other strong
            ``topic:`` / ``sefaria:`` ids -> ``content``, other strong ids -> ``all``; weak incidental / title ids
            are never carriers. Gate off: every strong id -> ``all`` (v9).
        :rtype: List[Tuple[str, str]]
        """
        strong = self.subj.get(doc_id, set()) - self.weak.get(doc_id, set())
        if not self.name_form_gate:
            return [(s, "all") for s in sorted(strong)]
        canon = self.canon_carrier_ids(doc_id)
        out = []
        for s in sorted(strong | canon):
            out.append((s, "canon" if s in canon else "content" if s.startswith(NAME_FORM_KINDS) else "all"))
        return out

    def strongly_carries_all(self, doc_id: str, festivals: Iterable[str]) -> bool:
        """Whether a record is a strong ``topic:<festival>`` carrier of every given festival.

        :param doc_id: Record id.
        :param festivals: Festival names.
        :returns: True when all are strong labels of the record.
        :rtype: bool
        """
        strong = self.subj.get(doc_id, set()) - self.weak.get(doc_id, set())
        return all(f"topic:{f}" in strong for f in festivals)

    def emit(self, anchor: Optional[str], family: str, doc_id: Optional[str] = None, positive: Optional[str] = None,
             anchor_doc_id: Optional[str] = None, **extra) -> bool:
        """Append one row after the anchor gate (record texts were gated in :meth:`gate_records`).

        :param anchor: Anchor text (cleaned here); ignored when ``anchor_doc_id`` is given.
        :param family: Family name.
        :param doc_id: Positive record (None for t2t).
        :param positive: Positive text (default: the record's masked semantic text).
        :param anchor_doc_id: Anchor record (d2d): the anchor is its masked semantic text, line structure kept.
        :param extra: Extra row fields.
        :returns: True when the row was kept.
        :rtype: bool
        """
        if anchor_doc_id:
            anchor = self.train[anchor_doc_id]
        else:
            anchor = clean_anchor(anchor or "")
            why = self.anchor_reject(anchor, doc_id)
            if why:
                self.drop[f"{family}:{why.split(':')[0]}"] += 1
                self.detail[f"{family}:{why}"] += 1
                return False
        kind = KIND.get(family, "q2d")
        if doc_id is None:
            positive = clean_anchor(positive or "")
            pwhy = self.anchor_reject(positive)  # t2t positives are short terms: same gate as anchors
            if pwhy:
                self.drop[f"{family}:positive_{pwhy.split(':')[0]}"] += 1
                self.detail[f"{family}:positive_{pwhy}"] += 1
                return False
            if self.festival_canon_drops and kind == "t2t" and festival_canon_pair(anchor, positive):
                self.drop[f"{family}:festival_canon_pair"] += 1
                self._example("t2t_festival_canon_pair", [anchor, positive])
                return False
        elif positive is None:
            positive = self.train[doc_id]
        if self.festival_canon_drops and kind == "q2d" and family != "content" and doc_id:
            canon_only = canon_only_festivals(anchor, doc_id, self.labels)
            if canon_only:
                self.drop[f"{family}:bare_festival_canon_positive"] += 1
                self._example(f"bare_festival_canon_positive:{family}", [anchor, doc_id, canon_only])
                return False
        row = {"anchor": anchor, "positive": positive, "family": family, "kind": kind,
               "mask_level": mask_level_of(family, anchor, self.short_anchor_subject), "doc_id": doc_id,
               "mask_ids": self.record_ids(doc_id) if doc_id else [],
               "anchor_mask_ids": [], "anchor_lang": script_of(anchor)}
        if doc_id:
            row["text_hash"] = self.rows[doc_id]["text_hash"]
        if anchor_doc_id:
            row["anchor_doc_id"] = anchor_doc_id
            row["anchor_mask_ids"] = self.record_ids(anchor_doc_id)
        if self.festival_guard and kind == "q2d" and family != "content" and doc_id:
            topics = self.anchor_festivals(anchor, doc_id)
            if topics:
                row["anchor_topic_ids"] = topics
        if self.mention_masks and extra.get("anchor_subject"):
            row["anchor_mention_id"] = MENTION_PREFIX + extra["anchor_subject"]
        row.update(extra)
        self.out.append(row)
        return True

    def _example(self, key: str, item: list, limit: int = 20) -> None:
        """Keep the first ``limit`` examples of a removal for the stats file.

        :param key: Removal name.
        :param item: Example (JSON-serialisable list).
        :param limit: Examples kept per key.
        """
        if len(self.examples[key]) < limit:
            self.examples[key].append(item)

    # ---- families ------------------------------------------------------------------------------------------
    def synthetic(self, rounds: List[str]) -> Counter:
        """``synthetic_<type>``: LLM queries of train-pool records -> the record.

        :param rounds: Synthetic query rounds.
        :returns: Source counts.
        :rtype: Counter
        """
        stats: Counter = Counter()
        pools = synthetic_pools(rounds)
        for rnd in rounds:
            base = AUDIT_ROOT / "synthetic_queries" / rnd
            for sub in ("out", "out_he"):
                for f in sorted(glob.glob(str(base / sub / "*.jsonl"))):
                    for o in iter_jsonl(Path(f)):
                        qs = [q for q in o.get("queries") or [] if (q.get("text") or "").strip()]
                        d = o["doc_id"]
                        stats["queries"] += len(qs)
                        if pools.get(d) != "train":
                            stats["skip_eval_pool"] += len(qs)
                            continue
                        if d not in self.train:
                            stats["skip_record_not_trainable"] += len(qs)
                            continue
                        for q in qs:
                            typ = q.get("type")
                            if typ not in SYNTHETIC_TYPES:
                                stats["skip_unknown_type"] += 1
                                continue
                            self.emit(q["text"], f"synthetic_{typ}", d, grade=q.get("grade"), round=rnd)
        return stats

    def subject(self, vocab: dict, graph: dict, refs: Dict[str, Set[str]], tags: Dict[str, Set[str]],
                k: float, k_by_kind: Optional[Dict[str, float]] = None) -> Counter:
        """``subject``: seen (and focus) subject names -> train records (sqrt quotas, <= 2 per record, sub-topics).

        Name filters, in order (each counted in the stats when it leaves a record without a name): leak / shelf-mark
        blocked names (``no_allowed_name``), :data:`build_subjects.NAME_STOPLIST` (``no_name_left:stoplist``), the
        name-form gate (canon carriers: law names only, content carriers: no law names; ``no_name_left:class_<cls>``),
        the bare-festival rule (a bare festival name needs a strong carrier of every festival it names;
        ``no_name_left:bare_festival``), then the literal-name rule (``all_names_literal:<kind>``).

        :param vocab: ``subject_vocab.json``.
        :param graph: Concept graph.
        :param refs: ``doc_id -> Sefaria slugs`` from frame refs (sub-topic gating for topics).
        :param tags: ``doc_id -> raw PGP tags`` (sub-topic gating for PGP subjects).
        :param k: Quota factor: carrier key ``(s, cls)`` gets ``min(n, ceil(k * sqrt(n)))`` pairs.
        :param k_by_kind: Quota factor per subject kind overriding ``k`` (``{"domain": 2.5}``: the plan C7 fallback
            mix with the KTIV domain quota halved); None = ``k`` for every kind.
        :returns: Stats.
        :rtype: Counter
        """
        stats: Counter = Counter()
        status = self.ev["status"]
        carriers: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for d in sorted(self.train):
            if not self.is_rep(d):
                continue
            for s, cls in self.carrier_classes(d):
                if status.get(s) in TRAINABLE_STATUSES and s in vocab:
                    carriers[(s, cls)].append(d)
        tables = {s: subject_name_table(s, vocab[s], graph) for s in sorted({s for s, _ in carriers})}
        blocked_names: Dict[str, Set[str]] = defaultdict(set)
        for s, (parent, cond) in tables.items():
            for n in parent + list(cond):
                why = self.anchor_reject(n)
                if why:
                    blocked_names[s].add(n)
                    self.detail[f"subject_name_blocked:{why}"] += 1
        stats["names_blocked"] = sum(len(v) for v in blocked_names.values())
        stats["subjects_all_names_blocked"] = sum(
            1 for s, (p, c) in tables.items() if set(p) | set(c) <= blocked_names[s])
        if self.name_stoplist:
            stop = {s: {n for n in p + list(c) if is_stoplisted(n)} for s, (p, c) in tables.items()}
            stats["names_stoplisted"] = sum(len(v) for v in stop.values())
            self.examples["subject_names_stoplisted"] = sorted({n for v in stop.values() for n in v})
        else:
            stop = {}
        law = {n: is_law_name(n) for p, c in tables.values() for n in p + list(c)}
        bare = {n: bare_festivals(n) for p, c in tables.values() for n in p + list(c)}
        for (s, cls), recs in carriers.items():
            stats[f"carriers_by_class:{cls}"] += len(recs)
        stats["carrier_keys"] = len(carriers)
        used: Counter = Counter()
        token_cache: Dict[str, Tuple[List[str], Set[str]]] = {}
        per_subject: Counter = Counter()
        per_key: Counter = Counter()
        for key in sorted(carriers, key=lambda x: (len(carriers[x]), x)):
            s, cls = key
            recs = list(carriers[key])
            self.rng.shuffle(recs)
            k_key = (k_by_kind or {}).get(s.split(":")[0], k)
            quota = min(len(recs), math.ceil(k_key * math.sqrt(len(recs))))
            parent, cond = tables[s]
            for d in recs:
                if per_key[key] >= quota:
                    break
                if used[d] >= MAX_SUBJECT_PAIRS_PER_RECORD:
                    stats["record_cap"] += 1
                    continue
                allowed = [n for n in parent if n not in blocked_names[s]]
                for n, (how, val) in cond.items():
                    if n in blocked_names[s]:
                        continue
                    ok = (val in refs.get(d, set()) or f"sefaria:{val}" in self.subj.get(d, set())) if how == "slug" \
                        else val in tags.get(d, set())
                    if ok:
                        allowed.append(n)
                        stats["sub_topic_name_allowed"] += 1
                if not allowed:
                    stats["no_allowed_name"] += 1
                    continue
                allowed = self._filter_names(allowed, s, cls, d, stop.get(s, set()), law, bare, stats)
                if not allowed:
                    continue
                if d not in token_cache:
                    toks = norm_tokens(self.train[d])
                    token_cache[d] = (toks, set(toks))
                toks, tokset = token_cache[d]
                fresh = []
                for n in allowed:
                    seq = norm_tokens(n)
                    if not (seq and set(seq) <= tokset and contains_seq(toks, seq)):
                        fresh.append(n)
                if not fresh:
                    stats[f"all_names_literal:{s.split(':')[0]}"] += 1
                    continue
                he = [n for n in fresh if script_of(n) == "he"]
                la = [n for n in fresh if script_of(n) != "he"]
                pool = he if he and (not la or self.rng.random() < 0.5) else la
                name = self.rng.choice(pool)
                stats["sub_topic_name_used"] += name in cond
                if self.emit(name, "subject", d, anchor_subject=s, subject_kind=s.split(":")[0]):
                    used[d] += 1
                    per_subject[s] += 1
                    per_key[key] += 1
        stats["subjects_with_carriers"] = len(tables)
        stats["subjects_with_pairs"] = sum(1 for s in tables if per_subject[s])
        for s, n in per_subject.items():
            stats[f"pairs_by_kind:{s.split(':')[0]}"] += n
        for (s, cls), n in per_key.items():
            stats[f"pairs_by_class:{cls}"] += n
            stats[f"pairs_by_kind_class:{s.split(':')[0]}:{cls}"] += n
        self.subject_top = per_subject.most_common(25)
        return stats

    def _filter_names(self, allowed: List[str], s: str, cls: str, d: str, stop: Set[str], law: Dict[str, bool],
                      bare: Dict[str, Set[str]], stats: Counter) -> List[str]:
        """Apply the stoplist, the name-form gate and the bare-festival rule to a carrier's usable names.

        :param allowed: Names left after the leak / sub-topic gates.
        :param s: Subject id.
        :param cls: Carrier class (``canon`` / ``content`` / ``all``).
        :param d: Carrier record.
        :param stop: Stoplisted names of ``s``.
        :param law: ``name -> is_law_name``.
        :param bare: ``name -> bare festivals`` (:func:`bare_festivals`).
        :param stats: Subject-family stats (updated).
        :returns: Remaining names (empty = no pair for this carrier key; the step that emptied it is counted).
        :rtype: List[str]
        """
        if stop:
            allowed = [n for n in allowed if n not in stop]
            if not allowed:
                stats["no_name_left:stoplist"] += 1
                return []
        if cls == "canon":
            allowed = [n for n in allowed if law[n]]
        elif cls == "content":
            allowed = [n for n in allowed if not law[n]]
        if not allowed:
            stats[f"no_name_left:class_{cls}"] += 1
            return []
        if self.festival_canon_drops:
            kept = [n for n in allowed if not bare[n] or self.strongly_carries_all(d, bare[n])]
            stats[f"bare_festival_names_skipped:{s.split(':')[0]}"] += len(allowed) - len(kept)
            if not kept:
                stats["no_name_left:bare_festival"] += 1
                stats[f"no_name_left:bare_festival:{s.split(':')[0]}"] += 1
                self._example("subject_bare_festival_no_name_left", [s, cls, d, allowed[:3]])
                return []
            allowed = kept
        return allowed

    def content(self, boiler: Set[str], spans_per_record: int, cap: int) -> Counter:
        """``content``: Hebrew transcription span -> the record's text without transcription / translation.

        :param boiler: Boilerplate labels (``corpus_semantic.stats.json``).
        :param spans_per_record: Spans sampled per record.
        :param cap: Max pairs (0 = no cap).
        :returns: Stats.
        :rtype: Counter
        """
        stats: Counter = Counter()
        n = 0
        for d in sorted(self.train):
            if not self.is_rep(d):
                continue
            label, trans = split_text(self.train[d])
            if not trans:
                continue
            stats["records_with_transcription"] += 1
            if len(label) < MIN_CHARS or not content_lines(label, boiler):
                stats["label_boilerplate"] += 1
                continue
            if hashlib.md5(label.encode("utf-8")).hexdigest() in self.held_hashes:
                # the label part restates a held-out record's whole text (same catalogue entry without a transcription)
                stats["label_equals_held_out_text"] += 1
                continue
            if self.norm_gate and norm_key(label) in self.held_norms:
                stats["label_norm_equals_held_out_text"] += 1
                continue
            if self.leak.check(label, anchor=False):
                # the short label part ("Fragment of a marriage contract.") can match an eval query the full text did not
                stats["label_leak"] += 1
                continue
            spans = list(dict.fromkeys(content_spans(trans, self.rng, spans_per_record)))
            if not spans:
                stats["transcription_too_short"] += 1
                continue
            ltoks = norm_tokens(label)
            grams = {tuple(ltoks[i:i + 4]) for i in range(len(ltoks) - 3)}
            for sp in spans:
                st = norm_tokens(sp)
                if any(tuple(st[i:i + 4]) in grams for i in range(len(st) - 3)):
                    stats["span_copied_in_label"] += 1
                    continue
                if cap and n >= cap:
                    stats["cap"] += 1
                    continue
                n += self.emit(sp, "content", d, positive=label)
        return stats

    def d2d(self, types: Dict[str, Set[str]], cap: int) -> Counter:
        """``d2d_cross_genre``: two train records sharing a topic / PGP / Sefaria subject, different works and genres.

        :param types: ``doc_id -> PGP document types``.
        :param cap: Target number of pairs.
        :returns: Stats.
        :rtype: Counter
        """
        stats: Counter = Counter()
        status = self.ev["status"]

        def genre(d: str) -> Set[str]:
            return {s for s in self.subj.get(d, set()) if s.startswith(("work:", "domain:"))} | \
                   {f"pgptype:{t}" for t in types.get(d, set())}

        carriers: Dict[str, List[str]] = defaultdict(list)
        for d in sorted(self.train):
            if self.is_rep(d) and genre(d):
                for s in self.subj.get(d, set()) - self.weak.get(d, set()):
                    if s.startswith(SUBJECT_KINDS_D2D) and status.get(s) in TRAINABLE_STATUSES:
                        carriers[s].append(d)
        carriers = {s: v for s, v in carriers.items() if len(v) >= 2}
        weight = {s: math.sqrt(len(v)) for s, v in carriers.items()}
        used: Counter = Counter()
        seen_pairs: Set[Tuple[str, str]] = set()
        toks: Dict[str, Set[str]] = {}
        per_subject: Counter = Counter()
        exhausted: Set[str] = set()
        n = 0
        # pass 1 spreads the cap over subjects by sqrt(size); later passes hand the unused part to subjects that
        # still produce pairs (small subjects run out of cross-genre record pairs quickly)
        for _ in range(D2D_PASSES):
            active = [s for s in sorted(carriers, key=lambda x: (len(carriers[x]), x)) if s not in exhausted]
            scale = (cap - n) / max(1e-9, sum(weight[s] for s in active))
            progress = 0
            for s in active:
                recs = carriers[s]
                quota, got = max(1, round(scale * weight[s])), 0
                for _ in range(30 * quota):
                    if got >= quota or n >= cap:
                        break
                    a, b = self.rng.sample(recs, 2)
                    if used[a] >= D2D_MAX_USES or used[b] >= D2D_MAX_USES or (min(a, b), max(a, b)) in seen_pairs:
                        stats["reject_used"] += 1
                        continue
                    if genre(a) & genre(b):
                        stats["reject_same_genre"] += 1
                        continue
                    ta = toks.setdefault(a, content_tokens(norm_tokens(self.train[a])))
                    tb = toks.setdefault(b, content_tokens(norm_tokens(self.train[b])))
                    if not ta or not tb or len(ta & tb) / len(ta | tb) >= D2D_MAX_JACCARD:
                        stats["reject_jaccard"] += 1
                        continue
                    if self.emit(None, "d2d_cross_genre", b, anchor_doc_id=a, shared_subject=s):
                        seen_pairs.add((min(a, b), max(a, b)))
                        used[a] += 1
                        used[b] += 1
                        got += 1
                        n += 1
                per_subject[s] += got
                progress += got
                if got < quota:
                    exhausted.add(s)
            if not progress or n >= cap:
                break
        stats["subjects_with_carriers"] = len(carriers)
        stats["subjects_with_pairs"] = sum(1 for v in per_subject.values() if v)
        stats["max_pairs_one_subject"] = max(per_subject.values(), default=0)
        return stats


# ---------------------------------------------------------------------------------------------------------------
# verification + stats
# ---------------------------------------------------------------------------------------------------------------
def verify(rows: Iterable[dict], ev: dict, sem_rows: Dict[str, dict], leak: LeakFilter,
           norm_gate: bool = False, labels: Optional[SubjectLabels] = None, festival_canon: bool = True,
           short_anchor_subject: bool = True) -> Counter:
    """Scan mixture rows; every counter except the ``info_`` ones must be zero.

    Besides the leak / identity checks: no held-out or linked subject id appears in ``mask_ids``,
    ``anchor_mask_ids``, ``anchor_topic_ids`` or ``anchor_mention_id`` (``M:`` ids are checked by their subject);
    ``anchor_topic_ids`` only holds ``topic:`` / ``M:topic:`` ids and never sits on a ``content`` / t2t / d2d row;
    ``anchor_mention_id`` is ``"M:" + anchor_subject``; ``mask_level`` follows :func:`mask_level_of`; and (with
    ``labels``) no q2d row except ``content`` has a bare festival anchor whose positive carries that festival only as
    canon (:func:`canon_only_festivals`, plan A7; an ``info_`` count when ``festival_canon`` is off).

    :param rows: Mixture rows.
    :param ev: :func:`load_eval` output (held-out ids, subjects, linked subjects).
    :param sem_rows: Semantic rows (:func:`load_semantic`): eligibility and text hashes.
    :param leak: Leak filter.
    :param norm_gate: Also count texts / records equal to a held-out record's text up to niqqud / case /
        punctuation / spacing (:func:`norm_key`) as violations (final run, ``--final-strict``).
    :param labels: Subject labels for the bare-festival / canon-positive check (None: check skipped).
    :param festival_canon: The mixture was built with the A7 drops (``--no-festival-canon-drops`` off): the check
        is a violation; otherwise informational.
    :param short_anchor_subject: The mixture was built with the C6 mask-level rule (``--legacy-mask-level`` off).
    :returns: Counts.
    :rtype: Counter
    """
    held_hashes = {r["text_hash"] for d, r in sem_rows.items() if d in ev["held"] and r["eligible"]}
    held_norms = held_norm_keys(sem_rows, ev["held"]) if norm_gate else set()
    norm_of: Dict[str, bool] = {}  # doc id -> its text's key is a held-out key
    blocked = ev["held_subjects"] | ev["linked"]
    out: Counter = Counter()
    checked: Set[str] = set()
    for r in rows:
        out["info_rows"] += 1
        for side in ("anchor", "positive"):
            text = r[side]
            if not text.strip():
                out[f"empty_{side}"] += 1
            for p in ID_LINES:
                out[f"{side}_contains_{p.rstrip(':').replace(' ', '_')}"] += p in text
            out[f"{side}_prod_shelfmark_regex"] += bool(PROD_SHELFMARK_RE.search(text))
            out[f"{side}_extra_id_regex"] += bool(EXTRA_ID_RE.search(text))
            is_record_text = (side == "positive" and r["doc_id"]) or (side == "anchor" and r.get("anchor_doc_id"))
            if is_record_text:
                out[f"info_{side}_bare_collection_token"] += bool(QUERY_SHELFMARK_RE.search(text))
                key = f"{side}:{r['doc_id'] if side == 'positive' else r['anchor_doc_id']}:{r['family'] == 'content'}"
                if key in checked:
                    continue
                checked.add(key)
            why = leak.check(text, anchor=(side == "anchor" and not is_record_text))
            out[f"{side}_leak"] += bool(why)
            if why:
                out[f"leak_detail:{side}:{why}"] += 1
                if out[f"{side}_leak"] <= 5:  # name the offending rows: the file is not written when this fires
                    print(f"verify: {side} leak ({why}) family={r['family']} doc_id={r.get('doc_id')} "
                          f"anchor={r['anchor'][:80]!r} text={text[:160]!r}", file=sys.stderr)
        for side in ("anchor", "positive"):
            out[f"{side}_equals_held_out_text"] += hashlib.md5(r[side].encode("utf-8")).hexdigest() in held_hashes
            if norm_gate:
                out[f"{side}_norm_equals_held_out_text"] += norm_key(r[side]) in held_norms
        out["anchor_bare_collection_token"] += bool(QUERY_SHELFMARK_RE.search(r["anchor"])) and not r.get("anchor_doc_id")
        for key in ("doc_id", "anchor_doc_id"):
            d = r.get(key)
            if d:
                out[f"{key}_held_out"] += d in ev["held"]
                out[f"{key}_unknown"] += d not in sem_rows
                out[f"{key}_same_text_as_held_out"] += sem_rows.get(d, {}).get("text_hash") in held_hashes
                out[f"{key}_not_eligible"] += not sem_rows.get(d, {}).get("eligible")
                if norm_gate:
                    if d not in norm_of:
                        norm_of[d] = norm_key(sem_rows.get(d, {}).get("text", "")) in held_norms
                    out[f"{key}_norm_text_as_held_out"] += norm_of[d]
        mask = set(r["mask_ids"]) | set(r["anchor_mask_ids"])
        topic_ids = set(r.get("anchor_topic_ids") or [])
        mention_id = r.get("anchor_mention_id")
        out["held_or_linked_subject_in_mask_ids"] += bool({_unmention(x) for x in mask} & blocked)
        out["held_or_linked_subject_in_anchor_topic_ids"] += bool({_unmention(x) for x in topic_ids} & blocked)
        out["held_or_linked_subject_in_anchor_mention_id"] += bool(mention_id) and _unmention(mention_id) in blocked
        out["bad_anchor_topic_id"] += any(not _unmention(x).startswith("topic:") for x in topic_ids)
        out["anchor_topic_ids_on_content_t2t_d2d"] += bool(topic_ids) and (r["family"] == "content"
                                                                           or r["kind"] != "q2d")
        out["bad_anchor_mention_id"] += bool(mention_id) and mention_id != MENTION_PREFIX + str(r.get("anchor_subject"))
        out["info_rows_anchor_topic_ids"] += bool(topic_ids)
        out[f"info_rows_anchor_topic_ids:{r['mask_level']}"] += bool(topic_ids)
        out["info_rows_anchor_mention_id"] += bool(mention_id)
        out["info_rows_mention_mask_ids"] += any(x.startswith(MENTION_PREFIX) for x in r["mask_ids"])
        if labels is not None and r["kind"] == "q2d" and r["family"] != "content" and r["doc_id"]:
            bad = bool(canon_only_festivals(r["anchor"], r["doc_id"], labels))
            out["bare_festival_anchor_canon_positive" if festival_canon
                else "info_bare_festival_anchor_canon_positive"] += bad
        out["anchor_equals_positive"] += r["anchor"] == r["positive"]
        out["bad_mask_level"] += r["mask_level"] != mask_level_of(r["family"], r["anchor"], short_anchor_subject)
        out["t2t_with_mask_ids"] += r["kind"] == "t2t" and bool(r["mask_ids"])
    return out


def _unmention(sid: str) -> str:
    """Subject id behind an ``M:<sid>`` mention id (other ids unchanged).

    :param sid: Mask id.
    :returns: The subject id.
    :rtype: str
    """
    return sid[len(MENTION_PREFIX):] if sid.startswith(MENTION_PREFIX) else sid


def mask_overlap(rows: List[dict], rng: random.Random, n: int = 20000) -> dict:
    """Share of random row pairs the trainer would mask (diagnostic for batch construction).

    :param rows: Final rows.
    :param rng: RNG.
    :param n: Sampled pairs.
    :returns: Overlap shares: any subject id, subject ids without ``domain:``, same ``D:`` id (``M:`` mention ids
        left out of these, as before), plus ``anchor_side_overlap``: row a's anchor-side ids (``anchor_topic_ids`` /
        ``anchor_mention_id``) hit row b's positive ids.
    :rtype: dict
    """
    rec = [r for r in rows if r["mask_ids"]]
    if len(rec) < 2:
        return {}
    c: Counter = Counter()
    for _ in range(n):
        a, b = rng.sample(rec, 2)
        sa = {x for x in a["mask_ids"] + a["anchor_mask_ids"] if not x.startswith(("D:", MENTION_PREFIX))}
        sb = {x for x in b["mask_ids"] + b["anchor_mask_ids"] if not x.startswith(("D:", MENTION_PREFIX))}
        c["subject_overlap"] += bool(sa & sb)
        c["subject_overlap_no_domain"] += bool({x for x in sa & sb if not x.startswith("domain:")})
        c["doc_overlap"] += bool({x for x in a["mask_ids"] if x.startswith("D:")} & set(b["mask_ids"]))
        anchor_side = set(a.get("anchor_topic_ids") or []) | ({a["anchor_mention_id"]} if a.get("anchor_mention_id")
                                                               else set())
        c["anchor_side_overlap"] += bool(anchor_side & set(b["mask_ids"]))
    return {k: round(v / n, 4) for k, v in c.items()}


def guard_stats(rows: List[dict], builder: "MixBuilder") -> Tuple[dict, dict]:
    """Festival-guard and mention-guard summaries of the final rows (plan A8 / finding 23).

    :param rows: Final rows.
    :param builder: The builder (pre-dedupe festival-guard decisions in ``builder.guard``).
    :returns: (``festival_guard``, ``mention_guard``) dicts for the stats file.
    :rtype: Tuple[dict, dict]
    """
    tagged = [r for r in rows if r.get("anchor_topic_ids")]
    festival = {
        "enabled": builder.festival_guard,
        "rows_tagged": len(tagged),
        "by_family": dict(Counter(r["family"] for r in tagged).most_common()),
        "by_mask_level": dict(Counter(r["mask_level"] for r in tagged)),
        "by_topic": dict(Counter(t for r in tagged for t in r["anchor_topic_ids"]
                                 if not t.startswith(MENTION_PREFIX)).most_common()),
        "decisions_pre_dedupe": dict(sorted(builder.guard.items())),
    }
    with_mentions = [r for r in rows if any(x.startswith(MENTION_PREFIX) for x in r["mask_ids"])]
    mention = {
        "enabled": builder.mention_masks and builder.labels is not None,
        "rows_with_mention_mask_ids": len(with_mentions),
        "mention_mask_ids": sum(1 for r in rows for x in r["mask_ids"] if x.startswith(MENTION_PREFIX)),
        "rows_anchor_mention_id": sum(1 for r in rows if r.get("anchor_mention_id")),
        "records_with_mentions": sum(1 for d in builder.train if builder.mention_ids(d)),
        "train_records": len(builder.train),
    }
    return festival, mention


def v10_rule_stats(rows: List[dict], builder: "MixBuilder", subject_stats: Counter,
                   labels: Optional[SubjectLabels], gate_strings: Dict[str, str]) -> dict:
    """What each v10 rule removed or changed (plan A7 / C2 / C4 / C6 / C9), for the stats file.

    :param rows: Final rows.
    :param builder: The builder (drop counters and removal examples).
    :param subject_stats: Stats of :meth:`MixBuilder.subject`.
    :param labels: Subject labels.
    :param gate_strings: Blocked gate queries.
    :returns: ``{festival_canon_drops, name_form_gate, name_stoplist, mask_level_rule, gate, schema}``.
    :rtype: dict
    """
    def drops(suffix: str) -> Dict[str, int]:
        return {k.split(":")[0]: v for k, v in sorted(builder.drop.items()) if k.endswith(":" + suffix)}

    content_canon = sum(1 for r in rows if r["family"] == "content"
                        and canon_only_festivals(r["anchor"], r["doc_id"], labels))
    not_carrying = Counter(r["family"] for r in rows if r["kind"] == "q2d" and r["family"] != "content"
                           and any(festival_carry(labels, r["doc_id"], f"topic:{f}") == "none"
                                   for f in bare_festivals(r["anchor"])))
    sub = {k: v for k, v in sorted(subject_stats.items())
           if k.startswith(("carriers_by_class", "pairs_by_class", "pairs_by_kind_class", "no_name_left",
                            "bare_festival_names_skipped", "names_stoplisted", "carrier_keys"))}
    short = [r for r in rows if r["family"] in SHORT_ANCHOR_FAMILIES]
    gate_drops: Counter = Counter()  # rows per family ("subject_name_blocked" counts vocabulary names)
    for k, v in builder.detail.items():
        if k.endswith(":gate") and "leak_" in k:
            gate_drops[k.split(":")[0]] += v
    return {
        "festival_canon_drops": {
            "enabled": builder.festival_canon_drops,
            "t2t_festival_canon_pair": drops("festival_canon_pair"),
            "q2d_bare_festival_canon_positive": drops("bare_festival_canon_positive"),
            "subject_records_no_name_left_bare_festival": subject_stats.get("no_name_left:bare_festival", 0),
            "info_content_rows_bare_festival_canon_positive": content_canon,
            "info_q2d_bare_festival_positive_not_carrying": dict(not_carrying.most_common()),
            "examples": {k: v for k, v in sorted(builder.examples.items()) if k != "subject_names_stoplisted"},
        },
        "name_form_gate": {"enabled": builder.name_form_gate, "sefaria_canon": builder.sefaria_canon, **sub},
        "name_stoplist": {"enabled": builder.name_stoplist,
                          "names_found_in_vocab": builder.examples.get("subject_names_stoplisted", [])},
        "mask_level_rule": {"short_anchor_subject": builder.short_anchor_subject,
                            "max_tokens": SHORT_ANCHOR_MAX_TOKENS,
                            "by_family_level": dict(Counter(f"{r['family']}:{r['mask_level']}" for r in short))},
        "gate": {"blocked_strings": len(gate_strings), "anchor_drops_by_family": dict(gate_drops.most_common())},
        "schema": {"records_with_weak_reason": len(labels.weak_reason) if labels else 0,
                   "records_with_canon": len(labels.canon) if labels else 0,
                   "records_with_mentions": len(labels.mentions) if labels else 0,
                   "gold_override_records": len(labels.gold_override) if labels else 0},
    }


def language_stats(rows: List[dict]) -> dict:
    """Anchor script shares per family and overall.

    :param rows: Final rows.
    :returns: ``{family: {n, he, latin, other}}`` with shares, plus ``ALL``.
    :rtype: dict
    """
    by: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by[r["family"]][r["anchor_lang"]] += 1
        by["ALL"][r["anchor_lang"]] += 1
    return {f: {"n": sum(c.values()), **{k: round(c[k] / sum(c.values()), 3) for k in ("he", "latin", "other")}}
            for f, c in sorted(by.items())}


def focus_report(rows: List[dict], ev: dict, graph: dict, v3_eval_dir: Path, v3_stats: Path) -> dict:
    """Final run: what the mixture now trains on the six focus subjects that the v3 mixture could not.

    A row "touches the six" when any of these holds: its subject ids (``mask_ids`` / ``anchor_mask_ids`` /
    ``anchor_subject`` / ``shared_subject``) include a focus subject or one of their v3 linked subjects; it is a t2t
    row on a Sefaria node that v3's 2-hop rule excluded; or its positive / anchor record was held out in v3.

    :param rows: Final mixture rows.
    :param ev: :func:`load_eval` output of the final eval dir (``focus``, ``focus_doc``).
    :param graph: Sefaria concept graph.
    :param v3_eval_dir: The v3 eval dir (its held-out records).
    :param v3_stats: The v3 mixture's ``.stats.json`` (pair counts by family), or a missing path.
    :returns: Counts by family and by focus subject, plus the by-family comparison with v3.
    :rtype: dict
    """
    focus = set(ev["focus"])
    doc = ev["focus_doc"] or {}
    v3_linked = set(doc.get("v3_linked", {}))
    v3_rule = {"held_subjects": focus, "linked": v3_linked,
               "patterns": {s["id"]: s["concept_pattern"] for s in doc.get("subjects", [])}}
    v3_excluded, _ = excluded_nodes(held_out_seed_nodes(v3_rule, graph), graph)
    v3_held = set(json.loads((v3_eval_dir / "held_out_records.json").read_text())["held_out"])
    why: Dict[str, Counter] = defaultdict(Counter)
    per_subject: Dict[str, Counter] = {s: Counter() for s in sorted(focus)}
    any_touch: Counter = Counter()
    for r in rows:
        ids = set(r["mask_ids"]) | set(r["anchor_mask_ids"]) | {r.get("anchor_subject"), r.get("shared_subject")}
        hits = {
            "focus_subject": bool(ids & focus),
            "v3_linked_subject": bool(ids & v3_linked),
            "t2t_v3_excluded_node": r["kind"] == "t2t" and any(c.split(":", 1)[1] in v3_excluded
                                                               for c in r.get("concept_ids") or []),
            "record_held_out_in_v3": r.get("doc_id") in v3_held or r.get("anchor_doc_id") in v3_held,
            "subject_anchor_is_focus": r.get("anchor_subject") in focus,
        }
        for k, v in hits.items():
            why[k][r["family"]] += v
        if any(hits.values()):
            any_touch[r["family"]] += 1
        for s in ids & focus:
            per_subject[s][r["family"]] += 1
    by_family = Counter(r["family"] for r in rows)
    v3_by_family = json.loads(v3_stats.read_text())["by_family"] if v3_stats.exists() else {}
    fams = sorted(set(by_family) | set(v3_by_family), key=lambda f: -by_family.get(f, 0))
    return {
        "rows_touching_six": sum(any_touch.values()), "rows_touching_six_by_family": dict(any_touch.most_common()),
        "by_test": {k: {"total": sum(c.values()), **{f: n for f, n in c.most_common() if n}} for k, c in why.items()},
        "per_focus_subject": {s: {"total": sum(c.values()), **dict(c.most_common())} for s, c in per_subject.items()},
        "v3_excluded_sefaria_nodes": len(v3_excluded & set(graph)),
        "by_family_vs_v3": {f: {"final": by_family.get(f, 0), "v3": v3_by_family.get(f, 0),
                                "delta": by_family.get(f, 0) - v3_by_family.get(f, 0)} for f in fams},
        "pairs": {"final": len(rows), "v3": sum(v3_by_family.values())},
    }


# ---------------------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------------------
def self_test() -> None:
    """Tiny checks of normalisation, masking and the leak filter on hand-made strings (no data files)."""
    assert strip_niqqud("נִשּׂוּאִים") == "נשואים"
    assert clean_name(" [Rabbi]  Shimon\n") == "Rabbi Shimon"
    assert norm_tokens("Rav Sa'adya's ‘Laws’ of LULAV—1!") == ["rav", "sa", "adya", "s", "laws", "of", "lulav", "1"]
    assert norm_tokens('ר"ת ושו׳ת') == ["ר", "ת", "ושו", "ת"]
    assert script_of("הלכות שבת") == "he" and script_of("Bavli Shabbat") == "latin" and script_of("123") == "other"
    masked, n = mask_ids("Joins with T-S 13J21.18 and CUL Or.1080 J24 (PGPID 37026). Own: ENA 1234.5", "ENA 1234.5")
    assert n == 4 and not id_hits(masked) and masked.count(MASK) == 4, masked
    assert id_hits("Shelf Mark: x") and id_hits("see Bodl. MS heb. d 66") and not id_hits("Letter about trade")
    for cited in ("T-S NS J401.21", "TS NS J 401", "T-S Loan 137", "CUL Add.3339a–c", "Bodl. Heb. f. 26/1-6",
                  "ENA NS I.2", "ENA L43", "AIU XII.118", "AIU IV C 2", "JRL Gaster heb. ms 1636/14",
                  "Gaster ar. 328", "Yevr. III B", "Antonin 1234", "PGPID 16178", "JTS: MS 9160",
                  "JTS Geniza Misc. 15", "Bodl. d79.35", "T-S NS J.98", "T-S K.149", "T-S AS152.236",
                  "T-S Arabic 30.123", "ENA Misc. 17", "Yevr. Arab. II 1545", "AIU D 75"):
        assert MASK in mask_ids(f"cf. {cited} here", "")[0], cited
    for plain in ("Shelomo Mosseri, Hayyim", "the AIU collection", "a note at JTS", "the ENA series"):
        assert mask_ids(plain, "")[1] == 0, plain
    assert own_mark_in("fragment T-S 13J21.18 recto", "T-S 13J21.18") and not own_mark_in("T-S", "T-S")
    leak = LeakFilter({"laws of lulav": "probe", "Shemini Atzeret liturgical poetry": "known_item",
                       "כתובה": "known_item", "two merchants pool their capital": "llm_t2"},
                      ["Sukkot", "עבד", "palm branches"], ["Piyyut Yotzer"],
                      {"topic:sukkot": r"sukk|lulav|(?<![א-ת])[בהולש]{0,2}(?:סוכה)(?![א-ת])"})
    assert leak.check("Laws of LULAV!", anchor=True) == "exact:probe"
    assert leak.check("piyyut yotzer", anchor=True) == "exact:linked_name"
    assert leak.check("piyyut yotzer", anchor=False) is None
    assert leak.check("ישב בסוכה", anchor=False) == "concept:topic:sukkot"
    assert leak.check("מסכות", anchor=False) is None  # Hebrew word boundary: not סוכה
    assert leak.check("trade in palm branches from Egypt", anchor=False) == "contains:held_out_name"
    assert leak.check("ויאמר עבד אברהם", anchor=False) is None  # ambiguous name: anchors only
    assert leak.check("עבד נאמן", anchor=True) == "contains:held_out_name"
    assert leak.check("the two merchants pool their capital in Fustat", anchor=True) == "contains:llm_t2"
    assert leak.check("Atzeret liturgical poetry", anchor=True) == "contained_in:known_item"
    assert leak.check("liturgical poetry for Shemini Atzeret day", anchor=True) == "jaccard:known_item"
    assert leak.check("כתובה של אלמנה", anchor=True) is None  # one shared token is not a near-copy
    assert leak.check("כתובה של אלמנה", anchor=True, min_shared=1) == "jaccard:known_item"
    assert leak.check("Shabbat liturgical poetry", anchor=True) is None
    # final run: focus queries are fuzzy eval strings (names only with focus_names="block"); no concept regex
    fev = {"subject_queries": {
        "topic:f": {"status": "focus", "queries": [{"text": "laws of lulav", "source": "probe"},
                                                   {"text": "Sukkot", "source": "name"},
                                                   {"text": "Feast of Tabernacles", "source": "name"},
                                                   {"text": "two merchants pool their capital", "source": "llm_t2"}]},
        "pgp:s": {"status": "seen", "queries": [{"text": "partnership", "source": "name"}]}},
        "known_item": [{"text": "a letter about the price of flax"}], "held_names": [], "patterns": {}}
    fblock, fcounts = build_leak_filter(fev, "v3", 3, 2, "block")
    assert fcounts["focus_name"] == 2 and fcounts["probe"] == 1 and fcounts["llm_t2"] == 1, fcounts
    assert fblock.check("SUKKOT", anchor=True) == "exact:focus_name"
    assert fblock.check("Laws of lulav", anchor=True) == "exact:probe"
    assert fblock.check("partnership", anchor=True) is None                     # seen names stay training labels
    assert fblock.check("piyyut for the festival of Sukkot", anchor=False) is None  # no concept regex any more
    assert fblock.check("hymns for the Feast of Tabernacles", anchor=False) == "contains:focus_name"
    assert fblock.check("the two merchants pool their capital", anchor=True) == "contains:llm_t2"
    assert fblock.check("דיני הסכך ושמחת לולב בגמרא", anchor=False) is None
    fev["subject_queries"]["topic:f"]["queries"].append({"text": "שמחת בית השואבה", "source": "probe"})
    plain, _ = build_leak_filter(fev, "v3", 3, 2, "block")
    pref, _ = build_leak_filter(fev, "v3", 3, 2, "block", he_prefixes=True)
    text = "דיני הסכך ושמחת בית השואבה בגמרא"  # the v3-final miss: "ו" + probe query inside a synthetic anchor
    assert plain.check(text, anchor=True) is None and pref.check(text, anchor=True) == "contains:probe"
    assert plain.check("ובשמחת בית השואבה", anchor=True) == "jaccard:probe"     # short texts: Jaccard already
    assert pref.check("ובשמחת בית השואבה", anchor=True) == "contains:probe"
    assert pref.check("שמחת בית השואבה בירושלים", anchor=False) == "contains:probe"
    assert pref.check("ושמחת בית המקדש", anchor=True) is None                      # rest of the sequence differs
    assert pref.check("ויאמר עבד אברהם", anchor=False) is None                      # short names: no prefix forms
    flabel, lcounts = build_leak_filter(fev, "v3", 3, 2, "label")
    assert "focus_name" not in lcounts and flabel.check("Sukkot", anchor=True) is None
    assert flabel.check("laws of lulav", anchor=True) == "exact:probe"
    # final-strict: two-token focus probe / llm_t2 queries inside anchors (prefix-aware), glued exact matches
    fev["subject_queries"]["topic:f"]["queries"] += [{"text": "בדיקת חמץ", "source": "probe"},
                                                     {"text": "Hoshana Rabbah", "source": "probe"},
                                                     {"text": "הושענא רבה", "source": "probe"},
                                                     {"text": "Haggadah", "source": "probe"},
                                                     {"text": "חג הפסח", "source": "name"}]
    loose, _ = build_leak_filter(fev, "v3", 3, 2, "block", he_prefixes=True)
    strict, scounts = build_leak_filter(fev, "v3", 3, 2, "block", he_prefixes=True, strict=True)
    assert scounts["focus_two_token_anchor_only"] == 3 and "focus_two_token_anchor_only" not in _, scounts
    anchor = "סוגיית בדיקת חמץ בתלמוד הבבלי"
    assert loose.check(anchor, anchor=True) is None and strict.check(anchor, anchor=True) == "contains:probe"
    assert strict.check(anchor, anchor=False) is None                       # record texts: the subject is trained
    assert strict.check("סדר תפילות להושענא רבה", anchor=True) == "contains:probe"   # prefixed first word
    assert loose.check("סדר תפילות להושענא רבה", anchor=True) is None
    assert strict.check("piyyutim for Hoshana Rabbah night", anchor=True) == "contains:probe"
    assert strict.check("Passover Haggadah as given by Maimonides", anchor=True) is None  # one-word queries: exact only
    assert strict.check("ספר חג פסח", anchor=True) is None                   # names keep the >= 3-token rule
    assert loose.check("Hosh'ana Rabbah", anchor=True) is None
    assert strict.check("Hosh'ana Rabbah", anchor=True) == "exact_glued:probe"
    assert strict.check("Hosh ana Rabbah piyyut", anchor=True) is None      # glued = whole-text exact only
    rows = {"t1": {"eligible": True, "text_hash": "a", "text": "Page from a Hebrew medical treatise"},
            "h1": {"eligible": True, "text_hash": "b", "text": "Page from a Hebrew medical treatise."},
            "t2": {"eligible": True, "text_hash": "c", "text": "Letter about the flax trade"},
            "x": {"eligible": False, "text_hash": "d", "text": ""}}
    gev = {"held": {"h1"}, "held_subjects": set(), "linked": set()}
    for gate, want in ((False, {"t1", "t2"}), (True, {"t2"})):
        mb = MixBuilder(gev, LeakFilter({}, [], [], {}), rows, {}, {}, {}, random.Random(0), norm_gate=gate)
        assert set(mb.train) == want and mb.gate_stats["norm_text_as_held_out"] == int(gate), mb.gate_stats
    vrow = {"anchor": "a query", "positive": rows["t1"]["text"], "family": "synthetic_implicit", "kind": "q2d",
            "mask_level": "subject", "doc_id": "t1", "mask_ids": ["D:t1"], "anchor_mask_ids": []}
    vchk = verify([vrow], gev, rows, LeakFilter({}, [], [], {}), norm_gate=True)
    assert vchk["positive_norm_equals_held_out_text"] == 1 and vchk["doc_id_norm_text_as_held_out"] == 1, vchk
    assert not verify([vrow], gev, rows, LeakFilter({}, [], [], {}))["positive_norm_equals_held_out_text"]
    seeds = {"a"}
    graph = {"a": {"links": {"is-a": ["hub"]}}, "b": {"links": {"related-to": ["a"]}},
             "c": {"links": {"related-to": ["b"]}}, "d": {"links": {"related-to": ["c"]}},
             "e": {"links": {"displays-under": ["a"]}}, "hub": {"links": {}}}
    graph.update({f"x{i}": {"links": {"is-a": ["hub"]}} for i in range(HOP_HUB_DEGREE + 1)})
    ex, literal = excluded_nodes(seeds, graph)
    assert {"a", "b", "c", "hub"} <= ex and "d" not in ex and "e" not in ex and "x0" not in ex, ex
    assert literal > len(ex & set(graph))
    self_test_v10()
    print("self-test ok")


def _toy_builder(**flags) -> Tuple["MixBuilder", SubjectLabels, Dict[str, dict], dict]:
    """A tiny in-memory builder for :func:`self_test_v10` (no data files).

    Records: P1 / P2 strong Passover carriers, C1 / C2 Passover canon-weak (tractate), L1 Passover incidental-weak,
    M1 a mention-only record (also mentions held-out Hanukkah), N1 nothing, E1 an Esther scroll (canon-weak Purim +
    ``sefaria:purim``), Q1 a strong Purim carrier, D1 a strong divorce carrier, X1 a held-out-subject carrier.

    :param flags: :class:`MixBuilder` keyword flags.
    :returns: (builder, labels, semantic rows, eval dict).
    :rtype: Tuple[MixBuilder, SubjectLabels, Dict[str, dict], dict]
    """
    texts = {"P1": "Illustrated seder service with songs and blessings", "P2": "Festival night liturgy, Yotzer poems",
             "C1": "Talmud Bavli tractate on leaven and its search", "C2": "Talmud Bavli folio on the paschal lamb",
             "L1": "Letter of a merchant about payments due after the holiday", "M1": "Commentary on the festival",
             "N1": "A list of goods and prices", "E1": "Scroll of the Book of Esther, chapters 3-4",
             "Q1": "Poem for the feast of lots", "D1": "Deed releasing a wife, signed by witnesses",
             "X1": "Poem for the lights"}
    rows = {d: {"eligible": True, "text_hash": hashlib.md5(t.encode("utf-8")).hexdigest(), "text": t}
            for d, t in texts.items()}
    pes, pur = "topic:pesach", "topic:purim"
    subjects = {"P1": {pes, "work:haggadah"}, "P2": {pes}, "C1": {pes}, "C2": {pes}, "L1": {pes}, "M1": set(),
                "N1": set(), "E1": {pur, "sefaria:purim"}, "Q1": {pur}, "D1": {"topic:divorce"},
                "X1": {"topic:hanukkah"}}
    weak = {d: set() for d in texts}
    weak.update({"C1": {pes}, "C2": {pes}, "L1": {pes}, "E1": {pur}})
    labels = SubjectLabels(subjects, weak,
                           {"C1": {pes: "canon"}, "C2": {pes: "canon"}, "L1": {pes: "incidental"}, "E1": {pur: "canon"}},
                           {"C1": {pes}, "C2": {pes}, "E1": {pur}},
                           {"M1": {pes, "topic:hanukkah"}, "N1": set()}, set())
    status = {s: "seen" for s in (pes, pur, "sefaria:purim", "topic:divorce", "work:haggadah", "topic:sukkot")}
    status["topic:hanukkah"] = "held_out"
    ev = {"held": set(), "held_subjects": {"topic:hanukkah"}, "linked": {"pgp:linked"}, "status": status}
    builder = MixBuilder(ev, LeakFilter({}, [], [], {}), rows, subjects, weak, {}, random.Random(0),
                         labels=labels, **flags)
    return builder, labels, rows, ev


def self_test_v10() -> None:
    """Torch-free checks of the v10 rules (plan A6 / A7 / C2 / C3 / C4 / C6 / C9 and finding 22), in memory."""
    # --- shared rules
    assert bare_festivals("Passover") == {"pesach"} and not bare_festivals("laws of Pesach")
    assert not bare_festivals("Talmud Bavli Sukkah") and not bare_festivals("Pesach 12a")
    assert not bare_festivals("פסח דף ה") and not bare_festivals('פסח ע"ב')
    assert bare_festivals("piyyut for Purim from the Book of Esther") == {"purim"}   # liturgy cue: not canon
    for cited in ("Bavli Sukkah chapters 1-5", "Pesachim daf 6 biur chametz", "Rosh Hashanah Gemara Tu Bishvat",
                  "גמרא ראש השנה על ניסן", "כותח הבבלי ותערובת חמץ", "Yerushalmi Pesahim on matzah",
                  "maseches Sukkah dofen akuma"):
        assert canon_reference(cited) and not bare_festivals(cited), cited      # Talmud known-item anchors
    for plain in ("Simchat Torah celebration letter", "Passover hametz removal before travel", "סדר הגדה של פסח"):
        assert not canon_reference(plain) and bare_festivals(plain), plain
    assert canon_reference("Book of Esther") and canon_reference("שבת דף ב") and not canon_reference("מדף ספרים")
    assert festival_canon_pair("Esther", "Purim") and festival_canon_pair("פורים", "אסתר")
    assert not festival_canon_pair("Purim", "פורים") and not festival_canon_pair("Laws of Purim", "Book of Esther")
    assert is_stoplisted("get") and is_stoplisted("מזון") and not is_stoplisted("bill of divorce")
    assert mask_level_of("synthetic_hebrew", "כתובה קראית") == "subject"
    assert mask_level_of("synthetic_translit", "ketubbah of a Karaite bride from Fustat") == "doc"
    assert mask_level_of("synthetic_specific", "Rif Yoma") == "doc" and mask_level_of("content", "a b") == "doc"
    assert mask_level_of("synthetic_hebrew", "כתובה קראית", short_anchor_subject=False) == "doc"
    assert mask_level_of("synthetic_implicit", "a long topical question about trade") == "subject"
    # --- record ids, mention masks (held-out mention dropped), festival carry
    b, labels, rows, ev = _toy_builder()
    assert "X1" not in b.train and set(b.train) == set(rows) - {"X1"}
    assert b.record_ids("M1") == ["M:topic:pesach", "D:M1"], b.record_ids("M1")
    assert b.subject_ids("P1") == ["topic:pesach", "work:haggadah"]
    carry = {d: festival_carry(labels, d, "topic:pesach") for d in ("P1", "C1", "L1", "M1", "N1")}
    assert carry == {"P1": "strong", "C1": "canon", "L1": "weak", "M1": "mention", "N1": "none"}, carry
    # --- anchor_festivals (A6): canon / folio anchors, canon-only / absent positives, held-out festivals
    both = ["M:topic:pesach", "topic:pesach"]
    assert b.anchor_festivals("Passover Haggadah with pictures", "P1") == both
    assert b.anchor_festivals("Passover Haggadah with pictures") == both          # no positive test
    assert b.anchor_festivals("Passover customs", "C1") == [] and b.anchor_festivals("Passover customs", "N1") == []
    assert b.anchor_festivals("Passover letter", "L1") == both and b.anchor_festivals("Pesach notes", "M1") == both
    assert b.anchor_festivals("Talmud Pesahim on leaven", "P1") == [] and b.anchor_festivals("Pesach 12a", "P1") == []
    assert b.anchor_festivals("Hanukkah and Passover songs", "P1") == both         # held-out Hanukkah never tagged
    assert b.guard["skip_not_trainable:hanukkah"] == 1 and b.guard["skip_positive_canon:pesach"] == 1, b.guard
    # --- emit: doc- and subject-level tagging, C6 mask level, drops (A7)
    assert b.emit("Passover Haggadah with pictures", "synthetic_specific", "P1")
    row = b.out[-1]
    assert row["mask_level"] == "doc" and row["anchor_topic_ids"] == both and "anchor_mention_id" not in row
    assert row["mask_ids"] == ["topic:pesach", "work:haggadah", "D:P1"]
    assert b.emit("הגדה של פסח", "synthetic_hebrew", "P1") and b.out[-1]["mask_level"] == "subject"
    assert b.out[-1]["anchor_topic_ids"] == both
    assert b.emit("Passover", "subject", "P2", anchor_subject="topic:pesach", subject_kind="topic")
    row = b.out[-1]
    assert row["mask_level"] == "subject" and row["anchor_topic_ids"] == both
    assert row["anchor_mention_id"] == "M:topic:pesach"
    assert b.emit("laws of Pesach", "subject", "C1", anchor_subject="topic:pesach", subject_kind="topic")
    assert "anchor_topic_ids" not in b.out[-1] and b.out[-1]["anchor_mention_id"] == "M:topic:pesach"
    assert b.emit("Talmud Pesahim on leaven", "synthetic_specific", "C1") and "anchor_topic_ids" not in b.out[-1]
    assert b.emit("בפסח אכלו מצות ומרור בלילה", "content", "C1", positive="label") \
        and "anchor_topic_ids" not in b.out[-1]                                   # content: exempt from A6 / A7
    n_out = len(b.out)
    assert not b.emit("Passover holiday customs", "synthetic_implicit", "C1")     # bare festival -> canon-only
    assert not b.emit("Esther", "t2t_parent", positive="Purim", concept_ids=["sefaria:esther", "sefaria:purim"])
    assert not b.emit("אסתר", "t2t_xling", positive="פורים", concept_ids=["sefaria:x"])
    assert b.emit("Purim", "t2t_xling", positive="פורים", concept_ids=["sefaria:purim"]) and len(b.out) == n_out + 1
    assert b.drop["synthetic_implicit:bare_festival_canon_positive"] == 1
    assert b.drop["t2t_parent:festival_canon_pair"] == 1 and b.drop["t2t_xling:festival_canon_pair"] == 1, b.drop
    assert "anchor_topic_ids" not in b.out[-1] and not b.out[-1]["mask_ids"]
    good = list(b.out)
    vgood = verify(good, ev, rows, LeakFilter({}, [], [], {}), labels=labels)
    assert not {k: v for k, v in vgood.items() if v and not k.startswith("info_")}, vgood
    assert vgood["info_rows_anchor_topic_ids"] == 3 and vgood["info_rows_anchor_mention_id"] == 2, vgood
    # --- flags: no festival guard, no mention masks, no drops, legacy mask level
    b2, *_ = _toy_builder(festival_guard=False)
    assert b2.emit("Passover Haggadah with pictures", "synthetic_specific", "P1") and "anchor_topic_ids" not in b2.out[-1]
    assert b2.emit("Passover", "subject", "P2", anchor_subject="topic:pesach")
    assert "anchor_topic_ids" not in b2.out[-1] and b2.out[-1]["anchor_mention_id"] == "M:topic:pesach"
    b3, *_ = _toy_builder(mention_masks=False)
    assert b3.record_ids("M1") == ["D:M1"] and b3.anchor_festivals("Pesach notes", "M1") == []
    assert b3.anchor_festivals("Passover letter", "L1") == ["topic:pesach"]
    assert b3.emit("Passover", "subject", "P2", anchor_subject="topic:pesach") and "anchor_mention_id" not in b3.out[-1]
    b4, *_ = _toy_builder(festival_canon_drops=False, short_anchor_subject=False)
    assert b4.emit("Passover holiday customs", "synthetic_implicit", "C1") and "anchor_topic_ids" not in b4.out[-1]
    assert b4.emit("Esther", "t2t_parent", positive="Purim", concept_ids=["sefaria:esther"])
    assert b4.emit("הגדה של פסח", "synthetic_hebrew", "P1") and b4.out[-1]["mask_level"] == "doc"
    v4 = verify(b4.out, ev, rows, LeakFilter({}, [], [], {}), labels=labels, festival_canon=False,
                short_anchor_subject=False)
    assert v4["info_bare_festival_anchor_canon_positive"] == 1 and not v4["bare_festival_anchor_canon_positive"], v4
    assert verify(b4.out, ev, rows, LeakFilter({}, [], [], {}), labels=labels)["bad_mask_level"] == 1
    assert verify(b4.out, ev, rows, LeakFilter({}, [], [], {}), labels=labels)["bare_festival_anchor_canon_positive"] == 1
    # --- verify catches planted held-out / linked ids and malformed guard fields
    base = dict(good[0])
    planted = [dict(base, anchor_topic_ids=["topic:hanukkah", "topic:pesach"]),
               dict(base, anchor_topic_ids=["M:topic:hanukkah"]),
               dict(base, family="subject", mask_level="subject", anchor_subject="topic:hanukkah",
                    anchor_mention_id="M:topic:hanukkah"),
               dict(base, mask_ids=base["mask_ids"][:-1] + ["M:pgp:linked", "D:P1"]),
               dict(base, anchor_topic_ids=["pgp:partnership"]),
               dict(good[5], anchor_topic_ids=["topic:pesach"]),
               dict(base, family="subject", mask_level="subject", anchor_subject="topic:pesach",
                    anchor_mention_id="M:topic:purim"),
               dict(good[1], mask_level="doc")]
    vbad = verify(planted, ev, rows, LeakFilter({}, [], [], {}), labels=labels)
    assert vbad["held_or_linked_subject_in_anchor_topic_ids"] == 2, vbad
    assert vbad["held_or_linked_subject_in_anchor_mention_id"] == 1 and vbad["held_or_linked_subject_in_mask_ids"] == 1
    assert vbad["bad_anchor_topic_id"] == 1 and vbad["anchor_topic_ids_on_content_t2t_d2d"] == 1, vbad
    assert vbad["bad_anchor_mention_id"] == 1 and vbad["bad_mask_level"] == 1, vbad
    # --- subject family: name-form gate (C2), bare-festival rule (A7), stoplist (C4)
    vocab = {"topic:pesach": {"names_en": ["Passover", "laws of Pesach"], "names_he": ["פסח", "הלכות פסח"]},
             "topic:purim": {"names_en": ["Purim"], "names_he": ["פורים"]},
             "sefaria:purim": {"names_en": ["Purim", "Laws of Purim"], "names_he": []},
             "topic:divorce": {"names_en": ["Get", "bill of divorce"], "names_he": []}}
    b5, *_ = _toy_builder()
    st = b5.subject(vocab, {}, {}, {}, k=10.0)
    pairs = {(r["doc_id"], r["anchor_subject"]): r["anchor"] for r in b5.out}
    assert {"P1", "P2", "C1", "C2", "Q1", "E1", "D1"} == {d for d, _ in pairs}, pairs
    assert all(is_law_name(a) for (d, _), a in pairs.items() if d in ("C1", "C2")), pairs   # canon: law names only
    assert not any(is_law_name(a) for (d, _), a in pairs.items() if d in ("P1", "P2")), pairs  # content: bare names
    assert pairs[("E1", "sefaria:purim")] == "Laws of Purim" and ("E1", "topic:purim") not in pairs, pairs
    assert pairs[("D1", "topic:divorce")] == "bill of divorce" and pairs[("Q1", "topic:purim")] in ("Purim", "פורים")
    assert st["no_name_left:class_canon"] >= 1 and st["names_stoplisted"] == 1, st
    assert all(r["anchor_mention_id"] == "M:" + r["anchor_subject"] for r in b5.out)
    assert verify(b5.out, ev, rows, LeakFilter({}, [], [], {}), labels=labels)["bare_festival_anchor_canon_positive"] == 0
    b8, *_ = _toy_builder()
    b8.subject(vocab, {}, {}, {}, k=10.0, k_by_kind={"topic": 0.1})                 # per-kind quota: 1 pair per key
    per_key8 = Counter((r["anchor_subject"], r["doc_id"] in ("C1", "C2")) for r in b8.out)
    assert per_key8[("topic:pesach", True)] == 1 and per_key8[("topic:pesach", False)] == 1, per_key8
    assert per_key8[("sefaria:purim", False)] == 1, per_key8                      # other kinds keep k
    bare_vocab = dict(vocab, **{"sefaria:purim": {"names_en": ["Purim"], "names_he": ["פורים"]}})
    b6, *_ = _toy_builder(name_form_gate=False, festival_canon_drops=False, name_stoplist=False)
    b6.subject(bare_vocab, {}, {}, {}, k=10.0)
    p6 = {(r["doc_id"], r["anchor_subject"]): r["anchor"] for r in b6.out}
    assert ("E1", "sefaria:purim") in p6 and not ({"C1", "C2"} & {d for d, _ in p6}), p6   # v9: bare name -> canon
    b7, *_ = _toy_builder(name_form_gate=False)
    st7 = b7.subject(bare_vocab, {}, {}, {}, k=10.0)
    assert ("E1", "sefaria:purim") not in {(r["doc_id"], r["anchor_subject"]) for r in b7.out}   # A7 alone closes it
    assert st7["no_name_left:bare_festival"] >= 1 and st7["no_name_left:bare_festival:sefaria"] >= 1, st7
    assert not any(r["anchor"] == "Get" for r in b7.out)
    # --- gate strings (C9): anchors only, every test; record texts untouched
    gs = gate_strings_from_rows([{"text": "documentary letters about flax", "blocked": True},
                                 {"text": "Purim", "blocked": False}, {"text": " ", "blocked": True}])
    assert gs == {"documentary letters about flax": "gate"}, gs
    spec = {"version": 1, "concepts": {"c1": {"queries": [{"text": "a", "form": "en", "blocked": False},
                                                          {"text": "seder plate songs", "form": "en", "blocked": True}]}}}
    assert gate_strings_from_spec(spec) == {"seder plate songs": "gate"}
    gleak = LeakFilter({"known item query here": "known_item"}, [], [], {}, anchor_strings=dict(gs, **{
        "known item query here": "gate"}))
    assert gleak.check("Documentary letters about flax!", anchor=True) == "exact:gate"
    assert gleak.check("Documentary letters about flax!", anchor=False) is None
    assert gleak.check("three documentary letters about flax from Fustat", anchor=True) == "contains:gate"
    assert gleak.check("three documentary letters about flax from Fustat", anchor=False) is None
    assert gleak.check("letters about flax", anchor=True) == "contained_in:gate"
    assert gleak.check("letters about flax", anchor=False) is None
    assert gleak.check("documentary letters about the flax trade", anchor=True) == "jaccard:gate"
    assert gleak.check("documentary letters about the flax trade", anchor=False) is None
    assert gleak.check("known item query here", anchor=False) == "exact:known_item"   # record-level source kept
    short = LeakFilter({}, [], [], {}, he_prefixes=True, anchor_strings={"כתובה קראית": "gate", "Purim": "gate"})
    assert short.check("כתובה קראית של נעמה בת ניסי", anchor=True) == "contains:gate"  # 2-token query inside
    assert short.check("וכתובה קראית מקהיר", anchor=True) == "contains:gate"           # Hebrew prefix
    assert short.check("כתובה קראית של נעמה בת ניסי", anchor=False) is None
    assert short.check("purim", anchor=True) == "exact:gate" and short.check("Purim piyyutim", anchor=True) is None
    gev = {"subject_queries": {}, "known_item": [], "held_names": [], "patterns": {}}
    _, gcounts = build_leak_filter(gev, "v3", 3, 2, gate_strings=gs)
    assert gcounts["gate_anchor_only"] == 1, gcounts
    print("self-test v10 ok")


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Build ``v3/train_mix_v3.jsonl`` + ``.stats.json``; refuse to write when verification finds a violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-festival-guard", action="store_true",
                        help="do not tag q2d rows with anchor_topic_ids (reproduces the v9 mixture)")
    parser.add_argument("--no-mention-masks", action="store_true",
                        help="ablation: no M:<sid> mention ids in mask ids, no anchor_mention_id, no M: twins in "
                             "anchor_topic_ids; mentions ignored by the festival-carry test (plan C3)")
    parser.add_argument("--no-name-form-gate", action="store_true",
                        help="ablation: subject family uses strong carriers with every name and one quota per "
                             "subject (v9) instead of canon / content carrier classes (plan C2)")
    parser.add_argument("--no-festival-canon-drops", action="store_true",
                        help="ablation: keep bare festival names on non-strong carriers, t2t festival <-> canon pairs "
                             "and q2d rows whose bare festival anchor has a canon-only positive (plan A7)")
    parser.add_argument("--no-name-stoplist", action="store_true",
                        help="ablation: allow build_subjects.NAME_STOPLIST names as subject anchors (plan C4)")
    parser.add_argument("--no-sefaria-canon", action="store_true",
                        help="name-form gate: treat sefaria:* carriers as canon only when listed in subjects_v1 canon")
    parser.add_argument("--legacy-mask-level", action="store_true",
                        help="synthetic_hebrew / synthetic_translit rows are always doc-level (v9); default: anchors of "
                             f"<= {SHORT_ANCHOR_MAX_TOKENS} normalised tokens are subject-level (plan C6). Also the rule "
                             "--verify checks; pass it to verify a v9 mixture")
    parser.add_argument("--gate-dir", default=None,
                        help="acceptance gate dir (eval_final/gate_v1): its queries.jsonl rows with blocked=true are "
                             "kept out of training anchors (plan C9)")
    parser.add_argument("--gate-spec", default=None,
                        help="gate_v1_spec.json: its blocked concept queries are kept out of training anchors (use "
                             "before the gate dir is built; unioned with --gate-dir when both are given)")
    parser.add_argument("--verify", metavar="JSONL", help="only scan an existing mixture file against the eval set")
    parser.add_argument("--semantic", default=str(V3 / "corpus_semantic.jsonl"))
    parser.add_argument("--subjects", default=str(V3 / "subjects_v1.jsonl"))
    parser.add_argument("--vocab", default=str(V3 / "subject_vocab.json"))
    parser.add_argument("--eval-dir", default=str(V3 / "eval"))
    parser.add_argument("--corpus-v1", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--merged", default=str(MERGED))
    parser.add_argument("--concept-graph", default=str(AUDIT_ROOT / "sefaria/derived/concept_graph.json"))
    parser.add_argument("--frame-refs", default=str(AUDIT_ROOT / "frame_refs.jsonl"))
    parser.add_argument("--curated-refs", default=str(AUDIT_ROOT / "sefaria/derived/curated_refs.jsonl"))
    parser.add_argument("--out", default=str(V3 / "train_mix_v3.jsonl"))
    parser.add_argument("--rounds", default="r1,r2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--subject-k", type=float, default=5.0, help="subject quota = ceil(k * sqrt(n_records))")
    parser.add_argument("--domain-k", type=float, default=None,
                        help="quota factor for domain:* subjects only (default: --subject-k); 2.5 builds the plan C7 "
                             "fallback mix with the KTIV domain quota halved")
    parser.add_argument("--content-spans", type=int, default=2, help="transcription spans sampled per record")
    parser.add_argument("--content-cap", type=int, default=0, help="max content pairs (0 = all)")
    parser.add_argument("--d2d-cap", type=int, default=3000)
    parser.add_argument("--t2t-cap", type=int, default=0, help="max t2t pairs (0 = all)")
    parser.add_argument("--min-contain", type=int, default=3)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--probe-scope", choices=("all", "v3"), default="all")
    parser.add_argument("--focus-names", choices=("block", "label"), default="block",
                        help="final run: keep the focus subjects' name queries out of training (block, default) or "
                             "allow them as training labels like seen-subject names (label)")
    parser.add_argument("--he-prefix-contain", choices=("auto", "on", "off"), default="auto",
                        help="Hebrew-prefix-aware containment of eval queries in the leak filter; auto = on when the "
                             "eval dir has focus subjects (final run: no concept regexes any more), off otherwise (v3)")
    parser.add_argument("--final-strict", choices=("auto", "on", "off"), default="auto",
                        help="final-run tightenings (auto = on when the eval dir has focus subjects, off for v3): "
                             "records whose text equals a held-out record's up to niqqud / case / punctuation / "
                             "spacing are not trained on; the focus subjects' two-token probe / llm_t2 queries are "
                             "blocked inside anchors; exact matches ignore spacing / punctuation inside words")
    parser.add_argument("--v3-eval-dir", default=str(V3 / "eval"), help="final run: v3 eval dir for focus_report")
    parser.add_argument("--v3-stats", default=str(V3 / "train_mix_v3.stats.json"),
                        help="final run: v3 mixture stats for the by-family comparison in focus_report")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    default_out, default_eval = str(V3 / "train_mix_v3.jsonl"), str(V3 / "eval")
    if not args.verify and Path(args.out) == Path(default_out) and Path(args.eval_dir) != Path(default_eval):
        raise SystemExit(f"--eval-dir {args.eval_dir} would overwrite the v3 mixture {default_out}: pass --out")
    rng = random.Random(args.seed)
    ev = load_eval(Path(args.eval_dir))
    if args.he_prefix_contain == "auto":  # recorded resolved in the stats params (the packager rebuilds the filter)
        args.he_prefix_contain = "on" if ev["focus"] else "off"
    if args.final_strict == "auto":  # same: recorded resolved, read back by the packager
        args.final_strict = "on" if ev["focus"] else "off"
    strict = args.final_strict == "on"
    gate_strings = load_gate_strings(Path(args.gate_dir) if args.gate_dir else None,
                                     Path(args.gate_spec) if args.gate_spec else None)
    if ev["focus"] and not gate_strings:
        print("WARNING: final-run eval dir but no blocked gate queries (--gate-dir / --gate-spec): gate validity (H11) "
              "is not protected by this mixture", flush=True)
    leak, leak_sources = build_leak_filter(ev, args.probe_scope, args.min_contain, args.min_shared, args.focus_names,
                                           args.he_prefix_contain == "on", strict, gate_strings)
    rows, order = load_semantic(Path(args.semantic))
    labels = load_subject_labels(Path(args.subjects))
    if args.verify:
        checks = verify(iter_jsonl(Path(args.verify)), ev, rows, leak, norm_gate=strict, labels=labels,
                        festival_canon=not args.no_festival_canon_drops,
                        short_anchor_subject=not args.legacy_mask_level)
        violations = {k: v for k, v in checks.items() if v and not k.startswith("info_")}
        print(json.dumps({"checks": dict(sorted(checks.items())), "violations": violations}, ensure_ascii=False,
                         indent=1))
        if violations:
            raise SystemExit(f"{args.verify}: {len(violations)} kinds of violation")
        return
    subj, weak = labels.subjects, labels.weak
    vocab = json.loads(Path(args.vocab).read_text())
    graph = json.loads(Path(args.concept_graph).read_text())
    boiler = set(json.loads(Path(args.semantic).with_suffix(".stats.json").read_text())["boilerplate_list"])
    marks = own_shelfmarks(Path(args.corpus_v1))
    tags, types = pgp_tags_types(order, Path(args.merged))
    sub_slugs = {s for t, slugs in TOPIC_SEFARIA.items() for s in slugs}
    refs = ref_slugs(Path(args.frame_refs), Path(args.curated_refs), sub_slugs)

    builder = MixBuilder(ev, leak, rows, subj, weak, marks, rng, norm_gate=strict,
                         festival_guard=not args.no_festival_guard, labels=labels,
                         mention_masks=not args.no_mention_masks, name_form_gate=not args.no_name_form_gate,
                         festival_canon_drops=not args.no_festival_canon_drops,
                         name_stoplist=not args.no_name_stoplist, sefaria_canon=not args.no_sefaria_canon,
                         short_anchor_subject=not args.legacy_mask_level)
    family_stats = {"synthetic": builder.synthetic(args.rounds.split(",")),
                    "subject": builder.subject(vocab, graph, refs, tags, args.subject_k,
                                               None if args.domain_k is None else {"domain": args.domain_k})}
    seeds = held_out_seed_nodes(ev, graph)
    excluded, literal_excluded = excluded_nodes(seeds, graph)
    t2t_rows, t2t_stats = t2t_pairs(graph, excluded, rng)
    rng.shuffle(t2t_rows)
    n_t2t = 0
    for r in t2t_rows:
        if args.t2t_cap and n_t2t >= args.t2t_cap:
            t2t_stats["cap"] += 1
            continue
        n_t2t += builder.emit(r["anchor"], r["family"], positive=r["positive"], concept_ids=r["concept_ids"])
    t2t_stats.update({"seed_nodes": len(seeds), "excluded_nodes_in_graph": len(excluded & set(graph)),
                      "literal_2hop_excluded_in_graph": literal_excluded, "graph_nodes": len(graph)})
    family_stats["t2t"] = t2t_stats
    family_stats["content"] = builder.content(boiler, args.content_spans, args.content_cap)
    family_stats["d2d"] = builder.d2d(types, args.d2d_cap)

    # dedupe exact (anchor, positive), shuffle
    seen: Set[Tuple[str, str]] = set()
    final = []
    for r in builder.out:
        key = (r["anchor"], hashlib.md5(r["positive"].encode("utf-8")).hexdigest())
        if key in seen:
            builder.drop[f"{r['family']}:duplicate_pair"] += 1
            continue
        seen.add(key)
        final.append(r)
    rng.shuffle(final)

    # diagnostic: what a 1-shared-token Jaccard rule would additionally drop (short anchors only)
    strict_extra: Counter = Counter()
    for r in final:
        if r["kind"] != "d2d" and leak.check(r["anchor"], anchor=True, min_shared=1):
            strict_extra[r["family"]] += 1

    # post-build assert (plan A7) is part of verify: 0 q2d rows with a bare festival anchor and a canon-only positive
    checks = verify(final, ev, rows, leak, norm_gate=strict, labels=labels,
                    festival_canon=not args.no_festival_canon_drops, short_anchor_subject=not args.legacy_mask_level)
    violations = {k: v for k, v in checks.items() if v and not k.startswith("info_")}
    by_family = Counter(r["family"] for r in final)
    festival_guard, mention_guard = guard_stats(final, builder)
    stats = {
        "pairs": len(final),
        "by_family": dict(by_family.most_common()),
        "by_kind": dict(Counter(r["kind"] for r in final)),
        "by_mask_level": dict(Counter(r["mask_level"] for r in final)),
        "anchor_language": language_stats(final),
        "records": dict(builder.gate_stats),
        "positive_records_used": len({r["doc_id"] for r in final if r["doc_id"]}),
        "families": {k: dict(v) for k, v in family_stats.items()},
        "subject_top": builder.subject_top,
        "dropped": dict(sorted(builder.drop.items())),
        "drop_detail_top": dict(builder.detail.most_common(60)),
        "leak_filter_sources": dict(leak_sources),
        "strict_jaccard_1token_would_also_drop": dict(strict_extra),
        "mask_overlap_random_pairs": mask_overlap(final, random.Random(0)),
        "festival_guard": festival_guard,
        "mention_guard": mention_guard,
        "v10_rules": v10_rule_stats(final, builder, family_stats["subject"], labels, gate_strings),
        "verification": {k: v for k, v in sorted(checks.items()) if not k.startswith("leak_detail")},
        "violations": violations,
        "params": vars(args),
    }
    if ev["focus"]:
        stats["focus_report"] = focus_report(final, ev, graph, Path(args.v3_eval_dir), Path(args.v3_stats))
    stats_path = Path(args.out).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: stats[k] for k in ("pairs", "by_family", "by_kind", "by_mask_level", "anchor_language",
                                            "records", "festival_guard", "mention_guard", "v10_rules", "violations",
                                            "focus_report") if k in stats},
                     ensure_ascii=False, indent=1))
    if violations:
        raise SystemExit(f"verification failed, {args.out} not written: {violations}")
    tmp = Path(args.out).with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"wrote {len(final)} rows -> {args.out}; stats -> {stats_path}")


if __name__ == "__main__":
    main()
