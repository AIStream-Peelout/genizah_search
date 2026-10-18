"""Frozen evaluation set v3 for the semantic text embedder (text track step 2 of 4).

Principle: score subject similarity on records AND subjects the model never trained on, so a gain means the model
generalises (the index keeps growing; new records and new subjects arrive all the time). This is the split of the
*evaluation* run only: once the recipe is proven, the production model is retrained on everything.

Writes ``AUDIT_ROOT/v3/eval/`` (json only, no model)::

    held_out_subjects.json   6 subjects held out of training entirely (+ the "linked" subjects that go with them)
    held_out_records.json    records that must never be trained on, with the reason(s) for each
    subject_queries.json     queries per subject: probe / llm_t2 implicit queries for held-out subjects, EN+HE names
                             for every subject; each subject tagged held_out / linked / seen; each query tagged
                             rel_mode bare / law
    subject_relevance.json   eligible carriers of every subject: strong / weak / canon / unjudged
    holiday_queries.json     name forms of every festival with relevant / canon / unjudged / alt_only /
                             held_out_relevant records (eval_v3's "holiday" section)
    known_item.jsonl         eval-pool synthetic queries {qid, doc_id, type, lang, text, round, shelfmark_query}
    memorization_pairs.json  seen subjects: size-matched train-side vs held-out carriers (memorisation gap)
    collection_sample.json   fixed 3,000 eligible unique-text records for the collection-lift metric
    eval_pool.jsonl          the frozen candidate pool: every eligible record {doc_id, text_hash, series, n_chars, dup_n, held_out}
    build_stats.json         counts printed by this script
    README.md                what every file and metric means

Run after ``semantic_text.py`` and ``build_subjects.py``::

    python3 build_eval_v3.py

Fields (Description, Alternate titles, Document Type) are parsed from the corpus column ``text_v3`` (the semantic_v3
text) after folding continuation lines, never from the embedded ``text`` (``semantic_v4`` has no Alternate titles /
Document Type lines). ``subjects_v1.jsonl`` files of the v10 schema (``weak_reason``, ``canon``, ``mentions``,
``gold_override``) give typed relevance; older files (only ``subjects`` / ``weak``) still load: no canon / mention
sets, and every weak festival carrier counts as canon (a miss) in the holiday set, as on 2026-10-08.

**Final mode** (``--final`` -> ``AUDIT_ROOT/v3/eval_final/``): the eval set of the FINAL run, which trains on the six
subjects above and their linked subjects. No subject is held out (``held_out_subjects.json`` has empty ``subjects`` /
``linked`` / ``concept_patterns``); held-out records = the same 20 % split + Sukkot gold + synthetic-query eval pool,
closed over identical text (no subject / linked-subject / concept-mention reasons). The six subjects get status
``focus`` in ``subject_queries.json`` (same probe / llm_t2 / name queries as v3) and are described in
``focus_subjects.json``; ``eval_v3`` scores them over held-out records only (train-side carriers are left out of the
ranking). The split, seeds, known-item, memorisation and collection rules are those of v3, and the builder checks
that the final held-out set is a subset of v3's with identical split / gold / eval-pool members (so a v3-trained model
never saw a final held-out record) and reports what moved to the train side::

    python3 build_eval_v3.py --final

On a newer corpus (``semantic_v4`` text, new records) that check cannot hold (new ids fall on the held-out side, and
regrouped identical texts move across the split); ``--v3-compare auto`` (default) then only reports the differences,
including ``v3_trainable_now_held_out``: records a v3-final-trained model (v9) may have trained on, which is why v9 is
scored with its trained ids ignored.
"""

import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from build_query_batches import is_test
from build_subjects import (FESTIVAL_INCIDENTAL_TYPES, FESTIVAL_RE, MERGED, PGP_DOCUMENTARY_TYPES, fold_field_lines,
                            is_law_name, iter_jsonl, pgp_info, text_field)
from embed_utils import AUDIT_ROOT
from eval_v3 import FILES, HOLIDAY_FILE, LONG_CHARS, SHORT_CHARS

HERE = Path(__file__).resolve().parent
V3 = AUDIT_ROOT / "v3"
PROBE = HERE / "concept_probe_v1.json"
HEB = re.compile(r"[֐-׿]")

# --------------------------------------------------------------------------------------------------------------
# held-out subjects (chosen from the candidate tables this script also writes; the criteria are re-checked below)
# --------------------------------------------------------------------------------------------------------------
HELD_OUT_TOPICS = {  # subject id -> (concept_probe_v1 key, reason)
    "topic:sukkot": ("sukkot", "Flagship: the owner's 'laws of lulav' example; carries the 125-record human-verified "
                               "Sukkot gold seed; 19 probe queries."),
    "topic:pesach": ("pesach", "Festival with the most probe queries after Sukkot/Shabbat (12) and >= 100 strong "
                               "records; little record overlap with Sukkot (12%, vs 22% for Yom Kippur and 34% for "
                               "Rosh Hashanah), so the two festival gates are close to independent."),
    "topic:kashrut": ("kashrut", "Largest non-calendar halakhic probe topic (dietary law / ritual slaughter): tests "
                                 "'laws of X' generalisation outside the festival cycle; its PGP twin pgp:kashrut is "
                                 "linked and held out with it."),
}
PROBE_TOPIC_REJECTED = {
    "topic:shabbat": "pervasive: ~250 seen Sefaria/Mishneh Torah subjects are Shabbat labour topics, holding it out "
                     "would strip a large part of the halakhic training signal",
    "topic:marriage": "its PGP twins (pgp:marriage 1,182, pgp:ketubba 652) are seen documentary subjects: the concept "
                      "would leak through them",
    "topic:divorce": "PGP twin pgp:divorce (219) stays seen",
    "topic:mourning": "fewer than 100 strong records; PGP twin pgp:mourning stays seen",
}
HELD_OUT_PGP = {
    "pgp:partnership": "Economy / legal instruments: partnership and commenda deeds (95% PGP 'Legal document'), "
                       "contained in no other subject.",
    "pgp:slavery": "Law / social history: sales, manumissions, court cases and letters about enslaved people; "
                   "pgp:manumission (fully inside it) is linked and held out too.",
    "pgp:geonic-academies": "Communal / religious institutions: letters to and from the Babylonian and Palestinian "
                            "yeshivot; mostly letters, under half inside pgp:communal-affairs.",
}
PGP_REJECTED = {
    "pgp:magic": "only 7% of its records are documentary (PGP types: amulets/charms are 'Paraliterary text')",
    "pgp:poll-tax": "100% of its records also carry pgp:taxes (the curation maps capitation tax -> poll-tax + taxes), "
                    "so it is a sub-label of a seen subject, not an unseen subject",
}
PGP_CRITERIA = {"min_eligible": 50, "max_eligible": 300, "min_documentary_share": 0.9, "max_containment": 0.5}
PGP_EXCLUDED_CATEGORIES = {"place", "letters"}  # places say where, letter-genre tags say what kind of document
PROBE_TWIN_PGP = {"pgp:marriage", "pgp:divorce", "pgp:mourning", "pgp:kashrut", "pgp:ketubba"}
LINK_OVERLAP = 0.5  # a seen subject is "linked" when >= this share of its eligible records carry a held-out subject

# Concept patterns (subject ids/names -> "linked"; also handed to T3 to drop training anchors that name the concept).
# Review 2026-10-05 added the unambiguous ritual terms the first version missed (ushpizin, water libation, charoset,
# afikoman, "dietary laws", lung adhesions / ~סירכ, ~שוחט, maidservant ...): train records such as "Dietary laws
# concerning animals and birds" or "הלכות בדיקת הריאה" and anchors such as "סירכה בין אונות הריאה" leaked the concept.
# Hebrew: a word may carry up to two prefix letters (ב ה ו ל ש) but must not sit inside another word, so "שפחה" does
# not match "משפחה" and "סכות" does not match "מסכות"; whole words also need a word end, stems (marked ~) do not.
HE_START, HE_END = r"(?<![\u05D0-\u05EA])[בהולש]{0,2}", r"(?![\u05D0-\u05EA])"
CONCEPT_TERMS = {  # held-out subject -> (English regex, Hebrew words / ~stems)
    "topic:sukkot": (r"sukk|succ|lulav|etrog|ethrog|hoshan|four[- ]species|willow|aravah|schach|sekhakh|tabernacle"
                     r"|ushpiz|water[- ]libation|simc?hat be(?:i)?t ha",
                     ["סוכות", "סכות", "סוכה", "סכה", "לולב", "אתרוג", "~הושענ", "ארבעה מינים", "סכך", "אושפיזין",
                      "ניסוך המים", "שמחת בית השואבה"]),
    "topic:pesach": (r"pesa[hḥc]|pessach|passover|chametz|hametz|ḥametz|hamets|matzah|matza\b|matzot|matsah|haggad"
                     r"|hagad|four[- ]questions|bedikat|seder[- ]night|passover[- ]night|unleavened bread|bitter herbs"
                     r"|charoset|[hḥ]aroset|afikoman|paschal",
                     ["פסח", "פסחים", "חמץ", "מצה", "חג המצות", "הגדה", "חרוסת", "אפיקומן", "מה נשתנה", "ליל הסדר",
                      "ארבע כוסות"]),
    "topic:kashrut": (r"kashr|kosher|shehit|shechit|shekhit|slaughter|terefa|tereif|treif|terefot|hullin|chullin"
                      r"|ḥullin|meat[- ]and[- ]milk|forbidden[- ]food|maakhalot|nikkur|porging"
                      r"|dietary (?:law|prohibition|rule|regulation|restriction)|inspection of the lung|lung adhesion"
                      r"|gid ha-?nasheh|sciatic nerve",
                      ["כשרות", "~שחיט", "טרפה", "טריפה", "טרפות", "טריפות", "חולין", "בשר בחלב", "מאכלות אסורות", "ניקור",
                       "~סירכ", "בדיקת הריאה", "גיד הנשה", "~שוחט"]),
    "pgp:partnership": (r"partner|\bcommenda|qir[aā][dḍ]|shirka|sharika", ["~שותפ", "עסקא"]),
    "pgp:slavery": (r"slave|manumi|jariya|ghulam|eunuch|freedwoman|freedman|maidservant|servant[- ]girl|concubine",
                    ["עבדות", "עבדים", "שפחה", "שפחות", "גט שחרור"]),  # bare עבד also spells Arabic names (Abd al-)
    "pgp:geonic-academies": (r"geonic|geonim|gaonate|yeshiv|academ(?:y|ies)|\bsura\b|pumbedita",
                             ["ישיבה", "ישיבת", "ישיבות", "גאונים", "סורא", "~פומבדית"]),
}


def concept_regex(english: str, hebrew: List[str]) -> str:
    """Join an English regex and Hebrew words/stems into one pattern with Hebrew word boundaries.

    :param english: English alternation (matched case-insensitively anywhere).
    :param hebrew: Hebrew whole words, or stems prefixed with ``~``.
    :returns: Combined regex string.
    :rtype: str
    """
    words = "|".join(w for w in hebrew if not w.startswith("~"))
    stems = "|".join(w[1:] for w in hebrew if w.startswith("~"))
    parts = [english, f"{HE_START}(?:{words}){HE_END}"]
    if stems:
        parts.append(f"{HE_START}(?:{stems})")
    return "|".join(parts)


CONCEPT_PATTERNS = {h: concept_regex(en, he) for h, (en, he) in CONCEPT_TERMS.items()}
FOCUS_SUBJECTS = list(HELD_OUT_TOPICS) + list(HELD_OUT_PGP)  # final mode: trained on, reported as the "focus" group
LINK_EXCLUDE = {  # seen subject -> why a pattern/overlap match is not really the same concept
    "sefaria:laws-of-slaughter-on-shabbat": "Shabbat labour (slaughtering as one of the 39 melakhot), not dietary law",
    "sefaria:laws-of-the-rest-of-animals-and-slaves": "Shabbat rest of one's animals and servants, not slavery",
}
CATALOGUE_PREFIXES = ("Description:", "Alternate titles:")
LAW_SOURCES = ("probe", "llm_t2")  # implicit / nuanced queries: canon carriers (tractate, law code) answer them
GOLD_FESTIVAL = "topic:sukkot"     # the festival the Sukkot gold seed (gold_override) forces strong

# Implicit search queries for the held-out PGP subjects (source "llm_t2", written for this eval by Claude, 2026-10-05).
# They describe the content and avoid the tag word and its catalogue synonyms (checked by assert_no_names).
PGP_QUERIES = {
    "pgp:partnership": {
        "en": ["two merchants pool their capital and agree to split the profits and losses",
               "agreement to run a shop together for three years, each side contributing a sum of dinars",
               "an investor supplies the money while a travelling trader does the work for a share of the gain",
               "dissolving a joint business venture and releasing one another from all further claims",
               "court settlement dividing the stock and debts of a shared trading enterprise",
               "brothers agree to trade with common funds and divide the earnings equally",
               "deed recording how much capital each associate brought into the business",
               "dispute over losses in a jointly owned venture brought before the rabbinical court"],
        "he": ["שטר עסקה בין שני סוחרים לחלוקת הריווח וההפסד",
               "מי שנתן מעות לחברו להתעסק בהן למחצית שכר",
               "פירוק עסק של שני סוחרים ומחילה הדדית על כל תביעה",
               "הסכם להשקעת כסף של שני אנשים בחנות אחת"],
    },
    "pgp:slavery": {
        "en": ["bill of sale for a Nubian maidservant bought at the market in Fustat",
               "a master frees his household servant girl and declares her children free",
               "court case over a purchased servant girl found to have a hidden defect",
               "letter asking a relative to buy a young male servant for the household",
               "a woman formerly owned by her master, now free, appears before the court",
               "converting a purchased maidservant to Judaism and the status of her offspring",
               "human beings bought and sold as property in medieval Egypt",
               "dispute over a concubine bought by a merchant and her share in his estate"],
        "he": ["שטר מכירת נערה נובית בשוק פסטאט",
               "שטר שבו אדון מוציא את משרתתו לחירות",
               "בני אדם שנקנו ונמכרו כרכוש",
               "תביעה בבית הדין על נערה שנקנתה ונמצא בה מום"],
    },
    "pgp:geonic-academies": {
        "en": ["letter from the heads of the Babylonian rabbinic schools in Baghdad to the communities of Egypt",
               "appeal for donations to support the scholars of Sura and Pumbedita",
               "the head of Palestinian Jewry in Jerusalem confirms a judge's appointment in Fustat",
               "honorific titles conferred by the Iraqi rabbinic leadership on a Fustat merchant",
               "questions of Jewish law sent from Kairouan to Baghdad and the replies",
               "struggle over the leadership of the Palestinian rabbinic school in the eleventh century",
               "funds collected in Egypt for the twice-yearly study assemblies in Iraq",
               "letter of thanks from the Baghdad religious authorities for money sent from Fustat"],
        "he": ["איגרת מראשי סורא ופומבדיתא לקהילות מצרים",
               "בקשת תרומות לחכמי בבל",
               "מינוי דיין בפסטאט מטעם ראש החבורה בירושלים",
               "שאלות ששלחו מקירואן לבבל והתשובות עליהן"],
    },
}
NAME_CAP_EN, NAME_CAP_HE = 6, 4
MEMO_MAX, MEMO_MIN = 50, 5
COLLECTION_N = 3000
# classmark tokens that would make a known-item query a shelf-mark lookup (the semantic text carries no shelf marks)
SHELFMARK_RE = re.compile(r"\b(?:T-?S|ENA|Mosseri|Bodl|Halper|Antonin|Yevr|CUL|JTS|AIU|Or\.|Taylor-Schechter|Gaster)\b")


# --------------------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------------------
def catalogue_text(row: dict) -> str:
    """The record's semantic_v3 text with continuation lines folded into their field line (field parsing input).

    Fields (Description, Alternate titles, Document Type) are always parsed from ``text_v3`` (the semantic_v3 text,
    present in every corpus written with ``--text-version``), never from the embedded ``text``: ``semantic_v4``
    folds Alternate titles into Description and drops Document Type, so parsing it would silently return "".
    Older corpora have no ``text_v3``; their ``text`` is the semantic_v3 text.

    :param row: ``corpus_semantic.jsonl`` row.
    :returns: Folded semantic_v3 text.
    :rtype: str
    """
    return fold_field_lines(row.get("text_v3") or row["text"])


def load_semantic(path: Path) -> Dict[str, dict]:
    """Semantic corpus rows without texts, plus the parsed catalogue fields.

    Added per row: ``n_chars`` (length of the embedded text), ``text_hash_v3`` (md5 of ``text_v3``, only when the
    corpus has that column), ``desc`` / ``alt`` / ``doc_type`` (Description, Alternate titles and Document Type of
    the folded semantic_v3 text, :func:`catalogue_text`) and ``catalogue`` (the Description and Alternate titles
    lines joined by newlines).

    :param path: ``corpus_semantic.jsonl``.
    :returns: ``doc_id -> row``.
    :rtype: Dict[str, dict]
    """
    rows = {}
    for row in iter_jsonl(path):
        folded = catalogue_text(row)
        if "text_v3" in row:
            row["text_hash_v3"] = hashlib.md5(row.pop("text_v3").encode("utf-8")).hexdigest()
        row["n_chars"] = len(row.pop("text"))
        row["desc"], row["alt"] = text_field(folded, "Description"), text_field(folded, "Alternate titles")
        row["doc_type"] = text_field(folded, "Document Type")
        row["catalogue"] = catalogue_lines(folded)
        rows[row["doc_id"]] = row
    return rows


class SubjectLabels(NamedTuple):
    """Per-record labels of ``subjects_v1.jsonl`` (missing keys of older files read as empty)."""

    subj: Dict[str, Set[str]]               # every subject id (strong and weak)
    weak: Dict[str, Set[str]]               # weak subject ids
    weak_reason: Dict[str, Dict[str, str]]  # weak subject id -> "canon" / "incidental" / "title"
    canon: Dict[str, Set[str]]              # topic ids matched on a canon frame / canon description
    mentions: Dict[str, Set[str]]           # subject ids the catalogue names but the record does not carry strong
    gold_override: Set[str]                 # records whose Sukkot label is the gold seed's override
    typed: bool                             # the file has weak_reason (v10 schema); False = untyped (v9) weak labels


def load_subjects(path: Path) -> SubjectLabels:
    """Subject labels per record (v10 schema; v9 files without the new keys still load).

    :param path: ``subjects_v1.jsonl``.
    :returns: :class:`SubjectLabels`.
    :rtype: SubjectLabels
    """
    subj, weak, reason, canon, mentions = {}, {}, {}, {}, {}
    gold: Set[str] = set()
    typed = False
    for row in iter_jsonl(path):
        d = row["doc_id"]
        subj[d] = set(row["subjects"])
        weak[d] = set(row["weak"])
        typed |= "weak_reason" in row
        reason[d] = dict(row.get("weak_reason") or {})
        canon[d] = set(row.get("canon") or []) & subj[d]
        mentions[d] = set(row.get("mentions") or [])
        if row.get("gold_override"):
            gold.add(d)
    return SubjectLabels(subj, weak, reason, canon, mentions, gold, typed)


def weak_kind(labels: SubjectLabels, doc_id: str, sid: str) -> Optional[str]:
    """Why a record carries a subject only weakly.

    :param labels: Subject labels.
    :param doc_id: Record id.
    :param sid: Subject id.
    :returns: ``canon`` / ``incidental`` / ``title``, ``untyped`` for a weak label of a pre-v10 file (or one missing
        from ``weak_reason``), None when the label is not weak.
    :rtype: Optional[str]
    """
    if sid not in labels.weak.get(doc_id, ()):
        return None
    return labels.weak_reason.get(doc_id, {}).get(sid, "untyped")


def rel_mode_of(text: str, source: str) -> str:
    """``rel_mode`` of a subject query: ``law`` for law-name forms and probe / llm_t2 queries, else ``bare``.

    :param text: Query text.
    :param source: Query source (``probe``, ``llm_t2``, ``name``).
    :returns: ``law`` or ``bare``.
    :rtype: str
    """
    return "law" if source in LAW_SOURCES or is_law_name(text) else "bare"


def subject_relevance(vocab: dict, rows: Dict[str, dict], labels: SubjectLabels,
                      strong: Dict[str, Set[str]], by: Dict[str, Set[str]],
                      concept_unjudged: Dict[str, Set[str]]) -> Dict[str, dict]:
    """``subject_relevance.json``: strong / weak / canon / unjudged eligible records per subject.

    ``canon`` = eligible carriers that list the subject in their ``canon`` (or whose weak reason is ``canon``);
    ``unjudged`` = eligible records whose ``mentions`` name the subject, plus title / incidental weak carriers, plus
    (held-out / focus subjects) the concept-pattern mentions, minus strong and canon carriers. ``eval_v3`` drops
    unjudged records from the ranking and treats canon carriers by the query's ``rel_mode``.

    :param vocab: ``subject_vocab.json``.
    :param rows: Semantic rows.
    :param labels: Subject labels.
    :param strong: Strong eligible carriers per subject.
    :param by: Eligible carriers (strong or weak) per subject.
    :param concept_unjudged: Concept-pattern mentions of held-out / focus subjects.
    :returns: ``subject -> {strong, weak, canon, unjudged}``.
    :rtype: Dict[str, dict]
    """
    canon: Dict[str, Set[str]] = defaultdict(set)
    unjudged: Dict[str, Set[str]] = defaultdict(set)
    for d, row in rows.items():
        if not row["eligible"]:
            continue
        for sid in labels.canon.get(d, ()):
            canon[sid].add(d)
        for sid, why in labels.weak_reason.get(d, {}).items():
            if why == "canon" and sid in labels.subj[d]:
                canon[sid].add(d)
            elif why in ("title", "incidental"):
                unjudged[sid].add(d)
        for sid in labels.mentions.get(d, ()):
            unjudged[sid].add(d)
    for sid, ids in concept_unjudged.items():
        unjudged[sid] |= ids
    out = {}
    for s in vocab:
        st = strong.get(s, set())
        out[s] = {"strong": sorted(st), "weak": sorted(by.get(s, set()) - st), "canon": sorted(canon[s]),
                  "unjudged": sorted(unjudged[s] - st - canon[s])}
    return out


# Holiday regression set (2026-10-08 handoff, P4): every name form of every festival, scored over the WHOLE pool
# (train side included -- these are trained names, the check is that they still find the festival, not its scroll
# or tractate). Relevant = strong carriers (the record names the festival outside a canon / incidental context);
# canon carriers (scripture / tractate / law-code frames) count as misses; title-only, incidental and mention-only
# records are unjudged. Not in the training leak filter, like seen-subject names. "hebrew_prefixed" (2026-10-09):
# prefixed forms users type (הפורים is in the 2026-10-07 chat sessions); rosh_hodesh joined the festival topics.
HOLIDAY_NAME_FORMS: Dict[str, Dict[str, List[str]]] = {
    "topic:pesach": {"english": ["Passover"], "translit": ["Pesach", "Pesah"], "hebrew": ["פסח"],
                     "hebrew_prefixed": ["בפסח", "לפסח", "חג הפסח"],
                     "ashkenazi": ["Peysakh"], "typo": ["Pesech"],
                     "descriptive": ["Passover Haggadah", "piyyutim for Passover"]},
    "topic:sukkot": {"english": ["Feast of Tabernacles", "Feast of Booths"], "translit": ["Sukkot", "Succot"],
                     "hebrew": ["סוכות"], "hebrew_prefixed": ["בסוכות", "לסוכות", "חג הסוכות"],
                     "ashkenazi": ["Sukkos"], "typo": ["Sukot"],
                     "descriptive": ["hoshanot for Sukkot"]},
    "topic:yom_kippur": {"english": ["Day of Atonement"], "translit": ["Yom Kippur", "Yom ha-Kippurim"],
                         "hebrew": ["יום הכפורים"], "hebrew_prefixed": ["ביום הכפורים", "ליום הכפורים"],
                         "typo": ["Yom Kipur"], "descriptive": ["prayers for Yom Kippur"]},
    "topic:rosh_hashanah": {"english": ["Jewish New Year"], "translit": ["Rosh Hashanah", "Rosh ha-Shanah"],
                            "hebrew": ["ראש השנה"], "hebrew_prefixed": ["בראש השנה", "לראש השנה"],
                            "ashkenazi": ["Rosh Hashono"], "typo": ["Rosh Hashnnah"],
                            "descriptive": ["prayers for Rosh Hashanah"]},
    "topic:purim": {"english": ["Feast of Lots"], "translit": ["Purim"], "hebrew": ["פורים"],
                    "hebrew_prefixed": ["הפורים", "בפורים", "לפורים"], "typo": ["Puirm"],
                    "descriptive": ["poem for Purim"]},
    "topic:hanukkah": {"english": ["Festival of Lights"], "translit": ["Hanukkah", "Chanukah"], "hebrew": ["חנוכה"],
                       "hebrew_prefixed": ["בחנוכה", "לחנוכה"], "typo": ["Hannukah"]},
    "topic:shavuot": {"english": ["Pentecost", "Feast of Weeks"], "translit": ["Shavuot"], "hebrew": ["חג השבועות"],
                      "hebrew_prefixed": ["בשבועות", "לשבועות"], "ashkenazi": ["Shavuos"]},
    "topic:simhat_torah": {"english": ["Rejoicing of the Torah"], "translit": ["Simchat Torah", "Simhat Torah"],
                           "hebrew": ["שמחת תורה"], "hebrew_prefixed": ["בשמחת תורה", "לשמחת תורה"],
                           "ashkenazi": ["Simchas Torah"], "typo": ["simchat tora"]},
    "topic:fast_days": {"english": ["Ninth of Av"], "translit": ["Tisha be-Av"], "hebrew": ["תשעה באב"],
                        "hebrew_prefixed": ["בתשעה באב", "לתשעה באב"]},
    "topic:rosh_hodesh": {"english": ["New Moon"], "translit": ["Rosh Hodesh", "Rosh Chodesh"],
                          "hebrew": ["ראש חודש"], "hebrew_prefixed": ["בראש חודש", "לראש חודש"],
                          "descriptive": ["prayers for the New Moon"]},
}


def is_incidental(row: dict, pgp_types_of: Set[str]) -> bool:
    """Whether a record's type makes a festival mention incidental (``build_subjects``' rule).

    :param row: Semantic row (``doc_type`` parsed from the semantic_v3 text).
    :param pgp_types_of: The record's PGP document types.
    :returns: True for letters / legal / documentary records.
    :rtype: bool
    """
    return row["doc_type"] in FESTIVAL_INCIDENTAL_TYPES or bool(pgp_types_of & PGP_DOCUMENTARY_TYPES)


def holiday_queries(rows: Dict[str, dict], labels: SubjectLabels, strong: Dict[str, Set[str]],
                    by: Dict[str, Set[str]], held: Set[str], types: Dict[str, Set[str]]) -> Tuple[dict, dict]:
    """``holiday_queries.json``: name-form queries and the judged record sets per festival.

    Per festival (eligible records only):

    * ``relevant`` = strong carriers, minus Sukkot-gold records (``gold_override``) that are canon or incidental;
    * ``canon`` = carriers whose weak reason is ``canon`` (the festival is named only on a scripture / tractate /
      law-code frame or description); a subjects file without ``weak_reason`` (v9 labels) puts every weak carrier
      here, which reproduces the 2026-10-08 set (weak = misses);
    * ``unjudged`` = title-only and incidental weak carriers, records whose ``mentions`` name the festival, and the
      gold records above (removed from the ranking);
    * ``alt_only`` = relevant records whose semantic_v3 catalogue names the festival under ``Alternate titles:`` but
      not under ``Description:`` (the field the result card shows); its share among relevant top-10 hits, against
      its base rate, measures the field-label shortcut (v8 0.29, v9 0.88 in the handoff);
    * ``held_out_relevant`` = relevant records that are held out (the held-out-only variant).

    :param rows: Semantic rows (parsed catalogue fields, :func:`load_semantic`).
    :param labels: Subject labels.
    :param strong: Eligible strong carriers per subject.
    :param by: Eligible carriers (strong or weak) per subject.
    :param held: Held-out doc ids.
    :param types: ``doc_id -> PGP types``.
    :returns: (the file's content ``{built, note, festivals: {sid: {...}}}``, counts per festival for build_stats).
    :rtype: Tuple[dict, dict]
    """
    eligible = {d for d, r in rows.items() if r["eligible"]}
    festivals, counts = {}, {}
    for sid in sorted(s for s in HOLIDAY_NAME_FORMS if strong.get(s)):
        rx = FESTIVAL_RE.get(sid.split(":", 1)[1])
        carriers, st = by.get(sid, set()), strong[sid]
        gold_unj = {d for d in st & labels.gold_override if sid == GOLD_FESTIVAL
                    and (sid in labels.canon.get(d, ()) or is_incidental(rows[d], types.get(d, set())))}
        relevant = st - gold_unj
        canon = {d for d in carriers if weak_kind(labels, d, sid) in ("canon", "untyped")} - relevant - gold_unj
        unjudged = ({d for d in carriers if weak_kind(labels, d, sid) in ("title", "incidental")}
                    | {d for d in eligible if sid in labels.mentions.get(d, ())} | gold_unj) - relevant - canon
        alt_only = {d for d in relevant if rx and rx.search(rows[d]["alt"]) and not rx.search(rows[d]["desc"])}
        festivals[sid] = {
            "queries": [{"text": q, "form": form} for form, qs in HOLIDAY_NAME_FORMS[sid].items() for q in qs],
            "relevant": sorted(relevant), "canon": sorted(canon), "unjudged": sorted(unjudged),
            "alt_only": sorted(alt_only), "held_out_relevant": sorted(relevant & held)}
        counts[sid] = {"n_queries": len(festivals[sid]["queries"]), "n_relevant": len(relevant),
                       "n_canon": len(canon), "n_unjudged": len(unjudged), "n_alt_only": len(alt_only),
                       "alt_only_base_rate": round(len(alt_only) / len(relevant), 4) if relevant else None,
                       "n_held_out_relevant": len(relevant & held), "n_gold_unjudged": len(gold_unj),
                       "n_strong_listed_canon": sum(sid in labels.canon.get(d, ()) for d in relevant),
                       "festival_regex": rx is not None}
    return ({"built": date.today().isoformat(),
             "note": "whole-pool regression check; relevant = strong carriers, canon = misses, unjudged = removed "
                     "from the ranking; held_out_relevant = the held-out-only variant",
             "festivals": festivals}, counts)


def merged_pgp_info(ids: Iterable[str], merged: Path) -> Tuple[Dict[str, Tuple[Set[str], Set[str]]], int]:
    """Raw PGP tags and PGP document types of the corpus records, looked up by ``canonical_id``.

    The sibling project re-merges ``merged_shelfmarks.jsonl`` in place (2026-10-06: 1,144 more records inserted in
    the middle, and 387 v3 corpus ids no longer present), so the file is no longer in corpus order. A corpus record
    absent from it gets no PGP data: none of the 387 had PGP data when the corpus was built (no ``pgp_type`` in
    ``corpus_v1.jsonl``, no ``pgp:`` subject in ``subjects_v1.jsonl``), and the v3 eval / mixture rebuilt with this
    lookup are identical to the published v3 artefacts (checked 2026-10-07).

    :param ids: Corpus doc ids.
    :param merged: ``merged_shelfmarks.jsonl``.
    :returns: (``doc_id -> (tags, types)`` for every id, number of ids absent from the merged file).
    :rtype: Tuple[Dict[str, Tuple[Set[str], Set[str]]], int]
    :raises ValueError: When a corpus id occurs twice in the merged file.
    """
    wanted = set(ids)
    out: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for record in iter_jsonl(merged):
        cid = record["canonical_id"]
        if cid in wanted:
            if cid in out:
                raise ValueError(f"{cid} occurs twice in {merged}")
            out[cid] = pgp_info(record)
    missing = wanted - set(out)
    for d in missing:
        out[d] = (set(), set())
    return out, len(missing)


def pgp_types(order: List[str], merged: Path) -> Dict[str, Set[str]]:
    """PGP document types per record (see :func:`merged_pgp_info`).

    :param order: Corpus doc ids.
    :param merged: ``merged_shelfmarks.jsonl``.
    :returns: ``doc_id -> PGP types``.
    :rtype: Dict[str, Set[str]]
    """
    info, missing = merged_pgp_info(order, merged)
    print(f"merged PGP info: {len(info) - missing} corpus records matched by id, {missing} absent (no PGP data)",
          flush=True)
    return {d: ty for d, (_, ty) in info.items()}


def synthetic_pools(rounds: Iterable[str]) -> Dict[str, str]:
    """``doc_id -> pool`` ("train"/"eval") over the synthetic query rounds' batch files.

    :param rounds: Round names (``r1``, ``r2``).
    :returns: Pool per record (eval wins if a record appears in both).
    :rtype: Dict[str, str]
    """
    pool: Dict[str, str] = {}
    for rnd in rounds:
        base = AUDIT_ROOT / "synthetic_queries" / rnd
        for sub in ("batches", "batches_he"):
            for f in sorted(glob.glob(str(base / sub / "*.jsonl"))):
                for b in iter_jsonl(Path(f)):
                    if pool.get(b["doc_id"]) != "eval":
                        pool[b["doc_id"]] = b.get("pool", "train")
    return pool


def shelfmarks(corpus_v1: Path, wanted: Set[str]) -> Dict[str, str]:
    """``Shelf Mark:`` values of selected records from the production-text corpus.

    :param corpus_v1: ``corpus_v1.jsonl``.
    :param wanted: Doc ids to look up.
    :returns: ``doc_id -> shelf mark``.
    :rtype: Dict[str, str]
    """
    out = {}
    for row in iter_jsonl(corpus_v1):
        if row["doc_id"] in wanted:
            out[row["doc_id"]] = next((ln[len("Shelf Mark:"):].strip() for ln in row["text"].split("\n")
                                       if ln.startswith("Shelf Mark:")), "")
    return out


# --------------------------------------------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------------------------------------------
def split_side(rows: Dict[str, dict]) -> Dict[str, bool]:
    """20 % test side, grouped by text: identical eligible texts land on the same side.

    Compatible with the v1/v2 rule (``md5(doc_id) % 5 == 0``): a record whose text is unique, and every ineligible
    record (never a positive or a candidate), keeps its old side; an eligible duplicate-text group follows its
    lexicographically smallest member's old side (still a 20 % expected rate, independent of group size).

    :param rows: Semantic corpus rows.
    :returns: ``doc_id -> on the held-out side``.
    :rtype: Dict[str, bool]
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for d, r in rows.items():
        if r["eligible"]:
            groups[r["text_hash"]].append(d)
    side = {d: is_test(d) for d in rows}
    for members in groups.values():
        if len(members) > 1:
            rep = is_test(min(members))
            for d in members:
                side[d] = rep
    return side


# --------------------------------------------------------------------------------------------------------------
# subject selection
# --------------------------------------------------------------------------------------------------------------
def carriers(subj: Dict[str, Set[str]], rows: Dict[str, dict], eligible_only: bool = True) -> Dict[str, Set[str]]:
    """Records carrying each subject (strong or weak).

    :param subj: ``doc_id -> subjects``.
    :param rows: Semantic corpus rows.
    :param eligible_only: Count eligible records only.
    :returns: ``subject -> doc ids``.
    :rtype: Dict[str, Set[str]]
    """
    by: Dict[str, Set[str]] = defaultdict(set)
    for d, ss in subj.items():
        if rows[d]["eligible"] or not eligible_only:
            for s in ss:
                by[s].add(d)
    return by


def probe_topic_table(by: Dict[str, Set[str]], weak: Dict[str, Set[str]], probe: dict) -> List[dict]:
    """Candidate table for the probe-topic hold-outs.

    :param by: Eligible carriers per subject.
    :param weak: Weak subjects per record.
    :param probe: ``concept_probe_v1.json``.
    :returns: One row per probe concept with a topic subject.
    :rtype: List[dict]
    """
    out = []
    sukkot = by.get("topic:sukkot", set())
    for key, concept in probe["concepts"].items():
        sid = f"topic:{key}"
        recs = by.get(sid, set())
        strong = {d for d in recs if sid not in weak[d]}
        out.append({"subject": sid, "n_eligible": len(recs), "n_strong_eligible": len(strong),
                    "n_probe_queries": len(concept["queries"]),
                    "overlap_with_sukkot": round(len(recs & sukkot) / len(recs), 3) if recs and sid != "topic:sukkot" else None,
                    "chosen": sid in HELD_OUT_TOPICS, "rejected_because": PROBE_TOPIC_REJECTED.get(sid)})
    return sorted(out, key=lambda r: -r["n_eligible"])


def pgp_table(by: Dict[str, Set[str]], subj: Dict[str, Set[str]], types: Dict[str, Set[str]], vocab: dict) -> List[dict]:
    """Candidate table for the documentary PGP hold-outs (subjects with >= 50 eligible records).

    :param by: Eligible carriers per subject.
    :param subj: ``doc_id -> subjects``.
    :param types: ``doc_id -> PGP types``.
    :param vocab: ``subject_vocab.json``.
    :returns: Rows with size, documentary share, largest containment in another subject, category, pass/fail.
    :rtype: List[dict]
    """
    out = []
    for sid, meta in vocab.items():
        recs = by.get(sid, set())
        if not sid.startswith("pgp:") or len(recs) < PGP_CRITERIA["min_eligible"]:
            continue
        doc_share = sum(bool(types[d] & PGP_DOCUMENTARY_TYPES) for d in recs) / len(recs)
        overlap = Counter(x for d in recs for x in subj[d] if x != sid)
        top, n_top = overlap.most_common(1)[0] if overlap else (None, 0)
        row = {"subject": sid, "category": meta.get("category"), "n_eligible": len(recs),
               "documentary_share": round(doc_share, 3), "max_containment_in": top,
               "max_containment": round(n_top / len(recs), 3)}
        reasons = []
        if len(recs) > PGP_CRITERIA["max_eligible"]:
            reasons.append("too large (removes too much training data)")
        if doc_share < PGP_CRITERIA["min_documentary_share"]:
            reasons.append("not documentary")
        if row["max_containment"] >= PGP_CRITERIA["max_containment"]:
            reasons.append(f"mostly inside {top}")
        if meta.get("category") in PGP_EXCLUDED_CATEGORIES:
            reasons.append(f"category {meta.get('category')}")
        if sid in PROBE_TWIN_PGP:
            reasons.append("twin of a probe topic")
        row["passes"] = not reasons
        row["fails"] = reasons
        row["chosen"] = sid in HELD_OUT_PGP
        row["note"] = PGP_REJECTED.get(sid)
        out.append(row)
    return sorted(out, key=lambda r: (not r["passes"], r["category"] or "", -r["n_eligible"]))


def subject_names(meta: dict) -> List[str]:
    """English + Hebrew names of a subject, deduplicated case-insensitively and capped.

    :param meta: ``subject_vocab.json`` entry.
    :returns: Up to :data:`NAME_CAP_EN` English and :data:`NAME_CAP_HE` Hebrew names.
    :rtype: List[str]
    """
    out, seen = [], set()
    for names, cap in ((meta.get("names_en") or [], NAME_CAP_EN), (meta.get("names_he") or [], NAME_CAP_HE)):
        kept = 0
        for n in names:
            key = " ".join(n.lower().split())
            if len(key) < 2 or key in seen or kept >= cap:
                continue
            seen.add(key)
            out.append(n.strip())
            kept += 1
    return out


def find_linked(held: List[str], by: Dict[str, Set[str]], vocab: dict) -> Dict[str, dict]:
    """Seen subjects that are effectively the same concept as a held-out subject.

    Linked when >= :data:`LINK_OVERLAP` of the subject's eligible records carry the held-out subject, or when its id
    or one of its names matches the held-out subject's :data:`CONCEPT_PATTERNS` entry.

    :param held: Held-out subject ids.
    :param by: Eligible carriers per subject (strong or weak).
    :param vocab: ``subject_vocab.json``.
    :returns: ``subject -> {linked_to, why}``.
    :rtype: Dict[str, dict]
    """
    out: Dict[str, dict] = {}
    for sid, meta in vocab.items():
        if sid in held or sid in LINK_EXCLUDE:
            continue
        recs = by.get(sid, set())
        label = " | ".join([sid] + (meta.get("names_en") or []) + (meta.get("names_he") or []))
        for h in held:
            why = []
            if recs and len(recs & by[h]) / len(recs) >= LINK_OVERLAP:
                why.append(f"overlap {len(recs & by[h]) / len(recs):.2f}")
            m = re.search(CONCEPT_PATTERNS[h], label, re.I)
            if m:
                why.append(f"name match '{m.group(0)}'")
            if why:
                entry = out.setdefault(sid, {"linked_to": [], "why": [], "n_eligible": len(recs)})
                entry["linked_to"].append(h)
                entry["why"].append(f"{h}: {', '.join(why)}")
    return out


def assert_no_names(queries: Dict[str, dict], vocab: dict) -> None:
    """Raise if an llm_t2 query contains its subject's tag word or a catalogue synonym.

    :param queries: :data:`PGP_QUERIES`.
    :param vocab: ``subject_vocab.json``.
    """
    for sid, langs in queries.items():
        meta = vocab[sid]
        names = [n.lower() for n in (meta.get("names_en") or []) + (meta.get("names_he") or []) + (meta.get("raw_tags") or [])
                 if len(n) >= 3]
        for q in langs["en"] + langs["he"]:
            bad = [n for n in names if n in q.lower()]
            if bad:
                raise ValueError(f"{sid}: query {q!r} names the subject ({bad})")


# --------------------------------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------------------------------
def known_items(rounds: Iterable[str], pools: Dict[str, str], marks: Dict[str, str]) -> List[dict]:
    """Eval-pool synthetic queries (English ``out`` + Hebrew ``out_he``) as known-item queries.

    :param rounds: Round names.
    :param pools: ``doc_id -> pool``.
    :param marks: ``doc_id -> shelf mark`` (to flag queries that name it).
    :returns: Rows ``{qid, doc_id, type, lang, text, round, shelfmark_query}``.
    :rtype: List[dict]
    """
    out = []
    for rnd in rounds:
        base = AUDIT_ROOT / "synthetic_queries" / rnd
        for sub in ("out", "out_he"):
            for f in sorted(glob.glob(str(base / sub / "*.jsonl"))):
                for o in iter_jsonl(Path(f)):
                    if pools.get(o["doc_id"]) != "eval":
                        continue
                    sm = re.sub(r"\W", "", marks.get(o["doc_id"], "").lower())
                    for i, q in enumerate(o.get("queries") or []):
                        text = (q.get("text") or "").strip()
                        if not text:
                            continue
                        own = len(sm) >= 6 and sm in re.sub(r"\W", "", text.lower())
                        out.append({"qid": f"{rnd}:{o['doc_id']}:{'he' if sub == 'out_he' else ''}{i}",
                                    "doc_id": o["doc_id"], "type": q.get("type"), "lang": q.get("lang"), "text": text,
                                    "round": rnd, "shelfmark_query": bool(own or SHELFMARK_RE.search(text))})
    return out


def memorization_pairs(seen: List[str], strong: Dict[str, Set[str]], held: Set[str], held_subject_recs: Set[str],
                       rows: Dict[str, dict]) -> Dict[str, dict]:
    """Size-matched train-side vs held-out carriers of each seen subject.

    Held-out side = records held out for split / gold / eval-pool / duplicate-closure reasons only (not because they
    carry a held-out or linked subject, which would bias the sample towards those topics).

    :param seen: Seen subject ids.
    :param strong: Strong eligible carriers per subject.
    :param held: All held-out doc ids.
    :param held_subject_recs: Records held out because of a held-out or linked subject.
    :param rows: Semantic corpus rows.
    :returns: ``subject -> {n, train, held_out}``.
    :rtype: Dict[str, dict]
    """
    out = {}
    for sid in seen:
        recs = sorted(strong.get(sid, set()))
        tr = [d for d in recs if d not in held]
        he = [d for d in recs if d in held and d not in held_subject_recs]
        n = min(len(tr), len(he), MEMO_MAX)
        if n < MEMO_MIN:
            continue
        rng = random.Random(f"memo:{sid}")
        out[sid] = {"n": n, "train": sorted(rng.sample(tr, n)), "held_out": sorted(rng.sample(he, n))}
    return out


def catalogue_lines(text: str) -> str:
    """The catalogue's own description lines of a record (``Description:`` / ``Alternate titles:``).

    :param text: Folded semantic_v3 text (:func:`catalogue_text`), so continuation lines are part of their field.
    :returns: Those lines joined by newlines (transcriptions and metadata left out).
    :rtype: str
    """
    return "\n".join(ln for ln in text.split("\n") if ln.startswith(CATALOGUE_PREFIXES))


def concept_mentions(rows: Dict[str, dict], subj: Dict[str, Set[str]], held_subjects: List[str]) -> Dict[str, Set[str]]:
    """Eligible records NOT labelled with a held-out subject whose catalogue lines name its concept.

    Subject labels are incomplete (only about half of the PGP records carry any tag, and KTIV frames miss records
    such as "Nissim Gaon on the laws of lulav"), so these records are both held out of training (they would teach
    the concept) and *unjudged* for that subject in the eval (neither relevant nor irrelevant). Transcription text is
    not searched: a Talmud page quoting "סוכה" in passing is not about Sukkot. The ``mentions`` of ``subjects_v1``
    generalise this to every subject (:func:`subject_relevance`); this pattern pass is kept for the held-out / focus
    subjects because it also decides which records v3 held out.

    :param rows: Semantic rows (``catalogue`` parsed from the folded semantic_v3 text, :func:`load_semantic`).
    :param subj: ``doc_id -> subjects``.
    :param held_subjects: Held-out subject ids (keys of :data:`CONCEPT_PATTERNS`).
    :returns: ``held-out subject -> doc ids``.
    :rtype: Dict[str, Set[str]]
    """
    rx = {h: re.compile(CONCEPT_PATTERNS[h], re.I) for h in held_subjects}
    out: Dict[str, Set[str]] = {h: set() for h in held_subjects}
    for d, row in rows.items():
        if not row["eligible"]:
            continue
        for h, r in rx.items():
            if h not in subj[d] and r.search(row["catalogue"]):
                out[h].add(d)
    return out


def pattern_counts(held_ids: Set[str], rows: Dict[str, dict], rounds: Iterable[str], pools: Dict[str, str],
                   semantic: Path) -> Dict[str, dict]:
    """How many training anchors / train-side texts name each held-out concept (for T3's leak filter).

    :param held_ids: Held-out doc ids.
    :param rows: Semantic corpus rows.
    :param rounds: Synthetic query rounds.
    :param pools: ``doc_id -> pool``.
    :param semantic: ``corpus_semantic.jsonl`` (texts are streamed again).
    :returns: ``held-out subject -> {train_pool_anchor_hits, train_side_text_hits}``.
    :rtype: Dict[str, dict]
    """
    rx = {h: re.compile(p, re.I) for h, p in CONCEPT_PATTERNS.items()}
    out = {h: Counter() for h in CONCEPT_PATTERNS}
    for rnd in rounds:
        base = AUDIT_ROOT / "synthetic_queries" / rnd
        for sub in ("out", "out_he"):
            for f in sorted(glob.glob(str(base / sub / "*.jsonl"))):
                for o in iter_jsonl(Path(f)):
                    if pools.get(o["doc_id"]) != "train" or o["doc_id"] in held_ids:
                        continue
                    for q in o.get("queries") or []:
                        for h, r in rx.items():
                            out[h]["train_pool_anchor_hits"] += bool(r.search(q.get("text") or ""))
    for row in iter_jsonl(semantic):
        if row["eligible"] and row["doc_id"] not in held_ids:
            for h, r in rx.items():
                out[h]["train_side_text_hits"] += bool(r.search(row["text"]))
    return {h: dict(c) for h, c in out.items()}


README = """# Frozen evaluation set v3 (semantic text embedder)

Built by `evals/embedding_audit/build_eval_v3.py` on {built}; scored by `evals/embedding_audit/eval_v3.py`
(pure numpy, importable on Colab). Inputs: `v3/corpus_semantic.jsonl` (semantic text: no Document ID / Shelf Mark
lines, no editor credits, shelf marks / PGPIDs cited inside other lines masked as `[shelfmark]`, `eligible` flag,
`text_hash`), `v3/subjects_v1.jsonl` + `v3/subject_vocab.json`
(subject labels), `concept_probe_v1.json`, and the synthetic query rounds `synthetic_queries/{{r1,r2}}`.

## Principle

The owner wants subjects close together ("laws of lulav" near Sukkot records; Hebrew and English names of the same
thing together) and does NOT want literal text-to-record or shelf-mark matching. The index keeps growing, so the
only gain that counts is one that generalises: the **gate** scores subjects whose records the model never saw in
training, and every metric ranks the full pool of eligible records.

**This split is for the EVALUATION run only.** Once the training recipe is proven on it, the production model is
retrained on everything (all records and all subjects, including the held-out ones), and is not re-scored on this
gate.

## Files

| file | content |
|---|---|
| `held_out_subjects.json` | The {n_held_subjects} held-out subjects (`subjects`: id, reason, record counts, query counts, concept pattern), the `linked` subjects held out with them (same concept under another label: e.g. `work:talmud-bavli-sukkah`, `sefaria:laws-of-the-sukkah`, `pgp:manumission`), the candidate tables the choice was made from (`selection`), and `pattern_counts`: how many train-side synthetic anchors / record texts still name each held-out concept. |
| `held_out_records.json` | `held_out`: every record that must never be an anchor or positive in training ({n_held_records} of {n_records}; {n_held_eligible} of {n_eligible} eligible). `by_reason`: `split` (20 % test side), `held_out_subject`, `linked_subject`, `concept_mention` (not labelled, but its Description / Alternate titles name a held-out concept, e.g. "Nissim Gaon on the laws of lulav"), `sukkot_gold`, `eval_pool`, `dup_closure` (identical text to a held-out record). `old_split_trained`: records held out here but on the train side of the v1/v2 rule (a v2-trained model saw them; pass them as `ignore_ids` when scoring such a model). |
| `subject_queries.json` | `subjects[id] = {{status, kind, queries[{{text, source, lang, rel_mode}}]}}`. status = `held_out` / `linked` / `seen`. source = `probe` (concept_probe_v1.json, topics), `llm_t2` (8 EN + 4 HE implicit queries written for the PGP hold-outs; they avoid the tag word and its synonyms), `name` (the subject's English and Hebrew names from subject_vocab, up to {cap_en} EN + {cap_he} HE). rel_mode = `law` for law-name forms (`build_subjects.is_law_name`: "laws of", "hilkhot", "הלכות", "tractate", "מסכת" ...) and every probe / llm_t2 query, else `bare`. |
| `subject_relevance.json` | `{{id: {{strong, weak, canon, unjudged}}}}`, eligible records only. `weak` = the label came only from a KTIV general title (free text, ~90 % precise), or a festival named only on a scripture / tractate / law-code frame or incidentally (letters, legal records); `canon` = carriers whose subjects_v1 `canon` lists the subject (named on a scripture / tractate / law-code frame or description: weak for festivals, strong otherwise); `unjudged` = records whose subjects_v1 `mentions` name the subject without a strong label, title / incidental weak carriers, and (held-out subjects) unlabelled records whose catalogue lines name the held-out concept (labels are incomplete: PGP tags cover about half of PGP records). Unjudged and non-relevant weak records are left out of that subject's ranking; canon carriers are misses for `bare` queries and relevant for `law` queries. |
| `holiday_queries.json` | `{{festivals: {{id: {{queries: [{{text, form}}], relevant, canon, unjudged, alt_only, held_out_relevant}}}}}}`: every name form (english / translit / hebrew / hebrew_prefixed / ashkenazi / typo / descriptive) of ten festivals, scored by `eval_v3`'s `holiday` section over the WHOLE pool (trained names: a regression check, not generalisation). Relevant = strong carriers; canon carriers are misses, so "Purim" -> Esther scores 0; unjudged (title-only, incidental, mention-only, Sukkot-gold records that are canon or incidental) are removed from the ranking. Reports `canon_share@20` and the alt-only share (relevant hits that name the festival only under Alternate titles, parsed from the semantic_v3 text) with its base rate and lift; `held_out_relevant` drives the held-out-only variant. |
| `known_item.jsonl` | {n_known} eval-pool synthetic queries (r1 `out` + `out_he`, pool == eval): `{{qid, doc_id, type, lang, text, round, shelfmark_query}}`. Types: implicit, topical, hebrew, specific, he_implicit, he_topical, translit. `shelfmark_query` flags queries naming a classmark (skipped by default). Not yet filtered by Isaac's review verdicts. |
| `memorization_pairs.json` | For {n_memo} seen subjects: `train` and `held_out` samples of equal size n (5 to {memo_max}) of the subject's carriers. Held-out side excludes records held out only because they carry a held-out/linked subject. |
| `collection_sample.json` | {n_coll} eligible records with unique text and a known collection series (seed 0), for the collection-lift metric. |
| `eval_pool.jsonl` | The frozen candidate pool: every eligible record `{{doc_id, text_hash, series, n_chars, dup_n, held_out}}` ({n_eligible} rows; plus `text_hash_v3` = md5 of the semantic_v3 text when the corpus has it). `dup_n` = eligible records sharing the text. |
| `build_stats.json` | Counts printed by the builder. |

## Metrics (`eval_v3.evaluate(doc_ids, doc_matrix, query_vectors_fn, eval_dir, corpus_rows)`)

Document vectors must be of the v3 **semantic** text (pass `corpus_rows` to have the text hashes checked).
`query_vectors_fn(list_of_texts) -> (n, d)` must add the model's query instruction itself (production uses the
Qwen3 "Instruct: Given a search query, retrieve relevant passages" prefix). Ties are always broken pessimistically.

* **Subject retrieval** (`subjects`). Every query ranks the whole eligible pool. Relevant = strong carriers of the
  subject (`bare` queries: minus canon carriers, which stay as misses; `law` queries: plus canon carriers); weak-only
  carriers and unjudged records are removed from that ranking. Per query: AP, P@10, R@100; mean per subject; macro
  over subjects. Reported for `held_out` (**the gate**: `subjects.gate`, also `macro_by_source` = probe / llm_t2 /
  name), `linked` and `seen` (with `macro_by_kind`). Subjects with fewer than 5 relevant pool records are skipped.
* **Paired comparison** (`compare(res_a, res_b, group, metric)`): two-stage bootstrap (resample subjects, then the
  paired per-query differences inside each), 1,000 resamples, 95 % CI of the macro difference. With 6 held-out
  subjects the interval is wide on purpose. `group="known_item"` clusters by target record.
* **Memorisation gap** (`memorisation`): for seen subjects, AP of the subject's name queries when the positives are the
  train-side sample vs the equal-size held-out sample (negatives identical: all non-carriers). `gap = AP_train -
  AP_held` with a subject-clustered bootstrap CI. A model that learns subjects has a gap near 0; one that memorises
  its training records has a large gap.
* **Known item** (`known_item`): rank of the target among the pool; identical-text twins count as hits; ties count
  against. MRR, R@1/10/100, median rank by query type and overall; `within_subject` = the target's rank among the
  carriers of each of its subjects (subjects with >= 10 carriers): can the model still pick the record out from its
  subject neighbours?
* **Collection lift** (`collection`): share of a sample record's top-10 neighbours from its own collection series
  (doc-id tokens before the first number, e.g. `Cambridge_CUL_T_S`) divided by the chance share sum(p_series^2)
  within the sample (`lift`; `lift_vs_pool` uses the pool's series shares). 1.0 = no clustering by collection;
  reported for all 3,000, for records >= {long_chars} chars, and by train/held-out side.
* **Hubness** (`hubness`): share of all subject queries' top-10 slots taken by records in duplicate-text groups or
  shorter than {short_chars} chars, vs that share of the pool (`ratio`), plus the 10 most frequent hub records.
* **Holiday** (`holiday`): P@10 / P@20 / MRR@100 / `canon_share@20` per name form, festival and overall (worst
  festival by P@10); alt-only share, base rate and lift weighted by hits; `held_out` = the same with relevant =
  held-out relevant records, train-side relevant records removed. `compare(..., group="holiday")` clusters by festival.

## Rules for the training steps (T3/T4)

* Never use a record in `held_out_records.json["held_out"]` as an anchor or a positive (nor its text as a negative
  source that is labelled by subject).
* Never use any query in `subject_queries.json` whose status is `held_out` or `linked` as a training anchor, and drop
  non-record pairs (Sefaria topic pairs, scholarship pages) for linked Sefaria slugs; filter remaining anchors with
  `held_out_subjects.json["concept_patterns"]` (counts of what that removes are in `pattern_counts`).
* Seen-subject names may be used as training labels; the memorisation gap measures exactly that.
"""

README_FINAL = """# Frozen evaluation set v3-final (semantic text embedder, FINAL run)

Built by `evals/embedding_audit/build_eval_v3.py --final` on {built}; scored by `evals/embedding_audit/eval_v3.py`.
Same inputs, split rule, seeds and query sets as the v3 eval set (`v3/eval`, see its README).

## What changed vs v3

The v3 run held six subjects out of training to prove the recipe generalises (`topic:sukkot`, `topic:pesach`,
`topic:kashrut`, `pgp:partnership`, `pgp:slavery`, `pgp:geonic-academies`, plus {n_v3_linked} linked subjects). The
FINAL run trains on all of them ("these are important topics"). So here:

* **No subject is held out**: `held_out_subjects.json` has empty `subjects`, `linked` and `concept_patterns`; there is
  no gate.
* **v3 comparison** (`build_stats.json["vs_v3"]`, mode `{v3_mode}`): in `strict` mode the checks below hold; in
  `report` mode (a newer corpus) {n_new_held} held-out records are not in v3's held-out set and {n_v3_trainable_held}
  of them were v3-trainable (pass a v3-final-trained model's trained ids as `ignore_ids` when scoring it).
* **Held-out records** ({n_held_records} of {n_records}; {n_held_eligible} of {n_eligible} eligible) = the v3 20 % split
  (`split`, grouped by text hash) + `sukkot_gold` + synthetic-query `eval_pool`, closed over identical text
  (`dup_closure`). They never appear on either side of a training pair. This set is a subset of v3's held-out set and
  its split / gold / eval-pool members are identical, so a v3-trained model (v3-run1) never saw any of them either.
  {n_moved} records that v3 held out only for subject reasons move to the train side (`build_stats.json["vs_v3"]`).
* **Focus group**: the six former held-out subjects have status `focus` in `subject_queries.json` (same probe /
  llm_t2 / name queries as v3) and are described in `focus_subjects.json`. `eval_v3` scores them over HELD-OUT
  records only: relevant = strong carriers that are held out; train-side carriers, weak-only carriers and the
  unjudged concept mentions are left out of the ranking (neither relevant nor irrelevant). Results:
  `subjects.focus` (macro, `macro_by_source` = probe / llm_t2 / name, per subject with `by_source`); it is the primary
  comparison (`subjects.primary_group == "focus"`).
* No focus query (probe, llm_t2, name) or known-item query is a training anchor or a near-copy of one, on either side
  of a pair (the T3 leak filter: exact, also ignoring spacing inside words; containment of queries of >= 3 tokens,
  and of two-token focus probe / llm_t2 queries inside anchors; content Jaccard >= 0.5). One-word focus queries
  ("Haggadah", "סוכות") do occur inside longer anchors and record texts, because the six subjects are trained on.
  Read the `name` source as near-in-distribution: the focus name queries themselves are blocked, but spelling
  variants of them are training labels of the same subjects ("Succos", "Booths", "חג פסח" vs the queries "Sukkos",
  "Tabernacles", "חג הפסח"); `probe` and `llm_t2` are the paraphrase test.

## Files

Same layout as v3 (`held_out_records.json`, `subject_queries.json` with `rel_mode`, `subject_relevance.json` with
`canon` / `unjudged`, `holiday_queries.json`, `known_item.jsonl`, `memorization_pairs.json`, `collection_sample.json`,
`eval_pool.jsonl`, `build_stats.json`; see the v3 README written by `build_eval_v3.py`) plus `focus_subjects.json`
({n_focus} subjects: counts of held-out positives / train-side carriers, queries, v3 reason, the v3 linked subjects,
the concept pattern used for the unjudged set). `held_out_records.json["by_reason"]` keeps only `split`,
`sukkot_gold`, `eval_pool`, `dup_closure`. Memorisation pairs: {n_memo} seen subjects (focus subjects excluded, as
held-out subjects were in v3). Known items: {n_known} queries (identical to v3). Collection sample: {n_coll} records
(identical to v3).

## Rules for the training steps

* Never use a record in `held_out_records.json["held_out"]` as an anchor or a positive, nor a record whose text
  equals a held-out record's (byte-identical, or up to niqqud / case / punctuation / spacing: `--final-strict`).
* Keep every focus-subject query (probe, llm_t2, name), every concept_probe_v1 query and every known-item query out
  of anchors and positives (fuzzy leak filter of `build_training_mix_v3.py`, see above).
* Every subject (focus included) may otherwise be trained on.
"""


def focus_entries(focus: List[str], vocab: dict, by: Dict[str, Set[str]], strong: Dict[str, Set[str]],
                  held: Set[str], rows: Dict[str, dict], mentions: Dict[str, Set[str]], subjects_q: Dict[str, dict],
                  v3_linked: Dict[str, dict], relevance: Dict[str, dict]) -> List[dict]:
    """``focus_subjects.json`` rows (final mode): sizes, held-out positives, queries, v3 links.

    :param focus: Focus subject ids.
    :param vocab: ``subject_vocab.json``.
    :param by: Eligible carriers per subject (strong or weak).
    :param strong: Strong eligible carriers per subject.
    :param held: Held-out doc ids.
    :param rows: Semantic corpus rows.
    :param mentions: Unjudged concept mentions per focus subject.
    :param subjects_q: Subject query entries.
    :param v3_linked: Subjects that v3 held out as linked (``find_linked`` on the six).
    :param relevance: ``subject_relevance.json`` content (canon / unjudged sets).
    :returns: One dict per focus subject.
    :rtype: List[dict]
    """
    out = []
    for sid in focus:
        st = strong.get(sid, set())
        out.append({
            "id": sid, "kind": vocab[sid]["kind"],
            "v3_reason": HELD_OUT_TOPICS[sid][1] if sid in HELD_OUT_TOPICS else HELD_OUT_PGP[sid],
            "names_en": vocab[sid].get("names_en"), "names_he": vocab[sid].get("names_he"),
            "n_eligible": len(by.get(sid, ())), "n_strong_eligible": len(st),
            "n_weak_eligible": len(by.get(sid, set()) - st),
            "n_strong_held_out": len(st & held), "n_strong_train_side": len(st - held),
            "n_gold": sum(rows[d]["sukkot_gold"] for d in by.get(sid, ())),
            "n_unjudged_concept_mentions": len(mentions[sid]),
            "n_unjudged_held_out": len(mentions[sid] & held),
            "n_canon_eligible": len(relevance[sid]["canon"]), "n_unjudged_eligible": len(relevance[sid]["unjudged"]),
            "n_queries_by_source": dict(Counter(q["source"] for q in subjects_q[sid]["queries"])),
            "n_queries_by_rel_mode": dict(Counter(q["rel_mode"] for q in subjects_q[sid]["queries"])),
            "queries": subjects_q[sid]["queries"],
            "v3_linked_subjects": sorted(s for s, v in v3_linked.items() if sid in v["linked_to"]),
            "concept_pattern": CONCEPT_PATTERNS[sid],
        })
    return out


def compare_with_v3(v3_dir: Path, held: Set[str], by_reason: Dict[str, List[str]], eligible: Set[str],
                    subj: Dict[str, Set[str]], focus: List[str], v3_linked: Dict[str, dict],
                    strict: bool = True) -> dict:
    """Check the final held-out set against v3's and count what moved to the train side.

    ``strict`` (the v3 corpus): the final set must be a subset of v3's with identical split / gold / eval-pool
    members. Report mode (a newer corpus: new records, and text regrouping that moves identical-text groups across
    the split) only counts the differences; ``v3_trainable_now_held_out`` are records a v3-final-trained model (v9)
    could have trained on that are now held out, which is why v9 is scored with its trained ids ignored.

    :param v3_dir: The v3 eval dir (``held_out_records.json``, ``eval_pool.jsonl``).
    :param held: Final held-out doc ids.
    :param by_reason: Final ``reason -> doc ids``.
    :param eligible: Eligible doc ids.
    :param subj: ``doc_id -> subjects``.
    :param focus: Focus subject ids.
    :param v3_linked: v3 linked subjects.
    :param strict: Raise on a difference (True) or only report it.
    :returns: Counts (``moved_*``) and the equality checks.
    :rtype: dict
    :raises ValueError: In strict mode, when the final set is not a subset of v3's or a shared reason's members
        differ.
    """
    v3 = json.loads((v3_dir / FILES["held_out_records"]).read_text(encoding="utf-8"))
    v3_held = set(v3["held_out"])
    v3_pool = {r["doc_id"] for r in iter_jsonl(v3_dir / FILES["eval_pool"])}
    subset = held <= v3_held
    same = {key: set(by_reason[key]) == set(v3["by_reason"][key]) for key in ("split", "sukkot_gold", "eval_pool")}
    if strict and not subset:
        raise ValueError(f"{len(held - v3_held)} final held-out records were on v3's train side")
    for key, ok in same.items():
        if strict and not ok:
            raise ValueError(f"reason {key!r} differs from v3")
    moved = (v3_held - held) & set(subj)  # v3 records gone from this corpus did not move anywhere
    gone = (v3_held - held) - set(subj)
    v3_reasons: Dict[str, Set[str]] = defaultdict(set)
    for k, ids in v3["by_reason"].items():
        for d in ids:
            v3_reasons[d].add(k)
    focus_set, linked_set = set(focus), set(v3_linked)
    new_held = held - v3_held
    return {
        "v3_dir": str(v3_dir), "mode": "strict" if strict else "report", "v3_n_held_out": len(v3_held),
        "final_n_held_out": len(held), "final_subset_of_v3": subset, "same_split_gold_eval_pool": all(same.values()),
        "same_by_reason": same, "final_held_not_v3_held": len(new_held),
        "final_held_not_v3_held_eligible": len(new_held & eligible),
        "v3_trainable_now_held_out": len(new_held & (v3_pool - v3_held)),
        "split_added_vs_v3": len(set(by_reason["split"]) - set(v3["by_reason"]["split"])),
        "split_removed_vs_v3": len(set(v3["by_reason"]["split"]) - set(by_reason["split"])),
        "dup_closure_v3": len(v3["by_reason"]["dup_closure"]), "dup_closure_final": len(by_reason["dup_closure"]),
        "moved_to_train": len(moved), "moved_to_train_eligible": len(moved & eligible),
        "moved_by_v3_reason": dict(sorted(Counter(k for d in moved for k in v3_reasons[d]).items())),
        "v3_held_gone_from_corpus": len(gone),
        "moved_carrying_focus_subject": sum(bool(subj[d] & focus_set) for d in moved),
        "moved_carrying_v3_linked_subject": sum(bool(subj[d] & linked_set) for d in moved),
        "moved_by_focus_subject": {s: sum(s in subj[d] for d in moved) for s in focus},
    }


def main() -> None:
    """Build every eval file and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semantic", default=str(V3 / "corpus_semantic.jsonl"))
    parser.add_argument("--subjects", default=str(V3 / "subjects_v1.jsonl"))
    parser.add_argument("--vocab", default=str(V3 / "subject_vocab.json"))
    parser.add_argument("--corpus-v1", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--merged", default=str(MERGED))
    parser.add_argument("--rounds", default="r1,r2")
    parser.add_argument("--final", action="store_true",
                        help="final-run eval set: no held-out subjects, the six reported as the focus group "
                             "(default --out-dir v3/eval_final)")
    parser.add_argument("--v3-eval-dir", default=str(V3 / "eval"),
                        help="final mode: the v3 eval dir the held-out set is checked against")
    parser.add_argument("--v3-compare", choices=("auto", "strict", "report"), default="auto",
                        help="final mode: 'strict' raises unless the held-out set is a subset of v3's with identical "
                             "split / gold / eval-pool members (the v3 corpus); 'report' only counts the differences "
                             "(a newer corpus: new records, regrouped identical texts); 'auto' = strict when every "
                             "corpus row is semantic_v3 text (no text_version, or semantic_v3), else report")
    parser.add_argument("--out-dir", default=None, help="default v3/eval (v3/eval_final with --final)")
    args = parser.parse_args()
    final = args.final
    out_dir = Path(args.out_dir or (V3 / ("eval_final" if final else "eval")))
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = args.rounds.split(",")

    rows = load_semantic(Path(args.semantic))
    labels = load_subjects(Path(args.subjects))
    subj, weak = labels.subj, labels.weak
    text_versions = Counter(r.get("text_version") or "semantic_v3 (unlabelled)" for r in rows.values())
    eligible_rows = [r for r in rows.values() if r["eligible"]]
    catalogue_counts = {"eligible_with_description": sum(bool(r["desc"]) for r in eligible_rows),
                        "eligible_with_alt_titles": sum(bool(r["alt"]) for r in eligible_rows),
                        "eligible_with_document_type": sum(bool(r["doc_type"]) for r in eligible_rows),
                        "field_source": "text_v3" if any("text_hash_v3" in r for r in rows.values()) else "text"}
    if not catalogue_counts["eligible_with_description"] or not catalogue_counts["eligible_with_alt_titles"]:
        raise ValueError(f"catalogue fields not parsed from the semantic_v3 text: {catalogue_counts}")
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    types = pgp_types(list(rows), Path(args.merged))
    by = carriers(subj, rows)
    by_all = carriers(subj, rows, eligible_only=False)
    strong = {s: {d for d in recs if s not in weak[d]} for s, recs in by.items()}

    # ---- held-out subjects (re-check the selection criteria on today's data) ----
    topic_rows = probe_topic_table(by, weak, probe)
    pgp_rows = pgp_table(by, subj, types, vocab)
    for row in pgp_rows:
        if row["chosen"] and not row["passes"]:
            raise ValueError(f"held-out PGP subject {row['subject']} no longer meets the criteria: {row['fails']}")
    for sid in HELD_OUT_TOPICS:
        if len(strong.get(sid, ())) < 100:
            raise ValueError(f"held-out topic {sid} has fewer than 100 strong eligible records")
    assert_no_names(PGP_QUERIES, vocab)
    # final mode: nothing is held out by subject; the same six subjects become the "focus" group
    held_subjects = [] if final else list(FOCUS_SUBJECTS)
    focus = list(FOCUS_SUBJECTS) if final else []
    linked = find_linked(held_subjects, by, vocab) if held_subjects else {}
    v3_linked = find_linked(FOCUS_SUBJECTS, by, vocab) if final else linked  # final: reported, not held out

    # ---- held-out records ----
    pools = synthetic_pools(rounds)
    side = split_side(rows)
    # concept mentions: v3 holds them out of training; final mode only keeps them unjudged for the focus subjects
    mentions = concept_mentions(rows, subj, held_subjects + focus)
    relevance = subject_relevance(vocab, rows, labels, strong, by, {h: mentions[h] for h in held_subjects + focus})
    mention_any = set().union(*mentions.values()) if held_subjects else set()
    reasons: Dict[str, Set[str]] = defaultdict(set)
    for d in rows:
        if side[d]:
            reasons[d].add("split")
        if rows[d]["sukkot_gold"]:
            reasons[d].add("sukkot_gold")
        if pools.get(d) == "eval":
            reasons[d].add("eval_pool")
        if subj[d] & set(held_subjects):
            reasons[d].add("held_out_subject")
        if subj[d] & set(linked):
            reasons[d].add("linked_subject")
        if d in mention_any and not subj[d] & set(held_subjects):
            reasons[d].add("concept_mention")
    groups: Dict[str, List[str]] = defaultdict(list)
    for d, r in rows.items():
        if r["eligible"]:
            groups[r["text_hash"]].append(d)
    for members in groups.values():
        if len(members) > 1 and any(reasons.get(d) for d in members):
            for d in members:
                if not reasons.get(d):
                    reasons[d].add("dup_closure")
    held = {d for d, rs in reasons.items() if rs}
    held_subject_recs = {d for d in held if reasons[d] & {"held_out_subject", "linked_subject", "concept_mention"}}
    old_held = {d for d in rows if is_test(d) or rows[d]["sukkot_gold"] or pools.get(d) == "eval"}
    reason_keys = ("split", "sukkot_gold", "eval_pool", "dup_closure") if final else \
        ("split", "held_out_subject", "linked_subject", "concept_mention", "sukkot_gold", "eval_pool", "dup_closure")
    by_reason = {k: sorted(d for d in held if k in reasons[d]) for k in reason_keys}
    eligible = {d for d, r in rows.items() if r["eligible"]}
    flips = sum(side[d] != is_test(d) for d in rows)
    held_records = {
        "built": date.today().isoformat(),
        "rule": ("FINAL run (no subject held out): held out = 20% split (md5(doc_id) % 5 == 0; an eligible "
                 "identical-text group follows its smallest doc_id) OR Sukkot gold OR synthetic-query eval pool; then "
                 "closed over eligible identical-text groups") if final else
                "held out = 20% split (md5(doc_id) % 5 == 0; an eligible identical-text group follows its smallest "
                "doc_id) OR carries a held-out/linked subject (strong or weak) OR its catalogue lines name a held-out "
                "concept (concept_patterns) OR Sukkot gold OR synthetic-query eval pool; then closed over eligible "
                "identical-text groups",
        "n_records": len(rows), "n_held_out": len(held), "n_eligible": len(eligible),
        "n_eligible_held_out": len(held & eligible),
        "n_by_reason": {k: len(v) for k, v in by_reason.items()},
        "split_flips_vs_old_rule": flips,
        "old_split_trained": sorted(held - old_held),
        "old_split_test_now_train": len(old_held - held),
        "held_out": sorted(held), "by_reason": by_reason,
    }

    # ---- queries ----
    subjects_q: Dict[str, dict] = {}
    special = held_subjects + focus
    for sid in special + sorted(linked) + sorted(s for s in vocab if s not in special and s not in linked):
        meta = vocab[sid]
        status = "held_out" if sid in held_subjects else "focus" if sid in focus else "linked" if sid in linked \
            else "seen"
        queries = []
        if sid in HELD_OUT_TOPICS:
            queries += [{"text": q, "source": "probe", "lang": "he" if HEB.search(q) else "en"}
                        for q in probe["concepts"][HELD_OUT_TOPICS[sid][0]]["queries"]]
        if sid in PGP_QUERIES:
            queries += [{"text": q, "source": "llm_t2", "lang": lang}
                        for lang in ("en", "he") for q in PGP_QUERIES[sid][lang]]
        queries += [{"text": n, "source": "name", "lang": "he" if HEB.search(n) else "en"} for n in subject_names(meta)]
        seen_q = set()
        queries = [q for q in queries if not (q["text"] in seen_q or seen_q.add(q["text"]))]
        for q in queries:
            q["rel_mode"] = rel_mode_of(q["text"], q["source"])
        subjects_q[sid] = {"status": status, "kind": meta["kind"], "n_strong_eligible": len(strong.get(sid, ())),
                           "queries": queries}

    # ---- held-out subject entries ----
    held_entries = []
    for sid in held_subjects:
        recs = by.get(sid, set())
        src = Counter(q["source"] for q in subjects_q[sid]["queries"])
        overlap = Counter(x for d in recs for x in subj[d] if x != sid and x not in linked)
        held_entries.append({
            "id": sid, "kind": vocab[sid]["kind"],
            "reason": HELD_OUT_TOPICS[sid][1] if sid in HELD_OUT_TOPICS else HELD_OUT_PGP[sid],
            "names_en": vocab[sid].get("names_en"), "names_he": vocab[sid].get("names_he"),
            "n_records": len(by_all.get(sid, ())), "n_eligible": len(recs), "n_strong_eligible": len(strong.get(sid, ())),
            "n_weak_eligible": len(recs) - len(strong.get(sid, ())),
            "n_gold": sum(rows[d]["sukkot_gold"] for d in recs), "n_queries_by_source": dict(src),
            "n_unjudged_concept_mentions": len(mentions[sid]),
            "linked_subjects": sorted(s for s, v in linked.items() if sid in v["linked_to"]),
            "top_seen_overlaps": overlap.most_common(8), "concept_pattern": CONCEPT_PATTERNS[sid],
        })
    held_ids = set(held)
    if final:
        held_subjects_doc = {
            "built": date.today().isoformat(), "mode": "final", "subjects": [], "linked": {}, "concept_patterns": {},
            "focus_file": "focus_subjects.json",
            "note": "FINAL run: no subject is held out of training. The six v3 held-out subjects (and their v3 linked "
                    "subjects) are trained on; the six are scored as the 'focus' group over held-out records only.",
        }
        focus_doc = {
            "built": date.today().isoformat(),
            "rule": "focus subjects are trained on; eval_v3 scores them over HELD-OUT records only: relevant = strong "
                    "carriers in held_out_records.json (rel_mode bare: minus canon carriers, which stay in the "
                    "ranking as misses; rel_mode law: plus canon carriers); train-side relevant carriers, weak-only "
                    "carriers and unjudged records (subject_relevance[id]['unjudged']: concept mentions, subjects_v1 "
                    "mentions, title / incidental weak carriers) are left out of the ranking. Their queries "
                    "(probe / llm_t2 / name) are kept out of the training text by the T3 leak filter.",
            "subjects": focus_entries(focus, vocab, by, strong, held_ids, rows, mentions, subjects_q, v3_linked,
                                      relevance),
            "v3_linked": v3_linked,
        }
    else:
        held_subjects_doc = {
            "built": date.today().isoformat(), "subjects": held_entries, "linked": linked,
            "concept_patterns": CONCEPT_PATTERNS,
            "selection": {"probe_topics": {"criteria": ">= 100 strong eligible records and >= 8 probe queries; one "
                                                       "festival with low Sukkot overlap + one non-calendar halakhic "
                                                       "topic; topics whose PGP twin stays seen are rejected",
                                           "candidates": topic_rows},
                          "pgp": {"criteria": {**PGP_CRITERIA, "excluded_categories": sorted(PGP_EXCLUDED_CATEGORIES),
                                               "rule": "one subject per category among those that pass; prefer clear, "
                                                       "non-overlapping concepts"},
                                  "candidates": pgp_rows, "rejected_notes": PGP_REJECTED}},
            "pattern_counts": pattern_counts(held_ids, rows, rounds, pools, Path(args.semantic)),
        }

    # ---- known items, memorisation, collection sample, pool ----
    # synthetic-query eval records that are gone from this corpus (ids dropped by a later merged file) cannot be held
    eval_docs = {d for d, p in pools.items() if p == "eval" and d in rows}
    eval_docs_missing = sorted(d for d, p in pools.items() if p == "eval" and d not in rows)
    ki = [q for q in known_items(rounds, pools, shelfmarks(Path(args.corpus_v1), eval_docs)) if q["doc_id"] in rows]
    seen = [s for s, v in subjects_q.items() if v["status"] == "seen"]
    memo = memorization_pairs(seen, strong, held, held_subject_recs, rows)
    dup_n = Counter(rows[d]["text_hash"] for d in eligible)
    coll_cands = sorted(d for d in eligible if dup_n[rows[d]["text_hash"]] == 1 and rows[d]["series"])
    coll = sorted(random.Random(0).sample(coll_cands, COLLECTION_N))
    holiday_doc, holiday_counts = holiday_queries(rows, labels, strong, by, held, types)

    # ---- integrity checks ----
    for members in groups.values():
        assert len({d in held for d in members}) == 1, f"text group split across sides: {members[:3]}"
    assert eval_docs <= held and {d for d in rows if rows[d]["sukkot_gold"]} <= held
    assert all(d in held for s in held_subjects + list(linked) for d in by_all.get(s, ()))
    assert all(d not in held for m in memo.values() for d in m["train"])
    assert all(d in held for m in memo.values() for d in m["held_out"])
    assert all(q["doc_id"] in held for q in ki)
    assert mention_any <= held
    for s, r in relevance.items():
        assert not set(r["unjudged"]) & (set(r["strong"]) | set(r["canon"])), f"{s}: unjudged overlaps strong/canon"
    for s, f in holiday_doc["festivals"].items():
        assert not set(f["unjudged"]) & (set(f["relevant"]) | set(f["canon"])) and set(f["held_out_relevant"]) <= held
    vs_v3 = None
    if final:
        assert not set(memo) & set(focus) and all(subjects_q[s]["status"] == "focus" for s in focus)
        strict = args.v3_compare == "strict" or (
            args.v3_compare == "auto" and set(text_versions) <= {"semantic_v3", "semantic_v3 (unlabelled)"})
        vs_v3 = compare_with_v3(Path(args.v3_eval_dir), held, by_reason, eligible, subj, focus, v3_linked, strict)

    # ---- write ----
    def dump(key: str, obj) -> None:
        (out_dir / FILES[key]).write_text(json.dumps(obj, ensure_ascii=False, indent=1 if key == "held_out_subjects"
                                                     else None), encoding="utf-8")

    dump("held_out_subjects", held_subjects_doc)
    if final:
        held_records["vs_v3"] = vs_v3
        (out_dir / "focus_subjects.json").write_text(json.dumps(focus_doc, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    dump("held_out_records", held_records)
    dump("subject_queries", {"built": date.today().isoformat(), "subjects": subjects_q})
    dump("subject_relevance", relevance)
    (out_dir / HOLIDAY_FILE).write_text(json.dumps(holiday_doc, ensure_ascii=False), encoding="utf-8")
    dump("memorization_pairs", {"built": date.today().isoformat(), "max_n": MEMO_MAX, "min_n": MEMO_MIN, "subjects": memo})
    dump("collection_sample", {"seed": 0, "n": len(coll), "criteria": "eligible, unique text among eligible records, "
                                                                      "non-empty series", "doc_ids": coll})
    with open(out_dir / FILES["known_item"], "w", encoding="utf-8") as fh:
        for q in ki:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out_dir / FILES["eval_pool"], "w", encoding="utf-8") as fh:
        for d in rows:
            if d in eligible:
                r = rows[d]
                pool_row = {"doc_id": d, "text_hash": r["text_hash"], "series": r["series"], "n_chars": r["n_chars"],
                            "dup_n": dup_n[r["text_hash"]], "held_out": d in held}
                if "text_hash_v3" in r:  # md5 of the semantic_v3 (v9) text: hash check for models scored on it
                    pool_row["text_hash_v3"] = r["text_hash_v3"]
                fh.write(json.dumps(pool_row, ensure_ascii=False) + "\n")
    status_counts = Counter(v["status"] for v in subjects_q.values())
    src_counts = Counter((v["status"], q["source"]) for v in subjects_q.values() for q in v["queries"])
    mode_counts = Counter((v["status"], q["rel_mode"]) for v in subjects_q.values() for q in v["queries"])
    rel_counts = {"subjects_with_canon": sum(bool(r["canon"]) for r in relevance.values()),
                  "canon_pairs": sum(len(r["canon"]) for r in relevance.values()),
                  "subjects_with_unjudged": sum(bool(r["unjudged"]) for r in relevance.values()),
                  "unjudged_pairs": sum(len(r["unjudged"]) for r in relevance.values()),
                  "strong_pairs_also_canon": sum(len(set(r["strong"]) & set(r["canon"])) for r in relevance.values())}
    stats = {
        "built": date.today().isoformat(),
        "held_out_subjects": {e["id"]: {k: e[k] for k in ("n_eligible", "n_strong_eligible", "n_weak_eligible",
                                                          "n_unjudged_concept_mentions", "n_queries_by_source",
                                                          "linked_subjects")}
                              for e in held_entries},
        "n_linked_subjects": len(linked),
        "records": {k: held_records[k] for k in ("n_records", "n_held_out", "n_eligible", "n_eligible_held_out",
                                                 "n_by_reason", "split_flips_vs_old_rule", "old_split_test_now_train")},
        "n_old_split_trained": len(held_records["old_split_trained"]),
        "n_eligible_train_side": len(eligible - held),
        "inputs": {"semantic": args.semantic, "subjects": args.subjects, "subjects_schema":
                   "v10 (weak_reason / canon / mentions)" if labels.typed else "v9 (untyped weak labels)",
                   "text_versions": dict(text_versions), "catalogue_fields": catalogue_counts},
        "subject_status": dict(status_counts),
        "queries_by_status_source": {f"{a}/{b}": n for (a, b), n in sorted(src_counts.items())},
        "queries_by_status_rel_mode": {f"{a}/{b}": n for (a, b), n in sorted(mode_counts.items())},
        "subject_relevance": rel_counts,
        "holiday": {"n_festivals": len(holiday_counts),
                    "n_queries": sum(c["n_queries"] for c in holiday_counts.values()),
                    "festivals": holiday_counts},
        "eval_pool_records_missing_from_corpus": eval_docs_missing,
        "known_item": {"n": len(ki), "by_type": dict(Counter(q["type"] for q in ki)),
                       "shelfmark_query": sum(q["shelfmark_query"] for q in ki),
                       "targets": len({q["doc_id"] for q in ki}),
                       "targets_eligible": len({q["doc_id"] for q in ki} & eligible)},
        "memorisation_subjects": len(memo), "memorisation_by_kind": dict(Counter(s.split(":")[0] for s in memo)),
        "collection_sample": len(coll), "collection_candidates": len(coll_cands),
        "collection_long": sum(rows[d]["n_chars"] >= LONG_CHARS for d in coll),
        "pool_dup_or_short_share": round(sum(dup_n[rows[d]["text_hash"]] > 1 or rows[d]["n_chars"] < SHORT_CHARS
                                             for d in eligible) / len(eligible), 4),
        "pattern_counts": held_subjects_doc.get("pattern_counts"),
    }
    if final:
        del stats["held_out_subjects"], stats["pattern_counts"]
        stats = {"mode": "final", **stats,
                 "focus_subjects": {e["id"]: {k: e[k] for k in ("n_eligible", "n_strong_eligible", "n_weak_eligible",
                                                               "n_strong_held_out", "n_strong_train_side",
                                                               "n_unjudged_concept_mentions", "n_unjudged_held_out",
                                                               "n_canon_eligible", "n_unjudged_eligible",
                                                               "n_queries_by_source", "n_queries_by_rel_mode")}
                                    for e in focus_doc["subjects"]},
                 "n_v3_linked_subjects": len(v3_linked), "vs_v3": vs_v3}
    (out_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    if final:
        readme = README_FINAL.format(
            built=date.today().isoformat(), n_v3_linked=len(v3_linked), n_records=len(rows), n_held_records=len(held),
            n_eligible=len(eligible), n_held_eligible=len(held & eligible), n_moved=vs_v3["moved_to_train"],
            n_focus=len(focus), n_memo=len(memo), n_known=len(ki), n_coll=len(coll), v3_mode=vs_v3["mode"],
            n_v3_trainable_held=vs_v3["v3_trainable_now_held_out"], n_new_held=vs_v3["final_held_not_v3_held"])
    else:
        readme = README.format(
            built=date.today().isoformat(), n_held_subjects=len(held_subjects), n_records=len(rows),
            n_held_records=len(held), n_eligible=len(eligible), n_held_eligible=len(held & eligible),
            cap_en=NAME_CAP_EN, cap_he=NAME_CAP_HE, n_known=len(ki), n_memo=len(memo), memo_max=MEMO_MAX,
            n_coll=len(coll), long_chars=LONG_CHARS, short_chars=SHORT_CHARS)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("linked:", json.dumps({s: v["why"] for s, v in sorted(linked.items())}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
