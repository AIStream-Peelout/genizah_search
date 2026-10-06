"""Frozen evaluation set v3 for the semantic text embedder (text track step 2 of 4).

Principle: score subject similarity on records AND subjects the model never trained on, so a gain means the model
generalises (the index keeps growing; new records and new subjects arrive all the time). This is the split of the
*evaluation* run only: once the recipe is proven, the production model is retrained on everything.

Writes ``AUDIT_ROOT/v3/eval/`` (json only, no model)::

    held_out_subjects.json   6 subjects held out of training entirely (+ the "linked" subjects that go with them)
    held_out_records.json    records that must never be trained on, with the reason(s) for each
    subject_queries.json     queries per subject: probe / llm_t2 implicit queries for held-out subjects, EN+HE names
                             for every subject; each subject tagged held_out / linked / seen
    subject_relevance.json   eligible carriers of every subject (strong labels; weak = KTIV general-title only)
    known_item.jsonl         eval-pool synthetic queries {qid, doc_id, type, lang, text, round, shelfmark_query}
    memorization_pairs.json  seen subjects: size-matched train-side vs held-out carriers (memorisation gap)
    collection_sample.json   fixed 3,000 eligible unique-text records for the collection-lift metric
    eval_pool.jsonl          the frozen candidate pool: every eligible record {doc_id, text_hash, series, n_chars, dup_n, held_out}
    build_stats.json         counts printed by this script
    README.md                what every file and metric means

Run after ``semantic_text.py`` and ``build_subjects.py``::

    python3 build_eval_v3.py
"""

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from build_query_batches import is_test
from build_subjects import MERGED, PGP_DOCUMENTARY_TYPES, iter_jsonl, pgp_info
from embed_utils import AUDIT_ROOT
from eval_v3 import FILES, LONG_CHARS, SHORT_CHARS

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
LINK_EXCLUDE = {  # seen subject -> why a pattern/overlap match is not really the same concept
    "sefaria:laws-of-slaughter-on-shabbat": "Shabbat labour (slaughtering as one of the 39 melakhot), not dietary law",
    "sefaria:laws-of-the-rest-of-animals-and-slaves": "Shabbat rest of one's animals and servants, not slavery",
}
CATALOGUE_PREFIXES = ("Description:", "Alternate titles:")

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
def load_semantic(path: Path) -> Dict[str, dict]:
    """Semantic corpus rows without text (``n_chars`` added).

    :param path: ``v3/corpus_semantic.jsonl``.
    :returns: ``doc_id -> row``.
    :rtype: Dict[str, dict]
    """
    rows = {}
    for row in iter_jsonl(path):
        row["n_chars"] = len(row.pop("text"))
        rows[row["doc_id"]] = row
    return rows


def load_subjects(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Subject ids and weak subject ids per record.

    :param path: ``v3/subjects_v1.jsonl``.
    :returns: (``doc_id -> subjects``, ``doc_id -> weak subjects``).
    :rtype: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
    """
    subj, weak = {}, {}
    for row in iter_jsonl(path):
        subj[row["doc_id"]] = set(row["subjects"])
        weak[row["doc_id"]] = set(row["weak"])
    return subj, weak


def pgp_types(order: List[str], merged: Path) -> Dict[str, Set[str]]:
    """PGP document types per record (merged file is in corpus order).

    :param order: Corpus doc ids in file order.
    :param merged: ``merged_shelfmarks.jsonl``.
    :returns: ``doc_id -> PGP types``.
    :rtype: Dict[str, Set[str]]
    """
    out = {}
    for doc_id, record in zip(order, iter_jsonl(merged)):
        if record["canonical_id"] != doc_id:
            raise ValueError(f"corpus/merged order mismatch at {doc_id} vs {record['canonical_id']}")
        out[doc_id] = pgp_info(record)[1]
    return out


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

    :param text: Semantic text.
    :returns: Those lines joined by newlines (transcriptions and metadata left out).
    :rtype: str
    """
    return "\n".join(ln for ln in text.split("\n") if ln.startswith(CATALOGUE_PREFIXES))


def concept_mentions(semantic: Path, subj: Dict[str, Set[str]], held_subjects: List[str]) -> Dict[str, Set[str]]:
    """Eligible records NOT labelled with a held-out subject whose catalogue lines name its concept.

    Subject labels are incomplete (only about half of the PGP records carry any tag, and KTIV frames miss records
    such as "Nissim Gaon on the laws of lulav"), so these records are both held out of training (they would teach
    the concept) and *unjudged* for that subject in the eval (neither relevant nor irrelevant). Transcription text is
    not searched: a Talmud page quoting "סוכה" in passing is not about Sukkot.

    :param semantic: ``corpus_semantic.jsonl``.
    :param subj: ``doc_id -> subjects``.
    :param held_subjects: Held-out subject ids (keys of :data:`CONCEPT_PATTERNS`).
    :returns: ``held-out subject -> doc ids``.
    :rtype: Dict[str, Set[str]]
    """
    rx = {h: re.compile(CONCEPT_PATTERNS[h], re.I) for h in held_subjects}
    out: Dict[str, Set[str]] = {h: set() for h in held_subjects}
    for row in iter_jsonl(semantic):
        if not row["eligible"]:
            continue
        cat = catalogue_lines(row["text"])
        for h, r in rx.items():
            if h not in subj[row["doc_id"]] and r.search(cat):
                out[h].add(row["doc_id"])
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
| `subject_queries.json` | `subjects[id] = {{status, kind, queries[{{text, source, lang}}]}}`. status = `held_out` / `linked` / `seen`. source = `probe` (concept_probe_v1.json, topics), `llm_t2` (8 EN + 4 HE implicit queries written for the PGP hold-outs; they avoid the tag word and its synonyms), `name` (the subject's English and Hebrew names from subject_vocab, up to {cap_en} EN + {cap_he} HE). |
| `subject_relevance.json` | `{{id: {{strong: [...], weak: [...]}}}}` (+ `unjudged` for held-out subjects), eligible records only. `weak` = the label came only from a KTIV general title (free text, ~90 % precise); `unjudged` = unlabelled records whose catalogue lines name the held-out concept (labels are incomplete: PGP tags cover about half of PGP records). Both are left out of that subject's ranking, neither relevant nor irrelevant. |
| `known_item.jsonl` | {n_known} eval-pool synthetic queries (r1 `out` + `out_he`, pool == eval): `{{qid, doc_id, type, lang, text, round, shelfmark_query}}`. Types: implicit, topical, hebrew, specific, he_implicit, he_topical, translit. `shelfmark_query` flags queries naming a classmark (skipped by default). Not yet filtered by Isaac's review verdicts. |
| `memorization_pairs.json` | For {n_memo} seen subjects: `train` and `held_out` samples of equal size n (5 to {memo_max}) of the subject's carriers. Held-out side excludes records held out only because they carry a held-out/linked subject. |
| `collection_sample.json` | {n_coll} eligible records with unique text and a known collection series (seed 0), for the collection-lift metric. |
| `eval_pool.jsonl` | The frozen candidate pool: every eligible record `{{doc_id, text_hash, series, n_chars, dup_n, held_out}}` ({n_eligible} rows). `dup_n` = eligible records sharing the text. |
| `build_stats.json` | Counts printed by the builder. |

## Metrics (`eval_v3.evaluate(doc_ids, doc_matrix, query_vectors_fn, eval_dir, corpus_rows)`)

Document vectors must be of the v3 **semantic** text (pass `corpus_rows` to have the text hashes checked).
`query_vectors_fn(list_of_texts) -> (n, d)` must add the model's query instruction itself (production uses the
Qwen3 "Instruct: Given a search query, retrieve relevant passages" prefix). Ties are always broken pessimistically.

* **Subject retrieval** (`subjects`). Every query ranks the whole eligible pool. Relevant = strong carriers of the
  subject; weak-only carriers and (held-out subjects) unjudged concept mentions are removed from that ranking. Per query: AP, P@10, R@100; mean per subject; macro
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

## Rules for the training steps (T3/T4)

* Never use a record in `held_out_records.json["held_out"]` as an anchor or a positive (nor its text as a negative
  source that is labelled by subject).
* Never use any query in `subject_queries.json` whose status is `held_out` or `linked` as a training anchor, and drop
  non-record pairs (Sefaria topic pairs, scholarship pages) for linked Sefaria slugs; filter remaining anchors with
  `held_out_subjects.json["concept_patterns"]` (counts of what that removes are in `pattern_counts`).
* Seen-subject names may be used as training labels; the memorisation gap measures exactly that.
"""


def main() -> None:
    """Build every eval file and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semantic", default=str(V3 / "corpus_semantic.jsonl"))
    parser.add_argument("--subjects", default=str(V3 / "subjects_v1.jsonl"))
    parser.add_argument("--vocab", default=str(V3 / "subject_vocab.json"))
    parser.add_argument("--corpus-v1", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--merged", default=str(MERGED))
    parser.add_argument("--rounds", default="r1,r2")
    parser.add_argument("--out-dir", default=str(V3 / "eval"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = args.rounds.split(",")

    rows = load_semantic(Path(args.semantic))
    subj, weak = load_subjects(Path(args.subjects))
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
    held_subjects = list(HELD_OUT_TOPICS) + list(HELD_OUT_PGP)
    linked = find_linked(held_subjects, by, vocab)

    # ---- held-out records ----
    pools = synthetic_pools(rounds)
    side = split_side(rows)
    mentions = concept_mentions(Path(args.semantic), subj, held_subjects)
    mention_any = set().union(*mentions.values())
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
    by_reason = {k: sorted(d for d in held if k in reasons[d])
                 for k in ("split", "held_out_subject", "linked_subject", "concept_mention", "sukkot_gold", "eval_pool",
                           "dup_closure")}
    eligible = {d for d, r in rows.items() if r["eligible"]}
    flips = sum(side[d] != is_test(d) for d in rows)
    held_records = {
        "built": date.today().isoformat(),
        "rule": "held out = 20% split (md5(doc_id) % 5 == 0; an eligible identical-text group follows its smallest "
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
    for sid in held_subjects + sorted(linked) + sorted(s for s in vocab if s not in held_subjects and s not in linked):
        meta = vocab[sid]
        status = "held_out" if sid in held_subjects else "linked" if sid in linked else "seen"
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
    held_subjects_doc = {
        "built": date.today().isoformat(), "subjects": held_entries, "linked": linked,
        "concept_patterns": CONCEPT_PATTERNS,
        "selection": {"probe_topics": {"criteria": ">= 100 strong eligible records and >= 8 probe queries; one festival "
                                                   "with low Sukkot overlap + one non-calendar halakhic topic; topics "
                                                   "whose PGP twin stays seen are rejected",
                                       "candidates": topic_rows},
                      "pgp": {"criteria": {**PGP_CRITERIA, "excluded_categories": sorted(PGP_EXCLUDED_CATEGORIES),
                                           "rule": "one subject per category among those that pass; prefer clear, "
                                                   "non-overlapping concepts"},
                              "candidates": pgp_rows, "rejected_notes": PGP_REJECTED}},
        "pattern_counts": pattern_counts(held_ids, rows, rounds, pools, Path(args.semantic)),
    }

    # ---- known items, memorisation, collection sample, pool ----
    eval_docs = {d for d, p in pools.items() if p == "eval"}
    ki = known_items(rounds, pools, shelfmarks(Path(args.corpus_v1), eval_docs))
    seen = [s for s, v in subjects_q.items() if v["status"] == "seen"]
    memo = memorization_pairs(seen, strong, held, held_subject_recs, rows)
    dup_n = Counter(rows[d]["text_hash"] for d in eligible)
    coll_cands = sorted(d for d in eligible if dup_n[rows[d]["text_hash"]] == 1 and rows[d]["series"])
    coll = sorted(random.Random(0).sample(coll_cands, COLLECTION_N))
    relevance = {s: {"strong": sorted(strong.get(s, ())), "weak": sorted(by.get(s, set()) - strong.get(s, set()))}
                 for s in vocab}
    for h in held_subjects:
        relevance[h]["unjudged"] = sorted(mentions[h])

    # ---- integrity checks ----
    for members in groups.values():
        assert len({d in held for d in members}) == 1, f"text group split across sides: {members[:3]}"
    assert eval_docs <= held and {d for d in rows if rows[d]["sukkot_gold"]} <= held
    assert all(d in held for s in held_subjects + list(linked) for d in by_all.get(s, ()))
    assert all(d not in held for m in memo.values() for d in m["train"])
    assert all(d in held for m in memo.values() for d in m["held_out"])
    assert all(q["doc_id"] in held for q in ki)
    assert mention_any <= held

    # ---- write ----
    def dump(key: str, obj) -> None:
        (out_dir / FILES[key]).write_text(json.dumps(obj, ensure_ascii=False, indent=1 if key == "held_out_subjects"
                                                     else None), encoding="utf-8")

    dump("held_out_subjects", held_subjects_doc)
    dump("held_out_records", held_records)
    dump("subject_queries", {"built": date.today().isoformat(), "subjects": subjects_q})
    dump("subject_relevance", relevance)
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
                fh.write(json.dumps({"doc_id": d, "text_hash": r["text_hash"], "series": r["series"],
                                     "n_chars": r["n_chars"], "dup_n": dup_n[r["text_hash"]], "held_out": d in held},
                                    ensure_ascii=False) + "\n")
    status_counts = Counter(v["status"] for v in subjects_q.values())
    src_counts = Counter((v["status"], q["source"]) for v in subjects_q.values() for q in v["queries"])
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
        "subject_status": dict(status_counts),
        "queries_by_status_source": {f"{a}/{b}": n for (a, b), n in sorted(src_counts.items())},
        "known_item": {"n": len(ki), "by_type": dict(Counter(q["type"] for q in ki)),
                       "shelfmark_query": sum(q["shelfmark_query"] for q in ki),
                       "targets": len({q["doc_id"] for q in ki}),
                       "targets_eligible": len({q["doc_id"] for q in ki} & eligible)},
        "memorisation_subjects": len(memo), "memorisation_by_kind": dict(Counter(s.split(":")[0] for s in memo)),
        "collection_sample": len(coll), "collection_candidates": len(coll_cands),
        "collection_long": sum(rows[d]["n_chars"] >= LONG_CHARS for d in coll),
        "pool_dup_or_short_share": round(sum(dup_n[rows[d]["text_hash"]] > 1 or rows[d]["n_chars"] < SHORT_CHARS
                                             for d in eligible) / len(eligible), 4),
        "pattern_counts": held_subjects_doc["pattern_counts"],
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "README.md").write_text(README.format(
        built=date.today().isoformat(), n_held_subjects=len(held_subjects), n_records=len(rows),
        n_held_records=len(held), n_eligible=len(eligible), n_held_eligible=len(held & eligible),
        cap_en=NAME_CAP_EN, cap_he=NAME_CAP_HE, n_known=len(ki), n_memo=len(memo), memo_max=MEMO_MAX,
        n_coll=len(coll), long_chars=LONG_CHARS, short_chars=SHORT_CHARS), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("linked:", json.dumps({s: v["why"] for s, v in sorted(linked.items())}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
