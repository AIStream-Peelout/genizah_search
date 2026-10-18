"""Subject table for the semantic text track: which corpus records are about what (text track step 1 of 4).

Every record gets a list of subject ids from five catalogue sources, none of which is the record's own text or
shelf mark, so training on them pulls records together by *subject* rather than by literal string or collection:

* ``topic:<t>``   — festival / halakhic topics from KTIV scholarly frames (``topic_v2``: build_corpus.py's
  ``TOPIC_FRAME_RULES`` re-implemented, matched per frame, extended with Hebrew-script frames and liturgical frames
  such as "Common Prayers"/"תפילות קבע" "Musaf Sukkot Amidah", "מוסף סוכות", Piyyut "(Yotzer for Shabbat)",
  "יוצר לשבת"), plus the human-verified Sukkot seed.
* ``work:<w>``    — the identified work and section, normalised across English and Hebrew catalogue entries
  ("[Talmud Bavli]: Sukkah 29 a" and "[חננאל ...]: סוכה" -> ``work:talmud-bavli-sukkah`` / ``work:rabbenu-hananel-sukkah``;
  "[תפילות קבע]: מוסף סוכות עמידה" and "[Common Prayers]: Musaf Sukkot Amidah" -> one id). Incipits are dropped:
  a single poem's first line is record identity, not a subject.
* ``domain:<d>``  — KTIV domain terms (each " # " level), synonyms merged, "Unidentified"-style values dropped.
* ``pgp:<s>``     — curated PGP tags: workflow / period / language-script / physical-form / person tags dropped,
  synonyms merged (get -> divorce, capitation tax -> poll-tax + taxes, illness:* -> medicine ...), subjects with
  fewer than ``--min-pgp`` records dropped. Every raw tag and its decision goes to ``results/pgp_tag_curation.csv``.
* ``sefaria:<slug>`` — Sefaria topics whose curated source refs overlap the record's frame refs
  (``frame_refs.jsonl`` x ``sefaria/derived/curated_refs.jsonl``), restricted to halakhic / holiday / ritual topics.

Topic rules also run over KTIV general titles (the cataloguer's free-text identification, e.g. "הלכות שחיטה");
topics found only there are listed again under ``weak`` so strict evaluations can drop them
(``--no-general-titles`` turns this off).

Festival topics (:data:`FESTIVAL_TOPICS`) also come from every record's description (FJP, PGP, TEI, Bodleian),
case-insensitively (:data:`FESTIVAL_TERMS`). Description, Alternate titles and Document Type are parsed from the
corpus ``text_v3`` column (the semantic_v3 text, falling back to ``text``) after folding continuation lines
(:func:`fold_field_lines`), never from the embedded semantic_v4 text, which has no such labels.

Weak labels are typed (``weak_reason``), because training masks and eval relevance need different things:

* ``canon``: a festival found only on a scripture / tractate / law-code frame ("[Bible]: Esther" is not Purim), in a
  canon description ("Sukkah 29a", "תלמוד ... סוכה לד ע"א") or a canon general title. KTIV domain labels appended to
  descriptions ("Bible: Texts", "Aramaic Targumim") are stripped first, and a liturgy / homily cue (piyyut, prayer,
  סידור, sermon ...) cancels the demotion (:func:`is_canon_context`).
  An FJP Document Type "bible" / "talmud" (a keyword guess) makes the mention canon too, unless a cue says otherwise.
* ``incidental``: named by a letter / legal / documentary record (FJP Document Type, PGP type; a PGP "Paraliterary
  text" only without a liturgy / homily cue), or only inside a date clause ("Dated ... Purim 1419", "Written on the eve
  of Sukkot").
* ``title``: found only in a KTIV general title (free text); ``--no-general-titles`` turns this off.

Every topic matched only on a canon frame is also listed in the record's ``canon``: festival ones are weak, the
others (``topic:marriage`` on "[Talmud Bavli]: Ketubbot") stay strong but are tagged, and every ``sefaria:*`` id is
listed too (Sefaria labels come only from canon frame refs; ``--no-sefaria-canon`` turns that off). Sukkot gold
records stay strong; ``gold_override`` says the gold flag overrode a weak label. ``mentions`` lists subjects whose kept
vocabulary names (or :data:`FESTIVAL_RE`) occur in the record's Description / Alternate titles but which the record
does not carry strongly (masked per anchor in training, unjudged in eval). ``--legacy-festivals`` reproduces v9's
``subjects`` / ``weak`` byte-for-byte (the new keys are still written).

Outputs (``AUDIT_ROOT/v3``): ``subjects_v1.jsonl`` ({doc_id, subjects, weak, weak_reason, canon, mentions,
gold_override}; ``weak == sorted(weak_reason)``), ``subject_vocab.json`` ({id: {kind, names_en, names_he, n_records,
n_eligible, source, ...}}), ``subjects_v1.stats.json``; and ``results/pgp_tag_curation.csv`` in the audit code dir.

Run after ``semantic_text.py`` (json only, no model)::

    python3 build_subjects.py --self-test
    python3 build_subjects.py
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Pattern, Set, Tuple

from build_train_pairs import TOPIC_PHRASES
from embed_utils import AUDIT_ROOT
# re-exported: one folding rule for the v4 text producer and every reader of the v3 fields (multi-line descriptions)
from semantic_text import FIELD_PREFIXES, fold_field_lines  # noqa: F401

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
V3 = AUDIT_ROOT / "v3"
RESULTS = Path(__file__).resolve().parent / "results"
HEB = re.compile(r"[֐-׿]")

# ---------------------------------------------------------------------------------------------------------------
# topic_v2
# ---------------------------------------------------------------------------------------------------------------
# Matched against each frame as "[<work>]: <body>" (continuation frames inherit the previous frame's work).
# The first alternatives of each rule are build_corpus.TOPIC_FRAME_RULES verbatim, so topic_v2 is a superset of v1.
HB = r"(?<![\u05D0-\u05EA])[בהוכלמשד]{0,2}"  # Hebrew word start, allowing up to two prefix letters (ב, ה, ו, ל ...)
HE = r"(?![\u05D0-\u05EA])"  # Hebrew word end
AY = r"[ʿ'’`]"  # ayin / apostrophe in transliterations ("Shavuʿot", "Neʿilah", "Tish'a be-Av")
# v9's rules, verbatim (``--legacy-festivals``); TOPIC_RULES_V2 below fixes two Hebrew bugs in them.
TOPIC_RULES_V2_LEGACY: Dict[str, str] = {
    "sukkot": r"\]:\s*Sukkah\b|Hoshanot|Shofar ve-Sukkah|\(Sukkot\)|Lulav"
              rf"|\bSukk(?:ah|ot)\b|Hoshana Rabbah|\bEtrog\b|{HB}(?:סוכה|סוכות|סכות|הושענ|לולב|אתרוג)",
    "pesach": r"\]:\s*Pesahim\b|Hametz u-Mat|Haggadah|\(Pesach\)|\(Passover\)"
              rf"|\bPesa[hc]h?\b|Passover|Qorban Pesah|{HB}(?:פסח(?:ים)?{HE}|חמץ{HE})",
    "yom_kippur": r"\]:\s*Yoma\b|Shevitat Asor|Avodat Yom|Kippur"
                  rf"|Ne'?ilah|High Holy Days|{HB}(?:יום ה?כי?פור|יוה['\"]+כ{HE}|יומא{HE}|נעילה{HE}|ימים נוראים)",
    "rosh_hashanah": r"\]:\s*Rosh ha-Shanah|Shofar ve-Sukkah"
                     rf"|Rosh ha-?Shanah|Rosh Hashanah|\bShofar\b|High Holy Days"
                     rf"|{HB}(?:ראש השנה|ר['\"]+ה{HE}|שופר{HE}|ימים נוראים)",
    "shabbat": r"\]:\s*Shabbat\b|\]:\s*Eruvin\b"
               rf"|\bShabbat\b|\bSabbath\b|\bEruvin\b|{HB}(?:שבת(?:ות)?{HE}|עי?רובין|עירוב{HE}|במה מדליקין)",
    "purim": r"\]:\s*Megillah\b|Megillah va-Hanukkah|\[Bible\]:\s*Esther|\(Purim\)"
             rf"|\bPurim\b|{HB}פורים{HE}|\]:\s*מגילה|הלכות מגילה|\[מקרא\]:\s*אסתר|מגילת אסתר",
    "hanukkah": rf"Megillah va-Hanukkah|\(Hanukkah\)|Hanuk+ah|Chanuk+ah|{HB}חנוכה",
    "marriage": r"\]:\s*Ketubbot\b|\]:\s*Qiddushin\b|\]:\s*Ishut\b"
                rf"|Shidduchin|engagement and marriage|{HB}(?:כתובות|קי?דושין|אישות|שידוכין|אירוסין|נישואין){HE}",
    "divorce": rf"\]:\s*Gittin\b|\]:\s*Gerushin\b|Get Halitzah|{HB}(?:גיטין|גטין|גירושין|גרושין){HE}",
    "kashrut": r"\]:\s*Hullin\b|\]:\s*Shehitah\b|Ma'akhalot Asurot"
               rf"|{HB}(?:חולין{HE}|שחיטה{HE}|מאכלות אסורות|טרפות{HE}|טריפות{HE})",
    "niddah": rf"\]:\s*Niddah\b|Mikva'ot|Issurei Bi'ah|{HB}(?:ני?דה{HE}|מקו?ואות{HE}|איסורי ביאה)",
    "tefillin_mezuzah": rf"Tefillin Mezuzah|\]:\s*Tefillin\b|\bTefillin\b|\bMezuz|{HB}(?:תפילין|מזוז)",
    "mourning": r"\]:\s*Mo'ed Qatan\b|\]:\s*Evel\b|Semahot"
                rf"|מועד קטן|{HB}(?:אבלות|אבלים){HE}|ברכת אבלים|הלכות אבל|מסכת שמחות|\]:\s*שמחות",
    # new in v2: festivals and liturgical occasions that only the liturgy frames carry
    "simhat_torah": rf"Simh?c?hat Torah|Shemini Atzeret|{HB}(?:שמחת תורה|שמיני עצרת)",
    "shavuot": rf"Shavu'?ot\b|Pentecost|{HB}(?:מוסף שבועות|יוצר לשבועות|תפילת שבועות|חג השבועות)",
    "selihot": rf"Selih?c?hot|{HB}סליחות{HE}",
    "grace_after_meals": rf"Grace after [Mm]eals|Birkat ha-?Mazon|{HB}ברכת המזון",
    "rosh_hodesh": rf"Rosh Hodesh|Qiddush ha-Hodesh|{HB}ראש חודש",
    "fast_days": rf"Ta'aniyyot|Tish'?a be-?Av|\]:\s*Ta'anit\b|{HB}(?:תשעה באב|הלכות תעני)",
}
TOPIC_RULES_V2: Dict[str, str] = {
    **TOPIC_RULES_V2_LEGACY,
    # HB allows a כ prefix, so v9 read "(ה)כפורים" (Yom Kippur) as Purim: no כ directly before פורים
    "purim": r"\]:\s*Megillah\b|Megillah va-Hanukkah|\[Bible\]:\s*Esther|\(Purim\)"
             rf"|\bPurim\b|{HB}(?<!כ)פורים{HE}|\]:\s*מגילה|הלכות מגילה|\[מקרא\]:\s*אסתר|מגילת אסתר",
    # hyphenated "יום-הכפורים", gershayim "יוה״כ" / "יו״כ"; "Yom Kippur Katan" (eve of Rosh Hodesh) is not Yom Kippur
    "yom_kippur": r"\]:\s*Yoma\b|Shevitat Asor|Avodat Yom|Kippur(?! Katan)"
                  rf"|Ne'?ilah|High Holy Days|{HB}(?:יום[- ]ה?כי?פור(?!\S* קטן)|יוה?['\"״]+כ{HE}|יומא{HE}|נעילה{HE}"
                  r"|ימים ה?נוראים)",
}
# Ambiguous outside liturgy ("Shevu'ot" / "שבועות" is also the tractate on oaths): applied to liturgical frames only,
# never to general titles.
TOPIC_RULES_LITURGY: Dict[str, str] = {
    "shavuot": rf"Shevu[`']ot|{HB}שבועות{HE}",
}
TOPIC_RE = {t: re.compile(p) for t, p in TOPIC_RULES_V2.items()}
TOPIC_RE_LEGACY = {t: re.compile(p) for t, p in TOPIC_RULES_V2_LEGACY.items()}
TOPIC_LIT_RE = {t: re.compile(p) for t, p in TOPIC_RULES_LITURGY.items()}
TOPIC_SEFARIA = {  # topic -> Sefaria slugs whose names we borrow
    "sukkot": ["sukkot", "sukkah", "lulav"], "pesach": ["passover", "haggadah"], "yom_kippur": ["yom-kippur"],
    "rosh_hashanah": ["rosh-hashanah", "shofar"], "shabbat": ["shabbat"], "purim": ["purim"],
    "hanukkah": ["hanukkah"], "marriage": ["marriage", "ketubah"], "divorce": ["divorce", "get"],
    "kashrut": ["kashrut"], "niddah": ["niddah"], "tefillin_mezuzah": ["tefillin", "mezuzah"],
    "mourning": ["mourning"], "simhat_torah": ["simchat-torah", "shemini-atzeret"], "shavuot": ["shavuot"],
    "selihot": ["selichot"], "grace_after_meals": ["birkat-hamazon"], "rosh_hodesh": ["rosh-chodesh"],
    "fast_days": ["fasting"],
}
TOPIC_EXTRA_PHRASES = {
    "hanukkah": ["Hanukkah", "חנוכה"], "simhat_torah": ["Simhat Torah", "Shemini Atzeret", "שמחת תורה"],
    "shavuot": ["Shavuot", "Feast of Weeks", "שבועות"], "selihot": ["Selihot", "penitential prayers", "סליחות"],
    "grace_after_meals": ["Grace after Meals", "ברכת המזון"], "rosh_hodesh": ["New Moon", "ראש חודש"],
    "fast_days": ["fast days", "Tisha be-Av", "תעניות"],
}

# ---------------------------------------------------------------------------------------------------------------
# festivals: what a bare festival-name query should find
# ---------------------------------------------------------------------------------------------------------------
# v9 learned "Purim" -> Esther / Megillah and "פסח" -> Pesahim because topic_v2 gives festival topics to scroll and
# tractate frames and those were the only carriers whose text lacks the name (see the 2026-10-08 holiday handoff).
# Festival topics found on a canon frame (scripture, tractate, law code) are therefore WEAK: still masked (never a
# negative), never a subject-pair carrier, left out of eval rankings. Descriptions of every record (FJP, PGP, TEI,
# Bodleian -- not only KTIV) are scanned for the festival too, case-insensitively.
FESTIVAL_TOPICS = ("pesach", "sukkot", "yom_kippur", "rosh_hashanah", "purim", "hanukkah", "shavuot", "simhat_torah",
                   "fast_days", "rosh_hodesh")
# Surface forms of the festival itself, never of its tractate or scroll ("Pesahim", "Megillah", "Esther" are absent).
# Left out on purpose: "new moon" (calendar texts and letters about sighting it, not Rosh Hodesh), "ר"ח" (also
# Rabbenu Hananel), bare "שבועות" (also oaths / tractate Shevu'ot), bare "matsa" (Hebrew "found").
FESTIVAL_TERMS: Dict[str, str] = {
    "pesach": rf"\bpesa[hḥc]h?\b(?! (?:al-?)?mas[iī][hḥ])|\bpessach\b|\bpeysa[kc]h\b|(?<!christian )passover"
              r"|\bhagg?add?ah?\b|\bmatz(?:ah|ot|os)\b|\bmatsot\b"
              r"|seder[- ]night|night of the seder|\b(?:[hḥ]|ch)a?metz\b|afikoman|unleavened bread"
              rf"|{HB}(?:פסח(?! ה?(?:אל)?מ[סש]יח)|חמץ|אפיקומן){HE}|הגדה (?:של|ל)פסח|חג המצות|ליל הסדר",
    "sukkot": r"\bsukk(?:ot|os|ah?)\b|\bsukot\b|\bsucc(?:ot|os|ah?)\b|tabernacles|feast of booths"
              rf"|\bhosha{AY}?n(?:ot|os)\b|\bhosha{AY}?nah? rabb|\blulav|\be?th?rog\b|four species"
              rf"|{HB}(?:סוכות|סכות|סוכה|לולב|אתרוג){HE}|(?<!בן )(?<!בר )(?<!ב\"ר )(?<!ב״ר ){HB}הושענ(?:ות|א|ה){HE}",
    "yom_kippur": rf"kipp?ur(?!\w* katan)|day of atonement|\bne{AY}?ilah?\b|high holy ?days|days of awe"
                  rf"|{HB}(?:יום[- ]ה?כי?פורי?ם?|כי?פור|נעילה|יוה?[\"״]כ|ימים ה?נוראים){HE}(?! קטן)",
    "rosh_hashanah": r"rosh ha-?sh?anah?|ro(?:sh|s) ha-?s(?:h)?ana\b|jewish new year"
                     r"|(?<!muslim )(?<!islamic )(?<!coptic )(?<!hijri )\bnew year\b|\bs(?:h)?ofar\b"
                     r"|high holy ?days|days of awe"
                     rf"|{HB}(?:ראש השנה|שופר|ר[\"״']ה|ימים ה?נוראים){HE}",
    "purim": rf"\bpurim\b|{HB}(?<!כ)פורים{HE}",
    "hanukkah": rf"\b[hḥc]h?anuk+ah?\b|{HB}חנוכה{HE}",
    "shavuot": rf"\bshavu{AY}?(?:ot|os)\b|\bshavous\b|pentecost|feast of weeks"
               rf"|{HB}חג ה?שבועות{HE}|{HB}ל(?:חג )?שבועות{HE}|{HB}(?:מוסף|יוצר|תפילת) ?ל?שבועות{HE}",
    "simhat_torah": rf"\bsim(?:[hḥ]|ch)at torah|simchas torah|shemini {AY}?a(?:tz|ts|ṣ|z|c)eret"
                    rf"|{HB}(?:שמחת תורה|שמיני עצרת){HE}",
    "fast_days": rf"\btish{AY}?a?h? b[e'’]?[-' ]?{AY}?av\b|\b(?:ninth|9th) of av\b|{HB}תשעה באב{HE}",
    "rosh_hodesh": rf"\brosh (?:[hḥ]|ch)odesh\b|{HB}ראש חו?דש{HE}",
}
FESTIVAL_RE = {t: re.compile(p, re.I) for t, p in FESTIVAL_TERMS.items()}
# Vocabulary names that are the festival's scroll or tractate, not the festival (see also CANON_NAME_RE)
FESTIVAL_NAME_DROP = {"purim": {"Scroll of Esther", "מגילת אסתר", "Megillah", "Megillat Esther", "מגילה"}}


def festival_mentions(text: str) -> Set[str]:
    """Festival topics whose surface forms occur in ``text`` (case-insensitive, Hebrew word boundaries).

    :param text: Any text (a query, a description line).
    :returns: Festival topic names, e.g. ``{"pesach"}``.
    :rtype: Set[str]
    """
    return {t for t, rx in FESTIVAL_RE.items() if rx.search(text)}


def text_field(text: str, label: str) -> str:
    """Value of one ``Label: value`` line of a semantic text (first occurrence), or "".

    Pass the semantic_v3 text through :func:`fold_field_lines` first to get multi-line values whole.

    :param text: Semantic text (``corpus_semantic.jsonl`` ``text_v3``, else ``text``).
    :param label: Field label, e.g. ``"Description"``.
    :returns: The value, stripped.
    :rtype: str
    """
    prefix = label + ":"
    return next((line[len(prefix):].strip() for line in text.split("\n") if line.startswith(prefix)), "")


def source_fields(sem: dict) -> Dict[str, str]:
    """Description, Alternate titles and Document Type of a corpus row, from its semantic_v3 text (folded).

    The merged record has no Document Type (HDA infers it from the description when rendering the v8 text), so it is
    read from the text as well.

    :param sem: ``corpus_semantic.jsonl`` row (``text_v3`` when the corpus was built with ``--text-version
        semantic_v4``, else ``text``).
    :returns: {"description", "alt_titles", "doc_type"}, "" when absent.
    :rtype: Dict[str, str]
    """
    folded = fold_field_lines(sem.get("text_v3") or sem["text"])
    return {"description": text_field(folded, "Description"), "alt_titles": text_field(folded, "Alternate titles"),
            "doc_type": text_field(folded, "Document Type")}


# ---------------------------------------------------------------------------------------------------------------
# work normalisation
# ---------------------------------------------------------------------------------------------------------------
LITURGY_WORKS = {"Piyyut", "Common Prayers", "Liturgical additions", "Brakhot", "Occasional prayer"}
TRACTATE_WORKS = {"Talmud Bavli", "Talmud Yerushalmi", "Mishnah", "Tosefta", "Hilkhot ha-Rif",
                  "Talmud Bavli Commentaries", "Rabbenu Hananel", "Mishnah Commentaries"}
BOOK_WORKS = {"Bible", "Targum Onqelos", "Aramaic Targumim", "Arabic Tafsir", "Tafsir Saadya Gaon", "Haftarot",
              "Biblical Exegesis", "Biblical Exegesis - Rabbanite", "Biblical Exegesis - Karaite", "Rashi on the Bible",
              "Derashot", "Massorah", "Yefet b. Eli, Bible Commentary"}
LAW_CODE_WORKS = {"Mishneh Torah", "Halakhot Gedolot", "Sefer ha-Mitzvot (Maimonides)"}
# frames whose festival topic names the scroll / tractate / law-code section (homilies are festival content)
CANON_WORKS = (TRACTATE_WORKS | BOOK_WORKS | LAW_CODE_WORKS) - {"Derashot"}
# bracket labels that are KTIV domains rather than works (the domain subject already covers them): no work id
GENERIC_WORKS = {"Halakhic", "Liturgy and Brakhot", "Court Records", "Contracts", "Secular Poetry", "Biblical Glossary",
                 "Halakhot ha-Rif and its Commentaries", "מעשה בית דין", "Documentary", "Rabbinic Literature",
                 "Halakhic Literature and Talmudic Commentaries"}
WORK_ALIASES = {
    "Moses b. Maimon, Rambam, Mishneh Torah": "Mishneh Torah", "משנה תורה": "Mishneh Torah",
    "פיוט": "Piyyut", "תפילות קבע": "Common Prayers", "תוספות של סידור": "Liturgical additions",
    "סדרי ברכות": "Brakhot", "תפילה מזדמנת": "Occasional prayer", "מקרא": "Bible", "תרגום אונקלוס": "Targum Onqelos",
    "תעודות אישיות ושטרות": "Documents", "מכתבים": "Letters", "לקח טוב": "Leqah Tov", "הפטרות": "Haftarot",
    "תפסיר רס''ג": "Tafsir Saadya Gaon", "פרשנות מקרא": "Biblical Exegesis",
    "פרשנות מקרא רבנית": "Biblical Exegesis - Rabbanite", "לא מזוהה, פרשנות מקרא רבנית": "Biblical Exegesis - Rabbanite",
    "Biblical Exegesis- Rabbanite": "Biblical Exegesis - Rabbanite", "Biblical Exegesis- Karaite": "Biblical Exegesis - Karaite",
    "פירוש רש''י למקרא": "Rashi on the Bible", "תלמוד בבלי": "Talmud Bavli", "תלמוד ירושלמי": "Talmud Yerushalmi",
    "משנה": "Mishnah", "תוספתא": "Tosefta", "הלכות הרי''ף": "Hilkhot ha-Rif",
    "Commentary on Talmud Bavli": "Talmud Bavli Commentaries", "Talmudic Commentaries": "Talmud Bavli Commentaries",
    "פירוש רבנו חננאל לתלמוד": "Rabbenu Hananel", "R. Hananel's Talmud Commentary": "Rabbenu Hananel",
    "הלכות גדולות": "Halakhot Gedolot", "ספר המצוות": "Sefer ha-Mitzvot (Maimonides)",
    "כתאב אלשראיע": "Sefer ha-Mitzvot (Maimonides)", "שירת חול": "Secular Poetry", "בראשית רבה": "Bereshit Rabbah",
    "זוהר": "Zohar", "פסיקתא דרב כהנא": "Pesiqta de-Rav Kahana",
    "Extracts/Abridgment Talmud Bavli": "Talmud Bavli", "Translation of Mishneh Torah": "Mishneh Torah",
    "Commentary on Mishnah": "Mishnah Commentaries", "Mishnaic Commentaries": "Mishnah Commentaries",
    "פירוש לתלמוד": "Talmud Bavli Commentaries", "Yefet's Commentary on the Bible": "Yefet b. Eli, Bible Commentary",
    "יפת בן עלי הלוי, פירוש על התורה": "Yefet b. Eli, Bible Commentary",
}
# Hebrew works whose generic last comma part needs the author kept ("יפת בן עלי הלוי, פירוש על התורה")
WORK_CONTAINS = [("תעודות אישיות ושטרות", "Documents")]
WORK_HE = {
    "Talmud Bavli": "תלמוד בבלי", "Talmud Yerushalmi": "תלמוד ירושלמי", "Mishnah": "משנה", "Tosefta": "תוספתא",
    "Hilkhot ha-Rif": "הלכות הרי\"ף", "Talmud Bavli Commentaries": "פירושים לתלמוד הבבלי",
    "Rabbenu Hananel": "פירוש רבנו חננאל לתלמוד", "Mishneh Torah": "משנה תורה", "Bible": "מקרא",
    "Targum Onqelos": "תרגום אונקלוס", "Tafsir Saadya Gaon": "תפסיר רס\"ג", "Haftarot": "הפטרות",
    "Biblical Exegesis": "פרשנות מקרא", "Rashi on the Bible": "פירוש רש\"י למקרא", "Piyyut": "פיוט",
    "Common Prayers": "תפילות קבע", "Liturgical additions": "תוספות של סידור", "Brakhot": "סדרי ברכות",
    "Documents": "תעודות אישיות ושטרות", "Letters": "מכתבים", "Leqah Tov": "לקח טוב",
    "Halakhot Gedolot": "הלכות גדולות", "Sefer ha-Mitzvot (Maimonides)": "ספר המצוות לרמב\"ם",
}
HE_TRACTATES = {
    "ברכות": "Berakhot", "פאה": "Pe'ah", "דמאי": "Demai", "כלאים": "Kilayim", "שביעית": "Shevi'it",
    "תרומות": "Terumot", "מעשרות": "Ma'aserot", "מעשר שני": "Ma'aser Sheni", "חלה": "Hallah", "ערלה": "Orlah",
    "ביכורים": "Bikkurim", "בכורים": "Bikkurim", "שבת": "Shabbat", "עירובין": "Eruvin", "ערובין": "Eruvin",
    "פסחים": "Pesahim", "שקלים": "Sheqalim", "יומא": "Yoma", "סוכה": "Sukkah", "ביצה": "Betzah",
    "ראש השנה": "Rosh ha-Shanah", "תענית": "Ta'anit", "מגילה": "Megillah", "מועד קטן": "Mo'ed Qatan",
    "חגיגה": "Hagigah", "יבמות": "Yevamot", "כתובות": "Ketubbot", "נדרים": "Nedarim", "נזיר": "Nazir",
    "סוטה": "Sotah", "גיטין": "Gittin", "קידושין": "Qiddushin", "קדושין": "Qiddushin", "בבא קמא": "Bava Qamma",
    "בבא מציעא": "Bava Metzi'a", "בבא בתרא": "Bava Batra", "סנהדרין": "Sanhedrin", "מכות": "Makkot",
    "שבועות": "Shevu'ot", "עדיות": "Eduyyot", "עבודה זרה": "Avodah Zarah", "אבות": "Avot", "הוריות": "Horayot",
    "זבחים": "Zevahim", "מנחות": "Menahot", "חולין": "Hullin", "בכורות": "Bekhorot", "ערכין": "Arakhin",
    "תמורה": "Temurah", "כריתות": "Keritot", "מעילה": "Me'ilah", "תמיד": "Tamid", "מדות": "Middot",
    "מידות": "Middot", "קנים": "Qinnim", "כלים": "Kelim", "אהלות": "Ohalot", "נגעים": "Nega'im", "פרה": "Parah",
    "טהרות": "Taharot", "מקואות": "Mikva'ot", "נדה": "Niddah", "נידה": "Niddah", "מכשירין": "Makhshirin",
    "זבים": "Zavim", "טבול יום": "Tevul Yom", "ידים": "Yadayim", "עוקצין": "Uqtzin",
}
HE_BOOKS = {
    "בראשית": "Genesis", "שמות": "Exodus", "ויקרא": "Leviticus", "במדבר": "Numbers", "דברים": "Deuteronomy",
    "יהושע": "Joshua", "שופטים": "Judges", "שמואל א": "I Samuel", "שמואל ב": "II Samuel", "מלכים א": "I Kings",
    "מלכים ב": "II Kings", "ישעיהו": "Isaiah", "ישעיה": "Isaiah", "ירמיהו": "Jeremiah", "ירמיה": "Jeremiah",
    "יחזקאל": "Ezekiel", "הושע": "Hosea", "יואל": "Joel", "עמוס": "Amos", "עובדיה": "Obadiah", "יונה": "Jonah",
    "מיכה": "Micah", "נחום": "Nahum", "חבקוק": "Habakkuk", "צפניה": "Zephaniah", "חגי": "Haggai",
    "זכריה": "Zechariah", "מלאכי": "Malachi", "תהלים": "Psalms", "תהילים": "Psalms", "משלי": "Proverbs",
    "איוב": "Job", "שיר השירים": "Song of Songs", "רות": "Ruth", "איכה": "Lamentations", "קהלת": "Ecclesiastes",
    "אסתר": "Esther", "דניאל": "Daniel", "עזרא": "Ezra", "נחמיה": "Nehemiah", "דברי הימים א": "I Chronicles",
    "דברי הימים ב": "II Chronicles",
}
KNOWN_TRACTATES = set(HE_TRACTATES.values()) | {"Semahot", "Soferim", "Kallah", "Derekh Eretz", "Eduyot"}
EN_BOOKS = sorted(set(HE_BOOKS.values()) | {"Chronicles", "Samuel", "Kings"}, key=len, reverse=True)
EN_BOOK_FIX = {"Ezechiel": "Ezekiel", "Michah": "Micah"}
# Hebrew liturgy phrases -> the English KTIV phrasing, so both catalogue languages land on one work id
LITURGY_HE = {
    "הגדה של פסח": "Haggadah of Pesach", "קריאת שמע וברכותיה": "Qeriat Shema and its benedictions",
    "וידוי ונפילת אפיים": "Vidduy and Nefilat Appayim", "ברכות התורה וההפטרות":
    "Benedictions for the reading of the Torah and the Haftarah", "פסוקי דזמרה": "Pesuqe d'zimrah",
    "ברכות השחר": "Early morning benedictions", "ברכת המזון": "Grace after meals",
    "ברכת המזון לחול ולשבת": "Grace after meals", "קידוש לסוכות": "Qiddush for Sukkot", "ימים נוראים": "High Holy Days",
    "שמיני עצרת": "Shemini Atzeret", "שמחת תורה": "Simhat Torah", "ראש השנה": "Rosh ha-Shanah",
    "יום הכיפורים": "Yom ha-Kippurim", "חול המועד": "Hol ha-Mo'ed", "ראש חודש": "Rosh Hodesh",
    "הושענא רבה": "Hoshana Rabbah", "סדר העבודה": "Avodah", "קודם התפילה": "Before the prayer",
    "שיר של יום": "Song of the day", "במה מדליקין": "Bameh madliqin", "פרקי אבות": "Pirqe Avot",
    "מזמורי תהלים": "Psalms", "אוסף פסוקים": "Collection of verses", "ברכת אבלים": "Mourners' benediction",
    "ברכות חנוכה": "Hanukkah benedictions", "קריאת התורה": "Torah Reading", "מעין שבע": "Me'ein Sheva",
    "פיוטים שאינן מהתפילה": "Non-liturgical piyyutim", "זמירות/בקשות לשבת ולמועדים":
    "Zemirot and baqqashot for Shabbat and festivals", "מוסף": "Musaf", "שחרית": "Shaharit", "ערבית": "Arvit",
    "מנחה": "Minhah", "נעילה": "Ne'ilah", "עמידה": "Amidah", "הלל": "Hallel", "סוכות": "Sukkot", "פסח": "Pesah",
    "שבועות": "Shavuot", "שבת": "Shabbat", "חול": "Weekday", "תחנון": "Tahanun", "קידוש": "Qiddush",
    "הבדלה": "Havdalah", "סליחות": "Selichot", "הושענות": "Hoshanot", "נשמת": "Nishmat",
}
LITURGY_EN_FIX = {"Selihot": "Selichot", "Yom Ha-Kippurim": "Yom ha-Kippurim"}
OCCASIONS_HE = {"שבת": "Shabbat", "סוכות": "Sukkot", "שבועות": "Shavuot", "פסח": "Pesah",
                "יום הכיפורים": "Yom ha-Kippurim", "ראש השנה": "Rosh ha-Shanah", "שמחת תורה": "Simhat Torah",
                "שמיני עצרת": "Shemini Atzeret"}
PIYYUT_GENRES_HE = {"הושענות": "Hoshanot", "יוצר": "Yotzer", "קדושתא": "Qedushta", "קרובה": "Qerovah",
                    "סליחות": "Selihot", "קינות": "Qinot"}
PIYYUT_HEAD_EN = re.compile(
    r"^(Hoshanot|Simhat Torah|Yotzer|Sukkot|Selihot|Yom Ha-Kippurim|Shevu[`']ot|Shemini Atzeret|Pesah|Qedushta|"
    r"Shabbat|Reshut|Ma'ariv|Qerovah|Silluq|Ofan|Zulat|Qinot|Avodah|Azharot|Purim|Hanukkah|Rosh ha-Shanah|"
    r"Versified Grace after Meals)\b(.*)$", re.I)
DOC_TYPES_EN = ["Ketubbot", "Shidduchin", "Bill of Testimony", "Deed of Sale", "Gittin", "Wills", "Gift Note",
                "Document forms", "Loan Note", "Bill of Consent", "Power of Attorney", "Cheque", "Purchase of Goods",
                "Bill of Guardianship", "Get Halitzah"]
DOC_TYPES_HE = {"כתובות": "Ketubbot", "שידוכין ואירוסין": "Shidduchin", "שטר מכר": "Deed of Sale", "גטין": "Gittin",
                "גיטין": "Gittin", "שטר פשרה": "Deed of Compromise", "שטר מחילה": "Deed of Release",
                "שטר מתנה": "Gift Note", "שטר הודאות": "Deed of Acknowledgment", "צוואות": "Wills"}
LETTER_TYPES = {"Business": "Business", "Public issues": "Public issues", "Family": "Family",
                "עסקים": "Business", "ענייני ציבור": "Public issues", "משפחה": "Family"}

# ---------------------------------------------------------------------------------------------------------------
# KTIV domains
# ---------------------------------------------------------------------------------------------------------------
DOMAIN_DROP = {"unspecified domain", "unidentified", "illegible", "cannot be determined from the catalogue",
               "unspecified (nature of text unclear after initial inspection)", "blank", "other",
               "ancillaries to the main work", "title pages", "indices", "table of contents"}
DOMAIN_ALIASES = {
    "piyut and its interpretation": "Piyyut", "talmud bavli: texts and anthologies": "Talmud Bavli",
    "bible: texts": "Bible", "bible: texts and translations": "Bible", "mishnah: texts": "Mishnah",
    "mishnah: texts and translations": "Mishnah", "mishnah: translations": "Mishnah",
    "halakhic": "Halakhic Literature and Talmudic Commentaries", "court records": "Court Documents",
    "court registers": "Court Documents", "calendars": "Calendar", "ethical literature": "Ethical Literature",
    "פרשנות מקרא": "Biblical Exegesis", "דתות אחרות": "Other Religions",
    "teaching aids,pen trials,writing exercises,scribblings,jotting": "Teaching Aids, Pen Trials, Writing Exercises",
}
DOMAIN_HE = {  # KTIV's own Hebrew labels for the same domains (they appear in the corpus descriptions)
    "Halakhic Literature and Talmudic Commentaries": "ספרות הלכתית ופרשנות תלמודית", "Rabbinic Literature": "ספרות חז\"ל",
    "Documentary": "תעודות", "Bible": "מקרא", "Piyyut": "פיוט", "Talmud Bavli": "תלמוד בבלי",
    "Mishneh Torah and its Commentaries": "משנה תורה ופירושיו", "Liturgy and Brakhot": "תפילה וברכות",
    "Letters": "מכתבים", "Common Prayers": "תפילות קבע", "Biblical Exegesis": "פרשנות מקרא", "Mishnah": "משנה",
    "Secular Poetry": "שירת חול", "Medicine": "רפואה", "Halakhot ha-Rif and its Commentaries": "הלכות הרי\"ף ופירושיו",
    "Personal Status Documents and Legal documents": "תעודות אישיות ושטרות", "Other Religions": "דתות אחרות",
    "Kabbalah": "קבלה", "Midrash": "מדרש", "Brakhot": "ברכות", "Liturgical additions": "תוספות של סידור",
}

# ---------------------------------------------------------------------------------------------------------------
# PGP tag curation
# ---------------------------------------------------------------------------------------------------------------
# (category, "subject|subject", "raw tag; raw tag") — raw tags are matched case-insensitively.
PGP_MERGES: List[Tuple[str, str, str]] = [
    ("family", "marriage", "marriage; wedding; remarriage; minor marriage; muslim marriage contract"),
    ("family", "ketubba|marriage", "ketubba; ketubbah"),
    ("family", "ketubba|marriage|karaites", "karaite ketubba"),
    ("family", "marriage|karaites", "qaraite marriage; karaite-rabbanite marriage"),
    ("family", "betrothal|marriage", "betrothal; engagement; shiddukhin; erusin; prenuptial"),
    ("family", "polygyny|marriage", "polygyny"),
    ("family", "marital-dispute|marriage",
     "marital dispute; marital conflict; marital strife; marital reconciliation; marital; distant husband; alimony"),
    ("family", "levirate-marriage|marriage", "levirate marriage; halisa"),
    ("family", "dowry|marriage", "dowry; dowry list; trousseau"),
    ("family", "divorce", "divorce; get; conditional divorce; aguna"),
    ("family", "widows", "widow; widows"),
    ("family", "orphans", "orphans; orphan; orphan girl"),
    ("family", "guardianship", "guardianship"),
    ("family", "family", "family; children; son; mother; distant son"),
    ("family", "women", "women's letters; women's letter; women; women's; women in business"),
    ("family", "pregnancy|women", "pregnancy; childbirth"),
    ("law", "responsa", "responsum; responsa; legal query"),
    ("law", "formulary", "formulary"),
    ("law", "testimony", "testimony"),
    ("law", "court-records", "court record; register; court notebook"),
    ("law", "muslim-courts", "muslim courts; muslim court; islamic court; qadi; qadi court; fatwa; mazalim"),
    ("law", "power-of-attorney", "power of attorney"),
    ("law", "inheritance", "inheritance; estate; female inheritance"),
    ("law", "wills|inheritance", "will; deathbed will"),
    ("law", "release", "release; quittance; bara'a"),
    ("law", "contracts", "contract"),
    ("law", "acknowledgment", "iqrar; acknowledgment"),
    ("law", "excommunication", "excommunication"),
    ("law", "oaths", "oath"),
    ("law", "slavery", "slavery; slaves; slave; jariya; ghulam; eunuch"),
    ("law", "manumission|slavery", "manumission; freedwoman; get shihrur"),
    ("law", "prison", "prison; prisoners; arrest"),
    ("law", "captives", "captives; ransom"),
    ("economy", "debt", "debt; loan; installments; interest"),
    ("economy", "partnership", "partnership; commenda"),
    ("economy", "sale", "bill of sale; sale"),
    ("economy", "real-estate", "real estate; property; house; rent; lease; hikr"),
    ("economy", "receipts", "receipt; commercial receipt"),
    ("economy", "receipts|taxes", "tax receipt"),
    ("economy", "order-of-payment", "order of payment; cheque; payment"),
    ("economy", "suftaja", "suftaja"),
    ("economy", "accounts", "account; accounts; ledger"),
    ("economy", "accounts|taxes", "fiscal register"),
    ("economy", "trade", "trade; business; commerce; merchant; commodities; prices; markets; broker; business dispute"),
    ("economy", "books|trade", "book trade"),
    ("economy", "books", "booklist; codices"),
    ("economy", "shipping|trade", "shipping; ship; ships; movement of ships; shipwreck; ottoman shipping; boat"),
    ("economy", "flax|textiles", "flax"),
    ("economy", "silk|textiles", "silk"),
    ("economy", "indigo|textiles", "indigo"),
    ("economy", "textiles",
     "textiles; clothing; clothes; cloth; garments; wool; cotton; tiraz; dye; dyeing; turban"),
    ("economy", "pepper|spices", "pepper"),
    ("economy", "spices", "spices; cinnamon; saffron"),
    ("economy", "sugar", "sugar; sugarcane"),
    ("economy", "wine", "wine"),
    ("economy", "cheese|food", "cheese"),
    ("economy", "wheat|food", "wheat; grain"),
    ("economy", "food", "food; bread; meat; honey; oil; olive oil; raisins"),
    ("economy", "coins", "coin; coins; numismatics; mint; ashrafi; muayyadi"),
    ("economy", "taxes", "tax; fiscal; customs; maks; kharaj; corvee"),
    ("economy", "poll-tax|taxes", "capitation tax; capitation; jizya; jaliya"),
    ("economy", "tax-farming|taxes", "damin; daman"),
    ("economy", "iqta", "iqta'"),
    ("economy", "waqf", "waqf"),
    ("economy", "heqdesh", "qodesh; heqdesh"),
    ("economy", "charity",
     "charity; donors; recipients; late donations; donations ledger ca.1800; alms; fundraising; contribution"),
    ("economy", "charity|medicine", "medical charity"),
    ("economy", "poverty", "poverty; begging"),
    ("medicine", "medicine|poverty", "illness: poverty"),
    ("medicine", "medicine|women", "illness: women's"),
    ("medicine", "physicians|medicine",
     "physician; physicians; illness: physician; illness: physicians; contracting for a cure"),
    ("medicine", "pharmacology|medicine",
     "prescription; illness: prescription; materia medica; mail-order medicine; druggist; pharmacy; drug store; "
     "medical books; tutty; syrup; opium"),
    ("medicine", "medicine",
     "medical; medicine; illness; illness letter 969-1517; late illness; heartsickness; leprosy; hospital; "
     "bloodletting; ophthalmic; bonesetter; liver; threatening to die"),
    ("medicine", "medicine|tiberias", "tiberias lepers"),
    ("medicine", "recipes", "recipe; recipes"),
    ("religion", "karaites", "karaite; karaites; qaraite; qaraites; qaraite-rabbanite relations"),
    ("religion", "synagogue|karaites", "dar simha"),
    ("religion", "synagogue", "synagogue; synagogues; late synagogues"),
    ("religion", "communal-affairs", "communal; jewish community; circular"),
    ("religion", "communal-strife", "communal strife; controversy; conflict; interconfessional conflict"),
    ("religion", "calendar", "calendar; calendrical; coptic calendar"),
    ("religion", "calendar|communal-strife", "calendar controversy"),
    ("religion", "education", "education; learning; teacher; apprenticeship"),
    ("religion", "geonic-academies", "yeshiva; babylonian academies; babylonian geonim; geonim; aluf"),
    ("religion", "nagid", "nagid"),
    ("religion", "magic", "magic; amulet; spell; curse; curses; jinn; occult"),
    ("religion", "divination", "prognostication; geomancy; divination; dreams"),
    ("religion", "astrology|divination", "astrology"),
    ("religion", "alchemy", "alchemy"),
    ("religion", "prayer", "prayer; liturgy"),
    ("religion", "fasting", "fasting"),
    ("religion", "kashrut", "kashrut; shehita; kosher; butchers"),
    ("religion", "circumcision", "circumcision"),
    ("religion", "conversion", "conversion; convert; converso; apostasy; apostate"),
    ("religion", "bible", "bible; arabic bible"),
    ("religion", "mourning", "mourning; death; funeral; funerals; memorial; memorial list; cemetery; dirge; report of death"),
    ("religion", "condolence|mourning", "condolence; grief; bereavement"),
    ("religion", "christians", "christians; christian"),
    ("religion", "mustarib", "musta'rib; musta'ribim"),
    ("letters", "recommendation", "recommendation; lor"),
    ("letters", "personal-letters", "personal; private"),
    ("letters", "petitions", "petition; petitioning"),
    ("letters", "appeals", "appeal"),
    ("letters", "state-documents", "decree; state; chancery; fatimid decree; tawqi'; ottoman state; government"),
    ("letters", "officials",
     "amir; wali; vizier; sultan; officials; consul; isfahsalar; sahib al-diwan; katib; turjuman; police; diwan; "
     "diwan al-abwab"),
    ("literature", "poetry", "poem; poetry; arabic poetry; judaeo-arabic poetry; panegyric; classical arabic poetry"),
    ("literature", "popular-literature", "popular literature; 'antar; al-faraj ba'd al-shidda"),
    ("history", "war", "war; military; army; arabic military report; navy; siege; janissary"),
    ("history", "crusades", "crusades; crusade; crusaders; franks; arabic crusades"),
    ("history", "byzantium", "byzantines; byzantium; byzantine merchants"),
    ("history", "travel", "travel; travels; pilgrim; pilgrims; hajj"),
    ("history", "violence", "violence; brawl"),
]
PLACE_ALIASES = {"dimyat": "damietta", "old cairo": "fustat", "constantinople": "istanbul",
                 "late jerusalem": "jerusalem", "late alexandria": "alexandria", "late palestine": "palestine",
                 "al-andalus": "spain", "rashid": "rosetta"}
PLACES = (
    "india; alexandria; damascus; jerusalem; yemen; sicily; tyre; tiberias; qayrawan; fayyum; ramla; salonica; venice; "
    "spain; toledo; almeria; bilbays; aleppo; safed; istanbul; constantinople; damietta; dimyat; ashqelon; al-mahalla; "
    "minyat zifta; dammuh; qalyub; qasr ibrim; rashid; cairo; fustat; old cairo; aden; tinnis; qus; crete; malij; "
    "hebron; pisa; palermo; livorno; marseilles; gaza; acre; tripoli; aydhab; baghdad; maghrib; libya; france; "
    "al-mahdiyya; maldives; java; rhodes; bulaq; minyat ghamr; damira; damsis; red sea; nile; giza; mosul; egypt; "
    "iraq; syria; palestine; al-andalus; late jerusalem; late alexandria; late palestine; qasr al-sham'; "
    "qasr al-rum; al-ashmunayn")
PGP_DROP: Dict[str, str] = {
    "workflow": "dimme; fgp stub; to edit; to examine; cmp; india book unedited; unedited; unedited 11th c; "
                "halfon-addenda; non-geniza; fotm; lost archive; late prosopography; joinfinder; interesting shape; "
                "bizarre",
    "unclear": "aodeh; nar; perahim; order; emet; or; c; b.; islamic",
    "period": "early modern; late; very late; fatimid; mamluk; ottoman; ottoman era; ayyubid; ikhshidid; 4800; "
              "dating tricks; numismatic dating",
    "language": "arabic; arabic script; late arabic script; late arabic; arabic address; arabic literary; late heb; "
                "late ja; ja; heb; ladino; ladino words; ladino literary; judaeo-persian; judaeo-persian literary; "
                "persian; persian literary; latin; latin script; greek; judaeo-greek; italian; spanish; french; german; "
                "yiddish; portuguese; syriac; coptic numerals; coptic alphanumeral; coptic; bialphabetic; macaronic; "
                "unknown language; ottoman turkish; arabic & hebrew; late ja literary; romance",
    "form": "list; legal; report; draft; copy; informal note; letter; literary with documentary value; literary letter; "
            "colophon; autograph; maimonides autograph; avraham maimonides autograph; seal; signature; 'alama; motto; "
            "glyph; pinholes; holes; hinge; reuse; arabic reused; palimpsest; printed; papyrus; vellum; parchment; "
            "paper; red ink; purple ink; ink; illumination; illustration; drawing; micrography; calligraphy; "
            "multihanded; fingerprint; ownership; memorandum; internal memorandum; scribal practice; cipher; daftar; "
            "inventory",
    "person": "karo y frances; ben na'im; mosul nasis; nasir; al-tahirti; al-taharti; al-tustari; cesana; bibas; castro; "
              "de curiel; gatenio; anatoli; romano; pinto; mondolfo; shtiwi; aghion; conforte; meyuhas; frances; "
              "alpalas; sholal; aripol; krispin; zussman; luria; fayruz; bialobos; haman & hefez; avraham haman; "
              "gavriel hefez; merkado karo; shim'on frances; moshe bibas; avraham castro; isaac luria; judge eliyyahu; "
              "yosef rosh ha-seder; saladin; al-malik al-afdal (1094-1121); al-mustansir; madmun; menashshe; yu'bas; "
              "obadiah the proselyte; nisaburi; mosul cantor; qaraite nasis; nasi; nasis; ibn habib's patient; "
              "al-mutanabbi; alemdar; halfon; nahray; nissim; isaac; samuel; ha-sefaradi",
}
PGP_PATTERNS: List[Tuple[str, str, Optional[str]]] = [
    # (category, regex, subjects or None for drop); checked after the explicit tables
    ("medicine", r"^illness\b", "medicine"),
    ("medicine", r"^epidemic\b", "epidemic|medicine"),
    ("medicine", r"^disability\b", "disability"),
    ("period", r"^\d{1,2}(st|nd|rd|th)\b.*\bc\b|^\d{1,2}(st|nd|rd|th) c", None),
    ("workflow", r"^ib[-\s\d]|^ib$|^india book|unedited|addenda|^to (edit|examine)|stub$", None),
    ("person", r"\s(b|bt)\.\s|^(ibn|abu|abū|ben)\s|maimonides|\bgaon$|\bha-(levi|kohen)$|\bkohen$", None),
]
PGP_SUBJECT_HE = {  # Hebrew names where Sefaria has none (or a different slug); reviewed by hand
    "charity": ["צדקה"], "synagogue": ["בית כנסת"], "testimony": ["עדות"], "court-records": ["פנקס בית דין"],
    "responsa": ["שאלות ותשובות", "שו\"ת"], "betrothal": ["שידוכין", "אירוסין"], "polygyny": ["ריבוי נשים"],
    "levirate-marriage": ["ייבום", "חליצה"], "ketubba": ["כתובה"], "sale": ["שטר מכר"], "power-of-attorney": ["הרשאה"],
    "release": ["שטר מחילה"], "wills": ["צוואה"], "heqdesh": ["הקדש"], "geonic-academies": ["ישיבות הגאונים", "גאונים"],
    "nagid": ["נגיד"], "communal-affairs": ["ענייני ציבור", "קהילה"], "epidemic": ["מגפה"], "pharmacology": ["רוקחות"],
    "silk": ["משי"], "sugar": ["סוכר"], "cheese": ["גבינה"], "textiles": ["אריגים", "בדים"], "shipping": ["ספנות"],
    "war": ["מלחמה"], "crusades": ["מסעי הצלב"], "recommendation": ["מכתב המלצה"], "condolence": ["ניחומים"],
    "personal-letters": ["מכתבים אישיים"], "state-documents": ["תעודות ממשלתיות"], "officials": ["פקידי השלטון"],
    "slavery": ["עבדות", "עבדים"], "manumission": ["שחרור עבדים", "גט שחרור"], "family": ["משפחה"],
    "christians": ["נוצרים"], "divorce": ["גירושין", "גט"], "debt": ["חוב", "הלוואה"], "dowry": ["נדוניה"],
    "taxes": ["מסים"], "poll-tax": ["מס הגולגולת", "ג'אליה"], "accounts": ["חשבונות"], "real-estate": ["נכסי דלא ניידי"],
    "fustat": ["פסטאט"], "cairo": ["קהיר"], "sicily": ["סיציליה", "צקליה"], "spain": ["ספרד"], "yemen": ["תימן"],
    "qayrawan": ["קירואן"], "tyre": ["צור"], "ramla": ["רמלה"], "aleppo": ["חלב"], "safed": ["צפת"],
    "istanbul": ["קושטא", "קונסטנטינופול"], "damietta": ["דמיאט"], "tiberias": ["טבריה"], "salonica": ["שאלוניקי"],
    "aden": ["עדן"], "rosetta": ["רשיד"], "medicine": ["רפואה", "חולי"], "mourning": ["אבלות"],
    "petitions": ["בקשה לשלטון", "עצומה"], "appeals": ["בקשת עזרה"], "receipts": ["שובר", "קבלה"],
    "formulary": ["ספר שטרות", "טופס שטר"], "marital-dispute": ["סכסוך בין בני זוג"],
    "muslim-courts": ["בית דין מוסלמי", "קאדי"], "prison": ["בית סוהר", "מאסר"], "popular-literature": ["ספרות עממית"],
    "alchemy": ["אלכימיה"], "divination": ["ניחוש", "גורלות"], "recipes": ["מרשמים"], "pepper": ["פלפל"],
    "order-of-payment": ["המחאה"], "suftaja": ["סופתג'ה"], "tax-farming": ["חכירת מסים"], "communal-strife": ["מחלוקת"],
    "mustarib": ["מוסתערבים"], "acknowledgment": ["שטר הודאה"], "waqf": ["וקף"], "iqta": ["אקטאע"],
    "byzantium": ["ביזנטיון"], "venice": ["ויניציאה", "ונציה"], "fayyum": ["פיום"], "bilbays": ["בלביס"],
    "qasr-ibrim": ["קצר אברים"],
}
PGP_SUBJECT_SEFARIA = {"ketubba": "ketubah", "slavery": "slaves", "taxes": "taxes", "dowry": "dowries",
                       "debt": "loans", "divorce": "divorce", "widows": "widows", "orphans": "orphans",
                       "physicians": "physicians", "conversion": "conversion"}
PGP_SUBJECT_EN = {"geonic-academies": "Geonic academies (yeshivot)", "poll-tax": "poll tax (jizya, jaliya)",
                  "tax-farming": "tax farming (daman)", "heqdesh": "heqdesh (Jewish pious foundations)",
                  "waqf": "waqf (Islamic pious foundation)", "suftaja": "suftaja (bill of exchange)",
                  "iqta": "iqta' (land-revenue grant)", "mustarib": "Musta'rib (Arabised Jews)",
                  "order-of-payment": "order of payment (cheque)", "rosetta": "Rosetta (Rashid)"}
PGP_DOCUMENTARY_TYPES = {"Letter", "Legal document", "List or table", "State document",
                         "Credit instrument or private receipt", "Legal query or responsum", "Inscription"}

# ---------------------------------------------------------------------------------------------------------------
# canon contexts, incidental mentions, law names, name filters
# ---------------------------------------------------------------------------------------------------------------
# KTIV domain labels appended to descriptions ("Passover Haggadah. Liturgical additions Bible: Texts and Translations
# Liturgy and Brakhot") or listed in Hebrew ("מקרא [טקסט];תפילות קבע") are classifications, not canon citations, so the
# scripture / Mishnah *text* labels are stripped before the canon test. A bare "Bible" / "מקרא" and the tractate /
# law-code labels ("Talmud Bavli: Texts and Anthologies", "Mishneh Torah and its Commentaries", "Halakhot ha-Rif and
# its Commentaries") stay: there the festival mention is the tractate or law-code section.
KTIV_DOMAIN_LABELS_EXTRA = ("Aramaic Targumim", "Arabic Tafsir", "מקרא [טקסט]", "מקרא [טקסט", "מקרא ותרגומים",
                            "משנה [טקסט]", "משנה [טקסט", "תרגומים ארמיים", "תפסיר ערבי")
KTIV_DOMAIN_LABELS = tuple(sorted({k for k, v in DOMAIN_ALIASES.items() if v in ("Bible", "Mishnah")}
                                  | set(KTIV_DOMAIN_LABELS_EXTRA), key=lambda s: (-len(s), s)))
KTIV_DOMAIN_TOKEN_RE = re.compile("|".join(re.escape(label) for label in KTIV_DOMAIN_LABELS), re.I)
_HE_TRACTATE_ALT = "|".join(re.escape(t) for t in sorted(HE_TRACTATES, key=lambda s: (-len(s), s)))
# A canon context (scripture, tractate, law code, or a folio of one): a festival named there is the tractate or scroll.
# Esther counts only in a citation context ("Book of Esther", "[Bible]: Esther", "Esther 3:4", "מגילת אסתר") -- in
# letters and ketubbot it is a personal name -- or as a whole name / anchor ("Esther", "אסתר"), so that the training mix
# can test vocabulary names and query anchors with this regex too.
FESTIVAL_CANON_RE = re.compile(
    r"\b(?:talmud|mishnah?|tosefta|bible|targum|tafsir|rif|alfasi|mishneh torah|halakhot gedolot|pesa[hḥ]im"
    r"|megill?ah|yoma)\b"
    rf"|\b(?:rosh ha-?shanah|sukkah?|yoma|pesa[hḥ]im|megill?ah|ta{AY}?anit|sh[ae]vu{AY}?ot) \d{{1,3}}[ab]?\b"
    r"|\b(?:book|scroll) of esther\b|\bmegill?at esther\b|\]:\s*esther\b|\besth(?:er|\.)\s*\d"
    r"|\bcommentary (?:on|to) (?:the book of )?esther\b|\btargum(?: sheni)? (?:on |to |of )?esther\b"
    r"|^\W*(?:esther|אסתר)\W*$"
    rf"|{HB}(?:תלמוד|משנה|מקרא|תוספתא|פסחים|יומא|מגילה|מסכת){HE}|{HB}מס['׳] ?(?:{_HE_TRACTATE_ALT}){HE}"
    r"|רי[\"״]ף|הלכות גדולות|רמב[\"״]ם|(?<![א-ת])ע[\"״][אב](?![א-ת])"
    r"|מגילת אסתר|ספר אסתר|\[מקרא\]:\s*אסתר|(?:פירוש|תרגום|תפסיר|שרח) (?:על |ל)?(?:מגילת )?אסתר"
    r"|אסתר\s+[א-י]['׳]?\s*[,:]",
    re.I)
# Liturgy / poetry / homily cues: a festival named beside one of them is festival content even next to a canon word
# ("פיוט פורים פירוש אסתר ג: א-ה", "Purim and Passover prayers ... Mishnah: Texts", "Sermon for Simhat Torah"; homilies
# are festival content, as Derashot frames are not canon)
LITURGY_CUE_RE = re.compile(
    r"piyy?u[tṭ]|liturg|prayer|haggad|kinot|qinot|selih|selic|mah[sz]or|siddur|hymn|\bʿ?amid(?:a|ah)\b|musaf"
    r"|qedusht|qerov|\bsermon|\bhomil|\bderash|\bderush"
    rf"|פיוט|תפיל|סידור|מחזור|{HB}הגד(?:ה|ות){HE}|קינות|סליח|עמידה|{HB}מוסף{HE}|קדושתא|{HB}דרש(?:ה|ות){HE}", re.I)
# A festival named only as a date ("Dated in the top left corner Purim 1419", "Written on the eve of Sukkot")
DATED_CLAUSE_RE = re.compile(r"\b(?:dated|dating|written on)\b[^.,;:|\n]{0,60}", re.I)
# Document types (the FJP Document Type keyword guess) whose festival mention is incidental (a letter "after
# Passover", a deed dated "on Hanukkah"), and those whose festival mention is the canon text (a tractate commentary,
# a haftarah commentary: "bible" / "talmud", canon unless a liturgy / homily cue says otherwise)
FESTIVAL_INCIDENTAL_TYPES = {"letter", "legal", "ketubah", "business", "medical"}
FESTIVAL_CANON_TYPES = {"bible", "talmud"}
# PGP "Paraliterary text" (colophons, ownership notes, title pages) is incidental unless the description has a liturgy /
# homily cue: 6 of the 9 such records naming a festival are liturgy or sermons ("ʿAmida for ... Rosh ha-Shana").
PARALITERARY_TYPE = "Paraliterary text"
WEAK_REASONS = ("canon", "incidental", "title")  # one reason per weak subject: the first (most specific) wins
# Law-name forms ("laws of Pesach", "הלכות סוכה", "hilkhot sukkah ve-lulav", "Mishnah tractate Sukkah", "Sabbath laws"):
# such a name or query is about the law / tractate, so canon carriers are relevant to it. Searched anywhere in the text.
LAW_NAME_RE = re.compile(
    r"(?<![\w-])(?:laws?|rules|regulations|prohibitions?)(?![\w-])"
    r"|\bhil[ck]h?[oa][ts]\b|\bhil\.|\bdin(?:ei|im)\b|\btractate\b|\bmas+e[kc]h?et\b|\bhala[kc]h"
    r"|(?<![א-ת])(?:הלכות|הלכתי(?:ות)?|הל['׳]|דיני|איסורי|מסכת)(?![א-ת])|^דין\s",
    re.I)
# Exact names never used as mention names or subject anchors: ordinary words ("אבל" but, "עד" until, "מזון" in Birkat
# ha-Mazon, "נשים" in Seder Nashim) or English words with another sense ("Get", "Praise", "Fast").
NAME_STOPLIST = {"אבל", "עד", "כשר", "מס", "ספר", "נשים", "מזון", "לוח", "עבד", "ברכה", "שמע", "Get", "Praise", "Fast"}
# Vocabulary names that are a canon work, not an occasion ("Mishnah Shabbat", "Scroll of Esther")
CANON_NAME_RE = re.compile(r"^(?:mishnah|talmud|tosefta|megillat|scroll of|tractate|משנה|תלמוד|מגילת|מסכת)\b", re.I)
# Occasions: a bare name means the day itself, never its tractate or scroll
OCCASION_TOPICS = set(FESTIVAL_TOPICS) | {"shabbat"}


def is_law_name(name: str) -> bool:
    """Whether a name or query is a law-name form (:data:`LAW_NAME_RE`).

    :param name: Vocabulary name or query text.
    :returns: True for "laws of Pesach", "הלכות סוכה", "Hilchot Chanukah", "Mishnah tractate Sukkah"; False for
        "Passover", "פסח", "son-in-law", "בית דין".
    :rtype: bool
    """
    return bool(LAW_NAME_RE.search(name))


def is_canon_context(text: str) -> bool:
    """Whether a description / title cites a canon text, so a festival named in it is the tractate or scroll.

    KTIV scripture / Mishnah text domain labels are stripped first (:data:`KTIV_DOMAIN_TOKEN_RE`), and a liturgy cue
    anywhere in the text cancels the canon reading (:data:`LITURGY_CUE_RE`).

    :param text: Description, Alternate titles part or KTIV general title.
    :returns: True in a canon context.
    :rtype: bool
    """
    return bool(FESTIVAL_CANON_RE.search(KTIV_DOMAIN_TOKEN_RE.sub(" ", text))) and not LITURGY_CUE_RE.search(text)


def incidental_record(doc_type: str, pgp_types: Set[str], description: str = "") -> bool:
    """Whether a record's type makes any festival it names incidental (letters, deeds, accounts, colophons ...).

    :param doc_type: ``Document Type`` value of the semantic_v3 text ("" when absent).
    :param pgp_types: PGP document types of the merged record (:func:`pgp_info`).
    :param description: The record's description (a liturgy / homily cue keeps a PGP "Paraliterary text" judged).
    :returns: True for an incidental type.
    :rtype: bool
    """
    if doc_type in FESTIVAL_INCIDENTAL_TYPES or pgp_types & PGP_DOCUMENTARY_TYPES:
        return True
    return PARALITERARY_TYPE in pgp_types and not LITURGY_CUE_RE.search(description)


def merge_reason(reasons: Dict[str, str], topic: str, reason: str) -> None:
    """Record a weak reason for a topic, keeping the most specific one (:data:`WEAK_REASONS` order).

    :param reasons: topic -> reason, updated in place.
    :param topic: Topic name.
    :param reason: "canon", "incidental" or "title".
    """
    if topic not in reasons or WEAK_REASONS.index(reason) < WEAK_REASONS.index(reasons[topic]):
        reasons[topic] = reason


def description_festival_reasons(description: str, incidental: bool, canon_type: bool = False
                                 ) -> Tuple[Set[str], Dict[str, str]]:
    """Festival topics named in a record's description: strong ones, and weak ones with their reason.

    Weak "incidental": the record's type is documentary (``incidental``), or the festival is named only inside a date
    clause (:data:`DATED_CLAUSE_RE`). Weak "canon": the description is a canon context (:func:`is_canon_context`), or
    the record's Document Type is scripture / Talmud (``canon_type``) and no liturgy / homily cue is present.

    :param description: The record's description (folded semantic_v3 ``Description:`` value).
    :param incidental: The record's type makes a festival mention incidental (:func:`incidental_record`).
    :param canon_type: The record's Document Type is in :data:`FESTIVAL_CANON_TYPES`.
    :returns: (strong topics, {weak topic: reason}).
    :rtype: Tuple[Set[str], Dict[str, str]]
    """
    found = festival_mentions(description)
    if not found:
        return set(), {}
    if incidental:
        return set(), dict.fromkeys(found, "incidental")
    undated = festival_mentions(DATED_CLAUSE_RE.sub(" ", description))
    reasons = dict.fromkeys(found - undated, "incidental")
    rest = found & undated
    if rest and (is_canon_context(description) or (canon_type and not LITURGY_CUE_RE.search(description))):
        reasons.update(dict.fromkeys(rest, "canon"))
        return set(), reasons
    return rest, reasons


def description_festivals(description: str, incidental: bool) -> Tuple[Set[str], Set[str]]:
    """Festival topics named in a record's description, split into strong and weak labels.

    :param description: The record's description.
    :param incidental: The record's type makes a festival mention incidental.
    :returns: (strong topics, weak topics); :func:`description_festival_reasons` also gives the weak reasons.
    :rtype: Tuple[Set[str], Set[str]]
    """
    strong, reasons = description_festival_reasons(description, incidental)
    return strong, set(reasons)


def title_topic_reasons(general: List[str], topic_re: Dict[str, Pattern]) -> Dict[str, str]:
    """Topics found in KTIV general titles, each with its weak reason.

    A festival title match is "canon" when the title names only the tractate / scroll (topic_v2 matches, the festival
    terms do not: "הלכות מגילה") or is a canon context ("Talmud Bavli Sukkah 29a"); every other match is "title".

    :param general: KTIV general titles.
    :param topic_re: topic_v2 regexes.
    :returns: {topic: "canon" | "title"}.
    :rtype: Dict[str, str]
    """
    reasons: Dict[str, str] = {}
    for title in general:
        for t, rx in topic_re.items():
            if rx.search(title):
                canon = t in FESTIVAL_RE and (not FESTIVAL_RE[t].search(title) or is_canon_context(title))
                merge_reason(reasons, t, "canon" if canon else "title")
    return reasons


def is_occasion_subject(sid: str) -> bool:
    """Whether a topic / Sefaria subject is an occasion (festival, fast, Rosh Hodesh, Shabbat).

    Occasion vocabularies drop canon-work names (:data:`CANON_NAME_RE`). Law sections (``sefaria:laws-of-*``,
    ``hilchot-*``, ``dinei-*``), law topics (marriage, divorce, kashrut, niddah, mourning, tefillin / mezuzah ...) and
    ``work:*`` keep them: there a name like "Tractate Niddah" names the subject's law, which the training mix pairs
    with canon carriers only (it is a :data:`LAW_NAME_RE` name).

    :param sid: Subject id.
    :returns: True for ``topic:<occasion>`` and for non-law Sefaria slugs of an occasion.
    :rtype: bool
    """
    kind, slug = sid.split(":", 1)
    if kind == "topic":
        return slug in OCCASION_TOPICS
    if kind != "sefaria" or slug.startswith(SEFARIA_PREFIXES):
        return False
    occasion_slugs = {s for t in OCCASION_TOPICS for s in TOPIC_SEFARIA.get(t, [])}
    return slug in occasion_slugs or bool(festival_mentions(slug.replace("-", " ")))


def vocab_name_kept(sid: str, name: str) -> bool:
    """Whether a vocabulary name stays on a subject (occasions lose their tractate / scroll names).

    :param sid: Subject id.
    :param name: Candidate name.
    :returns: False for "Mishnah Shabbat" on ``topic:shabbat``, "Scroll of Esther" / "Megillah" on ``topic:purim``.
    :rtype: bool
    """
    kind, slug = sid.split(":", 1)
    if kind == "topic" and name in FESTIVAL_NAME_DROP.get(slug, set()):
        return False
    return not (is_occasion_subject(sid) and CANON_NAME_RE.search(name))

# ---------------------------------------------------------------------------------------------------------------
# Sefaria
# ---------------------------------------------------------------------------------------------------------------
# Sefaria lists thematic sources for holidays/values (Psalm 126 under "Yom HaAtzmaut"), so only halakhic sections
# (laws-of-*, hilchot-*, dinei-*) and this hand-picked holiday / ritual / liturgy list are kept.
SEFARIA_PREFIXES = ("laws-of-", "hilchot-", "dinei-")
SEFARIA_EXTRA = {
    "sukkot", "lulav", "etrog", "sukkah", "willows", "myrtles", "the-four-species", "hoshana-rabbah",
    "shemini-atzeret", "simchat-torah", "shofar", "tefillin", "mezuzah", "tzitzit", "ketubah", "get", "divorce",
    "marriage", "kashrut", "meat-and-milk", "slaughter", "chametz", "matzah", "maror", "passover", "haggadah",
    "yom-kippur", "rosh-hashanah", "shabbat", "purim", "hanukkah", "shavuot", "fasting", "mourning", "burial",
    "circumcision", "niddah", "mikveh", "kiddush", "havdalah", "eruv", "megillah", "selichot", "birkat-hamazon",
    "rosh-chodesh", "hallel", "musaf", "shema", "psukei-dezimrah", "dowries", "yibbum", "chalitzah", "terumah",
    "tithes", "shmita", "orlah", "kilayim", "bikkurim", "challah", "seder", "four-cups", "four-questions",
    "the-four-children", "korban-pesach", "bedikat-chametz-biur-chametz", "high-holidays", "the-ten-days-of-repentance",
    "tashlich", "vidui", "shabbat-candles", "kabbalat-shabbat", "shabbat-prayers", "chanukkah", "al-hanisim",
    "sefirat-haomer", "omer-offering", "tisha-bav", "condolences", "mourners-kaddish", "immersion",
    "kiddush-and-havdalah", "mayim-achronim", "netilat-yadayim", "cup-for-the-blessing", "kiddush-hachodesh",
    "shemoneh-esrei", "birkot-hashachar", "arvit", "shacharit", "tachanun", "kaddish", "torah-reading",
    "blessings-over-the-torah", "blessings", "blessings-over-commandments", "blessings-over-physical-benefit", "minyan",
    "synagogues", "the-grooms-blessings", "yom-tov", "prohibitions-of-festival-days", "prohibitions-of-yom-kippur",
    "shalosh-regalim", "rishut-hayachid-rishut-harabim", "order-and-practice-of-prayer", "rosh-hashanah-and-yom-kippur-prayers",
}
SEFARIA_EXCLUDE = {"laws", "halakhah"}
_ADDR = re.compile(r"^(.*?)\s+(\d+[ab]?(?::\d+)*)(?:-(\d+[ab]?(?::\d+)*))?$")
_INF = 10 ** 9


def slugify(text: str) -> str:
    """Lower-case, keep letters (incl. Hebrew) and digits, join with hyphens.

    :param text: Label.
    :returns: Slug.
    :rtype: str
    """
    text = text.lower().replace("'", "").replace("’", "").replace("`", "").replace("\"", "")
    return "-".join(re.findall(r"[a-z0-9א-ת]+", text))


def as_list(value) -> List[str]:
    """Normalise a scalar-or-list catalogue field to a list of strings.

    :param value: Field value.
    :returns: List of strings.
    :rtype: List[str]
    """
    if not value:
        return []
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield rows of a JSONL file.

    :param path: File path.
    :yields: Row dicts.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def ktiv_fields(record: dict) -> Tuple[List[List[str]], List[str], List[str]]:
    """KTIV frames (grouped per scholarly entry, so continuation frames stay with their work), domains, titles.

    :param record: Merged record.
    :returns: (frames per entry, raw domain strings, general titles).
    :rtype: Tuple[List[List[str]], List[str], List[str]]
    """
    ktiv = (record.get("sources") or {}).get("ktiv") or {}
    frames, domains, general = [], [], []
    for entry in ktiv.get("scholarly_entries") or []:
        wc = ((entry.get("subsections") or {}).get("writing_characteristics") or {})
        frames.append(as_list(wc.get("frame")))
        domains += as_list(wc.get("domain"))
        general += as_list(wc.get("general_title"))
    return frames, domains, general


def pgp_info(record: dict) -> Tuple[Set[str], Set[str]]:
    """Raw PGP tags (comma-split) and PGP document types of a record.

    :param record: Merged record.
    :returns: (tags, types).
    :rtype: Tuple[Set[str], Set[str]]
    """
    docs = ((record.get("sources") or {}).get("pgp") or {}).get("documents") or []
    tags, types = set(), set()
    for doc in docs:
        tags.update(t.strip() for t in (doc.get("tags") or "").split(",") if t.strip())
        if doc.get("type"):
            types.add(doc["type"])
    return tags, types


# ---------------------------------------------------------------------------------------------------------------
# frames -> topics and works
# ---------------------------------------------------------------------------------------------------------------
FRAME_RE = re.compile(r"^\[(.*?)\]\]?\s*:?\s*(.*)$", re.S)


def parse_frames(entry_frames: List[str]) -> List[Tuple[str, str]]:
    """Split one entry's frames into (raw work, body); unbracketed frames continue the previous frame's work.

    KTIV frames were split on ";" by the scraper, so "(Hoshanot): ''...''" or "6 b – 56 b" without a
    "[work]:" prefix belongs to the frame before it.

    :param entry_frames: Frames of one scholarly entry, in order.
    :returns: List of (work, body); work is "" when nothing precedes an unbracketed frame.
    :rtype: List[Tuple[str, str]]
    """
    out, work = [], ""
    for frame in entry_frames:
        frame = frame.strip()
        m = FRAME_RE.match(frame)
        if m:
            work, body = m.group(1), m.group(2)
        else:
            body = frame
        out.append((work, body))
    return out


def canonical_work(raw: str) -> str:
    """Normalise a KTIV work name (English or Hebrew) to one canonical label.

    :param raw: Work as written inside the frame brackets.
    :returns: Canonical label, e.g. "Mishneh Torah", "Common Prayers", "Talmud Bavli Commentaries".
    :rtype: str
    """
    work = re.sub(r"\s*\[[^\]]*$", "", raw).replace("`", "'").replace("’", "'").strip(" ,;:")
    if work in WORK_ALIASES:
        return WORK_ALIASES[work]
    if work in WORK_ALIASES.values():
        return work
    for key, val in WORK_CONTAINS:
        if key in work:
            return val
    if "," in work:
        last = work.split(",")[-1].strip()
        # English "Author, Title" -> Title; Hebrew keeps the author unless the title is a known work
        if not HEB.search(work) or last in WORK_ALIASES:
            return WORK_ALIASES.get(last, last)
    return work


def clean_section(body: str) -> str:
    """Strip incipits, notes, parentheticals and numbering from a frame body.

    :param body: Frame text after "[work]:".
    :returns: Cleaned section label (may be empty).
    :rtype: str
    """
    text = re.sub(r"''.*$", "", body, flags=re.S)
    text = re.sub(r"\[[^\]]*\]?|\([^)]*\)?|@", " ", text)
    text = re.split(r"\d", text)[0]
    text = text.replace("?", "").replace("`", "'").replace("’", "'")
    return " ".join(text.split()).strip(" ,;:.–-/")


def prefix_match(text: str, table: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Longest key of ``table`` that ``text`` starts with (whole words).

    :param text: Section text.
    :param table: Hebrew -> English map.
    :returns: (matched key, mapped value) or None.
    :rtype: Optional[Tuple[str, str]]
    """
    for key in sorted(table, key=len, reverse=True):
        if text == key or text.startswith(key + " ") or text.startswith(key + ","):
            return key, table[key]
    return None


def translate_liturgy(section: str) -> str:
    """Translate a Hebrew liturgy section phrase-by-phrase into the English KTIV phrasing.

    :param section: Cleaned section, e.g. "מוסף סוכות עמידה".
    :returns: English label ("Musaf Sukkot Amidah"), or the input when any Hebrew is left untranslated.
    :rtype: str
    """
    out = section
    for he in sorted(LITURGY_HE, key=len, reverse=True):
        out = re.sub(rf"(?<![א-ת]){re.escape(he)}(?![א-ת])", LITURGY_HE[he], out)
    return section if HEB.search(out) else " ".join(out.split())


def piyyut_head(body: str) -> Optional[str]:
    """Genre/occasion of a Piyyut frame from its parenthetical head ("(Hoshanot)", "(יוצר לשבת)").

    Author heads ("(קליר)", "(יניי)") and anything unrecognised return None.

    :param body: Frame body.
    :returns: English genre/occasion label or None.
    :rtype: Optional[str]
    """
    for head in re.findall(r"\(((?:(?!'')[^()@:\[]){2,60})", body):
        head = head.strip(" '")
        m = PIYYUT_HEAD_EN.match(head)
        if m:
            label = m.group(1)
            rest = m.group(2).strip()
            label = {"shevu`ot": "Shavuot", "shevu'ot": "Shavuot", "yom ha-kippurim": "Yom ha-Kippurim"}.get(
                label.lower(), label)
            if label == "Hoshanot":
                return "Hoshanot"
            if rest.lower().startswith("for "):
                occ = rest[4:].strip()
                occ = {"Shevu`ot": "Shavuot", "Shevu'ot": "Shavuot", "Yom Ha-Kippurim": "Yom ha-Kippurim"}.get(occ, occ)
                return f"{label} for {occ}"
            return label
        if head in OCCASIONS_HE:
            return OCCASIONS_HE[head]
        for genre_he, genre in PIYYUT_GENRES_HE.items():
            if head == genre_he:
                return genre
            if head.startswith(genre_he + " ל"):
                if genre == "Hoshanot":
                    return "Hoshanot"
                occ = OCCASIONS_HE.get(head[len(genre_he) + 2:].strip())
                if occ:
                    return f"{genre} for {occ}"
    return None


def work_section(work: str, body: str) -> Optional[Tuple[str, Optional[str]]]:
    """Section label of a frame for its work family (tractate, book, hilkhot, liturgy section ...).

    :param work: Canonical work.
    :param body: Frame body.
    :returns: (english section, hebrew section or None); None when the family needs a section and none is found.
        ("", None) means the work stands on its own.
    :rtype: Optional[Tuple[str, Optional[str]]]
    """
    section = clean_section(body)
    if work in TRACTATE_WORKS:
        if HEB.search(section):
            hit = prefix_match(section, HE_TRACTATES)
            return (hit[1], hit[0]) if hit else None
        return (section, None) if section in KNOWN_TRACTATES else None
    if work in BOOK_WORKS:
        if HEB.search(section):
            hit = prefix_match(section, HE_BOOKS)
            return (hit[1], hit[0]) if hit else None
        section = " ".join(EN_BOOK_FIX.get(w, w) for w in section.split())
        book = next((b for b in EN_BOOKS if section == b or section.startswith(b + " ")), None)
        return (book, None) if book else None
    if work == "Mishneh Torah":
        if not section or len(section.split()) > 6:
            return None
        return (section, None) if not HEB.search(section) else (section, section)
    if work == "Piyyut":
        head = piyyut_head(body)
        return (head, None) if head else None
    if work in LITURGY_WORKS:
        if not section:
            return None
        section = " ".join(LITURGY_EN_FIX.get(w, w) for w in section.split())
        if HEB.search(section):
            en = translate_liturgy(section)
            return en, section
        return section, None
    if work == "Documents":
        if HEB.search(section):
            hit = prefix_match(section, DOC_TYPES_HE)
            return (hit[1], hit[0]) if hit else None
        en = next((t for t in DOC_TYPES_EN if section.startswith(t)), None)
        return (en, None) if en else None
    if work == "Letters":
        hit = prefix_match(section, LETTER_TYPES)
        return (hit[1], hit[0] if HEB.search(hit[0]) else None) if hit else None
    if work in GENERIC_WORKS:
        return None
    return "", None


def _reverse(table: Dict[str, str]) -> Dict[str, str]:
    """Invert a Hebrew -> English map, keeping the first Hebrew spelling of each English value.

    :param table: Hebrew -> English.
    :returns: English -> Hebrew.
    :rtype: Dict[str, str]
    """
    out: Dict[str, str] = {}
    for he, en in table.items():
        out.setdefault(en, he)
    return out


EN_HE_TRACTATES = _reverse(HE_TRACTATES)
EN_HE_BOOKS = _reverse(HE_BOOKS)
EN_HE_LITURGY = _reverse(LITURGY_HE)
EN_HE_PIYYUT = {**_reverse(PIYYUT_GENRES_HE), **_reverse(OCCASIONS_HE), "Selihot": "סליחות"}


def hebrew_section(work: str, en_sec: str) -> Optional[str]:
    """Hebrew rendering of an English section label, when every part of it has a known Hebrew catalogue form.

    :param work: Canonical work.
    :param en_sec: English section ("Sukkah", "Musaf Sukkot Amidah", "Yotzer for Shabbat").
    :returns: Hebrew section ("סוכה", "מוסף סוכות עמידה", "יוצר לשבת") or None.
    :rtype: Optional[str]
    """
    if work in TRACTATE_WORKS:
        return EN_HE_TRACTATES.get(en_sec)
    if work in BOOK_WORKS:
        return EN_HE_BOOKS.get(en_sec)
    if work == "Piyyut":
        if " for " in en_sec:
            genre, occ = en_sec.split(" for ", 1)
            if genre in EN_HE_PIYYUT and occ in EN_HE_PIYYUT:
                return f"{EN_HE_PIYYUT[genre]} ל{EN_HE_PIYYUT[occ]}"
            return None
        return EN_HE_PIYYUT.get(en_sec)
    if work in LITURGY_WORKS:
        out = en_sec
        for en in sorted(EN_HE_LITURGY, key=len, reverse=True):
            out = re.sub(rf"\b{re.escape(en)}\b", EN_HE_LITURGY[en], out)
        return None if re.search(r"[A-Za-z]", out) else out
    return None


def frame_is_canon(topic: str, work: str, text: str) -> bool:
    """Whether a topic_v2 match on one frame comes from the canon text rather than the occasion / practice itself.

    :param topic: Topic matched on the frame.
    :param work: Canonical work of the frame.
    :param text: Frame as "[<work>]: <body>".
    :returns: True when the work is scripture / tractate / law code (:data:`CANON_WORKS`), or, for festivals, when the
        frame names only the tractate / scroll (topic_v2 matches, the festival terms do not: "]: מגילה" under a Rif
        commentary, "Qiddush ha-Hodesh").
    :rtype: bool
    """
    return work in CANON_WORKS or (topic in FESTIVAL_RE and not FESTIVAL_RE[topic].search(text))


def frame_subjects(entries: List[List[str]], topic_re: Optional[Dict[str, Pattern]] = None
                   ) -> Tuple[Set[str], Set[str], Dict[str, Tuple[str, Optional[str]]]]:
    """Topics (topic_v2) and work subjects of one record's frames.

    A topic matched only on canon frames (:func:`frame_is_canon`, e.g. "[Bible]: Esther", "]: Pesahim",
    "[Talmud Bavli]: Ketubbot") is returned as a canon topic, not a topic.

    :param entries: Frames grouped per scholarly entry (:func:`ktiv_fields`).
    :param topic_re: topic_v2 regexes (default :data:`TOPIC_RE`; :data:`TOPIC_RE_LEGACY` reproduces v9).
    :returns: (topic names with a non-canon frame, canon-only topic names (festival or not),
        {work id: (english label, hebrew label or None)}).
    :rtype: Tuple[Set[str], Set[str], Dict[str, Tuple[str, Optional[str]]]]
    """
    topic_re = TOPIC_RE if topic_re is None else topic_re
    topics: Set[str] = set()
    canon: Set[str] = set()
    works: Dict[str, Tuple[str, Optional[str]]] = {}
    for entry in entries:
        for raw_work, body in parse_frames(entry):
            work = canonical_work(raw_work) if raw_work else ""
            text = f"[{raw_work}]: {body}" if raw_work else body
            for t, rx in topic_re.items():
                if not rx.search(text):
                    continue
                if frame_is_canon(t, work, text):
                    canon.add(t)
                else:
                    topics.add(t)
            if work in LITURGY_WORKS:
                topics.update(t for t, rx in TOPIC_LIT_RE.items() if rx.search(text))
            if not work:
                continue
            sec = work_section(work, body)
            if sec is None:
                continue
            en_sec, he_sec = sec
            if en_sec and not he_sec:
                he_sec = hebrew_section(work, en_sec)
            label = f"{work} {en_sec}".strip()
            he_work = WORK_HE.get(work, work if HEB.search(work) else None)
            he_label = f"{he_work} {he_sec}" if he_work and he_sec else (he_work if he_work and not en_sec else None)
            works[f"work:{slugify(label)}"] = (label, he_label)
    return topics, canon - topics, works


# ---------------------------------------------------------------------------------------------------------------
# domains, PGP tags
# ---------------------------------------------------------------------------------------------------------------
def domain_subjects(raw_domains: List[str]) -> Dict[str, str]:
    """KTIV domain subject ids (each " # " level), synonyms merged, uninformative values dropped.

    :param raw_domains: Raw domain strings.
    :returns: {id: english label}.
    :rtype: Dict[str, str]
    """
    out = {}
    for raw in raw_domains:
        for part in raw.split(" # "):
            part = part.strip()
            key = part.lower()
            if not part or key in DOMAIN_DROP:
                continue
            label = DOMAIN_ALIASES.get(key, part)
            out[f"domain:{slugify(label)}"] = label
    return out


def build_pgp_rules() -> Tuple[Dict[str, Tuple[str, List[str]]], Dict[str, str]]:
    """Explicit PGP tag tables -> lookup dicts.

    :returns: ({raw lower: (category, [subject slugs])}, {raw lower: drop category}).
    :rtype: Tuple[Dict[str, Tuple[str, List[str]]], Dict[str, str]]
    """
    merges: Dict[str, Tuple[str, List[str]]] = {}
    for category, subjects, tags in PGP_MERGES:
        for tag in tags.split(";"):
            merges[tag.strip().lower()] = (category, subjects.split("|"))
    for tag in PLACES.split(";"):
        tag = tag.strip().lower()
        merges.setdefault(tag, ("place", [slugify(PLACE_ALIASES.get(tag, tag))]))
    drops = {tag.strip().lower(): cat for cat, tags in PGP_DROP.items() for tag in tags.split(";")}
    return merges, drops


PGP_MERGE_TABLE, PGP_DROP_TABLE = build_pgp_rules()
PGP_PATTERN_RE = [(cat, re.compile(rx, re.I), subj) for cat, rx, subj in PGP_PATTERNS]


def curate_tag(tag: str) -> Tuple[str, List[str], str]:
    """Decide what one raw PGP tag means.

    :param tag: Raw tag.
    :returns: (category, subject slugs ([] = drop), rule) where rule names the table that decided.
    :rtype: Tuple[str, List[str], str]
    """
    key = tag.strip().lower()
    if key in PGP_MERGE_TABLE:
        cat, subjects = PGP_MERGE_TABLE[key]
        return cat, subjects, "table"
    if key in PGP_DROP_TABLE:
        return PGP_DROP_TABLE[key], [], "table"
    for cat, rx, subjects in PGP_PATTERN_RE:
        if rx.search(key):
            return cat, (subjects.split("|") if subjects else []), "pattern"
    return "unreviewed", [slugify(key)], "default"


# ---------------------------------------------------------------------------------------------------------------
# Sefaria
# ---------------------------------------------------------------------------------------------------------------
def _addr(text: str) -> Optional[List[int]]:
    """Parse a Sefaria address ("29a", "13:25", "29b:4") into integers (daf a/b -> 2*daf + side).

    :param text: Address.
    :returns: Integer list, or None.
    :rtype: Optional[List[int]]
    """
    out = []
    for part in text.split(":"):
        m = re.match(r"^(\d+)([ab])?$", part)
        if not m:
            return None
        out.append(int(m.group(1)) * 2 + (m.group(2) == "b") if m.group(2) else int(m.group(1)))
    return out


def parse_ref(ref: str) -> Optional[Tuple[str, tuple, tuple]]:
    """Parse a Sefaria ref into (book, low, high) comparable address tuples.

    Missing lower levels act as wildcards ("Sukkah 29a" covers every segment of 29a); an end address that omits
    leading levels inherits them ("Leviticus 13:25-43").

    :param ref: e.g. "Sukkah 29a-30b", "Mishneh Torah, Shofar, Sukkah and Lulav 7:1", "Leviticus 23:40".
    :returns: (book, low, high) or None.
    :rtype: Optional[Tuple[str, tuple, tuple]]
    """
    m = _ADDR.match(ref.strip())
    if not m:
        return None
    book, a, b = m.groups()
    start = _addr(a)
    end = _addr(b) if b else list(start or [])
    if start is None or end is None:
        return None
    if len(end) < len(start):
        end = start[: len(start) - len(end)] + end
    low = tuple(start) + (0,) * (4 - len(start))
    high = tuple(end) + (_INF,) * (4 - len(end))
    return book.strip(), low, high


def sefaria_whitelisted(slug: str, graph: dict) -> bool:
    """Keep only halakhic / holiday / ritual Sefaria topics.

    :param slug: Topic slug.
    :param graph: ``concept_graph.json``.
    :returns: True when the topic is kept.
    :rtype: bool
    """
    if slug in SEFARIA_EXCLUDE or slug not in graph:
        return False
    return slug.startswith(SEFARIA_PREFIXES) or slug in SEFARIA_EXTRA


def sefaria_subjects(frame_refs: Path, curated: Path, graph: dict) -> Tuple[Dict[str, Set[str]], Counter]:
    """Sefaria topics per record: curated topic refs that overlap the record's frame refs, whitelisted.

    :param frame_refs: ``frame_refs.jsonl``.
    :param curated: ``sefaria/derived/curated_refs.jsonl``.
    :param graph: Concept graph.
    :returns: ({doc_id: {slug}}, stats).
    :rtype: Tuple[Dict[str, Set[str]], Counter]
    """
    by_book: Dict[str, List[Tuple[tuple, tuple, str]]] = defaultdict(list)
    stats: Counter = Counter()
    for row in iter_jsonl(frame_refs):
        parsed = parse_ref(row["ref"])
        stats["frame_refs"] += 1
        if parsed:
            by_book[parsed[0]].append((parsed[1], parsed[2], row["doc_id"]))
    out: Dict[str, Set[str]] = defaultdict(set)
    for row in iter_jsonl(curated):
        stats["curated_refs"] += 1
        if not sefaria_whitelisted(row["topic"], graph):
            stats["curated_not_whitelisted"] += 1
            continue
        parsed = parse_ref(row["ref"])
        if not parsed:
            stats["curated_unparsed"] += 1
            continue
        book, low, high = parsed
        for dlow, dhigh, doc_id in by_book.get(book, []):
            if low <= dhigh and dlow <= high:
                out[doc_id].add(row["topic"])
    return out, stats


# ---------------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------------
def sefaria_names(slugs: Iterable[str], graph: dict, limit: int = 6) -> Tuple[List[str], List[str]]:
    """English and Hebrew names of Sefaria topics.

    :param slugs: Topic slugs.
    :param graph: Concept graph.
    :param limit: Max names per language per topic.
    :returns: (names_en, names_he).
    :rtype: Tuple[List[str], List[str]]
    """
    en, he = [], []
    for slug in slugs:
        node = graph.get(slug) or {}
        titles = [node.get("en") or "", node.get("he") or ""] + list(node.get("titles") or [])
        en += [t for t in titles if t and not HEB.search(t) and not t.startswith("#")][:limit]
        he += [t for t in titles if t and HEB.search(t)][:limit]
    return dedupe(en), dedupe(he)


def dedupe(items: Iterable[str]) -> List[str]:
    """Order-preserving, case-insensitive de-duplication.

    :param items: Strings.
    :returns: Unique strings.
    :rtype: List[str]
    """
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def title_index(graph: dict) -> Dict[str, str]:
    """Lower-cased English Sefaria title -> slug (first wins), for naming PGP / domain subjects.

    :param graph: Concept graph.
    :returns: Title -> slug.
    :rtype: Dict[str, str]
    """
    index: Dict[str, str] = {}
    for slug, node in graph.items():
        for title in [node.get("en") or ""] + list(node.get("titles") or []):
            if title and not HEB.search(title):
                index.setdefault(title.lower(), slug)
    return index


# ---------------------------------------------------------------------------------------------------------------
# per-record topic labels
# ---------------------------------------------------------------------------------------------------------------
class TopicLabels(NamedTuple):
    """Topic labels of one record (topic names, without the ``topic:`` prefix)."""

    strong: Set[str]  # carried strongly (subject-pair carriers, eval relevant)
    weak_reason: Dict[str, str]  # weak topic -> "canon" | "incidental" | "title"
    canon: Set[str]  # canon-only topics: festival ones are weak (reason canon), the others strong
    gold_override: bool  # the Sukkot gold flag turned a weak sukkot label strong
    gold_reason: str  # the weak reason it overrode ("" when none)
    description_strong: Set[str]  # strong only because the description names the festival (stats)
    description_weak: Set[str]  # weak because of the description (stats)
    title_only: Set[str]  # weak because of a KTIV general title only (stats)


def topic_labels(frame_topics: Set[str], frame_canon: Set[str], general: List[str], description: str,
                 incidental: bool, gold: bool, use_titles: bool = True, legacy: bool = False,
                 canon_type: bool = False) -> TopicLabels:
    """Strong and typed weak topic labels of one record.

    :param frame_topics: Topics with a non-canon frame (:func:`frame_subjects`).
    :param frame_canon: Canon-only frame topics (:func:`frame_subjects`).
    :param general: KTIV general titles.
    :param description: Folded semantic_v3 description (:func:`source_fields`).
    :param incidental: The record's type makes festival mentions incidental (:func:`incidental_record`).
    :param gold: The record is in the human-verified Sukkot seed.
    :param use_titles: Match topic_v2 rules against general titles (weak labels).
    :param legacy: v9 labels: every frame topic strong, no description scan, title-only topics weak.
    :param canon_type: The record's Document Type is scripture / Talmud (:data:`FESTIVAL_CANON_TYPES`).
    :returns: The record's labels.
    :rtype: TopicLabels
    """
    if legacy:
        strong = frame_topics | frame_canon
        if gold:
            strong.add("sukkot")
        title_only = set(title_topic_reasons(general, TOPIC_RE_LEGACY)) - strong if use_titles else set()
        return TopicLabels(strong, dict.fromkeys(title_only, "title"), set(), False, "", set(), set(), title_only)
    canon = {t for t in frame_canon if t not in FESTIVAL_RE}  # tractate / law-code topics stay strong, tagged
    strong = frame_topics | canon
    reasons = {t: "canon" for t in frame_canon if t in FESTIVAL_RE}
    desc_strong, desc_reasons = description_festival_reasons(description, incidental, canon_type)
    strong |= desc_strong
    for t, reason in desc_reasons.items():
        merge_reason(reasons, t, reason)
    title_reasons = title_topic_reasons(general, TOPIC_RE) if use_titles else {}
    for t, reason in title_reasons.items():
        merge_reason(reasons, t, reason)
    reasons = {t: r for t, r in reasons.items() if t not in strong}
    gold_reason = ""
    if gold:  # human-verified: stays strong, the override is flagged so the holiday eval can leave it unjudged
        gold_reason = reasons.pop("sukkot", "")
        strong.add("sukkot")
    canon |= {t for t, r in reasons.items() if r == "canon"}
    return TopicLabels(strong, reasons, canon, bool(gold_reason), gold_reason, desc_strong - frame_topics,
                       set(desc_reasons) & set(reasons),
                       (set(title_reasons) & set(reasons)) - set(desc_reasons) - frame_canon)


# ---------------------------------------------------------------------------------------------------------------
# mentions: subjects a record's catalogue text names without carrying them
# ---------------------------------------------------------------------------------------------------------------
NIQQUD_RE = re.compile(r"[֑-ׇ]")
MENTION_PREFIXES = "בהוכלמשד"  # Hebrew prefix letters tolerated on a name's first word (same set as HB)
MIN_MENTION_NAME_CHARS = 3


class MentionIndex(NamedTuple):
    """Vocabulary names as token sequences, indexed by first token."""

    by_first: Dict[str, List[Tuple[str, ...]]]
    by_seq: Dict[Tuple[str, ...], Set[str]]  # token sequence -> subject ids


def mention_tokens(text: str) -> List[str]:
    """Tokens for name matching: niqqud stripped, lower-cased, punctuation (incl. geresh / gershayim, ``_``) as spaces.

    Same rule as ``build_training_mix_v3.norm_tokens``, so a mention is a literal name in the mix's sense.

    :param text: Any text.
    :returns: Token list.
    :rtype: List[str]
    """
    return re.sub(r"[^\w\s]|_", " ", NIQQUD_RE.sub("", text).lower()).split()


def build_mention_index(vocab: Dict[str, dict]) -> MentionIndex:
    """Index the kept vocabulary names (English and Hebrew) for :func:`named_subjects`.

    Names shorter than :data:`MIN_MENTION_NAME_CHARS`, ``#``-names and :data:`NAME_STOPLIST` names (case-insensitive)
    are left out.

    :param vocab: ``subject_vocab.json`` entries.
    :returns: The index.
    :rtype: MentionIndex
    """
    stop = {n.casefold() for n in NAME_STOPLIST}
    by_seq: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    for sid, entry in vocab.items():
        for raw in (entry.get("names_en") or []) + (entry.get("names_he") or []):
            name = " ".join(NIQQUD_RE.sub("", re.sub(r"[\[\]]", "", raw)).split())
            if len(name) < MIN_MENTION_NAME_CHARS or name.startswith("#") or name.casefold() in stop:
                continue
            seq = tuple(mention_tokens(name))
            if seq and sum(map(len, seq)) >= MIN_MENTION_NAME_CHARS:
                by_seq[seq].add(sid)
    by_first: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for seq in by_seq:
        by_first[seq[0]].append(seq)
    return MentionIndex(dict(by_first), dict(by_seq))


def named_subjects(text: str, index: MentionIndex) -> Set[str]:
    """Subject ids whose names occur in ``text`` as whole words, plus festivals by :data:`FESTIVAL_RE`.

    A Hebrew first word may carry one or two prefix letters ("בשבת", "ולפורים").

    :param text: Description and Alternate titles of a record.
    :param index: :func:`build_mention_index` output.
    :returns: Subject ids (``topic:<festival>`` for festival terms).
    :rtype: Set[str]
    """
    toks = mention_tokens(text)
    found = {f"topic:{t}" for t in festival_mentions(text)}
    for i, tok in enumerate(toks):
        cands = [tok]
        if HEB.match(tok):
            cands += [tok[k:] for k in (1, 2) if len(tok) - k >= 2 and all(c in MENTION_PREFIXES for c in tok[:k])]
        for cand in cands:
            for seq in index.by_first.get(cand, ()):
                if len(seq) == 1 or tuple(toks[i + 1:i + len(seq)]) == seq[1:]:
                    found |= index.by_seq[seq]
    return found


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
V3_DOC_TYPE_SHARE = 0.526  # records with a Document Type line in v3/corpus_semantic.jsonl (2026-10-05: 38,691 / 73,543)
DOC_TYPE_SHARE_TOLERANCE = 0.2  # relative; a smaller share means the field labels were not found (plan A5)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Parsed options.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semantic", default=str(V3 / "corpus_semantic.jsonl"))
    parser.add_argument("--merged", default=str(MERGED))
    parser.add_argument("--frame-refs", default=str(AUDIT_ROOT / "frame_refs.jsonl"))
    parser.add_argument("--curated-refs", default=str(AUDIT_ROOT / "sefaria/derived/curated_refs.jsonl"))
    parser.add_argument("--concept-graph", default=str(AUDIT_ROOT / "sefaria/derived/concept_graph.json"))
    parser.add_argument("--out-dir", default=str(V3))
    parser.add_argument("--curation-csv", default=str(RESULTS / "pgp_tag_curation.csv"),
                        help="OUTPUT path of the PGP tag curation sheet (pass one under --out-dir for new builds)")
    parser.add_argument("--min-pgp", type=int, default=20, help="records a curated PGP subject needs")
    parser.add_argument("--min-records", type=int, default=5, help="records a topic/work/domain/sefaria subject needs")
    parser.add_argument("--no-general-titles", action="store_true",
                        help="match topic_v2 rules against frames only, not KTIV general titles (free-text identifications)")
    parser.add_argument("--legacy-festivals", action="store_true",
                        help="reproduce v9's subjects/weak: v9 topic_v2 rules, festival topics on canon frames stay "
                             "strong, no description scan, scroll/tractate names kept in the vocab, no canon list")
    parser.add_argument("--no-sefaria-canon", action="store_true",
                        help="do not list sefaria:* ids in a record's 'canon' (they come only from canon frame refs)")
    parser.add_argument("--no-mentions", action="store_true", help="skip the mentions pass ('mentions' left empty)")
    parser.add_argument("--doc-type-share-ref", type=float, default=V3_DOC_TYPE_SHARE,
                        help="expected share of records with a Document Type line (+-20%% relative); 0 = no check")
    parser.add_argument("--self-test", action="store_true", help="run the table-driven rule tests and exit")
    return parser.parse_args(argv)


def main() -> None:
    """Build subjects_v1.jsonl, subject_vocab.json and the PGP tag curation sheet; print coverage."""
    args = parse_args()
    if args.self_test:
        self_test()
        return
    legacy = args.legacy_festivals
    graph = json.loads(Path(args.concept_graph).read_text())
    titles = title_index(graph)
    sef_by_doc, sef_stats = sefaria_subjects(Path(args.frame_refs), Path(args.curated_refs), graph)

    per_doc: Dict[str, Dict[str, Set[str]]] = {}
    eligible: Dict[str, bool] = {}
    names_en: Dict[str, List[str]] = defaultdict(list)
    names_he: Dict[str, List[str]] = defaultdict(list)
    tag_counts: Counter = Counter()
    pgp_documentary: Set[str] = set()
    has_pgp: Set[str] = set()
    v1_missing: Counter = Counter()
    weak: Dict[str, Dict[str, str]] = {}  # doc -> {topic id: weak reason}
    canon_ids: Dict[str, Set[str]] = {}  # doc -> ids listed in its canon
    gold_override: Dict[str, str] = {}  # doc -> weak reason the gold flag overrode
    catalogue: Dict[str, str] = {}  # doc -> Description + Alternate titles (mentions pass)
    stats: Counter = Counter()
    for sem, record in zip(iter_jsonl(Path(args.semantic)), iter_jsonl(Path(args.merged))):
        doc_id = sem["doc_id"]
        if record["canonical_id"] != doc_id:
            raise ValueError(f"corpus/merged order mismatch at {doc_id} vs {record['canonical_id']}")
        eligible[doc_id] = bool(sem["eligible"])
        entries, raw_domains, general = ktiv_fields(record)
        frame_topics, frame_canon, works = frame_subjects(entries, TOPIC_RE_LEGACY if legacy else TOPIC_RE)
        tags, types = pgp_info(record)
        fields = source_fields(sem)
        stats["records_with_doc_type"] += bool(fields["doc_type"])
        stats["records_with_description"] += bool(fields["description"])
        labels = topic_labels(frame_topics, frame_canon, general, fields["description"],
                              incidental_record(fields["doc_type"], types, fields["description"]),
                              bool(sem.get("sukkot_gold")), use_titles=not args.no_general_titles, legacy=legacy,
                              canon_type=fields["doc_type"] in FESTIVAL_CANON_TYPES)
        topics = labels.strong | set(labels.weak_reason)
        stats["records_with_description_festival"] += bool(labels.description_strong)
        stats["records_with_weak_description_festival"] += bool(labels.description_weak)
        stats["records_with_title_only_topic"] += bool(labels.title_only)
        stats["records_with_canon_festival"] += any(r == "canon" for r in labels.weak_reason.values())
        weak[doc_id] = {f"topic:{t}": r for t, r in labels.weak_reason.items()}
        canon_ids[doc_id] = {f"topic:{t}" for t in labels.canon}
        if not legacy and not args.no_sefaria_canon:
            canon_ids[doc_id] |= {f"sefaria:{s}" for s in sef_by_doc.get(doc_id, ())}
        if labels.gold_override:
            gold_override[doc_id] = labels.gold_reason
        catalogue[doc_id] = f"{fields['description']}\n{fields['alt_titles']}"
        for t in set(sem.get("topics") or []) - topics:
            v1_missing[t] += 1
        domains = domain_subjects(raw_domains)
        if tags or types:
            has_pgp.add(doc_id)
        if types & PGP_DOCUMENTARY_TYPES:
            pgp_documentary.add(doc_id)
        pgp_subjects: Set[str] = set()
        for tag in tags:
            tag_counts[tag] += 1
            _, subjects, _ = curate_tag(tag)
            pgp_subjects.update(f"pgp:{s}" for s in subjects)
        for wid, (en, he) in works.items():
            names_en[wid].append(en)
            if he:
                names_he[wid].append(he)
        for did, label in domains.items():
            names_en[did].append(label)
        per_doc[doc_id] = {
            "topic": {f"topic:{t}" for t in topics}, "work": set(works), "domain": set(domains),
            "pgp": pgp_subjects, "sefaria": {f"sefaria:{s}" for s in sef_by_doc.get(doc_id, ())},
        }
        stats["records"] += 1
    if stats["records"] != len(eligible):
        raise ValueError("duplicate doc ids in corpus")
    if not legacy and not stats["records_with_description_festival"]:
        raise ValueError("no description names a festival: Description lines not found (text_v3 missing?)")
    doc_type_share = stats["records_with_doc_type"] / max(1, stats["records"])
    if args.doc_type_share_ref and abs(doc_type_share / args.doc_type_share_ref - 1) > DOC_TYPE_SHARE_TOLERANCE:
        raise ValueError(f"Document Type share {doc_type_share:.3f} is not within {DOC_TYPE_SHARE_TOLERANCE:.0%} of "
                         f"{args.doc_type_share_ref}: field labels not found (text_v3 missing?)")

    # thresholds
    counts: Counter = Counter()
    elig_counts: Counter = Counter()
    for doc_id, kinds in per_doc.items():
        for ids in kinds.values():
            counts.update(ids)
            if eligible[doc_id]:
                elig_counts.update(ids)
    keep = {sid for sid, n in counts.items()
            if n >= (args.min_pgp if sid.startswith("pgp:") else args.min_records)}

    # PGP curation sheet
    raw_by_subject: Dict[str, Counter] = defaultdict(Counter)
    category_of: Dict[str, str] = {}
    Path(args.curation_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.curation_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["raw_tag", "n_records", "decision", "category", "rule", "subjects", "kept_subjects"])
        for tag, n in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0].lower())):
            cat, subjects, rule = curate_tag(tag)
            if not subjects:
                decision = "drop"
            elif rule == "default":
                decision = "keep-unreviewed"  # own subject, not hand-checked; only matters if it reaches --min-pgp
            elif subjects == [slugify(tag)]:
                decision = "keep"
            else:
                decision = "merge->" + "|".join(subjects)
            kept = [s for s in subjects if f"pgp:{s}" in keep]
            writer.writerow([tag, n, decision, cat, rule, "|".join(subjects), "|".join(kept)])
            if subjects:  # a tag names its first (primary) subject; the others are broader parents
                raw_by_subject[subjects[0]][tag] += n
            for s in subjects:
                category_of.setdefault(s, cat)

    # vocabulary
    vocab: Dict[str, dict] = {}
    names_dropped: Dict[str, List[str]] = {}
    for sid in sorted(keep, key=lambda s: (s.split(":")[0], -counts[s], s)):
        kind, slug = sid.split(":", 1)
        entry = {"kind": kind, "n_records": counts[sid], "n_eligible": elig_counts[sid]}
        if kind == "topic":
            phrases = TOPIC_PHRASES.get(slug, []) + TOPIC_EXTRA_PHRASES.get(slug, [])
            sen, she = sefaria_names(TOPIC_SEFARIA.get(slug, []), graph)
            en = [p for p in phrases if not HEB.search(p)] + sen
            he = [p for p in phrases if HEB.search(p)] + she
            if not legacy:
                names_dropped[sid] = [n for n in en + he if not vocab_name_kept(sid, n)]
                en = [n for n in en if vocab_name_kept(sid, n)]
                he = [n for n in he if vocab_name_kept(sid, n)]
            entry.update(names_en=dedupe(en), names_he=dedupe(he),
                         source="KTIV frames + general titles (topic_v2 rules)" + (" + Sukkot seed" if slug == "sukkot" else ""))
        elif kind == "work":
            labels = [n for n, _ in Counter(names_en[sid]).most_common()]
            he_labels = [n for n, _ in Counter(names_he[sid]).most_common()]
            entry.update(names_en=dedupe(n for n in labels if not HEB.search(n)),
                         names_he=dedupe(he_labels + [n for n in labels if HEB.search(n)]), source="KTIV frames")
        elif kind == "domain":
            label = names_en[sid][0]
            sen, she = sefaria_names([titles[label.lower()]] if label.lower() in titles else [], graph, limit=3)
            entry.update(names_en=dedupe([label]), names_he=dedupe(([DOMAIN_HE[label]] if label in DOMAIN_HE else [])),
                         source="KTIV domain")
        elif kind == "pgp":
            display = PGP_SUBJECT_EN.get(slug, slug.replace("-", " "))
            if category_of.get(slug) == "place":
                display = display.title()
            raw = [t for t, _ in raw_by_subject[slug].most_common(10)]
            alt = [t for t in raw if ":" not in t and not re.search(r"\d", t)][:6]
            sef_slug = PGP_SUBJECT_SEFARIA.get(slug) or titles.get(slug.replace("-", " "))
            sen, she = sefaria_names([sef_slug] if sef_slug else [], graph, limit=4)
            entry.update(names_en=dedupe([display] + alt + sen), names_he=dedupe(PGP_SUBJECT_HE.get(slug, []) + she),
                         category=category_of.get(slug, "unreviewed"), raw_tags=raw, source="PGP tags (curated)")
        else:
            sen, she = sefaria_names([slug], graph)
            if not legacy:
                names_dropped[sid] = [n for n in sen + she if not vocab_name_kept(sid, n)]
                sen = [n for n in sen if vocab_name_kept(sid, n)]
                she = [n for n in she if vocab_name_kept(sid, n)]
            entry.update(names_en=sen, names_he=she, source="Sefaria topic via frame_refs x curated_refs")
        vocab[sid] = entry

    # subjects (+ the mentions pass, which needs the final vocabulary)
    index = None if args.no_mentions else build_mention_index(vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind_cover: Counter = Counter()
    kind_cover_elig: Counter = Counter()
    reason_counts: Counter = Counter()  # (subject id, weak reason)
    canon_counts: Counter = Counter()
    gold_counts: Counter = Counter()
    mention_counts: Counter = Counter()
    mention_dist: Counter = Counter()
    with open(out_dir / "subjects_v1.jsonl", "w", encoding="utf-8") as fh:
        for doc_id, kinds in per_doc.items():
            subjects = sorted(s for ids in kinds.values() for s in ids if s in keep)
            subject_set = set(subjects)
            weak_reason = {s: r for s, r in sorted(weak[doc_id].items()) if s in subject_set}
            weak_ids = sorted(weak_reason)
            strong = subject_set - set(weak_ids)
            canon = sorted(canon_ids[doc_id] & subject_set)
            mentions = sorted((named_subjects(catalogue[doc_id], index) & keep) - strong) if index else []
            gold = doc_id in gold_override and "topic:sukkot" in subject_set
            fh.write(json.dumps({"doc_id": doc_id, "subjects": subjects, "weak": weak_ids, "weak_reason": weak_reason,
                                 "canon": canon, "mentions": mentions, "gold_override": gold},
                                ensure_ascii=False) + "\n")
            stats["weak_assignments"] += len(weak_ids)
            reason_counts.update(weak_reason.items())
            canon_counts.update(canon)
            if gold:
                gold_counts[gold_override[doc_id]] += 1
            mention_counts.update(mentions)
            mention_dist[min(len(mentions), 20)] += 1
            stats["mention_assignments"] += len(mentions)
            stats["mention_assignments_eligible"] += len(mentions) * eligible[doc_id]
            stats["records_with_mention"] += bool(mentions)
            present = {s.split(":")[0] for s in subjects}
            for kind in present | ({"any"} if subjects else set()):
                kind_cover[kind] += 1
                kind_cover_elig[kind] += eligible[doc_id]
            stats["subject_assignments"] += len(subjects)
            if doc_id in pgp_documentary:
                stats["pgp_documentary"] += 1
                stats["pgp_documentary_any"] += bool(subjects)
                stats["pgp_documentary_pgp_subject"] += any(s.startswith("pgp:") for s in subjects)
                stats["pgp_documentary_eligible"] += eligible[doc_id]
                stats["pgp_documentary_eligible_any"] += bool(subjects) and eligible[doc_id]
            if doc_id in has_pgp:
                stats["pgp_records"] += 1
                stats["pgp_records_pgp_subject"] += any(s.startswith("pgp:") for s in subjects)
    (out_dir / "subject_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1))

    weak_counts: Counter = Counter()
    reason_totals: Counter = Counter()
    for (sid, reason), n in reason_counts.items():
        weak_counts[sid] += n
        reason_totals[reason] += n
    mention_kinds: Counter = Counter()
    for sid, n in mention_counts.items():
        mention_kinds[sid.split(":")[0]] += n
    by_kind = Counter(v["kind"] for v in vocab.values())
    n_eligible = sum(eligible.values())
    summary = {
        "records": stats["records"], "eligible": n_eligible,
        "subjects_by_kind": dict(by_kind),
        "records_with_subject_by_kind": dict(kind_cover),
        "eligible_with_subject_by_kind": dict(kind_cover_elig),
        "share_eligible_with_any": round(kind_cover_elig["any"] / max(1, n_eligible), 3),
        "pgp_documentary": stats["pgp_documentary"],
        "pgp_documentary_with_any": stats["pgp_documentary_any"],
        "pgp_documentary_with_pgp_subject": stats["pgp_documentary_pgp_subject"],
        "share_pgp_documentary_with_any": round(stats["pgp_documentary_any"] / max(1, stats["pgp_documentary"]), 3),
        "share_pgp_documentary_with_pgp_subject":
            round(stats["pgp_documentary_pgp_subject"] / max(1, stats["pgp_documentary"]), 3),
        "pgp_documentary_eligible": stats["pgp_documentary_eligible"],
        "pgp_documentary_eligible_with_any": stats["pgp_documentary_eligible_any"],
        "pgp_records": stats["pgp_records"], "pgp_records_with_pgp_subject": stats["pgp_records_pgp_subject"],
        "raw_pgp_tags": len(tag_counts),
        "pgp_tag_decisions": dict(Counter(curate_tag(t)[2] + ":" + ("drop" if not curate_tag(t)[1] else "keep")
                                          for t in tag_counts)),
        "mean_subjects_per_covered_record": round(stats["subject_assignments"] / max(1, kind_cover["any"]), 2),
        "topic_v1_labels_missing_in_v2": dict(v1_missing),
        "sefaria": dict(sef_stats),
        "top_subjects": {kind: [(s, counts[s]) for s in sorted((x for x in keep if x.startswith(kind + ":")),
                                                               key=lambda x: -counts[x])[:15]]
                         for kind in ("topic", "work", "domain", "pgp", "sefaria")},
        "records_with_doc_type": stats["records_with_doc_type"], "doc_type_share": round(doc_type_share, 4),
        "records_with_description": stats["records_with_description"],
        "records_with_title_only_topic": stats["records_with_title_only_topic"],
        "records_with_description_festival": stats["records_with_description_festival"],
        "records_with_weak_description_festival": stats["records_with_weak_description_festival"],
        "records_with_canon_festival": stats["records_with_canon_festival"],
        "festival_carriers": {t: {"strong": counts[f"topic:{t}"] - weak_counts[f"topic:{t}"],
                                  "weak": weak_counts[f"topic:{t}"],
                                  **{r: reason_counts[(f"topic:{t}", r)] for r in WEAK_REASONS},
                                  "canon_listed": canon_counts[f"topic:{t}"],
                                  "gold_override": sum(gold_counts.values()) if t == "sukkot" else 0}
                              for t in FESTIVAL_TOPICS},
        "gold_override_reasons": dict(gold_counts),
        "canon_topics": {sid: {"carriers": counts[sid], "canon": canon_counts[sid]}
                         for sid in sorted(keep) if sid.startswith("topic:")},
        "canon_sefaria_assignments": sum(n for s, n in canon_counts.items() if s.startswith("sefaria:")),
        "weak_reason_assignments": dict(reason_totals),
        "vocab_names_dropped": {sid: names for sid, names in names_dropped.items() if names},
        "mentions": {
            "computed": index is not None,
            "assignments": stats["mention_assignments"],
            "records_with_mention": stats["records_with_mention"],
            "mean_per_record": round(stats["mention_assignments"] / max(1, stats["records"]), 3),
            "mean_per_eligible_record": round(stats["mention_assignments_eligible"] / max(1, n_eligible), 3),
            "by_kind": dict(mention_kinds),
            "records_by_n_mentions_capped_20": dict(sorted(mention_dist.items())),
            "top_subjects": mention_counts.most_common(30),
        },
        "legacy_festivals": legacy, "sefaria_canon": not legacy and not args.no_sefaria_canon,
        "weak_assignments": stats["weak_assignments"],
        "min_pgp": args.min_pgp, "min_records": args.min_records, "general_titles": not args.no_general_titles,
    }
    (out_dir / "subjects_v1.stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# self-test (no files, no model)
# ---------------------------------------------------------------------------------------------------------------
# (text, festival topic it must name); from review findings 2, 4, 7, 8, 12 and the plan (A2-A4, C1)
FESTIVAL_POSITIVES = [
    ("פיוטים לשבועות", "shavuot"), ("חג שבועות", "shavuot"), ("קרובות לשבועות", "shavuot"), ("Shavuʿot", "shavuot"),
    ("Simḥat Torah", "simhat_torah"), ("Simchat Torah", "simhat_torah"), ("Shemini ʿAṣeret", "simhat_torah"),
    ('קדושתא למוסף ר"ה', "rosh_hashanah"), ("לר״ה", "rosh_hashanah"), ("9th of Av", "fast_days"),
    ("Tish'ah be-Av", "fast_days"), ("Tisha Be'av", "fast_days"), ("Tisha B'Av", "fast_days"),
    ("ninth of Av", "fast_days"), ("ḥametz", "pesach"), ("Hametz", "pesach"), ("chametz", "pesach"),
    ("matzot", "pesach"), ("matzah", "pesach"), ("Passover Haggadah", "pesach"), ("ליל הסדר", "pesach"),
    ("יוה״כ", "yom_kippur"), ('יו"כ', "yom_kippur"), ("Neʿila", "yom_kippur"), ("Ne'ilah", "yom_kippur"),
    ("יום-הכפורים", "yom_kippur"), ("ליום הכפורים", "yom_kippur"), ("יום הכיפורים", "yom_kippur"),
    ("ימים נוראים עמידה", "yom_kippur"), ("ימים נוראים עמידה", "rosh_hashanah"), ("High Holy Days", "yom_kippur"),
    ("High Holy Days", "rosh_hashanah"), ("Rosh ha-Shanah", "rosh_hashanah"), ("Jewish New Year", "rosh_hashanah"),
    ("Sukot", "sukkot"), ("Hosha'na Rabba", "sukkot"), ("Hoshanot", "sukkot"), ("הושענות", "sukkot"),
    ("Musaf for Sukkot", "sukkot"), ("מוסף לסוכות", "sukkot"), ("לפורים", "purim"), ("ופורים", "purim"),
    ("פיוט, לפורים?", "purim"), ("Purim", "purim"), ("Hanukkah", "hanukkah"), ("Ḥanukkah", "hanukkah"),
    ("ראש חודש", "rosh_hodesh"), ("מוסף לשבת ראש חדש", "rosh_hodesh"), ("Rosh Ḥodesh", "rosh_hodesh"),
    ("Rosh Chodesh", "rosh_hodesh"), ("Shemini Atzeret", "simhat_torah"),
]
# (text, festival topic it must NOT name)
FESTIVAL_NEGATIVES = [
    ("יום הכפורים", "purim"), ("ליום הכפורים", "purim"), ("יום-הכפורים", "purim"), ("מחוסר כפורים", "purim"),
    ("Ethical poem, Ashre adam, matsa hokhmah", "pesach"), ("אלתנקיח פי תחריר פסח אלמסיח", "pesach"),
    ("Pesach al-Masih", "pesach"), ("Muslim New Year = 1 Muḥarram 611 AH", "rosh_hashanah"),
    ("Islamic new year", "rosh_hashanah"), ("Coptic New Year", "rosh_hashanah"),
    ('תפילת יום כיפור קטן לערב ר"ח', "yom_kippur"), ("Yom Kippur Katan", "yom_kippur"),
    ("Moshe b. Shoshan", "sukkot"), ("ibn Shoshan", "sukkot"), ("shoshanim", "sukkot"),
    ("Shemuel b. Hoshana", "sukkot"), ("שמואל בן הושענא", "sukkot"), ("שבועות", "shavuot"),
    ("Shevu'ot 21a", "shavuot"), ("according to the observation of the new moon", "rosh_hodesh"),
    ("ר\"ח", "rosh_hodesh"), ("son-in-law", "pesach"),
]
# (text, is a canon context) for is_canon_context; findings 3, 11 and the plan A3
CANON_CASES = [
    ("Rosh ha-Shanah 3b-6a", True), ("Sukkah 39a", True), ("Talmud Bavli, Pesahim 3a", True),
    ("ו' תשובות ... (סוכה לד ע\"א, מד ע\"א, ועוד)", True), ("Book of Esther", True), ("Esther 3:4-6", True),
    ("[Bible]: Esther 3:1", True), ("מגילת אסתר", True), ("מסכת סוכה", True), ("מס' סוכה דף ב", True),
    ("Shavuʿot 21a", True), ("Mishneh Torah, Hametz u-Matzah", True), ("פירוש לתלמוד, סוכה", True),
    ("Passover Haggadah. Liturgical additions Bible: Texts and Translations Liturgy and Brakhot; Piyyut", False),
    ("פיוט פורים פירוש אסתר ג: א-ה בערבית-יהודית. Bible: Texts; Biblical Exegesis; Piyyut", False),
    ("משנה טהרות Jewish prayers, contains: Purim and Passover. Mishnah: Texts Liturgy and Brakhot", False),
    ("Musaf for Sukkot. Aramaic Targumim", False), ("מוסף לסוכות. מקרא [טקסט]", False),
    ("Letter of Esther bat Yosef, after Passover", False), ("Ketubbah of Esther daughter of Moshe", False),
    ("ספרים תלמודיים לפסח", False), ("Sukkah 3 is mentioned", True), ("Piyyut for Sukkot", False),
]
# (text, canon by FESTIVAL_CANON_RE alone): names / anchors, as the training mix tests them
CANON_NAME_CASES = [("Esther", True), ("אסתר", True), ("מגילה", True), ("Megillah", True), ("Scroll of Esther", True),
                    ("Purim", False), ("Esther bat Yosef", False), ("פורים", False)]
LAW_CASES = [
    ("laws of Pesach", True), ("Laws of the New Year", True), ("הלכות פסח", True), ("hilkhot sukkah ve-lulav", True),
    ("Hilchot Chanukah", True), ("Mishnah tractate Sukkah chapter three", True), ("Sabbath laws", True),
    ("marriage law", True), ("דיני ממונות", True), ("מסכת סוכה", True), ("Dinei Tefillah", True),
    ("halakhic rulings on Purim", True), ("הל' שבת", True),
    ("Passover", False), ("פסח", False), ("son-in-law", False), ("lawsuit", False), ("בית דין", False),
    ("Purim", False), ("Shabbat", False), ("Taanit", False), ("Bill of divorce", False),
]


def _failures_of(cases: Iterable[Tuple[str, bool]], check, label: str) -> List[str]:
    """Run a boolean check over a case table.

    :param cases: (input, expected) pairs.
    :param check: Function input -> bool.
    :param label: Table name for the messages.
    :returns: Failure messages.
    :rtype: List[str]
    """
    return [f"{label}: {text!r} -> {not want}, expected {want}" for text, want in cases if bool(check(text)) != want]


def self_test() -> None:
    """Table-driven tests of the festival, canon, incidental, law-name, title, gold, vocab-name and mention rules.

    :raises AssertionError: Listing every failed case.
    """
    fails: List[str] = []
    fails += [f"festival: {s!r} misses {t}" for s, t in FESTIVAL_POSITIVES if t not in festival_mentions(s)]
    fails += [f"festival: {s!r} names {t}" for s, t in FESTIVAL_NEGATIVES if t in festival_mentions(s)]
    fails += _failures_of(CANON_CASES, is_canon_context, "canon")
    fails += _failures_of(CANON_NAME_CASES, FESTIVAL_CANON_RE.search, "canon name")
    fails += _failures_of(LAW_CASES, is_law_name, "law name")
    # topic_v2 (frames): the Purim / Yom Kippur prefix bug is fixed, v9's rules keep it for --legacy-festivals
    topic_cases = [(TOPIC_RE["purim"], "ליום הכפורים", False), (TOPIC_RE_LEGACY["purim"], "ליום הכפורים", True),
                   (TOPIC_RE["purim"], "לפורים", True), (TOPIC_RE["yom_kippur"], "יום-הכפורים", True),
                   (TOPIC_RE_LEGACY["yom_kippur"], "יום-הכפורים", False), (TOPIC_RE["yom_kippur"], "יום כיפור קטן", False)]
    fails += [f"topic_v2: {rx.pattern[:30]}... on {s!r} -> {not want}" for rx, s, want in topic_cases
              if bool(rx.search(s)) != want]

    # descriptions: (description, incidental type) -> (strong, {weak: reason})
    desc_cases = [
        ("Order of payment, 3 dirhams for jarid. Written on the eve of Sukkot.", False, set(), {"sukkot": "incidental"}),
        ("Colophon. Dated in the top left corner Purim 1419 of the Seleucid Era", False, set(), {"purim": "incidental"}),
        ("Passover Haggadah. Liturgical additions Bible: Texts and Translations Liturgy and Brakhot; Piyyut", False,
         {"pesach"}, {}),
        ("Talmud Bavli, Sukkah 29a-b", False, set(), {"sukkot": "canon"}),
        ("Letter. He will come after Passover.", True, set(), {"pesach": "incidental"}),
        ("Piyyut for Purim", False, {"purim"}, {}),
        ("תפילת יום כיפור קטן לערב ר\"ח. Liturgical additions", False, set(), {}),
        ("Ethical poem, Ashre adam, matsa hokhmah. Ethical Literature", False, set(), {}),
        ("פיוט פורים פירוש אסתר ג: א-ה. Bible: Texts; Biblical Exegesis; Piyyut", False, {"purim"}, {}),
        ("Recto: ʿAmida for the additional service on Rosh ha-Shana. Verso: an ownership note", False,
         {"rosh_hashanah"}, {}),
    ]
    # Document Type "bible" / "talmud": canon unless a liturgy / homily cue
    desc_cases_typed = [
        ("Passover Haggadah. Liturgical additions Bible: Texts and Translations", {"pesach"}, {}),
        ("Sermon composed for the week of Parashat Bereshit or Simhat Torah", {"simhat_torah"}, {}),
        ("הפטרה לשמיני של פסח, הפטרה יום א' דשבועות. . Bible: Texts", set(), {"pesach": "canon"}),
        ("שרח להפטרת תשעה באב (ירמיהו ח: 22-18). . Biblical Exegesis", set(), {"fast_days": "canon"}),
    ]
    for desc, want_strong, want_weak in desc_cases_typed:
        got = description_festival_reasons(desc, False, canon_type=True)
        if got != (want_strong, want_weak):
            fails.append(f"description (bible/talmud type): {desc!r} -> {got}, expected {(want_strong, want_weak)}")
    for desc, inc, want_strong, want_weak in desc_cases:
        got = description_festival_reasons(desc, inc)
        if got != (want_strong, want_weak):
            fails.append(f"description: {desc!r} -> {got}, expected {(want_strong, want_weak)}")
    if description_festivals("Talmud Bavli, Sukkah 29a-b", False) != (set(), {"sukkot"}):
        fails.append("description_festivals wrapper")
    incidental_cases = [("", {"Paraliterary text"}, "Colophon of a book, dated Purim 1419", True),
                        ("", {"Paraliterary text"}, "From a volume of piyyuṭim for Yom Kippur. Title page", False),
                        ("letter", set(), "", True), ("", {"Letter"}, "", True), ("bible", set(), "", False),
                        ("liturgical", {"Literary text"}, "", False)]
    fails += [f"incidental_record: {c}" for c in incidental_cases if incidental_record(*c[:3]) != c[3]]

    # frames + titles + gold -> typed labels
    def labels_of(frames: List[List[str]], general: Optional[List[str]] = None, desc: str = "", inc: bool = False,
                  gold: bool = False, legacy: bool = False) -> TopicLabels:
        """Labels of a synthetic record.

        :param frames: Frames per scholarly entry.
        :param general: KTIV general titles.
        :param desc: Description.
        :param inc: Incidental type.
        :param gold: Sukkot gold record.
        :param legacy: v9 labels.
        :returns: Labels.
        :rtype: TopicLabels
        """
        topics, canon, _ = frame_subjects(frames, TOPIC_RE_LEGACY if legacy else TOPIC_RE)
        return topic_labels(topics, canon, general or [], desc, inc, gold, legacy=legacy)

    label_cases = [
        ("Esther frame", labels_of([["[Bible]: Esther 3:1-5"]]), set(), {"purim": "canon"}, {"purim"}, False),
        ("Ketubbot tractate", labels_of([["[Talmud Bavli]: Ketubbot 3 a"]]), {"marriage"}, {}, {"marriage"}, False),
        ("tractate + hoshanot", labels_of([["[Talmud Bavli]: Sukkah 29 a"], ["[Piyyut]: (Hoshanot) ''x''"]]),
         {"sukkot"}, {}, set(), False),
        ("ימים נוראים prayers", labels_of([["[תפילות קבע]: ימים נוראים עמידה"]]), {"yom_kippur", "rosh_hashanah"}, {},
         set(), False),
        ("Qiddush ha-Hodesh", labels_of([["[Moses b. Maimon, Rambam, Mishneh Torah]: Qiddush ha-Hodesh 1:1"]]),
         set(), {"rosh_hodesh": "canon"}, {"rosh_hodesh"}, False),
        ("gold canon-only", labels_of([["[Talmud Bavli Commentaries]: Sukkah"]], gold=True), {"sukkot"}, {}, set(),
         True),
        ("gold, no evidence", labels_of([], gold=True), {"sukkot"}, {}, set(), False),
        ("title piyyut", labels_of([], ["פיוטים לפורים"]), set(), {"purim": "title"}, set(), False),
        ("title law", labels_of([], ["הלכות מגילה"]), set(), {"purim": "canon"}, {"purim"}, False),
        ("title tractate", labels_of([], ["Talmud Bavli Sukkah 29a"]), set(), {"sukkot": "canon"}, {"sukkot"}, False),
        ("canon beats title", labels_of([["[Bible]: Esther 3:1"]], ["פיוטים לפורים"]), set(), {"purim": "canon"},
         {"purim"}, False),
        ("incidental letter", labels_of([], desc="Letter: after Passover", inc=True), set(),
         {"pesach": "incidental"}, set(), False),
        ("description beats frame canon", labels_of([["[Bible]: Esther"]], desc="Piyyut for Purim"), {"purim"}, {},
         set(), False),
        ("legacy Esther", labels_of([["[Bible]: Esther 3:1"]], legacy=True), {"purim"}, {}, set(), False),
        ("legacy YK as Purim", labels_of([["[Piyyut]: ליום הכפורים"]], legacy=True), {"purim", "yom_kippur"}, {},
         set(), False),
        ("new YK", labels_of([["[Piyyut]: ליום הכפורים"]]), {"yom_kippur"}, {}, set(), False),
    ]
    for name, got, strong, weak_reason, canon, gold in label_cases:
        if (got.strong, got.weak_reason, got.canon, got.gold_override) != (strong, weak_reason, canon, gold):
            fails.append(f"labels {name}: {got.strong, got.weak_reason, got.canon, got.gold_override}, expected "
                         f"{strong, weak_reason, canon, gold}")

    # vocabulary name filter (C1)
    name_cases = [("topic:shabbat", "Mishnah Shabbat", False), ("topic:purim", "Scroll of Esther", False),
                  ("topic:purim", "מגילת אסתר", False), ("topic:purim", "Megillah", False), ("topic:purim", "Purim", True),
                  ("sefaria:shabbat", "Mishnah Shabbat", False), ("sefaria:tisha-bav", "Talmud Taanit", False),
                  ("sefaria:laws-of-shabbat", "Mishnah Shabbat", True), ("topic:niddah", "Tractate Niddah", True),
                  ("work:talmud-bavli-sukkah", "Talmud Bavli Sukkah", True), ("topic:sukkot", "Sukkah", True)]
    fails += [f"vocab name: {sid} {n!r} -> {not want}" for sid, n, want in name_cases if vocab_name_kept(sid, n) != want]

    # mentions (C3)
    vocab = {"topic:shabbat": {"names_en": ["Shabbat", "Sabbath laws"], "names_he": ["שבת", "שַׁבָּת"]},
             "pgp:fustat": {"names_en": ["Fustat"], "names_he": ["פסטאט"]},
             "topic:mourning": {"names_en": [], "names_he": ["אבל", "אבלות"]},
             "pgp:divorce": {"names_en": ["bill of divorce", "Get"], "names_he": ["גט"]},
             "work:x": {"names_en": ["#hidden", "ab"], "names_he": []}}
    index = build_mention_index(vocab)
    mention_cases = [
        ("Letter written in Fustat on Shabbat", {"pgp:fustat", "topic:shabbat"}),
        ("ובשבת הגדול", {"topic:shabbat"}), ("אבל אין", set()), ("באבלות", {"topic:mourning"}),
        ("a bill of divorce", {"pgp:divorce"}), ("the get was written", set()), ("Sabbath", set()),
        ("Sabbath laws", {"topic:shabbat"}), ("Shabbatai b. Moshe", set()), ("Piyyut for Hanukkah", {"topic:hanukkah"}),
        ("ab cd", set()),
    ]
    for text, want in mention_cases:
        got = named_subjects(text, index)
        if got != want:
            fails.append(f"mentions: {text!r} -> {sorted(got)}, expected {sorted(want)}")

    # fields come from the folded semantic_v3 text, text_v3 before text
    sem = {"text": "Description: v4 text", "text_v3": "Description: first line\nsecond line\nLanguage: Hebrew\n"
                                                      "Document Type: letter\nAlternate titles: A; B"}
    want_fields = {"description": "first line second line", "alt_titles": "A; B", "doc_type": "letter"}
    if source_fields(sem) != want_fields:
        fails.append(f"source_fields: {source_fields(sem)}")
    if source_fields({"text": "Description: only v3"})["description"] != "only v3":
        fails.append("source_fields falls back to text")
    if "Scholarly identification:" in FIELD_PREFIXES and source_fields(
            {"text": "Description: d\nScholarly identification: [Talmud Bavli]: Sukkah 2 a"})["description"] != "d":
        fails.append("source_fields folds the Scholarly identification line into the description")
    if fails:
        raise AssertionError(f"{len(fails)} self-test failures:\n" + "\n".join(fails))
    n = (len(FESTIVAL_POSITIVES) + len(FESTIVAL_NEGATIVES) + len(CANON_CASES) + len(CANON_NAME_CASES) + len(LAW_CASES)
         + len(topic_cases) + len(desc_cases) + len(desc_cases_typed) + len(incidental_cases) + len(label_cases)
         + len(name_cases) + len(mention_cases) + 4)
    print(f"build_subjects self-test: {n} cases passed")


if __name__ == "__main__":
    main()
