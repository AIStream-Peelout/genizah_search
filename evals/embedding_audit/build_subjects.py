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

Outputs (``AUDIT_ROOT/v3``): ``subjects_v1.jsonl`` ({doc_id, subjects, weak}), ``subject_vocab.json``
({id: {kind, names_en, names_he, n_records, n_eligible, source, ...}}), ``subjects_v1.stats.json``; and
``results/pgp_tag_curation.csv`` in the audit code dir.

Run after ``semantic_text.py`` (json only, no model)::

    python3 build_subjects.py
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from build_train_pairs import TOPIC_PHRASES
from embed_utils import AUDIT_ROOT

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
TOPIC_RULES_V2: Dict[str, str] = {
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
# Ambiguous outside liturgy ("Shevu'ot" / "שבועות" is also the tractate on oaths): applied to liturgical frames only,
# never to general titles.
TOPIC_RULES_LITURGY: Dict[str, str] = {
    "shavuot": rf"Shevu[`']ot|{HB}שבועות{HE}",
}
TOPIC_RE = {t: re.compile(p) for t, p in TOPIC_RULES_V2.items()}
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
# work normalisation
# ---------------------------------------------------------------------------------------------------------------
LITURGY_WORKS = {"Piyyut", "Common Prayers", "Liturgical additions", "Brakhot", "Occasional prayer"}
TRACTATE_WORKS = {"Talmud Bavli", "Talmud Yerushalmi", "Mishnah", "Tosefta", "Hilkhot ha-Rif",
                  "Talmud Bavli Commentaries", "Rabbenu Hananel", "Mishnah Commentaries"}
BOOK_WORKS = {"Bible", "Targum Onqelos", "Aramaic Targumim", "Arabic Tafsir", "Tafsir Saadya Gaon", "Haftarot",
              "Biblical Exegesis", "Biblical Exegesis - Rabbanite", "Biblical Exegesis - Karaite", "Rashi on the Bible",
              "Derashot", "Massorah", "Yefet b. Eli, Bible Commentary"}
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


def frame_subjects(entries: List[List[str]]) -> Tuple[Set[str], Dict[str, Tuple[str, Optional[str]]]]:
    """Topics (topic_v2) and work subjects of one record's frames.

    :param entries: Frames grouped per scholarly entry (:func:`ktiv_fields`).
    :returns: (topic names, {work id: (english label, hebrew label or None)}).
    :rtype: Tuple[Set[str], Dict[str, Tuple[str, Optional[str]]]]
    """
    topics: Set[str] = set()
    works: Dict[str, Tuple[str, Optional[str]]] = {}
    for entry in entries:
        for raw_work, body in parse_frames(entry):
            work = canonical_work(raw_work) if raw_work else ""
            text = f"[{raw_work}]: {body}" if raw_work else body
            topics.update(t for t, rx in TOPIC_RE.items() if rx.search(text))
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
    return topics, works


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
# main
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Build subjects_v1.jsonl, subject_vocab.json and the PGP tag curation sheet; print coverage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semantic", default=str(V3 / "corpus_semantic.jsonl"))
    parser.add_argument("--merged", default=str(MERGED))
    parser.add_argument("--frame-refs", default=str(AUDIT_ROOT / "frame_refs.jsonl"))
    parser.add_argument("--curated-refs", default=str(AUDIT_ROOT / "sefaria/derived/curated_refs.jsonl"))
    parser.add_argument("--concept-graph", default=str(AUDIT_ROOT / "sefaria/derived/concept_graph.json"))
    parser.add_argument("--out-dir", default=str(V3))
    parser.add_argument("--curation-csv", default=str(RESULTS / "pgp_tag_curation.csv"))
    parser.add_argument("--min-pgp", type=int, default=20, help="records a curated PGP subject needs")
    parser.add_argument("--min-records", type=int, default=5, help="records a topic/work/domain/sefaria subject needs")
    parser.add_argument("--no-general-titles", action="store_true",
                        help="match topic_v2 rules against frames only, not KTIV general titles (free-text identifications)")
    args = parser.parse_args()

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
    weak: Dict[str, Set[str]] = {}  # subject ids whose only evidence is a KTIV general title (free text)
    stats: Counter = Counter()
    for sem, record in zip(iter_jsonl(Path(args.semantic)), iter_jsonl(Path(args.merged))):
        doc_id = sem["doc_id"]
        if record["canonical_id"] != doc_id:
            raise ValueError(f"corpus/merged order mismatch at {doc_id} vs {record['canonical_id']}")
        eligible[doc_id] = bool(sem["eligible"])
        entries, raw_domains, general = ktiv_fields(record)
        topics, works = frame_subjects(entries)
        if sem.get("sukkot_gold"):
            topics.add("sukkot")
        title_only: Set[str] = set()
        if not args.no_general_titles:
            title_only = {t for title in general for t, rx in TOPIC_RE.items() if rx.search(title)} - topics
            topics |= title_only
            stats["records_with_title_only_topic"] += bool(title_only)
            weak[doc_id] = {f"topic:{t}" for t in title_only}
        for t in set(sem.get("topics") or []) - topics:
            v1_missing[t] += 1
        domains = domain_subjects(raw_domains)
        tags, types = pgp_info(record)
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
    for sid in sorted(keep, key=lambda s: (s.split(":")[0], -counts[s], s)):
        kind, slug = sid.split(":", 1)
        entry = {"kind": kind, "n_records": counts[sid], "n_eligible": elig_counts[sid]}
        if kind == "topic":
            phrases = TOPIC_PHRASES.get(slug, []) + TOPIC_EXTRA_PHRASES.get(slug, [])
            sen, she = sefaria_names(TOPIC_SEFARIA.get(slug, []), graph)
            entry.update(names_en=dedupe([p for p in phrases if not HEB.search(p)] + sen),
                         names_he=dedupe([p for p in phrases if HEB.search(p)] + she),
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
            entry.update(names_en=sen, names_he=she, source="Sefaria topic via frame_refs x curated_refs")
        vocab[sid] = entry

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind_cover: Counter = Counter()
    kind_cover_elig: Counter = Counter()
    with open(out_dir / "subjects_v1.jsonl", "w", encoding="utf-8") as fh:
        for doc_id, kinds in per_doc.items():
            subjects = sorted(s for ids in kinds.values() for s in ids if s in keep)
            weak_ids = sorted(weak.get(doc_id, set()) & set(subjects))
            stats["weak_assignments"] += len(weak_ids)
            fh.write(json.dumps({"doc_id": doc_id, "subjects": subjects, "weak": weak_ids}, ensure_ascii=False) + "\n")
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
        "records_with_title_only_topic": stats["records_with_title_only_topic"],
        "weak_assignments": stats["weak_assignments"],
        "min_pgp": args.min_pgp, "min_records": args.min_records, "general_titles": not args.no_general_titles,
    }
    (out_dir / "subjects_v1.stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
