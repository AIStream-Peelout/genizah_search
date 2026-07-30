"""Domain terminology for Jewish holidays, liturgy, and Genizah document genres.

Scholarship on the Genizah transliterates Hebrew inconsistently (qinot /
kinnot / kinot), cites holidays by several names (Tisha B'Av / Ninth of Av /
ט' באב), and mixes Hebrew script with Latin. Retrieval and relevance checks
that compare a user's spelling against the corpus's spelling therefore miss
material that is genuinely present. Each entry below groups surface forms
that refer to the same concept.

Keep groups conservative: only forms that are genuinely interchangeable in
scholarly usage belong together. Over-broad grouping makes a topic look
covered when it is not.
"""

from typing import Dict, List, Set

# Each group: canonical concept -> equivalent surface forms (lowercased).
CONCEPT_ALIASES: Dict[str, List[str]] = {
    "tisha_bav": [
        "tisha b'av", "tisha bav", "tish b'av", "tishah b'av", "ninth of av",
        "9 av", "9th of av", "fast of av", "ט' באב", "תשעה באב",
    ],
    "qinot": [
        "qinot", "kinot", "kinnot", "qinnot", "qinah", "kinah", "qina", "kina",
        "lamentation", "lamentations", "eikhah", "ekhah", "eichah", "איכה", "קינות", "קינה",
    ],
    "tu_bav": [
        "tu b'av", "tu bav", "tu be-av", "tu be'av", "fifteenth of av", "15 av",
        "15th of av", "ט\"ו באב", "טו באב",
    ],
    "yom_kippur": [
        "yom kippur", "yom ha-kippurim", "day of atonement", "kippurim",
        "יום כיפור", "יום הכיפורים",
    ],
    "kol_nidre": ["kol nidre", "kol nidrei", "kol-nidré", "kol-nidre", "כל נדרי"],
    "rosh_hashanah": [
        "rosh hashanah", "rosh ha-shanah", "rosh-ha-shanah", "new year",
        "ראש השנה",
    ],
    "piyyut": [
        "piyyut", "piyyutim", "piyut", "piyyutic", "liturgical poem",
        "liturgical poetry", "payyetan", "paytan", "פיוט", "פיוטים",
    ],
    "passover": [
        "passover", "pesach", "pesah", "haggadah", "haggadot", "seder",
        "פסח", "הגדה",
    ],
    "shavuot": [
        "shavuot", "shavuoth", "shabuot", "pentecost", "weeks", "שבועות",
    ],
    "purim": ["purim", "esther", "megillah", "megillat esther", "פורים", "אסתר"],
    "sukkot": ["sukkot", "sukkoth", "tabernacles", "hoshana", "סוכות"],
    "hanukkah": ["hanukkah", "chanukah", "hanukah", "חנוכה"],
    "shabbat": ["shabbat", "sabbath", "shabbath", "שבת"],
    "zemirot": [
        "zemirot", "zemirah", "zemiroth", "table songs", "table hymns", "זמירות",
    ],
    "birkat_hamazon": [
        "birkat ha-mazon", "birkat hamazon", "grace after meals", "bentching",
        "benching", "ברכת המזון",
    ],
    "ketubba": [
        "ketubba", "ketubbah", "ketubah", "ketubbot", "ketubot", "kettubba",
        "marriage contract", "marriage contracts", "כתובה", "כתובות",
    ],
    "get_divorce": ["get", "gittin", "divorce deed", "bill of divorce", "גט", "גיטין"],
    "responsa": ["responsa", "responsum", "teshuvot", "she'elot u-teshuvot", "תשובות"],
    "targum": ["targum", "targumim", "targumic", "aramaic translation", "תרגום"],
    "masorah": [
        "masorah", "masora", "masoretic", "massorah", "vocalization",
        "tiberian", "מסורה",
    ],
}

# Reverse index: surface form -> concept key.
_FORM_TO_CONCEPT: Dict[str, str] = {
    form: concept for concept, forms in CONCEPT_ALIASES.items() for form in forms
}


def find_concepts(text: str) -> Set[str]:
    """Identify domain concepts named anywhere in a text.

    :param text: Lowercased, whitespace-normalized text (query or document).
    :returns: Concept keys whose surface forms occur in the text.
    :rtype: Set[str]
    """
    found: Set[str] = set()
    for form, concept in _FORM_TO_CONCEPT.items():
        if concept in found:
            continue
        if form in text:
            found.add(concept)
    return found


def expand_query_aliases(query: str, max_forms: int = 24) -> List[str]:
    """Return corpus spellings to search alongside the user's wording.

    Only forms the user did not already write are returned, so an exactly
    phrased query is not diluted by re-searching its own terms.

    :param query: Raw user query.
    :param max_forms: Upper bound on returned forms, bounding query size.
    :returns: Additional surface forms worth searching, longest first.
    :rtype: List[str]
    """
    normalized = " ".join(query.lower().split())
    forms: List[str] = []
    for concept in find_concepts(normalized):
        for form in CONCEPT_ALIASES[concept]:
            if form not in normalized and form not in forms:
                forms.append(form)
    # Longer, more specific phrases first when the budget is tight.
    forms.sort(key=len, reverse=True)
    return forms[:max_forms]


def aliases_for_concept(concept: str) -> List[str]:
    """Return the surface forms of a concept.

    :param concept: Concept key from :data:`CONCEPT_ALIASES`.
    :returns: Equivalent surface forms, or an empty list when unknown.
    :rtype: List[str]
    """
    return list(CONCEPT_ALIASES.get(concept, []))
