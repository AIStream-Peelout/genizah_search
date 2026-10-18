"""The scholar name standard: one display form and one identity key for every Scholar name.

Tooling for the knowledge-graph consolidation (``scripts/kg/consolidate_scholars.py``), not
backend code. Pure functions, no I/O.

Latin-script names (``standard_name``):

1. compose to Unicode NFC (a stored "h" + U+0323 becomes "ḥ"), strip, collapse whitespace,
   drop trailing punctuation;
2. given names first, surname last: ``"Hathaway, Jane"`` -> ``"Jane Hathaway"``; FJP style
   ``"Diez Merino L"`` (surname tokens followed only by initials) -> ``"L. Diez Merino"``.
   Names with several commas are left unchanged (and reported by ``needs_review``);
3. a standalone capital Latin letter, with or without a period, is an initial written in
   ``INITIAL_STYLE``: ``"S.D."`` -> ``"S. D."``, ``"J"`` -> ``"J."``; initials glued to the
   next word are split off (``"A.M.Haberman"`` -> ``"A. M. Haberman"``). Lower-case single
   letters are particles, not initials (``"Salmon b. Yeruḥam"`` keeps its ``b.``);
4. letters, diacritics, apostrophes, hyphens and capitalisation are kept as written (no title-casing:
   an all-caps name goes to review), except that in the given-first form the particles of
   ``LOWER_PARTICLES``/``LOWER_PREFIXES`` after the first token are written in lower case
   (``"Michel De Certeau"`` -> ``"Michel de Certeau"``).

Hebrew-script names lose their nikud and cantillation marks and nothing else changes.
``identity_key`` decides exact duplicates (word spaces count: ``"M. Ben-Sasson"`` and
``"M. Ben Sasson"`` differ; a Hebrew geresh between two letters of a word is a letter mark and
counts: ``ריצ'לר`` is not ``ריצלר``; ``near_key`` finds both kinds of pair for the review sheet);
``choose_display`` picks one display form for a cluster among its members' standard forms: the
form most likely to appear (the most edges), then the tie-breaks (Hebrew initials: geresh/gershayim
``ש' אסף`` over a period ``ש. אסף`` over bare ``ש אסף``; it never writes a form no member has, so
``ח. שירמן`` + ``ח שירמן`` stays ``ח. שירמן``);
``needs_review`` names what keeps a name out of automatic renaming and merging: only the easy,
unambiguous cases are automatic.
"""

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Set, Tuple

#: How initials are written: ``"dotted"`` ("S. D. Goitein") or ``"bare"`` ("S D Goitein").
INITIAL_STYLE = "dotted"

#: Lower-case name particles: no FJP surname ends in one ("Isaac ben E"), and they are not phrase words.
PARTICLES = frozenset({
    "al", "el", "ibn", "bin", "bint", "ben", "b", "bat", "bar", "ha",
    "de", "del", "della", "di", "da", "das", "dos", "du", "des",
    "van", "von", "der", "den", "ten", "ter", "zu",
})
#: Particles written in lower case when they are not the first token of the given-first form ("Michel de
#: Certeau", "N. R. M. de Lange", "Wout van Bekkum"), the form those names most often appear in; never the
#: last token (a surname: "Du", "De"). Kept as written: Ben-/Bar- surnames, ben, b., ibn, and di, della, le, la
#: and el-, which their bearers usually capitalise (Riccardo Di Segni, Alexander A. Di Lella, Charles Le Quesne,
#: Nadia Maria El-Cheikh; "Tamer el-Leithy" wins its cluster by frequency).
LOWER_PARTICLES = frozenset({"de", "den", "der", "van", "von", "del", "du"})
#: Hyphenated prefixes written in lower case when they are not the first token ("Abu Bakr al-Razi").
LOWER_PREFIXES = ("al-",)
#: Lower-case words a name string may contain without being a phrase: the particles, "of" ("Judah b.
#: Barzilai of Barcelona"), Romance articles and conjunctions, and the author joiners (their own reason).
NAME_WORDS = PARTICLES | {"of", "la", "le", "y", "i", "e", "and", "et", "und"}

#: Trailing characters dropped from a Latin name (a final initial gets its period back in step 3).
TRAILING_PUNCTUATION = ".,;:!*/|-–—"

#: Hebrew nikud and cantillation live here; only the combining marks (category Mn) are removed.
HEBREW_MARKS = (0x0591, 0x05C7)

LATIN_TITLES = ("prof", "professor", "professors", "dr", "mr", "mrs", "ms", "mme",
                "rev", "rabbi", "rabbis", "sir")
HEBREW_TITLES = ('ד"ר', "ד״ר", "דר'", "דר׳", "דר.", "פרופ'", "פרופ׳", "פרופ.", "פרופסור", "הרב", "ר'", "ר׳",
                 "מר", "גב'", "גב׳", "הגאון")
HEBREW_HONORIFIC_SUFFIXES = ('ז"ל', "ז״ל", 'זצ"ל', "זצ״ל", 'הי"ו', "הי״ו")

#: Hebrew letters (finals included), geresh-like marks, and the marks that write a Hebrew initial.
HEBREW_LETTER = "[\u05d0-\u05ea]"
HEBREW_GERESH = "\u05f3'\u2019"  # ׳ (geresh), apostrophe, right single quote
HEBREW_INITIAL_MARKS = HEBREW_GERESH + '\u05f4"'  # plus ״ (gershayim) and "
#: A geresh between two Hebrew letters (צ' in ריצ'לר): a letter mark the identity key keeps, as KEY_GERESH.
INNER_GERESH = re.compile(f"(?<={HEBREW_LETTER})[{HEBREW_GERESH}](?={HEBREW_LETTER})")
KEY_GERESH = "\u05f3"  # ׳ (geresh)
#: A token of Hebrew initials: "ש", "ש'", "ש.", 'ד"צ', "ד.צ." (each letter but the last needs a mark).
HEBREW_INITIALS = re.compile(f"(?:{HEBREW_LETTER}[{HEBREW_INITIAL_MARKS}.]+)*{HEBREW_LETTER}[{HEBREW_INITIAL_MARKS}.]*")
#: One Hebrew initial ("ש", "ש'", "ש."), and one with the conjunction ו glued on ("ומ'": "and M.").
HEBREW_ONE_INITIAL = re.compile(f"{HEBREW_LETTER}[{HEBREW_INITIAL_MARKS}.]*")
HEBREW_AND_INITIAL = re.compile(f"ו{HEBREW_LETTER}[{HEBREW_INITIAL_MARKS}.]+")

#: Arabic particles and kunyas: a "Surname, Given" name containing one is not a modern surname flip.
ARABIC_PARTICLES = frozenset({"ibn", "bin", "bint", "abu", "abū", "abd"})
#: ʿayn/hamza-like marks stripped before matching a particle ("'Abd", "ʿAbd", "‘Abū").
AYN_MARKS = "'‘’ʼʻʿʾ`"
#: Words that join several authors in one name string.
AUTHOR_JOINERS = (" and ", " et ", " und ")


def collapse(text: str) -> str:
    """Strip the text and collapse every run of whitespace to one space.

    :param text: Any text.
    :returns: The collapsed text.
    :rtype: str
    """
    return " ".join(text.split())


def is_hebrew(text: str) -> bool:
    """Return True when the text contains a Hebrew letter.

    :param text: A name.
    :returns: Whether any character is a Hebrew letter.
    :rtype: bool
    """
    return any(unicodedata.name(c, "").startswith("HEBREW LETTER") for c in text)


def strip_hebrew_marks(text: str) -> str:
    """Remove nikud and cantillation (combining marks U+0591-U+05C7), keep maqaf and geresh.

    :param text: A Hebrew-script name.
    :returns: The name without vowel points or accents.
    :rtype: str
    """
    low, high = HEBREW_MARKS
    return "".join(c for c in text if not (low <= ord(c) <= high and unicodedata.category(c) == "Mn"))


def strip_accents(text: str) -> str:
    """NFKD-decompose the text and drop every combining mark.

    :param text: Any text.
    :returns: The text without diacritics.
    :rtype: str
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.category(c).startswith("M"))


def is_latin_letter(char: str) -> bool:
    """Return True for one Latin-script letter, accented or not.

    :param char: One character.
    :returns: Whether it is a Latin letter.
    :rtype: bool
    """
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN")


def format_initial(letter: str) -> str:
    """Write one initial in the house style.

    :param letter: A single Latin letter.
    :returns: ``"J."`` (dotted) or ``"J"`` (bare).
    :rtype: str
    """
    return letter.upper() + ("." if INITIAL_STYLE == "dotted" else "")


def split_initials(token: str) -> Optional[List[str]]:
    """Split a token made only of initials ("J", "J.", "S.D.", "S.D") into formatted initials.

    :param token: One whitespace-separated token.
    :returns: The formatted initials, or None when the token is not only initials.
    :rtype: Optional[List[str]]
    """
    parts = token.split(".")
    if len(parts) > 1 and parts[-1] == "":
        parts = parts[:-1]
    if all(len(p) == 1 and is_latin_letter(p) and p.isupper() for p in parts):
        return [format_initial(p) for p in parts]
    return None


def unglue_initials(token: str) -> List[str]:
    """Split initials glued to the following word off it ("A.M.Haberman" -> "A.", "M.", "Haberman").

    Only capital Latin letters each followed by a period count as initials, and the rest must
    be a capitalised word of two or more characters; any other token comes back unchanged.

    :param token: One whitespace-separated token.
    :returns: The initials (with their periods) and the word, or ``[token]``.
    :rtype: List[str]
    """
    *initials, rest = token.split(".")
    if (initials and all(len(p) == 1 and is_latin_letter(p) and p.isupper() for p in initials)
            and len(rest) > 1 and is_latin_letter(rest[0]) and rest[0].isupper()):
        return [f"{p}." for p in initials] + [rest]
    return [token]


def flip_comma(text: str) -> str:
    """Turn "Surname, Given" into "Given Surname" (exactly one comma).

    :param text: A name with one comma.
    :returns: The given-first form; an empty side just drops the comma.
    :rtype: str
    """
    surname, given = (part.strip() for part in text.split(","))
    return collapse(f"{given} {surname}")


def is_all_caps(text: str) -> bool:
    """Return True when the text has cased letters and all of them are upper case ("E. ROMERO", "J. M.").

    :param text: A name.
    :returns: Whether every cased letter is upper case.
    :rtype: bool
    """
    cased = [c for c in text if c.lower() != c.upper()]
    return bool(cased) and all(c.isupper() for c in cased)


def lower_particles(words: List[str]) -> List[str]:
    """Write the particles after the first word in lower case ("A. De Biberstein Kazimirski" -> "A. de ...").

    A prefix glued to an all-caps word stays as written ("John AL-AZMEH", not "John al-AZMEH": a
    form that appears nowhere).

    :param words: The words of a given-first Latin name.
    :returns: The words with ``LOWER_PARTICLES`` (except the last word) and ``LOWER_PREFIXES`` lowered.
    :rtype: List[str]
    """
    out = words[:1]
    for index, word in enumerate(words[1:], 1):
        prefix = next((p for p in LOWER_PREFIXES if word.casefold().startswith(p) and len(word) > len(p)
                       and not is_all_caps(word[len(p):])), None)
        if word.casefold() in LOWER_PARTICLES and index < len(words) - 1:
            word = word.lower()
        elif prefix:
            word = prefix + word[len(prefix):]
        out.append(word)
    return out


def standard_tokens(text: str) -> List[Tuple[str, bool]]:
    """Tokenise a name and rewrite its initials in the house style.

    :param text: A Latin-script name.
    :returns: ``(token, is_initial)`` pairs.
    :rtype: List[Tuple[str, bool]]
    """
    tokens: List[Tuple[str, bool]] = []
    for token in text.split():
        initials = split_initials(token)
        if initials:
            tokens.extend((initial, True) for initial in initials)
        else:
            tokens.append((token, False))
    return tokens


def trailing_initials(tokens: List[Tuple[str, bool]]) -> int:
    """Count the FJP-style trailing initials ("Diez Merino L") that belong in front of the surname.

    A lead ending in a particle ("Isaac ben E") is not a surname: nothing moves.

    :param tokens: ``(token, is_initial)`` pairs of a name without a comma.
    :returns: How many trailing initials move to the front (0 when it is not surname tokens + initials).
    :rtype: int
    """
    count = 0
    while count < len(tokens) and tokens[len(tokens) - 1 - count][1]:
        count += 1
    lead = tokens[: len(tokens) - count]
    if (count and lead and not any(is_initial for _, is_initial in lead)
            and lead[-1][0].rstrip(".").casefold() not in PARTICLES):
        return count
    return 0


def initials_first(tokens: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
    """Move FJP-style trailing initials ("Diez Merino L") in front of the surname.

    :param tokens: ``(token, is_initial)`` pairs of a name without a comma.
    :returns: The reordered pairs, or the input when it is not surname tokens + initials.
    :rtype: List[Tuple[str, bool]]
    """
    count = trailing_initials(tokens)
    return tokens[-count:] + tokens[:-count] if count else tokens


def latin_text(raw: str) -> str:
    """NFC-compose, collapse whitespace and drop trailing punctuation (step 1 of the standard).

    :param raw: A Latin-script name as stored.
    :returns: The cleaned text, still in its stored order.
    :rtype: str
    """
    return collapse(unicodedata.normalize("NFC", raw)).rstrip(TRAILING_PUNCTUATION).rstrip()


def latin_tokens(text: str) -> List[Tuple[str, bool]]:
    """Flip a one-comma name, split glued initials and tokenise.

    :param text: Output of ``latin_text`` with at most one comma.
    :returns: ``(token, is_initial)`` pairs, before any FJP reordering.
    :rtype: List[Tuple[str, bool]]
    """
    if "," in text:
        text = flip_comma(text)
    text = " ".join(part for token in text.split() for part in unglue_initials(token))
    return standard_tokens(text)


def standard_name(raw: str) -> str:
    """Return the display form of a Scholar name under the standard.

    :param raw: The name as stored in the graph.
    :returns: The standard display form (see the module docstring).
    :rtype: str
    """
    text = collapse(unicodedata.normalize("NFC", raw))
    if is_hebrew(text):
        return collapse(strip_hebrew_marks(text))
    text = latin_text(text)
    if text.count(",") > 1:
        return text
    tokens = latin_tokens(text)
    if "," not in text:
        tokens = initials_first(tokens)
    words = [token for token, _ in tokens]
    return " ".join(words if is_all_caps(text) else lower_particles(words))


def is_punctuation(char: str) -> bool:
    """Return True for punctuation, symbols and modifier letters (ʿ ʾ ʼ), deleted by the key.

    :param char: One character.
    :returns: Whether the identity key deletes it.
    :rtype: bool
    """
    category = unicodedata.category(char)
    return category[0] in ("P", "S") or category == "Lm"


def identity_key(raw: str) -> str:
    """Return the key that decides exact duplicates (equal keys = same scholar).

    ``standard_name``, then NFKD minus combining marks, casefold, delete punctuation and
    apostrophe-like marks without inserting a space (``Ben-Zion`` = ``Benzion``) and ``?``,
    collapse whitespace. A Hebrew geresh between two letters of a word is kept (as ``׳``):
    ``צ'`` and ``צ`` are different letters, so ``ריצ'לר`` and ``ריצלר`` differ, while the marks of
    an initial (``ש' אסף`` = ``ש. אסף``, ``ד"צ`` = ``ד.צ.``) are deleted.

    :param raw: The name as stored in the graph.
    :returns: The identity key (empty for a name with no letters or digits).
    :rtype: str
    """
    text = strip_accents(standard_name(raw)).casefold()
    parts = INNER_GERESH.split(text)
    return collapse(KEY_GERESH.join("".join(c for c in part if not is_punctuation(c)) for part in parts))


def spaceless_key(raw: str) -> str:
    """Return ``identity_key`` with its word spaces removed.

    Names with different identity keys but the same spaceless key differ only in spacing
    (``"M. Ben-Sasson"`` / ``"M. Ben Sasson"``, ``"מ. ב. לרנר"`` / ``'מ"ב לרנר'``). They are
    not exact duplicates under the standard; the planner lists them for review.

    :param raw: The name as stored in the graph.
    :returns: The identity key without spaces (empty for a name with no letters or digits).
    :rtype: str
    """
    return identity_key(raw).replace(" ", "")


def near_key(raw: str) -> str:
    """Return ``spaceless_key`` without the inner Hebrew geresh.

    Names with different identity keys but the same near key differ only in spacing or in a
    geresh between two letters (``"בנימין ריצ'לר"`` / ``"בנימין ריצלר"``): maybe one person,
    maybe not (צ' is not צ), so the planner lists them for review instead of merging them.

    :param raw: The name as stored in the graph.
    :returns: The spaceless key without ``KEY_GERESH``.
    :rtype: str
    """
    return spaceless_key(raw).replace(KEY_GERESH, "")


def scripts_of(text: str) -> Set[str]:
    """Return the scripts (LATIN, HEBREW, ARABIC, ...) of the text's letters.

    :param text: A name.
    :returns: Script names taken from the Unicode character names; modifier letters are ignored.
    :rtype: Set[str]
    """
    return {unicodedata.name(c, "?").split(" ")[0] for c in text
            if unicodedata.category(c) in ("Lu", "Ll", "Lt", "Lo")}


def has_title(raw: str) -> bool:
    """Return True when the name carries an academic or honorific title (Dr., Prof., Rabbi, הרב).

    :param raw: The name as stored, or its standard form.
    :returns: Whether a title prefix (or a Hebrew honorific suffix) is present.
    :rtype: bool
    """
    tokens = collapse(raw).split()
    if not tokens:
        return False
    first = tokens[0]
    return (first.rstrip(".").casefold() in LATIN_TITLES or first in HEBREW_TITLES
            or tokens[-1] in HEBREW_HONORIFIC_SUFFIXES)


def is_capitalised_word(token: str) -> bool:
    """Return True for a word of two or more letters starting with a capital ("Pellicer", "Bru,").

    :param token: One whitespace-separated token.
    :returns: Whether it is a capitalised word (an initial is not).
    :rtype: bool
    """
    return token[:1].isupper() and sum(c.isalpha() for c in token) > 1


def comma_name_doubts(text: str) -> List[str]:
    """Reasons a one-comma name may not be "Surname, Given".

    :param text: Output of ``latin_text``.
    :returns: ``"French 'M.' before an initial"`` ("Wiet, M. G.", "Wiet, M G": M. is Monsieur) and/or
        ``"Arabic name with a comma"`` (a particle or kunya on either side, or ``al-``/``el-``
        opening the given part: medieval Arabic names are not Surname, Given).
    :rtype: List[str]
    """
    if text.count(",") != 1:
        return []
    given = text.split(",")[1].strip()
    initials = standard_tokens(given)
    monsieur = len(initials) > 1 and initials[0] == (format_initial("M"), True) and initials[1][1]
    arabic = given.casefold().startswith(("al-", "el-")) or any(
        word == "b." or word.lstrip(AYN_MARKS).casefold() in ARABIC_PARTICLES for word in text.replace(",", " ").split())
    return [reason for reason, doubtful in (("French 'M.' before an initial", monsieur),
                                            ("Arabic name with a comma", arabic)) if doubtful]


def fjp_initials(raw: str) -> int:
    """Count the trailing initials ``standard_name`` moves in front of the surname ("Marx A" -> 1).

    :param raw: The name as stored in the graph.
    :returns: The number of initials moved (0 for Hebrew, comma or non-FJP names).
    :rtype: int
    """
    text = latin_text(raw)
    return 0 if is_hebrew(text) or "," in text else trailing_initials(latin_tokens(text))


def initial_runs(tokens: List[str]) -> int:
    """Count the runs of consecutive initials ("A. Mediterranean Society S. D. Goitien" has 2).

    :param tokens: The words of a standard name.
    :returns: How many separate runs of initials tokens it has.
    :rtype: int
    """
    return sum(is_initials_token(t) and (i == 0 or not is_initials_token(tokens[i - 1])) for i, t in enumerate(tokens))


def is_phrase_word(word: str) -> bool:
    """Return True for a word that marks a phrase or title, not a name: "the" in any case, or a
    lower-case word that is not in ``NAME_WORDS`` ("scholars", "on", "family"). A word with a
    ``?`` is skipped (its lost letter may be the capital: "?ngeles").

    :param word: One word of a standard name.
    :returns: Whether the word does not belong in a personal name.
    :rtype: bool
    """
    cased = [c for c in word if c.lower() != c.upper()]
    return word.casefold() == "the" or (bool(cased) and not any(c.isupper() for c in cased) and "?" not in word
                                        and word.rstrip(".").casefold() not in NAME_WORDS)


def ends_after_particle(text: str) -> bool:
    """Return True when a Latin name ends in an initial right after a particle ("Isaac ben E").

    Such a name was probably cut off (a lost patronymic), so its last letter may not be an initial.

    :param text: Output of ``latin_text``.
    :returns: Whether the last token is an initial and the one before it a particle.
    :rtype: bool
    """
    if is_hebrew(text) or "," in text:
        return False
    tokens = latin_tokens(text)
    return len(tokens) > 1 and tokens[-1][1] and tokens[-2][0].rstrip(".").casefold() in PARTICLES


def comma_given(raw: str) -> Optional[str]:
    """Return the part after the comma of a one-comma name ("Gil, Palestine" -> "Palestine").

    :param raw: The name as stored in the graph.
    :returns: The given part, or None when the name has no comma or several.
    :rtype: Optional[str]
    """
    text = latin_text(raw)
    return text.split(",")[1].strip() if text.count(",") == 1 else None


#: Reasons for which the standard form is known to be unreliable: the review sheet proposes nothing.
UNRELIABLE_PROPOSAL = frozenset({
    "no upper-case letter", "contains a digit", "contains '<' or '>'", "several authors", "parentheses",
    "conjunction 'I'/'Y' between names", "several trailing initials", "initials in two separate runs",
    "phrase? (lower-case word or 'the')", "truncated name? (initial after a particle)", "Hebrew surname first",
    "French 'M.' before an initial", "Arabic name with a comma",
})


def review_reasons(raw: str) -> List[str]:
    """List why a name must not be changed or merged automatically.

    Besides the spec's reasons (empty, ``?``, title, several commas, single token, mixed
    scripts): no upper-case letter at all ("the editor"), a digit, ``<``/``>``, several authors
    (``and``/``et``/``und``/``&``/``;``, Hebrew ``ו`` glued to an initial: ``ומ'``), parentheses,
    the comma doubts of ``comma_name_doubts``, a Catalan/Spanish ``I``/``Y`` between two
    capitalised words ("Josep Pellicer I Bru"), two or more FJP trailing initials ("Haelst J V":
    van/del/Díaz look like initials), an all-caps name ("E. ROMERO", "J.M.": the standard does no
    title-casing), initials in two separate runs ("A Mediterranean
    Society S.D Goitien"), a phrase word (``is_phrase_word``: "R. Gamliel the Elder", "Mosseri
    family"), an initial right after a particle ("Isaac ben E": cut off?) and a Hebrew name ending
    in a one-letter initial ("אברמסון ש.": surname first, which the standard never reorders).

    :param raw: The name as stored in the graph.
    :returns: Reason codes, in a fixed order; empty when the name is fine.
    :rtype: List[str]
    """
    standard, text = standard_name(raw), latin_text(raw)
    cased = [c for c in text if c.lower() != c.upper()]
    words, tokens = text.split(), standard.split()
    checks = (
        ("empty", not identity_key(raw)),
        ("contains '?' (lost accent)", "?" in raw),
        ("title prefix", has_title(raw) or has_title(standard)),
        ("several commas", raw.count(",") > 1),
        ("single token", len(tokens) == 1),
        ("mixed scripts", len(scripts_of(raw)) > 1),
        ("no upper-case letter", bool(cased) and not any(c.isupper() for c in cased)),
        ("contains a digit", any(c.isdigit() for c in text)),
        ("contains '<' or '>'", "<" in text or ">" in text),
        ("several authors", any(j in f" {text.casefold()} " for j in AUTHOR_JOINERS) or "&" in text or ";" in text
         or any(HEBREW_AND_INITIAL.fullmatch(w) for w in words)),
        ("parentheses", "(" in text or ")" in text),
        ("conjunction 'I'/'Y' between names", any(
            w in ("I", "Y") and is_capitalised_word(words[i - 1]) and is_capitalised_word(words[i + 1])
            for i, w in enumerate(words[1:-1], 1))),
        ("several trailing initials", fjp_initials(raw) > 1),
        ("all caps", is_all_caps(text)),
        ("initials in two separate runs", initial_runs(tokens) > 1),
        ("phrase? (lower-case word or 'the')", any(c.isupper() for c in cased) and any(map(is_phrase_word, tokens))),
        ("truncated name? (initial after a particle)", ends_after_particle(text)),
        ("Hebrew surname first", is_hebrew(text) and len(tokens) > 1
         and HEBREW_ONE_INITIAL.fullmatch(tokens[-1]) is not None),
    )
    return [reason for reason, failed in checks if failed] + comma_name_doubts(text)


def needs_review(raw: str) -> Optional[str]:
    """Return why a name needs a human decision, or None.

    :param raw: The name as stored in the graph.
    :returns: The reasons joined with "; ", or None when the name can be handled automatically.
    :rtype: Optional[str]
    """
    reasons = review_reasons(raw)
    return "; ".join(reasons) if reasons else None


def diacritic_count(text: str) -> int:
    """Count diacritics: combining marks in the NFD form plus transliteration modifier letters.

    Schäfer has 1 and Schafer 0; Beʾeri (with the modifier letter ʾ) has 1 and Beeri 0.

    :param text: A name.
    :returns: The number of diacritics.
    :rtype: int
    """
    return sum(1 for c in unicodedata.normalize("NFD", text) if unicodedata.category(c)[0] == "M"
               or unicodedata.category(c) == "Lm")


def has_house_initials(name: str) -> bool:
    """Return True when every token made of initials is written in ``INITIAL_STYLE``.

    :param name: A display name.
    :returns: Whether all initials follow the house style.
    :rtype: bool
    """
    return all(split_initials(token) in (None, [token]) for token in name.split())


def initial_rank(name: str) -> int:
    """Rank how a name writes its initials (smaller is better).

    Latin: house style 0, otherwise 1. Hebrew: every initial with a geresh or gershayim
    (``ש' אסף``, ``ד"צ בנעט``) 0, the worst initial with a period (``ש. אסף``) 1, a bare initial
    (``ש אסף``) 2; a name without initials 0.

    :param name: A standard name.
    :returns: 0, 1 or 2.
    :rtype: int
    """
    if not is_hebrew(name):
        return 0 if has_house_initials(name) else 1
    return max((0 if any(c in HEBREW_INITIAL_MARKS for c in token) else 1 if "." in token else 2
                for token in name.split() if HEBREW_INITIALS.fullmatch(token)), default=0)


def is_initials_token(token: str) -> bool:
    """Return True for a token of Latin ("S.", "S.D.") or Hebrew ("ש'", 'ד"צ', "ח") initials.

    :param token: One whitespace-separated token.
    :returns: Whether the token is only initials.
    :rtype: bool
    """
    return split_initials(token) is not None or HEBREW_INITIALS.fullmatch(token) is not None


def initials_only(name: str) -> bool:
    """Return True when the standard name is initials + surname, no given name ("F. Rosenthal", "ש' אסף").

    :param name: A name, raw or standard.
    :returns: Whether its standard form starts with an initial and has a non-initial token.
    :rtype: bool
    """
    tokens = standard_name(name).split()
    return bool(tokens) and is_initials_token(tokens[0]) and not all(map(is_initials_token, tokens))


def choose_display(names: Iterable[str], degrees: Optional[Iterable[int]] = None) -> str:
    """Pick the cluster's display name among its members' standard names, deterministically.

    The form most likely to appear wins. Ranks each distinct standard name by: (a) no ``?`` (a
    lost accent is never a form that appears in print); (b) the most edges: the summed degree of
    the members with that form; then the tie-breaks (c) more diacritics (more information: Schäfer
    over Schafer, Beʾeri over Beeri); (d) initials in the house style, for Hebrew geresh/gershayim
    over a period over bare (``initial_rank``); (e) some member is already stored in that form
    (raw == standard); (f) lexicographic order. The result is always one of the members' standard
    forms: Hebrew initials are not rewritten (``["ח שירמן", "ח. שירמן"]`` -> ``"ח. שירמן"``).

    :param names: The members' current (raw) names.
    :param degrees: The members' degrees, in the same order (default: 1 each, i.e. member count).
    :returns: The canonical display name.
    :rtype: str
    :raises ValueError: When no names are given.
    """
    raw_names = list(names)
    if not raw_names:
        raise ValueError("choose_display needs at least one name")
    weight: Dict[str, int] = {}
    for raw, degree in zip(raw_names, [1] * len(raw_names) if degrees is None else degrees, strict=True):
        standard = standard_name(raw)
        weight[standard] = weight.get(standard, 0) + degree
    stored = set(raw_names)

    def rank(candidate: str) -> Tuple[bool, int, int, int, bool, str]:
        """Sort key: smaller is better.

        :param candidate: One distinct standard name.
        :returns: The rank tuple described above.
        :rtype: Tuple[bool, int, int, int, bool, str]
        """
        return ("?" in candidate, -weight[candidate], -diacritic_count(candidate), initial_rank(candidate),
                candidate not in stored, candidate)

    return min(weight, key=rank)
