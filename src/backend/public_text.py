"""Public form of catalogue free text (pure, stdlib only).

The one place that decides how catalogue text may appear on the public site:
the API (result cards, document modal) and the server-rendered fragment pages
all go through it.

* :func:`scrub_provider_mentions` removes names of, and links to, the
  unattributed catalogue provider (hosts in :data:`source_links.BLOCKED_HOSTS`).
* :func:`clean_block_text` / :func:`clean_inline_text` also strip formatting
  markup, keeping editorial brackets such as ``<and send it>``.
* The ``public_*`` helpers shape the short list and scalar display fields of
  :class:`DocumentMetadata` (alternative titles, subjects, genres, catalogue
  headings, century, visible date).
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, List, Optional, Tuple

from src.backend.source_links import BLOCKED_HOSTS

# ---------------------------------------------------------------------------
# Provider scrubbing
# ---------------------------------------------------------------------------

# URLs or bare host names on the unattributed provider's domains.
_BLOCKED_URL_RE = re.compile(
    r"(?:https?://)?(?:[\w-]+\.)*(?<![\w-])(?:" + "|".join(re.escape(host) for host in BLOCKED_HOSTS) + r")\b"
    r"(?:[^\s<>\"')\]]*[^\s<>\"')\].,;:!?])?",
    re.IGNORECASE,
)
# The provider acronym, and an attribution phrase built on it ("Information
# from FGP", "Data in part per the FJP", "with information from FGP").
_ACR = r"(?:the\s+)?F[GJ]P\b"
_ATTR = (
    r"\b(?:with\s+)?(?:Information|Info|Data|Description)(?:\s+in\s+part)?"
    r"\s+(?:from|per|via|according\s+to)\s+" + _ACR
)
# A noun the acronym can qualify ("FGP join finder", "the FGP bibliography").
_PROVIDER_NOUN = r"(?:join|image|photo|edition|bibliograph)\w*"
# A mention the replacement wording must not swallow ("on FGP images").
_NOT_A_PLACE = r"(?!\s+" + _PROVIDER_NOUN + r")"
# Ordered: whole attribution constructions first, so they go as a unit; the
# bare acronyms last.  Removing just the acronym would leave broken sentences
# ("Information.", "edition on.", "CUDL and.").
_PROVIDER_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # "See FGP for further information." -> the whole pointer sentence is dropped
    # (case-sensitive: a sentence-initial pointer; "…: see FGP." is reworded below).
    (re.compile(r"[ \t]*\bSee\s+(?:the\s+)?F[GJ]P\b[^.!?\n]*[.!?]?"), ""),
    # "FGP and CUDL" / "CUDL and FGP" / "PGP, FGP" -> the other source only
    (re.compile(r"\bF[GJ]P\s+(?:and|&)\s+", re.I), ""),
    (re.compile(r",?\s+(?:and|&)\s+" + _ACR, re.I), ""),
    (re.compile(r"\s*,\s*" + _ACR, re.I), ""),
    # "(Information in part from FGP)", "(with information from FGP)" -> dropped
    (re.compile(r"\s*\(\s*" + _ATTR + r"[^()]*\)", re.I), ""),
    # "Information from FGP." / "Information from FGP by X." -> dropped
    (re.compile(r"\s*" + _ATTR + r"(?:\s+by\s+[^.()\n]*)?\.?", re.I), ""),
    # "(not on FGP)", "(as catalogued on FGP)", "(according to FGP)" -> dropped
    (re.compile(r"\s*\([^()]*\b(?:on|in|at|see|to)\s+" + _ACR + r"[^()]*\)", re.I), ""),
    # "(FGP)", "(per FGP)", "(from the FJP)"
    (re.compile(r"\s*\(\s*(?:(?:per|from|via|see|according\s+to)\s+)?(?:the\s+)?F[GJ]P\s*\)", re.I), ""),
    # "According to the FGP bibliography, …", "found via FGP join finder" ->
    # "… another catalogue's bibliography": dropping only the acronym phrase
    # would leave "bibliography, …" / "found join finder".
    (
        re.compile(
            r"\b(per|from|via|according(?:\s+to)?)\s+" + _ACR + r"(?=\s+" + _PROVIDER_NOUN + r")", re.I
        ),
        r"\1 another catalogue's",
    ),
    # ", per FGP", " according (to) the FJP", " with information from FGP"
    (
        re.compile(
            r",?\s*\b(?:(?:with\s+)?information\s+)?(?:per|from|via|according(?:\s+to)?)\s+"
            + _ACR
            + _NOT_A_PLACE,
            re.I,
        ),
        "",
    ),
    # "edition on FGP", "listed as verso on FGP", "joins: see FGP" -> neutral wording
    (re.compile(r"\b(?:on|in|at)\s+" + _ACR + _NOT_A_PLACE, re.I), "in another catalogue"),
    (re.compile(r"\b(see)\s+" + _ACR + _NOT_A_PLACE, re.I), r"\1 another catalogue"),
    # "The Friedberg Genizah Project and the Ben-Zvi Institute" -> "the Ben-Zvi Institute"
    (
        re.compile(
            r"\b(?:the\s+)?Friedberg\s+(?:Genizah|Jewish\s+Manuscripts?)\s+(?:Project|Society)"
            r"(?:\s*\(\s*F[GJ]P\s*\))?(?:\s+and\s+|\s*,\s*)?",
            re.I,
        ),
        "",
    ),
    # "the Friedberg/NLI metadata" -> "the NLI metadata"
    (re.compile(r"\bFriedberg\s*/\s*", re.I), ""),
    (re.compile(r"\s*/\s*Friedberg\b", re.I), ""),
    (re.compile(r"\bF[GJ]P\b", re.I), ""),
)
# Applied only to text a provider pattern changed, so untouched catalogue text
# keeps its exact spacing and punctuation.
_TIDY_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]+([.,;:!?)])"), r"\1"),
    (re.compile(r"\([ \t]+"), "("),
    (re.compile(r"([,;])\1+"), r"\1"),
    # The doubled period left after a removed "(…)." (ellipses untouched).
    (re.compile(r"(?<![.])([.!?])\s*\.(?![.])"), r"\1"),
    # Punctuation left at the start of a line ("." of a removed "(…).").
    (re.compile(r"(^|\n)[ \t]*(?:[,;:]|\.(?!\.))[ \t]*"), r"\1"),
)
# Formatting markup found in catalogue descriptions and citations.  Narrow on
# purpose: editors use "<and send it>" for supplied text, which must survive.
_MARKUP_TAG_RE = re.compile(
    r"<(?:/?(?:b|i|em|strong|u|sup|sub|small|span|p|div|ol|ul|li)\b[^<>]*|br\s*/?|a\s+href\s*=[^<>]*|/a\s*)>",
    re.IGNORECASE,
)


def scrub_provider_mentions(text: Optional[str]) -> str:
    """Remove names of, and links to, the unattributed catalogue provider.

    Drops URLs on ``genizah.org`` / ``jewishmanuscripts.org`` (any subdomain),
    the acronyms FGP/FJP together with the construction around them (an
    attribution such as "Information from FGP.", "per FGP", "(not on FGP)",
    "See FGP for …", a conjunction such as "CUDL and FGP"), rewording a
    remaining place mention ("edition on FGP") to "in another catalogue" and a
    source qualifying a noun ("according to the FGP bibliography") to
    "another catalogue's bibliography", and
    the project names "Friedberg Genizah Project" / "Friedberg Jewish
    Manuscript Society".
    "Friedberg" as a manuscript collection ("Friedberg MSS 9-004/22", held in
    Toronto) or as an author's surname is kept: it identifies an object or a
    person, not the provider.

    :param text: Catalogue free text (description, citation, title).
    :return: The text without provider mentions, whitespace and punctuation tidied.
    :rtype: str
    """
    original = text or ""
    out = _BLOCKED_URL_RE.sub("", original)
    for pattern, repl in _PROVIDER_PATTERNS:
        out = pattern.sub(repl, out)
    if out != original:
        for pattern, repl in _TIDY_PATTERNS:
            out = pattern.sub(repl, out)
    return out.strip()


def strip_markup(text: Optional[str]) -> str:
    """Remove formatting tags and decode entities (no other change).

    :param text: Catalogue text that may carry ``<em>``, ``<a href>`` etc.
    :return: Plain text.
    :rtype: str
    """
    return html.unescape(_MARKUP_TAG_RE.sub("", text or ""))


def clean_block_text(text: Optional[str]) -> str:
    """Public form of a multi-paragraph catalogue text (e.g. a description).

    :param text: Raw text.
    :return: Markup-free, provider-scrubbed text with ``\\n`` line breaks and at
        most one blank line in a row.
    :rtype: str
    """
    plain = strip_markup(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [scrub_provider_mentions(line) for line in plain.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_inline_text(text: Optional[str]) -> str:
    """Public one-line form of a catalogue text (titles, meta tags, JSON-LD).

    :param text: Raw text.
    :return: Markup-free, provider-scrubbed text on one line.
    :rtype: str
    """
    return " ".join(scrub_provider_mentions(" ".join(strip_markup(text).split())).split())


# ---------------------------------------------------------------------------
# Display fields of DocumentMetadata
# ---------------------------------------------------------------------------

MAX_ALT_TITLES = 10
MAX_SUBJECTS = 20
MAX_GENRES = 20
MAX_SCHOLARLY_FRAMES = 10
# A century above this is a misread year (an Anno Mundi year read as CE,
# e.g. "48th century" for 27 Nisan 4790).
MAX_CENTURY = 21
# Stock alternative titles that say nothing about the fragment.
ALT_TITLE_PLACEHOLDERS: Tuple[str, ...] = (
    "אין בידנו לעת עתה פרטים על תוכנו של הקטע",
    "Unidentified",
    "תחום לא מזוהה",
)
_CENTURY_RE = re.compile(r"^\s*(\d+)\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
# An ISO 8601 date or interval as catalogued ("1125-12-09/1125-12-18", "1094/1095").
ISO_DATE_RE = re.compile(r"^-?\d{3,4}(?:-\d\d(?:-\d\d)?)?(?:/-?\d{3,4}(?:-\d\d(?:-\d\d)?)?)?$")


def normalise_phrase(text: str) -> str:
    """Case-fold, turn punctuation into spaces and collapse whitespace.

    :param text: Title, description or placeholder.
    :return: Normalised text, for comparisons only.
    :rtype: str
    """
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def public_list(values: Any, cap: Optional[int], exclude: Iterable[str] = ()) -> List[str]:
    """Clean, de-duplicate and cap a list-valued text field (subjects, genres …).

    :param values: Raw field value (list, scalar or missing).
    :param cap: Maximum number of entries kept (``None``: no cap).
    :param exclude: Values to leave out (compared after :func:`normalise_phrase`).
    :return: One-line, provider-scrubbed strings, first occurrences only.
    :rtype: List[str]
    """
    seen = {normalise_phrase(value) for value in exclude}
    raw = values if isinstance(values, (list, tuple)) else [values]
    out: List[str] = []
    for value in raw:
        if cap is not None and len(out) >= cap:
            break
        text = clean_inline_text(value) if isinstance(value, str) else ""
        key = normalise_phrase(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def public_alt_titles(values: Any, shelf_mark: Optional[str], description: Optional[str]) -> List[str]:
    """Alternative titles worth showing next to a record.

    :param values: Raw ``alt_titles``.
    :param shelf_mark: Display shelfmark (an alt title equal to it is dropped).
    :param description: Public description (an alt title it already contains,
        as whole words, is dropped).
    :return: At most :data:`MAX_ALT_TITLES` titles, stock placeholders removed.
    :rtype: List[str]
    """
    description_key = f" {normalise_phrase(description or '')} "
    exclude = list(ALT_TITLE_PLACEHOLDERS) + [shelf_mark or ""]
    titles = [
        title
        for title in public_list(values, cap=None, exclude=exclude)
        if f" {normalise_phrase(title)} " not in description_key
    ]
    return titles[:MAX_ALT_TITLES]


def public_century(value: Any) -> Optional[str]:
    """A plausible century label ("11th century"), or ``None``.

    :param value: Raw ``century``.
    :return: The cleaned label; ``None`` when empty or above :data:`MAX_CENTURY`.
    :rtype: Optional[str]
    """
    text = clean_inline_text(value) if isinstance(value, str) else ""
    match = _CENTURY_RE.match(text)
    if not text or (match and int(match.group(1)) > MAX_CENTURY):
        return None
    return text


def public_date_display(value: Any) -> Optional[str]:
    """Visible form of a catalogued date (cards, modal and fragment page alike).

    :param value: Raw ``date_display``.
    :return: The cleaned text with an ISO interval "a/b" written "a – b"
        (``"1125-12-09/1125-12-18"`` -> ``"1125-12-09 – 1125-12-18"``); other
        dates unchanged; ``None`` when empty.  The raw value stays the
        machine-readable form (schema.org ``temporalCoverage``).
    :rtype: Optional[str]
    """
    text = clean_inline_text(value) if isinstance(value, str) else ""
    if not text:
        return None
    return text.replace("/", " – ") if ISO_DATE_RE.match(text) else text
