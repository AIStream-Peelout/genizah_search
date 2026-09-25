"""Curated festival-day placement for the /sukkot "festival days" strip.

Each Sukkot fragment gets at most one slot on the strip: a preparations slot,
the first and second days, hol ha-moʿed, Hoshana Rabbah, Shemini Atzeret and
(diaspora) Simhat Torah.

Two kinds of placement exist:

* ``"named in the source"``: a hand-curated entry in :data:`NAMED_PLACEMENTS`.
  Every entry carries ``evidence`` phrases, and the placement is only used when
  one of them occurs in the seed row's human-sourced ``card_line`` or
  ``date_text``. If the wording is missing (for example after the seed is
  edited), the entry is rejected and the fragment falls back to its theme.
* ``"placed by theme"``: undated festival liturgy placed on the day its theme
  belongs to (:data:`THEME_FALLBACK`).

The 921 calendar proclamation is a special ``"contested"`` entry spanning the
second day (Stern 2019) and Hoshana Rabbah (Laufer 2025 and earlier scholars).

Sources for the curated entries: ``tech_notes.md`` section 3.2(e) (group
``"tech_notes"``) and the seed's own card wording (group ``"card wording"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: The strip's slots, in display order (Sukkot 5787).
FESTIVAL_DAYS: List[Dict[str, Any]] = [
    {"id": "prep", "tishri": None, "label": "Before the festival",
     "sublabel": "11–14 Tishri: preparations", "gregorian": "22–25 Sep 2026"},
    {"id": "d15", "tishri": 15, "label": "First day", "sublabel": "15 Tishri", "gregorian": "Sat 26 Sep"},
    {"id": "d16", "tishri": 16, "label": "Second day", "sublabel": "16 Tishri", "gregorian": "Sun 27 Sep"},
    {"id": "hol", "tishri": None, "label": "Hol ha-moʿed", "sublabel": "17–20 Tishri",
     "gregorian": "Mon 28 Sep – Thu 1 Oct"},
    {"id": "d21", "tishri": 21, "label": "Hoshana Rabbah", "sublabel": "21 Tishri · the Day of the Willow",
     "gregorian": "Fri 2 Oct"},
    {"id": "d22", "tishri": 22, "label": "Shemini Atzeret", "sublabel": "22 Tishri", "gregorian": "Sat 3 Oct"},
    {"id": "d23", "tishri": 23, "label": "Simhat Torah", "sublabel": "23 Tishri (diaspora)", "gregorian": "Sun 4 Oct"},
]

#: Valid day ids.
DAY_IDS: Tuple[str, ...] = tuple(day["id"] for day in FESTIVAL_DAYS)

BASIS_NAMED = "named in the source"
BASIS_THEME = "placed by theme"
BASIS_CONTESTED = "contested"

#: doc_id -> curated placement. ``evidence``: phrases, one of which must occur in
#: the row's card_line or date_text; ``group``: where the placement comes from.
NAMED_PLACEMENTS: Dict[str, Dict[str, Any]] = {
    # --- tech_notes section 3.2(e) ---
    "Cambridge_CUL_T_S_NS_J221": {
        "day": "d21", "evidence": ["יום ערבה", "Day of the Willow"], "group": "tech_notes",
        "why": "Recto: palm-branch transport (13 Tishri 1219, a preparations entry); verso: the 'יום ערבה' "
               "expenses. One slot only: the card names the Day of the Willow, so d21."},
    "Cambridge_CUL_T_S_AS_109_184": {
        "day": "hol", "evidence": ["sixth and seventh days"], "group": "tech_notes",
        "why": "Musaf rubrics for days 6-7 (20-21 Tishri); the card does not single out the seventh day, "
               "so hol ha-moʿed."},
    "Manchester_JRL_B_2562": {
        "day": "d21", "evidence": ["sixth and seventh days"], "group": "tech_notes",
        "why": "Hoshanot for the sixth and seventh days; the seventh is Hoshana Rabbah."},
    "Cambridge_CUL_T_S_8J17_24": {
        "day": "d21", "evidence": ["for the seventh day"], "group": "tech_notes",
        "why": "The cantor asks for the hoshanot 'for the seventh day'."},
    "Philadelphia_CAJS_Halper_251": {
        "day": "d21", "evidence": ["Hoshana Rabbah", "יום ערבה"], "group": "tech_notes",
        "why": "Booklet for 21-23 Tishri; it opens with the Day of the Willow."},
    "New_York_JTS_ENA_2114_1": {
        "day": "d22", "evidence": ["Shemini Atzeret"], "group": "tech_notes",
        "why": "Prayers for Shemini Atzeret."},
    "Cambridge_CUL_T_S_B13_16": {
        "day": "d22", "evidence": ["Shemini Atzeret"], "group": "tech_notes",
        "why": "Amidah for Shemini Atzeret (and Sukkot)."},
    "Cambridge_CUL_T_S_Misc_35_11": {
        "day": "d21", "evidence": ["Hoshana Rabbah"], "group": "tech_notes",
        "why": "The Hoshana Rabbah assembly of 1029 and the ban."},
    "Oxford_Bodleian_Bodl_MS_heb_b_3_5": {
        "day": "d16", "evidence": ["second day"], "group": "tech_notes",
        "why": "Damietta 1157: the man sailed away on the second day."},
    "Cambridge_CUL_T_S_12_305": {
        "day": "d16", "evidence": ["second day of Sukkot"], "group": "tech_notes",
        "why": "The bread riot 'on the second day'; the letter also tells of a Day of ʿArava quarrel."},
    "Cambridge_CUL_T_S_K2_8": {
        "day": "d21", "evidence": ["Willow: Wednesday"], "group": "tech_notes",
        "why": "Calendar entry '(day of the) Willow: Wednesday'."},
    # --- the seed's own card wording (not in the tech_notes list) ---
    "Cambridge_CUL_Or_1080_J105": {
        "day": "d21", "evidence": ["Hoshana Rabbah"], "group": "card wording",
        "why": "Appointment on the Mount of Olives on Hoshana Rabbah (Gil's edition)."},
    "Cambridge_CUL_T_S_NS_320_42": {
        "day": "d21", "evidence": ["day of ʿArava"], "group": "card wording",
        "why": "The head preached on the Mount 'on the day of ʿArava'."},
    "Cambridge_CUL_T_S_NS_159_124": {
        "day": "d21", "evidence": ["Hoshana Rabbah"], "group": "card wording",
        "why": "The hoshana for Hoshana Rabbah."},
    "Cambridge_Lewis_Gibson_L_G_Lit_I_33": {
        "day": "d21", "evidence": ["Hoshana Rabbah"], "group": "card wording",
        "why": "Yotser for Hoshana Rabbah."},
    "Paris_AIU_IV_B_59": {
        "day": "d16", "evidence": ["second day"], "group": "card wording",
        "why": "Derashot for the second day of Sukkot."},
    "Cambridge_CUL_T_S_K6_80": {
        "day": "hol", "evidence": ["intermediate days"], "group": "card wording",
        "why": "Piyyutim for the intermediate days."},
    "Cambridge_CUL_T_S_B14_42": {
        "day": "d22", "evidence": ["Shemini Atzeret"], "group": "card wording",
        "why": "Haftarah rubric for Shemini Atzeret."},
    "Cambridge_CUL_T_S_K27_65": {
        "day": "d22", "evidence": ["Shemini Atzeret"], "group": "card wording",
        "why": "Musaf Amidah for Shemini Atzeret."},
    "Cambridge_Lewis_Gibson_L_G_Glass_37": {
        "day": "d22", "evidence": ["Shemini Atzeret"], "group": "card wording",
        "why": "Qerova for Shemini Atzeret."},
    "Oxford_Bodleian_Bodl_MS_heb_e_39_1": {
        "day": "d23", "evidence": ["Simhat Torah"], "group": "card wording",
        "why": "'For שמחת תורה'."},
    "Oxford_Bodleian_Bodl_MS_heb_e_39_153": {
        "day": "d23", "evidence": ["Simhat Torah"], "group": "card wording",
        "why": "Piyyut for Simhat Torah."},
    "Cambridge_CUL_T_S_B18_25": {
        "day": "d23", "evidence": ["שמחת תורה"], "group": "card wording",
        "why": "Headed 'אפטרתא דיום שמחת תורה'; it also has the second-day haftarah, so one of two named days."},
}

#: The 921 proclamation: one contested bracket (Stern vs Laufer).
CONTESTED_921: Dict[str, Any] = {
    "day": "d16", "basis": BASIS_CONTESTED, "span": ["d16", "d21"],
    "label": "Stern 2019: second day · Laufer 2025 and earlier scholars: Hoshana Rabbah",
}

#: doc_id -> evidence phrases for the contested 921 placement.
CONTESTED_PLACEMENTS: Dict[str, List[str]] = {
    "Cambridge_CUL_T_S_NS_194_92": ["Day of Assembly"],
    "New_York_JTS_ENA_2555_1": ["Day of Assembly"],
}

#: Items named in tech_notes section 3.2(e) that are not rows of the seed (reported, not placed).
LISTED_NOT_IN_SEED: List[Dict[str, str]] = [
    {"shelfmark": "T-S 8J28.3", "day": "prep", "why": "Sukkot quires asked back the night after Yom Kippur"},
    {"shelfmark": "T-S 10J12.20", "day": "prep", "why": "letter from the eve of Sukkot"},
    {"shelfmark": "T-S 10J5.10", "day": "d23", "why": "the 1064 death on Simhat Torah night"},
    {"shelfmark": "T-S 10J11.13", "day": "d23", "why": "the 1064 death on Simhat Torah night"},
    {"shelfmark": "JRL C 127", "day": "d21", "why": "1844 accounts dated Hoshana Rabbah"},
]

#: theme id -> day id for undated festival material. Themes absent here get no slot.
THEME_FALLBACK: Dict[str, str] = {
    "hoshanot_hoshana_rabbah": "d21",
    "shemini_atzeret_geshem": "d22",
    "simhat_torah": "d23",
    "festival_prayers": "hol",
    "qerovot_piyyut": "hol",
    "haftarot_targum_bible": "hol",
    "homilies": "hol",
    "four_species": "prep",
}


def source_wording(fragment: Dict[str, Any]) -> str:
    """Return the human-sourced wording a curated placement is checked against.

    :param fragment: A seed fragment row.
    :return: ``card_line`` and ``date_text`` joined by a newline (missing parts are empty).
    """
    return "\n".join(str(fragment.get(key) or "") for key in ("card_line", "date_text"))


def has_evidence(fragment: Dict[str, Any], phrases: List[str]) -> bool:
    """Tell whether any evidence phrase occurs in the row's card_line or date_text.

    :param fragment: A seed fragment row.
    :param phrases: Candidate phrases; matching is case-insensitive.
    :return: True when at least one phrase is found.
    """
    wording = source_wording(fragment).casefold()
    return any(phrase.casefold() in wording for phrase in phrases)


def explain_placement(fragment: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Place a fragment on the strip and explain any rejected curated entry.

    :param fragment: A seed fragment row (needs ``doc_id``, ``theme``, ``card_line``; ``date_text`` optional).
    :return: ``(placement, problem)``. ``placement`` is the public ``festival_day`` dict or None;
        ``problem`` is a message when a curated entry existed but its wording was not found in the row.
    """
    doc_id = fragment.get("doc_id")
    problem: Optional[str] = None
    if doc_id in CONTESTED_PLACEMENTS:
        if has_evidence(fragment, CONTESTED_PLACEMENTS[doc_id]):
            return dict(CONTESTED_921, span=list(CONTESTED_921["span"])), None
        problem = f"contested 921 entry rejected: none of {CONTESTED_PLACEMENTS[doc_id]} in card_line/date_text"
    elif doc_id in NAMED_PLACEMENTS:
        entry = NAMED_PLACEMENTS[doc_id]
        if has_evidence(fragment, entry["evidence"]):
            return {"day": entry["day"], "basis": BASIS_NAMED}, None
        problem = f"curated {entry['day']} rejected: none of {entry['evidence']} in card_line/date_text"
    theme_day = THEME_FALLBACK.get(fragment.get("theme"))
    if theme_day is None:
        return None, problem
    return {"day": theme_day, "basis": BASIS_THEME}, problem


def place_fragment(fragment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the public ``festival_day`` value for a fragment.

    :param fragment: A seed fragment row.
    :return: ``{"day", "basis"}`` (plus ``span`` and ``label`` for the contested 921 entry), or None
        when the fragment has no natural place on the strip.
    """
    placement, _problem = explain_placement(fragment)
    return placement
