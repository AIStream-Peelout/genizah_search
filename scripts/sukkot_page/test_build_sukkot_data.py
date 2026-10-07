"""Unit tests for the /sukkot page builder and the festival-day placement.

Run with ``python3 -m pytest scripts/sukkot_page -q``. No network access: image
checks use a fake status function and thumbnails use a generated image.
"""

from __future__ import annotations

import copy
import io
import json
from typing import Any, Dict

import pytest

import build_sukkot_data as b
import festival_days as fd


def seed_row(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal seed fragment row that passes selection.

    :param overrides: Fields to replace.
    :return: Seed row dict.
    """
    row: Dict[str, Any] = {
        "doc_id": "Cambridge_CUL_T_S_X_1", "old_ids": ["Old_X_1"], "shelfmark": "T-S X1.1",
        "institution": "Cambridge University Library", "theme": "halakha", "subtheme": None, "genre": "law_learning",
        "clusters": [], "tags": [], "card_line": "Mishnah Sukkah 1:1", "card_source": "KTIV",
        "card_source_public": "KTIV (National Library of Israel)", "card_source_note": None,
        "card_source_text": None, "date_text": None, "date_ce_start": None, "date_ce_end": None,
        "date_precise": None, "date_kind": None, "date_caveat": None, "place": None,
        "images": {"ok_url": "https://storage.googleapis.com/cairo-genizah-es-json/images/x.jpg"},
        "has_human_transcription": False, "has_translation": False, "bibliography": [],
        "scholarship_note": None, "ai_read": {"available": False, "linkable": False, "ai_read_doc_id": None,
                                              "image_index": None, "n_agreed": None, "n_lines": None,
                                              "caveat": None, "zero_agreed_reads_public": []},
        "readiness": "ready", "readiness_flags": ["ready"], "notes": [], "holding_credit": "Image courtesy of CUL",
        "contested": False, "usable_now": True, "spot_check": None,
    }
    row.update(overrides)
    return row


def linkable_read(n_agreed: int = 6, caveat: Any = None) -> Dict[str, Any]:
    """Return a seed ``ai_read`` block for a linkable read.

    :param n_agreed: Agreed lines.
    :param caveat: Curated caveat.
    :return: ai_read dict.
    """
    return {"available": True, "linkable": True, "ai_read_doc_id": "Oxford_Bodleian_MS_heb_b_3_5",
            "only_under_old_id": True, "image_index": 0, "image_url": "https://example.org/9318_image.jpg",
            "n_agreed": n_agreed, "n_lines": 18, "vlm_model": "m", "label": "x", "caveat": caveat,
            "zero_agreed_reads_public": ["image 1: 0/10"]}


def minimal_seed() -> Dict[str, Any]:
    """Return a small seed with two fragments, one theme, one cluster and one ref.

    :return: Seed dict.
    """
    return {
        "genres": {"law_learning": {"label": "Law and learning", "order": 5}},
        "themes": {"halakha": {"title": "Law", "genre": "law_learning", "order": 8, "intro": "Probably so.",
                               "hedges": ["thus far"], "contested": [{"claim": "c", "positions": ["a", "b"]}],
                               "refs": ["ktiv"]}},
        "clusters": [{"id": "pair", "title": "Pair", "basis": "b", "caveat": None,
                      "members": ["Cambridge_CUL_T_S_X_1", "Cambridge_CUL_T_S_X_2", "Not_Included"],
                      "outside_seed": []},
                     {"id": "solo", "title": "Solo", "basis": "b", "caveat": None,
                      "members": ["Cambridge_CUL_T_S_X_1"], "outside_seed": []}],
        "refs": {"ktiv": "KTIV, National Library of Israel"},
        "fragments": [seed_row(clusters=["pair", "solo"]),
                      seed_row(doc_id="Cambridge_CUL_T_S_X_2", shelfmark="T-S X1.2", clusters=["pair"]),
                      seed_row(doc_id="Not_Included", shelfmark="T-S X1.3", usable_now=False)],
        "excluded_machine_only": [], "not_on_site": [],
    }


# --------------------------------------------------------------------------- shelfmarks


@pytest.mark.parametrize("raw, expected", [
    ("Cambridge University Library, Cambridge, England Ms. T-S 10 H 4.5", "T-S 10H4.5"),
    ("The Bodleian Libraries, University of Oxford, Oxford, England Ms. heb. d. 76.30", "Bodl. MS heb. d 76/30"),
    ("Cambridge CUL: Or.1080 7.5", "Or.1080 7.5"),
    ("Paris AIU: IV.B.46", "AIU IV.B.46"),
    ("New York JTS: ENA 2555.1", "ENA 2555.1"),
    ("Manchester: Rylands B 2562", "JRL B 2562"),
    ("Bodl. MS heb. b 3/5", "Bodl. MS heb. b 3/5"),
    ("T-S NS 159.124", "T-S NS 159.124"),
    ("T-S AS 109.184", "T-S AS 109.184"),
    ("T-S F1(1).3", "T-S F1(1).3"),
    ("AIU IV.A.157", "AIU IV.A.157"),
    ("RNL Yevr. III B 904", "RNL Yevr. III B 904"),
])
def test_normalise_shelfmark(raw: str, expected: str) -> None:
    """Institution prefixes are stripped and the page convention applied.

    :param raw: Input shelfmark.
    :param expected: Expected label.
    """
    assert b.normalise_shelfmark(raw) == expected


# --------------------------------------------------------------------------- festival days


def test_place_named_in_source() -> None:
    """A curated entry whose wording is in the card line is placed as named."""
    row = seed_row(doc_id="Cambridge_CUL_T_S_8J17_24", theme="hoshanot_hoshana_rabbah",
                   card_line='The cantor asks for the hoshanot "for the seventh day" of Sukkot')
    assert fd.place_fragment(row) == {"day": "d21", "basis": fd.BASIS_NAMED}


def test_place_rejects_curated_entry_without_wording() -> None:
    """Without its evidence wording a curated entry falls back to the theme and is explained."""
    row = seed_row(doc_id="Cambridge_CUL_T_S_B13_16", theme="festival_prayers", card_line="Amidah for Sukkot")
    placement, problem = fd.explain_placement(row)
    assert placement == {"day": "hol", "basis": fd.BASIS_THEME}
    assert problem and "rejected" in problem


def test_place_contested_921() -> None:
    """The 921 proclamation gets the contested bracket spanning d16 and d21."""
    row = seed_row(doc_id="Cambridge_CUL_T_S_NS_194_92", theme="calendar",
                   card_line="Ben Meir's son proclaims in Jerusalem, on the 'Day of Assembly', ...")
    placement = fd.place_fragment(row)
    assert placement["basis"] == "contested" and placement["span"] == ["d16", "d21"]
    assert placement["day"] == "d16" and "Stern 2019" in placement["label"] and "Laufer 2025" in placement["label"]


def test_place_contested_returns_a_copy() -> None:
    """Mutating a returned placement does not change the module constant."""
    row = seed_row(doc_id="Cambridge_CUL_T_S_NS_194_92", theme="calendar", card_line="the 'Day of Assembly'")
    fd.place_fragment(row)["span"].append("x")
    assert fd.CONTESTED_921["span"] == ["d16", "d21"]


@pytest.mark.parametrize("theme, day", [
    ("hoshanot_hoshana_rabbah", "d21"), ("shemini_atzeret_geshem", "d22"), ("simhat_torah", "d23"),
    ("festival_prayers", "hol"), ("qerovot_piyyut", "hol"), ("haftarot_targum_bible", "hol"),
    ("homilies", "hol"), ("four_species", "prep"),
])
def test_place_by_theme(theme: str, day: str) -> None:
    """Undated liturgy falls on its theme's day.

    :param theme: Theme id.
    :param day: Expected day id.
    """
    assert fd.place_fragment(seed_row(theme=theme)) == {"day": day, "basis": fd.BASIS_THEME}


@pytest.mark.parametrize("theme", ["halakha", "calendar", "mount_of_olives", "communal_sukkah",
                                   "dated_by_festival", "greetings_letters", "karaites", "later_customs_kabbalah"])
def test_place_none_for_other_themes(theme: str) -> None:
    """Themes without a natural day get no slot unless a day is named.

    :param theme: Theme id.
    """
    assert fd.place_fragment(seed_row(theme=theme)) is None


def test_date_text_counts_as_evidence() -> None:
    """Evidence may come from date_text as well as card_line."""
    row = seed_row(doc_id="Cambridge_CUL_T_S_K2_8", theme="calendar", card_line="A calendar",
                   date_text="(day of the) Willow: Wednesday")
    assert fd.place_fragment(row) == {"day": "d21", "basis": fd.BASIS_NAMED}


def test_curated_days_are_valid() -> None:
    """Every curated day id exists on the strip."""
    days = {entry["day"] for entry in fd.NAMED_PLACEMENTS.values()} | set(fd.THEME_FALLBACK.values())
    days |= {fd.CONTESTED_921["day"], *fd.CONTESTED_921["span"]}
    assert days <= set(fd.DAY_IDS)


# --------------------------------------------------------------------------- text helpers


def test_machine_note_drops_zero_agreed_and_instructions() -> None:
    """Caveats that name a zero-agreed read, or are instructions, do not ship."""
    caveat = "A second public read of image 0 has 0 of 12 lines agreed: never link or call it checked; link only image 1."
    assert b.filter_public_clauses(caveat) is None
    kept = b.filter_public_clauses("Machine text writes 'נפראו' for נפרצו on 10 agreed lines; say so if linked.")
    assert kept == "Machine text writes 'נפראו' for נפרצו on 10 agreed lines."


def test_split_clauses_keeps_abbreviations() -> None:
    """A full stop inside "Sept. 29" does not split a clause."""
    assert b.split_clauses("PGP's 'Sept. 29, 1219' is wrong: it is 2 Oct.") == ["PGP's 'Sept. 29, 1219' is wrong:",
                                                                                "it is 2 Oct."]


def test_public_card_source_note() -> None:
    """Credit instructions are removed and redundant notes dropped."""
    assert b.public_card_source_note("Catalogue note (credit as a catalogue note).", "catalogue note") is None
    assert (b.public_card_source_note("Description OPenn; the stanza is from a catalogue note (credit as a catalogue note).",
                                      "OPenn (Penn Libraries)")
            == "Description OPenn; the stanza is from a catalogue note.")
    assert b.public_card_source_note("From FJP cards", "KTIV") is None
    assert b.public_card_source_note("Re-check before quoting.", "PGP") is None


def test_clean_citation() -> None:
    """Spacing is repaired, truncation trimmed and non-citations rejected."""
    assert b.clean_citation("Sacha Stern,The Jewish Calendar Controversy of 921/2 CE(Leiden: Brill, 2019).")[0] == \
        "Sacha Stern, The Jewish Calendar Controversy of 921/2 CE (Leiden: Brill, 2019)."
    trimmed, _ = b.clean_citation("M. A. Friedman,Jewish Marriage in Palestine (Tel Aviv, 1980), vol.…")
    assert trimmed == "M. A. Friedman, Jewish Marriage in Palestine (Tel Aviv, 1980)"
    assert b.clean_citation("Short (x), vol.…") == (None, "truncated in the seed")
    assert b.clean_citation("Genazim team from catalogs")[0] is None
    assert b.clean_citation("Lieberman Catalog. page unknown")[0] is None
    assert b.public_bibliography_line([]) == (None, "no bibliography in the seed")


# --------------------------------------------------------------------------- selection and fragments


def test_select_fragments_exclusions() -> None:
    """Rows that are not usable, need a human check, or are listed exclusions stay out with reasons."""
    seed = minimal_seed()
    seed["fragments"] += [
        seed_row(doc_id="A", readiness_flags=["needs_human_check"]),
        seed_row(doc_id="B"),
        seed_row(doc_id="C", shelfmark="ENA NS 5.29"),
        seed_row(doc_id="D", shelfmark="Or.1080 3.54"),
    ]
    seed["excluded_machine_only"] = [{"doc_id": "B", "shelfmark": "x"}]
    seed["not_on_site"] = [{"shelfmark": "Or.1080 3.54"}]
    included, excluded = b.select_fragments(seed)
    assert [row["doc_id"] for row in included] == ["Cambridge_CUL_T_S_X_1", "Cambridge_CUL_T_S_X_2"]
    reasons = {row["doc_id"]: row["reasons"] for row in excluded}
    assert set(reasons) == {"Not_Included", "A", "B", "C", "D"}
    assert any("needs_human_check" in reason for reason in reasons["A"])
    assert any("excluded_machine_only" in reason for reason in reasons["B"])


def test_build_machine_read_rules() -> None:
    """Only linkable reads with agreed lines ship, with the fixed label and the old-id href."""
    read = b.build_machine_read(linkable_read(6))
    assert read["href"] == "/read?doc=Oxford_Bodleian_MS_heb_b_3_5&image=0"
    assert read["label"] == ("Machine reading, not a human transcription: 6 of 18 lines agreed by two machine readers")
    assert b.build_machine_read(linkable_read(0)) is None
    assert b.build_machine_read(dict(linkable_read(6), linkable=False)) is None
    assert b.build_machine_read(None) is None


def test_build_fragment_ships_no_internal_fields() -> None:
    """Internal seed fields and ai_read text never reach the card."""
    row = seed_row(ai_read=linkable_read(6, "Read only under the old id X; do not use the read's content."),
                   scholarship_note="FJP says", notes=["internal"], card_source="FJP")
    card = b.build_fragment(row, set(), {})
    assert b.find_internal_keys(card) == []
    assert card["ai_read_doc_id"] == "Oxford_Bodleian_MS_heb_b_3_5"
    assert card["text_status"] == "machine" and card["machine_read"]["note"] is None
    assert "old_ids" not in card and card["usable_now"] is True


# --------------------------------------------------------------------------- validators


def test_forbidden_term_in_shipped_field_fails() -> None:
    """"FJP" in a shipped field fails the build."""
    card = b.build_fragment(seed_row(card_line="Catalogued by FJP"), set(), {})
    hits = b.find_forbidden_terms({"fragments": [card]})
    assert hits and "card_line" in hits[0]


def test_forbidden_term_only_in_unshipped_field_passes() -> None:
    """"FJP" only in internal fields never reaches the output."""
    row = seed_row(card_source="FJP", card_source_text="FJP card", scholarship_note="Friedberg FJMS",
                   notes=["genizah.org"], spot_check={"source": "FGP"})
    card = b.build_fragment(row, set(), {})
    assert b.find_forbidden_terms({"fragments": [card]}) == []


@pytest.mark.parametrize("text", ["Friedberg", "fjms", "FGP", "Genazim", "see genizah.org"])
def test_forbidden_terms_anywhere(text: str) -> None:
    """Every forbidden name is caught in nested strings and in keys.

    :param text: Forbidden text.
    """
    assert b.find_forbidden_terms({"a": [{"b": text}]})
    assert b.find_forbidden_terms({text: 1})


def test_validate_machine_reads() -> None:
    """Zero-agreed reads and "checked" wording are rejected."""
    good = b.build_fragment(seed_row(ai_read=linkable_read(6)), set(), {})
    assert b.validate_machine_reads([good]) == []
    zero = copy.deepcopy(good)
    zero["machine_read"]["n_agreed"] = 0
    assert b.validate_machine_reads([zero])
    claim = copy.deepcopy(good)
    claim["machine_read"]["note"] = "Lines confirmed by a second reader."
    assert b.validate_machine_reads([claim])
    claim["machine_read"]["note"] = None
    claim["machine_read"]["label"] = "Checked machine reading"
    assert b.validate_machine_reads([claim])


def test_validate_card_sources() -> None:
    """Cards need a card_line and a human source; the distinct set is reported."""
    ok = b.build_fragment(seed_row(), set(), {})
    scholar = b.build_fragment(seed_row(card_source_public="Gil, Palestine (1983)"), set(), {})
    problems, distinct = b.validate_card_sources([ok, scholar])
    assert problems == [] and distinct == ["Gil, Palestine (1983)", "KTIV (National Library of Israel)"]
    empty = b.build_fragment(seed_row(card_line=" "), set(), {})
    machine = b.build_fragment(seed_row(card_source_public="machine reading"), set(), {})
    missing = b.build_fragment(seed_row(card_source_public=None), set(), {})
    problems, _ = b.validate_card_sources([empty, machine, missing])
    assert len(problems) == 3


def test_is_human_source() -> None:
    """Fixed labels and named scholars pass; machine or empty labels fail."""
    assert b.is_human_source("catalogue note") and b.is_human_source("Stern 2019")
    assert not b.is_human_source("AI transcription") and not b.is_human_source("")


def test_build_output_and_reference_checks() -> None:
    """The minimal seed builds cleanly; broken references and single-member clusters are caught."""
    seed = minimal_seed()
    output, report = b.build_output(seed, "2026-09-25T00:00:00+00:00")
    assert [cluster["id"] for cluster in output["clusters"]] == ["pair"]
    assert output["clusters"][0]["members"] == ["Cambridge_CUL_T_S_X_1", "Cambridge_CUL_T_S_X_2"]
    assert output["fragments"][0]["clusters"] == ["pair"]
    assert output["themes"][0]["fragment_ids"] == ["Cambridge_CUL_T_S_X_1", "Cambridge_CUL_T_S_X_2"]
    assert report["counts"]["excluded"] == 1
    validation = b.run_validations(output, seed, b.encode_output(output), None)
    assert validation["errors"] == []
    broken = copy.deepcopy(output)
    broken["themes"][0]["refs"].append("nope")
    broken["fragments"][0]["clusters"].append("solo")
    broken["fragments"][1]["festival_day"] = {"day": "d99", "basis": "named in the source"}
    problems = b.validate_references(broken)
    assert len(problems) == 3


def test_validate_hedges() -> None:
    """Themes must keep intro, hedges and contested positions verbatim."""
    seed = minimal_seed()
    output, _ = b.build_output(seed, "t")
    assert b.validate_hedges(output, seed["themes"]) == []
    output["themes"][0]["hedges"] = []
    output["themes"][0]["intro"] = "So."
    assert len(b.validate_hedges(output, seed["themes"])) == 3


def test_validate_size() -> None:
    """Output over the limit fails."""
    assert b.validate_size(b"x" * 10, limit=10) == []
    assert b.validate_size(b"x" * 11, limit=10)


def test_check_images_and_validate_images() -> None:
    """Each URL is checked once; any non-200 (or failed request) is a failure."""
    calls = []
    statuses = {"u1": 200, "u2": 404, "u3": None}

    def fake_status(url: str) -> Any:
        """Return a canned status.

        :param url: URL.
        :return: Status.
        """
        calls.append(url)
        return statuses[url]

    results = b.check_images(["u1", "u2", "u1", "u3"], fake_status, pause_s=0, sleep=lambda _s: None)
    assert calls == ["u1", "u2", "u3"]
    assert b.validate_images(results) == ["image u2 returned 404", "image u3 returned None"]


# --------------------------------------------------------------------------- thumbnails


def test_make_thumbnail_resizes_to_320_jpeg() -> None:
    """A large RGBA image becomes a 320 px wide RGB JPEG with the same aspect ratio."""
    from PIL import Image

    source = io.BytesIO()
    Image.new("RGBA", (1000, 500), (200, 100, 50, 128)).save(source, format="PNG")
    thumb = Image.open(io.BytesIO(b.make_thumbnail(source.getvalue())))
    assert thumb.format == "JPEG" and thumb.mode == "RGB" and thumb.size == (320, 160)


def test_make_thumbnail_does_not_upscale() -> None:
    """Images narrower than 320 px keep their size."""
    from PIL import Image

    source = io.BytesIO()
    Image.new("L", (200, 300), 128).save(source, format="PNG")
    assert Image.open(io.BytesIO(b.make_thumbnail(source.getvalue()))).size == (200, 300)


# --------------------------------------------------------------------------- the real seed


@pytest.mark.skipif(not b.DEFAULT_SEED.exists(), reason="seed not present")
def test_real_seed_builds_and_validates() -> None:
    """The committed seed builds with no validation errors (images not checked)."""
    seed = json.loads(b.DEFAULT_SEED.read_text(encoding="utf-8"))
    output, report = b.build_output(seed, "t")
    validation = b.run_validations(output, seed, b.encode_output(output), None)
    assert validation["errors"] == []
    assert report["festival_placement"]["rejected_curated_entries"] == []
    assert all(frag["usable_now"] for frag in output["fragments"])


def test_best_read_item_skips_missing_images():
    """A read on a missing scan is skipped in favour of the best read on a live image."""
    from build_sukkot_data import best_read_item
    items = [
        {"image_index": 1, "image_url": "dead", "ai_read": {"n_agreed": 19, "n_lines": 20}},
        {"image_index": 3, "image_url": "live", "ai_read": {"n_agreed": 18, "n_lines": 19}},
        {"image_index": 0, "image_url": "live2", "ai_read": {"n_agreed": 0, "n_lines": 10}},
    ]
    assert best_read_item(items)["image_index"] == 1
    assert best_read_item(items, image_ok=lambda u: u != "dead")["image_index"] == 3
    assert best_read_item(items, image_ok=lambda u: False) is None

