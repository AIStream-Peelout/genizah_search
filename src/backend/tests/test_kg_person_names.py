"""Tests for the scholar name standard (``scripts/kg/person_names.py``)."""

import unicodedata

import pytest

from scripts.kg import person_names
from scripts.kg.person_names import (
    choose_display, identity_key, near_key, needs_review, review_reasons, spaceless_key, standard_name)


@pytest.mark.parametrize("raw, expected", [
    ("Hathaway, Jane", "Jane Hathaway"),
    ("Zinger, Oded", "Oded Zinger"),
    ("Rustow, Marina", "Marina Rustow"),
    ("Chapira, Bernard", "Bernard Chapira"),
    ("Cohen, Mark R.", "Mark R. Cohen"),
    ("  Jane   Hathaway. ", "Jane Hathaway"),
])
def test_surname_comma_given_is_flipped(raw: str, expected: str) -> None:
    """"Surname, Given" becomes "Given Surname"; whitespace and trailing punctuation are cleaned.

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected


@pytest.mark.parametrize("raw", ["Hathaway, Jane", "Zinger, Oded", "Rustow, Marina", "Chapira, Bernard"])
def test_flipped_and_given_first_share_a_key(raw: str) -> None:
    """The four eval scholars match their given-first twins exactly.

    :param raw: "Surname, Given" form.
    :returns: None
    :rtype: None
    """
    assert identity_key(raw) == identity_key(standard_name(raw))


@pytest.mark.parametrize("raw, expected", [
    ("Goitein, S. D.", "S. D. Goitein"),
    ("S.D. Goitein", "S. D. Goitein"),
    ("Richards, D.S.", "D. S. Richards"),
    ("Marx A", "A. Marx"),
    ("Marx A.", "A. Marx"),
    ("Diez Merino L", "L. Diez Merino"),
    ("Goldberg J L", "J. L. Goldberg"),
    ("Estara J Arrant", "Estara J. Arrant"),
    ("J", "J."),
])
def test_initials_are_dotted_and_put_first(raw: str, expected: str) -> None:
    """Initials are "X." separated by spaces; FJP "Surname I J" puts them first.

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("A.M.Haberman", "A. M. Haberman"),
    ("A.M.HABERMAN", "A. M. HABERMAN"),
    ("Haberman, A.M.", "A. M. Haberman"),
    ("J.-P. Sartre", "J.-P. Sartre"),
    ("St.John Philby", "St.John Philby"),
])
def test_initials_glued_to_the_surname_are_split(raw: str, expected: str) -> None:
    """Initials glued to the next word are split off like any other initials; other dotted tokens stay.

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected


def test_glued_initials_share_the_spaced_key() -> None:
    """"A.M.Haberman" and "A. M. Haberman" are one exact cluster (census X012).

    :returns: None
    :rtype: None
    """
    assert identity_key("A.M.Haberman") == identity_key("A. M. Haberman") == "a m haberman"


def test_names_are_composed_to_nfc() -> None:
    """A decomposed stored name (h + U+0323) standardises to the composed form people type.

    :returns: None
    :rtype: None
    """
    decomposed, composed = "Simh\u0323a Assaf", "Sim\u1e25a Assaf"
    assert standard_name(decomposed) == composed
    assert identity_key(decomposed) == identity_key(composed) == identity_key("Simha Assaf")
    display = choose_display([decomposed, "Simha Assaf", "Assaf, Sim\u1e25a"])
    assert display == composed and unicodedata.is_normalized("NFC", display)
    assert standard_name("\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd") == "\u05e9\u05dc\u05d5\u05dd"


def test_spaceless_key_pairs_names_that_differ_only_by_spaces() -> None:
    """Word spaces count in the identity key; the spaceless key finds such pairs for review.

    :returns: None
    :rtype: None
    """
    pairs = [("M. Ben-Sasson", "M. Ben Sasson"), ("זאב בן-חיים", "זאב בן חיים"), ('מ"ב לרנר', "מ. ב. לרנר"),
             ("Alejandro Díez-Macho", "Alejandro Diez Macho")]
    for a, b in pairs:
        assert identity_key(a) != identity_key(b)
        assert spaceless_key(a) == spaceless_key(b)
    assert spaceless_key("Mark R. Cohen") != spaceless_key("M. R. Cohen")
    assert spaceless_key("???") == ""


def test_goitein_forms_share_a_key() -> None:
    """"Goitein, S. D." and "S.D. Goitein" are one exact cluster.

    :returns: None
    :rtype: None
    """
    assert identity_key("Goitein, S. D.") == identity_key("S.D. Goitein") == "s d goitein"


@pytest.mark.parametrize("raw, expected", [
    ("van Straten J", "J. van Straten"),
    ("del Valle C", "C. del Valle"),
    ("el-Leithy, Tamer", "Tamer el-Leithy"),
    ("Ben-Sasson, Menahem", "Menahem Ben-Sasson"),
    ("Salmon b. Yeruḥam", "Salmon b. Yeruḥam"),
    ("Abraham ibn Ezra", "Abraham ibn Ezra"),
    ("Kanāʿana, Sharīf", "Sharīf Kanāʿana"),
    ("Isaac ben E", "Isaac ben E."),
    ("Salmon b. Y", "Salmon b. Y."),
])
def test_particles_and_marks_are_kept_as_written(raw: str, expected: str) -> None:
    """Particles, hyphens, diacritics and ʿ stay; a lower-case "b." is not an initial.

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("E. ROMERO", "E. ROMERO"),
    ("MARX A", "A. MARX"),
    ("AL-QALQASHANDI, AHMAD", "AHMAD AL-QALQASHANDI"),
    ("LARREA PALACÍN, A. DE", "A. DE LARREA PALACÍN"),
    ("J.M.", "J. M."),
    ("o'brien, john", "john o'brien"),
    ("McKenzie, Judith", "Judith McKenzie"),
    ("Tamer el-Leithy", "Tamer el-Leithy"),
])
def test_names_are_never_title_cased(raw: str, expected: str) -> None:
    """No name is title-cased: all-caps and all-lower names are only reordered (and go to review).

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected


@pytest.mark.parametrize("raw", ["E. ROMERO", "JANE HATHAWAY", "AL-QALQASHANDI, AHMAD", "LARREA PALACÍN, A. DE",
                                 "MARX A", "J.M."])
def test_all_caps_names_go_to_review(raw: str) -> None:
    """Every all-caps Latin name is held for review with the reason "all caps".

    :param raw: Stored name.
    :returns: None
    :rtype: None
    """
    assert "all caps" in review_reasons(raw)


@pytest.mark.parametrize("raw, expected", [
    ("Michel De Certeau", "Michel de Certeau"),
    ("De Certeau, Michel", "Michel de Certeau"),
    ("De Certeau M", "M. de Certeau"),
    ("Johannes Den Heijer", "Johannes den Heijer"),
    ("Den Heijer, Johannes", "Johannes den Heijer"),
    ("A. De Biberstein Kazimirski", "A. de Biberstein Kazimirski"),
    ("Kazimirski, A. De Biberstein", "A. de Biberstein Kazimirski"),
    ("N. R. M. De Lange", "N. R. M. de Lange"),
    ("Wout Jacques Van Bekkum", "Wout Jacques van Bekkum"),
    ("Abu Bakr Al-Razi", "Abu Bakr al-Razi"),
    ("Van Bekkum, Wout", "Wout van Bekkum"),
    ("Menahem Ben-Sasson", "Menahem Ben-Sasson"),
    ("Moshe Bar-Asher", "Moshe Bar-Asher"),
    ("Abraham Ibn Ezra", "Abraham Ibn Ezra"),
    ("Salmon Ben Yeruḥam", "Salmon Ben Yeruḥam"),
    ("Riccardo Di Segni", "Riccardo Di Segni"),
    ("Charles Le Quesne", "Charles Le Quesne"),
    ("Nadia Maria El-Cheikh", "Nadia Maria El-Cheikh"),
    ("Thanh Du", "Thanh Du"),
    ("John AL-AZMEH", "John AL-AZMEH"),
    ("Abu Bakr AL-Razi", "Abu Bakr al-Razi"),
])
def test_particles_after_the_first_token_are_lower_case(raw: str, expected: str) -> None:
    """de/den/van/al- after the first token are written in lower case; Ben-/Bar-, ben, ibn and the
    particles their bearers capitalise (Di, Le, El-) stay as written, as does a particle-like surname.

    :param raw: Stored name.
    :param expected: Standard form.
    :returns: None
    :rtype: None
    """
    assert standard_name(raw) == expected
    assert identity_key(raw) == identity_key(expected)


def test_hebrew_loses_nikud_only() -> None:
    """Hebrew names lose vowel points and accents, keep punctuation and are never reordered.

    :returns: None
    :rtype: None
    """
    assert standard_name("שְׁלֹמֹה דֹּב  גּוֹיטַיְן") == "שלמה דב גויטין"
    assert standard_name("גויטיין, ש.ד.") == "גויטיין, ש.ד."
    assert standard_name("בן־ציון דינור") == "בן־ציון דינור"
    assert identity_key('ד"צ בנעט') == identity_key("ד.צ. בנעט")


def test_lost_accent_is_never_fixed_but_matches_and_is_reviewed() -> None:
    """Schäfer, Schafer and Scha?fer share a key; "?" is kept and sent to review; Schäfer wins.

    :returns: None
    :rtype: None
    """
    names = ["Peter Schafer", "Peter Scha?fer", "Schäfer, Peter"]
    assert standard_name("Schäfer, Peter") == "Peter Schäfer"
    assert standard_name("Peter Scha?fer") == "Peter Scha?fer"
    assert len({identity_key(n) for n in names + ["Peter Schäfer"]}) == 1
    assert "?" in needs_review("Peter Scha?fer")
    assert needs_review("Peter Schafer") is None
    assert choose_display(names) == "Peter Schäfer"


def test_different_people_keep_different_keys() -> None:
    """Never merge on surname + first initial: Mark R. Cohen, Martin Cohen and M. R. Cohen differ.

    :returns: None
    :rtype: None
    """
    keys = {identity_key(n) for n in ("Mark R. Cohen", "Martin Cohen", "Cohen M R", "Mark Cohen")}
    assert len(keys) == 4
    assert identity_key("Cohen, Mark R.") == identity_key("Mark R. Cohen")


def test_several_commas_are_left_unchanged_and_reviewed() -> None:
    """A name with several commas is not reordered and needs review.

    :returns: None
    :rtype: None
    """
    raw = "al-Hawary, Rached, and Wiet"
    assert standard_name(raw) == raw
    assert "several commas" in needs_review(raw)


@pytest.mark.parametrize("a, b", [
    ("Ben-Zion Schereschewsky", "Benzion Schereschewsky"),
    ("E.Y. 'Kutscher", "E. Y. Kutscher"),
    ("Tova Beʾeri", "Tova Beeri"),
    ("‘Alī Bahjat", "'Ali Bahjat"),
])
def test_key_deletes_punctuation_and_accents(a: str, b: str) -> None:
    """Punctuation, apostrophe-like marks and accents are deleted without inserting a space.

    :param a: One spelling.
    :param b: Another spelling of the same name.
    :returns: None
    :rtype: None
    """
    assert identity_key(a) == identity_key(b)


def test_key_does_not_equate_hyphen_and_space() -> None:
    """Deleting a hyphen joins the parts, so "Díez-Macho" and "Diez Macho" stay apart.

    :returns: None
    :rtype: None
    """
    assert identity_key("Alejandro Díez-Macho") != identity_key("Alejandro Diez Macho")


@pytest.mark.parametrize("raw, reason", [
    ("Dr. Jane Smith", "title prefix"),
    ("Smith, Prof. John", "title prefix"),
    ("הרב משה פיינשטיין", "title prefix"),
    ("Goitein", "single token"),
    ("", "empty"),
    ("???", "empty"),
    ("Moshe Gil / משה גיל", "mixed scripts"),
])
def test_needs_review_reasons(raw: str, reason: str) -> None:
    """Titles, single tokens, empty names and mixed scripts are flagged.

    :param raw: Stored name.
    :param reason: Expected reason.
    :returns: None
    :rtype: None
    """
    assert reason in needs_review(raw)


@pytest.mark.parametrize("raw", ["Jane Hathaway", "Hathaway, Jane", "S.D. Goitein", "ש ד גויטיין"])
def test_clean_names_need_no_review(raw: str) -> None:
    """Ordinary names pass.

    :param raw: Stored name.
    :returns: None
    :rtype: None
    """
    assert needs_review(raw) is None


@pytest.mark.parametrize("raw", [
    "Hathaway, Jane", "Goitein, S. D.", "Marx A", "Diez Merino L", "LARREA PALACÍN, A. DE", "J.M.",
    "Isaac ben E", "al-Hawary, Rached, and Wiet", "שְׁלֹמֹה דֹּב גּוֹיטַיְן", "Peter Scha?fer", "o'brien, john",
    "van Straten J", "Richards, D.S.", "USULA ÇAKUTİNOLU KOKEMU", "A.M.Haberman", "Simh\u0323a Assaf",
    "Muh\u0323ammad b. Mu\u0304sa\u0304 al-Khu\u0304wa\u0304rizmı\u0304", "De Certeau M",
    "Kazimirski, A. De Biberstein", "Abu Bakr Al-Razi",
])
def test_standard_is_idempotent(raw: str) -> None:
    """Standardising twice changes nothing, and the key of the standard form is the same key.

    :param raw: Stored name.
    :returns: None
    :rtype: None
    """
    once = standard_name(raw)
    assert standard_name(once) == once
    assert identity_key(once) == identity_key(raw)


def test_initial_style_can_switch_to_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    """INITIAL_STYLE = "bare" writes initials without periods and keeps the same keys.

    :param monkeypatch: Pytest fixture.
    :returns: None
    :rtype: None
    """
    dotted_key = identity_key("Goitein, S. D.")
    monkeypatch.setattr(person_names, "INITIAL_STYLE", "bare")
    assert standard_name("Goitein, S. D.") == "S D Goitein"
    assert standard_name("Marx A.") == "A Marx"
    assert standard_name("S D Goitein") == "S D Goitein"
    assert identity_key("S.D. Goitein") == dotted_key


def test_choose_display_rules() -> None:
    """(a) no "?", (b) most common, (c) diacritics, (e) stored form, (f) lexicographic.

    :returns: None
    :rtype: None
    """
    assert choose_display(["Peter Scha?fer", "Peter Schafer"]) == "Peter Schafer"
    assert choose_display(["Tova Beeri", "Tova Beʾeri"]) == "Tova Beʾeri"
    assert choose_display(["Tijana Krstic", "Krstić, Tijana", "Tijana Krstić"]) == "Tijana Krstić"
    # (d): the flipped form is lexicographically first but no node is stored that way.
    assert choose_display(["Ze'evi, Dror", "Dror Ze’evi"]) == "Dror Ze’evi"
    # (e): both forms are stored; two members standardise to the curly-apostrophe form.
    assert choose_display(["Ze’evi, Dror", "Dror Ze’evi", "Dror Ze'evi"]) == "Dror Ze’evi"
    # (f): nothing else separates them.
    assert choose_display(["Dror Ze’evi", "Dror Ze'evi"]) == "Dror Ze'evi"
    with pytest.raises(ValueError):
        choose_display([])


@pytest.mark.parametrize("raw, reason", [
    ("the editor", "no upper-case letter"),
    ("present writer", "no upper-case letter"),
    ("footnote 100", "contains a digit"),
    ("Footnote 100", "contains a digit"),
    ("<Person Name>", "contains '<' or '>'"),
    ("D.S. and Roth Loewinger", "several authors"),
    ("Greenfield, Stone and Eshel", "several authors"),
    ("Smith & Jones", "several authors"),
    ("Lévi et Schwab", "several authors"),
    ("Müller und Schmidt", "several authors"),
    ("Smith; Jones", "several authors"),
    ("Freimann, Alfred (Abraham Hayyim)", "parentheses"),
    ("Wiet, M. G.", "French 'M.' before an initial"),
    ("Wiet, M.G.", "French 'M.' before an initial"),
    ("Yaqut, Ibn 'Abdallah al-Hamawi", "Arabic name with a comma"),
    ("al-Akṣarayi, 'Abd al-Rahman ibn Yūsuf", "Arabic name with a comma"),
    ("al-Qalqašandī, 'Abū al-'Abbas Aḥmad", "Arabic name with a comma"),
    ("Al-ʿUksh, Saʻīd ʻAbd al-Salām", "Arabic name with a comma"),
    ("Saʿadya, b. Yosef", "Arabic name with a comma"),
    ("Haytham, al-Hasan", "Arabic name with a comma"),
    ("Josep Pellicer I Bru", "conjunction 'I'/'Y' between names"),
    ("José Ortega Y Gasset", "conjunction 'I'/'Y' between names"),
    ("Haelst J V", "several trailing initials"),
    ("Olmo Lete G D", "several trailing initials"),
    ("Esteban F D", "several trailing initials"),
    ("J.M.", "all caps"),
    ("E. AHSTOR", "all caps"),
    ("MARX A", "all caps"),
    ("Wiet, M G", "French 'M.' before an initial"),
    ("A Mediterranean Society S.D Goitien", "initials in two separate runs"),
    ("נ. וקופפר. א. אלוני", "initials in two separate runs"),
    ("שירת בני מערבא י' יהלום ומ' סוקולוף", "several authors"),
    ("R. Gamliel the Elder", "phrase? (lower-case word or 'the')"),
    ("The Jews of Egypt Mann", "phrase? (lower-case word or 'the')"),
    ("Mosseri family", "phrase? (lower-case word or 'the')"),
    ("Scholarship on the Non-Standard Tiberian Tradition of Vocalisation", "phrase? (lower-case word or 'the')"),
    ("Isaac ben E", "truncated name? (initial after a particle)"),
    ("Salmon b. Y", "truncated name? (initial after a particle)"),
    ("אברמסון ש.", "Hebrew surname first"),
    ("שייבר א", "Hebrew surname first"),
    ("דר. ב. לוין", "title prefix"),
    ("פרופ. ראובן כהן", "title prefix"),
])
def test_doubtful_names_need_review(raw: str, reason: str) -> None:
    """Lower-case phrases, digits, <>, several authors, parentheses, French M., Arabic comma names,
    Catalan I/Y, several FJP initials, all-caps fragments, split runs of initials, phrases, cut-off
    names, Hebrew surname-first names and Hebrew titles with a period are never changed automatically.

    :param raw: Stored name.
    :param reason: Expected reason.
    :returns: None
    :rtype: None
    """
    assert reason in needs_review(raw)


@pytest.mark.parametrize("raw", [
    "Marx A", "Diez Merino L", "el-Leithy, Tamer", "Michel De Certeau", "N. R. M. De Lange", "Smith, M. Leo",
    "Bates, Michael L.", "Salmon b. Yeruḥam", "Estara J Arrant", "Kanāʿana, Sharīf",
    "al-Šayyāl, Gamāl al-Din", "Abu-Lughod, Janet", "John I. Smith", "בנימין ריצ'לר", "Mark R. Cohen",
    "Judah b. Barzilai of Barcelona", "José Ortega y Gasset", "Sylvie Denise García de la Calle",
    "‘Abdarraḥmān b. Naṣr ash-Shayzari", 'מנחם כ"ץ', "ש' ד' גויטיין", "Salmon b. Yeruḥam ha-Levi",
])
def test_refined_rules_leave_easy_names_alone(raw: str) -> None:
    """One FJP initial, particles, "of", a real given name after M., one run of initials and Hebrew pass.

    :param raw: Stored name.
    :returns: None
    :rtype: None
    """
    assert needs_review(raw) is None


def test_fjp_reorder_is_automatic_only_for_one_trailing_initial() -> None:
    """"Marx A" is reordered automatically; "Haelst J V" keeps its proposal but goes to review.

    :returns: None
    :rtype: None
    """
    assert (standard_name("Marx A"), needs_review("Marx A")) == ("A. Marx", None)
    assert standard_name("Haelst J V") == "J. V. Haelst"
    assert needs_review("Haelst J V") == "several trailing initials"
    assert person_names.fjp_initials("Udovitch, A. L.") == 0


def test_hebrew_geresh_inside_a_word_is_a_letter() -> None:
    """A geresh between two letters is kept by the key (צ' is not צ); an initial's marks are not.

    :returns: None
    :rtype: None
    """
    assert identity_key("בנימין ריצ'לר") != identity_key("בנימין ריצלר")
    assert identity_key("בנימין ריצ'לר") == identity_key("בנימין ריצ׳לר") == identity_key("בנימין ריצ’לר")
    assert identity_key("ג'פרי כאן") != identity_key("גפרי כאן")
    assert identity_key("ש' אסף") == identity_key("ש׳ אסף") == identity_key("ש. אסף") == identity_key("ש אסף")
    assert identity_key("ש' ד' גויטיין") == identity_key("ש ד גויטיין")


@pytest.mark.parametrize("names, expected", [
    (["ש. אסף", "ש' אסף"], "ש' אסף"),
    (["ח שירמן", "ח. שירמן"], "ח. שירמן"),
    (["ח שירמן", "ח. שירמן", "ח׳ שירמן"], "ח׳ שירמן"),
    (["י זוסמן", "י' זוסמן"], "י' זוסמן"),
    (["ש ד גויטיין", "ש' ד' גויטיין"], "ש' ד' גויטיין"),
    (["ד.צ. בנעט", 'ד"צ בנעט'], 'ד"צ בנעט'),
])
def test_hebrew_initials_prefer_geresh_then_period(names: list, expected: str) -> None:
    """Among equally frequent forms, Hebrew initials display with geresh/gershayim over a period over bare.

    :param names: Cluster members.
    :param expected: The display name.
    :returns: None
    :rtype: None
    """
    assert choose_display(names) == expected
    assert choose_display(names, [3] * len(names)) == expected


def test_choose_display_prefers_the_most_frequent_form() -> None:
    """The form with the most edges wins over diacritics, Hebrew initial style and the stored form;
    those (and a lost accent, never chosen) only break ties.

    :returns: None
    :rtype: None
    """
    assert choose_display(["Tijana Krstic", "Krstić, Tijana", "Tijana Krstić"], [9, 1, 1]) == "Tijana Krstic"
    assert choose_display(["ש. אסף", "ש' אסף"], [5, 1]) == "ש. אסף"
    assert choose_display(["ח שירמן", "ח. שירמן", "ח׳ שירמן"], [4, 2, 1]) == "ח שירמן"
    assert choose_display(["Tamer El-Leithy", "Tamer el-Leithy", "el-Leithy, Tamer"], [5, 1, 7]) == "Tamer el-Leithy"
    assert choose_display(["Ze'evi, Dror", "Dror Ze’evi"], [4, 1]) == "Dror Ze'evi"
    assert choose_display(["Peter Scha?fer", "Peter Schafer"], [9, 1]) == "Peter Schafer"


def test_choose_display_weights_forms_by_degree() -> None:
    """Rule (e) sums the members' degrees, so one well-connected node beats two sparse ones.

    :returns: None
    :rtype: None
    """
    names = ["Ze’evi, Dror", "Dror Ze’evi", "Dror Ze'evi"]
    assert choose_display(names) == "Dror Ze’evi"
    assert choose_display(names, [1, 1, 5]) == "Dror Ze'evi"
    assert choose_display(names, [3, 0, 2]) == "Dror Ze’evi"


@pytest.mark.parametrize("name, expected", [
    ("F. Rosenthal", True), ("Rosenthal F", True), ("A. De Biberstein Kazimirski", True), ("ש' אסף", True),
    ('ד"צ בנעט', True), ("ח שירמן", True), ("Franz Rosenthal", False), ("Estara J. Arrant", False),
    ("J. M.", False), ('מנחם כ"ץ', False), ("Goitein", False),
])
def test_initials_only(name: str, expected: bool) -> None:
    """Initials followed by a surname, with no given name, in Latin or Hebrew script.

    :param name: A name.
    :param expected: Whether it is initials + surname.
    :returns: None
    :rtype: None
    """
    assert person_names.initials_only(name) is expected


def test_near_key_pairs_space_and_geresh_variants() -> None:
    """Names differing only by spaces or by a geresh between letters share a near key, not an identity key.

    :returns: None
    :rtype: None
    """
    for a, b in [("בנימין ריצ'לר", "בנימין ריצלר"), ("M. Ben-Sasson", "M. Ben Sasson"), ("ג'פרי כאן", "גפרי כאן")]:
        assert identity_key(a) != identity_key(b)
        assert near_key(a) == near_key(b)
    assert near_key("Mark R. Cohen") != near_key("M. R. Cohen")


def test_every_unreliable_proposal_reason_exists() -> None:
    """Each reason in UNRELIABLE_PROPOSAL is one review_reasons really returns (no typo hides a reason).

    :returns: None
    :rtype: None
    """
    examples = ["the editor", "Footnote 100", "<Name>", "Smith & Jones", "Freimann, Alfred (Abraham)",
                "Josep Pellicer I Bru", "Haelst J V", "A Mediterranean Society S.D Goitien", "Mosseri family",
                "Isaac ben E", "אברמסון ש.", "Wiet, M. G.", "Yaqut, Ibn 'Abdallah al-Hamawi"]
    produced = {reason for raw in examples for reason in review_reasons(raw)}
    assert person_names.UNRELIABLE_PROPOSAL <= produced
    assert not {"contains '?' (lost accent)", "title prefix", "single token"} & person_names.UNRELIABLE_PROPOSAL


@pytest.mark.parametrize("raw, reason", [
    ("Blau and Hopkins", "several authors"),
    ("M? ?ngeles Gallego Garc?a", "contains '?' (lost accent)"),
])
def test_phrase_rule_adds_no_second_reason(raw: str, reason: str) -> None:
    """Author joiners and words with a lost letter ("?ngeles") are not also reported as phrase words.

    :param raw: Stored name.
    :param reason: The only expected reason.
    :returns: None
    :rtype: None
    """
    assert needs_review(raw) == reason
