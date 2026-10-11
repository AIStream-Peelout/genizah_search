"""Unit tests for the pure deterministic quote check (spec cases 1-16, 24-29, 31).

Node, smoothing, verifier, metrics and stream cases (17-23, 30) live with the
graph tests. Sources mirror ``build_verification_sources`` output: bibliography
sources carry ``quoteable_text``; graph sources have no such key.
"""

import ast
import inspect

import pytest

from src.backend import quote_check
from src.backend.quote_check import (
    build_haystack,
    check_quotations,
    delete_sentences_at,
    delete_sentences_with_gaps,
    extract_quotations,
    insert_gap_markers,
    normalize_for_quote_match,
    quote_found,
    remove_gap_markers,
    sentences_overlapping,
    smoothing_violation,
    split_sentences,
)


def _deleted_draft(draft: str, sources: list, question: str = "") -> str:
    """Run the check and delete every sentence holding an unverified item.

    :param draft: Drafted answer text.
    :param sources: Verification source dictionaries.
    :param question: The user's question.
    :returns: The draft after deletion.
    :rtype: str
    """
    result = check_quotations(draft, sources, question)
    return delete_sentences_at(draft, result.unverified_offsets())[0]


def test_quote_check_never_imports_lms_agentic_search() -> None:
    """quote_check is imported by lms, so it must not import lms (cycle)."""
    tree = ast.parse(inspect.getsource(quote_check))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any("lms_agentic_search" in str(module) for module in modules)


def test_case_01_india_book_is_a_term_not_a_quotation() -> None:
    """A two-word title in quotes is a term: unchecked, nothing deleted."""
    draft = 'Goitein called it the "India Book." He began in 1948.'
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters of Medieval Jewish Traders*, p. 7",
        "quoteable_text": "Goitein started to collect the documents in 1948.",
        "evidence_text": "Original page text (quoteable): Goitein started to collect the documents in 1948.",
    }]

    result = check_quotations(draft, sources, "Who is S.D. Goitein?")

    assert result.counts["quotations"] == 0
    assert result.counts["terms_skipped"] == 1
    assert result.unverified_spans == []
    assert delete_sentences_at(draft, result.unverified_offsets()) == (draft, [])
    assert split_sentences(draft) == ['Goitein called it the "India Book."', "He began in 1948."]


def test_case_02_punctuation_inside_quote_is_ignored() -> None:
    """A period inside the closing mark matches a source reading 'India Book,'."""
    draft = 'Goitein edited the "documents of the India Book." Later scholars continued.'
    sources = [{
        "source_number": 1,
        "citation": "Friedman, *India Traders*, p. 1",
        "quoteable_text": "He assembled the documents of the India Book, which he never finished.",
        "evidence_text": "Original page text (quoteable): He assembled the documents of the India Book, which he never finished.",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["quotations"] == 1
    assert result.counts["verified"] == 1
    assert result.counts["unverified"] == 0
    assert result.verified[0]["type"] == "quote"
    assert result.verified[0]["text"] == "documents of the India Book."
    assert result.verified[0]["supported"] is True
    assert result.verified[0]["source_number"] == 1
    assert draft[result.verified[0]["start"]:result.verified[0]["end"]] == '"documents of the India Book."'


def test_case_03_curly_and_straight_quotes_match_each_other() -> None:
    """Curly draft quotes match straight source quotes, and the reverse."""
    straight_source = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 2",
        "quoteable_text": 'He read the "letters of the India traders" closely.',
        "evidence_text": 'Original page text (quoteable): He read the "letters of the India traders" closely.',
    }]
    curly_source = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 2",
        "quoteable_text": "He read the “letters of the India traders” closely.",
        "evidence_text": "Original page text (quoteable): He read the “letters of the India traders” closely.",
    }]

    curly_draft = "Goitein studied the “letters of the India traders” for decades."
    straight_draft = 'Goitein studied the "letters of the India traders" for decades.'

    assert check_quotations(curly_draft, straight_source, "").counts["verified"] == 1
    assert check_quotations(straight_draft, curly_source, "").counts["verified"] == 1


def test_case_04_nikud_and_latin_diacritics_are_ignored() -> None:
    """Pointed and unpointed Hebrew match both ways; Yijū matches Yiju."""
    pointed = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
    unpointed = "בראשית ברא אלהים"

    def source(text: str) -> list:
        """Build a one-source list around ``text``.

        :param text: Quoteable page text.
        :returns: Verification sources.
        :rtype: list
        """
        return [{
            "source_number": 1,
            "citation": "Fleischer, *Piyyut*, p. 4",
            "quoteable_text": f"{text} את השמים ואת הארץ",
            "evidence_text": f"Original page text (quoteable): {text} את השמים ואת הארץ",
        }]

    assert check_quotations(f'הפסוק "{pointed}" פותח את הספר.', source(unpointed), "").counts["verified"] == 1
    assert check_quotations(f'הפסוק "{unpointed}" פותח את הספר.', source(pointed), "").counts["verified"] == 1

    latin_source = [{
        "source_number": 1,
        "citation": "Goitein, *India Traders*, p. 52",
        "quoteable_text": "The merchant Abraham Ben Yiju of Mangalore traded in bronze.",
        "evidence_text": "Original page text (quoteable): The merchant Abraham Ben Yiju of Mangalore traded in bronze.",
    }]
    assert normalize_for_quote_match("Yijū") == normalize_for_quote_match("Yiju") == "yiju"
    assert check_quotations('He means "Abraham Ben Yijū of Mangalore" here.', latin_source, "").counts["verified"] == 1


def test_case_05_hebrew_acronyms_are_not_quotations() -> None:
    """Word-internal acronym marks (straight or gershayim) never pair."""
    assert extract_quotations('ראה כ"י אוקספורד, ע"ו ורמב"ם', min_words=1) == []
    assert extract_quotations("ראה כ״י אוקספורד, ע״ו ורמב״ם ור״ד", min_words=1) == []
    assert check_quotations('ראה כ"י אוקספורד, ע"ו ורמב"ם', [], "").counts["quotations"] == 0


def test_case_06_hebrew_prefix_quote_opens_a_quotation() -> None:
    """A mark after a prefix letter (ש"…") opens a quotation."""
    text = 'הוא כתב ש"נשתמרו בגניזה רבות מתשובותיהם" בספרו'

    quotations = extract_quotations(text)

    assert len(quotations) == 1
    start, end, body = quotations[0]
    assert body == "נשתמרו בגניזה רבות מתשובותיהם"
    assert text[start] == '"' and text[end - 1] == '"'


def test_case_07_quoted_work_title_is_exempt() -> None:
    """A quoted string found in a citation line is a title, not a quotation."""
    draft = 'In "Kol Nidre: Studies and Essays" he argues that the formula is old.'
    sources = [{
        "source_number": 1,
        "citation": "Gershon, *Kol Nidre: Studies and Essays*, p. 12",
        "quoteable_text": "The formula is attested early in the geonic period.",
        "evidence_text": "Original page text (quoteable): The formula is attested early in the geonic period.",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["exempt_title"] == 1
    assert result.counts["unverified"] == 0
    assert result.unverified_spans == []
    assert delete_sentences_at(draft, result.unverified_offsets())[0] == draft


def test_case_08_quote_echoed_from_question_is_exempt() -> None:
    """Repeating the user's own quoted words is not a fabricated quotation."""
    question = 'Why did Goitein call it "a Mediterranean society in the Genizah"?'
    draft = 'Goitein called it "a Mediterranean society in the Genizah" because the letters span the sea.'
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters of Medieval Jewish Traders*, p. 3",
        "quoteable_text": "The letters connect merchants across the sea.",
        "evidence_text": "Original page text (quoteable): The letters connect merchants across the sea.",
    }]

    result = check_quotations(draft, sources, question)

    assert result.counts["exempt_question"] == 1
    assert result.unverified_spans == []


def test_case_09_tisha_bav_marks_do_not_mis_pair() -> None:
    """A one-word quoted term consumes its marks, so prose never becomes a quote."""
    text = (
        'The word "הכפישה" is derived from Lamentations 3:16 ("ויגרס בחצץ שני"), '
        'and "the other one here" too.'
    )

    bodies = [body for _start, _end, body in extract_quotations(text)]

    assert bodies == ["ויגרס בחצץ שני", "the other one here"]
    assert not any("is derived from" in body for body in bodies)


def test_case_10_ellipsis_segments_must_match_in_order() -> None:
    """Ellipsis segments verify only when they occur in the source in order."""
    draft = 'Goitein\'s "seminal study of the documents … did more to illuminate the period" endures.'
    in_order = [{
        "source_number": 1,
        "citation": "Cohen, *Review*, p. 9",
        "quoteable_text": "His seminal study of the documents of the Geniza did more to illuminate the period than any other.",
        "evidence_text": "Original page text (quoteable): …",
    }]
    reversed_order = [{
        "source_number": 1,
        "citation": "Cohen, *Review*, p. 9",
        "quoteable_text": "Nothing did more to illuminate the period. Later came a seminal study of the documents.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    assert check_quotations(draft, in_order, "").counts["verified"] == 1
    assert check_quotations(draft, reversed_order, "").counts["unverified"] == 1


def test_case_10_quotation_opening_with_ellipsis_is_checked() -> None:
    """A quotation that starts with an ellipsis is extracted, not skipped."""
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 1",
        "quoteable_text": "In those years the court met every Monday in the synagogue.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    for ellipsis in ("…", "...", ". . ."):
        real = f'He wrote that "{ellipsis}the court met every Monday" there.'
        fake = f'He wrote that "{ellipsis}the court never met at all" there. Kept.'
        assert check_quotations(real, sources, "").counts["verified"] == 1
        assert _deleted_draft(fake, sources) == "Kept."


def test_case_11_ocr_line_break_hyphenation_is_absorbed() -> None:
    """Spaceless matching absorbs 'where- abouts' and 'Ara- bic' in the source."""
    draft = 'The editors traced "their whereabouts in the Arabic letters" carefully.'
    sources = [{
        "source_number": 1,
        "citation": "Gil, *Ishmael*, p. 33",
        "quoteable_text": "We traced their where- abouts in the Ara- bic letters of the period.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    assert check_quotations(draft, sources, "").counts["verified"] == 1


def test_case_12_fabricated_quote_sentence_deleted() -> None:
    """A quotation absent from every source deletes exactly its sentence."""
    draft = 'Goitein wrote that "there was no rabbinic court in Fustat at all." He later revised this.'
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *A Mediterranean Society*, vol. 2, p. 311",
        "quoteable_text": "The rabbinic court of Fustat met regularly in the synagogue.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    result = check_quotations(draft, sources, "")
    spans = result.unverified_offsets()
    new_draft, affected = delete_sentences_at(draft, spans)

    assert result.counts["unverified"] == 1
    assert result.unverified_spans[0][3] == "quote"
    assert len(sentences_overlapping(draft, spans)) == 1
    assert new_draft == "He later revised this."
    assert affected == [0]


def test_case_13_misquoted_date_range_is_deleted() -> None:
    """The 09-28 ketubbah misquote fails with no date and with a different date."""
    draft = (
        'It is "a rare surviving early medieval Iraqi ketubbah produced in Baghdad '
        'sometime between 889 and 988," as noted.'
    )
    no_date = [{
        "source_number": 1,
        "citation": "Friedman, *Jewish Marriage in Palestine*, p. 40",
        "quoteable_text": "This is a rare surviving early medieval Iraqi ketubbah produced in Baghdad, now in Cambridge.",
        "evidence_text": "Original page text (quoteable): …",
    }]
    other_date = [{
        "source_number": 1,
        "citation": "Friedman, *Jewish Marriage in Palestine*, p. 40",
        "quoteable_text": (
            "This is a rare surviving early medieval Iraqi ketubbah produced in Baghdad "
            "sometime between 889 and 998, now in Cambridge."
        ),
        "evidence_text": "Original page text (quoteable): …",
    }]

    assert _deleted_draft(draft, no_date) == ""
    assert _deleted_draft(draft, other_date) == ""


def test_case_14_shelf_marks_verified_deleted_or_exempt() -> None:
    """TS matches T-S; an unsourced mark deletes its sentence; a question echo is exempt."""
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 12",
        "quoteable_text": "A letter from Aden.",
        "evidence_text": "Original page text (quoteable): A letter from Aden.\nShelf marks cited in this source: T-S 12.243",
    }]

    verified = check_quotations("The letter TS 12.243 mentions him.", sources, "")
    assert verified.unverified_spans == []
    assert [(claim["type"], claim["text"], claim["source_number"]) for claim in verified.verified] == [
        ("shelf_mark", "TS 12.243", 1)
    ]

    missing_draft = "It is preserved as T-S 99.999. Other text stays."
    missing = check_quotations(missing_draft, sources, "")
    assert missing.unverified_spans == [(19, 29, "T-S 99.999", "shelf_mark")]
    assert missing.counts["shelf_marks_unverified"] == 1
    assert delete_sentences_at(missing_draft, missing.unverified_offsets())[0] == "Other text stays."

    echoed = check_quotations("T-S 8J22.22 is a letter.", sources, "What is in T-S 8J22.22?")
    assert echoed.unverified_spans == []
    assert echoed.verified == []


def test_case_15_hebrew_spelling_variant_verified_by_vowel_letter_fold() -> None:
    """והגהתי vs the source's והגהותי verifies: ו/י inside Hebrew words are folded.

    Isaac's decision on open item §9.2 (2026-10-11): plene/defective spelling
    is a "small thing off", not a fabrication.
    """
    draft = 'הוא כתב "משובשת בחשבונותיה בספר ההוא והגהתי". המשך הדברים נשאר.'
    sources = [{
        "source_number": 1,
        "citation": "Assaf, *Teshuvot ha-Geonim*, p. 5",
        "quoteable_text": "הנוסחה משובשת בחשבונותיה בספר ההוא והגהותי אותה",
        "evidence_text": "Original page text (quoteable): …",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["verified"] == 1
    assert result.unverified_spans == []
    assert build_haystack("והגהותי") == build_haystack("והגהתי")
    # The bracket rule is unaffected: normalisation itself does not fold.
    assert normalize_for_quote_match("במעשי[ו]") == "במעשיו"


def test_case_15_fold_keeps_word_initial_letter_and_other_letters() -> None:
    """The fold drops only non-initial ו/י; any other changed letter still fails."""
    # Word-initial ו ("and") survives the fold, so it still has to match.
    assert build_haystack("ויאמר") != build_haystack("יאמר")
    # A changed consonant is a different word.
    assert build_haystack("כתב") != build_haystack("כתם")
    hay = build_haystack("ויאמר משה אל העם זכור את היום הזה")
    assert quote_found("ויאמר משה אל העם", hay)
    assert quote_found("ויאמר מושה אל העם", hay)
    assert not quote_found("ויאמר משה אל הכהנים", hay)


def test_case_15_fabricated_hebrew_quote_still_deleted_under_fold() -> None:
    """An invented Hebrew quotation (like replay E2d) is still caught with the fold on."""
    draft = (
        'הכותב מוסיף "אין לסלוח על כך יתלוצצו ויתלעסו הרוסים". '
        "הדברים נכתבו בפוסטאט."
    )
    sources = [{
        "source_number": 1,
        "citation": "Assaf, *Teshuvot ha-Geonim*, p. 9",
        "quoteable_text": "אין לסלוח על כך שהשאלה לא נשלחה בזמן אל ישיבת סורא",
        "evidence_text": "Original page text (quoteable): …",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["unverified"] == 1
    assert _deleted_draft(draft, sources) == "הדברים נכתבו בפוסטאט."


def test_case_16_delete_sentences_keeps_markers_and_drops_empty_structure() -> None:
    """Bullets keep their marker; an emptied heading and paragraph disappear."""
    draft = (
        "## Background\n\n"
        '- Goitein wrote "there was no court at all" in 1950. He moved to Philadelphia.\n'
        '- Only "an invented quotation here".\n\n'
        "## Fabricated section\n\n"
        'The letter says "nothing of the kind exists here".\n\n\n\n'
        "## Conclusion\n\n"
        "Final paragraph stays."
    )
    spans = [(start, end) for start, end, _body in extract_quotations(draft)]

    new_draft, affected = delete_sentences_at(draft, spans)

    assert new_draft == (
        "## Background\n\n"
        "- He moved to Philadelphia.\n\n"
        "## Conclusion\n\n"
        "Final paragraph stays."
    )
    assert affected == [1]

    # A heading that never had a body of its own is structure, not debris.
    nested = '## Parent\n### Child\nKept sentence. Dropped "made up quote words".'
    nested_spans = [(start, end) for start, end, _body in extract_quotations(nested)]
    assert delete_sentences_at(nested, nested_spans) == ("## Parent\n### Child\nKept sentence.", [0])


def test_case_24_name_and_date_abbreviations_do_not_leave_remnants() -> None:
    """'b.' (ben) and 'd.' (died) do not split, so the whole sentence goes."""
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *A Mediterranean Society*, vol. 2, p. 5",
        "quoteable_text": "Nahray b. Nissim travelled between Fustat and Alexandria.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    nahray = 'Nahray b. Nissim wrote that "there was no court in Fustat at all." He travelled widely.'
    goitein = 'Goitein (d. 1985) wrote that "the court never met on Mondays". His work continues.'

    assert _deleted_draft(nahray, sources) == "He travelled widely."
    assert _deleted_draft(goitein, sources) == "His work continues."


def test_case_25_repeated_unverifiable_shelf_mark_deletes_every_sentence() -> None:
    """Every occurrence of an unverifiable mark is a span, not only the first."""
    draft = "The letter T-S 99.999 is early. A safe sentence. Later T-S 99.999. was copied."
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 12",
        "quoteable_text": "A letter from Aden.",
        "evidence_text": "Original page text (quoteable): A letter from Aden.",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["shelf_marks_unverified"] == 2
    assert [kind for _start, _end, _text, kind in result.unverified_spans] == ["shelf_mark", "shelf_mark"]
    assert delete_sentences_at(draft, result.unverified_offsets())[0] == "A safe sentence."


def test_case_26_editorial_interpolation_is_accepted() -> None:
    """A draft-side [word] is a gap; a source-side [word] keeps its content."""
    body = "the [Geniza] documents of the eleventh century show"
    draft = f'Goitein argued that "{body}" a busy trade.'

    plain_source = [{
        "source_number": 1,
        "citation": "Goitein, *Studies*, p. 8",
        "quoteable_text": "As the documents of the eleventh century show, trade flourished.",
        "evidence_text": "Original page text (quoteable): …",
    }]
    bracketed_source = [{
        "source_number": 1,
        "citation": "Goitein, *Studies*, p. 8",
        "quoteable_text": "As the [Geniza] documents of the eleventh century show, trade flourished.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    assert check_quotations(draft, plain_source, "").counts["verified"] == 1
    assert check_quotations(draft, bracketed_source, "").counts["verified"] == 1
    assert normalize_for_quote_match("במעשי[ו]") == "במעשיו"


def test_case_27_trivial_ellipsis_segment_proves_nothing() -> None:
    """A one-word segment is ignored, so the substantive clause must match."""
    draft = 'He wrote "the … there was no court in Fustat" in a letter.'
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters*, p. 1",
        "quoteable_text": "the merchants of Fustat sent the letter",
        "evidence_text": "Original page text (quoteable): …",
    }]

    assert check_quotations(draft, sources, "").counts["unverified"] == 1
    # Fail closed: with no segment of two or more words there is nothing to verify.
    assert not quote_found("the … of … and", build_haystack("the cat of the house and"))


def test_case_28_generated_summary_quote_is_not_a_title() -> None:
    """Only graph-source evidence joins the title haystack, never catalog summaries."""
    draft = 'The trip marked "a decisive turning point in Fatimid trade" for him.'
    bibliography_only = [{
        "source_number": 1,
        "citation": "Gil, *Jews in Islamic Countries*, p. 3",
        "quoteable_text": "",
        "evidence_text": (
            "Generated catalog summary (orientation only; never quote): "
            "The chapter treats a decisive turning point in Fatimid trade."
        ),
    }]

    result = check_quotations(draft, bibliography_only, "")
    assert result.counts["unverified"] == 1
    assert result.counts["exempt_title"] == 0

    with_graph = bibliography_only + [{
        "source_number": 2,
        "citation": "Neo4j knowledge graph record for Moshe Gil",
        "evidence_text": "Publication: a decisive turning point in Fatimid trade (1992)",
    }]
    assert check_quotations(draft, with_graph, "").counts["exempt_title"] == 1


def test_case_29_verified_quote_in_deleted_sentence_leaves_verified_list() -> None:
    """Re-checking the post-deletion draft drops a verified quote whose sentence went."""
    draft = (
        'He wrote "the court met every Monday" and also "the judges were paid by the community" '
        "in one letter. Another sentence stays."
    )
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *A Mediterranean Society*, vol. 2, p. 320",
        "quoteable_text": "In Fustat the court met every Monday and Thursday.",
        "evidence_text": "Original page text (quoteable): …",
    }]

    first = check_quotations(draft, sources, "")
    assert [claim["text"] for claim in first.verified] == ["the court met every Monday"]
    assert first.counts["unverified"] == 1

    new_draft, _affected = delete_sentences_at(draft, first.unverified_offsets())
    assert new_draft == "Another sentence stays."
    assert check_quotations(new_draft, sources, "").verified == []


def test_case_31_l46_footnote_numeral_omission_is_tolerated() -> None:
    """l46 Sukkot: a quote that leaves out the source's footnote '19' verifies.

    Isaac's decision on open item §9.3 (2026-10-11): an inline footnote
    marker is a "small thing off". Only the source side loses such numbers.
    """
    draft = (
        "Goitein and Friedman translate the letter: \"During Sukkot (the Feast of Tabernacles) "
        "a qinṭir arrived from the Land of the Zanj, which had in it new slaves.\" "
        "The writer then turns to prices."
    )
    quoteable = (
        "During Sukkot (the Feast of Tabernacles) a qințir (?) 19 arrived from the Land\n"
        "of the Zanj, which had in it new slaves. 20 I tried…"
    )
    sources = [{
        "source_number": 1,
        "citation": "Goitein, Friedman, *India Traders of the Middle Ages*, p. 457",
        "quoteable_text": quoteable,
        "evidence_text": f"Original page text (quoteable): {quoteable}",
    }]

    # ṭ (dot below) and ț (comma below) both decompose to t + a combining mark.
    assert normalize_for_quote_match("ṭ") == normalize_for_quote_match("ț") == "t"
    result = check_quotations(draft, sources, "")
    assert result.counts["verified"] == 1
    assert result.unverified_spans == []
    # Without the footnote tolerance the numeral would be the only blocker.
    body = extract_quotations(draft)[0][2]
    assert not quote_found(body, build_haystack(quoteable))


def test_case_31_footnote_tolerance_never_excuses_an_added_or_changed_number() -> None:
    """A quote that adds or changes a number still fails; only source-side markers drop."""
    quoteable = "The caravan reached Aden within 12 days of leaving Qus. 7 The goods were sold."
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *Letters of Medieval Jewish Traders*, p. 61",
        "quoteable_text": quoteable,
        "evidence_text": f"Original page text (quoteable): {quoteable}",
    }]

    def verified(draft: str) -> int:
        """Return how many quotations of a draft verify.

        :param draft: Draft holding one quotation.
        :returns: The verified count.
        :rtype: int
        """
        return check_quotations(draft, sources, "").counts["verified"]

    assert verified('He wrote "reached Aden within 12 days of leaving Qus."') == 1
    assert verified('He wrote "reached Aden within 13 days of leaving Qus."') == 0
    assert verified('He wrote "reached Aden within 12 days of leaving Qus in 1140."') == 0
    assert verified('He wrote "of leaving Qus. The goods were sold."') == 1
    # A dropped standalone two-digit number of the source is the accepted cost.
    assert verified('He wrote "reached Aden within days of leaving Qus."') == 1


def test_case_31_l55_ocr_hyphenation_quote_verified() -> None:
    """l55: 'for-\\nmulary' in the source matches 'formulary' in the draft."""
    draft = (
        'Friedman notes that the marriage formula is designated "the language or formulary '
        'of laymen" in Tannaitic sources.'
    )
    quoteable = (
        "…is designated in Tannaitic sources as leshon hedyot, the language or for-\n"
        "mulary of laymen.4 It primarily developed…"
    )
    sources = [{
        "source_number": 1,
        "citation": "Friedman, *Jewish Marriage in Palestine*, p. 2",
        "quoteable_text": quoteable,
        "evidence_text": f"Original page text (quoteable): {quoteable}",
    }]

    result = check_quotations(draft, sources, "")

    assert result.counts["verified"] == 1
    assert result.unverified_spans == []
    assert delete_sentences_at(draft, result.unverified_offsets()) == (draft, [])


def _gap_marked_paragraphs(draft: str) -> dict:
    """Delete every quoted sentence and mark the gaps in each affected paragraph.

    :param draft: Draft whose 3+-word quotations all count as unverified.
    :returns: Affected paragraph index mapped to its text with gap markers.
    :rtype: dict
    """
    spans = [(start, end) for start, end, _body in extract_quotations(draft)]
    new_draft, gaps = delete_sentences_with_gaps(draft, spans)
    paragraphs = new_draft.split("\n\n")
    return {index: insert_gap_markers(paragraphs[index], offsets) for index, offsets in gaps.items()}


def test_stage_c3_gap_offsets_mark_where_sentences_were_removed() -> None:
    """Gap offsets (for the C3 smoothing prompt) sit at each removal point."""
    draft = (
        'Intro stays. Goitein wrote "there was no court at all" then. Middle stays. '
        'He said "another made up line".\n'
        '- Bullet one. Bullet "fake quote words here".\n'
        '- "wholly fake bullet line" here.\n'
        "- Last bullet.\n\n"
        "Untouched paragraph.\n\n"
        'Final para "fake fake fake". Tail.'
    )

    assert _gap_marked_paragraphs(draft) == {
        0: "Intro stays. ⟦gap⟧ Middle stays. ⟦gap⟧\n- Bullet one. ⟦gap⟧\n- Last bullet.",
        2: "⟦gap⟧ Tail.",
    }
    # A whole line removed at the top of a paragraph is marked after the
    # next line's list marker, never before it.
    assert _gap_marked_paragraphs('- "wholly fake bullet line" here.\n- Kept bullet.') == {
        0: "- ⟦gap⟧ Kept bullet."
    }


def test_stage_c3_gap_companion_matches_delete_sentences_at() -> None:
    """delete_sentences_at keeps its (draft, affected) contract over the companion."""
    draft = (
        "## Background\n\n"
        '- Goitein wrote "there was no court at all" in 1950. He moved to Philadelphia.\n'
        '- Only "an invented quotation here".\n\n'
        "Final paragraph stays."
    )
    spans = [(start, end) for start, end, _body in extract_quotations(draft)]

    new_draft, gaps = delete_sentences_with_gaps(draft, spans)

    assert delete_sentences_at(draft, spans) == (new_draft, list(gaps))
    assert gaps == {1: [2, len("- He moved to Philadelphia.")]}
    assert delete_sentences_with_gaps("Nothing to delete here.", []) == ("Nothing to delete here.", {})
    assert "⟦" not in new_draft


# ----------------------------------------------------------------------------
# Case 17: the smoothing guard (pure; node-level call-health cases live with
# the graph tests)
# ----------------------------------------------------------------------------

# A sentence was removed between the two middle sentences, which may change;
# the first and last sentences are fixed.
_SMOOTH_FIXED_FIRST = "Goitein spent decades in Philadelphia."
_SMOOTH_FIXED_LAST = "The rest followed slowly."
_SMOOTH_BEFORE = "He read the letters closely."
_SMOOTH_AFTER = "He published the first volume later."
_SMOOTH_IN = f"{_SMOOTH_FIXED_FIRST} {_SMOOTH_BEFORE} ⟦gap⟧ {_SMOOTH_AFTER} {_SMOOTH_FIXED_LAST}"


def _smoothed(middle: str) -> str:
    """Build a rewrite of :data:`_SMOOTH_IN` whose gap region reads ``middle``.

    :param middle: Replacement for the two sentences around the gap.
    :returns: The rewritten paragraph with the fixed sentences intact.
    :rtype: str
    """
    return f"{_SMOOTH_FIXED_FIRST} {middle} {_SMOOTH_FIXED_LAST}"


def test_case_17_gap_markers_round_trip() -> None:
    """remove_gap_markers undoes insert_gap_markers, offsets included."""
    for paragraph, offsets in [
        ("Intro stays. Middle stays.\n- Bullet one.\n- Last bullet.", [13, 26, 40]),
        ("Tail.", [0]),
        ("- Kept bullet.", [2]),
        ("Lead only.", [10]),
    ]:
        assert remove_gap_markers(insert_gap_markers(paragraph, offsets)) == (paragraph, offsets)


def test_case_17_guard_accepts_however_at_the_gap() -> None:
    """Adding one connective at the gap passes (and re-casing "He" is fine)."""
    assert smoothing_violation(_SMOOTH_IN, _smoothed(f"{_SMOOTH_BEFORE} However, he published the first volume later.")) is None
    # Returning the paragraph unchanged, minus the marker, is also acceptable.
    assert smoothing_violation(_SMOOTH_IN, _smoothed(f"{_SMOOTH_BEFORE} {_SMOOTH_AFTER}")) is None


def test_case_17_guard_accepts_hebrew_connective_at_the_gap() -> None:
    """ובנוסף at a Hebrew gap passes; four new Hebrew words are the per-gap limit."""
    paragraph = "גויטיין קרא את המכתבים במשך עשרות שנים. ⟦gap⟧ הוא פרסם את ספרו על החברה הים תיכונית. הספר זכה להצלחה."
    smoothed = "גויטיין קרא את המכתבים במשך עשרות שנים. ובנוסף, הוא פרסם את ספרו על החברה הים תיכונית. הספר זכה להצלחה."
    four_new = "גויטיין קרא את המכתבים במשך עשרות שנים. ובנוסף לכך מאוחר יותר הוא פרסם את ספרו על החברה הים תיכונית. הספר זכה להצלחה."
    five_new = "גויטיין קרא את המכתבים במשך עשרות שנים. ובנוסף לכך מאוחר יותר ובהדרגה הוא פרסם את ספרו על החברה הים תיכונית. הספר זכה להצלחה."

    assert smoothing_violation(paragraph, smoothed) is None
    assert smoothing_violation(paragraph, four_new) is None
    assert smoothing_violation(paragraph, five_new) == "too_many_new_words"


def test_case_17_guard_accepts_the_c3a_k3_bridging_sentence() -> None:
    """C3a K3: the removed first sentence left "His transformation…" dangling;
    one bridging sentence of ordinary words is accepted although it is about
    81 chars longer (the superseded +80 rule would have rejected it)."""
    remainder = (
        "His transformation from a student of Islamic history into the foremost reader of the "
        "Geniza documents took two decades. In the preface to the India Book, Goitein recounts "
        "that he first went to the Geniza looking for Arabic texts. The volumes appeared only "
        "after his death."
    )
    bridge = "The preface to the India Book recounts how Goitein's career shifted dramatically."
    output = f"{bridge} {remainder}"

    assert 80 < len(output) - len(remainder) <= 160
    assert smoothing_violation(f"⟦gap⟧ {remainder}", output) is None
    # The smallest fix, naming the referent, passes too.
    assert smoothing_violation(f"⟦gap⟧ {remainder}", remainder.replace("His", "Goitein's", 1)) is None


@pytest.mark.parametrize("output, reason", [
    (_smoothed(f"{_SMOOTH_BEFORE} He published the first volume in 1952."), "new_number"),
    (_smoothed(f"{_SMOOTH_BEFORE} He published volume 2 later."), "new_number"),
    (_smoothed(f"{_SMOOTH_BEFORE} As Mann noted, he published the first volume later."), "new_proper_noun"),
    (_smoothed(f'{_SMOOTH_BEFORE} He published "the first volume" later.'), "new_quotation"),
    (_smoothed(f"{_SMOOTH_BEFORE} He (eventually) published the first volume later."), "new_parenthesis"),
    (_smoothed(f"{_SMOOTH_BEFORE} He published the first volume [later]."), "new_bracket"),
    (_smoothed(f"{_SMOOTH_BEFORE} **However**, he published the first volume later."), "new_bold"),
    (_smoothed(f"{_SMOOTH_BEFORE} He published `the first volume` later."), "new_backtick"),
    (_smoothed(f"{_SMOOTH_BEFORE} He published the first volume (doc:abc) later."), "new_parenthesis"),
    (_smoothed(f"{_SMOOTH_BEFORE} He published T-S 12.243 later."), "new_shelf_mark"),
    (_smoothed(f"{_SMOOTH_BEFORE} ⟦gap⟧ {_SMOOTH_AFTER}"), "marker_left"),
    (
        _smoothed(
            f"{_SMOOTH_BEFORE} {_SMOOTH_AFTER} It was received with enthusiasm by readers across "
            "many countries, who found in its pages an unusually vivid portrait of everyday life "
            "among the merchants and their families."
        ),
        "too_long",
    ),
    (f"{_SMOOTH_FIXED_FIRST} {_SMOOTH_FIXED_LAST}", "too_short"),
    (
        f"Goitein spent many decades in Philadelphia. {_SMOOTH_BEFORE} {_SMOOTH_AFTER} {_SMOOTH_FIXED_LAST}",
        "altered_sentence",
    ),
    (
        f"{_SMOOTH_FIXED_FIRST} {_SMOOTH_BEFORE} {_SMOOTH_AFTER} The rest followed quickly.",
        "altered_sentence",
    ),
    (_smoothed(f"{_SMOOTH_BEFORE} That took time. It was slow. {_SMOOTH_AFTER}"), "too_many_sentences"),
    (
        _smoothed(
            f"{_SMOOTH_BEFORE} Then, with great patience and remarkable care over many long years, "
            "he published the first volume later."
        ),
        "too_many_new_words",
    ),
], ids=[
    "c3a_in_1952", "new_number", "c3a_mann", "new_quotation", "new_parenthesis",
    "new_bracket", "new_bold", "new_backtick", "new_doc_link_parenthesis", "new_shelf_mark",
    "leftover_marker", "too_long", "too_short", "c3a_non_adjacent_first_altered",
    "c3a_non_adjacent_last_altered", "two_inserted_sentences", "too_many_new_words",
])
def test_case_17_guard_rejects(output: str, reason: str) -> None:
    """Each rewrite breaks one guard rule and is rejected with its reason."""
    assert smoothing_violation(_SMOOTH_IN, output) == reason


def test_case_17_guard_rejects_a_dropped_or_new_doc_link() -> None:
    """A doc: link may not be added even without other markup."""
    output = _smoothed(f"{_SMOOTH_BEFORE} He published the first volume, see doc:abc, later.")
    assert smoothing_violation(_SMOOTH_IN, output) == "new_doc_link"


def test_case_17_guard_rejects_a_removed_quotation() -> None:
    """Every input quotation (even one word) must survive verbatim."""
    paragraph = (
        f'{_SMOOTH_FIXED_FIRST} He called the merchants "the most mobile people of their age" in '
        f'one essay. ⟦gap⟧ {_SMOOTH_AFTER} {_SMOOTH_FIXED_LAST}'
    )
    dropped = _smoothed("He called the merchants highly mobile in one essay, and he published the first volume later.")
    restyled = _smoothed(
        "He called the merchants “the most mobile people of their age” in one essay. " + _SMOOTH_AFTER
    )
    kept = _smoothed(
        'He called the merchants "the most mobile people of their age" in one essay. However, '
        "he published the first volume later."
    )

    assert smoothing_violation(paragraph, dropped) == "removed_quotation"
    assert smoothing_violation(paragraph, restyled) == "removed_quotation"
    assert smoothing_violation(paragraph, kept) is None
    one_word = f'{_SMOOTH_FIXED_FIRST} He used the word "geniza" there. ⟦gap⟧ {_SMOOTH_AFTER} {_SMOOTH_FIXED_LAST}'
    assert smoothing_violation(one_word, _smoothed(f"He used the word geniza there. {_SMOOTH_AFTER}")) == "removed_quotation"


def test_case_17_guard_confines_edits_around_each_gap() -> None:
    """With two gaps, the sentence between them may change but a fixed sentence
    elsewhere may not; each gap allows one inserted sentence."""
    paragraph = (
        "One stays fixed here. Two may change. ⟦gap⟧ Three may change. Four stays fixed too. "
        "Five may change. ⟦gap⟧ Six may change."
    )
    edited = (
        "One stays fixed here. Two may change. Also, three may change. Four stays fixed too. "
        "Five may change. Still, six may change."
    )
    inserted = (
        "One stays fixed here. Two may change. That is all. Three may change. Four stays fixed too. "
        "Five may change. This too. Six may change."
    )
    altered = (
        "One stays fixed here. Two may change. Three may change. Four stays fixed as well. "
        "Five may change. Six may change."
    )

    assert smoothing_violation(paragraph, edited) is None
    assert smoothing_violation(paragraph, inserted) is None
    assert smoothing_violation(paragraph, altered) == "altered_sentence"


# ----------------------------------------------------------------------------
# Review fixes: matching tolerances, shelf-mark prose, splitter and pairing
# ----------------------------------------------------------------------------

_FUSTAT_COURT_SOURCES = [{
    "source_number": 1,
    "citation": "Goitein, *A Mediterranean Society*, vol. 2, p. 311",
    "quoteable_text": "The rabbinic court of Fustat met regularly in the synagogue.",
    "evidence_text": "Original page text (quoteable): …",
}]
_INVENTED = '"there was no court in Fustat at all"'


def _one_source(text: str) -> list:
    """Build a one-source list whose quoteable text is ``text``.

    :param text: Quoteable page text.
    :returns: Verification sources.
    :rtype: list
    """
    return [{
        "source_number": 1,
        "citation": "Gil, *A History of Palestine*, p. 1",
        "quoteable_text": text,
        "evidence_text": f"Original page text (quoteable): {text}",
    }]


@pytest.mark.parametrize("source_text, draft", [
    ("ידוע שהיהודים שישבו בפסטאט סחרו עם הודו",
     'גויטיין קובע ש"ישבו בפסטאט סחרו עם הודו" בימי הגאונים.'),
    ("בירושלים בזמן ההוא היו שתי קהילות",
     'הוא מספר ב"ירושלים בזמן ההוא היו שתי קהילות" בלבד.'),
    ("ויהודי מצרים ישבו בפסטאט ובאלכסנדריה",
     'הוא כותב "יהודי מצרים ישבו בפסטאט" בעיקר.'),
    ("ויש לציין כי הקהילה גדלה מאוד",
     'המחבר מעיר "יש לציין כי הקהילה גדלה" אז.'),
    ("הם שמחו מאוד וביום שובם מן הדרך ערכו סעודה",
     'נאמר "הם שמחו מאוד … יום שובם מן הדרך" במכתב.'),
], ids=["prefix_shin", "prefix_bet", "dropped_conjunction", "dropped_conjunction_yesh", "ellipsis_segment"])
def test_hebrew_quote_starting_inside_a_prefixed_source_word_verifies(source_text: str, draft: str) -> None:
    """A quote (or ellipsis segment) whose first word starts with ו/י verifies
    when the source has a prefix letter before it: the fold kept the draft's
    first letter but dropped the source's copy, which sits after the prefix."""
    result = check_quotations(draft, _one_source(source_text), "")

    assert result.counts["verified"] == 1
    assert result.unverified_spans == []


def test_hebrew_leading_letter_tolerance_still_catches_an_invented_quote() -> None:
    """Skipping a segment's leading ו/י never excuses any other changed word."""
    source = _one_source("ידוע שהיהודים שישבו בפסטאט סחרו עם הודו")

    result = check_quotations('גויטיין קובע ש"ישבו בפסטאט סחרו עם סין". זו עובדה.', source, "")

    assert result.counts["unverified"] == 1


@pytest.mark.parametrize("source_text, quote", [
    ("Saʿadya Gaon wrote the responsum to Fustat.", "Sa'adya Gaon wrote the responsum"),
    ("Sa‘adya Gaon wrote the responsum to Fustat.", "Saʿadya Gaon wrote the responsum"),
    ("Sa`adya Gaon wrote the responsum to Fustat.", "Saʿadya Gaon wrote the responsum"),
    ("A copy of the Qurʾān was bought there.", "copy of the Qur'an was bought"),
    ("A copy of the Qurʼan was bought there.", "copy of the Qur’an was bought"),
    ("Saʿadya Gaon wrote the responsum to Fustat.", "Saadya Gaon wrote the responsum"),
], ids=["ayin_vs_apostrophe", "curly_vs_ayin", "backtick_vs_ayin", "hamza_vs_apostrophe",
        "modifier_apostrophe_vs_curly", "ayin_vs_nothing"])
def test_ayin_and_hamza_transliteration_marks_count_as_punctuation(source_text: str, quote: str) -> None:
    """ʿ ʾ ʼ ʻ (spacing modifier letters) match ' ‘ ` or nothing at all."""
    result = check_quotations(f'He wrote "{quote}" once.', _one_source(source_text), "")

    assert result.counts["verified"] == 1


def test_ayin_tolerance_still_fails_a_changed_letter() -> None:
    """The ayin mark becomes a word break, not a wildcard letter."""
    assert normalize_for_quote_match("Saʿadya") == normalize_for_quote_match("Sa'adya") == "sa adya"
    result = check_quotations('He wrote "Sahadya Gaon wrote the responsum" once.',
                              _one_source("Saʿadya Gaon wrote the responsum."), "")
    assert result.counts["unverified"] == 1


def test_superscript_footnote_marker_of_the_source_may_be_omitted() -> None:
    """India¹⁹ is an inline footnote marker: NFKD alone would glue it into
    'india19', defeating the footnote tolerance."""
    source = _one_source("the merchants traded with India¹⁹ and Aden in those years")

    assert normalize_for_quote_match("India¹⁹ and") == "india 19 and"
    assert check_quotations('He wrote "the merchants traded with India and Aden" there.', source, "").counts[
        "verified"
    ] == 1
    assert check_quotations('He wrote "the merchants traded with India 20 and Aden" there.', source, "").counts[
        "unverified"
    ] == 1


@pytest.mark.parametrize("draft", [
    "Many fragments are now in the Mosseri collection at Cambridge.",
    "Cambridge bought the Mosseri Collection in 2006.",
    "Jacques Mosseri in 1909 acquired a large group of fragments.",
    "The Mosseri family kept the fragments in Cairo.",
    "The stones were covered in moss. The scribe noted this.",
    "Fragments from the L-G collection were catalogued later.",
    "Halper described these fragments (Halper 1924, p. 12).",
], ids=["mosseri_collection", "mosseri_collection_year", "mosseri_in", "mosseri_family",
        "moss_period", "lg_collection", "halper_year"])
def test_prose_that_matches_the_shelf_mark_pattern_is_never_deleted(draft: str) -> None:
    """A pattern match without an item number (or Halper plus a year) is
    prose, not a shelf mark, so nothing is deleted."""
    sources = [{
        "source_number": 1,
        "citation": "Halper, *Descriptive Catalogue of Genizah Fragments in Philadelphia*, p. 12",
        "quoteable_text": "The fragments were described in detail.",
        "evidence_text": "Original page text (quoteable): The fragments were described in detail.",
    }]
    full = f"{draft} Other text stays."

    result = check_quotations(full, sources, "")

    assert result.unverified_spans == []
    assert result.counts["shelf_marks_unverified"] == 0
    assert _deleted_draft(full, sources) == full


@pytest.mark.parametrize("mark", ["T-S 99.999", "Mosseri VII.200", "Halper 412", "ENA 2738.12"])
def test_unsourced_shelf_marks_with_an_item_number_are_still_deleted(mark: str) -> None:
    """The prose exemption never covers a real (numbered) shelf mark."""
    draft = f"It is preserved as {mark}. Other text stays."

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "Other text stays."


@pytest.mark.parametrize("lead", [
    "S. D. Goitein wrote that",
    "**S. D. Goitein**, p. 45:",
    "As M. Gil notes,",
    "The letter CUL Or. 1080 J 291 states",
    "The fragment T-S Ar. 30.278 says",
    "In ENA 2738, f. 12 the writer says",
], ids=["initials", "bold_initials", "single_initial", "cul_or", "ts_ar", "folio"])
def test_initials_and_shelf_mark_abbreviations_do_not_leave_remnants(lead: str) -> None:
    """The whole sentence goes: no 'S.', 'As M.' or 'CUL Or.' fragment stays
    (the shelf marks themselves are verified, only the quotation is not)."""
    sources = [dict(
        _FUSTAT_COURT_SOURCES[0],
        evidence_text=(
            "Original page text (quoteable): …\n"
            "Shelf marks cited in this source: CUL Or. 1080 J 291, T-S Ar. 30.278, ENA 2738"
        ),
    )]
    draft = f"{lead} {_INVENTED}. He travelled widely."

    assert check_quotations(draft, sources, "").counts["shelf_marks_unverified"] == 0
    assert _deleted_draft(draft, sources) == "He travelled widely."


def test_hebrew_initials_do_not_leave_remnants() -> None:
    """ש. ד. גויטיין is one name; deleting its sentence removes all of it."""
    draft = "פרופ' ש. ד. גויטיין כתב \"לא היה בית דין בפסטאט כלל\". זו עובדה ידועה."

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "זו עובדה ידועה."


def test_single_letter_after_an_apostrophe_still_ends_a_sentence() -> None:
    """Possessives and contractions are not initials."""
    assert split_sentences("This was Goitein's. Then he left.") == ["This was Goitein's.", "Then he left."]
    assert split_sentences("He didn’t. Then he left.") == ["He didn’t.", "Then he left."]


@pytest.mark.parametrize("draft, kept", [
    (f"**In short: the court met weekly.** Goitein wrote that {_INVENTED}. He travelled widely.",
     "**In short: the court met weekly.** He travelled widely."),
    (f"*Goitein argued this.* He wrote that {_INVENTED}. Later work followed.",
     "*Goitein argued this.* Later work followed."),
    (f"Goitein wrote that {_INVENTED}. ʿAbd al-Raḥmān travelled widely.", "ʿAbd al-Raḥmān travelled widely."),
    (f"Goitein wrote that {_INVENTED}. Ṭulayṭula was large.", "Ṭulayṭula was large."),
    (f"Goitein wrote that {_INVENTED}. ‘Old List’ came later.", "‘Old List’ came later."),
], ids=["closing_bold", "closing_italic", "ayin_capital", "dot_below_capital", "curly_single_opener"])
def test_neighbouring_sentence_survives_after_emphasis_or_transliterated_capital(draft: str, kept: str) -> None:
    """A boundary after closing emphasis, or before Ḥ/Ṭ/ʿ/‘, is still a
    boundary, so only the sentence with the invented quote is deleted."""
    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == kept


def test_sentence_split_before_bold_author_citation_keeps_the_verified_quote() -> None:
    """Spec C1: the splitter splits before **Author**, so deleting a fabricated
    sentence never takes the verified citation sentence after it."""
    sources = [{
        "source_number": 1,
        "citation": "Goitein, *A Mediterranean Society*, vol. 2, p. 7",
        "quoteable_text": "In Fustat the court met every Monday and Thursday in the synagogue.",
        "evidence_text": "Original page text (quoteable): …",
    }]
    citation = '**Goitein**, p. 7: "the court met every Monday and Thursday" in Fustat.'
    draft = f'He claimed that "the court never sat on any weekday at all". {citation}'

    assert _deleted_draft(draft, sources) == citation


@pytest.mark.parametrize("wrapper", ["**", "*", "_"], ids=["bold", "italic", "underscore_italic"])
def test_quotation_with_emphasis_inside_the_marks_is_checked(wrapper: str) -> None:
    """Emphasis inside the quote marks no longer hides a quotation."""
    sources = _one_source("The merchants of Fustat traded with India.")
    fabricated = f'He wrote "{wrapper}the merchants of Fustat never traded with anyone{wrapper}". Kept.'
    genuine = f'He wrote "{wrapper}the merchants of Fustat traded with India{wrapper}" once.'

    result = check_quotations(fabricated, sources, "")

    assert result.counts["quotations"] == 1
    assert result.counts["unverified"] == 1
    assert _deleted_draft(fabricated, sources) == "Kept."
    assert check_quotations(genuine, sources, "").counts["verified"] == 1
    assert check_quotations(f'He wrote {wrapper}"the merchants of Fustat traded with India"{wrapper} once.',
                            sources, "").counts["verified"] == 1


@pytest.mark.parametrize("draft", [
    'He used "Fustat as a name. Later he wrote "the court met every Monday" in a letter.',
    'He wrote "the court met every Monday” in Fustat. Later he noted "the judges were paid" by the community.',
], ids=["stray_straight_opener", "straight_opener_curly_closer"])
def test_fresh_opener_supersedes_an_unclosed_one(draft: str) -> None:
    """A stray unclosed mark cannot swallow later prose into a 'quotation'."""
    sources = _one_source("In Fustat the court met every Monday, and the judges were paid by the community.")

    result = check_quotations(draft, sources, "")

    assert result.unverified_spans == []
    assert result.counts["verified"] == 1


def test_quotation_pairing_resets_at_a_newline() -> None:
    """An unclosed mark on one line never pairs with a mark on the next."""
    draft = 'He spelled the town "Fustat in his notes.\nThe fragment is 12" long and torn. Kept sentence.'

    assert extract_quotations(draft) == []


def test_footnote_tolerance_never_drops_a_three_or_four_digit_number() -> None:
    """Only a standalone 1-2 digit source token may be omitted, never a year."""
    source = _one_source("The letter of Halfon b. Nethanel (1140) describes the voyage to Aden.")

    assert check_quotations('He wrote "Halfon b. Nethanel describes the voyage" there.', source, "").counts[
        "unverified"
    ] == 1


def test_hebrew_final_letter_fold_matches_ocr_medial_form() -> None:
    """An OCR'd source with a medial mem at word end matches the final form."""
    source = _one_source("נשתמרו בגניזה רבות מתשובותיהמ של הגאונים")

    assert check_quotations('הוא כתב "בגניזה רבות מתשובותיהם של הגאונים" בספרו.', source, "").counts[
        "verified"
    ] == 1


def test_heading_with_a_surviving_subsection_is_kept() -> None:
    """A parent heading goes only when every subsection under it vanished."""
    invented = 'Goitein wrote "there was no court in Fustat at all".'
    draft = f"## Courts\n### Fustat\n{invented}\n### Jerusalem\nThe court met weekly."
    blank_lines = f"## Courts\n\n### Fustat\n\n{invented}\n\n### Jerusalem\n\nThe court met weekly."
    all_gone = (
        f"## Courts\n### Fustat\n{invented}\n### Jerusalem\nHe said \"another invented line here\".\n"
        "## Next\nKept."
    )

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "## Courts\n### Jerusalem\nThe court met weekly."
    assert _deleted_draft(blank_lines, _FUSTAT_COURT_SOURCES) == (
        "## Courts\n\n### Jerusalem\n\nThe court met weekly."
    )
    assert _deleted_draft(all_gone, _FUSTAT_COURT_SOURCES) == "## Next\nKept."


@pytest.mark.parametrize("connective", [
    "Notably,", "Ultimately,", "Eventually,", "Importantly,", "Despite this,", "Over time,", "Among other things,",
])
def test_case_17_guard_accepts_common_sentence_initial_connectives(connective: str) -> None:
    """Ordinary discourse words at a gap are not new proper nouns."""
    output = _smoothed(f"{_SMOOTH_BEFORE} {connective} he published the first volume later.")

    assert smoothing_violation(_SMOOTH_IN, output) is None


@pytest.mark.parametrize("name", ["Mann", "Sicily"])
def test_case_17_guard_still_rejects_new_proper_nouns(name: str) -> None:
    """The connective allowlist never admits a new name."""
    output = _smoothed(f"{_SMOOTH_BEFORE} In {name}, he published the first volume later.")

    assert smoothing_violation(_SMOOTH_IN, output) == "new_proper_noun"


def test_case_17_guard_rejects_a_repeated_number_or_quotation() -> None:
    """Digits and quotations are compared as multisets: reusing a year to date
    a new event, or duplicating a quotation, is rejected."""
    dated = "In 1952 he moved to Philadelphia. ⟦gap⟧ He published the first volume later. The rest followed."
    quoted = (
        'He called them "the most mobile people" once. ⟦gap⟧ He published the first volume later. '
        "The rest followed."
    )

    assert smoothing_violation(
        dated, "In 1952 he moved to Philadelphia. In 1952 he published the first volume. The rest followed."
    ) == "new_number"
    assert smoothing_violation(
        quoted,
        'He called them "the most mobile people" once. He published "the most mobile people" later. '
        "The rest followed.",
    ) == "new_quotation"


def test_case_17_guard_matches_hebrew_words_behind_vav_and_he_prefixes() -> None:
    """Reusing paragraph words with a leading ו/ה adds no new Hebrew words."""
    paragraph = "גויטיין קרא את המכתבים במשך עשרות שנים. ⟦gap⟧ הוא פרסם את ספרו על החברה הים תיכונית. הספר זכה להצלחה."
    output = (
        "גויטיין קרא את המכתבים במשך עשרות שנים. והוא פרסם את ספרו והספר והמכתבים ובמשך והחברה והים "
        "תיכונית. הספר זכה להצלחה."
    )

    assert smoothing_violation(paragraph, output) is None


def test_case_17_guard_rejects_an_altered_final_or_glued_fixed_sentence() -> None:
    """The last fixed sentence must end the output unchanged, and a fixed
    sentence glued to inserted text is not intact."""
    paragraph = "He studied the letters. ⟦gap⟧ He wrote many books. He lived in Fustat."
    two_gaps = (
        "One stays fixed here. Two may change. ⟦gap⟧ Three may change. Four stays fixed too. "
        "Five may change. ⟦gap⟧ Six may change."
    )

    assert smoothing_violation(
        paragraph, "He studied the letters. He wrote many books. He lived in Aleppo."
    ) == "altered_sentence"
    assert smoothing_violation(
        two_gaps,
        "One stays fixed here. Two may change. Three may change.Four stays fixed too. "
        "Five may change. Six may change.",
    ) == "altered_sentence"


# ----------------------------------------------------------------------------
# Review fixes, round 2: acronyms, italic abbreviations, glued emphasis, ו/י
# ----------------------------------------------------------------------------

_FABRICATED_HEBREW = 'הוא קבע "לא היה בית דין בפסטאט כלל".'


@pytest.mark.parametrize("first", [
    'הפירוש הזה מופיע כבר אצל רש"י.',
    'כך כתב גויטיין ז"ל.',
    'הדבר נזכר בהלכות הרמב״ם.',
    'הוא עסק בתורה שבעל פה ובספרות חז"ל.',
    'הדבר מופיע כבר בתנ"ך.',
    'המכתב נכתב בשנת תתק"ב.',
    'במקור הערבי כתוב אלח׳ט.',
], ids=["rashi", "zal", "rambam_gershayim", "hazal", "tanakh", "year_numeral", "geresh"])
def test_hebrew_acronym_or_numeral_still_ends_its_sentence(first: str) -> None:
    """A single letter after ", ״ or ׳ ends an acronym or numeral, not an
    initial, so deleting the next sentence never takes this one with it."""
    draft = f"{first} {_FABRICATED_HEBREW} זו עובדה ידועה."

    assert split_sentences(draft)[0] == first
    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == f"{first} זו עובדה ידועה."


def test_latin_initials_still_do_not_end_a_sentence() -> None:
    """Control for the acronym rule: Latin initials stay inside the sentence."""
    assert split_sentences("As M. Cohen showed, S. D. Goitein read them. Then he left.") == [
        "As M. Cohen showed, S. D. Goitein read them.",
        "Then he left.",
    ]


@pytest.mark.parametrize("lead", [
    "Cohen *et al.* (2005) argued that",
    "As noted *ibid.* 45, Goitein wrote that",
    "Goitein (*cf.* Gil) wrote that",
    "Cohen **et al.** (2005) argued that",
], ids=["italic_et_al", "italic_ibid", "italic_cf", "bold_et_al"])
def test_italic_abbreviation_does_not_end_a_sentence(lead: str) -> None:
    """The period of an emphasised abbreviation is masked like a plain one,
    so the whole sentence goes and no 'Cohen *et al.*' remnant stays."""
    draft = f"{lead} {_INVENTED}. He travelled widely."

    assert len(split_sentences(draft)) == 2
    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "He travelled widely."


@pytest.mark.parametrize("draft, body", [
    ('He wrote "**quoted text here**" once.', "**quoted text here**"),
    ('**Goitein**"quoted text here" said', "quoted text here"),
    ('*Kitab*"quote text is here" and', "quote text is here"),
    ('**Goitein**"the documents of the India Book" said', "the documents of the India Book"),
], ids=["emphasis_inside", "glued_to_closing_bold", "glued_to_closing_italic", "india_book"])
def test_quotation_next_to_emphasis_on_either_side_is_extracted(draft: str, body: str) -> None:
    """Emphasis inside the marks, or closing emphasis glued to the opening
    mark, never hides a quotation."""
    assert [found for _start, _end, found in extract_quotations(draft)] == [body]


def test_fabricated_quote_glued_to_closing_bold_is_deleted() -> None:
    """A quotation right after **Author** is checked like any other."""
    draft = f"**Goitein**{_INVENTED} he wrote. Kept."

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "Kept."


@pytest.mark.parametrize("source_text", [
    "תכתוב את הספר הזה בעברית",
    "אכתוב את הספר הזה בעברית",
    "נכתוב את הספר הזה בעברית",
], ids=["tav", "alef", "nun"])
def test_leading_vav_or_yod_never_stands_for_a_non_prefix_letter(source_text: str) -> None:
    """A quote's leading ו/י may be skipped only behind a source prefix
    letter (ושהכלבמ), so a changed person or tense prefix still fails."""
    result = check_quotations('הוא אמר "יכתוב את הספר הזה" אז.', _one_source(source_text), "")

    assert result.counts["verified"] == 0
    assert result.counts["unverified"] == 1


# ----------------------------------------------------------------------------
# Round 3: splitter and matcher refinements from the round-2 recheck
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("lead", [
    "**No.**",
    "No.",
    "The colophon names the scribe as Hand B.",
    "**The colophon names the scribe as Hand B.**",
    "The letter dates from the reign of Baldwin I.",
    "The fragment sits in Box K.",
    "This is discussed in Appendix A.",
    "המכתב נדון בסעיף א.",
    'הפירוש מופיע אצל רש”י.',
], ids=["bold_no", "plain_no", "hand_b", "bold_hand_b", "baldwin_i", "box_k", "appendix_a", "hebrew_section", "curly_acronym"])
def test_sentence_ending_in_a_single_letter_or_no_keeps_its_own_boundary(lead: str) -> None:
    """A correct sentence ending in a lone letter, 'No.' or a curly-quoted
    acronym is not merged into the next sentence and deleted with it."""
    draft = f"{lead} Goitein wrote that {_INVENTED}. He travelled widely."

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == f"{lead} He travelled widely."


@pytest.mark.parametrize("draft", [
    "See no. 45 in the list. Next.",
    "In ENA 2738, f. 12 the writer says so. Next.",
    'The letter opens "R. Judah ha-Levi arrived in Alexandria" here. Next.',
    "Professor M. Cohen copied it. Next.",
    "Dr. M. Gil copied it. Next.",
    "The letter was edited by M. Gil in 1997. Next.",
], ids=["no_before_number", "folio", "quoted_initial", "title_initial", "abbreviated_title_initial", "lowercase_before_initial"])
def test_abbreviations_and_initials_before_a_name_do_not_split(draft: str) -> None:
    """'no. 45', 'f. 12' and name initials stay inside their sentence."""
    assert len(split_sentences(draft)) == 2


@pytest.mark.parametrize("quote,source_text", [
    ("יהיב ליה מאה זוזי", "ואודי דיהיב ליה מאה זוזי"),
    ("ישראל ויהודה כאחד", "בית דינא דישראל ויהודה כאחד"),
    ("יכון אלאמר כמא קלת", "ואנא אקול פיכון אלאמר כמא קלת"),
    ("וצל כתאבך אלינא אמס", "ואנא אקול פוצל כתאבך אלינא אמס"),
], ids=["aramaic_dalet", "aramaic_dalet_name", "judeo_arabic_fa", "judeo_arabic_fa_vav"])
def test_quote_behind_aramaic_dalet_or_judeo_arabic_fa_verifies(quote: str, source_text: str) -> None:
    """A faithful Aramaic or Judeo-Arabic quote that drops the attached ד/פ
    particle verifies even when it starts with ו/י."""
    result = check_quotations(f'הוא כתב "{quote}" שם.', _one_source(source_text), "")

    assert result.counts["verified"] == 1


def test_quotation_closed_against_opening_emphasis_is_extracted_and_checked() -> None:
    """'"…"**Goitein**' pairs, so a fabricated quote in that form is deleted."""
    assert [body for _s, _e, body in extract_quotations('"the documents of the India Book"**Goitein** said')] == [
        "the documents of the India Book"
    ]
    draft = f"{_INVENTED}**Goitein** said. Kept."

    assert _deleted_draft(draft, _FUSTAT_COURT_SOURCES) == "Kept."
