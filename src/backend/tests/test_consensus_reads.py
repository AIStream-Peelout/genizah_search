"""Tests for consensus reads: id remap, cleaning, visibility, dedupe, uncatalogued rule, search pass."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pytest
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import NotFoundError

from src.backend import ai_transcriptions as at
from src.backend import consensus_reads as cr

OLD = "Oxford_Bodleian_MS_heb_c_50_7"
NEW = "Oxford_Bodleian_Bodl_MS_heb_c_50_7"
DOC = "Manchester_JRL_G_66"
AGREED = "agreed"
UNCONF = "unconfirmed"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default surfacing thresholds and search switches for every test."""
    for name in (
        "AI_READ_MIN_AGREED_LINES", "AI_READ_MIN_AGREED_SHARE", "CONSENSUS_SEARCH_ENABLED", "AI_TRANSCRIPTIONS_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def _record(
    doc_id: str = DOC,
    image_index: int = 0,
    model: str = "v22i",
    lines: Sequence[Tuple[str, str]] = (("שלום רב לכם", AGREED), ("ברכה והצלחה", AGREED), ("עוד שורה", AGREED)),
    decoded_at: Optional[str] = "2026-10-02T11:21:37Z",
    published: bool = True,
    surfaced: bool = True,
    parsed: bool = True,
    sha: Optional[str] = None,
    n_agreed: Optional[int] = None,
) -> Dict[str, Any]:
    """Raw AI-index ``_source`` as stored (stored counters may lie on purpose)."""
    raw_lines = [{"index": i, "text": text, "status": status} for i, (text, status) in enumerate(lines)]
    return {
        "doc_id": doc_id,
        "source_index": "genizah_merged_v8",
        "image_index": image_index,
        "image_url": f"https://example.org/{doc_id}/{image_index}.jpg",
        "image_sha256": sha or f"sha-{doc_id}-{image_index}",
        "read_key": at.read_key(doc_id, image_index),
        "published": published,
        "surfaced": surfaced,
        "ai_read": {
            "vlm_model": model,
            "decoded_at": decoded_at,
            "parsed": parsed,
            "n_lines": len(raw_lines),
            "n_agreed": sum(1 for l in raw_lines if l["status"] == AGREED) if n_agreed is None else n_agreed,
            "lines": raw_lines,
        },
    }


# ---------------------------------------------------------------------------
# Ids and links
# ---------------------------------------------------------------------------


class TestOxfordRemap:
    def test_old_to_catalogue(self):
        assert cr.to_catalogue_doc_id(OLD) == NEW

    def test_catalogue_and_other_ids_unchanged(self):
        assert cr.to_catalogue_doc_id(NEW) == NEW
        assert cr.to_catalogue_doc_id(DOC) == DOC
        assert cr.to_catalogue_doc_id("Oxford_Bodleian_Ms_Heb_d_66_137") == "Oxford_Bodleian_Ms_Heb_d_66_137"
        assert cr.to_catalogue_doc_id(cr.OXFORD_OLD_PREFIX) == cr.OXFORD_OLD_PREFIX

    def test_catalogue_to_ai_forms(self):
        assert cr.to_ai_doc_ids(NEW) == [NEW, OLD]
        assert cr.to_ai_doc_ids(OLD) == [NEW, OLD]  # normalised first
        assert cr.to_ai_doc_ids(DOC) == [DOC]
        assert cr.to_ai_doc_ids("Oxford_Bodleian_Ms_Heb_d_66_137") == ["Oxford_Bodleian_Ms_Heb_d_66_137"]

    def test_read_url_encodes_and_pins_model(self):
        url = cr.read_url("Jerusalem_NLI_4°9463_8?x", 2, 5, "qwen3/v22i a50")
        assert url == "/read?doc=Jerusalem_NLI_4%C2%B09463_8%3Fx&image=2&line=5&model=qwen3%2Fv22i%20a50"
        assert cr.read_url(DOC, 0) == f"/read?doc={DOC}&image=0"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


class TestCleanAgreedLine:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Mixed Latin/Hebrew tokens are dropped whole, never glued ('למולא').
            ("לalmולא ואלה טוב", "ואלה טוב"),
            ("/dir עראקי ולא כתאב", "עראקי ולא כתאב"),
            ("אלא/dirך אצאל אללה", "אצאל אללה"),
            ("גויים ממא לג תוגבה דיני ישכם ילamma", "גויים ממא לג תוגבה דיני ישכם"),
            ("בו 棱 חתימות", "בו חתימות"),
            # Kept: geresh/gershayim/apostrophe, quotes, brackets, digits, Arabic, dot above.
            ("ג'רי ר\"ת נ״ע 12 [...] עולם.", "ג'רי ר\"ת נ״ע 12 [...] עולם."),
            ("אסرائيل אלאתנא עשר", "אסرائيل אלאתנא עשר"),
            ("תכצה לי באפ̇ט̇ל", "תכצה לי באפ̇ט̇ל"),
            ("כל־העם בְּרֵאשִׁית", "כל־העם בְּרֵאשִׁית"),
            # Invisible format characters are removed, not a reason to drop a word.
            ("כתב‏ו  ​שלום", "כתבו שלום"),
        ],
    )
    def test_cleaning(self, raw: str, expected: str):
        assert cr.clean_agreed_line(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "unfound_text", "[...] 12 .", "abc def ש", "سلام عليكم"])
    def test_rejected(self, raw: str):
        assert cr.clean_agreed_line(raw) is None

    def test_majority_junk_rejected_but_half_kept(self):
        assert cr.clean_agreed_line("tens tensי שנא") is None
        assert cr.clean_agreed_line("tensי שנא") == "שנא"


# ---------------------------------------------------------------------------
# Visibility and dedupe
# ---------------------------------------------------------------------------


class TestConsensusRead:
    def test_agreed_lines_only_in_index_order(self):
        rec = _record(lines=[("א" * 3, UNCONF), ("ברכה והצלחה", AGREED), ("שלום רב", AGREED), ("עוד", AGREED)])
        rec["ai_read"]["lines"].reverse()  # stored order must not matter
        read = cr.consensus_read(rec)
        assert [(l.index, l.text) for l in read.lines] == [(1, "ברכה והצלחה"), (2, "שלום רב"), (3, "עוד")]
        assert (read.n_agreed, read.n_lines) == (3, 4)
        assert read.doc_id == read.ai_doc_id == DOC and read.read_key == f"{DOC}__0"
        assert read.vlm_model == "v22i" and read.image_url.endswith("/0.jpg")

    def test_surfaced_flag_with_zero_agreed_lines_rejected(self):
        rec = _record(lines=[("שלום", UNCONF)] * 5, n_agreed=0)
        assert rec["surfaced"] is True
        assert cr.consensus_read(rec) is None

    def test_stored_counter_ignored(self):
        # Stored n_agreed claims 3 but the lines hold a single agreed line of 10: fails the rule.
        lines = [("שלום רב", AGREED)] + [("שורה", UNCONF)] * 9
        assert cr.consensus_read(_record(lines=lines, n_agreed=3)) is None
        # 3 of 20 agreed passes on the absolute count even though the share is 15%.
        lines = [("שלום רב", AGREED)] * 3 + [("שורה", UNCONF)] * 17
        assert cr.consensus_read(_record(lines=lines)).n_agreed == 3

    @pytest.mark.parametrize("override", [{"published": False}, {"surfaced": False}, {"parsed": False}])
    def test_flags(self, override: Dict[str, bool]):
        assert cr.consensus_read(_record(**override)) is None

    def test_junk_only_agreed_lines_rejected(self):
        rec = _record(lines=[("unfound_text", AGREED), ("abc def", AGREED), ("שלום", UNCONF)])
        assert cr.consensus_read(rec) is None

    def test_old_oxford_id_maps_to_catalogue(self):
        read = cr.consensus_read(_record(doc_id=OLD))
        assert read.doc_id == NEW and read.ai_doc_id == OLD and read.read_key == f"{OLD}__0"

    def test_thresholds_follow_env(self, monkeypatch: pytest.MonkeyPatch):
        lines = [("שלום רב", AGREED)] + [("שורה", UNCONF)] * 9
        monkeypatch.setenv("AI_READ_MIN_AGREED_LINES", "1")
        assert cr.consensus_read(_record(lines=lines)) is not None


class TestBestReads:
    def test_more_agreed_lines_beats_newer(self):
        older = _record(model="v21b", decoded_at="2026-09-01T00:00:00",
                        lines=[("שלום רב", AGREED)] * 4)
        newer = _record(model="v22i", decoded_at="2026-10-01T00:00:00Z",
                        lines=[("שלום רב", AGREED)] * 3)
        (best,) = cr.best_reads([newer, older])
        assert best.vlm_model == "v21b"

    def test_cleaned_count_decides(self):
        # Four stored agreed lines but two are junk: three clean lines win.
        junky = _record(model="v21b", lines=[("שלום רב", AGREED)] * 2 + [("abc", AGREED)] * 2)
        clean = _record(model="v20a", lines=[("שלום רב", AGREED)] * 3)
        (best,) = cr.best_reads([junky, clean])
        assert best.vlm_model == "v20a"

    def test_tie_newest_decoded_at_wins_across_formats(self):
        a = _record(model="v21b", decoded_at="2026-10-02T12:00:00")  # naive = UTC
        b = _record(model="v22i", decoded_at="2026-10-02T11:00:00Z")
        c = _record(model="v20a", decoded_at=None)
        (best,) = cr.best_reads([b, c, a])
        assert best.vlm_model == "v21b"

    def test_one_per_image_across_oxford_forms(self):
        # The same physical image under the old id (v6 image 0) and the v8 id (image 2).
        old = _record(doc_id=OLD, image_index=0, model="v21b", sha="same", lines=[("שלום רב", AGREED)] * 3)
        new = _record(doc_id=NEW, image_index=2, model="v22i", sha="same", lines=[("שלום רב", AGREED)] * 5)
        other = _record(doc_id=NEW, image_index=1, model="v22i")
        reads = cr.best_reads([old, new, other])
        assert [(r.ai_doc_id, r.image_index) for r in reads] == [(NEW, 1), (NEW, 2)]
        assert all(r.doc_id == NEW for r in reads)

    def test_sorted_by_image_index_and_invisible_dropped(self):
        recs = [_record(image_index=2), _record(image_index=0), _record(image_index=1, published=False)]
        assert [r.image_index for r in cr.best_reads(recs)] == [0, 2]

    def test_empty(self):
        assert cr.best_reads([]) == []


# ---------------------------------------------------------------------------
# Uncatalogued rule
# ---------------------------------------------------------------------------

RICH = "Letter from Shelomo b. Yehuda to Efrayim b. Shemarya. Dating: approximately 1030 CE."


class TestUncatalogued:
    @pytest.mark.parametrize(
        "description, thin",
        [
            (None, True), ("", True), ("   ", True), ("Piyyut", True), ("x" * 49, True), ("x" * 50, False),
            ("Newly treated and encapsulated, must be examined", True), ("קטע גניזה", True),
            ("A few words Blank verso", True), ("unidentified.", True),
            ("Newly treated and encapsulated, must be examined. A few words. Blank verso.", True),
            ("Newly treated and encapsulated, must be examined. Letter to Nahray b. Nissim.", False),
            (RICH, False),
        ],
    )
    def test_thin_description(self, description: Optional[str], thin: bool):
        assert cr.is_thin_description(description) is thin

    @pytest.mark.parametrize(
        "sources, description, has_tr, expected",
        [
            (["fjp"], RICH, True, True),             # no PGP/KTIV/Bodleian entry
            ("fjp", RICH, False, True),              # scalar keyword value
            (None, RICH, True, True),
            ([], RICH, True, True),
            (["fjp", "pgp"], "Piyyut", False, True),  # thin and no transcription
            (["fjp", "pgp"], "Piyyut", True, False),  # thin but transcribed
            (["fjp", "pgp"], RICH, False, False),     # rich description
            (["ktiv"], "", None, True),
            (["KTIV"], RICH, False, False),
            (["bodleian", "fjp"], RICH, False, False),
        ],
    )
    def test_truth_table(self, sources: Any, description: Optional[str], has_tr: Optional[bool], expected: bool):
        source = {"sources_present": sources, "description": description, "has_transcriptions": has_tr}
        assert cr.is_uncatalogued(source) is expected


# ---------------------------------------------------------------------------
# Terms and incipit
# ---------------------------------------------------------------------------


def _read(*lines: str, image_index: int = 0) -> cr.ConsensusRead:
    return cr.ConsensusRead(
        doc_id=DOC, ai_doc_id=DOC, image_index=image_index, read_key=f"{DOC}__{image_index}",
        vlm_model="v22i", n_agreed=len(lines), n_lines=len(lines), decoded_at=None, image_url=None,
        lines=[cr.ConsensusLine(index=i, text=t) for i, t in enumerate(lines)],
    )


class TestTerms:
    def test_stoplist_length_and_frequency_order(self):
        reads = [
            _read("אשר עלי אללה יוסף ברכה", "יוסף ברכה ישראל לי"),
            _read("יוסף נהוראי ישראל.", image_index=1),
        ]
        assert cr.consensus_terms(reads) == ["יוסף", "ברכה", "ישראל", "נהוראי"]

    def test_final_forms_and_points_fold_to_one_term(self):
        reads = [_read("שלום שלומ שָׁלוֹם", "דינר")]
        assert cr.consensus_terms(reads) == ["שלום", "דינר"]

    def test_stopword_written_with_medial_final_letter(self):
        assert cr.consensus_terms([_read("דלכ דלך כתאב")]) == ["כתאב"]

    def test_non_hebrew_and_mixed_tokens_excluded(self):
        reads = [_read("אסرائيل ותتزوג [...] 1234 ג'מיע כתב־יד")]
        assert cr.consensus_terms(reads) == ["ג'מיע", "כתב"]  # maqaf splits כתב־יד

    def test_limit(self):
        reads = [_read("אברהם יצחק יעקב משה אהרן")]
        assert cr.consensus_terms(reads, limit=2) == ["אברהם", "יצחק"]
        assert cr.consensus_terms(reads, limit=0) == []
        assert cr.consensus_terms([]) == []


class TestIncipit:
    def test_short_is_joined_without_ellipsis(self):
        reads = [_read("שלום רב", "לכם"), _read("ברכה", image_index=1)]
        assert cr.consensus_incipit(reads) == "שלום רב / לכם / ברכה"

    def test_cut_on_word_boundary(self):
        reads = [_read("אבגד הוזח טיכל מנסע", "פצקר שתאב")]
        out = cr.consensus_incipit(reads, max_chars=24)
        assert out == "אבגד הוזח טיכל מנסע…"
        assert len(out) <= 24

    def test_cut_never_leaves_separator(self):
        out = cr.consensus_incipit([_read("אבגד הוזח", "טיכל מנסע פצקר")], max_chars=13)
        assert out == "אבגד הוזח…"

    def test_next_char_space_keeps_whole_word(self):
        out = cr.consensus_incipit([_read("אבגד הוזח טיכל")], max_chars=10)
        assert out == "אבגד הוזח…"

    def test_single_long_word_hard_cut(self):
        out = cr.consensus_incipit([_read("א" * 30)], max_chars=10)
        assert out == "א" * 9 + "…"

    def test_empty(self):
        assert cr.consensus_incipit([]) == ""


# ---------------------------------------------------------------------------
# Service with a fake Elasticsearch
# ---------------------------------------------------------------------------


def _get(doc: Dict[str, Any], path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


class FakeES:
    """In-memory AI index (term/terms/ids filters, sort, search_after) plus catalogue ``mget``.

    A search with ``collapse`` (the agreed-lines search) returns ``search_response``.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        catalogue: Dict[str, Dict[str, Any]],
        search_response: Optional[Dict[str, Any]] = None,
        ai_exists: bool = True,
    ):
        self.records = [
            {"_id": at.record_id(r["doc_id"], r["image_index"], r["ai_read"]["vlm_model"]), "_source": r}
            for r in records
        ]
        self.catalogue = catalogue
        self.search_response = search_response
        self.ai_exists = ai_exists
        self.search_calls: List[Dict[str, Any]] = []
        self.mget_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _matches(hit: Dict[str, Any], clause: Dict[str, Any]) -> bool:
        (kind, body), = clause.items()
        if kind == "ids":
            return hit["_id"] in body["values"]
        (path, value), = body.items()
        actual = _get(hit["_source"], path)
        return actual in value if kind == "terms" else actual == value

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        self.search_calls.append(kwargs)
        if not self.ai_exists:
            raise NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        if "collapse" in kwargs:
            if isinstance(self.search_response, Exception):
                raise self.search_response
            return self.search_response
        filters = kwargs["query"]["bool"]["filter"]
        keys = [next(iter(s)) for s in kwargs.get("sort", [])]

        def sort_values(hit: Dict[str, Any]) -> List[Any]:
            return [_get(hit["_source"], k) for k in keys]

        hits = sorted((h for h in self.records if all(self._matches(h, f) for f in filters)), key=sort_values)
        if kwargs.get("search_after") is not None:
            hits = [h for h in hits if sort_values(h) > list(kwargs["search_after"])]
        hits = hits[: kwargs.get("size", 10)]
        return {"hits": {"hits": [{**h, "sort": sort_values(h)} for h in hits]}}

    def mget(self, **kwargs: Any) -> Dict[str, Any]:
        self.mget_calls.append(kwargs)
        return {"docs": [
            {"_id": i, "found": i in self.catalogue, "_source": self.catalogue.get(i, {})} for i in kwargs["ids"]
        ]}


def _cat(sources: Any, description: str = RICH, has_tr: bool = False) -> Dict[str, Any]:
    return {"sources_present": sources, "description": description, "has_transcriptions": has_tr}


FJP_DOC = "Cambridge_CUL_T_S_10J1_1"
THIN_DOC = "Cambridge_CUL_T_S_10J1_2"
RICH_DOC = "Cambridge_CUL_T_S_10J1_3"
BROKEN_DOC = "Cambridge_CUL_Or_1080_J115"
ORPHAN_DOC = "Oxford_Bodleian_Ms_Heb_d_66_137"


def _corpus() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    records = [
        _record(doc_id=FJP_DOC),
        _record(doc_id=FJP_DOC, model="v21b", lines=[("unfound_text", AGREED)] * 3),  # invisible checkpoint
        _record(doc_id=THIN_DOC, image_index=1),
        _record(doc_id=RICH_DOC),
        _record(doc_id=BROKEN_DOC, lines=[("שלום", UNCONF)] * 4, n_agreed=0),  # surfaced, 0 agreed
        _record(doc_id=OLD),
        _record(doc_id=ORPHAN_DOC),
        _record(doc_id="Cambridge_CUL_T_S_10J1_9", published=False),
    ]
    catalogue = {
        FJP_DOC: _cat(["fjp"]),
        THIN_DOC: _cat(["fjp", "pgp"], description="Piyyut"),
        RICH_DOC: _cat(["fjp", "pgp"]),
        BROKEN_DOC: _cat(["fjp"]),
        NEW: _cat(["bodleian", "fjp"], description="A few words Blank verso"),
        "Cambridge_CUL_T_S_10J1_9": _cat(["fjp"]),
    }
    return records, catalogue


def _service(es: FakeES, ttl: int = 3600) -> cr.ConsensusService:
    return cr.ConsensusService(es, catalogue_index="merged", ai_index="ai_v2", cache_ttl_seconds=ttl)


class TestReadsForDocument:
    def test_queries_both_oxford_forms_with_light_source(self):
        records = [
            _record(doc_id=OLD, image_index=0, model="v21b", sha="same"),
            _record(doc_id=NEW, image_index=2, model="v22i", sha="same", lines=[("שלום רב", AGREED)] * 4),
            _record(doc_id=NEW, image_index=1, model="v22i"),
        ]
        es = FakeES(records, {})
        reads = _service(es).reads_for_document(NEW)
        assert [(r.ai_doc_id, r.image_index) for r in reads] == [(NEW, 1), (NEW, 2)]
        (call,) = es.search_calls
        assert {"terms": {"doc_id": [NEW, OLD]}} in call["query"]["bool"]["filter"]
        assert {"term": {"published": True}} in call["query"]["bool"]["filter"]
        assert "ai_read.lines.htr_fragments" not in call["source_includes"]
        assert "ai_read.lines.bbox" not in call["source_includes"]
        assert "text_all" not in call["source_includes"]

    def test_pages_with_search_after(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cr, "PAGE_SIZE", 2)
        es = FakeES([_record(image_index=i) for i in range(5)], {})
        assert [r.image_index for r in _service(es).reads_for_document(DOC)] == [0, 1, 2, 3, 4]
        assert len(es.search_calls) == 3

    def test_missing_index(self):
        assert _service(FakeES([], {}, ai_exists=False)).reads_for_document(DOC) == []


class TestUncataloguedIds:
    def test_selection(self):
        records, catalogue = _corpus()
        assert _service(FakeES(records, catalogue)).uncatalogued_ai_doc_ids() == sorted([FJP_DOC, THIN_DOC, OLD])

    def test_catalogue_lookup_batched(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cr, "BATCH_SIZE", 2)
        records, catalogue = _corpus()
        es = FakeES(records, catalogue)
        _service(es).uncatalogued_ai_doc_ids()
        assert es.mget_calls and all(len(c["ids"]) <= 2 for c in es.mget_calls)
        assert all(c["source_includes"] == cr.CATALOGUE_SOURCE_FIELDS for c in es.mget_calls)
        assert all(c["index"] == "merged" for c in es.mget_calls)
        # Old Oxford id looked up under its catalogue form.
        assert NEW in {i for c in es.mget_calls for i in c["ids"]}

    def test_cached_until_ttl(self):
        records, catalogue = _corpus()
        es = FakeES(records, catalogue)
        service = _service(es)
        first = service.uncatalogued_ai_doc_ids()
        calls = (len(es.search_calls), len(es.mget_calls))
        assert service.uncatalogued_ai_doc_ids() == first
        assert (len(es.search_calls), len(es.mget_calls)) == calls
        expired = _service(es, ttl=0)
        expired.uncatalogued_ai_doc_ids()
        before = len(es.search_calls)
        expired.uncatalogued_ai_doc_ids()
        expired._refresh_thread.join(timeout=5)
        assert len(es.search_calls) > before

    def test_expired_set_is_served_while_one_background_refresh_runs(self):
        """The caller that finds the set expired never waits for the rebuild."""
        records, catalogue = _corpus()
        es = FakeES(records, catalogue)
        service = _service(es, ttl=0)
        first = service.uncatalogued_ai_doc_ids()
        stale = service._uncatalogued
        assert service._lock.acquire(blocking=False)  # pretend a refresh is running
        try:
            calls = len(es.search_calls)
            assert service.uncatalogued_ai_doc_ids() == first
            assert service._uncatalogued is stale and len(es.search_calls) == calls
        finally:
            service._lock.release()
        service.uncatalogued_ai_doc_ids()
        service._refresh_thread.join(timeout=5)
        assert service._uncatalogued is not stale
        assert service._uncatalogued.ai_doc_ids == first

    def test_failed_refresh_keeps_stale_set(self):
        records, catalogue = _corpus()
        es = FakeES(records, catalogue)
        service = _service(es, ttl=0)
        first = service.uncatalogued_ai_doc_ids()
        stale = service._uncatalogued

        def down(**kwargs: Any) -> Dict[str, Any]:
            raise ESConnectionError("elasticsearch unreachable")

        es.search = down  # type: ignore[method-assign]
        assert service.uncatalogued_ai_doc_ids() == first
        service._refresh_thread.join(timeout=5)
        assert service._uncatalogued is stale
        assert not service._lock.locked()

    def test_missing_ai_index(self):
        assert _service(FakeES([], {}, ai_exists=False)).uncatalogued_ai_doc_ids() == []


def _hit(doc_id: str, image_index: int, model: str, lines: List[Tuple[int, str]], score: float = 3.0) -> Dict[str, Any]:
    return {
        "_id": at.record_id(doc_id, image_index, model),
        "_score": score,
        "_source": {"doc_id": doc_id, "image_index": image_index, "ai_read": {"vlm_model": model}},
        "inner_hits": {"lines": {"hits": {"hits": [
            {"_source": {"index": i, "text": t, "agreement": 0.9, "status": AGREED, "bbox": [0, 0, 10, 10]},
             "highlight": {"ai_read.lines.text": [f"<em>{t}</em>"]}}
            for i, t in lines
        ]}}},
    }


class TestSearchUncatalogued:
    def _setup(self, hits: List[Dict[str, Any]], total: int = 7) -> Tuple[FakeES, cr.ConsensusService]:
        records, catalogue = _corpus()
        response = {"hits": {"total": {"value": 9}, "hits": hits}, "aggregations": {"reads": {"value": total}}}
        es = FakeES(records, catalogue, search_response=response)
        return es, _service(es)

    def test_non_hebrew_query_short_circuits(self):
        es, service = self._setup([])
        assert service.search_uncatalogued("letter of Nahray", 10, 0) == (0, [])
        assert service.search_uncatalogued("", 10, 0) == (0, [])
        assert es.search_calls == [] and es.mget_calls == []

    def test_kill_switch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CONSENSUS_SEARCH_ENABLED", "false")
        es, service = self._setup([])
        assert service.search_uncatalogued("שלום", 10, 0) == (0, [])
        assert es.search_calls == []

    def test_ai_master_switch(self, monkeypatch: pytest.MonkeyPatch):
        """AI_TRANSCRIPTIONS_ENABLED=false also turns the search pass off."""
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        es, service = self._setup([_hit(FJP_DOC, 0, "v22i", [(1, "שלום רב לכם")])])
        assert cr.search_enabled() is False
        assert service.search_uncatalogued("שלום", 10, 0) == (0, [])
        assert es.search_calls == [] and es.mget_calls == []

    def test_hit_shape_and_query(self):
        hits = [
            _hit(OLD, 0, "v22i", [(4, "abc שלום tensי"), (1, "שלום רב לכם")]),
            _hit(FJP_DOC, 0, "v22i", [(2, "unfound_text")]),  # nothing usable: skipped
        ]
        es, service = self._setup(hits)
        total, out = service.search_uncatalogued("שלום", 10, 20)
        assert total == 7
        assert out == [{
            "doc_id": NEW, "ai_doc_id": OLD, "image_index": 0, "line_index": 1,
            "snippet": "שלום רב לכם",
            "read_url": f"/read?doc={OLD}&image=0&line=1&model=v22i",
            "vlm_model": "v22i", "score": 3.0,
        }]
        call = es.search_calls[-1]
        assert call["index"] == "ai_v2" and call["collapse"] == {"field": "read_key"}
        assert call["from_"] == 20 and call["size"] == 10
        query = call["query"]["bool"]
        nested = query["must"][0]["nested"]["query"]["bool"]
        assert nested["filter"] == [{"term": {"ai_read.lines.status": "agreed"}}]
        assert query["should"][0]["match_phrase"].keys() == {"text_agreed"}
        assert "text_all" not in repr(call["query"])
        (ids,) = [f["ids"]["values"] for f in query["filter"] if "ids" in f]
        # Only visible records of uncatalogued fragments: not the junk checkpoint, not RICH/BROKEN/orphan.
        assert sorted(ids) == sorted([
            at.record_id(FJP_DOC, 0, "v22i"), at.record_id(THIN_DOC, 1, "v22i"), at.record_id(OLD, 0, "v22i"),
        ])

    def test_matched_line_cleaned_for_snippet(self):
        es, service = self._setup([_hit(THIN_DOC, 1, "v22i", [(3, "לalmולא שלום רב")])])
        _, (hit,) = service.search_uncatalogued("שלום", 5, 0)
        assert hit["snippet"] == "שלום רב" and hit["line_index"] == 3

    def test_encodes_odd_ids(self):
        odd = "Jerusalem_NLI_4°9463_8"
        es, service = self._setup([_hit(odd, 0, "v22i", [(0, "שלום רב")])])
        _, (hit,) = service.search_uncatalogued("שלום", 5, 0)
        assert hit["read_url"] == "/read?doc=Jerusalem_NLI_4%C2%B09463_8&image=0&line=0&model=v22i"

    def test_missing_index(self):
        records, catalogue = _corpus()
        service = _service(FakeES(records, catalogue, ai_exists=False))
        assert service.search_uncatalogued("שלום", 10, 0) == (0, [])

    def test_search_error_is_empty(self):
        records, catalogue = _corpus()
        error = NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        service = _service(FakeES(records, catalogue, search_response=error))
        assert service.search_uncatalogued("שלום", 10, 0) == (0, [])

    def test_no_uncatalogued_fragments(self):
        es = FakeES([_record(doc_id=RICH_DOC)], {RICH_DOC: _cat(["pgp"])})
        assert _service(es).search_uncatalogued("שלום", 10, 0) == (0, [])
        assert not any("collapse" in c for c in es.search_calls)
