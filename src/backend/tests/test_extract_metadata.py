"""Tests for the public DocumentMetadata built by ``ElasticsearchService._extract_metadata``.

The same model feeds the search result cards, the document modal (``/document``)
and the ``/fragment`` pages, so the display fields are checked here once.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from elasticsearch import NotFoundError

from src.backend import public_text as pt
from src.backend.search_service import ElasticsearchService

CAT = "genizah_merged_v8"
V10 = "genizah_merged_v10"


@pytest.fixture
def service() -> ElasticsearchService:
    """A search service without a cluster (``_extract_metadata`` needs none)."""
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.index_name = CAT
    return svc


def _record(**extra: Any) -> Dict[str, Any]:
    """A merged-index ``_source`` with the fields the display mapping reads.

    :param extra: Fields to add or override.
    :return: ``_source`` dict.
    """
    doc = {
        "doc_id": "Cambridge_CUL_T_S_F_1_2_74",
        "canonical_id": "Cambridge_CUL_T_S_F1_74",
        "shelf_mark": "T-S F 1(2).74",
        "description": "תלמוד בבלי [טקסט",
        "alt_titles": ["Talmud Bavli: Nedarim 2 b – 3 b", "תלמוד בבלי: נדרים ב ע\"ב – ג ע\"ב"],
        "subjects": ["Talmud Bavli", "Nedarim"],
        "genres": ["Talmud Bavli Commentaries"],
        "scholarly_frames": ["[Talmud Bavli]: Nedarim 2 b – 3 b", "Talmud manuscripts"],
        "document_type": "talmud",
        "date_display": "1100–1138",
        "century": "12th century",
        "language": "Hebrew",
        "institution": "Cambridge University Library",
        "tei_metadata": {"pgpids": ["2747"], "ktiv_images": ["https://storage.googleapis.com/b/KTIV/1/x.jpg"]},
        "embedding_vector": [0.1] * 4,
        "full_text_content": "FULL TEXT",
    }
    doc.update(extra)
    return doc


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------


def test_display_fields_are_mapped(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(_record(), CAT)
    assert meta.alt_titles == ["Talmud Bavli: Nedarim 2 b – 3 b", "תלמוד בבלי: נדרים ב ע\"ב – ג ע\"ב"]
    assert meta.subjects == ["Talmud Bavli", "Nedarim"]
    assert meta.genres == ["Talmud Bavli Commentaries"]
    # The frame repeating an alt title (brackets aside) is dropped.
    assert meta.scholarly_frames == ["Talmud manuscripts"]
    assert meta.document_type == "talmud"
    assert meta.date_display == "1100–1138"
    assert meta.century == "12th century"
    assert meta.date_info == {"date_display": "1100–1138", "century": "12th century"}
    assert meta.shelf_mark == "T-S F 1(2).74"
    assert meta.source_links == [
        {"source": "cudl", "label": "View on Cambridge Digital Library",
         "url": "https://cudl.lib.cam.ac.uk/view/MS-TS-F-00001-00002-00074", "kind": "item"},
        {"source": "pgp", "label": "View on Princeton Geniza Project",
         "url": "https://geniza.princeton.edu/en/documents/2747/", "kind": "item"},
    ]


def test_heavy_and_private_fields_never_reach_the_payload(service: ElasticsearchService) -> None:
    payload = service._extract_metadata(_record(), CAT).model_dump()
    assert "embedding_vector" not in payload and "full_text_content" not in payload
    assert "tei_metadata" not in payload
    assert "FULL TEXT" not in repr(payload) and "ktiv_images" not in repr(payload)


def test_missing_fields_give_empty_defaults(service: ElasticsearchService) -> None:
    meta = service._extract_metadata({"doc_id": "X_1"}, CAT)
    assert meta.alt_titles == [] and meta.subjects == [] and meta.genres == [] and meta.scholarly_frames == []
    assert meta.source_links == []
    assert meta.document_type is None and meta.date_display is None and meta.century is None
    assert meta.date_info is None and meta.description is None


def test_odd_value_types_are_tolerated(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(alt_titles="Single title", subjects=None, genres=[None, 3, "Poetry", ""], document_type=["x"], century=21),
        CAT,
    )
    assert meta.alt_titles == ["Single title"]
    assert meta.subjects == [] and meta.genres == ["Poetry"]
    assert meta.document_type is None and meta.century is None


def test_dates_without_century_and_with_start_end(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(date_display="1125-12-09/1125-12-18", date_start="1125-12-09", date_end="1125-12-18", century=""),
        CAT,
    )
    assert meta.date_info == {"date_display": "1125-12-09/1125-12-18", "date_start": "1125-12-09", "date_end": "1125-12-18"}
    assert meta.century is None
    # Cards and modal get the visible form the fragment page shows; date_info keeps the raw interval.
    assert meta.date_display == "1125-12-09 – 1125-12-18"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1125-12-09/1125-12-18", "1125-12-09 – 1125-12-18"),
        ("1094/1095", "1094 – 1095"),
        ("1125-12-09", "1125-12-09"),
        ("1100–1138", "1100–1138"),
        ("27 Nisan  4790", "27 Nisan 4790"),
        ("c. 1100/1150 (?)", "c. 1100/1150 (?)"),
        ("", None),
        (None, None),
    ],
)
def test_public_date_display(raw: Optional[str], expected: Optional[str]) -> None:
    assert pt.public_date_display(raw) == expected


def test_an_indexed_date_info_is_kept(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(_record(date_info={"display": "c. 1100"}), CAT)
    assert meta.date_info == {"display": "c. 1100"}
    assert meta.date_display == "1100–1138"


# ---------------------------------------------------------------------------
# Alternative titles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placeholder", pt.ALT_TITLE_PLACEHOLDERS)
def test_placeholder_alt_titles_are_dropped(service: ElasticsearchService, placeholder: str) -> None:
    meta = service._extract_metadata(_record(alt_titles=[placeholder, f" {placeholder}. ", "Materia medica"]), CAT)
    assert meta.alt_titles == ["Materia medica"]


def test_alt_titles_repeating_shelf_mark_or_description_are_dropped(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(
            shelf_mark="T-S NS 292.26",
            description="Documentary. Official document from Fustat.",
            alt_titles=["T-S NS 292.26", "t-s ns 292 26", "Documentary", "Official document", "Petition", "Doc"],
        ),
        CAT,
    )
    # "Doc" is not a whole word of the description, so it stays.
    assert meta.alt_titles == ["Petition", "Doc"]


def test_alt_titles_are_deduplicated_and_capped(service: ElasticsearchService) -> None:
    titles = [f"Title {n}" for n in range(15)]
    meta = service._extract_metadata(_record(alt_titles=["Title 0", "TITLE 0"] + titles), CAT)
    assert meta.alt_titles == titles[: pt.MAX_ALT_TITLES] == titles[:10]


def test_editorial_brackets_survive_and_markup_goes(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(_record(alt_titles=["Letter <t'lt>", "<em>Piyyut</em> for Shavuot"]), CAT)
    assert meta.alt_titles == ["Letter <t'lt>", "Piyyut for Shavuot"]


# ---------------------------------------------------------------------------
# Caps of the v10 lists
# ---------------------------------------------------------------------------


def test_list_caps(service: ElasticsearchService) -> None:
    many = [f"value {n}" for n in range(60)]
    meta = service._extract_metadata(_record(alt_titles=[], subjects=many, genres=many, scholarly_frames=many), CAT)
    assert meta.subjects == many[: pt.MAX_SUBJECTS] == many[:20]
    assert meta.genres == many[: pt.MAX_GENRES] == many[:20]
    assert meta.scholarly_frames == many[: pt.MAX_SCHOLARLY_FRAMES] == many[:10]


def test_frames_drop_raw_alt_titles_hidden_by_the_description(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(description="Responsa of Maimonides", alt_titles=["Responsa"], scholarly_frames=["[Responsa]", "Halakha"]),
        CAT,
    )
    assert meta.alt_titles == [] and meta.scholarly_frames == ["Halakha"]


# ---------------------------------------------------------------------------
# Provider scrubbing
# ---------------------------------------------------------------------------


def test_description_is_scrubbed(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(description="Responsum (per FGP). Information from FGP. See https://fgp.genizah.org/x?id=1 too."),
        CAT,
    )
    assert meta.description == "Responsum. See too."
    assert "FGP" not in meta.description and "genizah.org" not in meta.description


def test_text_fields_are_scrubbed(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(
            alt_titles=["Deed (Information from FGP)"],
            subjects=["court, per FGP"],
            genres=["Letters (FGP)"],
            scholarly_frames=["Friedberg Genizah Project record: Letters"],
        ),
        CAT,
    )
    assert meta.alt_titles == ["Deed"]
    assert meta.subjects == ["court"]
    assert meta.genres == ["Letters"]
    assert meta.scholarly_frames == ["record: Letters"]


def test_description_keeps_whole_sentences(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(description="Abū l-Faḍl wrote to him. According to the FGP bibliography, this manuscript was edited."),
        CAT,
    )
    assert meta.description == (
        "Abū l-Faḍl wrote to him. According to another catalogue's bibliography, this manuscript was edited."
    )


GINZEI_QEDEM = (
    'Oded Zinger, <a href="https://www.academia.edu/34167086/x">"A Karaite-Rabbanite Court Session,"</a> '
    "<em>Ginzei Qedem</em> 13 (n.p.: The Friedberg Genizah Project and the Ben-Zvi Institute, 2017), 95-116."
)


def test_bibliography_is_scrubbed(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(
            bibliography=[
                {"citation": GINZEI_QEDEM, "location": "FGP", "source": "fgp"},
                {"title": "Catalogue (FGP)", "location": "p. 12, per FGP", "authors": ["FGP", "Gil, Moshe"],
                 "relations": ["Mention", "FGP"], "source": "ktiv", "url": "https://fgp.genizah.org/x"},
                "Ginzei Qedem 2 (n.p.: The Friedberg Genizah Project and the Ben-Zvi Institute, 2006), 37-58.",
            ]
        ),
        CAT,
    )
    first, second, third = meta.bibliography_entries
    # The citation keeps the <em>/<a> markup the modal renders.
    assert first["citation"] == (
        'Oded Zinger, <a href="https://www.academia.edu/34167086/x">"A Karaite-Rabbanite Court Session,"</a> '
        "<em>Ginzei Qedem</em> 13 (n.p.: the Ben-Zvi Institute, 2017), 95-116."
    )
    assert first["location"] is None and first["source"] is None and first["title"] is None
    assert second == {
        "citation": second["citation"], "source": "ktiv", "title": "Catalogue", "authors": ["Gil, Moshe"],
        "year": None, "location": "p. 12", "relations": ["Mention"], "url": None,
    }
    assert third["citation"] == "Ginzei Qedem 2 (n.p.: the Ben-Zvi Institute, 2006), 37-58."
    assert meta.bibliography == [first["citation"], second["citation"], third["citation"]]
    payload = repr(meta.model_dump())
    assert "FGP" not in payload and "fgp" not in payload and "Friedberg" not in payload and "genizah.org" not in payload


def test_bibliography_text_without_provider_is_unchanged(service: ElasticsearchService) -> None:
    """Titles and authors stay exact keys for the /bibliography/work lookup."""
    entry = {"title": "גנזי קדם  - חדש vol. ג", "authors": ["Friedberg, Albert D"], "location": "12", "citation": "<em>X</em>"}
    out = service._extract_metadata(_record(bibliography=[entry]), CAT).bibliography_entries[0]
    assert out["title"] == "גנזי קדם  - חדש vol. ג"
    assert out["authors"] == ["Friedberg, Albert D"] and out["location"] == "12" and out["citation"] == "<em>X</em>"


def test_toronto_friedberg_collection_is_untouched(service: ElasticsearchService) -> None:
    meta = service._extract_metadata(
        _record(
            doc_id="Toronto_Friedberg_MSS_9_003",
            canonical_id="Toronto_Friedberg_MSS_9_003",
            shelf_mark="Friedberg MSS 9-003",
            description="Possibly an indirect join: Friedberg MSS 9-004/22 (PGPID 33737).",
            alt_titles=["Friedberg MSS 9-003", "Letter"],
        ),
        CAT,
    )
    assert meta.shelf_mark == "Friedberg MSS 9-003"
    assert meta.description == "Possibly an indirect join: Friedberg MSS 9-004/22 (PGPID 33737)."
    assert meta.alt_titles == ["Letter"]


def test_untouched_description_keeps_its_text(service: ElasticsearchService) -> None:
    text = "Letter in Judaeo-Arabic.\n\nVerso: accounts."
    assert service._extract_metadata(_record(description=text), CAT).description == text
    assert service._extract_metadata(_record(description=""), CAT).description == ""


# ---------------------------------------------------------------------------
# Century
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("11th century", "11th century"),
        ("21st century", "21st century"),
        ("1st century", "1st century"),
        ("48th century", None),  # Anno Mundi 4790 read as CE
        ("55th century", None),
        ("22nd century", None),
        ("", None),
        (None, None),
        ("late antiquity", "late antiquity"),
    ],
)
def test_century(service: ElasticsearchService, raw: Optional[str], expected: Optional[str]) -> None:
    meta = service._extract_metadata(_record(century=raw, date_display="27 Nisan  4790"), CAT)
    assert meta.century == expected
    assert meta.date_display == "27 Nisan 4790"
    assert meta.date_info == ({"date_display": "27 Nisan 4790", "century": expected} if expected else {"date_display": "27 Nisan 4790"})


# ---------------------------------------------------------------------------
# /document: v10 enrichment of catalogue records
# ---------------------------------------------------------------------------


class FakeDocumentES:
    """The two calls ``get_document_by_id`` makes: a catalogue search and the v10 GET."""

    def __init__(self, catalogue: Dict[str, Dict[str, Any]], v10: Optional[Dict[str, Dict[str, Any]]]) -> None:
        self.catalogue = catalogue
        self.v10 = v10
        self.searches: List[Dict[str, Any]] = []
        self.gets: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Exact doc id match on whichever index is asked (the shelf-mark fallback finds nothing)."""
        self.searches.append(kwargs)
        should = kwargs["query"]["bool"]["should"]
        doc_id = should[0].get("term", {}).get("doc_id")
        doc = self.catalogue.get(doc_id) if doc_id else None
        if doc is None:
            return {"hits": {"total": {"value": 0}, "hits": []}}
        source = {k: v for k, v in doc.items() if k not in kwargs.get("source_excludes", [])}
        return {"hits": {"total": {"value": 1}, "hits": [{"_id": doc_id, "_source": source}]}}

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        """Document GET on v10 (missing index or doc: 404)."""
        self.gets.append(kwargs)
        if self.v10 is None:
            raise NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        if kwargs["id"] not in self.v10:
            raise NotFoundError("not_found", MagicMock(status=404), {"found": False})
        doc = self.v10[kwargs["id"]]
        return {"_id": kwargs["id"], "found": True, "_source": {k: v for k, v in doc.items() if k in kwargs["source_includes"]}}


DOC_ID = "Cambridge_CUL_T_S_F_1_2_74"
V10_FIELDS = {
    "doc_id": DOC_ID,
    "alt_titles": ["Nedarim fragment"],
    "subjects": ["vows"],
    "genres": ["Talmud"],
    "scholarly_frames": ["Babylonian Talmud"],
}


@pytest.fixture(autouse=True)
def _default_enrichment_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the default ``METADATA_ENRICHMENT_INDEX``."""
    monkeypatch.delenv("METADATA_ENRICHMENT_INDEX", raising=False)


def _service_with(es: FakeDocumentES) -> ElasticsearchService:
    """A search service on a fake cluster, catalogue index v8.

    :param es: Fake cluster.
    :return: Service.
    """
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.index_name = CAT
    svc.es = es
    return svc


def test_document_reads_v10_fields_for_catalogue_records() -> None:
    es = FakeDocumentES({DOC_ID: _record(subjects=[], genres=[], scholarly_frames=[])}, {DOC_ID: dict(V10_FIELDS)})
    meta = _service_with(es).get_document_by_id(DOC_ID, enrich=True)
    assert meta.alt_titles == ["Nedarim fragment"]
    assert meta.subjects == ["vows"] and meta.genres == ["Talmud"] and meta.scholarly_frames == ["Babylonian Talmud"]
    assert es.gets == [{"index": V10, "id": DOC_ID, "source_includes": ["doc_id", "alt_titles", "subjects", "genres", "scholarly_frames"]}]
    assert es.searches[0]["source_excludes"] == ["embedding_vector"]


def test_document_from_v10_itself_needs_no_extra_get() -> None:
    es = FakeDocumentES({DOC_ID: _record()}, {DOC_ID: dict(V10_FIELDS)})
    meta = _service_with(es).get_document_by_id(DOC_ID, index_name=V10, enrich=True)
    assert es.gets == [] and meta.genres == ["Talmud Bavli Commentaries"]


def test_document_from_another_index_is_not_enriched() -> None:
    es = FakeDocumentES({DOC_ID: _record()}, {DOC_ID: dict(V10_FIELDS)})
    _service_with(es).get_document_by_id(DOC_ID, index_name="genizah_merged_v9", enrich=True)
    assert es.gets == []


@pytest.mark.parametrize("v10", [None, {}], ids=["missing-index", "missing-doc"])
def test_document_enrichment_degrades_silently(v10: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="src.backend.metadata_enrichment")
    es = FakeDocumentES({DOC_ID: _record()}, v10)
    meta = _service_with(es).get_document_by_id(DOC_ID, enrich=True)
    assert meta is not None and meta.genres == ["Talmud Bavli Commentaries"]
    assert len([r for r in caplog.records if r.name == "src.backend.metadata_enrichment"]) == 1


def test_document_enrichment_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METADATA_ENRICHMENT_INDEX", "")
    es = FakeDocumentES({DOC_ID: _record()}, {DOC_ID: dict(V10_FIELDS)})
    meta = _service_with(es).get_document_by_id(DOC_ID, enrich=True)
    assert es.gets == [] and meta.alt_titles[0] == "Talmud Bavli: Nedarim 2 b – 3 b"


def test_document_lookup_is_not_enriched_by_default() -> None:
    es = FakeDocumentES({DOC_ID: _record()}, {DOC_ID: dict(V10_FIELDS)})
    meta = _service_with(es).get_document_by_id(DOC_ID)
    assert es.gets == [] and meta.genres == ["Talmud Bavli Commentaries"]


def test_iiif_manifest_makes_no_enrichment_get() -> None:
    es = FakeDocumentES({DOC_ID: _record(image_urls=["https://images.example/1.jpg"])}, {DOC_ID: dict(V10_FIELDS)})
    manifest = _service_with(es).generate_iiif_manifest(DOC_ID)
    assert manifest is not None and es.gets == [] and len(es.searches) == 1


def test_document_route_enriches_and_manifest_route_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from src.backend import app as app_module

    record = _record(subjects=[], genres=[], scholarly_frames=[], image_urls=["https://images.example/1.jpg"])
    es = FakeDocumentES({DOC_ID: record}, {DOC_ID: dict(V10_FIELDS)})
    monkeypatch.setattr(app_module, "search_service", _service_with(es))
    client = TestClient(app_module.app)
    body = client.get(f"/document/{DOC_ID}").json()
    assert body["subjects"] == ["vows"] and body["genres"] == ["Talmud"] and len(es.gets) == 1
    assert client.get(f"/document/{DOC_ID}/manifest").status_code == 200
    assert len(es.gets) == 1


def test_placeholder_password_never_targets_a_remote_cluster() -> None:
    """Without src/backend/.env the conftest placeholder pins ES to a closed local port."""
    if os.environ.get("ELASTICSEARCH_PASSWORD") != "unused-placeholder":
        pytest.skip("a real password is configured (main checkout)")
    assert os.environ["ELASTICSEARCH_HOST"] == "127.0.0.1" and os.environ["ELASTICSEARCH_PORT"] == "9"
    from src.backend.search_service import search_service

    assert search_service.es_host == "127.0.0.1" and str(search_service.es_port) == "9"
