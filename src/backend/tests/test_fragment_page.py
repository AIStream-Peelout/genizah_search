"""Tests for the public ``/fragment/<doc_id>`` pages and the fragment sitemap routes."""

import html
import json
import re
from typing import Any, Dict, List, Optional

import pytest
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import NotFoundError
from unittest.mock import MagicMock

from src.backend import ai_transcriptions as at
from src.backend import fragment_page as fp
from src.backend.consensus_reads import ConsensusService
from src.backend.fragment_sitemaps import FragmentSitemaps
from src.backend.search_service import ElasticsearchService

CAT = "genizah_merged_v8"
AI = "genizah_ai_transcriptions_v2"
PUBLIC_HOST = {"host": "cairogenizah.ai"}
MODEL = "qwen3-vl-8b-heb-v21b-step1200"

PGP_ID = "Cambridge_CUL_T_S_8_108"
FJP_ID = "Manchester_JRL_G_66"
OXFORD_V8 = "Oxford_Bodleian_Bodl_MS_heb_a_2_14"
OXFORD_OLD = "Oxford_Bodleian_MS_heb_a_2_14"
ODD_ID = "Heidelberg_Cod_Heid_Or_??"
DEGREE_ID = "Jerusalem_NLI_4°9463_8"
EVIL_ID = "Evil_1"

LONG_TEXT = "A legal deed concerning the sale of half a house in Fustat, with witnesses and the court's validation."


def _pgp_doc() -> Dict[str, Any]:
    """A PGP + unattributed-provider record carrying every kind of private data."""
    return {
        "doc_id": PGP_ID,
        "shelf_mark": "T-S 8.108",
        "description": (
            "Legal document in the hand of Ḥalfon b. Menashshe. Account (FGP). "
            "Apparently a letter in Ladino, per FGP. See https://fgp.genizah.org/x?id=1 and "
            "www.jewishmanuscripts.org/item/9 for the Friedberg Genizah Project record. "
            "<em>Dated</em> 1125 CE."
        ),
        "date_display": "1125-12-09/1125-12-18",
        "date_start": "1125-12-09",
        "date_end": "1125-12-18",
        "century": "12th century",
        "language": "Aramaic, Judaeo-Arabic",
        "document_type": "legal",
        "institution": "Cambridge University Library",
        "collection": "Taylor-Schechter",
        "transcriptions": [{"editor": "Goitein, S. D.", "text": "בראה כאמלה\nואסתיפא בראה מני", "language": "Judaeo-Arabic"}],
        "has_transcriptions": True,
        "translations": [{"text": "Full acquittance <and send it> to him.", "target_language": "English"}],
        "bibliography": [
            {
                "citation": 'Goitein, <em>A Mediterranean Society</em>, vol. 1, <a href="https://doi.org/10.1/x">online</a>.',
                "source": "pgp",
                "url": "https://geniza.princeton.edu/documents/2747/",
                "title": None,
                "authors": [],
                "relations": ["Digital Edition"],
            },
            {
                "citation": "גרייף, אבנר, יהדות סיציליה",
                "title": "יהדות סיציליה בתקופה המוסלמית",
                "source": "Yitzchack Gila, Head of FGP Bibliography B team",
                "url": "https://fgp.genizah.org/FunctionPages/Bibliography/BibliographyRefInfo.aspx?id=1",
                "location": "79, 107",
                "year": 'תשמ"ה',
                "authors": [],
            },
            {
                "citation": "Ginzei Qedem 2 (n.p.: The Friedberg Genizah Project and the Ben-Zvi Institute, 2006), 37-58.",
                "source": "fjp",
                "url": "https://www.jewishmanuscripts.org/item/1",
            },
            {
                "citation": "Ktiv entry",
                "source": "ktiv",
                "title": "Sefer ha-Ma'asim",
                "authors": ["Author, A."],
                "year": "1990",
                "relations": ["Mention"],
            },
        ],
        "joins_data": {
            "joinedManuscripts": [{"shelfmark": "Cambridge, CUL: T-S 12.843", "index": 0, "source": "listOfJoins"}],
            "mainShelfmark": "Cambridge, CUL: T-S 8.108",
            "source": "Site User - Dr. Example Person",
            "metadata": {"pageUrl": "https://fgp.genizah.org/GeneralPages/Join/JoinDetails.aspx?InventoryId=1"},
        },
        "sources_present": ["pgp", "fjp"],
        "tei_metadata": {"sources_present": ["pgp", "fjp"], "pgpids": ["2747"]},
        "image_urls": [
            "https://storage.googleapis.com/b/T_S_8_108_1r.jpg 1440w",
            "https://storage.googleapis.com/b/T_S_8_108_1v.jpg",
            "https://fgp.genizah.org/images/blocked.jpg",
        ],
        "embedding_vector": [0.1, 0.2],
        "full_text_content": "SHOULD_NOT_BE_FETCHED",
    }


def _fjp_doc() -> Dict[str, Any]:
    """An unattributed-provider-only record with an empty description (uncatalogued)."""
    return {
        "doc_id": FJP_ID,
        "shelf_mark": "Manchester: G 66",
        "description": "",
        "language": "Other",
        "institution": "Rylands",
        "collection": "Manchester",
        "transcriptions": [],
        "has_transcriptions": False,
        "bibliography": [],
        "sources_present": ["fjp"],
        "tei_metadata": {"sources_present": ["fjp"], "image_preferred_source": "fjp"},
        "image_urls": [
            "https://storage.googleapis.com/b/G_66_1r.jpg",
            "https://storage.googleapis.com/b/G_66_1v.jpg",
        ],
    }


def _simple_doc(doc_id: str, **extra: Any) -> Dict[str, Any]:
    """A plain catalogued record.

    :param doc_id: Catalogue id.
    :param extra: Fields to override.
    :return: ``_source`` dict.
    """
    doc = {
        "doc_id": doc_id,
        "shelf_mark": doc_id.replace("_", " "),
        "description": LONG_TEXT,
        "institution": "Bodleian Libraries",
        "collection": "Oxford",
        "has_transcriptions": False,
        "sources_present": ["bodleian"],
        "image_urls": [f"https://storage.googleapis.com/b/{len(doc_id)}.jpg"],
    }
    doc.update(extra)
    return doc


def _catalogue() -> Dict[str, Dict[str, Any]]:
    """All catalogue fixtures by doc id."""
    docs = [
        _pgp_doc(),
        _fjp_doc(),
        _simple_doc(OXFORD_V8),
        _simple_doc(ODD_ID, sources_present=["ktiv"]),
        _simple_doc(DEGREE_ID, sources_present=["ktiv"]),
        _simple_doc(
            EVIL_ID,
            shelf_mark="<img src=x onerror=alert(1)>",
            description="</script><script>alert(1)</script> " + LONG_TEXT,
            institution="A & B <Library>",
        ),
    ]
    return {d["doc_id"]: d for d in docs}


def _ai_record(
    doc_id: str = FJP_ID,
    image_index: int = 0,
    image_url: str = "https://storage.googleapis.com/b/G_66_1r.jpg",
) -> Dict[str, Any]:
    """A published, surfaced AI read with agreed, junk and unconfirmed lines."""
    lines = [
        ("וקיים לעד ברוך אתה יי מלך על כל הארץ", "agreed"),
        ("אלהינו בעמך ישראל ובתפלתם שעה והשב", "agreed"),
        ("סודי ביותר לא מאושר", "unconfirmed"),
        ("לalmולא ומהרה תקבל ברצון", "agreed"),
        ("ישראל ברוך טוב לחיים", "agreed"),
    ]
    raw = [{"index": i, "text": t, "status": s} for i, (t, s) in enumerate(lines)]
    return {
        "doc_id": doc_id,
        "image_index": image_index,
        "image_url": image_url,
        "image_sha256": f"sha-{doc_id}-{image_index}",
        "read_key": at.read_key(doc_id, image_index),
        "published": True,
        "surfaced": True,
        "ai_read": {"vlm_model": MODEL, "decoded_at": "2026-10-02T11:21:37Z", "parsed": True, "lines": raw},
    }


def _get(source: Dict[str, Any], path: str) -> Any:
    """Dotted-path lookup in a ``_source`` dict."""
    node: Any = source
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


class FakeES:
    """In-memory catalogue + AI index for the calls the page and sitemap code make."""

    def __init__(self, catalogue: Dict[str, Dict[str, Any]], ai_records: List[Dict[str, Any]]):
        self.catalogue = catalogue
        self.ai_records = ai_records
        self.calls: List[Dict[str, Any]] = []
        self.fail_catalogue: Optional[Exception] = None
        self.fail_ai: Optional[Exception] = None

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs["index"] == CAT:
            if self.fail_catalogue is not None:
                raise self.fail_catalogue
            return self._catalogue_search(kwargs)
        if self.fail_ai is not None:
            raise self.fail_ai
        filters = kwargs["query"]["bool"]["filter"]
        hits = []
        for rec in self.ai_records:
            ok = True
            for clause in filters:
                (kind, body), = clause.items()
                (path, value), = body.items()
                actual = _get(rec, path)
                ok = ok and (actual in value if kind == "terms" else actual == value)
            if ok:
                hits.append({"_id": at.record_id(rec["doc_id"], rec["image_index"], MODEL), "_source": rec})
        return {"hits": {"hits": hits}}

    def _catalogue_search(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        query = kwargs["query"]
        if "exists" in query:  # sitemap scan
            ids = sorted(self.catalogue)
            after = kwargs.get("search_after")
            if after:
                ids = [i for i in ids if i > after[0]]
            ids = ids[: kwargs["size"]]
            return {"hits": {"hits": [{"fields": {"doc_id": [i]}, "sort": [i]} for i in ids]}}
        assert set(query) == {"term"} and set(query["term"]) == {"doc_id"}, query
        doc = self.catalogue.get(query["term"]["doc_id"])
        if doc is None:
            return {"hits": {"hits": []}}
        source = {k: v for k, v in doc.items() if k not in kwargs.get("source_excludes", [])}
        return {"hits": {"hits": [{"_id": doc["doc_id"], "_source": source}]}}

    def catalogue_lookups(self) -> List[str]:
        """Ids looked up with the exact ``term`` query."""
        return [c["query"]["term"]["doc_id"] for c in self.calls if c["index"] == CAT and "term" in c["query"]]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default surfacing thresholds and AI switches for every test."""
    for name in (
        "AI_READ_MIN_AGREED_LINES", "AI_READ_MIN_AGREED_SHARE", "CONSENSUS_SEARCH_ENABLED", "AI_TRANSCRIPTIONS_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake() -> FakeES:
    """Fake cluster with all fixtures and one AI read of the FJP-only fragment."""
    return FakeES(_catalogue(), [_ai_record()])


@pytest.fixture
def service(fake: FakeES) -> fp.FragmentPageService:
    """Page service wired like app.py, on the fake cluster."""
    metadata_source = ElasticsearchService.__new__(ElasticsearchService)
    return fp.FragmentPageService(
        fake, CAT, metadata_source=metadata_source, consensus_service=ConsensusService(fake, CAT, AI)
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake: FakeES, service: fp.FragmentPageService):
    """TestClient on the real app with the page service and sitemaps swapped for fakes."""
    from fastapi.testclient import TestClient
    from src.backend import app as app_module

    monkeypatch.setattr(app_module, "fragment_page_service", service)
    monkeypatch.setattr(app_module, "fragment_sitemaps", FragmentSitemaps(fake, CAT, urls_per_file=2))
    return TestClient(app_module.app)


def _json_ld(html_text: str) -> Dict[str, Any]:
    """Parse the page's JSON-LD block."""
    blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html_text, re.S)
    assert len(blocks) == 1
    return json.loads(blocks[0])


def _meta(html_text: str, attr: str, name: str) -> str:
    """Content of one ``<meta>`` tag."""
    match = re.search(rf'<meta {attr}="{re.escape(name)}" content="([^"]*)">', html_text)
    assert match, name
    return match.group(1)


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


class TestRender:
    def test_catalogued_record(self, client, fake):
        resp = client.get(f"/fragment/{PGP_ID}", headers=PUBLIC_HOST)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.headers["cache-control"] == "public, max-age=3600"
        assert "x-robots-tag" not in resp.headers
        page = resp.text
        assert "<title>T-S 8.108 · Cairo Genizah fragment · Cairo Genizah AI</title>" in page
        assert f'<link rel="canonical" href="https://cairogenizah.ai/fragment/{PGP_ID}">' in page
        assert _meta(page, "property", "og:url") == f"https://cairogenizah.ai/fragment/{PGP_ID}"
        assert _meta(page, "property", "og:image") == "https://storage.googleapis.com/b/T_S_8_108_1r.jpg"
        assert "<h1>T-S 8.108</h1>" in page
        assert f'href="/?doc={PGP_ID}">Open in Cairo Genizah AI</a>' in page
        # The app link is a duplicate of the homepage for crawlers: never followed.
        assert page.count(f'<a class="button" rel="nofollow" href="/?doc={PGP_ID}">') == 2
        assert "Cambridge University Library" in page and "Taylor-Schechter" in page
        assert "1125-12-09 – 1125-12-18" in page and "Aramaic, Judaeo-Arabic" in page and "Legal" in page
        assert 'dir="rtl" lang="he">בראה כאמלה' in page
        assert "Full acquittance &lt;and send it&gt; to him." in page  # editorial brackets kept, escaped
        assert "Cambridge, CUL: T-S 12.843" in page
        assert "Princeton Geniza Project" in page and "CC BY-NC 4.0" in page
        assert "KTIV, National Library of Israel" in page
        assert "<em>A Mediterranean Society</em>" in page
        assert "data-nosnippet" not in page and fp.MACHINE_READING_LABEL not in page
        assert 'loading="lazy"' in page  # second image
        assert "blocked.jpg" not in page
        meta_description = _meta(page, "name", "description")
        assert meta_description.startswith("Legal document in the hand of Ḥalfon b. Menashshe.")
        assert len(meta_description) <= 160

    def test_json_ld(self, client):
        ld = _json_ld(client.get(f"/fragment/{PGP_ID}").text)
        # ArchiveComponent: holdingArchive is only defined for that type.
        assert ld["@context"] == "https://schema.org" and ld["@type"] == ["Manuscript", "ArchiveComponent"]
        assert ld["name"] == ld["identifier"] == "T-S 8.108"
        assert ld["url"] == f"https://cairogenizah.ai/fragment/{PGP_ID}"
        assert ld["image"] == {"@type": "ImageObject", "contentUrl": "https://storage.googleapis.com/b/T_S_8_108_1r.jpg"}
        assert ld["holdingArchive"] == {"@type": "ArchiveOrganization", "name": "Cambridge University Library"}
        assert ld["inLanguage"] == ["Aramaic", "Judaeo-Arabic"]
        assert ld["temporalCoverage"] == "1125-12-09/1125-12-18"
        assert ld["isPartOf"] == {"@type": "Collection", "name": "Cairo Genizah"}
        assert ld["keywords"] == ["Legal"]  # catalogued: no machine-reading terms
        assert "יהדות סיציליה בתקופה המוסלמית" in ld["citation"]
        assert "Ginzei Qedem 2 (n.p.: the Ben-Zvi Institute, 2006), 37-58." in ld["citation"]

    def test_head(self, client):
        resp = client.head(f"/fragment/{PGP_ID}", headers=PUBLIC_HOST)
        assert resp.status_code == 200 and resp.content == b""
        assert resp.headers["content-type"] == "text/html; charset=utf-8"

    def test_page_cache_spares_elasticsearch(self, client, fake):
        assert client.get(f"/fragment/{PGP_ID}").status_code == 200
        calls = len(fake.calls)
        assert client.get(f"/fragment/{PGP_ID}").status_code == 200
        assert client.head(f"/fragment/{PGP_ID}").status_code == 200
        assert len(fake.calls) == calls


class TestLookup:
    def test_exact_doc_id_only(self, client, fake):
        """A shelfmark that get_document_by_id would fuzzy-match is a 404 here."""
        for path in ("/fragment/T-S%208.108", "/fragment/cambridge_cul_t_s_8_108", f"/fragment/{PGP_ID}%20"):
            resp = client.get(path)
            assert resp.status_code == 404, path
        assert all(c["index"] == CAT for c in fake.calls)
        assert fake.catalogue_lookups() == ["T-S 8.108", "cambridge_cul_t_s_8_108", PGP_ID + " "]
        client.get("/fragment/T-S 8.108")  # same decoded id: the cached 404 answers
        assert len(fake.catalogue_lookups()) == 3

    def test_not_found_page(self, client):
        resp = client.get("/fragment/No_Such_<b>Id</b>", headers=PUBLIC_HOST)
        assert resp.status_code == 404
        assert resp.headers["x-robots-tag"] == "noindex"
        assert '<meta name="robots" content="noindex">' in resp.text
        assert "No_Such_&lt;b&gt;Id&lt;/b&gt;" in resp.text and "<b>Id</b>" not in resp.text
        assert 'href="/"' in resp.text

    def test_empty_id_is_404(self, client, fake):
        assert client.get("/fragment/").status_code == 404
        assert fake.calls == []

    def test_oxford_old_id_redirects(self, client):
        resp = client.get(f"/fragment/{OXFORD_OLD}", follow_redirects=False)
        assert resp.status_code == 301
        assert resp.headers["location"] == f"/fragment/{OXFORD_V8}"
        assert client.get(f"/fragment/{OXFORD_V8}").status_code == 200

    def test_unresolved_old_oxford_id_is_404(self, client):
        resp = client.get("/fragment/Oxford_Bodleian_MS_heb_z_9_9", follow_redirects=False)
        assert resp.status_code == 404

    def test_percent_encoded_odd_ids_round_trip(self, client, fake):
        resp = client.get("/fragment/Heidelberg_Cod_Heid_Or_%3F%3F", headers=PUBLIC_HOST)
        assert resp.status_code == 200
        assert fake.catalogue_lookups()[-1] == ODD_ID
        assert '<link rel="canonical" href="https://cairogenizah.ai/fragment/Heidelberg_Cod_Heid_Or_%3F%3F">' in resp.text
        assert 'href="/?doc=Heidelberg_Cod_Heid_Or_%3F%3F"' in resp.text
        resp = client.get("/fragment/Jerusalem_NLI_4%C2%B09463_8")
        assert resp.status_code == 200
        assert "https://cairogenizah.ai/fragment/Jerusalem_NLI_4%C2%B09463_8" in resp.text


class TestAvailability:
    def test_es_down_is_503_not_404_and_not_cached(self, client, fake):
        fake.fail_catalogue = ESConnectionError("connection refused")
        resp = client.get(f"/fragment/{PGP_ID}")
        assert resp.status_code == 503
        assert resp.headers["retry-after"] == str(fp.RETRY_AFTER_SECONDS)
        assert resp.headers["x-robots-tag"] == "noindex"
        assert resp.headers["cache-control"] == "no-store"
        fake.fail_catalogue = None
        assert client.get(f"/fragment/{PGP_ID}").status_code == 200

    def test_missing_index_is_503(self, client, fake):
        fake.fail_catalogue = NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        assert client.get(f"/fragment/{PGP_ID}").status_code == 503

    def test_ai_index_failure_renders_page_without_reads_uncached(self, client, fake):
        fake.fail_ai = ESConnectionError("ai index down")
        resp = client.get(f"/fragment/{FJP_ID}")
        assert resp.status_code == 200 and "data-nosnippet" not in resp.text
        fake.fail_ai = None
        resp = client.get(f"/fragment/{FJP_ID}")
        assert "data-nosnippet" in resp.text

    @pytest.mark.parametrize(
        "host, noindex",
        [
            ("api.cairogenizah.ai", True),
            ("www.cairogenizah.ai", True),
            ("192.168.8.122:8000", True),
            ("cairogenizah.ai", False),
            ("CairoGenizah.ai:443", False),
            ("localhost:8010", False),
            ("127.0.0.1", False),
            ("[::1]:8010", False),
        ],
    )
    def test_noindex_outside_public_host(self, client, host, noindex):
        resp = client.get(f"/fragment/{PGP_ID}", headers={"host": host})
        assert resp.status_code == 200
        assert (resp.headers.get("x-robots-tag") == "noindex") is noindex


# ---------------------------------------------------------------------------
# Escaping and leaks
# ---------------------------------------------------------------------------


class TestEscaping:
    def test_json_ld_cannot_break_out_of_script(self, client):
        page = client.get(f"/fragment/{EVIL_ID}").text
        assert "<script>alert(1)" not in page
        assert page.count("</script>") == 1  # only the JSON-LD element's own end tag
        ld = _json_ld(page)
        assert ld["description"].startswith("</script><script>alert(1)</script>")
        assert ld["name"] == "<img src=x onerror=alert(1)>"
        assert ld["holdingArchive"]["name"] == "A & B <Library>"

    def test_html_is_autoescaped(self, client):
        page = client.get(f"/fragment/{EVIL_ID}").text
        assert "<img src=x onerror" not in page
        assert "<h1>&lt;img src=x onerror=alert(1)&gt;</h1>" in page
        assert "A &amp; B &lt;Library&gt;" in page
        assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in page

    def test_json_ld_script_body_escapes(self):
        body = fp.json_ld_script_body({"d": "</script><b>&"})
        assert "<" not in body and ">" not in body and "&" not in body
        assert json.loads(body) == {"d": "</script><b>&"}


class TestNoProviderLeak:
    FORBIDDEN_ANY_CASE = ("fjp", "fgp", "friedberg", "genizah.org", "jewishmanuscripts", "sources_present", "tei_metadata")
    FORBIDDEN = ("Yitzchack", "Head of", "Example Person", "Site User", "listOfJoins", "pageUrl", "SHOULD_NOT_BE_FETCHED")

    @pytest.mark.parametrize("doc_id", [PGP_ID, FJP_ID])
    def test_rendered_page_has_no_provider_data(self, client, doc_id):
        page = client.get(f"/fragment/{doc_id}").text
        lowered = page.lower()
        for needle in self.FORBIDDEN_ANY_CASE:
            assert needle not in lowered, needle
        for needle in self.FORBIDDEN:
            assert needle not in page, needle

    def test_fjp_only_record_is_shown_without_attribution(self, client):
        page = client.get(f"/fragment/{FJP_ID}").text
        assert "<h1>Manchester: G 66</h1>" in page
        assert "Catalogue information from" not in page  # no PGP/KTIV credit for this record
        assert "References recorded by" not in page

    def test_pgp_record_is_credited(self, client):
        page = client.get(f"/fragment/{PGP_ID}").text
        assert "Catalogue information from" in page and "the Princeton Geniza Project (PGP)" in page
        assert 'href="https://creativecommons.org/licenses/by-nc/4.0/"' in page


# ---------------------------------------------------------------------------
# Machine reading
# ---------------------------------------------------------------------------


class TestMachineReading:
    def test_section_present_with_reads(self, client):
        page = client.get(f"/fragment/{FJP_ID}").text
        section = re.search(r'<section class="panel machine" data-nosnippet[^>]*>(.*?)</section>', page, re.S)
        assert section, "machine-reading section with data-nosnippet"
        body = section.group(1)
        assert fp.MACHINE_READING_LABEL in body
        assert '<ol dir="rtl" lang="he">' in body
        assert '<li value="1">וקיים לעד ברוך אתה יי מלך על כל הארץ</li>' in body
        assert '<li value="4">ומהרה תקבל ברצון</li>' in body  # junk token dropped, line numbering kept
        assert "alm" not in body
        assert "סודי ביותר" not in page  # unconfirmed line never shown
        assert "Image 1 · 4 agreed lines of 5" in body
        assert f'<a href="/read?doc={FJP_ID}&amp;image=0&amp;model={MODEL}" rel="nofollow">See it on the manuscript</a>' in body

    def test_no_section_without_reads(self, client):
        page = client.get(f"/fragment/{OXFORD_V8}").text
        assert "data-nosnippet" not in page and fp.MACHINE_READING_LABEL not in page

    def test_machine_text_only_inside_the_nosnippet_section(self, client):
        """Snippet-facing metadata (meta, Open Graph, JSON-LD description) never carries
        machine-reading text; only the never-displayed JSON-LD keywords may."""
        page = client.get(f"/fragment/{FJP_ID}").text
        section_re = r'<section class="panel machine" data-nosnippet[^>]*>.*?</section>'
        ld_re = r'<script type="application/ld\+json">.*?</script>'
        assert re.search(section_re, page, re.S)
        outside = re.sub(ld_re, "", re.sub(section_re, "", page, flags=re.S), flags=re.S)
        read_texts = ["וקיים לעד ברוך אתה יי מלך על כל הארץ", "אלהינו בעמך ישראל ובתפלתם שעה והשב",
                      "ומהרה תקבל ברצון", "ישראל ברוך טוב לחיים"]
        read_words = {w for text in read_texts for w in text.split()}
        assert not any(w in outside for w in read_words)
        description = html.unescape(_meta(page, "name", "description"))
        assert description == (
            "Manchester: G 66: a Cairo Genizah manuscript fragment held at Rylands, Manchester. "
            "Images, catalogue record and an unchecked machine reading."
        )
        assert len(description) <= 160
        assert html.unescape(_meta(page, "property", "og:description")) == description
        ld = _json_ld(page)
        assert ld["description"] == description
        assert not re.search("[א-ת]", ld["description"])

    def test_uncatalogued_keywords_carry_consensus_terms(self, client):
        """An uncatalogued record's JSON-LD keywords include its consensus search terms."""
        ld = _json_ld(client.get(f"/fragment/{FJP_ID}").text)
        hebrew = [k for k in ld["keywords"] if re.search("[א-ת]", k)]
        assert hebrew and len(hebrew) <= fp.KEYWORD_TERMS_MAX
        assert "ברוך" in hebrew and "alm" not in " ".join(hebrew)

    def test_ai_master_switch_hides_readings(self, client, monkeypatch: pytest.MonkeyPatch):
        """AI_TRANSCRIPTIONS_ENABLED=false: no machine-reading section and no /read link."""
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        page = client.get(f"/fragment/{FJP_ID}").text
        assert "panel machine" not in page and fp.MACHINE_READING_LABEL not in page
        assert "/read?" not in page and "וקיים" not in page
        assert "machine reading" not in html.unescape(_meta(page, "name", "description"))

    def test_old_oxford_form_reads_reach_the_v8_page(self, fake, client):
        fake.ai_records.append(
            _ai_record(doc_id=OXFORD_OLD, image_index=0, image_url=fake.catalogue[OXFORD_V8]["image_urls"][0])
        )
        page = client.get(f"/fragment/{OXFORD_V8}").text
        assert "data-nosnippet" in page
        assert f"/read?doc={OXFORD_OLD}&amp;image=0" in page
        assert "Image 1 ·" in page
        # Catalogued (Bodleian, long description): the description stays the meta description.
        assert html.unescape(_meta(page, "name", "description")) == LONG_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestScrub:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Account (FGP)", "Account"),
            ("Responsum (per FGP).", "Responsum."),
            (
                "Documentary per FGP - needs examination. Apparently a letter in Ladino, per FGP.",
                "Documentary - needs examination. Apparently a letter in Ladino.",
            ),
            (
                "Ginzei Qedem 2 (n.p.: The Friedberg Genizah Project and the Ben-Zvi Institute, 2006), 37-58.",
                "Ginzei Qedem 2 (n.p.: the Ben-Zvi Institute, 2006), 37-58.",
            ),
            ("catalogued in the Friedberg/NLI metadata as a get", "catalogued in the NLI metadata as a get"),
            ("See https://fgp.genizah.org/a?b=1 for more.", "See for more."),
            ("Listed at www.jewishmanuscripts.org/x.", "Listed at."),
            ("Data from FJP.", ""),
            # Whole attribution constructions go, not just the acronym (production phrasings).
            ("(Information from CUDL.) See FGP for further information.", "(Information from CUDL.)"),
            (
                "Images may be found on the BL website (not on FGP): https://www.bl.uk/x",
                "Images may be found on the BL website: https://www.bl.uk/x",
            ),
            ("reused on the verso for piyyūṭ according FGP.", "reused on the verso for piyyūṭ."),
            ("(Information from CUDL and FGP.)", "(Information from CUDL.)"),
            ("Piyyut (qina) by Elʿazar ha-Qalir. Information from FGP.", "Piyyut (qina) by Elʿazar ha-Qalir."),
            (
                "the image listed as verso on FGP actually belongs to T-S K25.157.",
                "the image listed as verso in another catalogue actually belongs to T-S K25.157.",
            ),
            ("MCD (with information from FGP).", "MCD."),
            ("Information from FGP.", ""),
            ("Dream interpretation. In Judaeo-Arabic. Information from FGP.", "Dream interpretation. In Judaeo-Arabic."),
            ("(Information in part from Worman catalog and FGP.)", "(Information in part from Worman catalog.)"),
            ("(See Shivtiel/Niessen and FGP.)", "(See Shivtiel/Niessen.)"),
            (
                "There is a detailed description and edition on FGP.",
                "There is a detailed description and edition in another catalogue.",
            ),
            ("(Information in part from FGP).", ""),
            ("There are joins: see FGP.", "There are joins: see another catalogue."),
            ("Probably not a letter (as catalogued on FGP).", "Probably not a letter."),
            ("(Information from FGP and AA.)", "(Information from AA.)"),
        ],
    )
    def test_provider_mentions_removed(self, raw, expected):
        assert fp.scrub_provider_mentions(raw) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Possibly an indirect join: Friedberg MSS 9-004/22 (PGPID 33737).",
            "Friedberg, Albert D, An evaluation of Maimonides' Enumeration of the Commandments.",
            "b. […] known as Ibn Baqā  ( ? ) ,see",
            "Hosted at cairogenizah.org and geniza.princeton.edu.",
        ],
    )
    def test_other_text_untouched(self, text):
        assert fp.scrub_provider_mentions(text) == text

    def test_markup_stripped_but_editorial_brackets_kept(self):
        assert fp.clean_inline_text('<em>Title</em> <a href="https://x">link</a> <b>bold</b>') == "Title link bold"
        assert fp.strip_markup("He said <and send it> to me") == "He said <and send it> to me"
        assert fp.citation_segments("A <em>B</em> C") == [
            {"text": "A", "em": False}, {"text": "B", "em": True}, {"text": "C", "em": False}
        ]


class TestMetaDescription:
    def test_thin_placeholder_description_gets_a_sentence(self):
        out = fp.build_meta_description("T-S 8K20.2", "Cambridge University Library", "קטע גניזה")
        assert out == (
            "T-S 8K20.2: a Cairo Genizah manuscript fragment held at Cambridge University Library. "
            "Images and catalogue record on Cairo Genizah AI."
        )

    def test_reads_are_mentioned_never_quoted(self):
        out = fp.build_meta_description("ENA 2", "JTS", "", has_reads=True)
        assert out == (
            "ENA 2: a Cairo Genizah manuscript fragment held at JTS. "
            "Images, catalogue record and an unchecked machine reading."
        )

    def test_short_real_description_is_kept(self):
        out = fp.build_meta_description("ENA 1", "JTS", "Responsum.", has_reads=True)
        assert out == "ENA 1: Responsum. A Cairo Genizah manuscript fragment held at JTS."

    def test_long_description_is_shortened(self):
        out = fp.build_meta_description("X", "Y", "word " * 80)
        assert len(out) <= 160 and out.endswith("…")

    def test_placeholders(self):
        assert fp.is_placeholder_description("A few words. Blank verso.")
        assert fp.is_placeholder_description("Unidentified")
        assert not fp.is_placeholder_description("Responsum")


class TestSmallHelpers:
    def test_shorten(self):
        assert fp.shorten("short text", 160) == "short text"
        out = fp.shorten("word " * 60, 50)
        assert len(out) <= 50 and out.endswith("…") and not out.endswith(" …")

    def test_text_direction(self):
        assert fp.text_direction("בראה כאמלה") == ("rtl", "he")
        assert fp.text_direction("بسم الله") == ("rtl", "ar")
        assert fp.text_direction("My lord, teacher") == ("auto", None)
        assert fp.text_direction("") == ("auto", None)

    def test_page_cache_lru_ttl_and_budget(self):
        now = [0.0]
        cache = fp.PageCache(max_entries=2, max_bytes=10, ttl_seconds=5, clock=lambda: now[0])
        cache.put("a", fp.FragmentResponse(200, "1234"))
        cache.put("b", fp.FragmentResponse(200, "1234"))
        assert cache.get("a") is not None  # a is now most recent
        cache.put("c", fp.FragmentResponse(200, "1234"))
        assert cache.get("b") is None and cache.get("a") is not None and len(cache) == 2
        cache.put("big", fp.FragmentResponse(200, "x" * 11))
        assert cache.get("big") is None
        now[0] = 6.0
        assert cache.get("a") is None


# ---------------------------------------------------------------------------
# Sitemaps
# ---------------------------------------------------------------------------


class TestSitemapRoutes:
    def test_index_and_pages(self, client):
        resp = client.get("/sitemap-fragments.xml")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/xml")
        assert resp.headers["cache-control"] == "public, max-age=86400"
        locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
        assert locs == [f"https://cairogenizah.ai/sitemap-fragments-{n}.xml" for n in (1, 2, 3)]

        urls = []
        for n in (1, 2, 3):
            page = client.get(f"/sitemap-fragments-{n}.xml")
            assert page.status_code == 200 and "<urlset" in page.text
            urls += re.findall(r"<loc>(.*?)</loc>", page.text)
        assert "https://cairogenizah.ai/fragment/Jerusalem_NLI_4%C2%B09463_8" in urls
        assert not any("Heidelberg" in u for u in urls)  # '?' ids get a page but no sitemap entry
        assert len(urls) == 5

    @pytest.mark.parametrize("path", ["/sitemap-fragments-4.xml", "/sitemap-fragments-0.xml", "/sitemap-fragments-x.xml"])
    def test_out_of_range(self, client, path):
        assert client.get(path).status_code == 404

    def test_first_scan_failure_is_503(self, client, fake):
        fake.fail_catalogue = ESConnectionError("down")
        resp = client.get("/sitemap-fragments.xml")
        assert resp.status_code == 503 and resp.headers["retry-after"] == "300"
