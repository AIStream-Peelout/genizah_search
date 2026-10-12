"""Tests for the shared "View at the source" link builder (``src/backend/source_links.py``)."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from src.backend import source_links as sl
from src.backend.search_service import ElasticsearchService

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_PATH = REPO_ROOT / "docs" / "planned_features" / "source_links" / "sample_links.jsonl"
# Links in the handoff samples that need the PGP fragments.csv (not in the index).
NOT_IN_INDEX = ("other_inst", "bodleian_digital")
PGP_URL = "https://geniza.princeton.edu/en/documents/{}/"


def _samples() -> List[Dict[str, Any]]:
    """The 32 handoff sample records (4 per link source).

    :return: Rows with ``cid``, ``sm`` and ``links`` (``[source, url]`` pairs).
    """
    with SAMPLES_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


SAMPLES = _samples()


def _source_from_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    """An index ``_source`` holding what the sample's links were made from.

    :param row: Sample row.
    :return: ``_source`` with shelf mark, canonical id, pgpids, Bodleian URL and KTIV manifest.
    """
    source: Dict[str, Any] = {
        "doc_id": row["cid"],
        "canonical_id": row["cid"],
        "shelf_mark": row["sm"],
        "tei_metadata": {"pgpids": []},
    }
    for kind, url in row["links"]:
        if kind == "pgp":
            source["tei_metadata"]["pgpids"].append(re.search(r"/documents/(\d+)/", url).group(1))
        elif kind == "ktiv":
            sysnum = re.search(r"PNX_MANUSCRIPTS(\d+)", url).group(1)
            source["ktiv_iiif_manifest_url"] = f"https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS{sysnum}-1/manifest"
        elif kind == "bodleian":
            source["bodleian_catalogue_url"] = url
    return source


def _expected_from_sample(row: Dict[str, Any], ktiv: bool) -> List[Tuple[str, str]]:
    """The ``(source, url)`` pairs the serving side can derive for a sample.

    :param row: Sample row.
    :param ktiv: Whether KTIV links are enabled.
    :return: Expected pairs, in the samples' order (holder, KTIV, PGP).
    """
    expected: List[Tuple[str, str]] = []
    for kind, url in row["links"]:
        if kind in NOT_IN_INDEX:
            continue
        if kind == "penn" and "colenda" in url:  # PGP's Colenda page; the index gives the Halper rule
            url = sl.OPENN_HALPER_URL.format(row["cid"].rsplit("_", 1)[1])
        if kind == "ktiv":
            if not ktiv:
                continue
            url = url.replace("vid=MANUSCRIPTS", "vid=KTIV&scope=KTIV")  # the shape to be confirmed
        expected.append((kind, url))
    return expected


def _pairs(links: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """``(source, url)`` of each link.

    :param links: Output of :func:`sl.build_source_links`.
    :return: Pairs in order.
    """
    return [(link["source"], link["url"]) for link in links]


# ---------------------------------------------------------------------------
# Handoff samples
# ---------------------------------------------------------------------------


def test_samples_cover_every_source() -> None:
    kinds = {kind for row in SAMPLES for kind, _ in row["links"]}
    assert len(SAMPLES) == 32
    assert {"cudl", "bodleian", "manchester", "penn", "ktiv", "pgp"} <= kinds


@pytest.mark.parametrize("row", SAMPLES, ids=[row["cid"] for row in SAMPLES])
def test_samples_ktiv_off(row: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", False)
    assert _pairs(sl.build_source_links(_source_from_sample(row))) == _expected_from_sample(row, ktiv=False)


@pytest.mark.parametrize("row", SAMPLES, ids=[row["cid"] for row in SAMPLES])
def test_samples_ktiv_on(row: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", True)
    assert _pairs(sl.build_source_links(_source_from_sample(row))) == _expected_from_sample(row, ktiv=True)


def test_link_shape_labels_and_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", True)
    by_source = {}
    for row in SAMPLES:
        for link in sl.build_source_links(_source_from_sample(row)):
            assert set(link) == {"source", "label", "url", "kind"}
            by_source[link["source"]] = (link["label"], link["kind"])
    assert by_source == {
        "cudl": ("View on Cambridge Digital Library", "item"),
        "bodleian": ("View in the Bodleian catalogue", "catalogue"),
        "manchester": ("Search Manchester Digital Collections", "search"),
        "penn": ("View on OPenn (Penn Libraries)", "item"),
        "ktiv": ("View on KTIV (National Library of Israel)", "item"),
        "pgp": ("View on Princeton Geniza Project", "item"),
    }


# ---------------------------------------------------------------------------
# Ordering, de-duplication, KTIV switch
# ---------------------------------------------------------------------------

KTIV_MANIFEST = "https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS990051232320205171-1/manifest"
KTIV_URL = (
    "https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage"
    "?vid=KTIV&scope=KTIV&docId=PNX_MANUSCRIPTS990051232320205171"
)


def test_holder_first_then_ktiv_then_pgp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", True)
    source = {
        "doc_id": "Cambridge_CUL_T_S_8J41_4",
        "shelf_mark": "T-S 8J41.4",
        "tei_metadata": {"pgpids": ["3868", "3867"], "ktiv_iiif_manifest_url": KTIV_MANIFEST},
    }
    assert _pairs(sl.build_source_links(source)) == [
        ("cudl", "https://cudl.lib.cam.ac.uk/view/MS-TS-00008-J-00041-00004"),
        ("ktiv", KTIV_URL),
        ("pgp", PGP_URL.format("3868")),
        ("pgp", PGP_URL.format("3867")),
    ]


def test_ktiv_is_on_by_default() -> None:
    """The KTIV URL shape was confirmed in a browser (2026-10-12), so KTIV links ship."""
    assert sl.KTIV_LINKS_ENABLED is True
    source = {"doc_id": "New_York_JTS_ENA_409_8", "shelf_mark": "ENA 409.8", "ktiv_iiif_manifest_url": KTIV_MANIFEST}
    assert _pairs(sl.build_source_links(source)) == [("ktiv", KTIV_URL)]


def test_ktiv_switch_off_hides_ktiv_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the switch to False removes KTIV links (derived or indexed)."""
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", False)
    source = {"doc_id": "New_York_JTS_ENA_409_8", "shelf_mark": "ENA 409.8", "ktiv_iiif_manifest_url": KTIV_MANIFEST}
    assert sl.build_source_links(source) == []


@pytest.mark.parametrize("url", [
    "https://fjms.genizah.org/",
    "https://fjms.genizah.org/index.html?type=FragmentPage&id=123",
    "http://FJMS.Genizah.org/x",
    "https://fgp.genizah.org/FunctionPages/Bibliography/1",
    "https://www.jewishmanuscripts.org/item/9",
])
def test_old_friedberg_sites_are_never_linked(url: str) -> None:
    """No link to the old Friedberg sites survives, whatever field it comes from."""
    assert sl.is_blocked_url(url)
    indexed = {"doc_id": "x", "shelf_mark": "T-S 8J41.4", "source_links": [
        {"source": "other", "label": "View elsewhere", "url": url, "kind": "item"},
        {"source": "pgp", "label": "View on Princeton Geniza Project",
         "url": "https://geniza.princeton.edu/en/documents/1/", "kind": "item"},
    ]}
    urls = [link["url"] for link in sl.build_source_links(indexed)]
    assert url not in urls
    assert urls == ["https://geniza.princeton.edu/en/documents/1/"]
    assert not sl.is_blocked_url("https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage?vid=KTIV")
    assert not sl.is_blocked_url("https://cairogenizah.org/")


@pytest.mark.parametrize(
    "manifest, sysnum",
    [
        (KTIV_MANIFEST, "990051232320205171"),
        ("https://iiif.nli.org.il/IIIFv21/DOCID/990001492500205171-1/manifest", "990001492500205171"),  # no PNX_ prefix
        ("https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS12345-1/manifest", None),  # too short
        ("", None),
    ],
)
def test_ktiv_sysnum(manifest: str, sysnum: Any) -> None:
    assert sl.ktiv_sysnum({"ktiv_iiif_manifest_url": manifest}) == sysnum
    assert sl.ktiv_sysnum({"tei_metadata": {"ktiv_iiif_manifest_url": manifest}}) == sysnum


def test_duplicates_are_dropped() -> None:
    source = {"doc_id": "X_1", "shelf_mark": "X 1", "tei_metadata": {"pgpids": ["2747", "2747", " 2747 "]}}
    assert _pairs(sl.build_source_links(source)) == [("pgp", PGP_URL.format("2747"))]
    indexed = {"source_links": [{"source": "pgp", "url": PGP_URL.format("1")}, {"source": "pgp", "url": PGP_URL.format("1")}]}
    assert len(sl.build_source_links(indexed)) == 1


def test_bad_pgpids_are_skipped() -> None:
    source = {"doc_id": "X_1", "tei_metadata": {"pgpids": ["abc", "", None, "12/../x", 42]}}
    assert _pairs(sl.build_source_links(source)) == [("pgp", PGP_URL.format("42"))]
    assert _pairs(sl.build_source_links({"doc_id": "X_1", "tei_metadata": {"pgpids": "7"}})) == [("pgp", PGP_URL.format("7"))]


def test_bodleian_url_from_tei_metadata() -> None:
    url = "https://hebrew.bodleian.ox.ac.uk/catalog/volume_5#MS_Heb_a_3-part40"
    source = {"doc_id": "Oxford_Bodleian_Bodl_MS_heb_a_3_40", "tei_metadata": {"bodleian_catalogue_url": url}}
    assert sl.build_source_links(source) == [
        {"source": "bodleian", "label": "View in the Bodleian catalogue", "url": url, "kind": "catalogue"}
    ]


def test_empty_record() -> None:
    assert sl.build_source_links({}) == []
    assert sl.build_source_links({"doc_id": "Paris_AIU_XII_143", "shelf_mark": "AIU XII.143", "tei_metadata": None}) == []


# ---------------------------------------------------------------------------
# Blocked hosts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, blocked",
    [
        ("genizah.org", True),
        ("fgp.genizah.org", True),
        ("FGP.Genizah.ORG", True),
        ("www.jewishmanuscripts.org", True),
        ("jewishmanuscripts.org.", True),
        ("genizah.bodleian.ox.ac.uk", False),
        ("cairogenizah.org", False),
        ("cairogenizah.ai", False),
        ("geniza.princeton.edu", False),
        ("", False),
    ],
)
def test_is_blocked_host(host: str, blocked: bool) -> None:
    assert sl.is_blocked_host(host) is blocked


def test_indexed_links_pass_the_same_filter() -> None:
    source = {
        "doc_id": "Cambridge_CUL_T_S_10J2_11",
        "shelf_mark": "T-S 10J2.11",
        "tei_metadata": {"pgpids": ["7753"]},
        "source_links": [
            {"source": "other", "label": "Join record", "url": "https://fgp.genizah.org/GeneralPages/Join/1", "kind": "item"},
            {"source": "other", "url": "http://www.jewishmanuscripts.org/item/9"},
            {"source": "cudl", "url": "https://cudl.lib.cam.ac.uk/view/MS-TS-00010-J-00002-00011", "kind": "weird"},
            {"source": "manchester", "url": "https://luna.manchester.ac.uk/luna/servlet/view/search?q=x", "kind": "search"},
            {"source": "other", "label": "", "url": "https://www.bl.uk/manuscripts/x"},
            {"source": "bad", "url": "javascript:alert(1)"},
            {"url": None},
            "not a dict",
        ],
    }
    assert sl.build_source_links(source) == [
        {"source": "cudl", "label": "View on Cambridge Digital Library",
         "url": "https://cudl.lib.cam.ac.uk/view/MS-TS-00010-J-00002-00011", "kind": "item"},
        {"source": "manchester", "label": "Search Manchester Digital Collections",
         "url": "https://luna.manchester.ac.uk/luna/servlet/view/search?q=x", "kind": "search"},
        {"source": "other", "label": "View at bl.uk", "url": "https://www.bl.uk/manuscripts/x", "kind": "item"},
    ]


def test_indexed_list_replaces_the_derivation() -> None:
    source = {"doc_id": "Cambridge_CUL_T_S_8_108", "shelf_mark": "T-S 8.108", "tei_metadata": {"pgpids": ["2747"]}}
    assert len(sl.build_source_links(source)) == 2
    assert sl.build_source_links({**source, "source_links": []}) == []


def test_indexed_ktiv_links_follow_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"source_links": [{"source": "ktiv", "url": KTIV_URL}]}
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", False)
    assert sl.build_source_links(source) == []
    monkeypatch.setattr(sl, "KTIV_LINKS_ENABLED", True)
    assert _pairs(sl.build_source_links(source)) == [("ktiv", KTIV_URL)]


def test_derived_url_on_a_blocked_host_is_dropped() -> None:
    source = {"doc_id": "Oxford_X", "bodleian_catalogue_url": "https://fgp.genizah.org/catalog/1"}
    assert sl.build_source_links(source) == []
    kept = {"doc_id": "Oxford_X", "bodleian_catalogue_url": "https://genizah.bodleian.ox.ac.uk/catalog/volume_5#p1"}
    assert _pairs(sl.build_source_links(kept)) == [("bodleian", kept["bodleian_catalogue_url"])]


@pytest.mark.parametrize(
    "url, expected",
    [
        (" https://geniza.princeton.edu/en/documents/1/ ", "https://geniza.princeton.edu/en/documents/1/"),
        ("https://fgp.genizah.org/x", None),
        ("ftp://example.org/x", None),
        ("javascript:alert(1)", None),
        ("geniza.princeton.edu/x", None),
        ("", None),
        (None, None),
        (42, None),
    ],
)
def test_public_url(url: Any, expected: Any) -> None:
    assert sl.public_url(url) == expected


def test_bibliography_url_filter_uses_the_shared_blocklist() -> None:
    service = ElasticsearchService.__new__(ElasticsearchService)
    assert service._public_bibliography_url("https://fgp.genizah.org/FunctionPages/Bibliography/1") is None
    assert service._public_bibliography_url("https://www.jstor.org/stable/1") == "https://www.jstor.org/stable/1"
    assert not hasattr(ElasticsearchService, "_BLOCKED_BIBLIOGRAPHY_URL_HOSTS")


# ---------------------------------------------------------------------------
# CUDL rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shelf_mark, cudl_id",
    [
        ("T-S 10J2.11", "MS-TS-00010-J-00002-00011"),
        ("T-S 8.108", "MS-TS-00008-00108"),
        ("T-S K25.157", "MS-TS-K-00025-00157"),
        ("T-S NS 320.64a", "MS-TS-NS-00320-00064-A"),
        ("T-S NS J401", "MS-TS-NS-J-00401"),
        ("T-S 8Ja2.1", "MS-TS-00008-J-A-00002-00001"),
        ("T-S Ka1.1", "MS-TS-KA-00001-00001"),
        ("T-S Misc.24.137", "MS-TS-MISC-00024-00137"),
        # Parentheses are kept (the canonical id drops them, so the display mark is used)
        ("T-S Ar.18(1).1", "MS-TS-AR-00018-00001-00001"),
        ("T-S F2(1).60", "MS-TS-F-00002-00001-00060"),
        ("T-S F 1(2).74", "MS-TS-F-00001-00002-00074"),
        # Shelfmark styles
        ("Cambridge CUL: T-S AS 7.101", "MS-TS-AS-00007-00101"),
        ("Cambridge University Library, Cambridge, England Ms. T-S F 7.49", "MS-TS-F-00007-00049"),
        ("Or.1081 2.40", "MS-OR-01081-00002-00040"),
        ("Add.3335", "MS-ADD-03335"),
        # Mosseri: roman volume uppercased
        ("Moss. VII,5", "MS-MOSSERI-VII-00005"),
        ("Mosseri Ia,1", "MS-MOSSERI-IA-00001"),
        # Lewis-Gibson
        ("L-G Ar. I.11", "MS-LG-ARABIC-00001-00011"),
        ("L-G Misc. 51", "MS-LG-MISC-00051"),
        ("Cambridge Lewis-Gibson: L-G Bib. II.104", "MS-LG-BIBLE-00002-00104"),
        # Joins, ranges and broken marks get no id
        ("T-S 8J4.1-3", None),
        ("T-S 13J1.1 + T-S 13J1.2", None),
        ("T-S NS 9.56 and 57", None),
        ("T-S 12.843; T-S 12.844", None),
        ("T-S AS 145.minute fragments", None),
        ("Or.1080 10.20 Cambridge Lewis", None),
        ("ENA 409.8", None),
    ],
)
def test_cudl_rule(shelf_mark: str, cudl_id: Any) -> None:
    assert sl.cudl_from_shelfmark(shelf_mark) == cudl_id
    assert sl.cudl_item_id(shelf_mark) == cudl_id


# Expected ids below are the CUDL URLs PGP's fragments.csv gives for these shelfmarks.


@pytest.mark.parametrize(
    "shelf_mark, cudl_id",
    [
        ("T-S Ar.30.184.14", "MS-TS-AR-00030-00184-P-00014"),
        ("T-S Ar.30.184.5", "MS-TS-AR-00030-00184-P-00005"),
        ("Cambridge CUL: T-S Ar.30.184.1", "MS-TS-AR-00030-00184-P-00001"),
        ("T-S Ar. 30.184.20", "MS-TS-AR-00030-00184-P-00020"),
    ],
)
def test_cudl_ar_30_184_items_get_p_ids(shelf_mark: str, cudl_id: str) -> None:
    """Handoff "Known exceptions": T-S Ar.30.184.N -> …-P-000NN (the plain rule omits the "P")."""
    assert sl.cudl_from_shelfmark(shelf_mark) != cudl_id
    assert sl.cudl_item_id(shelf_mark) == cudl_id
    assert _pairs(sl.build_source_links({"doc_id": "Cambridge_CUL_X", "shelf_mark": shelf_mark})) == [
        ("cudl", f"https://cudl.lib.cam.ac.uk/view/{cudl_id}")
    ]


@pytest.mark.parametrize(
    "shelf_mark, parent_id",
    [
        ("Cambridge CUL: T-S 8J4.3.1", "MS-TS-00008-J-00004-00003"),
        ("T-S 8J4.3.2", "MS-TS-00008-J-00004-00003"),
        ("T-S 10J27.3.2", "MS-TS-00010-J-00027-00003"),
        ("T-S 13J4.15.1", "MS-TS-00013-J-00004-00015"),
        ("T-S Misc.28.79.1", "MS-TS-MISC-00028-00079"),
        ("T-S Ar.40.142.2", "MS-TS-AR-00040-00142"),
        ("T-S Ar.54.20.3", "MS-TS-AR-00054-00020"),
    ],
)
def test_cudl_parent_linked_sub_items_link_the_parent(shelf_mark: str, parent_id: str) -> None:
    """Handoff: these third-level sub-items live on their parent's CUDL page."""
    assert sl.cudl_from_shelfmark(shelf_mark) == parent_id + "-" + shelf_mark.rsplit(".", 1)[1].zfill(5)
    assert sl.cudl_item_id(shelf_mark) == parent_id


@pytest.mark.parametrize(
    "shelf_mark, cudl_id",
    [
        ("T-S K25.240.40", "MS-TS-K-00025-00240-00040"),
        ("T-S Misc.27.4.11", "MS-TS-MISC-00027-00004-00011"),
        ("T-S Misc.24.137.4", "MS-TS-MISC-00024-00137-00004"),
        ("T-S 12.59.1", "MS-TS-00012-00059-00001"),
        ("T-S NS 246.26.12", "MS-TS-NS-00246-00026-00012"),
    ],
)
def test_other_third_level_items_keep_their_own_id(shelf_mark: str, cudl_id: str) -> None:
    """Outside the parent-linked boxes, a third-level shelfmark has its own CUDL page."""
    assert sl.cudl_from_shelfmark(shelf_mark) == cudl_id
    assert sl.cudl_item_id(shelf_mark) == cudl_id


@pytest.mark.parametrize("shelf_mark", ["T-S Ar.30.184.3a", "T-S Ar.30.184.1-2", "T-S Ar.30.184.1 + T-S Ar.30.184.2"])
def test_odd_ar_30_184_marks_get_no_link(shelf_mark: str) -> None:
    """No guessed id for an Ar.30.184 mark that is not a plain item number."""
    assert sl.cudl_item_id(shelf_mark) is None
    assert sl.build_source_links({"doc_id": "Cambridge_CUL_X", "shelf_mark": shelf_mark}) == []


def test_cudl_only_for_cambridge_holders() -> None:
    """The holder is the canonical id prefix, not the (free-text) shelf mark or institution."""
    jts = {"doc_id": "New_York_JTS_T_S_AS_62_379", "canonical_id": "New_York_JTS_T_S_AS_62_379",
           "shelf_mark": "T-S AS 62.379", "institution": "Cambridge University Library"}
    assert sl.build_source_links(jts) == []
    cambridge = {"doc_id": "x", "canonical_id": "Cambridge_CUL_T_S_AS_62_379", "shelf_mark": "T-S AS 62.379"}
    assert _pairs(sl.build_source_links(cambridge)) == [("cudl", "https://cudl.lib.cam.ac.uk/view/MS-TS-AS-00062-00379")]


# ---------------------------------------------------------------------------
# Manchester and Penn rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shelf_mark, query",
    [
        ("Manchester: A 80", "%22A+80%22"),
        ("JRL Genizah Ar. 814", "%22Ar.+814%22"),
        ("The University of Manchester Library, Manchester, England Ms. B 8539a", "%22B+8539a%22"),
        ("JRL Gaster heb. ms 1", None),
        ("A 80", None),
    ],
)
def test_manchester_luna_search(shelf_mark: str, query: Any) -> None:
    links = sl.build_source_links({"doc_id": "Manchester_JRL_X", "shelf_mark": shelf_mark})
    assert _pairs(links) == ([("manchester", sl.LUNA_SEARCH_URL.format(query))] if query else [])


@pytest.mark.parametrize(
    "shelf_mark, number",
    [("Halper 367", "367"), ("Philadelphia Penn CAJS: Halper 230", "230"), ("Philadelphia Penn U.: E 16450", None)],
)
def test_penn_halper(shelf_mark: str, number: Any) -> None:
    links = sl.build_source_links({"doc_id": "Philadelphia_CAJS_X", "shelf_mark": shelf_mark})
    assert _pairs(links) == ([("penn", f"https://openn.library.upenn.edu/Data/0002/html/h{number}.html")] if number else [])


# ---------------------------------------------------------------------------
# /book-info labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, label",
    [
        ("https://www.jstor.org/stable/1", "View on JSTOR"),
        ("https://doi.org/10.1/x", "View via DOI"),
        ("https://search.worldcat.org/isbn/123", "View at search.worldcat.org"),
        ("www.academia.edu/123", "View on Academia.edu"),
        ("https://fgp.genizah.org/FunctionPages/Bibliography/1", None),
        ("jewishmanuscripts.org/item/1", None),
        ("https://fgp.genizah.org?id=1", None),
        ("https://user@fgp.genizah.org:443/x", None),
        (None, None),
        ("", None),
    ],
)
def test_link_label(url: Any, label: Any) -> None:
    assert sl.link_label(url) == label
    assert sl.is_blocked_url(url) is (bool(url) and label is None)
