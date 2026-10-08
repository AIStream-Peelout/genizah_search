"""Tests for canonical fragment URLs and the fragment sitemaps."""

from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import pytest
from elasticsearch import ConnectionError as ESConnectionError

from src.backend import fragment_sitemaps as fs
from src.backend.seo_urls import (
    CANONICAL_BASE,
    fragment_path,
    fragment_url,
    is_sitemap_eligible,
    sitemap_page_path,
)

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
ZWSP = "​"

# Real odd ids from genizah_merged_v8 (see the SEO plan's key facts).
ODD_IDS = [
    "Heidelberg_Cod_Heid_Or_??",
    "Cambridge_CUL_T_S_1*",
    "Jerusalem_NLI_4°9463_8",
    "Jerusalem_NLI_Sam_28=8",
    f"Cambridge_CUL_T_S_8J22{ZWSP}_22",
]


class FakeES:
    """In-memory ``search`` that sorts on ``doc_id`` and pages via ``search_after``."""

    def __init__(self, doc_ids: List[str], fail: Optional[Exception] = None) -> None:
        """Hold ids in memory.

        :param doc_ids: Ids held by the fake index (duplicates allowed).
        :param fail: Exception to raise from every ``search`` call, if any.
        """
        self.doc_ids = list(doc_ids)
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Return the next page of ids after ``search_after``.

        :param kwargs: Keyword arguments of ``Elasticsearch.search``.
        :return: Response shaped like ES with ``fields`` and ``sort`` per hit.
        """
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        ordered = sorted(self.doc_ids)
        if "search_after" in kwargs:
            ordered = [d for d in ordered if d > kwargs["search_after"][0]]
        page = ordered[: kwargs["size"]]
        return {"hits": {"hits": [
            {"_id": d, "_score": None, "fields": {"doc_id": [d]}, "sort": [d]} for d in page
        ]}}


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        """Start at t = 1000."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Read the clock.

        :return: Current fake time.
        """
        return self.now


def _locs(xml: str, tag: str) -> List[str]:
    """Parse sitemap XML and return its ``<loc>`` values.

    :param xml: Sitemap or sitemap index document.
    :param tag: Expected root tag without namespace (``urlset`` / ``sitemapindex``).
    :return: ``<loc>`` texts in document order.
    """
    root = ET.fromstring(xml.encode("utf-8"))
    assert root.tag == f"{{{NS['sm']}}}{tag}"
    return [el.text for el in root.iter(f"{{{NS['sm']}}}loc")]


# ----------------------------------------------------------------------
# seo_urls
# ----------------------------------------------------------------------


class TestSeoUrls:
    def test_canonical_base_is_fixed(self) -> None:
        """Canonical URLs always use https://cairogenizah.ai."""
        assert CANONICAL_BASE == "https://cairogenizah.ai"
        assert fragment_url("Cambridge_CUL_T_S_8J22_22") == (
            "https://cairogenizah.ai/fragment/Cambridge_CUL_T_S_8J22_22"
        )

    def test_degree_sign_and_equals_are_percent_encoded(self) -> None:
        """'°' and '=' in real ids are percent-encoded."""
        assert fragment_path("Jerusalem_NLI_4°9463_8") == "/fragment/Jerusalem_NLI_4%C2%B09463_8"
        assert fragment_path("Jerusalem_NLI_Sam_28=8") == "/fragment/Jerusalem_NLI_Sam_28%3D8"

    def test_nothing_is_kept_safe(self) -> None:
        """Slash, query, fragment, ampersand, space, star and ZWSP are all encoded."""
        assert fragment_path("a/b?c#d&e f*") == "/fragment/a%2Fb%3Fc%23d%26e%20f%2A"
        assert fragment_path(f"x{ZWSP}y") == "/fragment/x%E2%80%8By"

    @pytest.mark.parametrize("doc_id", ODD_IDS + ["Cambridge_CUL_T_S_8J22_22", "a/b", "קטע"])
    def test_path_round_trips(self, doc_id: str) -> None:
        """Decoding the path gives back the exact id.

        :param doc_id: Parametrized case.
        """
        assert unquote(fragment_path(doc_id)[len("/fragment/"):]) == doc_id

    def test_sitemap_page_path(self) -> None:
        """Sitemap file paths are 1-based and numbered."""
        assert sitemap_page_path(1) == "/sitemap-fragments-1.xml"
        assert sitemap_page_path(12) == "/sitemap-fragments-12.xml"

    @pytest.mark.parametrize("doc_id", [
        "Cambridge_CUL_T_S_8J22_22",
        "Jerusalem_NLI_4°9463_8",
        "Jerusalem_NLI_Sam_28=8",
        "Oxford_Bodleian_Bodl_MS_heb_d_66_137",
    ])
    def test_eligible(self, doc_id: str) -> None:
        """Ordinary ids and ids with '°' / '=' are listed.

        :param doc_id: Parametrized case.
        """
        assert is_sitemap_eligible(doc_id) is True

    @pytest.mark.parametrize("doc_id", [
        "",
        "Heidelberg_Cod_Heid_Or_??",
        "Cambridge_CUL_T_S_1*",
        f"Cambridge_CUL_T_S_8J22{ZWSP}_22",
        "x‌y",       # zero-width non-joiner
        "x‏y",       # right-to-left mark
        "﻿x",        # byte-order mark
        "x­y",       # soft hyphen
        "x\x00y",         # control
        "x\x7fy",         # DEL
        "x y",
        " x",
        "x\t",
        "x y",       # no-break space
        "x y",       # line separator
    ])
    def test_ineligible(self, doc_id: str) -> None:
        """Empty ids and ids with '?', '*', invisible, control or space chars are not listed.

        :param doc_id: Parametrized case.
        """
        assert is_sitemap_eligible(doc_id) is False


# ----------------------------------------------------------------------
# FragmentSitemaps
# ----------------------------------------------------------------------


class TestScan:
    def test_pages_through_search_after_with_ids_only(self) -> None:
        """The scan pages with search_after and fetches doc values only."""
        ids = [f"Doc_{i:02d}" for i in range(5)]
        es = FakeES(list(reversed(ids)))
        sm = fs.FragmentSitemaps(es, "genizah_merged_v8", batch_size=2)
        assert sm.scan_doc_ids() == ids
        assert len(es.calls) == 3  # 2 + 2 + 1, the short page ends the scan
        assert [c.get("search_after") for c in es.calls] == [None, ["Doc_01"], ["Doc_03"]]
        first = es.calls[0]
        assert first["index"] == "genizah_merged_v8"
        assert first["sort"] == [{"doc_id": "asc"}]
        assert first["source"] is False
        assert first["docvalue_fields"] == ["doc_id"]
        assert first["size"] == 2

    def test_exact_multiple_of_batch_needs_one_empty_page(self) -> None:
        """A full last page is followed by one empty request."""
        es = FakeES(["a", "b", "c", "d"])
        sm = fs.FragmentSitemaps(es, "idx", batch_size=2)
        assert sm.scan_doc_ids() == ["a", "b", "c", "d"]
        assert len(es.calls) == 3

    def test_default_batch_is_es_maximum(self) -> None:
        """Default scan batch is 10,000 hits."""
        es = FakeES(["a"])
        fs.FragmentSitemaps(es, "idx").scan_doc_ids()
        assert es.calls[0]["size"] == 10_000

    @pytest.mark.parametrize("kwargs", [
        {"urls_per_file": 0}, {"urls_per_file": 50_001}, {"batch_size": 0}, {"batch_size": 10_001},
    ])
    def test_limits_validated(self, kwargs: Dict[str, Any]) -> None:
        """Out-of-range file/batch sizes are rejected.

        :param kwargs: Parametrized case.
        """
        with pytest.raises(ValueError):
            fs.FragmentSitemaps(FakeES([]), "idx", **kwargs)


class TestRendering:
    IDS = ["Doc_c", "Doc_a", "Doc_b", "Doc_a", "Doc_e", "Doc_d"] + ODD_IDS

    def _sitemaps(self, **kwargs: Any) -> fs.FragmentSitemaps:
        """Sitemaps over :attr:`IDS` with two URLs per file."""
        return fs.FragmentSitemaps(FakeES(self.IDS), "idx", urls_per_file=2, **kwargs)

    def test_index_lists_every_file(self) -> None:
        """The index lists one entry per urlset file."""
        xml = self._sitemaps().index_xml()
        assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
        # 7 eligible ids (5 Doc_* + '°' + '=') at 2 per file -> 4 files
        assert _locs(xml, "sitemapindex") == [
            f"https://cairogenizah.ai/sitemap-fragments-{n}.xml" for n in range(1, 5)
        ]

    def test_pages_are_sorted_deduped_filtered_and_encoded(self) -> None:
        """Files hold sorted unique eligible ids, percent-encoded."""
        sm = self._sitemaps()
        all_locs = []
        for page in range(1, sm.page_count() + 1):
            locs = _locs(sm.page_xml(page), "urlset")
            assert 1 <= len(locs) <= 2
            all_locs.extend(locs)
        assert all_locs == [
            "https://cairogenizah.ai/fragment/Doc_a",
            "https://cairogenizah.ai/fragment/Doc_b",
            "https://cairogenizah.ai/fragment/Doc_c",
            "https://cairogenizah.ai/fragment/Doc_d",
            "https://cairogenizah.ai/fragment/Doc_e",
            "https://cairogenizah.ai/fragment/Jerusalem_NLI_4%C2%B09463_8",
            "https://cairogenizah.ai/fragment/Jerusalem_NLI_Sam_28%3D8",
        ]
        snap = sm.snapshot()
        assert snap.url_count == 7
        assert snap.excluded_doc_ids == (
            "Cambridge_CUL_T_S_1*",
            f"Cambridge_CUL_T_S_8J22{ZWSP}_22",
            "Heidelberg_Cod_Heid_Or_??",
        )
        joined = "".join(sm.page_xml(p) for p in range(1, 5))
        assert ZWSP not in joined and "%E2%80%8B" not in joined and "??" not in joined

    def test_no_lastmod(self) -> None:
        """No lastmod is emitted."""
        sm = self._sitemaps()
        assert "lastmod" not in sm.page_xml(1) and "lastmod" not in sm.index_xml()

    @pytest.mark.parametrize("page", [0, -1, 5, 99])
    def test_page_out_of_range_is_none(self, page: int) -> None:
        """Pages outside 1..N return None.

        :param page: Parametrized case.
        """
        assert self._sitemaps().page_xml(page) is None

    def test_base_url_is_xml_escaped_and_trailing_slash_dropped(self) -> None:
        """Every loc is XML-escaped; base URL trailing slash ignored."""
        sm = fs.FragmentSitemaps(FakeES(["a&b"]), "idx", base_url="https://example.org/x&y/")
        page = sm.page_xml(1)
        assert "<loc>https://example.org/x&amp;y/fragment/a%26b</loc>" in page
        assert _locs(page, "urlset") == ["https://example.org/x&y/fragment/a%26b"]
        assert _locs(sm.index_xml(), "sitemapindex") == [
            "https://example.org/x&y/sitemap-fragments-1.xml"
        ]

    def test_empty_index(self) -> None:
        """No ids gives an empty index and no pages."""
        sm = fs.FragmentSitemaps(FakeES([]), "idx")
        assert sm.page_count() == 0
        assert _locs(sm.index_xml(), "sitemapindex") == []
        assert sm.page_xml(1) is None

    def test_production_size_gives_two_files_within_limits(self) -> None:
        """~73.5k ids split into 45,000 + remainder under 50 MiB."""
        ids = [f"Cambridge_CUL_T_S_{i:06d}" for i in range(73_543)]
        sm = fs.FragmentSitemaps(FakeES(ids), "idx")
        assert sm.page_count() == 2
        first, second = sm.page_xml(1), sm.page_xml(2)
        assert first.count("<url>") == 45_000
        assert second.count("<url>") == 28_543
        assert len(first.encode("utf-8")) < fs.MAX_SITEMAP_BYTES
        assert "sitemap-fragments-3.xml" not in sm.index_xml()


class TestChunking:
    def test_byte_budget_splits_before_url_limit(self) -> None:
        """The byte budget splits files before the URL limit is hit."""
        entries = [fs.url_entry(f"https://x.org/fragment/{i}") for i in range(10)]
        overhead = len(fs.render_urlset([]).encode("utf-8"))
        per = len(entries[0].encode("utf-8"))
        groups = fs.chunk_entries(entries, max_urls=100, max_bytes=overhead + 3 * per)
        assert [len(g) for g in groups] == [3, 3, 3, 1]
        assert all(len(fs.render_urlset(g).encode("utf-8")) <= overhead + 3 * per for g in groups)

    def test_entry_larger_than_budget_rejected(self) -> None:
        """An entry that cannot fit any file is an error."""
        with pytest.raises(ValueError):
            fs.chunk_entries([fs.url_entry("https://x.org/" + "a" * 500)], max_urls=10, max_bytes=200)


class TestCache:
    def test_es_called_once_within_ttl(self) -> None:
        """All calls within the TTL share one scan."""
        es = FakeES(["a", "b", "c"])
        clock = FakeClock()
        sm = fs.FragmentSitemaps(es, "idx", urls_per_file=2, ttl_seconds=60, clock=clock)
        sm.index_xml()
        sm.page_xml(1)
        sm.page_xml(2)
        sm.page_xml(3)
        clock.now += 59
        sm.index_xml()
        assert len(es.calls) == 1

    def test_refresh_after_ttl_picks_up_new_ids(self) -> None:
        """After the TTL a rescan picks up new ids."""
        es = FakeES(["a"])
        clock = FakeClock()
        sm = fs.FragmentSitemaps(es, "idx", ttl_seconds=60, clock=clock)
        assert _locs(sm.page_xml(1), "urlset") == ["https://cairogenizah.ai/fragment/a"]
        es.doc_ids.append("b")
        clock.now += 61
        assert _locs(sm.page_xml(1), "urlset") == [
            "https://cairogenizah.ai/fragment/a", "https://cairogenizah.ai/fragment/b",
        ]
        assert len(es.calls) == 2

    def test_failed_refresh_serves_previous_snapshot_then_retries_later(self) -> None:
        """A failed refresh keeps the old snapshot and backs off."""
        es = FakeES(["a"])
        clock = FakeClock()
        sm = fs.FragmentSitemaps(es, "idx", ttl_seconds=60, clock=clock)
        before = sm.page_xml(1)
        es.fail = ESConnectionError("down")
        clock.now += 61
        assert sm.page_xml(1) == before
        assert len(es.calls) == 2
        clock.now += fs.RETRY_AFTER_FAILURE_S - 1
        sm.page_xml(1)
        assert len(es.calls) == 2  # back-off: no retry yet
        es.fail = None
        clock.now += 2
        sm.page_xml(1)
        assert len(es.calls) == 3

    def test_first_scan_failure_is_raised(self) -> None:
        """With nothing cached, a scan failure propagates."""
        sm = fs.FragmentSitemaps(FakeES(["a"], fail=ESConnectionError("down")), "idx")
        with pytest.raises(ESConnectionError):
            sm.index_xml()
