"""Keyword search's machine-reading second pass: AI-only matches strictly after every catalogue hit."""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest

from src.backend.search_service import (
    MACHINE_READING,
    MAX_RESULT_WINDOW,
    ElasticsearchService,
    SearchResponse,
    machine_reading_window,
)

CATALOGUE = "genizah_merged_v8"
HEBREW = "ירושלים"


def _cat_ids(n: int) -> List[str]:
    """Catalogue doc ids ``C00``, ``C01``, …

    :param n: How many.
    :return: Ids in rank order.
    """
    return [f"C{i:02d}" for i in range(n)]


def _ai_hit(doc_id: str, image_index: int = 0, line_index: int = 3) -> Dict[str, Any]:
    """One hit in the shape :meth:`ConsensusService.search_uncatalogued` returns.

    :param doc_id: Catalogue doc id.
    :param image_index: Image of the read.
    :param line_index: Matched agreed line.
    :return: Hit dict.
    """
    return {
        "doc_id": doc_id,
        "ai_doc_id": doc_id,
        "image_index": image_index,
        "line_index": line_index,
        "snippet": f"שורה של {doc_id}",
        "read_url": f"/read?doc={doc_id}&image={image_index}&line={line_index}&model=m",
        "vlm_model": "m",
        "score": 1.0,
    }


class FakeES:
    """Catalogue index double: the main keyword search, the candidate check and mget.

    :param catalogue_ids: Ids the catalogue query matches, in rank order.
    :param existing: Ids that exist in the catalogue (default: catalogue ids and ``A*`` ids).
    :param filtered_out: Ids the active filters exclude in the candidate check.
    :param relation: ``hits.total.relation`` of the main search.
    """

    def __init__(
        self,
        catalogue_ids: Sequence[str],
        existing: Optional[Set[str]] = None,
        filtered_out: Optional[Set[str]] = None,
        relation: str = "eq",
    ) -> None:
        """Store the fixture data.

        :param catalogue_ids: Ids the catalogue query matches, in rank order.
        :param existing: Ids that exist in the catalogue.
        :param filtered_out: Ids the active filters exclude.
        :param relation: Total relation of the main search.
        """
        self.catalogue_ids = list(catalogue_ids)
        self.existing = existing
        self.filtered_out = filtered_out or set()
        self.relation = relation
        self.searches: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.mgets: List[Dict[str, Any]] = []

    def _exists(self, doc_id: str) -> bool:
        """Whether a catalogue record exists.

        :param doc_id: Catalogue id.
        :return: Existence.
        """
        return doc_id in self.existing if self.existing is not None else True

    @staticmethod
    def _source(doc_id: str) -> Dict[str, Any]:
        """Catalogue ``_source`` of a record.

        :param doc_id: Catalogue id.
        :return: Minimal source.
        """
        return {"doc_id": doc_id, "shelf_mark": f"SM {doc_id}", "description": f"Description of {doc_id}"}

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Answer the main keyword search or the candidate check.

        :param kwargs: Search arguments.
        :return: Response dict.
        """
        query = kwargs["query"]
        filters = query.get("bool", {}).get("filter") or []
        ids_clause = next((f["ids"] for f in filters if isinstance(f, dict) and "ids" in f), None)
        if ids_clause is not None:
            self.checks.append(kwargs)
            hits = []
            for doc_id in ids_clause["values"]:
                if not self._exists(doc_id) or doc_id in self.filtered_out:
                    continue
                hit: Dict[str, Any] = {"_id": doc_id, "_score": 0.0}
                if doc_id in self.catalogue_ids:
                    hit["matched_queries"] = ["catalogue_match"]
                hits.append(hit)
            return {"hits": {"total": {"value": len(hits), "relation": "eq"}, "hits": hits}}
        self.searches.append(kwargs)
        start, size = kwargs["from_"], kwargs["size"]
        page = self.catalogue_ids[start:start + size]
        hits = [{"_id": d, "_score": 10.0 - i * 0.1, "_source": self._source(d)} for i, d in enumerate(page)]
        total = {"value": len(self.catalogue_ids), "relation": self.relation}
        return {"hits": {"total": total, "hits": hits}}

    def mget(self, **kwargs: Any) -> Dict[str, Any]:
        """Return catalogue records by ``_id``.

        :param kwargs: ``index``, ``ids``, ``source_excludes``.
        :return: ``{"docs": [...]}``.
        """
        self.mgets.append(kwargs)
        docs = []
        for doc_id in kwargs["ids"]:
            if self._exists(doc_id):
                docs.append({"_id": doc_id, "found": True, "_source": self._source(doc_id)})
            else:
                docs.append({"_id": doc_id, "found": False})
        return {"docs": docs}


class FakeConsensus:
    """Stands in for :class:`ConsensusService` (only what keyword search uses).

    :param hits: Hits in AI rank order.
    :param error: Exception to raise instead of answering.
    """

    def __init__(self, hits: Sequence[Dict[str, Any]], error: Optional[Exception] = None) -> None:
        """Store the fixture data.

        :param hits: Hits in AI rank order.
        :param error: Exception to raise instead of answering.
        """
        self.catalogue_index = CATALOGUE
        self.hits = list(hits)
        self.error = error
        self.calls: List[Tuple[str, int, int]] = []

    def search_uncatalogued(self, query: str, size: int, offset: int) -> Tuple[int, List[Dict[str, Any]]]:
        """Return a slice of the configured hits.

        :param query: Query.
        :param size: Page size.
        :param offset: Hits to skip.
        :return: ``(total, hits)``.
        """
        self.calls.append((query, size, offset))
        if self.error is not None:
            raise self.error
        return len(self.hits), self.hits[offset:offset + size]


def _service(es: FakeES, consensus: Optional[FakeConsensus]) -> ElasticsearchService:
    """Service over fakes, built without ``__init__`` (no network client).

    :param es: Catalogue double.
    :param consensus: Consensus double or ``None``.
    :return: Service.
    """
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.es = es
    svc.index_name = CATALOGUE
    svc.consensus_service = consensus
    return svc


def _search(
    svc: ElasticsearchService, page: int, size: int = 10, query: str = HEBREW,
    filters: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None,
) -> SearchResponse:
    """Run one keyword search page.

    :param svc: Service.
    :param page: 1-based page.
    :param size: Page size.
    :param query: Query.
    :param filters: Request filters.
    :param index_name: Index override.
    :return: Response.
    """
    req = SimpleNamespace(query=query, filters=filters, num_results=size, page=page)
    return asyncio.run(svc.search_by_keyword(req, index_name))


def _kinds(resp: SearchResponse) -> str:
    """``C`` per catalogue result and ``M`` per machine-reading result.

    :param resp: Response.
    :return: Kind string in result order.
    """
    return "".join("M" if r.match_source == MACHINE_READING else "C" for r in resp.results)


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the kill switch and the AI master switch to on regardless of the environment.

    :param monkeypatch: pytest fixture.
    """
    monkeypatch.setenv("CONSENSUS_SEARCH_ENABLED", "true")
    monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)


class TestWindow:
    def test_page_inside_catalogue(self) -> None:
        assert machine_reading_window(0, 10, 25) is None
        assert machine_reading_window(10, 10, 20) is None  # ends exactly on the last catalogue hit

    def test_page_straddling(self) -> None:
        assert machine_reading_window(20, 10, 25) == (0, 5)

    def test_page_starting_at_boundary(self) -> None:
        assert machine_reading_window(20, 10, 20) == (0, 10)

    def test_page_past_catalogue(self) -> None:
        assert machine_reading_window(30, 10, 25) == (5, 10)
        assert machine_reading_window(40, 10, 25) == (15, 10)

    def test_no_catalogue_hits(self) -> None:
        assert machine_reading_window(0, 10, 0) == (0, 10)
        assert machine_reading_window(10, 10, 0) == (10, 10)


class TestPaging:
    def _setup(self, n_catalogue: int = 25, n_ai: int = 20) -> Tuple[ElasticsearchService, FakeES, FakeConsensus]:
        """Service with ``n_catalogue`` catalogue hits and ``n_ai`` distinct AI-only fragments.

        :param n_catalogue: Catalogue hits.
        :param n_ai: Machine-reading hits.
        :return: ``(service, es, consensus)``.
        """
        es = FakeES(_cat_ids(n_catalogue))
        consensus = FakeConsensus([_ai_hit(f"A{i:02d}") for i in range(n_ai)])
        return _service(es, consensus), es, consensus

    def test_page_inside_catalogue_has_no_ai_hits_but_counts_them(self) -> None:
        svc, es, _ = self._setup()
        resp = _search(svc, page=1)
        assert _kinds(resp) == "C" * 10
        assert resp.total == 45 and resp.total_pages == 5 and resp.has_more is True
        assert es.mgets == []  # nothing to build on this page

    def test_page_straddling_boundary(self) -> None:
        svc, es, _ = self._setup()
        resp = _search(svc, page=3)
        assert _kinds(resp) == "C" * 5 + "M" * 5
        assert [r.doc_id for r in resp.results[5:]] == ["A00", "A01", "A02", "A03", "A04"]
        assert resp.has_more is True

    def test_page_past_catalogue_uses_ai_offset(self) -> None:
        svc, _, _ = self._setup()
        resp = _search(svc, page=4)
        assert _kinds(resp) == "M" * 10
        assert [r.doc_id for r in resp.results] == [f"A{i:02d}" for i in range(5, 15)]
        last = _search(svc, page=5)
        assert [r.doc_id for r in last.results] == [f"A{i:02d}" for i in range(15, 20)]
        assert last.has_more is False and last.total == 45

    def test_machine_reading_hits_always_follow_catalogue_hits(self) -> None:
        svc, _, _ = self._setup(n_catalogue=23, n_ai=9)
        seen: List[str] = []
        kinds = ""
        for page in range(1, 6):
            resp = _search(svc, page=page, size=7)
            kinds += _kinds(resp)
            seen.extend(r.doc_id for r in resp.results)
        assert kinds == "C" * 23 + "M" * 9
        assert len(seen) == len(set(seen)) == 32

    def test_ai_search_and_check_are_cached_per_query(self) -> None:
        svc, es, consensus = self._setup()
        for page in (1, 3, 4):
            _search(svc, page=page)
        assert len(consensus.calls) == 1
        assert len(es.checks) == 1
        _search(svc, page=1, query="שלום")
        assert len(consensus.calls) == 2

    def test_result_shape(self) -> None:
        svc, es, _ = self._setup(n_catalogue=0, n_ai=2)
        resp = _search(svc, page=1)
        hit = resp.results[0]
        assert hit.match_source == MACHINE_READING
        assert hit.machine_reading.image_index == 0
        assert hit.machine_reading.line_index == 3
        assert hit.machine_reading.snippet == "שורה של A00"
        assert hit.machine_reading.read_url.startswith("/read?doc=A00&image=0&line=3")
        assert hit.metadata.shelf_mark == "SM A00"  # built from the catalogue record
        assert hit.embedding is None
        assert es.mgets[0]["source_excludes"] == ["embedding_vector"]
        assert es.mgets[0]["index"] == CATALOGUE

    def test_catalogue_hits_have_no_match_source(self) -> None:
        svc, _, _ = self._setup()
        resp = _search(svc, page=3)
        assert all(r.match_source is None and r.machine_reading is None for r in resp.results[:5])


class TestWhenThePassDoesNotRun:
    def _svc(self, consensus: Optional[FakeConsensus], relation: str = "eq") -> ElasticsearchService:
        """Service with 3 catalogue hits.

        :param consensus: Consensus double.
        :param relation: Total relation of the main search.
        :return: Service.
        """
        return _service(FakeES(_cat_ids(3), relation=relation), consensus)

    def test_non_hebrew_query(self) -> None:
        consensus = FakeConsensus([_ai_hit("A00")])
        resp = _search(self._svc(consensus), page=1, query="Nahray")
        assert consensus.calls == []
        assert _kinds(resp) == "CCC" and resp.total == 3

    def test_kill_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONSENSUS_SEARCH_ENABLED", "false")
        consensus = FakeConsensus([_ai_hit("A00")])
        resp = _search(self._svc(consensus), page=1)
        assert consensus.calls == []
        assert _kinds(resp) == "CCC" and resp.total == 3

    def test_ai_master_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_TRANSCRIPTIONS_ENABLED=false turns the pass off even with its own switch on."""
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        consensus = FakeConsensus([_ai_hit("A00")])
        resp = _search(self._svc(consensus), page=1)
        assert consensus.calls == []
        assert _kinds(resp) == "CCC" and resp.total == 3

    def test_no_consensus_service(self) -> None:
        resp = _search(self._svc(None), page=1)
        assert _kinds(resp) == "CCC" and resp.total == 3

    def test_other_index(self) -> None:
        consensus = FakeConsensus([_ai_hit("A00")])
        _search(self._svc(consensus), page=1, index_name="bibliography_index")
        assert consensus.calls == []

    def test_lower_bound_total(self) -> None:
        """10,000+ catalogue hits: the slot after the last one is unreachable."""
        consensus = FakeConsensus([_ai_hit("A00")])
        resp = _search(self._svc(consensus, relation="gte"), page=1)
        assert consensus.calls == []
        assert resp.total == 3

    def test_consensus_failure_returns_catalogue_results(self) -> None:
        consensus = FakeConsensus([], error=ConnectionError("AI index down"))
        resp = _search(self._svc(consensus), page=1)
        assert len(consensus.calls) == 1
        assert _kinds(resp) == "CCC"
        assert resp.total == 3 and resp.has_more is False

    def test_catalogue_failure_after_first_pass_also_degrades(self) -> None:
        """An error in the check search is caught the same way."""
        es = FakeES(_cat_ids(3))

        def broken_check(**kwargs: Any) -> Dict[str, Any]:
            """Fail only the candidate check.

            :param kwargs: Search arguments.
            :return: Main search response.
            """
            if "ids" in str(kwargs["query"]):
                raise TimeoutError("check timed out")
            return FakeES.search(es, **kwargs)

        es.search = broken_check  # type: ignore[method-assign]
        resp = _search(_service(es, FakeConsensus([_ai_hit("A00")])), page=1)
        assert _kinds(resp) == "CCC" and resp.total == 3


class TestDuplicates:
    def test_fragment_already_in_catalogue_results_is_dropped(self) -> None:
        es = FakeES(_cat_ids(3))
        consensus = FakeConsensus([_ai_hit("C01"), _ai_hit("A00"), _ai_hit("C02")])
        resp = _search(_service(es, consensus), page=1)
        assert [r.doc_id for r in resp.results] == ["C00", "C01", "C02", "A00"]
        assert resp.total == 4

    def test_one_hit_per_fragment(self) -> None:
        """Two images of one fragment matched: the better-ranked one is kept."""
        consensus = FakeConsensus([_ai_hit("A00", image_index=2), _ai_hit("A01"), _ai_hit("A00", image_index=0)])
        resp = _search(_service(FakeES(_cat_ids(1)), consensus), page=1)
        assert [(r.doc_id, r.machine_reading and r.machine_reading.image_index) for r in resp.results] == [
            ("C00", None), ("A00", 2), ("A01", 0)
        ]
        assert resp.total == 3

    def test_filters_apply_to_machine_reading_hits(self) -> None:
        es = FakeES(_cat_ids(2), filtered_out={"A01"})
        consensus = FakeConsensus([_ai_hit("A00"), _ai_hit("A01"), _ai_hit("A02")])
        resp = _search(_service(es, consensus), page=1, filters={"language": "Judeo-Arabic"})
        assert [r.doc_id for r in resp.results] == ["C00", "C01", "A00", "A02"]
        check_filters = es.checks[0]["query"]["bool"]["filter"]
        assert check_filters[1:] == es.searches[0]["query"]["bool"]["filter"]

    def test_check_names_the_catalogue_text_clause(self) -> None:
        es = FakeES(_cat_ids(1))
        _search(_service(es, FakeConsensus([_ai_hit("A00")])), page=1)
        should = es.checks[0]["query"]["bool"]["should"][0]["bool"]
        text = ElasticsearchService.build_keyword_text_query(HEBREW)["bool"]
        assert should["_name"] == "catalogue_match"
        assert {k: v for k, v in should.items() if k != "_name"} == text
        assert es.checks[0]["_source"] is False

    def test_missing_catalogue_record_is_skipped(self) -> None:
        es = FakeES([], existing={"A01"})
        consensus = FakeConsensus([_ai_hit("A00"), _ai_hit("A01")])
        resp = _search(_service(es, consensus), page=1)
        assert [r.doc_id for r in resp.results] == ["A01"]
        assert resp.total == 1


class WindowedES(FakeES):
    """Catalogue double that enforces ``index.max_result_window`` like Elasticsearch."""

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Reject a main search whose ``from_ + size`` exceeds the result window.

        :param kwargs: Search arguments.
        :return: Response dict.
        :raises ValueError: For a request past ``max_result_window``.
        """
        if "from_" in kwargs and kwargs["from_"] + kwargs["size"] > MAX_RESULT_WINDOW:
            raise ValueError("Result window is too large")
        return super().search(**kwargs)


class TestResultWindow:
    def test_machine_reading_pages_past_the_result_window(self) -> None:
        """9,995 catalogue hits + 20 machine-reading hits: every page is reachable."""
        es = WindowedES(_cat_ids(9995))
        consensus = FakeConsensus([_ai_hit(f"A{i:02d}") for i in range(20)])
        svc = _service(es, consensus)
        straddle = _search(svc, page=1000)
        assert _kinds(straddle) == "C" * 5 + "M" * 5
        page = _search(svc, page=1001)
        assert _kinds(page) == "M" * 10
        assert [r.doc_id for r in page.results] == [f"A{i:02d}" for i in range(5, 15)]
        assert page.total == 10015 and page.has_more is True
        assert es.searches[-1]["from_"] == MAX_RESULT_WINDOW and es.searches[-1]["size"] == 0
        last = _search(svc, page=1002)
        assert _kinds(last) == "M" * 5
        assert last.total == 10015 and last.has_more is False

    def test_deep_page_with_lower_bound_total_is_empty_not_an_error(self) -> None:
        es = WindowedES(_cat_ids(30), relation="gte")
        resp = _search(_service(es, FakeConsensus([_ai_hit("A00")])), page=1002)
        assert resp.results == [] and resp.total == 30
