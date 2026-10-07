"""Tests for the scholar / AI / either transcription-source search filter."""

from typing import Any, Dict, List

from src.backend.search_service import ElasticsearchService
from src.backend import ai_transcriptions as at


def _service(ai_ids: List[str]) -> ElasticsearchService:
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc._surfaced_ai_doc_ids = lambda: list(ai_ids)  # type: ignore[attr-defined]
    return svc


def _clauses(svc: ElasticsearchService, source: str) -> List[Dict[str, Any]]:
    return svc._build_filters({"transcription_source": source})


class TestFilterClauses:
    def test_scholar_uses_has_transcriptions(self):
        assert _clauses(_service([]), "scholar") == [{"term": {"has_transcriptions": True}}]

    def test_ai_uses_doc_id_terms(self):
        clauses = _clauses(_service(["A", "B"]), "ai")
        assert clauses == [{"terms": {"doc_id": ["A", "B"]}}]

    def test_ai_with_no_data_matches_nothing(self):
        clauses = _clauses(_service([]), "ai")
        assert clauses == [{"bool": {"must_not": {"match_all": {}}}}]

    def test_either_is_scholar_or_ai(self):
        clauses = _clauses(_service(["A"]), "either")
        assert clauses == [{
            "bool": {
                "should": [{"term": {"has_transcriptions": True}}, {"terms": {"doc_id": ["A"]}}],
                "minimum_should_match": 1,
            }
        }]

    def test_unknown_source_is_ignored(self):
        assert _service([]) ._build_filters({"transcription_source": "bogus"}) == []
        assert _service([])._build_filters({"transcription_source": None}) == []


class FakeES:
    """search() returning surfaced doc_ids in ``search_after`` pages."""

    def __init__(self, ids: List[str], page: int = 2):
        self.pages = [ids[i:i + page] for i in range(0, len(ids), page)] or [[]]
        self.calls = 0

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        after = kwargs.get("search_after")
        idx = 0 if after is None else next(i + 1 for i, p in enumerate(self.pages) if p and p[-1] == after[0])
        page = self.pages[idx] if idx < len(self.pages) else []
        self.calls += 1
        return {"hits": {"hits": [{"_source": {"doc_id": d}, "sort": [d]} for d in page]}}


class TestSurfacedDocIds:
    def test_paginates_and_dedupes(self, monkeypatch):
        monkeypatch.setattr(at, "_surfaced_cache", {})
        svc = at.AiTranscriptionService(FakeES(["a", "b", "c", "c"]), index="x")
        assert svc.surfaced_doc_ids() == ["a", "b", "c"]

    def test_cached_within_ttl(self, monkeypatch):
        monkeypatch.setattr(at, "_surfaced_cache", {})
        fake = FakeES(["a", "b"])
        svc = at.AiTranscriptionService(fake, index="x")
        svc.surfaced_doc_ids()
        first = fake.calls
        svc.surfaced_doc_ids()
        assert fake.calls == first  # served from cache, no new queries
