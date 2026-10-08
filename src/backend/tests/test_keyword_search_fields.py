"""Keyword search must reach the v8 text: ``full_text_content`` and scholarly transcriptions."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from src.backend import search_service as ss
from src.backend.search_service import ElasticsearchService

LEGACY_FIELDS = [
    "transcription_full_text^3.0", "translation_full_text^2.5", "description^2.0", "title^2.5",
    "document_type^1.5", "content_type^1.5", "collection^1.2", "language^1.2", "script_type^1.1",
    "material^1.0",
]


class RecordingES:
    """Records ``search`` calls and answers with no hits."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Record the call.

        :param kwargs: Search arguments.
        :return: An empty response.
        """
        self.calls.append(kwargs)
        return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}


def _service() -> ElasticsearchService:
    """Service over a recording ES, with no consensus service wired.

    :return: Service built without ``__init__`` (no network client).
    """
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.es = RecordingES()
    svc.index_name = "genizah_merged_v8"
    svc.consensus_service = None
    return svc


def _parts(clause: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split the text clause into its multi_match, full_text_content and nested parts.

    :param clause: Output of :meth:`ElasticsearchService.build_keyword_text_query`.
    :return: ``{"multi_match", "full_text", "nested"}`` sub-clauses.
    """
    should = clause["bool"]["should"]
    dis_max = next(c["dis_max"] for c in should if "dis_max" in c)
    nested = next(c["nested"] for c in should if "nested" in c)
    multi_match = next(q["multi_match"] for q in dis_max["queries"] if "multi_match" in q)
    full_text = next(q["match"] for q in dis_max["queries"] if "match" in q)
    return {"multi_match": multi_match, "full_text": full_text, "nested": nested, "dis_max": dis_max}


class TestKeywordTextQuery:
    def test_any_part_may_match(self) -> None:
        """The parts are alternatives: one is enough."""
        clause = ElasticsearchService.build_keyword_text_query("ketubah")
        assert clause["bool"]["minimum_should_match"] == 1
        assert len(clause["bool"]["should"]) == 2

    def test_legacy_fields_are_kept(self) -> None:
        """The old flat fields stay (older indices), with fuzziness as before."""
        mm = _parts(ElasticsearchService.build_keyword_text_query("ketubah"))["multi_match"]
        assert mm["fields"] == LEGACY_FIELDS
        assert mm["query"] == "ketubah"
        assert mm["type"] == "best_fields"
        assert mm["fuzziness"] == "AUTO"

    def test_full_text_content_is_searched(self) -> None:
        """v8 ``full_text_content`` at boost 1.5, no fuzziness, maxed with the legacy fields."""
        parts = _parts(ElasticsearchService.build_keyword_text_query("ketubah"))
        assert parts["full_text"] == {"full_text_content": {"query": "ketubah", "boost": 1.5}}
        assert parts["dis_max"]["tie_breaker"] == 0.0

    def test_scholarly_transcriptions_are_searched(self) -> None:
        """Nested ``transcriptions.text``, best line counts, harmless on indices without the mapping."""
        nested = _parts(ElasticsearchService.build_keyword_text_query("לשריפתו"))["nested"]
        assert nested["path"] == "transcriptions"
        assert nested["query"] == {"match": {"transcriptions.text": {"query": "לשריפתו"}}}
        assert nested["score_mode"] == "max"
        assert nested["boost"] == 3.0
        assert nested["ignore_unmapped"] is True

    def test_ai_text_never_enters_the_catalogue_clause(self) -> None:
        """docs/ai_transcription_search.md: AI fields stay out of the default keyword clause."""
        dumped = json.dumps(ElasticsearchService.build_keyword_text_query("שלום"), ensure_ascii=False)
        for field in ("text_agreed", "text_all", "ai_read"):
            assert field not in dumped


class TestKeywordSearchSendsTheClause:
    def test_without_filters(self) -> None:
        """No filters: the text clause is the whole query, and the embedding vector is not fetched."""
        svc = _service()
        req = SimpleNamespace(query="Maimonides", filters=None, num_results=10, page=1)
        asyncio.run(svc.search_by_keyword(req))
        call = svc.es.calls[0]
        assert call["query"] == ElasticsearchService.build_keyword_text_query("Maimonides")
        assert call["source_excludes"] == ["embedding_vector"]
        assert call["from_"] == 0 and call["size"] == 10

    def test_with_filters(self) -> None:
        """Filters wrap the same text clause in ``bool.must``."""
        svc = _service()
        req = SimpleNamespace(query="Maimonides", filters={"language": "Hebrew"}, num_results=5, page=3)
        asyncio.run(svc.search_by_keyword(req))
        query = svc.es.calls[0]["query"]
        assert query["bool"]["must"] == [ElasticsearchService.build_keyword_text_query("Maimonides")]
        assert query["bool"]["filter"]
        assert svc.es.calls[0]["from_"] == 10


class TestHybridUsesTheClause:
    def test_keyword_function_filter(self, monkeypatch) -> None:
        """Hybrid search scores the keyword part with the same text clause."""

        async def fake_embedding(text: str, image: Any = None, use_cache: bool = False) -> np.ndarray:
            """Fixed query vector.

            :param text: Query text.
            :param image: Unused.
            :param use_cache: Unused.
            :return: A 3-d vector.
            """
            return np.array([0.1, 0.2, 0.3])

        monkeypatch.setattr(ss.embedding_client, "get_embedding", fake_embedding)
        svc = _service()
        req = SimpleNamespace(query="ketubah", filters=None, num_results=10, page=1, semanticWeight=50,
                              keywordWeight=50, include_embeddings=False)
        asyncio.run(svc.search_hybrid(req))
        functions = svc.es.calls[0]["query"]["function_score"]["functions"]
        keyword_fn = next(f for f in functions if "filter" in f and f["filter"] != {"match_all": {}})
        assert keyword_fn["filter"] == ElasticsearchService.build_keyword_text_query("ketubah")
        assert keyword_fn["weight"] == 0.5
