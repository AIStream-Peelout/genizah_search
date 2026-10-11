"""Opt-in semantic beta (genizah_merged_v10): embedder routing, eligibility filtering, contract check.

Everything runs on fakes (no network): a recording Elasticsearch client and
embedding clients whose HTTP layer is replaced.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pytest
from elasticsearch import NotFoundError

from src.backend import search_service as ss
from src.backend import semantic_beta as sb
from src.backend.embedding_client import EmbeddingClient
from src.backend.search_service import ElasticsearchService, SearchRequest

V8 = "genizah_merged_v8"
V10 = "genizah_merged_v10"
# The abandoned v9 beta index: same model, older revision, semantic_v3.
V9 = "genizah_merged_v9"
STOCK = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
TUNED = ("isaacmg/genizah-embed-qwen3-0.6b", "af0252255d3e9b8b8432e09de7c5f0689d7ada34")
TUNED_V9 = ("isaacmg/genizah-embed-qwen3-0.6b", "dbfab73f1e91a6b95eba728541cc5a391a88d933")
V10_ELIGIBLE = 65522
PROMPT = "Instruct: Given a search query, retrieve relevant passages\nQuery: "
CANARY_STRING = "genizah-embedding-canary-v1: Cairo Genizah semantic search."
DOC_CANARY_TEXT = "Description: a ketubah from Fustat\nDate: 1100"
QUERY_VECTOR = [0.1, 0.2, 0.3]
COSINE = "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0"


def _unit(seed: int, dims: int = 8) -> List[float]:
    """Deterministic unit vector.

    :param seed: RNG seed.
    :param dims: Vector length.
    :return: The vector as a list.
    """
    vector = np.random.default_rng(seed).normal(size=dims)
    return (vector / np.linalg.norm(vector)).tolist()


CANARY_VECTORS: Dict[Tuple[str, str], List[float]] = {
    (CANARY_STRING, "document"): _unit(1),
    (CANARY_STRING, "query"): _unit(2),
    (DOC_CANARY_TEXT, "document"): _unit(3),
}


def _beta_meta(**overrides: Any) -> Dict[str, Any]:
    """Beta index ``_meta`` shaped like genizah_merged_v10's.

    :param overrides: Keys to replace.
    :return: The ``_meta`` dict.
    """
    meta = {
        "model_name": TUNED[0], "embedding_model": TUNED[0],
        "model_revision": TUNED[1], "embedding_model_revision": TUNED[1],
        "text_representation_version": "semantic_v4",
        "query_prompt": PROMPT, "query_instruction": PROMPT,
        "canary": {"string": CANARY_STRING, "vector": CANARY_VECTORS[(CANARY_STRING, "document")],
                   "vector_sha256_6dp": "x"},
        "query_canary": {"string": CANARY_STRING, "vector": CANARY_VECTORS[(CANARY_STRING, "query")],
                         "prompt": PROMPT, "vector_sha256_6dp": "x"},
        "document_canary": {"doc_id": "987", "text": DOC_CANARY_TEXT,
                            "vector": CANARY_VECTORS[(DOC_CANARY_TEXT, "document")], "vector_sha256_6dp": "x"},
    }
    meta.update(overrides)
    return meta


def _mapping(meta: Optional[Dict[str, Any]], eligible: bool = False) -> Dict[str, Any]:
    """One index's ``mappings`` body.

    :param meta: ``_meta`` (``None`` for a legacy index without one).
    :param eligible: Whether the mapping has ``semantic_eligible``.
    :return: The body.
    """
    properties: Dict[str, Any] = {"embedding_vector": {"type": "dense_vector", "dims": 1024}}
    if eligible:
        properties["semantic_eligible"] = {"type": "boolean"}
    body: Dict[str, Any] = {"properties": properties}
    if meta is not None:
        body["_meta"] = meta
    return body


def _cluster_mappings() -> Dict[str, Dict[str, Any]]:
    """Mappings of the indices the fake cluster serves.

    :return: ``{index: mappings body}``.
    """
    stock = {"embedding_model": STOCK[0], "embedding_model_revision": STOCK[1]}
    return {
        V8: _mapping(stock),
        "genizah_merged_v7": _mapping(stock),
        "genizah_merged_v8_eligible": _mapping(stock, eligible=True),
        V10: _mapping(_beta_meta(), eligible=True),
        V9: _mapping(_beta_meta(model_name=TUNED_V9[0], embedding_model=TUNED_V9[0], model_revision=TUNED_V9[1],
                                embedding_model_revision=TUNED_V9[1], text_representation_version="semantic_v3"),
                     eligible=True),
        "genizah_merged_v1": _mapping(None),
        "other_model_index": _mapping({"model_name": "someone/else", "model_revision": "abc"}),
    }


class FakeIndices:
    """``es.indices`` answering ``get_mapping`` from a dict and counting reads."""

    def __init__(self, mappings: Dict[str, Dict[str, Any]]) -> None:
        """Store the mappings.

        :param mappings: ``{index: mappings body}``.
        """
        self.mappings = mappings
        self.reads: List[str] = []

    def get_mapping(self, index: str) -> Dict[str, Any]:
        """Return ``{index: {"mappings": ...}}`` like Elasticsearch.

        :param index: Index name.
        :return: The mapping response.
        :raises NotFoundError: For an unknown index.
        """
        self.reads.append(index)
        if index not in self.mappings:
            raise NotFoundError("index_not_found_exception", SimpleNamespace(status=404), {})
        return {index: {"mappings": self.mappings[index]}}


class FakeES:
    """Recording Elasticsearch client with a mapping API and canned hits."""

    def __init__(self, mappings: Optional[Dict[str, Dict[str, Any]]] = None,
                 hits: Optional[List[Dict[str, Any]]] = None, eligible_count: int = V10_ELIGIBLE) -> None:
        """Create the client.

        :param mappings: Index mappings (defaults to :func:`_cluster_mappings`).
        :param hits: Hits returned by every search.
        :param eligible_count: Answer of ``count``.
        """
        self.indices = FakeIndices(_cluster_mappings() if mappings is None else mappings)
        self.hits = hits or []
        self.eligible_count = eligible_count
        self.aggregations: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []
        self.counts: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Record a search.

        :param kwargs: Search arguments.
        :return: The canned hits (and canned aggregations).
        """
        self.calls.append(kwargs)
        return {"hits": {"total": {"value": len(self.hits), "relation": "eq"}, "hits": self.hits},
                "aggregations": self.aggregations}

    def count(self, **kwargs: Any) -> Dict[str, Any]:
        """Record a count.

        :param kwargs: Count arguments.
        :return: The configured count.
        """
        self.counts.append(kwargs)
        return {"count": self.eligible_count}


class RecordingSearchOnlyES:
    """Exactly what the v8 path may touch: ``search`` and nothing else (no mapping API)."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Record a search.

        :param kwargs: Search arguments.
        :return: No hits.
        """
        self.calls.append(kwargs)
        return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}


class FakeEmbeddingClient(EmbeddingClient):
    """Real client logic over a fake HTTP layer."""

    def __init__(self, identity: Tuple[str, str], vectors: Optional[Dict[Tuple[str, str], List[float]]] = None) -> None:
        """Create the client.

        :param identity: ``(model, revision)`` reported by ``/health``.
        :param vectors: Canary vectors it reproduces, by ``(text, mode)``.
        """
        super().__init__(base_url="http://fake-embedder:8001")
        self.identity = identity
        self.vectors = dict(CANARY_VECTORS if vectors is None else vectors)
        self.down = False
        self.embeds: List[Tuple[str, str, bool]] = []

    def _maybe_fail(self) -> None:
        """Raise a connection error while ``down``.

        :raises httpx.ConnectError: When down.
        """
        if self.down:
            raise httpx.ConnectError("connection refused")

    async def _post_embed(self, text: str, mode: str, use_cache: bool) -> np.ndarray:
        """Return the stored vector for canaries and a fixed vector otherwise.

        :param text: Text.
        :param mode: Encoding mode.
        :param use_cache: Cache flag.
        :return: The vector.
        """
        self._maybe_fail()
        self.embeds.append((text, mode, use_cache))
        return np.array(self.vectors.get((text, mode), QUERY_VECTOR))

    async def get_health(self) -> Dict[str, Any]:
        """Report the configured identity.

        :return: A ``/health``-shaped payload.
        """
        self._maybe_fail()
        return {"status": "healthy", "model_loaded": True, "model": self.identity[0],
                "revision": self.identity[1], "query_instruction": PROMPT}


def _beta(es: FakeES, client: Optional[FakeEmbeddingClient] = None, enabled: bool = True,
          **kwargs: Any) -> Tuple[sb.SemanticBeta, sb.SemanticRouter]:
    """Fresh beta state and router over one contract cache.

    :param es: Fake cluster (unused here; kept for symmetry with callers).
    :param client: Beta embedding client.
    :param enabled: ``SEMANTIC_BETA_ENABLED``.
    :param kwargs: Extra :class:`SemanticBeta` arguments.
    :return: ``(beta, router)``.
    """
    contracts = sb.IndexContractCache()
    beta = sb.SemanticBeta(index=V10, client=client or FakeEmbeddingClient(TUNED), enabled=enabled,
                           contracts=contracts, **kwargs)
    return beta, sb.SemanticRouter(contracts=contracts, beta=beta)


def _service(es: Any) -> ElasticsearchService:
    """Search service over a fake client, built without ``__init__``.

    :param es: Fake Elasticsearch client.
    :return: The service.
    """
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.es = es
    svc.index_name = V8
    svc.consensus_service = None
    return svc


@pytest.fixture
def default_client(monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddingClient:
    """Replace the default embedding client the search paths use.

    :param monkeypatch: Pytest monkeypatch.
    :return: The fake default client.
    """
    client = FakeEmbeddingClient(STOCK)
    monkeypatch.setattr(ss, "embedding_client", client)
    return client


@pytest.fixture
def cluster() -> FakeES:
    """Fake cluster with v1/v7/v8/v9/v10 and an unrelated-model index.

    :return: The fake client.
    """
    return FakeES()


@pytest.fixture
def beta_up(monkeypatch: pytest.MonkeyPatch, cluster: FakeES) -> sb.SemanticBeta:
    """A verified beta wired into the search service's router.

    :param monkeypatch: Pytest monkeypatch.
    :param cluster: Fake cluster.
    :return: The available beta.
    """
    beta, router = _beta(cluster)
    assert asyncio.run(beta.check(cluster)) is True
    monkeypatch.setattr(ss, "semantic_router", router)
    return beta


def _hybrid_request(**overrides: Any) -> SimpleNamespace:
    """Hybrid request like the /search-hybrid model.

    :param overrides: Fields to replace.
    :return: The request.
    """
    fields = dict(query="ketubah", filters=None, num_results=10, page=1, semanticWeight=60,
                  keywordWeight=40, include_embeddings=False)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hit(doc_id: str, shelf_mark: str, score: float = 1.8) -> Dict[str, Any]:
    """One Elasticsearch hit.

    :param doc_id: Record id.
    :param shelf_mark: Stored shelf mark.
    :param score: ``_score``.
    :return: The hit.
    """
    return {"_id": doc_id, "_score": score, "_source": {"doc_id": doc_id, "shelf_mark": shelf_mark}}


# A routed shelf-mark query needs a hit: with none it falls back to the search as typed.
SHELFMARK_HIT = _hit("Cambridge_CUL_T_S_13J20_20", "T-S 13J20.20", score=30.0)


def _last_search(es: FakeES, index: str) -> Dict[str, Any]:
    """The latest search sent to one index (not the catalogue lookup that follows beta results).

    :param es: Fake cluster.
    :param index: Index name.
    :return: That search's arguments.
    """
    return [call for call in es.calls if call["index"] == index][-1]


class CatalogueES(FakeES):
    """Fake cluster whose catalogue index (v8) holds only some of the beta records."""

    def __init__(self, hits: List[Dict[str, Any]], catalogue_ids: List[str]) -> None:
        """Create the client.

        :param hits: Hits of every non-lookup search.
        :param catalogue_ids: Doc ids the v8 ``terms`` lookup finds.
        """
        super().__init__(hits=hits)
        self.catalogue_ids = set(catalogue_ids)

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        """Answer a v8 ``terms`` lookup from ``catalogue_ids``, anything else with the canned hits.

        :param kwargs: Search arguments.
        :return: The response.
        """
        if kwargs["index"] != V8 or "terms" not in kwargs["query"]:
            return super().search(**kwargs)
        self.calls.append(kwargs)
        found = [_hit(i, "") for i in kwargs["query"]["terms"]["doc_id"] if i in self.catalogue_ids]
        return {"hits": {"total": {"value": len(found), "relation": "eq"}, "hits": found}}


# ---------------------------------------------------------------------------
# v8: unchanged
# ---------------------------------------------------------------------------

def _historical_semantic_query(vector: List[float], filter_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The semantic query body exactly as search() built it before the beta.

    :param vector: Query vector.
    :param filter_clauses: Filter clauses.
    :return: The body.
    """
    if filter_clauses:
        base_query = {"bool": {"filter": filter_clauses}}
    else:
        base_query = {"match_all": {}}
    return {"script_score": {"query": base_query, "script": {"source": COSINE, "params": {"query_vector": vector}}}}


def _historical_hybrid_query(vector: List[float], filter_clauses: List[Dict[str, Any]],
                             keyword: Dict[str, Any], sw: float, kw: float) -> Dict[str, Any]:
    """The hybrid query body exactly as search_hybrid() built it before the beta.

    :param vector: Query vector.
    :param filter_clauses: Filter clauses.
    :param keyword: Keyword text clause.
    :param sw: Semantic weight.
    :param kw: Keyword weight.
    :return: The body.
    """
    if filter_clauses:
        base_query = {"bool": {"filter": filter_clauses}}
    else:
        base_query = {"match_all": {}}
    return {"function_score": {
        "query": base_query,
        "functions": [
            {"filter": {"match_all": {}}, "weight": sw,
             "script_score": {"script": {"source": COSINE, "params": {"query_vector": vector}}}},
            {"filter": keyword, "weight": kw},
        ],
        "score_mode": "sum", "boost_mode": "multiply"}}


class TestDefaultPathUnchanged:
    @pytest.mark.parametrize("filters", [None, {"language": "Hebrew"}])
    def test_semantic_body_and_client(self, default_client: FakeEmbeddingClient, filters: Any,
                                      monkeypatch: pytest.MonkeyPatch, cluster: FakeES) -> None:
        """Default index: default client, no mapping read, byte-identical body."""
        beta, router = _beta(cluster)
        monkeypatch.setattr(ss, "semantic_router", router)
        es = RecordingSearchOnlyES()
        svc = _service(es)
        out = asyncio.run(svc.search(SearchRequest(query="ketubah", filters=filters)))
        assert default_client.embeds == [("ketubah", "query", False)]
        assert beta.client.embeds == []
        expected = _historical_semantic_query(QUERY_VECTOR, svc._build_filters(filters))
        assert json.dumps(es.calls[0]["query"]) == json.dumps(expected)
        assert es.calls[0]["index"] == V8
        assert out.semantic_beta is None and out.routed_to is None

    def test_default_results_have_no_catalogue_lookup(self, default_client: FakeEmbeddingClient) -> None:
        """v8 results are catalogue records: no extra lookup, catalogue_page stays None."""
        es = RecordingSearchOnlyES()
        svc = _service(es)
        es.search = lambda **kwargs: (es.calls.append(kwargs) or
                                      {"hits": {"total": {"value": 1, "relation": "eq"},
                                                "hits": [_hit("Cambridge_CUL_T_S_Ar_21_105", "T-S Ar. 21.105")]}})
        out = asyncio.run(svc.search(SearchRequest(query="ketubah")))
        assert len(es.calls) == 1 and out.results[0].catalogue_page is None
        out = asyncio.run(svc.search_hybrid(_hybrid_request()))
        assert len(es.calls) == 2 and out.results[0].catalogue_page is None

    def test_explicit_default_index_same_as_omitted(self, default_client: FakeEmbeddingClient) -> None:
        """Naming the default index explicitly takes the same path."""
        es = RecordingSearchOnlyES()
        asyncio.run(_service(es).search(SearchRequest(query="ketubah", index_name=V8)))
        assert json.dumps(es.calls[0]["query"]) == json.dumps(_historical_semantic_query(QUERY_VECTOR, []))

    def test_shelfmark_query_stays_semantic_on_default(self, default_client: FakeEmbeddingClient) -> None:
        """Shelf-mark routing is beta-only: v8 embeds a shelf-mark query as before."""
        es = RecordingSearchOnlyES()
        out = asyncio.run(_service(es).search(SearchRequest(query="T-S 13J20.20")))
        assert default_client.embeds == [("T-S 13J20.20", "query", False)]
        assert "script_score" in es.calls[0]["query"]
        assert out.routed_to is None

    @pytest.mark.parametrize("query", ["T-S 13J20.20", "T-S 13J", "Mosseri VII", "ENA 2558"])
    def test_shelfmark_and_prefix_stay_semantic_on_default(self, default_client: FakeEmbeddingClient,
                                                           query: str) -> None:
        """Prefix routing is beta-only too: v8 embeds shelf marks and prefixes as before."""
        es = RecordingSearchOnlyES()
        out = asyncio.run(_service(es).search(SearchRequest(query=query)))
        assert default_client.embeds == [(query, "query", False)]
        assert json.dumps(es.calls[0]["query"]) == json.dumps(_historical_semantic_query(QUERY_VECTOR, []))
        assert out.routed_to is None

    @pytest.mark.parametrize("query", ["T-S 13J20.20", "T-S 13J", "Mosseri VII"])
    def test_shelfmark_stays_hybrid_on_default(self, default_client: FakeEmbeddingClient, query: str) -> None:
        """Hybrid on v8: a shelf mark or prefix is embedded and searched with the historical body."""
        es = RecordingSearchOnlyES()
        svc = _service(es)
        out = asyncio.run(svc.search_hybrid(_hybrid_request(query=query)))
        assert default_client.embeds == [(query, "query", False)]
        expected = _historical_hybrid_query(
            QUERY_VECTOR, [], ElasticsearchService.build_keyword_text_query(query), 0.6, 0.4)
        assert json.dumps(es.calls[0]["query"]) == json.dumps(expected)
        assert out.routed_to is None and out.semantic_beta is None

    @pytest.mark.parametrize("filters", [None, {"language": "Hebrew"}])
    def test_hybrid_body(self, default_client: FakeEmbeddingClient, filters: Any) -> None:
        """Default-index hybrid body is byte-identical to the historical one."""
        es = RecordingSearchOnlyES()
        svc = _service(es)
        out = asyncio.run(svc.search_hybrid(_hybrid_request(filters=filters)))
        expected = _historical_hybrid_query(
            QUERY_VECTOR, svc._build_filters(filters), ElasticsearchService.build_keyword_text_query("ketubah"),
            0.6, 0.4,
        )
        assert json.dumps(es.calls[0]["query"]) == json.dumps(expected)
        assert out.semantic_beta is None

    def test_other_stock_index_uses_default_client(self, default_client: FakeEmbeddingClient,
                                                   beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """A non-default index built by the stock model keeps the default embedder and body."""
        asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name="genizah_merged_v7")))
        assert default_client.embeds == [("ketubah", "query", False)]
        assert json.dumps(cluster.calls[0]["query"]) == json.dumps(_historical_semantic_query(QUERY_VECTOR, []))


# ---------------------------------------------------------------------------
# Beta path
# ---------------------------------------------------------------------------

class TestBetaSearch:
    def test_beta_client_and_eligibility_filter(self, default_client: FakeEmbeddingClient,
                                                beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """v10 embeds with the beta client and filters to semantic_eligible records."""
        beta_up.client.embeds.clear()
        out = asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert beta_up.client.embeds == [("ketubah", "query", False)]
        assert default_client.embeds == []
        query = cluster.calls[-1]["query"]["script_score"]["query"]
        assert query == {"bool": {"filter": [{"term": {"semantic_eligible": True}}]}}
        assert cluster.calls[-1]["index"] == V10
        assert out.semantic_beta is True and out.index_name == V10

    def test_user_filters_kept(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                               cluster: FakeES) -> None:
        """User filters come first, the eligibility clause is appended."""
        svc = _service(cluster)
        asyncio.run(svc.search(SearchRequest(query="ketubah", index_name=V10, filters={"language": "Hebrew"})))
        filters = cluster.calls[-1]["query"]["script_score"]["query"]["bool"]["filter"]
        assert filters == svc._build_filters({"language": "Hebrew"}) + [{"term": {"semantic_eligible": True}}]

    def test_shelfmark_routed_on_beta(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                                      cluster: FakeES) -> None:
        """A bare shelf mark on v10 is answered by shelf-mark search, without embedding."""
        cluster.hits = [SHELFMARK_HIT]
        beta_up.client.embeds.clear()
        out = asyncio.run(_service(cluster).search(SearchRequest(query=" T-S 13J20.20. ", index_name=V10)))
        assert beta_up.client.embeds == [] and default_client.embeds == []
        assert "script_score" not in json.dumps(_last_search(cluster, V10)["query"])
        assert out.routed_to == "shelfmark" and out.semantic_beta is True
        assert out.filters_applied["shelf_mark"] == "T-S 13J20.20"

    def test_question_mentioning_shelfmark_not_routed(self, default_client: FakeEmbeddingClient,
                                                      beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """Only a bare shelf mark is routed; a question that mentions one stays semantic."""
        out = asyncio.run(_service(cluster).search(
            SearchRequest(query="letters related to T-S 13J20.20", index_name=V10)))
        assert "script_score" in cluster.calls[-1]["query"]
        assert out.routed_to is None

    @pytest.mark.parametrize("query", [
        "T-S 13J", "T-S NS 324", "T-S Ar. 30", "ENA 2558", "Bodl. MS Heb. d 66", "Mosseri VII", "Moss. IV",
    ])
    def test_shelfmark_prefix_routed_on_beta(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                                             cluster: FakeES, query: str) -> None:
        """A shelf-mark prefix or volume on v10 is answered by shelf-mark search (semantic and hybrid)."""
        cluster.hits = [SHELFMARK_HIT]
        beta_up.client.embeds.clear()
        svc = _service(cluster)
        for out in (
            asyncio.run(svc.search(SearchRequest(query=query, index_name=V10))),
            asyncio.run(svc.search_hybrid(_hybrid_request(query=query), V10)),
        ):
            assert out.routed_to == "shelfmark" and out.semantic_beta is True
            assert out.filters_applied["shelf_mark"] == query
            assert "script_score" not in json.dumps(_last_search(cluster, V10)["query"])
        assert beta_up.client.embeds == [] and default_client.embeds == []

    @pytest.mark.parametrize("query", ["Mosseri collection", "the Taylor-Schechter collection", "Mosseri"])
    def test_collection_name_stays_semantic_on_beta(self, default_client: FakeEmbeddingClient,
                                                    beta_up: sb.SemanticBeta, cluster: FakeES, query: str) -> None:
        """A bare collection name with no number is a topic, not a shelf mark."""
        beta_up.client.embeds.clear()
        out = asyncio.run(_service(cluster).search(SearchRequest(query=query, index_name=V10)))
        assert beta_up.client.embeds == [(query, "query", False)]
        assert "script_score" in cluster.calls[-1]["query"] and out.routed_to is None
        out = asyncio.run(_service(cluster).search_hybrid(_hybrid_request(query=query), V10))
        assert "function_score" in cluster.calls[-1]["query"] and out.routed_to is None

    def test_hybrid_shelfmark_routed_on_beta(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                                             cluster: FakeES) -> None:
        """A bare shelf mark sent to /search-hybrid on v10 is answered by shelf-mark search, without embedding."""
        cluster.hits = [SHELFMARK_HIT]
        beta_up.client.embeds.clear()
        svc = _service(cluster)
        out = asyncio.run(svc.search_hybrid(
            _hybrid_request(query=" T-S 13J20.20. ", filters={"language": "Hebrew"}, num_results=7), V10))
        assert beta_up.client.embeds == [] and default_client.embeds == []
        call = _last_search(cluster, V10)
        assert "function_score" not in json.dumps(call["query"]) and "script_score" not in json.dumps(call["query"])
        assert call["query"]["bool"]["filter"] == svc._build_filters({"language": "Hebrew"})
        assert out.routed_to == "shelfmark" and out.semantic_beta is True
        assert out.filters_applied["shelf_mark"] == "T-S 13J20.20" and out.page_size == 7

    def test_hybrid_question_mentioning_shelfmark_not_routed(self, default_client: FakeEmbeddingClient,
                                                             beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """Hybrid on v10: a question that mentions a shelf mark stays hybrid."""
        out = asyncio.run(_service(cluster).search_hybrid(
            _hybrid_request(query="letters related to T-S 13J20.20"), V10))
        assert "function_score" in cluster.calls[-1]["query"]
        assert out.routed_to is None and out.semantic_beta is True

    @pytest.mark.parametrize("query", ["Mosseri XI", "T-S 99J99.99"])
    def test_routed_query_without_hits_runs_as_typed(self, default_client: FakeEmbeddingClient,
                                                      beta_up: sb.SemanticBeta, cluster: FakeES, query: str) -> None:
        """No record has the shelf mark: semantic / hybrid answer instead of an empty routed page."""
        beta_up.client.embeds.clear()
        svc = _service(cluster)
        out = asyncio.run(svc.search(SearchRequest(query=query, index_name=V10)))
        shelfmark_call, semantic_call = cluster.calls[-2:]
        assert "script_score" not in json.dumps(shelfmark_call["query"])
        assert "script_score" in semantic_call["query"] and semantic_call["index"] == V10
        assert out.routed_to is None and out.semantic_beta is True
        out = asyncio.run(svc.search_hybrid(_hybrid_request(query=query), V10))
        assert "function_score" in cluster.calls[-1]["query"] and out.routed_to is None
        assert beta_up.client.embeds == [(query, "query", False)] * 2

    @pytest.mark.parametrize("mode", ["semantic", "hybrid", "routed"])
    def test_beta_results_flag_catalogue_pages(self, default_client: FakeEmbeddingClient,
                                               beta_up: sb.SemanticBeta, mode: str) -> None:
        """A beta result gets catalogue_page true only when its doc_id is in the catalogue index (v8)."""
        es = CatalogueES(
            hits=[_hit("Cambridge_CUL_T_S_13J20_20", "T-S 13J20.20"),
                  _hit("Cambridge_CUL_T_S_Ar_21_105", "T-S Ar. 21.105")],
            catalogue_ids=["Cambridge_CUL_T_S_13J20_20"],
        )
        svc = _service(es)
        query = "T-S 13J20.20" if mode == "routed" else "ketubah"
        if mode == "hybrid":
            out = asyncio.run(svc.search_hybrid(_hybrid_request(query=query), V10))
        else:
            out = asyncio.run(svc.search(SearchRequest(query=query, index_name=V10)))
        assert {r.doc_id: r.catalogue_page for r in out.results} == {
            "Cambridge_CUL_T_S_13J20_20": True, "Cambridge_CUL_T_S_Ar_21_105": False}
        lookup = es.calls[-1]
        assert lookup["index"] == V8 and lookup["query"] == {"terms": {"doc_id": [
            "Cambridge_CUL_T_S_13J20_20", "Cambridge_CUL_T_S_Ar_21_105"]}}

    def test_old_oxford_id_counts_when_its_catalogue_form_exists(self, default_client: FakeEmbeddingClient,
                                                                 beta_up: sb.SemanticBeta) -> None:
        """The fragment page redirects an old Oxford id to its catalogue form, so the flag follows that form."""
        from src.backend.consensus_reads import to_catalogue_doc_id

        old = "Oxford_Bodleian_MS_heb_d_66_1"
        mapped = to_catalogue_doc_id(old)
        assert mapped == "Oxford_Bodleian_Bodl_MS_heb_d_66_1"
        es = CatalogueES(hits=[_hit(old, "MS heb. d 66/1")], catalogue_ids=[mapped])
        out = asyncio.run(_service(es).search(SearchRequest(query="ketubah", index_name=V10)))
        assert out.results[0].catalogue_page is True

    def test_beta_hybrid_keyword_leg_has_v10_fields(self, default_client: FakeEmbeddingClient,
                                                    beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """The BM25 leg of /search-hybrid on v10 searches scholarly_frames, subjects and genres."""
        asyncio.run(_service(cluster).search_hybrid(_hybrid_request(), V10))
        keyword_fn = cluster.calls[-1]["query"]["function_score"]["functions"][1]["filter"]
        multi_match = keyword_fn["bool"]["should"][0]["dis_max"]["queries"][0]["multi_match"]
        assert multi_match["fields"][-3:] == ["scholarly_frames^2.0", "subjects^1.5", "genres^1.0"]
        assert multi_match["fuzziness"] == "AUTO"

    def test_beta_hybrid_filters_vector_function(self, default_client: FakeEmbeddingClient,
                                                 beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """Hybrid on v10: cosine only on eligible records, keyword matches still admitted."""
        out = asyncio.run(_service(cluster).search_hybrid(_hybrid_request(), V10))
        body = cluster.calls[-1]["query"]["function_score"]
        keyword = ElasticsearchService.build_keyword_text_query("ketubah")
        cosine_fn, keyword_fn = body["functions"]
        assert cosine_fn["filter"] == {"term": {"semantic_eligible": True}}
        assert keyword_fn == {"filter": keyword, "weight": 0.4}
        # match_all keeps the base score at 1: a filter-only bool scores 0 and
        # boost_mode multiply would zero every hybrid score.
        assert body["query"] == {"bool": {"must": [{"match_all": {}}], "filter": [
            {"bool": {"should": [{"term": {"semantic_eligible": True}}, keyword], "minimum_should_match": 1}}
        ]}}
        assert out.semantic_beta is True

    def test_beta_hybrid_vectorless_hits_get_query_sized_embeddings(
            self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta) -> None:
        """A page of only keyword-matched vector-less records still yields aligned query-sized vectors."""
        es = FakeES(hits=[
            {"_id": "a", "_score": 1.4, "_source": {"doc_id": "a", "semantic_eligible": False}},
            {"_id": "b", "_score": 1.3, "_source": {"doc_id": "b", "semantic_eligible": False}},
        ])
        out = asyncio.run(_service(es).search_hybrid(_hybrid_request(include_embeddings=True), V10))
        dims = len(QUERY_VECTOR)
        assert [len(r.embedding) for r in out.results] == [dims, dims]
        assert out.embedding_data.dimension == dims
        assert [len(v) for v in out.embedding_data.result_embeddings] == [dims, dims]

    def test_any_eligibility_mapped_index_is_filtered(self, default_client: FakeEmbeddingClient,
                                                      beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """Eligibility filtering follows the mapping, not the beta: a stock-model index with the field."""
        svc = _service(cluster)
        asyncio.run(svc.search(SearchRequest(query="ketubah", index_name="genizah_merged_v8_eligible")))
        assert default_client.embeds == [("ketubah", "query", False)]
        assert cluster.calls[-1]["query"]["script_score"]["query"] == {
            "bool": {"filter": [{"term": {"semantic_eligible": True}}]}}
        asyncio.run(svc.search_hybrid(_hybrid_request(filters={"language": "Hebrew"}), "genizah_merged_v8_eligible"))
        body = cluster.calls[-1]["query"]["function_score"]
        assert body["functions"][0]["filter"] == {"term": {"semantic_eligible": True}}
        assert body["query"]["bool"]["must"] == [{"match_all": {}}]
        assert body["query"]["bool"]["filter"][:-1] == svc._build_filters({"language": "Hebrew"})

    def test_beta_embedder_failure_is_503_and_rechecks(self, default_client: FakeEmbeddingClient,
                                                       beta_up: sb.SemanticBeta, cluster: FakeES) -> None:
        """A beta embedder outage takes the beta offline with a 503, not a 500."""
        beta_up._wake = asyncio.Event()
        beta_up.client.down = True
        with pytest.raises(sb.SemanticRouteError) as err:
            asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert err.value.status_code == 503
        assert beta_up.available is False and "ConnectError" in beta_up.reason
        assert beta_up._wake.is_set()
        assert cluster.calls == []
        # Further requests are refused up front until a re-check passes.
        with pytest.raises(sb.SemanticRouteError) as err:
            asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert err.value.status_code == 503
        beta_up.client.down = False
        assert asyncio.run(beta_up.check(cluster)) is True
        asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert cluster.calls


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    @pytest.mark.parametrize("index,status,needle", [
        ("other_model_index", 409, "someone/else"),
        ("genizah_merged_v1", 409, "records no embedding model"),
        ("no_such_index", 404, "not found"),
    ])
    def test_mismatched_index_never_searched(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                                             cluster: FakeES, index: str, status: int, needle: str) -> None:
        """No embedder for the index's vectors: a clear error, no embedding, no search."""
        beta_up.client.embeds.clear()
        for call in (
            lambda: _service(cluster).search(SearchRequest(query="ketubah", index_name=index)),
            lambda: _service(cluster).search_hybrid(_hybrid_request(), index),
        ):
            with pytest.raises(sb.SemanticRouteError) as err:
                asyncio.run(call())
            assert err.value.status_code == status
            assert needle in err.value.detail
        assert cluster.calls == [] and default_client.embeds == [] and beta_up.client.embeds == []

    def test_abandoned_v9_index_refused(self, default_client: FakeEmbeddingClient, beta_up: sb.SemanticBeta,
                                        cluster: FakeES) -> None:
        """genizah_merged_v9 (older revision) is no longer the beta: no embedder serves it, 409."""
        beta_up.client.embeds.clear()
        with pytest.raises(sb.SemanticRouteError) as err:
            asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V9)))
        assert err.value.status_code == 409 and TUNED_V9[1][:12] in err.value.detail
        assert cluster.calls == [] and beta_up.client.embeds == []

    def test_beta_unavailable_is_503(self, default_client: FakeEmbeddingClient, cluster: FakeES,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """v10 before the contract check passed: 503 with the reason."""
        beta, router = _beta(cluster)
        monkeypatch.setattr(ss, "semantic_router", router)
        with pytest.raises(sb.SemanticRouteError) as err:
            asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert err.value.status_code == 503 and "has not completed" in err.value.detail
        assert cluster.calls == []

    @pytest.mark.parametrize("query", ["ketubah", "T-S 13J"])
    def test_failed_revision_check_refuses_beta(self, default_client: FakeEmbeddingClient, cluster: FakeES,
                                                monkeypatch: pytest.MonkeyPatch, query: str) -> None:
        """Embedder still on the v9 revision: the check read the v10 contract but failed, so v10 is refused."""
        beta, router = _beta(cluster, FakeEmbeddingClient(TUNED_V9))
        assert asyncio.run(beta.check(cluster)) is False
        # The contract was read, so only ``available`` keeps the beta from serving.
        assert beta.identity == TUNED and TUNED_V9[1] in beta.reason
        monkeypatch.setattr(ss, "semantic_router", router)
        for call in (
            lambda: _service(cluster).search(SearchRequest(query=query, index_name=V10)),
            lambda: _service(cluster).search_hybrid(_hybrid_request(query=query), V10),
        ):
            with pytest.raises(sb.SemanticRouteError) as err:
                asyncio.run(call())
            assert err.value.status_code == 503 and TUNED_V9[1] in err.value.detail
        assert cluster.calls == [] and beta.client.embeds == [] and default_client.embeds == []

    def test_disabled(self, default_client: FakeEmbeddingClient, cluster: FakeES,
                      monkeypatch: pytest.MonkeyPatch) -> None:
        """SEMANTIC_BETA_ENABLED=false: the check never reads ES and v10 is refused."""
        monkeypatch.setenv("SEMANTIC_BETA_ENABLED", "false")
        beta, router = _beta(cluster, enabled=sb.env_flag("SEMANTIC_BETA_ENABLED", True))
        assert asyncio.run(beta.check(cluster)) is False
        assert cluster.indices.reads == [] and beta.client.embeds == []
        assert "disabled" in beta.status().reason
        monkeypatch.setattr(ss, "semantic_router", router)
        with pytest.raises(sb.SemanticRouteError) as err:
            asyncio.run(_service(cluster).search(SearchRequest(query="ketubah", index_name=V10)))
        assert err.value.status_code == 503 and "disabled" in err.value.detail
        # v8 is untouched by the flag.
        asyncio.run(_service(cluster).search(SearchRequest(query="ketubah")))
        assert default_client.embeds == [("ketubah", "query", False)]

    @pytest.mark.parametrize("raw,expected", [
        (None, True), ("", True), ("true", True), ("1", True), ("YES", True),
        ("false", False), ("0", False), ("no", False),
    ])
    def test_env_flag(self, monkeypatch: pytest.MonkeyPatch, raw: Optional[str], expected: bool) -> None:
        """Flag parsing, default on."""
        if raw is None:
            monkeypatch.delenv("SEMANTIC_BETA_ENABLED", raising=False)
        else:
            monkeypatch.setenv("SEMANTIC_BETA_ENABLED", raw)
        assert sb.env_flag("SEMANTIC_BETA_ENABLED", True) is expected


# ---------------------------------------------------------------------------
# Contract check
# ---------------------------------------------------------------------------

class TestContractCheck:
    def test_pass(self, cluster: FakeES) -> None:
        """All checks pass: available, with the contract and eligible count reported."""
        beta, _ = _beta(cluster)
        assert asyncio.run(beta.check(cluster)) is True
        status = beta.status().model_dump()
        assert status == {
            "available": True, "index": V10, "model": TUNED[0], "revision": TUNED[1],
            "text_representation_version": "semantic_v4", "eligible_documents": V10_ELIGIBLE, "reason": None,
        }
        assert cluster.counts == [{"index": V10, "query": {"term": {"semantic_eligible": True}}}]
        # Each canary re-embedded uncached in its own mode.
        assert beta.client.embeds == [
            (CANARY_STRING, "document", False), (CANARY_STRING, "query", False), (DOC_CANARY_TEXT, "document", False),
        ]

    def test_wrong_revision(self, cluster: FakeES) -> None:
        """The embedder serves another revision."""
        beta, _ = _beta(cluster, FakeEmbeddingClient((TUNED[0], "0" * 40)))
        assert asyncio.run(beta.check(cluster)) is False
        assert "0000000000" in beta.reason and TUNED[1] in beta.reason

    def test_stock_model_on_beta_url(self, cluster: FakeES) -> None:
        """The stock model answering on the beta URL is rejected."""
        beta, _ = _beta(cluster, FakeEmbeddingClient(STOCK))
        assert asyncio.run(beta.check(cluster)) is False
        assert STOCK[0] in beta.reason

    def test_wrong_text_version(self) -> None:
        """The index was built from another text representation (v9's semantic_v3 is no longer accepted)."""
        es = FakeES({**_cluster_mappings(), V10: _mapping(_beta_meta(text_representation_version="semantic_v3"),
                                                          eligible=True)})
        beta, _ = _beta(es)
        assert asyncio.run(beta.check(es)) is False
        assert "'semantic_v3', expected 'semantic_v4'" in beta.reason
        assert beta.client.embeds == []

    def test_text_version_is_configurable(self) -> None:
        """A beta configured for another text representation accepts an index that records it."""
        es = FakeES({**_cluster_mappings(), V10: _mapping(_beta_meta(text_representation_version="semantic_v5"),
                                                          eligible=True)})
        beta, _ = _beta(es, text_representation_version="semantic_v5")
        assert asyncio.run(beta.check(es)) is True
        assert beta.status().text_representation_version == "semantic_v5"

    def test_every_canary_cosine_logged_at_info(self, cluster: FakeES, caplog: pytest.LogCaptureFixture) -> None:
        """Each of the three canary cosines is logged at INFO so the check can be reported."""
        beta, _ = _beta(cluster)
        with caplog.at_level("INFO", logger=sb.logger.name):
            assert asyncio.run(beta.check(cluster)) is True
        lines = [r.getMessage() for r in caplog.records if r.levelname == "INFO" and "cosine" in r.getMessage()]
        assert lines == [
            f"Semantic beta {V10} canary cosine 1.000000 (document mode)",
            f"Semantic beta {V10} query_canary cosine 1.000000 (query mode)",
            f"Semantic beta {V10} document_canary cosine 1.000000 (document mode)",
        ]

    def test_failing_canary_cosine_logged_at_info(self, cluster: FakeES, caplog: pytest.LogCaptureFixture) -> None:
        """A cosine below the threshold is logged at INFO too, before the check stops."""
        vectors = dict(CANARY_VECTORS)
        vectors[(CANARY_STRING, "query")] = _unit(99)
        beta, _ = _beta(cluster, FakeEmbeddingClient(TUNED, vectors))
        with caplog.at_level("INFO", logger=sb.logger.name):
            assert asyncio.run(beta.check(cluster)) is False
        info = [r.getMessage() for r in caplog.records if r.levelname == "INFO" and "cosine" in r.getMessage()]
        assert len(info) == 2 and info[1].startswith(f"Semantic beta {V10} query_canary cosine ")

    @pytest.mark.parametrize("key,label", [
        ((CANARY_STRING, "document"), "canary"),
        ((CANARY_STRING, "query"), "query_canary"),
        ((DOC_CANARY_TEXT, "document"), "document_canary"),
    ])
    def test_low_canary_cosine(self, cluster: FakeES, key: Tuple[str, str], label: str) -> None:
        """Any of the three canaries not reproduced (cosine <= 0.99) blocks the beta."""
        vectors = dict(CANARY_VECTORS)
        vectors[key] = _unit(99)
        beta, _ = _beta(cluster, FakeEmbeddingClient(TUNED, vectors))
        assert asyncio.run(beta.check(cluster)) is False
        assert beta.reason.startswith(f"{V10}: {label} cosine")

    def test_nan_canary_cosine_rejected(self, cluster: FakeES, monkeypatch: pytest.MonkeyPatch) -> None:
        """A NaN cosine (e.g. from a zero vector) fails the check like a low one."""
        beta, _ = _beta(cluster)

        async def nan_cosine(*args: Any, **kwargs: Any) -> float:
            """Report a NaN cosine.

            :param args: Ignored.
            :param kwargs: Ignored.
            :return: NaN.
            """
            return float("nan")

        monkeypatch.setattr(beta.client, "canary_cosine", nan_cosine)
        assert asyncio.run(beta.check(cluster)) is False
        assert beta.reason == f"{V10}: canary cosine nan <= {sb.CANARY_MIN_COSINE} (document mode)"

    def test_missing_canary(self) -> None:
        """A _meta without the document canary cannot be verified."""
        meta = _beta_meta()
        del meta["document_canary"]
        es = FakeES({**_cluster_mappings(), V10: _mapping(meta, eligible=True)})
        beta, _ = _beta(es)
        assert asyncio.run(beta.check(es)) is False
        assert "document_canary" in beta.reason

    def test_down_then_recovers(self, cluster: FakeES) -> None:
        """Embedder down: unavailable with a reason; recovery flips it back."""
        client = FakeEmbeddingClient(TUNED)
        client.down = True
        beta, _ = _beta(cluster, client)
        assert asyncio.run(beta.check(cluster)) is False
        assert "ConnectError" in beta.status().reason
        assert beta.status().eligible_documents == V10_ELIGIBLE  # ES-side facts still reported
        client.down = False
        assert asyncio.run(beta.check(cluster)) is True
        assert beta.status().reason is None

    def test_missing_index(self) -> None:
        """The beta index does not exist."""
        mappings = _cluster_mappings()
        del mappings[V10]
        es = FakeES(mappings)
        beta, _ = _beta(es)
        assert asyncio.run(beta.check(es)) is False
        assert "not found" in beta.reason

    def test_background_loop_retries_until_embedder_appears(self, cluster: FakeES) -> None:
        """The startup loop keeps retrying and turns available once the embedder answers."""
        client = FakeEmbeddingClient(TUNED)
        client.down = True
        beta, _ = _beta(cluster, client, retry_initial_seconds=0.01, retry_max_seconds=0.02,
                        min_check_spacing_seconds=0.0)

        async def scenario() -> bool:
            """Start, fail a few times, recover, observe.

            :return: Final availability.
            """
            beta.start(cluster)
            await asyncio.sleep(0.05)
            assert beta.available is False
            client.down = False
            for _ in range(100):
                if beta.available:
                    break
                await asyncio.sleep(0.01)
            beta.stop()
            return beta.available

        assert asyncio.run(scenario()) is True

    def test_background_loop_exits_when_disabled(self, cluster: FakeES) -> None:
        """Disabled: the loop records the reason once and ends."""
        beta, _ = _beta(cluster, enabled=False)

        async def scenario() -> bool:
            """Run the loop to completion.

            :return: Whether the task finished.
            """
            task = beta.start(cluster)
            await asyncio.wait_for(task, timeout=1.0)
            return task.done()

        assert asyncio.run(scenario()) is True
        assert beta.available is False


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

class TestPieces:
    @pytest.mark.parametrize("query,expected", [
        ("T-S 13J20.20", True), ("t-s 8j5.1", True), ("ENA 2738.1", True), ("T-S NS 144.1.", True),
        ("Or. 1080 J291", True), ("Bodl. MS heb. a 2.17", True), ("ketubah", False),
        ("letters about T-S 13J20.20", False), ("", False), ("   ", False),
        ("Mosseri collection", False), ("Moss. poetry", False), ("L-G Arabic", False),
        ("Mosseri I.1", True),
        # Prefixes and volumes
        ("T-S 13J", True), ("T-S NS 324", True), ("T-S Ar. 30", True), ("ENA 2558", True),
        ("Bodl. MS Heb. d 66", True), ("Mosseri VII", True), ("Mosseri Ia", True), ("mosseri vii", True),
        ("Moss. IV.", True),
        # Bare collection or series names (no number) stay semantic
        ("the Taylor-Schechter collection", False), ("Taylor-Schechter collection", False), ("Mosseri", False),
        ("T-S NS", False), ("ENA", False), ("Mosseri Vienna", False), ("Bodl. MS Heb.", False),
        ("Mosseri VII letters", False),
        # Only I-X with an optional "a" is a volume, never an English word
        ("Mosseri IXa", True), ("Moss. Va", True), ("Mosseri Is", False), ("Mosseri It", False),
        ("Mosseri In", False), ("Mosseri If", False), ("Mosseri Id", False), ("L-G It", False), ("L-G In", False),
    ])
    def test_looks_like_shelfmark(self, query: str, expected: bool) -> None:
        """Only a query that is entirely one shelf mark, prefix or volume counts."""
        assert sb.looks_like_shelfmark(query) is expected

    def test_beta_defaults_target_v10(self) -> None:
        """Defaults: genizah_merged_v10, semantic_v4, the v9-named embedder service."""
        assert sb.DEFAULT_BETA_SEMANTIC_INDEX == V10
        assert sb.DEFAULT_BETA_TEXT_REPRESENTATION_VERSION == "semantic_v4"
        beta = sb.SemanticBeta(index=V10, client=FakeEmbeddingClient(TUNED), enabled=True,
                               contracts=sb.IndexContractCache())
        assert beta.text_representation_version == "semantic_v4"
        assert sb.semantic_beta.text_representation_version == sb.BETA_TEXT_REPRESENTATION_VERSION

    @pytest.mark.parametrize("env,expected", [
        ({}, f"{V10} semantic_v4 http://embedding-v9:8001"),
        ({"BETA_SEMANTIC_INDEX": "genizah_merged_v11", "BETA_TEXT_REPRESENTATION_VERSION": "semantic_v5"},
         "genizah_merged_v11 semantic_v5 http://embedding-v9:8001"),
    ])
    def test_beta_settings_read_from_env(self, env: Dict[str, str], expected: str) -> None:
        """The index and the expected text version come from the environment (fresh interpreter, no network)."""
        root = Path(__file__).resolve().parents[3]
        names = ("BETA_SEMANTIC_INDEX", "BETA_TEXT_REPRESENTATION_VERSION", "BETA_EMBEDDING_SERVICE_URL")
        child_env = {k: v for k, v in os.environ.items() if k not in names}
        child_env.update(env, PYTHONPATH=str(root))
        code = ("from src.backend import semantic_beta as sb; "
                "print(sb.semantic_beta.index, sb.semantic_beta.text_representation_version, "
                "sb.semantic_beta.client.base_url)")
        done = subprocess.run([sys.executable, "-c", code], cwd=root, env=child_env, capture_output=True,
                              text=True, timeout=60, check=True)
        assert done.stdout.strip().splitlines()[-1] == expected

    def test_contract_prefers_model_name_keys(self) -> None:
        """model_name/model_revision win over the older embedding_model keys."""
        contract = sb.contract_from_mapping("x", {"x": {"mappings": _mapping(
            {"model_name": "a", "model_revision": "1", "embedding_model": "b", "embedding_model_revision": "2"},
            eligible=True)}})
        assert contract.identity == ("a", "1") and contract.eligibility_mapped is True

    def test_contract_multi_index_disagreement(self) -> None:
        """A name spanning indices with different embedders is refused."""
        stock = {"embedding_model": STOCK[0], "embedding_model_revision": STOCK[1]}
        with pytest.raises(sb.SemanticRouteError) as err:
            sb.contract_from_mapping("genizah_*", {
                V8: {"mappings": _mapping(stock)}, V10: {"mappings": _mapping(_beta_meta(), eligible=True)}})
        assert err.value.status_code == 409

    def test_contract_cache_ttl(self, cluster: FakeES) -> None:
        """Mapping reads are cached per index for the TTL."""
        now = [0.0]
        cache = sb.IndexContractCache(ttl_seconds=10.0, clock=lambda: now[0])
        cache.get(cluster, V10)
        cache.get(cluster, V10)
        assert cluster.indices.reads == [V10]
        now[0] = 11.0
        cache.get(cluster, V10)
        assert cluster.indices.reads == [V10, V10]

    def test_zero_vectors_take_real_dimension(self) -> None:
        """Vector-less hits get zero vectors of the real dimension, not 768."""
        svc = _service(RecordingSearchOnlyES())
        hits = [{"_source": {"doc_id": "a", "embedding_vector": [1.0] * 1024}}, {"_source": {"doc_id": "b"}}]
        assert [len(v) for v in svc._get_document_embeddings(hits)] == [1024, 1024]


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

class TestEndpoints:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, cluster: FakeES, default_client: FakeEmbeddingClient):
        """TestClient over the real app with the search service and beta swapped for fakes.

        :param monkeypatch: Pytest monkeypatch.
        :param cluster: Fake cluster.
        :param default_client: Fake default embedder.
        :return: ``(TestClient, beta)``.
        """
        from fastapi.testclient import TestClient
        from src.backend import app as app_module

        beta, router = _beta(cluster)
        monkeypatch.setattr(ss, "semantic_router", router)
        monkeypatch.setattr(sb, "semantic_beta", beta)
        monkeypatch.setattr(app_module, "search_service", _service(cluster))
        return TestClient(app_module.app), beta

    def test_status_shape_unavailable(self, client) -> None:
        """Before the check: available false with a reason, every contract key present."""
        tc, _ = client
        body = tc.get("/semantic-beta").json()
        assert set(body) == {"available", "index", "model", "revision", "text_representation_version",
                             "eligible_documents", "reason"}
        assert body["available"] is False and body["index"] == V10 and body["reason"]

    def test_status_available(self, client, cluster: FakeES) -> None:
        """After a passing check the payload carries the contract."""
        tc, beta = client
        asyncio.run(beta.check(cluster))
        body = tc.get("/semantic-beta").json()
        assert body["available"] is True and body["model"] == TUNED[0] and body["reason"] is None
        assert body["eligible_documents"] == V10_ELIGIBLE

    def test_search_errors_keep_their_status(self, client) -> None:
        """Route errors reach the client as 503/409, not wrapped in a 500."""
        tc, _ = client
        resp = tc.post("/search", json={"query": "ketubah", "index_name": V10})
        assert resp.status_code == 503, resp.text
        resp = tc.post("/search-hybrid", json={"query": "ketubah", "index_name": "other_model_index"})
        assert resp.status_code == 409, resp.text

    def test_beta_search_response_flag(self, client, cluster: FakeES) -> None:
        """A v10 search answers in the normal shape plus semantic_beta: true."""
        tc, beta = client
        asyncio.run(beta.check(cluster))
        cluster.hits = [{"_id": "1", "_score": 1.8, "_source": {"doc_id": "1", "embedding_vector": [0.0] * 3}}]
        body = tc.post("/search", json={"query": "ketubah", "index_name": V10}).json()
        assert body["semantic_beta"] is True and body["index_name"] == V10
        assert body["results"][0]["doc_id"] == "1"
        body = tc.post("/search", json={"query": "ketubah"}).json()
        assert body["semantic_beta"] is None and body["index_name"] == V8

    def test_beta_hybrid_shelfmark_routed(self, client, cluster: FakeES) -> None:
        """POST /search-hybrid with a bare shelf mark on v10 answers with shelf-mark results."""
        tc, beta = client
        asyncio.run(beta.check(cluster))
        cluster.hits = [SHELFMARK_HIT]
        body = tc.post("/search-hybrid", json={"query": "T-S 13J", "index_name": V10}).json()
        assert body["routed_to"] == "shelfmark" and body["semantic_beta"] is True
        assert body["index_name"] == V10 and body["results"][0]["catalogue_page"] is True
        body = tc.post("/search-hybrid", json={"query": "T-S 13J"}).json()
        assert body["routed_to"] is None and body["semantic_beta"] is None and body["index_name"] == V8
        assert body["results"][0]["catalogue_page"] is None

    def test_filters_facets(self, client, cluster: FakeES) -> None:
        """GET /filters: genres/subjects empty on v8, filled from ?index_name=genizah_merged_v10."""
        tc, _ = client
        body = tc.get("/filters").json()
        assert body["genres"] == [] and body["subjects"] == []
        assert cluster.calls[-1]["index"] == V8
        cluster.aggregations = _facet_aggregations()
        body = tc.get("/filters", params={"index_name": V10}).json()
        assert cluster.calls[-1]["index"] == V10
        assert body["genres"] == ["Letter", "Legal document"] and body["subjects"] == ["Trade"]


# ---------------------------------------------------------------------------
# Facets (backend only)
# ---------------------------------------------------------------------------

def _facet_aggregations() -> Dict[str, Any]:
    """Aggregation response of a v10-like index.

    :return: ``aggregations`` with genre and subject buckets.
    """
    return {
        "genres": {"buckets": [{"key": "Letter", "doc_count": 900}, {"key": "Legal document", "doc_count": 400},
                               {"key": " ", "doc_count": 1}]},
        "subjects": {"buckets": [{"key": "Trade", "doc_count": 50}]},
    }


class TestFacets:
    def test_aggregations_requested(self, cluster: FakeES) -> None:
        """get_filter_options aggregates genres.keyword and subjects.keyword on the requested index."""
        svc = _service(cluster)
        svc.get_filter_options(V10)
        aggs = cluster.calls[-1]["aggs"]
        assert aggs["genres"] == {"terms": {"field": "genres.keyword", "size": 100}}
        assert aggs["subjects"] == {"terms": {"field": "subjects.keyword", "size": 100}}
        assert cluster.calls[-1]["index"] == V10 and cluster.calls[-1]["size"] == 0

    def test_empty_on_v8(self, cluster: FakeES) -> None:
        """v8 has no such fields: Elasticsearch returns empty buckets, so the facets are empty."""
        cluster.aggregations = {"genres": {"buckets": []}, "subjects": {"buckets": []}}
        options = _service(cluster).get_filter_options()
        assert cluster.calls[-1]["index"] == V8
        assert options.genres == [] and options.subjects == []

    def test_filled_on_v10(self, cluster: FakeES) -> None:
        """v10 buckets become the facet values (blank keys dropped)."""
        cluster.aggregations = _facet_aggregations()
        options = _service(cluster).get_filter_options(V10)
        assert options.genres == ["Letter", "Legal document"]
        assert options.subjects == ["Trade"]

    @pytest.mark.parametrize("filters,expected", [
        ({"genres": "Letter"}, [{"term": {"genres.keyword": "Letter"}}]),
        ({"genres": ["Letter", "Poetry"]}, [{"terms": {"genres.keyword": ["Letter", "Poetry"]}}]),
        ({"subjects": "Trade"}, [{"term": {"subjects.keyword": "Trade"}}]),
    ])
    def test_filter_on_facets(self, filters: Dict[str, Any], expected: List[Dict[str, Any]]) -> None:
        """On an index that maps them, a genre / subject filter is an exact term on its .keyword field."""
        svc = _service(RecordingSearchOnlyES())
        svc._aggregatable_field = lambda index, base: f"{base}.keyword"
        assert svc._build_filters(filters, V10) == expected
        # Without a target index (or on one lacking the fields) the keys are ignored.
        assert svc._build_filters(filters) == []

    def test_no_facet_filter_is_noop(self) -> None:
        """Without a genre / subject filter the clauses are exactly what they were."""
        svc = _service(RecordingSearchOnlyES())
        clauses = svc._build_filters({"language": "Hebrew", "has_images": True})
        assert clauses == [
            {"bool": {"should": [{"term": {"language": "Hebrew"}}, {"term": {"language.keyword": "Hebrew"}}],
                      "minimum_should_match": 1}},
            {"term": {"has_images": True}},
        ]
        assert svc._build_filters(None) == [] and svc._build_filters({}) == []
