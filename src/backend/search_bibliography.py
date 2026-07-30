import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from elasticsearch import Elasticsearch
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from src.backend.embedding_client import embedding_client
from src.backend.genizah_terminology import expand_query_aliases


logger = logging.getLogger(__name__)


class BibliographySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    num_results: int = Field(default=10, ge=1, le=50)
    page: int = Field(default=1, ge=1)
    include_embeddings: bool = Field(default=False)
    index_name: Optional[str] = Field(default=None)

class BibliographyHybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    semanticWeight: int = Field(default=60, ge=0, le=100)
    keywordWeight: int = Field(default=40, ge=0, le=100)
    num_results: int = Field(default=10, ge=1, le=50)
    page: int = Field(default=1, ge=1)
    include_embeddings: bool = Field(default=False)
    index_name: Optional[str] = Field(default=None)


class BibliographySearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: Optional[float] = None
    description: Optional[str] = None
    full_text: Optional[str] = None
    shelf_marks_mentioned: Optional[List[str]] = None
    author: Optional[str] = None
    authors: Optional[List[str]] = None
    title: Optional[str] = None
    # int for ordinary pages; str for roman-numeral or range labels
    # ("xxiv-xxv") that the bibliography index also contains.
    extracted_page_number: Optional[Union[int, str]] = None
    subject_keywords: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_details: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @field_validator("extracted_page_number", mode="before")
    @classmethod
    def coerce_page_number(cls, v):
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped or None
        return v


    @field_validator('shelf_marks_mentioned', mode='before')
    @classmethod
    def convert_shelf_marks(cls, v: Union[Dict[str, Any], List[str], None]) -> Optional[List[str]]:
        """Convert shelf_marks_mentioned from dict to list if needed"""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            # If it's a dict, extract the keys (shelf mark IDs)
            return list(v.keys())
        # Fallback: convert to string and wrap in list
        return [str(v)]

    @field_validator('authors', mode='before')
    @classmethod
    def convert_authors(cls, v: Union[str, List[str], None]) -> Optional[List[str]]:
        """Convert authors from string to list if needed"""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [v]
        return None


class BibliographySearchResponse(BaseModel):
    results: List[BibliographySearchResult]
    query: Optional[str] = None
    count: int
    processing_time_ms: float
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None
    has_more: Optional[bool] = None
    index_name: Optional[str] = None



class ElasticsearchBibliographyService:
    """Semantic search service for the Genizah bibliography index using Elasticsearch."""

    def __init__(self):
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "elastic.cairogenizah.ai")
        self.es_port = os.getenv("ELASTICSEARCH_PORT", "443")
        # Default to bibliography index; can be overridden per-request
        self.index_name = os.getenv("ELASTICSEARCH_BIBLIOGRAPHY_INDEX", "genizah_bibliography_v0.0.1")
        self.es: Optional[Elasticsearch] = None
        self._initialize_elasticsearch()

    def _initialize_elasticsearch(self) -> None:
        # Add retries/timeouts to be resilient to intermittent gateway issues.
        self.es = Elasticsearch(
            [f"https://{self.es_host}:{self.es_port}"],
            basic_auth=(os.getenv("ELASTICSEARCH_USER", "cairo_user"), os.getenv("ELASTICSEARCH_PASSWORD")),
            verify_certs=False,
            retry_on_status=[429, 502, 503, 504],
            max_retries=3,
            retry_on_timeout=True,
            request_timeout=30,
        )

    def _extract_core_fields(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the bibliography fields shared by all retrieval methods.

        :param source: Elasticsearch ``_source`` dictionary.
        :returns: Normalized core bibliography fields.
        :rtype: Dict[str, Any]
        """
        return {
            "description": source.get("description"),
            "full_text": source.get("full_text_content"),
            "shelf_marks_mentioned": source.get("shelf_marks_mentioned"),
            "author": source.get("author"),
            "authors": source.get("authors"),
            "title": source.get("title"),
            "extracted_page_number": source.get("extracted_page_number"),
            "subject_keywords": source.get("subject_keywords"),
        }

    @staticmethod
    def _person_name_variants(name: str) -> List[str]:
        """Generate conservative author-name variants for exact author filters.

        Covers no-initial forms and, for full given names, the initials forms
        under which the same person is catalogued elsewhere: "Shelomo Dov
        Goitein" must also match pages attributed to "S. D. Goitein",
        "S.D. Goitein", and "Goitein, S. D.".

        :param name: Canonical or user-supplied person name.
        :returns: Deduplicated name variants suitable for exact author filters.
        :rtype: List[str]
        """
        cleaned = " ".join(name.replace("’s", "").replace("'s", "").split()).strip(" ,.;:")
        tokens = cleaned.split()
        variants = [cleaned]
        without_initials = " ".join(token for token in tokens if len(token.strip(".")) > 1)
        if without_initials:
            variants.append(without_initials)
        if len(tokens) >= 2:
            variants.append(f"{tokens[0]} {tokens[-1]}")
            surname = tokens[-1]
            given = [token.strip(".,") for token in tokens[:-1]]
            initials = [token[0].upper() for token in given if token and token[0].isalpha()]
            if initials and all(len(token) > 1 for token in given):
                dotted = " ".join(f"{initial}." for initial in initials)
                compact = "".join(f"{initial}." for initial in initials)
                variants.extend([
                    f"{dotted} {surname}",
                    f"{compact} {surname}",
                    f"{surname}, {dotted}",
                ])
        return list(dict.fromkeys(variant for variant in variants if variant))

    def _result_from_hit(
        self,
        hit: Dict[str, Any],
        similarity_score: float,
        include_embeddings: bool,
        retrieval_details: Optional[Dict[str, Any]] = None,
    ) -> BibliographySearchResult:
        """Convert an Elasticsearch hit into the public result model.

        :param hit: Elasticsearch hit dictionary.
        :param similarity_score: Retrieval score normalized to the 0-1 range.
        :param include_embeddings: Whether to return the stored embedding vector.
        :param retrieval_details: Optional component ranks and raw scores.
        :returns: Normalized bibliography result.
        :rtype: BibliographySearchResult
        """
        source = hit.get("_source", {})
        core = self._extract_core_fields(source)
        embedding = source.get("embedding_vector", []) if include_embeddings else None
        doc_id = source.get("doc_id") or hit.get("_id")
        normalized_score = max(0.0, min(float(similarity_score), 1.0))
        return BibliographySearchResult(
            doc_id=doc_id,
            similarity_score=round(normalized_score, 4),
            distance=round(1.0 - normalized_score, 4),
            description=core.get("description"),
            full_text=core.get("full_text"),
            shelf_marks_mentioned=core.get("shelf_marks_mentioned"),
            author=core.get("author"),
            authors=core.get("authors"),
            title=core.get("title"),
            extracted_page_number=core.get("extracted_page_number"),
            subject_keywords=core.get("subject_keywords"),
            metadata={
                key: value for key, value in source.items() if key != "embedding_vector"
            } | {"_es_id": hit.get("_id")},
            retrieval_details=retrieval_details or {},
            embedding=embedding,
        )

    async def search(self, request: BibliographySearchRequest) -> BibliographySearchResponse:
        """Semantic vector search over the bibliography index using cosineSimilarity on 'embedding_vector'."""
        start_time = time.time()

        try:
            # Compute query embedding
            query_embedding = await embedding_client.get_embedding(request.query, image=None, use_cache=False)

            # Base query (no filters for now)
            base_query: Dict[str, Any] = {"match_all": {}}

            # Vector similarity via script_score
            es_query = {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": query_embedding.flatten().tolist()},
                    },
                }
            }

            # Pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            search_index = request.index_name or self.index_name

            response = self.es.search(
                index=search_index,
                query=es_query,
                size=page_size,
                from_=from_offset,
                _source=True,
            )

            results: List[BibliographySearchResult] = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                core = self._extract_core_fields(source)

                embedding: Optional[List[float]] = None
                if request.include_embeddings:
                    embedding = source.get("embedding_vector", [])

                doc_id = source.get("doc_id") or hit.get("_id")

                results.append(
                    BibliographySearchResult(
                        doc_id=doc_id,
                        similarity_score=round(hit.get("_score", 0.0) - 1.0, 4),
                        distance=round(2.0 - hit.get("_score", 0.0), 4),
                        description=core.get("description"),
                        full_text=core.get("full_text"),
                        shelf_marks_mentioned=core.get("shelf_marks_mentioned"),
                        author=core.get("author"),
                        authors=core.get("authors"),
                        title=core.get("title"),
                        extracted_page_number=core.get("extracted_page_number"),
                        subject_keywords=core.get("subject_keywords"),
                        metadata={k: v for k, v in source.items() if k not in {"embedding_vector"}},
                        embedding=embedding,
                    )
                )

            processing_time = (time.time() - start_time) * 1000

            # Total hits for pagination
            total_hits_value = 0
            try:
                total_info = response.get("hits", {}).get("total")
                if isinstance(total_info, dict):
                    total_hits_value = int(total_info.get("value", 0))
                elif isinstance(total_info, int):
                    total_hits_value = int(total_info)
            except Exception:
                total_hits_value = 0

            total_pages = max(1, int(np.ceil(total_hits_value / page_size))) if page_size else 1
            has_more = (page_number * page_size) < total_hits_value

            return BibliographySearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                processing_time_ms=round(processing_time, 2),
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index,
            )

        except Exception as e:
            # Log full details server-side but return a clean message to clients
            logger.error(f"Bibliography search failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, "status_code"):
                logger.error(f"ES status code: {e.status_code}")
            if hasattr(e, "info"):
                try:
                    logger.error(f"ES error info: {json.dumps(e.info, indent=2)}")
                except Exception:
                    logger.error("ES error info present but not JSON-serializable")
            if hasattr(e, "body"):
                logger.error("ES error body received (possibly HTML from gateway)")
            clean_message = "Elasticsearch gateway error (502/5xx). Please retry in a moment."
            raise HTTPException(status_code=502 if getattr(e, "status_code", 500) in [502, 503, 504] else 500,
                                detail=clean_message)

    async def search_hybrid(self, request: BibliographyHybridSearchRequest) -> BibliographySearchResponse:
        """Fuse independent lexical and semantic rankings with weighted RRF.

        Elasticsearch lexical and cosine scores have different, query-dependent
        scales. Weighted reciprocal-rank fusion preserves each retriever's
        ordering without pretending their raw scores are directly comparable.

        :param request: Hybrid bibliography search parameters.
        :returns: Fused, paginated bibliography results.
        :rtype: BibliographySearchResponse
        """
        if request.semanticWeight + request.keywordWeight != 100:
            raise HTTPException(status_code=400, detail="Semantic and keyword weights must sum to 100")

        start_time = time.time()

        try:
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size
            candidate_size = min(200, max(40, (from_offset + page_size) * 5))
            search_index = request.index_name or self.index_name

            keyword_fields: List[str] = [
                "author^6.0",
                "authors^6.0",
                "title^4.0",
                "full_text_content^2.5",
                "description^2.0",
                "subject_keywords^1.5",
                "shelf_marks_mentioned^1.0",
            ]

            lexical_hits: List[Dict[str, Any]] = []
            semantic_hits: List[Dict[str, Any]] = []

            if request.keywordWeight > 0:
                should_clauses: List[Dict[str, Any]] = [
                    {
                        "multi_match": {
                            "query": request.query,
                            "fields": keyword_fields,
                            "type": "best_fields",
                            # Fuzziness only on longer tokens: on short ones it
                            # matched "Tisha"→"Tasha" and similar near-names.
                            "fuzziness": "AUTO",
                            "prefix_length": 2,
                        }
                    },
                    {"match_phrase": {"author": {"query": request.query, "boost": 10.0}}},
                    {"match_phrase": {"title": {"query": request.query, "boost": 6.0}}},
                ]
                # Scholarship transliterates inconsistently (qinot/kinnot) and
                # names holidays several ways; search the corpus's spellings,
                # not only the user's.
                expansion_forms = expand_query_aliases(request.query)
                for form in expansion_forms:
                    should_clauses.append(
                        {"match_phrase": {"full_text_content": {"query": form, "boost": 3.0}}}
                    )
                    should_clauses.append(
                        {"match_phrase": {"title": {"query": form, "boost": 4.0}}}
                    )
                lexical_query: Dict[str, Any] = {
                    "bool": {"should": should_clauses, "minimum_should_match": 1}
                }
                lexical_response = self.es.search(
                    index=search_index,
                    query=lexical_query,
                    size=candidate_size,
                    _source=True,
                )
                lexical_hits = lexical_response.get("hits", {}).get("hits", [])

            if request.semanticWeight > 0:
                try:
                    query_embedding = await embedding_client.get_embedding(
                        request.query,
                        image=None,
                        use_cache=True,
                    )
                    semantic_query: Dict[str, Any] = {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                                "params": {"query_vector": query_embedding.flatten().tolist()},
                            },
                        }
                    }
                    semantic_response = self.es.search(
                        index=search_index,
                        query=semantic_query,
                        size=candidate_size,
                        _source=True,
                    )
                    semantic_hits = semantic_response.get("hits", {}).get("hits", [])
                except Exception as embedding_error:
                    logger.warning(
                        "Semantic bibliography retrieval unavailable; retaining lexical results: %s",
                        embedding_error,
                    )

            rrf_k = 60.0
            max_rrf = 1.0 / (rrf_k + 1.0)
            fused: Dict[str, Dict[str, Any]] = {}

            for mode, hits, weight in (
                ("keyword", lexical_hits, request.keywordWeight / 100.0),
                ("semantic", semantic_hits, request.semanticWeight / 100.0),
            ):
                for rank, hit in enumerate(hits, start=1):
                    source = hit.get("_source", {})
                    # Fuse on the immutable ES _id: the doc_id field is not
                    # guaranteed unique (page-id extraction bugs upstream), and
                    # keying on it lets duplicates accumulate RRF mass and
                    # collapse distinct pages.
                    fusion_key = hit.get("_id") or source.get("doc_id")
                    if not fusion_key:
                        continue
                    entry = fused.setdefault(fusion_key, {
                        "hit": hit,
                        "rrf_score": 0.0,
                        "details": {},
                    })
                    entry["rrf_score"] += weight / (rrf_k + rank)
                    entry["details"][f"{mode}_rank"] = rank
                    entry["details"][f"{mode}_raw_score"] = round(float(hit.get("_score") or 0.0), 6)
                    if mode == "semantic":
                        entry["details"]["semantic_cosine"] = round(
                            float(hit.get("_score") or 1.0) - 1.0,
                            6,
                        )

            ranked_entries = sorted(
                fused.values(),
                key=lambda entry: entry["rrf_score"],
                reverse=True,
            )
            selected_entries = ranked_entries[from_offset:from_offset + page_size]
            results = [
                self._result_from_hit(
                    entry["hit"],
                    similarity_score=entry["rrf_score"] / max_rrf,
                    include_embeddings=request.include_embeddings,
                    retrieval_details={
                        "fusion": "weighted_rrf",
                        **entry["details"],
                    },
                )
                for entry in selected_entries
            ]

            processing_time = (time.time() - start_time) * 1000
            total_hits_value = len(ranked_entries)
            total_pages = max(1, int(np.ceil(total_hits_value / page_size))) if page_size else 1
            has_more = (page_number * page_size) < total_hits_value

            return BibliographySearchResponse(
                results=results,
                query=f"Hybrid: {request.query} (Semantic: {request.semanticWeight}%, Keyword: {request.keywordWeight}%)",
                count=len(results),
                processing_time_ms=round(processing_time, 2),
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index,
            )

        except Exception as e:
            logger.error(f"Bibliography hybrid search failed: {e}")
            if hasattr(e, "info"):
                try:
                    logger.error(f"ES error info: {json.dumps(e.info, indent=2)}")
                except Exception:
                    logger.error("ES error info present but not JSON-serializable")
            if hasattr(e, "body"):
                logger.error("ES error body received (possibly HTML from gateway)")
            clean_message = "Elasticsearch gateway error (502/5xx). Please retry in a moment."
            raise HTTPException(status_code=502 if getattr(e, "status_code", 500) in [502, 503, 504] else 500,
                                detail=clean_message)

    async def search_by_author(
        self,
        author_name: str,
        query: str,
        num_results: int = 8,
        index_name: Optional[str] = None,
    ) -> BibliographySearchResponse:
        """Retrieve query-relevant chunks constrained to a resolved author.

        :param author_name: Canonical graph scholar name.
        :param query: Original user query used for semantic ranking.
        :param num_results: Maximum author-constrained chunks to return.
        :param index_name: Optional bibliography index override.
        :returns: Author-constrained bibliography results.
        :rtype: BibliographySearchResponse
        """
        start_time = time.time()
        variants = self._person_name_variants(author_name)
        search_index = index_name or self.index_name

        try:
            author_filter: Dict[str, Any] = {
                "bool": {
                    "should": [
                        {"terms": {"author.keyword": variants}},
                        {"terms": {"authors": variants}},
                        *[
                            {"match_phrase": {"author": {"query": variant}}}
                            for variant in variants
                        ],
                    ],
                    "minimum_should_match": 1,
                }
            }
            retrieval_mode = "author_constrained_semantic"
            try:
                query_embedding = await embedding_client.get_embedding(query, image=None, use_cache=True)
                search_query: Dict[str, Any] = {
                    "script_score": {
                        "query": {"bool": {"filter": [author_filter]}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                            "params": {"query_vector": query_embedding.flatten().tolist()},
                        },
                    }
                }
            except Exception as embedding_error:
                logger.warning(
                    "Embedding unavailable for author-constrained search; using lexical fallback: %s",
                    embedding_error,
                )
                retrieval_mode = "author_constrained_lexical_fallback"
                search_query = {
                    "bool": {
                        "filter": [author_filter],
                        "should": [{
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^4.0",
                                    "full_text_content^2.5",
                                    "description^2.0",
                                    "subject_keywords^1.5",
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }],
                        "minimum_should_match": 0,
                    }
                }
            # Diversify by work: collapse returns the best pages of EVERY
            # authored work, so a profile can never be built from one book's
            # pages while a scholar's other works rank just below the cutoff.
            response = self.es.search(
                index=search_index,
                query=search_query,
                size=num_results,
                collapse={
                    "field": "title.keyword",
                    "inner_hits": {"name": "work_pages", "size": 4, "_source": True},
                },
                _source=True,
            )
            work_page_lists: List[List[Dict[str, Any]]] = []
            for collapsed in response.get("hits", {}).get("hits", []):
                inner = (
                    collapsed.get("inner_hits", {})
                    .get("work_pages", {})
                    .get("hits", {})
                    .get("hits", [])
                )
                work_page_lists.append(inner or [collapsed])
            # Breadth-first: two pages of each work before a third of any.
            hits: List[Dict[str, Any]] = []
            for depth in range(4):
                for pages in work_page_lists:
                    if depth < len(pages) and len(hits) < num_results:
                        hits.append(pages[depth])
            results = [
                self._result_from_hit(
                    hit,
                    similarity_score=(
                        float(hit.get("_score") or 0.0) / 2.0
                        if retrieval_mode == "author_constrained_semantic"
                        else max(0.5, 1.0 - (rank * 0.04))
                    ),
                    include_embeddings=False,
                    retrieval_details={
                        "mode": retrieval_mode,
                        "canonical_author": author_name,
                        "author_variants": variants,
                        "raw_score": round(float(hit.get("_score") or 0.0), 6),
                        **(
                            {"semantic_cosine": round(float(hit.get("_score") or 1.0) - 1.0, 6)}
                            if retrieval_mode == "author_constrained_semantic"
                            else {}
                        ),
                    },
                )
                for rank, hit in enumerate(hits)
            ]
            total_info = response.get("hits", {}).get("total", 0)
            total = int(total_info.get("value", 0)) if isinstance(total_info, dict) else int(total_info or 0)
            return BibliographySearchResponse(
                results=results,
                query=f"Author-constrained: {author_name}; semantic query: {query}",
                count=len(results),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
                total=total,
                page=1,
                page_size=num_results,
                total_pages=max(1, int(np.ceil(total / num_results))) if num_results else 1,
                has_more=total > num_results,
                index_name=search_index,
            )
        except Exception as e:
            logger.error("Author-constrained bibliography search failed for %r: %s", author_name, e)
            clean_message = "Author-constrained bibliography search failed. Please retry in a moment."
            raise HTTPException(
                status_code=502 if getattr(e, "status_code", 500) in [502, 503, 504] else 500,
                detail=clean_message,
            )


# Global instance
bibliography_search_service = ElasticsearchBibliographyService()
