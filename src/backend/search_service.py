# Enhanced search_service.py with additional metadata fields

import asyncio
import os
import re
import threading
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Set, Tuple, Union
from google.cloud import aiplatform
import logging
import time
import json
from fastapi import HTTPException, Request, status
from src.backend.embedding_client import embedding_client
from src.backend.semantic_beta import (
    SemanticRouteError,
    ShelfmarkRouteRequest,
    looks_like_shelfmark,
    semantic_eligible_clause,
    semantic_router,
)
from src.backend.shelfmark_normalizer import ShelfmarkNormalizer
from src.backend.metadata_enrichment import enrich_source
from src.backend.public_text import (
    MAX_GENRES,
    MAX_SCHOLARLY_FRAMES,
    MAX_SUBJECTS,
    clean_block_text,
    clean_inline_text,
    public_alt_titles,
    public_century,
    public_date_display,
    public_list,
    scrub_provider_mentions,
)
from src.backend.source_links import build_source_links, public_url
from elasticsearch import Elasticsearch
from src.backend.models.pydantic_core import FilterOptions
from src.backend.consensus_reads import (
    ConsensusService,
    search_enabled as consensus_search_enabled,
    to_catalogue_doc_id,
)
from functools import lru_cache
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword search fields
# ---------------------------------------------------------------------------

# Flat text fields of the older indices. In genizah_merged_v8 ``title``,
# ``transcription_full_text``, ``translation_full_text``, ``content_type`` and
# ``material`` exist on 0 docs; they stay listed so older indices still work.
KEYWORD_SEARCH_FIELDS: List[str] = [
    "transcription_full_text^3.0",
    "translation_full_text^2.5",
    "description^2.0",
    "title^2.5",
    "document_type^1.5",
    "content_type^1.5",
    "collection^1.2",
    "language^1.2",
    "script_type^1.1",
    "material^1.0",
    # genizah_merged_v10 text fields (multilingual analyzer). They are absent
    # from v8 and older indices, where multi_match simply skips them.
    "scholarly_frames^2.0",
    "subjects^1.5",
    "genres^1.0",
]
# Keyword facets (genizah_merged_v10): filter key / aggregation name -> field.
# On indices without these fields the aggregations come back empty.
FACET_FIELDS: Dict[str, str] = {
    "genres": "genres.keyword",
    "subjects": "subjects.keyword",
}
FACET_SIZE = 100
# Mosseri marks are stored as "Moss. VII,5" / "Paris Mosseri: Moss. VIII399.1"
# with doc ids "Cambridge_Mosseri_VII_5", so a routed "Mosseri VII" is also
# searched as "Moss. VII" and vice versa (ElasticsearchService.mosseri_spellings).
MOSSERI_PREFIX = re.compile(r"^(?:Mosseri|Moss\.?)\s+(?=\S)", re.IGNORECASE)
# Boost of each prefix-child clause (shelf_mark, doc_id) of a routed shelf-mark
# query: a child mark (2 x 10 + contains 4 = 24 on v10) outranks fuzzy-only
# neighbours (about 9 per spelling, so 18 for "T-S" + "TS").
SHELFMARK_CHILD_BOOST = 10
# Whole-mark regexp boost on a routed query (15 otherwise): a stored prefix
# record ("Cambridge CUL: T-S NS 32": regexp + doc_id suffix 12 + contains 4)
# must stay above a child that also matches fuzzily (about 33 on v10).
SHELFMARK_WHOLE_BOOST_PREFIX = 25
# v8 concatenation of description + scholarly transcription + translation.
# Matched without fuzziness: AUTO fuzziness over whole Hebrew transcriptions
# matched 49 docs for a word that occurs in one (measured on v8).
FULL_TEXT_CONTENT_FIELD = "full_text_content"
FULL_TEXT_CONTENT_BOOST = 1.5
# v8 scholarly transcriptions (nested; 7,777 docs): exact matches rank highest.
TRANSCRIPTIONS_PATH = "transcriptions"
TRANSCRIPTIONS_BOOST = 3.0

# ---------------------------------------------------------------------------
# Machine-reading second pass of keyword search (docs/ai_transcription_search.md)
# ---------------------------------------------------------------------------

MACHINE_READING = "machine_reading"
# Upper bound on AI hits fetched per query. The uncatalogued set holds ~800
# visible records (one hit per image after collapsing), so this covers all.
MACHINE_READING_MAX_HITS = 1000
# Per-query cache of the de-duplicated AI hit list, so paging through one
# query does not repeat the AI search and the catalogue check.
MACHINE_READING_CACHE_TTL_SECONDS = 300.0
MACHINE_READING_CACHE_MAX_ENTRIES = 256
# The second pass runs in worker threads (asyncio.to_thread), so the shared
# per-query cache is read and written under this lock.
_MACHINE_READING_CACHE_LOCK = threading.Lock()
# Elasticsearch index.max_result_window (the default; genizah_merged_v8 uses it).
MAX_RESULT_WINDOW = 10000
_HEBREW_LETTER_RE = re.compile("[א-ת]")
_CATALOGUE_MATCH_NAME = "catalogue_match"


def machine_reading_window(page_start: int, page_size: int, catalogue_total: int) -> Optional[Tuple[int, int]]:
    """Slice of the machine-reading hit list that belongs on one results page.

    Machine-reading hits are numbered after the last catalogue hit, so global
    position ``catalogue_total + i`` is machine-reading hit ``i``.

    :param page_start: Global offset of the page's first result.
    :param page_size: Results per page.
    :param catalogue_total: Number of catalogue hits for the query.
    :return: ``(ai_offset, ai_size)``, or ``None`` when the page ends at or
        before the last catalogue hit.
    :rtype: Optional[Tuple[int, int]]
    """
    page_end = page_start + page_size
    if page_end <= catalogue_total:
        return None
    ai_offset = max(0, page_start - catalogue_total)
    ai_size = page_end - max(page_start, catalogue_total)
    return ai_offset, ai_size


# ---------------------------------------------------------------------------
# Vector query bodies
# ---------------------------------------------------------------------------

COSINE_SCRIPT_SOURCE = "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0"


def build_semantic_query(
    query_vector: List[float],
    filter_clauses: List[Dict[str, Any]],
    eligible_only: bool = False,
) -> Dict[str, Any]:
    """Build the ``script_score`` cosine query of semantic search.

    :param query_vector: Query embedding.
    :param filter_clauses: Output of ``_build_filters``.
    :param eligible_only: Restrict to ``semantic_eligible: true`` records, for
        indices whose vector-less records would make ``cosineSimilarity`` fail.
    :return: The query body; byte-identical to the historical one when
        ``eligible_only`` is false.
    :rtype: Dict[str, Any]
    """
    if eligible_only:
        filter_clauses = [*filter_clauses, semantic_eligible_clause()]
    if filter_clauses:
        base_query = {"bool": {"filter": filter_clauses}}
    else:
        base_query = {"match_all": {}}
    return {
        "script_score": {
            "query": base_query,
            "script": {
                "source": COSINE_SCRIPT_SOURCE,
                "params": {"query_vector": query_vector}
            }
        }
    }


def build_hybrid_query(
    query_vector: List[float],
    filter_clauses: List[Dict[str, Any]],
    keyword_text_query: Dict[str, Any],
    semantic_weight: float,
    keyword_weight: float,
    eligible_only: bool = False,
) -> Dict[str, Any]:
    """Build the weighted ``function_score`` query of hybrid search.

    On an eligibility-mapped index the cosine function only applies to
    ``semantic_eligible: true`` records, and a vector-less record enters the
    results only through a keyword match (scored by the keyword weight
    alone). The candidate set is narrowed to "eligible OR keyword match"
    because ``function_score`` gives a record that matches no function a
    neutral factor of 1, which would rank vector-less non-matches above
    genuine semantic hits. That base query keeps a ``match_all`` scoring
    clause: a filter-only ``bool`` scores 0, and ``boost_mode: multiply``
    would zero every score (measured on genizah_merged_v9).

    :param query_vector: Query embedding.
    :param filter_clauses: Output of ``_build_filters``.
    :param keyword_text_query: Output of ``build_keyword_text_query``.
    :param semantic_weight: Weight of the cosine function (0-1).
    :param keyword_weight: Weight of the keyword function (0-1).
    :param eligible_only: Apply the eligibility rules above.
    :return: The query body; byte-identical to the historical one when
        ``eligible_only`` is false.
    :rtype: Dict[str, Any]
    """
    if eligible_only:
        cosine_filter = semantic_eligible_clause()
        candidates = {"bool": {"should": [semantic_eligible_clause(), keyword_text_query], "minimum_should_match": 1}}
        base_query = {"bool": {"must": [{"match_all": {}}], "filter": [*filter_clauses, candidates]}}
    else:
        cosine_filter = {"match_all": {}}
        if filter_clauses:
            base_query = {"bool": {"filter": filter_clauses}}
        else:
            base_query = {"match_all": {}}
    return {
        "function_score": {
            "query": base_query,
            "functions": [
                {
                    "filter": cosine_filter,
                    "weight": semantic_weight,
                    "script_score": {
                        "script": {
                            "source": COSINE_SCRIPT_SOURCE,
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                {
                    "filter": keyword_text_query,
                    "weight": keyword_weight
                }
            ],
            "score_mode": "sum",
            "boost_mode": "multiply"
        }
    }


# Enhanced Pydantic models for API with additional metadata
class DocumentMetadata(BaseModel):
    """Rich document metadata matching new ES structure"""
    doc_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    period: Optional[str] = None
    date_info: Optional[Dict[str, Any]] = None
    location: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    institution: Optional[str] = None
    library: Optional[str] = None
    collection: Optional[str] = None
    collection_type: Optional[str] = None
    shelfmark: Optional[str] = None
    shelf_mark: Optional[str] = None  # Added for compatibility
    document_types: Optional[List[str]] = None
    document_type: Optional[str] = None
    content_type: Optional[str] = None
    transcription_full_text: Optional[str] = None
    translation_full_text: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    actual_image_url: Optional[str] = None  # Added actual image URL
    tags: Optional[List[str]] = None
    has_images: Optional[bool] = None
    has_description: Optional[bool] = None
    has_transcriptions: Optional[bool] = None
    has_translations: Optional[bool] = None
    has_date: Optional[bool] = None
    has_bib: Optional[bool] = None  # Added bibliography flag
    has_joins: Optional[bool] = None  # Added joins flag
    joins_data: Optional[Dict[str, Any]] = None  # Joins data from index
    transcription_completeness: Optional[str] = None
    transcription_count: Optional[int] = None
    total_transcription_lines: Optional[int] = None
    translation_count: Optional[int] = None
    donation_year: Optional[str] = None
    donor_surnames: Optional[List[str]] = None
    source_institution: Optional[str] = None
    physical_location: Optional[str] = None
    classmark: Optional[str] = None
    provenance: Optional[str] = None
    original_url: Optional[str] = None  # Added original URL
    sub_collection: Optional[str] = None
    indexed_at: Optional[str] = None
    
    # New fields from schema
    source_collection: Optional[str] = None
    date_certainty: Optional[str] = None
    main_language: Optional[str] = None
    other_languages: Optional[List[str]] = None
    script_type: Optional[str] = None
    height: Optional[float] = None
    width: Optional[float] = None
    condition: Optional[str] = None
    extent: Optional[str] = None
    repository: Optional[str] = None
    named_entities: Optional[Dict[str, Any]] = None
    transcriptions: Optional[List[Dict[str, Any]]] = None  # Changed from List[str]
    translations: Optional[List[Dict[str, Any]]] = None   # Changed from List[str] 
    bibliography: Optional[List[Any]] = None
    # Structured, source-tagged citations (genizah_merged_v7+); ``bibliography`` keeps the raw strings.
    bibliography_entries: Optional[List[Dict[str, Any]]] = None
    image_urls: Optional[List[str]] = None
    completeness_score: Optional[float] = None
    content_quality: Optional[str] = None
    miscellaneous_info: Optional[str] = None
    index_name: Optional[str] = None

    # Public display fields (the same values reach result cards, the document
    # modal and the /fragment pages). Free text is provider-scrubbed.
    # Alternative titles: placeholders and repeats of the shelf mark or the
    # description removed, at most 10.
    alt_titles: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)  # at most 20 (genizah_merged_v10)
    genres: List[str] = Field(default_factory=list)  # at most 20 (genizah_merged_v10)
    # Catalogue headings, without repeats of alt_titles, at most 10 (genizah_merged_v10).
    scholarly_frames: List[str] = Field(default_factory=list)
    # Visible date: an ISO interval reads "1125-12-09 – 1125-12-18" (the raw
    # value stays in date_info, the fragment page's temporalCoverage).
    date_display: Optional[str] = None
    century: Optional[str] = None  # implausible values ("48th century") dropped
    # "View at the source" links: [{"source", "label", "url", "kind"}] with kind
    # item | catalogue | search (src/backend/source_links.py).
    source_links: List[Dict[str, str]] = Field(default_factory=list)


class SecondaryDocumentMetadata(BaseModel):
    """
    Metadata model for secondary source / bibliography documents
    """
    doc_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    authors: Optional[List[str]] = None
    author: Optional[str] = None
    date_display: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    extracted_page_number: Optional[Union[str, int]] = None
    footnotes: Optional[Union[Dict[str, Any], List[Any]]] = None
    isbn: Optional[str] = None
    page_number: Optional[int] = None
    primary_image_index: Optional[int] = None
    shelf_marks_mentioned: Optional[List[str]] = None
    subject_keywords: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None
    actual_image_url: Optional[str] = None
    full_text_content: Optional[str] = None
    document_type: Optional[str] = None
    index_name: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=10, ge=1, le=20, description="Number of results")
    include_embeddings: Optional[bool] = Field(default=False, description="Include embedding vectors for visualization")
    page: Optional[int] = Field(default=1, ge=1, description="Page number for pagination (1-based)")
    index_name: Optional[str] = Field(default=None, description="Elasticsearch index to search (defaults to configured index)")


class MachineReadingMatch(BaseModel):
    """Where a machine-reading search hit matched: one agreed line of one image."""
    image_index: int
    line_index: int
    snippet: str  # the matched agreed line, cleaned (consensus_reads.clean_agreed_line)
    read_url: str  # relative /read?doc=…&image=…&line=…&model=… link


class SearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: Optional[float] = None
    metadata: Optional[Union[DocumentMetadata, SecondaryDocumentMetadata]] = None
    embedding: Optional[List[float]] = None  # Added for t-SNE visualization
    # Keyword search only: "machine_reading" marks a hit appended after every
    # catalogue hit because two machine readers agree on a matching line of an
    # uncatalogued fragment. None for catalogue hits.
    match_source: Optional[str] = None
    machine_reading: Optional[MachineReadingMatch] = None
    # Opt-in semantic beta only: whether the record's doc_id is also in the
    # catalogue index, the only index the public /fragment/ page reads. None
    # for every other search (catalogue-index results always have a page).
    catalogue_page: Optional[bool] = None


class EmbeddingData(BaseModel):
    """Embedding data for t-SNE visualization"""
    query_embedding: Optional[List[float]] = None
    # Raw vectors are optional: the explorer now keeps them server-side and
    # ships only 2D coordinates (tsne/umap) to the browser.
    result_embeddings: Optional[List[List[float]]] = None
    dimension: int
    tsne: Optional[List[List[float]]] = None
    umap: Optional[List[List[float]]] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: Optional[str] = None
    count: int  # count of results returned in this page
    filters_applied: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    embedding_data: Optional[EmbeddingData] = None
    # Explorer only: server-side handle to the sampled embeddings, so
    # /calculate and /similarities can reference them without re-upload.
    sample_id: Optional[str] = None
    # Pagination metadata
    total: Optional[int] = None  # total matching documents across all pages
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None
    has_more: Optional[bool] = None
    # Index information
    index_name: Optional[str] = None  # Name of the index that was searched
    # Opt-in semantic beta (genizah_merged_v10) only: True when the fine-tuned
    # beta embedder served the vector leg, or a beta query was routed to
    # shelf-mark search; None for every other search (the default v8 path included).
    semantic_beta: Optional[bool] = None
    # Set when a semantic or hybrid query was answered by another search type
    # instead, e.g. "shelfmark" for a bare shelf mark on the beta index.
    routed_to: Optional[str] = None


class ElasticsearchService:
    """Updated Elasticsearch service with enhanced metadata extraction"""

    def __init__(self):
        self.es_host = os.getenv('ELASTICSEARCH_HOST', 'elastic.cairogenizah.ai')
        self.es_port = os.getenv('ELASTICSEARCH_PORT', '443')
        # 'http' lets a dev backend run against the local experimentation container.
        self.es_scheme = os.getenv('ELASTICSEARCH_SCHEME', 'https')
        self.index_name = os.getenv('ELASTICSEARCH_INDEX', 'cairo_genizah_text_only_v1.0.1')
        self.es = None
        # Wired by app.py (``search_service.consensus_service = ...``). While it
        # is None, keyword search never runs the machine-reading second pass.
        self.consensus_service: Optional[ConsensusService] = None
        self._initialize_elasticsearch()
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch connection for ES 8.x"""
        # ES 8.x connection
        self.es = Elasticsearch(
            [f"{self.es_scheme}://{self.es_host}:{self.es_port}"],
            basic_auth=(os.getenv('ELASTICSEARCH_USER', 'cairo_user'), os.getenv('ELASTICSEARCH_PASSWORD')),
            verify_certs=False,
        )

    def _build_filters(self, filters: Optional[Dict[str, Any]], index: Optional[str] = None) -> List[Dict]:
        """Convert search filters to Elasticsearch query clauses for new structure.

        The ``genres`` / ``subjects`` facets (:data:`FACET_FIELDS`) only filter an
        index that maps them (genizah_merged_v10); elsewhere, as before they
        existed, those keys are ignored rather than matching nothing, so a model
        or client sending them cannot empty a v8 search or chat retrieval.

        :param filters: Filter keys and values from the request.
        :param index: Index the clauses will run against.
        :returns: Elasticsearch filter clauses.
        :rtype: List[Dict]
        """
        if not filters:
            return []

        filter_clauses = []

        # Updated field mappings for new ES structure
        filter_mappings = {
            'language': 'language',
            'main_language': 'main_language',
            'institution': 'institution',
            'library': 'library',
            'repository': 'repository',
            'collection': 'collection',
            'sub_collection': 'sub_collection',
            'collection_type': 'collection_type',
            'content_type': 'content_type',
            'document_type': 'document_type',
            'has_transcriptions': 'has_transcriptions',
            'has_bib': 'has_bib',
            'has_translations': 'has_translations',
            'has_images': 'has_images',
            'has_description': 'has_description',
            'has_date': 'has_date',
            # Note: has_joins is handled specially below to check both has_joins boolean and joins_data existence
            'transcription_completeness': 'transcription_completeness',
            'donation_year': 'donation_year',
            'source_institution': 'source_institution',
            'period': 'period',
            'physical_location': 'physical_location',
            'material': 'material',
            'script_type': 'script_type',
            'date_certainty': 'date_certainty',
            # Keyword facets of genizah_merged_v10 (exact term on .keyword),
            # only where the index maps them
            **{
                key: field
                for key, field in FACET_FIELDS.items()
                if filters.get(key) is not None
                and index
                and self._aggregatable_field(index, field.removesuffix(".keyword"))
            },
        }

        for filter_key, es_field in filter_mappings.items():
            if filter_key in filters and filters[filter_key] is not None:
                value = filters[filter_key]

                # For text fields that might have .keyword variants, try both
                # This handles cases where the field might be text or keyword
                text_fields_that_need_keyword_fallback = [
                    'language', 'main_language', 'institution', 'library', 'repository',
                    'collection', 'sub_collection', 'collection_type', 'content_type',
                    'document_type', 'material', 'script_type'
                ]
                
                # Handle array fields
                if filter_key in ['document_types', 'donor_surnames', 'other_languages']:
                    if isinstance(value, list):
                        if filter_key in text_fields_that_need_keyword_fallback:
                            # Try both .keyword and regular field
                            filter_clauses.append({
                                "bool": {
                                    "should": [
                                        {"terms": {es_field: value}},
                                        {"terms": {f"{es_field}.keyword": value}}
                                    ],
                                    "minimum_should_match": 1
                                }
                            })
                        else:
                            filter_clauses.append({"terms": {es_field: value}})
                    else:
                        if filter_key in text_fields_that_need_keyword_fallback:
                            # Try both .keyword and regular field
                            filter_clauses.append({
                                "bool": {
                                    "should": [
                                        {"term": {es_field: value}},
                                        {"term": {f"{es_field}.keyword": value}}
                                    ],
                                    "minimum_should_match": 1
                                }
                            })
                        else:
                            filter_clauses.append({"term": {es_field: value}})
                # Handle other array or single values
                elif isinstance(value, list):
                    if filter_key in text_fields_that_need_keyword_fallback:
                        # Try both .keyword and regular field
                        filter_clauses.append({
                            "bool": {
                                "should": [
                                    {"terms": {es_field: value}},
                                    {"terms": {f"{es_field}.keyword": value}}
                                ],
                                "minimum_should_match": 1
                            }
                        })
                    else:
                        filter_clauses.append({"terms": {es_field: value}})
                else:
                    if filter_key in text_fields_that_need_keyword_fallback:
                        # Try both .keyword and regular field
                        filter_clauses.append({
                            "bool": {
                                "should": [
                                    {"term": {es_field: value}},
                                    {"term": {f"{es_field}.keyword": value}}
                                ],
                                "minimum_should_match": 1
                            }
                        })
                    else:
                        filter_clauses.append({"term": {es_field: value}})

        # Special handling for has_joins - check for existence of joins_data field if has_joins boolean not available
        if 'has_joins' in filters and filters['has_joins'] is not None:
            if filters['has_joins']:
                # Filter for documents that have joins data
                filter_clauses.append({
                    "bool": {
                        "should": [
                            {"term": {"has_joins": True}},
                            {"exists": {"field": "joins_data"}}
                        ],
                        "minimum_should_match": 1
                    }
                })
            # If has_joins is False, we don't add a filter (show all documents including those without joins)

        # Date range filtering
        if 'date_range' in filters:
            date_filter = filters['date_range']
            if 'start' in date_filter or 'end' in date_filter:
                range_query = {"range": {"indexed_at": {}}}
                if 'start' in date_filter:
                    range_query["range"]["indexed_at"]["gte"] = date_filter['start']
                if 'end' in date_filter:
                    range_query["range"]["indexed_at"]["lte"] = date_filter['end']
                filter_clauses.append(range_query)

        # Transcription source: scholar (human/verified), ai (beta machine read),
        # or either. Scholar transcriptions live on the merged index
        # (``has_transcriptions``); AI reads live in a side index keyed by
        # ``doc_id``, so the AI clause is a ``terms`` over the surfaced ids.
        source = filters.get('transcription_source')
        if source in ('scholar', 'ai', 'either'):
            scholar_clause = {"term": {"has_transcriptions": True}}
            if source == 'scholar':
                filter_clauses.append(scholar_clause)
            else:
                ai_ids = self._surfaced_ai_doc_ids()
                ai_clause = {"terms": {"doc_id": ai_ids}} if ai_ids else {"bool": {"must_not": {"match_all": {}}}}
                if source == 'ai':
                    filter_clauses.append(ai_clause)
                else:  # either
                    filter_clauses.append({
                        "bool": {"should": [scholar_clause, ai_clause], "minimum_should_match": 1}
                    })

        return filter_clauses

    def _surfaced_ai_doc_ids(self) -> List[str]:
        """Fragment ids that currently have a visible AI transcription.

        Lazily builds a read-only :class:`AiTranscriptionService` over this
        instance's client; the service caches the id set for a short window.

        :return: Surfaced fragment ids (empty when the feature has no data).
        :rtype: List[str]
        """
        service = getattr(self, "_ai_service", None)
        if service is None:
            from src.backend.ai_transcriptions import AiTranscriptionService
            service = self._ai_service = AiTranscriptionService(self.es)
        return service.surfaced_doc_ids()

    def _format_dimensions(self, height: Optional[float], width: Optional[float]) -> Optional[str]:
        """Format dimensions for display"""
        if height is not None and width is not None:
            return f"{height} × {width} cm"
        elif height is not None:
            return f"H: {height} cm"
        elif width is not None:
            return f"W: {width} cm"
        return None

    def _generate_title(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a meaningful title from document ID and metadata"""
        # Clean up document ID for display
        clean_id = doc_id.replace("MS-TS-", "T-S ").replace("-", ".").replace("/", " Fragment ")

        # Use description if available
        if metadata.get('description'):
            # Extract first sentence of description for title
            desc = metadata['description']
            first_sentence = desc.split('.')[0]
            if len(first_sentence) < 100:
                return f"{clean_id}: {first_sentence}"

        # Fallback to language and document type
        language = metadata.get('language', metadata.get('main_language', ''))
        doc_type = metadata.get('document_type', '')

        title_parts = [clean_id]

        if doc_type:
            title_parts.append(doc_type.title())

        if language:
            if ';' in language:
                languages = [lang.strip() for lang in language.split(';')]
                lang_display = ' & '.join(languages)
            else:
                lang_display = language
            title_parts.append(f"({lang_display})")

        return " - ".join(title_parts) if len(title_parts) > 1 else title_parts[0]

    def _generate_description(self, metadata: Dict[str, Any]) -> str:
        """Use existing description or generate from metadata"""
        # Use existing description if available
        if metadata.get('description'):
            return metadata['description']

        # Fallback generation
        parts = []

        doc_type = metadata.get('document_type', 'document')
        language = metadata.get('language', metadata.get('main_language', ''))

        if language:
            if ';' in language:
                languages = [lang.strip() for lang in language.split(';')]
                lang_display = ' and '.join(languages)
            else:
                lang_display = language
            parts.append(f"A {doc_type} in {lang_display}")
        else:
            parts.append(f"A historical {doc_type}")

        parts.append("from the Cairo Genizah collection")

        institution = metadata.get('institution', metadata.get('repository', ''))
        if institution:
            institution_display = institution.replace('_', ' ').title()
            parts.append(f"housed at {institution_display}")

        return " ".join(parts) + "."

    # Removed _generate_image_urls - we now use raw URLs from Elasticsearch

    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from metadata"""
        tags = []

        # Add language tags
        if metadata.get('language'):
            languages = metadata['language'].split(';') if ';' in metadata['language'] else [metadata['language']]
            for lang in languages:
                tags.append(lang.strip().lower().replace(' ', '-'))

        if metadata.get('main_language'):
            tags.append(metadata['main_language'].lower())

        # Add other languages
        if metadata.get('other_languages'):
            for lang in metadata['other_languages']:
                tags.append(lang.lower().replace(' ', '-'))

        # Add document type tags
        if metadata.get('document_types'):
            tags.extend(metadata['document_types'])

        if metadata.get('document_type'):
            tags.append(metadata['document_type'])

        # Add collection tags
        if metadata.get('collection'):
            tags.append(metadata['collection'])
        if metadata.get('sub_collection'):
            tags.append(metadata['sub_collection'])
        if metadata.get('collection_type'):
            tags.append(metadata['collection_type'])
        if metadata.get('content_type'):
            tags.append(metadata['content_type'])

        # Add feature tags
        if metadata.get('has_images'):
            tags.append('illustrated')
        if metadata.get('has_transcriptions'):
            tags.append('transcribed')
        if metadata.get('has_translations'):
            tags.append('translated')
        if metadata.get('has_description'):
            tags.append('described')
        if metadata.get('has_bib'):
            tags.append('bibliography')

        # Add institutional tags
        if metadata.get('institution'):
            tags.append(metadata['institution'])
        if metadata.get('repository'):
            tags.append(metadata['repository'])
        if metadata.get('source_institution'):
            tags.append(metadata['source_institution'])

        # Add script type
        if metadata.get('script_type'):
            tags.append(metadata['script_type'])

        # Add material
        if metadata.get('material'):
            tags.append(metadata['material'].lower())

        # Add transcription completeness
        if metadata.get('transcription_completeness'):
            tags.append(f"transcription-{metadata['transcription_completeness']}")

        return list(set(tags))  # Remove duplicates
    
    @staticmethod
    def extract_text_field(field_value):
        """Handle fields that are now JSON arrays but used to be strings"""
        if field_value is None:
            return None
        if isinstance(field_value, list):
            # Join multiple transcriptions/translations with newlines
            return '\n\n'.join(str(item) for item in field_value if item) if field_value else None
        return str(field_value)  # Handle case where it's still a string
    


    # Providers whose citations are attributed on the public site. Entries from
    # any other provider are served WITHOUT attribution: ``source`` is None and
    # the raw provider / contributor strings stored in the index never leave
    # the API. Do not add providers here without checking their terms of use.
    _PUBLIC_BIBLIOGRAPHY_SOURCES = ("ktiv", "pgp")

    def _public_bibliography_url(self, url: Optional[str]) -> Optional[str]:
        """Return ``url`` only when it is safe to publish.

        Citation links on the unattributed provider's hosts (its function
        pages) are never served; the host blocklist lives in
        :data:`src.backend.source_links.BLOCKED_HOSTS`.

        :param url: Raw ``bibliography.url`` value from the index.
        :return: The URL, or ``None`` when empty, not http(s), or on a blocked host.
        """
        return public_url(url)

    @staticmethod
    def _public_joins_data(joins: Any) -> Optional[Dict[str, Any]]:
        """Reduce the index's ``joins_data`` to the fields the document page shows.

        The raw object also carries the provider's page URL (``metadata.pageUrl``)
        and a contributor string (``source``, sometimes a named person); neither
        may leave the API, for the same reason as the bibliography provider
        allowlist above.

        :param joins: Raw ``joins_data`` value from Elasticsearch.
        :return: ``{"mainShelfmark", "joinedManuscripts": [{"shelfmark", "index"}]}``,
            or ``None`` when there is nothing to show.
        """
        if not isinstance(joins, dict):
            return None
        joined = [
            {"shelfmark": j.get("shelfmark"), "index": j.get("index")}
            for j in (joins.get("joinedManuscripts") or [])
            if isinstance(j, dict) and j.get("shelfmark")
        ]
        main = joins.get("mainShelfmark") or None
        if not main and not joined:
            return None
        return {"mainShelfmark": main, "joinedManuscripts": joined}

    def public_bibliography_source(self, raw_source: Optional[str]) -> Optional[str]:
        """Map a raw ``bibliography.source`` value to its public attribution.

        The single place that decides whether a citation's provider may be
        named. :meth:`_normalise_bibliography_entry` and any other caller
        (e.g. the ``/bibliography/work`` lookup in ``bibliography_works.py``)
        must go through this rather than re-checking :data:`_PUBLIC_BIBLIOGRAPHY_SOURCES`
        themselves, so the allowlist only ever lives in one place.

        :param raw_source: Raw ``bibliography.source`` (or ``None``) as stored in the index.
        :return: ``"ktiv"``, ``"pgp"``, or ``None`` for any other/unknown provider.
        """
        normalised = (raw_source or "").strip().lower()
        return normalised if normalised in self._PUBLIC_BIBLIOGRAPHY_SOURCES else None

    @staticmethod
    def _public_bibliography_text(value: Any) -> Optional[str]:
        """Provider-scrubbed form of one bibliography text value.

        Only provider mentions are removed (:func:`scrub_provider_mentions`):
        markup stays for the modal, which renders ``<em>`` itself, and a
        ``title`` / ``authors`` value stays an exact key for the
        ``/bibliography/work`` lookup.

        :param value: Raw ``title``, ``location``, author or relation value.
        :return: The scrubbed text, or ``None`` when it is not a string or the
            scrub leaves nothing (``location`` "FGP").
        :rtype: Optional[str]
        """
        return (scrub_provider_mentions(value) if isinstance(value, str) else "") or None

    @classmethod
    def _public_bibliography_texts(cls, value: Any) -> List[str]:
        """Provider-scrubbed strings of a bibliography list field.

        :param value: Raw ``authors`` / ``relations`` value (string, list or missing).
        :return: The entries :meth:`_public_bibliography_text` keeps, order kept.
        :rtype: List[str]
        """
        raw = value if isinstance(value, (list, tuple)) else [value]
        texts = (cls._public_bibliography_text(item) for item in raw)
        return [text for text in texts if text]

    def _normalise_bibliography_entry(self, bib: Dict[str, Any], citation: str) -> Dict[str, Any]:
        """Shape one index bibliography entry for the document page.

        Free text is provider-scrubbed like the description, so no citation,
        title, location, author or relation names the unattributed provider.

        :param bib: Raw nested ``bibliography`` object from Elasticsearch (may be empty
            for legacy string-only entries).
        :param citation: Display string already derived for the legacy ``bibliography`` list.
        :return: Dict with ``citation``, ``source`` (``ktiv`` | ``pgp`` | ``None`` for
            unattributed entries), ``title``, ``authors``, ``year``, ``location``,
            ``relations`` and ``url``. The index's raw ``source`` value is never included.
        """
        return {
            "citation": scrub_provider_mentions(citation),
            "source": self.public_bibliography_source(bib.get("source")),
            "title": self._public_bibliography_text(bib.get("title")),
            "authors": self._public_bibliography_texts(bib.get("authors")),
            "year": bib.get("year") or None,
            "location": self._public_bibliography_text(bib.get("location")),
            "relations": self._public_bibliography_texts(bib.get("relations")),
            "url": self._public_bibliography_url(bib.get("url")),
        }

    @staticmethod
    def _public_dates(source: Dict[str, Any]) -> Dict[str, str]:
        """Public date fields of a merged-index record.

        :param source: Elasticsearch ``_source``.
        :return: The non-empty ``date_display``, ``date_start``, ``date_end`` and
            ``century`` values, cleaned; an implausible century is left out.
        :rtype: Dict[str, str]
        """
        dates = {
            key: clean_inline_text(source[key])
            for key in ("date_display", "date_start", "date_end")
            if isinstance(source.get(key), str)
        }
        dates["century"] = public_century(source.get("century")) or ""
        return {key: value for key, value in dates.items() if value}

    def _extract_metadata(self, source: Dict[str, Any], index_name: Optional[str] = None) -> DocumentMetadata:
        """Build the public :class:`DocumentMetadata` of one catalogue record.

        The single path from an Elasticsearch ``_source`` to what the result
        cards, the document modal and the ``/fragment`` pages show: provider
        data is filtered here (bibliography sources and URLs, joins, provider
        mentions in the description and the display fields), and the display
        fields (alternative titles, subjects, genres, catalogue headings,
        dates, document type, "View at the source" links) are shaped here.

        :param source: Elasticsearch ``_source`` (heavy fields may be excluded).
        :param index_name: Index the record came from.
        :return: Public metadata.
        :rtype: DocumentMetadata
        """
        # Handle bibliography field which can be a list of objects or strings
        bibliography_list = []
        bibliography_entries: List[Dict[str, Any]] = []
        bibliography_raw = source.get('bibliography', [])
        if bibliography_raw and isinstance(bibliography_raw, list):
            for bib in bibliography_raw:
                if isinstance(bib, dict):
                    # Try to construct a meaningful citation string
                    citation = (bib.get('citation') or 
                              bib.get('reference') or 
                              bib.get('text') or 
                              str(bib))
                    bibliography_list.append(scrub_provider_mentions(citation))
                    bibliography_entries.append(self._normalise_bibliography_entry(bib, citation))
                else:
                    bibliography_list.append(scrub_provider_mentions(str(bib)))
                    bibliography_entries.append(self._normalise_bibliography_entry({}, str(bib)))

        # Handle dimensions which might be a string or object
        dimensions = source.get('dimensions')
        if isinstance(dimensions, dict):
            # If it's a dict (e.g. {"height": ..., "width": ...}), convert to string representation
            parts = []
            if dimensions.get('height'): parts.append(f"H: {dimensions['height']}")
            if dimensions.get('width'): parts.append(f"W: {dimensions['width']}")
            dimensions = ", ".join(parts) if parts else None

        # Handle description which might be a list
        description = source.get('description')
        if isinstance(description, list):
            description = " ".join([str(d) for d in description if d])
        if description:
            description = clean_block_text(str(description))

        # Handle transcription text
        transcription_text = source.get('transcription_full_text')
        transcriptions_raw = source.get('transcriptions') or source.get('transcription')
        
        if not transcription_text and transcriptions_raw:
            if isinstance(transcriptions_raw, list):
                texts = []
                for trans in transcriptions_raw:
                    if isinstance(trans, dict):
                        # Extract text from transcription objects
                        text = (trans.get('text') or 
                            trans.get('content') or 
                            trans.get('transcription') or 
                            trans.get('value') or 
                            str(trans))
                        texts.append(text)
                    else:
                        texts.append(str(trans))
                transcription_text = '\n\n'.join(texts) if texts else None
            else:
                transcription_text = str(transcriptions_raw)
            
        # Handle translation text
        translation_text = source.get('translation_full_text')
        translations_raw = source.get('translations') or source.get('translation')
        
        if not translation_text and translations_raw:
            if isinstance(translations_raw, list):
                texts = []
                for trans in translations_raw:
                    if isinstance(trans, dict):
                        # Extract text from translation objects
                        text = (trans.get('text') or 
                            trans.get('content') or 
                            trans.get('translation') or
                            trans.get('value') or
                            str(trans))
                        texts.append(text)
                    else:
                        texts.append(str(trans))
                translation_text = '\n\n'.join(texts) if texts else None
            else:
                translation_text = str(translations_raw)

        # Handle image URLs
        # Logic to extract the best image URL
        image_url = source.get('image_url')
        thumbnail_url = source.get('thumbnail_url')
        
        # Process tags - ensure it's a list of strings
        tags = source.get('tags', [])
        if tags and isinstance(tags, str):
            tags = [tags]
            
        # Process transcriptions/translations raw data
        transcriptions_raw = source.get('transcriptions')
        translations_raw = source.get('translations')
        
        doc_id = source.get('doc_id', 'Unknown')

        # Ensure image_urls is populated for primary sources
        image_urls = source.get('image_urls')
        if not image_urls:
            # Fallback to single image fields
            single_image = source.get('actual_image_url') or source.get('image_url')
            if single_image:
                image_urls = [single_image]

        shelf_mark = source.get('shelf_mark', source.get('classmark', doc_id))
        raw_alt_titles = source.get('alt_titles')
        alt_titles = public_alt_titles(raw_alt_titles, str(shelf_mark or ""), description)
        # Catalogue headings never repeat an alternative title, shown or not.
        raw_alt_list = raw_alt_titles if isinstance(raw_alt_titles, list) else [raw_alt_titles]
        frames_exclude = alt_titles + [t for t in raw_alt_list if isinstance(t, str)]
        dates = self._public_dates(source)
        raw_document_type = source.get('document_type')
        document_type = clean_inline_text(raw_document_type) if isinstance(raw_document_type, str) else ""

        return DocumentMetadata(
            doc_id=doc_id,
            title=source.get('title'),
            description=description,
            language=source.get('language'),
            main_language=source.get('main_language'),
            other_languages=source.get('other_languages'),
            period=source.get('period'),
            date_info=source.get('date_info') or dates or None,
            # Visible form (ISO interval "a – b"); date_info keeps the raw value.
            date_display=public_date_display(dates.get('date_display')),
            century=dates.get('century'),
            location=source.get('physical_location'),
            material=source.get('material'),
            dimensions=dimensions,
            height=source.get('height'),
            width=source.get('width'),
            condition=source.get('condition'),
            extent=source.get('extent'),
            institution=source.get('institution'),
            repository=source.get('repository'),
            library=source.get('library'),
            collection=source.get('collection'),
            source_collection=source.get('source_collection'),
            collection_type=source.get('collection_type'),
            shelfmark=source.get('classmark', source.get('shelf_mark', doc_id)),
            shelf_mark=shelf_mark,
            document_types=source.get('document_types'),
            document_type=document_type or None,
            alt_titles=alt_titles,
            subjects=public_list(source.get('subjects'), MAX_SUBJECTS),
            genres=public_list(source.get('genres'), MAX_GENRES),
            scholarly_frames=public_list(source.get('scholarly_frames'), MAX_SCHOLARLY_FRAMES, exclude=frames_exclude),
            source_links=build_source_links(source),
            sub_collection=source.get('sub_collection'),
            bibliography=bibliography_list,
            bibliography_entries=bibliography_entries or None,
            transcription_full_text=transcription_text,
            translation_full_text=translation_text,
            miscellaneous_info=source.get('miscellaneous_info'),
            index_name=index_name,
            image_urls=image_urls,
            actual_image_url=source.get('actual_image_url') or source.get('image_url'),
            has_joins=source.get('has_joins'),
            joins_data=self._public_joins_data(source.get('joins_data'))
        )

    def _extract_secondary_metadata(self, source: Dict[str, Any], index_name: str) -> SecondaryDocumentMetadata:
        """Extract metadata for secondary source documents"""
        return SecondaryDocumentMetadata(
            doc_id=source.get('doc_id'),
            title=source.get('title'),
            description=source.get('description'),
            authors=source.get('authors'),
            author=source.get('author'),
            date_display=source.get('date_display'),
            date_end=source.get('date_end'),
            date_start=source.get('date_start'),
            extracted_page_number=source.get('extracted_page_number'),
            footnotes=source.get('footnotes'),
            isbn=source.get('isbn'),
            page_number=source.get('page_number'),
            primary_image_index=source.get('primary_image_index'),
            shelf_marks_mentioned=source.get('shelf_marks_mentioned'),
            subject_keywords=source.get('subject_keywords'),
            image_urls=source.get('image_urls'),
            actual_image_url=source.get('actual_image_url'),
            full_text_content=source.get('full_text_content'),
            document_type=source.get('document_type'),
            index_name=index_name
        )

    def generate_iiif_manifest(self, doc_id: str, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a IIIF Presentation 2.1 manifest for a document"""
        document = self.get_document_by_id(doc_id, index_name)
        if not document:
            return None

        # Base URL for the API (should be configured properly in production)
        api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')

        # Get images
        images = []

        # Prioritize the list of all images
        if document.image_urls:
            # Clean URLs similar to frontend logic
            for url in document.image_urls:
                if url and isinstance(url, str):
                    cleaned = url.split()[0]
                    if cleaned and not cleaned.endswith('w'):
                        images.append(cleaned)

        if not images and document.image_url:
            images.append(document.image_url)

        if not images:
            # Fallback placeholder if really needed, or return empty manifest
            return None

        manifest_id = f"{api_base_url}/document/{doc_id}/manifest"

        canvases = []
        for i, img_url in enumerate(images):
            canvas_id = f"{manifest_id}/canvas/{i}"

            # Simple canvas generation without deep zoom tiles for now
            # In a real IIIF setup, we'd have an image server (IIIF Image API)
            # Here we are just pointing to static images (Level 0 compliance-ish)

            canvas = {
                "@id": canvas_id,
                "@type": "sc:Canvas",
                "label": f"Page {i + 1}",
                "height": 1000,  # Placeholder dimensions as we might not know them
                "width": 1000,
                "images": [
                    {
                        "@type": "oa:Annotation",
                        "motivation": "sc:painting",
                        "on": canvas_id,
                        "resource": {
                            "@id": img_url,
                            "@type": "dctypes:Image",
                            "format": "image/jpeg",
                            "height": 1000,
                            "width": 1000
                        }
                    }
                ]
            }
            canvases.append(canvas)

        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@id": manifest_id,
            "@type": "sc:Manifest",
            "label": document.title or f"Document {doc_id}",
            "description": document.description or "",
            "metadata": [
                {"label": "Shelfmark", "value": document.shelfmark or doc_id},
                {"label": "Collection", "value": document.collection or "Unknown"},
                {"label": "Language", "value": document.language or "Unknown"}
            ],
            "sequences": [
                {
                    "@id": f"{manifest_id}/sequence/normal",
                    "@type": "sc:Sequence",
                    "label": "Current Page Order",
                    "canvases": canvases
                }
            ]
        }

        return manifest

    def _get_document_embeddings(self, hits: List[Dict], fallback_dimension: int = 768) -> List[List[float]]:
        """Extract embedding vectors from Elasticsearch hits.

        :param hits: Elasticsearch hits.
        :param fallback_dimension: Zero-vector length when no hit carries a vector.
        :return: One vector per hit, zero-filled for vector-less records.
        :rtype: List[List[float]]
        """
        embeddings = []
        # Zero vectors for vector-less records (e.g. keyword-only hybrid hits on
        # an eligibility-mapped index) take the dimension of the real ones.
        dimension = next(
            (len(hit["_source"]["embedding_vector"]) for hit in hits if hit["_source"].get("embedding_vector")),
            fallback_dimension,
        )
        for hit in hits:
            # Get the embedding from the document source
            embedding = hit["_source"].get("embedding_vector", [])
            if embedding:
                embeddings.append(embedding)
            else:
                # If no embedding found, create a zero vector or skip
                logger.warning(f"No embedding found for document {hit['_source'].get('doc_id', 'unknown')}")
                # You might want to generate an embedding on the fly or use a default
                embeddings.append([0.0] * dimension)
        return embeddings

    async def _beta_shelfmark_search(self, request: Any, search_index: str) -> SearchResponse:
        """Answer a bare shelf-mark (prefix) query on the beta index with shelf-mark search.

        Shelf marks were masked out of the beta vectors, so a cosine search
        for one would return noise. The callers fall back to their own search
        when this finds nothing (``total`` 0).

        :param request: The semantic or hybrid search request (``query``,
            ``filters``, ``num_results``, ``page``).
        :param search_index: The beta index.
        :return: One page of shelf-mark results marked ``routed_to="shelfmark"``.
        :rtype: SearchResponse
        """
        shelfmark_request = ShelfmarkRouteRequest.from_query(
            request.query, request.filters, request.num_results, getattr(request, "page", None)
        )
        response = await self.search_by_shelfmark(shelfmark_request, search_index)
        self._mark_catalogue_pages(response.results)
        response.semantic_beta = True
        response.routed_to = "shelfmark"
        return response

    def _mark_catalogue_pages(self, results: List[SearchResult]) -> None:
        """Set ``catalogue_page`` on beta results: is the record in the catalogue index?

        The public ``/fragment/`` page reads only the catalogue index
        (``self.index_name``, genizah_merged_v8), and some genizah_merged_v10
        ids are not in it (6,368 of 79,524), so the frontend links a beta
        result only when this is true. An old-form Oxford id counts when its
        catalogue form exists, since the page redirects it.

        :param results: One page of beta results, updated in place.
        """
        forms = {result.doc_id: {result.doc_id, to_catalogue_doc_id(result.doc_id)} for result in results}
        ids = sorted(set().union(*forms.values()))
        if not ids:
            return
        response = self.es.search(
            index=self.index_name, query={"terms": {"doc_id": ids}}, size=len(ids), _source=["doc_id"]
        )
        present = {hit["_source"].get("doc_id") for hit in response["hits"]["hits"]}
        for result in results:
            result.catalogue_page = bool(forms[result.doc_id] & present)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform vector similarity search with optional embedding data for visualization.

        The query is embedded by the embedder that built the target index
        (``semantic_router``): the default index keeps the default embedder
        unchanged; the opt-in beta index uses the beta embedder, filters to
        ``semantic_eligible`` records and answers bare shelf marks, shelf-mark
        prefixes and volumes with shelf-mark search (when it finds any).

        :param request: Search request; ``index_name`` selects the index.
        :return: Ranked results with pagination metadata.
        :rtype: SearchResponse
        :raises SemanticRouteError: When no verified embedder matches the index.
        :raises HTTPException: 500 when the search itself fails.
        """
        start_time = time.time()
        # Use provided index or default to configured index
        search_index = getattr(request, "index_name", None) or self.index_name
        route = await semantic_router.resolve(self.es, search_index, self.index_name, embedding_client)
        if route.beta and looks_like_shelfmark(request.query):
            routed = await self._beta_shelfmark_search(request, search_index)
            if routed.total:
                return routed
            # No record has that shelf mark: run the query as typed.

        try:
            # Generate query embedding with the index's embedder
            query_embedding = await semantic_router.embed_query(route, request.query)

            # Build filter clauses
            filter_clauses = self._build_filters(request.filters, search_index)

            # Build Elasticsearch query using script_score for vector similarity
            query = build_semantic_query(
                query_embedding.flatten().tolist(), filter_clauses, eligible_only=route.eligible_only
            )

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Execute search using ES 8.x syntax
            response = self.es.search(
                index=search_index,
                query=query,
                size=page_size,
                from_=from_offset,
                _source=True  # Ensure we get the full source including embeddings
            )

            # Extract embeddings if requested
            embedding_data = None
            if request.include_embeddings and response['hits']['hits']:
                result_embeddings = self._get_document_embeddings(response['hits']['hits'])
                embedding_data = EmbeddingData(
                    query_embedding=query_embedding.flatten().tolist(),
                    result_embeddings=result_embeddings,
                    dimension=len(query_embedding.flatten())
                )

            # Format results with rich metadata
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                # Debug logging for image URLs
                logger.info(f"[SEARCH] Document {source.get('doc_id', hit['_id'])}:")
                logger.info(f"  - Has actual_image_url: {bool(source.get('actual_image_url'))}")
                logger.info(f"  - Has image_urls: {bool(source.get('image_urls'))}")
                logger.info(f"  - image_urls type: {type(source.get('image_urls'))}")
                logger.info(f"  - image_urls length: {len(source.get('image_urls', [])) if isinstance(source.get('image_urls'), list) else 'N/A'}")
                if source.get('image_urls'):
                    logger.info(f"  - image_urls value: {source.get('image_urls')}")

                # Include embedding in result if requested
                embedding = None
                if request.include_embeddings:
                    embedding = source.get("embedding_vector", [])

                doc_id = source.get("doc_id") or hit["_id"]

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=round(hit["_score"] - 1.0, 4),
                    distance=round(2.0 - hit["_score"], 4),
                    metadata=metadata,
                    embedding=embedding
                ))
            if route.beta:
                self._mark_catalogue_pages(results)

            processing_time = (time.time() - start_time) * 1000

            # Total hits for pagination
            total_hits_value = 0
            try:
                total_info = response.get('hits', {}).get('total')
                if isinstance(total_info, dict):
                    total_hits_value = int(total_info.get('value', 0))
                elif isinstance(total_info, int):
                    total_hits_value = int(total_info)
            except Exception:
                total_hits_value = 0

            total_pages = max(1, int(np.ceil(total_hits_value / page_size))) if page_size else 1
            has_more = (page_number * page_size) < total_hits_value

            # Combine filters for response
            filters_applied = {"search_type": "semantic"}
            req_filters = getattr(request, 'filters', None)
            if req_filters:
                filters_applied.update(req_filters)

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied=filters_applied,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data,
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index,
                semantic_beta=True if route.beta else None
            )

        except SemanticRouteError:
            raise
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            logger.error(f"Full ES error type: {type(e).__name__}")
            logger.error(f"Full ES error message: {str(e)}")
            
            if hasattr(e, 'info'):
                logger.error(f"ES error info: {json.dumps(e.info, indent=2)}")
            if hasattr(e, 'body'):
                logger.error(f"ES error body: {e.body}")
            if hasattr(e, 'status_code'):
                logger.error(f"ES status code: {e.status_code}")
                
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    def _catalogue_record(self, source: Dict[str, Any], target_index: str, enrich: bool) -> DocumentMetadata:
        """Public metadata of one record shown on its own (document modal).

        With ``enrich``, a record from the catalogue index (``self.index_name``)
        gets its alternative titles, subjects, genres and catalogue headings
        from the enrichment index when that has them (one GET, never an error).

        :param source: Elasticsearch ``_source``.
        :param target_index: Index the record came from.
        :param enrich: Whether to read the enrichment index.
        :return: Public metadata.
        :rtype: DocumentMetadata
        """
        if enrich and target_index == self.index_name:
            source = enrich_source(self.es, source, target_index)
        return self._extract_metadata(source, target_index)

    def get_document_by_id(
        self, doc_id: str, index_name: Optional[str] = None, enrich: bool = False
    ) -> Optional[Union[DocumentMetadata, SecondaryDocumentMetadata]]:
        """Full public details of one record.

        Tries an exact ``doc_id`` match first, then shelf/class marks and the
        Elasticsearch ``_id``.  The embedding is never fetched.

        :param doc_id: Catalogue id (or a shelf mark).
        :param index_name: Index to read (default: the catalogue index).
        :param enrich: Complete a catalogue record's display fields from the
            enrichment index (one extra GET).  Only ``GET /document/{doc_id}``
            asks for it; internal callers such as the IIIF manifest do not
            read those fields.
        :return: The record's metadata, or ``None`` when absent or on errors.
        :rtype: Optional[Union[DocumentMetadata, SecondaryDocumentMetadata]]
        """
        try:
            target_index = index_name or self.index_name
            
            # First: exact doc_id (keyword or text)
            response = self.es.search(
                index=target_index,
                query={
                    "bool": {
                        "should": [
                            {"term": {"doc_id": doc_id}},
                            {"term": {"doc_id.keyword": doc_id}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                size=1,
                source_excludes=["embedding_vector"],
            )
            
            if response['hits']['total']['value'] > 0:
                source = response['hits']['hits'][0]['_source']
                
                # Check if this is a bibliography/secondary source index
                if 'bibliography' in target_index:
                    return self._extract_secondary_metadata(source, target_index)
                else:
                    return self._catalogue_record(source, target_index, enrich)
            # Second: try shelf/class marks and also ES _id
            response = self.es.search(
                index=target_index,
                query={
                    "bool": {
                        "should": [
                            {"ids": {"values": [doc_id]}},
                            {"term": {"shelf_mark.keyword": doc_id}},
                            {"term": {"shelf_mark": doc_id}},
                            {"term": {"shelfmark.keyword": doc_id}},
                            {"term": {"shelfmark": doc_id}},
                            {"term": {"classmark.keyword": doc_id}},
                            {"term": {"classmark": doc_id}},
                            {"match_phrase": {"shelf_mark": doc_id}},
                            {"match_phrase": {"shelfmark": doc_id}},
                            {"match_phrase": {"classmark": doc_id}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                size=1,
                source_excludes=["embedding_vector"],
            )

            if response['hits']['total']['value'] > 0:
                source = response['hits']['hits'][0]['_source']
                return self._catalogue_record(source, target_index, enrich)

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def get_filter_options(self, index_name: Optional[str] = None) -> FilterOptions:
        """Get the available filter options (facet values) of an index.

        The ``genres`` / ``subjects`` facets (:data:`FACET_FIELDS`) exist on
        genizah_merged_v10; on v8 and older indices their aggregations are
        empty, so those lists come back empty.

        :param index_name: Index to aggregate; defaults to the configured index.
        :return: Facet values; empty lists when Elasticsearch fails.
        :rtype: FilterOptions
        """
        try:
            # Simple aggregations - language is keyword, collection.keyword works
            aggs = {
                "languages": {"terms": {"field": "language", "size": 100}},
                "document_types": {"terms": {"field": "document_type.keyword", "size": 100}},
                "collections": {
                    "terms": {"field": "collection.keyword", "size": 100},
                    "aggs": {
                        "sub_collections": {
                            "terms": {"field": "sub_collection.keyword", "size": 100}
                        }
                    }
                },
                **{name: {"terms": {"field": field, "size": FACET_SIZE}} for name, field in FACET_FIELDS.items()},
            }

            response = self.es.search(
                index=index_name or self.index_name,
                size=0,
                aggs=aggs
            )

            aggregations = response.get("aggregations", {})
            
            # Extract languages (filter out empty/Unknown)
            languages = [
                bucket["key"] for bucket in aggregations.get("languages", {}).get("buckets", [])
                if bucket.get("key") and bucket["key"] != "Unknown" and str(bucket["key"]).strip()
            ]
            
            # Extract document types (filter out empty/Unknown)
            document_types = [
                bucket["key"] for bucket in aggregations.get("document_types", {}).get("buckets", [])
                if bucket.get("key") and bucket["key"] != "Unknown" and str(bucket["key"]).strip()
            ]
            
            # Extract collections (filter out empty/Unknown)
            collections = [
                bucket["key"] for bucket in aggregations.get("collections", {}).get("buckets", [])
                if bucket.get("key") and bucket["key"] != "Unknown" and str(bucket["key"]).strip()
            ]
            
            # Extract sub_collections grouped by collection
            sub_collections_dict = {}
            for coll_bucket in aggregations.get("collections", {}).get("buckets", []):
                collection_name = coll_bucket.get("key")
                if collection_name and collection_name != "Unknown" and str(collection_name).strip():
                    sub_collection_buckets = coll_bucket.get("sub_collections", {}).get("buckets", [])
                    if sub_collection_buckets:
                        sub_collections_dict[collection_name] = [
                            sub_bucket["key"] for sub_bucket in sub_collection_buckets
                            if sub_bucket.get("key") and sub_bucket["key"] != "Unknown" and str(sub_bucket["key"]).strip()
                        ]

            # Keyword facets (empty on indices without the fields)
            facets = {
                name: [
                    bucket["key"] for bucket in aggregations.get(name, {}).get("buckets", [])
                    if bucket.get("key") and str(bucket["key"]).strip()
                ]
                for name in FACET_FIELDS
            }

            return FilterOptions(
                languages=languages,
                periods=['early_medieval', 'late_medieval', 'early_modern'],
                document_types=document_types,
                institutions=[],  # Removed - redundant with collection
                collections=collections,
                sub_collections=sub_collections_dict if sub_collections_dict else None,
                **facets,
            )
        except Exception as e:
            logger.error(f"Could not get filter options from Elasticsearch: {e}", exc_info=True)
            return FilterOptions(
                languages=[],
                periods=['early_medieval', 'late_medieval', 'early_modern'],
                document_types=[],
                institutions=[],
                collections=[],
                sub_collections=None
            )

    @staticmethod
    def normalize_shelfmark(shelfmark: str) -> str:
        """Normalize shelf mark for search. Delegates to ShelfmarkNormalizer."""
        return ShelfmarkNormalizer.normalize_for_search(shelfmark)
    
    @staticmethod
    def get_search_variants(shelfmark: str) -> List[str]:
        """
        Get search variants for a shelf mark to help with liberal matching.
        
        Returns both the normalized version and variants that might match
        documents in the corpus. For example:
        - "Rylands Genizah Fragment 1" -> ["Manchester: Rylands Genizah Fragment 1", "Rylands Genizah Fragment 1"]
        - "T-S AS 1" -> ["T-S AS 1", "TS AS 1"]
        
        Args:
            shelfmark: Shelf mark string
            
        Returns:
            List of search variants to try
        """
        if not shelfmark:
            return [shelfmark]
        
        variants = set()
        normalized = shelfmark.strip()
        
        # Add normalized version
        normalized_version = ElasticsearchService.normalize_shelfmark(normalized)
        variants.add(normalized_version)
        variants.add(normalized)  # Also try original
        
        # Collection-specific variants
        
        # Manchester: If normalized added "Manchester: " prefix, also try without
        if normalized_version.startswith("Manchester: "):
            core = normalized_version[len("Manchester: "):].strip()
            variants.add(core)
            variants.add(f"Manchester {core}")  # Without colon
        
        # TS: Try both T-S and TS variants
        if re.search(r'\bT-S\b', normalized_version, re.IGNORECASE):
            variants.add(re.sub(r'\bT-S\b', 'TS', normalized_version, flags=re.IGNORECASE))
        if re.search(r'\bTS\b', normalized_version, re.IGNORECASE):
            variants.add(re.sub(r'\bTS\b', 'T-S', normalized_version, flags=re.IGNORECASE))
        
        return list(variants)

    @staticmethod
    def mosseri_spellings(shelfmark: str) -> List[str]:
        """Both spellings of a Mosseri shelf mark or volume.

        The catalogue stores ``"Moss. VII,5"`` (doc id
        ``Cambridge_Mosseri_VII_5``), while visitors type ``"Mosseri VII"``.

        :param shelfmark: Shelf mark as typed, e.g. ``"Mosseri VII"`` or ``"Moss. Ia"``.
        :return: ``["Moss. <rest>", "Mosseri <rest>"]``, or ``[]`` for other collections.
        :rtype: List[str]
        """
        match = MOSSERI_PREFIX.match((shelfmark or "").strip())
        if match is None:
            return []
        rest = shelfmark.strip()[match.end():]
        return [f"Moss. {rest}", f"Mosseri {rest}"]

    async def search_by_shelfmark(self, request, index_name: Optional[str] = None) -> SearchResponse:
        """Search documents by shelf mark with exact or partial matching.

        Requests of the shelf-mark endpoint and of chat retrieval get the best
        ``num_results`` of the top 30 hits as one page. A routed beta request
        (:class:`ShelfmarkRouteRequest`, which sets ``prefix`` and ``page``)
        also ranks the child marks of a prefix or volume first, tries both
        Mosseri spellings, and is paged with ``page``.

        :param request: Shelf-mark request (``shelf_mark``, ``exact_match``,
            ``num_results``, optional ``filters`` / ``include_embeddings`` /
            ``prefix`` / ``page``).
        :param index_name: Index to search; defaults to the configured index.
        :return: Ranked results with pagination metadata.
        :rtype: SearchResponse
        :raises HTTPException: 500 when the search fails.
        """
        start_time = time.time()
        prefix = getattr(request, "prefix", False)
        page = getattr(request, "page", None)
        page_size = request.num_results or 10

        try:
            # Normalize the input shelf mark
            normalized_shelfmark = self.normalize_shelfmark(request.shelf_mark)
            logger.info(f"Normalized shelf mark: '{request.shelf_mark}' -> '{normalized_shelfmark}'")
            
            # For partial matching, get search variants to try
            search_variants = []
            if not request.exact_match:
                search_variants = self.get_search_variants(request.shelf_mark)
                if prefix:
                    search_variants += self.mosseri_spellings(request.shelf_mark)
                # Remove duplicates and the normalized version (we'll add it separately)
                search_variants = [v for v in dict.fromkeys(search_variants) if v != normalized_shelfmark]
                if search_variants:
                    logger.info(f"Search variants for '{request.shelf_mark}': {search_variants}")
            
            # Build the query: exact / suffix matches carry high boosts so the
            # requested shelf mark ranks first; fuzzy neighbours only in partial mode.
            base_query = self._build_shelfmark_query(
                normalized_shelfmark, search_variants, request.exact_match, prefix=prefix
            )

            # Apply filters if provided
            filter_clauses = self._build_filters(getattr(request, 'filters', None), index_name or self.index_name)
            if filter_clauses:
                query = {
                    "bool": {
                        "must": [base_query],
                        "filter": filter_clauses
                    }
                }
            else:
                query = base_query

            # Use provided index or default to configured index
            search_index = index_name or self.index_name
            
            if page is None:
                # One page: re-rank the top 30 hits, then cut to the page size.
                window: Dict[str, Any] = {"size": max(page_size, 30)}
            else:
                # Paged like semantic / hybrid search. Constant-score clauses
                # tie a lot, so doc_id keeps the order stable between pages.
                window = {
                    "from_": (page - 1) * page_size,
                    "size": page_size,
                    "sort": ["_score", {"doc_id": "asc"}],
                    "track_scores": True,
                }

            # Execute search
            response = self.es.search(
                index=search_index,
                query=query,
                _source=True,  # Ensure we get the full source including embeddings
                **window,
            )

            # Extract embeddings if requested
            embedding_data = None
            include_embeddings = getattr(request, 'include_embeddings', False)
            if include_embeddings and response['hits']['hits']:
                result_embeddings = self._get_document_embeddings(response['hits']['hits'])
                # Note: No query embedding for shelf mark search
                if result_embeddings:
                    embedding_data = EmbeddingData(
                        query_embedding=None,
                        result_embeddings=result_embeddings,
                        dimension=len(result_embeddings[0]) if result_embeddings else 768
                    )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                # Debug logging for image URLs
                logger.info(f"[SHELFMARK] Document {source.get('doc_id', hit['_id'])}:")
                logger.info(f"  - Has actual_image_url: {bool(source.get('actual_image_url'))}")
                logger.info(f"  - Has image_urls: {bool(source.get('image_urls'))}")
                logger.info(f"  - image_urls type: {type(source.get('image_urls'))}")
                logger.info(f"  - image_urls length: {len(source.get('image_urls', [])) if isinstance(source.get('image_urls'), list) else 'N/A'}")
                if source.get('image_urls'):
                    logger.info(f"  - image_urls value: {source.get('image_urls')}")

                doc_id = source.get("doc_id") or hit["_id"]

                # Calculate relevance score based on field match (use normalized shelfmark and variants)
                relevance_score = self._calculate_shelfmark_relevance(
                    source, normalized_shelfmark, request.exact_match, search_variants
                )

                # Include embedding in result if requested
                embedding = None
                if include_embeddings:
                    embedding = source.get("embedding_vector", [])

                results.append((relevance_score, hit.get("_score") or 0.0, SearchResult(
                    doc_id=doc_id,
                    similarity_score=relevance_score,
                    distance=1.0 - relevance_score,  # Convert to distance
                    metadata=metadata,
                    embedding=embedding
                )))

            # One page: best shelf-mark match first, Elasticsearch score as
            # tie-breaker, then cut to the requested page size. Paged (routed
            # beta) requests keep Elasticsearch's order, which is the order
            # across pages, so "Load more" appends in a consistent order.
            if page is None:
                results.sort(key=lambda item: (item[0], item[1]), reverse=True)
            results = [item[2] for item in results[:page_size]]

            processing_time = (time.time() - start_time) * 1000

            # Total hits for pagination
            total_hits_value = 0
            try:
                total_info = response.get('hits', {}).get('total')
                if isinstance(total_info, dict):
                    total_hits_value = int(total_info.get('value', 0))
                elif isinstance(total_info, int):
                    total_hits_value = int(total_info)
            except Exception:
                total_hits_value = 0

            # Combine filters with metadata for response
            filters_applied = {"shelf_mark": request.shelf_mark, "exact_match": request.exact_match}
            req_filters = getattr(request, 'filters', None)
            if req_filters:
                filters_applied.update(req_filters)

            total_pages, has_more = 1, False
            if page is not None:
                total_pages = max(1, int(np.ceil(total_hits_value / page_size)))
                has_more = page * page_size < total_hits_value

            return SearchResponse(
                results=results,
                query=f"Shelf mark: {request.shelf_mark}",
                count=len(results),
                filters_applied=filters_applied,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data,
                total=total_hits_value,
                page=page or 1,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index
            )

        except Exception as e:
            logger.error(f"Shelf mark search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Shelf mark search failed: {str(e)}"
            )

    @staticmethod
    def shelfmark_key(shelfmark: str) -> str:
        """Collapse a shelf mark to the underscore form used in document ids.

        Punctuation and whitespace runs become a single underscore, so
        ``"T-S B11.93"``, ``"TS B11.93"`` and ``"T-S B11.93 "`` all give
        ``"T_S_B11_93"`` while ``"T-S B1.193"`` stays distinct.

        :param shelfmark: Shelf mark in any display form.
        :return: Underscore-joined alphanumeric key (empty for empty input).
        :rtype: str
        """
        return re.sub(r'[^0-9A-Za-z]+', '_', shelfmark or '').strip('_')

    @staticmethod
    def shelfmark_regex(shelfmark: str) -> str:
        """Lucene regular expression for a shelf mark, tolerant of spacing.

        Spaces become optional and a period may be followed by an optional
        space, so ``"L-G Ar. II.70"`` also matches ``"L-G Ar.II.70"``.
        Everything else is matched literally.

        :param shelfmark: Shelf mark in display form.
        :return: Regex fragment (no anchors) for a ``regexp`` query.
        :rtype: str
        """
        specials = set('\\.?+*|{}[]()"#@&<>~^$')
        out = []
        for ch in shelfmark or '':
            if ch == ' ':
                out.append(' ?')
            elif ch == '.':
                out.append(r'\. ?')
            elif ch in specials:
                out.append('\\' + ch)
            else:
                out.append(ch)
        return ''.join(out)

    @staticmethod
    def shelfmark_child_boundary(form: str) -> str:
        """Lucene class of the character that follows a prefix in its child marks.

        After a letter a child goes on with a non-letter (``"T-S 13J"`` →
        ``"T-S 13J20.20"``, ``"Moss. VII"`` → ``"Moss. VII,5"``), so volume
        VII never reaches VIII. After a digit it goes on with punctuation or a
        space (``"T-S NS 32"`` → ``"T-S NS 32.7"``), never ``"T-S NS 320"``.

        :param form: Shelf mark or prefix in display form.
        :return: A negated character class for a ``regexp`` query.
        :rtype: str
        """
        return "[^A-Za-z]" if form[-1:].isalpha() else "[^0-9A-Za-z]"

    def _build_shelfmark_query(
        self, normalized: str, variants: Optional[List[str]], exact_match: bool, prefix: bool = False
    ) -> Dict[str, Any]:
        """Elasticsearch query for a shelf-mark search.

        ``shelf_mark`` and ``doc_id`` are keyword fields, and stored shelf marks
        usually carry an institution prefix (``"Cambridge CUL: T-S B11.93"``),
        so the strong clauses are whole-value equality, a prefix-tolerant suffix
        regex on ``shelf_mark`` and a suffix wildcard on ``doc_id``.  Partial
        mode adds a contains regex and a low-boost fuzzy match for neighbours;
        the boosts keep the requested shelf mark on top whenever it is indexed.
        ``prefix`` (routed beta queries) adds anchored child clauses
        (:meth:`shelfmark_child_boundary`) so every mark under a prefix or
        volume outranks the contains and fuzzy matches, and lifts the
        whole-mark regexp so a stored prefix record stays above its children.

        :param normalized: Output of :meth:`normalize_shelfmark` for the query.
        :param variants: Extra spellings from :meth:`get_search_variants`.
        :param exact_match: Only strong clauses when ``True``.
        :param prefix: Also match the child marks of a prefix or volume.
        :return: ``bool`` query.
        :rtype: Dict[str, Any]
        """
        forms = [normalized] + [v for v in (variants or []) if v and v != normalized]
        whole_boost = SHELFMARK_WHOLE_BOOST_PREFIX if prefix else 15
        should: List[Dict[str, Any]] = []
        for form in forms:
            rx = self.shelfmark_regex(form)
            key = self.shelfmark_key(form)
            should.append({"term": {"shelf_mark": {"value": form, "boost": 20}}})
            should.append({"regexp": {"shelf_mark": {"value": f"(.*[ :])?{rx}", "case_insensitive": True,
                                                     "boost": whole_boost}}})
            if key:
                should.append({"term": {"doc_id": {"value": key, "boost": 12}}})
                should.append({"wildcard": {"doc_id": {"value": f"*_{key}", "case_insensitive": True, "boost": 12}}})
            if prefix and not exact_match:
                boundary = self.shelfmark_child_boundary(form)
                should.append({"regexp": {"shelf_mark": {
                    "value": f"(.*[ :])?{rx}{boundary}.*", "case_insensitive": True, "boost": SHELFMARK_CHILD_BOOST}}})
                if key:
                    should.append({"regexp": {"doc_id": {
                        "value": f"(.*_)?{key}{boundary}.*", "case_insensitive": True, "boost": SHELFMARK_CHILD_BOOST}}})
            if not exact_match:
                should.append({"regexp": {"shelf_mark": {"value": f".*{rx}.*", "case_insensitive": True, "boost": 4}}})
                should.append({"match": {"shelf_mark": {"query": form, "fuzziness": "AUTO", "boost": 1}}})
        return {"bool": {"should": should, "minimum_should_match": 1}}

    def _calculate_shelfmark_relevance(self, source: Dict[str, Any], query: str, exact_match: bool, search_variants: Optional[List[str]] = None) -> float:
        """Score how well a hit's shelf mark matches the query.

        Comparison is on :meth:`shelfmark_key` of the institution-stripped
        values, so spacing and punctuation differences do not matter.

        :param source: Elasticsearch ``_source`` of the hit.
        :param query: Normalized query shelf mark.
        :param exact_match: Unused; kept for call compatibility.
        :param search_variants: Extra query spellings.
        :return: 1.0 exact, 0.95 suffix (prefix-only difference), 0.6 contains,
            0.2 fuzzy neighbour.
        :rtype: float
        """
        forms = [query] + [v for v in (search_variants or []) if v and v != query]
        keys = [self.shelfmark_key(f).lower() for f in forms if self.shelfmark_key(f)]
        score = 0.2
        for field in ('shelf_mark', 'shelfmark', 'classmark', 'doc_id'):
            value = source.get(field)
            if not value or not isinstance(value, str):
                continue
            stripped = self.shelfmark_key(self.normalize_shelfmark(value)).lower()
            full = self.shelfmark_key(value).lower()
            for key in keys:
                if key in (stripped, full):
                    return 1.0
                if full.endswith('_' + key):
                    score = max(score, 0.95)
                elif key in full:
                    score = max(score, 0.6)
        return score

    @staticmethod
    def build_keyword_text_query(query: str) -> Dict[str, Any]:
        """Text clause of keyword search, also used as the keyword part of hybrid search.

        A document matches when any of these matches, and every match adds score:

        * the legacy fuzzy ``multi_match``, or an exact match on v8
          ``full_text_content``. These two sit in a ``dis_max``, so only the
          better of them counts, the same max-of-fields scoring ``best_fields``
          gives. That keeps the ranking of ordinary English queries as it was.
        * a nested match on a scholarly transcription (``transcriptions.text``).

        ``ignore_unmapped`` keeps the nested clause harmless on indices without
        that nested mapping (older merged indices, bibliography). AI reads are
        never part of this clause (docs/ai_transcription_search.md).

        :param query: Visitor's query.
        :return: Elasticsearch ``bool`` clause with ``minimum_should_match`` 1.
        :rtype: Dict[str, Any]
        """
        return {
            "bool": {
                "should": [
                    {
                        "dis_max": {
                            "queries": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": list(KEYWORD_SEARCH_FIELDS),
                                        "type": "best_fields",
                                        "fuzziness": "AUTO",
                                        "boost": 1.0,
                                    }
                                },
                                {
                                    "match": {
                                        FULL_TEXT_CONTENT_FIELD: {"query": query, "boost": FULL_TEXT_CONTENT_BOOST}
                                    }
                                },
                            ],
                            "tie_breaker": 0.0,
                        }
                    },
                    {
                        "nested": {
                            "path": TRANSCRIPTIONS_PATH,
                            "query": {"match": {f"{TRANSCRIPTIONS_PATH}.text": {"query": query}}},
                            "score_mode": "max",
                            "boost": TRANSCRIPTIONS_BOOST,
                            "ignore_unmapped": True,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _machine_reading_applies(self, query: str, search_index: str) -> bool:
        """Whether keyword search should run the machine-reading second pass.

        :param query: Visitor's query.
        :param search_index: Index the catalogue pass searched.
        :return: ``True`` when a consensus service is wired, the
            ``AI_TRANSCRIPTIONS_ENABLED`` master switch and the
            ``CONSENSUS_SEARCH_ENABLED`` kill switch are on, the query holds a
            Hebrew letter and the search targets the catalogue index the
            consensus service joins against.
        :rtype: bool
        """
        consensus = getattr(self, "consensus_service", None)
        return (
            consensus is not None
            and consensus_search_enabled()
            and bool(_HEBREW_LETTER_RE.search(query or ""))
            and search_index == consensus.catalogue_index
        )

    def _machine_reading_cache_store(self) -> Dict[Tuple[str, str, str], Tuple[float, List[Dict[str, Any]]]]:
        """Per-instance cache of :meth:`_machine_reading_hits`, created lazily.

        :return: ``{(index, query, filters_json): (monotonic stored_at, hits)}``.
        :rtype: Dict[Tuple[str, str, str], Tuple[float, List[Dict[str, Any]]]]
        """
        cache = getattr(self, "_machine_reading_cache", None)
        if cache is None:
            cache = self._machine_reading_cache = {}
        return cache

    def _catalogue_check(
        self,
        doc_ids: List[str],
        text_query: Dict[str, Any],
        filter_clauses: List[Dict[str, Any]],
        search_index: str,
    ) -> Set[str]:
        """Which candidate fragments may be appended as machine-reading hits.

        One search over the candidates: the active filters must hold, and the
        catalogue text clause (named) must **not** match. A fragment the
        catalogue query matches is already among the catalogue hits, so
        appending it again would duplicate it.

        :param doc_ids: Catalogue ``doc_id`` values (``_id`` in the catalogue index).
        :param text_query: :meth:`build_keyword_text_query` clause of this search.
        :param filter_clauses: :meth:`_build_filters` clauses of this search.
        :param search_index: Catalogue index.
        :return: The ``doc_id`` values that pass the filters and are not catalogue matches.
        :rtype: Set[str]
        """
        if not doc_ids:
            return set()
        named_text = {"bool": {**text_query["bool"], "_name": _CATALOGUE_MATCH_NAME}}
        response = self.es.search(
            index=search_index,
            query={"bool": {"filter": [{"ids": {"values": doc_ids}}, *filter_clauses], "should": [named_text]}},
            size=len(doc_ids),
            _source=False,
        )
        return {
            hit["_id"]
            for hit in response["hits"]["hits"]
            if _CATALOGUE_MATCH_NAME not in (hit.get("matched_queries") or ())
        }

    def _machine_reading_hits(
        self,
        query: str,
        text_query: Dict[str, Any],
        filter_clauses: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        search_index: str,
    ) -> List[Dict[str, Any]]:
        """Every machine-reading hit of a query that may be appended, in AI rank order.

        Takes :meth:`ConsensusService.search_uncatalogued` (agreed lines of
        uncatalogued fragments, one hit per image), keeps the best hit per
        catalogue ``doc_id`` (two images of one fragment can match), then keeps
        only fragments :meth:`_catalogue_check` allows. Deduplicating the whole
        list, rather than each page, makes the total exact and the pages
        consistent. That costs one AI search and one catalogue search per query.
        It is affordable because the uncatalogued set is small (about 800
        records), and the result is cached per query for
        :data:`MACHINE_READING_CACHE_TTL_SECONDS`.

        :param query: Visitor's query.
        :param text_query: Catalogue text clause of this search.
        :param filter_clauses: Catalogue filter clauses of this search.
        :param filters: Raw request filters (part of the cache key).
        :param search_index: Catalogue index.
        :return: Hit dicts as returned by ``search_uncatalogued``.
        :rtype: List[Dict[str, Any]]
        """
        key = (search_index, query, json.dumps(filters or {}, sort_keys=True, default=str))
        now = time.monotonic()
        with _MACHINE_READING_CACHE_LOCK:
            cache = self._machine_reading_cache_store()
            cached = cache.get(key)
        if cached is not None and now - cached[0] < MACHINE_READING_CACHE_TTL_SECONDS:
            return cached[1]

        total, raw_hits = self.consensus_service.search_uncatalogued(query, MACHINE_READING_MAX_HITS, 0)
        if total > MACHINE_READING_MAX_HITS:
            logger.info("Machine-reading pass for %r capped at %d of %d reads", query, MACHINE_READING_MAX_HITS, total)
        best_per_doc: Dict[str, Dict[str, Any]] = {}
        for hit in raw_hits:
            best_per_doc.setdefault(hit["doc_id"], hit)
        allowed = self._catalogue_check(list(best_per_doc), text_query, filter_clauses, search_index)
        hits = [hit for doc_id, hit in best_per_doc.items() if doc_id in allowed]

        with _MACHINE_READING_CACHE_LOCK:
            cache.pop(key, None)
            while len(cache) >= MACHINE_READING_CACHE_MAX_ENTRIES:
                cache.pop(next(iter(cache)))
            cache[key] = (now, hits)
        return hits

    def _machine_reading_results(self, hits: List[Dict[str, Any]], search_index: str) -> List[SearchResult]:
        """Turn machine-reading hits into ordinary results built from their catalogue records.

        Metadata goes through :meth:`_extract_metadata`, the same public path
        as catalogue hits (filtered joins and bibliography sources, no provider
        URLs). The machine-reading fields come from the AI hit only.

        :param hits: Hits of this page, in display order.
        :param search_index: Catalogue index (``_id`` == ``doc_id``).
        :return: Results with ``match_source="machine_reading"``; a hit whose
            catalogue record is missing is skipped.
        :rtype: List[SearchResult]
        """
        if not hits:
            return []
        response = self.es.mget(
            index=search_index, ids=[hit["doc_id"] for hit in hits], source_excludes=["embedding_vector"]
        )
        sources = {doc["_id"]: doc.get("_source") or {} for doc in response.get("docs", []) if doc.get("found")}
        results: List[SearchResult] = []
        for hit in hits:
            source = sources.get(hit["doc_id"])
            if source is None:
                continue
            results.append(SearchResult(
                doc_id=source.get("doc_id") or hit["doc_id"],
                similarity_score=0.0,  # AI scores are not comparable with catalogue scores
                distance=None,
                metadata=self._extract_metadata(source),
                embedding=None,
                match_source=MACHINE_READING,
                machine_reading=MachineReadingMatch(
                    image_index=hit["image_index"],
                    line_index=hit["line_index"],
                    snippet=hit["snippet"],
                    read_url=hit["read_url"],
                ),
            ))
        return results

    def _machine_reading_page(
        self,
        query: str,
        text_query: Dict[str, Any],
        filter_clauses: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        search_index: str,
        page_start: int,
        page_size: int,
        catalogue_total: int,
    ) -> Tuple[int, List[SearchResult]]:
        """Second pass of keyword search: machine-reading hits after every catalogue hit.

        Machine-reading hit ``i`` sits at global position ``catalogue_total + i``,
        so a page gets them only where it runs past the last catalogue hit
        (:func:`machine_reading_window`). Their count is added to the total on
        every page so pagination stays consistent.

        :param query: Visitor's query.
        :param text_query: Catalogue text clause of this search.
        :param filter_clauses: Catalogue filter clauses of this search.
        :param filters: Raw request filters.
        :param search_index: Index the catalogue pass searched.
        :param page_start: Global offset of the page.
        :param page_size: Results per page.
        :param catalogue_total: Exact number of catalogue hits.
        :return: ``(machine-reading total, results for this page)``;
            ``(0, [])`` when the pass does not apply.
        :rtype: Tuple[int, List[SearchResult]]
        """
        if not self._machine_reading_applies(query, search_index):
            return 0, []
        hits = self._machine_reading_hits(query, text_query, filter_clauses, filters, search_index)
        window = machine_reading_window(page_start, page_size, catalogue_total)
        if window is None:
            return len(hits), []
        ai_offset, ai_size = window
        return len(hits), self._machine_reading_results(hits[ai_offset:ai_offset + ai_size], search_index)

    async def search_by_keyword(self, request, index_name: Optional[str] = None) -> SearchResponse:
        """Search documents by keywords in text fields.

        Catalogue hits come first, in Elasticsearch order. For a Hebrew query on
        the catalogue index, a second pass appends machine-reading hits
        strictly after the last catalogue hit, across pages
        (:meth:`_machine_reading_page`). If the second pass fails, the
        catalogue results are returned on their own.

        :param request: ``KeywordSearchRequest`` (query, filters, num_results, page).
        :param index_name: Index to search; defaults to the configured index.
        :return: One page of results with pagination metadata.
        :rtype: SearchResponse
        """
        start_time = time.time()

        try:
            text_query = self.build_keyword_text_query(request.query)

            # Apply filters if provided
            filter_clauses = self._build_filters(getattr(request, 'filters', None), index_name or self.index_name)
            if filter_clauses:
                query = {
                    "bool": {
                        "must": [text_query],
                        "filter": filter_clauses
                    }
                }
            else:
                query = text_query

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Use provided index or default to configured index
            search_index = index_name or self.index_name
            
            # Execute search (the ~22KB embedding vector is never used here).
            # Pages that start past the catalogue hits only need the total; ES
            # rejects from + size > max_result_window (a size-0 search still
            # returns hits.total). from_offset/page_size stay unclamped for the
            # machine-reading window and the pagination arithmetic below.
            catalogue_from = min(from_offset, MAX_RESULT_WINDOW)
            catalogue_size = max(0, min(page_size, MAX_RESULT_WINDOW - from_offset))
            response = self.es.search(
                index=search_index,
                query=query,
                size=catalogue_size,
                from_=catalogue_from,
                source_excludes=["embedding_vector"]
            )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                # Debug logging for image URLs
                logger.info(f"[KEYWORD] Document {source.get('doc_id', hit['_id'])}:")
                logger.info(f"  - Has actual_image_url: {bool(source.get('actual_image_url'))}")
                logger.info(f"  - Has image_urls: {bool(source.get('image_urls'))}")
                logger.info(f"  - image_urls type: {type(source.get('image_urls'))}")
                logger.info(f"  - image_urls length: {len(source.get('image_urls', [])) if isinstance(source.get('image_urls'), list) else 'N/A'}")
                if source.get('image_urls'):
                    logger.info(f"  - image_urls value: {source.get('image_urls')}")

                doc_id = source.get("doc_id") or hit["_id"]

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=round(hit["_score"], 4),
                    distance=round(max(0, 10.0 - hit["_score"]), 4),  # Convert to distance-like metric
                    metadata=metadata,
                    embedding=None  # No embeddings for keyword search
                ))

            # Total hits for pagination
            total_hits_value = 0
            total_is_exact = True
            try:
                total_info = response.get('hits', {}).get('total')
                if isinstance(total_info, dict):
                    total_hits_value = int(total_info.get('value', 0))
                    total_is_exact = total_info.get('relation', 'eq') == 'eq'
                elif isinstance(total_info, int):
                    total_hits_value = int(total_info)
            except Exception:
                total_hits_value = 0

            # Second pass: machine-reading hits go strictly after the last
            # catalogue hit. With a lower-bound total (10,000+ catalogue hits)
            # that position is past max_result_window, so nothing is appended.
            # It runs in a worker thread: its blocking ES calls (and a first
            # build of the uncatalogued set) must not stall the event loop.
            if total_is_exact:
                try:
                    machine_total, machine_results = await asyncio.to_thread(
                        self._machine_reading_page,
                        request.query, text_query, filter_clauses, getattr(request, 'filters', None),
                        search_index, from_offset, page_size, total_hits_value,
                    )
                except Exception as exc:  # I/O boundary: never break catalogue search
                    logger.warning(f"Machine-reading second pass skipped for '{request.query}': {exc}")
                    machine_total, machine_results = 0, []
                results.extend(machine_results)
                total_hits_value += machine_total

            processing_time = (time.time() - start_time) * 1000

            total_pages = max(1, int(np.ceil(total_hits_value / page_size))) if page_size else 1
            has_more = (page_number * page_size) < total_hits_value

            # Combine filters with metadata for response
            filters_applied = {"search_type": "keyword"}
            req_filters = getattr(request, 'filters', None)
            if req_filters:
                filters_applied.update(req_filters)

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied=filters_applied,
                processing_time_ms=round(processing_time, 2),
                embedding_data=None,  # No embeddings for keyword search
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index
            )

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Keyword search failed: {str(e)}"
            )

    async def get_visualization_explorer_data(self, request, index_name: Optional[str] = None) -> SearchResponse:
        """
        Load documents for visualization explorer
        
        Loads a random sample of documents from the collection for full-page
        visualization exploration. Supports loading a configurable number of
        documents or the entire index.
        
        Args:
            request: VisualizationExplorerRequest with configuration
            index_name: Optional name of the Elasticsearch index to use
        """
        start_time = time.time()
        
        # Use provided index_name or fall back to default
        target_index = index_name if index_name else self.index_name
        
        try:
            # Determine how many documents to load
            if request.load_full_index:
                # Get total document count
                stats = self.es.indices.stats(index=target_index)
                total_docs = stats['indices'][target_index]['total']['docs']['count']
                num_docs_to_load = total_docs
            else:
                num_docs_to_load = min(request.num_documents, 10000)  # Cap at 10k for performance
            
            logger.info(f"Loading {num_docs_to_load} documents from index {target_index} for visualization explorer")
            
            # Use scroll API for large result sets
            if num_docs_to_load > 1000:
                # For large sets, use scroll API
                query = {
                    "match_all": {}
                }
                
                # Initial search
                response = self.es.search(
                    index=target_index,
                    query=query,
                    size=min(1000, num_docs_to_load),  # ES scroll size limit
                    scroll='5m',
                    _source=True
                )
                
                all_hits = response['hits']['hits']
                scroll_id = response.get('_scroll_id')
                
                # Continue scrolling if we need more documents
                while len(all_hits) < num_docs_to_load and scroll_id:
                    scroll_response = self.es.scroll(
                        scroll_id=scroll_id,
                        scroll='5m'
                    )
                    
                    hits = scroll_response['hits']['hits']
                    if not hits:
                        break
                    
                    all_hits.extend(hits)
                    
                    # Update scroll_id for next iteration
                    scroll_id = scroll_response.get('_scroll_id')
                    
                    # Safety check to prevent infinite loops
                    if len(all_hits) >= num_docs_to_load:
                        break
                
                # Clear scroll context
                if scroll_id:
                    self.es.clear_scroll(scroll_id=scroll_id)
                
                # Limit to requested number
                all_hits = all_hits[:num_docs_to_load]
                
            else:
                # For smaller sets, use regular search with random sampling
                query = {
                    "function_score": {
                        "query": {"match_all": {}},
                        "random_score": {}
                    }
                }
                
                response = self.es.search(
                    index=target_index,
                    query=query,
                    size=num_docs_to_load,
                    _source=True
                )
                
                all_hits = response['hits']['hits']
            
            # Embeddings stay server-side by default: register them under a
            # sample_id and (optionally) compute 2D coordinates right here, so
            # the browser never has to download or re-upload raw vectors.
            embedding_data = None
            sample_id = None
            compute_method = (getattr(request, 'compute_method', None) or '').lower() or None
            if all_hits:
                result_embeddings = self._get_document_embeddings(all_hits)
                if result_embeddings:
                    from src.backend.visualization_service import visualization_service
                    sample_id = visualization_service.register_sample(result_embeddings)
                    dimension = len(result_embeddings[0])
                    coords_by_method: Dict[str, List[List[float]]] = {}
                    if compute_method in ('tsne', 'umap'):
                        viz = visualization_service.calculate_visualization(
                            embeddings=result_embeddings,
                            method=compute_method,
                        )
                        coords_by_method[compute_method] = viz['coordinates']
                    if request.include_embeddings or coords_by_method:
                        embedding_data = EmbeddingData(
                            query_embedding=None,  # No query for explorer mode
                            result_embeddings=result_embeddings if request.include_embeddings else None,
                            dimension=dimension,
                            tsne=coords_by_method.get('tsne'),
                            umap=coords_by_method.get('umap'),
                        )
            
            # Format results with rich metadata
            results = []
            for hit in all_hits:
                doc_id = hit['_id']
                source = hit['_source']
                
                # Extract appropriate metadata based on index type
                if 'bibliography' in target_index:
                    metadata = self._extract_secondary_metadata(source, target_index)
                else:
                    metadata = self._extract_metadata(source, target_index)
                
                # Create search result
                search_result = SearchResult(
                    doc_id=doc_id,
                    similarity_score=1.0,  # All documents have equal weight in explorer
                    metadata=metadata,
                    embedding=source.get('embedding_vector') if request.include_embeddings else None
                )
                results.append(search_result)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return SearchResponse(
                results=results,
                count=len(results),
                processing_time_ms=processing_time_ms,
                embedding_data=embedding_data,
                sample_id=sample_id,
                page=1,
                page_size=len(results),
                total_pages=1,
                has_more=False,
                index_name=target_index
            )
            
        except Exception as e:
            logger.error(f"Visualization explorer data loading failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load visualization explorer data: {str(e)}"
            )

    async def search_hybrid(self, request, index_name: Optional[str] = None) -> SearchResponse:
        """Perform hybrid search combining semantic and keyword search with configurable weights.

        The vector leg uses the embedder that built the target index
        (``semantic_router``). On an eligibility-mapped index the cosine
        function only scores ``semantic_eligible`` records; vector-less
        records can still be found through the keyword leg
        (:func:`build_hybrid_query`). On the beta index a bare shelf mark,
        shelf-mark prefix or volume is answered by shelf-mark search (when it
        finds any), as in :meth:`search`.

        :param request: Hybrid request (query, weights, filters, paging).
        :param index_name: Target index; defaults to the configured index.
        :return: Ranked results with pagination metadata.
        :rtype: SearchResponse
        :raises SemanticRouteError: When no verified embedder matches the index.
        :raises HTTPException: 500 when the search itself fails.
        """
        start_time = time.time()
        # Use provided index or default to configured index
        search_index = index_name or self.index_name
        route = await semantic_router.resolve(self.es, search_index, self.index_name, embedding_client)
        if route.beta and looks_like_shelfmark(request.query):
            routed = await self._beta_shelfmark_search(request, search_index)
            if routed.total:
                return routed
            # No record has that shelf mark: run the query as typed.

        try:
            # Normalize weights to 0-1 range
            semantic_weight = request.semanticWeight / 100.0
            keyword_weight = request.keywordWeight / 100.0
            
            # Generate query embedding for semantic search
            query_embedding = await semantic_router.embed_query(route, request.query)

            # Build filter clauses
            filter_clauses = self._build_filters(request.filters, search_index)

            # Keyword part: same text clause as keyword search
            keyword_text_query = self.build_keyword_text_query(request.query)

            # Combine both with weights using function_score
            hybrid_query = build_hybrid_query(
                query_embedding.flatten().tolist(),
                filter_clauses,
                keyword_text_query,
                semantic_weight,
                keyword_weight,
                eligible_only=route.eligible_only,
            )

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Execute search
            response = self.es.search(
                index=search_index,
                query=hybrid_query,
                size=page_size,
                from_=from_offset,
                _source=True
            )

            # Extract embeddings if requested. Vector-less records (keyword-only
            # hits on an eligibility-mapped index) get zero vectors of the
            # query's dimension so per-result vectors stay aligned for t-SNE.
            embedding_data = None
            query_vector = query_embedding.flatten().tolist()
            result_embeddings: List[List[float]] = []
            if request.include_embeddings and response['hits']['hits']:
                result_embeddings = self._get_document_embeddings(
                    response['hits']['hits'], fallback_dimension=len(query_vector)
                )
                embedding_data = EmbeddingData(
                    query_embedding=query_vector,
                    result_embeddings=result_embeddings,
                    dimension=len(query_vector)
                )

            # Format results with rich metadata
            results = []
            for position, hit in enumerate(response['hits']['hits']):
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                # Debug logging for image URLs
                logger.info(f"[HYBRID] Document {source.get('doc_id', hit['_id'])}:")
                logger.info(f"  - Has actual_image_url: {bool(source.get('actual_image_url'))}")
                logger.info(f"  - Has image_urls: {bool(source.get('image_urls'))}")
                logger.info(f"  - image_urls type: {type(source.get('image_urls'))}")
                logger.info(f"  - image_urls length: {len(source.get('image_urls', [])) if isinstance(source.get('image_urls'), list) else 'N/A'}")
                if source.get('image_urls'):
                    logger.info(f"  - image_urls value: {source.get('image_urls')}")

                # Include embedding in result if requested
                embedding = result_embeddings[position] if request.include_embeddings else None

                doc_id = source.get("doc_id") or hit["_id"]

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=round(hit["_score"], 4),
                    distance=round(max(0, 10.0 - hit["_score"]), 4),  # Convert to distance-like metric
                    metadata=metadata,
                    embedding=embedding
                ))
            if route.beta:
                self._mark_catalogue_pages(results)

            processing_time = (time.time() - start_time) * 1000

            # Total hits for pagination
            total_hits_value = 0
            try:
                total_info = response.get('hits', {}).get('total')
                if isinstance(total_info, dict):
                    total_hits_value = int(total_info.get('value', 0))
                elif isinstance(total_info, int):
                    total_hits_value = int(total_info)
            except Exception:
                total_hits_value = 0

            total_pages = max(1, int(np.ceil(total_hits_value / page_size))) if page_size else 1
            has_more = (page_number * page_size) < total_hits_value

            # Combine filters for response
            filters_applied = {
                "search_type": "hybrid",
                "semantic_weight": request.semanticWeight,
                "keyword_weight": request.keywordWeight
            }
            req_filters = getattr(request, 'filters', None)
            if req_filters:
                filters_applied.update(req_filters)

            return SearchResponse(
                results=results,
                query=f"Hybrid search: {request.query}",
                count=len(results),
                filters_applied=filters_applied,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data,
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more,
                index_name=search_index,
                semantic_beta=True if route.beta else None
            )

        except SemanticRouteError:
            raise
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            logger.error(f"Full hybrid search error type: {type(e).__name__}")
            logger.error(f"Full hybrid search error message: {str(e)}")
            
            if hasattr(e, 'info'):
                logger.error(f"Hybrid search error info: {json.dumps(e.info, indent=2)}")
            if hasattr(e, 'body'):
                logger.error(f"Hybrid search error body: {e.body}")
            if hasattr(e, 'status_code'):
                logger.error(f"Hybrid search status code: {e.status_code}")
                
            raise HTTPException(
                status_code=500,
                detail=f"Hybrid search failed: {str(e)}"
            )

    def get_available_indices(self) -> List[Dict[str, Any]]:
        """Get list of available Elasticsearch indices"""
        try:
            # Get all indices
            indices = self.es.cat.indices(format='json')
            
            # Filter for relevant indices (those that look like document collections)
            relevant_indices = []
            for index_info in indices:
                index_name = index_info.get('index', '')
                # Skip system indices and hidden indices, but be more permissive
                if (not index_name.startswith('.') and 
                    not index_name.startswith('_') and
                    len(index_name) > 0):
                    
                    doc_count = int(index_info.get('docs.count', 0))
                    # Include indices with documents, or the default index even if empty
                    if doc_count > 0 or index_name == self.index_name:
                        relevant_indices.append({
                            "name": index_name,
                            "document_count": doc_count,
                            "size": index_info.get('store.size', 'unknown'),
                            "is_default": index_name == self.index_name,
                            "description": self._get_index_description(index_name)
                        })
            
            # Sort by document count (descending) and put default first
            relevant_indices.sort(key=lambda x: (not x['is_default'], -x['document_count']))
            
            return relevant_indices
            
        except Exception as e:
            logger.error(f"Failed to get available indices: {e}")
            # Return default index if we can't get the list
            return [{
                "name": self.index_name,
                "document_count": 0,
                "size": "unknown",
                "is_default": True,
                "description": "Default production index"
            }]
    
    def _get_index_description(self, index_name: str) -> str:
        """Generate a description for an index based on its name"""
        name_lower = index_name.lower()
        
        if 'prod' in name_lower or 'production' in name_lower:
            return "Production index with stable data"
        elif 'test' in name_lower or 'experimental' in name_lower:
            return "Test/experimental index for development"
        elif 'v1' in name_lower or 'v2' in name_lower:
            return f"Versioned index ({index_name})"
        elif 'text_only' in name_lower:
            return "Text-only index (no images)"
        elif 'full' in name_lower:
            return "Full index with all metadata"
        else:
            return f"Custom index: {index_name}"

    def get_stats(self):
        """Get index statistics"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            return {
                "status": "healthy",
                "document_count": doc_count,
                "backend": "elasticsearch",
                "index_name": self.index_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "backend": "elasticsearch"
            }

    def _group_shelfmarks_by_range(self, shelfmarks: List[Dict[str, Any]], range_size: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group shelfmarks by numeric ranges based on their numbering.
        For example: A-100 to A-199 -> "A-100", A-200 to A-299 -> "A-200", etc.
        
        Args:
            shelfmarks: List of shelfmark dicts with 'name' and 'count' keys
            range_size: Size of each range (default 100)
        
        Returns:
            Dict mapping range names to lists of shelfmarks in that range
        """
        ranges = {}
        
        for shelfmark in shelfmarks:
            name = shelfmark.get("name", "")
            if not name:
                continue
            
            # Try to extract numeric part from shelfmark
            # Patterns like: A-100, A-101, B-2000, MS-TS-NS-144.1, etc.
            # Extract the last numeric sequence
            match = re.search(r'(\d+)(?:\.\d+)?$', name)
            if not match:
                # No numeric part found, put in "Other" range
                range_key = "Other"
                if range_key not in ranges:
                    ranges[range_key] = []
                ranges[range_key].append(shelfmark)
                continue
            
            number = int(match.group(1))
            # Get the prefix (everything before the number)
            prefix = name[:match.start()]
            
            # Calculate which range this belongs to (round down to nearest range_size)
            range_start = (number // range_size) * range_size
            range_end = range_start + range_size - 1
            
            # Create range key like "A-100" or "B-2000"
            if prefix:
                range_key = f"{prefix}{range_start}"
            else:
                range_key = f"{range_start}"
            
            # Also store the range display name
            range_display = f"{prefix}{range_start}-{range_end}" if prefix else f"{range_start}-{range_end}"
            
            if range_key not in ranges:
                ranges[range_key] = {
                    "name": range_display,
                    "range_start": range_start,
                    "range_end": range_end,
                    "shelfmarks": []
                }
            
            ranges[range_key]["shelfmarks"].append(shelfmark)
        
        # Convert to list format with counts
        result = {}
        for range_key, range_data in ranges.items():
            if isinstance(range_data, dict) and "shelfmarks" in range_data:
                shelfmark_list = range_data["shelfmarks"]
                total_count = sum(s.get("count", 0) for s in shelfmark_list)
                result[range_key] = {
                    "name": range_data["name"],
                    "range_start": range_data["range_start"],
                    "range_end": range_data["range_end"],
                    "count": total_count,
                    "shelfmarks": shelfmark_list
                }
            else:
                # Handle "Other" case
                total_count = sum(s.get("count", 0) for s in range_data)
                result[range_key] = {
                    "name": "Other",
                    "range_start": None,
                    "range_end": None,
                    "count": total_count,
                    "shelfmarks": range_data
                }
        
        return result

    @lru_cache(maxsize=32)
    def _aggregatable_field(self, index: str, base: str) -> Optional[str]:
        """Name to use in a ``terms`` aggregation for a base field, from the mapping.

        Prefers a ``<base>.keyword`` subfield when one exists, then the field
        itself when it is mapped as ``keyword``. Returns ``None`` when neither
        exists, so callers never aggregate on a field that silently yields no
        buckets.

        :param index: Index to inspect.
        :param base: Base field name such as ``"shelf_mark"``.
        :return: Aggregatable field name or ``None``.
        :rtype: Optional[str]
        """
        cache = getattr(self, "_agg_field_cache", None)
        if cache is None:
            cache = self._agg_field_cache = {}
        key = (index, base)
        if key in cache:
            return cache[key]
        props = self.es.indices.get_mapping(index=index)[index]["mappings"].get("properties", {})
        spec = props.get(base) or {}
        if "keyword" in (spec.get("fields") or {}):
            resolved: Optional[str] = f"{base}.keyword"
        elif spec.get("type") == "keyword":
            resolved = base
        else:
            resolved = None
        cache[key] = resolved
        return resolved

    @staticmethod
    def prune_unbrowsable(hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Drop sub-collections and collections a visitor cannot click into.

        A sub-collection is browsable when it has shelfmarks or sub-sub-collections;
        a collection is browsable when at least one sub-collection is.

        :param hierarchy: Output of :meth:`get_collection_hierarchy` before pruning.
        :return: The same structure without dead entries.
        :rtype: Dict[str, Any]
        """
        pruned: Dict[str, Any] = {}
        for name, collection in hierarchy.items():
            subs = {
                sub_name: sub
                for sub_name, sub in (collection.get("sub_collections") or {}).items()
                if sub.get("shelfmarks") or sub.get("sub_sub_collections")
            }
            if subs:
                pruned[name] = {**collection, "sub_collections": subs}
        return pruned

    def get_collection_hierarchy(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection hierarchy using Elasticsearch aggregations
        Returns structure: collection -> sub_collection -> shelfmarks (or sub_sub_collections if > 1000 shelfmarks)
        
        This is very fast because it uses aggregations which are optimized
        in Elasticsearch.
        
        For sub-collections with > 1000 shelfmarks, they are automatically grouped
        into sub-sub-collections by numeric ranges (e.g., A-100, A-200, etc.)
        """
        try:
            target_index = index_name or self.index_name
            
            # Try multiple field name variations to handle different index mappings.
            # Use collection and sub_collection fields (NOT source_collection - that's old/incorrect metadata)
            field_variations = []
            resolved = tuple(self._aggregatable_field(target_index, base) for base in ("collection", "sub_collection", "shelf_mark"))
            if all(resolved):
                field_variations.append(resolved)
            field_variations += [
                ("collection.keyword", "sub_collection.keyword", "shelf_mark.keyword"),
                ("collection.keyword", "sub_collection.keyword", "shelfmark.keyword"),
                ("collection.keyword", "sub_collection.keyword", "classmark.keyword"),
                ("collection", "sub_collection", "shelf_mark"),
                ("collection", "sub_collection", "shelfmark"),
                ("collection", "sub_collection", "classmark"),
            ]
            
            for coll_field, sub_coll_field, shelf_field in field_variations:
                try:
                    # Use nested aggregations for collection -> sub_collection -> shelfmarks
                    aggs = {
                        "collections": {
                            "terms": {
                                "field": coll_field,
                                "size": 100,
                                "missing": "Unknown"  # Handle missing values
                            },
                            "aggs": {
                                "shelfmarks_direct": {
                                    "terms": {
                                        "field": shelf_field,
                                        "size": 2000,
                                        "min_doc_count": 1
                                    }
                                },
                                "sub_collections": {
                                    "terms": {
                                        "field": sub_coll_field,
                                        "size": 100,
                                        "missing": "Unknown"
                                    },
                                    "aggs": {
                                        "shelfmarks": {
                                            "terms": {
                                                "field": shelf_field,
                                                "size": 2000,
                                                "min_doc_count": 1
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    response = self.es.search(
                        index=target_index,
                        size=0,  # We don't need actual documents, just aggregations
                        query={"match_all": {}},  # Explicitly match all documents
                        aggs=aggs
                    )
                    
                    # Process aggregation results
                    hierarchy = {}
                    collection_buckets = response['aggregations']['collections']['buckets']
                    
                    # Debug logging to understand why shelfmarks are empty
                    if collection_buckets and len(collection_buckets) > 0:
                        first_collection = collection_buckets[0]
                        logger.info(f"First collection: {first_collection.get('key')}, doc_count: {first_collection.get('doc_count')}")
                        if 'sub_collections' in first_collection:
                            sub_cols = first_collection.get('sub_collections', {}).get('buckets', [])
                            if sub_cols:
                                first_sub = sub_cols[0]
                                logger.info(f"First sub-collection: {first_sub.get('key')}, doc_count: {first_sub.get('doc_count')}")
                                logger.info(f"Sub-collection aggregation keys: {list(first_sub.keys())}")
                                shelfmarks_agg = first_sub.get('shelfmarks', {})
                                if shelfmarks_agg:
                                    buckets = shelfmarks_agg.get('buckets', [])
                                    logger.info(f"Shelfmarks aggregation returned {len(buckets)} buckets for sub-collection '{first_sub.get('key')}' using field '{shelf_field}'")
                                    if len(buckets) == 0:
                                        logger.warning(f"No shelfmark buckets found! Field '{shelf_field}' may not exist, be empty, or have wrong mapping.")
                                        # Try to get a sample document to see what fields exist
                                        try:
                                            sample_query = {
                                                "bool": {
                                                    "must": [
                                                        {"term": {coll_field: first_collection.get('key')}},
                                                        {"term": {sub_coll_field: first_sub.get('key')}}
                                                    ]
                                                }
                                            }
                                            sample_resp = self.es.search(
                                                index=target_index,
                                                size=1,
                                                query=sample_query,
                                                _source=True
                                            )
                                            if sample_resp.get('hits', {}).get('hits'):
                                                sample_doc = sample_resp['hits']['hits'][0]['_source']
                                                logger.info(f"Sample document fields: {list(sample_doc.keys())}")
                                                logger.info(f"Sample doc shelf_mark: {sample_doc.get('shelf_mark')}, shelfmark: {sample_doc.get('shelfmark')}, classmark: {sample_doc.get('classmark')}")
                                        except Exception as e:
                                            logger.warning(f"Failed to get sample document: {e}")
                                else:
                                    logger.warning(f"No 'shelfmarks' aggregation found in sub-collection structure! Available keys: {list(first_sub.keys())}")
                    
                    for collection_bucket in collection_buckets:
                        collection_name = collection_bucket['key'] or "Unknown"
                        collection_count = collection_bucket['doc_count']
                        
                        # Process sub_collections
                        sub_collections = {}
                        sub_collection_buckets = collection_bucket.get('sub_collections', {}).get('buckets', [])
                        
                        # Documents of a collection that carry no sub_collection land in
                        # the "Unknown" bucket; keep their shelfmarks so they stay browsable.
                        other_shelfmarks: List[Dict[str, Any]] = []
                        other_count = 0

                        # If we have sub_collections, use them
                        if sub_collection_buckets:
                            for sub_collection_bucket in sub_collection_buckets:
                                sub_collection_name = sub_collection_bucket['key'] or "Unknown"
                                sub_collection_count = sub_collection_bucket['doc_count']
                                if sub_collection_name == "Unknown":
                                    other_count = sub_collection_count
                                    other_shelfmarks = [
                                        {"name": b['key'], "count": b['doc_count']}
                                        for b in sub_collection_bucket.get('shelfmarks', {}).get('buckets', [])
                                        if b.get('key')
                                    ]
                                
                                # Process shelfmarks
                                shelfmarks = []
                                shelfmark_buckets = sub_collection_bucket.get('shelfmarks', {}).get('buckets', [])
                                
                                for shelfmark_bucket in shelfmark_buckets:
                                    shelfmark_name = shelfmark_bucket['key'] or "Unknown"
                                    shelfmark_count = shelfmark_bucket['doc_count']
                                    
                                    shelfmarks.append({
                                        "name": shelfmark_name,
                                        "count": shelfmark_count
                                    })
                                
                                if sub_collection_name and sub_collection_name != "Unknown":
                                    shelfmark_count = len(shelfmarks)
                                    
                                    # If aggregation returned no shelfmarks but we have many documents,
                                    # try to fetch shelfmarks using get_shelfmark_distribution
                                    if shelfmark_count == 0 and sub_collection_count > 1000:
                                        logger.info(f"No shelfmarks from aggregation for '{sub_collection_name}', "
                                                   f"fetching directly (docs: {sub_collection_count})")
                                        try:
                                            dist = self.get_shelfmark_distribution(
                                                collection_name, 
                                                sub_collection_name, 
                                                index_name=target_index,
                                                size=10000  # Get more shelfmarks
                                            )
                                            if dist.get("buckets"):
                                                shelfmarks = [{
                                                    "name": b.get("key"),
                                                    "count": b.get("doc_count", 0)
                                                } for b in dist["buckets"] if b.get("key")]
                                                shelfmark_count = len(shelfmarks)
                                                logger.info(f"Fetched {shelfmark_count} shelfmarks for '{sub_collection_name}'")
                                        except Exception as e:
                                            logger.warning(f"Failed to fetch shelfmarks for '{sub_collection_name}': {e}")
                                    
                                    # If sub-collection has > 500 shelfmarks OR > 1000 documents, group into sub-sub-collections
                                    should_group = shelfmark_count > 500 or sub_collection_count > 1000
                                    
                                    if should_group:
                                        logger.info(f"Grouping large sub-collection '{sub_collection_name}' "
                                                   f"(shelfmarks: {shelfmark_count}, docs: {sub_collection_count})")
                                        if shelfmark_count > 0:
                                            sub_sub_collections = self._group_shelfmarks_by_range(shelfmarks, range_size=100)
                                            logger.info(f"Created {len(sub_sub_collections)} sub-sub-collections for '{sub_collection_name}'")
                                            sub_collections[sub_collection_name] = {
                                                "name": sub_collection_name,
                                                "count": sub_collection_count,
                                                "shelfmarks": [],  # Empty - use sub_sub_collections instead
                                                "sub_sub_collections": sub_sub_collections,
                                                "is_large": True
                                            }
                                        else:
                                            # No shelfmarks found, but large collection - mark as large but no sub-sub-collections
                                            logger.warning(f"Large sub-collection '{sub_collection_name}' has no shelfmarks to group")
                                            sub_collections[sub_collection_name] = {
                                                "name": sub_collection_name,
                                                "count": sub_collection_count,
                                                "shelfmarks": [],
                                                "is_large": True
                                            }
                                    else:
                                        sub_collections[sub_collection_name] = {
                                            "name": sub_collection_name,
                                            "count": sub_collection_count,
                                            "shelfmarks": shelfmarks,
                                            "is_large": False
                                        }
                        
                        # Named series exist but some documents have none: expose them too.
                        if sub_collections and other_shelfmarks:
                            if len(other_shelfmarks) > 500 or other_count > 1000:
                                sub_collections["_other"] = {
                                    "name": "Other (no series)",
                                    "count": other_count,
                                    "shelfmarks": [],
                                    "sub_sub_collections": self._group_shelfmarks_by_range(other_shelfmarks, range_size=100),
                                    "is_large": True
                                }
                            else:
                                sub_collections["_other"] = {
                                    "name": "Other (no series)",
                                    "count": other_count,
                                    "shelfmarks": other_shelfmarks,
                                    "is_large": False
                                }

                        # If no sub_collections, use shelfmarks directly under collection
                        if not sub_collections:
                            shelfmarks_direct = []
                            shelfmark_buckets_direct = collection_bucket.get('shelfmarks_direct', {}).get('buckets', [])
                            
                            for shelfmark_bucket in shelfmark_buckets_direct:
                                shelfmark_name = shelfmark_bucket['key'] or "Unknown"
                                shelfmark_count = shelfmark_bucket['doc_count']
                                
                                shelfmarks_direct.append({
                                    "name": shelfmark_name,
                                    "count": shelfmark_count
                                })
                            
                            # Create a default sub-collection for direct shelfmarks
                            if shelfmarks_direct:
                                # If direct shelfmarks > 1000 OR collection has > 1000 docs, group into sub-sub-collections
                                shelfmark_count = len(shelfmarks_direct)
                                should_group = shelfmark_count > 1000 or collection_count > 1000
                                
                                if should_group:
                                    logger.info(f"Grouping large direct shelfmarks collection "
                                               f"(shelfmarks: {shelfmark_count}, docs: {collection_count})")
                                    sub_sub_collections = self._group_shelfmarks_by_range(shelfmarks_direct, range_size=100)
                                    logger.info(f"Created {len(sub_sub_collections)} sub-sub-collections for direct shelfmarks")
                                    sub_collections["_all"] = {
                                        "name": "All Shelfmarks",
                                        "count": collection_count,
                                        "shelfmarks": [],
                                        "sub_sub_collections": sub_sub_collections,
                                        "is_large": True
                                    }
                                else:
                                    sub_collections["_all"] = {
                                        "name": "All Shelfmarks",
                                        "count": collection_count,
                                        "shelfmarks": shelfmarks_direct,
                                        "is_large": False
                                    }
                        
                        hierarchy[collection_name] = {
                            "name": collection_name,
                            "count": collection_count,
                            "sub_collections": sub_collections
                        }
                    
                    # If we got results, return them
                    if hierarchy:
                        logger.info(f"Successfully built hierarchy using fields: {coll_field}, {sub_coll_field}, {shelf_field}")
                        return self.prune_unbrowsable(hierarchy)
                    
                except Exception as e:
                    logger.warning(f"Failed aggregation with fields {coll_field}, {sub_coll_field}, {shelf_field}: {e}")
                    continue
            
            # If all variations failed, return empty hierarchy
            logger.warning(f"All field variations failed for index {target_index}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get collection hierarchy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_shelfmark_documents(self, shelfmark: str, include_embeddings: bool = True, index_name: Optional[str] = None) -> List[SearchResult]:
        """
        Get all documents for a specific shelfmark with optional embeddings
        This is used when a user selects a shelfmark from the collection browser
        """
        try:
            target_index = index_name or self.index_name
            
            # Build query to find all documents with this shelfmark
            # Try exact keyword matches and phrase matches across common fields
            query = {
                "bool": {
                    "should": [
                        {"term": {"shelf_mark.keyword": shelfmark}},
                        {"term": {"shelf_mark": shelfmark}},
                        {"match_phrase": {"shelf_mark": shelfmark}},
                        {"term": {"shelfmark.keyword": shelfmark}},
                        {"term": {"shelfmark": shelfmark}},
                        {"match_phrase": {"shelfmark": shelfmark}},
                        {"term": {"classmark.keyword": shelfmark}},
                        {"term": {"classmark": shelfmark}},
                        {"match_phrase": {"classmark": shelfmark}},
                        {"term": {"doc_id": shelfmark}}
                    ],
                    "minimum_should_match": 1
                }
            }
            
            # Search for all documents
            response = self.es.search(
                index=target_index,
                query=query,
                size=1000,  # Get all documents for this shelfmark
                _source=True
            )
            
            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)
                
                # Include embedding if requested
                embedding = None
                if include_embeddings:
                    embedding = source.get("embedding_vector", [])
                
                doc_id = source.get("doc_id") or hit["_id"]
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=1.0,
                    metadata=metadata,
                    embedding=embedding
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get shelfmark documents: {e}")
            return []

    def sample_docs_for_collection(self, collection: str, sub_collection: Optional[str] = None, index_name: Optional[str] = None, size: int = 5) -> List[Dict[str, Any]]:
        """
        Return a small sample of documents for a given collection/sub_collection with only relevant fields
        to help debug which shelfmark field is populated in the index.
        """
        target_index = index_name or self.index_name
        # Be lenient about keyword vs text fields by OR-ing both variants
        collection_filter = {
            "bool": {
                "should": [
                    {"term": {"collection": collection}},
                    {"term": {"collection.keyword": collection}}
                ],
                "minimum_should_match": 1
            }
        }
        must_filters = [collection_filter]
        if sub_collection is not None:
            must_filters.append({
                "bool": {
                    "should": [
                        {"term": {"sub_collection": sub_collection}},
                        {"term": {"sub_collection.keyword": sub_collection}}
                    ],
                    "minimum_should_match": 1
                }
            })

        query = {"bool": {"must": must_filters}}

        try:
            resp = self.es.search(
                index=target_index,
                query=query,
                size=size,
                _source=[
                    "doc_id",
                    "collection",
                    "sub_collection",
                    "shelf_mark",
                    "shelfmark",
                    "classmark",
                    "title",
                ],
            )
            return [hit.get("_source", {}) for hit in resp.get("hits", {}).get("hits", [])]
        except Exception as e:
            logger.error(f"Failed to sample docs for {collection}/{sub_collection}: {e}")
            return []

    def get_shelfmark_distribution(self, collection: str, sub_collection: Optional[str] = None, index_name: Optional[str] = None, size: int = 100) -> Dict[str, Any]:
        """
        Return a terms aggregation of shelf marks for a given collection/sub_collection.
        Tries preferred field order and reports which field succeeded.
        """
        target_index = index_name or self.index_name
        collection_filter = {
            "bool": {
                "should": [
                    {"term": {"collection": collection}},
                    {"term": {"collection.keyword": collection}}
                ],
                "minimum_should_match": 1
            }
        }
        must_filters = [collection_filter]
        if sub_collection is not None:
            must_filters.append({
                "bool": {
                    "should": [
                        {"term": {"sub_collection": sub_collection}},
                        {"term": {"sub_collection.keyword": sub_collection}}
                    ],
                    "minimum_should_match": 1
                }
            })

        base_query = {"bool": {"must": must_filters}}

        field_candidates = [
            "shelf_mark.keyword",
            "shelfmark.keyword",
            "classmark.keyword",
            "shelf_mark",
            "shelfmark",
            "classmark",
        ]

        for field in field_candidates:
            try:
                resp = self.es.search(
                    index=target_index,
                    size=0,
                    query=base_query,
                    aggs={
                        "shelfmarks": {
                            "terms": {
                                "field": field,
                                "size": size,
                                "min_doc_count": 1
                            },
                            "aggs": {
                                "sample_docs": {
                                    "top_hits": {
                                        "size": 5,
                                        "_source": ["doc_id"]
                                    }
                                }
                            }
                        },
                        "has_field": {"filter": {"exists": {"field": field}}}
                    }
                )
                buckets = resp.get("aggregations", {}).get("shelfmarks", {}).get("buckets", [])
                has_field = resp.get("aggregations", {}).get("has_field", {}).get("doc_count", 0)
                if buckets or has_field > 0:
                    # Enrich buckets with sample doc_ids
                    enriched = []
                    for b in buckets:
                        hits = b.get("sample_docs", {}).get("hits", {}).get("hits", [])
                        doc_ids = [h.get("_source", {}).get("doc_id") or h.get("_id") for h in hits if (h.get("_source") or h.get("_id"))]
                        e = dict(b)
                        e["doc_ids"] = [d for d in doc_ids if d]
                        enriched.append(e)
                    return {
                        "field_used": field,
                        "has_field_count": has_field,
                        "buckets": enriched
                    }
            except Exception as e:
                logger.warning(f"Shelfmark distribution failed on field {field}: {e}")
                continue

        return {"field_used": None, "has_field_count": 0, "buckets": []}

    def get_shelfmarks_list(self, collection: str, sub_collection: Optional[str] = None, index_name: Optional[str] = None, size: int = 1000) -> Dict[str, Any]:
        """
        Return a list of shelfmarks with counts for a given collection/sub_collection.
        First tries an aggregation; if empty, falls back to scanning documents and counting in Python.
        """
        # Try aggregation path first
        dist = self.get_shelfmark_distribution(collection, sub_collection, index_name, size)
        if dist.get("buckets"):
            return {
                "field_used": dist.get("field_used"),
                "shelfmarks": [{"name": b.get("key"), "count": b.get("doc_count", 0)} for b in dist["buckets"] if b.get("key")]
            }

        # Fallback: scan docs and count
        target_index = index_name or self.index_name
        collection_filter = {
            "bool": {
                "should": [
                    {"term": {"collection": collection}},
                    {"term": {"collection.keyword": collection}}
                ],
                "minimum_should_match": 1
            }
        }
        must_filters = [collection_filter]
        if sub_collection is not None:
            must_filters.append({
                "bool": {
                    "should": [
                        {"term": {"sub_collection": sub_collection}},
                        {"term": {"sub_collection.keyword": sub_collection}}
                    ],
                    "minimum_should_match": 1
                }
            })

        try:
            resp = self.es.search(
                index=target_index,
                query={"bool": {"must": must_filters}},
                size=min(size, 5000),
                _source=["shelf_mark", "shelfmark", "classmark", "doc_id"],
            )
        except Exception as e:
            logger.error(f"Fallback shelfmark scan failed: {e}")
            return {"field_used": None, "shelfmarks": []}

        counts: Dict[str, int] = {}
        def add(name: Optional[str]):
            if not name:
                return
            counts[name] = counts.get(name, 0) + 1

        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            # Prefer shelf_mark > shelfmark > classmark > doc_id
            value = src.get("shelf_mark") or src.get("shelfmark") or src.get("classmark") or src.get("doc_id")
            add(value)

        shelfmarks = sorted(
            ( {"name": k, "count": v} for k, v in counts.items() if k ),
            key=lambda x: (-x["count"], x["name"])
        )

        return {"field_used": dist.get("field_used"), "shelfmarks": shelfmarks}


# Global search service
search_service = ElasticsearchService()