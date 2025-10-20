# Enhanced search_service.py with additional metadata fields

import os
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
import logging
import time
import json
from fastapi import HTTPException, Request, status
from temp import NomicsEmbedding
from elasticsearch import Elasticsearch
from models.pydantic_core import FilterOptions

logger = logging.getLogger(__name__)


# Enhanced Pydantic models for API with additional metadata
class DocumentMetadata(BaseModel):
    """Rich document metadata matching new ES structure"""
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
    image_urls: Optional[List[str]] = None
    completeness_score: Optional[float] = None
    content_quality: Optional[str] = None
    miscellaneous_info: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=10, ge=1, le=20, description="Number of results")
    include_embeddings: Optional[bool] = Field(default=False, description="Include embedding vectors for visualization")
    page: Optional[int] = Field(default=1, ge=1, description="Page number for pagination (1-based)")


class SearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: Optional[float] = None
    metadata: Optional[DocumentMetadata] = None
    embedding: Optional[List[float]] = None  # Added for t-SNE visualization


class EmbeddingData(BaseModel):
    """Embedding data for t-SNE visualization"""
    query_embedding: Optional[List[float]] = None
    result_embeddings: List[List[float]]
    dimension: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: Optional[str] = None
    count: int  # count of results returned in this page
    filters_applied: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    embedding_data: Optional[EmbeddingData] = None
    # Pagination metadata
    total: Optional[int] = None  # total matching documents across all pages
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None
    has_more: Optional[bool] = None


class ElasticsearchService:
    """Updated Elasticsearch service with enhanced metadata extraction"""

    def __init__(self):
        self.es_host = os.getenv('ELASTICSEARCH_HOST', 'elastic.cairogenizah.ai')
        self.es_port = os.getenv('ELASTICSEARCH_PORT', '443')
        self.index_name = os.getenv('ELASTICSEARCH_INDEX', 'cairo_genizah_text_only_v1.0.1')
        self.es = None
        self._initialize_elasticsearch()
        self.embedding_model = NomicsEmbedding()
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch connection for ES 8.x"""
        # ES 8.x connection
        self.es = Elasticsearch(
            [f"https://{self.es_host}:{self.es_port}"],
            basic_auth=("cairo_user", os.getenv('ELASTICSEARCH_PASSWORD')),
            verify_certs=False,
        )

    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> List[Dict]:
        """Convert search filters to Elasticsearch query clauses for new structure"""
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
            'source_collection': 'source_collection',
            'collection_type': 'collection_type',
            'content_type': 'content_type',
            'document_type': 'document_type',
            'has_transcriptions': 'has_transcriptions',
            'has_bib': 'has_bib',
            'has_translations': 'has_translations',
            'has_images': 'has_images',
            'has_description': 'has_description',
            'has_date': 'has_date',
            'transcription_completeness': 'transcription_completeness',
            'donation_year': 'donation_year',
            'source_institution': 'source_institution',
            'period': 'period',
            'physical_location': 'physical_location',
            'material': 'material',
            'script_type': 'script_type',
            'date_certainty': 'date_certainty'
        }

        for filter_key, es_field in filter_mappings.items():
            if filter_key in filters and filters[filter_key] is not None:
                value = filters[filter_key]

                # Handle array fields
                if filter_key in ['document_types', 'donor_surnames', 'other_languages']:
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {es_field: value}})
                    else:
                        filter_clauses.append({"term": {es_field: value}})
                # Handle other array or single values
                elif isinstance(value, list):
                    filter_clauses.append({"terms": {es_field: value}})
                else:
                    filter_clauses.append({"term": {es_field: value}})

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

        return filter_clauses

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

    def _generate_image_urls(self, doc_id: str, metadata: Dict[str, Any]) -> tuple:
        """Generate image URLs - prioritize actual_image_url"""
        # Use actual image URL from metadata if available
        if metadata.get('actual_image_url'):
            image_url = metadata['actual_image_url']
            # Generate thumbnail from main image
            thumbnail_url = image_url.replace('/full/', '/400,/')
            return image_url, thumbnail_url

        # Use first image from image_urls array if available
        if metadata.get('image_urls') and len(metadata['image_urls']) > 0:
            image_url = metadata['image_urls'][0]
            # Try to generate thumbnail
            if '/full/' in image_url:
                thumbnail_url = image_url.replace('/full/', '/400,/')
            else:
                thumbnail_url = image_url
            return image_url, thumbnail_url

        # Fallback to placeholder images
        base_url = "https://images.unsplash.com"
        language = metadata.get('language', metadata.get('main_language', '')).lower()

        if 'hebrew' in language or 'heb' in language:
            image_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop"
        elif 'arabic' in language or 'ara' in language:
            image_url = f"{base_url}/photo-1544716278-ca5e3f4abd8c?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1544716278-ca5e3f4abd8c?w=400&h=300&fit=crop"
        else:
            image_url = f"{base_url}/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1507003211169-0a1dd7228f2d?w=400&h=300&fit=crop"

        return image_url, thumbnail_url

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
        if metadata.get('source_collection'):
            tags.append(metadata['source_collection'])
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
    

    def _extract_metadata(self, source: Dict[str, Any]) -> DocumentMetadata:
        """Extract and format document metadata from new ES structure"""
        doc_id = source.get('doc_id', 'Unknown')

        # Handle complex transcriptions (can be objects or strings)
        transcription_text = None
        transcriptions_raw = source.get('transcriptions', [])
        if transcriptions_raw:
            if isinstance(transcriptions_raw, list):
                texts = []
                for trans in transcriptions_raw:
                    if isinstance(trans, dict):
                        # Extract text from transcription objects - try common field names
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

        # Handle complex translations (can be objects or strings)
        translation_text = None
        translations_raw = source.get('translations', [])
        if translations_raw:
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

        # Handle complex bibliography (objects with citation field)
        bibliography_list = []
        bibliography_raw = source.get('bibliography', [])
        if bibliography_raw and isinstance(bibliography_raw, list):
            for bib in bibliography_raw:
                if isinstance(bib, dict):
                    # Extract citation text from the object
                    citation = (bib.get('citation') or 
                            bib.get('reference') or 
                            bib.get('text') or 
                            str(bib))
                    bibliography_list.append(citation)
                else:
                    bibliography_list.append(str(bib))

        # Generate enhanced metadata
        title = self._generate_title(doc_id, source)
        description = self._generate_description(source)
        image_url, thumbnail_url = self._generate_image_urls(doc_id, source)
        tags = self._extract_tags(source)
        dimensions = self._format_dimensions(source.get('height'), source.get('width'))

        return DocumentMetadata(
            title=title,
            description=description,
            language=source.get('language'),
            main_language=source.get('main_language'),
            other_languages=source.get('other_languages'),
            period=source.get('period'),
            date_info=source.get('date_info'),
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
            shelf_mark=source.get('shelf_mark', source.get('classmark', doc_id)),
            document_types=source.get('document_types'),
            document_type=source.get('document_type'),
            content_type=source.get('content_type'),
            script_type=source.get('script_type'),
            date_certainty=source.get('date_certainty'),
            transcription_full_text=transcription_text,
            translation_full_text=translation_text,
            transcriptions=transcriptions_raw,  # Keep raw data for complex handling
            translations=translations_raw,      # Keep raw data for complex handling
            bibliography=bibliography_list,     # Processed strings
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            actual_image_url=source.get('actual_image_url'),
            image_urls=source.get('image_urls', []),
            tags=tags,
            has_images=source.get('has_images'),
            has_description=source.get('has_description'),
            has_transcriptions=source.get('has_transcriptions'),
            has_bib=source.get('has_bib'),
            has_translations=source.get('has_translations'),
            has_date=source.get('has_date'),
            transcription_completeness=source.get('transcription_completeness'),
            transcription_count=source.get('transcription_count'),
            total_transcription_lines=source.get('total_transcription_lines'),
            translation_count=source.get('translation_count'),
            donation_year=source.get('donation_year'),
            donor_surnames=source.get('donor_surnames'),
            source_institution=source.get('source_institution'),
            physical_location=source.get('physical_location'),
            classmark=source.get('classmark'),
            provenance=source.get('provenance'),
            original_url=source.get('original_url'),
            indexed_at=source.get('indexed_at'),
            named_entities=source.get('named_entities'),
            completeness_score=source.get('completeness_score'),
            content_quality=source.get('content_quality'),
            miscellaneous_info=source.get('miscellaneous_info')
        )

    def _get_document_embeddings(self, hits: List[Dict]) -> List[List[float]]:
        """Extract embedding vectors from Elasticsearch hits"""
        embeddings = []
        for hit in hits:
            # Get the embedding from the document source
            embedding = hit["_source"].get("embedding_vector", [])
            if embedding:
                embeddings.append(embedding)
            else:
                # If no embedding found, create a zero vector or skip
                logger.warning(f"No embedding found for document {hit['_source'].get('doc_id', 'unknown')}")
                # You might want to generate an embedding on the fly or use a default
                embeddings.append([0.0] * 768)  # Assuming 768-dimensional embeddings
        return embeddings

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform vector similarity search with optional embedding data for visualization"""
        start_time = time.time()

        try:
            # Generate query embedding using your existing embedding model
            query_embedding = self.embedding_model.get_embeddings(
                None, request.query, use_cache=False
            )

            # Build filter clauses
            filter_clauses = self._build_filters(request.filters)

            # Build Elasticsearch query using script_score for vector similarity
            if filter_clauses:
                base_query = {"bool": {"filter": filter_clauses}}
            else:
                base_query = {"match_all": {}}

            # ES 8.x query structure
            query = {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": query_embedding.flatten().tolist()}
                    }
                }
            }

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Execute search using ES 8.x syntax
            response = self.es.search(
                index=self.index_name,
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

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied=request.filters,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data,
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more
            )

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

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get full document details by ID"""
        try:
            response = self.es.search(
                index=self.index_name,
                query={"term": {"doc_id": doc_id}},
                size=1
            )

            if response['hits']['total']['value'] > 0:
                source = response['hits']['hits'][0]['_source']
                return self._extract_metadata(source)

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def get_filter_options(self) -> FilterOptions:
        """Get available filter options from the updated index"""
        try:
            # Updated aggregations for new field structure
            aggs = {
                "languages": {"terms": {"field": "language", "size": 100}},
                "main_languages": {"terms": {"field": "main_language", "size": 100}},
                "institutions": {"terms": {"field": "institution", "size": 100}},
                "repositories": {"terms": {"field": "repository", "size": 100}},
                "libraries": {"terms": {"field": "library", "size": 100}},
                "collections": {"terms": {"field": "collection", "size": 100}},
                "source_collections": {"terms": {"field": "source_collection", "size": 100}},
                "collection_types": {"terms": {"field": "collection_type", "size": 100}},
                "content_types": {"terms": {"field": "content_type", "size": 100}},
                "document_types": {"terms": {"field": "document_type", "size": 100}},
                "transcription_completeness": {"terms": {"field": "transcription_completeness", "size": 10}},
                "materials": {"terms": {"field": "material", "size": 50}},
                "script_types": {"terms": {"field": "script_type", "size": 20}}
            }

            response = self.es.search(
                index=self.index_name,
                size=0,
                aggs=aggs
            )

            return FilterOptions(
                languages=[bucket["key"] for bucket in response["aggregations"]["languages"]["buckets"]],
                periods=['early_medieval', 'late_medieval', 'early_modern'],
                document_types=[bucket["key"] for bucket in response["aggregations"]["document_types"]["buckets"]],
                institutions=[bucket["key"] for bucket in response["aggregations"]["institutions"]["buckets"]],
                collections=[bucket["key"] for bucket in response["aggregations"]["collections"]["buckets"]]
            )
        except Exception as e:
            logger.warning(f"Could not get filter options: {e}")
            # Return defaults
            return FilterOptions(
                languages=['Hebrew', 'Judaeo-Arabic', 'Arabic', 'Aramaic'],
                periods=['early_medieval', 'late_medieval', 'early_modern'],
                document_types=['contract', 'marriage', 'court', 'fragment', 'tanakh', 'talmud'],
                institutions=['cambridge', 'JTS', "UPenn"],
                collections=['taylor_schechter']
            )

    async def search_by_shelfmark(self, request) -> SearchResponse:
        """Search documents by shelf mark with exact or partial matching"""
        start_time = time.time()

        try:
            # Build query based on exact_match preference
            if request.exact_match:
                # Exact match query - search in shelf_mark, shelfmark, and classmark fields
                query = {
                    "bool": {
                        "should": [
                            {"term": {"shelf_mark": request.shelf_mark}},
                            {"term": {"shelfmark": request.shelf_mark}},
                            {"term": {"classmark": request.shelf_mark}},
                            {"term": {"doc_id": request.shelf_mark}}  # Also search doc_id as fallback
                        ],
                        "minimum_should_match": 1
                    }
                }
            else:
                # Partial match query - use wildcard and prefix matching
                query = {
                    "bool": {
                        "should": [
                            {"wildcard": {"shelf_mark": f"*{request.shelf_mark}*"}},
                            {"wildcard": {"shelfmark": f"*{request.shelf_mark}*"}},
                            {"wildcard": {"classmark": f"*{request.shelf_mark}*"}},
                            {"wildcard": {"doc_id": f"*{request.shelf_mark}*"}},
                            {"prefix": {"shelf_mark": request.shelf_mark}},
                            {"prefix": {"shelfmark": request.shelf_mark}},
                            {"prefix": {"classmark": request.shelf_mark}},
                            {"prefix": {"doc_id": request.shelf_mark}}
                        ],
                        "minimum_should_match": 1
                    }
                }

            # Execute search
            response = self.es.search(
                index=self.index_name,
                query=query,
                size=request.num_results or 10,
                _source=True
            )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                doc_id = source.get("doc_id") or hit["_id"]

                # Calculate relevance score based on field match
                relevance_score = self._calculate_shelfmark_relevance(
                    source, request.shelf_mark, request.exact_match
                )

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=relevance_score,
                    distance=1.0 - relevance_score,  # Convert to distance
                    metadata=metadata,
                    embedding=None  # No embeddings for shelf mark search
                ))

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

            return SearchResponse(
                results=results,
                query=f"Shelf mark: {request.shelf_mark}",
                count=len(results),
                filters_applied={"shelf_mark": request.shelf_mark, "exact_match": request.exact_match},
                processing_time_ms=round(processing_time, 2),
                embedding_data=None,  # No embeddings for shelf mark search
                total=total_hits_value,
                page=1,
                page_size=request.num_results or 10,
                total_pages=1,
                has_more=False
            )

        except Exception as e:
            logger.error(f"Shelf mark search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Shelf mark search failed: {str(e)}"
            )

    def _calculate_shelfmark_relevance(self, source: Dict[str, Any], query: str, exact_match: bool) -> float:
        """Calculate relevance score for shelf mark matches"""
        score = 0.0
        
        # Check each shelf mark field
        shelf_fields = ['shelf_mark', 'shelfmark', 'classmark', 'doc_id']
        
        for field in shelf_fields:
            field_value = source.get(field, '')
            if not field_value:
                continue
                
            if exact_match:
                if field_value == query:
                    # Exact match gets highest score
                    if field == 'shelf_mark':
                        score = max(score, 1.0)
                    elif field == 'shelfmark':
                        score = max(score, 0.95)
                    elif field == 'classmark':
                        score = max(score, 0.9)
                    elif field == 'doc_id':
                        score = max(score, 0.85)
            else:
                # Partial match scoring
                if query.lower() in field_value.lower():
                    # Calculate score based on how much of the query matches
                    match_ratio = len(query) / len(field_value)
                    field_score = match_ratio * 0.8  # Base score for partial match
                    
                    # Boost score based on field priority
                    if field == 'shelf_mark':
                        field_score *= 1.0
                    elif field == 'shelfmark':
                        field_score *= 0.95
                    elif field == 'classmark':
                        field_score *= 0.9
                    elif field == 'doc_id':
                        field_score *= 0.85
                    
                    score = max(score, field_score)
        
        return min(score, 1.0)  # Cap at 1.0

    async def search_by_keyword(self, request) -> SearchResponse:
        """Search documents by keywords in text fields"""
        start_time = time.time()

        try:
            # Build keyword query that searches across multiple text fields
            query = {
                "multi_match": {
                    "query": request.query,
                    "fields": [
                        "transcription_full_text^3.0",
                        "translation_full_text^2.5", 
                        "description^2.0",
                        "title^2.5",
                        "document_type^1.5",
                        "content_type^1.5",
                        "collection^1.2",
                        "language^1.2",
                        "script_type^1.1",
                        "material^1.0"
                    ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "boost": 1.0
            }
        }

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Execute search
            response = self.es.search(
                index=self.index_name,
                query=query,
                size=page_size,
                from_=from_offset,
                _source=True
            )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                doc_id = source.get("doc_id") or hit["_id"]

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=round(hit["_score"], 4),
                    distance=round(max(0, 10.0 - hit["_score"]), 4),  # Convert to distance-like metric
                    metadata=metadata,
                    embedding=None  # No embeddings for keyword search
                ))

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

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied={"search_type": "keyword"},
                processing_time_ms=round(processing_time, 2),
                embedding_data=None,  # No embeddings for keyword search
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more
            )

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Keyword search failed: {str(e)}"
            )

    async def get_visualization_explorer_data(self, request) -> SearchResponse:
        """
        Load documents for visualization explorer
        
        Loads a random sample of documents from the collection for full-page
        visualization exploration. Supports loading a configurable number of
        documents or the entire index.
        """
        start_time = time.time()
        
        try:
            # Determine how many documents to load
            if request.load_full_index:
                # Get total document count
                stats = self.es.indices.stats(index=self.index_name)
                total_docs = stats['indices'][self.index_name]['total']['docs']['count']
                num_docs_to_load = total_docs
            else:
                num_docs_to_load = min(request.num_documents, 10000)  # Cap at 10k for performance
            
            logger.info(f"Loading {num_docs_to_load} documents for visualization explorer")
            
            # Use scroll API for large result sets
            if num_docs_to_load > 1000:
                # For large sets, use scroll API
                query = {
                    "match_all": {}
                }
                
                # Initial search
                response = self.es.search(
                    index=self.index_name,
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
                    index=self.index_name,
                    query=query,
                    size=num_docs_to_load,
                    _source=True
                )
                
                all_hits = response['hits']['hits']
            
            # Extract embeddings if requested
            embedding_data = None
            if request.include_embeddings and all_hits:
                result_embeddings = self._get_document_embeddings(all_hits)
                embedding_data = EmbeddingData(
                    query_embedding=None,  # No query for explorer mode
                    result_embeddings=result_embeddings,
                    dimension=len(result_embeddings[0]) if result_embeddings else 128
                )
            
            # Format results with rich metadata
            results = []
            for hit in all_hits:
                doc_id = hit['_id']
                source = hit['_source']
                metadata = self._extract_metadata(source)
                
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
                page=1,
                page_size=len(results),
                total_pages=1,
                has_more=False
            )
            
        except Exception as e:
            logger.error(f"Visualization explorer data loading failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load visualization explorer data: {str(e)}"
            )

    async def search_hybrid(self, request) -> SearchResponse:
        """Perform hybrid search combining semantic and keyword search with configurable weights"""
        start_time = time.time()

        try:
            # Normalize weights to 0-1 range
            semantic_weight = request.semanticWeight / 100.0
            keyword_weight = request.keywordWeight / 100.0
            
            # Generate query embedding for semantic search
            query_embedding = self.embedding_model.get_embeddings(
                None, request.query, use_cache=False
            )

            # Build filter clauses
            filter_clauses = self._build_filters(request.filters)

            # Build base query with filters
            if filter_clauses:
                base_query = {"bool": {"filter": filter_clauses}}
            else:
                base_query = {"match_all": {}}

            # Create semantic search query
            semantic_query = {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": query_embedding.flatten().tolist()}
                    }
                }
            }

            # Create keyword search query
            keyword_query = {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": [
                                    "transcription_full_text^3.0",
                                    "translation_full_text^2.5", 
                                    "description^2.0",
                                    "title^2.5",
                                    "document_type^1.5",
                                    "content_type^1.5",
                                    "collection^1.2",
                                    "language^1.2",
                                    "script_type^1.1",
                                    "material^1.0"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 1.0
                            }
                        }
                    ],
                    "filter": filter_clauses
                }
            }

            # Combine both queries with weights using function_score
            hybrid_query = {
                "function_score": {
                    "query": base_query,
                    "functions": [
                        {
                            "filter": {"match_all": {}},
                            "weight": semantic_weight,
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                                    "params": {"query_vector": query_embedding.flatten().tolist()}
                                }
                            }
                        },
                        {
                            "filter": {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": [
                                        "transcription_full_text^3.0",
                                        "translation_full_text^2.5", 
                                        "description^2.0",
                                        "title^2.5",
                                        "document_type^1.5",
                                        "content_type^1.5",
                                        "collection^1.2",
                                        "language^1.2",
                                        "script_type^1.1",
                                        "material^1.0"
                                    ],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            "weight": keyword_weight
                        }
                    ],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }

            # Calculate pagination
            page_number = request.page or 1
            page_size = request.num_results or 10
            from_offset = (page_number - 1) * page_size

            # Execute search
            response = self.es.search(
                index=self.index_name,
                query=hybrid_query,
                size=page_size,
                from_=from_offset,
                _source=True
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

                # Include embedding in result if requested
                embedding = None
                if request.include_embeddings:
                    embedding = source.get("embedding_vector", [])

                doc_id = source.get("doc_id") or hit["_id"]

                results.append(SearchResult(
                    doc_id=doc_id,
                    similarity_score=round(hit["_score"], 4),
                    distance=round(max(0, 10.0 - hit["_score"]), 4),  # Convert to distance-like metric
                    metadata=metadata,
                    embedding=embedding
                ))

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

            return SearchResponse(
                results=results,
                query=f"Hybrid: {request.query} (Semantic: {request.semanticWeight}%, Keyword: {request.keywordWeight}%)",
                count=len(results),
                filters_applied=request.filters,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data,
                total=total_hits_value,
                page=page_number,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more
            )

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


# Global search service
search_service = ElasticsearchService()