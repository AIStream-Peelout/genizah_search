# Enhanced search_service.py with embedding vectors for t-SNE visualization

import os
import numpy as np
from rate_limits import ProtectionService, RateLimitExceeded, FilterOptions
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
import logging
import time
import json
from fastapi import HTTPException, Request, status
from temp import NomicsEmbedding
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


# Enhanced Pydantic models for API with embedding support
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
    document_types: Optional[List[str]] = None
    document_type: Optional[str] = None
    content_type: Optional[str] = None
    transcription_full_text: Optional[str] = None
    translation_full_text: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: Optional[List[str]] = None
    has_images: Optional[bool] = None
    has_description: Optional[bool] = None
    has_transcriptions: Optional[bool] = None
    has_translations: Optional[bool] = None
    has_date: Optional[bool] = None
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
    original_url: Optional[str] = None
    indexed_at: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results")
    include_embeddings: Optional[bool] = Field(default=False, description="Include embedding vectors for visualization")


class SearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: float
    metadata: Optional[DocumentMetadata] = None
    embedding: Optional[List[float]] = None  # Added for t-SNE visualization


class EmbeddingData(BaseModel):
    """Embedding data for t-SNE visualization"""
    query_embedding: List[float]
    result_embeddings: List[List[float]]
    dimension: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    count: int
    filters_applied: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    embedding_data: Optional[EmbeddingData] = None  # Added for t-SNE visualization


class ElasticsearchService:
    """Updated Elasticsearch service with embedding vector support"""

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
            'institution': 'institution',
            'library': 'library',
            'collection': 'collection',
            'collection_type': 'collection_type',
            'content_type': 'content_type',
            'document_type': 'document_type',
            'has_transcriptions': 'has_transcriptions',
            "has_bib": "has_bib",
            'has_translations': 'has_translations',
            'has_images': 'has_images',
            'has_description': 'has_description',
            'has_date': 'has_date',
            'transcription_completeness': 'transcription_completeness',
            'donation_year': 'donation_year',
            'source_institution': 'source_institution',
            'period': 'period',
            'physical_location': 'physical_location'
        }

        for filter_key, es_field in filter_mappings.items():
            if filter_key in filters and filters[filter_key] is not None:
                value = filters[filter_key]

                # Handle document_types as array field
                if filter_key == 'document_types':
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {"document_types": value}})
                    else:
                        filter_clauses.append({"term": {"document_types": value}})
                # Handle donor_surnames as array field
                elif filter_key == 'donor_surnames':
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {"donor_surnames": value}})
                    else:
                        filter_clauses.append({"term": {"donor_surnames": value}})
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
        language = metadata.get('language', '')
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
        language = metadata.get('language', '')

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

        if metadata.get('institution'):
            institution = metadata['institution'].replace('_', ' ').title()
            parts.append(f"housed at {institution}")

        return " ".join(parts) + "."

    def _generate_image_urls(self, doc_id: str, metadata: Dict[str, Any]) -> tuple:
        """Generate image URLs - use actual image_url if available"""
        # Use actual image URL from metadata if available
        if metadata.get('actual_image_url'):
            image_url = metadata['actual_image_url']
            # Generate thumbnail from main image
            thumbnail_url = image_url.replace('/full/', '/400,/')
            return image_url, thumbnail_url

        # Fallback to placeholder images
        base_url = "https://images.unsplash.com"
        language = metadata.get('language', '').lower()

        if 'hebrew' in language:
            image_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop"
        elif 'arabic' in language:
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

        # Add document type tags
        if metadata.get('document_types'):
            tags.extend(metadata['document_types'])

        if metadata.get('document_type'):
            tags.append(metadata['document_type'])

        # Add collection tags
        if metadata.get('collection'):
            tags.append(metadata['collection'])
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
        if metadata.get('source_institution'):
            tags.append(metadata['source_institution'])

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

        # Generate enhanced metadata
        title = self._generate_title(doc_id, source)
        description = self._generate_description(source)
        image_url, thumbnail_url = self._generate_image_urls(doc_id, source)
        tags = self._extract_tags(source)

        return DocumentMetadata(
            title=title,
            description=description,
            language=source.get('language'),
            period=source.get('period'),
            date_info=source.get('date_info'),
            location=source.get('physical_location'),
            material=source.get('material'),
            dimensions=source.get('dimensions'),
            institution=source.get('institution'),
            library=source.get('library'),
            collection=source.get('collection'),
            collection_type=source.get('collection_type'),
            shelfmark=source.get('classmark', doc_id),
            document_types=source.get('document_types'),
            document_type=source.get('document_type'),
            content_type=source.get('content_type'),
            transcription_full_text=self.extract_text_field(source.get('transcriptions')),
            translation_full_text=self.extract_text_field(source.get('translations')),
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            tags=tags,
            has_images=source.get('has_images'),
            has_description=source.get('has_description'),
            has_transcriptions=source.get('has_transcriptions'),
            has_bib = source.get('has_transcriptions'),
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
            indexed_at=source.get('indexed_at')
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

            # Execute search using ES 8.x syntax
            response = self.es.search(
                index=self.index_name,
                query=query,
                size=request.num_results,
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

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied=request.filters,
                processing_time_ms=round(processing_time, 2),
                embedding_data=embedding_data
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
                "institutions": {"terms": {"field": "institution", "size": 100}},
                "libraries": {"terms": {"field": "library", "size": 100}},
                "collections": {"terms": {"field": "collection", "size": 100}},
                "collection_types": {"terms": {"field": "collection_type", "size": 100}},
                "content_types": {"terms": {"field": "content_type", "size": 100}},
                "document_types": {"terms": {"field": "document_type", "size": 100}},
                "transcription_completeness": {"terms": {"field": "transcription_completeness", "size": 10}}
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
                document_types=['contract', 'marriage', 'court', 'fragment'],
                institutions=['cambridge'],
                collections=['taylor_schechter']
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

# Global protection service
protection_service = ProtectionService()


# Dependency for rate limiting
async def check_rate_limits(request: Request):
    """Dependency to check rate limits"""
    try:
        await protection_service.check_limits(request)
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.message
        )