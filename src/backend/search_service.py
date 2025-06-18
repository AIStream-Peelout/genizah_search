import os
from rate_limits import ProtectionService, RateLimitExceeded, FilterOptions
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
import logging
import time
from fastapi import HTTPException, Request, status
from temp import NomicsEmbedding
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


# Enhanced Pydantic models for API
class DocumentMetadata(BaseModel):
    """Rich document metadata"""
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    period: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    institution: Optional[str] = None
    library: Optional[str] = None
    collection: Optional[str] = None
    collection_type: Optional[str] = None
    shelfmark: Optional[str] = None
    document_type: Optional[str] = None
    content_type: Optional[str] = None
    transcription: Optional[str] = None
    translation: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: Optional[List[str]] = None
    has_images: Optional[bool] = None
    has_description: Optional[bool] = None
    has_transcriptions: Optional[bool] = None
    has_translations: Optional[bool] = None
    has_date: Optional[bool] = None
    transcription_completeness: Optional[str] = None
    donation_year: Optional[str] = None
    donor_surname: Optional[List[str]] = None
    source_institution: Optional[str] = None
    crowding_tag: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results")


class SearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: float
    metadata: Optional[DocumentMetadata] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    count: int
    filters_applied: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class ElasticsearchService:
    """Cairo Genizah Elasticsearch search service for current data structure"""

    def __init__(self):
        self.es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        self.es_port = os.getenv('ELASTICSEARCH_PORT', '9200')
        self.index_name = os.getenv('ELASTICSEARCH_INDEX', 'cairo-genizah')
        self.es = None
        self._initialize_elasticsearch()

    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        try:
            self.es = Elasticsearch([f"http://{self.es_host}:{self.es_port}"])

            # Test connection
            if not self.es.ping():
                raise Exception("Cannot connect to Elasticsearch")

            logger.info("Elasticsearch service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise

    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> List[Dict]:
        """Convert search filters to Elasticsearch query clauses"""
        if not filters:
            return []

        filter_clauses = []

        # Direct field mappings for your current structure
        filter_mappings = {
            'language': 'language',
            'institution': 'institution',
            'library': 'library',
            'collection': 'collection',
            'collection_type': 'collection_type',
            'content_type': 'content_type',
            'has_transcriptions': 'has_transcriptions',
            'has_translations': 'has_translations',
            'has_images': 'has_images',
            'has_description': 'has_description',
            'has_date': 'has_date',
            'transcription_completeness': 'transcription_completeness',
            'donation_year': 'donation_year',
            'source_institution': 'source_institution'
        }

        for filter_key, es_field in filter_mappings.items():
            if filter_key in filters and filters[filter_key] is not None:
                value = filters[filter_key]

                # Handle donor_surname as array field
                if filter_key == 'donor_surname':
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {"donor_surname": value}})
                    else:
                        filter_clauses.append({"term": {"donor_surname": value}})
                # Handle other array or single values
                elif isinstance(value, list):
                    filter_clauses.append({"terms": {es_field: value}})
                else:
                    filter_clauses.append({"term": {es_field: value}})

        return filter_clauses

    def _generate_title(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a meaningful title from document ID and metadata"""
        # Clean up document ID for display
        clean_id = doc_id.replace("MS-TS-", "T-S ").replace("-", ".").replace("/", " Fragment ")

        # Add context based on metadata
        language = metadata.get('language', '')
        collection = metadata.get('collection', '')

        # Handle multi-language documents
        if language:
            if ';' in language:
                languages = [lang.strip() for lang in language.split(';')]
                lang_display = ' & '.join(languages)
            else:
                lang_display = language
        else:
            lang_display = ''

        if lang_display and collection:
            collection_display = collection.replace('_', ' ').title()
            return f"{clean_id} - {lang_display} Manuscript ({collection_display})"
        elif lang_display:
            return f"{clean_id} - {lang_display} Manuscript"
        elif collection:
            collection_display = collection.replace('_', ' ').title()
            return f"{clean_id} - {collection_display} Collection"
        else:
            return f"Document {clean_id}"

    def _generate_description(self, metadata: Dict[str, Any]) -> str:
        """Generate a description from available metadata"""
        parts = []

        # Start with content type or default
        if metadata.get('content_type'):
            if metadata['content_type'] == 'multimodal':
                parts.append("A multimodal historical document")
            else:
                parts.append(f"A {metadata['content_type']} document")
        else:
            parts.append("A historical manuscript")

        # Add language information
        if metadata.get('language'):
            language = metadata['language']
            if ';' in language:
                languages = [lang.strip() for lang in language.split(';')]
                lang_display = ' and '.join(languages)
            else:
                lang_display = language
            parts.append(f"written in {lang_display}")

        # Add collection context
        parts.append("from the Cairo Genizah collection")

        if metadata.get('collection'):
            collection_name = metadata['collection'].replace('_', ' ').title()
            parts.append(f"in the {collection_name}")

        if metadata.get('institution'):
            institution_name = metadata['institution'].replace('_', ' ').title()
            if metadata.get('library'):
                library_name = metadata['library'].replace('_', ' ').title()
                parts.append(f"housed at {library_name}, {institution_name}")
            else:
                parts.append(f"housed at {institution_name}")

        description = " ".join(parts) + "."

        # Add historical context
        if metadata.get('donation_year'):
            description += f" This document was donated in {metadata['donation_year']}"
            if metadata.get('donor_surname'):
                donors = metadata['donor_surname']
                if isinstance(donors, list):
                    donor_names = ' and '.join([name.title() for name in donors])
                else:
                    donor_names = donors.title()
                description += f" by {donor_names}"
            description += "."

        # Add content information
        content_info = []
        if metadata.get('has_images'):
            content_info.append("includes images")
        if metadata.get('has_transcriptions'):
            content_info.append("has transcriptions")
        if metadata.get('has_translations'):
            content_info.append("has translations")
        if metadata.get('has_description'):
            content_info.append("has detailed descriptions")

        if content_info:
            description += f" This document {', '.join(content_info)}."

        return description

    def _generate_image_urls(self, doc_id: str, metadata: Dict[str, Any]) -> tuple:
        """Generate image URLs based on document ID and metadata"""
        # TODO: Replace with your actual image serving logic
        # For now, use placeholder images based on metadata

        base_url = "https://images.unsplash.com"

        # Choose placeholder based on language and content
        language = metadata.get('language', '').lower()

        if 'hebrew' in language:
            # Hebrew manuscript placeholder
            image_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop"
        elif 'arabic' in language:
            # Arabic manuscript placeholder
            image_url = f"{base_url}/photo-1544716278-ca5e3f4abd8c?w=800&h=600&fit=crop"
            thumbnail_url = f"{base_url}/photo-1544716278-ca5e3f4abd8c?w=400&h=300&fit=crop"
        else:
            # General manuscript placeholder
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

        # Add institutional tags
        if metadata.get('institution'):
            tags.append(metadata['institution'])
        if metadata.get('crowding_tag'):
            tags.append(metadata['crowding_tag'])

        return list(set(tags))  # Remove duplicates

    def _extract_metadata(self, source: Dict[str, Any]) -> DocumentMetadata:
        """Extract and format document metadata from Elasticsearch source"""
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
            period=source.get('period'),  # You might add this field later
            date=source.get('date'),  # You might add this field later
            location=source.get('location'),  # You might add this field later
            material=source.get('material'),  # You might add this field later
            dimensions=source.get('dimensions'),  # You might add this field later
            institution=source.get('institution'),
            library=source.get('library'),
            collection=source.get('collection'),
            collection_type=source.get('collection_type'),
            shelfmark=doc_id,  # Use doc_id as shelfmark
            document_type=source.get('document_type'),
            content_type=source.get('content_type'),
            transcription=source.get('transcription'),
            translation=source.get('translation'),
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            tags=tags,
            has_images=source.get('has_images'),
            has_description=source.get('has_description'),
            has_transcriptions=source.get('has_transcriptions'),
            has_translations=source.get('has_translations'),
            has_date=source.get('has_date'),
            transcription_completeness=source.get('transcription_completeness'),
            donation_year=source.get('donation_year'),
            donor_surname=source.get('donor_surname'),
            source_institution=source.get('source_institution'),
            crowding_tag=source.get('crowding_tag')
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform vector similarity search with rich metadata"""
        start_time = time.time()

        try:
            # Generate query embedding using your existing embedding model
            from temp import NomicsEmbedding
            embedding_model = NomicsEmbedding()
            query_embedding = embedding_model.get_embeddings(
                None, request.query, use_cache=False
            )

            # Build filter clauses
            filter_clauses = self._build_filters(request.filters)

            # Build Elasticsearch query
            query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filter_clauses
                        }
                    } if filter_clauses else {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.flatten().tolist()}
                    }
                }
            }

            # Execute search - request all available fields
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": request.num_results,
                    "_source": True  # Get all fields
                }
            )

            # Format results with rich metadata
            results = []
            for hit in response['hits']['hits']:
                source = hit["_source"]
                metadata = self._extract_metadata(source)

                results.append(SearchResult(
                    doc_id=source["doc_id"],
                    similarity_score=round(hit["_score"] - 1.0, 4),
                    distance=round(2.0 - hit["_score"], 4),
                    metadata=metadata
                ))

            processing_time = (time.time() - start_time) * 1000

            return SearchResponse(
                results=results,
                query=request.query,
                count=len(results),
                filters_applied=request.filters,
                processing_time_ms=round(processing_time, 2)
            )

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get full document details by ID"""
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {"term": {"doc_id": doc_id}},
                    "size": 1
                }
            )

            if response['hits']['total']['value'] > 0:
                source = response['hits']['hits'][0]['_source']
                return self._extract_metadata(source)

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def get_filter_options(self) -> FilterOptions:
        """Get available filter options from the index"""
        try:
            # Get aggregations to find available filter values
            aggs = {
                "languages": {"terms": {"field": "language.keyword", "size": 100}},
                "institutions": {"terms": {"field": "institution", "size": 100}},
                "libraries": {"terms": {"field": "library", "size": 100}},
                "collections": {"terms": {"field": "collection", "size": 100}},
                "collection_types": {"terms": {"field": "collection_type", "size": 100}},
                "content_types": {"terms": {"field": "content_type", "size": 100}}
            }

            response = self.es.search(
                index=self.index_name,
                body={"size": 0, "aggs": aggs}
            )

            return FilterOptions(
                languages=[bucket["key"] for bucket in response["aggregations"]["languages"]["buckets"]],
                periods=['early_medieval', 'late_medieval', 'early_modern'],  # Static for now
                document_types=[bucket["key"] for bucket in response["aggregations"]["content_types"]["buckets"]],
                institutions=[bucket["key"] for bucket in response["aggregations"]["institutions"]["buckets"]],
                collections=[bucket["key"] for bucket in response["aggregations"]["collections"]["buckets"]]
            )
        except Exception as e:
            logger.warning(f"Could not get filter options: {e}")
            # Return defaults based on your sample data
            return FilterOptions(
                languages=['Hebrew', 'Judaeo-Arabic', 'Arabic', 'Judaeo-Arabic; Hebrew'],
                periods=['early_medieval', 'late_medieval', 'early_modern'],
                document_types=['multimodal'],
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
                "backend": "elasticsearch"
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
            detail=e.message)