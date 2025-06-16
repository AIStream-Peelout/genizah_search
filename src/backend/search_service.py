import os
from rate_limits import ProtectionService, RateLimitExceeded, FilterOptions
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
import logging
import time
from fastapi import HTTPException, Request, status
from temp import NomicsEmbedding
logger = logging.getLogger(__name__)

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results")


class SearchResult(BaseModel):
    doc_id: str
    similarity_score: float
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    count: int
    filters_applied: Optional[Dict[str, Any]] = None
    processing_time_ms: float

class SearchService:
    """Cairo Genizah search service"""

    def __init__(self):
        self.embedding_model = None
        self.index_endpoint = None
        self.deployed_index_id = os.getenv('DEPLOYED_INDEX_ID')
        self._initialize_vertex_ai()

    def _initialize_vertex_ai(self):
        """Initialize Vertex AI and embedding model"""
        try:
            # Initialize Vertex AI
            project_id = os.getenv('GCP_PROJECT_ID')
            region = os.getenv('GCP_REGION', 'us-central1')

            if not project_id:
                raise ValueError("GCP_PROJECT_ID environment variable required")

            aiplatform.init(project=project_id, location=region)

            # Initialize embedding model
            self.embedding_model = NomicsEmbedding()

            # Get index endpoint
            endpoint_name = os.getenv('VERTEX_ENDPOINT_NAME')
            if endpoint_name:
                self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=endpoint_name
                )

            logger.info("Search service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            raise

    def _build_restricts(self, filters: Optional[Dict[str, Any]]) -> Optional[List[Dict]]:
        """Convert filters to Vertex AI restricts"""
        if not filters:
            return None

        restricts = []

        # Map filter keys to namespaces
        filter_mappings = {
            'language': 'language',
            'period': 'period',
            'document_type': 'document_type',
            'institution': 'institution',
            'collection': 'collection',
            'has_transcriptions': 'has_transcriptions',
            'has_translations': 'has_translations',
            'transcription_completeness': 'transcription_completeness'
        }

        for filter_key, namespace in filter_mappings.items():
            if filter_key in filters and filters[filter_key]:
                value = filters[filter_key]
                allow_values = [value] if isinstance(value, str) else value
                restricts.append({
                    "namespace": namespace,
                    "allow": allow_values
                })

        return restricts if restricts else None

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform search with timing"""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.get_embeddings(
                None, request.query, use_cache=False
            )

            # Build restricts
            restricts = self._build_restricts(request.filters)

            # Perform search
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding.flatten().tolist()],
                num_neighbors=request.num_results,
                restricts=restricts
            )

            # Format results
            results = []
            for neighbor_list in response:
                for neighbor in neighbor_list:
                    results.append(SearchResult(
                        doc_id=neighbor.datapoint.datapoint_id,
                        similarity_score=round(1 - neighbor.distance, 4),
                        distance=round(neighbor.distance, 4),
                        metadata={}  # Could extract from restricts if needed
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
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    def get_filter_options(self) -> FilterOptions:
        """Get available filter options"""
        return FilterOptions(
            languages=['Hebrew', 'Arabic', 'Judeo-Arabic', 'Aramaic'],
            periods=['early_medieval', 'late_medieval', 'early_modern'],
            document_types=['legal', 'religious', 'marriage', 'business', 'letter', 'prayer'],
            institutions=['cambridge', 'princeton', 'oxford', 'british_library'],
            collections=['taylor_schechter', 'oriental', 'additional']
        )


# Global services
protection_service = ProtectionService()
search_service = SearchService()


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

