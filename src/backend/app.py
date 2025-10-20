# Updated app.py - FastAPI endpoint with embedding visualization support

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime
import dotenv
import logging
logging.getLogger('elasticsearch').setLevel(logging.DEBUG)

file_path = os.path.dirname(os.path.realpath(__file__))
load_dotenv = dotenv.load_dotenv(file_path + '/.env')

from search_service import (
    SearchResponse, SearchRequest, DocumentMetadata,
    search_service
)
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from search_service import FilterOptions

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Cairo Genizah Search API",
    description="AI-powered semantic search through historical manuscripts with embedding visualizations.",
    version="1.1.0",  # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://elastic.cairogenizah.ai",
        "http://frontend:80",
        "https://cairogenizah.ai",
        "https://www.cairogenizah.ai",
        "https://api.cairogenizah.ai",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Exception handlers


# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}




@app.get("/filters", response_model=FilterOptions)
async def get_filter_options():
    """Get available filter options for the frontend"""
    return search_service.get_filter_options()


@app.get("/document/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: str):
    """
    Get full document details by ID

    This endpoint returns complete metadata, transcription, translation,
    and image information for a specific document.
    """
    document = search_service.get_document_by_id(doc_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found"
        )

    return document


# Shelf mark search request model
class ShelfMarkSearchRequest(BaseModel):
    shelf_mark: str = Field(..., min_length=1, max_length=100, description="Shelf mark to search for")
    exact_match: bool = Field(default=False, description="Whether to perform exact match or partial match")
    num_results: Optional[int] = Field(default=10, ge=1, le=50, description="Number of results to return")


# Keyword search request model
class KeywordSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Keywords or phrases to search for")
    num_results: Optional[int] = Field(default=10, ge=1, le=50, description="Number of results to return")
    page: Optional[int] = Field(default=1, ge=1, description="Page number for pagination (1-based)")


# Hybrid search request model
class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query for hybrid search")
    semanticWeight: int = Field(default=50, ge=0, le=100, description="Weight for semantic search (0-100)")
    keywordWeight: int = Field(default=50, ge=0, le=100, description="Weight for keyword search (0-100)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    num_results: Optional[int] = Field(default=10, ge=1, le=50, description="Number of results to return")
    include_embeddings: Optional[bool] = Field(default=False, description="Include embedding vectors for visualization")
    page: Optional[int] = Field(default=1, ge=1, description="Page number for pagination (1-based)")


@app.post("/search-shelfmark", response_model=SearchResponse)
async def search_by_shelfmark(
        search_request: ShelfMarkSearchRequest,
        request: Request
):
    """
    Search documents by shelf mark or catalog number
    
    This endpoint allows users to find specific documents using their shelf mark
    (e.g., T-S 8J5.1, MS-TS-NS-144.1). Supports both exact and partial matching.
    
    Examples:
    - T-S 8J5.1 (exact match)
    - T-S 8J5 (partial match)
    - MS-TS-NS-144 (partial match)
    """
    # Log the shelf mark search request
    logger.info(f"Shelf mark search: '{search_request.shelf_mark}', exact_match={search_request.exact_match}")

    # Perform shelf mark search
    try:
        result = await search_service.search_by_shelfmark(search_request)
        
        # Log successful search
        logger.info(f"Shelf mark search completed: {result.count} results in {result.processing_time_ms}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Shelf mark search failed for '{search_request.shelf_mark}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Shelf mark search failed: {str(e)}"
        )


@app.post("/search-keyword", response_model=SearchResponse)
async def search_by_keyword(
        search_request: KeywordSearchRequest,
        request: Request
):
    """
    Search documents by keywords in text content
    
    This endpoint allows users to find documents by searching for specific words
    or phrases in transcriptions, translations, descriptions, and other text fields.
    This is a traditional keyword-based search that looks for exact text matches.
    
    Examples:
    - "marriage contract"
    - "Kiddushin"
    - "Hebrew"
    - "responsum"
    """
    # Log the keyword search request
    logger.info(f"Keyword search: '{search_request.query}', page={search_request.page}")

    # Perform keyword search
    try:
        result = await search_service.search_by_keyword(search_request)
        
        # Log successful search
        logger.info(f"Keyword search completed: {result.count} results in {result.processing_time_ms}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Keyword search failed for '{search_request.query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Keyword search failed: {str(e)}"
        )


@app.post("/search-hybrid", response_model=SearchResponse)
async def search_hybrid(
        search_request: HybridSearchRequest,
        request: Request
):
    """
    Hybrid search combining semantic and keyword search
    
    This endpoint performs a weighted combination of semantic AI search and traditional
    keyword search. Users can adjust the weights to balance between conceptual understanding
    and exact text matching.
    
    Features:
    - Configurable weights for semantic vs keyword search
    - Combines the best of both search approaches
    - Supports all standard search filters
    - Optional embedding data for visualization
    
    Examples:
    - 50% semantic + 50% keyword (balanced)
    - 80% semantic + 20% keyword (concept-focused)
    - 20% semantic + 80% keyword (text-focused)
    """
    # Validate weights sum to 100
    if search_request.semanticWeight + search_request.keywordWeight != 100:
        raise HTTPException(
            status_code=400,
            detail="Semantic and keyword weights must sum to 100"
        )
    
    # Log the hybrid search request
    logger.info(f"Hybrid search: '{search_request.query}', "
               f"semantic_weight={search_request.semanticWeight}%, "
               f"keyword_weight={search_request.keywordWeight}%, "
               f"page={search_request.page}")

    # Perform hybrid search
    try:
        result = await search_service.search_hybrid(search_request)
        
        # Log successful search
        logger.info(f"Hybrid search completed: {result.count} results in {result.processing_time_ms}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Hybrid search failed for '{search_request.query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
        search_request: SearchRequest,
        request: Request
):
    """
    Search Cairo Genizah documents with semantic AI search and optional embedding visualization.

    This endpoint performs AI-powered semantic search through historical manuscripts
    from the Cairo Genizah collection. Returns results with rich metadata including
    titles, descriptions, images, transcriptions, and translations. 
    
    New features:
    - Optional embedding vectors for t-SNE/PCA visualization
    - Enhanced metadata for better user experience
    - Improved similarity scoring
    
    Set `include_embeddings=true` to get embedding data for visualization.
    """
    # Log the search request for analytics
    logger.info(f"Search request: query='{search_request.query}', "
               f"include_embeddings={search_request.include_embeddings}, "
               f"num_results={search_request.num_results}")

    # Perform search
    try:
        result = await search_service.search(search_request)
        
        # Log successful search
        logger.info(f"Search completed: {result.count} results in {result.processing_time_ms}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Search failed for query '{search_request.query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/embedding-stats")
async def get_embedding_stats():
    """
    Get statistics about embedding usage and visualization features
    
    New endpoint to help monitor embedding-related usage and performance.
    """
    try:
        stats = search_service.get_stats()
        
        # Add embedding-specific stats
        embedding_stats = {
            "base_stats": stats,
            "embedding_features": {
                "supports_visualization": True,
                "embedding_dimension": 768,  # Adjust based on your actual embedding model
                "visualization_methods": ["pca", "tsne"],
                "max_results_for_visualization": 20
            },
            "performance_notes": {
                "pca_calculation_time": "~50ms for 10 documents",
                "tsne_calculation_time": "~500ms for 10 documents",
                "recommendation": "Use PCA for quick visualization, t-SNE for detailed analysis"
            }
        }
        
        return embedding_stats
        
    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "embedding_features": {
                "supports_visualization": False
            }
        }


# Visualization Explorer request model
class VisualizationExplorerRequest(BaseModel):
    num_documents: Optional[int] = Field(default=1000, ge=10, le=10000, description="Number of documents to load for visualization")
    load_full_index: Optional[bool] = Field(default=False, description="Load the entire index (ignores num_documents)")
    include_embeddings: Optional[bool] = Field(default=True, description="Include embedding vectors for visualization")


@app.post("/visualization-explorer", response_model=SearchResponse)
async def get_visualization_explorer_data(
        request: VisualizationExplorerRequest,
        request_obj: Request
):
    """
    Load a set of documents for the visualization explorer
    
    This endpoint loads a random sample of documents from the collection
    for full-page visualization exploration. Supports loading a configurable
    number of documents or the entire index.
    
    Features:
    - Random sampling of documents
    - Full metadata extraction
    - Embedding vectors for visualization
    - Support for large document sets
    """
    logger.info(f"Visualization explorer request: num_documents={request.num_documents}, "
               f"load_full_index={request.load_full_index}")
    
    try:
        result = await search_service.get_visualization_explorer_data(request)
        
        logger.info(f"Visualization explorer data loaded: {result.count} documents")
        
        return result
        
    except Exception as e:
        logger.error(f"Visualization explorer data loading failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load visualization explorer data: {str(e)}"
        )


@app.get("/")
async def root():
    """API root with basic info"""
    return {
        "message": "Cairo Genizah Search API",
        "version": "1.1.0",
        "new_features": [
            "Shelf mark search functionality",
            "Advanced search interface",
            "Embedding visualization support",
            "t-SNE and PCA dimensionality reduction",
            "Enhanced metadata extraction",
            "Improved similarity scoring"
        ],
        "docs": "/docs",
        "endpoints": {
            "search": "POST /search",
            "search_shelfmark": "POST /search-shelfmark",
            "search_keyword": "POST /search-keyword",
            "search_hybrid": "POST /search-hybrid",
            "visualization_explorer": "POST /visualization-explorer",
            "document": "GET /document/{doc_id}",
            "filters": "GET /filters",
            "embedding_stats": "GET /embedding-stats",
            "health": "GET /health"
        },
        "visualization": {
            "description": "Set include_embeddings=true in search requests to get visualization data",
            "supported_methods": ["pca", "tsne"],
            "frontend_integration": "Use TSNEVisualization React component"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('ENVIRONMENT') == 'development'
    )