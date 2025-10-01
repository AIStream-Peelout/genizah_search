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
    SearchResponse, SearchRequest, DocumentMetadata, protection_service,
    search_service, check_rate_limits
)
from rate_limits import RateLimitExceeded, UsageStats, FilterOptions

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Cairo Genizah Search API",
    description="AI-powered semantic search through historical manuscripts with embedding visualization",
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
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": exc.message, "type": "rate_limit"}
    )


# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/stats", response_model=UsageStats)
async def get_usage_stats(request: Request):
    """Get current usage statistics"""
    return await protection_service.get_usage_stats(request)


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


@app.post("/search", response_model=SearchResponse)
async def search_documents(
        search_request: SearchRequest,
        request: Request,
        _: None = Depends(check_rate_limits)
):
    """
    Search Cairo Genizah documents with semantic AI search and optional embedding visualization

    This endpoint performs AI-powered semantic search through historical manuscripts
    from the Cairo Genizah collection. Returns results with rich metadata including
    titles, descriptions, images, transcriptions, and translations. 
    
    New features:
    - Optional embedding vectors for t-SNE/PCA visualization
    - Enhanced metadata for better user experience
    - Improved similarity scoring
    
    Rate limits apply. Set `include_embeddings=true` to get embedding data for visualization.
    """
    # Record the query for billing/monitoring
    await protection_service.record_query(request)

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


@app.get("/")
async def root():
    """API root with basic info"""
    return {
        "message": "Cairo Genizah Search API",
        "version": "1.1.0",
        "new_features": [
            "Embedding visualization support",
            "t-SNE and PCA dimensionality reduction",
            "Enhanced metadata extraction",
            "Improved similarity scoring"
        ],
        "docs": "/docs",
        "endpoints": {
            "search": "POST /search",
            "document": "GET /document/{doc_id}",
            "stats": "GET /stats",
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