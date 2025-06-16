from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import hashlib
import logging
import os
from datetime import datetime
import dotenv
file_path = os.path.dirname(os.path.realpath(__file__))
load_dotenv = dotenv.load_dotenv(file_path + '/.env')

from search_service import (SearchService, SearchResponse, SearchRequest, protection_service,
                                        search_service, check_rate_limits)
from rate_limits import RateLimitExceeded, UsageStats, FilterOptions
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# FastAPI app
app = FastAPI(
    title="Cairo Genizah Search API",
    description="AI-powered semantic search through historical manuscripts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


@app.post("/search", response_model=SearchResponse)
async def search_documents(
        search_request: SearchRequest,
        request: Request,
        _: None = Depends(check_rate_limits)
):
    """
    Search Cairo Genizah documents with semantic AI search

    This endpoint performs AI-powered semantic search through historical manuscripts
    from the Cairo Genizah collection. Rate limits apply.
    """
    # Record the query for billing/monitoring
    await protection_service.record_query(request)

    # Perform search
    return await search_service.search(search_request)


@app.get("/")
async def root():
    """API root with basic info"""
    return {
        "message": "Cairo Genizah Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "search": "POST /search",
            "stats": "GET /stats",
            "filters": "GET /filters",
            "health": "GET /health"
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