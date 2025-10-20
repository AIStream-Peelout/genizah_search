from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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


class FilterOptions(BaseModel):
    languages: List[str]
    periods: List[str]
    document_types: List[str]
    institutions: List[str]
    collections: List[str]
