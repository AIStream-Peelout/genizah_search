"""
FastAPI microservice for query/document text embedding generation.

Serves the pinned Qwen3-Embedding-0.6B model that produced the vectors stored
in the current Elasticsearch indexes. See embedding_models.py for the full
embedding contract; /health reports the exact model, revision, and dims so
consumers can verify compatibility.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embedding_models import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EMBEDDING_DIMS,
    MAX_SEQ_LENGTH,
    QUERY_INSTRUCTION,
    Qwen3Embedding,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model: Optional[Qwen3Embedding] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the embedding model.

    :param app: The FastAPI application instance.
    """
    global embedding_model
    model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    revision = os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_MODEL_REVISION)
    logger.info(f"Initializing embedding model: {model_name}@{revision[:12]}")
    embedding_model = Qwen3Embedding(model_name=model_name, revision=revision)
    logger.info("Embedding model initialized successfully")

    yield

    logger.info("Shutting down embedding service")


app = FastAPI(
    title="Embedding Service",
    version="2.0.0",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    text: str = Field(..., min_length=1, description="Text to embed")
    mode: Literal["query", "document"] = Field(
        default="query",
        description=(
            "query: instruction-prefixed search-query embedding (the default; "
            "all retrieval callers embed queries). document: raw-text embedding "
            "matching how index vectors were built — used for canary checks."
        ),
    )
    image: Optional[str] = Field(
        default=None,
        description="Deprecated. The embedding model is text-only; images are ignored.",
    )
    use_cache: bool = Field(default=True, description="Whether to use cached embeddings")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Dimension of the embedding vector")
    cached: bool = Field(default=False, description="Whether the embedding was retrieved from cache")


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""

    texts: List[str] = Field(..., min_items=1, description="List of texts to embed")
    mode: Literal["query", "document"] = Field(default="query")
    images: Optional[List[Optional[str]]] = Field(
        default=None,
        description="Deprecated. The embedding model is text-only; images are ignored.",
    )
    use_cache: bool = Field(default=True, description="Whether to use cached embeddings")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Dimension of the embedding vectors")
    cached_count: int = Field(default=0, description="Number of embeddings retrieved from cache")


@app.get("/health")
async def health_check():
    """Report service health and the exact embedding contract being served.

    :returns: Health status plus model, revision, dims, and query instruction.
    :rtype: dict
    """
    return {
        "status": "healthy",
        "model_loaded": embedding_model is not None,
        "model": embedding_model.model_name if embedding_model else None,
        "revision": embedding_model.revision if embedding_model else None,
        "dimensions": EMBEDDING_DIMS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "query_instruction": QUERY_INSTRUCTION,
    }


def _embed_one(text: str, mode: str, use_cache: bool) -> tuple[List[float], bool]:
    """Embed one text with optional cache lookup.

    :param text: Text to embed.
    :param mode: ``query`` or ``document`` encoding mode.
    :param use_cache: Whether to read/write the on-disk cache.
    :returns: The embedding as a list and whether it came from cache.
    :rtype: tuple[List[float], bool]
    """
    cache_file = embedding_model.cache_path(mode, text)
    if use_cache and cache_file.exists():
        return list(np.asarray(embedding_model.load_cached(cache_file)).flatten()), True
    vector = embedding_model.embed([text], mode=mode)[0]
    if use_cache:
        embedding_model.save_cached(cache_file, vector)
    return list(vector.flatten()), False


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest):
    """Generate an embedding for a single text.

    :param request: Embedding request with text and encoding mode.
    :returns: The embedding vector.
    :rtype: EmbeddingResponse
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if request.image:
        logger.warning("Image supplied to text-only embedding service; ignoring it")

    try:
        embedding_list, cached = _embed_one(request.text, request.mode, request.use_cache)
        return EmbeddingResponse(
            embedding=embedding_list,
            dimension=len(embedding_list),
            cached=cached,
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    """Generate embeddings for multiple texts.

    :param request: Batch embedding request with texts and encoding mode.
    :returns: The embedding vectors.
    :rtype: BatchEmbeddingResponse
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if request.images and any(request.images):
        logger.warning("Images supplied to text-only embedding service; ignoring them")

    try:
        embeddings: List[List[float]] = []
        cached_count = 0
        for text in request.texts:
            embedding_list, cached = _embed_one(text, request.mode, request.use_cache)
            embeddings.append(embedding_list)
            cached_count += int(cached)
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            cached_count=cached_count,
        )
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate batch embeddings: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
