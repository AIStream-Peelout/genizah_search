"""
Client for the embedding service microservice
"""
import os
import logging
from typing import Optional, List
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for interacting with the embedding service"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8001")
        self.timeout = 60.0  # Longer timeout for model inference
        # Set when the startup canary check detects that the live embedder no
        # longer matches the vectors stored in Elasticsearch. While set, query
        # embedding refuses to run: serving semantic search over mismatched
        # vector spaces returns noise and silently poisons every consumer.
        self.drift_reason: Optional[str] = None

    async def _post_embed(self, text: str, mode: str, use_cache: bool) -> np.ndarray:
        """POST one embedding request without consulting the drift gate.

        :param text: Text to embed.
        :param mode: ``query`` or ``document`` encoding mode.
        :param use_cache: Whether the service may serve a cached vector.
        :returns: The embedding vector.
        :rtype: np.ndarray
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"text": text, "mode": mode, "use_cache": use_cache},
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])

    async def get_embedding(
        self,
        text: str,
        image: Optional[str] = None,
        use_cache: bool = True,
        mode: str = "query",
    ) -> np.ndarray:
        """Get an embedding for a single text.

        :param text: Text to embed.
        :param image: Deprecated; the embedding model is text-only and this
            argument is ignored (kept for caller compatibility).
        :param use_cache: Whether to use cached embeddings.
        :param mode: ``query`` (instruction-prefixed, the retrieval default)
            or ``document`` (raw text, matching stored index vectors).
        :returns: The embedding vector.
        :rtype: np.ndarray
        :raises RuntimeError: If embedding drift was detected at startup.
        """
        if self.drift_reason:
            raise RuntimeError(f"Semantic search disabled — embedding drift: {self.drift_reason}")
        try:
            return await self._post_embed(text, mode, use_cache)
        except httpx.HTTPError as e:
            logger.error(f"Failed to get embedding from service: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting embedding: {e}")
            raise

    async def verify_index_canary(self, es_client, index_name: str) -> Optional[str]:
        """Check the live embedder against an index's stored canary vector.

        Each re-embedded index records the embedding contract and a canary
        (string + expected document-mode vector) in its mapping ``_meta``.
        Reproducing the canary vector proves the serving embedder matches the
        one that built the index.

        :param es_client: Synchronous Elasticsearch client for the index.
        :param index_name: Index whose ``_meta`` canary to verify.
        :returns: A problem description, or ``None`` when compatible.
        :rtype: Optional[str]
        """
        mapping = es_client.indices.get_mapping(index=index_name)
        meta = list(mapping.values())[0].get("mappings", {}).get("_meta", {}) or {}
        canary = meta.get("canary") or {}
        canary_string = canary.get("string")
        canary_vector = canary.get("vector")
        if not canary_string or not canary_vector:
            return f"{index_name}: mapping _meta has no canary; cannot verify embedder compatibility"
        fresh = (await self._post_embed(canary_string, mode="document", use_cache=False)).flatten()
        stored = np.asarray(canary_vector, dtype=np.float64).flatten()
        cosine = float(np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
        if cosine <= 0.99:
            return (
                f"{index_name}: canary cosine {cosine:.4f} <= 0.99 — the serving embedder does "
                f"not match the model that built this index (expected "
                f"{meta.get('embedding_model')}@{str(meta.get('embedding_model_revision'))[:12]})"
            )
        logger.info("Embedding canary verified for %s (cosine %.6f)", index_name, cosine)
        return None
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        images: Optional[List[Optional[str]]] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for multiple texts (and optionally images)
        
        Args:
            texts: List of texts to embed
            images: Optional list of images (must match texts length)
            use_cache: Whether to use cached embeddings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "texts": texts,
                    "use_cache": use_cache
                }
                if images is not None:
                    payload["images"] = images
                
                response = await client.post(
                    f"{self.base_url}/embed/batch",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return np.array(data["embeddings"])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get batch embeddings from service: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting batch embeddings: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json().get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return False


# Global client instance
embedding_client = EmbeddingClient()


