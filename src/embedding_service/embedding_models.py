# embedding_models.py
"""Pinned single-vector text embedding model for Genizah semantic search.

The Elasticsearch indexes (``genizah_merged_v4``, ``bibliography_text_only_0.7``)
were embedded with exactly the configuration in this module (contract of
2026-07-24; each index carries the canary in its mapping ``_meta``).
Every value here — model id, revision, sequence length, normalization, and the
asymmetric query instruction — is part of that contract. Changing any of them
disconnects query vectors from the stored document vectors, which is the
silent-drift failure mode that broke the previous (colnomic mean-pooled)
embeddings. The revision is pinned so a container rebuild can never silently
change the embedder again; consumers verify compatibility at startup against
the canary vector stored in each index's mapping ``_meta``.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Embedding contract — must match the index-build configuration exactly.
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_SEQ_LENGTH = 8192
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages\nQuery: "
EMBEDDING_DIMS = 1024
# Long sequences batched together blow up attention memory; keep batches tiny.
ENCODE_BATCH_SIZE = 2

EmbeddingMode = Literal["query", "document"]


class Qwen3Embedding:
    """Qwen3-Embedding-0.6B wrapper with asymmetric query/document encoding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_MODEL_REVISION,
    ) -> None:
        """Load the pinned embedding model onto the best available device.

        :param model_name: Hugging Face model identifier.
        :param revision: Exact model commit hash; pinned so rebuilds cannot drift.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.device = self._get_device()
        self.cache_dir = Path("embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)

        logger.info(
            "Loading %s@%s on %s (max_seq_length=%s)",
            model_name, revision[:12], self.device, MAX_SEQ_LENGTH,
        )
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device,
        )
        # NOT the model's 32k default — document vectors were built at 8192.
        self.model.max_seq_length = MAX_SEQ_LENGTH

    @staticmethod
    def _get_device() -> str:
        """Return the best available torch device string.

        :returns: ``cuda:0``, ``mps``, or ``cpu``.
        :rtype: str
        """
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def cache_path(self, mode: EmbeddingMode, text: str) -> Path:
        """Build a cache path keyed by model, revision, mode, and text.

        Including revision and mode in the key guarantees stale entries from
        any previous embedder or convention can never be served.

        :param mode: ``query`` or ``document`` encoding mode.
        :param text: Text whose embedding is cached.
        :returns: Pickle cache file path.
        :rtype: Path
        """
        key = f"{self.model_name}@{self.revision}|{mode}|{text}"
        return self.cache_dir / f"emb_{hashlib.md5(key.encode()).hexdigest()}.pkl"

    def load_cached(self, path: Path) -> np.ndarray:
        """Load a cached embedding vector.

        :param path: Cache file path from :meth:`cache_path`.
        :returns: The cached embedding vector.
        :rtype: np.ndarray
        """
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def save_cached(self, path: Path, embedding: np.ndarray) -> None:
        """Persist an embedding vector to the cache.

        :param path: Cache file path from :meth:`cache_path`.
        :param embedding: Embedding vector to store.
        """
        with open(path, "wb") as handle:
            pickle.dump(embedding, handle)

    def embed(self, texts: List[str], mode: EmbeddingMode) -> np.ndarray:
        """Encode texts as L2-normalized 1024-dim vectors.

        Queries receive the retrieval instruction prefix; documents are encoded
        raw. Mixing the modes breaks compatibility with the stored vectors.

        :param texts: Texts to encode.
        :param mode: ``query`` (instruction-prefixed) or ``document`` (raw).
        :returns: Array of shape ``(len(texts), 1024)``.
        :rtype: np.ndarray
        :raises ValueError: If ``mode`` is not a known encoding mode.
        """
        if mode not in ("query", "document"):
            raise ValueError(f"Unknown embedding mode: {mode!r}")
        prompt = QUERY_INSTRUCTION if mode == "query" else None
        embeddings = self.model.encode(
            texts,
            prompt=prompt,
            normalize_embeddings=True,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
