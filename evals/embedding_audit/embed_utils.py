"""Shared helpers for the embedding audit probes: model registry + cached encoding.

All caches live on the NAS (``AUDIT_ROOT``) so the Studio's internal disk is
untouched. Encodings are cached per (model, mode, instruction, text) so probes
are resumable after the guard kills them.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

AUDIT_ROOT = Path(os.environ.get("AUDIT_ROOT", "/Volumes/home/studio_offload/genizah_search_embedding_audit"))
CACHE_DIR = AUDIT_ROOT / "cache"

DEFAULT_TASK = "Given a search query, retrieve relevant passages"
DOMAIN_TASK = (
    "Given a question about Jewish law, liturgy, festivals or history, retrieve Cairo Genizah "
    "fragment descriptions, transcriptions and scholarship that discuss the same topic"
)

# key -> (hf id, revision, instruction style). "qwen"/"e5" take "Instruct: ...\nQuery: "; "none" takes raw text.
TEXT_MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "qwen3-0.6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "style": "qwen"},
    "qwen3-4b": {"id": "Qwen/Qwen3-Embedding-4B", "revision": None, "style": "qwen"},
    "bge-m3": {"id": "BAAI/bge-m3", "revision": None, "style": "none"},
    "e5-large-instruct": {"id": "intfloat/multilingual-e5-large-instruct", "revision": None, "style": "qwen"},
}


def device() -> str:
    """Return the torch device to use (MPS when available; ``AUDIT_DEVICE`` overrides).

    CPU runs (``AUDIT_DEVICE=cpu``, optionally ``AUDIT_THREADS``) let probes proceed while the
    sibling's GPU-bound evals hold the shared GPU.

    :returns: ``mps`` or ``cpu``.
    :rtype: str
    """
    import torch

    forced = os.environ.get("AUDIT_DEVICE")
    if forced:
        if forced == "cpu" and os.environ.get("AUDIT_THREADS"):
            torch.set_num_threads(int(os.environ["AUDIT_THREADS"]))
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CachedTextEncoder:
    """Sentence-transformers encoder with an on-disk per-text cache."""

    def __init__(self, key: str, max_seq_length: int = 512, local_path: Optional[str] = None) -> None:
        """Load a registered model lazily.

        :param key: Key into :data:`TEXT_MODELS` (also used as the cache namespace).
        :param max_seq_length: Truncation length for documents/queries.
        :param local_path: Optional local checkpoint path overriding the hub id (fine-tuned models).
        """
        self.key = key
        self.spec = TEXT_MODELS.get(key, {"id": local_path, "revision": None, "style": "qwen"})
        self.local_path = local_path
        self.max_seq_length = max_seq_length
        self._model = None
        self.cache_dir = CACHE_DIR / "text" / key
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """Return the loaded SentenceTransformer, loading it on first use.

        :returns: The model.
        """
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": device()}
            if self.spec.get("revision"):
                kwargs["revision"] = self.spec["revision"]
            if self.key in ("qwen3-4b",):
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            self._model = SentenceTransformer(self.local_path or self.spec["id"], **kwargs)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def prefix(self, mode: str, task: Optional[str]) -> str:
        """Build the text prefix for a mode/instruction combination.

        :param mode: ``query`` or ``document``.
        :param task: Instruction text, or None for no instruction.
        :returns: Prefix string (empty for documents or instruction-free models).
        :rtype: str
        """
        if mode == "document" or task is None or self.spec["style"] == "none":
            return ""
        return f"Instruct: {task}\nQuery: "

    def _path(self, text: str, prefix: str) -> Path:
        """Return the cache file for one encoded text.

        :param text: Raw text.
        :param prefix: Prefix applied before encoding.
        :returns: Cache path.
        :rtype: Path
        """
        digest = hashlib.sha1(f"{self.max_seq_length}|{prefix}|{text}".encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.npy"

    def encode(self, texts: List[str], mode: str, task: Optional[str] = DEFAULT_TASK, batch_size: int = 8) -> np.ndarray:
        """Encode texts (L2-normalized), reading/writing the cache.

        :param texts: Texts to encode.
        :param mode: ``query`` or ``document``.
        :param task: Query instruction (ignored for documents).
        :param batch_size: Encode batch size.
        :returns: Array (n, dim) of normalized vectors.
        :rtype: np.ndarray
        """
        prefix = self.prefix(mode, task)
        paths = [self._path(t, prefix) for t in texts]
        missing = [i for i, p in enumerate(paths) if not p.exists()]
        for start in range(0, len(missing), 256):
            chunk = missing[start:start + 256]
            vecs = self.model.encode(
                [prefix + texts[i] for i in chunk], normalize_embeddings=True,
                batch_size=batch_size, convert_to_numpy=True,
            )
            for i, vec in zip(chunk, vecs):
                paths[i].parent.mkdir(exist_ok=True)
                np.save(paths[i], vec.astype(np.float32))
        return np.stack([np.load(p) for p in paths])
