"""
nexus/epistemic/embeddings.py
------------------------------
Local embedding engine using sentence-transformers.
Runs entirely on your Mac M1 via MPS (Metal Performance Shaders).
No API calls, no cost, no rate limits.

Model: all-MiniLM-L6-v2  (384-dim, ~80MB, very fast on M1)
  - Ideal for semantic similarity on infrastructure text
  - Qdrant collection vector_size must match: 384

Usage:
    embedder = get_embedder()
    vec = embedder.embed("SELECT * FROM users WHERE id = 42")
    vecs = embedder.embed_batch(["log line 1", "log line 2"])
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Union

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from nexus.config.settings import get_settings

log = structlog.get_logger(__name__)

# Thread lock — SentenceTransformer is not thread-safe during model load
_load_lock = threading.Lock()


class LocalEmbedder:
    """
    Wraps SentenceTransformer for consistent embedding across Nexus.

    Key design decisions:
    - Model loaded once, cached as singleton
    - M1 MPS acceleration used automatically when available
    - Returns plain Python lists (JSON-serialisable) for Qdrant
    - Normalises vectors to unit length (cosine similarity works correctly)
    """

    def __init__(self, model_name: str | None = None) -> None:
        cfg = get_settings()
        self.model_name = model_name or cfg.ollama.embedding_model
        self.vector_size: int = 384  # all-MiniLM-L6-v2 output dim

        log.info("Loading embedding model", model=self.model_name)
        with _load_lock:
            self._model = SentenceTransformer(self.model_name)

        # Try MPS (M1 Metal) acceleration
        try:
            import torch
            if torch.backends.mps.is_available():
                self._model = self._model.to("mps")
                log.info("Embedder using MPS (M1 Metal) acceleration")
            else:
                log.info("Embedder using CPU")
        except Exception:
            log.info("Embedder using CPU (torch MPS check failed)")

        log.info(
            "Embedding model ready",
            model=self.model_name,
            vector_size=self.vector_size,
        )

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string into a normalised 384-dim vector.
        Returns a plain Python list (not numpy) for JSON compatibility.
        """
        if not text or not text.strip():
            # Zero vector for empty inputs — avoids NaN in similarity search
            return [0.0] * self.vector_size

        vec: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=True,  # unit-length for cosine similarity
            show_progress_bar=False,
        )
        return vec.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """
        Embed a list of strings efficiently in batches.
        Much faster than calling embed() in a loop.

        Args:
            texts: List of strings to embed.
            batch_size: How many to process at once (tune for M1 RAM).

        Returns:
            List of 384-dim vectors, one per input text.
        """
        if not texts:
            return []

        # Replace empty strings with a placeholder so indices stay aligned
        cleaned = [t if t and t.strip() else "<empty>" for t in texts]

        vecs: np.ndarray = self._model.encode(
            cleaned,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,  # progress bar only for large batches
        )
        return vecs.tolist()

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity between two pre-computed vectors.
        Returns float in [-1, 1]; normalised vectors give [0, 1].
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Module-level singleton — import this, don't instantiate LocalEmbedder directly
# ---------------------------------------------------------------------------

_embedder_instance: LocalEmbedder | None = None
_embedder_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """
    Returns the global LocalEmbedder singleton.
    Thread-safe. Model loads only once per process.

    Usage anywhere in Nexus:
        from nexus.epistemic.embeddings import get_embedder
        vec = get_embedder().embed("some log line")
    """
    return LocalEmbedder()
