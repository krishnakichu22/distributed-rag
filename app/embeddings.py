from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


class Embedder:
    """Wraps a local sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        # download/load is lazy — happens on first .encode() call
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings → matrix shape (len(texts), EMBEDDING_DIM)."""
        model = self._ensure_loaded()
        return model.encode(texts, show_progress_bar=False)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience for a single string → 1-D vector."""
        return self.embed([text])[0]
