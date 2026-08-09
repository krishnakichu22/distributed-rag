from __future__ import annotations
from typing import Iterable
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app import config


class VectorStore:
    """Thin wrapper around qdrant-client for our specific use case."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
        dim: int | None = None,
    ) -> None:
        self.collection = collection or config.QDRANT_COLLECTION
        self.dim = dim or config.EMBEDDING_DIM
        self.client = QdrantClient(
            host=host or config.QDRANT_HOST,
            port=port or config.QDRANT_PORT,
        )

    # ── Collection setup ───────────────────────────────────────────
    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if it doesn't already exist.
        Pass recreate=True to wipe and start fresh — useful for testing.
        """
        exists = self.client.collection_exists(self.collection)

        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✓ Created Qdrant collection '{self.collection}' (dim={self.dim}, cosine)")
        else:
            print(f"✓ Using existing Qdrant collection '{self.collection}'")

    # ── Writing ────────────────────────────────────────────────────
    def upsert(
        self,
        texts: list[str],
        vectors: np.ndarray,
        source: str = "unknown",
    ) -> int:
        """
        Store text chunks + their vectors. Returns the number of points written.

        Each point has:
          id      — a unique uuid
          vector  — the embedding
          payload — the original text + source filename (so we can return text
                    later, since Qdrant only stores vectors+payload, not docs)
        """
        if len(texts) != len(vectors):
            raise ValueError(f"texts ({len(texts)}) and vectors ({len(vectors)}) length mismatch")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"text": text, "source": source},
            )
            for text, vector in zip(texts, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    # ── Reading ────────────────────────────────────────────────────
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Find the top_k chunks most similar to the query vector.
        Returns: list of {"text": str, "source": str, "score": float}
        """
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", "unknown"),
                "score": float(point.score),
            }
            for point in result.points
        ]

    # ── Diagnostics ────────────────────────────────────────────────
    def count(self) -> int:
        """How many points are currently in the collection."""
        return self.client.count(self.collection, exact=True).count
