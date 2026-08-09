from __future__ import annotations
import json
import uuid

import numpy as np
from redis.asyncio import Redis

from app import config


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticCache:
    """Redis-backed cache keyed by embedding similarity rather than exact text."""

    def __init__(
        self,
        redis_client: Redis,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis = redis_client
        self.threshold = threshold if threshold is not None else config.SEMANTIC_CACHE_THRESHOLD
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else config.CACHE_TTL_SECONDS

    async def lookup(self, question_vec: np.ndarray) -> dict | None:
        """
        Scan cached entries for the best match. Returns the cached
        {"answer", "sources", "score"} dict on a hit, else None.
        """
        best_score = -1.0
        best_entry: dict | None = None

        async for key in self.redis.scan_iter(match="cache:*"):
            raw = await self.redis.get(key)
            if raw is None:
                continue  # expired between SCAN and GET — race is harmless, just skip
            entry = json.loads(raw)
            cached_vec = np.array(entry["vector"], dtype=np.float32)
            score = _cosine_similarity(question_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.threshold:
            return {
                "answer": best_entry["answer"],
                "sources": best_entry["sources"],
                "score": best_score,
            }
        return None

    async def store(
        self,
        question: str,
        question_vec: np.ndarray,
        answer: str,
        sources: list[dict],
    ) -> None:
        """Save a question/answer pair. Expires automatically after ttl_seconds."""
        key = f"cache:{uuid.uuid4()}"
        value = json.dumps(
            {
                "question": question,
                "vector": question_vec.tolist(),
                "answer": answer,
                "sources": sources,
            }
        )
        await self.redis.set(key, value, ex=self.ttl_seconds)
