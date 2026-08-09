from __future__ import annotations
import json

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app import config

JOB_STATUS_TTL_SECONDS = 24 * 60 * 60  # keep job status around for a day


class JobQueue:
    """Wraps Redis Streams (the queue) + plain Redis keys (job status lookups)."""

    def __init__(
        self,
        redis_client: Redis,
        stream_name: str | None = None,
        group_name: str | None = None,
    ) -> None:
        self.redis = redis_client
        self.stream = stream_name or config.INGESTION_STREAM
        self.group = group_name or config.INGESTION_GROUP

    async def ensure_group(self) -> None:
        """
        Create the consumer group if it doesn't exist yet. mkstream=True
        also creates the stream itself on first run, so no separate setup
        step is needed. Safe to call every startup — BUSYGROUP just means
        "already exists," which we swallow.
        """
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # ── Producer side (called from main.py's /ingest handler) ─────────
    async def enqueue(self, job_id: str, filename: str, filepath: str) -> None:
        await self.redis.xadd(
            self.stream,
            {"job_id": job_id, "filename": filename, "filepath": filepath},
        )
        await self.set_status(job_id, "pending", filename=filename)

    # ── Consumer side (called from worker.py's main loop) ──────────────
    async def read_next(self, consumer_name: str, block_ms: int = 5000):
        """
        Block for up to block_ms waiting for one unclaimed job.
        Returns (message_id, fields_dict) or None if nothing arrived in time.
        """
        result = await self.redis.xreadgroup(
            groupname=self.group,
            consumername=consumer_name,
            streams={self.stream: ">"},
            count=1,
            block=block_ms,
        )
        if not result:
            return None
        _stream_name, messages = result[0]
        if not messages:
            return None
        message_id, fields = messages[0]
        return message_id, fields

    async def ack(self, message_id: str) -> None:
        await self.redis.xack(self.stream, self.group, message_id)

    # ── Job status (polled by GET /jobs/{job_id}) ──────────────────────
    async def set_status(self, job_id: str, status: str, **extra) -> None:
        payload = {"job_id": job_id, "status": status, **extra}
        await self.redis.set(f"job:{job_id}", json.dumps(payload), ex=JOB_STATUS_TTL_SECONDS)

    async def get_status(self, job_id: str) -> dict | None:
        raw = await self.redis.get(f"job:{job_id}")
        return json.loads(raw) if raw else None
