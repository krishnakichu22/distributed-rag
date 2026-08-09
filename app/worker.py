from __future__ import annotations
import asyncio
import os
import socket
import sys

from redis.asyncio import Redis

from app import config
from app.chunking import load_document, chunk_text
from app.embeddings import Embedder
from app.queue import JobQueue
from app.vectorstore import VectorStore

CONSUMER_NAME = f"worker-{socket.gethostname()}-{os.getpid()}"


async def process_job(
    queue: JobQueue,
    store: VectorStore,
    embedder: Embedder,
    job_id: str,
    filename: str,
    filepath: str,
) -> None:
    await queue.set_status(job_id, "processing", filename=filename)
    try:
        text = load_document(filepath)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        vectors = embedder.embed(chunks)
        n = store.upsert(chunks, vectors, source=filename)
        await queue.set_status(job_id, "done", filename=filename, chunks_ingested=n)
        print(f"[worker] job {job_id} done -- {n} chunks from {filename}")
    except Exception as exc:  # noqa: BLE001 - a job-specific failure must not kill the worker loop
        await queue.set_status(job_id, "error", filename=filename, error=str(exc))
        print(f"[worker] job {job_id} FAILED -- {exc}")


async def main() -> None:
    print(f"[worker] starting as consumer '{CONSUMER_NAME}'")

    redis_client = Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    queue = JobQueue(redis_client)
    await queue.ensure_group()

    store = VectorStore()
    store.ensure_collection()
    embedder = Embedder()  # model loads lazily on the first .embed() call

    print("[worker] waiting for ingestion jobs... (Ctrl+C to stop)")
    while True:
        result = await queue.read_next(CONSUMER_NAME, block_ms=5000)
        if result is None:
            continue  # nothing arrived within the block window -- loop so Ctrl+C is responsive
        message_id, fields = result
        await process_job(
            queue,
            store,
            embedder,
            job_id=fields["job_id"],
            filename=fields["filename"],
            filepath=fields["filepath"],
        )
        await queue.ack(message_id)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[worker] shutting down.")
