from __future__ import annotations
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from redis.asyncio import Redis

from app import config
from app.cache import SemanticCache
from app.embeddings import Embedder
from app.llm import GeminiLLM, build_grounded_prompt
from app.queue import JobQueue
from app.schemas import IngestResponse, JobStatusResponse, QueryRequest
from app.vectorstore import VectorStore

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, once at shutdown. Everything expensive to
    create (the embedding model, DB clients) is built ONCE here and
    reused for every request via app.state — not recreated per request.
    """
    config.require_gemini_key()
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    app.state.embedder = Embedder()
    app.state.store = VectorStore()
    app.state.store.ensure_collection()
    app.state.llm = GeminiLLM()

    app.state.redis = Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    app.state.cache = SemanticCache(app.state.redis)
    app.state.jobs = JobQueue(app.state.redis)
    await app.state.jobs.ensure_group()

    yield  # the app runs while paused here

    await app.state.redis.aclose()


app = FastAPI(title="Distributed RAG System", lifespan=lifespan)

# Local dev only: the UI is opened straight from the filesystem / a different
# port, so the browser treats it as a different origin. A real deployment
# would restrict this to the UI's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the single-page test UI."""
    return Path("ui/index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "qdrant_points": app.state.store.count(),
        "redis": await app.state.redis.ping(),
    }


# ── Ingestion (producer side of the queue) ──────────────────────────────
@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(file: UploadFile) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}")

    job_id = str(uuid.uuid4())
    # Path(...).name strips any directory components a malicious filename
    # might contain (e.g. "../../etc/passwd") -- never trust a client-supplied path.
    safe_name = Path(file.filename).name
    dest = Path(config.UPLOAD_DIR) / f"{job_id}_{safe_name}"
    dest.write_bytes(await file.read())

    await app.state.jobs.enqueue(job_id, safe_name, str(dest))
    return IngestResponse(job_id=job_id, filename=safe_name, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    status = await app.state.jobs.get_status(job_id)
    if status is None:
        raise HTTPException(404, "No job with that id")
    return JobStatusResponse(**status)


# ── Query (semantic cache -> Qdrant retrieval -> streamed Gemini answer) ─
def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The blank line at the end is required
    by the SSE spec -- it's how the browser knows one event ended."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _answer_stream(question: str):
    embedder: Embedder = app.state.embedder
    store: VectorStore = app.state.store
    cache: SemanticCache = app.state.cache
    llm: GeminiLLM = app.state.llm

    # embed() runs the local model -- CPU-bound, so it's pushed off the
    # event loop rather than blocking every other in-flight request.
    query_vec = await run_in_threadpool(embedder.embed_one, question)

    cached = await cache.lookup(query_vec)
    if cached is not None:
        yield _sse("sources", {"sources": cached["sources"], "cache_score": cached["score"]})
        yield _sse("token", {"text": cached["answer"]})
        yield _sse("done", {"cached": True})
        return

    hits = await run_in_threadpool(store.search, query_vec, config.TOP_K)
    yield _sse("sources", {"sources": hits, "cache_score": None})

    prompt = build_grounded_prompt(question, [h["text"] for h in hits])

    full_answer = ""
    async for piece in llm.astream(prompt):
        full_answer += piece
        yield _sse("token", {"text": piece})

    await cache.store(question, query_vec, full_answer, hits)
    yield _sse("done", {"cached": False})


@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(_answer_stream(req.question), media_type="text/event-stream")
