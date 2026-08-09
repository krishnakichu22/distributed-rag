# Distributed RAG

A distributed Retrieval-Augmented Generation (RAG) architecture built step-by-step from a simple Python implementation to a scalable distributed system using Qdrant, Redis, FastAPI, and Docker.

---

## Repository Structure

```text
distributed-rag/
├── README.md
├── requirements.txt
├── docker-compose.yml          # Qdrant + Redis (added Stage 2/4)
├── .gitignore
├── .env.example                # Config template
│
├── stage1_simple_rag.py        # Stage 1: Entire RAG pipeline in one file
│
├── app/                        # Added from Stage 3 onward
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── chunking.py             # Document → chunk processing
│   ├── embeddings.py           # Text → vector embeddings
│   ├── vectorstore.py          # Qdrant wrapper
│   ├── llm.py                  # Ollama wrapper
│   ├── cache.py                # Redis semantic cache
│   ├── pipeline.py             # Retrieval → generation pipeline
│   └── main.py                 # FastAPI app + SSE streaming
│
├── worker/
│   └── ingest.py               # Standalone ingestion worker
│
├── ui/
│   └── index.html              # Minimal streaming chat UI
│
├── data/
│   └── sample_docs/            # Sample documents for testing
│
```

---

# Build Order

## Stage 1 — Simple RAG

- Pure Python implementation
- Entire pipeline in a single file
- In-memory vector storage
- Basic retrieval + generation flow

File:
- `stage1_simple_rag.py`

A production-shaped Retrieval-Augmented Generation (RAG) system: upload
documents, ask natural-language questions, get answers streamed token-by-token
and grounded in what you uploaded — with a semantic cache for repeated
questions and a message queue so large uploads never block the API.

## Stack

- **FastAPI** — async query API, streamed via Server-Sent Events
- **Qdrant** — vector database (HNSW index, cosine similarity)
- **Redis** — semantic cache + Redis Streams job queue (same instance, two jobs)
- **Gemini 2.5 Flash** — LLM (free tier), streamed
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API call per chunk
- **Docker Compose** — runs Qdrant + Redis

## Architecture

Two independent processes, sharing state only through Redis and Qdrant:

```
Browser --> FastAPI (app/main.py) --> Qdrant (search) / Gemini (generate) / Redis (cache)
                |
                +-- enqueues ingestion jobs --> Redis Streams --> Worker (app/worker.py)
                                                                       |
                                                                       +--> Qdrant (upsert)
```


## Repository layout

```
app/
  config.py       typed settings, loaded from .env (pydantic-settings)
  chunking.py     load .txt/.pdf -> overlapping text chunks
  embeddings.py   local sentence-transformers wrapper
  vectorstore.py  Qdrant wrapper (collection setup, upsert, search)
  llm.py          Gemini wrapper: grounded-prompt builder + streaming
  cache.py        Redis semantic cache (cosine similarity over cached questions)
  queue.py        Redis Streams job queue + job status tracking
  main.py         FastAPI app: POST /ingest, POST /query (SSE), GET /jobs/{id}
  worker.py       separate process: consumes ingestion jobs, writes to Qdrant
ui/
  index.html      single-page test client (upload + streaming Q&A)
examples/
  01_inmemory_rag.py   Stage 1: the whole RAG loop, no infrastructure
  02_qdrant_rag.py     Stage 2: same loop, Qdrant instead of an in-memory list
  
docker-compose.yml  Qdrant + Redis
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your Gemini API key
cp .env.example .env
# open .env, paste your key (free at https://aistudio.google.com/apikey)

# 3. Start Qdrant + Redis
docker compose up -d

# 4. Start the API (terminal A)
uvicorn app.main:app --reload --port 8000

# 5. Start the ingestion worker (terminal B — yes, a separate process)
python -m app.worker

# 6. Open http://127.0.0.1:8000/
```

Upload `data/sample_docs/employee_handbook.txt`, wait for it to say "done",
then ask a question. Ask the same question twice and watch the response time
drop once the semantic cache kicks in.

### Stopping

```bash
docker compose down
```

## How this was built

This project was built in 5 stages, each one runnable end-to-end on its own,
each fixing a real limitation of the one before it:

> Built a RAG pipeline (in-memory). Embeddings didn't persist across runs →
> added a vector database (Qdrant). No one else could use it → wrapped it in
> a FastAPI service with streaming responses. Repeated questions were slow →
> added a Redis semantic cache. Large uploads blocked the API from answering
> questions → split ingestion into its own worker process with Redis Streams
> as the queue between them.

The two earliest stages are kept in `examples/` as standalone single-file
scripts — the fastest way to see the core RAG idea without any of the
infrastructure around it.
