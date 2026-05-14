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
└── notes/
    └── interview_prep.md       # Design decisions & interview Q&A
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

---

## Stage 2 — Qdrant Integration

- Replace in-memory storage with Qdrant
- Introduce Docker services
- Persistent vector database

Key additions:
- `docker-compose.yml`
- `app/vectorstore.py`

---

## Stage 3 — FastAPI + Streaming

- Wrap pipeline inside FastAPI
- Add API endpoints
- Implement Server-Sent Events (SSE) streaming
- Add minimal browser UI

Key additions:
- `app/main.py`
- `ui/index.html`

---

## Stage 4 — Redis Semantic Cache

- Add Redis caching layer
- Reduce repeated LLM calls
- Measure latency improvements

Key additions:
- `app/cache.py`

Features:
- Semantic similarity caching
- Faster repeated queries
- Before/after latency benchmarking

---

## Stage 5 — Distributed Ingestion Worker

- Separate ingestion from serving
- Add worker-based architecture
- Full multi-service Docker setup

Key additions:
- `worker/ingest.py`
- Updated `docker-compose.yml`

---

# Tech Stack

- Python
- FastAPI
- Qdrant
- Redis
- Ollama
- Sentence Transformers
- Docker
- SSE Streaming

---

# Goals

- Learn RAG systems incrementally
- Understand distributed AI architecture
- Build production-style retrieval systems
- Prepare for system design interviews
- Explore scalable LLM infrastructure

---