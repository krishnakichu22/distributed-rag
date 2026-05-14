┌──────────────┐
                          │   Web UI      │  ask questions, see streaming answers
                          └──────┬───────┘
                                 │ HTTP / SSE
                          ┌──────▼───────────┐
                          │  FastAPI service  │  ← the "query" side
                          │  - /query (stream)│
                          │  - /ingest        │
                          └──┬────┬────┬──────┘
                  cache?     │    │    │   retrieve
            ┌────────────────┘    │    └────────────┐
        ┌───▼────┐         embed  │             ┌───▼─────┐
        │ Redis  │         (local)│             │ Qdrant  │  vector DB
        │ cache  │                │             │         │
        └────────┘         ┌──────▼──────┐      └───▲─────┘
                           │  Ollama      │          │
                           │ (local LLM)  │          │ upsert
                           └──────────────┘   ┌──────┴────────┐
                                              │ Ingestion      │  ← the "write" side
                                              │ Worker         │  separate process
                                              │ parse→chunk→   │
                                              │ embed→store    │
                                              └────────────────┘