# distributed-rag
A distributed RAG Architecture 


Repository structure :

distributed-rag/
├── README.md
├── requirements.txt
├── docker-compose.yml          # Qdrant + Redis (added Stage 2/4)
├── .gitignore
├── .env.example                # config template
│
├── stage1_simple_rag.py        # Stage 1: the whole concept in one file
│
├── app/                        # grows from Stage 3 on
│   ├── __init__.py
│   ├── config.py               # central settings
│   ├── chunking.py             # document → chunks
│   ├── embeddings.py           # text → vectors (sentence-transformers)
│   ├── vectorstore.py          # Qdrant wrapper
│   ├── llm.py                  # Ollama wrapper
│   ├── cache.py                # Redis semantic cache
│   ├── pipeline.py             # ties retrieve → generate together
│   └── main.py                 # FastAPI app, endpoints, SSE streaming
│
├── worker/
│   └── ingest.py               # standalone ingestion worker
│
├── ui/
│   └── index.html              # minimal streaming chat UI
│
├── data/
│   └── sample_docs/            # sample text files to test with
│
└── notes/
    └── interview_prep.md       # your cheat sheet — design decisions & Q&A

Build order

Stage 1 → simple RAG in one file (Python only).
Stage 2 → swap in-memory store for Qdrant (first Docker service).
Stage 3 → FastAPI wrapper + SSE streaming.
Stage 4 → Redis + semantic cache, with before/after latency numbers.
Stage 5 → separate ingestion worker + full docker-compose.