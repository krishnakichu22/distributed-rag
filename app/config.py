from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Gemini (LLM) ─────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # ── Embeddings (local model) ────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    # ── Qdrant (vector store) ───────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # ── Redis (cache + job queue) ───────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    semantic_cache_threshold: float = 0.95  # cosine score to count as a cache HIT
    cache_ttl_seconds: int = 3600           # cached answers expire after 1 hour

    # ── Ingestion queue (Redis Streams) ─────────────────────────────
    ingestion_stream: str = "ingestion_jobs"
    ingestion_group: str = "ingestion_workers"
    upload_dir: str = "data/uploads"

    # ── Retrieval ────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 3


settings = Settings()

# ── Backwards-compatible module-level names ───────────────────────────
# app/vectorstore.py, embeddings.py etc. (and the tutorial scripts under
# examples/) were written against `config.GEMINI_API_KEY` style constants.
# Keeping both means those files don't need to change.
GEMINI_API_KEY = settings.gemini_api_key
GEMINI_MODEL = settings.gemini_model
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIM = settings.embedding_dim
QDRANT_HOST = settings.qdrant_host
QDRANT_PORT = settings.qdrant_port
QDRANT_COLLECTION = settings.qdrant_collection
REDIS_HOST = settings.redis_host
REDIS_PORT = settings.redis_port
SEMANTIC_CACHE_THRESHOLD = settings.semantic_cache_threshold
CACHE_TTL_SECONDS = settings.cache_ttl_seconds
INGESTION_STREAM = settings.ingestion_stream
INGESTION_GROUP = settings.ingestion_group
UPLOAD_DIR = settings.upload_dir
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K = settings.top_k


def require_gemini_key() -> None:
    """Fail fast with a clear message if the API key isn't set."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "paste-your-key-here":
        raise SystemExit(
            "GEMINI_API_KEY is missing.\n"
            "   1. Copy .env.example to .env\n"
            "   2. Get a free key at https://aistudio.google.com/apikey\n"
            "   3. Paste it into .env"
        )
