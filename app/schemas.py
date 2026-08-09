
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Returned immediately by POST /ingest — the job hasn't run yet."""

    job_id: str
    filename: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    """Returned by GET /jobs/{job_id}."""

    job_id: str
    status: str  # "pending" | "processing" | "done" | "error"
    filename: str | None = None
    chunks_ingested: int | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    """Body of POST /query."""

    question: str = Field(min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    """One retrieved chunk, sent to the client alongside the answer."""

    text: str
    source: str
    score: float
