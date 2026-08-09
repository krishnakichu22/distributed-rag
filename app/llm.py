from __future__ import annotations
import asyncio
import queue as thread_queue
import threading
from typing import AsyncIterator, Iterator

from google import genai

from app import config


def build_grounded_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Combine retrieved chunks + the question into the prompt Gemini sees.

    The instruction "using ONLY the context" + "say you don't know" is
    the main defense against hallucination — without it the model will
    happily answer from its own training data instead of your documents,
    which defeats the point of RAG.
    """
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


class GeminiLLM:
    """Thin wrapper around the Gemini client for both streaming and non-streaming calls."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or config.GEMINI_MODEL
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        """One-shot call — waits for the full answer, returns it as a string."""
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Yields the answer as it's generated, piece by piece.

        Gemini doesn't send one token at a time — each streamed chunk is
        usually a few words. We yield chunk.text as-is; main.py forwards
        each piece to the browser the moment it arrives, which is what
        makes the response feel instant instead of waiting ~3s for the
        whole thing.
        """
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=prompt):
            if chunk.text:
                yield chunk.text

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Async version of stream() for use inside FastAPI's async /query handler.

        Why this needs to exist at all:
          self.stream() above is a SYNC generator — each `next()` on it
          blocks the calling thread waiting on network I/O from Gemini.
          Calling it directly inside an `async def` endpoint would freeze
          FastAPI's entire event loop for the whole answer, stalling every
          other request the server is handling. There's no native
          async version of this SDK call, so we bridge it ourselves:
          run the blocking generator on a background thread, and have
          it push each piece onto a thread-safe queue.Queue. The async
          side awaits `queue.get()` in an executor (which DOES yield
          control back to the event loop) and re-yields each piece as
          it arrives. Net effect: real token-by-token streaming, without
          blocking anyone else's request.
        """
        q: thread_queue.Queue = thread_queue.Queue()
        DONE = object()

        def produce() -> None:
            try:
                for piece in self.stream(prompt):
                    q.put(piece)
            except Exception as exc:  # noqa: BLE001 - surface the error to the async side
                q.put(exc)
            finally:
                q.put(DONE)

        threading.Thread(target=produce, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
