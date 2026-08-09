import sys
from google import genai

# Windows PowerShell defaults stdout to cp1252, which can't print ✓ etc.
sys.stdout.reconfigure(encoding="utf-8")

from app import config
from app.chunking import load_document, chunk_text
from app.embeddings import Embedder
from app.vectorstore import VectorStore


DOCUMENT_PATH = "data/sample_docs/employee_handbook.txt"


def ingest(store: VectorStore, embedder: Embedder, path: str) -> None:
    """Read a document, chunk it, embed the chunks, upsert into Qdrant."""
    print(f"\n--- Ingesting {path} ---")
    text = load_document(path)
    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  → {len(chunks)} chunks")

    vectors = embedder.embed(chunks)
    print(f"  → embedded into {vectors.shape[1]}-dim vectors")

    n = store.upsert(chunks, vectors, source=path)
    print(f"  → upserted {n} points into Qdrant")
    print(f"  → total points in collection: {store.count()}\n")


def answer_question(
    store: VectorStore,
    embedder: Embedder,
    gemini: genai.Client,
    question: str,
) -> str:
    """Run the retrieve → augment → generate flow for a single question."""
    # STEP 4 — embed the question, search Qdrant for top-k similar chunks
    query_vec = embedder.embed_one(question)
    hits = store.search(query_vec, top_k=config.TOP_K)

    print("\n--- Retrieved chunks (score : preview) ---")
    for hit in hits:
        preview = hit["text"][:70].replace("\n", " ")
        print(f"  {hit['score']:.3f} : {preview}...")
    print("------------------------------------------\n")

    # STEP 5 — build the grounded prompt and ask Gemini
    context = "\n\n".join(hit["text"] for hit in hits)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = gemini.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def main() -> None:
    config.require_gemini_key()

    # Setup: open clients (Qdrant, Gemini) and load the embedding model
    print("Setting up...")
    store = VectorStore()
    store.ensure_collection(recreate=True)  # wipe-and-start-fresh for the demo

    embedder = Embedder()
    gemini = genai.Client(api_key=config.GEMINI_API_KEY)

    # Ingest the sample document (one-time, up front for this demo)
    ingest(store, embedder, DOCUMENT_PATH)

    # Interactive loop
    print("✓ RAG is ready. Ask questions about the employee handbook.")
    print("  (type 'quit' to exit)\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            print("Bye.")
            break

        answer = answer_question(store, embedder, gemini, question)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
