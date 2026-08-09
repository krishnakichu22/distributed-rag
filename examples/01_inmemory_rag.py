import os
import sys
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# Windows PowerShell defaults stdout to cp1252, which can't print ✓ etc.
sys.stdout.reconfigure(encoding="utf-8")

# load .env
load_dotenv()

DOCUMENT_PATH = "data/sample_docs/employee_handbook.txt"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE = 400                       # characters per chunk
CHUNK_OVERLAP = 50                     # characters shared between neighbours
TOP_K = 3                              # how many chunks to retrieve per query


# Load the document
def load_document(path: str) -> str:
    """Read a plain-text file and return its full contents as one string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Split into chunks
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:                       # skip empty/whitespace-only pieces
            chunks.append(chunk)
        start += chunk_size - overlap   # step forward, but leave an overlap
    return chunks



# Embed the chunks (LOCAL, free, no API call)
def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False)


# Retrieve the most relevant chunks for a question
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question_vec, chunk_vecs, chunks, top_k: int) -> list[str]:
    scored = []
    for chunk, vec in zip(chunks, chunk_vecs):
        score = cosine_similarity(question_vec, vec)
        scored.append((score, chunk))

    # sort highest-score first, keep the best few
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]

    print("\n--- Retrieved chunks (score : preview) ---")
    for score, chunk in top:
        preview = chunk[:70].replace("\n", " ")
        print(f"  {score:.3f} : {preview}...")
    print("------------------------------------------\n")

    return [chunk for _, chunk in top]


# Generate an answer with LLM (Gemini)
def generate_answer(client: genai.Client, question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY
    the context below. If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question: {question}

    Answer:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def main():
    print("Loading local embedding model (first run downloads ~80MB)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Gemini...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    document = load_document(DOCUMENT_PATH)
    chunks = chunk_text(document, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Document split into {len(chunks)} chunks.")

    # ingestion
    chunk_vectors = embed_texts(embedder, chunks)
    print(f"Embedded chunks into {chunk_vectors.shape[1]}-dim vectors.")

    print("\n✓ RAG is ready. Ask questions about the employee handbook.")
    print("  (type 'quit' to exit)\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            print("Bye.")
            break

        question_vector = embed_texts(embedder, [question])[0]
        relevant = retrieve(question_vector, chunk_vectors, chunks, TOP_K)

        answer = generate_answer(client, question, relevant)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()

# stage 1 of simple RAG implementation: no vector database, just in-memory lists and numpy arrays.