from pathlib import Path

from pypdf import PdfReader


def load_document(path: str | Path) -> str:
    """Read a .txt or .pdf file and return its full text contents."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Cut text into ~chunk_size pieces with `overlap` chars shared between neighbours."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
