import re
from pypdf import PdfReader
from io import BytesIO


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to break at sentence boundary near the end
        boundary = _find_sentence_boundary(text, end, search_range=80)
        if boundary:
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def _find_sentence_boundary(text: str, pos: int, search_range: int = 80) -> int | None:
    """Find the nearest sentence-ending punctuation around pos."""
    search_start = max(0, pos - search_range)
    search_end = min(len(text), pos + search_range)
    region = text[search_start:search_end]

    # Find all sentence boundaries in the search region
    boundaries = [m.end() for m in re.finditer(r'[.!?。]\s', region)]

    if not boundaries:
        return None

    # Pick the boundary closest to the target position
    target = pos - search_start
    closest = min(boundaries, key=lambda b: abs(b - target))
    return search_start + closest
