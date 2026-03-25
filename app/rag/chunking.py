import re
from dataclasses import dataclass
from pypdf import PdfReader
from io import BytesIO
from app.config import DEFAULT_CHILD_SIZE, DEFAULT_PARENT_SIZE, DEFAULT_OVERLAP


@dataclass
class ChunkResult:
    """A child chunk with reference to its parent context."""
    child_text: str
    parent_text: str
    section: str
    chunk_index: int


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section-based splitting
# ---------------------------------------------------------------------------

SECTION_PATTERNS = re.compile(
    r'(?:^|\n)'
    r'(?:'
    r'#{1,6}\s+'
    r'|หมวด(?:ที่)?\s*\d+'
    r'|ข้อ\s*\d+'
    r'|บทที่\s*\d+'
    r'|\d+\.\s+[A-Zก-ฮ]'
    r'|[A-Z][A-Z\s]{3,}$'
    r')'
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text by section headings. Returns list of (section_name, section_body)."""
    text = text.strip()
    if not text:
        return []

    matches = list(SECTION_PATTERNS.finditer(text))

    if not matches:
        return [("(untitled)", text)]

    sections = []

    if matches[0].start() > 0:
        pre_text = text[:matches[0].start()].strip()
        if pre_text:
            sections.append(("(intro)", pre_text))

    for i, match in enumerate(matches):
        heading_end = text.find("\n", match.start() + 1)
        if heading_end == -1:
            heading_end = len(text)
        section_name = text[match.start():heading_end].strip()

        body_start = heading_end
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        sections.append((section_name, f"{section_name}\n{body}" if body else section_name))

    return sections


# ---------------------------------------------------------------------------
# Parent-child chunking
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    child_size: int = DEFAULT_CHILD_SIZE,
    parent_size: int = DEFAULT_PARENT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkResult]:
    """
    Smart chunking: section-based → parent chunks → child chunks.
    Child carries reference to its parent for full context retrieval.
    """
    text = text.strip()
    if not text:
        return []

    sections = split_into_sections(text)
    results = []
    global_index = 0

    for section_name, section_body in sections:
        parents = _chunk_text(section_body, chunk_size=parent_size, overlap=100)

        for parent in parents:
            children = _chunk_text(parent, chunk_size=child_size, overlap=overlap)

            for child in children:
                results.append(ChunkResult(
                    child_text=child,
                    parent_text=parent,
                    section=section_name,
                    chunk_index=global_index,
                ))
                global_index += 1

    return results


# ---------------------------------------------------------------------------
# Basic chunking (internal + backward compat)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple chunking — kept for backward compatibility."""
    return _chunk_text(text, chunk_size, overlap)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
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

        boundary = _find_sentence_boundary(text, end, search_range=80)
        if boundary:
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def _find_sentence_boundary(text: str, pos: int, search_range: int = 80) -> int | None:
    search_start = max(0, pos - search_range)
    search_end = min(len(text), pos + search_range)
    region = text[search_start:search_end]

    boundaries = [m.end() for m in re.finditer(r'[.!?。]\s', region)]

    if not boundaries:
        return None

    target = pos - search_start
    closest = min(boundaries, key=lambda b: abs(b - target))
    return search_start + closest
